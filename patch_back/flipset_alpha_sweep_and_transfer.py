# -*- coding: utf-8 -*-
"""
flipset_alpha_sweep_and_transfer.py

Two works on top of your existing subspace_patching_transfer script:

Work 1) Flip-set (defined at alpha=1) -> alpha sweep on flip-set only
        Report flip-rate / margins (and delta-margins vs baseline).

Work 2) Transfer-donor subspace patching on flip-set:
        donor comes from other example (same task or other tasks),
        not self donor.

This script dynamically imports your existing script (the one you pasted),
and reuses its utilities:
  - load_aux_modules
  - get_transformer_layers
  - orthonormalize_np, project_cpu
  - DecodeStepHiddenCaptureHook, SubspacePatchHook
  - forced_choice_decode_aligned
  - load_selected_tasks_eval_only / maybe_compute_Qs (optional)
and uses loto8.LastTokenRemovalHook from the imported loto8 module.

Recommended: patch_window=steps_0 for transfer donor (robust across cache types).

Example:
python flipset_alpha_sweep_and_transfer.py \
  --base_script_path subspace_patching_transfer.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --device cuda --dtype fp16 \
  --layer 10 \
  --task aqua --n_eval 1024 --flipset_max 128 \
  --Qs_path Q_shared_layer10.npy \
  --alpha_list 0,0.05,0.1,0.2,0.5,1.0 \
  --run_alpha_sweep 1 \
  --run_transfer_patching 1 \
  --donor_source cross_task_eval \
  --donor_tasks gsm8k,commonsenseqa,strategyqa \
  --donor_n_eval 64 \
  --patch_window steps_0 \
  --out_json flipset_sweep_transfer.json

A) 只跑 flip-set 上的 α sweep（Work 1）
python flipset_alpha_sweep_and_transfer.py \
  --base_script_path subspace_patching_transfer.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --device cuda --dtype fp16 \
  --layer 10 \
  --task aqua --n_eval 256 --flipset_max 128 \
  --Qs_path Q_shared_layer10.npy \
  --run_alpha_sweep 1 \
  --alpha_list 0,0.02,0.05,0.1,0.2,0.5,1.0 \
  --run_transfer_patching 0 \
  --out_json flipset_alpha_sweep.json


你会在输出 JSON 里得到：

alpha_sweep_summary_on_flipset：每个 alpha 的 flip_rate / mean_margin / mean Δmargin(vs baseline)

alpha_sweep_rows_by_alpha：逐样本结果（方便你画曲线）

B) 跑 transfer donor patching（Work 2）
1) donor 来自同 task 的 eval 其他样本
python flipset_alpha_sweep_and_transfer.py \
  --base_script_path subspace_patching_transfer.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --device cuda --dtype fp16 \
  --layer 10 \
  --task aqua --n_eval 256 --flipset_max 64 \
  --Qs_path Q_shared_layer10.npy \
  --run_alpha_sweep 0 \
  --run_transfer_patching 1 \
  --donor_source same_task_eval \
  --donor_n_eval 128 \
  --patch_window steps_0 \
  --run_self_patch_ref 1 \
  --out_json flipset_transfer_same_task.json

2) donor 来自别的 task
python flipset_alpha_sweep_and_transfer.py \
  --base_script_path subspace_patching_transfer.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --device cuda --dtype fp16 \
  --layer 10 \
  --task aqua --n_eval 256 --flipset_max 64 \
  --Qs_path Q_shared_layer10.npy \
  --run_alpha_sweep 0 \
  --run_transfer_patching 1 \
  --donor_source cross_task_eval \
  --donor_tasks gsm8k,commonsenseqa,strategyqa \
  --donor_n_eval 64 \
  --patch_window steps_0 \
  --run_self_patch_ref 1 \
  --out_json flipset_transfer_cross_task.json

我刻意做的几个“稳健性处理”

flip-set 固定按 α=1 定义（你说的 common+reasonable 定义）。

transfer patching 默认建议 --patch_window steps_0：
因为在 Transformers 新 cache 路径下，step>0 往往变成 per-candidate 循环（batch 维度不同），你原脚本里也提到 beyond step0 在 fallback path “less meaningful”。
所以这个脚本会：

如果你请求 steps_01/full，会先 probe 一次是否 batched-candidate；

不满足就自动降级到 {0}，避免 shape mismatch 和语义歧义。

"""

from __future__ import annotations

import os
import json
import argparse
import importlib.util
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def import_module_from_path(module_name: str, file_path: str):
    import sys
    import os
    import importlib.util

    file_path = os.path.abspath(file_path)
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_name} from {file_path}")

    mod = importlib.util.module_from_spec(spec)

    # 关键：先注册到 sys.modules，dataclasses/typing 才能找到模块命名空间
    sys.modules[module_name] = mod
    try:
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    except Exception:
        # 失败就清理，避免 sys.modules 里残留半初始化模块
        sys.modules.pop(module_name, None)
        raise

    return mod


# # -----------------------------------------------------------------------------
# # Dynamic import of your existing script
# # -----------------------------------------------------------------------------
# def import_module_from_path(module_name: str, file_path: str):
#     spec = importlib.util.spec_from_file_location(module_name, file_path)
#     if spec is None or spec.loader is None:
#         raise ImportError(f"Cannot import {module_name} from {file_path}")
#     mod = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(mod)  # type: ignore[attr-defined]
#     return mod


def parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def parse_float_list(s: str) -> List[float]:
    out: List[float] = []
    for x in (s or "").split(","):
        x = x.strip()
        if not x:
            continue
        out.append(float(x))
    return out


def build_candidate_texts(base_mod: Any, candidate_labels: List[str], style: str) -> List[str]:
    # reuse base helper if present; fallback to same logic
    if hasattr(base_mod, "build_candidate_texts"):
        return base_mod.build_candidate_texts(candidate_labels, style)
    if style == "raw":
        return [lab for lab in candidate_labels]
    if style == "space_letter":
        return [" " + lab for lab in candidate_labels]
    raise ValueError(f"Unknown candidate_text_style={style}")


def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def fc_to_dict(res: Any) -> Dict[str, Any]:
    # base_mod.FCResult is a dataclass; but be defensive
    if hasattr(res, "__dict__"):
        return dict(res.__dict__)
    return {
        "pred_label": getattr(res, "pred_label", ""),
        "correct": bool(getattr(res, "correct", False)),
        "margin": float(getattr(res, "margin", float("nan"))),
        "scores": dict(getattr(res, "scores", {})),
    }


def summarize_alpha_sweep(rows_by_alpha: Dict[float, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    rows_by_alpha[alpha] = list of per-example dicts with:
      - baseline: {correct, margin, pred_label}
      - ablated:  {correct, margin, pred_label}
      - flip (bool): baseline_correct & (not ablated_correct)
    """
    out: Dict[str, Any] = {}
    for a, rows in rows_by_alpha.items():
        n = len(rows)
        if n == 0:
            out[str(a)] = {"n": 0}
            continue
        flip_rate = float(np.mean([1.0 if r["flip"] else 0.0 for r in rows]))
        ablt_acc = float(np.mean([1.0 if r["ablated"]["correct"] else 0.0 for r in rows]))
        pred_change = float(np.mean([1.0 if r["ablated"]["pred_label"] != r["baseline"]["pred_label"] else 0.0 for r in rows]))
        margins = np.array([float(r["ablated"]["margin"]) for r in rows], dtype=np.float32)
        base_margins = np.array([float(r["baseline"]["margin"]) for r in rows], dtype=np.float32)
        dm = margins - base_margins

        out[str(a)] = {
            "n": n,
            "flip_rate": flip_rate,
            "ablated_acc": ablt_acc,
            "pred_change_rate": pred_change,
            "mean_margin": float(np.mean(margins)),
            "median_margin": float(np.median(margins)),
            "mean_delta_margin_vs_baseline": float(np.mean(dm)),
            "median_delta_margin_vs_baseline": float(np.median(dm)),
        }
    return out


def summarize_patching(rows: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    """
    rows entries should contain:
      - ablated_1: {correct, margin}
      - <key>: {correct, margin}
    """
    n = len(rows)
    if n == 0:
        return {"n": 0}
    rescued = int(sum(1 for r in rows if bool(r[key]["correct"])))
    dm = []
    for r in rows:
        dm.append(float(r[key]["margin"]) - float(r["ablated_1"]["margin"]))
    dm = np.array(dm, dtype=np.float32)
    return {
        "n": n,
        "rescued": rescued,
        "rescued_pct": 100.0 * rescued / n,
        "mean_delta_margin_vs_ablated": float(np.mean(dm)),
        "median_delta_margin_vs_ablated": float(np.median(dm)),
    }


@torch.no_grad()
def detect_batched_candidate_path(
    base_mod: Any,
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_module: torch.nn.Module,
    prompt: str,
    candidate_labels: List[str],
    candidate_texts: List[str],
) -> bool:
    """
    Try to detect whether forced_choice_decode_aligned is using the "legacy cache batched-candidate"
    path (where step>=1 hidden has batch=K), vs fallback Cache-object path.
    """
    if len(candidate_labels) < 2:
        return False
    cap = base_mod.DecodeStepHiddenCaptureHook(capture_steps=[0, 1])
    gold = candidate_labels[0]
    _ = base_mod.forced_choice_decode_aligned(
        model, tok, prompt,
        candidate_labels, candidate_texts, gold,
        layer_module=layer_module,
        capture_hook=cap,
        add_special_tokens_prompt=True,
    )
    h1 = cap.hidden_by_step.get(1, None)
    if h1 is None:
        return False
    # In batched path, step1 often has batch=K (num candidates)
    return int(h1.shape[0]) == int(len(candidate_labels))


def main():
    ap = argparse.ArgumentParser()

    # --- where to import your existing script ---
    ap.add_argument("--base_script_path", type=str, required=True,
                    help="Path to your existing script (the one you pasted), e.g. subspace_patching_transfer.py")

    # --- model / data ---
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--layer", type=int, required=True, help="Layer index to hook (0-based)")

    ap.add_argument("--task", type=str, default="aqua", help="Target eval task to build flip-set from")
    ap.add_argument("--n_eval", type=int, default=256, help="How many eval examples to scan to find flips")

    # flip-set controls
    ap.add_argument("--flipset_max", type=int, default=128,
                    help="How many flip examples to actually use downstream (alpha sweep / patching)")

    # candidates
    ap.add_argument("--candidate_labels", type=str, default="ABCDE")
    ap.add_argument("--candidate_text_style", type=str, default="space_letter", choices=["space_letter", "raw"])
    ap.add_argument("--add_special_tokens_prompt", type=int, default=1)

    # Q_shared
    ap.add_argument("--Qs_path", type=str, default="", help="Path to Q_shared .npy [d,k]")
    ap.add_argument("--compute_Qs", type=int, default=0)
    ap.add_argument("--Qs_out", type=str, default="Q_shared_computed.npy")
    ap.add_argument("--basis_tasks", type=str, default="gsm8k,commonsenseqa,strategyqa,openbookqa,qasc,boolq,piqa")
    ap.add_argument("--basis_n_subspace", type=int, default=2048)
    ap.add_argument("--calib_batch_size", type=int, default=8)
    ap.add_argument("--calib_max_new_tokens", type=int, default=128)
    ap.add_argument("--per_task_max_states", type=int, default=20000)
    ap.add_argument("--max_prompt_len", type=int, default=1024)
    ap.add_argument("--variance_threshold", type=float, default=0.95)
    ap.add_argument("--min_dim", type=int, default=8)
    ap.add_argument("--max_dim", type=int, default=1024)
    ap.add_argument("--tau", type=float, default=0.001)
    ap.add_argument("--m_shared", type=str, default="all")

    # modules
    ap.add_argument("--loto8_path", type=str, default="disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py")
    ap.add_argument("--dataloaders_path", type=str, default="benchmark_dataloaders.py")

    # prompt formatting
    ap.add_argument("--answer_prefix", type=str, default="\nFinal answer:")
    ap.add_argument("--template_randomization", type=int, default=1)
    ap.add_argument("--shuffle_choices", type=int, default=1)
    ap.add_argument("--seed", type=int, default=123)

    # Work 1: alpha sweep on flip-set
    ap.add_argument("--run_alpha_sweep", type=int, default=1)
    ap.add_argument("--alpha_list", type=str, default="0,0.05,0.1,0.2,0.5,1.0")

    # Work 2: transfer donor patching
    ap.add_argument("--run_transfer_patching", type=int, default=1)
    ap.add_argument("--patch_window", type=str, default="steps_0", choices=["steps_0", "steps_01", "full_steps"],
                    help="Patch steps for transfer donor. Recommend steps_0 for robustness.")
    ap.add_argument("--run_self_patch_ref", type=int, default=1, help="Also run self-donor patch as a reference")

    ap.add_argument("--donor_source", type=str, default="same_task_eval",
                    choices=["same_task_eval", "cross_task_eval"],
                    help="Where to draw donor examples from")
    ap.add_argument("--donor_tasks", type=str, default="",
                    help="Comma-separated donor tasks (used when donor_source=cross_task_eval)")
    ap.add_argument("--donor_n_eval", type=int, default=64, help="How many donor eval examples per donor task")
    ap.add_argument("--donor_pick", type=str, default="cyclic", choices=["cyclic", "random"])
    ap.add_argument("--donor_require_gold_in_candidates", type=int, default=0,
                    help="Filter donor examples to those whose gold is in candidate_labels")
    ap.add_argument("--donor_require_baseline_correct", type=int, default=0,
                    help="Filter donor examples to those baseline-correct (only if gold in candidate_labels)")

    ap.add_argument("--out_json", type=str, default="flipset_alpha_sweep_transfer.json")
    args = ap.parse_args()

    # -------------------------------------------------------------------------
    # Import your base script and aux modules
    # -------------------------------------------------------------------------
    base_mod = import_module_from_path("base_subspace_patching_transfer", args.base_script_path)
    loto8, dl = base_mod.load_aux_modules(args.loto8_path, args.dataloaders_path)

    # seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # model
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch_dtype,
            device_map=None,
        ).to(args.device)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch_dtype,
            device_map=None,
        ).to(args.device)
    model.eval()

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    layers, path_used = base_mod.get_transformer_layers(model)
    if args.layer < 0 or args.layer >= len(layers):
        raise ValueError(f"--layer {args.layer} out of range for layers at {path_used} (n={len(layers)})")
    layer_module = layers[args.layer]
    print(f"[Info] Hooking layer={args.layer} at path {path_used}")

    # candidates
    candidate_labels = list(args.candidate_labels.strip())
    candidate_texts = build_candidate_texts(base_mod, candidate_labels, args.candidate_text_style)

    cand_token_ids = [tok.encode(ct, add_special_tokens=False) for ct in candidate_texts]
    cand_lens = [len(x) for x in cand_token_ids]
    max_len = max(cand_lens) if cand_lens else 1

    steps_0 = {0}
    steps_01 = {0, 1} if max_len >= 2 else {0}
    full_steps = set(range(max_len))

    patch_steps_user = steps_0 if args.patch_window == "steps_0" else (steps_01 if args.patch_window == "steps_01" else full_steps)

    print("[Info] Candidate token lens:", {lab: l for lab, l in zip(candidate_labels, cand_lens)})
    print(f"[Info] patch steps requested ({args.patch_window}): {sorted(list(patch_steps_user))}")

    # -------------------------------------------------------------------------
    # Load / compute Q_shared
    # -------------------------------------------------------------------------
    Qs: Optional[np.ndarray] = None
    if args.Qs_path:
        Qs = base_mod.orthonormalize_np(np.load(args.Qs_path).astype(np.float32))
        print(f"[Info] Loaded Q_shared from {args.Qs_path}  shape={Qs.shape}")
    elif args.compute_Qs:
        tasks = parse_csv_list(args.basis_tasks)
        Qs = base_mod.maybe_compute_Qs(
            loto8=loto8,
            dl=dl,
            model=model,
            tokenizer=tok,
            layer_idx=args.layer,
            seed=args.seed,
            tasks=tasks,
            n_subspace=args.basis_n_subspace,
            template_randomization=bool(args.template_randomization),
            shuffle_choices=bool(args.shuffle_choices),
            answer_prefix=args.answer_prefix,
            calib_batch_size=args.calib_batch_size,
            calib_max_new_tokens=args.calib_max_new_tokens,
            per_task_max_states=args.per_task_max_states,
            max_prompt_len=args.max_prompt_len,
            variance_threshold=args.variance_threshold,
            min_dim=args.min_dim,
            max_dim=args.max_dim,
            tau=args.tau,
            m_shared=args.m_shared,
            out_path=args.Qs_out,
        )
    else:
        raise RuntimeError("Provide --Qs_path or set --compute_Qs=1")
    assert Qs is not None
    d, k = Qs.shape

    # -------------------------------------------------------------------------
    # Load eval examples for target task (build flip-set from alpha=1)
    # -------------------------------------------------------------------------
    _, eval_by, meta_by = base_mod.load_selected_tasks_eval_only(
        dl,
        task=args.task,
        n_eval=args.n_eval,
        seed=args.seed,
        template_randomization=bool(args.template_randomization),
        template_seed=args.seed + 999,
        shuffle_choices=bool(args.shuffle_choices),
        answer_prefix=args.answer_prefix
    )
    eval_examples = eval_by[args.task]
    eval_meta = meta_by.get(args.task, {})
    print(f"[Info] Loaded eval examples: task={args.task}, n={len(eval_examples)} meta={eval_meta}")
    if len(eval_examples) == 0:
        raise RuntimeError("No eval examples loaded. Check dataset availability / splits / extraction.")

    # -------------------------------------------------------------------------
    # Scan ALL eval examples (up to n_eval loaded) to build flip-set @ alpha=1
    # -------------------------------------------------------------------------
    scan_rows: List[Dict[str, Any]] = []
    flip_examples: List[Any] = []
    baseline_cache: Dict[str, Dict[str, Any]] = {}
    ablated1_cache: Dict[str, Dict[str, Any]] = {}

    for ex in eval_examples:
        prompt = ex.prompt
        gold = (ex.gold or "").strip().upper()
        ex_id = ex.ex_id

        # skip if gold not in candidate labels (keeps "correct" meaningful)
        if gold not in candidate_labels:
            scan_rows.append({
                "ex_id": ex_id,
                "gold": gold,
                "skipped_reason": f"gold_not_in_candidates (gold={gold})",
            })
            continue

        base_res = base_mod.forced_choice_decode_aligned(
            model, tok, prompt,
            candidate_labels, candidate_texts, gold,
            layer_module=layer_module,
            add_special_tokens_prompt=bool(args.add_special_tokens_prompt),
        )
        remove1 = loto8.LastTokenRemovalHook(Qs, alpha=1.0, stats=loto8.HookStats("remove_shared_a1"))
        ablt1_res = base_mod.forced_choice_decode_aligned(
            model, tok, prompt,
            candidate_labels, candidate_texts, gold,
            layer_module=layer_module,
            removal_hook=remove1,
            add_special_tokens_prompt=bool(args.add_special_tokens_prompt),
        )

        base_d = fc_to_dict(base_res)
        ablt1_d = fc_to_dict(ablt1_res)
        baseline_cache[ex_id] = base_d
        ablated1_cache[ex_id] = ablt1_d

        scan_rows.append({
            "ex_id": ex_id,
            "gold": gold,
            "baseline": base_d,
            "ablated_1": ablt1_d,
        })

        if bool(base_d["correct"]) and (not bool(ablt1_d["correct"])):
            flip_examples.append(ex)

    n_scanned = len([r for r in scan_rows if "baseline" in r])
    n_flips_total = len(flip_examples)
    n_flips_used = min(int(args.flipset_max), n_flips_total)
    flip_used = flip_examples[:n_flips_used]

    print(f"[FlipSet@alpha=1] flips_total={n_flips_total}  flips_used={n_flips_used}  (scanned_with_gold={n_scanned})")
    if n_flips_used == 0:
        out = {
            "meta": {
                "note": "No flip examples found (baseline correct, ablated wrong) under alpha=1.0",
                "model": args.model,
                "layer": args.layer,
                "task": args.task,
                "candidate_labels": candidate_labels,
                "candidate_text_style": args.candidate_text_style,
                "Qs_path": args.Qs_path or args.Qs_out,
                "Qs_shape": [int(d), int(k)],
                "flipset_alpha_def": 1.0,
                "n_eval_loaded": len(eval_examples),
                "n_flips_total": 0,
            },
            "scan_rows": scan_rows,
        }
        ensure_dir(args.out_json)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"[Done] No flips. Wrote {args.out_json}")
        return

    # -------------------------------------------------------------------------
    # Work 1: Alpha sweep on flip-set only
    # -------------------------------------------------------------------------
    alpha_rows_by_alpha: Dict[float, List[Dict[str, Any]]] = {}
    alpha_summary: Dict[str, Any] = {}

    if bool(args.run_alpha_sweep):
        alphas = parse_float_list(args.alpha_list)
        if len(alphas) == 0:
            raise ValueError("--alpha_list is empty; provide e.g. '0,0.1,0.2,0.5,1.0'")

        print(f"[AlphaSweep] running on flip-set (size={len(flip_used)}), alphas={alphas}")

        for a in alphas:
            rows_a: List[Dict[str, Any]] = []
            for ex in flip_used:
                ex_id = ex.ex_id
                prompt = ex.prompt
                gold = (ex.gold or "").strip().upper()

                base_d = baseline_cache[ex_id]  # baseline correct by construction
                remove = loto8.LastTokenRemovalHook(Qs, alpha=float(a), stats=loto8.HookStats(f"remove_shared_a{a:g}"))
                ablt = base_mod.forced_choice_decode_aligned(
                    model, tok, prompt,
                    candidate_labels, candidate_texts, gold,
                    layer_module=layer_module,
                    removal_hook=remove,
                    add_special_tokens_prompt=bool(args.add_special_tokens_prompt),
                )
                ablt_d = fc_to_dict(ablt)

                rows_a.append({
                    "ex_id": ex_id,
                    "gold": gold,
                    "baseline": base_d,
                    "ablated": ablt_d,
                    "flip": bool(base_d["correct"]) and (not bool(ablt_d["correct"])),
                })
            alpha_rows_by_alpha[float(a)] = rows_a

        alpha_summary = summarize_alpha_sweep(alpha_rows_by_alpha)

        # quick print
        print("[AlphaSweep Summary] (on flip-set)")
        for a_str, s in alpha_summary.items():
            print(f"  alpha={a_str:>6s}  flip_rate={s.get('flip_rate', float('nan')):.3f}  "
                  f"mean_margin={s.get('mean_margin', float('nan')):.3f}  "
                  f"meanΔm(vs base)={s.get('mean_delta_margin_vs_baseline', float('nan')):.3f}")

    # -------------------------------------------------------------------------
    # Work 2: Transfer donor patching on flip-set
    # -------------------------------------------------------------------------
    patch_rows: List[Dict[str, Any]] = []
    patch_summary: Dict[str, Any] = {}
    donors_meta: List[Dict[str, Any]] = []
    patch_steps_final = set(patch_steps_user)

    if bool(args.run_transfer_patching):
        # Detect whether batched-candidate path is available.
        # If user requested steps beyond 0, but we are NOT batched, auto downgrade to step0.
        if any(s > 0 for s in patch_steps_user):
            probe_prompt = flip_used[0].prompt
            batched = detect_batched_candidate_path(
                base_mod, model, tok, layer_module, probe_prompt, candidate_labels, candidate_texts
            )
            if not batched:
                print("[Warn] Non-legacy cache path detected (no batched candidates). "
                      "Patching steps beyond 0 can cause shape mismatch / be ill-defined. "
                      "Auto-downgrading transfer patch_steps -> {0}.")
                patch_steps_final = {0}

        print(f"[TransferPatch] patch_steps_final={sorted(list(patch_steps_final))}")

        # -------------------------
        # Build donor pool
        # -------------------------
        donor_examples: List[Any] = []
        donor_tasks = [args.task] if args.donor_source == "same_task_eval" else parse_csv_list(args.donor_tasks)
        if args.donor_source == "cross_task_eval" and len(donor_tasks) == 0:
            raise ValueError("--donor_tasks must be provided when --donor_source=cross_task_eval")

        for t in donor_tasks:
            _, eval_by_d, meta_by_d = base_mod.load_selected_tasks_eval_only(
                dl,
                task=t,
                n_eval=int(args.donor_n_eval),
                seed=args.seed + 777,
                template_randomization=bool(args.template_randomization),
                template_seed=args.seed + 999 + 777,
                shuffle_choices=bool(args.shuffle_choices),
                answer_prefix=args.answer_prefix
            )
            exs = eval_by_d[t]
            donor_examples.extend(exs)
            print(f"[Donors] loaded task={t} n={len(exs)} meta={meta_by_d.get(t, {})}")

        # Optional filters
        if bool(args.donor_require_gold_in_candidates) or bool(args.donor_require_baseline_correct):
            filtered: List[Any] = []
            for ex in donor_examples:
                gold = (ex.gold or "").strip().upper()
                if bool(args.donor_require_gold_in_candidates) and (gold not in candidate_labels):
                    continue
                if bool(args.donor_require_baseline_correct):
                    if gold not in candidate_labels:
                        continue
                    base_res = base_mod.forced_choice_decode_aligned(
                        model, tok, ex.prompt,
                        candidate_labels, candidate_texts, gold,
                        layer_module=layer_module,
                        add_special_tokens_prompt=bool(args.add_special_tokens_prompt),
                    )
                    if not bool(getattr(base_res, "correct", False)):
                        continue
                filtered.append(ex)
            donor_examples = filtered

        if len(donor_examples) == 0:
            raise RuntimeError("Donor pool is empty after loading/filtering.")

        # Precompute donor shared vectors (by decode step)
        # Use dummy gold for capture run if needed.
        rng = np.random.default_rng(args.seed + 9999)
        donor_bank: List[Dict[str, Any]] = []

        # To keep memory bounded, we only keep up to N donors = max( len(flip_used), donor_n_eval*#tasks )
        # but you can change this if you want.
        max_donors_keep = min(len(donor_examples), max(len(flip_used), 256))
        donor_indices = list(range(len(donor_examples)))
        rng.shuffle(donor_indices)
        donor_indices = donor_indices[:max_donors_keep]

        print(f"[Donors] building donor_bank size={len(donor_indices)} (from total={len(donor_examples)})")

        for j, di in enumerate(donor_indices):
            exd = donor_examples[di]
            exd_id = exd.ex_id
            prompt_d = exd.prompt
            gold_d = (exd.gold or "").strip().upper()
            gold_for_run = gold_d if gold_d in candidate_labels else candidate_labels[0]

            cap = base_mod.DecodeStepHiddenCaptureHook(capture_steps=patch_steps_final)
            _ = base_mod.forced_choice_decode_aligned(
                model, tok, prompt_d,
                candidate_labels, candidate_texts, gold_for_run,
                layer_module=layer_module,
                capture_hook=cap,
                add_special_tokens_prompt=bool(args.add_special_tokens_prompt),
            )

            donor_by_step = {}
            for t in patch_steps_final:
                if t not in cap.hidden_by_step:
                    continue
                donor_by_step[int(t)] = base_mod.project_cpu(cap.hidden_by_step[int(t)], Qs)

            if len(donor_by_step) == 0:
                continue

            donor_bank.append({
                "donor_ex_id": exd_id,
                "donor_gold": gold_d,
                "donor_by_step": donor_by_step,  # CPU tensors
            })

        if len(donor_bank) == 0:
            raise RuntimeError("Donor bank is empty (failed to capture donor states).")

        donors_meta = [{
            "n_donor_bank": len(donor_bank),
            "donor_source": args.donor_source,
            "donor_tasks": donor_tasks,
            "donor_n_eval": int(args.donor_n_eval),
            "donor_pick": args.donor_pick,
            "donor_require_gold_in_candidates": bool(args.donor_require_gold_in_candidates),
            "donor_require_baseline_correct": bool(args.donor_require_baseline_correct),
        }]

        # -------------------------
        # Run transfer patching on flip-set
        # -------------------------
        print(f"[TransferPatch] running on flip-set size={len(flip_used)}")

        def pick_donor(i: int) -> Dict[str, Any]:
            if args.donor_pick == "cyclic":
                return donor_bank[i % len(donor_bank)]
            # random
            return donor_bank[int(rng.integers(0, len(donor_bank)))]

        for i, ex in enumerate(flip_used):
            ex_id = ex.ex_id
            prompt = ex.prompt
            gold = (ex.gold or "").strip().upper()

            base_d = baseline_cache[ex_id]
            ablt1_d = ablated1_cache[ex_id]

            # Transfer donor
            donor_item = pick_donor(i)
            donor_by_step = donor_item["donor_by_step"]

            remove1 = loto8.LastTokenRemovalHook(Qs, alpha=1.0, stats=loto8.HookStats("remove_shared_a1_transfer"))
            patched_transfer = base_mod.forced_choice_decode_aligned(
                model, tok, prompt,
                candidate_labels, candidate_texts, gold,
                layer_module=layer_module,
                removal_hook=remove1,
                patch_hook=base_mod.SubspacePatchHook(Qs, donor_by_step=donor_by_step, patch_steps=patch_steps_final),
                add_special_tokens_prompt=bool(args.add_special_tokens_prompt),
            )
            patched_transfer_d = fc_to_dict(patched_transfer)

            row = {
                "ex_id": ex_id,
                "gold": gold,
                "baseline": base_d,
                "ablated_1": ablt1_d,
                "patched_transfer": patched_transfer_d,
                "transfer_donor_ex_id": donor_item["donor_ex_id"],
                "transfer_donor_gold": donor_item["donor_gold"],
            }

            # Optional: self donor patch reference (same patch steps)
            if bool(args.run_self_patch_ref):
                cap_self = base_mod.DecodeStepHiddenCaptureHook(capture_steps=patch_steps_final)
                _ = base_mod.forced_choice_decode_aligned(
                    model, tok, prompt,
                    candidate_labels, candidate_texts, gold,
                    layer_module=layer_module,
                    capture_hook=cap_self,
                    add_special_tokens_prompt=bool(args.add_special_tokens_prompt),
                )
                self_donor = {}
                for t in patch_steps_final:
                    if t in cap_self.hidden_by_step:
                        self_donor[int(t)] = base_mod.project_cpu(cap_self.hidden_by_step[int(t)], Qs)

                remove1s = loto8.LastTokenRemovalHook(Qs, alpha=1.0, stats=loto8.HookStats("remove_shared_a1_selfref"))
                patched_self = base_mod.forced_choice_decode_aligned(
                    model, tok, prompt,
                    candidate_labels, candidate_texts, gold,
                    layer_module=layer_module,
                    removal_hook=remove1s,
                    patch_hook=base_mod.SubspacePatchHook(Qs, donor_by_step=self_donor, patch_steps=patch_steps_final),
                    add_special_tokens_prompt=bool(args.add_special_tokens_prompt),
                )
                row["patched_self"] = fc_to_dict(patched_self)

            patch_rows.append(row)

            print(f"[Flip {i+1}/{len(flip_used)}] ex_id={ex_id} gold={gold} "
                  f"ablt1={ablt1_d['pred_label']}({ablt1_d['correct']}) "
                  f"transfer={patched_transfer_d['pred_label']}({patched_transfer_d['correct']}) "
                  f"donor={donor_item['donor_ex_id']}")

        patch_summary = {
            "patched_transfer": summarize_patching(patch_rows, "patched_transfer"),
        }
        if bool(args.run_self_patch_ref):
            patch_summary["patched_self"] = summarize_patching(patch_rows, "patched_self")

        print("[TransferPatch Summary]")
        for kname, sval in patch_summary.items():
            print(f"  {kname:>16s}: rescued={sval.get('rescued', 0)}/{sval.get('n', 0)} "
                  f"({sval.get('rescued_pct', float('nan')):.1f}%) "
                  f"meanΔm(vs ablt)={sval.get('mean_delta_margin_vs_ablated', float('nan')):.3f}")

    # -------------------------------------------------------------------------
    # Write output
    # -------------------------------------------------------------------------
    out = {
        "meta": {
            "model": args.model,
            "device": args.device,
            "dtype": args.dtype,
            "layer": args.layer,
            "layers_path": path_used,
            "task": args.task,
            "eval_meta": eval_meta,
            "seed": args.seed,

            "candidate_labels": candidate_labels,
            "candidate_text_style": args.candidate_text_style,
            "candidate_token_lens": {lab: int(l) for lab, l in zip(candidate_labels, cand_lens)},
            "max_candidate_token_len": int(max_len),

            "Qs_path": args.Qs_path or args.Qs_out,
            "Qs_shape": [int(d), int(k)],

            "flipset_definition": {
                "alpha": 1.0,
                "criterion": "baseline correct AND ablated(alpha=1) wrong",
                "n_eval_loaded": int(len(eval_examples)),
                "flipset_total": int(n_flips_total),
                "flipset_used": int(n_flips_used),
            },

            "alpha_sweep": {
                "enabled": bool(args.run_alpha_sweep),
                "alpha_list": parse_float_list(args.alpha_list),
            },

            "transfer_patching": {
                "enabled": bool(args.run_transfer_patching),
                "patch_window_requested": args.patch_window,
                "patch_steps_requested": sorted(list(patch_steps_user)),
                "patch_steps_final": sorted(list(patch_steps_final)),
                "run_self_patch_ref": bool(args.run_self_patch_ref),
            },
        },

        # lightweight: keep scan_rows (can be big but still manageable at n_eval~256)
        "scan_rows": scan_rows,

        # Work 1
        "alpha_sweep_summary_on_flipset": alpha_summary,
        "alpha_sweep_rows_by_alpha": {
            str(a): rows for a, rows in alpha_rows_by_alpha.items()
        } if bool(args.run_alpha_sweep) else {},

        # Work 2
        "donors_meta": donors_meta,
        "transfer_patching_summary_on_flipset": patch_summary,
        "transfer_patching_rows": patch_rows,
    }

    ensure_dir(args.out_json)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[Done] Wrote {args.out_json}")


if __name__ == "__main__":
    main()
