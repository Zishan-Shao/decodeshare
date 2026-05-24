"""
disturb_energy_matched_sharedness_kmatch.py

Energy-matched controls (reviewer-friendly) for shared-subspace removal, using
`benchmark_dataloader(s).py` for loading + `eval_perf.py` for shared-basis estimation
and decode-aligned forced-choice evaluation (including gold logprob/margin metrics).

Fixes / changes vs earlier versions
-----------------------------------
- Uses benchmark loader's `Example` (dataset/ex_id/prompt/gold). **No `choices=` anywhere.**
  This fixes: `TypeError: Example.__init__() got an unexpected keyword argument 'choices'`.
- Uses eval_perf's:
    * decode-state collection (`collect_decode_states`)
    * shared-basis estimation (`compute_shared_basis_from_states`)
    * forced-choice + gold metrics (`forced_choice_logprob_eval`)
- Always computes k-match at least once (for alpha=1), even if `--kmatch_per_alpha=0`.

What this script does
---------------------
1) Load tasks with `load_selected_tasks()`.
2) Collect decode last-token hidden states per task and compute a shared basis Q_shared.
3) Build:
   - Structural control (fixed dim k=k_use): top-k non-shared components orthogonalized vs Q_shared.
   - Energy-matched control (k-match): choose k_c so mean removed energy matches shared removal
     on a prompt-boundary decode-last calibration distribution.
4) Forced-choice eval (no free-form generation parsing), with optional alpha sweep.

Outputs
-------
Writes:
  --out_json : structured results
  --out_txt  : summary text
  --out_md   : markdown tables
  --out_tex  : latex tables
"""
from __future__ import annotations

import os
import sys
import json
import math
import argparse
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.append(THIS_DIR)
PARENT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)


Example = None
load_selected_tasks = None
_bench_import_err: Optional[Exception] = None
for modname in ["decodeshare.benchmark_dataloaders"]:
    try:
        _m = __import__(modname, fromlist=["Example", "load_selected_tasks"])
        Example = getattr(_m, "Example")
        load_selected_tasks = getattr(_m, "load_selected_tasks")
        break
    except Exception as e:
        _bench_import_err = e
        continue
if Example is None or load_selected_tasks is None:
    raise ImportError(
        "Failed to import decodeshare.benchmark_dataloaders."
    ) from _bench_import_err

try:
    from decodeshare import eval_perf as ep  # type: ignore
except Exception as e:
    raise ImportError("Failed to import decodeshare.eval_perf.") from e


try:
    from decodeshare.subspace import get_model_layers  # type: ignore
except Exception as e:
    raise ImportError(
        "Failed to import decodeshare.subspace.get_model_layers."
    ) from e


def parse_csv_list(s: str) -> List[str]:
    items = [x.strip() for x in (s or "").split(",")]
    return [x for x in items if x]


def parse_csv_floats(s: str) -> List[float]:
    items = parse_csv_list(s)
    out: List[float] = []
    for x in items:
        try:
            out.append(float(x))
        except Exception as e:
            raise ValueError(f"Bad float in --alphas: '{x}'") from e
    return out


def ensure_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def maybe_apply_chat_template(tok, user_prompt: str, use_chat_template: bool) -> str:
    if not use_chat_template:
        return user_prompt
    if not hasattr(tok, "apply_chat_template"):
        return user_prompt
    msgs = [{"role": "user", "content": user_prompt}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        return user_prompt


def subsample_rows_np(x: np.ndarray, n_max: int, seed: int) -> np.ndarray:
    if n_max <= 0 or x.shape[0] <= n_max:
        return x
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.shape[0], size=int(n_max), replace=False)
    return x[idx]


def balance_tasks_np(task_to_x: Dict[str, np.ndarray], seed: int) -> Tuple[Dict[str, np.ndarray], int]:
    sizes = {t: int(v.shape[0]) for t, v in task_to_x.items()}
    if not sizes:
        return {}, 0
    min_n = min(sizes.values())
    out: Dict[str, np.ndarray] = {}
    for t, v in task_to_x.items():
        out[t] = subsample_rows_np(v, min_n, seed=ep.stable_int_seed(seed, "balance", t))
    return out, min_n


def project_out_np(A: np.ndarray, Q: np.ndarray) -> np.ndarray:
    if Q is None or Q.size == 0:
        return A
    return A - Q @ (Q.T @ A)


def projection_energy_stats(H: np.ndarray, Q: np.ndarray) -> Dict[str, float]:
    Hf = H.astype(np.float32, copy=False)
    Qf = Q.astype(np.float32, copy=False)
    Z = Hf @ Qf
    proj_e = np.sum(Z * Z, axis=1)
    tot_e = np.sum(Hf * Hf, axis=1) + 1e-12
    ratio = proj_e / tot_e
    return {
        "mean_ratio": float(ratio.mean()),
        "mean_energy": float(proj_e.mean()),
    }


def make_random_orthonormal(
    D: int, k: int, seed: int, orthogonal_to: Optional[np.ndarray] = None
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal(size=(D, int(k))).astype(np.float32)
    if orthogonal_to is not None and orthogonal_to.size > 0:
        A = project_out_np(A, orthogonal_to.astype(np.float32, copy=False))
    Q = ep.orthonormalize_np(A)
    return Q[:, :k]


def k_match_from_curve(mean_by_k: np.ndarray, target: float, k_min: int, k_max: int) -> int:
    if mean_by_k.size == 0:
        return int(k_min)
    idx = int(np.argmin(np.abs(mean_by_k - float(target)))) + 1
    idx = max(int(k_min), idx)
    idx = min(int(k_max), idx)
    return int(idx)


class PromptBoundaryDecodeCollector:
    """
    Collect last-token hidden states on the PROMPT-BOUNDARY decode step.
    Hook condition: seq_len == 1
    """
    def __init__(self, layer_indices: List[int]):
        self.layer_indices = list(layer_indices)
        self.cur_task: Optional[str] = None
        self.storage: Dict[str, Dict[int, List[np.ndarray]]] = {}

    def set_current_task(self, task: str) -> None:
        self.cur_task = task
        self.storage.setdefault(task, {})

    def make_hook(self, layer_idx: int):
        def _hook(module, inputs, output):
            if self.cur_task is None:
                return output
            hs = output[0] if isinstance(output, tuple) else output
            if not isinstance(hs, torch.Tensor) or hs.ndim != 3:
                return output
            if hs.shape[1] != 1:
                return output
            x = hs[:, -1, :]
            self.storage.setdefault(self.cur_task, {}).setdefault(layer_idx, []).append(
                x.detach().float().cpu().numpy()
            )
            return output
        return _hook

    def get(self, task: str, layer_idx: int) -> Optional[np.ndarray]:
        chunks = self.storage.get(task, {}).get(layer_idx, [])
        if not chunks:
            return None
        return np.concatenate(chunks, axis=0)


@torch.no_grad()
def collect_prompt_boundary_decode_states(
    model,
    tok,
    prompts: List[str],
    collector: PromptBoundaryDecodeCollector,
    *,
    batch_size: int,
    max_prompt_len: int,
    use_chat_template: bool,
) -> None:
    device = next(model.parameters()).device
    model.eval()
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    for i in tqdm(range(0, len(prompts), batch_size), desc="CollectPromptBoundary"):
        batch_raw = prompts[i:i + batch_size]
        batch = [maybe_apply_chat_template(tok, p, use_chat_template) for p in batch_raw]
        inputs = tok(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(max_prompt_len),
        ).to(device)
        ids = inputs["input_ids"]
        attn = inputs["attention_mask"]
        _past, _logits = ep.cache_decode_aligned_boundary(model, ids, attn)


def fmt_ci(acc: float, lo: float, hi: float) -> str:
    return f"{acc * 100:.1f} [{lo * 100:.1f}, {hi * 100:.1f}]"


def main() -> None:
    ap = argparse.ArgumentParser()


    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--use_chat_template", type=int, default=0)


    ap.add_argument(
        "--tasks",
        type=str,
        default="gsm8k,commonsenseqa,strategyqa,aqua,arc_challenge,openbookqa,qasc,logiqa,boolq",
    )
    ap.add_argument("--n_prompts", type=int, default=128)
    ap.add_argument("--eval_n", type=int, default=256)
    ap.add_argument("--max_prompt_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=4)


    ap.add_argument("--calib_max_new_tokens", type=int, default=128)
    ap.add_argument("--per_task_max_states", type=int, default=20000)
    ap.add_argument("--pca_var", type=float, default=0.95)
    ap.add_argument("--tau", type=float, default=0.001)
    ap.add_argument("--m_shared", type=str, default="all")
    ap.add_argument("--k_eval", type=int, default=0)


    ap.add_argument("--answer_prefix", type=str, default="\nFinal answer:")
    ap.add_argument("--fc_prefix_mode", type=str, default="auto", choices=["auto", "always", "never"])
    ap.add_argument("--warmup_tokens", type=int, default=0)
    ap.add_argument("--warmup_phrase", type=str, default=" Let's think step by step.")
    ap.add_argument("--template_randomization", type=int, default=0)
    ap.add_argument("--shuffle_choices", type=int, default=0)
    ap.add_argument("--template_seed", type=int, default=123)
    ap.add_argument("--save_scores", type=int, default=1, help="Include scores_sum in JSON (bigger files).")


    ap.add_argument("--alphas", type=str, default="1.0")
    ap.add_argument("--kmatch_per_alpha", type=int, default=int(os.getenv("KMATCH_PER_ALPHA", "0")))
    ap.add_argument("--alpha_ctrl_cap", type=float, default=0.0)


    ap.add_argument("--bootstrap_iters", type=int, default=5000)
    ap.add_argument("--perm_iters", type=int, default=10000)
    ap.add_argument("--ci_alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)


    ap.add_argument("--out_json", type=str, default=os.path.join(THIS_DIR, "results", "energy_kmatch_results.json"))
    ap.add_argument("--out_txt", type=str, default=os.path.join(THIS_DIR, "results", "energy_kmatch_summary.txt"))
    ap.add_argument("--out_md", type=str, default=os.path.join(THIS_DIR, "results", "energy_kmatch_summary.md"))
    ap.add_argument("--out_tex", type=str, default=os.path.join(THIS_DIR, "results", "energy_kmatch_tables.tex"))

    args = ap.parse_args()

    for p in [args.out_json, args.out_txt, args.out_md, args.out_tex]:
        ensure_dir(p)

    ep.set_global_seed(int(args.seed))

    layer_indices = [int(args.layer)]
    use_chat_template = bool(int(args.use_chat_template))
    kmatch_per_alpha = bool(int(args.kmatch_per_alpha))
    save_scores = bool(int(args.save_scores))

    alphas = parse_csv_floats(args.alphas)
    if not alphas:
        alphas = [1.0]


    model, tok = ep.load_model_and_tokenizer(args.model, args.device, args.dtype)
    hidden_dim = getattr(model.config, "hidden_size", None) or getattr(model.config, "n_embd", None)
    if hidden_dim is None:
        raise RuntimeError("Cannot infer hidden_dim from model.config")

    tasks = parse_csv_list(args.tasks)
    if not tasks:
        raise ValueError("--tasks is empty")

    print(f"[Env] model={args.model} dtype={args.dtype} device={args.device} layer={layer_indices} hidden_dim={hidden_dim}")
    print(f"[Cfg] tasks={tasks}")
    print(f"[Cfg] alphas={alphas} kmatch_per_alpha={int(kmatch_per_alpha)}")


    sub_by, eval_by, meta = load_selected_tasks(
        tasks=tasks,
        n_subspace=int(args.n_prompts),
        n_eval=int(args.eval_n),
        seed=int(args.seed),
        template_seed=int(args.template_seed),
        template_randomization=bool(int(args.template_randomization)),
        shuffle_choices=bool(int(args.shuffle_choices)),
        add_answer_prefix=False,
        answer_prefix=args.answer_prefix,
    )


    layers, _ = get_model_layers(model)
    dec_col = ep.DecodeLastTokenCollector(layer_indices)
    handles = []
    for li in layer_indices:
        if li >= len(layers):
            raise ValueError(f"Layer {li} out of range (n_layers={len(layers)})")
        handles.append(layers[li].register_forward_hook(dec_col.make_hook(li)))

    decode_task_states: Dict[str, np.ndarray] = {}
    try:
        for task, exs in sub_by.items():
            prompts = [ex.prompt for ex in exs]
            prompts = [maybe_apply_chat_template(tok, p, use_chat_template) for p in prompts]

            print(f"[CollectDecode] task={task} prompts={len(prompts)}")
            dec_col.set_current_task(task)

            ep.collect_decode_states(
                model,
                tok,
                prompts,
                dec_col,
                batch_size=int(args.batch_size),
                max_new_tokens=int(args.calib_max_new_tokens),
                max_prompt_len=int(args.max_prompt_len),
                decoding="greedy",
                temperature=1.0,
                top_p=1.0,
                top_k=0,
            )

            acts = dec_col.get(task, int(args.layer))
            if acts is None or acts.shape[0] == 0:
                print(f"[CollectDecode][WARN] task={task}: no decode states; skipping in basis")
                continue

            acts = acts.astype(np.float32, copy=False)
            acts = subsample_rows_np(
                acts,
                int(args.per_task_max_states),
                seed=ep.stable_int_seed(args.seed, "subsample_decode", task),
            )
            decode_task_states[task] = acts
            print(f"[CollectDecode] task={task} states={acts.shape[0]} x {acts.shape[1]}")
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass
        dec_col.set_capture(False, None)

    if len(decode_task_states) < 2:
        raise RuntimeError("Need >=2 tasks with decode states to compute shared basis.")


    joint_subspace, shared_idx, extra = ep.compute_shared_basis_from_states(
        decode_task_states,
        pca_var=float(args.pca_var),
        min_dim=1,
        max_dim=int(hidden_dim),
        tau=float(args.tau),
        m_shared=args.m_shared,
        seed=int(args.seed),
    )
    cross_dim = int(extra.get("cross_dim", joint_subspace.shape[1]))
    shared_idx = sorted(list(shared_idx))
    k_shared = len(shared_idx)
    if k_shared <= 0:
        raise RuntimeError("No shared basis found; try smaller tau or different m_shared.")

    k_use = k_shared if int(args.k_eval) <= 0 else min(int(args.k_eval), k_shared)
    if k_use <= 0:
        raise RuntimeError("k_use=0 (check --k_eval).")

    Q_shared_full = ep.orthonormalize_np(joint_subspace[:, shared_idx].astype(np.float32, copy=False))
    Q_shared = Q_shared_full[:, :k_use]


    nonshared_idx = [i for i in range(cross_dim) if i not in set(shared_idx)]
    if len(nonshared_idx) < k_use:
        raise RuntimeError(f"Non-shared pool too small: {len(nonshared_idx)} < k_use={k_use}")
    B_pool = joint_subspace[:, nonshared_idx].astype(np.float32, copy=False)
    B_pool_ortho = project_out_np(B_pool, Q_shared)
    Q_pool = ep.orthonormalize_np(B_pool_ortho)
    K_pool = int(Q_pool.shape[1])
    if K_pool < k_use:
        raise RuntimeError(f"Q_pool rank too small: K_pool={K_pool} < k_use={k_use}")
    Q_ctrl_struct = Q_pool[:, :k_use]


    pb_col = PromptBoundaryDecodeCollector(layer_indices)
    layers, _ = get_model_layers(model)
    handles = []
    for li in layer_indices:
        handles.append(layers[li].register_forward_hook(pb_col.make_hook(li)))

    prompt_boundary_task_states: Dict[str, np.ndarray] = {}
    try:
        for task, exs in sub_by.items():
            if task not in decode_task_states:
                continue
            prompts = [ex.prompt for ex in exs]
            prompts = [maybe_apply_chat_template(tok, p, use_chat_template) for p in prompts]
            pb_col.set_current_task(task)
            collect_prompt_boundary_decode_states(
                model,
                tok,
                prompts,
                pb_col,
                batch_size=int(args.batch_size),
                max_prompt_len=int(args.max_prompt_len),
                use_chat_template=use_chat_template,
            )
            acts = pb_col.get(task, int(args.layer))
            if acts is None or acts.shape[0] == 0:
                raise RuntimeError(f"No prompt-boundary states for task={task}")
            prompt_boundary_task_states[task] = acts.astype(np.float32, copy=False)
            print(f"[EnergyCalib] task={task} states={acts.shape[0]} x {acts.shape[1]}")
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass

    pb_bal, n_bal = balance_tasks_np(prompt_boundary_task_states, seed=int(args.seed) + 999)
    if n_bal <= 0:
        raise RuntimeError("No prompt-boundary states after balancing.")
    tasks_used = sorted(list(pb_bal.keys()))
    H_calib = np.concatenate([pb_bal[t] for t in tasks_used], axis=0)
    print(f"[EnergyCalib] balanced per task={n_bal}, pooled H_calib={H_calib.shape[0]} x {H_calib.shape[1]}")


    stats_shared = projection_energy_stats(H_calib, Q_shared)
    stats_ctrl_struct = projection_energy_stats(H_calib, Q_ctrl_struct)

    Z_pool = H_calib @ Q_pool
    cum_energy = np.cumsum(Z_pool * Z_pool, axis=1)
    mean_by_k = cum_energy.mean(axis=0)


    target_alpha1 = float(stats_shared["mean_energy"])
    k_c_alpha1 = k_match_from_curve(mean_by_k, target_alpha1, k_min=k_use, k_max=K_pool)


    Es = float(stats_shared["mean_energy"])
    Ec = float(stats_ctrl_struct["mean_energy"])
    alpha_match_scale = math.sqrt(Es / max(Ec, 1e-12))

    print("\n" + "=" * 80)
    print("[Sanity / Energy]")
    print(f"  cross_dim={cross_dim}  k_shared={k_shared}  k_use={k_use}  K_pool={K_pool}")
    print(f"  k_c(alpha=1, K-match)= {k_c_alpha1}")
    print(f"  shared mean||P h||^2={Es:.4e}  ctrl_struct mean||P h||^2={Ec:.4e}  alpha_match_scale={alpha_match_scale:.4f}")
    if k_c_alpha1 == K_pool:
        ratio = float(mean_by_k[-1] / max(Es, 1e-12))
        print(f"  [WARN] Pool saturates at K_pool; max_energy/target={ratio:.4f}")
    print("=" * 80 + "\n")


    warmup_ids: List[int] = []
    if int(args.warmup_tokens) > 0:
        base_ids = tok.encode(args.warmup_phrase, add_special_tokens=False)
        if len(base_ids) == 0:
            raise ValueError("warmup_phrase tokenizes to empty; choose a different phrase.")
        W = int(args.warmup_tokens)
        warmup_ids = (base_ids * (W // len(base_ids) + 1))[:W]
    print(f"[FC] warmup_ids_len={len(warmup_ids)} warmup_phrase={repr(args.warmup_phrase)}")


    def task_has_candidates(t: str) -> bool:
        try:
            return len(ep.candidate_strings(t)) > 0
        except Exception:
            return False

    eval_tasks = [t for t in tasks if t in eval_by and task_has_candidates(t)]
    if not eval_tasks:
        raise RuntimeError("No forced-choice tasks available for evaluation (check --tasks).")


    eval_examples: Dict[str, List[Any]] = {}
    warmup_token_ids_cache: Dict[str, Optional[np.ndarray]] = {}
    for task in eval_tasks:
        exs = eval_by[task]
        if use_chat_template:
            exs = [Example(ex.dataset, ex.ex_id, maybe_apply_chat_template(tok, ex.prompt, True), ex.gold) for ex in exs]
        eval_examples[task] = exs
        N = len(exs)
        if warmup_ids:
            warmup_token_ids_cache[task] = np.tile(np.array(warmup_ids, dtype=np.int64)[None, :], (N, 1))
        else:
            warmup_token_ids_cache[task] = None


    baseline_cache: Dict[str, Dict[str, Any]] = {}
    for task in eval_tasks:
        exs = eval_examples[task]
        run_base = ep.forced_choice_logprob_eval(
            model,
            tok,
            exs,
            task,
            layer_indices=layer_indices,
            basis_np=None,
            alpha=0.0,
            batch_size=int(args.batch_size),
            max_prompt_len=int(args.max_prompt_len),
            warmup_token_ids=warmup_token_ids_cache[task],
            answer_prefix=args.answer_prefix,
            prefix_mode=args.fc_prefix_mode,
            save_scores=save_scores,
        )
        acc_b, lo_b, hi_b = ep.bootstrap_ci_mean(
            np.array(run_base["correct"], dtype=np.float32),
            iters=int(args.bootstrap_iters),
            alpha=float(args.ci_alpha),
            seed=ep.stable_int_seed(args.seed, task, "baseline"),
        )
        baseline_cache[task] = {
            **run_base,
            "acc": float(acc_b),
            "ci_low": float(lo_b),
            "ci_high": float(hi_b),
        }


    results: Dict[str, Any] = {
        "config": {
            "model": args.model,
            "device": args.device,
            "dtype": args.dtype,
            "layer_indices": layer_indices,
            "seed": int(args.seed),
            "tasks": tasks,
            "n_prompts": int(args.n_prompts),
            "eval_n": int(args.eval_n),
            "max_prompt_len": int(args.max_prompt_len),
            "batch_size": int(args.batch_size),
            "calib_max_new_tokens": int(args.calib_max_new_tokens),
            "per_task_max_states": int(args.per_task_max_states),
            "pca_var": float(args.pca_var),
            "tau": float(args.tau),
            "m_shared": args.m_shared,
            "k_eval": int(args.k_eval),
            "answer_prefix": args.answer_prefix,
            "fc_prefix_mode": args.fc_prefix_mode,
            "warmup_tokens": int(args.warmup_tokens),
            "warmup_phrase": args.warmup_phrase,
            "template_randomization": int(args.template_randomization),
            "shuffle_choices": int(args.shuffle_choices),
            "template_seed": int(args.template_seed),
            "alphas": alphas,
            "kmatch_per_alpha": int(kmatch_per_alpha),
            "k_c_alpha1": int(k_c_alpha1),
            "alpha_ctrl_cap": float(args.alpha_ctrl_cap),
            "save_scores": int(save_scores),
        },
        "basis": {
            "hidden_dim": int(hidden_dim),
            "cross_dim": int(cross_dim),
            "shared_idx": shared_idx,
            "k_shared": int(k_shared),
            "k_use": int(k_use),
            "K_pool": int(K_pool),
        },
        "energy_calib": {
            "tasks_used": tasks_used,
            "balanced_per_task": int(n_bal),
            "stats_shared": stats_shared,
            "stats_ctrl_struct": stats_ctrl_struct,
            "alpha_match_scale": float(alpha_match_scale),
            "k_c_alpha1": int(k_c_alpha1),
        },
        "baseline": {},
        "alpha_runs": {},
    }


    for task in eval_tasks:
        b = baseline_cache[task]
        results["baseline"][task] = {
            "n": int(len(eval_examples[task])),
            "acc": float(b["acc"]),
            "ci_low": float(b["ci_low"]),
            "ci_high": float(b["ci_high"]),
            "metrics_summary": b.get("metrics_summary", {}),
        }

    summary_lines: List[str] = []
    md_lines: List[str] = []
    tex_blocks: List[str] = []

    summary_lines.append("=" * 80)
    summary_lines.append("ENERGY K-MATCH + ALPHA SWEEP (decode-aligned forced-choice, eval_perf)")
    summary_lines.append("=" * 80)
    summary_lines.append(f"Model={args.model} dtype={args.dtype} device={args.device} layer={layer_indices}")
    summary_lines.append(f"Tasks={tasks}")
    summary_lines.append(f"cross_dim={cross_dim} k_shared={k_shared} k_use={k_use} K_pool={K_pool}")
    summary_lines.append(f"k_c(alpha=1)={k_c_alpha1}  alpha_match_scale={alpha_match_scale:.4f}")
    summary_lines.append("")

    md_lines.append("# Energy K-Match + Alpha Sweep (decode-aligned)")
    md_lines.append("")
    md_lines.append(f"- **Model**: `{args.model}`")
    md_lines.append(f"- **Layer**: `{layer_indices}`  | **dtype**: `{args.dtype}`  | **device**: `{args.device}`")
    md_lines.append(f"- **Tasks**: `{', '.join(tasks)}`")
    md_lines.append(f"- **cross_dim**: `{cross_dim}`  | **k_shared**: `{k_shared}`  | **k_use**: `{k_use}`  | **K_pool**: `{K_pool}`")
    md_lines.append(f"- **k_c(alpha=1)**: `{k_c_alpha1}`  | **kmatch_per_alpha**: `{int(kmatch_per_alpha)}`")
    md_lines.append(f"- **alpha_match_scale**: `{alpha_match_scale:.4f}`")
    md_lines.append(f"- **answer_prefix**: `{args.answer_prefix}`  | **prefix_mode**: `{args.fc_prefix_mode}`  | **warmup_tokens**: `{args.warmup_tokens}`")
    md_lines.append("")


    base_headers = ["Task", "N", "Baseline acc (95% CI)"]
    base_rows = []
    for task in eval_tasks:
        b = baseline_cache[task]
        base_rows.append([task, str(len(eval_examples[task])), fmt_ci(b["acc"], b["ci_low"], b["ci_high"])])
    md_lines.append("## Baseline")
    md_lines.append("")

    md_lines.append(ep.md_table(base_rows, base_headers))
    md_lines.append("")


    for alpha_shared in alphas:
        alpha_shared_f = float(alpha_shared)

        alpha_ctrl = alpha_shared_f * alpha_match_scale
        if float(args.alpha_ctrl_cap) > 0:
            cap = abs(float(args.alpha_ctrl_cap))
            alpha_ctrl = float(np.clip(alpha_ctrl, -cap, cap))


        if kmatch_per_alpha:
            target = (alpha_shared_f ** 2) * Es
            k_c = k_match_from_curve(mean_by_k, target, k_min=k_use, k_max=K_pool)
        else:
            k_c = k_c_alpha1
        Q_ctrl_energy = Q_pool[:, :k_c]

        alpha_key = f"{alpha_shared_f:.6g}"
        results["alpha_runs"][alpha_key] = {
            "alpha_shared": alpha_shared_f,
            "alpha_ctrl": float(alpha_ctrl),
            "k_c": int(k_c),
            "by_task": {},
        }

        print("\n" + "-" * 80)
        print(f"[Alpha] alpha_shared={alpha_shared_f:.4g}  alpha_ctrl={alpha_ctrl:.4g}  k_c={k_c}  (kmatch_per_alpha={int(kmatch_per_alpha)})")
        print("-" * 80)

        headers = ["Task", "N", "Baseline", f"Shared(alpha={alpha_shared_f:g})", "Ctrl alpha-match", f"Ctrl k-match(k={k_c})"]
        md_rows: List[List[str]] = []
        tex_rows: List[List[str]] = []

        for task in eval_tasks:
            exs = eval_examples[task]
            N = len(exs)
            if N == 0:
                continue

            warm = warmup_token_ids_cache[task]
            base = baseline_cache[task]


            run_shared = ep.forced_choice_logprob_eval(
                model,
                tok,
                exs,
                task,
                layer_indices=layer_indices,
                basis_np=Q_shared,
                alpha=float(alpha_shared_f),
                batch_size=int(args.batch_size),
                max_prompt_len=int(args.max_prompt_len),
                warmup_token_ids=warm,
                answer_prefix=args.answer_prefix,
                prefix_mode=args.fc_prefix_mode,
                save_scores=save_scores,
            )
            acc_s, lo_s, hi_s = ep.bootstrap_ci_mean(
                np.array(run_shared["correct"], dtype=np.float32),
                iters=int(args.bootstrap_iters),
                alpha=float(args.ci_alpha),
                seed=ep.stable_int_seed(args.seed, task, "shared", alpha_key),
            )


            run_ctrl_alpha = ep.forced_choice_logprob_eval(
                model,
                tok,
                exs,
                task,
                layer_indices=layer_indices,
                basis_np=Q_ctrl_struct,
                alpha=float(alpha_ctrl),
                batch_size=int(args.batch_size),
                max_prompt_len=int(args.max_prompt_len),
                warmup_token_ids=warm,
                answer_prefix=args.answer_prefix,
                prefix_mode=args.fc_prefix_mode,
                save_scores=save_scores,
            )
            acc_ca, lo_ca, hi_ca = ep.bootstrap_ci_mean(
                np.array(run_ctrl_alpha["correct"], dtype=np.float32),
                iters=int(args.bootstrap_iters),
                alpha=float(args.ci_alpha),
                seed=ep.stable_int_seed(args.seed, task, "ctrl_alpha", alpha_key),
            )


            run_ctrl_k = ep.forced_choice_logprob_eval(
                model,
                tok,
                exs,
                task,
                layer_indices=layer_indices,
                basis_np=Q_ctrl_energy,
                alpha=1.0,
                batch_size=int(args.batch_size),
                max_prompt_len=int(args.max_prompt_len),
                warmup_token_ids=warm,
                answer_prefix=args.answer_prefix,
                prefix_mode=args.fc_prefix_mode,
                save_scores=save_scores,
            )
            acc_ck, lo_ck, hi_ck = ep.bootstrap_ci_mean(
                np.array(run_ctrl_k["correct"], dtype=np.float32),
                iters=int(args.bootstrap_iters),
                alpha=float(args.ci_alpha),
                seed=ep.stable_int_seed(args.seed, task, "ctrl_kmatch", alpha_key),
            )


            base_arr = np.array(base["correct"], dtype=np.float32)
            sh_arr = np.array(run_shared["correct"], dtype=np.float32)
            ca_arr = np.array(run_ctrl_alpha["correct"], dtype=np.float32)
            ck_arr = np.array(run_ctrl_k["correct"], dtype=np.float32)

            seed0 = ep.stable_int_seed(args.seed, task, "paired", alpha_key)
            paired = {
                "shared_vs_base": ep.summarize_paired(base_arr, sh_arr, f"{task}:shared_vs_base@a={alpha_key}", int(args.bootstrap_iters), int(args.perm_iters), float(args.ci_alpha), seed0 + 1),
                "ctrl_alpha_vs_base": ep.summarize_paired(base_arr, ca_arr, f"{task}:ctrl_alpha_vs_base@a={alpha_key}", int(args.bootstrap_iters), int(args.perm_iters), float(args.ci_alpha), seed0 + 2),
                "shared_vs_ctrl_alpha": ep.summarize_paired(ca_arr, sh_arr, f"{task}:shared_vs_ctrl_alpha@a={alpha_key}", int(args.bootstrap_iters), int(args.perm_iters), float(args.ci_alpha), seed0 + 3),
                "shared_vs_ctrl_kmatch": ep.summarize_paired(ck_arr, sh_arr, f"{task}:shared_vs_ctrl_kmatch@a={alpha_key}", int(args.bootstrap_iters), int(args.perm_iters), float(args.ci_alpha), seed0 + 4),
            }


            if save_scores:
                runs_out = {
                    "baseline": base,
                    "shared": {"acc": float(acc_s), "ci_low": float(lo_s), "ci_high": float(hi_s), **run_shared},
                    "ctrl_alpha": {"acc": float(acc_ca), "ci_low": float(lo_ca), "ci_high": float(hi_ca), **run_ctrl_alpha},
                    "ctrl_kmatch": {"acc": float(acc_ck), "ci_low": float(lo_ck), "ci_high": float(hi_ck), **run_ctrl_k},
                }
            else:
                runs_out = {
                    "baseline": {"acc": float(base["acc"]), "ci_low": float(base["ci_low"]), "ci_high": float(base["ci_high"]), "metrics_summary": base.get("metrics_summary", {})},
                    "shared": {"acc": float(acc_s), "ci_low": float(lo_s), "ci_high": float(hi_s), "metrics_summary": run_shared.get("metrics_summary", {})},
                    "ctrl_alpha": {"acc": float(acc_ca), "ci_low": float(lo_ca), "ci_high": float(hi_ca), "metrics_summary": run_ctrl_alpha.get("metrics_summary", {})},
                    "ctrl_kmatch": {"acc": float(acc_ck), "ci_low": float(lo_ck), "ci_high": float(hi_ck), "metrics_summary": run_ctrl_k.get("metrics_summary", {})},
                }

            results["alpha_runs"][alpha_key]["by_task"][task] = {
                "n": int(N),
                "runs": runs_out,
                "paired": paired,
            }

            print(
                f"  {task:12s} "
                f"base={fmt_ci(base['acc'], base['ci_low'], base['ci_high'])}  "
                f"shared={fmt_ci(acc_s, lo_s, hi_s)}  "
                f"ctrl_alpha={fmt_ci(acc_ca, lo_ca, hi_ca)}  "
                f"ctrlk={fmt_ci(acc_ck, lo_ck, hi_ck)}"
            )

            md_rows.append([
                task,
                str(N),
                fmt_ci(base["acc"], base["ci_low"], base["ci_high"]),
                fmt_ci(float(acc_s), float(lo_s), float(hi_s)),
                fmt_ci(float(acc_ca), float(lo_ca), float(hi_ca)),
                fmt_ci(float(acc_ck), float(lo_ck), float(hi_ck)),
            ])
            tex_rows.append([
                task,
                str(N),
                f"{base['acc']*100:.1f}",
                f"{acc_s*100:.1f}",
                f"{acc_ca*100:.1f}",
                f"{acc_ck*100:.1f}",
            ])

        summary_lines.append("-" * 80)
        summary_lines.append(f"alpha_shared={alpha_shared_f:.6g}  alpha_ctrl={alpha_ctrl:.6g}  k_c={k_c} (kmatch_per_alpha={int(kmatch_per_alpha)})")
        summary_lines.append("")

        md_lines.append(f"## Alpha = {alpha_shared_f:g}")
        md_lines.append("")
        md_lines.append(f"- alpha_ctrl (alpha-match): `{alpha_ctrl:.6g}`")
        md_lines.append(f"- k_c (k-match, alpha=1): `{k_c}` (kmatch_per_alpha={int(kmatch_per_alpha)})")
        md_lines.append("")

        md_lines.append(ep.md_table(md_rows, headers))
        md_lines.append("")

        tex_blocks.append(
            ep.latex_table(
                headers=headers,
                rows=tex_rows,
                caption=f"Forced-choice accuracy (\\%) at $\\alpha={alpha_shared_f:g}$ (k-match uses $\\alpha=1$).",
                label=f"tab:energy_kmatch_alpha_{alpha_key.replace('.', '_')}",
                colspec="l" + "c" * (len(headers) - 1),
            )
        )


    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with open(args.out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    with open(args.out_tex, "w", encoding="utf-8") as f:
        f.write("\n\n".join(tex_blocks))

    print("\n" + "\n".join(summary_lines))
    print("\n" + "=" * 80)
    print("[Done]")
    print(f"[Done] JSON: {args.out_json}")
    print(f"[Done] TXT : {args.out_txt}")
    print(f"[Done] MD  : {args.out_md}")
    print(f"[Done] TEX : {args.out_tex}")
    print("=" * 80)


if __name__ == "__main__":
    main()
