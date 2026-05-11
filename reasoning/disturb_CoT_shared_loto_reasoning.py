# -*- coding: utf-8 -*-
"""disturb_CoT_shared_loto_reasoning.py (refactored)

This is a lightweight *LOTO* runner that delegates:
  - **Data loading / prompt construction / parsing** -> `benchmark_dataloaders.py`
  - **Evaluation (generation + forced-choice)**      -> `eval_perf.py`
  - **Shared space computation**                     -> `eval_perf.compute_decode_prefill_shared_bases()`

Why this rewrite:
  - Your original script contained a lot of duplicated evaluation loops and
    dataset loading logic.
  - The attached `eval_perf.py` already implements:
      * decode-aligned generation
      * robust forced-choice logprob evaluation (with warmup + answer-prefix anchoring)
      * decode/prefill shared-basis computation across tasks

What this file keeps:
  - CLI interface
  - LOTO folding logic (held-out vs all)
  - Warmup-token caching orchestration (optional)
  - JSON/Markdown outputs

Important difference vs the original monolithic script:
  - This refactor uses the *single* decode-only removal hook from `eval_perf.py`.
    It does **not** include the original “staged removal during the first N reasoning tokens”
    because that logic is not implemented in `eval_perf.py`.
    If you still need staged removal, the cleanest way is to implement it once inside
    `eval_perf.py` so every experiment script can reuse it.

Expected file layout
--------------------
Put these three files in the same folder (or ensure they are on PYTHONPATH):
  - disturb_CoT_shared_loto_reasoning.py   (this file)
  - eval_perf.py                          (attached)
  - benchmark_dataloaders.py              (attached)

Example
-------

NOTE: if there is no shuffle_choices, fp32 dtype, and add_answer_prefix, the results will be different from the original script.

CUDA_VISIBLE_DEVICES=3 python disturb_CoT_shared_loto_reasoning.py \
  --model meta-llama/Llama-2-7b-chat-hf --device cuda --model_dtype fp32 \
  --mode loto --loto_eval_mode heldout \
  --tasks gsm8k,commonsenseqa,strategyqa,aqua,arc_challenge,openbookqa,qasc,logiqa \
  --layer 10 --n_subspace 128 --n_eval 2048 \
  --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
  --max_new_tokens 256 --do_sample 0 \
  --use_forced_choice 1 --fc_warmup_tokens 0 --fc_prefix_mode auto \
  --add_answer_prefix 1 --answer_prefix $'\nFinal answer:' \
  --out_json results.json --out_md results.md
  
  
  
# thie one is correct

 CUDA_VISIBLE_DEVICES=3 python disturb_CoT_shared_loto_reasoning.py \
    --model meta-llama/Llama-2-7b-chat-hf --device cuda --model_dtype fp32 \
    --mode loto --loto_eval_mode heldout \
    --tasks gsm8k,commonsenseqa,strategyqa,aqua,arc_challenge,openbookqa,qasc,logiqa \
    --layer 10 --n_subspace 128 --n_eval 2048 \
    --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
    --reasoning_tokens 128 --max_new_tokens 256 \
    --template_randomization 1 --shuffle_choices 1 \
    --add_answer_prefix 1 --answer_prefix $'\nFinal answer:' \
    --use_forced_choice 1 \
    --fc_warmup_tokens 0 \
    --fc_prefix_mode auto --fc_answer_prefix $'\nFinal answer:' \
    --do_sample 0 \
    --out_json energy_balance_loto8_reasoning_fc_eval2048.json --out_md energy_balance_loto8_reasoning_fc_eval2048.md

  
"""

from __future__ import annotations

import os
import sys
import json
import argparse
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


# -----------------------------------------------------------------------------
# Import attached utilities (eval_perf + benchmark_dataloaders)
# -----------------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# Make sure local dir wins (so `eval_perf.py` next to this script is found)
sys.path.insert(0, THIS_DIR)
# Your original script added project root for `joint_subspace_large.*`
sys.path.insert(0, os.path.join(THIS_DIR, ".."))

try:
    import eval_perf as EP  # attached
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Failed to import eval_perf.py.\n"
        "Put eval_perf.py next to this script, or ensure it's on PYTHONPATH.\n"
        f"Import error: {e}"
    ) from e


# Re-export commonly used names for readability
Example = EP.Example
load_selected_tasks = EP.load_selected_tasks


# -----------------------------------------------------------------------------
# Small compatibility patch: eval_perf.candidate_strings has a known piqa mismatch
# -----------------------------------------------------------------------------
_ORIG_CANDIDATE_STRINGS = EP.candidate_strings

def _candidate_strings_patched(task: str) -> List[str]:
    t = (task or "").strip().lower()
    if t == "piqa":
        return ["A", "B"]
    # keep eval_perf defaults otherwise
    return _ORIG_CANDIDATE_STRINGS(task)

EP.candidate_strings = _candidate_strings_patched  # monkey-patch within this process


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def json_default(o: Any) -> Any:
    # Use eval_perf's helper if present
    try:
        return EP.json_default(o)
    except Exception:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        return str(o)


def fmt_acc(acc: float, lo: float, hi: float) -> str:
    return f"{acc*100:.1f} [{lo*100:.1f}, {hi*100:.1f}]"


def _strip_large_basis_extra(extra: Dict[str, Any], *, keep_full_pca_info: bool) -> Dict[str, Any]:
    """Avoid writing huge arrays into JSON by default."""
    out = dict(extra or {})
    if not keep_full_pca_info and "full_pca_info" in out:
        out["full_pca_info"] = {"_omitted": True}
    return out


def render_loto_heldout_table(results: Dict[str, Any]) -> str:
    """Markdown table across folds: one row per held-out task."""
    folds = results.get("folds", {})
    header = [
        "Held-out",
        "n",
        "Protocol",
        "Baseline",
        "Decode-shared",
        "Prefill-shared",
        "Random",
        "Δ(Decode-Prefill) [CI]",
        "p",
    ]

    rows: List[List[str]] = []
    for holdout, fold in folds.items():
        # In heldout mode, eval_results should contain only this task.
        eval_res = fold.get("eval", {})
        if holdout not in eval_res:
            continue
        r = eval_res[holdout]

        stat = r.get("paired", {}).get("decode_minus_prefill", {})
        md = float(stat.get("mean_diff", float("nan")))
        lo = float(stat.get("ci_low", float("nan")))
        hi = float(stat.get("ci_high", float("nan")))
        p = float(stat.get("p_value", float("nan")))

        rows.append(
            [
                holdout,
                str(r.get("n", "")),
                str(r.get("protocol", "")),
                fmt_acc(r["baseline"]["acc"], r["baseline"]["ci"][0], r["baseline"]["ci"][1]),
                fmt_acc(r["decode_shared"]["acc"], r["decode_shared"]["ci"][0], r["decode_shared"]["ci"][1]),
                fmt_acc(r["prefill_shared"]["acc"], r["prefill_shared"]["ci"][0], r["prefill_shared"]["ci"][1]),
                fmt_acc(r["random"]["acc"], r["random"]["ci"][0], r["random"]["ci"][1]),
                f"{md*100:+.1f} [{lo*100:+.1f}, {hi*100:+.1f}]",
                f"{p:.3g}",
            ]
        )

    # Format markdown
    if not rows:
        return "(no fold results)"

    # compute widths
    cols = list(zip(*([header] + rows)))
    widths = [max(len(str(x)) for x in col) for col in cols]

    def fmt_row(r: List[str]) -> str:
        return "| " + " | ".join(str(x).ljust(w) for x, w in zip(r, widths)) + " |"

    lines = [fmt_row(header), "|-" + "-|-".join("-" * w for w in widths) + "-|" ]
    for r in rows:
        lines.append(fmt_row(r))
    return "\n".join(lines)


def build_fc_warmup_cache(
    *,
    model,
    tok,
    eval_by: Dict[str, List[Example]],
    tasks: List[str],
    use_forced_choice: bool,
    do_generation: bool,
    fc_warmup_tokens: int,
    fc_warmup_decoding: str,
    fc_warmup_temperature: float,
    fc_warmup_top_p: float,
    fc_warmup_top_k: int,
    fc_warmup_ban_eos: bool,
    base_seed: int,
    batch_size: int,
    max_prompt_len: int,
    cache: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Compute warmup tokens only when needed, and cache per task globally."""

    warmup_by_task: Dict[str, np.ndarray] = {}

    if not use_forced_choice or bool(do_generation) or int(fc_warmup_tokens) <= 0:
        return warmup_by_task

    for task in tasks:
        # Only MC/YesNo tasks that actually use forced-choice
        if len(EP.candidate_strings(task)) == 0:
            continue
        if task in cache:
            warmup_by_task[task] = cache[task]
            continue

        prompts = [ex.prompt for ex in eval_by[task]]
        if len(prompts) == 0:
            continue

        seed = EP.stable_int_seed(base_seed, task, "fc_warmup")
        warm_ids = EP.precompute_fc_warmup_tokens(
            model,
            tok,
            prompts,
            warmup_tokens=int(fc_warmup_tokens),
            batch_size=int(batch_size),
            max_prompt_len=int(max_prompt_len),
            decoding=str(fc_warmup_decoding),
            temperature=float(fc_warmup_temperature),
            top_p=float(fc_warmup_top_p),
            top_k=int(fc_warmup_top_k),
            ban_eos=bool(fc_warmup_ban_eos),
            seed=int(seed),
        )
        cache[task] = warm_ids
        warmup_by_task[task] = warm_ids

    return warmup_by_task


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()

    # Model
    ap.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model_dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--trust_remote_code", type=int, default=0, choices=[0, 1])
    ap.add_argument("--device_map", type=str, default="", help="Optional HF device_map, e.g. 'auto' for multi-GPU sharding.")
    ap.add_argument("--max_memory_per_gpu_gb", type=float, default=0.0, help="Per-GPU cap used only when --device_map is set.")
    ap.add_argument("--cpu_offload_gb", type=float, default=0.0, help="Optional CPU max_memory used only when --device_map is set.")

    # Experiment structure
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument(
        "--tasks",
        type=str,
        default="gsm8k,commonsenseqa,strategyqa,aqua,arc_challenge,openbookqa,qasc,logiqa",
    )
    ap.add_argument("--mode", type=str, default="loto", choices=["all", "loto"])
    ap.add_argument("--loto_eval_mode", type=str, default="heldout", choices=["heldout", "all"])
    ap.add_argument(
        "--loto_only",
        type=str,
        default="",
        help="Optional: only run this held-out task (e.g., 'gsm8k'). Empty means run all folds.",
    )

    # Data sizes
    ap.add_argument("--n_subspace", type=int, default=128)
    ap.add_argument("--n_eval", type=int, default=256)

    # Shared space (PCA/sharedness)
    ap.add_argument("--pca_var", type=float, default=0.95)
    ap.add_argument("--min_dim", type=int, default=1)
    ap.add_argument("--max_dim", type=int, default=4096)
    ap.add_argument("--tau", type=float, default=0.001)
    ap.add_argument("--m_shared", type=str, default="all")
    ap.add_argument(
        "--match_state_count",
        type=int,
        default=0,
        choices=[0, 1],
        help="If 1, match decode/prefill state counts before computing subspaces.",
    )
    ap.add_argument(
        "--k_eval",
        type=int,
        default=0,
        help="If >0, use this k for evaluation (capped at min(k_decode,k_prefill)). 0 => use matched min(k_decode,k_prefill).",
    )
    ap.add_argument(
        "--save_full_pca_info",
        type=int,
        default=0,
        choices=[0, 1],
        help="If 1, keep full_pca_info in JSON. Otherwise it is omitted to keep JSON small.",
    )

    # Activation collection
    ap.add_argument("--calib_decode_max_new_tokens", type=int, default=128)
    ap.add_argument("--per_task_max_states", type=int, default=20000)

    # Intervention
    ap.add_argument("--alpha_remove", type=float, default=1.0)
    # NOTE: kept for CLI compatibility with old script (not used in this refactor)
    ap.add_argument(
        "--reasoning_tokens",
        type=int,
        default=128,
        help="(compat) Not used in this refactor; staged removal is not implemented in eval_perf.py.",
    )

    # Generation decoding
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--do_sample", type=int, default=0, choices=[0, 1])

    # Template randomization
    ap.add_argument("--template_randomization", type=int, default=1, choices=[0, 1])
    ap.add_argument("--template_seed", type=int, default=1234)
    ap.add_argument("--shuffle_choices", type=int, default=0, choices=[0, 1])
    ap.add_argument("--add_answer_prefix", type=int, default=0, choices=[0, 1])
    ap.add_argument("--answer_prefix", type=str, default="\nFinal answer:")

    # Forced-choice
    ap.add_argument(
        "--use_forced_choice",
        type=int,
        default=0,
        choices=[0, 1],
        help="If 1, use forced-choice for tasks with discrete candidates (MC/YesNo). gsm8k stays generation.",
    )
    ap.add_argument("--fc_warmup_tokens", type=int, default=0)
    ap.add_argument("--fc_warmup_decoding", type=str, default="greedy", choices=["greedy", "sample"])
    ap.add_argument("--fc_warmup_seed", type=int, default=123)
    ap.add_argument("--fc_warmup_ban_eos", type=int, default=1, choices=[0, 1])
    ap.add_argument("--fc_warmup_temperature", type=float, default=0.7)
    ap.add_argument("--fc_warmup_top_p", type=float, default=0.9)
    ap.add_argument("--fc_warmup_top_k", type=int, default=0)
    ap.add_argument("--fc_prefix_mode", type=str, default="auto", choices=["auto", "always", "never"])
    ap.add_argument("--fc_answer_prefix", type=str, default="\nFinal answer:")
    ap.add_argument(
        "--fc_save_scores",
        type=int,
        default=1,
        choices=[0, 1],
        help="If 1, save per-example candidate score sums (can make JSON very large).",
    )

    # Misc runtime
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_prompt_len", type=int, default=512)

    # Stats
    ap.add_argument("--bootstrap_iters", type=int, default=5000)
    ap.add_argument("--perm_iters", type=int, default=10000)
    ap.add_argument("--ci_alpha", type=float, default=0.05)

    # Seeds
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample_seed", type=int, default=12345)

    # Output
    ap.add_argument("--out_json", type=str, default=os.path.join(THIS_DIR, "energy_balance_loto_refactored.json"))
    ap.add_argument("--out_md", type=str, default=os.path.join(THIS_DIR, "energy_balance_loto_refactored.md"))

    args = ap.parse_args()

    # Seeds
    EP.set_global_seed(int(args.seed))

    # Normalize prefixes (treat '0'/'none' as empty)
    args.answer_prefix = EP.normalize_answer_prefix(args.answer_prefix)
    args.fc_answer_prefix = EP.normalize_answer_prefix(args.fc_answer_prefix)

    # Booleans
    args.trust_remote_code = bool(args.trust_remote_code)
    args.do_sample = bool(args.do_sample)
    args.template_randomization = bool(args.template_randomization)
    args.shuffle_choices = bool(args.shuffle_choices)
    args.add_answer_prefix = bool(args.add_answer_prefix)
    args.use_forced_choice = bool(args.use_forced_choice)
    args.match_state_count = bool(args.match_state_count)
    args.save_full_pca_info = bool(args.save_full_pca_info)
    args.fc_save_scores = bool(args.fc_save_scores)

    tasks = [t.strip() for t in str(args.tasks).split(",") if t.strip()]
    if len(tasks) < 2:
        raise RuntimeError("Need at least 2 tasks in --tasks.")

    # Generation modes
    gen_modes = ["greedy"] + (["sample"] if args.do_sample else [])

    # If use_forced_choice=1 => do_generation=False (MC tasks use forced-choice)
    do_generation = (not bool(args.use_forced_choice))

    print("\n" + "=" * 80)
    print("[Env] Refactored LOTO runner (delegating to eval_perf.py + benchmark_dataloaders.py)")
    print(f"[Env] DEVICE={args.device}  MODEL={args.model}  dtype={args.model_dtype} trust_remote_code={args.trust_remote_code}")
    print(f"[Env] layer={args.layer}  tasks={tasks}")
    print(f"[Env] mode={args.mode} loto_eval_mode={args.loto_eval_mode} loto_only={args.loto_only!r}")
    print(f"[Env] data: n_subspace={args.n_subspace} n_eval={args.n_eval}")
    print(f"[Env] shared-space: pca_var={args.pca_var} tau={args.tau} m_shared={args.m_shared} match_state_count={args.match_state_count} k_eval={args.k_eval}")
    print(f"[Env] eval: alpha_remove={args.alpha_remove} do_generation={do_generation} gen_modes={gen_modes}")
    print(f"[Env] forced_choice={args.use_forced_choice} warmup_tokens={args.fc_warmup_tokens} prefix_mode={args.fc_prefix_mode} fc_answer_prefix={args.fc_answer_prefix!r}")
    if args.reasoning_tokens:
        print("[Note] --reasoning_tokens is kept for compatibility but is NOT used in this refactor.")
    print("=" * 80 + "\n")

    # Load model
    model, tok = EP.load_model_and_tokenizer(
        model_name=str(args.model),
        device=str(args.device),
        dtype=str(args.model_dtype),
        trust_remote_code=bool(args.trust_remote_code),
        device_map=(str(args.device_map).strip() or None),
        max_memory_per_gpu_gb=float(args.max_memory_per_gpu_gb),
        cpu_offload_gb=float(args.cpu_offload_gb),
    )

    # Load data (benchmark_dataloaders)
    sub_by, eval_by, meta_by = load_selected_tasks(
        tasks=tasks,
        n_subspace=int(args.n_subspace),
        n_eval=int(args.n_eval),
        seed=int(args.seed),
        template_seed=int(args.template_seed),
        template_randomization=bool(args.template_randomization),
        shuffle_choices=bool(args.shuffle_choices),
        add_answer_prefix=bool(args.add_answer_prefix),
        answer_prefix=str(args.answer_prefix),
    )

    print("=" * 80)
    print(f"[Data] Loaded tasks: {list(sub_by.keys())}")
    print(f"[Data] Meta: {json.dumps(meta_by, ensure_ascii=False, indent=2)}")
    print("=" * 80 + "\n")

    # Global forced-choice warmup cache (reusable across folds)
    global_warmup_cache: Dict[str, np.ndarray] = {}

    # Results
    results: Dict[str, Any] = {
        "config": {
            "model": str(args.model),
            "device": str(args.device),
            "model_dtype": str(args.model_dtype),
            "trust_remote_code": bool(args.trust_remote_code),
            "device_map": str(args.device_map),
            "max_memory_per_gpu_gb": float(args.max_memory_per_gpu_gb),
            "cpu_offload_gb": float(args.cpu_offload_gb),
            "layer": int(args.layer),
            "tasks": tasks,
            "mode": str(args.mode),
            "loto_eval_mode": str(args.loto_eval_mode),
            "loto_only": str(args.loto_only),
            "n_subspace": int(args.n_subspace),
            "n_eval": int(args.n_eval),
            "pca_var": float(args.pca_var),
            "min_dim": int(args.min_dim),
            "max_dim": int(args.max_dim),
            "tau": float(args.tau),
            "m_shared": str(args.m_shared),
            "match_state_count": bool(args.match_state_count),
            "k_eval": int(args.k_eval),
            "calib_decode_max_new_tokens": int(args.calib_decode_max_new_tokens),
            "per_task_max_states": int(args.per_task_max_states),
            "alpha_remove": float(args.alpha_remove),
            "reasoning_tokens": int(args.reasoning_tokens),
            "generation": {
                "max_new_tokens": int(args.max_new_tokens),
                "gen_modes": gen_modes,
                "temperature": float(args.temperature),
                "top_p": float(args.top_p),
                "top_k": int(args.top_k),
                "sample_seed": int(args.sample_seed),
            },
            "forced_choice": {
                "use_forced_choice": bool(args.use_forced_choice),
                "fc_warmup_tokens": int(args.fc_warmup_tokens),
                "fc_warmup_decoding": str(args.fc_warmup_decoding),
                "fc_warmup_seed": int(args.fc_warmup_seed),
                "fc_warmup_ban_eos": bool(args.fc_warmup_ban_eos),
                "fc_warmup_temperature": float(args.fc_warmup_temperature),
                "fc_warmup_top_p": float(args.fc_warmup_top_p),
                "fc_warmup_top_k": int(args.fc_warmup_top_k),
                "fc_prefix_mode": str(args.fc_prefix_mode),
                "fc_answer_prefix": str(args.fc_answer_prefix),
                "fc_save_scores": bool(args.fc_save_scores),
            },
            "runtime": {
                "batch_size": int(args.batch_size),
                "max_prompt_len": int(args.max_prompt_len),
            },
            "stats": {
                "bootstrap_iters": int(args.bootstrap_iters),
                "perm_iters": int(args.perm_iters),
                "ci_alpha": float(args.ci_alpha),
            },
            "seed": int(args.seed),
            "dataset_meta": meta_by,
        }
    }

    def run_one_fold(*, fold_name: str, train_tasks: List[str], eval_tasks: List[str]) -> Dict[str, Any]:
        print("\n" + "=" * 90)
        print(f"[Fold] {fold_name}")
        print(f"[Fold] train_tasks={train_tasks}")
        print(f"[Fold] eval_tasks ={eval_tasks}")
        print("=" * 90)

        sub_train = {t: sub_by[t] for t in train_tasks}

        # Compute shared bases (decode + prefill) from train tasks only
        bases = EP.compute_decode_prefill_shared_bases(
            model,
            tok,
            sub_train,
            layer_idx=int(args.layer),
            batch_size=int(args.batch_size),
            max_prompt_len=int(args.max_prompt_len),
            calib_decode_max_new_tokens=int(args.calib_decode_max_new_tokens),
            per_task_max_states=int(args.per_task_max_states),
            pca_var=float(args.pca_var),
            min_dim=int(args.min_dim),
            max_dim=int(args.max_dim),
            tau=float(args.tau),
            m_shared=str(args.m_shared),
            seed=EP.stable_int_seed(int(args.seed), fold_name, "bases"),
            match_state_count=bool(args.match_state_count),
            k_eval=int(args.k_eval),
        )

        # Warmup tokens (forced-choice only) — cached per task globally
        warmup_by_task = build_fc_warmup_cache(
            model=model,
            tok=tok,
            eval_by=eval_by,
            tasks=eval_tasks,
            use_forced_choice=bool(args.use_forced_choice),
            do_generation=bool(do_generation),
            fc_warmup_tokens=int(args.fc_warmup_tokens),
            fc_warmup_decoding=str(args.fc_warmup_decoding),
            fc_warmup_temperature=float(args.fc_warmup_temperature),
            fc_warmup_top_p=float(args.fc_warmup_top_p),
            fc_warmup_top_k=int(args.fc_warmup_top_k),
            fc_warmup_ban_eos=bool(args.fc_warmup_ban_eos),
            base_seed=EP.stable_int_seed(int(args.seed), int(args.fc_warmup_seed)),
            batch_size=int(args.batch_size),
            max_prompt_len=int(args.max_prompt_len),
            cache=global_warmup_cache,
        )

        # Evaluate
        eval_results = EP.evaluate_tasks_once(
            model,
            tok,
            eval_by,
            tasks=eval_tasks,
            layer_idx=int(args.layer),
            bases=bases,
            alpha_remove=float(args.alpha_remove),
            batch_size=int(args.batch_size),
            max_prompt_len=int(args.max_prompt_len),
            do_generation=bool(do_generation),
            fc_warmup_by_task=warmup_by_task,
            fc_answer_prefix=str(args.fc_answer_prefix),
            fc_prefix_mode=str(args.fc_prefix_mode),
            fc_save_scores=bool(args.fc_save_scores),
            gen_modes=gen_modes,
            gen_max_new_tokens=int(args.max_new_tokens),
            gen_temperature=float(args.temperature),
            gen_top_p=float(args.top_p),
            gen_top_k=int(args.top_k),
            gen_seed=int(args.sample_seed),
            bootstrap_iters=int(args.bootstrap_iters),
            perm_iters=int(args.perm_iters),
            ci_alpha=float(args.ci_alpha),
            seed=int(args.seed),
        )

        # Build fold summary table (markdown) — note: this is per-fold; can be large if eval_tasks is large.
        try:
            table_md = EP.build_summary_table(eval_results, k=int(bases.k_eval))
        except Exception:
            table_md = "(failed to render summary table)"

        fold_out: Dict[str, Any] = {
            "fold_name": fold_name,
            "train_tasks": list(train_tasks),
            "eval_tasks": list(eval_tasks),
            "bases": {
                "k_decode": int(bases.k_decode),
                "k_prefill": int(bases.k_prefill),
                "k_eval": int(bases.k_eval),
                "similarity_full": bases.similarity_full,
                "similarity_k": bases.similarity_k,
                "energy": bases.energy,
                "extra_decode": _strip_large_basis_extra(bases.extra_decode, keep_full_pca_info=bool(args.save_full_pca_info)),
                "extra_prefill": _strip_large_basis_extra(bases.extra_prefill, keep_full_pca_info=bool(args.save_full_pca_info)),
            },
            "eval": eval_results,
            "summary_table_md": table_md,
        }
        return fold_out

    if str(args.mode).lower() == "all":
        fold = run_one_fold(fold_name="all_tasks", train_tasks=tasks, eval_tasks=tasks)
        results["all_tasks"] = fold

    else:
        folds: Dict[str, Any] = {}
        for holdout in tasks:
            if args.loto_only and holdout != args.loto_only:
                continue

            train_tasks = [t for t in tasks if t != holdout]
            eval_tasks = [holdout] if str(args.loto_eval_mode).lower() == "heldout" else list(tasks)
            fold_name = f"loto_holdout={holdout}"

            fold = run_one_fold(fold_name=fold_name, train_tasks=train_tasks, eval_tasks=eval_tasks)
            folds[holdout] = fold

            # Mild hygiene between folds
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        results["folds"] = folds

    # ---------------------------------------------------------------------
    # Save JSON
    # ---------------------------------------------------------------------
    with open(str(args.out_json), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=json_default)

    # ---------------------------------------------------------------------
    # Save Markdown summary
    # ---------------------------------------------------------------------
    md_lines: List[str] = []
    md_lines.append("# Energy-balance + LOTO Summary (refactored)\n")
    md_lines.append(f"- Model: `{args.model}` dtype={args.model_dtype} device={args.device}\n")
    md_lines.append(f"- Tasks: {tasks}\n")
    md_lines.append(f"- Mode: {args.mode} (loto_eval_mode={args.loto_eval_mode})\n")
    md_lines.append(f"- Template randomization: {bool(args.template_randomization)} (seed={args.template_seed}), shuffle_choices={bool(args.shuffle_choices)}\n")
    md_lines.append(f"- Sharedness: pca_var={args.pca_var}, tau={args.tau}, m_shared={args.m_shared}, k_eval={args.k_eval or 'auto'}\n")
    md_lines.append(f"- Calibration: calib_decode_max_new_tokens={args.calib_decode_max_new_tokens}, per_task_max_states={args.per_task_max_states}\n")
    md_lines.append(f"- Evaluation: forced_choice={bool(args.use_forced_choice)} warmup_tokens={args.fc_warmup_tokens}\n")
    md_lines.append("")

    if str(args.mode).lower() == "loto" and str(args.loto_eval_mode).lower() == "heldout" and "folds" in results:
        md_lines.append("## LOTO held-out performance\n")
        md_lines.append(render_loto_heldout_table(results))
        md_lines.append("")

        # Optionally also include per-fold basis diagnostics summary
        md_lines.append("## Basis diagnostics (per fold)\n")
        for holdout, fold in results.get("folds", {}).items():
            b = fold.get("bases", {})
            simk = b.get("similarity_k", {})
            md_lines.append(f"### Holdout: {holdout}\n")
            md_lines.append(
                "- k_decode={k_decode}, k_prefill={k_prefill}, k_eval={k_eval}\n".format(
                    k_decode=b.get("k_decode"), k_prefill=b.get("k_prefill"), k_eval=b.get("k_eval")
                )
            )
            md_lines.append(
                "- Similarity(k): max_cos={max_cos:.3f}, mean_cos={mean_cos:.3f}, min_cos={min_cos:.3f}\n".format(
                    max_cos=float(simk.get("max_cos", float("nan"))),
                    mean_cos=float(simk.get("mean_cos", float("nan"))),
                    min_cos=float(simk.get("min_cos", float("nan"))),
                )
            )
            md_lines.append("")
    else:
        # all-tasks mode: include the full summary table once
        if "all_tasks" in results:
            md_lines.append("## All-tasks performance\n")
            md_lines.append(results["all_tasks"].get("summary_table_md", "(no table)"))
            md_lines.append("")

    with open(str(args.out_md), "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print("\n" + "=" * 80)
    print("[Done]")
    print(f"[Done] JSON: {args.out_json}")
    print(f"[Done] MD  : {args.out_md}")
    print("=" * 80)


if __name__ == "__main__":
    main()
