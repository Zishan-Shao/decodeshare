# -*- coding: utf-8 -*-
"""
h3_killer_counterfactual_grid_reasoning_v2.py (refactored, forced-choice only)

This refactor removes ALL generation-evaluation code and delegates to attached modules:
  - Data loading                 -> benchmark_dataloaders.py (via eval_perf.load_selected_tasks)
  - Shared space computation     -> eval_perf.compute_decode_prefill_shared_bases()
  - Forced-choice evaluation     -> eval_perf.forced_choice_logprob_eval()

Evaluation protocol (forced-choice only):
  prompt_core (with any trailing answer_prefix stripped)
    -> teacher-forced warmup tokens (fixed ids from warmup_phrase, optional)
    -> teacher-force answer_prefix tokens (prefix_mode=always)
    -> score candidates immediately via logprob

This matches the “score at answer slot” requirement.

Expected files next to this script (or on PYTHONPATH):
  - eval_perf.py                    (attached)
  - benchmark_dataloaders.py        (attached)

Example:
CUDA_VISIBLE_DEVICES=3 python h3_killer_counterfactual_grid_reasoning.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --device cuda --model_dtype fp32 \
  --tasks gsm8k,commonsenseqa,strategyqa,piqa,arc_challenge,openbookqa,qasc,logiqa,boolq \
  --layer 10 --n_subspace 128 --n_eval 256 \
  --calib_decode_max_new_tokens 512 --per_task_max_states 20000 \
  --answer_prefix $'\\nFinal answer:' \
  --warmup_tokens 0 \
  --template_randomization 1 --shuffle_choices 1 \
  --out_json h3_grid_reasoning.json
  
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
# Import attached utilities
# -----------------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.join(THIS_DIR, ".."))  # project root (for joint_subspace_large.*)

try:
    from decodeshare import eval_perf as EP
except Exception as e:  # pragma: no cover
    raise RuntimeError(f"Failed to import decodeshare.eval_perf: {e}") from e

Example = EP.Example
load_selected_tasks = EP.load_selected_tasks


# -----------------------------------------------------------------------------
# Patch: eval_perf.candidate_strings has a known piqa mismatch (should be A/B)
# -----------------------------------------------------------------------------
_ORIG_CANDIDATE_STRINGS = EP.candidate_strings

def _candidate_strings_patched(task: str) -> List[str]:
    t = (task or "").strip().lower()
    if t == "piqa":
        return ["A", "B"]
    return _ORIG_CANDIDATE_STRINGS(task)

EP.candidate_strings = _candidate_strings_patched  # monkey-patch locally


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def json_default(o: Any) -> Any:
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


def strip_last_answer_prefix(prompt: str, answer_prefix: str) -> Tuple[str, bool]:
    """
    Remove the last occurrence of answer_prefix and everything after it.
    Returns (prompt_core, found_flag).
    """
    ap = EP.normalize_answer_prefix(answer_prefix)
    if not ap:
        return prompt, False
    idx = prompt.rfind(ap)
    if idx == -1:
        return prompt, False
    return prompt[:idx], True


def principal_angles_deg(Qa: np.ndarray, Qb: np.ndarray) -> Dict[str, float]:
    """
    Principal angles between orthonormal column spaces Qa(d×k) and Qb(d×k).
    Returns mean/p50/p95 in degrees.
    """
    if Qa.size == 0 or Qb.size == 0:
        return {"mean": float("nan"), "p50": float("nan"), "p95": float("nan")}
    M = Qa.T @ Qb
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    theta = np.degrees(np.arccos(s))
    return {
        "mean": float(np.mean(theta)),
        "p50": float(np.percentile(theta, 50)),
        "p95": float(np.percentile(theta, 95)),
    }


def make_fixed_warmup_ids(tok, warmup_phrase: str, warmup_tokens: int) -> List[int]:
    """
    Teacher-forced fixed warmup tokens, built by repeating tokenized warmup_phrase.
    No model generation is used.
    """
    W = int(warmup_tokens)
    if W <= 0:
        return []
    base_ids = tok(warmup_phrase, add_special_tokens=False).input_ids
    if not base_ids:
        base_ids = tok(" ", add_special_tokens=False).input_ids
    if not base_ids:
        return []
    rep = (W + len(base_ids) - 1) // len(base_ids)
    return (base_ids * rep)[:W]


def warmup_matrix(N: int, warmup_ids: List[int]) -> Optional[np.ndarray]:
    """
    Convert warmup_ids (shared across examples) into [N,W] int64 matrix.
    """
    if N <= 0 or not warmup_ids:
        return None
    w = np.array(warmup_ids, dtype=np.int64)[None, :]
    return np.repeat(w, repeats=N, axis=0)


def tokenization_sanity(tok, task_names: List[str]) -> Dict[str, Any]:
    """
    Print / record candidate tokenization lengths for a few tasks.
    Uses eval_perf.cand_token_ids (which tries leading-space first).
    """
    out: Dict[str, Any] = {}
    for t in task_names:
        cands = EP.candidate_strings(t)
        if not cands:
            continue
        lens = []
        for c in cands:
            ids = EP.cand_token_ids(tok, c)
            lens.append({"cand": c, "n_tokens": int(len(ids)), "ids_head": ids[:6]})
        out[t] = lens
    return out


def run_fc_condition(
    *,
    model,
    tok,
    task: str,
    examples: List[Example],
    basis_np: Optional[np.ndarray],
    alpha_remove: float,
    layer_idx: int,
    batch_size: int,
    max_prompt_len: int,
    answer_prefix: str,
    prefix_mode: str,
    warmup_ids: List[int],
    bootstrap_iters: int,
    ci_alpha: float,
    seed: int,
    save_scores: bool,
) -> Dict[str, Any]:
    """
    Forced-choice logprob eval for one condition; returns acc + CI + raw eval_perf outputs.
    """
    # Strip answer_prefix from prompts to ensure we score at the answer slot,
    # then force answer_prefix immediately before scoring (prefix_mode=always recommended).
    stripped: List[Example] = []
    for ex in examples:
        core, _found = strip_last_answer_prefix(ex.prompt, answer_prefix)
        stripped.append(Example(dataset=ex.dataset, ex_id=ex.ex_id, prompt=core, gold=ex.gold))

    warm_ids_mat = warmup_matrix(len(stripped), warmup_ids)

    out = EP.forced_choice_logprob_eval(
        model,
        tok,
        stripped,
        task,
        layer_indices=[int(layer_idx)],
        basis_np=basis_np,
        alpha=float(alpha_remove),
        batch_size=int(batch_size),
        max_prompt_len=int(max_prompt_len),
        warmup_token_ids=warm_ids_mat,
        answer_prefix=str(answer_prefix),
        prefix_mode=str(prefix_mode),
        save_scores=bool(save_scores),
    )

    correct = np.array(out["correct"], dtype=np.float32)
    acc, lo, hi = EP.bootstrap_ci_mean(correct, iters=int(bootstrap_iters), alpha=float(ci_alpha), seed=int(seed))

    return {
        "acc": float(acc),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "correct": out["correct"],
        "hook_stats": out.get("hook_stats", {}),
        "metrics_summary": out.get("metrics_summary", {}),
        "preds": out.get("preds", []),
        "golds": out.get("golds", []),
        "cands": out.get("cands", []),
        # optionally huge:
        "scores_sum": out.get("scores_sum", None),
    }



def main() -> None:
    ap = argparse.ArgumentParser()

    # Model
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model_dtype", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--trust_remote_code", type=int, default=0, choices=[0, 1])

    # Data
    ap.add_argument("--tasks", type=str, default="gsm8k,commonsenseqa,strategyqa,piqa,arc_challenge,openbookqa,qasc,logiqa,boolq")
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--n_subspace", type=int, default=128)
    ap.add_argument("--n_eval", type=int, default=2048)
    ap.add_argument("--max_prompt_len", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=8)

    # Shared-basis calibration
    ap.add_argument("--calib_decode_max_new_tokens", type=int, default=512)
    ap.add_argument("--per_task_max_states", type=int, default=20000)

    # Sharedness / PCA
    ap.add_argument("--pca_var", type=float, default=0.95)
    ap.add_argument("--min_dim", type=int, default=16)
    ap.add_argument("--max_dim", type=int, default=4096)
    ap.add_argument("--tau", type=float, default=1e-3)
    ap.add_argument("--m_shared", type=str, default="all")
    ap.add_argument("--match_state_count", type=int, default=0, choices=[0, 1])
    ap.add_argument("--k_eval", type=int, default=0, help="If >0, force matched k_eval; else auto=min(k_decode,k_prefill).")

    # Forced-choice protocol knobs
    ap.add_argument("--answer_prefix", type=str, default="\nFinal answer:")
    ap.add_argument("--prefix_mode", type=str, default="always", choices=["always", "auto", "never"])
    ap.add_argument("--warmup_tokens", type=int, default=128)
    ap.add_argument("--warmup_phrase", type=str, default="Let's think step by step.\n")

    # Template randomization
    ap.add_argument("--template_randomization", type=int, default=1, choices=[0, 1])
    ap.add_argument("--shuffle_choices", type=int, default=1, choices=[0, 1])
    ap.add_argument("--template_seed", type=int, default=999)

    # Intervention control
    ap.add_argument("--alpha_remove", type=float, default=1.0)
    ap.add_argument("--fc_save_scores", type=int, default=1, choices=[0, 1])

    # Stats
    ap.add_argument("--bootstrap_iters", type=int, default=2000)
    ap.add_argument("--perm_iters", type=int, default=10000)
    ap.add_argument("--ci_alpha", type=float, default=0.05)

    # Seed / output
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_json", type=str, default="")

    args = ap.parse_args()
    args.trust_remote_code = bool(args.trust_remote_code)
    args.match_state_count = bool(args.match_state_count)
    args.template_randomization = bool(args.template_randomization)
    args.shuffle_choices = bool(args.shuffle_choices)
    args.fc_save_scores = bool(args.fc_save_scores)

    EP.set_global_seed(int(args.seed))

    tasks = [t.strip() for t in str(args.tasks).split(",") if t.strip()]

    # Normalize answer prefix
    args.answer_prefix = EP.normalize_answer_prefix(args.answer_prefix)

    # 1) Load model/tokenizer
    model, tok = EP.load_model_and_tokenizer(
        model_name=str(args.model),
        device=str(args.device),
        dtype=str(args.model_dtype),
        trust_remote_code=bool(args.trust_remote_code),
    )

    # 2) Load data
    # IMPORTANT: do NOT append answer_prefix here; we handle answer slot in forced-choice eval.
    sub_by, eval_by, meta = load_selected_tasks(
        tasks=tasks,
        n_subspace=int(args.n_subspace),
        n_eval=int(args.n_eval),
        seed=int(args.seed),
        template_seed=int(args.seed) + int(args.template_seed),
        template_randomization=bool(args.template_randomization),
        shuffle_choices=bool(args.shuffle_choices),
        add_answer_prefix=False,
        answer_prefix=str(args.answer_prefix),
    )

    # 3) Fixed warmup ids
    warmup_ids = make_fixed_warmup_ids(tok, str(args.warmup_phrase), int(args.warmup_tokens))
    print(f"[Warmup] teacher_forced_fixed W={len(warmup_ids)} phrase={args.warmup_phrase.strip()!r}")

    # 4) Compute decode vs prefill shared bases (and matched k + random control)
    bases = EP.compute_decode_prefill_shared_bases(
        model,
        tok,
        sub_by,
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
        seed=int(EP.stable_int_seed(int(args.seed), "bases", int(args.layer))),
        match_state_count=bool(args.match_state_count),
        k_eval=int(args.k_eval),
    )

    Q_dec = bases.Q_decode_k
    Q_pre = bases.Q_prefill_k
    Q_ctl = bases.Q_rand_k
    k = int(bases.k_eval)

    ang = principal_angles_deg(Q_dec, Q_pre)
    print("\n" + "=" * 90)
    print("[H3 Bases]")
    print(f"  layer={args.layer}  k_decode={bases.k_decode}  k_prefill={bases.k_prefill}  k_eval(matched)={k}")
    print(f"  similarity_k: {bases.similarity_k}")
    print(f"  principal_angles(deg): mean={ang['mean']:.2f}  p50={ang['p50']:.2f}  p95={ang['p95']:.2f}")
    print("=" * 90)

    # 5) Candidate tokenization sanity
    sanity_tasks = ["commonsenseqa", "arc_challenge", "piqa", "boolq", "strategyqa"]
    tok_sanity = tokenization_sanity(tok, sanity_tasks)
    for t, info in tok_sanity.items():
        pretty = [(x["cand"], x["n_tokens"]) for x in info]
        print(f"[CandTok] {t}: {pretty}")

    # 6) Evaluate H3 grid (forced-choice only)
    results: Dict[str, Any] = {
        "model": str(args.model),
        "device": str(args.device),
        "dtype": str(args.model_dtype),
        "layer": int(args.layer),
        "seed": int(args.seed),
        "tasks": tasks,
        "data_meta": meta,
        "warmup": {
            "warmup_tokens": int(args.warmup_tokens),
            "warmup_phrase": str(args.warmup_phrase),
            "warmup_ids_len": int(len(warmup_ids)),
        },
        "protocol": {
            "forced_choice_only": True,
            "answer_prefix": str(args.answer_prefix),
            "prefix_mode": str(args.prefix_mode),
        },
        "bases": {
            "k_decode": int(bases.k_decode),
            "k_prefill": int(bases.k_prefill),
            "k_eval": int(k),
            "similarity_full": bases.similarity_full,
            "similarity_k": bases.similarity_k,
            "energy": bases.energy,
            "angles_deg": ang,
        },
        "tokenization_sanity": tok_sanity,
        "by_task": {},
    }

    for task in tasks:
        cands = EP.candidate_strings(task)
        if not cands:
            print(f"[Skip] {task}: no forced-choice candidates")
            continue

        exs = eval_by.get(task, [])
        if not exs:
            print(f"[Skip] {task}: no eval examples")
            continue

        print("\n" + "=" * 90)
        print(f"[H3 Grid | ForcedChoice] {task}  n={len(exs)}  W={len(warmup_ids)}  prefix_mode={args.prefix_mode}")
        print("=" * 90)

        # baseline
        r_base = run_fc_condition(
            model=model, tok=tok,
            task=task, examples=exs,
            basis_np=None,
            alpha_remove=float(args.alpha_remove),
            layer_idx=int(args.layer),
            batch_size=int(args.batch_size),
            max_prompt_len=int(args.max_prompt_len),
            answer_prefix=str(args.answer_prefix),
            prefix_mode=str(args.prefix_mode),
            warmup_ids=warmup_ids,
            bootstrap_iters=int(args.bootstrap_iters),
            ci_alpha=float(args.ci_alpha),
            seed=int(EP.stable_int_seed(int(args.seed), task, "baseline", "ci")),
            save_scores=bool(args.fc_save_scores),
        )

        # decode-est / decode-int
        r_dec = run_fc_condition(
            model=model, tok=tok,
            task=task, examples=exs,
            basis_np=Q_dec,
            alpha_remove=float(args.alpha_remove),
            layer_idx=int(args.layer),
            batch_size=int(args.batch_size),
            max_prompt_len=int(args.max_prompt_len),
            answer_prefix=str(args.answer_prefix),
            prefix_mode=str(args.prefix_mode),
            warmup_ids=warmup_ids,
            bootstrap_iters=int(args.bootstrap_iters),
            ci_alpha=float(args.ci_alpha),
            seed=int(EP.stable_int_seed(int(args.seed), task, "decode", "ci")),
            save_scores=bool(args.fc_save_scores),
        )

        # prefill-est / decode-int
        r_pre = run_fc_condition(
            model=model, tok=tok,
            task=task, examples=exs,
            basis_np=Q_pre,
            alpha_remove=float(args.alpha_remove),
            layer_idx=int(args.layer),
            batch_size=int(args.batch_size),
            max_prompt_len=int(args.max_prompt_len),
            answer_prefix=str(args.answer_prefix),
            prefix_mode=str(args.prefix_mode),
            warmup_ids=warmup_ids,
            bootstrap_iters=int(args.bootstrap_iters),
            ci_alpha=float(args.ci_alpha),
            seed=int(EP.stable_int_seed(int(args.seed), task, "prefill", "ci")),
            save_scores=bool(args.fc_save_scores),
        )

        # random control / decode-int
        r_ctl = run_fc_condition(
            model=model, tok=tok,
            task=task, examples=exs,
            basis_np=Q_ctl,
            alpha_remove=float(args.alpha_remove),
            layer_idx=int(args.layer),
            batch_size=int(args.batch_size),
            max_prompt_len=int(args.max_prompt_len),
            answer_prefix=str(args.answer_prefix),
            prefix_mode=str(args.prefix_mode),
            warmup_ids=warmup_ids,
            bootstrap_iters=int(args.bootstrap_iters),
            ci_alpha=float(args.ci_alpha),
            seed=int(EP.stable_int_seed(int(args.seed), task, "control", "ci")),
            save_scores=bool(args.fc_save_scores),
        )

        # Paired stats (decode vs prefill is the key H3 comparison)
        base_arr = np.array(r_base["correct"], dtype=np.float32)
        dec_arr = np.array(r_dec["correct"], dtype=np.float32)
        pre_arr = np.array(r_pre["correct"], dtype=np.float32)
        ctl_arr = np.array(r_ctl["correct"], dtype=np.float32)

        seed_p = int(EP.stable_int_seed(int(args.seed), task, "paired"))
        paired = {
            "decode_vs_baseline": EP.summarize_paired(
                base_arr, dec_arr,
                label=f"{task}:decode_vs_baseline",
                bootstrap_iters=int(args.bootstrap_iters),
                perm_iters=int(args.perm_iters),
                alpha=float(args.ci_alpha),
                seed=seed_p + 1,
            ),
            "prefill_vs_baseline": EP.summarize_paired(
                base_arr, pre_arr,
                label=f"{task}:prefill_vs_baseline",
                bootstrap_iters=int(args.bootstrap_iters),
                perm_iters=int(args.perm_iters),
                alpha=float(args.ci_alpha),
                seed=seed_p + 2,
            ),
            "rand_vs_baseline": EP.summarize_paired(
                base_arr, ctl_arr,
                label=f"{task}:rand_vs_baseline",
                bootstrap_iters=int(args.bootstrap_iters),
                perm_iters=int(args.perm_iters),
                alpha=float(args.ci_alpha),
                seed=seed_p + 3,
            ),
            "decode_minus_prefill": EP.summarize_paired(
                pre_arr, dec_arr,
                label=f"{task}:decode_minus_prefill",
                bootstrap_iters=int(args.bootstrap_iters),
                perm_iters=int(args.perm_iters),
                alpha=float(args.ci_alpha),
                seed=seed_p + 4,
            ),
        }

        print(f"  baseline(decode)            : {fmt_acc(r_base['acc'], r_base['ci_low'], r_base['ci_high'])}")
        print(f"  decode-est / decode-int     : {fmt_acc(r_dec['acc'], r_dec['ci_low'], r_dec['ci_high'])}")
        print(f"  prefill-est / decode-int    : {fmt_acc(r_pre['acc'], r_pre['ci_low'], r_pre['ci_high'])}")
        print(f"  random-control / decode-int : {fmt_acc(r_ctl['acc'], r_ctl['ci_low'], r_ctl['ci_high'])}")

        dmp = paired["decode_minus_prefill"]
        print(
            f"  [Stats] decode - prefill: Δ={dmp['mean_diff']:+.3f} "
            f"CI[{dmp['ci_low']:+.3f}, {dmp['ci_high']:+.3f}] p={dmp['p_value']:.3g}"
        )

        results["by_task"][task] = {
            "n": int(len(exs)),
            "candidates": cands,
            "baseline": r_base,
            "decode": r_dec,
            "prefill": r_pre,
            "control": r_ctl,
            "paired": paired,
        }

    # 7) Save JSON
    if not args.out_json:
        safe_model = str(args.model).replace("/", "_")
        args.out_json = f"h3_grid_v2_forcedchoice_{safe_model}_layer{args.layer}_k{k}_W{len(warmup_ids)}_seed{args.seed}.json"

    with open(str(args.out_json), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=json_default)

    print("\n" + "=" * 80)
    print("[Done]")
    print(f"[Done] wrote {args.out_json}")
    print("=" * 80)


if __name__ == "__main__":
    main()
