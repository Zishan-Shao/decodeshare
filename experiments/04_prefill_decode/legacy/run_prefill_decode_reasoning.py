# -*- coding: utf-8 -*-
"""
prefill_vs_decode_alignment_experiment_reasoning.py

This one is with golden xx and entropy stuff, so it is better than pure forced choice

Refactor goals (per your request):
  - Remove ALL generation code paths. Default evaluation is forced-choice only.
  - Use attached scripts for:
      * dataloading: benchmark_dataloaders.py
      * forced-choice evaluation + warmup: eval_perf.py
      * shared-space computation (decode vs prefill): eval_perf.py

Behavior:
  1) Load tasks via benchmark_dataloaders.load_selected_tasks()
  2) Compute shared bases (decode-est / prefill-est) via eval_perf.compute_decode_prefill_shared_bases()
  3) Forced-choice evaluation only:
        baseline / decode_full / prefill_full / decode_k / prefill_k / rand_k
     with bootstrap CIs + paired sign-flip permutation test on (decode_k - prefill_k)

Important default:
  - We STRIP the answer prefix from evaluation prompts (last occurrence at end) to ensure
    warmup happens BEFORE answer_prefix, and scoring happens AT answer slot.
  - Then we let eval_perf.forced_choice_logprob_eval() add answer_prefix back according to fc_prefix_mode.

Notes:
  - Tasks without discrete candidates (e.g., gsm8k) will be skipped in evaluation,
    but can still be included in basis estimation.
"""

from __future__ import annotations

import os
import sys
import json
import argparse
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch


# -----------------------------------------------------------------------------
# Path setup: assume eval_perf.py and benchmark_dataloaders.py are in the same dir
# -----------------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.append(THIS_DIR)
# If your project modules live one level up, keep it:
PARENT_DIR = os.path.join(THIS_DIR, "..")
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)


# -----------------------------------------------------------------------------
# Imports: attached scripts
# -----------------------------------------------------------------------------
try:
    from decodeshare import eval_perf as EP
except Exception as e:  # pragma: no cover
    raise RuntimeError("Failed to import decodeshare.eval_perf.") from e

try:
    from decodeshare.benchmark_dataloaders import Example, load_selected_tasks
except Exception as e:  # pragma: no cover
    raise RuntimeError("Failed to import decodeshare.benchmark_dataloaders.") from e


# -----------------------------------------------------------------------------
# Monkeypatch: fix PIQA candidates (A/B, not ABCD)
#   (eval_perf.candidate_strings currently lists piqa in ABCD in some versions)
# -----------------------------------------------------------------------------
def _patched_candidate_strings(task: str) -> List[str]:
    t = (task or "").strip().lower()
    if t == "piqa":
        return ["A", "B"]
    return EP.candidate_strings(task)

EP.candidate_strings = _patched_candidate_strings  # type: ignore


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _norm_prefix_arg(x: Any) -> str:
    """Treat '0'/'none'/'null' as empty prefix (helps with CLI habits)."""
    if x is None:
        return ""
    s = str(x)
    if s.strip().lower() in {"0", "none", "null", "false"}:
        return ""
    return s


def strip_last_answer_prefix(prompt: str, answer_prefix: str) -> Tuple[str, bool]:
    """
    Remove the *last* occurrence of answer_prefix ONLY if it is at the end (ignoring trailing whitespace).
    Returns (prompt_without_prefix, found_flag).

    This avoids:
      - warmup happening after answer_prefix
      - double-prefix when prefix_mode auto/always
    """
    if not answer_prefix:
        return prompt, False

    p0 = prompt or ""
    ap = answer_prefix

    # First, handle "endswith ignoring trailing whitespace"
    pr = p0.rstrip()
    apr = ap.rstrip()
    if apr and pr.endswith(apr):
        idx = pr.rfind(apr)
        if idx >= 0:
            return pr[:idx].rstrip(), True

    # Otherwise try raw rfind, but only strip if tail is essentially prefix + whitespace
    idx = p0.rfind(ap)
    if idx == -1:
        return prompt, False
    tail = p0[idx:]
    if tail.strip() == ap.strip():
        return p0[:idx].rstrip(), True

    return prompt, False


def strip_eval_prompts(
    eval_by: Dict[str, List[Example]],
    *,
    answer_prefix: str,
    enabled: bool,
) -> Dict[str, List[Example]]:
    """
    Return a new eval_by with stripped prompts (without mutating originals).
    """
    if not enabled or not answer_prefix:
        return eval_by

    out: Dict[str, List[Example]] = {}
    for task, exs in eval_by.items():
        new_list: List[Example] = []
        for ex in exs:
            core, _ = strip_last_answer_prefix(ex.prompt, answer_prefix)
            new_list.append(Example(dataset=ex.dataset, ex_id=ex.ex_id, prompt=core, gold=ex.gold))
        out[task] = new_list
    return out


def md_table(rows: List[List[str]], header: List[str]) -> str:
    out = []
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "---|" * len(header))
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def latex_table(rows: List[List[str]], header: List[str], caption: str, label: str, colspec: str) -> str:
    def esc(s: str) -> str:
        return s.replace("%", "\\%").replace("_", "\\_")

    header_esc = [esc(h) for h in header]
    body = []
    for r in rows:
        body.append(" & ".join(esc(x) for x in r) + " \\\\")
    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\small\n"
        f"\\begin{{tabular}}{{{colspec}}}\n"
        "\\toprule\n"
        + " & ".join(header_esc)
        + " \\\\\n\\midrule\n"
        + "\n".join(body)
        + "\n\\bottomrule\n"
        "\\end{tabular}\n"
        f"\\caption{{{esc(caption)}}}\n"
        f"\\label{{{esc(label)}}}\n"
        "\\end{table}\n"
    )


def main():
    ap = argparse.ArgumentParser()

    # Model
    ap.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--trust_remote_code", type=int, default=0, choices=[0, 1])

    # Tasks / data
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument(
        "--tasks",
        type=str,
        default="gsm8k,commonsenseqa,strategyqa,aqua,arc_challenge,openbookqa,qasc,logiqa,boolq,piqa",
        help="Comma-separated tasks for basis estimation and (by default) eval. Tasks without candidates are skipped in eval.",
    )
    ap.add_argument("--n_prompts", type=int, default=128, help="n_subspace for basis estimation (per task)")
    ap.add_argument("--eval_n", type=int, default=256, help="n_eval for evaluation (per task)")
    ap.add_argument(
        "--eval_tasks",
        type=str,
        default="",
        help="Optional: evaluate only these tasks (comma-separated). Empty => auto (tasks with discrete candidates).",
    )

    # Basis estimation
    ap.add_argument("--calib_decode_max_new_tokens", type=int, default=128)
    ap.add_argument("--per_task_max_states", type=int, default=20000)
    ap.add_argument("--pca_var", type=float, default=0.95)
    ap.add_argument("--min_dim", type=int, default=1)
    ap.add_argument("--max_dim", type=int, default=4096)
    ap.add_argument("--tau", type=float, default=1e-3)
    ap.add_argument("--m_shared", type=str, default="all")
    ap.add_argument("--match_state_count", type=int, default=0, choices=[0, 1])
    ap.add_argument(
        "--k_eval",
        type=int,
        default=0,
        help="If >0, force dimension-matched evaluation to use k_eval (clamped to <=k_match). 0 => use k_match.",
    )

    # Prompt template knobs (passed to benchmark_dataloaders)
    ap.add_argument("--template_randomization", type=int, default=0, choices=[0, 1])
    ap.add_argument("--template_seed", type=int, default=1234)
    ap.add_argument("--shuffle_choices", type=int, default=0, choices=[0, 1])
    ap.add_argument("--add_answer_prefix", type=int, default=0, choices=[0, 1])
    ap.add_argument("--answer_prefix", type=str, default="\nFinal answer:")

    # Forced-choice
    ap.add_argument("--alpha_remove", type=float, default=1.0)

    ap.add_argument("--fc_warmup_tokens", type=int, default=0)
    ap.add_argument("--fc_warmup_decoding", type=str, default="greedy", choices=["greedy", "sample"])
    ap.add_argument("--fc_warmup_seed", type=int, default=123)
    ap.add_argument("--fc_warmup_ban_eos", type=int, default=1, choices=[0, 1])
    ap.add_argument("--fc_warmup_temperature", type=float, default=0.7)
    ap.add_argument("--fc_warmup_top_p", type=float, default=0.9)
    ap.add_argument("--fc_warmup_top_k", type=int, default=0)

    ap.add_argument("--fc_prefix_mode", type=str, default="auto", choices=["auto", "always", "never"])
    ap.add_argument("--fc_answer_prefix", type=str, default="\nFinal answer:")
    ap.add_argument("--fc_save_scores", type=int, default=1, choices=[0, 1])

    # Important: strip prompts to keep scoring at answer slot
    ap.add_argument(
        "--strip_eval_answer_prefix",
        type=int,
        default=1,
        choices=[0, 1],
        help="If 1 (recommended), strip trailing fc_answer_prefix from eval prompts before warmup+scoring.",
    )

    # Runtime
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_prompt_len", type=int, default=512)

    # Stats
    ap.add_argument("--bootstrap_iters", type=int, default=5000)
    ap.add_argument("--perm_iters", type=int, default=10000)
    ap.add_argument("--ci_alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)

    # Outputs
    ap.add_argument("--out_json", type=str, default=os.path.join(THIS_DIR, "prefill_vs_decode_alignment_forced_choice.json"))
    ap.add_argument("--out_txt", type=str, default=os.path.join(THIS_DIR, "prefill_vs_decode_alignment_forced_choice.txt"))
    ap.add_argument("--out_md", type=str, default=os.path.join(THIS_DIR, "prefill_vs_decode_alignment_forced_choice.md"))
    ap.add_argument("--out_tex", type=str, default=os.path.join(THIS_DIR, "prefill_vs_decode_alignment_forced_choice.tex"))

    args = ap.parse_args()

    # Normalize prefix args
    args.answer_prefix = _norm_prefix_arg(args.answer_prefix)
    args.fc_answer_prefix = _norm_prefix_arg(args.fc_answer_prefix)

    # Seeds
    EP.set_global_seed(int(args.seed))

    tasks = [t.strip() for t in str(args.tasks).split(",") if t.strip()]
    if len(tasks) < 2:
        raise RuntimeError("Need at least 2 tasks in --tasks.")

    # Decide eval_tasks
    if str(args.eval_tasks).strip():
        eval_tasks = [t.strip() for t in str(args.eval_tasks).split(",") if t.strip()]
    else:
        # default: only tasks with discrete candidates
        eval_tasks = [t for t in tasks if len(EP.candidate_strings(t)) > 0]

    if not eval_tasks:
        raise RuntimeError(
            "No eval tasks resolved. "
            "If you included only non-discrete tasks (e.g., gsm8k), forced-choice eval will skip them."
        )

    # Warnings for risky configs
    if args.fc_prefix_mode == "never":
        print(
            "[WARN] fc_prefix_mode=never. If prompts do not already end with an answer prefix, "
            "forced-choice accuracy can collapse toward chance."
        )
    if int(args.fc_warmup_tokens) > 0 and args.fc_prefix_mode == "never":
        print(
            "[WARN] fc_warmup_tokens>0 with fc_prefix_mode=never usually yields near-chance, "
            "because scoring happens at an arbitrary deep decode position."
        )
    if int(args.fc_warmup_tokens) > 0 and int(args.add_answer_prefix) == 1 and int(args.strip_eval_answer_prefix) == 0:
        print(
            "[WARN] You set add_answer_prefix=1 but strip_eval_answer_prefix=0 and warmup_tokens>0. "
            "This can move scoring away from the answer slot (prefix already in prompt, then warmup after it). "
            "Recommended: --strip_eval_answer_prefix 1 (default)."
        )

    print(f"[Env] model={args.model} device={args.device} dtype={args.dtype} trust_remote_code={bool(args.trust_remote_code)}")
    print(f"[Env] layer={args.layer}")
    print(f"[Env] tasks(basis)={tasks}")
    print(f"[Env] eval_tasks={eval_tasks}")
    print(
        f"[Env] forced-choice only | warmup_tokens={args.fc_warmup_tokens} warmup_decoding={args.fc_warmup_decoding} "
        f"prefix_mode={args.fc_prefix_mode} fc_answer_prefix={args.fc_answer_prefix!r} strip_eval_answer_prefix={bool(args.strip_eval_answer_prefix)}"
    )

    # 1) Load model/tokenizer
    model, tok = EP.load_model_and_tokenizer(
        args.model,
        args.device,
        args.dtype,
        trust_remote_code=bool(args.trust_remote_code),
    )

    # 2) Load data (benchmark_dataloaders)
    sub_by, eval_by, meta_by = load_selected_tasks(
        tasks=tasks,
        n_subspace=int(args.n_prompts),
        n_eval=int(args.eval_n),
        seed=int(args.seed),
        template_seed=int(args.template_seed),
        template_randomization=bool(args.template_randomization),
        shuffle_choices=bool(args.shuffle_choices),
        add_answer_prefix=bool(args.add_answer_prefix),
        answer_prefix=args.answer_prefix,
    )

    # 3) Strip eval prompts for correct answer-slot probing (optional but recommended)
    eval_by_core = strip_eval_prompts(
        eval_by,
        answer_prefix=args.fc_answer_prefix,
        enabled=bool(args.strip_eval_answer_prefix),
    )

    # 4) Compute shared bases (decode vs prefill) via eval_perf
    print("\n" + "=" * 80)
    print("[Basis] Computing decode-est vs prefill-est shared bases via eval_perf.compute_decode_prefill_shared_bases()")
    print("=" * 80)
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
        seed=int(args.seed),
        match_state_count=bool(args.match_state_count),
        k_eval=int(args.k_eval),
    )

    print(f"[Basis] k_decode={bases.k_decode} k_prefill={bases.k_prefill} k_eval={bases.k_eval}")
    print("[Basis] similarity_full:", json.dumps(bases.similarity_full, indent=2))
    print("[Basis] similarity_k   :", json.dumps(bases.similarity_k, indent=2))
    print("[Basis] energy summary :", json.dumps(bases.energy, indent=2))

    # 5) Precompute warmup tokens (baseline-generated, teacher-forced)
    warmup_by_task: Dict[str, np.ndarray] = {}
    if int(args.fc_warmup_tokens) > 0:
        print("\n" + "=" * 80)
        print(f"[FC Warmup] Precomputing baseline warmup tokens W={args.fc_warmup_tokens} via eval_perf.precompute_fc_warmup_tokens()")
        print("=" * 80)

        for task in eval_tasks:
            if len(EP.candidate_strings(task)) == 0:
                continue
            prompts = [ex.prompt for ex in eval_by_core[task]]
            warm_ids = EP.precompute_fc_warmup_tokens(
                model,
                tok,
                prompts,
                warmup_tokens=int(args.fc_warmup_tokens),
                batch_size=int(args.batch_size),
                max_prompt_len=int(args.max_prompt_len),
                decoding=str(args.fc_warmup_decoding),
                temperature=float(args.fc_warmup_temperature),
                top_p=float(args.fc_warmup_top_p),
                top_k=int(args.fc_warmup_top_k),
                ban_eos=bool(args.fc_warmup_ban_eos),
                seed=EP.stable_int_seed(int(args.seed), int(args.fc_warmup_seed), task, "warmup"),
            )
            warmup_by_task[task] = warm_ids

            if warm_ids.shape[0] > 0 and warm_ids.shape[1] > 0:
                demo = tok.decode(warm_ids[0].tolist(), skip_special_tokens=True)
                print(f"[FC Warmup] {task}: warmup_ids shape={warm_ids.shape}; demo text[:120]={demo[:120]!r}")

    # 6) Forced-choice evaluation (no generation)
    layer_indices = [int(args.layer)]
    eval_results: Dict[str, Any] = {}

    for task in eval_tasks:
        exs = eval_by_core.get(task, [])
        n = len(exs)
        print("\n" + "-" * 80)
        print(f"[Eval] task={task} n={n}")
        print("-" * 80)

        if len(EP.candidate_strings(task)) == 0:
            print(f"[Skip] {task}: no forced-choice candidates in eval_perf.candidate_strings()")
            continue

        warm_ids = warmup_by_task.get(task, None)

        # ---- baseline ----
        base_detail = EP.forced_choice_logprob_eval(
            model, tok, exs, task,
            layer_indices=layer_indices,
            basis_np=None,
            alpha=0.0,
            batch_size=int(args.batch_size),
            max_prompt_len=int(args.max_prompt_len),
            warmup_token_ids=warm_ids,
            answer_prefix=args.fc_answer_prefix,
            prefix_mode=str(args.fc_prefix_mode),
            save_scores=bool(args.fc_save_scores),
        )
        base_arr = np.array(base_detail["correct"], dtype=np.float32)
        b_acc, b_lo, b_hi = EP.bootstrap_ci_mean(
            base_arr,
            int(args.bootstrap_iters),
            float(args.ci_alpha),
            seed=EP.stable_int_seed(int(args.seed), task, "forced_choice", "baseline", float(args.alpha_remove)),
        )
        print(f"  baseline acc={EP.fmt_acc(b_acc, b_lo, b_hi)} {base_detail.get('hook_stats')}")

        # ---- decode/prefill FULL ----
        dec_full_detail = EP.forced_choice_logprob_eval(
            model, tok, exs, task,
            layer_indices=layer_indices,
            basis_np=bases.Q_decode_full,
            alpha=float(args.alpha_remove),
            batch_size=int(args.batch_size),
            max_prompt_len=int(args.max_prompt_len),
            warmup_token_ids=warm_ids,
            answer_prefix=args.fc_answer_prefix,
            prefix_mode=str(args.fc_prefix_mode),
            save_scores=bool(args.fc_save_scores),
        )
        dec_full_arr = np.array(dec_full_detail["correct"], dtype=np.float32)
        df_acc, df_lo, df_hi = EP.bootstrap_ci_mean(
            dec_full_arr,
            int(args.bootstrap_iters),
            float(args.ci_alpha),
            seed=EP.stable_int_seed(int(args.seed), task, "forced_choice", "decode_full", float(args.alpha_remove)),
        )
        print(f"  decode_full acc={EP.fmt_acc(df_acc, df_lo, df_hi)} {dec_full_detail.get('hook_stats')}")

        pre_full_detail = EP.forced_choice_logprob_eval(
            model, tok, exs, task,
            layer_indices=layer_indices,
            basis_np=bases.Q_prefill_full,
            alpha=float(args.alpha_remove),
            batch_size=int(args.batch_size),
            max_prompt_len=int(args.max_prompt_len),
            warmup_token_ids=warm_ids,
            answer_prefix=args.fc_answer_prefix,
            prefix_mode=str(args.fc_prefix_mode),
            save_scores=bool(args.fc_save_scores),
        )
        pre_full_arr = np.array(pre_full_detail["correct"], dtype=np.float32)
        pf_acc, pf_lo, pf_hi = EP.bootstrap_ci_mean(
            pre_full_arr,
            int(args.bootstrap_iters),
            float(args.ci_alpha),
            seed=EP.stable_int_seed(int(args.seed), task, "forced_choice", "prefill_full", float(args.alpha_remove)),
        )
        print(f"  prefill_full acc={EP.fmt_acc(pf_acc, pf_lo, pf_hi)} {pre_full_detail.get('hook_stats')}")

        # ---- decode/prefill k (dimension matched / k_eval) ----
        dec_k_detail = EP.forced_choice_logprob_eval(
            model, tok, exs, task,
            layer_indices=layer_indices,
            basis_np=bases.Q_decode_k,
            alpha=float(args.alpha_remove),
            batch_size=int(args.batch_size),
            max_prompt_len=int(args.max_prompt_len),
            warmup_token_ids=warm_ids,
            answer_prefix=args.fc_answer_prefix,
            prefix_mode=str(args.fc_prefix_mode),
            save_scores=bool(args.fc_save_scores),
        )
        dec_k_arr = np.array(dec_k_detail["correct"], dtype=np.float32)
        dk_acc, dk_lo, dk_hi = EP.bootstrap_ci_mean(
            dec_k_arr,
            int(args.bootstrap_iters),
            float(args.ci_alpha),
            seed=EP.stable_int_seed(int(args.seed), task, "forced_choice", "decode_k", float(args.alpha_remove)),
        )
        print(f"  decode_k acc={EP.fmt_acc(dk_acc, dk_lo, dk_hi)}")

        pre_k_detail = EP.forced_choice_logprob_eval(
            model, tok, exs, task,
            layer_indices=layer_indices,
            basis_np=bases.Q_prefill_k,
            alpha=float(args.alpha_remove),
            batch_size=int(args.batch_size),
            max_prompt_len=int(args.max_prompt_len),
            warmup_token_ids=warm_ids,
            answer_prefix=args.fc_answer_prefix,
            prefix_mode=str(args.fc_prefix_mode),
            save_scores=bool(args.fc_save_scores),
        )
        pre_k_arr = np.array(pre_k_detail["correct"], dtype=np.float32)
        pk_acc, pk_lo, pk_hi = EP.bootstrap_ci_mean(
            pre_k_arr,
            int(args.bootstrap_iters),
            float(args.ci_alpha),
            seed=EP.stable_int_seed(int(args.seed), task, "forced_choice", "prefill_k", float(args.alpha_remove)),
        )
        print(f"  prefill_k acc={EP.fmt_acc(pk_acc, pk_lo, pk_hi)}")

        rnd_k_detail = EP.forced_choice_logprob_eval(
            model, tok, exs, task,
            layer_indices=layer_indices,
            basis_np=bases.Q_rand_k,
            alpha=float(args.alpha_remove),
            batch_size=int(args.batch_size),
            max_prompt_len=int(args.max_prompt_len),
            warmup_token_ids=warm_ids,
            answer_prefix=args.fc_answer_prefix,
            prefix_mode=str(args.fc_prefix_mode),
            save_scores=bool(args.fc_save_scores),
        )
        rnd_k_arr = np.array(rnd_k_detail["correct"], dtype=np.float32)
        rk_acc, rk_lo, rk_hi = EP.bootstrap_ci_mean(
            rnd_k_arr,
            int(args.bootstrap_iters),
            float(args.ci_alpha),
            seed=EP.stable_int_seed(int(args.seed), task, "forced_choice", "rand_k", float(args.alpha_remove)),
        )
        print(f"  rand_k acc={EP.fmt_acc(rk_acc, rk_lo, rk_hi)}")

        # ---- paired test (decode_k - prefill_k) ----
        stat_dk_vs_pk = EP.summarize_paired(
            baseline_correct=pre_k_arr,
            treatment=dec_k_arr,
            iters_boot=int(args.bootstrap_iters),
            iters_perm=int(args.perm_iters),
            alpha=float(args.ci_alpha),
            seed=EP.stable_int_seed(int(args.seed), task, "forced_choice", "paired", "dk_vs_pk", float(args.alpha_remove)),
        )

        print(
            f"  [Paired] decode_k - prefill_k: "
            f"Δ={stat_dk_vs_pk['mean_diff']*100:+.1f} "
            f"CI[{stat_dk_vs_pk['ci_low']*100:+.1f},{stat_dk_vs_pk['ci_high']*100:+.1f}] "
            f"p={stat_dk_vs_pk['p_value']:.3g}"
        )

        eval_results[task] = {
            "protocol": "forced_choice",
            "n": int(n),
            "baseline": {"acc": float(b_acc), "ci": [float(b_lo), float(b_hi)], "detail": base_detail},
            "decode_full": {"acc": float(df_acc), "ci": [float(df_lo), float(df_hi)], "k": int(bases.k_decode), "detail": dec_full_detail},
            "prefill_full": {"acc": float(pf_acc), "ci": [float(pf_lo), float(pf_hi)], "k": int(bases.k_prefill), "detail": pre_full_detail},
            "decode_k": {"acc": float(dk_acc), "ci": [float(dk_lo), float(dk_hi)], "k": int(bases.k_eval), "detail": dec_k_detail},
            "prefill_k": {"acc": float(pk_acc), "ci": [float(pk_lo), float(pk_hi)], "k": int(bases.k_eval), "detail": pre_k_detail},
            "rand_k": {"acc": float(rk_acc), "ci": [float(rk_lo), float(rk_hi)], "k": int(bases.k_eval), "detail": rnd_k_detail},
            "paired": {"decode_k_minus_prefill_k": stat_dk_vs_pk},
        }

    # 7) Build tables
    rows_k = []
    for task, r in eval_results.items():
        stat = r["paired"]["decode_k_minus_prefill_k"]
        rows_k.append([
            task,
            r["protocol"],
            str(r["n"]),
            EP.fmt_acc(r["baseline"]["acc"], r["baseline"]["ci"][0], r["baseline"]["ci"][1]),
            EP.fmt_acc(r["decode_k"]["acc"], r["decode_k"]["ci"][0], r["decode_k"]["ci"][1]),
            EP.fmt_acc(r["prefill_k"]["acc"], r["prefill_k"]["ci"][0], r["prefill_k"]["ci"][1]),
            EP.fmt_acc(r["rand_k"]["acc"], r["rand_k"]["ci"][0], r["rand_k"]["ci"][1]),
            f"{stat['mean_diff']*100:+.1f} [{stat['ci_low']*100:+.1f}, {stat['ci_high']*100:+.1f}]",
            f"{stat['p_value']:.3g}",
        ])

    header_k = [
        "Task",
        "Protocol",
        "n",
        "Baseline",
        f"Decode-shared (k={bases.k_eval})",
        f"Prefill-shared (k={bases.k_eval})",
        f"Random (k={bases.k_eval})",
        "Δ(Decode-Prefill) [CI]",
        "p",
    ]
    md_k = md_table(rows_k, header_k)

    # Native-k table (reference)
    rows_nat = []
    for task, r in eval_results.items():
        delta = (r["decode_full"]["acc"] - r["prefill_full"]["acc"]) * 100.0
        rows_nat.append([
            task,
            r["protocol"],
            str(r["n"]),
            EP.fmt_acc(r["baseline"]["acc"], r["baseline"]["ci"][0], r["baseline"]["ci"][1]),
            EP.fmt_acc(r["decode_full"]["acc"], r["decode_full"]["ci"][0], r["decode_full"]["ci"][1]),
            EP.fmt_acc(r["prefill_full"]["acc"], r["prefill_full"]["ci"][0], r["prefill_full"]["ci"][1]),
            f"{delta:+.1f}",
            "(n/a)",
        ])

    header_nat = [
        "Task",
        "Protocol",
        "n",
        "Baseline",
        f"Decode-shared (k={bases.k_decode})",
        f"Prefill-shared (k={bases.k_prefill})",
        "Δ(Decode-Prefill)",
        "p",
    ]
    md_nat = md_table(rows_nat, header_nat)

    tex_k = latex_table(
        rows_k,
        header_k,
        caption=(
            f"Prefill-vs-Decode alignment experiment (forced-choice only). "
            f"Dimension-matched bases (k={bases.k_eval}). "
            f"Warmup W={int(args.fc_warmup_tokens)}, prefix_mode={args.fc_prefix_mode}."
        ),
        label="tab:prefill-vs-decode-kmatch-fc",
        colspec="llrcccccc",
    )
    tex_nat = latex_table(
        rows_nat,
        header_nat,
        caption="Native shared-k reference table (no dimension matching). Forced-choice only.",
        label="tab:prefill-vs-decode-native-fc",
        colspec="llrccccc",
    )

    # 8) Save
    results = {
        "config": {
            "model": args.model,
            "dtype": args.dtype,
            "device": args.device,
            "trust_remote_code": bool(args.trust_remote_code),
            "layer": int(args.layer),
            "tasks": tasks,
            "eval_tasks": eval_tasks,
            "n_prompts": int(args.n_prompts),
            "eval_n": int(args.eval_n),
            "calib_decode_max_new_tokens": int(args.calib_decode_max_new_tokens),
            "per_task_max_states": int(args.per_task_max_states),
            "pca_var": float(args.pca_var),
            "tau": float(args.tau),
            "m_shared": str(args.m_shared),
            "match_state_count": bool(args.match_state_count),
            "k_eval": int(args.k_eval),
            "alpha_remove": float(args.alpha_remove),
            "prompt": {
                "template_randomization": bool(args.template_randomization),
                "template_seed": int(args.template_seed),
                "shuffle_choices": bool(args.shuffle_choices),
                "add_answer_prefix": bool(args.add_answer_prefix),
                "answer_prefix": args.answer_prefix,
            },
            "forced_choice": {
                "enabled": True,
                "warmup_tokens": int(args.fc_warmup_tokens),
                "warmup_decoding": str(args.fc_warmup_decoding),
                "warmup_seed": int(args.fc_warmup_seed),
                "warmup_ban_eos": bool(args.fc_warmup_ban_eos),
                "warmup_temperature": float(args.fc_warmup_temperature),
                "warmup_top_p": float(args.fc_warmup_top_p),
                "warmup_top_k": int(args.fc_warmup_top_k),
                "prefix_mode": str(args.fc_prefix_mode),
                "answer_prefix": args.fc_answer_prefix,
                "strip_eval_answer_prefix": bool(args.strip_eval_answer_prefix),
                "save_scores": bool(args.fc_save_scores),
            },
            "stats": {
                "bootstrap_iters": int(args.bootstrap_iters),
                "perm_iters": int(args.perm_iters),
                "ci_alpha": float(args.ci_alpha),
                "seed": int(args.seed),
            },
            "dataset_meta": meta_by,
        },
        "basis": {
            "k_decode": int(bases.k_decode),
            "k_prefill": int(bases.k_prefill),
            "k_eval": int(bases.k_eval),
            "similarity_full": bases.similarity_full,
            "similarity_k": bases.similarity_k,
            "energy": bases.energy,
            "extra_decode": bases.extra_decode,
            "extra_prefill": bases.extra_prefill,
        },
        "eval": eval_results,
        "tables": {
            "markdown_kmatch": md_k,
            "markdown_native": md_nat,
            "latex_kmatch": tex_k,
            "latex_native": tex_nat,
        },
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=EP.json_default)

    summary_lines = []
    summary_lines.append("[Summary]")
    summary_lines.append(f"Model={args.model} dtype={args.dtype} device={args.device} layer={args.layer} trust_remote_code={bool(args.trust_remote_code)}")
    summary_lines.append(f"Tasks(basis)={tasks}")
    summary_lines.append(f"Eval tasks={eval_tasks}")
    summary_lines.append(f"Decode shared_k={bases.k_decode} Prefill shared_k={bases.k_prefill} k_eval={bases.k_eval}")
    summary_lines.append(f"Forced-choice warmup W={int(args.fc_warmup_tokens)} prefix_mode={args.fc_prefix_mode} prefix={args.fc_answer_prefix!r}")
    summary_lines.append(f"Strip eval answer_prefix: {bool(args.strip_eval_answer_prefix)}")
    summary_lines.append("")
    summary_lines.append("## Dimension-matched results (k_eval)")
    summary_lines.append(md_k)
    summary_lines.append("")
    summary_lines.append("## Native-k reference")
    summary_lines.append(md_nat)
    summary_lines.append("")

    with open(args.out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")
    with open(args.out_tex, "w", encoding="utf-8") as f:
        f.write(tex_k + "\n" + tex_nat + "\n")

    print("\n" + "=" * 80)
    print("\n".join(summary_lines[:14]))
    print("...")
    print(f"[Done] JSON: {args.out_json}")
    print(f"[Done] TXT : {args.out_txt}")
    print(f"[Done] MD  : {args.out_md}")
    print(f"[Done] TEX : {args.out_tex}")
    print("=" * 80)


if __name__ == "__main__":
    main()
