#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REBUTTAL_DIR = os.path.normpath(os.path.join(THIS_DIR, ".."))
if REBUTTAL_DIR not in sys.path:
    sys.path.append(REBUTTAL_DIR)

import exp_ranking_flip_steering as rank_base


def _derive_out_md(out_json: str) -> str:
    root, _ = os.path.splitext(out_json)
    return root + ".md"


def _fmt(x: float, digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "nan"
    return f"{x:.{digits}f}"


def _bootstrap_mean_ci(values: List[float], *, n_boot: int, seed: int) -> Dict[str, float]:
    arr = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "ci95_lo": float("nan"), "ci95_hi": float("nan")}
    mean = float(np.mean(arr))
    if arr.size == 1 or n_boot <= 0:
        return {"mean": mean, "ci95_lo": mean, "ci95_hi": mean}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, arr.size, size=arr.size)
        draws.append(float(np.mean(arr[idx])))
    lo, hi = np.percentile(np.asarray(draws, dtype=np.float64), [2.5, 97.5])
    return {"mean": mean, "ci95_lo": float(lo), "ci95_hi": float(hi)}


def _summarize_seed_scores(scores_by_seed: Dict[int, float], *, positive_threshold: float) -> Dict[str, Any]:
    seeds = sorted(scores_by_seed.keys())
    vals = np.asarray([scores_by_seed[s] for s in seeds], dtype=np.float64)
    ddof = 1 if vals.size > 1 else 0
    return {
        "n_templates": int(vals.size),
        "mean": float(np.mean(vals)) if vals.size else float("nan"),
        "median": float(np.median(vals)) if vals.size else float("nan"),
        "worst": float(np.min(vals)) if vals.size else float("nan"),
        "std": float(np.std(vals, ddof=ddof)) if vals.size else float("nan"),
        "range": float(np.max(vals) - np.min(vals)) if vals.size else float("nan"),
        "positive_rate": float(np.mean(vals > positive_threshold)) if vals.size else float("nan"),
        "by_seed": {int(s): float(scores_by_seed[s]) for s in seeds},
    }


def _selection_report_from_raw(
    raw: Dict[str, Any],
    *,
    positive_threshold: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> Dict[str, Any]:
    vectors = raw.get("vectors", {})
    if not isinstance(vectors, dict) or not vectors:
        raise RuntimeError("rankflip_json has no vectors")

    sample_vec = next(iter(vectors.values()))
    has_real_by_seed = (
        isinstance(sample_vec.get("score_real_summary"), dict)
        and isinstance(sample_vec["score_real_summary"].get("by_seed"), dict)
    )
    has_task_real = isinstance(sample_vec.get("delta_real_decode_by_seed"), dict)
    if not has_real_by_seed or not has_task_real:
        raise RuntimeError(
            "rankflip_json must come from rebuttal/exp_ranking_flip_steering.py "
            "or another raw output with *_summary and delta_real_decode_by_seed fields"
        )

    def _rank_score(payload: Dict[str, Any], key: str) -> float:
        if key in payload:
            return float(payload[key])
        mean_key = f"{key}_mean"
        if mean_key in payload:
            return float(payload[mean_key])
        raise KeyError(key)

    chosen_trad_name = max(vectors.keys(), key=lambda n: _rank_score(vectors[n], "score_rank_trad"))
    chosen_decode_name = max(vectors.keys(), key=lambda n: _rank_score(vectors[n], "score_rank_decode"))

    trad_scores = {int(k): float(v) for k, v in vectors[chosen_trad_name]["score_real_summary"]["by_seed"].items()}
    decode_scores = {int(k): float(v) for k, v in vectors[chosen_decode_name]["score_real_summary"]["by_seed"].items()}
    common_seeds = sorted(set(trad_scores.keys()) & set(decode_scores.keys()))

    seed_diffs = [decode_scores[s] - trad_scores[s] for s in common_seeds]
    seed_win_rate = float(np.mean([x > 0 for x in seed_diffs])) if seed_diffs else float("nan")

    task_win_diffs: List[float] = []
    trad_task = vectors[chosen_trad_name].get("delta_real_decode_by_seed", {})
    decode_task = vectors[chosen_decode_name].get("delta_real_decode_by_seed", {})
    for sk in common_seeds:
        t_map = trad_task.get(str(sk), trad_task.get(int(sk), {}))
        d_map = decode_task.get(str(sk), decode_task.get(int(sk), {}))
        for task in sorted(set(t_map.keys()) & set(d_map.keys())):
            task_win_diffs.append(float(d_map[task]) - float(t_map[task]))

    trad_summary = _summarize_seed_scores(trad_scores, positive_threshold=positive_threshold)
    decode_summary = _summarize_seed_scores(decode_scores, positive_threshold=positive_threshold)

    diff_report = {
        "mean_advantage": _bootstrap_mean_ci(seed_diffs, n_boot=bootstrap_samples, seed=bootstrap_seed + 1),
        "win_rate_across_templates": seed_win_rate,
        "n_common_templates": int(len(common_seeds)),
        "positive_rate_advantage": float(decode_summary["positive_rate"] - trad_summary["positive_rate"])
        if np.isfinite(decode_summary["positive_rate"]) and np.isfinite(trad_summary["positive_rate"])
        else float("nan"),
        "worst_advantage": float(decode_summary["worst"] - trad_summary["worst"])
        if np.isfinite(decode_summary["worst"]) and np.isfinite(trad_summary["worst"])
        else float("nan"),
        "std_advantage": float(trad_summary["std"] - decode_summary["std"])
        if np.isfinite(decode_summary["std"]) and np.isfinite(trad_summary["std"])
        else float("nan"),
        "range_advantage": float(trad_summary["range"] - decode_summary["range"])
        if np.isfinite(decode_summary["range"]) and np.isfinite(trad_summary["range"])
        else float("nan"),
        "task_level_win_rate": float(np.mean([x > 0 for x in task_win_diffs])) if task_win_diffs else float("nan"),
        "task_level_mean_advantage": _bootstrap_mean_ci(task_win_diffs, n_boot=bootstrap_samples, seed=bootstrap_seed + 2),
    }

    return {
        "selection": {
            "chosen_by_trad": chosen_trad_name,
            "chosen_by_decode": chosen_decode_name,
        },
        "deployment": {
            "trad_selected": trad_summary,
            "decode_selected": decode_summary,
            "decode_minus_trad": diff_report,
        },
    }


def _evaluate_vector_on_seeds(
    *,
    model,
    tokenizer,
    steering: rank_base.SteeringVector,
    eval_sets_by_seed: Dict[int, Dict[str, List[Any]]],
    base_acc_by_seed: Dict[int, Dict[str, float]],
    tasks: List[str],
    args: argparse.Namespace,
) -> Tuple[Dict[int, Dict[str, float]], Dict[int, float]]:
    per_task_by_seed: Dict[int, Dict[str, float]] = {}
    score_by_seed: Dict[int, float] = {}
    for seed, eval_by in eval_sets_by_seed.items():
        d: Dict[str, float] = {}
        for task in tasks:
            res = rank_base.evaluate_with_steering(
                model=model,
                tokenizer=tokenizer,
                examples=eval_by[task],
                decoding=args.decoding,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                device=args.device,
                batch_size=args.batch_size,
                max_prompt_len=args.max_prompt_len,
                reasoning_token_threshold=args.reasoning_tokens,
                steering=steering,
                phase_mode="decode",
                staged=bool(args.staged),
                global_seed=args.seed,
                sample_seed=(args.sample_seed if args.decoding == "sample" else None),
            )
            d[task] = float(res["accuracy"] - base_acc_by_seed[seed][task])
        per_task_by_seed[seed] = d
        score_by_seed[seed] = rank_base.agg_task_scores(d, agg=args.agg)
    return per_task_by_seed, score_by_seed


def _run_direct_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    if rank_base.load_selected_tasks is None:
        raise RuntimeError(
            "benchmark_dataloaders is required for this script. "
            f"Import error was: {getattr(rank_base, '_IMPORT_ERR', 'unknown')}"
        )

    rank_base.set_global_seed(args.seed)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if not tasks:
        raise ValueError("Empty --tasks")

    max_vectors = None if args.max_vectors <= 0 else int(args.max_vectors)
    vecs = rank_base.load_vectors_from_manifest(args.vectors_manifest, max_vectors=max_vectors)
    if args.filter_regex:
        import re
        pat = re.compile(args.filter_regex)
        vecs = [v for v in vecs if pat.search(v.name) or pat.search(v.concept)]
        if not vecs:
            raise RuntimeError("No vectors after --filter_regex")

    rank_seeds = rank_base._dedup_keep_order(rank_base._parse_csv_ints(args.template_seeds_rank) or [int(args.template_seed_rank)])
    real_seeds = rank_base._dedup_keep_order(rank_base._parse_csv_ints(args.template_seeds_real) or [int(args.template_seed_real)])

    model, tokenizer = rank_base.load_model_and_tokenizer(args.model, args.device, args.model_dtype)
    hidden_dim = rank_base.infer_hidden_dim(model)
    if hidden_dim is not None:
        for sv in vecs:
            if sv.vec.shape[0] != hidden_dim:
                raise ValueError(f"Vector dim mismatch for {sv.name}: {sv.vec.shape[0]} != {hidden_dim}")

    def load_eval(template_seed: int) -> Dict[str, List[Any]]:
        _sub_by, eval_by, _meta = rank_base.load_selected_tasks(
            tasks=tasks,
            n_subspace=1,
            n_eval=args.n_eval,
            seed=args.seed,
            template_seed=template_seed,
            template_randomization=bool(args.template_randomization),
            shuffle_choices=bool(args.shuffle_choices),
            add_answer_prefix=bool(args.add_answer_prefix),
            answer_prefix=args.answer_prefix,
        )
        return eval_by

    eval_rank_by_seed = {s: load_eval(s) for s in rank_seeds}
    eval_real_by_seed = {s: load_eval(s) for s in real_seeds}

    base_rank_by_seed: Dict[int, Dict[str, float]] = {}
    base_real_by_seed: Dict[int, Dict[str, float]] = {}
    for seed, eval_by in eval_rank_by_seed.items():
        base_rank_by_seed[seed] = {}
        for task in tasks:
            res = rank_base.evaluate_with_steering(
                model=model,
                tokenizer=tokenizer,
                examples=eval_by[task],
                decoding=args.decoding,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                device=args.device,
                batch_size=args.batch_size,
                max_prompt_len=args.max_prompt_len,
                reasoning_token_threshold=args.reasoning_tokens,
                steering=None,
                phase_mode="none",
                staged=False,
                global_seed=args.seed,
                sample_seed=(args.sample_seed if args.decoding == "sample" else None),
            )
            base_rank_by_seed[seed][task] = float(res["accuracy"])

    for seed, eval_by in eval_real_by_seed.items():
        base_real_by_seed[seed] = {}
        for task in tasks:
            res = rank_base.evaluate_with_steering(
                model=model,
                tokenizer=tokenizer,
                examples=eval_by[task],
                decoding=args.decoding,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                device=args.device,
                batch_size=args.batch_size,
                max_prompt_len=args.max_prompt_len,
                reasoning_token_threshold=args.reasoning_tokens,
                steering=None,
                phase_mode="none",
                staged=False,
                global_seed=args.seed,
                sample_seed=(args.sample_seed if args.decoding == "sample" else None),
            )
            base_real_by_seed[seed][task] = float(res["accuracy"])

    ranking: Dict[str, Any] = {"vectors": {}}
    for sv in vecs:
        trad_scores: Dict[int, float] = {}
        decode_scores: Dict[int, float] = {}
        for seed, eval_by in eval_rank_by_seed.items():
            trad_task: Dict[str, float] = {}
            decode_task: Dict[str, float] = {}
            for task in tasks:
                res_trad = rank_base.evaluate_with_steering(
                    model=model,
                    tokenizer=tokenizer,
                    examples=eval_by[task],
                    decoding=args.decoding,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    device=args.device,
                    batch_size=args.batch_size,
                    max_prompt_len=args.max_prompt_len,
                    reasoning_token_threshold=args.reasoning_tokens,
                    steering=sv,
                    phase_mode=args.trad_mode,
                    staged=bool(args.staged),
                    global_seed=args.seed,
                    sample_seed=(args.sample_seed if args.decoding == "sample" else None),
                )
                res_decode = rank_base.evaluate_with_steering(
                    model=model,
                    tokenizer=tokenizer,
                    examples=eval_by[task],
                    decoding=args.decoding,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    device=args.device,
                    batch_size=args.batch_size,
                    max_prompt_len=args.max_prompt_len,
                    reasoning_token_threshold=args.reasoning_tokens,
                    steering=sv,
                    phase_mode=args.decode_mode,
                    staged=bool(args.staged),
                    global_seed=args.seed,
                    sample_seed=(args.sample_seed if args.decoding == "sample" else None),
                )
                trad_task[task] = float(res_trad["accuracy"] - base_rank_by_seed[seed][task])
                decode_task[task] = float(res_decode["accuracy"] - base_rank_by_seed[seed][task])
            trad_scores[seed] = rank_base.agg_task_scores(trad_task, agg=args.agg)
            decode_scores[seed] = rank_base.agg_task_scores(decode_task, agg=args.agg)
        ranking["vectors"][sv.name] = {
            "concept": sv.concept,
            "layer": sv.layer,
            "alpha": sv.alpha,
            "score_rank_trad_mean": float(np.mean(list(trad_scores.values()))),
            "score_rank_decode_mean": float(np.mean(list(decode_scores.values()))),
            "score_rank_trad_by_seed": {int(k): float(v) for k, v in trad_scores.items()},
            "score_rank_decode_by_seed": {int(k): float(v) for k, v in decode_scores.items()},
        }

    chosen_trad_name = max(ranking["vectors"].keys(), key=lambda n: float(ranking["vectors"][n]["score_rank_trad_mean"]))
    chosen_decode_name = max(ranking["vectors"].keys(), key=lambda n: float(ranking["vectors"][n]["score_rank_decode_mean"]))
    name_to_vec = {sv.name: sv for sv in vecs}

    trad_task_by_seed, trad_real_scores = _evaluate_vector_on_seeds(
        model=model,
        tokenizer=tokenizer,
        steering=name_to_vec[chosen_trad_name],
        eval_sets_by_seed=eval_real_by_seed,
        base_acc_by_seed=base_real_by_seed,
        tasks=tasks,
        args=args,
    )
    decode_task_by_seed, decode_real_scores = _evaluate_vector_on_seeds(
        model=model,
        tokenizer=tokenizer,
        steering=name_to_vec[chosen_decode_name],
        eval_sets_by_seed=eval_real_by_seed,
        base_acc_by_seed=base_real_by_seed,
        tasks=tasks,
        args=args,
    )

    raw_like = {
        "config": vars(args),
        "vectors": {
            chosen_trad_name: {
                "score_rank_trad": ranking["vectors"][chosen_trad_name]["score_rank_trad_mean"],
                "score_rank_decode": ranking["vectors"][chosen_trad_name]["score_rank_decode_mean"],
                "score_real_summary": {"by_seed": trad_real_scores},
                "delta_real_decode_by_seed": trad_task_by_seed,
            },
            chosen_decode_name: {
                "score_rank_trad": ranking["vectors"][chosen_decode_name]["score_rank_trad_mean"],
                "score_rank_decode": ranking["vectors"][chosen_decode_name]["score_rank_decode_mean"],
                "score_real_summary": {"by_seed": decode_real_scores},
                "delta_real_decode_by_seed": decode_task_by_seed,
            },
        },
    }

    deployment = _selection_report_from_raw(
        raw_like,
        positive_threshold=args.positive_threshold,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    deployment["ranking"] = ranking
    deployment["config"] = vars(args)
    return deployment


def _write_markdown(report: Dict[str, Any], out_md: str) -> None:
    lines: List[str] = []
    cfg = report.get("config", {})
    lines.append("# Steering Robustness — Selected Deployment")
    lines.append("")
    lines.append(f"- `vectors_manifest`: `{cfg.get('vectors_manifest', '')}`")
    lines.append(f"- `tasks`: `{cfg.get('tasks', '')}`")
    lines.append(f"- `template_seeds_rank`: `{cfg.get('template_seeds_rank', cfg.get('template_seed_rank', ''))}`")
    lines.append(f"- `template_seeds_real`: `{cfg.get('template_seeds_real', cfg.get('template_seed_real', ''))}`")
    lines.append("")

    sel = report.get("selection", {})
    if sel:
        lines.append("## Selection")
        lines.append("")
        lines.append(f"- `chosen_by_trad`: `{sel.get('chosen_by_trad', '')}`")
        lines.append(f"- `chosen_by_decode`: `{sel.get('chosen_by_decode', '')}`")
        lines.append("")

    dep = report.get("deployment", {})
    trad = dep.get("trad_selected", {})
    dec = dep.get("decode_selected", {})
    if trad and dec:
        lines.append("## Deployment")
        lines.append("")
        lines.append("| selected-by | mean | worst | std | range | positive-rate | n |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        lines.append(
            "| TRAD | "
            + " | ".join(
                [
                    _fmt(trad.get("mean", float("nan"))),
                    _fmt(trad.get("worst", float("nan"))),
                    _fmt(trad.get("std", float("nan"))),
                    _fmt(trad.get("range", float("nan"))),
                    _fmt(trad.get("positive_rate", float("nan"))),
                    str(trad.get("n_templates", 0)),
                ]
            )
            + " |"
        )
        lines.append(
            "| DECODE | "
            + " | ".join(
                [
                    _fmt(dec.get("mean", float("nan"))),
                    _fmt(dec.get("worst", float("nan"))),
                    _fmt(dec.get("std", float("nan"))),
                    _fmt(dec.get("range", float("nan"))),
                    _fmt(dec.get("positive_rate", float("nan"))),
                    str(dec.get("n_templates", 0)),
                ]
            )
            + " |"
        )
        lines.append("")

    diff = dep.get("decode_minus_trad", {})
    if diff:
        lines.append("## DECODE − TRAD")
        lines.append("")
        lines.append(f"- `mean_advantage`: `{_fmt(diff['mean_advantage']['mean'])}`")
        lines.append(f"- `mean_advantage_ci95`: `[{_fmt(diff['mean_advantage']['ci95_lo'])}, {_fmt(diff['mean_advantage']['ci95_hi'])}]`")
        lines.append(f"- `worst_advantage`: `{_fmt(diff.get('worst_advantage', float('nan')) )}`")
        lines.append(f"- `std_advantage`: `{_fmt(diff.get('std_advantage', float('nan')) )}`")
        lines.append(f"- `range_advantage`: `{_fmt(diff.get('range_advantage', float('nan')) )}`")
        lines.append(f"- `positive_rate_advantage`: `{_fmt(diff.get('positive_rate_advantage', float('nan')) )}`")
        lines.append(f"- `template_win_rate`: `{_fmt(diff.get('win_rate_across_templates', float('nan')), 3)}`")
        lines.append(f"- `task_level_win_rate`: `{_fmt(diff.get('task_level_win_rate', float('nan')), 3)}`")
        lines.append("")

    os.makedirs(os.path.dirname(out_md) or ".", exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rankflip_json", type=str, default="", help="Analyze an existing raw rankflip JSON instead of re-running.")
    ap.add_argument("--out_json", type=str, required=True)
    ap.add_argument("--out_md", type=str, default="")
    ap.add_argument("--positive_threshold", type=float, default=0.0)
    ap.add_argument("--bootstrap_samples", type=int, default=10000)
    ap.add_argument("--bootstrap_seed", type=int, default=123)

    ap.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    ap.add_argument("--device", type=str, default="cuda" if rank_base.torch.cuda.is_available() else "cpu")
    ap.add_argument("--model_dtype", type=str, default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--vectors_manifest", type=str, default="")
    ap.add_argument("--max_vectors", type=int, default=0)
    ap.add_argument("--filter_regex", type=str, default="")
    ap.add_argument("--tasks", type=str, default="commonsenseqa,arc_challenge,openbookqa,qasc,logiqa")
    ap.add_argument("--n_eval", type=int, default=128)
    ap.add_argument("--template_randomization", type=int, default=1, choices=[0, 1])
    ap.add_argument("--shuffle_choices", type=int, default=1, choices=[0, 1])
    ap.add_argument("--answer_prefix", type=str, default="\nFinal answer:")
    ap.add_argument("--add_answer_prefix", type=int, default=1, choices=[0, 1])
    ap.add_argument("--template_seed_rank", type=int, default=1234)
    ap.add_argument("--template_seed_real", type=int, default=5678)
    ap.add_argument("--template_seeds_rank", type=str, default="")
    ap.add_argument("--template_seeds_real", type=str, default="")
    ap.add_argument("--decoding", type=str, default="greedy", choices=["greedy", "sample"])
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--reasoning_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_prompt_len", type=int, default=512)
    ap.add_argument("--sample_seed", type=int, default=12345)
    ap.add_argument("--trad_mode", type=str, default="prefill", choices=["prefill", "both"])
    ap.add_argument("--decode_mode", type=str, default="decode", choices=["decode", "both"])
    ap.add_argument("--staged", type=int, default=1, choices=[0, 1])
    ap.add_argument("--agg", type=str, default="mean", choices=["mean", "min", "median"])
    ap.add_argument("--seed", type=int, default=42)
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    out_md = args.out_md or _derive_out_md(args.out_json)

    if args.rankflip_json:
        with open(os.path.expanduser(args.rankflip_json), "r", encoding="utf-8") as f:
            raw = json.load(f)
        report = _selection_report_from_raw(
            raw,
            positive_threshold=args.positive_threshold,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
        report["config"] = dict(raw.get("config", {}))
        report["config"]["rankflip_json"] = args.rankflip_json
    else:
        if not args.vectors_manifest:
            raise ValueError("--vectors_manifest is required when --rankflip_json is not provided")
        report = _run_direct_experiment(args)

    os.makedirs(os.path.dirname(os.path.expanduser(args.out_json)) or ".", exist_ok=True)
    with open(os.path.expanduser(args.out_json), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    _write_markdown(report, os.path.expanduser(out_md))
    print(f"[Saved] {args.out_json}")
    print(f"[Saved] {out_md}")


if __name__ == "__main__":
    main()
