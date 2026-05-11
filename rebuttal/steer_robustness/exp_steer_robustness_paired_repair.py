#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REBUTTAL_DIR = os.path.normpath(os.path.join(THIS_DIR, ".."))
if REBUTTAL_DIR not in sys.path:
    sys.path.append(REBUTTAL_DIR)

import exp_repair_controls_steering as repair_base


def _derive_out_md(out_json: str) -> str:
    root, _ = os.path.splitext(out_json)
    return root + ".md"


def _atomic_write_json(payload: Dict[str, Any], path: str) -> None:
    path = os.path.expanduser(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _fmt(x: float, digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "nan"
    return f"{x:.{digits}f}"


def _parse_metric_preference(metric: str) -> int:
    lower_is_better = {"std_delta", "range_delta", "negative_rate"}
    return -1 if metric in lower_is_better else 1


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


def _summarize_template_values(
    deltas: List[float],
    accs: Optional[List[float]],
    *,
    positive_threshold: float,
) -> Dict[str, float]:
    arr = np.asarray(deltas, dtype=np.float64)
    arr_acc = np.asarray(accs or [], dtype=np.float64)
    ddof = 1 if arr.size > 1 else 0
    out = {
        "n_templates": int(arr.size),
        "mean_delta": float(np.mean(arr)) if arr.size else float("nan"),
        "median_delta": float(np.median(arr)) if arr.size else float("nan"),
        "worst_delta": float(np.min(arr)) if arr.size else float("nan"),
        "std_delta": float(np.std(arr, ddof=ddof)) if arr.size else float("nan"),
        "range_delta": float(np.max(arr) - np.min(arr)) if arr.size else float("nan"),
        "positive_rate": float(np.mean(arr > positive_threshold)) if arr.size else float("nan"),
        "negative_rate": float(np.mean(arr < positive_threshold)) if arr.size else float("nan"),
    }
    if arr_acc.size:
        acc_ddof = 1 if arr_acc.size > 1 else 0
        out.update(
            {
                "mean_acc": float(np.mean(arr_acc)),
                "worst_acc": float(np.min(arr_acc)),
                "std_acc": float(np.std(arr_acc, ddof=acc_ddof)),
                "range_acc": float(np.max(arr_acc) - np.min(arr_acc)),
            }
        )
    return out


def _aggregate_paired_report(
    results: Dict[str, Any],
    *,
    positive_threshold: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> Dict[str, Any]:
    vectors = results.get("vectors", {})
    if not isinstance(vectors, dict) or not vectors:
        return {}

    methods = list(next(iter(vectors.values()))["summary"].keys())
    aggregate: Dict[str, Any] = {"methods": {}, "comparisons": {}, "shared_beats_all_controls": {}}

    metric_keys = [
        "mean_delta",
        "worst_delta",
        "std_delta",
        "range_delta",
        "positive_rate",
        "negative_rate",
    ]

    for method in methods:
        method_summaries = [vectors[vn]["summary"][method] for vn in vectors]
        aggregate["methods"][method] = {}
        for metric in metric_keys:
            vals = [float(s[metric]) for s in method_summaries if metric in s]
            aggregate["methods"][method][metric] = _bootstrap_mean_ci(
                vals,
                n_boot=bootstrap_samples,
                seed=bootstrap_seed + 17 * (methods.index(method) + 1) + 101 * (metric_keys.index(metric) + 1),
            )
        aggregate["methods"][method]["n_vectors"] = int(len(method_summaries))

    comparators = [m for m in methods if m != "shared"]
    comparison_metrics = ["mean_delta", "worst_delta", "std_delta", "range_delta", "positive_rate"]
    for comparator in comparators:
        comp: Dict[str, Any] = {"n_vectors": 0}
        for metric in comparison_metrics:
            sign = _parse_metric_preference(metric)
            diffs: List[float] = []
            wins = 0
            n = 0
            for vn in vectors:
                s_shared = float(vectors[vn]["summary"]["shared"][metric])
                s_other = float(vectors[vn]["summary"][comparator][metric])
                if not np.isfinite(s_shared) or not np.isfinite(s_other):
                    continue
                diff = sign * (s_shared - s_other)
                diffs.append(float(diff))
                n += 1
                if diff > 0:
                    wins += 1
            comp[f"{metric}_advantage"] = _bootstrap_mean_ci(
                diffs,
                n_boot=bootstrap_samples,
                seed=bootstrap_seed + 1009 * (comparators.index(comparator) + 1) + 67 * (comparison_metrics.index(metric) + 1),
            )
            comp[f"{metric}_winrate"] = (float(wins) / float(n)) if n else float("nan")
            comp["n_vectors"] = max(comp["n_vectors"], int(n))
        aggregate["comparisons"][f"shared_vs_{comparator}"] = comp

    controls = [m for m in methods if m not in {"orig", "shared"}]
    trio = {"worst_delta": 0, "std_delta": 0, "range_delta": 0}
    for vn in vectors:
        passed = True
        for control in controls:
            if control not in vectors[vn]["summary"]:
                continue
            if float(vectors[vn]["summary"]["shared"]["worst_delta"]) <= float(vectors[vn]["summary"][control]["worst_delta"]):
                passed = False
                break
            if float(vectors[vn]["summary"]["shared"]["std_delta"]) >= float(vectors[vn]["summary"][control]["std_delta"]):
                passed = False
                break
            if float(vectors[vn]["summary"]["shared"]["range_delta"]) >= float(vectors[vn]["summary"][control]["range_delta"]):
                passed = False
                break
        if passed:
            trio["worst_delta"] += 1
            trio["std_delta"] += 1
            trio["range_delta"] += 1
    n_vectors = len(vectors)
    aggregate["shared_beats_all_controls"] = {
        "n_vectors": int(n_vectors),
        "robustness_triple_winrate": (float(trio["worst_delta"]) / float(n_vectors)) if n_vectors else float("nan"),
        "positive_threshold": float(positive_threshold),
    }
    return aggregate


def _write_markdown(report: Dict[str, Any], out_md: str) -> None:
    lines: List[str] = []
    cfg = report.get("config", {})
    lines.append("# Steering Robustness — Paired Repair")
    lines.append("")
    lines.append(f"- `vectors_manifest`: `{cfg.get('vectors_manifest', '')}`")
    lines.append(f"- `tasks_eval`: `{cfg.get('tasks_eval', '')}`")
    lines.append(f"- `template_seeds`: `{cfg.get('template_seeds', '')}`")
    lines.append(f"- `positive_threshold`: `{cfg.get('positive_threshold', 0.0)}`")
    lines.append("")

    agg = report.get("aggregate", {})
    methods = agg.get("methods", {})
    if methods:
        lines.append("## Aggregate")
        lines.append("")
        lines.append("| method | mean Δ | worst Δ | std | range | positive-rate | n |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for method, stats in methods.items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        method,
                        _fmt(stats["mean_delta"]["mean"]),
                        _fmt(stats["worst_delta"]["mean"]),
                        _fmt(stats["std_delta"]["mean"]),
                        _fmt(stats["range_delta"]["mean"]),
                        _fmt(stats["positive_rate"]["mean"]),
                        str(stats["n_vectors"]),
                    ]
                )
                + " |"
            )
        lines.append("")

    comparisons = agg.get("comparisons", {})
    if comparisons:
        lines.append("## Shared vs Comparator")
        lines.append("")
        lines.append("| comparison | mean-adv | worst-adv | std-adv | range-adv | pos-adv | worst-win | std-win | range-win |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for name, stats in comparisons.items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        name,
                        _fmt(stats["mean_delta_advantage"]["mean"]),
                        _fmt(stats["worst_delta_advantage"]["mean"]),
                        _fmt(stats["std_delta_advantage"]["mean"]),
                        _fmt(stats["range_delta_advantage"]["mean"]),
                        _fmt(stats["positive_rate_advantage"]["mean"]),
                        _fmt(stats["worst_delta_winrate"]),
                        _fmt(stats["std_delta_winrate"]),
                        _fmt(stats["range_delta_winrate"]),
                    ]
                )
                + " |"
            )
        lines.append("")

    key = agg.get("shared_beats_all_controls", {})
    if key:
        lines.append("## Joint Robustness")
        lines.append("")
        lines.append(
            f"- `shared` beats all non-`orig` controls on worst/std/range for "
            f"`{_fmt(key.get('robustness_triple_winrate', float('nan')), 3)}` of vectors."
        )
        lines.append("")

    os.makedirs(os.path.dirname(out_md) or ".", exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def _validate_resume_config(existing_cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    checks = {
        "vectors_manifest": args.vectors_manifest,
        "tasks_eval": args.tasks_eval,
        "template_seeds": args.template_seeds,
        "basis_layers": args.basis_layers,
        "positive_threshold": float(args.positive_threshold),
    }
    mismatches = []
    for key, expected in checks.items():
        existing = existing_cfg.get(key)
        if existing is not None and existing != expected:
            mismatches.append(f"{key}: existing={existing!r} current={expected!r}")
    if mismatches:
        raise RuntimeError(
            "Resume checkpoint is incompatible with current arguments:\n- "
            + "\n- ".join(mismatches)
        )


def _load_resume_vectors(
    checkpoint_json: str,
    *,
    valid_names: List[str],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    checkpoint_json = os.path.expanduser(checkpoint_json)
    if not checkpoint_json or not os.path.exists(checkpoint_json):
        return {}
    with open(checkpoint_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        return {}
    _validate_resume_config(payload.get("config", {}), args)
    vectors = payload.get("vectors", {})
    if not isinstance(vectors, dict):
        return {}
    kept = {name: vectors[name] for name in valid_names if name in vectors}
    if kept:
        print(f"[Resume] Loaded {len(kept)}/{len(valid_names)} vectors from {checkpoint_json}")
    return kept


def _save_checkpoint(
    results: Dict[str, Any],
    *,
    checkpoint_json: str,
    out_md: str,
    positive_threshold: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
    total_vectors: int,
) -> None:
    results["aggregate"] = _aggregate_paired_report(
        results,
        positive_threshold=positive_threshold,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    results["checkpoint"] = {
        "completed_vectors": int(len(results.get("vectors", {}))),
        "total_vectors": int(total_vectors),
        "is_complete": bool(len(results.get("vectors", {})) >= total_vectors),
    }
    _atomic_write_json(results, checkpoint_json)
    if out_md:
        _write_markdown(results, out_md)
    print(
        f"[Checkpoint] {len(results.get('vectors', {}))}/{total_vectors} -> "
        f"{os.path.expanduser(checkpoint_json)}"
    )


def _run_direct_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    if repair_base.load_selected_tasks is None:
        raise RuntimeError(
            "benchmark_dataloaders is required for this script. "
            f"Import error was: {getattr(repair_base, '_IMPORT_ERR', 'unknown')}"
        )

    repair_base.set_global_seed(args.seed)

    max_vectors = None if args.max_vectors <= 0 else int(args.max_vectors)
    vecs = repair_base.load_vectors_from_manifest(args.vectors_manifest, max_vectors=max_vectors)
    if args.filter_regex:
        import re
        pat = re.compile(args.filter_regex)
        vecs = [v for v in vecs if pat.search(v.name) or pat.search(v.concept)]
        if not vecs:
            raise RuntimeError("No vectors after --filter_regex")
    valid_vector_names = [v.name for v in vecs]

    tasks_eval = [t.strip() for t in args.tasks_eval.split(",") if t.strip()]
    tasks_sub = [t.strip() for t in args.tasks_subspace.split(",") if t.strip()]
    template_seeds = repair_base._dedup_keep_order(repair_base._parse_csv_ints(args.template_seeds))
    if not template_seeds:
        raise ValueError("Empty --template_seeds")

    subspace_template_seed = int(args.subspace_template_seed) if args.subspace_template_seed is not None else int(template_seeds[0])
    if args.subspace_shuffle_choices < 0:
        subspace_shuffle_choices = bool(args.shuffle_choices)
    else:
        subspace_shuffle_choices = bool(args.subspace_shuffle_choices)

    if args.basis_layers.strip().lower() == "auto":
        basis_layers = sorted(set(int(v.layer) for v in vecs))
    else:
        basis_layers = repair_base._dedup_keep_order(repair_base._parse_csv_ints(args.basis_layers))
    if not basis_layers:
        raise ValueError("No basis layers specified")

    model, tokenizer = repair_base.load_model_and_tokenizer(args.model, args.device, args.model_dtype)
    hidden_dim = repair_base.infer_hidden_dim(model)
    if hidden_dim is not None:
        for v in vecs:
            if v.vec.shape[0] != hidden_dim:
                raise ValueError(f"Vector dim mismatch for {v.name}: {v.vec.shape[0]} != {hidden_dim}")

    calib_max_new_tokens = int(args.calib_decode_max_new_tokens)
    if calib_max_new_tokens <= 0:
        calib_max_new_tokens = int(args.reasoning_tokens)

    sub_by, _eval_dummy, _meta = repair_base.load_selected_tasks(
        tasks=tasks_sub,
        n_subspace=max(1, args.n_subspace),
        n_eval=1,
        seed=args.seed,
        template_seed=subspace_template_seed,
        template_randomization=bool(args.template_randomization),
        shuffle_choices=subspace_shuffle_choices,
        add_answer_prefix=bool(args.add_answer_prefix),
        answer_prefix=args.answer_prefix,
    )
    prompts_by_task = {t: [ex.prompt for ex in sub_by[t]] for t in tasks_sub if t in sub_by}

    bases_by_layer: Dict[int, Dict[str, Any]] = {}
    for layer in basis_layers:
        Q_shared = None
        if args.shared_basis_npy_pattern:
            spath = repair_base._format_layer_pattern(os.path.expanduser(args.shared_basis_npy_pattern), layer)
            if os.path.exists(spath):
                Q_shared = repair_base.orthonormalize_np(np.load(spath).astype(np.float32))

        Q_pca = None
        if args.pca_basis_npy_pattern:
            ppath = repair_base._format_layer_pattern(os.path.expanduser(args.pca_basis_npy_pattern), layer)
            if os.path.exists(ppath):
                Q_pca = repair_base.orthonormalize_np(np.load(ppath).astype(np.float32))

        info = None
        if Q_shared is None or Q_pca is None:
            info = repair_base.estimate_shared_and_pca_bases(
                model=model,
                tokenizer=tokenizer,
                prompts_by_task=prompts_by_task,
                layer_idx=layer,
                calib_batch_size=args.batch_size,
                calib_max_new_tokens=calib_max_new_tokens,
                per_task_max_states=args.per_task_max_states,
                max_prompt_len=args.max_prompt_len,
                decoding="greedy",
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                seed=args.seed,
                pca_var=args.pca_var,
                pca_max_dim=args.pca_max_dim,
                pca_max_rows=args.pca_max_rows,
                tau=args.tau,
                m_shared=args.m_shared,
            )
            joint = info["joint_subspace"]
            shared_indices = list(info["shared_indices"])
            if Q_shared is None:
                if not shared_indices:
                    raise RuntimeError(f"No shared components selected at layer {layer}")
                if args.shared_dim > 0:
                    shared_indices = shared_indices[: int(args.shared_dim)]
                Q_shared = repair_base.orthonormalize_np(joint[:, shared_indices])
            if Q_pca is None:
                Q_pca = repair_base.orthonormalize_np(joint[:, : Q_shared.shape[1]])

        assert Q_shared is not None and Q_pca is not None
        d = int(Q_shared.shape[0])
        k = int(min(Q_shared.shape[1], Q_pca.shape[1]))
        Q_shared = Q_shared[:, :k]
        Q_pca = Q_pca[:, :k]

        Q_pca_prefill = None
        pca_prefill_info = None
        if bool(args.include_pca_prefill):
            pinfo = repair_base.estimate_prefill_pca_basis(
                model=model,
                tokenizer=tokenizer,
                prompts_by_task=prompts_by_task,
                layer_idx=layer,
                calib_batch_size=args.batch_size,
                per_task_max_states=args.per_task_max_states,
                max_prompt_len=args.max_prompt_len,
                seed=args.seed,
                k=k,
                pca_max_dim=args.pca_max_dim,
                pca_max_rows=args.pca_max_rows,
            )
            Q_pca_prefill = pinfo["Q"]
            pca_prefill_info = {
                "n_rows": int(pinfo["n_rows"]),
                "cos_singulars_vs_decode_pca": repair_base._subspace_cos_singulars(Q_pca, Q_pca_prefill),
                "tasks_used": pinfo["tasks_used"],
            }

        Q_rand = repair_base.rand_orthonormal(d, k, seed=repair_base.stable_int_seed(args.seed, "Q_rand", layer, d, k))
        bases_by_layer[layer] = {
            "Q_shared": Q_shared,
            "Q_pca": Q_pca,
            "Q_pca_prefill": Q_pca_prefill,
            "Q_rand": Q_rand,
            "k": k,
            "d": d,
            "info": info,
            "pca_prefill_info": pca_prefill_info,
        }

    eval_sets: Dict[int, Dict[str, List[Any]]] = {}
    base_acc: Dict[int, Dict[str, float]] = {}
    for tseed in template_seeds:
        _sub_dummy, eval_by, _meta = repair_base.load_selected_tasks(
            tasks=tasks_eval,
            n_subspace=1,
            n_eval=args.n_eval,
            seed=args.seed,
            template_seed=tseed,
            template_randomization=bool(args.template_randomization),
            shuffle_choices=bool(args.shuffle_choices),
            add_answer_prefix=bool(args.add_answer_prefix),
            answer_prefix=args.answer_prefix,
        )
        eval_sets[tseed] = eval_by
        base_acc[tseed] = {}
        for task in tasks_eval:
            acc0 = repair_base.eval_decode_steering(
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
                v_np=None,
                alpha=0.0,
                layer_idx=basis_layers[0],
                staged=False,
                sample_seed=(args.sample_seed if args.decoding == "sample" else None),
            )
            base_acc[tseed][task] = float(acc0)

    methods = ["orig", "shared", "rand", "pca", "shrink"]
    if bool(args.include_pca_prefill):
        methods = ["orig", "shared", "rand", "pca", "pca_prefill", "shrink"]

    checkpoint_json = os.path.expanduser(args.resume_json or args.out_json)
    out_md = os.path.expanduser(args.out_md or _derive_out_md(args.out_json))

    results: Dict[str, Any] = {
        "config": vars(args),
        "basis_layers": basis_layers,
        "basis_summary": {
            str(layer): {
                "k": bases_by_layer[layer]["k"],
                "d": bases_by_layer[layer]["d"],
                "pca_prefill_info": bases_by_layer[layer]["pca_prefill_info"],
            }
            for layer in basis_layers
        },
        "vectors": _load_resume_vectors(
            checkpoint_json,
            valid_names=valid_vector_names,
            args=args,
        ),
    }

    staged = bool(args.staged)
    norm_match = bool(args.norm_match)
    completed_since_save = 0
    for sv in vecs:
        if sv.name in results["vectors"]:
            continue
        basis = bases_by_layer[sv.layer]
        v = sv.vec.astype(np.float32, copy=False)
        v_shared = repair_base.project_out(v, basis["Q_shared"], alpha_proj=args.alpha_proj)
        v_rand = repair_base.project_out(v, basis["Q_rand"], alpha_proj=args.alpha_proj)
        v_pca = repair_base.project_out(v, basis["Q_pca"], alpha_proj=args.alpha_proj)
        v_pca_prefill = None
        if "pca_prefill" in methods:
            v_pca_prefill = repair_base.project_out(v, basis["Q_pca_prefill"], alpha_proj=args.alpha_proj)

        norm_v = float(np.linalg.norm(v) + 1e-12)
        norm_shared = float(np.linalg.norm(v_shared) + 1e-12)
        gamma = norm_shared / norm_v
        v_shrink = (gamma * v).astype(np.float32, copy=False)

        if norm_match:
            def _scale_to(x: np.ndarray, target: float) -> np.ndarray:
                nx = float(np.linalg.norm(x) + 1e-12)
                return (x * (target / nx)).astype(np.float32, copy=False)
            v_rand = _scale_to(v_rand, norm_shared)
            v_pca = _scale_to(v_pca, norm_shared)
            if v_pca_prefill is not None:
                v_pca_prefill = _scale_to(v_pca_prefill, norm_shared)

        repaired = {"orig": v, "shared": v_shared, "rand": v_rand, "pca": v_pca, "shrink": v_shrink}
        if v_pca_prefill is not None:
            repaired["pca_prefill"] = v_pca_prefill

        per_method_template_delta = {m: [] for m in methods}
        per_method_template_acc = {m: [] for m in methods}
        per_method_task_template_delta = {m: {t: [] for t in tasks_eval} for m in methods}

        for tseed in template_seeds:
            eval_by = eval_sets[tseed]
            base_by_task = base_acc[tseed]
            for method in methods:
                deltas = []
                accs = []
                for task in tasks_eval:
                    acc_m = repair_base.eval_decode_steering(
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
                        v_np=repaired[method],
                        alpha=sv.alpha,
                        layer_idx=sv.layer,
                        staged=staged,
                        sample_seed=(args.sample_seed if args.decoding == "sample" else None),
                    )
                    delta = float(acc_m - base_by_task[task])
                    deltas.append(delta)
                    accs.append(float(acc_m))
                    per_method_task_template_delta[method][task].append(delta)
                per_method_template_delta[method].append(float(np.mean(deltas)) if deltas else float("nan"))
                per_method_template_acc[method].append(float(np.mean(accs)) if accs else float("nan"))

        summary = {
            method: _summarize_template_values(
                per_method_template_delta[method],
                per_method_template_acc[method],
                positive_threshold=args.positive_threshold,
            )
            for method in methods
        }

        results["vectors"][sv.name] = {
            "concept": sv.concept,
            "layer": sv.layer,
            "alpha": sv.alpha,
            "template_tag": sv.template_tag,
            "norms": {
                "norm_v": norm_v,
                "norm_shared": norm_shared,
                "gamma_shrink": gamma,
            },
            "template_seeds": template_seeds,
            "per_method_template_delta": per_method_template_delta,
            "per_method_template_acc": per_method_template_acc,
            "per_method_task_template_delta": per_method_task_template_delta,
            "summary": summary,
        }
        completed_since_save += 1
        if args.save_every_vectors > 0 and completed_since_save >= int(args.save_every_vectors):
            _save_checkpoint(
                results,
                checkpoint_json=checkpoint_json,
                out_md=out_md,
                positive_threshold=args.positive_threshold,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
                total_vectors=len(vecs),
            )
            completed_since_save = 0

    results["aggregate"] = _aggregate_paired_report(
        results,
        positive_threshold=args.positive_threshold,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    results["checkpoint"] = {
        "completed_vectors": int(len(results.get("vectors", {}))),
        "total_vectors": int(len(vecs)),
        "is_complete": True,
    }
    return results


def _analyze_existing(args: argparse.Namespace) -> Dict[str, Any]:
    with open(os.path.expanduser(args.repair_json), "r", encoding="utf-8") as f:
        results = json.load(f)

    vectors = results.get("vectors", {})
    if not isinstance(vectors, dict) or not vectors:
        raise RuntimeError("repair_json has no vectors")

    for vn, payload in vectors.items():
        summary = payload.get("summary", {})
        template_delta = payload.get("per_method_template_delta", {})
        template_acc = payload.get("per_method_template_acc", {})
        for method, deltas in template_delta.items():
            if method not in summary or "positive_rate" not in summary[method]:
                summary[method] = _summarize_template_values(
                    deltas,
                    template_acc.get(method, []),
                    positive_threshold=args.positive_threshold,
                )
        payload["summary"] = summary

    results["config"] = dict(results.get("config", {}))
    results["config"]["positive_threshold"] = float(args.positive_threshold)
    results["aggregate"] = _aggregate_paired_report(
        results,
        positive_threshold=args.positive_threshold,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    return results


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repair_json", type=str, default="", help="If set, analyze an existing repair-controls JSON instead of re-running.")
    ap.add_argument("--out_json", type=str, required=True)
    ap.add_argument("--out_md", type=str, default="")
    ap.add_argument("--resume_json", type=str, default="", help="Checkpoint path for direct runs; defaults to --out_json.")
    ap.add_argument("--save_every_vectors", type=int, default=1)

    ap.add_argument("--positive_threshold", type=float, default=0.0)
    ap.add_argument("--bootstrap_samples", type=int, default=10000)
    ap.add_argument("--bootstrap_seed", type=int, default=123)

    ap.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    ap.add_argument("--device", type=str, default="cuda" if repair_base.torch.cuda.is_available() else "cpu")
    ap.add_argument("--model_dtype", type=str, default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--vectors_manifest", type=str, default="")
    ap.add_argument("--max_vectors", type=int, default=0)
    ap.add_argument("--filter_regex", type=str, default="")
    ap.add_argument("--tasks_eval", type=str, default="commonsenseqa,arc_challenge,openbookqa,qasc,logiqa")
    ap.add_argument("--n_eval", type=int, default=128)
    ap.add_argument("--tasks_subspace", type=str, default="gsm8k,commonsenseqa,strategyqa,aqua,arc_challenge,openbookqa,qasc,logiqa")
    ap.add_argument("--n_subspace", type=int, default=128)
    ap.add_argument("--template_randomization", type=int, default=1, choices=[0, 1])
    ap.add_argument("--shuffle_choices", type=int, default=1, choices=[0, 1])
    ap.add_argument("--answer_prefix", type=str, default="\nFinal answer:")
    ap.add_argument("--add_answer_prefix", type=int, default=1, choices=[0, 1])
    ap.add_argument("--template_seeds", type=str, default="1234,2345,3456,4567,5678")
    ap.add_argument("--subspace_template_seed", type=int, default=None)
    ap.add_argument("--subspace_shuffle_choices", type=int, default=-1, choices=[-1, 0, 1])
    ap.add_argument("--decoding", type=str, default="greedy", choices=["greedy", "sample"])
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--reasoning_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_prompt_len", type=int, default=512)
    ap.add_argument("--sample_seed", type=int, default=12345)
    ap.add_argument("--staged", type=int, default=1, choices=[0, 1])
    ap.add_argument("--alpha_proj", type=float, default=1.0)
    ap.add_argument("--norm_match", type=int, default=1, choices=[0, 1])
    ap.add_argument("--basis_layers", type=str, default="auto")
    ap.add_argument("--shared_basis_npy_pattern", type=str, default="")
    ap.add_argument("--pca_basis_npy_pattern", type=str, default="")
    ap.add_argument("--include_pca_prefill", type=int, default=0, choices=[0, 1])
    ap.add_argument("--calib_decode_max_new_tokens", type=int, default=-1)
    ap.add_argument("--per_task_max_states", type=int, default=20000)
    ap.add_argument("--pca_var", type=float, default=0.95)
    ap.add_argument("--pca_max_dim", type=int, default=4096)
    ap.add_argument("--pca_max_rows", type=int, default=200000)
    ap.add_argument("--tau", type=float, default=0.001)
    ap.add_argument("--m_shared", type=str, default="all")
    ap.add_argument("--shared_dim", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    return ap


def main() -> None:
    ap = build_argparser()
    args = ap.parse_args()
    out_md = args.out_md or _derive_out_md(args.out_json)

    if args.repair_json:
        report = _analyze_existing(args)
    else:
        if not args.vectors_manifest:
            raise ValueError("--vectors_manifest is required when --repair_json is not provided")
        report = _run_direct_experiment(args)

    os.makedirs(os.path.dirname(os.path.expanduser(args.out_json)) or ".", exist_ok=True)
    with open(os.path.expanduser(args.out_json), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    _write_markdown(report, os.path.expanduser(out_md))
    print(f"[Saved] {args.out_json}")
    print(f"[Saved] {out_md}")


if __name__ == "__main__":
    main()
