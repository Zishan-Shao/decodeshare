#!/usr/bin/env python3
"""
Summarize results from disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py

Key upgrades vs the older summarizer:
  - Recursively scans results_dir (so it works with src/results/disturb_cot/**).
  - Summarizes multiple models cleanly (overview + per-model sections).
  - Treats EACH json as a run (so you don't silently collapse different configs).
  
  Most common:

python summarize_disturb_cot_results.py \
  --results_dir ../src/results/disturb_cot_reason \
  --recursive \
  --pattern "*.json"


If you only want the loto8 runs:

python summarize_disturb_cot_results.py \
  --results_dir src/results/disturb_cot \
  --recursive \
  --pattern "*.json" \
  --contains "loto8"
  
  
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import argparse
import math


# -----------------------------
# Formatting helpers
# -----------------------------
def fmt_acc(acc: float, lo: float, hi: float) -> str:
    """Format accuracy with confidence interval."""
    return f"{acc*100:.1f} [{lo*100:.1f}, {hi*100:.1f}]"


def fmt_pvalue(p: float) -> str:
    """Format p-value."""
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "N/A"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def mean(xs: List[float]) -> Optional[float]:
    xs2 = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs2:
        return None
    return sum(xs2) / len(xs2)


# -----------------------------
# IO
# -----------------------------
def load_json_file(filepath: str) -> Optional[Dict[str, Any]]:
    """Load and parse a JSON file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load {filepath}: {e}")
        return None


def default_results_dir() -> str:
    """
    Try a few sensible defaults:
      - src/results/disturb_cot   (common repo layout)
      - results/disturb_cot       (older layout)
    """
    candidates = [
        Path("src/results/disturb_cot"),
        Path("results/disturb_cot"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    # fallback
    return "src/results/disturb_cot"


def find_json_files(results_dir: Path, pattern: str, recursive: bool) -> List[Path]:
    if recursive:
        return sorted(results_dir.rglob(pattern))
    return sorted(results_dir.glob(pattern))


# -----------------------------
# Model naming
# -----------------------------
def normalize_model_name(model_str: str) -> str:
    """
    Convert HF repo id into a short readable name.
      e.g. "meta-llama/Llama-2-7b-chat-hf" -> "Llama-2-7b-chat-hf"
           "Qwen/Qwen2.5-7B-Instruct"     -> "Qwen2.5-7B-Instruct"
    """
    if not model_str:
        return "unknown"
    s = str(model_str).strip()
    if "/" in s:
        s = s.split("/")[-1]
    return s


def build_run_signature(cfg: Dict[str, Any]) -> str:
    """
    A compact identifier to distinguish runs of the same model.
    Keep it short but informative.
    """
    layer = (cfg.get("layer_indices") or ["?"])[0]
    tau = cfg.get("tau", "?")
    m = cfg.get("m_shared", "?")
    tr = int(bool(cfg.get("template_randomization", 0)))
    sc = int(bool(cfg.get("shuffle_choices", 0)))
    rand_type = cfg.get("rand_type", "?")
    dtype = cfg.get("model_dtype", "?")
    mode = cfg.get("mode", "?")
    loto_eval_mode = cfg.get("loto_eval_mode", "")
    suffix = f",loto_eval={loto_eval_mode}" if loto_eval_mode else ""
    return f"{mode}{suffix} | layer={layer} | tau={tau} | m={m} | tr={tr} sc={sc} | rand={rand_type} | dtype={dtype}"


# -----------------------------
# Markdown tables
# -----------------------------
def render_table(header: List[str], rows: List[List[str]]) -> str:
    """Render a markdown table."""
    if not rows:
        return "No data available.\n"

    cols = list(zip(*([header] + rows)))
    widths = [max(len(str(x)) for x in col) for col in cols]

    def fmt_row(r):
        return "| " + " | ".join(str(x).ljust(w) for x, w in zip(r, widths)) + " |"

    lines = [fmt_row(header), "|-" + "-|-".join("-" * w for w in widths) + "-|"]
    for r in rows:
        lines.append(fmt_row(r))

    return "\n".join(lines) + "\n"


# -----------------------------
# Extract per-run detailed rows
# -----------------------------
def summarize_loto_results(results: Dict[str, Any], decoding: str = "greedy") -> List[List[str]]:
    """
    Extract LOTO held-out results into table rows.
    Assumes fold keys are holdout task names, and we evaluate heldout dataset inside fold["by_dataset"][holdout].
    """
    rows: List[List[str]] = []
    folds = results.get("folds", {})
    if not isinstance(folds, dict) or not folds:
        return rows

    for holdout, fold in folds.items():
        block = (fold.get("by_dataset", {}) or {}).get(holdout, None)
        if block is None:
            continue

        runs = block.get("runs", {}) or {}
        run_key = f"{decoding}/baseline"
        if run_key not in runs:
            continue

        b = runs.get(f"{decoding}/baseline", {})
        s = runs.get(f"{decoding}/shared_full", {})
        r = runs.get(f"{decoding}/rand_full", {})

        paired_tests = (block.get("paired_tests", {}) or {}).get(decoding, {}) or {}
        stat = paired_tests.get("shared_full_vs_baseline", {}) or {}

        rows.append([
            str(holdout),
            str(block.get("n", "?")),
            fmt_acc(b.get("accuracy", 0), b.get("ci_low", 0), b.get("ci_high", 0)),
            fmt_acc(s.get("accuracy", 0), s.get("ci_low", 0), s.get("ci_high", 0)) if s else "N/A",
            fmt_acc(r.get("accuracy", 0), r.get("ci_low", 0), r.get("ci_high", 0)) if r else "N/A",
            f"{safe_float(stat.get('mean_diff', 0), 0)*100:+.1f} "
            f"[{safe_float(stat.get('ci_low', 0), 0)*100:+.1f}, {safe_float(stat.get('ci_high', 0), 0)*100:+.1f}]"
            if stat else "N/A",
            fmt_pvalue(stat.get("p_value", None)) if stat else "N/A",
        ])

    return rows


def summarize_all_tasks_results(results: Dict[str, Any], decoding: str = "greedy") -> List[List[str]]:
    """Extract all-tasks results into table rows."""
    rows: List[List[str]] = []
    fold = results.get("all_tasks", None)
    if not isinstance(fold, dict):
        return rows

    by_dataset = fold.get("by_dataset", {}) or {}
    for task_name, block in sorted(by_dataset.items()):
        runs = block.get("runs", {}) or {}
        run_key = f"{decoding}/baseline"
        if run_key not in runs:
            continue

        b = runs.get(f"{decoding}/baseline", {})
        s = runs.get(f"{decoding}/shared_full", {})
        r = runs.get(f"{decoding}/rand_full", {})

        paired_tests = (block.get("paired_tests", {}) or {}).get(decoding, {}) or {}
        stat = paired_tests.get("shared_full_vs_baseline", {}) or {}

        rows.append([
            str(task_name),
            str(block.get("n", "?")),
            fmt_acc(b.get("accuracy", 0), b.get("ci_low", 0), b.get("ci_high", 0)),
            fmt_acc(s.get("accuracy", 0), s.get("ci_low", 0), s.get("ci_high", 0)) if s else "N/A",
            fmt_acc(r.get("accuracy", 0), r.get("ci_low", 0), r.get("ci_high", 0)) if r else "N/A",
            f"{safe_float(stat.get('mean_diff', 0), 0)*100:+.1f} "
            f"[{safe_float(stat.get('ci_low', 0), 0)*100:+.1f}, {safe_float(stat.get('ci_high', 0), 0)*100:+.1f}]"
            if stat else "N/A",
            fmt_pvalue(stat.get("p_value", None)) if stat else "N/A",
        ])

    return rows


# -----------------------------
# Compute overview stats (for cross-model comparison)
# -----------------------------
def overview_for_loto(results: Dict[str, Any], decoding: str = "greedy") -> Optional[Dict[str, Any]]:
    folds = results.get("folds", {})
    if not isinstance(folds, dict) or not folds:
        return None

    baseline_acc, shared_acc, rand_acc = [], [], []
    diffs, pvals = [], []
    cross_dims, shared_ks = [], []
    er_shared, er_rand = [], []

    for holdout, fold in folds.items():
        block = (fold.get("by_dataset", {}) or {}).get(holdout, None)
        if block is None:
            continue
        runs = block.get("runs", {}) or {}
        if f"{decoding}/baseline" not in runs:
            continue

        b = runs.get(f"{decoding}/baseline", {})
        s = runs.get(f"{decoding}/shared_full", {})
        r = runs.get(f"{decoding}/rand_full", {})
        baseline_acc.append(safe_float(b.get("accuracy", None)))
        shared_acc.append(safe_float(s.get("accuracy", None)))
        rand_acc.append(safe_float(r.get("accuracy", None)))

        stat = ((block.get("paired_tests", {}) or {}).get(decoding, {}) or {}).get("shared_full_vs_baseline", {}) or {}
        diffs.append(safe_float(stat.get("mean_diff", None)))
        pvals.append(safe_float(stat.get("p_value", None)))

        basis = fold.get("basis", {}) or {}
        cross_dims.append(safe_float(basis.get("cross_dim", None)))
        shared_ks.append(safe_float(basis.get("shared_k", None)))

        sanity = basis.get("sanity", {}) or {}
        ers = (sanity.get("energy_ratio_shared", {}) or {}).get("mean", None)
        err = (sanity.get("energy_ratio_rand", {}) or {}).get("mean", None)
        er_shared.append(safe_float(ers, None))
        er_rand.append(safe_float(err, None))

    if not baseline_acc:
        return None

    sig = 0
    total_p = 0
    for p in pvals:
        if p is None or (isinstance(p, float) and math.isnan(p)):
            continue
        total_p += 1
        if p < 0.05:
            sig += 1

    return {
        "n_holdouts": len(baseline_acc),
        "baseline_mean": mean(baseline_acc),
        "shared_mean": mean(shared_acc),
        "rand_mean": mean(rand_acc),
        "diff_mean": mean(diffs),
        "sig": sig,
        "sig_denom": total_p,
        "cross_dim_mean": mean(cross_dims),
        "shared_k_mean": mean(shared_ks),
        "er_shared_mean": mean(er_shared),
        "er_rand_mean": mean(er_rand),
    }


def overview_for_all(results: Dict[str, Any], decoding: str = "greedy") -> Optional[Dict[str, Any]]:
    fold = results.get("all_tasks", None)
    if not isinstance(fold, dict):
        return None

    by_dataset = fold.get("by_dataset", {}) or {}
    baseline_acc, shared_acc, rand_acc = [], [], []
    diffs, pvals = [], []

    for task, block in by_dataset.items():
        runs = block.get("runs", {}) or {}
        if f"{decoding}/baseline" not in runs:
            continue

        b = runs.get(f"{decoding}/baseline", {})
        s = runs.get(f"{decoding}/shared_full", {})
        r = runs.get(f"{decoding}/rand_full", {})

        baseline_acc.append(safe_float(b.get("accuracy", None)))
        shared_acc.append(safe_float(s.get("accuracy", None)))
        rand_acc.append(safe_float(r.get("accuracy", None)))

        stat = ((block.get("paired_tests", {}) or {}).get(decoding, {}) or {}).get("shared_full_vs_baseline", {}) or {}
        diffs.append(safe_float(stat.get("mean_diff", None)))
        pvals.append(safe_float(stat.get("p_value", None)))

    if not baseline_acc:
        return None

    sig = 0
    total_p = 0
    for p in pvals:
        if p is None or (isinstance(p, float) and math.isnan(p)):
            continue
        total_p += 1
        if p < 0.05:
            sig += 1

    # basis stats (single fold)
    basis = fold.get("basis", {}) or {}
    sanity = basis.get("sanity", {}) or {}
    return {
        "n_tasks": len(baseline_acc),
        "baseline_mean": mean(baseline_acc),
        "shared_mean": mean(shared_acc),
        "rand_mean": mean(rand_acc),
        "diff_mean": mean(diffs),
        "sig": sig,
        "sig_denom": total_p,
        "cross_dim": safe_float(basis.get("cross_dim", None)),
        "shared_k": safe_float(basis.get("shared_k", None)),
        "er_shared_mean": safe_float((sanity.get("energy_ratio_shared", {}) or {}).get("mean", None)),
        "er_rand_mean": safe_float((sanity.get("energy_ratio_rand", {}) or {}).get("mean", None)),
    }


# -----------------------------
# Markdown generation
# -----------------------------
def generate_summary_markdown(experiments: List[Dict[str, Any]], output_file: str, decoding: str = "greedy") -> None:
    md: List[str] = []
    md.append("# Disturb CoT Results Summary\n")
    md.append(f"Generated from {len(experiments)} JSON file(s)\n")
    md.append(f"- Decoding summarized: **{decoding}**\n\n")

    # -----------------------------
    # Overview tables (cross-model)
    # -----------------------------
    loto_rows: List[List[str]] = []
    all_rows: List[List[str]] = []

    for exp in experiments:
        res = exp["results"]
        cfg = exp["config"]
        model_short = exp["model_short"]
        mode = exp["mode"]
        sig = exp["signature"]
        filename = exp["filename"]

        if mode == "loto":
            ov = overview_for_loto(res, decoding=decoding)
            if ov is not None:
                loto_rows.append([
                    model_short,
                    sig,
                    str(ov["n_holdouts"]),
                    f"{(ov['baseline_mean'] or 0)*100:.1f}",
                    f"{(ov['shared_mean'] or 0)*100:.1f}",
                    f"{(ov['rand_mean'] or 0)*100:.1f}",
                    f"{(ov['diff_mean'] or 0)*100:+.2f}",
                    f"{ov['sig']}/{ov['sig_denom']}",
                    f"{ov['cross_dim_mean']:.1f}" if ov["cross_dim_mean"] is not None else "N/A",
                    f"{ov['shared_k_mean']:.1f}" if ov["shared_k_mean"] is not None else "N/A",
                    f"{ov['er_shared_mean']:.4f}" if ov["er_shared_mean"] is not None else "N/A",
                    f"{ov['er_rand_mean']:.4f}" if ov["er_rand_mean"] is not None else "N/A",
                    filename,
                ])
        elif mode == "all":
            ov = overview_for_all(res, decoding=decoding)
            if ov is not None:
                all_rows.append([
                    model_short,
                    sig,
                    str(ov["n_tasks"]),
                    f"{(ov['baseline_mean'] or 0)*100:.1f}",
                    f"{(ov['shared_mean'] or 0)*100:.1f}",
                    f"{(ov['rand_mean'] or 0)*100:.1f}",
                    f"{(ov['diff_mean'] or 0)*100:+.2f}",
                    f"{ov['sig']}/{ov['sig_denom']}",
                    f"{ov['cross_dim']:.0f}" if ov["cross_dim"] is not None else "N/A",
                    f"{ov['shared_k']:.0f}" if ov["shared_k"] is not None else "N/A",
                    f"{ov['er_shared_mean']:.4f}" if ov["er_shared_mean"] is not None else "N/A",
                    f"{ov['er_rand_mean']:.4f}" if ov["er_rand_mean"] is not None else "N/A",
                    filename,
                ])

    if loto_rows:
        md.append("## Overview: LOTO runs\n\n")
        md.append(render_table(
            header=[
                "Model", "Run signature", "#holdouts",
                "Baseline(%)", "Shared(%)", "Rand(%)", "Δ mean(pp)",
                "#sig(p<0.05)", "cross_dim", "shared_k",
                "ER(shared)", "ER(rand)", "File"
            ],
            rows=sorted(loto_rows, key=lambda r: (r[0], r[-1]))
        ))
        md.append("\n")

    if all_rows:
        md.append("## Overview: ALL runs\n\n")
        md.append(render_table(
            header=[
                "Model", "Run signature", "#tasks",
                "Baseline(%)", "Shared(%)", "Rand(%)", "Δ mean(pp)",
                "#sig(p<0.05)", "cross_dim", "shared_k",
                "ER(shared)", "ER(rand)", "File"
            ],
            rows=sorted(all_rows, key=lambda r: (r[0], r[-1]))
        ))
        md.append("\n")

    # -----------------------------
    # Detailed sections grouped by model
    # -----------------------------
    by_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for exp in experiments:
        by_model[exp["model_short"]].append(exp)

    for model_short in sorted(by_model.keys()):
        md.append(f"\n# Model: {model_short}\n\n")
        runs = sorted(by_model[model_short], key=lambda x: (x["mode"], x["filename"]))

        for exp in runs:
            res = exp["results"]
            cfg = exp["config"]
            mode = exp["mode"]
            sig = exp["signature"]

            md.append(f"## Run: `{exp['filename']}`\n\n")
            md.append(f"- **Model (raw)**: `{cfg.get('model', 'unknown')}`\n")
            md.append(f"- **Signature**: {sig}\n")
            md.append(f"- **Tasks**: {', '.join(cfg.get('tasks', []))}\n")
            md.append(f"- **Layer**: {((cfg.get('layer_indices') or ['?'])[0])}\n")
            md.append(f"- **Sharedness**: pca_var={cfg.get('pca_var','?')}, tau={cfg.get('tau','?')}, m_shared={cfg.get('m_shared','?')}\n")
            md.append(f"- **Calibration**: calib_decode_max_new_tokens={cfg.get('calib_decode_max_new_tokens','?')}, per_task_max_states={cfg.get('per_task_max_states','?')}\n")
            md.append(f"- **Eval**: n_subspace={cfg.get('n_subspace','?')}, n_eval={cfg.get('n_eval','?')}, max_new_tokens={cfg.get('max_new_tokens','?')}\n")
            md.append("\n")

            if mode == "loto" and "folds" in res:
                md.append(f"### LOTO Held-out Performance ({decoding})\n\n")
                header = ["Held-out", "n", "Baseline", "Shared(full)", "Rand(full)", "Δ(shared-baseline)", "p(shared-baseline)"]
                rows = summarize_loto_results(res, decoding=decoding)
                md.append(render_table(header, rows))
                md.append("\n")

                # Basis statistics
                md.append("### Basis Statistics\n\n")
                md.append("| Held-out | Cross-dim | Shared-k | ER(shared) mean | ER(rand) mean |\n")
                md.append("|----------|-----------|----------|-----------------|---------------|\n")
                for holdout, fold in (res.get("folds", {}) or {}).items():
                    basis = fold.get("basis", {}) or {}
                    sanity = basis.get("sanity", {}) or {}
                    er_s = (sanity.get("energy_ratio_shared", {}) or {}).get("mean", None)
                    er_r = (sanity.get("energy_ratio_rand", {}) or {}).get("mean", None)
                    md.append(
                        f"| {holdout} | {basis.get('cross_dim', 'N/A')} | {basis.get('shared_k', 'N/A')} | "
                        f"{(safe_float(er_s, None) if er_s is not None else None) if er_s is not None else ''}"
                    )
                    # keep stable formatting
                    er_s_val = safe_float(er_s, None)
                    er_r_val = safe_float(er_r, None)
                    md.pop()  # remove partial line above
                    md.append(
                        f"| {holdout} | {basis.get('cross_dim', 'N/A')} | {basis.get('shared_k', 'N/A')} | "
                        f"{(f'{er_s_val:.4f}' if er_s_val is not None else 'N/A')} | "
                        f"{(f'{er_r_val:.4f}' if er_r_val is not None else 'N/A')} |\n"
                    )
                md.append("\n")

            elif mode == "all" and "all_tasks" in res:
                md.append(f"### All-Tasks Performance ({decoding})\n\n")
                header = ["Task", "n", "Baseline", "Shared(full)", "Rand(full)", "Δ(shared-baseline)", "p(shared-baseline)"]
                rows = summarize_all_tasks_results(res, decoding=decoding)
                md.append(render_table(header, rows))
                md.append("\n")

                # Basis statistics (single fold)
                fold = res.get("all_tasks", {}) or {}
                basis = fold.get("basis", {}) or {}
                sanity = basis.get("sanity", {}) or {}
                er_s_val = safe_float((sanity.get("energy_ratio_shared", {}) or {}).get("mean", None), None)
                er_r_val = safe_float((sanity.get("energy_ratio_rand", {}) or {}).get("mean", None), None)

                md.append("### Basis Statistics\n\n")
                md.append("| cross_dim | shared_k | ER(shared) mean | ER(rand) mean |\n")
                md.append("|----------:|---------:|----------------:|--------------:|\n")
                md.append(
                    f"| {basis.get('cross_dim','N/A')} | {basis.get('shared_k','N/A')} | "
                    f"{(f'{er_s_val:.4f}' if er_s_val is not None else 'N/A')} | "
                    f"{(f'{er_r_val:.4f}' if er_r_val is not None else 'N/A')} |\n"
                )
                md.append("\n")

    # Write
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("".join(md))

    print(f"Summary written to: {output_file}")


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Summarize disturb CoT experiment results (multi-model)")
    parser.add_argument(
        "--results_dir",
        type=str,
        default=default_results_dir(),
        help="Directory containing JSON result files (supports recursive scan)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output markdown file path (default: <results_dir>/COMPREHENSIVE_SUMMARY.md)"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.json",
        help="Glob pattern for JSON files (default: *.json)"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan results_dir (recommended for src/results/disturb_cot)"
    )
    parser.add_argument(
        "--no_recursive",
        action="store_true",
        help="Disable recursive scan"
    )
    parser.add_argument(
        "--contains",
        type=str,
        default="",
        help="Only include files whose name contains this substring (case-insensitive). Example: 'loto8'"
    )
    parser.add_argument(
        "--decoding",
        type=str,
        default="greedy",
        choices=["greedy", "sample"],
        help="Which decoding branch to summarize (default: greedy)"
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory does not exist: {results_dir}")
        return

    recursive = True
    if args.no_recursive:
        recursive = False
    if args.recursive:
        recursive = True

    json_files = find_json_files(results_dir, args.pattern, recursive=recursive)

    if args.contains.strip():
        key = args.contains.strip().lower()
        json_files = [p for p in json_files if key in p.name.lower()]

    if not json_files:
        print(f"Warning: No JSON files found in {results_dir} (pattern={args.pattern}, recursive={recursive})")
        return

    print(f"Found {len(json_files)} JSON file(s) in {results_dir} (pattern={args.pattern}, recursive={recursive})")

    experiments: List[Dict[str, Any]] = []
    for p in json_files:
        data = load_json_file(str(p))
        if not data or not isinstance(data, dict):
            continue
        cfg = data.get("config", {}) or {}
        mode = cfg.get("mode", "unknown")
        model_raw = cfg.get("model", "")
        model_short = normalize_model_name(model_raw)
        signature = build_run_signature(cfg)

        # Basic sanity filter: keep only files that look like your script's output
        looks_like_disturb = ("folds" in data) or ("all_tasks" in data)
        if not looks_like_disturb:
            # skip random jsons in directory
            continue

        experiments.append({
            "filename": p.name,
            "filepath": str(p),
            "results": data,
            "config": cfg,
            "mode": mode,
            "model_raw": model_raw,
            "model_short": model_short,
            "signature": signature,
        })

    if not experiments:
        print("Error: No valid disturb_cot JSON results loaded (did you point to the right folder/pattern?)")
        return

    output_file = args.output
    if output_file is None:
        output_file = str(results_dir / "COMPREHENSIVE_SUMMARY.md")

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    generate_summary_markdown(experiments, output_file=output_file, decoding=args.decoding)

    print(f"\nDone! Processed {len(experiments)} result file(s)")


if __name__ == "__main__":
    main()
















# #!/usr/bin/env python3
# """
# Summarize results from disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py

# This script processes all JSON result files in results/disturb_cot/ and generates
# comprehensive summary reports.

# # Basic usage (default: results/disturb_cot/)
# python summarize_disturb_cot_results.py

# # Specify custom directory
# python summarize_disturb_cot_results.py --results_dir ../src/results/disturb_cot

# # Custom output file
# python summarize_disturb_cot_results.py --output src/results/disturb_cot/COMPREHENSIVE_SUMMARY.md

# # Custom file pattern
# python summarize_disturb_cot_results.py --pattern "*loto*.json"

# # Filter for no_aqua files only
# python summarize_disturb_cot_results.py --no_aqua

# """

# import os
# import json
# import glob
# from pathlib import Path
# from typing import Dict, List, Any, Optional
# from collections import defaultdict
# import argparse


# def fmt_acc(acc: float, lo: float, hi: float) -> str:
#     """Format accuracy with confidence interval."""
#     return f"{acc*100:.1f} [{lo*100:.1f}, {hi*100:.1f}]"


# def fmt_pvalue(p: float) -> str:
#     """Format p-value."""
#     if p < 0.001:
#         return "<0.001"
#     return f"{p:.3f}"


# def load_json_file(filepath: str) -> Optional[Dict[str, Any]]:
#     """Load and parse a JSON file."""
#     try:
#         with open(filepath, "r", encoding="utf-8") as f:
#             return json.load(f)
#     except Exception as e:
#         print(f"Warning: Failed to load {filepath}: {e}")
#         return None


# def extract_model_name_from_path(filepath: str) -> str:
#     """Extract model name from file path."""
#     basename = os.path.basename(filepath)
#     # Try to extract model name from filename patterns
#     if "llama2" in basename.lower() or "llama-2" in basename.lower():
#         return "Llama-2-7b-chat-hf"
#     elif "qwen" in basename.lower():
#         if "instruct" in basename.lower():
#             return "Qwen2.5-7B-Instruct"
#         return "Qwen2.5-7B"
#     elif "gemma" in basename.lower():
#         if "12b" in basename.lower():
#             return "gemma-3-12b-it"
#         return "gemma-2-2b-it"
#     return "unknown"


# def summarize_loto_results(results: Dict[str, Any], decoding: str = "greedy") -> List[List[str]]:
#     """Extract LOTO held-out results into table rows."""
#     rows = []
#     if "folds" not in results:
#         return rows
    
#     for holdout, fold in results["folds"].items():
#         block = fold.get("by_dataset", {}).get(holdout, None)
#         if block is None:
#             continue
        
#         run_key = f"{decoding}/baseline"
#         if run_key not in block.get("runs", {}):
#             continue
        
#         b = block["runs"][f"{decoding}/baseline"]
#         s = block["runs"].get(f"{decoding}/shared_full", {})
#         r = block["runs"].get(f"{decoding}/rand_full", {})
        
#         paired_tests = block.get("paired_tests", {}).get(decoding, {})
#         stat = paired_tests.get("shared_full_vs_baseline", {})
        
#         rows.append([
#             holdout,
#             str(block.get("n", "?")),
#             fmt_acc(b.get("accuracy", 0), b.get("ci_low", 0), b.get("ci_high", 0)),
#             fmt_acc(s.get("accuracy", 0), s.get("ci_low", 0), s.get("ci_high", 0)) if s else "N/A",
#             fmt_acc(r.get("accuracy", 0), r.get("ci_low", 0), r.get("ci_high", 0)) if r else "N/A",
#             f"{stat.get('mean_diff', 0)*100:+.1f} [{stat.get('ci_low', 0)*100:+.1f}, {stat.get('ci_high', 0)*100:+.1f}]" if stat else "N/A",
#             fmt_pvalue(stat.get("p_value", 1.0)) if stat else "N/A",
#         ])
    
#     return rows


# def summarize_all_tasks_results(results: Dict[str, Any], decoding: str = "greedy") -> List[List[str]]:
#     """Extract all-tasks results into table rows."""
#     rows = []
#     if "all_tasks" not in results:
#         return rows
    
#     fold = results["all_tasks"]
#     by_dataset = fold.get("by_dataset", {})
    
#     for task_name, block in sorted(by_dataset.items()):
#         run_key = f"{decoding}/baseline"
#         if run_key not in block.get("runs", {}):
#             continue
        
#         b = block["runs"][f"{decoding}/baseline"]
#         s = block["runs"].get(f"{decoding}/shared_full", {})
#         r = block["runs"].get(f"{decoding}/rand_full", {})
        
#         paired_tests = block.get("paired_tests", {}).get(decoding, {})
#         stat = paired_tests.get("shared_full_vs_baseline", {})
        
#         rows.append([
#             task_name,
#             str(block.get("n", "?")),
#             fmt_acc(b.get("accuracy", 0), b.get("ci_low", 0), b.get("ci_high", 0)),
#             fmt_acc(s.get("accuracy", 0), s.get("ci_low", 0), s.get("ci_high", 0)) if s else "N/A",
#             fmt_acc(r.get("accuracy", 0), r.get("ci_low", 0), r.get("ci_high", 0)) if r else "N/A",
#             f"{stat.get('mean_diff', 0)*100:+.1f} [{stat.get('ci_low', 0)*100:+.1f}, {stat.get('ci_high', 0)*100:+.1f}]" if stat else "N/A",
#             fmt_pvalue(stat.get("p_value", 1.0)) if stat else "N/A",
#         ])
    
#     return rows


# def render_table(header: List[str], rows: List[List[str]]) -> str:
#     """Render a markdown table."""
#     if not rows:
#         return "No data available.\n"
    
#     cols = list(zip(*([header] + rows)))
#     widths = [max(len(str(x)) for x in col) for col in cols]
    
#     def fmt_row(r):
#         return "| " + " | ".join(str(x).ljust(w) for x, w in zip(r, widths)) + " |"
    
#     lines = [fmt_row(header), "|-" + "-|-".join("-"*w for w in widths) + "-|"]
#     for r in rows:
#         lines.append(fmt_row(r))
    
#     return "\n".join(lines) + "\n"


# def generate_summary_markdown(all_results: List[Dict[str, Any]], output_file: str):
#     """Generate comprehensive markdown summary from all result files."""
#     md_lines = []
#     md_lines.append("# Disturb CoT Results Summary\n")
#     md_lines.append(f"Generated from {len(all_results)} result file(s)\n")
    
#     # Group by model and mode
#     by_model_mode = defaultdict(list)
#     for result_data in all_results:
#         results = result_data["results"]
#         config = results.get("config", {})
#         model = config.get("model", "unknown")
#         mode = config.get("mode", "unknown")
#         key = f"{model}_{mode}"
#         by_model_mode[key].append(result_data)
    
#     for key, result_list in sorted(by_model_mode.items()):
#         # Use first result for config info
#         first_result = result_list[0]["results"]
#         config = first_result.get("config", {})
#         model = config.get("model", "unknown")
#         mode = config.get("mode", "unknown")
        
#         md_lines.append(f"\n## Model: {model} | Mode: {mode}\n")
#         md_lines.append(f"- **Tasks**: {', '.join(config.get('tasks', []))}\n")
#         md_lines.append(f"- **Layer**: {config.get('layer_indices', ['?'])[0]}\n")
#         md_lines.append(f"- **Template randomization**: {config.get('template_randomization', '?')}\n")
#         md_lines.append(f"- **Shuffle choices**: {config.get('shuffle_choices', '?')}\n")
#         md_lines.append(f"- **Sharedness**: tau={config.get('tau', '?')}, m_shared={config.get('m_shared', '?')}\n")
#         md_lines.append(f"- **Calibration**: max_new_tokens={config.get('calib_decode_max_new_tokens', '?')}, per_task_max_states={config.get('per_task_max_states', '?')}\n")
#         md_lines.append("")
        
#         # LOTO mode
#         if mode == "loto" and "folds" in first_result:
#             md_lines.append("### LOTO Held-out Performance (Greedy)\n")
#             header = ["Held-out", "n", "Baseline", "Shared(full)", "Rand(full)", "Δ(shared-baseline)", "p(shared-baseline)"]
#             rows = summarize_loto_results(first_result, decoding="greedy")
#             md_lines.append(render_table(header, rows))
        
#         # All-tasks mode
#         elif mode == "all" and "all_tasks" in first_result:
#             md_lines.append("### All-Tasks Performance (Greedy)\n")
#             header = ["Task", "n", "Baseline", "Shared(full)", "Rand(full)", "Δ(shared-baseline)", "p(shared-baseline)"]
#             rows = summarize_all_tasks_results(first_result, decoding="greedy")
#             md_lines.append(render_table(header, rows))
        
#         # Basis statistics
#         if mode == "loto" and "folds" in first_result:
#             md_lines.append("### Basis Statistics\n")
#             md_lines.append("| Held-out | Cross-dim | Shared-k | Energy Ratio (shared) | Energy Ratio (rand) |\n")
#             md_lines.append("|----------|-----------|----------|------------------------|---------------------|\n")
#             for holdout, fold in first_result["folds"].items():
#                 basis = fold.get("basis", {})
#                 sanity = basis.get("sanity", {})
#                 er_s = sanity.get("energy_ratio_shared", {})
#                 er_r = sanity.get("energy_ratio_rand", {})
#                 md_lines.append(
#                     f"| {holdout} | {basis.get('cross_dim', '?')} | {basis.get('shared_k', '?')} | "
#                     f"{er_s.get('mean', 0):.4f} | {er_r.get('mean', 0):.4f} |\n"
#                 )
#             md_lines.append("")
        
#         # File sources
#         md_lines.append("### Source Files\n")
#         for result_data in result_list:
#             md_lines.append(f"- `{result_data['filename']}`\n")
#         md_lines.append("")
    
#     # Write to file
#     with open(output_file, "w", encoding="utf-8") as f:
#         f.write("".join(md_lines))
    
#     print(f"Summary written to: {output_file}")


# def main():
#     parser = argparse.ArgumentParser(description="Summarize disturb CoT experiment results")
#     parser.add_argument(
#         "--results_dir",
#         type=str,
#         default="results/disturb_cot",
#         help="Directory containing JSON result files"
#     )
#     parser.add_argument(
#         "--output",
#         type=str,
#         default="results/disturb_cot/SUMMARY.md",
#         help="Output markdown file path"
#     )
#     parser.add_argument(
#         "--pattern",
#         type=str,
#         default="*_results_*.json",
#         help="Glob pattern for JSON files (default: *_results_*.json)"
#     )
#     parser.add_argument(
#         "--no_aqua",
#         action="store_true",
#         help="Filter for files containing 'no_aqua' in filename"
#     )
    
#     args = parser.parse_args()
    
#     # Find all JSON files
#     results_dir = Path(args.results_dir)
#     if not results_dir.exists():
#         print(f"Error: Results directory does not exist: {results_dir}")
#         return
    
#     json_files = list(results_dir.glob(args.pattern))
    
#     # Filter for no_aqua files if requested
#     if args.no_aqua:
#         json_files = [f for f in json_files if "no_aqua" in f.name]
#         if not json_files:
#             print(f"Warning: No JSON files found matching pattern '{args.pattern}' with 'no_aqua' in {results_dir}")
#             return
    
#     if not json_files:
#         print(f"Warning: No JSON files found matching pattern '{args.pattern}' in {results_dir}")
#         return
    
#     print(f"Found {len(json_files)} JSON file(s)")
    
#     # Load all results
#     all_results = []
#     for json_file in sorted(json_files):
#         print(f"Loading: {json_file.name}")
#         results = load_json_file(str(json_file))
#         if results:
#             all_results.append({
#                 "filename": json_file.name,
#                 "filepath": str(json_file),
#                 "results": results
#             })
    
#     if not all_results:
#         print("Error: No valid results loaded")
#         return
    
#     # Generate summary
#     output_path = Path(args.output)
#     output_path.parent.mkdir(parents=True, exist_ok=True)
#     generate_summary_markdown(all_results, str(output_path))
    
#     print(f"\nDone! Processed {len(all_results)} result file(s)")


# if __name__ == "__main__":
#     main()

