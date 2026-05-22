#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
summarize_energy_kmatch_outputs.py

Summarize JSON outputs produced by disturb_energy_matched_sharedness_kmatch.py.

Input JSON schema (expected):
{
  "config": {...},
  "sanity": {
     "overlap_shared_ctrl_struct": float,
     "overlap_shared_ctrl_energy": float,
     "energy_shared": {"mean_ratio":..., "mean_energy":...},
     "energy_ctrl_struct": {...},
     "energy_ctrl_energy": {...},
     "alpha_match": {"alpha_shared":..., "alpha_ctrl":..., ...}
  },
  "by_task": {
     "<task>": {
        "n": int,
        "runs": {
           "<condition>": {
              "accuracy": float, "ci_low": float, "ci_high": float,
              "alpha": float, "hookstats": {layer: {...}}  # optional
           }, ...
        },
        "paired": {
           "<comparison_key>": {"mean_diff": float, "ci_low": float, "ci_high": float, "p_value": float, ...}
        }
     }, ...
  }
}

Outputs:
- Markdown summary file (default)
- Optionally CSVs:
    - <prefix>_runs.csv (one row per JSON)
    - <prefix>_tasks.csv (one row per task per JSON)

Example:
  python analysis/summarize_energy_kmatch_outputs.py \
    --results_dir ./results/energy_kmatch \
    --pattern "*.json" \
    --output ./results/energy_kmatch/SUMMARY.md \
    --write_csv 1 \
    --csv_prefix ./results/energy_kmatch/summary
    
1) 最常用：汇总一个目录下所有 out_json
python analysis/summarize_energy_kmatch_outputs.py \
  --results_dir ../../outputs/02_decode_ablation/energy_kmatch_alpha_sweep \
  --pattern "*.json" \
  --output ../../outputs/02_decode_ablation/energy_kmatch_alpha_sweep/SUMMARY.md
  
2) 同时输出 CSV（方便你画图）
python analysis/summarize_energy_kmatch_outputs.py \
  --results_dir ./results/energy_kmatch \
  --pattern "*.json" \
  --output ./results/energy_kmatch/SUMMARY.md \
  --write_csv 1 \
  --csv_prefix ./results/energy_kmatch/summary

3) 只看某几个任务
python analysis/summarize_energy_kmatch_outputs.py \
  --results_dir ./results/energy_kmatch \
  --tasks commonsenseqa,strategyqa,aqua \
  --output ./results/energy_kmatch/SUMMARY_small.md

输出里你能快速读到什么

Emean(ctrl_energy)/Emean(shared)：能量匹配到底匹没匹上（≈1 越好）

Mean Δ(shared-ctrlE)：如果 shared_full 比 ctrl_energy 更差很多，说明“共享子空间”更关键；
如果差不多，说明可能只是“去掉能量”导致的一般性损伤。

sig/neg_sig(shared-ctrlE)：按 task 统计显著性（p<0.05），可以直接当 reviewer-friendly 证据。    
    


"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

# -----------------------------
# Basic helpers
# -----------------------------
def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[Warn] failed to load {path}: {e}")
        return None

def safe_get(d: Dict[str, Any], *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def pct(x: Optional[float]) -> str:
    if x is None:
        return "N/A"
    return f"{x*100:.1f}"

def pp(x: Optional[float]) -> str:
    """x is mean_diff in probability space -> percentage points"""
    if x is None:
        return "N/A"
    return f"{x*100:+.1f}pp"

def sci(x: Optional[float]) -> str:
    if x is None:
        return "N/A"
    return f"{x:.3e}"

def f4(x: Optional[float]) -> str:
    if x is None:
        return "N/A"
    return f"{x:.4f}"

def f3(x: Optional[float]) -> str:
    if x is None:
        return "N/A"
    return f"{x:.3f}"

def render_md_table(header: List[str], rows: List[List[str]]) -> str:
    if not rows:
        return "No data.\n"
    cols = list(zip(*([header] + rows)))
    widths = [max(len(str(v)) for v in col) for col in cols]

    def row(r):
        return "| " + " | ".join(str(v).ljust(w) for v, w in zip(r, widths)) + " |"

    sep = "|-" + "-|-".join("-" * w for w in widths) + "-|"
    out = [row(header), sep]
    out.extend(row(r) for r in rows)
    return "\n".join(out) + "\n"

def short_model_name(model_raw: str) -> str:
    s = (model_raw or "").lower()
    if "llama" in s and "2" in s and "7b" in s:
        return "Llama-2-7b-chat-hf"
    if "qwen" in s and "2.5" in s and "7b" in s:
        return "Qwen2.5-7B-Instruct" if "instruct" in s else "Qwen2.5-7B"
    if "gemma" in s and "12b" in s:
        return "gemma-3-12b-it"
    return model_raw or "unknown"

def cond_acc(run_task: Dict[str, Any], cond: str) -> Optional[float]:
    val = safe_get(run_task, "runs", cond, "accuracy", default=None)
    if val is None:
        val = safe_get(run_task, "runs", cond, "acc", default=None)
    return val

def cond_ci(run_task: Dict[str, Any], cond: str) -> Tuple[Optional[float], Optional[float]]:
    lo = safe_get(run_task, "runs", cond, "ci_low", default=None)
    hi = safe_get(run_task, "runs", cond, "ci_high", default=None)
    return lo, hi

def paired_stat(run_task: Dict[str, Any], key: str) -> Optional[Dict[str, Any]]:
    return safe_get(run_task, "paired", key, default=None)

def is_sig(stat: Optional[Dict[str, Any]], p_th: float) -> bool:
    if not stat:
        return False
    p = stat.get("p_value", None)
    return (p is not None) and (p < p_th)

def mean(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    return sum(xs) / len(xs)

# -----------------------------
# Extract & summarize one JSON
# -----------------------------
DEFAULT_CONDS = ["baseline", "shared_alpha", "ctrl_alpha", "shared_full", "ctrl_struct", "ctrl_energy"]
DEFAULT_PAIRED_KEYS = [
    "shared_alpha_vs_base",
    "ctrl_alpha_vs_base",
    "shared_alpha_vs_ctrl_alpha",
    "shared_vs_base",
    "ctrl_struct_vs_base",
    "ctrl_energy_vs_base",
    "shared_vs_ctrl_struct",
    "shared_vs_ctrl_energy",
]

def summarize_one_run(
    res: Dict[str, Any],
    *,
    filename: str,
    p_th: float,
    tasks_filter: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Returns (run_row, task_rows)."""
    cfg = res.get("config", {})
    sanity = res.get("sanity", {})
    by_task = res.get("by_task", {})

    model_raw = cfg.get("model", "unknown")
    model = short_model_name(model_raw)
    layer_indices = cfg.get("layer_indices", [])
    layer = layer_indices[0] if isinstance(layer_indices, list) and layer_indices else cfg.get("layer", None)

    cross_dim = cfg.get("cross_dim", None)
    k_shared = cfg.get("k_shared", None)
    k_c = cfg.get("k_c", None)

    alpha_shared = safe_get(sanity, "alpha_match", "alpha_shared", default=cfg.get("alpha_shared_base", None))
    alpha_ctrl = safe_get(sanity, "alpha_match", "alpha_ctrl", default=cfg.get("alpha_ctrl_alpha_match", None))

    overlap_struct = sanity.get("overlap_shared_ctrl_struct", None)
    overlap_energy = sanity.get("overlap_shared_ctrl_energy", None)

    Es_ratio = safe_get(sanity, "energy_shared", "mean_ratio", default=None)
    EcS_ratio = safe_get(sanity, "energy_ctrl_struct", "mean_ratio", default=None)
    EcE_ratio = safe_get(sanity, "energy_ctrl_energy", "mean_ratio", default=None)

    Es_E = safe_get(sanity, "energy_shared", "mean_energy", default=None)
    EcE_E = safe_get(sanity, "energy_ctrl_energy", "mean_energy", default=None)
    energy_match_ratio = (EcE_E / Es_E) if (Es_E is not None and abs(Es_E) > 1e-12 and EcE_E is not None) else None

    # Per task rows
    task_rows: List[Dict[str, Any]] = []
    tasks = sorted(by_task.keys())
    if tasks_filter:
        tasks = [t for t in tasks if t in set(tasks_filter)]

    # aggregate accuracies for run-level means
    accs = {c: [] for c in DEFAULT_CONDS if c != "baseline"}  # include baseline separately
    accs["baseline"] = []

    # significance counters (task-level)
    # Example: shared_vs_ctrl_energy: how many tasks where shared differs from ctrl_energy significantly
    sig_counts = {k: 0 for k in DEFAULT_PAIRED_KEYS}
    neg_sig_counts = {k: 0 for k in DEFAULT_PAIRED_KEYS}  # mean_diff < 0 and sig
    pos_sig_counts = {k: 0 for k in DEFAULT_PAIRED_KEYS}

    for task in tasks:
        blk = by_task[task]
        n = blk.get("n", None)

        row = {
            "file": filename,
            "model": model,
            "model_raw": model_raw,
            "layer": layer,
            "task": task,
            "n": n,
        }

        # accuracies
        for c in DEFAULT_CONDS:
            a = cond_acc(blk, c)
            lo, hi = cond_ci(blk, c)
            row[f"{c}_acc"] = a
            row[f"{c}_ci_low"] = lo
            row[f"{c}_ci_high"] = hi
            if a is not None:
                accs[c].append(float(a))

        # paired stats
        for key in DEFAULT_PAIRED_KEYS:
            st = paired_stat(blk, key)
            if st:
                md = st.get("mean_diff", None)
                row[f"{key}_mean_diff"] = md
                row[f"{key}_ci_low"] = st.get("ci_low", None)
                row[f"{key}_ci_high"] = st.get("ci_high", None)
                row[f"{key}_p"] = st.get("p_value", None)
                if is_sig(st, p_th):
                    sig_counts[key] += 1
                    if md is not None and md < 0:
                        neg_sig_counts[key] += 1
                    if md is not None and md > 0:
                        pos_sig_counts[key] += 1

        task_rows.append(row)

    # run-level mean accuracies
    mean_baseline = mean(accs["baseline"])
    mean_shared_full = mean(accs.get("shared_full", []))
    mean_ctrl_struct = mean(accs.get("ctrl_struct", []))
    mean_ctrl_energy = mean(accs.get("ctrl_energy", []))
    mean_shared_alpha = mean(accs.get("shared_alpha", []))
    mean_ctrl_alpha = mean(accs.get("ctrl_alpha", []))

    # run-level mean diffs (unweighted across tasks)
    def mean_diff_key(key: str) -> Optional[float]:
        vals = []
        for r in task_rows:
            v = r.get(f"{key}_mean_diff", None)
            if v is not None:
                vals.append(float(v))
        return mean(vals)

    run_row = {
        "file": filename,
        "model": model,
        "model_raw": model_raw,
        "layer": layer,
        "cross_dim": cross_dim,
        "k_shared": k_shared,
        "k_c": k_c,
        "alpha_shared": alpha_shared,
        "alpha_ctrl": alpha_ctrl,
        "overlap_struct": overlap_struct,
        "overlap_energy": overlap_energy,
        "Eratio_shared": Es_ratio,
        "Eratio_ctrl_struct": EcS_ratio,
        "Eratio_ctrl_energy": EcE_ratio,
        "Emean_shared": Es_E,
        "Emean_ctrl_energy": EcE_E,
        "Emean_ctrl_energy_over_shared": energy_match_ratio,
        "n_tasks": len(tasks),
        "mean_acc_baseline": mean_baseline,
        "mean_acc_shared_full": mean_shared_full,
        "mean_acc_ctrl_struct": mean_ctrl_struct,
        "mean_acc_ctrl_energy": mean_ctrl_energy,
        "mean_acc_shared_alpha": mean_shared_alpha,
        "mean_acc_ctrl_alpha": mean_ctrl_alpha,
        # mean diffs for key comparisons
        "mean_diff_shared_vs_ctrl_energy": mean_diff_key("shared_vs_ctrl_energy"),
        "mean_diff_shared_vs_ctrl_struct": mean_diff_key("shared_vs_ctrl_struct"),
        "mean_diff_shared_vs_base": mean_diff_key("shared_vs_base"),
        "mean_diff_ctrl_energy_vs_base": mean_diff_key("ctrl_energy_vs_base"),
        "mean_diff_ctrl_struct_vs_base": mean_diff_key("ctrl_struct_vs_base"),
        "mean_diff_shared_alpha_vs_ctrl_alpha": mean_diff_key("shared_alpha_vs_ctrl_alpha"),
        # significance counts
        **{f"sig_{k}": sig_counts[k] for k in sig_counts},
        **{f"neg_sig_{k}": neg_sig_counts[k] for k in neg_sig_counts},
        **{f"pos_sig_{k}": pos_sig_counts[k] for k in pos_sig_counts},
    }

    return run_row, task_rows


def summarize_alpha_sweep_run(
    res: Dict[str, Any],
    *,
    filename: str,
    p_th: float,
    tasks_filter: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Summarize the current run_energy_kmatch_reasoning.py schema.

    Current JSONs store one file with baseline metrics plus alpha_runs:
    alpha -> by_task -> runs {baseline, shared, ctrl_alpha, ctrl_kmatch}.
    For compatibility with the existing report renderer, each alpha is treated
    as one run, with ctrl_alpha mapped to ctrl_struct and ctrl_kmatch mapped to
    ctrl_energy.
    """
    cfg = res.get("config", {})
    basis = res.get("basis", {})
    energy = res.get("energy_calib", {})
    alpha_runs = res.get("alpha_runs", {})

    model_raw = cfg.get("model", "unknown")
    model = short_model_name(model_raw)
    layer_indices = cfg.get("layer_indices", [])
    layer = layer_indices[0] if isinstance(layer_indices, list) and layer_indices else cfg.get("layer", None)
    cross_dim = basis.get("cross_dim", None)
    k_shared = basis.get("k_shared", None)

    run_rows: List[Dict[str, Any]] = []
    task_rows: List[Dict[str, Any]] = []
    task_filter_set = set(tasks_filter or [])

    def alpha_sort_key(x: str):
        try:
            return float(x)
        except Exception:
            return x

    for alpha_key in sorted(alpha_runs.keys(), key=alpha_sort_key):
        alpha_blk = alpha_runs[alpha_key] or {}
        by_task_in = alpha_blk.get("by_task", {})
        by_task: Dict[str, Any] = {}

        for task, blk in by_task_in.items():
            if task_filter_set and task not in task_filter_set:
                continue
            runs_in = blk.get("runs", {})
            paired_in = blk.get("paired", {})
            by_task[task] = {
                "n": blk.get("n"),
                "runs": {
                    "baseline": runs_in.get("baseline", {}),
                    "shared_full": runs_in.get("shared", {}),
                    "ctrl_struct": runs_in.get("ctrl_alpha", {}),
                    "ctrl_energy": runs_in.get("ctrl_kmatch", {}),
                    "shared_alpha": runs_in.get("shared", {}),
                    "ctrl_alpha": runs_in.get("ctrl_alpha", {}),
                },
                "paired": {
                    "shared_vs_base": paired_in.get("shared_vs_base", {}),
                    "shared_vs_ctrl_struct": paired_in.get("shared_vs_ctrl_alpha", {}),
                    "shared_vs_ctrl_energy": paired_in.get("shared_vs_ctrl_kmatch", {}),
                    "shared_alpha_vs_ctrl_alpha": paired_in.get("shared_vs_ctrl_alpha", {}),
                    "ctrl_struct_vs_base": paired_in.get("ctrl_alpha_vs_base", {}),
                    "ctrl_energy_vs_base": paired_in.get("ctrl_kmatch_vs_base", {}),
                },
            }

        normalized = {
            "config": {
                **cfg,
                "cross_dim": cross_dim,
                "k_shared": k_shared,
                "k_c": alpha_blk.get("k_c"),
            },
            "sanity": {
                "energy_shared": energy.get("stats_shared", {}),
                "energy_ctrl_struct": energy.get("stats_ctrl_struct", {}),
                "alpha_match": {
                    "alpha_shared": alpha_blk.get("alpha_shared"),
                    "alpha_ctrl": alpha_blk.get("alpha_ctrl"),
                },
            },
            "by_task": by_task,
        }
        rr, trs = summarize_one_run(
            normalized,
            filename=f"{filename}::alpha={alpha_key}",
            p_th=p_th,
            tasks_filter=tasks_filter,
        )
        rr["alpha_key"] = alpha_key
        rr["model"] = model
        rr["model_raw"] = model_raw
        rr["layer"] = layer
        run_rows.append(rr)
        task_rows.extend(trs)

    return run_rows, task_rows

# -----------------------------
# Markdown report
# -----------------------------
def build_markdown(
    run_rows: List[Dict[str, Any]],
    task_rows: List[Dict[str, Any]],
    *,
    p_th: float,
) -> str:
    md: List[str] = []
    md.append("# Energy K-Match Outputs Summary\n\n")
    md.append(f"- Total runs: **{len(run_rows)}**\n")
    md.append(f"- p-value threshold for `sig_*`: **{p_th}**\n\n")

    # Overview table (one row per run)
    header = [
        "Model", "Layer", "Tasks",
        "k_shared", "k_c",
        "Emean(ctrlE)/Emean(shared)",
        "α_shared", "α_ctrl",
        "MeanAcc Base", "MeanAcc Shared(full)", "MeanAcc Ctrl(struct)", "MeanAcc Ctrl(energy)",
        "Δ(shared-ctrlE) mean",
        "sig(shared-ctrlE)", "neg_sig(shared-ctrlE)",
    ]
    rows = []
    for r in run_rows:
        rows.append([
            str(r.get("model", "")),
            str(r.get("layer", "")),
            str(r.get("n_tasks", "")),
            str(r.get("k_shared", "")),
            str(r.get("k_c", "")),
            f3(r.get("Emean_ctrl_energy_over_shared", None)),
            f4(r.get("alpha_shared", None)),
            f4(r.get("alpha_ctrl", None)),
            pct(r.get("mean_acc_baseline", None)),
            pct(r.get("mean_acc_shared_full", None)),
            pct(r.get("mean_acc_ctrl_struct", None)),
            pct(r.get("mean_acc_ctrl_energy", None)),
            pp(r.get("mean_diff_shared_vs_ctrl_energy", None)),
            str(r.get("sig_shared_vs_ctrl_energy", 0)),
            str(r.get("neg_sig_shared_vs_ctrl_energy", 0)),
        ])
    md.append("## Overview: Runs\n\n")
    md.append(render_md_table(header, rows))
    md.append("\n")

    # Group by model for detailed sections
    by_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in run_rows:
        by_model[str(r.get("model", "unknown"))].append(r)

    # Index task rows by file
    tasks_by_file: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for tr in task_rows:
        tasks_by_file[tr["file"]].append(tr)

    for model in sorted(by_model.keys()):
        md.append(f"# Model: {model}\n\n")
        for rr in by_model[model]:
            file = rr["file"]
            md.append(f"## Run: `{file}`\n\n")

            # Sanity / config block
            md.append("### Config & sanity\n\n")
            md.append(
                "- "
                f"layer={rr.get('layer')} cross_dim={rr.get('cross_dim')} "
                f"k_shared={rr.get('k_shared')} k_c={rr.get('k_c')}  "
                f"overlap_struct={f3(rr.get('overlap_struct'))} overlap_energy={f3(rr.get('overlap_energy'))}\n"
            )
            md.append(
                "- "
                f"E_ratio(shared)={f4(rr.get('Eratio_shared'))} "
                f"E_ratio(ctrl_struct)={f4(rr.get('Eratio_ctrl_struct'))} "
                f"E_ratio(ctrl_energy)={f4(rr.get('Eratio_ctrl_energy'))}\n"
            )
            md.append(
                "- "
                f"E_mean(shared)={sci(rr.get('Emean_shared'))} "
                f"E_mean(ctrl_energy)={sci(rr.get('Emean_ctrl_energy'))} "
                f"ratio={f3(rr.get('Emean_ctrl_energy_over_shared'))}\n"
            )
            md.append(
                "- "
                f"alpha_shared={f4(rr.get('alpha_shared'))} alpha_ctrl(alpha-match)={f4(rr.get('alpha_ctrl'))}\n\n"
            )

            # Per-task accuracy table
            md.append("### Per-task accuracies\n\n")
            trows = tasks_by_file.get(file, [])
            trows = sorted(trows, key=lambda x: x["task"])

            acc_header = [
                "Task", "n",
                "Base", "Shared(full)", "Ctrl(struct)", "Ctrl(energy)",
                "Shared(alpha)", "Ctrl(alpha)",
                "Δ(shared-ctrlE)", "p",
            ]
            acc_rows = []
            for tr in trows:
                st = {
                    "md": tr.get("shared_vs_ctrl_energy_mean_diff", None),
                    "p": tr.get("shared_vs_ctrl_energy_p", None),
                }
                acc_rows.append([
                    tr["task"],
                    str(tr.get("n", "")),
                    pct(tr.get("baseline_acc")),
                    pct(tr.get("shared_full_acc")),
                    pct(tr.get("ctrl_struct_acc")),
                    pct(tr.get("ctrl_energy_acc")),
                    pct(tr.get("shared_alpha_acc")),
                    pct(tr.get("ctrl_alpha_acc")),
                    pp(st["md"]),
                    (f"{st['p']:.4g}" if isinstance(st["p"], (int, float)) else "N/A"),
                ])
            md.append(render_md_table(acc_header, acc_rows))
            md.append("\n")

            # Paired stats table (key comparisons)
            md.append("### Paired stats (mean diff in pp)\n\n")
            paired_header = [
                "Task",
                "Δ(shared-base)", "p",
                "Δ(ctrlE-base)", "p",
                "Δ(shared-ctrlE)", "p",
                "Δ(sharedα-ctrlα)", "p",
            ]
            paired_rows = []
            for tr in trows:
                def cell(key: str) -> Tuple[str, str]:
                    mdv = tr.get(f"{key}_mean_diff", None)
                    pv = tr.get(f"{key}_p", None)
                    return pp(mdv), (f"{pv:.4g}" if isinstance(pv, (int, float)) else "N/A")

                s1, p1 = cell("shared_vs_base")
                s2, p2 = cell("ctrl_energy_vs_base")
                s3, p3 = cell("shared_vs_ctrl_energy")
                s4, p4 = cell("shared_alpha_vs_ctrl_alpha")

                paired_rows.append([tr["task"], s1, p1, s2, p2, s3, p3, s4, p4])

            md.append(render_md_table(paired_header, paired_rows))
            md.append("\n")

            # Run-level quick read
            md.append("### Run-level quick read\n\n")
            md.append(
                f"- MeanAcc: base={pct(rr.get('mean_acc_baseline'))}, "
                f"shared_full={pct(rr.get('mean_acc_shared_full'))}, "
                f"ctrl_struct={pct(rr.get('mean_acc_ctrl_struct'))}, "
                f"ctrl_energy={pct(rr.get('mean_acc_ctrl_energy'))}\n"
            )
            md.append(
                f"- Mean Δ(shared-ctrlE)={pp(rr.get('mean_diff_shared_vs_ctrl_energy'))}  "
                f"(sig tasks={rr.get('sig_shared_vs_ctrl_energy',0)}, neg_sig tasks={rr.get('neg_sig_shared_vs_ctrl_energy',0)})\n"
            )
            md.append("\n---\n\n")

    return "".join(md)

# -----------------------------
# CSV writing (optional)
# -----------------------------
def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    # stable field order
    keys = sorted({k for r in rows for k in r.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            vals = []
            for k in keys:
                v = r.get(k, "")
                # basic CSV escaping
                s = "" if v is None else str(v)
                if any(ch in s for ch in [",", '"', "\n"]):
                    s = '"' + s.replace('"', '""') + '"'
                vals.append(s)
            f.write(",".join(vals) + "\n")

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, required=True, help="Directory containing out_json files")
    ap.add_argument("--pattern", type=str, default="*.json", help="Glob pattern under results_dir")
    ap.add_argument("--output", type=str, default="SUMMARY.md", help="Markdown output path")
    ap.add_argument("--p_threshold", type=float, default=0.05, help="p-value threshold for significance counts")
    ap.add_argument("--tasks", type=str, default="", help="Comma-separated task filter (optional)")

    ap.add_argument("--write_csv", type=int, default=0, help="Whether to also write CSVs (0/1)")
    ap.add_argument("--csv_prefix", type=str, default="summary", help="Prefix path for CSV outputs (no extension)")

    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise SystemExit(f"Not found: {results_dir}")

    files = sorted(results_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files found: dir={results_dir} pattern={args.pattern}")

    tasks_filter = [t.strip() for t in args.tasks.split(",") if t.strip()] if args.tasks.strip() else None

    run_rows: List[Dict[str, Any]] = []
    task_rows: List[Dict[str, Any]] = []

    for p in files:
        j = load_json(p)
        if not j:
            continue
        if isinstance(j.get("alpha_runs"), dict):
            rrs, trs = summarize_alpha_sweep_run(
                j,
                filename=p.name,
                p_th=args.p_threshold,
                tasks_filter=tasks_filter,
            )
            run_rows.extend(rrs)
            task_rows.extend(trs)
        else:
            rr, trs = summarize_one_run(
                j,
                filename=p.name,
                p_th=args.p_threshold,
                tasks_filter=tasks_filter,
            )
            run_rows.append(rr)
            task_rows.extend(trs)

    if not run_rows:
        raise SystemExit("No valid JSON files parsed.")

    md = build_markdown(run_rows, task_rows, p_th=args.p_threshold)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"[Done] wrote markdown: {out_path}")

    if int(args.write_csv) == 1:
        prefix = Path(args.csv_prefix)
        write_csv(Path(str(prefix) + "_runs.csv"), run_rows)
        write_csv(Path(str(prefix) + "_tasks.csv"), task_rows)
        print(f"[Done] wrote CSVs: {str(prefix)}_runs.csv and {str(prefix)}_tasks.csv")

if __name__ == "__main__":
    main()
