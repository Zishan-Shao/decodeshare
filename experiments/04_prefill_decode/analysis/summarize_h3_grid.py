# -*- coding: utf-8 -*-
"""
summarize_h3_grid.py

Analyze H3 grid json outputs from:
  - run_h3_grid_reasoning.py (full 2x2 + controls)
  - legacy H3 grid outputs (decode-intervene only)

Outputs:
  - Per-task accuracy table
  - Per-task delta-accuracy (pp) relative to the matching baseline protocol
  - Macro + micro averages
  - Key H3 contrast summary
  - Optional CSV / LaTeX export

Usage examples:
  python analysis/summarize_h3_grid.py --inputs "h3_grid_v3_*.json"
  python analysis/summarize_h3_grid.py --inputs run1.json,run2.json --out_csv summary.csv --out_latex table.tex

A) 分析单个 json
python analysis/summarize_h3_grid.py --inputs h3_grid_v3_*.json

B) 输出 CSV（每个 task 一行）
python analysis/summarize_h3_grid.py --inputs h3_grid_v3_run.json --out_csv h3_summary.csv

C) 输出 LaTeX 表格（默认是 ΔAcc(pp)）
python analysis/summarize_h3_grid.py --inputs h3_grid_v3_run.json --out_latex h3_table.tex --latex_mode delta


如果想导出 raw accuracy（百分数）：
python analysis/summarize_h3_grid.py --inputs h3_grid_v3_run.json --out_latex h3_acc_table.tex --latex_mode acc

D) 同时分析多个 runs（多个文件会逐个打印；导出时会自动按文件名加后缀）
python analysis/summarize_h3_grid.py --inputs "h3_grid_v3_meta-llama_Llama-2-7b-chat-hf_layer10_k48_W0_seed0.json" --out_csv out.csv --out_latex out.tex --latex_mode acc

python analysis/summarize_h3_grid.py --inputs "/home/zs89/decodeshare/results/h3_grid/h3_grid_v3_Qwen_Qwen2.5-7B-Instruct_layer10_k20_W0_seed0.json" --out_csv Qwen_Qwen2.5-7B-Instruct_layer10_out.csv --out_latex Qwen_Qwen2.5-7B-Instruct_layer10_out.tex --latex_mode acc


"""

from __future__ import annotations

import argparse
import glob
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# -----------------------------
# Helpers
# -----------------------------
def _safe_get(d: Dict[str, Any], path: List[str]) -> Optional[Any]:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _extract_acc_ci(entry: Optional[Dict[str, Any]]) -> Tuple[float, float, float]:
    """
    Return (acc, ci_low, ci_high). If missing, return NaNs.
    acc is assumed to be in [0,1].
    """
    if not isinstance(entry, dict):
        return float("nan"), float("nan"), float("nan")
    acc = float(entry.get("acc", float("nan")))
    lo = float(entry.get("ci_low", float("nan")))
    hi = float(entry.get("ci_high", float("nan")))
    return acc, lo, hi


def _extract_correct(entry: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
    """
    Return correct array as float32 of shape [N], or None if not present.
    """
    if not isinstance(entry, dict):
        return None
    arr = entry.get("correct", None)
    if arr is None:
        return None
    try:
        x = np.asarray(arr, dtype=np.float32)
        if x.ndim != 1:
            return None
        return x
    except Exception:
        return None


def bootstrap_ci_mean(x: np.ndarray, iters: int = 2000, alpha: float = 0.05, seed: int = 0) -> Tuple[float, float, float]:
    """
    Nonparametric bootstrap CI for mean(x).
    x: 1D array of 0/1 correctness.
    """
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = x.size
    means = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        idx = rng.integers(0, n, size=n)
        means[i] = float(np.mean(x[idx]))
    m = float(np.mean(x))
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return m, lo, hi


def pp(x: float) -> float:
    """prob -> percentage points"""
    return 100.0 * x


def fmt_acc_ci(acc: float, lo: float, hi: float) -> str:
    if any(math.isnan(v) for v in [acc, lo, hi]):
        return "NaN"
    return f"{pp(acc):.1f} [{pp(lo):.1f},{pp(hi):.1f}]"


def fmt_pp(x: float) -> str:
    if math.isnan(x):
        return "NaN"
    return f"{x:+.1f}"


# -----------------------------
# Canonical condition names
# -----------------------------
COND_DEC_DEC = "Dec-est/Dec-int"
COND_PRE_DEC = "Pre-est/Dec-int"
COND_RAND_DEC = "Rand/Dec-int"

COND_DEC_PRE = "Dec-est/Pre-int"
COND_PRE_PRE = "Pre-est/Pre-int"
COND_RAND_PRE = "Rand/Pre-int"


@dataclass
class TaskMetrics:
    task: str

    # baselines (two protocols)
    base_dec_acc: float
    base_dec_lo: float
    base_dec_hi: float

    base_pre_acc: float
    base_pre_lo: float
    base_pre_hi: float

    # conditions
    cond: Dict[str, Tuple[float, float, float]]          # acc/lo/hi
    delta_pp: Dict[str, float]                           # delta vs matching baseline in percentage points

    # optionally correctness arrays (for micro aggregation)
    correct_by_key: Dict[str, np.ndarray]                # e.g. "base_dec", "Dec-est/Dec-int", ...


def detect_version(run: Dict[str, Any]) -> str:
    """
    Returns 'v3' if it looks like v3, else 'v2'.
    """
    tasks = run.get("tasks", {})
    if not isinstance(tasks, dict) or len(tasks) == 0:
        return "unknown"
    # peek one task entry
    any_task = next(iter(tasks.values()))
    if isinstance(any_task, dict) and ("baseline_dec_proto" in any_task or "decode_intervene" in any_task):
        return "v3"
    if isinstance(any_task, dict) and ("baseline" in any_task and "decode" in any_task):
        return "v2"
    return "unknown"


def parse_task_v3(task: str, te: Dict[str, Any]) -> TaskMetrics:
    base_dec = _safe_get(te, ["baseline_dec_proto"])
    base_pre = _safe_get(te, ["baseline_pre_proto"])

    base_dec_acc, base_dec_lo, base_dec_hi = _extract_acc_ci(base_dec)
    base_pre_acc, base_pre_lo, base_pre_hi = _extract_acc_ci(base_pre)

    # decode-intervene conditions
    dec_dec = _safe_get(te, ["decode_intervene", "dec_est_dec_int"])
    pre_dec = _safe_get(te, ["decode_intervene", "pre_est_dec_int"])
    rand_dec = _safe_get(te, ["decode_intervene", "rand_ctl_dec_int"])

    # prefill-intervene conditions
    dec_pre = _safe_get(te, ["prefill_intervene", "dec_est_pre_int"])
    pre_pre = _safe_get(te, ["prefill_intervene", "pre_est_pre_int"])
    rand_pre = _safe_get(te, ["prefill_intervene", "rand_ctl_pre_int"])

    cond: Dict[str, Tuple[float, float, float]] = {}
    for name, entry in [
        (COND_DEC_DEC, dec_dec),
        (COND_PRE_DEC, pre_dec),
        (COND_RAND_DEC, rand_dec),
        (COND_DEC_PRE, dec_pre),
        (COND_PRE_PRE, pre_pre),
        (COND_RAND_PRE, rand_pre),
    ]:
        cond[name] = _extract_acc_ci(entry)

    # deltas: decode-int uses base_dec; prefill-int uses base_pre
    delta_pp: Dict[str, float] = {}
    for name, (acc, _lo, _hi) in cond.items():
        if math.isnan(acc):
            delta_pp[name] = float("nan")
            continue
        if name.endswith("Dec-int"):
            delta_pp[name] = pp(acc - base_dec_acc)
        elif name.endswith("Pre-int"):
            delta_pp[name] = pp(acc - base_pre_acc)
        else:
            delta_pp[name] = float("nan")

    correct_by_key: Dict[str, np.ndarray] = {}
    cd = _extract_correct(base_dec)
    cp = _extract_correct(base_pre)
    if cd is not None:
        correct_by_key["base_dec"] = cd
    if cp is not None:
        correct_by_key["base_pre"] = cp

    for name, entry in [
        (COND_DEC_DEC, dec_dec),
        (COND_PRE_DEC, pre_dec),
        (COND_RAND_DEC, rand_dec),
        (COND_DEC_PRE, dec_pre),
        (COND_PRE_PRE, pre_pre),
        (COND_RAND_PRE, rand_pre),
    ]:
        c = _extract_correct(entry)
        if c is not None:
            correct_by_key[name] = c

    return TaskMetrics(
        task=task,
        base_dec_acc=base_dec_acc, base_dec_lo=base_dec_lo, base_dec_hi=base_dec_hi,
        base_pre_acc=base_pre_acc, base_pre_lo=base_pre_lo, base_pre_hi=base_pre_hi,
        cond=cond,
        delta_pp=delta_pp,
        correct_by_key=correct_by_key,
    )


def parse_task_v2(task: str, te: Dict[str, Any]) -> TaskMetrics:
    # v2 only had one protocol; we map it into base_dec and leave base_pre = base_dec
    base = _safe_get(te, ["baseline"])
    dec = _safe_get(te, ["decode"])
    pre = _safe_get(te, ["prefill"])
    ctl = _safe_get(te, ["control"])

    base_acc, base_lo, base_hi = _extract_acc_ci(base)

    cond: Dict[str, Tuple[float, float, float]] = {
        COND_DEC_DEC: _extract_acc_ci(dec),    # decode-est/decode-int
        COND_PRE_DEC: _extract_acc_ci(pre),    # prefill-est/decode-int
        COND_RAND_DEC: _extract_acc_ci(ctl),   # control-rand/decode-int
        COND_DEC_PRE: (float("nan"), float("nan"), float("nan")),
        COND_PRE_PRE: (float("nan"), float("nan"), float("nan")),
        COND_RAND_PRE: (float("nan"), float("nan"), float("nan")),
    }

    delta_pp: Dict[str, float] = {}
    for name, (acc, _lo, _hi) in cond.items():
        if math.isnan(acc):
            delta_pp[name] = float("nan")
        else:
            # v2 only meaningful for Dec-int arm
            delta_pp[name] = pp(acc - base_acc)

    correct_by_key: Dict[str, np.ndarray] = {}
    cb = _extract_correct(base)
    if cb is not None:
        correct_by_key["base_dec"] = cb
        correct_by_key["base_pre"] = cb  # same
    for name, entry in [
        (COND_DEC_DEC, dec),
        (COND_PRE_DEC, pre),
        (COND_RAND_DEC, ctl),
    ]:
        c = _extract_correct(entry)
        if c is not None:
            correct_by_key[name] = c

    return TaskMetrics(
        task=task,
        base_dec_acc=base_acc, base_dec_lo=base_lo, base_dec_hi=base_hi,
        base_pre_acc=base_acc, base_pre_lo=base_lo, base_pre_hi=base_hi,  # alias
        cond=cond,
        delta_pp=delta_pp,
        correct_by_key=correct_by_key,
    )


def analyze_one_file(path: str, bootstrap_iters: int, seed: int) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        run = json.load(f)

    ver = detect_version(run)
    tasks = run.get("tasks", {})
    if not isinstance(tasks, dict) or len(tasks) == 0:
        raise ValueError(f"No tasks found in json: {path}")

    metrics: List[TaskMetrics] = []
    for task, te in tasks.items():
        if not isinstance(te, dict):
            continue
        if ver == "v3":
            metrics.append(parse_task_v3(task, te))
        elif ver == "v2":
            metrics.append(parse_task_v2(task, te))
        else:
            # try v3 first, fallback v2
            if "baseline_dec_proto" in te or "decode_intervene" in te:
                metrics.append(parse_task_v3(task, te))
            elif "baseline" in te and "decode" in te:
                metrics.append(parse_task_v2(task, te))

    # Sort tasks for stable printing
    metrics.sort(key=lambda m: m.task)

    # Macro averages over tasks (ignore NaNs)
    def macro_mean(vals: List[float]) -> float:
        x = np.asarray([v for v in vals if not math.isnan(v)], dtype=np.float64)
        return float(np.mean(x)) if x.size > 0 else float("nan")

    macro = {
        "base_dec_acc": macro_mean([m.base_dec_acc for m in metrics]),
        "base_pre_acc": macro_mean([m.base_pre_acc for m in metrics]),
        "delta_pp": {name: macro_mean([m.delta_pp.get(name, float("nan")) for m in metrics])
                     for name in [COND_DEC_DEC, COND_PRE_DEC, COND_RAND_DEC, COND_DEC_PRE, COND_PRE_PRE, COND_RAND_PRE]},
    }

    # Micro averages (concatenate correctness arrays if present)
    # We'll compute micro for decode-proto group and prefill-proto group separately.
    def micro_from_keys(keys: List[str]) -> Tuple[float, float, float, int]:
        arrs = []
        for m in metrics:
            for k in keys:
                if k in m.correct_by_key:
                    arrs.append(m.correct_by_key[k])
        if len(arrs) == 0:
            return float("nan"), float("nan"), float("nan"), 0
        x = np.concatenate(arrs, axis=0)
        acc, lo, hi = bootstrap_ci_mean(x, iters=bootstrap_iters, alpha=0.05, seed=seed)
        return float(acc), float(lo), float(hi), int(x.size)

    micro: Dict[str, Any] = {}

    # baselines
    bdec_acc, bdec_lo, bdec_hi, bdec_n = micro_from_keys(["base_dec"])
    bpre_acc, bpre_lo, bpre_hi, bpre_n = micro_from_keys(["base_pre"])
    micro["baseline_dec_proto"] = {"acc": bdec_acc, "lo": bdec_lo, "hi": bdec_hi, "n": bdec_n}
    micro["baseline_pre_proto"] = {"acc": bpre_acc, "lo": bpre_lo, "hi": bpre_hi, "n": bpre_n}

    # conditions (each condition concatenates across tasks)
    for cond_name in [COND_DEC_DEC, COND_PRE_DEC, COND_RAND_DEC, COND_DEC_PRE, COND_PRE_PRE, COND_RAND_PRE]:
        acc, lo, hi, n = micro_from_keys([cond_name])
        micro[cond_name] = {"acc": acc, "lo": lo, "hi": hi, "n": n}

    # Key contrasts (macro)
    h3_contrast_macro = macro["delta_pp"][COND_DEC_DEC] - macro["delta_pp"][COND_PRE_DEC]
    cross_prefill_macro = macro["delta_pp"][COND_DEC_PRE] - macro["delta_pp"][COND_PRE_PRE]

    # Baseline sanity: base_pre - base_dec (per task)
    sanity_gaps = [(m.task, pp(m.base_pre_acc - m.base_dec_acc)) for m in metrics]
    max_gap = max((abs(g) for _t, g in sanity_gaps), default=float("nan"))

    out = {
        "path": path,
        "version": ver,
        "meta": {
            "model": run.get("model", None),
            "layer": run.get("layer", None),
            "k_match": run.get("k_match", None),
            "angles_deg": run.get("angles_deg", None),
            "alpha_remove": run.get("alpha_remove", None),
            "warmup_tokens": run.get("warmup_tokens", None),
        },
        "metrics": metrics,
        "macro": macro,
        "micro": micro,
        "contrasts": {
            "H3_macro_pp__Delta(Dec-est/Dec-int) - Delta(Pre-est/Dec-int)": float(h3_contrast_macro),
            "Cross_macro_pp__Delta(Dec-est/Pre-int) - Delta(Pre-est/Pre-int)": float(cross_prefill_macro),
            "max_abs_baseline_gap_pp__base_pre - base_dec": float(max_gap),
        },
        "sanity_gaps_pp": sanity_gaps,
    }
    return out


def print_report(run_out: Dict[str, Any], show_tasks: bool = True) -> None:
    path = run_out["path"]
    meta = run_out["meta"]
    metrics: List[TaskMetrics] = run_out["metrics"]
    macro = run_out["macro"]
    micro = run_out["micro"]
    contrasts = run_out["contrasts"]

    print("\n" + "=" * 120)
    print(f"[FILE] {path}")
    print(f"[META] model={meta.get('model')} layer={meta.get('layer')} k={meta.get('k_match')} "
          f"alpha={meta.get('alpha_remove')} warmup={meta.get('warmup_tokens')}")
    if meta.get("angles_deg") is not None:
        ang = meta["angles_deg"]
        print(f"[ANGLES deg] mean={ang.get('mean'):.2f} p50={ang.get('p50'):.2f} p95={ang.get('p95'):.2f}")
    print(f"[VERSION] {run_out.get('version')}")

    print("\n[Macro averages across tasks]")
    print(f"  baseline_dec_proto: {pp(macro['base_dec_acc']):.2f}")
    print(f"  baseline_pre_proto: {pp(macro['base_pre_acc']):.2f}")
    for name, v in macro["delta_pp"].items():
        print(f"  Δpp {name:<14}: {fmt_pp(v)}")

    print("\n[Micro averages across all examples (concatenated)]")
    bd = micro["baseline_dec_proto"]
    bp = micro["baseline_pre_proto"]
    print(f"  baseline_dec_proto: {fmt_acc_ci(bd['acc'], bd['lo'], bd['hi'])} (n={bd['n']})")
    print(f"  baseline_pre_proto: {fmt_acc_ci(bp['acc'], bp['lo'], bp['hi'])} (n={bp['n']})")
    for name in [COND_DEC_DEC, COND_PRE_DEC, COND_RAND_DEC, COND_DEC_PRE, COND_PRE_PRE, COND_RAND_PRE]:
        m = micro[name]
        if math.isnan(m["acc"]):
            continue
        print(f"  {name:<16}: {fmt_acc_ci(m['acc'], m['lo'], m['hi'])} (n={m['n']})")

    print("\n[Key contrasts (macro, percentage points)]")
    for k, v in contrasts.items():
        print(f"  {k}: {fmt_pp(v)}")

    if show_tasks:
        print("\n[Per-task table]")
        header = (
            "task".ljust(16) +
            " | base_dec".ljust(18) +
            " | base_pre".ljust(18) +
            " | Δ Dec/Dec".ljust(12) +
            " | Δ Pre/Dec".ljust(12) +
            " | Δ Rand/Dec".ljust(12) +
            " | Δ Dec/Pre".ljust(12) +
            " | Δ Pre/Pre".ljust(12) +
            " | Δ Rand/Pre".ljust(12)
        )
        print(header)
        print("-" * len(header))
        for m in metrics:
            row = (
                m.task.ljust(16) +
                " | " + fmt_acc_ci(m.base_dec_acc, m.base_dec_lo, m.base_dec_hi).ljust(16) +
                " | " + fmt_acc_ci(m.base_pre_acc, m.base_pre_lo, m.base_pre_hi).ljust(16) +
                " | " + fmt_pp(m.delta_pp.get(COND_DEC_DEC, float('nan'))).ljust(10) +
                " | " + fmt_pp(m.delta_pp.get(COND_PRE_DEC, float('nan'))).ljust(10) +
                " | " + fmt_pp(m.delta_pp.get(COND_RAND_DEC, float('nan'))).ljust(10) +
                " | " + fmt_pp(m.delta_pp.get(COND_DEC_PRE, float('nan'))).ljust(10) +
                " | " + fmt_pp(m.delta_pp.get(COND_PRE_PRE, float('nan'))).ljust(10) +
                " | " + fmt_pp(m.delta_pp.get(COND_RAND_PRE, float('nan'))).ljust(10)
            )
            print(row)

        # Baseline sanity gaps
        gaps = run_out.get("sanity_gaps_pp", [])
        if gaps:
            worst = max(gaps, key=lambda x: abs(x[1]))
            print("\n[Sanity] base_pre - base_dec (pp) per task (should be ~0); worst:")
            print(f"  worst_task={worst[0]} gap_pp={worst[1]:+.2f}")


def export_csv(run_out: Dict[str, Any], out_csv: str) -> None:
    """
    Export a wide CSV with per-task accuracies and deltas.
    """
    metrics: List[TaskMetrics] = run_out["metrics"]
    rows = []
    for m in metrics:
        row = {
            "task": m.task,
            "base_dec_acc": m.base_dec_acc,
            "base_pre_acc": m.base_pre_acc,
            "delta_pp_Dec-est_Dec-int": m.delta_pp.get(COND_DEC_DEC, float("nan")),
            "delta_pp_Pre-est_Dec-int": m.delta_pp.get(COND_PRE_DEC, float("nan")),
            "delta_pp_Rand_Dec-int": m.delta_pp.get(COND_RAND_DEC, float("nan")),
            "delta_pp_Dec-est_Pre-int": m.delta_pp.get(COND_DEC_PRE, float("nan")),
            "delta_pp_Pre-est_Pre-int": m.delta_pp.get(COND_PRE_PRE, float("nan")),
            "delta_pp_Rand_Pre-int": m.delta_pp.get(COND_RAND_PRE, float("nan")),
        }
        rows.append(row)

    # pure stdlib CSV write (no pandas dependency)
    import csv
    outp = Path(out_csv)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def export_latex_table(run_out: Dict[str, Any], out_tex: str, mode: str = "delta") -> None:
    """
    Export a simple LaTeX tabular.
    mode:
      - 'delta': exports ΔAcc (pp) for the 2x2 grid (and controls if present)
      - 'acc'  : exports raw accuracies (%)
    """
    metrics: List[TaskMetrics] = run_out["metrics"]
    outp = Path(out_tex)
    outp.parent.mkdir(parents=True, exist_ok=True)

    # Choose columns for LaTeX
    cols = [
        ("Task", None),
        (COND_DEC_DEC, COND_DEC_DEC),
        (COND_PRE_DEC, COND_PRE_DEC),
        (COND_DEC_PRE, COND_DEC_PRE),
        (COND_PRE_PRE, COND_PRE_PRE),
        (COND_RAND_DEC, COND_RAND_DEC),
        (COND_RAND_PRE, COND_RAND_PRE),
    ]

    lines: List[str] = []
    lines.append(r"\begin{tabular}{lrrrrrr}")
    lines.append(r"\toprule")
    header = " & ".join([c[0] for c in cols]) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for m in metrics:
        vals = [m.task]
        for _title, key in cols[1:]:
            if mode == "delta":
                v = m.delta_pp.get(key, float("nan"))
                vals.append(f"{v:+.1f}" if not math.isnan(v) else "")
            else:
                acc, lo, hi = m.cond.get(key, (float("nan"), float("nan"), float("nan")))
                vals.append(f"{pp(acc):.1f}" if not math.isnan(acc) else "")
        lines.append(" & ".join(vals) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    with open(outp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=str, required=True,
                    help="Comma-separated file paths or glob patterns. e.g. 'h3_grid_v3_*.json,run2.json'")
    ap.add_argument("--bootstrap_iters", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_task_table", action="store_true")

    ap.add_argument("--out_csv", type=str, default="")
    ap.add_argument("--out_latex", type=str, default="")
    ap.add_argument("--latex_mode", type=str, default="delta", choices=["delta", "acc"])

    args = ap.parse_args()

    # Expand inputs (comma-separated, each may be glob)
    patterns = [p.strip() for p in args.inputs.split(",") if p.strip()]
    files: List[str] = []
    for p in patterns:
        m = glob.glob(p)
        if len(m) == 0 and Path(p).exists():
            files.append(p)
        else:
            files.extend(sorted(m))
    if len(files) == 0:
        raise SystemExit(f"No input files found from: {args.inputs}")

    for fp in files:
        out = analyze_one_file(fp, bootstrap_iters=args.bootstrap_iters, seed=args.seed)
        print_report(out, show_tasks=(not args.no_task_table))

        if args.out_csv:
            # if multiple files, suffix by filename stem
            out_csv = args.out_csv
            if len(files) > 1:
                stem = Path(fp).stem
                out_csv = str(Path(args.out_csv).with_name(f"{Path(args.out_csv).stem}_{stem}{Path(args.out_csv).suffix or '.csv'}"))
            export_csv(out, out_csv)
            print(f"[CSV] wrote {out_csv}")

        if args.out_latex:
            out_tex = args.out_latex
            if len(files) > 1:
                stem = Path(fp).stem
                out_tex = str(Path(args.out_latex).with_name(f"{Path(args.out_latex).stem}_{stem}{Path(args.out_latex).suffix or '.tex'}"))
            export_latex_table(out, out_tex, mode=args.latex_mode)
            print(f"[LaTeX] wrote {out_tex}")


if __name__ == "__main__":
    main()
