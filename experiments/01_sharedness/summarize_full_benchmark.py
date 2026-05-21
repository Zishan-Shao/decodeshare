#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summarize Hypothesis-1 evidence from a full_benchmark result directory.

H1 (Shared workspace exists at decode time):
  The decode-time shared set size |S_l(tau, m)| exceeds chance under BOTH nulls.
  Falsification: if |S_l(tau, m)| is not in the tail of either null distribution, reject H1.

This script aggregates results for the three variants used in the repo:
  - full:      *_exist.(json|txt)
  - pooled:    *_exist_pooled.(json|txt)
  - loosened:  *_exist_loosened.(json|txt)

It prefers JSON if available, but can fall back to TXT logs (which contain the two p-values).

Usage:
  python experiments/01_sharedness/summarize_full_benchmark.py \
    --results_dir paper_artifacts/h1_results/results/full_benchmark \
    --out_dir paper_artifacts/h1_results/results/full_benchmark \
    --alpha 0.05
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def fmt_float(x: Optional[float], nd: int = 4) -> str:
    if x is None:
        return "N/A"
    return f"{float(x):.{nd}f}"


def fmt_int(x: Any) -> str:
    try:
        return str(int(x))
    except Exception:
        return "N/A"


def latex_escape(s: str) -> str:
    s = str(s)
    s = s.replace("\\", "\\textbackslash{}")
    s = s.replace("&", "\\&")
    s = s.replace("%", "\\%")
    s = s.replace("$", "\\$")
    s = s.replace("#", "\\#")
    s = s.replace("_", "\\_")
    s = s.replace("{", "\\{")
    s = s.replace("}", "\\}")
    s = s.replace("~", "\\textasciitilde{}")
    s = s.replace("^", "\\textasciicircum{}")
    return s


def detect_variant(stem: str) -> str:
    if "_exist_loosened" in stem:
        return "loosened"
    if "_exist_pooled" in stem:
        return "pooled"
    if "_exist" in stem:
        return "full"
    return "unknown"


def base_stub(stem: str) -> str:
    for suf in ["_exist_loosened", "_exist_pooled", "_exist"]:
        stem = stem.replace(suf, "")
    return stem


def parse_hidden_dim_from_txt(txt: str) -> Optional[int]:
    m = re.search(r"hidden_dim\s*=\s*(\d+)", txt)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def parse_txt_result(txt_path: Path) -> Dict[str, Any]:
    txt = txt_path.read_text(encoding="utf-8", errors="ignore")

    # p-values appear twice: Null-1 then Null-2
    pvals = re.findall(r"p-value \(null>=obs\) = ([0-9.eE+-]+)", txt)
    p1 = float(pvals[0]) if len(pvals) >= 1 else None
    p2 = float(pvals[1]) if len(pvals) >= 2 else None

    # trials
    perm_trials = None
    scr_trials = None
    m = re.search(r"\[Null-1\].*trials=(\d+)", txt)
    if m:
        perm_trials = int(m.group(1))
    m = re.search(r"\[Null-2\].*trials=(\d+)", txt)
    if m:
        scr_trials = int(m.group(1))

    # observed numbers
    model = None
    m = re.search(r"\[Env\]\s+model=([^\s]+)", txt)
    if m:
        model = m.group(1)

    layer = None
    m = re.search(r"\[Env\].*layer=?\s*\[?(\d+)\]?", txt)
    if m:
        layer = int(m.group(1))

    cross_dim = None
    m = re.search(r"\[PCA\]\s+cross_dim=(\d+)", txt)
    if m:
        cross_dim = int(m.group(1))

    shared_count = None
    m = re.search(r"OBS\s+shared_count=(\d+)\s*/\s*cross_dim=(\d+)", txt)
    if m:
        shared_count = int(m.group(1))
        cross_dim = int(m.group(2))

    tau = None
    m_shared = None
    m = re.search(r"tau=([0-9.eE+-]+)\s+m_shared=([^\s]+)", txt)
    if m:
        tau = float(m.group(1))
        m_shared = m.group(2)

    hidden_dim = parse_hidden_dim_from_txt(txt)

    return {
        "source": "txt",
        "path": str(txt_path),
        "model": model,
        "layer": layer,
        "hidden_dim": hidden_dim,
        "cross_dim": cross_dim,
        "shared_count": shared_count,
        "tau": tau,
        "m_shared": m_shared,
        "null_perm_trials": perm_trials,
        "null_scramble_trials": scr_trials,
        "p_null1_perm": p1,
        "p_null2_scramble": p2,
    }


def parse_json_result(json_path: Path) -> Dict[str, Any]:
    obj = json.loads(json_path.read_text(encoding="utf-8"))
    cfg = obj.get("config", {}) if isinstance(obj, dict) else {}
    obs = obj.get("observed", {}) if isinstance(obj, dict) else {}
    return {
        "source": "json",
        "path": str(json_path),
        "model": cfg.get("model"),
        "layer": cfg.get("layer"),
        "cross_dim": obs.get("cross_dim"),
        "shared_count": obs.get("shared_count"),
        "tau": cfg.get("tau"),
        "m_shared": cfg.get("m_shared"),
        "null_perm_trials": cfg.get("null_perm_trials"),
        "null_scramble_trials": cfg.get("null_scramble_trials"),
        "p_null1_perm": obs.get("p_null1_perm"),
        "p_null2_scramble": obs.get("p_null2_scramble"),
        "out_txt": cfg.get("out_txt"),
    }


def resolve_out_txt(results_dir: Path, out_txt: Optional[str]) -> Optional[Path]:
    if not out_txt:
        return None
    p = Path(out_txt)
    if p.is_absolute() and p.exists():
        return p
    # Most out_txt are relative like "results/full_benchmark/foo.txt"
    cand = results_dir / Path(out_txt).name
    return cand if cand.exists() else None


def choose_best_record(results_dir: Path, stub: str, variant: str) -> Dict[str, Any]:
    """
    Prefer JSON if it exists and doesn't obviously point to a different variant's out_txt.
    Otherwise fall back to TXT.
    """
    json_path = results_dir / f"{stub}_{'exist' if variant=='full' else ('exist_'+variant)}.json"
    txt_path = results_dir / f"{stub}_{'exist' if variant=='full' else ('exist_'+variant)}.txt"

    rec_json = parse_json_result(json_path) if json_path.exists() else None
    rec_txt = parse_txt_result(txt_path) if txt_path.exists() else None

    if rec_json is not None:
        out_txt = resolve_out_txt(results_dir, rec_json.get("out_txt"))
        if out_txt is not None:
            out_variant = detect_variant(out_txt.stem)
            if out_variant != variant and rec_txt is not None:
                rec_txt["note"] = f"json out_txt points to {out_txt.name} (variant={out_variant})"
                return rec_txt
        return rec_json

    if rec_txt is not None:
        return rec_txt

    return {"source": "missing", "path": "", "model": stub, "variant": variant}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--label", type=str, default="tab:h1_full_benchmark")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # enumerate stubs from any file in the directory
    stubs: Dict[str, set] = {}
    for p in sorted(results_dir.glob("**/*")):
        if not p.is_file() or p.suffix not in [".json", ".txt"]:
            continue
        v = detect_variant(p.stem)
        if v == "unknown":
            continue
        stub = base_stub(p.stem)
        stubs.setdefault(stub, set()).add(v)

    variants = ["full", "pooled", "loosened"]
    rows: List[Dict[str, Any]] = []
    for stub in sorted(stubs.keys()):
        for v in variants:
            if v not in stubs[stub]:
                continue
            rec = choose_best_record(results_dir, stub, v)
            rec["variant"] = v
            rec["stub"] = stub
            rows.append(rec)

    alpha = float(args.alpha)
    for r in rows:
        sc = r.get("shared_count")
        cd = r.get("cross_dim")
        try:
            r["shared_frac_cross"] = (float(sc) / float(cd)) if (sc is not None and cd) else None
        except Exception:
            r["shared_frac_cross"] = None

        p1 = r.get("p_null1_perm")
        p2 = r.get("p_null2_scramble")
        r["pass_null1"] = bool(p1 is not None and float(p1) < alpha)
        r["pass_null2"] = bool(p2 is not None and float(p2) < alpha)
        r["supports_H1"] = bool(r["pass_null1"] and r["pass_null2"] and (sc is not None and int(sc) > 0))

        n2 = r.get("null_scramble_trials")
        try:
            r["p2_min"] = 1.0 / (int(n2) + 1) if n2 else None
        except Exception:
            r["p2_min"] = None

    # CSV
    out_csv = out_dir / "H1_full_benchmark_summary.csv"
    cols = [
        "stub","variant","model","layer","tau","m_shared",
        "cross_dim","shared_count","shared_frac_cross",
        "null_perm_trials","p_null1_perm","pass_null1",
        "null_scramble_trials","p_null2_scramble","p2_min","pass_null2",
        "supports_H1","source","note","path"
    ]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

    # Markdown
    out_md = out_dir / "H1_full_benchmark_summary.md"
    md: List[str] = []
    md.append("# H1 summary: decode-time shared workspace exists (full_benchmark)\n")
    md.append(f"- results_dir: `{results_dir}`\n")
    md.append(f"- alpha: {alpha}\n\n")
    md.append("H1 support criterion used here: `supports_H1 = (p_null1_perm < alpha) AND (p_null2_scramble < alpha) AND (shared_count > 0)`.\n\n")
    md.append("| Model | Variant | Layer | tau | m_shared | cross_dim | |S| | |S|/cross_dim | p1 (perm) | p2 (scramble) | H1 |\n")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:--:|\n")
    for r in rows:
        model = r.get("model") or r.get("stub")
        md.append(
            f"| {model} | {r.get('variant')} | {r.get('layer')} | {r.get('tau')} | {r.get('m_shared')} | "
            f"{r.get('cross_dim')} | {r.get('shared_count')} | {fmt_float(r.get('shared_frac_cross'), nd=3)} | "
            f"{fmt_float(r.get('p_null1_perm'), nd=6)} | {fmt_float(r.get('p_null2_scramble'), nd=6)} | "
            f"{'PASS' if r.get('supports_H1') else 'FAIL'} |\n"
        )
    md.append("\nNotes:\n")
    md.append("- p2_min = 1/(N_scramble+1) is the minimum attainable p-value for Null-2 at finite trials; when p2==p2_min, it indicates zero exceedances in the scramble null.\n")
    out_md.write_text("".join(md), encoding="utf-8")

    # LaTeX
    out_tex = out_dir / "H1_full_benchmark_summary.tex"
    tex: List[str] = []
    tex.append("% Auto-generated by experiments/01_sharedness/summarize_full_benchmark.py\n")
    tex.append("% Requires: \\usepackage{booktabs}\n")
    tex.append("\\begin{table}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{4.5pt}\n")
    tex.append("\\begin{tabular}{lllrrrrrrr}\n\\toprule\n")
    tex.append("Model & Variant & Layer & $\\tau$ & $m$ & cross & $|S|$ & $|S|/cross$ & $p_1$ & $p_2$ \\\\\n")
    tex.append("\\midrule\n")
    for r in rows:
        model = r.get("model") or r.get("stub")
        tex.append(
            f"{latex_escape(model)} & {latex_escape(str(r.get('variant')))} & {fmt_int(r.get('layer'))} "
            f"& {fmt_float(r.get('tau'), nd=4)} & {latex_escape(str(r.get('m_shared')))} "
            f"& {fmt_int(r.get('cross_dim'))} & {fmt_int(r.get('shared_count'))} "
            f"& {fmt_float(r.get('shared_frac_cross'), nd=3)} "
            f"& {fmt_float(r.get('p_null1_perm'), nd=6)} & {fmt_float(r.get('p_null2_scramble'), nd=6)} \\\\\n"
        )
    tex.append("\\bottomrule\n\\end{tabular}\n")
    tex.append(
        "\\caption{"
        "Hypothesis H1 (decode-time shared workspace exists) evaluated on full\\_benchmark. "
        "We report the decode-time shared set size $|S_\\ell(\\tau,m)|$ and its null p-values: "
        "$p_1$ (Null-1 permutation) and $p_2$ (Null-2 orthogonal scramble). "
        f"We treat a run as supporting H1 when $p_1<{alpha}$ and $p_2<{alpha}$ and $|S|>0$."
        "}\n"
    )
    tex.append(f"\\label{{{latex_escape(args.label)}}}\n")
    tex.append("\\end{table}\n")
    out_tex.write_text("".join(tex), encoding="utf-8")

    # Evidence-chain snippet (LaTeX)
    out_chain = out_dir / "H1_evidence_chain.tex"
    chain = (
        "% Auto-generated evidence-chain snippet for H1.\n"
        "\\paragraph{H1: shared decode-time workspace exists.} "
        "We test whether the decode-time shared set size $|S_\\ell(\\tau,m)|$ is larger than expected by chance under two nulls: "
        "(i) Null-1 (permutation of task-specific relative-variance scores) and "
        "(ii) Null-2 (per-task orthogonal scramble followed by re-estimation). "
        "We reject H1 when $|S_\\ell(\\tau,m)|$ is not in the tail of either null distribution; equivalently, when either $p_1$ or $p_2$ is not significant. "
        f"Table~\\ref{{{args.label}}} reports $|S|$, $p_1$, and $p_2$ across the full/pooled/loosened settings.\n"
    )
    out_chain.write_text(chain, encoding="utf-8")

    print("[OK] wrote:")
    print(" -", out_csv)
    print(" -", out_md)
    print(" -", out_tex)
    print(" -", out_chain)


if __name__ == "__main__":
    main()
