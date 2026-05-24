"""
summarize_multibench_full.py

Summarize multibench steering-control outputs into:
  - a rich Markdown report (includes candidate calibration details, per-template breakdown)
  - LaTeX tables (main + deltas + per-template appendix tables)
  - information-rich plots (beta curves + per-template heatmaps + candidate bars)

Expected directory layout (example):
  <root>/
    run_config.json
    aggregate_summary.csv
    boolq/
      boolq_summary.csv
      boolq_diag.json
      boolq_report.json
    rte/
      rte_summary.csv
      rte_diag.json
      rte_report.json
    sst2/
      sst2_summary.csv
      sst2_diag.json
      sst2_report.json
    summary_pack/   (optional; recommended output directory)

Usage:
  python summarize_multibench_full.py --root_dir results/steer_repair_multibench
  python summarize_multibench_full.py --root_dir results/steer_repair_multibench --out_dir results/steer_repair_multibench/summary_pack

Requires: pandas, matplotlib

Notes:
  - This script reads *_report.json to recover per-template stats for each method.
  - It also reads *_diag.json to include candidate-calibration ranks and sharedness(beta).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt


METHOD_BETA_RE = re.compile(r"^decode_beta([0-9.]+)$")


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def fmt(x: float, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "nan"
    return f"{x:.{nd}f}"


def latex_escape(s: str) -> str:
    return (str(s)
            .replace("\\", "\\\\")
            .replace("_", "\\_")
            .replace("%", "\\%")
            .replace("&", "\\&")
            .replace("#", "\\#")
            .replace("{", "\\{")
            .replace("}", "\\}")
            )


def method_to_beta(method: str) -> Optional[float]:
    if method == "decode_est":
        return 0.0
    if method == "decode_fixed":
        return 1.0
    m = METHOD_BETA_RE.match(method)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def beta_to_method(beta: float) -> str:
    if abs(beta - 0.0) < 1e-9:
        return "decode_est"
    if abs(beta - 1.0) < 1e-9:
        return "decode_fixed"

    if abs(beta - 0.25) < 1e-9:
        return "decode_beta0.25"
    if abs(beta - 0.5) < 1e-9:
        return "decode_beta0.5"
    if abs(beta - 0.75) < 1e-9:
        return "decode_beta0.75"
    s = f"{beta:.3f}".rstrip("0").rstrip(".")
    return f"decode_beta{s}"


def choose_mid_beta(methods: List[str], target: float = 0.5) -> Optional[str]:
    cands: List[Tuple[float, str]] = []
    for m in methods:
        b = method_to_beta(m)
        if b is not None:
            cands.append((abs(b - target), m))
    if not cands:
        return None
    cands.sort(key=lambda x: x[0])
    return cands[0][1]


def savefig(path: Path) -> None:
    try:
        plt.tight_layout()
    except Exception:
        pass
    try:
        plt.savefig(path, format="pdf", dpi=300)
    except Exception:

        png = path.with_suffix(".png")
        plt.savefig(png, dpi=300)
    plt.close()


def load_task(task_dir: Path) -> Tuple[pd.DataFrame, dict, dict]:
    """
    Load:
      <task>_summary.csv  (aggregated)
      <task>_diag.json    (candidate calibration, sharedness per beta)
      <task>_report.json  (per-template stats per method)

    Returns: (df_summary, diag, report)
    """
    task = task_dir.name
    summ_path = task_dir / f"{task}_summary.csv"
    diag_path = task_dir / f"{task}_diag.json"
    rep_path = task_dir / f"{task}_report.json"

    for p in (summ_path, diag_path, rep_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    df = pd.read_csv(summ_path)
    diag = read_json(diag_path)
    report = read_json(rep_path)


    df["beta"] = df["method"].apply(method_to_beta)


    sh_map: Dict[str, float] = {}
    beta_to_sh: Dict[float, float] = {}
    for dv in diag.get("decode_variants", []):
        name = dv.get("name")
        if name is None:
            continue
        b = float(dv.get("beta"))
        sh = float(dv.get("sharedness"))
        sh_map[str(name)] = sh
        beta_to_sh[b] = sh

    df["sharedness_method"] = df["method"].map(lambda m: sh_map.get(str(m), float("nan")))
    df["sharedness_decode_est"] = float(diag.get("sharedness_decode_est", float("nan")))


    chosen = diag.get("cand_calibration", {}).get("chosen", {})
    df["cand_calib_acc"] = float(chosen.get("acc", float("nan")))


    diag["_beta_to_sharedness"] = beta_to_sh

    return df, diag, report


def collect_all(root_dir: Path) -> Tuple[pd.DataFrame, Dict[str, dict], Dict[str, dict], Dict[str, dict]]:
    """
    Returns:
      all_df: concatenated summary.csv across tasks
      diags: task -> diag
      reports: task -> report
      meta: run_config + (optional) aggregate_summary
    """
    diags: Dict[str, dict] = {}
    reports: Dict[str, dict] = {}
    dfs: List[pd.DataFrame] = []

    for child in sorted(root_dir.iterdir()):
        if not child.is_dir():
            continue
        task = child.name
        summ = child / f"{task}_summary.csv"
        diag = child / f"{task}_diag.json"
        rep = child / f"{task}_report.json"
        if summ.exists() and diag.exists() and rep.exists():
            df, dj, rj = load_task(child)
            dfs.append(df)
            diags[task] = dj
            reports[task] = rj

    if not dfs:
        raise RuntimeError(
            f"No task outputs found under {root_dir}. Expected <task>/<task>_summary.csv, _diag.json, _report.json"
        )

    all_df = pd.concat(dfs, axis=0, ignore_index=True)

    meta: Dict[str, dict] = {}
    run_cfg = root_dir / "run_config.json"
    if run_cfg.exists():
        meta["run_config"] = read_json(run_cfg)
    agg = root_dir / "aggregate_summary.csv"
    if agg.exists():
        try:
            meta["aggregate_summary"] = pd.read_csv(agg)
        except Exception:
            pass

    return all_df, diags, reports, meta


def per_template_df_from_report(report: dict) -> pd.DataFrame:
    """
    report["methods"][method]["per_template_last"] is a list of per-template dicts.
    Returns DF with columns: method, template_id, mean, std, anti.
    """
    rows = []
    methods = report.get("methods", {})
    for mname, mrep in methods.items():
        pt = mrep.get("per_template_last", [])
        for tid, tstat in enumerate(pt):
            rows.append({
                "method": mname,
                "template_id": tid,
                "mean": float(tstat.get("mean", float("nan"))),
                "std": float(tstat.get("std", float("nan"))),
                "anti": float(tstat.get("anti", float("nan"))),
            })
    return pd.DataFrame(rows)


def worst_template_id(pt_df: pd.DataFrame, method: str) -> Optional[int]:
    d = pt_df[pt_df["method"] == method]
    if d.empty:
        return None

    idx = d["mean"].idxmin()
    return int(d.loc[idx, "template_id"])


def select_methods_for_main(df_task: pd.DataFrame) -> List[str]:
    """
    Select decode_est, mid-beta (~0.5), decode_fixed, rand_matched if present.
    """
    methods = sorted(df_task["method"].unique().tolist())
    sel: List[str] = []
    if "decode_est" in methods:
        sel.append("decode_est")
    mid = choose_mid_beta(methods, target=0.5)
    if mid and mid not in sel:
        sel.append(mid)
    if "decode_fixed" in methods:
        sel.append("decode_fixed")
    if "rand_matched" in methods:
        sel.append("rand_matched")
    return sel


def recommend_beta(df_task: pd.DataFrame, prefer: str = "worst_then_anti", min_mu_frac: float = 0.9) -> Optional[float]:
    """
    Recommend a beta (among decode_* methods) by a simple robust criterion.
    - Compute baseline mu0 = mu(beta=0).
    - Filter betas with mu >= min_mu_frac * mu0 (avoid too much efficacy loss).
    - Among remaining, maximize worst_case_mean; tie-break by minimizing anti_worst, then min std.
    Returns recommended beta, or None.
    """

    d = df_task[df_task["beta"].notna()].copy()
    if d.empty:
        return None
    mu0_row = d[d["beta"] == 0.0]
    if mu0_row.empty:
        return None
    mu0 = float(mu0_row.iloc[0]["mean_of_means"])
    thr = min_mu_frac * mu0
    c = d[d["mean_of_means"] >= thr].copy()
    if c.empty:
        c = d.copy()


    c = c.sort_values(
        by=["worst_case_mean", "anti_worst", "std_across_templates", "mean_of_means"],
        ascending=[False, True, True, False],
    )
    return float(c.iloc[0]["beta"])


def write_latex_tables(all_df: pd.DataFrame, diags: Dict[str, dict], reports: Dict[str, dict], out_tex: Path) -> None:
    """
    Write:
      - main multibench table (selected methods)
      - delta table (decode_fixed - decode_est)
      - per-template appendix tables (selected methods per task)
      - candidate calibration top-k table per task (top 5)
    """
    lines: List[str] = []
    lines.append("% Auto-generated by summarize_multibench_full.py")
    lines.append("% Requires: \\usepackage{booktabs,multirow}")
    lines.append("")


    main_rows = []
    for task in sorted(all_df["task"].unique().tolist()):
        dft = all_df[all_df["task"] == task].copy()
        sel = select_methods_for_main(dft)
        dft = dft[dft["method"].isin(sel)].copy()
        order = {m: i for i, m in enumerate(sel)}
        dft["_ord"] = dft["method"].map(order)
        dft = dft.sort_values("_ord").drop(columns=["_ord"])
        main_rows.append(dft)
    main_df = pd.concat(main_rows, axis=0, ignore_index=True)


    task_info = {}
    for task in sorted(main_df["task"].unique().tolist()):
        row0 = main_df[main_df["task"] == task].iloc[0]
        cand_name = row0.get("cand_name", "")
        cand_pos = row0.get("cand_pos", "")
        cand_neg = row0.get("cand_neg", "")
        cand = f"{cand_name} ({cand_pos}/{cand_neg})"
        sh = float(row0.get("sharedness_decode_est", float("nan")))
        task_info[task] = {"cand": cand, "sh": sh}

    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{5pt}")
    lines.append("\\begin{tabular}{lll l rrrr}")
    lines.append("\\toprule")
    lines.append("Task & Candidates & $\\mathrm{sh}(v)$ & Method & $\\mu$ & $\\sigma_{\\text{tmpl}}$ & worst & anti$_{\\text{worst}}$ \\\\")
    lines.append("\\midrule")

    for task in sorted(main_df["task"].unique().tolist()):
        dft = main_df[main_df["task"] == task].copy()
        cand = latex_escape(task_info[task]["cand"])
        shv = fmt(task_info[task]["sh"], 3)
        nrows = len(dft)
        first = True
        for _, r in dft.iterrows():
            method = r["method"]
            b = method_to_beta(method)
            if method == "decode_est":
                mdisp = "decode\\_est ($\\beta{=}0$)"
            elif method == "decode_fixed":
                mdisp = "decode\\_fixed ($\\beta{=}1$)"
            elif method == "rand_matched":
                mdisp = "rand\\_matched"
            elif b is not None:
                mdisp = f"decode\\_beta{fmt(b,2)} ($\\beta={fmt(b,2)}$)"
                mdisp = mdisp.replace("0.50", "0.5").replace("0.00", "0").replace("1.00", "1")
            else:
                mdisp = latex_escape(method)

            mu = fmt(float(r["mean_of_means"]), 4)
            sig = fmt(float(r["std_across_templates"]), 4)
            worst = fmt(float(r["worst_case_mean"]), 4)
            antiw = fmt(float(r["anti_worst"]), 4)

            if first:
                lines.append(f"\\multirow{{{nrows}}}{{*}}{{{latex_escape(task)}}} & \\multirow{{{nrows}}}{{*}}{{{cand}}} & \\multirow{{{nrows}}}{{*}}{{{shv}}} & {mdisp} & {mu} & {sig} & {worst} & {antiw} \\\\")
                first = False
            else:
                lines.append(f" &  &  & {mdisp} & {mu} & {sig} & {worst} & {antiw} \\\\")
        lines.append("\\midrule")
    if lines[-1] == "\\midrule":
        lines[-1] = "\\bottomrule"
    else:
        lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{\\textbf{Multibench robustness under KV-cache decode.} We report correct-signed margin shift aggregated across prompt templates: mean $\\mu$, template std $\\sigma_{\\text{tmpl}}$, worst-case template mean (worst), and worst-case anti-steer rate (anti$_{\\text{worst}}$; lower is better). $\\mathrm{sh}(v)=\\|B^\\top v\\|/\\|v\\|$ is the overlap between the steering direction and the decode-time shared basis (estimated from neutral prompts). Partial projection uses $v_\\beta = v - \\beta BB^\\top v$.}")
    lines.append("\\label{tab:multibench}")
    lines.append("\\end{table}")
    lines.append("")


    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{6pt}")
    lines.append("\\begin{tabular}{lrrrr}")
    lines.append("\\toprule")
    lines.append("Task & $\\Delta \\mu$ & $\\Delta \\sigma_{\\text{tmpl}}$ & $\\Delta$worst & $\\Delta$anti$_{\\text{worst}}$ \\\\")
    lines.append("\\midrule")
    for task in sorted(all_df["task"].unique().tolist()):
        dft = all_df[all_df["task"] == task]
        if not ("decode_est" in dft["method"].values and "decode_fixed" in dft["method"].values):
            continue
        est = dft[dft["method"] == "decode_est"].iloc[0]
        fix = dft[dft["method"] == "decode_fixed"].iloc[0]
        d_mu = float(fix["mean_of_means"]) - float(est["mean_of_means"])
        d_sig = float(fix["std_across_templates"]) - float(est["std_across_templates"])
        d_worst = float(fix["worst_case_mean"]) - float(est["worst_case_mean"])
        d_anti = float(fix["anti_worst"]) - float(est["anti_worst"])
        lines.append(f"{latex_escape(task)} & {fmt(d_mu,4)} & {fmt(d_sig,4)} & {fmt(d_worst,4)} & {fmt(d_anti,4)} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{\\textbf{Effect of fully removing the shared component of the steering vector.} $\\Delta$ compares decode\\_fixed ($\\beta{=}1$) vs decode\\_est ($\\beta{=}0$). Negative $\\Delta$anti$_{\\text{worst}}$ indicates fewer worst-case failures.}")
    lines.append("\\label{tab:multibench_delta}")
    lines.append("\\end{table}")
    lines.append("")


    for task in sorted(reports.keys()):
        report = reports[task]
        pt = per_template_df_from_report(report)

        dft = all_df[all_df["task"] == task]
        sel = select_methods_for_main(dft)
        sel_no_rand = [m for m in sel if m != "rand_matched"]
        pt = pt[pt["method"].isin(sel_no_rand)].copy()
        if pt.empty:
            continue

        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        lines.append("\\small")
        lines.append("\\setlength{\\tabcolsep}{5pt}")

        methods = sel_no_rand
        cols = "l" + "r" * (len(methods) * 2)
        lines.append(f"\\begin{{tabular}}{{{cols}}}")
        lines.append("\\toprule")
        header1 = ["Template"]
        for m in methods:
            header1 += [latex_escape(m) + " mean", latex_escape(m) + " anti"]
        lines.append(" & ".join(header1) + " \\\\")
        lines.append("\\midrule")

        n_templates = int(report.get("n_templates", pt["template_id"].max() + 1 if not pt.empty else 0))
        for tid in range(n_templates):
            row = [f"T{tid}"]
            for m in methods:
                r = pt[(pt["template_id"] == tid) & (pt["method"] == m)]
                if r.empty:
                    row += ["--", "--"]
                else:
                    row += [fmt(float(r.iloc[0]["mean"]), 4), fmt(float(r.iloc[0]["anti"]), 4)]
            lines.append(" & ".join(row) + " \\\\")
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append(f"\\caption{{\\textbf{{Per-template breakdown for {latex_escape(task)}.}} Mean correct-signed margin shift and failure rate (anti) at the final $\\lambda$ for selected methods.}}")
        lines.append(f"\\label{{tab:multibench_{latex_escape(task)}_pertemplate}}")
        lines.append("\\end{table}")
        lines.append("")


    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{6pt}")
    lines.append("\\begin{tabular}{llrr}")
    lines.append("\\toprule")
    lines.append("Task & Candidate pair & Baseline acc & Single-token \\\\")
    lines.append("\\midrule")
    for task in sorted(diags.keys()):
        dj = diags[task]
        tested = dj.get("cand_calibration", {}).get("candidates_tested", [])
        topk = tested[:5] if isinstance(tested, list) else []
        if not topk:
            continue
        for i, c in enumerate(topk):
            pair = f"{c.get('pos','?')}/{c.get('neg','?')} ({c.get('name','?')})"
            acc = float(c.get("acc", float("nan")))
            st = 1 if bool(c.get("single_token", False)) else 0
            task_cell = latex_escape(task) if i == 0 else ""
            lines.append(f"{task_cell} & {latex_escape(pair)} & {fmt(acc,3)} & {st} \\\\")
        lines.append("\\midrule")
    if lines[-1] == "\\midrule":
        lines[-1] = "\\bottomrule"
    else:
        lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{\\textbf{Candidate calibration (top-5).} For each task, we select a single-token forced-choice pair by maximizing baseline forced-choice accuracy on a small balanced subset.}")
    lines.append("\\label{tab:cand_calib_top5}")
    lines.append("\\end{table}")

    out_tex.write_text("\n".join(lines), encoding="utf-8")


def write_markdown_report(
    all_df: pd.DataFrame,
    diags: Dict[str, dict],
    reports: Dict[str, dict],
    meta: Dict[str, dict],
    out_md: Path,
    tex_filename: str,
    figs: List[str],
) -> None:
    lines: List[str] = []
    lines.append("# Multibench summary (full)\n")


    run_cfg = meta.get("run_config", {})
    if run_cfg:
        lines.append("## Run config\n")
        keys = ["model", "dtype", "device", "layer", "basis_source", "basis_k", "basis_max_states", "v_est_templates", "betas", "lambdas", "n_rand"]
        for k in keys:
            if k in run_cfg:
                lines.append(f"- **{k}**: `{run_cfg[k]}`")
        lines.append("")

    tasks = sorted(diags.keys())
    lines.append("## Tasks\n")
    lines.append(", ".join(tasks) + "\n")


    lines.append("## Candidate calibration\n")
    lines.append("Each task chooses a forced-choice candidate pair by maximizing baseline forced-choice accuracy on a small balanced subset.\n")
    lines.append("| Task | Chosen | Baseline acc | sh(v) | Top-3 candidates (acc) |")
    lines.append("|---|---|---:|---:|---|")
    for task in tasks:
        dj = diags[task]
        cand = dj.get("cand", {})
        chosen = dj.get("cand_calibration", {}).get("chosen", {})
        acc = float(chosen.get("acc", float("nan")))
        sh = float(dj.get("sharedness_decode_est", float("nan")))
        tested = dj.get("cand_calibration", {}).get("candidates_tested", [])
        top3 = []
        if isinstance(tested, list):
            for c in tested[:3]:
                top3.append(f"{c.get('pos','?')}/{c.get('neg','?')}({fmt(float(c.get('acc',0.0)),3)})")
        chosen_str = f"{cand.get('name','?')} ({cand.get('pos','?')}/{cand.get('neg','?')})"
        lines.append(f"| {task} | {chosen_str} | {acc:.3f} | {sh:.3f} | " + ", ".join(top3) + " |")
    lines.append("")


    lines.append("## Main results (template robustness)\n")
    lines.append("Metrics are computed on correct-signed margin shift aggregated across templates at the final $\\lambda$: mean $\\mu$, template std $\\sigma_{tmpl}$, worst-case template mean, and worst-case anti-steer rate (lower is better).\n")
    lines.append("| Task | Method | beta | mu | sigma_tmpl | worst | anti_worst | worst template id |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for task in tasks:
        dft = all_df[all_df["task"] == task].copy()
        sel = select_methods_for_main(dft)

        pt = per_template_df_from_report(reports[task])
        for m in sel:
            row = dft[dft["method"] == m]
            if row.empty:
                continue
            r = row.iloc[0]
            b = method_to_beta(m)
            b_disp = "" if b is None else f"{b:.2f}".rstrip("0").rstrip(".")
            worst_tid = worst_template_id(pt, m)
            lines.append(
                f"| {task} | {m} | {b_disp if b_disp else ''} | "
                f"{float(r['mean_of_means']):.4f} | {float(r['std_across_templates']):.4f} | "
                f"{float(r['worst_case_mean']):.4f} | {float(r['anti_worst']):.4f} | "
                f"{worst_tid if worst_tid is not None else ''} |"
            )
    lines.append("")


    lines.append("## Recommended beta (simple robust heuristic)\n")
    lines.append("We recommend a beta per task by maximizing worst-case mean shift while preserving at least 90\\% of the baseline mean shift (beta=0). This is a reporting aid (not used to tune results).\n")
    lines.append("| Task | recommended beta | recommended method |")
    lines.append("|---|---:|---|")
    for task in tasks:
        dft = all_df[all_df["task"] == task].copy()
        b = recommend_beta(dft, min_mu_frac=0.9)
        if b is None:
            lines.append(f"| {task} |  |  |")
        else:
            lines.append(f"| {task} | {b:.2f} | {beta_to_method(b)} |")
    lines.append("")


    lines.append("## Per-template breakdown (selected methods)\n")
    for task in tasks:
        report = reports[task]
        pt = per_template_df_from_report(report)
        dft = all_df[all_df["task"] == task]
        sel = select_methods_for_main(dft)
        sel_no_rand = [m for m in sel if m != "rand_matched"]
        pt = pt[pt["method"].isin(sel_no_rand)].copy()
        if pt.empty:
            continue
        lines.append(f"### {task}\n")
        lines.append("| template | " + " | ".join([f"{m} mean" for m in sel_no_rand]) + " |")
        lines.append("|---|" + "|".join(["---:"] * len(sel_no_rand)) + "|")
        n_templates = int(report.get("n_templates", pt["template_id"].max() + 1))
        for tid in range(n_templates):
            row = [f"T{tid}"]
            for m in sel_no_rand:
                r = pt[(pt["template_id"] == tid) & (pt["method"] == m)]
                row.append(fmt(float(r.iloc[0]["mean"]), 4) if not r.empty else "")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
        lines.append("| template | " + " | ".join([f"{m} anti" for m in sel_no_rand]) + " |")
        lines.append("|---|" + "|".join(["---:"] * len(sel_no_rand)) + "|")
        for tid in range(n_templates):
            row = [f"T{tid}"]
            for m in sel_no_rand:
                r = pt[(pt["template_id"] == tid) & (pt["method"] == m)]
                row.append(fmt(float(r.iloc[0]["anti"]), 4) if not r.empty else "")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")


    lines.append("## LaTeX tables\n")
    lines.append(f"Written to `{tex_filename}` (requires `booktabs` and `multirow`).\n")
    lines.append("## Figures\n")
    for fn in figs:
        lines.append(f"- `{fn}`")
    lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")


def plot_metric_vs_beta(all_df: pd.DataFrame, metric: str, out_path: Path, ylabel: str, title: str) -> None:
    plt.figure()
    for task in sorted(all_df["task"].unique().tolist()):
        dft = all_df[(all_df["task"] == task) & (all_df["beta"].notna())].copy()
        if dft.empty:
            continue
        dft = dft.sort_values("beta")
        plt.plot(dft["beta"].values, dft[metric].values, marker="o", label=task)
    plt.xlabel("beta")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    savefig(out_path)


def plot_metric_vs_beta_with_baseline(all_df: pd.DataFrame, metric: str, out_path: Path, ylabel: str, title: str) -> None:
    """
    Like plot_metric_vs_beta but also overlays rand_matched per task as horizontal dashed line.
    """
    plt.figure()
    for task in sorted(all_df["task"].unique().tolist()):
        dft = all_df[(all_df["task"] == task) & (all_df["beta"].notna())].copy()
        if dft.empty:
            continue
        dft = dft.sort_values("beta")
        plt.plot(dft["beta"].values, dft[metric].values, marker="o", label=f"{task} decode")

        rnd = all_df[(all_df["task"] == task) & (all_df["method"] == "rand_matched")]
        if not rnd.empty:
            y = float(rnd.iloc[0][metric])
            plt.axhline(y, linestyle="--", linewidth=1.0)

    plt.xlabel("beta")
    plt.ylabel(ylabel)
    plt.title(title + " (dashed=rand)")
    plt.legend()
    savefig(out_path)


def plot_sharedness_vs_beta(diags: Dict[str, dict], out_path: Path) -> None:
    plt.figure()
    for task in sorted(diags.keys()):
        beta_to_sh = diags[task].get("_beta_to_sharedness", {})
        if not beta_to_sh:
            continue
        xs = sorted(beta_to_sh.keys())
        ys = [beta_to_sh[x] for x in xs]
        plt.plot(xs, ys, marker="o", label=task)
    plt.xlabel("beta")
    plt.ylabel("sharedness(v_beta)")
    plt.title("Sharedness vs beta")
    plt.legend()
    savefig(out_path)


def plot_sharedness_vs_metric(all_df: pd.DataFrame, metric: str, out_path: Path, ylabel: str, title: str) -> None:
    plt.figure()
    for task in sorted(all_df["task"].unique().tolist()):
        dft = all_df[(all_df["task"] == task) & (all_df["beta"].notna())].copy()
        if dft.empty:
            continue
        dft = dft.sort_values("beta")
        plt.plot(dft["sharedness_method"].values, dft[metric].values, marker="o", label=task)
    plt.xlabel("sharedness(v_beta)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    savefig(out_path)


def plot_candidate_bars(diag: dict, task: str, out_path: Path, topk: int = 8) -> None:
    tested = diag.get("cand_calibration", {}).get("candidates_tested", [])
    if not isinstance(tested, list) or not tested:
        return
    top = tested[:topk]
    labels = [f"{c.get('pos','?')}/{c.get('neg','?')}" for c in top]
    accs = [float(c.get("acc", 0.0)) for c in top]
    chosen = diag.get("cand_calibration", {}).get("chosen", {})
    chosen_pair = f"{chosen.get('pos','?')}/{chosen.get('neg','?')}"
    colors = ["C0" if lab != chosen_pair else "C3" for lab in labels]

    plt.figure()
    plt.bar(range(len(labels)), accs, color=colors)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel("baseline forced-choice acc")
    plt.title(f"{task}: candidate calibration (top-{len(labels)})")
    savefig(out_path)


def plot_template_heatmap(pt_df: pd.DataFrame, methods: List[str], task: str, value_col: str, out_path: Path, title: str) -> None:
    """
    Heatmap with y=template_id, x=method(beta order), value = mean or anti.
    """
    d = pt_df[pt_df["method"].isin(methods)].copy()
    if d.empty:
        return


    def key(m):
        b = method_to_beta(m)
        return 9.0 if b is None else b
    methods_sorted = sorted(methods, key=key)


    piv = d.pivot_table(index="template_id", columns="method", values=value_col, aggfunc="first")
    piv = piv.reindex(columns=methods_sorted)

    plt.figure()
    plt.imshow(piv.values, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(piv.columns)), [c.replace("decode_", "") for c in piv.columns], rotation=45, ha="right")
    plt.yticks(range(len(piv.index)), [f"T{int(i)}" for i in piv.index])
    plt.xlabel("method / beta")
    plt.ylabel("template")
    plt.title(title)
    savefig(out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", type=str, required=True, help="Output dir containing per-task subdirs")
    ap.add_argument("--out_dir", type=str, default="", help="output dir (default: <root>/summary_pack)")
    args = ap.parse_args()

    root = Path(args.root_dir)
    if not root.exists():
        raise SystemExit(f"root_dir does not exist: {root}")

    out = Path(args.out_dir) if args.out_dir.strip() else (root / "summary_pack")
    safe_mkdir(out)

    all_df, diags, reports, meta = collect_all(root)


    tex_path = out / "tables_multibench_full.tex"
    write_latex_tables(all_df, diags, reports, tex_path)

    figs: List[str] = []


    p = out / "fig_beta_vs_mean_of_means.pdf"
    plot_metric_vs_beta_with_baseline(all_df, "mean_of_means", p, ylabel="mean of means", title="Effect vs beta")
    figs.append(p.name)

    p = out / "fig_beta_vs_worst_case_mean.pdf"
    plot_metric_vs_beta_with_baseline(all_df, "worst_case_mean", p, ylabel="worst-case template mean", title="Worst-case vs beta")
    figs.append(p.name)

    p = out / "fig_beta_vs_anti_worst.pdf"
    plot_metric_vs_beta_with_baseline(all_df, "anti_worst", p, ylabel="anti-worst (failure rate)", title="Worst-case failure vs beta")
    figs.append(p.name)

    p = out / "fig_beta_vs_sigma_tmpl.pdf"
    plot_metric_vs_beta(all_df, "std_across_templates", p, ylabel="template std", title="Template sensitivity vs beta")
    figs.append(p.name)

    p = out / "fig_beta_vs_slope.pdf"
    if "slope" in all_df.columns:
        plot_metric_vs_beta_with_baseline(all_df, "slope", p, ylabel="slope vs lambda", title="Steering strength vs beta")
        figs.append(p.name)


    p = out / "fig_beta_vs_sharedness.pdf"
    plot_sharedness_vs_beta(diags, p)
    figs.append(p.name)


    p = out / "fig_sharedness_vs_worst.pdf"
    plot_sharedness_vs_metric(all_df, "worst_case_mean", p, ylabel="worst-case mean", title="Worst-case vs sharedness")
    figs.append(p.name)

    p = out / "fig_sharedness_vs_anti_worst.pdf"
    plot_sharedness_vs_metric(all_df, "anti_worst", p, ylabel="anti-worst", title="Worst-case failure vs sharedness")
    figs.append(p.name)


    for task in sorted(diags.keys()):

        p = out / f"fig_candidate_acc_{task}.pdf"
        plot_candidate_bars(diags[task], task, p, topk=8)
        figs.append(p.name)


        pt = per_template_df_from_report(reports[task])

        methods = sorted([m for m in pt["method"].unique().tolist() if m.startswith("decode_")])
        if methods:
            p = out / f"fig_heatmap_mean_{task}.pdf"
            plot_template_heatmap(pt, methods, task, value_col="mean", out_path=p, title=f"{task}: per-template mean shift")
            figs.append(p.name)

            p = out / f"fig_heatmap_anti_{task}.pdf"
            plot_template_heatmap(pt, methods, task, value_col="anti", out_path=p, title=f"{task}: per-template anti (failure rate)")
            figs.append(p.name)


    md_path = out / "summary_multibench_full.md"
    write_markdown_report(
        all_df=all_df,
        diags=diags,
        reports=reports,
        meta=meta,
        out_md=md_path,
        tex_filename=tex_path.name,
        figs=figs,
    )

    print(f"Wrote: {md_path}")
    print(f"Wrote: {tex_path}")
    for fn in figs:
        print(f"Wrote: {out / fn}")


if __name__ == "__main__":
    main()
