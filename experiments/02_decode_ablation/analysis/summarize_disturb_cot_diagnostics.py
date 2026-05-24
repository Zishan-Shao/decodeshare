#!/usr/bin/env python3

"""
summarize_disturb_cot_diagnostics.py

Diagnostic summary for disturb_CoT_* results.

Summarizes (per task):
  - extraction_rate
  - eos_rate
  - avg_new_tokens
  - (optional) accuracy
  - heuristic flags for "parse failure" / "output behavior shift"

Also supports reading confidence intervals (CI) for staged runs (and others),
IF the JSON stores CI fields inside each run dict.

Expected structure (same as your current script):
  block["runs"][f"{decoding}/{mode}"] is a dict containing metrics.

Modes:
  baseline / shared_full / shared_staged / rand_full / rand_staged

Usage example:
  python analysis/summarize_disturb_cot_diagnostics.py \
    --results_dir ../../outputs/02_decode_ablation/loto \
    --pattern "*.json" \
    --output ../../outputs/02_decode_ablation/loto/DIAGNOSTIC_SUMMARY.md
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict


def fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return "N/A"
    return f"{x * 100:.1f}"

def fmt_len(x: Optional[float]) -> str:
    if x is None:
        return "N/A"
    return f"{x:.1f}"

def fmt_pp(delta: Optional[float]) -> str:
    if delta is None:
        return "N/A"
    return f"{delta * 100:+.1f}pp"

def fmt_delta(x: Optional[float], x0: Optional[float], kind: str) -> str:
    """
    kind:
      - "pp" : (x - x0) in percentage points
      - "raw": (x - x0) raw
      - "ratio": x/x0
    """
    if x is None or x0 is None:
        return "N/A"
    if kind == "pp":
        return fmt_pp(x - x0)
    if kind == "raw":
        return f"{(x - x0):+.2f}"
    if kind == "ratio":
        if abs(x0) < 1e-12:
            return "N/A"
        return f"{(x / x0):.2f}x"
    raise ValueError(kind)

def render_md_table(header: List[str], rows: List[List[str]]) -> str:
    if not rows:
        return "No data available.\n"
    cols = list(zip(*([header] + rows)))
    widths = [max(len(str(v)) for v in col) for col in cols]

    def fmt_row(r: List[str]) -> str:
        return "| " + " | ".join(str(v).ljust(w) for v, w in zip(r, widths)) + " |"

    sep = "|-" + "-|-".join("-" * w for w in widths) + "-|"
    out = [fmt_row(header), sep]
    out.extend(fmt_row(r) for r in rows)
    return "\n".join(out) + "\n"


def _maybe_div100(x: Optional[float]) -> Optional[float]:
    """If x looks like percent (e.g., 78.7), convert to proportion (0.787)."""
    if x is None:
        return None
    try:
        x = float(x)
    except Exception:
        return None
    return x / 100.0 if x > 1.5 else x

def _normalize_ci(ci: Any, *, is_prob: bool) -> Optional[Tuple[float, float]]:
    """
    Accept CI formats:
      - [lo, hi] (proportion 0..1) or (percent 0..100)
      - {"low":..., "high":...} or {"ci_low":..., "ci_high":...}
    """
    if ci is None:
        return None

    lo = hi = None
    if isinstance(ci, (list, tuple)) and len(ci) == 2:
        lo, hi = ci[0], ci[1]
    elif isinstance(ci, dict):
        if "low" in ci and "high" in ci:
            lo, hi = ci["low"], ci["high"]
        elif "ci_low" in ci and "ci_high" in ci:
            lo, hi = ci["ci_low"], ci["ci_high"]

    try:
        lo = float(lo) if lo is not None else None
        hi = float(hi) if hi is not None else None
    except Exception:
        return None

    if lo is None or hi is None:
        return None


    if is_prob and (lo > 1.5 or hi > 1.5):
        lo /= 100.0
        hi /= 100.0

    return (lo, hi)

def _pick_ci_field(r: Dict[str, Any], metric: str) -> Optional[Tuple[float, float]]:
    """
    Tries common CI key names.
    """

    cand_keys = [
        f"{metric}_ci",
        f"{metric}CI",
        f"{metric}_confint",
    ]


    if metric == "accuracy":
        cand_keys = ["accuracy_ci", "acc_ci", "ci", "acc_ci95", "accuracy_ci95"] + cand_keys
    elif metric == "extraction_rate":
        cand_keys = ["extraction_rate_ci", "extr_ci", "extraction_ci", "ci_extraction", "ci_extr"] + cand_keys
    elif metric == "eos_rate":
        cand_keys = ["eos_rate_ci", "eos_ci", "ci_eos"] + cand_keys
    elif metric == "avg_new_tokens":
        cand_keys = ["avg_new_tokens_ci", "len_ci", "avg_len_ci", "ci_len", "tokens_ci"] + cand_keys

    for k in cand_keys:
        if k in r:
            is_prob = metric in ("accuracy", "extraction_rate", "eos_rate")
            return _normalize_ci(r.get(k), is_prob=is_prob)

    return None

def fmt_ci_prob(ci: Optional[Tuple[float, float]]) -> str:
    if not ci:
        return ""
    lo, hi = ci
    return f"[{lo*100:.1f}, {hi*100:.1f}]"

def fmt_ci_len(ci: Optional[Tuple[float, float]]) -> str:
    if not ci:
        return ""
    lo, hi = ci
    return f"[{lo:.1f}, {hi:.1f}]"


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[Warn] failed to load {path}: {e}")
        return None

def short_model_name(raw: str) -> str:
    s = (raw or "").lower()
    if "llama" in s and "2" in s and "7b" in s:
        return "Llama-2-7b-chat-hf"
    if "qwen" in s and "2.5" in s and "7b" in s:
        return "Qwen2.5-7B-Instruct" if "instruct" in s else "Qwen2.5-7B"
    if "gemma" in s and "12b" in s:
        return "gemma-3-12b-it"
    return raw or "unknown"

def run_signature(cfg: Dict[str, Any]) -> str:

    layer = cfg.get("layer", cfg.get("layer_indices", ["?"]))
    if isinstance(layer, list):
        layer = layer[0] if layer else "?"
    parts = [
        f"mode={cfg.get('mode','?')}",
        f"loto_eval={cfg.get('loto_eval_mode','?')}",
        f"layer={layer}",
        f"tau={cfg.get('tau','?')}",
        f"m={cfg.get('m_shared','?')}",
        f"tr={int(bool(cfg.get('template_randomization', False)))}",
        f"sc={int(bool(cfg.get('shuffle_choices', False)))}",
        f"rand={cfg.get('rand_type','?')}",
        f"dtype={cfg.get('model_dtype', cfg.get('dtype','?'))}",
    ]
    return " | ".join(parts)


def get_task_blocks(results: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Return list of (task_key, block) where block has:
      - n
      - runs: dict
    """
    cfg = results.get("config", {})
    mode = cfg.get("mode", "unknown")

    out: List[Tuple[str, Dict[str, Any]]] = []

    if mode == "all":
        fold = results.get("all_tasks", {})
        by_dataset = fold.get("by_dataset", {})
        for task, block in sorted(by_dataset.items()):
            out.append((task, block))
        return out

    if mode == "loto":
        folds = results.get("folds", {})
        loto_eval_mode = cfg.get("loto_eval_mode", "heldout")
        for holdout, fold in sorted(folds.items()):
            by_dataset = fold.get("by_dataset", {})
            if loto_eval_mode == "heldout":
                block = by_dataset.get(holdout)
                if block is not None:
                    out.append((holdout, block))
            else:
                for task, block in sorted(by_dataset.items()):
                    out.append((f"{holdout}->{task}", block))
        return out


    if "all_tasks" in results:
        fold = results.get("all_tasks", {})
        by_dataset = fold.get("by_dataset", {})
        for task, block in sorted(by_dataset.items()):
            out.append((task, block))
    return out

def get_run_metrics(block: Dict[str, Any], decoding: str, mode: str) -> Optional[Dict[str, Any]]:
    """
    Returns dict:
      accuracy, extraction_rate, eos_rate, avg_new_tokens
      accuracy_ci, extraction_rate_ci, eos_rate_ci, avg_new_tokens_ci
    """
    runs = block.get("runs", {})
    key = f"{decoding}/{mode}"
    if key not in runs:
        return None

    r = runs[key] or {}

    acc = r.get("accuracy", r.get("acc", None))
    ex  = r.get("extraction_rate", r.get("extr", r.get("extraction", None)))
    eos = r.get("eos_rate", r.get("eos", None))
    ln  = r.get("avg_new_tokens", r.get("avg_len", r.get("length", None)))

    acc = _maybe_div100(acc)
    ex  = _maybe_div100(ex)
    eos = _maybe_div100(eos)

    ln_f = None
    if ln is not None:
        try:
            ln_f = float(ln)
        except Exception:
            ln_f = None

    return {
        "accuracy": acc,
        "accuracy_ci": _pick_ci_field(r, "accuracy"),
        "extraction_rate": ex,
        "extraction_rate_ci": _pick_ci_field(r, "extraction_rate"),
        "eos_rate": eos,
        "eos_rate_ci": _pick_ci_field(r, "eos_rate"),
        "avg_new_tokens": ln_f,
        "avg_new_tokens_ci": _pick_ci_field(r, "avg_new_tokens"),
    }


def diagnose_flags(
    base: Dict[str, Any],
    cond: Dict[str, Any],
    *,
    parse_drop_thresh: float,
    extr_floor: float,
    eos_delta_thresh: float,
    len_ratio_thresh: float,
    require_base_extr: float,
) -> List[str]:
    flags: List[str] = []

    b_ex = base.get("extraction_rate", None)
    c_ex = cond.get("extraction_rate", None)
    if b_ex is not None and c_ex is not None:
        if b_ex >= require_base_extr and (b_ex - c_ex) >= parse_drop_thresh:
            flags.append("PARSE_DROP")
        if c_ex <= extr_floor:
            flags.append("PARSE low")

    b_eos = base.get("eos_rate", None)
    c_eos = cond.get("eos_rate", None)
    if b_eos is not None and c_eos is not None:
        if abs(c_eos - b_eos) >= eos_delta_thresh:
            flags.append("EOS shift")

    b_len = base.get("avg_new_tokens", None)
    c_len = cond.get("avg_new_tokens", None)
    if b_len is not None and c_len is not None:
        if b_len > 1e-6 and (c_len / b_len) <= len_ratio_thresh:
            flags.append("LEN_DROP")


    seen = set()
    out = []
    for f in flags:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def summarize_one_file(
    results: Dict[str, Any],
    *,
    decoding: str,
    parse_drop_thresh: float,
    extr_floor: float,
    eos_delta_thresh: float,
    len_ratio_thresh: float,
    require_base_extr: float,
    include_accuracy: bool,
) -> str:
    cfg = results.get("config", {})
    model_raw = cfg.get("model", "unknown")
    model = short_model_name(model_raw)
    sig = run_signature(cfg)

    task_blocks = get_task_blocks(results)

    conds = ["baseline", "shared_full", "shared_staged", "rand_full", "rand_staged"]
    agg = {c: {"acc": [], "ex": [], "eos": [], "len": []} for c in conds}

    rows_ex: List[List[str]] = []
    rows_eos: List[List[str]] = []
    rows_len: List[List[str]] = []
    rows_acc: List[List[str]] = []


    flag_counts = {c: defaultdict(int) for c in ["shared_full", "shared_staged", "rand_full", "rand_staged"]}
    flagged_tasks = {c: 0 for c in ["shared_full", "shared_staged", "rand_full", "rand_staged"]}

    for task, block in task_blocks:
        n = block.get("n", None)
        n_str = str(n) if n is not None else "?"

        base = get_run_metrics(block, decoding, "baseline") or {}


        if base.get("accuracy") is not None: agg["baseline"]["acc"].append(float(base["accuracy"]))
        if base.get("extraction_rate") is not None: agg["baseline"]["ex"].append(float(base["extraction_rate"]))
        if base.get("eos_rate") is not None: agg["baseline"]["eos"].append(float(base["eos_rate"]))
        if base.get("avg_new_tokens") is not None: agg["baseline"]["len"].append(float(base["avg_new_tokens"]))

        def base_cell(metric_name: str) -> str:
            v = base.get(metric_name, None)
            ci = base.get(metric_name + "_ci", None)
            if metric_name in ("extraction_rate", "eos_rate", "accuracy"):
                main = fmt_pct(v)
                ci_str = (" " + fmt_ci_prob(ci)) if ci else ""
                return (main + ci_str).strip()
            if metric_name == "avg_new_tokens":
                main = fmt_len(v)
                ci_str = (" " + fmt_ci_len(ci)) if ci else ""
                return (main + ci_str).strip()
            return "N/A"

        def cond_cell(metric_name: str, cond_name: str) -> str:
            m = get_run_metrics(block, decoding, cond_name)
            if m is None:
                return "N/A"

            v = m.get(metric_name, None)
            v0 = base.get(metric_name, None)
            ci = m.get(metric_name + "_ci", None)

            if metric_name in ("extraction_rate", "eos_rate", "accuracy"):
                main = fmt_pct(v)
                ci_str = (" " + fmt_ci_prob(ci)) if ci else ""
                d = fmt_delta(v, v0, "pp")
                return f"{main}{ci_str} ({d})".strip()

            if metric_name == "avg_new_tokens":
                main = fmt_len(v)
                ci_str = (" " + fmt_ci_len(ci)) if ci else ""
                d = fmt_delta(v, v0, "raw")
                r_ = fmt_delta(v, v0, "ratio")
                return f"{main}{ci_str} ({d}, {r_})".strip()

            return "N/A"


        flags_sf: List[str] = []
        flags_ss: List[str] = []
        flags_rf: List[str] = []
        flags_rs: List[str] = []

        for c in ["shared_full", "shared_staged", "rand_full", "rand_staged"]:
            m = get_run_metrics(block, decoding, c)
            if m is None:
                continue


            if m.get("accuracy") is not None: agg[c]["acc"].append(float(m["accuracy"]))
            if m.get("extraction_rate") is not None: agg[c]["ex"].append(float(m["extraction_rate"]))
            if m.get("eos_rate") is not None: agg[c]["eos"].append(float(m["eos_rate"]))
            if m.get("avg_new_tokens") is not None: agg[c]["len"].append(float(m["avg_new_tokens"]))

            flags = diagnose_flags(
                base=base,
                cond=m,
                parse_drop_thresh=parse_drop_thresh,
                extr_floor=extr_floor,
                eos_delta_thresh=eos_delta_thresh,
                len_ratio_thresh=len_ratio_thresh,
                require_base_extr=require_base_extr,
            )
            if flags:
                flagged_tasks[c] += 1
            for f in flags:
                flag_counts[c][f] += 1

            if c == "shared_full":   flags_sf = flags
            if c == "shared_staged": flags_ss = flags
            if c == "rand_full":     flags_rf = flags
            if c == "rand_staged":   flags_rs = flags

        def flags_str(flags: List[str]) -> str:
            return ",".join(flags) if flags else "-"


        rows_ex.append([
            task, n_str,
            base_cell("extraction_rate"),
            cond_cell("extraction_rate", "shared_full"),
            cond_cell("extraction_rate", "shared_staged"),
            cond_cell("extraction_rate", "rand_full"),
            cond_cell("extraction_rate", "rand_staged"),
            f"SF:{flags_str(flags_sf)} SS:{flags_str(flags_ss)} RF:{flags_str(flags_rf)} RS:{flags_str(flags_rs)}",
        ])

        rows_eos.append([
            task, n_str,
            base_cell("eos_rate"),
            cond_cell("eos_rate", "shared_full"),
            cond_cell("eos_rate", "shared_staged"),
            cond_cell("eos_rate", "rand_full"),
            cond_cell("eos_rate", "rand_staged"),
        ])

        rows_len.append([
            task, n_str,
            base_cell("avg_new_tokens"),
            cond_cell("avg_new_tokens", "shared_full"),
            cond_cell("avg_new_tokens", "shared_staged"),
            cond_cell("avg_new_tokens", "rand_full"),
            cond_cell("avg_new_tokens", "rand_staged"),
        ])

        if include_accuracy:
            rows_acc.append([
                task, n_str,
                base_cell("accuracy"),
                cond_cell("accuracy", "shared_full"),
                cond_cell("accuracy", "shared_staged"),
                cond_cell("accuracy", "rand_full"),
                cond_cell("accuracy", "rand_staged"),
            ])


    def mean_or_na(xs: List[float]) -> str:
        if not xs:
            return "N/A"
        return f"{(sum(xs)/len(xs))*100:.1f}"

    def mean_len_or_na(xs: List[float]) -> str:
        if not xs:
            return "N/A"
        return f"{(sum(xs)/len(xs)):.1f}"

    overview_header = ["Condition", "Avg Acc(%)", "Avg Extr(%)", "Avg EOS(%)", "Avg Len"]
    overview_rows: List[List[str]] = []
    for c in ["baseline", "shared_full", "shared_staged", "rand_full", "rand_staged"]:
        overview_rows.append([
            c,
            mean_or_na(agg[c]["acc"]),
            mean_or_na(agg[c]["ex"]),
            mean_or_na(agg[c]["eos"]),
            mean_len_or_na(agg[c]["len"]),
        ])


    flag_header = ["Condition", "Total flagged tasks", "PARSE_DROP", "PARSE low", "EOS shift", "LEN_DROP"]
    flag_rows: List[List[str]] = []
    for c in ["shared_full", "shared_staged", "rand_full", "rand_staged"]:
        cnts = flag_counts[c]
        flag_rows.append([
            c,
            str(flagged_tasks[c]),
            str(cnts.get("PARSE_DROP", 0)),
            str(cnts.get("PARSE low", 0)),
            str(cnts.get("EOS shift", 0)),
            str(cnts.get("LEN_DROP", 0)),
        ])


    md: List[str] = []
    md.append(f"## Model: {model}\n")
    md.append(f"- **Model (raw)**: `{model_raw}`\n")
    md.append(f"- **Signature**: {sig}\n")
    md.append(f"- **Decoding**: `{decoding}`\n\n")

    md.append("### Overview (means across tasks)\n")
    md.append(render_md_table(overview_header, overview_rows))
    md.append("\n")

    md.append("### Diagnostic flags (heuristic counts)\n")
    md.append(
        f"- Heuristics: baseline_extr>={require_base_extr:.2f} & extr_drop>={parse_drop_thresh:.2f} -> `PARSE_DROP`; "
        f"extr<={extr_floor:.2f} -> `PARSE low`; |Delta eos|>={eos_delta_thresh:.2f} -> `EOS shift`; "
        f"len_ratio<={len_ratio_thresh:.2f} -> `LEN_DROP`\n\n"
    )
    md.append(render_md_table(flag_header, flag_rows))
    md.append("\n")

    md.append("### Extraction rate per task (%, with CI if available; Delta vs baseline)\n")
    md.append(render_md_table(
        ["Task", "n", "Baseline", "Shared(full)", "Shared(staged)", "Rand(full)", "Rand(staged)", "Flags"],
        rows_ex
    ))
    md.append("\n")

    md.append("### EOS rate per task (%, with CI if available; Delta vs baseline)\n")
    md.append(render_md_table(
        ["Task", "n", "Baseline", "Shared(full)", "Shared(staged)", "Rand(full)", "Rand(staged)"],
        rows_eos
    ))
    md.append("\n")

    md.append("### Avg new tokens per task (with CI if available; Delta and ratio vs baseline)\n")
    md.append(render_md_table(
        ["Task", "n", "Baseline", "Shared(full)", "Shared(staged)", "Rand(full)", "Rand(staged)"],
        rows_len
    ))
    md.append("\n")

    if include_accuracy:
        md.append("### Accuracy per task (%, with CI if available; Delta vs baseline) [context]\n")
        md.append(render_md_table(
            ["Task", "n", "Baseline", "Shared(full)", "Shared(staged)", "Rand(full)", "Rand(staged)"],
            rows_acc
        ))
        md.append("\n")

    return "".join(md)


def main():
    ap = argparse.ArgumentParser(description="Diagnostic summarizer for disturb_cot results")
    ap.add_argument("--results_dir", type=str, default="results/disturb_cot", help="Directory of JSON files")
    ap.add_argument("--pattern", type=str, default="*.json", help="Glob pattern (default: *.json)")
    ap.add_argument("--output", type=str, default="results/disturb_cot/DIAGNOSTIC_SUMMARY.md", help="Output markdown path")
    ap.add_argument("--decoding", type=str, default="greedy", choices=["greedy", "sample"], help="Which decoding key to summarize")
    ap.add_argument("--include_accuracy", action="store_true", help="Also include per-task accuracy table")


    ap.add_argument("--parse_drop_thresh", type=float, default=0.20)
    ap.add_argument("--extr_floor", type=float, default=0.50)
    ap.add_argument("--require_base_extr", type=float, default=0.80)
    ap.add_argument("--eos_delta_thresh", type=float, default=0.20)
    ap.add_argument("--len_ratio_thresh", type=float, default=0.50)


    ap.add_argument("--only_model_substr", type=str, default="", help="Only include runs whose model string contains this substring")
    ap.add_argument("--only_mode", type=str, default="", choices=["", "all", "loto"], help="Only include a mode")

    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise SystemExit(f"Results dir not found: {results_dir}")

    files = sorted(results_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files match pattern={args.pattern} under {results_dir}")

    loaded: List[Tuple[str, Dict[str, Any]]] = []
    for p in files:
        j = load_json(p)
        if not j:
            continue
        cfg = j.get("config", {})
        model_raw = str(cfg.get("model", ""))
        if args.only_model_substr and args.only_model_substr.lower() not in model_raw.lower():
            continue
        if args.only_mode and cfg.get("mode", "") != args.only_mode:
            continue
        loaded.append((p.name, j))

    if not loaded:
        raise SystemExit("No valid JSONs after filtering.")


    by_model: Dict[str, List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
    for fname, j in loaded:
        cfg = j.get("config", {})
        m = short_model_name(cfg.get("model", "unknown"))
        by_model[m].append((fname, j))

    md: List[str] = []
    md.append("# Disturb CoT Diagnostic Summary\n\n")
    md.append(f"Generated from **{len(loaded)}** JSON file(s). Decoding summarized: **{args.decoding}**\n\n")

    for model in sorted(by_model.keys()):
        md.append(f"# Model: {model}\n\n")
        for fname, j in by_model[model]:
            cfg = j.get("config", {})
            md.append(f"## Run: `{fname}`\n\n")
            md.append(f"- Source file: `{fname}`\n")
            md.append(f"- Tasks: {', '.join(cfg.get('tasks', []))}\n\n")
            md.append(
                summarize_one_file(
                    j,
                    decoding=args.decoding,
                    parse_drop_thresh=args.parse_drop_thresh,
                    extr_floor=args.extr_floor,
                    eos_delta_thresh=args.eos_delta_thresh,
                    len_ratio_thresh=args.len_ratio_thresh,
                    require_base_extr=args.require_base_extr,
                    include_accuracy=bool(args.include_accuracy),
                )
            )
            md.append("\n---\n\n")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(md), encoding="utf-8")
    print(f"[Done] wrote: {out_path}")


if __name__ == "__main__":
    main()
