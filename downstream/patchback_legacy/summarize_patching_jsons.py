#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import glob
import argparse
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

'''
1) 只总结某个目录（不递归）
python summarize_patching_jsons.py \
  --dir results/subspace_patching_transfer/runs_layer10_seed123 \
  --pattern "*.json"

2) 总结整个 results（递归扫所有 json：subspace + openanswer + flipset）
python summarize_patching_jsons.py \
  --dir results \
  --pattern "**/*.json" \
  --recursive

3) 不 dedupe（保留同一个 task 的多个不同 donor/seed/run）
python summarize_patching_jsons.py \
  --dir results \
  --pattern "**/*.json" \
  --recursive \
  --no_dedupe

'''

# -----------------------------
# Helpers
# -----------------------------

def safe_get(d: Any, path: str, default=None):
    cur: Any = d
    for p in path.split("."):
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        elif isinstance(cur, list):
            try:
                idx = int(p)
                cur = cur[idx]
            except Exception:
                return default
        else:
            return default
    return cur

def as_label_str(cand):
    if isinstance(cand, list):
        try:
            return "".join(str(x) for x in cand)
        except Exception:
            return str(cand)
    return str(cand) if cand is not None else ""

def is_computeq_file(filename: str) -> bool:
    s = filename.lower()
    return ("computeq" in s) or ("compute_q" in s) or ("computeqs" in s) or ("compute_qs" in s)

def fmt(x, nd=3):
    if x is None:
        return ""
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)

def pct(x, nd=1):
    if x is None:
        return ""
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)

def detect_kind(obj: Dict[str, Any]) -> str:
    """
    Detect which script produced this JSON.
    """
    # flipset script signatures
    if isinstance(obj.get("transfer_patching_rows"), list) or isinstance(obj.get("alpha_sweep_summary_on_flipset"), dict):
        return "flipset"
    if isinstance(obj.get("transfer_patching_summary_on_flipset"), dict):
        return "flipset"

    # openanswer script signatures
    evm = safe_get(obj, "meta.eval_mode", None)
    if evm is not None:
        return "openanswer"

    # default: subspace_patching_transfer (MC)
    return "subspace_mc"

def extract_task(obj: Dict[str, Any]) -> str:
    t = safe_get(obj, "meta.task", "") or ""
    if t:
        return str(t)
    # fallback filename stem
    fn = obj.get("_file_name", "unknown.json")
    return os.path.splitext(fn)[0]

def extract_eval_mode(obj: Dict[str, Any]) -> str:
    evm = safe_get(obj, "meta.eval_mode", "")
    return str(evm) if evm else ""

def extract_hf_meta(obj: Dict[str, Any]) -> Tuple[str, str]:
    """
    openanswer HF loader meta might sit in meta.eval_meta or meta.eval_meta.hf_id
    """
    ev_meta = safe_get(obj, "meta.eval_meta", {}) or {}
    hf_id = ev_meta.get("hf_id", "") if isinstance(ev_meta, dict) else ""
    hf_split = ev_meta.get("split", "") if isinstance(ev_meta, dict) else ""
    return str(hf_id), str(hf_split)

def extract_qshape_str(obj: Dict[str, Any]) -> str:
    qshape = safe_get(obj, "meta.Qs_shape", None)
    if isinstance(qshape, list) and len(qshape) == 2:
        return f"{qshape[0]}x{qshape[1]}"
    return str(qshape) if qshape else ""

def extract_patch_desc(obj: Dict[str, Any], kind: str) -> str:
    """
    Provide a small string describing patch window / steps.
    """
    if kind == "openanswer":
        steps = safe_get(obj, "meta.patch_steps", None)
        if isinstance(steps, list):
            return "steps=" + ",".join(str(int(x)) for x in steps)
        return ""
    if kind == "flipset":
        steps = safe_get(obj, "meta.transfer_patching.patch_steps_final", None)
        if isinstance(steps, list):
            return "steps=" + ",".join(str(int(x)) for x in steps)
        # older outputs: maybe only requested
        steps2 = safe_get(obj, "meta.transfer_patching.patch_steps_requested", None)
        if isinstance(steps2, list):
            return "steps=" + ",".join(str(int(x)) for x in steps2)
        return ""
    # subspace_mc
    # we typically care about patched_0 vs patched_full; keep blank
    return ""

def _pick_ablated_dict(row: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Different scripts use:
      - ablated
      - ablated_1 (flipset scan_rows)
    """
    if isinstance(row.get("ablated"), dict):
        return row["ablated"], "ablated"
    if isinstance(row.get("ablated_1"), dict):
        return row["ablated_1"], "ablated_1"
    if isinstance(row.get("ablated1"), dict):
        return row["ablated1"], "ablated1"
    return None, ""

def summarize_scan_rows(scan_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute from scan_rows:
      base_correct, ablt_correct,
      flips (base correct -> ablt wrong),
      anti_flips (base wrong -> ablt correct),
      both_correct, both_wrong,
      skipped (skipped_reason or missing baseline/ablated/correct).
    Works for:
      - subspace_mc: baseline vs ablated
      - openanswer: baseline vs ablated
      - flipset: baseline vs ablated_1
    """
    base_correct = 0
    ablt_correct = 0
    flips = 0
    anti_flips = 0
    both_correct = 0
    both_wrong = 0
    skipped = 0
    counted = 0

    for r in scan_rows:
        if not isinstance(r, dict):
            continue
        if "skipped_reason" in r:
            skipped += 1
            continue

        b = r.get("baseline", None)
        a, _akey = _pick_ablated_dict(r)

        if not isinstance(b, dict) or not isinstance(a, dict):
            skipped += 1
            continue

        bc = b.get("correct", None)
        ac = a.get("correct", None)

        if not isinstance(bc, bool) or not isinstance(ac, bool):
            skipped += 1
            continue

        counted += 1
        base_correct += int(bc)
        ablt_correct += int(ac)

        if bc and (not ac):
            flips += 1
        if (not bc) and ac:
            anti_flips += 1
        if bc and ac:
            both_correct += 1
        if (not bc) and (not ac):
            both_wrong += 1

    n = len(scan_rows)
    n_eff = counted

    return {
        "scan_n": n,
        "scan_skipped": skipped,
        "scan_effective": n_eff,
        "base_correct_scan": base_correct,
        "ablt_correct_scan": ablt_correct,
        "base_acc_scan": (base_correct / n_eff) if n_eff > 0 else None,
        "ablt_acc_scan": (ablt_correct / n_eff) if n_eff > 0 else None,
        "flips_scan": flips,
        "anti_flips_scan": anti_flips,
        "both_correct_scan": both_correct,
        "both_wrong_scan": both_wrong,
    }

def _mean_key_from_summary(s: Dict[str, Any]) -> Optional[float]:
    for k in ["mean_dmargin", "mean_delta_margin_vs_ablated", "mean_delta_margin_vs_baseline"]:
        if k in s:
            try:
                return float(s[k])
            except Exception:
                return None
    return None

def summarize_rescue_from_rows(
    rows: List[Dict[str, Any]],
    method: str,
    *,
    ablated_key_candidates: Tuple[str, ...] = ("ablated", "ablated_1", "ablated1"),
) -> Optional[Dict[str, Any]]:
    """
    Fallback: compute rescued stats from raw rows, if summary block missing.
    rows can be:
      - flip_rows in subspace/openanswer: contains "ablated"
      - transfer_patching_rows in flipset: contains "ablated_1"
    """
    if not rows:
        return None

    rescued = 0
    n = 0
    dms: List[float] = []
    have_margin = True

    for r in rows:
        if not isinstance(r, dict) or method not in r:
            continue
        m = r.get(method, None)
        if not isinstance(m, dict):
            continue

        # pick ablated dict
        ablt = None
        for ak in ablated_key_candidates:
            if isinstance(r.get(ak), dict):
                ablt = r[ak]
                break

        if ablt is None:
            # cannot compute rescue reliably
            continue

        mc = m.get("correct", None)
        if not isinstance(mc, bool):
            continue

        n += 1
        rescued += int(mc)

        # delta margin if available
        if have_margin:
            try:
                dm = float(m.get("margin")) - float(ablt.get("margin"))
                dms.append(dm)
            except Exception:
                have_margin = False

    if n == 0:
        return None

    out = {
        "n": n,
        "rescued": rescued,
        "rescued_pct": 100.0 * rescued / n,
        "mean_dmargin": (sum(dms) / len(dms)) if (have_margin and dms) else None,
    }
    return out

def extract_method_summary(
    obj: Dict[str, Any],
    kind: str,
    method: str,
) -> Tuple[Optional[float], Optional[float], Optional[int], Optional[int]]:
    """
    Return (rescued_pct, mean_dmargin, rescued, n).

    Priority:
      - subspace/openanswer: summary_on_flips[method]
      - flipset: transfer_patching_summary_on_flipset[method]
      - fallback: compute from flip_rows / transfer_patching_rows
    """
    # 1) summary dict from subspace/openanswer
    s = safe_get(obj, f"summary_on_flips.{method}", None)
    if isinstance(s, dict):
        rp = s.get("rescued_pct", None)
        rescued = s.get("rescued", None)
        n = s.get("n", None)
        if rp is None and isinstance(rescued, (int, float)) and isinstance(n, (int, float)) and n:
            rp = 100.0 * float(rescued) / float(n)
        mdm = _mean_key_from_summary(s)
        return (rp, mdm, rescued if isinstance(rescued, int) else None, n if isinstance(n, int) else None)

    # 2) flipset transfer summary
    s2 = safe_get(obj, f"transfer_patching_summary_on_flipset.{method}", None)
    if isinstance(s2, dict):
        rp = s2.get("rescued_pct", None)
        rescued = s2.get("rescued", None)
        n = s2.get("n", None)
        if rp is None and isinstance(rescued, (int, float)) and isinstance(n, (int, float)) and n:
            rp = 100.0 * float(rescued) / float(n)
        mdm = _mean_key_from_summary(s2)
        return (rp, mdm, rescued if isinstance(rescued, int) else None, n if isinstance(n, int) else None)

    # 3) fallback from rows
    if kind == "flipset":
        rows = obj.get("transfer_patching_rows", [])
        c = summarize_rescue_from_rows(rows, method, ablated_key_candidates=("ablated_1", "ablated", "ablated1"))
    else:
        rows = obj.get("flip_rows", [])
        c = summarize_rescue_from_rows(rows, method, ablated_key_candidates=("ablated", "ablated_1", "ablated1"))

    if not isinstance(c, dict):
        return (None, None, None, None)

    return (
        c.get("rescued_pct", None),
        c.get("mean_dmargin", None),
        c.get("rescued", None),
        c.get("n", None),
    )

def pick_primary_patch_method(r: Dict[str, Any]) -> str:
    """
    For unified summary across scripts:
      subspace_mc: patched_0 is primary
      openanswer: patched_self is primary
      flipset: patched_transfer is primary (if exists), else patched_self
    """
    for m in ["patched_0", "patched_self", "patched_transfer", "patched_full"]:
        if r.get(f"{m}_rescued_pct") is not None:
            return m
    return ""

def dedupe_key(obj: Dict[str, Any], kind: str) -> Tuple[str, str, str, str, str, str]:
    """
    Default dedupe key: (kind, task, eval_mode, layer, seed, extra)
    extra tries to separate multiple flipset donor configs, openanswer hf_id/split, etc.
    """
    task = extract_task(obj)
    eval_mode = extract_eval_mode(obj)
    layer = str(safe_get(obj, "meta.layer", ""))
    seed = str(safe_get(obj, "meta.seed", ""))
    extra = ""

    if kind == "flipset":
        a_en = safe_get(obj, "meta.alpha_sweep.enabled", None)
        t_en = safe_get(obj, "meta.transfer_patching.enabled", None)
        donor_source = safe_get(obj, "donors_meta.0.donor_source", "")
        donor_tasks = safe_get(obj, "donors_meta.0.donor_tasks", [])
        if isinstance(donor_tasks, list):
            donor_tasks = ",".join(str(x) for x in donor_tasks)
        extra = f"a{int(bool(a_en))}t{int(bool(t_en))}:{donor_source}:{donor_tasks}"
    elif kind == "openanswer":
        hf_id, hf_split = extract_hf_meta(obj)
        pdesc = extract_patch_desc(obj, kind)
        extra = f"{hf_id}:{hf_split}:{pdesc}"
    else:
        cand = as_label_str(safe_get(obj, "meta.candidate_labels", ""))
        extra = f"cand={cand}"

    return (kind, task, eval_mode, layer, seed, extra)


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default=".", help="Directory containing result json files")
    ap.add_argument("--pattern", type=str, default="*.json", help="Glob pattern (relative to --dir unless absolute)")
    ap.add_argument("--recursive", action="store_true", help="Enable recursive glob (useful with **/*.json)")

    ap.add_argument("--out_csv", type=str, default="summary.csv")
    ap.add_argument("--out_md", type=str, default="summary.md")
    ap.add_argument("--out_paper_md", type=str, default="paper_table.md")

    ap.add_argument("--out_alpha_csv", type=str, default="alpha_sweep.csv")
    ap.add_argument("--out_alpha_md", type=str, default="alpha_sweep.md")

    ap.add_argument("--keep_computeq", action="store_true",
                    help="If set, DO NOT skip '*computeQ*.json' files.")
    ap.add_argument("--no_dedupe", action="store_true",
                    help="If set, do NOT dedupe; keep all json files.")
    ap.add_argument("--pick_latest", action="store_true", default=True,
                    help="When deduping, keep the latest file by mtime (default).")

    args = ap.parse_args()

    # Resolve files
    glob_path = os.path.join(args.dir, args.pattern) if not os.path.isabs(args.pattern) else args.pattern
    recursive = bool(args.recursive) or ("**" in glob_path)
    all_files = sorted(glob.glob(glob_path, recursive=recursive))
    if not all_files:
        raise SystemExit(f"No json files found with pattern: {glob_path} (recursive={recursive})")

    # Filter computeQ
    filtered_files: List[str] = []
    skipped_computeq: List[str] = []
    for fp in all_files:
        base = os.path.basename(fp)
        if (not args.keep_computeq) and is_computeq_file(base):
            skipped_computeq.append(fp)
            continue
        filtered_files.append(fp)

    if not filtered_files:
        raise SystemExit("After filtering computeQ files, no json files remain. Use --keep_computeq to include them.")

    # Load jsons
    loaded: List[Dict[str, Any]] = []
    load_errors: List[Tuple[str, str]] = []

    for fp in filtered_files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                obj = json.load(f)
            obj["_file_path"] = fp
            obj["_file_name"] = os.path.basename(fp)
            obj["_mtime"] = os.path.getmtime(fp)
            loaded.append(obj)
        except Exception as e:
            load_errors.append((fp, str(e)))

    if not loaded:
        raise SystemExit("No json could be loaded successfully.")

    # Dedupe
    kept: List[Dict[str, Any]] = []
    deduped_out: List[Dict[str, Any]] = []

    if args.no_dedupe:
        kept = loaded
    else:
        by_key: Dict[Tuple[str, str, str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
        for obj in loaded:
            kind = detect_kind(obj)
            k = dedupe_key(obj, kind)
            by_key[k].append(obj)

        for k, objs in by_key.items():
            if len(objs) == 1:
                kept.append(objs[0])
            else:
                objs_sorted = sorted(objs, key=lambda o: float(o.get("_mtime", 0.0)), reverse=True)
                kept.append(objs_sorted[0])
                deduped_out.extend(objs_sorted[1:])

        kept.sort(key=lambda o: (
            detect_kind(o),
            str(extract_task(o)),
            str(extract_eval_mode(o)),
            str(o.get("_file_name", "")),
        ))

    # Unified method list across 3 scripts
    methods = [
        # subspace_mc
        "patched_0",
        "patched_01",
        "patched_full",

        # openanswer
        "patched_self",

        # flipset
        "patched_transfer",

        # controls
        "control_time_shuffled",
        "control_shared_mismatch",
        "control_shared_perm",
        "control_shared_signflip",
        "control_shared_randvec",
        "control_rand_subspace",
        "control_patch_nonshared",
    ]

    # Build rows
    rows: List[Dict[str, Any]] = []
    alpha_rows: List[Dict[str, Any]] = []

    for obj in kept:
        kind = detect_kind(obj)
        meta = obj.get("meta", {}) or {}

        task = extract_task(obj)
        eval_mode = extract_eval_mode(obj)
        layer = safe_get(obj, "meta.layer", None)
        seed = safe_get(obj, "meta.seed", None)
        model = safe_get(obj, "meta.model", None)
        cand = safe_get(obj, "meta.candidate_labels", None)
        qshape = extract_qshape_str(obj)
        patch_desc = extract_patch_desc(obj, kind)

        hf_id, hf_split = ("", "")
        if kind == "openanswer":
            hf_id, hf_split = extract_hf_meta(obj)

        # flipset donor meta
        donor_source = safe_get(obj, "donors_meta.0.donor_source", "")
        donor_tasks = safe_get(obj, "donors_meta.0.donor_tasks", [])
        donor_pick = safe_get(obj, "donors_meta.0.donor_pick", "")
        n_donor_bank = safe_get(obj, "donors_meta.0.n_donor_bank", None)
        if isinstance(donor_tasks, list):
            donor_tasks = ",".join(str(x) for x in donor_tasks)
        else:
            donor_tasks = str(donor_tasks) if donor_tasks else ""

        # scan stats
        scan_rows = obj.get("scan_rows", [])
        scan_stats = summarize_scan_rows(scan_rows) if isinstance(scan_rows, list) else {}

        r: Dict[str, Any] = {
            "kind": kind,
            "file": obj.get("_file_name", ""),
            "path": obj.get("_file_path", ""),
            "task": task,
            "eval_mode": eval_mode,
            "layer": layer,
            "seed": seed,
            "model": model,
            "hf_id": hf_id,
            "hf_split": hf_split,
            "candidate_labels": as_label_str(cand),
            "Qs_shape": qshape,
            "patch_desc": patch_desc,
            "donor_source": donor_source,
            "donor_tasks": donor_tasks,
            "donor_pick": donor_pick,
            "n_donor_bank": n_donor_bank,
        }
        r.update(scan_stats)

        # method stats
        for m in methods:
            rescued_pct, mean_dmargin, rescued, n = extract_method_summary(obj, kind, m)
            r[f"{m}_rescued"] = rescued
            r[f"{m}_n"] = n
            r[f"{m}_rescued_pct"] = rescued_pct
            r[f"{m}_mean_dmargin"] = mean_dmargin

        # primary patch method + diffs
        primary = pick_primary_patch_method(r)
        r["patched_primary_method"] = primary
        r["patched_primary_rescued_pct"] = r.get(f"{primary}_rescued_pct") if primary else None
        r["patched_primary_mean_dmargin"] = r.get(f"{primary}_mean_dmargin") if primary else None

        ts = r.get("control_time_shuffled_rescued_pct")
        rv = r.get("control_shared_randvec_rescued_pct")
        pp = r.get("patched_primary_rescued_pct")

        r["diff_time_shuffled_minus_patched_primary_rescued_pct"] = (ts - pp) if (ts is not None and pp is not None) else None
        r["diff_patched_primary_minus_shared_randvec_rescued_pct"] = (pp - rv) if (pp is not None and rv is not None) else None

        rows.append(r)

        # alpha sweep rows (flipset only)
        if kind == "flipset":
            alpha_sum = obj.get("alpha_sweep_summary_on_flipset", {})
            if isinstance(alpha_sum, dict) and alpha_sum:
                for a_str, s in alpha_sum.items():
                    if not isinstance(s, dict):
                        continue
                    try:
                        a_val = float(a_str)
                    except Exception:
                        # keep raw
                        a_val = None
                    alpha_rows.append({
                        "file": obj.get("_file_name", ""),
                        "task": task,
                        "layer": layer,
                        "seed": seed,
                        "alpha": a_val if a_val is not None else a_str,
                        "n": s.get("n", None),
                        "flip_rate": s.get("flip_rate", None),
                        "ablated_acc": s.get("ablated_acc", None),
                        "pred_change_rate": s.get("pred_change_rate", None),
                        "mean_margin": s.get("mean_margin", None),
                        "mean_delta_margin_vs_baseline": s.get("mean_delta_margin_vs_baseline", None),
                    })

    # Sort rows
    rows.sort(key=lambda x: (str(x.get("kind", "")), str(x.get("task", "")), str(x.get("eval_mode", "")), str(x.get("file", ""))))
    alpha_rows.sort(key=lambda x: (str(x.get("task", "")), str(x.get("file", "")), float(x["alpha"]) if isinstance(x.get("alpha"), (int, float)) else 1e9))

    # -----------------------------
    # Write summary.csv
    # -----------------------------
    import csv

    base_cols = [
        "kind", "file", "task", "eval_mode", "layer", "seed",
        "hf_id", "hf_split",
        "candidate_labels", "Qs_shape", "patch_desc",
        "donor_source", "donor_tasks", "donor_pick", "n_donor_bank",
        "scan_effective", "scan_skipped",
        "base_acc_scan", "ablt_acc_scan",
        "flips_scan", "anti_flips_scan",
        "both_correct_scan", "both_wrong_scan",
        "patched_primary_method",
        "patched_primary_rescued_pct", "patched_primary_mean_dmargin",
        "diff_time_shuffled_minus_patched_primary_rescued_pct",
        "diff_patched_primary_minus_shared_randvec_rescued_pct",
    ]

    method_cols = []
    for m in methods:
        method_cols += [
            f"{m}_rescued", f"{m}_n",
            f"{m}_rescued_pct", f"{m}_mean_dmargin",
        ]

    cols = base_cols + method_cols

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for rr in rows:
            w.writerow({c: rr.get(c, "") for c in cols})

    # -----------------------------
    # Write summary.md (compact)
    # -----------------------------
    md_cols = [
        "kind", "task", "eval_mode", "file",
        "base_acc_scan", "ablt_acc_scan",
        "flips_scan",
        "patched_primary_method", "patched_primary_rescued_pct",
        "control_time_shuffled_rescued_pct",
        "control_shared_randvec_rescued_pct",
        "control_rand_subspace_rescued_pct",
        "control_patch_nonshared_rescued_pct",
    ]

    def md_cell(v, is_pct=False):
        if v is None:
            return ""
        if isinstance(v, float):
            return (f"{v:.1f}" if is_pct else f"{v:.3f}")
        return str(v)

    md_lines = []
    md_lines.append("| " + " | ".join(md_cols) + " |")
    md_lines.append("| " + " | ".join(["---"] * len(md_cols)) + " |")
    for rr in rows:
        row_cells = []
        for c in md_cols:
            is_pct = c.endswith("_rescued_pct")
            row_cells.append(md_cell(rr.get(c), is_pct=is_pct))
        md_lines.append("| " + " | ".join(row_cells) + " |")

    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    # -----------------------------
    # Write paper_table.md
    # -----------------------------
    key_methods = [
        ("patched_0", "Patched@0"),
        ("patched_full", "Patched@full"),
        ("patched_self", "Patched(self)"),
        ("patched_transfer", "Patched(transfer)"),
        ("control_time_shuffled", "Cross-example donor"),
        ("control_shared_mismatch", "Donor mismatch"),
        ("control_shared_perm", "Shared coeff permute"),
        ("control_shared_signflip", "Shared coeff signflip"),
        ("control_shared_randvec", "Rand vec in shared"),
        ("control_rand_subspace", "Rand subspace"),
        ("control_patch_nonshared", "Nonshared patch"),
    ]

    paper_cols = ["kind", "task", "eval_mode", "base_acc_scan", "ablt_acc_scan", "flips_scan"]
    for _, name in key_methods:
        paper_cols.append(f"{name} (rescue%, Δm)")

    paper_lines = []
    paper_lines.append("| " + " | ".join(paper_cols) + " |")
    paper_lines.append("| " + " | ".join(["---"] * len(paper_cols)) + " |")

    for rr in rows:
        cells = [
            str(rr.get("kind", "")),
            str(rr.get("task", "")),
            str(rr.get("eval_mode", "")),
            fmt(rr.get("base_acc_scan"), 3),
            fmt(rr.get("ablt_acc_scan"), 3),
            fmt(rr.get("flips_scan"), 0),
        ]
        for m, _name in key_methods:
            rp = rr.get(f"{m}_rescued_pct")
            dm = rr.get(f"{m}_mean_dmargin")
            if rp is None and dm is None:
                cells.append("")
            else:
                dm_str = fmt(dm, 3) if dm is not None else "-"
                cells.append(f"{pct(rp,1)}%, {dm_str}")
        paper_lines.append("| " + " | ".join(cells) + " |")

    with open(args.out_paper_md, "w", encoding="utf-8") as f:
        f.write("\n".join(paper_lines) + "\n")

    # -----------------------------
    # Alpha sweep outputs (flipset only)
    # -----------------------------
    if alpha_rows:
        alpha_cols = [
            "file", "task", "layer", "seed", "alpha",
            "n", "flip_rate", "ablated_acc", "pred_change_rate",
            "mean_margin", "mean_delta_margin_vs_baseline",
        ]
        with open(args.out_alpha_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=alpha_cols)
            w.writeheader()
            for ar in alpha_rows:
                w.writerow({c: ar.get(c, "") for c in alpha_cols})

        # md
        md_a_cols = ["task", "file", "alpha", "n", "flip_rate", "ablated_acc", "mean_delta_margin_vs_baseline"]
        lines = []
        lines.append("| " + " | ".join(md_a_cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(md_a_cols)) + " |")
        for ar in alpha_rows:
            def _cell(v, pct_flag=False):
                if v is None:
                    return ""
                if isinstance(v, float):
                    return f"{v:.3f}" if not pct_flag else f"{100.0*v:.1f}"
                return str(v)
            lines.append("| " + " | ".join([
                str(ar.get("task", "")),
                str(ar.get("file", "")),
                str(ar.get("alpha", "")),
                str(ar.get("n", "")),
                _cell(ar.get("flip_rate"), pct_flag=True),  # show as %
                _cell(ar.get("ablated_acc"), pct_flag=True),
                _cell(ar.get("mean_delta_margin_vs_baseline"), pct_flag=False),
            ]) + " |")
        with open(args.out_alpha_md, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    # -----------------------------
    # Terminal print
    # -----------------------------
    print(f"[OK] Found {len(all_files)} json files from {glob_path} (recursive={recursive})")
    if skipped_computeq and (not args.keep_computeq):
        print(f"[OK] Skipped {len(skipped_computeq)} computeQ json(s)")
    if load_errors:
        print(f"[Warn] {len(load_errors)} json failed to load (showing up to 3):")
        for fp, err in load_errors[:3]:
            print(f"  - {fp}: {err}")
    if (not args.no_dedupe) and deduped_out:
        print(f"[OK] Deduped: kept {len(kept)} / {len(loaded)} (dropped {len(deduped_out)})")

    print(f"[OK] Wrote: {args.out_csv}")
    print(f"[OK] Wrote: {args.out_md}")
    print(f"[OK] Wrote: {args.out_paper_md}")
    if alpha_rows:
        print(f"[OK] Wrote: {args.out_alpha_csv}")
        print(f"[OK] Wrote: {args.out_alpha_md}")
    print()

    # quick stdout view (compact)
    header = [
        "kind", "task", "eval_mode", "file",
        "base_acc_scan", "ablt_acc_scan",
        "flips_scan",
        "patched_primary_method", "patched_primary_rescued_pct",
        "control_time_shuffled_rescued_pct",
        "control_shared_randvec_rescued_pct",
    ]
    print("\t".join(header))
    for rr in rows:
        out_cells = []
        for h in header:
            if h.endswith("_rescued_pct"):
                out_cells.append(pct(rr.get(h), 1))
            elif h.endswith("_acc_scan"):
                out_cells.append(fmt(rr.get(h), 3))
            else:
                out_cells.append(str(rr.get(h, "")))
        print("\t".join(out_cells))


if __name__ == "__main__":
    main()
