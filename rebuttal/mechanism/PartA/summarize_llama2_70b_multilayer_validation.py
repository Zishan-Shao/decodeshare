#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
summarize_llama2_70b_multilayer_validation.py

Aggregate A3/A4/A5 JSONs from a 70B multi-layer validation sweep into a compact
markdown + JSON summary that is easy to use in rebuttal drafting.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple


THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_repo_root(start_dir: str) -> str:
    cur = os.path.abspath(start_dir)
    for _ in range(10):
        if os.path.isdir(os.path.join(cur, "src")) and os.path.isdir(os.path.join(cur, "reasoning")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return os.path.normpath(os.path.join(start_dir, "..", "..", ".."))


ROOT_DIR = _find_repo_root(THIS_DIR)
_LAYER_RE = re.compile(r"layer(\d+)")


def _split_csv(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def _parse_layer_groups(spec: str) -> Dict[int, str]:
    layer_to_group: Dict[int, str] = {}
    raw = str(spec).strip()
    if not raw:
        return layer_to_group
    for block in raw.split(";"):
        block = block.strip()
        if not block or ":" not in block:
            continue
        group, layers_csv = block.split(":", 1)
        for part in _split_csv(layers_csv):
            layer = int(part)
            layer_to_group[layer] = str(group).strip()
    return layer_to_group


def _atomic_json_dump(obj: Any, out_path: str) -> None:
    out_path = os.path.expanduser(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, out_path)


def _atomic_text_dump(text: str, out_path: str) -> None:
    out_path = os.path.expanduser(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, out_path)


def _md_table(rows: List[List[str]], header: List[str]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def _fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return ""
    return f"{x * 100:.1f}"


def _fmt_diff(stat: Optional[Dict[str, Any]]) -> str:
    if not stat:
        return ""
    try:
        mean = float(stat["mean_diff"]) * 100.0
        lo = float(stat["ci_low"]) * 100.0
        hi = float(stat["ci_high"]) * 100.0
        p = float(stat["p_value"])
        return f"{mean:+.1f} [{lo:+.1f}, {hi:+.1f}] (p={p:.3g})"
    except Exception:
        return ""


def _get_layer(obj: Dict[str, Any], path: str) -> Optional[int]:
    cfg = obj.get("config", {}) or {}
    if "layer" in cfg:
        try:
            return int(cfg["layer"])
        except Exception:
            pass
    m = _LAYER_RE.search(os.path.basename(path))
    if m:
        return int(m.group(1))
    return None


def _get_acc_ci_mean(by_condition: Dict[str, Any], name: str) -> Optional[float]:
    cond = (by_condition or {}).get(name, {}) or {}
    ci = cond.get("acc_ci", {}) or {}
    if "mean" in ci:
        try:
            return float(ci["mean"])
        except Exception:
            return None
    if "acc" in cond:
        try:
            return float(cond["acc"])
        except Exception:
            return None
    return None


def _task_block(obj: Dict[str, Any], task: str) -> Optional[Dict[str, Any]]:
    eval_block = obj.get("eval", {}) or {}
    if task in eval_block:
        return eval_block[task]
    return None


def _extract_a3_summary(obj: Dict[str, Any], *, main_task: str, source_type: str, path: str) -> Optional[Dict[str, Any]]:
    task = _task_block(obj, main_task)
    if task is None:
        return None
    by_condition = task.get("by_condition", {}) or {}
    paired = task.get("paired_vs_baseline", {}) or {}
    diagnostics = obj.get("diagnostics", {}) or {}
    max_overlap = diagnostics.get("max_overlap", {}) or {}
    return {
        "source_type": source_type,
        "path": os.path.relpath(path, ROOT_DIR),
        "task": main_task,
        "n": int(task.get("n", 0) or 0),
        "baseline": _get_acc_ci_mean(by_condition, "baseline"),
        "shared": _get_acc_ci_mean(by_condition, "shared"),
        "ctrl_energy": _get_acc_ci_mean(by_condition, "ctrl_energy"),
        "rand_energy": _get_acc_ci_mean(by_condition, "rand_energy"),
        "shared_stat": paired.get("shared"),
        "ctrl_energy_stat": paired.get("ctrl_energy"),
        "rand_energy_stat": paired.get("rand_energy"),
        "shared_vs_ctrl_energy_overlap": max_overlap.get("shared_vs_ctrl_energy"),
        "shared_energy_ratio": diagnostics.get("energy_ratio_shared"),
        "ctrl_energy_ratio": diagnostics.get("energy_ratio_ctrl_energy"),
        "rand_energy_ratio": diagnostics.get("energy_ratio_rand_energy"),
    }


def _extract_a4_summary(obj: Dict[str, Any], *, eval_specs: Sequence[str], main_task: str, path: str) -> Optional[Dict[str, Any]]:
    evaluations = obj.get("evaluations", {}) or {}
    out: Dict[str, Any] = {"path": os.path.relpath(path, ROOT_DIR), "eval_specs": {}}
    found = False
    for spec in eval_specs:
        block = evaluations.get(spec, {}) or {}
        pooled = block.get("pooled", {}) or {}
        pooled_paired = pooled.get("paired_vs_baseline", {}) or {}
        pooled_by = pooled.get("by_condition", {}) or {}
        spec_out: Dict[str, Any] = {}
        if "shared" in pooled_paired:
            spec_out["scope"] = "pooled"
            spec_out["baseline"] = _get_acc_ci_mean(pooled_by, "baseline")
            spec_out["shared"] = _get_acc_ci_mean(pooled_by, "shared")
            spec_out["ctrl_energy"] = _get_acc_ci_mean(pooled_by, "ctrl_energy")
            spec_out["rand_energy"] = _get_acc_ci_mean(pooled_by, "rand_energy")
            spec_out["shared_stat"] = pooled_paired.get("shared")
            spec_out["ctrl_energy_stat"] = pooled_paired.get("ctrl_energy")
            spec_out["rand_energy_stat"] = pooled_paired.get("rand_energy")
            found = True
        else:
            tasks = block.get("tasks", {}) or {}
            task = tasks.get(main_task, {}) or {}
            by_condition = task.get("by_condition", {}) or {}
            paired = task.get("paired_vs_baseline", {}) or {}
            if "shared" in paired:
                spec_out["scope"] = main_task
                spec_out["baseline"] = _get_acc_ci_mean(by_condition, "baseline")
                spec_out["shared"] = _get_acc_ci_mean(by_condition, "shared")
                spec_out["ctrl_energy"] = _get_acc_ci_mean(by_condition, "ctrl_energy")
                spec_out["rand_energy"] = _get_acc_ci_mean(by_condition, "rand_energy")
                spec_out["shared_stat"] = paired.get("shared")
                spec_out["ctrl_energy_stat"] = paired.get("ctrl_energy")
                spec_out["rand_energy_stat"] = paired.get("rand_energy")
                found = True
        if spec_out:
            out["eval_specs"][spec] = spec_out
    return out if found else None


def _extract_a5_summary(obj: Dict[str, Any], path: str) -> Optional[Dict[str, Any]]:
    pooled = ((obj.get("eval_meta", {}) or {}).get("pooled", {}) or {})
    decomp = obj.get("decomposition", {}) or {}
    if "shared_full" not in pooled or "resid_only" not in pooled:
        return None
    return {
        "path": os.path.relpath(path, ROOT_DIR),
        "k_shared": decomp.get("k_shared"),
        "k_fmt": decomp.get("k_fmt"),
        "k_resid": decomp.get("k_resid"),
        "fmt_vs_full_drop_share": ((decomp.get("pooled_drop_shares", {}) or {}).get("fmt_vs_full_drop_share")),
        "resid_vs_full_drop_share": ((decomp.get("pooled_drop_shares", {}) or {}).get("resid_vs_full_drop_share")),
        "rand_resid_vs_full_drop_share": ((decomp.get("pooled_drop_shares", {}) or {}).get("rand_resid_vs_full_drop_share")),
        "baseline": ((pooled.get("baseline", {}) or {}).get("acc_ci", {}) or {}).get("mean"),
        "shared_full_stat": ((pooled.get("shared_full", {}) or {}).get("paired_vs_baseline")),
        "fmt_only_stat": ((pooled.get("fmt_only", {}) or {}).get("paired_vs_baseline")),
        "resid_only_stat": ((pooled.get("resid_only", {}) or {}).get("paired_vs_baseline")),
        "rand_fmt_stat": ((pooled.get("rand_fmt_shared", {}) or {}).get("paired_vs_baseline")),
        "rand_resid_stat": ((pooled.get("rand_resid_shared", {}) or {}).get("paired_vs_baseline")),
    }


def _scan_jsons(root_dirs: Sequence[str]) -> List[str]:
    out: List[str] = []
    for root in root_dirs:
        root = os.path.expanduser(root)
        if not os.path.exists(root):
            continue
        if os.path.isfile(root) and root.endswith(".json"):
            out.append(root)
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            for fn in filenames:
                if fn.endswith(".json"):
                    out.append(os.path.join(dirpath, fn))
    return sorted(set(out))


def _pick_latest(paths: List[str]) -> Optional[str]:
    if not paths:
        return None
    return max(paths, key=lambda p: os.path.getmtime(p))


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize 70B multi-layer validation artifacts.")
    ap.add_argument("--root_dirs", type=str, required=True, help="CSV of JSON files or directories to scan.")
    ap.add_argument("--layer_groups", type=str, default="")
    ap.add_argument("--main_task", type=str, default="commonsenseqa")
    ap.add_argument("--a4_eval_specs", type=str, default="number_rewrite,text_rewrite")
    ap.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(ROOT_DIR, "results", "rebuttal_scaling", "llama2_70b_multilayer_validation", "summary"),
    )
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    layer_to_group = _parse_layer_groups(args.layer_groups)
    a4_eval_specs = _split_csv(args.a4_eval_specs)
    json_paths = _scan_jsons(_split_csv(args.root_dirs))

    discovered: Dict[int, Dict[str, List[str]]] = {}
    for path in json_paths:
        base = os.path.basename(path)
        kind = None
        if base.startswith("exp_A3_eval_saved_basis_layer"):
            kind = "a3_eval"
        elif base.startswith("exp_A3_causal_layer"):
            kind = "a3_basis"
        elif base.startswith("exp_A4_option_text_layer"):
            kind = "a4"
        elif base.startswith("exp_A5_probe_split_layer"):
            kind = "a5"
        else:
            continue

        try:
            obj = _load_json(path)
        except Exception:
            continue
        layer = _get_layer(obj, path)
        if layer is None:
            continue
        discovered.setdefault(int(layer), {}).setdefault(kind, []).append(path)

    summary_layers: Dict[str, Any] = {}
    ordered_layers = sorted(discovered.keys())
    for layer in ordered_layers:
        kinds = discovered[layer]
        rec: Dict[str, Any] = {
            "group": layer_to_group.get(layer, ""),
            "artifacts": {k: sorted(v) for k, v in kinds.items()},
        }

        a3_eval_path = _pick_latest(kinds.get("a3_eval", []))
        a3_basis_path = _pick_latest(kinds.get("a3_basis", []))
        if a3_eval_path:
            rec["a3_eval"] = _extract_a3_summary(
                _load_json(a3_eval_path),
                main_task=args.main_task,
                source_type="a3_eval_saved_basis",
                path=a3_eval_path,
            )
        if a3_basis_path:
            rec["a3_basis"] = _extract_a3_summary(
                _load_json(a3_basis_path),
                main_task=args.main_task,
                source_type="a3_basis_smoke",
                path=a3_basis_path,
            )
        a4_path = _pick_latest(kinds.get("a4", []))
        if a4_path:
            rec["a4"] = _extract_a4_summary(
                _load_json(a4_path),
                eval_specs=a4_eval_specs,
                main_task=args.main_task,
                path=a4_path,
            )
        a5_path = _pick_latest(kinds.get("a5", []))
        if a5_path:
            rec["a5"] = _extract_a5_summary(_load_json(a5_path), a5_path)

        summary_layers[str(layer)] = rec

    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    tag = ("_" + str(args.tag).strip()) if str(args.tag).strip() else ""
    out_json = os.path.join(out_dir, f"llama2_70b_multilayer_summary{tag}.json")
    out_md = os.path.join(out_dir, f"llama2_70b_multilayer_summary{tag}.md")

    a3_rows: List[List[str]] = []
    a4_rows: List[List[str]] = []
    a5_rows: List[List[str]] = []

    for layer in ordered_layers:
        rec = summary_layers[str(layer)]
        a3 = rec.get("a3_eval") or rec.get("a3_basis")
        if a3:
            a3_rows.append(
                [
                    str(layer),
                    rec.get("group", ""),
                    str(a3.get("source_type", "")),
                    str(a3.get("n", "")),
                    _fmt_pct(a3.get("baseline")),
                    _fmt_pct(a3.get("shared")),
                    _fmt_diff(a3.get("shared_stat")),
                    _fmt_diff(a3.get("ctrl_energy_stat")),
                    _fmt_diff(a3.get("rand_energy_stat")),
                ]
            )

        a4 = rec.get("a4") or {}
        if a4:
            row = [str(layer), rec.get("group", "")]
            for spec in a4_eval_specs:
                spec_rec = ((a4.get("eval_specs", {}) or {}).get(spec, {}) or {})
                row.extend(
                    [
                        str(spec_rec.get("scope", "")),
                        _fmt_diff(spec_rec.get("shared_stat")),
                        _fmt_diff(spec_rec.get("ctrl_energy_stat")),
                        _fmt_diff(spec_rec.get("rand_energy_stat")),
                    ]
                )
            a4_rows.append(row)

        a5 = rec.get("a5") or {}
        if a5:
            k_shared = a5.get("k_shared")
            k_fmt = a5.get("k_fmt")
            k_resid = a5.get("k_resid")
            decomp = ""
            if k_shared is not None and k_fmt is not None and k_resid is not None:
                decomp = f"{int(k_fmt)}/{int(k_resid)}/{int(k_shared)}"
            a5_rows.append(
                [
                    str(layer),
                    rec.get("group", ""),
                    decomp,
                    _fmt_diff(a5.get("shared_full_stat")),
                    _fmt_diff(a5.get("fmt_only_stat")),
                    _fmt_diff(a5.get("resid_only_stat")),
                    _fmt_diff(a5.get("rand_resid_stat")),
                    (
                        ""
                        if a5.get("resid_vs_full_drop_share") is None
                        else f"{float(a5['resid_vs_full_drop_share']):.2f}"
                    ),
                ]
            )

    payload = {
        "config": {
            "root_dirs": _split_csv(args.root_dirs),
            "layer_groups": args.layer_groups,
            "main_task": args.main_task,
            "a4_eval_specs": a4_eval_specs,
        },
        "layers": summary_layers,
    }
    _atomic_json_dump(payload, out_json)

    md: List[str] = []
    md.append("# Llama-2-70B Multi-layer Validation Summary")
    md.append("")
    md.append(f"Main task for A3 selection: `{args.main_task}`")
    md.append("")
    md.append(f"Scanned roots: `{', '.join(_split_csv(args.root_dirs))}`")
    md.append("")

    md.append("## A3 main effect")
    if a3_rows:
        md.append(
            _md_table(
                a3_rows,
                ["Layer", "Group", "Source", "n", "Baseline", "Shared", "ΔShared", "ΔCtrl(E)", "ΔRand(E)"],
            )
        )
    else:
        md.append("No A3 artifacts found.")
    md.append("")

    md.append("## A4 formatting / answer-routing checks")
    if a4_rows:
        header = ["Layer", "Group"]
        for spec in a4_eval_specs:
            header.extend([f"{spec} scope", f"{spec} ΔShared", f"{spec} ΔCtrl(E)", f"{spec} ΔRand(E)"])
        md.append(_md_table(a4_rows, header))
    else:
        md.append("No A4 artifacts found.")
    md.append("")

    md.append("## A5 probe split checks")
    if a5_rows:
        md.append(
            _md_table(
                a5_rows,
                ["Layer", "Group", "k_fmt/k_resid/k_shared", "ΔFull", "ΔFmt", "ΔResid", "ΔRandResid", "Resid/Full"],
            )
        )
    else:
        md.append("No A5 artifacts found.")
    md.append("")

    md.append(f"JSON: `{os.path.relpath(out_json, ROOT_DIR)}`")
    md.append("")
    _atomic_text_dump("\n".join(md), out_md)
    print(f"[Saved] {out_json}")
    print(f"[Saved] {out_md}")


if __name__ == "__main__":
    main()
