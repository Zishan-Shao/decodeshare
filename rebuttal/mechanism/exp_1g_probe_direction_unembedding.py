# -*- coding: utf-8 -*-
"""
exp_1g_probe_direction_unembedding.py

Probe-direction unembedding analysis.

This script is designed to make the unembedding view directly comparable to the
linear-classifier view:

1. For the format/readout slice, it reuses the saved probe coefficients from the
   A5 causal split run (e.g. answer_readout / option_letter / newline learned on
   Q_shared), maps those classifier normals back into hidden space, and probes the
   unembedding along those directions.
2. For reasoning-style tags, it fits fresh logistic probes on open-answer decode
   states using a chosen saved basis (typically Q_resid), then again maps the
   learned classifier normals back into hidden space and probes the unembedding.

The goal is to report a classifier-aligned unembedding signature, not a
whole-subspace aggregate signature.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional

import numpy as np
import torch


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(THIS_DIR, "..", ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
REASONING_DIR = os.path.join(ROOT_DIR, "reasoning")
PARTA_DIR = os.path.join(THIS_DIR, "PartA")

for p in [SRC_DIR, REASONING_DIR, ROOT_DIR, THIS_DIR, PARTA_DIR]:
    if p not in sys.path:
        sys.path.append(p)

import eval_perf as EP  # noqa: E402
from benchmark_dataloaders import load_selected_tasks  # noqa: E402
from exp_1_logit_lens_vocab_signature import (  # noqa: E402
    _get_unembedding_weight,
    _md_escape,
    _md_table,
    _safe_convert_id_to_token,
    _safe_decode_one,
    _tag_histogram,
    _tags_for_token,
)
from exp_1d_open_answer_reasoning_probe import (  # noqa: E402
    DecodeTokenProbeCollector,
    _collect_decode_token_records,
    _reasoning_tags,
    _split_indices,
    _subsample_task_records,
)
from exp_A5_probe_split_causal import _fit_eval_binary_probe  # noqa: E402


def _atomic_json_dump(obj: Any, out_path: str) -> None:
    out_path = os.path.expanduser(out_path)
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=os.path.basename(out_path) + ".", suffix=".tmp", dir=out_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, default=EP.json_default)
        os.replace(tmp_path, out_path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def _split_csv(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def _top_tokens_for_hidden_direction(model, tok, d_hidden: np.ndarray, topk: int, include_special: bool) -> Dict[str, Any]:
    W = _get_unembedding_weight(model).detach().float()
    d = np.asarray(d_hidden, dtype=np.float32)
    d = d / max(float(np.linalg.norm(d)), 1e-12)
    d_t = torch.as_tensor(d, device=W.device, dtype=W.dtype)
    scores = (W @ d_t).detach().cpu().numpy().astype(np.float32)
    special = set(getattr(tok, "all_special_ids", []) or [])

    def _entries_from_ids(ids: np.ndarray) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for tid in ids.tolist():
            if (not include_special) and int(tid) in special:
                continue
            raw = _safe_convert_id_to_token(tok, int(tid))
            dec = _safe_decode_one(tok, int(tid))
            out.append(
                {
                    "id": int(tid),
                    "score": float(scores[int(tid)]),
                    "raw_token": raw,
                    "decoded": dec,
                    "decoded_repr": repr(dec),
                    "tags": _tags_for_token(raw, dec),
                }
            )
            if len(out) >= int(topk):
                break
        return out

    pos_idx = np.argsort(-scores)
    neg_idx = np.argsort(scores)
    top_pos = _entries_from_ids(pos_idx)
    top_neg = _entries_from_ids(neg_idx)
    return {
        "top_positive": top_pos,
        "top_negative": top_neg,
        "tag_hist_positive": _tag_histogram(top_pos),
        "tag_hist_negative": _tag_histogram(top_neg),
    }


def _collect_open_answer_records(
    *,
    model,
    tok,
    tasks: List[str],
    layer: int,
    seed: int,
    n_prompts: int,
    template_seed: int,
    template_randomization: bool,
    shuffle_choices: bool,
    add_answer_prefix: bool,
    answer_prefix: str,
    batch_size: int,
    max_prompt_len: int,
    max_new_tokens: int,
    per_task_max_states: int,
) -> Dict[str, Any]:
    sub_by, _eval_dummy, meta_by = load_selected_tasks(
        tasks=tasks,
        n_subspace=max(1, int(n_prompts)),
        n_eval=1,
        seed=int(seed),
        template_randomization=bool(template_randomization),
        template_seed=int(template_seed),
        shuffle_choices=bool(shuffle_choices),
        add_answer_prefix=bool(add_answer_prefix),
        answer_prefix=str(answer_prefix),
    )

    layers, _ = EP.get_model_layers(model)
    collector = DecodeTokenProbeCollector(int(layer))
    handle = layers[int(layer)].register_forward_hook(collector.make_hook())
    prompt_base = 0
    records_by_task: Dict[str, Any] = {}
    try:
        for task, exs in sub_by.items():
            collector.set_current_task(task)
            prompts = [ex.prompt for ex in exs]
            prompt_base = _collect_decode_token_records(
                model=model,
                tok=tok,
                prompts=prompts,
                collector=collector,
                batch_size=int(batch_size),
                max_new_tokens=int(max_new_tokens),
                max_prompt_len=int(max_prompt_len),
                prompt_base=int(prompt_base),
            )
            rec = collector.get(task)
            rec = _subsample_task_records(
                rec,
                int(per_task_max_states),
                seed=EP.stable_int_seed(seed, task, "probe_dir_unembed"),
            )
            records_by_task[task] = rec
            print(f"[Collected] task={task} states={rec.states.shape[0]}")
    finally:
        try:
            handle.remove()
        except Exception:
            pass
    return {"records_by_task": records_by_task, "meta_by_task": meta_by}


def _fit_reasoning_probe_rows(
    *,
    tok,
    records_by_task: Dict[str, Any],
    Q_basis: np.ndarray,
    tags: List[str],
    min_pos: int,
    seed: int,
    test_size: float,
) -> List[Dict[str, Any]]:
    H = np.concatenate([np.asarray(rec.states, dtype=np.float32) for rec in records_by_task.values() if rec.states.size > 0], axis=0)
    groups = np.concatenate([np.asarray(rec.prompt_ids, dtype=np.int64) for rec in records_by_task.values() if rec.states.size > 0], axis=0)
    token_ids = np.concatenate([np.asarray(rec.token_ids, dtype=np.int64) for rec in records_by_task.values() if rec.states.size > 0], axis=0)
    X = H @ np.asarray(Q_basis, dtype=np.float32)
    reasoning_rows = [_reasoning_tags(_safe_convert_id_to_token(tok, int(tid)), _safe_decode_one(tok, int(tid))) for tid in token_ids.tolist()]

    probe_rows: List[Dict[str, Any]] = []
    for tag_name in tags:
        y = np.array([int(r[tag_name]) for r in reasoning_rows], dtype=np.int64)
        n_total = int(y.shape[0])
        n_pos = int(y.sum())
        n_neg = int(n_total - n_pos)
        if n_pos < int(min_pos) or n_neg < int(min_pos):
            probe_rows.append(
                {
                    "tag": tag_name,
                    "skipped": f"Too few positives/negatives: pos={n_pos} neg={n_neg}",
                    "n_total": n_total,
                    "n_pos": n_pos,
                }
            )
            continue
        train_idx, test_idx, split_mode = _split_indices(y, groups, seed=EP.stable_int_seed(seed, tag_name), test_size=float(test_size))
        fit = _fit_eval_binary_probe(X, y, train_idx, test_idx)
        probe_rows.append(
            {
                "tag": tag_name,
                "basis_key": "reasoning_basis",
                "k": int(Q_basis.shape[1]),
                "n_total": n_total,
                "n_pos": n_pos,
                "split_mode": split_mode,
                "roc_auc": float(fit["roc_auc"]),
                "avg_precision": float(fit["avg_precision"]),
                "balanced_acc": float(fit["balanced_acc"]),
                "coef_basis_coords": np.asarray(fit["coef_shared_coords"], dtype=np.float32),
                "intercept": float(fit["intercept"]),
            }
        )
    return probe_rows


def _format_probe_direction_rows(
    *,
    format_json: str,
    format_tags: List[str],
) -> Dict[str, Any]:
    obj = json.load(open(os.path.expanduser(str(format_json)), "r", encoding="utf-8"))
    probe_fit = obj["probe_fit"]
    rows_by_tag = {str(r["tag"]): r for r in probe_fit["probe_rows"]}
    selected: List[Dict[str, Any]] = []
    missing: Dict[str, str] = {}
    for tag in format_tags:
        if tag not in rows_by_tag:
            missing[tag] = "missing"
            continue
        selected.append(rows_by_tag[tag])
    return {"selected_rows": selected, "missing": missing, "config": obj.get("config", {}), "probe_fit": probe_fit}


def _render_token_rows(entries: List[Dict[str, Any]]) -> List[List[str]]:
    rows: List[List[str]] = []
    for rank, e in enumerate(entries, start=1):
        rows.append(
            [
                str(rank),
                f"{e.get('score', 0.0):+.4f}",
                str(e.get("id")),
                _md_escape(e.get("raw_token", "")),
                _md_escape(e.get("decoded_repr", "")),
                _md_escape(",".join(e.get("tags", []) or [])),
            ]
        )
    return rows


def _render_md(
    *,
    config: Dict[str, Any],
    format_rows: List[Dict[str, Any]],
    reasoning_rows: List[Dict[str, Any]],
) -> str:
    lines: List[str] = []
    lines.append("# Exp-1g: Probe-Direction Unembedding")
    lines.append("")
    lines.append("## Config")
    lines.append("```json")
    lines.append(json.dumps(config, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")

    lines.append("## Format / readout probe directions")
    metric_rows: List[List[str]] = []
    for row in format_rows:
        metric_rows.append(
            [
                str(row["tag"]),
                str(row["n_pos"]),
                f"{row['roc_auc']:.3f}",
                f"{row['avg_precision']:.3f}",
                f"{row['balanced_acc']:.3f}",
            ]
        )
    lines.append(_md_table(metric_rows, ["tag", "n_pos", "ROC-AUC", "AP", "BalAcc"]))
    lines.append("")
    for row in format_rows:
        lines.append(f"### format tag={row['tag']}")
        lines.append("Top positive tokens:")
        lines.append(_md_table(_render_token_rows(row["unembedding"]["top_positive"]), ["rank", "score", "id", "raw", "decoded", "tags"]))
        lines.append("")
        lines.append("Positive tag histogram:")
        lines.append("```json")
        lines.append(json.dumps(row["unembedding"]["tag_hist_positive"], ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")

    lines.append("## Residual reasoning-style probe directions")
    metric_rows = []
    for row in reasoning_rows:
        if "skipped" in row:
            metric_rows.append([str(row["tag"]), str(row["n_pos"]), row["skipped"], "-", "-"])
        else:
            metric_rows.append(
                [
                    str(row["tag"]),
                    str(row["n_pos"]),
                    f"{row['roc_auc']:.3f}",
                    f"{row['avg_precision']:.3f}",
                    f"{row['balanced_acc']:.3f}",
                ]
            )
    lines.append(_md_table(metric_rows, ["tag", "n_pos", "ROC-AUC / note", "AP", "BalAcc"]))
    lines.append("")
    for row in reasoning_rows:
        if "skipped" in row:
            continue
        lines.append(f"### residual tag={row['tag']}")
        lines.append("Top positive tokens:")
        lines.append(_md_table(_render_token_rows(row["unembedding"]["top_positive"]), ["rank", "score", "id", "raw", "decoded", "tags"]))
        lines.append("")
        lines.append("Positive tag histogram:")
        lines.append("```json")
        lines.append(json.dumps(row["unembedding"]["tag_hist_positive"], ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--format_json", type=str, required=True, help="A5 JSON containing saved format/readout probe coefficients.")
    ap.add_argument("--bases_npz", type=str, required=True, help="Saved A5 bases NPZ containing Q_shared / Q_resid.")
    ap.add_argument("--format_tags", type=str, default="answer_readout,option_letter,newline")
    ap.add_argument("--reasoning_basis_key", type=str, default="Q_resid")
    ap.add_argument("--reasoning_tags", type=str, default="reasoning_marker,step_marker,digit,equation_symbol")
    ap.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--trust_remote_code", type=int, default=0, choices=[0, 1])
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--include_special", type=int, default=0, choices=[0, 1])

    ap.add_argument("--tasks", type=str, default="gsm8k,strategyqa,aqua")
    ap.add_argument("--n_prompts", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_prompt_len", type=int, default=512)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--per_task_max_states", type=int, default=6000)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--min_pos", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--template_seed", type=int, default=1234)
    ap.add_argument("--template_randomization", type=int, default=1, choices=[0, 1])
    ap.add_argument("--shuffle_choices", type=int, default=1, choices=[0, 1])
    ap.add_argument("--add_answer_prefix", type=int, default=0, choices=[0, 1])
    ap.add_argument("--answer_prefix", type=str, default="\nFinal answer:")

    ap.add_argument("--out_dir", type=str, default="results/rebuttal_mechanism/probe_dir_unembed")
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    format_tags = _split_csv(args.format_tags)
    reasoning_tags = _split_csv(args.reasoning_tags)
    tasks = _split_csv(args.tasks)
    if not format_tags:
        raise ValueError("Empty --format_tags")
    if not reasoning_tags:
        raise ValueError("Empty --reasoning_tags")
    if not tasks:
        raise ValueError("Empty --tasks")

    out_dir = os.path.expanduser(str(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)
    tag = ("_" + str(args.tag).strip()) if str(args.tag).strip() else ""
    base = f"exp_1g_probe_direction_unembedding_layer{int(args.layer)}{tag}"
    json_path = os.path.join(out_dir, base + ".json")
    md_path = os.path.join(out_dir, base + ".md")

    arrs = np.load(os.path.expanduser(str(args.bases_npz)))
    if "Q_shared" not in arrs.files:
        raise KeyError("Q_shared missing from --bases_npz")
    if str(args.reasoning_basis_key) not in arrs.files:
        raise KeyError(f"{args.reasoning_basis_key!r} missing from --bases_npz")
    Q_shared = np.asarray(arrs["Q_shared"], dtype=np.float32)
    Q_reason = np.asarray(arrs[str(args.reasoning_basis_key)], dtype=np.float32)

    EP.set_global_seed(int(args.seed))
    model, tok = EP.load_model_and_tokenizer(
        args.model,
        args.device,
        args.dtype,
        trust_remote_code=bool(args.trust_remote_code),
    )

    format_info = _format_probe_direction_rows(format_json=str(args.format_json), format_tags=format_tags)
    format_rows: List[Dict[str, Any]] = []
    for row in format_info["selected_rows"]:
        coef = np.asarray(row["coef_shared_coords"], dtype=np.float32)
        d_hidden = np.asarray(Q_shared @ coef, dtype=np.float32)
        unembedding = _top_tokens_for_hidden_direction(
            model=model,
            tok=tok,
            d_hidden=d_hidden,
            topk=int(args.topk),
            include_special=bool(args.include_special),
        )
        format_rows.append(
            {
                "tag": str(row["tag"]),
                "n_pos": int(row["n_pos"]),
                "roc_auc": float(row["roc_auc"]),
                "avg_precision": float(row["avg_precision"]),
                "balanced_acc": float(row["balanced_acc"]),
                "coef_shared_coords": coef,
                "hidden_dir_norm": float(np.linalg.norm(d_hidden)),
                "unembedding": unembedding,
            }
        )

    reason_collect = _collect_open_answer_records(
        model=model,
        tok=tok,
        tasks=tasks,
        layer=int(args.layer),
        seed=int(args.seed),
        n_prompts=int(args.n_prompts),
        template_seed=int(args.template_seed),
        template_randomization=bool(args.template_randomization),
        shuffle_choices=bool(args.shuffle_choices),
        add_answer_prefix=bool(args.add_answer_prefix),
        answer_prefix=str(args.answer_prefix),
        batch_size=int(args.batch_size),
        max_prompt_len=int(args.max_prompt_len),
        max_new_tokens=int(args.max_new_tokens),
        per_task_max_states=int(args.per_task_max_states),
    )
    reasoning_rows = _fit_reasoning_probe_rows(
        tok=tok,
        records_by_task=reason_collect["records_by_task"],
        Q_basis=Q_reason,
        tags=reasoning_tags,
        min_pos=int(args.min_pos),
        seed=int(args.seed),
        test_size=float(args.test_size),
    )
    for row in reasoning_rows:
        if "skipped" in row:
            continue
        coef = np.asarray(row["coef_basis_coords"], dtype=np.float32)
        d_hidden = np.asarray(Q_reason @ coef, dtype=np.float32)
        row["hidden_dir_norm"] = float(np.linalg.norm(d_hidden))
        row["unembedding"] = _top_tokens_for_hidden_direction(
            model=model,
            tok=tok,
            d_hidden=d_hidden,
            topk=int(args.topk),
            include_special=bool(args.include_special),
        )

    results = {
        "config": {
            "format_json": str(args.format_json),
            "bases_npz": str(args.bases_npz),
            "format_tags": format_tags,
            "reasoning_basis_key": str(args.reasoning_basis_key),
            "reasoning_tags": reasoning_tags,
            "model": str(args.model),
            "device": str(args.device),
            "dtype": str(args.dtype),
            "layer": int(args.layer),
            "topk": int(args.topk),
            "include_special": bool(args.include_special),
            "tasks": tasks,
            "n_prompts": int(args.n_prompts),
            "batch_size": int(args.batch_size),
            "max_prompt_len": int(args.max_prompt_len),
            "max_new_tokens": int(args.max_new_tokens),
            "per_task_max_states": int(args.per_task_max_states),
            "test_size": float(args.test_size),
            "min_pos": int(args.min_pos),
            "seed": int(args.seed),
            "template_seed": int(args.template_seed),
            "template_randomization": bool(args.template_randomization),
            "shuffle_choices": bool(args.shuffle_choices),
            "add_answer_prefix": bool(args.add_answer_prefix),
            "answer_prefix": str(args.answer_prefix),
        },
        "format_probe_rows": format_rows,
        "format_missing": format_info["missing"],
        "reasoning_dataset": {
            "tasks": {task: int(rec.states.shape[0]) for task, rec in reason_collect["records_by_task"].items()},
            "meta_by_task": reason_collect["meta_by_task"],
        },
        "reasoning_probe_rows": reasoning_rows,
    }

    _atomic_json_dump(results, json_path)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_render_md(config=results["config"], format_rows=format_rows, reasoning_rows=reasoning_rows))
    print(f"[Saved] {json_path}")
    print(f"[Saved] {md_path}")


if __name__ == "__main__":
    main()
