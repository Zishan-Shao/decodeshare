# -*- coding: utf-8 -*-
"""
exp_1e_saved_basis_focus_vocab.py

Tag-conditioned unembedding / logit-lens analysis for saved bases such as
`Q_fmt` and `Q_resid`. This is intended to be more reviewer-facing than the
full-vocabulary top-token dump by explicitly surfacing answer-readout,
reasoning-like, digit, and punctuation families.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from typing import Any, Dict, List

import numpy as np
import torch


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(THIS_DIR, "..", ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
REASONING_DIR = os.path.join(ROOT_DIR, "reasoning")

for p in [SRC_DIR, REASONING_DIR, ROOT_DIR, THIS_DIR]:
    if p not in sys.path:
        sys.path.append(p)

import eval_perf as EP  # noqa: E402
from exp_1_logit_lens_vocab_signature import (  # noqa: E402
    _get_unembedding_weight,
    _md_escape,
    _md_table,
    _safe_convert_id_to_token,
    _safe_decode_one,
    _tag_histogram,
    _tags_for_token,
)


TAG_ORDER = [
    "option_letter",
    "answer_marker",
    "yes_no",
    "reasoning_marker",
    "digit",
    "newline",
    "bracket",
    "punct",
    "whitespace",
]


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


def _basis_projection_scores(model, Q: np.ndarray) -> np.ndarray:
    W = _get_unembedding_weight(model).detach().float()
    Q_t = torch.as_tensor(np.asarray(Q, dtype=np.float32), device=W.device, dtype=W.dtype)
    proj = W @ Q_t
    # RMS over basis dimensions makes scores comparable across bases of different width
    # (e.g. Q_fmt with k=3 vs Q_resid with k=29).
    scores = torch.linalg.norm(proj, dim=1) / max(float(np.sqrt(max(1, int(Q.shape[1])))), 1.0)
    return scores.detach().cpu().numpy().astype(np.float32)


def _collect_vocab_entries(tok, scores: np.ndarray, include_special: bool) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    vocab_size = int(scores.shape[0])
    special = set(getattr(tok, "all_special_ids", []) or [])
    for tid in range(vocab_size):
        if (not include_special) and (tid in special):
            continue
        raw = _safe_convert_id_to_token(tok, tid)
        dec = _safe_decode_one(tok, tid)
        tags = _tags_for_token(raw, dec)
        out.append(
            {
                "id": int(tid),
                "score": float(scores[tid]),
                "raw_token": raw,
                "decoded": dec,
                "decoded_repr": repr(dec),
                "tags": tags,
            }
        )
    return out


def _top_by_tag(entries: List[Dict[str, Any]], tag: str, topk: int) -> List[Dict[str, Any]]:
    hits = [e for e in entries if tag in (e.get("tags", []) or [])]
    hits.sort(key=lambda e: (-float(e["score"]), int(e["id"])))
    return hits[: int(topk)]


def _family_summary(entries: List[Dict[str, Any]], tag: str, topk: int) -> Dict[str, Any]:
    hits = [e for e in entries if tag in (e.get("tags", []) or [])]
    if not hits:
        return {
            "tag": tag,
            "n_vocab_hits": 0,
            "max_score": 0.0,
            "mean_score": 0.0,
            "top_tokens": [],
        }
    scores = np.asarray([float(e["score"]) for e in hits], dtype=np.float32)
    top_hits = sorted(hits, key=lambda e: (-float(e["score"]), int(e["id"])))[: int(topk)]
    return {
        "tag": tag,
        "n_vocab_hits": int(len(hits)),
        "max_score": float(scores.max()),
        "mean_score": float(scores.mean()),
        "top_tokens": top_hits,
    }


def _render_md(config: Dict[str, Any], basis_meta: Dict[str, Any], summaries: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Exp-1e: Saved-Basis Focus Vocab Signature")
    lines.append("")
    lines.append("## Config")
    lines.append("```json")
    lines.append(json.dumps(config, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Basis")
    lines.append("```json")
    lines.append(json.dumps(basis_meta, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")

    rows = []
    for tag in TAG_ORDER:
        fam = summaries["families"].get(tag, {})
        rows.append(
            [
                tag,
                str(fam.get("n_vocab_hits", 0)),
                f"{fam.get('max_score', 0.0):.4f}",
                f"{fam.get('mean_score', 0.0):.4f}",
            ]
        )
    lines.append("## Family Summary")
    lines.append(_md_table(rows, ["tag", "n_vocab_hits", "max_score", "mean_score"]))
    lines.append("")

    lines.append("## Top Tokens By Tag")
    for tag in TAG_ORDER:
        fam = summaries["families"].get(tag, {})
        top = fam.get("top_tokens", []) or []
        lines.append(f"### tag={tag}")
        rows = []
        for rank, e in enumerate(top, start=1):
            rows.append(
                [
                    str(rank),
                    f"{e.get('score', 0.0):.4f}",
                    str(e.get("id")),
                    _md_escape(e.get("raw_token", "")),
                    _md_escape(e.get("decoded_repr", "")),
                    _md_escape(",".join(e.get("tags", []) or [])),
                ]
            )
        lines.append(_md_table(rows, ["rank", "score", "id", "raw", "decoded", "tags"]))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--basis_npz", type=str, required=True)
    ap.add_argument("--basis_keys", type=str, default="Q_fmt,Q_resid")
    ap.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--trust_remote_code", type=int, default=0, choices=[0, 1])
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--include_special", type=int, default=0, choices=[0, 1])
    ap.add_argument("--layer", type=int, default=-1, help="Only used for filenames/metadata.")
    ap.add_argument("--out_dir", type=str, default="results/rebuttal_mechanism/focus_vocab_saved")
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    basis_keys = _split_csv(args.basis_keys)
    if not basis_keys:
        raise ValueError("Empty --basis_keys")

    arrs = np.load(os.path.expanduser(str(args.basis_npz)))
    out_dir = os.path.expanduser(str(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)
    tag = ("_" + str(args.tag).strip()) if str(args.tag).strip() else ""

    model, tok = EP.load_model_and_tokenizer(
        args.model,
        args.device,
        args.dtype,
        trust_remote_code=bool(args.trust_remote_code),
    )

    config = {
        "basis_npz": str(args.basis_npz),
        "basis_keys": basis_keys,
        "model": str(args.model),
        "device": str(args.device),
        "dtype": str(args.dtype),
        "layer": int(args.layer),
        "topk": int(args.topk),
        "include_special": bool(args.include_special),
        "tags": TAG_ORDER,
    }

    all_results: Dict[str, Any] = {"config": config, "by_basis_key": {}}
    for basis_key in basis_keys:
        if basis_key not in arrs.files:
            raise KeyError(f"basis_key={basis_key!r} missing from {args.basis_npz}")
        Q = np.asarray(arrs[basis_key], dtype=np.float32)
        basis_meta = {
            "basis_key": basis_key,
            "shape": list(Q.shape),
            "saved_basis_npz": str(args.basis_npz),
        }
        scores = _basis_projection_scores(model, Q)
        entries = _collect_vocab_entries(tok, scores, include_special=bool(args.include_special))
        families = {tag_name: _family_summary(entries, tag_name, topk=int(args.topk)) for tag_name in TAG_ORDER}
        summary = {
            "basis_meta": basis_meta,
            "aggregate_tag_hist": _tag_histogram(sorted(entries, key=lambda e: -float(e["score"]))[: int(args.topk)]),
            "families": families,
        }
        md = _render_md(config, basis_meta, summary)
        base = f"exp_1e_saved_basis_focus_vocab_{basis_key}_layer{int(args.layer)}{tag}"
        json_path = os.path.join(out_dir, base + ".json")
        md_path = os.path.join(out_dir, base + ".md")
        _atomic_json_dump(summary, json_path)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md)
        all_results["by_basis_key"][basis_key] = {
            "basis_meta": basis_meta,
            "saved_json": os.path.relpath(json_path, ROOT_DIR),
            "saved_md": os.path.relpath(md_path, ROOT_DIR),
        }
        print(f"[Saved] {json_path}")
        print(f"[Saved] {md_path}")

    summary_json = os.path.join(out_dir, f"exp_1e_saved_basis_focus_vocab_all_layer{int(args.layer)}{tag}.json")
    _atomic_json_dump(all_results, summary_json)
    print(f"[Saved] {summary_json}")


if __name__ == "__main__":
    main()
