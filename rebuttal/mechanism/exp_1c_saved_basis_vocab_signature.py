# -*- coding: utf-8 -*-
"""
exp_1c_saved_basis_vocab_signature.py

Run the logit-lens / vocabulary-signature analysis on one or more saved bases
stored in an NPZ, e.g. `Q_fmt` / `Q_resid` from exp_A5_probe_split_causal.py.
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
    _compute_vocab_signature,
    _render_md_summary,
)


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--basis_npz", type=str, required=True)
    ap.add_argument("--basis_keys", type=str, default="Q_fmt,Q_resid")
    ap.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--trust_remote_code", type=int, default=0, choices=[0, 1])
    ap.add_argument("--topk", type=int, default=40)
    ap.add_argument("--include_special", type=int, default=0, choices=[0, 1])
    ap.add_argument("--per_direction", type=int, default=1, choices=[0, 1])
    ap.add_argument("--md_max_dirs", type=int, default=8)
    ap.add_argument("--layer", type=int, default=-1, help="Only used for filenames/metadata.")
    ap.add_argument("--out_dir", type=str, default="results/rebuttal_mechanism/logit_lens_saved")
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

    all_results: Dict[str, Any] = {
        "config": {
            "basis_npz": str(args.basis_npz),
            "basis_keys": basis_keys,
            "model": str(args.model),
            "device": str(args.device),
            "dtype": str(args.dtype),
            "layer": int(args.layer),
            "topk": int(args.topk),
            "include_special": bool(args.include_special),
            "per_direction": bool(args.per_direction),
            "md_max_dirs": int(args.md_max_dirs),
        },
        "by_basis_key": {},
    }

    for basis_key in basis_keys:
        if basis_key not in arrs.files:
            raise KeyError(f"basis_key={basis_key!r} missing from {args.basis_npz}")
        Q = np.asarray(arrs[basis_key], dtype=np.float32)
        basis_meta = {
            "basis_key": basis_key,
            "shape": list(Q.shape),
            "saved_basis_npz": str(args.basis_npz),
        }
        signature = _compute_vocab_signature(
            model=model,
            tok=tok,
            Q=Q,
            topk=int(args.topk),
            include_special=bool(args.include_special),
            per_direction=bool(args.per_direction),
        )
        md = _render_md_summary(
            config=all_results["config"],
            basis_meta=basis_meta,
            signature=signature,
            md_max_dirs=int(args.md_max_dirs),
        )
        base = f"exp_1c_saved_basis_vocab_signature_{basis_key}_layer{int(args.layer)}{tag}"
        json_path = os.path.join(out_dir, base + ".json")
        md_path = os.path.join(out_dir, base + ".md")
        _atomic_json_dump(
            {
                "config": all_results["config"],
                "basis_meta": basis_meta,
                "signature": signature,
            },
            json_path,
        )
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md)
        all_results["by_basis_key"][basis_key] = {
            "basis_meta": basis_meta,
            "saved_json": os.path.relpath(json_path, ROOT_DIR),
            "saved_md": os.path.relpath(md_path, ROOT_DIR),
        }
        print(f"[Saved] {json_path}")
        print(f"[Saved] {md_path}")

    summary_json = os.path.join(out_dir, f"exp_1c_saved_basis_vocab_signature_all_layer{int(args.layer)}{tag}.json")
    _atomic_json_dump(all_results, summary_json)
    print(f"[Saved] {summary_json}")


if __name__ == "__main__":
    main()
