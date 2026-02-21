#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
exp_ranking_flip_steering_layer28_rand100_full.py

One-shot helper to:
  1) Generate N synthetic (random) steering vectors at a fixed layer (default: layer 28),
  2) Run `rebuttal/exp_ranking_flip_steering.py` on that manifest,
  3) Save outputs under a single output directory.

Note: These vectors are *synthetic* (random). This is useful to stress-test the pipeline and
produce a large-N ranking table, but it is not a substitute for real steering vectors.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np


def _infer_dim_from_reference(repo_root: Path) -> Optional[int]:
    # Prefer a known layer-28 Llama-2 vector if present.
    candidates = [
        repo_root / "brittleness" / "results" / "sharedspace_solid_llama2_7b_chat" / "v_pirate_decode_layer28.npy",
        repo_root / "brittleness" / "results" / "mvp_pirate_v5_story_clean" / "v_pirate_decode_layer28.npy",
    ]
    for p in candidates:
        if not p.exists():
            continue
        try:
            arr = np.load(str(p), mmap_mode="r")
        except Exception:
            continue
        if getattr(arr, "ndim", None) == 1 and int(arr.shape[0]) > 0:
            return int(arr.shape[0])
    return None


def _write_jsonl_manifest(path: Path, lines: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Auto-generated JSONL manifest (one JSON object per line)\n")
        for obj in lines:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--model_dtype", type=str, default="fp32", choices=["fp32", "fp16"])

    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--n_vectors", type=int, default=100)
    ap.add_argument("--dim", type=int, default=0, help="Vector dimension (0 = infer from repo ref, else default 4096).")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--vector_seed", type=int, default=0, help="Seed for synthetic vector generation.")

    ap.add_argument("--tasks", type=str, default="commonsenseqa,arc_challenge,openbookqa,qasc,logiqa")
    ap.add_argument("--n_eval", type=int, default=128)
    ap.add_argument("--template_seeds_rank", type=str, default="1234,2345,3456")
    ap.add_argument("--template_seeds_real", type=str, default="4567,5678,6789")

    ap.add_argument("--decoding", type=str, default="greedy", choices=["greedy", "sample"])
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--reasoning_tokens", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_prompt_len", type=int, default=512)
    ap.add_argument("--sample_seed", type=int, default=12345)

    ap.add_argument("--trad_mode", type=str, default="prefill", choices=["prefill", "both"])
    ap.add_argument("--decode_mode", type=str, default="decode", choices=["decode", "both"])
    ap.add_argument("--staged", type=int, default=1, choices=[0, 1])
    ap.add_argument("--agg", type=str, default="mean", choices=["mean", "min", "median"])
    ap.add_argument("--seed", type=int, default=42, help="Seed passed to the experiment script (data/template RNG).")

    ap.add_argument("--out_dir", type=str, default="", help="Default: results/rebuttal_rankflip_layer<layer>_rand<n>_<ts>/")
    ap.add_argument("--out_json", type=str, default="", help="Default: <out_dir>/ranking_flip_layer<layer>_rand<n>.json")
    args = ap.parse_args()

    this_dir = Path(__file__).resolve().parent
    repo_root = this_dir.parent
    exp_script = repo_root / "rebuttal" / "exp_ranking_flip_steering.py"
    if not exp_script.exists():
        raise FileNotFoundError(str(exp_script))

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else (repo_root / "results" / f"rebuttal_rankflip_layer{args.layer}_rand{args.n_vectors}_{run_id}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    dim = int(args.dim)
    if dim <= 0:
        dim = _infer_dim_from_reference(repo_root) or 4096

    vectors_dir = out_dir / "vectors"
    vectors_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(args.vector_seed))
    manifest_lines: List[dict] = []
    for i in range(int(args.n_vectors)):
        v = rng.standard_normal(size=(dim,), dtype=np.float32)
        v /= float(np.linalg.norm(v) + 1e-12)
        fname = f"rand_l{int(args.layer)}_{i:04d}.npy"
        fpath = vectors_dir / fname
        np.save(str(fpath), v)
        manifest_lines.append(
            {
                "name": f"rand_l{int(args.layer)}_{i:04d}",
                "concept": "rand",
                "layer": int(args.layer),
                "alpha": float(args.alpha),
                "path": str(fpath.relative_to(out_dir).as_posix()),
            }
        )

    manifest_path = out_dir / f"steering_vectors_layer{int(args.layer)}_rand{int(args.n_vectors)}.jsonl"
    _write_jsonl_manifest(manifest_path, manifest_lines)

    out_json = (
        Path(args.out_json)
        if args.out_json
        else (out_dir / f"ranking_flip_layer{int(args.layer)}_rand{int(args.n_vectors)}.json")
    )

    cmd = [
        sys.executable,
        str(exp_script),
        "--model",
        args.model,
        "--device",
        args.device,
        "--model_dtype",
        args.model_dtype,
        "--vectors_manifest",
        str(manifest_path),
        "--max_vectors",
        "0",
        "--tasks",
        args.tasks,
        "--n_eval",
        str(int(args.n_eval)),
        "--template_seeds_rank",
        args.template_seeds_rank,
        "--template_seeds_real",
        args.template_seeds_real,
        "--decoding",
        args.decoding,
        "--max_new_tokens",
        str(int(args.max_new_tokens)),
        "--reasoning_tokens",
        str(int(args.reasoning_tokens)),
        "--batch_size",
        str(int(args.batch_size)),
        "--max_prompt_len",
        str(int(args.max_prompt_len)),
        "--sample_seed",
        str(int(args.sample_seed)),
        "--trad_mode",
        args.trad_mode,
        "--decode_mode",
        args.decode_mode,
        "--staged",
        str(int(args.staged)),
        "--agg",
        args.agg,
        "--seed",
        str(int(args.seed)),
        "--out_json",
        str(out_json),
    ]

    (out_dir / "commands.sh").write_text(
        "# Auto-generated\n" + " ".join(shlex.quote(c) for c in cmd) + "\n",
        encoding="utf-8",
    )

    print(f"[Prep] out_dir={out_dir}")
    print(f"[Prep] vectors={args.n_vectors} layer={args.layer} dim={dim} alpha={args.alpha} vector_seed={args.vector_seed}")
    print(f"[Prep] vectors_manifest={manifest_path}")
    print(f"[Run] out_json={out_json}")
    print("[Run] cmd:")
    print("  " + " ".join(shlex.quote(c) for c in cmd))
    sys.stdout.flush()

    # Important: run from repo root so relative imports in the experiment script work.
    subprocess.run(cmd, cwd=str(repo_root), check=True)

    print(f"[Done] Saved: {out_json}")


if __name__ == "__main__":
    main()

