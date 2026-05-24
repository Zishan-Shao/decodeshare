#!/usr/bin/env python3

"""Full ranking-flip runner with manifest discovery and layer sweeps."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None


def _sanitize_name(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "vec"


def _load_layer_from_json(path: Path) -> Optional[int]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(obj, dict):
        v = obj.get("layer", None)
        if isinstance(v, int):
            return int(v)
        cfg = obj.get("config", None)
        if isinstance(cfg, dict) and isinstance(cfg.get("layer", None), int):
            return int(cfg["layer"])
        cfg2 = obj.get("config", None)
        if isinstance(cfg2, dict) and isinstance(cfg2.get("config", None), dict):
            v2 = cfg2["config"].get("layer", None)
            if isinstance(v2, int):
                return int(v2)
    return None


def _infer_layer(vec_path: Path, repo_root: Path) -> Optional[int]:
    m = re.search(r"(?:layer|l)(\d+)", vec_path.name)
    if m:
        return int(m.group(1))
    m = re.search(r"(?:layer|l)(\d+)", str(vec_path))
    if m:
        return int(m.group(1))

    cur = vec_path.parent.resolve()
    root = repo_root.resolve()
    for _ in range(20):
        if cur == cur.parent:
            break
        for run_file in ("run_config.json", "run_meta.json"):
            cand = cur / run_file
            if cand.exists():
                layer = _load_layer_from_json(cand)
                if layer is not None:
                    return layer
        if cur == root:
            break
        cur = cur.parent
    return None


def _infer_concept(vec_path: Path) -> str:
    s = str(vec_path).lower()
    for c in ("pirate", "boolq", "rte", "sst2", "fixed"):
        if c in s:
            return c
    return _sanitize_name(vec_path.parent.name)


def _iter_candidate_vector_paths(repo_root: Path) -> Iterable[Path]:

    dirs = [
        repo_root / "brittleness" / "old" / "steer_repair_multibench",
        repo_root / "brittleness" / "old" / "steer_repair_multibench_v2",
        repo_root / "brittleness" / "old" / "steer_repair_multibench_v2_enhanced",
        repo_root / "brittleness" / "results" / "steer_repair_multibench_v3",
        repo_root / "brittleness" / "results" / "sharedspace_solid_llama2_7b_chat",
        repo_root / "brittleness" / "results" / "mvp_pirate_v5_story_clean",
    ]
    for d in dirs:
        if not d.exists():
            continue
        yield from sorted(d.rglob("*.npy"))


def _is_vector_npy(path: Path) -> bool:
    try:
        arr = np.load(str(path), mmap_mode="r")
    except Exception:
        return False

    return getattr(arr, "ndim", None) == 1 and int(arr.shape[0]) >= 16


def _build_manifest(
    *,
    repo_root: Path,
    out_manifest: Path,
    max_vectors: int,
) -> Tuple[Path, int]:
    items: List[Dict[str, Any]] = []
    seen_names: set[str] = set()

    for p in _iter_candidate_vector_paths(repo_root):
        if not _is_vector_npy(p):
            continue
        layer = _infer_layer(p, repo_root)
        if layer is None:
            continue
        rel = p.relative_to(repo_root).as_posix()
        base_name = _sanitize_name(rel.replace("/", "__").removesuffix(".npy"))
        name = base_name
        k = 2
        while name in seen_names:
            name = f"{base_name}__{k}"
            k += 1
        seen_names.add(name)

        items.append(
            {
                "name": name,
                "concept": _infer_concept(p),
                "layer": int(layer),
                "alpha": 1.0,
                "path": str(p.resolve()),
            }
        )
        if max_vectors > 0 and len(items) >= max_vectors:
            break

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with out_manifest.open("w", encoding="utf-8") as f:
        f.write("# Auto-generated steering vectors manifest (JSONL; 1 JSON object per line)\n")
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    return out_manifest, len(items)


def _parse_csv_ints(s: str) -> List[int]:
    s = str(s or "").strip()
    if not s:
        return []
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _parse_csv_strs(s: str) -> List[str]:
    s = str(s or "").strip()
    if not s:
        return []
    return [p.strip() for p in s.split(",") if p.strip()]


def _model_tag(model_name: str) -> str:

    s = model_name.replace("/", "__")
    return _sanitize_name(s)


def _infer_dim_from_reference(repo_root: Path) -> Optional[int]:

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


def _maybe_generate_random_vectors(*, vectors_dir: Path, n_vectors: int, dim: int, vector_seed: int) -> List[Path]:
    vectors_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []

    existing = sorted(vectors_dir.glob("rand_*.npy"))
    if len(existing) >= n_vectors:
        return existing[:n_vectors]

    for i in range(n_vectors):
        fpath = vectors_dir / f"rand_{i:04d}.npy"
        if fpath.exists():
            paths.append(fpath)
            continue

        rng_i = np.random.default_rng(int(vector_seed) + int(i))
        v = rng_i.standard_normal(size=(dim,), dtype=np.float32)
        v /= float(np.linalg.norm(v) + 1e-12)
        np.save(str(fpath), v)
        paths.append(fpath)
    return paths


def _rankdata_average(a: np.ndarray) -> np.ndarray:
    n = a.shape[0]
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and a[order[j + 1]] == a[order[i]]:
            j += 1
        r = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = r
        i = j + 1
    return ranks


def _spearmanr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size != b.size or a.size < 2:
        return float("nan")
    ra = _rankdata_average(a)
    rb = _rankdata_average(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(np.sqrt(np.sum(ra * ra) * np.sum(rb * rb)))
    if denom <= 0:
        return float("nan")
    return float(np.sum(ra * rb) / denom)


def _log_choose(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _hypergeom_p_geq(*, n_pop: int, n_pos: int, n_draw: int, k_geq: int) -> float:

    if k_geq <= 0:
        return 1.0
    if n_pos <= 0:
        return 0.0
    if n_draw <= 0:
        return 0.0
    k_max = min(n_draw, n_pos)
    if k_geq > k_max:
        return 0.0
    den = _log_choose(n_pop, n_draw)
    tot = 0.0
    for k in range(k_geq, k_max + 1):
        tot += math.exp(_log_choose(n_pos, k) + _log_choose(n_pop - n_pos, n_draw - k) - den)
    return float(min(max(tot, 0.0), 1.0))


def _compute_decision_summary(results: Dict[str, Any], *, k_list: List[int]) -> Dict[str, Any]:
    vecs = results.get("vectors", {})
    if not isinstance(vecs, dict) or not vecs:
        raise RuntimeError("Missing/empty vectors in results JSON.")

    names = sorted(vecs.keys())
    trad = np.array([float(vecs[n]["score_rank_trad"]) for n in names], dtype=np.float64)
    dec = np.array([float(vecs[n]["score_rank_decode"]) for n in names], dtype=np.float64)
    real = np.array([float(vecs[n]["score_real"]) for n in names], dtype=np.float64)

    idx_trad = np.argsort(-trad)
    idx_dec = np.argsort(-dec)
    idx_real = np.argsort(-real)

    n_pop = int(len(names))
    n_pos_total = int(np.sum(real > 0))

    out: Dict[str, Any] = {
        "n_vectors": n_pop,
        "n_pos_total": n_pos_total,
        "pos_rate": float(n_pos_total / max(1, n_pop)),
        "correlations": {
            "spearman_trad_vs_decode": _spearmanr(trad, dec),
            "spearman_trad_vs_real": _spearmanr(trad, real),
            "spearman_decode_vs_real": _spearmanr(dec, real),
        },
        "topk": {},
        "oracle": {},
    }

    for k in k_list:
        k = int(k)
        k = min(k, n_pop)
        sel_tr = idx_trad[:k]
        sel_de = idx_dec[:k]
        sel_or = idx_real[:k]

        mean_tr = float(real[sel_tr].mean())
        mean_de = float(real[sel_de].mean())
        mean_or = float(real[sel_or].mean())

        npos_tr = int(np.sum(real[sel_tr] > 0))
        npos_de = int(np.sum(real[sel_de] > 0))

        out["topk"][str(k)] = {
            "mean_real_trad": mean_tr,
            "mean_real_decode": mean_de,
            "mean_real_oracle": mean_or,
            "regret_trad": float(mean_or - mean_tr),
            "regret_decode": float(mean_or - mean_de),
            "delta_decode_minus_trad": float(mean_de - mean_tr),
            "n_pos_trad": npos_tr,
            "n_pos_decode": npos_de,
            "p_geq_npos_trad": _hypergeom_p_geq(n_pop=n_pop, n_pos=n_pos_total, n_draw=k, k_geq=npos_tr),
            "p_geq_npos_decode": _hypergeom_p_geq(n_pop=n_pop, n_pos=n_pos_total, n_draw=k, k_geq=npos_de),
        }


    def _topk_names(idx: np.ndarray, k: int) -> List[str]:
        return [names[i] for i in idx[:k]]

    for k in [10, 20, 50]:
        if k > n_pop:
            continue
        a = set(_topk_names(idx_trad, k))
        b = set(_topk_names(idx_dec, k))
        c = set(_topk_names(idx_real, k))
        out["oracle"][str(k)] = {
            "overlap_trad_decode": int(len(a & b)),
            "overlap_trad_real": int(len(a & c)),
            "overlap_decode_real": int(len(b & c)),
        }

    return out


def _merge_shards(shard_paths: List[Path], *, merged_out: Path) -> Dict[str, Any]:
    merged: Optional[Dict[str, Any]] = None
    merged_vectors: Dict[str, Any] = {}
    merged_from: List[str] = []

    for sp in shard_paths:
        obj = json.loads(sp.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            raise RuntimeError(f"Shard JSON is not an object: {sp}")
        if merged is None:
            merged = obj
        else:

            a = (merged.get("config") or {})
            b = (obj.get("config") or {})
            for key in ["model", "tasks", "n_eval", "template_seeds_rank", "template_seeds_real", "decoding"]:
                if key in a and key in b and a[key] != b[key]:
                    raise RuntimeError(f"Shard config mismatch for key={key}: {sp}")

        vecs = obj.get("vectors", {})
        if not isinstance(vecs, dict):
            raise RuntimeError(f"Shard missing vectors dict: {sp}")
        for name, rec in vecs.items():
            if name in merged_vectors:
                raise RuntimeError(f"Duplicate vector name across shards: {name}")
            merged_vectors[name] = rec
        merged_from.append(str(sp.name))

    assert merged is not None
    merged["vectors"] = merged_vectors
    merged["merged_from"] = merged_from


    cfg = merged.get("config", {})
    if isinstance(cfg, dict):
        cfg = dict(cfg)
        cfg["out_json"] = str(merged_out)
        cfg["start_idx"] = 0
        cfg["end_idx"] = -1
        cfg["resume"] = 0
        merged["config"] = cfg


    names = sorted(merged_vectors.keys())
    trad = np.array([float(merged_vectors[n]["score_rank_trad"]) for n in names], dtype=np.float64)
    dec = np.array([float(merged_vectors[n]["score_rank_decode"]) for n in names], dtype=np.float64)
    real = np.array([float(merged_vectors[n]["score_real"]) for n in names], dtype=np.float64)
    merged["correlations"] = {
        "spearman_trad_vs_decode": _spearmanr(trad, dec),
        "spearman_trad_vs_real": _spearmanr(trad, real),
        "spearman_decode_vs_real": _spearmanr(dec, real),
    }
    merged["progress"] = {
        "n_vectors_total": int(len(names)),
        "n_vectors_done": int(len(names)),
        "merged_shards": int(len(shard_paths)),
    }

    merged_out.parent.mkdir(parents=True, exist_ok=True)
    merged_out.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return merged


def _count_done_vectors(out_json: Path) -> int:
    if not out_json.exists():
        return 0
    try:
        obj = json.loads(out_json.read_text(encoding="utf-8"))
    except Exception:
        return 0
    if not isinstance(obj, dict):
        return 0
    vecs = obj.get("vectors", {})
    if not isinstance(vecs, dict):
        return 0
    return int(len(vecs))


def _write_sweep_summary(out_root: Path, summary_rows: List[Dict[str, Any]]) -> None:
    (out_root / "sweep_summary.json").write_text(
        json.dumps({"rows": summary_rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not summary_rows:
        return
    cols = sorted({k for r in summary_rows for k in r.keys()})
    lines = [",".join(cols)]
    for r in summary_rows:
        lines.append(",".join(str(r.get(c, "")) for c in cols))
    (out_root / "sweep_summary.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    ap.add_argument("--models", type=str, default="", help="Optional comma-separated model list (model sweep).")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--model_dtype", type=str, default="fp32", choices=["fp32", "fp16"])

    ap.add_argument(
        "--candidate_pool",
        type=str,
        default="auto",
        choices=["auto", "random"],
        help="Vector candidate pool. 'auto' scans repo for existing vectors; 'random' generates synthetic unit vectors.",
    )
    ap.add_argument("--vectors_manifest", type=str, default="", help="If empty and --candidate_pool=auto: auto-build a manifest.")
    ap.add_argument("--max_vectors", type=int, default=100, help="Cap the number of vectors in the run (0=no limit).")
    ap.add_argument("--filter_regex", type=str, default="", help="Optional regex filter applied by the experiment script.")

    ap.add_argument("--layers", type=str, default="", help="Comma-separated layer list for a layer sweep, e.g. 16,20,24,28,31.")
    ap.add_argument("--n_vectors", type=int, default=100, help="(random pool) Number of candidate vectors.")
    ap.add_argument("--vector_seed", type=int, default=0, help="(random pool) RNG seed for candidate vectors.")
    ap.add_argument("--dim", type=int, default=0, help="(random pool) Vector dim (0=infer from repo ref; else use value).")
    ap.add_argument("--alpha", type=float, default=1.0, help="(random pool) Steering alpha for all candidates.")

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
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--shards", type=int, default=1, help="Number of vector shards to run (for multi-GPU parallelism).")
    ap.add_argument("--cuda_devices", type=str, default="", help="Comma-separated CUDA_VISIBLE_DEVICES ids for shards, e.g. 0,1,2,3.")
    ap.add_argument("--resume", type=int, default=1, choices=[0, 1], help="Pass --resume to the experiment script.")
    ap.add_argument("--save_every", type=int, default=1, help="Pass --save_every to the experiment script.")
    ap.add_argument("--k_list", type=str, default="1,5,10,20", help="k values for decision utility summaries.")
    ap.add_argument("--poll_seconds", type=int, default=60,
                    help="In sweep mode: poll per-shard JSONs every N seconds and show a progress bar (0 disables).")
    ap.add_argument("--dry_run", type=int, default=0, choices=[0, 1], help="If 1: write commands, but do not execute.")
    ap.add_argument("--keep_going", type=int, default=0, choices=[0, 1], help="If 1: continue sweep on failures.")
    ap.add_argument("--skip_existing", type=int, default=1, choices=[0, 1], help="If 1: skip runs with existing merged output.")

    ap.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="If empty: outputs/steering_rank_flip/<run>/ under the repo root.",
    )
    ap.add_argument("--out_json", type=str, default="", help="If empty: <out_dir>/ranking_flip_full.json")
    args = ap.parse_args()

    this_dir = Path(__file__).resolve().parent
    downstream_root = this_dir.parent
    repo_root = downstream_root.parent
    exp_script = this_dir / "exp_rank_flip.py"
    if not exp_script.exists():
        raise FileNotFoundError(str(exp_script))

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    layers = _parse_csv_ints(args.layers)
    models = _parse_csv_strs(args.models) or [args.model]
    k_list = _parse_csv_ints(args.k_list) or [1, 5, 10, 20]


    if layers:
        out_root = Path(args.out_dir) if args.out_dir else (repo_root / "outputs" / "steering_rank_flip" / f"sweep_{run_id}")
        out_root.mkdir(parents=True, exist_ok=True)

        summary_rows: List[Dict[str, Any]] = []

        for model_name in models:
            model_dir = out_root / _model_tag(model_name)
            model_dir.mkdir(parents=True, exist_ok=True)

            if args.candidate_pool != "random":
                raise RuntimeError("Layer/model sweeps require --candidate_pool=random (fixed candidate pool).")

            dim = int(args.dim) if int(args.dim) > 0 else (_infer_dim_from_reference(downstream_root) or 4096)
            n_vectors = int(args.n_vectors)
            vectors_dir = model_dir / f"rand_vectors_n{n_vectors}_d{dim}_seed{int(args.vector_seed)}"
            vec_paths = _maybe_generate_random_vectors(
                vectors_dir=vectors_dir,
                n_vectors=n_vectors,
                dim=dim,
                vector_seed=int(args.vector_seed),
            )

            for layer in layers:
                run_dir = model_dir / f"layer{int(layer)}"
                run_dir.mkdir(parents=True, exist_ok=True)

                merged_out = run_dir / "ranking_flip.json"
                if bool(args.skip_existing) and merged_out.exists():
                    print(f"[Skip] {model_name} layer={layer}: {merged_out} exists")
                    continue

                manifest_lines: List[dict] = []
                for i, vp in enumerate(vec_paths):
                    manifest_lines.append(
                        {
                            "name": f"rand_{i:04d}",
                            "concept": "rand",
                            "layer": int(layer),
                            "alpha": float(args.alpha),
                            "path": str(vp.resolve()),
                        }
                    )

                manifest_path = run_dir / f"steering_vectors_rand_n{n_vectors}_seed{int(args.vector_seed)}_layer{int(layer)}.jsonl"
                _write_jsonl_manifest(manifest_path, manifest_lines)

                shards = max(int(args.shards), 1)
                cuda_devices = _parse_csv_strs(args.cuda_devices)
                shard_paths: List[Path] = []
                shard_cmds: List[List[str]] = []

                for s in range(shards):
                    start = int((s * n_vectors) // shards)
                    end = int(((s + 1) * n_vectors) // shards)
                    shard_out = run_dir / f"ranking_flip_shard{s:02d}.json"
                    shard_paths.append(shard_out)
                    cmd = [
                        sys.executable,
                        str(exp_script),
                        "--model",
                        model_name,
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
                        str(args.n_eval),
                        "--template_seeds_rank",
                        args.template_seeds_rank,
                        "--template_seeds_real",
                        args.template_seeds_real,
                        "--decoding",
                        args.decoding,
                        "--max_new_tokens",
                        str(args.max_new_tokens),
                        "--reasoning_tokens",
                        str(args.reasoning_tokens),
                        "--batch_size",
                        str(args.batch_size),
                        "--max_prompt_len",
                        str(args.max_prompt_len),
                        "--sample_seed",
                        str(args.sample_seed),
                        "--trad_mode",
                        args.trad_mode,
                        "--decode_mode",
                        args.decode_mode,
                        "--staged",
                        str(args.staged),
                        "--agg",
                        args.agg,
                        "--seed",
                        str(args.seed),
                        "--out_json",
                        str(shard_out),
                        "--start_idx",
                        str(start),
                        "--end_idx",
                        str(end),
                        "--resume",
                        str(int(args.resume)),
                        "--save_every",
                        str(int(args.save_every)),
                    ]
                    if args.filter_regex:
                        cmd += ["--filter_regex", args.filter_regex]
                    shard_cmds.append(cmd)


                (run_dir / "commands_shards.sh").write_text(
                    "# Auto-generated\n" + "\n".join(" ".join(shlex.quote(c) for c in cmd) for cmd in shard_cmds) + "\n",
                    encoding="utf-8",
                )

                if bool(args.dry_run):
                    print(f"[Dry-run] Prepared: {run_dir}")
                    continue

                procs: List[subprocess.Popen] = []
                log_files = []
                for s, cmd in enumerate(shard_cmds):
                    env = os.environ.copy()
                    if cuda_devices:
                        if s >= len(cuda_devices):
                            raise RuntimeError(f"--cuda_devices has {len(cuda_devices)} entries but --shards={shards}")
                        env["CUDA_VISIBLE_DEVICES"] = str(cuda_devices[s])
                    log_path = run_dir / f"shard{s:02d}.log"
                    log_f = open(log_path, "w", encoding="utf-8")
                    log_files.append(log_f)
                    print(f"[Run] {model_name} layer={layer} shard={s} -> {shard_paths[s].name} (log={log_path.name})")
                    procs.append(
                        subprocess.Popen(cmd, cwd=str(repo_root), env=env, stdout=log_f, stderr=subprocess.STDOUT)
                    )

                poll_s = int(getattr(args, "poll_seconds", 0) or 0)
                pbar = None
                last_total_done = 0
                if poll_s > 0:
                    if tqdm is not None:
                        pbar = tqdm(total=n_vectors, desc=f"{_model_tag(model_name)}:layer{int(layer)}", unit="vec")
                    else:
                        print(f"[Progress] Polling every {poll_s}s (tqdm not available).")

                if poll_s > 0:
                    while True:
                        alive = any(p.poll() is None for p in procs)
                        shard_done = [_count_done_vectors(pth) for pth in shard_paths]
                        total_done = int(sum(shard_done))
                        if total_done < last_total_done:
                            total_done = last_total_done
                        last_total_done = total_done

                        if pbar is not None:
                            pbar.n = total_done
                            try:
                                pbar.set_postfix_str(" ".join(f"s{i}={c}" for i, c in enumerate(shard_done)))
                            except Exception:
                                pass
                            pbar.refresh()
                        else:
                            print(f"[Progress] {model_name} layer={layer}: {total_done}/{n_vectors} vectors done")

                        if not alive:
                            break
                        time.sleep(float(poll_s))

                ok = True
                for p in procs:
                    rc = p.wait()
                    ok = ok and (rc == 0)
                for f in log_files:
                    try:
                        f.close()
                    except Exception:
                        pass
                if pbar is not None:
                    try:
                        pbar.n = n_vectors
                        pbar.refresh()
                        pbar.close()
                    except Exception:
                        pass

                if not ok:
                    msg = f"One or more shards failed for model={model_name} layer={layer}"
                    if bool(args.keep_going):
                        print(f"[Warn] {msg}")
                        continue
                    raise RuntimeError(msg)

                merged = _merge_shards(shard_paths, merged_out=merged_out)
                decision = _compute_decision_summary(merged, k_list=k_list)
                (run_dir / "decision_summary.json").write_text(
                    json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )

                summary_rows.append(
                    {
                        "model": model_name,
                        "layer": int(layer),
                        "n_vectors": decision["n_vectors"],
                        "n_pos_total": decision["n_pos_total"],
                        **{f"corr_{k}": v for k, v in decision["correlations"].items()},
                        **{f"k{kk}_mean_real_trad": decision["topk"][str(kk)]["mean_real_trad"] for kk in k_list},
                        **{f"k{kk}_mean_real_decode": decision["topk"][str(kk)]["mean_real_decode"] for kk in k_list},
                        **{f"k{kk}_regret_trad": decision["topk"][str(kk)]["regret_trad"] for kk in k_list},
                        **{f"k{kk}_regret_decode": decision["topk"][str(kk)]["regret_decode"] for kk in k_list},
                        **{f"k{kk}_delta_decode_minus_trad": decision["topk"][str(kk)]["delta_decode_minus_trad"] for kk in k_list},
                        **{f"k{kk}_npos_trad": decision["topk"][str(kk)]["n_pos_trad"] for kk in k_list},
                        **{f"k{kk}_npos_decode": decision["topk"][str(kk)]["n_pos_decode"] for kk in k_list},
                    }
                )
                _write_sweep_summary(out_root, summary_rows)


        _write_sweep_summary(out_root, summary_rows)

        print(f"[Done] Wrote sweep summary under: {out_root}")
        return


    out_dir = Path(args.out_dir) if args.out_dir else (repo_root / "outputs" / "steering_rank_flip" / f"full_{run_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.candidate_pool == "random":
        raise RuntimeError("For random candidate pool, pass --layers (even a single layer) to enable sweep mode.")

    if args.vectors_manifest:
        manifest = Path(os.path.expanduser(args.vectors_manifest)).resolve()
        if not manifest.exists():
            raise FileNotFoundError(str(manifest))
        n_vecs = None
    else:
        manifest, n_vecs = _build_manifest(
            repo_root=downstream_root,
            out_manifest=out_dir / "steering_vectors_full.jsonl",
            max_vectors=int(args.max_vectors),
        )
        if n_vecs is not None and n_vecs < 30:
            raise RuntimeError(
                f"Auto-manifest only found {n_vecs} vectors (<30). "
                "Pass --vectors_manifest to use your own JSONL with 30-100 vectors."
            )

    out_json = Path(args.out_json) if args.out_json else (out_dir / "ranking_flip_full.json")

    cmd: List[str] = [
        sys.executable,
        str(exp_script),
        "--model",
        args.model,
        "--device",
        args.device,
        "--model_dtype",
        args.model_dtype,
        "--vectors_manifest",
        str(manifest),
        "--max_vectors",
        str(args.max_vectors),
        "--tasks",
        args.tasks,
        "--n_eval",
        str(args.n_eval),
        "--template_seeds_rank",
        args.template_seeds_rank,
        "--template_seeds_real",
        args.template_seeds_real,
        "--decoding",
        args.decoding,
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--reasoning_tokens",
        str(args.reasoning_tokens),
        "--batch_size",
        str(args.batch_size),
        "--max_prompt_len",
        str(args.max_prompt_len),
        "--sample_seed",
        str(args.sample_seed),
        "--trad_mode",
        args.trad_mode,
        "--decode_mode",
        args.decode_mode,
        "--staged",
        str(args.staged),
        "--agg",
        args.agg,
        "--seed",
        str(args.seed),
        "--out_json",
        str(out_json),
    ]
    if args.filter_regex:
        cmd += ["--filter_regex", args.filter_regex]

    (out_dir / "commands.sh").write_text(
        "# Auto-generated\n" + " ".join(shlex.quote(c) for c in cmd) + "\n",
        encoding="utf-8",
    )

    print(f"[Run] out_dir={out_dir}")
    print(f"[Run] vectors_manifest={manifest}" + (f" (n_vectors={n_vecs})" if n_vecs is not None else ""))
    print(f"[Run] out_json={out_json}")
    print("[Run] cmd:")
    print("  " + " ".join(cmd))
    sys.stdout.flush()

    if not bool(args.dry_run):
        subprocess.run(cmd, cwd=str(repo_root), check=True)

    print(f"[Done] Saved: {out_json}")


if __name__ == "__main__":
    main()
