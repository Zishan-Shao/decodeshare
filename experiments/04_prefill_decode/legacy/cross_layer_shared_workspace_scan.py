# -*- coding: utf-8 -*-
"""cross_layer_shared_workspace_scan.py

Scan a *small set of middle layers* (4–6 layers recommended) and estimate a
decode-time shared workspace per layer, then analyze how these layerwise
workspaces relate.

Why this supports the paper story:
  - Your main paper already argues a decode-time shared subspace exists at (many)
    layers and is causally important.
  - This script adds an *extra* analysis layer: are the per-layer shared
    workspaces independent, or do they form a coherent "workspace tube" carried
    by the residual stream (persisting / slowly rotating across layers)?

What this script outputs:
  1) Per-layer: cross-task PCA dim, shared basis size, and energy ratio.
  2) Pairwise subspace similarity between layers (principal-angle statistics).
  3) Optional "tube" basis: eigenvectors of the average projector
     P_avg = (1/L) Σ_l Q_l Q_l^T, which highlights directions consistently
     present across scanned layers.

It reuses your existing decode-aligned estimator (A3 + pooled PCA + shareness)
from:
  disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py

--layers 8,9,10,11,12 \

Example:
  CUDA_VISIBLE_DEVICES=0 python cross_layer_shared_workspace_scan.py \
  --model meta-llama/Llama-2-7b-chat-hf --device cuda --model_dtype fp16 \
  --tasks gsm8k,commonsenseqa,strategyqa,openbookqa,qasc,boolq,piqa \
  --layers 21,22,23,24,25,26,27,28,29,30,31,32 \
  --n_subspace 128 --n_eval 128 \
  --calib_decode_max_new_tokens 128 --per_task_max_states 12000 \
  --pca_var 0.95 --max_dim 2048 --tau 0.001 --m_shared all \
  --tube_thresh 0.80 --tube_cap_per_layer 128 \
  --out_dir cross_layer_scan_out_later_layers


Notes:
  - Keep --layers to 4–6 to control runtime.
  - This script is *analysis-only*: it does not run causal ablations; it is
    intended as a clean “relationship between layerwise workspaces” appendix
    experiment.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

# ---- Reuse the parent H3 helper module ----
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from h3_decode_subspace_helpers import (
    load_model_and_tokenizer,
    compute_shared_subspace_decode_aligned,
    orthonormalize_np,
    energy_ratio_stats,
    stable_int_seed,
)

from decodeshare.benchmark_dataloaders import load_selected_tasks


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_layers_arg(layers: str, center: int, n_scan: int) -> List[int]:
    """Parse --layers or derive from --center_layer/--n_scan."""
    if layers and layers.strip():
        out = [int(x.strip()) for x in layers.split(",") if x.strip()]
        if len(out) == 0:
            raise ValueError("--layers parsed to empty list")
        return out
    # derive: symmetric window around center
    n_scan = int(n_scan)
    if n_scan <= 0:
        raise ValueError("--n_scan must be > 0")
    # choose n_scan layers: center-2..center+3 for n_scan=6, etc.
    half = n_scan // 2
    start = int(center) - half
    out = list(range(start, start + n_scan))
    return out


def _principal_angle_stats(Qa: np.ndarray, Qb: np.ndarray, cos_thresh: float = 0.90) -> Dict[str, float]:
    """Return principal-angle style stats between two subspaces."""
    if Qa.size == 0 or Qb.size == 0:
        return {
            "k_a": float(Qa.shape[1] if Qa.ndim == 2 else 0),
            "k_b": float(Qb.shape[1] if Qb.ndim == 2 else 0),
            "min_k": 0.0,
            "mean_cos2": 0.0,
            "max_cos": 0.0,
            "p50_angle_deg": 90.0,
            "mean_angle_deg": 90.0,
            "near_intersection_dim": 0.0,
        }

    # M shape: [k_a, k_b]
    M = Qa.T @ Qb
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    angles = np.arccos(s) * (180.0 / math.pi)

    return {
        "k_a": float(Qa.shape[1]),
        "k_b": float(Qb.shape[1]),
        "min_k": float(min(Qa.shape[1], Qb.shape[1])),
        "mean_cos2": float(np.mean(s**2)) if s.size else 0.0,
        "max_cos": float(np.max(s)) if s.size else 0.0,
        "p50_angle_deg": float(np.percentile(angles, 50)) if angles.size else 90.0,
        "mean_angle_deg": float(np.mean(angles)) if angles.size else 90.0,
        "near_intersection_dim": float(np.sum(s >= float(cos_thresh))),
    }


def _subsample_rows(x: np.ndarray, n_max: int, seed: int) -> np.ndarray:
    if n_max is None or n_max <= 0 or x.shape[0] <= n_max:
        return x
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.shape[0], size=int(n_max), replace=False)
    return x[idx]


def _sample_calib_states(task_acts: Dict[str, Dict[int, np.ndarray]], layer: int, *,
                         max_total: int, seed: int) -> np.ndarray:
    """Sample a balanced set of decode states across tasks for energy diagnostics."""
    tasks = [t for t in task_acts.keys() if layer in task_acts[t]]
    if not tasks:
        return np.zeros((0, 0), dtype=np.float32)
    per_task = max(1, int(max_total) // max(len(tasks), 1))
    chunks = []
    for t in tasks:
        X = task_acts[t][layer]
        ss = stable_int_seed(seed, "energy", layer, t)
        Xs = _subsample_rows(X, per_task, seed=ss)
        chunks.append(Xs)
    return np.concatenate(chunks, axis=0)


def _compute_consensus_tube(
    Q_by_layer: Dict[int, np.ndarray],
    *,
    cap_per_layer: int,
    tube_thresh: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Compute a cross-layer consensus subspace via the average projector.

    P_avg = (1/L) Σ_l Q_l Q_l^T.

    We avoid building the D×D matrix by using A = [Q_1, Q_2, ..., Q_L] (D×K),
    then P_avg = (1/L) A A^T. The eigenvalues of P_avg are λ_i / L, where λ_i
    are eigenvalues of A^T A.
    """
    layers = sorted(Q_by_layer.keys())
    if not layers:
        return np.zeros((0, 0), dtype=np.float32), {
            "n_layers": 0,
            "K_total": 0,
            "tube_dim": 0,
            "tube_thresh": float(tube_thresh),
            "top_eigs": [],
        }

    # concatenate capped bases
    A_list = []
    k_list = []
    for l in layers:
        Q = Q_by_layer[l]
        k = int(min(Q.shape[1], int(cap_per_layer)))
        k_list.append(k)
        if k > 0:
            A_list.append(Q[:, :k])
    if not A_list:
        return np.zeros((Q_by_layer[layers[0]].shape[0], 0), dtype=np.float32), {
            "n_layers": len(layers),
            "K_total": 0,
            "tube_dim": 0,
            "tube_thresh": float(tube_thresh),
            "top_eigs": [],
            "per_layer_caps": {str(l): int(k) for l, k in zip(layers, k_list)},
        }

    A = np.concatenate(A_list, axis=1).astype(np.float32, copy=False)  # [D, K_total]
    K_total = int(A.shape[1])
    L = float(len(layers))

    # Gram matrix (K_total x K_total) is small enough for np.linalg.eigh in our intended regimes.
    G = (A.T @ A).astype(np.float64, copy=False)
    evals, evecs = np.linalg.eigh(G)  # ascending
    evals = evals[::-1]
    evecs = evecs[:, ::-1]

    # consensus eigenvalues of P_avg in [0,1]
    consensus = evals / max(L, 1.0)
    tube_dim = int(np.sum(consensus >= float(tube_thresh)))

    if tube_dim <= 0:
        meta = {
            "n_layers": int(len(layers)),
            "K_total": int(K_total),
            "tube_dim": 0,
            "tube_thresh": float(tube_thresh),
            "top_eigs": [float(x) for x in consensus[: min(20, consensus.shape[0])]],
            "per_layer_caps": {str(l): int(min(Q_by_layer[l].shape[1], int(cap_per_layer))) for l in layers},
        }
        return np.zeros((A.shape[0], 0), dtype=np.float32), meta

    # Build tube basis U from eigenpairs of G:
    # For each eigenpair (λ, v): u = A v / sqrt(λ)
    lam = evals[:tube_dim]
    V = evecs[:, :tube_dim]
    # numerical safety
    lam = np.maximum(lam, 1e-12)
    U = (A @ V) / np.sqrt(lam[None, :])
    U = orthonormalize_np(U)

    meta = {
        "n_layers": int(len(layers)),
        "K_total": int(K_total),
        "tube_dim": int(U.shape[1]),
        "tube_thresh": float(tube_thresh),
        "top_eigs": [float(x) for x in consensus[: min(50, consensus.shape[0])]],
        "per_layer_caps": {str(l): int(min(Q_by_layer[l].shape[1], int(cap_per_layer))) for l in layers},
    }
    return U.astype(np.float32, copy=False), meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--model_dtype", type=str, default="fp16", choices=["fp16", "fp32"])

    ap.add_argument(
        "--tasks",
        type=str,
        default="gsm8k,commonsenseqa,strategyqa,piqa,arc_challenge,openbookqa,qasc,logiqa,boolq",
    )

    # Layer scan control
    ap.add_argument("--layers", type=str, default="", help="Comma-separated layer indices, e.g., 8,9,10,11,12")
    ap.add_argument("--center_layer", type=int, default=10, help="Used if --layers not provided")
    ap.add_argument("--n_scan", type=int, default=6, help="Used if --layers not provided")

    # Data sizes
    ap.add_argument("--n_subspace", type=int, default=64)
    ap.add_argument("--n_eval", type=int, default=64)

    # Decode collection
    ap.add_argument("--calib_decode_max_new_tokens", type=int, default=128)
    ap.add_argument("--per_task_max_states", type=int, default=12000)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_prompt_len", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=0)

    # Sharedness estimator
    ap.add_argument("--pca_var", type=float, default=0.95)
    ap.add_argument("--min_dim", type=int, default=16)
    ap.add_argument("--max_dim", type=int, default=512)
    ap.add_argument("--tau", type=float, default=1e-3)
    ap.add_argument("--m_shared", type=str, default="all")

    # Prompt formatting
    ap.add_argument("--template_randomization", type=int, default=1)
    ap.add_argument("--shuffle_choices", type=int, default=1)
    ap.add_argument("--add_answer_prefix", type=int, default=1)
    ap.add_argument("--answer_prefix", type=str, default="\nFinal answer:")

    # Diagnostics
    ap.add_argument("--energy_sample_total", type=int, default=6000)
    ap.add_argument("--cos_thresh", type=float, default=0.90)

    # Tube extraction
    ap.add_argument("--tube_thresh", type=float, default=0.80, help="Eigenvalue threshold on P_avg for tube directions")
    ap.add_argument("--tube_cap_per_layer", type=int, default=128, help="Cap columns per layer when building tube")

    # Output
    ap.add_argument("--out_dir", type=str, default="cross_layer_scan_out")
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()
    _set_global_seed(args.seed)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    layers = _parse_layers_arg(args.layers, args.center_layer, args.n_scan)
    layers = sorted(list(dict.fromkeys(layers)))  # unique + sorted

    os.makedirs(args.out_dir, exist_ok=True)

    print("\n" + "=" * 100)
    print(f"[Config] model={args.model} dtype={args.model_dtype} device={args.device}")
    print(f"[Config] tasks={tasks}")
    print(f"[Config] layers={layers} (n={len(layers)})")
    print("=" * 100 + "\n")

    # 1) Load model + tokenizer
    model, tok = load_model_and_tokenizer(args.model, device=args.device, model_dtype=args.model_dtype)

    # 2) Load prompts
    sub_by, _eval_by, _meta_by = load_selected_tasks(
        tasks=tasks,
        n_subspace=args.n_subspace,
        n_eval=args.n_eval,
        seed=args.seed,
        template_seed=args.seed + 999,
        template_randomization=bool(args.template_randomization),
        shuffle_choices=bool(args.shuffle_choices),
        add_answer_prefix=bool(args.add_answer_prefix),
        answer_prefix=args.answer_prefix,
    )

    prompts_by_task = {k: [ex.prompt for ex in sub_by[k]] for k in tasks if k in sub_by}

    # 3) For each layer, estimate Q_shared
    Q_by_layer: Dict[int, np.ndarray] = {}
    layer_stats: Dict[str, Any] = {}

    for layer in layers:
        print("\n" + "#" * 100)
        print(f"[Layer {layer}] Estimating decode-time shared workspace ...")
        print("#" * 100)

        joint_subspace, shared_indices, extra, task_acts = compute_shared_subspace_decode_aligned(
            model=model,
            tokenizer=tok,
            prompts_by_task=prompts_by_task,
            layer_indices=[int(layer)],
            calib_decoding="greedy",
            calib_batch_size=args.batch_size,
            calib_max_new_tokens=args.calib_decode_max_new_tokens,
            per_task_max_states=args.per_task_max_states,
            max_prompt_len=args.max_prompt_len,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            global_seed=args.seed,
            variance_threshold=args.pca_var,
            min_dim=args.min_dim,
            max_dim=args.max_dim,
            tau=args.tau,
            m_shared=args.m_shared,
        )

        cross_dim = int(extra.get("cross_dim", 0))
        k_shared = int(len(shared_indices))
        if k_shared <= 0:
            print(f"[Layer {layer}] WARNING: shared basis is empty. Skipping overlap contributions.")
            Q = np.zeros((joint_subspace.shape[0], 0), dtype=np.float32)
        else:
            Q = orthonormalize_np(joint_subspace[:, shared_indices])

        # energy ratio diagnostic on a small balanced sample
        calib_states = _sample_calib_states(
            task_acts, layer,
            max_total=int(args.energy_sample_total),
            seed=args.seed,
        )
        er = energy_ratio_stats(calib_states, Q) if calib_states.size else {"mean": float("nan"), "p50": float("nan"), "p95": float("nan")}

        # Persist
        Q_by_layer[int(layer)] = Q
        layer_stats[str(layer)] = {
            "layer": int(layer),
            "cross_dim": int(cross_dim),
            "k_shared": int(k_shared),
            "tau": float(args.tau),
            "m_shared": args.m_shared,
            "energy_ratio": er,
            "tasks_used": extra.get("tasks_used", []),
        }

        # save basis
        np.save(os.path.join(args.out_dir, f"Q_shared_layer{layer}_k{k_shared}.npy"), Q)
        with open(os.path.join(args.out_dir, f"layer{layer}_meta.json"), "w", encoding="utf-8") as f:
            json.dump(layer_stats[str(layer)], f, indent=2)

        print(
            f"[Layer {layer}] cross_dim={cross_dim} k_shared={k_shared} "
            f"energy_ratio(mean/p50/p95)={er.get('mean', float('nan')):.4f}/{er.get('p50', float('nan')):.4f}/{er.get('p95', float('nan')):.4f}"
        )

    # 4) Pairwise overlap
    overlap: Dict[str, Dict[str, Any]] = {}
    for i, li in enumerate(layers):
        overlap[str(li)] = {}
        for lj in layers:
            st = _principal_angle_stats(Q_by_layer[li], Q_by_layer[lj], cos_thresh=float(args.cos_thresh))
            overlap[str(li)][str(lj)] = st

    # 5) Consensus tube (optional but useful)
    Q_tube, tube_meta = _compute_consensus_tube(
        Q_by_layer,
        cap_per_layer=int(args.tube_cap_per_layer),
        tube_thresh=float(args.tube_thresh),
    )
    if Q_tube.size:
        np.save(os.path.join(args.out_dir, f"Q_tube_thresh{args.tube_thresh:.2f}_k{Q_tube.shape[1]}.npy"), Q_tube)

    tube_overlap: Dict[str, Any] = {}
    if Q_tube.size:
        for l in layers:
            tube_overlap[str(l)] = _principal_angle_stats(Q_by_layer[l], Q_tube, cos_thresh=float(args.cos_thresh))

    # 6) Print a compact summary table
    print("\n" + "=" * 100)
    print("[Summary] Per-layer shared workspace stats")
    print("=" * 100)
    print(f"{'layer':>6}  {'cross':>5}  {'kS':>4}  {'Emean':>7}  {'Ep50':>7}  {'Ep95':>7}")
    for l in layers:
        st = layer_stats[str(l)]
        er = st["energy_ratio"]
        print(f"{l:>6d}  {st['cross_dim']:>5d}  {st['k_shared']:>4d}  {er['mean']:>7.4f}  {er['p50']:>7.4f}  {er['p95']:>7.4f}")

    print("\n" + "=" * 100)
    print("[Summary] Pairwise overlap: mean_cos2 (higher = more aligned)")
    print("=" * 100)
    header = "layers "+" ".join([f"{l:>8d}" for l in layers])
    print(header)
    for li in layers:
        row = [f"{li:>6d}"]
        for lj in layers:
            row.append(f"{overlap[str(li)][str(lj)]['mean_cos2']:.3f}".rjust(8))
        print(" ".join(row))

    print("\n" + "=" * 100)
    print("[Summary] Pairwise overlap: near-intersection dim @ cos>=%.2f" % float(args.cos_thresh))
    print("=" * 100)
    print(header)
    for li in layers:
        row = [f"{li:>6d}"]
        for lj in layers:
            row.append(f"{overlap[str(li)][str(lj)]['near_intersection_dim']:.0f}".rjust(8))
        print(" ".join(row))

    if Q_tube.size:
        print("\n" + "=" * 100)
        print("[Tube] Consensus tube extracted from average projector")
        print("=" * 100)
        print(f"tube_dim={Q_tube.shape[1]}  tube_thresh={args.tube_thresh}  cap_per_layer={args.tube_cap_per_layer}")
        print(f"top consensus eigvals (P_avg): {[round(x, 4) for x in tube_meta.get('top_eigs', [])[:10]]}")

    # 7) Save JSON
    out = {
        "model": args.model,
        "dtype": args.model_dtype,
        "device": args.device,
        "tasks": tasks,
        "layers": layers,
        "estimator": {
            "calib_decode_max_new_tokens": int(args.calib_decode_max_new_tokens),
            "per_task_max_states": int(args.per_task_max_states),
            "pca_var": float(args.pca_var),
            "min_dim": int(args.min_dim),
            "max_dim": int(args.max_dim),
            "tau": float(args.tau),
            "m_shared": args.m_shared,
        },
        "layer_stats": layer_stats,
        "pairwise_overlap": overlap,
        "tube": {
            "meta": tube_meta,
            "tube_overlap": tube_overlap,
        },
    }
    out_path = os.path.join(args.out_dir, "cross_layer_shared_workspace_scan.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n[Done] Wrote: {out_path}")


if __name__ == "__main__":
    main()
