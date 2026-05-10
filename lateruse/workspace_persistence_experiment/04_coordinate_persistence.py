
# -*- coding: utf-8 -*-
"""
04_coordinate_persistence.py

Phase 4 (Q2): 功能持久性 —— shared coords 是否可被跨层线性预测？

输入：
  --coords_train / --coords_test：由 03_collect_shared_coords.py 生成

输出：
  - results_coord_persistence.csv
  - R² vs layer distance 的图（figs/q2_r2_vs_distance.png）
  - 同时给一个打乱 baseline（shuffle Z_tgt 行）做对照

运行示例：
  python 04_coordinate_persistence.py --coords_train results_run1/coords_train.pt --coords_test results_run1/coords_test.pt
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from subspace_utils import fit_linear_map_ridge, r2_score


def build_pairs(layers: List[int], mode: str = "adjacent", max_hop: int = 1) -> List[Tuple[int, int]]:
    """
    mode:
      - "adjacent": 按 layers 列表的相邻项配对（默认），可用 max_hop 扩展到 i->i+h
      - "all": 所有 i<j
    """
    mode = mode.lower().strip()
    pairs = []
    if mode == "adjacent":
        max_hop = int(max_hop)
        max_hop = max(1, max_hop)
        for i in range(len(layers)):
            for h in range(1, max_hop + 1):
                j = i + h
                if j < len(layers):
                    pairs.append((layers[i], layers[j]))
    elif mode == "all":
        for i in range(len(layers)):
            for j in range(i + 1, len(layers)):
                pairs.append((layers[i], layers[j]))
    else:
        raise ValueError("mode must be adjacent or all")
    return pairs


def stack_Z(trajectories: List[Dict[int, np.ndarray]], ell: int, m: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    将多个 prompt 的 (Z_ell, Z_m) 逐步拼成一个大矩阵：
      Z_ell: [N, k_ell]
      Z_m:   [N, k_m]
    """
    Z1_list = []
    Z2_list = []
    for tr in trajectories:
        Z1 = tr[ell]
        Z2 = tr[m]
        T = min(Z1.shape[0], Z2.shape[0])
        if T == 0:
            continue
        Z1_list.append(Z1[:T])
        Z2_list.append(Z2[:T])
    if len(Z1_list) == 0:
        raise RuntimeError(f"No samples for pair ({ell}->{m}).")
    Z1_all = np.concatenate(Z1_list, axis=0)
    Z2_all = np.concatenate(Z2_list, axis=0)
    return Z1_all, Z2_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coords_train", type=str, required=True)
    ap.add_argument("--coords_test", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default=None,
                    help="输出目录（默认与 coords_train 同目录）")
    ap.add_argument("--pair_mode", type=str, default="adjacent", choices=["adjacent", "all"])
    ap.add_argument("--max_hop", type=int, default=1, help="pair_mode=adjacent 时有效：i->i+h 最大 h")
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--shuffle_baseline", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    train_payload = torch.load(args.coords_train, map_location="cpu")
    test_payload = torch.load(args.coords_test, map_location="cpu")
    layers: List[int] = list(train_payload["layers"])
    k_shared_by_layer = train_payload["k_shared_by_layer"]

    traj_train: List[Dict[int, np.ndarray]] = train_payload["trajectories"]
    traj_test: List[Dict[int, np.ndarray]] = test_payload["trajectories"]

    out_dir = args.out_dir or os.path.dirname(args.coords_train)
    fig_dir = os.path.join(out_dir, "figs")
    os.makedirs(fig_dir, exist_ok=True)

    pairs = build_pairs(layers, mode=args.pair_mode, max_hop=args.max_hop)
    print(f"[info] pairs={pairs}")

    rng = np.random.default_rng(int(args.seed))

    rows = []
    for (ell, m) in pairs:
        Z1_tr, Z2_tr = stack_Z(traj_train, ell, m)
        Z1_te, Z2_te = stack_Z(traj_test, ell, m)

        A = fit_linear_map_ridge(Z1_tr, Z2_tr, ridge=args.ridge)
        Z2_hat = Z1_te @ A.T
        r2 = r2_score(Z2_te, Z2_hat)

        r2_shuf = None
        if args.shuffle_baseline:
            perm = rng.permutation(Z2_te.shape[0])
            r2_shuf = r2_score(Z2_te[perm], Z2_hat)

        rows.append({
            "src_layer": ell,
            "tgt_layer": m,
            "distance": abs(m - ell),
            "k_src": int(k_shared_by_layer[ell]),
            "k_tgt": int(k_shared_by_layer[m]),
            "N_train": int(Z1_tr.shape[0]),
            "N_test": int(Z1_te.shape[0]),
            "ridge": float(args.ridge),
            "r2": float(r2),
            "r2_shuffled": None if r2_shuf is None else float(r2_shuf),
        })

    df = pd.DataFrame(rows).sort_values(["distance", "src_layer", "tgt_layer"])
    out_csv = os.path.join(out_dir, "results_coord_persistence.csv")
    df.to_csv(out_csv, index=False)
    print(f"[done] saved {out_csv}")
    print(df)

    # Plot R2 vs distance (mean ± se)
    uniq = np.sort(df["distance"].unique())
    mean_r2 = []
    se_r2 = []
    mean_shuf = []
    se_shuf = []

    for d in uniq:
        vals = df[df["distance"] == d]["r2"].to_numpy()
        mean_r2.append(vals.mean())
        se_r2.append(vals.std(ddof=1) / max(len(vals) ** 0.5, 1.0))

        if args.shuffle_baseline and df["r2_shuffled"].notna().any():
            v2 = df[df["distance"] == d]["r2_shuffled"].dropna().to_numpy()
            mean_shuf.append(v2.mean())
            se_shuf.append(v2.std(ddof=1) / max(len(v2) ** 0.5, 1.0))
        else:
            mean_shuf.append(np.nan)
            se_shuf.append(np.nan)

    plt.figure(figsize=(7, 4))
    plt.errorbar(uniq, mean_r2, yerr=se_r2, marker="o", label="R² (linear map)")
    if args.shuffle_baseline:
        plt.errorbar(uniq, mean_shuf, yerr=se_shuf, marker="o", label="R² baseline (shuffled)")
    plt.xlabel("layer distance |Δℓ|")
    plt.ylabel("R²")
    plt.title("Functional persistence: linear predictability of shared coords")
    plt.legend()
    plt.tight_layout()
    out_fig = os.path.join(fig_dir, "q2_r2_vs_distance.png")
    plt.savefig(out_fig, dpi=200)
    plt.close()
    print(f"[saved] {out_fig}")


if __name__ == "__main__":
    main()
