
# -*- coding: utf-8 -*-
"""
02_structural_persistence.py

Phase 2 (Q1): 结构持久性 —— 共享子空间跨层是否重叠？

输入：
  --shared_pt 由 01_estimate_shared_subspaces.py 生成的 shared_subspaces.pt

输出：
  - layer×layer 相似度 heatmap（mean sigma^2）
  - 相似度 vs layer distance 的 1D plot
  - 可选：与 nonshared control 的对照曲线

运行示例：
  python 02_structural_persistence.py --shared_pt results_run1/shared_subspaces.pt --out_dir results_run1
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from subspace_utils import subspace_similarity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shared_pt", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default=None,
                    help="输出目录（默认与 shared_pt 同目录）")
    ap.add_argument("--title", type=str, default="Structural persistence: subspace similarity")
    ap.add_argument("--save_npy", type=int, default=1)
    args = ap.parse_args()

    payload = torch.load(args.shared_pt, map_location="cpu")
    layers: List[int] = list(payload["layers"])
    Qs: Dict[int, torch.Tensor] = payload["Q_shared_by_layer"]
    Qctrl: Dict[int, torch.Tensor] = payload["Q_control_pool_by_layer"]

    out_dir = args.out_dir or os.path.dirname(args.shared_pt)
    fig_dir = os.path.join(out_dir, "figs")
    os.makedirs(fig_dir, exist_ok=True)

    L = len(layers)
    S = np.zeros((L, L), dtype=np.float64)
    S_ctrl = np.zeros((L, L), dtype=np.float64)

    # compute pairwise similarities
    for i, li in enumerate(layers):
        Qa = Qs[li].numpy()
        for j, lj in enumerate(layers):
            Qb = Qs[lj].numpy()
            sim, _ = subspace_similarity(Qa, Qb, r=None)
            S[i, j] = sim

            # control: compare Qa with nonshared pool at layer lj, matched rank r=min(k_li,k_lj)
            r = min(Qa.shape[1], Qb.shape[1])
            Qb_ctrl_pool = Qctrl[lj].numpy()
            if Qb_ctrl_pool.shape[1] < r:
                raise RuntimeError(
                    f"Control pool too small at layer {lj}: need {r}, have {Qb_ctrl_pool.shape[1]}"
                )
            Qb_ctrl = Qb_ctrl_pool[:, :r]
            sim_ctrl, _ = subspace_similarity(Qa, Qb_ctrl, r=r)
            S_ctrl[i, j] = sim_ctrl

    if args.save_npy:
        np.save(os.path.join(out_dir, "similarity_matrix.npy"), S)
        np.save(os.path.join(out_dir, "similarity_matrix_control.npy"), S_ctrl)

    # heatmap
    plt.figure(figsize=(7, 6))
    plt.imshow(S, interpolation="nearest")
    plt.colorbar(label="mean(sigma^2)")
    plt.xticks(range(L), [str(x) for x in layers], rotation=45, ha="right")
    plt.yticks(range(L), [str(x) for x in layers])
    plt.xlabel("layer")
    plt.ylabel("layer")
    plt.title(args.title)
    plt.tight_layout()
    out_path = os.path.join(fig_dir, "q1_heatmap_similarity.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[saved] {out_path}")

    # 1D similarity vs distance
    # distance 用“层编号差值”
    dists = []
    sims = []
    sims_ctrl = []
    for i, li in enumerate(layers):
        for j, lj in enumerate(layers):
            if j <= i:
                continue
            dist = abs(lj - li)
            dists.append(dist)
            sims.append(S[i, j])
            sims_ctrl.append(S_ctrl[i, j])

    dists = np.asarray(dists)
    sims = np.asarray(sims)
    sims_ctrl = np.asarray(sims_ctrl)

    uniq = np.unique(dists)
    mean_sim = []
    se_sim = []
    mean_ctrl = []
    se_ctrl = []
    for d in uniq:
        m = sims[dists == d]
        c = sims_ctrl[dists == d]
        mean_sim.append(m.mean())
        se_sim.append(m.std(ddof=1) / max(len(m) ** 0.5, 1.0))
        mean_ctrl.append(c.mean())
        se_ctrl.append(c.std(ddof=1) / max(len(c) ** 0.5, 1.0))

    plt.figure(figsize=(7, 4))
    plt.errorbar(uniq, mean_sim, yerr=se_sim, marker="o", label="shared vs shared")
    plt.errorbar(uniq, mean_ctrl, yerr=se_ctrl, marker="o", label="shared vs nonshared(control)")
    plt.xlabel("layer distance |Δℓ|")
    plt.ylabel("mean(sigma^2)")
    plt.title("Structural persistence vs layer distance")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(fig_dir, "q1_similarity_vs_distance.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[saved] {out_path}")

    # also print quick summary
    print("[summary] similarity matrix diag mean =", float(np.mean(np.diag(S))))
    print("[summary] offdiag mean =", float(np.mean(S[np.triu_indices(L, k=1)])))
    print("[summary] control offdiag mean =", float(np.mean(S_ctrl[np.triu_indices(L, k=1)])))


if __name__ == "__main__":
    main()
