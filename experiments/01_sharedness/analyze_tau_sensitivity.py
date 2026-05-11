# -*- coding: utf-8 -*-
"""
analyze_tau_sensitivity.py

对 Llama 做 sweep：
  - energy: pca_var (PCA 解释方差阈值)
  - tau: relvar threshold

输出：
  - CSV：每个 (pca_var, tau) 的 cross_dim / shared_count / shared_ratio
  - PNG：shared_ratio 的 heatmap（y=pca_var, x=tau）

示例：
  python analyze_tau_sensitivity.py \
    --acts_dir results/acts/<LLAMA_TAG>/layer10_... \
    --pca_vars 0.8,0.9,0.95,0.97,0.99 \
    --taus 1e-4,2e-4,5e-4,1e-3,2e-3,5e-3,1e-2 \
    --min_dim 1 --max_dim 4096 --seed 123 \
    --out_csv results/exp3/llama_sens.csv \
    --out_png results/exp3/llama_sens.png
"""

import os
import json
import csv
import argparse
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from sharedness_base import (
    compute_cross_task_subspace,
    compute_relvar_in_basis,
    compute_shared_indices_from_relvar,
)

def _parse_float_list(s: str) -> List[float]:
    items = []
    for x in str(s).split(","):
        xx = x.strip()
        if xx:
            items.append(float(xx))
    return items

def load_acts(acts_dir: str) -> Tuple[Dict[str, np.ndarray], Dict]:
    meta_path = os.path.join(acts_dir, "meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    X_by_task: Dict[str, np.ndarray] = {}
    for t in meta["tasks"]:
        npy = os.path.join(acts_dir, meta["files"][t])
        X = np.load(npy)
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        X = X - X.mean(axis=0, keepdims=True)
        X_by_task[t] = X
    return X_by_task, meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts_dir", type=str, required=True)
    ap.add_argument("--pca_vars", type=str, required=True, help="逗号分隔，如 0.8,0.9,0.95")
    ap.add_argument("--taus", type=str, required=True, help="逗号分隔，如 1e-4,5e-4,1e-3")
    ap.add_argument("--min_dim", type=int, default=1)
    ap.add_argument("--max_dim", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--out_png", type=str, required=True)
    args = ap.parse_args()

    pca_vars = _parse_float_list(args.pca_vars)
    taus = _parse_float_list(args.taus)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_png)), exist_ok=True)

    X_by_task, meta = load_acts(args.acts_dir)
    layer = int(meta["config"]["layer"])
    tasks = list(X_by_task.keys())
    m_shared = len(tasks)

    print(f"[Acts] dir={args.acts_dir}")
    print(f"[Tasks] {tasks}")
    print(f"[Cfg] layer={layer} min_dim={args.min_dim} max_dim={args.max_dim}")

    rows = []

    # 为了效率：每个 pca_var 只算一次 PCA + relvar；tau sweep 只做阈值计数
    for pv in pca_vars:
        task_acts = {t: {layer: X_by_task[t]} for t in tasks}
        joint_subspace, cross_dim, _, _ = compute_cross_task_subspace(
            task_acts,
            variance_threshold=float(pv),
            min_dim=int(args.min_dim),
            max_dim=int(args.max_dim),
            return_full_pca=True,
        )
        if joint_subspace is None or int(cross_dim) <= 0:
            print(f"[Warn] pca_var={pv} failed")
            for tau in taus:
                rows.append({
                    "pca_var": pv, "tau": tau,
                    "cross_dim": 0,
                    "shared_count": 0,
                    "shared_ratio": 0.0,
                })
            continue

        Q = joint_subspace.astype(np.float32, copy=False)
        k = int(cross_dim)
        relvar_by_task = {t: compute_relvar_in_basis(X_by_task[t], Q) for t in tasks}

        for tau in taus:
            shared_idx = compute_shared_indices_from_relvar(relvar_by_task, tau=float(tau), m_shared=int(m_shared))
            sc = int(len(shared_idx))
            rows.append({
                "pca_var": float(pv),
                "tau": float(tau),
                "cross_dim": int(k),
                "shared_count": int(sc),
                "shared_ratio": (float(sc) / float(k)) if k > 0 else 0.0,
            })
        print(f"[pca_var={pv}] cross_dim={k} done")

    # save CSV
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[Save] {args.out_csv}")

    # heatmap: rows=pca_var, cols=tau
    pca_vars_sorted = sorted(set([r["pca_var"] for r in rows]))
    taus_sorted = sorted(set([r["tau"] for r in rows]))

    grid = np.zeros((len(pca_vars_sorted), len(taus_sorted)), dtype=np.float64)
    lookup = {(r["pca_var"], r["tau"]): r["shared_ratio"] for r in rows}
    for i, pv in enumerate(pca_vars_sorted):
        for j, tau in enumerate(taus_sorted):
            grid[i, j] = float(lookup.get((pv, tau), 0.0))

    plt.figure()
    plt.imshow(grid, aspect="auto", origin="lower")
    plt.colorbar(label="shared_ratio")
    plt.xticks(np.arange(len(taus_sorted)), [str(t) for t in taus_sorted], rotation=45, ha="right")
    plt.yticks(np.arange(len(pca_vars_sorted)), [str(p) for p in pca_vars_sorted])
    plt.xlabel("tau")
    plt.ylabel("pca_var (energy)")
    plt.title("Sensitivity sweep: shared_ratio")
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=200)
    print(f"[Save] {args.out_png}")

if __name__ == "__main__":
    main()
