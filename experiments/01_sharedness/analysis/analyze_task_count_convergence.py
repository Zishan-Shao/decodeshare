"""Analyze shared-subspace convergence as the number of tasks increases."""

import os
import json
import csv
import argparse
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from decodeshare.sharedness import compute_cross_task_subspace

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

def _orthonormalize(Q: np.ndarray) -> np.ndarray:
    Q = Q.astype(np.float64, copy=False)
    Q, _ = np.linalg.qr(Q)
    return Q.astype(np.float32, copy=False)

def compute_basis(X_by_task: Dict[str, np.ndarray], layer: int, tasks: List[str],
                  pca_var: float, min_dim: int, max_dim: int) -> Tuple[np.ndarray, int]:
    task_acts = {t: {layer: X_by_task[t]} for t in tasks}
    joint_subspace, cross_dim, _, _ = compute_cross_task_subspace(
        task_acts,
        variance_threshold=float(pca_var),
        min_dim=int(min_dim),
        max_dim=int(max_dim),
        return_full_pca=True,
    )
    if joint_subspace is None or int(cross_dim) <= 0:
        return None, 0
    Q = joint_subspace.astype(np.float32, copy=False)
    Q = _orthonormalize(Q)
    return Q, int(cross_dim)


def subspace_overlap(Qa: np.ndarray, Qb: np.ndarray) -> float:
    if Qa is None or Qb is None:
        return 0.0
    M = Qa.T @ Qb
    s = np.linalg.svd(M, compute_uv=False)
    k = min(Qa.shape[1], Qb.shape[1])
    return float((s[:k] ** 2).sum() / max(k, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts_dir", type=str, required=True)
    ap.add_argument("--pca_var", type=float, default=0.95)
    ap.add_argument("--min_dim", type=int, default=1)
    ap.add_argument("--max_dim", type=int, default=4096)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--out_png", type=str, required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_png)), exist_ok=True)

    X_by_task, meta = load_acts(args.acts_dir)
    layer = int(meta["config"]["layer"])
    tasks_all = list(X_by_task.keys())
    T = len(tasks_all)

    print(f"[Acts] dir={args.acts_dir}")
    print(f"[Cfg] layer={layer} pca_var={args.pca_var} repeats={args.repeats} seed={args.seed}")
    print(f"[Tasks] T={T} tasks={tasks_all}")


    Q_full, k_full = compute_basis(X_by_task, layer, tasks_all, args.pca_var, args.min_dim, args.max_dim)
    if Q_full is None:
        raise RuntimeError("Failed to compute full-task basis.")
    print(f"[Full] k_full={k_full}")

    rng = np.random.default_rng(int(args.seed))

    rows = []
    for n in range(2, T + 1):
        overlaps = []
        ks = []
        for r in range(int(args.repeats)):
            subset = rng.choice(tasks_all, size=n, replace=False).tolist()
            Qn, kn = compute_basis(X_by_task, layer, subset, args.pca_var, args.min_dim, args.max_dim)
            ov = subspace_overlap(Qn, Q_full)
            overlaps.append(ov)
            ks.append(kn)

        row = {
            "n_tasks": n,
            "overlap_mean": float(np.mean(overlaps)),
            "overlap_std": float(np.std(overlaps, ddof=0)),
            "cross_dim_mean": float(np.mean(ks)),
            "cross_dim_std": float(np.std(ks, ddof=0)),
        }
        rows.append(row)
        print(f"[n={n}] overlap={row['overlap_mean']:.4f}+/-{row['overlap_std']:.4f} "
              f"k={row['cross_dim_mean']:.1f}+/-{row['cross_dim_std']:.1f}")


    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[Save] {args.out_csv}")


    xs = [r["n_tasks"] for r in rows]
    ys = [r["overlap_mean"] for r in rows]
    yerr = [r["overlap_std"] for r in rows]

    plt.figure()
    plt.errorbar(xs, ys, yerr=yerr, marker="o")
    plt.xlabel("#tasks used to estimate subspace")
    plt.ylabel("overlap to full-task subspace")
    plt.title("Subspace convergence vs #tasks")
    plt.ylim(0.0, 1.05)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=200)
    print(f"[Save] {args.out_png}")

if __name__ == "__main__":
    main()
