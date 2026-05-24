"""Compare within-category sharedness against mixed-category task samples."""

import os
import json
import csv
import argparse
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from decodeshare.sharedness import (
    compute_cross_task_subspace,
    compute_relvar_in_basis,
    compute_shared_indices_from_relvar,
)

TASK2CAT = {
    "gsm8k": "math",
    "aqua": "math",
    "commonsenseqa": "commonsense",
    "piqa": "commonsense",
    "strategyqa": "reasoning",
    "boolq": "reasoning",
    "arc_challenge": "science",
    "openbookqa": "science",
    "qasc": "science",
}

GROUPS = {
    "math": ["gsm8k", "aqua"],
    "commonsense": ["commonsenseqa", "piqa"],
    "reasoning": ["strategyqa", "boolq"],
    "science": ["arc_challenge", "openbookqa", "qasc"],
}

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

def sharedness_metrics(
    X_by_task: Dict[str, np.ndarray],
    layer: int,
    tasks: List[str],
    pca_var: float,
    tau: float,
    min_dim: int,
    max_dim: int,
) -> Dict:
    task_acts = {t: {layer: X_by_task[t]} for t in tasks}

    joint_subspace, cross_dim, _, _ = compute_cross_task_subspace(
        task_acts,
        variance_threshold=float(pca_var),
        min_dim=int(min_dim),
        max_dim=int(max_dim),
        return_full_pca=True,
    )
    if joint_subspace is None or int(cross_dim) <= 0:
        return {
            "cross_dim": 0,
            "shared_count": 0,
            "shared_ratio": 0.0,
        }

    Q = joint_subspace.astype(np.float32, copy=False)
    relvar_by_task = {t: compute_relvar_in_basis(X_by_task[t], Q) for t in tasks}

    m_shared = len(tasks)
    shared_idx = compute_shared_indices_from_relvar(relvar_by_task, tau=float(tau), m_shared=int(m_shared))
    shared_count = int(len(shared_idx))
    k = int(cross_dim)
    return {
        "cross_dim": k,
        "shared_count": shared_count,
        "shared_ratio": (float(shared_count) / float(k)) if k > 0 else 0.0,
    }

def sample_mixed_tasks(all_tasks: List[str], k: int, rng: np.random.Generator) -> List[str]:
    """Internal helper for this experiment."""
    assert k >= 2
    for _ in range(10_000):
        pick = rng.choice(all_tasks, size=k, replace=False).tolist()
        cats = {TASK2CAT.get(t, "unknown") for t in pick}
        if len(cats) >= 2:
            return pick
    return rng.choice(all_tasks, size=k, replace=False).tolist()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts_dir", type=str, required=True)
    ap.add_argument("--pca_var", type=float, default=0.95)
    ap.add_argument("--tau", type=float, default=0.001)
    ap.add_argument("--min_dim", type=int, default=1)
    ap.add_argument("--max_dim", type=int, default=4096)
    ap.add_argument("--n_mixed", type=int, default=50)
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--out_png", type=str, required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_png)), exist_ok=True)

    X_by_task, meta = load_acts(args.acts_dir)
    layer = int(meta["config"]["layer"])
    all_tasks = list(X_by_task.keys())

    print(f"[Acts] dir={args.acts_dir}")
    print(f"[Acts] tasks={all_tasks}")
    print(f"[Cfg] layer={layer} pca_var={args.pca_var} tau={args.tau} n_mixed={args.n_mixed}")

    rows = []

    for gname, g_tasks0 in GROUPS.items():
        g_tasks = [t for t in g_tasks0 if t in all_tasks]
        if len(g_tasks) < 2:
            print(f"[Skip] group={gname} has <2 tasks available: {g_tasks}")
            continue

        within = sharedness_metrics(
            X_by_task, layer, g_tasks,
            pca_var=args.pca_var, tau=args.tau, min_dim=args.min_dim, max_dim=args.max_dim
        )

        rng = np.random.default_rng(int(args.seed) + hash(gname) % 10_000)
        mixed_metrics = []
        for _ in range(int(args.n_mixed)):
            mtasks = sample_mixed_tasks(all_tasks, k=len(g_tasks), rng=rng)
            mm = sharedness_metrics(
                X_by_task, layer, mtasks,
                pca_var=args.pca_var, tau=args.tau, min_dim=args.min_dim, max_dim=args.max_dim
            )
            mixed_metrics.append(mm)

        mixed_ratio = np.array([m["shared_ratio"] for m in mixed_metrics], dtype=np.float64)
        mixed_count = np.array([m["shared_count"] for m in mixed_metrics], dtype=np.float64)

        row = {
            "group": gname,
            "k_tasks": len(g_tasks),
            "within_tasks": ",".join(g_tasks),
            "within_cross_dim": within["cross_dim"],
            "within_shared_count": within["shared_count"],
            "within_shared_ratio": within["shared_ratio"],
            "mixed_shared_ratio_mean": float(mixed_ratio.mean()) if len(mixed_ratio) else 0.0,
            "mixed_shared_ratio_std": float(mixed_ratio.std(ddof=0)) if len(mixed_ratio) else 0.0,
            "mixed_shared_count_mean": float(mixed_count.mean()) if len(mixed_count) else 0.0,
            "mixed_shared_count_std": float(mixed_count.std(ddof=0)) if len(mixed_count) else 0.0,
        }
        rows.append(row)

        print(f"[Group={gname}] within_ratio={row['within_shared_ratio']:.4f} "
              f"mixed_ratio_mean={row['mixed_shared_ratio_mean']:.4f}+/-{row['mixed_shared_ratio_std']:.4f}")


    fieldnames = list(rows[0].keys()) if rows else [
        "group","k_tasks","within_tasks","within_cross_dim","within_shared_count","within_shared_ratio",
        "mixed_shared_ratio_mean","mixed_shared_ratio_std","mixed_shared_count_mean","mixed_shared_count_std"
    ]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[Save] {args.out_csv}")


    if rows:
        labels = [r["group"] for r in rows]
        x = np.arange(len(labels))
        within = [r["within_shared_ratio"] for r in rows]
        mmean = [r["mixed_shared_ratio_mean"] for r in rows]
        mstd = [r["mixed_shared_ratio_std"] for r in rows]

        width = 0.35
        plt.figure()
        plt.bar(x - width/2, within, width, label="within-category")
        plt.bar(x + width/2, mmean, width, yerr=mstd, label="mixed-category (mean+/-std)")

        plt.xticks(x, labels, rotation=0)
        plt.ylabel("shared_ratio = shared_count / cross_dim")
        plt.title("Within-category vs Mixed-category sharedness")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.out_png, dpi=200)
        print(f"[Save] {args.out_png}")

if __name__ == "__main__":
    main()
