# -*- coding: utf-8 -*-
"""06_shared_residual_flow_analysis.py

新增分析（Q3）：跨层 shared 与 residual(非共享/控制池) 流之间的相关性。

你在 01 里已经为每层估计了：
  - Q_shared_by_layer[ell] : shared basis
  - Q_control_pool_by_layer[ell] : matched nonshared/control basis（可视作 residual 子空间的一组低维探针）

本脚本做两件事：

(A) 结构层面（subspace-level）
    计算跨层 shared(ℓ) vs residual(m) 的 principal-angle 相似度 heatmap。

(B) 功能层面（coordinate-level）
    在逐步 decode 过程中收集：
      shared_state(ℓ), resid_state(ℓ), shared_update(ℓ), resid_update(ℓ)
    并在 held-out 上报告线性可预测性（R²）：
      shared_state(ℓ)  -> resid_state(m)
      shared_state(ℓ)  -> resid_update(m)
    （可选再加反向：resid -> shared）

输出 deliverables
----------------
- figs/q3_struct_shared_vs_resid_heatmap.png
- figs/q3_r2_shared_to_resid_state_heatmap.png
- figs/q3_r2_shared_to_resid_update_heatmap.png
- figs/q3_r2_vs_distance.png
- shared_residual_train.pt / shared_residual_test.pt  （供 07 因果脚本复用）
- shared_residual_r2_results.csv

运行示例
--------
python 06_shared_residual_flow_analysis.py \
  --shared_pt results_run1/shared_subspaces.pt \
  --model YOUR_MODEL \
  --tasks aqua,commonsenseqa,strategyqa,arc_challenge,openbookqa \
  --n_train_per_task 40 --n_test_per_task 40 \
  --max_new_tokens 16 \
  --out_dir results_run1

注意
----
- residual 这里采用 control_pool 作为“非共享方向探针”，维数通常 64。
  它不是完整 d-k 的正交补，但对相关/流分析足够稳健且可控。
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from data_utils import build_mixture_prompts, list_supported_tasks
from model_utils import load_model_and_tokenizer
from subspace_utils import fit_linear_map_ridge, r2_score, subspace_similarity
from flow_utils import move_bases_to_device, greedy_collect_shared_residual_features


def parse_csv_list(s: str) -> List[str]:
    if s is None or s.strip() == "":
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def build_xy(
    examples: List[Dict],
    src_layer: int,
    tgt_layer: int,
    src_key: str,
    tgt_key: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """把 list[example] 展平成 (X,Y)。

    example[src_key][src_layer] : [T,k_src]
    example[tgt_key][tgt_layer] : [T,k_tgt]
    """
    Xs: List[np.ndarray] = []
    Ys: List[np.ndarray] = []
    for ex in examples:
        X = ex[src_key][src_layer]
        Y = ex[tgt_key][tgt_layer]
        T = min(X.shape[0], Y.shape[0])
        if T <= 0:
            continue
        Xs.append(X[:T])
        Ys.append(Y[:T])
    if len(Xs) == 0:
        return np.zeros((0, 1), dtype=np.float32), np.zeros((0, 1), dtype=np.float32)
    return np.concatenate(Xs, axis=0), np.concatenate(Ys, axis=0)


def eval_r2_pair(
    train_exs: List[Dict],
    test_exs: List[Dict],
    src_layer: int,
    tgt_layer: int,
    src_key: str,
    tgt_key: str,
    *,
    ridge: float = 1e-3,
    shuffle_train: bool = False,
    seed: int = 0,
) -> float:
    Xtr, Ytr = build_xy(train_exs, src_layer, tgt_layer, src_key, tgt_key)
    Xte, Yte = build_xy(test_exs, src_layer, tgt_layer, src_key, tgt_key)
    if Xtr.shape[0] == 0 or Xte.shape[0] == 0:
        return float("nan")

    if shuffle_train:
        rng = np.random.RandomState(seed)
        idx = np.arange(Xtr.shape[0])
        rng.shuffle(idx)
        Xtr = Xtr[idx]

    A = fit_linear_map_ridge(Xtr, Ytr, ridge=ridge)  # [k_tgt, k_src]
    Yhat = Xte @ A.T
    return r2_score(Yte, Yhat)


def heatmap_save(mat: np.ndarray, layers: List[int], path: str, title: str, cbar: str):
    L = len(layers)
    plt.figure(figsize=(7.2, 6))
    plt.imshow(mat, interpolation="nearest")
    plt.colorbar(label=cbar)
    plt.xticks(range(L), [str(x) for x in layers], rotation=45, ha="right")
    plt.yticks(range(L), [str(x) for x in layers])
    plt.xlabel("target layer m")
    plt.ylabel("source layer ℓ")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_r2_vs_distance(
    layers: List[int],
    R2: np.ndarray,
    R2_shuf: np.ndarray,
    path: str,
    title: str,
):
    L = len(layers)
    dists: List[int] = []
    vals: List[float] = []
    vals_shuf: List[float] = []

    for i in range(L):
        for j in range(L):
            if i == j:
                continue
            dist = abs(layers[j] - layers[i])
            v = float(R2[i, j])
            vs = float(R2_shuf[i, j])
            if np.isfinite(v):
                dists.append(dist)
                vals.append(v)
                vals_shuf.append(vs)

    if len(dists) == 0:
        print("[warn] no valid R2 values to plot distance curve")
        return

    dists = np.asarray(dists)
    vals = np.asarray(vals)
    vals_shuf = np.asarray(vals_shuf)

    uniq = np.unique(dists)
    mean_v, se_v, mean_s, se_s = [], [], [], []
    for d in uniq:
        m = vals[dists == d]
        s = vals_shuf[dists == d]
        mean_v.append(float(np.mean(m)))
        se_v.append(float(np.std(m, ddof=1) / max(len(m) ** 0.5, 1.0)))
        mean_s.append(float(np.mean(s)))
        se_s.append(float(np.std(s, ddof=1) / max(len(s) ** 0.5, 1.0)))

    plt.figure(figsize=(7.2, 4))
    plt.errorbar(uniq, mean_v, yerr=se_v, marker="o", label="true")
    plt.errorbar(uniq, mean_s, yerr=se_s, marker="o", label="shuffle baseline")
    plt.xlabel("layer distance |Δℓ|")
    plt.ylabel("R²")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shared_pt", type=str, required=True)
    ap.add_argument("--model", type=str, default=None,
                    help="可选：覆盖 shared_pt 里保存的 model 字段")
    ap.add_argument("--tasks", type=str, required=True,
                    help=f"Comma-separated tasks. Supported: {','.join(list_supported_tasks())}")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split_source", type=str, default="eval", choices=["subspace", "eval"])
    ap.add_argument("--n_train_per_task", type=int, default=40)
    ap.add_argument("--n_test_per_task", type=int, default=40)
    ap.add_argument("--template_randomization", type=int, default=1)
    ap.add_argument("--template_seed", type=int, default=0)
    ap.add_argument("--shuffle_choices", type=int, default=1)
    ap.add_argument("--add_answer_prefix", type=int, default=1)
    ap.add_argument("--answer_prefix", type=str, default="Final answer:")
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="auto")
    ap.add_argument("--cache_dir", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--compute_reverse", type=int, default=0,
                    help="是否额外计算 resid->shared（会多跑一倍回归）")
    args = ap.parse_args()

    payload = torch.load(args.shared_pt, map_location="cpu")
    layers: List[int] = list(payload["layers"])
    Qs_cpu: Dict[int, torch.Tensor] = payload["Q_shared_by_layer"]
    Qr_cpu: Dict[int, torch.Tensor] = payload["Q_control_pool_by_layer"]
    model_name = args.model or payload.get("model", None)
    if model_name is None:
        raise ValueError("Need --model or shared_pt must contain model field.")

    tasks = parse_csv_list(args.tasks)
    if len(tasks) == 0:
        raise ValueError("Need at least 1 task.")

    out_dir = args.out_dir or os.path.dirname(args.shared_pt)
    fig_dir = os.path.join(out_dir, "figs")
    os.makedirs(fig_dir, exist_ok=True)

    # (A) Structural: shared(ℓ) vs residual(m)
    L = len(layers)
    S_sr = np.zeros((L, L), dtype=np.float64)
    for i, li in enumerate(layers):
        Qa = Qs_cpu[li].numpy()
        for j, lj in enumerate(layers):
            Qb = Qr_cpu[lj].numpy()
            r = min(Qa.shape[1], Qb.shape[1])
            sim, _ = subspace_similarity(Qa, Qb, r=r)
            S_sr[i, j] = sim

    out_struct = os.path.join(fig_dir, "q3_struct_shared_vs_resid_heatmap.png")
    heatmap_save(
        S_sr,
        layers,
        out_struct,
        title="Q3 Structural: similarity between shared(ℓ) and residual/control(m)",
        cbar="mean(sigma^2)",
    )
    print(f"[saved] {out_struct}")

    # Load model
    model, tok = load_model_and_tokenizer(model_name, device=args.device, dtype=args.dtype, cache_dir=args.cache_dir)

    # Move bases to device
    Qs = move_bases_to_device(Qs_cpu, device=args.device, dtype=torch.float32)
    Qr = move_bases_to_device(Qr_cpu, device=args.device, dtype=torch.float32)

    # Build prompts
    train_prompts = build_mixture_prompts(
        tasks,
        n_per_task=args.n_train_per_task,
        seed=args.seed + 123,
        split=args.split_source,
        template_randomization=bool(args.template_randomization),
        template_seed=args.template_seed,
        shuffle_choices=bool(args.shuffle_choices),
        add_answer_prefix=bool(args.add_answer_prefix),
        answer_prefix=args.answer_prefix,
    )
    test_prompts = build_mixture_prompts(
        tasks,
        n_per_task=args.n_test_per_task,
        seed=args.seed + 456,
        split=args.split_source,
        template_randomization=bool(args.template_randomization),
        template_seed=args.template_seed,
        shuffle_choices=bool(args.shuffle_choices),
        add_answer_prefix=bool(args.add_answer_prefix),
        answer_prefix=args.answer_prefix,
    )

    # Collect features (train/test)
    def collect(prompts: List[str], split_name: str) -> List[Dict]:
        out: List[Dict] = []
        it = tqdm(prompts, desc=f"collect shared/resid {split_name}", total=len(prompts))
        for p in it:
            feats, gen_ids = greedy_collect_shared_residual_features(
                model, tok, p, layers, Qs, Qr,
                max_new_tokens=args.max_new_tokens,
                device=args.device,
                stop_on_eos=True,
            )
            out.append({
                "shared_state": feats["shared_state"],
                "resid_state": feats["resid_state"],
                "shared_update": feats["shared_update"],
                "resid_update": feats["resid_update"],
                "gen_ids": gen_ids,
            })
        return out

    train_exs = collect(train_prompts, "train")
    test_exs = collect(test_prompts, "test")

    # Save datasets for reuse
    train_path = os.path.join(out_dir, "shared_residual_train.pt")
    test_path = os.path.join(out_dir, "shared_residual_test.pt")
    meta = {
        "shared_pt": args.shared_pt,
        "model": model_name,
        "tasks": tasks,
        "layers": layers,
        "split_source": args.split_source,
        "n_train_per_task": args.n_train_per_task,
        "n_test_per_task": args.n_test_per_task,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
    }
    torch.save({"layers": layers, "examples": train_exs, "meta": meta}, train_path)
    torch.save({"layers": layers, "examples": test_exs, "meta": meta}, test_path)
    print(f"[saved] {train_path}")
    print(f"[saved] {test_path}")

    # (B) Functional: R2 shared->resid
    R2_state = np.full((L, L), np.nan, dtype=np.float64)
    R2_state_shuf = np.full((L, L), np.nan, dtype=np.float64)
    R2_update = np.full((L, L), np.nan, dtype=np.float64)
    R2_update_shuf = np.full((L, L), np.nan, dtype=np.float64)

    rows = []

    for i, li in enumerate(layers):
        for j, lj in enumerate(layers):
            r2_sr = eval_r2_pair(train_exs, test_exs, li, lj, "shared_state", "resid_state",
                                 ridge=args.ridge, shuffle_train=False, seed=args.seed)
            r2_sr_sh = eval_r2_pair(train_exs, test_exs, li, lj, "shared_state", "resid_state",
                                    ridge=args.ridge, shuffle_train=True, seed=args.seed)
            R2_state[i, j] = r2_sr
            R2_state_shuf[i, j] = r2_sr_sh
            rows.append({
                "src_layer": li, "tgt_layer": lj,
                "src": "shared_state", "tgt": "resid_state",
                "r2": r2_sr, "r2_shuf": r2_sr_sh,
            })

            r2_su = eval_r2_pair(train_exs, test_exs, li, lj, "shared_state", "resid_update",
                                 ridge=args.ridge, shuffle_train=False, seed=args.seed)
            r2_su_sh = eval_r2_pair(train_exs, test_exs, li, lj, "shared_state", "resid_update",
                                    ridge=args.ridge, shuffle_train=True, seed=args.seed)
            R2_update[i, j] = r2_su
            R2_update_shuf[i, j] = r2_su_sh
            rows.append({
                "src_layer": li, "tgt_layer": lj,
                "src": "shared_state", "tgt": "resid_update",
                "r2": r2_su, "r2_shuf": r2_su_sh,
            })

            if args.compute_reverse:
                r2_rs = eval_r2_pair(train_exs, test_exs, li, lj, "resid_state", "shared_state",
                                     ridge=args.ridge, shuffle_train=False, seed=args.seed)
                r2_rs_sh = eval_r2_pair(train_exs, test_exs, li, lj, "resid_state", "shared_state",
                                        ridge=args.ridge, shuffle_train=True, seed=args.seed)
                rows.append({
                    "src_layer": li, "tgt_layer": lj,
                    "src": "resid_state", "tgt": "shared_state",
                    "r2": r2_rs, "r2_shuf": r2_rs_sh,
                })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "shared_residual_r2_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"[saved] {csv_path}")

    # Heatmaps
    out_hm1 = os.path.join(fig_dir, "q3_r2_shared_to_resid_state_heatmap.png")
    heatmap_save(
        R2_state,
        layers,
        out_hm1,
        title="Q3 Functional: R²(shared_state(ℓ) -> resid_state(m))",
        cbar="R²",
    )
    print(f"[saved] {out_hm1}")

    out_hm2 = os.path.join(fig_dir, "q3_r2_shared_to_resid_update_heatmap.png")
    heatmap_save(
        R2_update,
        layers,
        out_hm2,
        title="Q3 Functional: R²(shared_state(ℓ) -> resid_update(m))",
        cbar="R²",
    )
    print(f"[saved] {out_hm2}")

    # Distance curve（用 shared_state -> resid_update 作为主曲线，通常更像“流/更新”）
    out_dist = os.path.join(fig_dir, "q3_r2_vs_distance.png")
    plot_r2_vs_distance(
        layers,
        R2_update,
        R2_update_shuf,
        out_dist,
        title="Q3: Predictability of residual updates from shared coords vs layer distance",
    )
    print(f"[saved] {out_dist}")

    # quick summary
    offdiag = R2_update.copy()
    np.fill_diagonal(offdiag, np.nan)
    print("[summary] mean R²(shared->resid_update) offdiag =", float(np.nanmean(offdiag)))


if __name__ == "__main__":
    main()
