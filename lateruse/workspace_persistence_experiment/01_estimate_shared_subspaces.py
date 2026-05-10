
# -*- coding: utf-8 -*-
"""
01_estimate_shared_subspaces.py

Phase 1: 在多个层估计共享子空间 Q_S(ℓ)。

输出（保存到 --out_dir/shared_subspaces.pt）：
  - layers: 层列表（0-based）
  - tasks: 任务列表
  - Q_shared_by_layer: dict[int -> torch.Tensor(d,kℓ)]
  - k_shared_by_layer: dict[int -> int]
  - eigvals_by_layer: dict[int -> torch.Tensor(r_union)]
  - Q_control_pool_by_layer: dict[int -> torch.Tensor(d,k_pool)]
  - config: 运行配置（便于复现实验）

运行示例：
  python 01_estimate_shared_subspaces.py \
    --model meta-llama/Llama-2-7b-hf \
    --tasks aqua,commonsenseqa,strategyqa,arc_challenge,openbookqa \
    --layers 2,5,8,11,14,17,20,23 \
    --n_subspace 256 --pca_dim 128 \
    --answer_prefix "Final answer:" \
    --out_dir results_run1
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import torch

from data_utils import list_supported_tasks, load_task, get_prompts
from model_utils import load_model_and_tokenizer, get_num_layers, get_hidden_size, collect_last_token_states_multi_layer
from subspace_utils import pca_basis, estimate_shared_subspace


def parse_csv_list(s: str) -> List[str]:
    if s is None or s.strip() == "":
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_int_list(s: str) -> List[int]:
    if s is None or s.strip() == "":
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def auto_layers(num_layers: int, n: int = 8) -> List[int]:
    # 在 [0, num_layers-1] 均匀取 n 个层（避开 embedding 层）
    if n <= 0:
        raise ValueError("n must be positive")
    if num_layers <= n:
        return list(range(num_layers))
    # 尽量覆盖全深度，略偏中后层也可以
    idx = torch.linspace(0, num_layers - 1, steps=n).round().to(torch.int64).tolist()
    # 去重保持顺序
    out = []
    for i in idx:
        if i not in out:
            out.append(int(i))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="HF model name or local path")
    ap.add_argument("--tasks", type=str, required=True,
                    help=f"Comma-separated tasks. Supported: {','.join(list_supported_tasks())}")
    ap.add_argument("--layers", type=str, default="",
                    help="Comma-separated 0-based layer ids. Empty => auto 8 layers.")
    ap.add_argument("--n_subspace", type=int, default=256)
    ap.add_argument("--n_eval", type=int, default=0, help="这里不需要 eval；保留接口以兼容 loader")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--template_randomization", type=int, default=1)
    ap.add_argument("--template_seed", type=int, default=0)
    ap.add_argument("--shuffle_choices", type=int, default=1)
    ap.add_argument("--add_answer_prefix", type=int, default=1)
    ap.add_argument("--answer_prefix", type=str, default="Final answer:")
    ap.add_argument("--pca_dim", type=int, default=128, help="每个任务 PCA 子空间维数 p")
    ap.add_argument("--k_shared", type=int, default=-1, help="共享子空间维数 k；-1 表示自动")
    ap.add_argument("--tau_shared", type=float, default=-1.0, help="自动选 k 的阈值 tau；-1 表示使用默认")
    ap.add_argument("--max_k", type=int, default=64)
    ap.add_argument("--min_k", type=int, default=1)
    ap.add_argument("--control_pool_dim", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="auto", help="auto/float16/bfloat16/float32")
    ap.add_argument("--cache_dir", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default="results_shared")
    args = ap.parse_args()

    tasks = parse_csv_list(args.tasks)
    if len(tasks) < 2:
        raise ValueError("Need at least 2 tasks to estimate shared subspace.")

    model, tok = load_model_and_tokenizer(args.model, device=args.device, dtype=args.dtype, cache_dir=args.cache_dir)
    num_layers = get_num_layers(model)
    d = get_hidden_size(model)

    layers = parse_int_list(args.layers)
    if len(layers) == 0:
        layers = auto_layers(num_layers, n=8)

    # sanitize layers
    layers = [int(x) for x in layers]
    for ell in layers:
        if ell < 0 or ell >= num_layers:
            raise ValueError(f"Layer {ell} out of range [0,{num_layers-1}]")

    print(f"[info] model={args.model} num_layers={num_layers} hidden_size={d}")
    print(f"[info] tasks={tasks}")
    print(f"[info] layers={layers}")

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) 为每个 task 收集各层的 last-token states，并做 PCA
    Q_task_by_layer: Dict[int, List[torch.Tensor]] = {ell: [] for ell in layers}

    for ti, task in enumerate(tasks):
        sub_exs, _, meta = load_task(
            task,
            n_subspace=args.n_subspace,
            n_eval=max(args.n_eval, 0),
            seed=args.seed + 1000 * ti,
            template_randomization=bool(args.template_randomization),
            template_seed=args.template_seed,
            shuffle_choices=bool(args.shuffle_choices),
            add_answer_prefix=bool(args.add_answer_prefix),
            answer_prefix=args.answer_prefix,
        )
        prompts = get_prompts(sub_exs)
        print(f"[task={task}] loaded {len(prompts)} prompts | meta keys={list(meta.keys())[:6]}...")

        X_by_layer = collect_last_token_states_multi_layer(
            model, tok, prompts, layers,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=args.device,
            show_progress=True,
        )

        # PCA per layer
        for ell in layers:
            X = X_by_layer[ell].to(args.device)  # PCA 可用 GPU
            Q, S = pca_basis(X, k=args.pca_dim, center=True)
            Q_task_by_layer[ell].append(Q.detach().float().cpu())
            # 解释方差统计可选：这里不强行保存，避免文件过大

        # 释放显存
        del X_by_layer
        torch.cuda.empty_cache() if args.device.startswith("cuda") else None

    # 2) 每个层估计共享子空间
    Q_shared_by_layer: Dict[int, torch.Tensor] = {}
    k_shared_by_layer: Dict[int, int] = {}
    eigvals_by_layer: Dict[int, torch.Tensor] = {}
    Q_control_pool_by_layer: Dict[int, torch.Tensor] = {}

    k_shared = None if args.k_shared < 0 else int(args.k_shared)
    tau_shared = None if args.tau_shared < 0 else float(args.tau_shared)

    for ell in layers:
        res = estimate_shared_subspace(
            Q_task_by_layer[ell],
            k_shared=k_shared,
            tau_shared=tau_shared,
            max_k=args.max_k,
            min_k=args.min_k,
            control_pool_dim=args.control_pool_dim,
            control_seed=args.seed + 999,
        )
        Q_shared_by_layer[ell] = res.Q_shared.cpu()
        k_shared_by_layer[ell] = int(res.k_shared)
        eigvals_by_layer[ell] = res.eigvals.cpu()
        Q_control_pool_by_layer[ell] = res.Q_control_pool.cpu()
        print(f"[layer={ell}] k_shared={res.k_shared} | top eigvals={res.eigvals[:5].tolist()}")

    # 3) 保存
    out_path = os.path.join(args.out_dir, "shared_subspaces.pt")
    payload = {
        "model": args.model,
        "layers": layers,
        "tasks": tasks,
        "Q_shared_by_layer": Q_shared_by_layer,
        "k_shared_by_layer": k_shared_by_layer,
        "eigvals_by_layer": eigvals_by_layer,
        "Q_control_pool_by_layer": Q_control_pool_by_layer,
        "config": vars(args),
    }
    torch.save(payload, out_path)
    print(f"[done] saved to {out_path}")


if __name__ == "__main__":
    main()
