
# -*- coding: utf-8 -*-
"""
03_collect_shared_coords.py

Phase 3 (Q2 数据收集): 逐步 decode，收集共享坐标 z_t(ℓ)=Q_S(ℓ)^T h_t(ℓ)。

输入：
  --shared_pt: 01 输出 shared_subspaces.pt

输出：
  - coords_train.pt / coords_test.pt：
      {
        "layers": [...],
        "k_shared_by_layer": {ell: k},
        "trajectories": [
           {ell: np.ndarray[T,k_ell], ...},   # 每个 prompt 一个 dict
           ...
        ],
        "meta": {...}
      }

运行示例：
  python 03_collect_shared_coords.py \
    --shared_pt results_run1/shared_subspaces.pt \
    --tasks aqua,commonsenseqa,strategyqa,arc_challenge,openbookqa \
    --n_train_per_task 40 --n_test_per_task 40 \
    --max_new_tokens 16 \
    --out_dir results_run1
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

from data_utils import build_mixture_prompts, list_supported_tasks
from model_utils import load_model_and_tokenizer, generate_and_collect_hidden_states


def parse_csv_list(s: str) -> List[str]:
    if s is None or s.strip() == "":
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def project_trajectory(traj_h: Dict[int, List[torch.Tensor]], Qs: Dict[int, torch.Tensor]) -> Dict[int, np.ndarray]:
    """
    traj_h: layer -> list[t] of h_t [d] (CPU torch)
    Qs: layer -> Q_shared [d,k] (CPU torch)
    返回 layer -> Z [T,k] (numpy float32)
    """
    out: Dict[int, np.ndarray] = {}
    for ell, steps in traj_h.items():
        if len(steps) == 0:
            out[ell] = np.zeros((0, Qs[ell].shape[1]), dtype=np.float32)
            continue
        H = torch.stack(steps, dim=0).float()  # [T,d]
        Q = Qs[ell].float()                    # [d,k]
        Z = (H @ Q).cpu().numpy().astype(np.float32)  # [T,k]
        out[ell] = Z
    return out


def collect_split(
    *,
    split_name: str,
    prompts: List[str],
    model,
    tok,
    layers: List[int],
    Qs: Dict[int, torch.Tensor],
    max_new_tokens: int,
    device: str,
    do_sample: bool,
    temperature: float,
    stop_on_eos: bool,
) -> List[Dict[int, np.ndarray]]:
    trajectories: List[Dict[int, np.ndarray]] = []
    it = tqdm(prompts, desc=f"collect {split_name}", total=len(prompts))
    for p in it:
        traj_h, gen_ids = generate_and_collect_hidden_states(
            model, tok, p, layers,
            max_new_tokens=max_new_tokens,
            device=device,
            do_sample=do_sample,
            temperature=temperature,
            stop_on_eos=stop_on_eos,
        )
        traj_z = project_trajectory(traj_h, Qs)
        trajectories.append(traj_z)
    return trajectories


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shared_pt", type=str, required=True)
    ap.add_argument("--model", type=str, default=None,
                    help="可选：覆盖 shared_pt 里保存的 model 字段")
    ap.add_argument("--tasks", type=str, required=True,
                    help=f"Comma-separated tasks. Supported: {','.join(list_supported_tasks())}")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split_source", type=str, default="eval", choices=["subspace", "eval"],
                    help="从哪个 split 抽 prompt 用于轨迹/回归。建议 eval（更独立）")
    ap.add_argument("--n_train_per_task", type=int, default=40)
    ap.add_argument("--n_test_per_task", type=int, default=40)
    ap.add_argument("--template_randomization", type=int, default=1)
    ap.add_argument("--template_seed", type=int, default=0)
    ap.add_argument("--shuffle_choices", type=int, default=1)
    ap.add_argument("--add_answer_prefix", type=int, default=1)
    ap.add_argument("--answer_prefix", type=str, default="Final answer:")
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--do_sample", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--stop_on_eos", type=int, default=1)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="auto")
    ap.add_argument("--cache_dir", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default=None,
                    help="输出目录（默认与 shared_pt 同目录）")
    args = ap.parse_args()

    payload = torch.load(args.shared_pt, map_location="cpu")
    layers: List[int] = list(payload["layers"])
    Qs: Dict[int, torch.Tensor] = payload["Q_shared_by_layer"]
    k_shared_by_layer = payload["k_shared_by_layer"]
    model_name = args.model or payload.get("model", None)
    if model_name is None:
        raise ValueError("Need --model or shared_pt must contain model field.")

    tasks = parse_csv_list(args.tasks)
    if len(tasks) == 0:
        raise ValueError("Need at least 1 task for collecting trajectories.")

    out_dir = args.out_dir or os.path.dirname(args.shared_pt)
    os.makedirs(out_dir, exist_ok=True)

    model, tok = load_model_and_tokenizer(model_name, device=args.device, dtype=args.dtype, cache_dir=args.cache_dir)

    # build prompts
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

    # collect
    traj_train = collect_split(
        split_name="train",
        prompts=train_prompts,
        model=model,
        tok=tok,
        layers=layers,
        Qs=Qs,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        do_sample=bool(args.do_sample),
        temperature=args.temperature,
        stop_on_eos=bool(args.stop_on_eos),
    )
    traj_test = collect_split(
        split_name="test",
        prompts=test_prompts,
        model=model,
        tok=tok,
        layers=layers,
        Qs=Qs,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        do_sample=bool(args.do_sample),
        temperature=args.temperature,
        stop_on_eos=bool(args.stop_on_eos),
    )

    # save
    out_train = os.path.join(out_dir, "coords_train.pt")
    out_test = os.path.join(out_dir, "coords_test.pt")
    meta = {
        "shared_pt": args.shared_pt,
        "model": model_name,
        "tasks": tasks,
        "layers": layers,
        "split_source": args.split_source,
        "n_train_per_task": args.n_train_per_task,
        "n_test_per_task": args.n_test_per_task,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": bool(args.do_sample),
        "temperature": args.temperature,
        "stop_on_eos": bool(args.stop_on_eos),
        "seed": args.seed,
    }

    torch.save(
        {"layers": layers, "k_shared_by_layer": k_shared_by_layer, "trajectories": traj_train, "meta": meta},
        out_train
    )
    torch.save(
        {"layers": layers, "k_shared_by_layer": k_shared_by_layer, "trajectories": traj_test, "meta": meta},
        out_test
    )
    print(f"[done] saved train -> {out_train}")
    print(f"[done] saved test  -> {out_test}")


if __name__ == "__main__":
    main()
