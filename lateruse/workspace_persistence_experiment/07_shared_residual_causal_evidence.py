# -*- coding: utf-8 -*-
"""07_shared_residual_causal_evidence.py

Q3 增强（因果证据）：跨层 shared ↔ residual 之间是否存在可测的因果影响？

核心思想
--------
- 仅靠相关/可预测性（06）不足以证明信息“驱动”关系。
- 这里做一个最小但稳健的干预：

  在源层 src_layer 的 decode residual stream 上，
  *消融*（移除）某个组件：
    - shared: 投影到 Q_shared(src) 的分量
    - residual: 投影到 Q_control_pool(src) 的分量

  然后在相同 continuation token 序列（teacher forcing）下，
  观测目标层 m 的 residual 特征（state/update）如何变化。

输出 deliverables
----------------
- shared_residual_causal.csv
- figs/q3_causal_effect_vs_layer.png
- 还会打印：平均 token logprob 变化（说明干预确实影响输出）

运行示例
--------
python 07_shared_residual_causal_evidence.py \
  --shared_pt results_run1/shared_subspaces.pt \
  --model YOUR_MODEL \
  --tasks aqua,commonsenseqa,strategyqa,arc_challenge,openbookqa \
  --src_layer 11 \
  --ablate shared \
  --n_prompts 80 \
  --max_new_tokens 16 \
  --answer_prefix "Final answer:" \
  --out_dir results_run1

建议
----
- 为了减少 stochasticity，本脚本默认使用 greedy 生成 continuation，
  并对 baseline/ablation 都采用 teacher forcing，保证跨条件 token 对齐。
- 如果你已经用 06 发现某些距离/层段耦合强，可以把 src_layer 设在那段。
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from data_utils import build_mixture_prompts, list_supported_tasks
from model_utils import load_model_and_tokenizer, get_transformer_layers
from flow_utils import (
    move_bases_to_device,
    greedy_generate_ids,
    teacher_forced_collect_shared_residual_features,
    ComponentAblationHook,
)


def parse_csv_list(s: str) -> List[str]:
    if s is None or s.strip() == "":
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shared_pt", type=str, required=True)
    ap.add_argument("--model", type=str, default=None,
                    help="可选：覆盖 shared_pt 里保存的 model 字段")
    ap.add_argument("--tasks", type=str, required=True,
                    help=f"Comma-separated tasks. Supported: {','.join(list_supported_tasks())}")
    ap.add_argument("--src_layer", type=int, required=True)
    ap.add_argument("--ablate", type=str, default="shared", choices=["shared", "residual"],
                    help="在 src_layer 消融 shared 或 residual(control_pool) 组件")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split_source", type=str, default="eval", choices=["subspace", "eval"])
    ap.add_argument("--n_prompts", type=int, default=80)
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
    args = ap.parse_args()

    payload = torch.load(args.shared_pt, map_location="cpu")
    layers: List[int] = list(payload["layers"])
    Qs_cpu: Dict[int, torch.Tensor] = payload["Q_shared_by_layer"]
    Qr_cpu: Dict[int, torch.Tensor] = payload["Q_control_pool_by_layer"]

    model_name = args.model or payload.get("model", None)
    if model_name is None:
        raise ValueError("Need --model or shared_pt must contain model field.")

    if args.src_layer not in layers:
        raise ValueError(f"src_layer={args.src_layer} not in shared_pt layers={layers}")

    tasks = parse_csv_list(args.tasks)
    if len(tasks) == 0:
        raise ValueError("Need at least 1 task")

    out_dir = args.out_dir or os.path.dirname(args.shared_pt)
    fig_dir = os.path.join(out_dir, "figs")
    os.makedirs(fig_dir, exist_ok=True)

    # model
    model, tok = load_model_and_tokenizer(model_name, device=args.device, dtype=args.dtype, cache_dir=args.cache_dir)
    blocks = get_transformer_layers(model)

    # bases
    Qs = move_bases_to_device(Qs_cpu, device=args.device, dtype=torch.float32)
    Qr = move_bases_to_device(Qr_cpu, device=args.device, dtype=torch.float32)

    # prompts mixture
    prompts_all = build_mixture_prompts(
        tasks,
        n_per_task=args.n_prompts,
        seed=args.seed,
        split=args.split_source,
        template_randomization=bool(args.template_randomization),
        template_seed=args.template_seed,
        shuffle_choices=bool(args.shuffle_choices),
        add_answer_prefix=bool(args.add_answer_prefix),
        answer_prefix=args.answer_prefix,
    )
    prompts = prompts_all[: int(args.n_prompts) * max(1, len(tasks))]

    # aggregators per target layer
    sum_diff2_state = {ell: 0.0 for ell in layers}
    sum_base2_state = {ell: 0.0 for ell in layers}
    sum_diff2_update = {ell: 0.0 for ell in layers}
    sum_base2_update = {ell: 0.0 for ell in layers}

    total_tokens = 0
    sum_logprob_base = 0.0
    sum_logprob_ab = 0.0

    # choose ablation basis
    if args.ablate == "shared":
        Q_ab = Qs[args.src_layer]
    else:
        Q_ab = Qr[args.src_layer]

    hook = ComponentAblationHook(Q=Q_ab, steps=None)

    it = tqdm(prompts, desc="causal runs", total=len(prompts))
    for p in it:
        # 1) decide continuation ids under baseline greedy
        cont = greedy_generate_ids(
            model, tok, p,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            stop_on_eos=True,
        )
        if len(cont) == 0:
            continue

        # 2) baseline teacher-forced (no hooks)
        feats_base, lps_base = teacher_forced_collect_shared_residual_features(
            model, tok, p, cont, layers, Qs, Qr, device=args.device
        )

        # 3) ablated teacher-forced (hook on src_layer)
        hook.reset()
        h = blocks[args.src_layer].register_forward_hook(hook)
        try:
            feats_ab, lps_ab = teacher_forced_collect_shared_residual_features(
                model, tok, p, cont, layers, Qs, Qr, device=args.device
            )
        finally:
            h.remove()

        T = min(len(lps_base), len(lps_ab))
        if T <= 0:
            continue

        # token logprob aggregation
        sum_logprob_base += float(np.sum(lps_base[:T]))
        sum_logprob_ab += float(np.sum(lps_ab[:T]))
        total_tokens += int(T)

        # residual feature changes per target layer
        for ell in layers:
            base_state = feats_base["resid_state"][ell]
            ab_state = feats_ab["resid_state"][ell]
            base_upd = feats_base["resid_update"][ell]
            ab_upd = feats_ab["resid_update"][ell]

            TT = min(base_state.shape[0], ab_state.shape[0], base_upd.shape[0], ab_upd.shape[0], T)
            if TT <= 0:
                continue

            ds = ab_state[:TT] - base_state[:TT]
            du = ab_upd[:TT] - base_upd[:TT]

            sum_diff2_state[ell] += float(np.sum(ds ** 2))
            sum_base2_state[ell] += float(np.sum(base_state[:TT] ** 2))

            sum_diff2_update[ell] += float(np.sum(du ** 2))
            sum_base2_update[ell] += float(np.sum(base_upd[:TT] ** 2))

    # summarize per layer
    rows = []
    for ell in layers:
        diff2_s = sum_diff2_state[ell]
        base2_s = sum_base2_state[ell]
        diff2_u = sum_diff2_update[ell]
        base2_u = sum_base2_update[ell]

        norm_s = diff2_s / (base2_s + 1e-12)
        norm_u = diff2_u / (base2_u + 1e-12)
        rows.append({
            "src_layer": int(args.src_layer),
            "tgt_layer": int(ell),
            "ablate": args.ablate,
            "norm_mse_change_resid_state": float(norm_s),
            "norm_mse_change_resid_update": float(norm_u),
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "shared_residual_causal.csv")
    df.to_csv(csv_path, index=False)
    print(f"[saved] {csv_path}")

    # output-level evidence: continuation log-likelihood shift
    if total_tokens > 0:
        avg_lp_base = sum_logprob_base / float(total_tokens)
        avg_lp_ab = sum_logprob_ab / float(total_tokens)
        print("[output] avg token logprob baseline =", float(avg_lp_base))
        print("[output] avg token logprob ablated  =", float(avg_lp_ab))
        print("[output] delta (ab - base)          =", float(avg_lp_ab - avg_lp_base))
    else:
        print("[warn] no tokens processed; cannot compute output logprob shift")

    # plot causal effect vs target layer
    xs = np.asarray(layers, dtype=np.int64)
    ys_state = np.asarray([df[df.tgt_layer == int(ell)]["norm_mse_change_resid_state"].values[0] for ell in layers], dtype=np.float64)
    ys_upd = np.asarray([df[df.tgt_layer == int(ell)]["norm_mse_change_resid_update"].values[0] for ell in layers], dtype=np.float64)

    plt.figure(figsize=(7.2, 4))
    plt.plot(xs, ys_state, marker="o", label="effect on resid_state")
    plt.plot(xs, ys_upd, marker="o", label="effect on resid_update")
    plt.axvline(args.src_layer, linestyle="--")
    plt.xlabel("target layer m")
    plt.ylabel("normalized mean-square change")
    plt.title(f"Q3 Causal: ablate {args.ablate} at layer {args.src_layer}")
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(fig_dir, "q3_causal_effect_vs_layer.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"[saved] {fig_path}")


if __name__ == "__main__":
    main()
