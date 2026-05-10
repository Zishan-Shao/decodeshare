
# -*- coding: utf-8 -*-
"""
05_cross_layer_patching_optional.py  (可选/增强版)

强因果检验（可选）：
- 在目标层 m 做 shared-subspace ablation（移除共享分量）
- 再用源层 ℓ 的共享坐标，经线性映射 A_{ℓ->m} 预测目标层共享分量，并 patch 回去
- 观察 next-token forced-choice（A/B/C/D/E）上的 logprob/margin 是否被“救回”

⚠️ 重要限制/假设：
1) 这个脚本专门面向“答案是单个字母”的 MC 任务（默认 aqua: A-E）。
2) 评估的是 answer_prefix 后的“下一个 token”。为了让下一个 token 更可能是字母，
   建议 answer_prefix 以冒号+空格结尾，例如 "Final answer: "（带空格）。
3) patch 发生在 prefill 的最后一个 token 位置（answer_prefix 的末尾），属于最小实现。

运行示例：
  python 05_cross_layer_patching_optional.py \
    --shared_pt results_run1/shared_subspaces.pt \
    --model meta-llama/Llama-2-7b-hf \
    --task aqua \
    --src_layer 11 --tgt_layer 14 \
    --n_calib 200 --n_test 200 \
    --answer_prefix "Final answer: " \
    --out_dir results_run1
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd

from data_utils import load_task
from model_utils import load_model_and_tokenizer, get_transformer_layers
from subspace_utils import fit_linear_map_ridge


def get_choice_letters(task: str) -> List[str]:
    task = task.lower().strip()
    if task in {"aqua", "commonsenseqa"}:
        return list("ABCDE")
    if task in {"arc_challenge", "openbookqa"}:
        return list("ABCD")
    # 其他任务（如 strategyqa yes/no）不在此脚本范围
    raise ValueError(f"Task {task} not supported for letter forced-choice patching.")


@dataclass
class PatchCtx:
    z_src: torch.Tensor | None = None


def make_hooks_for_mode(
    *,
    blocks: List[torch.nn.Module],
    src_layer: int,
    tgt_layer: int,
    Qs_src: torch.Tensor,   # [d,k1] on device
    Qs_tgt: torch.Tensor,   # [d,k2] on device
    A: torch.Tensor,        # [k2,k1] on device
    mode: str,              # "baseline"|"ablate"|"patch"
    ctx: PatchCtx,
):
    hooks = []

    def hook_src(module, inputs, output):
        hs = output[0] if isinstance(output, (tuple, list)) else output  # [B,seq,d]
        x = hs[:, -1, :].float()  # [B,d]
        # z_src = x @ Qs_src  [B,k1]
        ctx.z_src = x @ Qs_src
        return output

    def hook_tgt(module, inputs, output):
        hs = output[0] if isinstance(output, (tuple, list)) else output
        B, S, d = hs.shape
        x = hs[:, -1, :].float()     # [B,d]
        z = x @ Qs_tgt               # [B,k2]
        shared = z @ Qs_tgt.T        # [B,d]
        if mode == "ablate":
            x_new = x - shared
        elif mode == "patch":
            if ctx.z_src is None:
                raise RuntimeError("ctx.z_src is None in patch mode; ensure src_layer < tgt_layer.")
            z_pred = ctx.z_src @ A.T            # [B,k2]
            pred = z_pred @ Qs_tgt.T            # [B,d]
            x_new = x - shared + pred
        else:
            return output

        hs2 = hs.clone()
        hs2[:, -1, :] = x_new.to(hs2.dtype)
        if isinstance(output, (tuple, list)):
            # 兼容 tuple 输出
            out = (hs2,) + tuple(output[1:])
            return out
        return hs2

    # register
    if mode in {"patch"}:
        hooks.append(blocks[src_layer].register_forward_hook(hook_src))
    if mode in {"ablate", "patch"}:
        hooks.append(blocks[tgt_layer].register_forward_hook(hook_tgt))
    return hooks


def next_token_logits_with_mode(
    model,
    tok,
    prompts: List[str],
    *,
    src_layer: int,
    tgt_layer: int,
    Qs_src: torch.Tensor,
    Qs_tgt: torch.Tensor,
    A: torch.Tensor,
    mode: str,
    device: str,
    batch_size: int,
    max_length: int,
) -> torch.Tensor:
    """
    对一批 prompts 做 prefill forward，返回最后位置的 next-token logits: [N, V]
    mode: baseline/ablate/patch
    """
    blocks = get_transformer_layers(model)
    ctx = PatchCtx()
    all_logits = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        enc = tok(batch_prompts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        ctx.z_src = None

        hooks = make_hooks_for_mode(
            blocks=blocks,
            src_layer=src_layer,
            tgt_layer=tgt_layer,
            Qs_src=Qs_src,
            Qs_tgt=Qs_tgt,
            A=A,
            mode=mode,
            ctx=ctx,
        )
        try:
            with torch.inference_mode():
                out = model(**enc, use_cache=False, return_dict=True)
                logits = out.logits[:, -1, :].detach().float().cpu()
                all_logits.append(logits)
        finally:
            for h in hooks:
                h.remove()

    return torch.cat(all_logits, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shared_pt", type=str, required=True)
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--task", type=str, default="aqua")
    ap.add_argument("--src_layer", type=int, required=True)
    ap.add_argument("--tgt_layer", type=int, required=True)
    ap.add_argument("--n_calib", type=int, default=200)
    ap.add_argument("--n_test", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--answer_prefix", type=str, default="Final answer: ")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="auto")
    ap.add_argument("--cache_dir", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default=None)
    args = ap.parse_args()

    if args.src_layer >= args.tgt_layer:
        raise ValueError("This minimal patching implementation assumes src_layer < tgt_layer.")

    payload = torch.load(args.shared_pt, map_location="cpu")
    Qs = payload["Q_shared_by_layer"]
    if args.src_layer not in Qs or args.tgt_layer not in Qs:
        raise ValueError("src_layer/tgt_layer must be included in shared_pt layers list.")

    # load model
    model, tok = load_model_and_tokenizer(args.model, device=args.device, dtype=args.dtype, cache_dir=args.cache_dir)

    # load data
    sub_exs, eval_exs, meta = load_task(
        args.task,
        n_subspace=max(args.n_calib, 1),
        n_eval=max(args.n_test, 1),
        seed=args.seed,
        template_randomization=True,
        template_seed=0,
        shuffle_choices=True,
        add_answer_prefix=True,
        answer_prefix=args.answer_prefix,
    )
    calib = sub_exs[:args.n_calib]
    test = eval_exs[:args.n_test]

    prompts_calib = [ex.prompt for ex in calib]
    prompts_test = [ex.prompt for ex in test]
    gold = [ex.gold for ex in test]

    letters = get_choice_letters(args.task)

    # token ids for letters
    letter_token_ids = {}
    for L in letters:
        ids = tok.encode(L, add_special_tokens=False)
        if len(ids) != 1:
            # 仍然继续，但只用第一个 token（最小实现）
            print(f"[warn] letter {L} tokenized into {ids}; using first token only.")
        letter_token_ids[L] = ids[0]

    # Fit A_{src->tgt} on calibration prompts using prompt last-token states
    # 这里直接用 logits_with_mode 的内部 hooks + Q 也行，但更简单：
    # 我们用 baseline forward 抓取 src/tgt 的 hidden states，然后投影得到 Z，再拟合 A。
    # 为了最小实现，这里用 next-token logits 收集函数在 patch 模式里已经能抓 z_src，
    # 但它不返回 z。我们实现一个更直接的采集：用 forward hooks 收集 hs_last。
    from model_utils import collect_last_token_states_multi_layer

    X = collect_last_token_states_multi_layer(
        model, tok, prompts_calib, [args.src_layer, args.tgt_layer],
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
        show_progress=True,
    )
    Qs_src = Qs[args.src_layer].to(args.device).float()
    Qs_tgt = Qs[args.tgt_layer].to(args.device).float()

    Z_src = (X[args.src_layer].float().to(args.device) @ Qs_src).cpu().numpy()
    Z_tgt = (X[args.tgt_layer].float().to(args.device) @ Qs_tgt).cpu().numpy()

    A_np = fit_linear_map_ridge(Z_src, Z_tgt, ridge=1e-3)  # [k2,k1]
    A = torch.tensor(A_np, device=args.device, dtype=torch.float32)

    # Evaluate baseline/ablate/patch on test prompts
    logits_base = next_token_logits_with_mode(
        model, tok, prompts_test,
        src_layer=args.src_layer, tgt_layer=args.tgt_layer,
        Qs_src=Qs_src, Qs_tgt=Qs_tgt, A=A,
        mode="baseline", device=args.device,
        batch_size=args.batch_size, max_length=args.max_length,
    )
    logits_ablate = next_token_logits_with_mode(
        model, tok, prompts_test,
        src_layer=args.src_layer, tgt_layer=args.tgt_layer,
        Qs_src=Qs_src, Qs_tgt=Qs_tgt, A=A,
        mode="ablate", device=args.device,
        batch_size=args.batch_size, max_length=args.max_length,
    )
    logits_patch = next_token_logits_with_mode(
        model, tok, prompts_test,
        src_layer=args.src_layer, tgt_layer=args.tgt_layer,
        Qs_src=Qs_src, Qs_tgt=Qs_tgt, A=A,
        mode="patch", device=args.device,
        batch_size=args.batch_size, max_length=args.max_length,
    )

    def score_logits(logits: torch.Tensor) -> Tuple[float, float]:
        """
        返回 (accuracy, mean_margin)
        margin = logp(correct) - max_{wrong} logp(wrong)
        """
        logp = F.log_softmax(logits, dim=-1)  # [N,V]
        margins = []
        correct = 0
        for i in range(logits.shape[0]):
            g = gold[i]
            if g not in letters:
                continue
            lp = {L: float(logp[i, letter_token_ids[L]].item()) for L in letters}
            pred = max(lp.items(), key=lambda kv: kv[1])[0]
            if pred == g:
                correct += 1
            wrong_best = max([lp[L] for L in letters if L != g])
            margins.append(lp[g] - wrong_best)
        acc = correct / max(len(margins), 1)
        margin = float(np.mean(margins)) if margins else float("nan")
        return acc, margin

    acc_b, mar_b = score_logits(logits_base)
    acc_a, mar_a = score_logits(logits_ablate)
    acc_p, mar_p = score_logits(logits_patch)

    df = pd.DataFrame([
        {"mode": "baseline", "accuracy": acc_b, "mean_margin": mar_b},
        {"mode": "ablate_shared@tgt", "accuracy": acc_a, "mean_margin": mar_a},
        {"mode": "patch_pred_shared@tgt", "accuracy": acc_p, "mean_margin": mar_p},
    ])
    print(df)

    out_dir = args.out_dir or os.path.dirname(args.shared_pt)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"patching_{args.task}_src{args.src_layer}_tgt{args.tgt_layer}.csv")
    df.to_csv(out_csv, index=False)
    print(f"[done] saved {out_csv}")


if __name__ == "__main__":
    main()
