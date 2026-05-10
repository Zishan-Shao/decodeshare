# -*- coding: utf-8 -*-
"""flow_utils.py

用于分析 shared 子空间与 residual(非共享/控制池) 的跨层关系的一组工具：
- greedy 生成并收集每步的 shared/residual 坐标（state & update）
- teacher-forced 收集（用于因果/干预对齐）
- 组件消融 hook（移除投影到给定基的分量）

约定
----
- layer 索引为 0-based transformer block id。
- 对于某个 block ell，在 decode step 中：
    pre  = hidden_states[ell]   (进入该 block 的 residual stream)
    post = hidden_states[ell+1] (该 block 输出 residual stream)
    delta = post - pre          (该 block 的 residual update)

特征
----
对每个层 ell，我们用两套基：
- Qs[ell] : shared basis (d x kS)
- Qr[ell] : residual/control pool basis (d x kR)

并记录：
- shared_state  = post @ Qs
- resid_state   = post @ Qr
- shared_update = (post-pre) @ Qs
- resid_update  = (post-pre) @ Qr

注意：本模块只依赖 torch/transformers，不依赖具体任务数据。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


def move_bases_to_device(
    Q_by_layer: Dict[int, torch.Tensor],
    *,
    device: str,
    dtype: torch.dtype = torch.float32,
) -> Dict[int, torch.Tensor]:
    """把 {layer: Q[d,k]} 移到 device，并转 float32（默认）。"""
    out: Dict[int, torch.Tensor] = {}
    for ell, Q in Q_by_layer.items():
        out[int(ell)] = Q.to(device=device, dtype=dtype, non_blocking=True).contiguous()
    return out


def _extract_step_features(
    hs: Tuple[torch.Tensor, ...],
    layers: Sequence[int],
    Qs: Dict[int, torch.Tensor],
    Qr: Dict[int, torch.Tensor],
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """从单个 decode step 的 hidden_states 中提取特征（torch tensors on device, float32）。"""
    shared_state: Dict[int, torch.Tensor] = {}
    resid_state: Dict[int, torch.Tensor] = {}
    shared_update: Dict[int, torch.Tensor] = {}
    resid_update: Dict[int, torch.Tensor] = {}

    # hs[i] : [B, 1, d]  (decode step)
    for ell in layers:
        pre = hs[ell][0, -1, :].float()      # [d]
        post = hs[ell + 1][0, -1, :].float() # [d]
        delta = post - pre

        Qs_ = Qs[ell]  # [d,kS]
        Qr_ = Qr[ell]  # [d,kR]

        shared_state[ell] = post @ Qs_
        resid_state[ell] = post @ Qr_
        shared_update[ell] = delta @ Qs_
        resid_update[ell] = delta @ Qr_

    return shared_state, resid_state, shared_update, resid_update


def greedy_collect_shared_residual_features(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    layers: Sequence[int],
    Qs: Dict[int, torch.Tensor],
    Qr: Dict[int, torch.Tensor],
    *,
    max_new_tokens: int = 16,
    device: str = "cuda",
    stop_on_eos: bool = True,
) -> Tuple[Dict[str, Dict[int, np.ndarray]], List[int]]:
    """Greedy 生成，并在每个 decode step 收集 shared/residual 特征。

    返回：
      features: {
        'shared_state': {ell: [T,kS]},
        'resid_state': {ell: [T,kR]},
        'shared_update': {ell: [T,kS]},
        'resid_update': {ell: [T,kR]},
      }
      gen_ids: 生成 token id 序列（长度 T；等于收集到的步数）
    """
    layers = [int(x) for x in layers]

    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attn = enc.get("attention_mask", None)
    if attn is not None:
        attn = attn.to(device)

    feats = {
        "shared_state": {ell: [] for ell in layers},
        "resid_state": {ell: [] for ell in layers},
        "shared_update": {ell: [] for ell in layers},
        "resid_update": {ell: [] for ell in layers},
    }
    gen_ids: List[int] = []

    with torch.inference_mode():
        # prefill
        out = model(
            input_ids=input_ids,
            attention_mask=attn,
            use_cache=True,
            output_hidden_states=False,
            return_dict=True,
        )
        past = out.past_key_values
        logits = out.logits[:, -1, :]  # [1,V]

        cur = torch.argmax(logits, dim=-1, keepdim=True)  # [1,1]

        for _t in range(int(max_new_tokens)):
            # 记录当前 token
            tid = int(cur.item())
            gen_ids.append(tid)
            if stop_on_eos and (tokenizer.eos_token_id is not None) and tid == int(tokenizer.eos_token_id):
                break

            out = model(
                input_ids=cur,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = out.past_key_values
            hs = out.hidden_states

            ss, rs, su, ru = _extract_step_features(hs, layers, Qs, Qr)
            for ell in layers:
                feats["shared_state"][ell].append(ss[ell].detach().cpu())
                feats["resid_state"][ell].append(rs[ell].detach().cpu())
                feats["shared_update"][ell].append(su[ell].detach().cpu())
                feats["resid_update"][ell].append(ru[ell].detach().cpu())

            # 下一 token
            logits = out.logits[:, -1, :]
            cur = torch.argmax(logits, dim=-1, keepdim=True)

    # stack to numpy
    out_np: Dict[str, Dict[int, np.ndarray]] = {}
    for key in feats:
        out_np[key] = {}
        for ell, lst in feats[key].items():
            if len(lst) == 0:
                # 没有任何 step（例如立刻 eos）
                k = int(Qs[ell].shape[1]) if "shared" in key else int(Qr[ell].shape[1])
                out_np[key][ell] = np.zeros((0, k), dtype=np.float32)
            else:
                out_np[key][ell] = torch.stack(lst, dim=0).numpy().astype(np.float32)

    return out_np, gen_ids


def greedy_generate_ids(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 16,
    device: str = "cuda",
    stop_on_eos: bool = True,
) -> List[int]:
    """只做 greedy 生成，返回生成 token ids（长度<=max_new_tokens）。"""
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attn = enc.get("attention_mask", None)
    if attn is not None:
        attn = attn.to(device)

    gen_ids: List[int] = []
    with torch.inference_mode():
        out = model(
            input_ids=input_ids,
            attention_mask=attn,
            use_cache=True,
            output_hidden_states=False,
            return_dict=True,
        )
        past = out.past_key_values
        logits = out.logits[:, -1, :]
        cur = torch.argmax(logits, dim=-1, keepdim=True)  # [1,1]

        for _t in range(int(max_new_tokens)):
            tid = int(cur.item())
            gen_ids.append(tid)
            if stop_on_eos and (tokenizer.eos_token_id is not None) and tid == int(tokenizer.eos_token_id):
                break

            out = model(
                input_ids=cur,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=False,
                return_dict=True,
            )
            past = out.past_key_values
            logits = out.logits[:, -1, :]
            cur = torch.argmax(logits, dim=-1, keepdim=True)

    return gen_ids


def teacher_forced_collect_shared_residual_features(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    continuation_ids: Sequence[int],
    layers: Sequence[int],
    Qs: Dict[int, torch.Tensor],
    Qr: Dict[int, torch.Tensor],
    *,
    device: str = "cuda",
) -> Tuple[Dict[str, Dict[int, np.ndarray]], np.ndarray]:
    """Teacher-forced：给定 continuation token ids，逐步前向并收集特征。

    返回：
      features 同 greedy_collect...
      token_logprobs: [T] 每个 continuation token 在该位置的 logprob

    注意：token i 的概率来自“上一步”的 logits：
      - token0 用 prefill logits
      - token_i (i>0) 用处理 token_{i-1} 后的 logits
    """
    layers = [int(x) for x in layers]

    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attn = enc.get("attention_mask", None)
    if attn is not None:
        attn = attn.to(device)

    feats = {
        "shared_state": {ell: [] for ell in layers},
        "resid_state": {ell: [] for ell in layers},
        "shared_update": {ell: [] for ell in layers},
        "resid_update": {ell: [] for ell in layers},
    }

    token_logprobs: List[float] = []

    with torch.inference_mode():
        # prefill
        out = model(
            input_ids=input_ids,
            attention_mask=attn,
            use_cache=True,
            output_hidden_states=False,
            return_dict=True,
        )
        past = out.past_key_values
        prev_logits = out.logits[:, -1, :].float()  # [1,V]

        for tid in continuation_ids:
            tid_int = int(tid)
            # logprob of tid under prev_logits
            lp = torch.log_softmax(prev_logits, dim=-1)[0, tid_int].item()
            token_logprobs.append(float(lp))

            cur = torch.tensor([[tid_int]], device=device)
            out = model(
                input_ids=cur,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = out.past_key_values
            hs = out.hidden_states

            ss, rs, su, ru = _extract_step_features(hs, layers, Qs, Qr)
            for ell in layers:
                feats["shared_state"][ell].append(ss[ell].detach().cpu())
                feats["resid_state"][ell].append(rs[ell].detach().cpu())
                feats["shared_update"][ell].append(su[ell].detach().cpu())
                feats["resid_update"][ell].append(ru[ell].detach().cpu())

            prev_logits = out.logits[:, -1, :].float()

    out_np: Dict[str, Dict[int, np.ndarray]] = {}
    for key in feats:
        out_np[key] = {}
        for ell, lst in feats[key].items():
            if len(lst) == 0:
                k = int(Qs[ell].shape[1]) if "shared" in key else int(Qr[ell].shape[1])
                out_np[key][ell] = np.zeros((0, k), dtype=np.float32)
            else:
                out_np[key][ell] = torch.stack(lst, dim=0).numpy().astype(np.float32)

    return out_np, np.asarray(token_logprobs, dtype=np.float32)


@dataclass
class ComponentAblationHook:
    """移除 output last-token 上投影到 Q 的分量。

    用法：
      hook = ComponentAblationHook(Q)
      h = blocks[layer].register_forward_hook(hook)
      hook.reset()  # 在每个新序列前

    仅在 seq_len==1 的 decode steps 生效；prefill 不动。
    """

    Q: torch.Tensor                 # [d,k] on device float32
    steps: Optional[Sequence[int]] = None  # 只在这些 step 生效；None => all decode steps

    step: int = 0

    def reset(self):
        self.step = 0

    def __call__(self, module, inputs, output):
        hs = output[0] if isinstance(output, (tuple, list)) else output
        if hs.dim() != 3:
            return output
        # only decode steps
        if hs.shape[1] != 1:
            return output

        t = self.step
        self.step += 1
        if (self.steps is not None) and (t not in set(map(int, self.steps))):
            return output

        x = hs[:, -1, :].float()  # [B,d]
        proj = (x @ self.Q) @ self.Q.T
        x_new = x - proj

        hs2 = hs.clone()
        hs2[:, -1, :] = x_new.to(hs2.dtype)
        if isinstance(output, (tuple, list)):
            return (hs2,) + tuple(output[1:])
        return hs2
