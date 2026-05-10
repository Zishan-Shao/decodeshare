
# -*- coding: utf-8 -*-
"""
model_utils.py

与 HuggingFace Transformers 模型交互的工具：
- 加载模型/分词器
- 找到“transformer blocks”的列表（兼容常见架构）
- 用 forward hooks 抓取指定层的 last-token hidden state（用于子空间估计）
- 逐步 decode 收集每步 hidden state（用于功能持久性）

注意：
  - layer 索引约定：0-based，指第 ell 个 transformer block。
  - output_hidden_states 的索引：hidden_states[0] 通常是 embedding 输出；
    第 ell 个 block 的输出一般在 hidden_states[ell+1]。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


# ----------------------------
# Model loading
# ----------------------------

def load_model_and_tokenizer(
    model_name_or_path: str,
    *,
    device: str = "cuda",
    dtype: str = "auto",
    cache_dir: Optional[str] = None,
) -> Tuple[torch.nn.Module, "AutoTokenizer"]:
    """
    加载 HF causal LM 与 tokenizer。

    dtype:
      - "auto": transformers 自行决定
      - "float16"/"bfloat16"/"float32"
    """
    if dtype == "auto":
        torch_dtype = "auto"
    else:
        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if dtype not in dtype_map:
            raise ValueError(f"Unknown dtype: {dtype}")
        torch_dtype = dtype_map[dtype]

    tok = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, cache_dir=cache_dir)
    # causal LM 常用 left padding 方便取 last position
    tok.padding_side = "left"
    if tok.pad_token is None:
        # 绝大多数 causal LM 没有 pad_token，通常用 eos 代替
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=None,
        cache_dir=cache_dir,
    )
    model.eval()
    model.to(device)
    return model, tok


# ----------------------------
# Locate transformer layers
# ----------------------------

def get_transformer_layers(model: torch.nn.Module) -> List[torch.nn.Module]:
    """
    尝试从不同架构中拿到 transformer block 列表。
    常见：
      - LLaMA/Mistral/Qwen2: model.model.layers
      - GPT-2: model.transformer.h
      - GPT-NeoX: model.gpt_neox.layers
      - OPT: model.model.decoder.layers
    """
    # LLaMA / Mistral / Qwen2 / Yi / etc.
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
        return list(layers)

    # OPT-like
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return list(model.model.decoder.layers)

    # GPT-2 / GPT-J / GPT-Neo style
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)

    # GPT-NeoX
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return list(model.gpt_neox.layers)

    raise RuntimeError(
        "Unsupported model architecture: cannot find transformer layers list. "
        "Please extend get_transformer_layers() for your model."
    )


def get_num_layers(model: torch.nn.Module) -> int:
    return len(get_transformer_layers(model))


def get_hidden_size(model: torch.nn.Module) -> int:
    cfg = getattr(model, "config", None)
    for attr in ["hidden_size", "n_embd", "d_model"]:
        if cfg is not None and hasattr(cfg, attr):
            return int(getattr(cfg, attr))
    # fallback：尝试从 embedding weight
    if hasattr(model, "get_input_embeddings"):
        emb = model.get_input_embeddings()
        return int(emb.embedding_dim)
    raise RuntimeError("Cannot infer hidden size.")


# ----------------------------
# Collect last-token states (for subspace estimation)
# ----------------------------

@dataclass
class _HookStore:
    layers: List[int]
    per_layer_batches: Dict[int, List[torch.Tensor]]

    def __init__(self, layers: List[int]):
        self.layers = list(layers)
        self.per_layer_batches = {ell: [] for ell in self.layers}

    def append(self, ell: int, x_last: torch.Tensor) -> None:
        # x_last: [B, d] on CPU float32 recommended
        self.per_layer_batches[ell].append(x_last)

    def finalize(self) -> Dict[int, torch.Tensor]:
        out = {}
        for ell, chunks in self.per_layer_batches.items():
            if len(chunks) == 0:
                raise RuntimeError(f"No activations captured for layer {ell}.")
            out[ell] = torch.cat(chunks, dim=0)
        return out


def collect_last_token_states_multi_layer(
    model: torch.nn.Module,
    tokenizer,
    prompts: Sequence[str],
    layers: Sequence[int],
    *,
    batch_size: int = 8,
    max_length: int = 512,
    device: str = "cuda",
    show_progress: bool = True,
) -> Dict[int, torch.Tensor]:
    """
    对一组 prompts，抓取多个层的 last-token hidden state（prefill）。

    返回 dict: layer -> X [N, d] (CPU float32)
    """
    layers = list(map(int, layers))
    blocks = get_transformer_layers(model)
    L = len(blocks)
    for ell in layers:
        if ell < 0 or ell >= L:
            raise ValueError(f"Layer {ell} out of range [0,{L-1}]")

    store = _HookStore(layers)
    hooks = []

    def make_hook(ell: int):
        def _hook(module, inputs, output):
            hs = output[0] if isinstance(output, (tuple, list)) else output
            # hs: [B, seq, d]
            x_last = hs[:, -1, :].detach().float().cpu()
            store.append(ell, x_last)
        return _hook

    for ell in layers:
        hooks.append(blocks[ell].register_forward_hook(make_hook(ell)))

    try:
        rng = range(0, len(prompts), batch_size)
        it = tqdm(rng, desc="collect last-token states", disable=not show_progress)
        with torch.inference_mode():
            for i in it:
                batch_prompts = list(prompts[i:i + batch_size])
                enc = tokenizer(
                    batch_prompts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                _ = model(**enc, use_cache=False)  # 不需要 hidden_states，靠 hook 抓
        return store.finalize()
    finally:
        for h in hooks:
            h.remove()


# ----------------------------
# Step-by-step decode trajectories (for functional persistence)
# ----------------------------

def generate_and_collect_hidden_states(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    layers: Sequence[int],
    *,
    max_new_tokens: int = 16,
    device: str = "cuda",
    do_sample: bool = False,
    temperature: float = 1.0,
    stop_on_eos: bool = True,
) -> Tuple[Dict[int, List[torch.Tensor]], List[int]]:
    """
    逐步 greedy/sample decode，并收集每个 decode step 的 hidden state（seq_len==1）。
    返回：
      traj: dict layer -> list[t] of h_t (CPU float32 tensor [d])
      gen_ids: 生成的 token ids 列表（长度 T）
    """
    layers = list(map(int, layers))
    num_layers = get_num_layers(model)
    for ell in layers:
        if ell < 0 or ell >= num_layers:
            raise ValueError(f"Layer {ell} out of range [0,{num_layers-1}]")

    # encode prompt
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attn = enc.get("attention_mask", None)
    if attn is not None:
        attn = attn.to(device)

    traj = {ell: [] for ell in layers}
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
        logits = out.logits[:, -1, :]  # [1, V]
        if do_sample:
            probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # [1,1]
        else:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)  # [1,1]

        for _t in range(int(max_new_tokens)):
            out = model(
                input_ids=next_id,  # [1,1]
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = out.past_key_values
            hs = out.hidden_states  # tuple, len = num_layers + 1 (incl embeddings)

            # record hidden states at each selected layer
            for ell in layers:
                # hidden_states[0] is embeddings; block ell is usually at ell+1
                h = hs[ell + 1][0, -1, :].detach().float().cpu()
                traj[ell].append(h)

            logits = out.logits[:, -1, :]
            if do_sample:
                probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            else:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)

            tid = int(next_id.item())
            gen_ids.append(tid)
            if stop_on_eos and (tokenizer.eos_token_id is not None) and tid == int(tokenizer.eos_token_id):
                break

    return traj, gen_ids
