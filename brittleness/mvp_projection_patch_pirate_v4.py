
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mvp_projection_patch_pirate_v4.py

A "fixed" MVP script for user-facing generation utility (Pirate style) under true KV-cache decode.

v4 improvements (vs your v0/v1 MVP):
  1) --v_mode prefill|decode (default: decode)
     - decode mode estimates steering vectors from prompt-boundary seq_len==1 calls (KV-aligned).
  2) Chat template support (auto for Llama-2-chat style models/tokenizers)
     - v estimation and evaluation prompts are formatted consistently.
  3) Pirate anchor option for v estimation
     - makes the estimated direction more lexically "pirate" so regex metrics detect it.
  4) Optional alpha sweep + early-token-only injection
     - --alphas "1,2,3" to evaluate multiple strengths
     - --inject_first_n N: inject only first N generated tokens (plus prompt-boundary call)

Outputs:
  - results.csv: per-example outputs + pirate_hits/success
  - summary.md/json: mean±std + worst-case across templates (per decoding and method)

Example:
  CUDA_VISIBLE_DEVICES=0 python mvp_projection_patch_pirate_v4.py \
    --model meta-llama/Llama-2-7b-chat-hf --device cuda --dtype fp32 \
    --layer 10 --v_mode decode \
    --alpha 3.0 --betas 0,1 \
    --basis_k 128 --calib_max_new_tokens 128 --max_new_tokens 256 \
    --batch_size 8 --do_greedy 1 --do_sample 1 --sample_seeds 1,2 \
    --inject_first_n 64 \
    --out_dir results/mvp_pirate_v4 \
    --pirate_anchor "Start your answer with 'Ahoy matey!' and include: ahoy, matey, arrr, aye, cap'n."

CUDA_VISIBLE_DEVICES=0 python mvp_projection_patch_pirate_v4.py \
  --model meta-llama/Llama-2-7b-chat-hf --device cuda --dtype fp32 \
  --layer 10 --v_mode decode --v_n 16 --v_decode_steps 16 \
  --pirate_anchor "Start your answer with 'Ahoy matey!' and include: ahoy, matey, arrr, aye, cap'n." \
  --alphas 6,10,14 \
  --inject_first_n 64 \
  --pirate_threshold 1 \
  --do_greedy 1 --do_sample 0 \
  --out_dir results/mvp_pirate_v4_fix


Notes:
  - Pirate success metric is lexicon-based and configurable.
  - This script is intentionally self-contained.

"""

import argparse
import csv
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import re

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# -----------------------------
# Defaults: prompts/templates
# -----------------------------
BASE_PROMPTS = [
    "Explain why the sky looks blue during the day.",
    "Give practical tips to improve sleep quality.",
    "Describe how a refrigerator works.",
    "Summarize the plot of a hypothetical mystery story in 5 sentences.",
    "Explain what inflation means to a high school student.",
    "Give advice for preparing for a job interview.",
    "Explain photosynthesis in simple terms.",
    "Describe the pros and cons of remote work.",
    "Provide a short guide to learning a new language effectively.",
    "Explain how vaccines help protect communities.",
    "Describe what a neural network is at a high level.",
    "Give a simple recipe for making pancakes.",
    "Explain why exercise benefits mental health.",
    "Describe how to plan a weekly schedule productively.",
    "Explain what climate change means and why it matters.",
    "Give tips for resolving conflicts in a team.",
    "Explain what debugging is and how to do it systematically.",
    "Describe the water cycle.",
    "Give a short explanation of what a database index does.",
    "Explain the difference between correlation and causation.",
    "Explain what an API is and why it is useful.",
    "Describe how GPS location is determined.",
    "Explain how a microwave heats food.",
    "Describe the basics of public-key cryptography.",
    "Explain what a compiler does.",
    "Give a brief guide to writing clear emails.",
    "Explain what overfitting is in machine learning.",
    "Describe the difference between RAM and storage.",
    "Explain what latency is and why it matters.",
    "Explain what version control is and why it matters.",
    "Explain what a cache is and why it helps.",
    "Explain the idea of supply and demand.",
    "Explain what a queue and a stack are.",
    "Explain what a probability distribution is.",
    "Explain what a hypothesis test is.",
    "Describe what an operating system does.",
    "Explain recursion with a simple example.",
    "Describe a simple approach to time management.",
    "Explain why privacy matters online.",
    "Describe how a thermostat controls temperature.",
]

TEMPLATES = [
    "{q}",
    "Please answer the following question:\n{q}",
    "I need help with this:\n{q}",
    "Give a clear explanation:\n{q}",
    "Explain it step by step:\n{q}",
    "Answer as if speaking to a beginner:\n{q}",
    "Provide a concise but complete answer:\n{q}",
    "Write your answer in two short paragraphs:\n{q}",
    "Use bullet points when helpful:\n{q}",
    "Answer in a friendly tone:\n{q}",
]


# -----------------------------
# Pirate metric (expanded lexicon)
# -----------------------------
_PIRATE_PATTERNS = [
    r"\bahoy\b",
    r"\bmatey\b",
    r"\bavast\b",
    r"\bscallywag\b",
    r"\blandlubber\b",
    r"\byo-?ho\b",
    r"\bbooty\b",
    r"\bplunder\b",
    r"\bbuccaneer\b",
    r"\bprivateer\b",
    r"\bseadog\b",
    r"\baye\b",
    r"\bcap['’]?n\b",
    r"\bshiver\s+me\s+timbers\b",
    r"\bscurvy\b",
    r"\bar{2,}\b",     # arrr
    r"\byar{1,}\b",    # yar
    r"\bme\s+heart(?:y|ies)\b",
    r"\bdead\s+men\s+tell\s+no\s+tales\b",
    r"\bwalk\s+the\s+plank\b",
]
_PIRATE_REGEXES = [re.compile(p, re.IGNORECASE) for p in _PIRATE_PATTERNS]


def pirate_hits(text: str) -> int:
    t = text.strip().lower()
    hits = 0
    for rx in _PIRATE_REGEXES:
        if rx.search(t) is not None:
            hits += 1
    return hits


# -----------------------------
# Utils: seeding, blocks, chat formatting
# -----------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def tok(tokenizer, text: str, max_len: int, use_chat: bool):
    # chat template 产物通常已经包含 special tokens，不要重复加
    return tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
        add_special_tokens=(not use_chat),
    )

def get_block(model, layer_idx: int):
    # LLaMA-like HF: model.model.layers[i]
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    # GPT-2-like: model.transformer.h[i]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer_idx]
    raise ValueError("Cannot locate transformer blocks; adapt get_block() for your model.")


def get_model_device(model) -> torch.device:
    emb = model.get_input_embeddings()
    if emb is not None and hasattr(emb, "weight"):
        return emb.weight.device
    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    return torch.device("cpu")


def supports_chat_template(tokenizer) -> bool:
    return hasattr(tokenizer, "apply_chat_template")


def should_use_chat_template(model_name: str, tokenizer, flag: str) -> bool:
    """
    flag: 'auto'|'on'|'off'
    """
    if flag == "off":
        return False
    if flag == "on":
        return supports_chat_template(tokenizer)
    # auto
    if not supports_chat_template(tokenizer):
        return False
    low = model_name.lower()
    # heuristic: llama-2 chat, instruct models
    return ("chat" in low) or ("instruct" in low) or ("assistant" in low)


def format_chat(tokenizer, user_text: str, system_text: str = "You are a helpful assistant.") -> str:
    """
    Returns a single string prompt. Works only if tokenizer supports apply_chat_template.
    """
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    # add_generation_prompt makes the assistant prefix appear (important for chat models)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# -----------------------------
# Hooks for collect / add
# -----------------------------
class CollectLastTokenHook:
    """
    Records last-token hidden states. If decode_only=True, records only when seq_len==1.
    """
    def __init__(self, decode_only: bool):
        self.decode_only = decode_only
        self.records: List[torch.Tensor] = []

    def __call__(self, module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        if not isinstance(h, torch.Tensor) or h.ndim != 3:
            return output
        if self.decode_only and h.shape[1] != 1:
            return output
        self.records.append(h[:, -1, :].detach())
        return output


class AddVectorHook:
    """
    Adds alpha*v on seq_len==1 calls. Optional early-token window.
    - inject_first_n == 0: inject for all decode calls
    - inject_first_n > 0: inject only while step < inject_first_n
      (step counts decode steps within each generation run)
    """
    def __init__(self, v: torch.Tensor, alpha: float, inject_first_n: int = 0):
        self.v = v.detach()
        self.alpha = float(alpha)
        self.inject_first_n = int(inject_first_n)
        self._cache = {}
        self.step = 0  # increments each seq_len==1 call

    def reset(self):
        self.step = 0

    def _v_on(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (device, dtype)
        if key not in self._cache:
            self._cache[key] = self.v.to(device=device, dtype=dtype)
        return self._cache[key]

    def __call__(self, module, inputs, output):
        if isinstance(output, tuple):
            h = output[0]
            rest = output[1:]
        else:
            h = output
            rest = None

        if not isinstance(h, torch.Tensor) or h.ndim != 3:
            return output

        # Only apply on KV-cached decode calls (seq_len==1)
        if h.shape[1] != 1:
            return output

        do_inject = (self.inject_first_n <= 0) or (self.step < self.inject_first_n)
        self.step += 1

        if not do_inject:
            return output

        v = self._v_on(h.device, h.dtype)
        h2 = h.clone()
        h2[:, -1, :] = h2[:, -1, :] + self.alpha * v

        if rest is None:
            return h2
        return (h2, *rest)


# -----------------------------
# Prompt construction
# -----------------------------
def make_eval_prompts(base_prompts: List[str], templates: List[str]) -> List[Tuple[int, str]]:
    """
    Returns list of (template_id, prompt_text_without_chat_wrapping)
    length = len(base_prompts) * len(templates)
    """
    out = []
    for tid, tpl in enumerate(templates):
        for q in base_prompts:
            out.append((tid, tpl.format(q=q)))
    return out


def make_pirate_instruction(anchor: str) -> str:
    # For v estimation only: strengthen lexical style signal with anchors.
    # Keep it short; chat models often follow this well.
    base = "Reply like a pirate."
    if anchor.strip():
        base += " " + anchor.strip()
    return base


def make_v_est_pair(text: str, *, anchor: str) -> Tuple[str, str]:
    """
    Returns (pirate_variant, normal_variant) user-texts (without chat wrapping).
    """
    pirate = make_pirate_instruction(anchor) + "\n\n" + text
    normal = text
    return pirate, normal


# -----------------------------
# v estimation: prefill / decode
# -----------------------------
@torch.inference_mode()
def collect_last_token_state_prefill(model, tokenizer, prompt_text: str, layer: int, max_prompt_tokens: int) -> torch.Tensor:
    device = get_model_device(model)
    block = get_block(model, layer)
    hook = CollectLastTokenHook(decode_only=False)
    handle = block.register_forward_hook(hook)
    try:
        #toks = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_prompt_tokens, add_special_tokens=True)
        toks = tok(tokenizer, prompt_text, max_prompt_tokens, use_chat=True)
        input_ids = toks["input_ids"].to(device)
        _ = model(input_ids=input_ids, use_cache=False)
        if len(hook.records) < 1:
            raise RuntimeError("No records captured in prefill.")
        return hook.records[-1].squeeze(0).float()
    finally:
        handle.remove()


@torch.inference_mode()
def collect_last_token_state_decode_prompt_boundary(model, tokenizer, prompt_text: str, layer: int, max_prompt_tokens: int) -> torch.Tensor:
    """
    Prefill T-1 tokens with cache, then feed last prompt token as seq_len==1 decode call.
    Returns last token hidden state captured on seq_len==1 call.
    """
    device = get_model_device(model)
    block = get_block(model, layer)
    hook = CollectLastTokenHook(decode_only=True)
    handle = block.register_forward_hook(hook)
    try:
        #toks = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_prompt_tokens, add_special_tokens=True)
        toks = tok(tokenizer, prompt_text, max_prompt_tokens, use_chat=True)
        input_ids = toks["input_ids"].to(device)
        T = input_ids.shape[1]
        past = None
        if T > 1:
            out_prefill = model(input_ids=input_ids[:, :-1], use_cache=True)
            past = out_prefill.past_key_values
        _ = model(input_ids=input_ids[:, -1:], past_key_values=past, use_cache=True)
        if len(hook.records) < 1:
            raise RuntimeError("No decode-only record captured.")
        return hook.records[-1].squeeze(0).float()
    finally:
        handle.remove()


@torch.inference_mode()
def estimate_v_mean_diff(
    model,
    tokenizer,
    texts: List[str],
    *,
    layer: int,
    v_mode: str,
    max_prompt_tokens: int,
    use_chat: bool,
    system_text: str,
    pirate_anchor: str,
    decode_steps: int,
) -> torch.Tensor:
    """
    Estimate steering direction v = mean(h_pirate) - mean(h_normal)
    using either prefill or decode-aligned prompt-boundary hidden states.
    """
    states_p = []
    states_n = []

    for t in texts:
        pirate_u, normal_u = make_v_est_pair(t, anchor=pirate_anchor)

        if use_chat:
            pirate_prompt = format_chat(tokenizer, pirate_u, system_text=system_text)
            normal_prompt = format_chat(tokenizer, normal_u, system_text=system_text)
        else:
            pirate_prompt = pirate_u
            normal_prompt = normal_u

        if v_mode == "prefill":
            hp = collect_last_token_state_prefill(model, tokenizer, pirate_prompt, layer, max_prompt_tokens)
            hn = collect_last_token_state_prefill(model, tokenizer, normal_prompt, layer, max_prompt_tokens)
        elif v_mode == "decode":
            # hp = collect_last_token_state_decode_prompt_boundary(model, tokenizer, pirate_prompt, layer, max_prompt_tokens)
            # hn = collect_last_token_state_decode_prompt_boundary(model, tokenizer, normal_prompt, layer, max_prompt_tokens)
            hp = collect_mean_decode_state_over_steps(
                model, tokenizer, pirate_prompt,
                layer=layer, max_prompt_tokens=max_prompt_tokens,
                use_chat=use_chat, system_text=system_text,
                decode_steps=decode_steps,
            )
            hn = collect_mean_decode_state_over_steps(
                model, tokenizer, normal_prompt,
                layer=layer, max_prompt_tokens=max_prompt_tokens,
                use_chat=use_chat, system_text=system_text,
                decode_steps=decode_steps,
            )
        else:
            raise ValueError("--v_mode must be prefill or decode")

        states_p.append(hp)
        states_n.append(hn)

    Hp = torch.stack(states_p, dim=0)
    Hn = torch.stack(states_n, dim=0)
    v = (Hp.mean(dim=0) - Hn.mean(dim=0)).float()
    v = v / (v.norm() + 1e-12)
    return v


# -----------------------------
# Basis estimation B: decode PCA from no-steer generation traces
# -----------------------------
@torch.inference_mode()
def generate_collect_decode_states(
    model,
    tokenizer,
    prompts: List[str],
    *,
    layer: int,
    max_prompt_tokens: int,
    max_new_tokens: int,
    use_chat: bool,
    system_text: str,
    temperature: float,
    top_p: float,
    seed: int,
) -> torch.Tensor:
    """
    Generate with no steering and collect seq_len==1 hidden states at the specified layer.
    Returns X: [n_states, d] float32 on CPU.
    """
    seed_everything(seed)
    device = get_model_device(model)
    block = get_block(model, layer)
    hook = CollectLastTokenHook(decode_only=True)
    handle = block.register_forward_hook(hook)
    try:
        for p in prompts:
            user_text = p
            prompt_text = format_chat(tokenizer, user_text, system_text=system_text) if use_chat else user_text

            # toks = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_prompt_tokens, add_special_tokens=True)
            toks = tok(tokenizer, prompt_text, max_prompt_tokens, use_chat=True)
            input_ids = toks["input_ids"].to(device)
            T = input_ids.shape[1]

            # Prefill all but last token to initialize cache
            past = None
            if T > 1:
                out_prefill = model(input_ids=input_ids[:, :-1], use_cache=True)
                past = out_prefill.past_key_values

            # First seq_len==1 call: last prompt token
            out = model(input_ids=input_ids[:, -1:], past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :]

            # Now decode loop for max_new_tokens
            prev = None
            for _ in range(max_new_tokens):
                if prev is None:
                    # sample from first logits
                    next_id = sample_next_token(logits, temperature=temperature, top_p=top_p)
                else:
                    out = model(input_ids=prev, past_key_values=past, use_cache=True)
                    past = out.past_key_values
                    logits = out.logits[:, -1, :]
                    next_id = sample_next_token(logits, temperature=temperature, top_p=top_p)

                prev = next_id
                # early stop if EOS
                if int(next_id.item()) == tokenizer.eos_token_id:
                    break

        if len(hook.records) == 0:
            raise RuntimeError("No decode states were recorded for basis estimation.")
        X = torch.cat([r.float().cpu() for r in hook.records], dim=0)  # [n, d]
        return X
    finally:
        handle.remove()


@torch.inference_mode()
def collect_mean_decode_state_over_steps(
    model, tokenizer, prompt_text: str,
    *, layer: int, max_prompt_tokens: int, use_chat: bool, system_text: str,
    decode_steps: int,
) -> torch.Tensor:
    device = get_model_device(model)
    block = get_block(model, layer)

    hook = CollectLastTokenHook(decode_only=True)
    handle = block.register_forward_hook(hook)
    try:
        toks = tok(tokenizer, prompt_text, max_prompt_tokens, use_chat)
        input_ids = toks["input_ids"].to(device)
        T = input_ids.shape[1]

        past = None
        if T > 1:
            out_prefill = model(input_ids=input_ids[:, :-1], use_cache=True)
            past = out_prefill.past_key_values

        # prompt-boundary (seq_len=1)
        out = model(input_ids=input_ids[:, -1:], past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :]

        # greedy decode a few steps to expose style tokens
        prev = None
        for _ in range(int(decode_steps)):
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
            if int(next_id.item()) == tokenizer.eos_token_id:
                break
            out = model(input_ids=next_id, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :]

        if len(hook.records) < 1:
            raise RuntimeError("No decode-only record captured in collect_mean_decode_state_over_steps().")

        H = torch.cat([r.float().cpu() for r in hook.records], dim=0)  # [n_steps, d]
        return H.mean(dim=0).float()
    finally:
        handle.remove()


@torch.inference_mode()
def pca_basis(X: torch.Tensor, k: int) -> torch.Tensor:
    """
    X: [n, d] float32 on CPU or GPU
    Returns B: [d, k] float32 on CPU with orthonormal columns.
    """
    X = X.float()
    n, d = X.shape
    q = int(min(k, n - 1, d))
    if q < 1:
        raise RuntimeError(f"Not enough states for PCA: n={n}, d={d}, k={k}")
    # do PCA on CPU to reduce GPU memory spikes
    Xc = X
    U, S, V = torch.pca_lowrank(Xc, q=q, center=True, niter=2)
    B = V[:, :q].contiguous()
    B, _ = torch.linalg.qr(B, mode="reduced")
    return B.cpu().float()


def project_out(B: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    v_fixed = v - B(B^T v) , B is orthonormal [d,k]
    """
    B = B.to(v.device, dtype=torch.float32)
    v32 = v.float()
    v_fixed = v32 - B @ (B.t() @ v32)
    # rescale to match ||v||
    v_fixed = v_fixed / (v_fixed.norm() + 1e-12) * (v32.norm() + 1e-12)
    return v_fixed.to(dtype=v.dtype)


def sharedness(B: torch.Tensor, v: torch.Tensor) -> float:
    B = B.to(v.device, dtype=torch.float32)
    v32 = v.float()
    return float((B.t() @ v32).norm().item() / (v32.norm().item() + 1e-12))


# -----------------------------
# Sampling helper
# -----------------------------
@torch.inference_mode()
def sample_next_token(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """
    logits: [1, V]
    returns next_id: [1,1]
    """
    if temperature <= 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / max(temperature, 1e-6)
    probs = torch.softmax(logits, dim=-1)

    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cum = torch.cumsum(sorted_probs, dim=-1)
        mask = cum > top_p
        # keep at least 1 token
        mask[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(mask, 0.0)
        sorted_probs = sorted_probs / (sorted_probs.sum(dim=-1, keepdim=True) + 1e-12)
        next_sorted = torch.multinomial(sorted_probs, num_samples=1)
        next_id = torch.gather(sorted_idx, dim=-1, index=next_sorted)
        return next_id
    else:
        return torch.multinomial(probs, num_samples=1)


# -----------------------------
# Generation evaluation
# -----------------------------
@dataclass
class EvalCfg:
    temperature: float = 0.7
    top_p: float = 0.9


@torch.inference_mode()
def generate_one(
    model,
    tokenizer,
    prompt_text: str,
    *,
    max_prompt_tokens: int,
    max_new_tokens: int,
    decoding: str,  # 'greedy'|'sample'
    seed: Optional[int],
    hook: Optional[AddVectorHook],
    layer: int,
    use_chat: bool,
    system_text: str,
    eval_cfg: EvalCfg,
) -> Tuple[str, int, bool]:
    """
    Returns: (generated_text, new_tokens, ended_by_eos)
    """
    if seed is not None:
        seed_everything(seed)

    device = get_model_device(model)
    block = get_block(model, layer)
    handle = None
    if hook is not None:
        hook.reset()
        handle = block.register_forward_hook(hook)

    try:
        # Wrap as chat if needed
        prompt = format_chat(tokenizer, prompt_text, system_text=system_text) if use_chat else prompt_text
        # toks = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_prompt_tokens, add_special_tokens=True)
        toks = tok(tokenizer, prompt, max_prompt_tokens, use_chat=True)
        input_ids = toks["input_ids"].to(device)
        T = input_ids.shape[1]

        past = None
        if T > 1:
            out_prefill = model(input_ids=input_ids[:, :-1], use_cache=True)
            past = out_prefill.past_key_values

        # prompt-boundary decode call (seq_len==1) for last prompt token
        out = model(input_ids=input_ids[:, -1:], past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :]

        gen_ids = []
        ended = False
        prev = None

        for _ in range(max_new_tokens):
            if decoding == "greedy":
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                next_id = sample_next_token(logits, temperature=eval_cfg.temperature, top_p=eval_cfg.top_p)

            tok = int(next_id.item())
            gen_ids.append(tok)

            if tok == tokenizer.eos_token_id:
                ended = True
                break

            prev = next_id
            out = model(input_ids=prev, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :]

        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        return text, len(gen_ids), ended
    finally:
        if handle is not None:
            handle.remove()


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def parse_float_list(s: str) -> List[float]:
    xs = []
    for x in s.split(","):
        x = x.strip()
        if x:
            xs.append(float(x))
    return xs


def parse_int_list(s: str) -> List[int]:
    xs = []
    for x in s.split(","):
        x = x.strip()
        if x:
            xs.append(int(x))
    return xs


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="fp32", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--device_map", type=str, default=None, help="Optional HF device_map like 'auto'.")

    ap.add_argument("--layer", type=int, default=10)

    # v estimation
    ap.add_argument("--v_mode", type=str, default="decode", choices=["prefill", "decode"])
    ap.add_argument("--v_n", type=int, default=32)
    ap.add_argument("--v_max_prompt_tokens", type=int, default=512)
    ap.add_argument("--pirate_anchor", type=str, default="Use at least two of: ahoy, matey, arrr, aye, cap'n.",
                    help="Anchor string used ONLY for v estimation to encourage lexical pirate tokens.")

    # basis
    ap.add_argument("--basis_k", type=int, default=128)
    ap.add_argument("--basis_n_prompts", type=int, default=30)
    ap.add_argument("--calib_max_new_tokens", type=int, default=128)
    ap.add_argument("--basis_max_states", type=int, default=20000)

    # eval generation
    ap.add_argument("--max_prompt_tokens", type=int, default=512)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=4)  # currently used for loop chunking
    ap.add_argument("--do_greedy", type=int, default=1)
    ap.add_argument("--do_sample", type=int, default=1)
    ap.add_argument("--sample_seeds", type=str, default="1,2")

    ap.add_argument("--alpha", type=float, default=3.0)
    ap.add_argument("--alphas", type=str, default="", help="Optional comma list; overrides --alpha if set.")
    ap.add_argument("--inject_first_n", type=int, default=0, help="Inject only first N decode steps; 0=all.")
    ap.add_argument("--n_rand", type=int, default=1)

    # metric
    ap.add_argument("--pirate_threshold", type=int, default=2)
    ap.add_argument("--v_decode_steps", type=int, default=16,
                help="In v_mode=decode, average hidden states over prompt-boundary + first N generated tokens.")

    # chat template
    ap.add_argument("--chat_template", type=str, default="auto", choices=["auto", "on", "off"])
    ap.add_argument("--system_text", type=str, default="You are a helpful assistant.")

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default="results/mvp_pirate_v4")

    args = ap.parse_args()
    seed_everything(args.seed)
    ensure_dir(args.out_dir)

    # dtype
    if args.dtype == "fp16":
        torch_dtype = torch.float16
    elif args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    # load
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kw = {"torch_dtype": torch_dtype}
    # newer HF prefers 'dtype', but keep compatibility
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch_dtype, device_map=args.device_map)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, **kw, device_map=args.device_map)

    model.eval()
    if args.device_map is None:
        model.to(args.device)

    use_chat = should_use_chat_template(args.model, tokenizer, args.chat_template)

    print(f"[Load] model={args.model} device={args.device} dtype={args.dtype} layer={args.layer} use_chat={use_chat} v_mode={args.v_mode}")
    if use_chat:
        print("[Chat] using tokenizer.apply_chat_template for prompts.")

    # choose alphas
    alphas = parse_float_list(args.alphas) if args.alphas.strip() else [float(args.alpha)]

    # prepare prompts
    base_prompts = BASE_PROMPTS
    templates = TEMPLATES
    eval_prompts = make_eval_prompts(base_prompts, templates)
    print(f"[Data] base_prompts={len(base_prompts)} templates={len(templates)} eval_prompts={len(eval_prompts)}")

    # v estimation texts: just use first v_n base prompts (content only, not templates)
    v_texts = base_prompts[: min(args.v_n, len(base_prompts))]
    print(f"[v] estimating v with n={len(v_texts)} layer={args.layer} mode={args.v_mode}")
    v = estimate_v_mean_diff(
        model, tokenizer, v_texts,
        layer=args.layer,
        v_mode=args.v_mode,
        max_prompt_tokens=args.v_max_prompt_tokens,
        use_chat=use_chat,
        system_text=args.system_text,
        pirate_anchor=args.pirate_anchor,
        decode_steps=args.v_decode_steps,
    ).to(get_model_device(model))

    v_path = os.path.join(args.out_dir, f"v_pirate_{args.v_mode}_layer{args.layer}.npy")
    np.save(v_path, v.detach().cpu().numpy())
    print(f"[v] saved {v_path} ||v||={float(v.norm().item()):.4f}")

    # basis estimation prompts: use some templates to diversify (no pirate)
    calib_prompts = []
    for i in range(min(args.basis_n_prompts, len(base_prompts))):
        tid = i % len(templates)
        calib_prompts.append(templates[tid].format(q=base_prompts[i]))

    print(f"[B] collecting decode states for PCA: n_prompts={len(calib_prompts)} max_new_tokens={args.calib_max_new_tokens} basis_k={args.basis_k}")
    X = generate_collect_decode_states(
        model, tokenizer, calib_prompts,
        layer=args.layer,
        max_prompt_tokens=args.max_prompt_tokens,
        max_new_tokens=args.calib_max_new_tokens,
        use_chat=use_chat,
        system_text=args.system_text,
        temperature=0.7,
        top_p=0.9,
        seed=args.seed + 123,
    )
    print(f"[B] collected states X shape={tuple(X.shape)}")

    # subsample states if needed
    if args.basis_max_states > 0 and X.shape[0] > args.basis_max_states:
        idx = torch.randperm(X.shape[0])[: args.basis_max_states]
        X = X[idx]
        print(f"[B] subsampled states to {tuple(X.shape)}")

    B = pca_basis(X, k=args.basis_k)
    b_path = os.path.join(args.out_dir, f"B_decode_pca_k{B.shape[1]}_layer{args.layer}.npy")
    np.save(b_path, B.numpy())
    print(f"[B] saved {b_path} shape={tuple(B.shape)}")

    print(f"[Sharedness] ||B^T v||/||v|| = {sharedness(B, v):.4f}")
    v_fixed = project_out(B, v)
    print(f"[Sharedness] ||B^T v_fixed||/||v_fixed|| = {sharedness(B, v_fixed):.4f}")

    # random control(s) energy matched
    v_norm = float(v.float().norm().item())
    rand_vs = []
    for rid in range(max(args.n_rand, 0)):
        r = torch.randn_like(v.float())
        r = r / (r.norm() + 1e-12) * v_norm
        rand_vs.append(r.to(v.dtype))

    # evaluation configs
    eval_cfg = EvalCfg()

    sample_seeds = parse_int_list(args.sample_seeds) if args.sample_seeds.strip() else [1, 2]
    methods = []  # list of dict with method_name, vector_or_none, rand_id
    methods.append({"name": "no_steer", "v": None, "rand_id": None})

    for a in alphas:
        methods.append({"name": f"v_orig_a{a:g}", "v": v, "alpha": a, "rand_id": None})
        methods.append({"name": f"v_fixed_a{a:g}", "v": v_fixed, "alpha": a, "rand_id": None})
    for rid, rv in enumerate(rand_vs):
        for a in alphas:
            methods.append({"name": f"rand{rid}_a{a:g}", "v": rv, "alpha": a, "rand_id": rid})

    # evaluate
    rows = []
    def eval_one_setting(method, decoding: str, seed: Optional[int]):
        hook = None
        if method["v"] is not None:
            hook = AddVectorHook(method["v"], alpha=float(method["alpha"]), inject_first_n=args.inject_first_n)

        for (tid, prompt_text) in eval_prompts:
            out_text, new_tokens, ended = generate_one(
                model, tokenizer, prompt_text,
                max_prompt_tokens=args.max_prompt_tokens,
                max_new_tokens=args.max_new_tokens,
                decoding=decoding,
                seed=seed,
                hook=hook,
                layer=args.layer,
                use_chat=use_chat,
                system_text=args.system_text,
                eval_cfg=eval_cfg,
            )
            hits = pirate_hits(out_text)
            succ = 1 if hits >= args.pirate_threshold else 0
            rows.append({
                "method": method["name"],
                "decoding": decoding,
                "seed": seed if seed is not None else "",
                "template_id": tid,
                "pirate_hits": hits,
                "success": succ,
                "new_tokens": new_tokens,
                "ended_by_eos": int(ended),
                "text": out_text,
            })

    if args.do_greedy:
        for m in methods:
            print(f"[Eval] method={m['name']} decoding=greedy")
            eval_one_setting(m, "greedy", None)

    if args.do_sample:
        for s in sample_seeds:
            for m in methods:
                print(f"[Eval] method={m['name']} decoding=sample seed={s}")
                eval_one_setting(m, "sample", s)

    # save csv
    out_csv = os.path.join(args.out_dir, "results.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) if rows else []
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[Save] wrote {out_csv} rows={len(rows)}")

    # summarize across templates: mean±std and worst-case (template min) per decoding and method
    def summarize(decoding: str):
        # method -> list of template success rates
        summary = {}
        for m in methods:
            name = m["name"]
            # gather by template
            t_rates = []
            for tid in range(len(templates)):
                subset = [r for r in rows if r["decoding"] == decoding and r["method"] == name and int(r["template_id"]) == tid]
                if len(subset) == 0:
                    continue
                rate = sum(int(r["success"]) for r in subset) / len(subset)
                t_rates.append(rate)
            if len(t_rates) == 0:
                continue
            mean = float(np.mean(t_rates))
            std = float(np.std(t_rates, ddof=0))
            worst = float(np.min(t_rates))
            summary[name] = {"mean": mean, "std": std, "worst": worst, "per_template": t_rates}
        return summary

    summ_g = summarize("greedy") if args.do_greedy else {}
    summ_s = summarize("sample") if args.do_sample else {}

    # Write summary.md
    md_path = os.path.join(args.out_dir, "summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# MVP v4: Pirate steering projection repair\n\n")
        f.write(f"- model: {args.model}\n")
        f.write(f"- layer: {args.layer}\n")
        f.write(f"- v_mode: {args.v_mode}\n")
        f.write(f"- pirate_threshold: {args.pirate_threshold}\n")
        f.write(f"- inject_first_n: {args.inject_first_n}\n")
        f.write(f"- basis_k: {B.shape[1]}\n")
        f.write(f"- sharedness(v): {sharedness(B, v):.6f}\n")
        f.write(f"- sharedness(v_fixed): {sharedness(B, v_fixed):.6f}\n\n")

        # Table
        f.write("| Method | Greedy mean ± std | Greedy worst | Sample mean ± std | Sample worst |\n")
        f.write("| --- | --- | --- | --- | --- |\n")

        all_method_names = [m["name"] for m in methods]
        # show a nice ordering: no_steer, v_orig..., v_fixed..., rand...
        def sort_key(name: str):
            if name == "no_steer":
                return (0, name)
            if name.startswith("v_orig"):
                return (1, name)
            if name.startswith("v_fixed"):
                return (2, name)
            if name.startswith("rand"):
                return (3, name)
            return (9, name)

        for name in sorted(all_method_names, key=sort_key):
            g = summ_g.get(name, None)
            s = summ_s.get(name, None)
            g_str = f"{g['mean']:.3f} ± {g['std']:.3f}" if g else ""
            g_w = f"{g['worst']:.3f}" if g else ""
            s_str = f"{s['mean']:.3f} ± {s['std']:.3f}" if s else ""
            s_w = f"{s['worst']:.3f}" if s else ""
            f.write(f"| {name} | {g_str} | {g_w} | {s_str} | {s_w} |\n")

    print(f"[Save] wrote {md_path}")

    # Write summary.json
    js_path = os.path.join(args.out_dir, "summary.json")
    payload = {
        "config": vars(args),
        "sharedness_v": sharedness(B, v),
        "sharedness_v_fixed": sharedness(B, v_fixed),
        "greedy": summ_g,
        "sample": summ_s,
    }
    with open(js_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[Save] wrote {js_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
