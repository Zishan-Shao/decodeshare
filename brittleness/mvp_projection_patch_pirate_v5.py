#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mvp_projection_patch_pirate_v5.py

v5 = v4 + debug/fail-fast + smoke test + incremental CSV + chat tokenize fix
核心能力：
- chat template 下 tokenize 修复（use_chat 时不重复 add_special_tokens）
- v_mode=decode 下支持 v_decode_steps：平均 prompt-boundary + 前N步decode隐藏态来估 v
- probe_injection：算完 v 后立刻检查 next-token logits 是否被注入推动（10秒内发现问题）
- fail-fast：探针失败直接退出
- smoke test：短生成 + alpha sweep（默认 sample），可自动选择最佳 alpha
- 评估：子集控制、增量写CSV、早停、遇到一次成功就停



# A) 只做 v + probe（最快定位问题）
CUDA_VISIBLE_DEVICES=0 python mvp_projection_patch_pirate_v5.py \
  --layer 28 --dtype fp32 \
  --v_mode decode --v_decode_steps 16 --v_n 8 \
  --probe_alpha 12 --fail_fast 1 --debug_only 1 \
  --out_dir results/mvp_pirate_v5_probe


# B) smoke test + 自动选 alpha + 小子集 eval（最快把“故事”跑出来）
CUDA_VISIBLE_DEVICES=0 python mvp_projection_patch_pirate_v5.py \
  --layer 28 --dtype fp32 \
  --v_mode decode --v_decode_steps 16 --v_n 16 \
  --pirate_threshold 1 \
  --temperature 1.0 --top_p 0.95 \
  --smoke_test 1 --smoke_decoding sample --smoke_alphas 10,20,40,60,80,120 \
  --auto_use_best_alpha 1 --stop_on_smoke_success 1 \
  --do_greedy 0 --do_sample 1 --sample_seeds 1 \
  --inject_first_n 64 \
  --eval_n_base 2 --eval_n_templates 2 --max_eval_prompts 10 \
  --early_abort_after 40 \
  --out_dir results/mvp_pirate_v5_smoke_then_eval


"""

import argparse
import csv
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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
    if flag == "off":
        return False
    if flag == "on":
        return supports_chat_template(tokenizer)
    if not supports_chat_template(tokenizer):
        return False
    low = model_name.lower()
    return ("chat" in low) or ("instruct" in low) or ("assistant" in low)


def format_chat(tokenizer, user_text: str, system_text: str = "You are a helpful assistant.") -> str:
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def tok(tokenizer, text: str, max_len: int, use_chat: bool):
    """
    关键修复：chat template 产物通常已经包含 special tokens，
    tokenize 时不要再 add_special_tokens=True，避免边界 token 乱掉。
    """
    return tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
        add_special_tokens=(not use_chat),
    )


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def parse_float_list(s: str) -> List[float]:
    xs: List[float] = []
    for x in s.split(","):
        x = x.strip()
        if x:
            xs.append(float(x))
    return xs


def parse_int_list(s: str) -> List[int]:
    xs: List[int] = []
    for x in s.split(","):
        x = x.strip()
        if x:
            xs.append(int(x))
    return xs


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
    Debug counters:
      - n_seen_seq1: number of seq_len==1 calls observed
      - n_applied : number of times injection actually applied
    """
    def __init__(self, v: torch.Tensor, alpha: float, inject_first_n: int = 0):
        self.v = v.detach()
        self.alpha = float(alpha)
        self.inject_first_n = int(inject_first_n)
        self._cache = {}
        self.step = 0
        self.n_seen_seq1 = 0
        self.n_applied = 0

    def reset(self):
        self.step = 0
        self.n_seen_seq1 = 0
        self.n_applied = 0

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

        if h.shape[1] != 1:
            return output

        self.n_seen_seq1 += 1

        do_inject = (self.inject_first_n <= 0) or (self.step < self.inject_first_n)
        self.step += 1
        if not do_inject:
            return output

        self.n_applied += 1

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
    out: List[Tuple[int, str]] = []
    for tid, tpl in enumerate(templates):
        for q in base_prompts:
            out.append((tid, tpl.format(q=q)))
    return out


def make_pirate_instruction(anchor: str) -> str:
    base = "Reply like a pirate."
    if anchor.strip():
        base += " " + anchor.strip()
    return base


def make_v_est_pair(text: str, *, anchor: str) -> Tuple[str, str]:
    pirate = make_pirate_instruction(anchor) + "\n\n" + text
    normal = text
    return pirate, normal


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
        mask[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(mask, 0.0)
        sorted_probs = sorted_probs / (sorted_probs.sum(dim=-1, keepdim=True) + 1e-12)
        next_sorted = torch.multinomial(sorted_probs, num_samples=1)
        next_id = torch.gather(sorted_idx, dim=-1, index=next_sorted)
        return next_id
    else:
        return torch.multinomial(probs, num_samples=1)


# -----------------------------
# v estimation
# -----------------------------
@torch.inference_mode()
def collect_last_token_state_prefill(model, tokenizer, prompt_text: str, layer: int, max_prompt_tokens: int, use_chat: bool) -> torch.Tensor:
    device = get_model_device(model)
    block = get_block(model, layer)
    hook = CollectLastTokenHook(decode_only=False)
    handle = block.register_forward_hook(hook)
    try:
        toks = tok(tokenizer, prompt_text, max_prompt_tokens, use_chat)
        input_ids = toks["input_ids"].to(device)
        _ = model(input_ids=input_ids, use_cache=False)
        if len(hook.records) < 1:
            raise RuntimeError("No records captured in prefill.")
        return hook.records[-1].squeeze(0).float()
    finally:
        handle.remove()


@torch.inference_mode()
def collect_last_token_state_decode_prompt_boundary(model, tokenizer, prompt_text: str, layer: int, max_prompt_tokens: int, use_chat: bool) -> torch.Tensor:
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
        _ = model(input_ids=input_ids[:, -1:], past_key_values=past, use_cache=True)
        if len(hook.records) < 1:
            raise RuntimeError("No decode-only record captured.")
        return hook.records[-1].squeeze(0).float()
    finally:
        handle.remove()


@torch.inference_mode()
def collect_mean_decode_state_over_steps(
    model, tokenizer, prompt_text: str,
    *, layer: int, max_prompt_tokens: int, use_chat: bool,
    decode_steps: int,
) -> torch.Tensor:
    """
    v_mode=decode 改进：平均 prompt-boundary(seq_len=1) + 前 N 个 decode steps 的 hidden state。
    这样更对齐“风格词(ahoy/matey/arrr)”出现阶段。
    """
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

        # greedy decode a few steps (只用于估 v)
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
def estimate_v_mean_diff(
    model,
    tokenizer,
    texts: List[str],
    *,
    layer: int,
    v_mode: str,
    v_decode_steps: int,
    max_prompt_tokens: int,
    use_chat: bool,
    system_text: str,
    pirate_anchor: str,
) -> torch.Tensor:
    states_p: List[torch.Tensor] = []
    states_n: List[torch.Tensor] = []

    for t in texts:
        pirate_u, normal_u = make_v_est_pair(t, anchor=pirate_anchor)

        if use_chat:
            pirate_prompt = format_chat(tokenizer, pirate_u, system_text=system_text)
            normal_prompt = format_chat(tokenizer, normal_u, system_text=system_text)
        else:
            pirate_prompt = pirate_u
            normal_prompt = normal_u

        if v_mode == "prefill":
            hp = collect_last_token_state_prefill(model, tokenizer, pirate_prompt, layer, max_prompt_tokens, use_chat)
            hn = collect_last_token_state_prefill(model, tokenizer, normal_prompt, layer, max_prompt_tokens, use_chat)
        elif v_mode == "decode":
            if v_decode_steps and v_decode_steps > 0:
                hp = collect_mean_decode_state_over_steps(
                    model, tokenizer, pirate_prompt,
                    layer=layer, max_prompt_tokens=max_prompt_tokens, use_chat=use_chat,
                    decode_steps=v_decode_steps,
                )
                hn = collect_mean_decode_state_over_steps(
                    model, tokenizer, normal_prompt,
                    layer=layer, max_prompt_tokens=max_prompt_tokens, use_chat=use_chat,
                    decode_steps=v_decode_steps,
                )
            else:
                hp = collect_last_token_state_decode_prompt_boundary(model, tokenizer, pirate_prompt, layer, max_prompt_tokens, use_chat)
                hn = collect_last_token_state_decode_prompt_boundary(model, tokenizer, normal_prompt, layer, max_prompt_tokens, use_chat)
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
    seed_everything(seed)
    device = get_model_device(model)
    block = get_block(model, layer)
    hook = CollectLastTokenHook(decode_only=True)
    handle = block.register_forward_hook(hook)
    try:
        for p in prompts:
            prompt_text = format_chat(tokenizer, p, system_text=system_text) if use_chat else p
            toks = tok(tokenizer, prompt_text, max_prompt_tokens, use_chat)
            input_ids = toks["input_ids"].to(device)
            T = input_ids.shape[1]

            past = None
            if T > 1:
                out_prefill = model(input_ids=input_ids[:, :-1], use_cache=True)
                past = out_prefill.past_key_values

            # prompt-boundary
            out = model(input_ids=input_ids[:, -1:], past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :]

            prev = None
            for _ in range(max_new_tokens):
                if prev is None:
                    next_id = sample_next_token(logits, temperature=temperature, top_p=top_p)
                else:
                    out = model(input_ids=prev, past_key_values=past, use_cache=True)
                    past = out.past_key_values
                    logits = out.logits[:, -1, :]
                    next_id = sample_next_token(logits, temperature=temperature, top_p=top_p)

                prev = next_id
                if int(next_id.item()) == tokenizer.eos_token_id:
                    break

        if len(hook.records) == 0:
            raise RuntimeError("No decode states were recorded for basis estimation.")
        X = torch.cat([r.float().cpu() for r in hook.records], dim=0)
        return X
    finally:
        handle.remove()


@torch.inference_mode()
def pca_basis(X: torch.Tensor, k: int) -> torch.Tensor:
    X = X.float()
    n, d = X.shape
    q = int(min(k, n - 1, d))
    if q < 1:
        raise RuntimeError(f"Not enough states for PCA: n={n}, d={d}, k={k}")
    U, S, V = torch.pca_lowrank(X, q=q, center=True, niter=2)
    B = V[:, :q].contiguous()
    B, _ = torch.linalg.qr(B, mode="reduced")
    return B.cpu().float()


def project_out(B: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    B = B.to(v.device, dtype=torch.float32)
    v32 = v.float()
    v_fixed = v32 - B @ (B.t() @ v32)
    v_fixed = v_fixed / (v_fixed.norm() + 1e-12) * (v32.norm() + 1e-12)
    return v_fixed.to(dtype=v.dtype)


def sharedness(B: torch.Tensor, v: torch.Tensor) -> float:
    B = B.to(v.device, dtype=torch.float32)
    v32 = v.float()
    return float((B.t() @ v32).norm().item() / (v32.norm().item() + 1e-12))


# -----------------------------
# Debug probe: logits delta on next token
# -----------------------------
@torch.inference_mode()
def probe_injection(
    model, tokenizer, user_text: str,
    *, layer: int, max_prompt_tokens: int, use_chat: bool, system_text: str,
    v: torch.Tensor, alpha: float, inject_first_n: int,
    topk: int = 15,
):
    device = get_model_device(model)
    block = get_block(model, layer)

    prompt = format_chat(tokenizer, user_text, system_text=system_text) if use_chat else user_text
    toks = tok(tokenizer, prompt, max_prompt_tokens, use_chat)
    input_ids = toks["input_ids"].to(device)
    T = input_ids.shape[1]

    past = None
    if T > 1:
        out_prefill = model(input_ids=input_ids[:, :-1], use_cache=True)
        past = out_prefill.past_key_values

    # logits without hook
    out0 = model(input_ids=input_ids[:, -1:], past_key_values=past, use_cache=True)
    logits0 = out0.logits[0, -1, :].float().cpu()

    # logits with hook
    hook = AddVectorHook(v, alpha=alpha, inject_first_n=inject_first_n)
    hook.reset()
    h = block.register_forward_hook(hook)
    out1 = model(input_ids=input_ids[:, -1:], past_key_values=past, use_cache=True)
    h.remove()
    logits1 = out1.logits[0, -1, :].float().cpu()

    d = logits1 - logits0
    delta_norm = float(d.norm().item())

    print(f"[Probe] hook n_seen_seq1={hook.n_seen_seq1} n_applied={hook.n_applied} ||Δlogits||={delta_norm:.4f}")

    topv, topi = torch.topk(d, k=min(topk, d.numel()))
    print("[Probe] Top Δlogits tokens:")
    for dv, tid in zip(topv.tolist(), topi.tolist()):
        s = tokenizer.decode([tid]).replace("\n", "\\n")
        print(f"  Δ={dv:+.3f}  id={tid}  tok={repr(s)}")

    # Multi-token candidates: print per-token delta + sum(delta) as a rough proxy
    cands = [" ahoy", "Ahoy", " matey", "Matey", " arrr", "Arrr", " aye", "Aye", " cap'n", " Cap'n"]
    print("[Probe] Pirate candidates (multi-token; show per-token Δlogit and sum):")
    pirate_sum_deltas: List[float] = []
    for s in cands:
        ids = tokenizer.encode(s, add_special_tokens=False)
        deltas = [float(d[i].item()) for i in ids]
        pirate_sum_deltas.append(float(sum(deltas)))
        decoded = [tokenizer.decode([i]).replace("\n", "\\n") for i in ids]
        pairs = ", ".join([f"{repr(tok)}:{dl:+.3f}" for tok, dl in zip(decoded, deltas)])
        print(f"  {repr(s)} ids={ids}  sumΔ={sum(deltas):+.3f}  parts=[{pairs}]")

    return {
        "n_seen_seq1": hook.n_seen_seq1,
        "n_applied": hook.n_applied,
        "delta_norm": delta_norm,
        "pirate_sum_deltas": pirate_sum_deltas,
    }


# -----------------------------
# Generation evaluation
# -----------------------------
@dataclass
class EvalCfg:
    temperature: float = 0.9
    top_p: float = 0.95


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
) -> Tuple[str, int, bool, Optional[Dict[str, int]]]:
    if seed is not None:
        seed_everything(seed)

    device = get_model_device(model)
    block = get_block(model, layer)
    handle = None
    if hook is not None:
        hook.reset()
        handle = block.register_forward_hook(hook)

    try:
        prompt = format_chat(tokenizer, prompt_text, system_text=system_text) if use_chat else prompt_text
        toks = tok(tokenizer, prompt, max_prompt_tokens, use_chat)
        input_ids = toks["input_ids"].to(device)
        T = input_ids.shape[1]

        past = None
        if T > 1:
            out_prefill = model(input_ids=input_ids[:, :-1], use_cache=True)
            past = out_prefill.past_key_values

        # prompt-boundary decode call
        out = model(input_ids=input_ids[:, -1:], past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :]

        gen_ids = []
        ended = False

        for _ in range(max_new_tokens):
            if decoding == "greedy":
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                next_id = sample_next_token(logits, temperature=eval_cfg.temperature, top_p=eval_cfg.top_p)

            tok_id = int(next_id.item())
            gen_ids.append(tok_id)

            if tok_id == tokenizer.eos_token_id:
                ended = True
                break

            out = model(input_ids=next_id, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :]

        text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        hook_stats = None
        if hook is not None:
            hook_stats = {"n_seen_seq1": hook.n_seen_seq1, "n_applied": hook.n_applied}

        return text, len(gen_ids), ended, hook_stats
    finally:
        if handle is not None:
            handle.remove()


@torch.inference_mode()
def smoke_test_alphas(
    model, tokenizer,
    prompts: List[str],
    *,
    layer: int,
    max_prompt_tokens: int,
    max_new_tokens: int,
    use_chat: bool,
    system_text: str,
    v: torch.Tensor,
    alphas: List[float],
    inject_first_n: int,
    decoding: str,
    seed: int,
    pirate_threshold: int,
    eval_cfg: EvalCfg,
    stop_on_success: bool = True,
):
    best = None  # (hits, alpha, preview)
    print(f"[Smoke] decoding={decoding} seed={seed} max_new_tokens={max_new_tokens} alphas={alphas}")
    for a in alphas:
        hook = AddVectorHook(v, alpha=float(a), inject_first_n=inject_first_n)
        max_hits = 0
        best_preview = ""
        for p in prompts:
            out_text, new_tokens, ended, hook_stats = generate_one(
                model, tokenizer, p,
                max_prompt_tokens=max_prompt_tokens,
                max_new_tokens=max_new_tokens,
                decoding=decoding,
                seed=seed,
                hook=hook,
                layer=layer,
                use_chat=use_chat,
                system_text=system_text,
                eval_cfg=eval_cfg,
            )
            hits = pirate_hits(out_text)
            max_hits = max(max_hits, hits)
            if hits > 0 and not best_preview:
                best_preview = out_text.replace("\n", " ")[:160]

        print(f"[Smoke] alpha={a:g}  max_hits={max_hits}  preview={best_preview}")

        if best is None or max_hits > best[0]:
            best = (max_hits, a, best_preview)

        if stop_on_success and max_hits >= pirate_threshold:
            print(f"[Smoke] SUCCESS at alpha={a:g} (max_hits={max_hits} >= threshold={pirate_threshold})")
            return best

    print(f"[Smoke] Best so far: hits={best[0]} alpha={best[1]:g} preview={best[2]}")
    return best


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="fp32", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--device_map", type=str, default=None)

    ap.add_argument("--layer", type=int, default=10)

    # v estimation
    ap.add_argument("--v_mode", type=str, default="decode", choices=["prefill", "decode"])
    ap.add_argument("--v_n", type=int, default=32)
    ap.add_argument("--v_max_prompt_tokens", type=int, default=512)
    ap.add_argument("--v_decode_steps", type=int, default=16,
                    help="For v_mode=decode, average states over prompt-boundary + first N decode steps. Set 0 to disable.")
    ap.add_argument("--pirate_anchor", type=str,
                    default="Start your answer with 'Ahoy matey!' and include: ahoy, matey, arrr, aye, cap'n.",
                    help="Anchor string used ONLY for v estimation.")

    # basis
    ap.add_argument("--basis_k", type=int, default=128)
    ap.add_argument("--basis_n_prompts", type=int, default=30)
    ap.add_argument("--calib_max_new_tokens", type=int, default=128)
    ap.add_argument("--basis_max_states", type=int, default=20000)

    # eval generation
    ap.add_argument("--max_prompt_tokens", type=int, default=512)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--do_greedy", type=int, default=1)
    ap.add_argument("--do_sample", type=int, default=1)
    ap.add_argument("--sample_seeds", type=str, default="1,2")

    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)

    ap.add_argument("--alpha", type=float, default=3.0)
    ap.add_argument("--alphas", type=str, default="", help="Optional comma list; overrides --alpha if set.")
    ap.add_argument("--inject_first_n", type=int, default=0, help="Inject only first N decode steps; 0=all.")
    ap.add_argument("--n_rand", type=int, default=1)

    # metric
    ap.add_argument("--pirate_threshold", type=int, default=2)

    # chat template
    ap.add_argument("--chat_template", type=str, default="auto", choices=["auto", "on", "off"])
    ap.add_argument("--system_text", type=str, default="You are a helpful assistant.")

    # debug / fail-fast
    ap.add_argument("--fail_fast", type=int, default=1)
    ap.add_argument("--debug_only", type=int, default=0)
    ap.add_argument("--probe_alpha", type=float, default=12.0)
    ap.add_argument("--probe_text", type=str, default="Explain why the sky looks blue during the day.")
    ap.add_argument("--probe_min_delta_norm", type=float, default=0.10)

    # smoke test
    ap.add_argument("--smoke_test", type=int, default=1)
    ap.add_argument("--smoke_alphas", type=str, default="6,10,14,20,30,40,60,80,120")
    ap.add_argument("--smoke_decoding", type=str, default="sample", choices=["greedy", "sample"])
    ap.add_argument("--smoke_seed", type=int, default=1)
    ap.add_argument("--smoke_new_tokens", type=int, default=64)
    ap.add_argument("--smoke_n_prompts", type=int, default=2)
    ap.add_argument("--stop_on_smoke_success", type=int, default=1)
    ap.add_argument("--auto_use_best_alpha", type=int, default=1)

    # fast eval subset
    ap.add_argument("--eval_n_base", type=int, default=0, help="0=all, else only first N base prompts")
    ap.add_argument("--eval_n_templates", type=int, default=0, help="0=all, else only first N templates")
    ap.add_argument("--max_eval_prompts", type=int, default=0, help="0=all, else cap total eval prompts")

    # incremental CSV + early stop
    ap.add_argument("--flush_every", type=int, default=20)
    ap.add_argument("--print_hit_examples", type=int, default=1)
    ap.add_argument("--early_abort_after", type=int, default=0, help="0=off; else check after this many rows per method/decoding")
    ap.add_argument("--early_abort_if_all_zero", type=int, default=1)
    ap.add_argument("--stop_on_first_success", type=int, default=0)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default="results/mvp_pirate_v5")

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

    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch_dtype, device_map=args.device_map)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch_dtype, device_map=args.device_map)

    model.eval()
    if args.device_map is None:
        model.to(args.device)

    use_chat = should_use_chat_template(args.model, tokenizer, args.chat_template)

    print(f"[Load] model={args.model} device={args.device} dtype={args.dtype} layer={args.layer} use_chat={use_chat} v_mode={args.v_mode} v_decode_steps={args.v_decode_steps}")
    if use_chat:
        print("[Chat] using tokenizer.apply_chat_template (tokenize with add_special_tokens=False).")

    # choose alphas
    alphas = parse_float_list(args.alphas) if args.alphas.strip() else [float(args.alpha)]

    # eval prompts (possibly subset)
    base_prompts = BASE_PROMPTS[:args.eval_n_base] if args.eval_n_base > 0 else BASE_PROMPTS
    templates = TEMPLATES[:args.eval_n_templates] if args.eval_n_templates > 0 else TEMPLATES
    eval_prompts = make_eval_prompts(base_prompts, templates)
    if args.max_eval_prompts > 0:
        eval_prompts = eval_prompts[:args.max_eval_prompts]

    print(f"[Data] base_prompts={len(base_prompts)} templates={len(templates)} eval_prompts={len(eval_prompts)}")

    # v estimation texts (固定从 BASE_PROMPTS 取前 v_n 个，避免子集影响 v)
    v_texts = BASE_PROMPTS[: min(args.v_n, len(BASE_PROMPTS))]
    print(f"[v] estimating v with n={len(v_texts)} layer={args.layer} mode={args.v_mode}")
    v = estimate_v_mean_diff(
        model, tokenizer, v_texts,
        layer=args.layer,
        v_mode=args.v_mode,
        v_decode_steps=args.v_decode_steps,
        max_prompt_tokens=args.v_max_prompt_tokens,
        use_chat=use_chat,
        system_text=args.system_text,
        pirate_anchor=args.pirate_anchor,
    ).to(get_model_device(model))

    v_path = os.path.join(args.out_dir, f"v_pirate_{args.v_mode}_layer{args.layer}.npy")
    np.save(v_path, v.detach().cpu().numpy())
    print(f"[v] saved {v_path} ||v||={float(v.norm().item()):.4f}")

    # eval cfg (sampling)
    eval_cfg = EvalCfg(temperature=float(args.temperature), top_p=float(args.top_p))

    # probe (fail-fast)
    print("[Sanity] probing injection effect on next-token logits...")
    probe = probe_injection(
        model, tokenizer, args.probe_text,
        layer=args.layer,
        max_prompt_tokens=args.max_prompt_tokens,
        use_chat=use_chat,
        system_text=args.system_text,
        v=v,
        alpha=args.probe_alpha,
        inject_first_n=max(64, args.inject_first_n),
        topk=15,
    )

    if args.fail_fast:
        if probe["n_applied"] <= 0:
            print("[FailFast] hook did not apply any injection (n_applied=0). Exiting.")
            raise SystemExit(2)
        if probe["delta_norm"] < args.probe_min_delta_norm:
            print(f"[FailFast] ||Δlogits|| too small ({probe['delta_norm']:.4f} < {args.probe_min_delta_norm:.4f}). Exiting.")
            raise SystemExit(2)

    if args.debug_only:
        print("[DebugOnly] Done (v + probe). Exiting.")
        return

    # smoke test (short alpha sweep)
    if args.smoke_test:
        smoke_alphas = parse_float_list(args.smoke_alphas)
        smoke_prompts = [TEMPLATES[0].format(q=BASE_PROMPTS[i]) for i in range(min(args.smoke_n_prompts, len(BASE_PROMPTS)))]
        best = smoke_test_alphas(
            model, tokenizer, smoke_prompts,
            layer=args.layer,
            max_prompt_tokens=args.max_prompt_tokens,
            max_new_tokens=args.smoke_new_tokens,
            use_chat=use_chat,
            system_text=args.system_text,
            v=v,
            alphas=smoke_alphas,
            inject_first_n=max(32, args.inject_first_n),
            decoding=args.smoke_decoding,
            seed=args.smoke_seed,
            pirate_threshold=args.pirate_threshold,
            eval_cfg=eval_cfg,
            stop_on_success=bool(args.stop_on_smoke_success),
        )

        if args.fail_fast and (best is None or best[0] <= 0):
            print("[FailFast] Smoke test still produced 0 pirate hits for all alphas. Exiting.")
            raise SystemExit(2)

        if args.auto_use_best_alpha and best is not None:
            alphas = [float(best[1])]
            print(f"[AutoAlpha] Using best alpha={alphas[0]:g} for full eval.")

    # basis estimation prompts
    calib_prompts: List[str] = []
    for i in range(min(args.basis_n_prompts, len(BASE_PROMPTS))):
        tid = i % len(TEMPLATES)
        calib_prompts.append(TEMPLATES[tid].format(q=BASE_PROMPTS[i]))

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

    # random control(s)
    v_norm = float(v.float().norm().item())
    rand_vs = []
    for rid in range(max(args.n_rand, 0)):
        r = torch.randn_like(v.float())
        r = r / (r.norm() + 1e-12) * v_norm
        rand_vs.append(r.to(v.dtype))

    # methods
    methods = [{"name": "no_steer", "v": None, "rand_id": None}]
    for a in alphas:
        methods.append({"name": f"v_orig_a{a:g}", "v": v, "alpha": a, "rand_id": None})
        methods.append({"name": f"v_fixed_a{a:g}", "v": v_fixed, "alpha": a, "rand_id": None})
    for rid, rv in enumerate(rand_vs):
        for a in alphas:
            methods.append({"name": f"rand{rid}_a{a:g}", "v": rv, "alpha": a, "rand_id": rid})

    sample_seeds = parse_int_list(args.sample_seeds) if args.sample_seeds.strip() else [1, 2]

    # incremental CSV
    out_csv = os.path.join(args.out_dir, "results.csv")
    f_csv = open(out_csv, "w", newline="", encoding="utf-8")
    fieldnames = [
        "method", "decoding", "seed", "template_id",
        "pirate_hits", "success", "new_tokens", "ended_by_eos",
        "text", "hook_n_seen_seq1", "hook_n_applied"
    ]
    w = csv.DictWriter(f_csv, fieldnames=fieldnames)
    w.writeheader()
    f_csv.flush()

    rows: List[Dict] = []

    def record_row(r: Dict):
        rows.append(r)
        w.writerow(r)
        if len(rows) % args.flush_every == 0:
            f_csv.flush()

    def maybe_early_abort(method_name: str, decoding: str):
        if args.early_abort_after <= 0 or not args.early_abort_if_all_zero:
            return
        subset = [r for r in rows if r["method"] == method_name and r["decoding"] == decoding]
        if len(subset) >= args.early_abort_after:
            if max(int(r["pirate_hits"]) for r in subset) == 0:
                print(f"[EarlyAbort] method={method_name} decoding={decoding} still all pirate_hits=0 after {len(subset)} rows. Exiting.")
                raise SystemExit(3)

    def eval_one_setting(method, decoding: str, seed: Optional[int]):
        hook = None
        if method["v"] is not None:
            hook = AddVectorHook(method["v"], alpha=float(method["alpha"]), inject_first_n=args.inject_first_n)

        for (tid, prompt_text) in eval_prompts:
            out_text, new_tokens, ended, hook_stats = generate_one(
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

            hn_seen = hook_stats["n_seen_seq1"] if hook_stats else ""
            hn_appl = hook_stats["n_applied"] if hook_stats else ""

            row = {
                "method": method["name"],
                "decoding": decoding,
                "seed": seed if seed is not None else "",
                "template_id": tid,
                "pirate_hits": hits,
                "success": succ,
                "new_tokens": new_tokens,
                "ended_by_eos": int(ended),
                "text": out_text,
                "hook_n_seen_seq1": hn_seen,
                "hook_n_applied": hn_appl,
            }
            record_row(row)

            if args.print_hit_examples and hits > 0:
                preview = out_text.replace("\n", " ")[:160]
                print(f"[HIT] method={method['name']} dec={decoding} seed={seed} tid={tid} hits={hits} :: {preview}")

            if args.stop_on_first_success and succ == 1:
                print(f"[StopOnFirstSuccess] method={method['name']} decoding={decoding} seed={seed} tid={tid}")
                raise SystemExit(0)

            maybe_early_abort(method["name"], decoding)

    # Evaluate + ensure csv closed
    try:
        if args.do_greedy:
            for m in methods:
                print(f"[Eval] method={m['name']} decoding=greedy")
                eval_one_setting(m, "greedy", None)

        if args.do_sample:
            for s in sample_seeds:
                for m in methods:
                    print(f"[Eval] method={m['name']} decoding=sample seed={s}")
                    eval_one_setting(m, "sample", s)

    finally:
        f_csv.flush()
        f_csv.close()
        print(f"[Save] wrote {out_csv} rows={len(rows)}")

    # summarize across templates
    def summarize(decoding: str):
        summary: Dict[str, Dict] = {}
        for m in methods:
            name = m["name"]
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

    md_path = os.path.join(args.out_dir, "summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# MVP v5: Pirate steering debug + smoke test\n\n")
        f.write(f"- model: {args.model}\n")
        f.write(f"- layer: {args.layer}\n")
        f.write(f"- v_mode: {args.v_mode}\n")
        f.write(f"- v_decode_steps: {args.v_decode_steps}\n")
        f.write(f"- pirate_threshold: {args.pirate_threshold}\n")
        f.write(f"- inject_first_n: {args.inject_first_n}\n")
        f.write(f"- temperature/top_p: {args.temperature}/{args.top_p}\n")
        f.write(f"- basis_k: {B.shape[1]}\n")
        f.write(f"- sharedness(v): {sharedness(B, v):.6f}\n")
        f.write(f"- sharedness(v_fixed): {sharedness(B, v_fixed):.6f}\n\n")

        f.write("| Method | Greedy mean ± std | Greedy worst | Sample mean ± std | Sample worst |\n")
        f.write("| --- | --- | --- | --- | --- |\n")

        all_method_names = [m["name"] for m in methods]

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

    js_path = os.path.join(args.out_dir, "summary.json")
    payload = {
        "config": vars(args),
        "probe": probe,
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
