#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mvp_projection_patch_pirate.py (clean + solid shared-space experiment)

What it does:
  - Estimate style direction v (pirate vs normal) at a chosen layer (KV-cache aligned).
  - Collect decode hidden states on no_steer traces, build PCA basis B_full.
  - Sweep basis_ks, compute sharedness(v) and v_fixed_k = v - B_k(B_k^T v).
  - Evaluate: no_steer, v_orig, v_fixed_k for each k, rand controls
    under greedy and/or sampling.
  - Save:
      * results.csv (per-example, includes generated text)
      * sharedness.csv
      * summary_stats.csv (aggregated metrics)
      * summary.md
      * summary.tex (\\input{examples.tex})
      * examples.tex (representative examples for paper)

Stability improvements:
  - Smoke test uses multiple seeds and a softer threshold by default to reliably find alpha.
  - If smoke fails to hit threshold for all alphas, it still picks best-alpha and continues (no fail-fast).
  - No probe / no fail-fast / no token-level debug prints.

# example:
CUDA_VISIBLE_DEVICES=0 python mvp_projection_patch_pirate.py \
  --layer 28 --dtype fp32 \
  --v_mode decode --v_decode_steps 16 --v_n 16 \
  --basis_ks 16,32,64,128 \
  --pirate_threshold 2 \
  --temperature 0.9 --top_p 0.9 \
  --smoke_test 1 --smoke_decoding sample --smoke_alphas 30,40,45,50,55,60 \
  --inject_start_step 1 --inject_first_n 24 \
  --do_greedy 1 --do_sample 1 --sample_seeds 1,2,3 \
  --eval_n_base 20 --eval_n_templates 5 \
  --out_dir results/sharedspace_solid_llama2_7b_chat

"""

import argparse
import csv
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

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
# Pirate metric (regex lexicon)
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
    return sum(1 for rx in _PIRATE_REGEXES if rx.search(t) is not None)


# -----------------------------
# Utils
# -----------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def parse_int_list_allow_empty(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    return parse_int_list(s)


def get_block(model, layer_idx: int):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
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


def should_use_chat_template(model_name: str, tokenizer) -> bool:
    if not supports_chat_template(tokenizer):
        return False
    low = model_name.lower()
    return ("chat" in low) or ("instruct" in low) or ("assistant" in low)


def format_chat(tokenizer, user_text: str, system_text: str) -> str:
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def tok(tokenizer, text: str, max_len: int, use_chat: bool):
    # Key fix: chat template text already includes special tokens
    return tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
        add_special_tokens=(not use_chat),
    )


# -----------------------------
# Hooks
# -----------------------------
class CollectLastTokenHook:
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
    Inject alpha*v on seq_len==1 calls.

    inject_start_step:
      0 = include prompt-boundary step
      1 = skip boundary injection (often makes text less 'glued')
    inject_first_n:
      number of decode steps after start to inject; 0 = all
    """
    def __init__(self, v: torch.Tensor, alpha: float, inject_first_n: int, inject_start_step: int):
        self.v = v.detach()
        self.alpha = float(alpha)
        self.inject_first_n = int(inject_first_n)
        self.inject_start_step = int(inject_start_step)
        self._cache: Dict[Tuple[torch.device, torch.dtype], torch.Tensor] = {}
        self.step = 0

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
        if h.shape[1] != 1:
            return output

        within_window = (self.inject_first_n <= 0) or (self.step < self.inject_start_step + self.inject_first_n)
        do_inject = (self.step >= self.inject_start_step) and within_window
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
def make_eval_items(base_prompts: List[str], templates: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    pid = 0
    for tid, tpl in enumerate(templates):
        for bid, q in enumerate(base_prompts):
            out.append({
                "prompt_id": pid,
                "template_id": tid,
                "base_id": bid,
                "base_prompt": q,
                "prompt_text": tpl.format(q=q),
            })
            pid += 1
    return out


def make_pirate_instruction(anchor: str) -> str:
    base = "Reply like a pirate."
    if anchor.strip():
        base += " " + anchor.strip()
    return base


def make_v_est_pair(text: str, anchor: str) -> Tuple[str, str]:
    pirate = make_pirate_instruction(anchor) + "\n\n" + text
    normal = text
    return pirate, normal


# -----------------------------
# Sampling
# -----------------------------
@torch.inference_mode()
def sample_next_token(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
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

    return torch.multinomial(probs, num_samples=1)


# -----------------------------
# v estimation (KV-aligned decode)
# -----------------------------
@torch.inference_mode()
def collect_mean_decode_state_over_steps(
    model, tokenizer, prompt_text: str,
    *, layer: int, max_prompt_tokens: int, use_chat: bool,
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

        out = model(input_ids=input_ids[:, -1:], past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :]

        for _ in range(int(decode_steps)):
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
            if int(next_id.item()) == tokenizer.eos_token_id:
                break
            out = model(input_ids=next_id, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :]

        H = torch.cat([r.float().cpu() for r in hook.records], dim=0)  # [n, d]
        return H.mean(dim=0).float()
    finally:
        handle.remove()


@torch.inference_mode()
def estimate_v_mean_diff(
    model, tokenizer, texts: List[str],
    *, layer: int, v_decode_steps: int,
    max_prompt_tokens: int, use_chat: bool, system_text: str, pirate_anchor: str
) -> torch.Tensor:
    states_p: List[torch.Tensor] = []
    states_n: List[torch.Tensor] = []

    for t in texts:
        pirate_u, normal_u = make_v_est_pair(t, pirate_anchor)
        if use_chat:
            pirate_prompt = format_chat(tokenizer, pirate_u, system_text)
            normal_prompt = format_chat(tokenizer, normal_u, system_text)
        else:
            pirate_prompt = pirate_u
            normal_prompt = normal_u

        hp = collect_mean_decode_state_over_steps(
            model, tokenizer, pirate_prompt,
            layer=layer, max_prompt_tokens=max_prompt_tokens, use_chat=use_chat,
            decode_steps=max(1, int(v_decode_steps)),
        )
        hn = collect_mean_decode_state_over_steps(
            model, tokenizer, normal_prompt,
            layer=layer, max_prompt_tokens=max_prompt_tokens, use_chat=use_chat,
            decode_steps=max(1, int(v_decode_steps)),
        )
        states_p.append(hp)
        states_n.append(hn)

    Hp = torch.stack(states_p, dim=0)
    Hn = torch.stack(states_n, dim=0)
    v = (Hp.mean(dim=0) - Hn.mean(dim=0)).float()
    v = v / (v.norm() + 1e-12)
    return v


# -----------------------------
# Basis estimation from decode traces
# -----------------------------
@torch.inference_mode()
def generate_collect_decode_states(
    model, tokenizer, prompts: List[str],
    *, layer: int, max_prompt_tokens: int, max_new_tokens: int,
    use_chat: bool, system_text: str, temperature: float, top_p: float, seed: int
) -> torch.Tensor:
    seed_everything(seed)
    device = get_model_device(model)
    block = get_block(model, layer)
    hook = CollectLastTokenHook(decode_only=True)
    handle = block.register_forward_hook(hook)
    try:
        for p in prompts:
            prompt_text = format_chat(tokenizer, p, system_text) if use_chat else p
            toks = tok(tokenizer, prompt_text, max_prompt_tokens, use_chat)
            input_ids = toks["input_ids"].to(device)
            T = input_ids.shape[1]

            past = None
            if T > 1:
                out_prefill = model(input_ids=input_ids[:, :-1], use_cache=True)
                past = out_prefill.past_key_values

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

        X = torch.cat([r.float().cpu() for r in hook.records], dim=0)
        return X
    finally:
        handle.remove()


@torch.inference_mode()
def pca_basis_full(X: torch.Tensor, q: int) -> torch.Tensor:
    X = X.float()
    n, d = X.shape
    q_eff = int(min(q, n - 1, d))
    if q_eff < 1:
        raise RuntimeError(f"Not enough states for PCA: n={n}, d={d}, q={q}")
    _, _, V = torch.pca_lowrank(X, q=q_eff, center=True, niter=2)
    B = V[:, :q_eff].contiguous()
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
# Generation / evaluation
# -----------------------------
@dataclass
class EvalCfg:
    temperature: float
    top_p: float


@torch.inference_mode()
def generate_one(
    model, tokenizer, prompt_text: str,
    *, max_prompt_tokens: int, max_new_tokens: int,
    decoding: str, seed: Optional[int],
    hook: Optional[AddVectorHook],
    layer: int, use_chat: bool, system_text: str,
    eval_cfg: EvalCfg
) -> Tuple[str, int, bool]:
    if seed is not None:
        seed_everything(seed)

    device = get_model_device(model)
    block = get_block(model, layer)
    handle = None
    if hook is not None:
        hook.reset()
        handle = block.register_forward_hook(hook)

    try:
        prompt = format_chat(tokenizer, prompt_text, system_text) if use_chat else prompt_text
        toks = tok(tokenizer, prompt, max_prompt_tokens, use_chat)
        input_ids = toks["input_ids"].to(device)
        T = input_ids.shape[1]

        past = None
        if T > 1:
            out_prefill = model(input_ids=input_ids[:, :-1], use_cache=True)
            past = out_prefill.past_key_values

        out = model(input_ids=input_ids[:, -1:], past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :]

        gen_ids: List[int] = []
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
        return text, len(gen_ids), ended
    finally:
        if handle is not None:
            handle.remove()


# -----------------------------
# Smoke test (robust alpha selection)
# -----------------------------
@torch.inference_mode()
def smoke_select_alpha(
    model, tokenizer,
    smoke_items: List[str],
    *, layer: int, use_chat: bool, system_text: str,
    max_prompt_tokens: int, max_new_tokens: int,
    v: torch.Tensor,
    alphas: List[float],
    inject_first_n: int, inject_start_step: int,
    decoding: str, seeds: List[int],
    threshold: int,
    eval_cfg: EvalCfg,
) -> Tuple[float, int]:
    """
    Returns (best_alpha, best_hits).
    - If any alpha reaches threshold, choose the smallest alpha that reaches it.
    - Otherwise choose alpha with max_hits (no failure).
    """
    best_alpha = alphas[0]
    best_hits = -1

    # we prefer smaller alpha if it meets threshold
    for a in alphas:
        hook = AddVectorHook(v, alpha=a, inject_first_n=inject_first_n, inject_start_step=inject_start_step)
        max_hits = 0
        for sd in seeds:
            for p in smoke_items:
                out_text, _, _ = generate_one(
                    model, tokenizer, p,
                    max_prompt_tokens=max_prompt_tokens,
                    max_new_tokens=max_new_tokens,
                    decoding=decoding,
                    seed=sd,
                    hook=hook,
                    layer=layer,
                    use_chat=use_chat,
                    system_text=system_text,
                    eval_cfg=eval_cfg,
                )
                max_hits = max(max_hits, pirate_hits(out_text))
                if max_hits >= threshold:
                    return a, max_hits  # early success
        if max_hits > best_hits:
            best_hits = max_hits
            best_alpha = a

    return best_alpha, best_hits


# -----------------------------
# Reporting helpers
# -----------------------------
def mean_std(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    return float(np.mean(xs)), float(np.std(xs, ddof=0))


def summarize_by_template(rows: List[Dict[str, Any]], method: str, decoding: str, templates_n: int) -> Dict[str, Any]:
    t_rates, t_hits = [], []
    for tid in range(templates_n):
        subset = [r for r in rows if r["method"] == method and r["decoding"] == decoding and int(r["template_id"]) == tid]
        if not subset:
            continue
        t_rates.append(sum(int(r["success"]) for r in subset) / len(subset))
        t_hits.append(sum(int(r["pirate_hits"]) for r in subset) / len(subset))
    if not t_rates:
        return {}
    ms, ss = mean_std(t_rates)
    mh, sh = mean_std(t_hits)
    return {
        "mean_success": ms, "std_success": ss, "worst_success": float(np.min(t_rates)),
        "mean_hits": mh, "std_hits": sh, "worst_hits": float(np.min(t_hits)),
    }


def group_key(r: Dict[str, Any]) -> Tuple:
    return (r["prompt_id"], r["template_id"], r["base_id"], r["decoding"], r["seed"])


def pick_representative_examples(
    rows: List[Dict[str, Any]],
    *,
    method_no: str,
    method_orig: str,
    method_fixed_kmax: str,
    want_greedy_and_sample: bool,
    max_examples_each: int,
) -> Dict[str, List[Any]]:
    idx: Dict[Tuple[str, Tuple], Dict[str, Any]] = {(r["method"], group_key(r)): r for r in rows}
    keys = sorted({group_key(r) for r in rows})

    def bundle_for(k: Tuple, methods: List[str]) -> Optional[Dict[str, Any]]:
        ref = None
        for m in methods:
            rr = idx.get((m, k))
            if rr is not None:
                ref = rr
                break
        if ref is None:
            return None
        b = {
            "prompt_text": ref["prompt_text"],
            "decoding": ref["decoding"],
            "seed": ref["seed"],
            "outputs": {m: idx.get((m, k)) for m in methods if idx.get((m, k)) is not None},
        }
        return b

    cats = {
        "steering_works": [],
        "projection_removes": [],
        "fixed_still_hits": [],
        "decoding_brittleness": [],
    }

    for k in keys:
        r_o = idx.get((method_orig, k))
        r_n = idx.get((method_no, k))
        if r_o and r_n and int(r_o["success"]) == 1 and int(r_n["success"]) == 0:
            b = bundle_for(k, [method_no, method_orig, method_fixed_kmax])
            if b:
                cats["steering_works"].append(b)

    for k in keys:
        r_o = idx.get((method_orig, k))
        r_f = idx.get((method_fixed_kmax, k))
        if r_o and r_f and int(r_o["success"]) == 1 and int(r_f["success"]) == 0:
            b = bundle_for(k, [method_no, method_orig, method_fixed_kmax])
            if b:
                cats["projection_removes"].append(b)

    for k in keys:
        r_f = idx.get((method_fixed_kmax, k))
        if r_f and int(r_f["success"]) == 1:
            b = bundle_for(k, [method_no, method_orig, method_fixed_kmax])
            if b:
                cats["fixed_still_hits"].append(b)

    if want_greedy_and_sample:
        by_core: Dict[Tuple[int, int, int, Any], Dict[str, Tuple]] = {}
        for r in rows:
            core = (r["prompt_id"], r["template_id"], r["base_id"], r["seed"])
            by_core.setdefault(core, {})
            by_core[core][r["decoding"]] = group_key(r)

        for _, dmap in by_core.items():
            kg = dmap.get("greedy")
            ks = dmap.get("sample")
            if kg and ks:
                rg = idx.get((method_orig, kg))
                rs = idx.get((method_orig, ks))
                if rg and rs and int(rg["success"]) == 0 and int(rs["success"]) == 1:
                    bg = bundle_for(kg, [method_no, method_orig, method_fixed_kmax])
                    bs = bundle_for(ks, [method_no, method_orig, method_fixed_kmax])
                    if bg and bs:
                        cats["decoding_brittleness"].append({"greedy": bg, "sample": bs})

    # sort for representativeness
    def score(b: Dict[str, Any], m: str) -> int:
        rr = b["outputs"].get(m)
        return int(rr["pirate_hits"]) if rr else 0

    cats["steering_works"].sort(key=lambda b: score(b, method_orig), reverse=True)
    cats["projection_removes"].sort(key=lambda b: score(b, method_orig), reverse=True)
    cats["fixed_still_hits"].sort(key=lambda b: score(b, method_fixed_kmax), reverse=True)

    for k in list(cats.keys()):
        cats[k] = cats[k][:max_examples_each]

    return cats


def latex_escape(s: str) -> str:
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in s)


def tex_quote_block(text: str, max_chars: int) -> str:
    t = text.strip()
    if len(t) > max_chars:
        t = t[:max_chars] + "…"
    return "\\begin{quote}\\small\n" + latex_escape(t) + "\n\\end{quote}\n"


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="fp32", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--device_map", type=str, default=None)

    ap.add_argument("--layer", type=int, default=28)

    # v estimation
    ap.add_argument("--v_mode", type=str, default="decode", choices=["decode"])  # keep simple
    ap.add_argument("--v_n", type=int, default=16)
    ap.add_argument("--v_max_prompt_tokens", type=int, default=512)
    ap.add_argument("--v_decode_steps", type=int, default=16)
    ap.add_argument("--pirate_anchor", type=str,
                    default="Start your answer with 'Ahoy matey!' and include at least two of: ahoy, matey, arrr, aye.")

    # basis sweep
    ap.add_argument("--basis_ks", type=str, default="16,32,64,128")
    ap.add_argument("--basis_n_prompts", type=int, default=30)
    ap.add_argument("--calib_max_new_tokens", type=int, default=128)
    ap.add_argument("--basis_max_states", type=int, default=20000)

    # eval prompts subset
    ap.add_argument("--eval_n_base", type=int, default=0)
    ap.add_argument("--eval_n_templates", type=int, default=0)

    # generation / decoding
    ap.add_argument("--max_prompt_tokens", type=int, default=512)
    ap.add_argument("--max_new_tokens", type=int, default=192)
    ap.add_argument("--do_greedy", type=int, default=1)
    ap.add_argument("--do_sample", type=int, default=1)
    ap.add_argument("--sample_seeds", type=str, default="1,2,3")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_p", type=float, default=0.9)

    # alpha selection
    ap.add_argument("--alpha", type=float, default=50.0)
    ap.add_argument("--alphas", type=str, default="")
    ap.add_argument("--smoke_test", type=int, default=1)
    ap.add_argument("--smoke_alphas", type=str, default="30,40,45,50,55,60")
    ap.add_argument("--smoke_decoding", type=str, default="sample", choices=["sample", "greedy"])
    ap.add_argument("--smoke_seeds", type=str, default="1,2,3")
    ap.add_argument("--smoke_new_tokens", type=int, default=64)
    ap.add_argument("--smoke_n_base", type=int, default=4)
    ap.add_argument("--smoke_n_templates", type=int, default=2)
    ap.add_argument("--smoke_threshold", type=int, default=-1,
                    help="If -1, use max(1, pirate_threshold-1) for robust alpha selection.")

    # injection window
    ap.add_argument("--inject_start_step", type=int, default=1)
    ap.add_argument("--inject_first_n", type=int, default=24)

    # metric threshold
    ap.add_argument("--pirate_threshold", type=int, default=2)

    # controls + saving
    ap.add_argument("--n_rand", type=int, default=1)
    ap.add_argument("--save_text", type=int, default=1)

    # output / examples
    ap.add_argument("--examples_per_category", type=int, default=2)
    ap.add_argument("--example_max_chars", type=int, default=700)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default="results/sharedspace_solid")

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

    # load model/tokenizer
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

    use_chat = should_use_chat_template(args.model, tokenizer)
    system_text = "You are a helpful assistant."

    print(f"[Load] model={args.model} layer={args.layer} dtype={args.dtype} use_chat={use_chat}")

    # eval prompts
    base_prompts = BASE_PROMPTS[:args.eval_n_base] if args.eval_n_base > 0 else BASE_PROMPTS
    templates = TEMPLATES[:args.eval_n_templates] if args.eval_n_templates > 0 else TEMPLATES
    eval_items = make_eval_items(base_prompts, templates)
    print(f"[Data] base_prompts={len(base_prompts)} templates={len(templates)} eval_prompts={len(eval_items)}")

    # estimate v
    v_texts = BASE_PROMPTS[: min(args.v_n, len(BASE_PROMPTS))]
    v = estimate_v_mean_diff(
        model, tokenizer, v_texts,
        layer=args.layer,
        v_decode_steps=args.v_decode_steps,
        max_prompt_tokens=args.v_max_prompt_tokens,
        use_chat=use_chat,
        system_text=system_text,
        pirate_anchor=args.pirate_anchor,
    ).to(get_model_device(model))
    np.save(os.path.join(args.out_dir, f"v_pirate_decode_layer{args.layer}.npy"), v.detach().cpu().numpy())

    # alpha selection
    alphas = parse_float_list(args.alphas) if args.alphas.strip() else [float(args.alpha)]
    eval_cfg = EvalCfg(temperature=float(args.temperature), top_p=float(args.top_p))

    if args.smoke_test:
        smoke_alphas = parse_float_list(args.smoke_alphas)
        smoke_seeds = parse_int_list(args.smoke_seeds)
        smoke_threshold = args.smoke_threshold if args.smoke_threshold >= 0 else max(1, args.pirate_threshold - 1)

        smoke_base = BASE_PROMPTS[: min(args.smoke_n_base, len(BASE_PROMPTS))]
        smoke_tpls = TEMPLATES[: min(args.smoke_n_templates, len(TEMPLATES))]
        smoke_items = [tpl.format(q=q) for tpl in smoke_tpls for q in smoke_base]

        best_alpha, best_hits = smoke_select_alpha(
            model, tokenizer, smoke_items,
            layer=args.layer,
            use_chat=use_chat,
            system_text=system_text,
            max_prompt_tokens=args.max_prompt_tokens,
            max_new_tokens=args.smoke_new_tokens,
            v=v,
            alphas=smoke_alphas,
            inject_first_n=args.inject_first_n,
            inject_start_step=args.inject_start_step,
            decoding=args.smoke_decoding,
            seeds=smoke_seeds,
            threshold=smoke_threshold,
            eval_cfg=eval_cfg,
        )
        alphas = [best_alpha]
        print(f"[Alpha] selected alpha={best_alpha:g} (smoke max_hits={best_hits}, smoke_threshold={smoke_threshold})")

    # basis ks
    basis_ks = parse_int_list_allow_empty(args.basis_ks)
    if not basis_ks:
        basis_ks = [128]
    basis_ks = sorted(set(int(k) for k in basis_ks if int(k) > 0))
    max_k = max(basis_ks)

    # calib prompts for PCA basis
    calib_prompts: List[str] = []
    for i in range(min(args.basis_n_prompts, len(BASE_PROMPTS))):
        tid = i % len(TEMPLATES)
        calib_prompts.append(TEMPLATES[tid].format(q=BASE_PROMPTS[i]))

    X = generate_collect_decode_states(
        model, tokenizer, calib_prompts,
        layer=args.layer,
        max_prompt_tokens=args.max_prompt_tokens,
        max_new_tokens=args.calib_max_new_tokens,
        use_chat=use_chat,
        system_text=system_text,
        temperature=0.7,
        top_p=0.9,
        seed=args.seed + 123,
    )
    if args.basis_max_states > 0 and X.shape[0] > args.basis_max_states:
        idx = torch.randperm(X.shape[0])[: args.basis_max_states]
        X = X[idx]

    B_full = pca_basis_full(X, q=max_k)
    q_eff = B_full.shape[1]
    if q_eff < max_k:
        basis_ks = [k for k in basis_ks if k <= q_eff]

    np.save(os.path.join(args.out_dir, f"B_decode_pca_q{q_eff}_layer{args.layer}.npy"), B_full.numpy())

    # v_fixed per k + sharedness
    fixed_vectors: Dict[int, torch.Tensor] = {}
    shared_rows: List[Dict[str, Any]] = []
    for k in basis_ks:
        Bk = B_full[:, :k]
        sh_v = sharedness(Bk, v)
        v_fixed = project_out(Bk, v)
        sh_vf = sharedness(Bk, v_fixed)
        fixed_vectors[k] = v_fixed.to(get_model_device(model))
        shared_rows.append({"basis_k": k, "sharedness_v": sh_v, "sharedness_v_fixed": sh_vf})
        np.save(os.path.join(args.out_dir, f"v_fixed_k{k}_layer{args.layer}.npy"), v_fixed.detach().cpu().numpy())

    with open(os.path.join(args.out_dir, "sharedness.json"), "w", encoding="utf-8") as f:
        json.dump(shared_rows, f, indent=2)

    with open(os.path.join(args.out_dir, "sharedness.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["basis_k", "sharedness_v", "sharedness_v_fixed"])
        w.writeheader()
        for r in shared_rows:
            w.writerow(r)

    # rand controls
    v_norm = float(v.float().norm().item())
    rand_vs: List[torch.Tensor] = []
    for _ in range(max(args.n_rand, 0)):
        r = torch.randn_like(v.float())
        r = r / (r.norm() + 1e-12) * v_norm
        rand_vs.append(r.to(v.dtype).to(get_model_device(model)))

    # methods
    alpha0 = alphas[0]
    methods: List[Dict[str, Any]] = [{"name": "no_steer", "kind": "baseline", "basis_k": "", "v": None}]
    methods.append({"name": f"v_orig_a{alpha0:g}", "kind": "v_orig", "basis_k": "", "v": v, "alpha": alpha0})
    for k in basis_ks:
        methods.append({"name": f"v_fixed_k{k}_a{alpha0:g}", "kind": "v_fixed", "basis_k": k, "v": fixed_vectors[k], "alpha": alpha0})
    for rid, rv in enumerate(rand_vs):
        methods.append({"name": f"rand{rid}_a{alpha0:g}", "kind": "rand", "basis_k": "", "v": rv, "alpha": alpha0})

    # decodings + seeds
    decodings: List[str] = []
    if args.do_greedy:
        decodings.append("greedy")
    if args.do_sample:
        decodings.append("sample")
    sample_seeds = parse_int_list(args.sample_seeds) if args.sample_seeds.strip() else [1]

    decoding_seeds: List[Tuple[str, Any]] = []
    for d in decodings:
        if d == "greedy":
            decoding_seeds.append((d, ""))  # deterministic
        else:
            for s in sample_seeds:
                decoding_seeds.append((d, s))

    # results.csv
    results_csv = os.path.join(args.out_dir, "results.csv")
    rows: List[Dict[str, Any]] = []
    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "method", "kind", "basis_k", "decoding", "seed",
            "prompt_id", "template_id", "base_id",
            "pirate_hits", "success", "new_tokens", "ended_by_eos",
            "base_prompt", "prompt_text", "text"
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for (dec, sd) in decoding_seeds:
            for m in methods:
                hook = None
                if m["v"] is not None:
                    hook = AddVectorHook(
                        m["v"], alpha=float(m["alpha"]),
                        inject_first_n=args.inject_first_n,
                        inject_start_step=args.inject_start_step,
                    )

                for item in eval_items:
                    out_text, new_tokens, ended = generate_one(
                        model, tokenizer, item["prompt_text"],
                        max_prompt_tokens=args.max_prompt_tokens,
                        max_new_tokens=args.max_new_tokens,
                        decoding=dec,
                        seed=(None if sd == "" else int(sd)),
                        hook=hook,
                        layer=args.layer,
                        use_chat=use_chat,
                        system_text=system_text,
                        eval_cfg=eval_cfg,
                    )

                    hits = pirate_hits(out_text)
                    succ = 1 if hits >= args.pirate_threshold else 0

                    r = {
                        "method": m["name"],
                        "kind": m["kind"],
                        "basis_k": m["basis_k"],
                        "decoding": dec,
                        "seed": sd,
                        "prompt_id": item["prompt_id"],
                        "template_id": item["template_id"],
                        "base_id": item["base_id"],
                        "pirate_hits": hits,
                        "success": succ,
                        "new_tokens": new_tokens,
                        "ended_by_eos": int(ended),
                        "base_prompt": item["base_prompt"],
                        "prompt_text": item["prompt_text"],
                        "text": out_text if args.save_text else "",
                    }
                    rows.append(r)
                    w.writerow(r)

    # summary_stats.csv
    templates_n = len(templates)
    summary_rows: List[Dict[str, Any]] = []
    for dec in decodings:
        for m in methods:
            s = summarize_by_template(rows, m["name"], dec, templates_n)
            if not s:
                continue
            summary_rows.append({
                "method": m["name"],
                "kind": m["kind"],
                "basis_k": m["basis_k"],
                "decoding": dec,
                "mean_success": s["mean_success"],
                "std_success": s["std_success"],
                "worst_success": s["worst_success"],
                "mean_hits": s["mean_hits"],
                "std_hits": s["std_hits"],
                "worst_hits": s["worst_hits"],
            })

    summary_stats_csv = os.path.join(args.out_dir, "summary_stats.csv")
    with open(summary_stats_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["method","kind","basis_k","decoding","mean_success","std_success","worst_success","mean_hits","std_hits","worst_hits"]
        )
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    # representative examples (use max-k fixed)
    kmax = max(basis_ks) if basis_ks else None
    method_no = "no_steer"
    method_orig = f"v_orig_a{alpha0:g}"
    method_fixed_kmax = f"v_fixed_k{kmax}_a{alpha0:g}" if kmax is not None else method_orig

    cats = pick_representative_examples(
        rows,
        method_no=method_no,
        method_orig=method_orig,
        method_fixed_kmax=method_fixed_kmax,
        want_greedy_and_sample=("greedy" in decodings and "sample" in decodings),
        max_examples_each=int(args.examples_per_category),
    )

    # examples.tex
    examples_tex = os.path.join(args.out_dir, "examples.tex")
    with open(examples_tex, "w", encoding="utf-8") as f:
        f.write("% Auto-generated representative examples\n\n")

        def write_bundle(title: str, b: Dict[str, Any]):
            f.write(f"\\subsection*{{{latex_escape(title)}}}\n")
            f.write(f"\\textbf{{Prompt}}: {latex_escape(b['prompt_text'])}\\\\\n")
            f.write(f"\\textbf{{Decoding}}: {latex_escape(str(b['decoding']))}, \\textbf{{Seed}}: {latex_escape(str(b['seed']))}\\\\\n\n")
            for mn in [method_no, method_orig, method_fixed_kmax]:
                rr = b["outputs"].get(mn)
                if rr is None:
                    continue
                f.write(f"\\textbf{{{latex_escape(mn)}}} (hits={rr['pirate_hits']}, success={rr['success']}):\n")
                f.write(tex_quote_block(rr.get("text",""), int(args.example_max_chars)))
                f.write("\n")

        if cats["steering_works"]:
            f.write("\\section*{Representative Examples}\n")
            f.write("\\subsection*{Steering works (v\\_orig succeeds, no\\_steer fails)}\n")
            for i, b in enumerate(cats["steering_works"], 1):
                write_bundle(f"Steering works #{i}", b)

        if cats["projection_removes"]:
            f.write("\\subsection*{Projection removes style (v\\_orig succeeds, v\\_fixed fails)}\n")
            for i, b in enumerate(cats["projection_removes"], 1):
                write_bundle(f"Projection removes #{i}", b)

        if cats["fixed_still_hits"]:
            f.write("\\subsection*{Repair is imperfect (v\\_fixed still hits)}\n")
            for i, b in enumerate(cats["fixed_still_hits"], 1):
                write_bundle(f"v\\_fixed still hits #{i}", b)

        if cats["decoding_brittleness"]:
            f.write("\\subsection*{Decoding brittleness (greedy fails, sample succeeds)}\n")
            for i, pair in enumerate(cats["decoding_brittleness"], 1):
                bg = pair["greedy"]
                bs = pair["sample"]
                f.write(f"\\subsection*{{Decoding brittleness #{i}}}\n")
                f.write(f"\\textbf{{Prompt}}: {latex_escape(bg['prompt_text'])}\\\\\n\n")
                f.write("\\textbf{Greedy}\\\\\n")
                for mn in [method_no, method_orig, method_fixed_kmax]:
                    rr = bg["outputs"].get(mn)
                    if rr is None:
                        continue
                    f.write(f"\\textbf{{{latex_escape(mn)}}} (hits={rr['pirate_hits']}, success={rr['success']}):\n")
                    f.write(tex_quote_block(rr.get("text",""), int(args.example_max_chars)))
                f.write("\n\\textbf{Sample}\\\\\n")
                for mn in [method_no, method_orig, method_fixed_kmax]:
                    rr = bs["outputs"].get(mn)
                    if rr is None:
                        continue
                    f.write(f"\\textbf{{{latex_escape(mn)}}} (hits={rr['pirate_hits']}, success={rr['success']}):\n")
                    f.write(tex_quote_block(rr.get("text",""), int(args.example_max_chars)))
                f.write("\n")

    # summary.md
    md_path = os.path.join(args.out_dir, "summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Shared-space experiment summary\n\n")
        f.write(f"- model: {args.model}\n")
        f.write(f"- layer: {args.layer}\n")
        f.write(f"- v_decode_steps: {args.v_decode_steps}\n")
        f.write(f"- alpha used: {alpha0}\n")
        f.write(f"- inject_start_step: {args.inject_start_step}\n")
        f.write(f"- inject_first_n: {args.inject_first_n}\n")
        f.write(f"- pirate_threshold: {args.pirate_threshold}\n")
        f.write(f"- decodings: {decodings} (sample_seeds={sample_seeds})\n\n")

        f.write("## Sharedness sweep\n\n")
        f.write("| basis_k | sharedness(v) | sharedness(v_fixed) |\n")
        f.write("| --- | --- | --- |\n")
        for r in shared_rows:
            f.write(f"| {r['basis_k']} | {r['sharedness_v']:.4f} | {r['sharedness_v_fixed']:.4f} |\n")

        f.write("\n## Aggregated metrics (per-template mean±std, worst)\n\n")
        f.write("| decoding | method | kind | basis_k | mean_success ± std | worst_success | mean_hits ± std | worst_hits |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for r in summary_rows:
            kstr = str(r["basis_k"]) if r["basis_k"] != "" else "-"
            f.write(
                f"| {r['decoding']} | {r['method']} | {r['kind']} | {kstr} | "
                f"{r['mean_success']:.3f} ± {r['std_success']:.3f} | {r['worst_success']:.3f} | "
                f"{r['mean_hits']:.3f} ± {r['std_hits']:.3f} | {r['worst_hits']:.3f} |\n"
            )

        f.write("\n## LaTeX examples\n\n")
        f.write(f"- examples: `{os.path.basename(examples_tex)}`\n")

    # summary.tex
    tex_path = os.path.join(args.out_dir, "summary.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated LaTeX summary\n")
        f.write("\\section*{Shared-space steering experiment}\n")
        f.write("\\subsection*{Setup}\n")
        f.write("\\begin{itemize}\n")
        f.write(f"  \\item Model: {latex_escape(args.model)}\n")
        f.write(f"  \\item Layer: {args.layer}\n")
        f.write(f"  \\item Alpha: {alpha0}\n")
        f.write(f"  \\item Inject window: start\\_step={args.inject_start_step}, first\\_n={args.inject_first_n}\n")
        f.write(f"  \\item Pirate threshold: {args.pirate_threshold}\n")
        f.write("\\end{itemize}\n\n")

        f.write("\\subsection*{Sharedness sweep}\n")
        f.write("\\begin{tabular}{rcc}\n\\hline\n")
        f.write("basis\\_k & sharedness($v$) & sharedness($v_{\\mathrm{fixed}}$)\\\\\n\\hline\n")
        for r in shared_rows:
            f.write(f"{r['basis_k']} & {r['sharedness_v']:.4f} & {r['sharedness_v_fixed']:.4f}\\\\\n")
        f.write("\\hline\n\\end{tabular}\n\n")

        f.write("\\subsection*{Aggregated metrics}\n")
        f.write("\\begin{tabular}{lllrcccc}\n\\hline\n")
        f.write("dec & method & kind & k & meanSuc & worstSuc & meanHits & worstHits\\\\\n\\hline\n")
        for r in summary_rows:
            kstr = str(r["basis_k"]) if r["basis_k"] != "" else "-"
            f.write(
                f"{latex_escape(r['decoding'])} & {latex_escape(r['method'])} & {latex_escape(r['kind'])} & {latex_escape(kstr)} & "
                f"{r['mean_success']:.3f} $\\pm$ {r['std_success']:.3f} & {r['worst_success']:.3f} & "
                f"{r['mean_hits']:.3f} $\\pm$ {r['std_hits']:.3f} & {r['worst_hits']:.3f}\\\\\n"
            )
        f.write("\\hline\n\\end{tabular}\n\n")

        f.write("\\subsection*{Representative examples}\n")
        f.write(f"\\input{{{latex_escape(os.path.basename(examples_tex))}}}\n")

    # meta
    payload = {
        "config": vars(args),
        "alpha_used": alpha0,
        "sharedness": shared_rows,
        "example_methods": {"no_steer": method_no, "v_orig": method_orig, "v_fixed_kmax": method_fixed_kmax},
    }
    with open(os.path.join(args.out_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[Done] wrote: results.csv, sharedness.csv, summary_stats.csv, summary.md, summary.tex, examples.tex")


if __name__ == "__main__":
    main()
