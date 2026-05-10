#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
steering_decodeshare_full.py

End-to-end, decode-aligned (KV-cache, seq_len==1) steering repair experiment.

What it does:
1) Estimate decode-time shared subspace Q via DecodeShare (pooled PCA + relative variance threshold).
2) Estimate steering vector v for each task via decode-aligned CAA (mean difference of decode states).
3) Repair: v_beta = (I - beta Q Q^T) v; plus energy-matched random control.
4) Evaluate template robustness on BoolQ / RTE / SST-2 via forced-choice margin shift,
   strictly using cache-advanced decode scoring.
5) Statistics: paired bootstrap CI + paired sign-flip permutation test (worst-template focus).

Targets: Llama / Qwen / Falcon (HF transformers).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# --------------------------
# Repro utils
# --------------------------

def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sanitize_model_id(mid: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", mid)


# --------------------------
# Model / layer access
# --------------------------

def get_transformer_layers(model: torch.nn.Module) -> List[torch.nn.Module]:
    """
    Return the list of block modules for common decoder-only HF models.
    Works for:
      - Llama/Qwen: model.model.layers
      - Falcon: model.transformer.h
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    # fallback: try common attribute names
    for name in ["layers", "h", "blocks"]:
        if hasattr(model, name):
            obj = getattr(model, name)
            if isinstance(obj, (list, torch.nn.ModuleList)):
                return list(obj)
    raise ValueError("Unsupported model architecture: cannot locate transformer layers.")


class LayerHook:
    """
    Context manager for a forward hook that can (a) capture activations, (b) modify outputs.
    Handles outputs that are either Tensor or tuple(Tensor, ...).
    """
    def __init__(
        self,
        layer: torch.nn.Module,
        fn,
    ):
        self.layer = layer
        self.fn = fn
        self.handle = None

    def __enter__(self):
        self.handle = self.layer.register_forward_hook(self.fn)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            self.handle.remove()
        return False


# --------------------------
# Chat template wrapper
# --------------------------

def wrap_as_chat(tokenizer, user_text: str, system_text: Optional[str] = None) -> str:
    """
    Use tokenizer.chat_template if available; otherwise return plain text.
    Ensures instruct/chat models (Llama-chat, Qwen-instruct, Falcon-instruct) behave more stably.
    """
    if getattr(tokenizer, "chat_template", None):
        msgs = []
        if system_text:
            msgs.append({"role": "system", "content": system_text})
        msgs.append({"role": "user", "content": user_text})
        # add_generation_prompt=True puts assistant prefix at end
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    # plain LM
    if system_text:
        return system_text + "\n\n" + user_text
    return user_text


# --------------------------
# Cache-advanced decode utilities (seq_len==1 alignment)
# --------------------------

@torch.inference_mode()
def prefill_prefix(
    model,
    input_ids_prefix: torch.Tensor,
) -> Optional[Tuple]:
    """
    Prefill on prefix tokens (len>=1). Return past_key_values or None if empty prefix.
    """
    if input_ids_prefix is None or input_ids_prefix.numel() == 0 or input_ids_prefix.shape[1] == 0:
        return None
    out = model(input_ids=input_ids_prefix, use_cache=True)
    return out.past_key_values


@torch.inference_mode()
def decode_step(
    model,
    input_ids_1tok: torch.Tensor,              # [B,1]
    past_key_values: Optional[Tuple],
) -> Tuple[torch.Tensor, Optional[Tuple]]:
    """
    One cached decode step with seq_len==1.
    Return (logits[B,1,V], new_past).
    """
    out = model(input_ids=input_ids_1tok, past_key_values=past_key_values, use_cache=True)
    return out.logits, out.past_key_values


@torch.inference_mode()
def cache_advanced_margin(
    model,
    tokenizer,
    prompt_text: str,
    cand_a: str,
    cand_b: str,
    hook_ctx: Optional[Any] = None,
    max_prompt_tokens: int = 512,
) -> float:
    """
    Compute margin = logP(cand_a | prompt) - logP(cand_b | prompt)
    using cache-advanced scoring so that the decisive logit is produced by seq_len==1 decode.

    Supports multi-token candidates by teacher-forcing them and accumulating logprobs,
    still with seq_len==1 at every step.
    """
    # tokenize prompt
    enc = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_prompt_tokens)
    input_ids = enc["input_ids"].to(model.device)
    if input_ids.shape[1] < 1:
        raise ValueError("Prompt tokenized to empty input_ids.")

    # tokenize candidates (no special tokens)
    cand_a_ids = tokenizer.encode(cand_a, add_special_tokens=False)
    cand_b_ids = tokenizer.encode(cand_b, add_special_tokens=False)
    if len(cand_a_ids) == 0 or len(cand_b_ids) == 0:
        raise ValueError("Candidate tokenization produced empty ids.")

    def score_candidate(cand_ids: List[int]) -> float:
        # split prompt into prefix and last token (cache-advance)
        if input_ids.shape[1] == 1:
            prefix = input_ids[:, :0]
            last_tok = input_ids[:, :1]
        else:
            prefix = input_ids[:, :-1]
            last_tok = input_ids[:, -1:]

        past = prefill_prefix(model, prefix)
        total_lp = 0.0

        cur_in = last_tok  # [1,1]
        for tid in cand_ids:
            # apply hook during decode
            if hook_ctx is None:
                logits, past = decode_step(model, cur_in, past)
            else:
                with hook_ctx:
                    logits, past = decode_step(model, cur_in, past)

            # logits -> predict next token
            logp = torch.log_softmax(logits[:, -1, :], dim=-1)  # [B,V]
            total_lp += float(logp[0, tid].item())

            # next input is the forced token
            cur_in = torch.tensor([[tid]], device=logits.device, dtype=torch.long)

        return total_lp

    lp_a = score_candidate(cand_a_ids)
    lp_b = score_candidate(cand_b_ids)
    return lp_a - lp_b


# --------------------------
# Task: datasets, templates, label mapping, prompts
# --------------------------

STEER_TASKS = ["boolq", "rte", "sst2"]

DEFAULT_STEER_TEMPLATES = {
    "boolq": [
        "Passage:\n{passage}\n\nQuestion: {question}\nAnswer:",
        "Read the passage and answer Yes or No.\n\nPassage: {passage}\n\nQ: {question}\nA:",
        "Given the passage, answer the question with Yes or No.\n\n{passage}\n\nQuestion: {question}\nAnswer:",
    ],
    "rte": [
        "Premise: {premise}\nHypothesis: {hypothesis}\nIs the hypothesis entailed by the premise? Answer True or False.\nAnswer:",
        "Textual entailment.\n\nP: {premise}\nH: {hypothesis}\nDoes P entail H? True or False.\nLabel:",
        "Decide entailment.\nPremise: {premise}\nHypothesis: {hypothesis}\nFinal answer (True/False):",
    ],
    "sst2": [
        "Review: {sentence}\nSentiment (Good/Bad):",
        "Classify sentiment as Good or Bad.\nSentence: {sentence}\nSentiment:",
        "Is the sentiment positive or negative? Answer Good for positive, Bad for negative.\n\n{sentence}\nAnswer:",
    ],
}

# Candidate-pair search space (we will calibrate best pair per model)
CANDIDATE_PAIRS = {
    "boolq": [(" Yes", " No"), (" True", " False"), (" true", " false"), (" yes", " no")],
    "rte":   [(" True", " False"), (" Yes", " No")],
    "sst2":  [(" Good", " Bad"), (" Positive", " Negative"), (" positive", " negative")],
}

def load_steer_dataset(task: str, split: str = "validation"):
    if task == "boolq":
        return load_dataset("boolq", split=split)
    if task == "rte":
        # GLUE RTE
        return load_dataset("glue", "rte", split=split)
    if task == "sst2":
        return load_dataset("glue", "sst2", split=split)
    raise ValueError(f"Unknown steer task: {task}")


def get_label(task: str, ex: Dict[str, Any]) -> int:
    """
    Return label in {0,1} with a consistent meaning:
      - boolq: 1=True(Yes), 0=False(No)
      - rte:   1=entailment(True), 0=not_entailment(False)
      - sst2:  1=positive(Good), 0=negative(Bad)
    """
    if task == "boolq":
        return int(bool(ex["answer"]))
    if task == "rte":
        # GLUE: label 0 = entailment, 1 = not_entailment
        # convert to (1 entailment, 0 not)
        return 1 if int(ex["label"]) == 0 else 0
    if task == "sst2":
        return 1 if int(ex["label"]) == 1 else 0
    raise ValueError(task)


def render_user_text(task: str, template: str, ex: Dict[str, Any]) -> str:
    if task == "boolq":
        return template.format(passage=ex["passage"], question=ex["question"])
    if task == "rte":
        return template.format(premise=ex["sentence1"], hypothesis=ex["sentence2"])
    if task == "sst2":
        return template.format(sentence=ex["sentence"])
    raise ValueError(task)


# --------------------------
# Decode-aligned CAA steering vector estimation
# --------------------------

@torch.inference_mode()
def get_decode_state_at_layer(
    model,
    tokenizer,
    prompt_text: str,
    layer_idx: int,
    max_prompt_tokens: int = 512,
) -> torch.Tensor:
    """
    Return decode-aligned hidden state h_l for the *prompt boundary* cached decode call (seq_len==1),
    captured at the output of block[layer_idx]. Shape: [d]
    """
    layers = get_transformer_layers(model)
    if not (0 <= layer_idx < len(layers)):
        raise ValueError(f"layer_idx {layer_idx} out of range (0..{len(layers)-1}).")

    enc = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_prompt_tokens)
    input_ids = enc["input_ids"].to(model.device)
    if input_ids.shape[1] == 1:
        prefix = input_ids[:, :0]
        last_tok = input_ids[:, :1]
    else:
        prefix = input_ids[:, :-1]
        last_tok = input_ids[:, -1:]

    captured: Dict[str, torch.Tensor] = {}

    def capture_hook(_module, _inp, out):
        # out: Tensor [B,1,d] or tuple(out0,...)
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] != 1:
            return out
        captured["h"] = h[:, -1, :].detach()
        return out

    past = prefill_prefix(model, prefix)

    with LayerHook(layers[layer_idx], capture_hook):
        _logits, _past2 = decode_step(model, last_tok, past)

    if "h" not in captured:
        raise RuntimeError("Failed to capture decode state (did you hit seq_len==1?).")
    return captured["h"][0]  # [d]


@torch.inference_mode()
def estimate_caa_vector(
    model,
    tokenizer,
    task: str,
    dataset,
    layer_idx: int,
    template_for_caa: str,
    n_per_class: int,
    seed: int,
    system_text: Optional[str] = "You are a helpful assistant.",
    max_prompt_tokens: int = 512,
) -> torch.Tensor:
    """
    Decode-aligned CAA: v = mean(h|label=1) - mean(h|label=0), using prompt-boundary decode states.
    """
    rng = np.random.default_rng(seed)
    idxs = rng.permutation(len(dataset)).tolist()

    pos_states = []
    neg_states = []
    for i in idxs:
        ex = dataset[i]
        y = get_label(task, ex)
        user_text = render_user_text(task, template_for_caa, ex)
        prompt = wrap_as_chat(tokenizer, user_text, system_text=system_text)

        h = get_decode_state_at_layer(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt,
            layer_idx=layer_idx,
            max_prompt_tokens=max_prompt_tokens,
        )
        if y == 1 and len(pos_states) < n_per_class:
            pos_states.append(h)
        elif y == 0 and len(neg_states) < n_per_class:
            neg_states.append(h)

        if len(pos_states) >= n_per_class and len(neg_states) >= n_per_class:
            break

    if len(pos_states) < n_per_class or len(neg_states) < n_per_class:
        raise RuntimeError(f"Not enough samples for CAA in {task}: got pos={len(pos_states)} neg={len(neg_states)}")

    pos = torch.stack(pos_states, dim=0).float().mean(dim=0)
    neg = torch.stack(neg_states, dim=0).float().mean(dim=0)
    v = pos - neg
    v = v / (v.norm(p=2) + 1e-12)
    return v  # float32, on model.device (because h on model.device)


# --------------------------
# DecodeShare shared basis estimation (pooled PCA + relative variance threshold)
# We'll estimate Q_shared using decode-time states collected by greedy decoding for K steps,
# strictly recording only seq_len==1 cached decode calls.
# --------------------------

BASIS_TASKS_DEFAULT = ["gsm8k", "commonsenseqa", "strategyqa", "aqua", "openbookqa", "qasc", "piqa", "boolq"]

def load_basis_dataset(task: str, split: str = "train"):
    # keep it simple; if split not exist, fall back
    if task == "gsm8k":
        try:
            return load_dataset("gsm8k", "main", split=split)
        except Exception:
            return load_dataset("gsm8k", "main", split="train")
    if task == "commonsenseqa":
        try:
            return load_dataset("commonsense_qa", split=split)
        except Exception:
            return load_dataset("commonsense_qa", split="train")
    if task == "strategyqa":
        try:
            return load_dataset("strategyqa", split=split)
        except Exception:
            return load_dataset("strategyqa", split="train")
    if task == "aqua":
        # HF dataset name
        try:
            return load_dataset("aqua_rat", split=split)
        except Exception:
            return load_dataset("aqua_rat", split="train")
    if task == "openbookqa":
        try:
            return load_dataset("openbookqa", "main", split=split)
        except Exception:
            return load_dataset("openbookqa", "main", split="train")
    if task == "qasc":
        try:
            return load_dataset("qasc", split=split)
        except Exception:
            return load_dataset("qasc", split="train")
    if task == "piqa":
        try:
            return load_dataset("piqa", split=split)
        except Exception:
            return load_dataset("piqa", split="train")
    if task == "boolq":
        return load_dataset("boolq", split="train")
    raise ValueError(f"Unknown basis task: {task}")


def prompt_for_basis(task: str, ex: Dict[str, Any]) -> str:
    # Simple prompts (not scored), only to induce decode states.
    if task == "gsm8k":
        return f"Solve the math problem.\n\n{ex['question']}\nAnswer:"
    if task == "commonsenseqa":
        q = ex["question"]
        choices = ex["choices"]
        # choices: {'label': [...], 'text': [...]}
        lines = []
        for lab, txt in zip(choices["label"], choices["text"]):
            lines.append(f"{lab}. {txt}")
        return "Question: " + q + "\nChoices:\n" + "\n".join(lines) + "\nAnswer:"
    if task == "strategyqa":
        return f"Question: {ex['question']}\nAnswer Yes or No.\nAnswer:"
    if task == "aqua":
        opts = ex.get("options", [])
        opt_lines = "\n".join([str(o) for o in opts])
        return f"Question: {ex['question']}\nOptions:\n{opt_lines}\nAnswer:"
    if task == "openbookqa":
        stem = ex["question_stem"]
        choices = ex["choices"]
        lines = []
        for lab, txt in zip(choices["label"], choices["text"]):
            lines.append(f"{lab}. {txt}")
        return f"Question: {stem}\nChoices:\n" + "\n".join(lines) + "\nAnswer:"
    if task == "qasc":
        # qasc schema varies; handle common
        q = ex.get("question", ex.get("combinedfact", ""))
        choices = ex.get("choices", None)
        if choices and isinstance(choices, dict) and "label" in choices and "text" in choices:
            lines = [f"{lab}. {txt}" for lab, txt in zip(choices["label"], choices["text"])]
            return f"Question: {q}\nChoices:\n" + "\n".join(lines) + "\nAnswer:"
        return f"Question: {q}\nAnswer:"
    if task == "piqa":
        return f"Goal: {ex['goal']}\nChoice A: {ex['sol1']}\nChoice B: {ex['sol2']}\nAnswer:"
    if task == "boolq":
        return f"Passage:\n{ex['passage']}\n\nQuestion: {ex['question']}\nAnswer Yes or No.\nAnswer:"
    raise ValueError(task)


@torch.inference_mode()
def collect_decode_states_greedy(
    model,
    tokenizer,
    prompts: List[str],
    layer_idx: int,
    K: int,
    system_text: Optional[str],
    max_prompt_tokens: int,
    seed: int,
) -> torch.Tensor:
    """
    For each prompt:
      - wrap as chat (if possible)
      - cache-advance so step0 is a seq_len==1 decode call at prompt boundary
      - greedy decode for up to K steps
      - record layer output hidden state at layer_idx for each seq_len==1 call
    Returns: Tensor [N_states, d] (float32 on CPU)
    """
    rng = np.random.default_rng(seed)
    layers = get_transformer_layers(model)
    d = None

    all_states: List[torch.Tensor] = []

    for p in tqdm(prompts, desc="Collect decode states", leave=False):
        text = wrap_as_chat(tokenizer, p, system_text=system_text)
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_prompt_tokens)
        input_ids = enc["input_ids"].to(model.device)
        if input_ids.shape[1] < 1:
            continue

        # split prompt into prefix and last token
        if input_ids.shape[1] == 1:
            prefix = input_ids[:, :0]
            cur_in = input_ids[:, :1]
        else:
            prefix = input_ids[:, :-1]
            cur_in = input_ids[:, -1:]  # [1,1]

        past = prefill_prefix(model, prefix)

        captured: Dict[str, torch.Tensor] = {}

        def cap_hook(_module, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if h.shape[1] != 1:
                return out
            captured["h"] = h[:, -1, :].detach()
            return out

        for _step in range(K):
            captured.clear()
            with LayerHook(layers[layer_idx], cap_hook):
                logits, past = decode_step(model, cur_in, past)

            if "h" not in captured:
                break
            h = captured["h"][0]  # [d] on device
            if d is None:
                d = h.shape[0]
            all_states.append(h.float().cpu())

            # greedy next token
            next_id = int(torch.argmax(logits[0, -1, :]).item())
            if next_id == tokenizer.eos_token_id:
                break
            cur_in = torch.tensor([[next_id]], device=model.device, dtype=torch.long)

    if len(all_states) == 0:
        raise RuntimeError("No decode states collected.")
    X = torch.stack(all_states, dim=0)  # [N,d], float32 cpu
    return X


def orthonormalize(Q: torch.Tensor) -> torch.Tensor:
    # QR orthonormalization
    Q = Q.float()
    q, _r = torch.linalg.qr(Q, mode="reduced")
    return q


@dataclass
class DecodeShareResult:
    Q_pooled: torch.Tensor      # [d,k]
    Q_shared: torch.Tensor      # [d,|S|]
    shared_idx: np.ndarray      # indices in pooled basis
    r_scores: Dict[str, np.ndarray]  # task -> [k] relative variance


def decode_share_estimate(
    task_states: Dict[str, torch.Tensor],   # each [N_t,d] float32 cpu
    rho: float,
    tau: float,
    m_shared: int,
    seed: int,
    pca_q_max: Optional[int] = None,
    device_for_pca: str = "cpu",
) -> DecodeShareResult:
    """
    Algorithm 1 style:
      1) task-center each X_t
      2) balance to n_min by subsampling
      3) pool, run PCA via torch.pca_lowrank
      4) choose k by variance retention rho using total_var=||X||_F^2
      5) compute r_{t,i}; shared set S where >=tau for >=m_shared tasks
    """
    rng = np.random.default_rng(seed)
    tasks = list(task_states.keys())
    if len(tasks) < 2:
        raise ValueError("Need >=2 tasks for DecodeShare.")

    # task-center + balance
    centered: Dict[str, torch.Tensor] = {}
    n_min = min(x.shape[0] for x in task_states.values())
    for t in tasks:
        X = task_states[t]
        # subsample to n_min
        if X.shape[0] > n_min:
            idx = rng.choice(X.shape[0], size=n_min, replace=False)
            X = X[idx]
        mu = X.mean(dim=0, keepdim=True)
        centered[t] = X - mu

    X_pool = torch.cat([centered[t] for t in tasks], dim=0)  # [(T*n_min), d] cpu float32
    n, d = X_pool.shape

    # PCA low-rank (randomized)
    Xp = X_pool.to(device_for_pca)
    total_var = float((Xp.float() ** 2).sum().item())  # ||X||_F^2

    # choose q (upper bound on returned PCs)
    q = pca_q_max if pca_q_max is not None else min(d, 2048)
    q = int(min(q, d, n - 1))
    if q <= 0:
        raise ValueError("Invalid PCA q. Try smaller prompts or bigger dataset.")

    # torch.pca_lowrank returns U,S,V
    U, S, V = torch.pca_lowrank(Xp.float(), q=q, center=False, niter=2)
    # explained variance ratios using total_var
    s2 = (S ** 2)
    cum = torch.cumsum(s2, dim=0) / max(total_var, 1e-12)
    k = int(torch.searchsorted(cum, torch.tensor(rho, device=cum.device)).item() + 1)
    k = min(k, V.shape[1])
    if k == V.shape[1] and float(cum[-1].item()) < rho:
        print(f"[WARN] PCA q={q} insufficient to reach rho={rho:.3f}. "
              f"Reached {float(cum[-1].item()):.3f}. Consider increasing --pca_q_max.")

    Q_pooled = V[:, :k].contiguous()  # [d,k] on device_for_pca
    Q_pooled = orthonormalize(Q_pooled).to("cpu")

    # per-task r scores
    r_scores: Dict[str, np.ndarray] = {}
    above = np.zeros((len(tasks), k), dtype=np.int32)

    for ti, t in enumerate(tasks):
        Xt = centered[t]  # cpu
        Z = (Xt @ Q_pooled)  # [n_min,k]
        v = Z.var(dim=0, unbiased=False) + 1e-12
        r = (v / v.sum()).cpu().numpy()
        r_scores[t] = r
        above[ti, :] = (r >= tau).astype(np.int32)

    counts = above.sum(axis=0)  # [k]
    shared_idx = np.where(counts >= m_shared)[0]
    Q_shared = Q_pooled[:, shared_idx]
    Q_shared = orthonormalize(Q_shared).to("cpu")

    return DecodeShareResult(
        Q_pooled=Q_pooled,
        Q_shared=Q_shared,
        shared_idx=shared_idx,
        r_scores=r_scores,
    )


# --------------------------
# Steering injection hook (decode-only)
# --------------------------

class DecodeOnlySteerHook:
    """
    A context manager that registers a forward hook on a chosen layer,
    and adds lambda * v to the layer output hidden state, only when seq_len==1.
    """
    def __init__(self, model, layer_idx: int, v: torch.Tensor, lam: float):
        self.model = model
        self.layers = get_transformer_layers(model)
        self.layer_idx = layer_idx
        self.v = v
        self.lam = float(lam)
        self.handle = None

    def _hook(self, module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] != 1:
            return out

        # move v to correct device/dtype on the fly
        v = self.v.to(device=h.device, dtype=h.dtype)
        h2 = h + self.lam * v.view(1, 1, -1)

        if isinstance(out, tuple):
            return (h2,) + out[1:]
        return h2

    def __enter__(self):
        self.handle = self.layers[self.layer_idx].register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            self.handle.remove()
        return False


# --------------------------
# Steering repair vectors + random control
# --------------------------

def repair_vector(v: torch.Tensor, Q_shared: torch.Tensor, beta: float) -> torch.Tensor:
    """
    v_beta = v - beta * Q Q^T v
    """
    # keep in float32 for stability
    v32 = v.float()
    Q = Q_shared.float().to(v32.device)
    proj = Q @ (Q.t() @ v32)
    out = v32 - float(beta) * proj
    out = out / (out.norm(p=2) + 1e-12)
    return out


def random_energy_matched_control(v: torch.Tensor, Q_shared: torch.Tensor, seed: int) -> torch.Tensor:
    """
    Random control like Table-style "R":
      - sample random orthonormal basis Q_rand with same dim as Q_shared
      - choose alpha so that ||alpha * Q_rand^T v|| ~= ||Q_shared^T v||
      - v_R = v - alpha * Q_rand Q_rand^T v
      - normalize
    """
    v32 = v.float()
    d = v32.shape[0]
    k = Q_shared.shape[1]
    if k == 0:
        return v32 / (v32.norm() + 1e-12)

    g = torch.Generator(device=v32.device)
    g.manual_seed(seed)

    A = torch.randn(d, k, generator=g, device=v32.device, dtype=torch.float32)
    Qr = orthonormalize(A)  # [d,k]
    # match removed energy magnitude
    e_shared = float((Q_shared.float().to(v32.device).t() @ v32).norm().item())
    e_rand = float((Qr.t() @ v32).norm().item()) + 1e-12
    alpha = e_shared / e_rand
    out = v32 - alpha * (Qr @ (Qr.t() @ v32))
    out = out / (out.norm(p=2) + 1e-12)
    return out


def shared_overlap(v: torch.Tensor, Q_shared: torch.Tensor) -> float:
    """
    sh(v) = ||Q^T v|| / ||v||
    """
    v32 = v.float()
    Q = Q_shared.float().to(v32.device)
    num = float((Q.t() @ v32).norm().item())
    den = float(v32.norm().item()) + 1e-12
    return num / den


# --------------------------
# Candidate calibration (pick best pair under baseline forced-choice acc)
# --------------------------

@torch.inference_mode()
def forced_choice_predict(
    model,
    tokenizer,
    prompt_text: str,
    cand_a: str,
    cand_b: str,
    hook_ctx: Optional[Any],
    max_prompt_tokens: int,
) -> int:
    """
    Return 1 if cand_a wins else 0 (based on margin > 0).
    """
    m = cache_advanced_margin(
        model=model,
        tokenizer=tokenizer,
        prompt_text=prompt_text,
        cand_a=cand_a,
        cand_b=cand_b,
        hook_ctx=hook_ctx,
        max_prompt_tokens=max_prompt_tokens,
    )
    return 1 if m > 0 else 0


def pick_best_candidate_pair(
    model,
    tokenizer,
    task: str,
    dataset,
    templates: List[str],
    candidate_pairs: List[Tuple[str, str]],
    n_calib: int,
    seed: int,
    system_text: Optional[str],
    max_prompt_tokens: int,
) -> Tuple[str, str]:
    """
    Choose the candidate token pair (A,B) with best baseline accuracy on a balanced subset.
    """
    rng = np.random.default_rng(seed)
    idxs = rng.permutation(len(dataset)).tolist()

    # balanced subset
    pos = []
    neg = []
    for i in idxs:
        ex = dataset[i]
        y = get_label(task, ex)
        if y == 1 and len(pos) < n_calib // 2:
            pos.append(ex)
        elif y == 0 and len(neg) < n_calib // 2:
            neg.append(ex)
        if len(pos) >= n_calib // 2 and len(neg) >= n_calib // 2:
            break
    subset = pos + neg
    rng.shuffle(subset)

    best = None
    best_acc = -1.0
    tmpl = templates[0]  # use a fixed template for calibration
    for (a, b) in candidate_pairs:
        correct = 0
        for ex in subset:
            user_text = render_user_text(task, tmpl, ex)
            prompt = wrap_as_chat(tokenizer, user_text, system_text=system_text)
            pred = forced_choice_predict(model, tokenizer, prompt, a, b, hook_ctx=None,
                                         max_prompt_tokens=max_prompt_tokens)
            y = get_label(task, ex)
            # interpret cand_a as label=1, cand_b as label=0
            if pred == y:
                correct += 1
        acc = correct / max(len(subset), 1)
        if acc > best_acc:
            best_acc = acc
            best = (a, b)

    if best is None:
        raise RuntimeError("Failed to pick candidate pair.")
    print(f"[CandCalib] task={task} best_pair={best} calib_acc={best_acc:.3f}")
    return best


# --------------------------
# Evaluation metrics + statistics
# --------------------------

@dataclass
class TemplateMetrics:
    mu: float
    anti: float
    per_example_shift: np.ndarray  # shape [n]


@dataclass
class RobustnessMetrics:
    mu: float
    sigma_tmpl: float
    worst: float
    antiworst: float
    worst_template_idx: int
    per_template: List[TemplateMetrics]


def compute_robustness_metrics(shifts_by_template: List[np.ndarray]) -> RobustnessMetrics:
    per_template: List[TemplateMetrics] = []
    mus = []
    antis = []
    for arr in shifts_by_template:
        mu = float(np.mean(arr))
        anti = float(np.mean(arr < 0.0))
        per_template.append(TemplateMetrics(mu=mu, anti=anti, per_example_shift=arr))
        mus.append(mu)
        antis.append(anti)

    mu_all = float(np.mean(mus))
    sigma = float(np.std(mus, ddof=0))
    worst_idx = int(np.argmin(mus))
    worst = float(mus[worst_idx])
    antiworst = float(antis[worst_idx])
    return RobustnessMetrics(
        mu=mu_all,
        sigma_tmpl=sigma,
        worst=worst,
        antiworst=antiworst,
        worst_template_idx=worst_idx,
        per_template=per_template,
    )


def paired_signflip_pvalue(deltas: np.ndarray, n_perm: int, seed: int, one_sided: bool = True) -> float:
    """
    Paired randomization (sign-flip) test on per-example deltas.
    H0: delta distribution symmetric about 0.
    """
    rng = np.random.default_rng(seed)
    obs = float(np.mean(deltas))
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=deltas.shape[0])
        m = float(np.mean(deltas * signs))
        if one_sided:
            if m >= obs:
                count += 1
        else:
            if abs(m) >= abs(obs):
                count += 1
    return (count + 1.0) / (n_perm + 1.0)


def paired_bootstrap_ci_mean(deltas: np.ndarray, n_boot: int, seed: int, alpha: float = 0.05) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = deltas.shape[0]
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(float(np.mean(deltas[idx])))
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return lo, hi


def bootstrap_ci_worst_diff(
    shifts_a: List[np.ndarray],
    shifts_b: List[np.ndarray],
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """
    Bootstrap CI for (worst_b - worst_a), where worst is min over template mean shifts.
    Paired over examples within each template.
    """
    rng = np.random.default_rng(seed)
    T = len(shifts_a)
    assert T == len(shifts_b)
    boots = []
    for _ in range(n_boot):
        mus_a = []
        mus_b = []
        for t in range(T):
            a = shifts_a[t]
            b = shifts_b[t]
            n = a.shape[0]
            idx = rng.integers(0, n, size=n)
            mus_a.append(float(np.mean(a[idx])))
            mus_b.append(float(np.mean(b[idx])))
        worst_a = float(np.min(mus_a))
        worst_b = float(np.min(mus_b))
        boots.append(worst_b - worst_a)
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return lo, hi


# --------------------------
# Main experiment runner
# --------------------------

@torch.inference_mode()
def eval_task_across_templates(
    model,
    tokenizer,
    task: str,
    dataset,
    templates: List[str],
    cand_a: str,
    cand_b: str,
    v: torch.Tensor,
    lam: float,
    layer_idx: int,
    system_text: Optional[str],
    n_eval: int,
    seed: int,
    max_prompt_tokens: int,
) -> RobustnessMetrics:
    """
    Compute per-example margin shift across templates:
      shift = (margin_with_steer - margin_base)
      margin = logP(A)-logP(B)
    """
    rng = np.random.default_rng(seed)
    idxs = rng.permutation(len(dataset))[:n_eval].tolist()
    subset = [dataset[i] for i in idxs]

    shifts_by_template: List[np.ndarray] = []

    for tmpl in templates:
        shifts = []
        for ex in subset:
            user_text = render_user_text(task, tmpl, ex)
            prompt = wrap_as_chat(tokenizer, user_text, system_text=system_text)

            base = cache_advanced_margin(model, tokenizer, prompt, cand_a, cand_b,
                                         hook_ctx=None, max_prompt_tokens=max_prompt_tokens)

            hook = DecodeOnlySteerHook(model, layer_idx=layer_idx, v=v, lam=lam)
            steered = cache_advanced_margin(model, tokenizer, prompt, cand_a, cand_b,
                                            hook_ctx=hook, max_prompt_tokens=max_prompt_tokens)

            shifts.append(steered - base)

        shifts_by_template.append(np.array(shifts, dtype=np.float64))

    return compute_robustness_metrics(shifts_by_template)


def tune_lambda_simple(
    model,
    tokenizer,
    task: str,
    dataset,
    templates: List[str],
    cand_a: str,
    cand_b: str,
    v: torch.Tensor,
    layer_idx: int,
    system_text: Optional[str],
    n_tune: int,
    seed: int,
    max_prompt_tokens: int,
    lam_grid: List[float],
) -> float:
    """
    Simple lambda tuning: choose lam that maximizes mean mu across templates on a small subset,
    while avoiding extreme antiworst (optional soft constraint).
    """
    best_lam = lam_grid[0]
    best_score = -1e9
    for lam in lam_grid:
        mets = eval_task_across_templates(
            model=model,
            tokenizer=tokenizer,
            task=task,
            dataset=dataset,
            templates=templates,
            cand_a=cand_a,
            cand_b=cand_b,
            v=v,
            lam=lam,
            layer_idx=layer_idx,
            system_text=system_text,
            n_eval=n_tune,
            seed=seed,
            max_prompt_tokens=max_prompt_tokens,
        )
        # score: prioritize worst-case positivity, then overall mu
        score = mets.worst * 3.0 + mets.mu * 1.0 - mets.antiworst * 0.5
        if score > best_score:
            best_score = score
            best_lam = lam
    print(f"[TuneLambda] task={task} best_lam={best_lam} score={best_score:.4f}")
    return best_lam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="HF model id")
    ap.add_argument("--layer", type=int, default=10, help="which layer to steer / estimate Q_shared")
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device_map", type=str, default="auto", help="HF device_map (auto / cuda / cpu)")
    ap.add_argument("--out_dir", type=str, default="./steer_out")

    # DecodeShare basis
    ap.add_argument("--basis_tasks", type=str, default=",".join(BASIS_TASKS_DEFAULT))
    ap.add_argument("--basis_split", type=str, default="train")
    ap.add_argument("--basis_n_prompts", type=int, default=128)
    ap.add_argument("--basis_K", type=int, default=16)
    ap.add_argument("--rho", type=float, default=0.95)
    ap.add_argument("--tau", type=float, default=1e-3)
    ap.add_argument("--m_shared", type=int, default=8, help=">= how many tasks must exceed tau to be shared")
    ap.add_argument("--pca_q_max", type=int, default=2048, help="max PCs computed by torch.pca_lowrank")
    ap.add_argument("--pca_device", type=str, default="cpu", choices=["cpu", "cuda"])

    # Steering eval
    ap.add_argument("--steer_tasks", type=str, default="boolq,rte,sst2")
    ap.add_argument("--eval_n", type=int, default=512)
    ap.add_argument("--caa_n_per_class", type=int, default=256)
    ap.add_argument("--cand_calib_n", type=int, default=256)
    ap.add_argument("--max_prompt_tokens", type=int, default=512)

    # Lambda tuning
    ap.add_argument("--tune_lambda", action="store_true")
    ap.add_argument("--tune_n", type=int, default=128)
    ap.add_argument("--lam_grid", type=str, default="0,2,4,6,8,10,12")

    # Stats
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--n_perm", type=int, default=10000)

    # misc
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--system_text", type=str, default="You are a helpful assistant.")
    args = ap.parse_args()

    seed_all(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # dtype
    if args.dtype == "float16":
        torch_dtype = torch.float16
    elif args.dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    print(f"[LoadModel] {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
        trust_remote_code=True,
    )
    model.eval()

    layers = get_transformer_layers(model)
    if not (0 <= args.layer < len(layers)):
        raise ValueError(f"--layer out of range: got {args.layer}, model has {len(layers)} layers")

    # ----------------------
    # 1) Estimate Q_shared
    # ----------------------
    basis_tasks = [t.strip() for t in args.basis_tasks.split(",") if t.strip()]
    if args.m_shared <= 0 or args.m_shared > len(basis_tasks):
        print(f"[WARN] m_shared={args.m_shared} invalid for #tasks={len(basis_tasks)}; "
              f"clamping to {len(basis_tasks)}")
        m_shared = len(basis_tasks)
    else:
        m_shared = args.m_shared

    task_states: Dict[str, torch.Tensor] = {}
    for t in basis_tasks:
        ds = load_basis_dataset(t, split=args.basis_split)
        n = min(args.basis_n_prompts, len(ds))
        # sample prompts
        rng = np.random.default_rng(args.seed)
        idxs = rng.choice(len(ds), size=n, replace=False)
        prompts = [prompt_for_basis(t, ds[int(i)]) for i in idxs]
        X = collect_decode_states_greedy(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            layer_idx=args.layer,
            K=args.basis_K,
            system_text=args.system_text,
            max_prompt_tokens=args.max_prompt_tokens,
            seed=args.seed + 7,
        )
        task_states[t] = X
        print(f"[BasisStates] task={t} X={tuple(X.shape)}")

    dsr = decode_share_estimate(
        task_states=task_states,
        rho=args.rho,
        tau=args.tau,
        m_shared=m_shared,
        seed=args.seed + 99,
        pca_q_max=args.pca_q_max,
        device_for_pca=args.pca_device,
    )
    Q_shared = dsr.Q_shared  # cpu float32
    print(f"[DecodeShare] pooled_k={dsr.Q_pooled.shape[1]} shared_dim={Q_shared.shape[1]} "
          f"(m_shared={m_shared}, tau={args.tau}, rho={args.rho})")

    basis_path = os.path.join(args.out_dir, f"{sanitize_model_id(args.model)}_layer{args.layer}_Qshared.pt")
    torch.save({"Q_shared": Q_shared, "meta": vars(args), "shared_idx": dsr.shared_idx}, basis_path)
    print(f"[Saved] {basis_path}")

    # ----------------------
    # 2) Steering eval for each task
    # ----------------------
    steer_tasks = [t.strip() for t in args.steer_tasks.split(",") if t.strip()]
    lam_grid = [float(x) for x in args.lam_grid.split(",") if x.strip()]

    summary: Dict[str, Any] = {}
    for task in steer_tasks:
        if task not in STEER_TASKS:
            print(f"[SKIP] unknown steer task: {task}")
            continue

        print(f"\n====================\n[Task] {task}\n====================")
        ds = load_steer_dataset(task, split="validation")
        templates = DEFAULT_STEER_TEMPLATES[task]

        # candidate calibration (pick best A/B for this tokenizer/model)
        cand_a, cand_b = pick_best_candidate_pair(
            model=model,
            tokenizer=tokenizer,
            task=task,
            dataset=ds,
            templates=templates,
            candidate_pairs=CANDIDATE_PAIRS[task],
            n_calib=args.cand_calib_n,
            seed=args.seed + 3,
            system_text=args.system_text,
            max_prompt_tokens=args.max_prompt_tokens,
        )

        # estimate steering vector v via decode-aligned CAA
        v = estimate_caa_vector(
            model=model,
            tokenizer=tokenizer,
            task=task,
            dataset=ds,
            layer_idx=args.layer,
            template_for_caa=templates[0],
            n_per_class=args.caa_n_per_class,
            seed=args.seed + 11,
            system_text=args.system_text,
            max_prompt_tokens=args.max_prompt_tokens,
        )

        # shared overlap
        sh = shared_overlap(v, Q_shared.to(v.device))
        print(f"[Overlap] sh(v)={sh:.4f}  (dim(Q_shared)={Q_shared.shape[1]})")

        # repair vectors
        v0 = v
        v05 = repair_vector(v, Q_shared, beta=0.5)
        v1 = repair_vector(v, Q_shared, beta=1.0)
        vR = random_energy_matched_control(v, Q_shared, seed=args.seed + 17)

        sh0 = shared_overlap(v0, Q_shared.to(v0.device))
        sh05 = shared_overlap(v05, Q_shared.to(v05.device))
        sh1 = shared_overlap(v1, Q_shared.to(v1.device))
        print(f"[Overlap] sh(v0)={sh0:.4f} sh(v0.5)={sh05:.4f} sh(v1)={sh1:.4f}")

        # tune lambda if requested
        if args.tune_lambda:
            lam = tune_lambda_simple(
                model=model,
                tokenizer=tokenizer,
                task=task,
                dataset=ds,
                templates=templates,
                cand_a=cand_a,
                cand_b=cand_b,
                v=v0,
                layer_idx=args.layer,
                system_text=args.system_text,
                n_tune=args.tune_n,
                seed=args.seed + 23,
                max_prompt_tokens=args.max_prompt_tokens,
                lam_grid=lam_grid,
            )
        else:
            lam = lam_grid[-1] if len(lam_grid) > 0 else 8.0
            print(f"[Lambda] use lam={lam} (set --tune_lambda to tune)")

        # evaluate
        mets0 = eval_task_across_templates(
            model, tokenizer, task, ds, templates, cand_a, cand_b,
            v=v0, lam=lam, layer_idx=args.layer,
            system_text=args.system_text,
            n_eval=args.eval_n, seed=args.seed + 31, max_prompt_tokens=args.max_prompt_tokens
        )
        mets05 = eval_task_across_templates(
            model, tokenizer, task, ds, templates, cand_a, cand_b,
            v=v05, lam=lam, layer_idx=args.layer,
            system_text=args.system_text,
            n_eval=args.eval_n, seed=args.seed + 31, max_prompt_tokens=args.max_prompt_tokens
        )
        mets1 = eval_task_across_templates(
            model, tokenizer, task, ds, templates, cand_a, cand_b,
            v=v1, lam=lam, layer_idx=args.layer,
            system_text=args.system_text,
            n_eval=args.eval_n, seed=args.seed + 31, max_prompt_tokens=args.max_prompt_tokens
        )
        metsR = eval_task_across_templates(
            model, tokenizer, task, ds, templates, cand_a, cand_b,
            v=vR, lam=lam, layer_idx=args.layer,
            system_text=args.system_text,
            n_eval=args.eval_n, seed=args.seed + 31, max_prompt_tokens=args.max_prompt_tokens
        )

        def fmt(m: RobustnessMetrics) -> str:
            return (f"mu={m.mu:+.4f}  sigma_tmpl={m.sigma_tmpl:.4f}  "
                    f"worst={m.worst:+.4f}  antiworst={m.antiworst:.4f}  "
                    f"worst_t={m.worst_template_idx}")

        print("\n[Results]")
        print(f"  beta=0.0 : {fmt(mets0)}")
        print(f"  beta=0.5 : {fmt(mets05)}")
        print(f"  beta=1.0 : {fmt(mets1)}")
        print(f"  controlR: {fmt(metsR)}")

        # ----------------------
        # 3) Significance tests (focus: worst-template improvement beta=1 vs beta=0)
        # ----------------------
        worst_t0 = mets0.worst_template_idx
        # use the SAME template index worst_t0 for paired test to avoid "moving target"
        delta_worst_template = (mets1.per_template[worst_t0].per_example_shift -
                                mets0.per_template[worst_t0].per_example_shift)

        p_sf = paired_signflip_pvalue(delta_worst_template, n_perm=args.n_perm,
                                      seed=args.seed + 101, one_sided=True)
        ci_sf = paired_bootstrap_ci_mean(delta_worst_template, n_boot=args.n_boot,
                                         seed=args.seed + 202)

        # bootstrap CI for worst-diff (min over templates)
        shifts0 = [tm.per_example_shift for tm in mets0.per_template]
        shifts1 = [tm.per_example_shift for tm in mets1.per_template]
        ci_worst = bootstrap_ci_worst_diff(shifts0, shifts1, n_boot=args.n_boot, seed=args.seed + 303)

        print("\n[Significance] (beta=1 vs beta=0)")
        print(f"  Worst-template-fixed (t={worst_t0}) mean(Δshift) CI95={ci_sf}, signflip p={p_sf:.4g}")
        print(f"  Worst-over-templates Δworst CI95={ci_worst}  (bootstrap over examples)")

        summary[task] = {
            "cand_pair": (cand_a, cand_b),
            "lambda": lam,
            "overlap": {"sh_v": sh0, "sh_v05": sh05, "sh_v1": sh1},
            "beta0": mets0.__dict__,
            "beta05": mets05.__dict__,
            "beta1": mets1.__dict__,
            "controlR": metsR.__dict__,
            "sig": {
                "worst_template_fixed": {
                    "t": worst_t0,
                    "mean_delta_shift": float(np.mean(delta_worst_template)),
                    "ci95": ci_sf,
                    "p_signflip": p_sf,
                },
                "worst_over_templates": {
                    "ci95": ci_worst,
                },
            },
        }

    out_json = os.path.join(args.out_dir, f"{sanitize_model_id(args.model)}_layer{args.layer}_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[Done] Wrote {out_json}")


if __name__ == "__main__":
    main()