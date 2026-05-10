
# -*- coding: utf-8 -*-
"""
h3_killer_counterfactual_grid_reasoning_v2.py

 NOTE：n_eval should be 2048, and use_forced_choice should be 1. 这个代码在reasoning folder里面也有一个，那个也是对的，实验是用那个跑的

Fixes for making H3 "prefill vs decode distribution" story solid:

1) Forced-choice scoring MUST be at the answer slot.
   - If you append answer_prefix to prompts, do NOT then generate warmup tokens *after* it.
     That moves you away from the answer slot and baseline accuracy collapses to ~chance.
   - Instead: (prompt_without_answer_prefix) -> (teacher-forced decode warmup tokens, optional)
     -> (teacher-force answer_prefix tokens) -> score candidates immediately.

2) Warmup should be TEACHER-FORCED (fixed tokens), not greedy-generated per condition.
   Greedy warmup diverges across interventions and makes the "decision probe" ill-defined.

3) Control condition should default to alpha=1 (dimension-matched). If you energy-match via alpha-scaling,
   large alpha (>1) can become an "over-subtraction" that is itself destructive.

This script runs an H3 grid:
  baseline(decode)
  decode-est / decode-int
  prefill-est / decode-int
  random-control / decode-int
(+ optional energy-match variants)

It also prints:
  - k_dec_shared, k_pre_shared, matched k
  - principal angles between bases (first k)
  - candidate tokenization sanity
  - baseline forced-choice sanity (should not be ~chance on CSQA)

Requires your project modules:
  - disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py
  - benchmark_dataloaders.py
  - joint_subspace_large.disturb_cross_task_all_shared.get_model_layers


Example:
    CUDA_VISIBLE_DEVICES=0 python h3_killer_counterfactual_grid_reasoning_v2.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --device cuda --model_dtype fp16 \
    --tasks gsm8k,commonsenseqa,strategyqa,piqa,arc_challenge,openbookqa,qasc,logiqa,boolq \
    --layer 10 --n_subspace 128 --n_eval 256 \
    --calib_decode_max_new_tokens 512 --per_task_max_states 20000 \
    --answer_prefix $'\nFinal answer:' \
    --warmup_tokens 0 \
    --template_randomization 1 --shuffle_choices 1


"""

from __future__ import annotations

import argparse
import math
import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

# ---- Import your existing pipeline bits ----
from disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8 import (
    HookStats,
    GenerationState,
    load_model_and_tokenizer,
    compute_shared_subspace_decode_aligned,
    register_hooks_for_condition,
    remove_hooks,
    bootstrap_ci_mean,
    orthonormalize_np,
)

# ---- Import task loading from benchmark_dataloaders ----
from benchmark_dataloaders import (
    Example,
    load_selected_tasks,
    is_correct,
    stable_int_seed,
)

from joint_subspace_large.disturb_cross_task_all_shared import get_model_layers


# -----------------------------
# Candidate sets for forced-choice
# -----------------------------
CHOICE_LABELS: Dict[str, List[str]] = {
    "commonsenseqa": list("ABCDE"),
    "aqua": list("ABCDE"),
    "arc_challenge": list("ABCD"),
    "openbookqa": list("ABCD"),
    "qasc": list("ABCDEFGH"),
    "logiqa": list("ABCD"),
    "piqa": list("AB"),
    "strategyqa": ["YES", "NO"],
    "boolq": ["YES", "NO"],
    # gsm8k is free-form; skip forced-choice
}


def candidate_texts_for_task(task: str) -> Tuple[List[str], List[str]]:
    labels = CHOICE_LABELS[task]
    if task in ["strategyqa", "boolq"]:
        # prefer lower-case tokens for Llama (often single token with leading space)
        texts = [" yes", " no"]
        labels = ["YES", "NO"]  # keep labels canonical
        return labels, texts
    # multiple-choice letters with leading space for stable tokenization
    texts = [f" {c}" for c in labels]
    return labels, texts


def split_at_answer_prefix(prompt: str, answer_prefix: str) -> Tuple[str, bool]:
    """
    If prompt already contains answer_prefix, remove the *last* occurrence.
    Returns (prompt_without_prefix, found_flag).
    """
    if not answer_prefix:
        return prompt, False
    idx = prompt.rfind(answer_prefix)
    if idx == -1:
        return prompt, False
    # Remove the suffix starting at idx
    return prompt[:idx], True


def principal_angles_deg(Qa: np.ndarray, Qb: np.ndarray) -> Dict[str, float]:
    """
    Compute principal angles between column-orthonormal bases Qa (d×k) and Qb (d×k').
    Returns summary in degrees.
    """
    if Qa.size == 0 or Qb.size == 0:
        return {"mean": float("nan"), "p50": float("nan"), "p95": float("nan")}
    M = Qa.T @ Qb
    # singular values are cos(theta)
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    theta = np.degrees(np.arccos(s))
    return {
        "mean": float(np.mean(theta)),
        "p50": float(np.percentile(theta, 50)),
        "p95": float(np.percentile(theta, 95)),
    }


# -----------------------------
# Prefill last-token state collection for basis estimation
# -----------------------------
class PrefillLastTokenCollector:
    def __init__(self):
        self.states: List[torch.Tensor] = []

    def __call__(self, module: torch.nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor):
        # output: [B, T, d]
        if not torch.is_tensor(output) or output.ndim != 3:
            return
        # collect last token only
        self.states.append(output[:, -1, :].detach())

    def pop_all(self) -> torch.Tensor:
        if not self.states:
            return torch.empty((0, 0))
        x = torch.cat(self.states, dim=0)
        self.states.clear()
        return x


@torch.no_grad()
def collect_prefill_lasttoken_states(
    model: torch.nn.Module,
    tokenizer,
    prompts: List[str],
    *,
    device: str,
    layer_idx: int,
    max_prompt_len: int,
    batch_size: int,
) -> np.ndarray:
    """
    Collect layer output (after block) at last token during prefill (seq_len>1).
    Returns numpy array [N, d].
    """
    model.eval()
    layers, _ = get_model_layers(model)
    if layer_idx < 0 or layer_idx >= len(layers):
        raise ValueError(f"layer_idx out of range: {layer_idx} vs {len(layers)}")

    collector = PrefillLastTokenCollector()
    handle = layers[layer_idx].register_forward_hook(collector)
    try:
        out_states: List[np.ndarray] = []
        for i in tqdm(range(0, len(prompts), batch_size), desc="CalibPrefill"):
            batch = prompts[i:i+batch_size]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=max_prompt_len,
                padding=True,
            ).to(device)
            _ = model(
                input_ids=enc["input_ids"],
                attention_mask=enc.get("attention_mask", None),
                use_cache=False,
            )
            st = collector.pop_all()
            if st.numel() == 0:
                continue
            out_states.append(st.float().cpu().numpy())
        if not out_states:
            return np.zeros((0, 0), dtype=np.float32)
        return np.concatenate(out_states, axis=0).astype(np.float32, copy=False)
    finally:
        try:
            handle.remove()
        except Exception:
            pass


def pooled_shared_basis_from_task_mats(
    mats: Dict[str, np.ndarray],
    *,
    pca_var: float,
    min_dim: int,
    max_dim: int,
    tau: float,
    m_shared: str,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Build shared basis from per-task matrices mats[task] = [n_t, d].
    Implements task-centering, balancing, pooled PCA, and sharedness criterion (relative variance >= tau).

    Returns:
      Q_shared: [d, k_shared] orthonormal
      extra: dict with k_pca, k_shared, shared_indices, r_stats, ...
    """
    rng = np.random.RandomState(seed)

    tasks = list(mats.keys())
    if not tasks:
        return np.zeros((0, 0), dtype=np.float32), {"k_pca": 0, "k_shared": 0}

    # balance to min count
    n_min = min(mats[t].shape[0] for t in tasks)
    d = mats[tasks[0]].shape[1]
    Xs = []
    X_task = {}
    for t in tasks:
        X = mats[t]
        if X.shape[1] != d:
            raise ValueError("Hidden dim mismatch across tasks")
        if X.shape[0] > n_min:
            idx = rng.choice(X.shape[0], size=n_min, replace=False)
            X = X[idx]
        # task-center
        mu = X.mean(axis=0, keepdims=True)
        Xc = X - mu
        X_task[t] = Xc.astype(np.float32, copy=False)
        Xs.append(Xc)
    X_pool = np.concatenate(Xs, axis=0).astype(np.float32, copy=False)

    # PCA via SVD on pooled matrix
    # X_pool: [N, d] with N <= tasks*n_min
    X_t = torch.from_numpy(X_pool).float()
    # center already, so compute SVD
    # Use CPU SVD to avoid GPU memory spikes
    U, S, Vh = torch.linalg.svd(X_t, full_matrices=False)
    # explained variance ratio from singular values: var_i ∝ S_i^2
    s2 = (S**2).cpu().numpy()
    total = float(np.sum(s2) + 1e-12)
    cumsum = np.cumsum(s2) / total

    k_pca = int(np.searchsorted(cumsum, pca_var) + 1)
    k_pca = max(k_pca, int(min_dim))
    k_pca = min(k_pca, int(max_dim), Vh.shape[0])
    Q = Vh[:k_pca, :].T.contiguous().cpu().numpy()  # [d, k_pca]

    # Per-task relative variance contributions r_{t,i}
    r_by_task: Dict[str, np.ndarray] = {}
    for t in tasks:
        Z = X_task[t] @ Q  # [n_min, k_pca]
        v = np.var(Z, axis=0, ddof=0)  # [k_pca]
        Vtot = float(np.sum(v) + 1e-12)
        r = v / Vtot
        r_by_task[t] = r

    # Determine m (task count) from m_shared string
    if m_shared == "all":
        m_req = len(tasks)
    elif m_shared == "half":
        m_req = max(1, len(tasks) // 2)
    else:
        # allow integer string
        try:
            m_req = int(m_shared)
        except Exception:
            m_req = len(tasks)

    # shared indices
    shared = []
    for i in range(k_pca):
        cnt = 0
        for t in tasks:
            if r_by_task[t][i] >= tau:
                cnt += 1
        if cnt >= m_req:
            shared.append(i)

    Qs = Q[:, shared] if shared else np.zeros((d, 0), dtype=np.float32)
    Qs = orthonormalize_np(Qs)

    extra = {
        "k_pca": int(k_pca),
        "k_shared": int(Qs.shape[1]),
        "shared_indices": [int(i) for i in shared],
        "m_req": int(m_req),
        "tasks": tasks,
    }
    return Qs, extra


# -----------------------------
# Decode-aligned forced-choice with teacher-forced warmup and answer prefix
# -----------------------------
@torch.no_grad()
def forced_choice_one(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    *,
    cand_texts: List[str],
    device: str,
    max_prompt_len: int,
    answer_prefix: str,
    warmup_ids: List[int],
) -> List[float]:
    """
    Score candidates at the answer slot:
      prompt_core -> (decode warmup tokens, teacher-forced) -> (teacher-force answer_prefix)
      -> score candidates immediately.

    Uses cache-advanced boundary alignment:
      - prefill prompt_core[:-1]
      - decode on last token of prompt_core to start cache-1 regime
    """
    model.eval()

    # tokenize prompt
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_prompt_len).to(device)
    input_ids = enc["input_ids"]  # [1, T]
    attn_mask = enc.get("attention_mask", None)
    if attn_mask is None:
        attn_mask = torch.ones_like(input_ids)

    eos = tokenizer.eos_token_id
    B, T = input_ids.shape
    assert B == 1

    # prefill + boundary decode to enter seq_len==1 regime
    if T >= 2:
        prefix_ids = input_ids[:, :-1]
        out0 = model(input_ids=prefix_ids, attention_mask=attn_mask[:, :-1], use_cache=True)
        past = out0.past_key_values
        last_id = input_ids[:, -1:]
        out1 = model(input_ids=last_id, attention_mask=attn_mask, past_key_values=past, use_cache=True)
        logits = out1.logits[:, -1, :]
        past = out1.past_key_values
        cur_attn = attn_mask
    else:
        out1 = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=True)
        logits = out1.logits[:, -1, :]
        past = out1.past_key_values
        cur_attn = attn_mask

    # teacher-forced warmup in decode regime
    for tid in warmup_ids:
        tid_t = torch.tensor([[tid]], device=device, dtype=torch.long)
        cur_attn = torch.cat([cur_attn, torch.ones((1, 1), device=device, dtype=cur_attn.dtype)], dim=1)
        outw = model(input_ids=tid_t, attention_mask=cur_attn, past_key_values=past, use_cache=True)
        logits = outw.logits[:, -1, :]
        past = outw.past_key_values

    # teacher-force answer_prefix to create a clean answer slot
    if answer_prefix:
        ap_ids = tokenizer(answer_prefix, add_special_tokens=False).input_ids
        for tid in ap_ids:
            tid_t = torch.tensor([[tid]], device=device, dtype=torch.long)
            cur_attn = torch.cat([cur_attn, torch.ones((1, 1), device=device, dtype=cur_attn.dtype)], dim=1)
            outa = model(input_ids=tid_t, attention_mask=cur_attn, past_key_values=past, use_cache=True)
            logits = outa.logits[:, -1, :]
            past = outa.past_key_values

    # score candidates (fast path: single-token)
    scores: List[float] = []
    logp = torch.log_softmax(logits.float(), dim=-1)[0]  # [V]
    for cand in cand_texts:
        cand_ids = tokenizer(cand, add_special_tokens=False).input_ids
        if len(cand_ids) == 1:
            scores.append(float(logp[cand_ids[0]].item()))
        else:
            # slow path: recompute from scratch for this candidate to avoid cache branching issues
            # (rare; typically labels are one token)
            score = 0.0
            # rebuild state
            score = score_multitok_candidate(
                model, tokenizer, prompt, cand_ids,
                device=device, max_prompt_len=max_prompt_len,
                answer_prefix=answer_prefix, warmup_ids=warmup_ids
            )
            scores.append(float(score))
    return scores


@torch.no_grad()
def score_multitok_candidate(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    cand_ids: List[int],
    *,
    device: str,
    max_prompt_len: int,
    answer_prefix: str,
    warmup_ids: List[int],
) -> float:
    """
    Safe but slower multi-token candidate scoring: recompute base cache then roll candidate tokens.
    """
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_prompt_len).to(device)
    input_ids = enc["input_ids"]
    attn_mask = enc.get("attention_mask", None)
    if attn_mask is None:
        attn_mask = torch.ones_like(input_ids)

    B, T = input_ids.shape
    assert B == 1

    if T >= 2:
        out0 = model(input_ids=input_ids[:, :-1], attention_mask=attn_mask[:, :-1], use_cache=True)
        past = out0.past_key_values
        out1 = model(input_ids=input_ids[:, -1:], attention_mask=attn_mask, past_key_values=past, use_cache=True)
        logits = out1.logits[:, -1, :]
        past = out1.past_key_values
        cur_attn = attn_mask
    else:
        out1 = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=True)
        logits = out1.logits[:, -1, :]
        past = out1.past_key_values
        cur_attn = attn_mask

    for tid in warmup_ids:
        tid_t = torch.tensor([[tid]], device=device, dtype=torch.long)
        cur_attn = torch.cat([cur_attn, torch.ones((1, 1), device=device, dtype=cur_attn.dtype)], dim=1)
        outw = model(input_ids=tid_t, attention_mask=cur_attn, past_key_values=past, use_cache=True)
        logits = outw.logits[:, -1, :]
        past = outw.past_key_values

    if answer_prefix:
        ap_ids = tokenizer(answer_prefix, add_special_tokens=False).input_ids
        for tid in ap_ids:
            tid_t = torch.tensor([[tid]], device=device, dtype=torch.long)
            cur_attn = torch.cat([cur_attn, torch.ones((1, 1), device=device, dtype=cur_attn.dtype)], dim=1)
            outa = model(input_ids=tid_t, attention_mask=cur_attn, past_key_values=past, use_cache=True)
            logits = outa.logits[:, -1, :]
            past = outa.past_key_values

    score = 0.0
    for j, tid in enumerate(cand_ids):
        logp = torch.log_softmax(logits.float(), dim=-1)[0, tid].item()
        score += float(logp)
        if j == len(cand_ids) - 1:
            break
        tid_t = torch.tensor([[tid]], device=device, dtype=torch.long)
        cur_attn = torch.cat([cur_attn, torch.ones((1, 1), device=device, dtype=cur_attn.dtype)], dim=1)
        outc = model(input_ids=tid_t, attention_mask=cur_attn, past_key_values=past, use_cache=True)
        logits = outc.logits[:, -1, :]
        past = outc.past_key_values
    return float(score)


@torch.no_grad()
def eval_forced_choice_decode(
    model: torch.nn.Module,
    tokenizer,
    examples: List[Example],
    *,
    device: str,
    max_prompt_len: int,
    answer_prefix: str,
    warmup_ids: List[int],
) -> Tuple[np.ndarray, float, float, float]:
    """
    Returns (correct_arr, acc, ci_low, ci_high).
    """
    correct: List[float] = []
    used = 0
    for ex in tqdm(examples, desc=f"ForcedChoice(decode,warmup)"):
        if ex.dataset not in CHOICE_LABELS:
            continue

        labels, cand_texts = candidate_texts_for_task(ex.dataset)

        # Ensure we score at answer slot: remove answer_prefix from prompt if it already exists
        core_prompt, _found = split_at_answer_prefix(ex.prompt, answer_prefix)

        scores = forced_choice_one(
            model, tokenizer, core_prompt,
            cand_texts=cand_texts,
            device=device,
            max_prompt_len=max_prompt_len,
            answer_prefix=answer_prefix,
            warmup_ids=warmup_ids,
        )
        pred = labels[int(np.argmax(np.asarray(scores, dtype=np.float64)))]
        correct.append(float(is_correct(ex.dataset, pred, ex.gold)))
        used += 1

    correct_arr = np.array(correct, dtype=np.float32)
    acc, lo, hi = bootstrap_ci_mean(correct_arr, iters=1000, alpha=0.05, seed=0)
    return correct_arr, float(acc), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--model_dtype", type=str, default="fp16", choices=["fp32", "fp16"])

    ap.add_argument("--tasks", type=str, default="gsm8k,commonsenseqa,strategyqa,piqa,arc_challenge,openbookqa,qasc,logiqa,boolq")
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--n_subspace", type=int, default=128)
    ap.add_argument("--n_eval", type=int, default=256)

    ap.add_argument("--calib_decode_max_new_tokens", type=int, default=512)
    ap.add_argument("--per_task_max_states", type=int, default=20000)

    ap.add_argument("--max_prompt_len", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=8)

    # H3 params
    ap.add_argument("--pca_var", type=float, default=0.95)
    ap.add_argument("--min_dim", type=int, default=16)
    ap.add_argument("--max_dim", type=int, default=4096)
    ap.add_argument("--tau", type=float, default=1e-3)
    ap.add_argument("--m_shared", type=str, default="all")

    ap.add_argument("--answer_prefix", type=str, default="\nFinal answer:")
    ap.add_argument("--warmup_tokens", type=int, default=0)
    ap.add_argument("--warmup_phrase", type=str, default="Let's think step by step.\n")

    ap.add_argument("--template_randomization", type=int, default=1)
    ap.add_argument("--shuffle_choices", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)

    # controls
    ap.add_argument("--alpha_remove", type=float, default=1.0)
    ap.add_argument("--report_energy_match", type=int, default=0)

    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    # 1) Load model/tokenizer
    model, tok = load_model_and_tokenizer(args.model, device=args.device, model_dtype=args.model_dtype)

    # 2) Load data (IMPORTANT: do NOT append answer_prefix here; we handle it in scoring)
    sub_by, eval_by, _meta = load_selected_tasks(
        tasks=tasks,
        n_subspace=args.n_subspace,
        n_eval=args.n_eval,
        seed=args.seed,
        template_seed=args.seed + 999,
        template_randomization=bool(args.template_randomization),
        shuffle_choices=bool(args.shuffle_choices),
        add_answer_prefix=False,
        answer_prefix=args.answer_prefix,
    )

    # 3) Build warmup token ids (teacher-forced, fixed)
    warmup_ids: List[int] = []
    if args.warmup_tokens > 0:
        base_ids = tok(args.warmup_phrase, add_special_tokens=False).input_ids
        if len(base_ids) == 0:
            base_ids = tok(" ", add_special_tokens=False).input_ids
        rep = (args.warmup_tokens + len(base_ids) - 1) // max(len(base_ids), 1)
        warmup_ids = (base_ids * rep)[: args.warmup_tokens]
    print(f"[Warmup] mode=teacher_forced_fixed W={len(warmup_ids)} tokens, phrase='{args.warmup_phrase.strip()}'")

    # 4) Compute decode-estimated shared basis
    prompts_by_task = {k: [ex.prompt for ex in sub_by[k]] for k in sub_by.keys()}
    joint_dec, shared_dec_idx, extra_dec, _ = compute_shared_subspace_decode_aligned(
        model=model,
        tokenizer=tok,
        prompts_by_task=prompts_by_task,
        layer_indices=[args.layer],
        calib_decoding="greedy",
        calib_batch_size=args.batch_size,
        calib_max_new_tokens=args.calib_decode_max_new_tokens,
        per_task_max_states=args.per_task_max_states,
        max_prompt_len=args.max_prompt_len,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        global_seed=args.seed,
        variance_threshold=args.pca_var,
        min_dim=args.min_dim,
        max_dim=args.max_dim,
        tau=args.tau,
        m_shared=args.m_shared,
    )
    Q_dec_full = orthonormalize_np(joint_dec[:, shared_dec_idx])
    print(f"[Shared-Decode] k_pca={extra_dec.get('cross_dim')} k_shared={Q_dec_full.shape[1]} (tau={args.tau}, m_shared={args.m_shared})")

    # 5) Compute prefill-estimated shared basis (prefill last token once per prompt)
    mats_pre: Dict[str, np.ndarray] = {}
    for t in tasks:
        prompts = [ex.prompt for ex in sub_by[t]]
        X = collect_prefill_lasttoken_states(
            model=model,
            tokenizer=tok,
            prompts=prompts,
            device=args.device,
            layer_idx=args.layer,
            max_prompt_len=args.max_prompt_len,
            batch_size=args.batch_size,
        )
        mats_pre[t] = X
        print(f"[PrefillStates] {t}: {tuple(X.shape)}")
    Q_pre_full, extra_pre = pooled_shared_basis_from_task_mats(
        mats_pre,
        pca_var=args.pca_var,
        min_dim=args.min_dim,
        max_dim=args.max_dim,
        tau=args.tau,
        m_shared=args.m_shared,
        seed=args.seed + 1234,
    )
    print(f"[Shared-Prefill] k_pca={extra_pre.get('k_pca')} k_shared={Q_pre_full.shape[1]} (tau={args.tau}, m_shared={args.m_shared})")

    # 6) Match dimension
    k = min(Q_dec_full.shape[1], Q_pre_full.shape[1])
    if k <= 0:
        raise RuntimeError("Matched k is 0; cannot run H3 grid.")
    Q_dec = Q_dec_full[:, :k]
    Q_pre = Q_pre_full[:, :k]
    Q_ctrl = orthonormalize_np(np.random.RandomState(args.seed + 2026).randn(Q_dec.shape[0], k).astype(np.float32))

    ang = principal_angles_deg(Q_dec, Q_pre)
    print(f"[Match] k = min(k_dec, k_pre) = {k}")
    print(f"[Angles] mean={ang['mean']:.2f}° p50={ang['p50']:.2f}° p95={ang['p95']:.2f}°")

    # 7) Candidate tokenization sanity
    for t in ["commonsenseqa", "arc_challenge", "piqa", "boolq"]:
        if t in CHOICE_LABELS:
            _lbl, cand = candidate_texts_for_task(t)
            lens = [len(tok(c, add_special_tokens=False).input_ids) for c in cand]
            print(f"[CandTok] {t}: {list(zip(cand, lens))}")

    # 8) Evaluate tasks (forced-choice only)
    def run_cond(examples: List[Example], Q: Optional[np.ndarray], name: str) -> Dict[str, Any]:
        if Q is None:
            handles, _setter, _stats = ([], None, [])
        else:
            handles, _setter, _stats = register_hooks_for_condition(
                model=model,
                layer_indices=[args.layer],
                Q_np=Q,
                condition="full",  # full decode removal
                alpha=args.alpha_remove,
                reasoning_token_threshold=10**9,  # disable staged gating
            )
        try:
            corr, acc, lo, hi = eval_forced_choice_decode(
                model, tok, examples,
                device=args.device,
                max_prompt_len=args.max_prompt_len,
                answer_prefix=args.answer_prefix,
                warmup_ids=warmup_ids,
            )
            return {"name": name, "acc": acc, "ci_low": lo, "ci_high": hi, "correct": corr.tolist()}
        finally:
            remove_hooks(handles)

    results: Dict[str, Any] = {
        "model": args.model,
        "layer": args.layer,
        "k_match": k,
        "angles_deg": ang,
        "tasks": {},
    }

    for t in tasks:
        if t not in CHOICE_LABELS:
            print(f"[Skip] {t}: not a discrete-choice task")
            continue
        exs = eval_by[t]
        print("\n" + "=" * 90)
        print(f"[H3-Grid v2 | ForcedChoice+TeacherWarmup] {t} (n={len(exs)}, W={len(warmup_ids)})")
        print("=" * 90)

        r_base = run_cond(exs, None, "baseline(decode)")
        r_dec = run_cond(exs, Q_dec, "decode-est/decode-int")
        r_pre = run_cond(exs, Q_pre, "prefill-est/decode-int")
        r_ctl = run_cond(exs, Q_ctrl, "control-rand/decode-int")

        def pct(x: float) -> float:
            return 100.0 * x

        print(f"  {r_base['name']:<28}: {pct(r_base['acc']):5.1f} [{pct(r_base['ci_low']):.1f},{pct(r_base['ci_high']):.1f}]")
        print(f"  {r_dec['name']:<28}: {pct(r_dec['acc']):5.1f} [{pct(r_dec['ci_low']):.1f},{pct(r_dec['ci_high']):.1f}]")
        print(f"  {r_pre['name']:<28}: {pct(r_pre['acc']):5.1f} [{pct(r_pre['ci_low']):.1f},{pct(r_pre['ci_high']):.1f}]")
        print(f"  {r_ctl['name']:<28}: {pct(r_ctl['acc']):5.1f} [{pct(r_ctl['ci_low']):.1f},{pct(r_ctl['ci_high']):.1f}]")

        results["tasks"][t] = {
            "baseline": r_base,
            "decode": r_dec,
            "prefill": r_pre,
            "control": r_ctl,
        }

    out = f"h3_grid_v2_{args.model.replace('/','_')}_layer{args.layer}_k{k}_W{len(warmup_ids)}_seed{args.seed}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Done] wrote {out}")


if __name__ == "__main__":
    main()
