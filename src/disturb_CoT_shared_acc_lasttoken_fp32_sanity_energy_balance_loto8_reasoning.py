# -*- coding: utf-8 -*-
"""
disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8_reasoning.py

NOTE: n_eval should be 2048, and use_forced_choice should be 1. 这个代码在增加n_eval和use_forced_choice之后，结果就对了。

This is a *reasoning + LOTO* script with **generation** evaluation (as in your original)
AND an improved **forced-choice** evaluation path for MC/YesNo tasks.

Why this file exists:
  - In many CoT benchmarks, free-form generation + parsing can be noisy (format sensitivity).
  - Forced-choice logprob is often a cleaner metric BUT it is very easy to accidentally
    evaluate at the wrong "decision point" (e.g., after warmup tokens without re-anchoring
    with an answer prefix), producing chance-level baselines.

This version improves the LOTO script's forced-choice so results "make sense":
  1) Decode-aligned prompt boundary for evaluation:
        process prompt[:-1] as prefill, then prompt[-1:] as a decode call (seq_len==1).
     This ensures decode-only hooks (your interventions) can affect the *first* decision logits.
  2) Optional forced-choice warmup tokens (teacher-forced, shared across conditions).
  3) Robust answer-prefix policy:
        --fc_prefix_mode auto|always|never
     Default "auto" makes forced-choice *hard to accidentally break*:
        - if warmup_tokens > 0 => ALWAYS add fc_answer_prefix after warmup
        - else add prefix only if the prompt doesn't already end with it
  4) Forced-choice correctness uses benchmark_dataloaders.is_correct() for consistency.

Notes:
  - gsm8k stays generation-only (numeric free-form).
  - For MC tasks, you can choose protocol with --use_forced_choice 1 (recommended for debugging).
  - This script can import from your existing benchmark_dataloaders (no duplication).

Example (LOTO heldout + forced-choice on MC tasks):
  CUDA_VISIBLE_DEVICES=3 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8_reasoning.py \
    --model meta-llama/Llama-2-7b-chat-hf --device cuda --model_dtype fp32 \
    --mode loto --loto_eval_mode heldout \
    --tasks gsm8k,commonsenseqa,strategyqa,aqua,arc_challenge,openbookqa,qasc,logiqa \
    --layer 10 --n_subspace 128 --n_eval 2048 \
    --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
    --reasoning_tokens 128 --max_new_tokens 256 \
    --template_randomization 1 --shuffle_choices 1 \
    --add_answer_prefix 1 --answer_prefix $'\nFinal answer:' \
    --use_forced_choice 1 \
    --fc_warmup_tokens 0 \
    --fc_prefix_mode auto --fc_answer_prefix $'\nFinal answer:' \
    --do_sample 0 \
    --out_json energy_balance_loto8_reasoning_fc_eval2048.json --out_md energy_balance_loto8_reasoning_fc_eval2048.md

If you want the forced-choice "deep decision point" style:
  --fc_warmup_tokens 128 --fc_prefix_mode auto   # auto will re-anchor with answer prefix after warmup

"""

import os
import sys
import re
import json
import math
import random
import argparse
from typing import Dict, List, Tuple, Optional, Any, DefaultDict
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------
# Import your shared-subspace utilities (from your project)
# ---------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(THIS_DIR, ".."))

from joint_subspace_large.disturb_cross_task_all_shared import (  # noqa: E402
    get_model_layers,
    compute_cross_task_subspace,
    find_fully_shared_basis_improved,
)

# ---------------------------------------------------------------------
# Import data loading and evaluation utilities from benchmark_dataloaders
# ---------------------------------------------------------------------
# You can keep your local benchmark_dataloaders.py in the same dir, or on PYTHONPATH.
# This script uses Example/load_selected_tasks/parse_prediction/is_correct/stable_int_seed.
try:
    from benchmark_dataloaders import *  # noqa: F401,F403
    from benchmark_dataloaders import (  # noqa: E402
        stable_int_seed as stable_int_seed_bdl,
        is_correct as is_correct_bool,
    )
except Exception:
    # Fallback: some users keep renamed copies (e.g., benchmark_dataloaders_aqua_prefix_default.py)
    try:
        from benchmark_dataloaders_aqua_prefix_default import *  # type: ignore  # noqa: F401,F403
        from benchmark_dataloaders_aqua_prefix_default import (  # type: ignore  # noqa: E402
            stable_int_seed as stable_int_seed_bdl,
            is_correct as is_correct_bool,
        )
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Failed to import benchmark_dataloaders.\n"
            "Put benchmark_dataloaders.py next to this script, or ensure it's on PYTHONPATH.\n"
            "As a fallback, we also try benchmark_dataloaders_aqua_prefix_default.py."
        ) from e


# -----------------------------
# Repro / utils
# -----------------------------
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Use stable_int_seed from benchmark_dataloaders (keeps consistency with your loaders)
stable_int_seed = stable_int_seed_bdl


def json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return str(o)


def _norm_prefix_arg(x: Any) -> str:
    """Treat '0'/'none'/'null' as empty prefix (helps with CLI habits)."""
    if x is None:
        return ""
    s = str(x)
    if s.strip().lower() in {"0", "none", "null", "false"}:
        return ""
    return s


# Wrapper to convert is_correct bool to int for compatibility
def is_correct(dataset: str, pred: str, gold: str) -> int:
    return int(is_correct_bool(dataset, pred, gold))


# -----------------------------
# Stats: bootstrap + paired test
# -----------------------------
def bootstrap_ci_mean(values: np.ndarray, iters: int, alpha: float, seed: int) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    m = float(values.mean())
    boots = []
    for _ in range(iters):
        idx = rng.integers(0, n, size=n)
        boots.append(float(values[idx].mean()))
    lo = float(np.percentile(boots, 100 * (alpha / 2)))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return m, lo, hi


def paired_bootstrap_ci_diff(baseline: np.ndarray, treatment: np.ndarray, iters: int, alpha: float, seed: int) -> Tuple[float, float, float]:
    assert baseline.shape == treatment.shape
    diffs = treatment - baseline
    obs = float(diffs.mean())
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boots = []
    for _ in range(iters):
        idx = rng.integers(0, n, size=n)
        boots.append(float(diffs[idx].mean()))
    lo = float(np.percentile(boots, 100 * (alpha / 2)))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return obs, lo, hi


def signflip_permutation_test(baseline: np.ndarray, treatment: np.ndarray, iters: int, seed: int) -> float:
    """Two-sided sign-flip permutation test on paired diffs."""
    assert baseline.shape == treatment.shape
    diffs = treatment - baseline
    obs = float(diffs.mean())
    rng = np.random.default_rng(seed)
    n = len(diffs)
    if n == 0:
        return float("nan")
    count = 0
    for _ in range(iters):
        signs = rng.choice([-1.0, 1.0], size=n)
        perm = float((diffs * signs).mean())
        if abs(perm) >= abs(obs):
            count += 1
    return float((count + 1) / (iters + 1))


def summarize_paired(
    baseline_correct: np.ndarray,
    treat_correct: np.ndarray,
    label: str,
    bootstrap_iters: int,
    perm_iters: int,
    alpha: float,
    seed: int,
) -> Dict[str, Any]:
    md, lo, hi = paired_bootstrap_ci_diff(
        baseline_correct, treat_correct,
        iters=bootstrap_iters, alpha=alpha,
        seed=seed + 123
    )
    p = signflip_permutation_test(
        baseline_correct, treat_correct,
        iters=perm_iters, seed=seed + 456
    )
    return {
        "label": label,
        "mean_diff": md,      # treatment - baseline
        "ci_low": lo,
        "ci_high": hi,
        "p_value": p,
    }


def fmt_acc(acc: float, lo: float, hi: float) -> str:
    return f"{acc*100:.1f} [{lo*100:.1f}, {hi*100:.1f}]"


# -----------------------------
# Decoding utilities (top-p/top-k)
# -----------------------------
def top_p_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p <= 0.0 or top_p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumprobs = torch.cumsum(probs, dim=-1)
    mask = cumprobs > top_p
    mask[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
    filtered = torch.full_like(logits, float("-inf"))
    filtered.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)
    return filtered


def top_k_filtering(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k is None or top_k <= 0:
        return logits
    top_k = min(top_k, logits.size(-1))
    values, _ = torch.topk(logits, top_k, dim=-1)
    min_values = values[:, -1].unsqueeze(-1)
    return torch.where(logits < min_values, torch.full_like(logits, float("-inf")), logits)


def _choose_next_token(
    logits: torch.Tensor,
    *,
    decoding: str,
    temperature: float,
    top_p: float,
    top_k: int,
    eos_token_id: int,
    ban_eos: bool,
) -> torch.Tensor:
    """Return next token ids [B,1]."""
    assert decoding in ["greedy", "sample"]
    if ban_eos:
        logits = logits.clone()
        logits[:, eos_token_id] = float("-inf")

    if decoding == "greedy":
        return torch.argmax(logits, dim=-1, keepdim=True)

    lt = logits / max(temperature, 1e-6)
    lt = top_k_filtering(lt, top_k=top_k)
    lt = top_p_filtering(lt, top_p=top_p)
    probs = torch.softmax(lt, dim=-1)
    return torch.multinomial(probs, num_samples=1)


# -----------------------------
# Decode-aligned prompt boundary helper
# -----------------------------
@torch.no_grad()
def _cache_advanced_prompt_boundary(model, ids: torch.Tensor, attn: torch.Tensor):
    """
    Compute (past, logits_next) such that the last prompt token is processed with seq_len==1.

    This matters because your interventions are decode-only (seq_len==1). Without this,
    the *first* next-token logits after the prompt would be produced by a prefill call
    and would NOT be affected by interventions.
    """
    if ids.ndim != 2:
        raise ValueError(f"ids must be 2D [B,T], got {ids.shape}")
    _, T = ids.shape
    if T == 0:
        raise ValueError("Empty prompt")
    if T == 1:
        out1 = model(input_ids=ids, attention_mask=attn, use_cache=True)
        return out1.past_key_values, out1.logits[:, -1, :]

    out0 = model(input_ids=ids[:, :-1], attention_mask=attn[:, :-1], use_cache=True)
    out1 = model(
        input_ids=ids[:, -1:],
        attention_mask=attn,
        use_cache=True,
        past_key_values=out0.past_key_values,
    )
    return out1.past_key_values, out1.logits[:, -1, :]


# -----------------------------
# Decode last-token activation collector (A3)
# -----------------------------
class DecodeLastTokenActivationCollector:
    """
    Collect last-token hidden states ONLY during decode forward passes (seq_len==1).
    storage[task][layer_idx] -> list of np arrays [B', D]
    """
    def __init__(self, layer_indices: List[int]):
        self.layer_indices = list(layer_indices)
        self._cur_task: Optional[str] = None
        self.capture_enabled: bool = False
        self.active_mask: Optional[torch.Tensor] = None
        self.storage: DefaultDict[str, DefaultDict[int, List[np.ndarray]]] = defaultdict(lambda: defaultdict(list))

    def set_current_task(self, task_name: str) -> None:
        self._cur_task = task_name

    def set_capture(self, enabled: bool, active_mask: Optional[torch.Tensor] = None) -> None:
        self.capture_enabled = bool(enabled)
        self.active_mask = active_mask

    def make_hook(self, layer_idx: int):
        def _hook(module, inputs, output):
            if (not self.capture_enabled) or (self._cur_task is None):
                return output
            hs = output[0] if isinstance(output, tuple) else output
            if not isinstance(hs, torch.Tensor) or hs.ndim != 3:
                return output
            if hs.shape[1] != 1:
                return output
            x = hs[:, -1, :]  # [B, D]
            if self.active_mask is not None:
                m = self.active_mask.bool()
                if m.numel() == x.shape[0]:
                    x = x[m]
            if x.numel() == 0:
                return output
            self.storage[self._cur_task][layer_idx].append(x.detach().float().cpu().numpy())
            return output
        return _hook

    def get_task_activations(self, task: str, layer_idx: int) -> Optional[np.ndarray]:
        chunks = self.storage.get(task, {}).get(layer_idx, [])
        if not chunks:
            return None
        return np.concatenate(chunks, axis=0)


def _subsample_rows_np(x: np.ndarray, n_max: int, seed: int) -> np.ndarray:
    if n_max is None or n_max <= 0 or x.shape[0] <= n_max:
        return x
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.shape[0], size=n_max, replace=False)
    return x[idx]


@torch.no_grad()
def collect_decode_last_token_states(
    model,
    tokenizer,
    prompts: List[str],
    collector: DecodeLastTokenActivationCollector,
    batch_size: int,
    max_new_tokens: int,
    decoding: str,
    temperature: float,
    top_p: float,
    top_k: int,
    max_prompt_len: int,
) -> None:
    assert decoding in ["greedy", "sample"]
    device = next(model.parameters()).device
    eos = tokenizer.eos_token_id
    model.eval()

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for i in tqdm(range(0, len(prompts), batch_size), desc="CollectDecode"):
        # batch = prompts[i:i+batch_size]
        # inputs = tokenizer(
        #     batch,
        #     return_tensors="pt",
        #     padding=True,
        #     truncation=True,
        #     max_length=max_prompt_len,
        # ).to(device)
        use_template = bool(getattr(tokenizer, "chat_template", None))

        batch = prompts[i:i+batch_size]
        batch = [render_prompt(tokenizer, p, add_generation_prompt=True) for p in batch]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_len,
            add_special_tokens=not use_template,
        ).to(device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        B, _T0 = input_ids.shape

        unfinished = torch.ones(B, dtype=torch.bool, device=device)

        # Decode-aligned boundary: process prompt last token via seq_len==1 call
        collector.set_capture(False, None)
        past, logits = _cache_advanced_prompt_boundary(model, input_ids, attention_mask)

        for _ in range(max_new_tokens):
            next_token = _choose_next_token(
                logits,
                decoding=decoding,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                eos_token_id=eos,
                ban_eos=False,
            )

            next_token = torch.where(
                unfinished.unsqueeze(-1),
                next_token,
                torch.full_like(next_token, eos),
            )

            unfinished = unfinished & (next_token.squeeze(-1) != eos)
            if not bool(unfinished.any().item()):
                break

            attention_mask = torch.cat(
                [attention_mask, torch.ones((B, 1), device=attention_mask.device, dtype=attention_mask.dtype)],
                dim=1,
            )

            collector.set_capture(True, unfinished)
            out = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                use_cache=True,
                past_key_values=past,
            )
            logits = out.logits[:, -1, :]
            past = out.past_key_values

        collector.set_capture(False, None)


def render_prompt(tokenizer, user_prompt: str, *, add_generation_prompt: bool = True, system_prompt: str | None = None):
    tmpl = getattr(tokenizer, "chat_template", None)
    if not tmpl:
        return user_prompt

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    except Exception:
        # 常见：Gemma-7b-it 报 system role unsupported
        messages = [{"role": "user", "content": user_prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)


# -----------------------------
# Basis utilities
# -----------------------------
def orthonormalize_np(basis: np.ndarray) -> np.ndarray:
    q, _ = np.linalg.qr(basis.astype(np.float32, copy=False))
    return q.astype(np.float32, copy=False)


def max_offdiag(Q: np.ndarray) -> float:
    G = Q.T @ Q
    k = G.shape[0]
    G = G - np.eye(k, dtype=G.dtype)
    return float(np.max(np.abs(G))) if k > 0 else 0.0


def max_overlap(Qa: np.ndarray, Qb: np.ndarray) -> float:
    if Qa.size == 0 or Qb.size == 0:
        return 0.0
    M = Qa.T @ Qb
    return float(np.max(np.abs(M)))


def energy_ratio_stats(states: np.ndarray, Q: np.ndarray) -> Dict[str, float]:
    eps = 1e-12
    H = states.astype(np.float32, copy=False)
    proj = H @ Q
    num = np.sum(proj * proj, axis=1)
    den = np.sum(H * H, axis=1) + eps
    r = num / den
    return {"mean": float(np.mean(r)), "p50": float(np.percentile(r, 50)), "p95": float(np.percentile(r, 95))}


def infer_component_variances(contributions: Dict[str, Any], tasks: List[str], cross_dim: int) -> np.ndarray:
    candidates = []
    for t in tasks:
        d = contributions.get(t, {})
        v = None
        for key in ["variances", "component_variances", "per_component_variance", "var", "vars"]:
            if key in d:
                v = np.asarray(d[key], dtype=np.float64)
                break
        if v is None:
            for _, val in d.items():
                if isinstance(val, (list, np.ndarray)) and len(val) >= cross_dim:
                    vv = np.asarray(val, dtype=np.float64)
                    if vv.ndim == 1:
                        v = vv
                        break
        if v is None or v.ndim != 1 or v.shape[0] < cross_dim:
            raise KeyError(f"Cannot infer per-component variances for task={t}. keys={list(d.keys())}")
        candidates.append(v[:cross_dim])
    pooled = np.mean(np.stack(candidates, axis=0), axis=0)
    return pooled


def select_rand_indices(
    rand_type: str,
    cross_dim: int,
    shared_indices: List[int],
    pooled_var: Optional[np.ndarray],
    k: int,
    seed: int,
) -> List[int]:
    rng = np.random.default_rng(seed)
    shared_set = set(shared_indices)
    nonshared = [i for i in range(cross_dim) if i not in shared_set]
    if len(nonshared) < k:
        raise RuntimeError(f"Not enough nonshared components: nonshared={len(nonshared)} < k={k}")

    if rand_type == "joint_nonshared_uniform" or pooled_var is None:
        return list(rng.choice(nonshared, size=k, replace=False))

    if rand_type == "joint_nonshared_topk":
        idx_sorted = sorted(nonshared, key=lambda i: pooled_var[i], reverse=True)
        return idx_sorted[:k]

    if rand_type == "joint_nonshared_varmatch":
        shared_vars = [(i, pooled_var[i]) for i in shared_indices]
        shared_vars.sort(key=lambda x: x[1])
        nonshared_sorted = sorted(nonshared, key=lambda i: pooled_var[i])
        nonshared_vals = [pooled_var[i] for i in nonshared_sorted]
        import bisect
        chosen = []
        for _, v in shared_vars:
            j = bisect.bisect_left(nonshared_vals, v)
            cand_pos = []
            if 0 <= j < len(nonshared_sorted):
                cand_pos.append(j)
            if 0 <= j - 1 < len(nonshared_sorted):
                cand_pos.append(j - 1)
            best = None
            best_d = None
            for p in cand_pos:
                d = abs(nonshared_vals[p] - v)
                if best is None or d < best_d - 1e-12 or (abs(d - best_d) < 1e-12 and rng.random() < 0.5):
                    best = p
                    best_d = d
            if best is None:
                best = rng.integers(0, len(nonshared_sorted))
            chosen_idx = nonshared_sorted.pop(best)
            nonshared_vals.pop(best)
            chosen.append(chosen_idx)
            if len(chosen) >= k:
                break
        if len(chosen) < k:
            remaining = nonshared_sorted
            extra = list(rng.choice(remaining, size=(k - len(chosen)), replace=False))
            chosen.extend(extra)
        return chosen

    raise ValueError(f"Unknown rand_type={rand_type}")


@torch.no_grad()
def compute_shared_subspace_decode_aligned(
    model,
    tokenizer,
    prompts_by_task: Dict[str, List[str]],
    layer_indices: List[int],
    *,
    calib_decoding: str,
    calib_batch_size: int,
    calib_max_new_tokens: int,
    per_task_max_states: int,
    max_prompt_len: int,
    temperature: float,
    top_p: float,
    top_k: int,
    global_seed: int,
    variance_threshold: float,
    min_dim: int,
    max_dim: int,
    tau: float,
    m_shared: str,
) -> Tuple[np.ndarray, List[int], Dict[str, Any], Dict[str, Dict[int, np.ndarray]]]:

    print("\n" + "=" * 80)
    print("[Subspace-A3] Collecting DECODE last-token activations for shared subspace estimation ...")
    print(f"[Subspace-A3] calib_decoding={calib_decoding}, max_new_tokens={calib_max_new_tokens}, per_task_max_states={per_task_max_states}")
    print("=" * 80)

    layers, _ = get_model_layers(model)
    collector = DecodeLastTokenActivationCollector(layer_indices)

    handles = []
    for layer_idx in layer_indices:
        if layer_idx >= len(layers):
            print(f"[Subspace-A3] Warn: layer_idx={layer_idx} out of range, skipping")
            continue
        handles.append(layers[layer_idx].register_forward_hook(collector.make_hook(layer_idx)))

    try:
        for task_name, prompts in prompts_by_task.items():
            print(f"[Subspace-A3] Task={task_name}, prompts={len(prompts)}")
            collector.set_current_task(task_name)
            collect_decode_last_token_states(
                model,
                tokenizer,
                prompts=prompts,
                collector=collector,
                batch_size=calib_batch_size,
                max_new_tokens=calib_max_new_tokens,
                decoding=calib_decoding,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_prompt_len=max_prompt_len,
            )
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass
        collector.set_capture(False, None)

    task_activations: Dict[str, Dict[int, np.ndarray]] = {}
    for task_name in prompts_by_task.keys():
        layer_dict = {}
        for layer_idx in layer_indices:
            acts = collector.get_task_activations(task_name, layer_idx)
            if acts is None or acts.shape[0] == 0:
                continue
            ss = stable_int_seed(global_seed, task_name, layer_idx, "subsample")
            acts = _subsample_rows_np(acts, per_task_max_states, seed=ss)
            layer_dict[layer_idx] = acts
            print(f"[Subspace-A3]  collected {task_name} layer={layer_idx}: {acts.shape[0]} x {acts.shape[1]}")
        if layer_dict:
            task_activations[task_name] = layer_dict

    if not task_activations:
        raise RuntimeError("[Subspace-A3] No decode activations collected. Check hooks/layers/generation loop.")

    joint_subspace, cross_dim, contributions, full_pca_info = compute_cross_task_subspace(
        task_activations,
        variance_threshold=variance_threshold,
        min_dim=min_dim,
        max_dim=max_dim,
        return_full_pca=True,
    )
    if joint_subspace is None or cross_dim <= 0:
        raise RuntimeError("[Subspace-A3] Failed to compute cross-task subspace.")

    tasks = list(task_activations.keys())

    # m_shared: "all" or int string (>=2)
    if m_shared == "all":
        min_tasks = len(tasks)
    else:
        try:
            min_tasks = max(2, int(m_shared))
        except Exception:
            min_tasks = len(tasks)

    shared_indices = find_fully_shared_basis_improved(
        contributions,
        tasks,
        cross_dim,
        min_tasks_shared=min_tasks,
        relative_threshold=tau,
        top_k_components=cross_dim,
    )

    if not shared_indices and min_tasks != 2:
        # fallback to >=2 tasks shared
        print("[Subspace-A3] No shared basis for requested m_shared; falling back to min_tasks_shared=2.")
        shared_indices = find_fully_shared_basis_improved(
            contributions,
            tasks,
            cross_dim,
            min_tasks_shared=2,
            relative_threshold=tau,
            top_k_components=cross_dim,
        )

    print(f"[Subspace-A3] cross_dim={cross_dim}, shared_basis_count={len(shared_indices)} (m_shared={m_shared}, tau={tau})")
    extra = {
        "cross_dim": int(cross_dim),
        "tasks_used": tasks,
        "task_contributions": contributions,
        "full_pca_info": full_pca_info,
        "calib": {
            "calib_decoding": calib_decoding,
            "calib_max_new_tokens": calib_max_new_tokens,
            "per_task_max_states": per_task_max_states,
        },
        "m_shared": m_shared,
        "tau": float(tau),
    }
    return joint_subspace.astype(np.float32, copy=False), shared_indices, extra, task_activations


# -----------------------------
# Intervention hooks (last-token removal + staged)
# -----------------------------
class GenerationState:
    def __init__(self, batch_size: int, device: torch.device, reasoning_threshold: int):
        self.batch_size = batch_size
        self.device = device
        self.reasoning_threshold = int(reasoning_threshold)
        self.unfinished = torch.ones(batch_size, dtype=torch.bool, device=device)
        self.gen_steps = torch.zeros(batch_size, dtype=torch.long, device=device)

    def current_reasoning_mask(self) -> torch.Tensor:
        return self.unfinished & (self.gen_steps < self.reasoning_threshold)

    def step_update(self, next_tokens: torch.Tensor, eos_token_id: int) -> None:
        next_tokens = next_tokens.squeeze(-1)
        active = self.unfinished.clone()
        self.gen_steps[active] += 1
        newly_finished = active & (next_tokens == eos_token_id)
        self.unfinished[newly_finished] = False

    def clone(self) -> "GenerationState":
        st = GenerationState(self.batch_size, self.device, self.reasoning_threshold)
        st.unfinished = self.unfinished.clone()
        st.gen_steps = self.gen_steps.clone()
        return st


class HookStats:
    def __init__(self, name: str):
        self.name = name
        self.decode_calls = 0
        self.intervened = 0


class LastTokenRemovalHook:
    def __init__(self, Q_np: np.ndarray, alpha: float, stats: HookStats):
        self.alpha = float(alpha)
        self.stats = stats
        self.Q = torch.tensor(orthonormalize_np(Q_np), dtype=torch.float32)
        self.Q_device: Optional[torch.Tensor] = None

    def _Q(self, device: torch.device) -> torch.Tensor:
        if self.Q_device is None or self.Q_device.device != device:
            self.Q_device = self.Q.to(device=device)
        return self.Q_device

    def __call__(self, module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        if not isinstance(hs, torch.Tensor) or hs.ndim != 3:
            return output
        if hs.shape[1] != 1:
            return output

        self.stats.decode_calls += 1

        Q = self._Q(hs.device)
        x = hs[:, -1, :].float()
        proj = (x @ Q) @ Q.T
        hs2 = hs.clone()
        hs2[:, -1, :] = (x - self.alpha * proj).to(dtype=hs.dtype)

        self.stats.intervened += 1
        if isinstance(output, tuple):
            return (hs2,) + output[1:]
        return hs2


class LastTokenStagedRemovalHook(LastTokenRemovalHook):
    def __init__(self, Q_np: np.ndarray, alpha: float, stats: HookStats, reasoning_threshold: int):
        super().__init__(Q_np, alpha, stats)
        self.state: Optional[GenerationState] = None
        self.reasoning_threshold = int(reasoning_threshold)

    def set_state(self, st: Optional[GenerationState]) -> None:
        self.state = st

    def __call__(self, module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        if not isinstance(hs, torch.Tensor) or hs.ndim != 3:
            return output
        if hs.shape[1] != 1:
            return output

        self.stats.decode_calls += 1

        if self.state is None:
            return super().__call__(module, inputs, output)

        mask = self.state.current_reasoning_mask()
        if not bool(mask.any().item()):
            return output

        Q = self._Q(hs.device)
        x = hs[:, -1, :].float()
        x_sel = x[mask]
        proj = (x_sel @ Q) @ Q.T
        x[mask] = x_sel - self.alpha * proj

        hs2 = hs.clone()
        hs2[:, -1, :] = x.to(dtype=hs.dtype)

        self.stats.intervened += 1
        if isinstance(output, tuple):
            return (hs2,) + output[1:]
        return hs2


# -----------------------------
# Hook registration
# -----------------------------
def register_hooks_for_condition(
    model,
    layer_indices: List[int],
    Q_np: Optional[np.ndarray],
    condition: str,   # "baseline" | "full" | "staged"
    alpha: float,
    reasoning_token_threshold: int,
) -> Tuple[List[Any], Optional[Any], List[HookStats]]:
    assert condition in ["baseline", "full", "staged"]
    if condition == "baseline":
        return [], None, []

    assert Q_np is not None
    layers, _ = get_model_layers(model)

    handles = []
    staged_hooks: List[LastTokenStagedRemovalHook] = []
    hook_stats: List[HookStats] = []

    for layer_idx in layer_indices:
        if layer_idx >= len(layers):
            print(f"[Warn] layer_idx={layer_idx} out of range, skipping")
            continue

        if condition == "full":
            stats = HookStats(name=f"full@{layer_idx}")
            hk = LastTokenRemovalHook(Q_np, alpha=alpha, stats=stats)
            handles.append(layers[layer_idx].register_forward_hook(hk))
            hook_stats.append(stats)
        else:
            stats = HookStats(name=f"staged@{layer_idx}")
            hk = LastTokenStagedRemovalHook(Q_np, alpha=alpha, stats=stats, reasoning_threshold=reasoning_token_threshold)
            staged_hooks.append(hk)
            handles.append(layers[layer_idx].register_forward_hook(hk))
            hook_stats.append(stats)

    def setter(state_or_none: Optional[GenerationState]) -> None:
        for hk in staged_hooks:
            hk.set_state(state_or_none)

    return handles, (setter if condition == "staged" else None), hook_stats


def remove_hooks(handles: List[Any]) -> None:
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass


# -----------------------------
# Generation + per-example stats (decode-aligned evaluation)
# -----------------------------
@torch.no_grad()
def generate_continuations(
    model,
    tokenizer,
    prompts: List[str],
    decoding: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    device: str,
    batch_size: int,
    max_prompt_len: int,
    reasoning_token_threshold: int,
    state_setter: Optional[Any] = None,
    sample_seed: Optional[int] = None,
):
    assert decoding in ["greedy", "sample"]
    model.eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos = tokenizer.eos_token_id

    if decoding == "sample" and sample_seed is not None:
        torch.manual_seed(sample_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(sample_seed)

    continuations: List[str] = []
    eos_hit: List[int] = []
    new_tok: List[int] = []

    for i in tqdm(range(0, len(prompts), batch_size), desc=f"Generate({decoding})"):
        # batch = prompts[i:i+batch_size]
        # inputs = tokenizer(
        #     batch,
        #     return_tensors="pt",
        #     padding=True,
        #     truncation=True,
        #     max_length=max_prompt_len,
        # ).to(device)
        use_template = bool(getattr(tokenizer, "chat_template", None))

        batch = prompts[i:i+batch_size]
        batch = [render_prompt(tokenizer, p, add_generation_prompt=True) for p in batch]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_len,
            add_special_tokens=not use_template,
        ).to(device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        B, T0 = input_ids.shape

        state = GenerationState(B, input_ids.device, reasoning_token_threshold)
        if state_setter is not None:
            state_setter(state)

        # Decode-aligned boundary
        past, logits = _cache_advanced_prompt_boundary(model, input_ids, attention_mask)

        generated = input_ids

        for _ in range(max_new_tokens):
            next_token = _choose_next_token(
                logits,
                decoding=decoding,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                eos_token_id=eos,
                ban_eos=False,
            )

            next_token = torch.where(
                state.unfinished.unsqueeze(-1),
                next_token,
                torch.full_like(next_token, eos),
            )

            generated = torch.cat([generated, next_token], dim=1)

            state.step_update(next_token, eos_token_id=eos)
            if not bool(state.unfinished.any().item()):
                break

            attention_mask = torch.cat(
                [attention_mask, torch.ones((B, 1), device=attention_mask.device, dtype=attention_mask.dtype)],
                dim=1,
            )

            out = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                use_cache=True,
                past_key_values=past,
            )
            logits = out.logits[:, -1, :]
            past = out.past_key_values

        if state_setter is not None:
            state_setter(None)

        for b in range(B):
            L = int(state.gen_steps[b].item())
            cont_ids = generated[b, T0:T0+L]
            txt = tokenizer.decode(cont_ids, skip_special_tokens=True)
            continuations.append(txt)
            eos_hit.append(int(not bool(state.unfinished[b].item())))
            new_tok.append(L)

    return continuations, np.array(eos_hit, dtype=np.int32), np.array(new_tok, dtype=np.int32)


def evaluate_condition_generation(
    model,
    tokenizer,
    examples: List["Example"],
    Q_np: Optional[np.ndarray],
    condition: str,     # baseline/full/staged
    decoding: str,       # greedy/sample
    alpha: float,
    layer_indices: List[int],
    reasoning_token_threshold: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    device: str,
    batch_size: int,
    max_prompt_len: int,
    bootstrap_iters: int,
    ci_alpha: float,
    global_seed: int,
    sample_seed: Optional[int] = None,
    save_generation_details: bool = False,
) -> Dict[str, Any]:

    handles, state_setter, hook_stats = register_hooks_for_condition(
        model=model,
        layer_indices=layer_indices,
        Q_np=Q_np,
        condition=condition,
        alpha=alpha,
        reasoning_token_threshold=reasoning_token_threshold,
    )

    try:
        prompts = [ex.prompt for ex in examples]
        continuations, eos_hit, new_tok = generate_continuations(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            decoding=decoding,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            device=device,
            batch_size=batch_size,
            max_prompt_len=max_prompt_len,
            reasoning_token_threshold=reasoning_token_threshold,
            state_setter=state_setter,
            sample_seed=sample_seed,
        )

        correct, extracted = [], []
        preds = []
        for ex, cont in zip(examples, continuations):
            pred = parse_prediction(ex.dataset, cont)
            preds.append(pred)
            extracted.append(int(pred != ""))
            correct.append(is_correct(ex.dataset, pred, ex.gold))

        correct_arr = np.array(correct, dtype=np.float32)
        extracted_arr = np.array(extracted, dtype=np.float32)

        seed = stable_int_seed(global_seed, examples[0].dataset if examples else "na", condition, decoding, sample_seed or 0)
        acc, lo, hi = bootstrap_ci_mean(correct_arr, iters=bootstrap_iters, alpha=ci_alpha, seed=seed)

        out = {
            "protocol": "generation",
            "condition": condition,
            "decoding": decoding,
            "sample_seed": sample_seed,
            "accuracy": float(acc),
            "ci_low": float(lo),
            "ci_high": float(hi),
            "correct": correct_arr.tolist(),
            "extraction_rate": float(extracted_arr.mean()),
            "eos_rate": float(np.mean(eos_hit)),
            "avg_new_tokens": float(np.mean(new_tok)),
            "hook_stats": [{"name": s.name, "decode_calls": s.decode_calls, "intervened": s.intervened} for s in hook_stats],
        }
        if save_generation_details:
            out["example_ids"] = [str(ex.ex_id) for ex in examples]
            out["prompts"] = [str(ex.prompt) for ex in examples]
            out["golds"] = [str(ex.gold) for ex in examples]
            out["preds"] = [str(x) for x in preds]
            out["continuations"] = [str(x) for x in continuations]
        return out
    finally:
        remove_hooks(handles)


def _truncate_text(s: Any, limit: int) -> str:
    text = "" if s is None else str(s)
    if int(limit) <= 0 or len(text) <= int(limit):
        return text
    return text[: int(limit)] + "...[truncated]"


def select_representative_generation_examples(
    *,
    examples: List["Example"],
    block: Dict[str, Any],
    decoding: str,
    max_examples: int,
    prompt_char_limit: int,
    continuation_char_limit: int,
) -> List[Dict[str, Any]]:
    if int(max_examples) <= 0:
        return []

    modes = ["baseline", "shared_full", "shared_staged", "rand_full", "rand_staged"]
    runs: Dict[str, Dict[str, Any]] = {}
    for mode in modes:
        run = block["runs"].get(f"{decoding}/{mode}")
        if not run:
            return []
        if "continuations" not in run or "preds" not in run:
            return []
        runs[mode] = run

    n = len(examples)
    bools: Dict[str, List[bool]] = {
        mode: [bool(x) for x in runs[mode]["correct"]]
        for mode in modes
    }

    priorities = [
        (
            "baseline_correct_shared_full_wrong_rand_full_correct",
            lambda i: bools["baseline"][i] and (not bools["shared_full"][i]) and bools["rand_full"][i],
        ),
        (
            "baseline_correct_shared_staged_wrong_rand_staged_correct",
            lambda i: bools["baseline"][i] and (not bools["shared_staged"][i]) and bools["rand_staged"][i],
        ),
        (
            "baseline_correct_shared_full_wrong",
            lambda i: bools["baseline"][i] and (not bools["shared_full"][i]),
        ),
        (
            "shared_full_wrong_rand_full_correct",
            lambda i: (not bools["shared_full"][i]) and bools["rand_full"][i],
        ),
        (
            "baseline_vs_shared_full_changed",
            lambda i: bools["baseline"][i] != bools["shared_full"][i],
        ),
    ]

    chosen: List[Dict[str, Any]] = []
    seen_ids = set()

    for reason, predicate in priorities:
        for i in range(n):
            if len(chosen) >= int(max_examples):
                return chosen

            ex = examples[i]
            ex_id = str(ex.ex_id)
            if ex_id in seen_ids:
                continue
            if not predicate(i):
                continue

            item = {
                "selection_reason": reason,
                "ex_id": ex_id,
                "gold": str(ex.gold),
                "prompt": _truncate_text(ex.prompt, prompt_char_limit),
                "by_condition": {},
            }
            for mode in modes:
                run = runs[mode]
                item["by_condition"][mode] = {
                    "correct": bool(bools[mode][i]),
                    "pred": str(run["preds"][i]),
                    "continuation": _truncate_text(run["continuations"][i], continuation_char_limit),
                }
            chosen.append(item)
            seen_ids.add(ex_id)

    return chosen


# -----------------------------
# Forced-choice utilities (improved)
# -----------------------------
def candidate_strings(task: str) -> List[str]:
    """
    Candidate labels/strings for forced-choice scoring.
    Must match benchmark_dataloaders gold label conventions.
    """
    t = task.strip().lower()

    if t in ["commonsenseqa", "aqua"]:
        return list("ABCDE")
    if t in ["arc_challenge", "openbookqa", "logiqa"]:
        return list("ABCD")
    if t == "qasc":
        return list("ABCDEFGH")
    if t == "piqa":
        return ["A", "B"]
    if t == "boolq":
        # many loaders use A/B (A=yes, B=no)
        return ["A", "B"]
    if t == "strategyqa":
        return ["Yes", "No"]
    return []


def _context_ends_with_whitespace(s: str) -> bool:
    return len(s) > 0 and s[-1].isspace()


def cand_token_ids(tokenizer, s: str, *, leading_space: bool) -> List[int]:
    """
    Encode candidate string into token ids, optionally with a leading space.

    This is a common source of accidental chance-level baselines:
      - If your prompt ends with 'Final answer:' (no space), you usually want leading_space=True.
      - If your prompt already ends with whitespace/newline, leading_space=False is safer.
    """
    text = (" " + s) if leading_space else s
    ids = tokenizer.encode(text, add_special_tokens=False)
    # Fallback if something weird happens
    if not ids and leading_space:
        ids = tokenizer.encode(s, add_special_tokens=False)
    return ids


# def _should_add_fc_prefix(
#     *,
#     prompt: str,
#     warmup_tokens: int,
#     prefix_mode: str,
#     fc_answer_prefix: str,
# ) -> bool:
#     """
#     Decide whether to teacher-force answer prefix before scoring candidates.
#     """
#     if not fc_answer_prefix:
#         return False
#     assert prefix_mode in {"auto", "always", "never"}
#     if prefix_mode == "never":
#         return False
#     if prefix_mode == "always":
#         return True

#     # auto:
#     # If warmup>0, ALWAYS re-anchor the decision point after warmup.
#     if warmup_tokens > 0:
#         return True

#     # Otherwise, add prefix only if prompt doesn't already end with it (ignoring trailing whitespace).
#     p = prompt.rstrip()
#     ap = fc_answer_prefix.rstrip()
#     return (ap != "") and (not p.endswith(ap))

def _should_add_fc_prefix(
    *,
    prompt: str,
    warmup_tokens: int,
    prefix_mode: str,
    fc_answer_prefix: str,
    use_chat_template: bool,
) -> bool:
    """
    Decide whether to teacher-force answer prefix before scoring candidates.

    IMPORTANT:
    - For chat-template models (Gemma-it / Llama-chat), generation starts at the assistant turn,
      so checking whether the *raw prompt string* endswith(prefix) is NOT a reliable decision-point test.
      In that case, 'auto' should default to adding the prefix (unless prefix_mode=='never').
    """
    if not fc_answer_prefix:
        return False
    assert prefix_mode in {"auto", "always", "never"}

    if prefix_mode == "never":
        return False
    if prefix_mode == "always":
        return True

    # auto:
    # If using chat template, ALWAYS add prefix to anchor decision point in assistant turn.
    if use_chat_template:
        return True

    # Non-chat: if warmup>0, always re-anchor after warmup.
    if warmup_tokens > 0:
        return True

    # Otherwise, add prefix only if prompt doesn't already end with it (ignoring trailing whitespace).
    p = (prompt or "").rstrip()
    ap = fc_answer_prefix.rstrip()
    return (ap != "") and (not p.endswith(ap))


@torch.no_grad()
def precompute_fc_warmup_tokens(
    model,
    tokenizer,
    prompts: List[str],
    *,
    warmup_tokens: int,
    batch_size: int,
    max_prompt_len: int,
    decoding: str,
    temperature: float,
    top_p: float,
    top_k: int,
    ban_eos: bool,
    seed: int,
) -> np.ndarray:
    """Generate warmup tokens under baseline (no intervention). Returns [N,W] int64."""
    assert warmup_tokens >= 0
    if warmup_tokens == 0:
        return np.zeros((len(prompts), 0), dtype=np.int64)

    device = next(model.parameters()).device
    eos = tokenizer.eos_token_id
    model.eval()

    if decoding == "sample":
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    out_tokens = np.zeros((len(prompts), warmup_tokens), dtype=np.int64)

    for i in tqdm(range(0, len(prompts), batch_size), desc=f"FCWarmupGen(W={warmup_tokens})"):
        # batch = prompts[i : i + batch_size]
        # inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_prompt_len).to(device)
        use_template = bool(getattr(tokenizer, "chat_template", None))
        batch = prompts[i:i+batch_size]
        batch = [render_prompt(tokenizer, p, add_generation_prompt=True) for p in batch]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_len,
            add_special_tokens=not use_template,
        ).to(device)

        ids = inputs["input_ids"]
        attn = inputs["attention_mask"]
        B = ids.shape[0]

        past, logits = _cache_advanced_prompt_boundary(model, ids, attn)

        toks = []
        for _ in range(warmup_tokens):
            next_tok = _choose_next_token(
                logits,
                decoding=decoding,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                eos_token_id=eos,
                ban_eos=ban_eos,
            )
            toks.append(next_tok)
            attn = torch.cat([attn, torch.ones((B, 1), device=device, dtype=attn.dtype)], dim=1)
            out = model(input_ids=next_tok, attention_mask=attn, use_cache=True, past_key_values=past)
            logits = out.logits[:, -1, :]
            past = out.past_key_values

        toks_mat = torch.cat(toks, dim=1)  # [B,W]
        out_tokens[i : i + B, :] = toks_mat.detach().cpu().numpy().astype(np.int64, copy=False)

    return out_tokens


# @torch.no_grad()
# def evaluate_condition_forced_choice(
#     model,
#     tokenizer,
#     examples: List["Example"],
#     task_name: str,
#     Q_np: Optional[np.ndarray],
#     condition: str,  # baseline/full/staged
#     *,
#     alpha: float,
#     layer_indices: List[int],
#     reasoning_token_threshold: int,
#     batch_size: int,
#     max_prompt_len: int,
#     bootstrap_iters: int,
#     ci_alpha: float,
#     global_seed: int,
#     # forced-choice knobs
#     warmup_token_ids: Optional[np.ndarray],
#     fc_warmup_tokens: int,
#     fc_prefix_mode: str,
#     fc_answer_prefix: str,
# ) -> Dict[str, Any]:
#     """
#     Forced-choice accuracy by sum logprob of candidate strings.

#     This is decode-aligned and compatible with decode-only interventions.
#     """
#     device = next(model.parameters()).device
#     model.eval()

#     tokenizer.padding_side = "left"
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#     eos = tokenizer.eos_token_id

#     cands = candidate_strings(task_name)
#     if len(cands) == 0:
#         raise ValueError(f"Task '{task_name}' has no forced-choice candidates.")

#     # Register hooks (baseline/full/staged)
#     handles, state_setter, hook_stats = register_hooks_for_condition(
#         model=model,
#         layer_indices=layer_indices,
#         Q_np=Q_np,
#         condition=condition,
#         alpha=alpha,
#         reasoning_token_threshold=reasoning_token_threshold,
#     )

#     prompts = [ex.prompt for ex in examples]
#     golds = [ex.gold for ex in examples]
#     correct = np.zeros(len(examples), dtype=np.float32)

#     # Useful diagnostics
#     avg_margin = []
#     avg_best_lp = []

#     try:
#         for i in tqdm(range(0, len(examples), batch_size), desc=f"ForcedChoice({task_name}/{condition})"):
#             batch_ex = examples[i : i + batch_size]
#             batch_prompts = [ex.prompt for ex in batch_ex]
#             batch_golds = [ex.gold for ex in batch_ex]

#             # inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_prompt_len).to(device)
#             use_template = bool(getattr(tokenizer, "chat_template", None))
#             batch = prompts[i:i+batch_size]
#             batch = [render_prompt(tokenizer, p, add_generation_prompt=True) for p in batch]

#             inputs = tokenizer(
#                 batch,
#                 return_tensors="pt",
#                 padding=True,
#                 truncation=True,
#                 max_length=max_prompt_len,
#                 add_special_tokens=not use_template,
#             ).to(device)

#             ids = inputs["input_ids"]
#             attn = inputs["attention_mask"]
#             B = ids.shape[0]
#             # Decide per-example whether to add answer prefix (robust to template randomization).
#             # If prompts are mixed (some already contain the prefix, some not), we split into two
#             # sub-batches to avoid duplicated prefixes (which can tank accuracy).
#             batch_prompts_raw = [ex.prompt for ex in batch]  # 用于 need_prefix 判断
#             need_prefix = [
#                 _should_add_fc_prefix(
#                     prompt=p,
#                     warmup_tokens=fc_warmup_tokens,
#                     prefix_mode=fc_prefix_mode,
#                     fc_answer_prefix=fc_answer_prefix,
#                 )
#                 for p in batch_prompts
#             ]

#             warm_slice = None
#             if warmup_token_ids is not None and fc_warmup_tokens > 0:
#                 warm_slice = warmup_token_ids[i : i + B]

#             scores_full = torch.full((B, len(cands)), float("-inf"), device=device)

#             # Evaluate two groups: add_prefix=False and add_prefix=True
#             for add_prefix in [False, True]:
#                 idxs = [j for j, flag in enumerate(need_prefix) if bool(flag) == add_prefix]
#                 if not idxs:
#                     continue

#                 ids_g = ids[idxs]
#                 attn_g = attn[idxs]
#                 Bg = ids_g.shape[0]

#                 # Staged state for this sub-batch
#                 if state_setter is not None:
#                     st = GenerationState(Bg, ids_g.device, reasoning_token_threshold)
#                     state_setter(st)
#                 else:
#                     st = None

#                 # Decode-aligned boundary for this sub-batch
#                 past_g, logits_g = _cache_advanced_prompt_boundary(model, ids_g, attn_g)

#                 # Teacher-force warmup tokens (baseline-generated, shared across conditions)
#                 if warm_slice is not None:
#                     warm_g = torch.tensor(warm_slice[idxs], dtype=torch.long, device=device)
#                     for t in range(warm_g.shape[1]):
#                         tok_t = warm_g[:, t : t + 1]
#                         attn_g = torch.cat(
#                             [attn_g, torch.ones((Bg, 1), device=device, dtype=attn_g.dtype)],
#                             dim=1,
#                         )
#                         out = model(input_ids=tok_t, attention_mask=attn_g, use_cache=True, past_key_values=past_g)
#                         logits_g = out.logits[:, -1, :]
#                         past_g = out.past_key_values
#                         if st is not None:
#                             st.step_update(tok_t, eos_token_id=eos)

#                 # Teacher-force answer prefix if needed for this sub-batch
#                 prefix_used_g = fc_answer_prefix if (add_prefix and fc_answer_prefix) else ""
#                 if prefix_used_g:
#                     prefix_ids = tokenizer.encode(prefix_used_g, add_special_tokens=False)
#                     for pid in prefix_ids:
#                         inp = torch.full((Bg, 1), pid, dtype=torch.long, device=device)
#                         attn_g = torch.cat(
#                             [attn_g, torch.ones((Bg, 1), device=device, dtype=attn_g.dtype)],
#                             dim=1,
#                         )
#                         out = model(input_ids=inp, attention_mask=attn_g, use_cache=True, past_key_values=past_g)
#                         logits_g = out.logits[:, -1, :]
#                         past_g = out.past_key_values
#                         if st is not None:
#                             st.step_update(inp, eos_token_id=eos)

#                 # Determine candidate tokenization (leading space or not) based on the *expected* context end.
#                 if prefix_used_g:
#                     leading_space_g = not _context_ends_with_whitespace(prefix_used_g)
#                 else:
#                     leading_space_g = not _context_ends_with_whitespace(batch_prompts[idxs[0]])

#                 cand_ids_list = [cand_token_ids(tokenizer, s, leading_space=leading_space_g) for s in cands]

#                 # Score candidates by logprob for this sub-batch
#                 scores_g = torch.zeros(Bg, len(cands), device=device)
#                 for ci, cand_ids in enumerate(cand_ids_list):
#                     if len(cand_ids) == 0:
#                         scores_g[:, ci] = float("-inf")
#                         continue

#                     past_c = past_g
#                     attn_c = attn_g
#                     logits_c = logits_g

#                     # For staged: each candidate path starts from the same state (before candidate tokens)
#                     if state_setter is not None:
#                         cand_state = st.clone()
#                         state_setter(cand_state)
#                     else:
#                         cand_state = None

#                     lp = torch.zeros(Bg, device=device)
#                     for ti, tok_id in enumerate(cand_ids):
#                         logp = torch.log_softmax(logits_c, dim=-1)
#                         lp = lp + logp[:, tok_id]
#                         if ti < len(cand_ids) - 1:
#                             inp = torch.full((Bg, 1), tok_id, dtype=torch.long, device=device)
#                             attn_c = torch.cat(
#                                 [attn_c, torch.ones((Bg, 1), device=device, dtype=attn_c.dtype)],
#                                 dim=1,
#                             )
#                             out = model(input_ids=inp, attention_mask=attn_c, use_cache=True, past_key_values=past_c)
#                             logits_c = out.logits[:, -1, :]
#                             past_c = out.past_key_values
#                             if cand_state is not None:
#                                 cand_state.step_update(inp, eos_token_id=eos)

#                     scores_g[:, ci] = lp

#                 scores_full[idxs, :] = scores_g

#             # pick best candidate for each example
#             pred_idx = torch.argmax(scores_full, dim=1).detach().cpu().numpy().tolist()
#             preds = [cands[j] for j in pred_idx]

#             # simple margin diagnostics
#             top2 = torch.topk(scores_full, k=min(2, scores_full.shape[1]), dim=1).values.detach().cpu().numpy()
#             if top2.shape[1] >= 2:
#                 avg_margin.extend((top2[:, 0] - top2[:, 1]).tolist())
#             avg_best_lp.extend(top2[:, 0].tolist())

#             for b, (pred, gold) in enumerate(zip(preds, batch_golds)):
#                 correct[i + b] = float(is_correct(task_name, pred, gold))

#         # clear state pointer for staged
#         if state_setter is not None:
#             state_setter(None)

#         correct_arr = correct.astype(np.float32, copy=False)
#         seed = stable_int_seed(global_seed, task_name, "forced_choice", condition)
#         acc, lo, hi = bootstrap_ci_mean(correct_arr, iters=bootstrap_iters, alpha=ci_alpha, seed=seed)

#         return {
#             "protocol": "forced_choice",
#             "condition": condition,
#             "accuracy": float(acc),
#             "ci_low": float(lo),
#             "ci_high": float(hi),
#             "correct": correct_arr.tolist(),
#             "fc": {
#                 "candidates": cands,
#                 "warmup_tokens": int(fc_warmup_tokens),
#                 "prefix_mode": fc_prefix_mode,
#                 "answer_prefix": fc_answer_prefix,
#                 "leading_space": bool(not _context_ends_with_whitespace((fc_answer_prefix if fc_answer_prefix else (prompts[0] if prompts else "")))),
#                 "avg_best_logprob": float(np.mean(avg_best_lp)) if avg_best_lp else float("nan"),
#                 "avg_margin": float(np.mean(avg_margin)) if avg_margin else float("nan"),
#             },
#             # generation-only fields: keep for schema compatibility
#             "extraction_rate": 1.0,
#             "eos_rate": float("nan"),
#             "avg_new_tokens": float("nan"),
#             "hook_stats": [{"name": s.name, "decode_calls": s.decode_calls, "intervened": s.intervened} for s in hook_stats],
#         }
#     finally:
#         remove_hooks(handles)

@torch.no_grad()
def evaluate_condition_forced_choice(
    model,
    tokenizer,
    examples: List["Example"],
    task_name: str,
    Q_np: Optional[np.ndarray],
    condition: str,  # baseline/full/staged
    *,
    alpha: float,
    layer_indices: List[int],
    reasoning_token_threshold: int,
    batch_size: int,
    max_prompt_len: int,
    bootstrap_iters: int,
    ci_alpha: float,
    global_seed: int,
    # forced-choice knobs
    warmup_token_ids: Optional[np.ndarray],
    fc_warmup_tokens: int,
    fc_prefix_mode: str,
    fc_answer_prefix: str,
) -> Dict[str, Any]:
    """
    Forced-choice accuracy by sum logprob of candidate strings.

    Fixes vs your current version:
      - Keep RAW prompts (for prefix decision) separate from RENDERED prompts (for tokenization / whitespace).
      - For chat_template models, auto-prefix defaults to True (anchors decision point in assistant turn).
      - Remove buggy 'batch_prompts_raw = [ex.prompt for ex in batch]' usage.
    """
    device = next(model.parameters()).device
    model.eval()

    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos = tokenizer.eos_token_id

    cands = candidate_strings(task_name)
    if len(cands) == 0:
        raise ValueError(f"Task '{task_name}' has no forced-choice candidates.")

    handles, state_setter, hook_stats = register_hooks_for_condition(
        model=model,
        layer_indices=layer_indices,
        Q_np=Q_np,
        condition=condition,
        alpha=alpha,
        reasoning_token_threshold=reasoning_token_threshold,
    )

    correct = np.zeros(len(examples), dtype=np.float32)

    # diagnostics
    avg_margin: List[float] = []
    avg_best_lp: List[float] = []

    try:
        for i in tqdm(range(0, len(examples), batch_size), desc=f"ForcedChoice({task_name}/{condition})"):
            batch_ex = examples[i : i + batch_size]
            batch_prompts_raw = [ex.prompt for ex in batch_ex]   # for prefix decision
            batch_golds = [ex.gold for ex in batch_ex]

            use_template = bool(getattr(tokenizer, "chat_template", None))
            batch_prompts_rendered = [render_prompt(tokenizer, p, add_generation_prompt=True) for p in batch_prompts_raw]

            inputs = tokenizer(
                batch_prompts_rendered,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_prompt_len,
                add_special_tokens=not use_template,  # IMPORTANT: avoid double BOS when apply_chat_template used
            ).to(device)

            ids = inputs["input_ids"]
            attn = inputs["attention_mask"]
            B = ids.shape[0]

            # per-example: whether we should teacher-force answer prefix before scoring candidates
            need_prefix = [
                _should_add_fc_prefix(
                    prompt=p,
                    warmup_tokens=fc_warmup_tokens,
                    prefix_mode=fc_prefix_mode,
                    fc_answer_prefix=fc_answer_prefix,
                    use_chat_template=use_template,
                )
                for p in batch_prompts_raw
            ]

            warm_slice = None
            if warmup_token_ids is not None and fc_warmup_tokens > 0:
                warm_slice = warmup_token_ids[i : i + B]  # [B,W]

            scores_full = torch.full((B, len(cands)), float("-inf"), device=device)

            # Evaluate two groups to avoid duplicated prefix: need_prefix False / True
            for add_prefix_flag in [False, True]:
                idxs = [j for j, flag in enumerate(need_prefix) if bool(flag) == add_prefix_flag]
                if not idxs:
                    continue

                ids_g = ids[idxs]
                attn_g = attn[idxs]
                Bg = ids_g.shape[0]

                # staged state for this sub-batch
                if state_setter is not None:
                    st = GenerationState(Bg, ids_g.device, reasoning_token_threshold)
                    state_setter(st)
                else:
                    st = None

                # decode-aligned boundary
                past_g, logits_g = _cache_advanced_prompt_boundary(model, ids_g, attn_g)

                # teacher-force warmup tokens (baseline-generated, shared across conditions)
                if warm_slice is not None:
                    warm_g = torch.tensor(warm_slice[idxs], dtype=torch.long, device=device)  # [Bg,W]
                    for t in range(warm_g.shape[1]):
                        tok_t = warm_g[:, t : t + 1]
                        attn_g = torch.cat(
                            [attn_g, torch.ones((Bg, 1), device=device, dtype=attn_g.dtype)],
                            dim=1,
                        )
                        out = model(input_ids=tok_t, attention_mask=attn_g, use_cache=True, past_key_values=past_g)
                        logits_g = out.logits[:, -1, :]
                        past_g = out.past_key_values
                        if st is not None:
                            st.step_update(tok_t, eos_token_id=eos)

                # teacher-force answer prefix (anchors decision point)
                prefix_used_g = fc_answer_prefix if (add_prefix_flag and fc_answer_prefix) else ""
                if prefix_used_g:
                    prefix_ids = tokenizer.encode(prefix_used_g, add_special_tokens=False)
                    for pid in prefix_ids:
                        inp = torch.full((Bg, 1), pid, dtype=torch.long, device=device)
                        attn_g = torch.cat(
                            [attn_g, torch.ones((Bg, 1), device=device, dtype=attn_g.dtype)],
                            dim=1,
                        )
                        out = model(input_ids=inp, attention_mask=attn_g, use_cache=True, past_key_values=past_g)
                        logits_g = out.logits[:, -1, :]
                        past_g = out.past_key_values
                        if st is not None:
                            st.step_update(inp, eos_token_id=eos)

                # Determine whether candidates should be tokenized with a leading space
                # (should reflect the REAL context end: prefix (if used) else rendered prompt end).
                if prefix_used_g:
                    leading_space_g = not _context_ends_with_whitespace(prefix_used_g)
                else:
                    leading_space_g = not _context_ends_with_whitespace(batch_prompts_rendered[idxs[0]])

                cand_ids_list = [cand_token_ids(tokenizer, s, leading_space=leading_space_g) for s in cands]

                # Score candidates
                scores_g = torch.zeros(Bg, len(cands), device=device)
                for ci, cand_ids in enumerate(cand_ids_list):
                    if len(cand_ids) == 0:
                        scores_g[:, ci] = float("-inf")
                        continue

                    past_c = past_g
                    attn_c = attn_g
                    logits_c = logits_g

                    # for staged: each candidate path starts from the same state
                    if state_setter is not None and st is not None:
                        cand_state = st.clone()
                        state_setter(cand_state)
                    else:
                        cand_state = None

                    lp = torch.zeros(Bg, device=device)
                    for ti, tok_id in enumerate(cand_ids):
                        logp = torch.log_softmax(logits_c, dim=-1)
                        lp = lp + logp[:, tok_id]

                        if ti < len(cand_ids) - 1:
                            inp = torch.full((Bg, 1), tok_id, dtype=torch.long, device=device)
                            attn_c = torch.cat(
                                [attn_c, torch.ones((Bg, 1), device=device, dtype=attn_c.dtype)],
                                dim=1,
                            )
                            out = model(input_ids=inp, attention_mask=attn_c, use_cache=True, past_key_values=past_c)
                            logits_c = out.logits[:, -1, :]
                            past_c = out.past_key_values
                            if cand_state is not None:
                                cand_state.step_update(inp, eos_token_id=eos)

                    scores_g[:, ci] = lp

                scores_full[idxs, :] = scores_g

                # restore staged pointer (hygiene)
                if state_setter is not None:
                    state_setter(st)

            # pick best candidate for each example
            pred_idx = torch.argmax(scores_full, dim=1).detach().cpu().numpy().tolist()
            preds = [cands[j] for j in pred_idx]

            # margin diagnostics
            top2 = torch.topk(scores_full, k=min(2, scores_full.shape[1]), dim=1).values.detach().cpu().numpy()
            if top2.shape[1] >= 2:
                avg_margin.extend((top2[:, 0] - top2[:, 1]).tolist())
            avg_best_lp.extend(top2[:, 0].tolist())

            for b, (pred, gold) in enumerate(zip(preds, batch_golds)):
                correct[i + b] = float(is_correct(task_name, pred, gold))

        if state_setter is not None:
            state_setter(None)

        correct_arr = correct.astype(np.float32, copy=False)
        seed = stable_int_seed(global_seed, task_name, "forced_choice", condition)
        acc, lo, hi = bootstrap_ci_mean(correct_arr, iters=bootstrap_iters, alpha=ci_alpha, seed=seed)

        # diagnostic leading_space (best-effort)
        if fc_answer_prefix and fc_prefix_mode != "never":
            diag_leading_space = bool(not _context_ends_with_whitespace(fc_answer_prefix))
        else:
            if examples:
                use_template0 = bool(getattr(tokenizer, "chat_template", None))
                p0 = render_prompt(tokenizer, examples[0].prompt, add_generation_prompt=True) if use_template0 else examples[0].prompt
                diag_leading_space = bool(not _context_ends_with_whitespace(p0))
            else:
                diag_leading_space = False

        return {
            "protocol": "forced_choice",
            "condition": condition,
            "accuracy": float(acc),
            "ci_low": float(lo),
            "ci_high": float(hi),
            "correct": correct_arr.tolist(),
            "fc": {
                "candidates": cands,
                "warmup_tokens": int(fc_warmup_tokens),
                "prefix_mode": fc_prefix_mode,
                "answer_prefix": fc_answer_prefix,
                "leading_space": diag_leading_space,
                "avg_best_logprob": float(np.mean(avg_best_lp)) if avg_best_lp else float("nan"),
                "avg_margin": float(np.mean(avg_margin)) if avg_margin else float("nan"),
                "use_chat_template": bool(getattr(tokenizer, "chat_template", None)),
            },
            # keep schema compatibility
            "extraction_rate": 1.0,
            "eos_rate": float("nan"),
            "avg_new_tokens": float("nan"),
            "hook_stats": [{"name": s.name, "decode_calls": s.decode_calls, "intervened": s.intervened} for s in hook_stats],
        }

    finally:
        remove_hooks(handles)


# -----------------------------
# Model loading
# -----------------------------
# def load_model_and_tokenizer(model_name: str, device: str, model_dtype: str):
#     dtype = torch.float32 if model_dtype == "fp32" else torch.float16
#     try:
#         model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
#     except TypeError:
#         model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)

#     tok = AutoTokenizer.from_pretrained(model_name)
#     tok.padding_side = "left"
#     if tok.pad_token is None:
#         tok.pad_token = tok.eos_token

#     model = model.to(device)
#     model.eval()
#     if hasattr(model.config, "use_cache"):
#         model.config.use_cache = True
#     return model, tok

def load_model_and_tokenizer(model_name: str, device: str, model_dtype: str):
    dtype = torch.float32 if model_dtype == "fp32" else torch.float16
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)

    tok = AutoTokenizer.from_pretrained(model_name)

    # IMPORTANT for left padding + truncation:
    tok.padding_side = "left"
    tok.truncation_side = "left"

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = model.to(device)
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    return model, tok


# -----------------------------
# Running one fold (basis estimation + eval)
# -----------------------------
def run_fold(
    *,
    fold_name: str,
    model,
    tokenizer,
    sub_by: Dict[str, List["Example"]],
    eval_by: Dict[str, List["Example"]],
    train_tasks: List[str],
    eval_tasks: List[str],
    layer_indices: List[int],
    args,
) -> Dict[str, Any]:

    # 1) basis estimation on train_tasks only
    prompts_by_task = {k: [ex.prompt for ex in sub_by[k]] for k in train_tasks}
    joint_subspace, shared_indices, extra, task_acts = compute_shared_subspace_decode_aligned(
        model=model,
        tokenizer=tokenizer,
        prompts_by_task=prompts_by_task,
        layer_indices=layer_indices,
        calib_decoding="greedy",
        calib_batch_size=args.batch_size,
        calib_max_new_tokens=args.calib_decode_max_new_tokens,
        per_task_max_states=args.per_task_max_states,
        max_prompt_len=args.max_prompt_len,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        global_seed=args.seed,
        variance_threshold=args.pca_var,
        min_dim=args.min_dim,
        max_dim=args.max_dim,
        tau=args.tau,
        m_shared=args.m_shared,
    )

    cross_dim = int(extra["cross_dim"])
    if len(shared_indices) == 0:
        raise RuntimeError(f"[{fold_name}] No shared basis found (shared_indices empty). Try relax tau or m_shared.")

    shared_basis = joint_subspace[:, shared_indices]  # [D, kS]
    Q_shared = orthonormalize_np(shared_basis)
    kS = Q_shared.shape[1]

    pooled_var = infer_component_variances(extra["task_contributions"], extra["tasks_used"], cross_dim)

    rand_indices = select_rand_indices(
        rand_type=args.rand_type,
        cross_dim=cross_dim,
        shared_indices=shared_indices,
        pooled_var=pooled_var,
        k=kS,
        seed=stable_int_seed(args.seed, fold_name, "rand_idx", args.rand_type),
    )
    rand_basis = joint_subspace[:, rand_indices]
    Q_rand = orthonormalize_np(rand_basis)

    # 2) SANITY: orthonormality + overlap
    sanity = {
        "cross_dim": cross_dim,
        "shared_basis_dim": int(kS),
        "rand_type": args.rand_type,
        "orthonorm_max_offdiag_shared": max_offdiag(Q_shared),
        "orthonorm_max_offdiag_rand": max_offdiag(Q_rand),
        "max_overlap_shared_rand": max_overlap(Q_shared, Q_rand),
    }

    # 3) SANITY: energy ratios on calibration decode states
    layer = layer_indices[0]
    pool = []
    for t in extra["tasks_used"]:
        X = task_acts[t][layer]
        ss = stable_int_seed(args.seed, fold_name, "energy_sample", t)
        Xs = _subsample_rows_np(X, n_max=min(4000, X.shape[0]), seed=ss)
        pool.append(Xs)
    calib_states = np.concatenate(pool, axis=0)
    er_s = energy_ratio_stats(calib_states, Q_shared)
    er_r = energy_ratio_stats(calib_states, Q_rand)
    sanity["energy_ratio_shared"] = er_s
    sanity["energy_ratio_rand"] = er_r

    print(f"\n[{fold_name}] cross_dim={cross_dim} shared_dim={kS}")
    print(f"[{fold_name}][Sanity] Orthonormality max offdiag: shared={sanity['orthonorm_max_offdiag_shared']:.2e}, rand={sanity['orthonorm_max_offdiag_rand']:.2e}")
    print(f"[{fold_name}][Sanity] Max overlap |Q_shared^T Q_rand| = {sanity['max_overlap_shared_rand']:.2e}")
    print(f"[{fold_name}][Sanity] Energy ratio on calib decode states: shared mean={er_s['mean']:.4f}, rand mean={er_r['mean']:.4f}")

    # 4) Eval
    CONDITIONS = ["baseline", "shared_full", "shared_staged", "rand_full", "rand_staged"]

    by_dataset: Dict[str, Any] = {}
    fold_representative_examples: List[Dict[str, Any]] = []

    # Cache warmup tokens per dataset within this fold (only if forced-choice enabled)
    warmup_cache: Dict[str, np.ndarray] = {}

    for task_name in eval_tasks:
        eval_exs = eval_by[task_name]
        print("\n" + "-" * 80)
        print(f"[{fold_name}][Eval] Dataset={task_name} (n={len(eval_exs)})")
        print("-" * 80)

        # Decide protocol:
        # - gsm8k: generation only
        # - others: forced-choice if enabled and candidates exist; otherwise generation
        has_cands = len(candidate_strings(task_name)) > 0
        use_fc = bool(args.use_forced_choice) and has_cands

        block: Dict[str, Any] = {"n": len(eval_exs), "protocol": ("forced_choice" if use_fc else "generation"), "runs": {}, "paired_tests": {}}

        if use_fc:
            # Precompute warmup tokens once per dataset (baseline-only)
            if args.fc_warmup_tokens > 0:
                if task_name not in warmup_cache:
                    prompts = [ex.prompt for ex in eval_exs]
                    warm_ids = precompute_fc_warmup_tokens(
                        model=model,
                        tokenizer=tokenizer,
                        prompts=prompts,
                        warmup_tokens=args.fc_warmup_tokens,
                        batch_size=args.batch_size,
                        max_prompt_len=args.max_prompt_len,
                        decoding=args.fc_warmup_decoding,
                        temperature=args.fc_warmup_temperature,
                        top_p=args.fc_warmup_top_p,
                        top_k=args.fc_warmup_top_k,
                        ban_eos=bool(args.fc_warmup_ban_eos),
                        seed=stable_int_seed(args.seed, args.fc_warmup_seed, fold_name, task_name, "fc_warmup"),
                    )
                    warmup_cache[task_name] = warm_ids
                    if args.fc_debug_print and warm_ids.shape[0] > 0:
                        demo = tokenizer.decode(warm_ids[0].tolist(), skip_special_tokens=True)
                        print(f"[{fold_name}][FC Warmup] {task_name}: warmup_ids shape={warm_ids.shape}; demo text[:120]={demo[:120]!r}")
            else:
                warmup_cache[task_name] = np.zeros((len(eval_exs), 0), dtype=np.int64)

            warm_ids = warmup_cache.get(task_name, None)

            # Run conditions (no decoding loop for forced-choice)
            for cond in CONDITIONS:
                if cond == "baseline":
                    Q = None
                    cond_name = "baseline"
                    mode = "baseline"
                elif cond == "shared_full":
                    Q = Q_shared
                    cond_name = "full"
                    mode = "shared_full"
                elif cond == "shared_staged":
                    Q = Q_shared
                    cond_name = "staged"
                    mode = "shared_staged"
                elif cond == "rand_full":
                    Q = Q_rand
                    cond_name = "full"
                    mode = "rand_full"
                elif cond == "rand_staged":
                    Q = Q_rand
                    cond_name = "staged"
                    mode = "rand_staged"
                else:
                    raise ValueError(cond)

                run = evaluate_condition_forced_choice(
                    model=model,
                    tokenizer=tokenizer,
                    examples=eval_exs,
                    task_name=task_name,
                    Q_np=Q,
                    condition=cond_name,
                    alpha=args.alpha_remove,
                    layer_indices=layer_indices,
                    reasoning_token_threshold=args.reasoning_tokens,
                    batch_size=args.batch_size,
                    max_prompt_len=args.max_prompt_len,
                    bootstrap_iters=args.bootstrap_iters,
                    ci_alpha=args.ci_alpha,
                    global_seed=args.seed,
                    warmup_token_ids=warm_ids,
                    fc_warmup_tokens=args.fc_warmup_tokens,
                    fc_prefix_mode=args.fc_prefix_mode,
                    fc_answer_prefix=args.fc_answer_prefix,
                )
                block["runs"][f"forced_choice/{mode}"] = run
                fc_meta = run.get("fc", {})
                print(
                    f"[{fold_name}][{task_name}] forced_choice/{mode}: "
                    f"acc={fmt_acc(run['accuracy'], run['ci_low'], run['ci_high'])} "
                    f"(W={fc_meta.get('warmup_tokens', 0)}, prefix_mode={fc_meta.get('prefix_mode')}, "
                    f"avg_margin={fc_meta.get('avg_margin', float('nan')):.3f})"
                )

            # Paired tests (forced-choice)
            base = np.array(block["runs"]["forced_choice/baseline"]["correct"], dtype=np.float32)
            shared_full = np.array(block["runs"]["forced_choice/shared_full"]["correct"], dtype=np.float32)
            rand_full = np.array(block["runs"]["forced_choice/rand_full"]["correct"], dtype=np.float32)
            seed0 = stable_int_seed(args.seed, fold_name, task_name, "forced_choice", "paired")

            block["paired_tests"]["forced_choice"] = {
                "shared_full_vs_baseline": summarize_paired(
                    base, shared_full,
                    label=f"{fold_name}:{task_name}:forced_choice:shared_full_vs_baseline",
                    bootstrap_iters=args.bootstrap_iters,
                    perm_iters=args.perm_iters,
                    alpha=args.ci_alpha,
                    seed=seed0 + 1,
                ),
                "rand_full_vs_baseline": summarize_paired(
                    base, rand_full,
                    label=f"{fold_name}:{task_name}:forced_choice:rand_full_vs_baseline",
                    bootstrap_iters=args.bootstrap_iters,
                    perm_iters=args.perm_iters,
                    alpha=args.ci_alpha,
                    seed=seed0 + 2,
                ),
                "shared_full_vs_rand_full": summarize_paired(
                    rand_full, shared_full,
                    label=f"{fold_name}:{task_name}:forced_choice:shared_full_vs_rand_full",
                    bootstrap_iters=args.bootstrap_iters,
                    perm_iters=args.perm_iters,
                    alpha=args.ci_alpha,
                    seed=seed0 + 3,
                ),
            }
            stat = block["paired_tests"]["forced_choice"]["shared_full_vs_baseline"]
            print(f"[{fold_name}][Stats] {task_name} (forced_choice) shared_full_vs_baseline: "
                  f"Δ={stat['mean_diff']:+.3f} CI[{stat['ci_low']:+.3f}, {stat['ci_high']:+.3f}] p={stat['p_value']:.3g}")

        else:
            # Generation protocol (original behavior)
            DECODINGS = ["greedy"] + (["sample"] if args.do_sample else [])

            for decoding in DECODINGS:
                for cond in CONDITIONS:
                    if cond == "baseline":
                        Q = None
                        cond_name = "baseline"
                        mode = "baseline"
                    elif cond == "shared_full":
                        Q = Q_shared
                        cond_name = "full"
                        mode = "shared_full"
                    elif cond == "shared_staged":
                        Q = Q_shared
                        cond_name = "staged"
                        mode = "shared_staged"
                    elif cond == "rand_full":
                        Q = Q_rand
                        cond_name = "full"
                        mode = "rand_full"
                    elif cond == "rand_staged":
                        Q = Q_rand
                        cond_name = "staged"
                        mode = "rand_staged"
                    else:
                        raise ValueError(cond)

                    run = evaluate_condition_generation(
                        model=model,
                        tokenizer=tokenizer,
                        examples=eval_exs,
                        Q_np=Q,
                        condition=cond_name,
                        decoding=decoding,
                        alpha=args.alpha_remove,
                        layer_indices=layer_indices,
                        reasoning_token_threshold=args.reasoning_tokens,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        device=args.device,
                        batch_size=args.batch_size,
                        max_prompt_len=args.max_prompt_len,
                        bootstrap_iters=args.bootstrap_iters,
                        ci_alpha=args.ci_alpha,
                        global_seed=args.seed,
                        sample_seed=(args.sample_seed if decoding == "sample" else None),
                        save_generation_details=bool(args.save_generation_details or args.save_generation_examples > 0),
                    )
                    block["runs"][f"{decoding}/{mode}"] = run
                    print(
                        f"[{fold_name}][{task_name}] {decoding}/{mode}: "
                        f"acc={fmt_acc(run['accuracy'], run['ci_low'], run['ci_high'])} "
                        f"extr={run['extraction_rate']*100:.1f}% eos={run['eos_rate']*100:.1f}% "
                        f"avg_new_tok={run['avg_new_tokens']:.1f}"
                    )

            # Paired tests (focus on FULL shared vs baseline/rand)
            for decoding in DECODINGS:
                base = np.array(block["runs"][f"{decoding}/baseline"]["correct"], dtype=np.float32)
                shared_full = np.array(block["runs"][f"{decoding}/shared_full"]["correct"], dtype=np.float32)
                rand_full = np.array(block["runs"][f"{decoding}/rand_full"]["correct"], dtype=np.float32)
                seed0 = stable_int_seed(args.seed, fold_name, task_name, decoding, "paired")

                block["paired_tests"][decoding] = {
                    "shared_full_vs_baseline": summarize_paired(
                        base, shared_full,
                        label=f"{fold_name}:{task_name}:{decoding}:shared_full_vs_baseline",
                        bootstrap_iters=args.bootstrap_iters,
                        perm_iters=args.perm_iters,
                        alpha=args.ci_alpha,
                        seed=seed0 + 1,
                    ),
                    "rand_full_vs_baseline": summarize_paired(
                        base, rand_full,
                        label=f"{fold_name}:{task_name}:{decoding}:rand_full_vs_baseline",
                        bootstrap_iters=args.bootstrap_iters,
                        perm_iters=args.perm_iters,
                        alpha=args.ci_alpha,
                        seed=seed0 + 2,
                    ),
                    "shared_full_vs_rand_full": summarize_paired(
                        rand_full, shared_full,
                        label=f"{fold_name}:{task_name}:{decoding}:shared_full_vs_rand_full",
                        bootstrap_iters=args.bootstrap_iters,
                        perm_iters=args.perm_iters,
                        alpha=args.ci_alpha,
                        seed=seed0 + 3,
                    ),
                }
                stat = block["paired_tests"][decoding]["shared_full_vs_baseline"]
                print(f"[{fold_name}][Stats] {task_name} ({decoding}) shared_full_vs_baseline: "
                      f"Δ={stat['mean_diff']:+.3f} CI[{stat['ci_low']:+.3f}, {stat['ci_high']:+.3f}] p={stat['p_value']:.3g}")

                if int(args.save_generation_examples) > 0:
                    reps = select_representative_generation_examples(
                        examples=eval_exs,
                        block=block,
                        decoding=decoding,
                        max_examples=int(args.save_generation_examples),
                        prompt_char_limit=int(args.generation_prompt_char_limit),
                        continuation_char_limit=int(args.generation_continuation_char_limit),
                    )
                    if reps:
                        enriched = []
                        for item in reps:
                            rec = {
                                "fold_name": fold_name,
                                "task": task_name,
                                "protocol": "generation",
                                "decoding": decoding,
                            }
                            rec.update(item)
                            enriched.append(rec)
                        block.setdefault("representative_examples", {})[decoding] = enriched
                        fold_representative_examples.extend(enriched)

        by_dataset[task_name] = block

    return {
        "fold_name": fold_name,
        "train_tasks": train_tasks,
        "eval_tasks": eval_tasks,
        "basis": {
            "cross_dim": sanity["cross_dim"],
            "shared_k": sanity["shared_basis_dim"],
            "shared_indices_count": len(shared_indices),
            "rand_type": sanity["rand_type"],
            "sanity": sanity,
        },
        "by_dataset": by_dataset,
        "representative_examples": fold_representative_examples,
        "extra": extra,
    }


def render_loto_heldout_table(results: Dict[str, Any], decoding: str = "greedy") -> str:
    """
    Simple markdown table: each row is a held-out task, showing baseline vs shared_full on held-out only.

    If the held-out task was evaluated with forced-choice, pass decoding="forced_choice".
    """
    rows = []
    header = ["Held-out", "n", "Protocol", "Baseline", "Shared(full)", "Rand(full)", "Δ(shared-baseline)", "p(shared-baseline)"]
    for holdout, fold in results.get("folds", {}).items():
        block = fold["by_dataset"].get(holdout, None)
        if block is None:
            continue

        protocol = block.get("protocol", "generation")
        if protocol == "forced_choice":
            key_prefix = "forced_choice"
            b = block["runs"][f"{key_prefix}/baseline"]
            s = block["runs"][f"{key_prefix}/shared_full"]
            r = block["runs"][f"{key_prefix}/rand_full"]
            stat = block["paired_tests"][key_prefix]["shared_full_vs_baseline"]
        else:
            key_prefix = decoding
            b = block["runs"][f"{key_prefix}/baseline"]
            s = block["runs"][f"{key_prefix}/shared_full"]
            r = block["runs"][f"{key_prefix}/rand_full"]
            stat = block["paired_tests"][key_prefix]["shared_full_vs_baseline"]

        rows.append([
            holdout,
            str(block["n"]),
            protocol,
            fmt_acc(b["accuracy"], b["ci_low"], b["ci_high"]),
            fmt_acc(s["accuracy"], s["ci_low"], s["ci_high"]),
            fmt_acc(r["accuracy"], r["ci_low"], r["ci_high"]),
            f"{stat['mean_diff']*100:+.1f} [{stat['ci_low']*100:+.1f}, {stat['ci_high']*100:+.1f}]",
            f"{stat['p_value']:.3g}",
        ])

    cols = list(zip(*([header] + rows))) if rows else [header]
    widths = [max(len(str(x)) for x in col) for col in cols]

    def fmt_row(r):
        return "| " + " | ".join(str(x).ljust(w) for x, w in zip(r, widths)) + " |"

    lines = [fmt_row(header), "|-" + "-|-".join("-"*w for w in widths) + "-|"]
    for r in rows:
        lines.append(fmt_row(r))
    return "\n".join(lines)


def infer_hidden_dim(model) -> Optional[int]:
    cfg = getattr(model, "config", None)

    # common fields
    for k in ("hidden_size", "n_embd", "dim", "d_model", "model_dim", "embed_dim"):
        v = getattr(cfg, k, None)
        if isinstance(v, int) and v > 0:
            return v

    # multimodal / nested text_config
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is not None:
        for k in ("hidden_size", "n_embd", "dim", "d_model", "model_dim", "embed_dim"):
            v = getattr(text_cfg, k, None)
            if isinstance(v, int) and v > 0:
                return v

    # fallback: embedding weight
    try:
        emb = model.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight") and isinstance(emb.weight, torch.Tensor) and emb.weight.ndim == 2:
            return int(emb.weight.shape[1])
        if emb is not None and hasattr(emb, "embedding_dim"):
            return int(emb.embedding_dim)
    except Exception:
        pass

    return None


# # -----------------------------
# # Main
# # -----------------------------
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
#     ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
#     ap.add_argument("--model_dtype", type=str, default="fp32", choices=["fp32", "fp16"])
#     ap.add_argument("--layer", type=int, default=10)
#     ap.add_argument("--tasks", type=str, default="gsm8k,commonsenseqa,strategyqa,aqua,arc_challenge,openbookqa,qasc,logiqa")
#     ap.add_argument("--mode", type=str, default="loto", choices=["all", "loto"])
#     ap.add_argument("--loto_eval_mode", type=str, default="heldout", choices=["heldout", "all"])
#     ap.add_argument("--loto_only", type=str, default="", help="Optional: only run this held-out task (e.g., 'gsm8k'). Empty means run all folds.")

#     # Subspace estimation sizes
#     ap.add_argument("--n_subspace", type=int, default=128)
#     ap.add_argument("--n_eval", type=int, default=256)

#     # PCA/sharedness
#     ap.add_argument("--pca_var", type=float, default=0.95)
#     ap.add_argument("--min_dim", type=int, default=1)
#     ap.add_argument("--max_dim", type=int, default=4096)
#     ap.add_argument("--tau", type=float, default=0.001, help="Relative threshold for shared component selection.")
#     ap.add_argument("--m_shared", type=str, default="all", help="Sharedness requirement: 'all' or an int (>=2).")

#     # Activation collection
#     ap.add_argument("--calib_decode_max_new_tokens", type=int, default=128)
#     ap.add_argument("--per_task_max_states", type=int, default=20000)

#     # Intervention
#     ap.add_argument("--alpha_remove", type=float, default=1.0)
#     ap.add_argument("--reasoning_tokens", type=int, default=128)
#     ap.add_argument("--max_new_tokens", type=int, default=256)

#     # Decoding (generation)
#     ap.add_argument("--temperature", type=float, default=0.7)
#     ap.add_argument("--top_p", type=float, default=0.9)
#     ap.add_argument("--top_k", type=int, default=0)
#     ap.add_argument("--batch_size", type=int, default=4)
#     ap.add_argument("--max_prompt_len", type=int, default=512)
#     ap.add_argument("--do_sample", type=int, default=0, choices=[0, 1])

#     # Random controls
#     ap.add_argument("--rand_type", type=str, default="joint_nonshared_varmatch",
#                     choices=["joint_nonshared_uniform", "joint_nonshared_topk", "joint_nonshared_varmatch"])

#     # Template randomization
#     ap.add_argument("--template_randomization", type=int, default=1, choices=[0, 1])
#     ap.add_argument("--template_seed", type=int, default=1234)
#     ap.add_argument("--shuffle_choices", type=int, default=0, choices=[0, 1])
#     ap.add_argument("--add_answer_prefix", type=int, default=0, choices=[0, 1])
#     ap.add_argument("--answer_prefix", type=str, default="\nFinal answer:")

#     # Forced-choice (new)
#     ap.add_argument("--use_forced_choice", type=int, default=0, choices=[0, 1], help="If 1, use forced-choice for tasks with discrete candidates (MC/YesNo). gsm8k stays generation.")
#     ap.add_argument("--fc_warmup_tokens", type=int, default=0, help="Teacher-forced warmup tokens before scoring candidates (baseline-generated, shared across conditions).")
#     ap.add_argument("--fc_warmup_decoding", type=str, default="greedy", choices=["greedy", "sample"])
#     ap.add_argument("--fc_warmup_seed", type=int, default=123)
#     ap.add_argument("--fc_warmup_ban_eos", type=int, default=1, choices=[0, 1])
#     ap.add_argument("--fc_warmup_temperature", type=float, default=0.7)
#     ap.add_argument("--fc_warmup_top_p", type=float, default=0.9)
#     ap.add_argument("--fc_warmup_top_k", type=int, default=0)
#     ap.add_argument("--fc_prefix_mode", type=str, default="auto", choices=["auto", "always", "never"])
#     ap.add_argument("--fc_answer_prefix", type=str, default="\nFinal answer:")
#     ap.add_argument("--fc_debug_print", type=int, default=0, choices=[0, 1], help="If 1, print warmup demo text and other FC diagnostics.")

#     # Stats
#     ap.add_argument("--bootstrap_iters", type=int, default=5000)
#     ap.add_argument("--perm_iters", type=int, default=10000)
#     ap.add_argument("--ci_alpha", type=float, default=0.05)
#     ap.add_argument("--seed", type=int, default=42)
#     ap.add_argument("--sample_seed", type=int, default=12345)

#     # Output
#     ap.add_argument("--out_json", type=str, default=os.path.join(THIS_DIR, "energy_balance_loto8_reasoning_results.json"))
#     ap.add_argument("--out_md", type=str, default=os.path.join(THIS_DIR, "energy_balance_loto8_reasoning_summary.md"))

#     args = ap.parse_args()
#     set_global_seed(args.seed)

#     # Normalize prefix args
#     args.answer_prefix = _norm_prefix_arg(args.answer_prefix)
#     args.fc_answer_prefix = _norm_prefix_arg(args.fc_answer_prefix)

#     tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
#     if len(tasks) < 2:
#         raise RuntimeError("Need at least 2 tasks in --tasks.")

#     args.do_sample = bool(args.do_sample)
#     args.template_randomization = bool(args.template_randomization)
#     args.shuffle_choices = bool(args.shuffle_choices)
#     args.add_answer_prefix = bool(args.add_answer_prefix)
#     args.use_forced_choice = bool(args.use_forced_choice)
#     args.fc_debug_print = bool(args.fc_debug_print)

#     # Guardrails for forced-choice
#     if args.use_forced_choice and args.fc_warmup_tokens > 0 and args.fc_prefix_mode == "never" and args.fc_answer_prefix:
#         print(
#             "[Warn][ForcedChoice] fc_warmup_tokens>0 but fc_prefix_mode=never.\n"
#             "  This often evaluates at a 'random mid-continuation' position and can yield chance-level baselines.\n"
#             "  Recommended: --fc_prefix_mode auto (default) or always."
#         )

#     layer_indices = [args.layer]

#     print(f"[Env] DEVICE={args.device}")
#     print(f"[Env] MODEL={args.model} dtype={args.model_dtype}")
#     print(f"[Env] layer_indices={layer_indices}")
#     print(f"[Env] tasks={tasks}")
#     print(f"[Env] mode={args.mode} loto_eval_mode={args.loto_eval_mode}")
#     print(f"[Env] template_randomization={args.template_randomization} shuffle_choices={args.shuffle_choices} add_answer_prefix={args.add_answer_prefix}")
#     print(f"[Env] forced_choice={args.use_forced_choice} fc_warmup_tokens={args.fc_warmup_tokens} fc_prefix_mode={args.fc_prefix_mode} fc_answer_prefix={args.fc_answer_prefix!r}")

#     model, tokenizer = load_model_and_tokenizer(args.model, args.device, args.model_dtype)

#     hidden_dim = infer_hidden_dim(model)
#     if hidden_dim is None:
#         print(f"[Warn] Could not infer hidden_dim (config_class={type(model.config)}). Continue anyway.")
#     else:
#         print(f"[Env] hidden_dim={hidden_dim}")

#     # Load datasets using benchmark_dataloaders
#     sub_by, eval_by, meta_by = load_selected_tasks(
#         tasks=tasks,
#         n_subspace=args.n_subspace,
#         n_eval=args.n_eval,
#         seed=args.seed,
#         template_seed=args.template_seed,
#         template_randomization=args.template_randomization,
#         shuffle_choices=args.shuffle_choices,
#         add_answer_prefix=args.add_answer_prefix,
#         answer_prefix=args.answer_prefix,
#     )
#     print("\n" + "=" * 80)
#     print(f"[Data] Loaded tasks: {list(sub_by.keys())}")
#     print(f"[Data] Meta: {json.dumps(meta_by, indent=2, ensure_ascii=False)}")
#     print("=" * 80)

#     results: Dict[str, Any] = {
#         "config": {
#             "model": args.model,
#             "device": args.device,
#             "model_dtype": args.model_dtype,
#             "layer_indices": layer_indices,
#             "tasks": tasks,
#             "mode": args.mode,
#             "loto_eval_mode": args.loto_eval_mode,
#             "n_subspace": args.n_subspace,
#             "n_eval": args.n_eval,
#             "pca_var": args.pca_var,
#             "tau": args.tau,
#             "m_shared": args.m_shared,
#             "per_task_max_states": args.per_task_max_states,
#             "calib_decode_max_new_tokens": args.calib_decode_max_new_tokens,
#             "reasoning_tokens": args.reasoning_tokens,
#             "max_new_tokens": args.max_new_tokens,
#             "alpha_remove": args.alpha_remove,
#             "rand_type": args.rand_type,
#             "template_randomization": args.template_randomization,
#             "template_seed": args.template_seed,
#             "shuffle_choices": args.shuffle_choices,
#             "add_answer_prefix": args.add_answer_prefix,
#             "answer_prefix": args.answer_prefix,
#             "do_sample": args.do_sample,
#             "temperature": args.temperature,
#             "top_p": args.top_p,
#             "top_k": args.top_k,
#             "batch_size": args.batch_size,
#             "max_prompt_len": args.max_prompt_len,
#             "bootstrap_iters": args.bootstrap_iters,
#             "perm_iters": args.perm_iters,
#             "ci_alpha": args.ci_alpha,
#             "seed": args.seed,
#             "sample_seed": args.sample_seed,
#             "dataset_meta": meta_by,
#             "forced_choice": {
#                 "use_forced_choice": args.use_forced_choice,
#                 "fc_warmup_tokens": args.fc_warmup_tokens,
#                 "fc_warmup_decoding": args.fc_warmup_decoding,
#                 "fc_warmup_seed": args.fc_warmup_seed,
#                 "fc_warmup_ban_eos": bool(args.fc_warmup_ban_eos),
#                 "fc_warmup_temperature": args.fc_warmup_temperature,
#                 "fc_warmup_top_p": args.fc_warmup_top_p,
#                 "fc_warmup_top_k": args.fc_warmup_top_k,
#                 "fc_prefix_mode": args.fc_prefix_mode,
#                 "fc_answer_prefix": args.fc_answer_prefix,
#             },
#         }
#     }

#     if args.mode == "all":
#         fold = run_fold(
#             fold_name="all_tasks",
#             model=model,
#             tokenizer=tokenizer,
#             sub_by=sub_by,
#             eval_by=eval_by,
#             train_tasks=tasks,
#             eval_tasks=tasks,
#             layer_indices=layer_indices,
#             args=args,
#         )
#         results["all_tasks"] = fold

#     else:
#         folds = {}
#         for holdout in tasks:
#             if args.loto_only and holdout != args.loto_only:
#                 continue
#             train_tasks = [t for t in tasks if t != holdout]
#             eval_tasks = [holdout] if args.loto_eval_mode == "heldout" else list(tasks)
#             fold_name = f"loto_holdout={holdout}"
#             print("\n" + "=" * 90)
#             print(f"[LOTO] Running fold: holdout={holdout} train={train_tasks} eval={eval_tasks}")
#             print("=" * 90)

#             fold = run_fold(
#                 fold_name=fold_name,
#                 model=model,
#                 tokenizer=tokenizer,
#                 sub_by=sub_by,
#                 eval_by=eval_by,
#                 train_tasks=train_tasks,
#                 eval_tasks=eval_tasks,
#                 layer_indices=layer_indices,
#                 args=args,
#             )
#             folds[holdout] = fold

#             # Mild hygiene between folds
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()

#         results["folds"] = folds

#     # Save JSON
#     with open(args.out_json, "w", encoding="utf-8") as f:
#         json.dump(results, f, ensure_ascii=False, indent=2, default=json_default)

#     # Save a small markdown summary (especially useful for LOTO heldout-mode)
#     md_lines = []
#     md_lines.append("# Energy-balance + LOTO(8) Summary\n")
#     md_lines.append(f"- Model: `{args.model}` dtype={args.model_dtype} device={args.device}\n")
#     md_lines.append(f"- Tasks: {tasks}\n")
#     md_lines.append(f"- Mode: {args.mode}\n")
#     md_lines.append(f"- Template randomization: {args.template_randomization} (seed={args.template_seed}), shuffle_choices={args.shuffle_choices}\n")
#     md_lines.append(f"- Sharedness: pca_var={args.pca_var}, tau={args.tau}, m_shared={args.m_shared}\n")
#     md_lines.append(f"- Calibration decode max_new_tokens={args.calib_decode_max_new_tokens}, per_task_max_states={args.per_task_max_states}\n")
#     md_lines.append(f"- Evaluation: forced_choice={args.use_forced_choice} (MC tasks only)\n")
#     md_lines.append("")

#     if args.mode == "loto" and args.loto_eval_mode == "heldout" and "folds" in results:
#         md_lines.append("## LOTO held-out performance\n")
#         # If forced-choice is enabled, many heldouts (except gsm8k) will be forced-choice
#         md_lines.append(render_loto_heldout_table(results, decoding="greedy"))
#         md_lines.append("")

#     with open(args.out_md, "w", encoding="utf-8") as f:
#         f.write("\n".join(md_lines))

#     print("\n" + "=" * 80)
#     print("[Done]")
#     print(f"[Done] JSON: {args.out_json}")
#     print(f"[Done] MD  : {args.out_md}")
#     print("=" * 80)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model_dtype", type=str, default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--tasks", type=str, default="gsm8k,commonsenseqa,strategyqa,aqua,arc_challenge,openbookqa,qasc,logiqa")
    ap.add_argument("--mode", type=str, default="loto", choices=["all", "loto"])
    ap.add_argument("--loto_eval_mode", type=str, default="heldout", choices=["heldout", "all"])
    ap.add_argument("--loto_only", type=str, default="", help="Optional: only run this held-out task (e.g., 'gsm8k'). Empty means run all folds.")

    # Subspace estimation sizes
    ap.add_argument("--n_subspace", type=int, default=128)

    # ✅ 改动1：默认 n_eval=2048（你注释里说这是正确设置）
    ap.add_argument("--n_eval", type=int, default=2048)

    # PCA/sharedness
    ap.add_argument("--pca_var", type=float, default=0.95)
    ap.add_argument("--min_dim", type=int, default=1)
    ap.add_argument("--max_dim", type=int, default=4096)
    ap.add_argument("--tau", type=float, default=0.001, help="Relative threshold for shared component selection.")
    ap.add_argument("--m_shared", type=str, default="all", help="Sharedness requirement: 'all' or an int (>=2).")

    # Activation collection
    ap.add_argument("--calib_decode_max_new_tokens", type=int, default=128)
    ap.add_argument("--per_task_max_states", type=int, default=20000)

    # Intervention
    ap.add_argument("--alpha_remove", type=float, default=1.0)
    ap.add_argument("--reasoning_tokens", type=int, default=128)
    ap.add_argument("--max_new_tokens", type=int, default=256)

    # Decoding (generation)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_prompt_len", type=int, default=512)
    ap.add_argument("--do_sample", type=int, default=0, choices=[0, 1])

    # Random controls
    ap.add_argument("--rand_type", type=str, default="joint_nonshared_varmatch",
                    choices=["joint_nonshared_uniform", "joint_nonshared_topk", "joint_nonshared_varmatch"])

    # Template randomization
    ap.add_argument("--template_randomization", type=int, default=1, choices=[0, 1])
    ap.add_argument("--template_seed", type=int, default=1234)
    ap.add_argument("--shuffle_choices", type=int, default=0, choices=[0, 1])
    ap.add_argument("--add_answer_prefix", type=int, default=0, choices=[0, 1])
    ap.add_argument("--answer_prefix", type=str, default="\nFinal answer:")

    # Forced-choice (new)
    # ✅ 改动2：默认 use_forced_choice=1（你注释里说这是正确设置）
    ap.add_argument("--use_forced_choice", type=int, default=1, choices=[0, 1],
                    help="If 1, use forced-choice for tasks with discrete candidates (MC/YesNo). gsm8k stays generation.")
    ap.add_argument("--fc_warmup_tokens", type=int, default=0,
                    help="Teacher-forced warmup tokens before scoring candidates (baseline-generated, shared across conditions).")
    ap.add_argument("--fc_warmup_decoding", type=str, default="greedy", choices=["greedy", "sample"])
    ap.add_argument("--fc_warmup_seed", type=int, default=123)
    ap.add_argument("--fc_warmup_ban_eos", type=int, default=1, choices=[0, 1])
    ap.add_argument("--fc_warmup_temperature", type=float, default=0.7)
    ap.add_argument("--fc_warmup_top_p", type=float, default=0.9)
    ap.add_argument("--fc_warmup_top_k", type=int, default=0)
    ap.add_argument("--fc_prefix_mode", type=str, default="auto", choices=["auto", "always", "never"])
    ap.add_argument("--fc_answer_prefix", type=str, default="\nFinal answer:")
    ap.add_argument("--fc_debug_print", type=int, default=0, choices=[0, 1],
                    help="If 1, print warmup demo text and other FC diagnostics.")

    # Stats
    ap.add_argument("--bootstrap_iters", type=int, default=5000)
    ap.add_argument("--perm_iters", type=int, default=10000)
    ap.add_argument("--ci_alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample_seed", type=int, default=12345)

    # Optional generation logging
    ap.add_argument("--save_generation_details", type=int, default=0, choices=[0, 1],
                    help="If 1, save prompts/preds/continuations for generation runs into the output JSON.")
    ap.add_argument("--save_generation_examples", type=int, default=0,
                    help="If >0, save up to this many representative generation examples per generation dataset.")
    ap.add_argument("--generation_prompt_char_limit", type=int, default=500)
    ap.add_argument("--generation_continuation_char_limit", type=int, default=1200)
    ap.add_argument("--out_examples_jsonl", type=str, default="",
                    help="Optional sidecar JSONL path for representative generation examples. "
                         "If empty and save_generation_examples>0, derive from out_json.")

    # Output
    ap.add_argument("--out_json", type=str, default=os.path.join(THIS_DIR, "energy_balance_loto8_reasoning_results.json"))
    ap.add_argument("--out_md", type=str, default=os.path.join(THIS_DIR, "energy_balance_loto8_reasoning_summary.md"))

    args = ap.parse_args()
    set_global_seed(args.seed)

    # Normalize prefix args
    args.answer_prefix = _norm_prefix_arg(args.answer_prefix)
    args.fc_answer_prefix = _norm_prefix_arg(args.fc_answer_prefix)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if len(tasks) < 2:
        raise RuntimeError("Need at least 2 tasks in --tasks.")

    args.do_sample = bool(args.do_sample)
    args.template_randomization = bool(args.template_randomization)
    args.shuffle_choices = bool(args.shuffle_choices)
    args.add_answer_prefix = bool(args.add_answer_prefix)
    args.use_forced_choice = bool(args.use_forced_choice)
    args.fc_debug_print = bool(args.fc_debug_print)
    args.save_generation_details = bool(args.save_generation_details)

    if args.save_generation_examples > 0 and not args.out_examples_jsonl:
        args.out_examples_jsonl = os.path.splitext(args.out_json)[0] + ".examples.jsonl"

    # Guardrails for forced-choice
    if args.use_forced_choice and args.fc_warmup_tokens > 0 and args.fc_prefix_mode == "never" and args.fc_answer_prefix:
        print(
            "[Warn][ForcedChoice] fc_warmup_tokens>0 but fc_prefix_mode=never.\n"
            "  This often evaluates at a 'random mid-continuation' position and can yield chance-level baselines.\n"
            "  Recommended: --fc_prefix_mode auto (default) or always."
        )

    layer_indices = [args.layer]

    print(f"[Env] DEVICE={args.device}")
    print(f"[Env] MODEL={args.model} dtype={args.model_dtype}")
    print(f"[Env] layer_indices={layer_indices}")
    print(f"[Env] tasks={tasks}")
    print(f"[Env] mode={args.mode} loto_eval_mode={args.loto_eval_mode}")
    print(f"[Env] template_randomization={args.template_randomization} shuffle_choices={args.shuffle_choices} add_answer_prefix={args.add_answer_prefix}")
    print(f"[Env] forced_choice={args.use_forced_choice} fc_warmup_tokens={args.fc_warmup_tokens} fc_prefix_mode={args.fc_prefix_mode} fc_answer_prefix={args.fc_answer_prefix!r}")

    model, tokenizer = load_model_and_tokenizer(args.model, args.device, args.model_dtype)

    hidden_dim = infer_hidden_dim(model)
    if hidden_dim is None:
        print(f"[Warn] Could not infer hidden_dim (config_class={type(model.config)}). Continue anyway.")
    else:
        print(f"[Env] hidden_dim={hidden_dim}")

    # Load datasets using benchmark_dataloaders
    sub_by, eval_by, meta_by = load_selected_tasks(
        tasks=tasks,
        n_subspace=args.n_subspace,
        n_eval=args.n_eval,
        seed=args.seed,
        template_seed=args.template_seed,
        template_randomization=args.template_randomization,
        shuffle_choices=args.shuffle_choices,
        add_answer_prefix=args.add_answer_prefix,
        answer_prefix=args.answer_prefix,
    )
    print("\n" + "=" * 80)
    print(f"[Data] Loaded tasks: {list(sub_by.keys())}")
    print(f"[Data] Meta: {json.dumps(meta_by, indent=2, ensure_ascii=False)}")
    print("=" * 80)

    results: Dict[str, Any] = {
        "config": {
            "model": args.model,
            "device": args.device,
            "model_dtype": args.model_dtype,
            "layer_indices": layer_indices,
            "tasks": tasks,
            "mode": args.mode,
            "loto_eval_mode": args.loto_eval_mode,
            "n_subspace": args.n_subspace,
            "n_eval": args.n_eval,
            "pca_var": args.pca_var,
            "tau": args.tau,
            "m_shared": args.m_shared,
            "per_task_max_states": args.per_task_max_states,
            "calib_decode_max_new_tokens": args.calib_decode_max_new_tokens,
            "reasoning_tokens": args.reasoning_tokens,
            "max_new_tokens": args.max_new_tokens,
            "alpha_remove": args.alpha_remove,
            "rand_type": args.rand_type,
            "template_randomization": args.template_randomization,
            "template_seed": args.template_seed,
            "shuffle_choices": args.shuffle_choices,
            "add_answer_prefix": args.add_answer_prefix,
            "answer_prefix": args.answer_prefix,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "batch_size": args.batch_size,
            "max_prompt_len": args.max_prompt_len,
            "bootstrap_iters": args.bootstrap_iters,
            "perm_iters": args.perm_iters,
            "ci_alpha": args.ci_alpha,
            "seed": args.seed,
            "sample_seed": args.sample_seed,
            "dataset_meta": meta_by,
            "forced_choice": {
                "use_forced_choice": args.use_forced_choice,
                "fc_warmup_tokens": args.fc_warmup_tokens,
                "fc_warmup_decoding": args.fc_warmup_decoding,
                "fc_warmup_seed": args.fc_warmup_seed,
                "fc_warmup_ban_eos": bool(args.fc_warmup_ban_eos),
                "fc_warmup_temperature": args.fc_warmup_temperature,
                "fc_warmup_top_p": args.fc_warmup_top_p,
                "fc_warmup_top_k": args.fc_warmup_top_k,
                "fc_prefix_mode": args.fc_prefix_mode,
                "fc_answer_prefix": args.fc_answer_prefix,
            },
            "generation_logging": {
                "save_generation_details": args.save_generation_details,
                "save_generation_examples": args.save_generation_examples,
                "generation_prompt_char_limit": args.generation_prompt_char_limit,
                "generation_continuation_char_limit": args.generation_continuation_char_limit,
                "out_examples_jsonl": args.out_examples_jsonl,
            },
        }
    }

    if args.mode == "all":
        fold = run_fold(
            fold_name="all_tasks",
            model=model,
            tokenizer=tokenizer,
            sub_by=sub_by,
            eval_by=eval_by,
            train_tasks=tasks,
            eval_tasks=tasks,
            layer_indices=layer_indices,
            args=args,
        )
        results["all_tasks"] = fold

    else:
        folds = {}
        for holdout in tasks:
            if args.loto_only and holdout != args.loto_only:
                continue
            train_tasks = [t for t in tasks if t != holdout]
            eval_tasks = [holdout] if args.loto_eval_mode == "heldout" else list(tasks)
            fold_name = f"loto_holdout={holdout}"
            print("\n" + "=" * 90)
            print(f"[LOTO] Running fold: holdout={holdout} train={train_tasks} eval={eval_tasks}")
            print("=" * 90)

            fold = run_fold(
                fold_name=fold_name,
                model=model,
                tokenizer=tokenizer,
                sub_by=sub_by,
                eval_by=eval_by,
                train_tasks=train_tasks,
                eval_tasks=eval_tasks,
                layer_indices=layer_indices,
                args=args,
            )
            folds[holdout] = fold

            # Mild hygiene between folds
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        results["folds"] = folds

    # Save JSON
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=json_default)

    if args.save_generation_examples > 0 and args.out_examples_jsonl:
        rep_examples = []
        for fold in results.get("folds", {}).values():
            for item in fold.get("representative_examples", []):
                rep_examples.append(item)
        if "all_tasks" in results:
            rep_examples.extend(results["all_tasks"].get("representative_examples", []))
        with open(args.out_examples_jsonl, "w", encoding="utf-8") as f:
            for item in rep_examples:
                f.write(json.dumps(item, ensure_ascii=False, default=json_default) + "\n")

    # Save a small markdown summary (especially useful for LOTO heldout-mode)
    md_lines = []
    md_lines.append("# Energy-balance + LOTO(8) Summary\n")
    md_lines.append(f"- Model: `{args.model}` dtype={args.model_dtype} device={args.device}\n")
    md_lines.append(f"- Tasks: {tasks}\n")
    md_lines.append(f"- Mode: {args.mode}\n")
    md_lines.append(f"- Template randomization: {args.template_randomization} (seed={args.template_seed}), shuffle_choices={args.shuffle_choices}\n")
    md_lines.append(f"- Sharedness: pca_var={args.pca_var}, tau={args.tau}, m_shared={args.m_shared}\n")
    md_lines.append(f"- Calibration decode max_new_tokens={args.calib_decode_max_new_tokens}, per_task_max_states={args.per_task_max_states}\n")
    md_lines.append(f"- Evaluation: forced_choice={args.use_forced_choice} (MC tasks only)\n")
    md_lines.append("")

    if args.mode == "loto" and args.loto_eval_mode == "heldout" and "folds" in results:
        md_lines.append("## LOTO held-out performance\n")
        md_lines.append(render_loto_heldout_table(results, decoding="greedy"))
        md_lines.append("")

    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print("\n" + "=" * 80)
    print("[Done]")
    print(f"[Done] JSON: {args.out_json}")
    print(f"[Done] MD  : {args.out_md}")
    print("=" * 80)


if __name__ == "__main__":
    main()
