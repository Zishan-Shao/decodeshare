# -*- coding: utf-8 -*-
"""
exp_unique_vs_IH_TC.py

Goal: argue your shared-subspace intervention is *not* reducible to
(1) Induction Heads (IH) and (2) Circuit-style sparse head subsets (TC).

SOLIDIFY PATCHES (3 changes):
  (1) Shared-subspace definition robustness:
      - default max_dim increased (avoid truncating PCA below pca_var target)
      - tau can be auto-scaled by realized cross_dim (tau_mode=auto) so threshold remains conservative
        even when cross_dim is small (e.g., k=512).
  (2) Match mainline A3 estimator:
      - balance per-task counts to n_min per layer
      - task-center each task's states before pooled PCA
  (3) TC-search scoring:
      - replace discrete accuracy-drop head scoring with continuous logprob-margin drop (default)
        to avoid degenerate all-zeros head scores.

We compare three interventions at *decode-time*:
  - Ours: remove shared decode-aligned basis Q_shared at a chosen layer.
  - IH baseline: detect induction-like heads (attn-based if available; else decode-aligned ablation probe).
  - TC baseline: select heads causally important for an anchor task by single-head ablation scoring.

Metrics:
  - Generation accuracy (existing CoT generation + answer extraction).
  - Forced-choice accuracy (cache-advanced prompt-boundary alignment; decision-level evaluation).
  - Optional overlap analysis between Q_shared and head-write subspaces (o_proj column subspace).

This script imports and reuses utilities from your existing pipeline file:
  - disturb_CoT_shared_acc_lasttoken_*_loto8.py
and benchmark dataloaders.

Run (example):
    CUDA_VISIBLE_DEVICES=1 python exp_unique_vs_IH_TC.py \
        --model meta-llama/Llama-2-7b-chat-hf --device cuda --model_dtype fp32 \
        --tasks gsm8k,commonsenseqa,strategyqa,openbookqa,qasc,boolq,piqa \
        --layer 10 --n_subspace 128 --n_eval 256 \
        --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
        --reasoning_tokens 128 --max_new_tokens 256 \
        --do_sample 0 \
        --ih_topk 4 --tc_topk 4 --tc_anchor_task commonsenseqa \
        --max_dim 4096 --tau_mode auto --tau_ref_dim 3000 \
        --balance_tasks 1 --task_center 1 \
        --tc_score margin

    
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

# -----------------------------------------------------------------------------
# Import your existing pipeline bits
# -----------------------------------------------------------------------------
# Prefer the "energy_balanced" filename if present; fallback to your fp32 sanity script.
try:
    from disturb_CoT_shared_acc_lasttoken_energy_balanced_loto8 import (  # type: ignore
        HookStats,
        GenerationState,
        load_model_and_tokenizer,
        register_hooks_for_condition,
        remove_hooks,
        generate_continuations,
        bootstrap_ci_mean,
        summarize_paired,
        orthonormalize_np,
        DecodeLastTokenActivationCollector,
        collect_decode_last_token_states,
    )
except Exception:
    from disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8 import (  # type: ignore
        HookStats,
        GenerationState,
        load_model_and_tokenizer,
        register_hooks_for_condition,
        remove_hooks,
        generate_continuations,
        bootstrap_ci_mean,
        summarize_paired,
        orthonormalize_np,
        DecodeLastTokenActivationCollector,
        collect_decode_last_token_states,
    )

# -----------------------------------------------------------------------------
# Import task loading from benchmark dataloaders
# -----------------------------------------------------------------------------
# Keep your original import, but add a safe fallback to your uploaded module name.
try:
    from decodeshare.benchmark_dataloaders import (  # type: ignore
        Example,
        load_selected_tasks,
        parse_prediction,
        is_correct,
        stable_int_seed,
    )
except Exception:
    from decodeshare.benchmark_dataloaders import (  # type: ignore
        Example,
        load_selected_tasks,
        parse_prediction,
        is_correct,
        stable_int_seed,
    )

# -----------------------------------------------------------------------------
# Import cross-task PCA + sharedness scorer (same as your main pipeline)
# -----------------------------------------------------------------------------
from decodeshare.joint_subspace_large.disturb_cross_task_all_shared import (  # type: ignore
    get_model_layers,
    compute_cross_task_subspace,
    find_fully_shared_basis_improved,
)

# -----------------------------
# Helpers: numeric utilities
# -----------------------------
def _subsample_rows_np(x: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Deterministic subsample without replacement to exactly n rows."""
    if x is None:
        return x
    if n <= 0 or x.shape[0] <= n:
        return x
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(x.shape[0], size=int(n), replace=False)
    return x[idx]


def _logsumexp(vals: List[float]) -> float:
    """Stable logsumexp over python floats."""
    if len(vals) == 0:
        return float("-inf")
    m = max(vals)
    if not math.isfinite(m):
        return m
    s = 0.0
    for v in vals:
        s += math.exp(v - m)
    return m + math.log(max(s, 1e-300))


def _effective_tau(base_tau: float, cross_dim: int, *, mode: str, ref_dim: int) -> float:
    """
    tau scaling heuristic:
      - fixed: use base_tau
      - auto: tau_eff = base_tau * (ref_dim / cross_dim)

    Motivation: tau=1e-3 is conservative when k≈2500–3000; for smaller k we should
    raise tau to avoid "everything is shared" artifacts.
    """
    base_tau = float(base_tau)
    cross_dim = int(max(cross_dim, 1))
    ref_dim = int(max(ref_dim, 1))
    if mode == "fixed":
        tau_eff = base_tau
    else:
        tau_eff = base_tau * (float(ref_dim) / float(cross_dim))
    # clamp to sane range
    tau_eff = float(min(max(tau_eff, 1e-6), 0.2))
    return tau_eff


def _min_tasks_from_m_shared(m_shared: str, tasks: List[str]) -> int:
    ms = str(m_shared).strip().lower()
    if ms == "all":
        return len(tasks)
    if ms == "half":
        return max(2, int(math.ceil(len(tasks) / 2)))
    # try integer
    try:
        v = int(ms)
        return max(2, min(v, len(tasks)))
    except Exception:
        return len(tasks)


def collect_task_activations_decode_only(
    model: torch.nn.Module,
    tokenizer,
    prompts_by_task: Dict[str, List[str]],
    *,
    layer_indices: List[int],
    calib_decoding: str,
    calib_batch_size: int,
    calib_max_new_tokens: int,
    per_task_max_states: int,
    max_prompt_len: int,
    temperature: float,
    top_p: float,
    top_k: int,
    global_seed: int,
) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Collect decode-time (seq_len==1) last-token states per task and per layer,
    without performing PCA inside this helper.

    Returns:
      task_activations[task][layer] = np.ndarray [n_states, d]
    """
    layers, _ = get_model_layers(model)
    collector = DecodeLastTokenActivationCollector(layer_indices)

    handles = []
    for layer_idx in layer_indices:
        if layer_idx >= len(layers):
            print(f"[Collect-A3] Warn: layer_idx={layer_idx} out of range, skipping")
            continue
        handles.append(layers[layer_idx].register_forward_hook(collector.make_hook(layer_idx)))

    try:
        for task_name, prompts in prompts_by_task.items():
            print(f"[Collect-A3] Task={task_name}, prompts={len(prompts)}")
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
        layer_dict: Dict[int, np.ndarray] = {}
        for layer_idx in layer_indices:
            acts = collector.get_task_activations(task_name, layer_idx)
            if acts is None or acts.shape[0] == 0:
                continue
            # cap per-task states
            ss = stable_int_seed(global_seed, task_name, layer_idx, "cap_states")
            acts = _subsample_rows_np(acts, int(per_task_max_states), seed=ss)
            layer_dict[int(layer_idx)] = acts
            print(f"[Collect-A3]  collected {task_name} layer={layer_idx}: {acts.shape[0]} x {acts.shape[1]}")
        if layer_dict:
            task_activations[task_name] = layer_dict

    if not task_activations:
        raise RuntimeError("[Collect-A3] No decode activations collected. Check hooks/layers/generation loop.")
    return task_activations


def balance_and_task_center(
    task_activations: Dict[str, Dict[int, np.ndarray]],
    *,
    layer_indices: List[int],
    tasks_order: List[str],
    do_balance: bool,
    do_center: bool,
    global_seed: int,
) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Apply mainline A3 preprocessing:
      - (optional) balance each task to n_min per layer
      - (optional) task-center: subtract per-task mean vector
    """
    out: Dict[str, Dict[int, np.ndarray]] = {t: {} for t in tasks_order if t in task_activations}

    for layer_idx in layer_indices:
        # gather available tasks for this layer
        avail = []
        for t in tasks_order:
            if t in task_activations and layer_idx in task_activations[t]:
                a = task_activations[t][layer_idx]
                if a is not None and a.shape[0] > 0:
                    avail.append(t)

        if len(avail) == 0:
            continue

        n_min = None
        if do_balance:
            n_min = min(int(task_activations[t][layer_idx].shape[0]) for t in avail)
            n_min = int(max(n_min, 1))
            print(f"[A3] layer={layer_idx} balance enabled: n_min={n_min} across tasks={avail}")

        for t in avail:
            a = task_activations[t][layer_idx]
            a2 = a
            if do_balance and n_min is not None:
                ss = stable_int_seed(global_seed, t, layer_idx, "balance_min")
                a2 = _subsample_rows_np(a2, n_min, seed=ss)

            if do_center:
                mu = a2.mean(axis=0, keepdims=True)
                a2 = a2 - mu

            out[t][layer_idx] = a2

    # prune empties
    out = {t: ld for t, ld in out.items() if len(ld) > 0}
    if not out:
        raise RuntimeError("[A3] After balance/center, no activations remain.")
    return out


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
    "strategyqa": ["YES", "NO"],
    # gsm8k is not multiple-choice here; we skip forced-choice by default
}


def candidate_texts_for_task(task: str) -> Tuple[List[str], List[str]]:
    """
    Return (labels, texts_to_append).
    We append a leading space so that tokenization is stable after 'Final answer:'.
    """
    labels = CHOICE_LABELS[task]
    if task == "strategyqa":
        texts = [" YES", " NO"]
    else:
        texts = [f" {c}" for c in labels]
    return labels, texts


# -----------------------------
# Head ablation hooks (pre-hook on attention o_proj)
# -----------------------------
def _get_attn_o_proj(block: torch.nn.Module) -> torch.nn.Module:
    """
    Find the attention output projection module in a transformer block.

    Supports common HF module names:
      - LLaMA/Mistral: block.self_attn.o_proj
      - Qwen2: block.self_attn.o_proj (typical)
      - Some models: block.attn.o_proj / out_proj / dense
    """
    attn = None
    for name in ["self_attn", "attn", "attention", "self_attention", "mha"]:
        if hasattr(block, name):
            attn = getattr(block, name)
            break
    if attn is None:
        raise AttributeError("Cannot find attention module in block; tried self_attn/attn/attention/...")

    for name in ["o_proj", "out_proj", "dense", "proj"]:
        if hasattr(attn, name):
            return getattr(attn, name)
    raise AttributeError("Cannot find attention output projection; tried o_proj/out_proj/dense/proj")


def _get_num_heads_and_head_dim(model: torch.nn.Module, block: Optional[torch.nn.Module] = None) -> Tuple[int, int]:
    """
    Infer (num_heads, head_dim) for reshaping the pre-o_proj input [B,1,H] into [B,1,n_heads,head_dim].
    """
    if block is not None:
        for path in ["self_attn", "attn", "attention", "self_attention", "mha"]:
            if hasattr(block, path):
                attn = getattr(block, path)
                for attr in ["num_heads", "n_heads", "num_attention_heads"]:
                    if hasattr(attn, attr):
                        num_heads = int(getattr(attn, attr))
                        break
                else:
                    num_heads = None
                if num_heads is not None:
                    hidden = None
                    if hasattr(attn, "hidden_size"):
                        hidden = int(getattr(attn, "hidden_size"))
                    elif hasattr(model.config, "hidden_size"):
                        hidden = int(model.config.hidden_size)
                    if hidden is None:
                        break
                    head_dim = hidden // num_heads
                    return num_heads, head_dim

    cfg = getattr(model, "config", None)
    if cfg is None:
        raise ValueError("Model has no config; cannot infer head shapes")
    hidden = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None)
    num_heads = getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head", None)
    if hidden is None or num_heads is None:
        raise ValueError(f"Cannot infer heads from config: hidden={hidden}, num_heads={num_heads}")
    hidden = int(hidden)
    num_heads = int(num_heads)
    if hidden % num_heads != 0:
        raise ValueError(f"hidden_size={hidden} not divisible by num_heads={num_heads}")
    return num_heads, hidden // num_heads


class HeadAblationPreHook:
    """
    Zero out selected head slices in the concatenated attention output
    *before* o_proj, i.e. ablate heads at the "write" interface.

    Designed to be registered via:
        o_proj.register_forward_pre_hook(hook)
    """
    def __init__(
        self,
        head_indices: List[int],
        num_heads: int,
        head_dim: int,
        stats: HookStats,
    ):
        self.head_indices = sorted(set(int(h) for h in head_indices))
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.stats = stats
        self.state: Optional[GenerationState] = None

    def set_state(self, st: Optional[GenerationState]) -> None:
        self.state = st

    def __call__(self, module: torch.nn.Module, inputs: Tuple[torch.Tensor, ...]):
        if not inputs:
            return None
        x = inputs[0]
        if (not torch.is_tensor(x)) or x.ndim != 3:
            return None
        if x.shape[1] != 1:
            return None

        self.stats.decode_calls += 1

        if self.state is not None:
            mask = self.state.current_reasoning_mask()
            if not bool(mask.any().item()):
                return None
        else:
            mask = None

        B, S, H = x.shape
        if H != self.num_heads * self.head_dim:
            if H % self.num_heads != 0:
                return None
            self.head_dim = H // self.num_heads

        x2 = x.clone()
        xv = x2.view(B, S, self.num_heads, self.head_dim)

        if mask is None:
            xv[:, :, self.head_indices, :] = 0
        else:
            mask = mask.to(dtype=torch.bool)
            xv_sel = xv[mask]
            if xv_sel.numel() != 0:
                xv_sel = xv_sel.clone()
                xv_sel[:, :, self.head_indices, :] = 0
                xv[mask] = xv_sel

        self.stats.intervened += 1
        x2 = xv.view(B, S, H)

        if len(inputs) == 1:
            return (x2,)
        return (x2,) + tuple(inputs[1:])


def register_head_ablation_hooks(
    model: torch.nn.Module,
    head_map: Dict[int, List[int]],
    condition: str,  # "baseline" | "full" | "staged"
    reasoning_token_threshold: int,
) -> Tuple[List[Any], Optional[Any], List[HookStats]]:
    """
    Register head-ablation hooks.
    head_map: {layer_idx: [head_idx,...]}.

    condition:
      - baseline: no hooks
      - full: always ablate at decode-time
      - staged: ablate only for examples within GenerationState.current_reasoning_mask()
    """
    assert condition in ["baseline", "full", "staged"]
    if condition == "baseline":
        return [], None, []

    layers, _ = get_model_layers(model)

    handles: List[Any] = []
    staged_hooks: List[HeadAblationPreHook] = []
    stats_list: List[HookStats] = []

    for layer_idx, heads in head_map.items():
        if layer_idx >= len(layers):
            print(f"[Warn] layer_idx={layer_idx} out of range (n_layers={len(layers)}), skipping")
            continue

        block = layers[layer_idx]
        o_proj = _get_attn_o_proj(block)
        n_heads, head_dim = _get_num_heads_and_head_dim(model, block=block)

        stats = HookStats(name=f"head_{condition}@{layer_idx}")
        hk = HeadAblationPreHook(head_indices=heads, num_heads=n_heads, head_dim=head_dim, stats=stats)
        if condition == "staged":
            staged_hooks.append(hk)

        handles.append(o_proj.register_forward_pre_hook(hk))
        stats_list.append(stats)

    def setter(state_or_none: Optional[GenerationState]) -> None:
        for hk in staged_hooks:
            hk.set_state(state_or_none)

    return handles, (setter if condition == "staged" else None), stats_list


# -----------------------------
# Forced-choice: cache-advanced with prompt-boundary alignment
# -----------------------------
def ensure_cache_for_model(past_key_values):
    if past_key_values is None:
        return None
    try:
        from transformers.cache_utils import Cache, DynamicCache  # type: ignore
    except Exception:
        return past_key_values

    try:
        if isinstance(past_key_values, Cache):
            return past_key_values
    except Exception:
        pass

    if isinstance(past_key_values, (tuple, list)) and hasattr(DynamicCache, "from_legacy_cache"):
        try:
            return DynamicCache.from_legacy_cache(past_key_values)
        except Exception:
            return past_key_values

    return past_key_values


def clone_past_key_values(past_key_values):
    if past_key_values is None:
        return None
    try:
        import copy
        return copy.deepcopy(past_key_values)
    except Exception:
        return past_key_values


def decode_one_token_with_past(
    model: torch.nn.Module,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    past_key_values,
):
    past = ensure_cache_for_model(past_key_values)
    try:
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past,
            use_cache=False,
        )
    except Exception:
        past2 = clone_past_key_values(past)
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past2,
            use_cache=True,
        )


@torch.no_grad()
def score_candidates_cache_advanced(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    cand_texts: List[str],
    *,
    device: str,
    max_prompt_len: int,
    reasoning_token_threshold: int,
    state_setter: Optional[Any] = None,
    warmup_text: str = "",
) -> List[float]:
    model.eval()

    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_len,
    ).to(device)

    input_ids: torch.Tensor = enc["input_ids"]   # [1, T]
    attn_mask: torch.Tensor = enc.get("attention_mask", None)
    if attn_mask is None:
        attn_mask = torch.ones_like(input_ids)

    B, T = input_ids.shape
    assert B == 1, "This helper scores one prompt at a time for simplicity"
    eos = tokenizer.eos_token_id

    cand_ids_list: List[List[int]] = [
        tokenizer(c, add_special_tokens=False).input_ids for c in cand_texts
    ]
    all_single_token = all((len(ids) == 1) for ids in cand_ids_list if len(ids) > 0)

    def _compute_base_after_warmup(state: Optional[GenerationState]):
        if T >= 2:
            prefix_ids = input_ids[:, :-1]
            out0 = model(input_ids=prefix_ids, attention_mask=attn_mask[:, :-1], use_cache=True)
            past = ensure_cache_for_model(out0.past_key_values)

            last_id = input_ids[:, -1:]
            out1 = model(input_ids=last_id, attention_mask=attn_mask, past_key_values=past, use_cache=True)
            logits = out1.logits[:, -1, :]  # [1, V]
            past = ensure_cache_for_model(out1.past_key_values)
            cur_attn = attn_mask
        else:
            out1 = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=True)
            logits = out1.logits[:, -1, :]
            past = ensure_cache_for_model(out1.past_key_values)
            cur_attn = attn_mask

        if warmup_text:
            warm_ids = tokenizer(warmup_text, add_special_tokens=False).input_ids
            for tid in warm_ids:
                tid_t = torch.tensor([[tid]], device=device, dtype=input_ids.dtype)

                if state is not None:
                    state.step_update(tid_t, eos_token_id=eos)

                cur_attn = torch.cat(
                    [cur_attn, torch.ones((1, 1), device=cur_attn.device, dtype=cur_attn.dtype)],
                    dim=1,
                )
                outw = model(input_ids=tid_t, attention_mask=cur_attn, past_key_values=past, use_cache=True)
                logits = outw.logits[:, -1, :]
                past = ensure_cache_for_model(outw.past_key_values)

        return logits, past, cur_attn

    # Fast path
    if all_single_token:
        if state_setter is not None:
            state0 = GenerationState(1, input_ids.device, reasoning_token_threshold)
            state_setter(state0)
        else:
            state0 = None
        try:
            logits, _past, _cur_attn = _compute_base_after_warmup(state0)
            logp = torch.log_softmax(logits.float(), dim=-1)[0]  # [V]
            scores: List[float] = []
            for ids in cand_ids_list:
                if len(ids) != 1:
                    scores.append(float("-inf"))
                else:
                    scores.append(float(logp[ids[0]].item()))
            return scores
        finally:
            if state_setter is not None:
                state_setter(None)

    # Slow path
    scores: List[float] = []
    for cand_ids in cand_ids_list:
        if len(cand_ids) == 0:
            scores.append(float("-inf"))
            continue

        if state_setter is not None:
            st = GenerationState(1, input_ids.device, reasoning_token_threshold)
            state_setter(st)
        else:
            st = None

        try:
            logits, past, cur_attn = _compute_base_after_warmup(st)
            score = 0.0

            for j, tid in enumerate(cand_ids):
                logp_t = torch.log_softmax(logits.float(), dim=-1)[0, tid].item()
                score += float(logp_t)

                tid_t = torch.tensor([[tid]], device=device, dtype=input_ids.dtype)
                if st is not None:
                    st.step_update(tid_t, eos_token_id=eos)

                if j == len(cand_ids) - 1:
                    break

                cur_attn = torch.cat(
                    [cur_attn, torch.ones((1, 1), device=cur_attn.device, dtype=cur_attn.dtype)],
                    dim=1,
                )
                outc = model(input_ids=tid_t, attention_mask=cur_attn, past_key_values=past, use_cache=True)
                logits = outc.logits[:, -1, :]
                past = ensure_cache_for_model(outc.past_key_values)

            scores.append(score)
        finally:
            if state_setter is not None:
                state_setter(None)

    return scores


def forced_choice_predict(
    model: torch.nn.Module,
    tokenizer,
    example: Example,
    *,
    device: str,
    max_prompt_len: int,
    reasoning_token_threshold: int,
    state_setter: Optional[Any] = None,
    warmup_text: str = "",
) -> Tuple[str, List[float]]:
    if example.dataset not in CHOICE_LABELS:
        raise ValueError(f"Dataset {example.dataset} has no forced-choice candidate set configured")

    labels, cand_texts = candidate_texts_for_task(example.dataset)
    scores = score_candidates_cache_advanced(
        model, tokenizer, example.prompt, cand_texts,
        device=device,
        max_prompt_len=max_prompt_len,
        reasoning_token_threshold=reasoning_token_threshold,
        state_setter=state_setter,
        warmup_text=warmup_text,
    )
    best = int(np.argmax(np.array(scores, dtype=np.float64)))
    return labels[best], scores


# -----------------------------
# Induction head identification
# -----------------------------
@torch.no_grad()
def induction_scores_via_attn(
    model: torch.nn.Module,
    tokenizer,
    *,
    device: str,
    seq_len: int = 64,
    batch_size: int = 4,
    n_batches: int = 4,
) -> np.ndarray:
    model.eval()
    half = seq_len // 2
    if 2 * half != seq_len:
        raise ValueError("seq_len must be even")

    vocab = int(tokenizer.vocab_size)
    special = set()
    for tid in [tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.bos_token_id]:
        if tid is not None:
            special.add(int(tid))
    pool = [i for i in range(vocab) if i not in special]
    if len(pool) < 1000:
        pool = list(range(vocab))

    all_scores = None
    count = 0

    for _ in range(n_batches):
        base = torch.tensor(
            np.random.choice(pool, size=(batch_size, half), replace=True),
            device=device,
            dtype=torch.long,
        )
        inp = torch.cat([base, base], dim=1)  # [B, seq_len]
        attn_mask = torch.ones_like(inp)

        out = model(
            input_ids=inp,
            attention_mask=attn_mask,
            use_cache=False,
            output_attentions=True,
        )
        attns = out.attentions
        if attns is None or len(attns) == 0 or attns[0] is None:
            raise RuntimeError("Model did not return attention tensors (attentions is None/empty or contains None)")

        L = len(attns)
        H = attns[0].shape[1]
        if all_scores is None:
            all_scores = torch.zeros((L, H), device="cpu", dtype=torch.float64)

        diag_i = torch.arange(half, seq_len, device=device)
        diag_j = diag_i - half

        for l in range(L):
            a = attns[l]  # [B,H,T,T]
            s = a[:, :, diag_i, diag_j].mean(dim=(0, 2))  # [H]
            all_scores[l] += s.detach().double().cpu()
        count += 1

    all_scores = (all_scores / max(count, 1)).numpy()
    return all_scores


@torch.no_grad()
def induction_scores_via_ablation(
    model: torch.nn.Module,
    tokenizer,
    *,
    device: str,
    layer_idx: int,
    seq_len: int = 64,
    batch_size: int = 16,
    n_batches: int = 8,
    seed: int = 0,
) -> np.ndarray:
    assert seq_len % 2 == 0 and seq_len >= 4, "seq_len must be even and >= 4"
    half = seq_len // 2

    layers, _ = get_model_layers(model)
    block = layers[layer_idx]
    o_proj = _get_attn_o_proj(block)
    n_heads, head_dim = _get_num_heads_and_head_dim(model, block=block)

    rng = np.random.default_rng(int(seed))

    vocab = int(tokenizer.vocab_size)
    special = set()
    for tid in [tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.bos_token_id]:
        if tid is not None:
            special.add(int(tid))
    pool = [i for i in range(vocab) if i not in special]
    if len(pool) < 1000:
        pool = list(range(vocab))

    head_scores = np.zeros((n_heads,), dtype=np.float64)
    head_counts = np.zeros((n_heads,), dtype=np.int64)

    model.eval()

    for _ in range(int(n_batches)):
        base_np = rng.choice(pool, size=(int(batch_size), half), replace=True)
        base = torch.tensor(base_np, device=device, dtype=torch.long)  # [B, half]

        prompt = torch.cat([base, base[:, :-1], base[:, :1]], dim=1)  # [B, seq_len]
        prefix = prompt[:, :-1]
        last = prompt[:, -1:]
        target = base[:, 1]

        attn_prefix = torch.ones_like(prefix)
        attn_full = torch.ones_like(prompt)

        out0 = model(input_ids=prefix, attention_mask=attn_prefix, use_cache=True)
        past = ensure_cache_for_model(out0.past_key_values)

        out1 = decode_one_token_with_past(model, input_ids=last, attention_mask=attn_full, past_key_values=past)
        logits_base = out1.logits[:, -1, :]
        logp_base = torch.log_softmax(logits_base.float(), dim=-1)
        nll_base = -logp_base[torch.arange(base.shape[0], device=device), target]

        for h in range(n_heads):
            stats = HookStats(name=f"ih_ablate_l{layer_idx}_h{h}")
            hk = HeadAblationPreHook([h], num_heads=n_heads, head_dim=head_dim, stats=stats)
            handle = o_proj.register_forward_pre_hook(hk)
            try:
                out_h = decode_one_token_with_past(model, input_ids=last, attention_mask=attn_full, past_key_values=past)
                logits_h = out_h.logits[:, -1, :]
                logp_h = torch.log_softmax(logits_h.float(), dim=-1)
                nll_h = -logp_h[torch.arange(base.shape[0], device=device), target]
                head_scores[h] += float((nll_h - nll_base).mean().item())
                head_counts[h] += 1
            finally:
                try:
                    handle.remove()
                except Exception:
                    pass

    head_scores = head_scores / np.maximum(head_counts, 1)
    return head_scores


def pick_top_heads_from_scores(scores: np.ndarray, topk: int, layer_scope: Optional[int] = None) -> Dict[int, List[int]]:
    topk = int(topk)
    if topk <= 0:
        return {}

    if scores.ndim == 1:
        if layer_scope is None:
            raise ValueError("layer_scope must be provided for 1D scores")
        idx = np.argsort(-scores)[:topk]
        return {int(layer_scope): [int(i) for i in idx]}

    L, H = scores.shape
    if layer_scope is not None:
        row = scores[int(layer_scope)]
        idx = np.argsort(-row)[:topk]
        return {int(layer_scope): [int(i) for i in idx]}

    flat = scores.reshape(-1)
    idx = np.argsort(-flat)[:topk]
    head_map: Dict[int, List[int]] = {}
    for k in idx:
        l = int(k // H)
        h = int(k % H)
        head_map.setdefault(l, []).append(h)
    return head_map


# -----------------------------
# TC baseline: task-specific head selection at a layer
# -----------------------------
def _gold_margin_from_scores(dataset: str, scores: List[float], gold: str) -> Optional[float]:
    """
    Compute margin = logp(gold) - logsumexp(logp(others)).
    Returns None if gold cannot be matched.
    """
    if dataset not in CHOICE_LABELS:
        return None
    labels, _ = candidate_texts_for_task(dataset)

    g = str(gold).strip()
    # normalize common cases
    g_up = g.upper()
    if g_up in labels:
        gold_label = g_up
    else:
        # try raw
        gold_label = g
    if gold_label not in labels:
        return None

    gi = labels.index(gold_label)
    gold_score = float(scores[gi])
    others = [float(s) for j, s in enumerate(scores) if j != gi]
    return gold_score - _logsumexp(others)


@torch.no_grad()
def select_task_specific_heads(
    model: torch.nn.Module,
    tokenizer,
    examples: List[Example],
    *,
    device: str,
    layer_idx: int,
    topk: int,
    condition: str,  # "full" or "staged"
    reasoning_token_threshold: int,
    max_prompt_len: int,
    warmup_text: str = "",
    n_search: int = 64,
    score_mode: str = "margin",  # "margin" or "acc"
) -> Dict[int, List[int]]:
    """
    Circuits-like baseline:
      - Find a small set of heads in *one layer* that are most causally important for one anchor task.
      - Score each head by either:
          * acc drop (legacy) OR
          * mean logprob margin drop (default; robust & non-degenerate).
    """
    assert condition in ["full", "staged"]
    assert score_mode in ["margin", "acc"]

    layers, _ = get_model_layers(model)
    block = layers[layer_idx]
    n_heads, _ = _get_num_heads_and_head_dim(model, block=block)

    if n_search is not None and n_search > 0:
        examples = examples[: min(len(examples), int(n_search))]

    if len(examples) == 0:
        print("[TC-search] empty example list; returning empty head_map.")
        return {}

    dataset_name = examples[0].dataset
    if dataset_name not in CHOICE_LABELS:
        print(f"[TC-search] dataset {dataset_name} not in CHOICE_LABELS; returning empty head_map.")
        return {}

    # Baseline stats
    base_correct = []
    base_margins = []

    for ex in tqdm(examples, desc=f"TC-search baseline ({dataset_name})"):
        pred, scores = forced_choice_predict(
            model, tokenizer, ex,
            device=device,
            max_prompt_len=max_prompt_len,
            reasoning_token_threshold=reasoning_token_threshold,
            state_setter=None,
            warmup_text=warmup_text,
        )
        base_correct.append(is_correct(ex.dataset, pred, ex.gold))
        m = _gold_margin_from_scores(ex.dataset, scores, ex.gold)
        if m is not None and math.isfinite(m):
            base_margins.append(float(m))

    if len(base_correct) == 0:
        print("[TC-search] no usable forced-choice examples; returning empty head_map.")
        return {}

    base_acc = float(np.mean(base_correct))
    base_margin = float(np.mean(base_margins)) if base_margins else float("nan")
    print(f"[TC-search] anchor baseline acc={base_acc:.4f} (n={len(base_correct)})")
    if score_mode == "margin":
        print(f"[TC-search] anchor baseline mean_margin={base_margin:.6f} (n_margin={len(base_margins)})")

    drops = np.zeros((n_heads,), dtype=np.float64)

    # Single-head ablations
    for h in range(n_heads):
        head_map = {int(layer_idx): [int(h)]}
        handles, state_setter, _stats = register_head_ablation_hooks(
            model=model,
            head_map=head_map,
            condition=condition,
            reasoning_token_threshold=reasoning_token_threshold,
        )
        try:
            corr_h = []
            margins_h = []
            for ex in examples:
                pred, scores = forced_choice_predict(
                    model, tokenizer, ex,
                    device=device,
                    max_prompt_len=max_prompt_len,
                    reasoning_token_threshold=reasoning_token_threshold,
                    state_setter=state_setter,
                    warmup_text=warmup_text,
                )
                corr_h.append(is_correct(ex.dataset, pred, ex.gold))
                m = _gold_margin_from_scores(ex.dataset, scores, ex.gold)
                if m is not None and math.isfinite(m):
                    margins_h.append(float(m))

            acc_h = float(np.mean(corr_h)) if corr_h else float("nan")
            if score_mode == "acc":
                drops[h] = base_acc - acc_h
            else:
                mean_m_h = float(np.mean(margins_h)) if margins_h else float("nan")
                # If margins are unavailable, fall back to acc drop rather than produce NaN
                if not math.isfinite(base_margin) or not math.isfinite(mean_m_h):
                    drops[h] = base_acc - acc_h
                else:
                    drops[h] = base_margin - mean_m_h
        finally:
            remove_hooks(handles)

    idx = np.argsort(-drops)[: int(topk)]
    print(f"[TC-search] score_mode={score_mode} top heads @layer{layer_idx}: {[(int(i), float(drops[i])) for i in idx]}")
    return {int(layer_idx): [int(i) for i in idx]}


# -----------------------------
# Overlap analysis (Q_shared vs head-write subspace)
# -----------------------------
def head_write_subspace(
    model: torch.nn.Module,
    *,
    layer_idx: int,
    head_indices: List[int],
) -> np.ndarray:
    layers, _ = get_model_layers(model)
    block = layers[layer_idx]
    o_proj = _get_attn_o_proj(block)
    n_heads, head_dim = _get_num_heads_and_head_dim(model, block=block)

    W = o_proj.weight.detach().float().cpu().numpy()  # [out, in]
    D_out, D_in = W.shape
    assert D_out == D_in, "Expected square o_proj weight"
    assert D_in == n_heads * head_dim, f"Expected {n_heads*head_dim} in_features, got {D_in}"

    cols = []
    for h in head_indices:
        s = int(h) * head_dim
        e = s + head_dim
        cols.append(W[:, s:e])
    A = np.concatenate(cols, axis=1) if cols else np.zeros((D_out, 0), dtype=np.float32)
    if A.shape[1] == 0:
        return A
    Q = orthonormalize_np(A)
    return Q


def subspace_overlap_stats(Qa: np.ndarray, Qb: np.ndarray) -> Dict[str, float]:
    if Qa.size == 0 or Qb.size == 0:
        return {"mean_cos2": 0.0, "max_abs": 0.0}
    M = Qa.T @ Qb
    fro_sq = float(np.sum(M * M))
    denom = float(min(Qa.shape[1], Qb.shape[1]))
    mean_cos2 = fro_sq / max(denom, 1.0)
    max_abs = float(np.max(np.abs(M)))
    return {"mean_cos2": mean_cos2, "max_abs": max_abs}


# -----------------------------
# Generic evaluation helpers
# -----------------------------
def fmt_ci(acc: float, lo: float, hi: float) -> str:
    if any(map(lambda x: x is None or (isinstance(x, float) and math.isnan(x)), [acc, lo, hi])):
        return "nan"
    return f"{acc*100:.1f} [{lo*100:.1f},{hi*100:.1f}]"


@torch.no_grad()
def eval_generation(
    model: torch.nn.Module,
    tokenizer,
    examples: List[Example],
    *,
    register_fn,
    decoding: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    device: str,
    batch_size: int,
    max_prompt_len: int,
    reasoning_token_threshold: int,
    bootstrap_iters: int,
    ci_alpha: float,
    global_seed: int,
    sample_seed: Optional[int],
) -> Dict[str, Any]:
    handles, state_setter, hook_stats = register_fn()
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
        for ex, cont in zip(examples, continuations):
            pred = parse_prediction(ex.dataset, cont)
            extracted.append(int(pred != ""))
            correct.append(is_correct(ex.dataset, pred, ex.gold))

        correct_arr = np.array(correct, dtype=np.float32)
        extracted_arr = np.array(extracted, dtype=np.float32)
        seed = stable_int_seed(global_seed, examples[0].dataset if examples else "na", "gen", decoding, sample_seed or 0)
        acc, lo, hi = bootstrap_ci_mean(correct_arr, iters=bootstrap_iters, alpha=ci_alpha, seed=seed)

        return {
            "accuracy": float(acc),
            "ci_low": float(lo),
            "ci_high": float(hi),
            "correct": correct_arr.tolist(),
            "extraction_rate": float(extracted_arr.mean()),
            "eos_rate": float(np.mean(eos_hit)),
            "avg_new_tokens": float(np.mean(new_tok)),
            "hook_stats": [{"name": s.name, "decode_calls": s.decode_calls, "intervened": s.intervened} for s in hook_stats],
        }
    finally:
        remove_hooks(handles)


@torch.no_grad()
def eval_forced_choice(
    model: torch.nn.Module,
    tokenizer,
    examples: List[Example],
    *,
    register_fn,
    device: str,
    max_prompt_len: int,
    reasoning_token_threshold: int,
    warmup_text: str,
    bootstrap_iters: int,
    ci_alpha: float,
    global_seed: int,
) -> Dict[str, Any]:
    handles, state_setter, hook_stats = register_fn()
    try:
        correct = []
        used = 0
        for ex in tqdm(examples, desc=f"ForcedChoice({examples[0].dataset if examples else 'na'})"):
            if ex.dataset not in CHOICE_LABELS:
                continue
            pred, _scores = forced_choice_predict(
                model, tokenizer, ex,
                device=device,
                max_prompt_len=max_prompt_len,
                reasoning_token_threshold=reasoning_token_threshold,
                state_setter=state_setter,
                warmup_text=warmup_text,
            )
            correct.append(is_correct(ex.dataset, pred, ex.gold))
            used += 1

        correct_arr = np.array(correct, dtype=np.float32)
        seed = stable_int_seed(global_seed, examples[0].dataset if examples else "na", "fc")
        acc, lo, hi = bootstrap_ci_mean(correct_arr, iters=bootstrap_iters, alpha=ci_alpha, seed=seed)

        return {
            "n_used": int(used),
            "accuracy": float(acc),
            "ci_low": float(lo),
            "ci_high": float(hi),
            "correct": correct_arr.tolist(),
            "hook_stats": [{"name": s.name, "decode_calls": s.decode_calls, "intervened": s.intervened} for s in hook_stats],
        }
    finally:
        remove_hooks(handles)


# -----------------------------
# Main experiment driver
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--model_dtype", type=str, default="fp32", choices=["fp32", "fp16"])

    ap.add_argument("--mode", type=str, default="all", choices=["all", "loto"],
                    help="all: estimate Q_shared once using all tasks; loto: leave-one-task-out basis estimation.")

    ap.add_argument("--tasks", type=str, default="gsm8k,commonsenseqa,strategyqa,aqua,arc_challenge,openbookqa,qasc,logiqa")
    ap.add_argument("--layer", type=int, default=10)

    ap.add_argument("--n_subspace", type=int, default=128)
    ap.add_argument("--n_eval", type=int, default=256)

    ap.add_argument("--calib_decode_max_new_tokens", type=int, default=128)
    ap.add_argument("--per_task_max_states", type=int, default=20000)

    ap.add_argument("--reasoning_tokens", type=int, default=128)
    ap.add_argument("--max_new_tokens", type=int, default=256)

    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--do_sample", type=int, default=0)
    ap.add_argument("--sample_seed", type=int, default=123)

    ap.add_argument("--template_randomization", type=int, default=1)
    ap.add_argument("--shuffle_choices", type=int, default=1)
    ap.add_argument("--add_answer_prefix", type=int, default=1)
    ap.add_argument("--answer_prefix", type=str, default="\nFinal answer:")

    # Shared-subspace params
    ap.add_argument("--pca_var", type=float, default=0.95)
    ap.add_argument("--min_dim", type=int, default=16)

    # SOLID change: default max_dim should not truncate PCA below pca_var
    ap.add_argument("--max_dim", type=int, default=4096,
                    help="Max PCA rank cap. Set <=0 to auto(use hidden_size). Default 4096 to avoid truncation.")

    ap.add_argument("--tau", type=float, default=1e-3)
    ap.add_argument("--m_shared", type=str, default="all")  # or "half" etc

    # SOLID change: tau scaling for small cross_dim
    ap.add_argument("--tau_mode", type=str, default="auto", choices=["fixed", "auto"])
    ap.add_argument("--tau_ref_dim", type=int, default=3000)

    # SOLID change: enforce A3 balance + task-centering by default
    ap.add_argument("--balance_tasks", type=int, default=1,
                    help="If 1, balance per-task states to n_min per layer before pooled PCA.")
    ap.add_argument("--task_center", type=int, default=1,
                    help="If 1, subtract per-task mean vector before pooled PCA (task-centering).")

    ap.add_argument("--alpha_remove", type=float, default=1.0)

    # Baselines
    ap.add_argument("--ih_topk", type=int, default=4)
    ap.add_argument("--ih_scope", type=str, default="layer", choices=["layer", "all"])
    ap.add_argument("--ih_method", type=str, default="attn", choices=["attn", "ablation"])

    ap.add_argument("--tc_anchor_task", type=str, default="commonsenseqa")
    ap.add_argument("--tc_topk", type=int, default=4)
    ap.add_argument("--tc_search_n", type=int, default=256)

    # SOLID change: TC search score mode
    ap.add_argument("--tc_score", type=str, default="margin", choices=["margin", "acc"])

    # Forced-choice options
    ap.add_argument("--forced_choice_warmup", type=str, default="")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_prompt_len", type=int, default=1024)

    # Stats
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap_iters", type=int, default=1000)
    ap.add_argument("--ci_alpha", type=float, default=0.05)
    ap.add_argument("--perm_iters", type=int, default=2000)

    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    layer_indices = [int(args.layer)]

    # 1) Load model/tokenizer
    model, tok = load_model_and_tokenizer(args.model, device=args.device, model_dtype=args.model_dtype)

    # infer hidden size for max_dim sanity
    hidden = getattr(model.config, "hidden_size", None) or getattr(model.config, "n_embd", None)
    hidden = int(hidden) if hidden is not None else 4096
    max_dim_eff = int(args.max_dim)
    if max_dim_eff <= 0:
        max_dim_eff = hidden
    max_dim_eff = min(max_dim_eff, hidden)

    # 2) Load data
    sub_by, eval_by, meta_by = load_selected_tasks(
        tasks=tasks,
        n_subspace=args.n_subspace,
        n_eval=args.n_eval,
        seed=args.seed,
        template_seed=args.seed + 999,
        template_randomization=bool(args.template_randomization),
        shuffle_choices=bool(args.shuffle_choices),
        add_answer_prefix=bool(args.add_answer_prefix),
        answer_prefix=args.answer_prefix,
    )

    # 3) IH baseline
    ih_head_map: Dict[int, List[int]] = {}
    if args.ih_topk > 0:
        if args.ih_method == "attn":
            try:
                scores = induction_scores_via_attn(model, tok, device=args.device)
            except Exception as e:
                print(f"[IH] attention-based scoring failed: {type(e).__name__}: {e}")
                print("[IH] falling back to ablation-based scoring on the target layer (heuristic)")
                scores_1d = induction_scores_via_ablation(
                    model, tok, device=args.device, layer_idx=args.layer, seed=args.seed + 2026,
                )
                ih_head_map = pick_top_heads_from_scores(scores_1d, topk=args.ih_topk, layer_scope=args.layer)
            else:
                if args.ih_scope == "layer":
                    ih_head_map = pick_top_heads_from_scores(scores, topk=args.ih_topk, layer_scope=args.layer)
                else:
                    ih_head_map = pick_top_heads_from_scores(scores, topk=args.ih_topk, layer_scope=None)
        else:
            scores_1d = induction_scores_via_ablation(
                model, tok, device=args.device, layer_idx=args.layer, seed=args.seed + 2026,
            )
            ih_head_map = pick_top_heads_from_scores(scores_1d, topk=args.ih_topk, layer_scope=args.layer)
    print(f"[IH] selected head_map={ih_head_map}")

    DECODINGS = ["greedy"] + (["sample"] if bool(args.do_sample) else [])
    CONDITIONS = ["baseline", "shared_full", "shared_staged", "ih_staged", "tc_staged"]

    def compute_Q_shared_for_tasks(train_tasks: List[str]) -> np.ndarray:
        """
        SOLID shared basis computation:
          - collect decode-only last-token states
          - cap per-task states
          - balance to n_min per layer
          - task-center
          - pooled PCA to pca_var (up to max_dim_eff)
          - shared set with tau (fixed or auto-scaled by cross_dim)
        """
        prompts_by_task = {k: [ex.prompt for ex in sub_by[k]] for k in train_tasks}

        print("\n" + "=" * 80)
        print("[Subspace-A3-SOLID] Collecting decode-only last-token activations ...")
        print(f"[Subspace-A3-SOLID] calib_decoding=greedy, max_new_tokens={args.calib_decode_max_new_tokens}, "
              f"per_task_max_states={args.per_task_max_states}, balance={bool(args.balance_tasks)}, center={bool(args.task_center)}")
        print("=" * 80)

        raw_task_acts = collect_task_activations_decode_only(
            model=model,
            tokenizer=tok,
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
        )

        proc_task_acts = balance_and_task_center(
            raw_task_acts,
            layer_indices=layer_indices,
            tasks_order=train_tasks,
            do_balance=bool(args.balance_tasks),
            do_center=bool(args.task_center),
            global_seed=args.seed,
        )

        joint_subspace, cross_dim, contributions, full_pca_info = compute_cross_task_subspace(
            proc_task_acts,
            variance_threshold=float(args.pca_var),
            min_dim=int(args.min_dim),
            max_dim=int(max_dim_eff),
            return_full_pca=True,
        )

        # warn if likely truncated
        if int(cross_dim) >= int(max_dim_eff) and int(max_dim_eff) < int(hidden):
            print(f"[Warn] cross_dim hit max_dim cap (cross_dim={cross_dim}, max_dim={max_dim_eff}). "
                  f"pca_var={args.pca_var} may be truncated; consider increasing --max_dim.")

        tau_eff = _effective_tau(float(args.tau), int(cross_dim), mode=str(args.tau_mode), ref_dim=int(args.tau_ref_dim))
        min_tasks = _min_tasks_from_m_shared(args.m_shared, train_tasks)

        print(f"[Basis-SOLID] cross_dim={cross_dim}, pca_var_target={args.pca_var}, max_dim_eff={max_dim_eff}")
        print(f"[Basis-SOLID] m_shared={args.m_shared} -> min_tasks_shared={min_tasks}, tau_base={args.tau} "
              f"(mode={args.tau_mode}, ref_dim={args.tau_ref_dim}) -> tau_eff={tau_eff:.6f}")

        shared_indices = find_fully_shared_basis_improved(
            contributions,
            train_tasks,
            int(cross_dim),
            min_tasks_shared=int(min_tasks),
            relative_threshold=float(tau_eff),
            top_k_components=int(cross_dim),
        )

        if not shared_indices and int(min_tasks) != 2:
            print("[Basis-SOLID] No shared basis for requested m_shared; falling back to min_tasks_shared=2.")
            shared_indices = find_fully_shared_basis_improved(
                contributions,
                train_tasks,
                int(cross_dim),
                min_tasks_shared=2,
                relative_threshold=float(tau_eff),
                top_k_components=int(cross_dim),
            )

        Q = orthonormalize_np(joint_subspace[:, shared_indices]) if shared_indices else np.zeros((hidden, 0), dtype=np.float32)
        print(f"[Basis-SOLID] shared_k={len(shared_indices)} / cross_dim={cross_dim}  (ratio={len(shared_indices)/max(int(cross_dim),1):.4f})")
        return Q

    def choose_tc_anchor(train_tasks: List[str]) -> Optional[str]:
        if args.tc_anchor_task in train_tasks and args.tc_anchor_task in CHOICE_LABELS:
            return args.tc_anchor_task
        for t in train_tasks:
            if t in CHOICE_LABELS:
                return t
        return None

    # 4) folds
    folds: List[Tuple[str, List[str], List[str]]] = []
    if args.mode == "all":
        folds.append(("all", tasks, tasks))
    else:
        for held_out in tasks:
            train = [t for t in tasks if t != held_out]
            folds.append((f"loto_{held_out}", train, [held_out]))

    results: Dict[str, Any] = {
        "model": args.model,
        "layer": args.layer,
        "tasks": tasks,
        "mode": args.mode,
        "folds": {},
    }

    for fold_name, train_tasks, eval_tasks in folds:
        print("\n" + "=" * 100)
        print(f"[Fold] {fold_name}  train={train_tasks}  eval={eval_tasks}")
        print("=" * 100)

        # 4.1) shared basis
        Q_shared = compute_Q_shared_for_tasks(train_tasks)

        # 4.2) TC baseline head selection
        tc_head_map: Dict[int, List[int]] = {}
        if args.tc_topk > 0:
            anchor = choose_tc_anchor(train_tasks)
            if anchor is None:
                print("[TC] no suitable forced-choice anchor in train_tasks; skipping TC baseline.")
            else:
                anchor_exs = [ex for ex in sub_by[anchor] if ex.dataset in CHOICE_LABELS]
                tc_head_map = select_task_specific_heads(
                    model=model,
                    tokenizer=tok,
                    examples=anchor_exs,
                    device=args.device,
                    layer_idx=args.layer,
                    topk=args.tc_topk,
                    condition="staged",
                    reasoning_token_threshold=args.reasoning_tokens,
                    max_prompt_len=args.max_prompt_len,
                    warmup_text=args.forced_choice_warmup,
                    n_search=args.tc_search_n,
                    score_mode=str(args.tc_score),
                )
        print(f"[TC] selected head_map={tc_head_map}")

        # 4.3) Overlap analysis
        try:
            ih_heads_layer = ih_head_map.get(args.layer, [])
            tc_heads_layer = tc_head_map.get(args.layer, [])
            if ih_heads_layer:
                Q_ih_write = head_write_subspace(model, layer_idx=args.layer, head_indices=ih_heads_layer)
                print(f"[Overlap] shared vs IH-write: {subspace_overlap_stats(Q_shared, Q_ih_write)}")
            if tc_heads_layer:
                Q_tc_write = head_write_subspace(model, layer_idx=args.layer, head_indices=tc_heads_layer)
                print(f"[Overlap] shared vs TC-write: {subspace_overlap_stats(Q_shared, Q_tc_write)}")
        except Exception as e:
            print(f"[Overlap] skipped due to error: {type(e).__name__}: {e}")

        fold_block: Dict[str, Any] = {"train_tasks": train_tasks, "eval_tasks": eval_tasks, "by_dataset": {}}

        for task in eval_tasks:
            eval_exs = eval_by[task]
            print("\n" + "-" * 90)
            print(f"[Eval] fold={fold_name} task={task} n={len(eval_exs)}")
            print("-" * 90)

            task_block: Dict[str, Any] = {"n": len(eval_exs), "gen": {}, "forced_choice": {}, "paired": {}}

            for decoding in DECODINGS:
                for cond in CONDITIONS:
                    if cond == "baseline":
                        def _reg():
                            return [], None, []
                    elif cond == "shared_full":
                        def _reg(Q=Q_shared):
                            return register_hooks_for_condition(
                                model=model,
                                layer_indices=layer_indices,
                                Q_np=Q,
                                condition="full",
                                alpha=args.alpha_remove,
                                reasoning_token_threshold=args.reasoning_tokens,
                            )
                    elif cond == "shared_staged":
                        def _reg(Q=Q_shared):
                            return register_hooks_for_condition(
                                model=model,
                                layer_indices=layer_indices,
                                Q_np=Q,
                                condition="staged",
                                alpha=args.alpha_remove,
                                reasoning_token_threshold=args.reasoning_tokens,
                            )
                    elif cond == "ih_staged":
                        hm = ih_head_map
                        def _reg(hm=hm):
                            return register_head_ablation_hooks(
                                model=model,
                                head_map=hm,
                                condition="staged",
                                reasoning_token_threshold=args.reasoning_tokens,
                            )
                    elif cond == "tc_staged":
                        hm = tc_head_map
                        def _reg(hm=hm):
                            return register_head_ablation_hooks(
                                model=model,
                                head_map=hm,
                                condition="staged",
                                reasoning_token_threshold=args.reasoning_tokens,
                            )
                    else:
                        raise ValueError(cond)

                    run_gen = eval_generation(
                        model=model,
                        tokenizer=tok,
                        examples=eval_exs,
                        register_fn=_reg,
                        decoding=decoding,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        device=args.device,
                        batch_size=args.batch_size,
                        max_prompt_len=args.max_prompt_len,
                        reasoning_token_threshold=args.reasoning_tokens,
                        bootstrap_iters=args.bootstrap_iters,
                        ci_alpha=args.ci_alpha,
                        global_seed=args.seed,
                        sample_seed=(args.sample_seed if decoding == "sample" else None),
                    )
                    task_block["gen"][f"{decoding}/{cond}"] = run_gen
                    print(f"[Gen] {decoding}/{cond}: acc={fmt_ci(run_gen['accuracy'], run_gen['ci_low'], run_gen['ci_high'])} "
                          f"extr={run_gen['extraction_rate']*100:.1f}% eos={run_gen['eos_rate']*100:.1f}% newtok={run_gen['avg_new_tokens']:.1f}")

                    if task in CHOICE_LABELS:
                        run_fc = eval_forced_choice(
                            model=model,
                            tokenizer=tok,
                            examples=eval_exs,
                            register_fn=_reg,
                            device=args.device,
                            max_prompt_len=args.max_prompt_len,
                            reasoning_token_threshold=args.reasoning_tokens,
                            warmup_text=args.forced_choice_warmup,
                            bootstrap_iters=args.bootstrap_iters,
                            ci_alpha=args.ci_alpha,
                            global_seed=args.seed,
                        )
                        task_block["forced_choice"][f"{cond}"] = run_fc
                        print(f"[FC ] {cond}: acc={fmt_ci(run_fc['accuracy'], run_fc['ci_low'], run_fc['ci_high'])} n_used={run_fc['n_used']}")

            # Paired tests on forced-choice
            if task in CHOICE_LABELS and "baseline" in task_block["forced_choice"] and "shared_full" in task_block["forced_choice"]:
                base = np.array(task_block["forced_choice"]["baseline"]["correct"], dtype=np.float32)
                shared = np.array(task_block["forced_choice"]["shared_full"]["correct"], dtype=np.float32)
                ih = np.array(task_block["forced_choice"]["ih_staged"]["correct"], dtype=np.float32) if "ih_staged" in task_block["forced_choice"] else None
                tc = np.array(task_block["forced_choice"]["tc_staged"]["correct"], dtype=np.float32) if "tc_staged" in task_block["forced_choice"] else None
                seed0 = stable_int_seed(args.seed, fold_name, task, "paired_fc")

                task_block["paired"]["shared_full_vs_baseline_fc"] = summarize_paired(
                    base, shared,
                    label=f"{fold_name}:{task}:shared_full_vs_baseline_fc",
                    bootstrap_iters=args.bootstrap_iters,
                    perm_iters=args.perm_iters,
                    alpha=args.ci_alpha,
                    seed=seed0 + 1,
                )
                if ih is not None and len(ih) == len(shared):
                    task_block["paired"]["shared_full_vs_ih_fc"] = summarize_paired(
                        ih, shared,
                        label=f"{fold_name}:{task}:shared_full_vs_ih_fc",
                        bootstrap_iters=args.bootstrap_iters,
                        perm_iters=args.perm_iters,
                        alpha=args.ci_alpha,
                        seed=seed0 + 2,
                    )
                if tc is not None and len(tc) == len(shared):
                    task_block["paired"]["shared_full_vs_tc_fc"] = summarize_paired(
                        tc, shared,
                        label=f"{fold_name}:{task}:shared_full_vs_tc_fc",
                        bootstrap_iters=args.bootstrap_iters,
                        perm_iters=args.perm_iters,
                        alpha=args.ci_alpha,
                        seed=seed0 + 3,
                    )

            fold_block["by_dataset"][task] = task_block

        results["folds"][fold_name] = fold_block

    out_path = f"unique_IH_TC_{args.model.replace('/', '_')}_layer{args.layer}_mode{args.mode}_seed{args.seed}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Done] wrote: {out_path}")


if __name__ == "__main__":
    main()
