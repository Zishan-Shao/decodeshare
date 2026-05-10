# -*- coding: utf-8 -*-
"""
disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py

Decode-last-token (decode-only) shared-subspace removal with:
  (1) Energy-balance sanity checks (shared vs random bases).
  (2) Template randomization (prompt wording + optional choice shuffling).
  (3) Leave-One-Task-Out (LOTO) basis estimation across 8 CoT-style HF datasets.

Default tasks (8):
  - gsm8k (gsm8k/main)                     [numeric]
  - commonsenseqa (commonsense_qa)         [MC A-E]
  - strategyqa (ChilleD/StrategyQA)        [Yes/No]
  - aqua (aqua_rat)                        [MC A-E]
  - arc_challenge (ai2_arc/ARC-Challenge)  [MC A-D]
  - openbookqa (openbookqa/main)           [MC A-D]
  - qasc (qasc)                            [MC A-H]
  - logiqa (logiqa)                        [MC A-D]

What LOTO means here:
  - For each held-out task t, estimate the shared subspace Q_shared using the other (N-1) tasks only.
  - Then evaluate interventions on the held-out task (and optionally all tasks).
  - This avoids leakage of the evaluated task into subspace estimation.

Key features inherited from your energy_balance script:
  - A3 decode-aligned basis: collect decode-phase last-token states (seq_len==1) using KV-cache decode calls.
  - pooled PCA -> sharedness -> shared basis
  - intervention: last-token removal only (no rotation), alpha=1.0 default
  - staged gating by generated token count
  - evaluation: EM accuracy + bootstrap CI + paired sign-flip permutation test (paired by example)
  - greedy main + sampling robustness (optional)
  - SANITY:
      * orthonormality of shared/rand bases
      * overlap between shared and rand
      * energy ratio on calibration decode states (shared vs rand)
      * hook stats (decode_calls / intervened)
      * extraction rate, eos rate, avg_new_tokens

Requirements:
  pip install transformers datasets numpy torch tqdm

Also requires your project utilities:
  from joint_subspace_large.disturb_cross_task_all_shared import (
      get_model_layers, compute_cross_task_subspace, find_fully_shared_basis_improved
  )


[Data] task=gsm8k loaded_prompts=128
[Data] task=commonsenseqa loaded_prompts=128
[Data] task=strategyqa loaded_prompts=128
[Data] task=aqua loaded_prompts=128
[Data] task=openbookqa loaded_prompts=128
[Data] task=qasc loaded_prompts=128
[Data] task=boolq loaded_prompts=128
[Data] task=piqa loaded_prompts=128

gsm8k,commonsenseqa,strategyqa,aqua,piqa,arc_challenge,openbookqa,qasc,logiqa,boolq

Example usage:

  # Single (all-tasks) basis estimation + evaluation:
  CUDA_VISIBLE_DEVICES=1 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
    --model meta-llama/Llama-2-7b-chat-hf --device cuda --model_dtype fp32 \
    --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
    --mode all \
    --n_subspace 128 --n_eval 256 --layer 10 \
    --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
    --reasoning_tokens 128 --max_new_tokens 256 \
    --template_randomization 1 --shuffle_choices 1 \
    --do_sample 0 --out_json results/disturb_cot/all_tasks_energy_balance_results_llama2-7b-chat-hf_aligned.json --out_md results/disturb_cot/all_tasks_energy_balance_summary_llama2-7b-chat-hf_aligned.md

  # LOTO (estimate basis on N-1 tasks, evaluate held-out only):
  CUDA_VISIBLE_DEVICES=1 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
    --model meta-llama/Llama-2-7b-chat-hf --device cuda --model_dtype fp32 \
    --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
    --mode loto \
    --loto_eval_mode heldout \
    --n_subspace 128 --n_eval 256 --layer 10 \
    --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
    --reasoning_tokens 128 --max_new_tokens 256 \
    --template_randomization 1 --shuffle_choices 1 \
    --do_sample 0 --out_json results/disturb_cot/energy_balance_loto8_results_llama2-7b-chat-hf_aligned.json --out_md results/disturb_cot/energy_balance_loto8_summary_llama2-7b-chat-hf_aligned.md

Notes:
  - qasc/logiqa schemas can vary slightly across HF versions; this script includes schema inference + fallbacks.
  - If a dataset fails to load, you can remove it from --tasks.

# A) LOTO（只评估 heldout），只跑 greedy（最省时间，最适合先出结论）
CUDA_VISIBLE_DEVICES=1 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
  --model meta-llama/Llama-2-7b-chat-hf --device cuda --model_dtype fp32 \
  --mode loto --loto_eval_mode heldout \
  --tasks gsm8k,commonsenseqa,strategyqa,aqua,arc_challenge,openbookqa,qasc,logiqa \
  --layer 10 --n_subspace 128 --n_eval 256 \
  --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
  --reasoning_tokens 128 --max_new_tokens 256 \
  --template_randomization 1 --shuffle_choices 1 \
  --add_answer_prefix 1 --answer_prefix $'\nFinal answer:' \
  --do_sample 0 --out_json energy_balance_loto8_results_llama2-7b-chat-hf.json --out_md energy_balance_loto8_summary_llama2-7b-chat-hf.md

# B) 只跑某一个 holdout（debug / 快速迭代）
CUDA_VISIBLE_DEVICES=1 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
  --model meta-llama/Llama-2-7b-chat-hf --device cuda --model_dtype fp32 \
  --mode loto --loto_eval_mode heldout --loto_only commonsenseqa \
  --layer 10 --n_subspace 128 --n_eval 256 \
  --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
  --reasoning_tokens 128 --max_new_tokens 256 \
  --template_randomization 1 --shuffle_choices 1 \
  --add_answer_prefix 1 --answer_prefix 0 \
  --do_sample 0

# C) 非 LOTO（all tasks 一把估 basis）
CUDA_VISIBLE_DEVICES=0 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
  --model meta-llama/Llama-2-7b-chat-hf --device cuda --model_dtype fp32 \
  --mode all \
  --tasks gsm8k,commonsenseqa,strategyqa,aqua,arc_challenge,openbookqa,qasc,logiqa \
  --layer 10 --n_subspace 128 --n_eval 256 \
  --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
  --reasoning_tokens 128 --max_new_tokens 256 \
  --template_randomization 1 --shuffle_choices 1 \
  --add_answer_prefix 1 --answer_prefix $'\nFinal answer:' \
  --do_sample 0

"""

import os
import sys
import re
import json
import math
import random
import argparse
import hashlib
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
# from benchmark_dataloaders import (  # noqa: E402
#     Example,
#     load_selected_tasks,
#     parse_prediction,
#     is_correct as is_correct_bool,
#     stable_int_seed as stable_int_seed_bdl,
# )
from benchmark_dataloaders import *
from benchmark_dataloaders import (
    stable_int_seed as stable_int_seed_bdl,
    is_correct as is_correct_bool,
)

# -----------------------------
# Repro / utils
# -----------------------------
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Use stable_int_seed from benchmark_dataloaders
stable_int_seed = stable_int_seed_bdl

def json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return str(o)

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

# Wrapper to convert is_correct bool to int for compatibility
def is_correct(dataset: str, pred: str, gold: str) -> int:
    return int(is_correct_bool(dataset, pred, gold))

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

# -----------------------------
# Decode last-token activation collector (A3)
# -----------------------------
class DecodeLastTokenActivationCollector:
    """
    Collect last-token hidden states ONLY during decode forward passes (seq_len==1).
    storage[task][layer_idx] -> list of chunks [B', D]
    """
    def __init__(self, layer_indices: List[int], acts_resident: str = "cpu"):
        self.layer_indices = list(layer_indices)
        self.acts_resident = str(acts_resident)
        self._cur_task: Optional[str] = None
        self.capture_enabled: bool = False
        self.active_mask: Optional[torch.Tensor] = None
        self.storage: DefaultDict[str, DefaultDict[int, List[Any]]] = defaultdict(lambda: defaultdict(list))

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
                if torch.is_tensor(m) and m.device != x.device:
                    m = m.to(device=x.device, non_blocking=True)
                if m.numel() == x.shape[0]:
                    x = x[m]
            if x.numel() == 0:
                return output
            if self.acts_resident == "gpu":
                self.storage[self._cur_task][layer_idx].append(x.detach().to(dtype=torch.float16))
            else:
                self.storage[self._cur_task][layer_idx].append(x.detach().float().cpu().numpy())
            return output
        return _hook

    def get_task_activations(self, task: str, layer_idx: int, *, clear: bool = False) -> Optional[np.ndarray]:
        chunks = self.storage.get(task, {}).get(layer_idx, [])
        if not chunks:
            return None
        first = chunks[0]
        if torch.is_tensor(first):
            x = torch.cat(chunks, dim=0).float().cpu().numpy()
        else:
            x = np.concatenate(chunks, axis=0)
        if clear:
            try:
                del self.storage[task][layer_idx]
                if not self.storage[task]:
                    del self.storage[task]
            except Exception:
                pass
        return x.astype(np.float32, copy=False)

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

    for i in tqdm(range(0, len(prompts), batch_size), desc="CollectDecode"):
        batch = prompts[i:i+batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_len,
        ).to(device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        B, _T0 = input_ids.shape

        unfinished = torch.ones(B, dtype=torch.bool, device=device)

        # Prefill (no capture)
        collector.set_capture(False, None)
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        logits = out.logits[:, -1, :]
        past = out.past_key_values

        for _ in range(max_new_tokens):
            if decoding == "greedy":
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                lt = logits / max(temperature, 1e-6)
                lt = top_k_filtering(lt, top_k=top_k)
                lt = top_p_filtering(lt, top_p=top_p)
                probs = torch.softmax(lt, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

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
    collector = DecodeLastTokenActivationCollector(layer_indices, acts_resident="gpu")
    task_activations: Dict[str, Dict[int, np.ndarray]] = {}

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
            layer_dict = {}
            for layer_idx in layer_indices:
                acts = collector.get_task_activations(task_name, layer_idx, clear=True)
                if acts is None or acts.shape[0] == 0:
                    continue
                ss = stable_int_seed(global_seed, task_name, layer_idx, "subsample")
                acts = _subsample_rows_np(acts, per_task_max_states, seed=ss)
                layer_dict[layer_idx] = acts
                print(f"[Subspace-A3]  collected {task_name} layer={layer_idx}: {acts.shape[0]} x {acts.shape[1]}")
            if layer_dict:
                task_activations[task_name] = layer_dict
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass
        collector.set_capture(False, None)

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
# Generation + per-example stats
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
        batch = prompts[i:i+batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_len,
        ).to(device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        B, T0 = input_ids.shape

        state = GenerationState(B, input_ids.device, reasoning_token_threshold)
        if state_setter is not None:
            state_setter(state)

        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        logits = out.logits[:, -1, :]
        past = out.past_key_values

        generated = input_ids

        for _ in range(max_new_tokens):
            if decoding == "greedy":
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                lt = logits / max(temperature, 1e-6)
                lt = top_k_filtering(lt, top_k=top_k)
                lt = top_p_filtering(lt, top_p=top_p)
                probs = torch.softmax(lt, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

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
# Evaluate
# -----------------------------
def evaluate_condition(
    model,
    tokenizer,
    examples: List[Example],
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
        for ex, cont in zip(examples, continuations):
            pred = parse_prediction(ex.dataset, cont)
            extracted.append(int(pred != ""))
            correct.append(is_correct(ex.dataset, pred, ex.gold))

        correct_arr = np.array(correct, dtype=np.float32)
        extracted_arr = np.array(extracted, dtype=np.float32)

        seed = stable_int_seed(global_seed, examples[0].dataset if examples else "na", condition, decoding, sample_seed or 0)
        acc, lo, hi = bootstrap_ci_mean(correct_arr, iters=bootstrap_iters, alpha=ci_alpha, seed=seed)

        return {
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
    finally:
        remove_hooks(handles)

# -----------------------------
# Model loading
# -----------------------------
def _parse_max_memory_map(spec: str):
    spec = str(spec or "").strip()
    if not spec:
        return None
    out = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid max_memory_map item: {item!r}")
        key, val = item.split(":", 1)
        key = key.strip()
        val = val.strip()
        if not key or not val:
            raise ValueError(f"Invalid max_memory_map item: {item!r}")
        key_norm = key.lower()
        if key_norm == "cpu":
            out["cpu"] = f"{int(float(val))}GiB"
        else:
            out[int(key)] = f"{int(float(val))}GiB"
    return out or None


def load_model_and_tokenizer(
    model_name: str,
    device: str,
    model_dtype: str,
    device_map: Optional[str] = None,
    max_memory_per_gpu_gb: float = 0.0,
    cpu_offload_gb: float = 0.0,
    max_memory_map: str = "",
):
    dtype = torch.float32 if model_dtype == "fp32" else torch.float16
    model_kwargs: Dict[str, Any] = {}
    device_map = (str(device_map).strip() or None)
    if device_map is not None:
        model_kwargs["device_map"] = device_map
        model_kwargs["low_cpu_mem_usage"] = True
        parsed_max_mem = _parse_max_memory_map(max_memory_map)
        if parsed_max_mem:
            if not torch.cuda.is_available():
                raise RuntimeError("device_map was requested but CUDA is unavailable.")
            model_kwargs["max_memory"] = parsed_max_mem
        elif float(max_memory_per_gpu_gb) > 0.0:
            if not torch.cuda.is_available():
                raise RuntimeError("device_map was requested but CUDA is unavailable.")
            max_mem = {i: f"{int(float(max_memory_per_gpu_gb))}GiB" for i in range(torch.cuda.device_count())}
            if float(cpu_offload_gb) > 0.0:
                max_mem["cpu"] = f"{int(float(cpu_offload_gb))}GiB"
            model_kwargs["max_memory"] = max_mem
        elif float(cpu_offload_gb) > 0.0:
            model_kwargs["max_memory"] = {"cpu": f"{int(float(cpu_offload_gb))}GiB"}
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, **model_kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, **model_kwargs)

    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if device_map is None:
        model = model.to(device)
    model.eval()
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
    sub_by: Dict[str, List[Example]],
    eval_by: Dict[str, List[Example]],
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
    DECODINGS = ["greedy"] + (["sample"] if args.do_sample else [])
    CONDITIONS = ["baseline", "shared_full", "shared_staged", "rand_full", "rand_staged"]

    by_dataset: Dict[str, Any] = {}
    for task_name in eval_tasks:
        eval_exs = eval_by[task_name]
        print("\n" + "-" * 80)
        print(f"[{fold_name}][Eval] Dataset={task_name} (n={len(eval_exs)})")
        print("-" * 80)

        block: Dict[str, Any] = {"n": len(eval_exs), "runs": {}, "paired_tests": {}}

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

                run = evaluate_condition(
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
                )
                block["runs"][f"{decoding}/{mode}"] = run
                print(f"[{fold_name}][{task_name}] {decoding}/{mode}: acc={fmt_acc(run['accuracy'], run['ci_low'], run['ci_high'])} "
                      f"extr={run['extraction_rate']*100:.1f}% eos={run['eos_rate']*100:.1f}% avg_new_tok={run['avg_new_tokens']:.1f}")

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
        "extra": extra,
    }

def render_loto_heldout_table(results: Dict[str, Any], decoding: str = "greedy") -> str:
    """
    Simple markdown table: each row is a held-out task, showing baseline vs shared_full on held-out only.
    """
    rows = []
    header = ["Held-out", "n", "Baseline", "Shared(full)", "Rand(full)", "Δ(shared-baseline)", "p(shared-baseline)"]
    for holdout, fold in results.get("folds", {}).items():
        block = fold["by_dataset"].get(holdout, None)
        if block is None:
            continue
        b = block["runs"][f"{decoding}/baseline"]
        s = block["runs"][f"{decoding}/shared_full"]
        r = block["runs"][f"{decoding}/rand_full"]
        stat = block["paired_tests"][decoding]["shared_full_vs_baseline"]
        rows.append([
            holdout,
            str(block["n"]),
            fmt_acc(b["accuracy"], b["ci_low"], b["ci_high"]),
            fmt_acc(s["accuracy"], s["ci_low"], s["ci_high"]),
            fmt_acc(r["accuracy"], r["ci_low"], r["ci_high"]),
            f"{stat['mean_diff']*100:+.1f} [{stat['ci_low']*100:+.1f}, {stat['ci_high']*100:+.1f}]",
            f"{stat['p_value']:.3g}",
        ])

    # widths
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

    # 1) 常见字段（大多数纯文本LM）
    for k in ("hidden_size", "n_embd", "dim", "d_model", "model_dim", "embed_dim"):
        v = getattr(cfg, k, None)
        if isinstance(v, int) and v > 0:
            return v

    # 2) Gemma3 / 多模态：hidden_size 在 text_config 里
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is not None:
        for k in ("hidden_size", "n_embd", "dim", "d_model", "model_dim", "embed_dim"):
            v = getattr(text_cfg, k, None)
            if isinstance(v, int) and v > 0:
                return v

    # 3) 最终兜底：直接从 input embedding 的 weight 维度读
    try:
        emb = model.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight") and isinstance(emb.weight, torch.Tensor) and emb.weight.ndim == 2:
            return int(emb.weight.shape[1])
        if emb is not None and hasattr(emb, "embedding_dim"):
            return int(emb.embedding_dim)
    except Exception:
        pass

    return None


# -----------------------------
# Main
# -----------------------------
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

    # Decoding
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

    # Stats
    ap.add_argument("--bootstrap_iters", type=int, default=5000)
    ap.add_argument("--perm_iters", type=int, default=10000)
    ap.add_argument("--ci_alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample_seed", type=int, default=12345)

    # Output
    ap.add_argument("--out_json", type=str, default=os.path.join(THIS_DIR, "energy_balance_loto8_results.json"))
    ap.add_argument("--out_md", type=str, default=os.path.join(THIS_DIR, "energy_balance_loto8_summary.md"))

    args = ap.parse_args()
    set_global_seed(args.seed)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if len(tasks) < 2:
        raise RuntimeError("Need at least 2 tasks in --tasks.")

    args.do_sample = bool(args.do_sample)
    args.template_randomization = bool(args.template_randomization)
    args.shuffle_choices = bool(args.shuffle_choices)
    args.add_answer_prefix = bool(args.add_answer_prefix)

    layer_indices = [args.layer]

    print(f"[Env] DEVICE={args.device}")
    print(f"[Env] MODEL={args.model} dtype={args.model_dtype}")
    print(f"[Env] layer_indices={layer_indices}")
    print(f"[Env] tasks={tasks}")
    print(f"[Env] mode={args.mode} loto_eval_mode={args.loto_eval_mode}")
    print(f"[Env] template_randomization={args.template_randomization} shuffle_choices={args.shuffle_choices} add_answer_prefix={args.add_answer_prefix}")

    model, tokenizer = load_model_and_tokenizer(args.model, args.device, args.model_dtype)



    # hidden_dim = getattr(model.config, "hidden_size", None) or getattr(model.config, "n_embd", None)
    # if hidden_dim is None:
    #     raise RuntimeError("Could not infer hidden_dim from model.config")
    # print(f"[Env] hidden_dim={hidden_dim}")

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

    # Save a small markdown summary (especially useful for LOTO heldout-mode)
    md_lines = []
    md_lines.append("# Energy-balance + LOTO(8) Summary\n")
    md_lines.append(f"- Model: `{args.model}` dtype={args.model_dtype} device={args.device}\n")
    md_lines.append(f"- Tasks: {tasks}\n")
    md_lines.append(f"- Mode: {args.mode}\n")
    md_lines.append(f"- Template randomization: {args.template_randomization} (seed={args.template_seed}), shuffle_choices={args.shuffle_choices}\n")
    md_lines.append(f"- Sharedness: pca_var={args.pca_var}, tau={args.tau}, m_shared={args.m_shared}\n")
    md_lines.append(f"- Calibration decode max_new_tokens={args.calib_decode_max_new_tokens}, per_task_max_states={args.per_task_max_states}\n")
    md_lines.append("")

    if args.mode == "loto" and args.loto_eval_mode == "heldout" and "folds" in results:
        md_lines.append("## LOTO held-out performance (greedy)\n")
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
