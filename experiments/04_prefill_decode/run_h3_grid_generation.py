# -*- coding: utf-8 -*-
"""
h3_killer_counterfactual_generation.py

Generation-only version of the H3 “killer counterfactual” 2×2 grid experiment.

What this script does (end-to-end):
  1) Estimate two shared bases at the same layer ℓ:
       Q_S_dec  from decode distribution  (KV-cached, seq_len==1 calls during rollouts)
       Q_S_pre  from prefill distribution (single full prompt forward, seq_len>1)
     using the same sharedness criterion (per-task relative variance usage + τ, m_shared).

  2) Dimension-match: k = min(k_dec, k_pre) and truncate both shared bases to k.

  3) Energy-match at the *decode locus* via α-scaling (optional):
       alpha(Q) = sqrt(E_ref / E_Q)
     where E_Q = mean ||Q^T h||^2 on a decode-calibration state distribution.

  4) Run generation evaluation (NO forced-choice anywhere):
       - baseline (no hook)
       - decode-intervene: decode-est / decode-int, prefill-est / decode-int, control(decode) / decode-int
       - prefill-intervene: decode-est / prefill-int, prefill-est / prefill-int, control(prefill) / prefill-int
     Metrics use your repo’s dataloader parse_prediction + is_correct (paper-consistent).

Dependencies:
  pip install torch transformers datasets numpy tqdm

Expected repo context:
  This script imports:
    - benchmark_dataloaders_aqua_prefix_default.load_selected_tasks
  If your repo uses a different module name, edit `import_dataloaders()`.

Example:
  CUDA_VISIBLE_DEVICES=0 python h3_killer_counterfactual_grid_generate.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --tasks gsm8k,commonsenseqa,strategyqa,piqa,arc_challenge,openbookqa,qasc,logiqa,boolq \
    --layer 10 --n_subspace 128 --n_eval 256 \
    --calib_decode_max_new_tokens 512 --per_task_max_states 20000 \
    --answer_prefix $'\nFinal answer:' \
    --template_randomization 1 --shuffle_choices 1 \
    --gen_max_new_tokens 256 --gen_decoding greedy
"""

from __future__ import annotations

import os
import json
import math
import argparse
import random
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# -----------------------------------------------------------------------------
# Repro helpers
# -----------------------------------------------------------------------------

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_int_seed(*items: Any) -> int:
    s = "|".join(map(str, items)).encode("utf-8")
    h = hashlib.md5(s).hexdigest()
    return int(h[:8], 16)


def json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return str(o)


# -----------------------------------------------------------------------------
# Model layer discovery (works for Llama/Qwen/Gemma and many HF decoders)
# -----------------------------------------------------------------------------

def get_transformer_blocks(model) -> List[torch.nn.Module]:
    """Return a list of transformer block modules to hook."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return list(model.model.decoder.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return list(model.gpt_neox.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "blocks"):
        return list(model.transformer.blocks)
    raise ValueError(
        "Cannot locate transformer blocks for hooking. "
        "Please extend get_transformer_blocks() for your model class."
    )


# -----------------------------------------------------------------------------
# Basis math
# -----------------------------------------------------------------------------

def orthonormalize_np(basis: np.ndarray) -> np.ndarray:
    q, _ = np.linalg.qr(basis.astype(np.float32, copy=False))
    return q.astype(np.float32, copy=False)


def principal_angles_deg(Qa: np.ndarray, Qb: np.ndarray) -> np.ndarray:
    """Return principal angles (degrees) between two k-dim subspaces."""
    if Qa.size == 0 or Qb.size == 0:
        return np.array([], dtype=np.float64)
    Qa = orthonormalize_np(Qa)
    Qb = orthonormalize_np(Qb)
    M = Qa.T @ Qb
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    ang = np.degrees(np.arccos(s))
    return ang


def energy_stats(states: np.ndarray, Q: np.ndarray) -> Dict[str, float]:
    """Report mean/median/p95 energy ratio r(h;Q) = ||Q^T h||^2 / ||h||^2."""
    eps = 1e-12
    H = states.astype(np.float32, copy=False)
    Q = orthonormalize_np(Q)
    proj = H @ Q
    num = np.sum(proj * proj, axis=1)
    den = np.sum(H * H, axis=1) + eps
    r = num / den
    return {
        "mean": float(np.mean(r)),
        "p50": float(np.percentile(r, 50)),
        "p95": float(np.percentile(r, 95)),
    }


# -----------------------------------------------------------------------------
# Shared subspace estimator (self-contained version of Def.4)
# -----------------------------------------------------------------------------

@dataclass
class SharedSubspaceResult:
    Q_joint: np.ndarray                # [D, k_pca]
    eigvals: np.ndarray                # [k_pca]
    shared_indices: List[int]          # indices in [0, k_pca)
    per_task_rel: Dict[str, np.ndarray]  # task -> [k_pca]


def _balance_and_center(task_states: Dict[str, np.ndarray], seed: int) -> Tuple[Dict[str, np.ndarray], int]:
    """Task-center each X_t and subsample to the same row count."""
    rng = np.random.default_rng(seed)
    centered: Dict[str, np.ndarray] = {}
    n_min = None
    for t, X in task_states.items():
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[0] == 0:
            continue
        mu = X.mean(axis=0, keepdims=True)
        Xc = X - mu
        centered[t] = Xc
        n_min = Xc.shape[0] if n_min is None else min(n_min, Xc.shape[0])
    if not centered:
        raise RuntimeError("No valid task states provided.")
    assert n_min is not None
    balanced: Dict[str, np.ndarray] = {}
    for t, Xc in centered.items():
        if Xc.shape[0] == n_min:
            balanced[t] = Xc
        else:
            idx = rng.choice(Xc.shape[0], size=n_min, replace=False)
            balanced[t] = Xc[idx]
    return balanced, int(n_min)


def _pca_from_cov(X: np.ndarray, device: str = "cpu") -> Tuple[np.ndarray, np.ndarray]:
    """PCA via covariance eigen-decomposition.

    Returns:
      eigvals_desc: [D]
      eigvecs_desc: [D,D]
    """
    Xt = torch.from_numpy(X).to(device=device, dtype=torch.float32)
    n = Xt.shape[0]
    C = (Xt.T @ Xt) / max(n, 1)
    w, V = torch.linalg.eigh(C)  # ascending
    w = w.flip(0)
    V = V.flip(1)
    return w.detach().cpu().numpy(), V.detach().cpu().numpy()


def estimate_shared_subspace(
    task_states: Dict[str, np.ndarray],
    *,
    rho: float,
    tau: float,
    m_shared: str,
    pca_device: str,
    seed: int,
) -> SharedSubspaceResult:
    """Estimate pooled PCA basis + shared component set."""
    balanced, _ = _balance_and_center(task_states, seed=seed)
    tasks = sorted(balanced.keys())

    X_pool = np.concatenate([balanced[t] for t in tasks], axis=0)
    eigvals, eigvecs = _pca_from_cov(X_pool, device=pca_device)

    total = float(np.sum(np.maximum(eigvals, 0.0)))
    if total <= 0:
        raise RuntimeError("Non-positive total variance in PCA.")

    csum = np.cumsum(np.maximum(eigvals, 0.0)) / total
    k_pca = int(np.searchsorted(csum, rho, side="left") + 1)
    k_pca = max(1, min(k_pca, eigvecs.shape[1]))

    Q_joint = eigvecs[:, :k_pca].astype(np.float32, copy=False)
    eigvals_k = eigvals[:k_pca].astype(np.float32, copy=False)

    per_task_rel: Dict[str, np.ndarray] = {}
    for t in tasks:
        Z = balanced[t] @ Q_joint
        v = np.var(Z, axis=0, ddof=0).astype(np.float64)
        Vt = float(np.sum(v)) + 1e-12
        per_task_rel[t] = (v / Vt).astype(np.float32)

    if m_shared == "all":
        m_req = len(tasks)
    else:
        try:
            m_req = max(2, int(m_shared))
        except Exception:
            m_req = len(tasks)

    shared_idx: List[int] = []
    for i in range(k_pca):
        ct = sum(1 for t in tasks if float(per_task_rel[t][i]) >= tau)
        if ct >= m_req:
            shared_idx.append(i)

    return SharedSubspaceResult(
        Q_joint=orthonormalize_np(Q_joint),
        eigvals=eigvals_k,
        shared_indices=shared_idx,
        per_task_rel=per_task_rel,
    )


def select_nonshared_control(
    *,
    Q_joint: np.ndarray,
    eigvals: np.ndarray,
    shared_idx: List[int],
    k: int,
    method: str,
    seed: int,
) -> np.ndarray:
    """Pick a non-shared control subspace from the same joint PCA basis."""
    rng = np.random.default_rng(seed)
    shared_set = set(shared_idx)
    nonshared = [i for i in range(Q_joint.shape[1]) if i not in shared_set]
    if len(nonshared) < k:
        raise RuntimeError(f"Not enough nonshared components: {len(nonshared)} < k={k}")

    eigvals = np.asarray(eigvals, dtype=np.float64)

    if method == "uniform":
        idx = list(rng.choice(nonshared, size=k, replace=False))
        return orthonormalize_np(Q_joint[:, idx])

    if method == "topk":
        nonshared_sorted = sorted(nonshared, key=lambda i: eigvals[i], reverse=True)
        idx = nonshared_sorted[:k]
        return orthonormalize_np(Q_joint[:, idx])

    if method == "varmatch":
        # Greedy match shared eigenvalues to nearest nonshared eigenvalues
        shared_sorted = sorted(shared_idx, key=lambda i: eigvals[i])
        nonshared_sorted = sorted(nonshared, key=lambda i: eigvals[i])
        nonshared_vals = [eigvals[i] for i in nonshared_sorted]
        import bisect

        chosen: List[int] = []
        for si in shared_sorted:
            target = eigvals[si]
            j = bisect.bisect_left(nonshared_vals, target)
            cand_pos = [p for p in (j - 1, j) if 0 <= p < len(nonshared_sorted)]
            if not cand_pos:
                break
            best_p = min(cand_pos, key=lambda p: abs(nonshared_vals[p] - target))
            chosen.append(nonshared_sorted.pop(best_p))
            nonshared_vals.pop(best_p)
            if len(chosen) >= k:
                break
        if len(chosen) < k:
            rem = nonshared_sorted
            extra = list(rng.choice(rem, size=(k - len(chosen)), replace=False))
            chosen.extend(extra)
        return orthonormalize_np(Q_joint[:, chosen])

    raise ValueError(f"Unknown control method: {method}")


# -----------------------------------------------------------------------------
# Activation collectors
# -----------------------------------------------------------------------------

class DecodeLastTokenCollector:
    """Collect last-token hidden states only on decode calls (seq_len == 1)."""
    def __init__(self):
        self.enabled = False
        self.buf: List[np.ndarray] = []
        self.active_mask: Optional[torch.Tensor] = None

    def set(self, enabled: bool, active_mask: Optional[torch.Tensor] = None):
        self.enabled = bool(enabled)
        self.active_mask = active_mask

    def hook(self, module, inputs, output):
        if not self.enabled:
            return output
        hs = output[0] if isinstance(output, tuple) else output
        if not isinstance(hs, torch.Tensor) or hs.ndim != 3:
            return output
        if hs.shape[1] != 1:
            return output
        x = hs[:, -1, :]
        if self.active_mask is not None:
            m = self.active_mask.bool()
            if m.numel() == x.shape[0]:
                x = x[m]
        if x.numel() == 0:
            return output
        self.buf.append(x.detach().float().cpu().numpy())
        return output

    def pop(self) -> Optional[np.ndarray]:
        if not self.buf:
            return None
        out = np.concatenate(self.buf, axis=0)
        self.buf = []
        return out


class PrefillLastTokenCollector:
    """Collect last-token hidden state on prefill calls (seq_len > 1)."""
    def __init__(self):
        self.enabled = False
        self.buf: List[np.ndarray] = []

    def set(self, enabled: bool):
        self.enabled = bool(enabled)

    def hook(self, module, inputs, output):
        if not self.enabled:
            return output
        hs = output[0] if isinstance(output, tuple) else output
        if not isinstance(hs, torch.Tensor) or hs.ndim != 3:
            return output
        if hs.shape[1] <= 1:
            return output
        x = hs[:, -1, :]
        if x.numel() == 0:
            return output
        self.buf.append(x.detach().float().cpu().numpy())
        return output

    def pop(self) -> Optional[np.ndarray]:
        if not self.buf:
            return None
        out = np.concatenate(self.buf, axis=0)
        self.buf = []
        return out


# -----------------------------------------------------------------------------
# Intervention hooks
# -----------------------------------------------------------------------------

class HookStats:
    def __init__(self, name: str):
        self.name = name
        self.calls = 0
        self.intervened = 0


class DecodeLastTokenRemovalHook:
    """Apply h <- h - alpha * Q Q^T h only when seq_len == 1."""
    def __init__(self, Q_np: np.ndarray, alpha: float, stats: HookStats):
        self.alpha = float(alpha)
        self.stats = stats
        self.Q = torch.tensor(orthonormalize_np(Q_np), dtype=torch.float32)
        self.Q_dev: Optional[torch.Tensor] = None

    def _Q(self, device: torch.device) -> torch.Tensor:
        if self.Q_dev is None or self.Q_dev.device != device:
            self.Q_dev = self.Q.to(device=device)
        return self.Q_dev

    def __call__(self, module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        if not isinstance(hs, torch.Tensor) or hs.ndim != 3:
            return output
        if hs.shape[1] != 1:
            return output
        self.stats.calls += 1
        Q = self._Q(hs.device)
        x = hs[:, -1, :].float()
        proj = (x @ Q) @ Q.T
        hs2 = hs.clone()
        hs2[:, -1, :] = (x - self.alpha * proj).to(dtype=hs.dtype)
        self.stats.intervened += 1
        if isinstance(output, tuple):
            return (hs2,) + output[1:]
        return hs2


class PrefillLastTokenRemovalHook:
    """Apply h <- h - alpha * Q Q^T h only when seq_len > 1 (prefill)."""
    def __init__(self, Q_np: np.ndarray, alpha: float, stats: HookStats):
        self.alpha = float(alpha)
        self.stats = stats
        self.Q = torch.tensor(orthonormalize_np(Q_np), dtype=torch.float32)
        self.Q_dev: Optional[torch.Tensor] = None

    def _Q(self, device: torch.device) -> torch.Tensor:
        if self.Q_dev is None or self.Q_dev.device != device:
            self.Q_dev = self.Q.to(device=device)
        return self.Q_dev

    def __call__(self, module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        if not isinstance(hs, torch.Tensor) or hs.ndim != 3:
            return output
        if hs.shape[1] <= 1:
            return output
        self.stats.calls += 1
        Q = self._Q(hs.device)
        x = hs[:, -1, :].float()
        proj = (x @ Q) @ Q.T
        hs2 = hs.clone()
        hs2[:, -1, :] = (x - self.alpha * proj).to(dtype=hs.dtype)
        self.stats.intervened += 1
        if isinstance(output, tuple):
            return (hs2,) + output[1:]
        return hs2


def register_hook(
    model,
    *,
    layer_idx: int,
    locus: str,
    Q: np.ndarray,
    alpha: float,
    name: str,
) -> Tuple[List[Any], HookStats]:
    """Register exactly one hook at layer_idx; return handles + stats."""
    blocks = get_transformer_blocks(model)
    if not (0 <= layer_idx < len(blocks)):
        raise ValueError(f"layer_idx={layer_idx} out of range: 0..{len(blocks)-1}")
    stats = HookStats(name)
    if locus == "decode":
        hk = DecodeLastTokenRemovalHook(Q, alpha, stats)
    elif locus == "prefill":
        hk = PrefillLastTokenRemovalHook(Q, alpha, stats)
    else:
        raise ValueError(f"Unknown locus: {locus}")
    handle = blocks[layer_idx].register_forward_hook(hk)
    return [handle], stats


def remove_hooks(handles: List[Any]) -> None:
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def import_dataloaders():
    """Import load_selected_tasks from the DecodeShare package."""
    candidates = [
        "decodeshare.benchmark_dataloaders",
    ]
    last_err = None
    for mod in candidates:
        try:
            m = __import__(mod, fromlist=["load_selected_tasks"])
            return m
        except Exception as e:
            last_err = e
    raise ImportError(
        "Could not import benchmark dataloaders. Tried: "
        + ", ".join(candidates)
        + f". Last error: {last_err}"
    )


# -----------------------------------------------------------------------------
# Calibration state collection
# -----------------------------------------------------------------------------

def _subsample_rows_np(x: np.ndarray, n_max: int, seed: int) -> np.ndarray:
    if n_max is None or n_max <= 0 or x.shape[0] <= n_max:
        return x
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.shape[0], size=n_max, replace=False)
    return x[idx]


@torch.no_grad()
def collect_decode_states_for_task(
    *,
    model,
    tokenizer,
    prompts: List[str],
    layer_idx: int,
    batch_size: int,
    max_new_tokens: int,
    max_prompt_len: int,
    temperature: float,
    top_p: float,
    top_k: int,
    decoding: str,
    seed: int,
) -> np.ndarray:
    """Collect decode-time last-token states (seq_len==1 calls) during greedy/sample rollouts."""
    assert decoding in {"greedy", "sample"}
    device = next(model.parameters()).device
    eos = tokenizer.eos_token_id
    model.eval()

    blocks = get_transformer_blocks(model)
    collector = DecodeLastTokenCollector()
    handle = blocks[layer_idx].register_forward_hook(collector.hook)

    all_chunks: List[np.ndarray] = []
    rng = np.random.default_rng(seed)

    try:
        for i in tqdm(range(0, len(prompts), batch_size), desc="CalibDecode"):
            batch = prompts[i : i + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_prompt_len,
            ).to(device)
            input_ids = inputs["input_ids"]
            attn = inputs["attention_mask"]
            B = input_ids.shape[0]

            # Prefill (no capture)
            collector.set(False, None)
            out = model(input_ids=input_ids, attention_mask=attn, use_cache=True)
            logits = out.logits[:, -1, :]
            past = out.past_key_values

            unfinished = torch.ones(B, dtype=torch.bool, device=device)

            # Decode rollout; collect states on every seq_len==1 call
            for _step in range(max_new_tokens):
                if decoding == "greedy":
                    next_tok = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    lt = logits / max(temperature, 1e-6)
                    if top_k and top_k > 0:
                        v, _ = torch.topk(lt, min(top_k, lt.shape[-1]), dim=-1)
                        minv = v[:, -1].unsqueeze(-1)
                        lt = torch.where(lt < minv, torch.full_like(lt, float("-inf")), lt)
                    if top_p and 0.0 < top_p < 1.0:
                        sorted_logits, sorted_idx = torch.sort(lt, descending=True, dim=-1)
                        probs = torch.softmax(sorted_logits, dim=-1)
                        cum = torch.cumsum(probs, dim=-1)
                        mask = cum > top_p
                        mask[..., 0] = False
                        sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
                        lt2 = torch.full_like(lt, float("-inf"))
                        lt2.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)
                        lt = lt2
                    probs = torch.softmax(lt, dim=-1)
                    next_tok = torch.multinomial(probs, num_samples=1)

                next_tok = torch.where(
                    unfinished.unsqueeze(-1),
                    next_tok,
                    torch.full_like(next_tok, eos),
                )

                unfinished = unfinished & (next_tok.squeeze(-1) != eos)
                if not bool(unfinished.any().item()):
                    break

                attn = torch.cat([attn, torch.ones((B, 1), device=device, dtype=attn.dtype)], dim=1)

                collector.set(True, unfinished)
                out = model(
                    input_ids=next_tok,
                    attention_mask=attn,
                    use_cache=True,
                    past_key_values=past,
                )
                logits = out.logits[:, -1, :]
                past = out.past_key_values

            collector.set(False, None)
            chunk = collector.pop()
            if chunk is not None and chunk.shape[0] > 0:
                all_chunks.append(chunk)
    finally:
        try:
            handle.remove()
        except Exception:
            pass

    if not all_chunks:
        return np.zeros((0, 1), dtype=np.float32)
    return np.concatenate(all_chunks, axis=0)


@torch.no_grad()
def collect_prefill_last_states_for_task(
    *,
    model,
    tokenizer,
    prompts: List[str],
    layer_idx: int,
    batch_size: int,
    max_prompt_len: int,
) -> np.ndarray:
    """Collect prefill last-token states from a single full prompt forward (seq_len>1)."""
    device = next(model.parameters()).device
    model.eval()

    blocks = get_transformer_blocks(model)
    collector = PrefillLastTokenCollector()
    handle = blocks[layer_idx].register_forward_hook(collector.hook)

    all_chunks: List[np.ndarray] = []
    try:
        for i in tqdm(range(0, len(prompts), batch_size), desc="CalibPrefill"):
            batch = prompts[i : i + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_prompt_len,
            ).to(device)
            input_ids = inputs["input_ids"]
            attn = inputs["attention_mask"]
            collector.set(True)
            _ = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
            collector.set(False)
            chunk = collector.pop()
            if chunk is not None and chunk.shape[0] > 0:
                all_chunks.append(chunk)
    finally:
        try:
            handle.remove()
        except Exception:
            pass

    if not all_chunks:
        return np.zeros((0, 1), dtype=np.float32)
    return np.concatenate(all_chunks, axis=0)


# -----------------------------------------------------------------------------
# Utility: collect decode prompt-boundary states for energy calibration
# -----------------------------------------------------------------------------

@torch.no_grad()
def collect_prompt_boundary_decode_states(
    *,
    model,
    tokenizer,
    prompts: List[str],
    layer_idx: int,
    batch_size: int,
    max_prompt_len: int,
) -> np.ndarray:
    """Collect h_ℓ states at the *prompt-boundary decode step*."""
    device = next(model.parameters()).device
    model.eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    blocks = get_transformer_blocks(model)
    collector = DecodeLastTokenCollector()
    handle = blocks[layer_idx].register_forward_hook(collector.hook)

    chunks: List[np.ndarray] = []
    try:
        for i in tqdm(range(0, len(prompts), batch_size), desc="EnergyCalib(decode-boundary)"):
            batch = prompts[i : i + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_prompt_len,
            ).to(device)
            input_ids = inputs["input_ids"]
            attn = inputs["attention_mask"]
            B, T = input_ids.shape
            if T < 2:
                continue

            prefix_ids = input_ids[:, :-1]
            prefix_attn = attn[:, :-1]
            last_ids = input_ids[:, -1:].contiguous()
            full_attn = attn

            out0 = model(input_ids=prefix_ids, attention_mask=prefix_attn, use_cache=True)
            past = out0.past_key_values

            collector.set(True, torch.ones(B, dtype=torch.bool, device=device))
            _ = model(input_ids=last_ids, attention_mask=full_attn, use_cache=True, past_key_values=past)
            collector.set(False, None)
            x = collector.pop()
            if x is not None and x.shape[0] > 0:
                chunks.append(x)
    finally:
        try:
            handle.remove()
        except Exception:
            pass
        collector.set(False, None)

    if not chunks:
        return np.zeros((0, 1), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


# -----------------------------------------------------------------------------
# Generation evaluation (HF generate)
# -----------------------------------------------------------------------------

@torch.no_grad()
def generate_continuations_hf(
    *,
    model,
    tokenizer,
    prompts: List[str],
    batch_size: int,
    max_prompt_len: int,
    max_new_tokens: int,
    decoding: str = "greedy",   # "greedy" | "sample"
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    sample_seed: Optional[int] = None,
):
    assert decoding in ["greedy", "sample"]
    device = next(model.parameters()).device
    model.eval()

    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos = tokenizer.eos_token_id

    if decoding == "sample" and sample_seed is not None:
        torch.manual_seed(int(sample_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(sample_seed))

    all_text: List[str] = []
    all_eos_hit: List[int] = []
    all_new_tok: List[int] = []

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
        attn = inputs["attention_mask"]
        B, T0 = input_ids.shape

        gen_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            max_new_tokens=int(max_new_tokens),
            do_sample=(decoding == "sample"),
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            eos_token_id=eos,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )

        for b in range(B):
            out = gen_ids[b]
            cont = out[T0:]
            eos_pos = (cont == eos).nonzero(as_tuple=False)
            if eos_pos.numel() > 0:
                L = int(eos_pos[0].item()) + 1
                eos_hit = 1
                cont2 = cont[:L]
            else:
                L = int(cont.shape[0])
                eos_hit = 0
                cont2 = cont
            txt = tokenizer.decode(cont2, skip_special_tokens=True)
            all_text.append(txt)
            all_eos_hit.append(eos_hit)
            all_new_tok.append(L)

    return all_text, np.array(all_eos_hit, dtype=np.int32), np.array(all_new_tok, dtype=np.int32)


def score_generation_with_dataloader(
    *,
    task: str,
    continuations: List[str],
    gold: List[str],
    eos_hit: np.ndarray,
    new_tok: np.ndarray,
    dl_module,
) -> Dict[str, float]:
    """Uses dl_module.parse_prediction + dl_module.is_correct."""
    preds = [dl_module.parse_prediction(task, c) for c in continuations]
    correct = np.array(
        [1 if dl_module.is_correct(task, p, g) else 0 for p, g in zip(preds, gold)],
        dtype=np.int32,
    )
    extracted = np.array([1 if str(p).strip() != "" else 0 for p in preds], dtype=np.int32)

    return {
        "acc": float(correct.mean()) if len(correct) else float("nan"),
        "extraction_rate": float(extracted.mean()) if len(extracted) else float("nan"),
        "eos_rate": float(eos_hit.mean()) if len(eos_hit) else float("nan"),
        "avg_new_tokens": float(new_tok.mean()) if len(new_tok) else float("nan"),
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--model_dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--trust_remote_code", type=int, default=0)

    ap.add_argument("--tasks", type=str, default="commonsenseqa,strategyqa,arc_challenge")
    ap.add_argument("--layer", type=int, default=10)

    ap.add_argument("--n_subspace", type=int, default=128)
    ap.add_argument("--n_eval", type=int, default=256)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--template_seed", type=int, default=1234)
    ap.add_argument("--template_randomization", type=int, default=1)
    ap.add_argument("--shuffle_choices", type=int, default=1)
    ap.add_argument("--add_answer_prefix", type=int, default=1)
    ap.add_argument("--answer_prefix", type=str, default="\nFinal answer:")

    # Subspace calibration
    ap.add_argument("--calib_decode_max_new_tokens", type=int, default=128)
    ap.add_argument("--calib_decoding", type=str, default="greedy", choices=["greedy", "sample"])
    ap.add_argument("--calib_batch_size", type=int, default=8)
    ap.add_argument("--eval_batch_size", type=int, default=8)
    ap.add_argument("--max_prompt_len", type=int, default=1024)
    ap.add_argument("--per_task_max_states", type=int, default=20000)

    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=0)

    # Sharedness / PCA
    ap.add_argument("--rho", type=float, default=0.95)
    ap.add_argument("--tau", type=float, default=1e-3)
    ap.add_argument("--m_shared", type=str, default="all")
    ap.add_argument("--pca_device", type=str, default="cpu", choices=["cpu", "cuda"])

    # Control selection
    ap.add_argument("--control_method", type=str, default="varmatch", choices=["uniform", "topk", "varmatch"])

    # Energy matching
    ap.add_argument("--do_energy_match", type=int, default=1)
    ap.add_argument("--energy_calib_on", type=str, default="decode_boundary", choices=["decode_boundary", "decode_rollout"])
    ap.add_argument("--alpha_cap", type=float, default=5.0)

    # Generation eval
    ap.add_argument("--gen_max_new_tokens", type=int, default=256)
    ap.add_argument("--gen_decoding", type=str, default="greedy", choices=["greedy", "sample"])
    ap.add_argument("--gen_temperature", type=float, default=1.0)
    ap.add_argument("--gen_top_p", type=float, default=1.0)
    ap.add_argument("--gen_top_k", type=int, default=0)
    ap.add_argument("--gen_sample_seed", type=int, default=123)

    # Output
    ap.add_argument("--out_json", type=str, default="h3_generation_grid_results.json")

    args = ap.parse_args()
    set_global_seed(args.seed)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if not tasks:
        raise ValueError("No tasks specified")

    # Load model/tokenizer
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.model_dtype]
    print(f"[Load] model={args.model} device={args.device} dtype={args.model_dtype}")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=bool(args.trust_remote_code))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=bool(args.trust_remote_code),
        low_cpu_mem_usage=True,
    ).to(args.device)
    model.eval()

    # Load tasks/dataloaders
    dl = import_dataloaders()
    if not (hasattr(dl, "parse_prediction") and hasattr(dl, "is_correct")):
        raise RuntimeError("Your dataloader module must provide parse_prediction(task, text) and is_correct(task, pred, gold).")

    sub_by, eval_by, meta_by = dl.load_selected_tasks(
        tasks=tasks,
        n_subspace=args.n_subspace,
        n_eval=args.n_eval,
        seed=args.seed,
        template_randomization=bool(args.template_randomization),
        template_seed=args.template_seed,
        shuffle_choices=bool(args.shuffle_choices),
        add_answer_prefix=bool(args.add_answer_prefix),
        answer_prefix=args.answer_prefix,
    )
    prompts_calib = {t: [ex.prompt for ex in sub_by[t]] for t in tasks}

    # ---------------------------------------------------------------------
    # 1) Estimate decode-shared basis
    # ---------------------------------------------------------------------
    decode_states_by_task: Dict[str, np.ndarray] = {}
    for t in tasks:
        states = collect_decode_states_for_task(
            model=model,
            tokenizer=tok,
            prompts=prompts_calib[t],
            layer_idx=args.layer,
            batch_size=args.calib_batch_size,
            max_new_tokens=args.calib_decode_max_new_tokens,
            max_prompt_len=args.max_prompt_len,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            decoding=args.calib_decoding,
            seed=stable_int_seed(args.seed, t, "decode"),
        )
        states = _subsample_rows_np(states, args.per_task_max_states, seed=stable_int_seed(args.seed, t, "subsample_decode"))
        decode_states_by_task[t] = states
        print(f"[DecodeStates] {t}: {states.shape}")

    res_dec = estimate_shared_subspace(
        decode_states_by_task,
        rho=args.rho,
        tau=args.tau,
        m_shared=args.m_shared,
        pca_device=args.pca_device,
        seed=stable_int_seed(args.seed, "pca_dec"),
    )
    k_dec = len(res_dec.shared_indices)
    print(f"[Shared-Decode] k_pca={res_dec.Q_joint.shape[1]} k_shared={k_dec} (tau={args.tau}, m_shared={args.m_shared})")

    # ---------------------------------------------------------------------
    # 2) Estimate prefill-shared basis
    # ---------------------------------------------------------------------
    prefill_states_by_task: Dict[str, np.ndarray] = {}
    for t in tasks:
        states = collect_prefill_last_states_for_task(
            model=model,
            tokenizer=tok,
            prompts=prompts_calib[t],
            layer_idx=args.layer,
            batch_size=args.calib_batch_size,
            max_prompt_len=args.max_prompt_len,
        )
        states = _subsample_rows_np(states, args.per_task_max_states, seed=stable_int_seed(args.seed, t, "subsample_prefill"))
        prefill_states_by_task[t] = states
        print(f"[PrefillStates] {t}: {states.shape}")

    res_pre = estimate_shared_subspace(
        prefill_states_by_task,
        rho=args.rho,
        tau=args.tau,
        m_shared=args.m_shared,
        pca_device=args.pca_device,
        seed=stable_int_seed(args.seed, "pca_pre"),
    )
    k_pre = len(res_pre.shared_indices)
    print(f"[Shared-Prefill] k_pca={res_pre.Q_joint.shape[1]} k_shared={k_pre} (tau={args.tau}, m_shared={args.m_shared})")

    # Dimension-match
    k = int(min(k_dec, k_pre))
    if k <= 0:
        raise RuntimeError(f"No shared components after matching: k_dec={k_dec}, k_pre={k_pre}")
    print(f"[Match] k = min(k_dec, k_pre) = {k}")

    # Truncate shared bases to k (take highest-eigenvalue shared components)
    idx_dec_sorted = sorted(res_dec.shared_indices, key=lambda i: float(res_dec.eigvals[i]), reverse=True)[:k]
    idx_pre_sorted = sorted(res_pre.shared_indices, key=lambda i: float(res_pre.eigvals[i]), reverse=True)[:k]
    Q_dec_k = orthonormalize_np(res_dec.Q_joint[:, idx_dec_sorted])
    Q_pre_k = orthonormalize_np(res_pre.Q_joint[:, idx_pre_sorted])

    Q_ctrl_decode_k = select_nonshared_control(
        Q_joint=res_dec.Q_joint,
        eigvals=res_dec.eigvals,
        shared_idx=idx_dec_sorted,
        k=k,
        method=args.control_method,
        seed=stable_int_seed(args.seed, "ctrl_decode"),
    )
    Q_ctrl_prefill_k = select_nonshared_control(
        Q_joint=res_pre.Q_joint,
        eigvals=res_pre.eigvals,
        shared_idx=idx_pre_sorted,
        k=k,
        method=args.control_method,
        seed=stable_int_seed(args.seed, "ctrl_prefill"),
    )

    # Principal angles (diagnostic)
    angles = principal_angles_deg(Q_dec_k, Q_pre_k)
    ang_summary = {
        "mean": float(np.mean(angles)) if angles.size else float("nan"),
        "p50": float(np.percentile(angles, 50)) if angles.size else float("nan"),
        "p95": float(np.percentile(angles, 95)) if angles.size else float("nan"),
    }
    print(f"[Angles] mean={ang_summary['mean']:.2f}° p50={ang_summary['p50']:.2f}° p95={ang_summary['p95']:.2f}°")

    # ---------------------------------------------------------------------
    # 3) Energy calibration for α-matching (decode locus)
    # ---------------------------------------------------------------------
    all_calib_prompts: List[str] = []
    for t in tasks:
        all_calib_prompts.extend(prompts_calib[t])

    if len(all_calib_prompts) > 512:
        rng = np.random.default_rng(stable_int_seed(args.seed, "energy_calib_sub"))
        all_calib_prompts = list(rng.choice(np.array(all_calib_prompts, dtype=object), size=512, replace=False))

    if args.energy_calib_on == "decode_boundary":
        H_energy = collect_prompt_boundary_decode_states(
            model=model,
            tokenizer=tok,
            prompts=all_calib_prompts,
            layer_idx=args.layer,
            batch_size=args.calib_batch_size,
            max_prompt_len=args.max_prompt_len,
        )
    else:
        H_energy = np.concatenate(list(decode_states_by_task.values()), axis=0)
        H_energy = _subsample_rows_np(H_energy, 50000, seed=stable_int_seed(args.seed, "energy_rollout_sub"))

    def mean_proj_energy(H: np.ndarray, Q: np.ndarray) -> float:
        if H.size == 0:
            return float("nan")
        Q = orthonormalize_np(Q)
        proj = H.astype(np.float32, copy=False) @ Q
        e = np.sum(proj * proj, axis=1)
        return float(np.mean(e))

    E_dec = mean_proj_energy(H_energy, Q_dec_k)
    E_pre = mean_proj_energy(H_energy, Q_pre_k)
    E_ctrl = mean_proj_energy(H_energy, Q_ctrl_decode_k)
    eps = 1e-12

    alpha_dec = 1.0
    alpha_pre_raw = float(math.sqrt((E_dec + eps) / (E_pre + eps))) if args.do_energy_match else 1.0
    alpha_ctrl_raw = float(math.sqrt((E_dec + eps) / (E_ctrl + eps))) if args.do_energy_match else 1.0
    alpha_pre = float(min(max(alpha_pre_raw, 0.0), args.alpha_cap))
    alpha_ctrl = float(min(max(alpha_ctrl_raw, 0.0), args.alpha_cap))

    print(f"[Energy] E_dec={E_dec:.3e} E_pre={E_pre:.3e} E_ctrl={E_ctrl:.3e}")
    if args.do_energy_match:
        print(f"[Energy] alpha_pre(match)={alpha_pre:.3f} alpha_ctrl(match)={alpha_ctrl:.3f} (alpha_dec=1.0)")

    energy_diag = {
        "decode_boundary_states": int(H_energy.shape[0]),
        "E_dec": E_dec,
        "E_pre": E_pre,
        "E_ctrl": E_ctrl,
        "alpha_dec": alpha_dec,
        "alpha_pre_raw": alpha_pre_raw,
        "alpha_pre": alpha_pre,
        "alpha_ctrl_raw": alpha_ctrl_raw,
        "alpha_ctrl": alpha_ctrl,
        "ratio_dec": energy_stats(H_energy, Q_dec_k) if H_energy.size else {},
        "ratio_pre": energy_stats(H_energy, Q_pre_k) if H_energy.size else {},
        "ratio_ctrl": energy_stats(H_energy, Q_ctrl_decode_k) if H_energy.size else {},
    }

    # ---------------------------------------------------------------------
    # 4) Generation-only 2×2 grid evaluation
    # ---------------------------------------------------------------------
    results: Dict[str, Any] = {
        "config": vars(args),
        "tasks": tasks,
        "k_dec": k_dec,
        "k_pre": k_pre,
        "k_matched": k,
        "angles_deg": angles.astype(np.float32).tolist(),
        "angles_summary": ang_summary,
        "energy": energy_diag,
        "per_task_generation": {},
    }

    def run_gen_with_hook(locus: str, name: str, Q: np.ndarray, alpha: float, prompts: List[str], gold: List[str]) -> Dict[str, Any]:
        handles, st = register_hook(model, layer_idx=args.layer, locus=locus, Q=Q, alpha=alpha, name=name)
        try:
            cont, eos, nt = generate_continuations_hf(
                model=model,
                tokenizer=tok,
                prompts=prompts,
                batch_size=args.eval_batch_size,
                max_prompt_len=args.max_prompt_len,
                max_new_tokens=args.gen_max_new_tokens,
                decoding=args.gen_decoding,
                temperature=args.gen_temperature,
                top_p=args.gen_top_p,
                top_k=args.gen_top_k,
                sample_seed=args.gen_sample_seed,
            )
        finally:
            remove_hooks(handles)
        m = score_generation_with_dataloader(
            task=task_name,
            continuations=cont,
            gold=gold,
            eos_hit=eos,
            new_tok=nt,
            dl_module=dl,
        )
        m["hook_calls"] = int(st.calls)
        m["hook_intervened"] = int(st.intervened)
        return m

    for task_name in tasks:
        exs = eval_by[task_name]
        prompts_eval = [ex.prompt for ex in exs]
        gold = [ex.gold for ex in exs]

        # Baseline
        cont0, eos0, nt0 = generate_continuations_hf(
            model=model,
            tokenizer=tok,
            prompts=prompts_eval,
            batch_size=args.eval_batch_size,
            max_prompt_len=args.max_prompt_len,
            max_new_tokens=args.gen_max_new_tokens,
            decoding=args.gen_decoding,
            temperature=args.gen_temperature,
            top_p=args.gen_top_p,
            top_k=args.gen_top_k,
            sample_seed=args.gen_sample_seed,
        )
        m0 = score_generation_with_dataloader(
            task=task_name,
            continuations=cont0,
            gold=gold,
            eos_hit=eos0,
            new_tok=nt0,
            dl_module=dl,
        )

        per = {
            "n": len(prompts_eval),
            "baseline": m0,

            # decode-intervene quadrants
            "decode-est/decode-int": run_gen_with_hook("decode", "decode-est/decode-int", Q_dec_k, 1.0, prompts_eval, gold),
            "prefill-est/decode-int": run_gen_with_hook("decode", "prefill-est/decode-int", Q_pre_k, 1.0, prompts_eval, gold),
            "control(decode)/decode-int": run_gen_with_hook("decode", "control(decode)/decode-int", Q_ctrl_decode_k, 1.0, prompts_eval, gold),

            "prefill-est/decode-int_energy_match": run_gen_with_hook("decode", "prefill-est/decode-int(E)", Q_pre_k, alpha_pre, prompts_eval, gold),
            "control(decode)/decode-int_energy_match": run_gen_with_hook("decode", "control(decode)/decode-int(E)", Q_ctrl_decode_k, alpha_ctrl, prompts_eval, gold),

            # prefill-intervene quadrants
            "decode-est/prefill-int": run_gen_with_hook("prefill", "decode-est/prefill-int", Q_dec_k, 1.0, prompts_eval, gold),
            "prefill-est/prefill-int": run_gen_with_hook("prefill", "prefill-est/prefill-int", Q_pre_k, 1.0, prompts_eval, gold),
            "control(prefill)/prefill-int": run_gen_with_hook("prefill", "control(prefill)/prefill-int", Q_ctrl_prefill_k, 1.0, prompts_eval, gold),
        }

        results["per_task_generation"][task_name] = per

        def pct(x: float) -> str:
            return "nan" if not (x == x) else f"{100.0 * x:.1f}"

        print(
            f"\n[GEN-GRID] {task_name} (n={len(prompts_eval)})\n"
            f"  baseline_acc={pct(per['baseline']['acc'])}  "
            f"ex={pct(per['baseline']['extraction_rate'])}  "
            f"eos={pct(per['baseline']['eos_rate'])}  "
            f"len={per['baseline']['avg_new_tokens']:.1f}\n"
            f"  dd(acc)={pct(per['decode-est/decode-int']['acc'])}  "
            f"pd(acc)={pct(per['prefill-est/decode-int']['acc'])}  "
            f"ctrl(acc)={pct(per['control(decode)/decode-int']['acc'])}\n"
            f"  dp(acc)={pct(per['decode-est/prefill-int']['acc'])}  "
            f"pp(acc)={pct(per['prefill-est/prefill-int']['acc'])}  "
            f"ctrlp(acc)={pct(per['control(prefill)/prefill-int']['acc'])}\n"
        )

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=json_default)
    print(f"[Saved] {args.out_json}")


if __name__ == "__main__":
    main()
