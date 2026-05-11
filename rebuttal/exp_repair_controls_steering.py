
# -*- coding: utf-8 -*-
"""
exp_repair_controls_steering.py  (multi-layer capable)

Repair-vs-controls experiment for steering vectors:

  - Original steering vector v
  - Shared-subspace repair:      v_shared = (I - alpha_proj * Q_shared Q_shared^T) v
  - Random-subspace control:    v_rand   = (I - alpha_proj * Q_rand   Q_rand^T)   v
  - PCA-subspace control:       v_pca    = (I - alpha_proj * Q_pca    Q_pca^T)    v
  - Prefill-PCA control:        v_pca_prefill = (I - alpha_proj * Q_pca_prefill Q_pca_prefill^T) v
  - Shrinkage control:          v_shrink = gamma * v  (gamma chosen to norm-match v_shared by default)

Evaluate each method under **true KV-cached decoding** (decode-only steering),
across multiple prompt templates (template seeds), and report:
  - mean delta utility (vs baseline no-steer)
  - worst-case delta across templates (min over template seeds)
  - template variance / std (across template seeds)
  - range across templates (max-min)

This is designed to match DECODESHARE's protocol constraints:
  - estimate Q_shared from **decode-time hidden states** (seq_len==1)
  - interventions applied only during decode forward passes
  - template randomization + choice shuffling
  - focus on worst-case + variance, not only mean

PROJECT DEPENDENCY (recommended / aligns with your current code):
  benchmark_dataloaders providing:
    - Example
    - load_selected_tasks(...)
    - parse_prediction(...)
    - is_correct(...)
    - stable_int_seed(...)

Steering vector manifest format (JSONL, one per line), e.g.:
{"name":"truthful_l10_seed0","concept":"truthful","layer":10,"alpha":1.0,"path":"vectors/truthful_l10_seed0.npy"}

Optional precomputed bases:
  --shared_basis_npy_pattern "bases/Q_shared_layer{layer}.npy"
If provided, the script will load Q_shared per layer; it will still estimate Q_pca unless you also pass:
  --pca_basis_npy_pattern "bases/Q_pca_layer{layer}.npy"

If you want the script to estimate bases automatically for all layers appearing in your vector set:
  --basis_layers auto   (default)

"""

import os
import sys
import re
import json
import math
import argparse
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple, DefaultDict
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# -----------------------------
# Local imports (repo layout)
# -----------------------------
# This script lives in `rebuttal/`, while `benchmark_dataloaders.py` lives in `src/`.
# Make it work whether you run from repo root or from within `rebuttal/`.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.normpath(os.path.join(THIS_DIR, "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)


# -----------------------------
# Repro / stable seed
# -----------------------------
def stable_int_seed_fallback(*items: Any) -> int:
    s = "|".join(str(x) for x in items)
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


try:
    from benchmark_dataloaders import (
        Example,
        load_selected_tasks,
        parse_prediction,
        is_correct as is_correct_bool,
        stable_int_seed as stable_int_seed_project,
    )
    stable_int_seed = stable_int_seed_project
except Exception as e:
    Example = None  # type: ignore
    load_selected_tasks = None  # type: ignore
    parse_prediction = None  # type: ignore
    is_correct_bool = None  # type: ignore
    stable_int_seed = stable_int_seed_fallback
    _IMPORT_ERR = e


def set_global_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_hidden_dim(model) -> Optional[int]:
    cfg = getattr(model, "config", None)
    for k in ("hidden_size", "n_embd", "dim", "d_model", "model_dim", "embed_dim"):
        v = getattr(cfg, k, None)
        if isinstance(v, int) and v > 0:
            return v
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is not None:
        for k in ("hidden_size", "n_embd", "dim", "d_model", "model_dim", "embed_dim"):
            v = getattr(text_cfg, k, None)
            if isinstance(v, int) and v > 0:
                return v
    try:
        emb = model.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight") and isinstance(emb.weight, torch.Tensor) and emb.weight.ndim == 2:
            return int(emb.weight.shape[1])
        if emb is not None and hasattr(emb, "embedding_dim"):
            return int(emb.embedding_dim)
    except Exception:
        pass
    return None


def get_model_layers(model) -> List[torch.nn.Module]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return list(model.gpt_neox.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    if hasattr(model, "transformer") and hasattr(model.transformer, "blocks"):
        return list(model.transformer.blocks)
    raise RuntimeError(f"Cannot locate transformer layers for model class: {type(model)}")


# -----------------------------
# Prompt rendering
# -----------------------------
def render_prompt(tokenizer, user_prompt: str, *, add_generation_prompt: bool = True, system_prompt: Optional[str] = None) -> str:
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
        messages = [{"role": "user", "content": user_prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)


# -----------------------------
# Sampling utils
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
# Steering vector spec + loader
# -----------------------------
@dataclass
class SteeringVector:
    name: str
    concept: str
    layer: int
    alpha: float
    vec: np.ndarray
    template_tag: Optional[str] = None


def _load_vector_file(path: str) -> np.ndarray:
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.endswith(".npy"):
        v = np.load(path)
        return np.asarray(v, dtype=np.float32).reshape(-1)
    if path.endswith(".pt") or path.endswith(".pth"):
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):
            for k in ["vector", "v", "direction", "dir"]:
                if k in obj:
                    obj = obj[k]
                    break
        if isinstance(obj, torch.Tensor):
            return obj.detach().float().cpu().numpy().reshape(-1)
        if isinstance(obj, np.ndarray):
            return obj.astype(np.float32).reshape(-1)
        raise TypeError(f"Unsupported .pt content type: {type(obj)}")
    raise ValueError(f"Unsupported vector file extension: {path}")


def _resolve_manifest_vector_path(path_str: str, *, base_dir: str) -> str:
    raw = os.path.expanduser(str(path_str))
    if os.path.isabs(raw):
        return raw
    candidates = [
        os.path.abspath(raw),
        os.path.abspath(os.path.join(base_dir, raw)),
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    return candidates[-1]


def load_vectors_from_manifest(manifest_path: str, *, max_vectors: Optional[int] = None) -> List[SteeringVector]:
    manifest_path = os.path.expanduser(manifest_path)
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(manifest_path)

    base_dir = os.path.dirname(os.path.abspath(manifest_path))

    vecs: List[SteeringVector] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            item = json.loads(line)
            name = str(item.get("name", ""))
            concept = str(item.get("concept", item.get("attr", "unknown")))
            layer = int(item["layer"])
            alpha = float(item.get("alpha", 1.0))
            path = _resolve_manifest_vector_path(str(item["path"]), base_dir=base_dir)
            template_tag = item.get("template_tag", None)
            v = _load_vector_file(path)
            vecs.append(SteeringVector(name=name, concept=concept, layer=layer, alpha=alpha, vec=v, template_tag=template_tag))
            if max_vectors is not None and len(vecs) >= max_vectors:
                break
    if not vecs:
        raise RuntimeError(f"No vectors loaded from {manifest_path}")
    return vecs


# -----------------------------
# Linear algebra helpers
# -----------------------------
def orthonormalize_np(basis: np.ndarray) -> np.ndarray:
    q, _ = np.linalg.qr(basis.astype(np.float32, copy=False))
    return q.astype(np.float32, copy=False)


def project_out(v: np.ndarray, Q: np.ndarray, alpha_proj: float = 1.0) -> np.ndarray:
    if Q.size == 0:
        return v.copy()
    proj = Q @ (Q.T @ v)
    return (v - float(alpha_proj) * proj).astype(np.float32, copy=False)


def rand_orthonormal(d: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    M = rng.standard_normal(size=(d, k)).astype(np.float32)
    return orthonormalize_np(M)


# -----------------------------
# Decode-time activation collector (for Q_shared / Q_pca)
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


class PrefillLastTokenActivationCollector:
    """
    Collect last-token hidden states ONLY during prompt prefill forward passes (seq_len > 1).
    storage[task][layer_idx] -> list of np arrays [B, D]
    """

    def __init__(self, layer_indices: List[int]):
        self.layer_indices = list(layer_indices)
        self._cur_task: Optional[str] = None
        self.capture_enabled: bool = False
        self.storage: DefaultDict[str, DefaultDict[int, List[np.ndarray]]] = defaultdict(lambda: defaultdict(list))

    def set_current_task(self, task_name: str) -> None:
        self._cur_task = task_name

    def set_capture(self, enabled: bool) -> None:
        self.capture_enabled = bool(enabled)

    def make_hook(self, layer_idx: int):
        def _hook(module, inputs, output):
            if (not self.capture_enabled) or (self._cur_task is None):
                return output
            hs = output[0] if isinstance(output, tuple) else output
            if not isinstance(hs, torch.Tensor) or hs.ndim != 3:
                return output
            if hs.shape[1] <= 1:
                return output
            x = hs[:, -1, :]  # [B, D]
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


def _parse_csv_ints(s: str) -> List[int]:
    s = str(s or "").strip()
    if not s:
        return []
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _dedup_keep_order(items: List[int]) -> List[int]:
    seen = set()
    out: List[int] = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


@torch.no_grad()
def collect_decode_last_token_states(
    model,
    tokenizer,
    prompts: List[str],
    collector: DecodeLastTokenActivationCollector,
    *,
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


@torch.no_grad()
def collect_prefill_last_token_states(
    model,
    tokenizer,
    prompts: List[str],
    collector: PrefillLastTokenActivationCollector,
    *,
    batch_size: int,
    max_prompt_len: int,
) -> None:
    device = next(model.parameters()).device
    model.eval()

    for i in tqdm(range(0, len(prompts), batch_size), desc="CollectPrefill"):
        use_template = bool(getattr(tokenizer, "chat_template", None))
        batch = prompts[i:i + batch_size]
        batch = [render_prompt(tokenizer, p, add_generation_prompt=True) for p in batch]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_len,
            add_special_tokens=not use_template,
        ).to(device)

        collector.set_capture(True)
        _out = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], use_cache=True)
        collector.set_capture(False)


def estimate_shared_and_pca_bases(
    *,
    model,
    tokenizer,
    prompts_by_task: Dict[str, List[str]],
    layer_idx: int,
    calib_batch_size: int,
    calib_max_new_tokens: int,
    per_task_max_states: int,
    max_prompt_len: int,
    decoding: str,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    # PCA
    pca_var: float,
    pca_max_dim: int,
    pca_max_rows: int,
    # sharedness
    tau: float,
    m_shared: str,   # "all" or int
) -> Dict[str, Any]:
    """
    Returns dict with:
      - joint_subspace: [D, cross_dim]
      - cross_dim: int
      - per_task_vars: {task: [cross_dim] variances}
      - shared_indices: List[int]
      - mean: [D] mean of pooled states
    """
    layers = get_model_layers(model)
    if layer_idx < 0 or layer_idx >= len(layers):
        raise ValueError(f"layer_idx out of range: {layer_idx} (n_layers={len(layers)})")

    collector = DecodeLastTokenActivationCollector([layer_idx])
    handle = layers[layer_idx].register_forward_hook(collector.make_hook(layer_idx))

    try:
        for task, prompts in prompts_by_task.items():
            collector.set_current_task(task)
            collect_decode_last_token_states(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                collector=collector,
                batch_size=calib_batch_size,
                max_new_tokens=calib_max_new_tokens,
                decoding=decoding,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_prompt_len=max_prompt_len,
            )
    finally:
        try:
            handle.remove()
        except Exception:
            pass
        collector.set_capture(False, None)

    # Gather + subsample
    task_X: Dict[str, np.ndarray] = {}
    for task in prompts_by_task.keys():
        X = collector.get_task_activations(task, layer_idx)
        if X is None or X.shape[0] == 0:
            continue
        X = _subsample_rows_np(X, per_task_max_states, seed=stable_int_seed(seed, task, layer_idx, "subsample"))
        task_X[task] = X.astype(np.float32, copy=False)

    if not task_X:
        raise RuntimeError(f"No decode states collected at layer {layer_idx} for basis estimation.")

    # Pool across tasks, center
    X_all = np.concatenate(list(task_X.values()), axis=0)
    if pca_max_rows > 0 and X_all.shape[0] > pca_max_rows:
        X_all = _subsample_rows_np(X_all, pca_max_rows, seed=stable_int_seed(seed, layer_idx, "pca_pool"))
    mean = X_all.mean(axis=0, keepdims=True).astype(np.float32)
    Xc = (X_all - mean).astype(np.float32, copy=False)

    # PCA via torch.pca_lowrank (CPU by default)
    Xc_t = torch.from_numpy(Xc)
    q = min(pca_max_dim, Xc_t.shape[1], Xc_t.shape[0] - 1)
    if q <= 0:
        raise RuntimeError("Not enough rows for PCA.")
    U, S, V = torch.pca_lowrank(Xc_t, q=q, center=False)
    s2 = (S ** 2).cpu().numpy()
    ratio = s2 / max(np.sum(s2), 1e-12)
    cum = np.cumsum(ratio)
    cross_dim = int(np.searchsorted(cum, pca_var) + 1)
    cross_dim = max(1, min(cross_dim, V.shape[1]))
    joint = V[:, :cross_dim].contiguous().cpu().numpy().astype(np.float32, copy=False)

    # Per-task component variances (on centered task states)
    per_task_vars: Dict[str, np.ndarray] = {}
    for task, X in task_X.items():
        Xct = (X - mean).astype(np.float32, copy=False)
        Z = Xct @ joint  # [n, cross_dim]
        per_task_vars[task] = np.var(Z, axis=0)

    # Sharedness selection
    tasks = list(per_task_vars.keys())
    if m_shared == "all":
        m_req = len(tasks)
    else:
        try:
            m_req = max(2, int(m_shared))
        except Exception:
            m_req = len(tasks)

    shared_indices: List[int] = []
    for j in range(cross_dim):
        active = 0
        for t in tasks:
            vt = per_task_vars[t]
            mx = float(np.max(vt)) + 1e-12
            if float(vt[j]) >= float(tau) * mx:
                active += 1
        if active >= m_req:
            shared_indices.append(j)

    # Sort shared indices by pooled variance
    if shared_indices:
        pooled = np.mean(np.stack([per_task_vars[t] for t in tasks], axis=0), axis=0)
        shared_indices.sort(key=lambda j: float(pooled[j]), reverse=True)

    return {
        "joint_subspace": joint,
        "cross_dim": cross_dim,
        "per_task_vars": {k: v.tolist() for k, v in per_task_vars.items()},
        "shared_indices": shared_indices,
        "mean": mean.reshape(-1).astype(np.float32).tolist(),
        "tasks_used": tasks,
    }


# -----------------------------
# Prefill-PCA basis estimation (control for prefill/decode mismatch)
# -----------------------------
def estimate_prefill_pca_basis(
    *,
    model,
    tokenizer,
    prompts_by_task: Dict[str, List[str]],
    layer_idx: int,
    calib_batch_size: int,
    per_task_max_states: int,
    max_prompt_len: int,
    seed: int,
    k: int,
    pca_max_dim: int,
    pca_max_rows: int,
) -> Dict[str, Any]:
    """
    Estimate PCA basis from prompt prefill distribution (seq_len>1 last-token states).
    Returns dict with:
      - Q: [D,k] orthonormal basis
      - n_rows: pooled rows used (after subsampling)
      - d: hidden dim
      - tasks_used: List[str]
    """
    layers = get_model_layers(model)
    if layer_idx < 0 or layer_idx >= len(layers):
        raise ValueError(f"layer_idx out of range: {layer_idx} (n_layers={len(layers)})")

    collector = PrefillLastTokenActivationCollector([layer_idx])
    handle = layers[layer_idx].register_forward_hook(collector.make_hook(layer_idx))

    try:
        for task, prompts in prompts_by_task.items():
            collector.set_current_task(task)
            collect_prefill_last_token_states(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                collector=collector,
                batch_size=calib_batch_size,
                max_prompt_len=max_prompt_len,
            )
    finally:
        try:
            handle.remove()
        except Exception:
            pass
        collector.set_capture(False)

    # Gather + subsample
    task_X: Dict[str, np.ndarray] = {}
    for task in prompts_by_task.keys():
        X = collector.get_task_activations(task, layer_idx)
        if X is None or X.shape[0] == 0:
            continue
        X = _subsample_rows_np(X, per_task_max_states, seed=stable_int_seed(seed, task, layer_idx, "prefill_subsample"))
        task_X[task] = X.astype(np.float32, copy=False)

    if not task_X:
        raise RuntimeError(f"No prefill states collected at layer {layer_idx} for PCA estimation.")

    # Pool across tasks, center
    X_all = np.concatenate(list(task_X.values()), axis=0)
    if pca_max_rows > 0 and X_all.shape[0] > pca_max_rows:
        X_all = _subsample_rows_np(X_all, pca_max_rows, seed=stable_int_seed(seed, layer_idx, "prefill_pca_pool"))
    mean = X_all.mean(axis=0, keepdims=True).astype(np.float32)
    Xc = (X_all - mean).astype(np.float32, copy=False)

    # PCA (top-k)
    Xc_t = torch.from_numpy(Xc)
    d = int(Xc_t.shape[1])
    k = int(k)
    q_max = min(int(pca_max_dim), int(d), int(Xc_t.shape[0] - 1))
    if k <= 0:
        raise ValueError(f"k must be > 0 (got {k})")
    if q_max < k:
        raise RuntimeError(
            f"Prefill PCA cannot produce k={k} comps at layer {layer_idx}: "
            f"n_rows={int(Xc_t.shape[0])} d={d} pca_max_dim={int(pca_max_dim)} (max_k={q_max}). "
            f"Reduce --shared_dim / relax shared selection, or increase --n_subspace / --per_task_max_states."
        )

    torch.manual_seed(int(stable_int_seed(seed, "prefill_pca_lowrank", layer_idx, k)))
    _U, _S, V = torch.pca_lowrank(Xc_t, q=k, center=False)
    Q = V[:, :k].contiguous().cpu().numpy().astype(np.float32, copy=False)
    Q = orthonormalize_np(Q)

    return {
        "Q": Q,
        "n_rows": int(Xc.shape[0]),
        "d": int(d),
        "tasks_used": list(task_X.keys()),
    }


def _subspace_cos_singulars(Qa: np.ndarray, Qb: np.ndarray) -> Dict[str, float]:
    if Qa.size == 0 or Qb.size == 0:
        return {"mean": float("nan"), "min": float("nan"), "max": float("nan")}
    M = Qa.T @ Qb
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    return {"mean": float(np.mean(s)), "min": float(np.min(s)), "max": float(np.max(s))}


# -----------------------------
# Steering hooks + generation (decode-only)
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
        self.prefill_calls = 0
        self.decode_calls = 0
        self.intervened = 0


class LastTokenSteeringHook:
    """
    Decode-only steering: add alpha * v on seq_len==1 forwards.
    If staged=True, only apply while (unfinished & gen_steps < reasoning_threshold).
    """
    def __init__(self, v_np: np.ndarray, alpha: float, stats: HookStats, *, staged: bool, reasoning_threshold: int):
        self.v = torch.tensor(v_np.astype(np.float32, copy=False))
        self.v_device: Optional[torch.Tensor] = None
        self.alpha = float(alpha)
        self.stats = stats
        self.staged = bool(staged)
        self.reasoning_threshold = int(reasoning_threshold)
        self.state: Optional[GenerationState] = None

    def set_state(self, st: Optional[GenerationState]) -> None:
        self.state = st

    def _v(self, device: torch.device) -> torch.Tensor:
        if self.v_device is None or self.v_device.device != device:
            self.v_device = self.v.to(device=device)
        return self.v_device

    def __call__(self, module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        if not isinstance(hs, torch.Tensor) or hs.ndim != 3:
            return output

        seq_len = hs.shape[1]
        if seq_len == 1:
            self.stats.decode_calls += 1
        else:
            self.stats.prefill_calls += 1
            return output  # decode-only

        if self.staged and self.state is not None:
            mask = self.state.current_reasoning_mask()
            if not bool(mask.any().item()):
                return output
            x = hs[:, -1, :].float()
            v = self._v(hs.device)
            x_sel = x[mask]
            x_sel = x_sel + self.alpha * v
            x[mask] = x_sel
            hs2 = hs.clone()
            hs2[:, -1, :] = x.to(dtype=hs.dtype)
            self.stats.intervened += 1
            if isinstance(output, tuple):
                return (hs2,) + output[1:]
            return hs2

        # non-staged
        x = hs[:, -1, :].float()
        v = self._v(hs.device)
        hs2 = hs.clone()
        hs2[:, -1, :] = (x + self.alpha * v).to(dtype=hs.dtype)
        self.stats.intervened += 1
        if isinstance(output, tuple):
            return (hs2,) + output[1:]
        return hs2


def register_decode_only_steering_hook(
    model,
    *,
    layer_idx: int,
    v_np: np.ndarray,
    alpha: float,
    staged: bool,
    reasoning_threshold: int,
) -> Tuple[List[Any], Optional[Any], List[HookStats]]:
    layers = get_model_layers(model)
    if layer_idx < 0 or layer_idx >= len(layers):
        raise ValueError(f"layer_idx out of range: {layer_idx} (n_layers={len(layers)})")

    stats = HookStats(name=f"decode_only{'_staged' if staged else ''}@{layer_idx}")
    hook = LastTokenSteeringHook(v_np=v_np, alpha=alpha, stats=stats, staged=staged, reasoning_threshold=reasoning_threshold)
    handle = layers[layer_idx].register_forward_hook(hook)

    def setter(st: Optional[GenerationState]) -> None:
        hook.set_state(st)

    return [handle], (setter if staged else None), [stats]


def remove_hooks(handles: List[Any]) -> None:
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass


@torch.no_grad()
def generate_continuations(
    model,
    tokenizer,
    prompts: List[str],
    *,
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


def load_model_and_tokenizer(model_name: str, device: str, model_dtype: str):
    dtype = torch.float32 if model_dtype == "fp32" else torch.float16
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)

    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = model.to(device)
    model.eval()
    model.config.use_cache = True
    return model, tok


def _is_correct(dataset: str, pred: str, gold: str) -> int:
    if is_correct_bool is None:
        raise RuntimeError(f"benchmark_dataloaders.is_correct not available: {_IMPORT_ERR}")
    return int(is_correct_bool(dataset, pred, gold))


@torch.no_grad()
def eval_decode_steering(
    *,
    model,
    tokenizer,
    examples: List[Any],
    decoding: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    device: str,
    batch_size: int,
    max_prompt_len: int,
    reasoning_token_threshold: int,
    # steering:
    v_np: Optional[np.ndarray],
    alpha: float,
    layer_idx: int,
    staged: bool,
    sample_seed: Optional[int],
) -> float:
    if v_np is None:
        handles, setter = [], None
    else:
        handles, setter, _stats = register_decode_only_steering_hook(
            model=model, layer_idx=layer_idx, v_np=v_np, alpha=alpha,
            staged=staged, reasoning_threshold=reasoning_token_threshold
        )

    try:
        prompts = [ex.prompt for ex in examples]
        continuations, _eos_hit, _new_tok = generate_continuations(
            model=model, tokenizer=tokenizer, prompts=prompts,
            decoding=decoding, max_new_tokens=max_new_tokens,
            temperature=temperature, top_p=top_p, top_k=top_k,
            device=device, batch_size=batch_size, max_prompt_len=max_prompt_len,
            reasoning_token_threshold=reasoning_token_threshold,
            state_setter=setter, sample_seed=sample_seed,
        )

        correct = []
        for ex, cont in zip(examples, continuations):
            if parse_prediction is None:
                raise RuntimeError(f"benchmark_dataloaders.parse_prediction not available: {_IMPORT_ERR}")
            pred = parse_prediction(ex.dataset, cont)
            correct.append(_is_correct(ex.dataset, pred, ex.gold))
        return float(np.mean(correct)) if correct else float("nan")
    finally:
        remove_hooks(handles)


# -----------------------------
# Experiment driver
# -----------------------------
def _format_layer_pattern(pat: str, layer: int) -> str:
    if "{layer}" in pat:
        return pat.format(layer=layer)
    return pat


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model_dtype", type=str, default="fp32", choices=["fp32", "fp16"])

    ap.add_argument("--vectors_manifest", type=str, required=True)
    ap.add_argument("--max_vectors", type=int, default=0)
    ap.add_argument("--filter_regex", type=str, default="")

    # Tasks for evaluation + subspace estimation
    ap.add_argument("--tasks_eval", type=str, default="commonsenseqa,arc_challenge,openbookqa,qasc,logiqa")
    ap.add_argument("--n_eval", type=int, default=128)

    ap.add_argument("--tasks_subspace", type=str, default="gsm8k,commonsenseqa,strategyqa,aqua,arc_challenge,openbookqa,qasc,logiqa")
    ap.add_argument("--n_subspace", type=int, default=128)

    # Templates for evaluation (variance)
    ap.add_argument("--template_randomization", type=int, default=1, choices=[0, 1])
    ap.add_argument("--shuffle_choices", type=int, default=1, choices=[0, 1])
    ap.add_argument("--answer_prefix", type=str, default="\nFinal answer:")
    ap.add_argument("--add_answer_prefix", type=int, default=1, choices=[0, 1])
    ap.add_argument("--template_seeds", type=str, default="1234,2345,3456,4567,5678",
                    help="Comma-separated template seeds used to measure template variance.")
    ap.add_argument(
        "--subspace_template_seed",
        type=int,
        default=None,
        help="Template seed used to build basis-estimation prompts. Default: first seed from --template_seeds.",
    )
    ap.add_argument(
        "--subspace_shuffle_choices",
        type=int,
        default=-1,
        choices=[-1, 0, 1],
        help="Whether to shuffle choices when building basis-estimation prompts (-1 follows --shuffle_choices).",
    )

    # Decoding
    ap.add_argument("--decoding", type=str, default="greedy", choices=["greedy", "sample"])
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--reasoning_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_prompt_len", type=int, default=512)
    ap.add_argument("--sample_seed", type=int, default=12345)
    ap.add_argument("--staged", type=int, default=1, choices=[0, 1])

    # Repair knobs
    ap.add_argument("--alpha_proj", type=float, default=1.0, help="Projection amount in (I - alpha Q Q^T).")
    ap.add_argument("--norm_match", type=int, default=1, choices=[0, 1],
                    help="If 1: scale control repaired vectors to match ||v_shared|| for fair energy budget.")

    # Which layers to build bases for
    ap.add_argument("--basis_layers", type=str, default="auto",
                    help="'auto' = unique layers in vectors; else comma-separated layers like '10,15,24'.")

    # Precomputed bases (optional patterns)
    ap.add_argument("--shared_basis_npy_pattern", type=str, default="",
                    help="Optional: pattern for Q_shared .npy, e.g. 'bases/Q_shared_layer{layer}.npy'.")
    ap.add_argument("--pca_basis_npy_pattern", type=str, default="",
                    help="Optional: pattern for Q_pca .npy, e.g. 'bases/Q_pca_layer{layer}.npy'.")
    ap.add_argument("--include_pca_prefill", type=int, default=0, choices=[0, 1],
                    help="If 1: add control 'pca_prefill' using prefill-distribution PCA as Q_pca.")

    # Basis estimation config (only used if bases not loaded)
    ap.add_argument("--calib_decode_max_new_tokens", type=int, default=-1,
                    help="Max decode steps collected for basis estimation. -1 uses --reasoning_tokens.")
    ap.add_argument("--per_task_max_states", type=int, default=20000)
    ap.add_argument("--pca_var", type=float, default=0.95)
    ap.add_argument("--pca_max_dim", type=int, default=4096)
    ap.add_argument("--pca_max_rows", type=int, default=200000, help="Max pooled rows used for PCA (0=no limit).")
    ap.add_argument("--tau", type=float, default=0.001)
    ap.add_argument("--m_shared", type=str, default="all")
    ap.add_argument("--shared_dim", type=int, default=0, help="0 means use all shared components; else top-k.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_json", type=str, default="repair_controls_results.json")

    args = ap.parse_args()
    set_global_seed(args.seed)

    if load_selected_tasks is None:
        raise RuntimeError(
            "benchmark_dataloaders is required for this script. "
            f"Import error was: {_IMPORT_ERR}"
        )

    max_vectors = None if args.max_vectors <= 0 else int(args.max_vectors)
    vecs = load_vectors_from_manifest(args.vectors_manifest, max_vectors=max_vectors)

    if args.filter_regex:
        pat = re.compile(args.filter_regex)
        vecs = [v for v in vecs if pat.search(v.name) or pat.search(v.concept)]
        if not vecs:
            raise RuntimeError("No vectors after --filter_regex")

    tasks_eval = [t.strip() for t in args.tasks_eval.split(",") if t.strip()]
    tasks_sub = [t.strip() for t in args.tasks_subspace.split(",") if t.strip()]
    template_seeds = _dedup_keep_order(_parse_csv_ints(args.template_seeds))
    if not template_seeds:
        raise ValueError("Empty --template_seeds")

    subspace_template_seed = int(args.subspace_template_seed) if args.subspace_template_seed is not None else int(template_seeds[0])
    if args.subspace_shuffle_choices < 0:
        subspace_shuffle_choices = bool(args.shuffle_choices)
    else:
        subspace_shuffle_choices = bool(args.subspace_shuffle_choices)

    # Decide which layers to build bases for
    if args.basis_layers.strip().lower() == "auto":
        basis_layers = sorted(set(int(v.layer) for v in vecs))
    else:
        basis_layers = _dedup_keep_order(_parse_csv_ints(args.basis_layers))
    if not basis_layers:
        raise ValueError("No basis layers specified.")

    model, tokenizer = load_model_and_tokenizer(args.model, args.device, args.model_dtype)
    hidden_dim = infer_hidden_dim(model)
    if hidden_dim is None:
        print("[Warn] Could not infer hidden_dim; will skip dimension check.")
    else:
        for v in vecs:
            if v.vec.shape[0] != hidden_dim:
                raise ValueError(f"Vector dim mismatch for {v.name}: {v.vec.shape[0]} != {hidden_dim}")

    # -------------------
    # Prepare prompts_by_task for basis estimation (once)
    # -------------------
    print("[Basis] Preparing prompts for basis estimation ...")
    calib_max_new_tokens = int(args.calib_decode_max_new_tokens)
    if calib_max_new_tokens <= 0:
        calib_max_new_tokens = int(args.reasoning_tokens)

    sub_by, _eval_by_dummy, _meta = load_selected_tasks(
        tasks=tasks_sub,
        n_subspace=max(1, args.n_subspace),
        n_eval=1,  # (compat) loader may not accept 0
        seed=args.seed,
        template_seed=subspace_template_seed,
        template_randomization=bool(args.template_randomization),
        shuffle_choices=subspace_shuffle_choices,
        add_answer_prefix=bool(args.add_answer_prefix),
        answer_prefix=args.answer_prefix,
    )
    prompts_by_task = {t: [ex.prompt for ex in sub_by[t]] for t in tasks_sub if t in sub_by}
    print(f"[Basis] Using subspace_template_seed={subspace_template_seed} shuffle_choices={int(subspace_shuffle_choices)}")

    # -------------------
    # Load / estimate per-layer bases
    # -------------------
    bases_by_layer: Dict[int, Dict[str, Any]] = {}

    for layer in basis_layers:
        print("\n" + "=" * 80)
        print(f"[Basis] Building bases for layer={layer} ...")

        # 1) Q_shared
        Q_shared = None
        if args.shared_basis_npy_pattern:
            spath = _format_layer_pattern(os.path.expanduser(args.shared_basis_npy_pattern), layer)
            if os.path.exists(spath):
                Q_shared = np.load(spath).astype(np.float32)
                if Q_shared.ndim != 2:
                    raise ValueError(f"{spath} must contain a 2D array [D,k]")
                Q_shared = orthonormalize_np(Q_shared)
                print(f"  - loaded Q_shared from {spath} (k={Q_shared.shape[1]})")
            else:
                print(f"  - shared basis file not found: {spath} (will estimate)")

        # 2) Q_pca (optional load)
        Q_pca = None
        if args.pca_basis_npy_pattern:
            ppath = _format_layer_pattern(os.path.expanduser(args.pca_basis_npy_pattern), layer)
            if os.path.exists(ppath):
                Q_pca = np.load(ppath).astype(np.float32)
                if Q_pca.ndim != 2:
                    raise ValueError(f"{ppath} must contain a 2D array [D,k]")
                Q_pca = orthonormalize_np(Q_pca)
                print(f"  - loaded Q_pca from {ppath} (k={Q_pca.shape[1]})")

        # If either basis missing, estimate from decode states (also provides joint_subspace)
        info = None
        if Q_shared is None or Q_pca is None:
            print("  - estimating decode-time PCA (+ shared selection if needed) ...")
            info = estimate_shared_and_pca_bases(
                model=model,
                tokenizer=tokenizer,
                prompts_by_task=prompts_by_task,
                layer_idx=layer,
                calib_batch_size=args.batch_size,
                calib_max_new_tokens=calib_max_new_tokens,
                per_task_max_states=args.per_task_max_states,
                max_prompt_len=args.max_prompt_len,
                decoding="greedy",
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                seed=args.seed,
                pca_var=args.pca_var,
                pca_max_dim=args.pca_max_dim,
                pca_max_rows=args.pca_max_rows,
                tau=args.tau,
                m_shared=args.m_shared,
            )
            joint = info["joint_subspace"]
            shared_indices = info["shared_indices"]

            if Q_shared is None:
                if not shared_indices:
                    raise RuntimeError(f"No shared components selected at layer {layer}. Try relax --tau or --m_shared.")
                if args.shared_dim > 0:
                    shared_indices = shared_indices[:int(args.shared_dim)]
                Q_shared = orthonormalize_np(joint[:, shared_indices])
                print(f"  - estimated Q_shared k={Q_shared.shape[1]} (cross_dim={info['cross_dim']})")

            if Q_pca is None:
                k = Q_shared.shape[1]
                Q_pca = orthonormalize_np(joint[:, :k])
                print(f"  - estimated Q_pca top-{k} comps (cross_dim={info['cross_dim']})")

        assert Q_shared is not None and Q_pca is not None
        d = Q_shared.shape[0]
        k = Q_shared.shape[1]
        if Q_pca.shape[1] != k:
            # align dims (take min)
            k2 = min(k, Q_pca.shape[1])
            Q_shared = Q_shared[:, :k2]
            Q_pca = Q_pca[:, :k2]
            k = k2
            print(f"  - [Warn] aligned basis dims to k={k}")

        # Optional: PCA control estimated from prefill (prompt) distribution
        Q_pca_prefill = None
        pca_prefill_info = None
        if bool(args.include_pca_prefill):
            print(f"  - estimating Q_pca_prefill top-{k} comps from prefill states ...")
            pinfo = estimate_prefill_pca_basis(
                model=model,
                tokenizer=tokenizer,
                prompts_by_task=prompts_by_task,
                layer_idx=layer,
                calib_batch_size=args.batch_size,
                per_task_max_states=args.per_task_max_states,
                max_prompt_len=args.max_prompt_len,
                seed=args.seed,
                k=k,
                pca_max_dim=args.pca_max_dim,
                pca_max_rows=args.pca_max_rows,
            )
            Q_pca_prefill = pinfo["Q"]
            cos = _subspace_cos_singulars(Q_pca, Q_pca_prefill)
            pca_prefill_info = {
                "n_rows": int(pinfo["n_rows"]),
                "cos_singulars_vs_decode_pca": cos,
                "tasks_used": pinfo["tasks_used"],
            }
            print(
                f"  - estimated Q_pca_prefill (n_rows={pinfo['n_rows']})  "
                f"cos(mean/min) vs decode-PCA={cos['mean']:.3f}/{cos['min']:.3f}"
            )

        Q_rand = rand_orthonormal(d, k, seed=stable_int_seed(args.seed, "Q_rand", layer, d, k))
        bases_by_layer[layer] = {
            "Q_shared": Q_shared,
            "Q_pca": Q_pca,
            "Q_pca_prefill": Q_pca_prefill,
            "Q_rand": Q_rand,
            "k": int(k),
            "d": int(d),
            "info": info,  # may be None if both loaded
            "pca_prefill_info": pca_prefill_info,
        }

    # -------------------
    # Pre-load evaluation sets for each template seed and baseline acc
    # -------------------
    eval_sets: Dict[int, Dict[str, List[Any]]] = {}
    base_acc: Dict[int, Dict[str, float]] = {}

    for tseed in template_seeds:
        print("\n" + "=" * 80)
        print(f"[Data] Loading eval set for template_seed={tseed} ...")
        _sub_by_dummy, eval_by, _meta = load_selected_tasks(
            tasks=tasks_eval,
            n_subspace=1,  # (compat) loader may not accept 0
            n_eval=args.n_eval,
            seed=args.seed,
            template_seed=tseed,
            template_randomization=bool(args.template_randomization),
            shuffle_choices=bool(args.shuffle_choices),
            add_answer_prefix=bool(args.add_answer_prefix),
            answer_prefix=args.answer_prefix,
        )
        eval_sets[tseed] = eval_by

        # baseline no steering (layer idx doesn't matter when v_np=None)
        base_acc[tseed] = {}
        for t in tasks_eval:
            acc0 = eval_decode_steering(
                model=model, tokenizer=tokenizer, examples=eval_by[t],
                decoding=args.decoding, max_new_tokens=args.max_new_tokens,
                temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                device=args.device, batch_size=args.batch_size, max_prompt_len=args.max_prompt_len,
                reasoning_token_threshold=args.reasoning_tokens,
                v_np=None, alpha=0.0, layer_idx=basis_layers[0], staged=False,
                sample_seed=(args.sample_seed if args.decoding == "sample" else None),
            )
            base_acc[tseed][t] = acc0
            print(f"  [Baseline] {t}: acc={acc0*100:.1f}")

    # -------------------
    # Evaluate: original vs repaired vs controls
    # -------------------
    methods = ["orig", "shared", "rand", "pca", "shrink"]
    if bool(args.include_pca_prefill):
        methods = ["orig", "shared", "rand", "pca", "pca_prefill", "shrink"]
    staged = bool(args.staged)
    norm_match = bool(args.norm_match)

    results: Dict[str, Any] = {
        "config": vars(args),
        "basis_layers": basis_layers,
        "basis_summary": {
            str(layer): {
                "k": bases_by_layer[layer]["k"],
                "d": bases_by_layer[layer]["d"],
                "pca_prefill_info": bases_by_layer[layer].get("pca_prefill_info", None),
            }
            for layer in basis_layers
        },
        "vectors": {},
    }

    for sv in vecs:
        if sv.layer not in bases_by_layer:
            raise RuntimeError(f"Vector layer {sv.layer} missing from bases_by_layer. Add it to --basis_layers.")
        Q_shared = bases_by_layer[sv.layer]["Q_shared"]
        Q_pca = bases_by_layer[sv.layer]["Q_pca"]
        Q_pca_prefill = bases_by_layer[sv.layer].get("Q_pca_prefill", None)
        Q_rand = bases_by_layer[sv.layer]["Q_rand"]

        print("\n" + "#" * 80)
        print(f"[Vector] {sv.name} concept={sv.concept} layer={sv.layer} alpha={sv.alpha}")
        print("#" * 80)

        # Build repaired vectors
        v = sv.vec.astype(np.float32, copy=False)
        v_shared = project_out(v, Q_shared, alpha_proj=args.alpha_proj)
        v_rand = project_out(v, Q_rand, alpha_proj=args.alpha_proj)
        v_pca = project_out(v, Q_pca, alpha_proj=args.alpha_proj)
        v_pca_prefill = None
        if bool(args.include_pca_prefill):
            if Q_pca_prefill is None:
                raise RuntimeError(f"include_pca_prefill=1 but Q_pca_prefill is missing for layer {sv.layer}.")
            v_pca_prefill = project_out(v, Q_pca_prefill, alpha_proj=args.alpha_proj)

        # Shrinkage: norm-match shared by default
        norm_v = float(np.linalg.norm(v) + 1e-12)
        norm_shared = float(np.linalg.norm(v_shared) + 1e-12)
        gamma = norm_shared / norm_v
        v_shrink = (gamma * v).astype(np.float32, copy=False)

        # Optionally norm-match rand/pca too (energy-budget control)
        if norm_match:
            def _scale_to(x: np.ndarray, target: float) -> np.ndarray:
                nx = float(np.linalg.norm(x) + 1e-12)
                return (x * (target / nx)).astype(np.float32, copy=False)
            v_rand = _scale_to(v_rand, norm_shared)
            v_pca = _scale_to(v_pca, norm_shared)
            if v_pca_prefill is not None:
                v_pca_prefill = _scale_to(v_pca_prefill, norm_shared)

        repaired = {
            "orig": v,
            "shared": v_shared,
            "rand": v_rand,
            "pca": v_pca,
            "shrink": v_shrink,
        }
        if v_pca_prefill is not None:
            repaired["pca_prefill"] = v_pca_prefill

        # Evaluate per template seed
        per_method_template_delta: Dict[str, List[float]] = {m: [] for m in methods}
        per_method_template_acc: Dict[str, List[float]] = {m: [] for m in methods}
        per_method_task_template_delta: Dict[str, Dict[str, List[float]]] = {
            m: {t: [] for t in tasks_eval} for m in methods
        }

        for tseed in template_seeds:
            eval_by = eval_sets[tseed]
            base_by_task = base_acc[tseed]

            for m in methods:
                deltas = []
                accs = []
                for t in tasks_eval:
                    acc_m = eval_decode_steering(
                        model=model, tokenizer=tokenizer, examples=eval_by[t],
                        decoding=args.decoding, max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                        device=args.device, batch_size=args.batch_size, max_prompt_len=args.max_prompt_len,
                        reasoning_token_threshold=args.reasoning_tokens,
                        v_np=repaired[m], alpha=sv.alpha, layer_idx=sv.layer, staged=staged,
                        sample_seed=(args.sample_seed if args.decoding == "sample" else None),
                    )
                    delta = float(acc_m - base_by_task[t])
                    deltas.append(delta)
                    accs.append(acc_m)
                    per_method_task_template_delta[m][t].append(delta)

                per_method_template_delta[m].append(float(np.mean(deltas)) if deltas else float("nan"))
                per_method_template_acc[m].append(float(np.mean(accs)) if accs else float("nan"))

            name_str = "/".join(methods)
            val_str = " / ".join(f"{per_method_template_delta[m][-1]:+.3f}" for m in methods)
            print(f"[template_seed={tseed}] mean-delta({name_str}) = {val_str}")

        summary = {}
        for m in methods:
            arr = np.array(per_method_template_delta[m], dtype=np.float64)
            arr_acc = np.array(per_method_template_acc[m], dtype=np.float64)
            ddof = 1 if arr.size > 1 else 0
            summary[m] = {
                "mean_delta": float(np.mean(arr)),
                "worst_delta": float(np.min(arr)),
                "std_delta": float(np.std(arr, ddof=ddof)),
                "var_delta": float(np.var(arr, ddof=ddof)),
                "range_delta": float(np.max(arr) - np.min(arr)),
                "mean_acc": float(np.mean(arr_acc)),
                "worst_acc": float(np.min(arr_acc)),
                "std_acc": float(np.std(arr_acc, ddof=ddof)),
                "range_acc": float(np.max(arr_acc) - np.min(arr_acc)),
            }

        results["vectors"][sv.name] = {
            "concept": sv.concept,
            "layer": sv.layer,
            "alpha": sv.alpha,
            "template_tag": sv.template_tag,
            "norms": {
                "norm_v": norm_v,
                "norm_shared": norm_shared,
                "gamma_shrink": gamma,
            },
            "template_seeds": template_seeds,
            "per_method_template_delta": per_method_template_delta,
            "per_method_template_acc": per_method_template_acc,
            "per_method_task_template_delta": per_method_task_template_delta,
            "summary": summary,
        }

        print("\n[Summary: template-robust deltas vs baseline (mean / worst / std / range)]")
        for m in methods:
            s = summary[m]
            print(f"  {m:6s}  mean={s['mean_delta']:+.3f}  worst={s['worst_delta']:+.3f}  std={s['std_delta']:.3f}  range={s['range_delta']:.3f}")

    # -------------------
    # Aggregate summary across vectors
    # -------------------
    if results["vectors"]:
        agg: Dict[str, Any] = {"methods": {}, "wins": {}}
        for m in methods:
            mean_deltas = [results["vectors"][vn]["summary"][m]["mean_delta"] for vn in results["vectors"].keys()]
            worst_deltas = [results["vectors"][vn]["summary"][m]["worst_delta"] for vn in results["vectors"].keys()]
            std_deltas = [results["vectors"][vn]["summary"][m]["std_delta"] for vn in results["vectors"].keys()]
            agg["methods"][m] = {
                "n_vectors": int(len(mean_deltas)),
                "mean_of_mean_delta": float(np.mean(mean_deltas)),
                "mean_of_worst_delta": float(np.mean(worst_deltas)),
                "mean_of_std_delta": float(np.mean(std_deltas)),
        }

        # win rates for shared vs each control on worst-case metric
        controls = [c for c in ["rand", "pca", "pca_prefill", "shrink"] if c in methods]
        wins: Dict[str, float] = {}
        for c in controls:
            n = 0
            w = 0
            for vn in results["vectors"].keys():
                s_shared = results["vectors"][vn]["summary"]["shared"]["worst_delta"]
                s_c = results["vectors"][vn]["summary"][c]["worst_delta"]
                if (not isinstance(s_shared, float)) or (not isinstance(s_c, float)):
                    continue
                n += 1
                if s_shared > s_c:
                    w += 1
            wins[f"shared_vs_{c}_worst_delta_winrate"] = (float(w) / float(n)) if n else float("nan")

        agg["wins"] = wins

        # Optional: direct PCA(decode) vs PCA(prefill) comparison
        if "pca" in methods and "pca_prefill" in methods:
            diffs_mean = []
            diffs_worst = []
            diffs_std = []
            n = 0
            w = 0
            for vn in results["vectors"].keys():
                s_pca = results["vectors"][vn]["summary"]["pca"]
                s_pre = results["vectors"][vn]["summary"]["pca_prefill"]
                dm = float(s_pre["mean_delta"] - s_pca["mean_delta"])
                dw = float(s_pre["worst_delta"] - s_pca["worst_delta"])
                ds = float(s_pre["std_delta"] - s_pca["std_delta"])
                diffs_mean.append(dm)
                diffs_worst.append(dw)
                diffs_std.append(ds)
                n += 1
                if dw > 0:
                    w += 1
            agg["comparisons"] = {
                "pca_prefill_minus_pca": {
                    "n_vectors": int(n),
                    "mean_of_mean_delta_diff": float(np.mean(diffs_mean)) if diffs_mean else float("nan"),
                    "mean_of_worst_delta_diff": float(np.mean(diffs_worst)) if diffs_worst else float("nan"),
                    "mean_of_std_delta_diff": float(np.mean(diffs_std)) if diffs_std else float("nan"),
                    "worst_delta_winrate": (float(w) / float(n)) if n else float("nan"),
                }
            }

        results["aggregate"] = agg

        print("\n[Aggregate across vectors]")
        for m in methods:
            s = agg["methods"][m]
            print(f"  {m:6s}  mean(mean_delta)={s['mean_of_mean_delta']:+.3f}  mean(worst_delta)={s['mean_of_worst_delta']:+.3f}  mean(std_delta)={s['mean_of_std_delta']:.3f}")
        for k, v in wins.items():
            print(f"  {k}: {v:.3f}" if isinstance(v, float) and not math.isnan(v) else f"  {k}: {v}")
        if "comparisons" in agg:
            c = agg["comparisons"]["pca_prefill_minus_pca"]
            print(
                "  pca_prefill_minus_pca: "
                f"mean(mean_delta_diff)={c['mean_of_mean_delta_diff']:+.3f}  "
                f"mean(worst_delta_diff)={c['mean_of_worst_delta_diff']:+.3f}  "
                f"mean(std_delta_diff)={c['mean_of_std_delta_diff']:+.3f}  "
                f"worst_delta_winrate={c['worst_delta_winrate']:.3f}"
            )

    out_path = os.path.expanduser(args.out_json)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[Saved] {out_path}")


if __name__ == "__main__":
    main()
