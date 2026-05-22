# -*- coding: utf-8 -*-
"""
exp_pca_mismatch.py

Compare PCA bases estimated from:
  - Prefill distribution: last-token hidden states from the *prompt prefill* forward (seq_len > 1)
  - Decode distribution: last-token hidden states from *KV-cached decode* forwards (seq_len == 1)

This mirrors the "prefill/decode mismatch" story used in the H3-grid experiments, but focuses
specifically on PCA (not sharedness selection or intervention outcomes).

For each layer and each requested k, the script reports:
  - principal angles between the prefill-PCA and decode-PCA subspaces
  - subspace similarity (cosine singular values of Qp^T Qd)
  - cross-distribution variance explained (e.g., how much decode variance is captured by prefill PCs)

Usage (quick sanity, small k):
  python downstream/prefill_decode_mismatch/exp_pca_mismatch.py \
    --model meta-llama/Llama-2-7b-chat-hf --device cuda \
    --layers 28 --tasks_subspace commonsenseqa,arc_challenge,openbookqa,qasc,logiqa \
    --n_subspace 64 --template_seed 1234 \
    --calib_decode_max_new_tokens 64 \
    --ks 32,64,128 \
    --out_json outputs/prefill_decode_mismatch/pca_prefill_decode_mismatch_layer28.json

"""

from __future__ import annotations

import os
import sys
import re
import json
import math
import argparse
import hashlib
from typing import Any, Dict, List, Optional, Tuple, DefaultDict
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# -----------------------------
# Local imports (repo layout)
# -----------------------------
# This script lives in `downstream/prefill_decode_mismatch/`; public releases
# keep benchmark_dataloaders.py with the experiment/downstream bundles.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _candidate in [
    os.path.join(THIS_DIR, "..", "..", "src"),
    os.path.join(THIS_DIR, "..", "brittleness"),
    os.path.join(THIS_DIR, "..", "patch_back"),
    os.path.join(THIS_DIR, "..", "..", "experiments", "02_decode_ablation"),
]:
    _candidate = os.path.normpath(_candidate)
    if os.path.isfile(os.path.join(_candidate, "benchmark_dataloaders.py")) and _candidate not in sys.path:
        sys.path.append(_candidate)


# -----------------------------
# Repro / stable seed
# -----------------------------
def stable_int_seed_fallback(*items: Any) -> int:
    s = "|".join(str(x) for x in items)
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


try:
    from benchmark_dataloaders import (
        load_selected_tasks,
        stable_int_seed as stable_int_seed_project,
    )
    stable_int_seed = stable_int_seed_project
except Exception as e:
    load_selected_tasks = None  # type: ignore
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
# Prompt rendering (chat template safe)
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
# Sampling utils (optional)
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
# Collect prefill + decode last-token states
# -----------------------------
class PhaseLastTokenActivationCollector:
    """
    One collector that can capture either:
      - phase="prefill": capture last-token hidden states from prefill forward (seq_len>1)
      - phase="decode":  capture last-token hidden states from decode forwards (seq_len==1)

    storage_prefill[task][layer] -> list of np arrays [B', D]
    storage_decode[task][layer]  -> list of np arrays [B', D]
    """
    def __init__(self, layer_indices: List[int]):
        self.layer_indices = list(layer_indices)
        self._cur_task: Optional[str] = None
        self.phase: Optional[str] = None  # None | "prefill" | "decode"
        self.active_mask: Optional[torch.Tensor] = None
        self.storage_prefill: DefaultDict[str, DefaultDict[int, List[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
        self.storage_decode: DefaultDict[str, DefaultDict[int, List[np.ndarray]]] = defaultdict(lambda: defaultdict(list))

    def set_current_task(self, task_name: str) -> None:
        self._cur_task = task_name

    def set_phase(self, phase: Optional[str], active_mask: Optional[torch.Tensor] = None) -> None:
        if phase is not None:
            phase = str(phase)
            if phase not in ["prefill", "decode"]:
                raise ValueError(f"Unknown phase={phase}")
        self.phase = phase
        self.active_mask = active_mask

    def make_hook(self, layer_idx: int):
        def _hook(module, inputs, output):
            if self.phase is None or self._cur_task is None:
                return output
            hs = output[0] if isinstance(output, tuple) else output
            if not isinstance(hs, torch.Tensor) or hs.ndim != 3:
                return output

            seq_len = int(hs.shape[1])
            if self.phase == "prefill":
                if seq_len <= 1:
                    return output
            else:  # decode
                if seq_len != 1:
                    return output

            x = hs[:, -1, :]  # [B, D]
            if self.active_mask is not None:
                m = self.active_mask.bool()
                if m.numel() == x.shape[0]:
                    x = x[m]
            if x.numel() == 0:
                return output

            arr = x.detach().float().cpu().numpy()
            if self.phase == "prefill":
                self.storage_prefill[self._cur_task][layer_idx].append(arr)
            else:
                self.storage_decode[self._cur_task][layer_idx].append(arr)
            return output

        return _hook

    def get_task_activations(self, task: str, layer_idx: int, *, phase: str) -> Optional[np.ndarray]:
        if phase == "prefill":
            chunks = self.storage_prefill.get(task, {}).get(layer_idx, [])
        elif phase == "decode":
            chunks = self.storage_decode.get(task, {}).get(layer_idx, [])
        else:
            raise ValueError(f"Unknown phase={phase}")
        if not chunks:
            return None
        return np.concatenate(chunks, axis=0)


@torch.no_grad()
def collect_prefill_and_decode_last_token_states(
    model,
    tokenizer,
    prompts_by_task: Dict[str, List[str]],
    *,
    layer_indices: List[int],
    batch_size: int,
    max_prompt_len: int,
    calib_decode_max_new_tokens: int,
    decoding: str,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
) -> Tuple[Dict[str, Dict[int, np.ndarray]], Dict[str, Dict[int, np.ndarray]]]:
    """
    Returns:
      - prefill_states_by_task[layer] = np.ndarray [N, D]
      - decode_states_by_task[layer]  = np.ndarray [M, D]
    """
    assert decoding in ["greedy", "sample"]
    device = next(model.parameters()).device
    model.eval()

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos = tokenizer.eos_token_id

    layers = get_model_layers(model)
    for li in layer_indices:
        if li < 0 or li >= len(layers):
            raise ValueError(f"layer_idx out of range: {li} (n_layers={len(layers)})")

    collector = PhaseLastTokenActivationCollector(layer_indices)
    handles = [layers[li].register_forward_hook(collector.make_hook(li)) for li in layer_indices]

    # deterministic sampling (if enabled)
    if decoding == "sample":
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    try:
        for task, prompts in prompts_by_task.items():
            collector.set_current_task(task)

            for i in tqdm(range(0, len(prompts), batch_size), desc=f"Collect({task})"):
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

                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]
                B, _T0 = input_ids.shape
                unfinished = torch.ones(B, dtype=torch.bool, device=device)

                # Prefill forward (capture prefill states)
                collector.set_phase("prefill", None)
                out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
                collector.set_phase(None, None)

                logits = out.logits[:, -1, :]
                past = out.past_key_values

                # KV-cached decode loop (capture decode states)
                for _ in range(calib_decode_max_new_tokens):
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

                    collector.set_phase("decode", unfinished)
                    out = model(
                        input_ids=next_token,
                        attention_mask=attention_mask,
                        use_cache=True,
                        past_key_values=past,
                    )
                    collector.set_phase(None, None)
                    logits = out.logits[:, -1, :]
                    past = out.past_key_values
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass
        collector.set_phase(None, None)

    prefill_by_task: Dict[str, Dict[int, np.ndarray]] = {}
    decode_by_task: Dict[str, Dict[int, np.ndarray]] = {}
    for task in prompts_by_task.keys():
        prefill_by_task[task] = {}
        decode_by_task[task] = {}
        for li in layer_indices:
            Xp = collector.get_task_activations(task, li, phase="prefill")
            Xd = collector.get_task_activations(task, li, phase="decode")
            if Xp is not None:
                prefill_by_task[task][li] = Xp.astype(np.float32, copy=False)
            if Xd is not None:
                decode_by_task[task][li] = Xd.astype(np.float32, copy=False)
    return prefill_by_task, decode_by_task


# -----------------------------
# PCA + metrics
# -----------------------------
def _subsample_rows_np(x: np.ndarray, n_max: int, seed: int) -> np.ndarray:
    if n_max is None or n_max <= 0 or x.shape[0] <= n_max:
        return x
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.shape[0], size=n_max, replace=False)
    return x[idx]


def pool_center_rows(
    mats_by_task: Dict[str, np.ndarray],
    *,
    pca_max_rows: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    X_all = np.concatenate(list(mats_by_task.values()), axis=0)
    if pca_max_rows > 0 and X_all.shape[0] > pca_max_rows:
        X_all = _subsample_rows_np(X_all, pca_max_rows, seed=seed)
    mean = X_all.mean(axis=0, keepdims=True).astype(np.float32, copy=False)
    Xc = (X_all - mean).astype(np.float32, copy=False)
    return Xc, mean.reshape(-1).astype(np.float32, copy=False)


def pca_basis_lowrank(Xc: np.ndarray, *, k: int, seed: int, max_dim: int = 4096) -> np.ndarray:
    """
    Return PCA basis Q [D, k] using torch.pca_lowrank on CPU.
    """
    if Xc.ndim != 2:
        raise ValueError("Xc must be 2D")
    n, d = Xc.shape
    if n <= 1:
        raise RuntimeError("Not enough rows for PCA.")
    k = int(k)
    q = min(int(max_dim), int(d), int(n - 1), int(k))
    if q <= 0:
        raise RuntimeError("q<=0 in PCA.")

    torch.manual_seed(int(seed))
    X_t = torch.from_numpy(Xc)  # float32 on CPU
    _U, _S, V = torch.pca_lowrank(X_t, q=q, center=False)
    Q = V[:, :q].contiguous().cpu().numpy().astype(np.float32, copy=False)
    if Q.shape[1] < k:
        # Caller asked for larger k than possible; return what we have.
        return Q
    return Q[:, :k]


def principal_angles_deg(Qa: np.ndarray, Qb: np.ndarray) -> Dict[str, float]:
    """
    Qa/Qb: [D,k] orthonormal columns (or approximately).
    Return summary stats over k principal angles in degrees.
    """
    if Qa.size == 0 or Qb.size == 0:
        return {"mean": float("nan"), "p50": float("nan"), "p95": float("nan"), "min": float("nan"), "max": float("nan")}
    M = Qa.T @ Qb
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    theta = np.degrees(np.arccos(s))
    return {
        "mean": float(np.mean(theta)),
        "p50": float(np.percentile(theta, 50)),
        "p95": float(np.percentile(theta, 95)),
        "min": float(np.min(theta)),
        "max": float(np.max(theta)),
    }


def subspace_cos_singulars(Qa: np.ndarray, Qb: np.ndarray) -> Dict[str, float]:
    if Qa.size == 0 or Qb.size == 0:
        return {"mean": float("nan"), "min": float("nan"), "max": float("nan")}
    M = Qa.T @ Qb
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    return {"mean": float(np.mean(s)), "min": float(np.min(s)), "max": float(np.max(s))}


def explained_variance_ratio(
    Xc: np.ndarray,
    Q: np.ndarray,
    *,
    chunk_rows: int = 8192,
) -> float:
    """
    Return ratio in [0,1] (may exceed slightly due to numeric error):
      E[||Proj_Q(x)||^2] / E[||x||^2] for centered rows x.
    """
    if Xc.size == 0 or Q.size == 0:
        return float("nan")
    n = int(Xc.shape[0])
    tot = 0.0
    proj = 0.0
    for i in range(0, n, chunk_rows):
        chunk = Xc[i:i + chunk_rows].astype(np.float32, copy=False)
        tot += float(np.sum(chunk * chunk, dtype=np.float64))
        z = chunk @ Q
        proj += float(np.sum(z * z, dtype=np.float64))
    if n <= 0:
        return float("nan")
    tot_per = tot / float(n)
    proj_per = proj / float(n)
    if tot_per <= 1e-30:
        return float("nan")
    return float(proj_per / tot_per)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model_dtype", type=str, default="fp32", choices=["fp32", "fp16"])

    ap.add_argument("--layers", type=str, default="28", help="Comma-separated layer indices, e.g. '10,28'.")
    ap.add_argument("--filter_regex", type=str, default="", help="Optional regex to filter tasks.")

    # Calibration tasks/data
    ap.add_argument("--tasks_subspace", type=str, default="gsm8k,commonsenseqa,strategyqa,aqua,arc_challenge,openbookqa,qasc,logiqa")
    ap.add_argument("--n_subspace", type=int, default=128)
    ap.add_argument("--template_seed", type=int, default=1234)
    ap.add_argument("--template_randomization", type=int, default=1, choices=[0, 1])
    ap.add_argument("--shuffle_choices", type=int, default=1, choices=[0, 1])
    ap.add_argument("--answer_prefix", type=str, default="\nFinal answer:")
    ap.add_argument("--add_answer_prefix", type=int, default=1, choices=[0, 1])

    # Decode collection
    ap.add_argument("--calib_decode_max_new_tokens", type=int, default=128)
    ap.add_argument("--decoding", type=str, default="greedy", choices=["greedy", "sample"])
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=0)

    # PCA
    ap.add_argument("--ks", type=str, default="32,64,128,256", help="Comma-separated k values to compare.")
    ap.add_argument("--pca_max_rows", type=int, default=200000, help="Max pooled rows used per phase (0=no limit).")
    ap.add_argument("--per_task_max_states", type=int, default=20000, help="Max rows per task per phase (0=no limit).")
    ap.add_argument("--pca_max_dim", type=int, default=4096, help="Max q passed to torch.pca_lowrank.")

    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_prompt_len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_json", type=str, default="pca_prefill_decode_mismatch.json")

    args = ap.parse_args()
    set_global_seed(args.seed)

    if load_selected_tasks is None:
        raise RuntimeError(
            "benchmark_dataloaders is required for this script. "
            f"Import error was: {_IMPORT_ERR}"
        )

    layers = _dedup_keep_order(_parse_csv_ints(args.layers))
    if not layers:
        raise ValueError("Empty --layers")

    tasks = [t.strip() for t in args.tasks_subspace.split(",") if t.strip()]
    if args.filter_regex:
        pat = re.compile(args.filter_regex)
        tasks = [t for t in tasks if pat.search(t)]
    if not tasks:
        raise ValueError("Empty --tasks_subspace after filtering.")

    ks = _dedup_keep_order(_parse_csv_ints(args.ks))
    ks = [k for k in ks if k > 0]
    if not ks:
        raise ValueError("Empty --ks")
    max_k_req = int(max(ks))

    # Load prompts (subspace prompts only)
    print("[Data] Loading prompts for basis estimation ...")
    sub_by, _eval_by_dummy, _meta = load_selected_tasks(
        tasks=tasks,
        n_subspace=max(1, int(args.n_subspace)),
        n_eval=1,  # (compat) loader may not accept 0
        seed=int(args.seed),
        template_seed=int(args.template_seed),
        template_randomization=bool(args.template_randomization),
        shuffle_choices=bool(args.shuffle_choices),
        add_answer_prefix=bool(args.add_answer_prefix),
        answer_prefix=str(args.answer_prefix),
    )
    prompts_by_task = {t: [ex.prompt for ex in sub_by[t]] for t in tasks if t in sub_by}
    if not prompts_by_task:
        raise RuntimeError("No prompts loaded (check tasks / dataloader).")

    model, tokenizer = load_model_and_tokenizer(args.model, args.device, args.model_dtype)
    hidden_dim = infer_hidden_dim(model)
    if hidden_dim is None:
        print("[Warn] Could not infer hidden_dim.")
    else:
        print(f"[Model] hidden_dim={hidden_dim}")

    print(f"[Collect] phases=prefill+decode  decoding={args.decoding}  decode_steps={int(args.calib_decode_max_new_tokens)}")
    prefill_by_task, decode_by_task = collect_prefill_and_decode_last_token_states(
        model=model,
        tokenizer=tokenizer,
        prompts_by_task=prompts_by_task,
        layer_indices=layers,
        batch_size=int(args.batch_size),
        max_prompt_len=int(args.max_prompt_len),
        calib_decode_max_new_tokens=int(args.calib_decode_max_new_tokens),
        decoding=str(args.decoding),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
        seed=stable_int_seed(args.seed, "decode_sampling"),
    )

    results: Dict[str, Any] = {
        "config": vars(args),
        "layers": layers,
        "tasks_used": list(prompts_by_task.keys()),
        "by_layer": {},
    }

    per_task_max_states = int(args.per_task_max_states)
    pca_max_rows = int(args.pca_max_rows)
    pca_max_dim = int(args.pca_max_dim)

    for layer in layers:
        print("\n" + "=" * 80)
        print(f"[Layer] {layer}")

        # Build per-task matrices for each phase at this layer
        mats_pre: Dict[str, np.ndarray] = {}
        mats_dec: Dict[str, np.ndarray] = {}

        for task in prompts_by_task.keys():
            Xp = prefill_by_task.get(task, {}).get(layer, None)
            Xd = decode_by_task.get(task, {}).get(layer, None)
            if Xp is not None and Xp.shape[0] > 0:
                Xp = _subsample_rows_np(Xp, per_task_max_states, seed=stable_int_seed(args.seed, "prefill", task, layer))
                mats_pre[task] = Xp
            if Xd is not None and Xd.shape[0] > 0:
                Xd = _subsample_rows_np(Xd, per_task_max_states, seed=stable_int_seed(args.seed, "decode", task, layer))
                mats_dec[task] = Xd

        if not mats_pre:
            raise RuntimeError(f"No prefill states collected at layer {layer}.")
        if not mats_dec:
            raise RuntimeError(f"No decode states collected at layer {layer}.")

        Xp_c, mu_p = pool_center_rows(mats_pre, pca_max_rows=pca_max_rows, seed=stable_int_seed(args.seed, "pool_prefill", layer))
        Xd_c, mu_d = pool_center_rows(mats_dec, pca_max_rows=pca_max_rows, seed=stable_int_seed(args.seed, "pool_decode", layer))

        n_pre, d_pre = Xp_c.shape
        n_dec, d_dec = Xd_c.shape
        if d_pre != d_dec:
            raise ValueError(f"Hidden dim mismatch: prefill d={d_pre} vs decode d={d_dec}")

        max_k_possible = min(max_k_req, d_pre, n_pre - 1, n_dec - 1)
        if max_k_possible <= 0:
            raise RuntimeError(f"Not enough rows for PCA at layer {layer}: n_pre={n_pre} n_dec={n_dec}")

        if max_k_possible < max_k_req:
            print(f"[Warn] requested max_k={max_k_req} but only possible k={max_k_possible} (n_pre={n_pre}, n_dec={n_dec}, d={d_pre})")

        # Compute bases (up to max_k_possible)
        Qp = pca_basis_lowrank(Xp_c, k=max_k_possible, seed=stable_int_seed(args.seed, "pca_prefill", layer), max_dim=pca_max_dim)
        Qd = pca_basis_lowrank(Xd_c, k=max_k_possible, seed=stable_int_seed(args.seed, "pca_decode", layer), max_dim=pca_max_dim)

        layer_res: Dict[str, Any] = {
            "n_rows_prefill": int(n_pre),
            "n_rows_decode": int(n_dec),
            "d": int(d_pre),
            "k_max": int(max_k_possible),
            "angles_deg": {},
            "cos_singulars": {},
            "explained_var_ratio": {},
        }

        for k in ks:
            if k > max_k_possible:
                continue
            Qa = Qp[:, :k]
            Qb = Qd[:, :k]
            ang = principal_angles_deg(Qa, Qb)
            cos = subspace_cos_singulars(Qa, Qb)

            ev = {
                "decode_by_decode": explained_variance_ratio(Xd_c, Qb),
                "decode_by_prefill": explained_variance_ratio(Xd_c, Qa),
                "prefill_by_prefill": explained_variance_ratio(Xp_c, Qa),
                "prefill_by_decode": explained_variance_ratio(Xp_c, Qb),
            }

            layer_res["angles_deg"][str(k)] = ang
            layer_res["cos_singulars"][str(k)] = cos
            layer_res["explained_var_ratio"][str(k)] = ev

            print(
                f"[k={k:4d}] angles_deg(mean/p50/p95)={ang['mean']:.2f}/{ang['p50']:.2f}/{ang['p95']:.2f}  "
                f"cos(mean/min)={cos['mean']:.3f}/{cos['min']:.3f}  "
                f"EV_decode(prefill/dec)={ev['decode_by_prefill']:.3f}/{ev['decode_by_decode']:.3f}"
            )

        results["by_layer"][str(layer)] = layer_res

    out_path = os.path.expanduser(args.out_json)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[Saved] {out_path}")


if __name__ == "__main__":
    main()
