# -*- coding: utf-8 -*-
"""prefill_vs_decode_alignment_experiment_v3.py

Prefill-vs-Decode shared-basis *alignment* experiment (estimation distribution mismatch).

This is v3 (scheme-2): add a **decode-warmup** option for forced-choice evaluation.

Motivation (scheme-2)
--------------------
The v1/v2 forced-choice protocol scores candidates immediately at the prompt boundary.
For many MC tasks (A/B/C/D/E, Yes/No), answers are 1 token, so the decision is made
almost entirely from the *prompt-boundary* next-token state.

However, the estimator–intervention mismatch we care about is primarily between:
  - D_prefill: prompt-time (seq_len>1) last-token states
  - D_decode: *later* decode states (seq_len==1) after several cached decode steps,
              especially on self-generated tokens.

To probe deeper into D_decode (where misalignment is largest), v3 optionally:
  1) Generates W warmup tokens **once** under baseline (no intervention),
  2) Then *teacher-forces* those same warmup tokens for **all** conditions,
  3) Scores the multiple-choice candidates *after* the warmup context.

This keeps the evaluation "forced-choice" (no free-form generation during scoring)
while moving the decision point away from the prompt boundary.

Key new CLI args:
  --fc_warmup_tokens W         (default 0)
  --fc_warmup_decoding {greedy,sample}   (default greedy)
  --fc_warmup_ban_eos 0/1      (default 1)
  --fc_warmup_seed SEED        (default 123)

All other features from v2 remain:
  - decode vs prefill basis estimation
  - dimension-matched k_match tables (decode_shared_k vs prefill_shared_k)
  - optional state-count matching for basis estimation
  - random control basis
  - outputs: JSON + TXT + Markdown + LaTeX tables

Run (example):
  python prefill_vs_decode_alignment_experiment_v3_fixed.py \
  --model meta-llama/Llama-2-7b-chat-hf --device cuda --dtype fp32 \
  --layer 10 --n_prompts 128 --eval_n 256 \
  --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
  --pca_var 0.95 --tau 0.001 --m_shared all \
  --do_generation 0 \
  --fc_warmup_tokens 128 \
  --fc_add_answer_prefix 1 \
  --fc_answer_prefix $'\nFinal answer:'
"""

import os
import sys
import re
import json
import math
import random
import argparse
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
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


# -----------------------------
# Repro / utils
# -----------------------------
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


def summarize_paired(baseline_correct: np.ndarray, treat_correct: np.ndarray, bootstrap_iters: int, perm_iters: int, alpha: float, seed: int) -> Dict[str, Any]:
    md, lo, hi = paired_bootstrap_ci_diff(baseline_correct, treat_correct, iters=bootstrap_iters, alpha=alpha, seed=seed + 123)
    p = signflip_permutation_test(baseline_correct, treat_correct, iters=perm_iters, seed=seed + 456)
    return {"mean_diff": md, "ci_low": lo, "ci_high": hi, "p_value": p}


def fmt_acc(acc: float, lo: float, hi: float) -> str:
    return f"{acc*100:.1f} [{lo*100:.1f}, {hi*100:.1f}]"


# -----------------------------
# Dataset / prompts
# -----------------------------
@dataclass
class Example:
    dataset: str
    ex_id: str
    prompt: str
    gold: str


def safe_upper(x: Any) -> str:
    return str(x).strip().upper()


def normalize_number_str(s: str) -> Optional[str]:
    if s is None:
        return None
    s = str(s).strip().replace(",", "")
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    if not m:
        return None
    num = m.group(0)
    if "." in num:
        num = num.rstrip("0").rstrip(".")
    return num


def parse_gsm8k_gold(answer_field: str) -> str:
    if answer_field is None:
        return ""
    txt = str(answer_field).replace(",", "")
    m = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", txt)
    if m:
        return normalize_number_str(m.group(1)) or ""
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", txt)
    return normalize_number_str(nums[-1]) if nums else ""


def build_prompt_gsm8k(question: str) -> str:
    return (
        f"Question: {question}\n"
        "Let's think step by step.\n"
        'At the end, write exactly one line in the format: "Final answer: <number>".\n'
    )


def build_prompt_commonsenseqa(question: str, choices: Dict[str, List[str]]) -> str:
    labels = choices["label"]
    texts = choices["text"]
    lines = [f"{lab}) {txt}" for lab, txt in zip(labels, texts)]
    return (
        f"Question: {question}\n"
        "Choices:\n" + "\n".join(lines) + "\n"
        "Reason step by step.\n"
        'At the end, write exactly one line in the format: "Final answer: <A/B/C/D/E>".\n'
    )


def build_prompt_strategyqa(question: str) -> str:
    return (
        f"Question: {question}\n"
        "Please reason step by step.\n"
        'At the end, write exactly one line in the format: "Final answer: Yes" or "Final answer: No".\n'
    )


def build_prompt_aqua(question: str, options: List[str]) -> str:
    labels = ["A", "B", "C", "D", "E"]
    lines = []
    for i, opt in enumerate(options[:5]):
        lab = labels[i]
        opt_clean = re.sub(r"^[A-E]\)?\s*[:\-]?\s*", "", str(opt).strip(), flags=re.IGNORECASE)
        lines.append(f"{lab}) {opt_clean}")
    return (
        f"Question: {question}\n"
        "Choices:\n" + "\n".join(lines) + "\n"
        "Please reason step by step.\n"
        'At the end, write exactly one line in the format: "Final answer: <A/B/C/D/E>".\n'
    )


def sample_hf_split(ds_split, n: int, seed: int):
    n = min(n, len(ds_split))
    if n <= 0:
        return ds_split.select([])
    return ds_split.shuffle(seed=seed).select(range(n))


def load_gsm8k_examples(n_prompts: int, eval_n: int, seed: int):
    ds = load_dataset("gsm8k", "main")
    sub_split = "train" if "train" in ds else list(ds.keys())[0]
    eval_split = "test" if "test" in ds else ("validation" if "validation" in ds else sub_split)
    sub_rows = sample_hf_split(ds[sub_split], n_prompts, seed + 1)
    eval_rows = sample_hf_split(ds[eval_split], eval_n, seed + 2)
    sub_exs, eval_exs = [], []
    for i, ex in enumerate(sub_rows):
        sub_exs.append(Example("gsm8k", f"gsm8k-{sub_split}-{i}", build_prompt_gsm8k(ex["question"]), parse_gsm8k_gold(ex["answer"])))
    for i, ex in enumerate(eval_rows):
        eval_exs.append(Example("gsm8k", f"gsm8k-{eval_split}-{i}", build_prompt_gsm8k(ex["question"]), parse_gsm8k_gold(ex["answer"])))
    meta = {"hf_id": "gsm8k/main", "subspace_split": sub_split, "eval_split": eval_split}
    return sub_exs, eval_exs, meta


def load_commonsenseqa_examples(n_prompts: int, eval_n: int, seed: int):
    ds = load_dataset("commonsense_qa")
    sub_split = "train" if "train" in ds else list(ds.keys())[0]
    eval_split = "validation" if "validation" in ds else ("test" if "test" in ds else sub_split)
    sub_rows = sample_hf_split(ds[sub_split], n_prompts, seed + 11)
    eval_rows = sample_hf_split(ds[eval_split], eval_n, seed + 12)
    sub_exs, eval_exs = [], []
    for i, ex in enumerate(sub_rows):
        sub_exs.append(Example("commonsenseqa", f"csqa-{sub_split}-{i}", build_prompt_commonsenseqa(ex["question"], ex["choices"]), safe_upper(ex["answerKey"])))
    for i, ex in enumerate(eval_rows):
        eval_exs.append(Example("commonsenseqa", f"csqa-{eval_split}-{i}", build_prompt_commonsenseqa(ex["question"], ex["choices"]), safe_upper(ex["answerKey"])))
    meta = {"hf_id": "commonsense_qa", "subspace_split": sub_split, "eval_split": eval_split}
    return sub_exs, eval_exs, meta


def load_aqua_examples(n_prompts: int, eval_n: int, seed: int):
    ds = load_dataset("aqua_rat")
    sub_split = "train" if "train" in ds else list(ds.keys())[0]
    eval_split = "test" if "test" in ds else ("validation" if "validation" in ds else sub_split)
    sub_rows = sample_hf_split(ds[sub_split], n_prompts, seed + 21)
    eval_rows = sample_hf_split(ds[eval_split], eval_n, seed + 22)
    def gold(ex: dict) -> str:
        if "correct" in ex:
            return safe_upper(ex["correct"])
        if "answer" in ex:
            return safe_upper(ex["answer"])
        return ""
    sub_exs, eval_exs = [], []
    for i, ex in enumerate(sub_rows):
        sub_exs.append(Example("aqua", f"aqua-{sub_split}-{i}", build_prompt_aqua(ex["question"], ex["options"]), gold(ex)))
    for i, ex in enumerate(eval_rows):
        eval_exs.append(Example("aqua", f"aqua-{eval_split}-{i}", build_prompt_aqua(ex["question"], ex["options"]), gold(ex)))
    meta = {"hf_id": "aqua_rat", "subspace_split": sub_split, "eval_split": eval_split}
    return sub_exs, eval_exs, meta


def load_strategyqa_examples(n_prompts: int, eval_n: int, seed: int):
    hf_id = "ChilleD/StrategyQA"
    ds = load_dataset(hf_id)
    sub_split = "train" if "train" in ds else list(ds.keys())[0]
    eval_split = "test" if "test" in ds else ("validation" if "validation" in ds else sub_split)
    sub_rows = sample_hf_split(ds[sub_split], n_prompts, seed + 31)
    eval_rows = sample_hf_split(ds[eval_split], eval_n, seed + 32)
    def to_yesno(v: Any) -> str:
        if isinstance(v, bool):
            return "YES" if v else "NO"
        if isinstance(v, (int, np.integer)):
            return "YES" if int(v) == 1 else "NO"
        s = str(v).strip().lower()
        if s in ["true", "yes", "1"]:
            return "YES"
        if s in ["false", "no", "0"]:
            return "NO"
        if "yes" in s:
            return "YES"
        if "no" in s:
            return "NO"
        return ""
    sub_exs, eval_exs = [], []
    for i, ex in enumerate(sub_rows):
        sub_exs.append(Example("strategyqa", f"strategyqa-{sub_split}-{i}", build_prompt_strategyqa(ex["question"]), to_yesno(ex["answer"])))
    for i, ex in enumerate(eval_rows):
        eval_exs.append(Example("strategyqa", f"strategyqa-{eval_split}-{i}", build_prompt_strategyqa(ex["question"]), to_yesno(ex["answer"])))
    meta = {"hf_id": hf_id, "subspace_split": sub_split, "eval_split": eval_split}
    return sub_exs, eval_exs, meta


def load_all_examples(n_prompts: int, eval_n: int, seed: int):
    loaders = {
        "gsm8k": load_gsm8k_examples,
        "commonsenseqa": load_commonsenseqa_examples,
        "strategyqa": load_strategyqa_examples,
        "aqua": load_aqua_examples,
    }
    sub_by, eval_by, meta_by = {}, {}, {}
    for name, fn in loaders.items():
        print(f"[Data] Loading {name} (n_prompts={n_prompts}, eval_n={eval_n}) ...")
        sub_exs, eval_exs, meta = fn(n_prompts, eval_n, seed)
        sub_by[name] = sub_exs
        eval_by[name] = eval_exs
        meta_by[name] = meta
        print(f"[Data] {name}: subspace={len(sub_exs)} eval={len(eval_exs)} meta={meta}")
    return sub_by, eval_by, meta_by


# -----------------------------
# Orthonormal basis + subspace diagnostics
# -----------------------------
def orthonormalize_np(basis: np.ndarray) -> np.ndarray:
    q, _ = np.linalg.qr(basis.astype(np.float32, copy=False))
    return q.astype(np.float32, copy=False)


def random_orthonormal_basis_np(dim: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((dim, k), dtype=np.float32)
    return orthonormalize_np(a)


def subspace_similarity(Qa: np.ndarray, Qb: np.ndarray) -> Dict[str, float]:
    # principal angle cosines via svd of Qa^T Qb
    M = Qa.T @ Qb
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    return {
        "max_cos": float(np.max(s)) if s.size else float("nan"),
        "mean_cos": float(np.mean(s)) if s.size else float("nan"),
        "min_cos": float(np.min(s)) if s.size else float("nan"),
        "fro_norm": float(np.linalg.norm(M, ord="fro")),
    }


def energy_ratio_stats(states: np.ndarray, Q: np.ndarray) -> Dict[str, float]:
    eps = 1e-12
    H = states.astype(np.float32, copy=False)
    proj = H @ Q
    num = np.sum(proj * proj, axis=1)
    den = np.sum(H * H, axis=1) + eps
    r = num / den
    return {
        "mean": float(np.mean(r)),
        "p50": float(np.percentile(r, 50)),
        "p95": float(np.percentile(r, 95)),
    }


# -----------------------------
# Decode vs prefill activation collectors (same as v2)
# -----------------------------
from collections import defaultdict
from typing import DefaultDict


class DecodeLastTokenCollector:
    """Collect last-token hidden states during decode passes only (seq_len==1)."""

    def __init__(self, layer_indices: List[int]):
        self.layer_indices = list(layer_indices)
        self._cur_task: Optional[str] = None
        self.capture_enabled: bool = False
        self.active_mask: Optional[torch.Tensor] = None
        self.storage: DefaultDict[str, DefaultDict[int, List[np.ndarray]]] = defaultdict(lambda: defaultdict(list))

    def set_current_task(self, task: str) -> None:
        self._cur_task = task

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
            x = hs[:, -1, :]
            if self.active_mask is not None:
                m = self.active_mask.bool()
                if m.numel() == x.shape[0]:
                    x = x[m]
            if x.numel() == 0:
                return output
            self.storage[self._cur_task][layer_idx].append(x.detach().float().cpu().numpy())
            return output

        return _hook

    def get(self, task: str, layer_idx: int) -> Optional[np.ndarray]:
        chunks = self.storage.get(task, {}).get(layer_idx, [])
        if not chunks:
            return None
        return np.concatenate(chunks, axis=0)


class PrefillLastTokenCollector:
    """Collect last-token hidden states during prefill passes only (seq_len>1)."""

    def __init__(self, layer_indices: List[int]):
        self.layer_indices = list(layer_indices)
        self._cur_task: Optional[str] = None
        self.capture_enabled: bool = True
        self.storage: DefaultDict[str, DefaultDict[int, List[np.ndarray]]] = defaultdict(lambda: defaultdict(list))

    def set_current_task(self, task: str) -> None:
        self._cur_task = task

    def make_hook(self, layer_idx: int):
        def _hook(module, inputs, output):
            if (not self.capture_enabled) or (self._cur_task is None):
                return output
            hs = output[0] if isinstance(output, tuple) else output
            if not isinstance(hs, torch.Tensor) or hs.ndim != 3:
                return output
            if hs.shape[1] <= 1:
                return output
            x = hs[:, -1, :]
            if x.numel() == 0:
                return output
            self.storage[self._cur_task][layer_idx].append(x.detach().float().cpu().numpy())
            return output

        return _hook

    def get(self, task: str, layer_idx: int) -> Optional[np.ndarray]:
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


# -----------------------------
# Decode collection loop
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
    """Return next_token [B,1]. If ban_eos, avoid selecting EOS."""
    assert decoding in ["greedy", "sample"]
    if ban_eos:
        logits = logits.clone()
        logits[:, eos_token_id] = float("-inf")

    if decoding == "greedy":
        # If ban_eos, EOS is already -inf.
        next_tok = torch.argmax(logits, dim=-1, keepdim=True)
        return next_tok

    lt = logits / max(temperature, 1e-6)
    lt = top_k_filtering(lt, top_k)
    lt = top_p_filtering(lt, top_p)
    probs = torch.softmax(lt, dim=-1)
    next_tok = torch.multinomial(probs, num_samples=1)
    return next_tok


@torch.no_grad()
def _cache_advanced_prompt_boundary(model, ids: torch.Tensor, attn: torch.Tensor):
    """Compute (past, logits_next) such that the last prompt token is processed with seq_len==1."""
    # ids: [B,T], attn: [B,T]
    if ids.ndim != 2:
        raise ValueError(f"ids must be 2D [B,T], got {ids.shape}")
    B, T = ids.shape
    if T == 0:
        raise ValueError("Empty prompt")
    if T == 1:
        out1 = model(input_ids=ids, attention_mask=attn, use_cache=True)
        return out1.past_key_values, out1.logits[:, -1, :]
    # prefix (seq_len=T-1)
    out0 = model(input_ids=ids[:, :-1], attention_mask=attn[:, :-1], use_cache=True)
    # last token (seq_len=1)
    out1 = model(input_ids=ids[:, -1:], attention_mask=attn, use_cache=True, past_key_values=out0.past_key_values)
    return out1.past_key_values, out1.logits[:, -1, :]


@torch.no_grad()
def collect_decode_states(
    model,
    tok,
    prompts: List[str],
    collector: DecodeLastTokenCollector,
    *,
    batch_size: int,
    max_new_tokens: int,
    max_prompt_len: int,
    decoding: str,
    temperature: float,
    top_p: float,
    top_k: int,
) -> None:
    device = next(model.parameters()).device
    eos = tok.eos_token_id
    model.eval()
    for i in tqdm(range(0, len(prompts), batch_size), desc="CollectDecode"):
        batch = prompts[i : i + batch_size]
        inputs = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_prompt_len).to(device)
        ids = inputs["input_ids"]
        attn = inputs["attention_mask"]
        B = ids.shape[0]

        unfinished = torch.ones(B, dtype=torch.bool, device=device)

        # Prefill (no capture)
        collector.set_capture(False, None)
        past, logits = _cache_advanced_prompt_boundary(model, ids, attn)

        for _ in range(max_new_tokens):
            next_tok = _choose_next_token(
                logits,
                decoding=decoding,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                eos_token_id=eos,
                ban_eos=False,
            )
            next_tok = torch.where(unfinished.unsqueeze(-1), next_tok, torch.full_like(next_tok, eos))
            unfinished = unfinished & (next_tok.squeeze(-1) != eos)
            if not bool(unfinished.any().item()):
                break
            attn = torch.cat([attn, torch.ones((B, 1), device=device, dtype=attn.dtype)], dim=1)

            collector.set_capture(True, unfinished)
            out = model(input_ids=next_tok, attention_mask=attn, use_cache=True, past_key_values=past)
            logits = out.logits[:, -1, :]
            past = out.past_key_values

        collector.set_capture(False, None)


@torch.no_grad()
def collect_prefill_states(
    model,
    tok,
    prompts: List[str],
    collector: PrefillLastTokenCollector,
    *,
    batch_size: int,
    max_prompt_len: int,
) -> None:
    device = next(model.parameters()).device
    model.eval()
    for i in tqdm(range(0, len(prompts), batch_size), desc="CollectPrefill"):
        batch = prompts[i : i + batch_size]
        inputs = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_prompt_len).to(device)
        _ = model(**inputs)


def _balanced_concat(task_to_states: Dict[str, np.ndarray], seed: int) -> Tuple[np.ndarray, int]:
    sizes = {t: v.shape[0] for t, v in task_to_states.items()}
    n_min = min(sizes.values())
    out = []
    for t, X in task_to_states.items():
        rng = np.random.default_rng(stable_int_seed(seed, t, "bal"))
        if X.shape[0] > n_min:
            idx = rng.choice(X.shape[0], size=n_min, replace=False)
            out.append(X[idx])
        else:
            out.append(X)
    return np.concatenate(out, axis=0), n_min


def compute_shared_basis_from_states(
    task_states: Dict[str, np.ndarray],
    *,
    pca_var: float,
    min_dim: int,
    max_dim: int,
    tau: float,
    m_shared: str,
    seed: int,
) -> Tuple[np.ndarray, List[int], Dict[str, Any]]:
    """Compute joint subspace + shared indices from task->states dict (single layer)."""
    X_joint, n_bal = _balanced_concat(task_states, seed)
    tasks = list(task_states.keys())

    # Build dict in the format expected by compute_cross_task_subspace: task -> {layer_idx -> states}
    # We'll use layer key 0 internally.
    task_dict = {t: {0: task_states[t]} for t in tasks}

    joint_subspace, cross_dim, contributions, full_pca_info = compute_cross_task_subspace(
        task_dict,
        variance_threshold=pca_var,
        min_dim=min_dim,
        max_dim=max_dim,
        return_full_pca=True,
    )
    if joint_subspace is None or cross_dim <= 0:
        raise RuntimeError("Failed to compute cross-task subspace")

    # Shared components across all tasks by default
    min_tasks_shared = len(tasks) if m_shared == "all" else int(m_shared)
    shared_idx = find_fully_shared_basis_improved(
        contributions,
        tasks,
        cross_dim,
        min_tasks_shared=min_tasks_shared,
        relative_threshold=tau,
        top_k_components=cross_dim,
    )

    extra = {
        "tasks_used": tasks,
        "n_balanced": int(n_bal),
        "cross_dim": int(cross_dim),
        "task_contributions": contributions,
        "full_pca_info": full_pca_info,
    }
    return joint_subspace.astype(np.float32, copy=False), shared_idx, extra


# -----------------------------
# Intervention hooks (decode-only; same as v2)
# -----------------------------
class HookStats:
    def __init__(self, name: str):
        self.name = name
        self.decode_calls = 0
        self.intervened = 0

    def report(self):
        return {"name": self.name, "decode_calls": int(self.decode_calls), "intervened": int(self.intervened)}


class LastTokenRemovalHook:
    """Remove projection onto Q on decode passes only (seq_len==1)."""

    def __init__(self, Q_np: np.ndarray, alpha: float, stats: HookStats):
        self.alpha = float(alpha)
        self.stats = stats
        self.enabled = True
        self.Q_cpu = torch.tensor(orthonormalize_np(Q_np), dtype=torch.float32)
        self.Q_dev: Optional[torch.Tensor] = None

    def set_enabled(self, flag: bool) -> None:
        self.enabled = bool(flag)

    def _Q(self, device: torch.device) -> torch.Tensor:
        if self.Q_dev is None or self.Q_dev.device != device:
            self.Q_dev = self.Q_cpu.to(device=device)
        return self.Q_dev

    def __call__(self, module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        if not isinstance(hs, torch.Tensor) or hs.ndim != 3:
            return output
        if hs.shape[1] != 1:
            return output

        self.stats.decode_calls += 1
        if not self.enabled:
            return output

        Q = self._Q(hs.device)
        x = hs[:, -1, :].float()
        proj = (x @ Q) @ Q.T
        hs2 = hs.clone()
        hs2[:, -1, :] = (x - self.alpha * proj).to(dtype=hs.dtype)
        self.stats.intervened += 1
        if isinstance(output, tuple):
            return (hs2,) + output[1:]
        return hs2


def register_hooks(model, layer_indices: List[int], basis_np: Optional[np.ndarray], alpha: float, name: str):
    if basis_np is None:
        return [], [], HookStats(name), (lambda flag: None)
    layers, _ = get_model_layers(model)
    stats = HookStats(name)
    handles = []
    hooks = []
    for li in layer_indices:
        hk = LastTokenRemovalHook(basis_np, alpha, stats)
        hooks.append(hk)
        handles.append(layers[li].register_forward_hook(hk))

    def toggle(flag: bool):
        for hk in hooks:
            hk.set_enabled(flag)

    return handles, hooks, stats, toggle


def remove_hooks(handles):
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass


# -----------------------------
# Forced-choice logprob eval (decode-aligned) + v3 warmup
# -----------------------------
def candidate_strings(task: str) -> List[str]:
    if task in ["commonsenseqa", "aqua"]:
        return ["A", "B", "C", "D", "E"]
    if task == "strategyqa":
        return ["Yes", "No"]
    return []


def cand_token_ids(tok, s: str) -> List[int]:
    # Prefer leading-space variant
    ids = tok.encode(" " + s, add_special_tokens=False)
    if not ids:
        ids = tok.encode(s, add_special_tokens=False)
    return ids


@torch.no_grad()
def precompute_fc_warmup_tokens(
    model,
    tok,
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
    """Generate W warmup tokens under baseline (no intervention). Returns [N,W] int64 on CPU."""
    assert warmup_tokens >= 0
    if warmup_tokens == 0:
        return np.zeros((len(prompts), 0), dtype=np.int64)

    device = next(model.parameters()).device
    eos = tok.eos_token_id
    model.eval()

    if decoding == "sample":
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    out_tokens = np.zeros((len(prompts), warmup_tokens), dtype=np.int64)

    for i in tqdm(range(0, len(prompts), batch_size), desc=f"WarmupGen(W={warmup_tokens})"):
        batch = prompts[i : i + batch_size]
        inputs = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_prompt_len).to(device)
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


@torch.no_grad()
def forced_choice_logprob_eval(
    model,
    tok,
    examples: List[Example],
    task: str,
    *,
    layer_indices: List[int],
    basis_np: Optional[np.ndarray],
    alpha: float,
    batch_size: int,
    max_prompt_len: int,
    warmup_token_ids: Optional[np.ndarray],
    answer_prefix: str,
) -> Dict[str, Any]:
    """Forced-choice accuracy by logprob (decode-aligned) with optional warmup teacher-forcing."""

    device = next(model.parameters()).device
    model.eval()
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    prompts = [ex.prompt for ex in examples]
    golds = [ex.gold for ex in examples]
    cands = candidate_strings(task)
    cand_ids_list = [cand_token_ids(tok, s) for s in cands]

    handles, hooks, stats, toggle = register_hooks(
        model,
        layer_indices=layer_indices,
        basis_np=basis_np,
        alpha=alpha,
        name=f"fc_full@{layer_indices[0]}",
    )

    correct = np.zeros(len(prompts), dtype=np.float32)

    try:
        for i in tqdm(range(0, len(prompts), batch_size), desc=f"ForcedChoice({task})"):
            batch_prompts = prompts[i : i + batch_size]
            batch_golds = golds[i : i + batch_size]
            inputs = tok(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_prompt_len).to(device)
            ids = inputs["input_ids"]
            attn = inputs["attention_mask"]
            B = ids.shape[0]

            # Optional warmup tokens: teacher force the same baseline-generated tokens for all conditions.
            warm_ids = None
            W = 0
            if warmup_token_ids is not None:
                warm = warmup_token_ids[i : i + B]
                if warm is not None:
                    warm_ids = torch.tensor(warm, dtype=torch.long, device=device)
                    W = int(warm_ids.shape[1])

            # Prompt boundary in decode mode
            past, logits = _cache_advanced_prompt_boundary(model, ids, attn)

            # Teacher-force warmup tokens (decode steps) BEFORE scoring candidates.
            if warm_ids is not None and W > 0:
                for t in range(W):
                    tok_t = warm_ids[:, t : t + 1]
                    attn = torch.cat([attn, torch.ones((B, 1), device=device, dtype=attn.dtype)], dim=1)
                    out = model(input_ids=tok_t, attention_mask=attn, use_cache=True, past_key_values=past)
                    logits = out.logits[:, -1, :]
                    past = out.past_key_values


            # Optional: move the decision point to the *answer slot*.
            # Without this, scoring single-token candidates (A/B/Yes/No) right after warmup
            # is usually meaningless because the model is still mid-reasoning.
            if answer_prefix:
                prefix_ids = tok.encode(answer_prefix, add_special_tokens=False)
                if len(prefix_ids) > 0:
                    for pid in prefix_ids:
                        inp = torch.full((B, 1), pid, dtype=torch.long, device=device)
                        attn = torch.cat([attn, torch.ones((B, 1), device=device, dtype=attn.dtype)], dim=1)
                        out = model(input_ids=inp, attention_mask=attn, use_cache=True, past_key_values=past)
                        logits = out.logits[:, -1, :]
                        past = out.past_key_values

            # Score candidates
            scores = torch.zeros(B, len(cands), device=device)
            for ci, cand_ids in enumerate(cand_ids_list):
                if len(cand_ids) == 0:
                    scores[:, ci] = float("-inf")
                    continue

                past_c = past
                attn_c = attn
                logits_c = logits

                lp = torch.zeros(B, device=device)
                for ti, tok_id in enumerate(cand_ids):
                    # logprob for this token under current logits
                    logp = torch.log_softmax(logits_c, dim=-1)
                    lp = lp + logp[:, tok_id]

                    # advance cache with teacher-forced token (unless last)
                    if ti < len(cand_ids) - 1:
                        inp = torch.full((B, 1), tok_id, dtype=torch.long, device=device)
                        attn_c = torch.cat([attn_c, torch.ones((B, 1), device=device, dtype=attn_c.dtype)], dim=1)
                        out = model(input_ids=inp, attention_mask=attn_c, use_cache=True, past_key_values=past_c)
                        logits_c = out.logits[:, -1, :]
                        past_c = out.past_key_values

                scores[:, ci] = lp

            pred_idx = torch.argmax(scores, dim=1).detach().cpu().numpy().tolist()
            preds = [cands[j] for j in pred_idx]
            for b, (pred, gold) in enumerate(zip(preds, batch_golds)):
                correct[i + b] = 1.0 if safe_upper(pred) == safe_upper(gold) else 0.0

        return {
            "acc": float(correct.mean()),
            "correct": correct.tolist(),
            "hook_stats": stats.report(),
        }
    finally:
        remove_hooks(handles)


# -----------------------------
# Build dimension-matched bases
# -----------------------------
def _build_shared_basis_from_joint(joint: np.ndarray, shared_idx: List[int], k: int) -> np.ndarray:
    if k <= 0:
        raise ValueError("k must be positive")
    if len(shared_idx) < k:
        raise ValueError(f"Need at least {k} shared components, got {len(shared_idx)}")
    idx = sorted(shared_idx)[:k]
    return orthonormalize_np(joint[:, idx])


# -----------------------------
# Model load
# -----------------------------
def load_model_and_tokenizer(model_name: str, device: str, dtype: str):
    torch_dtype = torch.float32 if dtype == "fp32" else torch.float16
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch_dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = model.to(device)
    model.eval()
    model.config.use_cache = True
    return model, tok


# -----------------------------
# Markdown/LaTeX table helpers
# -----------------------------
def md_table(rows: List[List[str]], header: List[str]) -> str:
    out = []
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "---|" * len(header))
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def latex_table(rows: List[List[str]], header: List[str], caption: str, label: str, colspec: str) -> str:
    # Minimal ICML-friendly table (booktabs assumed by ICML style)
    def esc(s: str) -> str:
        return s.replace("%", "\\%").replace("_", "\\_")
    header_esc = [esc(h) for h in header]
    body = []
    for r in rows:
        body.append(" & ".join(esc(x) for x in r) + " \\")
    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\small\n"
        f"\\begin{{tabular}}{{{colspec}}}\n"
        "\\toprule\n"
        + " & ".join(header_esc)
        + " \\\n+\\midrule\n"
        + "\n".join(body)
        + "\n\\bottomrule\n"
        "\\end{tabular}\n"
        f"\\caption{{{esc(caption)}}}\n"
        f"\\label{{{esc(label)}}}\n"
        "\\end{table}\n"
    )


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--n_prompts", type=int, default=128)
    ap.add_argument("--eval_n", type=int, default=256)
    ap.add_argument("--calib_decode_max_new_tokens", type=int, default=128)
    ap.add_argument("--per_task_max_states", type=int, default=20000)
    ap.add_argument("--pca_var", type=float, default=0.95)
    ap.add_argument("--min_dim", type=int, default=1)
    ap.add_argument("--max_dim", type=int, default=4096)
    ap.add_argument("--tau", type=float, default=0.001)
    ap.add_argument("--m_shared", type=str, default="all")
    ap.add_argument("--alpha_remove", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_prompt_len", type=int, default=512)
    ap.add_argument("--bootstrap_iters", type=int, default=5000)
    ap.add_argument("--perm_iters", type=int, default=10000)
    ap.add_argument("--ci_alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--do_generation", type=int, default=0)
    ap.add_argument("--match_state_count", type=int, default=0)

    # Warmup-forced-choice (v3)
    ap.add_argument("--fc_warmup_tokens", type=int, default=0, help="Teacher-force W baseline-generated decode tokens before scoring candidates (probe deeper decode distribution).")
    ap.add_argument("--fc_warmup_decoding", type=str, default="greedy", choices=["greedy", "sample"], help="How to generate warmup tokens under baseline.")
    ap.add_argument("--fc_warmup_seed", type=int, default=123, help="Seed used when generating warmup tokens (if sampling).")
    ap.add_argument("--fc_warmup_ban_eos", type=int, default=1, help="If 1, ban EOS during warmup token generation.")
    ap.add_argument(
        "--fc_add_answer_prefix",
        type=int,
        default=1,
        help="If 1, teacher-force `--fc_answer_prefix` after warmup and BEFORE scoring candidates (recommended). "
             "Set 0 to reproduce the old (often chance-level) forced-choice protocol.",
    )
    ap.add_argument(
        "--fc_answer_prefix",
        type=str,
        default="\nFinal answer:",
        help="String to teacher-force after warmup before scoring candidates, to align the scoring point with the 'Final answer:' slot.",
    )


    ap.add_argument("--out_json", type=str, default=os.path.join(THIS_DIR, "prefill_vs_decode_alignment.json"))
    ap.add_argument("--out_txt", type=str, default=os.path.join(THIS_DIR, "prefill_vs_decode_alignment.txt"))
    ap.add_argument("--out_md", type=str, default=os.path.join(THIS_DIR, "prefill_vs_decode_alignment.md"))
    ap.add_argument("--out_tex", type=str, default=os.path.join(THIS_DIR, "prefill_vs_decode_alignment.tex"))
    args = ap.parse_args()

    set_global_seed(args.seed)

    layer_indices = [args.layer]
    print(f"[Env] model={args.model} device={args.device} dtype={args.dtype} layer={layer_indices}")

    model, tok = load_model_and_tokenizer(args.model, args.device, args.dtype)
    hidden_dim = getattr(model.config, "hidden_size", None) or getattr(model.config, "n_embd", None)
    if hidden_dim is None:
        raise RuntimeError("Could not infer hidden_dim")

    # Load datasets
    sub_by, eval_by, meta_by = load_all_examples(args.n_prompts, args.eval_n, args.seed)

    # 1) Collect DECODE states
    print("\n" + "=" * 80)
    print("[Basis] Estimating SHARED basis on D_decode (seq_len==1 decode steps)")
    print("=" * 80)

    decode_col = DecodeLastTokenCollector(layer_indices)
    layers, _ = get_model_layers(model)
    handles = []
    for li in layer_indices:
        handles.append(layers[li].register_forward_hook(decode_col.make_hook(li)))
    try:
        decode_task_states: Dict[str, np.ndarray] = {}
        for task, sub_exs in sub_by.items():
            decode_col.set_current_task(task)
            prompts = [ex.prompt for ex in sub_exs]
            collect_decode_states(
                model,
                tok,
                prompts,
                decode_col,
                batch_size=args.batch_size,
                max_new_tokens=args.calib_decode_max_new_tokens,
                max_prompt_len=args.max_prompt_len,
                decoding="greedy",
                temperature=1.0,
                top_p=1.0,
                top_k=0,
            )
            X = decode_col.get(task, layer_indices[0])
            if X is None:
                raise RuntimeError(f"No decode states for task={task}")
            X = _subsample_rows_np(X, args.per_task_max_states, seed=stable_int_seed(args.seed, task, "decode"))
            decode_task_states[task] = X
            print(f"[Collect][decode] task={task} states={X.shape[0]} x {X.shape[1]}")
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass
        decode_col.set_capture(False, None)

    # 2) Collect PREFILL states
    print("\n" + "=" * 80)
    print("[Basis] Estimating SHARED basis on D_prefill (seq_len>1 prefill tokens)")
    print("=" * 80)

    pre_col = PrefillLastTokenCollector(layer_indices)
    handles = []
    for li in layer_indices:
        handles.append(layers[li].register_forward_hook(pre_col.make_hook(li)))
    try:
        pre_task_states: Dict[str, np.ndarray] = {}
        for task, sub_exs in sub_by.items():
            pre_col.set_current_task(task)
            prompts = [ex.prompt for ex in sub_exs]
            collect_prefill_states(model, tok, prompts, pre_col, batch_size=args.batch_size, max_prompt_len=args.max_prompt_len)
            X = pre_col.get(task, layer_indices[0])
            if X is None:
                raise RuntimeError(f"No prefill states for task={task}")
            X = _subsample_rows_np(X, args.per_task_max_states, seed=stable_int_seed(args.seed, task, "prefill"))
            pre_task_states[task] = X
            print(f"[Collect][prefill] task={task} states={X.shape[0]} x {X.shape[1]}")
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass

    # Optional state-count matching for basis estimation
    state_match_info = {"enabled": bool(args.match_state_count)}
    if args.match_state_count:
        n_decode_min = min(v.shape[0] for v in decode_task_states.values())
        n_prefill_min = min(v.shape[0] for v in pre_task_states.values())
        n_match = min(n_decode_min, n_prefill_min)
        state_match_info.update({"n_decode_min": int(n_decode_min), "n_prefill_min": int(n_prefill_min), "n_match": int(n_match)})
        print("\n" + "=" * 80)
        print("[Basis] Re-estimating BOTH bases with STATE-COUNT matching")
        print(f"  decode n_min={n_decode_min} prefill n_min={n_prefill_min} => using n_match={n_match}")
        print("=" * 80)

        # downsample decode states to n_match per task
        new_decode = {}
        for t, X in decode_task_states.items():
            new_decode[t] = _subsample_rows_np(X, n_match, seed=stable_int_seed(args.seed, t, "decode_match"))
        decode_task_states = new_decode
        # downsample prefill states to n_match per task
        new_pre = {}
        for t, X in pre_task_states.items():
            new_pre[t] = _subsample_rows_np(X, n_match, seed=stable_int_seed(args.seed, t, "prefill_match"))
        pre_task_states = new_pre

    # Compute bases
    joint_decode, shared_idx_decode, extra_decode = compute_shared_basis_from_states(
        decode_task_states,
        pca_var=args.pca_var,
        min_dim=args.min_dim,
        max_dim=args.max_dim,
        tau=args.tau,
        m_shared=args.m_shared,
        seed=args.seed + 100,
    )
    joint_prefill, shared_idx_prefill, extra_prefill = compute_shared_basis_from_states(
        pre_task_states,
        pca_var=args.pca_var,
        min_dim=args.min_dim,
        max_dim=args.max_dim,
        tau=args.tau,
        m_shared=args.m_shared,
        seed=args.seed + 200,
    )

    k_decode = len(shared_idx_decode)
    k_prefill = len(shared_idx_prefill)
    k_match = min(k_decode, k_prefill)
    if k_match <= 0:
        raise RuntimeError("No shared components found (k_match<=0)")

    Q_decode_full = _build_shared_basis_from_joint(joint_decode, shared_idx_decode, k_decode)
    Q_prefill_full = _build_shared_basis_from_joint(joint_prefill, shared_idx_prefill, k_prefill)
    Q_decode_km = _build_shared_basis_from_joint(joint_decode, shared_idx_decode, k_match)
    Q_prefill_km = _build_shared_basis_from_joint(joint_prefill, shared_idx_prefill, k_match)
    Q_rand_km = random_orthonormal_basis_np(hidden_dim, k_match, seed=stable_int_seed(args.seed, "rand", k_match))

    print("\n" + "=" * 80)
    print("[Diag] Subspace similarity (Q_decode_shared vs Q_prefill_shared)")
    print("=" * 80)
    print(json.dumps(subspace_similarity(Q_decode_full, Q_prefill_full), indent=2))
    print("\n[Diag] Subspace similarity (dimension-matched k_match)")
    print(json.dumps(subspace_similarity(Q_decode_km, Q_prefill_km), indent=2))

    # Energy diagnostics on held-out collected states
    # Use pooled decode states
    decode_pool = np.concatenate([_subsample_rows_np(X, 4000, seed=stable_int_seed(args.seed, t, "er_d")) for t, X in decode_task_states.items()], axis=0)
    pre_pool = np.concatenate([_subsample_rows_np(X, 4000, seed=stable_int_seed(args.seed, t, "er_p")) for t, X in pre_task_states.items()], axis=0)
    er_decode_on_decode = energy_ratio_stats(decode_pool, Q_decode_full)
    er_prefill_on_decode = energy_ratio_stats(decode_pool, Q_prefill_full)
    er_prefill_on_prefill = energy_ratio_stats(pre_pool, Q_prefill_full)
    er_decode_on_prefill = energy_ratio_stats(pre_pool, Q_decode_full)

    er_decode_km_on_decode = energy_ratio_stats(decode_pool, Q_decode_km)
    er_prefill_km_on_decode = energy_ratio_stats(decode_pool, Q_prefill_km)
    er_rand_km_on_decode = energy_ratio_stats(decode_pool, Q_rand_km)
    er_prefill_km_on_prefill = energy_ratio_stats(pre_pool, Q_prefill_km)
    er_decode_km_on_prefill = energy_ratio_stats(pre_pool, Q_decode_km)

    print("\n" + "=" * 80)
    print("[Diag] Energy ratio r(h,Q)=||Q^T h||^2 / ||h||^2")
    print(f"  (FULL)   On DECODE states:  Q_decode_shared mean={er_decode_on_decode['mean']:.4f},  Q_prefill_shared mean={er_prefill_on_decode['mean']:.4f}")
    print(f"  (FULL)   On PREFILL states: Q_prefill_shared mean={er_prefill_on_prefill['mean']:.4f}, Q_decode_shared mean={er_decode_on_prefill['mean']:.4f}")
    print(f"  (k={k_match}) On DECODE states:  Q_decode_km mean={er_decode_km_on_decode['mean']:.4f},  Q_prefill_km mean={er_prefill_km_on_decode['mean']:.4f}, rand mean={er_rand_km_on_decode['mean']:.4f}")
    print(f"  (k={k_match}) On PREFILL states: Q_prefill_km mean={er_prefill_km_on_prefill['mean']:.4f}, Q_decode_km mean={er_decode_km_on_prefill['mean']:.4f}")

    # v3: optional forced-choice warmup tokens per task
    warmup_by_task: Dict[str, np.ndarray] = {}
    if args.fc_warmup_tokens > 0:
        print("\n" + "=" * 80)
        print(f"[FC Warmup] Precomputing baseline warmup tokens: W={args.fc_warmup_tokens} (decoding={args.fc_warmup_decoding}, ban_eos={bool(args.fc_warmup_ban_eos)})")
        print("=" * 80)
        for task in ["commonsenseqa", "strategyqa", "aqua"]:
            prompts = [ex.prompt for ex in eval_by[task]]
            warm_ids = precompute_fc_warmup_tokens(
                model,
                tok,
                prompts,
                warmup_tokens=args.fc_warmup_tokens,
                batch_size=args.batch_size,
                max_prompt_len=args.max_prompt_len,
                decoding=args.fc_warmup_decoding,
                temperature=0.7,
                top_p=0.9,
                top_k=0,
                ban_eos=bool(args.fc_warmup_ban_eos),
                seed=stable_int_seed(args.seed, args.fc_warmup_seed, task, "warmup"),
            )
            warmup_by_task[task] = warm_ids
            # print a tiny decoded snippet for sanity
            if warm_ids.shape[0] > 0:
                demo = tok.decode(warm_ids[0].tolist(), skip_special_tokens=True)
                print(f"[FC Warmup] {task}: warmup_ids shape={warm_ids.shape}; example[0] warmup text (first 120 chars): {demo[:120]!r}")

    # Forced-choice evaluation (k_match + native-k reference)
    fc_tasks = ["commonsenseqa", "strategyqa", "aqua"]
    fc_results = {}
    for task in fc_tasks:
        exs = eval_by[task]
        n = len(exs)
        print("\n" + "-" * 80)
        print(f"[Task] {task} n={n}")
        print("-" * 80)

        warm_ids = warmup_by_task.get(task, None) if args.fc_warmup_tokens > 0 else None

        # baseline
        base = forced_choice_logprob_eval(
            model,
            tok,
            exs,
            task,
            layer_indices=layer_indices,
            basis_np=None,
            alpha=args.alpha_remove,
            batch_size=args.batch_size,
            max_prompt_len=args.max_prompt_len,
            warmup_token_ids=warm_ids,
            answer_prefix=(args.fc_answer_prefix if bool(args.fc_add_answer_prefix) else ""),
        )
        b_acc, b_lo, b_hi = bootstrap_ci_mean(np.array(base["correct"], dtype=np.float32), args.bootstrap_iters, args.ci_alpha, seed=stable_int_seed(args.seed, task, "fc", "baseline"))
        # Quick sanity: if we are near chance, the forced-choice decision point is likely misaligned
        # (e.g., the model is still mid-reasoning). In that case, decode interventions won't show up clearly.
        cands_for_task = candidate_strings(task)
        if cands_for_task:
            chance = 1.0 / float(len(cands_for_task))
            if abs(b_acc - chance) < 0.02:
                print(
                    f"  [WARN] baseline acc ~ chance (acc={b_acc*100:.1f}%, chance={chance*100:.1f}%). "
                    "This usually means the scoring point is not aligned with the answer, or warmup is too short. "
                    "Try: increase --fc_warmup_tokens; keep --fc_add_answer_prefix=1; or switch to full generation eval."
                )

        print(f"  [FC] baseline acc={fmt_acc(b_acc, b_lo, b_hi)} {base['hook_stats']}")

        # decode-shared (native)
        dec_full = forced_choice_logprob_eval(
            model,
            tok,
            exs,
            task,
            layer_indices=layer_indices,
            basis_np=Q_decode_full,
            alpha=args.alpha_remove,
            batch_size=args.batch_size,
            max_prompt_len=args.max_prompt_len,
            warmup_token_ids=warm_ids,
            answer_prefix=(args.fc_answer_prefix if bool(args.fc_add_answer_prefix) else ""),
        )
        d_acc, d_lo, d_hi = bootstrap_ci_mean(np.array(dec_full["correct"], dtype=np.float32), args.bootstrap_iters, args.ci_alpha, seed=stable_int_seed(args.seed, task, "fc", "decode_full"))
        print(f"  [FC] decode_shared_full acc={fmt_acc(d_acc, d_lo, d_hi)} {dec_full['hook_stats']}")

        # prefill-shared (native)
        pre_full = forced_choice_logprob_eval(
            model,
            tok,
            exs,
            task,
            layer_indices=layer_indices,
            basis_np=Q_prefill_full,
            alpha=args.alpha_remove,
            batch_size=args.batch_size,
            max_prompt_len=args.max_prompt_len,
            warmup_token_ids=warm_ids,
            answer_prefix=(args.fc_answer_prefix if bool(args.fc_add_answer_prefix) else ""),
        )
        p_acc, p_lo, p_hi = bootstrap_ci_mean(np.array(pre_full["correct"], dtype=np.float32), args.bootstrap_iters, args.ci_alpha, seed=stable_int_seed(args.seed, task, "fc", "prefill_full"))
        print(f"  [FC] prefill_shared_full acc={fmt_acc(p_acc, p_lo, p_hi)} {pre_full['hook_stats']}")

        # dimension-matched
        dec_km = forced_choice_logprob_eval(
            model,
            tok,
            exs,
            task,
            layer_indices=layer_indices,
            basis_np=Q_decode_km,
            alpha=args.alpha_remove,
            batch_size=args.batch_size,
            max_prompt_len=args.max_prompt_len,
            warmup_token_ids=warm_ids,
            answer_prefix=(args.fc_answer_prefix if bool(args.fc_add_answer_prefix) else ""),
        )
        dk_acc, dk_lo, dk_hi = bootstrap_ci_mean(np.array(dec_km["correct"], dtype=np.float32), args.bootstrap_iters, args.ci_alpha, seed=stable_int_seed(args.seed, task, "fc", "decode_km"))
        print(f"  [FC] decode_shared_km acc={fmt_acc(dk_acc, dk_lo, dk_hi)}")

        pre_km = forced_choice_logprob_eval(
            model,
            tok,
            exs,
            task,
            layer_indices=layer_indices,
            basis_np=Q_prefill_km,
            alpha=args.alpha_remove,
            batch_size=args.batch_size,
            max_prompt_len=args.max_prompt_len,
            warmup_token_ids=warm_ids,
            answer_prefix=(args.fc_answer_prefix if bool(args.fc_add_answer_prefix) else ""),
        )
        pk_acc, pk_lo, pk_hi = bootstrap_ci_mean(np.array(pre_km["correct"], dtype=np.float32), args.bootstrap_iters, args.ci_alpha, seed=stable_int_seed(args.seed, task, "fc", "prefill_km"))
        print(f"  [FC] prefill_shared_km acc={fmt_acc(pk_acc, pk_lo, pk_hi)}")

        rand_km = forced_choice_logprob_eval(
            model,
            tok,
            exs,
            task,
            layer_indices=layer_indices,
            basis_np=Q_rand_km,
            alpha=args.alpha_remove,
            batch_size=args.batch_size,
            max_prompt_len=args.max_prompt_len,
            warmup_token_ids=warm_ids,
            answer_prefix=(args.fc_answer_prefix if bool(args.fc_add_answer_prefix) else ""),
        )
        rk_acc, rk_lo, rk_hi = bootstrap_ci_mean(np.array(rand_km["correct"], dtype=np.float32), args.bootstrap_iters, args.ci_alpha, seed=stable_int_seed(args.seed, task, "fc", "rand_km"))
        print(f"  [FC] rand_km acc={fmt_acc(rk_acc, rk_lo, rk_hi)}")

        # Paired tests (decode_km vs prefill_km)
        base_arr = np.array(base["correct"], dtype=np.float32)
        dk_arr = np.array(dec_km["correct"], dtype=np.float32)
        pk_arr = np.array(pre_km["correct"], dtype=np.float32)
        stat_dk_vs_pk = summarize_paired(pk_arr, dk_arr, args.bootstrap_iters, args.perm_iters, args.ci_alpha, seed=stable_int_seed(args.seed, task, "paired", "dk_vs_pk"))

        fc_results[task] = {
            "n": n,
            "baseline": {"acc": b_acc, "ci": [b_lo, b_hi]},
            "decode_full": {"acc": d_acc, "ci": [d_lo, d_hi], "k": k_decode},
            "prefill_full": {"acc": p_acc, "ci": [p_lo, p_hi], "k": k_prefill},
            "decode_km": {"acc": dk_acc, "ci": [dk_lo, dk_hi], "k": k_match},
            "prefill_km": {"acc": pk_acc, "ci": [pk_lo, pk_hi], "k": k_match},
            "rand_km": {"acc": rk_acc, "ci": [rk_lo, rk_hi], "k": k_match},
            "paired": {"decode_km_minus_prefill_km": stat_dk_vs_pk},
        }

    # Build Markdown/LaTeX tables
    warm_str = f" (warmup W={args.fc_warmup_tokens})" if args.fc_warmup_tokens > 0 else ""

    # k_match table
    km_rows = []
    for task in fc_tasks:
        r = fc_results[task]
        stat = r["paired"]["decode_km_minus_prefill_km"]
        km_rows.append(
            [
                task,
                str(r["n"]),
                fmt_acc(r["baseline"]["acc"], r["baseline"]["ci"][0], r["baseline"]["ci"][1]),
                fmt_acc(r["decode_km"]["acc"], r["decode_km"]["ci"][0], r["decode_km"]["ci"][1]),
                fmt_acc(r["prefill_km"]["acc"], r["prefill_km"]["ci"][0], r["prefill_km"]["ci"][1]),
                fmt_acc(r["rand_km"]["acc"], r["rand_km"]["ci"][0], r["rand_km"]["ci"][1]),
                f"{stat['mean_diff']*100:+.1f} [{stat['ci_low']*100:+.1f}, {stat['ci_high']*100:+.1f}]",
                f"{stat['p_value']:.3g}",
            ]
        )

    km_header = [
        "Task",
        "n",
        "Baseline",
        f"Decode-shared (k={k_match})",
        f"Prefill-shared (k={k_match})",
        f"Random (k={k_match})",
        "Δ(Decode-Prefill) [CI]",
        "p",
    ]
    md_km = md_table(km_rows, km_header)

    tex_km = latex_table(
        km_rows,
        km_header,
        caption=(
            f"Forced-choice accuracy after removing a fully-shared subspace during decode steps, using dimension-matched bases (k={k_match})."
            f" Warmup tokens{warm_str} are baseline-generated and teacher-forced for all conditions."
        ),
        label="tab:prefill-vs-decode-kmatch",
        colspec="lrcccccc",
    )

    # native table
    nat_rows = []
    for task in fc_tasks:
        r = fc_results[task]
        # paired decode_full - prefill_full (note: different k)
        base = np.array([1.0] * r["n"], dtype=np.float32)  # dummy, we only report Δ via acc diff? keep simple
        # We'll compute paired using stored correct arrays? Not stored for nat in this compact v3.
        # Use acc difference with conservative CI omitted here.
        delta = (r["decode_full"]["acc"] - r["prefill_full"]["acc"]) * 100
        nat_rows.append(
            [
                task,
                str(r["n"]),
                fmt_acc(r["baseline"]["acc"], r["baseline"]["ci"][0], r["baseline"]["ci"][1]),
                fmt_acc(r["decode_full"]["acc"], r["decode_full"]["ci"][0], r["decode_full"]["ci"][1]),
                fmt_acc(r["prefill_full"]["acc"], r["prefill_full"]["ci"][0], r["prefill_full"]["ci"][1]),
                f"{delta:+.1f}",
                "(n/a)",
            ]
        )

    nat_header = [
        "Task",
        "n",
        "Baseline",
        f"Decode-shared (k={k_decode})",
        f"Prefill-shared (k={k_prefill})",
        "Δ(Decode-Prefill)",
        "p",
    ]
    md_nat = md_table(nat_rows, nat_header)
    tex_nat = latex_table(
        nat_rows,
        nat_header,
        caption=(
            "Native shared-k reference table (no dimension matching). "
            f"Warmup tokens{warm_str} are baseline-generated and teacher-forced for all conditions."
        ),
        label="tab:prefill-vs-decode-native",
        colspec="lrccccc",
    )

    # Save JSON
    results = {
        "config": {
            "model": args.model,
            "dtype": args.dtype,
            "device": args.device,
            "layer": args.layer,
            "n_prompts": args.n_prompts,
            "eval_n": args.eval_n,
            "calib_decode_max_new_tokens": args.calib_decode_max_new_tokens,
            "per_task_max_states": args.per_task_max_states,
            "pca_var": args.pca_var,
            "tau": args.tau,
            "m_shared": args.m_shared,
            "alpha_remove": args.alpha_remove,
            "match_state_count": state_match_info,
            "fc_warmup_tokens": args.fc_warmup_tokens,
            "fc_warmup_decoding": args.fc_warmup_decoding,
            "fc_warmup_ban_eos": bool(args.fc_warmup_ban_eos),
            "fc_warmup_seed": args.fc_warmup_seed,
            "fc_add_answer_prefix": bool(args.fc_add_answer_prefix),
            "fc_answer_prefix": args.fc_answer_prefix,
            "seed": args.seed,
            "dataset_meta": meta_by,
        },
        "basis": {
            "decode": {"cross_dim": extra_decode["cross_dim"], "shared_k": k_decode, "n_balanced": extra_decode["n_balanced"]},
            "prefill": {"cross_dim": extra_prefill["cross_dim"], "shared_k": k_prefill, "n_balanced": extra_prefill["n_balanced"]},
            "k_match": k_match,
            "subspace_similarity_full": subspace_similarity(Q_decode_full, Q_prefill_full),
            "subspace_similarity_kmatch": subspace_similarity(Q_decode_km, Q_prefill_km),
            "energy": {
                "full": {
                    "decode_on_decode": er_decode_on_decode,
                    "prefill_on_decode": er_prefill_on_decode,
                    "prefill_on_prefill": er_prefill_on_prefill,
                    "decode_on_prefill": er_decode_on_prefill,
                },
                "kmatch": {
                    "decode_on_decode": er_decode_km_on_decode,
                    "prefill_on_decode": er_prefill_km_on_decode,
                    "rand_on_decode": er_rand_km_on_decode,
                    "prefill_on_prefill": er_prefill_km_on_prefill,
                    "decode_on_prefill": er_decode_km_on_prefill,
                },
            },
        },
        "forced_choice": fc_results,
        "tables": {"markdown_kmatch": md_km, "markdown_native": md_nat, "latex_kmatch": tex_km, "latex_native": tex_nat},
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=json_default)

    # Save TXT summary
    summary_lines = []
    summary_lines.append("[Summary]")
    summary_lines.append(f"Model={args.model} dtype={args.dtype} device={args.device} layer={args.layer}")
    summary_lines.append(f"Decode shared_k={k_decode} Prefill shared_k={k_prefill} k_match={k_match}")
    if args.match_state_count:
        summary_lines.append(f"State-count matching: enabled (n_match={state_match_info.get('n_match')})")
    else:
        summary_lines.append("State-count matching: disabled")
    summary_lines.append(f"FC warmup tokens: {args.fc_warmup_tokens}")
    summary_lines.append(f"FC add answer prefix: {bool(args.fc_add_answer_prefix)}")
    summary_lines.append(f"FC answer prefix: {args.fc_answer_prefix!r}")
    summary_lines.append("")
    summary_lines.append("# PREFILL vs DECODE BASIS (estimation distribution mismatch test)")
    summary_lines.append("")
    summary_lines.append(f"- Model: `{args.model}`  layer={args.layer}  dtype={args.dtype}  device={args.device}")
    summary_lines.append(f"- Decode shared_k={k_decode}, Prefill shared_k={k_prefill}, k_match={k_match}")
    summary_lines.append(f"- State-count matching: {'enabled' if args.match_state_count else 'disabled'}")
    summary_lines.append(f"- Forced-choice warmup tokens: {args.fc_warmup_tokens} (baseline-generated, teacher-forced)")
    summary_lines.append(f"- Forced-choice answer prefix: {args.fc_answer_prefix!r} (teacher-forced before scoring; enabled={bool(args.fc_add_answer_prefix)})")
    summary_lines.append("")
    summary_lines.append("## Dimension-matched forced-choice (k_match)")
    summary_lines.append(f"### Forced-choice accuracy with dimension-matched shared bases (k={k_match})")
    summary_lines.append(md_km)
    summary_lines.append("")
    summary_lines.append("## Native-k forced-choice (reference)")
    summary_lines.append("### Forced-choice accuracy with native shared-k (reference)")
    summary_lines.append(md_nat)
    summary_lines.append("")

    with open(args.out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    # Save MD and TEX
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")
    with open(args.out_tex, "w", encoding="utf-8") as f:
        f.write(tex_km + "\n" + tex_nat + "\n")

    print("\n" + "=" * 80)
    print("\n".join(summary_lines[:10]))
    print("...")
    print(f"[Done] JSON: {args.out_json}")
    print(f"[Done] TXT : {args.out_txt}")
    print(f"[Done] MD  : {args.out_md}")
    print(f"[Done] TEX : {args.out_tex}")
    print("=" * 80)


if __name__ == "__main__":
    main()
