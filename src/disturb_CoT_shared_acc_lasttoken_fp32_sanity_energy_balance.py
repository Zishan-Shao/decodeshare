# -*- coding: utf-8 -*-
"""
disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance.py

Last-token (decode-only) shared-subspace removal with full sanity checks.

Key features:
  - A3 decode-aligned basis: collect decode-phase (seq_len==1) last-token states on calibration prompts
  - pooled PCA -> sharedness -> shared basis
  - intervention: last-token removal only (no rotation), alpha=1.0 default
  - staged gating by generated token count
  - evaluation: task EM accuracy + bootstrap CI + paired sign-flip permutation test (per-example pairing)
  - greedy main + sampling robustness
  - SANITY:
      * orthonormality of shared/rand bases
      * overlap between shared and rand
      * energy ratio on calibration decode states (shared vs rand)
      * hook stats (decode_calls / intervened)
      * extraction rate, eos rate, avg_new_tokens

Requirements:
  pip install transformers datasets numpy torch tqdm

Also requires your project utilities:
  from joint_subspace_large.disturb_cross_task_all_shared import get_model_layers, compute_cross_task_subspace, find_fully_shared_basis_improved

"facebook/opt-6.7b",
"Qwen/Qwen2.5-7B"
"Qwen/Qwen2.5-7B-Instruct"

"google/gemma-3-12b-it"

google/gemma-2-2b-it

Run (example):
  CUDA_VISIBLE_DEVICES=1 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --device cuda \
    --model_dtype fp32 \
    --n_subspace 128 \
    --n_eval 256 \
    --layer 10 \
    --reasoning_tokens 128 \
    --max_new_tokens 256 \
    --rand_type joint_nonshared_topk

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
    # Fix np.int64 / np.float32 etc.
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
def bootstrap_ci_mean(
    values: np.ndarray,
    iters: int,
    alpha: float,
    seed: int,
) -> Tuple[float, float, float]:
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

def paired_bootstrap_ci_diff(
    baseline: np.ndarray,
    treatment: np.ndarray,
    iters: int,
    alpha: float,
    seed: int,
) -> Tuple[float, float, float]:
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

def signflip_permutation_test(
    baseline: np.ndarray,
    treatment: np.ndarray,
    iters: int,
    seed: int,
) -> float:
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

def load_gsm8k_examples(n_subspace: int, n_eval: int, seed: int):
    ds = load_dataset("gsm8k", "main")
    sub_split = "train" if "train" in ds else list(ds.keys())[0]
    eval_split = "test" if "test" in ds else ("validation" if "validation" in ds else sub_split)

    sub_rows = sample_hf_split(ds[sub_split], n_subspace, seed + 1)
    eval_rows = sample_hf_split(ds[eval_split], n_eval, seed + 2)

    sub_exs, eval_exs = [], []
    for i, ex in enumerate(sub_rows):
        gold = parse_gsm8k_gold(ex["answer"])
        sub_exs.append(Example("gsm8k", f"gsm8k-{sub_split}-{i}", build_prompt_gsm8k(ex["question"]), gold))
    for i, ex in enumerate(eval_rows):
        gold = parse_gsm8k_gold(ex["answer"])
        eval_exs.append(Example("gsm8k", f"gsm8k-{eval_split}-{i}", build_prompt_gsm8k(ex["question"]), gold))

    meta = {"hf_id": "gsm8k/main", "subspace_split": sub_split, "eval_split": eval_split}
    return sub_exs, eval_exs, meta

def load_commonsenseqa_examples(n_subspace: int, n_eval: int, seed: int):
    ds = load_dataset("commonsense_qa")
    sub_split = "train" if "train" in ds else list(ds.keys())[0]
    eval_split = "validation" if "validation" in ds else ("test" if "test" in ds else sub_split)

    sub_rows = sample_hf_split(ds[sub_split], n_subspace, seed + 11)
    eval_rows = sample_hf_split(ds[eval_split], n_eval, seed + 12)

    sub_exs, eval_exs = [], []
    for i, ex in enumerate(sub_rows):
        gold = safe_upper(ex["answerKey"])
        sub_exs.append(Example("commonsenseqa", f"csqa-{sub_split}-{i}", build_prompt_commonsenseqa(ex["question"], ex["choices"]), gold))
    for i, ex in enumerate(eval_rows):
        gold = safe_upper(ex["answerKey"])
        eval_exs.append(Example("commonsenseqa", f"csqa-{eval_split}-{i}", build_prompt_commonsenseqa(ex["question"], ex["choices"]), gold))

    meta = {"hf_id": "commonsense_qa", "subspace_split": sub_split, "eval_split": eval_split}
    return sub_exs, eval_exs, meta

def load_aqua_examples(n_subspace: int, n_eval: int, seed: int):
    ds = load_dataset("aqua_rat")
    sub_split = "train" if "train" in ds else list(ds.keys())[0]
    eval_split = "test" if "test" in ds else ("validation" if "validation" in ds else sub_split)

    sub_rows = sample_hf_split(ds[sub_split], n_subspace, seed + 21)
    eval_rows = sample_hf_split(ds[eval_split], n_eval, seed + 22)

    def get_gold(ex: dict) -> str:
        if "correct" in ex:
            return safe_upper(ex["correct"])
        if "answer" in ex:
            return safe_upper(ex["answer"])
        return ""

    sub_exs, eval_exs = [], []
    for i, ex in enumerate(sub_rows):
        sub_exs.append(Example("aqua", f"aqua-{sub_split}-{i}", build_prompt_aqua(ex["question"], ex["options"]), get_gold(ex)))
    for i, ex in enumerate(eval_rows):
        eval_exs.append(Example("aqua", f"aqua-{eval_split}-{i}", build_prompt_aqua(ex["question"], ex["options"]), get_gold(ex)))

    meta = {"hf_id": "aqua_rat", "subspace_split": sub_split, "eval_split": eval_split}
    return sub_exs, eval_exs, meta

def load_strategyqa_examples(n_subspace: int, n_eval: int, seed: int):
    hf_id = "ChilleD/StrategyQA"
    ds = load_dataset(hf_id)
    sub_split = "train" if "train" in ds else list(ds.keys())[0]
    eval_split = "test" if "test" in ds else ("validation" if "validation" in ds else sub_split)

    sub_rows = sample_hf_split(ds[sub_split], n_subspace, seed + 31)
    eval_rows = sample_hf_split(ds[eval_split], n_eval, seed + 32)

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

    meta = {"hf_id": hf_id, "subspace_split": sub_split, "eval_split": eval_split, "question_field": "question", "answer_field": "answer"}
    return sub_exs, eval_exs, meta

def load_all_examples(n_subspace: int, n_eval: int, seed: int):
    loaders = {
        "gsm8k": load_gsm8k_examples,
        "commonsenseqa": load_commonsenseqa_examples,
        "strategyqa": load_strategyqa_examples,
        "aqua": load_aqua_examples,
    }
    sub_by, eval_by, meta_by = {}, {}, {}
    for name, fn in loaders.items():
        print(f"[Data] Loading {name} (subspace={n_subspace}, eval={n_eval}) ...")
        sub_exs, eval_exs, meta = fn(n_subspace, n_eval, seed)
        sub_by[name] = sub_exs
        eval_by[name] = eval_exs
        meta_by[name] = meta
        print(f"[Data] {name}: subspace={len(sub_exs)}, eval={len(eval_exs)} meta={meta}")
    return sub_by, eval_by, meta_by

# -----------------------------
# Answer parsing / correctness
# -----------------------------
def extract_final_answer_line(text: str) -> str:
    m = re.search(r"Final answer\s*:\s*(.*)", text, flags=re.IGNORECASE)
    if not m:
        return ""
    s = (m.group(1) or "").strip()
    if not s:
        return ""
    lines = s.splitlines()
    return lines[0].strip() if lines else ""

def parse_prediction(dataset: str, continuation_text: str) -> str:
    ans_line = extract_final_answer_line(continuation_text)

    if dataset == "gsm8k":
        pred = normalize_number_str(ans_line)
        if pred is not None:
            return pred
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", continuation_text.replace(",", ""))
        return normalize_number_str(nums[-1]) if nums else ""

    if dataset in ["commonsenseqa", "aqua"]:
        m = re.search(r"\b([A-E])\b", ans_line.upper())
        if m:
            return m.group(1)
        m2 = re.search(r"\b([A-E])\b", continuation_text.upper())
        return m2.group(1) if m2 else ""

    if dataset == "strategyqa":
        t = ans_line.strip().lower()
        if "yes" in t:
            return "YES"
        if "no" in t:
            return "NO"
        t2 = continuation_text.lower()
        if "yes" in t2 and "no" in t2:
            return "YES" if t2.find("yes") < t2.find("no") else "NO"
        if "yes" in t2:
            return "YES"
        if "no" in t2:
            return "NO"
        return ""

    return ""

def is_correct(dataset: str, pred: str, gold: str) -> int:
    if dataset == "gsm8k":
        return int(pred != "" and gold != "" and pred == gold)
    if dataset in ["commonsenseqa", "aqua", "strategyqa"]:
        return int(pred != "" and pred.upper() == gold.upper())
    return 0

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
from collections import defaultdict
from typing import DefaultDict

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
        B, T0 = input_ids.shape

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

def infer_component_variances(contributions: Dict[str, Any], tasks: List[str], cross_dim: int) -> np.ndarray:
    """
    Robustly infer per-component variance vector from your contributions dict.
    We expect each task has a 1D vector length cross_dim under some key.
    """
    candidates = []
    for t in tasks:
        d = contributions.get(t, {})
        v = None
        for key in ["variances", "component_variances", "per_component_variance", "var", "vars"]:
            if key in d:
                v = np.asarray(d[key], dtype=np.float64)
                break
        if v is None:
            # fallback: find first 1D list/array of length cross_dim
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

def orthonormalize_np(basis: np.ndarray) -> np.ndarray:
    q, _ = np.linalg.qr(basis.astype(np.float32, copy=False))
    return q.astype(np.float32, copy=False)

def max_offdiag(Q: np.ndarray) -> float:
    G = Q.T @ Q
    k = G.shape[0]
    G = G - np.eye(k, dtype=G.dtype)
    return float(np.max(np.abs(G)))

def max_overlap(Qa: np.ndarray, Qb: np.ndarray) -> float:
    # max |Qa^T Qb|
    M = Qa.T @ Qb
    return float(np.max(np.abs(M)))

def energy_ratio_stats(states: np.ndarray, Q: np.ndarray) -> Dict[str, float]:
    # states: [N, D], Q: [D, k] orthonormal
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
        # Match pooled variance distribution of shared indices
        shared_vars = [(i, pooled_var[i]) for i in shared_indices]
        shared_vars.sort(key=lambda x: x[1])  # ascending by variance

        nonshared_sorted = sorted(nonshared, key=lambda i: pooled_var[i])  # ascending
        nonshared_vals = [pooled_var[i] for i in nonshared_sorted]

        import bisect
        chosen = []
        for _, v in shared_vars:
            j = bisect.bisect_left(nonshared_vals, v)
            cand_pos = []
            if 0 <= j < len(nonshared_sorted):
                cand_pos.append(j)
            if 0 <= j-1 < len(nonshared_sorted):
                cand_pos.append(j-1)
            # choose nearest; random tie-break
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
            # fill remainder uniformly
            remaining = nonshared_sorted
            extra = list(rng.choice(remaining, size=(k-len(chosen)), replace=False))
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

    # Build activation dict
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
    shared_indices = find_fully_shared_basis_improved(
        contributions,
        tasks,
        cross_dim,
        min_tasks_shared=len(tasks),
        relative_threshold=0.001,
        top_k_components=cross_dim,
    )
    if not shared_indices:
        print("[Subspace-A3] No fully-shared basis across ALL tasks; fallback to >=2 tasks shared.")
        shared_indices = find_fully_shared_basis_improved(
            contributions,
            tasks,
            cross_dim,
            min_tasks_shared=2,
            relative_threshold=0.001,
            top_k_components=cross_dim,
        )

    print(f"[Subspace-A3] cross_dim={cross_dim}, shared_basis_count={len(shared_indices)}")
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
        # Orthonormalize once (safety)
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

        # full intervene
        Q = self._Q(hs.device)  # [D,k]
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
# Generation (always EOS-aware) + per-example stats
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

        # Prefill
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

            # Force EOS for finished sequences
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

        # per-example slice by its own gen_steps (so we don't keep forced eos padding)
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

        preds, correct, extracted = [], [], []
        for ex, cont in zip(examples, continuations):
            pred = parse_prediction(ex.dataset, cont)
            preds.append(pred)
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
            # keep only a tiny sample for debugging to keep json small
            "debug_samples": [{"pred": preds[j], "gold": examples[j].gold, "cont": continuations[j][:300]} for j in range(min(3, len(examples)))],
        }
    finally:
        remove_hooks(handles)

# -----------------------------
# Model loading
# -----------------------------
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

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model_dtype", type=str, default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--n_subspace", type=int, default=128)
    ap.add_argument("--n_eval", type=int, default=256)
    ap.add_argument("--pca_var", type=float, default=0.95)
    ap.add_argument("--min_dim", type=int, default=1)
    ap.add_argument("--max_dim", type=int, default=4096)
    ap.add_argument("--alpha_remove", type=float, default=1.0)
    ap.add_argument("--reasoning_tokens", type=int, default=128)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_prompt_len", type=int, default=512)
    ap.add_argument("--bootstrap_iters", type=int, default=5000)
    ap.add_argument("--perm_iters", type=int, default=10000)
    ap.add_argument("--ci_alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample_seed", type=int, default=12345)
    ap.add_argument("--rand_type", type=str, default="joint_nonshared_varmatch",
                    choices=["joint_nonshared_uniform", "joint_nonshared_topk", "joint_nonshared_varmatch"])
    ap.add_argument("--out_json", type=str, default=os.path.join(THIS_DIR, "level0_acc_results.json"))
    ap.add_argument("--out_txt", type=str, default=os.path.join(THIS_DIR, "level0_acc_summary.txt"))
    args = ap.parse_args()

    set_global_seed(args.seed)

    layer_indices = [args.layer]

    print(f"[Env] DEVICE={args.device}")
    print(f"[Env] MODEL={args.model}")
    print(f"[Env] model_dtype={args.model_dtype}")
    print(f"[Env] layer_indices={layer_indices}")

    model, tokenizer = load_model_and_tokenizer(args.model, args.device, args.model_dtype)
    hidden_dim = getattr(model.config, "hidden_size", None) or getattr(model.config, "n_embd", None)
    if hidden_dim is None:
        raise RuntimeError("Could not infer hidden_dim from model.config")
    print(f"[Env] hidden_dim={hidden_dim}")

    # Load datasets
    sub_by, eval_by, meta_by = load_all_examples(args.n_subspace, args.n_eval, args.seed)
    print("\n" + "=" * 80)
    print(f"[Data] Loaded tasks: {list(sub_by.keys())}")
    print(f"[Data] Meta: {json.dumps(meta_by, indent=2, ensure_ascii=False)}")
    print("=" * 80)

    # A3: decode-aligned basis on subspace split prompts
    prompts_by_task = {k: [ex.prompt for ex in v] for k, v in sub_by.items()}
    joint_subspace, shared_indices, extra, task_acts = compute_shared_subspace_decode_aligned(
        model=model,
        tokenizer=tokenizer,
        prompts_by_task=prompts_by_task,
        layer_indices=layer_indices,
        calib_decoding="greedy",
        calib_batch_size=args.batch_size,
        calib_max_new_tokens=args.reasoning_tokens,
        per_task_max_states=20000,
        max_prompt_len=args.max_prompt_len,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        global_seed=args.seed,
        variance_threshold=args.pca_var,
        min_dim=args.min_dim,
        max_dim=args.max_dim,
    )

    cross_dim = int(extra["cross_dim"])
    if len(shared_indices) == 0:
        raise RuntimeError("No shared basis found (shared_indices empty). Try relax threshold or min_tasks_shared.")

    # Build shared basis
    shared_basis = joint_subspace[:, shared_indices]  # [D, kS]
    Q_shared = orthonormalize_np(shared_basis)

    # Build pooled variance for stronger random controls
    pooled_var = infer_component_variances(extra["task_contributions"], extra["tasks_used"], cross_dim)

    # Build random basis from joint nonshared components (variance-matched/topk/uniform)
    kS = Q_shared.shape[1]
    rand_indices = select_rand_indices(
        rand_type=args.rand_type,
        cross_dim=cross_dim,
        shared_indices=shared_indices,
        pooled_var=pooled_var,
        k=kS,
        seed=stable_int_seed(args.seed, "rand_idx", args.rand_type),
    )
    rand_basis = joint_subspace[:, rand_indices]
    Q_rand = orthonormalize_np(rand_basis)

    # SANITY: orthonormality + overlap
    print(f"[Subspace] cross_dim={cross_dim}, shared_basis_dim={kS}, rand_type={args.rand_type}")
    print(f"[Sanity] Orthonormality max offdiag: shared={max_offdiag(Q_shared):.3e}, rand={max_offdiag(Q_rand):.3e}")
    print(f"[Sanity] Max overlap |Q_shared^T Q_rand| = {max_overlap(Q_shared, Q_rand):.3e}")

    # SANITY: energy ratio on calibration decode states (use pooled sample across tasks)
    # collect a small sample of states from task_acts (layer_indices only)
    layer = layer_indices[0]
    pool = []
    for t in extra["tasks_used"]:
        X = task_acts[t][layer]
        ss = stable_int_seed(args.seed, "energy_sample", t)
        Xs = _subsample_rows_np(X, n_max=4000, seed=ss)
        pool.append(Xs)
    calib_states = np.concatenate(pool, axis=0)  # [N, D]
    er_s = energy_ratio_stats(calib_states, Q_shared)
    er_r = energy_ratio_stats(calib_states, Q_rand)
    print("[Sanity] Energy ratio on calib decode states:")
    print(f"         shared mean={er_s['mean']:.4f} (p50={er_s['p50']:.4f}, p95={er_s['p95']:.4f})")
    print(f"         rand   mean={er_r['mean']:.4f} (p50={er_r['p50']:.4f}, p95={er_r['p95']:.4f})")
    if er_r["mean"] > 0 and er_s["mean"] / er_r["mean"] > 3.0:
        print("[Sanity][WARN] shared/rand energy ratio gap is large. "
              "If your claim is specifically 'sharedness > random', consider rand_type=joint_nonshared_varmatch or joint_nonshared_topk.")

    # Eval
    DECODINGS = ["greedy", "sample"]
    CONDITIONS = ["baseline", "shared_full", "shared_staged", "rand_full", "rand_staged"]

    results: Dict[str, Any] = {
        "config": {
            "model": args.model,
            "device": args.device,
            "model_dtype": args.model_dtype,
            "layer_indices": layer_indices,
            "pca_variance_threshold": args.pca_var,
            "n_subspace_per_task": args.n_subspace,
            "n_eval_per_task": args.n_eval,
            "alpha_remove": args.alpha_remove,
            "reasoning_token_threshold": args.reasoning_tokens,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "bootstrap_iters": args.bootstrap_iters,
            "perm_iters": args.perm_iters,
            "ci_alpha": args.ci_alpha,
            "seed": args.seed,
            "sample_seed": args.sample_seed,
            "shared_basis_count": int(kS),
            "cross_dim": int(cross_dim),
            "tasks_used_for_subspace": extra.get("tasks_used"),
            "dataset_meta": meta_by,
            "rand_type": args.rand_type,
            "energy_ratio_shared": er_s,
            "energy_ratio_rand": er_r,
        },
        "by_dataset": {},
    }

    print("\n" + "=" * 80)
    print("[Eval] Running Level-0 evaluations (accuracy + CI, greedy + sampling) ...")
    print("=" * 80)

    for task_name, eval_exs in eval_by.items():
        print("\n" + "-" * 80)
        print(f"[Eval] Dataset={task_name} (n={len(eval_exs)})")
        print("-" * 80)

        block: Dict[str, Any] = {"n": len(eval_exs), "runs": {}, "paired_tests": {}}

        for decoding in DECODINGS:
            for cond in CONDITIONS:
                if cond == "baseline":
                    Q = None
                    mode = "baseline"
                    cond_name = "baseline"
                elif cond == "shared_full":
                    Q = Q_shared
                    mode = "shared_full"
                    cond_name = "full"
                elif cond == "shared_staged":
                    Q = Q_shared
                    mode = "shared_staged"
                    cond_name = "staged"
                elif cond == "rand_full":
                    Q = Q_rand
                    mode = "rand_full"
                    cond_name = "full"
                elif cond == "rand_staged":
                    Q = Q_rand
                    mode = "rand_staged"
                    cond_name = "staged"
                else:
                    raise ValueError(cond)

                print(f"[Eval] condition={mode}, decoding={decoding}")
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
                print(f"  acc={fmt_acc(run['accuracy'], run['ci_low'], run['ci_high'])} "
                      f"extr={run['extraction_rate']*100:.1f}% eos={run['eos_rate']*100:.1f}% avg_new_tok={run['avg_new_tokens']:.1f}")
                for hs in run.get("hook_stats", []):
                    print(f"  [HookStats] {hs['name']} decode_calls={hs['decode_calls']} intervened={hs['intervened']}")

        # Paired tests
        for decoding in DECODINGS:
            base = np.array(block["runs"][f"{decoding}/baseline"]["correct"], dtype=np.float32)
            shared_full = np.array(block["runs"][f"{decoding}/shared_full"]["correct"], dtype=np.float32)
            rand_full = np.array(block["runs"][f"{decoding}/rand_full"]["correct"], dtype=np.float32)

            seed0 = stable_int_seed(args.seed, task_name, decoding, "paired")
            block["paired_tests"][decoding] = {
                "shared_full_vs_baseline": summarize_paired(
                    base, shared_full,
                    label=f"{task_name}:{decoding}:shared_full_vs_baseline",
                    bootstrap_iters=args.bootstrap_iters,
                    perm_iters=args.perm_iters,
                    alpha=args.ci_alpha,
                    seed=seed0 + 1,
                ),
                "rand_full_vs_baseline": summarize_paired(
                    base, rand_full,
                    label=f"{task_name}:{decoding}:rand_full_vs_baseline",
                    bootstrap_iters=args.bootstrap_iters,
                    perm_iters=args.perm_iters,
                    alpha=args.ci_alpha,
                    seed=seed0 + 2,
                ),
                "shared_full_vs_rand_full": summarize_paired(
                    rand_full, shared_full,
                    label=f"{task_name}:{decoding}:shared_full_vs_rand_full",
                    bootstrap_iters=args.bootstrap_iters,
                    perm_iters=args.perm_iters,
                    alpha=args.ci_alpha,
                    seed=seed0 + 3,
                ),
            }

            print(f"\n[Stats] Paired tests ({decoding})")
            for k, stat in block["paired_tests"][decoding].items():
                print(f"  {k}: Δ={stat['mean_diff']:+.3f} CI[{stat['ci_low']:+.3f}, {stat['ci_high']:+.3f}] p={stat['p_value']:.4g}")

        results["by_dataset"][task_name] = block

    # Save JSON/TXT
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=json_default)

    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("LEVEL-0 ACCURACY SUMMARY (decode-last-token removal, staged by generated tokens)")
    summary_lines.append("=" * 80)
    summary_lines.append(f"Model: {args.model} (dtype={args.model_dtype})  Device: {args.device}")
    summary_lines.append(f"Layer(s): {layer_indices}")
    summary_lines.append(f"cross_dim={cross_dim} shared_basis_dim={kS} rand_type={args.rand_type}")
    summary_lines.append(f"Energy ratio(shared) mean={er_s['mean']:.4f}, rand mean={er_r['mean']:.4f}")
    summary_lines.append(f"Reasoning token threshold (staged): {args.reasoning_tokens}")
    summary_lines.append(f"Removal alpha: {args.alpha_remove} (1.0 = full removal)")
    summary_lines.append("")

    with open(args.out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print("\n" + "\n".join(summary_lines))
    print("\n" + "=" * 80)
    print("[Done]")
    print(f"[Done] JSON: {args.out_json}")
    print(f"[Done] TXT : {args.out_txt}")
    print("=" * 80)

if __name__ == "__main__":
    main()
