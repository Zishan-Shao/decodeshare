# coding: utf-8
"""
disturb_energy_matched_sharedness_kmatch.py

Energy-matched controls (reviewer-friendly) for shared-subspace removal.

Goal
----
Show BOTH:
(1) Structural effect at fixed dimension k = k_shared:
    shared(k_shared) vs control_struct(k_shared)  (nonshared, top-k)
(2) Energy-matched effect with alpha fixed to 1 and control dimension k_c chosen to match
    removed energy on a calibration distribution:
    shared(k_shared, alpha=1) vs control_energy(k_c, alpha=1) where
    E||P_c h||^2 ≈ E||P_s h||^2

Additionally, evaluate with forced-choice log-probability (no "Final answer" extraction) to
separate "general generation collapse" from "decision/reasoning loss".

===============================
MODIFICATIONS (requested)
===============================
(1) STRICT DECODE-ONLY HOOK:
    The removal hook intervenes ONLY when seq_len == 1 (hs.shape[1] == 1).

(2) DECODE-ALIGNED FORCED-CHOICE PROMPT-END LOGITS:
    To make "prompt-end next-token logits" come from a seq_len==1 forward pass:
      - Prefill x1:T-1 -> past
      - Feed xT as first decode input (seq_len=1) with that past
    Candidate rollout proceeds token-by-token (each seq_len=1).

(3) ENERGY MATCHING CALIBRATION DISTRIBUTION:
    Energy matching is calibrated on PROMPT-BOUNDARY DECODE-LAST states:
      - For each prompt x1:T, collect hidden state at layer ℓ when feeding xT
        with past from prefill x1:T-1 (seq_len=1).
    This matches the forced-choice intervention point distribution.

Key fixes vs earlier buggy versions
-----------------------------------
- Forced-choice supports MULTI-token labels.
- Orthonormal bases & overlaps sanity checks.
- Energy ratio is ||P h||^2 / ||h||^2 in [0,1].
- Energy matching for alpha=1 uses K-match (choose k_c), not alpha>1.

Requires your project utilities:
  joint_subspace_large/disturb_cross_task_all_shared.py providing:
    - get_model_layers
    - compute_cross_task_subspace
    - find_fully_shared_basis_improved
    
CUDA_VISIBLE_DEVICES=1 python disturb_energy_matched_sharedness_kmatch.py \
  --model meta-llama/Llama-2-7b-chat-hf --device cuda --dtype fp32 \
  --layer 10 \
  --n_prompts 128 --calib_max_new_tokens 128 --per_task_max_states 20000 \
  --pca_var 0.95 --tau 0.001 --m_shared all \
  --eval_n 256 --max_prompt_len 512 \
  --control_basis joint_nonshared_topk \
  --seed 42 \
  --use_chat_template 0
  
"""

import os
import sys
import re
import json
import random
import math
import argparse
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------------------
# Import project utilities
# ---------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(THIS_DIR, ".."))

from decodeshare.joint_subspace_large.disturb_cross_task_all_shared import (  # noqa: E402
    get_model_layers,
    compute_cross_task_subspace,
    find_fully_shared_basis_improved,
)

# -----------------------------
# Repro / stats
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

def bootstrap_ci_mean(values: np.ndarray, iters: int, alpha: float, seed: int) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = int(values.shape[0])
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    obs = float(values.mean())
    boots = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        idx = rng.integers(0, n, size=n)
        boots[i] = values[idx].mean()
    lo = float(np.percentile(boots, 100 * (alpha / 2)))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return obs, lo, hi

def paired_bootstrap_ci_diff(baseline: np.ndarray, treatment: np.ndarray, iters: int, alpha: float, seed: int) -> Tuple[float, float, float]:
    assert baseline.shape == treatment.shape
    diffs = treatment - baseline
    rng = np.random.default_rng(seed)
    n = int(diffs.shape[0])
    obs = float(diffs.mean())
    boots = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        idx = rng.integers(0, n, size=n)
        boots[i] = diffs[idx].mean()
    lo = float(np.percentile(boots, 100 * (alpha / 2)))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return obs, lo, hi

def signflip_permutation_test(baseline: np.ndarray, treatment: np.ndarray, iters: int, seed: int) -> float:
    assert baseline.shape == treatment.shape
    diffs = treatment - baseline
    obs = float(diffs.mean())
    rng = np.random.default_rng(seed)
    n = int(diffs.shape[0])
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
        baseline_correct, treat_correct, iters=bootstrap_iters, alpha=alpha, seed=seed + 123
    )
    p = signflip_permutation_test(baseline_correct, treat_correct, iters=perm_iters, seed=seed + 456)
    return {"label": label, "mean_diff": md, "ci_low": lo, "ci_high": hi, "p_value": p}

def fmt_acc(acc: float, lo: float, hi: float) -> str:
    return f"{acc*100:.1f} [{lo*100:.1f}, {hi*100:.1f}]"

def fmt_diff(stat: Dict[str, Any]) -> str:
    return f"{stat['mean_diff']*100:+.1f} [{stat['ci_low']*100:+.1f}, {stat['ci_high']*100:+.1f}]"

# -----------------------------
# Data / prompts
# -----------------------------
@dataclass
class Example:
    dataset: str
    ex_id: str
    prompt: str
    gold: str

def safe_upper(x: Any) -> str:
    return str(x).strip().upper()

def build_calib_prompt_gsm8k(q: str) -> str:
    return (
        f"Question: {q}\n"
        "Let's think step by step.\n"
        'At the end, write exactly one line in the format: "Final answer: <number>".\n'
    )

def build_calib_prompt_csqa(q: str, choices: Dict[str, List[str]]) -> str:
    labels = choices["label"]
    texts = choices["text"]
    lines = [f"{lab}) {txt}" for lab, txt in zip(labels, texts)]
    return (
        f"Question: {q}\n"
        "Choices:\n" + "\n".join(lines) + "\n"
        "Reason step by step.\n"
        'At the end, write exactly one line in the format: "Final answer: <A/B/C/D/E>".\n'
    )

def build_calib_prompt_strategyqa(q: str) -> str:
    return (
        f"Question: {q}\n"
        "Please reason step by step.\n"
        'At the end, write exactly one line in the format: "Final answer: Yes" or "Final answer: No".\n'
    )

def build_calib_prompt_aqua(q: str, options: List[str]) -> str:
    labels = ["A", "B", "C", "D", "E"]
    lines = []
    for i, opt in enumerate(options[:5]):
        lab = labels[i]
        opt_clean = re.sub(r"^[A-E]\)?\s*[:\-]?\s*", "", str(opt).strip(), flags=re.IGNORECASE)
        lines.append(f"{lab}) {opt_clean}")
    return (
        f"Question: {q}\n"
        "Choices:\n" + "\n".join(lines) + "\n"
        "Please reason step by step.\n"
        'At the end, write exactly one line in the format: "Final answer: <A/B/C/D/E>".\n'
    )

def build_fc_prompt_csqa(q: str, choices: Dict[str, List[str]]) -> str:
    labels = choices["label"]
    texts = choices["text"]
    lines = [f"{lab}) {txt}" for lab, txt in zip(labels, texts)]
    return (
        f"Question: {q}\n"
        "Choices:\n" + "\n".join(lines) + "\n"
        "Select the correct option.\n"
        "Answer:"
    )

def build_fc_prompt_aqua(q: str, options: List[str]) -> str:
    labels = ["A", "B", "C", "D", "E"]
    lines = []
    for i, opt in enumerate(options[:5]):
        lab = labels[i]
        opt_clean = re.sub(r"^[A-E]\)?\s*[:\-]?\s*", "", str(opt).strip(), flags=re.IGNORECASE)
        lines.append(f"{lab}) {opt_clean}")
    return (
        f"Question: {q}\n"
        "Choices:\n" + "\n".join(lines) + "\n"
        "Select the correct option.\n"
        "Answer:"
    )

def build_fc_prompt_strategyqa(q: str) -> str:
    return (
        f"Question: {q}\n"
        "Answer yes or no.\n"
        "Answer:"
    )

def build_fc_prompt_arc(q: str, choices: Dict[str, List[str]]) -> str:
    labels = choices["label"]
    texts = choices["text"]
    lines = [f"{lab}) {txt}" for lab, txt in zip(labels, texts)]
    return (
        f"Question: {q}\n"
        "Choices:\n" + "\n".join(lines) + "\n"
        "Select the correct option.\n"
        "Answer:"
    )

def build_fc_prompt_openbookqa(q_stem: str, choices: Dict[str, List[str]]) -> str:
    labels = choices["label"]
    texts = choices["text"]
    lines = [f"{lab}) {txt}" for lab, txt in zip(labels, texts)]
    return (
        f"Question: {q_stem}\n"
        "Choices:\n" + "\n".join(lines) + "\n"
        "Select the correct option.\n"
        "Answer:"
    )

def build_fc_prompt_qasc(q: str, choices: Dict[str, List[str]]) -> str:
    labels = choices["label"]  # A-H
    texts = choices["text"]
    lines = [f"{lab}) {txt}" for lab, txt in zip(labels, texts)]
    return (
        f"Question: {q}\n"
        "Choices:\n" + "\n".join(lines) + "\n"
        "Select the correct option.\n"
        "Answer:"
    )

def build_fc_prompt_logiqa(context: str, query: str, options: List[str]) -> str:
    labels = ["A", "B", "C", "D"]
    lines = [f"{labels[i]}) {opt}" for i, opt in enumerate(options[:4])]
    return (
        f"Context: {context}\n"
        f"Question: {query}\n"
        "Choices:\n" + "\n".join(lines) + "\n"
        "Select the correct option.\n"
        "Answer:"
    )

def build_fc_prompt_boolq(passage: str, question: str) -> str:
    return (
        f"Passage: {passage}\n"
        f"Question: {question}\n"
        "Answer yes or no.\n"
        "Answer:"
    )


def sample_hf_split(ds_split, n: int, seed: int):
    n = min(int(n), len(ds_split))
    if n <= 0:
        return ds_split.select([])
    return ds_split.shuffle(seed=seed).select(range(n))

def load_calib_prompts(n_prompts: int, seed: int) -> Dict[str, List[str]]:
    prompts_by: Dict[str, List[str]] = {}

    ds = load_dataset("gsm8k", "main")
    split = "train" if "train" in ds else list(ds.keys())[0]
    rows = sample_hf_split(ds[split], n_prompts, seed + 1)
    prompts_by["gsm8k"] = [build_calib_prompt_gsm8k(ex["question"]) for ex in rows]

    ds = load_dataset("commonsense_qa")
    split = "train" if "train" in ds else list(ds.keys())[0]
    rows = sample_hf_split(ds[split], n_prompts, seed + 11)
    prompts_by["commonsenseqa"] = [build_calib_prompt_csqa(ex["question"], ex["choices"]) for ex in rows]

    ds = load_dataset("ChilleD/StrategyQA")
    split = "train" if "train" in ds else list(ds.keys())[0]
    rows = sample_hf_split(ds[split], n_prompts, seed + 21)
    prompts_by["strategyqa"] = [build_calib_prompt_strategyqa(ex["question"]) for ex in rows]

    ds = load_dataset("aqua_rat")
    split = "train" if "train" in ds else list(ds.keys())[0]
    rows = sample_hf_split(ds[split], n_prompts, seed + 31)
    prompts_by["aqua"] = [build_calib_prompt_aqua(ex["question"], ex["options"]) for ex in rows]

    return prompts_by

# def load_forced_choice_eval(n_eval: int, seed: int) -> Dict[str, List[Example]]:
#     out: Dict[str, List[Example]] = {}

#     ds = load_dataset("commonsense_qa")
#     split = "validation" if "validation" in ds else ("test" if "test" in ds else list(ds.keys())[0])
#     rows = sample_hf_split(ds[split], n_eval, seed + 101)
#     exs = []
#     for i, ex in enumerate(rows):
#         exs.append(Example("commonsenseqa", f"csqa-{split}-{i}", build_fc_prompt_csqa(ex["question"], ex["choices"]), safe_upper(ex["answerKey"])))
#     out["commonsenseqa"] = exs

#     ds = load_dataset("ChilleD/StrategyQA")
#     split = "test" if "test" in ds else ("validation" if "validation" in ds else list(ds.keys())[0])
#     rows = sample_hf_split(ds[split], n_eval, seed + 111)

#     def to_yesno(v: Any) -> str:
#         if isinstance(v, bool):
#             return "YES" if v else "NO"
#         if isinstance(v, (int, np.integer)):
#             return "YES" if int(v) == 1 else "NO"
#         s = str(v).strip().lower()
#         if s in ["true", "yes", "1"]:
#             return "YES"
#         if s in ["false", "no", "0"]:
#             return "NO"
#         if "yes" in s:
#             return "YES"
#         if "no" in s:
#             return "NO"
#         return ""

#     exs = []
#     for i, ex in enumerate(rows):
#         exs.append(Example("strategyqa", f"strategyqa-{split}-{i}", build_fc_prompt_strategyqa(ex["question"]), to_yesno(ex["answer"])))
#     out["strategyqa"] = exs

#     ds = load_dataset("aqua_rat")
#     split = "test" if "test" in ds else ("validation" if "validation" in ds else list(ds.keys())[0])
#     rows = sample_hf_split(ds[split], n_eval, seed + 121)

#     def get_gold(ex: dict) -> str:
#         if "correct" in ex:
#             return safe_upper(ex["correct"])
#         if "answer" in ex:
#             return safe_upper(ex["answer"])
#         return ""

#     exs = []
#     for i, ex in enumerate(rows):
#         exs.append(Example("aqua", f"aqua-{split}-{i}", build_fc_prompt_aqua(ex["question"], ex["options"]), get_gold(ex)))
#     out["aqua"] = exs

#     return out
def load_forced_choice_eval(n_eval: int, seed: int, tasks: Optional[List[str]] = None) -> Dict[str, List[Example]]:
    out: Dict[str, List[Example]] = {}
    tasks = tasks or ["commonsenseqa", "strategyqa", "aqua"]

    if "commonsenseqa" in tasks:
        ds = load_dataset("commonsense_qa")
        split = "validation" if "validation" in ds else ("test" if "test" in ds else list(ds.keys())[0])
        rows = sample_hf_split(ds[split], n_eval, seed + 101)
        exs = []
        for i, ex in enumerate(rows):
            exs.append(Example("commonsenseqa", f"csqa-{split}-{i}",
                               build_fc_prompt_csqa(ex["question"], ex["choices"]),
                               safe_upper(ex["answerKey"])))
        out["commonsenseqa"] = exs

    if "strategyqa" in tasks:
        ds = load_dataset("ChilleD/StrategyQA")
        split = "test" if "test" in ds else ("validation" if "validation" in ds else list(ds.keys())[0])
        rows = sample_hf_split(ds[split], n_eval, seed + 111)
        def to_yesno(v: Any) -> str:
            if isinstance(v, bool): return "YES" if v else "NO"
            if isinstance(v, (int, np.integer)): return "YES" if int(v) == 1 else "NO"
            s = str(v).strip().lower()
            if s in ["true", "yes", "1"]: return "YES"
            if s in ["false", "no", "0"]: return "NO"
            return ""
        exs = []
        for i, ex in enumerate(rows):
            exs.append(Example("strategyqa", f"strategyqa-{split}-{i}",
                               build_fc_prompt_strategyqa(ex["question"]),
                               to_yesno(ex["answer"])))
        out["strategyqa"] = exs

    if "aqua" in tasks:
        ds = load_dataset("aqua_rat")
        split = "test" if "test" in ds else ("validation" if "validation" in ds else list(ds.keys())[0])
        rows = sample_hf_split(ds[split], n_eval, seed + 121)
        def get_gold(ex: dict) -> str:
            if "correct" in ex: return safe_upper(ex["correct"])
            if "answer" in ex: return safe_upper(ex["answer"])
            return ""
        exs = []
        for i, ex in enumerate(rows):
            exs.append(Example("aqua", f"aqua-{split}-{i}",
                               build_fc_prompt_aqua(ex["question"], ex["options"]),
                               get_gold(ex)))
        out["aqua"] = exs

    # -------- NEW: ARC-Challenge (A-D) --------
    if "arc_challenge" in tasks:
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge")  # question/choices/answerKey :contentReference[oaicite:5]{index=5}
        split = "validation" if "validation" in ds else ("train" if "train" in ds else list(ds.keys())[0])
        rows = sample_hf_split(ds[split], n_eval, seed + 201)
        exs = []
        for i, ex in enumerate(rows):
            exs.append(Example("arc_challenge", f"arc-{split}-{i}",
                               build_fc_prompt_arc(ex["question"], ex["choices"]),
                               safe_upper(ex["answerKey"])))
        out["arc_challenge"] = exs

    # -------- NEW: OpenBookQA (A-D) --------
    if "openbookqa" in tasks:
        ds = load_dataset("allenai/openbookqa", "main")  # question_stem/choices/answerKey :contentReference[oaicite:6]{index=6}
        split = "validation" if "validation" in ds else ("train" if "train" in ds else list(ds.keys())[0])
        rows = sample_hf_split(ds[split], n_eval, seed + 211)
        exs = []
        for i, ex in enumerate(rows):
            exs.append(Example("openbookqa", f"obqa-{split}-{i}",
                               build_fc_prompt_openbookqa(ex["question_stem"], ex["choices"]),
                               safe_upper(ex["answerKey"])))
        out["openbookqa"] = exs

    # -------- NEW: QASC (A-H) --------
    if "qasc" in tasks:
        ds = load_dataset("allenai/qasc")  # choices A-H :contentReference[oaicite:7]{index=7}
        split = "validation" if "validation" in ds else ("train" if "train" in ds else list(ds.keys())[0])
        rows = sample_hf_split(ds[split], n_eval, seed + 221)
        exs = []
        for i, ex in enumerate(rows):
            exs.append(Example("qasc", f"qasc-{split}-{i}",
                               build_fc_prompt_qasc(ex["question"], ex["choices"]),
                               safe_upper(ex["answerKey"])))
        out["qasc"] = exs

    # -------- NEW: LogiQA (A-D, correct_option=0..3) --------
    if "logiqa" in tasks:
        ds = load_dataset("lucasmccabe/logiqa")  # context/query/options/correct_option :contentReference[oaicite:8]{index=8}
        split = "validation" if "validation" in ds else ("test" if "test" in ds else list(ds.keys())[0])
        rows = sample_hf_split(ds[split], n_eval, seed + 231)
        labels = ["A", "B", "C", "D"]
        exs = []
        for i, ex in enumerate(rows):
            gold_idx = int(ex["correct_option"])
            gold = labels[gold_idx] if 0 <= gold_idx < 4 else ""
            exs.append(Example("logiqa", f"logiqa-{split}-{i}",
                               build_fc_prompt_logiqa(ex["context"], ex["query"], ex["options"]),
                               gold))
        out["logiqa"] = exs

    # -------- NEW: BoolQ (YES/NO) --------
    if "boolq" in tasks:
        ds = load_dataset("google/boolq")  # question/passage/answer :contentReference[oaicite:9]{index=9}
        split = "validation" if "validation" in ds else ("train" if "train" in ds else list(ds.keys())[0])
        rows = sample_hf_split(ds[split], n_eval, seed + 241)
        exs = []
        for i, ex in enumerate(rows):
            gold = "YES" if bool(ex["answer"]) else "NO"
            exs.append(Example("boolq", f"boolq-{split}-{i}",
                               build_fc_prompt_boolq(ex["passage"], ex["question"]),
                               gold))
        out["boolq"] = exs

    return out


# -----------------------------
# Chat template helper
# -----------------------------
def maybe_apply_chat_template(tok: AutoTokenizer, user_prompt: str, use_chat_template: bool) -> str:
    if not use_chat_template:
        return user_prompt
    if not hasattr(tok, "apply_chat_template"):
        return user_prompt
    msgs = [{"role": "user", "content": user_prompt}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        return user_prompt

# -----------------------------
# Collectors
# -----------------------------
class DecodeLastTokenActivationCollector:
    """Collect last-token hidden states ONLY during decode steps (seq_len == 1)."""
    def __init__(self, layer_indices: List[int]):
        self.layer_indices = list(layer_indices)
        self._cur_task: Optional[str] = None
        self.capture_enabled: bool = False
        self.active_mask: Optional[torch.Tensor] = None  # [B] bool
        self.storage: Dict[str, Dict[int, List[np.ndarray]]] = {}

    def set_current_task(self, task: str) -> None:
        self._cur_task = task
        if task not in self.storage:
            self.storage[task] = {}

    def set_capture(self, enabled: bool, active_mask: Optional[torch.Tensor]) -> None:
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
                m = self.active_mask
                if m.dtype != torch.bool:
                    m = m.bool()
                if m.numel() == x.shape[0]:
                    x = x[m]
            if x.numel() == 0:
                return output
            self.storage.setdefault(self._cur_task, {}).setdefault(layer_idx, []).append(
                x.detach().float().cpu().numpy()
            )
            return output
        return _hook

    def get_task_activations(self, task: str, layer_idx: int) -> Optional[np.ndarray]:
        chunks = self.storage.get(task, {}).get(layer_idx, [])
        if not chunks:
            return None
        return np.concatenate(chunks, axis=0)

class PromptBoundaryDecodeLastTokenCollector:
    """
    Collect last-token hidden states on the PROMPT-BOUNDARY decode step:
      - Prefill x1:T-1 -> past
      - Decode with xT (seq_len=1) -> collect h at desired layer
    """
    def __init__(self, layer_indices: List[int]):
        self.layer_indices = list(layer_indices)
        self._cur_task: Optional[str] = None
        self.storage: Dict[str, Dict[int, List[np.ndarray]]] = {}

    def set_current_task(self, task: str) -> None:
        self._cur_task = task
        if task not in self.storage:
            self.storage[task] = {}

    def make_hook(self, layer_idx: int):
        def _hook(module, inputs, output):
            if self._cur_task is None:
                return output
            hs = output[0] if isinstance(output, tuple) else output
            if not isinstance(hs, torch.Tensor) or hs.ndim != 3:
                return output
            # Only record seq_len==1 calls (prompt-boundary decode)
            if hs.shape[1] != 1:
                return output
            x = hs[:, -1, :]
            self.storage.setdefault(self._cur_task, {}).setdefault(layer_idx, []).append(
                x.detach().float().cpu().numpy()
            )
            return output
        return _hook

    def get_task_activations(self, task: str, layer_idx: int) -> Optional[np.ndarray]:
        chunks = self.storage.get(task, {}).get(layer_idx, [])
        if not chunks:
            return None
        return np.concatenate(chunks, axis=0)

# -----------------------------
# Collect decode states by greedy cached decoding (for basis estimation)
# -----------------------------
@torch.no_grad()
def collect_decode_last_token_states(
    model,
    tok: AutoTokenizer,
    prompts: List[str],
    collector: DecodeLastTokenActivationCollector,
    *,
    batch_size: int,
    max_new_tokens: int,
    max_prompt_len: int,
    use_chat_template: bool,
) -> None:
    device = next(model.parameters()).device
    model.eval()
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    eos = tok.eos_token_id

    for i in tqdm(range(0, len(prompts), batch_size), desc="CollectDecode"):
        batch_raw = prompts[i:i+batch_size]
        batch = [maybe_apply_chat_template(tok, p, use_chat_template) for p in batch_raw]
        inputs = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_prompt_len).to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        B = int(input_ids.shape[0])

        unfinished = torch.ones(B, dtype=torch.bool, device=device)

        collector.set_capture(False, None)
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        logits = out.logits[:, -1, :]
        past = out.past_key_values

        for _ in range(int(max_new_tokens)):
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            next_token = torch.where(
                unfinished.unsqueeze(-1),
                next_token,
                torch.full_like(next_token, eos),
            )
            unfinished = unfinished & (next_token.squeeze(-1) != eos)
            if not bool(unfinished.any().item()):
                break

            attention_mask = torch.cat(
                [attention_mask, torch.ones((B, 1), device=device, dtype=attention_mask.dtype)],
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
# MOD (3): Collect prompt-boundary decode-last states for energy calibration
# -----------------------------
@torch.no_grad()
def collect_prompt_boundary_decode_last_states(
    model,
    tok: AutoTokenizer,
    prompts: List[str],
    *,
    batch_size: int,
    max_prompt_len: int,
    use_chat_template: bool,
) -> None:
    """
    Execute prompt-boundary decode forward passes:
      - prefill input_ids[:, :-1] with use_cache=True
      - decode input_ids[:, -1:] (seq_len=1) with past from prefill
    The hidden states for the decode call are captured by a forward hook (seq_len==1).
    """
    device = next(model.parameters()).device
    model.eval()
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    for i in tqdm(range(0, len(prompts), batch_size), desc="PromptBoundaryDecode"):
        batch_raw = prompts[i:i+batch_size]
        batch = [maybe_apply_chat_template(tok, p, use_chat_template) for p in batch_raw]
        inputs = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_prompt_len).to(device)
        input_ids = inputs["input_ids"]            # [B, L]
        attention_mask = inputs["attention_mask"]  # [B, L]

        lengths = attention_mask.long().sum(dim=1)  # [B]
        idx_long = torch.nonzero(lengths > 1, as_tuple=False).view(-1)
        idx_short = torch.nonzero(lengths <= 1, as_tuple=False).view(-1)

        if idx_long.numel() > 0:
            ids_long = input_ids[idx_long]
            mask_long = attention_mask[idx_long]

            # Prefill x1:T-1
            out_pre = model(
                input_ids=ids_long[:, :-1],
                attention_mask=mask_long[:, :-1],
                use_cache=True,
            )
            past = out_pre.past_key_values

            # Prompt-boundary decode step: feed xT (seq_len=1)
            _ = model(
                input_ids=ids_long[:, -1:],
                attention_mask=mask_long,  # full prompt mask
                past_key_values=past,
                use_cache=True,
            )

        if idx_short.numel() > 0:
            # Rare; prompt length 1: directly run seq_len==1 (also collected)
            ids_short = input_ids[idx_short, -1:]
            mask_short = attention_mask[idx_short, -1:]
            _ = model(input_ids=ids_short, attention_mask=mask_short, use_cache=True)

# -----------------------------
# Basis helpers
# -----------------------------
def subsample_rows(x: np.ndarray, n_max: int, seed: int) -> np.ndarray:
    if n_max is None or n_max <= 0 or x.shape[0] <= n_max:
        return x
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.shape[0], size=int(n_max), replace=False)
    return x[idx]

def balance_tasks(task_to_x: Dict[str, np.ndarray], seed: int) -> Tuple[Dict[str, np.ndarray], int]:
    sizes = {t: int(x.shape[0]) for t, x in task_to_x.items()}
    min_n = min(sizes.values())
    out = {}
    for t, x in task_to_x.items():
        out[t] = subsample_rows(x, min_n, seed=stable_int_seed(seed, "balance", t))
    return out, min_n

def qr_orthonormal(A: np.ndarray) -> np.ndarray:
    if A.size == 0:
        raise ValueError("Empty matrix for QR.")
    Q, _ = np.linalg.qr(A.astype(np.float64, copy=False))
    Q = Q.astype(np.float32, copy=False)
    return Q

def project_out(A: np.ndarray, Q_ortho: np.ndarray) -> np.ndarray:
    if Q_ortho is None or Q_ortho.size == 0:
        return A
    return A - Q_ortho @ (Q_ortho.T @ A)

def make_random_orthonormal(D: int, k: int, seed: int, orthogonal_to: Optional[np.ndarray] = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal(size=(D, int(k))).astype(np.float32)
    if orthogonal_to is not None and orthogonal_to.size > 0:
        A = project_out(A, orthogonal_to.astype(np.float32, copy=False))
    Q = qr_orthonormal(A)
    return Q[:, :k]

def orthonormality_max_offdiag(Q: np.ndarray) -> float:
    G = Q.T @ Q
    I = np.eye(G.shape[0], dtype=G.dtype)
    off = np.abs(G - I)
    off[np.diag_indices_from(off)] = 0.0
    return float(off.max()) if off.size else 0.0

def max_overlap(Qa: np.ndarray, Qb: np.ndarray) -> float:
    if Qa.size == 0 or Qb.size == 0:
        return 0.0
    M = Qa.T @ Qb
    return float(np.abs(M).max())

def projection_energy_stats(H: np.ndarray, Q: np.ndarray) -> Dict[str, float]:
    H = H.astype(np.float32, copy=False)
    Q = Q.astype(np.float32, copy=False)
    Z = H @ Q
    proj_e = np.sum(Z * Z, axis=1)
    tot_e = np.sum(H * H, axis=1) + 1e-12
    ratio = proj_e / tot_e
    return {
        "mean_ratio": float(ratio.mean()),
        "p50_ratio": float(np.percentile(ratio, 50)),
        "p95_ratio": float(np.percentile(ratio, 95)),
        "mean_energy": float(proj_e.mean()),
    }

# -----------------------------
# Intervention hook with stats
# -----------------------------
class HookStats:
    def __init__(self, name: str):
        self.name = name
        self.calls = 0
        self.intervened = 0
        self.skipped = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "calls": int(self.calls),
            "intervened": int(self.intervened),
            "skipped": int(self.skipped),
        }

class LastTokenRemovalHook:
    """
    Decode-only last-token removal:
        h <- h - alpha * Q Q^T h
    applied ONLY when seq_len == 1 (hs.shape[1] == 1).
    """
    def __init__(self, Q_np: np.ndarray, alpha: float, stats: HookStats):
        self.alpha = float(alpha)
        self.stats = stats
        self.Q_cpu = torch.tensor(Q_np, dtype=torch.float32, device="cpu").contiguous()
        self.Q_device: Optional[torch.Tensor] = None

    def _Q(self, device: torch.device) -> torch.Tensor:
        if self.Q_device is None or self.Q_device.device != device:
            self.Q_device = self.Q_cpu.to(device=device, dtype=torch.float32)
        return self.Q_device

    def __call__(self, module, inputs, output):
        self.stats.calls += 1
        hs = output[0] if isinstance(output, tuple) else output
        if not isinstance(hs, torch.Tensor) or hs.ndim != 3:
            return output

        # MOD (1): strict decode-only
        if hs.shape[1] != 1:
            self.stats.skipped += 1
            return output

        self.stats.intervened += 1

        Q = self._Q(hs.device)
        hs2 = hs.clone()
        x = hs2[:, -1, :].float()
        proj = (x @ Q) @ Q.T
        x_new = x - self.alpha * proj
        hs2[:, -1, :] = x_new.to(dtype=hs2.dtype)

        if isinstance(output, tuple):
            return (hs2,) + output[1:]
        return hs2

def register_removal_hooks(model, layer_indices: List[int], Q: np.ndarray, alpha: float, name: str) -> Tuple[List[Any], Dict[int, HookStats]]:
    layers, _ = get_model_layers(model)
    handles = []
    stats_by_layer: Dict[int, HookStats] = {}
    for li in layer_indices:
        if li >= len(layers):
            print(f"[Hook][WARN] layer={li} out of range, skipping")
            continue
        st = HookStats(f"{name}@{li}")
        hk = LastTokenRemovalHook(Q, alpha=alpha, stats=st)
        h = layers[li].register_forward_hook(hk)
        handles.append(h)
        stats_by_layer[li] = st
    return handles, stats_by_layer

def remove_hooks(handles: List[Any]) -> None:
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass


# # -----------------------------
# # Forced-choice evaluation (decode-aligned)
# # -----------------------------
# def get_candidates(task: str) -> Tuple[List[str], List[str]]:
#     if task in ["commonsenseqa", "aqua"]:
#         labels = ["A", "B", "C", "D", "E"]
#         texts = [f" {x}" for x in labels]
#         return labels, texts
#     if task == "strategyqa":
#         labels = ["YES", "NO"]
#         texts = [" Yes", " No"]
#         return labels, texts
#     raise ValueError(f"Unknown forced-choice task: {task}")

def get_candidates(task: str) -> Tuple[List[str], List[str]]:
    # 5-way
    if task in ["commonsenseqa", "aqua"]:
        labels = ["A", "B", "C", "D", "E"]
        texts = [f" {x}" for x in labels]
        return labels, texts

    # 4-way
    if task in ["arc_challenge", "openbookqa", "logiqa"]:
        labels = ["A", "B", "C", "D"]
        texts = [f" {x}" for x in labels]
        return labels, texts

    # 8-way (QASC)
    if task in ["qasc"]:
        labels = ["A", "B", "C", "D", "E", "F", "G", "H"]
        texts = [f" {x}" for x in labels]
        return labels, texts

    # yes/no
    if task in ["strategyqa", "boolq"]:
        labels = ["YES", "NO"]
        texts = [" Yes", " No"]
        return labels, texts

    raise ValueError(f"Unknown forced-choice task: {task}")


@torch.no_grad()
def forced_choice_eval(
    model,
    tok: AutoTokenizer,
    examples: List[Example],
    *,
    layer_indices: List[int],
    condition_name: str,
    Q: Optional[np.ndarray],
    alpha: float,
    batch_size: int,
    max_prompt_len: int,
    use_chat_template: bool,
    bootstrap_iters: int,
    perm_iters: int,
    ci_alpha: float,
    seed: int,
) -> Dict[str, Any]:
    """
    MOD (2): Decode-aligned prompt-end logits.
    For prompt tokens x1:T:
      - prefill x1:T-1 -> past
      - decode step with xT (seq_len=1) -> logits for next token after prompt
    Candidate rollout: seq_len=1 steps with cache.
    """
    device = next(model.parameters()).device
    model.eval()
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    task = examples[0].dataset if examples else "unknown"
    labels, cand_texts = get_candidates(task)
    cand_ids = [tok.encode(ct, add_special_tokens=False) for ct in cand_texts]

    handles: List[Any] = []
    hookstats: Dict[int, HookStats] = {}
    if Q is not None:
        handles, hookstats = register_removal_hooks(model, layer_indices, Q, alpha=alpha, name=condition_name)

    correct: List[int] = []
    preds: List[str] = []

    def score_group(input_ids_g: torch.Tensor, attention_mask_g: torch.Tensor) -> torch.Tensor:
        Bg = int(input_ids_g.shape[0])
        Lg = int(input_ids_g.shape[1])

        if Lg == 1:
            out_last = model(input_ids=input_ids_g, attention_mask=attention_mask_g, use_cache=True)
            logits0 = out_last.logits[:, -1, :]
            past0 = out_last.past_key_values
            attn0 = attention_mask_g
        else:
            out_pre = model(
                input_ids=input_ids_g[:, :-1],
                attention_mask=attention_mask_g[:, :-1],
                use_cache=True,
            )
            past_pre = out_pre.past_key_values
            out_last = model(
                input_ids=input_ids_g[:, -1:],
                attention_mask=attention_mask_g,
                past_key_values=past_pre,
                use_cache=True,
            )
            logits0 = out_last.logits[:, -1, :]
            past0 = out_last.past_key_values
            attn0 = attention_mask_g

        scores_g = torch.zeros((Bg, len(labels)), device=device, dtype=torch.float32)

        for ci, tok_list in enumerate(cand_ids):
            logits_cur = logits0
            past_cur = past0
            attn_cur = attn0
            score = torch.zeros((Bg,), device=device, dtype=torch.float32)

            for j, tid in enumerate(tok_list):
                logp = torch.log_softmax(logits_cur.float(), dim=-1)
                score += logp[:, int(tid)]
                if j < len(tok_list) - 1:
                    inp = torch.full((Bg, 1), int(tid), device=device, dtype=input_ids_g.dtype)
                    attn_cur = torch.cat(
                        [attn_cur, torch.ones((Bg, 1), device=device, dtype=attn_cur.dtype)],
                        dim=1,
                    )
                    out2 = model(
                        input_ids=inp,
                        attention_mask=attn_cur,
                        past_key_values=past_cur,
                        use_cache=True,
                    )
                    logits_cur = out2.logits[:, -1, :]
                    past_cur = out2.past_key_values

            scores_g[:, ci] = score

        return scores_g

    try:
        for i in tqdm(range(0, len(examples), batch_size), desc=f"ForcedChoice({task},{condition_name})"):
            batch_ex = examples[i:i+batch_size]
            prompts_raw = [ex.prompt for ex in batch_ex]
            prompts = [maybe_apply_chat_template(tok, p, use_chat_template) for p in prompts_raw]

            inputs = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_prompt_len).to(device)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            B = int(input_ids.shape[0])
            lengths = attention_mask.long().sum(dim=1)
            idx_long = torch.nonzero(lengths > 1, as_tuple=False).view(-1)
            idx_short = torch.nonzero(lengths <= 1, as_tuple=False).view(-1)

            scores = torch.zeros((B, len(labels)), device=device, dtype=torch.float32)

            if idx_long.numel() > 0:
                scores[idx_long] = score_group(input_ids[idx_long], attention_mask[idx_long])

            if idx_short.numel() > 0:
                scores[idx_short] = score_group(
                    input_ids[idx_short, -1:].contiguous(),
                    attention_mask[idx_short, -1:].contiguous(),
                )

            pred_idx = torch.argmax(scores, dim=-1).tolist()
            for b, ex in enumerate(batch_ex):
                plab = labels[int(pred_idx[b])]
                preds.append(plab)
                correct.append(int(plab.upper() == ex.gold.upper()))

        correct_arr = np.array(correct, dtype=np.float32)
        acc, lo, hi = bootstrap_ci_mean(correct_arr, iters=bootstrap_iters, alpha=ci_alpha, seed=seed)

        return {
            "task": task,
            "condition": condition_name,
            "alpha": float(alpha),
            "accuracy": float(acc),
            "ci_low": float(lo),
            "ci_high": float(hi),
            "correct": correct_arr.tolist(),
            "preds": preds,
            "hookstats": {int(k): v.to_dict() for k, v in hookstats.items()},
        }
    finally:
        remove_hooks(handles)

# -----------------------------
# Model loading
# -----------------------------
def load_model_and_tokenizer(model_name: str, device: str, dtype: str):
    if dtype == "fp32":
        torch_dtype = torch.float32
    elif dtype == "fp16":
        torch_dtype = torch.float16
    else:
        raise ValueError("dtype must be fp32 or fp16")

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
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--n_prompts", type=int, default=128)
    ap.add_argument("--calib_max_new_tokens", type=int, default=128)
    ap.add_argument("--per_task_max_states", type=int, default=20000)
    ap.add_argument("--pca_var", type=float, default=0.95)
    ap.add_argument("--tau", type=float, default=0.001)
    ap.add_argument("--m_shared", type=str, default="all")  # "all" or integer
    ap.add_argument("--control_basis", type=str, default="joint_nonshared_topk", choices=["joint_nonshared_topk"])
    ap.add_argument("--eval_n", type=int, default=2048)
    ap.add_argument("--max_prompt_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--bootstrap_iters", type=int, default=5000)
    ap.add_argument("--perm_iters", type=int, default=10000)
    ap.add_argument("--ci_alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_chat_template", type=int, default=0)
    ap.add_argument("--alpha_shared_base", type=float, default=1.0)
    ap.add_argument("--out_json", type=str, default=os.path.join(THIS_DIR, "energy_kmatch_results.json"))
    ap.add_argument("--out_txt", type=str, default=os.path.join(THIS_DIR, "energy_kmatch_summary.txt"))
    ap.add_argument(
        "--eval_tasks",
        type=str,
        default="commonsenseqa,strategyqa,aqua",
        help="Comma-separated eval task names (forced-choice): commonsenseqa,strategyqa,aqua,arc_challenge,openbookqa,qasc,logiqa,boolq",
    )

    args = ap.parse_args()

    set_global_seed(args.seed)

    layer_indices = [int(args.layer)]
    use_chat_template = bool(int(args.use_chat_template))

    model, tok = load_model_and_tokenizer(args.model, args.device, args.dtype)
    hidden_dim = getattr(model.config, "hidden_size", None) or getattr(model.config, "n_embd", None)
    if hidden_dim is None:
        raise RuntimeError("Cannot infer hidden_dim from model.config")

    print(f"[Env] model={args.model} device={args.device} dtype={args.dtype} hidden_dim={hidden_dim} layer={layer_indices}")
    print(f"[Data] n_prompts={args.n_prompts} eval_n={args.eval_n} use_chat_template={int(use_chat_template)}")

    # 1) Load calibration prompts + eval examples
    calib_prompts_by = load_calib_prompts(args.n_prompts, seed=args.seed)
    #eval_by = load_forced_choice_eval(args.eval_n, seed=args.seed)
    eval_tasks = [t.strip() for t in args.eval_tasks.split(",") if t.strip()]
    eval_by = load_forced_choice_eval(args.eval_n, seed=args.seed, tasks=eval_tasks)
    print(f"[Data] eval forced-choice tasks={list(eval_by.keys())}")

    print(f"[Data] calib tasks={list(calib_prompts_by.keys())}")
    print(f"[Data] eval forced-choice tasks={list(eval_by.keys())}")

    # 2) Collect decode last-token states (basis estimation)
    layers, _ = get_model_layers(model)
    dec_col = DecodeLastTokenActivationCollector(layer_indices)

    handles = []
    for li in layer_indices:
        if li >= len(layers):
            print(f"[Collect][WARN] layer={li} out of range; skipping")
            continue
        handles.append(layers[li].register_forward_hook(dec_col.make_hook(li)))

    try:
        for task, prompts in calib_prompts_by.items():
            print(f"[CollectDecode] task={task} prompts={len(prompts)}")
            dec_col.set_current_task(task)
            collect_decode_last_token_states(
                model, tok, prompts, dec_col,
                batch_size=args.batch_size,
                max_new_tokens=args.calib_max_new_tokens,
                max_prompt_len=args.max_prompt_len,
                use_chat_template=use_chat_template,
            )
    finally:
        remove_hooks(handles)
        dec_col.set_capture(False, None)

    task_acts: Dict[str, Dict[int, np.ndarray]] = {}
    for task in calib_prompts_by.keys():
        layer_dict: Dict[int, np.ndarray] = {}
        for li in layer_indices:
            acts = dec_col.get_task_activations(task, li)
            if acts is None or acts.shape[0] == 0:
                continue
            acts = acts.astype(np.float32, copy=False)
            acts = subsample_rows(acts, args.per_task_max_states, seed=stable_int_seed(args.seed, "subsample_decode", task, li))
            layer_dict[li] = acts
            print(f"[CollectDecode] task={task} layer={li} states={acts.shape[0]} x {acts.shape[1]}")
        if layer_dict:
            task_acts[task] = layer_dict
    if not task_acts:
        raise RuntimeError("No decode activations collected.")

    li0 = layer_indices[0]
    task_to_x = {t: task_acts[t][li0] for t in task_acts.keys()}
    task_to_x_bal, n_bal = balance_tasks(task_to_x, seed=args.seed)
    print(f"[Fair] balanced decode states per task = {n_bal}")
    task_acts_bal = {t: {li0: x} for t, x in task_to_x_bal.items()}

    # 3) Cross-task subspace + shared indices
    joint_subspace, cross_dim, contributions, full_pca_info = compute_cross_task_subspace(
        task_acts_bal,
        variance_threshold=args.pca_var,
        min_dim=1,
        max_dim=hidden_dim,
        return_full_pca=True,
    )
    if joint_subspace is None or cross_dim <= 0:
        raise RuntimeError("compute_cross_task_subspace failed.")
    joint_subspace = joint_subspace.astype(np.float32, copy=False)

    tasks = list(task_acts_bal.keys())
    if args.m_shared.lower() == "all":
        m_shared = len(tasks)
    else:
        m_shared = int(args.m_shared)

    shared_idx = find_fully_shared_basis_improved(
        contributions,
        tasks,
        cross_dim,
        min_tasks_shared=m_shared,
        relative_threshold=args.tau,
        top_k_components=cross_dim,
    )
    if not shared_idx:
        raise RuntimeError("No shared basis found. Try smaller tau or smaller m_shared.")

    shared_idx = sorted(shared_idx)
    k_shared = len(shared_idx)
    print("\n" + "=" * 80)
    print("[Subspace]")
    print(f"  cross_dim={cross_dim} shared_k={k_shared} tau={args.tau} m_shared={args.m_shared}")
    print("=" * 80)

    B_shared_raw = joint_subspace[:, shared_idx]
    Q_shared = qr_orthonormal(B_shared_raw)

    nonshared_idx = [i for i in range(cross_dim) if i not in set(shared_idx)]
    if len(nonshared_idx) < k_shared:
        raise RuntimeError("Nonshared pool too small for structural control.")

    B_pool_raw = joint_subspace[:, nonshared_idx]
    B_pool_ortho = project_out(B_pool_raw, Q_shared)
    Q_pool = qr_orthonormal(B_pool_ortho)
    K_pool = int(Q_pool.shape[1])

    Q_ctrl_struct = Q_pool[:, :k_shared]

    # 4) MOD (3): Collect PROMPT-BOUNDARY DECODE-last states for energy matching calibration
    pb_col = PromptBoundaryDecodeLastTokenCollector(layer_indices)
    layers, _ = get_model_layers(model)
    handles = []
    for li in layer_indices:
        if li >= len(layers):
            continue
        handles.append(layers[li].register_forward_hook(pb_col.make_hook(li)))

    try:
        for task, prompts in calib_prompts_by.items():
            pb_col.set_current_task(task)
            collect_prompt_boundary_decode_last_states(
                model, tok, prompts,
                batch_size=args.batch_size,
                max_prompt_len=args.max_prompt_len,
                use_chat_template=use_chat_template,
            )
    finally:
        remove_hooks(handles)

    pb_task_to_x: Dict[str, np.ndarray] = {}
    for task in calib_prompts_by.keys():
        acts = pb_col.get_task_activations(task, li0)
        if acts is None or acts.shape[0] == 0:
            raise RuntimeError(f"No prompt-boundary decode states collected for task={task}")
        acts = acts.astype(np.float32, copy=False)
        pb_task_to_x[task] = acts
        print(f"[PromptBoundaryDecode] task={task} layer={li0} states={acts.shape[0]} x {acts.shape[1]}")

    pb_task_to_x_bal, n_pb_bal = balance_tasks(pb_task_to_x, seed=args.seed + 999)
    H_calib = np.concatenate([pb_task_to_x_bal[t] for t in tasks], axis=0)
    print(f"[Fair] balanced prompt-boundary decode-last states per task = {n_pb_bal} pooled={H_calib.shape[0]} x {H_calib.shape[1]}")

    # 5) K-match (alpha=1) to match removed energy on H_calib
    stats_shared = projection_energy_stats(H_calib, Q_shared)
    target_removed_mean = stats_shared["mean_energy"]
    
    # (NEW) Control-1: alpha-scaling mean-match (decode-aligned on H_calib)
    # Match mean removed energy at fixed dimension k = k_shared using alpha_C (may be > 1).
    alpha_shared = float(args.alpha_shared_base)
    stats_ctrl_struct = projection_energy_stats(H_calib, Q_ctrl_struct)  # ensure available before alpha-match
    Es = float(stats_shared["mean_energy"])
    Ec_struct = float(stats_ctrl_struct["mean_energy"])
    alpha_ctrl = alpha_shared * math.sqrt(Es / max(Ec_struct, 1e-12))
    removed_shared_mean = (alpha_shared ** 2) * Es
    removed_ctrl_alpha_mean = (alpha_ctrl ** 2) * Ec_struct
    print(f"  alpha_shared={alpha_shared} alpha_ctrl={alpha_ctrl} removed_shared_mean={removed_shared_mean} removed_ctrl_alpha_mean={removed_ctrl_alpha_mean}")


    Z_pool = H_calib @ Q_pool
    cum_energy = np.cumsum(Z_pool * Z_pool, axis=1)
    mean_by_k = cum_energy.mean(axis=0)

    k_c = int(np.argmin(np.abs(mean_by_k - target_removed_mean)) + 1)
    k_c = max(k_c, k_shared)
    k_c = min(k_c, K_pool)
    Q_ctrl_energy = Q_pool[:, :k_c]

    Q_rand_struct = make_random_orthonormal(hidden_dim, k_shared, seed=args.seed + 1234, orthogonal_to=Q_shared)
    Q_rand_energy = make_random_orthonormal(hidden_dim, k_c, seed=args.seed + 5678, orthogonal_to=Q_shared)

    # 6) Sanity checks on H_calib
    # stats_ctrl_struct = projection_energy_stats(H_calib, Q_ctrl_struct)
    stats_ctrl_energy = projection_energy_stats(H_calib, Q_ctrl_energy)

    print("\n" + "=" * 80)
    print("[Sanity]")
    print(f"  cross_dim={cross_dim} shared_k={k_shared} control_basis={args.control_basis} energy_match=K-match(alpha=1)")
    print(f"  Q_pool dim={K_pool}, k_c={k_c}")
    print(f"  Orthonormality max offdiag: shared={orthonormality_max_offdiag(Q_shared):.3e}, ctrl_struct={orthonormality_max_offdiag(Q_ctrl_struct):.3e}, ctrl_energy={orthonormality_max_offdiag(Q_ctrl_energy):.3e}, rand={orthonormality_max_offdiag(Q_rand_struct):.3e}")
    print(f"  Max overlap |Q_shared^T Q_ctrl_struct| = {max_overlap(Q_shared, Q_ctrl_struct):.3e}")
    print(f"  Max overlap |Q_shared^T Q_ctrl_energy| = {max_overlap(Q_shared, Q_ctrl_energy):.3e}")
    print(f"  Max overlap |Q_shared^T Q_rand|        = {max_overlap(Q_shared, Q_rand_struct):.3e}")
    print("  Energy ratio on CALIBRATION PROMPT-BOUNDARY DECODE-last states (forced-choice prompt-end source):")
    print(f"    shared      mean={stats_shared['mean_ratio']:.4f} (p50={stats_shared['p50_ratio']:.4f}, p95={stats_shared['p95_ratio']:.4f})  mean||P h||^2={stats_shared['mean_energy']:.4e}")
    print(f"    ctrl_struct mean={stats_ctrl_struct['mean_ratio']:.4f} (p50={stats_ctrl_struct['p50_ratio']:.4f}, p95={stats_ctrl_struct['p95_ratio']:.4f})  mean||P h||^2={stats_ctrl_struct['mean_energy']:.4e}")
    print(f"    ctrl_energy mean={stats_ctrl_energy['mean_ratio']:.4f} (p50={stats_ctrl_energy['p50_ratio']:.4f}, p95={stats_ctrl_energy['p95_ratio']:.4f})  mean||P h||^2={stats_ctrl_energy['mean_energy']:.4e}")
    
    print("  (Control-1) Alpha-scaling mean match at fixed k=k_shared (decode-aligned, may allow alpha>1):")
    print(f"    alpha_shared={alpha_shared:.4f}  removed_mean(shared)=alpha^2*E||P_s h||^2={removed_shared_mean:.4e}")
    print(f"    alpha_ctrl  ={alpha_ctrl:.4f}  removed_mean(ctrl)  =alpha^2*E||P_c h||^2={removed_ctrl_alpha_mean:.4e}")
    if alpha_ctrl > 2.5:
        print("  [Sanity][WARN] alpha_ctrl is quite large (over-subtraction risk). Consider reporting K-match as primary.")

    print("  Removed-energy mean match check (alpha=1 => removed_mean = E||P h||^2):")
    print(f"    target(shared) removed_mean={target_removed_mean:.4e}")
    print(f"    ctrl_energy    removed_mean={stats_ctrl_energy['mean_energy']:.4e}  (ratio={stats_ctrl_energy['mean_energy']/max(target_removed_mean,1e-12):.4f})")
    print("=" * 80)

    # 7) Forced-choice evaluation with paired stats
    conditions: List[Tuple[str, Optional[np.ndarray], float]] = [
        ("baseline", None, 0.0),
        ("shared_alpha", Q_shared, alpha_shared),
        ("ctrl_alpha", Q_ctrl_struct, alpha_ctrl),
        ("shared_full", Q_shared, 1.0),
        ("ctrl_struct", Q_ctrl_struct, 1.0),
        ("ctrl_energy", Q_ctrl_energy, 1.0),
        ("rand_struct", Q_rand_struct, 1.0),
        ("rand_energy", Q_rand_energy, 1.0),
    ]

    results: Dict[str, Any] = {
        "config": {
            "model": args.model,
            "device": args.device,
            "dtype": args.dtype,
            "layer_indices": layer_indices,
            "seed": int(args.seed),
            "n_prompts": int(args.n_prompts),
            "calib_max_new_tokens": int(args.calib_max_new_tokens),
            "per_task_max_states": int(args.per_task_max_states),
            "pca_var": float(args.pca_var),
            "tau": float(args.tau),
            "m_shared": args.m_shared,
            "control_basis": args.control_basis,
            "eval_n": int(args.eval_n),
            "max_prompt_len": int(args.max_prompt_len),
            "batch_size": int(args.batch_size),
            "bootstrap_iters": int(args.bootstrap_iters),
            "perm_iters": int(args.perm_iters),
            "ci_alpha": float(args.ci_alpha),
            "use_chat_template": int(use_chat_template),
            "cross_dim": int(cross_dim),
            "k_shared": int(k_shared),
            "k_c": int(k_c),
            "alpha_shared_base": float(alpha_shared),
            "alpha_ctrl_alpha_match": float(alpha_ctrl),
            "energy_match_space": "prompt_boundary_decode_last",  # MOD (3)
            "forced_choice_decode_aligned": True,  # MOD (2)
            "hook_decode_only": True,              # MOD (1)
        },
        "sanity": {
            "overlap_shared_ctrl_struct": max_overlap(Q_shared, Q_ctrl_struct),
            "overlap_shared_ctrl_energy": max_overlap(Q_shared, Q_ctrl_energy),
            "energy_shared": stats_shared,
            "energy_ctrl_struct": stats_ctrl_struct,
            "energy_ctrl_energy": stats_ctrl_energy,
            "alpha_match": {
                "alpha_shared": float(alpha_shared),
                "alpha_ctrl": float(alpha_ctrl),
                "removed_shared_mean": float(removed_shared_mean),
                "removed_ctrl_mean": float(removed_ctrl_alpha_mean),
            },
        },
        "by_task": {},
    }

    summary_lines: List[str] = []
    summary_lines.append("=" * 80)
    summary_lines.append("ENERGY K-MATCHED CONTROL (alpha=1) + FORCED-CHOICE EVAL")
    summary_lines.append("=" * 80)
    summary_lines.append(f"Model={args.model} dtype={args.dtype} device={args.device} layer={layer_indices}")
    summary_lines.append(f"cross_dim={cross_dim} k_shared={k_shared} k_c={k_c} tau={args.tau} m_shared={args.m_shared}")
    summary_lines.append(f"Alpha-match (fixed k=k_shared): alpha_shared={alpha_shared:.4f}, alpha_ctrl={alpha_ctrl:.4f}")
    summary_lines.append(f"EnergyRatio(shared) mean={stats_shared['mean_ratio']:.4f}, ctrl_struct mean={stats_ctrl_struct['mean_ratio']:.4f}, ctrl_energy mean={stats_ctrl_energy['mean_ratio']:.4f}")
    summary_lines.append(f"RemovedEnergyMean(shared)={stats_shared['mean_energy']:.4e}, ctrl_energy={stats_ctrl_energy['mean_energy']:.4e}")
    summary_lines.append(f"MaxOverlap |Q_s^T Q_c_struct|={max_overlap(Q_shared, Q_ctrl_struct):.3e}, |Q_s^T Q_c_energy|={max_overlap(Q_shared, Q_ctrl_energy):.3e}")
    summary_lines.append("")

    for task, exs in eval_by.items():
        print("\n" + "-" * 80)
        print(f"[ForcedChoice] task={task} n={len(exs)}")
        print("-" * 80)

        block: Dict[str, Any] = {"n": int(len(exs)), "runs": {}, "paired": {}}

        for cname, Qc, alpha in conditions:
            run = forced_choice_eval(
                model, tok, exs,
                layer_indices=layer_indices,
                condition_name=cname,
                Q=Qc,
                alpha=alpha,
                batch_size=args.batch_size,
                max_prompt_len=args.max_prompt_len,
                use_chat_template=use_chat_template,
                bootstrap_iters=args.bootstrap_iters,
                perm_iters=args.perm_iters,
                ci_alpha=args.ci_alpha,
                seed=stable_int_seed(args.seed, task, cname),
            )
            block["runs"][cname] = run
            print(f"  {cname:11s} acc={fmt_acc(run['accuracy'], run['ci_low'], run['ci_high'])} alpha={run.get('alpha', float('nan')):.4g} hook={run.get('hookstats', {})}")
            #print(f"  {cname:11s} acc={fmt_acc(run['accuracy'], run['ci_low'], run['ci_high'])} hook={run.get('hookstats', {})}")

        base = np.array(block["runs"]["baseline"]["correct"], dtype=np.float32)
        def arr(name: str) -> np.ndarray:
            return np.array(block["runs"][name]["correct"], dtype=np.float32)

        seed0 = stable_int_seed(args.seed, task, "paired")
        block["paired"] = {
            "shared_alpha_vs_base": summarize_paired(base, arr("shared_alpha"), f"{task}:shared_alpha_vs_base", args.bootstrap_iters, args.perm_iters, args.ci_alpha, seed0 + 11),
            "ctrl_alpha_vs_base": summarize_paired(base, arr("ctrl_alpha"), f"{task}:ctrl_alpha_vs_base", args.bootstrap_iters, args.perm_iters, args.ci_alpha, seed0 + 12),
            "shared_alpha_vs_ctrl_alpha": summarize_paired(arr("ctrl_alpha"), arr("shared_alpha"), f"{task}:shared_alpha_vs_ctrl_alpha", args.bootstrap_iters, args.perm_iters, args.ci_alpha, seed0 + 13),
            "shared_vs_base": summarize_paired(base, arr("shared_full"), f"{task}:shared_vs_base", args.bootstrap_iters, args.perm_iters, args.ci_alpha, seed0 + 1),
            "ctrl_struct_vs_base": summarize_paired(base, arr("ctrl_struct"), f"{task}:ctrl_struct_vs_base", args.bootstrap_iters, args.perm_iters, args.ci_alpha, seed0 + 2),
            "ctrl_energy_vs_base": summarize_paired(base, arr("ctrl_energy"), f"{task}:ctrl_energy_vs_base", args.bootstrap_iters, args.perm_iters, args.ci_alpha, seed0 + 3),
            "shared_vs_ctrl_struct": summarize_paired(arr("ctrl_struct"), arr("shared_full"), f"{task}:shared_vs_ctrl_struct", args.bootstrap_iters, args.perm_iters, args.ci_alpha, seed0 + 4),
            "shared_vs_ctrl_energy": summarize_paired(arr("ctrl_energy"), arr("shared_full"), f"{task}:shared_vs_ctrl_energy", args.bootstrap_iters, args.perm_iters, args.ci_alpha, seed0 + 5),
        }

        print("  [Paired]")
        for k, st in block["paired"].items():
            print(f"    {k:22s} Δ={fmt_diff(st)} p={st['p_value']:.4g}")

        results["by_task"][task] = block

        summary_lines.append(f"[ForcedChoice] {task} n={len(exs)}")
        for cname in ["baseline", "shared_alpha", "ctrl_alpha"]:
            r = block["runs"][cname]
            summary_lines.append(f"  {cname:11s} {fmt_acc(r['accuracy'], r['ci_low'], r['ci_high'])}  (alpha={r.get('alpha', float('nan')):.4g})")
         
        for cname in ["baseline", "shared_full", "ctrl_struct", "ctrl_energy"]:
            r = block["runs"][cname]
            summary_lines.append(f"  {cname:11s} {fmt_acc(r['accuracy'], r['ci_low'], r['ci_high'])}")
        summary_lines.append(f"  Δ(shared_alpha-base) {fmt_diff(block['paired']['shared_alpha_vs_base'])} p={block['paired']['shared_alpha_vs_base']['p_value']:.4g}")
        summary_lines.append(f"  Δ(ctrl_alpha-base)   {fmt_diff(block['paired']['ctrl_alpha_vs_base'])} p={block['paired']['ctrl_alpha_vs_base']['p_value']:.4g}")
        summary_lines.append(f"  Δ(shared_alpha-ctrl_alpha) {fmt_diff(block['paired']['shared_alpha_vs_ctrl_alpha'])} p={block['paired']['shared_alpha_vs_ctrl_alpha']['p_value']:.4g}")
        summary_lines.append(f"  Δ(shared-base)       {fmt_diff(block['paired']['shared_vs_base'])} p={block['paired']['shared_vs_base']['p_value']:.4g}")
        summary_lines.append(f"  Δ(ctrl_struct-base)  {fmt_diff(block['paired']['ctrl_struct_vs_base'])} p={block['paired']['ctrl_struct_vs_base']['p_value']:.4g}")
        summary_lines.append(f"  Δ(ctrl_energy-base)  {fmt_diff(block['paired']['ctrl_energy_vs_base'])} p={block['paired']['ctrl_energy_vs_base']['p_value']:.4g}")
        summary_lines.append(f"  Δ(shared-ctrl_struct){fmt_diff(block['paired']['shared_vs_ctrl_struct'])} p={block['paired']['shared_vs_ctrl_struct']['p_value']:.4g}")
        summary_lines.append(f"  Δ(shared-ctrl_energy){fmt_diff(block['paired']['shared_vs_ctrl_energy'])} p={block['paired']['shared_vs_ctrl_energy']['p_value']:.4g}")
        summary_lines.append("")

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

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
