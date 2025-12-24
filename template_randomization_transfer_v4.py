
# -*- coding: utf-8 -*-
"""
template_randomization_transfer_v2.py

Template randomization experiment upgraded to match the "energy_balance" mainline:

  1) SAME basis estimation pipeline as the main experiment:
     - collect DECODE-only (seq_len==1) last-token states on calibration prompts
     - compute_cross_task_subspace -> find_fully_shared_basis_improved (sharedness)
  2) Cross-template *causal transfer matrix* (A -> B):
     - estimate shared basis on template seed A
     - intervene during forced-choice scoring on prompts rendered with template seed B
     - report Δacc = baseline(B) - acc(shared(A→B))  (and random control)
  3) Geometric evidence:
     - principal angles between shared subspaces across templates (k-matched)
     - energy ratio r(h,Q)=||Q^T h||^2 / ||h||^2 cross-evaluation (A basis on B states)
  4) Outputs:
     - template_transfer_v2.json
     - template_transfer_v2.md
     - template_transfer_v2.tex

Run example:
  python template_randomization_transfer_v4.py \
    --model meta-llama/Llama-2-7b-chat-hf --device cuda --dtype fp32 \
    --layer 10 --n_prompts 128 --eval_n 256 \
    --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
    --pca_var 0.95 --tau 0.001 --m_shared all \
    --template_seeds 0 1 2 3 \
    --fc_batch_size 4

Notes:
- Forced-choice evaluation is decode-only (teacher-forced candidate tokens), so the intervention
  happens strictly on decode steps (seq_len==1), matching your "decode-aligned" story.
- The transfer matrix is a *robustness* check: if the basis is merely "template directions",
  the effect should be mostly diagonal (A->A), and weak off-diagonal. If it's task/workspace
  directions, the effect should transfer across templates.
"""

import os
import sys
import re
import json
import math
import time
import argparse
import random
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------
# Forced-choice suffix
# ---------------------------------------------------------------------
# Forced-choice in this script scores the probability of answer candidates as
# the *next tokens* after a prompt prefix. Therefore the prompt must end with a
# fixed "answer prefix" such that the next token is one of the answer labels.
#
# If this suffix is missing, forced-choice will be near chance because the
# model is still in the "reasoning" part of the prompt (expecting to generate
# thoughts rather than an answer label).
ANSWER_PREFIX_DEFAULT = "\nFinal answer:"

# ---------------------------------------------------------------------
# Try importing *exactly* the same utilities as the main energy_balance script
# ---------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# Add parent(s) to PYTHONPATH (robust to different repo layouts)
for up in [1, 2, 3]:
    sys.path.append(os.path.abspath(os.path.join(THIS_DIR, *([".."] * up))))

HAVE_PROJECT_UTILS = False
try:
    from joint_subspace_large.disturb_cross_task_all_shared import (  # noqa: E402
        get_model_layers,
        compute_cross_task_subspace,
        find_fully_shared_basis_improved,
    )
    HAVE_PROJECT_UTILS = True
except Exception as e:
    HAVE_PROJECT_UTILS = False
    _IMPORT_ERR = str(e)

# -----------------------------
# Repro / misc utils
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
    # r(h,Q)=||Q^T h||^2 / ||h||^2
    eps = 1e-12
    H = states.astype(np.float32, copy=False)
    proj = H @ Q
    num = np.sum(proj * proj, axis=1)
    den = np.sum(H * H, axis=1) + eps
    r = num / den
    return {
        "mean": float(np.mean(r)) if r.size else float("nan"),
        "p50": float(np.percentile(r, 50)) if r.size else float("nan"),
        "p95": float(np.percentile(r, 95)) if r.size else float("nan"),
    }

def principal_angle_stats(Qa: np.ndarray, Qb: np.ndarray) -> Dict[str, float]:
    """
    Principal angles between two k-dimensional subspaces (k-matched).
    We return cosines and mean angle in degrees.
    """
    if Qa.size == 0 or Qb.size == 0:
        return {"max_cos": float("nan"), "mean_cos": float("nan"), "min_cos": float("nan"),
                "mean_angle_deg": float("nan"), "fro_norm": float("nan")}
    # cosines are singular values of Qa^T Qb
    M = Qa.T @ Qb
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    angles = np.degrees(np.arccos(s))
    Pa = Qa @ Qa.T
    Pb = Qb @ Qb.T
    fro = float(np.linalg.norm(Pa - Pb, ord="fro"))
    return {
        "max_cos": float(np.max(s)),
        "mean_cos": float(np.mean(s)),
        "min_cos": float(np.min(s)),
        "mean_angle_deg": float(np.mean(angles)),
        "fro_norm": fro,
    }

def _subsample_rows_np(x: np.ndarray, n_max: int, seed: int) -> np.ndarray:
    if n_max is None or n_max <= 0 or x.shape[0] <= n_max:
        return x
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.shape[0], size=n_max, replace=False)
    return x[idx]

def _shuffle_and_take(x: np.ndarray, n: int, seed: int) -> np.ndarray:
    if x.shape[0] <= n:
        return x
    rng = np.random.default_rng(seed)
    idx = rng.permutation(x.shape[0])[:n]
    return x[idx]

# -----------------------------
# Dataset: raw examples (prompt rendered later via templates)
# -----------------------------
@dataclass
class RawExample:
    dataset: str
    ex_id: str
    question: str
    gold: str
    # optional fields
    choices: Optional[Dict[str, List[str]]] = None
    options: Optional[List[str]] = None

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

def sample_hf_split(ds_split, n: int, seed: int):
    n = min(n, len(ds_split))
    if n <= 0:
        return ds_split.select([])
    return ds_split.shuffle(seed=seed).select(range(n))

def load_all_raw_examples(n_prompts: int, eval_n: int, seed: int) -> Tuple[
    Dict[str, List[RawExample]], Dict[str, List[RawExample]], Dict[str, Any]
]:
    sub_by, eval_by, meta_by = {}, {}, {}

    # gsm8k
    ds = load_dataset("gsm8k", "main")
    sub_split = "train" if "train" in ds else list(ds.keys())[0]
    eval_split = "test" if "test" in ds else ("validation" if "validation" in ds else sub_split)
    sub_rows = sample_hf_split(ds[sub_split], n_prompts, seed + 1)
    eval_rows = sample_hf_split(ds[eval_split], eval_n, seed + 2)
    sub_by["gsm8k"] = [RawExample("gsm8k", f"gsm8k-{sub_split}-{i}", ex["question"], parse_gsm8k_gold(ex["answer"])) for i, ex in enumerate(sub_rows)]
    eval_by["gsm8k"] = [RawExample("gsm8k", f"gsm8k-{eval_split}-{i}", ex["question"], parse_gsm8k_gold(ex["answer"])) for i, ex in enumerate(eval_rows)]
    meta_by["gsm8k"] = {"hf_id": "gsm8k/main", "subspace_split": sub_split, "eval_split": eval_split}

    # commonsense_qa
    ds = load_dataset("commonsense_qa")
    sub_split = "train" if "train" in ds else list(ds.keys())[0]
    eval_split = "validation" if "validation" in ds else ("test" if "test" in ds else sub_split)
    sub_rows = sample_hf_split(ds[sub_split], n_prompts, seed + 11)
    eval_rows = sample_hf_split(ds[eval_split], eval_n, seed + 12)
    sub_by["commonsenseqa"] = [
        RawExample("commonsenseqa", f"csqa-{sub_split}-{i}", ex["question"], safe_upper(ex["answerKey"]), choices=ex["choices"])
        for i, ex in enumerate(sub_rows)
    ]
    eval_by["commonsenseqa"] = [
        RawExample("commonsenseqa", f"csqa-{eval_split}-{i}", ex["question"], safe_upper(ex["answerKey"]), choices=ex["choices"])
        for i, ex in enumerate(eval_rows)
    ]
    meta_by["commonsenseqa"] = {"hf_id": "commonsense_qa", "subspace_split": sub_split, "eval_split": eval_split}

    # strategyqa (ChilleD/StrategyQA)
    ds = load_dataset("ChilleD/StrategyQA")
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

    sub_by["strategyqa"] = [RawExample("strategyqa", f"strategyqa-{sub_split}-{i}", ex["question"], to_yesno(ex["answer"])) for i, ex in enumerate(sub_rows)]
    eval_by["strategyqa"] = [RawExample("strategyqa", f"strategyqa-{eval_split}-{i}", ex["question"], to_yesno(ex["answer"])) for i, ex in enumerate(eval_rows)]
    meta_by["strategyqa"] = {"hf_id": "ChilleD/StrategyQA", "subspace_split": sub_split, "eval_split": eval_split}

    # aqua_rat
    ds = load_dataset("aqua_rat")
    sub_split = "train" if "train" in ds else list(ds.keys())[0]
    eval_split = "test" if "test" in ds else ("validation" if "validation" in ds else sub_split)
    sub_rows = sample_hf_split(ds[sub_split], n_prompts, seed + 21)
    eval_rows = sample_hf_split(ds[eval_split], eval_n, seed + 22)

    def get_gold_aqua(ex: dict) -> str:
        if "correct" in ex:
            return safe_upper(ex["correct"])
        if "answer" in ex:
            return safe_upper(ex["answer"])
        return ""

    sub_by["aqua"] = [RawExample("aqua", f"aqua-{sub_split}-{i}", ex["question"], get_gold_aqua(ex), options=ex["options"]) for i, ex in enumerate(sub_rows)]
    eval_by["aqua"] = [RawExample("aqua", f"aqua-{eval_split}-{i}", ex["question"], get_gold_aqua(ex), options=ex["options"]) for i, ex in enumerate(eval_rows)]
    meta_by["aqua"] = {"hf_id": "aqua_rat", "subspace_split": sub_split, "eval_split": eval_split}

    return sub_by, eval_by, meta_by

# -----------------------------
# Templates (4 variants per dataset)
# -----------------------------
def template_id_for_ex(ex_id: str, template_seed: int, n_templates: int = 4) -> int:
    # deterministic: ex_id + template_seed
    h = stable_int_seed("template", ex_id, template_seed)
    return int(h % n_templates)

def build_prompt(dataset: str, ex: RawExample, template_id: int) -> str:
    if dataset == "gsm8k":
        T = [
            ("Question: {q}\nLet's think step by step.\nAt the end, write exactly one line: "
             "\"Final answer: <number>\".\n"),
            ("Solve the following math problem.\nProblem: {q}\nReason step by step.\n"
             "Conclude with exactly: Final answer: <number>\n"),
            ("You are a careful mathematician.\n{q}\nWork it out step by step.\n"
             "Finish with: Final answer: <number>\n"),
            ("Question: {q}\nExplain your reasoning.\nProvide final line as: Final answer: <number>\n"),
        ]
        return T[template_id].format(q=ex.question)

    if dataset == "commonsenseqa":
        assert ex.choices is not None
        labels = ex.choices["label"]
        texts = ex.choices["text"]
        lines = [f"{lab}) {txt}" for lab, txt in zip(labels, texts)]
        choices_block = "\n".join(lines)
        T = [
            ("Question: {q}\nChoices:\n{c}\nReason step by step.\n"
             "End with exactly: Final answer: <A/B/C/D/E>\n"),
            ("Select the best answer.\nQ: {q}\nOptions:\n{c}\nExplain briefly.\n"
             "Final line: Final answer: <A/B/C/D/E>\n"),
            ("You are given a multiple-choice question.\n{q}\n{c}\n"
             "Think step by step.\nAnswer with: Final answer: <A/B/C/D/E>\n"),
            ("Question: {q}\nPossible answers:\n{c}\nShow reasoning.\n"
             "Provide: Final answer: <A/B/C/D/E>\n"),
        ]
        return T[template_id].format(q=ex.question, c=choices_block)

    if dataset == "strategyqa":
        T = [
            ("Question: {q}\nPlease reason step by step.\nFinal line: Final answer: Yes or No\n"),
            ("Decide whether the statement is true.\n{q}\nExplain.\nFinal answer as: Final answer: Yes/No\n"),
            ("Answer the following with reasoning.\nQ: {q}\nConclude: Final answer: Yes or Final answer: No\n"),
            ("Question: {q}\nThink carefully.\nEnd with: Final answer: Yes/No\n"),
        ]
        return T[template_id].format(q=ex.question)

    if dataset == "aqua":
        assert ex.options is not None
        labels = ["A", "B", "C", "D", "E"]
        opts = []
        for i, opt in enumerate(ex.options[:5]):
            lab = labels[i]
            opt_clean = re.sub(r"^[A-E]\)?\s*[:\-]?\s*", "", str(opt).strip(), flags=re.IGNORECASE)
            opts.append(f"{lab}) {opt_clean}")
        choices_block = "\n".join(opts)
        T = [
            ("Question: {q}\nChoices:\n{c}\nReason step by step.\n"
             "Final line: Final answer: <A/B/C/D/E>\n"),
            ("Solve and pick the correct option.\nProblem: {q}\nOptions:\n{c}\n"
             "Show steps.\nFinal answer: <A/B/C/D/E>\n"),
            ("You are solving a word problem.\n{q}\n{c}\nExplain your reasoning.\n"
             "End with: Final answer: <A/B/C/D/E>\n"),
            ("Question: {q}\nOptions:\n{c}\nWork step by step.\n"
             "Conclude with: Final answer: <A/B/C/D/E>\n"),
        ]
        return T[template_id].format(q=ex.question, c=choices_block)

    raise ValueError(f"Unknown dataset={dataset}")

def render_prompts(
    exs: List[RawExample],
    template_seed: int,
    *,
    add_answer_prefix: bool = True,
    answer_prefix: str = ANSWER_PREFIX_DEFAULT,
) -> List[str]:
    """Render prompts with a deterministic random template.

    IMPORTANT: Forced-choice scoring assumes the prompt ends with a fixed answer
    prefix (e.g., "\nFinal answer:") so that the next token is the answer label.
    """
    out: List[str] = []
    for ex in exs:
        tid = template_id_for_ex(ex.ex_id, template_seed, n_templates=4)
        p = build_prompt(ex.dataset, ex, tid)
        if add_answer_prefix:
            p = p + str(answer_prefix)
        out.append(p)
    return out

# -----------------------------
# Decode-only last-token activation collector (seq_len==1 only)
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
# Shared subspace estimation (mainline)
# -----------------------------
def _get_layers_fallback(model):
    # Generic fallback for common HF causal LM architectures
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    raise RuntimeError("Could not locate transformer layers for hook registration.")

def get_layers(model):
    if HAVE_PROJECT_UTILS:
        layers, _ = get_model_layers(model)
        return layers
    return _get_layers_fallback(model)

def infer_pooled_component_variance(
    task_acts: Dict[str, np.ndarray],
    joint_subspace: np.ndarray,   # [D, cross_dim]
) -> np.ndarray:
    # pooled variance of each component across tasks
    vars_all = []
    Q = joint_subspace.astype(np.float32, copy=False)
    for _t, X in task_acts.items():
        Z = X.astype(np.float32, copy=False) @ Q
        vars_all.append(np.var(Z, axis=0))
    return np.mean(np.stack(vars_all, axis=0), axis=0).astype(np.float64, copy=False)

def select_rand_indices_varmatch(
    pooled_var: np.ndarray,
    shared_idx_sorted: List[int],
    cross_dim: int,
    seed: int,
) -> List[int]:
    # Choose nonshared components whose pooled variance matches shared components (nearest neighbor, without replacement)
    rng = np.random.default_rng(seed)
    shared_set = set(shared_idx_sorted)
    nonshared = [i for i in range(cross_dim) if i not in shared_set]
    if len(nonshared) < len(shared_idx_sorted):
        raise RuntimeError("Not enough nonshared components for varmatch random basis.")
    nonshared_sorted = sorted(nonshared, key=lambda i: pooled_var[i])
    nonshared_vals = [pooled_var[i] for i in nonshared_sorted]
    import bisect
    chosen = []
    for i_s in shared_idx_sorted:
        v = pooled_var[i_s]
        j = bisect.bisect_left(nonshared_vals, v)
        cand = []
        if 0 <= j < len(nonshared_sorted):
            cand.append(j)
        if 0 <= j-1 < len(nonshared_sorted):
            cand.append(j-1)
        best = None
        best_d = None
        for p in cand:
            d = abs(nonshared_vals[p] - v)
            if best is None or d < best_d - 1e-12 or (abs(d - best_d) < 1e-12 and rng.random() < 0.5):
                best = p
                best_d = d
        if best is None:
            best = int(rng.integers(0, len(nonshared_sorted)))
        chosen_idx = nonshared_sorted.pop(best)
        nonshared_vals.pop(best)
        chosen.append(chosen_idx)
    return chosen

def select_rand_indices_uniform(
    cross_dim: int,
    shared_idx_sorted: List[int],
    k: int,
    seed: int,
) -> List[int]:
    rng = np.random.default_rng(seed)
    shared_set = set(shared_idx_sorted)
    nonshared = [i for i in range(cross_dim) if i not in shared_set]
    if len(nonshared) < k:
        raise RuntimeError("Not enough nonshared components for random basis.")
    return list(rng.choice(nonshared, size=k, replace=False))

@torch.no_grad()
def estimate_basis_for_template_seed(
    model,
    tokenizer,
    sub_by_task: Dict[str, List[RawExample]],
    layer_idx: int,
    template_seed: int,
    *,
    calib_decoding: str,
    calib_batch_size: int,
    calib_max_new_tokens: int,
    per_task_max_states: int,
    max_prompt_len: int,
    temperature: float,
    top_p: float,
    top_k: int,
    variance_threshold: float,
    min_dim: int,
    max_dim: int,
    tau: float,
    m_shared: str,
    global_seed: int,
) -> Dict[str, Any]:
    """
    Returns a dict containing joint_subspace, shared indices, pooled_var, and small state sample (for cross-energy).
    """
    t0 = time.time()

    # Render calibration prompts for this template seed
    prompts_by_task = {t: render_prompts(exs, template_seed) for t, exs in sub_by_task.items()}

    # Hook collector
    layers = get_layers(model)
    collector = DecodeLastTokenActivationCollector([layer_idx])
    handles = []
    for li in [layer_idx]:
        if li >= len(layers):
            raise RuntimeError(f"layer_idx={li} out of range (n_layers={len(layers)})")
        handles.append(layers[li].register_forward_hook(collector.make_hook(li)))

    # Collect decode states
    try:
        for task, prompts in prompts_by_task.items():
            collector.set_current_task(task)
            collect_decode_last_token_states(
                model, tokenizer, prompts, collector,
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

    # Build per-task activations (balanced)
    task_acts_layer: Dict[str, np.ndarray] = {}
    counts = {}
    for task in sub_by_task.keys():
        X = collector.get_task_activations(task, layer_idx)
        if X is None or X.shape[0] == 0:
            continue
        X = _subsample_rows_np(X, per_task_max_states, seed=stable_int_seed(global_seed, "subsample", template_seed, task))
        counts[task] = X.shape[0]
        task_acts_layer[task] = X.astype(np.float32, copy=False)

    if len(task_acts_layer) < 2:
        raise RuntimeError(f"Too few tasks with activations for template_seed={template_seed}: {list(task_acts_layer.keys())}")

    n_min = min(counts.values())
    # shuffle+take to balance
    for task in list(task_acts_layer.keys()):
        task_acts_layer[task] = _shuffle_and_take(task_acts_layer[task], n_min, seed=stable_int_seed(global_seed, "balance", template_seed, task))

    # Convert to compute_cross_task_subspace input format: {task: {layer: X}}
    task_activations_dict = {t: {layer_idx: X} for t, X in task_acts_layer.items()}
    tasks_used = list(task_activations_dict.keys())

    if not HAVE_PROJECT_UTILS:
        raise RuntimeError(
            "Project utils import failed; cannot run v2 with mainline compute_cross_task_subspace.\n"
            f"Import error: {_IMPORT_ERR}"
        )

    joint_subspace, cross_dim, contributions, full_pca_info = compute_cross_task_subspace(
        task_activations_dict,
        variance_threshold=variance_threshold,
        min_dim=min_dim,
        max_dim=max_dim,
        return_full_pca=True,
    )
    if joint_subspace is None or int(cross_dim) <= 0:
        raise RuntimeError("compute_cross_task_subspace returned empty subspace.")

    joint_subspace = joint_subspace.astype(np.float32, copy=False)
    cross_dim = int(cross_dim)

    # sharedness (same as main experiment)
    if m_shared == "all":
        min_tasks_shared = len(tasks_used)
    else:
        min_tasks_shared = int(m_shared)

    shared_indices = find_fully_shared_basis_improved(
        contributions,
        tasks_used,
        cross_dim,
        min_tasks_shared=min_tasks_shared,
        relative_threshold=float(tau),
        top_k_components=cross_dim,
    )
    if not shared_indices and min_tasks_shared > 2:
        # fallback (optional)
        shared_indices = find_fully_shared_basis_improved(
            contributions,
            tasks_used,
            cross_dim,
            min_tasks_shared=2,
            relative_threshold=float(tau),
            top_k_components=cross_dim,
        )

    if not shared_indices:
        raise RuntimeError(f"No shared components found for template_seed={template_seed}. Try smaller tau or m_shared.")

    # pooled per-component variance (for sorting + random var-match)
    pooled_var = infer_pooled_component_variance(task_acts_layer, orthonormalize_np(joint_subspace))

    # sort shared indices by pooled variance descending
    shared_sorted = sorted(shared_indices, key=lambda i: pooled_var[i], reverse=True)

    # small pooled state sample (for cross-energy eval)
    pool = []
    for t in tasks_used:
        pool.append(_subsample_rows_np(task_acts_layer[t], n_max=min(4000, n_min), seed=stable_int_seed(global_seed, "energy_pool", template_seed, t)))
    states_sample = np.concatenate(pool, axis=0).astype(np.float32, copy=False)

    dt = time.time() - t0

    return {
        "template_seed": int(template_seed),
        "tasks_used": tasks_used,
        "n_min": int(n_min),
        "counts_raw": {k: int(v) for k, v in counts.items()},
        "cross_dim": int(cross_dim),
        "shared_k": int(len(shared_sorted)),
        "joint_subspace": joint_subspace,        # [D, cross_dim]
        "shared_sorted": shared_sorted,          # indices into cross components, sorted by var
        "pooled_var": pooled_var,                # [cross_dim]
        "states_sample": states_sample,          # [N, D]
        "time_sec": float(dt),
    }

# -----------------------------
# Intervention hook for forced-choice (decode-only, seq_len==1)
# -----------------------------
class LastTokenSubspaceRemovalHook:
    def __init__(self, Q_np: np.ndarray, alpha: float = 1.0):
        self.alpha = float(alpha)
        Q = orthonormalize_np(Q_np)
        self.Q_cpu = torch.tensor(Q, dtype=torch.float32, device="cpu")
        self.Q_dev: Optional[torch.Tensor] = None
        self.decode_calls = 0
        self.intervened = 0

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
        self.decode_calls += 1
        Q = self._Q(hs.device)
        x = hs[:, -1, :].float()
        proj = (x @ Q) @ Q.T
        hs2 = hs.clone()
        hs2[:, -1, :] = (x - self.alpha * proj).to(dtype=hs.dtype)
        self.intervened += 1
        if isinstance(output, tuple):
            return (hs2,) + output[1:]
        return hs2

def register_hook(model, layer_idx: int, hook_obj: LastTokenSubspaceRemovalHook):
    layers = get_layers(model)
    if layer_idx >= len(layers):
        raise RuntimeError(f"layer_idx={layer_idx} out of range (n_layers={len(layers)})")
    return layers[layer_idx].register_forward_hook(hook_obj)

def forced_choice_batch(
    model,
    tokenizer,
    prompts: List[str],
    candidate_strings: List[str],
    *,
    hook: Optional["LastTokenRemovalHook"] = None,
    max_prompt_len: int = 512,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Teacher-forced forced-choice using *decode-step* scoring.

    We prefill each prompt once (seq_len>1, no intervention), then score each
    candidate by feeding its tokens one-by-one with KV cache (seq_len==1),
    so decode-aligned hooks intervene on the same distribution as generation.

    IMPORTANT: when using past_key_values, we must pass an attention_mask whose
    length equals (prompt_len + decoded_steps_so_far). Passing a length-1 mask
    breaks position_ids / attention masking in Llama-family models and will
    collapse scores (often yielding chance-level baselines).
    """
    device = next(model.parameters()).device
    model.eval()

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize prompts (left padded)
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_len,
    ).to(device)

    input_ids = inputs["input_ids"]
    attn0 = inputs["attention_mask"]
    B = input_ids.shape[0]

    # Reset hook stats for this batch
    if hook is not None:
        hook.stats.decode_calls = 0
        hook.stats.intervened = 0

    # Prefill: get cache + logits for next token after prompt
    out = model(input_ids=input_ids, attention_mask=attn0, use_cache=True)
    logits0 = out.logits[:, -1, :]         # [B, V]
    past0 = out.past_key_values

    # Tokenize candidates once (no specials)
    tok_ids: List[List[int]] = []
    for c in candidate_strings:
        ids = tokenizer(c, add_special_tokens=False).input_ids
        tok_ids.append([int(x) for x in ids])

    # Score each candidate: sum log p(token_t | prompt, previous candidate tokens)
    scores: List[torch.Tensor] = []
    for cand_ids in tok_ids:
        if len(cand_ids) == 0:
            scores.append(torch.full((B,), -1e9, device=device))
            continue

        logits = logits0
        past = past0
        attn = attn0  # will be extended (new tensor) per step
        score = torch.zeros((B,), device=device)

        for step, tid in enumerate(cand_ids):
            logp = torch.log_softmax(logits, dim=-1)
            score = score + logp[:, tid]

            # Advance cache to get logits for next token (if needed)
            if step < len(cand_ids) - 1:
                inp = torch.full((B, 1), tid, device=device, dtype=torch.long)
                attn = torch.cat(
                    [attn, torch.ones((B, 1), device=device, dtype=attn.dtype)],
                    dim=1,
                )
                out2 = model(
                    input_ids=inp,
                    attention_mask=attn,
                    use_cache=True,
                    past_key_values=past,
                )
                logits = out2.logits[:, -1, :]
                past = out2.past_key_values

        scores.append(score)

    scores_t = torch.stack(scores, dim=1)  # [B, C]
    pred_idx = torch.argmax(scores_t, dim=1).detach().cpu().numpy()

    stats = {
        "decode_calls": int(getattr(hook, "stats", None).decode_calls if hook is not None else 0),
        "intervened": int(getattr(hook, "stats", None).intervened if hook is not None else 0),
    }
    return pred_idx, stats
def forced_choice_candidates_for_task(task: str) -> Tuple[List[str], List[str]]:
    if task in ["commonsenseqa", "aqua"]:
        labels = ["A", "B", "C", "D", "E"]
        cands = [" " + x for x in labels]
        return cands, labels
    if task == "strategyqa":
        labels = ["YES", "NO"]
        cands = [" Yes", " No"]
        return cands, labels
    raise ValueError(task)

def gold_label_for_task(task: str, ex: RawExample) -> str:
    if task == "strategyqa":
        return ex.gold.upper()
    return ex.gold.upper()

def eval_forced_choice(
    model,
    tokenizer,
    eval_exs: List[RawExample],
    task: str,
    template_seed: int,
    Q_np: Optional[np.ndarray],
    layer_idx: int,
    alpha: float,
    batch_size: int,
    max_prompt_len: int,
    device: str,
) -> Dict[str, Any]:
    prompts = render_prompts(eval_exs, template_seed)
    cand_strs, cand_labels = forced_choice_candidates_for_task(task)

    pred_idx, hk_stats = forced_choice_batch(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        candidates=cand_strs,
        Q_np=Q_np,
        layer_idx=layer_idx,
        alpha=alpha,
        batch_size=batch_size,
        max_prompt_len=max_prompt_len,
        device=device,
    )
    preds = [cand_labels[i] for i in pred_idx.tolist()]
    golds = [gold_label_for_task(task, ex) for ex in eval_exs]
    correct = np.array([int(p == g) for p, g in zip(preds, golds)], dtype=np.int32)

    acc = float(np.mean(correct)) if correct.size else float("nan")
    return {
        "n": int(len(eval_exs)),
        "accuracy": acc,
        "correct": correct.tolist(),
        "hook_stats": hk_stats,
    }

# -----------------------------
# Output formatting: matrices -> Markdown + LaTeX
# -----------------------------
def format_matrix_md(mat: np.ndarray, row_labels: List[str], col_labels: List[str], fmt: str = "{:+.1f}") -> str:
    header = "| | " + " | ".join(col_labels) + " |"
    sep = "|" + "---|" * (len(col_labels) + 1)
    lines = [header, sep]
    for i, rlab in enumerate(row_labels):
        row = [rlab] + [fmt.format(mat[i, j]) for j in range(len(col_labels))]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)

def format_matrix_tex(mat: np.ndarray, row_labels: List[str], col_labels: List[str], fmt: str = "{:+.1f}") -> str:
    cols = "l" + "r" * len(col_labels)
    lines = []
    lines.append("\\begin{tabular}{" + cols + "}")
    lines.append("\\toprule")
    lines.append(" & " + " & ".join(col_labels) + " \\\\")
    lines.append("\\midrule")
    for i, rlab in enumerate(row_labels):
        vals = [fmt.format(mat[i, j]) for j in range(len(col_labels))]
        lines.append(f"{rlab} & " + " & ".join(vals) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)

# -----------------------------
# Model loading
# -----------------------------
def load_model_and_tokenizer(model_name: str, device: str, dtype_str: str):
    dtype = torch.float32 if dtype_str == "fp32" else torch.float16
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    # Keep the *end* of long prompts, since we often append an answer prefix
    # (e.g., "\nFinal answer:") right before forced-choice scoring.
    tok.truncation_side = "left"
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
    ap.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--layer", type=int, default=10)

    ap.add_argument("--n_prompts", type=int, default=128)
    ap.add_argument("--eval_n", type=int, default=256)

    ap.add_argument("--calib_decode_max_new_tokens", type=int, default=128)
    ap.add_argument("--per_task_max_states", type=int, default=20000)
    ap.add_argument("--max_prompt_len", type=int, default=512)

    ap.add_argument("--pca_var", type=float, default=0.95)
    ap.add_argument("--min_dim", type=int, default=1)
    ap.add_argument("--max_dim", type=int, default=4096)
    ap.add_argument("--tau", type=float, default=0.001)
    ap.add_argument("--m_shared", type=str, default="all")  # "all" or integer

    ap.add_argument("--template_seeds", type=int, nargs="+", default=[0, 1, 2, 3])

    ap.add_argument("--alpha_remove", type=float, default=1.0)
    ap.add_argument("--calib_batch_size", type=int, default=4)
    ap.add_argument("--calib_decoding", type=str, default="greedy", choices=["greedy", "sample"])
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=0)

    ap.add_argument("--fc_batch_size", type=int, default=4)
    ap.add_argument("--rand_type", type=str, default="varmatch", choices=["varmatch", "uniform"])

    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--out_json", type=str, default=os.path.join(THIS_DIR, "template_transfer_v2.json"))
    ap.add_argument("--out_md", type=str, default=os.path.join(THIS_DIR, "template_transfer_v2.md"))
    ap.add_argument("--out_tex", type=str, default=os.path.join(THIS_DIR, "template_transfer_v2.tex"))
    args = ap.parse_args()

    set_global_seed(args.seed)

    print(f"[Env] model={args.model} device={args.device} dtype={args.dtype} layer={args.layer}")
    print(f"[Env] HAVE_PROJECT_UTILS={HAVE_PROJECT_UTILS}")
    if not HAVE_PROJECT_UTILS:
        print(f"[Env] Import error: {_IMPORT_ERR}")

    model, tok = load_model_and_tokenizer(args.model, args.device, args.dtype)

    # Load datasets once (fixed underlying examples across templates)
    sub_by, eval_by, meta_by = load_all_raw_examples(args.n_prompts, args.eval_n, args.seed)
    tasks_all = list(sub_by.keys())
    tasks_fc = ["commonsenseqa", "strategyqa", "aqua"]  # forced-choice tasks

    print(f"[Data] tasks_all={tasks_all} tasks_fc={tasks_fc}")
    print(f"[Data] meta={json.dumps(meta_by, ensure_ascii=False)}")

    # 1) Estimate basis for each template seed
    per_seed = {}
    for s in args.template_seeds:
        print("\n" + "=" * 80)
        print(f"[Basis] Estimating shared basis for template_seed={s} (decode-only states)")
        print("=" * 80)
        info = estimate_basis_for_template_seed(
            model=model,
            tokenizer=tok,
            sub_by_task=sub_by,
            layer_idx=args.layer,
            template_seed=int(s),
            calib_decoding=args.calib_decoding,
            calib_batch_size=args.calib_batch_size,
            calib_max_new_tokens=args.calib_decode_max_new_tokens,
            per_task_max_states=args.per_task_max_states,
            max_prompt_len=args.max_prompt_len,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            variance_threshold=args.pca_var,
            min_dim=args.min_dim,
            max_dim=args.max_dim,
            tau=args.tau,
            m_shared=args.m_shared,
            global_seed=args.seed,
        )
        print(f"[Basis] seed={s} cross_dim={info['cross_dim']} shared_k={info['shared_k']} n_min={info['n_min']} time={info['time_sec']:.1f}s")
        per_seed[int(s)] = info

    # k-match across seeds
    shared_ks = [per_seed[s]["shared_k"] for s in per_seed.keys()]
    k_match = int(min(shared_ks))
    print(f"\n[Match] shared_k per seed={shared_ks} => k_match={k_match}")

    # Build k-matched Q_shared and Q_rand for each seed
    for s, info in per_seed.items():
        Q_joint = orthonormalize_np(info["joint_subspace"])  # [D,cross_dim] (ensure ortho)
        cross_dim = int(info["cross_dim"])
        pooled_var = info["pooled_var"]
        shared_sorted = info["shared_sorted"][:k_match]

        Q_shared_km = orthonormalize_np(Q_joint[:, shared_sorted])

        if args.rand_type == "varmatch":
            rand_idx = select_rand_indices_varmatch(
                pooled_var=pooled_var,
                shared_idx_sorted=shared_sorted,
                cross_dim=cross_dim,
                seed=stable_int_seed(args.seed, "rand", s),
            )
            rand_idx = rand_idx[:k_match]
        else:
            rand_idx = select_rand_indices_uniform(
                cross_dim=cross_dim,
                shared_idx_sorted=shared_sorted,
                k=k_match,
                seed=stable_int_seed(args.seed, "rand", s),
            )

        Q_rand_km = orthonormalize_np(Q_joint[:, rand_idx])

        # sanity
        print(f"[Sanity][seed={s}] offdiag shared={max_offdiag(Q_shared_km):.3e} rand={max_offdiag(Q_rand_km):.3e} overlap={max_overlap(Q_shared_km, Q_rand_km):.3e}")

        # energy on its own states
        states = info["states_sample"]
        er_s = energy_ratio_stats(states, Q_shared_km)
        er_r = energy_ratio_stats(states, Q_rand_km)

        info["Q_joint_ortho"] = Q_joint
        info["Q_shared_km"] = Q_shared_km
        info["Q_rand_km"] = Q_rand_km
        info["rand_idx_km"] = rand_idx
        info["shared_idx_km"] = shared_sorted
        info["energy_self"] = {"shared": er_s, "rand": er_r}

    # 2) Geometric matrices: principal angles + cross-energy
    seeds = [int(s) for s in args.template_seeds]
    seeds_str = [str(s) for s in seeds]
    nS = len(seeds)

    mat_angle_mean = np.zeros((nS, nS), dtype=np.float32)
    mat_cos_mean = np.zeros((nS, nS), dtype=np.float32)
    mat_energy = np.zeros((nS, nS), dtype=np.float32)

    angle_stats = {}
    for i, a in enumerate(seeds):
        Qa = per_seed[a]["Q_shared_km"]
        for j, b in enumerate(seeds):
            Qb = per_seed[b]["Q_shared_km"]
            st = principal_angle_stats(Qa, Qb)
            mat_angle_mean[i, j] = float(st["mean_angle_deg"])
            mat_cos_mean[i, j] = float(st["mean_cos"])
            angle_stats[f"{a}->{b}"] = st

            # cross-energy: energy of Qa on states from template b
            states_b = per_seed[b]["states_sample"]
            er = energy_ratio_stats(states_b, Qa)
            mat_energy[i, j] = float(er["mean"])

    # 3) Cross-template causal transfer (forced-choice)
    # Baseline depends only on eval template seed B
    baseline_by_B = {}
    shared_by_A_B = {}
    rand_by_A_B = {}

    for b in seeds:
        print("\n" + "=" * 80)
        print(f"[Eval] Baseline forced-choice for eval_template_seed={b}")
        print("=" * 80)
        baseline_by_B[b] = {}
        for task in tasks_fc:
            res = eval_forced_choice(
                model=model, tokenizer=tok,
                eval_exs=eval_by[task],
                task=task,
                template_seed=b,
                Q_np=None,
                layer_idx=args.layer,
                alpha=args.alpha_remove,
                batch_size=args.fc_batch_size,
                max_prompt_len=args.max_prompt_len,
                device=args.device,
            )
            baseline_by_B[b][task] = res
            print(f"[Baseline][B={b}][{task}] acc={res['accuracy']*100:.1f} n={res['n']}")

    # Transfer A->B for shared and random
    for a in seeds:
        shared_by_A_B[a] = {}
        rand_by_A_B[a] = {}
        Qa = per_seed[a]["Q_shared_km"]
        Ra = per_seed[a]["Q_rand_km"]

        for b in seeds:
            print("\n" + "-" * 80)
            print(f"[Eval] Transfer A->B : A={a} (basis) -> B={b} (eval prompts)")
            print("-" * 80)
            shared_by_A_B[a][b] = {}
            rand_by_A_B[a][b] = {}

            for task in tasks_fc:
                res_s = eval_forced_choice(
                    model=model, tokenizer=tok,
                    eval_exs=eval_by[task],
                    task=task,
                    template_seed=b,
                    Q_np=Qa,
                    layer_idx=args.layer,
                    alpha=args.alpha_remove,
                    batch_size=args.fc_batch_size,
                    max_prompt_len=args.max_prompt_len,
                    device=args.device,
                )
                res_r = eval_forced_choice(
                    model=model, tokenizer=tok,
                    eval_exs=eval_by[task],
                    task=task,
                    template_seed=b,
                    Q_np=Ra,
                    layer_idx=args.layer,
                    alpha=args.alpha_remove,
                    batch_size=args.fc_batch_size,
                    max_prompt_len=args.max_prompt_len,
                    device=args.device,
                )
                shared_by_A_B[a][b][task] = res_s
                rand_by_A_B[a][b][task] = res_r
                print(f"[A={a}->B={b}][{task}] shared acc={res_s['accuracy']*100:.1f} rand acc={res_r['accuracy']*100:.1f}")

    # Build transfer matrices (micro-avg across tasks_fc)
    mat_drop_shared = np.zeros((nS, nS), dtype=np.float32)
    mat_drop_rand = np.zeros((nS, nS), dtype=np.float32)

    def micro_acc_from_blocks(blocks: Dict[str, Any]) -> float:
        all_corr = []
        for t in tasks_fc:
            all_corr.extend(blocks[t]["correct"])
        arr = np.array(all_corr, dtype=np.float32)
        return float(arr.mean()) if arr.size else float("nan")

    for i, a in enumerate(seeds):
        for j, b in enumerate(seeds):
            base_acc = micro_acc_from_blocks(baseline_by_B[b])
            sh_acc = micro_acc_from_blocks(shared_by_A_B[a][b])
            rd_acc = micro_acc_from_blocks(rand_by_A_B[a][b])

            mat_drop_shared[i, j] = (base_acc - sh_acc) * 100.0  # percentage points
            mat_drop_rand[i, j] = (base_acc - rd_acc) * 100.0

    # Package results
    out = {
        "config": {
            "model": args.model,
            "device": args.device,
            "dtype": args.dtype,
            "layer": args.layer,
            "n_prompts": args.n_prompts,
            "eval_n": args.eval_n,
            "calib_decode_max_new_tokens": args.calib_decode_max_new_tokens,
            "per_task_max_states": args.per_task_max_states,
            "pca_var": args.pca_var,
            "tau": args.tau,
            "m_shared": args.m_shared,
            "template_seeds": seeds,
            "k_match": k_match,
            "alpha_remove": args.alpha_remove,
            "calib_decoding": args.calib_decoding,
            "calib_batch_size": args.calib_batch_size,
            "fc_batch_size": args.fc_batch_size,
            "rand_type": args.rand_type,
            "seed": args.seed,
            "have_project_utils": HAVE_PROJECT_UTILS,
        },
        "dataset_meta": meta_by,
        "per_seed": {
            str(s): {
                "template_seed": s,
                "tasks_used": per_seed[s]["tasks_used"],
                "n_min": per_seed[s]["n_min"],
                "counts_raw": per_seed[s]["counts_raw"],
                "cross_dim": per_seed[s]["cross_dim"],
                "shared_k": per_seed[s]["shared_k"],
                "k_match": k_match,
                "energy_self": per_seed[s]["energy_self"],
                "time_sec": per_seed[s]["time_sec"],
            }
            for s in seeds
        },
        "geometry": {
            "principal_angle_stats": angle_stats,
            "mean_angle_deg_matrix": mat_angle_mean,
            "mean_cos_matrix": mat_cos_mean,
            "energy_mean_matrix": mat_energy,
        },
        "forced_choice": {
            "tasks": tasks_fc,
            "baseline_by_eval_seed": baseline_by_B,
            "shared_by_basis_seed": shared_by_A_B,
            "rand_by_basis_seed": rand_by_A_B,
            "drop_shared_pp_matrix": mat_drop_shared,
            "drop_rand_pp_matrix": mat_drop_rand,
        },
    }

    # Write JSON
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=json_default)

    # Build Markdown
    md = []
    md.append("# Template Randomization Transfer (v2)\n")
    md.append(f"- Model: `{args.model}`  layer={args.layer}  dtype={args.dtype}  device={args.device}\n")
    md.append(f"- template_seeds: {seeds}\n")
    md.append(f"- k_match: {k_match}\n")
    md.append(f"- rand_type: {args.rand_type}\n")
    md.append("\n## Cross-template causal transfer matrix\n")
    md.append("Entry = **Δacc (pp)** = baseline(B) − accuracy(shared(A→B)), micro-avg over {commonsenseqa, strategyqa, aqua}.\n")
    md.append(format_matrix_md(mat_drop_shared, seeds_str, seeds_str, fmt="{:+.1f}"))
    md.append("\n\n### Random control (Δacc)\n")
    md.append(format_matrix_md(mat_drop_rand, seeds_str, seeds_str, fmt="{:+.1f}"))
    md.append("\n\n## Geometry: principal angles\n")
    md.append("Entry = mean principal angle (degrees) between k-matched shared subspaces.\n")
    md.append(format_matrix_md(mat_angle_mean, seeds_str, seeds_str, fmt="{:.1f}"))
    md.append("\n\n## Geometry: cross-energy\n")
    md.append("Entry = mean energy ratio r(h,Q_A) on states sampled from template B.\n")
    md.append(format_matrix_md(mat_energy, seeds_str, seeds_str, fmt="{:.3f}"))
    md.append("\n")

    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    # Build LaTeX (3 tables)
    tex = []
    tex.append("% Auto-generated by template_randomization_transfer_v2.py\n")
    tex.append("\\begin{table}[t]\n\\centering\n\\small\n")
    tex.append(format_matrix_tex(mat_drop_shared, seeds_str, seeds_str, fmt="{:+.1f}"))
    tex.append("\\caption{Cross-template causal transfer. Entry is $\\Delta$acc (pp) = baseline(B) $-$ acc(shared basis from A applied to prompts rendered with template B). Micro-average over \\{commonsenseqa, strategyqa, aqua\\}.}\n")
    tex.append("\\label{tab:template-transfer-shared}\n\\end{table}\n\n")

    tex.append("\\begin{table}[t]\n\\centering\n\\small\n")
    tex.append(format_matrix_tex(mat_drop_rand, seeds_str, seeds_str, fmt="{:+.1f}"))
    tex.append("\\caption{Random-basis control for template transfer. Same as Table~\\ref{tab:template-transfer-shared} but using a variance-matched nonshared random basis from template seed A.}\n")
    tex.append("\\label{tab:template-transfer-rand}\n\\end{table}\n\n")

    tex.append("\\begin{table}[t]\n\\centering\n\\small\n")
    tex.append(format_matrix_tex(mat_angle_mean, seeds_str, seeds_str, fmt="{:.1f}"))
    tex.append("\\caption{Geometry of shared subspaces across templates. Entry is the mean principal angle (degrees) between k-matched shared subspaces estimated from template seeds A and B (lower is more aligned).}\n")
    tex.append("\\label{tab:template-transfer-angles}\n\\end{table}\n\n")

    tex.append("\\begin{table}[t]\n\\centering\n\\small\n")
    tex.append(format_matrix_tex(mat_energy, seeds_str, seeds_str, fmt="{:.3f}"))
    tex.append("\\caption{Cross-energy evaluation. Entry is the mean energy ratio $r(h,Q_A)=\\|Q_A^\\top h\\|^2/\\|h\\|^2$ computed on decode states sampled from prompts rendered with template seed B.}\n")
    tex.append("\\label{tab:template-transfer-energy}\n\\end{table}\n")

    with open(args.out_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(tex))

    print("\n" + "=" * 80)
    print("[Done]")
    print(f"[Done] JSON: {args.out_json}")
    print(f"[Done] MD  : {args.out_md}")
    print(f"[Done] TEX : {args.out_tex}")
    print("=" * 80)

if __name__ == "__main__":
    main()
