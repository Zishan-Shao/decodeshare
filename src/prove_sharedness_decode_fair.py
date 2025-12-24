# -*- coding: utf-8 -*-
"""
prove_sharedness_decode_fair.py

Fair existence test for shared basis / shared subspace on DECODE last-token states.

What this script proves (reviewer-friendly):
  1) We collect decode-phase (seq_len==1) last-token hidden states under baseline decoding.
  2) We estimate a pooled (cross-task) PCA subspace.
  3) We define "shared components" as those whose relative variance contribution exceeds tau
     in at least m tasks (default: m = #tasks, i.e., fully shared).
  4) We report significance under two nulls:
       - Null-1 (fast): independently permute per-task relative variance profiles across components.
       - Null-2 (stronger, slower): independently apply per-task orthogonal "scramble"
         (dimension permutation + sign flips) to activations, recompute pooled PCA, then sharedness.

Benchmarks (calibration prompts) included by default:
  - gsm8k
  - commonsenseqa
  - strategyqa
  - aqua
  - arc_challenge (ai2_arc / ARC-Challenge)
  - openbookqa
  - qasc
  - boolq
  - piqa

New feature:
  - --out_txt: tee stdout prints into a local .txt file (summary + all prints).
    Note: tqdm progress bars default to stderr, so the txt log stays mostly clean.

Run example:
  CUDA_VISIBLE_DEVICES=1 python prove_sharedness_decode_fair.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --device cuda \
    --model_dtype fp32 \
    --layer 10 \
    --n_prompts 128 \
    --calib_max_new_tokens 128 \
    --max_prompt_len 512 \
    --per_task_max_states 20000 \
    --tau 0.001 \
    --m_shared all \
    --null_perm_trials 2000 \
    --null_scramble_trials 100 \
    --out_json results/exists/prove_existence.json \
    --out_txt  results/exists/prove_existence.txt
"""

import os
import re
import json
import random
import argparse
import hashlib
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------
# Import your project utilities
# ---------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(THIS_DIR, ".."))

from joint_subspace_large.disturb_cross_task_all_shared import (  # noqa: E402
    get_model_layers,
    compute_cross_task_subspace,
)

# -----------------------------
# Stdout tee (save prints to txt)
# -----------------------------
class TeeStdout:
    def __init__(self, *streams):
        self.streams = streams
        self.encoding = getattr(streams[0], "encoding", "utf-8")

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):
        return any(getattr(s, "isatty", lambda: False)() for s in self.streams)


def _should_write_txt(path: Optional[str]) -> bool:
    if path is None:
        return False
    p = str(path).strip()
    if p == "" or p.lower() in {"none", "null"}:
        return False
    return True


# -----------------------------
# Repro utils
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


def to_py(obj: Any):
    """JSON-safe conversion."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return obj


# -----------------------------
# Prompt builders
# -----------------------------
def build_prompt_gsm8k(question: str) -> str:
    return (
        f"Question: {question}\n"
        "Let's think step by step.\n"
        'At the end, write exactly one line in the format: "Final answer: <number>".\n'
    )


def build_prompt_mc(question: str, labels: List[str], texts: List[str]) -> str:
    labels = [str(x).strip() for x in labels]
    texts = [str(x).strip() for x in texts]
    lines = [f"{lab}) {txt}" for lab, txt in zip(labels, texts)]
    allowed = "/".join(labels)
    return (
        f"Question: {question}\n"
        "Choices:\n" + "\n".join(lines) + "\n"
        "Reason step by step.\n"
        f'At the end, write exactly one line in the format: "Final answer: <{allowed}>".\n'
    )


def build_prompt_commonsenseqa(question: str, choices: Dict[str, List[str]]) -> str:
    return build_prompt_mc(question, choices["label"], choices["text"])


def build_prompt_strategyqa(question: str) -> str:
    return (
        f"Question: {question}\n"
        "Please reason step by step.\n"
        'At the end, write exactly one line in the format: "Final answer: Yes" or "Final answer: No".\n'
    )


def build_prompt_aqua(question: str, options: List[str]) -> str:
    labels = ["A", "B", "C", "D", "E"]
    lines_lab = []
    lines_txt = []
    for i, opt in enumerate(options[:5]):
        lab = labels[i]
        opt_clean = re.sub(r"^[A-E]\)?\s*[:\-]?\s*", "", str(opt).strip(), flags=re.IGNORECASE)
        lines_lab.append(lab)
        lines_txt.append(opt_clean)
    return build_prompt_mc(question, lines_lab, lines_txt)


def build_prompt_boolq(passage: str, question: str) -> str:
    return (
        f"Passage: {passage}\n"
        f"Question: {question}\n"
        "Please reason step by step.\n"
        'At the end, write exactly one line in the format: "Final answer: Yes" or "Final answer: No".\n'
    )


def build_prompt_piqa(goal: str, sol1: str, sol2: str) -> str:
    return (
        f"Goal: {goal}\n"
        "Choices:\n"
        f"A) {sol1}\n"
        f"B) {sol2}\n"
        "Please reason step by step.\n"
        'At the end, write exactly one line in the format: "Final answer: A" or "Final answer: B".\n'
    )


# -----------------------------
# Dataset helpers
# -----------------------------
def sample_hf_split(ds_split, n: int, seed: int):
    n = min(int(n), len(ds_split))
    if n <= 0:
        return ds_split.select([])
    return ds_split.shuffle(seed=seed).select(range(n))


def _pick_split(ds) -> str:
    # prefer train, else first available
    if isinstance(ds, dict) or hasattr(ds, "keys"):
        if "train" in ds:
            return "train"
        return list(ds.keys())[0]
    return "train"


def _try_load_dataset(path: str, name: Optional[str] = None):
    try:
        if name is None:
            return load_dataset(path)
        return load_dataset(path, name)
    except Exception as e:
        print(f"[Warn] failed to load dataset: {path}" + (f"/{name}" if name else "") + f" :: {repr(e)}")
        return None


def load_calib_prompts(n_prompts: int, seed: int) -> Dict[str, List[str]]:
    prompts: Dict[str, List[str]] = {}

    # gsm8k
    ds = _try_load_dataset("gsm8k", "main")
    if ds is not None:
        split = _pick_split(ds)
        rows = sample_hf_split(ds[split], n_prompts, seed + 1)
        prompts["gsm8k"] = [build_prompt_gsm8k(ex["question"]) for ex in rows]

    # commonsenseqa
    ds = _try_load_dataset("commonsense_qa")
    if ds is not None:
        split = _pick_split(ds)
        rows = sample_hf_split(ds[split], n_prompts, seed + 11)
        prompts["commonsenseqa"] = [build_prompt_commonsenseqa(ex["question"], ex["choices"]) for ex in rows]

    # strategyqa
    ds = _try_load_dataset("ChilleD/StrategyQA")
    if ds is not None:
        split = _pick_split(ds)
        rows = sample_hf_split(ds[split], n_prompts, seed + 21)
        prompts["strategyqa"] = [build_prompt_strategyqa(ex["question"]) for ex in rows]

    # aqua
    ds = _try_load_dataset("aqua_rat")
    if ds is not None:
        split = _pick_split(ds)
        rows = sample_hf_split(ds[split], n_prompts, seed + 31)
        prompts["aqua"] = [build_prompt_aqua(ex["question"], ex["options"]) for ex in rows]

    # ARC-Challenge
    ds = _try_load_dataset("ai2_arc", "ARC-Challenge")
    if ds is not None:
        split = _pick_split(ds)
        rows = sample_hf_split(ds[split], n_prompts, seed + 41)
        arc_prompts = []
        for ex in rows:
            q = ex.get("question", {})
            stem = q.get("stem", "") if isinstance(q, dict) else str(q)
            choices = q.get("choices", {}) if isinstance(q, dict) else {}
            labels = choices.get("label", [])
            texts = choices.get("text", [])
            if stem and labels and texts:
                arc_prompts.append(build_prompt_mc(stem, labels, texts))
        if len(arc_prompts) > 0:
            prompts["arc_challenge"] = arc_prompts

    # OpenBookQA
    ds = _try_load_dataset("openbookqa", "main")
    if ds is not None:
        split = _pick_split(ds)
        rows = sample_hf_split(ds[split], n_prompts, seed + 51)
        ob_prompts = []
        for ex in rows:
            q = ex.get("question_stem", "")
            ch = ex.get("choices", {})
            labels = ch.get("label", [])
            texts = ch.get("text", [])
            if q and labels and texts:
                ob_prompts.append(build_prompt_mc(q, labels, texts))
        if len(ob_prompts) > 0:
            prompts["openbookqa"] = ob_prompts

    # QASC
    ds = _try_load_dataset("qasc")
    if ds is not None:
        split = _pick_split(ds)
        rows = sample_hf_split(ds[split], n_prompts, seed + 61)
        qasc_prompts = []
        for ex in rows:
            q = ex.get("question", "")
            ch = ex.get("choices", {})
            labels = ch.get("label", [])
            texts = ch.get("text", [])
            if q and labels and texts:
                qasc_prompts.append(build_prompt_mc(q, labels, texts))
        if len(qasc_prompts) > 0:
            prompts["qasc"] = qasc_prompts

    # BoolQ
    ds = _try_load_dataset("boolq")
    if ds is not None:
        split = _pick_split(ds)
        rows = sample_hf_split(ds[split], n_prompts, seed + 71)
        bq_prompts = []
        for ex in rows:
            passage = ex.get("passage", "")
            q = ex.get("question", "")
            if passage and q:
                bq_prompts.append(build_prompt_boolq(passage, q))
        if len(bq_prompts) > 0:
            prompts["boolq"] = bq_prompts

    # PIQA
    ds = _try_load_dataset("piqa")
    if ds is not None:
        split = _pick_split(ds)
        rows = sample_hf_split(ds[split], n_prompts, seed + 81)
        piqa_prompts = []
        for ex in rows:
            goal = ex.get("goal", "")
            sol1 = ex.get("sol1", "")
            sol2 = ex.get("sol2", "")
            if goal and sol1 and sol2:
                piqa_prompts.append(build_prompt_piqa(goal, sol1, sol2))
        if len(piqa_prompts) > 0:
            prompts["piqa"] = piqa_prompts

    if len(prompts) == 0:
        raise RuntimeError("No datasets could be loaded; check HF datasets access / network / cache.")

    return prompts


# -----------------------------
# Decode last-token activation collector
# -----------------------------
from collections import defaultdict
from typing import DefaultDict

class DecodeLastTokenActivationCollector:
    """
    Collect last-token hidden states only during decode forward passes (seq_len == 1).
    storage[task][layer] -> list of [b, D] numpy chunks
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
                m = self.active_mask
                if m.dtype != torch.bool:
                    m = m.bool()
                if m.numel() == x.shape[0]:
                    x = x[m]
            if x.numel() == 0:
                return output

            self.storage[self._cur_task][layer_idx].append(x.detach().float().cpu().numpy())
            return output
        return _hook

    def get(self, task: str, layer: int) -> Optional[np.ndarray]:
        chunks = self.storage.get(task, {}).get(layer, [])
        if not chunks:
            return None
        return np.concatenate(chunks, axis=0)


# -----------------------------
# Sampling filters
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
# Decode collection
# -----------------------------
@torch.no_grad()
def collect_decode_last_token_states(
    model,
    tokenizer,
    prompts: List[str],
    collector: DecodeLastTokenActivationCollector,
    *,
    batch_size: int,
    max_prompt_len: int,
    calib_max_new_tokens: int,
    decoding: str,
    temperature: float,
    top_p: float,
    top_k: int,
) -> None:
    assert decoding in ["greedy", "sample"]
    device = next(model.parameters()).device
    model.eval()

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos = tokenizer.eos_token_id

    for i in tqdm(range(0, len(prompts), batch_size), desc="CollectDecode"):
        batch = prompts[i:i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_len,
        ).to(device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        B, _ = input_ids.shape

        unfinished = torch.ones(B, dtype=torch.bool, device=device)

        # Prefill (no capture)
        collector.set_capture(False, None)
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        logits = out.logits[:, -1, :]
        past = out.past_key_values

        # Decode loop (capture)
        for _step in range(int(calib_max_new_tokens)):
            if decoding == "greedy":
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                lt = logits / max(float(temperature), 1e-6)
                lt = top_k_filtering(lt, top_k=int(top_k))
                lt = top_p_filtering(lt, top_p=float(top_p))
                probs = torch.softmax(lt, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            # force eos for finished seqs (so shapes stay consistent)
            next_token = torch.where(
                unfinished.unsqueeze(-1),
                next_token,
                torch.full_like(next_token, eos),
            )

            newly_finished = unfinished & (next_token.squeeze(-1) == eos)
            unfinished[newly_finished] = False

            if not bool(unfinished.any().item()):
                break

            attention_mask = torch.cat(
                [attention_mask, torch.ones((B, 1), device=device, dtype=attention_mask.dtype)],
                dim=1,
            )

            # capture only unfinished sequences
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
# Sharedness computation
# -----------------------------
def center_and_balance(
    X_by_task: Dict[str, np.ndarray],
    *,
    per_task_max_states: int,
    balance_to: str,
    seed: int,
) -> Tuple[Dict[str, np.ndarray], int]:
    """
    1) cap each task to per_task_max_states (subsample)
    2) balance all tasks to same count (min or fixed int)
    3) task-wise centering
    """
    rng = np.random.default_rng(seed)

    # cap
    capped: Dict[str, np.ndarray] = {}
    for t, X in X_by_task.items():
        if X.shape[0] > per_task_max_states:
            idx = rng.choice(X.shape[0], size=per_task_max_states, replace=False)
            X = X[idx]
        capped[t] = X.astype(np.float32, copy=False)

    # balance
    if balance_to == "min":
        n0 = min(X.shape[0] for X in capped.values())
    else:
        n0 = int(balance_to)
        n0 = min(n0, min(X.shape[0] for X in capped.values()))

    balanced: Dict[str, np.ndarray] = {}
    for t, X in capped.items():
        if X.shape[0] > n0:
            idx = rng.choice(X.shape[0], size=n0, replace=False)
            X = X[idx]
        # task-wise centering
        X = X - X.mean(axis=0, keepdims=True)
        balanced[t] = X.astype(np.float32, copy=False)

    return balanced, n0


def compute_shared_indices_from_relvar(
    relvar_by_task: Dict[str, np.ndarray],
    *,
    tau: float,
    m_shared: int,
) -> List[int]:
    tasks = list(relvar_by_task.keys())
    rel = np.stack([relvar_by_task[t] for t in tasks], axis=0)  # [T, k]
    ok = (rel >= float(tau)).astype(np.int32)                   # [T, k]
    cnt = ok.sum(axis=0)                                        # [k]
    idx = np.where(cnt >= int(m_shared))[0]
    return idx.tolist()


def compute_relvar_in_basis(X: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    X: [n, D], Q: [D, k] (assumed approximately orthonormal; in practice PCA basis)
    returns relvar: [k] where relvar[i] = Var(XQ[:,i]) / sum_j Var(XQ[:,j])
    """
    Z = X @ Q  # [n, k]
    v = np.var(Z, axis=0)
    s = float(v.sum()) + 1e-12
    return (v / s).astype(np.float32, copy=False)


# -----------------------------
# Nulls
# -----------------------------
def null_perm_sharedcount(
    relvar_by_task: Dict[str, np.ndarray],
    *,
    tau: float,
    m_shared: int,
    trials: int,
    seed: int,
) -> Tuple[np.ndarray, float]:
    """
    Null-1: independently permute each task's relvar profile over components.
    Preserves each task's marginal distribution but destroys cross-task alignment.
    """
    rng = np.random.default_rng(seed)
    tasks = list(relvar_by_task.keys())
    k = relvar_by_task[tasks[0]].shape[0]

    counts = np.zeros(int(trials), dtype=np.int32)
    for b in range(int(trials)):
        ok_sum = np.zeros(k, dtype=np.int32)
        for t in tasks:
            perm = rng.permutation(k)
            rv = relvar_by_task[t][perm]
            ok_sum += (rv >= float(tau)).astype(np.int32)
        counts[b] = int((ok_sum >= int(m_shared)).sum())

    return counts, float(counts.mean())


def scramble_features_orthogonal(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Apply an orthogonal transform implemented as a permutation + sign flip on dimensions.
    This preserves per-task spectrum/energy but destroys alignment across tasks if done independently.
    """
    D = X.shape[1]
    perm = rng.permutation(D)
    signs = rng.choice([-1.0, 1.0], size=D).astype(np.float32)
    Xs = X[:, perm] * signs[None, :]
    return Xs.astype(np.float32, copy=False)


# -----------------------------
# Model loader
# -----------------------------
def load_model_and_tokenizer(model_name: str, device: str, model_dtype: str):
    if model_dtype == "fp32":
        dtype = torch.float32
    elif model_dtype == "fp16":
        dtype = torch.float16
    else:
        raise ValueError("model_dtype must be fp32 or fp16")

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
    default_out_json = os.path.join(THIS_DIR, "sharedness_existence.json")
    default_out_txt = os.path.splitext(default_out_json)[0] + ".txt"

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model_dtype", type=str, default="fp32", choices=["fp32", "fp16"])

    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--n_prompts", type=int, default=128)

    ap.add_argument("--max_prompt_len", type=int, default=512)
    ap.add_argument("--calib_max_new_tokens", type=int, default=128)
    ap.add_argument("--calib_decoding", type=str, default="greedy", choices=["greedy", "sample"])
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=4)

    ap.add_argument("--pca_var", type=float, default=0.95)
    ap.add_argument("--min_dim", type=int, default=1)
    ap.add_argument("--max_dim", type=int, default=4096)

    ap.add_argument("--tau", type=float, default=0.001)  # 0.1% rel var threshold
    ap.add_argument("--m_shared", type=str, default="all")  # "all" or an int

    ap.add_argument("--per_task_max_states", type=int, default=20000)
    ap.add_argument("--balance_to", type=str, default="min")  # "min" or an int

    ap.add_argument("--null_perm_trials", type=int, default=2000)
    ap.add_argument("--null_scramble_trials", type=int, default=0)  # strong but slow

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_json", type=str, default=default_out_json)
    ap.add_argument("--out_txt", type=str, default=default_out_txt,
                    help='Tee stdout prints into this txt file. Use "" or "none" to disable.')

    args = ap.parse_args()

    # setup tee stdout early
    orig_stdout = sys.stdout
    txt_f = None
    if _should_write_txt(args.out_txt):
        out_dir = os.path.dirname(os.path.abspath(args.out_txt))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        txt_f = open(args.out_txt, "w", encoding="utf-8")
        sys.stdout = TeeStdout(orig_stdout, txt_f)

    try:
        print(f"[Cmd] {' '.join(sys.argv)}")
        if txt_f is not None:
            print(f"[Log] tee stdout -> {args.out_txt}")

        set_global_seed(args.seed)

        layer_indices = [int(args.layer)]
        print(f"[Env] model={args.model} device={args.device} dtype={args.model_dtype} layer={layer_indices}")

        model, tok = load_model_and_tokenizer(args.model, args.device, args.model_dtype)
        layers, _ = get_model_layers(model)
        if args.layer >= len(layers):
            raise RuntimeError(f"layer={args.layer} out of range, num_layers={len(layers)}")

        hidden_dim = getattr(model.config, "hidden_size", None) or getattr(model.config, "n_embd", None)
        if hidden_dim is None:
            raise RuntimeError("Cannot infer hidden_dim")
        print(f"[Env] hidden_dim={hidden_dim}")

        # Load calibration prompts
        prompts_by_task = load_calib_prompts(args.n_prompts, args.seed)
        tasks = list(prompts_by_task.keys())
        print(f"[Data] tasks={tasks} n_prompts_per_task(target)={args.n_prompts}")
        for t in tasks:
            print(f"[Data] task={t} loaded_prompts={len(prompts_by_task[t])}")

        # Collector + hooks
        collector = DecodeLastTokenActivationCollector(layer_indices)
        handles = []
        for li in layer_indices:
            handles.append(layers[li].register_forward_hook(collector.make_hook(li)))

        # Collect decode states
        try:
            with torch.inference_mode():
                for task in tasks:
                    print(f"[Collect] task={task}")
                    collector.set_current_task(task)
                    collect_decode_last_token_states(
                        model=model,
                        tokenizer=tok,
                        prompts=prompts_by_task[task],
                        collector=collector,
                        batch_size=args.batch_size,
                        max_prompt_len=args.max_prompt_len,
                        calib_max_new_tokens=args.calib_max_new_tokens,
                        decoding=args.calib_decoding,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                    )
        finally:
            for h in handles:
                try:
                    h.remove()
                except Exception:
                    pass
            collector.set_capture(False, None)

        # Build X_by_task (single layer)
        X_raw: Dict[str, np.ndarray] = {}
        for task in tasks:
            X = collector.get(task, args.layer)
            if X is None or X.shape[0] == 0:
                raise RuntimeError(f"No activations collected for task={task}, layer={args.layer}")
            X_raw[task] = X
            print(f"[Collect] task={task} states={X.shape[0]} x {X.shape[1]}")

        # Fair preprocessing: cap, balance, task-center
        X_by_task, n0 = center_and_balance(
            X_raw,
            per_task_max_states=int(args.per_task_max_states),
            balance_to=str(args.balance_to),
            seed=args.seed + 999,
        )
        print(f"[Fair] balanced states per task = {n0}")

        # Build dict expected by compute_cross_task_subspace
        task_acts: Dict[str, Dict[int, np.ndarray]] = {t: {args.layer: X_by_task[t]} for t in tasks}

        # Pooled PCA
        joint_subspace, cross_dim, contributions, full_pca_info = compute_cross_task_subspace(
            task_acts,
            variance_threshold=float(args.pca_var),
            min_dim=int(args.min_dim),
            max_dim=int(args.max_dim),
            return_full_pca=True,
        )
        if joint_subspace is None or int(cross_dim) <= 0:
            raise RuntimeError("compute_cross_task_subspace failed")

        Q = joint_subspace.astype(np.float32, copy=False)  # [D, k]
        k = int(cross_dim)
        print(f"[PCA] cross_dim={k} / {hidden_dim}  (pca_var={args.pca_var})")

        # Compute per-task relative variance in this basis
        relvar_by_task: Dict[str, np.ndarray] = {}
        for t in tasks:
            relvar_by_task[t] = compute_relvar_in_basis(X_by_task[t], Q)

        # Sharedness threshold
        if args.m_shared == "all":
            m_shared = len(tasks)
        else:
            m_shared = int(args.m_shared)

        shared_idx = compute_shared_indices_from_relvar(relvar_by_task, tau=float(args.tau), m_shared=m_shared)
        obs_shared_count = int(len(shared_idx))

        # Diagnostics
        avg_rel = np.mean(np.stack([relvar_by_task[t] for t in tasks], axis=0), axis=0)
        top10 = np.argsort(-avg_rel)[:10].tolist()

        print("\n" + "=" * 80)
        print("[Observed Sharedness]")
        print("=" * 80)
        print(f"tasks={tasks}")
        print(f"tau={args.tau} m_shared={m_shared} (all={len(tasks)})")
        print(f"OBS shared_count={obs_shared_count} / cross_dim={k}")
        print("Top-10 components by avg relvar:", top10)
        print("Top-10 avg relvar:", [float(avg_rel[i]) for i in top10])

        # Null-1
        null1_counts, _ = null_perm_sharedcount(
            relvar_by_task,
            tau=float(args.tau),
            m_shared=m_shared,
            trials=int(args.null_perm_trials),
            seed=args.seed + 12345,
        )
        p1 = float((np.sum(null1_counts >= obs_shared_count) + 1) / (len(null1_counts) + 1))

        print("\n" + "=" * 80)
        print("[Null-1] relvar-permutation (fast)")
        print("=" * 80)
        print(f"trials={len(null1_counts)} null_mean={float(null1_counts.mean()):.2f} "
              f"p95={float(np.percentile(null1_counts, 95)):.2f} max={int(null1_counts.max())}")
        print(f"p-value (null>=obs) = {p1:.4g}")

        # Null-2
        null2_counts = []
        if int(args.null_scramble_trials) > 0:
            print("\n" + "=" * 80)
            print("[Null-2] per-task orthogonal scramble + recompute PCA (stronger, slower)")
            print("=" * 80)

            rng = np.random.default_rng(args.seed + 777)
            for b in range(int(args.null_scramble_trials)):
                X_scr: Dict[str, np.ndarray] = {}
                for t in tasks:
                    Xs = scramble_features_orthogonal(X_by_task[t], rng)
                    Xs = Xs - Xs.mean(axis=0, keepdims=True)
                    X_scr[t] = Xs.astype(np.float32, copy=False)

                task_acts_scr = {t: {args.layer: X_scr[t]} for t in tasks}
                joint2, k2, _, _ = compute_cross_task_subspace(
                    task_acts_scr,
                    variance_threshold=float(args.pca_var),
                    min_dim=int(args.min_dim),
                    max_dim=int(args.max_dim),
                    return_full_pca=True,
                )
                if joint2 is None or int(k2) <= 0:
                    null2_counts.append(0)
                    continue

                Q2 = joint2.astype(np.float32, copy=False)
                rel2 = {t: compute_relvar_in_basis(X_scr[t], Q2) for t in tasks}
                idx2 = compute_shared_indices_from_relvar(rel2, tau=float(args.tau), m_shared=m_shared)
                null2_counts.append(int(len(idx2)))
                print(f"  trial={b+1}/{args.null_scramble_trials}: cross_dim={int(k2)} shared_count={int(len(idx2))}")

            null2_counts = np.array(null2_counts, dtype=np.int32)
            p2 = float((np.sum(null2_counts >= obs_shared_count) + 1) / (len(null2_counts) + 1))
            print(f"[Null-2] mean={float(null2_counts.mean()):.2f} p95={float(np.percentile(null2_counts, 95)):.2f} "
                  f"max={int(null2_counts.max())}")
            print(f"[Null-2] p-value (null>=obs) = {p2:.4g}")
        else:
            p2 = None

        # Save JSON
        out_json_dir = os.path.dirname(os.path.abspath(args.out_json))
        if out_json_dir:
            os.makedirs(out_json_dir, exist_ok=True)

        out = {
            "config": {
                "model": args.model,
                "device": args.device,
                "model_dtype": args.model_dtype,
                "layer": int(args.layer),
                "n_prompts": int(args.n_prompts),
                "max_prompt_len": int(args.max_prompt_len),
                "calib_max_new_tokens": int(args.calib_max_new_tokens),
                "calib_decoding": args.calib_decoding,
                "temperature": float(args.temperature),
                "top_p": float(args.top_p),
                "top_k": int(args.top_k),
                "batch_size": int(args.batch_size),
                "pca_var": float(args.pca_var),
                "tau": float(args.tau),
                "m_shared": ("all" if args.m_shared == "all" else int(args.m_shared)),
                "per_task_max_states": int(args.per_task_max_states),
                "balance_to": args.balance_to,
                "null_perm_trials": int(args.null_perm_trials),
                "null_scramble_trials": int(args.null_scramble_trials),
                "seed": int(args.seed),
                "out_txt": (None if not _should_write_txt(args.out_txt) else args.out_txt),
            },
            "observed": {
                "tasks": tasks,
                "balanced_states_per_task": int(n0),
                "cross_dim": int(k),
                "shared_count": int(obs_shared_count),
                "shared_indices": shared_idx,
                "p_null1_perm": float(p1),
                "p_null2_scramble": (None if p2 is None else float(p2)),
            },
            "null1_perm_counts": null1_counts.astype(np.int32).tolist(),
            "null2_scramble_counts": (None if p2 is None else null2_counts.astype(np.int32).tolist()),
        }

        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2, default=to_py)

        print("\n" + "=" * 80)
        print("[Done]")
        print(f"Saved JSON: {args.out_json}")
        if _should_write_txt(args.out_txt):
            print(f"Saved TXT : {args.out_txt}")
        print("=" * 80)

    finally:
        # restore stdout / close log file
        try:
            sys.stdout.flush()
        except Exception:
            pass
        sys.stdout = orig_stdout
        if txt_f is not None:
            try:
                txt_f.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
