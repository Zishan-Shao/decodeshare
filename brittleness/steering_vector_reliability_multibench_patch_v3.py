

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
steering_vector_reliability_multibench_patch_v3.py

v3 = v2 + two minimal, high-yield fixes to improve SST-2 & RTE without big engineering:

(A) Per-task candidate calibration (still single-token when possible)
    We try a small set of semantically appropriate candidate pairs per task and pick the one
    with best *baseline forced-choice accuracy* on a small balanced calibration subset,
    averaged across templates. This avoids a "Yes/No" prior mismatch.

(B) Fix the known-bad RTE template (T1) by making it explicit forced-choice.
    All RTE templates are in a consistent forced-choice format with an explicit mapping:
      "Return '{POS}' if entailment holds; otherwise return '{NEG}'. Answer ({POS}/{NEG}): "

Retains:
  - True KV-cache decode measurement (prompt-boundary seq_len==1)
  - Partial projection beta sweep: v_beta = v - beta * B B^T v
  - Task-agnostic ("neutral") shared basis B by default

Optional extra benchmarks (add via --tasks):
  - qnli (GLUE)
  - imdb (sentiment)

Example:
 CUDA_VISIBLE_DEVICES=1 python steering_vector_reliability_multibench_patch_v3.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --device cuda --dtype fp32 \
  --layer 10 \
  --tasks boolq,rte,sst2 \
  --calib_per_class 256 --eval_per_class 128 \
  --basis_source neutral --basis_k 512 --basis_max_states 1024 \
  --betas 0,0.25,0.5,0.75,1.0 \
  --lambdas 0,0.5,1.0 \
  --n_rand 5 \
  --cand_calib_per_class 32 --cand_calib_templates all \
  --out_dir results/steer_repair_multibench_v3 \
  --show_per_template 1


"""

import argparse
import inspect
import json
import os
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# -----------------------------
# Reproducibility
# -----------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Model helpers
# -----------------------------
def get_block(model, layer_idx: int):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer_idx]
    raise ValueError("Cannot locate transformer blocks; adapt get_block() for your model.")


def get_model_device(model) -> torch.device:
    emb = model.get_input_embeddings()
    if emb is not None and hasattr(emb, "weight"):
        return emb.weight.device
    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    return torch.device("cpu")


def load_model_and_tokenizer(model_name: str, dtype_str: str, device: str, device_map: Optional[str]):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if dtype_str == "fp16":
        torch_dtype = torch.float16
    elif dtype_str == "bf16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    sig = inspect.signature(AutoModelForCausalLM.from_pretrained)
    dtype_kw = "dtype" if "dtype" in sig.parameters else "torch_dtype"

    kw = {dtype_kw: torch_dtype}
    if device_map is not None:
        kw["device_map"] = device_map

    model = AutoModelForCausalLM.from_pretrained(model_name, **kw)
    model.eval()

    if device_map is None:
        model.to(device)

    return model, tokenizer


# -----------------------------
# Hooks
# -----------------------------
class CollectLastTokenHook:
    def __init__(self, decode_only: bool):
        self.decode_only = decode_only
        self.records: List[torch.Tensor] = []

    def __call__(self, module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        if not isinstance(h, torch.Tensor) or h.ndim != 3:
            return output
        if self.decode_only and h.shape[1] != 1:
            return output
        self.records.append(h[:, -1, :].detach())
        return output


class AddVectorHook:
    def __init__(self, v: torch.Tensor, alpha: float, decode_only: bool = True):
        self.v = v.detach()
        self.alpha = float(alpha)
        self.decode_only = decode_only
        self._cache = {}

    def _v_on(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (device, dtype)
        if key not in self._cache:
            self._cache[key] = self.v.to(device=device, dtype=dtype)
        return self._cache[key]

    def __call__(self, module, inputs, output):
        if isinstance(output, tuple):
            h = output[0]
            rest = output[1:]
        else:
            h = output
            rest = None

        if not isinstance(h, torch.Tensor) or h.ndim != 3:
            return output
        if self.decode_only and h.shape[1] != 1:
            return output

        v = self._v_on(h.device, h.dtype)
        h2 = h.clone()
        h2[:, -1, :] = h2[:, -1, :] + self.alpha * v

        if rest is None:
            return h2
        return (h2, *rest)


# -----------------------------
# Neutral prompts for basis estimation
# -----------------------------
NEUTRAL_BASE_QUESTIONS = [
    "Explain how a refrigerator works.",
    "What are practical ways to improve sleep quality?",
    "Describe the water cycle in simple terms.",
    "Explain the difference between correlation and causation.",
    "Give a short overview of how computers represent numbers.",
    "Explain what inflation means and why it matters.",
    "What is a database index used for?",
    "How does photosynthesis work at a high level?",
    "Give tips for learning a new language effectively.",
    "Explain why regular exercise can improve mood.",
    "Describe how vaccines help protect communities.",
    "How does a neural network learn from data?",
    "Explain what debugging is and how to approach it.",
    "Describe pros and cons of remote work.",
    "Explain how to plan a weekly schedule productively.",
    "What is the greenhouse effect?",
    "Give a brief guide to writing clear emails.",
    "Explain what an API is.",
    "Describe how GPS location is determined.",
    "Explain why the sky looks blue.",
    "What causes seasons on Earth?",
    "Explain the idea of supply and demand.",
    "Explain how a microwave heats food.",
    "Describe the basics of public-key cryptography.",
    "Explain what a compiler does.",
    "Describe how to resolve conflicts in a team.",
    "Explain what version control is.",
    "Explain what overfitting is in machine learning.",
    "Describe the difference between RAM and storage.",
    "Explain why privacy matters online.",
    "Explain what latency is and why it matters.",
    "Describe what a cache is and why it helps.",
]

NEUTRAL_WRAPPERS = [
    "{q}",
    "Please answer the following:\n{q}",
    "Provide a concise explanation:\n{q}",
    "Give a helpful response:\n{q}",
]


def build_neutral_prompts(n: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    qs = NEUTRAL_BASE_QUESTIONS.copy()
    rng.shuffle(qs)
    prompts = []
    i = 0
    while len(prompts) < n:
        q = qs[i % len(qs)]
        w = NEUTRAL_WRAPPERS[(i // len(qs)) % len(NEUTRAL_WRAPPERS)]
        prompts.append(w.format(q=q))
        i += 1
    rng.shuffle(prompts)
    return prompts[:n]


# -----------------------------
# Candidate pairs + calibration
# -----------------------------
@dataclass
class CandidatePair:
    name: str
    pos: str  # label=True
    neg: str  # label=False


def default_candidate_pool(task_name: str) -> List[CandidatePair]:
    base = [
        CandidatePair("Yes/No", "Yes", "No"),
        CandidatePair("True/False", "True", "False"),
    ]
    if task_name in ("csqa_pair", "arc_challenge_pair"):
        return [CandidatePair("1/2", "1", "2")]
    if task_name in ("sst2", "imdb"):
        base += [
            CandidatePair("Positive/Negative", "Positive", "Negative"),
            CandidatePair("Good/Bad", "Good", "Bad"),
        ]
    if task_name in ("rte", "qnli"):
        base += [
            CandidatePair("Entails/Not", "Entails", "Not"),
            CandidatePair("Entailed/Not", "Entailed", "Not"),
        ]
    return base


# -----------------------------
# Task specs
# Templates include {POS}/{NEG} and end with trailing space
# -----------------------------
@dataclass
class TaskSpec:
    name: str
    ds_path: str
    ds_config: Optional[str]
    split_calib: str
    split_eval: str
    templates: List[str]
    parse_ex: Callable[[object, object], Optional[Tuple[Dict[str, str], bool]]]
    candidate_pool_fn: Callable[[str], List[CandidatePair]]
    # Optional custom loader for derived tasks (e.g., paired MCQ -> binary)
    custom_loader: Optional[Callable[["TaskSpec", int, int, str], List[Dict]]] = None


def make_boolq_task() -> TaskSpec:
    templates = [
        "Passage:\n{passage}\n\nQuestion: {question}\nAnswer ({POS}/{NEG}): ",
        "Read the passage and answer the question.\n\nPassage: {passage}\nQ: {question}\nA ({POS}/{NEG}): ",
        "Answer with only '{POS}' or '{NEG}'.\n\n{passage}\n\nQuestion: {question}\nFinal answer ({POS}/{NEG}): ",
    ]

    def parse_ex(ds, ex):
        if ex.get("passage", None) is None or ex.get("question", None) is None:
            return None
        return {"passage": ex["passage"], "question": ex["question"]}, bool(ex["answer"])

    return TaskSpec("boolq", "boolq", None, "train", "validation", templates, parse_ex, default_candidate_pool)


def make_glue_rte_task() -> TaskSpec:
    templates = [
        "Premise:\n{premise}\n\nHypothesis:\n{hypothesis}\n\nDoes the premise entail the hypothesis?\nAnswer ({POS}/{NEG}): ",
        # v3 fixed T1:
        "Reply with only '{POS}' or '{NEG}'.\nReturn '{POS}' if the premise entails the hypothesis; otherwise return '{NEG}'.\n\nPremise: {premise}\nHypothesis: {hypothesis}\nAnswer ({POS}/{NEG}): ",
        "One-token answer: '{POS}' or '{NEG}'.\n\nPremise: {premise}\nHypothesis: {hypothesis}\nEntailment ({POS}/{NEG}): ",
    ]

    def parse_ex(ds, ex):
        s1 = ex.get("sentence1", None)
        s2 = ex.get("sentence2", None)
        lab = ex.get("label", None)
        if s1 is None or s2 is None or lab is None or int(lab) < 0:
            return None
        try:
            names = ds.features["label"].names
            label_name = names[int(lab)]
            y = (label_name == "entailment")
        except Exception:
            y = (int(lab) == 0)
        return {"premise": s1, "hypothesis": s2}, bool(y)

    return TaskSpec("rte", "glue", "rte", "train", "validation", templates, parse_ex, default_candidate_pool)


def make_glue_sst2_task() -> TaskSpec:
    templates = [
        "Text: {sentence}\n\nIs the sentiment positive?\nAnswer ({POS}/{NEG}): ",
        "Reply with only '{POS}' or '{NEG}'.\nReturn '{POS}' if sentiment is positive; otherwise '{NEG}'.\n\nSentence: {sentence}\nAnswer ({POS}/{NEG}): ",
        "One-word answer: '{POS}' or '{NEG}'.\n\nSentence: {sentence}\nSentiment ({POS}/{NEG}): ",
    ]

    def parse_ex(ds, ex):
        s = ex.get("sentence", None)
        lab = ex.get("label", None)
        if s is None or lab is None or int(lab) < 0:
            return None
        try:
            names = ds.features["label"].names
            label_name = names[int(lab)]
            y = (label_name == "positive")
        except Exception:
            y = (int(lab) == 1)
        return {"sentence": s}, bool(y)

    return TaskSpec("sst2", "glue", "sst2", "train", "validation", templates, parse_ex, default_candidate_pool)


def make_glue_qnli_task() -> TaskSpec:
    templates = [
        "Question: {question}\nSentence: {sentence}\n\nDoes the sentence entail the answer to the question?\nAnswer ({POS}/{NEG}): ",
        "Reply with only '{POS}' or '{NEG}'.\nReturn '{POS}' if the sentence entails; otherwise '{NEG}'.\n\nQ: {question}\nS: {sentence}\nAnswer ({POS}/{NEG}): ",
        "One-word answer: '{POS}' or '{NEG}'.\n\nQuestion: {question}\nSentence: {sentence}\nEntails ({POS}/{NEG}): ",
    ]

    def parse_ex(ds, ex):
        q = ex.get("question", None)
        s = ex.get("sentence", None)
        lab = ex.get("label", None)
        if q is None or s is None or lab is None or int(lab) < 0:
            return None
        try:
            names = ds.features["label"].names
            label_name = names[int(lab)]
            y = (label_name == "entailment")
        except Exception:
            y = (int(lab) == 0)
        return {"question": q, "sentence": s}, bool(y)

    return TaskSpec("qnli", "glue", "qnli", "train", "validation", templates, parse_ex, default_candidate_pool)


def make_imdb_task() -> TaskSpec:
    templates = [
        "Review:\n{text}\n\nSentiment ({POS}/{NEG}): ",
        "Reply with only '{POS}' or '{NEG}'.\nReturn '{POS}' if sentiment is positive; otherwise '{NEG}'.\n\n{text}\n\nAnswer ({POS}/{NEG}): ",
        "One-word answer: '{POS}' or '{NEG}'.\n\n{text}\n\nSentiment ({POS}/{NEG}): ",
    ]

    def parse_ex(ds, ex):
        txt = ex.get("text", None)
        lab = ex.get("label", None)
        if txt is None or lab is None:
            return None
        if len(txt.strip()) < 20:
            return None
        y = (int(lab) == 1)
        return {"text": txt}, bool(y)

    return TaskSpec("imdb", "imdb", None, "train", "test", templates, parse_ex, default_candidate_pool)


def make_strategyqa_task() -> TaskSpec:
    """
    Verification (binary).
    Uses a common HF copy: tasksource/strategy-qa
    """
    templates = [
        "Question: {question}\n\nAnswer ({POS}/{NEG}): ",
        "Reply with only '{POS}' or '{NEG}'.\n\nQ: {question}\nA ({POS}/{NEG}): ",
        "One-token answer: '{POS}' or '{NEG}'.\n\n{question}\n\nAnswer ({POS}/{NEG}): ",
    ]

    def parse_ex(ds, ex):
        q = ex.get("question", None)
        ans = ex.get("answer", None)
        if q is None or ans is None:
            return None
        # allow bool or string
        if isinstance(ans, str):
            a = ans.strip().lower()
            if a in ("yes", "true", "1"):
                y = True
            elif a in ("no", "false", "0"):
                y = False
            else:
                return None
        else:
            y = bool(ans)
        return {"question": str(q)}, bool(y)

    # Match the dataset id used in existing LOTO configs/logs.
    return TaskSpec("strategyqa", "ChilleD/StrategyQA", None, "train", "validation", templates, parse_ex, default_candidate_pool)


def _load_balanced_paired_mcq(
    task: TaskSpec,
    *,
    n_per_class: int,
    seed: int,
    split: str,
    question_getter: Callable[[object], str],
    choices_getter: Callable[[object], List[str]],
    answer_index_getter: Callable[[object], int],
) -> List[Dict]:
    """
    Build a balanced binary dataset from an MCQ benchmark by sampling:
      - one correct option
      - one randomly chosen incorrect option
      - randomize order to create a balanced label
    """
    ds = load_dataset(task.ds_path, task.ds_config, split=split)
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)

    pos, neg = [], []
    for i in idxs:
        ex = ds[int(i)]
        try:
            q = question_getter(ex)
            choices = choices_getter(ex)
            aidx = int(answer_index_getter(ex))
        except Exception:
            continue

        if not q or not isinstance(choices, list) or len(choices) < 3:
            continue
        if aidx < 0 or aidx >= len(choices):
            continue

        wrong = [j for j in range(len(choices)) if j != aidx]
        if not wrong:
            continue
        widx = rng.choice(wrong)

        opt_correct = str(choices[aidx])
        opt_wrong = str(choices[widx])

        # Randomize order to create both labels.
        if rng.random() < 0.5:
            opt1, opt2 = opt_correct, opt_wrong
            y = True
        else:
            opt1, opt2 = opt_wrong, opt_correct
            y = False

        item = {"fields": {"question": str(q), "option1": opt1, "option2": opt2}, "label": bool(y)}
        (pos if y else neg).append(item)

        if len(pos) >= n_per_class and len(neg) >= n_per_class:
            break

    take = min(n_per_class, len(pos), len(neg))
    if take == 0:
        raise RuntimeError(f"[{task.name}] Not enough paired MCQ examples in split={split}: pos={len(pos)}, neg={len(neg)}")
    out = pos[:take] + neg[:take]
    rng.shuffle(out)
    return out


def make_csqa_pair_task() -> TaskSpec:
    """
    CommonsenseQA (CSQA) as paired binary (correct vs one distractor).
    Dataset: commonsense_qa
    """
    templates = [
        "Question: {question}\n\nOption 1: {option1}\nOption 2: {option2}\n\nReply with only '{POS}' or '{NEG}'. Return '{POS}' if Option 1 is correct; otherwise return '{NEG}'.\nAnswer ({POS}/{NEG}): ",
        "Choose the correct option.\n\n{question}\n(1) {option1}\n(2) {option2}\n\nReply with only '{POS}' or '{NEG}'. Return '{POS}' if (1) is correct; otherwise return '{NEG}'.\nAnswer ({POS}/{NEG}): ",
        "One-token answer: '{POS}' or '{NEG}'. Return '{POS}' for option 1; '{NEG}' for option 2.\n\nQ: {question}\n1) {option1}\n2) {option2}\n\nAnswer ({POS}/{NEG}): ",
    ]

    def parse_ex(ds, ex):
        if ex.get("question") is None:
            return None
        return {"question": ex["question"], "option1": ex["option1"], "option2": ex["option2"]}, bool(ex["label"])

    def loader(task: TaskSpec, n_per_class: int, seed: int, split: str) -> List[Dict]:
        def q_get(ex):
            q = ex.get("question", {})
            return str(q.get("stem", "")) if isinstance(q, dict) else str(q)

        def c_get(ex):
            ch = ex.get("choices", {})
            if not isinstance(ch, dict):
                return []
            texts = ch.get("text", [])
            return [str(t) for t in texts] if isinstance(texts, list) else []

        def a_get(ex):
            ch = ex.get("choices", {})
            if not isinstance(ch, dict):
                raise ValueError
            labels = ch.get("label", [])
            key = str(ex.get("answerKey", "")).strip()
            if key in labels:
                return labels.index(key)
            return int(key)

        return _load_balanced_paired_mcq(
            task,
            n_per_class=n_per_class,
            seed=seed,
            split=split,
            question_getter=q_get,
            choices_getter=c_get,
            answer_index_getter=a_get,
        )

    return TaskSpec("csqa_pair", "commonsense_qa", None, "train", "validation", templates, parse_ex, default_candidate_pool, custom_loader=loader)


def make_arc_challenge_pair_task() -> TaskSpec:
    """
    ARC-Challenge (ARC-C) as paired binary (correct vs one distractor).
    Dataset: ai2_arc / ARC-Challenge
    """
    templates = [
        "Question: {question}\n\nOption 1: {option1}\nOption 2: {option2}\n\nReply with only '{POS}' or '{NEG}'. Return '{POS}' if Option 1 is correct; otherwise return '{NEG}'.\nAnswer ({POS}/{NEG}): ",
        "Choose the correct option.\n\n{question}\n(1) {option1}\n(2) {option2}\n\nReply with only '{POS}' or '{NEG}'. Return '{POS}' if (1) is correct; otherwise return '{NEG}'.\nAnswer ({POS}/{NEG}): ",
        "One-token answer: '{POS}' or '{NEG}'. Return '{POS}' for option 1; '{NEG}' for option 2.\n\nQ: {question}\n1) {option1}\n2) {option2}\n\nAnswer ({POS}/{NEG}): ",
    ]

    def parse_ex(ds, ex):
        if ex.get("question") is None:
            return None
        return {"question": ex["question"], "option1": ex["option1"], "option2": ex["option2"]}, bool(ex["label"])

    def loader(task: TaskSpec, n_per_class: int, seed: int, split: str) -> List[Dict]:
        def q_get(ex):
            q = ex.get("question", "")
            if isinstance(q, dict):
                return str(q.get("stem", ""))
            return str(q)

        def c_get(ex):
            ch = ex.get("choices", {})
            if not isinstance(ch, dict):
                return []
            texts = ch.get("text", [])
            return [str(t) for t in texts] if isinstance(texts, list) else []

        def a_get(ex):
            ch = ex.get("choices", {})
            if not isinstance(ch, dict):
                raise ValueError
            labels = ch.get("label", [])
            key = str(ex.get("answerKey", "")).strip()
            if key in labels:
                return labels.index(key)
            return int(key)

        return _load_balanced_paired_mcq(
            task,
            n_per_class=n_per_class,
            seed=seed,
            split=split,
            question_getter=q_get,
            choices_getter=c_get,
            answer_index_getter=a_get,
        )

    return TaskSpec(
        "arc_challenge_pair",
        "ai2_arc",
        "ARC-Challenge",
        "train",
        "validation",
        templates,
        parse_ex,
        default_candidate_pool,
        custom_loader=loader,
    )


TASK_BUILDERS: Dict[str, Callable[[], TaskSpec]] = {
    "boolq": make_boolq_task,
    "strategyqa": make_strategyqa_task,
    "csqa_pair": make_csqa_pair_task,
    "arc_challenge_pair": make_arc_challenge_pair_task,
    "rte": make_glue_rte_task,
    "sst2": make_glue_sst2_task,
    "qnli": make_glue_qnli_task,  # optional
    "imdb": make_imdb_task,       # optional
}


# -----------------------------
# Balanced sampling
# -----------------------------
def _balanced_from_dataset(task: TaskSpec, ds, *, n_per_class: int, seed: int) -> List[Dict]:
    pos_idx, neg_idx = [], []

    for i in range(len(ds)):
        parsed = task.parse_ex(ds, ds[i])
        if parsed is None:
            continue
        _, y = parsed
        (pos_idx if y else neg_idx).append(i)

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise RuntimeError(f"[{task.name}] Not enough labeled examples in split={split}: pos={len(pos_idx)}, neg={len(neg_idx)}")

    rng = random.Random(seed)
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    take = min(n_per_class, len(pos_idx), len(neg_idx))
    pos_idx = pos_idx[:take]
    neg_idx = neg_idx[:take]
    idx = pos_idx + neg_idx
    rng.shuffle(idx)

    out = []
    for i in idx:
        parsed = task.parse_ex(ds, ds[int(i)])
        if parsed is None:
            continue
        fields, y = parsed
        out.append({"fields": fields, "label": bool(y)})
    return out


def load_balanced_examples(task: TaskSpec, *, n_per_class: int, seed: int, split: str) -> List[Dict]:
    def _try_splits(splits: List[str]) -> List[Dict]:
        last_err = None
        for sp in splits:
            try:
                if getattr(task, "custom_loader", None) is not None:
                    return task.custom_loader(task, n_per_class, seed, sp)
                ds = load_dataset(task.ds_path, task.ds_config, split=sp)
                return _balanced_from_dataset(task, ds, n_per_class=n_per_class, seed=seed)
            except Exception as e:
                last_err = e
                continue
        if last_err is not None:
            raise last_err
        raise RuntimeError(f"[{task.name}] failed to load any splits: {splits}")

    # split name fallback (some datasets use 'test' instead of 'validation', etc.)
    if split == task.split_calib:
        return _try_splits([split, "train", "validation", "test"])
    return _try_splits([split, "validation", "test", "train"])


def format_prompt(task: TaskSpec, ex: Dict, template_id: int, cand: CandidatePair) -> str:
    return task.templates[template_id].format(**ex["fields"], POS=cand.pos, NEG=cand.neg)


# -----------------------------
# Candidate ids + margin
# -----------------------------
@torch.no_grad()
def get_candidate_token_ids(tokenizer, cand: CandidatePair) -> Tuple[List[int], List[int]]:
    pos_ids = tokenizer(cand.pos, add_special_tokens=False, return_tensors="pt")["input_ids"][0].tolist()
    neg_ids = tokenizer(cand.neg, add_special_tokens=False, return_tensors="pt")["input_ids"][0].tolist()
    if len(pos_ids) == 0 or len(neg_ids) == 0:
        raise RuntimeError("Candidate tokenization produced empty ids.")
    return pos_ids, neg_ids


@torch.no_grad()
def prompt_boundary_logits(model, input_ids: torch.Tensor):
    T = input_ids.shape[1]
    past = None
    if T > 1:
        out_prefill = model(input_ids=input_ids[:, :-1], use_cache=True)
        past = out_prefill.past_key_values
    out_last = model(input_ids=input_ids[:, -1:], past_key_values=past, use_cache=True)
    logits0 = out_last.logits[:, -1, :].squeeze(0)
    past0 = out_last.past_key_values
    return logits0, past0


@torch.no_grad()
def score_sequence_from_logits(model, first_logits: torch.Tensor, past0, token_ids: List[int]) -> float:
    device = first_logits.device
    score = 0.0
    t0 = token_ids[0]
    score += torch.log_softmax(first_logits, dim=-1)[t0].item()

    past = past0
    prev = t0
    for t in token_ids[1:]:
        out = model(input_ids=torch.tensor([[prev]], device=device), past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :].squeeze(0)
        score += torch.log_softmax(logits, dim=-1)[t].item()
        prev = t
    return float(score)


@torch.no_grad()
def compute_raw_margin_pos_minus_neg(
    model,
    tokenizer,
    prompt_text: str,
    pos_ids: List[int],
    neg_ids: List[int],
    max_prompt_tokens: int,
) -> float:
    device = get_model_device(model)
    toks = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_tokens,
        add_special_tokens=True,
    )
    input_ids = toks["input_ids"].to(device)
    logits0, past0 = prompt_boundary_logits(model, input_ids)

    if len(pos_ids) == 1 and len(neg_ids) == 1:
        return float((logits0[pos_ids[0]] - logits0[neg_ids[0]]).item())

    s_pos = score_sequence_from_logits(model, logits0, past0, pos_ids)
    s_neg = score_sequence_from_logits(model, logits0, past0, neg_ids)
    return float(s_pos - s_neg)


@torch.no_grad()
def baseline_forced_choice_acc(
    model,
    tokenizer,
    task: TaskSpec,
    examples: List[Dict],
    cand: CandidatePair,
    *,
    template_ids: List[int],
    max_prompt_tokens: int,
) -> float:
    pos_ids, neg_ids = get_candidate_token_ids(tokenizer, cand)
    correct = 0
    total = 0
    for tid in template_ids:
        for ex in examples:
            prompt = format_prompt(task, ex, tid, cand)
            margin = compute_raw_margin_pos_minus_neg(model, tokenizer, prompt, pos_ids, neg_ids, max_prompt_tokens)
            pred = (margin > 0.0)
            gold = bool(ex["label"])
            correct += int(pred == gold)
            total += 1
    return correct / max(total, 1)


@torch.no_grad()
def choose_best_candidate_pair(
    model,
    tokenizer,
    task: TaskSpec,
    calib_examples: List[Dict],
    *,
    max_prompt_tokens: int,
    cand_calib_per_class: int,
    cand_calib_templates: str,
    require_single_token: bool,
    seed: int,
) -> Tuple[CandidatePair, Dict]:
    pool = task.candidate_pool_fn(task.name)

    rng = random.Random(seed)
    exs = calib_examples.copy()
    rng.shuffle(exs)
    pos = [e for e in exs if e["label"]]
    neg = [e for e in exs if not e["label"]]
    take = min(cand_calib_per_class, len(pos), len(neg))
    subset = pos[:take] + neg[:take]
    rng.shuffle(subset)

    tids = [0] if cand_calib_templates == "0" else list(range(len(task.templates)))

    scored = []
    for cand in pool:
        pos_ids, neg_ids = get_candidate_token_ids(tokenizer, cand)
        single = (len(pos_ids) == 1 and len(neg_ids) == 1)
        if require_single_token and (not single):
            continue
        acc = baseline_forced_choice_acc(
            model, tokenizer, task, subset, cand,
            template_ids=tids, max_prompt_tokens=max_prompt_tokens
        )
        scored.append({
            "name": cand.name,
            "pos": cand.pos,
            "neg": cand.neg,
            "pos_ids": pos_ids,
            "neg_ids": neg_ids,
            "single_token": single,
            "acc": float(acc),
        })

    if len(scored) == 0:
        # fallback: allow multi-token
        for cand in pool:
            pos_ids, neg_ids = get_candidate_token_ids(tokenizer, cand)
            acc = baseline_forced_choice_acc(
                model, tokenizer, task, subset, cand,
                template_ids=tids, max_prompt_tokens=max_prompt_tokens
            )
            scored.append({
                "name": cand.name,
                "pos": cand.pos,
                "neg": cand.neg,
                "pos_ids": pos_ids,
                "neg_ids": neg_ids,
                "single_token": (len(pos_ids) == 1 and len(neg_ids) == 1),
                "acc": float(acc),
            })

    def rank_key(x):
        return (x["acc"], 1.0 if x["single_token"] else 0.0, - (len(x["pos_ids"]) + len(x["neg_ids"])))
    scored_sorted = sorted(scored, key=rank_key, reverse=True)

    best = scored_sorted[0]
    best_cand = CandidatePair(best["name"], best["pos"], best["neg"])
    info = {
        "chosen": best,
        "candidates_tested": scored_sorted,
        "subset_size": int(len(subset)),
        "template_ids": tids,
        "require_single_token": bool(require_single_token),
    }
    return best_cand, info


# -----------------------------
# Prefill vs decode state collection + vector estimation
# -----------------------------
@torch.no_grad()
def collect_last_token_state_prefill(
    model,
    tokenizer,
    prompt_text: str,
    layer: int,
    max_prompt_tokens: int,
) -> torch.Tensor:
    device = get_model_device(model)
    block = get_block(model, layer)
    hook = CollectLastTokenHook(decode_only=False)
    handle = block.register_forward_hook(hook)
    try:
        toks = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_prompt_tokens, add_special_tokens=True)
        input_ids = toks["input_ids"].to(device)
        _ = model(input_ids=input_ids, use_cache=False)
        if len(hook.records) < 1:
            raise RuntimeError("No records captured in prefill; check hook placement.")
        return hook.records[-1].squeeze(0)
    finally:
        handle.remove()


@torch.no_grad()
def collect_last_token_state_decode_prompt_boundary(
    model,
    tokenizer,
    prompt_text: str,
    layer: int,
    max_prompt_tokens: int,
) -> torch.Tensor:
    device = get_model_device(model)
    block = get_block(model, layer)
    hook = CollectLastTokenHook(decode_only=True)
    handle = block.register_forward_hook(hook)
    try:
        toks = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_prompt_tokens, add_special_tokens=True)
        input_ids = toks["input_ids"].to(device)
        T = input_ids.shape[1]
        past = None
        if T > 1:
            out_prefill = model(input_ids=input_ids[:, :-1], use_cache=True)
            past = out_prefill.past_key_values
        _ = model(input_ids=input_ids[:, -1:], past_key_values=past, use_cache=True)
        if len(hook.records) < 1:
            raise RuntimeError("No decode-only record captured; check that seq_len==1 decode happened.")
        return hook.records[-1].squeeze(0)
    finally:
        handle.remove()


@torch.no_grad()
def estimate_mean_diff_vector(
    model,
    tokenizer,
    task: TaskSpec,
    examples: List[Dict],
    cand: CandidatePair,
    *,
    layer: int,
    template_ids: List[int],
    max_prompt_tokens: int,
    mode: str,
    prefill_append_gold_answer: bool,
) -> Tuple[torch.Tensor, Dict]:
    states = []
    labels = []
    norms = []
    for idx, ex in enumerate(examples):
        tid = template_ids[idx % len(template_ids)]
        prompt = format_prompt(task, ex, tid, cand)
        if mode == "prefill" and prefill_append_gold_answer:
            prompt = prompt + (cand.pos if ex["label"] else cand.neg)

        if mode == "prefill":
            h = collect_last_token_state_prefill(model, tokenizer, prompt, layer=layer, max_prompt_tokens=max_prompt_tokens)
        elif mode == "decode":
            h = collect_last_token_state_decode_prompt_boundary(model, tokenizer, prompt, layer=layer, max_prompt_tokens=max_prompt_tokens)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        states.append(h.float())
        labels.append(bool(ex["label"]))
        norms.append(float(h.float().norm().item()))

    H = torch.stack(states, dim=0)
    y = torch.tensor(labels, device=H.device, dtype=torch.bool)
    v = (H[y].mean(dim=0) - H[~y].mean(dim=0)).contiguous()

    stats = {
        "n": int(H.shape[0]),
        "n_pos": int(y.sum().item()),
        "n_neg": int((~y).sum().item()),
        "avg_state_norm": float(np.mean(norms)),
        "v_norm": float(v.norm().item()),
        "template_ids_used": template_ids,
        "mode": mode,
        "prefill_append_gold_answer": bool(prefill_append_gold_answer),
        "cand": {"name": cand.name, "pos": cand.pos, "neg": cand.neg},
    }
    return v, stats


# -----------------------------
# Shared basis estimation (decode-only)
# -----------------------------
@torch.no_grad()
def collect_decode_states_for_prompts(
    model,
    tokenizer,
    prompts: List[str],
    *,
    layer: int,
    max_prompt_tokens: int,
) -> torch.Tensor:
    device = get_model_device(model)
    block = get_block(model, layer)
    hook = CollectLastTokenHook(decode_only=True)
    handle = block.register_forward_hook(hook)
    try:
        for p in prompts:
            toks = tokenizer(p, return_tensors="pt", truncation=True, max_length=max_prompt_tokens, add_special_tokens=True)
            input_ids = toks["input_ids"].to(device)
            T = input_ids.shape[1]
            past = None
            if T > 1:
                out_prefill = model(input_ids=input_ids[:, :-1], use_cache=True)
                past = out_prefill.past_key_values
            _ = model(input_ids=input_ids[:, -1:], past_key_values=past, use_cache=True)

        if len(hook.records) == 0:
            raise RuntimeError("No decode states recorded for basis estimation.")
        return torch.cat([r.float() for r in hook.records], dim=0)
    finally:
        handle.remove()


@torch.no_grad()
def pca_basis(X: torch.Tensor, k: int) -> torch.Tensor:
    n, d = X.shape
    q = int(min(k, n - 1, d))
    if q < 1:
        raise RuntimeError(f"PCA basis q too small (n={n}, d={d}, k={k}).")
    U, S, V = torch.pca_lowrank(X, q=q, center=True, niter=2)
    B = V[:, :q].contiguous()
    B, _ = torch.linalg.qr(B, mode="reduced")
    return B


@torch.no_grad()
def estimate_shared_basis_decode(
    model,
    tokenizer,
    task: TaskSpec,
    calib_examples: List[Dict],
    cand: CandidatePair,
    *,
    layer: int,
    max_prompt_tokens: int,
    k: int,
    basis_source: str,
    basis_templates: str,
    max_states: int,
    seed: int,
) -> torch.Tensor:
    if basis_source not in ("neutral", "task"):
        raise ValueError("--basis_source must be 'neutral' or 'task' (or use --basis_source multitask which precomputes a shared basis).")
    if basis_templates not in ("0", "all"):
        raise ValueError("--basis_templates must be '0' or 'all'")

    rng = random.Random(seed)
    if basis_source == "neutral":
        n = max_states if max_states > 0 else 1024
        prompts = build_neutral_prompts(n=n, seed=seed)
    else:
        tids = [0] if basis_templates == "0" else list(range(len(task.templates)))
        prompts = []
        for ex in calib_examples:
            for tid in tids:
                prompts.append(format_prompt(task, ex, tid, cand))
        rng.shuffle(prompts)
        if max_states > 0:
            prompts = prompts[:max_states]

    X = collect_decode_states_for_prompts(model, tokenizer, prompts, layer=layer, max_prompt_tokens=max_prompt_tokens)
    return pca_basis(X, k=k)


@torch.no_grad()
def estimate_shared_basis_decode_from_prompts(
    model,
    tokenizer,
    prompts: List[str],
    *,
    layer: int,
    max_prompt_tokens: int,
    k: int,
) -> torch.Tensor:
    X = collect_decode_states_for_prompts(model, tokenizer, prompts, layer=layer, max_prompt_tokens=max_prompt_tokens)
    return pca_basis(X, k=k)


def build_multitask_basis_prompts(
    task_names: List[str],
    *,
    seed: int,
    n_per_class: int,
    basis_templates: str,
) -> List[str]:
    rng = random.Random(seed)
    prompts: List[str] = []
    for tname in task_names:
        if tname not in TASK_BUILDERS:
            continue
        task = TASK_BUILDERS[tname]()
        exs = load_balanced_examples(task, n_per_class=n_per_class, seed=seed + 1000 + hash(tname) % 10000, split=task.split_calib)
        cand = task.candidate_pool_fn(task.name)[0]
        tids = [0] if basis_templates == "0" else list(range(len(task.templates)))
        for ex in exs:
            for tid in tids:
                prompts.append(format_prompt(task, ex, tid, cand))
    rng.shuffle(prompts)
    return prompts


# -----------------------------
# Vector utilities
# -----------------------------
def project_partial(B: torch.Tensor, v: torch.Tensor, beta: float) -> torch.Tensor:
    return v - float(beta) * (B @ (B.T @ v))


def rescale_to(v: torch.Tensor, target_norm: float) -> torch.Tensor:
    return v / (v.norm() + 1e-12) * float(target_norm)


def sharedness(B: torch.Tensor, v: torch.Tensor) -> float:
    return float((B.T @ v).norm().item() / (v.norm().item() + 1e-12))


def fmt_beta(beta: float) -> str:
    if abs(beta - 0.0) < 1e-9:
        return "0"
    if abs(beta - 1.0) < 1e-9:
        return "1"
    return f"{beta:.2f}".rstrip("0").rstrip(".")


# -----------------------------
# Sign calibration
# -----------------------------
@torch.no_grad()
def mean_correct_shift_on_subset(
    model,
    tokenizer,
    task: TaskSpec,
    examples: List[Dict],
    cand: CandidatePair,
    *,
    layer: int,
    template_id: int,
    v: torch.Tensor,
    lam: float,
    max_prompt_tokens: int,
    pos_ids: List[int],
    neg_ids: List[int],
) -> float:
    block = get_block(model, layer)
    signs = np.array([1.0 if ex["label"] else -1.0 for ex in examples], dtype=np.float32)

    base = []
    for ex in examples:
        prompt = format_prompt(task, ex, template_id, cand)
        base.append(compute_raw_margin_pos_minus_neg(model, tokenizer, prompt, pos_ids, neg_ids, max_prompt_tokens))
    base = np.array(base, dtype=np.float32) * signs

    if lam == 0.0:
        steered = base.copy()
    else:
        hook = AddVectorHook(v=v, alpha=lam, decode_only=True)
        handle = block.register_forward_hook(hook)
        try:
            steered_raw = []
            for ex in examples:
                prompt = format_prompt(task, ex, template_id, cand)
                steered_raw.append(compute_raw_margin_pos_minus_neg(model, tokenizer, prompt, pos_ids, neg_ids, max_prompt_tokens))
            steered_raw = np.array(steered_raw, dtype=np.float32)
        finally:
            handle.remove()
        steered = steered_raw * signs

    return float((steered - base).mean())


@torch.no_grad()
def calibrate_sign_and_match_energy(
    model,
    tokenizer,
    task: TaskSpec,
    calib_small: List[Dict],
    cand: CandidatePair,
    *,
    layer: int,
    template_id: int,
    v: torch.Tensor,
    target_norm: float,
    max_prompt_tokens: int,
    pos_ids: List[int],
    neg_ids: List[int],
) -> torch.Tensor:
    v2 = rescale_to(v, target_norm)
    ms = mean_correct_shift_on_subset(
        model, tokenizer, task, calib_small, cand,
        layer=layer, template_id=template_id, v=v2, lam=1.0,
        max_prompt_tokens=max_prompt_tokens, pos_ids=pos_ids, neg_ids=neg_ids
    )
    return (-v2) if (ms < 0) else v2


# -----------------------------
# Evaluation
# -----------------------------
@torch.no_grad()
def eval_lambda_sweep(
    model,
    tokenizer,
    task: TaskSpec,
    eval_examples: List[Dict],
    cand: CandidatePair,
    *,
    layer: int,
    max_prompt_tokens: int,
    vecs: Dict[str, torch.Tensor],
    lambdas: List[float],
    n_rand: int,
    rand_seed: int,
    pos_ids: List[int],
    neg_ids: List[int],
) -> Dict:
    block = get_block(model, layer)
    last_lam = max(lambdas)
    signs = np.array([1.0 if ex["label"] else -1.0 for ex in eval_examples], dtype=np.float32)

    base_corr_by_t: List[np.ndarray] = []
    for tid in range(len(task.templates)):
        raw = []
        for ex in eval_examples:
            prompt = format_prompt(task, ex, tid, cand)
            raw.append(compute_raw_margin_pos_minus_neg(model, tokenizer, prompt, pos_ids, neg_ids, max_prompt_tokens))
        raw = np.array(raw, dtype=np.float32)
        base_corr_by_t.append(raw * signs)

    baseline_acc_by_t = [float((bc > 0.0).mean()) for bc in base_corr_by_t]

    def compute_shift_corr_for_vector(tid: int, v: torch.Tensor, lam: float) -> np.ndarray:
        base_corr = base_corr_by_t[tid]
        if lam == 0.0:
            return np.zeros_like(base_corr)

        hook = AddVectorHook(v=v, alpha=lam, decode_only=True)
        handle = block.register_forward_hook(hook)
        try:
            raw = []
            for ex in eval_examples:
                prompt = format_prompt(task, ex, tid, cand)
                raw.append(compute_raw_margin_pos_minus_neg(model, tokenizer, prompt, pos_ids, neg_ids, max_prompt_tokens))
            raw = np.array(raw, dtype=np.float32)
        finally:
            handle.remove()

        return raw * signs - base_corr

    def stats_from_shift(shift: np.ndarray) -> Dict[str, float]:
        return {"mean": float(shift.mean()), "std": float(shift.std()), "anti": float((shift < 0).mean())}

    def robust_from_template_stats(tstats: List[Dict[str, float]]) -> Dict[str, float]:
        means = np.array([t["mean"] for t in tstats], dtype=np.float32)
        antis = np.array([t["anti"] for t in tstats], dtype=np.float32)
        return {
            "mean_of_means": float(means.mean()),
            "std_across_templates": float(means.std(ddof=0)),
            "worst_case_mean": float(means.min()),
            "anti_mean": float(antis.mean()),
            "anti_worst": float(antis.max()),
        }

    results: Dict[str, Dict] = {
        "task": task.name,
        "cand": {"name": cand.name, "pos": cand.pos, "neg": cand.neg},
        "n_eval": int(len(eval_examples)),
        "n_templates": int(len(task.templates)),
        "lambdas": list(lambdas),
        "last_lambda": float(last_lam),
        "baseline": {
            "acc_by_template": baseline_acc_by_t,
            "acc_mean": float(np.mean(baseline_acc_by_t)) if len(baseline_acc_by_t) else 0.0,
            "acc_worst": float(np.min(baseline_acc_by_t)) if len(baseline_acc_by_t) else 0.0,
        },
        "methods": {},
    }

    # Non-random methods
    for name, v in [(k, vecs[k]) for k in vecs.keys() if not k.startswith("rand")]:
        pts = []
        per_template_last = []
        for tid in range(len(task.templates)):
            for lam in lambdas:
                shift = compute_shift_corr_for_vector(tid, v, lam)
                pts.append((lam, float(shift.mean())))
                if lam == last_lam:
                    per_template_last.append(stats_from_shift(shift))

        xs = np.array([p[0] for p in pts], dtype=np.float32)
        ys = np.array([p[1] for p in pts], dtype=np.float32)
        slope = float(np.cov(xs, ys, bias=True)[0, 1] / (xs.var() + 1e-12))
        results["methods"][name] = {"slope": slope, "per_template_last": per_template_last, "robust_last": robust_from_template_stats(per_template_last)}

    # Random control averaged over draws
    if "rand_ref" in vecs and n_rand > 0:
        v_ref = vecs["rand_ref"].to(get_model_device(model)).float()
        target_norm = float(v_ref.norm().item())
        rng = np.random.RandomState(rand_seed)

        slopes = []
        per_template_last_draws: List[List[Dict[str, float]]] = [[] for _ in range(len(task.templates))]
        for r in range(n_rand):
            vr = torch.from_numpy(rng.randn(v_ref.shape[0]).astype(np.float32)).to(v_ref.device)
            vr = rescale_to(vr, target_norm)

            pts = []
            per_template_last = []
            for tid in range(len(task.templates)):
                for lam in lambdas:
                    shift = compute_shift_corr_for_vector(tid, vr, lam)
                    pts.append((lam, float(shift.mean())))
                    if lam == last_lam:
                        per_template_last.append(stats_from_shift(shift))

            xs = np.array([p[0] for p in pts], dtype=np.float32)
            ys = np.array([p[1] for p in pts], dtype=np.float32)
            slope = float(np.cov(xs, ys, bias=True)[0, 1] / (xs.var() + 1e-12))
            slopes.append(slope)

            for tid in range(len(task.templates)):
                per_template_last_draws[tid].append(per_template_last[tid])

        per_template_last_agg = []
        for tid in range(len(task.templates)):
            means = np.array([d["mean"] for d in per_template_last_draws[tid]], dtype=np.float32)
            stds = np.array([d["std"] for d in per_template_last_draws[tid]], dtype=np.float32)
            antis = np.array([d["anti"] for d in per_template_last_draws[tid]], dtype=np.float32)
            per_template_last_agg.append({
                "mean": float(means.mean()),
                "std": float(stds.mean()),
                "anti": float(antis.mean()),
                "mean_std_across_draws": float(means.std(ddof=1)) if len(means) > 1 else 0.0,
                "anti_std_across_draws": float(antis.std(ddof=1)) if len(antis) > 1 else 0.0,
            })

        results["methods"]["rand_matched"] = {
            "slope_mean": float(np.mean(slopes)),
            "slope_std": float(np.std(slopes, ddof=1)) if len(slopes) > 1 else 0.0,
            "per_template_last": per_template_last_agg,
            "robust_last": robust_from_template_stats(per_template_last_agg),
        }

    return results


def print_task_report(task: TaskSpec, report: Dict, *, show_per_template: bool = True) -> None:
    cand = report.get("cand", {})
    print(f"\n=== Task: {task.name} | cand={cand.get('name','?')}({cand.get('pos','?')}/{cand.get('neg','?')}) | templates={report['n_templates']} | n_eval={report['n_eval']} | lambdas={report['lambdas']} (last={report['last_lambda']}) ===")
    if "baseline" in report:
        b = report["baseline"]
        print(f"Baseline acc (mean,worst): {b.get('acc_mean', 0.0):.3f}, {b.get('acc_worst', 0.0):.3f} | per-template: {b.get('acc_by_template', [])}")

    hdr = (
        "Method".ljust(26)
        + " | " + "slope".rjust(12)
        + " | " + "robust(mean±std,worst)".ljust(30)
        + " | " + "anti(mean,worst)".ljust(18)
    )
    print(hdr)
    print("-" * len(hdr))

    for mname, mrep in report["methods"].items():
        if mname == "rand_matched":
            slope_str = f"{mrep['slope_mean']:+0.4f}±{mrep['slope_std']:0.4f}"
        else:
            slope_str = f"{mrep['slope']:+0.4f}"
        rb = mrep["robust_last"]
        robust_str = f"{rb['mean_of_means']:+0.3f}±{rb['std_across_templates']:0.3f},{rb['worst_case_mean']:+0.3f}"
        anti_str = f"{rb['anti_mean']:.3f},{rb['anti_worst']:.3f}"
        print(mname.ljust(26) + " | " + slope_str.rjust(12) + " | " + robust_str.ljust(30) + " | " + anti_str.ljust(18))

    if show_per_template:
        print("\nPer-template stats at last lambda (mean_shift, std_shift, anti):")
        for mname, mrep in report["methods"].items():
            print(f"  - {mname}:")
            for tid, tstat in enumerate(mrep["per_template_last"]):
                if mname == "rand_matched":
                    ms = tstat["mean"]
                    ss = tstat["std"]
                    aa = tstat["anti"]
                    ms_sd = tstat.get("mean_std_across_draws", 0.0)
                    aa_sd = tstat.get("anti_std_across_draws", 0.0)
                    print(f"      T{tid}: {ms:+0.3f}±{ms_sd:0.3f}, {ss:0.3f}, {aa:0.3f}±{aa_sd:0.3f}")
                else:
                    print(f"      T{tid}: {tstat['mean']:+0.3f}, {tstat['std']:0.3f}, {tstat['anti']:.3f}")


# -----------------------------
# Run one task
# -----------------------------
@torch.inference_mode()
def run_one_task(
    model,
    tokenizer,
    task: TaskSpec,
    *,
    seed: int,
    layer: int,
    max_prompt_tokens: int,
    calib_per_class: int,
    eval_per_class: int,
    prefill_append_gold_answer: bool,
    calib_sign_n: int,
    basis_k: int,
    basis_source: str,
    basis_templates: str,
    basis_max_states: int,
    basis_override: Optional[torch.Tensor],
    betas: List[float],
    lambdas: List[float],
    n_rand: int,
    cand_calib_per_class: int,
    cand_calib_templates: str,
    cand_require_single_token: bool,
    v_est_templates: str,
    out_dir: Optional[str],
    show_per_template: bool,
) -> Dict:
    calib = load_balanced_examples(task, n_per_class=calib_per_class, seed=seed, split=task.split_calib)
    evalset = load_balanced_examples(task, n_per_class=eval_per_class, seed=seed + 999, split=task.split_eval)

    print(f"\n[{task.name}] calib={len(calib)} (balanced), eval={len(evalset)} (balanced)")

    # Candidate calibration
    chosen_cand, cand_info = choose_best_candidate_pair(
        model, tokenizer, task, calib,
        max_prompt_tokens=max_prompt_tokens,
        cand_calib_per_class=min(cand_calib_per_class, calib_per_class),
        cand_calib_templates=cand_calib_templates,
        require_single_token=bool(cand_require_single_token),
        seed=seed + 2024,
    )
    pos_ids, neg_ids = get_candidate_token_ids(tokenizer, chosen_cand)
    print(f"[{task.name}] Candidate calibration -> chosen {chosen_cand.name} ({chosen_cand.pos}/{chosen_cand.neg}), ids={pos_ids}/{neg_ids}, single={(len(pos_ids)==1 and len(neg_ids)==1)}")
    for i, c in enumerate(cand_info["candidates_tested"][: min(3, len(cand_info["candidates_tested"]))]):
        print(f"    top{i+1}: {c['name']} ({c['pos']}/{c['neg']}) acc={c['acc']:.3f} single={c['single_token']} ids={c['pos_ids']}/{c['neg_ids']}")

    # Basis
    if basis_override is not None:
        B = basis_override
    else:
        B = estimate_shared_basis_decode(
            model, tokenizer, task, calib, chosen_cand,
            layer=layer, max_prompt_tokens=max_prompt_tokens, k=basis_k,
            basis_source=basis_source, basis_templates=basis_templates,
            max_states=basis_max_states, seed=seed + 12345
        )
    device = get_model_device(model)
    B = B.to(device=device, dtype=torch.float32)

    # v estimation templates
    v_tids = [0] if v_est_templates == "0" else list(range(len(task.templates)))

    # decode v
    v_dec_raw, st_dec = estimate_mean_diff_vector(
        model, tokenizer, task, calib, chosen_cand,
        layer=layer, template_ids=v_tids, max_prompt_tokens=max_prompt_tokens,
        mode="decode", prefill_append_gold_answer=False
    )
    v_dec_raw = v_dec_raw.to(device=device)

    target_norm = float(v_dec_raw.float().norm().item())
    calib_small = calib[: max(1, min(calib_sign_n, len(calib)))]

    vecs: Dict[str, torch.Tensor] = {}
    decode_variants = []
    for beta in betas:
        v_beta_raw = project_partial(B, v_dec_raw, beta)
        v_beta = calibrate_sign_and_match_energy(
            model, tokenizer, task, calib_small, chosen_cand,
            layer=layer, template_id=0, v=v_beta_raw, target_norm=target_norm,
            max_prompt_tokens=max_prompt_tokens, pos_ids=pos_ids, neg_ids=neg_ids
        )
        name = "decode_est" if abs(beta) < 1e-9 else ("decode_fixed" if abs(beta - 1.0) < 1e-9 else f"decode_beta{fmt_beta(beta)}")
        vecs[name] = v_beta
        decode_variants.append({"beta": float(beta), "name": name, "sharedness": sharedness(B, v_beta)})

    vecs["rand_ref"] = vecs["decode_est"]

    diag = {
        "task": task.name,
        "layer": int(layer),
        "basis_k": int(B.shape[1]),
        "basis_source": basis_source,
        "basis_templates": basis_templates,
        "basis_max_states": int(basis_max_states),
        "v_est_templates": v_est_templates,
        "cand": {"name": chosen_cand.name, "pos": chosen_cand.pos, "neg": chosen_cand.neg},
        "cand_ids": {"pos_ids": pos_ids, "neg_ids": neg_ids},
        "cand_calibration": cand_info,
        "decode_stats": st_dec,
        "target_norm": target_norm,
        "decode_variants": decode_variants,
        "sharedness_decode_est": sharedness(B, vecs["decode_est"]),
    }

    print(f"[{task.name}] Diagnostics: sharedness(decode_est)={diag['sharedness_decode_est']:.4f}")
    for dv in decode_variants:
        print(f"  - {dv['name']}: beta={dv['beta']:.2f} sharedness={dv['sharedness']:.4f}")

    report = eval_lambda_sweep(
        model, tokenizer, task, evalset, chosen_cand,
        layer=layer, max_prompt_tokens=max_prompt_tokens,
        vecs=vecs, lambdas=lambdas,
        n_rand=n_rand, rand_seed=seed + 7777,
        pos_ids=pos_ids, neg_ids=neg_ids
    )

    print_task_report(task, report, show_per_template=show_per_template)

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, f"{task.name}_B.npy"), B.detach().cpu().numpy())
        for dv in decode_variants:
            np.save(os.path.join(out_dir, f"{task.name}_{dv['name']}.npy"), vecs[dv["name"]].detach().cpu().numpy())

        with open(os.path.join(out_dir, f"{task.name}_diag.json"), "w", encoding="utf-8") as f:
            json.dump(diag, f, indent=2)
        with open(os.path.join(out_dir, f"{task.name}_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        import csv
        rows = []
        for mname, mrep in report["methods"].items():
            rb = mrep["robust_last"]
            if mname == "rand_matched":
                slope = mrep["slope_mean"]
                slope_std = mrep["slope_std"]
            else:
                slope = mrep["slope"]
                slope_std = 0.0
            rows.append({
                "task": task.name,
                "cand_name": chosen_cand.name,
                "cand_pos": chosen_cand.pos,
                "cand_neg": chosen_cand.neg,
                "method": mname,
                "slope": float(slope),
                "slope_std": float(slope_std),
                "mean_of_means": rb["mean_of_means"],
                "std_across_templates": rb["std_across_templates"],
                "worst_case_mean": rb["worst_case_mean"],
                "anti_mean": rb["anti_mean"],
                "anti_worst": rb["anti_worst"],
            })
        csv_path = os.path.join(out_dir, f"{task.name}_summary.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)

    return {"diag": diag, "report": report}


# -----------------------------
# Main
# -----------------------------
def parse_floats_csv(s: str) -> List[float]:
    out = []
    for x in s.split(","):
        x = x.strip()
        if x:
            out.append(float(x))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--device_map", type=str, default=None)

    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_prompt_tokens", type=int, default=512)

    ap.add_argument(
        "--tasks",
        type=str,
        default="strategyqa,sst2,csqa_pair,arc_challenge_pair",
        help="Comma-separated tasks or 'all'. Default: strategyqa,sst2,csqa_pair,arc_challenge_pair. Extra: boolq,rte,qnli,imdb",
    )

    ap.add_argument("--calib_per_class", type=int, default=64)
    ap.add_argument("--eval_per_class", type=int, default=512)

    ap.add_argument("--prefill_append_gold_answer", type=int, default=1)
    ap.add_argument("--calib_sign_n", type=int, default=32)

    ap.add_argument("--basis_k", type=int, default=64)
    ap.add_argument("--basis_source", type=str, default="neutral", choices=["neutral", "task", "multitask"])
    ap.add_argument("--basis_templates", type=str, default="all", choices=["0", "all"])
    ap.add_argument("--basis_max_states", type=int, default=1024)
    ap.add_argument(
        "--basis_tasks",
        type=str,
        default="",
        help="Comma-separated task names used to build a shared basis when --basis_source multitask. If empty, uses --tasks.",
    )
    ap.add_argument(
        "--basis_tasks_per_class",
        type=int,
        default=32,
        help="Balanced examples per class per basis-task when --basis_source multitask.",
    )
    ap.add_argument(
        "--basis_holdout_current_task",
        type=int,
        default=0,
        help="If 1 and --basis_source=multitask, estimate a separate basis per eval task that excludes that task from --basis_tasks (leave-one-out).",
    )

    ap.add_argument("--betas", type=str, default="0,0.25,0.5,0.75,1.0")
    ap.add_argument("--lambdas", type=str, default="0,0.5,1.0")
    ap.add_argument("--n_rand", type=int, default=5)

    ap.add_argument("--cand_calib_per_class", type=int, default=32)
    ap.add_argument("--cand_calib_templates", type=str, default="all", choices=["0", "all"])
    ap.add_argument("--cand_require_single_token", type=int, default=1)

    ap.add_argument("--v_est_templates", type=str, default="0", choices=["0", "all"],
                    help="Estimate v using template 0 only (robustness test) or all templates (more stable).")

    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--show_per_template", type=int, default=1)

    args = ap.parse_args()
    seed_everything(args.seed)

    lambdas = parse_floats_csv(args.lambdas)
    if len(lambdas) < 2:
        raise ValueError("Provide at least 2 lambdas, e.g. --lambdas 0,0.5,1.0")
    betas = sorted(set([max(0.0, min(1.0, b)) for b in parse_floats_csv(args.betas)]))
    if len(betas) == 0:
        raise ValueError("Provide at least one beta.")

    if args.tasks.strip().lower() == "all":
        task_names = list(TASK_BUILDERS.keys())
    else:
        task_names = [t.strip().lower() for t in args.tasks.split(",") if t.strip()]
    unknown = [t for t in task_names if t not in TASK_BUILDERS]
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}. Available: {sorted(TASK_BUILDERS.keys())} or 'all'.")

    out_dir = args.out_dir
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2)

    print(f"[Load] model={args.model} dtype={args.dtype} device={args.device} device_map={args.device_map}")
    model, tokenizer = load_model_and_tokenizer(args.model, args.dtype, args.device, args.device_map)
    device = get_model_device(model)
    print(f"[Info] model_device={device} | layer={args.layer}")

    basis_override_by_task: Dict[str, Optional[torch.Tensor]] = {t: None for t in task_names}
    if args.basis_source == "multitask":
        if args.basis_tasks.strip():
            basis_task_names = [t.strip().lower() for t in args.basis_tasks.split(",") if t.strip()]
        else:
            basis_task_names = task_names
        unknown_b = [t for t in basis_task_names if t not in TASK_BUILDERS]
        if unknown_b:
            raise ValueError(f"Unknown --basis_tasks: {unknown_b}. Available: {sorted(TASK_BUILDERS.keys())}")
        per_task = bool(args.basis_holdout_current_task)
        cache: Dict[Tuple[str, ...], torch.Tensor] = {}
        for tname in task_names:
            use_tasks = basis_task_names
            if per_task:
                use_tasks = [t for t in basis_task_names if t != tname]
                if len(use_tasks) == 0:
                    use_tasks = basis_task_names
            key = tuple(use_tasks)
            if key not in cache:
                prompts = build_multitask_basis_prompts(
                    list(use_tasks),
                    seed=args.seed + 4242,
                    n_per_class=min(int(args.basis_tasks_per_class), int(args.calib_per_class)),
                    basis_templates=args.basis_templates,
                )
                if args.basis_max_states > 0:
                    prompts = prompts[: int(args.basis_max_states)]
                print(f"[Basis] multitask prompts={len(prompts)} tasks={list(use_tasks)}")
                cache[key] = estimate_shared_basis_decode_from_prompts(
                    model,
                    tokenizer,
                    prompts,
                    layer=args.layer,
                    max_prompt_tokens=args.max_prompt_tokens,
                    k=args.basis_k,
                ).to(device=device, dtype=torch.float32)
            basis_override_by_task[tname] = cache[key]

    all_outputs = []
    for tname in task_names:
        task = TASK_BUILDERS[tname]()
        task_out_dir = os.path.join(out_dir, tname) if out_dir is not None else None
        try:
            out = run_one_task(
                model, tokenizer, task,
                seed=args.seed,
                layer=args.layer,
                max_prompt_tokens=args.max_prompt_tokens,
                calib_per_class=args.calib_per_class,
                eval_per_class=args.eval_per_class,
                prefill_append_gold_answer=bool(args.prefill_append_gold_answer),
                calib_sign_n=args.calib_sign_n,
                basis_k=args.basis_k,
                basis_source=args.basis_source,
                basis_templates=args.basis_templates,
                basis_max_states=args.basis_max_states,
                basis_override=basis_override_by_task.get(tname, None),
                betas=betas,
                lambdas=lambdas,
                n_rand=args.n_rand,
                cand_calib_per_class=args.cand_calib_per_class,
                cand_calib_templates=args.cand_calib_templates,
                cand_require_single_token=bool(args.cand_require_single_token),
                v_est_templates=args.v_est_templates,
                out_dir=task_out_dir,
                show_per_template=bool(args.show_per_template),
            )
            all_outputs.append(out)
        except Exception as e:
            print(f"\n[Skip] task={tname} failed: {type(e).__name__}: {e}")
            continue

    # Aggregate summary
    if out_dir is not None and len(all_outputs) > 0:
        agg_rows = []
        for out in all_outputs:
            task = out["diag"]["task"]
            cand = out["diag"]["cand"]
            report = out["report"]
            for mname, mrep in report["methods"].items():
                rb = mrep["robust_last"]
                if mname == "rand_matched":
                    slope = mrep["slope_mean"]
                    slope_std = mrep["slope_std"]
                else:
                    slope = mrep["slope"]
                    slope_std = 0.0
                agg_rows.append({
                    "task": task,
                    "cand_name": cand["name"],
                    "cand_pos": cand["pos"],
                    "cand_neg": cand["neg"],
                    "method": mname,
                    "slope": float(slope),
                    "slope_std": float(slope_std),
                    "mean_of_means": rb["mean_of_means"],
                    "std_across_templates": rb["std_across_templates"],
                    "worst_case_mean": rb["worst_case_mean"],
                    "anti_mean": rb["anti_mean"],
                    "anti_worst": rb["anti_worst"],
                    "basis_source": out["diag"]["basis_source"],
                    "basis_k": out["diag"]["basis_k"],
                    "v_est_templates": out["diag"]["v_est_templates"],
                    "sharedness_decode_est": out["diag"]["sharedness_decode_est"],
                })

        import csv
        agg_csv = os.path.join(out_dir, "aggregate_summary.csv")
        with open(agg_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
            w.writeheader()
            for r in agg_rows:
                w.writerow(r)

        with open(os.path.join(out_dir, "aggregate_summary.json"), "w", encoding="utf-8") as f:
            json.dump(agg_rows, f, indent=2)

        print(f"\n[Saved] aggregate summaries to: {agg_csv}")

    print("\nDone.")


if __name__ == "__main__":
    main()
