# -*- coding: utf-8 -*-
"""
benchmark_dataloaders.py

Robust dataset loading + prompt construction for common HF benchmarks.

Key features:
  - No trust_remote_code (explicitly enforced with compatibility fallback).
  - Robust schema inference for MC choices / answer keys.
  - Automatic eval split fallback (prevents eval=0 -> NaNs).
  - Template randomization + deterministic choice shuffling.
  - Optional evaluation helpers: parse_prediction(), is_correct().
  - HF datasets>=4.x compatibility:
      * If a dataset repo relies on a loading script (*.py) and fails with
        "Dataset scripts are no longer supported", automatically fallback to
        revision="refs/convert/parquet" when possible.

Dependencies:
  pip install datasets numpy

Usage (example):
  from benchmark_dataloaders import load_selected_tasks
"""

from __future__ import annotations

import re
import json
import math
import string
import random
import hashlib
import ast
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable

import numpy as np
from datasets import load_dataset


# ============================================================
# HF datasets>=4.x: auto-parquet fallback (no scripts)
# ============================================================

AUTO_PARQUET_REVISION = "refs/convert/parquet"


def _is_dataset_script_unsupported_error(e: Exception) -> bool:
    msg = str(e)
    # 更鲁棒一些：不同版本可能有大小写/措辞差异
    needles = [
        "Dataset scripts are no longer supported",
        "dataset scripts are no longer supported",
        "Loading a dataset script is no longer supported",
        "loading a dataset script is no longer supported",
    ]
    return any(x in msg for x in needles)


def _load_dataset_compat(*args, **kwargs):
    """
    兼容不同 datasets 版本的参数差异。
    重点：trust_remote_code 在部分版本里不存在；若报 TypeError，则去掉该参数重试。
    """
    try:
        return load_dataset(*args, **kwargs)
    except TypeError as te:
        msg = str(te)
        # 只对 trust_remote_code 这个参数做兼容兜底，避免吞掉其它真实错误
        if "trust_remote_code" in msg and ("unexpected keyword" in msg or "got an unexpected keyword argument" in msg):
            kwargs2 = dict(kwargs)
            kwargs2.pop("trust_remote_code", None)
            return load_dataset(*args, **kwargs2)
        raise


def load_hf_dataset(
    hf_id: str,
    cfg: Optional[str] = None,
    *,
    revision: Optional[str] = None,
    **kwargs,
) -> Tuple[Any, Optional[str]]:
    """
    Load a HF dataset in a way compatible with datasets>=4.x.

    Returns:
      (ds, used_revision)

    Behavior:
      - Try (hf_id, cfg, revision) first.
      - If it fails with "Dataset scripts are no longer supported" and revision is None,
        retry with revision="refs/convert/parquet".
      - Explicitly enforces trust_remote_code=False (with compat fallback).
    """
    # 显式禁止 remote code（并允许在老版本 datasets 下自动去掉该参数重试）
    if "trust_remote_code" not in kwargs:
        kwargs["trust_remote_code"] = False

    def _call(rev: Optional[str]):
        if cfg is None:
            if rev is None:
                return _load_dataset_compat(hf_id, **kwargs)
            return _load_dataset_compat(hf_id, revision=rev, **kwargs)
        else:
            if rev is None:
                return _load_dataset_compat(hf_id, cfg, **kwargs)
            return _load_dataset_compat(hf_id, cfg, revision=rev, **kwargs)

    try:
        ds = _call(revision)
        return ds, revision
    except Exception as e:
        if revision is None and _is_dataset_script_unsupported_error(e):
            ds = _call(AUTO_PARQUET_REVISION)
            return ds, AUTO_PARQUET_REVISION
        raise


# ============================================================
# Core dataclass
# ============================================================

@dataclass
class Example:
    dataset: str
    ex_id: str
    prompt: str
    gold: str  # canonical gold label/answer; may be "" if unlabeled and require_gold=False


# ============================================================
# Repro / stable seeding
# ============================================================

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def stable_int_seed(*items: Any) -> int:
    s = "|".join(map(str, items)).encode("utf-8")
    h = hashlib.md5(s).hexdigest()
    return int(h[:8], 16)


# ============================================================
# Templates
# ============================================================

MC_LETTERS_8 = list("ABCDEFGH")
MC_LETTERS_5 = list("ABCDE")
MC_LETTERS_4 = list("ABCD")

GSM8K_TEMPLATES = [
    "Question: {q}\nLet's think step by step.\nAt the end, write exactly one line in the format: \"Final answer: <number>\".\n",
    "Solve the problem carefully.\nProblem: {q}\nShow your reasoning.\nAt the end, output one line: \"Final answer: <number>\".\n",
    "You are a math tutor.\nQuestion: {q}\nReason step by step.\nFinish with: \"Final answer: <number>\".\n",
]

YN_TEMPLATES = [
    "Question: {q}\nPlease reason step by step.\nAt the end, write exactly one line: \"Final answer: Yes\" or \"Final answer: No\".\n",
    "Decide whether the statement is true.\nQuestion: {q}\nExplain briefly.\nEnd with: \"Final answer: Yes\" or \"Final answer: No\".\n",
    "Question: {q}\nThink step by step.\nProvide a single final line: \"Final answer: Yes\" or \"Final answer: No\".\n",
]

MC_TEMPLATES = [
    "Question: {q}\nChoices:\n{choices}\nReason step by step.\nAt the end, write exactly one line: \"Final answer: <{letters}>\".\n",
    "Multiple-choice question:\n{q}\nOptions:\n{choices}\nThink step by step.\nFinish with one line: \"Final answer: <{letters}>\".\n",
    "Answer the question.\n{q}\n{choices}\nShow reasoning.\nConclude with: \"Final answer: <{letters}>\".\n",
]

QA_TEMPLATES = [
    "Use the context to answer the question.\nContext:\n{context}\n\nQuestion: {q}\nLet's think step by step.\nAt the end, write exactly one line: \"Final answer: <answer>\".\n",
    "Read the passages and answer.\nContext:\n{context}\n\nQ: {q}\nExplain your reasoning.\nEnd with: \"Final answer: <answer>\".\n",
    "Answer the question based only on the provided context.\n{context}\n\nQuestion: {q}\nReason step by step.\nFinish with: \"Final answer: <answer>\".\n",
]

LM_NEXTWORD_TEMPLATES = [
    "Predict the next word in the text.\nContext:\n{context}\n\nAt the end, write exactly one line: \"Final answer: <word>\".\n",
    "Given the context, what is the next word?\nContext:\n{context}\n\nFinish with exactly one line: \"Final answer: <word>\".\n",
    "Next-word prediction.\nText so far:\n{context}\n\nEnd with: \"Final answer: <word>\".\n",
]


def choose_template_id(ex_id: str, num_templates: int, seed: int) -> int:
    if num_templates <= 1:
        return 0
    return stable_int_seed(seed, "tmpl", ex_id) % num_templates


def maybe_add_answer_prefix(prompt: str, add_prefix: bool, answer_prefix: str) -> str:
    if not add_prefix:
        return prompt
    ap = "" if answer_prefix is None else str(answer_prefix)
    # Common CLI mistake: --answer_prefix 0 (string) while add_prefix=1
    # Treat "0"/"none"/"null" as empty.
    if ap.strip().lower() in {"0", "none", "null", "false"}:
        return prompt
    if not ap.startswith("\n"):
        ap = "\n" + ap
    return prompt.rstrip() + ap


def format_choices(lines: List[str]) -> str:
    return "\n".join(lines)


def build_prompt_gsm8k(q: str, template_id: int) -> str:
    tmpl = GSM8K_TEMPLATES[template_id % len(GSM8K_TEMPLATES)]
    return tmpl.format(q=q)


def build_prompt_yesno(q: str, template_id: int) -> str:
    tmpl = YN_TEMPLATES[template_id % len(YN_TEMPLATES)]
    return tmpl.format(q=q)


def build_prompt_mc(q: str, option_texts: List[str], option_labels: List[str], template_id: int) -> str:
    choice_lines = [f"{lab}) {txt}" for lab, txt in zip(option_labels, option_texts)]
    letters = "/".join(option_labels)
    tmpl = MC_TEMPLATES[template_id % len(MC_TEMPLATES)]
    return tmpl.format(q=q, choices=format_choices(choice_lines), letters=letters)


def build_prompt_qa(q: str, context: str, template_id: int) -> str:
    tmpl = QA_TEMPLATES[template_id % len(QA_TEMPLATES)]
    return tmpl.format(q=q, context=context)


def build_prompt_nextword(context: str, template_id: int) -> str:
    tmpl = LM_NEXTWORD_TEMPLATES[template_id % len(LM_NEXTWORD_TEMPLATES)]
    return tmpl.format(context=context)


# ============================================================
# Choice shuffling (deterministic)
# ============================================================

def safe_upper(x: Any) -> str:
    return str(x).strip().upper()


def shuffle_choices_if_needed(
    option_texts: List[str],
    option_labels: List[str],
    gold_label: str,
    do_shuffle: bool,
    seed: int,
    ex_id: str,
) -> Tuple[List[str], List[str], str]:
    if not do_shuffle:
        return option_texts, option_labels, gold_label

    rng = np.random.default_rng(stable_int_seed(seed, "shuffle", ex_id))
    idx = np.arange(len(option_texts))
    rng.shuffle(idx)

    texts2 = [option_texts[i] for i in idx]
    new_labels = MC_LETTERS_8[: len(texts2)]

    old_pos = option_labels.index(gold_label) if gold_label in option_labels else None
    if old_pos is None:
        new_gold = ""
    else:
        new_pos = int(np.where(idx == old_pos)[0][0])
        new_gold = new_labels[new_pos]

    return texts2, new_labels, new_gold


# ============================================================
# HF split sampling
# ============================================================

def sample_hf_split(ds_split, n: int, seed: int):
    n = min(n, len(ds_split))
    if n <= 0:
        return ds_split.select([])
    return ds_split.shuffle(seed=seed).select(range(n))


# ============================================================
# Robust schema inference for MC tasks
# ============================================================

def _extract_question_text(ex: dict) -> str:
    if "question_stem" in ex:
        return str(ex["question_stem"])
    q = ex.get("question", None)
    if isinstance(q, dict):
        for k in ["stem", "question", "text"]:
            if k in q:
                return str(q[k])
    if isinstance(q, str):
        return q
    for k in ["query", "prompt", "text", "sentence", "problem"]:
        if k in ex:
            return str(ex[k])
    return str(q) if q is not None else ""


def _extract_choices(ex: dict) -> Tuple[List[str], List[str]]:
    """
    Robust extractor that supports many HF schemas:
      - question.choices as dict(text,label) or list[dict]
      - top-level choices as dict or list[dict] or list[str]
      - options/endings/candidates/answers keys
      - dict keyed by 'A'/'B'/... or '1'/'2'/...
      - string blob containing lines or "A) ... B) ..." patterns
    """

    def _from_obj(obj: Any) -> Tuple[List[str], List[str]]:
        if obj is None:
            return [], []

        if isinstance(obj, dict):
            if "text" in obj and isinstance(obj.get("text"), (list, tuple)):
                texts = [str(x) for x in obj.get("text", [])]
                labels = [safe_upper(x) for x in obj.get("label", [])] if "label" in obj else MC_LETTERS_8[:len(texts)]
                if len(labels) != len(texts):
                    labels = MC_LETTERS_8[:len(texts)]
                return texts, labels

            keys = list(obj.keys())
            letter_keys = []
            for k in keys:
                if isinstance(k, str) and re.fullmatch(r"[A-H]", k.strip().upper()):
                    letter_keys.append(k.strip().upper())
            if letter_keys and len(letter_keys) == len(keys) and len(letter_keys) >= 2:
                letter_keys = sorted(letter_keys, key=lambda x: MC_LETTERS_8.index(x))
                texts = [str(obj[k]) for k in letter_keys]
                labels = letter_keys
                return texts, labels

            digit_keys = []
            for k in keys:
                if isinstance(k, str) and k.strip().isdigit():
                    digit_keys.append(int(k.strip()))
            if digit_keys and len(digit_keys) == len(keys) and len(digit_keys) >= 2:
                digit_keys_sorted = sorted(digit_keys)
                texts = [str(obj[str(k)]) for k in digit_keys_sorted]
                labels = [str(k) for k in digit_keys_sorted]
                return texts, labels

            opt_pairs = []
            for k, v in obj.items():
                if not isinstance(k, str):
                    continue
                m = re.fullmatch(r"(?:option|choice|answer|ending|cand|candidate|opt)[_\-\s]?(\d+)", k.strip().lower())
                if m:
                    opt_pairs.append((int(m.group(1)), v))
            if opt_pairs:
                opt_pairs.sort(key=lambda x: x[0])
                texts = [str(v) for _, v in opt_pairs]
                labels = MC_LETTERS_8[:len(texts)]
                return texts, labels

            for kk in ["choices", "options", "candidates", "answers", "endings", "answer_choices", "answerChoices"]:
                if kk in obj:
                    texts, labels = _from_obj(obj[kk])
                    if texts:
                        return texts, labels

            return [], []

        if isinstance(obj, (list, tuple)):
            if len(obj) == 0:
                return [], []
            if isinstance(obj[0], dict):
                texts, labels = [], []
                for i, c in enumerate(obj):
                    txt = c.get("text", c.get("content", c.get("answer", c.get("option", ""))))
                    lab = c.get("label", c.get("key", c.get("id", MC_LETTERS_8[i])))
                    texts.append(str(txt))
                    labels.append(safe_upper(lab))
                if len(labels) != len(texts):
                    labels = MC_LETTERS_8[:len(texts)]
                return texts, labels
            texts = [str(x) for x in obj]
            labels = MC_LETTERS_8[:len(texts)]
            return texts, labels

        if isinstance(obj, str):
            s = obj.strip()
            if not s:
                return [], []

            pat = re.compile(
                r"([A-H])\s*[\)\.\:\-]\s*(.*?)(?=(?:\b[A-H]\s*[\)\.\:\-])|$)",
                re.IGNORECASE | re.DOTALL
            )
            segs = pat.findall(s)
            if segs and len(segs) >= 2:
                labels = [safe_upper(a) for a, _ in segs]
                texts = [" ".join(b.split()).strip() for _, b in segs]
                return texts, labels

            lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
            if len(lines) >= 2:
                cleaned = [re.sub(r"^[A-H]\s*[\)\.\:\-]\s*", "", ln, flags=re.IGNORECASE).strip() for ln in lines]
                labels = MC_LETTERS_8[:len(cleaned)]
                return cleaned, labels

            return [], []

        return [], []

    q = ex.get("question", None)
    if isinstance(q, dict):
        for key in ["choices", "options", "answer_choices", "answerChoices"]:
            if key in q:
                texts, labels = _from_obj(q[key])
                if texts:
                    return texts, labels

    if "choices" in ex:
        texts, labels = _from_obj(ex["choices"])
        if texts:
            return texts, labels

    for k in ["options", "endings", "candidates", "answers", "answer_choices", "answerChoices"]:
        if k in ex:
            texts, labels = _from_obj(ex[k])
            if texts:
                return texts, labels

    texts = []
    for i in range(8):
        for k in [f"option{i}", f"choice{i}", f"ending{i}", f"answer{i}", f"candidate{i}"]:
            if k in ex:
                texts.append(str(ex[k]))
                break
    if texts:
        return texts, MC_LETTERS_8[:len(texts)]

    return [], []


def _extract_answer_key(ex: dict) -> str:
    for k in [
        "answerKey", "answer_key",
        "label", "answer", "correct", "gold",
        "correct_answer", "correct_option", "correctChoice",
        "answer_idx", "answer_index",
    ]:
        if k not in ex:
            continue

        v = ex[k]
        if isinstance(v, dict):
            for kk in ["label", "key", "answerKey", "answer", "gold", "correct"]:
                if kk in v:
                    v = v[kk]
                    break

        if v is None:
            continue

        if isinstance(v, (int, np.integer)):
            return str(int(v))

        s = str(v).strip().upper()
        if not s:
            continue

        m = re.search(r"\b([A-H])\b", s)
        if m:
            return m.group(1)

        md = re.search(r"\b(\d{1,2})\b", s)
        if md:
            return md.group(1)

        return s

    return ""


def _canonicalize_mc(texts: List[str], labels: List[str], gold0: Any) -> Tuple[List[str], List[str], str]:
    texts = [str(x) for x in (texts or [])]
    n = len(texts)
    if n == 0:
        return [], [], ""

    canon = MC_LETTERS_8[:n]
    lbls = [safe_upper(x) for x in (labels or [])]
    if len(lbls) != n:
        lbls = canon

    g0 = safe_upper(gold0)

    numeric_lbls = []
    for l in lbls:
        if l.isdigit():
            numeric_lbls.append(int(l))
    indexing = None
    if len(numeric_lbls) == n:
        if min(numeric_lbls) == 0 and max(numeric_lbls) == n - 1:
            indexing = "0"
        elif min(numeric_lbls) == 1 and max(numeric_lbls) == n:
            indexing = "1"

    def map_digit(j: int) -> str:
        if indexing == "0":
            return canon[j] if 0 <= j < n else ""
        if indexing == "1":
            return canon[j - 1] if 1 <= j <= n else ""
        if 0 <= j < n:
            return canon[j]
        if 1 <= j <= n:
            return canon[j - 1]
        return ""

    if g0 in lbls:
        return texts, canon, canon[lbls.index(g0)]

    if g0 in canon:
        return texts, canon, g0

    if g0.isdigit():
        return texts, canon, map_digit(int(g0))

    m = re.search(r"\b([A-H])\b", g0)
    if m and m.group(1) in canon:
        return texts, canon, m.group(1)

    return texts, canon, ""


# ============================================================
# Generic split fallback helper
# ============================================================

def _build_from_splits_with_fallback(
    ds,
    dataset_name: str,
    build_one: Callable[[dict, str], Tuple[str, str]],
    *,
    n_subspace: int,
    n_eval: int,
    seed: int,
    sub_candidates: List[str],
    eval_candidates: List[str],
    require_gold_eval: bool = True,
) -> Tuple[List[Example], List[Example], Dict[str, Any]]:
    """
    Try multiple splits; pick the first that yields non-empty examples.
    Eval split is required to yield at least one labeled example if require_gold_eval=True,
    except when n_eval <= 0.
    """

    def make_examples(split_name: str, rows, require_gold: bool) -> List[Example]:
        out: List[Example] = []
        for i, ex in enumerate(rows):
            ex_id = f"{dataset_name}-{split_name}-{i}"
            p, g = build_one(ex, ex_id)
            p = (p or "").strip()
            g = (g or "").strip()
            if not p:
                continue
            if require_gold and not g:
                continue
            out.append(Example(dataset=dataset_name, ex_id=ex_id, prompt=p, gold=g))
        return out

    sub_exs: List[Example] = []
    sub_split: Optional[str] = None
    for j, sp in enumerate(sub_candidates):
        if sp not in ds:
            continue
        rows = sample_hf_split(ds[sp], n_subspace, seed + 1000 + 13 * j)
        sub_exs = make_examples(sp, rows, require_gold=False)
        if len(sub_exs) > 0:
            sub_split = sp
            break
    if sub_split is None:
        raise RuntimeError(
            f"[{dataset_name}] Could not build ANY subspace prompts from splits={list(ds.keys())}. "
            f"Check schema extraction for this dataset."
        )

    if n_eval <= 0:
        meta = {
            "prompt_format": "raw_text",
            "subspace_split": sub_split,
            "eval_split": None,
            "available_splits": list(ds.keys()),
            "eval_skipped": True,
            "require_gold_eval": require_gold_eval,
            "n_eval_requested": int(n_eval),
        }
        return sub_exs, [], meta

    eval_exs: List[Example] = []
    eval_split: Optional[str] = None
    for j, sp in enumerate(eval_candidates):
        if sp not in ds:
            continue
        rows = sample_hf_split(ds[sp], n_eval, seed + 2000 + 17 * j)
        eval_exs = make_examples(sp, rows, require_gold=require_gold_eval)
        if len(eval_exs) > 0:
            eval_split = sp
            break
    if eval_split is None:
        raise RuntimeError(
            f"[{dataset_name}] Could not build ANY eval examples (require_gold={require_gold_eval}) "
            f"from candidate splits={eval_candidates}. Available splits={list(ds.keys())}. "
            f"Likely the 'test' split is unlabeled; prefer validation/dev."
        )

    meta = {
        "prompt_format": "raw_text",  # ⭐明确：loader 只输出纯文本 prompt
        "subspace_split": sub_split,
        "eval_split": eval_split,
        "available_splits": list(ds.keys()),
        "eval_skipped": False,
        "require_gold_eval": require_gold_eval,
        "n_eval_requested": int(n_eval),
    }
    return sub_exs, eval_exs, meta


# ============================================================
# Helpers for LogiQA2.0 / JSON-in-text style datasets
# ============================================================

def _maybe_parse_json_blob_example(ex: dict) -> dict:
    if not isinstance(ex, dict):
        return ex
    if "text" not in ex or not isinstance(ex["text"], str):
        return ex
    s = ex["text"].strip()
    if len(s) < 2 or s[0] != "{" or s[-1] != "}":
        return ex
    if not any(k in s for k in ["\"question\"", "\"options\"", "\"answer\"", "\"correct\""]):
        return ex
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return ex


def _extract_text_field_for_lm(ex: dict) -> str:
    if not isinstance(ex, dict):
        return ""
    for k in ["text", "sentence", "content", "data"]:
        v = ex.get(k, None)
        if isinstance(v, str) and v.strip():
            return v
    for _, v in ex.items():
        if isinstance(v, str) and v.strip():
            return v
    return ""


# ============================================================
# Text cleaning helpers
# ============================================================

def _first_nonempty_line(text: str) -> str:
    if text is None:
        return ""
    for ln in str(text).splitlines():
        ln = ln.strip()
        if ln:
            return ln
    return str(text).strip()


def _strip_repeated_choice_prefix(opt: Any, letters: str) -> str:
    s = "" if opt is None else str(opt).strip()
    if not s:
        return ""
    pat = re.compile(rf"^\s*([{letters}{letters.lower()}])\s*[\)\.\:\-]\s*")
    prev = None
    while prev != s:
        prev = s
        s = pat.sub("", s).strip()
    s = s.strip().strip(",").strip()
    return s


# ============================================================
# Individual task loaders
# ============================================================

def load_gsm8k(
    *,
    n_subspace: int,
    n_eval: int,
    seed: int,
    template_randomization: bool,
    template_seed: int,
    add_answer_prefix: bool,
    answer_prefix: str,
) -> Tuple[List[Example], List[Example], Dict[str, Any]]:
    ds, used_rev = load_hf_dataset("gsm8k", "main")

    def parse_gsm8k_gold(answer_field: str) -> str:
        if answer_field is None:
            return ""
        txt = str(answer_field).replace(",", "")
        m = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", txt)
        if m:
            return m.group(1).strip()
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", txt)
        return nums[-1].strip() if nums else ""

    def build_one(ex: dict, ex_id: str) -> Tuple[str, str]:
        q = str(ex.get("question", "") or "")
        tid = choose_template_id(ex_id, len(GSM8K_TEMPLATES), template_seed) if template_randomization else 0
        p = build_prompt_gsm8k(q, tid)
        p = maybe_add_answer_prefix(p, add_answer_prefix, answer_prefix)
        g = parse_gsm8k_gold(ex.get("answer", ""))
        return p, g

    sub_exs, eval_exs, meta = _build_from_splits_with_fallback(
        ds, "gsm8k", build_one,
        n_subspace=n_subspace, n_eval=n_eval, seed=seed,
        sub_candidates=["train", "test"],
        eval_candidates=["test", "train"],
        require_gold_eval=True,
    )
    meta["hf_id"] = "gsm8k/main"
    if used_rev:
        meta["hf_revision"] = used_rev
    return sub_exs, eval_exs, meta


def load_commonsenseqa(
    *,
    n_subspace: int,
    n_eval: int,
    seed: int,
    template_randomization: bool,
    template_seed: int,
    shuffle_choices: bool,
    add_answer_prefix: bool,
    answer_prefix: str,
) -> Tuple[List[Example], List[Example], Dict[str, Any]]:
    ds, used_rev = load_hf_dataset("commonsense_qa")

    def build_one(ex: dict, ex_id: str) -> Tuple[str, str]:
        q = str(ex.get("question", "") or "")
        choices = ex.get("choices", {})
        labels = [safe_upper(x) for x in choices.get("label", [])]
        texts = [str(x) for x in choices.get("text", [])]
        if len(texts) == 0:
            return "", ""
        if len(labels) != len(texts) or any(l not in MC_LETTERS_8 for l in labels):
            labels = MC_LETTERS_5[:len(texts)]
        gold = safe_upper(ex.get("answerKey", ""))

        texts2, labels2, gold2 = shuffle_choices_if_needed(texts, labels, gold, shuffle_choices, seed, ex_id)
        tid = choose_template_id(ex_id, len(MC_TEMPLATES), template_seed) if template_randomization else 0
        p = build_prompt_mc(q, texts2, labels2, tid)
        p = maybe_add_answer_prefix(p, add_answer_prefix, answer_prefix)
        return p, gold2

    sub_exs, eval_exs, meta = _build_from_splits_with_fallback(
        ds, "commonsenseqa", build_one,
        n_subspace=n_subspace, n_eval=n_eval, seed=seed,
        sub_candidates=["train", "validation", "test"],
        eval_candidates=["validation", "test", "train"],
        require_gold_eval=True,
    )
    meta["hf_id"] = "commonsense_qa"
    if used_rev:
        meta["hf_revision"] = used_rev
    return sub_exs, eval_exs, meta


def load_strategyqa(
    *,
    n_subspace: int,
    n_eval: int,
    seed: int,
    template_randomization: bool,
    template_seed: int,
    add_answer_prefix: bool,
    answer_prefix: str,
) -> Tuple[List[Example], List[Example], Dict[str, Any]]:
    hf_id = "ChilleD/StrategyQA"
    ds, used_rev = load_hf_dataset(hf_id)

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

    def build_one(ex: dict, ex_id: str) -> Tuple[str, str]:
        q = str(ex.get("question", "") or "")
        tid = choose_template_id(ex_id, len(YN_TEMPLATES), template_seed) if template_randomization else 0
        p = build_prompt_yesno(q, tid)
        p = maybe_add_answer_prefix(p, add_answer_prefix, answer_prefix)
        g = to_yesno(ex.get("answer", None))
        return p, g

    sub_exs, eval_exs, meta = _build_from_splits_with_fallback(
        ds, "strategyqa", build_one,
        n_subspace=n_subspace, n_eval=n_eval, seed=seed,
        sub_candidates=["train", "validation", "test"],
        eval_candidates=["test", "validation", "train"],
        require_gold_eval=True,
    )
    meta["hf_id"] = hf_id
    if used_rev:
        meta["hf_revision"] = used_rev
    return sub_exs, eval_exs, meta


def load_aqua(
    *,
    n_subspace: int,
    n_eval: int,
    seed: int,
    template_randomization: bool,
    template_seed: int,
    shuffle_choices: bool,
    add_answer_prefix: bool,
    answer_prefix: str,
) -> Tuple[List[Example], List[Example], Dict[str, Any]]:
    """
    AQuA-RAT (aqua_rat) multiple-choice (A-E).

    Important gotcha:
      - Many 'options' strings already include a leading "A)"/"B)"/... prefix.
        If you shuffle choices but don't strip these prefixes, you leak the ORIGINAL
        label into the option text, causing systematic label mismatch and very low accuracy.
    """
    ds, used_rev = load_hf_dataset("aqua_rat")

    def normalize_gold(v: Any) -> str:
        s = safe_upper(v)
        if not s:
            return ""
        m = re.search(r"[A-E]", s)
        if m:
            return m.group(0)
        if s.isdigit():
            j = int(s)
            if 0 <= j < 5:
                return MC_LETTERS_5[j]
            if 1 <= j <= 5:
                return MC_LETTERS_5[j - 1]
        return s

    def get_gold(ex: dict) -> str:
        if "correct" in ex:
            return normalize_gold(ex["correct"])
        if "answer" in ex:
            return normalize_gold(ex["answer"])
        return ""

    def build_one(ex: dict, ex_id: str) -> Tuple[str, str]:
        q = str(ex.get("question", "") or "")

        opts = ex.get("options", [])
        if isinstance(opts, str):
            import json as _json
            s = opts.strip()
            if s.startswith("["):
                try:
                    opts = _json.loads(s)
                except Exception:
                    pass
            if isinstance(opts, str):
                marks = list(re.finditer(r"\b[A-E]\)", s))
                parts = []
                for i, m in enumerate(marks):
                    a = m.end()
                    b = marks[i + 1].start() if i + 1 < len(marks) else len(s)
                    parts.append(s[a:b].strip())
                opts = [p for p in parts if p]

        if not isinstance(opts, list) or len(opts) == 0:
            return "", ""

        texts: List[str] = []
        for opt in opts[:5]:
            opt_clean = _strip_repeated_choice_prefix(opt, letters="ABCDE")
            if opt_clean:
                texts.append(opt_clean)

        if len(texts) == 0:
            return "", ""

        labels = MC_LETTERS_5[:len(texts)]
        gold = get_gold(ex)

        texts2, labels2, gold2 = shuffle_choices_if_needed(texts, labels, gold, shuffle_choices, seed, ex_id)
        tid = choose_template_id(ex_id, len(MC_TEMPLATES), template_seed) if template_randomization else 0
        p = build_prompt_mc(q, texts2, labels2, tid)
        # force for AQuA (improves extraction)
        p = maybe_add_answer_prefix(p, True, answer_prefix)
        return p, gold2

    sub_exs, eval_exs, meta = _build_from_splits_with_fallback(
        ds, "aqua", build_one,
        n_subspace=n_subspace, n_eval=n_eval, seed=seed,
        sub_candidates=["train", "validation", "test"],
        eval_candidates=["test", "validation", "train"],
        require_gold_eval=True,
    )
    meta["hf_id"] = "aqua_rat"
    if used_rev:
        meta["hf_revision"] = used_rev
    meta["options_prefix_stripped"] = True
    meta["force_answer_prefix"] = True
    return sub_exs, eval_exs, meta


def load_arc_challenge(
    *,
    n_subspace: int,
    n_eval: int,
    seed: int,
    template_randomization: bool,
    template_seed: int,
    shuffle_choices: bool,
    add_answer_prefix: bool,
    answer_prefix: str,
) -> Tuple[List[Example], List[Example], Dict[str, Any]]:
    cfg = "ARC-Challenge"
    try:
        ds, used_rev = load_hf_dataset("ai2_arc", cfg)
    except Exception:
        cfg = None
        ds, used_rev = load_hf_dataset("ai2_arc")

    def build_one(ex: dict, ex_id: str) -> Tuple[str, str]:
        q = _extract_question_text(ex)
        texts, labels = _extract_choices(ex)
        gold0 = _extract_answer_key(ex)
        texts, canon_labels, gold = _canonicalize_mc(texts, labels, gold0)
        if not texts:
            return "", ""
        texts2, labels2, gold2 = shuffle_choices_if_needed(texts, canon_labels, gold, shuffle_choices, seed, ex_id)
        tid = choose_template_id(ex_id, len(MC_TEMPLATES), template_seed) if template_randomization else 0
        p = build_prompt_mc(q, texts2, labels2, tid)
        p = maybe_add_answer_prefix(p, add_answer_prefix, answer_prefix)
        return p, gold2

    sub_exs, eval_exs, meta = _build_from_splits_with_fallback(
        ds, "arc_challenge", build_one,
        n_subspace=n_subspace, n_eval=n_eval, seed=seed,
        sub_candidates=["train", "validation", "test"],
        eval_candidates=["test", "validation", "train"],
        require_gold_eval=True,
    )
    meta["hf_id"] = "ai2_arc" + (f"/{cfg}" if cfg else "")
    if used_rev:
        meta["hf_revision"] = used_rev
    return sub_exs, eval_exs, meta


def load_openbookqa(
    *,
    n_subspace: int,
    n_eval: int,
    seed: int,
    template_randomization: bool,
    template_seed: int,
    shuffle_choices: bool,
    add_answer_prefix: bool,
    answer_prefix: str,
) -> Tuple[List[Example], List[Example], Dict[str, Any]]:
    cfg = "main"
    try:
        ds, used_rev = load_hf_dataset("openbookqa", cfg)
    except Exception:
        cfg = None
        ds, used_rev = load_hf_dataset("openbookqa")

    def build_one(ex: dict, ex_id: str) -> Tuple[str, str]:
        q = _extract_question_text(ex)
        texts, labels = _extract_choices(ex)
        gold0 = _extract_answer_key(ex)
        texts, canon_labels, gold = _canonicalize_mc(texts, labels, gold0)
        if not texts:
            return "", ""
        texts2, labels2, gold2 = shuffle_choices_if_needed(texts, canon_labels, gold, shuffle_choices, seed, ex_id)
        tid = choose_template_id(ex_id, len(MC_TEMPLATES), template_seed) if template_randomization else 0
        p = build_prompt_mc(q, texts2, labels2, tid)
        p = maybe_add_answer_prefix(p, add_answer_prefix, answer_prefix)
        return p, gold2

    sub_exs, eval_exs, meta = _build_from_splits_with_fallback(
        ds, "openbookqa", build_one,
        n_subspace=n_subspace, n_eval=n_eval, seed=seed,
        sub_candidates=["train", "validation", "test"],
        eval_candidates=["test", "validation", "train"],
        require_gold_eval=True,
    )
    meta["hf_id"] = "openbookqa" + (f"/{cfg}" if cfg else "")
    if used_rev:
        meta["hf_revision"] = used_rev
    return sub_exs, eval_exs, meta


def load_qasc(
    *,
    n_subspace: int,
    n_eval: int,
    seed: int,
    template_randomization: bool,
    template_seed: int,
    shuffle_choices: bool,
    add_answer_prefix: bool,
    answer_prefix: str,
) -> Tuple[List[Example], List[Example], Dict[str, Any]]:
    ds, used_rev = load_hf_dataset("qasc")

    def build_one(ex: dict, ex_id: str) -> Tuple[str, str]:
        q = _extract_question_text(ex)
        texts, labels = _extract_choices(ex)
        gold0 = _extract_answer_key(ex)
        texts, canon_labels, gold = _canonicalize_mc(texts, labels, gold0)
        if not texts:
            return "", ""
        texts2, labels2, gold2 = shuffle_choices_if_needed(texts, canon_labels, gold, shuffle_choices, seed, ex_id)
        tid = choose_template_id(ex_id, len(MC_TEMPLATES), template_seed) if template_randomization else 0
        p = build_prompt_mc(q, texts2, labels2, tid)
        p = maybe_add_answer_prefix(p, add_answer_prefix, answer_prefix)
        return p, gold2

    sub_exs, eval_exs, meta = _build_from_splits_with_fallback(
        ds, "qasc", build_one,
        n_subspace=n_subspace, n_eval=n_eval, seed=seed,
        sub_candidates=["train", "validation", "dev", "test"],
        eval_candidates=["validation", "dev", "train", "test"],
        require_gold_eval=True,
    )
    meta["hf_id"] = "qasc"
    if used_rev:
        meta["hf_revision"] = used_rev
    return sub_exs, eval_exs, meta


def load_logiqa(
    *,
    n_subspace: int,
    n_eval: int,
    seed: int,
    template_randomization: bool,
    template_seed: int,
    shuffle_choices: bool,
    add_answer_prefix: bool,
    answer_prefix: str,
) -> Tuple[List[Example], List[Example], Dict[str, Any]]:
    candidates: List[Tuple[str, Optional[str], Optional[str]]] = [
        ("lucasmccabe/logiqa", None, AUTO_PARQUET_REVISION),
        ("EleutherAI/logiqa", "logiqa", AUTO_PARQUET_REVISION),
        ("lucasmccabe/logiqa", None, None),
        ("EleutherAI/logiqa", "logiqa", None),
        ("datatune/LogiQA2.0", None, None),
    ]

    errors: List[str] = []

    def build_one(raw_ex: dict, ex_id: str) -> Tuple[str, str]:
        ex = _maybe_parse_json_blob_example(raw_ex)

        q = _extract_question_text(ex).strip()

        ctx = ""
        for k in ["context", "passage", "article", "text"]:
            if k in ex and isinstance(ex[k], str) and ex[k].strip():
                ctx = ex[k].strip()
                break

        if ctx and q:
            q_full = f"{ctx}\n\nQuestion: {q}".strip() if ctx not in q else q
        elif ctx:
            q_full = ctx.strip()
        else:
            q_full = q.strip()

        texts, labels = _extract_choices(ex)
        if not texts:
            for k in ["options", "answers", "candidates", "endings"]:
                v = ex.get(k, None)
                if isinstance(v, list) and len(v) >= 2:
                    texts = [str(x) for x in v]
                    labels = MC_LETTERS_8[:len(texts)]
                    break
            if not texts:
                return "", ""

        gold0 = _extract_answer_key(ex)
        texts, canon_labels, gold = _canonicalize_mc(texts, labels, gold0)

        if shuffle_choices and gold:
            texts2, labels2, gold2 = shuffle_choices_if_needed(texts, canon_labels, gold, True, seed, ex_id)
        else:
            texts2, labels2, gold2 = texts, canon_labels, gold

        tid = choose_template_id(ex_id, len(MC_TEMPLATES), template_seed) if template_randomization else 0
        p = build_prompt_mc(q_full, texts2, labels2, tid)
        p = maybe_add_answer_prefix(p, add_answer_prefix, answer_prefix)
        return p, gold2

    for hf_id, cfg, rev in candidates:
        try:
            ds, used_rev = load_hf_dataset(hf_id, cfg, revision=rev)
        except Exception as e:
            errors.append(f"[logiqa] load failed: {hf_id}" + (f"/{cfg}" if cfg else "") + (f" (rev={rev})" if rev else "") + f" -> {repr(e)}")
            continue

        try:
            sub_exs, eval_exs, meta = _build_from_splits_with_fallback(
                ds, "logiqa", build_one,
                n_subspace=n_subspace, n_eval=n_eval, seed=seed,
                sub_candidates=["train", "validation", "dev", "test"],
                eval_candidates=["validation", "dev", "train", "test"],
                require_gold_eval=True,
            )
            meta["hf_id"] = hf_id + (f"/{cfg}" if cfg else "")
            if used_rev:
                meta["hf_revision"] = used_rev
            return sub_exs, eval_exs, meta
        except Exception as e:
            errors.append(f"[logiqa] build failed: {hf_id}" + (f"/{cfg}" if cfg else "") + (f" (rev={rev})" if rev else "") + f" -> {repr(e)}")
            continue

    raise RuntimeError("All LogiQA candidates failed.\n" + "\n".join(errors))


def load_boolq(
    *,
    n_subspace: int,
    n_eval: int,
    seed: int,
    template_randomization: bool,
    template_seed: int,
    shuffle_choices: bool,
    add_answer_prefix: bool,
    answer_prefix: str,
) -> Tuple[List[Example], List[Example], Dict[str, Any]]:
    try:
        ds, used_rev = load_hf_dataset("boolq")
        hf_id = "boolq"
    except Exception:
        ds, used_rev = load_hf_dataset("super_glue", "boolq")
        hf_id = "super_glue/boolq"

    def to_ab_gold(ans: Any) -> str:
        if isinstance(ans, bool):
            return "A" if ans else "B"
        if isinstance(ans, (int, np.integer)):
            return "A" if int(ans) == 1 else "B"
        s = str(ans).strip().lower()
        if s in ["true", "yes", "1"]:
            return "A"
        if s in ["false", "no", "0"]:
            return "B"
        return ""

    def build_one(ex: dict, ex_id: str) -> Tuple[str, str]:
        passage = str(ex.get("passage", ex.get("context", ex.get("text", ""))) or "")
        question = str(ex.get("question", ex.get("query", "")) or "")
        q_full = f"Passage: {passage}\nQuestion: {question}".strip()

        option_texts = ["Yes", "No"]
        option_labels = MC_LETTERS_8[:2]
        gold = to_ab_gold(ex.get("answer", ex.get("label", None)))

        texts2, labels2, gold2 = shuffle_choices_if_needed(
            option_texts, option_labels, gold,
            do_shuffle=shuffle_choices, seed=seed, ex_id=ex_id
        )
        tid = choose_template_id(ex_id, len(MC_TEMPLATES), template_seed) if template_randomization else 0
        p = build_prompt_mc(q_full, texts2, labels2, tid)
        p = maybe_add_answer_prefix(p, add_answer_prefix, answer_prefix)
        return p, gold2

    sub_exs, eval_exs, meta = _build_from_splits_with_fallback(
        ds, "boolq", build_one,
        n_subspace=n_subspace, n_eval=n_eval, seed=seed,
        sub_candidates=["train", "validation", "test"],
        eval_candidates=["validation", "test", "train"],
        require_gold_eval=True,
    )
    meta["hf_id"] = hf_id
    if used_rev:
        meta["hf_revision"] = used_rev
    return sub_exs, eval_exs, meta


def load_piqa(
    *,
    n_subspace: int,
    n_eval: int,
    seed: int,
    template_randomization: bool,
    template_seed: int,
    shuffle_choices: bool,
    add_answer_prefix: bool,
    answer_prefix: str,
) -> Tuple[List[Example], List[Example], Dict[str, Any]]:
    ds, used_rev = load_hf_dataset("piqa")
    hf_id = "piqa"

    def to_ab_gold(label: Any) -> str:
        if isinstance(label, (int, np.integer)):
            return "A" if int(label) == 0 else ("B" if int(label) == 1 else "")
        s = str(label).strip().upper()
        if s == "0":
            return "A"
        if s == "1":
            return "B"
        if s in ["A", "B"]:
            return s
        return ""

    def build_one(ex: dict, ex_id: str) -> Tuple[str, str]:
        goal = str(ex.get("goal", ex.get("question", "")) or "").strip()
        sol1 = str(ex.get("sol1", ex.get("ending0", ex.get("choice0", ""))) or "").strip()
        sol2 = str(ex.get("sol2", ex.get("ending1", ex.get("choice1", ""))) or "").strip()

        option_texts = [sol1, sol2]
        option_labels = MC_LETTERS_8[:2]
        gold = to_ab_gold(ex.get("label", ex.get("answer", None)))

        texts2, labels2, gold2 = shuffle_choices_if_needed(
            option_texts, option_labels, gold,
            do_shuffle=shuffle_choices, seed=seed, ex_id=ex_id
        )

        tid = choose_template_id(ex_id, len(MC_TEMPLATES), template_seed) if template_randomization else 0
        p = build_prompt_mc(goal, texts2, labels2, tid)
        p = maybe_add_answer_prefix(p, add_answer_prefix, answer_prefix)
        return p, gold2

    sub_exs, eval_exs, meta = _build_from_splits_with_fallback(
        ds, "piqa", build_one,
        n_subspace=n_subspace, n_eval=n_eval, seed=seed,
        sub_candidates=["train", "validation", "test"],
        eval_candidates=["validation", "test", "train"],
        require_gold_eval=True,
    )
    meta["hf_id"] = hf_id
    if used_rev:
        meta["hf_revision"] = used_rev
    return sub_exs, eval_exs, meta


def load_wikitext(
    *,
    n_subspace: int,
    n_eval: int,
    seed: int,
    template_randomization: bool,
    template_seed: int,
    add_answer_prefix: bool,
    answer_prefix: str,
    prefix_words: int = 32,
) -> Tuple[List[Example], List[Example], Dict[str, Any]]:
    # 更稳的 id：优先用官方 wikitext（很多环境没有 Salesforce/wikitext）
    cfg = "wikitext-2-raw-v1"
    tried = []
    ds = None
    used_rev = None
    hf_id = None

    for _hf_id in ["wikitext", "Salesforce/wikitext"]:
        try:
            _ds, _used_rev = load_hf_dataset(_hf_id, cfg)
            ds, used_rev, hf_id = _ds, _used_rev, _hf_id
            break
        except Exception as e:
            tried.append(f"{_hf_id}/{cfg} -> {repr(e)}")

    if ds is None:
        raise RuntimeError("Failed to load wikitext. Tried:\n" + "\n".join(tried))

    def build_one(ex: dict, ex_id: str) -> Tuple[str, str]:
        txt = str(ex.get("text", "") or "")
        txt = " ".join(txt.split())
        toks = txt.split()
        if len(toks) < prefix_words + 1:
            return "", ""
        context = " ".join(toks[:prefix_words])
        gold = toks[prefix_words]

        tid = choose_template_id(ex_id, len(LM_NEXTWORD_TEMPLATES), template_seed) if template_randomization else 0
        p = build_prompt_nextword(context, tid)
        p = maybe_add_answer_prefix(p, add_answer_prefix, answer_prefix)
        return p, gold

    sub_exs, eval_exs, meta = _build_from_splits_with_fallback(
        ds, "wikitext", build_one,
        n_subspace=n_subspace, n_eval=n_eval, seed=seed,
        sub_candidates=["train", "validation", "test"],
        eval_candidates=["validation", "test", "train"],
        require_gold_eval=True,
    )
    meta["hf_id"] = f"{hf_id}/{cfg}"
    if used_rev:
        meta["hf_revision"] = used_rev
    meta["task_type"] = "next_word"
    meta["prefix_words"] = int(prefix_words)
    return sub_exs, eval_exs, meta


def load_ptb(
    *,
    n_subspace: int,
    n_eval: int,
    seed: int,
    template_randomization: bool,
    template_seed: int,
    add_answer_prefix: bool,
    answer_prefix: str,
    prefix_words: int = 32,
) -> Tuple[List[Example], List[Example], Dict[str, Any]]:
    hf_id = "ptb-text-only/ptb_text_only"
    ds, used_rev = load_hf_dataset(hf_id)

    def build_one(ex: dict, ex_id: str) -> Tuple[str, str]:
        txt = _extract_text_field_for_lm(ex)
        txt = " ".join(str(txt).split())
        toks = txt.split()
        if len(toks) < prefix_words + 1:
            return "", ""
        context = " ".join(toks[:prefix_words])
        gold = toks[prefix_words]

        tid = choose_template_id(ex_id, len(LM_NEXTWORD_TEMPLATES), template_seed) if template_randomization else 0
        p = build_prompt_nextword(context, tid)
        p = maybe_add_answer_prefix(p, add_answer_prefix, answer_prefix)
        return p, gold

    sub_exs, eval_exs, meta = _build_from_splits_with_fallback(
        ds, "ptb", build_one,
        n_subspace=n_subspace, n_eval=n_eval, seed=seed,
        sub_candidates=["train", "validation", "test"],
        eval_candidates=["validation", "test", "train"],
        require_gold_eval=True,
    )
    meta["hf_id"] = hf_id
    if used_rev:
        meta["hf_revision"] = used_rev
    meta["task_type"] = "next_word"
    meta["prefix_words"] = int(prefix_words)
    return sub_exs, eval_exs, meta


# ============================================================
# Task registry
# ============================================================

TASK_LOADERS = {
    "gsm8k": load_gsm8k,
    "commonsenseqa": load_commonsenseqa,
    "strategyqa": load_strategyqa,
    "aqua": load_aqua,
    "arc_challenge": load_arc_challenge,
    "openbookqa": load_openbookqa,
    "qasc": load_qasc,
    "logiqa": load_logiqa,
    "boolq": load_boolq,
    "piqa": load_piqa,
    "wikitext": load_wikitext,
    "ptb": load_ptb,
}

_NO_SHUFFLE_ARG = {"gsm8k", "strategyqa", "wikitext", "ptb"}


def load_selected_tasks(
    *,
    tasks: List[str],
    n_subspace: int,
    n_eval: int,
    seed: int,
    template_randomization: bool = True,
    template_seed: int = 1234,
    shuffle_choices: bool = False,
    add_answer_prefix: bool = False,
    answer_prefix: str = "\nFinal answer:",
) -> Tuple[Dict[str, List[Example]], Dict[str, List[Example]], Dict[str, Any]]:
    """
    Returns:
      sub_by[task]  -> list[Example] for subspace collection (gold may be "")
      eval_by[task] -> list[Example] for evaluation (gold required)
      meta_by[task] -> metadata (hf_id, chosen splits, etc.)
    """
    set_global_seed(seed)

    sub_by: Dict[str, List[Example]] = {}
    eval_by: Dict[str, List[Example]] = {}
    meta_by: Dict[str, Any] = {}

    for name in tasks:
        if name not in TASK_LOADERS:
            raise ValueError(f"Unknown task '{name}'. Known: {sorted(TASK_LOADERS.keys())}")

        fn = TASK_LOADERS[name]

        if name in _NO_SHUFFLE_ARG:
            sub_exs, eval_exs, meta = fn(
                n_subspace=n_subspace,
                n_eval=n_eval,
                seed=seed,
                template_randomization=template_randomization,
                template_seed=template_seed,
                add_answer_prefix=add_answer_prefix,
                answer_prefix=answer_prefix,
            )
        else:
            sub_exs, eval_exs, meta = fn(
                n_subspace=n_subspace,
                n_eval=n_eval,
                seed=seed,
                template_randomization=template_randomization,
                template_seed=template_seed,
                shuffle_choices=shuffle_choices,
                add_answer_prefix=add_answer_prefix,
                answer_prefix=answer_prefix,
            )

        sub_by[name] = sub_exs
        eval_by[name] = eval_exs
        meta_by[name] = meta

    return sub_by, eval_by, meta_by


# ============================================================
# Optional: evaluation helpers (reuse across scripts)
# ============================================================

_FINAL_RE = re.compile(r"final\s*answer\s*:\s*(.*)", re.IGNORECASE)


def _extract_final_span(text: str) -> str:
    if text is None:
        return ""
    t = str(text)
    ms = list(_FINAL_RE.finditer(t))
    if ms:
        return ms[-1].group(1).strip()
    lines = [ln.strip() for ln in t.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _normalize_number_str(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().replace(",", "")
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", s)
    if not nums:
        return ""
    num = nums[-1]
    if "." in num:
        num = num.rstrip("0").rstrip(".")
    return num


def _digits_to_letter(n: int, letters: str) -> str:
    L = len(letters)
    if 1 <= n <= L:
        return letters[n - 1]
    if 0 <= n < L:
        return letters[n]
    return ""


def _normalize_gold_mc(gold: Any, letters: str) -> str:
    if gold is None:
        return ""
    if isinstance(gold, (int, np.integer)):
        return _digits_to_letter(int(gold), letters)
    s = str(gold).strip().upper()
    s2 = re.sub(r"^[\(\[\{<\s]+|[\)\]\}>\s\.\:;,\!]+$", "", s)
    if s2 in set(letters):
        return s2
    if s2.isdigit():
        return _digits_to_letter(int(s2), letters)
    s3 = re.sub(r"[^A-Z0-9]+", "", s)
    if len(s3) == 1 and s3 in set(letters):
        return s3
    if s3.isdigit():
        return _digits_to_letter(int(s3), letters)
    return ""


def _normalize_pred_mc(pred_span: str, letters: str) -> str:
    if pred_span is None:
        return ""
    s = str(pred_span).strip().upper()

    m = re.search(rf"\b([{letters}])\b", s)
    if m:
        return m.group(1)

    m = re.search(rf"([{letters}])\s*[\)\]\.\:\,;]", s)
    if m:
        return m.group(1)

    m = re.search(rf"(OPTION|CHOICE|ANSWER|ANS|选项|答案)\s*[:：]?\s*([{letters}])", s)
    if m:
        return m.group(2)

    m = re.search(rf"([{letters}])", s[:32])
    if m:
        return m.group(1)

    md = re.search(r"\b(\d{1,2})\b", s)
    if md:
        return _digits_to_letter(int(md.group(1)), letters)

    s2 = re.sub(r"[^A-Z0-9]+", "", s)
    if s2 and s2[0] in set(letters):
        return s2[0]
    if s2.isdigit():
        return _digits_to_letter(int(s2), letters)

    return ""


def _normalize_yesno_pred(span: str) -> str:
    if span is None:
        return ""
    s = str(span).strip().lower()
    if re.search(r"\byes\b", s) or "true" in s:
        return "YES"
    if re.search(r"\bno\b", s) or "false" in s:
        return "NO"
    if "entail" in s and "not" not in s:
        return "YES"
    if "unknown" in s or "not entail" in s or "not_entail" in s or "cannot be entailed" in s:
        return "NO"
    m = re.search(r"\b([AB])\b", s.upper())
    if m:
        return "YES" if m.group(1) == "A" else "NO"
    return ""


def _normalize_yesno_gold(gold: Any) -> str:
    if gold is None:
        return ""
    if isinstance(gold, bool):
        return "YES" if gold else "NO"
    if isinstance(gold, (int, np.integer)):
        return "YES" if int(gold) == 1 else "NO"
    s = str(gold).strip().lower()
    if s in {"yes", "true", "1"}:
        return "YES"
    if s in {"no", "false", "0"}:
        return "NO"
    if "entail" in s and "not" not in s:
        return "YES"
    if "unknown" in s or "not entail" in s or "not_entail" in s:
        return "NO"
    if s in {"a"}:
        return "YES"
    if s in {"b"}:
        return "NO"
    return ""


def _normalize_freeform(s: str) -> str:
    if s is None:
        return ""
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def _normalize_nextword(s: Any) -> str:
    if s is None:
        return ""
    t = str(s).strip()
    if not t:
        return ""
    first = t.split()[0]
    first = first.strip().strip("“”\"'`.,;:!?()[]{}")
    return first.lower()


_MC_LABELS = {
    "aqua": "ABCDE",
    "arc_challenge": "ABCD",
    "openbookqa": "ABCD",
    "qasc": "ABCDEFGH",
    "logiqa": "ABCD",
    "boolq": "AB",
    "commonsenseqa": "ABCDE",
    "piqa": "AB",
}

_FREEFORM_QA = set()
_RULE_TASKS = {"rule_taker", "ruletaker"}
_NEXTWORD_TASKS = {"wikitext", "ptb"}


def parse_prediction(task: str, generated_text: str) -> str:
    """
    Parse model continuation into a canonical prediction string.

    NOTE: When prompts end with an "answer prefix" (e.g. '\\nFinal answer:'),
    many LMs output the answer at the *beginning* of the continuation and then
    continue explaining. This parser therefore falls back to looking at the
    first line / early window if parsing from the final line fails.
    """
    t = (task or "").strip().lower()
    span = _extract_final_span(generated_text)

    if t == "gsm8k":
        p = _normalize_number_str(span)
        if p:
            return p
        return _normalize_number_str(_first_nonempty_line(generated_text))

    if t in _NEXTWORD_TASKS:
        return span.strip()

    if t in _FREEFORM_QA:
        return span.strip()

    if t in _RULE_TASKS:
        p = _normalize_yesno_pred(span)
        if p:
            return p
        return _normalize_yesno_pred(_first_nonempty_line(generated_text))

    if t == "strategyqa":
        p = _normalize_yesno_pred(span)
        if p:
            return p
        return _normalize_yesno_pred(_first_nonempty_line(generated_text))

    if t in _MC_LABELS:
        letters = _MC_LABELS[t]
        pred = _normalize_pred_mc(span, letters)
        if pred:
            return pred

        first = _first_nonempty_line(generated_text)
        pred = _normalize_pred_mc(first, letters)
        if pred:
            return pred

        pred = _normalize_pred_mc(str(generated_text)[:128], letters)
        if pred:
            return pred

        return _normalize_pred_mc(generated_text, letters)

    return span.strip()


def is_correct(task: str, pred: Any, gold: Any) -> bool:
    t = (task or "").strip().lower()

    if t == "gsm8k":
        return _normalize_number_str(pred) == _normalize_number_str(gold) and _normalize_number_str(pred) != ""

    if t in _NEXTWORD_TASKS:
        p = _normalize_nextword(pred)
        g = _normalize_nextword(gold)
        return bool(p) and bool(g) and (p == g)

    if t in _FREEFORM_QA:
        p = _normalize_freeform(pred)
        g = _normalize_freeform(gold)
        return bool(p) and bool(g) and (p == g or (g in p))

    if t in _RULE_TASKS or t == "strategyqa":
        p = _normalize_yesno_pred(pred)
        g = _normalize_yesno_gold(gold)
        return (p != "") and (g != "") and (p == g)

    if t in _MC_LABELS:
        letters = _MC_LABELS[t]
        p = _normalize_pred_mc(pred, letters)
        g = _normalize_gold_mc(gold, letters)
        return (p != "") and (g != "") and (p == g)

    return False


# ============================================================
# Minimal CLI smoke-test (optional)
# ============================================================

if __name__ == "__main__":
    tasks = ["aqua"]
    sub_by, eval_by, meta_by = load_selected_tasks(
        tasks=tasks,
        n_subspace=16,
        n_eval=32,
        seed=42,
        template_randomization=True,
        template_seed=1234,
        shuffle_choices=True,
        add_answer_prefix=True,
        answer_prefix="\nFinal answer:",
    )
    print(json.dumps(meta_by, indent=2, ensure_ascii=False))
    for t in tasks:
        print(f"{t}: subspace={len(sub_by[t])}, eval={len(eval_by[t])}")
