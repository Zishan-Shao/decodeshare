# -*- coding: utf-8 -*-
"""
benchmark_dataloaders.py

Robust dataset loading + prompt construction for common HF benchmarks.

Key features:
  - No trust_remote_code.
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
    return "Dataset scripts are no longer supported" in msg


def _is_features_dataclass_error(e: Exception) -> bool:
    """Heuristic for a known HF cache / feature-deserialization failure."""
    msg = str(e)
    return "must be called with a dataclass type or instance" in msg
def _is_arrow_cast_error(e: Exception) -> bool:
    """
    Heuristic for pyarrow/datasets casting failures that can happen with stale caches
    or feature schema mismatches across dataset versions.
    """
    msg = str(e)
    return ("CastError" in msg) or ("Couldn't cast" in msg) or ("because column names don't match" in msg)



def load_hf_dataset(
    hf_id: str,
    cfg: Optional[str] = None,
    *,
    revision: Optional[str] = None,
    **kwargs,
) -> Tuple[Any, Optional[str]]:
    """
    Load a HF dataset with robust fallbacks.

    Returns:
      (ds, used_revision)

    Fallback policy:
      1) Try (hf_id, cfg, revision) first.
      2) If revision is None and load fails for *any* reason, try
         revision="refs/convert/parquet" (many dataset repos provide this).
      3) If we hit a HF cache / DatasetInfo parsing error (often seen as
         TypeError: "must be called with a dataclass type or instance"),
         retry once with download_mode="force_redownload".

    Notes:
      - We keep trust_remote_code=False (default) for safety.
      - If parquet fallback doesn't exist, the original exception is re-raised.
    """

    def _call(rev: Optional[str], extra_kwargs: Optional[dict] = None):
        kw = dict(kwargs)
        if extra_kwargs:
            kw.update(extra_kwargs)

        if cfg is None:
            if rev is None:
                return load_dataset(hf_id, **kw)
            return load_dataset(hf_id, revision=rev, **kw)
        else:
            if rev is None:
                return load_dataset(hf_id, cfg, **kw)
            return load_dataset(hf_id, cfg, revision=rev, **kw)

    try:
        ds = _call(revision)
        return ds, revision
    except Exception as e0:
        # Broad compatibility fallback: parquet branch
        if revision is None:
            try:
                ds = _call(AUTO_PARQUET_REVISION)
                return ds, AUTO_PARQUET_REVISION
            except Exception as e1:
                # Rare but annoying: cache deserialization issues across env versions.
                if (_is_features_dataclass_error(e0) or _is_features_dataclass_error(e1) or _is_arrow_cast_error(e0) or _is_arrow_cast_error(e1)):
                    # Try parquet + force redownload
                    try:
                        ds = _call(
                            AUTO_PARQUET_REVISION,
                            extra_kwargs={"download_mode": "force_redownload"},
                        )
                        return ds, AUTO_PARQUET_REVISION
                    except Exception:
                        pass
                    # Try original (no revision) + force redownload
                    try:
                        ds = _call(
                            None,
                            extra_kwargs={"download_mode": "force_redownload"},
                        )
                        return ds, None
                    except Exception:
                        pass

        # If user pinned a revision, only force-redownload on the specific error signature.
        if _is_features_dataclass_error(e0):
            try:
                ds = _call(revision, extra_kwargs={"download_mode": "force_redownload"})
                return ds, revision
            except Exception:
                pass

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

def _maybe_parse_json_blob_example(ex: dict) -> dict:
    """
    有些数据集（或 parquet 转换版本）会把一整条样本 JSON 塞进 ex["text"] 里。
    这个函数尝试把它解析出来并 merge 回原 dict。
    """
    if not isinstance(ex, dict):
        return ex

    # 如果已经是正常结构，就直接返回
    if any(k in ex for k in ("prompt", "question", "choices", "answer", "answerKey")):
        return ex

    for k in ("text", "json", "data"):
        v = ex.get(k)
        if not isinstance(v, str):
            continue
        s = v.strip()
        if not s:
            continue
        looks_like_json = (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]"))
        if not looks_like_json:
            continue

        obj = None
        try:
            obj = json.loads(s)
        except Exception:
            try:
                obj = ast.literal_eval(s)
            except Exception:
                obj = None

        if isinstance(obj, dict):
            merged = dict(ex)
            merged.update(obj)
            return merged

    return ex


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
    build_one,
    *,
    n_subspace: int,
    n_eval: int,
    seed: int,
    sub_candidates: List[str],
    eval_candidates: List[str],
    require_gold_eval: bool = True,
) -> Tuple[List[Example], List[Example], Dict]:
    """
    Build (subspace_examples, eval_examples, meta) from a HF DatasetDict.

    Key behavior:
      - If n_eval <= 0, we **do not** try to build eval examples at all.
        This is important for calibration-mix generation, where we only need prompts.
      - For eval, we try candidate splits in order and (optionally) require gold labels.
    """

    def make_examples(split: str, n: int, require_gold: bool) -> List[Example]:
        if n <= 0:
            return []
        if split not in ds:
            return []
        ds_split = ds[split]
        # Oversample a bit to survive filtering (missing prompts / missing gold)
        probe_n = min(len(ds_split), max(n * 4, n))
        rows = sample_hf_split(
            ds_split,
            n=probe_n,
            seed=stable_int_seed(f"{dataset_name}:{split}", seed),
        )
        exs: List[Example] = []
        for i, ex in enumerate(rows):
            ex_id = str(ex.get("id", ex.get("idx", i)))
            prompt, gold = build_one(ex, ex_id)
            if not prompt:
                continue
            if require_gold and not gold:
                continue
            #exs.append(Example(id=ex_id, prompt=prompt, gold=gold, meta={"split": split}))
            exs.append(Example(dataset=dataset_name, ex_id=ex_id, prompt=prompt, gold=gold))
            if len(exs) >= n:
                break
        return exs

    # ---- 1) subspace prompts (no gold required) ----
    sub_split = None
    sub_exs: List[Example] = []
    for split in sub_candidates:
        if split not in ds:
            continue
        sub_exs = make_examples(split, n_subspace, require_gold=False)
        if len(sub_exs) > 0:
            sub_split = split
            break

    if sub_split is None:
        raise RuntimeError(
            f"[{dataset_name}] Could not build ANY subspace prompts from available splits={list(ds.keys())}. "
            f"Check schema extraction for this dataset."
        )

    # ---- 2) eval examples (optionally require gold); skip entirely if n_eval <= 0 ----
    eval_split = None
    eval_exs: List[Example] = []
    if n_eval > 0:
        for split in eval_candidates:
            if split not in ds:
                continue
            eval_exs = make_examples(split, n_eval, require_gold=require_gold_eval)
            if len(eval_exs) > 0:
                eval_split = split
                break
        if eval_split is None:
            raise RuntimeError(
                f"[{dataset_name}] Could not build ANY eval examples (require_gold={require_gold_eval}) "
                f"from candidate splits={eval_candidates}. Available splits={list(ds.keys())}. "
                f"Likely the 'test' split is unlabeled; prefer validation/dev."
            )

    meta = {
        "hf_dataset": dataset_name,
        "available_splits": list(ds.keys()),
        "subspace_split": sub_split,
        "eval_split": eval_split,
        "n_subspace": len(sub_exs),
        "n_eval": len(eval_exs),
    }
    return sub_exs, eval_exs, meta



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
    n_subspace: int = 512,
    n_eval: int = 512,
    seed: int = 0,
    template_randomization: bool = True,
    template_seed: int = 1234,
    shuffle_choices: bool = False,
    add_answer_prefix: bool = False,
    answer_prefix: str = "\nFinal answer:",
) -> Tuple[List[Example], List[Example], Dict]:
    """
    CommonsenseQA (multiple choice).
    Returns prompts that end with the "Answer:"-style slot (option label expected as gold).
    """
    ds, used_rev = load_hf_dataset("commonsense_qa")
    hf_id = "commonsense_qa"

    def build_one(ex: Dict, ex_id: str) -> Tuple[str, str]:
        q = _extract_question_text(ex).strip()
        texts, labels = _extract_choices(ex)
        gold0 = _extract_answer_key(ex).strip()
        if not q or not texts:
            return "", ""

        # Optionally shuffle choices (shuffle raw labels+texts together so gold0 still matches)
        if shuffle_choices and len(texts) > 1:
            rng = random.Random(stable_int_seed(f"commonsenseqa:{ex_id}", seed))
            order = list(range(len(texts)))
            rng.shuffle(order)
            texts = [texts[i] for i in order]
            labels = [labels[i] for i in order]

        texts2, canon_labels, gold = _canonicalize_mc(texts, labels, gold0)

        tid = choose_template_id(ex_id, len(MC_TEMPLATES), template_seed) if template_randomization else 0
        prompt = build_prompt_mc(q, option_texts=texts2, option_labels=canon_labels, template_id=tid)

        if add_answer_prefix:
            prompt = prompt + answer_prefix

        return prompt, gold

    sub_exs, eval_exs, meta = _build_from_splits_with_fallback(
        ds,
        "commonsenseqa",
        build_one,
        n_subspace=n_subspace,
        n_eval=n_eval,
        seed=seed,
        sub_candidates=["train", "validation", "test"],
        eval_candidates=["validation", "dev", "test", "train"],
        require_gold_eval=True,
    )
    meta.update({"hf_id": hf_id, "revision": used_rev})
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
        # opts = ex.get("options", [])
        # if not isinstance(opts, list) or len(opts) == 0:
        #     return "", ""
        opts = ex.get("options", [])
        if isinstance(opts, str):
            # split by "A)"..."E)" markers
            import re, json
            s = opts.strip()
            if s.startswith("["):
                try:
                    opts = json.loads(s)
                except Exception:
                    pass
            if isinstance(opts, str):
                # find "A) ... B) ... C) ..." spans
                marks = list(re.finditer(r"\b[A-E]\)", s))
                parts = []
                for i,m in enumerate(marks):
                    a = m.end()
                    b = marks[i+1].start() if i+1 < len(marks) else len(s)
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
        p = maybe_add_answer_prefix(p, True, answer_prefix)  # force for AQuA (improves extraction)
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
    # OpenBookQA has had occasional HF cache / feature-schema issues on some environments.
    # We try a small set of HF ids/configs + our load_hf_dataset fallbacks.
    hf_cfg_candidates = [
        ("openbookqa", "main"),
        ("openbookqa", None),
        ("allenai/openbookqa", "main"),
        ("allenai/openbookqa", None),
    ]
    ds = None
    used_rev = None
    cfg = None
    last_err = None
    for _hf_id, _cfg in hf_cfg_candidates:
        try:
            if _cfg is None:
                _ds, _rev = load_hf_dataset(_hf_id)
            else:
                _ds, _rev = load_hf_dataset(_hf_id, _cfg)
            ds, used_rev, cfg = _ds, _rev, _cfg
            hf_id = _hf_id
            break
        except Exception as e:
            last_err = e
            continue
    if ds is None:
        raise last_err

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
    """
    LogiQA loader compatible with datasets>=4.x:
      - Prefer loading from auto parquet branch when a dataset repo has loading scripts.
      - Also supports datasets that store JSON per-row in a single 'text' column (e.g. LogiQA2.0 variants).
    """
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
    ds, used_rev = load_hf_dataset("lighteval/piqa", trust_remote_code=True) # load_dataset("lighteval/piqa", split="validation")#
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
    hf_id = "Salesforce/wikitext"
    cfg = "wikitext-2-raw-v1"
    ds, used_rev = load_hf_dataset(hf_id, cfg)

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
# Code & extra math benchmarks (for calibration mix)
# ============================================================

def load_humaneval(
    *,
    n_subspace: int,
    n_eval: int,
    seed: int,
    template_randomization: bool = True,
    template_seed: int = 1234,
    shuffle_choices: bool = False,  # unused, for API parity
    add_answer_prefix: bool = False,
    answer_prefix: str = "\nFinal answer:",
) -> Tuple[List[Example], List[Example], Dict[str, Any]]:
    """
    HumanEval (code generation). Primarily used for calibration diversity.

    NOTE:
      - We set require_gold_eval=False because proper evaluation usually needs
        pass@k (running unit tests), which is out-of-scope for this simple loader.
      - Gold code is still returned when available (canonical_solution).
    """
    dataset_name = "humaneval"
    hf_id = "openai_humaneval"
    ds, used_rev = load_hf_dataset(hf_id)

    def build_one(ex: dict, ex_id: str) -> Tuple[str, str]:
        ex = _maybe_parse_json_blob_example(ex)
        prompt = (ex.get("prompt") or ex.get("question") or "").strip()
        gold = (ex.get("canonical_solution") or ex.get("solution") or "").strip()
        if not prompt:
            return "", ""

        # Add a lightweight instruction wrapper (model-agnostic).
        p = (
            "You are a helpful coding assistant.\n"
            "Complete the following Python function.\n\n"
            + prompt.rstrip()
        )
        if add_answer_prefix:
            # For code tasks, a code block prefix is usually more appropriate than "Final answer:".
            p = p.rstrip() + "\n\n```python\n"
        return p, gold

    sub_exs, eval_exs, meta = _build_from_splits_with_fallback(
        ds,
        dataset_name,
        build_one,
        n_subspace=n_subspace,
        n_eval=max(1, n_eval),
        seed=seed,
        sub_candidates=["test", "validation", "train"],
        eval_candidates=["test", "validation", "train"],
        require_gold_eval=False,
    )
    meta["hf_id"] = hf_id
    if used_rev:
        meta["hf_revision"] = used_rev
    meta["task_type"] = "code"
    return sub_exs, eval_exs, meta


def load_mbpp(
    *,
    n_subspace: int,
    n_eval: int,
    seed: int,
    template_randomization: bool = True,
    template_seed: int = 1234,
    shuffle_choices: bool = False,  # unused
    add_answer_prefix: bool = False,
    answer_prefix: str = "\nFinal answer:",
) -> Tuple[List[Example], List[Example], Dict[str, Any]]:
    """
    MBPP (Mostly Basic Programming Problems) for code-style calibration.
    """
    dataset_name = "mbpp"
    hf_id = "mbpp"
    ds, used_rev = load_hf_dataset(hf_id)

    def build_one(ex: dict, ex_id: str) -> Tuple[str, str]:
        ex = _maybe_parse_json_blob_example(ex)
        text = (ex.get("prompt") or ex.get("text") or ex.get("question") or "").strip()
        gold = (ex.get("code") or ex.get("solution") or "").strip()
        if not text:
            return "", ""
        p = (
            "You are a helpful coding assistant.\n"
            "Write Python code that solves the following problem.\n\n"
            f"Problem:\n{text.rstrip()}\n"
        )
        if add_answer_prefix:
            p = p.rstrip() + "\n\n```python\n"
        return p, gold

    sub_exs, eval_exs, meta = _build_from_splits_with_fallback(
        ds,
        dataset_name,
        build_one,
        n_subspace=n_subspace,
        n_eval=max(1, n_eval),
        seed=seed,
        sub_candidates=["train", "validation", "test"],
        eval_candidates=["test", "validation", "train"],
        require_gold_eval=False,
    )
    meta["hf_id"] = hf_id
    if used_rev:
        meta["hf_revision"] = used_rev
    meta["task_type"] = "code"
    return sub_exs, eval_exs, meta


def _extract_boxed_answer_from_solution(solution: str) -> str:
    """
    Best-effort extraction for MATH-style solutions where final answer is in \\boxed{...}.
    """
    if not isinstance(solution, str):
        return ""
    s = solution
    key = r"\boxed{"
    i = s.find(key)
    if i < 0:
        # sometimes \\boxed( ... )
        m = re.search(r"\\boxed\s*[\{\(]([^}\)]*)[\}\)]", s)
        return (m.group(1).strip() if m else "")
    j = i + len(key)
    depth = 1
    out = []
    while j < len(s) and depth > 0:
        ch = s[j]
        if ch == "{":
            depth += 1
            out.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
            out.append(ch)
        else:
            out.append(ch)
        j += 1
    return "".join(out).strip()


def load_competition_math(
    *,
    n_subspace: int,
    n_eval: int,
    seed: int,
    template_randomization: bool = True,
    template_seed: int = 1234,
    shuffle_choices: bool = False,  # unused
    add_answer_prefix: bool = False,
    answer_prefix: str = "\nFinal answer:",
) -> Tuple[List[Example], List[Example], Dict[str, Any]]:
    """
    Competition MATH dataset (math reasoning / LaTeX solutions).
    Primarily used for calibration diversity.
    """
    dataset_name = "competition_math"
    hf_id = "qwedsacf/competition_math"
    ds, used_rev = load_hf_dataset(hf_id)

    def build_one(ex: dict, ex_id: str) -> Tuple[str, str]:
        ex = _maybe_parse_json_blob_example(ex)
        prob = (ex.get("problem") or ex.get("question") or "").strip()
        sol = (ex.get("solution") or "").strip()
        gold = _extract_boxed_answer_from_solution(sol)
        if not prob:
            return "", ""
        p = (
            "Solve the following math problem.\n"
            "Give only the final answer.\n\n"
            f"Problem:\n{prob.rstrip()}\n"
        )
        if add_answer_prefix:
            p = p.rstrip() + answer_prefix
        return p, gold

    sub_exs, eval_exs, meta = _build_from_splits_with_fallback(
        ds,
        dataset_name,
        build_one,
        n_subspace=n_subspace,
        n_eval=max(1, n_eval),
        seed=seed,
        sub_candidates=["train", "validation", "test"],
        eval_candidates=["test", "validation", "train"],
        # relaxed: some rows may miss boxed parsing, and math evaluation needs more work anyway
        require_gold_eval=False,
    )
    meta["hf_id"] = hf_id
    if used_rev:
        meta["hf_revision"] = used_rev
    meta["task_type"] = "math"
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
    "humaneval": load_humaneval,
    "mbpp": load_mbpp,
    "competition_math": load_competition_math,
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


# def is_correct(task: str, pred: Any, gold: Any) -> bool:
#     t = (task or "").strip().lower()

#     if t == "gsm8k":
#         return _normalize_number_str(pred) == _normalize_number_str(gold) and _normalize_number_str(pred) != ""
def _norm_bool(s: str) -> str:
    s = s.strip().lower()
    # 常见等价表达
    if s in {"yes", "y", "true", "t", "1"}: return "true"
    if s in {"no", "n", "false", "f", "0"}: return "false"
    # 宽松匹配（生成里常出现句子）
    if "yes" in s and "no" not in s: return "true"
    if "no" in s and "yes" not in s: return "false"
    if "true" in s and "false" not in s: return "true"
    if "false" in s and "true" not in s: return "false"
    return s

def _extract_last_number(s: str) -> str:
    import re
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", s.replace(",", ""))
    return nums[-1] if nums else ""

def is_correct(task: str, pred: str, gold: str) -> bool:
    if gold is None:
        return False
    t = task.lower()
    p = str(pred).strip()
    g = str(gold).strip()

    if t == "boolq":
        return _norm_bool(p) == _norm_bool(g)

    if t == "gsm8k":
        # gold 可能形如 "#### 42"
        gnum = _extract_last_number(g)
        pnum = _extract_last_number(p)
        if not gnum or not pnum:
            return False
        try:
            # 数值相等（允许 1e-4 容忍）
            return abs(float(gnum) - float(pnum)) < 1e-4
        except Exception:
            return False

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
