# -*- coding: utf-8 -*-
"""
prove_sharedness_decode_full.py

Self-contained runner that reuses utilities from prove_sharedness_decode_fair.py (imported as base),
but DOES NOT call base.main() (avoids args parsing bugs when imported/wrapped).

Pipeline (same as fair):
  1) collect decode-phase (seq_len==1) last-token hidden states under cached decoding
  2) pooled PCA subspace
  3) sharedness via relvar threshold tau in >= m tasks
  4) significance under nulls (perm; optional scramble+recompute)

Plus: full benchmark prompt loader (math/code) and chat-template-aware decode collection for Gemma3-*-it.

CUDA_VISIBLE_DEVICES=1 python prove_sharedness_decode_full.py   --model google/gemma-3-12b-it   --device cuda   --model_dtype fp32   --layer 10   --n_prompts 128   --calib_max_new_tokens 128   --max_prompt_len 512   --per_task_max_states 20000   --tau 0.001   --m_shared all   --null_perm_trials 2000   --null_scramble_trials 100   --out_json results/full_benchmark/gemma3_exist.json   --out_txt  results/full_benchmark/gemma3_exist.txt

CUDA_VISIBLE_DEVICES=1 python prove_sharedness_decode_full.py   --model meta/llama-2-7b-chat-hf   --device cuda   --model_dtype fp32   --layer 10   --n_prompts 128   --calib_max_new_tokens 128   --max_prompt_len 512   --per_task_max_states 20000   --tau 0.001   --m_shared all   --null_perm_trials 2000   --null_scramble_trials 100   --out_json results/full_benchmark/llama2-7b-chat-hf_exist.json   --out_txt  results/full_benchmark/llama2-7b-chat-hf_exist.txt

CUDA_VISIBLE_DEVICES=1 python prove_sharedness_decode_full.py   --model Qwen/Qwen2.5-7B-Instruct   --device cuda   --model_dtype fp32   --layer 10   --n_prompts 128   --calib_max_new_tokens 128   --max_prompt_len 512   --per_task_max_states 20000   --tau 0.001   --m_shared all   --null_perm_trials 2000   --null_scramble_trials 100   --out_json results/full_benchmark/Qwen2.5-7B-Instruct_exist.json   --out_txt  results/full_benchmark/Qwen2.5-7B-Instruct_exist.txt

"""

from __future__ import annotations

from typing import Dict, List, Optional, Any, Tuple
import os
import re
import json
import random
import argparse
import hashlib
import sys
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

import prove_sharedness_decode_fair as base


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


def to_py(obj: Any):
    # JSON-safe conversion
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return obj


# -----------------------------
# Prompt builders (new tasks)
# -----------------------------
def _clean_text(x: Any) -> str:
    s = "" if x is None else str(x)
    s = s.replace("\r\n", "\n")
    return s.strip()


def _get_first(ex: dict, keys: List[str], default: str = "") -> str:
    for k in keys:
        if k in ex and ex[k] is not None and str(ex[k]).strip() != "":
            return _clean_text(ex[k])
    return default


def build_prompt_math_openanswer(problem: str) -> str:
    problem = _clean_text(problem)
    return (
        "You are a careful mathematician.\n"
        "Solve the following problem. Show your reasoning, then give the final answer.\n"
        "Return the final answer on the last line as: Final Answer: <answer>\n\n"
        f"Problem:\n{problem}\n\n"
        "Solution:\n"
    )


def build_prompt_code_generation(spec: str, starter_code: str = "") -> str:
    spec = _clean_text(spec)
    starter_code = _clean_text(starter_code)
    if starter_code:
        return (
            "You are a helpful coding assistant.\n"
            "Write a correct Python 3 solution for the following programming task.\n"
            "Return ONLY code (no explanation).\n\n"
            f"Task:\n{spec}\n\n"
            f"Starter code:\n```python\n{starter_code}\n```\n\n"
            "Code:\n```python\n"
        )
    return (
        "You are a helpful coding assistant.\n"
        "Write a correct Python 3 solution for the following programming task.\n"
        "Return ONLY code (no explanation).\n\n"
        f"Task:\n{spec}\n\n"
        "Code:\n```python\n"
    )


def build_prompt_humaneval(prompt: str) -> str:
    prompt = _clean_text(prompt)
    return (
        "Complete the following Python function.\n"
        "Return ONLY code.\n\n"
        "```python\n"
        f"{prompt}\n"
    )


def build_prompt_mbpp(text: str, starter: str = "") -> str:
    text = _clean_text(text)
    starter = _clean_text(starter)
    if starter:
        return (
            "Write a Python 3 function that satisfies the description.\n"
            "Return ONLY code.\n\n"
            f"Description:\n{text}\n\n"
            f"Starter:\n```python\n{starter}\n```\n\n"
            "Code:\n```python\n"
        )
    return (
        "Write a Python 3 function that satisfies the description.\n"
        "Return ONLY code.\n\n"
        f"Description:\n{text}\n\n"
        "Code:\n```python\n"
    )


# -----------------------------
# Full benchmark prompt loader
# -----------------------------
def _try_load_any(paths: List[tuple[str, Optional[str]]]):
    """
    Try (path, name) in order; returns first successful dataset object or None.
    """
    for path, name in paths:
        ds = base._try_load_dataset(path, name)
        if ds is not None:
            return ds, path, name
    return None, None, None



def _safe_sample(ds, n_prompts: int, seed: int):
    split = base._pick_split(ds)
    rows = base.sample_hf_split(ds[split], n_prompts, seed)
    return rows


def load_calib_prompts_full(n_prompts: int, seed: int) -> Dict[str, List[str]]:
    prompts: Dict[str, List[str]] = {}

    # original 9 tasks (copied, robust to partial failure)
    ds = base._try_load_dataset("gsm8k", "main")
    if ds is not None:
        rows = _safe_sample(ds, n_prompts, seed + 1)
        prompts["gsm8k"] = [base.build_prompt_gsm8k(ex["question"]) for ex in rows]

    ds = base._try_load_dataset("commonsense_qa")
    if ds is not None:
        rows = _safe_sample(ds, n_prompts, seed + 11)
        prompts["commonsenseqa"] = [base.build_prompt_commonsenseqa(ex["question"], ex["choices"]) for ex in rows]

    ds = base._try_load_dataset("ChilleD/StrategyQA")
    if ds is not None:
        rows = _safe_sample(ds, n_prompts, seed + 21)
        prompts["strategyqa"] = [base.build_prompt_strategyqa(ex["question"]) for ex in rows]

    ds = base._try_load_dataset("aqua_rat")
    if ds is not None:
        rows = _safe_sample(ds, n_prompts, seed + 31)
        prompts["aqua"] = [base.build_prompt_aqua(ex["question"], ex["options"]) for ex in rows]

    ds = base._try_load_dataset("ai2_arc", "ARC-Challenge")
    if ds is not None:
        rows = _safe_sample(ds, n_prompts, seed + 41)
        arc_prompts = []
        for ex in rows:
            stem = _clean_text(ex.get("question", ""))
            choices = ex.get("choices", {}) or {}
            labels = choices.get("label", []) if isinstance(choices, dict) else []
            texts = choices.get("text", []) if isinstance(choices, dict) else []
            if stem and labels and texts:
                arc_prompts.append(base.build_prompt_mc(stem, labels, texts))
        if arc_prompts:
            prompts["arc_challenge"] = arc_prompts

    ds = base._try_load_dataset("openbookqa")
    if ds is not None:
        rows = _safe_sample(ds, n_prompts, seed + 51)
        ob_prompts = []
        for ex in rows:
            q = ex.get("question_stem", "")
            ch = ex.get("choices", {})
            labels = ch.get("label", [])
            texts = ch.get("text", [])
            if q and labels and texts:
                ob_prompts.append(base.build_prompt_mc(q, labels, texts))
        if ob_prompts:
            prompts["openbookqa"] = ob_prompts

    ds = base._try_load_dataset("qasc")
    if ds is not None:
        rows = _safe_sample(ds, n_prompts, seed + 61)
        qasc_prompts = []
        for ex in rows:
            q = ex.get("question", "")
            ch = ex.get("choices", {})
            labels = ch.get("label", [])
            texts = ch.get("text", [])
            if q and labels and texts:
                qasc_prompts.append(base.build_prompt_mc(q, labels, texts))
        if qasc_prompts:
            prompts["qasc"] = qasc_prompts

    ds = base._try_load_dataset("boolq")
    if ds is not None:
        rows = _safe_sample(ds, n_prompts, seed + 71)
        bq_prompts = []
        for ex in rows:
            passage = ex.get("passage", "")
            q = ex.get("question", "")
            if passage and q:
                bq_prompts.append(base.build_prompt_boolq(passage, q))
        if bq_prompts:
            prompts["boolq"] = bq_prompts

    ds = base._try_load_dataset("piqa")
    if ds is not None:
        rows = _safe_sample(ds, n_prompts, seed + 81)
        piqa_prompts = []
        for ex in rows:
            goal = ex.get("goal", "")
            sol1 = ex.get("sol1", "")
            sol2 = ex.get("sol2", "")
            if goal and sol1 and sol2:
                piqa_prompts.append(base.build_prompt_piqa(goal, sol1, sol2))
        if piqa_prompts:
            prompts["piqa"] = piqa_prompts

    # # extra benchmarks
    # ds, used_path, used_name = _try_load_any([
    #     ("EleutherAI/hendrycks_math", None),
    #     ("hendrycks/competition_math", None),
    #     ("qwedsacf/competition_math", None),
    #     ("Maxwell-Jia/MATH", None),
    # ])
    # if ds is not None:
    #     rows = _safe_sample(ds, n_prompts, seed + 101)
    #     out = []
    #     for ex in rows:
    #         problem = _get_first(ex, ["problem", "question", "prompt"], default="")
    #         if problem:
    #             out.append(build_prompt_math_openanswer(problem))
    #     if out:
    #         prompts["math"] = out
    #     print(f"[Info] loaded MATH from {used_path}" + (f"/{used_name}" if used_name else ""))

    # 1) MATH (Hendrycks et al.) — try per-subject configs if needed
    #    If you want ONE combined "math" task, keep only one config (e.g., algebra).
    #    If you want broader coverage, include several configs and store them as separate tasks.
    hendrycks_configs = [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]
    ds, used_path, used_name = _try_load_any(
        [( "EleutherAI/hendrycks_math", cfg) for cfg in hendrycks_configs]
        + [
            ("hendrycks/competition_math", None),
            ("qwedsacf/competition_math", None),
            ("Maxwell-Jia/MATH", None),
        ]
    )

    if ds is not None:
        rows = _safe_sample(ds, n_prompts, seed + 101)
        out = []
        for ex in rows:
            problem = _get_first(ex, ["problem", "question", "prompt"], default="")
            if problem:
                out.append(build_prompt_math_openanswer(problem))

        # put under a name that reflects the config if we used hendrycks_math
        if out:
            task_name = "math"
            if used_path == "EleutherAI/hendrycks_math" and used_name:
                task_name = f"math_{used_name}"
            prompts[task_name] = out

        print(f"[Info] loaded MATH from {used_path}" + (f"/{used_name}" if used_name else ""))

    ds, used_path, used_name = _try_load_any([
        ("HuggingFaceH4/aime_2024", None),
        ("AI-MO/aimo-validation-aime", None),
        ("GY2233/AIME-2024-2025", None),
        ("math-ai/aime24", None),
    ])
    if ds is not None:
        rows = _safe_sample(ds, n_prompts, seed + 111)
        out = []
        for ex in rows:
            problem = _get_first(ex, ["problem", "question", "prompt"], default="")
            if problem:
                out.append(build_prompt_math_openanswer(problem))
        if out:
            prompts["aime"] = out
        print(f"[Info] loaded AIME from {used_path}" + (f"/{used_name}" if used_name else ""))

    # ds, used_path, used_name = _try_load_any([
    #     ("RUC-AIBOX/OlymMATH", None),
    # ])
    # if ds is not None:
    #     rows = _safe_sample(ds, n_prompts, seed + 121)
    #     out = []
    #     for ex in rows:
    #         problem = _get_first(ex, ["problem", "question", "prompt"], default="")
    #         if problem:
    #             out.append(build_prompt_math_openanswer(problem))
    #     if out:
    #         prompts["olymmath"] = out
    #     print(f"[Info] loaded OlymMATH from {used_path}" + (f"/{used_name}" if used_name else ""))

    # 3) OlymMATH (olympiad-level) — requires config
    olymmath_configs = ["en-hard"]  # or ["en-hard", "en-easy", "zh-hard", "zh-easy", "lean"]
    ds, used_path, used_name = _try_load_any([("RUC-AIBOX/OlymMATH", cfg) for cfg in olymmath_configs])

    if ds is not None:
        rows = _safe_sample(ds, n_prompts, seed + 121)
        out = []
        for ex in rows:
            problem = _get_first(ex, ["problem", "question", "prompt"], default="")
            if problem:
                out.append(build_prompt_math_openanswer(problem))

        if out:
            task_name = "olymmath"
            if used_name:
                task_name = f"olymmath_{used_name}"
            prompts[task_name] = out

        print(f"[Info] loaded OlymMATH from {used_path}" + (f"/{used_name}" if used_name else ""))


    ds, used_path, used_name = _try_load_any([
        ("livecodebench/code_generation", None),
        ("livecodebench/code_generation_lite", None),
        ("livecodebench/code_generation_lite", "v1"),
    ])
    if ds is not None:
        rows = _safe_sample(ds, n_prompts, seed + 131)
        out = []
        for ex in rows:
            spec = _get_first(ex, ["question", "prompt", "instruction", "problem", "description"], default="")
            starter = _get_first(ex, ["starter_code", "code_starter", "skeleton", "template"], default="")
            if spec:
                out.append(build_prompt_code_generation(spec, starter))
        if out:
            prompts["livecodebench"] = out
        print(f"[Info] loaded LiveCodeBench from {used_path}" + (f"/{used_name}" if used_name else ""))

    ds, used_path, used_name = _try_load_any([
        ("openai/openai_humaneval", None),
        ("codeparrot/instructhumaneval", None),
    ])
    if ds is not None:
        rows = _safe_sample(ds, n_prompts, seed + 141)
        out = []
        for ex in rows:
            p = _get_first(ex, ["prompt"], default="")
            if p:
                out.append(build_prompt_humaneval(p))
        if out:
            prompts["humaneval"] = out
        print(f"[Info] loaded HumanEval from {used_path}" + (f"/{used_name}" if used_name else ""))

    ds, used_path, used_name = _try_load_any([
        ("google-research-datasets/mbpp", None),
        ("Muennighoff/mbpp", None),
        ("claudios/google-research-datasets__mbpp", None),
    ])
    if ds is not None:
        rows = _safe_sample(ds, n_prompts, seed + 151)
        out = []
        for ex in rows:
            desc = _get_first(ex, ["text", "prompt", "question", "description"], default="")
            starter = _get_first(ex, ["code", "starter_code"], default="")
            if desc:
                out.append(build_prompt_mbpp(desc, starter))
        if out:
            prompts["mbpp"] = out
        print(f"[Info] loaded MBPP from {used_path}" + (f"/{used_name}" if used_name else ""))

    if len(prompts) == 0:
        raise RuntimeError("No datasets could be loaded; check HF datasets access / network / cache.")
    return prompts


# -----------------------------
# Decode last-token activation collector (same behavior as base)
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
# Chat-template-aware decode collection (Gemma3-*-it friendly)
# -----------------------------
def _top_p_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    # reuse base helpers if present
    if hasattr(base, "top_p_filtering"):
        return base.top_p_filtering(logits, top_p)
    # minimal implementation
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


def _top_k_filtering(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if hasattr(base, "top_k_filtering"):
        return base.top_k_filtering(logits, top_k)
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
    *,
    batch_size: int,
    max_prompt_len: int,
    calib_max_new_tokens: int,
    decoding: str,
    temperature: float,
    top_p: float,
    top_k: int,
) -> None:
    """
    Collect last-token hidden states during cached decode (seq_len==1).

    Fixes:
      - chat_template path may return Tensor instead of dict; normalize to dict.
      - left padding => left truncation to preserve prompt tail (choices/question end).
    """
    assert decoding in ["greedy", "sample"]
    device = next(model.parameters()).device
    model.eval()

    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos = tokenizer.eos_token_id
    if eos is None:
        raise RuntimeError("tokenizer.eos_token_id is None")

    chat_tmpl = getattr(tokenizer, "chat_template", None)
    use_chat = bool(chat_tmpl and str(chat_tmpl).strip() and hasattr(tokenizer, "apply_chat_template"))

    def _normalize_to_dict(enc) -> Dict[str, torch.Tensor]:
        """
        Make sure enc is a dict with 'input_ids' and 'attention_mask', both torch.Tensor on device.
        Handles:
          - torch.Tensor (assumed input_ids)
          - BatchEncoding / dict
          - dict of lists
        """
        # Case 1: tokenizer.apply_chat_template returned a Tensor directly
        if isinstance(enc, torch.Tensor):
            input_ids = enc.to(device)
            attention_mask = torch.ones_like(input_ids, device=device)
            return {"input_ids": input_ids, "attention_mask": attention_mask}

        # Case 2: BatchEncoding is dict-like; plain dict also
        if isinstance(enc, dict):
            out: Dict[str, torch.Tensor] = {}
            for k, v in enc.items():
                if isinstance(v, torch.Tensor):
                    out[k] = v.to(device)
                else:
                    out[k] = torch.tensor(v, device=device)
            if "input_ids" not in out:
                raise RuntimeError(f"Tokenizer output missing input_ids. Keys={list(out.keys())}")
            if "attention_mask" not in out:
                out["attention_mask"] = (out["input_ids"] != tokenizer.pad_token_id).long()
            return out

        # Case 3: other types (very rare) -> try best-effort
        if hasattr(enc, "to") and hasattr(enc, "__getitem__"):
            try:
                enc = enc.to(device)
                if "attention_mask" not in enc:
                    enc["attention_mask"] = (enc["input_ids"] != tokenizer.pad_token_id).long()
                return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
            except Exception as e:
                raise RuntimeError(f"Unexpected tokenizer output type: {type(enc)}; err={e}")

        raise RuntimeError(f"Unexpected tokenizer output type: {type(enc)}")

    def _encode_batch(batch_prompts: List[str]) -> Dict[str, torch.Tensor]:
        if use_chat:
            convs = [[{"role": "user", "content": p}] for p in batch_prompts]

            # Prefer tokenize=True if it returns a usable structure; otherwise fallback
            try:
                enc = tokenizer.apply_chat_template(
                    convs,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_prompt_len,
                )
            except Exception:
                rendered = tokenizer.apply_chat_template(convs, tokenize=False, add_generation_prompt=True)
                if isinstance(rendered, str):
                    rendered = [rendered]
                enc = tokenizer(
                    rendered,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_prompt_len,
                    add_special_tokens=False,  # avoid double BOS/EOS
                )
        else:
            enc = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_prompt_len,
            )

        return _normalize_to_dict(enc)

    for i in tqdm(range(0, len(prompts), batch_size), desc="CollectDecode"):
        batch = prompts[i : i + batch_size]
        inputs = _encode_batch(batch)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        B = input_ids.shape[0]
        unfinished = torch.ones(B, dtype=torch.bool, device=device)

        # Prefill (no capture)
        collector.set_capture(False, None)
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        logits = out.logits[:, -1, :]
        past = getattr(out, "past_key_values", None)

        # Decode loop
        for _step in range(int(calib_max_new_tokens)):
            if decoding == "greedy":
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                lt = logits / max(float(temperature), 1e-6)
                lt = _top_k_filtering(lt, top_k=int(top_k))
                lt = _top_p_filtering(lt, top_p=float(top_p))
                probs = torch.softmax(lt, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            next_token = torch.where(
                unfinished.unsqueeze(-1),
                next_token,
                torch.full_like(next_token, int(eos)),
            )

            newly_finished = unfinished & (next_token.squeeze(-1) == int(eos))
            unfinished[newly_finished] = False
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
            past = getattr(out, "past_key_values", None)

        collector.set_capture(False, None)


# -----------------------------
# Sharedness helpers (reuse base if exists; else implement minimal)
# -----------------------------
def center_and_balance(
    X_by_task: Dict[str, np.ndarray],
    *,
    per_task_max_states: int,
    balance_to: str,
    seed: int,
) -> Tuple[Dict[str, np.ndarray], int]:
    # Use base implementation if present
    if hasattr(base, "center_and_balance"):
        return base.center_and_balance(X_by_task, per_task_max_states=per_task_max_states, balance_to=balance_to, seed=seed)

    rng = np.random.default_rng(seed)
    capped: Dict[str, np.ndarray] = {}
    for t, X in X_by_task.items():
        if X.shape[0] > per_task_max_states:
            idx = rng.choice(X.shape[0], size=per_task_max_states, replace=False)
            X = X[idx]
        capped[t] = X.astype(np.float32, copy=False)

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
        X = X - X.mean(axis=0, keepdims=True)
        balanced[t] = X.astype(np.float32, copy=False)

    return balanced, n0


def compute_relvar_in_basis(X: np.ndarray, Q: np.ndarray) -> np.ndarray:
    if hasattr(base, "compute_relvar_in_basis"):
        return base.compute_relvar_in_basis(X, Q)
    Z = X @ Q
    v = np.var(Z, axis=0)
    s = float(v.sum()) + 1e-12
    return (v / s).astype(np.float32, copy=False)


def compute_shared_indices_from_relvar(relvar_by_task: Dict[str, np.ndarray], tau: float, m_shared: int) -> List[int]:
    if hasattr(base, "compute_shared_indices_from_relvar"):
        return base.compute_shared_indices_from_relvar(relvar_by_task, tau=tau, m_shared=m_shared)
    tasks = list(relvar_by_task.keys())
    rel = np.stack([relvar_by_task[t] for t in tasks], axis=0)  # [T,k]
    ok = (rel >= float(tau)).astype(np.int32)
    cnt = ok.sum(axis=0)
    idx = np.where(cnt >= int(m_shared))[0]
    return idx.tolist()


def null_perm_sharedcount(relvar_by_task: Dict[str, np.ndarray], tau: float, m_shared: int, trials: int, seed: int):
    if hasattr(base, "null_perm_sharedcount"):
        return base.null_perm_sharedcount(relvar_by_task, tau=tau, m_shared=m_shared, trials=trials, seed=seed)
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
    if hasattr(base, "scramble_features_orthogonal"):
        return base.scramble_features_orthogonal(X, rng)
    D = X.shape[1]
    perm = rng.permutation(D)
    signs = rng.choice([-1.0, 1.0], size=D).astype(np.float32)
    return (X[:, perm] * signs[None, :]).astype(np.float32, copy=False)


# -----------------------------
# Model loader (reuse base if possible)
# -----------------------------
def load_model_and_tokenizer(model_name: str, device: str, model_dtype: str):
    if hasattr(base, "load_model_and_tokenizer"):
        return base.load_model_and_tokenizer(model_name, device, model_dtype)

    from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
    if model_dtype == "fp32":
        dtype = torch.float32
    elif model_dtype == "fp16":
        dtype = torch.float16
    else:
        raise ValueError("model_dtype must be fp32 or fp16")

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)

    tok = AutoTokenizer.from_pretrained(model_name)
    if isinstance(tok, bool):
        tok = LlamaTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = model.to(device)
    model.eval()
    model.config.use_cache = True
    return model, tok


def infer_hidden_dim(model) -> Optional[int]:
    if hasattr(base, "infer_hidden_dim"):
        return base.infer_hidden_dim(model)
    cfg = getattr(model, "config", None)
    for k in ("hidden_size", "n_embd", "dim", "d_model", "model_dim", "embed_dim"):
        v = getattr(cfg, k, None)
        if isinstance(v, int) and v > 0:
            return v
    try:
        emb = model.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight") and isinstance(emb.weight, torch.Tensor):
            return int(emb.weight.shape[1])
    except Exception:
        pass
    return None


# -----------------------------
# Main runner
# -----------------------------
def main():
    default_out_json = os.path.join(os.getcwd(), "sharedness_existence_full.json")
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

    ap.add_argument("--tau", type=float, default=0.001)
    ap.add_argument("--m_shared", type=str, default="all")  # "all" or int

    ap.add_argument("--per_task_max_states", type=int, default=20000)
    ap.add_argument("--balance_to", type=str, default="min")  # "min" or int

    ap.add_argument("--null_perm_trials", type=int, default=2000)
    ap.add_argument("--null_scramble_trials", type=int, default=0)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tasks", type=str, default="all",
                    help='Task subset: "all" or comma-separated keys from loaded prompts (e.g., "gsm8k,boolq").')

    ap.add_argument("--out_json", type=str, default=default_out_json)
    ap.add_argument("--out_txt", type=str, default=default_out_txt,
                    help='Tee stdout prints into this txt file. Use "" or "none" to disable.')

    args = ap.parse_args()

    # tee stdout early
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

        set_global_seed(int(args.seed))

        print(f"[Env] model={args.model} device={args.device} dtype={args.model_dtype} layer={args.layer}")

        # model
        model, tok = load_model_and_tokenizer(args.model, args.device, args.model_dtype)
        layers, _arch = base.get_model_layers(model) if hasattr(base, "get_model_layers") else (None, None)
        if layers is None:
            raise RuntimeError("base.get_model_layers not found; cannot locate transformer layers.")
        if args.layer >= len(layers):
            raise RuntimeError(f"layer={args.layer} out of range, num_layers={len(layers)}")

        hidden_dim = infer_hidden_dim(model)
        if hidden_dim is None:
            print(f"[Warn] Could not infer hidden_dim. Continue anyway.")
        else:
            print(f"[Env] hidden_dim={hidden_dim}")

        # load prompts (full suite)
        prompts_by_task = load_calib_prompts_full(int(args.n_prompts), int(args.seed))
        all_tasks = list(prompts_by_task.keys())

        if args.tasks != "all":
            want = [t.strip() for t in args.tasks.split(",") if t.strip()]
            prompts_by_task = {t: prompts_by_task[t] for t in want if t in prompts_by_task}
            if not prompts_by_task:
                raise RuntimeError(f"--tasks={args.tasks} produced empty set. Available: {all_tasks}")

        tasks = list(prompts_by_task.keys())
        print(f"[Data] tasks={tasks} n_prompts_per_task(target)={args.n_prompts}")
        for t in tasks:
            print(f"[Data] task={t} loaded_prompts={len(prompts_by_task[t])}")

        # collector + hooks
        collector = DecodeLastTokenActivationCollector([int(args.layer)])
        handles = []
        for li in [int(args.layer)]:
            handles.append(layers[li].register_forward_hook(collector.make_hook(li)))

        # collect decode states
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
                        batch_size=int(args.batch_size),
                        max_prompt_len=int(args.max_prompt_len),
                        calib_max_new_tokens=int(args.calib_max_new_tokens),
                        decoding=args.calib_decoding,
                        temperature=float(args.temperature),
                        top_p=float(args.top_p),
                        top_k=int(args.top_k),
                    )
        finally:
            for h in handles:
                try:
                    h.remove()
                except Exception:
                    pass
            collector.set_capture(False, None)

        # build X_by_task (single layer)
        X_raw: Dict[str, np.ndarray] = {}
        for task in tasks:
            X = collector.get(task, int(args.layer))
            if X is None or X.shape[0] == 0:
                raise RuntimeError(f"No activations collected for task={task}, layer={args.layer}")
            X_raw[task] = X
            print(f"[Collect] task={task} states={X.shape[0]} x {X.shape[1]}")

        # fair preprocessing: cap, balance, task-center
        X_by_task, n0 = center_and_balance(
            X_raw,
            per_task_max_states=int(args.per_task_max_states),
            balance_to=str(args.balance_to),
            seed=int(args.seed) + 999,
        )
        print(f"[Fair] balanced states per task = {n0}")

        # pooled PCA via base.compute_cross_task_subspace (must exist)
        if not hasattr(base, "compute_cross_task_subspace"):
            raise RuntimeError("base.compute_cross_task_subspace not found. Ensure prove_sharedness_decode_fair imports it.")

        task_acts: Dict[str, Dict[int, np.ndarray]] = {t: {int(args.layer): X_by_task[t]} for t in tasks}
        joint_subspace, cross_dim, contributions, full_pca_info = base.compute_cross_task_subspace(
            task_acts,
            variance_threshold=float(args.pca_var),
            min_dim=int(args.min_dim),
            max_dim=int(args.max_dim),
            return_full_pca=True,
        )
        if joint_subspace is None or int(cross_dim) <= 0:
            raise RuntimeError("compute_cross_task_subspace failed")

        Q = joint_subspace.astype(np.float32, copy=False)
        k = int(cross_dim)
        print(f"[PCA] cross_dim={k} / {Q.shape[0]}  (pca_var={args.pca_var})")

        # relvar profiles
        relvar_by_task: Dict[str, np.ndarray] = {}
        for t in tasks:
            relvar_by_task[t] = compute_relvar_in_basis(X_by_task[t], Q)

        # sharedness threshold
        if args.m_shared == "all":
            m_shared = len(tasks)
        else:
            m_shared = int(args.m_shared)

        shared_idx = compute_shared_indices_from_relvar(relvar_by_task, tau=float(args.tau), m_shared=m_shared)
        obs_shared_count = int(len(shared_idx))

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

        # Null-1: relvar permutation
        null1_counts, _ = null_perm_sharedcount(
            relvar_by_task,
            tau=float(args.tau),
            m_shared=m_shared,
            trials=int(args.null_perm_trials),
            seed=int(args.seed) + 12345,
        )
        p1 = float((np.sum(null1_counts >= obs_shared_count) + 1) / (len(null1_counts) + 1))

        print("\n" + "=" * 80)
        print("[Null-1] relvar-permutation (fast)")
        print("=" * 80)
        print(f"trials={len(null1_counts)} null_mean={float(null1_counts.mean()):.2f} "
              f"p95={float(np.percentile(null1_counts, 95)):.2f} max={int(null1_counts.max())}")
        print(f"p-value (null>=obs) = {p1:.4g}")

        # Null-2: scramble + recompute PCA (optional)
        null2_counts = None
        p2 = None
        if int(args.null_scramble_trials) > 0:
            print("\n" + "=" * 80)
            print("[Null-2] per-task orthogonal scramble + recompute PCA (stronger, slower)")
            print("=" * 80)
            rng = np.random.default_rng(int(args.seed) + 777)
            tmp = []
            for b in range(int(args.null_scramble_trials)):
                X_scr: Dict[str, np.ndarray] = {}
                for t in tasks:
                    Xs = scramble_features_orthogonal(X_by_task[t], rng)
                    Xs = Xs - Xs.mean(axis=0, keepdims=True)
                    X_scr[t] = Xs.astype(np.float32, copy=False)
                task_acts_scr = {t: {int(args.layer): X_scr[t]} for t in tasks}
                joint2, k2, _, _ = base.compute_cross_task_subspace(
                    task_acts_scr,
                    variance_threshold=float(args.pca_var),
                    min_dim=int(args.min_dim),
                    max_dim=int(args.max_dim),
                    return_full_pca=True,
                )
                if joint2 is None or int(k2) <= 0:
                    tmp.append(0)
                    continue
                Q2 = joint2.astype(np.float32, copy=False)
                rel2 = {t: compute_relvar_in_basis(X_scr[t], Q2) for t in tasks}
                idx2 = compute_shared_indices_from_relvar(rel2, tau=float(args.tau), m_shared=m_shared)
                tmp.append(int(len(idx2)))
                print(f"  trial={b+1}/{args.null_scramble_trials}: cross_dim={int(k2)} shared_count={int(len(idx2))}")
            null2_counts = np.array(tmp, dtype=np.int32)
            p2 = float((np.sum(null2_counts >= obs_shared_count) + 1) / (len(null2_counts) + 1))
            print(f"[Null-2] mean={float(null2_counts.mean()):.2f} p95={float(np.percentile(null2_counts, 95)):.2f} "
                  f"max={int(null2_counts.max())}")
            print(f"[Null-2] p-value (null>=obs) = {p2:.4g}")

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
                "tasks": ("all" if args.tasks == "all" else args.tasks),
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
            "null1_perm_counts": np.asarray(null1_counts, dtype=np.int32).tolist(),
            "null2_scramble_counts": (None if null2_counts is None else np.asarray(null2_counts, dtype=np.int32).tolist()),
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
