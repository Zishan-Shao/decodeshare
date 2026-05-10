#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sanity_check_eval_models_v3.py

Smoke-test runner to verify:
  1) HF model loading works for decoder-only and seq2seq models
  2) benchmark_dataloaders_ext produces prompts + gold labels
  3) generation + parsing + scoring runs end-to-end

Key improvements vs v2:
  - Robust parsing: prefers "Final answer:" span, then first-line, then fallback
  - BoolQ handled as MC (A/B) if gold looks like a letter; otherwise normalizes yes/no/true/false
  - GSM8K scoring uses numeric comparison (handles "#### 42", commas, etc.)
  - Optional per-task max_new_tokens overrides (GSM8K default higher)
  - Debug print uses the *same* sampled examples the tqdm loop evaluates (no double generation)

Example:
  CUDA_VISIBLE_DEVICES=0 python sanity_check_eval_models_v3.py \
    --models google/gemma-3-12b-it \
    --tasks gsm8k,boolq,piqa,arc_challenge,openbookqa,commonsenseqa \
    --n_eval 8 \
    --max_new_tokens 64 \
    --device cuda \
    --debug_first 1
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from benchmark_dataloaders_ext import load_selected_tasks, Example


# ----------------------------
# Model loading helpers
# ----------------------------

@dataclass
class LoadedModel:
    kind: str  # "causal" or "seq2seq"
    model: torch.nn.Module
    tok: AutoTokenizer


def load_model_any(model_id: str, device: str, dtype: str = "fp16") -> LoadedModel:
    """Load a text-capable model. Tries causal LM first, then seq2seq."""
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    if dtype == "fp16":
        torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32
    elif dtype == "bf16":
        torch_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    else:
        torch_dtype = torch.float32

    # 1) Try causal LM (Gemma/Llama-like)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="auto" if device.startswith("cuda") else None,
        )
        model.eval()
        return LoadedModel(kind="causal", model=model, tok=tok)
    except Exception:
        pass

    # 2) Try seq2seq (T5/BART-like)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map="auto" if device.startswith("cuda") else None,
    )
    model.eval()
    return LoadedModel(kind="seq2seq", model=model, tok=tok)


@torch.no_grad()
def generate_one(
    loaded: LoadedModel,
    prompt: str,
    device: str,
    max_new_tokens: int,
) -> str:
    tok = loaded.tok
    inputs = tok(prompt, return_tensors="pt").to(device)
    gen = loaded.model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    if loaded.kind == "seq2seq":
        # seq2seq output doesn't include input prompt
        return tok.decode(gen[0], skip_special_tokens=True)

    # causal: decode only newly generated tokens
    in_len = inputs["input_ids"].shape[-1]
    new_ids = gen[0][in_len:]
    return tok.decode(new_ids, skip_special_tokens=True)


# ----------------------------
# Parsing + scoring
# ----------------------------

MC_RE = re.compile(r"\b([A-E])\b", flags=re.IGNORECASE)

def _extract_after_final_answer(text: str) -> str:
    m = re.search(r"final\s*answer\s*:\s*(.*)", text, flags=re.IGNORECASE)
    return m.group(1).strip() if m else ""


def _parse_mc(text: str) -> str:
    # 1) Prefer "Final answer: X"
    tail = _extract_after_final_answer(text)
    if tail:
        m = MC_RE.search(tail)
        if m:
            return m.group(1).upper()

    # 2) Prefer first non-empty line starting with a letter
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        m = re.match(r"^([A-E])\b", s.upper())
        if m:
            return m.group(1)
        break

    # 3) Fallback: last A-E token anywhere (least reliable)
    cands = MC_RE.findall(text)
    return cands[-1].upper() if cands else ""


def _parse_number(text: str) -> str:
    # Prefer final answer span; otherwise whole text.
    tail = _extract_after_final_answer(text)
    src = tail if tail else text
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", src.replace(",", ""))
    return nums[-1] if nums else ""


def _norm_bool(s: str) -> str:
    s = s.strip().lower()
    if s in {"yes", "y", "true", "t", "1"}:
        return "true"
    if s in {"no", "n", "false", "f", "0"}:
        return "false"
    if "yes" in s and "no" not in s:
        return "true"
    if "no" in s and "yes" not in s:
        return "false"
    if "true" in s and "false" not in s:
        return "true"
    if "false" in s and "true" not in s:
        return "false"
    return s


def parse_prediction(task: str, gen_text: str, gold: Optional[str] = None) -> str:
    t = task.lower()

    if t == "gsm8k":
        return _parse_number(gen_text)

    if t == "boolq":
        # Some loaders turn BoolQ into A/B MC; others keep true/false.
        g = str(gold).strip().upper() if gold is not None else ""
        if g in {"A", "B"}:
            return _parse_mc(gen_text)
        # Otherwise normalize booleans
        tail = _extract_after_final_answer(gen_text)
        s = tail if tail else gen_text
        return _norm_bool(s)

    # Most other tasks here are MC.
    return _parse_mc(gen_text)


def is_correct(task: str, pred: str, gold: str) -> bool:
    if gold is None:
        return False

    t = task.lower()
    g = str(gold).strip()
    p = str(pred).strip()

    if t == "gsm8k":
        # gold might be "#### 42" or include commas
        gnum = _parse_number(g)
        pnum = _parse_number(p)
        if not gnum or not pnum:
            return False
        try:
            return abs(float(gnum) - float(pnum)) < 1e-4
        except Exception:
            return False

    if t == "boolq":
        gu = g.upper()
        pu = p.upper()
        if gu in {"A", "B"}:
            return pu == gu
        return _norm_bool(p) == _norm_bool(g)

    return p.strip().lower() == g.strip().lower()


def _max_new_tokens_for_task(task: str, default: int, overrides: Dict[str, int]) -> int:
    t = task.lower()
    if t in overrides:
        return overrides[t]
    # sensible default bump for GSM8K if not overridden
    if t == "gsm8k" and default < 128:
        return 256
    return default


def eval_model_on_tasks(
    loaded: LoadedModel,
    device: str,
    tasks: List[str],
    n_eval: int,
    seed: int,
    max_new_tokens: int,
    per_task_max_new_tokens: Dict[str, int],
    debug_first: int = 0,
) -> Dict[str, float]:
    _, eval_by, _ = load_selected_tasks(
        tasks=tasks,
        n_subspace=1,
        n_eval=n_eval,
        seed=seed,
        template_randomization=True,
        template_seed=seed + 123,
        shuffle_choices=True,
        add_answer_prefix=True,
        answer_prefix="\nFinal answer:",
    )

    results: Dict[str, float] = {}
    for task in tasks:
        exs: List[Example] = eval_by.get(task, [])
        correct = 0
        total = 0

        # Debug print on the SAME examples we evaluate (first k in the list)
        dbg_k = min(debug_first, len(exs))
        for i in range(dbg_k):
            ex = exs[i]
            mnt = _max_new_tokens_for_task(task, max_new_tokens, per_task_max_new_tokens)
            out = generate_one(loaded, ex.prompt, device=device, max_new_tokens=mnt)
            pred = parse_prediction(task, out, gold=ex.gold)
            print("\n---", task, f"(debug {i+1}/{dbg_k})", "---")
            print("GOLD:", ex.gold)
            print("PRED:", pred)
            print("OUT :", out[:400])

        pbar = tqdm(exs, desc=f"[{task}] {loaded.kind}", leave=False)
        for ex in pbar:
            mnt = _max_new_tokens_for_task(task, max_new_tokens, per_task_max_new_tokens)
            out = generate_one(loaded, ex.prompt, device=device, max_new_tokens=mnt)
            pred = parse_prediction(task, out, gold=ex.gold)
            ok = is_correct(task, pred, ex.gold)
            correct += int(ok)
            total += 1
            pbar.set_postfix({"acc": f"{(correct/max(total,1)):.3f}"})

        results[task] = correct / max(total, 1)

    return results


def _parse_task_overrides(s: str) -> Dict[str, int]:
    """
    Parse overrides string like:
      "gsm8k=256,boolq=64"
    """
    out: Dict[str, int] = {}
    s = (s or "").strip()
    if not s:
        return out
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Bad --per_task_max_new_tokens item: '{part}' (expected task=NUM)")
        k, v = part.split("=", 1)
        k = k.strip().lower()
        v = v.strip()
        out[k] = int(v)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--models",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf,google/gemma-3-12b-it",
        help="Comma-separated HF model IDs to sanity-check.",
    )
    p.add_argument(
        "--tasks",
        type=str,
        default="gsm8k,boolq,piqa,arc_challenge,openbookqa,commonsenseqa",
        help="Comma-separated tasks (must exist in benchmark_dataloaders_ext.TASK_LOADERS).",
    )
    p.add_argument("--n_eval", type=int, default=8, help="Evaluation examples per task.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument(
        "--per_task_max_new_tokens",
        type=str,
        default="",
        help='Optional overrides like "gsm8k=256,boolq=64".',
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--dtype",
        type=str,
        default="fp32",
        choices=["fp16", "bf16", "fp32"],
        help="Model dtype. fp16/bf16 recommended on CUDA.",
    )
    p.add_argument(
        "--debug_first",
        type=int,
        default=0,
        help="Print debug for first K examples per task (0 disables).",
    )
    args = p.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    per_task_overrides = _parse_task_overrides(args.per_task_max_new_tokens)

    for mid in models:
        print("\n" + "=" * 80)
        print(f"Loading model: {mid}")
        loaded = load_model_any(mid, device=args.device, dtype=args.dtype)
        if not hasattr(loaded.model, "hf_device_map") and args.device:
            loaded.model.to(args.device)

        print(f"Model kind: {loaded.kind}")
        print(f"Running sanity evaluation on tasks: {tasks} (n_eval={args.n_eval})")

        res = eval_model_on_tasks(
            loaded=loaded,
            device=args.device,
            tasks=tasks,
            n_eval=args.n_eval,
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
            per_task_max_new_tokens=per_task_overrides,
            debug_first=args.debug_first,
        )

        print("Results (very rough sanity metrics):")
        for t, acc in res.items():
            print(f"  {t}: {acc:.3f}")


if __name__ == "__main__":
    main()
