# -*- coding: utf-8 -*-
"""
prove_sharedness_decode_full.py

Wrapper around prove_sharedness_decode_fair.py:
- Keeps the exact same CLI / main pipeline (decode last-token states -> pooled PCA -> sharedness -> nulls).
- Replaces ONLY the calibration prompt loader to cover a fuller benchmark suite:

Math (open-answer / CoT strong):
  - MATH (Hendrycks et al.)  (tries EleutherAI/hendrycks_math etc.)
  - AIME 2024 (HuggingFaceH4/aime_2024 etc.)
  - OlymMATH (RUC-AIBOX/OlymMATH)

Code (generation; no repo environment):
  - LiveCodeBench code_generation (+ lite fallback)
  - HumanEval (openai/openai_humaneval)
  - MBPP (google-research-datasets/mbpp / Muennighoff/mbpp)

Usage: identical to prove_sharedness_decode_fair.py
  python prove_sharedness_decode_full.py --model ... --layer ... --n_prompts ... (etc)

Note:
- If a dataset cannot be loaded (network / gated / not cached), it is skipped with a warning.
- If none of the datasets can be loaded, we raise like the original script.


Run example:
  CUDA_VISIBLE_DEVICES=0 python prove_sharedness_decode_full.py \
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
    --out_json results/full_benchmark/prove_existence.json \
    --out_txt  results/full_benchmark/prove_existence.txt

"""

from __future__ import annotations

from typing import Dict, List, Optional, Any
import re

import prove_sharedness_decode_fair as base


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
    # 强制“最终答案”形式，尽量减少输出长度分散
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
    # HumanEval 的 prompt 本身是 python 代码片段（含函数签名/docstring）
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

    # 0) 先加载你原来的 9 个（gsm8k / commonsenseqa / ...）
    #    这里不直接调用 base.load_calib_prompts()，避免它因“原始9个全加载失败”而提前 raise。
    #    我们把原始逻辑“复制一份”最简调用：复用 base 里已有 builder。
    # gsm8k
    ds = base._try_load_dataset("gsm8k", "main")
    if ds is not None:
        rows = _safe_sample(ds, n_prompts, seed + 1)
        prompts["gsm8k"] = [base.build_prompt_gsm8k(ex["question"]) for ex in rows]

    # commonsenseqa
    ds = base._try_load_dataset("commonsense_qa")
    if ds is not None:
        rows = _safe_sample(ds, n_prompts, seed + 11)
        prompts["commonsenseqa"] = [base.build_prompt_commonsenseqa(ex["question"], ex["choices"]) for ex in rows]

    # strategyqa
    ds = base._try_load_dataset("ChilleD/StrategyQA")
    if ds is not None:
        rows = _safe_sample(ds, n_prompts, seed + 21)
        prompts["strategyqa"] = [base.build_prompt_strategyqa(ex["question"]) for ex in rows]

    # aqua
    ds = base._try_load_dataset("aqua_rat")
    if ds is not None:
        rows = _safe_sample(ds, n_prompts, seed + 31)
        prompts["aqua"] = [base.build_prompt_aqua(ex["question"], ex["options"]) for ex in rows]

    # arc_challenge
    ds = base._try_load_dataset("ai2_arc", "ARC-Challenge")
    if ds is not None:
        rows = _safe_sample(ds, n_prompts, seed + 41)
        arc_prompts = []
        for ex in rows:
            q = ex.get("question", {})
            stem = q.get("stem", "") if isinstance(q, dict) else str(q)
            choices = q.get("choices", {}) if isinstance(q, dict) else {}
            labels = choices.get("label", [])
            texts = choices.get("text", [])
            if stem and labels and texts:
                arc_prompts.append(base.build_prompt_mc(stem, labels, texts))
        if arc_prompts:
            prompts["arc_challenge"] = arc_prompts

    # openbookqa
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

    # qasc
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

    # boolq
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

    # piqa
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

    # 1) MATH (Hendrycks et al.)
    # try a few common HF ids
    ds, used_path, used_name = _try_load_any([
        ("EleutherAI/hendrycks_math", None),
        ("hendrycks/competition_math", None),
        ("qwedsacf/competition_math", None),
        ("Maxwell-Jia/MATH", None),
    ])
    if ds is not None:
        rows = _safe_sample(ds, n_prompts, seed + 101)
        out = []
        for ex in rows:
            problem = _get_first(ex, ["problem", "question", "prompt"], default="")
            if problem:
                out.append(build_prompt_math_openanswer(problem))
        if out:
            prompts["math"] = out
        print(f"[Info] loaded MATH from {used_path}" + (f"/{used_name}" if used_name else ""))

    # 2) AIME (AIME 2024 / AIME 2022-2024 collections etc.)
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

    # 3) OlymMATH (olympiad-level)
    ds, used_path, used_name = _try_load_any([
        ("RUC-AIBOX/OlymMATH", None),
    ])
    if ds is not None:
        rows = _safe_sample(ds, n_prompts, seed + 121)
        out = []
        for ex in rows:
            problem = _get_first(ex, ["problem", "question", "prompt"], default="")
            if problem:
                out.append(build_prompt_math_openanswer(problem))
        if out:
            prompts["olymmath"] = out
        print(f"[Info] loaded OlymMATH from {used_path}" + (f"/{used_name}" if used_name else ""))

    # 4) LiveCodeBench (code_generation)
    # main + lite fallback
    ds, used_path, used_name = _try_load_any([
        ("livecodebench/code_generation", None),
        ("livecodebench/code_generation_lite", None),
        ("livecodebench/code_generation_lite", "v1"),  # some repos expose configs
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

    # 5) HumanEval
    ds, used_path, used_name = _try_load_any([
        ("openai/openai_humaneval", None),
        ("codeparrot/instructhumaneval", None),  # instruction-friendly variant
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

    # 6) MBPP
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
            starter = _get_first(ex, ["code", "starter_code"], default="")  # 有些版本叫 code
            if desc:
                out.append(build_prompt_mbpp(desc, starter))
        if out:
            prompts["mbpp"] = out
        print(f"[Info] loaded MBPP from {used_path}" + (f"/{used_name}" if used_name else ""))

    if len(prompts) == 0:
        raise RuntimeError("No datasets could be loaded; check HF datasets access / network / cache.")

    # log coverage
    print("[Data] Loaded tasks:")
    for k, v in prompts.items():
        print(f"  - {k}: {len(v)} prompts")

    return prompts


# -----------------------------
# Monkeypatch base loader + run base.main()
# -----------------------------

base.load_calib_prompts = load_calib_prompts_full

if __name__ == "__main__":
    # IMPORTANT: base.main() 里用 __name__=="__main__" 才 parse args
    # 这里把 base 模块的 __name__ 临时改成 "__main__"
    base.__dict__["__name__"] = "__main__"
    base.main()

