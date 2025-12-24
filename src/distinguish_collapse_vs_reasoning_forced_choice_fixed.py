
# -*- coding: utf-8 -*-
"""
distinguish_collapse_vs_reasoning_forced_choice_fixed.py

Fixes vs your original:
  1) 修复 forced-choice + staged hook 的 mask 维度不匹配崩溃（IndexError）。
     - 关键原因：staged hook 使用的 GenerationState.mask 与实际 forward 的 batch size 不一致时，
       直接 x[mask] 会报错。
     - 现在：如果 mask 维度不匹配，则退化为对整个 batch 施加干预（full removal），保证不崩溃。
  2) forced-choice 评分改为“支持多 token 候选”的严格 logprob 求和（仍然保持 seq_len==1 decode pass，
     以保证 hook 会触发）。
     - 这也避免了原来 all_single==False 时走 per-example(单条)路径导致 batch size 变成 1 的情况。
  3) choice prompt 加了 “只输出选项字母/Yes-No” 的指令，否则 forced-choice 很容易接近随机猜（≈20%）。
  4) generation 部分的 extraction 指标改为：是否能解析出非空 pred（而不是硬找 "Final answer:"）。
     - 你原来的 prompt 把 "Final answer:" 放在 prompt 末尾，continuation 里通常不会再出现该字符串，
       导致 extraction 永远是 0。

用法与原脚本一致，例如：
  python distinguish_collapse_vs_reasoning_forced_choice_fixed.py \
    --model meta-llama/Llama-2-7b-chat-hf --device cuda --dtype fp32 \
    --layer 10 --n_prompts 128 --eval_n 256 \
    --calib_max_new_tokens 128 --per_task_max_states 20000 \
    --pca_var 0.95 --tau 0.001 --m_shared all \
    --reasoning_tokens 128 --max_new_tokens 256
"""

import os
import re
import json
import math
import argparse
import random
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(os.path.join(THIS_DIR, ".."))

from joint_subspace_large.disturb_cross_task_all_shared import (  # noqa: E402
    get_model_layers,
    compute_cross_task_subspace,
    find_fully_shared_basis_improved,
)

# -----------------------------
# small utils
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

def json_dump_safe(obj: Any, path: str) -> None:
    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        return str(o)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_default)

# -----------------------------
# stats (lightweight)
# -----------------------------
def bootstrap_ci_mean(values: np.ndarray, iters: int, alpha: float, seed: int) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    obs = float(values.mean())
    boots = []
    for _ in range(iters):
        idx = rng.integers(0, n, size=n)
        boots.append(float(values[idx].mean()))
    lo = float(np.percentile(boots, 100 * (alpha / 2)))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return obs, lo, hi

def paired_bootstrap_ci_diff(baseline: np.ndarray, treat: np.ndarray, iters: int, alpha: float, seed: int):
    assert baseline.shape == treat.shape
    rng = np.random.default_rng(seed)
    diffs = treat - baseline
    obs = float(diffs.mean())
    n = len(diffs)
    boots = []
    for _ in range(iters):
        idx = rng.integers(0, n, size=n)
        boots.append(float(diffs[idx].mean()))
    lo = float(np.percentile(boots, 100 * (alpha / 2)))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return obs, lo, hi

def signflip_permutation_test(baseline: np.ndarray, treat: np.ndarray, iters: int, seed: int) -> float:
    assert baseline.shape == treat.shape
    rng = np.random.default_rng(seed)
    diffs = (treat - baseline).astype(np.float32)
    obs = float(diffs.mean())
    n = len(diffs)
    count = 0
    for _ in range(iters):
        signs = rng.choice([-1.0, 1.0], size=n).astype(np.float32)
        stat = float((diffs * signs).mean())
        if abs(stat) >= abs(obs):
            count += 1
    return float((count + 1) / (iters + 1))

def summarize_paired(b: np.ndarray, t: np.ndarray, iters_boot: int, iters_perm: int, alpha: float, seed: int):
    md, lo, hi = paired_bootstrap_ci_diff(b, t, iters=iters_boot, alpha=alpha, seed=seed + 7)
    p = signflip_permutation_test(b, t, iters=iters_perm, seed=seed + 19)
    return {"mean_diff": md, "ci_low": lo, "ci_high": hi, "p_value": p}

def fmt_ci(x, lo, hi) -> str:
    return f"{x*100:.1f} [{lo*100:.1f}, {hi*100:.1f}]"

# -----------------------------
# prompts / examples
# -----------------------------
TASKS = ["gsm8k", "commonsenseqa", "strategyqa", "aqua"]

@dataclass
class Example:
    dataset: str
    ex_id: str
    prompt_cot: str
    prompt_choice: str
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

def build_prompt_gsm8k_cot(q: str) -> str:
    return f"Question: {q}\nLet's think step by step.\nFinal answer:"
def build_prompt_gsm8k_choice(q: str) -> str:
    # 这里 gsm8k 不做 forced-choice（脚本里也不会跑），保留。
    return f"Question: {q}\nAnswer:"

def build_prompt_csqa_cot(q: str, choices: Dict[str, List[str]]) -> str:
    labels = choices["label"]; texts = choices["text"]
    lines = [f"{a}) {b}" for a, b in zip(labels, texts)]
    return f"Question: {q}\nChoices:\n" + "\n".join(lines) + "\nReason step by step.\nFinal answer:"
def build_prompt_csqa_choice(q: str, choices: Dict[str, List[str]]) -> str:
    # 关键：明确要求只输出字母，否则 forced-choice 往往接近随机。
    labels = choices["label"]; texts = choices["text"]
    lines = [f"{a}) {b}" for a, b in zip(labels, texts)]
    return (
        f"Question: {q}\nChoices:\n" + "\n".join(lines)
        + "\nAnswer with only the letter (A, B, C, D, or E).\nAnswer:"
    )

def build_prompt_stqa_cot(q: str) -> str:
    return f"Question: {q}\nPlease reason step by step.\nFinal answer:"
def build_prompt_stqa_choice(q: str) -> str:
    # 关键：明确 Yes/No only
    return f"Question: {q}\nAnswer with only Yes or No.\nAnswer:"

def build_prompt_aqua_cot(q: str, opts: List[str]) -> str:
    labels = ["A","B","C","D","E"]
    lines = []
    for i, opt in enumerate(opts[:5]):
        clean = re.sub(r"^[A-E]\)?\s*[:\-]?\s*", "", str(opt).strip(), flags=re.I)
        lines.append(f"{labels[i]}) {clean}")
    return f"Question: {q}\nChoices:\n" + "\n".join(lines) + "\nPlease reason step by step.\nFinal answer:"
def build_prompt_aqua_choice(q: str, opts: List[str]) -> str:
    labels = ["A","B","C","D","E"]
    lines = []
    for i, opt in enumerate(opts[:5]):
        clean = re.sub(r"^[A-E]\)?\s*[:\-]?\s*", "", str(opt).strip(), flags=re.I)
        lines.append(f"{labels[i]}) {clean}")
    return (
        f"Question: {q}\nChoices:\n" + "\n".join(lines)
        + "\nAnswer with only the letter (A, B, C, D, or E).\nAnswer:"
    )

def sample_hf_split(ds_split, n: int, seed: int):
    n = min(n, len(ds_split))
    return ds_split.shuffle(seed=seed).select(range(n))

def load_examples(n_prompts: int, n_eval: int, seed: int) -> Tuple[Dict[str, List[Example]], Dict[str, List[Example]]]:
    sub_by, eval_by = {}, {}

    ds = load_dataset("gsm8k", "main")
    sub_rows = sample_hf_split(ds["train"], n_prompts, seed + 1)
    ev_rows = sample_hf_split(ds["test"], n_eval, seed + 2)
    sub_by["gsm8k"] = [Example("gsm8k", f"gsm8k-sub-{i}", build_prompt_gsm8k_cot(ex["question"]),
                              build_prompt_gsm8k_choice(ex["question"]), parse_gsm8k_gold(ex["answer"])) for i, ex in enumerate(sub_rows)]
    eval_by["gsm8k"] = [Example("gsm8k", f"gsm8k-ev-{i}", build_prompt_gsm8k_cot(ex["question"]),
                               build_prompt_gsm8k_choice(ex["question"]), parse_gsm8k_gold(ex["answer"])) for i, ex in enumerate(ev_rows)]

    ds = load_dataset("commonsense_qa")
    sub_rows = sample_hf_split(ds["train"], n_prompts, seed + 11)
    ev_rows = sample_hf_split(ds["validation"], n_eval, seed + 12)
    sub_by["commonsenseqa"] = [Example("commonsenseqa", f"csqa-sub-{i}",
                                      build_prompt_csqa_cot(ex["question"], ex["choices"]),
                                      build_prompt_csqa_choice(ex["question"], ex["choices"]),
                                      safe_upper(ex["answerKey"])) for i, ex in enumerate(sub_rows)]
    eval_by["commonsenseqa"] = [Example("commonsenseqa", f"csqa-ev-{i}",
                                       build_prompt_csqa_cot(ex["question"], ex["choices"]),
                                       build_prompt_csqa_choice(ex["question"], ex["choices"]),
                                       safe_upper(ex["answerKey"])) for i, ex in enumerate(ev_rows)]

    ds = load_dataset("ChilleD/StrategyQA")
    sub_rows = sample_hf_split(ds["train"], n_prompts, seed + 21)
    ev_rows = sample_hf_split(ds["test"], n_eval, seed + 22)

    def to_yesno(v: Any) -> str:
        if isinstance(v, bool):
            return "YES" if v else "NO"
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

    sub_by["strategyqa"] = [Example("strategyqa", f"stqa-sub-{i}",
                                   build_prompt_stqa_cot(ex["question"]),
                                   build_prompt_stqa_choice(ex["question"]),
                                   to_yesno(ex["answer"])) for i, ex in enumerate(sub_rows)]
    eval_by["strategyqa"] = [Example("strategyqa", f"stqa-ev-{i}",
                                    build_prompt_stqa_cot(ex["question"]),
                                    build_prompt_stqa_choice(ex["question"]),
                                    to_yesno(ex["answer"])) for i, ex in enumerate(ev_rows)]

    ds = load_dataset("aqua_rat")
    sub_rows = sample_hf_split(ds["train"], n_prompts, seed + 31)
    ev_rows = sample_hf_split(ds["test"], n_eval, seed + 32)

    def aqua_gold(ex: dict) -> str:
        if "correct" in ex:
            return safe_upper(ex["correct"])
        if "answer" in ex:
            return safe_upper(ex["answer"])
        return ""

    sub_by["aqua"] = [Example("aqua", f"aqua-sub-{i}",
                             build_prompt_aqua_cot(ex["question"], ex["options"]),
                             build_prompt_aqua_choice(ex["question"], ex["options"]),
                             aqua_gold(ex)) for i, ex in enumerate(sub_rows)]
    eval_by["aqua"] = [Example("aqua", f"aqua-ev-{i}",
                              build_prompt_aqua_cot(ex["question"], ex["options"]),
                              build_prompt_aqua_choice(ex["question"], ex["options"]),
                              aqua_gold(ex)) for i, ex in enumerate(ev_rows)]
    return sub_by, eval_by

# -----------------------------
# hooks (same as your last-token decode-pass intervention)
# -----------------------------
def orthonormalize_basis_np(B: np.ndarray) -> np.ndarray:
    Q, _ = np.linalg.qr(B.astype(np.float64, copy=False))
    return Q.astype(np.float32, copy=False)

class GenerationState:
    def __init__(self, batch_size: int, device: torch.device, reasoning_threshold: int):
        self.unfinished = torch.ones(batch_size, dtype=torch.bool, device=device)
        self.gen_steps = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.reasoning_threshold = int(reasoning_threshold)
    def current_reasoning_mask(self) -> torch.Tensor:
        return self.unfinished & (self.gen_steps < self.reasoning_threshold)
    def step_update(self, next_tokens: torch.Tensor, eos_id: int) -> None:
        t = next_tokens.squeeze(-1)
        active = self.unfinished.clone()
        self.gen_steps[active] += 1
        self.unfinished[active & (t == eos_id)] = False

class HookStats:
    def __init__(self, name: str):
        self.name = name
        self.decode_calls = 0
        self.intervened = 0
    def report(self) -> str:
        return f"{self.name} decode_calls={self.decode_calls} intervened={self.intervened}"

class LastTokenRemovalHook:
    def __init__(self, basis_np: np.ndarray, alpha: float, stats: HookStats):
        self.alpha = float(alpha)
        self.stats = stats
        Q = orthonormalize_basis_np(basis_np)
        self.Q_cpu = torch.tensor(Q, dtype=torch.float32)
        self.Q_dev = None
    def _Q(self, device):
        if self.Q_dev is None or self.Q_dev.device != device:
            self.Q_dev = self.Q_cpu.to(device=device)
        return self.Q_dev
    def __call__(self, module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        if not isinstance(hs, torch.Tensor) or hs.ndim != 3 or hs.shape[1] != 1:
            return output
        B = hs.shape[0]
        self.stats.decode_calls += B
        self.stats.intervened += B
        hs2 = hs.clone()
        x = hs2[:, -1, :].float()
        Q = self._Q(hs2.device)
        proj = (x @ Q) @ Q.T
        hs2[:, -1, :] = (x - self.alpha * proj).to(dtype=hs2.dtype)
        if isinstance(output, tuple):
            return (hs2,) + output[1:]
        return hs2

class LastTokenStagedRemovalHook(LastTokenRemovalHook):
    """
    修复点：mask 维度不匹配时不再 x[mask]，而是退化为对整个 batch 施加干预。
    同时避免 state=None 时调用 super() 导致 stats 双计数。
    """
    def __init__(self, basis_np: np.ndarray, alpha: float, reasoning_tokens: int, stats: HookStats):
        super().__init__(basis_np, alpha, stats)
        self.state = None
        self.reasoning_tokens = int(reasoning_tokens)
    def set_state(self, st):
        self.state = st

    def __call__(self, module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        if not isinstance(hs, torch.Tensor) or hs.ndim != 3 or hs.shape[1] != 1:
            return output

        B = hs.shape[0]
        self.stats.decode_calls += B

        # get mask if possible; otherwise intervene on all samples (safe fallback)
        mask: Optional[torch.Tensor] = None
        if self.state is not None:
            try:
                m = self.state.current_reasoning_mask()
                if isinstance(m, torch.Tensor) and m.ndim == 1 and m.numel() == B:
                    mask = m.to(device=hs.device, dtype=torch.bool)
            except Exception:
                mask = None

        if mask is None:
            mask = torch.ones(B, device=hs.device, dtype=torch.bool)

        n_on = int(mask.sum().item())
        self.stats.intervened += n_on
        if n_on == 0:
            return output

        hs2 = hs.clone()
        x = hs2[:, -1, :].float()
        Q = self._Q(hs2.device)

        xs = x[mask]
        proj = (xs @ Q) @ Q.T
        x[mask] = xs - self.alpha * proj
        hs2[:, -1, :] = x.to(dtype=hs2.dtype)

        if isinstance(output, tuple):
            return (hs2,) + output[1:]
        return hs2

def register_hooks(model, layer_indices, basis_np, alpha, staged, reasoning_tokens, name):
    layers, _ = get_model_layers(model)
    stats = HookStats(name)
    handles = []
    staged_hooks = []
    if basis_np is None:
        return [], None, stats
    for li in layer_indices:
        if staged:
            hk = LastTokenStagedRemovalHook(basis_np, alpha, reasoning_tokens, stats)
            staged_hooks.append(hk)
        else:
            hk = LastTokenRemovalHook(basis_np, alpha, stats)
        handles.append(layers[li].register_forward_hook(hk))
    def setter(st):
        for hk in staged_hooks:
            hk.set_state(st)
    return handles, (setter if staged else None), stats

def remove_hooks(handles):
    for h in handles:
        try: h.remove()
        except Exception: pass

# -----------------------------
# A3 decode-aligned basis estimation (reuse decode states)
# -----------------------------
class DecodeLastTokenActivationCollector:
    def __init__(self, layer_indices):
        self.layer_indices = list(layer_indices)
        self._cur_task = None
        self.capture_enabled = False
        self.active_mask = None
        self.storage = defaultdict(lambda: defaultdict(list))
    def set_current_task(self, t): self._cur_task = t
    def set_capture(self, enabled, active_mask=None):
        self.capture_enabled = bool(enabled); self.active_mask = active_mask
    def make_hook(self, layer_idx):
        def _hook(module, inputs, output):
            if (not self.capture_enabled) or (self._cur_task is None):
                return output
            hs = output[0] if isinstance(output, tuple) else output
            if not isinstance(hs, torch.Tensor) or hs.ndim != 3 or hs.shape[1] != 1:
                return output
            x = hs[:, -1, :]
            if self.active_mask is not None and self.active_mask.numel() == x.shape[0]:
                x = x[self.active_mask.bool()]
            if x.numel() == 0:
                return output
            self.storage[self._cur_task][layer_idx].append(x.detach().float().cpu().numpy())
            return output
        return _hook
    def get_task_activations(self, task, layer_idx):
        chunks = self.storage.get(task, {}).get(layer_idx, [])
        if not chunks: return None
        return np.concatenate(chunks, axis=0)

def top_p_filtering(logits, top_p):
    if top_p <= 0.0 or top_p >= 1.0: return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumprobs = torch.cumsum(probs, dim=-1)
    mask = cumprobs > top_p
    mask[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
    filtered = torch.full_like(logits, float("-inf"))
    filtered.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)
    return filtered

def top_k_filtering(logits, top_k):
    if top_k is None or top_k <= 0: return logits
    top_k = min(top_k, logits.size(-1))
    values, _ = torch.topk(logits, top_k, dim=-1)
    min_values = values[:, -1].unsqueeze(-1)
    return torch.where(logits < min_values, torch.full_like(logits, float("-inf")), logits)

@torch.no_grad()
def collect_decode_states(model, tok, prompts, collector, batch_size, max_new_tokens, temperature, top_p, top_k, max_prompt_len):
    device = next(model.parameters()).device
    eos = tok.eos_token_id
    for i in tqdm(range(0, len(prompts), batch_size), desc="CollectDecode"):
        batch = prompts[i:i+batch_size]
        inputs = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_prompt_len).to(device)
        ids = inputs["input_ids"]; attn = inputs["attention_mask"]
        B = ids.shape[0]
        unfinished = torch.ones(B, dtype=torch.bool, device=device)
        collector.set_capture(False, None)
        out = model(input_ids=ids, attention_mask=attn, use_cache=True)
        logits = out.logits[:, -1, :]
        past = out.past_key_values
        for _ in range(max_new_tokens):
            next_tok = torch.argmax(logits, dim=-1, keepdim=True)
            next_tok = torch.where(unfinished.unsqueeze(-1), next_tok, torch.full_like(next_tok, eos))
            unfinished = unfinished & (next_tok.squeeze(-1) != eos)
            if not bool(unfinished.any().item()): break
            attn = torch.cat([attn, torch.ones((B, 1), device=device, dtype=attn.dtype)], dim=1)
            collector.set_capture(True, unfinished)
            out = model(input_ids=next_tok, attention_mask=attn, use_cache=True, past_key_values=past)
            logits = out.logits[:, -1, :]
            past = out.past_key_values
        collector.set_capture(False, None)

def balance_states(states_by_task: Dict[str, np.ndarray], seed: int) -> Dict[str, np.ndarray]:
    nmin = min(v.shape[0] for v in states_by_task.values())
    rng = np.random.default_rng(seed)
    out = {}
    for t, x in states_by_task.items():
        if x.shape[0] == nmin:
            out[t] = x
        else:
            idx = rng.choice(x.shape[0], size=nmin, replace=False)
            out[t] = x[idx]
    print(f"[Fair] balanced states per task = {nmin}")
    return out

def compute_shared_basis_decode_aligned(model, tok, sub_by, layer_indices, n_prompts, calib_max_new_tokens,
                                        per_task_max_states, max_prompt_len, pca_var, tau, m_shared, seed):
    layers, _ = get_model_layers(model)
    collector = DecodeLastTokenActivationCollector(layer_indices)
    handles = []
    for li in layer_indices:
        handles.append(layers[li].register_forward_hook(collector.make_hook(li)))
    try:
        for t in TASKS:
            collector.set_current_task(t)
            prompts = [ex.prompt_cot for ex in sub_by[t][:n_prompts]]
            collect_decode_states(model, tok, prompts, collector, batch_size=4,
                                 max_new_tokens=calib_max_new_tokens, temperature=0.7, top_p=0.9, top_k=0,
                                 max_prompt_len=max_prompt_len)
    finally:
        for h in handles:
            try: h.remove()
            except Exception: pass

    layer = layer_indices[0]
    states = {}
    for t in TASKS:
        x = collector.get_task_activations(t, layer)
        if x is None or x.shape[0] == 0:
            raise RuntimeError(f"No states for task={t}")
        if x.shape[0] > per_task_max_states:
            rng = np.random.default_rng(seed + stable_int_seed(t, layer))
            idx = rng.choice(x.shape[0], size=per_task_max_states, replace=False)
            x = x[idx]
        states[t] = x.astype(np.float32, copy=False)
        print(f"[Collect] task={t} states={states[t].shape[0]} x {states[t].shape[1]}")

    states = balance_states(states, seed=seed + 999)
    task_acts = {t: {layer: states[t]} for t in TASKS}
    joint_subspace, cross_dim, contrib, full_pca_info = compute_cross_task_subspace(
        task_acts, variance_threshold=pca_var, min_dim=1, max_dim=4096, return_full_pca=True
    )
    if joint_subspace is None or cross_dim <= 0:
        raise RuntimeError("PCA failed")

    m_req = len(TASKS) if (m_shared == "all") else int(m_shared)
    shared_indices = find_fully_shared_basis_improved(
        contrib, TASKS, cross_dim, min_tasks_shared=m_req, relative_threshold=tau, top_k_components=cross_dim
    )
    if not shared_indices:
        shared_indices = find_fully_shared_basis_improved(
            contrib, TASKS, cross_dim, min_tasks_shared=2, relative_threshold=tau, top_k_components=cross_dim
        )
    if not shared_indices:
        raise RuntimeError("shared_indices empty")
    return joint_subspace.astype(np.float32, copy=False), shared_indices, states, {"cross_dim": int(cross_dim)}

# -----------------------------
# Generation eval (collapse diagnosis)
# -----------------------------
def extract_final_answer(text: str) -> str:
    m = re.search(r"Final answer\s*:\s*(.*)", text, flags=re.I)
    if not m: return ""
    s = (m.group(1) or "").strip().splitlines()[0].strip()
    return s

def parse_pred(dataset: str, cont: str) -> str:
    s = extract_final_answer(cont)
    if dataset == "gsm8k":
        n = normalize_number_str(s)
        if n is not None: return n
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", cont.replace(",", ""))
        return normalize_number_str(nums[-1]) if nums else ""
    if dataset in ["commonsenseqa", "aqua"]:
        m = re.search(r"\b([A-E])\b", s.upper())
        if m: return m.group(1)
        m2 = re.search(r"\b([A-E])\b", cont.upper())
        return m2.group(1) if m2 else ""
    if dataset == "strategyqa":
        t = s.lower()
        if "yes" in t: return "YES"
        if "no" in t: return "NO"
        t2 = cont.lower()
        if "yes" in t2 and "no" in t2:
            return "YES" if t2.find("yes") < t2.find("no") else "NO"
        if "yes" in t2: return "YES"
        if "no" in t2: return "NO"
        return ""
    return ""

def is_correct(dataset: str, pred: str, gold: str) -> int:
    if dataset == "gsm8k":
        return int(pred != "" and gold != "" and pred == gold)
    return int(pred != "" and pred.upper() == gold.upper())

@torch.no_grad()
def generate_with_hooks(model, tok, prompts, layer_indices, basis_np, alpha, staged, reasoning_tokens,
                        max_new_tokens, decoding, temperature, top_p, top_k, batch_size, max_prompt_len, seed):
    device = next(model.parameters()).device
    eos = tok.eos_token_id
    if decoding == "sample":
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    handles, setter, stats = register_hooks(
        model, layer_indices, basis_np, alpha, staged, reasoning_tokens,
        name=f"{'staged' if staged else 'full'}@{layer_indices[0]}"
    )
    continuations = []
    total_new = 0
    eos_hits = 0
    try:
        for i in tqdm(range(0, len(prompts), batch_size), desc=f"Generate({decoding})"):
            batch = prompts[i:i+batch_size]
            inputs = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_prompt_len).to(device)
            ids = inputs["input_ids"]; attn = inputs["attention_mask"]
            B, T0 = ids.shape
            st = GenerationState(B, ids.device, reasoning_tokens) if setter else None
            if setter: setter(st)

            out = model(input_ids=ids, attention_mask=attn, use_cache=True)
            logits = out.logits[:, -1, :]
            past = out.past_key_values

            generated = ids
            unfinished = torch.ones(B, dtype=torch.bool, device=device)

            for _ in range(max_new_tokens):
                if decoding == "greedy":
                    next_tok = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    lt = logits / max(temperature, 1e-6)
                    lt = top_k_filtering(lt, top_k)
                    lt = top_p_filtering(lt, top_p)
                    probs = torch.softmax(lt, dim=-1)
                    next_tok = torch.multinomial(probs, num_samples=1)

                next_tok = torch.where(unfinished.unsqueeze(-1), next_tok, torch.full_like(next_tok, eos))
                generated = torch.cat([generated, next_tok], dim=1)

                unfinished = unfinished & (next_tok.squeeze(-1) != eos)
                if st is not None:
                    st.step_update(next_tok, eos_id=eos)
                if not bool(unfinished.any().item()):
                    break

                attn = torch.cat([attn, torch.ones((B, 1), device=device, dtype=attn.dtype)], dim=1)
                out = model(input_ids=next_tok, attention_mask=attn, use_cache=True, past_key_values=past)
                logits = out.logits[:, -1, :]
                past = out.past_key_values

            if setter: setter(None)

            cont_ids = generated[:, T0:]
            total_new += int(cont_ids.shape[1] * B)
            eos_hits += int((~unfinished).sum().item())

            for b in range(B):
                continuations.append(tok.decode(cont_ids[b], skip_special_tokens=True))

        return {
            "continuations": continuations,
            "avg_new_tok": float(total_new / max(1, len(prompts))),
            "eos_rate": float(eos_hits / max(1, len(prompts))),
            "hook_stats": stats.report(),
        }
    finally:
        remove_hooks(handles)

# -----------------------------
# Forced-choice logprob eval (confound-free)
# -----------------------------
def candidate_strings(task: str) -> List[str]:
    if task in ["commonsenseqa", "aqua"]:
        return ["A","B","C","D","E"]
    if task == "strategyqa":
        return ["Yes","No"]
    return []

def cand_token_ids(tok, s: str) -> List[int]:
    # 保持与你原版一致：默认假设 prompt "Answer:" 后面紧跟一个空格再输出。
    ids = tok.encode(" " + s, add_special_tokens=False)
    if not ids:
        ids = tok.encode(s, add_special_tokens=False)
    return ids

def repeat_past_key_values(past, repeats: int):
    """
    Repeat KV cache along batch dim so we can score C candidates in parallel.

    Works for:
      - New HF Cache objects (DynamicCache, EncoderDecoderCache, etc.)
      - Legacy tuple-of-tuples past_key_values
    """
    if past is None:
        return None

    # NEW: HF Cache API (DynamicCache is default in recent transformers)
    if hasattr(past, "batch_repeat_interleave"):
        # Note: this mutates in-place. That's fine here because past_prompt is batch-local.
        past.batch_repeat_interleave(repeats)
        return past

    # Legacy tuple/list format
    if isinstance(past, (tuple, list)):
        rep = []
        for layer_past in past:
            if layer_past is None:
                rep.append(None)
                continue
            if isinstance(layer_past, (tuple, list)):
                items = []
                for x in layer_past:
                    if torch.is_tensor(x) and x.ndim >= 1:
                        items.append(x.repeat_interleave(repeats, dim=0))
                    else:
                        items.append(x)
                rep.append(tuple(items))
            else:
                rep.append(layer_past)
        return tuple(rep)

    # Fallback: unknown cache type
    return past


@torch.no_grad()
def forced_choice_logprob_eval(model, tok, examples: List[Example], task: str,
                               layer_indices, basis_np, alpha, staged, reasoning_tokens,
                               batch_size, max_prompt_len, bootstrap_iters, perm_iters, ci_alpha, seed):
    cands = candidate_strings(task)
    if not cands:
        return {"skipped": True}

    cand_ids: List[List[int]] = [cand_token_ids(tok, c) for c in cands]
    if any(len(x) == 0 for x in cand_ids):
        raise RuntimeError(f"Empty candidate tokenization for task={task}, cands={cands}")

    C = len(cands)
    max_cand_len = max(len(x) for x in cand_ids)

    handles, setter, stats = register_hooks(
        model, layer_indices, basis_np, alpha, staged, reasoning_tokens,
        name=f"fc_{'staged' if staged else 'full'}@{layer_indices[0]}"
    )

    device = next(model.parameters()).device
    eos = tok.eos_token_id
    correct: List[float] = []

    try:
        for i in tqdm(range(0, len(examples), batch_size), desc=f"ForcedChoice({task})"):
            batch = examples[i:i+batch_size]
            prompts = [ex.prompt_choice for ex in batch]
            golds = [ex.gold.upper() for ex in batch]

            inp = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_prompt_len).to(device)
            ids = inp["input_ids"]; attn = inp["attention_mask"]
            B, T = ids.shape

            # 1) prompt prefill split: ensure first-cand logits comes from seq_len==1 pass
            if setter:
                st_prompt = GenerationState(B, ids.device, reasoning_tokens)
                setter(st_prompt)

            if T <= 1:
                # Extremely short prompt edge-case: fallback (no guarantee hook triggers)
                out_full = model(input_ids=ids, attention_mask=attn, use_cache=True)
                logits0 = out_full.logits[:, -1, :]
                past_prompt = out_full.past_key_values
            else:
                out1 = model(input_ids=ids[:, :-1], attention_mask=attn[:, :-1], use_cache=True)
                past1 = out1.past_key_values
                out2 = model(input_ids=ids[:, -1:], attention_mask=attn, use_cache=True, past_key_values=past1)
                logits0 = out2.logits[:, -1, :]
                past_prompt = out2.past_key_values

            logp0 = torch.log_softmax(logits0, dim=-1)  # [B, V]
            scores = torch.zeros((B, C), device=device, dtype=torch.float32)

            # score token 0 for each cand
            for ci in range(C):
                scores[:, ci] += logp0[:, cand_ids[ci][0]].float()

            # 2) score remaining tokens (if any) with decode passes (seq_len==1), in parallel over candidates
            if max_cand_len > 1:
                # Expand batch: for each prompt b, create C copies for candidates (order: [b0c0..b0cC-1, b1c0..])
                Bc = B * C
                if setter:
                    st_cand = GenerationState(Bc, ids.device, reasoning_tokens)
                    setter(st_cand)

                # repeat past & attn for candidates
                past_c = repeat_past_key_values(past_prompt, repeats=C)
                attn_c = attn.repeat_interleave(C, dim=0)

                # index helper: map (b,ci) -> flat index
                idx_bc = torch.arange(Bc, device=device).view(B, C)

                # For step=1..max_len-1: feed previous token to obtain logits for current token
                for step in range(1, max_cand_len):
                    # build prev-token input ids for each expanded sequence
                    prev_tokens = []
                    for ci in range(C):
                        if step - 1 < len(cand_ids[ci]):
                            prev_tokens.append(cand_ids[ci][step - 1])
                        else:
                            prev_tokens.append(eos)
                    prev_tokens = torch.tensor(prev_tokens, device=device, dtype=torch.long)  # [C]
                    prev_tokens = prev_tokens.repeat(B).view(Bc, 1)  # [B*C, 1]

                    attn_c = torch.cat([attn_c, torch.ones((Bc, 1), device=device, dtype=attn_c.dtype)], dim=1)

                    out = model(input_ids=prev_tokens, attention_mask=attn_c, use_cache=True, past_key_values=past_c)
                    logits = out.logits[:, -1, :]
                    logp = torch.log_softmax(logits, dim=-1)

                    # update scores for candidates that have this token
                    for ci in range(C):
                        if step < len(cand_ids[ci]):
                            tok_id = cand_ids[ci][step]
                            idx = idx_bc[:, ci]  # [B]
                            scores[:, ci] += logp[idx, tok_id].float()

                    past_c = out.past_key_values
                    # advance staged counter if needed (not crucial, but keeps semantics sane)
                    if setter:
                        st_cand.step_update(prev_tokens, eos_id=eos)

            # done with hooks state for this batch
            if setter:
                setter(None)

            pred_idx = torch.argmax(scores, dim=-1).detach().cpu().numpy().tolist()
            for bi, pi in enumerate(pred_idx):
                pred = cands[pi].upper()
                correct.append(1.0 if pred == golds[bi] else 0.0)

        corr = np.array(correct, dtype=np.float32)
        acc, lo, hi = bootstrap_ci_mean(corr, iters=bootstrap_iters, alpha=ci_alpha, seed=stable_int_seed(seed, task, "fc"))
        return {
            "accuracy": float(acc), "ci_low": float(lo), "ci_high": float(hi),
            "correct": corr.tolist(),
            "hook_stats": stats.report(),
            "cand_token_lens": [len(x) for x in cand_ids],
        }
    finally:
        remove_hooks(handles)

# -----------------------------
# Main
# -----------------------------
def load_model_tokenizer(model_name: str, dtype: torch.dtype, device: str):
    # transformers 版本差异：有的用 dtype，有的用 torch_dtype。两者都尝试。
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model = model.to(device)
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, default="fp32", choices=["fp32","fp16"])
    ap.add_argument("--layer", type=int, default=10)

    ap.add_argument("--n_prompts", type=int, default=128)
    ap.add_argument("--eval_n", type=int, default=256)
    ap.add_argument("--calib_max_new_tokens", type=int, default=128)
    ap.add_argument("--per_task_max_states", type=int, default=20000)

    ap.add_argument("--pca_var", type=float, default=0.95)
    ap.add_argument("--tau", type=float, default=0.001)
    ap.add_argument("--m_shared", type=str, default="all")

    ap.add_argument("--reasoning_tokens", type=int, default=128)
    ap.add_argument("--max_new_tokens", type=int, default=256)

    ap.add_argument("--bootstrap_iters", type=int, default=5000)
    ap.add_argument("--perm_iters", type=int, default=10000)
    ap.add_argument("--ci_alpha", type=float, default=0.05)

    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_prompt_len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--out_json", type=str, default=os.path.join(THIS_DIR, "collapse_vs_reasoning.json"))
    ap.add_argument("--out_txt", type=str, default=os.path.join(THIS_DIR, "collapse_vs_reasoning.txt"))
    args = ap.parse_args()

    set_global_seed(args.seed)
    dtype = torch.float32 if args.dtype == "fp32" else torch.float16

    model, tok = load_model_tokenizer(args.model, dtype=dtype, device=args.device)
    model.eval()
    model.config.use_cache = True

    layer_indices = [args.layer]
    print(f"[Env] model={args.model} device={args.device} dtype={args.dtype} layer={layer_indices}")

    sub_by, eval_by = load_examples(args.n_prompts, args.eval_n, args.seed)

    # compute decode-aligned shared basis
    joint_subspace, shared_indices, states_by_task, extra = compute_shared_basis_decode_aligned(
        model, tok, sub_by, layer_indices,
        n_prompts=args.n_prompts,
        calib_max_new_tokens=args.calib_max_new_tokens,
        per_task_max_states=args.per_task_max_states,
        max_prompt_len=args.max_prompt_len,
        pca_var=args.pca_var,
        tau=args.tau,
        m_shared=args.m_shared,
        seed=args.seed
    )
    shared_basis = joint_subspace[:, shared_indices]
    # simple random control (same dim)
    D = shared_basis.shape[0]; k = shared_basis.shape[1]
    rng = np.random.default_rng(args.seed + 999)
    A = rng.standard_normal(size=(D, k)).astype(np.float32)
    Qr, _ = np.linalg.qr(A)
    rand_basis = Qr.astype(np.float32, copy=False)

    results = {
        "config": {
            "model": args.model, "dtype": args.dtype, "device": args.device,
            "layer_indices": layer_indices,
            "cross_dim": extra["cross_dim"],
            "shared_k": int(k),
            "tau": args.tau, "m_shared": args.m_shared,
            "reasoning_tokens": args.reasoning_tokens,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
        },
        "by_task": {}
    }

    CONDITIONS = [
        ("baseline", None, 0.0, False),
        ("shared_full", shared_basis, 1.0, False),
        ("shared_staged", shared_basis, 1.0, True),
        ("rand_full", rand_basis, 1.0, False),
        ("rand_staged", rand_basis, 1.0, True),
    ]

    for task in TASKS:
        exs = eval_by[task]
        print("\n" + "-"*80)
        print(f"[Task] {task} n={len(exs)}")
        print("-"*80)

        block = {"generation": {}, "forced_choice": {}, "paired": {}}

        # A) generation/extraction (diagnose collapse)
        prompts = [ex.prompt_cot for ex in exs]
        for name, basis, alpha, staged in CONDITIONS:
            print(f"[Gen] {name}")
            out = generate_with_hooks(
                model, tok, prompts,
                layer_indices=layer_indices,
                basis_np=basis, alpha=alpha,
                staged=staged, reasoning_tokens=args.reasoning_tokens,
                max_new_tokens=args.max_new_tokens,
                decoding="greedy",
                temperature=0.7, top_p=0.9, top_k=0,
                batch_size=args.batch_size, max_prompt_len=args.max_prompt_len,
                seed=args.seed + stable_int_seed(task, name)
            )
            preds = [parse_pred(task, c) for c in out["continuations"]]
            correct = np.array([is_correct(task, p, ex.gold) for p, ex in zip(preds, exs)], dtype=np.float32)

            # FIX: extraction success should be "能否解析出答案"，而不是 continuation 中是否包含 Final answer:
            extr = np.array([1.0 if p != "" else 0.0 for p in preds], dtype=np.float32)

            acc, lo, hi = bootstrap_ci_mean(correct, args.bootstrap_iters, args.ci_alpha, stable_int_seed(args.seed, task, name, "gen"))
            extr_m, extr_lo, extr_hi = bootstrap_ci_mean(extr, args.bootstrap_iters, args.ci_alpha, stable_int_seed(args.seed, task, name, "extr"))
            print(f"  acc={fmt_ci(acc, lo, hi)} extr={fmt_ci(extr_m, extr_lo, extr_hi)} eos={out['eos_rate']:.3f} avg_new={out['avg_new_tok']:.1f} {out['hook_stats']}")
            block["generation"][name] = {
                "accuracy": float(acc), "ci_low": float(lo), "ci_high": float(hi),
                "extraction": float(extr_m), "ex_ci_low": float(extr_lo), "ex_ci_high": float(extr_hi),
                "eos_rate": out["eos_rate"], "avg_new_tok": out["avg_new_tok"],
                "correct": correct.tolist(),
                "extract_ok": extr.tolist(),
                "hook_stats": out["hook_stats"],
            }

        # B) forced-choice (confound-free) for applicable tasks
        if task in ["commonsenseqa", "strategyqa", "aqua"]:
            for name, basis, alpha, staged in CONDITIONS:
                print(f"[FC] {name}")
                fc = forced_choice_logprob_eval(
                    model, tok, exs, task,
                    layer_indices=layer_indices,
                    basis_np=basis, alpha=alpha,
                    staged=staged, reasoning_tokens=args.reasoning_tokens,
                    batch_size=args.batch_size, max_prompt_len=args.max_prompt_len,
                    bootstrap_iters=args.bootstrap_iters, perm_iters=args.perm_iters,
                    ci_alpha=args.ci_alpha, seed=args.seed + stable_int_seed(task, name, "fc")
                )
                print(f"  acc={fmt_ci(fc['accuracy'], fc['ci_low'], fc['ci_high'])} {fc['hook_stats']} cand_lens={fc.get('cand_token_lens')}")
                block["forced_choice"][name] = fc

            # paired tests on forced-choice (more reliable vs collapse)
            b = np.array(block["forced_choice"]["baseline"]["correct"], dtype=np.float32)
            s = np.array(block["forced_choice"]["shared_full"]["correct"], dtype=np.float32)
            r = np.array(block["forced_choice"]["rand_full"]["correct"], dtype=np.float32)
            seed0 = stable_int_seed(args.seed, task, "paired_fc")
            block["paired"]["fc_shared_full_vs_baseline"] = summarize_paired(b, s, args.bootstrap_iters, args.perm_iters, args.ci_alpha, seed0 + 1)
            block["paired"]["fc_rand_full_vs_baseline"] = summarize_paired(b, r, args.bootstrap_iters, args.perm_iters, args.ci_alpha, seed0 + 2)
            block["paired"]["fc_shared_full_vs_rand_full"] = summarize_paired(r, s, args.bootstrap_iters, args.perm_iters, args.ci_alpha, seed0 + 3)

            print("[FC Paired]")
            for kname, stat in block["paired"].items():
                print(f"  {kname}: Δ={stat['mean_diff']:+.3f} CI[{stat['ci_low']:+.3f},{stat['ci_high']:+.3f}] p={stat['p_value']:.4g}")

        results["by_task"][task] = block

    json_dump_safe(results, args.out_json)

    # summary txt
    lines = []
    lines.append("="*80)
    lines.append("COLLAPSE vs REASONING (Generation+Extraction vs Forced-Choice Logprob)")
    lines.append("="*80)
    lines.append(f"Model={args.model} dtype={args.dtype} device={args.device} layer={layer_indices}")
    lines.append(f"cross_dim={extra['cross_dim']} shared_k={k} tau={args.tau} m_shared={args.m_shared}")
    lines.append("")

    for task in TASKS:
        blk = results["by_task"][task]
        lines.append("-"*80)
        lines.append(f"[{task}]")
        lines.append("Generation (collapse-sensitive):")
        for name in ["baseline","shared_full","shared_staged","rand_full","rand_staged"]:
            g = blk["generation"][name]
            lines.append(f"  {name:12s} acc={fmt_ci(g['accuracy'], g['ci_low'], g['ci_high'])} "
                         f"extr={fmt_ci(g['extraction'], g['ex_ci_low'], g['ex_ci_high'])} eos={g['eos_rate']:.3f} avg_new={g['avg_new_tok']:.1f}")
        if task in ["commonsenseqa","strategyqa","aqua"]:
            lines.append("Forced-choice (confound-free):")
            for name in ["baseline","shared_full","shared_staged","rand_full","rand_staged"]:
                fc = blk["forced_choice"][name]
                lines.append(f"  {name:12s} acc={fmt_ci(fc['accuracy'], fc['ci_low'], fc['ci_high'])}")
            if "paired" in blk:
                lines.append("Forced-choice paired tests:")
                for kname, stat in blk["paired"].items():
                    lines.append(f"  {kname}: Δ={stat['mean_diff']:+.3f} CI[{stat['ci_low']:+.3f},{stat['ci_high']:+.3f}] p={stat['p_value']:.4g}")
        lines.append("")

    with open(args.out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n" + "\n".join(lines))
    print(f"[Done] JSON: {args.out_json}")
    print(f"[Done] TXT : {args.out_txt}")

if __name__ == "__main__":
    main()
