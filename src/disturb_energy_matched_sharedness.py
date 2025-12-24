# -*- coding: utf-8 -*-
"""
disturb_energy_matched_sharedness.py

这个代码是prior的
Energy-exact matched control experiment (with sanity checks) that WORKS with forced-choice logprob eval.

Key fixes vs your current script:
  (A) Hook intervenes on the LAST POSITION hidden state for BOTH:
        - prefill (seq_len > 1)  --> affects logits[:, -1, :] (critical for forced-choice first token)
        - decode  (seq_len == 1)
      This is the main reason your previous forced-choice showed Δ=0 exactly.

  (B) Forced-choice scoring uses correct next-token logprob:
        log p(t0 | prompt) from logits(prompt)[-1]
        then (if multi-token candidate) cached decoding to score subsequent tokens.

  (C) Energy matching is computed on prefill-last states (distribution that decides forced-choice),
      while PCA/sharedness is estimated from decode-last-token states (A3 alignment).

Outputs:
  - energy_matched_results.json
  - energy_matched_summary.txt

Example:
  CUDA_VISIBLE_DEVICES=1 python disturb_energy_matched_sharedness.py \
    --model meta-llama/Llama-2-7b-chat-hf --device cuda --dtype fp32 \
    --layer 10 \
    --n_prompts 128 --calib_max_new_tokens 128 --per_task_max_states 20000 \
    --pca_var 0.95 --tau 0.001 --m_shared all \
    --eval_n 256 \
    --control_basis joint_nonshared_topk --energy_match mean \
    --use_chat_template 0 \
    --do_generation_eval 0

Notes:
  - If you want strictly alpha<=1 for both shared/control, use --energy_match min
  - If your model is chat-tuned and you want best baseline, try --use_chat_template 1
"""

import os
import re
import json
import math
import random
import argparse
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================================
# Repro / small utils
# =============================================================================
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


def to_jsonable(x: Any) -> Any:
    """Convert numpy / torch scalars to plain Python types recursively."""
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if torch.is_tensor(x) and x.numel() == 1:
        return float(x.detach().cpu().item())
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    return x


# =============================================================================
# Stats: bootstrap CI + paired tests
# =============================================================================
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


def paired_bootstrap_ci_diff(baseline: np.ndarray, treat: np.ndarray, iters: int, alpha: float, seed: int) -> Tuple[float, float, float]:
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
    diffs = treat - baseline
    obs = float(diffs.mean())
    n = len(diffs)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(iters):
        signs = rng.choice([-1.0, 1.0], size=n)
        perm = float((diffs * signs).mean())
        if abs(perm) >= abs(obs):
            count += 1
    return float((count + 1) / (iters + 1))


def summarize_paired(baseline: np.ndarray, treat: np.ndarray, label: str, bootstrap_iters: int, perm_iters: int, ci_alpha: float, seed: int) -> Dict[str, Any]:
    md, lo, hi = paired_bootstrap_ci_diff(baseline, treat, iters=bootstrap_iters, alpha=ci_alpha, seed=seed + 11)
    p = signflip_permutation_test(baseline, treat, iters=perm_iters, seed=seed + 29)
    return {"label": label, "mean_diff": md, "ci_low": lo, "ci_high": hi, "p_value": p}


def fmt_acc(acc: float, lo: float, hi: float) -> str:
    return f"{acc*100:.1f} [{lo*100:.1f}, {hi*100:.1f}]"


def fmt_diff(stat: Dict[str, Any]) -> str:
    return f"{stat['mean_diff']*100:+.1f} [{stat['ci_low']*100:+.1f}, {stat['ci_high']*100:+.1f}]"


# =============================================================================
# Datasets / prompts
# =============================================================================
@dataclass
class FCExample:
    task: str
    ex_id: str
    prompt: str
    gold: str


def safe_upper(x: Any) -> str:
    return str(x).strip().upper()


def maybe_apply_chat(tokenizer: AutoTokenizer, prompt: str, use_chat_template: bool) -> str:
    if not use_chat_template:
        return prompt
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        msgs = [{"role": "user", "content": prompt}]
        # add_generation_prompt=True makes the assistant turn start where next token is generated
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return prompt


def build_prompt_csqa(question: str, choices: Dict[str, List[str]], for_calib: bool) -> str:
    labels = choices["label"]
    texts = choices["text"]
    lines = [f"{lab}) {txt}" for lab, txt in zip(labels, texts)]
    tail = "Select the correct option (A, B, C, D, or E)."
    if for_calib:
        tail += " Give the answer and then a short explanation."
    return f"Question: {question}\nChoices:\n" + "\n".join(lines) + f"\n{tail}\nAnswer:"


def build_prompt_aqua(question: str, options: List[str], for_calib: bool) -> str:
    labels = ["A", "B", "C", "D", "E"]
    lines = []
    for i, opt in enumerate(options[:5]):
        lab = labels[i]
        opt_clean = re.sub(r"^[A-E]\)?\s*[:\-]?\s*", "", str(opt).strip(), flags=re.IGNORECASE)
        lines.append(f"{lab}) {opt_clean}")
    tail = "Select the correct option (A, B, C, D, or E)."
    if for_calib:
        tail += " Give the answer and then a short explanation."
    return f"Question: {question}\nChoices:\n" + "\n".join(lines) + f"\n{tail}\nAnswer:"


def build_prompt_strategyqa(question: str, for_calib: bool) -> str:
    tail = "Answer with Yes or No."
    if for_calib:
        tail += " Then give a short explanation."
    return f"Question: {question}\n{tail}\nAnswer:"


def build_prompt_gsm8k(question: str, for_calib: bool) -> str:
    # only used for calibration / PCA, not forced-choice eval
    tail = "Solve the problem."
    if for_calib:
        tail += " Give the final numeric answer and a short explanation."
    return f"Question: {question}\n{tail}\nAnswer:"


def sample_hf_split(ds_split, n: int, seed: int):
    n = min(n, len(ds_split))
    if n <= 0:
        return ds_split.select([])
    return ds_split.shuffle(seed=seed).select(range(n))


def load_calib_prompts_and_fc_eval(seed: int, n_prompts: int, eval_n: int) -> Tuple[Dict[str, List[str]], Dict[str, List[FCExample]], Dict[str, Any]]:
    """
    Returns:
      prompts_by_task: prompts for calibration decode collection (all 4 tasks)
      eval_fc_by_task: forced-choice evaluation examples (csqa/strategyqa/aqua)
      meta: dataset info
    """
    meta: Dict[str, Any] = {}

    # GSM8K (calib only)
    ds_gsm = load_dataset("gsm8k", "main")
    split_train = "train" if "train" in ds_gsm else list(ds_gsm.keys())[0]
    rows = sample_hf_split(ds_gsm[split_train], n_prompts, seed + 1)
    gsm_prompts = [build_prompt_gsm8k(ex["question"], for_calib=True) for ex in rows]
    meta["gsm8k"] = {"hf_id": "gsm8k/main", "calib_split": split_train}

    # CommonsenseQA
    ds_csqa = load_dataset("commonsense_qa")
    split_train = "train" if "train" in ds_csqa else list(ds_csqa.keys())[0]
    split_eval = "validation" if "validation" in ds_csqa else ("test" if "test" in ds_csqa else split_train)
    rows_cal = sample_hf_split(ds_csqa[split_train], n_prompts, seed + 11)
    rows_eval = sample_hf_split(ds_csqa[split_eval], eval_n, seed + 12)

    csqa_prompts = [build_prompt_csqa(ex["question"], ex["choices"], for_calib=True) for ex in rows_cal]
    csqa_eval = []
    for i, ex in enumerate(rows_eval):
        p = build_prompt_csqa(ex["question"], ex["choices"], for_calib=False)
        csqa_eval.append(FCExample("commonsenseqa", f"csqa-{split_eval}-{i}", p, safe_upper(ex["answerKey"])))
    meta["commonsenseqa"] = {"hf_id": "commonsense_qa", "calib_split": split_train, "eval_split": split_eval}

    # StrategyQA (your validated HF repo)
    hf_id = "ChilleD/StrategyQA"
    ds_sq = load_dataset(hf_id)
    split_train = "train" if "train" in ds_sq else list(ds_sq.keys())[0]
    split_eval = "test" if "test" in ds_sq else ("validation" if "validation" in ds_sq else split_train)
    rows_cal = sample_hf_split(ds_sq[split_train], n_prompts, seed + 21)
    rows_eval = sample_hf_split(ds_sq[split_eval], eval_n, seed + 22)

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

    sq_prompts = [build_prompt_strategyqa(ex["question"], for_calib=True) for ex in rows_cal]
    sq_eval = []
    for i, ex in enumerate(rows_eval):
        p = build_prompt_strategyqa(ex["question"], for_calib=False)
        sq_eval.append(FCExample("strategyqa", f"strategyqa-{split_eval}-{i}", p, to_yesno(ex["answer"])))
    meta["strategyqa"] = {"hf_id": hf_id, "calib_split": split_train, "eval_split": split_eval}

    # AQuA
    ds_aq = load_dataset("aqua_rat")
    split_train = "train" if "train" in ds_aq else list(ds_aq.keys())[0]
    split_eval = "test" if "test" in ds_aq else ("validation" if "validation" in ds_aq else split_train)
    rows_cal = sample_hf_split(ds_aq[split_train], n_prompts, seed + 31)
    rows_eval = sample_hf_split(ds_aq[split_eval], eval_n, seed + 32)

    def get_gold_aqua(ex: dict) -> str:
        if "correct" in ex:
            return safe_upper(ex["correct"])
        if "answer" in ex:
            return safe_upper(ex["answer"])
        return ""

    aq_prompts = [build_prompt_aqua(ex["question"], ex["options"], for_calib=True) for ex in rows_cal]
    aq_eval = []
    for i, ex in enumerate(rows_eval):
        p = build_prompt_aqua(ex["question"], ex["options"], for_calib=False)
        aq_eval.append(FCExample("aqua", f"aqua-{split_eval}-{i}", p, get_gold_aqua(ex)))
    meta["aqua"] = {"hf_id": "aqua_rat", "calib_split": split_train, "eval_split": split_eval}

    prompts_by_task = {
        "gsm8k": gsm_prompts,
        "commonsenseqa": csqa_prompts,
        "strategyqa": sq_prompts,
        "aqua": aq_prompts,
    }
    eval_fc_by_task = {
        "commonsenseqa": csqa_eval,
        "strategyqa": sq_eval,
        "aqua": aq_eval,
    }
    return prompts_by_task, eval_fc_by_task, meta


# =============================================================================
# Model layers (generic-ish)
# =============================================================================
def get_decoder_layers(model) -> List[torch.nn.Module]:
    # LLaMA / Mistral / Qwen2 style
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    # GPT-2 style
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    # GPT-NeoX style
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return list(model.gpt_neox.layers)
    raise RuntimeError("Cannot locate decoder layers on this model. Please extend get_decoder_layers().")


# =============================================================================
# Collect activations (decode-last-token for PCA + prefill-last-token for energy match)
# =============================================================================
class CalibActivationCollector:
    """
    Collect:
      - prefill_last: last-position hidden state during prompt prefill (seq_len > 1)
      - decode_last : last-position hidden state during cached decoding (seq_len == 1)

    storage[task]["prefill_last"] -> list of [B, D]
    storage[task]["decode_last"]  -> list of [b_active, D]
    """

    def __init__(self, layer_indices: List[int]):
        self.layer_indices = list(layer_indices)
        self.cur_task: Optional[str] = None
        self.capture_prefill_last = False
        self.capture_decode_last = False
        self.active_mask: Optional[torch.Tensor] = None
        self.storage: Dict[str, Dict[str, Dict[int, List[np.ndarray]]]] = {}

    def set_task(self, task: str) -> None:
        self.cur_task = task
        if task not in self.storage:
            self.storage[task] = {"prefill_last": {}, "decode_last": {}}

    def set_capture(self, prefill_last: bool, decode_last: bool, active_mask: Optional[torch.Tensor]) -> None:
        self.capture_prefill_last = bool(prefill_last)
        self.capture_decode_last = bool(decode_last)
        self.active_mask = active_mask

    def make_hook(self, layer_idx: int):
        def _hook(module, inputs, output):
            if self.cur_task is None:
                return output

            hs = output[0] if isinstance(output, tuple) else output
            if not isinstance(hs, torch.Tensor) or hs.ndim != 3:
                return output

            # prefill capture: seq_len > 1, take last position
            if self.capture_prefill_last and hs.shape[1] > 1:
                x = hs[:, -1, :].detach().float().cpu().numpy()
                self.storage[self.cur_task]["prefill_last"].setdefault(layer_idx, []).append(x)

            # decode capture: seq_len == 1, take last position; optionally mask unfinished
            if self.capture_decode_last and hs.shape[1] == 1:
                x = hs[:, -1, :]
                if self.active_mask is not None and self.active_mask.numel() == x.shape[0]:
                    m = self.active_mask.bool()
                    x = x[m]
                if x.numel() > 0:
                    x_np = x.detach().float().cpu().numpy()
                    self.storage[self.cur_task]["decode_last"].setdefault(layer_idx, []).append(x_np)

            return output

        return _hook

    def get_concat(self, task: str, kind: str, layer_idx: int) -> Optional[np.ndarray]:
        chunks = self.storage.get(task, {}).get(kind, {}).get(layer_idx, [])
        if not chunks:
            return None
        return np.concatenate(chunks, axis=0)

    def clear(self) -> None:
        self.storage.clear()
        self.cur_task = None
        self.capture_prefill_last = False
        self.capture_decode_last = False
        self.active_mask = None


@torch.no_grad()
def collect_calib_states(
    model,
    tokenizer,
    prompts_by_task: Dict[str, List[str]],
    layer_indices: List[int],
    *,
    calib_batch_size: int,
    calib_max_new_tokens: int,
    max_prompt_len: int,
    use_chat_template: bool,
    seed: int,
) -> Tuple[Dict[str, Dict[int, np.ndarray]], Dict[str, Dict[int, np.ndarray]]]:
    """
    Returns:
      decode_states[task][layer]  -> [N, D] (many)
      prefill_states[task][layer] -> [N_prompt, D] (one per prompt)
    """
    device = next(model.parameters()).device
    model.eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos = tokenizer.eos_token_id

    layers = get_decoder_layers(model)

    collector = CalibActivationCollector(layer_indices)
    handles = []
    for li in layer_indices:
        if li >= len(layers):
            raise ValueError(f"layer_idx={li} out of range (layers={len(layers)})")
        handles.append(layers[li].register_forward_hook(collector.make_hook(li)))

    try:
        for task, prompts in prompts_by_task.items():
            collector.set_task(task)
            # apply chat template if requested
            prompts2 = [maybe_apply_chat(tokenizer, p, use_chat_template) for p in prompts]

            for i in tqdm(range(0, len(prompts2), calib_batch_size), desc=f"CollectDecode({task})"):
                batch = prompts2[i:i + calib_batch_size]
                inputs = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_prompt_len,
                ).to(device)

                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]
                B = input_ids.shape[0]

                # PREFILL: capture prefill-last (this controls first-token logits)
                collector.set_capture(prefill_last=True, decode_last=False, active_mask=None)
                out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
                logits = out.logits[:, -1, :]
                past = out.past_key_values

                unfinished = torch.ones(B, dtype=torch.bool, device=device)

                # DECODE: capture decode-last for PCA alignment
                for _ in range(calib_max_new_tokens):
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

                    collector.set_capture(prefill_last=False, decode_last=True, active_mask=unfinished)

                    out = model(
                        input_ids=next_token,
                        attention_mask=attention_mask,
                        use_cache=True,
                        past_key_values=past,
                    )
                    logits = out.logits[:, -1, :]
                    past = out.past_key_values

            collector.set_capture(False, False, None)

    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass
        collector.set_capture(False, False, None)

    decode_states: Dict[str, Dict[int, np.ndarray]] = {}
    prefill_states: Dict[str, Dict[int, np.ndarray]] = {}
    for task in prompts_by_task.keys():
        decode_states[task] = {}
        prefill_states[task] = {}
        for li in layer_indices:
            d = collector.get_concat(task, "decode_last", li)
            p = collector.get_concat(task, "prefill_last", li)
            if d is not None:
                decode_states[task][li] = d.astype(np.float32, copy=False)
            if p is not None:
                prefill_states[task][li] = p.astype(np.float32, copy=False)

    return decode_states, prefill_states


def _subsample_rows(x: np.ndarray, n_max: int, seed: int) -> np.ndarray:
    if n_max <= 0 or x.shape[0] <= n_max:
        return x
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.shape[0], size=n_max, replace=False)
    return x[idx]


# =============================================================================
# PCA + sharedness on decode distribution
# =============================================================================
def compute_pca_sharedness(
    decode_states: Dict[str, Dict[int, np.ndarray]],
    layer_idx: int,
    *,
    pca_var: float,
    tau: float,
    m_shared: str,
    per_task_max_states: int,
    device: str,
    seed: int,
    chunk_rows: int = 4096,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], Dict[str, np.ndarray]]:
    """
    Returns:
      Q_joint: [D, K] joint PCA basis (columns)
      eigvals: [K]   eigenvalues
      mean_vec: [D]  pooled mean used for centering
      shared_indices: list of indices in [0..K-1]
      relvar_by_task: task -> [K] relative variance contribution
    """
    tasks = list(decode_states.keys())

    # collect & subsample per task
    mats = {}
    for t in tasks:
        x = decode_states[t].get(layer_idx, None)
        if x is None or x.shape[0] == 0:
            raise RuntimeError(f"No decode states for task={t}, layer={layer_idx}")
        x = _subsample_rows(x, per_task_max_states, seed=stable_int_seed(seed, t, "sub") + 7)
        mats[t] = x

    # balance
    n_bal = min(m.shape[0] for m in mats.values())
    mats_bal = {}
    for t in tasks:
        m = mats[t]
        if m.shape[0] > n_bal:
            mats_bal[t] = _subsample_rows(m, n_bal, seed=stable_int_seed(seed, t, "bal") + 13)
        else:
            mats_bal[t] = m
    X = np.concatenate([mats_bal[t] for t in tasks], axis=0).astype(np.float32, copy=False)
    N, D = X.shape

    # pooled mean
    mean_vec = X.mean(axis=0).astype(np.float32, copy=False)

    # compute covariance on GPU/CPU by chunks: C = (Xc^T Xc) / (N-1)
    dev = torch.device(device)
    C = torch.zeros((D, D), dtype=torch.float32, device=dev)
    mean_t = torch.from_numpy(mean_vec).to(dev)

    for s in range(0, N, chunk_rows):
        e = min(N, s + chunk_rows)
        x_chunk = torch.from_numpy(X[s:e]).to(dev)
        x_chunk = x_chunk - mean_t
        C += x_chunk.T @ x_chunk
        del x_chunk

    C = C / max(N - 1, 1)

    # eigen decomposition (symmetric)
    evals, evecs = torch.linalg.eigh(C)  # ascending
    idx = torch.argsort(evals, descending=True)
    evals = evals[idx]
    evecs = evecs[:, idx]  # columns are eigenvectors

    # choose K by explained variance
    total = torch.sum(evals).clamp_min(1e-12)
    cum = torch.cumsum(evals, dim=0) / total
    K = int((cum < pca_var).sum().item()) + 1
    K = min(K, D)

    Q_joint = evecs[:, :K].contiguous()   # [D, K]
    eigvals = evals[:K].contiguous()      # [K]

    # per-task relvar
    relvar_by_task: Dict[str, np.ndarray] = {}
    Q_cpu = Q_joint.detach().cpu()
    mean_cpu = torch.from_numpy(mean_vec).float()

    for t in tasks:
        Xt = torch.from_numpy(mats_bal[t]).float()
        Z = (Xt - mean_cpu) @ Q_cpu  # [n, K]
        var = torch.var(Z, dim=0, unbiased=False)  # [K]
        denom = torch.sum(var).clamp_min(1e-12)
        rel = (var / denom).numpy()
        relvar_by_task[t] = rel.astype(np.float32, copy=False)

    # shared indices
    if m_shared == "all":
        m_req = len(tasks)
    else:
        m_req = int(m_shared)

    shared_mask = np.zeros((K,), dtype=np.int32)
    for j in range(K):
        c = 0
        for t in tasks:
            if relvar_by_task[t][j] > tau:
                c += 1
        shared_mask[j] = c

    shared_indices = [j for j in range(K) if shared_mask[j] >= m_req]

    return (
        Q_joint.detach().cpu().numpy().astype(np.float32, copy=False),
        eigvals.detach().cpu().numpy().astype(np.float32, copy=False),
        mean_vec,
        shared_indices,
        relvar_by_task,
    )


# =============================================================================
# Control basis selection (nonshared PCs)
# =============================================================================
def pick_control_indices_nonshared_topk(K: int, shared_indices: List[int], k_need: int) -> List[int]:
    shared_set = set(shared_indices)
    nonshared = [i for i in range(K) if i not in shared_set]
    if len(nonshared) < k_need:
        raise RuntimeError(f"Not enough nonshared components: have {len(nonshared)} need {k_need}")
    return nonshared[:k_need]


def pick_control_indices_nonshared_varmatch(eigvals: np.ndarray, shared_indices: List[int], k_need: int, seed: int) -> List[int]:
    """
    Greedy eigenvalue-matching:
      For each shared component (sorted by eigval desc), pick a nonshared component with closest eigval.
    """
    K = len(eigvals)
    shared_set = set(shared_indices)
    nonshared = [i for i in range(K) if i not in shared_set]
    if len(nonshared) < k_need:
        raise RuntimeError(f"Not enough nonshared components: have {len(nonshared)} need {k_need}")

    shared_sorted = sorted(shared_indices, key=lambda i: float(eigvals[i]), reverse=True)[:k_need]
    avail = nonshared.copy()
    rng = np.random.default_rng(seed + 17)
    rng.shuffle(avail)

    picked = []
    for si in shared_sorted:
        target = float(eigvals[si])
        # find closest in avail
        best = min(avail, key=lambda j: abs(float(eigvals[j]) - target))
        picked.append(best)
        avail.remove(best)
    return picked


def orthonormality_max_offdiag(Q: np.ndarray) -> float:
    G = Q.T @ Q
    I = np.eye(G.shape[0], dtype=np.float32)
    off = np.abs(G - I)
    off[np.eye(G.shape[0], dtype=bool)] = 0.0
    return float(off.max())


def max_overlap(Q1: np.ndarray, Q2: np.ndarray) -> float:
    M = np.abs(Q1.T @ Q2)
    return float(M.max())


def projection_energy(h: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    h: [N, D]
    Q: [D, k] orthonormal
    Returns: [N] energies ||P_Q h||^2 = ||h Q||^2
    """
    z = h @ Q
    return np.sum(z * z, axis=1)


def energy_ratio(h: np.ndarray, Q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    eproj = projection_energy(h, Q)
    etot = np.sum(h * h, axis=1) + eps
    return eproj / etot


def compute_energy_match_alphas(
    h_prefill: np.ndarray,
    Q_shared: np.ndarray,
    Q_ctrl: np.ndarray,
    *,
    energy_match: str,
    alpha_shared_base: float = 1.0,
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Match mean removed-energy:
      removed_energy_mean = alpha^2 * E[ ||P_Q h||^2 ]  (since delta = alpha * P_Q h)
    """
    Es = float(np.mean(projection_energy(h_prefill, Q_shared)))
    Ec = float(np.mean(projection_energy(h_prefill, Q_ctrl)))

    if Es <= 0 or Ec <= 0:
        raise RuntimeError(f"Non-positive projection energy: Es={Es} Ec={Ec}")

    if energy_match == "mean":
        alpha_s = float(alpha_shared_base)
        target = (alpha_s ** 2) * Es
        alpha_c = math.sqrt(target / Ec)
    elif energy_match == "min":
        # conservative: choose target so both alphas <= 1 when alpha_shared_base==1
        alpha_s = float(alpha_shared_base)
        target_full_shared = (alpha_s ** 2) * Es
        target = min(target_full_shared, Ec)  # ensure alpha_c<=1
        alpha_c = math.sqrt(target / Ec)
        alpha_s = math.sqrt(target / Es)
    else:
        raise ValueError("energy_match must be 'mean' or 'min'")

    info = {
        "Es_proj_mean": Es,
        "Ec_proj_mean": Ec,
        "removed_target_mean": float((alpha_s ** 2) * Es),
        "alpha_shared": alpha_s,
        "alpha_ctrl": alpha_c,
    }
    return alpha_s, alpha_c, info


# =============================================================================
# Intervention hook (LAST POSITION in all forward passes)
# =============================================================================
class HookStats:
    def __init__(self, name: str):
        self.name = name
        self.calls = 0
        self.intervened = 0

    def as_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "calls": int(self.calls), "intervened": int(self.intervened)}


class LastPosSubspaceRemovalHook:
    """
    Apply on LAST POSITION only:
      h_last <- h_last - alpha * Q Q^T h_last
    This affects:
      - prefill last token (seq_len>1) --> first-token logits (critical for forced-choice)
      - decode steps (seq_len==1)

    This is the key fix vs decode-only hooks.
    """

    def __init__(self, Q_np: np.ndarray, alpha: float, stats: HookStats):
        self.alpha = float(alpha)
        self.stats = stats

        Q = torch.tensor(Q_np, dtype=torch.float32)
        # Orthonormalize once (in case numeric)
        q, _ = torch.linalg.qr(Q)
        self.Q_cpu = q.contiguous()
        self.Q_device: Optional[torch.Tensor] = None

    def _Q(self, device: torch.device) -> torch.Tensor:
        if self.Q_device is None or self.Q_device.device != device:
            self.Q_device = self.Q_cpu.to(device=device, dtype=torch.float32)
        return self.Q_device

    def __call__(self, module, inputs, output):
        self.stats.calls += 1

        if isinstance(output, tuple):
            hs = output[0]
            rest = output[1:]
        else:
            hs = output
            rest = None

        if not isinstance(hs, torch.Tensor) or hs.ndim != 3:
            return output

        # intervene on last position
        self.stats.intervened += 1
        hs2 = hs.clone()
        x = hs2[:, -1, :]  # [B, D]
        Q = self._Q(hs2.device)

        x_fp32 = x.float()
        proj = (x_fp32 @ Q) @ Q.T
        x_new = x_fp32 - self.alpha * proj
        hs2[:, -1, :] = x_new.to(dtype=hs2.dtype)

        if rest is None:
            return hs2
        return (hs2,) + rest


def register_hooks(model, layer_indices: List[int], hook_obj) -> List[Any]:
    layers = get_decoder_layers(model)
    handles = []
    for li in layer_indices:
        if li >= len(layers):
            raise ValueError(f"layer_idx={li} out of range")
        handles.append(layers[li].register_forward_hook(hook_obj))
    return handles


def remove_hooks(handles: List[Any]) -> None:
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass


# =============================================================================
# Forced-choice logprob evaluation
# =============================================================================
@torch.no_grad()
def forced_choice_eval(
    model,
    tokenizer,
    examples: List[FCExample],
    *,
    candidates: List[str],
    condition_name: str,
    Q: Optional[np.ndarray],
    alpha: float,
    layer_indices: List[int],
    max_prompt_len: int,
    batch_size: int,
    bootstrap_iters: int,
    perm_iters: int,
    ci_alpha: float,
    seed: int,
    use_chat_template: bool,
) -> Dict[str, Any]:
    device = next(model.parameters()).device
    model.eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare hook if needed
    hook_stats = HookStats(condition_name)
    handles = []
    if Q is not None:
        hook = LastPosSubspaceRemovalHook(Q_np=Q, alpha=alpha, stats=hook_stats)
        handles = register_hooks(model, layer_indices, hook)

    try:
        prompts = [maybe_apply_chat(tokenizer, ex.prompt, use_chat_template) for ex in examples]
        golds = [ex.gold for ex in examples]

        # Tokenize candidates once (with leading space)
        cand_token_ids: List[List[int]] = []
        for c in candidates:
            toks = tokenizer.encode(" " + c, add_special_tokens=False)
            if len(toks) == 0:
                raise RuntimeError(f"Empty tokenization for candidate='{c}'")
            cand_token_ids.append(toks)

        preds = []
        correct = []

        # Sanity: measure if logits(prompt)[-1] is changing under hook
        logits_diff_debug = None

        for i in tqdm(range(0, len(prompts), batch_size), desc=f"ForcedChoice({examples[0].task})"):
            batch_prompts = prompts[i:i + batch_size]
            batch_golds = golds[i:i + batch_size]
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_prompt_len,
            ).to(device)

            out = model(**inputs, use_cache=True)
            logits_next = out.logits[:, -1, :]  # [B, V]
            past_base = out.past_key_values
            attn_base = inputs["attention_mask"]

            # Optional logits diff sanity on very first batch:
            # Run baseline logits for same batch without hooks (only if hooks enabled)
            if logits_diff_debug is None and Q is not None:
                # Temporarily remove hook handles to compute baseline logits
                remove_hooks(handles)
                out0 = model(**inputs, use_cache=True)
                logits0 = out0.logits[:, -1, :]
                # re-register
                handles = register_hooks(model, layer_indices, hook)
                logits_diff_debug = float(torch.mean(torch.abs(logits_next - logits0)).detach().cpu().item())

            logp0 = torch.log_softmax(logits_next, dim=-1)  # [B, V]

            # score each candidate
            cand_scores = []
            B = logits_next.shape[0]

            for toks in cand_token_ids:
                # score first token from prefill logits
                s = logp0[:, toks[0]].clone()

                # score subsequent tokens (if any) via cached decoding
                past = past_base
                attn = attn_base

                for j in range(len(toks) - 1):
                    tid = toks[j]
                    tid_next = toks[j + 1]
                    # extend attention mask by 1
                    attn = torch.cat([attn, torch.ones((B, 1), device=device, dtype=attn.dtype)], dim=1)
                    inp = torch.full((B, 1), tid, device=device, dtype=inputs["input_ids"].dtype)
                    outj = model(input_ids=inp, attention_mask=attn, past_key_values=past, use_cache=True)
                    logitsj = outj.logits[:, -1, :]
                    past = outj.past_key_values
                    s = s + torch.log_softmax(logitsj, dim=-1)[:, tid_next]

                cand_scores.append(s)

            # pick argmax
            scores = torch.stack(cand_scores, dim=1)  # [B, C]
            best = torch.argmax(scores, dim=1).detach().cpu().numpy().tolist()

            for b in range(len(best)):
                pred = candidates[int(best[b])]
                preds.append(pred)
                correct.append(1 if safe_upper(pred) == safe_upper(batch_golds[b]) else 0)

        correct_arr = np.array(correct, dtype=np.float32)
        acc, lo, hi = bootstrap_ci_mean(correct_arr, iters=bootstrap_iters, alpha=ci_alpha,
                                        seed=stable_int_seed(seed, examples[0].task, condition_name, "ci"))

        return {
            "task": examples[0].task,
            "condition": condition_name,
            "alpha": float(alpha),
            "accuracy": float(acc),
            "ci_low": float(lo),
            "ci_high": float(hi),
            "correct": correct_arr.tolist(),
            "preds": preds,
            "hook_stats": hook_stats.as_dict(),
            "sanity_logits_mean_abs_diff_vs_nohook_firstbatch": logits_diff_debug,
        }

    finally:
        remove_hooks(handles)


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--n_prompts", type=int, default=128)
    ap.add_argument("--calib_max_new_tokens", type=int, default=128)
    ap.add_argument("--per_task_max_states", type=int, default=20000)
    ap.add_argument("--max_prompt_len", type=int, default=512)
    ap.add_argument("--pca_var", type=float, default=0.95)
    ap.add_argument("--tau", type=float, default=0.001)
    ap.add_argument("--m_shared", type=str, default="all")  # "all" or int
    ap.add_argument("--eval_n", type=int, default=256)
    ap.add_argument("--control_basis", type=str, default="joint_nonshared_topk",
                    choices=["joint_nonshared_topk", "joint_nonshared_varmatch"])
    ap.add_argument("--energy_match", type=str, default="mean", choices=["mean", "min"])
    ap.add_argument("--bootstrap_iters", type=int, default=5000)
    ap.add_argument("--perm_iters", type=int, default=10000)
    ap.add_argument("--ci_alpha", type=float, default=0.05)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--use_chat_template", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_json", type=str, default="energy_matched_results.json")
    ap.add_argument("--out_txt", type=str, default="energy_matched_summary.txt")
    ap.add_argument("--do_generation_eval", type=int, default=0)  # kept for compatibility; not implemented here
    args = ap.parse_args()

    set_global_seed(args.seed)

    # load model
    dtype = torch.float32 if args.dtype == "fp32" else torch.float16
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = model.to(args.device)
    model.eval()
    model.config.use_cache = True

    hidden_dim = getattr(model.config, "hidden_size", None) or getattr(model.config, "n_embd", None)
    if hidden_dim is None:
        raise RuntimeError("Cannot infer hidden_dim from model.config")

    layer_indices = [args.layer]

    print(f"[Env] model={args.model} device={args.device} dtype={args.dtype} hidden_dim={hidden_dim} layer={layer_indices}")

    # load data
    prompts_by_task, eval_fc_by_task, meta = load_calib_prompts_and_fc_eval(
        seed=args.seed, n_prompts=args.n_prompts, eval_n=args.eval_n
    )
    print(f"[Data] calib tasks={list(prompts_by_task.keys())} n_prompts_per_task={args.n_prompts}")
    print(f"[Data] eval forced-choice tasks={list(eval_fc_by_task.keys())} eval_n={args.eval_n}")

    # collect states
    decode_states, prefill_states = collect_calib_states(
        model=model,
        tokenizer=tok,
        prompts_by_task=prompts_by_task,
        layer_indices=layer_indices,
        calib_batch_size=args.batch_size,
        calib_max_new_tokens=args.calib_max_new_tokens,
        max_prompt_len=args.max_prompt_len,
        use_chat_template=bool(args.use_chat_template),
        seed=args.seed,
    )

    # build matrices for PCA (decode distribution)
    # note: decode_states includes gsm8k too, which is fine for pooled PCA/sharedness
    #       forced-choice eval only uses csqa/strategyqa/aqua
    Q_joint, eigvals, mean_vec, shared_indices, relvar_by_task = compute_pca_sharedness(
        decode_states=decode_states,
        layer_idx=args.layer,
        pca_var=args.pca_var,
        tau=args.tau,
        m_shared=args.m_shared,
        per_task_max_states=args.per_task_max_states,
        device=args.device,
        seed=args.seed,
    )
    K = Q_joint.shape[1]
    k_shared = len(shared_indices)
    if k_shared == 0:
        raise RuntimeError("shared_indices is empty. Try smaller --tau or m_shared=2/3.")

    print("\n" + "=" * 80)
    print("[Subspace]")
    print(f"  cross_dim={K} shared_k={k_shared} tau={args.tau} m_shared={args.m_shared}")
    print("=" * 80)

    Q_shared = Q_joint[:, shared_indices]  # [D, k_shared]

    # pick control indices from nonshared PCs
    if args.control_basis == "joint_nonshared_topk":
        ctrl_idx = pick_control_indices_nonshared_topk(K, shared_indices, k_need=k_shared)
    else:
        ctrl_idx = pick_control_indices_nonshared_varmatch(eigvals, shared_indices, k_need=k_shared, seed=args.seed)

    Q_ctrl = Q_joint[:, ctrl_idx]

    # Orthonormal sanity (after PCA they are orthonormal-ish, but check)
    off_s = orthonormality_max_offdiag(Q_shared)
    off_c = orthonormality_max_offdiag(Q_ctrl)
    ov = max_overlap(Q_shared, Q_ctrl)

    # Energy on prefill-last states (THIS controls forced-choice)
    # Pool prefill-last states across tasks equally (balance by prompts)
    pre_list = []
    for t in prompts_by_task.keys():
        hp = prefill_states[t].get(args.layer, None)
        if hp is None:
            continue
        # hp has ~n_prompts rows; keep as-is
        pre_list.append(hp)
    H_prefill = np.concatenate(pre_list, axis=0).astype(np.float32, copy=False)

    er_s = energy_ratio(H_prefill, Q_shared)
    er_c = energy_ratio(H_prefill, Q_ctrl)

    # energy match alphas based on prefill-last
    alpha_shared, alpha_ctrl, em_info = compute_energy_match_alphas(
        H_prefill, Q_shared, Q_ctrl,
        energy_match=args.energy_match,
        alpha_shared_base=1.0,
    )

    # Print sanity block
    print("\n" + "=" * 80)
    print("[Sanity]")
    print(f"  cross_dim={K} shared_k={k_shared} control_basis={args.control_basis} energy_match={args.energy_match}")
    print(f"  Orthonormality max offdiag: shared={off_s:.3e}, ctrl={off_c:.3e}")
    print(f"  Max overlap |Q_shared^T Q_ctrl| = {ov:.3e}")
    print("  Energy ratio on PREFILL-last states (where forced-choice logits come from):")
    print(f"    shared mean={float(er_s.mean()):.4f} (p50={float(np.percentile(er_s, 50)):.4f}, p95={float(np.percentile(er_s, 95)):.4f})")
    print(f"    ctrl   mean={float(er_c.mean()):.4f} (p50={float(np.percentile(er_c, 50)):.4f}, p95={float(np.percentile(er_c, 95)):.4f})")
    print("  Removed-energy (mean) match check on PREFILL-last:")
    print(f"    alpha_shared={alpha_shared:.4f}, E||P_s h||^2={em_info['Es_proj_mean']:.4e}, removed_mean={alpha_shared**2 * em_info['Es_proj_mean']:.4e}")
    print(f"    alpha_ctrl  ={alpha_ctrl:.4f}, E||P_c h||^2={em_info['Ec_proj_mean']:.4e}, removed_mean={alpha_ctrl**2 * em_info['Ec_proj_mean']:.4e}")
    if alpha_ctrl > 2.5:
        print("  [Sanity][WARN] alpha_ctrl is quite large. Consider --energy_match min or --control_basis joint_nonshared_varmatch.")
    print("=" * 80)

    # forced-choice eval
    results = {
        "config": {
            "model": args.model,
            "device": args.device,
            "dtype": args.dtype,
            "layer_indices": layer_indices,
            "n_prompts": args.n_prompts,
            "calib_max_new_tokens": args.calib_max_new_tokens,
            "per_task_max_states": args.per_task_max_states,
            "pca_var": args.pca_var,
            "tau": args.tau,
            "m_shared": args.m_shared,
            "eval_n": args.eval_n,
            "control_basis": args.control_basis,
            "energy_match": args.energy_match,
            "alpha_shared": alpha_shared,
            "alpha_ctrl": alpha_ctrl,
            "shared_k": k_shared,
            "cross_dim": K,
            "use_chat_template": int(bool(args.use_chat_template)),
            "seed": args.seed,
            "dataset_meta": meta,
        },
        "sanity": {
            "orth_offdiag_shared": off_s,
            "orth_offdiag_ctrl": off_c,
            "max_overlap": ov,
            "energy_ratio_shared_mean": float(er_s.mean()),
            "energy_ratio_ctrl_mean": float(er_c.mean()),
            "energy_ratio_shared_p50": float(np.percentile(er_s, 50)),
            "energy_ratio_ctrl_p50": float(np.percentile(er_c, 50)),
            "energy_ratio_shared_p95": float(np.percentile(er_s, 95)),
            "energy_ratio_ctrl_p95": float(np.percentile(er_c, 95)),
            "removed_energy_target_mean": em_info["removed_target_mean"],
        },
        "by_task": {},
    }

    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("ENERGY-EXACT MATCHED CONTROL (FORCED-CHOICE) SUMMARY")
    summary_lines.append("=" * 80)
    summary_lines.append(f"Model={args.model} dtype={args.dtype} device={args.device} layer={layer_indices}")
    summary_lines.append(f"cross_dim={K} shared_k={k_shared} control_basis={args.control_basis} energy_match={args.energy_match}")
    summary_lines.append(f"alpha_shared={alpha_shared:.4f} alpha_ctrl={alpha_ctrl:.4f}")
    summary_lines.append(f"EnergyRatio(shared) mean={float(er_s.mean()):.4f}, ctrl mean={float(er_c.mean()):.4f}")
    summary_lines.append(f"MaxOverlap |Q_s^T Q_c|={ov:.3e}")
    summary_lines.append("")

    for task, exs in eval_fc_by_task.items():
        print("\n" + "-" * 80)
        print(f"[ForcedChoice] task={task} n={len(exs)}")
        print("-" * 80)

        if task in ["commonsenseqa", "aqua"]:
            candidates = ["A", "B", "C", "D", "E"]
        elif task == "strategyqa":
            candidates = ["Yes", "No"]
        else:
            raise ValueError(task)

        base = forced_choice_eval(
            model=model,
            tokenizer=tok,
            examples=exs,
            candidates=candidates,
            condition_name="baseline",
            Q=None,
            alpha=0.0,
            layer_indices=layer_indices,
            max_prompt_len=args.max_prompt_len,
            batch_size=args.batch_size,
            bootstrap_iters=args.bootstrap_iters,
            perm_iters=args.perm_iters,
            ci_alpha=args.ci_alpha,
            seed=args.seed,
            use_chat_template=bool(args.use_chat_template),
        )
        shared = forced_choice_eval(
            model=model,
            tokenizer=tok,
            examples=exs,
            candidates=candidates,
            condition_name="shared_full",
            Q=Q_shared,
            alpha=alpha_shared,
            layer_indices=layer_indices,
            max_prompt_len=args.max_prompt_len,
            batch_size=args.batch_size,
            bootstrap_iters=args.bootstrap_iters,
            perm_iters=args.perm_iters,
            ci_alpha=args.ci_alpha,
            seed=args.seed + 1,
            use_chat_template=bool(args.use_chat_template),
        )
        ctrl = forced_choice_eval(
            model=model,
            tokenizer=tok,
            examples=exs,
            candidates=candidates,
            condition_name="ctrl_full",
            Q=Q_ctrl,
            alpha=alpha_ctrl,
            layer_indices=layer_indices,
            max_prompt_len=args.max_prompt_len,
            batch_size=args.batch_size,
            bootstrap_iters=args.bootstrap_iters,
            perm_iters=args.perm_iters,
            ci_alpha=args.ci_alpha,
            seed=args.seed + 2,
            use_chat_template=bool(args.use_chat_template),
        )

        base_arr = np.array(base["correct"], dtype=np.float32)
        sh_arr = np.array(shared["correct"], dtype=np.float32)
        ct_arr = np.array(ctrl["correct"], dtype=np.float32)

        seed0 = stable_int_seed(args.seed, "paired", task)

        stat_sh_base = summarize_paired(base_arr, sh_arr, f"{task}:shared_vs_base",
                                        args.bootstrap_iters, args.perm_iters, args.ci_alpha, seed0 + 10)
        stat_ct_base = summarize_paired(base_arr, ct_arr, f"{task}:ctrl_vs_base",
                                        args.bootstrap_iters, args.perm_iters, args.ci_alpha, seed0 + 20)
        stat_sh_ct = summarize_paired(ct_arr, sh_arr, f"{task}:shared_vs_ctrl",
                                      args.bootstrap_iters, args.perm_iters, args.ci_alpha, seed0 + 30)

        # Print quick
        print(f"  baseline    acc={fmt_acc(base['accuracy'], base['ci_low'], base['ci_high'])}")
        print(f"  shared_full acc={fmt_acc(shared['accuracy'], shared['ci_low'], shared['ci_high'])}  [HookStats] {shared['hook_stats']}")
        print(f"  ctrl_full   acc={fmt_acc(ctrl['accuracy'], ctrl['ci_low'], ctrl['ci_high'])}  [HookStats] {ctrl['hook_stats']}")
        print("  [Paired]")
        print(f"    shared_full_vs_baseline: Δ={fmt_diff(stat_sh_base)} p={stat_sh_base['p_value']:.4g}")
        print(f"    ctrl_full_vs_baseline  : Δ={fmt_diff(stat_ct_base)} p={stat_ct_base['p_value']:.4g}")
        print(f"    shared_full_vs_ctrl_full: Δ={fmt_diff(stat_sh_ct)} p={stat_sh_ct['p_value']:.4g}")

        # Extra sanity: logits diff should not be None for shared/ctrl
        if shared.get("sanity_logits_mean_abs_diff_vs_nohook_firstbatch", None) is None:
            print("  [Sanity][WARN] logits-diff sanity is None (unexpected).")
        else:
            d = shared["sanity_logits_mean_abs_diff_vs_nohook_firstbatch"]
            if d is not None and d < 1e-6:
                print("  [Sanity][WARN] shared hook does NOT change prompt-next logits (diff ~ 0). Something is still wrong.")

        results["by_task"][task] = {
            "baseline": base,
            "shared_full": shared,
            "ctrl_full": ctrl,
            "paired": {
                "shared_full_vs_baseline": stat_sh_base,
                "ctrl_full_vs_baseline": stat_ct_base,
                "shared_full_vs_ctrl_full": stat_sh_ct,
            },
        }

        summary_lines.append(f"[ForcedChoice] {task} n={len(exs)}")
        summary_lines.append(f"  baseline    {fmt_acc(base['accuracy'], base['ci_low'], base['ci_high'])}")
        summary_lines.append(f"  shared_full {fmt_acc(shared['accuracy'], shared['ci_low'], shared['ci_high'])}")
        summary_lines.append(f"  ctrl_full   {fmt_acc(ctrl['accuracy'], ctrl['ci_low'], ctrl['ci_high'])}")
        summary_lines.append(f"  Δ(shared-base) {fmt_diff(stat_sh_base)} p={stat_sh_base['p_value']:.4g}")
        summary_lines.append(f"  Δ(ctrl-base)   {fmt_diff(stat_ct_base)} p={stat_ct_base['p_value']:.4g}")
        summary_lines.append(f"  Δ(shared-ctrl) {fmt_diff(stat_sh_ct)} p={stat_sh_ct['p_value']:.4g}")
        summary_lines.append("")

    # Save
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(results), f, ensure_ascii=False, indent=2)

    with open(args.out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print("\n" + "\n".join(summary_lines))
    print("\n" + "=" * 80)
    print("[Done]")
    print(f"[Done] JSON: {os.path.abspath(args.out_json)}")
    print(f"[Done] TXT : {os.path.abspath(args.out_txt)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
