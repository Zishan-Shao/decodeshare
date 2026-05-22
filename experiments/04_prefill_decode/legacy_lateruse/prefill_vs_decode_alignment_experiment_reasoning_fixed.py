# -*- coding: utf-8 -*-
"""
prefill_vs_decode_alignment_experiment_reasoning_fixed.py

This is a patched version of your `prefill_vs_decode_alignment_experiment_reasoning.py`
focused on making forced-choice results "make sense" when using warmup tokens and/or
when prompts do not already end with an explicit answer prefix.

Key fixes vs your original script:
  1) Forced-choice now supports a prefix policy via --fc_prefix_mode {auto,always,never}.
     - auto (recommended): add the answer prefix when it is needed:
         * always add after warmup tokens (because the decision point moved deep into decode)
         * if no warmup, add only if the prompt does NOT already end with the prefix
     - always: always add the prefix before scoring candidates
     - never: never add (can lead to near-chance accuracy if prompts don't end with prefix)

     For backward compatibility, --fc_add_answer_prefix is still accepted:
       * 1 -> --fc_prefix_mode always
       * 0 -> --fc_prefix_mode auto   (NOTE: old semantics were "never"; use --fc_prefix_mode never now)

  2) Forced-choice correctness uses benchmark_dataloaders.is_correct() (robust normalization),
     instead of raw string equality.

  3) Adds runtime warnings for configurations that typically yield chance-level forced-choice accuracy.

All other experiment logic is unchanged: shared subspace estimation (decode vs prefill),
decode-only projection removal hook, bootstrap CIs and sign-flip permutation tests, and
generation evaluation for gsm8k (or for all tasks if --do_generation=1).

"""

import os
import sys
import json
import random
import argparse
import hashlib
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------
# Import your shared-subspace utilities (from your project)
# ---------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.append(THIS_DIR)
sys.path.append(os.path.join(THIS_DIR, ".."))

from decodeshare.joint_subspace_large.disturb_cross_task_all_shared import (  # noqa: E402
    get_model_layers,
    compute_cross_task_subspace,
    find_fully_shared_basis_improved,
)

# ---------------------------------------------------------------------
# Import dataset loaders / parsing helpers
# ---------------------------------------------------------------------
# The user may have attached a dataloader file with a different name.
# We try the canonical import first, then fall back.
try:
    from benchmark_dataloaders import (  # noqa: E402
        Example,
        load_selected_tasks as bdl_load_selected_tasks,
        parse_prediction as parse_prediction_generation,
        is_correct as is_correct_any,
    )
except Exception:
    try:
        from benchmark_dataloaders_aqua_prefix_default import (  # noqa: E402
            Example,
            load_selected_tasks as bdl_load_selected_tasks,
            parse_prediction as parse_prediction_generation,
            is_correct as is_correct_any,
        )
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Failed to import benchmark dataloaders.\n"
            "Expected `benchmark_dataloaders.py` (or fallback `benchmark_dataloaders_aqua_prefix_default.py`) "
            "to be on PYTHONPATH / in the same directory."
        ) from e


# -----------------------------
# Repro / utils
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


def safe_upper(x: Any) -> str:
    return str(x).strip().upper()


# -----------------------------
# Stats: bootstrap + paired test
# -----------------------------
def bootstrap_ci_mean(values: np.ndarray, iters: int, alpha: float, seed: int) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    m = float(values.mean())
    boots = []
    for _ in range(iters):
        idx = rng.integers(0, n, size=n)
        boots.append(float(values[idx].mean()))
    lo = float(np.percentile(boots, 100 * (alpha / 2)))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return m, lo, hi


def paired_bootstrap_ci_diff(baseline: np.ndarray, treatment: np.ndarray, iters: int, alpha: float, seed: int) -> Tuple[float, float, float]:
    assert baseline.shape == treatment.shape
    diffs = treatment - baseline
    obs = float(diffs.mean())
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boots = []
    for _ in range(iters):
        idx = rng.integers(0, n, size=n)
        boots.append(float(diffs[idx].mean()))
    lo = float(np.percentile(boots, 100 * (alpha / 2)))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return obs, lo, hi


def signflip_permutation_test(baseline: np.ndarray, treatment: np.ndarray, iters: int, seed: int) -> float:
    """Two-sided sign-flip permutation test on paired diffs."""
    assert baseline.shape == treatment.shape
    diffs = treatment - baseline
    obs = float(diffs.mean())
    rng = np.random.default_rng(seed)
    n = len(diffs)
    if n == 0:
        return float("nan")
    count = 0
    for _ in range(iters):
        signs = rng.choice([-1.0, 1.0], size=n)
        perm = float((diffs * signs).mean())
        if abs(perm) >= abs(obs):
            count += 1
    return float((count + 1) / (iters + 1))


def summarize_paired(baseline_correct: np.ndarray, treat_correct: np.ndarray, bootstrap_iters: int, perm_iters: int, alpha: float, seed: int) -> Dict[str, Any]:
    md, lo, hi = paired_bootstrap_ci_diff(baseline_correct, treat_correct, iters=bootstrap_iters, alpha=alpha, seed=seed + 123)
    p = signflip_permutation_test(baseline_correct, treat_correct, iters=perm_iters, seed=seed + 456)
    return {"mean_diff": md, "ci_low": lo, "ci_high": hi, "p_value": p}


def fmt_acc(acc: float, lo: float, hi: float) -> str:
    return f"{acc*100:.1f} [{lo*100:.1f}, {hi*100:.1f}]"


# -----------------------------
# Task loading (9 benchmarks)
# -----------------------------
TASKS_9 = [
    "gsm8k",
    "commonsenseqa",
    "strategyqa",
    "aqua",
    "arc_challenge",
    "openbookqa",
    "qasc",
    "logiqa",
    "boolq",
]
TASK_DEFAULT_9 = "gsm8k,commonsenseqa,strategyqa,aqua,arc_challenge,openbookqa,qasc,logiqa,boolq"


def load_selected_tasks_9(
    tasks: List[str],
    n_prompts: int,
    eval_n: int,
    *,
    seed: int,
    template_seed: int,
    template_randomization: bool,
    shuffle_choices: bool,
    add_answer_prefix: bool,
    answer_prefix: str,
) -> Tuple[Dict[str, List[Example]], Dict[str, List[Example]], Dict[str, Any]]:
    """
    Compatibility wrapper to keep the rest of the script unchanged.

    Uses benchmark_dataloaders.load_selected_tasks():
      - n_prompts -> n_subspace
      - eval_n    -> n_eval
    """
    print(f"[Data] Loading tasks via benchmark_dataloaders: tasks={tasks}")
    sub_by, eval_by, meta_by = bdl_load_selected_tasks(
        tasks=tasks,
        n_subspace=n_prompts,
        n_eval=eval_n,
        seed=seed,
        template_randomization=template_randomization,
        template_seed=template_seed,
        shuffle_choices=shuffle_choices,
        add_answer_prefix=add_answer_prefix,
        answer_prefix=answer_prefix,
    )

    for name in tasks:
        sub_exs = sub_by.get(name, [])
        eval_exs = eval_by.get(name, [])
        meta = meta_by.get(name, {})
        print(f"[Data] {name}: subspace={len(sub_exs)} eval={len(eval_exs)} meta={meta}")

    return sub_by, eval_by, meta_by


# -----------------------------
# Orthonormal basis + subspace diagnostics
# -----------------------------
def orthonormalize_np(basis: np.ndarray) -> np.ndarray:
    q, _ = np.linalg.qr(basis.astype(np.float32, copy=False))
    return q.astype(np.float32, copy=False)


def random_orthonormal_basis_np(dim: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((dim, k), dtype=np.float32)
    return orthonormalize_np(a)


def subspace_similarity(Qa: np.ndarray, Qb: np.ndarray) -> Dict[str, float]:
    M = Qa.T @ Qb
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    return {
        "max_cos": float(np.max(s)) if s.size else float("nan"),
        "mean_cos": float(np.mean(s)) if s.size else float("nan"),
        "min_cos": float(np.min(s)) if s.size else float("nan"),
        "fro_norm": float(np.linalg.norm(M, ord="fro")),
    }


def energy_ratio_stats(states: np.ndarray, Q: np.ndarray) -> Dict[str, float]:
    eps = 1e-12
    H = states.astype(np.float32, copy=False)
    proj = H @ Q
    num = np.sum(proj * proj, axis=1)
    den = np.sum(H * H, axis=1) + eps
    r = num / den
    return {"mean": float(np.mean(r)), "p50": float(np.percentile(r, 50)), "p95": float(np.percentile(r, 95))}


# -----------------------------
# Decode vs prefill activation collectors
# -----------------------------
from collections import defaultdict
from typing import DefaultDict


class DecodeLastTokenCollector:
    """Collect last-token hidden states during decode passes only (seq_len==1)."""

    def __init__(self, layer_indices: List[int]):
        self.layer_indices = list(layer_indices)
        self._cur_task: Optional[str] = None
        self.capture_enabled: bool = False
        self.active_mask: Optional[torch.Tensor] = None
        self.storage: DefaultDict[str, DefaultDict[int, List[np.ndarray]]] = defaultdict(lambda: defaultdict(list))

    def set_current_task(self, task: str) -> None:
        self._cur_task = task

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
            x = hs[:, -1, :]
            if self.active_mask is not None:
                m = self.active_mask.bool()
                if m.numel() == x.shape[0]:
                    x = x[m]
            if x.numel() == 0:
                return output
            self.storage[self._cur_task][layer_idx].append(x.detach().float().cpu().numpy())
            return output

        return _hook

    def get(self, task: str, layer_idx: int) -> Optional[np.ndarray]:
        chunks = self.storage.get(task, {}).get(layer_idx, [])
        if not chunks:
            return None
        return np.concatenate(chunks, axis=0)


class PrefillLastTokenCollector:
    """Collect last-token hidden states during prefill passes only (seq_len>1)."""

    def __init__(self, layer_indices: List[int]):
        self.layer_indices = list(layer_indices)
        self._cur_task: Optional[str] = None
        self.capture_enabled: bool = True
        self.storage: DefaultDict[str, DefaultDict[int, List[np.ndarray]]] = defaultdict(lambda: defaultdict(list))

    def set_current_task(self, task: str) -> None:
        self._cur_task = task

    def make_hook(self, layer_idx: int):
        def _hook(module, inputs, output):
            if (not self.capture_enabled) or (self._cur_task is None):
                return output
            hs = output[0] if isinstance(output, tuple) else output
            if not isinstance(hs, torch.Tensor) or hs.ndim != 3:
                return output
            if hs.shape[1] <= 1:
                return output
            x = hs[:, -1, :]
            if x.numel() == 0:
                return output
            self.storage[self._cur_task][layer_idx].append(x.detach().float().cpu().numpy())
            return output

        return _hook

    def get(self, task: str, layer_idx: int) -> Optional[np.ndarray]:
        chunks = self.storage.get(task, {}).get(layer_idx, [])
        if not chunks:
            return None
        return np.concatenate(chunks, axis=0)


def _subsample_rows_np(x: np.ndarray, n_max: int, seed: int) -> np.ndarray:
    if n_max is None or n_max <= 0 or x.shape[0] <= n_max:
        return x
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.shape[0], size=n_max, replace=False)
    return x[idx]


# -----------------------------
# Decode collection loop + decode-aligned prompt boundary
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


def _choose_next_token(
    logits: torch.Tensor,
    *,
    decoding: str,
    temperature: float,
    top_p: float,
    top_k: int,
    eos_token_id: int,
    ban_eos: bool,
) -> torch.Tensor:
    """Return next_token [B,1]. If ban_eos, avoid selecting EOS."""
    assert decoding in ["greedy", "sample"]
    if ban_eos:
        logits = logits.clone()
        logits[:, eos_token_id] = float("-inf")

    if decoding == "greedy":
        return torch.argmax(logits, dim=-1, keepdim=True)

    lt = logits / max(temperature, 1e-6)
    lt = top_k_filtering(lt, top_k)
    lt = top_p_filtering(lt, top_p)
    probs = torch.softmax(lt, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def _cache_advanced_prompt_boundary(model, ids: torch.Tensor, attn: torch.Tensor):
    """Compute (past, logits_next) such that the last prompt token is processed with seq_len==1."""
    if ids.ndim != 2:
        raise ValueError(f"ids must be 2D [B,T], got {ids.shape}")
    _, T = ids.shape
    if T == 0:
        raise ValueError("Empty prompt")
    if T == 1:
        out1 = model(input_ids=ids, attention_mask=attn, use_cache=True)
        return out1.past_key_values, out1.logits[:, -1, :]
    out0 = model(input_ids=ids[:, :-1], attention_mask=attn[:, :-1], use_cache=True)
    out1 = model(input_ids=ids[:, -1:], attention_mask=attn, use_cache=True, past_key_values=out0.past_key_values)
    return out1.past_key_values, out1.logits[:, -1, :]


@torch.no_grad()
def collect_decode_states(
    model,
    tok,
    prompts: List[str],
    collector: DecodeLastTokenCollector,
    *,
    batch_size: int,
    max_new_tokens: int,
    max_prompt_len: int,
    decoding: str,
    temperature: float,
    top_p: float,
    top_k: int,
) -> None:
    device = next(model.parameters()).device
    eos = tok.eos_token_id
    model.eval()
    for i in tqdm(range(0, len(prompts), batch_size), desc="CollectDecode"):
        batch = prompts[i : i + batch_size]
        inputs = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_prompt_len).to(device)
        ids = inputs["input_ids"]
        attn = inputs["attention_mask"]
        B = ids.shape[0]

        unfinished = torch.ones(B, dtype=torch.bool, device=device)

        collector.set_capture(False, None)
        past, logits = _cache_advanced_prompt_boundary(model, ids, attn)

        for _ in range(max_new_tokens):
            next_tok = _choose_next_token(
                logits,
                decoding=decoding,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                eos_token_id=eos,
                ban_eos=False,
            )
            next_tok = torch.where(unfinished.unsqueeze(-1), next_tok, torch.full_like(next_tok, eos))
            unfinished = unfinished & (next_tok.squeeze(-1) != eos)
            if not bool(unfinished.any().item()):
                break
            attn = torch.cat([attn, torch.ones((B, 1), device=device, dtype=attn.dtype)], dim=1)

            collector.set_capture(True, unfinished)
            out = model(input_ids=next_tok, attention_mask=attn, use_cache=True, past_key_values=past)
            logits = out.logits[:, -1, :]
            past = out.past_key_values

        collector.set_capture(False, None)


@torch.no_grad()
def collect_prefill_states(
    model,
    tok,
    prompts: List[str],
    collector: PrefillLastTokenCollector,
    *,
    batch_size: int,
    max_prompt_len: int,
) -> None:
    device = next(model.parameters()).device
    model.eval()
    for i in tqdm(range(0, len(prompts), batch_size), desc="CollectPrefill"):
        batch = prompts[i : i + batch_size]
        inputs = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_prompt_len).to(device)
        _ = model(**inputs)


def _balanced_concat(task_to_states: Dict[str, np.ndarray], seed: int) -> Tuple[np.ndarray, int]:
    sizes = {t: v.shape[0] for t, v in task_to_states.items()}
    n_min = min(sizes.values())
    out = []
    for t, X in task_to_states.items():
        rng = np.random.default_rng(stable_int_seed(seed, t, "bal"))
        if X.shape[0] > n_min:
            idx = rng.choice(X.shape[0], size=n_min, replace=False)
            out.append(X[idx])
        else:
            out.append(X)
    return np.concatenate(out, axis=0), int(n_min)


def compute_shared_basis_from_states(
    task_states: Dict[str, np.ndarray],
    *,
    pca_var: float,
    min_dim: int,
    max_dim: int,
    tau: float,
    m_shared: str,
    seed: int,
) -> Tuple[np.ndarray, List[int], Dict[str, Any]]:
    """Compute joint subspace + shared indices from task->states dict (single layer)."""
    _X_joint, n_bal = _balanced_concat(task_states, seed)
    tasks = list(task_states.keys())
    task_dict = {t: {0: task_states[t]} for t in tasks}

    joint_subspace, cross_dim, contributions, full_pca_info = compute_cross_task_subspace(
        task_dict,
        variance_threshold=pca_var,
        min_dim=min_dim,
        max_dim=max_dim,
        return_full_pca=True,
    )
    if joint_subspace is None or cross_dim <= 0:
        raise RuntimeError("Failed to compute cross-task subspace")

    min_tasks_shared = len(tasks) if m_shared == "all" else int(m_shared)
    shared_idx = find_fully_shared_basis_improved(
        contributions,
        tasks,
        cross_dim,
        min_tasks_shared=min_tasks_shared,
        relative_threshold=tau,
        top_k_components=cross_dim,
    )

    extra = {
        "tasks_used": tasks,
        "n_balanced": int(n_bal),
        "cross_dim": int(cross_dim),
        "task_contributions": contributions,
        "full_pca_info": full_pca_info,
    }
    return joint_subspace.astype(np.float32, copy=False), shared_idx, extra


# -----------------------------
# Intervention hooks (decode-only)
# -----------------------------
class HookStats:
    def __init__(self, name: str):
        self.name = name
        self.decode_calls = 0
        self.intervened = 0

    def report(self):
        return {"name": self.name, "decode_calls": int(self.decode_calls), "intervened": int(self.intervened)}


class LastTokenRemovalHook:
    """Remove projection onto Q on decode passes only (seq_len==1)."""

    def __init__(self, Q_np: np.ndarray, alpha: float, stats: HookStats):
        self.alpha = float(alpha)
        self.stats = stats
        self.enabled = True
        self.Q_cpu = torch.tensor(orthonormalize_np(Q_np), dtype=torch.float32)
        self.Q_dev: Optional[torch.Tensor] = None

    def set_enabled(self, flag: bool) -> None:
        self.enabled = bool(flag)

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

        self.stats.decode_calls += 1
        if not self.enabled:
            return output

        Q = self._Q(hs.device)
        x = hs[:, -1, :].float()
        proj = (x @ Q) @ Q.T
        hs2 = hs.clone()
        hs2[:, -1, :] = (x - self.alpha * proj).to(dtype=hs.dtype)
        self.stats.intervened += 1
        if isinstance(output, tuple):
            return (hs2,) + output[1:]
        return hs2


def register_hooks(model, layer_indices: List[int], basis_np: Optional[np.ndarray], alpha: float, name: str):
    if basis_np is None:
        return [], [], HookStats(name), (lambda flag: None)
    layers, _ = get_model_layers(model)
    stats = HookStats(name)
    handles = []
    hooks = []
    for li in layer_indices:
        hk = LastTokenRemovalHook(basis_np, alpha, stats)
        hooks.append(hk)
        handles.append(layers[li].register_forward_hook(hk))

    def toggle(flag: bool):
        for hk in hooks:
            hk.set_enabled(flag)

    return handles, hooks, stats, toggle


def remove_hooks(handles):
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass


# -----------------------------
# Forced-choice logprob eval (decode-aligned) + warmup
# -----------------------------
def candidate_strings(task: str) -> List[str]:
    """
    Candidate labels/strings used for forced-choice scoring.

    These should be consistent with benchmark_dataloaders' MC label conventions.
    """
    task = (task or "").strip().lower()
    if task in ["commonsenseqa", "aqua"]:
        return list("ABCDE")
    if task in ["arc_challenge", "openbookqa", "logiqa"]:
        return list("ABCD")
    if task == "qasc":
        return list("ABCDEFGH")
    if task == "boolq":
        # benchmark_dataloaders boolq gold is A/B (A=Yes, B=No)
        return ["A", "B"]
    if task == "strategyqa":
        # benchmark_dataloaders strategyqa gold normalized to YES/NO
        return ["Yes", "No"]
    return []


def _normalize_answer_prefix(prefix: str) -> str:
    """Match benchmark_dataloaders behavior a bit: treat '0/none/null/false' as empty."""
    if prefix is None:
        return ""
    s = str(prefix)
    if s.strip().lower() in {"0", "none", "null", "false"}:
        return ""
    # keep as-is; do not force leading newline here (some users may want none)
    return s


def _prompt_endswith_prefix(prompt: str, prefix: str) -> bool:
    if not prefix:
        return False
    p = (prompt or "").rstrip()
    ap = (prefix or "").rstrip()
    return p.endswith(ap)


def cand_token_ids(tok, s: str) -> List[int]:
    """
    Tokenize a candidate as it would usually appear after 'Final answer:'.

    We intentionally include a leading space so that SentencePiece tokenizers
    produce the correct "word-start" token (e.g., ▁A, ▁Yes).
    """
    ids = tok.encode(" " + s, add_special_tokens=False)
    if not ids:
        ids = tok.encode(s, add_special_tokens=False)
    return ids


@torch.no_grad()
def precompute_fc_warmup_tokens(
    model,
    tok,
    prompts: List[str],
    *,
    warmup_tokens: int,
    batch_size: int,
    max_prompt_len: int,
    decoding: str,
    temperature: float,
    top_p: float,
    top_k: int,
    ban_eos: bool,
    seed: int,
) -> np.ndarray:
    """Generate W warmup tokens under baseline (no intervention). Returns [N,W] int64 on CPU."""
    assert warmup_tokens >= 0
    if warmup_tokens == 0:
        return np.zeros((len(prompts), 0), dtype=np.int64)

    device = next(model.parameters()).device
    eos = tok.eos_token_id
    model.eval()

    if decoding == "sample":
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    out_tokens = np.zeros((len(prompts), warmup_tokens), dtype=np.int64)

    for i in tqdm(range(0, len(prompts), batch_size), desc=f"WarmupGen(W={warmup_tokens})"):
        batch = prompts[i : i + batch_size]
        inputs = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_prompt_len).to(device)
        ids = inputs["input_ids"]
        attn = inputs["attention_mask"]
        B = ids.shape[0]

        past, logits = _cache_advanced_prompt_boundary(model, ids, attn)

        toks = []
        for _ in range(warmup_tokens):
            next_tok = _choose_next_token(
                logits,
                decoding=decoding,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                eos_token_id=eos,
                ban_eos=ban_eos,
            )
            toks.append(next_tok)
            attn = torch.cat([attn, torch.ones((B, 1), device=device, dtype=attn.dtype)], dim=1)
            out = model(input_ids=next_tok, attention_mask=attn, use_cache=True, past_key_values=past)
            logits = out.logits[:, -1, :]
            past = out.past_key_values

        toks_mat = torch.cat(toks, dim=1)  # [B,W]
        out_tokens[i : i + B, :] = toks_mat.detach().cpu().numpy().astype(np.int64, copy=False)

    return out_tokens


@torch.no_grad()
def forced_choice_logprob_eval(
    model,
    tok,
    examples: List[Example],
    task: str,
    *,
    layer_indices: List[int],
    basis_np: Optional[np.ndarray],
    alpha: float,
    batch_size: int,
    max_prompt_len: int,
    warmup_token_ids: Optional[np.ndarray],
    answer_prefix: str,
    prefix_mode: str,  # auto|always|never
) -> Dict[str, Any]:
    """Forced-choice accuracy by logprob (decode-aligned) with optional warmup teacher-forcing."""
    prefix_mode = (prefix_mode or "auto").strip().lower()
    if prefix_mode not in {"auto", "always", "never"}:
        raise ValueError(f"Unknown prefix_mode={prefix_mode!r}")

    device = next(model.parameters()).device
    model.eval()
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    prompts = [ex.prompt for ex in examples]
    golds = [ex.gold for ex in examples]
    cands = candidate_strings(task)
    if len(cands) == 0:
        raise ValueError(f"Task '{task}' has no forced-choice candidates. Use generation eval instead.")
    cand_ids_list = [cand_token_ids(tok, s) for s in cands]

    answer_prefix = _normalize_answer_prefix(answer_prefix)

    handles, _hooks, stats, _toggle = register_hooks(
        model,
        layer_indices=layer_indices,
        basis_np=basis_np,
        alpha=alpha,
        name=f"fc_full@{layer_indices[0]}",
    )

    correct = np.zeros(len(prompts), dtype=np.float32)

    try:
        for i in tqdm(range(0, len(prompts), batch_size), desc=f"ForcedChoice({task})"):
            batch_prompts = prompts[i : i + batch_size]
            batch_golds = golds[i : i + batch_size]
            inputs = tok(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_prompt_len).to(device)
            ids = inputs["input_ids"]
            attn = inputs["attention_mask"]
            B = ids.shape[0]

            warm_ids = None
            W = 0
            if warmup_token_ids is not None:
                warm = warmup_token_ids[i : i + B]
                if warm is not None:
                    warm_ids = torch.tensor(warm, dtype=torch.long, device=device)
                    W = int(warm_ids.shape[1])

            past, logits = _cache_advanced_prompt_boundary(model, ids, attn)

            # Teacher-force warmup tokens before scoring candidates
            if warm_ids is not None and W > 0:
                for t in range(W):
                    tok_t = warm_ids[:, t : t + 1]
                    attn = torch.cat([attn, torch.ones((B, 1), device=device, dtype=attn.dtype)], dim=1)
                    out = model(input_ids=tok_t, attention_mask=attn, use_cache=True, past_key_values=past)
                    logits = out.logits[:, -1, :]
                    past = out.past_key_values

            # Decide whether to teacher-force answer_prefix before scoring candidates.
            # auto policy:
            #   - if warmup was used, ALWAYS add prefix (decision point moved; otherwise scoring is near-chance)
            #   - if no warmup, add only if prompt doesn't already end with prefix
            do_prefix = False
            if prefix_mode == "always":
                do_prefix = bool(answer_prefix)
            elif prefix_mode == "never":
                do_prefix = False
            else:  # auto
                if bool(answer_prefix):
                    if W > 0:
                        do_prefix = True
                    else:
                        # Per-task prompts are usually consistent; check the first prompt in batch.
                        do_prefix = not _prompt_endswith_prefix(batch_prompts[0], answer_prefix)

            if do_prefix and answer_prefix:
                prefix_ids = tok.encode(answer_prefix, add_special_tokens=False)
                if len(prefix_ids) > 0:
                    for pid in prefix_ids:
                        inp = torch.full((B, 1), pid, dtype=torch.long, device=device)
                        attn = torch.cat([attn, torch.ones((B, 1), device=device, dtype=attn.dtype)], dim=1)
                        out = model(input_ids=inp, attention_mask=attn, use_cache=True, past_key_values=past)
                        logits = out.logits[:, -1, :]
                        past = out.past_key_values

            # Score candidates by logprob
            scores = torch.zeros(B, len(cands), device=device)
            for ci, cand_ids in enumerate(cand_ids_list):
                if len(cand_ids) == 0:
                    scores[:, ci] = float("-inf")
                    continue

                past_c = past
                attn_c = attn
                logits_c = logits

                lp = torch.zeros(B, device=device)
                for ti, tok_id in enumerate(cand_ids):
                    logp = torch.log_softmax(logits_c, dim=-1)
                    lp = lp + logp[:, tok_id]
                    if ti < len(cand_ids) - 1:
                        inp = torch.full((B, 1), tok_id, dtype=torch.long, device=device)
                        attn_c = torch.cat([attn_c, torch.ones((B, 1), device=device, dtype=attn_c.dtype)], dim=1)
                        out = model(input_ids=inp, attention_mask=attn_c, use_cache=True, past_key_values=past_c)
                        logits_c = out.logits[:, -1, :]
                        past_c = out.past_key_values

                scores[:, ci] = lp

            pred_idx = torch.argmax(scores, dim=1).detach().cpu().numpy().tolist()
            preds = [cands[j] for j in pred_idx]
            for b, (pred, gold) in enumerate(zip(preds, batch_golds)):
                # Robust normalization from benchmark_dataloaders
                correct[i + b] = 1.0 if is_correct_any(task, pred, gold) else 0.0

        return {"acc": float(correct.mean()), "correct": correct.tolist(), "hook_stats": stats.report()}
    finally:
        remove_hooks(handles)


# -----------------------------
# Free-form generation eval (for gsm8k, or if --do_generation 1)
# -----------------------------
@torch.no_grad()
def generate_decode_aligned(
    model,
    tok,
    prompts: List[str],
    *,
    batch_size: int,
    max_new_tokens: int,
    max_prompt_len: int,
    decoding: str,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: Optional[int] = None,
) -> List[str]:
    """Generate continuations using decode-aligned prompt boundary caching (seq_len==1 decode)."""
    assert decoding in ["greedy", "sample"]
    device = next(model.parameters()).device
    eos = tok.eos_token_id
    model.eval()

    if decoding == "sample" and seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    outs: List[str] = []
    for i in tqdm(range(0, len(prompts), batch_size), desc=f"Gen({decoding})"):
        batch = prompts[i : i + batch_size]
        inputs = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_prompt_len).to(device)
        ids = inputs["input_ids"]
        attn = inputs["attention_mask"]
        B = ids.shape[0]

        past, logits = _cache_advanced_prompt_boundary(model, ids, attn)

        unfinished = torch.ones(B, dtype=torch.bool, device=device)
        gen_steps = torch.zeros(B, dtype=torch.long, device=device)
        gen = torch.full((B, max_new_tokens), eos, dtype=torch.long, device=device)

        for t in range(max_new_tokens):
            next_tok = _choose_next_token(
                logits,
                decoding=decoding,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                eos_token_id=eos,
                ban_eos=False,
            )
            next_tok = torch.where(unfinished.unsqueeze(-1), next_tok, torch.full_like(next_tok, eos))
            tok_ids = next_tok.squeeze(-1)

            gen[unfinished, t] = tok_ids[unfinished]
            gen_steps[unfinished] += 1
            unfinished = unfinished & (tok_ids != eos)
            if not bool(unfinished.any().item()):
                break

            attn = torch.cat([attn, torch.ones((B, 1), device=device, dtype=attn.dtype)], dim=1)
            out = model(input_ids=next_tok, attention_mask=attn, use_cache=True, past_key_values=past)
            logits = out.logits[:, -1, :]
            past = out.past_key_values

        for b in range(B):
            L = int(gen_steps[b].item())
            cont_ids = gen[b, :L].tolist()
            txt = tok.decode(cont_ids, skip_special_tokens=True)
            outs.append(txt)

    return outs


@torch.no_grad()
def generation_eval(
    model,
    tok,
    examples: List[Example],
    task: str,
    *,
    layer_indices: List[int],
    basis_np: Optional[np.ndarray],
    alpha: float,
    batch_size: int,
    max_prompt_len: int,
    max_new_tokens: int,
    decoding: str,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
) -> Dict[str, Any]:
    """Free-form generation accuracy using benchmark_dataloaders.parse_prediction/is_correct."""
    prompts = [ex.prompt for ex in examples]
    golds = [ex.gold for ex in examples]

    handles, _hooks, stats, _toggle = register_hooks(
        model,
        layer_indices=layer_indices,
        basis_np=basis_np,
        alpha=alpha,
        name=f"gen_full@{layer_indices[0]}",
    )

    try:
        conts = generate_decode_aligned(
            model,
            tok,
            prompts,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            max_prompt_len=max_prompt_len,
            decoding=decoding,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
        )
        correct = np.zeros(len(examples), dtype=np.float32)
        extracted = np.zeros(len(examples), dtype=np.float32)

        for i, (cont, gold) in enumerate(zip(conts, golds)):
            pred = parse_prediction_generation(task, cont)
            extracted[i] = 1.0 if pred != "" else 0.0
            correct[i] = 1.0 if is_correct_any(task, pred, gold) else 0.0

        return {
            "acc": float(correct.mean()),
            "correct": correct.tolist(),
            "extraction_rate": float(extracted.mean()),
            "hook_stats": stats.report(),
        }
    finally:
        remove_hooks(handles)


# -----------------------------
# Build dimension-matched bases
# -----------------------------
def _build_shared_basis_from_joint(joint: np.ndarray, shared_idx: List[int], k: int) -> np.ndarray:
    if k <= 0:
        raise ValueError("k must be positive")
    if len(shared_idx) < k:
        raise ValueError(f"Need at least {k} shared components, got {len(shared_idx)}")
    idx = sorted(shared_idx)[:k]
    return orthonormalize_np(joint[:, idx])


# -----------------------------
# Model load
# -----------------------------
def load_model_and_tokenizer(model_name: str, device: str, dtype: str, trust_remote_code: bool = False):
    if dtype == "fp32":
        torch_dtype = torch.float32
    elif dtype == "fp16":
        torch_dtype = torch.float16
    elif dtype == "bf16":
        torch_dtype = torch.bfloat16
    else:
        raise ValueError(f"Unknown dtype: {dtype}")

    model_kwargs = {"torch_dtype": torch_dtype, "trust_remote_code": trust_remote_code}
    tok_kwargs = {"trust_remote_code": trust_remote_code}

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    tok = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)

    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = model.to(device)
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    return model, tok


# -----------------------------
# Markdown/LaTeX table helpers
# -----------------------------
def md_table(rows: List[List[str]], header: List[str]) -> str:
    out = []
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "---|" * len(header))
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def latex_table(rows: List[List[str]], header: List[str], caption: str, label: str, colspec: str) -> str:
    def esc(s: str) -> str:
        return s.replace("%", "\\%").replace("_", "\\_")

    header_esc = [esc(h) for h in header]
    body = []
    for r in rows:
        body.append(" & ".join(esc(x) for x in r) + " \\\\")
    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\small\n"
        f"\\begin{{tabular}}{{{colspec}}}\n"
        "\\toprule\n"
        + " & ".join(header_esc)
        + " \\\\\n\\midrule\n"
        + "\n".join(body)
        + "\n\\bottomrule\n"
        "\\end{tabular}\n"
        f"\\caption{{{esc(caption)}}}\n"
        f"\\label{{{esc(label)}}}\n"
        "\\end{table}\n"
    )


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--trust_remote_code", type=int, default=0, choices=[0, 1])

    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--tasks", type=str, default=TASK_DEFAULT_9, help=f"Comma-separated tasks. Default: {TASK_DEFAULT_9}")

    ap.add_argument("--n_prompts", type=int, default=128)
    ap.add_argument("--eval_n", type=int, default=256)

    ap.add_argument("--calib_decode_max_new_tokens", type=int, default=128)
    ap.add_argument("--per_task_max_states", type=int, default=20000)

    ap.add_argument("--pca_var", type=float, default=0.95)
    ap.add_argument("--min_dim", type=int, default=1)
    ap.add_argument("--max_dim", type=int, default=4096)
    ap.add_argument("--tau", type=float, default=0.001)
    ap.add_argument("--m_shared", type=str, default="all")

    ap.add_argument("--alpha_remove", type=float, default=1.0)

    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_prompt_len", type=int, default=512)

    ap.add_argument("--bootstrap_iters", type=int, default=5000)
    ap.add_argument("--perm_iters", type=int, default=10000)
    ap.add_argument("--ci_alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--do_generation", type=int, default=0, choices=[0, 1])
    ap.add_argument("--match_state_count", type=int, default=0, choices=[0, 1])

    # Prompt-building knobs (passed through to benchmark_dataloaders)
    ap.add_argument("--template_randomization", type=int, default=0, choices=[0, 1])
    ap.add_argument("--template_seed", type=int, default=1234)
    ap.add_argument("--shuffle_choices", type=int, default=0, choices=[0, 1])
    ap.add_argument("--add_answer_prefix", type=int, default=0, choices=[0, 1])
    ap.add_argument("--answer_prefix", type=str, default="\nFinal answer:")

    # Warmup-forced-choice
    ap.add_argument("--fc_warmup_tokens", type=int, default=0)
    ap.add_argument("--fc_warmup_decoding", type=str, default="greedy", choices=["greedy", "sample"])
    ap.add_argument("--fc_warmup_seed", type=int, default=123)
    ap.add_argument("--fc_warmup_ban_eos", type=int, default=1)

    # NEW: prefix policy for forced-choice scoring
    ap.add_argument("--fc_prefix_mode", type=str, default="auto", choices=["auto", "always", "never"])
    ap.add_argument("--fc_answer_prefix", type=str, default="\nFinal answer:")

    # Backward compatibility (deprecated)
    ap.add_argument(
        "--fc_add_answer_prefix",
        type=int,
        default=None,
        choices=[0, 1],
        help="DEPRECATED. Overrides --fc_prefix_mode: 1->always, 0->auto (old scripts used 0 to mean never; now use --fc_prefix_mode never).",
    )

    # Generation-eval decoding knobs (only used when --do_generation=1 or task has no candidates)
    ap.add_argument("--gen_decoding", type=str, default="greedy", choices=["greedy", "sample"])
    ap.add_argument("--gen_temperature", type=float, default=0.7)
    ap.add_argument("--gen_top_p", type=float, default=0.9)
    ap.add_argument("--gen_top_k", type=int, default=0)
    ap.add_argument("--gen_max_new_tokens", type=int, default=256)
    ap.add_argument("--gen_seed", type=int, default=12345)

    ap.add_argument("--out_json", type=str, default=os.path.join(THIS_DIR, "prefill_vs_decode_alignment_9tasks_fixed.json"))
    ap.add_argument("--out_txt", type=str, default=os.path.join(THIS_DIR, "prefill_vs_decode_alignment_9tasks_fixed.txt"))
    ap.add_argument("--out_md", type=str, default=os.path.join(THIS_DIR, "prefill_vs_decode_alignment_9tasks_fixed.md"))
    ap.add_argument("--out_tex", type=str, default=os.path.join(THIS_DIR, "prefill_vs_decode_alignment_9tasks_fixed.tex"))

    args = ap.parse_args()
    set_global_seed(args.seed)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if len(tasks) < 2:
        raise RuntimeError("Need at least 2 tasks in --tasks.")
    for t in tasks:
        if t not in TASKS_9:
            raise ValueError(f"Unknown task '{t}'. Supported: {sorted(TASKS_9)}")

    # Resolve forced-choice prefix policy with backward-compatible flag
    fc_prefix_mode = (args.fc_prefix_mode or "auto").strip().lower()
    if args.fc_add_answer_prefix is not None:
        fc_prefix_mode = "always" if int(args.fc_add_answer_prefix) == 1 else "auto"

    if (not bool(args.do_generation)) and fc_prefix_mode == "never":
        # Configuration warning: near-chance is expected unless prompts end with answer_prefix already.
        print("[WARN] Forced-choice prefix_mode=never. If your prompts do NOT already end with an explicit answer prefix, forced-choice accuracy will often be near chance.")

    if (not bool(args.do_generation)) and (args.fc_warmup_tokens > 0) and (fc_prefix_mode == "never"):
        print("[WARN] fc_warmup_tokens>0 with prefix_mode=never is *almost guaranteed* to produce near-chance accuracy, because you're scoring labels at an arbitrary deep decode position.")

    layer_indices = [args.layer]
    print(f"[Env] model={args.model} device={args.device} dtype={args.dtype} layer={layer_indices} trust_remote_code={bool(args.trust_remote_code)}")
    print(f"[Env] tasks={tasks}")
    print(f"[Env] prompt template_randomization={bool(args.template_randomization)} shuffle_choices={bool(args.shuffle_choices)} add_answer_prefix={bool(args.add_answer_prefix)}")
    if not bool(args.do_generation):
        print(f"[Env] forced-choice warmup_tokens={args.fc_warmup_tokens} prefix_mode={fc_prefix_mode} answer_prefix={args.fc_answer_prefix!r}")

    model, tok = load_model_and_tokenizer(args.model, args.device, args.dtype, trust_remote_code=bool(args.trust_remote_code))

    layers, _ = get_model_layers(model)
    if args.layer < 0 or args.layer >= len(layers):
        raise ValueError(f"--layer {args.layer} out of range for this model. num_layers={len(layers)}")

    hidden_dim = getattr(model.config, "hidden_size", None) or getattr(model.config, "n_embd", None)
    if hidden_dim is None:
        raise RuntimeError("Could not infer hidden_dim from model.config")
    print(f"[Env] hidden_dim={hidden_dim} num_layers={len(layers)}")

    # Load datasets (benchmark_dataloaders)
    sub_by, eval_by, meta_by = load_selected_tasks_9(
        tasks=tasks,
        n_prompts=args.n_prompts,
        eval_n=args.eval_n,
        seed=args.seed,
        template_seed=args.template_seed,
        template_randomization=bool(args.template_randomization),
        shuffle_choices=bool(args.shuffle_choices),
        add_answer_prefix=bool(args.add_answer_prefix),
        answer_prefix=args.answer_prefix,
    )

    # 1) Collect DECODE states
    print("\n" + "=" * 80)
    print("[Basis] Estimating SHARED basis on D_decode (seq_len==1 decode steps)")
    print("=" * 80)

    decode_col = DecodeLastTokenCollector(layer_indices)
    handles = []
    for li in layer_indices:
        handles.append(layers[li].register_forward_hook(decode_col.make_hook(li)))
    try:
        decode_task_states: Dict[str, np.ndarray] = {}
        for task, sub_exs in sub_by.items():
            decode_col.set_current_task(task)
            prompts = [ex.prompt for ex in sub_exs]
            collect_decode_states(
                model,
                tok,
                prompts,
                decode_col,
                batch_size=args.batch_size,
                max_new_tokens=args.calib_decode_max_new_tokens,
                max_prompt_len=args.max_prompt_len,
                decoding="greedy",
                temperature=1.0,
                top_p=1.0,
                top_k=0,
            )
            X = decode_col.get(task, layer_indices[0])
            if X is None:
                raise RuntimeError(f"No decode states for task={task}")
            X = _subsample_rows_np(X, args.per_task_max_states, seed=stable_int_seed(args.seed, task, "decode"))
            decode_task_states[task] = X
            print(f"[Collect][decode] task={task} states={X.shape[0]} x {X.shape[1]}")
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass
        decode_col.set_capture(False, None)

    # 2) Collect PREFILL states
    print("\n" + "=" * 80)
    print("[Basis] Estimating SHARED basis on D_prefill (seq_len>1 prefill tokens)")
    print("=" * 80)

    pre_col = PrefillLastTokenCollector(layer_indices)
    handles = []
    for li in layer_indices:
        handles.append(layers[li].register_forward_hook(pre_col.make_hook(li)))
    try:
        pre_task_states: Dict[str, np.ndarray] = {}
        for task, sub_exs in sub_by.items():
            pre_col.set_current_task(task)
            prompts = [ex.prompt for ex in sub_exs]
            collect_prefill_states(model, tok, prompts, pre_col, batch_size=args.batch_size, max_prompt_len=args.max_prompt_len)
            X = pre_col.get(task, layer_indices[0])
            if X is None:
                raise RuntimeError(f"No prefill states for task={task}")
            X = _subsample_rows_np(X, args.per_task_max_states, seed=stable_int_seed(args.seed, task, "prefill"))
            pre_task_states[task] = X
            print(f"[Collect][prefill] task={task} states={X.shape[0]} x {X.shape[1]}")
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass

    # Optional state-count matching for basis estimation
    state_match_info = {"enabled": bool(args.match_state_count)}
    if args.match_state_count:
        n_decode_min = min(v.shape[0] for v in decode_task_states.values())
        n_prefill_min = min(v.shape[0] for v in pre_task_states.values())
        n_match = min(n_decode_min, n_prefill_min)
        state_match_info.update({"n_decode_min": int(n_decode_min), "n_prefill_min": int(n_prefill_min), "n_match": int(n_match)})
        print("\n" + "=" * 80)
        print("[Basis] Re-estimating BOTH bases with STATE-COUNT matching")
        print(f"  decode n_min={n_decode_min} prefill n_min={n_prefill_min} => using n_match={n_match}")
        print("=" * 80)

        decode_task_states = {t: _subsample_rows_np(X, n_match, seed=stable_int_seed(args.seed, t, "decode_match")) for t, X in decode_task_states.items()}
        pre_task_states = {t: _subsample_rows_np(X, n_match, seed=stable_int_seed(args.seed, t, "prefill_match")) for t, X in pre_task_states.items()}

    # Compute bases
    joint_decode, shared_idx_decode, extra_decode = compute_shared_basis_from_states(
        decode_task_states,
        pca_var=args.pca_var,
        min_dim=args.min_dim,
        max_dim=args.max_dim,
        tau=args.tau,
        m_shared=args.m_shared,
        seed=args.seed + 100,
    )
    joint_prefill, shared_idx_prefill, extra_prefill = compute_shared_basis_from_states(
        pre_task_states,
        pca_var=args.pca_var,
        min_dim=args.min_dim,
        max_dim=args.max_dim,
        tau=args.tau,
        m_shared=args.m_shared,
        seed=args.seed + 200,
    )

    k_decode = len(shared_idx_decode)
    k_prefill = len(shared_idx_prefill)
    k_match = min(k_decode, k_prefill)
    if k_match <= 0:
        raise RuntimeError("No shared components found (k_match<=0)")

    Q_decode_full = _build_shared_basis_from_joint(joint_decode, shared_idx_decode, k_decode)
    Q_prefill_full = _build_shared_basis_from_joint(joint_prefill, shared_idx_prefill, k_prefill)
    Q_decode_km = _build_shared_basis_from_joint(joint_decode, shared_idx_decode, k_match)
    Q_prefill_km = _build_shared_basis_from_joint(joint_prefill, shared_idx_prefill, k_match)
    Q_rand_km = random_orthonormal_basis_np(hidden_dim, k_match, seed=stable_int_seed(args.seed, "rand", k_match))

    print("\n" + "=" * 80)
    print("[Diag] Subspace similarity (Q_decode_shared vs Q_prefill_shared)")
    print("=" * 80)
    print(json.dumps(subspace_similarity(Q_decode_full, Q_prefill_full), indent=2))
    print("\n[Diag] Subspace similarity (dimension-matched k_match)")
    print(json.dumps(subspace_similarity(Q_decode_km, Q_prefill_km), indent=2))

    decode_pool = np.concatenate([_subsample_rows_np(X, 4000, seed=stable_int_seed(args.seed, t, "er_d")) for t, X in decode_task_states.items()], axis=0)
    pre_pool = np.concatenate([_subsample_rows_np(X, 4000, seed=stable_int_seed(args.seed, t, "er_p")) for t, X in pre_task_states.items()], axis=0)

    er_decode_on_decode = energy_ratio_stats(decode_pool, Q_decode_full)
    er_prefill_on_decode = energy_ratio_stats(decode_pool, Q_prefill_full)
    er_prefill_on_prefill = energy_ratio_stats(pre_pool, Q_prefill_full)
    er_decode_on_prefill = energy_ratio_stats(pre_pool, Q_decode_full)

    er_decode_km_on_decode = energy_ratio_stats(decode_pool, Q_decode_km)
    er_prefill_km_on_decode = energy_ratio_stats(decode_pool, Q_prefill_km)
    er_rand_km_on_decode = energy_ratio_stats(decode_pool, Q_rand_km)
    er_prefill_km_on_prefill = energy_ratio_stats(pre_pool, Q_prefill_km)
    er_decode_km_on_prefill = energy_ratio_stats(pre_pool, Q_decode_km)

    print("\n" + "=" * 80)
    print("[Diag] Energy ratio r(h,Q)=||Q^T h||^2 / ||h||^2")
    print(f"  (FULL)   On DECODE states:  Q_decode_shared mean={er_decode_on_decode['mean']:.4f},  Q_prefill_shared mean={er_prefill_on_decode['mean']:.4f}")
    print(f"  (FULL)   On PREFILL states: Q_prefill_shared mean={er_prefill_on_prefill['mean']:.4f}, Q_decode_shared mean={er_decode_on_prefill['mean']:.4f}")
    print(f"  (k={k_match}) On DECODE states:  Q_decode_km mean={er_decode_km_on_decode['mean']:.4f},  Q_prefill_km mean={er_prefill_km_on_decode['mean']:.4f}, rand mean={er_rand_km_on_decode['mean']:.4f}")
    print(f"  (k={k_match}) On PREFILL states: Q_prefill_km mean={er_prefill_km_on_prefill['mean']:.4f}, Q_decode_km mean={er_decode_km_on_prefill['mean']:.4f}")

    # Warmup tokens per task (only used for forced-choice)
    warmup_by_task: Dict[str, np.ndarray] = {}
    if args.fc_warmup_tokens > 0 and not bool(args.do_generation):
        print("\n" + "=" * 80)
        print(f"[FC Warmup] Precomputing baseline warmup tokens: W={args.fc_warmup_tokens} (decoding={args.fc_warmup_decoding}, ban_eos={bool(args.fc_warmup_ban_eos)})")
        print("=" * 80)
        for task in tasks:
            if len(candidate_strings(task)) == 0:
                continue
            prompts = [ex.prompt for ex in eval_by[task]]
            warm_ids = precompute_fc_warmup_tokens(
                model,
                tok,
                prompts,
                warmup_tokens=args.fc_warmup_tokens,
                batch_size=args.batch_size,
                max_prompt_len=args.max_prompt_len,
                decoding=args.fc_warmup_decoding,
                temperature=0.7,
                top_p=0.9,
                top_k=0,
                ban_eos=bool(args.fc_warmup_ban_eos),
                seed=stable_int_seed(args.seed, args.fc_warmup_seed, task, "warmup"),
            )
            warmup_by_task[task] = warm_ids
            if warm_ids.shape[0] > 0 and warm_ids.shape[1] > 0:
                demo = tok.decode(warm_ids[0].tolist(), skip_special_tokens=True)
                print(f"[FC Warmup] {task}: warmup_ids shape={warm_ids.shape}; example[0] warmup text (first 120 chars): {demo[:120]!r}")

    # Evaluation across tasks
    eval_results: Dict[str, Any] = {}
    for task in tasks:
        exs = eval_by[task]
        n = len(exs)
        print("\n" + "-" * 80)
        print(f"[Task] {task} n={n}")
        print("-" * 80)

        use_generation = bool(args.do_generation) or (len(candidate_strings(task)) == 0)
        protocol = "generation" if use_generation else "forced_choice"
        warm_ids = warmup_by_task.get(task, None) if (not use_generation and args.fc_warmup_tokens > 0) else None

        # baseline
        if use_generation:
            base = generation_eval(
                model, tok, exs, task,
                layer_indices=layer_indices,
                basis_np=None,
                alpha=args.alpha_remove,
                batch_size=args.batch_size,
                max_prompt_len=args.max_prompt_len,
                max_new_tokens=args.gen_max_new_tokens,
                decoding=args.gen_decoding,
                temperature=args.gen_temperature,
                top_p=args.gen_top_p,
                top_k=args.gen_top_k,
                seed=stable_int_seed(args.seed, args.gen_seed, task, "gen_base"),
            )
        else:
            base = forced_choice_logprob_eval(
                model, tok, exs, task,
                layer_indices=layer_indices,
                basis_np=None,
                alpha=args.alpha_remove,
                batch_size=args.batch_size,
                max_prompt_len=args.max_prompt_len,
                warmup_token_ids=warm_ids,
                answer_prefix=args.fc_answer_prefix,
                prefix_mode=fc_prefix_mode,
            )

        base_arr = np.array(base["correct"], dtype=np.float32)
        b_acc, b_lo, b_hi = bootstrap_ci_mean(base_arr, args.bootstrap_iters, args.ci_alpha, seed=stable_int_seed(args.seed, task, protocol, "baseline"))
        print(f"  [{protocol}] baseline acc={fmt_acc(b_acc, b_lo, b_hi)} {base.get('hook_stats')}")

        # decode_full
        if use_generation:
            dec_full = generation_eval(
                model, tok, exs, task,
                layer_indices=layer_indices,
                basis_np=Q_decode_full,
                alpha=args.alpha_remove,
                batch_size=args.batch_size,
                max_prompt_len=args.max_prompt_len,
                max_new_tokens=args.gen_max_new_tokens,
                decoding=args.gen_decoding,
                temperature=args.gen_temperature,
                top_p=args.gen_top_p,
                top_k=args.gen_top_k,
                seed=stable_int_seed(args.seed, args.gen_seed, task, "gen_dec_full"),
            )
        else:
            dec_full = forced_choice_logprob_eval(
                model, tok, exs, task,
                layer_indices=layer_indices,
                basis_np=Q_decode_full,
                alpha=args.alpha_remove,
                batch_size=args.batch_size,
                max_prompt_len=args.max_prompt_len,
                warmup_token_ids=warm_ids,
                answer_prefix=args.fc_answer_prefix,
                prefix_mode=fc_prefix_mode,
            )
        d_arr = np.array(dec_full["correct"], dtype=np.float32)
        d_acc, d_lo, d_hi = bootstrap_ci_mean(d_arr, args.bootstrap_iters, args.ci_alpha, seed=stable_int_seed(args.seed, task, protocol, "decode_full"))
        print(f"  [{protocol}] decode_shared_full acc={fmt_acc(d_acc, d_lo, d_hi)} {dec_full.get('hook_stats')}")

        # prefill_full
        if use_generation:
            pre_full = generation_eval(
                model, tok, exs, task,
                layer_indices=layer_indices,
                basis_np=Q_prefill_full,
                alpha=args.alpha_remove,
                batch_size=args.batch_size,
                max_prompt_len=args.max_prompt_len,
                max_new_tokens=args.gen_max_new_tokens,
                decoding=args.gen_decoding,
                temperature=args.gen_temperature,
                top_p=args.gen_top_p,
                top_k=args.gen_top_k,
                seed=stable_int_seed(args.seed, args.gen_seed, task, "gen_pre_full"),
            )
        else:
            pre_full = forced_choice_logprob_eval(
                model, tok, exs, task,
                layer_indices=layer_indices,
                basis_np=Q_prefill_full,
                alpha=args.alpha_remove,
                batch_size=args.batch_size,
                max_prompt_len=args.max_prompt_len,
                warmup_token_ids=warm_ids,
                answer_prefix=args.fc_answer_prefix,
                prefix_mode=fc_prefix_mode,
            )
        p_arr = np.array(pre_full["correct"], dtype=np.float32)
        p_acc, p_lo, p_hi = bootstrap_ci_mean(p_arr, args.bootstrap_iters, args.ci_alpha, seed=stable_int_seed(args.seed, task, protocol, "prefill_full"))
        print(f"  [{protocol}] prefill_shared_full acc={fmt_acc(p_acc, p_lo, p_hi)} {pre_full.get('hook_stats')}")

        # decode_km
        if use_generation:
            dec_km = generation_eval(
                model, tok, exs, task,
                layer_indices=layer_indices,
                basis_np=Q_decode_km,
                alpha=args.alpha_remove,
                batch_size=args.batch_size,
                max_prompt_len=args.max_prompt_len,
                max_new_tokens=args.gen_max_new_tokens,
                decoding=args.gen_decoding,
                temperature=args.gen_temperature,
                top_p=args.gen_top_p,
                top_k=args.gen_top_k,
                seed=stable_int_seed(args.seed, args.gen_seed, task, "gen_dec_km"),
            )
        else:
            dec_km = forced_choice_logprob_eval(
                model, tok, exs, task,
                layer_indices=layer_indices,
                basis_np=Q_decode_km,
                alpha=args.alpha_remove,
                batch_size=args.batch_size,
                max_prompt_len=args.max_prompt_len,
                warmup_token_ids=warm_ids,
                answer_prefix=args.fc_answer_prefix,
                prefix_mode=fc_prefix_mode,
            )
        dk_arr = np.array(dec_km["correct"], dtype=np.float32)
        dk_acc, dk_lo, dk_hi = bootstrap_ci_mean(dk_arr, args.bootstrap_iters, args.ci_alpha, seed=stable_int_seed(args.seed, task, protocol, "decode_km"))
        print(f"  [{protocol}] decode_shared_km acc={fmt_acc(dk_acc, dk_lo, dk_hi)}")

        # prefill_km
        if use_generation:
            pre_km = generation_eval(
                model, tok, exs, task,
                layer_indices=layer_indices,
                basis_np=Q_prefill_km,
                alpha=args.alpha_remove,
                batch_size=args.batch_size,
                max_prompt_len=args.max_prompt_len,
                max_new_tokens=args.gen_max_new_tokens,
                decoding=args.gen_decoding,
                temperature=args.gen_temperature,
                top_p=args.gen_top_p,
                top_k=args.gen_top_k,
                seed=stable_int_seed(args.seed, args.gen_seed, task, "gen_pre_km"),
            )
        else:
            pre_km = forced_choice_logprob_eval(
                model, tok, exs, task,
                layer_indices=layer_indices,
                basis_np=Q_prefill_km,
                alpha=args.alpha_remove,
                batch_size=args.batch_size,
                max_prompt_len=args.max_prompt_len,
                warmup_token_ids=warm_ids,
                answer_prefix=args.fc_answer_prefix,
                prefix_mode=fc_prefix_mode,
            )
        pk_arr = np.array(pre_km["correct"], dtype=np.float32)
        pk_acc, pk_lo, pk_hi = bootstrap_ci_mean(pk_arr, args.bootstrap_iters, args.ci_alpha, seed=stable_int_seed(args.seed, task, protocol, "prefill_km"))
        print(f"  [{protocol}] prefill_shared_km acc={fmt_acc(pk_acc, pk_lo, pk_hi)}")

        # rand_km
        if use_generation:
            rand_km = generation_eval(
                model, tok, exs, task,
                layer_indices=layer_indices,
                basis_np=Q_rand_km,
                alpha=args.alpha_remove,
                batch_size=args.batch_size,
                max_prompt_len=args.max_prompt_len,
                max_new_tokens=args.gen_max_new_tokens,
                decoding=args.gen_decoding,
                temperature=args.gen_temperature,
                top_p=args.gen_top_p,
                top_k=args.gen_top_k,
                seed=stable_int_seed(args.seed, args.gen_seed, task, "gen_rand_km"),
            )
        else:
            rand_km = forced_choice_logprob_eval(
                model, tok, exs, task,
                layer_indices=layer_indices,
                basis_np=Q_rand_km,
                alpha=args.alpha_remove,
                batch_size=args.batch_size,
                max_prompt_len=args.max_prompt_len,
                warmup_token_ids=warm_ids,
                answer_prefix=args.fc_answer_prefix,
                prefix_mode=fc_prefix_mode,
            )
        rk_arr = np.array(rand_km["correct"], dtype=np.float32)
        rk_acc, rk_lo, rk_hi = bootstrap_ci_mean(rk_arr, args.bootstrap_iters, args.ci_alpha, seed=stable_int_seed(args.seed, task, protocol, "rand_km"))
        print(f"  [{protocol}] rand_km acc={fmt_acc(rk_acc, rk_lo, rk_hi)}")

        # Paired tests (decode_km vs prefill_km; paired by example)
        stat_dk_vs_pk = summarize_paired(
            baseline_correct=pk_arr,
            treat_correct=dk_arr,
            bootstrap_iters=args.bootstrap_iters,
            perm_iters=args.perm_iters,
            alpha=args.ci_alpha,
            seed=stable_int_seed(args.seed, task, protocol, "paired", "dk_vs_pk"),
        )

        eval_results[task] = {
            "protocol": protocol,
            "n": n,
            "baseline": {"acc": b_acc, "ci": [b_lo, b_hi], "detail": base},
            "decode_full": {"acc": d_acc, "ci": [d_lo, d_hi], "k": k_decode, "detail": dec_full},
            "prefill_full": {"acc": p_acc, "ci": [p_lo, p_hi], "k": k_prefill, "detail": pre_full},
            "decode_km": {"acc": dk_acc, "ci": [dk_lo, dk_hi], "k": k_match, "detail": dec_km},
            "prefill_km": {"acc": pk_acc, "ci": [pk_lo, pk_hi], "k": k_match, "detail": pre_km},
            "rand_km": {"acc": rk_acc, "ci": [rk_lo, rk_hi], "k": k_match, "detail": rand_km},
            "paired": {"decode_km_minus_prefill_km": stat_dk_vs_pk},
        }

    # Tables: all tasks, dimension-matched
    warm_str = f"W={args.fc_warmup_tokens}" if (args.fc_warmup_tokens > 0 and not bool(args.do_generation)) else "W=0"
    rows_km = []
    for task in tasks:
        r = eval_results[task]
        stat = r["paired"]["decode_km_minus_prefill_km"]
        rows_km.append(
            [
                task,
                r["protocol"],
                str(r["n"]),
                fmt_acc(r["baseline"]["acc"], r["baseline"]["ci"][0], r["baseline"]["ci"][1]),
                fmt_acc(r["decode_km"]["acc"], r["decode_km"]["ci"][0], r["decode_km"]["ci"][1]),
                fmt_acc(r["prefill_km"]["acc"], r["prefill_km"]["ci"][0], r["prefill_km"]["ci"][1]),
                fmt_acc(r["rand_km"]["acc"], r["rand_km"]["ci"][0], r["rand_km"]["ci"][1]),
                f"{stat['mean_diff']*100:+.1f} [{stat['ci_low']*100:+.1f}, {stat['ci_high']*100:+.1f}]",
                f"{stat['p_value']:.3g}",
            ]
        )

    header_km = [
        "Task",
        "Protocol",
        "n",
        "Baseline",
        f"Decode-shared (k={k_match})",
        f"Prefill-shared (k={k_match})",
        f"Random (k={k_match})",
        "Δ(Decode-Prefill) [CI]",
        "p",
    ]
    md_km = md_table(rows_km, header_km)
    tex_km = latex_table(
        rows_km,
        header_km,
        caption=(
            f"Prefill-vs-Decode alignment experiment across 9 benchmarks. "
            f"Dimension-matched bases (k={k_match}). Forced-choice warmup={warm_str} (if protocol=forced_choice). "
            f"Forced-choice prefix_mode={fc_prefix_mode}."
        ),
        label="tab:prefill-vs-decode-9tasks-kmatch",
        colspec="llrcccccc",
    )

    # Native-k (decode_full vs prefill_full) reference table
    rows_nat = []
    for task in tasks:
        r = eval_results[task]
        delta = (r["decode_full"]["acc"] - r["prefill_full"]["acc"]) * 100
        rows_nat.append(
            [
                task,
                r["protocol"],
                str(r["n"]),
                fmt_acc(r["baseline"]["acc"], r["baseline"]["ci"][0], r["baseline"]["ci"][1]),
                fmt_acc(r["decode_full"]["acc"], r["decode_full"]["ci"][0], r["decode_full"]["ci"][1]),
                fmt_acc(r["prefill_full"]["acc"], r["prefill_full"]["ci"][0], r["prefill_full"]["ci"][1]),
                f"{delta:+.1f}",
                "(n/a)",
            ]
        )

    header_nat = [
        "Task",
        "Protocol",
        "n",
        "Baseline",
        f"Decode-shared (k={k_decode})",
        f"Prefill-shared (k={k_prefill})",
        "Δ(Decode-Prefill)",
        "p",
    ]
    md_nat = md_table(rows_nat, header_nat)
    tex_nat = latex_table(
        rows_nat,
        header_nat,
        caption="Native shared-k reference table (no dimension matching).",
        label="tab:prefill-vs-decode-9tasks-native",
        colspec="llrccccc",
    )

    results = {
        "config": {
            "model": args.model,
            "dtype": args.dtype,
            "device": args.device,
            "trust_remote_code": bool(args.trust_remote_code),
            "layer": args.layer,
            "tasks": tasks,
            "n_prompts": args.n_prompts,
            "eval_n": args.eval_n,
            "calib_decode_max_new_tokens": args.calib_decode_max_new_tokens,
            "per_task_max_states": args.per_task_max_states,
            "pca_var": args.pca_var,
            "tau": args.tau,
            "m_shared": args.m_shared,
            "alpha_remove": args.alpha_remove,
            "match_state_count": state_match_info,
            "prompt": {
                "template_randomization": bool(args.template_randomization),
                "template_seed": args.template_seed,
                "shuffle_choices": bool(args.shuffle_choices),
                "add_answer_prefix": bool(args.add_answer_prefix),
                "answer_prefix": args.answer_prefix,
            },
            "forced_choice": {
                "enabled": not bool(args.do_generation),
                "fc_warmup_tokens": args.fc_warmup_tokens,
                "fc_warmup_decoding": args.fc_warmup_decoding,
                "fc_warmup_ban_eos": bool(args.fc_warmup_ban_eos),
                "fc_warmup_seed": args.fc_warmup_seed,
                "fc_prefix_mode": fc_prefix_mode,
                "fc_answer_prefix": args.fc_answer_prefix,
            },
            "generation": {
                "enabled": bool(args.do_generation),
                "gen_decoding": args.gen_decoding,
                "gen_temperature": args.gen_temperature,
                "gen_top_p": args.gen_top_p,
                "gen_top_k": args.gen_top_k,
                "gen_max_new_tokens": args.gen_max_new_tokens,
                "gen_seed": args.gen_seed,
            },
            "seed": args.seed,
            "dataset_meta": meta_by,
        },
        "basis": {
            "decode": {"cross_dim": extra_decode["cross_dim"], "shared_k": k_decode, "n_balanced": extra_decode["n_balanced"]},
            "prefill": {"cross_dim": extra_prefill["cross_dim"], "shared_k": k_prefill, "n_balanced": extra_prefill["n_balanced"]},
            "k_match": k_match,
            "subspace_similarity_full": subspace_similarity(Q_decode_full, Q_prefill_full),
            "subspace_similarity_kmatch": subspace_similarity(Q_decode_km, Q_prefill_km),
            "energy": {
                "full": {
                    "decode_on_decode": er_decode_on_decode,
                    "prefill_on_decode": er_prefill_on_decode,
                    "prefill_on_prefill": er_prefill_on_prefill,
                    "decode_on_prefill": er_decode_on_prefill,
                },
                "kmatch": {
                    "decode_on_decode": er_decode_km_on_decode,
                    "prefill_on_decode": er_prefill_km_on_decode,
                    "rand_on_decode": er_rand_km_on_decode,
                    "prefill_on_prefill": er_prefill_km_on_prefill,
                    "decode_on_prefill": er_decode_km_on_prefill,
                },
            },
        },
        "eval": eval_results,
        "tables": {"markdown_kmatch": md_km, "markdown_native": md_nat, "latex_kmatch": tex_km, "latex_native": tex_nat},
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=json_default)

    # Save TXT/MD/TEX summaries
    summary_lines = []
    summary_lines.append("[Summary]")
    summary_lines.append(f"Model={args.model} dtype={args.dtype} device={args.device} layer={args.layer} trust_remote_code={bool(args.trust_remote_code)}")
    summary_lines.append(f"Tasks={tasks}")
    summary_lines.append(f"Decode shared_k={k_decode} Prefill shared_k={k_prefill} k_match={k_match}")
    summary_lines.append(f"State-count matching: {'enabled' if args.match_state_count else 'disabled'}")
    if not bool(args.do_generation):
        summary_lines.append(f"Forced-choice warmup tokens: {args.fc_warmup_tokens}")
        summary_lines.append(f"Forced-choice prefix_mode: {fc_prefix_mode} prefix={args.fc_answer_prefix!r}")
    else:
        summary_lines.append("Evaluation protocol: GENERATION for all tasks (--do_generation=1)")
    summary_lines.append("")
    summary_lines.append("## Dimension-matched results (k_match)")
    summary_lines.append(md_km)
    summary_lines.append("")
    summary_lines.append("## Native-k reference")
    summary_lines.append(md_nat)
    summary_lines.append("")

    with open(args.out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")
    with open(args.out_tex, "w", encoding="utf-8") as f:
        f.write(tex_km + "\n" + tex_nat + "\n")

    print("\n" + "=" * 80)
    print("\n".join(summary_lines[:12]))
    print("...")
    print(f"[Done] JSON: {args.out_json}")
    print(f"[Done] TXT : {args.out_txt}")
    print(f"[Done] MD  : {args.out_md}")
    print(f"[Done] TEX : {args.out_tex}")
    print("=" * 80)


if __name__ == "__main__":
    main()
