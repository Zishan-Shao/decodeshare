"""Shared decode-stage LOTO utilities used by DecodeShare experiment runners."""

from __future__ import annotations

import bisect
import random
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import numpy as np
import torch

from decodeshare.benchmark_dataloaders import (
    is_correct as is_correct_bool,
    stable_int_seed as stable_int_seed_bdl,
)
from decodeshare.subspace import (
    compute_cross_task_subspace,
    find_fully_shared_basis_improved,
    get_model_layers,
)


stable_int_seed = stable_int_seed_bdl


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return str(o)


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


def summarize_paired(
    baseline_correct: np.ndarray,
    treat_correct: np.ndarray,
    label: str,
    bootstrap_iters: int,
    perm_iters: int,
    alpha: float,
    seed: int,
) -> Dict[str, Any]:
    md, lo, hi = paired_bootstrap_ci_diff(
        baseline_correct, treat_correct,
        iters=bootstrap_iters, alpha=alpha,
        seed=seed + 123
    )
    p = signflip_permutation_test(
        baseline_correct, treat_correct,
        iters=perm_iters, seed=seed + 456
    )
    return {
        "label": label,
        "mean_diff": md,
        "ci_low": lo,
        "ci_high": hi,
        "p_value": p,
    }


def fmt_acc(acc: float, lo: float, hi: float) -> str:
    return f"{acc*100:.1f} [{lo*100:.1f}, {hi*100:.1f}]"


def is_correct(dataset: str, pred: str, gold: str) -> int:
    return int(is_correct_bool(dataset, pred, gold))


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


class DecodeLastTokenActivationCollector:
    """
    Collect last-token hidden states ONLY during decode forward passes (seq_len==1).
    storage[task][layer_idx] -> list of np arrays [B', D]
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

    def get_task_activations(self, task: str, layer_idx: int) -> Optional[np.ndarray]:
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


def orthonormalize_np(basis: np.ndarray) -> np.ndarray:
    q, _ = np.linalg.qr(basis.astype(np.float32, copy=False))
    return q.astype(np.float32, copy=False)


def max_offdiag(Q: np.ndarray) -> float:
    G = Q.T @ Q
    k = G.shape[0]
    G = G - np.eye(k, dtype=G.dtype)
    return float(np.max(np.abs(G))) if k > 0 else 0.0


def max_overlap(Qa: np.ndarray, Qb: np.ndarray) -> float:
    if Qa.size == 0 or Qb.size == 0:
        return 0.0
    M = Qa.T @ Qb
    return float(np.max(np.abs(M)))


def energy_ratio_stats(states: np.ndarray, Q: np.ndarray) -> Dict[str, float]:
    eps = 1e-12
    H = states.astype(np.float32, copy=False)
    proj = H @ Q
    num = np.sum(proj * proj, axis=1)
    den = np.sum(H * H, axis=1) + eps
    r = num / den
    return {"mean": float(np.mean(r)), "p50": float(np.percentile(r, 50)), "p95": float(np.percentile(r, 95))}


def infer_component_variances(contributions: Dict[str, Any], tasks: List[str], cross_dim: int) -> np.ndarray:
    candidates = []
    for t in tasks:
        d = contributions.get(t, {})
        v = None
        for key in ["variances", "component_variances", "per_component_variance", "var", "vars"]:
            if key in d:
                v = np.asarray(d[key], dtype=np.float64)
                break
        if v is None:
            for _, val in d.items():
                if isinstance(val, (list, np.ndarray)) and len(val) >= cross_dim:
                    vv = np.asarray(val, dtype=np.float64)
                    if vv.ndim == 1:
                        v = vv
                        break
        if v is None or v.ndim != 1 or v.shape[0] < cross_dim:
            raise KeyError(f"Cannot infer per-component variances for task={t}. keys={list(d.keys())}")
        candidates.append(v[:cross_dim])
    pooled = np.mean(np.stack(candidates, axis=0), axis=0)
    return pooled


def select_rand_indices(
    rand_type: str,
    cross_dim: int,
    shared_indices: List[int],
    pooled_var: Optional[np.ndarray],
    k: int,
    seed: int,
) -> List[int]:
    rng = np.random.default_rng(seed)
    shared_set = set(shared_indices)
    nonshared = [i for i in range(cross_dim) if i not in shared_set]
    if len(nonshared) < k:
        raise RuntimeError(f"Not enough nonshared components: nonshared={len(nonshared)} < k={k}")

    if rand_type == "joint_nonshared_uniform" or pooled_var is None:
        return list(rng.choice(nonshared, size=k, replace=False))

    if rand_type == "joint_nonshared_topk":
        idx_sorted = sorted(nonshared, key=lambda i: pooled_var[i], reverse=True)
        return idx_sorted[:k]

    if rand_type == "joint_nonshared_varmatch":
        shared_vars = [(i, pooled_var[i]) for i in shared_indices]
        shared_vars.sort(key=lambda x: x[1])
        nonshared_sorted = sorted(nonshared, key=lambda i: pooled_var[i])
        nonshared_vals = [pooled_var[i] for i in nonshared_sorted]
        chosen = []
        for _, v in shared_vars:
            j = bisect.bisect_left(nonshared_vals, v)
            cand_pos = []
            if 0 <= j < len(nonshared_sorted):
                cand_pos.append(j)
            if 0 <= j - 1 < len(nonshared_sorted):
                cand_pos.append(j - 1)
            best = None
            best_d = None
            for p in cand_pos:
                d = abs(nonshared_vals[p] - v)
                if best is None or d < best_d - 1e-12 or (abs(d - best_d) < 1e-12 and rng.random() < 0.5):
                    best = p
                    best_d = d
            if best is None:
                best = rng.integers(0, len(nonshared_sorted))
            chosen_idx = nonshared_sorted.pop(best)
            nonshared_vals.pop(best)
            chosen.append(chosen_idx)
            if len(chosen) >= k:
                break
        if len(chosen) < k:
            remaining = nonshared_sorted
            extra = list(rng.choice(remaining, size=(k - len(chosen)), replace=False))
            chosen.extend(extra)
        return chosen

    raise ValueError(f"Unknown rand_type={rand_type}")


def compute_shared_subspace_decode_aligned(
    model,
    tokenizer,
    prompts_by_task: Dict[str, List[str]],
    layer_indices: List[int],
    *,
    calib_decoding: str,
    calib_batch_size: int,
    calib_max_new_tokens: int,
    per_task_max_states: int,
    max_prompt_len: int,
    temperature: float,
    top_p: float,
    top_k: int,
    global_seed: int,
    variance_threshold: float,
    min_dim: int,
    max_dim: int,
    tau: float,
    m_shared: str,
    collect_fn: Any,
) -> Tuple[np.ndarray, List[int], Dict[str, Any], Dict[str, Dict[int, np.ndarray]]]:

    print("\n" + "=" * 80)
    print("[Subspace-A3] Collecting DECODE last-token activations for shared subspace estimation ...")
    print(f"[Subspace-A3] calib_decoding={calib_decoding}, max_new_tokens={calib_max_new_tokens}, per_task_max_states={per_task_max_states}")
    print("=" * 80)

    layers, _ = get_model_layers(model)
    collector = DecodeLastTokenActivationCollector(layer_indices)

    handles = []
    for layer_idx in layer_indices:
        if layer_idx >= len(layers):
            print(f"[Subspace-A3] Warn: layer_idx={layer_idx} out of range, skipping")
            continue
        handles.append(layers[layer_idx].register_forward_hook(collector.make_hook(layer_idx)))

    try:
        for task_name, prompts in prompts_by_task.items():
            print(f"[Subspace-A3] Task={task_name}, prompts={len(prompts)}")
            collector.set_current_task(task_name)
            collect_fn(
                model,
                tokenizer,
                prompts=prompts,
                collector=collector,
                batch_size=calib_batch_size,
                max_new_tokens=calib_max_new_tokens,
                decoding=calib_decoding,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_prompt_len=max_prompt_len,
            )
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass
        collector.set_capture(False, None)

    task_activations: Dict[str, Dict[int, np.ndarray]] = {}
    for task_name in prompts_by_task.keys():
        layer_dict = {}
        for layer_idx in layer_indices:
            acts = collector.get_task_activations(task_name, layer_idx)
            if acts is None or acts.shape[0] == 0:
                continue
            ss = stable_int_seed(global_seed, task_name, layer_idx, "subsample")
            acts = _subsample_rows_np(acts, per_task_max_states, seed=ss)
            layer_dict[layer_idx] = acts
            print(f"[Subspace-A3]  collected {task_name} layer={layer_idx}: {acts.shape[0]} x {acts.shape[1]}")
        if layer_dict:
            task_activations[task_name] = layer_dict

    if not task_activations:
        raise RuntimeError("[Subspace-A3] No decode activations collected. Check hooks/layers/generation loop.")

    joint_subspace, cross_dim, contributions, full_pca_info = compute_cross_task_subspace(
        task_activations,
        variance_threshold=variance_threshold,
        min_dim=min_dim,
        max_dim=max_dim,
        return_full_pca=True,
    )
    if joint_subspace is None or cross_dim <= 0:
        raise RuntimeError("[Subspace-A3] Failed to compute cross-task subspace.")

    tasks = list(task_activations.keys())


    if m_shared == "all":
        min_tasks = len(tasks)
    else:
        try:
            min_tasks = max(2, int(m_shared))
        except Exception:
            min_tasks = len(tasks)

    shared_indices = find_fully_shared_basis_improved(
        contributions,
        tasks,
        cross_dim,
        min_tasks_shared=min_tasks,
        relative_threshold=tau,
        top_k_components=cross_dim,
    )

    if not shared_indices and min_tasks != 2:

        print("[Subspace-A3] No shared basis for requested m_shared; falling back to min_tasks_shared=2.")
        shared_indices = find_fully_shared_basis_improved(
            contributions,
            tasks,
            cross_dim,
            min_tasks_shared=2,
            relative_threshold=tau,
            top_k_components=cross_dim,
        )

    print(f"[Subspace-A3] cross_dim={cross_dim}, shared_basis_count={len(shared_indices)} (m_shared={m_shared}, tau={tau})")
    extra = {
        "cross_dim": int(cross_dim),
        "tasks_used": tasks,
        "task_contributions": contributions,
        "full_pca_info": full_pca_info,
        "calib": {
            "calib_decoding": calib_decoding,
            "calib_max_new_tokens": calib_max_new_tokens,
            "per_task_max_states": per_task_max_states,
        },
        "m_shared": m_shared,
        "tau": float(tau),
    }
    return joint_subspace.astype(np.float32, copy=False), shared_indices, extra, task_activations

class GenerationState:
    def __init__(self, batch_size: int, device: torch.device, reasoning_threshold: int):
        self.batch_size = batch_size
        self.device = device
        self.reasoning_threshold = int(reasoning_threshold)
        self.unfinished = torch.ones(batch_size, dtype=torch.bool, device=device)
        self.gen_steps = torch.zeros(batch_size, dtype=torch.long, device=device)

    def current_reasoning_mask(self) -> torch.Tensor:
        return self.unfinished & (self.gen_steps < self.reasoning_threshold)

    def step_update(self, next_tokens: torch.Tensor, eos_token_id: int) -> None:
        next_tokens = next_tokens.squeeze(-1)
        active = self.unfinished.clone()
        self.gen_steps[active] += 1
        newly_finished = active & (next_tokens == eos_token_id)
        self.unfinished[newly_finished] = False

    def clone(self) -> "GenerationState":
        st = GenerationState(self.batch_size, self.device, self.reasoning_threshold)
        st.unfinished = self.unfinished.clone()
        st.gen_steps = self.gen_steps.clone()
        return st


class HookStats:
    def __init__(self, name: str):
        self.name = name
        self.decode_calls = 0
        self.intervened = 0


class LastTokenRemovalHook:
    def __init__(self, Q_np: np.ndarray, alpha: float, stats: HookStats):
        self.alpha = float(alpha)
        self.stats = stats
        self.Q = torch.tensor(orthonormalize_np(Q_np), dtype=torch.float32)
        self.Q_device: Optional[torch.Tensor] = None

    def _Q(self, device: torch.device) -> torch.Tensor:
        if self.Q_device is None or self.Q_device.device != device:
            self.Q_device = self.Q.to(device=device)
        return self.Q_device

    def __call__(self, module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        if not isinstance(hs, torch.Tensor) or hs.ndim != 3:
            return output
        if hs.shape[1] != 1:
            return output

        self.stats.decode_calls += 1

        Q = self._Q(hs.device)
        x = hs[:, -1, :].float()
        proj = (x @ Q) @ Q.T
        hs2 = hs.clone()
        hs2[:, -1, :] = (x - self.alpha * proj).to(dtype=hs.dtype)

        self.stats.intervened += 1
        if isinstance(output, tuple):
            return (hs2,) + output[1:]
        return hs2


class LastTokenStagedRemovalHook(LastTokenRemovalHook):
    def __init__(self, Q_np: np.ndarray, alpha: float, stats: HookStats, reasoning_threshold: int):
        super().__init__(Q_np, alpha, stats)
        self.state: Optional[GenerationState] = None
        self.reasoning_threshold = int(reasoning_threshold)

    def set_state(self, st: Optional[GenerationState]) -> None:
        self.state = st

    def __call__(self, module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        if not isinstance(hs, torch.Tensor) or hs.ndim != 3:
            return output
        if hs.shape[1] != 1:
            return output

        self.stats.decode_calls += 1

        if self.state is None:
            return super().__call__(module, inputs, output)

        mask = self.state.current_reasoning_mask()
        if not bool(mask.any().item()):
            return output

        Q = self._Q(hs.device)
        x = hs[:, -1, :].float()
        x_sel = x[mask]
        proj = (x_sel @ Q) @ Q.T
        x[mask] = x_sel - self.alpha * proj

        hs2 = hs.clone()
        hs2[:, -1, :] = x.to(dtype=hs.dtype)

        self.stats.intervened += 1
        if isinstance(output, tuple):
            return (hs2,) + output[1:]
        return hs2


def register_hooks_for_condition(
    model,
    layer_indices: List[int],
    Q_np: Optional[np.ndarray],
    condition: str,
    alpha: float,
    reasoning_token_threshold: int,
) -> Tuple[List[Any], Optional[Any], List[HookStats]]:
    assert condition in ["baseline", "full", "staged"]
    if condition == "baseline":
        return [], None, []

    assert Q_np is not None
    layers, _ = get_model_layers(model)

    handles = []
    staged_hooks: List[LastTokenStagedRemovalHook] = []
    hook_stats: List[HookStats] = []

    for layer_idx in layer_indices:
        if layer_idx >= len(layers):
            print(f"[Warn] layer_idx={layer_idx} out of range, skipping")
            continue

        if condition == "full":
            stats = HookStats(name=f"full@{layer_idx}")
            hk = LastTokenRemovalHook(Q_np, alpha=alpha, stats=stats)
            handles.append(layers[layer_idx].register_forward_hook(hk))
            hook_stats.append(stats)
        else:
            stats = HookStats(name=f"staged@{layer_idx}")
            hk = LastTokenStagedRemovalHook(Q_np, alpha=alpha, stats=stats, reasoning_threshold=reasoning_token_threshold)
            staged_hooks.append(hk)
            handles.append(layers[layer_idx].register_forward_hook(hk))
            hook_stats.append(stats)

    def setter(state_or_none: Optional[GenerationState]) -> None:
        for hk in staged_hooks:
            hk.set_state(state_or_none)

    return handles, (setter if condition == "staged" else None), hook_stats


def remove_hooks(handles: List[Any]) -> None:
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass


def infer_hidden_dim(model) -> Optional[int]:
    cfg = getattr(model, "config", None)


    for k in ("hidden_size", "n_embd", "dim", "d_model", "model_dim", "embed_dim"):
        v = getattr(cfg, k, None)
        if isinstance(v, int) and v > 0:
            return v


    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is not None:
        for k in ("hidden_size", "n_embd", "dim", "d_model", "model_dim", "embed_dim"):
            v = getattr(text_cfg, k, None)
            if isinstance(v, int) and v > 0:
                return v


    try:
        emb = model.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight") and isinstance(emb.weight, torch.Tensor) and emb.weight.ndim == 2:
            return int(emb.weight.shape[1])
        if emb is not None and hasattr(emb, "embedding_dim"):
            return int(emb.embedding_dim)
    except Exception:
        pass

    return None
