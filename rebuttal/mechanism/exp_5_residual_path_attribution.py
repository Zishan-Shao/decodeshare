# -*- coding: utf-8 -*-
"""
exp_5_residual_path_attribution.py

Mechanism experiment M5 (residual-path attribution):
  Can an effect attributed to shared subspace be better explained as being in the
  residual stream than in a single module branch?

Core idea
---------
Run the decode-shared basis Q removal on:
  - attn_delta: attention branch delta (layer.self_attn output)
  - mlp_delta : MLP branch delta (layer.mlp output)
  - sum_delta : simultaneously remove both branch deltas
  - res_in    : pre-hook on layer input (incoming residual stream before any branch split)
  - res_out   : post-hook on layer output (merged residual stream after branch recombination)

Timing follows Exp-2/4:
  - lateN window aligned to answer-entry stage
  - midN placebo window in the middle of warmup

Design points (for a stronger residual claim):
  1) If residual-stream mechanism dominates, res_in/res_out should show a consistent
     pattern that is not uniquely branch-specific.
  2) If branch effects are an artifact of residual mixing, attn_delta and mlp_delta
     remain weak/inconsistent while sum_delta and residual-stream conditions are clearer.
  3) Random subspace control (optional) checks direction specificity.
"""

from __future__ import annotations

import os
import sys
import json
import math
import argparse
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Iterable

import numpy as np
import torch


# -----------------------------------------------------------------------------
# Repo-local imports
# -----------------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_repo_root(start_dir: str) -> str:
    cur = os.path.abspath(start_dir)
    for _ in range(10):
        if os.path.isdir(os.path.join(cur, "src")) and os.path.isdir(os.path.join(cur, "reasoning")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return os.path.normpath(os.path.join(start_dir, "..", ".."))


ROOT_DIR = _find_repo_root(THIS_DIR)
SRC_DIR = os.path.join(ROOT_DIR, "src")
REASONING_DIR = os.path.join(ROOT_DIR, "reasoning")

for p in [SRC_DIR, REASONING_DIR, ROOT_DIR]:
    if p not in sys.path:
        sys.path.append(p)

try:
    import eval_perf as EP  # reasoning/eval_perf.py
except Exception as e:  # pragma: no cover
    raise RuntimeError("Failed to import `reasoning/eval_perf.py` as module `eval_perf`.") from e

try:
    from benchmark_dataloaders import load_selected_tasks  # src/benchmark_dataloaders.py
except Exception as e:  # pragma: no cover
    raise RuntimeError("Failed to import `src/benchmark_dataloaders.py` as module `benchmark_dataloaders`.") from e


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------
def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        if o.ndim == 0:
            return float(o.detach().cpu().item())
        return o.detach().cpu().tolist()
    return str(o)


def _atomic_json_dump(obj: Any, out_path: str) -> None:
    out_path = os.path.expanduser(out_path)
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=os.path.basename(out_path) + ".", suffix=".tmp", dir=out_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)
        os.replace(tmp_path, out_path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def _atomic_text_dump(text: str, out_path: str) -> None:
    out_path = os.path.expanduser(out_path)
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=os.path.basename(out_path) + ".", suffix=".tmp", dir=out_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_path, out_path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def _split_csv(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def _safe_upper(x: Any) -> str:
    return str(x).strip().upper()


def _decode_backslash_escapes(s: str) -> str:
    # Convenience for CLI usage: allow passing "\\n" to mean newline.
    return str(s).replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r")


def _md_table(rows: List[List[str]], header: List[str]) -> str:
    out = []
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "---|" * len(header))
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def _pct(x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "nan"
    return f"{100.0 * float(x):.1f}"


# -----------------------------------------------------------------------------
# Decode-step window hooks
# -----------------------------------------------------------------------------
class WindowHookStats:
    def __init__(self, name: str, *, location: str, module_ref: str):
        self.name = str(name)
        self.location = str(location)
        self.module_ref = str(module_ref)
        self.decode_calls = 0
        self.intervened = 0
        self.step_hist: Dict[int, int] = {}

    def report(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "location": self.location,
            "module_ref": self.module_ref,
            "decode_calls": int(self.decode_calls),
            "intervened": int(self.intervened),
            "step_hist": {str(k): int(v) for k, v in sorted(self.step_hist.items())},
        }


class _BaseWindowedHook:
    """
    Shared pieces for decode-step windowing:
      - step counter resets whenever seq_len != 1 (prefill path)
    """

    def __init__(self, *, alpha: float, window_start: int, window_end: int, stats: WindowHookStats):
        self.alpha = float(alpha)
        self.window_start = int(window_start)
        self.window_end = int(window_end)
        if self.window_end <= self.window_start:
            raise ValueError(f"Invalid window: start={self.window_start}, end={self.window_end}")
        self.stats = stats
        self.enabled = True
        self._step = 0

    def reset_steps(self) -> None:
        self._step = 0

    def set_enabled(self, flag: bool) -> None:
        self.enabled = bool(flag)

    def _on_prefill(self) -> None:
        self._step = 0

    def _on_decode_step(self) -> int:
        step_idx = int(self._step)
        self._step += 1
        self.stats.decode_calls += 1
        return step_idx

    def _should_intervene(self, step_idx: int) -> bool:
        if not self.enabled:
            return False
        return bool(self.window_start <= int(step_idx) < self.window_end)

    def _bump_hist(self, step_idx: int) -> None:
        self.stats.intervened += 1
        self.stats.step_hist[step_idx] = int(self.stats.step_hist.get(step_idx, 0) + 1)


class WindowedLastTokenProjectionHook(_BaseWindowedHook):
    """
    Forward hook: remove projection onto Q from module output last token on decode steps.
    """

    def __init__(self, Q_np: np.ndarray, alpha: float, window_start: int, window_end: int, stats: WindowHookStats):
        super().__init__(alpha=alpha, window_start=window_start, window_end=window_end, stats=stats)
        self.Q_cpu = torch.tensor(EP.orthonormalize_np(Q_np), dtype=torch.float32)
        self.Q_dev: Optional[torch.Tensor] = None

    def _Q(self, device: torch.device) -> torch.Tensor:
        if self.Q_dev is None or self.Q_dev.device != device:
            self.Q_dev = self.Q_cpu.to(device=device)
        return self.Q_dev

    def __call__(self, module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        if not isinstance(hs, torch.Tensor) or hs.ndim != 3:
            return output
        if hs.shape[1] != 1:
            self._on_prefill()
            return output

        step_idx = self._on_decode_step()
        if not self._should_intervene(step_idx):
            return output

        Q = self._Q(hs.device)
        x = hs[:, -1, :].float()
        proj = (x @ Q) @ Q.T
        hs2 = hs.clone()
        hs2[:, -1, :] = (x - self.alpha * proj).to(dtype=hs.dtype)

        self._bump_hist(step_idx)
        if isinstance(output, tuple):
            return (hs2,) + output[1:]
        return hs2


class WindowedLastTokenPreInputProjectionHook(_BaseWindowedHook):
    """
    Forward pre-hook: remove projection onto Q from module input last token.
    This affects the residual stream input to the module.
    """

    def __init__(self, Q_np: np.ndarray, alpha: float, window_start: int, window_end: int, stats: WindowHookStats):
        super().__init__(alpha=alpha, window_start=window_start, window_end=window_end, stats=stats)
        self.Q_cpu = torch.tensor(EP.orthonormalize_np(Q_np), dtype=torch.float32)
        self.Q_dev: Optional[torch.Tensor] = None

    def _Q(self, device: torch.device) -> torch.Tensor:
        if self.Q_dev is None or self.Q_dev.device != device:
            self.Q_dev = self.Q_cpu.to(device=device)
        return self.Q_dev

    def __call__(self, module, inputs):
        if not inputs:
            return None
        hs = inputs[0]
        if not isinstance(hs, torch.Tensor) or hs.ndim != 3:
            return None
        if hs.shape[1] != 1:
            self._on_prefill()
            return None

        step_idx = self._on_decode_step()
        if not self._should_intervene(step_idx):
            return None

        Q = self._Q(hs.device)
        x = hs[:, -1, :].float()
        proj = (x @ Q) @ Q.T
        hs2 = hs.clone()
        hs2[:, -1, :] = (x - self.alpha * proj).to(dtype=hs.dtype)

        self._bump_hist(step_idx)
        if len(inputs) == 1:
            return (hs2,)
        return (hs2,) + inputs[1:]


def _remove_hooks(handles: Sequence[Any]) -> None:
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass


def _find_attn_module(layer) -> Tuple[torch.nn.Module, str]:
    for attr in ["self_attn", "attn", "attention"]:
        if hasattr(layer, attr):
            mod = getattr(layer, attr)
            if isinstance(mod, torch.nn.Module):
                return mod, str(attr)
    raise RuntimeError("Could not find an attention module on this layer.")


def _find_mlp_module(layer) -> Tuple[torch.nn.Module, str]:
    for attr in ["mlp", "feed_forward", "ffn", "mlp_layer"]:
        if hasattr(layer, attr):
            mod = getattr(layer, attr)
            if isinstance(mod, torch.nn.Module):
                return mod, str(attr)
    raise RuntimeError("Could not find an MLP/FFN module on this layer.")


def _register_condition_hooks(
    model,
    *,
    layer_idx: int,
    alpha: float,
    window_start: int,
    window_end: int,
    components: Sequence[str],
    component_bases: Dict[str, np.ndarray],
    condition_name: str,
) -> Tuple[List[Any], List[_BaseWindowedHook], List[WindowHookStats], List[str]]:
    """
    Register one or more hooks for a condition.
    components values:
      - "attn_delta"
      - "mlp_delta"
      - "res_in"
      - "res_out"
      - "rand_attn_delta"

    Returns:
      handles, hooks, stats, resolved_components
    """
    layers, _arch = EP.get_model_layers(model)
    if int(layer_idx) < 0 or int(layer_idx) >= len(layers):
        raise ValueError(f"layer_idx={layer_idx} out of range: num_layers={len(layers)}")
    layer = layers[int(layer_idx)]

    handles: List[Any] = []
    hooks: List[_BaseWindowedHook] = []
    stats: List[WindowHookStats] = []
    resolved_components: List[str] = []

    for comp in components:
        comp = str(comp).strip().lower()
        if not comp:
            continue
        if comp not in component_bases:
            raise ValueError(f"No basis provided for component={comp!r} in condition={condition_name!r}")
        Q_np = component_bases[comp]
        if Q_np is None:
            continue

        # Shared: attn_delta and random_attn_delta use the same insertion point.
        if comp in {"attn_delta", "rand_attn_delta"}:
            mod, attr = _find_attn_module(layer)
            stats_obj = WindowHookStats(
                f"{condition_name}_{comp}",
                location=comp,
                module_ref=f"layers[{int(layer_idx)}].{attr}",
            )
            hk = WindowedLastTokenProjectionHook(Q_np, float(alpha), int(window_start), int(window_end), stats_obj)
            h = mod.register_forward_hook(hk)
            handles.append(h)
            hooks.append(hk)
            stats.append(stats_obj)
            resolved_components.append(comp)
            continue

        if comp == "mlp_delta":
            mod, attr = _find_mlp_module(layer)
            stats_obj = WindowHookStats(
                f"{condition_name}_{comp}",
                location=comp,
                module_ref=f"layers[{int(layer_idx)}].{attr}",
            )
            hk = WindowedLastTokenProjectionHook(Q_np, float(alpha), int(window_start), int(window_end), stats_obj)
            h = mod.register_forward_hook(hk)
            handles.append(h)
            hooks.append(hk)
            stats.append(stats_obj)
            resolved_components.append(comp)
            continue

        if comp == "res_in":
            stats_obj = WindowHookStats(f"{condition_name}_{comp}", location="res_in", module_ref=f"layers[{int(layer_idx)}]")
            hk = WindowedLastTokenPreInputProjectionHook(Q_np, float(alpha), int(window_start), int(window_end), stats_obj)
            h = layer.register_forward_pre_hook(hk)
            handles.append(h)
            hooks.append(hk)
            stats.append(stats_obj)
            resolved_components.append(comp)
            continue

        if comp == "res_out":
            stats_obj = WindowHookStats(f"{condition_name}_{comp}", location="res_out", module_ref=f"layers[{int(layer_idx)}]")
            hk = WindowedLastTokenProjectionHook(Q_np, float(alpha), int(window_start), int(window_end), stats_obj)
            h = layer.register_forward_hook(hk)
            handles.append(h)
            hooks.append(hk)
            stats.append(stats_obj)
            resolved_components.append(comp)
            continue

        raise ValueError(f"Unknown component={comp!r} (condition={condition_name!r})")

    return handles, hooks, stats, resolved_components


# -----------------------------------------------------------------------------
# Forced-choice eval for one condition
# -----------------------------------------------------------------------------
def _is_correct_any(task: str, pred: str, gold: str) -> bool:
    try:
        return bool(EP.is_correct(task, pred, gold))
    except Exception:
        return _safe_upper(pred) == _safe_upper(gold)


@torch.no_grad()
def forced_choice_logprob_eval_condition(
    model,
    tok,
    examples: List[EP.Example],
    task: str,
    *,
    layer_idx: int,
    component_bases: Dict[str, np.ndarray],
    components: Sequence[str],
    alpha: float,
    window_start: int,
    window_end: int,
    batch_size: int,
    max_prompt_len: int,
    warmup_token_ids: Optional[np.ndarray],
    answer_prefix: str,
    prefix_mode: str = "auto",
    condition_name: str = "cond",
) -> Dict[str, Any]:
    """
    Same protocol as EP.forced_choice_logprob_eval, but with a condition-specific hook set.
    """
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
    cands = EP.candidate_strings(task)
    if len(cands) == 0:
        raise ValueError(f"Task '{task}' has no forced-choice candidates.")

    cand_ids_list = [EP.cand_token_ids(tok, s) for s in cands]
    cand_lens = np.array([max(1, len(x)) for x in cand_ids_list], dtype=np.float32)
    answer_prefix = EP.normalize_answer_prefix(answer_prefix)

    handles: List[Any] = []
    hooks: List[_BaseWindowedHook] = []
    hook_stats: List[WindowHookStats] = []

    if components:
        handles, hooks, hook_stats, resolved = _register_condition_hooks(
            model,
            layer_idx=int(layer_idx),
            alpha=float(alpha),
            window_start=int(window_start),
            window_end=int(window_end),
            components=components,
            component_bases=component_bases,
            condition_name=condition_name,
        )
        if not resolved:
            components = ()
    else:
        resolved = []

    N = len(prompts)
    C = len(cands)
    correct = np.zeros(N, dtype=np.float32)
    gold_logprob_sum = np.full(N, np.nan, dtype=np.float32)
    gold_margin_sum = np.full(N, np.nan, dtype=np.float32)
    gold_logprob_mean = np.full(N, np.nan, dtype=np.float32)
    gold_margin_mean = np.full(N, np.nan, dtype=np.float32)
    entropy = np.full(N, np.nan, dtype=np.float32)
    top1_gap = np.full(N, np.nan, dtype=np.float32)
    pred_labels: List[str] = [""] * N

    def gold_index_for(gold_label: str) -> int:
        for j, c in enumerate(cands):
            if _is_correct_any(task, c, gold_label):
                return j
        g = _safe_upper(gold_label)
        for j, c in enumerate(cands):
            if _safe_upper(c) == g:
                return j
        return -1

    def score_subset(idxs: np.ndarray, do_prefix: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if hooks:
            for hk in hooks:
                hk.reset_steps()

        sub_prompts = [prompts[int(j)] for j in idxs.tolist()]
        sub_golds = [golds[int(j)] for j in idxs.tolist()]
        inputs = tok(sub_prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_prompt_len).to(device)
        ids = inputs["input_ids"]
        attn = inputs["attention_mask"]
        B = ids.shape[0]

        warm_ids = None
        W = 0
        if warmup_token_ids is not None:
            warm = warmup_token_ids[idxs]
            if warm is not None and warm.size > 0:
                warm_ids = torch.tensor(warm, dtype=torch.long, device=device)
                W = int(warm_ids.shape[1])

        past, logits = EP.cache_decode_aligned_boundary(model, ids, attn)

        if warm_ids is not None and W > 0:
            for t in range(W):
                tok_t = warm_ids[:, t : t + 1]
                attn = torch.cat([attn, torch.ones((B, 1), device=device, dtype=attn.dtype)], dim=1)
                out = model(input_ids=tok_t, attention_mask=attn, use_cache=True, past_key_values=past)
                logits = out.logits[:, -1, :]
                past = out.past_key_values

        if do_prefix and answer_prefix:
            prefix_ids = tok.encode(answer_prefix, add_special_tokens=False)
            for pid in prefix_ids:
                inp = torch.full((B, 1), int(pid), dtype=torch.long, device=device)
                attn = torch.cat([attn, torch.ones((B, 1), device=device, dtype=attn.dtype)], dim=1)
                out = model(input_ids=inp, attention_mask=attn, use_cache=True, past_key_values=past)
                logits = out.logits[:, -1, :]
                past = out.past_key_values

        # Candidate logprobs on the final score token
        scores = torch.full((B, C), float("-inf"), device=device)
        logp0 = torch.log_softmax(logits, dim=-1)
        for ci, cand_ids in enumerate(cand_ids_list):
            if len(cand_ids) == 0:
                continue
            lp = logp0[:, int(cand_ids[0])]
            if len(cand_ids) > 1:
                past_c = past
                attn_c = attn
                logits_c = logits
                for ti, tok_id in enumerate(cand_ids):
                    if ti == 0:
                        continue
                    inp = torch.full((B, 1), int(tok_id), dtype=torch.long, device=device)
                    attn_c = torch.cat([attn_c, torch.ones((B, 1), device=device, dtype=attn_c.dtype)], dim=1)
                    out = model(input_ids=inp, attention_mask=attn_c, use_cache=True, past_key_values=past_c)
                    logits_c = out.logits[:, -1, :]
                    past_c = out.past_key_values
                    lp = lp + torch.log_softmax(logits_c, dim=-1)[:, int(tok_id)]
            scores[:, ci] = lp

        pred_idx = torch.argmax(scores, dim=1)
        probs = torch.softmax(scores, dim=1)
        ent = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)
        top2 = torch.topk(scores, k=min(2, C), dim=1).values
        gap = top2[:, 0] - (top2[:, 1] if top2.shape[1] > 1 else top2[:, 0])

        gidx = np.array([gold_index_for(str(g)) for g in sub_golds], dtype=np.int64)
        return (
            scores.detach().cpu().numpy().astype(np.float32, copy=False),
            pred_idx.detach().cpu().numpy().astype(np.int64, copy=False),
            ent.detach().cpu().numpy().astype(np.float32, copy=False),
            gap.detach().cpu().numpy().astype(np.float32, copy=False),
            gidx,
        )

    try:
        for i in range(0, N, int(batch_size)):
            j0 = i
            j1 = min(N, i + int(batch_size))
            idxs = np.arange(j0, j1, dtype=np.int64)
            B = int(idxs.shape[0])

            W = 0
            if warmup_token_ids is not None and warmup_token_ids.shape[1] > 0:
                W = int(warmup_token_ids.shape[1])

            if prefix_mode == "never":
                need_prefix = np.zeros(B, dtype=bool)
            elif prefix_mode == "always":
                need_prefix = np.ones(B, dtype=bool) if bool(answer_prefix) else np.zeros(B, dtype=bool)
            else:  # auto
                if not answer_prefix:
                    need_prefix = np.zeros(B, dtype=bool)
                elif W > 0:
                    need_prefix = np.ones(B, dtype=bool)
                else:
                    need_prefix = np.array([not EP.prompt_endswith_prefix(prompts[int(j)], answer_prefix) for j in idxs.tolist()], dtype=bool)

            for flag in [False, True]:
                sub = idxs[need_prefix == flag]
                if sub.size == 0:
                    continue
                scores_sum, pred_idx, ent, gap, gidx = score_subset(sub, do_prefix=bool(flag))
                for row_pos, ex_idx in enumerate(sub.tolist()):
                    pred_label = cands[int(pred_idx[row_pos])]
                    gold_label = str(golds[int(ex_idx)])
                    pred_labels[int(ex_idx)] = str(pred_label)
                    correct[int(ex_idx)] = 1.0 if _is_correct_any(task, str(pred_label), gold_label) else 0.0
                    entropy[int(ex_idx)] = float(ent[row_pos])
                    top1_gap[int(ex_idx)] = float(gap[row_pos])
                    gi = int(gidx[row_pos])
                    if gi < 0:
                        continue
                    row = scores_sum[row_pos, :]
                    gold_lp = float(row[gi])
                    best_wrong = float(np.max(np.delete(row, gi))) if C > 1 else float("-inf")
                    gold_logprob_sum[int(ex_idx)] = gold_lp
                    gold_margin_sum[int(ex_idx)] = gold_lp - best_wrong
                    den = float(cand_lens[gi]) if 0 <= gi < int(cand_lens.size) else float("nan")
                    gold_logprob_mean[int(ex_idx)] = gold_lp / den
                    gold_margin_mean[int(ex_idx)] = (gold_lp - best_wrong) / den

        out: Dict[str, Any] = {
            "acc": float(correct.mean()) if N > 0 else float("nan"),
            "correct": correct.tolist(),
            "hook_stats": [h.report() for h in hook_stats],
            "hook_components": [s.location for s in hook_stats],
            "cands": cands,
            "preds": pred_labels,
            "golds": [str(g) for g in golds],
            "metrics_summary": {
                "gold_logprob_sum_mean": float(np.nanmean(gold_logprob_sum)) if N > 0 else float("nan"),
                "gold_margin_sum_mean": float(np.nanmean(gold_margin_sum)) if N > 0 else float("nan"),
                "gold_logprob_mean_mean": float(np.nanmean(gold_logprob_mean)) if N > 0 else float("nan"),
                "gold_margin_mean_mean": float(np.nanmean(gold_margin_mean)) if N > 0 else float("nan"),
                "entropy_mean": float(np.nanmean(entropy)) if N > 0 else float("nan"),
                "top1_gap_mean": float(np.nanmean(top1_gap)) if N > 0 else float("nan"),
            },
        }
        return out
    finally:
        _remove_hooks(handles)


# -----------------------------------------------------------------------------
# Summaries
# -----------------------------------------------------------------------------
@dataclass
class TemplateStats:
    mean: float
    worst: float
    best: float
    std: float
    range: float
    regret_at_1: float
    worst_gap: float
    best_gap: float


def _summarize_template_stats(accs: List[float]) -> TemplateStats:
    a = np.asarray(accs, dtype=np.float64)
    mean = float(np.mean(a)) if a.size else float("nan")
    worst = float(np.min(a)) if a.size else float("nan")
    best = float(np.max(a)) if a.size else float("nan")
    std = float(np.std(a)) if a.size else float("nan")
    rng = float(best - worst) if a.size else float("nan")
    regret = float(best - mean) if a.size else float("nan")
    worst_gap = float(mean - worst) if a.size else float("nan")
    best_gap = float(best - worst) if a.size else float("nan")
    return TemplateStats(mean, worst, best, std, rng, regret, worst_gap, best_gap)


def _macro_avg(stats_by_task: Dict[str, TemplateStats]) -> TemplateStats:
    keys = list(stats_by_task.keys())
    if not keys:
        return _summarize_template_stats([])
    mean = float(np.mean([stats_by_task[t].mean for t in keys]))
    worst = float(np.mean([stats_by_task[t].worst for t in keys]))
    best = float(np.mean([stats_by_task[t].best for t in keys]))
    std = float(np.mean([stats_by_task[t].std for t in keys]))
    rng = float(np.mean([stats_by_task[t].range for t in keys]))
    regret = float(np.mean([stats_by_task[t].regret_at_1 for t in keys]))
    worst_gap = float(np.mean([stats_by_task[t].worst_gap for t in keys]))
    best_gap = float(np.mean([stats_by_task[t].best_gap for t in keys]))
    return TemplateStats(mean, worst, best, std, rng, regret, worst_gap, best_gap)


def _load_basis_from_npz(path: str, *, k: int = 0, key: str = "") -> np.ndarray:
    z = np.load(os.path.expanduser(path))
    if key and key in z:
        Q = z[key]
    elif "Q_shared" in z:
        Q = z["Q_shared"]
    elif "Q" in z:
        Q = z["Q"]
    else:
        raise KeyError(f"Could not find basis array in npz. Available keys: {list(z.keys())}")
    Q = np.asarray(Q, dtype=np.float32)
    if int(k) > 0:
        Q = Q[:, : int(k)]
    return EP.orthonormalize_np(Q)


def _rand_orthonormal_basis(dim: int, k: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    k_eff = int(k) if int(k) > 0 else int(dim)
    z = rng.standard_normal((int(dim), max(1, int(k_eff))), dtype=np.float64)
    z = z - z.mean(axis=0, keepdims=True)
    return EP.orthonormalize_np(z.astype(np.float32))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()

    # Model
    ap.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--trust_remote_code", type=int, default=0, choices=[0, 1])

    # Layer + basis
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--basis_npz", type=str, default="", help="npz path with Q_shared.")
    ap.add_argument("--basis_key", type=str, default="", help="Optional: npz key override.")
    ap.add_argument("--k_basis", type=int, default=32, help="Use first k columns (0=all).")
    ap.add_argument("--alpha_remove", type=float, default=1.0)

    # Tasks / data
    ap.add_argument("--tasks", type=str, default="commonsenseqa,arc_challenge,openbookqa,qasc,logiqa,boolq")
    ap.add_argument("--eval_n", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--template_seeds", type=str, default="1234,2345,3456")
    ap.add_argument("--template_randomization", type=int, default=1, choices=[0, 1])
    ap.add_argument("--shuffle_choices", type=int, default=0, choices=[0, 1], help="Keep choices fixed to isolate template variance.")
    ap.add_argument("--add_answer_prefix", type=int, default=1, choices=[0, 1])
    ap.add_argument("--answer_prefix", type=str, default="\\nFinal answer:")

    # Eval
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_prompt_len", type=int, default=512)

    # Forced-choice warmup + prefix anchoring
    ap.add_argument("--fc_warmup_tokens", type=int, default=32)
    ap.add_argument("--fc_warmup_decoding", type=str, default="greedy", choices=["greedy", "sample"])
    ap.add_argument("--fc_warmup_ban_eos", type=int, default=0, choices=[0, 1])
    ap.add_argument("--fc_warmup_seed", type=int, default=123)
    ap.add_argument("--fc_prefix_mode", type=str, default="auto", choices=["auto", "always", "never"])
    ap.add_argument("--fc_answer_prefix", type=str, default="\\nFinal answer:")

    # Window control (late + optional placebo)
    ap.add_argument("--window_n", type=int, default=4, help="N for late/mid windows.")
    ap.add_argument("--add_mid_window", type=int, default=1, choices=[0, 1], help="Run mid-N placebo too.")
    ap.add_argument(
        "--mid_window_start",
        type=int,
        default=-1,
        help="Optional explicit mid-window start (decode-step coords). If <0, auto center.",
    )
    ap.add_argument("--exclude_final_step", type=int, default=1, choices=[0, 1], help="Exclude final boundary step.")

    # Residual/path options
    ap.add_argument("--include_residual_in", type=int, default=1, choices=[0, 1], help="Add res_in condition.")
    ap.add_argument("--include_residual_out", type=int, default=1, choices=[0, 1], help="Add res_out condition.")
    ap.add_argument("--include_sum_delta", type=int, default=1, choices=[0, 1], help="Add attn+mlp sum_delta condition.")
    ap.add_argument("--add_random_control", type=int, default=0, choices=[0, 1], help="Add random-attn-delta control.")
    ap.add_argument("--random_seed", type=int, default=2024, help="Seed for random-control basis.")

    # Output
    ap.add_argument("--out_dir", type=str, default="results/rebuttal_mechanism/m5_residual_path")
    ap.add_argument("--tag", type=str, default="")

    args = ap.parse_args()

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but not available.\n"
            f"torch={torch.__version__} torch.version.cuda={getattr(torch.version, 'cuda', None)}"
        )

    tasks = _split_csv(args.tasks)
    if not tasks:
        raise ValueError("Empty --tasks")
    template_seeds = [int(x) for x in _split_csv(args.template_seeds)]
    if not template_seeds:
        raise ValueError("Empty --template_seeds")

    out_dir = os.path.expanduser(str(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)
    tag = ("_" + str(args.tag).strip()) if str(args.tag).strip() else ""

    EP.set_global_seed(int(args.seed))
    model, tok = EP.load_model_and_tokenizer(
        args.model,
        args.device,
        args.dtype,
        trust_remote_code=bool(args.trust_remote_code),
    )

    answer_prefix = _decode_backslash_escapes(str(args.answer_prefix))
    fc_answer_prefix = _decode_backslash_escapes(str(args.fc_answer_prefix))

    if not str(args.basis_npz).strip():
        raise ValueError("--basis_npz is required.")

    Q = _load_basis_from_npz(str(args.basis_npz), k=int(args.k_basis), key=str(args.basis_key).strip())
    model_hidden = int(getattr(model.config, "hidden_size", Q.shape[0]))
    k_effective = int(Q.shape[1]) if Q.ndim == 2 else int(Q.shape[0])
    Q_rand = _rand_orthonormal_basis(model_hidden, k_effective, seed=int(args.random_seed))
    basis_meta = {"basis_npz": str(args.basis_npz), "basis_key": str(args.basis_key), "k_basis": int(args.k_basis)}

    # Decode-step indexing for forced-choice:
    W = int(args.fc_warmup_tokens)
    prefix_mode = str(args.fc_prefix_mode).strip().lower()
    if prefix_mode == "auto" and W == 0 and bool(args.exclude_final_step):
        raise ValueError(
            "With --fc_prefix_mode auto and --fc_warmup_tokens 0, prefix insertion differs per prompt "
            "and final-step windowing can be unstable. Set W>=1, or set --fc_prefix_mode always/never, or "
            "--exclude_final_step 0."
        )
    P = 0
    if prefix_mode != "never" and str(fc_answer_prefix).strip():
        P = len(tok.encode(EP.normalize_answer_prefix(str(fc_answer_prefix)), add_special_tokens=False))
    total_steps = 1 + max(0, W) + max(0, P)
    windowable_steps = int(total_steps - (1 if bool(args.exclude_final_step) else 0))
    if windowable_steps <= 0:
        raise ValueError("windowable_steps must be >0. Increase warmup/prefix length or set --exclude_final_step 0.")

    Nw = int(args.window_n)
    if Nw <= 0:
        raise ValueError("--window_n must be >0")
    if windowable_steps < Nw:
        raise ValueError(
            f"Need windowable_steps >= window_n, got windowable_steps={windowable_steps}, window_n={Nw}."
        )

    late0, late1 = int(windowable_steps - Nw), int(windowable_steps)
    windows: Dict[str, Tuple[int, int]] = {f"late{Nw}": (late0, late1)}
    if bool(args.add_mid_window):
        if int(args.mid_window_start) >= 0:
            mid0 = int(args.mid_window_start)
        else:
            mid0 = int(max(0, (windowable_steps // 2) - (Nw // 2)))
            mid0 = int(min(mid0, windowable_steps - Nw))
        if mid0 < 0 or mid0 + Nw > windowable_steps:
            raise ValueError("Bad mid window. Use --mid_window_start or disable --add_mid_window.")
        windows[f"mid{Nw}"] = (mid0, mid0 + Nw)

    # Base condition specs
    # key -> components; each component has a basis key.
    # random control is optional and uses Q_rand.
    cond_components: Dict[str, List[str]] = {}
    cond_bases: Dict[str, Dict[str, np.ndarray]] = {}

    # Branch-only and residual-path controls
    cond_components["attn_delta"] = ["attn_delta"]
    cond_bases["attn_delta"] = {"attn_delta": Q}

    cond_components["mlp_delta"] = ["mlp_delta"]
    cond_bases["mlp_delta"] = {"mlp_delta": Q}

    cond_components["res_in"] = ["res_in"]
    cond_bases["res_in"] = {"res_in": Q}

    cond_components["res_out"] = ["res_out"]
    cond_bases["res_out"] = {"res_out": Q}

    if bool(args.include_sum_delta):
        cond_components["attn_mlp_sum_delta"] = ["attn_delta", "mlp_delta"]
        cond_bases["attn_mlp_sum_delta"] = {"attn_delta": Q, "mlp_delta": Q}

    if bool(args.add_random_control):
        cond_components["random_attn_delta"] = ["rand_attn_delta"]
        cond_bases["random_attn_delta"] = {"rand_attn_delta": Q_rand}

    if not bool(args.include_residual_in):
        cond_components.pop("res_in", None)
        cond_bases.pop("res_in", None)
    if not bool(args.include_residual_out):
        cond_components.pop("res_out", None)
        cond_bases.pop("res_out", None)

    # Optional to simplify: disable branch-sum if only one branch requested.
    if len(cond_components) == 0:
        raise ValueError("No intervention conditions configured.")

    # Expand to windowed conditions
    conds: Dict[str, Dict[str, Any]] = {"baseline": {"components": [], "bases": {}, "window": None}}
    cond_order: List[str] = ["baseline"]
    for wname, (w0, w1) in windows.items():
        for cname, comps in cond_components.items():
            cond_key = f"{cname}_{wname}"
            conds[cond_key] = {
                "components": list(comps),
                "bases": cond_bases[cname],
                "window": (int(w0), int(w1)),
            }
            cond_order.append(cond_key)

    # Only forced-choice tasks
    tasks_eff = [t for t in tasks if len(EP.candidate_strings(t)) > 0]
    if not tasks_eff:
        raise RuntimeError("No tasks with candidates in --tasks.")

    acc_by: Dict[str, Dict[str, Dict[int, float]]] = {t: {c: {} for c in conds.keys()} for t in tasks_eff}
    hook_by: Dict[str, Dict[str, Dict[int, Any]]] = {t: {c: {} for c in conds.keys()} for t in tasks_eff}
    metrics_by: Dict[str, Dict[str, Dict[int, Any]]] = {t: {c: {} for c in conds.keys()} for t in tasks_eff}

    for tseed in template_seeds:
        _sub_by, eval_by_all, meta_by = load_selected_tasks(
            tasks=tasks_eff,
            n_subspace=1,
            n_eval=max(1, int(args.eval_n)),
            seed=int(args.seed),
            template_seed=int(tseed),
            template_randomization=bool(args.template_randomization),
            shuffle_choices=bool(args.shuffle_choices),
            add_answer_prefix=bool(args.add_answer_prefix),
            answer_prefix=str(answer_prefix),
        )

        for task in tasks_eff:
            examples = eval_by_all[task]
            warmup_token_ids = None
            if int(args.fc_warmup_tokens) > 0:
                warmup_token_ids = EP.precompute_fc_warmup_tokens(
                    model,
                    tok,
                    [ex.prompt for ex in examples],
                    warmup_tokens=int(args.fc_warmup_tokens),
                    batch_size=int(args.batch_size),
                    max_prompt_len=int(args.max_prompt_len),
                    decoding=str(args.fc_warmup_decoding),
                    temperature=1.0,
                    top_p=1.0,
                    top_k=0,
                    ban_eos=bool(args.fc_warmup_ban_eos),
                    seed=EP.stable_int_seed(args.seed, tseed, task, "warmup", int(args.fc_warmup_seed)),
                )

            for cond in cond_order:
                meta = conds[cond]
                comps = meta.get("components", [])
                bases = meta.get("bases", {})
                w = meta.get("window")
                if w is None:
                    w0, w1 = 0, 1
                else:
                    w0, w1 = int(w[0]), int(w[1])

                out_fc = forced_choice_logprob_eval_condition(
                    model,
                    tok,
                    examples,
                    task,
                    layer_idx=int(args.layer),
                    component_bases=bases,
                    components=comps,
                    alpha=float(args.alpha_remove) if comps else 0.0,
                    window_start=int(w0),
                    window_end=int(w1) if comps else 1,
                    batch_size=int(args.batch_size),
                    max_prompt_len=int(args.max_prompt_len),
                    warmup_token_ids=warmup_token_ids,
                    answer_prefix=str(fc_answer_prefix),
                    prefix_mode=str(args.fc_prefix_mode),
                    condition_name=str(cond),
                )
                acc_by[task][cond][int(tseed)] = float(out_fc["acc"])
                hook_by[task][cond][int(tseed)] = out_fc.get("hook_stats", [])
                metrics_by[task][cond][int(tseed)] = out_fc.get("metrics_summary", {})

            if str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()

    stats_by_task_cond: Dict[str, Dict[str, TemplateStats]] = {t: {} for t in tasks_eff}
    for task in tasks_eff:
        for cond in conds.keys():
            accs = [acc_by[task][cond][int(s)] for s in template_seeds if int(s) in acc_by[task][cond]]
            stats_by_task_cond[task][cond] = _summarize_template_stats(accs)

    macro_by_cond: Dict[str, TemplateStats] = {}
    for cond in conds.keys():
        macro_by_cond[cond] = _macro_avg({t: stats_by_task_cond[t][cond] for t in tasks_eff})

    attribution: Dict[str, Any] = {
        "macro_std": {c: float(macro_by_cond[c].std) for c in cond_order},
        "macro_range": {c: float(macro_by_cond[c].range) for c in cond_order},
        "std_improvement_vs_baseline": {
            c: float(macro_by_cond["baseline"].std - macro_by_cond[c].std) for c in cond_order if c != "baseline"
        },
        "range_improvement_vs_baseline": {
            c: float(macro_by_cond["baseline"].range - macro_by_cond[c].range) for c in cond_order if c != "baseline"
        },
        "windows": {k: [int(v[0]), int(v[1])] for k, v in windows.items()},
        "conditions": cond_components,
    }

    out = {
        "config": {
            "model": args.model,
            "device": args.device,
            "dtype": args.dtype,
            "trust_remote_code": bool(args.trust_remote_code),
            "layer": int(args.layer),
            "alpha_remove": float(args.alpha_remove),
            "tasks": tasks_eff,
            "eval_n": int(args.eval_n),
            "seed": int(args.seed),
            "template_seeds": template_seeds,
            "template_randomization": bool(args.template_randomization),
            "shuffle_choices": bool(args.shuffle_choices),
            "add_answer_prefix": bool(args.add_answer_prefix),
            "answer_prefix": str(answer_prefix),
            "forced_choice": {
                "fc_warmup_tokens": int(args.fc_warmup_tokens),
                "fc_warmup_decoding": str(args.fc_warmup_decoding),
                "fc_warmup_ban_eos": bool(args.fc_warmup_ban_eos),
                "fc_warmup_seed": int(args.fc_warmup_seed),
                "fc_prefix_mode": str(args.fc_prefix_mode),
                "fc_answer_prefix": str(fc_answer_prefix),
            },
            "window": {
                "window_n": int(Nw),
                "add_mid_window": bool(args.add_mid_window),
                "mid_window_start": int(args.mid_window_start),
                "total_steps": int(total_steps),
                "exclude_final_step": bool(args.exclude_final_step),
                "windowable_steps": int(windowable_steps),
                "windows": {k: [int(v[0]), int(v[1])] for k, v in windows.items()},
            },
            "residual": {
                "include_residual_in": bool(args.include_residual_in),
                "include_residual_out": bool(args.include_residual_out),
                "include_sum_delta": bool(args.include_sum_delta),
                "add_random_control": bool(args.add_random_control),
                "random_seed": int(args.random_seed),
            },
            "basis": basis_meta,
            "cond_order": list(cond_order),
            "conditions": {
                k: {"components": v.get("components"), "uses_basis": bool(v.get("bases"))} for k, v in conds.items()
            },
        },
        "dataset_meta": meta_by if "meta_by" in locals() else {},
        "acc_by_task_cond_seed": acc_by,
        "hook_by_task_cond_seed": hook_by,
        "metrics_by_task_cond_seed": metrics_by,
        "template_stats_by_task_cond": {
            t: {c: stats_by_task_cond[t][c].__dict__ for c in stats_by_task_cond[t].keys()} for t in stats_by_task_cond.keys()
        },
        "macro_by_cond": {c: macro_by_cond[c].__dict__ for c in macro_by_cond.keys()},
        "attribution": attribution,
    }

    base = f"exp_5_residual_path_late{Nw}" + (f"_mid{Nw}" if bool(args.add_mid_window) else "")
    out_json = os.path.join(out_dir, f"{base}_layer{int(args.layer)}{tag}.json")
    _atomic_json_dump(out, out_json)

    def _table_for(metric: str, title: str) -> str:
        rows = []
        for task in tasks_eff:
            r = [task]
            for c in cond_order:
                st = stats_by_task_cond[task][c]
                r.append(_pct(getattr(st, metric)))
            rows.append(r)
        macro_row = ["macro"]
        for c in cond_order:
            macro_row.append(_pct(getattr(macro_by_cond[c], metric)))
        rows.append(macro_row)
        return "\n".join([f"### {title}", _md_table(rows, ["Task"] + cond_order), ""])

    md: List[str] = []
    md.append("# Exp-5 (M5): Residual-path attribution (late window + placebo mid)")
    md.append("")
    md.append("Hypothesis: decode-shared effect may live in residual stream merge points, not a single branch.")
    md.append("")
    md.append("Intervention components:")
    md.append("- `attn_delta`: hook on `layer.self_attn` output (attention branch delta)")
    md.append("- `mlp_delta` : hook on `layer.mlp` output (MLP branch delta)")
    md.append("- `attn_mlp_sum_delta`: simultaneously remove shared component from both deltas")
    md.append("- `res_in`    : pre-hook on layer input residual stream")
    md.append("- `res_out`   : post-hook on layer output residual stream")
    if bool(args.add_random_control):
        md.append("- `random_attn_delta`: random subspace control (same rank as Q)")
    md.append("")
    md.append("Windows (decode-step coords; applied only on KV-cached decode passes):")
    for wname, (w0, w1) in windows.items():
        md.append(f"- `{wname}`: [{int(w0)},{int(w1)})")
    md.append("")
    md.append("Decode-step indexing (forced-choice):")
    md.append("- step 0: decode-aligned boundary (prompt[-1] with past_key_values)")
    md.append(f"- steps 1..{W}: warmup teacher-forcing tokens (W={W})")
    md.append(f"- steps {W+1}..{W+P}: answer_prefix teacher-forcing tokens (P={P})")
    md.append(f"- total_steps = {total_steps}")
    if bool(args.exclude_final_step):
        md.append(f"- windowable_steps = total_steps - 1 = {windowable_steps} (exclude final pre-scoring step)")
    md.append("")
    md.append("## Config")
    md.append("```json")
    md.append(json.dumps(out["config"], ensure_ascii=False, indent=2, default=_json_default))
    md.append("```")
    md.append("")
    md.append("## Results (across template seeds)")
    md.append(_table_for("mean", "Mean accuracy (%) across templates"))
    md.append(_table_for("worst", "Worst-case accuracy (%) across templates"))
    md.append(_table_for("std", "Template std of accuracy (%) (lower is better)"))
    md.append(_table_for("range", "Template range of accuracy (%) (lower is better)"))
    md.append("## Attribution summary (macro; higher improvement = better)")
    md.append("```json")
    md.append(json.dumps(attribution, ensure_ascii=False, indent=2))
    md.append("```")
    md.append("")
    md.append(f"JSON: `{os.path.relpath(out_json, ROOT_DIR)}`")

    md_path = os.path.join(out_dir, f"{base}_layer{int(args.layer)}{tag}.md")
    _atomic_text_dump("\n".join(md).rstrip() + "\n", md_path)

    print(f"[Saved] {out_json}")
    print(f"[Saved] {md_path}")


if __name__ == "__main__":
    main()
