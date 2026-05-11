# -*- coding: utf-8 -*-
"""
exp_6_stepwise_delta_shared_alignment.py

Mechanism experiment M6 (direct geometry control):
- Collect decode-time delta streams of attn/MLP branches at layer L.
- Measure step-wise projection energy into shared basis Q (from exp-1).
- Measure step-wise subspace alignment via principal angles to Q
  (using a capped state reservoir to keep runtime stable).

This is a non-causal diagnostic aimed at strengthening the residual-path narrative:
it asks whether both branches project to Q in the same localized steps,
or whether attribution differences are weak/noisy enough to support a distributed view.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# ----------------------------------------------------------------------------
# Repo-local imports
# ----------------------------------------------------------------------------
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

for _p in [SRC_DIR, REASONING_DIR, ROOT_DIR]:
    if _p not in sys.path:
        sys.path.append(_p)

try:
    import eval_perf as EP  # reasoning/eval_perf.py
except Exception as e:  # pragma: no cover
    raise RuntimeError("Failed to import `reasoning/eval_perf.py` as module `eval_perf`.") from e

try:
    from benchmark_dataloaders import load_selected_tasks  # src/benchmark_dataloaders.py
except Exception as e:  # pragma: no cover
    raise RuntimeError("Failed to import `src/benchmark_dataloaders.py` as module `benchmark_dataloaders`.") from e


# ----------------------------------------------------------------------------
# IO helpers
# ----------------------------------------------------------------------------
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


def _decode_backslash_escapes(s: str) -> str:
    return str(s).replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r")


def _md_table(rows: List[List[str]], header: List[str]) -> str:
    out = []
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "---|" * len(header))
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def _fmt(x: float, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "nan"
    return f"{float(x):.{nd}f}"


def _safe_mean_std(vals: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray([float(v) for v in vals], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.nanmean(arr)), float(np.nanstd(arr, ddof=0))


# ----------------------------------------------------------------------------
# Module collectors
# ----------------------------------------------------------------------------
class StepEnergyAccumulator:
    """Accumulates projection-energy and keeps a capped sample bank for subspace fit."""

    def __init__(self, q_shared: np.ndarray, *, max_state_samples: int, random_seed: int):
        self.q_shared = np.asarray(q_shared, dtype=np.float32)
        self.max_state_samples = max(0, int(max_state_samples))
        self.rng = np.random.default_rng(int(random_seed))
        self.n = 0
        self.sum_ratio = 0.0
        self.sumsq_ratio = 0.0
        self.align_rows: List[np.ndarray] = []

    def add_batch(self, x: np.ndarray) -> None:
        if x.size == 0:
            return
        x = np.asarray(x, dtype=np.float32)
        proj = x @ self.q_shared
        num = np.sum(proj * proj, axis=1)
        den = np.sum(x * x, axis=1) + 1e-12
        ratio = num / den

        self.n += int(x.shape[0])
        self.sum_ratio += float(np.nansum(ratio))
        self.sumsq_ratio += float(np.nansum(ratio * ratio))

        if self.max_state_samples <= 0:
            return
        m = int(len(self.align_rows))
        cap = int(self.max_state_samples)
        k = int(x.shape[0])
        if m < cap:
            take = min(k, cap - m)
            if take > 0:
                self.align_rows.extend([v.astype(np.float32, copy=True) for v in x[:take]])
                k = k - take
                x = x[take:]
            if k <= 0:
                return

        for row in x:
            m = int(len(self.align_rows))
            j = int(self.rng.integers(0, m + 1))
            if j < cap:
                if m < cap:
                    self.align_rows.append(row.astype(np.float32, copy=True))
                else:
                    self.align_rows[j] = row.astype(np.float32, copy=True)

    def mean_ratio(self) -> float:
        if self.n <= 0:
            return float("nan")
        return float(self.sum_ratio / self.n)

    def std_ratio(self) -> float:
        if self.n <= 1:
            return float("nan")
        mean = self.sum_ratio / self.n
        var = max(0.0, self.sumsq_ratio / self.n - mean * mean)
        return float(math.sqrt(var))

    def sample_matrix(self) -> Optional[np.ndarray]:
        if not self.align_rows:
            return None
        return np.stack(self.align_rows, axis=0)


def _orth_basis_from_states(X: np.ndarray, k: int) -> Optional[np.ndarray]:
    if X is None:
        return None
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        return None
    n, d = X.shape
    k = int(max(0, min(int(k), n, d)))
    if k <= 1:
        return None
    Xc = X - np.mean(X, axis=0, keepdims=True)
    try:
        _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    except Exception:
        return None
    if vt.size == 0:
        return None
    V = vt[:k].T
    if V.size == 0:
        return None
    return EP.orthonormalize_np(V)


class DecodeStepCollector:
    """Collects step-wise module outputs on decode steps (`seq_len == 1`)."""

    def __init__(self, name: str, q_shared: np.ndarray, step_count: int, max_state_samples: int, random_seed: int):
        self.name = str(name)
        self.q_shared = np.asarray(q_shared, dtype=np.float32)
        self.step_count = int(step_count)
        self.max_state_samples = int(max_state_samples)
        self.random_seed = int(random_seed)
        self.step = 0
        self.enabled = False
        self._accumulators: Dict[int, StepEnergyAccumulator] = {}
        self._seed = int(random_seed)
        self.reset()

    @property
    def accumulators(self) -> Dict[int, StepEnergyAccumulator]:
        return self._accumulators

    def reset(self) -> None:
        self._accumulators = {
            int(s): StepEnergyAccumulator(self.q_shared, max_state_samples=self.max_state_samples, random_seed=self.random_seed + 1009 * int(s) + 17)
            for s in range(int(self.step_count))
        }
        self.step = 0
        self.enabled = False

    def set_capture(self, enabled: bool, step: Optional[int] = None) -> None:
        self.enabled = bool(enabled)
        if step is not None:
            self.step = int(step)

    def set_step(self, step: int) -> None:
        self.step = int(step)

    def _step_acc(self) -> Optional[StepEnergyAccumulator]:
        return self._accumulators.get(int(self.step))

    def make_hook(self):
        def _hook(module, inputs, output):
            if not self.enabled:
                return output
            hs = output[0] if isinstance(output, tuple) else output
            if not isinstance(hs, torch.Tensor) or hs.ndim != 3:
                return output
            if hs.shape[1] != 1:
                return output
            x = hs[:, -1, :]
            if x.numel() == 0:
                return output
            acc = self._step_acc()
            if acc is None:
                return output
            try:
                xn = x.detach().float().to(torch.float32)
                acc.add_batch(xn.detach().cpu().numpy())
            except Exception:
                pass
            return output

        return _hook


def _find_layer_module(model, layer_idx: int, kind: str) -> Tuple[torch.nn.Module, str]:
    layers, _ = EP.get_model_layers(model)
    if int(layer_idx) < 0 or int(layer_idx) >= len(layers):
        raise ValueError(f"layer_idx={int(layer_idx)} out of range: num_layers={len(layers)}")
    layer = layers[int(layer_idx)]
    if kind == "attn":
        for attr in ["self_attn", "attn", "attention"]:
            if hasattr(layer, attr):
                mod = getattr(layer, attr)
                if isinstance(mod, torch.nn.Module):
                    return mod, f"layers[{int(layer_idx)}].{attr}"
        raise RuntimeError(f"Could not find attention module on layer={int(layer_idx)}")
    if kind == "mlp":
        for attr in ["mlp", "feed_forward", "ffn", "mlp_layer"]:
            if hasattr(layer, attr):
                mod = getattr(layer, attr)
                if isinstance(mod, torch.nn.Module):
                    return mod, f"layers[{int(layer_idx)}].{attr}"
        raise RuntimeError(f"Could not find MLP module on layer={int(layer_idx)}")
    raise ValueError(f"Unknown module kind {kind}")


def _load_basis(path: str, *, k: int, key: str = "") -> np.ndarray:
    z = np.load(os.path.expanduser(path))
    if key and key in z:
        Q = z[key]
    elif "Q_shared" in z:
        Q = z["Q_shared"]
    elif "Q" in z:
        Q = z["Q"]
    else:
        raise KeyError(f"Could not find basis array in {path}. Available keys: {list(z.keys())}")
    Q = np.asarray(Q, dtype=np.float32)
    if int(k) > 0:
        Q = Q[:, : int(k)]
    return EP.orthonormalize_np(Q)


@torch.no_grad()
def _collect_decode_steps(
    model,
    tok,
    prompts: List[str],
    examples_warmup_ids: Optional[np.ndarray],
    *,
    attn_collector: DecodeStepCollector,
    mlp_collector: DecodeStepCollector,
    fc_answer_prefix: str,
    max_prompt_len: int,
    warmup_tokens: int,
    batch_size: int,
    do_prefix: bool,
    answer_prefix: str,
    attention_mask: Optional[torch.Tensor] = None,
) -> None:
    device = next(model.parameters()).device
    model.eval()

    if warmup_tokens > 0 and examples_warmup_ids is None:
        raise RuntimeError("warmup_tokens > 0 but warmup tokens are None")

    P = 0
    if do_prefix and str(answer_prefix).strip():
        P = len(tok.encode(EP.normalize_answer_prefix(str(answer_prefix)), add_special_tokens=False))

    for i in range(0, len(prompts), int(batch_size)):
        batch_prompts = prompts[i : i + int(batch_size)]
        batch_size_b = len(batch_prompts)
        warm = None
        if examples_warmup_ids is not None:
            warm = torch.tensor(examples_warmup_ids[i : i + batch_size_b], dtype=torch.long, device=device)

        inputs = tok(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=int(max_prompt_len)).to(device)
        ids = inputs["input_ids"]
        attn = inputs["attention_mask"]
        if attn is None:
            attn = torch.ones_like(ids)

        # Step 0: decode-aligned boundary.
        attn_collector.set_capture(False)
        mlp_collector.set_capture(False)
        attn_collector.set_step(0)
        mlp_collector.set_step(0)
        if ids.shape[1] == 1:
            attn_collector.set_capture(True)
            mlp_collector.set_capture(True)
            out1 = model(input_ids=ids, attention_mask=attn, use_cache=True)
            past = out1.past_key_values
            logits = out1.logits
        else:
            out0 = model(input_ids=ids[:, :-1], attention_mask=attn[:, :-1], use_cache=True)
            attn_collector.set_capture(True)
            mlp_collector.set_capture(True)
            out1 = model(input_ids=ids[:, -1:], attention_mask=attn, use_cache=True, past_key_values=out0.past_key_values)
            past = out1.past_key_values
            logits = out1.logits

        if warmup_tokens == 0 and not do_prefix:
            attn_collector.set_capture(False)
            mlp_collector.set_capture(False)
            _ = logits
            continue

        step = 0
        if warmup_tokens > 0 and warm is not None:
            for t in range(warmup_tokens):
                step += 1
                tok_t = warm[:, t : t + 1]
                attn = torch.cat([attn, torch.ones((batch_size_b, 1), device=device, dtype=attn.dtype)], dim=1)
                attn_collector.set_step(step)
                mlp_collector.set_step(step)
                attn_collector.set_capture(True)
                mlp_collector.set_capture(True)
                out = model(input_ids=tok_t, attention_mask=attn, use_cache=True, past_key_values=past)
                logits = out.logits
                past = out.past_key_values
                _ = logits

        if do_prefix and str(fc_answer_prefix).strip():
            prefix_ids = tok.encode(EP.normalize_answer_prefix(str(fc_answer_prefix)), add_special_tokens=False)
            for pid in prefix_ids:
                step += 1
                inp = torch.full((batch_size_b, 1), int(pid), dtype=torch.long, device=device)
                attn = torch.cat([attn, torch.ones((batch_size_b, 1), device=device, dtype=attn.dtype)], dim=1)
                attn_collector.set_step(step)
                mlp_collector.set_step(step)
                attn_collector.set_capture(True)
                mlp_collector.set_capture(True)
                out = model(input_ids=inp, attention_mask=attn, use_cache=True, past_key_values=past)
                logits = out.logits
                past = out.past_key_values
                _ = logits

        attn_collector.set_capture(False)
        mlp_collector.set_capture(False)


def _align_summary(X: Optional[np.ndarray], q_shared: np.ndarray, k_basis: int) -> Dict[str, float]:
    if X is None or X.shape[0] < 2:
        return {"max_cos": float("nan"), "mean_cos": float("nan"), "min_cos": float("nan"), "fro_norm": float("nan")}

    Q_step = _orth_basis_from_states(X, k=min(int(k_basis), int(X.shape[1])))
    if Q_step is None or Q_step.size == 0:
        return {"max_cos": float("nan"), "mean_cos": float("nan"), "min_cos": float("nan"), "fro_norm": float("nan")}
    return EP.subspace_similarity(q_shared, Q_step)


def _build_window_dict(window_n: int, windowable_steps: int, add_mid_window: bool, mid_window_start: int) -> Dict[str, Tuple[int, int]]:
    w = int(window_n)
    if w <= 0:
        raise ValueError("window_n must be > 0")
    if windowable_steps < w:
        raise ValueError(f"Need windowable_steps >= window_n, got windowable_steps={windowable_steps}, window_n={w}")

    late0, late1 = windowable_steps - w, windowable_steps
    windows = {f"late{w}": (late0, late1)}

    if add_mid_window:
        if int(mid_window_start) >= 0:
            m0 = int(mid_window_start)
        else:
            m0 = int(max(0, (windowable_steps // 2) - (w // 2)))
            m0 = int(min(m0, windowable_steps - w))
        if m0 < 0 or m0 + w > windowable_steps:
            raise ValueError("Bad mid window; adjust --mid_window_start or disable --add_mid_window.")
        windows[f"mid{w}"] = (m0, m0 + w)

    return windows


def _aggregate_window(steps_dict: Dict[int, Dict[str, float]], s0: int, s1: int) -> Dict[str, float]:
    keys = [int(s) for s in range(int(s0), int(s1)) if int(s) in steps_dict]
    if not keys:
        return {
            "energy_mean": float("nan"),
            "energy_std": float("nan"),
            "align_max_cos": float("nan"),
            "align_mean_cos": float("nan"),
            "align_fro_norm": float("nan"),
        }
    em = [steps_dict[s].get("energy_mean", float("nan")) for s in keys]
    es = [steps_dict[s].get("energy_std", float("nan")) for s in keys]
    am = [steps_dict[s].get("align_max_cos", float("nan")) for s in keys]
    ame = [steps_dict[s].get("align_mean_cos", float("nan")) for s in keys]
    af = [steps_dict[s].get("align_fro_norm", float("nan")) for s in keys]

    return {
        "energy_mean": float(np.nanmean(em)) if len(em) else float("nan"),
        "energy_std": float(np.nanmean(es)) if len(es) else float("nan"),
        "align_max_cos": float(np.nanmean(am)) if len(am) else float("nan"),
        "align_mean_cos": float(np.nanmean(ame)) if len(ame) else float("nan"),
        "align_fro_norm": float(np.nanmean(af)) if len(af) else float("nan"),
    }


@dataclass
class TemplateStats:
    mean: float
    std: float
    n: int

    def as_dict(self) -> Dict[str, float]:
        return {"mean": float(self.mean), "std": float(self.std), "n": int(self.n)}


def _summary_from_vals(values: Sequence[float]) -> TemplateStats:
    arr = np.asarray([float(v) for v in values], dtype=np.float64)
    if arr.size == 0:
        return TemplateStats(float("nan"), float("nan"), 0)
    return TemplateStats(float(np.nanmean(arr)), float(np.nanstd(arr)), int(arr.size))


def main() -> None:
    ap = argparse.ArgumentParser()

    # Model
    ap.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--trust_remote_code", type=int, default=0, choices=[0, 1])

    # Layer + shared basis
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--basis_npz", type=str, required=True)
    ap.add_argument("--basis_key", type=str, default="")
    ap.add_argument("--k_basis", type=int, default=32)

    # Dataset
    ap.add_argument("--tasks", type=str, default="commonsenseqa,arc_challenge,openbookqa,qasc,logiqa,boolq")
    ap.add_argument("--eval_n", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--template_seeds", type=str, default="1234,2345,3456")
    ap.add_argument("--template_randomization", type=int, default=1, choices=[0, 1])
    ap.add_argument("--shuffle_choices", type=int, default=0, choices=[0, 1])
    ap.add_argument("--add_answer_prefix", type=int, default=1, choices=[0, 1])
    ap.add_argument("--answer_prefix", type=str, default="\\nFinal answer:")

    # Eval
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_prompt_len", type=int, default=512)

    # Forced-choice warmup / prefix schedule (for alignment probe)
    ap.add_argument("--fc_warmup_tokens", type=int, default=32)
    ap.add_argument("--fc_warmup_decoding", type=str, default="greedy", choices=["greedy", "sample"])
    ap.add_argument("--fc_warmup_ban_eos", type=int, default=0, choices=[0, 1])
    ap.add_argument("--fc_warmup_seed", type=int, default=123)
    ap.add_argument("--fc_prefix_mode", type=str, default="auto", choices=["auto", "always", "never"])
    ap.add_argument("--fc_answer_prefix", type=str, default="\\nFinal answer:")

    # Window + steps
    ap.add_argument("--window_n", type=int, default=4)
    ap.add_argument("--add_mid_window", type=int, default=1, choices=[0, 1])
    ap.add_argument("--mid_window_start", type=int, default=-1)
    ap.add_argument("--exclude_final_step", type=int, default=1, choices=[0, 1])

    # Geometry sampling controls
    ap.add_argument("--align_samples_per_step", type=int, default=128)
    ap.add_argument("--k_align", type=int, default=16)

    # Output
    ap.add_argument("--out_dir", type=str, default="results/rebuttal_mechanism/m6_stepwise_delta_alignment")
    ap.add_argument("--tag", type=str, default="")

    args = ap.parse_args()

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

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

    Q_shared = _load_basis(str(args.basis_npz), k=int(args.k_basis), key=str(args.basis_key).strip())
    k_basis = int(Q_shared.shape[1]) if Q_shared.ndim == 2 else int(Q_shared.shape[0])

    # Decode-step indexing used by warmup + answer prefix tokens.
    W = int(args.fc_warmup_tokens)
    P = 0
    if str(fc_answer_prefix).strip():
        P = len(tok.encode(EP.normalize_answer_prefix(str(fc_answer_prefix)), add_special_tokens=False))

    total_steps = 1 + max(0, W) + max(0, P)
    windowable_steps = int(total_steps - (1 if bool(args.exclude_final_step) else 0))
    if windowable_steps <= 0:
        raise ValueError("windowable_steps must be > 0. Increase warmup/prefix length or set --exclude_final_step 0.")

    windows = _build_window_dict(int(args.window_n), int(windowable_steps), bool(args.add_mid_window), int(args.mid_window_start))

    # For fc_prefix_mode=auto and W==0, we either need per-prompt prefix checks or disable.
    prefix_mode = str(args.fc_prefix_mode).strip().lower()
    if prefix_mode not in {"auto", "always", "never"}:
        raise ValueError(f"Unknown fc_prefix_mode={prefix_mode}")
    if prefix_mode == "auto" and W == 0 and not str(fc_answer_prefix).strip():
        args.fc_prefix_mode = "never"

    tasks_eff = [t for t in tasks if len(EP.candidate_strings(t)) > 0]
    if not tasks_eff:
        raise RuntimeError("No tasks with multiple-choice candidates in --tasks.")

    # module refs
    attn_mod, attn_ref = _find_layer_module(model, int(args.layer), "attn")
    mlp_mod, mlp_ref = _find_layer_module(model, int(args.layer), "mlp")

    step_stats_by_task_seed: Dict[str, Dict[int, Dict[str, Dict[str, Dict[str, float]]]]] = {t: {} for t in tasks_eff}
    window_by_task_seed: Dict[str, Dict[int, Dict[str, Dict[str, Dict[str, float]]]]] = {t: {} for t in tasks_eff}

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
            examples = eval_by_all.get(task, [])
            if not examples:
                continue

            warmup_ids = None
            prompts = [ex.prompt for ex in examples]
            if int(args.fc_warmup_tokens) > 0:
                warmup_ids = EP.precompute_fc_warmup_tokens(
                    model,
                    tok,
                    prompts,
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

            # Build prefix flags (mostly all True when W>0).
            if str(args.fc_prefix_mode).strip().lower() == "never" or not str(fc_answer_prefix).strip():
                do_prefix = False
            elif str(args.fc_prefix_mode).strip().lower() == "always":
                do_prefix = True
            else:
                if W > 0:
                    do_prefix = True
                else:
                    # Rare mixed-case fallback, keep conservative behaviour:
                    do_prefix = bool(any(not EP.prompt_endswith_prefix(p, fc_answer_prefix) for p in prompts))

            attn_col = DecodeStepCollector(
                "attn_delta",
                q_shared=Q_shared,
                step_count=int(total_steps),
                max_state_samples=max(0, int(args.align_samples_per_step)),
                random_seed=EP.stable_int_seed(args.seed, tseed, task, "attn", total_steps),
            )
            mlp_col = DecodeStepCollector(
                "mlp_delta",
                q_shared=Q_shared,
                step_count=int(total_steps),
                max_state_samples=max(0, int(args.align_samples_per_step)),
                random_seed=EP.stable_int_seed(args.seed, tseed, task, "mlp", total_steps),
            )

            h_attn = attn_mod.register_forward_hook(attn_col.make_hook())
            h_mlp = mlp_mod.register_forward_hook(mlp_col.make_hook())

            try:
                _collect_decode_steps(
                    model,
                    tok,
                    prompts,
                    warmup_ids,
                    attn_collector=attn_col,
                    mlp_collector=mlp_col,
                    fc_answer_prefix=fc_answer_prefix,
                    max_prompt_len=int(args.max_prompt_len),
                    warmup_tokens=int(args.fc_warmup_tokens),
                    batch_size=int(args.batch_size),
                    do_prefix=bool(do_prefix),
                    answer_prefix=str(fc_answer_prefix),
                )
            finally:
                try:
                    h_attn.remove()
                except Exception:
                    pass
                try:
                    h_mlp.remove()
                except Exception:
                    pass

            # Step-level summaries (for windowable steps only)
            step_summary: Dict[str, Dict[str, Dict[str, float]]] = {
                "attn_delta": {},
                "mlp_delta": {},
            }

            for step_idx in range(windowable_steps):
                for comp, collector in (("attn_delta", attn_col), ("mlp_delta", mlp_col)):
                    acc = collector.accumulators.get(step_idx)
                    if acc is None:
                        s = {
                            "n": 0,
                            "energy_mean": float("nan"),
                            "energy_std": float("nan"),
                        }
                    else:
                        align = _align_summary(acc.sample_matrix(), Q_shared, k_basis=min(int(args.k_align), k_basis))
                        s = {
                            "n": int(acc.n),
                            "energy_mean": float(acc.mean_ratio()),
                            "energy_std": float(acc.std_ratio()),
                            "align_max_cos": float(align["max_cos"]),
                            "align_mean_cos": float(align["mean_cos"]),
                            "align_min_cos": float(align["min_cos"]),
                            "align_fro_norm": float(align["fro_norm"]),
                        }
                    step_summary[comp][str(step_idx)] = s

            # Window summaries (late/mid)
            win_summary: Dict[str, Dict[str, Dict[str, float]]] = {"attn_delta": {}, "mlp_delta": {}}
            for comp in ("attn_delta", "mlp_delta"):
                comp_steps = step_summary[comp]
                for wname, (w0, w1) in windows.items():
                    comp_steps_int = {int(k): v for k, v in [(kk, vv) for kk, vv in comp_steps.items()]}
                    win = _aggregate_window(comp_steps_int, s0=int(w0), s1=int(w1))
                    win_summary[comp][wname] = win

            step_stats_by_task_seed[task][int(tseed)] = {
                "attn_delta": step_summary["attn_delta"],
                "mlp_delta": step_summary["mlp_delta"],
            }
            window_by_task_seed[task][int(tseed)] = win_summary

            if str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()

    # Aggregate across template seeds
    task_window_macro: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for comp in ("attn_delta", "mlp_delta"):
        task_window_macro[comp] = {}
        for wname in windows.keys():
            for metric in ["energy_mean", "energy_std", "align_max_cos", "align_mean_cos", "align_fro_norm"]:
                vals = []
                for t in tasks_eff:
                    for seed in template_seeds:
                        v = window_by_task_seed.get(t, {}).get(int(seed), {}).get(comp, {}).get(wname, {}).get(metric, float("nan"))
                        vals.append(v)
                m, s = _safe_mean_std([v for v in vals if not (isinstance(v, float) and math.isnan(v))])
                task_window_macro[comp][f"{wname}.{metric}"] = {"macro_mean": m, "macro_std": s, "n": len(vals)}

    # Compare late vs mid if both exist
    diff_late_mid: Dict[str, Dict[str, float]] = {}
    if all(k in windows for k in (f"late{int(args.window_n)}", f"mid{int(args.window_n)}")):
        for comp in ("attn_delta", "mlp_delta"):
            late_energy = []
            mid_energy = []
            late_align = []
            mid_align = []
            for t in tasks_eff:
                for seed in template_seeds:
                    win = window_by_task_seed.get(t, {}).get(int(seed), {}).get(comp, {})
                    le = win.get(f"late{int(args.window_n)}", {}).get("energy_mean")
                    lm = win.get(f"mid{int(args.window_n)}", {}).get("energy_mean")
                    la = win.get(f"late{int(args.window_n)}", {}).get("align_mean_cos")
                    ma = win.get(f"mid{int(args.window_n)}", {}).get("align_mean_cos")
                    if isinstance(le, (float, int)) and not (isinstance(le, float) and math.isnan(float(le))):
                        late_energy.append(float(le))
                    if isinstance(lm, (float, int)) and not (isinstance(lm, float) and math.isnan(float(lm))):
                        mid_energy.append(float(lm))
                    if isinstance(la, (float, int)) and not (isinstance(la, float) and math.isnan(float(la))):
                        late_align.append(float(la))
                    if isinstance(ma, (float, int)) and not (isinstance(ma, float) and math.isnan(float(ma))):
                        mid_align.append(float(ma))
            le_m, le_s = _safe_mean_std(late_energy)
            lm_m, lm_s = _safe_mean_std(mid_energy)
            la_m, la_s = _safe_mean_std(late_align)
            ma_m, ma_s = _safe_mean_std(mid_align)
            diff_late_mid[comp] = {
                "late_energy_mean": le_m,
                "mid_energy_mean": lm_m,
                "late_minus_mid_energy_mean": le_m - lm_m,
                "late_minus_mid_energy_std": math.sqrt(max(0.0, le_s * le_s + lm_s * lm_s)),
                "late_align_mean_cos": la_m,
                "mid_align_mean_cos": ma_m,
                "late_minus_mid_align_mean_cos": la_m - ma_m,
                "late_minus_mid_align_std": math.sqrt(max(0.0, la_s * la_s + ma_s * ma_s)),
            }

    # Build final output
    out = {
        "config": {
            "model": str(args.model),
            "device": str(args.device),
            "dtype": str(args.dtype),
            "trust_remote_code": bool(args.trust_remote_code),
            "layer": int(args.layer),
            "k_basis": int(k_basis),
            "basis_npz": str(args.basis_npz),
            "basis_key": str(args.basis_key),
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
            "windows": {k: [int(v[0]), int(v[1])] for k, v in windows.items()},
            "align_samples_per_step": int(args.align_samples_per_step),
            "k_align": int(args.k_align),
            "collect_steps": {
                "total_steps": int(total_steps),
                "windowable_steps": int(windowable_steps),
                "exclude_final_step": bool(args.exclude_final_step),
            },
        },
        "dataset_meta": meta_by if "meta_by" in locals() else {},
        "step_stats_by_task_seed": step_stats_by_task_seed,
        "window_stats_by_task_seed": window_by_task_seed,
        "window_macro_stats": task_window_macro,
        "late_vs_mid": diff_late_mid,
    }

    base = f"exp_6_delta_shared_stepwise_late{int(args.window_n)}" + (f"_mid{int(args.window_n)}" if bool(args.add_mid_window) else "")
    out_json = os.path.join(out_dir, f"{base}_layer{int(args.layer)}{tag}.json")
    _atomic_json_dump(out, out_json)

    # Markdown summary
    md = []
    md.append("# Exp-6 (M6): Step-wise energy/alignment diagnostic")
    md.append("")
    md.append("Goal: quantify where `Q_shared` aligns with branch deltas over decode steps,")
    md.append("without introducing extra causal intervention conditions.")
    md.append("")
    md.append("## Decode-step indexing")
    md.append(f"- total_steps = 1 + W + P = {total_steps}")
    md.append(f"- windowable_steps = total_steps - 1 (exclude final) = {windowable_steps}")
    md.append(f"- W = {W}, P = {P}, steps are [0, windowable_steps)")
    md.append("")
    md.append("## Windows")
    for wname, (w0, w1) in windows.items():
        md.append(f"- `{wname}`: [{int(w0)},{int(w1)})")

    # Macro summary table: late/mid by component
    rows = []
    for comp in ("attn_delta", "mlp_delta"):
        for metric in ["energy_mean", "energy_std", "align_max_cos", "align_mean_cos", "align_fro_norm"]:
            if metric == "energy_mean":
                label = "Late energy"
            rows.append([comp, metric, _fmt(task_window_macro.get(comp, {}).get(f"late{int(args.window_n)}.{metric}", {}).get("macro_mean", float("nan"))), _fmt(task_window_macro.get(comp, {}).get(f"mid{int(args.window_n)}.{metric}", {}).get("macro_mean", float("nan")))])
    md.append("")
    md.append("## Window-level macro (late vs mid)")
    md.append(_md_table(rows, ["Component", "Metric", f"Late({args.window_n})", f"Mid({args.window_n})"]))

    if diff_late_mid:
        md.append("")
        md.append("## Late - mid contrast")
        rows = []
        for comp, d in diff_late_mid.items():
            rows.append([
                comp,
                _fmt(d.get("late_minus_mid_energy_mean", float("nan"))),
                _fmt(d.get("late_minus_mid_align_mean_cos", float("nan"))),
            ])
        md.append(_md_table(rows, ["Component", "Energy (late-mid)", "AlignMeanCos (late-mid)"]))

    md.append("")
    md.append(f"JSON: `{os.path.relpath(out_json, ROOT_DIR)}`")
    out_md = os.path.join(out_dir, f"{base}_layer{int(args.layer)}{tag}.md")
    _atomic_text_dump("\n".join(md).rstrip() + "\n", out_md)

    print(f"[Saved] {out_json}")
    print(f"[Saved] {out_md}")


if __name__ == "__main__":
    main()
