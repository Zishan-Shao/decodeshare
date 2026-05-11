# -*- coding: utf-8 -*-
"""
exp2_subspace_convergence_enhanced.py

Compute subspace convergence curves for 3 activation modes in one run:
  (a) decode-last     : decode phase seq_len==1 (like sharedness_base.py)
  (b) prefill-last    : prompt prefill only, take last prompt token hidden state
  (c) decode-step-t   : decode phase, take hidden state at a specific decode step t (0-index)

Outputs:
  - CSV (long-form): mode, n_tasks, overlap_mean, overlap_std, cross_dim_mean, cross_dim_std
  - PNG: a single plot with 3 curves (+ error bars)

Key detail:
  compute_cross_task_subspace may return a basis that is not strictly orthonormal.
  For overlap, we compare subspaces, so we orthonormalize via QR and compute overlap
  using principal angles (SVD of Qa^T Qb). This guarantees overlap in [0,1].


Run example:
CUDA_VISIBLE_DEVICES=0 python exp2.5_subspace_convergence_enhanced.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --device cuda \
  --model_dtype fp32 \
  --layer 10 \
  --loader fair \
  --n_prompts 128 \
  --max_prompt_len 512 \
  --calib_max_new_tokens 128 \
  --batch_size 4 \
  --per_task_max_states 20000 \
  --pca_var 0.95 \
  --repeats 20 \
  --seed 123 \
  --decode_step_t 8 \
  --out_csv results/exp2.5/llama_exp2.5_enhanced_fair.csv \
  --out_png results/exp2.5/llama_exp2.5_enhanced_fair.png
    



如果你要跑你 full benchmark（run_full_benchmark.py 那套 loader），把 --loader full：

--loader full

"""

from __future__ import annotations
import os
import sys
import csv
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

import sharedness_base as base


# -------------------------
# Prompt loader selection
# -------------------------
def load_prompts(loader: str, n_prompts: int, seed: int) -> Dict[str, List[str]]:
    """
    loader:
      - fair: base.load_calib_prompts
      - full: tries to import run_full_benchmark and use its load_calib_prompts_full
    """
    loader = (loader or "fair").strip().lower()
    if loader == "fair":
        return base.load_calib_prompts(n_prompts, seed)

    if loader == "full":
        try:
            import run_full_benchmark as full
            if hasattr(full, "load_calib_prompts_full"):
                return full.load_calib_prompts_full(n_prompts, seed)
            # if full monkeypatches base.load_calib_prompts, we can use it too
            if hasattr(full, "base") and hasattr(full.base, "load_calib_prompts"):
                return full.base.load_calib_prompts(n_prompts, seed)
        except Exception as e:
            print(f"[Warn] cannot use full loader: {e}. Falling back to fair.")
        return base.load_calib_prompts(n_prompts, seed)

    raise ValueError(f"Unknown --loader {loader}. Use fair|full.")


# -------------------------
# Collectors
# -------------------------
from collections import defaultdict
from typing import DefaultDict

class PrefillLastTokenCollector:
    """
    Collect prefill-phase last prompt token hidden states.
    storage[task][layer] -> list of [b, D]
    """
    def __init__(self, layer_indices: List[int]):
        self.layer_indices = list(layer_indices)
        self._cur_task: Optional[str] = None
        self.capture_enabled: bool = False
        self.storage: DefaultDict[str, DefaultDict[int, List[np.ndarray]]] = defaultdict(lambda: defaultdict(list))

    def set_current_task(self, task_name: str) -> None:
        self._cur_task = task_name

    def set_capture(self, enabled: bool) -> None:
        self.capture_enabled = bool(enabled)

    def make_hook(self, layer_idx: int):
        def _hook(module, inputs, output):
            if (not self.capture_enabled) or (self._cur_task is None):
                return output
            hs = output[0] if isinstance(output, tuple) else output
            if not isinstance(hs, torch.Tensor) or hs.ndim != 3:
                return output
            if hs.shape[1] <= 1:
                return output  # prefill should have seq_len > 1 typically

            x = hs[:, -1, :]  # [B, D] last prompt token
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


class DecodeStepTCollector:
    """
    Collect decode-phase hidden states ONLY at a specific decode step t (0-index),
    where each decode forward pass has seq_len == 1.
    """
    def __init__(self, layer_indices: List[int], target_step: int):
        self.layer_indices = list(layer_indices)
        self.target_step = int(target_step)
        self._cur_task: Optional[str] = None
        self.capture_enabled: bool = False
        self.active_mask: Optional[torch.Tensor] = None
        self.cur_step: int = -1
        self.storage: DefaultDict[str, DefaultDict[int, List[np.ndarray]]] = defaultdict(lambda: defaultdict(list))

    def set_current_task(self, task_name: str) -> None:
        self._cur_task = task_name

    def set_step(self, step: int) -> None:
        self.cur_step = int(step)

    def set_capture(self, enabled: bool, active_mask: Optional[torch.Tensor] = None) -> None:
        self.capture_enabled = bool(enabled)
        self.active_mask = active_mask

    def make_hook(self, layer_idx: int):
        def _hook(module, inputs, output):
            if (not self.capture_enabled) or (self._cur_task is None):
                return output
            if self.cur_step != self.target_step:
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


# -------------------------
# Collection routines
# -------------------------
@torch.no_grad()
def collect_prefill_last_token_states(
    model,
    tokenizer,
    prompts: List[str],
    collector: PrefillLastTokenCollector,
    *,
    batch_size: int,
    max_prompt_len: int,
) -> None:
    device = next(model.parameters()).device
    model.eval()

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for i in tqdm(range(0, len(prompts), batch_size), desc="CollectPrefill"):
        batch = prompts[i:i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_len,
        ).to(device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        collector.set_capture(True)
        _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        collector.set_capture(False)


@torch.no_grad()
def collect_decode_step_t_states(
    model,
    tokenizer,
    prompts: List[str],
    collector: DecodeStepTCollector,
    *,
    batch_size: int,
    max_prompt_len: int,
    decoding: str,
    temperature: float,
    top_p: float,
    top_k: int,
) -> None:
    """
    Generate until step==target_step and capture only that step. To reach step t,
    we need t+1 decode iterations after prefill logits.
    """
    assert decoding in ["greedy", "sample"]
    device = next(model.parameters()).device
    model.eval()

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos = tokenizer.eos_token_id

    target_t = int(collector.target_step)

    for i in tqdm(range(0, len(prompts), batch_size), desc=f"CollectDecodeStepT(t={target_t})"):
        batch = prompts[i:i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_len,
        ).to(device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        B, _ = input_ids.shape

        unfinished = torch.ones(B, dtype=torch.bool, device=device)

        # Prefill (no capture)
        collector.set_capture(False, None)
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        logits = out.logits[:, -1, :]
        past = out.past_key_values

        # Decode loop until target step
        for step in range(target_t + 1):
            if not bool(unfinished.any().item()):
                break

            if decoding == "greedy":
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                lt = logits / max(float(temperature), 1e-6)
                lt = base.top_k_filtering(lt, top_k=int(top_k))
                lt = base.top_p_filtering(lt, top_p=float(top_p))
                probs = torch.softmax(lt, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            next_token = next_token * unfinished.unsqueeze(-1) + (eos * (~unfinished)).unsqueeze(-1)

            # append to attention mask
            attention_mask = torch.cat(
                [attention_mask, torch.ones((B, 1), device=device, dtype=attention_mask.dtype)],
                dim=1,
            )

            # capture only unfinished at this step
            collector.set_step(step)
            collector.set_capture(True, unfinished)

            out = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                use_cache=True,
                past_key_values=past,
            )
            collector.set_capture(False, None)

            logits = out.logits[:, -1, :]
            past = out.past_key_values

            unfinished = unfinished & (next_token.squeeze(-1) != eos)

        collector.set_capture(False, None)


# -------------------------
# Subspace / overlap utils
# -------------------------
def _orthonormalize(Q: np.ndarray) -> np.ndarray:
    # QR to orthonormal columns
    Q = Q.astype(np.float64, copy=False)
    Q, _ = np.linalg.qr(Q)
    return Q.astype(np.float32, copy=False)

def compute_basis(
    X_by_task: Dict[str, np.ndarray],
    layer: int,
    tasks: List[str],
    pca_var: float,
    min_dim: int,
    max_dim: int,
) -> Tuple[Optional[np.ndarray], int]:
    task_acts = {t: {layer: X_by_task[t]} for t in tasks}
    joint_subspace, cross_dim, _, _ = base.compute_cross_task_subspace(
        task_acts,
        variance_threshold=float(pca_var),
        min_dim=int(min_dim),
        max_dim=int(max_dim),
        return_full_pca=True,
    )
    if joint_subspace is None or int(cross_dim) <= 0:
        return None, 0
    Q = joint_subspace.astype(np.float32, copy=False)
    Q = _orthonormalize(Q)
    return Q, int(cross_dim)

def subspace_overlap(Qa: Optional[np.ndarray], Qb: Optional[np.ndarray]) -> float:
    """
    overlap in [0,1]:
      overlap = sum_i cos^2(theta_i) / k, where cos(theta_i) are singular values of Qa^T Qb
    """
    if Qa is None or Qb is None:
        return 0.0
    M = Qa.T @ Qb
    s = np.linalg.svd(M, compute_uv=False)
    k = min(Qa.shape[1], Qb.shape[1])
    if k <= 0:
        return 0.0
    return float((s[:k] ** 2).sum() / float(k))

def fair_preprocess(
    X_raw: Dict[str, np.ndarray],
    per_task_max_states: int,
    balance_to: str,
    seed: int,
) -> Tuple[Dict[str, np.ndarray], int]:
    return base.center_and_balance(
        X_raw,
        per_task_max_states=int(per_task_max_states),
        balance_to=str(balance_to),
        seed=int(seed),
    )


# -------------------------
# Main pipeline
# -------------------------
@dataclass
class ModeResult:
    mode: str
    rows: List[dict]

def run_one_mode(
    mode: str,
    model,
    tok,
    layers,
    prompts_by_task: Dict[str, List[str]],
    layer_idx: int,
    *,
    batch_size: int,
    max_prompt_len: int,
    calib_max_new_tokens: int,
    decoding: str,
    temperature: float,
    top_p: float,
    top_k: int,
    per_task_max_states: int,
    balance_to: str,
    pca_var: float,
    min_dim: int,
    max_dim: int,
    repeats: int,
    seed: int,
    decode_step_t: int,
) -> ModeResult:
    tasks_all = list(prompts_by_task.keys())
    T = len(tasks_all)
    rng = np.random.default_rng(int(seed) + (hash(mode) % 10000))

    # --- collect X_raw per task ---
    X_raw: Dict[str, np.ndarray] = {}

    handles = []
    if mode == "decode-last":
        collector = base.DecodeLastTokenActivationCollector([layer_idx])
        handles.append(layers[layer_idx].register_forward_hook(collector.make_hook(layer_idx)))
        try:
            with torch.inference_mode():
                for task in tasks_all:
                    collector.set_current_task(task)
                    base.collect_decode_last_token_states(
                        model=model,
                        tokenizer=tok,
                        prompts=prompts_by_task[task],
                        collector=collector,
                        batch_size=int(batch_size),
                        max_prompt_len=int(max_prompt_len),
                        calib_max_new_tokens=int(calib_max_new_tokens),
                        decoding=str(decoding),
                        temperature=float(temperature),
                        top_p=float(top_p),
                        top_k=int(top_k),
                    )
        finally:
            for h in handles:
                try: h.remove()
                except Exception: pass
            collector.set_capture(False, None)

        for task in tasks_all:
            X = collector.get(task, layer_idx)
            if X is None or X.shape[0] == 0:
                raise RuntimeError(f"[{mode}] No activations for task={task}")
            X_raw[task] = X

    elif mode == "prefill-last":
        collector = PrefillLastTokenCollector([layer_idx])
        handles.append(layers[layer_idx].register_forward_hook(collector.make_hook(layer_idx)))
        try:
            with torch.inference_mode():
                for task in tasks_all:
                    collector.set_current_task(task)
                    collect_prefill_last_token_states(
                        model=model,
                        tokenizer=tok,
                        prompts=prompts_by_task[task],
                        collector=collector,
                        batch_size=int(batch_size),
                        max_prompt_len=int(max_prompt_len),
                    )
        finally:
            for h in handles:
                try: h.remove()
                except Exception: pass
            collector.set_capture(False)

        for task in tasks_all:
            X = collector.get(task, layer_idx)
            if X is None or X.shape[0] == 0:
                raise RuntimeError(f"[{mode}] No activations for task={task}")
            X_raw[task] = X

    elif mode == "decode-step-t":
        collector = DecodeStepTCollector([layer_idx], target_step=int(decode_step_t))
        handles.append(layers[layer_idx].register_forward_hook(collector.make_hook(layer_idx)))
        try:
            with torch.inference_mode():
                for task in tasks_all:
                    collector.set_current_task(task)
                    collect_decode_step_t_states(
                        model=model,
                        tokenizer=tok,
                        prompts=prompts_by_task[task],
                        collector=collector,
                        batch_size=int(batch_size),
                        max_prompt_len=int(max_prompt_len),
                        decoding=str(decoding),
                        temperature=float(temperature),
                        top_p=float(top_p),
                        top_k=int(top_k),
                    )
        finally:
            for h in handles:
                try: h.remove()
                except Exception: pass
            collector.set_capture(False, None)

        for task in tasks_all:
            X = collector.get(task, layer_idx)
            if X is None or X.shape[0] == 0:
                raise RuntimeError(f"[{mode}] No activations for task={task}")
            X_raw[task] = X

    else:
        raise ValueError(f"Unknown mode {mode}")

    # --- fair preprocessing (cap, balance, center) ---
    X_by_task, _ = fair_preprocess(
        X_raw,
        per_task_max_states=int(per_task_max_states),
        balance_to=str(balance_to),
        seed=int(seed) + 999,
    )

    # --- full basis ---
    Q_full, k_full = compute_basis(X_by_task, layer_idx, tasks_all, pca_var, min_dim, max_dim)
    if Q_full is None or k_full <= 0:
        raise RuntimeError(f"[{mode}] Failed to compute full basis.")

    # --- convergence curve ---
    rows: List[dict] = []
    for n in range(2, T + 1):
        overlaps = []
        ks = []
        for _r in range(int(repeats)):
            subset = rng.choice(tasks_all, size=n, replace=False).tolist()
            Qn, kn = compute_basis(X_by_task, layer_idx, subset, pca_var, min_dim, max_dim)
            overlaps.append(subspace_overlap(Qn, Q_full))
            ks.append(kn)

        rows.append({
            "mode": mode,
            "n_tasks": n,
            "overlap_mean": float(np.mean(overlaps)),
            "overlap_std": float(np.std(overlaps, ddof=0)),
            "cross_dim_mean": float(np.mean(ks)),
            "cross_dim_std": float(np.std(ks, ddof=0)),
        })

    return ModeResult(mode=mode, rows=rows)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model_dtype", type=str, default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--layer", type=int, default=10)

    ap.add_argument("--loader", type=str, default="fair", choices=["fair", "full"])
    ap.add_argument("--n_prompts", type=int, default=128)
    ap.add_argument("--max_prompt_len", type=int, default=512)
    ap.add_argument("--calib_max_new_tokens", type=int, default=128)

    ap.add_argument("--calib_decoding", type=str, default="greedy", choices=["greedy", "sample"])
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=4)

    ap.add_argument("--per_task_max_states", type=int, default=20000)
    ap.add_argument("--balance_to", type=str, default="min")

    ap.add_argument("--pca_var", type=float, default=0.95)
    ap.add_argument("--min_dim", type=int, default=1)
    ap.add_argument("--max_dim", type=int, default=4096)

    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--decode_step_t", type=int, default=8)

    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--out_png", type=str, required=True)

    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_png)), exist_ok=True)

    base.set_global_seed(int(args.seed))

    print(f"[Env] model={args.model} device={args.device} dtype={args.model_dtype} layer={args.layer}")
    model, tok = base.load_model_and_tokenizer(args.model, args.device, args.model_dtype)
    layers, _ = base.get_model_layers(model)
    if int(args.layer) >= len(layers):
        raise RuntimeError(f"layer={args.layer} out of range, num_layers={len(layers)}")

    prompts_by_task = load_prompts(args.loader, int(args.n_prompts), int(args.seed))
    tasks = list(prompts_by_task.keys())
    print(f"[Data] tasks={tasks} n_prompts(target)={args.n_prompts}")
    for t in tasks:
        print(f"[Data] task={t} loaded_prompts={len(prompts_by_task[t])}")

    modes = ["decode-last", "prefill-last", "decode-step-t"]
    all_rows: List[dict] = []

    for mode in modes:
        print("\n" + "=" * 80)
        print(f"[Run Mode] {mode}")
        print("=" * 80)

        res = run_one_mode(
            mode=mode,
            model=model,
            tok=tok,
            layers=layers,
            prompts_by_task=prompts_by_task,
            layer_idx=int(args.layer),
            batch_size=int(args.batch_size),
            max_prompt_len=int(args.max_prompt_len),
            calib_max_new_tokens=int(args.calib_max_new_tokens),
            decoding=str(args.calib_decoding),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            top_k=int(args.top_k),
            per_task_max_states=int(args.per_task_max_states),
            balance_to=str(args.balance_to),
            pca_var=float(args.pca_var),
            min_dim=int(args.min_dim),
            max_dim=int(args.max_dim),
            repeats=int(args.repeats),
            seed=int(args.seed),
            decode_step_t=int(args.decode_step_t),
        )
        all_rows.extend(res.rows)

        # quick preview
        last = res.rows[-1]
        print(f"[{mode}] n={last['n_tasks']} overlap={last['overlap_mean']:.4f}±{last['overlap_std']:.4f} "
              f"cross_dim={last['cross_dim_mean']:.1f}±{last['cross_dim_std']:.1f}")

    # save CSV
    fieldnames = ["mode", "n_tasks", "overlap_mean", "overlap_std", "cross_dim_mean", "cross_dim_std"]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f"[Save] {args.out_csv}")

    # plot: 3 curves on one figure
    plt.figure()
    for mode in modes:
        rs = [r for r in all_rows if r["mode"] == mode]
        xs = [r["n_tasks"] for r in rs]
        ys = [r["overlap_mean"] for r in rs]
        yerr = [r["overlap_std"] for r in rs]
        plt.errorbar(xs, ys, yerr=yerr, marker="o", label=mode)

    plt.xlabel("#tasks used to estimate subspace")
    plt.ylabel("overlap to full-task subspace")
    title = f"Subspace convergence (layer={args.layer}, pca_var={args.pca_var})"
    if "decode-step-t" in modes:
        title += f", decode_step_t={args.decode_step_t}"
    plt.title(title)
    plt.ylim(0.0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=200)
    print(f"[Save] {args.out_png}")


if __name__ == "__main__":
    main()
