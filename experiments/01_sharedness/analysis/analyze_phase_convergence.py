# -*- coding: utf-8 -*-
"""
analyze_phase_convergence.py

Upgraded EXP2:
- still computes subspace convergence overlap curves for 3 modes:
    (a) decode-last
    (b) prefill-last
    (c) decode-step-t
- additionally computes "ALL-TASK sharedness" in each estimated subspace:
    shared_count_all: number of basis dimensions shared by ALL tasks (m_shared = T_all)
    shared_ratio_all: shared_count_all / cross_dim
  using the SAME criterion as existence: relvar >= tau across tasks.

Key point:
- "prefill has a stable subspace" does NOT imply "prefill has our desired all-task shared subspace".
  This script produces a direct curve showing shared_ratio_all vs #tasks.

Outputs:
- out_csv_long: per repeat rows (mode, n_tasks, rep, overlap, cross_dim, shared_count_all, shared_ratio_all)
- out_csv_summary: aggregated mean/std per (mode,n_tasks)
- out_png: 2-panel figure (left: overlap, right: shared_ratio_all)
- out_png_single_*: optional separate figures if you pass --out_png_overlap / --out_png_shared

CUDA_VISIBLE_DEVICES=0 python analysis/analyze_phase_convergence.py \
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
  --balance_to min \
  --pca_var 0.95 \
  --tau 0.001 \
  --repeats 20 \
  --seed 123 \
  --decode_step_t 8 \
  --out_csv results/exp2.75/llama_exp2.75_distinguish_long.csv \
  --out_png results/exp2.75/llama_exp2.75_distinguish.png

"""

from __future__ import annotations
import os
import sys
import csv
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from decodeshare import sharedness as base


# -------------------------
# Prompt loader selection
# -------------------------
def load_prompts(loader: str, n_prompts: int, seed: int) -> Dict[str, List[str]]:
    loader = (loader or "fair").strip().lower()
    if loader == "fair":
        return base.load_calib_prompts(n_prompts, seed)
    if loader == "full":
        try:
            exp_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if exp_dir not in sys.path:
                sys.path.insert(0, exp_dir)
            import run_full_benchmark as full
            if hasattr(full, "load_calib_prompts_full"):
                return full.load_calib_prompts_full(n_prompts, seed)
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
    storage[task][layer] -> list of [B, D]
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
                return output

            x = hs[:, -1, :]  # [B, D]
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

        collector.set_capture(True)
        _ = model(**inputs, use_cache=False)
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

            attention_mask = torch.cat(
                [attention_mask, torch.ones((B, 1), device=device, dtype=attention_mask.dtype)],
                dim=1,
            )

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
# Subspace / sharedness utils
# -------------------------
def _orthonormalize(Q: np.ndarray) -> np.ndarray:
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
    Q = _orthonormalize(joint_subspace.astype(np.float32, copy=False))
    return Q, int(cross_dim)

def subspace_overlap(Qa: Optional[np.ndarray], Qb: Optional[np.ndarray]) -> float:
    if Qa is None or Qb is None:
        return 0.0
    M = Qa.T @ Qb
    s = np.linalg.svd(M, compute_uv=False)
    k = min(Qa.shape[1], Qb.shape[1])
    if k <= 0:
        return 0.0
    return float((s[:k] ** 2).sum() / float(k))

def all_task_sharedness(
    X_by_task: Dict[str, np.ndarray],
    Q: np.ndarray,
    tasks_all: List[str],
    tau: float,
) -> Tuple[int, float]:
    """
    Evaluate 'all-task shared' dims in basis Q:
      relvar(task, dim) >= tau for ALL tasks in tasks_all.
    Returns: (shared_count, shared_ratio=shared_count/cross_dim)
    """
    relvar_by_task = {t: base.compute_relvar_in_basis(X_by_task[t], Q) for t in tasks_all}
    m_shared = len(tasks_all)  # ALL tasks
    shared_idx = base.compute_shared_indices_from_relvar(relvar_by_task, tau=float(tau), m_shared=int(m_shared))
    sc = int(len(shared_idx))
    k = int(Q.shape[1])
    sr = float(sc) / float(k) if k > 0 else 0.0
    return sc, sr

def fair_preprocess(
    X_raw: Dict[str, np.ndarray],
    per_task_max_states: int,
    balance_to: str,
    seed: int,
) -> Dict[str, np.ndarray]:
    X_bal, _ = base.center_and_balance(
        X_raw,
        per_task_max_states=int(per_task_max_states),
        balance_to=str(balance_to),
        seed=int(seed),
    )
    return X_bal


# -------------------------
# Run one mode
# -------------------------
@dataclass
class ModeLongRow:
    mode: str
    n_tasks: int
    rep: int
    overlap: float
    cross_dim: int
    shared_count_all: int
    shared_ratio_all: float

def collect_acts_for_mode(
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
    seed: int,
    decode_step_t: int,
) -> Dict[str, np.ndarray]:
    tasks_all = list(prompts_by_task.keys())

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

    X_by_task = fair_preprocess(
        X_raw,
        per_task_max_states=int(per_task_max_states),
        balance_to=str(balance_to),
        seed=int(seed) + 999 + (hash(mode) % 1000),
    )
    return X_by_task


def summarize_long_rows(rows: List[ModeLongRow]) -> List[dict]:
    # group by (mode, n_tasks)
    groups: Dict[Tuple[str, int], List[ModeLongRow]] = {}
    for r in rows:
        groups.setdefault((r.mode, r.n_tasks), []).append(r)

    out = []
    for (mode, n), rs in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        overlaps = np.array([x.overlap for x in rs], dtype=np.float64)
        ks = np.array([x.cross_dim for x in rs], dtype=np.float64)
        scs = np.array([x.shared_count_all for x in rs], dtype=np.float64)
        srs = np.array([x.shared_ratio_all for x in rs], dtype=np.float64)

        out.append({
            "mode": mode,
            "n_tasks": n,
            "overlap_mean": float(overlaps.mean()),
            "overlap_std": float(overlaps.std(ddof=0)),
            "cross_dim_mean": float(ks.mean()),
            "cross_dim_std": float(ks.std(ddof=0)),
            "shared_count_all_mean": float(scs.mean()),
            "shared_count_all_std": float(scs.std(ddof=0)),
            "shared_ratio_all_mean": float(srs.mean()),
            "shared_ratio_all_std": float(srs.std(ddof=0)),
        })
    return out


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
    ap.add_argument("--tau", type=float, default=0.001)
    ap.add_argument("--min_dim", type=int, default=1)
    ap.add_argument("--max_dim", type=int, default=4096)

    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--decode_step_t", type=int, default=8)

    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--out_png", type=str, required=True)
    ap.add_argument("--out_png_overlap", type=str, default="")
    ap.add_argument("--out_png_shared", type=str, default="")

    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_png)), exist_ok=True)
    if args.out_png_overlap:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_png_overlap)), exist_ok=True)
    if args.out_png_shared:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_png_shared)), exist_ok=True)

    base.set_global_seed(int(args.seed))

    print(f"[Env] model={args.model} device={args.device} dtype={args.model_dtype} layer={args.layer}")
    model, tok = base.load_model_and_tokenizer(args.model, args.device, args.model_dtype)
    layers, _ = base.get_model_layers(model)
    if int(args.layer) >= len(layers):
        raise RuntimeError(f"layer={args.layer} out of range, num_layers={len(layers)}")

    prompts_by_task = load_prompts(args.loader, int(args.n_prompts), int(args.seed))
    tasks_all = list(prompts_by_task.keys())
    T = len(tasks_all)
    if T < 2:
        raise RuntimeError("Need at least 2 tasks.")
    print(f"[Data] tasks={tasks_all} (T={T})")
    for t in tasks_all:
        print(f"[Data] task={t} loaded_prompts={len(prompts_by_task[t])}")

    modes = ["decode-last", "prefill-last", "decode-step-t"]

    # collect once per mode (expensive); then run many PCA + metrics (cheap)
    X_mode: Dict[str, Dict[str, np.ndarray]] = {}
    for mode in modes:
        print("\n" + "=" * 80)
        print(f"[Collect Mode] {mode}")
        print("=" * 80)
        X_by_task = collect_acts_for_mode(
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
            seed=int(args.seed),
            decode_step_t=int(args.decode_step_t),
        )
        X_mode[mode] = X_by_task
        # quick stats
        n0 = min(v.shape[0] for v in X_by_task.values())
        d = next(iter(X_by_task.values())).shape[1]
        print(f"[{mode}] balanced_states_per_task={n0}, dim={d}")

    long_rows: List[ModeLongRow] = []

    for mode in modes:
        print("\n" + "=" * 80)
        print(f"[Analyze Mode] {mode}")
        print("=" * 80)

        X_by_task = X_mode[mode]
        rng = np.random.default_rng(int(args.seed) + (hash(mode) % 10000))

        # full basis (all tasks)
        Q_full, k_full = compute_basis(X_by_task, int(args.layer), tasks_all, args.pca_var, args.min_dim, args.max_dim)
        if Q_full is None or k_full <= 0:
            raise RuntimeError(f"[{mode}] Failed to compute full basis.")
        sc_full, sr_full = all_task_sharedness(X_by_task, Q_full, tasks_all, tau=float(args.tau))
        print(f"[{mode}] FULL: cross_dim={k_full}, shared_count_all={sc_full}, shared_ratio_all={sr_full:.6f}")

        # subsets
        for n in range(2, T + 1):
            for rep in range(int(args.repeats)):
                subset = rng.choice(tasks_all, size=n, replace=False).tolist()

                Qn, kn = compute_basis(X_by_task, int(args.layer), subset, args.pca_var, args.min_dim, args.max_dim)
                if Qn is None or kn <= 0:
                    ov = 0.0
                    sc = 0
                    sr = 0.0
                else:
                    ov = subspace_overlap(Qn, Q_full)
                    # IMPORTANT: evaluate sharedness across ALL tasks, not just subset
                    sc, sr = all_task_sharedness(X_by_task, Qn, tasks_all, tau=float(args.tau))

                long_rows.append(ModeLongRow(
                    mode=mode, n_tasks=n, rep=rep,
                    overlap=float(ov),
                    cross_dim=int(kn),
                    shared_count_all=int(sc),
                    shared_ratio_all=float(sr),
                ))

            # quick print per n
            rs = [r for r in long_rows if r.mode == mode and r.n_tasks == n]
            srm = float(np.mean([r.shared_ratio_all for r in rs]))
            ovm = float(np.mean([r.overlap for r in rs]))
            print(f"[{mode}] n={n}: overlap_mean={ovm:.4f}, shared_ratio_all_mean={srm:.6f}")

    # write long CSV
    out_csv_long = args.out_csv
    with open(out_csv_long, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mode","n_tasks","rep","overlap","cross_dim","shared_count_all","shared_ratio_all"])
        for r in long_rows:
            w.writerow([r.mode, r.n_tasks, r.rep, r.overlap, r.cross_dim, r.shared_count_all, r.shared_ratio_all])
    print(f"[Save] long CSV: {out_csv_long}")

    # write summary CSV
    summary = summarize_long_rows(long_rows)
    out_csv_summary = out_csv_long.replace(".csv", ".summary.csv")
    with open(out_csv_summary, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        for row in summary:
            w.writerow(row)
    print(f"[Save] summary CSV: {out_csv_summary}")

    # plot (2 panels)
    def _get_series(mode: str, key_mean: str, key_std: str):
        rs = [r for r in summary if r["mode"] == mode]
        xs = [r["n_tasks"] for r in rs]
        ys = [r[key_mean] for r in rs]
        yerr = [r[key_std] for r in rs]
        return xs, ys, yerr

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # left: overlap
    ax = axes[0]
    for mode in modes:
        xs, ys, yerr = _get_series(mode, "overlap_mean", "overlap_std")
        ax.errorbar(xs, ys, yerr=yerr, marker="o", label=mode)
    ax.set_xlabel("#tasks used to estimate subspace")
    ax.set_ylabel("overlap to full-task subspace")
    ax.set_title(f"Subspace convergence (layer={args.layer}, pca_var={args.pca_var})")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True)
    ax.legend()

    # right: all-task shared ratio
    ax = axes[1]
    for mode in modes:
        xs, ys, yerr = _get_series(mode, "shared_ratio_all_mean", "shared_ratio_all_std")
        ax.errorbar(xs, ys, yerr=yerr, marker="o", label=mode)
    ax.set_xlabel("#tasks used to estimate subspace")
    ax.set_ylabel("shared_ratio_all = shared_count_all / cross_dim")
    ax.set_title(f"ALL-task sharedness (tau={args.tau}, m_shared=ALL)")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.savefig(args.out_png, dpi=200)
    print(f"[Save] figure: {args.out_png}")

    # optional single plots
    if args.out_png_overlap:
        plt.figure()
        for mode in modes:
            xs, ys, yerr = _get_series(mode, "overlap_mean", "overlap_std")
            plt.errorbar(xs, ys, yerr=yerr, marker="o", label=mode)
        plt.xlabel("#tasks used to estimate subspace")
        plt.ylabel("overlap to full-task subspace")
        plt.title(f"Subspace convergence (layer={args.layer}, pca_var={args.pca_var})")
        plt.ylim(0.0, 1.05)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.out_png_overlap, dpi=200)
        print(f"[Save] overlap figure: {args.out_png_overlap}")

    if args.out_png_shared:
        plt.figure()
        for mode in modes:
            xs, ys, yerr = _get_series(mode, "shared_ratio_all_mean", "shared_ratio_all_std")
            plt.errorbar(xs, ys, yerr=yerr, marker="o", label=mode)
        plt.xlabel("#tasks used to estimate subspace")
        plt.ylabel("shared_ratio_all = shared_count_all / cross_dim")
        plt.title(f"ALL-task sharedness (tau={args.tau}, m_shared=ALL)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.out_png_shared, dpi=200)
        print(f"[Save] sharedness figure: {args.out_png_shared}")


if __name__ == "__main__":
    main()
