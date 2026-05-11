# -*- coding: utf-8 -*-
"""
exp_A3_causal_decode_only_controls.py

Part A / Mechanism (causal test + matched controls):
Run a decode-only causal intervention (remove a decode-shared subspace) and compare
against strong matched controls:
  - structural control: remove a nonshared subspace of the same dimension (joint_nonshared_topk)
  - energy-matched control: choose k_c for nonshared such that removed energy matches shared
  - random controls: random_struct (dim-matched) and random_energy (energy-matched)

This is designed to be "reviewer-hard":
  - decode-only (seq_len==1) hook
  - forced-choice logprob evaluation (decode-aligned boundary)
  - controls that match dimension and/or removed energy

Typical run
-----------
CUDA_VISIBLE_DEVICES=0 python rebuttal/mechanism/PartA/exp_A3_causal_decode_only_controls.py \\
  --model meta-llama/Llama-2-7b-chat-hf --device cuda --dtype fp16 \\
  --layer 10 \\
  --tasks_subspace gsm8k,commonsenseqa,strategyqa,aqua,arc_challenge,openbookqa,qasc,logiqa,boolq \\
  --tasks_eval commonsenseqa,arc_challenge,openbookqa,qasc,logiqa,boolq \\
  --n_prompts 128 --eval_n 256 \\
  --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \\
  --pca_var 0.95 --min_dim 8 --max_dim 256 --tau 0.001 --m_shared all \\
  --k_eval 128 --alpha_remove 1.0 \\
  --fc_prefix_mode auto --fc_answer_prefix $'\\nFinal answer:' \\
  --bootstrap_iters 5000 --perm_iters 10000 --seed 42 \\
  --out_dir results/rebuttal_mechanism/partA_causal
"""

from __future__ import annotations

import os
import sys
import json
import math
import argparse
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    return os.path.normpath(os.path.join(start_dir, "..", "..", ".."))


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


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _split_csv(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def _fmt_acc(acc: float, lo: float, hi: float) -> str:
    return f"{acc*100:.1f} [{lo*100:.1f}, {hi*100:.1f}]"


def _fmt_diff(stat: Dict[str, Any]) -> str:
    return f"{stat['mean_diff']*100:+.1f} [{stat['ci_low']*100:+.1f}, {stat['ci_high']*100:+.1f}] (p={stat['p_value']:.3g})"


def _md_table(rows: List[List[str]], header: List[str]) -> str:
    out = []
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "---|" * len(header))
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def _energy_ratio_mean(states: np.ndarray, Q: np.ndarray) -> float:
    eps = 1e-12
    H = states.astype(np.float32, copy=False)
    Q = Q.astype(np.float32, copy=False)
    proj = H @ Q
    num = np.sum(proj * proj, axis=1)
    den = np.sum(H * H, axis=1) + eps
    return float(np.mean(num / den))


def _make_random_orthonormal(D: int, k: int, seed: int, orthogonal_to: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Random orthonormal basis in R^D with optional orthogonalization to an existing basis.
    """
    rng = np.random.default_rng(int(seed))
    D = int(D)
    k = int(k)
    if k <= 0:
        raise ValueError("k must be positive")

    ortho = None
    if orthogonal_to is not None:
        ortho = np.asarray(orthogonal_to, dtype=np.float32)
        if ortho.ndim != 2 or ortho.shape[0] != D:
            raise ValueError(f"orthogonal_to must have shape [D, r], got {ortho.shape}")

    for _ in range(20):
        A = rng.standard_normal((D, k), dtype=np.float32)
        if ortho is not None and ortho.shape[1] > 0:
            # Remove components along ortho
            A = A - ortho @ (ortho.T @ A)
        Q, R = np.linalg.qr(A)
        diag = np.abs(np.diag(R))
        if diag.size == 0 or float(np.min(diag)) < 1e-8:
            continue
        return Q.astype(np.float32, copy=False)
    raise RuntimeError("Failed to sample a full-rank random basis after many tries.")


def _max_overlap(Qa: np.ndarray, Qb: np.ndarray) -> float:
    M = Qa.T @ Qb
    return float(np.max(np.abs(M))) if M.size else float("nan")


def _choose_energy_matched_k(
    *,
    states: np.ndarray,
    Q_ordered: np.ndarray,
    target_er: float,
) -> Tuple[int, List[float]]:
    """
    Given an ordered orthonormal basis Q_ordered [D, Kmax], pick k that matches target_er
    in terms of mean energy ratio on `states`.
    Returns (k_best, ers_by_k) where ers_by_k[k-1] = ER(first k cols).
    """
    eps = 1e-12
    H = states.astype(np.float32, copy=False)
    Q = Q_ordered.astype(np.float32, copy=False)
    den = np.sum(H * H, axis=1) + eps  # [N]

    # Project once: [N, Kmax]
    Y = H @ Q
    Y2 = Y * Y
    cum = np.cumsum(Y2, axis=1)  # [N, Kmax]
    ers = np.mean(cum / den[:, None], axis=0)  # [Kmax]
    # Pick k with minimal abs diff
    diffs = np.abs(ers - float(target_er))
    k_best = int(np.argmin(diffs) + 1)
    return k_best, [float(x) for x in ers.tolist()]


@torch.no_grad()
def _collect_decode_states_by_task(
    *,
    model,
    tok,
    sub_by: Dict[str, List[Any]],
    layer_idx: int,
    batch_size: int,
    max_prompt_len: int,
    calib_decode_max_new_tokens: int,
    decoding: str,
    temperature: float,
    top_p: float,
    top_k: int,
    per_task_max_states: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    layers_mods, _ = EP.get_model_layers(model)
    if int(layer_idx) < 0 or int(layer_idx) >= len(layers_mods):
        raise ValueError(f"layer_idx={layer_idx} out of range: num_layers={len(layers_mods)}")

    col = EP.DecodeLastTokenCollector([int(layer_idx)])
    handle = layers_mods[int(layer_idx)].register_forward_hook(col.make_hook(int(layer_idx)))
    out: Dict[str, np.ndarray] = {}
    try:
        for task, sub_exs in sub_by.items():
            col.set_current_task(task)
            prompts = [ex.prompt for ex in sub_exs]
            EP.collect_decode_states(
                model,
                tok,
                prompts,
                col,
                batch_size=int(batch_size),
                max_new_tokens=int(calib_decode_max_new_tokens),
                max_prompt_len=int(max_prompt_len),
                decoding=str(decoding),
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
            )
            X = col.get(task, int(layer_idx))
            if X is None or X.shape[0] == 0:
                raise RuntimeError(f"No decode states collected for task={task}")
            X = EP.subsample_rows_np(X, int(per_task_max_states), seed=EP.stable_int_seed(seed, task, "decode"))
            out[task] = X
    finally:
        try:
            handle.remove()
        except Exception:
            pass
        col.set_capture(False, None)
    return out


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
    ap.add_argument("--device_map", type=str, default="", help="Optional HF device_map, e.g. 'auto' for multi-GPU sharding.")
    ap.add_argument("--max_memory_per_gpu_gb", type=float, default=0.0, help="Uniform per-GPU cap used only when --device_map is set.")
    ap.add_argument(
        "--max_memory_map",
        type=str,
        default="",
        help="Optional per-device memory map, e.g. '0:72,1:28,2:20,cpu:220'. Takes precedence over --max_memory_per_gpu_gb.",
    )
    ap.add_argument("--cpu_offload_gb", type=float, default=0.0, help="Optional CPU max_memory used only when --device_map is set.")

    # Basis tasks / eval tasks
    ap.add_argument("--tasks_subspace", type=str, default="gsm8k,commonsenseqa,strategyqa,aqua,arc_challenge,openbookqa,qasc,logiqa,boolq")
    ap.add_argument("--tasks_eval", type=str, default="commonsenseqa,arc_challenge,openbookqa,qasc,logiqa,boolq")
    ap.add_argument("--n_prompts", type=int, default=128)
    ap.add_argument("--eval_n", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--template_seed", type=int, default=1234)
    ap.add_argument("--template_randomization", type=int, default=1, choices=[0, 1])
    ap.add_argument("--shuffle_choices", type=int, default=1, choices=[0, 1])
    ap.add_argument("--add_answer_prefix", type=int, default=1, choices=[0, 1])
    ap.add_argument("--answer_prefix", type=str, default="\nFinal answer:")

    # Layer / decode collection
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_prompt_len", type=int, default=512)
    ap.add_argument("--calib_decode_max_new_tokens", type=int, default=128)
    ap.add_argument("--decoding", type=str, default="greedy", choices=["greedy", "sample"])
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--per_task_max_states", type=int, default=20000)
    ap.add_argument("--energy_calib_max_rows", type=int, default=50000, help="Max pooled rows used for energy matching (0=no limit).")

    # Sharedness / subspace params
    ap.add_argument("--pca_var", type=float, default=0.95)
    ap.add_argument("--min_dim", type=int, default=8)
    ap.add_argument("--max_dim", type=int, default=256)
    ap.add_argument("--tau", type=float, default=0.001)
    ap.add_argument("--m_shared", type=str, default="all")
    ap.add_argument(
        "--auto_tau_if_no_nonshared",
        type=int,
        default=1,
        choices=[0, 1],
        help="If 1 and shared covers all components, auto-increase effective tau to create a nonshared pool for controls.",
    )
    ap.add_argument(
        "--auto_tau_quantile",
        type=float,
        default=0.5,
        help="Quantile over per-component sharedness score used to choose tau when auto-tau is triggered (0.5=median).",
    )

    # Intervention params
    ap.add_argument("--k_eval", type=int, default=128, help="Requested shared-k to remove (will clamp if needed). 0=use all possible.")
    ap.add_argument("--alpha_remove", type=float, default=1.0)

    # Forced-choice settings
    ap.add_argument("--fc_prefix_mode", type=str, default="auto", choices=["auto", "always", "never"])
    ap.add_argument("--fc_answer_prefix", type=str, default="\nFinal answer:")
    ap.add_argument("--fc_warmup_tokens", type=int, default=0)
    ap.add_argument("--fc_warmup_decoding", type=str, default="greedy", choices=["greedy", "sample"])
    ap.add_argument("--fc_warmup_ban_eos", type=int, default=0, choices=[0, 1])
    ap.add_argument("--fc_warmup_seed", type=int, default=123)
    ap.add_argument("--fc_save_scores", type=int, default=0, choices=[0, 1])

    # Stats
    ap.add_argument("--bootstrap_iters", type=int, default=5000)
    ap.add_argument("--perm_iters", type=int, default=10000)
    ap.add_argument("--alpha", type=float, default=0.05, help="CI alpha (e.g., 0.05 => 95%% CI).")

    # Output
    ap.add_argument("--out_dir", type=str, default="results/rebuttal_mechanism/partA_causal")
    ap.add_argument("--tag", type=str, default="")

    args = ap.parse_args()

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested (--device cuda) but torch.cuda.is_available()==False.\n"
            "This usually means your NVIDIA driver is missing/too old for your installed torch build.\n"
            f"torch={torch.__version__}  torch.version.cuda={getattr(torch.version, 'cuda', None)}"
        )

    tasks_sub = _split_csv(args.tasks_subspace)
    tasks_eval = _split_csv(args.tasks_eval)
    if not tasks_sub:
        raise ValueError("Empty --tasks_subspace")
    if not tasks_eval:
        raise ValueError("Empty --tasks_eval")

    out_dir = os.path.expanduser(str(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)
    tag = ("_" + str(args.tag).strip()) if str(args.tag).strip() else ""

    EP.set_global_seed(int(args.seed))
    model, tok = EP.load_model_and_tokenizer(
        args.model,
        args.device,
        args.dtype,
        trust_remote_code=bool(args.trust_remote_code),
        device_map=(str(args.device_map).strip() or None),
        max_memory_per_gpu_gb=float(args.max_memory_per_gpu_gb),
        max_memory_map=str(args.max_memory_map),
        cpu_offload_gb=float(args.cpu_offload_gb),
    )

    # Load prompts + eval examples
    sub_by, eval_by_all, meta_by = load_selected_tasks(
        tasks=_split_csv(args.tasks_subspace),
        n_subspace=max(1, int(args.n_prompts)),
        n_eval=max(1, int(args.eval_n)),
        seed=int(args.seed),
        template_randomization=bool(args.template_randomization),
        template_seed=int(args.template_seed),
        shuffle_choices=bool(args.shuffle_choices),
        add_answer_prefix=bool(args.add_answer_prefix),
        answer_prefix=str(args.answer_prefix),
    )

    # Collect decode states (on-policy, seq_len==1 steps) for sharedness estimation
    decode_task_states = _collect_decode_states_by_task(
        model=model,
        tok=tok,
        sub_by=sub_by,
        layer_idx=int(args.layer),
        batch_size=int(args.batch_size),
        max_prompt_len=int(args.max_prompt_len),
        calib_decode_max_new_tokens=int(args.calib_decode_max_new_tokens),
        decoding=str(args.decoding),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
        per_task_max_states=int(args.per_task_max_states),
        seed=int(args.seed),
    )

    # Compute joint subspace + shared indices
    joint, shared_idx, extra = EP.compute_shared_basis_from_states(
        decode_task_states,
        pca_var=float(args.pca_var),
        min_dim=int(args.min_dim),
        max_dim=int(args.max_dim),
        tau=float(args.tau),
        m_shared=str(args.m_shared),
        seed=int(args.seed) + 100,
    )
    cross_dim = int(extra.get("cross_dim", joint.shape[1]))
    shared_sorted = sorted(int(i) for i in (shared_idx or []))
    if not shared_sorted:
        raise RuntimeError("No shared components found (shared_idx empty). Try smaller tau or m_shared.")

    # Nonshared indices ordered by pooled variance (sum/mean of per-task raw variances)
    contrib = extra.get("task_contributions", {}) or {}
    tasks_used = list(extra.get("tasks_used", [])) or list(contrib.keys())
    tasks_used = [t for t in tasks_used if t in contrib]
    # Compute per-task relative contributions (raw_variances / total_variance) for auto-tau fallback.
    rel_rows = []
    rel_tasks = []
    for t in tasks_used:
        cd = contrib.get(t, {}) or {}
        rv = cd.get("raw_variances", None)
        tv = cd.get("total_variance", None)
        if rv is None:
            continue
        rv_arr = np.asarray(rv, dtype=np.float64)
        if rv_arr.ndim != 1 or rv_arr.shape[0] != cross_dim:
            continue
        if tv is None:
            tv = float(np.sum(rv_arr))
        tv_f = float(tv)
        if not np.isfinite(tv_f) or tv_f <= 0:
            continue
        rel_rows.append(rv_arr / tv_f)
        rel_tasks.append(t)
    rel_mat = np.stack(rel_rows, axis=0) if rel_rows else None  # [T, cross_dim]

    pooled_var = np.zeros(cross_dim, dtype=np.float64)
    n_tasks = 0
    for _t, cd in contrib.items():
        rv = cd.get("raw_variances", None)
        if rv is None:
            continue
        arr = np.asarray(rv, dtype=np.float64)
        if arr.shape[0] != cross_dim:
            continue
        pooled_var += arr
        n_tasks += 1
    if n_tasks > 0:
        pooled_var /= float(n_tasks)
    shared_set = set(shared_sorted)
    nonshared = [i for i in range(cross_dim) if i not in shared_set]
    tau_effective = float(args.tau)
    auto_tau_note = ""
    if not nonshared:
        if not bool(args.auto_tau_if_no_nonshared):
            raise RuntimeError(
                "No nonshared components available (shared covers all cross_dim). "
                "Increase max_dim or adjust tau/m_shared (or set --auto_tau_if_no_nonshared 1)."
            )

        if rel_mat is None or rel_mat.shape[0] == 0:
            raise RuntimeError(
                "Auto-tau was requested but could not compute relative contributions from task_contributions. "
                "Try rerunning with different tau/m_shared."
            )

        # Effective min_tasks_shared (as in find_fully_shared_basis_improved)
        m_raw = str(args.m_shared).strip().lower()
        if m_raw == "all":
            min_tasks_shared = int(rel_mat.shape[0])
        else:
            try:
                min_tasks_shared = int(m_raw)
            except Exception:
                min_tasks_shared = int(rel_mat.shape[0])
        min_tasks_shared = max(1, min(int(min_tasks_shared), int(rel_mat.shape[0])))

        # Sharedness score per component: the m-th largest relative contribution across tasks.
        # Component is "shared" iff score > tau (strict, matching variance > total*tau).
        sort_rel = np.sort(rel_mat, axis=0)  # ascending [T, K]
        kth_idx = int(rel_mat.shape[0] - min_tasks_shared)  # 0-based index into ascending sort
        score = sort_rel[kth_idx, :]  # [K]

        def _try_quantile(q: float) -> Optional[Tuple[float, List[int], List[int]]]:
            q = float(min(max(q, 0.0), 1.0))
            tau_eff = float(np.quantile(score, q))
            shared_eff = [i for i in range(cross_dim) if float(score[i]) > tau_eff]
            if not shared_eff or len(shared_eff) >= cross_dim:
                return None
            nonshared_eff = [i for i in range(cross_dim) if i not in set(shared_eff)]
            if not nonshared_eff:
                return None
            return tau_eff, shared_eff, nonshared_eff

        qs = [float(args.auto_tau_quantile), 0.75, 0.25, 0.9, 0.1, 0.95, 0.05]
        chosen = None
        for q in qs:
            chosen = _try_quantile(q)
            if chosen is not None:
                break

        if chosen is None:
            # Degenerate case (e.g., score nearly constant). Fallback to a deterministic split.
            order = np.argsort(pooled_var)[::-1].tolist()
            k_half = max(1, int(cross_dim // 2))
            shared_eff = order[:k_half]
            nonshared_eff = order[k_half:]
            tau_eff = float(args.tau)
            print(
                f"[Warn] Shared covers all components at tau={float(args.tau):.4g}. "
                "Auto-tau could not create a clean split; falling back to a variance-based half split "
                f"(shared={len(shared_eff)}, nonshared={len(nonshared_eff)})."
            )
            tau_used = None
            auto_tau_note = "fallback_variance_half_split"
        else:
            tau_eff, shared_eff, nonshared_eff = chosen
            tau_used = float(tau_eff)
            tau_effective = float(tau_eff)
            auto_tau_note = "auto_tau_quantile_split"
            print(
                f"[AutoTau] Shared covered all {cross_dim} components at tau={float(args.tau):.4g}. "
                f"Using tau_eff={tau_eff:.4g} (min_tasks_shared={min_tasks_shared}/{rel_mat.shape[0]}) -> "
                f"shared={len(shared_eff)}, nonshared={len(nonshared_eff)}."
            )

        # Order by pooled variance within each pool (more stable / reviewer-friendly)
        shared_sorted = sorted(shared_eff, key=lambda i: float(pooled_var[i]), reverse=True)
        nonshared_sorted = sorted(nonshared_eff, key=lambda i: float(pooled_var[i]), reverse=True)
    else:
        nonshared_sorted = sorted(nonshared, key=lambda i: float(pooled_var[i]), reverse=True)

    # Choose k to remove (clamp so we can compare to nonshared controls)
    k_req = int(args.k_eval)
    if k_req <= 0:
        k_req = len(shared_sorted)
    k_use = min(k_req, len(shared_sorted), len(nonshared_sorted))
    if k_use <= 0:
        raise RuntimeError("k_use<=0 after clamping.")

    Q_shared = EP.orthonormalize_np(joint[:, shared_sorted[:k_use]])
    Q_ctrl_struct = EP.orthonormalize_np(joint[:, nonshared_sorted[:k_use]])

    # Energy matching calibration pool (balanced across tasks)
    decode_bal, _n_min = EP.balance_task_states(decode_task_states, seed=EP.stable_int_seed(args.seed, "bal_energy"))
    H = np.concatenate([decode_bal[t] for t in decode_bal.keys()], axis=0)
    if int(args.energy_calib_max_rows) > 0:
        H = EP.subsample_rows_np(H, int(args.energy_calib_max_rows), seed=EP.stable_int_seed(args.seed, "energy_pool"))

    er_shared = _energy_ratio_mean(H, Q_shared)

    # Energy-matched nonshared: choose k_c over prefixes of ordered nonshared basis
    Q_nonshared_ordered = EP.orthonormalize_np(joint[:, nonshared_sorted])  # [D, K_nonshared]
    k_c, ers_by_k = _choose_energy_matched_k(states=H, Q_ordered=Q_nonshared_ordered, target_er=er_shared)
    Q_ctrl_energy = EP.orthonormalize_np(Q_nonshared_ordered[:, :k_c])

    # Random controls (optionally orthogonal to Q_shared)
    D = int(Q_shared.shape[0])
    Q_rand_struct = _make_random_orthonormal(D, k_use, seed=int(args.seed) + 1234, orthogonal_to=Q_shared)
    Q_rand_energy = _make_random_orthonormal(D, k_c, seed=int(args.seed) + 5678, orthogonal_to=Q_shared)

    diagnostics = {
        "cross_dim": int(cross_dim),
        "k_shared_total": int(len(shared_sorted)),
        "tau_requested": float(args.tau),
        "tau_effective": float(tau_effective),
        "auto_tau_note": str(auto_tau_note),
        "m_shared": str(args.m_shared),
        "auto_tau_if_no_nonshared": bool(args.auto_tau_if_no_nonshared),
        "auto_tau_quantile": float(args.auto_tau_quantile),
        "k_eval_used": int(k_use),
        "k_energy_matched": int(k_c),
        "energy_ratio_shared": float(er_shared),
        "energy_ratio_ctrl_struct": float(_energy_ratio_mean(H, Q_ctrl_struct)),
        "energy_ratio_ctrl_energy": float(_energy_ratio_mean(H, Q_ctrl_energy)),
        "energy_ratio_rand_struct": float(_energy_ratio_mean(H, Q_rand_struct)),
        "energy_ratio_rand_energy": float(_energy_ratio_mean(H, Q_rand_energy)),
        "max_overlap": {
            "shared_vs_ctrl_struct": _max_overlap(Q_shared, Q_ctrl_struct),
            "shared_vs_ctrl_energy": _max_overlap(Q_shared, Q_ctrl_energy),
            "shared_vs_rand_struct": _max_overlap(Q_shared, Q_rand_struct),
            "shared_vs_rand_energy": _max_overlap(Q_shared, Q_rand_energy),
        },
    }

    # Save bases for reuse
    basis_npz = os.path.join(out_dir, f"exp_A3_bases_layer{int(args.layer)}{tag}.npz")
    np.savez(
        basis_npz,
        Q_shared=Q_shared.astype(np.float32),
        Q_ctrl_struct=Q_ctrl_struct.astype(np.float32),
        Q_ctrl_energy=Q_ctrl_energy.astype(np.float32),
        Q_rand_struct=Q_rand_struct.astype(np.float32),
        Q_rand_energy=Q_rand_energy.astype(np.float32),
        shared_idx_used=np.array(shared_sorted[:k_use], dtype=np.int32),
        nonshared_idx_struct=np.array(nonshared_sorted[:k_use], dtype=np.int32),
        nonshared_idx_energy=np.array(nonshared_sorted[:k_c], dtype=np.int32),
    )

    # Evaluate tasks (forced-choice only)
    eval_tasks_effective: List[str] = []
    for t in tasks_eval:
        if t not in eval_by_all:
            continue
        if len(EP.candidate_strings(t)) == 0:
            continue
        eval_tasks_effective.append(t)
    if not eval_tasks_effective:
        raise RuntimeError("No eval tasks with forced-choice candidates. Check --tasks_eval.")

    conditions = [
        ("baseline", None, 0.0),
        ("shared", Q_shared, float(args.alpha_remove)),
        ("ctrl_struct", Q_ctrl_struct, float(args.alpha_remove)),
        ("ctrl_energy", Q_ctrl_energy, float(args.alpha_remove)),
        ("rand_struct", Q_rand_struct, float(args.alpha_remove)),
        ("rand_energy", Q_rand_energy, float(args.alpha_remove)),
    ]

    eval_results: Dict[str, Any] = {}
    for task in eval_tasks_effective:
        examples = eval_by_all[task]
        prompts = [ex.prompt for ex in examples]

        warmup_token_ids = None
        if int(args.fc_warmup_tokens) > 0:
            warmup_token_ids = EP.precompute_fc_warmup_tokens(
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
                seed=int(args.fc_warmup_seed),
            )

        per_task: Dict[str, Any] = {"n": int(len(examples)), "by_condition": {}, "paired_vs_baseline": {}}
        corr_by_cond: Dict[str, np.ndarray] = {}

        for name, Q, alpha in conditions:
            out_fc = EP.forced_choice_logprob_eval(
                model,
                tok,
                examples,
                task,
                layer_indices=[int(args.layer)],
                basis_np=Q,
                alpha=float(alpha),
                batch_size=int(args.batch_size),
                max_prompt_len=int(args.max_prompt_len),
                warmup_token_ids=warmup_token_ids,
                answer_prefix=str(args.fc_answer_prefix),
                prefix_mode=str(args.fc_prefix_mode),
                save_scores=bool(args.fc_save_scores),
            )
            corr = np.asarray(out_fc["correct"], dtype=np.float32)
            acc, lo, hi = EP.bootstrap_ci_mean(corr, iters=int(args.bootstrap_iters), alpha=float(args.alpha), seed=int(args.seed) + 11)

            corr_by_cond[name] = corr
            per_task["by_condition"][name] = {
                "acc": float(out_fc["acc"]),
                "acc_ci": {"mean": float(acc), "lo": float(lo), "hi": float(hi)},
                "hook_stats": out_fc.get("hook_stats", {}),
                "metrics_summary": out_fc.get("metrics_summary", {}),
            }

        if "baseline" not in corr_by_cond:
            raise RuntimeError(f"Missing baseline results for task={task}")
        base_corr = corr_by_cond["baseline"]
        for name, _Q, _alpha in conditions:
            if name == "baseline":
                continue
            treat_corr = corr_by_cond[name]
            per_task["paired_vs_baseline"][name] = EP.summarize_paired(
                base_corr,
                treat_corr,
                label=name,
                bootstrap_iters=int(args.bootstrap_iters),
                perm_iters=int(args.perm_iters),
                alpha=float(args.alpha),
                seed=int(args.seed) + 999,
            )

        eval_results[task] = per_task

    # Render MD summary
    md_rows = []
    header = ["Task", "n", "Baseline", "Shared", "ΔShared", "Ctrl(E)", "ΔCtrl(E)", "Rand(E)", "ΔRand(E)"]
    for task in eval_tasks_effective:
        pt = eval_results[task]
        n = pt["n"]
        b = pt["by_condition"]["baseline"]["acc_ci"]
        sh = pt["by_condition"]["shared"]["acc_ci"]
        ce = pt["by_condition"]["ctrl_energy"]["acc_ci"]
        re = pt["by_condition"]["rand_energy"]["acc_ci"]
        dsh = pt["paired_vs_baseline"]["shared"]
        dce = pt["paired_vs_baseline"]["ctrl_energy"]
        dre = pt["paired_vs_baseline"]["rand_energy"]
        md_rows.append(
            [
                task,
                str(n),
                _fmt_acc(b["mean"], b["lo"], b["hi"]),
                _fmt_acc(sh["mean"], sh["lo"], sh["hi"]),
                _fmt_diff(dsh),
                _fmt_acc(ce["mean"], ce["lo"], ce["hi"]),
                _fmt_diff(dce),
                _fmt_acc(re["mean"], re["lo"], re["hi"]),
                _fmt_diff(dre),
            ]
        )

    out = {
        "config": {
            "model": args.model,
            "device": args.device,
            "dtype": args.dtype,
            "trust_remote_code": bool(args.trust_remote_code),
            "device_map": str(args.device_map),
            "max_memory_per_gpu_gb": float(args.max_memory_per_gpu_gb),
            "max_memory_map": str(args.max_memory_map),
            "cpu_offload_gb": float(args.cpu_offload_gb),
            "layer": int(args.layer),
            "tasks_subspace": tasks_sub,
            "tasks_eval": tasks_eval,
            "n_prompts": int(args.n_prompts),
            "eval_n": int(args.eval_n),
            "seed": int(args.seed),
            "template_seed": int(args.template_seed),
            "template_randomization": bool(args.template_randomization),
            "shuffle_choices": bool(args.shuffle_choices),
            "add_answer_prefix": bool(args.add_answer_prefix),
            "answer_prefix": str(args.answer_prefix),
            "decode_collect": {
                "calib_decode_max_new_tokens": int(args.calib_decode_max_new_tokens),
                "decoding": str(args.decoding),
                "temperature": float(args.temperature),
                "top_p": float(args.top_p),
                "top_k": int(args.top_k),
                "per_task_max_states": int(args.per_task_max_states),
            },
            "sharedness": {
                "pca_var": float(args.pca_var),
                "min_dim": int(args.min_dim),
                "max_dim": int(args.max_dim),
                "tau": float(args.tau),
                "m_shared": str(args.m_shared),
            },
            "intervention": {"k_eval": int(args.k_eval), "alpha_remove": float(args.alpha_remove)},
            "forced_choice": {
                "fc_prefix_mode": str(args.fc_prefix_mode),
                "fc_answer_prefix": str(args.fc_answer_prefix),
                "fc_warmup_tokens": int(args.fc_warmup_tokens),
            },
            "stats": {"bootstrap_iters": int(args.bootstrap_iters), "perm_iters": int(args.perm_iters), "alpha": float(args.alpha)},
        },
        "dataset_meta": meta_by,
        "diagnostics": diagnostics,
        "saved_basis_npz": os.path.relpath(basis_npz, ROOT_DIR),
        "eval_tasks_effective": eval_tasks_effective,
        "eval": eval_results,
    }

    out_json = os.path.join(out_dir, f"exp_A3_causal_layer{int(args.layer)}{tag}.json")
    _atomic_json_dump(out, out_json)

    md = []
    md.append("# Exp-A3: Decode-only causal test with matched controls")
    md.append("")
    md.append("Key comparisons (all decode-only interventions):")
    md.append("- `Shared`: remove decode-shared subspace (k_eval)")
    md.append("- `Ctrl(E)`: remove nonshared prefix with energy matched to Shared")
    md.append("- `Rand(E)`: remove random subspace with energy matched to Shared")
    md.append("")
    md.append("## Diagnostics")
    md.append("```json")
    md.append(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    md.append("```")
    md.append("")
    md.append("## Forced-choice results")
    md.append(_md_table(md_rows, header))
    md.append("")
    md.append(f"JSON: `{os.path.relpath(out_json, ROOT_DIR)}`")
    md.append(f"Bases: `{os.path.relpath(basis_npz, ROOT_DIR)}`")
    md_path = os.path.join(out_dir, f"exp_A3_causal_layer{int(args.layer)}{tag}.md")
    _atomic_text_dump("\n".join(md).rstrip() + "\n", md_path)

    print(f"[Saved] {out_json}")
    print(f"[Saved] {md_path}")


if __name__ == "__main__":
    main()
