# -*- coding: utf-8 -*-
"""
exp_A2_geometric_subspace_misalignment.py

Part A / Mechanism (geometry):
Quantify the geometric mismatch between *prefill* vs *decode* hidden-state subspaces.

High-level claim:
  The activation distribution at KV-cached decode time (seq_len==1, on-policy generation)
  lives in a subspace that is substantially misaligned with the prefill distribution.
  This makes "off-policy" (prefill-estimated) subspaces/vectors unreliable at decode time.

This script computes, per layer:
  - principal angles between Prefill-PCA(k) and Decode-PCA(k)
  - cross-distribution explained-variance ratios:
      EV(decode data | decode PCs) vs EV(decode data | prefill PCs)
      EV(prefill data | prefill PCs) vs EV(prefill data | decode PCs)

Typical run
-----------
CUDA_VISIBLE_DEVICES=0 python rebuttal/mechanism/PartA/exp_A2_geometric_subspace_misalignment.py \\
  --model meta-llama/Llama-2-7b-chat-hf --device cuda --dtype fp16 \\
  --layers 10,28 \\
  --tasks gsm8k,commonsenseqa,strategyqa,aqua,arc_challenge,openbookqa,qasc,logiqa,boolq \\
  --n_prompts 128 --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \\
  --ks 32,64,128 \\
  --out_dir results/rebuttal_mechanism/partA_geometry
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
# Small utilities
# -----------------------------------------------------------------------------
def _split_csv(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def _parse_int_csv(s: str) -> List[int]:
    out: List[int] = []
    for x in _split_csv(s):
        try:
            out.append(int(x))
        except Exception:
            raise ValueError(f"Bad int in csv: {x!r}")
    return out


def _dedup_keep_order(xs: Sequence[Any]) -> List[Any]:
    seen = set()
    out = []
    for x in xs:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _subsample_rows_np(X: np.ndarray, n_max: int, seed: int) -> np.ndarray:
    if n_max is None or int(n_max) <= 0 or X.shape[0] <= int(n_max):
        return X
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(X.shape[0], size=int(n_max), replace=False)
    return X[idx]


def _center_rows(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.mean(X, axis=0, keepdims=True).astype(np.float32, copy=False)
    return (X.astype(np.float32, copy=False) - mu), mu.squeeze(0)


def _explained_var_ratio(Xc: np.ndarray, Q: np.ndarray) -> float:
    eps = 1e-12
    Xc = Xc.astype(np.float32, copy=False)
    Q = Q.astype(np.float32, copy=False)
    proj = Xc @ Q
    num = float(np.sum(proj * proj))
    den = float(np.sum(Xc * Xc) + eps)
    return num / den


def _pca_basis_lowrank(Xc: np.ndarray, k: int, *, seed: int, device: torch.device) -> np.ndarray:
    Xc = Xc.astype(np.float32, copy=False)
    n, d = Xc.shape
    q = int(min(k, d, max(1, n - 1)))
    if q <= 0:
        raise ValueError(f"Invalid PCA q={q} for shape n={n}, d={d}")
    torch.manual_seed(int(seed))
    Xt = torch.tensor(Xc, dtype=torch.float32, device=device)
    # center=False because Xc is already centered
    _U, _S, V = torch.pca_lowrank(Xt, q=q, center=False, niter=2)
    Q = V[:, :q].detach().cpu().numpy().astype(np.float32, copy=False)
    return Q


def _principal_angles_deg(Qa: np.ndarray, Qb: np.ndarray) -> Dict[str, float]:
    M = Qa.T @ Qb
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    ang = np.degrees(np.arccos(s))
    return {
        "mean": float(np.mean(ang)) if ang.size else float("nan"),
        "p50": float(np.percentile(ang, 50)) if ang.size else float("nan"),
        "p95": float(np.percentile(ang, 95)) if ang.size else float("nan"),
        "min": float(np.min(ang)) if ang.size else float("nan"),
        "max": float(np.max(ang)) if ang.size else float("nan"),
    }


def _md_table(rows: List[List[str]], header: List[str]) -> str:
    out = []
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "---|" * len(header))
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


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

    # Layers and tasks
    ap.add_argument("--layers", type=str, default="10", help="Comma-separated layer indices.")
    ap.add_argument("--tasks", type=str, default="gsm8k,commonsenseqa,strategyqa,aqua,arc_challenge,openbookqa,qasc,logiqa,boolq")
    ap.add_argument("--n_prompts", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--template_seed", type=int, default=1234)
    ap.add_argument("--template_randomization", type=int, default=1, choices=[0, 1])
    ap.add_argument("--shuffle_choices", type=int, default=1, choices=[0, 1])
    ap.add_argument("--add_answer_prefix", type=int, default=1, choices=[0, 1])
    ap.add_argument("--answer_prefix", type=str, default="\nFinal answer:")

    # Decode collection params
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_prompt_len", type=int, default=512)
    ap.add_argument("--calib_decode_max_new_tokens", type=int, default=128)
    ap.add_argument("--decoding", type=str, default="greedy", choices=["greedy", "sample"])
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--per_task_max_states", type=int, default=20000)

    # PCA / metrics
    ap.add_argument("--ks", type=str, default="32,64,128", help="Comma-separated k values to report.")
    ap.add_argument("--pca_max_rows", type=int, default=80000, help="Max pooled rows used for PCA per phase (0=no limit).")
    ap.add_argument("--pca_device", type=str, default="cpu", choices=["cpu", "cuda"])

    # Output
    ap.add_argument("--out_dir", type=str, default="results/rebuttal_mechanism/partA_geometry")
    ap.add_argument("--tag", type=str, default="")

    args = ap.parse_args()

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested (--device cuda) but torch.cuda.is_available()==False.\n"
            "This usually means your NVIDIA driver is missing/too old for your installed torch build.\n"
            f"torch={torch.__version__}  torch.version.cuda={getattr(torch.version, 'cuda', None)}"
        )
    if str(args.pca_device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--pca_device cuda requested but CUDA is not available.")

    tasks = _split_csv(args.tasks)
    if not tasks:
        raise ValueError("Empty --tasks")

    layers = _dedup_keep_order(_parse_int_csv(args.layers))
    if not layers:
        raise ValueError("Empty --layers")

    ks = _dedup_keep_order(_parse_int_csv(args.ks))
    ks = [int(k) for k in ks if int(k) > 0]
    if not ks:
        raise ValueError("Empty --ks")
    k_max_req = int(max(ks))

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

    # Load prompts (subspace prompts only)
    sub_by, _eval_by_dummy, meta_by = load_selected_tasks(
        tasks=tasks,
        n_subspace=max(1, int(args.n_prompts)),
        n_eval=1,  # (compat) loader may not accept 0
        seed=int(args.seed),
        template_randomization=bool(args.template_randomization),
        template_seed=int(args.template_seed),
        shuffle_choices=bool(args.shuffle_choices),
        add_answer_prefix=bool(args.add_answer_prefix),
        answer_prefix=str(args.answer_prefix),
    )

    # Collect prefill states for all requested layers
    layers_mods, _ = EP.get_model_layers(model)
    pre_col = EP.PrefillLastTokenCollector([int(li) for li in layers])
    pre_handles = [layers_mods[int(li)].register_forward_hook(pre_col.make_hook(int(li))) for li in layers]
    try:
        for task, sub_exs in sub_by.items():
            pre_col.set_current_task(task)
            prompts = [ex.prompt for ex in sub_exs]
            EP.collect_prefill_states(
                model,
                tok,
                prompts,
                pre_col,
                batch_size=int(args.batch_size),
                max_prompt_len=int(args.max_prompt_len),
            )
    finally:
        for h in pre_handles:
            try:
                h.remove()
            except Exception:
                pass

    # Collect decode states for all requested layers
    dec_col = EP.DecodeLastTokenCollector([int(li) for li in layers])
    dec_handles = [layers_mods[int(li)].register_forward_hook(dec_col.make_hook(int(li))) for li in layers]
    try:
        for task, sub_exs in sub_by.items():
            dec_col.set_current_task(task)
            prompts = [ex.prompt for ex in sub_exs]
            EP.collect_decode_states(
                model,
                tok,
                prompts,
                dec_col,
                batch_size=int(args.batch_size),
                max_new_tokens=int(args.calib_decode_max_new_tokens),
                max_prompt_len=int(args.max_prompt_len),
                decoding=str(args.decoding),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                top_k=int(args.top_k),
            )
    finally:
        for h in dec_handles:
            try:
                h.remove()
            except Exception:
                pass
        dec_col.set_capture(False, None)

    pca_device = torch.device(str(args.pca_device))

    results: Dict[str, Any] = {
        "config": {
            "model": args.model,
            "device": args.device,
            "dtype": args.dtype,
            "trust_remote_code": bool(args.trust_remote_code),
            "layers": layers,
            "tasks": tasks,
            "n_prompts": int(args.n_prompts),
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
            },
            "per_task_max_states": int(args.per_task_max_states),
            "pca": {"ks": ks, "pca_max_rows": int(args.pca_max_rows), "pca_device": str(args.pca_device)},
        },
        "dataset_meta": meta_by,
        "by_layer": {},
    }

    md_rows: List[List[str]] = []

    for li in layers:
        # Build per-task matrices for each phase at this layer
        mats_pre: Dict[str, np.ndarray] = {}
        mats_dec: Dict[str, np.ndarray] = {}
        for task in tasks:
            Xp = pre_col.get(task, int(li))
            Xd = dec_col.get(task, int(li))
            if Xp is not None and Xp.shape[0] > 0:
                Xp = _subsample_rows_np(Xp, int(args.per_task_max_states), seed=EP.stable_int_seed(args.seed, "prefill", task, li))
                mats_pre[task] = Xp
            if Xd is not None and Xd.shape[0] > 0:
                Xd = _subsample_rows_np(Xd, int(args.per_task_max_states), seed=EP.stable_int_seed(args.seed, "decode", task, li))
                mats_dec[task] = Xd

        if not mats_pre or not mats_dec:
            raise RuntimeError(f"Missing states at layer {li}: prefill_tasks={len(mats_pre)} decode_tasks={len(mats_dec)}")

        mats_pre_bal, n_pre_min = EP.balance_task_states(mats_pre, seed=EP.stable_int_seed(args.seed, "bal_pre", li))
        mats_dec_bal, n_dec_min = EP.balance_task_states(mats_dec, seed=EP.stable_int_seed(args.seed, "bal_dec", li))
        Xp = np.concatenate([mats_pre_bal[t] for t in mats_pre_bal.keys()], axis=0)
        Xd = np.concatenate([mats_dec_bal[t] for t in mats_dec_bal.keys()], axis=0)

        # Optional pooled subsampling for PCA
        Xp = _subsample_rows_np(Xp, int(args.pca_max_rows), seed=EP.stable_int_seed(args.seed, "pool_pre", li)) if int(args.pca_max_rows) > 0 else Xp
        Xd = _subsample_rows_np(Xd, int(args.pca_max_rows), seed=EP.stable_int_seed(args.seed, "pool_dec", li)) if int(args.pca_max_rows) > 0 else Xd

        Xp_c, _mu_p = _center_rows(Xp)
        Xd_c, _mu_d = _center_rows(Xd)

        n_pre, d_pre = Xp_c.shape
        n_dec, d_dec = Xd_c.shape
        if d_pre != d_dec:
            raise ValueError(f"Hidden dim mismatch at layer {li}: prefill d={d_pre} vs decode d={d_dec}")

        k_max = int(min(k_max_req, d_pre, max(1, n_pre - 1), max(1, n_dec - 1)))
        if k_max <= 0:
            raise RuntimeError(f"Not enough rows for PCA at layer {li}: n_pre={n_pre} n_dec={n_dec} d={d_pre}")

        Qp = _pca_basis_lowrank(Xp_c, k=k_max, seed=EP.stable_int_seed(args.seed, "pca_pre", li), device=pca_device)
        Qd = _pca_basis_lowrank(Xd_c, k=k_max, seed=EP.stable_int_seed(args.seed, "pca_dec", li), device=pca_device)

        layer_res: Dict[str, Any] = {
            "n_rows_prefill_pool": int(n_pre),
            "n_rows_decode_pool": int(n_dec),
            "n_min_per_task_prefill": int(n_pre_min),
            "n_min_per_task_decode": int(n_dec_min),
            "d": int(d_pre),
            "k_max": int(k_max),
            "by_k": {},
        }

        for k in ks:
            if int(k) > k_max:
                continue
            Qa = Qp[:, : int(k)]
            Qb = Qd[:, : int(k)]
            ang = _principal_angles_deg(Qa, Qb)

            ev = {
                "decode_by_decode": _explained_var_ratio(Xd_c, Qb),
                "decode_by_prefill": _explained_var_ratio(Xd_c, Qa),
                "prefill_by_prefill": _explained_var_ratio(Xp_c, Qa),
                "prefill_by_decode": _explained_var_ratio(Xp_c, Qb),
            }
            layer_res["by_k"][str(int(k))] = {"angles_deg": ang, "explained_var_ratio": ev}

            # A compact row for the MD summary (use mean principal angle)
            md_rows.append(
                [
                    str(li),
                    str(k),
                    f"{ang['mean']:.2f}",
                    f"{ev['decode_by_prefill']:.3f}",
                    f"{ev['decode_by_decode']:.3f}",
                ]
            )

        results["by_layer"][str(li)] = layer_res

    out_json = os.path.join(out_dir, f"exp_A2_geometry{tag}.json")
    _atomic_json_dump(results, out_json)

    md = []
    md.append("# Exp-A2: Geometric subspace misalignment (prefill vs decode)")
    md.append("")
    md.append("Columns:")
    md.append("- `mean_angle_deg`: mean principal angle between Prefill-PCA(k) and Decode-PCA(k)")
    md.append("- `EV_decode|prefill`: explained-variance ratio of decode data under prefill PCs")
    md.append("- `EV_decode|decode`: explained-variance ratio of decode data under decode PCs (upper bound)")
    md.append("")
    md.append(_md_table(md_rows, ["layer", "k", "mean_angle_deg", "EV_decode|prefill", "EV_decode|decode"]))
    md.append("")
    md.append(f"JSON: `{os.path.relpath(out_json, ROOT_DIR)}`")
    md_path = os.path.join(out_dir, f"exp_A2_geometry{tag}.md")
    _atomic_text_dump("\n".join(md).rstrip() + "\n", md_path)

    print(f"[Saved] {out_json}")
    print(f"[Saved] {md_path}")


if __name__ == "__main__":
    main()
