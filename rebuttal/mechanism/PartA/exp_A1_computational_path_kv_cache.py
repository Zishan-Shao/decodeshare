# -*- coding: utf-8 -*-
"""
exp_A1_computational_path_kv_cache.py

Part A / Mechanism (computational path):
Show that *decode-only* interventions (hooks that only fire when seq_len==1) are
NOT measurable under a naive "prefill forward" protocol, but DO take effect under
the KV-cached decode path (decode-aligned prompt boundary).

This is the protocol-level mechanism:
  - With past_key_values, Transformers expects you to pass only the *new* token(s).
  - Many decode-time interventions are naturally defined on seq_len==1 states.
  - Therefore, evaluating them with a seq_len>1 forward (prefill) can be off-policy.

What the script reports
-----------------------
For each prompt:
  (1) Baseline logits equivalence:
      logits(prefill_full_prompt) ≈ logits(decode_aligned_boundary)
  (2) Intervention visibility:
      With a decode-only removal hook registered, prefill logits don't change,
      but decode-aligned logits do change (hook fires at seq_len==1).

Typical run
-----------
# Recommended: use an already-estimated decode basis (from exp_1) to make it paper-aligned.
CUDA_VISIBLE_DEVICES=0 python rebuttal/mechanism/PartA/exp_A1_computational_path_kv_cache.py \\
  --model meta-llama/Llama-2-7b-chat-hf --device cuda --dtype fp16 \\
  --layer 10 --alpha 1.0 \\
  --basis_npz results/rebuttal_mechanism/logit_lens_l10/basis_layer10_tseed1234.npz \\
  --tasks commonsenseqa,arc_challenge,openbookqa,qasc,logiqa \\
  --n_prompts 32 --template_seed 1234 --shuffle_choices 1 \\
  --out_dir results/rebuttal_mechanism/partA_comp_path
"""

from __future__ import annotations

import os
import sys
import json
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


def _random_orthonormal(D: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    A = rng.standard_normal((int(D), int(k)), dtype=np.float32)
    Q, _ = np.linalg.qr(A)
    return Q.astype(np.float32, copy=False)


def _get_unembedding_weight(model) -> torch.Tensor:
    if hasattr(model, "lm_head") and getattr(model.lm_head, "weight", None) is not None:
        W = model.lm_head.weight
    else:
        emb = model.get_output_embeddings()
        if emb is None or getattr(emb, "weight", None) is None:
            raise RuntimeError("Model has no output embeddings weight.")
        W = emb.weight
    return W


def _decode_token(tok, tid: int) -> str:
    try:
        return tok.decode([int(tid)], clean_up_tokenization_spaces=False)
    except Exception:
        try:
            return tok.decode([int(tid)])
        except Exception:
            return ""


def _topk_delta_tokens(tok, delta_logits_1d: torch.Tensor, topk: int, *, exclude_special: bool = True) -> List[Dict[str, Any]]:
    v = delta_logits_1d.detach().float()
    if v.ndim != 1:
        raise ValueError("delta_logits_1d must be 1D")

    if exclude_special:
        for sid in getattr(tok, "all_special_ids", []) or []:
            if 0 <= int(sid) < v.numel():
                v[int(sid)] = 0.0

    idx = torch.topk(v.abs(), k=min(int(topk), int(v.numel())), largest=True).indices
    out: List[Dict[str, Any]] = []
    for tid in idx.detach().cpu().tolist():
        tid = int(tid)
        out.append(
            {
                "id": tid,
                "delta": float(delta_logits_1d[tid].detach().float().cpu().item()),
                "decoded": _decode_token(tok, tid),
                "decoded_repr": repr(_decode_token(tok, tid)),
            }
        )
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

    # Data (prompts)
    ap.add_argument("--tasks", type=str, default="commonsenseqa,arc_challenge,openbookqa,qasc,logiqa")
    ap.add_argument("--n_prompts", type=int, default=32, help="Prompts per task to test the computational-path effect.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--template_seed", type=int, default=1234)
    ap.add_argument("--template_randomization", type=int, default=1, choices=[0, 1])
    ap.add_argument("--shuffle_choices", type=int, default=1, choices=[0, 1])
    ap.add_argument("--add_answer_prefix", type=int, default=1, choices=[0, 1])
    ap.add_argument("--answer_prefix", type=str, default="\nFinal answer:")

    # Protocol / hook
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=1.0, help="Removal strength: x <- x - alpha * Proj_Q(x).")
    ap.add_argument("--basis_npz", type=str, default="", help="Optional .npz with key 'Q' (saved by exp_1).")
    ap.add_argument("--basis_k", type=int, default=32, help="Used only if --basis_npz is not provided (random basis).")

    # Tokenization / batching
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_prompt_len", type=int, default=512)

    # Reporting
    ap.add_argument("--topk_tokens", type=int, default=25)
    ap.add_argument("--show_examples", type=int, default=6, help="How many prompts to include token-delta examples for.")
    ap.add_argument("--exclude_special", type=int, default=1, choices=[0, 1])

    # Output
    ap.add_argument("--out_dir", type=str, default="results/rebuttal_mechanism/partA_comp_path")
    ap.add_argument("--tag", type=str, default="")

    args = ap.parse_args()

    # Avoid a long stack trace when CUDA isn't actually usable.
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested (--device cuda) but torch.cuda.is_available()==False.\n"
            "This usually means your NVIDIA driver is missing/too old for your installed torch build.\n"
            f"torch={torch.__version__}  torch.version.cuda={getattr(torch.version, 'cuda', None)}"
        )

    tasks = _split_csv(args.tasks)
    if not tasks:
        raise ValueError("Empty --tasks")

    out_dir = os.path.expanduser(str(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)
    tag = ("_" + str(args.tag).strip()) if str(args.tag).strip() else ""

    model, tok = EP.load_model_and_tokenizer(
        args.model,
        args.device,
        args.dtype,
        trust_remote_code=bool(args.trust_remote_code),
    )

    # Basis Q
    basis_info: Dict[str, Any] = {}
    if str(args.basis_npz).strip():
        p = os.path.expanduser(str(args.basis_npz))
        arr = np.load(p)
        if "Q" not in arr:
            raise ValueError(f"--basis_npz missing key 'Q': {p}")
        Q = np.asarray(arr["Q"], dtype=np.float32)
        basis_info = {"mode": "npz", "path": p, "k": int(Q.shape[1]), "d": int(Q.shape[0])}
    else:
        hidden_dim = int(_get_unembedding_weight(model).shape[1])
        Q = _random_orthonormal(hidden_dim, int(args.basis_k), seed=int(args.seed))
        basis_info = {"mode": "random", "k": int(Q.shape[1]), "d": int(Q.shape[0]), "seed": int(args.seed)}

    # Prompts to test
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
    prompts: List[str] = []
    for t in tasks:
        for ex in sub_by.get(t, []):
            prompts.append(ex.prompt)
    if not prompts:
        raise RuntimeError("No prompts loaded. Check tasks / dataloader.")

    # Register decode-only hook (seq_len==1)
    handles, _hooks, stats, toggle = EP.register_hooks(
        model,
        layer_indices=[int(args.layer)],
        basis_np=Q,
        alpha=float(args.alpha),
        name="A1_decode_only_removal",
    )

    device = next(model.parameters()).device
    model.eval()

    def _counts() -> Tuple[int, int]:
        rep = stats.report()
        return int(rep.get("decode_calls", 0)), int(rep.get("intervened", 0))

    records: List[Dict[str, Any]] = []
    shown = 0
    try:
        # Critical for avoiding OOM: inference only (no autograd graph), and avoid
        # materializing unused KV caches during prefill.
        with torch.inference_mode():
            for i in range(0, len(prompts), int(args.batch_size)):
                batch = prompts[i : i + int(args.batch_size)]
                inputs = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=int(args.max_prompt_len)).to(device)
                ids = inputs["input_ids"]
                attn = inputs["attention_mask"]
                B = int(ids.shape[0])

                # (A) Baseline (hook disabled)
                toggle(False)
                dc0, iv0 = _counts()

                # Prefill logits: no need to keep KV cache here (saves a lot of memory).
                out_prefill = model(input_ids=ids, attention_mask=attn, use_cache=False)
                logits_prefill = out_prefill.logits[:, -1, :]
                dc1, iv1 = _counts()
                del out_prefill

                past, logits_decode_aligned = EP.cache_decode_aligned_boundary(model, ids, attn)
                dc2, iv2 = _counts()
                del past

                # (B) Intervention (hook enabled)
                toggle(True)
                out_prefill_i = model(input_ids=ids, attention_mask=attn, use_cache=False)
                logits_prefill_i = out_prefill_i.logits[:, -1, :]
                dc3, iv3 = _counts()
                del out_prefill_i

                past_i, logits_decode_aligned_i = EP.cache_decode_aligned_boundary(model, ids, attn)
                dc4, iv4 = _counts()
                del past_i

                # Per-example metrics
                base_equiv = (logits_prefill - logits_decode_aligned).detach().float()
                prefill_effect = (logits_prefill_i - logits_prefill).detach().float()
                decode_effect = (logits_decode_aligned_i - logits_decode_aligned).detach().float()

                base_equiv_max = torch.amax(base_equiv.abs(), dim=1).detach().cpu().numpy()
                prefill_effect_max = torch.amax(prefill_effect.abs(), dim=1).detach().cpu().numpy()
                decode_effect_max = torch.amax(decode_effect.abs(), dim=1).detach().cpu().numpy()
                decode_effect_l2 = torch.linalg.norm(decode_effect, dim=1).detach().cpu().numpy()

                for b in range(B):
                    rec: Dict[str, Any] = {
                        "prompt_idx": int(i + b),
                        "base_logits_equiv_maxabs": float(base_equiv_max[b]),
                        "prefill_effect_maxabs": float(prefill_effect_max[b]),
                        "decode_aligned_effect_maxabs": float(decode_effect_max[b]),
                        "decode_aligned_effect_l2": float(decode_effect_l2[b]),
                    }
                    if shown < int(args.show_examples):
                        rec["top_delta_tokens_decode_aligned"] = _topk_delta_tokens(
                            tok,
                            decode_effect[b],
                            topk=int(args.topk_tokens),
                            exclude_special=bool(args.exclude_special),
                        )
                        shown += 1
                    records.append(rec)

                # Store hook call counts per stage (batch-level; same for all examples in batch)
                records.append(
                    {
                        "batch_marker": True,
                        "batch_start": int(i),
                        "batch_size": int(B),
                        "hook_counts": {
                            "prefill_baseline": {"decode_calls": int(dc1 - dc0), "intervened": int(iv1 - iv0)},
                            "decode_aligned_baseline": {"decode_calls": int(dc2 - dc1), "intervened": int(iv2 - iv1)},
                            "prefill_intervene": {"decode_calls": int(dc3 - dc2), "intervened": int(iv3 - iv2)},
                            "decode_aligned_intervene": {"decode_calls": int(dc4 - dc3), "intervened": int(iv4 - iv3)},
                        },
                    }
                )

    finally:
        EP.remove_hooks(handles)

    # Aggregate stats
    vals_equiv = np.array([r["base_logits_equiv_maxabs"] for r in records if "base_logits_equiv_maxabs" in r], dtype=np.float64)
    vals_pref = np.array([r["prefill_effect_maxabs"] for r in records if "prefill_effect_maxabs" in r], dtype=np.float64)
    vals_dec = np.array([r["decode_aligned_effect_maxabs"] for r in records if "decode_aligned_effect_maxabs" in r], dtype=np.float64)

    summary = {
        "n_prompts_total": int(len(prompts)),
        "base_logits_equiv_maxabs": {"mean": float(np.mean(vals_equiv)), "p50": float(np.percentile(vals_equiv, 50)), "p95": float(np.percentile(vals_equiv, 95)), "max": float(np.max(vals_equiv))},
        "prefill_effect_maxabs": {"mean": float(np.mean(vals_pref)), "p50": float(np.percentile(vals_pref, 50)), "p95": float(np.percentile(vals_pref, 95)), "max": float(np.max(vals_pref))},
        "decode_aligned_effect_maxabs": {"mean": float(np.mean(vals_dec)), "p50": float(np.percentile(vals_dec, 50)), "p95": float(np.percentile(vals_dec, 95)), "max": float(np.max(vals_dec))},
    }

    out = {
        "config": {
            "model": args.model,
            "device": args.device,
            "dtype": args.dtype,
            "trust_remote_code": bool(args.trust_remote_code),
            "layer": int(args.layer),
            "alpha": float(args.alpha),
            "tasks": tasks,
            "n_prompts": int(args.n_prompts),
            "seed": int(args.seed),
            "template_seed": int(args.template_seed),
            "template_randomization": bool(args.template_randomization),
            "shuffle_choices": bool(args.shuffle_choices),
            "add_answer_prefix": bool(args.add_answer_prefix),
            "answer_prefix": str(args.answer_prefix),
            "batch_size": int(args.batch_size),
            "max_prompt_len": int(args.max_prompt_len),
        },
        "basis": basis_info,
        "dataset_meta": meta_by,
        "summary": summary,
        "records": records,
    }

    out_json = os.path.join(out_dir, f"exp_A1_comp_path_layer{int(args.layer)}{tag}.json")
    _atomic_json_dump(out, out_json)

    md_lines = []
    md_lines.append("# Exp-A1: Computational path (prefill vs KV-cached decode boundary)")
    md_lines.append("")
    md_lines.append("Key claim: decode-only (seq_len==1) interventions are invisible under a prefill forward, but visible under decode-aligned boundary caching.")
    md_lines.append("")
    md_lines.append("## Summary")
    md_lines.append("```json")
    md_lines.append(json.dumps(summary, ensure_ascii=False, indent=2))
    md_lines.append("```")
    md_lines.append("")
    md_lines.append("## What to look for")
    md_lines.append("- `base_logits_equiv_maxabs` should be ~0 (prefill logits == decode-aligned logits).")
    md_lines.append("- `prefill_effect_maxabs` should be ~0 (decode-only hook never fires on seq_len>1).")
    md_lines.append("- `decode_aligned_effect_maxabs` should be >0 (hook fires at seq_len==1 boundary).")
    md_lines.append("")
    md_lines.append(f"JSON: `{os.path.relpath(out_json, ROOT_DIR)}`")
    md_path = os.path.join(out_dir, f"exp_A1_comp_path_layer{int(args.layer)}{tag}.md")
    _atomic_text_dump("\n".join(md_lines).rstrip() + "\n", md_path)

    print(f"[Saved] {out_json}")
    print(f"[Saved] {md_path}")


if __name__ == "__main__":
    main()
