# -*- coding: utf-8 -*-
"""
exp_1_logit_lens_vocab_signature.py

Mechanism experiment M1 (high ROI): Logit-lens / vocabulary signature for the decode-shared subspace.

Core idea
---------
Given a decode-shared basis Q = [q_1 ... q_k] (hidden_dim x k), project each direction into
vocabulary-logit space via the unembedding / lm_head matrix W_U (vocab_size x hidden_dim):

  s_i = W_U q_i        (vocab_size,)

Then inspect the top-k tokens by |s_i| (and pos/neg separately), and tag them by simple
format categories (option letters, brackets, answer markers, whitespace/newlines, punctuation, ...).

Optionally compute an aggregate "signature" across directions:

  s_agg[t] = || (W_U Q)[t, :] ||_2

and measure stability across prompt template seeds (same examples, different templates).

Typical run
-----------
CUDA_VISIBLE_DEVICES=7 python rebuttal/mechanism/exp_1_logit_lens_vocab_signature.py \\
  --model meta-llama/Llama-2-7b-chat-hf --device cuda --dtype fp16 \\
  --layer 10 \\
  --tasks gsm8k,commonsenseqa,strategyqa,aqua,arc_challenge,openbookqa,qasc,logiqa,boolq \\
  --n_prompts 128 --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \\
  --pca_var 0.95 --min_dim 8 --max_dim 256 --tau 0.001 --m_shared all \\
  --k_analyze 32 --topk 40 \\
  --template_seed 1234 --template_randomization 1 --shuffle_choices 1 \\
  --add_answer_prefix 1 --answer_prefix $'\\nFinal answer:' \\
  --out_dir results/rebuttal_mechanism/logit_lens_l10_seed1234
"""

from __future__ import annotations

import os
import sys
import re
import json
import math
import time
import string
import argparse
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


# -----------------------------------------------------------------------------
# Repo-local imports (this script lives in rebuttal/mechanism/)
# -----------------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(THIS_DIR, "..", ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
REASONING_DIR = os.path.join(ROOT_DIR, "reasoning")

for p in [SRC_DIR, REASONING_DIR, ROOT_DIR]:
    if p not in sys.path:
        sys.path.append(p)

try:
    import eval_perf as EP  # reasoning/eval_perf.py
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Failed to import `reasoning/eval_perf.py` as module `eval_perf`.\n"
        "Make sure you run inside the repo and that `reasoning/` is present."
    ) from e

try:
    from benchmark_dataloaders import load_selected_tasks  # src/benchmark_dataloaders.py
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Failed to import `src/benchmark_dataloaders.py` as module `benchmark_dataloaders`.\n"
        "Make sure `src/` is present and on PYTHONPATH."
    ) from e


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------
def _atomic_json_dump(obj: Any, out_path: str) -> None:
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
# Parsing
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


# -----------------------------------------------------------------------------
# Token tagging (reviewer-friendly categories)
# -----------------------------------------------------------------------------
_PUNCT_ASCII = set(string.punctuation)
_PUNCT_CJK = set("，。！？；：、…·“”‘’—–")
_BRACKETS = set("()[]{}<>（）【】《》")
_OPTION_LETTERS = set("ABCDE")
_YES_NO = {"yes", "no"}


def _safe_convert_id_to_token(tok, tid: int) -> str:
    try:
        s = tok.convert_ids_to_tokens([int(tid)])[0]
        return str(s)
    except Exception:
        try:
            s = tok.convert_ids_to_tokens(int(tid))
            return str(s)
        except Exception:
            return f"<id:{int(tid)}>"


def _safe_decode_one(tok, tid: int) -> str:
    try:
        return tok.decode([int(tid)], clean_up_tokenization_spaces=False)
    except Exception:
        try:
            return tok.decode([int(tid)])
        except Exception:
            return ""


def _tags_for_token(raw_tok: str, decoded: str) -> List[str]:
    s = decoded
    s_strip = s.strip()
    tags: List[str] = []

    if s == "":
        tags.append("empty")

    if "\n" in s:
        tags.append("newline")

    # SentencePiece (▁...) or GPT2/BPE (Ġ...) often marks leading whitespace
    if s.startswith(" ") or raw_tok.startswith("▁") or raw_tok.startswith("Ġ"):
        tags.append("leading_space")

    if s_strip == "":
        tags.append("whitespace")

    if any(ch in _BRACKETS for ch in s_strip):
        tags.append("bracket")

    if s_strip in _OPTION_LETTERS:
        tags.append("option_letter")
    if re.fullmatch(r"[A-E][\)\.\:]", s_strip):
        tags.append("option_punct")

    low = s_strip.lower()
    if low in _YES_NO:
        tags.append("yes_no")

    if any(ch.isdigit() for ch in s_strip):
        tags.append("digit")

    # "Answer:" / "Final answer:" markers
    if re.search(r"(?i)\bfinal\s*answer\b", s) or re.search(r"(?i)\banswer\s*:", s):
        tags.append("answer_marker")

    # Basic reasoning markers (keep conservative)
    if re.search(r"(?i)\btherefore\b|\bthus\b|\bhence\b|\bbecause\b", s_strip):
        tags.append("reasoning_marker")

    if s_strip and all((ch in _PUNCT_ASCII) or (ch in _PUNCT_CJK) for ch in s_strip):
        tags.append("punct")

    return tags


def _tag_histogram(token_entries: List[Dict[str, Any]]) -> Dict[str, int]:
    h: Dict[str, int] = {}
    for e in token_entries:
        for t in e.get("tags", []) or []:
            h[t] = h.get(t, 0) + 1
    return dict(sorted(h.items(), key=lambda kv: (-kv[1], kv[0])))


def _md_escape(s: str) -> str:
    # Keep tables readable (escape pipes + newlines)
    return str(s).replace("|", "\\|").replace("\n", "\\n").replace("\r", "\\r")


# -----------------------------------------------------------------------------
# Model helpers
# -----------------------------------------------------------------------------
def _get_unembedding_weight(model) -> torch.Tensor:
    if hasattr(model, "lm_head") and getattr(model.lm_head, "weight", None) is not None:
        W = model.lm_head.weight
    else:
        if not hasattr(model, "get_output_embeddings"):
            raise RuntimeError("Model has no lm_head and no get_output_embeddings().")
        emb = model.get_output_embeddings()
        if emb is None or getattr(emb, "weight", None) is None:
            raise RuntimeError("get_output_embeddings() returned None or has no weight.")
        W = emb.weight
    if not isinstance(W, torch.Tensor) or W.ndim != 2:
        raise RuntimeError(f"Unexpected unembedding weight: type={type(W)} shape={getattr(W, 'shape', None)}")
    return W


# -----------------------------------------------------------------------------
# Basis estimation (decode shared)
# -----------------------------------------------------------------------------
@torch.no_grad()
def _estimate_decode_shared_basis(
    *,
    model,
    tok,
    sub_by: Dict[str, List[Any]],
    layer_idx: int,
    batch_size: int,
    max_prompt_len: int,
    calib_decode_max_new_tokens: int,
    per_task_max_states: int,
    pca_var: float,
    min_dim: int,
    max_dim: int,
    tau: float,
    m_shared: str,
    seed: int,
    k_analyze: int,
    decoding: str,
    temperature: float,
    top_p: float,
    top_k: int,
) -> Tuple[np.ndarray, List[int], Dict[str, Any]]:
    layers, _ = EP.get_model_layers(model)
    if int(layer_idx) < 0 or int(layer_idx) >= len(layers):
        raise ValueError(f"layer_idx={layer_idx} out of range: num_layers={len(layers)}")

    collector = EP.DecodeLastTokenCollector([int(layer_idx)])
    handle = layers[int(layer_idx)].register_forward_hook(collector.make_hook(int(layer_idx)))

    decode_task_states: Dict[str, np.ndarray] = {}
    try:
        for task, sub_exs in sub_by.items():
            collector.set_current_task(task)
            prompts = [ex.prompt for ex in sub_exs]
            EP.collect_decode_states(
                model,
                tok,
                prompts,
                collector,
                batch_size=int(batch_size),
                max_new_tokens=int(calib_decode_max_new_tokens),
                max_prompt_len=int(max_prompt_len),
                decoding=str(decoding),
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
            )
            X = collector.get(task, int(layer_idx))
            if X is None or X.shape[0] == 0:
                raise RuntimeError(f"No decode states collected for task={task}.")
            X = EP.subsample_rows_np(X, int(per_task_max_states), seed=EP.stable_int_seed(seed, task, "decode"))
            decode_task_states[task] = X
    finally:
        try:
            handle.remove()
        except Exception:
            pass
        collector.set_capture(False, None)

    joint, shared_idx, extra = EP.compute_shared_basis_from_states(
        decode_task_states,
        pca_var=float(pca_var),
        min_dim=int(min_dim),
        max_dim=int(max_dim),
        tau=float(tau),
        m_shared=str(m_shared),
        seed=int(seed) + 100,
    )
    if not shared_idx:
        raise RuntimeError("No shared components found (shared_idx empty). Try smaller tau or m_shared.")

    idx_sorted = sorted(shared_idx)
    k_use = min(int(k_analyze), len(idx_sorted))
    if k_use <= 0:
        raise RuntimeError(f"Bad k_analyze={k_analyze}.")
    idx_use = idx_sorted[:k_use]
    Q = EP.orthonormalize_np(joint[:, idx_use])

    # Keep basis metadata JSON/MD-friendly (avoid huge PCA matrices).
    contrib = extra.get("task_contributions", {}) or {}
    contrib_summary: Dict[str, Any] = {}
    for task, cd in contrib.items():
        norm = cd.get("normalized", None)
        try:
            norm_arr = np.asarray(norm, dtype=np.float64)
            top = np.argsort(norm_arr)[-5:][::-1].tolist()
            top_vals = [float(norm_arr[i]) for i in top]
        except Exception:
            top, top_vals = [], []
        contrib_summary[str(task)] = {
            "sample_count": int(cd.get("sample_count", 0) or 0),
            "top_components": top,
            "top_normalized": top_vals,
        }

    basis_meta = {
        "tasks_used": list(extra.get("tasks_used", []) or []),
        "n_balanced": int(extra.get("n_balanced", 0) or 0),
        "cross_dim": int(extra.get("cross_dim", 0) or 0),
        "k_shared_total": int(len(idx_sorted)),
        "k_analyze": int(k_use),
        "shared_idx_sorted": idx_sorted,
        "shared_idx_used": idx_use,
        "task_contrib_top5": contrib_summary,
    }
    return Q.astype(np.float32, copy=False), idx_use, basis_meta


# -----------------------------------------------------------------------------
# Logit-lens projection + reporting
# -----------------------------------------------------------------------------
@torch.no_grad()
def _token_list_from_scores(
    *,
    tok,
    scores: torch.Tensor,
    token_ids: torch.Tensor,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    ids = token_ids.detach().cpu().tolist()
    vals = scores[token_ids].detach().float().cpu().tolist()
    for tid, v in zip(ids, vals):
        raw = _safe_convert_id_to_token(tok, int(tid))
        dec = _safe_decode_one(tok, int(tid))
        out.append(
            {
                "id": int(tid),
                "raw_token": raw,
                "decoded": dec,
                "decoded_repr": repr(dec),
                "score": float(v),
                "tags": _tags_for_token(raw, dec),
            }
        )
    return out


@torch.no_grad()
def _compute_vocab_signature(
    *,
    model,
    tok,
    Q: np.ndarray,
    topk: int,
    include_special: bool,
    per_direction: bool,
) -> Dict[str, Any]:
    W = _get_unembedding_weight(model)
    vocab_size, hidden_dim = int(W.shape[0]), int(W.shape[1])
    if int(Q.shape[0]) != hidden_dim:
        raise ValueError(f"Hidden dim mismatch: Q.shape[0]={Q.shape[0]} vs W.hidden_dim={hidden_dim}")

    special_ids = set(getattr(tok, "all_special_ids", []) or [])
    valid_mask = torch.ones(vocab_size, dtype=torch.bool, device=W.device)
    if not bool(include_special) and special_ids:
        ids = torch.tensor(sorted(int(x) for x in special_ids if 0 <= int(x) < vocab_size), device=W.device)
        if ids.numel() > 0:
            valid_mask[ids] = False

    Q_t = torch.tensor(Q, device=W.device, dtype=W.dtype)
    k = int(Q_t.shape[1])

    # Aggregate signature: L2 norm across directions (per token)
    agg_sumsq = torch.zeros(vocab_size, device=W.device, dtype=torch.float32)

    dirs: List[Dict[str, Any]] = []
    for i in range(k):
        q = Q_t[:, i]
        scores = torch.mv(W, q)  # [vocab]
        agg_sumsq += scores.detach().float().pow(2)

        if not bool(per_direction):
            continue

        # Mask special tokens out of selection (by setting to -inf in the selection view)
        scores_pos = scores.detach().clone()
        scores_pos[~valid_mask] = float("-inf")
        scores_neg = (-scores.detach()).clone()
        scores_neg[~valid_mask] = float("-inf")
        scores_abs = scores.detach().abs().clone()
        scores_abs[~valid_mask] = float("-inf")

        top_pos = torch.topk(scores_pos, k=min(int(topk), vocab_size), largest=True).indices
        top_neg = torch.topk(scores_neg, k=min(int(topk), vocab_size), largest=True).indices
        top_abs = torch.topk(scores_abs, k=min(int(topk), vocab_size), largest=True).indices

        top_pos_list = _token_list_from_scores(tok=tok, scores=scores, token_ids=top_pos)
        top_neg_list = _token_list_from_scores(tok=tok, scores=scores, token_ids=top_neg)
        top_abs_list = _token_list_from_scores(tok=tok, scores=scores, token_ids=top_abs)

        dirs.append(
            {
                "dir": int(i),
                "top_pos": top_pos_list,
                "top_neg": top_neg_list,
                "top_abs": top_abs_list,
                "tag_hist_top_abs": _tag_histogram(top_abs_list),
            }
        )

    agg = agg_sumsq.sqrt()
    agg_sel = agg.detach().clone()
    agg_sel[~valid_mask] = float("-inf")
    agg_top = torch.topk(agg_sel, k=min(int(topk), vocab_size), largest=True).indices
    agg_top_list = _token_list_from_scores(tok=tok, scores=agg, token_ids=agg_top)

    return {
        "vocab_size": int(vocab_size),
        "hidden_dim": int(hidden_dim),
        "k": int(k),
        "aggregate_top": agg_top_list,
        "aggregate_tag_hist": _tag_histogram(agg_top_list),
        "per_direction": dirs,
    }


def _pairwise_overlap(sets: Dict[str, set]) -> Dict[str, Any]:
    keys = list(sets.keys())
    out_pairs = []
    jaccs = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            sa, sb = sets[a], sets[b]
            inter = len(sa & sb)
            uni = len(sa | sb)
            j = float(inter / uni) if uni > 0 else float("nan")
            out_pairs.append({"a": a, "b": b, "inter": int(inter), "union": int(uni), "jaccard": j})
            if not math.isnan(j):
                jaccs.append(j)
    return {
        "pairs": out_pairs,
        "mean_jaccard": float(np.mean(jaccs)) if jaccs else float("nan"),
        "min_jaccard": float(np.min(jaccs)) if jaccs else float("nan"),
        "max_jaccard": float(np.max(jaccs)) if jaccs else float("nan"),
    }


def _md_table(rows: List[List[str]], header: List[str]) -> str:
    out = []
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "---|" * len(header))
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def _render_md_summary(
    *,
    config: Dict[str, Any],
    basis_meta: Dict[str, Any],
    signature: Dict[str, Any],
    md_max_dirs: int,
) -> str:
    lines: List[str] = []
    lines.append("# Exp-1: Logit-lens / vocabulary signature")
    lines.append("")
    lines.append("## Config")
    lines.append("```json")
    lines.append(json.dumps(config, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Basis")
    lines.append("```json")
    lines.append(json.dumps(basis_meta, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")

    # Aggregate signature
    lines.append("## Aggregate top tokens")
    rows = []
    for r, e in enumerate(signature.get("aggregate_top", []) or []):
        rows.append(
            [
                str(r + 1),
                f"{e.get('score', 0.0):+.4f}",
                str(e.get("id")),
                _md_escape(e.get("raw_token", "")),
                _md_escape(e.get("decoded_repr", "")),
                _md_escape(",".join(e.get("tags", []) or [])),
            ]
        )
    lines.append(_md_table(rows, ["rank", "score", "id", "raw", "decoded", "tags"]))
    lines.append("")
    lines.append("Tag histogram (aggregate top-k):")
    lines.append("```json")
    lines.append(json.dumps(signature.get("aggregate_tag_hist", {}), ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")

    # Per-direction
    per_dir = signature.get("per_direction", []) or []
    if per_dir:
        lines.append("## Per-direction top-|score| tokens")
        for d in per_dir[: int(md_max_dirs)]:
            di = int(d.get("dir", -1))
            lines.append(f"### dir={di}")
            top_abs = d.get("top_abs", []) or []
            rows = []
            for r, e in enumerate(top_abs):
                rows.append(
                    [
                        str(r + 1),
                        f"{e.get('score', 0.0):+.4f}",
                        str(e.get("id")),
                        _md_escape(e.get("raw_token", "")),
                        _md_escape(e.get("decoded_repr", "")),
                        _md_escape(",".join(e.get("tags", []) or [])),
                    ]
                )
            lines.append(_md_table(rows, ["rank", "score", "id", "raw", "decoded", "tags"]))
            lines.append("")
            lines.append("Tag histogram (top-|score|):")
            lines.append("```json")
            lines.append(json.dumps(d.get("tag_hist_top_abs", {}), ensure_ascii=False, indent=2))
            lines.append("```")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()

    # Model
    ap.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--trust_remote_code", type=int, default=0, choices=[0, 1])

    # Tasks / prompts
    ap.add_argument("--tasks", type=str, default="gsm8k,commonsenseqa,strategyqa,aqua,arc_challenge,openbookqa,qasc,logiqa,boolq")
    ap.add_argument("--n_prompts", type=int, default=128, help="Prompts per task for basis estimation.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--template_seed", type=int, default=1234)
    ap.add_argument("--template_seeds", type=str, default="", help="Optional CSV of template seeds for stability, e.g. '1234,2345,3456'.")
    ap.add_argument("--template_randomization", type=int, default=1, choices=[0, 1])
    ap.add_argument("--shuffle_choices", type=int, default=1, choices=[0, 1])
    ap.add_argument("--add_answer_prefix", type=int, default=1, choices=[0, 1])
    ap.add_argument("--answer_prefix", type=str, default="\nFinal answer:")

    # Decode collection
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_prompt_len", type=int, default=512)
    ap.add_argument("--calib_decode_max_new_tokens", type=int, default=128)
    ap.add_argument("--decoding", type=str, default="greedy", choices=["greedy", "sample"])
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--per_task_max_states", type=int, default=20000)

    # Subspace params
    ap.add_argument("--pca_var", type=float, default=0.95)
    ap.add_argument("--min_dim", type=int, default=8)
    ap.add_argument("--max_dim", type=int, default=256)
    ap.add_argument("--tau", type=float, default=0.001)
    ap.add_argument("--m_shared", type=str, default="all")
    ap.add_argument("--k_analyze", type=int, default=32, help="How many shared directions to analyze (<= shared_k).")

    # Projection / reporting
    ap.add_argument("--topk", type=int, default=40, help="Top-k tokens to report.")
    ap.add_argument("--include_special", type=int, default=0, choices=[0, 1])
    ap.add_argument("--per_direction", type=int, default=1, choices=[0, 1])
    ap.add_argument("--md_max_dirs", type=int, default=8, help="Max directions rendered in MD summary.")

    # Outputs
    ap.add_argument("--out_dir", type=str, default="results/rebuttal_mechanism/logit_lens")
    ap.add_argument("--tag", type=str, default="", help="Optional tag appended to output filenames.")

    args = ap.parse_args()

    EP.set_global_seed(int(args.seed))
    tasks = _split_csv(args.tasks)
    if not tasks:
        raise ValueError("Empty --tasks")

    if args.template_seeds.strip():
        template_seeds = _dedup_keep_order(_parse_int_csv(args.template_seeds))
    else:
        template_seeds = [int(args.template_seed)]

    out_dir = os.path.expanduser(str(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)
    tag = ("_" + args.tag.strip()) if args.tag.strip() else ""

    # Load model once (basis estimation differs only by prompts/templates)
    model, tok = EP.load_model_and_tokenizer(
        args.model,
        args.device,
        args.dtype,
        trust_remote_code=bool(args.trust_remote_code),
    )

    results: Dict[str, Any] = {
        "config": {
            "model": args.model,
            "device": args.device,
            "dtype": args.dtype,
            "trust_remote_code": bool(args.trust_remote_code),
            "layer": int(args.layer),
            "tasks": tasks,
            "n_prompts": int(args.n_prompts),
            "seed": int(args.seed),
            "template_randomization": bool(args.template_randomization),
            "shuffle_choices": bool(args.shuffle_choices),
            "add_answer_prefix": bool(args.add_answer_prefix),
            "answer_prefix": str(args.answer_prefix),
            "decode_collect": {
                "batch_size": int(args.batch_size),
                "max_prompt_len": int(args.max_prompt_len),
                "calib_decode_max_new_tokens": int(args.calib_decode_max_new_tokens),
                "decoding": str(args.decoding),
                "temperature": float(args.temperature),
                "top_p": float(args.top_p),
                "top_k": int(args.top_k),
                "per_task_max_states": int(args.per_task_max_states),
            },
            "subspace": {
                "pca_var": float(args.pca_var),
                "min_dim": int(args.min_dim),
                "max_dim": int(args.max_dim),
                "tau": float(args.tau),
                "m_shared": str(args.m_shared),
                "k_analyze": int(args.k_analyze),
            },
            "projection": {
                "topk": int(args.topk),
                "include_special": bool(args.include_special),
                "per_direction": bool(args.per_direction),
            },
            "template_seeds": template_seeds,
        },
        "by_template_seed": {},
        "stability": {},
        "saved": {},
    }

    sig_sets: Dict[str, set] = {}

    t0 = time.time()
    for ts in template_seeds:
        print("\n" + "=" * 80)
        print(f"[TemplateSeed] {ts}")
        print("=" * 80)

        sub_by, _eval_by_dummy, meta_by = load_selected_tasks(
            tasks=tasks,
            n_subspace=max(1, int(args.n_prompts)),
            n_eval=1,  # (compat) loader may not accept 0
            seed=int(args.seed),
            template_randomization=bool(args.template_randomization),
            template_seed=int(ts),
            shuffle_choices=bool(args.shuffle_choices),
            add_answer_prefix=bool(args.add_answer_prefix),
            answer_prefix=str(args.answer_prefix),
        )

        Q, shared_idx_used, basis_meta = _estimate_decode_shared_basis(
            model=model,
            tok=tok,
            sub_by=sub_by,
            layer_idx=int(args.layer),
            batch_size=int(args.batch_size),
            max_prompt_len=int(args.max_prompt_len),
            calib_decode_max_new_tokens=int(args.calib_decode_max_new_tokens),
            per_task_max_states=int(args.per_task_max_states),
            pca_var=float(args.pca_var),
            min_dim=int(args.min_dim),
            max_dim=int(args.max_dim),
            tau=float(args.tau),
            m_shared=str(args.m_shared),
            seed=int(args.seed),
            k_analyze=int(args.k_analyze),
            decoding=str(args.decoding),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            top_k=int(args.top_k),
        )

        signature = _compute_vocab_signature(
            model=model,
            tok=tok,
            Q=Q,
            topk=int(args.topk),
            include_special=bool(args.include_special),
            per_direction=bool(args.per_direction),
        )

        # Save basis for reuse
        basis_npz = os.path.join(out_dir, f"basis_layer{int(args.layer)}_tseed{int(ts)}{tag}.npz")
        np.savez(
            basis_npz,
            Q=Q.astype(np.float32),
            shared_idx_used=np.array(shared_idx_used, dtype=np.int32),
            shared_idx_sorted=np.array(basis_meta.get("shared_idx_sorted", []), dtype=np.int32),
            tasks=np.array(tasks),
            meta=json.dumps(meta_by, ensure_ascii=False),
            basis_meta=json.dumps(basis_meta, ensure_ascii=False),
        )

        # Track stability using aggregate top token ids
        agg_ids = [int(e["id"]) for e in signature.get("aggregate_top", []) or []]
        sig_sets[str(ts)] = set(agg_ids)

        results["by_template_seed"][str(ts)] = {
            "meta_by_task": meta_by,
            "basis_meta": basis_meta,
            "signature": signature,
            "saved_basis_npz": os.path.relpath(basis_npz, ROOT_DIR),
        }

        # Emit an MD summary per seed (kept compact)
        md = _render_md_summary(
            config=results["config"],
            basis_meta=basis_meta,
            signature=signature,
            md_max_dirs=int(args.md_max_dirs),
        )
        md_path = os.path.join(out_dir, f"summary_layer{int(args.layer)}_tseed{int(ts)}{tag}.md")
        _atomic_text_dump(md, md_path)
        print(f"[Saved] {md_path}")

    results["stability"] = {"aggregate_topk_overlap": _pairwise_overlap(sig_sets)} if len(sig_sets) >= 2 else {}
    results["timing_sec"] = float(time.time() - t0)

    out_json = os.path.join(out_dir, f"exp_1_logit_lens_vocab_signature_layer{int(args.layer)}{tag}.json")
    _atomic_json_dump(results, out_json)
    print(f"\n[Saved] {out_json}")


if __name__ == "__main__":
    main()
