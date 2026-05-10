# -*- coding: utf-8 -*-
"""
experiment2_subspace_patching_transfer.py


它核心在回答：“跨任务共享子空间（decodeshare）是否是因果的、且是结构性的？”

它覆盖/控制了这些点：
共享子空间的构造是否合理
从多个任务的 decode last-token 激活做 cross-task PCA + shared basis 筛选（你输出了 cross_dim、shared_basis_count、验证正交性等）。
删掉 shared 分量是否会导致错误（因果性）
ablated：在特定层把 P_Q(x) 去掉（α=1），看 baseline→ablated 的变化，构造 flips（baseline 对、ablated 错）。
能否用 patch “修复” ablation 造成的错误（定位到层与步）
patched_0 / patched_01 / patched_full：把 donor 的 P_Q(x) patch 回去，看能否 rescue flips，并测 Δmargin。

关键负对照（排除“能量注入/随便 patch 都行”）
control_rand_subspace：随机子空间 + 能量匹配向量（通常失败）
control_shared_randvec：同一个 shared 子空间里能量匹配随机向量（你已经看到它也失败）→ 强力排除“只要在 Q_shared 里加能量就好”
control_patch_nonshared：正交补空间（应失败）
你还加了 donor cos-sim、flip label 分布 → 解释为何 “cross-example donor” 会强（donor 太相似）。

一句话总结：
这是“机制主实验”：证明 Q_shared 是因果通道，且需要 on-manifold 的结构性 donor，而非能量/随机方向。

Experiment 2: Subspace patching (transfer) — "workspace feels earned"

Fixes / checks added (2026-01):
1) Benchmark eval accuracy correctness:
   - Always scan ALL eval examples loaded (n_eval), regardless of max_flips.
   - Print both accuracy fraction and correct-count (so "50多" is visible).
   - Loader is strictly from benchmark_dataloaders.py (dl.load_selected_tasks),
     not from loto utilities.

2) patched_full vs patched_01 identical:
   - Print candidate tokenization lengths and derived patch step sets.
   - Warn if steps_01 == full_steps (then patched_full == patched_01 by design).
   - At runtime, optionally compute max|scores_full - scores_01| for flips.

Notes:
- Forced-choice is decode-aligned:
  Prefill x1:T-1 (seq_len>1)
  Step 0: decode with xT (seq_len==1) => logits for first candidate token
  Step 1..: decode with candidate token j (seq_len==1) => logits for next token

- full_steps is defined as all decode-step indices that occur during scoring:
  If max candidate token length is L, total decode calls = L (step 0..L-1).

CUDA_VISIBLE_DEVICES=1 python subspace_patching_transfer.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --device cuda --dtype fp32 \
  --layer 10 \
  --compute_Qs 1 \
  --Qs_out Q_shared_layer10.npy \
  --basis_tasks gsm8k,commonsenseqa,strategyqa,openbookqa,qasc,boolq,piqa \
  --basis_n_subspace 128 \
  --out_json patch_results.json

"""

from __future__ import annotations

import os
import json
import argparse
import importlib.util
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Iterable, Set

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter, defaultdict
from collections import Counter
import math

# =============================================================================
# Dynamic imports: reuse the two attached scripts
# =============================================================================

def _import_from_path(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_name} from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def load_aux_modules(loto8_path: str, dataloaders_path: str):
    loto8 = _import_from_path("loto8", loto8_path)
    dl = _import_from_path("benchmark_dataloaders", dataloaders_path)
    return loto8, dl


# =============================================================================
# Model layer access (fallback)
# =============================================================================

def _getattr_nested(obj: Any, path: str) -> Any:
    cur = obj
    for p in path.split("."):
        if not hasattr(cur, p):
            return None
        cur = getattr(cur, p)
    return cur


def get_transformer_layers(model: torch.nn.Module) -> Tuple[List[torch.nn.Module], str]:
    """
    Returns (layers, path_used) for common HF causal decoders.
    """
    candidate_paths = [
        "model.layers",
        "model.model.layers",
        "model.decoder.layers",
        "model.model.decoder.layers",
        "transformer.h",
        "model.transformer.h",
        "gpt_neox.layers",
        "model.gpt_neox.layers",
    ]
    for path in candidate_paths:
        layers = _getattr_nested(model, path)
        if layers is None:
            continue
        if isinstance(layers, (torch.nn.ModuleList, list, tuple)) and len(layers) > 0:
            return list(layers), path
    raise RuntimeError(
        "Could not locate transformer block list on this model. "
        "Edit get_transformer_layers() to match your architecture."
    )


# =============================================================================
# KV-cache utilities
# =============================================================================

def repeat_past_key_values(past_key_values: Any, repeat: int) -> Any:
    """
    Repeat batch dimension of past_key_values (assumes current batch=1).
    Works for most HF causal models where past_key_values is
      tuple[num_layers] of tuples[tensors...]
    """
    if past_key_values is None:
        return None

    def _repeat_tensor(t: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(t):
            return t
        reps = [repeat] + [1] * (t.ndim - 1)
        return t.repeat(*reps)

    if isinstance(past_key_values, (tuple, list)):
        out = []
        for layer in past_key_values:
            if isinstance(layer, (tuple, list)):
                out_layer = []
                for item in layer:
                    out_layer.append(_repeat_tensor(item) if torch.is_tensor(item) else item)
                out.append(tuple(out_layer))
            else:
                out.append(layer)
        return tuple(out) if isinstance(past_key_values, tuple) else out

    return past_key_values


# =============================================================================
# Hooks: capture + patch (decode-only, seq_len==1)
# =============================================================================

class DecodeStepHiddenCaptureHook:
    """
    Captures the last-token hidden state x = h[:, -1, :] at selected decode steps.
    The step counter increments ONLY on seq_len==1 forward calls.

    Stores CPU float32 tensors in `hidden_by_step[t]` with shape [B, d].
    """
    def __init__(self, capture_steps: Optional[Iterable[int]] = None):
        self.capture_steps = None if capture_steps is None else set(int(s) for s in capture_steps)
        self.step = 0
        self.hidden_by_step: Dict[int, torch.Tensor] = {}

    def reset(self):
        self.step = 0
        self.hidden_by_step = {}

    def __call__(self, module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(hs) or hs.ndim != 3:
            return output
        if hs.shape[1] != 1:
            return output  # strict decode-only

        t = self.step
        self.step += 1

        if self.capture_steps is None or t in self.capture_steps:
            self.hidden_by_step[t] = hs[:, -1, :].detach().float().cpu()
        return output


class SubspacePatchHook:
    """
    Decode-only hook at ONE layer.
    Replaces the component in span(Q) at selected decode steps with a donor component.

    For step t in patch_steps:
      s_t = P_Q(x_t)
      x'_t = x_t - s_t + p_t   where p_t is donor_by_step[t].
    """
    def __init__(self, Q_np: np.ndarray, donor_by_step: Dict[int, torch.Tensor], patch_steps: Iterable[int]):
        Q = np.asarray(Q_np, dtype=np.float32)
        if Q.ndim != 2:
            raise ValueError(f"Q_np must be 2D [d,k], got shape={Q.shape}")
        self.Q_cpu = torch.tensor(Q, dtype=torch.float32, device="cpu").contiguous()  # [d,k]
        self.Q_dev: Optional[torch.Tensor] = None
        self.donor_by_step = {int(k): v.detach().float().cpu() for k, v in donor_by_step.items()}
        self.patch_steps: Set[int] = set(int(s) for s in patch_steps)
        self.step = 0

    def reset(self):
        self.step = 0

    def _Q(self, device: torch.device) -> torch.Tensor:
        if self.Q_dev is None or self.Q_dev.device != device:
            self.Q_dev = self.Q_cpu.to(device=device)
        return self.Q_dev

    def __call__(self, module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(hs) or hs.ndim != 3:
            return output
        if hs.shape[1] != 1:
            return output

        t = self.step
        self.step += 1
        if t not in self.patch_steps:
            return output
        if t not in self.donor_by_step:
            return output

        Q = self._Q(hs.device)          # [d,k]
        x = hs[:, -1, :].float()        # [B,d]
        s = (x @ Q) @ Q.T               # [B,d]
        p = self.donor_by_step[t].to(device=hs.device, dtype=torch.float32)  # [B,d]
        if p.shape != x.shape:
            raise RuntimeError(f"Donor shape mismatch at step={t}: donor={tuple(p.shape)} vs x={tuple(x.shape)}")
        x_new = x - s + p

        hs2 = hs.clone()
        hs2[:, -1, :] = x_new.to(dtype=hs2.dtype)
        if isinstance(output, tuple):
            return (hs2,) + output[1:]
        return hs2


# =============================================================================
# Subspace math utilities
# =============================================================================

def orthonormalize_np(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=np.float32)
    q, _ = np.linalg.qr(M)
    return q.astype(np.float32, copy=False)


def project_cpu(x_cpu: torch.Tensor, Q_np: np.ndarray) -> torch.Tensor:
    """
    x_cpu: torch float32 on CPU, shape [B,d]
    Q_np: numpy float32 [d,k], assumed orthonormal
    Returns p = P_Q x = (xQ)Q^T on CPU float32, shape [B,d]
    """
    if x_cpu.device.type != "cpu":
        x_cpu = x_cpu.cpu()
    x = x_cpu.float()
    Q = torch.tensor(Q_np, dtype=torch.float32, device="cpu")
    return (x @ Q) @ Q.T


def sample_random_orthonormal_basis(d: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, k)).astype(np.float32)
    return orthonormalize_np(A)


def sample_random_orthonormal_complement(Q_shared: np.ndarray, k: int, seed: int) -> np.ndarray:
    """
    Returns a random orthonormal basis Q_ns with shape [d,k] approximately orthogonal to Q_shared.
    """
    Qs = orthonormalize_np(Q_shared)
    d = Qs.shape[0]
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, k + 16)).astype(np.float32)
    A = A - Qs @ (Qs.T @ A)
    Q, _ = np.linalg.qr(A)
    return Q[:, :k].astype(np.float32, copy=False)


def energy_matched_random_vector_in_subspace(
    Q_sub: np.ndarray,
    target_norms: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    """
    Draw random vectors r in span(Q_sub) with per-row L2 norms matched to target_norms.

    Q_sub: [d,k] orthonormal
    target_norms: [B] CPU tensor float32
    Returns: r_cpu [B,d] float32
    """
    Q = torch.tensor(Q_sub, dtype=torch.float32, device="cpu")  # [d,k]
    B = int(target_norms.shape[0])
    k = int(Q.shape[1])
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((B, k)).astype(np.float32)
    z = torch.tensor(z, dtype=torch.float32, device="cpu")  # [B,k]
    eps = 1e-12
    z = z / (torch.linalg.norm(z, dim=1)[:, None] + eps)
    z = z * target_norms[:, None]
    return z @ Q.T  # [B,d]


# =============================================================================
# Forced-choice scoring (decode-aligned, batched candidates)
# =============================================================================

@dataclass
class FCResult:
    pred_label: str
    correct: bool
    margin: float
    scores: Dict[str, float]  # label -> logprob


def _maybe_call_model(model: AutoModelForCausalLM, **kwargs):
    """
    Try to force legacy cache if supported by this Transformers version,
    otherwise fall back.
    """
    # Always request dict output for consistency
    kwargs = dict(kwargs)
    kwargs.setdefault("return_dict", True)
    try:
        # Some versions support this and will return tuple past_key_values
        return model(**kwargs, return_legacy_cache=True)
    except TypeError:
        return model(**kwargs)


@torch.no_grad()
def forced_choice_decode_aligned(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    candidate_labels: List[str],
    candidate_texts: List[str],
    gold_label: str,
    *,
    layer_module: torch.nn.Module,
    removal_hook: Optional[Any] = None,
    patch_hook: Optional[SubspacePatchHook] = None,
    capture_hook: Optional[DecodeStepHiddenCaptureHook] = None,
    add_special_tokens_prompt: bool = True,
) -> FCResult:
    """
    Cache-advanced decode-aligned forced-choice scoring.

    Hook order: removal_hook -> patch_hook -> capture_hook
    """
    model.eval()
    device = next(model.parameters()).device

    toks = tokenizer(prompt, return_tensors="pt", add_special_tokens=add_special_tokens_prompt)
    input_ids = toks["input_ids"].to(device)
    attn = torch.ones_like(input_ids)

    cand_token_ids: List[List[int]] = [tokenizer.encode(ct, add_special_tokens=False) for ct in candidate_texts]
    if any(len(x) == 0 for x in cand_token_ids):
        raise RuntimeError("Some candidate_text tokenized to empty list. Check candidate_texts.")

    def _register_hooks():
        handles = []
        if removal_hook is not None:
            handles.append(layer_module.register_forward_hook(removal_hook))
        if patch_hook is not None:
            patch_hook.reset()
            handles.append(layer_module.register_forward_hook(patch_hook))
        if capture_hook is not None:
            capture_hook.reset()
            handles.append(layer_module.register_forward_hook(capture_hook))
        return handles

    # -------------------------------------------------------------------------
    # Fast path: batched candidates if past_key_values is legacy (tuple/list)
    # -------------------------------------------------------------------------
    handles = _register_hooks()
    try:
        # Prefill + step0 once
        if input_ids.shape[1] > 1:
            out_pre = _maybe_call_model(
                model,
                input_ids=input_ids[:, :-1],
                attention_mask=attn[:, :-1],
                use_cache=True,
            )
            past = out_pre.past_key_values
            out0 = _maybe_call_model(
                model,
                input_ids=input_ids[:, -1:],
                attention_mask=attn,
                past_key_values=past,
                use_cache=True,
            )
        else:
            out0 = _maybe_call_model(
                model,
                input_ids=input_ids,
                attention_mask=attn,
                use_cache=True,
            )

        past0 = out0.past_key_values
        logits0 = out0.logits[:, -1, :]             # [1,V]
        logp0 = torch.log_softmax(logits0, dim=-1)  # [1,V]
        attn0 = attn

        legacy_cache = isinstance(past0, (tuple, list))

        if legacy_cache:
            K = len(cand_token_ids)
            max_len = max(len(x) for x in cand_token_ids)

            fill_id = tokenizer.eos_token_id
            if fill_id is None:
                fill_id = tokenizer.pad_token_id
            if fill_id is None:
                fill_id = 0

            tok_mat = torch.full((K, max_len), int(fill_id), dtype=torch.long, device=device)
            lengths = torch.tensor([len(x) for x in cand_token_ids], dtype=torch.long, device=device)
            for i, ids in enumerate(cand_token_ids):
                tok_mat[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)

            # First token score from logits0
            first_tokens = tok_mat[:, 0]               # [K]
            scores = logp0[0, first_tokens].clone()    # [K]

            # Remaining tokens (batched) with repeated legacy cache
            if max_len > 1:
                past = repeat_past_key_values(past0, K)
                attn_k = attn0.repeat(K, 1)

                for j in range(max_len - 1):
                    inp = tok_mat[:, j].unsqueeze(1)  # [K,1]
                    attn_k = torch.cat(
                        [attn_k, torch.ones((K, 1), device=device, dtype=attn_k.dtype)],
                        dim=1
                    )
                    outj = _maybe_call_model(
                        model,
                        input_ids=inp,
                        attention_mask=attn_k,
                        past_key_values=past,
                        use_cache=True,
                    )
                    past = outj.past_key_values
                    logpj = torch.log_softmax(outj.logits[:, -1, :], dim=-1)  # [K,V]
                    next_tok = tok_mat[:, j + 1]                              # [K]
                    mask = (lengths > (j + 1)).float()                        # [K]
                    scores = scores + logpj[torch.arange(K, device=device), next_tok] * mask

            score_by_label = {lab: float(scores[i].item()) for i, lab in enumerate(candidate_labels)}
            pred_idx = int(torch.argmax(scores).item())
            pred_label = candidate_labels[pred_idx]

            if gold_label in candidate_labels:
                gold_idx = candidate_labels.index(gold_label)
                gold_score = scores[gold_idx]
                other_scores = torch.cat([scores[:gold_idx], scores[gold_idx + 1:]], dim=0)
                margin = float((gold_score - torch.max(other_scores)).item()) if other_scores.numel() > 0 else float("nan")
                correct = (pred_label == gold_label)
            else:
                margin = float("nan")
                correct = False

            return FCResult(pred_label=pred_label, correct=correct, margin=margin, scores=score_by_label)

    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # Fallback: unbatched per-candidate (robust for Cache objects)
    # NOTE: This path is correct for scoring; patching beyond step0 is less
    # meaningful unless donor vectors are also computed per-candidate.
    # -------------------------------------------------------------------------
    scores_list: List[float] = []

    for cand_ids in cand_token_ids:
        handles = _register_hooks()
        try:
            if input_ids.shape[1] > 1:
                out_pre = _maybe_call_model(
                    model,
                    input_ids=input_ids[:, :-1],
                    attention_mask=attn[:, :-1],
                    use_cache=True,
                )
                past = out_pre.past_key_values
                out0 = _maybe_call_model(
                    model,
                    input_ids=input_ids[:, -1:],
                    attention_mask=attn,
                    past_key_values=past,
                    use_cache=True,
                )
            else:
                out0 = _maybe_call_model(
                    model,
                    input_ids=input_ids,
                    attention_mask=attn,
                    use_cache=True,
                )

            logits0 = out0.logits[:, -1, :]
            logp0 = torch.log_softmax(logits0, dim=-1)
            past = out0.past_key_values
            attn_cur = attn

            s = logp0[0, cand_ids[0]].clone()

            for j in range(len(cand_ids) - 1):
                tid = cand_ids[j]
                tid_next = cand_ids[j + 1]
                attn_cur = torch.cat(
                    [attn_cur, torch.ones((1, 1), device=device, dtype=attn_cur.dtype)],
                    dim=1
                )
                inp = torch.tensor([[tid]], device=device, dtype=input_ids.dtype)
                outj = _maybe_call_model(
                    model,
                    input_ids=inp,
                    attention_mask=attn_cur,
                    past_key_values=past,
                    use_cache=True,
                )
                past = outj.past_key_values
                s = s + torch.log_softmax(outj.logits[:, -1, :], dim=-1)[0, tid_next]

            scores_list.append(float(s.item()))
        finally:
            for h in handles:
                try:
                    h.remove()
                except Exception:
                    pass

    scores = torch.tensor(scores_list, device="cpu", dtype=torch.float32)
    score_by_label = {lab: float(scores[i].item()) for i, lab in enumerate(candidate_labels)}
    pred_idx = int(torch.argmax(scores).item())
    pred_label = candidate_labels[pred_idx]

    if gold_label in candidate_labels:
        gold_idx = candidate_labels.index(gold_label)
        gold_score = scores[gold_idx]
        other_scores = torch.cat([scores[:gold_idx], scores[gold_idx + 1:]], dim=0)
        margin = float((gold_score - torch.max(other_scores)).item()) if other_scores.numel() > 0 else float("nan")
        correct = (pred_label == gold_label)
    else:
        margin = float("nan")
        correct = False

    return FCResult(pred_label=pred_label, correct=correct, margin=margin, scores=score_by_label)


def shuffle_coeffs_in_subspace(p0_cpu: torch.Tensor, Q_sub: np.ndarray, seed: int, mode: str = "permute") -> torch.Tensor:
    """
    p0_cpu: [B,d] CPU float32, assumed in span(Q_sub)
    Returns a new vector in span(Q_sub) with same norm per row but shuffled structure.
    """
    Q = torch.tensor(Q_sub, dtype=torch.float32, device="cpu")  # [d,k]
    c = p0_cpu @ Q  # [B,k]
    k = c.shape[1]
    rng = np.random.default_rng(seed)

    if mode == "permute":
        perm = torch.tensor(rng.permutation(k), dtype=torch.long)
        c2 = c[:, perm]
    elif mode == "signflip":
        signs = torch.tensor(rng.choice([-1.0, 1.0], size=(k,)).astype(np.float32))
        c2 = c * signs[None, :]
    else:
        raise ValueError(mode)

    return c2 @ Q.T  # [B,d]


# =============================================================================
# Experiment runner helpers
# =============================================================================

def build_candidate_texts(candidate_labels: List[str], style: str) -> List[str]:
    if style == "raw":
        return [lab for lab in candidate_labels]
    if style == "space_letter":
        return [" " + lab for lab in candidate_labels]
    raise ValueError(f"Unknown candidate_text_style={style}")


def summarize_rescue(rows: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n": 0, "rescued": 0, "rescued_pct": float("nan"), "mean_dmargin": float("nan")}
    rescued = 0
    dms = []
    for r in rows:
        if r[key]["correct"]:
            rescued += 1
        dms.append(r[key]["margin"] - r["ablated"]["margin"])
    return {
        "n": n,
        "rescued": rescued,
        "rescued_pct": 100.0 * rescued / n,
        "mean_dmargin": float(np.mean(dms)),
        "median_dmargin": float(np.median(dms)),
    }


def _tokens_debug(tok: AutoTokenizer, ids: List[int]) -> str:
    try:
        pieces = tok.convert_ids_to_tokens(ids)
        return " ".join(pieces)
    except Exception:
        return "<convert_ids_to_tokens failed>"


def maybe_compute_Qs(
    *,
    loto8: Any,
    dl: Any,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    layer_idx: int,
    seed: int,
    tasks: List[str],
    n_subspace: int,
    template_randomization: bool,
    shuffle_choices: bool,
    answer_prefix: str,
    calib_batch_size: int,
    calib_max_new_tokens: int,
    per_task_max_states: int,
    max_prompt_len: int,
    variance_threshold: float,
    min_dim: int,
    max_dim: int,
    tau: float,
    m_shared: str,
    out_path: str,
) -> np.ndarray:
    print("\n[Compute Q_shared] Loading prompts for basis estimation ...")

    orig_bfs = getattr(dl, "_build_from_splits_with_fallback", None)

    if orig_bfs is not None:
        def _bfs_noeval(ds, dataset_name, build_one, *, n_subspace, n_eval, seed,
                        sub_candidates, eval_candidates, require_gold_eval=True):
            if n_eval and int(n_eval) > 0:
                return orig_bfs(
                    ds, dataset_name, build_one,
                    n_subspace=n_subspace, n_eval=n_eval, seed=seed,
                    sub_candidates=sub_candidates, eval_candidates=eval_candidates,
                    require_gold_eval=require_gold_eval,
                )

            def make_examples(split_name, rows, require_gold):
                out = []
                for i, ex in enumerate(rows):
                    ex_id = f"{dataset_name}-{split_name}-{i}"
                    p, g = build_one(ex, ex_id)
                    p = (p or "").strip()
                    g = (g or "").strip()
                    if not p:
                        continue
                    if require_gold and not g:
                        continue
                    out.append(dl.Example(dataset=dataset_name, ex_id=ex_id, prompt=p, gold=g))
                return out

            sub_exs = []
            sub_split = None
            for j, sp in enumerate(sub_candidates):
                if sp not in ds:
                    continue
                if hasattr(dl, "sample_hf_split"):
                    rows = dl.sample_hf_split(ds[sp], n_subspace, seed + 1000 + 13 * j)
                else:
                    rows = list(ds[sp])[: int(n_subspace)]
                sub_exs = make_examples(sp, rows, require_gold=False)
                if len(sub_exs) > 0:
                    sub_split = sp
                    break
            if sub_split is None:
                raise RuntimeError(
                    f"[{dataset_name}] Could not build ANY subspace prompts from splits={list(ds.keys())}. "
                    f"Check schema extraction for this dataset."
                )

            meta = {"subspace_split": sub_split, "eval_split": None, "available_splits": list(ds.keys())}
            return sub_exs, [], meta

        dl._build_from_splits_with_fallback = _bfs_noeval  # type: ignore[attr-defined]

    try:
        sub_by, _, meta_by = dl.load_selected_tasks(
            tasks=tasks,
            n_subspace=n_subspace,
            n_eval=0,
            seed=seed,
            template_randomization=template_randomization,
            template_seed=seed + 999,
            shuffle_choices=shuffle_choices,
            add_answer_prefix=True,
            answer_prefix=answer_prefix,
        )
    finally:
        if orig_bfs is not None:
            dl._build_from_splits_with_fallback = orig_bfs  # type: ignore[attr-defined]

    prompts_by_task = {t: [ex.prompt for ex in sub_by[t] if ex.prompt] for t in tasks}
    for t in tasks:
        print(f"  task={t:>14s}: prompts={len(prompts_by_task[t])}")

    print("\n[Compute Q_shared] Estimating decode-aligned shared subspace ...")
    joint_subspace, shared_indices, extra, task_acts = loto8.compute_shared_subspace_decode_aligned(
        model=model,
        tokenizer=tokenizer,
        prompts_by_task=prompts_by_task,
        layer_indices=[layer_idx],
        calib_decoding="greedy",
        calib_batch_size=calib_batch_size,
        calib_max_new_tokens=calib_max_new_tokens,
        per_task_max_states=per_task_max_states,
        max_prompt_len=max_prompt_len,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        global_seed=seed,
        variance_threshold=variance_threshold,
        min_dim=min_dim,
        max_dim=max_dim,
        tau=tau,
        m_shared=m_shared,
    )
    if not shared_indices:
        raise RuntimeError("No shared_indices found by shared-subspace estimator. Try adjusting tau/m_shared or max_dim.")

    Q_shared = joint_subspace[:, shared_indices].astype(np.float32, copy=False)
    Q_shared = orthonormalize_np(Q_shared)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.save(out_path, Q_shared)
    print(f"[Compute Q_shared] Saved Q_shared to {out_path}  shape={Q_shared.shape}")
    return Q_shared


# =============================================================================
# Dataloader robustness helper: eval-only
# =============================================================================

def load_selected_tasks_eval_only(
    dl: Any, *,
    task: str,
    n_eval: int,
    seed: int,
    template_randomization: bool,
    template_seed: int,
    shuffle_choices: bool,
    answer_prefix: str
):
    """
    Guarantee eval examples come from benchmark_dataloaders.load_selected_tasks
    even if it insists on building subspace prompts when n_subspace==0.

    We monkeypatch dl._build_from_splits_with_fallback so that when n_subspace<=0
    it only builds eval examples.
    """
    orig_bfs = getattr(dl, "_build_from_splits_with_fallback", None)
    if orig_bfs is None:
        return dl.load_selected_tasks(
            tasks=[task],
            n_subspace=0,
            n_eval=n_eval,
            seed=seed,
            template_randomization=template_randomization,
            template_seed=template_seed,
            shuffle_choices=shuffle_choices,
            add_answer_prefix=True,
            answer_prefix=answer_prefix,
        )

    def _bfs_eval_only(ds, dataset_name, build_one, *, n_subspace, n_eval, seed,
                       sub_candidates, eval_candidates, require_gold_eval=True):
        if n_subspace and int(n_subspace) > 0:
            return orig_bfs(
                ds, dataset_name, build_one,
                n_subspace=n_subspace, n_eval=n_eval, seed=seed,
                sub_candidates=sub_candidates, eval_candidates=eval_candidates,
                require_gold_eval=require_gold_eval,
            )

        def make_examples(split_name, rows, require_gold):
            out = []
            for i, ex in enumerate(rows):
                ex_id = f"{dataset_name}-{split_name}-{i}"
                p, g = build_one(ex, ex_id)
                p = (p or "").strip()
                g = (g or "").strip()
                if not p:
                    continue
                if require_gold and not g:
                    continue
                out.append(dl.Example(dataset=dataset_name, ex_id=ex_id, prompt=p, gold=g))
            return out

        eval_exs = []
        eval_split = None
        for j, sp in enumerate(eval_candidates):
            if sp not in ds:
                continue
            if hasattr(dl, "sample_hf_split"):
                rows = dl.sample_hf_split(ds[sp], n_eval, seed + 2000 + 13 * j)
            else:
                rows = list(ds[sp])[: int(n_eval)]
            eval_exs = make_examples(sp, rows, require_gold=require_gold_eval)
            if len(eval_exs) > 0:
                eval_split = sp
                break

        if eval_split is None:
            raise RuntimeError(
                f"[{dataset_name}] Could not build ANY eval examples from candidate splits={eval_candidates}. "
                f"Available splits={list(ds.keys())}."
            )

        meta = {"subspace_split": None, "eval_split": eval_split, "available_splits": list(ds.keys())}
        return [], eval_exs, meta

    dl._build_from_splits_with_fallback = _bfs_eval_only  # type: ignore[attr-defined]
    try:
        return dl.load_selected_tasks(
            tasks=[task],
            n_subspace=0,
            n_eval=n_eval,
            seed=seed,
            template_randomization=template_randomization,
            template_seed=template_seed,
            shuffle_choices=shuffle_choices,
            add_answer_prefix=True,
            answer_prefix=answer_prefix,
        )
    finally:
        dl._build_from_splits_with_fallback = orig_bfs  # type: ignore[attr-defined]


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="HF model id or local path")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--layer", type=int, required=True, help="Layer index to hook (0-based)")
    ap.add_argument("--task", type=str, default="aqua", help="Eval task (default: aqua)")

    ap.add_argument("--n_eval", type=int, default=256, help="Number of eval examples to scan")
    ap.add_argument("--max_flips", type=int, default=64, help="Max number of flip examples to run patching on")

    ap.add_argument("--candidate_labels", type=str, default="ABCDE")
    ap.add_argument("--candidate_text_style", type=str, default="space_letter", choices=["space_letter", "raw"])
    ap.add_argument("--add_special_tokens_prompt", type=int, default=1)

    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--Qs_path", type=str, default="", help="Path to Q_shared .npy [d,k].")
    ap.add_argument("--compute_Qs", type=int, default=0)
    ap.add_argument("--Qs_out", type=str, default="Q_shared_computed.npy")

    ap.add_argument("--basis_tasks", type=str, default="gsm8k,commonsenseqa,strategyqa,arc_challenge,openbookqa,qasc,logiqa")
    ap.add_argument("--basis_n_subspace", type=int, default=2048)
    ap.add_argument("--calib_batch_size", type=int, default=8)
    ap.add_argument("--calib_max_new_tokens", type=int, default=128)
    ap.add_argument("--per_task_max_states", type=int, default=20000)
    ap.add_argument("--max_prompt_len", type=int, default=1024)
    ap.add_argument("--variance_threshold", type=float, default=0.95)
    ap.add_argument("--min_dim", type=int, default=8)
    ap.add_argument("--max_dim", type=int, default=1024)
    ap.add_argument("--tau", type=float, default=0.001)
    ap.add_argument("--m_shared", type=str, default="all")

    ap.add_argument("--loto8_path", type=str, default="disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py")
    ap.add_argument("--dataloaders_path", type=str, default="benchmark_dataloaders.py")

    ap.add_argument("--answer_prefix", type=str, default="\nFinal answer:")
    ap.add_argument("--template_randomization", type=int, default=1)
    ap.add_argument("--shuffle_choices", type=int, default=1)

    ap.add_argument("--out_json", type=str, default="patching_results.json")
    args = ap.parse_args()

    # Load helper modules
    loto8, dl = load_aux_modules(args.loto8_path, args.dataloaders_path)

    # Seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load model
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch_dtype,
            device_map=None,
        ).to(args.device)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch_dtype,
            device_map=None,
        ).to(args.device)

    model.eval()

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    # Find layer module for hooking
    layers, path_used = get_transformer_layers(model)
    if args.layer < 0 or args.layer >= len(layers):
        raise ValueError(f"--layer {args.layer} out of range for layers at {path_used} (n={len(layers)})")
    layer_module = layers[args.layer]
    print(f"[Info] Hooking layer={args.layer} at path {path_used}")

    # Load or compute Q_shared
    Qs: Optional[np.ndarray] = None
    if args.Qs_path:
        Qs = orthonormalize_np(np.load(args.Qs_path).astype(np.float32))
        print(f"[Info] Loaded Q_shared from {args.Qs_path}  shape={Qs.shape}")
    elif args.compute_Qs:
        tasks = [t.strip() for t in args.basis_tasks.split(",") if t.strip()]
        Qs = maybe_compute_Qs(
            loto8=loto8,
            dl=dl,
            model=model,
            tokenizer=tok,
            layer_idx=args.layer,
            seed=args.seed,
            tasks=tasks,
            n_subspace=args.basis_n_subspace,
            template_randomization=bool(args.template_randomization),
            shuffle_choices=bool(args.shuffle_choices),
            answer_prefix=args.answer_prefix,
            calib_batch_size=args.calib_batch_size,
            calib_max_new_tokens=args.calib_max_new_tokens,
            per_task_max_states=args.per_task_max_states,
            max_prompt_len=args.max_prompt_len,
            variance_threshold=args.variance_threshold,
            min_dim=args.min_dim,
            max_dim=args.max_dim,
            tau=args.tau,
            m_shared=args.m_shared,
            out_path=args.Qs_out,
        )
    else:
        raise RuntimeError("Provide --Qs_path or set --compute_Qs=1")

    assert Qs is not None
    d, k = Qs.shape

    # Candidate setup
    candidate_labels = list(args.candidate_labels.strip())
    candidate_texts = build_candidate_texts(candidate_labels, style=args.candidate_text_style)

    cand_token_ids = [tok.encode(ct, add_special_tokens=False) for ct in candidate_texts]
    cand_lens = [len(x) for x in cand_token_ids]
    max_len = max(cand_lens)

    print("\n[Info] Candidate tokenization debug:")
    for lab, ct, ids in zip(candidate_labels, candidate_texts, cand_token_ids):
        print(f"  label={lab} text={ct!r} len={len(ids)} ids={ids} toks={_tokens_debug(tok, ids)}")
    print(f"[Info] max candidate token length = {max_len}")

    # Patch windows derived from max_len:
    # total decode-step calls = max_len (step indices 0..max_len-1)
    full_steps = set(range(max_len))
    steps_0 = {0}
    steps_01 = {0, 1} if max_len >= 2 else {0}

    print(f"[Info] patch windows: steps_0={sorted(list(steps_0))}, steps_01={sorted(list(steps_01))}, full_steps={sorted(list(full_steps))}")
    if steps_01 == full_steps:
        print("[Warn] steps_01 == full_steps, so patched_01 will be IDENTICAL to patched_full (by design). "
              "Reason: max candidate token length <= 2. If you want them different, use longer candidate texts.")

    # Load evaluation examples (benchmark dataloaders only)
    sub_by, eval_by, meta_by = load_selected_tasks_eval_only(
        dl,
        task=args.task,
        n_eval=args.n_eval,
        seed=args.seed,
        template_randomization=bool(args.template_randomization),
        template_seed=args.seed + 999,
        shuffle_choices=bool(args.shuffle_choices),
        answer_prefix=args.answer_prefix,
    )
    eval_examples = eval_by[args.task]
    eval_meta = meta_by.get(args.task, {})
    print(f"\n[Info] Loaded eval examples: task={args.task}, n={len(eval_examples)}  meta={eval_meta}")

    if len(eval_examples) == 0:
        raise RuntimeError("No eval examples loaded. Check HF dataset availability / splits / dataloader extraction.")

    # -------------------------------------------------------------------------
    # Scan for flips (baseline correct, ablated wrong)
    # IMPORTANT FIX: do NOT break early when we have enough flips;
    # keep scanning to get correct benchmark accuracy.
    # -------------------------------------------------------------------------
    scan_rows: List[Dict[str, Any]] = []
    flip_examples_all: List[Any] = []
    flip_examples_used: List[Any] = []

    for ex in eval_examples:
        prompt = ex.prompt
        gold = (ex.gold or "").strip().upper()

        # Skip if gold not in candidate set (keeps accuracy meaningful)
        if gold not in candidate_labels:
            scan_rows.append({
                "ex_id": ex.ex_id,
                "gold": gold,
                "baseline": {"pred_label": "", "correct": False, "margin": float("nan"), "scores": {}},
                "ablated": {"pred_label": "", "correct": False, "margin": float("nan"), "scores": {}},
                "skipped_reason": f"gold_not_in_candidates (gold={gold})",
            })
            continue

        base = forced_choice_decode_aligned(
            model, tok, prompt,
            candidate_labels, candidate_texts, gold,
            layer_module=layer_module,
            add_special_tokens_prompt=bool(args.add_special_tokens_prompt),
        )

        stats = loto8.HookStats("remove_shared")
        remove = loto8.LastTokenRemovalHook(Qs, alpha=1.0, stats=stats)
        ablt = forced_choice_decode_aligned(
            model, tok, prompt,
            candidate_labels, candidate_texts, gold,
            layer_module=layer_module,
            removal_hook=remove,
            add_special_tokens_prompt=bool(args.add_special_tokens_prompt),
        )

        scan_rows.append({
            "ex_id": ex.ex_id,
            "gold": gold,
            "baseline": base.__dict__,
            "ablated": ablt.__dict__,
        })

        if base.correct and (not ablt.correct):
            flip_examples_all.append(ex)
            if len(flip_examples_used) < args.max_flips:
                flip_examples_used.append(ex)

    n_scanned = len(scan_rows)
    base_correct_n = int(sum(1 for r in scan_rows if r.get("baseline", {}).get("correct") is True))
    ablt_correct_n = int(sum(1 for r in scan_rows if r.get("ablated", {}).get("correct") is True))
    base_acc = base_correct_n / n_scanned if n_scanned else float("nan")
    ablt_acc = ablt_correct_n / n_scanned if n_scanned else float("nan")

    n_flips_total = len(flip_examples_all)
    n_flips_used = len(flip_examples_used)

    print(f"\n[Scan] n_scanned={n_scanned}  baseline_acc={base_acc:.3f} ({base_correct_n}/{n_scanned})  "
          f"ablated_acc={ablt_acc:.3f} ({ablt_correct_n}/{n_scanned})  "
          f"flips_total={n_flips_total}  flips_used={n_flips_used}")

    if n_flips_used == 0:
        out = {
            "meta": {
                "model": args.model,
                "device": args.device,
                "dtype": args.dtype,
                "layer": args.layer,
                "task": args.task,
                "eval_meta": eval_meta,
                "candidate_labels": candidate_labels,
                "candidate_text_style": args.candidate_text_style,
                "candidate_token_lens": {lab: int(l) for lab, l in zip(candidate_labels, cand_lens)},
                "max_candidate_token_len": int(max_len),
                "patch_windows": {
                    "steps_0": sorted(list(steps_0)),
                    "steps_01": sorted(list(steps_01)),
                    "full_steps": sorted(list(full_steps)),
                },
                "seed": args.seed,
                "Qs_path": args.Qs_path or args.Qs_out,
                "Qs_shape": [int(d), int(k)],
                "n_scanned": n_scanned,
                "baseline_acc": base_acc,
                "baseline_correct_n": base_correct_n,
                "ablated_acc": ablt_acc,
                "ablated_correct_n": ablt_correct_n,
                "n_flips_total": n_flips_total,
                "n_flips_used": 0,
                "layers_path": path_used,
            },
            "scan_rows": scan_rows,
            "flip_rows": [],
        }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"[Done] No flips found. Wrote {args.out_json}")
        return

    # Nonshared control basis (k dims), chosen as random orthogonal complement to Qs
    Q_nonshared = sample_random_orthonormal_complement(Qs, k=k, seed=args.seed + 2024)

    # Pre-capture step0 donor shared vectors for time-shuffled control (only for used flips)
    flip_donor_shared_step0: List[torch.Tensor] = []
    for ex in flip_examples_used:
        cap0 = DecodeStepHiddenCaptureHook(capture_steps=[0])
        _ = forced_choice_decode_aligned(
            model, tok, ex.prompt,
            candidate_labels, candidate_texts, ex.gold.strip().upper(),
            layer_module=layer_module,
            capture_hook=cap0,
            add_special_tokens_prompt=bool(args.add_special_tokens_prompt),
        )
        h0 = cap0.hidden_by_step.get(0, None)
        if h0 is None:
            raise RuntimeError("Failed to capture step0 hidden state. Check hook/layer compatibility.")
        flip_donor_shared_step0.append(project_cpu(h0, Qs))

    # --- diagnostics (print once) ---
    flip_golds = [ex.gold.strip().upper() for ex in flip_examples_used]
    print("[Flip gold distribution]", Counter(flip_golds))

    # # donor cosine sim (print once)
    # def _cos(u: torch.Tensor, v: torch.Tensor, eps: float = 1e-12) -> float:
    #     uu = float(torch.linalg.norm(u) + eps)
    #     vv = float(torch.linalg.norm(v) + eps)
    #     return float((u @ v.T).item() / (uu * vv))

    # cos_vals = []
    # for i in range(len(flip_donor_shared_step0)):
    #     for j in range(i + 1, len(flip_donor_shared_step0)):
    #         cos_vals.append(_cos(flip_donor_shared_step0[i], flip_donor_shared_step0[j]))
    # if cos_vals:
    #     cos_vals = np.array(cos_vals, dtype=np.float32)
    #     print(f"[Donor cos sim] mean={cos_vals.mean():.3f}  median={np.median(cos_vals):.3f}  p90={np.quantile(cos_vals,0.9):.3f}")

    # donor pools for label-mismatch control
    donors_by_gold = defaultdict(list)
    for g, p in zip(flip_golds, flip_donor_shared_step0):
        donors_by_gold[g].append(p)
    all_donors = list(flip_donor_shared_step0)

    # Patching runs on flips_used
    flip_rows: List[Dict[str, Any]] = []

    def _max_abs_score_diff(a: FCResult, b: FCResult) -> float:
        m = 0.0
        for lab in candidate_labels:
            m = max(m, abs(float(a.scores.get(lab, 0.0)) - float(b.scores.get(lab, 0.0))))
        return m

    for idx, ex in enumerate(flip_examples_used):
        prompt = ex.prompt
        gold = ex.gold.strip().upper()

        base = forced_choice_decode_aligned(
            model, tok, prompt,
            candidate_labels, candidate_texts, gold,
            layer_module=layer_module,
            add_special_tokens_prompt=bool(args.add_special_tokens_prompt),
        )
        ablt = forced_choice_decode_aligned(
            model, tok, prompt,
            candidate_labels, candidate_texts, gold,
            layer_module=layer_module,
            removal_hook=loto8.LastTokenRemovalHook(Qs, alpha=1.0, stats=loto8.HookStats("remove_shared")),
            add_special_tokens_prompt=bool(args.add_special_tokens_prompt),
        )

        # Capture baseline hidden states for all decode steps in full_steps
        cap_all = DecodeStepHiddenCaptureHook(capture_steps=full_steps)
        _ = forced_choice_decode_aligned(
            model, tok, prompt,
            candidate_labels, candidate_texts, gold,
            layer_module=layer_module,
            capture_hook=cap_all,
            add_special_tokens_prompt=bool(args.add_special_tokens_prompt),
        )

        if 0 not in cap_all.hidden_by_step:
            raise RuntimeError("Capture did not record step 0. This should never happen if decode-aligned is correct.")

        donor_shared = {t: project_cpu(cap_all.hidden_by_step[t], Qs) for t in cap_all.hidden_by_step.keys()}

        patched0 = forced_choice_decode_aligned(
            model, tok, prompt,
            candidate_labels, candidate_texts, gold,
            layer_module=layer_module,
            removal_hook=loto8.LastTokenRemovalHook(Qs, alpha=1.0, stats=loto8.HookStats("remove_shared")),
            patch_hook=SubspacePatchHook(Qs, donor_by_step=donor_shared, patch_steps=steps_0),
            add_special_tokens_prompt=bool(args.add_special_tokens_prompt),
        )
        patched01 = forced_choice_decode_aligned(
            model, tok, prompt,
            candidate_labels, candidate_texts, gold,
            layer_module=layer_module,
            removal_hook=loto8.LastTokenRemovalHook(Qs, alpha=1.0, stats=loto8.HookStats("remove_shared")),
            patch_hook=SubspacePatchHook(Qs, donor_by_step=donor_shared, patch_steps=steps_01),
            add_special_tokens_prompt=bool(args.add_special_tokens_prompt),
        )
        patched_full = forced_choice_decode_aligned(
            model, tok, prompt,
            candidate_labels, candidate_texts, gold,
            layer_module=layer_module,
            removal_hook=loto8.LastTokenRemovalHook(Qs, alpha=1.0, stats=loto8.HookStats("remove_shared")),
            patch_hook=SubspacePatchHook(Qs, donor_by_step=donor_shared, patch_steps=full_steps),
            add_special_tokens_prompt=bool(args.add_special_tokens_prompt),
        )

        # Extra debug: if full and 01 should match, quantify it
        diff_01_full = _max_abs_score_diff(patched01, patched_full)
        if idx == 0:
            print(f"[Debug] max|scores(patched_full)-scores(patched_01)| = {diff_01_full:.6e}")

        # Controls for patch window {0}
        p0 = donor_shared[0]  # [B=1,d] or [B=K,d] depending on internal batching, but step0 is [1,d]
        target_norms = torch.linalg.norm(p0, dim=1).cpu().float()

        p0_cpu = donor_shared[0].cpu().float()  # already [1,d] in span(Qs)

        ctrl_shared_perm = forced_choice_decode_aligned(
            model, tok, prompt,
            candidate_labels, candidate_texts, gold,
            layer_module=layer_module,
            removal_hook=loto8.LastTokenRemovalHook(Qs, alpha=1.0, stats=loto8.HookStats("remove_shared")),
            patch_hook=SubspacePatchHook(Qs, donor_by_step={0: shuffle_coeffs_in_subspace(p0_cpu, Qs, seed=args.seed + 9300 + idx, mode="permute")}, patch_steps={0}),
            add_special_tokens_prompt=bool(args.add_special_tokens_prompt),
        )

        ctrl_shared_signflip = forced_choice_decode_aligned(
            model, tok, prompt,
            candidate_labels, candidate_texts, gold,
            layer_module=layer_module,
            removal_hook=loto8.LastTokenRemovalHook(Qs, alpha=1.0, stats=loto8.HookStats("remove_shared")),
            patch_hook=SubspacePatchHook(Qs, donor_by_step={0: shuffle_coeffs_in_subspace(p0_cpu, Qs, seed=args.seed + 9400 + idx, mode="signflip")}, patch_steps={0}),
            add_special_tokens_prompt=bool(args.add_special_tokens_prompt),
        )


        # (1) Random-subspace energy-matched patch
        Q_rand = sample_random_orthonormal_basis(d=d, k=k, seed=args.seed + 9000 + idx)
        r0 = energy_matched_random_vector_in_subspace(Q_rand, target_norms=target_norms, seed=args.seed + 9100 + idx)
        ctrl_rand = forced_choice_decode_aligned(
            model, tok, prompt,
            candidate_labels, candidate_texts, gold,
            layer_module=layer_module,
            removal_hook=loto8.LastTokenRemovalHook(Qs, alpha=1.0, stats=loto8.HookStats("remove_shared")),
            patch_hook=SubspacePatchHook(Q_rand, donor_by_step={0: r0}, patch_steps={0}),
            add_special_tokens_prompt=bool(args.add_special_tokens_prompt),
        )

        # (2) Time-shuffled donor (from another example)
        donor_from = (idx + 1) % len(flip_examples_used)
        shuffled_p0 = flip_donor_shared_step0[donor_from]
        ctrl_shuffled = forced_choice_decode_aligned(
            model, tok, prompt,
            candidate_labels, candidate_texts, gold,
            layer_module=layer_module,
            removal_hook=loto8.LastTokenRemovalHook(Qs, alpha=1.0, stats=loto8.HookStats("remove_shared")),
            patch_hook=SubspacePatchHook(Qs, donor_by_step={0: shuffled_p0}, patch_steps={0}),
            add_special_tokens_prompt=bool(args.add_special_tokens_prompt),
        )


        def _cos(u: torch.Tensor, v: torch.Tensor, eps: float = 1e-12) -> float:
            # u,v: [1,d] CPU float32
            uu = float(torch.linalg.norm(u) + eps)
            vv = float(torch.linalg.norm(v) + eps)
            return float((u @ v.T).item() / (uu * vv))

        flip_golds = [ex.gold.strip().upper() for ex in flip_examples_used]
        print("[Flip gold distribution]", Counter(flip_golds))

        from collections import defaultdict
        donors_by_gold = defaultdict(list)
        for g, p in zip(flip_golds, flip_donor_shared_step0):
            donors_by_gold[g].append(p)
        all_donors = list(flip_donor_shared_step0)

        # (2b) label-mismatched donor
        rng = np.random.default_rng(args.seed + 9500 + idx)
        other_labels = [lab for lab in donors_by_gold.keys() if lab != gold and len(donors_by_gold[lab]) > 0]
        if other_labels:
            pick_lab = other_labels[int(rng.integers(0, len(other_labels)))]
            pick_list = donors_by_gold[pick_lab]
            shuffled_p0_mismatch = pick_list[int(rng.integers(0, len(pick_list)))]
        else:
            shuffled_p0_mismatch = all_donors[int(rng.integers(0, len(all_donors)))]

        ctrl_shared_mismatch = forced_choice_decode_aligned(
            model, tok, prompt,
            candidate_labels, candidate_texts, gold,
            layer_module=layer_module,
            removal_hook=loto8.LastTokenRemovalHook(Qs, alpha=1.0, stats=loto8.HookStats("remove_shared")),
            patch_hook=SubspacePatchHook(Qs, donor_by_step={0: shuffled_p0_mismatch}, patch_steps={0}),
            add_special_tokens_prompt=bool(args.add_special_tokens_prompt),
        )

        # donor cosine similarity stats
        cos_vals = []
        for i in range(len(flip_donor_shared_step0)):
            for j in range(i+1, len(flip_donor_shared_step0)):
                cos_vals.append(_cos(flip_donor_shared_step0[i], flip_donor_shared_step0[j]))
        if cos_vals:
            cos_vals = np.array(cos_vals, dtype=np.float32)
            print(f"[Donor cos sim] mean={cos_vals.mean():.3f}  median={np.median(cos_vals):.3f}  p90={np.quantile(cos_vals,0.9):.3f}")


        # (3) Patch nonshared instead of shared
        h0 = cap_all.hidden_by_step[0]  # [1,d] CPU
        p0_ns = project_cpu(h0, Q_nonshared)
        ctrl_nonshared = forced_choice_decode_aligned(
            model, tok, prompt,
            candidate_labels, candidate_texts, gold,
            layer_module=layer_module,
            removal_hook=loto8.LastTokenRemovalHook(Qs, alpha=1.0, stats=loto8.HookStats("remove_shared")),
            patch_hook=SubspacePatchHook(Q_nonshared, donor_by_step={0: p0_ns}, patch_steps={0}),
            add_special_tokens_prompt=bool(args.add_special_tokens_prompt),
        )

        # (4) Random vector IN shared subspace (energy-matched)
        r0_shared = energy_matched_random_vector_in_subspace(
            Qs, target_norms=target_norms, seed=args.seed + 9200 + idx
        )
        ctrl_shared_randvec = forced_choice_decode_aligned(
            model, tok, prompt,
            candidate_labels, candidate_texts, gold,
            layer_module=layer_module,
            removal_hook=loto8.LastTokenRemovalHook(Qs, alpha=1.0, stats=loto8.HookStats("remove_shared")),
            patch_hook=SubspacePatchHook(Qs, donor_by_step={0: r0_shared}, patch_steps={0}),
            add_special_tokens_prompt=bool(args.add_special_tokens_prompt),
        )

        flip_rows.append({
            "ex_id": ex.ex_id,
            "gold": gold,
            "baseline": base.__dict__,
            "ablated": ablt.__dict__,
            "patched_0": patched0.__dict__,
            "patched_01": patched01.__dict__,
            "patched_full": patched_full.__dict__,
            "debug_max_abs_diff_patched01_vs_full": float(diff_01_full),

            "control_rand_subspace": ctrl_rand.__dict__,
            "control_time_shuffled": ctrl_shuffled.__dict__,
            "control_shared_mismatch": ctrl_shared_mismatch.__dict__,
            "control_shared_perm": ctrl_shared_perm.__dict__,
            "control_shared_signflip": ctrl_shared_signflip.__dict__,
            "control_shared_randvec": ctrl_shared_randvec.__dict__,
            "control_patch_nonshared": ctrl_nonshared.__dict__,
        })


        print(
            f"[Flip {idx+1}/{len(flip_examples_used)}] ex_id={ex.ex_id} gold={gold} "
            f"base={base.pred_label}({base.correct}) ablt={ablt.pred_label}({ablt.correct}) "
            f"patch0={patched0.pred_label}({patched0.correct})"
        )

    summary = {
        "patched_0": summarize_rescue(flip_rows, "patched_0"),
        "patched_01": summarize_rescue(flip_rows, "patched_01"),
        "patched_full": summarize_rescue(flip_rows, "patched_full"),
        "control_rand_subspace": summarize_rescue(flip_rows, "control_rand_subspace"),
        "control_shared_randvec": summarize_rescue(flip_rows, "control_shared_randvec"),
        "control_time_shuffled": summarize_rescue(flip_rows, "control_time_shuffled"),
        "control_shared_mismatch": summarize_rescue(flip_rows, "control_shared_mismatch"),
        "control_shared_perm": summarize_rescue(flip_rows, "control_shared_perm"),
        "control_shared_signflip": summarize_rescue(flip_rows, "control_shared_signflip"),
        "control_patch_nonshared": summarize_rescue(flip_rows, "control_patch_nonshared"),
    }


    print("\n[Summary on flips_used]")
    for name, v in summary.items():
        print(f"  {name:>22s}: rescued={v['rescued']}/{v['n']} ({v['rescued_pct']:.1f}%)  mean Δmargin={v['mean_dmargin']:.3f}")

    out = {
        "meta": {
            "model": args.model,
            "device": args.device,
            "dtype": args.dtype,
            "layer": args.layer,
            "task": args.task,
            "eval_meta": eval_meta,
            "candidate_labels": candidate_labels,
            "candidate_text_style": args.candidate_text_style,
            "candidate_token_lens": {lab: int(l) for lab, l in zip(candidate_labels, cand_lens)},
            "max_candidate_token_len": int(max_len),
            "patch_windows": {
                "steps_0": sorted(list(steps_0)),
                "steps_01": sorted(list(steps_01)),
                "full_steps": sorted(list(full_steps)),
                "note": "If steps_01 == full_steps then patched_01 == patched_full by design.",
            },
            "add_special_tokens_prompt": bool(args.add_special_tokens_prompt),
            "seed": args.seed,
            "Qs_path": args.Qs_path or args.Qs_out,
            "Qs_shape": [int(d), int(k)],
            "n_scanned": n_scanned,
            "baseline_acc": base_acc,
            "baseline_correct_n": base_correct_n,
            "ablated_acc": ablt_acc,
            "ablated_correct_n": ablt_correct_n,
            "n_flips_total": n_flips_total,
            "n_flips_used": n_flips_used,
            "layers_path": path_used,
        },
        "summary_on_flips": summary,
        "scan_rows": scan_rows,
        "flip_rows": flip_rows,
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n[Done] Wrote {args.out_json}")


if __name__ == "__main__":
    main()
