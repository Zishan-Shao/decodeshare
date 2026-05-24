"""H3 prefill/decode estimator and intervention grid experiment."""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm


from h3_decode_subspace_helpers import (
    load_model_and_tokenizer,
    compute_shared_subspace_decode_aligned,
    bootstrap_ci_mean,
    orthonormalize_np,
)


from decodeshare.benchmark_dataloaders import (
    Example,
    load_selected_tasks,
    is_correct,
)

from decodeshare.subspace import get_model_layers


CHOICE_LABELS: Dict[str, List[str]] = {
    "commonsenseqa": list("ABCDE"),
    "aqua": list("ABCDE"),
    "arc_challenge": list("ABCD"),
    "openbookqa": list("ABCD"),
    "qasc": list("ABCDEFGH"),
    "logiqa": list("ABCD"),
    "piqa": list("AB"),
    "strategyqa": ["YES", "NO"],
    "boolq": ["YES", "NO"],

}


def candidate_texts_for_task(task: str) -> Tuple[List[str], List[str]]:
    labels = CHOICE_LABELS[task]
    if task in ["strategyqa", "boolq"]:
        texts = [" yes", " no"]
        labels = ["YES", "NO"]
        return labels, texts
    texts = [f" {c}" for c in labels]
    return labels, texts


def split_at_answer_prefix(prompt: str, answer_prefix: str) -> Tuple[str, bool]:
    """If prompt already contains answer_prefix, remove the last occurrence."""
    if not answer_prefix:
        return prompt, False
    idx = prompt.rfind(answer_prefix)
    if idx == -1:
        return prompt, False
    return prompt[:idx], True


def principal_angles_deg(Qa: np.ndarray, Qb: np.ndarray) -> Dict[str, float]:
    if Qa.size == 0 or Qb.size == 0:
        return {"mean": float("nan"), "p50": float("nan"), "p95": float("nan")}
    M = Qa.T @ Qb
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    theta = np.degrees(np.arccos(s))
    return {
        "mean": float(np.mean(theta)),
        "p50": float(np.percentile(theta, 50)),
        "p95": float(np.percentile(theta, 95)),
    }


class PrefillLastTokenCollector:
    def __init__(self):
        self.states: List[torch.Tensor] = []

    def __call__(self, module: torch.nn.Module, inputs: Tuple[torch.Tensor, ...], output):

        h = _extract_hidden_tensor(output)
        if h is None or (not torch.is_tensor(h)) or h.ndim != 3:
            return
        self.states.append(h[:, -1, :].detach())

    def pop_all(self) -> torch.Tensor:
        if not self.states:
            return torch.empty((0, 0))
        x = torch.cat(self.states, dim=0)
        self.states.clear()
        return x


@torch.no_grad()
def collect_prefill_lasttoken_states(
    model: torch.nn.Module,
    tokenizer,
    prompts: List[str],
    *,
    device: str,
    layer_idx: int,
    max_prompt_len: int,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    layers, _ = get_model_layers(model)
    if layer_idx < 0 or layer_idx >= len(layers):
        raise ValueError(f"layer_idx out of range: {layer_idx} vs {len(layers)}")

    collector = PrefillLastTokenCollector()
    handle = layers[layer_idx].register_forward_hook(collector)
    try:
        out_states: List[np.ndarray] = []
        for i in tqdm(range(0, len(prompts), batch_size), desc="CalibPrefill"):
            batch = prompts[i:i + batch_size]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=max_prompt_len,
                padding=True,
            ).to(device)
            _ = model(
                input_ids=enc["input_ids"],
                attention_mask=enc.get("attention_mask", None),
                use_cache=False,
            )
            st = collector.pop_all()
            if st.numel() == 0:
                continue
            out_states.append(st.float().cpu().numpy())
        if not out_states:
            return np.zeros((0, 0), dtype=np.float32)
        return np.concatenate(out_states, axis=0).astype(np.float32, copy=False)
    finally:
        try:
            handle.remove()
        except Exception:
            pass


def pooled_shared_basis_from_task_mats(
    mats: Dict[str, np.ndarray],
    *,
    pca_var: float,
    min_dim: int,
    max_dim: int,
    tau: float,
    m_shared: str,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    rng = np.random.RandomState(seed)
    tasks = list(mats.keys())
    if not tasks:
        return np.zeros((0, 0), dtype=np.float32), {"k_pca": 0, "k_shared": 0}

    n_min = min(mats[t].shape[0] for t in tasks)
    d = mats[tasks[0]].shape[1]
    Xs = []
    X_task: Dict[str, np.ndarray] = {}
    for t in tasks:
        X = mats[t]
        if X.shape[1] != d:
            raise ValueError("Hidden dim mismatch across tasks")
        if X.shape[0] > n_min:
            idx = rng.choice(X.shape[0], size=n_min, replace=False)
            X = X[idx]
        mu = X.mean(axis=0, keepdims=True)
        Xc = (X - mu).astype(np.float32, copy=False)
        X_task[t] = Xc
        Xs.append(Xc)
    X_pool = np.concatenate(Xs, axis=0).astype(np.float32, copy=False)


    X_t = torch.from_numpy(X_pool).float()
    _, S, Vh = torch.linalg.svd(X_t, full_matrices=False)
    s2 = (S ** 2).cpu().numpy()
    total = float(np.sum(s2) + 1e-12)
    cumsum = np.cumsum(s2) / total

    k_pca = int(np.searchsorted(cumsum, pca_var) + 1)
    k_pca = max(k_pca, int(min_dim))
    k_pca = min(k_pca, int(max_dim), Vh.shape[0])
    Q = Vh[:k_pca, :].T.contiguous().cpu().numpy()

    r_by_task: Dict[str, np.ndarray] = {}
    for t in tasks:
        Z = X_task[t] @ Q
        v = np.var(Z, axis=0, ddof=0)
        Vtot = float(np.sum(v) + 1e-12)
        r_by_task[t] = v / Vtot

    if m_shared == "all":
        m_req = len(tasks)
    elif m_shared == "half":
        m_req = max(1, len(tasks) // 2)
    else:
        try:
            m_req = int(m_shared)
        except Exception:
            m_req = len(tasks)

    shared = []
    for i in range(k_pca):
        cnt = 0
        for t in tasks:
            if r_by_task[t][i] >= tau:
                cnt += 1
        if cnt >= m_req:
            shared.append(i)

    Qs = Q[:, shared] if shared else np.zeros((d, 0), dtype=np.float32)
    Qs = orthonormalize_np(Qs)

    extra = {
        "k_pca": int(k_pca),
        "k_shared": int(Qs.shape[1]),
        "shared_indices": [int(i) for i in shared],
        "m_req": int(m_req),
        "tasks": tasks,
    }
    return Qs, extra


def _extract_hidden_tensor(output) -> Optional[torch.Tensor]:
    """
    Robustly extract hidden states tensor [B,T,d] from module output.
    Supports Tensor, tuple/list with tensor first element.
    """
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and len(output) > 0 and torch.is_tensor(output[0]):
        return output[0]
    return None


def _replace_hidden_tensor(output, new_h: torch.Tensor):
    """Put new_h back into output, preserving tuple/list structure if needed."""
    if torch.is_tensor(output):
        return new_h
    if isinstance(output, tuple):
        if len(output) == 0:
            return output
        lst = list(output)
        lst[0] = new_h
        return tuple(lst)
    if isinstance(output, list):
        if len(output) == 0:
            return output
        output[0] = new_h
        return output
    return output


class LastTokenSubspaceRemover:
    """
    Remove subspace Q from the *last token only* of the activation output.
    Locus:
      - 'decode': apply only when seq_len == 1
      - 'prefill': apply only when seq_len > 1
    """
    def __init__(self, Q_np: np.ndarray, alpha: float, locus: str):
        assert locus in ("decode", "prefill")
        self.alpha = float(alpha)
        self.locus = locus
        self.Q_cpu = torch.from_numpy(Q_np).float().contiguous()
        self.Q_by_device: Dict[str, torch.Tensor] = {}

    def _get_Q(self, device: torch.device) -> torch.Tensor:
        key = str(device)
        if key not in self.Q_by_device:
            self.Q_by_device[key] = self.Q_cpu.to(device=device, non_blocking=True)
        return self.Q_by_device[key]

    def __call__(self, module: torch.nn.Module, inputs: Tuple[torch.Tensor, ...], output):
        h = _extract_hidden_tensor(output)
        if h is None or (not torch.is_tensor(h)) or h.ndim != 3:
            return output

        T = h.shape[1]
        if self.locus == "decode":
            if T != 1:
                return output
        else:
            if T <= 1:
                return output

        Q = self._get_Q(h.device)


        dtype = h.dtype
        v = h[:, -1, :]
        v32 = v.float()


        vQ = torch.matmul(v32, Q)
        proj = torch.matmul(vQ, Q.t())
        v_new = v32 - self.alpha * proj


        h_new = h.clone()
        h_new[:, -1, :] = v_new.to(dtype=dtype)

        return _replace_hidden_tensor(output, h_new)


def register_lasttoken_hooks(
    model: torch.nn.Module,
    *,
    layer_indices: List[int],
    Q_np: np.ndarray,
    alpha: float,
    locus: str,
) -> List[Any]:
    layers, _ = get_model_layers(model)
    handles = []
    remover = LastTokenSubspaceRemover(Q_np=Q_np, alpha=alpha, locus=locus)
    for li in layer_indices:
        if li < 0 or li >= len(layers):
            raise ValueError(f"layer_idx out of range: {li} vs {len(layers)}")
        h = layers[li].register_forward_hook(remover)
        handles.append(h)
    return handles


def remove_handles(handles: List[Any]) -> None:
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass


@torch.no_grad()
def score_multitok_candidate(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    cand_ids: List[int],
    *,
    device: str,
    max_prompt_len: int,
    answer_prefix: str,
    warmup_ids: List[int],
    cache_init: str,
) -> float:
    """
    Safe but slower multi-token candidate scoring: recompute base cache then roll candidate tokens.
    Uses the same cache_init mode as the main scorer.
    """
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_prompt_len).to(device)
    input_ids = enc["input_ids"]
    attn_mask = enc.get("attention_mask", None)
    if attn_mask is None:
        attn_mask = torch.ones_like(input_ids)

    B, T = input_ids.shape
    assert B == 1

    if cache_init == "prefill_full":
        out = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=True)
        logits = out.logits[:, -1, :]
        past = out.past_key_values
        cur_attn = attn_mask
    elif cache_init == "decode_boundary":
        if T >= 2:
            out0 = model(input_ids=input_ids[:, :-1], attention_mask=attn_mask[:, :-1], use_cache=True)
            past = out0.past_key_values
            out1 = model(input_ids=input_ids[:, -1:], attention_mask=attn_mask, past_key_values=past, use_cache=True)
            logits = out1.logits[:, -1, :]
            past = out1.past_key_values
            cur_attn = attn_mask
        else:
            out1 = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=True)
            logits = out1.logits[:, -1, :]
            past = out1.past_key_values
            cur_attn = attn_mask
    else:
        raise ValueError(f"Unknown cache_init={cache_init}")

    for tid in warmup_ids:
        tid_t = torch.tensor([[tid]], device=device, dtype=torch.long)
        cur_attn = torch.cat([cur_attn, torch.ones((1, 1), device=device, dtype=cur_attn.dtype)], dim=1)
        outw = model(input_ids=tid_t, attention_mask=cur_attn, past_key_values=past, use_cache=True)
        logits = outw.logits[:, -1, :]
        past = outw.past_key_values

    if answer_prefix:
        ap_ids = tokenizer(answer_prefix, add_special_tokens=False).input_ids
        for tid in ap_ids:
            tid_t = torch.tensor([[tid]], device=device, dtype=torch.long)
            cur_attn = torch.cat([cur_attn, torch.ones((1, 1), device=device, dtype=cur_attn.dtype)], dim=1)
            outa = model(input_ids=tid_t, attention_mask=cur_attn, past_key_values=past, use_cache=True)
            logits = outa.logits[:, -1, :]
            past = outa.past_key_values

    score = 0.0
    for j, tid in enumerate(cand_ids):
        lp = torch.log_softmax(logits.float(), dim=-1)[0, tid].item()
        score += float(lp)
        if j == len(cand_ids) - 1:
            break
        tid_t = torch.tensor([[tid]], device=device, dtype=torch.long)
        cur_attn = torch.cat([cur_attn, torch.ones((1, 1), device=device, dtype=cur_attn.dtype)], dim=1)
        outc = model(input_ids=tid_t, attention_mask=cur_attn, past_key_values=past, use_cache=True)
        logits = outc.logits[:, -1, :]
        past = outc.past_key_values
    return float(score)


@torch.no_grad()
def forced_choice_one(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    *,
    cand_texts: List[str],
    device: str,
    max_prompt_len: int,
    answer_prefix: str,
    warmup_ids: List[int],
    cache_init: str,
) -> List[float]:
    """Internal helper for this experiment."""
    model.eval()

    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_prompt_len).to(device)
    input_ids = enc["input_ids"]
    attn_mask = enc.get("attention_mask", None)
    if attn_mask is None:
        attn_mask = torch.ones_like(input_ids)

    B, T = input_ids.shape
    assert B == 1

    if cache_init == "prefill_full":
        out = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=True)
        logits = out.logits[:, -1, :]
        past = out.past_key_values
        cur_attn = attn_mask
    elif cache_init == "decode_boundary":
        if T >= 2:
            prefix_ids = input_ids[:, :-1]
            out0 = model(input_ids=prefix_ids, attention_mask=attn_mask[:, :-1], use_cache=True)
            past = out0.past_key_values
            last_id = input_ids[:, -1:]
            out1 = model(input_ids=last_id, attention_mask=attn_mask, past_key_values=past, use_cache=True)
            logits = out1.logits[:, -1, :]
            past = out1.past_key_values
            cur_attn = attn_mask
        else:
            out1 = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=True)
            logits = out1.logits[:, -1, :]
            past = out1.past_key_values
            cur_attn = attn_mask
    else:
        raise ValueError(f"Unknown cache_init={cache_init}")


    for tid in warmup_ids:
        tid_t = torch.tensor([[tid]], device=device, dtype=torch.long)
        cur_attn = torch.cat([cur_attn, torch.ones((1, 1), device=device, dtype=cur_attn.dtype)], dim=1)
        outw = model(input_ids=tid_t, attention_mask=cur_attn, past_key_values=past, use_cache=True)
        logits = outw.logits[:, -1, :]
        past = outw.past_key_values


    if answer_prefix:
        ap_ids = tokenizer(answer_prefix, add_special_tokens=False).input_ids
        for tid in ap_ids:
            tid_t = torch.tensor([[tid]], device=device, dtype=torch.long)
            cur_attn = torch.cat([cur_attn, torch.ones((1, 1), device=device, dtype=cur_attn.dtype)], dim=1)
            outa = model(input_ids=tid_t, attention_mask=cur_attn, past_key_values=past, use_cache=True)
            logits = outa.logits[:, -1, :]
            past = outa.past_key_values


    scores: List[float] = []
    logp = torch.log_softmax(logits.float(), dim=-1)[0]
    for cand in cand_texts:
        cand_ids = tokenizer(cand, add_special_tokens=False).input_ids
        if len(cand_ids) == 1:
            scores.append(float(logp[cand_ids[0]].item()))
        else:
            s = score_multitok_candidate(
                model, tokenizer, prompt, cand_ids,
                device=device, max_prompt_len=max_prompt_len,
                answer_prefix=answer_prefix, warmup_ids=warmup_ids,
                cache_init=cache_init
            )
            scores.append(float(s))
    return scores


@torch.no_grad()
def eval_forced_choice(
    model: torch.nn.Module,
    tokenizer,
    examples: List[Example],
    *,
    device: str,
    max_prompt_len: int,
    answer_prefix: str,
    warmup_ids: List[int],
    cache_init: str,
) -> Tuple[np.ndarray, float, float, float]:
    correct: List[float] = []
    for ex in tqdm(examples, desc=f"ForcedChoice(cache_init={cache_init})"):
        if ex.dataset not in CHOICE_LABELS:
            continue

        labels, cand_texts = candidate_texts_for_task(ex.dataset)

        core_prompt, _found = split_at_answer_prefix(ex.prompt, answer_prefix)

        scores = forced_choice_one(
            model, tokenizer, core_prompt,
            cand_texts=cand_texts,
            device=device,
            max_prompt_len=max_prompt_len,
            answer_prefix=answer_prefix,
            warmup_ids=warmup_ids,
            cache_init=cache_init,
        )
        pred = labels[int(np.argmax(np.asarray(scores, dtype=np.float64)))]
        correct.append(float(is_correct(ex.dataset, pred, ex.gold)))

    correct_arr = np.array(correct, dtype=np.float32)
    acc, lo, hi = bootstrap_ci_mean(correct_arr, iters=1000, alpha=0.05, seed=0)
    return correct_arr, float(acc), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--model_dtype", type=str, default="fp16", choices=["fp32", "fp16"])

    ap.add_argument("--tasks", type=str, default="gsm8k,commonsenseqa,strategyqa,piqa,arc_challenge,openbookqa,qasc,logiqa,boolq")
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--n_subspace", type=int, default=128)
    ap.add_argument("--n_eval", type=int, default=256)

    ap.add_argument("--calib_decode_max_new_tokens", type=int, default=512)
    ap.add_argument("--per_task_max_states", type=int, default=20000)

    ap.add_argument("--max_prompt_len", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=8)


    ap.add_argument("--pca_var", type=float, default=0.95)
    ap.add_argument("--min_dim", type=int, default=16)
    ap.add_argument("--max_dim", type=int, default=4096)
    ap.add_argument("--tau", type=float, default=1e-3)
    ap.add_argument("--m_shared", type=str, default="all")

    ap.add_argument("--answer_prefix", type=str, default="\nFinal answer:")
    ap.add_argument("--warmup_tokens", type=int, default=0)
    ap.add_argument("--warmup_phrase", type=str, default="Let's think step by step.\n")

    ap.add_argument("--template_randomization", type=int, default=1)
    ap.add_argument("--shuffle_choices", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)


    ap.add_argument("--alpha_remove", type=float, default=1.0)


    ap.add_argument("--run_prefill_intervene", type=int, default=1)
    ap.add_argument("--run_decode_intervene", type=int, default=1)
    ap.add_argument("--out_json", type=str, default="", help="Output JSON path. Defaults to the historical filename in the current directory.")

    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]


    model, tok = load_model_and_tokenizer(args.model, device=args.device, model_dtype=args.model_dtype)


    sub_by, eval_by, _meta = load_selected_tasks(
        tasks=tasks,
        n_subspace=args.n_subspace,
        n_eval=args.n_eval,
        seed=args.seed,
        template_seed=args.seed + 999,
        template_randomization=bool(args.template_randomization),
        shuffle_choices=bool(args.shuffle_choices),
        add_answer_prefix=False,
        answer_prefix=args.answer_prefix,
    )


    warmup_ids: List[int] = []
    if args.warmup_tokens > 0:
        base_ids = tok(args.warmup_phrase, add_special_tokens=False).input_ids
        if len(base_ids) == 0:
            base_ids = tok(" ", add_special_tokens=False).input_ids
        rep = (args.warmup_tokens + len(base_ids) - 1) // max(len(base_ids), 1)
        warmup_ids = (base_ids * rep)[: args.warmup_tokens]
    print(f"[Warmup] teacher_forced_fixed W={len(warmup_ids)} tokens, phrase='{args.warmup_phrase.strip()}'")


    prompts_by_task = {k: [ex.prompt for ex in sub_by[k]] for k in sub_by.keys()}
    joint_dec, shared_dec_idx, extra_dec, _ = compute_shared_subspace_decode_aligned(
        model=model,
        tokenizer=tok,
        prompts_by_task=prompts_by_task,
        layer_indices=[args.layer],
        calib_decoding="greedy",
        calib_batch_size=args.batch_size,
        calib_max_new_tokens=args.calib_decode_max_new_tokens,
        per_task_max_states=args.per_task_max_states,
        max_prompt_len=args.max_prompt_len,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        global_seed=args.seed,
        variance_threshold=args.pca_var,
        min_dim=args.min_dim,
        max_dim=args.max_dim,
        tau=args.tau,
        m_shared=args.m_shared,
    )
    Q_dec_full = orthonormalize_np(joint_dec[:, shared_dec_idx])
    print(f"[Shared-Decode] k_shared={Q_dec_full.shape[1]} (tau={args.tau}, m_shared={args.m_shared})")


    mats_pre: Dict[str, np.ndarray] = {}
    for t in tasks:
        prompts = [ex.prompt for ex in sub_by[t]]
        X = collect_prefill_lasttoken_states(
            model=model,
            tokenizer=tok,
            prompts=prompts,
            device=args.device,
            layer_idx=args.layer,
            max_prompt_len=args.max_prompt_len,
            batch_size=args.batch_size,
        )
        mats_pre[t] = X
        print(f"[PrefillStates] {t}: {tuple(X.shape)}")
    Q_pre_full, extra_pre = pooled_shared_basis_from_task_mats(
        mats_pre,
        pca_var=args.pca_var,
        min_dim=args.min_dim,
        max_dim=args.max_dim,
        tau=args.tau,
        m_shared=args.m_shared,
        seed=args.seed + 1234,
    )
    print(f"[Shared-Prefill] k_shared={Q_pre_full.shape[1]} (tau={args.tau}, m_shared={args.m_shared})")


    k = min(Q_dec_full.shape[1], Q_pre_full.shape[1])
    if k <= 0:
        raise RuntimeError("Matched k is 0; cannot run H3 grid.")
    Q_dec = Q_dec_full[:, :k]
    Q_pre = Q_pre_full[:, :k]
    Q_ctrl_dec = orthonormalize_np(np.random.RandomState(args.seed + 2026).randn(Q_dec.shape[0], k).astype(np.float32))
    Q_ctrl_pre = orthonormalize_np(np.random.RandomState(args.seed + 2027).randn(Q_dec.shape[0], k).astype(np.float32))

    ang = principal_angles_deg(Q_dec, Q_pre)
    print(f"[Match] k = {k}")
    print(f"[Angles] mean={ang['mean']:.2f} deg p50={ang['p50']:.2f} deg p95={ang['p95']:.2f} deg")


    for t in ["commonsenseqa", "arc_challenge", "piqa", "boolq"]:
        if t in CHOICE_LABELS:
            _lbl, cand = candidate_texts_for_task(t)
            lens = [len(tok(c, add_special_tokens=False).input_ids) for c in cand]
            print(f"[CandTok] {t}: {list(zip(cand, lens))}")


    def run_cond(
        examples: List[Example],
        Q: Optional[np.ndarray],
        *,
        name: str,
        intervene_locus: Optional[str],
        cache_init: str,
    ) -> Dict[str, Any]:
        handles: List[Any] = []
        try:
            if Q is not None and intervene_locus is not None:
                handles = register_lasttoken_hooks(
                    model=model,
                    layer_indices=[args.layer],
                    Q_np=Q,
                    alpha=args.alpha_remove,
                    locus=intervene_locus,
                )
            corr, acc, lo, hi = eval_forced_choice(
                model, tok, examples,
                device=args.device,
                max_prompt_len=args.max_prompt_len,
                answer_prefix=args.answer_prefix,
                warmup_ids=warmup_ids,
                cache_init=cache_init,
            )
            return {"name": name, "acc": acc, "ci_low": lo, "ci_high": hi, "correct": corr.tolist()}
        finally:
            remove_handles(handles)

    results: Dict[str, Any] = {
        "model": args.model,
        "layer": args.layer,
        "k_match": k,
        "angles_deg": ang,
        "alpha_remove": float(args.alpha_remove),
        "warmup_tokens": int(len(warmup_ids)),
        "tasks": {},
    }

    def pct(x: float) -> float:
        return 100.0 * x

    for t in tasks:
        if t not in CHOICE_LABELS:
            print(f"[Skip] {t}: not a discrete-choice task")
            continue

        exs = eval_by[t]
        print("\n" + "=" * 100)
        print(f"[H3-Grid v3 | 2x2 + controls] {t} (n={len(exs)}, W={len(warmup_ids)})")
        print("=" * 100)


        r_base_decproto = run_cond(
            exs, None,
            name="baseline(dec-proto)",
            intervene_locus=None,
            cache_init="decode_boundary",
        )
        r_base_preproto = run_cond(
            exs, None,
            name="baseline(pre-proto)",
            intervene_locus=None,
            cache_init="prefill_full",
        )

        print(f"  {r_base_decproto['name']:<26}: {pct(r_base_decproto['acc']):5.1f} [{pct(r_base_decproto['ci_low']):.1f},{pct(r_base_decproto['ci_high']):.1f}]")
        print(f"  {r_base_preproto['name']:<26}: {pct(r_base_preproto['acc']):5.1f} [{pct(r_base_preproto['ci_low']):.1f},{pct(r_base_preproto['ci_high']):.1f}]")


        decode_arm: Dict[str, Any] = {}
        if args.run_decode_intervene:
            r_dec_dec = run_cond(
                exs, Q_dec,
                name="Dec-est / Dec-int",
                intervene_locus="decode",
                cache_init="decode_boundary",
            )
            r_pre_dec = run_cond(
                exs, Q_pre,
                name="Pre-est / Dec-int",
                intervene_locus="decode",
                cache_init="decode_boundary",
            )
            r_ctl_dec = run_cond(
                exs, Q_ctrl_dec,
                name="Rand-ctl / Dec-int",
                intervene_locus="decode",
                cache_init="decode_boundary",
            )

            print("  --- decode-intervene ---")
            print(f"  {r_dec_dec['name']:<26}: {pct(r_dec_dec['acc']):5.1f} [{pct(r_dec_dec['ci_low']):.1f},{pct(r_dec_dec['ci_high']):.1f}]")
            print(f"  {r_pre_dec['name']:<26}: {pct(r_pre_dec['acc']):5.1f} [{pct(r_pre_dec['ci_low']):.1f},{pct(r_pre_dec['ci_high']):.1f}]")
            print(f"  {r_ctl_dec['name']:<26}: {pct(r_ctl_dec['acc']):5.1f} [{pct(r_ctl_dec['ci_low']):.1f},{pct(r_ctl_dec['ci_high']):.1f}]")

            decode_arm = {
                "dec_est_dec_int": r_dec_dec,
                "pre_est_dec_int": r_pre_dec,
                "rand_ctl_dec_int": r_ctl_dec,
            }


        prefill_arm: Dict[str, Any] = {}
        if args.run_prefill_intervene:
            r_dec_pre = run_cond(
                exs, Q_dec,
                name="Dec-est / Pre-int",
                intervene_locus="prefill",
                cache_init="prefill_full",
            )
            r_pre_pre = run_cond(
                exs, Q_pre,
                name="Pre-est / Pre-int",
                intervene_locus="prefill",
                cache_init="prefill_full",
            )
            r_ctl_pre = run_cond(
                exs, Q_ctrl_pre,
                name="Rand-ctl / Pre-int",
                intervene_locus="prefill",
                cache_init="prefill_full",
            )

            print("  --- prefill-intervene ---")
            print(f"  {r_dec_pre['name']:<26}: {pct(r_dec_pre['acc']):5.1f} [{pct(r_dec_pre['ci_low']):.1f},{pct(r_dec_pre['ci_high']):.1f}]")
            print(f"  {r_pre_pre['name']:<26}: {pct(r_pre_pre['acc']):5.1f} [{pct(r_pre_pre['ci_low']):.1f},{pct(r_pre_pre['ci_high']):.1f}]")
            print(f"  {r_ctl_pre['name']:<26}: {pct(r_ctl_pre['acc']):5.1f} [{pct(r_ctl_pre['ci_low']):.1f},{pct(r_ctl_pre['ci_high']):.1f}]")

            prefill_arm = {
                "dec_est_pre_int": r_dec_pre,
                "pre_est_pre_int": r_pre_pre,
                "rand_ctl_pre_int": r_ctl_pre,
            }

        results["tasks"][t] = {
            "baseline_dec_proto": r_base_decproto,
            "baseline_pre_proto": r_base_preproto,
            "decode_intervene": decode_arm,
            "prefill_intervene": prefill_arm,
        }

    out = args.out_json or f"h3_grid_v3_{args.model.replace('/','_')}_layer{args.layer}_k{k}_W{len(warmup_ids)}_seed{args.seed}.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[Done] wrote {out}")


if __name__ == "__main__":
    main()
