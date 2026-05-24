"""Reasoning LOTO provenance experiment for downstream brittleness checks."""

import os
import sys
import re
import json
import math
import random
import argparse
from typing import Dict, List, Tuple, Optional, Any, DefaultDict
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(THIS_DIR, ".."))

from decodeshare.subspace import (  # noqa: E402
    get_model_layers,
    compute_cross_task_subspace,
    find_fully_shared_basis_improved,
)


try:
    from decodeshare.benchmark_dataloaders import *  # noqa: F401,F403
    from decodeshare.benchmark_dataloaders import (  # noqa: E402
        stable_int_seed as stable_int_seed_bdl,
        is_correct as is_correct_bool,
    )
except Exception as e:  # pragma: no cover
    raise RuntimeError("Failed to import decodeshare.benchmark_dataloaders.") from e

from decodeshare.decode_loto import (  # noqa: E402
    DecodeLastTokenActivationCollector,
    GenerationState,
    HookStats,
    LastTokenRemovalHook,
    LastTokenStagedRemovalHook,
    _subsample_rows_np,
    compute_shared_subspace_decode_aligned as _compute_shared_subspace_decode_aligned,
    bootstrap_ci_mean,
    energy_ratio_stats,
    fmt_acc,
    infer_component_variances,
    infer_hidden_dim,
    is_correct,
    json_default,
    max_offdiag,
    max_overlap,
    orthonormalize_np,
    paired_bootstrap_ci_diff,
    register_hooks_for_condition,
    remove_hooks,
    select_rand_indices,
    set_global_seed,
    signflip_permutation_test,
    summarize_paired,
    top_k_filtering,
    top_p_filtering,
)


stable_int_seed = stable_int_seed_bdl


def _norm_prefix_arg(x: Any) -> str:
    """Treat '0'/'none'/'null' as empty prefix (helps with CLI habits)."""
    if x is None:
        return ""
    s = str(x)
    if s.strip().lower() in {"0", "none", "null", "false"}:
        return ""
    return s


def _choose_next_token(
    logits: torch.Tensor,
    *,
    decoding: str,
    temperature: float,
    top_p: float,
    top_k: int,
    eos_token_id: int,
    ban_eos: bool,
) -> torch.Tensor:
    """Return next token ids [B,1]."""
    assert decoding in ["greedy", "sample"]
    if ban_eos:
        logits = logits.clone()
        logits[:, eos_token_id] = float("-inf")

    if decoding == "greedy":
        return torch.argmax(logits, dim=-1, keepdim=True)

    lt = logits / max(temperature, 1e-6)
    lt = top_k_filtering(lt, top_k=top_k)
    lt = top_p_filtering(lt, top_p=top_p)
    probs = torch.softmax(lt, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def _cache_advanced_prompt_boundary(model, ids: torch.Tensor, attn: torch.Tensor):
    """
    Compute (past, logits_next) such that the last prompt token is processed with seq_len==1.

    This matters because your interventions are decode-only (seq_len==1). Without this,
    the *first* next-token logits after the prompt would be produced by a prefill call
    and would NOT be affected by interventions.
    """
    if ids.ndim != 2:
        raise ValueError(f"ids must be 2D [B,T], got {ids.shape}")
    _, T = ids.shape
    if T == 0:
        raise ValueError("Empty prompt")
    if T == 1:
        out1 = model(input_ids=ids, attention_mask=attn, use_cache=True)
        return out1.past_key_values, out1.logits[:, -1, :]

    out0 = model(input_ids=ids[:, :-1], attention_mask=attn[:, :-1], use_cache=True)
    out1 = model(
        input_ids=ids[:, -1:],
        attention_mask=attn,
        use_cache=True,
        past_key_values=out0.past_key_values,
    )
    return out1.past_key_values, out1.logits[:, -1, :]


@torch.no_grad()
def collect_decode_last_token_states(
    model,
    tokenizer,
    prompts: List[str],
    collector: DecodeLastTokenActivationCollector,
    batch_size: int,
    max_new_tokens: int,
    decoding: str,
    temperature: float,
    top_p: float,
    top_k: int,
    max_prompt_len: int,
) -> None:
    assert decoding in ["greedy", "sample"]
    device = next(model.parameters()).device
    eos = tokenizer.eos_token_id
    model.eval()

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for i in tqdm(range(0, len(prompts), batch_size), desc="CollectDecode"):
        batch = prompts[i:i+batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_len,
        ).to(device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        B, _T0 = input_ids.shape

        unfinished = torch.ones(B, dtype=torch.bool, device=device)


        collector.set_capture(False, None)
        past, logits = _cache_advanced_prompt_boundary(model, input_ids, attention_mask)

        for _ in range(max_new_tokens):
            next_token = _choose_next_token(
                logits,
                decoding=decoding,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                eos_token_id=eos,
                ban_eos=False,
            )

            next_token = torch.where(
                unfinished.unsqueeze(-1),
                next_token,
                torch.full_like(next_token, eos),
            )

            unfinished = unfinished & (next_token.squeeze(-1) != eos)
            if not bool(unfinished.any().item()):
                break

            attention_mask = torch.cat(
                [attention_mask, torch.ones((B, 1), device=attention_mask.device, dtype=attention_mask.dtype)],
                dim=1,
            )

            collector.set_capture(True, unfinished)
            out = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                use_cache=True,
                past_key_values=past,
            )
            logits = out.logits[:, -1, :]
            past = out.past_key_values

        collector.set_capture(False, None)
def compute_shared_subspace_decode_aligned(*args, **kwargs):
    return _compute_shared_subspace_decode_aligned(
        *args, collect_fn=collect_decode_last_token_states, **kwargs
    )


@torch.no_grad()


@torch.no_grad()
def generate_continuations(
    model,
    tokenizer,
    prompts: List[str],
    decoding: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    device: str,
    batch_size: int,
    max_prompt_len: int,
    reasoning_token_threshold: int,
    state_setter: Optional[Any] = None,
    sample_seed: Optional[int] = None,
):
    assert decoding in ["greedy", "sample"]
    model.eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos = tokenizer.eos_token_id

    if decoding == "sample" and sample_seed is not None:
        torch.manual_seed(sample_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(sample_seed)

    continuations: List[str] = []
    eos_hit: List[int] = []
    new_tok: List[int] = []

    for i in tqdm(range(0, len(prompts), batch_size), desc=f"Generate({decoding})"):
        batch = prompts[i:i+batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_len,
        ).to(device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        B, T0 = input_ids.shape

        state = GenerationState(B, input_ids.device, reasoning_token_threshold)
        if state_setter is not None:
            state_setter(state)


        past, logits = _cache_advanced_prompt_boundary(model, input_ids, attention_mask)

        generated = input_ids

        for _ in range(max_new_tokens):
            next_token = _choose_next_token(
                logits,
                decoding=decoding,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                eos_token_id=eos,
                ban_eos=False,
            )

            next_token = torch.where(
                state.unfinished.unsqueeze(-1),
                next_token,
                torch.full_like(next_token, eos),
            )

            generated = torch.cat([generated, next_token], dim=1)

            state.step_update(next_token, eos_token_id=eos)
            if not bool(state.unfinished.any().item()):
                break

            attention_mask = torch.cat(
                [attention_mask, torch.ones((B, 1), device=attention_mask.device, dtype=attention_mask.dtype)],
                dim=1,
            )

            out = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                use_cache=True,
                past_key_values=past,
            )
            logits = out.logits[:, -1, :]
            past = out.past_key_values

        if state_setter is not None:
            state_setter(None)

        for b in range(B):
            L = int(state.gen_steps[b].item())
            cont_ids = generated[b, T0:T0+L]
            txt = tokenizer.decode(cont_ids, skip_special_tokens=True)
            continuations.append(txt)
            eos_hit.append(int(not bool(state.unfinished[b].item())))
            new_tok.append(L)

    return continuations, np.array(eos_hit, dtype=np.int32), np.array(new_tok, dtype=np.int32)


def evaluate_condition_generation(
    model,
    tokenizer,
    examples: List["Example"],
    Q_np: Optional[np.ndarray],
    condition: str,
    decoding: str,
    alpha: float,
    layer_indices: List[int],
    reasoning_token_threshold: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    device: str,
    batch_size: int,
    max_prompt_len: int,
    bootstrap_iters: int,
    ci_alpha: float,
    global_seed: int,
    sample_seed: Optional[int] = None,
) -> Dict[str, Any]:

    handles, state_setter, hook_stats = register_hooks_for_condition(
        model=model,
        layer_indices=layer_indices,
        Q_np=Q_np,
        condition=condition,
        alpha=alpha,
        reasoning_token_threshold=reasoning_token_threshold,
    )

    try:
        prompts = [ex.prompt for ex in examples]
        continuations, eos_hit, new_tok = generate_continuations(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            decoding=decoding,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            device=device,
            batch_size=batch_size,
            max_prompt_len=max_prompt_len,
            reasoning_token_threshold=reasoning_token_threshold,
            state_setter=state_setter,
            sample_seed=sample_seed,
        )

        correct, extracted = [], []
        for ex, cont in zip(examples, continuations):
            pred = parse_prediction(ex.dataset, cont)
            extracted.append(int(pred != ""))
            correct.append(is_correct(ex.dataset, pred, ex.gold))

        correct_arr = np.array(correct, dtype=np.float32)
        extracted_arr = np.array(extracted, dtype=np.float32)

        seed = stable_int_seed(global_seed, examples[0].dataset if examples else "na", condition, decoding, sample_seed or 0)
        acc, lo, hi = bootstrap_ci_mean(correct_arr, iters=bootstrap_iters, alpha=ci_alpha, seed=seed)

        return {
            "protocol": "generation",
            "condition": condition,
            "decoding": decoding,
            "sample_seed": sample_seed,
            "accuracy": float(acc),
            "ci_low": float(lo),
            "ci_high": float(hi),
            "correct": correct_arr.tolist(),
            "extraction_rate": float(extracted_arr.mean()),
            "eos_rate": float(np.mean(eos_hit)),
            "avg_new_tokens": float(np.mean(new_tok)),
            "hook_stats": [{"name": s.name, "decode_calls": s.decode_calls, "intervened": s.intervened} for s in hook_stats],
        }
    finally:
        remove_hooks(handles)


def candidate_strings(task: str) -> List[str]:
    """
    Candidate labels/strings for forced-choice scoring.
    Must match benchmark_dataloaders gold label conventions.
    """
    t = task.strip().lower()

    if t in ["commonsenseqa", "aqua"]:
        return list("ABCDE")
    if t in ["arc_challenge", "openbookqa", "logiqa"]:
        return list("ABCD")
    if t == "qasc":
        return list("ABCDEFGH")
    if t == "piqa":
        return ["A", "B"]
    if t == "boolq":

        return ["A", "B"]
    if t == "strategyqa":
        return ["Yes", "No"]
    return []


def _context_ends_with_whitespace(s: str) -> bool:
    return len(s) > 0 and s[-1].isspace()


def cand_token_ids(tokenizer, s: str, *, leading_space: bool) -> List[int]:
    """
    Encode candidate string into token ids, optionally with a leading space.

    This is a common source of accidental chance-level baselines:
      - If your prompt ends with 'Final answer:' (no space), you usually want leading_space=True.
      - If your prompt already ends with whitespace/newline, leading_space=False is safer.
    """
    text = (" " + s) if leading_space else s
    ids = tokenizer.encode(text, add_special_tokens=False)

    if not ids and leading_space:
        ids = tokenizer.encode(s, add_special_tokens=False)
    return ids


def _should_add_fc_prefix(
    *,
    prompt: str,
    warmup_tokens: int,
    prefix_mode: str,
    fc_answer_prefix: str,
) -> bool:
    """
    Decide whether to teacher-force answer prefix before scoring candidates.
    """
    if not fc_answer_prefix:
        return False
    assert prefix_mode in {"auto", "always", "never"}
    if prefix_mode == "never":
        return False
    if prefix_mode == "always":
        return True


    if warmup_tokens > 0:
        return True


    p = prompt.rstrip()
    ap = fc_answer_prefix.rstrip()
    return (ap != "") and (not p.endswith(ap))


@torch.no_grad()
def precompute_fc_warmup_tokens(
    model,
    tokenizer,
    prompts: List[str],
    *,
    warmup_tokens: int,
    batch_size: int,
    max_prompt_len: int,
    decoding: str,
    temperature: float,
    top_p: float,
    top_k: int,
    ban_eos: bool,
    seed: int,
) -> np.ndarray:
    """Generate warmup tokens under baseline (no intervention). Returns [N,W] int64."""
    assert warmup_tokens >= 0
    if warmup_tokens == 0:
        return np.zeros((len(prompts), 0), dtype=np.int64)

    device = next(model.parameters()).device
    eos = tokenizer.eos_token_id
    model.eval()

    if decoding == "sample":
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    out_tokens = np.zeros((len(prompts), warmup_tokens), dtype=np.int64)

    for i in tqdm(range(0, len(prompts), batch_size), desc=f"FCWarmupGen(W={warmup_tokens})"):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_prompt_len).to(device)
        ids = inputs["input_ids"]
        attn = inputs["attention_mask"]
        B = ids.shape[0]

        past, logits = _cache_advanced_prompt_boundary(model, ids, attn)

        toks = []
        for _ in range(warmup_tokens):
            next_tok = _choose_next_token(
                logits,
                decoding=decoding,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                eos_token_id=eos,
                ban_eos=ban_eos,
            )
            toks.append(next_tok)
            attn = torch.cat([attn, torch.ones((B, 1), device=device, dtype=attn.dtype)], dim=1)
            out = model(input_ids=next_tok, attention_mask=attn, use_cache=True, past_key_values=past)
            logits = out.logits[:, -1, :]
            past = out.past_key_values

        toks_mat = torch.cat(toks, dim=1)
        out_tokens[i : i + B, :] = toks_mat.detach().cpu().numpy().astype(np.int64, copy=False)

    return out_tokens


@torch.no_grad()
def evaluate_condition_forced_choice(
    model,
    tokenizer,
    examples: List["Example"],
    task_name: str,
    Q_np: Optional[np.ndarray],
    condition: str,
    *,
    alpha: float,
    layer_indices: List[int],
    reasoning_token_threshold: int,
    batch_size: int,
    max_prompt_len: int,
    bootstrap_iters: int,
    ci_alpha: float,
    global_seed: int,

    warmup_token_ids: Optional[np.ndarray],
    fc_warmup_tokens: int,
    fc_prefix_mode: str,
    fc_answer_prefix: str,
) -> Dict[str, Any]:
    """
    Forced-choice accuracy by sum logprob of candidate strings.

    This is decode-aligned and compatible with decode-only interventions.
    """
    device = next(model.parameters()).device
    model.eval()

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos = tokenizer.eos_token_id

    cands = candidate_strings(task_name)
    if len(cands) == 0:
        raise ValueError(f"Task '{task_name}' has no forced-choice candidates.")


    handles, state_setter, hook_stats = register_hooks_for_condition(
        model=model,
        layer_indices=layer_indices,
        Q_np=Q_np,
        condition=condition,
        alpha=alpha,
        reasoning_token_threshold=reasoning_token_threshold,
    )

    prompts = [ex.prompt for ex in examples]
    golds = [ex.gold for ex in examples]
    correct = np.zeros(len(examples), dtype=np.float32)


    avg_margin = []
    avg_best_lp = []

    try:
        for i in tqdm(range(0, len(examples), batch_size), desc=f"ForcedChoice({task_name}/{condition})"):
            batch_ex = examples[i : i + batch_size]
            batch_prompts = [ex.prompt for ex in batch_ex]
            batch_golds = [ex.gold for ex in batch_ex]

            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_prompt_len).to(device)
            ids = inputs["input_ids"]
            attn = inputs["attention_mask"]
            B = ids.shape[0]


            need_prefix = [
                _should_add_fc_prefix(
                    prompt=p,
                    warmup_tokens=fc_warmup_tokens,
                    prefix_mode=fc_prefix_mode,
                    fc_answer_prefix=fc_answer_prefix,
                )
                for p in batch_prompts
            ]

            warm_slice = None
            if warmup_token_ids is not None and fc_warmup_tokens > 0:
                warm_slice = warmup_token_ids[i : i + B]

            scores_full = torch.full((B, len(cands)), float("-inf"), device=device)


            for add_prefix in [False, True]:
                idxs = [j for j, flag in enumerate(need_prefix) if bool(flag) == add_prefix]
                if not idxs:
                    continue

                ids_g = ids[idxs]
                attn_g = attn[idxs]
                Bg = ids_g.shape[0]


                if state_setter is not None:
                    st = GenerationState(Bg, ids_g.device, reasoning_token_threshold)
                    state_setter(st)
                else:
                    st = None


                past_g, logits_g = _cache_advanced_prompt_boundary(model, ids_g, attn_g)


                if warm_slice is not None:
                    warm_g = torch.tensor(warm_slice[idxs], dtype=torch.long, device=device)
                    for t in range(warm_g.shape[1]):
                        tok_t = warm_g[:, t : t + 1]
                        attn_g = torch.cat(
                            [attn_g, torch.ones((Bg, 1), device=device, dtype=attn_g.dtype)],
                            dim=1,
                        )
                        out = model(input_ids=tok_t, attention_mask=attn_g, use_cache=True, past_key_values=past_g)
                        logits_g = out.logits[:, -1, :]
                        past_g = out.past_key_values
                        if st is not None:
                            st.step_update(tok_t, eos_token_id=eos)


                prefix_used_g = fc_answer_prefix if (add_prefix and fc_answer_prefix) else ""
                if prefix_used_g:
                    prefix_ids = tokenizer.encode(prefix_used_g, add_special_tokens=False)
                    for pid in prefix_ids:
                        inp = torch.full((Bg, 1), pid, dtype=torch.long, device=device)
                        attn_g = torch.cat(
                            [attn_g, torch.ones((Bg, 1), device=device, dtype=attn_g.dtype)],
                            dim=1,
                        )
                        out = model(input_ids=inp, attention_mask=attn_g, use_cache=True, past_key_values=past_g)
                        logits_g = out.logits[:, -1, :]
                        past_g = out.past_key_values
                        if st is not None:
                            st.step_update(inp, eos_token_id=eos)


                if prefix_used_g:
                    leading_space_g = not _context_ends_with_whitespace(prefix_used_g)
                else:
                    leading_space_g = not _context_ends_with_whitespace(batch_prompts[idxs[0]])

                cand_ids_list = [cand_token_ids(tokenizer, s, leading_space=leading_space_g) for s in cands]


                scores_g = torch.zeros(Bg, len(cands), device=device)
                for ci, cand_ids in enumerate(cand_ids_list):
                    if len(cand_ids) == 0:
                        scores_g[:, ci] = float("-inf")
                        continue

                    past_c = past_g
                    attn_c = attn_g
                    logits_c = logits_g


                    if state_setter is not None:
                        cand_state = st.clone()
                        state_setter(cand_state)
                    else:
                        cand_state = None

                    lp = torch.zeros(Bg, device=device)
                    for ti, tok_id in enumerate(cand_ids):
                        logp = torch.log_softmax(logits_c, dim=-1)
                        lp = lp + logp[:, tok_id]
                        if ti < len(cand_ids) - 1:
                            inp = torch.full((Bg, 1), tok_id, dtype=torch.long, device=device)
                            attn_c = torch.cat(
                                [attn_c, torch.ones((Bg, 1), device=device, dtype=attn_c.dtype)],
                                dim=1,
                            )
                            out = model(input_ids=inp, attention_mask=attn_c, use_cache=True, past_key_values=past_c)
                            logits_c = out.logits[:, -1, :]
                            past_c = out.past_key_values
                            if cand_state is not None:
                                cand_state.step_update(inp, eos_token_id=eos)

                    scores_g[:, ci] = lp

                scores_full[idxs, :] = scores_g


            pred_idx = torch.argmax(scores_full, dim=1).detach().cpu().numpy().tolist()
            preds = [cands[j] for j in pred_idx]


            top2 = torch.topk(scores_full, k=min(2, scores_full.shape[1]), dim=1).values.detach().cpu().numpy()
            if top2.shape[1] >= 2:
                avg_margin.extend((top2[:, 0] - top2[:, 1]).tolist())
            avg_best_lp.extend(top2[:, 0].tolist())

            for b, (pred, gold) in enumerate(zip(preds, batch_golds)):
                correct[i + b] = float(is_correct(task_name, pred, gold))


        if state_setter is not None:
            state_setter(None)

        correct_arr = correct.astype(np.float32, copy=False)
        seed = stable_int_seed(global_seed, task_name, "forced_choice", condition)
        acc, lo, hi = bootstrap_ci_mean(correct_arr, iters=bootstrap_iters, alpha=ci_alpha, seed=seed)

        return {
            "protocol": "forced_choice",
            "condition": condition,
            "accuracy": float(acc),
            "ci_low": float(lo),
            "ci_high": float(hi),
            "correct": correct_arr.tolist(),
            "fc": {
                "candidates": cands,
                "warmup_tokens": int(fc_warmup_tokens),
                "prefix_mode": fc_prefix_mode,
                "answer_prefix": fc_answer_prefix,
                "leading_space": bool(not _context_ends_with_whitespace((fc_answer_prefix if fc_answer_prefix else (prompts[0] if prompts else "")))),
                "avg_best_logprob": float(np.mean(avg_best_lp)) if avg_best_lp else float("nan"),
                "avg_margin": float(np.mean(avg_margin)) if avg_margin else float("nan"),
            },

            "extraction_rate": 1.0,
            "eos_rate": float("nan"),
            "avg_new_tokens": float("nan"),
            "hook_stats": [{"name": s.name, "decode_calls": s.decode_calls, "intervened": s.intervened} for s in hook_stats],
        }
    finally:
        remove_hooks(handles)


def load_model_and_tokenizer(model_name: str, device: str, model_dtype: str):
    dtype = torch.float32 if model_dtype == "fp32" else torch.float16
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)

    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = model.to(device)
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    return model, tok


def run_fold(
    *,
    fold_name: str,
    model,
    tokenizer,
    sub_by: Dict[str, List["Example"]],
    eval_by: Dict[str, List["Example"]],
    train_tasks: List[str],
    eval_tasks: List[str],
    layer_indices: List[int],
    args,
) -> Dict[str, Any]:


    prompts_by_task = {k: [ex.prompt for ex in sub_by[k]] for k in train_tasks}
    joint_subspace, shared_indices, extra, task_acts = compute_shared_subspace_decode_aligned(
        model=model,
        tokenizer=tokenizer,
        prompts_by_task=prompts_by_task,
        layer_indices=layer_indices,
        calib_decoding="greedy",
        calib_batch_size=args.batch_size,
        calib_max_new_tokens=args.calib_decode_max_new_tokens,
        per_task_max_states=args.per_task_max_states,
        max_prompt_len=args.max_prompt_len,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        global_seed=args.seed,
        variance_threshold=args.pca_var,
        min_dim=args.min_dim,
        max_dim=args.max_dim,
        tau=args.tau,
        m_shared=args.m_shared,
    )

    cross_dim = int(extra["cross_dim"])
    if len(shared_indices) == 0:
        raise RuntimeError(f"[{fold_name}] No shared basis found (shared_indices empty). Try relax tau or m_shared.")

    shared_basis = joint_subspace[:, shared_indices]
    Q_shared = orthonormalize_np(shared_basis)
    kS = Q_shared.shape[1]

    pooled_var = infer_component_variances(extra["task_contributions"], extra["tasks_used"], cross_dim)

    rand_indices = select_rand_indices(
        rand_type=args.rand_type,
        cross_dim=cross_dim,
        shared_indices=shared_indices,
        pooled_var=pooled_var,
        k=kS,
        seed=stable_int_seed(args.seed, fold_name, "rand_idx", args.rand_type),
    )
    rand_basis = joint_subspace[:, rand_indices]
    Q_rand = orthonormalize_np(rand_basis)


    sanity = {
        "cross_dim": cross_dim,
        "shared_basis_dim": int(kS),
        "rand_type": args.rand_type,
        "orthonorm_max_offdiag_shared": max_offdiag(Q_shared),
        "orthonorm_max_offdiag_rand": max_offdiag(Q_rand),
        "max_overlap_shared_rand": max_overlap(Q_shared, Q_rand),
    }


    layer = layer_indices[0]
    pool = []
    for t in extra["tasks_used"]:
        X = task_acts[t][layer]
        ss = stable_int_seed(args.seed, fold_name, "energy_sample", t)
        Xs = _subsample_rows_np(X, n_max=min(4000, X.shape[0]), seed=ss)
        pool.append(Xs)
    calib_states = np.concatenate(pool, axis=0)
    er_s = energy_ratio_stats(calib_states, Q_shared)
    er_r = energy_ratio_stats(calib_states, Q_rand)
    sanity["energy_ratio_shared"] = er_s
    sanity["energy_ratio_rand"] = er_r

    print(f"\n[{fold_name}] cross_dim={cross_dim} shared_dim={kS}")
    print(f"[{fold_name}][Sanity] Orthonormality max offdiag: shared={sanity['orthonorm_max_offdiag_shared']:.2e}, rand={sanity['orthonorm_max_offdiag_rand']:.2e}")
    print(f"[{fold_name}][Sanity] Max overlap |Q_shared^T Q_rand| = {sanity['max_overlap_shared_rand']:.2e}")
    print(f"[{fold_name}][Sanity] Energy ratio on calib decode states: shared mean={er_s['mean']:.4f}, rand mean={er_r['mean']:.4f}")


    CONDITIONS = ["baseline", "shared_full", "shared_staged", "rand_full", "rand_staged"]

    by_dataset: Dict[str, Any] = {}


    warmup_cache: Dict[str, np.ndarray] = {}

    for task_name in eval_tasks:
        eval_exs = eval_by[task_name]
        print("\n" + "-" * 80)
        print(f"[{fold_name}][Eval] Dataset={task_name} (n={len(eval_exs)})")
        print("-" * 80)


        has_cands = len(candidate_strings(task_name)) > 0
        use_fc = bool(args.use_forced_choice) and has_cands

        block: Dict[str, Any] = {"n": len(eval_exs), "protocol": ("forced_choice" if use_fc else "generation"), "runs": {}, "paired_tests": {}}

        if use_fc:

            if args.fc_warmup_tokens > 0:
                if task_name not in warmup_cache:
                    prompts = [ex.prompt for ex in eval_exs]
                    warm_ids = precompute_fc_warmup_tokens(
                        model=model,
                        tokenizer=tokenizer,
                        prompts=prompts,
                        warmup_tokens=args.fc_warmup_tokens,
                        batch_size=args.batch_size,
                        max_prompt_len=args.max_prompt_len,
                        decoding=args.fc_warmup_decoding,
                        temperature=args.fc_warmup_temperature,
                        top_p=args.fc_warmup_top_p,
                        top_k=args.fc_warmup_top_k,
                        ban_eos=bool(args.fc_warmup_ban_eos),
                        seed=stable_int_seed(args.seed, args.fc_warmup_seed, fold_name, task_name, "fc_warmup"),
                    )
                    warmup_cache[task_name] = warm_ids
                    if args.fc_debug_print and warm_ids.shape[0] > 0:
                        demo = tokenizer.decode(warm_ids[0].tolist(), skip_special_tokens=True)
                        print(f"[{fold_name}][FC Warmup] {task_name}: warmup_ids shape={warm_ids.shape}; demo text[:120]={demo[:120]!r}")
            else:
                warmup_cache[task_name] = np.zeros((len(eval_exs), 0), dtype=np.int64)

            warm_ids = warmup_cache.get(task_name, None)


            for cond in CONDITIONS:
                if cond == "baseline":
                    Q = None
                    cond_name = "baseline"
                    mode = "baseline"
                elif cond == "shared_full":
                    Q = Q_shared
                    cond_name = "full"
                    mode = "shared_full"
                elif cond == "shared_staged":
                    Q = Q_shared
                    cond_name = "staged"
                    mode = "shared_staged"
                elif cond == "rand_full":
                    Q = Q_rand
                    cond_name = "full"
                    mode = "rand_full"
                elif cond == "rand_staged":
                    Q = Q_rand
                    cond_name = "staged"
                    mode = "rand_staged"
                else:
                    raise ValueError(cond)

                run = evaluate_condition_forced_choice(
                    model=model,
                    tokenizer=tokenizer,
                    examples=eval_exs,
                    task_name=task_name,
                    Q_np=Q,
                    condition=cond_name,
                    alpha=args.alpha_remove,
                    layer_indices=layer_indices,
                    reasoning_token_threshold=args.reasoning_tokens,
                    batch_size=args.batch_size,
                    max_prompt_len=args.max_prompt_len,
                    bootstrap_iters=args.bootstrap_iters,
                    ci_alpha=args.ci_alpha,
                    global_seed=args.seed,
                    warmup_token_ids=warm_ids,
                    fc_warmup_tokens=args.fc_warmup_tokens,
                    fc_prefix_mode=args.fc_prefix_mode,
                    fc_answer_prefix=args.fc_answer_prefix,
                )
                block["runs"][f"forced_choice/{mode}"] = run
                fc_meta = run.get("fc", {})
                print(
                    f"[{fold_name}][{task_name}] forced_choice/{mode}: "
                    f"acc={fmt_acc(run['accuracy'], run['ci_low'], run['ci_high'])} "
                    f"(W={fc_meta.get('warmup_tokens', 0)}, prefix_mode={fc_meta.get('prefix_mode')}, "
                    f"avg_margin={fc_meta.get('avg_margin', float('nan')):.3f})"
                )


            base = np.array(block["runs"]["forced_choice/baseline"]["correct"], dtype=np.float32)
            shared_full = np.array(block["runs"]["forced_choice/shared_full"]["correct"], dtype=np.float32)
            rand_full = np.array(block["runs"]["forced_choice/rand_full"]["correct"], dtype=np.float32)
            seed0 = stable_int_seed(args.seed, fold_name, task_name, "forced_choice", "paired")

            block["paired_tests"]["forced_choice"] = {
                "shared_full_vs_baseline": summarize_paired(
                    base, shared_full,
                    label=f"{fold_name}:{task_name}:forced_choice:shared_full_vs_baseline",
                    bootstrap_iters=args.bootstrap_iters,
                    perm_iters=args.perm_iters,
                    alpha=args.ci_alpha,
                    seed=seed0 + 1,
                ),
                "rand_full_vs_baseline": summarize_paired(
                    base, rand_full,
                    label=f"{fold_name}:{task_name}:forced_choice:rand_full_vs_baseline",
                    bootstrap_iters=args.bootstrap_iters,
                    perm_iters=args.perm_iters,
                    alpha=args.ci_alpha,
                    seed=seed0 + 2,
                ),
                "shared_full_vs_rand_full": summarize_paired(
                    rand_full, shared_full,
                    label=f"{fold_name}:{task_name}:forced_choice:shared_full_vs_rand_full",
                    bootstrap_iters=args.bootstrap_iters,
                    perm_iters=args.perm_iters,
                    alpha=args.ci_alpha,
                    seed=seed0 + 3,
                ),
            }
            stat = block["paired_tests"]["forced_choice"]["shared_full_vs_baseline"]
            print(f"[{fold_name}][Stats] {task_name} (forced_choice) shared_full_vs_baseline: "
                  f"Delta={stat['mean_diff']:+.3f} CI[{stat['ci_low']:+.3f}, {stat['ci_high']:+.3f}] p={stat['p_value']:.3g}")

        else:

            DECODINGS = ["greedy"] + (["sample"] if args.do_sample else [])

            for decoding in DECODINGS:
                for cond in CONDITIONS:
                    if cond == "baseline":
                        Q = None
                        cond_name = "baseline"
                        mode = "baseline"
                    elif cond == "shared_full":
                        Q = Q_shared
                        cond_name = "full"
                        mode = "shared_full"
                    elif cond == "shared_staged":
                        Q = Q_shared
                        cond_name = "staged"
                        mode = "shared_staged"
                    elif cond == "rand_full":
                        Q = Q_rand
                        cond_name = "full"
                        mode = "rand_full"
                    elif cond == "rand_staged":
                        Q = Q_rand
                        cond_name = "staged"
                        mode = "rand_staged"
                    else:
                        raise ValueError(cond)

                    run = evaluate_condition_generation(
                        model=model,
                        tokenizer=tokenizer,
                        examples=eval_exs,
                        Q_np=Q,
                        condition=cond_name,
                        decoding=decoding,
                        alpha=args.alpha_remove,
                        layer_indices=layer_indices,
                        reasoning_token_threshold=args.reasoning_tokens,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        device=args.device,
                        batch_size=args.batch_size,
                        max_prompt_len=args.max_prompt_len,
                        bootstrap_iters=args.bootstrap_iters,
                        ci_alpha=args.ci_alpha,
                        global_seed=args.seed,
                        sample_seed=(args.sample_seed if decoding == "sample" else None),
                    )
                    block["runs"][f"{decoding}/{mode}"] = run
                    print(
                        f"[{fold_name}][{task_name}] {decoding}/{mode}: "
                        f"acc={fmt_acc(run['accuracy'], run['ci_low'], run['ci_high'])} "
                        f"extr={run['extraction_rate']*100:.1f}% eos={run['eos_rate']*100:.1f}% "
                        f"avg_new_tok={run['avg_new_tokens']:.1f}"
                    )


            for decoding in DECODINGS:
                base = np.array(block["runs"][f"{decoding}/baseline"]["correct"], dtype=np.float32)
                shared_full = np.array(block["runs"][f"{decoding}/shared_full"]["correct"], dtype=np.float32)
                rand_full = np.array(block["runs"][f"{decoding}/rand_full"]["correct"], dtype=np.float32)
                seed0 = stable_int_seed(args.seed, fold_name, task_name, decoding, "paired")

                block["paired_tests"][decoding] = {
                    "shared_full_vs_baseline": summarize_paired(
                        base, shared_full,
                        label=f"{fold_name}:{task_name}:{decoding}:shared_full_vs_baseline",
                        bootstrap_iters=args.bootstrap_iters,
                        perm_iters=args.perm_iters,
                        alpha=args.ci_alpha,
                        seed=seed0 + 1,
                    ),
                    "rand_full_vs_baseline": summarize_paired(
                        base, rand_full,
                        label=f"{fold_name}:{task_name}:{decoding}:rand_full_vs_baseline",
                        bootstrap_iters=args.bootstrap_iters,
                        perm_iters=args.perm_iters,
                        alpha=args.ci_alpha,
                        seed=seed0 + 2,
                    ),
                    "shared_full_vs_rand_full": summarize_paired(
                        rand_full, shared_full,
                        label=f"{fold_name}:{task_name}:{decoding}:shared_full_vs_rand_full",
                        bootstrap_iters=args.bootstrap_iters,
                        perm_iters=args.perm_iters,
                        alpha=args.ci_alpha,
                        seed=seed0 + 3,
                    ),
                }
                stat = block["paired_tests"][decoding]["shared_full_vs_baseline"]
                print(f"[{fold_name}][Stats] {task_name} ({decoding}) shared_full_vs_baseline: "
                      f"Delta={stat['mean_diff']:+.3f} CI[{stat['ci_low']:+.3f}, {stat['ci_high']:+.3f}] p={stat['p_value']:.3g}")

        by_dataset[task_name] = block

    return {
        "fold_name": fold_name,
        "train_tasks": train_tasks,
        "eval_tasks": eval_tasks,
        "basis": {
            "cross_dim": sanity["cross_dim"],
            "shared_k": sanity["shared_basis_dim"],
            "shared_indices_count": len(shared_indices),
            "rand_type": sanity["rand_type"],
            "sanity": sanity,
        },
        "by_dataset": by_dataset,
        "extra": extra,
    }


def render_loto_heldout_table(results: Dict[str, Any], decoding: str = "greedy") -> str:
    """
    Simple markdown table: each row is a held-out task, showing baseline vs shared_full on held-out only.

    If the held-out task was evaluated with forced-choice, pass decoding="forced_choice".
    """
    rows = []
    header = ["Held-out", "n", "Protocol", "Baseline", "Shared(full)", "Rand(full)", "Delta(shared-baseline)", "p(shared-baseline)"]
    for holdout, fold in results.get("folds", {}).items():
        block = fold["by_dataset"].get(holdout, None)
        if block is None:
            continue

        protocol = block.get("protocol", "generation")
        if protocol == "forced_choice":
            key_prefix = "forced_choice"
            b = block["runs"][f"{key_prefix}/baseline"]
            s = block["runs"][f"{key_prefix}/shared_full"]
            r = block["runs"][f"{key_prefix}/rand_full"]
            stat = block["paired_tests"][key_prefix]["shared_full_vs_baseline"]
        else:
            key_prefix = decoding
            b = block["runs"][f"{key_prefix}/baseline"]
            s = block["runs"][f"{key_prefix}/shared_full"]
            r = block["runs"][f"{key_prefix}/rand_full"]
            stat = block["paired_tests"][key_prefix]["shared_full_vs_baseline"]

        rows.append([
            holdout,
            str(block["n"]),
            protocol,
            fmt_acc(b["accuracy"], b["ci_low"], b["ci_high"]),
            fmt_acc(s["accuracy"], s["ci_low"], s["ci_high"]),
            fmt_acc(r["accuracy"], r["ci_low"], r["ci_high"]),
            f"{stat['mean_diff']*100:+.1f} [{stat['ci_low']*100:+.1f}, {stat['ci_high']*100:+.1f}]",
            f"{stat['p_value']:.3g}",
        ])

    cols = list(zip(*([header] + rows))) if rows else [header]
    widths = [max(len(str(x)) for x in col) for col in cols]

    def fmt_row(r):
        return "| " + " | ".join(str(x).ljust(w) for x, w in zip(r, widths)) + " |"

    lines = [fmt_row(header), "|-" + "-|-".join("-"*w for w in widths) + "-|"]
    for r in rows:
        lines.append(fmt_row(r))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model_dtype", type=str, default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--tasks", type=str, default="gsm8k,commonsenseqa,strategyqa,aqua,arc_challenge,openbookqa,qasc,logiqa")
    ap.add_argument("--mode", type=str, default="loto", choices=["all", "loto"])
    ap.add_argument("--loto_eval_mode", type=str, default="heldout", choices=["heldout", "all"])
    ap.add_argument("--loto_only", type=str, default="", help="Optional: only run this held-out task (e.g., 'gsm8k'). Empty means run all folds.")


    ap.add_argument("--n_subspace", type=int, default=128)
    ap.add_argument("--n_eval", type=int, default=256)


    ap.add_argument("--pca_var", type=float, default=0.95)
    ap.add_argument("--min_dim", type=int, default=1)
    ap.add_argument("--max_dim", type=int, default=4096)
    ap.add_argument("--tau", type=float, default=0.001, help="Relative threshold for shared component selection.")
    ap.add_argument("--m_shared", type=str, default="all", help="Sharedness requirement: 'all' or an int (>=2).")


    ap.add_argument("--calib_decode_max_new_tokens", type=int, default=128)
    ap.add_argument("--per_task_max_states", type=int, default=20000)


    ap.add_argument("--alpha_remove", type=float, default=1.0)
    ap.add_argument("--reasoning_tokens", type=int, default=128)
    ap.add_argument("--max_new_tokens", type=int, default=256)


    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_prompt_len", type=int, default=512)
    ap.add_argument("--do_sample", type=int, default=0, choices=[0, 1])


    ap.add_argument("--rand_type", type=str, default="joint_nonshared_varmatch",
                    choices=["joint_nonshared_uniform", "joint_nonshared_topk", "joint_nonshared_varmatch"])


    ap.add_argument("--template_randomization", type=int, default=1, choices=[0, 1])
    ap.add_argument("--template_seed", type=int, default=1234)
    ap.add_argument("--shuffle_choices", type=int, default=0, choices=[0, 1])
    ap.add_argument("--add_answer_prefix", type=int, default=0, choices=[0, 1])
    ap.add_argument("--answer_prefix", type=str, default="\nFinal answer:")


    ap.add_argument("--use_forced_choice", type=int, default=0, choices=[0, 1], help="If 1, use forced-choice for tasks with discrete candidates (MC/YesNo). gsm8k stays generation.")
    ap.add_argument("--fc_warmup_tokens", type=int, default=0, help="Teacher-forced warmup tokens before scoring candidates (baseline-generated, shared across conditions).")
    ap.add_argument("--fc_warmup_decoding", type=str, default="greedy", choices=["greedy", "sample"])
    ap.add_argument("--fc_warmup_seed", type=int, default=123)
    ap.add_argument("--fc_warmup_ban_eos", type=int, default=1, choices=[0, 1])
    ap.add_argument("--fc_warmup_temperature", type=float, default=0.7)
    ap.add_argument("--fc_warmup_top_p", type=float, default=0.9)
    ap.add_argument("--fc_warmup_top_k", type=int, default=0)
    ap.add_argument("--fc_prefix_mode", type=str, default="auto", choices=["auto", "always", "never"])
    ap.add_argument("--fc_answer_prefix", type=str, default="\nFinal answer:")
    ap.add_argument("--fc_debug_print", type=int, default=0, choices=[0, 1], help="If 1, print warmup demo text and other FC diagnostics.")


    ap.add_argument("--bootstrap_iters", type=int, default=5000)
    ap.add_argument("--perm_iters", type=int, default=10000)
    ap.add_argument("--ci_alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample_seed", type=int, default=12345)


    ap.add_argument("--out_json", type=str, default=os.path.join(THIS_DIR, "energy_balance_loto8_reasoning_results.json"))
    ap.add_argument("--out_md", type=str, default=os.path.join(THIS_DIR, "energy_balance_loto8_reasoning_summary.md"))

    args = ap.parse_args()
    set_global_seed(args.seed)


    args.answer_prefix = _norm_prefix_arg(args.answer_prefix)
    args.fc_answer_prefix = _norm_prefix_arg(args.fc_answer_prefix)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if len(tasks) < 2:
        raise RuntimeError("Need at least 2 tasks in --tasks.")

    args.do_sample = bool(args.do_sample)
    args.template_randomization = bool(args.template_randomization)
    args.shuffle_choices = bool(args.shuffle_choices)
    args.add_answer_prefix = bool(args.add_answer_prefix)
    args.use_forced_choice = bool(args.use_forced_choice)
    args.fc_debug_print = bool(args.fc_debug_print)


    if args.use_forced_choice and args.fc_warmup_tokens > 0 and args.fc_prefix_mode == "never" and args.fc_answer_prefix:
        print(
            "[Warn][ForcedChoice] fc_warmup_tokens>0 but fc_prefix_mode=never.\n"
            "  This often evaluates at a 'random mid-continuation' position and can yield chance-level baselines.\n"
            "  Recommended: --fc_prefix_mode auto (default) or always."
        )

    layer_indices = [args.layer]

    print(f"[Env] DEVICE={args.device}")
    print(f"[Env] MODEL={args.model} dtype={args.model_dtype}")
    print(f"[Env] layer_indices={layer_indices}")
    print(f"[Env] tasks={tasks}")
    print(f"[Env] mode={args.mode} loto_eval_mode={args.loto_eval_mode}")
    print(f"[Env] template_randomization={args.template_randomization} shuffle_choices={args.shuffle_choices} add_answer_prefix={args.add_answer_prefix}")
    print(f"[Env] forced_choice={args.use_forced_choice} fc_warmup_tokens={args.fc_warmup_tokens} fc_prefix_mode={args.fc_prefix_mode} fc_answer_prefix={args.fc_answer_prefix!r}")

    model, tokenizer = load_model_and_tokenizer(args.model, args.device, args.model_dtype)

    hidden_dim = infer_hidden_dim(model)
    if hidden_dim is None:
        print(f"[Warn] Could not infer hidden_dim (config_class={type(model.config)}). Continue anyway.")
    else:
        print(f"[Env] hidden_dim={hidden_dim}")


    sub_by, eval_by, meta_by = load_selected_tasks(
        tasks=tasks,
        n_subspace=args.n_subspace,
        n_eval=args.n_eval,
        seed=args.seed,
        template_seed=args.template_seed,
        template_randomization=args.template_randomization,
        shuffle_choices=args.shuffle_choices,
        add_answer_prefix=args.add_answer_prefix,
        answer_prefix=args.answer_prefix,
    )
    print("\n" + "=" * 80)
    print(f"[Data] Loaded tasks: {list(sub_by.keys())}")
    print(f"[Data] Meta: {json.dumps(meta_by, indent=2, ensure_ascii=False)}")
    print("=" * 80)

    results: Dict[str, Any] = {
        "config": {
            "model": args.model,
            "device": args.device,
            "model_dtype": args.model_dtype,
            "layer_indices": layer_indices,
            "tasks": tasks,
            "mode": args.mode,
            "loto_eval_mode": args.loto_eval_mode,
            "n_subspace": args.n_subspace,
            "n_eval": args.n_eval,
            "pca_var": args.pca_var,
            "tau": args.tau,
            "m_shared": args.m_shared,
            "per_task_max_states": args.per_task_max_states,
            "calib_decode_max_new_tokens": args.calib_decode_max_new_tokens,
            "reasoning_tokens": args.reasoning_tokens,
            "max_new_tokens": args.max_new_tokens,
            "alpha_remove": args.alpha_remove,
            "rand_type": args.rand_type,
            "template_randomization": args.template_randomization,
            "template_seed": args.template_seed,
            "shuffle_choices": args.shuffle_choices,
            "add_answer_prefix": args.add_answer_prefix,
            "answer_prefix": args.answer_prefix,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "batch_size": args.batch_size,
            "max_prompt_len": args.max_prompt_len,
            "bootstrap_iters": args.bootstrap_iters,
            "perm_iters": args.perm_iters,
            "ci_alpha": args.ci_alpha,
            "seed": args.seed,
            "sample_seed": args.sample_seed,
            "dataset_meta": meta_by,
            "forced_choice": {
                "use_forced_choice": args.use_forced_choice,
                "fc_warmup_tokens": args.fc_warmup_tokens,
                "fc_warmup_decoding": args.fc_warmup_decoding,
                "fc_warmup_seed": args.fc_warmup_seed,
                "fc_warmup_ban_eos": bool(args.fc_warmup_ban_eos),
                "fc_warmup_temperature": args.fc_warmup_temperature,
                "fc_warmup_top_p": args.fc_warmup_top_p,
                "fc_warmup_top_k": args.fc_warmup_top_k,
                "fc_prefix_mode": args.fc_prefix_mode,
                "fc_answer_prefix": args.fc_answer_prefix,
            },
        }
    }

    if args.mode == "all":
        fold = run_fold(
            fold_name="all_tasks",
            model=model,
            tokenizer=tokenizer,
            sub_by=sub_by,
            eval_by=eval_by,
            train_tasks=tasks,
            eval_tasks=tasks,
            layer_indices=layer_indices,
            args=args,
        )
        results["all_tasks"] = fold

    else:
        folds = {}
        for holdout in tasks:
            if args.loto_only and holdout != args.loto_only:
                continue
            train_tasks = [t for t in tasks if t != holdout]
            eval_tasks = [holdout] if args.loto_eval_mode == "heldout" else list(tasks)
            fold_name = f"loto_holdout={holdout}"
            print("\n" + "=" * 90)
            print(f"[LOTO] Running fold: holdout={holdout} train={train_tasks} eval={eval_tasks}")
            print("=" * 90)

            fold = run_fold(
                fold_name=fold_name,
                model=model,
                tokenizer=tokenizer,
                sub_by=sub_by,
                eval_by=eval_by,
                train_tasks=train_tasks,
                eval_tasks=eval_tasks,
                layer_indices=layer_indices,
                args=args,
            )
            folds[holdout] = fold


            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        results["folds"] = folds


    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=json_default)


    md_lines = []
    md_lines.append("# Energy-balance + LOTO(8) Summary\n")
    md_lines.append(f"- Model: `{args.model}` dtype={args.model_dtype} device={args.device}\n")
    md_lines.append(f"- Tasks: {tasks}\n")
    md_lines.append(f"- Mode: {args.mode}\n")
    md_lines.append(f"- Template randomization: {args.template_randomization} (seed={args.template_seed}), shuffle_choices={args.shuffle_choices}\n")
    md_lines.append(f"- Sharedness: pca_var={args.pca_var}, tau={args.tau}, m_shared={args.m_shared}\n")
    md_lines.append(f"- Calibration decode max_new_tokens={args.calib_decode_max_new_tokens}, per_task_max_states={args.per_task_max_states}\n")
    md_lines.append(f"- Evaluation: forced_choice={args.use_forced_choice} (MC tasks only)\n")
    md_lines.append("")

    if args.mode == "loto" and args.loto_eval_mode == "heldout" and "folds" in results:
        md_lines.append("## LOTO held-out performance\n")

        md_lines.append(render_loto_heldout_table(results, decoding="greedy"))
        md_lines.append("")

    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print("\n" + "=" * 80)
    print("[Done]")
    print(f"[Done] JSON: {args.out_json}")
    print(f"[Done] MD  : {args.out_md}")
    print("=" * 80)


if __name__ == "__main__":
    main()
