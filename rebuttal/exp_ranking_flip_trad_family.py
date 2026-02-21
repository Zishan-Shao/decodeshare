#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
exp_ranking_flip_trad_family.py

Ranking-flip with a TRAD-family (avoid a "strawman TRAD"):
  - TRAD-A: prefill-only (KV-cached generation; steering only on prefill forward)
  - TRAD-B: always-on (KV-cached generation; steering on both prefill + decode forwards)
  - TRAD-C: no-cache full recomputation (no KV cache; full forward each step)

Plus:
  - DECODE: decode-only (KV-cached generation; steering only on decode steps)
  - REAL: held-out templates (KV-cached generation; decode-only)

This script is intentionally separate from `exp_ranking_flip_steering.py` to avoid
changing any previously-run experiments. It can optionally reuse an existing JSON
from `exp_ranking_flip_steering.py` to avoid re-running DECODE/REAL.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

import exp_ranking_flip_steering as rf


def _parse_csv_ints(s: str) -> List[int]:
    s = str(s or "").strip()
    if not s:
        return []
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _dedup_keep_order(items: List[int]) -> List[int]:
    seen = set()
    out: List[int] = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _mean_by_key(dicts: List[Dict[str, float]]) -> Dict[str, float]:
    if not dicts:
        return {}
    keys = list(dicts[0].keys())
    out: Dict[str, float] = {}
    for k in keys:
        vals = [d[k] for d in dicts if k in d and not (isinstance(d[k], float) and np.isnan(d[k]))]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out


def _summarize_seed_scores(scores_by_seed: Dict[int, float]) -> Dict[str, Any]:
    seeds = sorted(scores_by_seed.keys())
    vals = np.array([scores_by_seed[s] for s in seeds], dtype=np.float64)
    n = int(vals.size)
    mean = float(np.mean(vals)) if n else float("nan")
    std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
    return {"n_seeds": n, "mean": mean, "std": std, "by_seed": {int(s): float(scores_by_seed[s]) for s in seeds}}


@torch.no_grad()
def generate_continuations_nocache(
    model,
    tokenizer,
    prompts: List[str],
    *,
    decoding: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    device: str,
    batch_size: int,
    max_prompt_len: int,
    sample_seed: Optional[int] = None,
    tqdm_inner: bool = True,
) -> Tuple[List[str], List[int], List[int]]:
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

    use_template = bool(getattr(tokenizer, "chat_template", None))
    it = range(0, len(prompts), batch_size)
    for i in tqdm(it, desc=f"GenerateNoCache({decoding})", disable=not bool(tqdm_inner), leave=False):
        batch = prompts[i : i + batch_size]
        batch = [rf.render_prompt(tokenizer, p, add_generation_prompt=True) for p in batch]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_len,
            add_special_tokens=not use_template,
        ).to(device)

        generated = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        B, T0 = generated.shape

        unfinished = torch.ones(B, dtype=torch.bool, device=generated.device)
        gen_steps = torch.zeros(B, dtype=torch.long, device=generated.device)

        for _ in range(int(max_new_tokens)):
            out = model(input_ids=generated, attention_mask=attention_mask, use_cache=False)
            logits = out.logits[:, -1, :]

            if decoding == "greedy":
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                lt = logits / max(float(temperature), 1e-6)
                lt = rf.top_k_filtering(lt, top_k=int(top_k))
                lt = rf.top_p_filtering(lt, top_p=float(top_p))
                probs = torch.softmax(lt, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            next_token = torch.where(
                unfinished.unsqueeze(-1),
                next_token,
                torch.full_like(next_token, eos),
            )

            generated = torch.cat([generated, next_token], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((B, 1), device=attention_mask.device, dtype=attention_mask.dtype)],
                dim=1,
            )

            active = unfinished.clone()
            gen_steps[active] += 1
            newly_finished = active & (next_token.squeeze(-1) == eos)
            unfinished[newly_finished] = False
            if not bool(unfinished.any().item()):
                break

        for b in range(B):
            L = int(gen_steps[b].item())
            cont_ids = generated[b, T0 : T0 + L]
            continuations.append(tokenizer.decode(cont_ids, skip_special_tokens=True))
            if L > 0:
                eos_hit.append(int(cont_ids[-1].item() == eos))
            else:
                eos_hit.append(0)
            new_tok.append(L)

    return continuations, eos_hit, new_tok


@torch.no_grad()
def evaluate_with_steering_nocache(
    *,
    model,
    tokenizer,
    examples: List[Any],
    decoding: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    device: str,
    batch_size: int,
    max_prompt_len: int,
    steering: Optional[rf.SteeringVector],
    phase_mode: str,  # "prefill" | "both" | "none"
    sample_seed: Optional[int] = None,
    tqdm_inner: bool = True,
) -> Dict[str, Any]:
    if phase_mode == "none" or steering is None:
        handles, _, hook_stats = [], None, []
    else:
        handles, _, hook_stats = rf.register_steering_hooks(
            model=model,
            layer_idx=int(steering.layer),
            v_np=steering.vec,
            alpha=float(steering.alpha),
            phase_mode=str(phase_mode),
            staged=False,
            reasoning_threshold=0,
        )

    try:
        prompts = [ex.prompt for ex in examples]
        continuations, eos_hit, new_tok = generate_continuations_nocache(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            decoding=decoding,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            device=str(device),
            batch_size=int(batch_size),
            max_prompt_len=int(max_prompt_len),
            sample_seed=sample_seed,
            tqdm_inner=bool(tqdm_inner),
        )

        if rf.parse_prediction is None or rf.is_correct_bool is None:
            raise RuntimeError(f"benchmark_dataloaders missing parse_prediction/is_correct: {rf._IMPORT_ERR}")

        correct: List[int] = []
        for ex, cont in zip(examples, continuations):
            pred = rf.parse_prediction(ex.dataset, cont)
            correct.append(int(rf.is_correct_bool(ex.dataset, pred, ex.gold)))
        correct_arr = np.array(correct, dtype=np.float32)
        acc = float(correct_arr.mean()) if len(correct_arr) else float("nan")
        return {
            "accuracy": acc,
            "n": int(len(correct_arr)),
            "eos_rate": float(np.mean(eos_hit)) if len(eos_hit) else float("nan"),
            "avg_new_tokens": float(np.mean(new_tok)) if len(new_tok) else float("nan"),
            "hook_stats": [
                {
                    "name": s.name,
                    "prefill_calls": int(s.prefill_calls),
                    "decode_calls": int(s.decode_calls),
                    "intervened": int(s.intervened),
                }
                for s in hook_stats
            ],
        }
    finally:
        rf.remove_hooks(handles)


def _coerce_seed_keyed(d: Any) -> Dict[int, Dict[str, float]]:
    if not isinstance(d, dict):
        return {}
    out: Dict[int, Dict[str, float]] = {}
    for sk, tv in d.items():
        try:
            s = int(sk)
        except Exception:
            continue
        if not isinstance(tv, dict):
            continue
        out[s] = {str(t): float(v) for t, v in tv.items()}
    return out


def _need_baseline(baseline: Dict[int, Dict[str, float]], *, seeds: List[int], tasks: List[str]) -> bool:
    return not all((s in baseline) and all(t in baseline[s] for t in tasks) for s in seeds)


def _decision_summary(
    *,
    vecs: Dict[str, Any],
    rank_keys: List[str],
    real_key: str,
    k_list: List[int],
) -> Dict[str, Any]:
    names = sorted(vecs.keys())
    real = np.array([float(vecs[n][real_key]) for n in names], dtype=np.float64)
    idx_real = np.argsort(-real)

    out: Dict[str, Any] = {"k_list": [int(k) for k in k_list], "rank_keys": list(rank_keys), "topk": {}}

    for rk in rank_keys:
        scores = np.array([float(vecs[n][rk]) for n in names], dtype=np.float64)
        idx = np.argsort(-scores)
        out["topk"][rk] = {}
        for k in k_list:
            k = int(min(int(k), len(names)))
            sel = idx[:k]
            sel_or = idx_real[:k]
            mean_real = float(real[sel].mean())
            mean_or = float(real[sel_or].mean())
            out["topk"][rk][str(k)] = {
                "mean_real": mean_real,
                "oracle_mean_real": mean_or,
                "regret": float(mean_or - mean_real),
                "npos": int(np.sum(real[sel] > 0)),
            }

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_json", type=str, default="", help="Optional existing JSON from exp_ranking_flip_steering.py to reuse DECODE/REAL/TRAD-A results.")

    ap.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model_dtype", type=str, default="fp32", choices=["fp32", "fp16"])

    ap.add_argument("--vectors_manifest", type=str, required=True)
    ap.add_argument("--max_vectors", type=int, default=0, help="0 means no limit.")
    ap.add_argument("--filter_regex", type=str, default="", help="Optional regex to filter vector names/concepts.")

    ap.add_argument("--tasks", type=str, default="commonsenseqa,arc_challenge,openbookqa,qasc,logiqa")
    ap.add_argument("--n_eval", type=int, default=128)

    ap.add_argument("--template_randomization", type=int, default=1, choices=[0, 1])
    ap.add_argument("--shuffle_choices", type=int, default=1, choices=[0, 1])
    ap.add_argument("--answer_prefix", type=str, default="\nFinal answer:")
    ap.add_argument("--add_answer_prefix", type=int, default=1, choices=[0, 1])

    ap.add_argument("--template_seed_rank", type=int, default=1234)
    ap.add_argument("--template_seed_real", type=int, default=5678)
    ap.add_argument("--template_seeds_rank", type=str, default="")
    ap.add_argument("--template_seeds_real", type=str, default="")

    ap.add_argument("--decoding", type=str, default="greedy", choices=["greedy", "sample"])
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--reasoning_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--nocache_batch_size", type=int, default=0, help="If >0: batch size for TRAD-C no-cache eval.")
    ap.add_argument("--max_prompt_len", type=int, default=512)
    ap.add_argument("--sample_seed", type=int, default=12345)

    ap.add_argument("--staged", type=int, default=1, choices=[0, 1])
    ap.add_argument("--decode_mode", type=str, default="decode", choices=["decode", "both"])
    ap.add_argument("--agg", type=str, default="mean", choices=["mean", "min", "median"])
    ap.add_argument("--do_trad_both", type=int, default=1, choices=[0, 1], help="Compute TRAD-B (prefill+decode, cached).")
    ap.add_argument("--do_trad_nocache", type=int, default=1, choices=[0, 1], help="Compute TRAD-C (no-cache full recomputation).")

    ap.add_argument("--k_list", type=str, default="1,5,10,20")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--out_json", type=str, default="ranking_flip_trad_family.json")
    ap.add_argument("--start_idx", type=int, default=0)
    ap.add_argument("--end_idx", type=int, default=-1)
    ap.add_argument("--resume", type=int, default=1, choices=[0, 1])
    ap.add_argument("--save_every", type=int, default=1)
    ap.add_argument("--save_every_seconds", type=int, default=0)
    ap.add_argument("--tqdm_outer", type=int, default=1, choices=[0, 1])
    ap.add_argument("--tqdm_inner", type=int, default=1, choices=[0, 1])
    args = ap.parse_args()

    rf.set_global_seed(int(args.seed))
    rf.TQDM_OUTER = bool(int(args.tqdm_outer))
    rf.TQDM_INNER = bool(int(args.tqdm_inner))

    out_path = os.path.expanduser(str(args.out_json))
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    # Load existing outputs if resuming, else optionally seed from base_json.
    results: Dict[str, Any]
    existing = rf._load_json_if_exists(out_path) if bool(args.resume) else None
    if existing is not None:
        if not isinstance(existing, dict):
            raise RuntimeError(f"--resume=1 but out_json is not a JSON object: {out_path}")
        results = existing
        results["config_trad_family"] = vars(args)
        print(f"[Resume] Loaded existing out_json: {out_path}")
    elif args.base_json:
        base_path = os.path.expanduser(str(args.base_json))
        base_obj = rf._load_json_if_exists(base_path)
        if base_obj is None:
            raise FileNotFoundError(base_path)
        if not isinstance(base_obj, dict):
            raise RuntimeError(f"--base_json is not a JSON object: {base_path}")
        results = base_obj
        results["base_json"] = str(base_path)
        results["config_trad_family"] = vars(args)
        print(f"[Init] Seeded from base_json: {base_path}")
    else:
        results = {"config_trad_family": vars(args), "vectors": {}}

    tasks = [t.strip() for t in str(args.tasks).split(",") if t.strip()]
    if not tasks:
        raise ValueError("Empty --tasks")

    rank_seeds = _parse_csv_ints(args.template_seeds_rank) or [int(args.template_seed_rank)]
    real_seeds = _parse_csv_ints(args.template_seeds_real) or [int(args.template_seed_real)]
    rank_seeds = _dedup_keep_order(rank_seeds)
    real_seeds = _dedup_keep_order(real_seeds)
    results["template_seeds_rank"] = rank_seeds
    results["template_seeds_real"] = real_seeds

    max_vectors = None if int(args.max_vectors) <= 0 else int(args.max_vectors)
    vecs_all = rf.load_vectors_from_manifest(str(args.vectors_manifest), max_vectors=max_vectors)
    if args.filter_regex:
        pat = re.compile(str(args.filter_regex))
        vecs_all = [v for v in vecs_all if pat.search(v.name) or pat.search(v.concept)]
        if not vecs_all:
            raise RuntimeError("No vectors after --filter_regex")

    n_total = len(vecs_all)
    start_idx = max(int(args.start_idx), 0)
    end_idx = int(args.end_idx)
    if end_idx < 0 or end_idx > n_total:
        end_idx = n_total
    if start_idx > n_total:
        start_idx = n_total
    vecs = vecs_all[start_idx:end_idx]

    # Load model
    model, tokenizer = rf.load_model_and_tokenizer(str(args.model), str(args.device), str(args.model_dtype))

    # Load evaluation sets
    if rf.load_selected_tasks is None:
        raise RuntimeError(f"benchmark_dataloaders.load_selected_tasks missing: {rf._IMPORT_ERR}")

    def load_eval(template_seed: int) -> Dict[str, List[Any]]:
        _sub_by, eval_by, _meta = rf.load_selected_tasks(
            tasks=tasks,
            n_subspace=1,
            n_eval=int(args.n_eval),
            seed=int(args.seed),
            template_seed=int(template_seed),
            template_randomization=bool(args.template_randomization),
            shuffle_choices=bool(args.shuffle_choices),
            add_answer_prefix=bool(args.add_answer_prefix),
            answer_prefix=str(args.answer_prefix),
        )
        return eval_by

    print(f"[Data] Loading ranking eval sets (template_seeds={rank_seeds}) ...")
    eval_rank_by_seed: Dict[int, Dict[str, List[Any]]] = {s: load_eval(s) for s in rank_seeds}
    print(f"[Data] Loading REAL eval sets (template_seeds={real_seeds}) ...")
    eval_real_by_seed: Dict[int, Dict[str, List[Any]]] = {s: load_eval(s) for s in real_seeds}

    # Baselines (reuse if present)
    base_rank_cached_by_seed = _coerce_seed_keyed(results.get("baseline_rank_by_seed"))
    base_real_cached_by_seed = _coerce_seed_keyed(results.get("baseline_real_by_seed"))
    base_rank_nocache_by_seed = _coerce_seed_keyed(results.get("baseline_rank_nocache_by_seed"))

    staged = bool(int(args.staged))
    sample_seed = int(args.sample_seed) if str(args.decoding) == "sample" else None
    nocache_bs = int(args.nocache_batch_size) if int(args.nocache_batch_size) > 0 else int(args.batch_size)

    if _need_baseline(base_rank_cached_by_seed, seeds=rank_seeds, tasks=tasks):
        print(f"\n[Baseline] Cached baseline on ranking templates (n_seeds={len(rank_seeds)}) ...")
        for s in rank_seeds:
            base_rank_cached_by_seed.setdefault(s, {})
            for t in tasks:
                if t in base_rank_cached_by_seed[s]:
                    continue
                res = rf.evaluate_with_steering(
                    model=model,
                    tokenizer=tokenizer,
                    examples=eval_rank_by_seed[s][t],
                    decoding=str(args.decoding),
                    max_new_tokens=int(args.max_new_tokens),
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    top_k=int(args.top_k),
                    device=str(args.device),
                    batch_size=int(args.batch_size),
                    max_prompt_len=int(args.max_prompt_len),
                    reasoning_token_threshold=int(args.reasoning_tokens),
                    steering=None,
                    phase_mode="none",
                    staged=False,
                    global_seed=int(args.seed),
                    sample_seed=sample_seed,
                )
                base_rank_cached_by_seed[s][t] = float(res["accuracy"])

    results["baseline_rank_mean"] = {
        t: float(np.mean([base_rank_cached_by_seed[s][t] for s in rank_seeds])) for t in tasks
    }

    if _need_baseline(base_real_cached_by_seed, seeds=real_seeds, tasks=tasks):
        print(f"\n[Baseline] Cached baseline on REAL templates (n_seeds={len(real_seeds)}) ...")
        for s in real_seeds:
            base_real_cached_by_seed.setdefault(s, {})
            for t in tasks:
                if t in base_real_cached_by_seed[s]:
                    continue
                res = rf.evaluate_with_steering(
                    model=model,
                    tokenizer=tokenizer,
                    examples=eval_real_by_seed[s][t],
                    decoding=str(args.decoding),
                    max_new_tokens=int(args.max_new_tokens),
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    top_k=int(args.top_k),
                    device=str(args.device),
                    batch_size=int(args.batch_size),
                    max_prompt_len=int(args.max_prompt_len),
                    reasoning_token_threshold=int(args.reasoning_tokens),
                    steering=None,
                    phase_mode="none",
                    staged=False,
                    global_seed=int(args.seed),
                    sample_seed=sample_seed,
                )
                base_real_cached_by_seed[s][t] = float(res["accuracy"])

    results["baseline_real_mean"] = {
        t: float(np.mean([base_real_cached_by_seed[s][t] for s in real_seeds])) for t in tasks
    }

    if _need_baseline(base_rank_nocache_by_seed, seeds=rank_seeds, tasks=tasks):
        if bool(int(args.do_trad_nocache)):
            print(f"\n[Baseline] No-cache baseline on ranking templates (TRAD-C) (n_seeds={len(rank_seeds)}) ...")
            for s in rank_seeds:
                base_rank_nocache_by_seed.setdefault(s, {})
                for t in tasks:
                    if t in base_rank_nocache_by_seed[s]:
                        continue
                    res = evaluate_with_steering_nocache(
                        model=model,
                        tokenizer=tokenizer,
                        examples=eval_rank_by_seed[s][t],
                        decoding=str(args.decoding),
                        max_new_tokens=int(args.max_new_tokens),
                        temperature=float(args.temperature),
                        top_p=float(args.top_p),
                        top_k=int(args.top_k),
                        device=str(args.device),
                        batch_size=nocache_bs,
                        max_prompt_len=int(args.max_prompt_len),
                        steering=None,
                        phase_mode="none",
                        sample_seed=sample_seed,
                        tqdm_inner=bool(int(args.tqdm_inner)),
                    )
                    base_rank_nocache_by_seed[s][t] = float(res["accuracy"])

    results["baseline_rank_nocache_mean"] = {
        t: float(np.mean([base_rank_nocache_by_seed[s][t] for s in rank_seeds])) for t in tasks
        if not _need_baseline(base_rank_nocache_by_seed, seeds=rank_seeds, tasks=[t])
    }

    results["baseline_rank_by_seed"] = base_rank_cached_by_seed
    results["baseline_real_by_seed"] = base_real_cached_by_seed
    results["baseline_rank_nocache_by_seed"] = base_rank_nocache_by_seed
    results.setdefault("vectors", {})

    # Save a baseline-only checkpoint early.
    rf._atomic_json_dump(results, out_path)
    last_save_t = time.time()

    done_names = set(results["vectors"].keys()) if isinstance(results.get("vectors"), dict) else set()
    save_every = int(args.save_every)
    save_every_seconds = int(args.save_every_seconds) if int(args.save_every_seconds) > 0 else 0
    n_since_save = 0

    def _needs_any(sv: rf.SteeringVector) -> bool:
        r0 = results["vectors"].get(sv.name, {})
        if not isinstance(r0, dict):
            return True
        need = ("score_rank_trad" not in r0) or ("score_rank_decode" not in r0) or ("score_real" not in r0)
        if bool(int(args.do_trad_both)):
            need = need or ("score_rank_trad_both" not in r0)
        if bool(int(args.do_trad_nocache)):
            need = need or ("score_rank_trad_nocache" not in r0)
        return need

    vecs_todo = [sv for sv in vecs if _needs_any(sv)]

    vec_iter = vecs_todo
    if bool(int(args.tqdm_outer)):
        vec_iter = tqdm(vecs_todo, desc="Vectors(TRAD-family)", unit="vec")

    def _eval_rank_cached(*, phase_mode: str) -> Tuple[Dict[int, Dict[str, float]], Dict[str, float], Dict[str, Any]]:
        per_task_by_seed: Dict[int, Dict[str, float]] = {}
        score_by_seed: Dict[int, float] = {}
        for s in rank_seeds:
            d: Dict[str, float] = {}
            for t in tasks:
                res = rf.evaluate_with_steering(
                    model=model,
                    tokenizer=tokenizer,
                    examples=eval_rank_by_seed[s][t],
                    decoding=str(args.decoding),
                    max_new_tokens=int(args.max_new_tokens),
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    top_k=int(args.top_k),
                    device=str(args.device),
                    batch_size=int(args.batch_size),
                    max_prompt_len=int(args.max_prompt_len),
                    reasoning_token_threshold=int(args.reasoning_tokens),
                    steering=sv,
                    phase_mode=str(phase_mode),
                    staged=staged,
                    global_seed=int(args.seed),
                    sample_seed=sample_seed,
                )
                d[t] = float(res["accuracy"] - base_rank_cached_by_seed[s][t])
            per_task_by_seed[s] = d
            score_by_seed[s] = rf.agg_task_scores(d, agg=str(args.agg))
        per_task_mean = _mean_by_key(list(per_task_by_seed.values()))
        summ = _summarize_seed_scores(score_by_seed)
        return per_task_by_seed, per_task_mean, summ

    def _eval_real_cached() -> Tuple[Dict[int, Dict[str, float]], Dict[str, float], Dict[str, Any]]:
        per_task_by_seed: Dict[int, Dict[str, float]] = {}
        score_by_seed: Dict[int, float] = {}
        for s in real_seeds:
            d: Dict[str, float] = {}
            for t in tasks:
                res = rf.evaluate_with_steering(
                    model=model,
                    tokenizer=tokenizer,
                    examples=eval_real_by_seed[s][t],
                    decoding=str(args.decoding),
                    max_new_tokens=int(args.max_new_tokens),
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    top_k=int(args.top_k),
                    device=str(args.device),
                    batch_size=int(args.batch_size),
                    max_prompt_len=int(args.max_prompt_len),
                    reasoning_token_threshold=int(args.reasoning_tokens),
                    steering=sv,
                    phase_mode="decode",
                    staged=staged,
                    global_seed=int(args.seed),
                    sample_seed=sample_seed,
                )
                d[t] = float(res["accuracy"] - base_real_cached_by_seed[s][t])
            per_task_by_seed[s] = d
            score_by_seed[s] = rf.agg_task_scores(d, agg=str(args.agg))
        per_task_mean = _mean_by_key(list(per_task_by_seed.values()))
        summ = _summarize_seed_scores(score_by_seed)
        return per_task_by_seed, per_task_mean, summ

    for sv in vec_iter:
        rec = results["vectors"].get(sv.name, {})
        if not isinstance(rec, dict):
            rec = {}

        # Alias old TRAD-A name if present.
        if "score_rank_trad" in rec and "score_rank_trad_prefill" not in rec:
            rec["score_rank_trad_prefill"] = rec["score_rank_trad"]

        # TRAD-A (prefill-only, cached) a.k.a. the original TRAD in exp_ranking_flip_steering.py
        if "score_rank_trad" not in rec:
            per_task_by_seed, per_task_mean, summ = _eval_rank_cached(phase_mode="prefill")
            rec["delta_rank_trad_by_seed"] = per_task_by_seed
            rec["delta_rank_trad"] = per_task_mean
            rec["score_rank_trad"] = float(summ["mean"])
            rec["score_rank_trad_summary"] = summ
            rec["score_rank_trad_prefill"] = float(summ["mean"])

        # DECODE (decode-only, cached) on ranking templates
        if "score_rank_decode" not in rec:
            per_task_by_seed, per_task_mean, summ = _eval_rank_cached(phase_mode=str(args.decode_mode))
            rec["delta_rank_decode_by_seed"] = per_task_by_seed
            rec["delta_rank_decode"] = per_task_mean
            rec["score_rank_decode"] = float(summ["mean"])
            rec["score_rank_decode_summary"] = summ

        # REAL (held-out templates, decode-only, cached)
        if "score_real" not in rec:
            per_task_by_seed, per_task_mean, summ = _eval_real_cached()
            rec["delta_real_decode_by_seed"] = per_task_by_seed
            rec["delta_real_decode"] = per_task_mean
            rec["score_real"] = float(summ["mean"])
            rec["score_real_summary"] = summ

        # Compute TRAD-B (always-on, cached)
        if bool(int(args.do_trad_both)) and "score_rank_trad_both" not in rec:
            per_task_by_seed, per_task_mean, summ = _eval_rank_cached(phase_mode="both")
            rec["delta_rank_trad_both_by_seed"] = per_task_by_seed
            rec["delta_rank_trad_both"] = per_task_mean
            rec["score_rank_trad_both"] = float(summ["mean"])
            rec["score_rank_trad_both_summary"] = summ

        # Compute TRAD-C (no-cache full recomputation)
        if bool(int(args.do_trad_nocache)) and "score_rank_trad_nocache" not in rec:
            per_task_by_seed: Dict[int, Dict[str, float]] = {}
            score_by_seed: Dict[int, float] = {}
            for s in rank_seeds:
                d = {}
                for t in tasks:
                    res = evaluate_with_steering_nocache(
                        model=model,
                        tokenizer=tokenizer,
                        examples=eval_rank_by_seed[s][t],
                        decoding=str(args.decoding),
                        max_new_tokens=int(args.max_new_tokens),
                        temperature=float(args.temperature),
                        top_p=float(args.top_p),
                        top_k=int(args.top_k),
                        device=str(args.device),
                        batch_size=nocache_bs,
                        max_prompt_len=int(args.max_prompt_len),
                        steering=sv,
                        phase_mode="prefill",
                        sample_seed=sample_seed,
                        tqdm_inner=bool(int(args.tqdm_inner)),
                    )
                    d[t] = float(res["accuracy"] - base_rank_nocache_by_seed[s][t])
                per_task_by_seed[s] = d
                score_by_seed[s] = rf.agg_task_scores(d, agg=str(args.agg))

            per_task_mean = _mean_by_key(list(per_task_by_seed.values()))
            summ = _summarize_seed_scores(score_by_seed)
            rec["delta_rank_trad_nocache_by_seed"] = per_task_by_seed
            rec["delta_rank_trad_nocache"] = per_task_mean
            rec["score_rank_trad_nocache"] = float(summ["mean"])
            rec["score_rank_trad_nocache_summary"] = summ

        # Store back
        rec.setdefault("concept", sv.concept)
        rec.setdefault("layer", int(sv.layer))
        rec.setdefault("alpha", float(sv.alpha))
        results["vectors"][sv.name] = rec

        n_since_save += 1
        now_t = time.time()
        should_save = (save_every > 0 and n_since_save >= save_every) or (
            save_every_seconds > 0 and (now_t - last_save_t) >= float(save_every_seconds)
        )
        if should_save:
            rf._atomic_json_dump(results, out_path)
            n_since_save = 0
            last_save_t = now_t

    # Summaries / correlations across all available vectors in the output
    vecs_out = results.get("vectors", {})
    if not isinstance(vecs_out, dict) or not vecs_out:
        raise RuntimeError("No vectors in results; nothing to summarize.")

    # Ensure TRAD-A alias exists for correlation/decision code (works for base_json-only runs too).
    for n, r in list(vecs_out.items()):
        if isinstance(r, dict) and "score_rank_trad" in r and "score_rank_trad_prefill" not in r:
            r["score_rank_trad_prefill"] = float(r["score_rank_trad"])

    def _pair_spearman(key_a: str, key_b: str) -> Dict[str, Any]:
        a = []
        b = []
        for n in sorted(vecs_out.keys()):
            r = vecs_out[n]
            if not isinstance(r, dict):
                continue
            if key_a not in r or key_b not in r:
                continue
            a.append(float(r[key_a]))
            b.append(float(r[key_b]))
        a_np = np.array(a, dtype=np.float64)
        b_np = np.array(b, dtype=np.float64)
        return {"n": int(len(a)), "rho": float(rf.spearmanr(a_np, b_np))}

    results["correlations_trad_family"] = {
        "trad_prefill_vs_real": _pair_spearman("score_rank_trad_prefill", "score_real"),
        "decode_vs_real": _pair_spearman("score_rank_decode", "score_real"),
        "trad_prefill_vs_decode": _pair_spearman("score_rank_trad_prefill", "score_rank_decode"),
    }
    if bool(int(args.do_trad_both)):
        results["correlations_trad_family"]["trad_both_vs_real"] = _pair_spearman("score_rank_trad_both", "score_real")
        results["correlations_trad_family"]["trad_both_vs_decode"] = _pair_spearman("score_rank_trad_both", "score_rank_decode")
    if bool(int(args.do_trad_nocache)):
        results["correlations_trad_family"]["trad_nocache_vs_real"] = _pair_spearman("score_rank_trad_nocache", "score_real")
        results["correlations_trad_family"]["trad_nocache_vs_decode"] = _pair_spearman("score_rank_trad_nocache", "score_rank_decode")

    k_list = _parse_csv_ints(args.k_list) or [1, 5, 10, 20]
    # Decision summary computed on vectors that have REAL and the ranking key.
    rank_keys = ["score_rank_trad_prefill"]
    if bool(int(args.do_trad_both)):
        rank_keys.append("score_rank_trad_both")
    if bool(int(args.do_trad_nocache)):
        rank_keys.append("score_rank_trad_nocache")
    rank_keys.append("score_rank_decode")

    vecs_for_decision = {}
    for n, r in vecs_out.items():
        if not isinstance(r, dict) or "score_real" not in r:
            continue
        if any(k in r for k in rank_keys):
            vecs_for_decision[n] = r

    results["decision_trad_family"] = _decision_summary(
        vecs=vecs_for_decision,
        rank_keys=rank_keys,
        real_key="score_real",
        k_list=k_list,
    )

    rf._atomic_json_dump(results, out_path)

    c = results["correlations_trad_family"]
    print("\n" + "-" * 80)
    print("[TRAD-family] Spearman(ranking_signal, REAL)")
    print(f"TRAD-A prefill: rho={c['trad_prefill_vs_real']['rho']:.3f} (n={c['trad_prefill_vs_real']['n']})")
    if "trad_both_vs_real" in c:
        print(f"TRAD-B both:   rho={c['trad_both_vs_real']['rho']:.3f} (n={c['trad_both_vs_real']['n']})")
    if "trad_nocache_vs_real" in c:
        print(f"TRAD-C nocache:rho={c['trad_nocache_vs_real']['rho']:.3f} (n={c['trad_nocache_vs_real']['n']})")
    print(f"DECODE:        rho={c['decode_vs_real']['rho']:.3f} (n={c['decode_vs_real']['n']})")
    print("-" * 80)
    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()
