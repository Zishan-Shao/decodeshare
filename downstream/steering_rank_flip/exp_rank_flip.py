
# -*- coding: utf-8 -*-
"""
exp_rank_flip.py

Ranking-flip experiment for steering vectors under:
  (A) "Traditional" prefill-only intervention (apply steering only during the prefill forward),
  (B) "Decode protocol" decode-only intervention (apply steering only during KV-cached decode steps),
and compare both rankings against a "real" held-out-template decode-only evaluation.

This script is designed to plug into the same project layout as:
  disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py

It assumes you have:
  - benchmark_dataloaders.py (or package) providing:
      * Example dataclass with fields: prompt, gold, dataset (and optionally id)
      * load_selected_tasks(...)
      * parse_prediction(dataset, continuation) -> str
      * is_correct(dataset, pred, gold) -> bool/int
      * stable_int_seed(...) -> int
  - transformers, torch, numpy, tqdm

Steering vectors are loaded from a JSONL manifest, one JSON per line, e.g.:

{"name":"truthful_l10_seed0","concept":"truthful","layer":10,"alpha":1.0,"path":"vectors/truthful_l10_seed0.npy"}
{"name":"refusal_l15_seed123","concept":"refusal","layer":15,"alpha":0.8,"path":"vectors/refusal_l15_seed123.pt"}

Vector files:
  - .npy: np.ndarray shape [D]
  - .pt/.pth: torch.Tensor [D] OR dict with one of keys: ["vector","v","direction"]

Outputs:
  - JSON with per-vector scores and ranking correlations.
"""

import os
import sys
import re
import json
import math
import argparse
import hashlib
import tempfile
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

'''

python exp_rank_flip.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --device cuda \
  --vectors_manifest steering_vectors_example.jsonl \
  --tasks commonsenseqa,arc_challenge,openbookqa,qasc,logiqa \
  --n_eval 128 \
  --template_seed_rank 1234 \
  --template_seed_real 5678 \
  --trad_mode prefill \
  --decode_mode decode \
  --staged 1 \
  --reasoning_tokens 128 \
  --decoding greedy \
  --out_json ranking_flip_results.json

python exp_rank_flip.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --device cuda \
  --vectors_manifest steering_vectors_example.jsonl \
  --tasks commonsenseqa,arc_challenge,openbookqa,qasc,logiqa \
  --n_eval 128 \
  --template_seeds_rank 1234,2345,3456 \
  --template_seeds_real 4567,5678,6789 \
  --trad_mode prefill \
  --decode_mode decode \
  --staged 1 \
  --reasoning_tokens 128 \
  --decoding greedy \
  --out_json ranking_flip_results.json

'''

# -----------------------------
# Local imports (repo layout)
# -----------------------------
# This script lives in `downstream/steering_rank_flip/`; public releases keep
# benchmark_dataloaders.py with the experiment/downstream bundles. Make direct
# script execution work from either the repo root or this directory.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _candidate in [
    os.path.join(THIS_DIR, "..", "brittleness"),
    os.path.join(THIS_DIR, "..", "patch_back"),
    os.path.join(THIS_DIR, "..", "..", "experiments", "02_decode_ablation"),
]:
    _candidate = os.path.normpath(_candidate)
    if os.path.isfile(os.path.join(_candidate, "benchmark_dataloaders.py")) and _candidate not in sys.path:
        sys.path.append(_candidate)

TQDM_OUTER = False
TQDM_INNER = True


# -----------------------------
# Repro / stable seed
# -----------------------------
def stable_int_seed_fallback(*items: Any) -> int:
    s = "|".join(str(x) for x in items)
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _atomic_json_dump(obj: Any, out_path: str) -> None:
    out_path = os.path.expanduser(out_path)
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=os.path.basename(out_path) + ".", suffix=".tmp", dir=out_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, out_path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def _load_json_if_exists(path: str) -> Optional[Any]:
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _config_compatible(old_cfg: Dict[str, Any], new_cfg: Dict[str, Any]) -> Tuple[bool, List[str]]:
    ignore = {
        "out_json",
        "resume",
        "start_idx",
        "end_idx",
        "save_every",
        "save_every_seconds",
        "tqdm_outer",
        "tqdm_inner",
    }
    diffs: List[str] = []
    keys = sorted(set(old_cfg.keys()) | set(new_cfg.keys()))
    for k in keys:
        if k in ignore:
            continue
        if k not in old_cfg or k not in new_cfg:
            continue
        if old_cfg[k] != new_cfg[k]:
            diffs.append(k)
    return (len(diffs) == 0), diffs


# -----------------------------
# Optional project imports
# -----------------------------
try:
    from benchmark_dataloaders import (
        Example,
        load_selected_tasks,
        parse_prediction,
        is_correct as is_correct_bool,
        stable_int_seed as stable_int_seed_project,
    )
    stable_int_seed = stable_int_seed_project
except Exception as e:
    Example = None  # type: ignore
    load_selected_tasks = None  # type: ignore
    parse_prediction = None  # type: ignore
    is_correct_bool = None  # type: ignore
    stable_int_seed = stable_int_seed_fallback
    _IMPORT_ERR = e


def set_global_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_hidden_dim(model) -> Optional[int]:
    cfg = getattr(model, "config", None)
    for k in ("hidden_size", "n_embd", "dim", "d_model", "model_dim", "embed_dim"):
        v = getattr(cfg, k, None)
        if isinstance(v, int) and v > 0:
            return v
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is not None:
        for k in ("hidden_size", "n_embd", "dim", "d_model", "model_dim", "embed_dim"):
            v = getattr(text_cfg, k, None)
            if isinstance(v, int) and v > 0:
                return v
    try:
        emb = model.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight") and isinstance(emb.weight, torch.Tensor) and emb.weight.ndim == 2:
            return int(emb.weight.shape[1])
        if emb is not None and hasattr(emb, "embedding_dim"):
            return int(emb.embedding_dim)
    except Exception:
        pass
    return None


def get_model_layers(model) -> List[torch.nn.Module]:
    """
    Best-effort layer list extraction for common decoder-only HF models.
    This returns the list of transformer blocks whose forward() outputs hidden states.
    """
    # LLaMA / Qwen / Gemma / Mistral / etc.
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    # GPT-NeoX
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return list(model.gpt_neox.layers)
    # GPT-2 / GPT-J style
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    # Falcon
    if hasattr(model, "transformer") and hasattr(model.transformer, "blocks"):
        return list(model.transformer.blocks)
    # MPT
    if hasattr(model, "transformer") and hasattr(model.transformer, "blocks"):
        return list(model.transformer.blocks)
    raise RuntimeError(f"Cannot locate transformer layers for model class: {type(model)}")


# -----------------------------
# Prompt rendering (chat template safe)
# -----------------------------
def render_prompt(tokenizer, user_prompt: str, *, add_generation_prompt: bool = True, system_prompt: Optional[str] = None) -> str:
    tmpl = getattr(tokenizer, "chat_template", None)
    if not tmpl:
        return user_prompt
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    except Exception:
        # Some models don't support system role
        messages = [{"role": "user", "content": user_prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)


# -----------------------------
# Sampling utils
# -----------------------------
def top_p_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p <= 0.0 or top_p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumprobs = torch.cumsum(probs, dim=-1)
    mask = cumprobs > top_p
    mask[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
    filtered = torch.full_like(logits, float("-inf"))
    filtered.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)
    return filtered


def top_k_filtering(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k is None or top_k <= 0:
        return logits
    top_k = min(top_k, logits.size(-1))
    values, _ = torch.topk(logits, top_k, dim=-1)
    min_values = values[:, -1].unsqueeze(-1)
    return torch.where(logits < min_values, torch.full_like(logits, float("-inf")), logits)


# -----------------------------
# Steering vector spec + loader
# -----------------------------
@dataclass
class SteeringVector:
    name: str
    concept: str
    layer: int
    alpha: float
    vec: np.ndarray
    template_tag: Optional[str] = None


def _load_vector_file(path: str) -> np.ndarray:
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.endswith(".npy"):
        v = np.load(path)
        return np.asarray(v, dtype=np.float32).reshape(-1)
    if path.endswith(".pt") or path.endswith(".pth"):
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):
            for k in ["vector", "v", "direction", "dir"]:
                if k in obj:
                    obj = obj[k]
                    break
        if isinstance(obj, torch.Tensor):
            return obj.detach().float().cpu().numpy().reshape(-1)
        if isinstance(obj, np.ndarray):
            return obj.astype(np.float32).reshape(-1)
        raise TypeError(f"Unsupported .pt content type: {type(obj)}")
    raise ValueError(f"Unsupported vector file extension: {path}")


def load_vectors_from_manifest(manifest_path: str, *, max_vectors: Optional[int] = None) -> List[SteeringVector]:
    manifest_path = os.path.expanduser(manifest_path)
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(manifest_path)

    base_dir = os.path.dirname(os.path.abspath(manifest_path))

    vecs: List[SteeringVector] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            item = json.loads(line)
            name = str(item.get("name", ""))
            concept = str(item.get("concept", item.get("attr", "unknown")))
            layer = int(item["layer"])
            alpha = float(item.get("alpha", 1.0))
            path = os.path.expanduser(str(item["path"]))
            if not os.path.isabs(path):
                path = os.path.join(base_dir, path)
            template_tag = item.get("template_tag", None)
            v = _load_vector_file(path)
            vecs.append(SteeringVector(name=name, concept=concept, layer=layer, alpha=alpha, vec=v, template_tag=template_tag))
            if max_vectors is not None and len(vecs) >= max_vectors:
                break
    if not vecs:
        raise RuntimeError(f"No vectors loaded from {manifest_path}")
    return vecs


# -----------------------------
# Generation state + hook
# -----------------------------
class GenerationState:
    def __init__(self, batch_size: int, device: torch.device, reasoning_threshold: int):
        self.batch_size = batch_size
        self.device = device
        self.reasoning_threshold = int(reasoning_threshold)
        self.unfinished = torch.ones(batch_size, dtype=torch.bool, device=device)
        self.gen_steps = torch.zeros(batch_size, dtype=torch.long, device=device)

    def current_reasoning_mask(self) -> torch.Tensor:
        return self.unfinished & (self.gen_steps < self.reasoning_threshold)

    def step_update(self, next_tokens: torch.Tensor, eos_token_id: int) -> None:
        next_tokens = next_tokens.squeeze(-1)
        active = self.unfinished.clone()
        self.gen_steps[active] += 1
        newly_finished = active & (next_tokens == eos_token_id)
        self.unfinished[newly_finished] = False


class HookStats:
    def __init__(self, name: str):
        self.name = name
        self.prefill_calls = 0
        self.decode_calls = 0
        self.intervened = 0


class LastTokenSteeringHook:
    """
    Add alpha * v to the last token hidden state at the chosen layer.
    phase_mode:
      - "prefill": apply only on the prefill forward (seq_len > 1)
      - "decode":  apply only on decode forwards (seq_len == 1)
      - "both":    apply on both
    If staged=True, only apply on decode steps where (unfinished & gen_steps < reasoning_threshold).
    """
    def __init__(self, v_np: np.ndarray, alpha: float, stats: HookStats,
                 *, phase_mode: str, staged: bool, reasoning_threshold: int):
        assert phase_mode in ["prefill", "decode", "both"]
        self.v = torch.tensor(v_np.astype(np.float32, copy=False))
        self.v_device: Optional[torch.Tensor] = None
        self.alpha = float(alpha)
        self.stats = stats
        self.phase_mode = phase_mode
        self.staged = bool(staged)
        self.reasoning_threshold = int(reasoning_threshold)
        self.state: Optional[GenerationState] = None

    def set_state(self, st: Optional[GenerationState]) -> None:
        self.state = st

    def _v(self, device: torch.device) -> torch.Tensor:
        if self.v_device is None or self.v_device.device != device:
            self.v_device = self.v.to(device=device)
        return self.v_device

    def __call__(self, module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        if not isinstance(hs, torch.Tensor) or hs.ndim != 3:
            return output

        seq_len = hs.shape[1]
        is_decode = (seq_len == 1)
        if is_decode:
            self.stats.decode_calls += 1
        else:
            self.stats.prefill_calls += 1

        if self.phase_mode == "prefill" and is_decode:
            return output
        if self.phase_mode == "decode" and (not is_decode):
            return output

        if self.staged and is_decode and self.state is not None:
            mask = self.state.current_reasoning_mask()
            if not bool(mask.any().item()):
                return output
            x = hs[:, -1, :].float()
            v = self._v(hs.device)
            x_sel = x[mask]
            x_sel = x_sel + self.alpha * v
            x[mask] = x_sel
            hs2 = hs.clone()
            hs2[:, -1, :] = x.to(dtype=hs.dtype)
            self.stats.intervened += 1
            if isinstance(output, tuple):
                return (hs2,) + output[1:]
            return hs2

        # non-staged (or prefill)
        x = hs[:, -1, :].float()
        v = self._v(hs.device)
        hs2 = hs.clone()
        hs2[:, -1, :] = (x + self.alpha * v).to(dtype=hs.dtype)
        self.stats.intervened += 1
        if isinstance(output, tuple):
            return (hs2,) + output[1:]
        return hs2


def register_steering_hooks(
    model,
    *,
    layer_idx: int,
    v_np: np.ndarray,
    alpha: float,
    phase_mode: str,
    staged: bool,
    reasoning_threshold: int,
) -> Tuple[List[Any], Optional[Any], List[HookStats]]:
    layers = get_model_layers(model)
    if layer_idx < 0 or layer_idx >= len(layers):
        raise ValueError(f"layer_idx out of range: {layer_idx} (n_layers={len(layers)})")

    stats = HookStats(name=f"{phase_mode}{'_staged' if staged else ''}@{layer_idx}")
    hook = LastTokenSteeringHook(
        v_np=v_np,
        alpha=alpha,
        stats=stats,
        phase_mode=phase_mode,
        staged=staged,
        reasoning_threshold=reasoning_threshold,
    )
    handle = layers[layer_idx].register_forward_hook(hook)

    def setter(st: Optional[GenerationState]) -> None:
        hook.set_state(st)

    return [handle], (setter if staged else None), [stats]


def remove_hooks(handles: List[Any]) -> None:
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass


# -----------------------------
# Manual KV-cached generation (same spirit as your script)
# -----------------------------
@torch.no_grad()
def generate_continuations(
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

    inner_pos = 1 if TQDM_OUTER else 0
    for i in tqdm(
        range(0, len(prompts), batch_size),
        desc=f"Generate({decoding})",
        disable=not TQDM_INNER,
        position=inner_pos,
        leave=False,
    ):
        use_template = bool(getattr(tokenizer, "chat_template", None))
        batch = prompts[i:i+batch_size]
        batch = [render_prompt(tokenizer, p, add_generation_prompt=True) for p in batch]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_len,
            add_special_tokens=not use_template,
        ).to(device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        B, T0 = input_ids.shape

        state = GenerationState(B, input_ids.device, reasoning_token_threshold)
        if state_setter is not None:
            state_setter(state)

        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        logits = out.logits[:, -1, :]
        past = out.past_key_values

        generated = input_ids

        for _ in range(max_new_tokens):
            if decoding == "greedy":
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                lt = logits / max(temperature, 1e-6)
                lt = top_k_filtering(lt, top_k=top_k)
                lt = top_p_filtering(lt, top_p=top_p)
                probs = torch.softmax(lt, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

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
    model.config.use_cache = True
    return model, tok


# -----------------------------
# Evaluation helper
# -----------------------------
def _is_correct(dataset: str, pred: str, gold: str) -> int:
    if is_correct_bool is None:
        raise RuntimeError(f"benchmark_dataloaders.is_correct not available: {_IMPORT_ERR}")
    return int(is_correct_bool(dataset, pred, gold))


@torch.no_grad()
def evaluate_with_steering(
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
    reasoning_token_threshold: int,
    # steering:
    steering: Optional[SteeringVector],
    phase_mode: str,       # "prefill" | "decode" | "both" | "none"
    staged: bool,
    # repro:
    global_seed: int,
    sample_seed: Optional[int] = None,
) -> Dict[str, Any]:
    if phase_mode == "none" or steering is None:
        handles, state_setter, hook_stats = [], None, []
    else:
        handles, state_setter, hook_stats = register_steering_hooks(
            model=model,
            layer_idx=steering.layer,
            v_np=steering.vec,
            alpha=steering.alpha,
            phase_mode=phase_mode,
            staged=staged,
            reasoning_threshold=reasoning_token_threshold,
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

        correct = []
        for ex, cont in zip(examples, continuations):
            if parse_prediction is None:
                raise RuntimeError(f"benchmark_dataloaders.parse_prediction not available: {_IMPORT_ERR}")
            pred = parse_prediction(ex.dataset, cont)
            correct.append(_is_correct(ex.dataset, pred, ex.gold))
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
                } for s in hook_stats
            ],
            "correct": correct_arr.tolist(),
        }
    finally:
        remove_hooks(handles)


# -----------------------------
# Ranking + correlation utils
# -----------------------------
def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks for ties, 1..n (like scipy.stats.rankdata(method='average'))."""
    n = a.shape[0]
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and a[order[j+1]] == a[order[i]]:
            j += 1
        rank = 0.5 * (i + j) + 1.0
        for k in range(i, j+1):
            ranks[order[k]] = rank
        i = j + 1
    return ranks


def spearmanr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size != b.size or a.size < 2:
        return float("nan")
    ra = _rankdata(a)
    rb = _rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(np.sqrt(np.sum(ra * ra) * np.sum(rb * rb)))
    if denom <= 0:
        return float("nan")
    return float(np.sum(ra * rb) / denom)


def agg_task_scores(per_task: Dict[str, float], *, agg: str = "mean") -> float:
    vals = [v for v in per_task.values() if not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return float("nan")
    if agg == "mean":
        return float(np.mean(vals))
    if agg == "min":
        return float(np.min(vals))
    if agg == "median":
        return float(np.median(vals))
    raise ValueError(f"Unknown agg={agg}")


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
        vals = [d[k] for d in dicts if k in d and not (isinstance(d[k], float) and math.isnan(d[k]))]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out


def _summarize_scores(scores_by_seed: Dict[int, float]) -> Dict[str, Any]:
    seeds = sorted(scores_by_seed.keys())
    vals = np.array([scores_by_seed[s] for s in seeds], dtype=np.float64)
    n = int(vals.size)
    mean = float(np.mean(vals)) if n else float("nan")
    if n <= 1:
        std = 0.0
        ci95 = [mean, mean]
    else:
        std = float(np.std(vals, ddof=1))
        sem = std / math.sqrt(n)
        half = 1.96 * sem
        ci95 = [mean - half, mean + half]
    return {
        "n_seeds": n,
        "mean": mean,
        "std": std,
        "ci95": ci95,
        "by_seed": {int(s): float(scores_by_seed[s]) for s in seeds},
    }


# -----------------------------
# Main experiment
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model_dtype", type=str, default="fp32", choices=["fp32", "fp16"])

    ap.add_argument("--vectors_manifest", type=str, required=True)
    ap.add_argument("--max_vectors", type=int, default=0, help="0 means no limit.")
    ap.add_argument("--filter_regex", type=str, default="", help="Optional regex to filter vector names/concepts.")

    # Evaluation tasks/datasets
    ap.add_argument("--tasks", type=str, default="commonsenseqa,arc_challenge,openbookqa,qasc,logiqa")
    ap.add_argument("--n_eval", type=int, default=128)

    # Templates
    ap.add_argument("--template_randomization", type=int, default=1, choices=[0, 1])
    ap.add_argument("--shuffle_choices", type=int, default=1, choices=[0, 1])
    ap.add_argument("--answer_prefix", type=str, default="\nFinal answer:")
    ap.add_argument("--add_answer_prefix", type=int, default=1, choices=[0, 1])

    ap.add_argument("--template_seed_rank", type=int, default=1234, help="Template seed used to compute ranking scores.")
    ap.add_argument("--template_seed_real", type=int, default=5678, help="Held-out template seed for 'real' eval.")
    ap.add_argument(
        "--template_seeds_rank",
        type=str,
        default="",
        help="Optional comma-separated list of template seeds for ranking (overrides --template_seed_rank).",
    )
    ap.add_argument(
        "--template_seeds_real",
        type=str,
        default="",
        help="Optional comma-separated list of template seeds for REAL eval (overrides --template_seed_real).",
    )

    # Decoding
    ap.add_argument("--decoding", type=str, default="greedy", choices=["greedy", "sample"])
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--reasoning_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_prompt_len", type=int, default=512)
    ap.add_argument("--sample_seed", type=int, default=12345)

    # Protocol knobs
    ap.add_argument("--trad_mode", type=str, default="prefill", choices=["prefill", "both"],
                    help="Traditional evaluation mode: prefill-only or both (prefill+decode).")
    ap.add_argument(
        "--trad_backend",
        type=str,
        default="",
        help="(Deprecated) Kept for backward compatibility with older commands/docs; ignored.",
    )
    ap.add_argument("--decode_mode", type=str, default="decode", choices=["decode", "both"],
                    help="Decode-protocol evaluation mode (usually 'decode').")
    ap.add_argument("--staged", type=int, default=1, choices=[0, 1],
                    help="Whether to stage steering (apply only during first --reasoning_tokens decode steps).")

    ap.add_argument("--agg", type=str, default="mean", choices=["mean", "min", "median"],
                    help="How to aggregate across tasks into one scalar score per vector.")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_json", type=str, default="ranking_flip_results.json")

    ap.add_argument("--start_idx", type=int, default=0, help="Start index (0-based) into the loaded vector list.")
    ap.add_argument("--end_idx", type=int, default=-1, help="End index (exclusive). -1 means end.")
    ap.add_argument("--resume", type=int, default=0, choices=[0, 1],
                    help="If --out_json exists, load it and skip vectors already present (requires compatible config).")
    ap.add_argument(
        "--save_every",
        type=int,
        default=1,
        help="Write partial --out_json every N evaluated vectors (0 disables partial saves; final save still happens).",
    )
    ap.add_argument("--save_every_seconds", type=int, default=0,
                    help="Also write partial --out_json if at least this many seconds elapsed since last save (0 disables).")
    ap.add_argument("--tqdm_outer", type=int, default=1, choices=[0, 1],
                    help="Show an outer tqdm progress bar over vectors.")
    ap.add_argument("--tqdm_inner", type=int, default=1, choices=[0, 1],
                    help="Show inner tqdm bars for per-batch generation (Generate(...)).")

    args = ap.parse_args()
    set_global_seed(args.seed)

    global TQDM_OUTER, TQDM_INNER
    TQDM_OUTER = bool(int(getattr(args, "tqdm_outer", 1)))
    TQDM_INNER = bool(int(getattr(args, "tqdm_inner", 1)))

    out_path = os.path.expanduser(args.out_json)
    existing = _load_json_if_exists(out_path) if bool(args.resume) else None
    if existing is not None:
        if not isinstance(existing, dict):
            raise RuntimeError(f"--resume=1 but existing out_json is not a JSON object: {out_path}")
        old_cfg = existing.get("config", {})
        if isinstance(old_cfg, dict):
            ok, diffs = _config_compatible(old_cfg, vars(args))
            if not ok:
                raise RuntimeError(
                    f"--resume=1 but config mismatch in {out_path} for keys={diffs}. "
                    "Refuse to merge incompatible runs; use a new --out_json."
                )
        else:
            raise RuntimeError(f"--resume=1 but missing/invalid config in {out_path}; use a new --out_json.")

    if load_selected_tasks is None:
        raise RuntimeError(
            "benchmark_dataloaders is required for this script. "
            f"Import error was: {_IMPORT_ERR}"
        )

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if not tasks:
        raise ValueError("Empty --tasks")

    max_vectors = None if args.max_vectors <= 0 else int(args.max_vectors)
    vecs = load_vectors_from_manifest(args.vectors_manifest, max_vectors=max_vectors)

    if args.filter_regex:
        pat = re.compile(args.filter_regex)
        vecs = [v for v in vecs if pat.search(v.name) or pat.search(v.concept)]
        if not vecs:
            raise RuntimeError("No vectors after --filter_regex")

    # Slice vectors (used for multi-GPU sharding).
    vecs_all = list(vecs)
    n_total = len(vecs_all)
    start_idx = max(int(args.start_idx), 0)
    end_idx = int(args.end_idx)
    if end_idx < 0 or end_idx > n_total:
        end_idx = n_total
    if start_idx > n_total:
        start_idx = n_total
    vecs = vecs_all[start_idx:end_idx]

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model, args.device, args.model_dtype)
    hidden_dim = infer_hidden_dim(model)
    if hidden_dim is None:
        print("[Warn] Could not infer hidden_dim; will skip dimension check.")
    else:
        for v in vecs:
            if v.vec.shape[0] != hidden_dim:
                raise ValueError(f"Vector dim mismatch for {v.name}: {v.vec.shape[0]} != {hidden_dim}")

    # -------------------
    # Load evaluation sets for ranking and real
    # -------------------
    def load_eval(template_seed: int) -> Dict[str, List[Any]]:
        _sub_by, eval_by, _meta = load_selected_tasks(
            tasks=tasks,
            n_subspace=1,  # (compat) loader may not accept 0,                 # We don't need subspace prompts here
            n_eval=args.n_eval,
            seed=args.seed,
            template_seed=template_seed,
            template_randomization=bool(args.template_randomization),
            shuffle_choices=bool(args.shuffle_choices),
            add_answer_prefix=bool(args.add_answer_prefix),
            answer_prefix=args.answer_prefix,
        )
        return eval_by

    rank_seeds = _parse_csv_ints(args.template_seeds_rank) or [int(args.template_seed_rank)]
    real_seeds = _parse_csv_ints(args.template_seeds_real) or [int(args.template_seed_real)]
    rank_seeds = _dedup_keep_order(rank_seeds)
    real_seeds = _dedup_keep_order(real_seeds)

    print(f"[Data] Loading ranking eval sets (template_seeds={rank_seeds}) ...")
    eval_rank_by_seed: Dict[int, Dict[str, List[Any]]] = {s: load_eval(s) for s in rank_seeds}
    print(f"[Data] Loading REAL eval sets (template_seeds={real_seeds}) ...")
    eval_real_by_seed: Dict[int, Dict[str, List[Any]]] = {s: load_eval(s) for s in real_seeds}

    # Baseline accuracies (no steering) on both template regimes
    base_rank_by_seed: Dict[int, Dict[str, float]] = {}
    base_real_by_seed: Dict[int, Dict[str, float]] = {}

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

    reused_baseline = False
    if existing is not None:
        cand_rank = _coerce_seed_keyed(existing.get("baseline_rank_by_seed"))
        cand_real = _coerce_seed_keyed(existing.get("baseline_real_by_seed"))
        need_rank = all((s in cand_rank) and all(t in cand_rank[s] for t in tasks) for s in rank_seeds)
        need_real = all((s in cand_real) and all(t in cand_real[s] for t in tasks) for s in real_seeds)
        if need_rank and need_real:
            base_rank_by_seed = cand_rank
            base_real_by_seed = cand_real
            reused_baseline = True
            print("\n[Baseline] Reusing baseline from existing out_json (--resume=1).")

    if not reused_baseline:
        print(f"\n[Baseline] Evaluating baseline (no steering) on ranking templates (n_seeds={len(rank_seeds)}) ...")
        for s in rank_seeds:
            base_rank_by_seed[s] = {}
            for t in tasks:
                res = evaluate_with_steering(
                    model=model, tokenizer=tokenizer, examples=eval_rank_by_seed[s][t],
                    decoding=args.decoding, max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                    device=args.device, batch_size=args.batch_size, max_prompt_len=args.max_prompt_len,
                    reasoning_token_threshold=args.reasoning_tokens,
                    steering=None, phase_mode="none", staged=False,
                    global_seed=args.seed, sample_seed=(args.sample_seed if args.decoding == "sample" else None),
                )
                base_rank_by_seed[s][t] = float(res["accuracy"])

    base_rank_mean: Dict[str, float] = {
        t: float(np.mean([base_rank_by_seed[s][t] for s in rank_seeds])) for t in tasks
    }
    print("[Baseline] Ranking template baseline (mean acc over seeds):")
    for t in tasks:
        print(f"  - {t}: acc={base_rank_mean[t]*100:.1f} (n_eval={args.n_eval})")

    if not reused_baseline:
        print(f"\n[Baseline] Evaluating baseline (no steering) on REAL templates (n_seeds={len(real_seeds)}) ...")
        for s in real_seeds:
            base_real_by_seed[s] = {}
            for t in tasks:
                res = evaluate_with_steering(
                    model=model, tokenizer=tokenizer, examples=eval_real_by_seed[s][t],
                    decoding=args.decoding, max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                    device=args.device, batch_size=args.batch_size, max_prompt_len=args.max_prompt_len,
                    reasoning_token_threshold=args.reasoning_tokens,
                    steering=None, phase_mode="none", staged=False,
                    global_seed=args.seed, sample_seed=(args.sample_seed if args.decoding == "sample" else None),
                )
                base_real_by_seed[s][t] = float(res["accuracy"])

    base_real_mean: Dict[str, float] = {
        t: float(np.mean([base_real_by_seed[s][t] for s in real_seeds])) for t in tasks
    }
    print("[Baseline] REAL template baseline (mean acc over seeds):")
    for t in tasks:
        print(f"  - {t}: acc={base_real_mean[t]*100:.1f} (n_eval={args.n_eval})")

    # -------------------
    # Evaluate vectors under each protocol
    # -------------------
    if existing is None:
        results: Dict[str, Any] = {
            "config": vars(args),
            "template_seeds_rank": rank_seeds,
            "template_seeds_real": real_seeds,
            "baseline_rank_mean": base_rank_mean,
            "baseline_real_mean": base_real_mean,
            "baseline_rank_by_seed": base_rank_by_seed,
            "baseline_real_by_seed": base_real_by_seed,
            "vectors": {},
        }
    else:
        # Resume: keep existing per-vector results, and ensure baseline/config are present for consistency.
        results = existing
        results["config"] = vars(args)
        results["template_seeds_rank"] = rank_seeds
        results["template_seeds_real"] = real_seeds
        results["baseline_rank_mean"] = base_rank_mean
        results["baseline_real_mean"] = base_real_mean
        results["baseline_rank_by_seed"] = base_rank_by_seed
        results["baseline_real_by_seed"] = base_real_by_seed
        if "vectors" not in results or not isinstance(results["vectors"], dict):
            results["vectors"] = {}

    results["progress"] = {
        "n_vectors_total": int(n_total),
        "n_vectors_slice": int(len(vecs)),
        "start_idx": int(start_idx),
        "end_idx": int(end_idx),
        "n_vectors_done": int(len(results.get("vectors", {}))) if isinstance(results.get("vectors"), dict) else 0,
    }

    # Save a baseline-only checkpoint so that long runs can be safely resumed even if interrupted
    # before finishing the first vector.
    _atomic_json_dump(results, out_path)
    last_save_t = time.time()

    score_trad = []
    score_decode = []
    score_real = []
    names = []

    staged = bool(args.staged)

    done_names = set(results.get("vectors", {}).keys()) if isinstance(results.get("vectors"), dict) else set()
    save_every = int(args.save_every)
    save_every_seconds = int(getattr(args, "save_every_seconds", 0) or 0)
    n_since_save = 0

    vecs_todo = [sv for sv in vecs if sv.name not in done_names]
    if done_names:
        n_skipped = int(len(vecs) - len(vecs_todo))
        if n_skipped > 0:
            print(f"[Resume] Skipping {n_skipped}/{len(vecs)} already-computed vectors in this slice.")

    vec_iter = vecs_todo
    if bool(getattr(args, "tqdm_outer", 1)):
        vec_iter = tqdm(vecs_todo, desc="Vectors", unit="vec", position=0)

    for sv in vec_iter:
        if hasattr(vec_iter, "set_postfix_str"):
            try:
                vec_iter.set_postfix_str(sv.name)
            except Exception:
                pass

        print("\n" + "=" * 80)
        print(f"[Vector] {sv.name} concept={sv.concept} layer={sv.layer} alpha={sv.alpha}")
        print("=" * 80)

        per_task_trad = {}
        per_task_decode = {}
        per_task_real = {}

        # TRAD on ranking templates
        per_task_trad_by_seed: Dict[int, Dict[str, float]] = {}
        score_trad_by_seed: Dict[int, float] = {}
        for s in rank_seeds:
            d = {}
            for t in tasks:
                res = evaluate_with_steering(
                    model=model, tokenizer=tokenizer, examples=eval_rank_by_seed[s][t],
                    decoding=args.decoding, max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                    device=args.device, batch_size=args.batch_size, max_prompt_len=args.max_prompt_len,
                    reasoning_token_threshold=args.reasoning_tokens,
                    steering=sv, phase_mode=args.trad_mode, staged=staged,
                    global_seed=args.seed, sample_seed=(args.sample_seed if args.decoding == "sample" else None),
                )
                d[t] = float(res["accuracy"] - base_rank_by_seed[s][t])
            per_task_trad_by_seed[s] = d
            score_trad_by_seed[s] = agg_task_scores(d, agg=args.agg)

        # DECODE protocol on ranking templates
        per_task_decode_by_seed: Dict[int, Dict[str, float]] = {}
        score_decode_by_seed: Dict[int, float] = {}
        for s in rank_seeds:
            d = {}
            for t in tasks:
                res = evaluate_with_steering(
                    model=model, tokenizer=tokenizer, examples=eval_rank_by_seed[s][t],
                    decoding=args.decoding, max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                    device=args.device, batch_size=args.batch_size, max_prompt_len=args.max_prompt_len,
                    reasoning_token_threshold=args.reasoning_tokens,
                    steering=sv, phase_mode=args.decode_mode, staged=staged,
                    global_seed=args.seed, sample_seed=(args.sample_seed if args.decoding == "sample" else None),
                )
                d[t] = float(res["accuracy"] - base_rank_by_seed[s][t])
            per_task_decode_by_seed[s] = d
            score_decode_by_seed[s] = agg_task_scores(d, agg=args.agg)

        # REAL: decode-only on held-out templates
        per_task_real_by_seed: Dict[int, Dict[str, float]] = {}
        score_real_by_seed: Dict[int, float] = {}
        for s in real_seeds:
            d = {}
            for t in tasks:
                res = evaluate_with_steering(
                    model=model, tokenizer=tokenizer, examples=eval_real_by_seed[s][t],
                    decoding=args.decoding, max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                    device=args.device, batch_size=args.batch_size, max_prompt_len=args.max_prompt_len,
                    reasoning_token_threshold=args.reasoning_tokens,
                    steering=sv, phase_mode="decode", staged=staged,
                    global_seed=args.seed, sample_seed=(args.sample_seed if args.decoding == "sample" else None),
                )
                d[t] = float(res["accuracy"] - base_real_by_seed[s][t])
            per_task_real_by_seed[s] = d
            score_real_by_seed[s] = agg_task_scores(d, agg=args.agg)

        per_task_trad = _mean_by_key(list(per_task_trad_by_seed.values()))
        per_task_decode = _mean_by_key(list(per_task_decode_by_seed.values()))
        per_task_real = _mean_by_key(list(per_task_real_by_seed.values()))

        summ_trad = _summarize_scores(score_trad_by_seed)
        summ_dec = _summarize_scores(score_decode_by_seed)
        summ_real = _summarize_scores(score_real_by_seed)

        s_trad = float(summ_trad["mean"])
        s_dec = float(summ_dec["mean"])
        s_real = float(summ_real["mean"])

        results["vectors"][sv.name] = {
            "concept": sv.concept,
            "layer": sv.layer,
            "alpha": sv.alpha,
            "template_tag": sv.template_tag,
            "delta_rank_trad": per_task_trad,
            "delta_rank_decode": per_task_decode,
            "delta_real_decode": per_task_real,
            "score_rank_trad": s_trad,
            "score_rank_decode": s_dec,
            "score_real": s_real,
            "delta_rank_trad_by_seed": per_task_trad_by_seed,
            "delta_rank_decode_by_seed": per_task_decode_by_seed,
            "delta_real_decode_by_seed": per_task_real_by_seed,
            "score_rank_trad_summary": summ_trad,
            "score_rank_decode_summary": summ_dec,
            "score_real_summary": summ_real,
        }

        names.append(sv.name)
        score_trad.append(s_trad)
        score_decode.append(s_dec)
        score_real.append(s_real)

        print(
            f"[Scores] agg={args.agg}  "
            f"trad={s_trad:+.4f}±{float(summ_trad['std']):.4f}  "
            f"decode={s_dec:+.4f}±{float(summ_dec['std']):.4f}  "
            f"real={s_real:+.4f}±{float(summ_real['std']):.4f}"
        )

        n_since_save += 1
        now_t = time.time()
        should_save = (save_every > 0 and n_since_save >= save_every) or (
            save_every_seconds > 0 and (now_t - last_save_t) >= float(save_every_seconds)
        )
        if should_save:
            results["progress"]["n_vectors_done"] = int(len(results.get("vectors", {})))
            _atomic_json_dump(results, out_path)
            n_since_save = 0
            last_save_t = now_t

    # Compute correlations on *all* vectors currently saved in the results (useful for sharded/resume runs).
    all_vecs = results.get("vectors", {})
    if not isinstance(all_vecs, dict) or not all_vecs:
        raise RuntimeError("No vectors in results; nothing to summarize.")
    all_names = sorted(all_vecs.keys())
    score_trad = np.array([float(all_vecs[n]["score_rank_trad"]) for n in all_names], dtype=np.float64)
    score_decode = np.array([float(all_vecs[n]["score_rank_decode"]) for n in all_names], dtype=np.float64)
    score_real = np.array([float(all_vecs[n]["score_real"]) for n in all_names], dtype=np.float64)

    rho_trad_decode = spearmanr(score_trad, score_decode)
    rho_trad_real = spearmanr(score_trad, score_real)
    rho_decode_real = spearmanr(score_decode, score_real)

    results["correlations"] = {
        "spearman_trad_vs_decode": rho_trad_decode,
        "spearman_trad_vs_real": rho_trad_real,
        "spearman_decode_vs_real": rho_decode_real,
    }

    # Print top-k for sanity
    def topk(idx_scores: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        order = np.argsort(-idx_scores)
        out = []
        for j in order[:min(k, len(order))]:
            out.append((all_names[j], float(idx_scores[j])))
        return out

    print("\n" + "=" * 80)
    print("[Ranking flip] Top vectors by TRAD score (rank template):")
    for n, s in topk(score_trad, k=10):
        print(f"  {n:40s} {s:+.4f}")
    print("\n[Ranking flip] Top vectors by DECODE score (rank template):")
    for n, s in topk(score_decode, k=10):
        print(f"  {n:40s} {s:+.4f}")
    print("\n[REAL] Top vectors by REAL score (held-out template, decode):")
    for n, s in topk(score_real, k=10):
        print(f"  {n:40s} {s:+.4f}")

    print("\n" + "-" * 80)
    print(f"Spearman(trad, decode) = {rho_trad_decode:.3f}")
    print(f"Spearman(trad, real)   = {rho_trad_real:.3f}")
    print(f"Spearman(decode, real) = {rho_decode_real:.3f}")
    print("-" * 80)

    results["progress"]["n_vectors_done"] = int(len(all_names))
    _atomic_json_dump(results, out_path)
    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()
