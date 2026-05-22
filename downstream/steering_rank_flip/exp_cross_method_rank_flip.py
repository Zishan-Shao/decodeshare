
# -*- coding: utf-8 -*-
"""
exp_cross_method_rank_flip.py

A1: Cross-method candidate pools + ranking-flip under TRAD vs DECODE vs REAL.

Goal
----
Show that DecodeShare-style *decode-aligned* evaluation/ranking predicts REAL KV-cached decode
performance better than a prefill/TRAD proxy — and that this holds across *different sources of
candidate steering vectors*:

  (1) CAA (Contrastive Activation Addition) candidate pool
  (2) Instruction activation steering candidate pool
  (3) SAE-based steering candidate pool (SAIF/SAE-SSV style: SAE features -> decoder directions)

This script is designed to be:
  - "one-shot runnable": build vectors + run rank-flip for all pools + emit a summary table
  - protocol-faithful: KV-cached decode generation; TRAD=apply only during prefill; DECODE=apply only
    during decode steps (seq_len==1); REAL=held-out templates + decode-only.
  - rigorous: multi template seeds for ranking and real evaluation; reports Spearman, top-k regret, etc.

Dependencies
------------
pip install transformers datasets numpy torch tqdm

AND your project must provide benchmark_dataloaders with:
  - Example dataclass: .prompt, .gold, .dataset
  - load_selected_tasks(...)
  - parse_prediction(dataset, continuation) -> str
  - is_correct(dataset, pred, gold) -> bool/int
  - stable_int_seed(...) -> int

If benchmark_dataloaders is missing, the script exits with a clear error.

Typical usage
-------------
CUDA_VISIBLE_DEVICES=0 python exp_cross_method_rank_flip.py \
  --model meta-llama/Llama-2-7b-chat-hf --device cuda --model_dtype fp16 \
  --layer 28 \
  --tasks_eval commonsenseqa,arc_challenge,openbookqa,qasc,logiqa \
  --n_eval 128 \
  --template_seeds_rank 1234,2345,3456 \
  --template_seeds_real 4567,5678,6789 \
  --decoding greedy --reasoning_tokens 128 --max_new_tokens 256 \
  --n_vec_caa 32 --n_vec_instr 64 --n_vec_sae 64 --subset_size 96 \
  --sae_train_samples 20000 --sae_latent_dim 8192 --sae_steps 3000 \
  --out_dir outputs/steering_rank_flip/cross_method

Notes on method fidelity
------------------------
- CAA: the original CAA computes a steering vector as the mean difference of residual-stream activations
  between positive and negative examples, and adds it during inference after the user's prompt.
  Here, we follow the same *contrastive mean-difference* definition, using contrastive "correct vs wrong"
  answer continuations as positive/negative pairs (MCQ tasks). (This makes vectors non-trivial on the same
  evaluation suite, which is important for rank-flip power.)
- Instruction steering: we follow the definition from activation steering for instruction following:
  vector = mean activation(with instruction) - mean activation(without instruction), extracted at the
  last prefill token that conditions generation.
- SAE steering: we train a simple sparse autoencoder on layer activations and use decoder column vectors
  as candidate directions (feature steering). This matches the core mechanism used by SAE-based steering
  lines (e.g., SAIF / SAE-SSV): represent activations in sparse latent space and steer with feature directions.

Outputs
-------
Under out_dir/, you get:
  vectors/{method}/...npy
  manifests/{method}.jsonl
  rankflip/{method}.json
  summary.md  (paper-facing ranking table)
"""

import os
import sys
import re
import json
import math
import time
import argparse
import hashlib
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# -----------------------------
# Local imports (repo layout)
# -----------------------------
# This script lives in `downstream/steering_rank_flip/`; public releases keep
# benchmark_dataloaders.py with the experiment/downstream bundles.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _candidate in [
    os.path.join(THIS_DIR, "..", "..", "src"),
    os.path.join(THIS_DIR, "..", "brittleness"),
    os.path.join(THIS_DIR, "..", "patch_back"),
    os.path.join(THIS_DIR, "..", "..", "experiments", "02_decode_ablation"),
]:
    _candidate = os.path.normpath(_candidate)
    if os.path.isfile(os.path.join(_candidate, "benchmark_dataloaders.py")) and _candidate not in sys.path:
        sys.path.append(_candidate)


# -----------------------------
# Project imports (required)
# -----------------------------
try:
    from benchmark_dataloaders import (
        Example,
        load_selected_tasks,
        parse_prediction,
        is_correct as is_correct_bool,
        stable_int_seed as stable_int_seed_project,
    )
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "benchmark_dataloaders is required for this script.\n"
        "Please ensure your repo `src/` is on PYTHONPATH (this script auto-adds ../../src when present).\n"
        f"Import error: {e}"
    )

stable_int_seed = stable_int_seed_project


# -----------------------------
# Repro utils
# -----------------------------
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def json_dump(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


# -----------------------------
# Model utils
# -----------------------------
def infer_hidden_dim(model) -> Optional[int]:
    cfg = getattr(model, "config", None)
    for k in ("hidden_size", "n_embd", "dim", "d_model", "model_dim", "embed_dim"):
        v = getattr(cfg, k, None)
        if isinstance(v, int) and v > 0:
            return int(v)
    try:
        emb = model.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight") and isinstance(emb.weight, torch.Tensor) and emb.weight.ndim == 2:
            return int(emb.weight.shape[1])
    except Exception:
        pass
    return None


def get_model_layers(model) -> List[torch.nn.Module]:
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


def load_model_and_tokenizer(model_name: str, device: str, model_dtype: str):
    dtype = torch.float16 if model_dtype == "fp16" else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)

    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = model.to(device)
    model.eval()
    model.config.use_cache = True
    return model, tok


def render_chat(
    tokenizer,
    user_content: str,
    *,
    assistant_content: Optional[str] = None,
    system_content: Optional[str] = None,
    add_generation_prompt: bool = True,
) -> str:
    tmpl = getattr(tokenizer, "chat_template", None)
    if not tmpl:
        # Plain text fallback
        if assistant_content is None:
            return user_content
        return user_content + "\n" + assistant_content

    messages = []
    if system_content:
        messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": user_content})
    if assistant_content is not None:
        messages.append({"role": "assistant", "content": assistant_content})
        add_generation_prompt = False  # assistant message already present

    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    except Exception:
        # Some templates don't support system role
        messages = [{"role": "user", "content": user_content}]
        if assistant_content is not None:
            messages.append({"role": "assistant", "content": assistant_content})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=(assistant_content is None))


# -----------------------------
# Activation extraction (mean / matrix)
# -----------------------------
@torch.no_grad()
def mean_last_token_activation(
    model,
    tokenizer,
    texts: List[str],
    *,
    layer_idx: int,
    device: str,
    batch_size: int,
    max_prompt_len: int,
) -> np.ndarray:
    """
    Returns mean_{examples} h_{layer}(last_token), where h is the layer output hidden state.
    Efficient: accumulates sum without storing all activations.
    """
    model.eval()
    layers = get_model_layers(model)
    if layer_idx < 0 or layer_idx >= len(layers):
        raise ValueError(f"layer_idx out of range: {layer_idx} (n_layers={len(layers)})")

    hid_dim = infer_hidden_dim(model)
    if hid_dim is None:
        raise RuntimeError("Could not infer hidden dim for mean_last_token_activation")

    acc = torch.zeros(hid_dim, device=device, dtype=torch.float32)
    count = 0

    def hook_fn(module, inputs, output):
        nonlocal acc, count
        hs = output[0] if isinstance(output, tuple) else output
        if not isinstance(hs, torch.Tensor) or hs.ndim != 3:
            return output
        last = hs[:, -1, :].detach().float()  # [B, D]
        acc += last.sum(dim=0)
        count += last.shape[0]
        return output

    h = layers[layer_idx].register_forward_hook(hook_fn)
    try:
        use_template = bool(getattr(tokenizer, "chat_template", None))
        for i in tqdm(range(0, len(texts), batch_size), desc=f"MeanActs@L{layer_idx}"):
            batch = texts[i:i+batch_size]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_prompt_len,
                add_special_tokens=not use_template,
            ).to(device)
            _ = model(**enc, use_cache=False)
        if count == 0:
            raise RuntimeError("No activations collected (count==0)")
        mean = (acc / float(count)).detach().float().cpu().numpy()
        return mean
    finally:
        try:
            h.remove()
        except Exception:
            pass


@torch.no_grad()
def collect_last_token_activations_matrix(
    model,
    tokenizer,
    texts: List[str],
    *,
    layer_idx: int,
    device: str,
    batch_size: int,
    max_prompt_len: int,
    max_samples: int,
    dtype: str = "fp16",
) -> torch.Tensor:
    """
    Collect per-example last-token activations at layer layer_idx.
    Returns tensor [N, D] on CPU (float16 by default).
    """
    model.eval()
    layers = get_model_layers(model)
    if layer_idx < 0 or layer_idx >= len(layers):
        raise ValueError(f"layer_idx out of range: {layer_idx} (n_layers={len(layers)})")

    hid_dim = infer_hidden_dim(model)
    if hid_dim is None:
        raise RuntimeError("Could not infer hidden dim for collect_last_token_activations_matrix")

    out_chunks: List[torch.Tensor] = []
    remaining = int(max_samples)

    def hook_fn(module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        if not isinstance(hs, torch.Tensor) or hs.ndim != 3:
            return output
        last = hs[:, -1, :].detach()
        # store to CPU float16
        last = last.to(dtype=torch.float16 if dtype == "fp16" else torch.float32).cpu()
        out_chunks.append(last)
        return output

    h = layers[layer_idx].register_forward_hook(hook_fn)
    try:
        use_template = bool(getattr(tokenizer, "chat_template", None))
        for i in tqdm(range(0, len(texts), batch_size), desc=f"CollectActs@L{layer_idx}"):
            if remaining <= 0:
                break
            batch = texts[i:i+batch_size]
            # if we only need a few samples, truncate batch
            if len(batch) > remaining:
                batch = batch[:remaining]
            remaining -= len(batch)

            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_prompt_len,
                add_special_tokens=not use_template,
            ).to(device)
            _ = model(**enc, use_cache=False)
        if not out_chunks:
            raise RuntimeError("No activations collected for SAE training.")
        X = torch.cat(out_chunks, dim=0)  # [N, D]
        return X
    finally:
        try:
            h.remove()
        except Exception:
            pass


# -----------------------------
# Candidate pools
# -----------------------------
CHOICE_SETS: Dict[str, List[str]] = {
    "commonsenseqa": list("ABCDE"),
    "arc_challenge": list("ABCD"),
    "openbookqa": list("ABCD"),
    "qasc": list("ABCDEFGH"),
    "logiqa": list("ABCD"),
    "aqua": list("ABCDE"),
}


def sample_wrong_label(dataset: str, gold: str, rng: random.Random) -> str:
    cs = CHOICE_SETS.get(dataset, list("ABCD"))
    cs = [c for c in cs if c != gold]
    if not cs:
        # fallback (should not happen)
        cs = [c for c in list("ABCD") if c != gold] or ["B"]
    return rng.choice(cs)


def normalize_vec(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    if not math.isfinite(n) or n < eps:
        return v
    return (v / n).astype(np.float32)


def load_corpus_examples(
    *,
    tasks: List[str],
    n_subspace: int,
    seed: int,
    template_seed: int,
    template_randomization: bool,
    shuffle_choices: bool,
    add_answer_prefix: bool,
    answer_prefix: str,
) -> List[Any]:
    """
    Returns a flat list of Examples from the "subspace" split (preferred) across tasks.
    """
    # Handle signature drift robustly.
    import inspect
    sig = inspect.signature(load_selected_tasks)
    kwargs = dict(
        tasks=tasks,
        n_subspace=int(n_subspace),
        n_eval=1,  # minimal; we only need subspace prompts
        seed=int(seed),
        template_seed=int(template_seed),
        template_randomization=bool(template_randomization),
        shuffle_choices=bool(shuffle_choices),
        add_answer_prefix=bool(add_answer_prefix),
        answer_prefix=str(answer_prefix),
    )
    kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

    sub_by, _eval_by, _meta = load_selected_tasks(**kwargs)
    corpus: List[Any] = []
    for t in tasks:
        corpus.extend(list(sub_by.get(t, [])))
    if not corpus:
        raise RuntimeError("Empty corpus from load_selected_tasks; check tasks / loader.")
    return corpus


def build_pool_caa(
    *,
    model,
    tokenizer,
    corpus: List[Any],
    layer_idx: int,
    n_vec: int,
    subset_size: int,
    answer_prefix: str,
    alpha: float,
    device: str,
    batch_size: int,
    max_prompt_len: int,
    out_vec_dir: str,
    out_manifest_path: str,
    seed: int,
    skip_if_exists: bool,
) -> None:
    """
    CAA-style vectors: v = mean(h(prompt + correct_answer)) - mean(h(prompt + wrong_answer)).
    We generate many candidate vectors by bootstrapping subsets + varying wrong answers.

    This follows the core CAA definition (contrastive mean-difference in residual activations).
    """
    ensure_dir(out_vec_dir)
    ensure_dir(os.path.dirname(out_manifest_path))

    rng = random.Random(seed)
    if skip_if_exists and os.path.exists(out_manifest_path):
        print(f"[CAA] Manifest exists, skipping build: {out_manifest_path}")
        return

    with open(out_manifest_path, "w", encoding="utf-8") as mf:
        for i in range(n_vec):
            subset = [corpus[rng.randrange(len(corpus))] for _ in range(subset_size)]
            pos_texts: List[str] = []
            neg_texts: List[str] = []
            for ex in subset:
                gold = str(ex.gold).strip()
                wrong = sample_wrong_label(str(ex.dataset), gold, rng)
                user = str(ex.prompt)
                pos_assistant = f"{answer_prefix} {gold}"
                neg_assistant = f"{answer_prefix} {wrong}"
                pos_texts.append(render_chat(tokenizer, user, assistant_content=pos_assistant, add_generation_prompt=False))
                neg_texts.append(render_chat(tokenizer, user, assistant_content=neg_assistant, add_generation_prompt=False))

            mu_pos = mean_last_token_activation(
                model, tokenizer, pos_texts,
                layer_idx=layer_idx, device=device, batch_size=batch_size, max_prompt_len=max_prompt_len
            )
            mu_neg = mean_last_token_activation(
                model, tokenizer, neg_texts,
                layer_idx=layer_idx, device=device, batch_size=batch_size, max_prompt_len=max_prompt_len
            )
            v = normalize_vec(mu_pos - mu_neg)

            vec_path = os.path.join(out_vec_dir, f"caa_vec_{i:04d}_l{layer_idx}.npy")
            np.save(vec_path, v)

            item = dict(
                name=f"CAA_correct_vs_wrong_{i:04d}_l{layer_idx}",
                concept="correct_vs_wrong",
                method="CAA",
                layer=int(layer_idx),
                alpha=float(alpha),
                path=vec_path,
                subset_size=int(subset_size),
                seed=int(seed),
            )
            mf.write(json.dumps(item) + "\n")

    print(f"[CAA] Wrote manifest: {out_manifest_path} (n_vec={n_vec})")


INSTRUCTION_TEMPLATES: List[str] = [
    # keep these short to avoid dominating prompt content
    "Follow the instruction carefully. Your final output MUST be exactly one option letter (A/B/C/D/...).",
    "You must answer with exactly one multiple-choice letter. Do not add any other text.",
    "Important: output only a single letter choice. No explanation.",
    "Return only the letter corresponding to the correct option.",
    "Answer with exactly one character: the option letter.",
]


def build_pool_instruction(
    *,
    model,
    tokenizer,
    corpus: List[Any],
    layer_idx: int,
    n_vec: int,
    subset_size: int,
    alpha: float,
    device: str,
    batch_size: int,
    max_prompt_len: int,
    out_vec_dir: str,
    out_manifest_path: str,
    seed: int,
    skip_if_exists: bool,
) -> None:
    """
    Activation Steering for instruction following:
      v = mean(h(with_instruction)) - mean(h(without_instruction))
    extracted at the last token of the prefill input that conditions generation.

    We generate many candidates by bootstrapping subsets and rotating instruction templates.
    """
    ensure_dir(out_vec_dir)
    ensure_dir(os.path.dirname(out_manifest_path))

    rng = random.Random(seed + 1337)
    if skip_if_exists and os.path.exists(out_manifest_path):
        print(f"[INSTR] Manifest exists, skipping build: {out_manifest_path}")
        return

    with open(out_manifest_path, "w", encoding="utf-8") as mf:
        for i in range(n_vec):
            subset = [corpus[rng.randrange(len(corpus))] for _ in range(subset_size)]
            instr = INSTRUCTION_TEMPLATES[i % len(INSTRUCTION_TEMPLATES)]

            with_texts: List[str] = []
            without_texts: List[str] = []
            for ex in subset:
                user_base = str(ex.prompt)
                user_with = instr + "\n\n" + user_base
                # For extraction, mimic the actual generation conditioning:
                # include generation prompt so the last token corresponds to "assistant start" in chat templates.
                with_texts.append(render_chat(tokenizer, user_with, assistant_content=None, add_generation_prompt=True))
                without_texts.append(render_chat(tokenizer, user_base, assistant_content=None, add_generation_prompt=True))

            mu_with = mean_last_token_activation(
                model, tokenizer, with_texts,
                layer_idx=layer_idx, device=device, batch_size=batch_size, max_prompt_len=max_prompt_len
            )
            mu_without = mean_last_token_activation(
                model, tokenizer, without_texts,
                layer_idx=layer_idx, device=device, batch_size=batch_size, max_prompt_len=max_prompt_len
            )
            v = normalize_vec(mu_with - mu_without)

            vec_path = os.path.join(out_vec_dir, f"instr_vec_{i:04d}_l{layer_idx}.npy")
            np.save(vec_path, v)

            item = dict(
                name=f"INSTR_vec_{i:04d}_l{layer_idx}",
                concept=f"instruction_template_{i % len(INSTRUCTION_TEMPLATES)}",
                method="ActivationSteering-Instr",
                layer=int(layer_idx),
                alpha=float(alpha),
                path=vec_path,
                subset_size=int(subset_size),
                seed=int(seed + 1337),
            )
            mf.write(json.dumps(item) + "\n")

    print(f"[INSTR] Wrote manifest: {out_manifest_path} (n_vec={n_vec})")


# -----------------------------
# SAE training + feature vectors
# -----------------------------
class SparseAutoencoder(torch.nn.Module):
    def __init__(self, d_in: int, d_latent: int):
        super().__init__()
        self.encoder = torch.nn.Linear(d_in, d_latent, bias=True)
        self.decoder = torch.nn.Linear(d_latent, d_in, bias=True)

        # Small init helps stability
        torch.nn.init.normal_(self.encoder.weight, mean=0.0, std=0.01)
        torch.nn.init.zeros_(self.encoder.bias)
        torch.nn.init.normal_(self.decoder.weight, mean=0.0, std=0.01)
        torch.nn.init.zeros_(self.decoder.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_pre = self.encoder(x)
        z = torch.relu(z_pre)
        x_hat = self.decoder(z)
        return x_hat, z


def train_sae(
    X_cpu: torch.Tensor,
    *,
    d_latent: int,
    steps: int,
    batch_size: int,
    lr: float,
    l1_coef: float,
    device: str,
    seed: int,
    log_every: int = 200,
) -> SparseAutoencoder:
    """
    Train a simple ReLU SAE with L1 penalty on latents.
    X_cpu: [N, D] on CPU (float16/float32).
    """
    set_global_seed(seed)
    X = X_cpu  # keep on CPU; move batches to GPU
    N, D = int(X.shape[0]), int(X.shape[1])

    sae = SparseAutoencoder(D, d_latent).to(device)
    opt = torch.optim.AdamW(sae.parameters(), lr=lr)

    # Shuffle indices once per epoch-ish
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)

    def get_batch(step: int) -> torch.Tensor:
        # simple cycling
        start = (step * batch_size) % N
        end = start + batch_size
        if end <= N:
            b = idx[start:end]
        else:
            b = np.concatenate([idx[start:], idx[:(end - N)]], axis=0)
        xb = X[b]
        return xb.to(device=device, dtype=torch.float32)

    sae.train()
    for step in range(steps):
        xb = get_batch(step)
        x_hat, z = sae(xb)
        mse = torch.mean((x_hat - xb) ** 2)
        l1 = torch.mean(torch.abs(z))
        loss = mse + l1_coef * l1

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        opt.step()

        if (step + 1) % log_every == 0 or step == 0 or step == steps - 1:
            with torch.no_grad():
                sparsity = float((z > 0).float().mean().item())
            print(f"[SAE] step {step+1:6d}/{steps}  loss={loss.item():.6f}  mse={mse.item():.6f}  l1={l1.item():.6f}  act_frac={sparsity:.4f}")

    sae.eval()
    return sae


def build_pool_sae_features(
    *,
    model,
    tokenizer,
    corpus: List[Any],
    layer_idx: int,
    n_vec: int,
    alpha: float,
    device: str,
    batch_size: int,
    max_prompt_len: int,
    sae_train_samples: int,
    sae_latent_dim: int,
    sae_steps: int,
    sae_batch_size: int,
    sae_lr: float,
    sae_l1: float,
    out_vec_dir: str,
    out_manifest_path: str,
    seed: int,
    skip_if_exists: bool,
    cache_acts_path: str,
    cache_sae_path: str,
) -> None:
    """
    SAE-based candidate pool: train SAE on layer activations, then use decoder columns as vectors.
    Feature selection: pick features with non-trivial activation frequency; rank by decoder column norm.
    """
    ensure_dir(out_vec_dir)
    ensure_dir(os.path.dirname(out_manifest_path))
    ensure_dir(os.path.dirname(cache_acts_path))
    ensure_dir(os.path.dirname(cache_sae_path))

    if skip_if_exists and os.path.exists(out_manifest_path):
        print(f"[SAE] Manifest exists, skipping build: {out_manifest_path}")
        return

    # 1) collect activations
    if os.path.exists(cache_acts_path):
        print(f"[SAE] Loading cached activations: {cache_acts_path}")
        X_cpu = torch.load(cache_acts_path, map_location="cpu")
        assert isinstance(X_cpu, torch.Tensor)
    else:
        rng = random.Random(seed + 2025)
        # Use plain conditioning prompts (with generation prompt), similar to instruction extraction
        texts = []
        for _ in range(sae_train_samples):
            ex = corpus[rng.randrange(len(corpus))]
            texts.append(render_chat(tokenizer, str(ex.prompt), assistant_content=None, add_generation_prompt=True))
        X_cpu = collect_last_token_activations_matrix(
            model, tokenizer, texts,
            layer_idx=layer_idx, device=device, batch_size=batch_size, max_prompt_len=max_prompt_len,
            max_samples=sae_train_samples, dtype="fp16"
        )
        torch.save(X_cpu, cache_acts_path)
        print(f"[SAE] Saved activations: {cache_acts_path}  shape={tuple(X_cpu.shape)} dtype={X_cpu.dtype}")

    N, D = int(X_cpu.shape[0]), int(X_cpu.shape[1])

    # 2) train SAE (or load)
    if os.path.exists(cache_sae_path):
        print(f"[SAE] Loading cached SAE weights: {cache_sae_path}")
        obj = torch.load(cache_sae_path, map_location="cpu")
        sae = SparseAutoencoder(D, sae_latent_dim)
        sae.load_state_dict(obj["state_dict"])
        sae = sae.to(device)
        sae.eval()
    else:
        # Important: SAE training needs gradients. Keep this explicitly enabled even if callers
        # wrap higher-level routines in `torch.no_grad()`.
        with torch.enable_grad():
            sae = train_sae(
                X_cpu,
                d_latent=sae_latent_dim,
                steps=sae_steps,
                batch_size=sae_batch_size,
                lr=sae_lr,
                l1_coef=sae_l1,
                device=device,
                seed=seed + 2026,
            )
        torch.save({"state_dict": sae.state_dict(), "d_in": D, "d_latent": sae_latent_dim}, cache_sae_path)
        print(f"[SAE] Saved SAE weights: {cache_sae_path}")

    # 3) feature selection
    # Compute activation freq on a small subset for speed
    subN = min(N, 8192)
    with torch.no_grad():
        Xsub = X_cpu[:subN].to(device=device, dtype=torch.float32)
        _, Z = sae(Xsub)
        freq = (Z > 0).float().mean(dim=0).detach().cpu().numpy()  # [M]
        dec_w = sae.decoder.weight.detach().cpu().numpy()          # [D, M]
    col_norm = np.linalg.norm(dec_w, axis=0)                   # [M]

    # Keep features that are neither dead nor always-on.
    good = np.where((freq > 0.001) & (freq < 0.5) & np.isfinite(col_norm) & (col_norm > 1e-6))[0]
    if good.size < n_vec:
        # fallback: relax
        good = np.where((freq > 0.0) & np.isfinite(col_norm) & (col_norm > 1e-8))[0]
    if good.size == 0:
        raise RuntimeError("SAE produced no usable features (all dead?). Try higher sae_steps or lower sae_l1.")

    # Rank by decoder norm (simple, stable)
    order = good[np.argsort(-col_norm[good])]
    pick = order[:n_vec]

    with open(out_manifest_path, "w", encoding="utf-8") as mf:
        for j, feat_id in enumerate(pick):
            v = dec_w[:, int(feat_id)].astype(np.float32)
            v = normalize_vec(v)

            vec_path = os.path.join(out_vec_dir, f"sae_feat_{j:04d}_fid{int(feat_id)}_l{layer_idx}.npy")
            np.save(vec_path, v)

            item = dict(
                name=f"SAE_feat_{j:04d}_fid{int(feat_id)}_l{layer_idx}",
                concept=f"sae_feature_{int(feat_id)}",
                method="SAE-feature",
                layer=int(layer_idx),
                alpha=float(alpha),
                path=vec_path,
                sae_latent_dim=int(sae_latent_dim),
                sae_steps=int(sae_steps),
                sae_l1=float(sae_l1),
                seed=int(seed + 2026),
                freq=float(freq[int(feat_id)]),
                dec_col_norm=float(col_norm[int(feat_id)]),
            )
            mf.write(json.dumps(item) + "\n")

    print(f"[SAE] Wrote manifest: {out_manifest_path} (n_vec={n_vec})")


# -----------------------------
# Rank-flip evaluation (multi-seed)
# -----------------------------
@dataclass
class SteeringVector:
    name: str
    concept: str
    method: str
    layer: int
    alpha: float
    vec: np.ndarray
    path: str


def load_vectors_from_manifest(manifest_path: str) -> List[SteeringVector]:
    vecs: List[SteeringVector] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            it = json.loads(line)
            path = str(it["path"])
            v = np.load(path).astype(np.float32).reshape(-1)
            vecs.append(
                SteeringVector(
                    name=str(it.get("name", os.path.basename(path))),
                    concept=str(it.get("concept", "unknown")),
                    method=str(it.get("method", "unknown")),
                    layer=int(it["layer"]),
                    alpha=float(it.get("alpha", 1.0)),
                    vec=v,
                    path=path,
                )
            )
    if not vecs:
        raise RuntimeError(f"No vectors loaded from {manifest_path}")
    return vecs


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
      - "none":    apply nowhere

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

    for i in tqdm(range(0, len(prompts), batch_size), desc=f"Generate({decoding})"):
        batch_raw = prompts[i:i+batch_size]
        # Re-wrap as chat prompt w/ generation prompt (this matches evaluation)
        batch = [render_chat(tokenizer, p, assistant_content=None, add_generation_prompt=True) for p in batch_raw]

        use_template = bool(getattr(tokenizer, "chat_template", None))
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


@torch.no_grad()
def evaluate_with_steering(
    *,
    model,
    tokenizer,
    examples: List[Any],
    decoding: str,
    max_new_tokens: int,
    reasoning_token_threshold: int,
    temperature: float,
    top_p: float,
    top_k: int,
    device: str,
    batch_size: int,
    max_prompt_len: int,
    steering_vec: Optional[SteeringVector],
    phase_mode: str,   # "prefill" | "decode" | "both" | "none"
    staged: bool,
    sample_seed: Optional[int],
) -> Dict[str, Any]:
    if phase_mode == "none" or steering_vec is None:
        handles, state_setter, hook_stats = [], None, []
    else:
        handles, state_setter, hook_stats = register_steering_hooks(
            model=model,
            layer_idx=steering_vec.layer,
            v_np=steering_vec.vec,
            alpha=steering_vec.alpha,
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
            pred = parse_prediction(ex.dataset, cont)
            correct.append(int(is_correct_bool(ex.dataset, pred, ex.gold)))
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
        }
    finally:
        remove_hooks(handles)


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
    ra = _rankdata(a); rb = _rankdata(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    denom = float(np.sqrt(np.sum(ra * ra) * np.sum(rb * rb)))
    if denom <= 0:
        return float("nan")
    return float(np.sum(ra * rb) / denom)


def agg_task_scores(per_task: Dict[str, float], agg: str) -> float:
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


def load_eval_by_seed(
    *,
    tasks: List[str],
    n_eval: int,
    seed: int,
    template_seed: int,
    template_randomization: bool,
    shuffle_choices: bool,
    add_answer_prefix: bool,
    answer_prefix: str,
) -> Dict[str, List[Any]]:
    import inspect
    sig = inspect.signature(load_selected_tasks)
    kwargs = dict(
        tasks=tasks,
        n_subspace=1,
        n_eval=int(n_eval),
        seed=int(seed),
        template_seed=int(template_seed),
        template_randomization=bool(template_randomization),
        shuffle_choices=bool(shuffle_choices),
        add_answer_prefix=bool(add_answer_prefix),
        answer_prefix=str(answer_prefix),
    )
    kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    _sub_by, eval_by, _meta = load_selected_tasks(**kwargs)
    return eval_by


def run_rankflip_for_pool(
    *,
    pool_name: str,
    manifest_path: str,
    out_json_path: str,
    model,
    tokenizer,
    tasks: List[str],
    n_eval: int,
    template_seeds_rank: List[int],
    template_seeds_real: List[int],
    template_randomization: bool,
    shuffle_choices: bool,
    add_answer_prefix: bool,
    answer_prefix: str,
    decoding: str,
    max_new_tokens: int,
    reasoning_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    device: str,
    batch_size: int,
    max_prompt_len: int,
    staged: bool,
    trad_mode: str,
    decode_mode: str,
    agg: str,
    sample_seed: int,
    seed: int,
) -> Dict[str, Any]:
    """
    Returns a result dict (also saved to out_json_path):
      - per-vector mean/std scores for TRAD/DECODE/REAL
      - correlations
      - selection metrics (regret@1, top10 overlap)
    """
    set_global_seed(seed)

    vecs = load_vectors_from_manifest(manifest_path)
    hid_dim = infer_hidden_dim(model)
    if hid_dim is not None:
        for v in vecs:
            if v.vec.shape[0] != hid_dim:
                raise ValueError(f"[{pool_name}] dim mismatch for {v.name}: {v.vec.shape[0]} != {hid_dim}")

    # Pre-load eval sets and baselines per seed
    def baseline_for_seed(eval_by_task: Dict[str, List[Any]]) -> Dict[str, float]:
        base = {}
        for t in tasks:
            res = evaluate_with_steering(
                model=model, tokenizer=tokenizer, examples=eval_by_task[t],
                decoding=decoding, max_new_tokens=max_new_tokens,
                reasoning_token_threshold=reasoning_tokens,
                temperature=temperature, top_p=top_p, top_k=top_k,
                device=device, batch_size=batch_size, max_prompt_len=max_prompt_len,
                steering_vec=None, phase_mode="none", staged=False,
                sample_seed=(sample_seed if decoding == "sample" else None),
            )
            base[t] = float(res["accuracy"])
        return base

    eval_rank = {}
    base_rank = {}
    for s in template_seeds_rank:
        eval_by = load_eval_by_seed(
            tasks=tasks, n_eval=n_eval, seed=seed, template_seed=s,
            template_randomization=template_randomization, shuffle_choices=shuffle_choices,
            add_answer_prefix=add_answer_prefix, answer_prefix=answer_prefix,
        )
        eval_rank[s] = eval_by
        base_rank[s] = baseline_for_seed(eval_by)

    eval_real = {}
    base_real = {}
    for s in template_seeds_real:
        eval_by = load_eval_by_seed(
            tasks=tasks, n_eval=n_eval, seed=seed, template_seed=s,
            template_randomization=template_randomization, shuffle_choices=shuffle_choices,
            add_answer_prefix=add_answer_prefix, answer_prefix=answer_prefix,
        )
        eval_real[s] = eval_by
        base_real[s] = baseline_for_seed(eval_by)

    # Evaluate each vector
    per_vec: Dict[str, Any] = {}
    scores_trad = []
    scores_decode = []
    scores_real = []
    names = []

    for sv in vecs:
        # Per seed scalar score (aggregated across tasks)
        trad_seed_scores = []
        decode_seed_scores = []
        real_seed_scores = []

        # TRAD rank seeds
        for s in template_seeds_rank:
            per_task = {}
            for t in tasks:
                res = evaluate_with_steering(
                    model=model, tokenizer=tokenizer, examples=eval_rank[s][t],
                    decoding=decoding, max_new_tokens=max_new_tokens,
                    reasoning_token_threshold=reasoning_tokens,
                    temperature=temperature, top_p=top_p, top_k=top_k,
                    device=device, batch_size=batch_size, max_prompt_len=max_prompt_len,
                    steering_vec=sv, phase_mode=trad_mode, staged=staged,
                    sample_seed=(sample_seed if decoding == "sample" else None),
                )
                per_task[t] = float(res["accuracy"] - base_rank[s][t])
            trad_seed_scores.append(agg_task_scores(per_task, agg))

        # DECODE rank seeds
        for s in template_seeds_rank:
            per_task = {}
            for t in tasks:
                res = evaluate_with_steering(
                    model=model, tokenizer=tokenizer, examples=eval_rank[s][t],
                    decoding=decoding, max_new_tokens=max_new_tokens,
                    reasoning_token_threshold=reasoning_tokens,
                    temperature=temperature, top_p=top_p, top_k=top_k,
                    device=device, batch_size=batch_size, max_prompt_len=max_prompt_len,
                    steering_vec=sv, phase_mode=decode_mode, staged=staged,
                    sample_seed=(sample_seed if decoding == "sample" else None),
                )
                per_task[t] = float(res["accuracy"] - base_rank[s][t])
            decode_seed_scores.append(agg_task_scores(per_task, agg))

        # REAL held-out seeds (decode-only always)
        for s in template_seeds_real:
            per_task = {}
            for t in tasks:
                res = evaluate_with_steering(
                    model=model, tokenizer=tokenizer, examples=eval_real[s][t],
                    decoding=decoding, max_new_tokens=max_new_tokens,
                    reasoning_token_threshold=reasoning_tokens,
                    temperature=temperature, top_p=top_p, top_k=top_k,
                    device=device, batch_size=batch_size, max_prompt_len=max_prompt_len,
                    steering_vec=sv, phase_mode="decode", staged=staged,
                    sample_seed=(sample_seed if decoding == "sample" else None),
                )
                per_task[t] = float(res["accuracy"] - base_real[s][t])
            real_seed_scores.append(agg_task_scores(per_task, agg))

        s_trad_mean = float(np.mean(trad_seed_scores)); s_trad_std = float(np.std(trad_seed_scores, ddof=0))
        s_dec_mean  = float(np.mean(decode_seed_scores)); s_dec_std  = float(np.std(decode_seed_scores, ddof=0))
        s_real_mean = float(np.mean(real_seed_scores)); s_real_std = float(np.std(real_seed_scores, ddof=0))

        per_vec[sv.name] = dict(
            method=sv.method,
            concept=sv.concept,
            layer=sv.layer,
            alpha=sv.alpha,
            path=sv.path,
            score_rank_trad_mean=s_trad_mean,
            score_rank_trad_std=s_trad_std,
            score_rank_decode_mean=s_dec_mean,
            score_rank_decode_std=s_dec_std,
            score_real_mean=s_real_mean,
            score_real_std=s_real_std,
        )
        names.append(sv.name)
        scores_trad.append(s_trad_mean)
        scores_decode.append(s_dec_mean)
        scores_real.append(s_real_mean)

    scores_trad = np.array(scores_trad, dtype=np.float64)
    scores_decode = np.array(scores_decode, dtype=np.float64)
    scores_real = np.array(scores_real, dtype=np.float64)

    rho_trad_decode = spearmanr(scores_trad, scores_decode)
    rho_trad_real   = spearmanr(scores_trad, scores_real)
    rho_decode_real = spearmanr(scores_decode, scores_real)

    # selection metrics
    best_real = float(np.max(scores_real))
    idx_trad = int(np.argmax(scores_trad))
    idx_dec  = int(np.argmax(scores_decode))
    regret_trad = float(best_real - scores_real[idx_trad])
    regret_dec  = float(best_real - scores_real[idx_dec])

    def topk_idx(x: np.ndarray, k: int) -> List[int]:
        return list(np.argsort(-x)[:min(k, x.shape[0])])

    top10_trad = set(topk_idx(scores_trad, 10))
    top10_dec  = set(topk_idx(scores_decode, 10))
    top10_real = set(topk_idx(scores_real, 10))

    overlap_trad_real = int(len(top10_trad & top10_real))
    overlap_dec_real  = int(len(top10_dec & top10_real))

    result = dict(
        pool=pool_name,
        manifest=manifest_path,
        config=dict(
            tasks=tasks,
            n_eval=n_eval,
            template_seeds_rank=template_seeds_rank,
            template_seeds_real=template_seeds_real,
            decoding=decoding,
            max_new_tokens=max_new_tokens,
            reasoning_tokens=reasoning_tokens,
            staged=staged,
            trad_mode=trad_mode,
            decode_mode=decode_mode,
            agg=agg,
        ),
        correlations=dict(
            spearman_trad_vs_decode=float(rho_trad_decode),
            spearman_trad_vs_real=float(rho_trad_real),
            spearman_decode_vs_real=float(rho_decode_real),
        ),
        selection=dict(
            best_real=float(best_real),
            chosen_by_trad=names[idx_trad],
            chosen_by_decode=names[idx_dec],
            regret_at_1_trad=float(regret_trad),
            regret_at_1_decode=float(regret_dec),
            top10_overlap_trad_real=overlap_trad_real,
            top10_overlap_decode_real=overlap_dec_real,
        ),
        vectors=per_vec,
    )

    ensure_dir(os.path.dirname(out_json_path))
    json_dump(result, out_json_path)
    print(f"[{pool_name}] Saved rankflip JSON: {out_json_path}")

    # quick print
    print(f"\n[{pool_name}] Spearman(trad, real)={rho_trad_real:.3f}  Spearman(decode, real)={rho_decode_real:.3f}  (n={len(names)})")
    print(f"[{pool_name}] regret@1 TRAD={regret_trad:+.4f}  DECODE={regret_dec:+.4f}  top10_overlap TRAD={overlap_trad_real}  DECODE={overlap_dec_real}")

    return result


def write_summary_md(all_results: List[Dict[str, Any]], out_path: str) -> None:
    lines = []
    lines.append("# A1 Cross-method candidate pools: RankFlip summary\n")
    lines.append("This table summarizes protocol-level ranking generalization under KV-cached decode.\n")
    lines.append("Metrics are computed per pool using mean scores across template seeds.\n")

    lines.append("| Pool | #Vec | Spearman(TRAD,REAL) | Spearman(DECODE,REAL) | Spearman(TRAD,DECODE) | regret@1 TRAD | regret@1 DECODE | top10 overlap TRAD∩REAL | top10 overlap DECODE∩REAL |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    for r in all_results:
        n = len(r["vectors"])
        c = r["correlations"]
        s = r["selection"]
        lines.append(
            f"| {r['pool']} | {n} | {c['spearman_trad_vs_real']:.3f} | {c['spearman_decode_vs_real']:.3f} | {c['spearman_trad_vs_decode']:.3f} | "
            f"{s['regret_at_1_trad']:.4f} | {s['regret_at_1_decode']:.4f} | {s['top10_overlap_trad_real']} | {s['top10_overlap_decode_real']} |"
        )

    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[Summary] Wrote: {out_path}")


# -----------------------------
# Main
# -----------------------------
def parse_int_list(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model_dtype", type=str, default="fp16", choices=["fp16", "fp32"])

    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--alpha", type=float, default=1.0, help="Global steering scale applied to all vectors (vectors are normalized).")

    ap.add_argument("--tasks_eval", type=str, default="commonsenseqa,arc_challenge,openbookqa,qasc,logiqa")
    ap.add_argument("--n_eval", type=int, default=128)

    ap.add_argument("--template_seeds_rank", type=str, default="1234,2345,3456")
    ap.add_argument("--template_seeds_real", type=str, default="4567,5678,6789")
    ap.add_argument("--template_randomization", type=int, default=1, choices=[0, 1])
    ap.add_argument("--shuffle_choices", type=int, default=1, choices=[0, 1])
    ap.add_argument("--answer_prefix", type=str, default="\nFinal answer:")
    ap.add_argument("--add_answer_prefix", type=int, default=1, choices=[0, 1])

    # Decoding protocol
    ap.add_argument("--decoding", type=str, default="greedy", choices=["greedy", "sample"])
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--reasoning_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_prompt_len", type=int, default=512)
    ap.add_argument("--sample_seed", type=int, default=12345)

    ap.add_argument("--staged", type=int, default=1, choices=[0, 1])
    ap.add_argument("--trad_mode", type=str, default="prefill", choices=["prefill", "both"])
    ap.add_argument("--decode_mode", type=str, default="decode", choices=["decode", "both"])
    ap.add_argument("--agg", type=str, default="mean", choices=["mean", "min", "median"])

    # Pool sizes
    ap.add_argument("--n_vec", type=int, default=64, help="Vectors per method/pool")
    ap.add_argument("--n_vec_caa", type=int, default=0, help="CAA vectors (0 = use --n_vec)")
    ap.add_argument("--n_vec_instr", type=int, default=0, help="Instruction vectors (0 = use --n_vec)")
    ap.add_argument("--n_vec_sae", type=int, default=0, help="SAE vectors (0 = use --n_vec)")
    ap.add_argument("--subset_size", type=int, default=96, help="Bootstrap subset size per vector (CAA / INSTR extraction).")
    ap.add_argument("--n_corpus", type=int, default=512, help="Corpus examples per task for vector extraction.")

    # SAE params
    ap.add_argument("--sae_train_samples", type=int, default=20000)
    ap.add_argument("--sae_latent_dim", type=int, default=8192)
    ap.add_argument("--sae_steps", type=int, default=3000)
    ap.add_argument("--sae_batch_size", type=int, default=256)
    ap.add_argument("--sae_lr", type=float, default=2e-4)
    ap.add_argument("--sae_l1", type=float, default=1e-3)

    ap.add_argument("--out_dir", type=str, default="outputs/steering_rank_flip/cross_method")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_if_exists", type=int, default=1, choices=[0, 1])

    args = ap.parse_args()
    set_global_seed(args.seed)

    out_dir = ensure_dir(args.out_dir)
    vec_root = ensure_dir(os.path.join(out_dir, "vectors"))
    man_root = ensure_dir(os.path.join(out_dir, "manifests"))
    rf_root  = ensure_dir(os.path.join(out_dir, "rankflip"))

    tasks = [t.strip() for t in args.tasks_eval.split(",") if t.strip()]
    if not tasks:
        raise ValueError("Empty --tasks_eval")

    rank_seeds = parse_int_list(args.template_seeds_rank)
    real_seeds = parse_int_list(args.template_seeds_real)
    if not rank_seeds or not real_seeds:
        raise ValueError("template_seeds_rank and template_seeds_real must be non-empty comma-separated lists.")

    # Load model once
    model, tokenizer = load_model_and_tokenizer(args.model, args.device, args.model_dtype)
    hid_dim = infer_hidden_dim(model)
    print(f"[Model] {args.model}  hidden_dim={hid_dim}  layer={args.layer}  dtype={args.model_dtype}")

    # Build corpus (from subspace prompts)
    corpus = load_corpus_examples(
        tasks=tasks,
        n_subspace=args.n_corpus,
        seed=args.seed,
        template_seed=0,
        template_randomization=bool(args.template_randomization),
        shuffle_choices=bool(args.shuffle_choices),
        add_answer_prefix=bool(args.add_answer_prefix),
        answer_prefix=args.answer_prefix,
    )
    print(f"[Corpus] loaded {len(corpus)} examples (tasks={tasks}, per_task≈{args.n_corpus})")

    # ---------------------------------------
    # Build three pools
    # ---------------------------------------
    skip = bool(args.skip_if_exists)
    layer = int(args.layer)
    alpha = float(args.alpha)
    n_vec_caa = int(args.n_vec_caa) if int(args.n_vec_caa) > 0 else int(args.n_vec)
    n_vec_instr = int(args.n_vec_instr) if int(args.n_vec_instr) > 0 else int(args.n_vec)
    n_vec_sae = int(args.n_vec_sae) if int(args.n_vec_sae) > 0 else int(args.n_vec)
    print(f"[Pools] n_vec: CAA={n_vec_caa} INSTR={n_vec_instr} SAE={n_vec_sae}")

    # (1) CAA
    caa_vec_dir = ensure_dir(os.path.join(vec_root, "caa"))
    caa_manifest = os.path.join(man_root, "caa.jsonl")
    build_pool_caa(
        model=model, tokenizer=tokenizer, corpus=corpus, layer_idx=layer,
        n_vec=n_vec_caa, subset_size=args.subset_size, answer_prefix=args.answer_prefix,
        alpha=alpha, device=args.device, batch_size=args.batch_size, max_prompt_len=args.max_prompt_len,
        out_vec_dir=caa_vec_dir, out_manifest_path=caa_manifest,
        seed=args.seed, skip_if_exists=skip
    )

    # (2) Instruction activation steering
    instr_vec_dir = ensure_dir(os.path.join(vec_root, "instr"))
    instr_manifest = os.path.join(man_root, "instr.jsonl")
    build_pool_instruction(
        model=model, tokenizer=tokenizer, corpus=corpus, layer_idx=layer,
        n_vec=n_vec_instr, subset_size=args.subset_size, alpha=alpha,
        device=args.device, batch_size=args.batch_size, max_prompt_len=args.max_prompt_len,
        out_vec_dir=instr_vec_dir, out_manifest_path=instr_manifest,
        seed=args.seed, skip_if_exists=skip
    )

    # (3) SAE-based
    sae_vec_dir = ensure_dir(os.path.join(vec_root, "sae"))
    sae_manifest = os.path.join(man_root, "sae.jsonl")
    cache_acts = os.path.join(out_dir, "cache", f"acts_l{layer}_n{args.sae_train_samples}.pt")
    cache_sae = os.path.join(out_dir, "cache", f"sae_l{layer}_M{args.sae_latent_dim}_steps{args.sae_steps}.pt")
    build_pool_sae_features(
        model=model, tokenizer=tokenizer, corpus=corpus, layer_idx=layer,
        n_vec=n_vec_sae, alpha=alpha, device=args.device, batch_size=args.batch_size, max_prompt_len=args.max_prompt_len,
        sae_train_samples=args.sae_train_samples, sae_latent_dim=args.sae_latent_dim, sae_steps=args.sae_steps,
        sae_batch_size=args.sae_batch_size, sae_lr=args.sae_lr, sae_l1=args.sae_l1,
        out_vec_dir=sae_vec_dir, out_manifest_path=sae_manifest,
        seed=args.seed, skip_if_exists=skip,
        cache_acts_path=cache_acts, cache_sae_path=cache_sae,
    )

    # ---------------------------------------
    # Run rank-flip for each pool
    # ---------------------------------------
    all_results: List[Dict[str, Any]] = []
    staged = bool(args.staged)

    for pool_name, manifest in [("CAA", caa_manifest), ("INSTR", instr_manifest), ("SAE", sae_manifest)]:
        out_json = os.path.join(rf_root, f"rankflip_{pool_name.lower()}.json")
        res = run_rankflip_for_pool(
            pool_name=pool_name,
            manifest_path=manifest,
            out_json_path=out_json,
            model=model,
            tokenizer=tokenizer,
            tasks=tasks,
            n_eval=args.n_eval,
            template_seeds_rank=rank_seeds,
            template_seeds_real=real_seeds,
            template_randomization=bool(args.template_randomization),
            shuffle_choices=bool(args.shuffle_choices),
            add_answer_prefix=bool(args.add_answer_prefix),
            answer_prefix=args.answer_prefix,
            decoding=args.decoding,
            max_new_tokens=args.max_new_tokens,
            reasoning_tokens=args.reasoning_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            device=args.device,
            batch_size=args.batch_size,
            max_prompt_len=args.max_prompt_len,
            staged=staged,
            trad_mode=args.trad_mode,
            decode_mode=args.decode_mode,
            agg=args.agg,
            sample_seed=args.sample_seed,
            seed=args.seed,
        )
        all_results.append(res)

    # Write summary
    summary_md = os.path.join(out_dir, "summary.md")
    write_summary_md(all_results, summary_md)

    print("\n[Done] A1 cross-method rank-flip complete.")
    print(f"  - Summary: {summary_md}")
    print(f"  - Rankflip JSONs: {rf_root}")
    print(f"  - Manifests: {man_root}")
    print(f"  - Vectors: {vec_root}")


if __name__ == "__main__":
    main()
