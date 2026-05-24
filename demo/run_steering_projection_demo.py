#!/usr/bin/env python3

"""Small DecodeShare demo: project a steering vector away from a decode-shared basis.

The full paper experiments are large. This script is deliberately compact:
it uses a Llama-style model, estimates a small decode-time PCA basis from short
KV-cached rollouts, builds one contrastive steering vector, and writes an HTML
report showing how much of that vector lies in the shared decode channel.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


torch = None
AutoModelForCausalLM = None
AutoTokenizer = None


CALIBRATION_PROMPTS: List[Tuple[str, str]] = [
    ("math", "If a train travels 60 miles in 1.5 hours, what is its average speed?"),
    ("math", "A store gives a 20 percent discount on a 45 dollar item. What is the new price?"),
    ("commonsense", "Why might a person carry an umbrella on a cloudy morning?"),
    ("commonsense", "Which object is more useful for cutting paper, a spoon or scissors?"),
    ("logic", "If all bloops are razzes and all razzes are lazzes, are all bloops lazzes?"),
    ("logic", "Mina is older than Jo. Jo is older than Kai. Who is youngest?"),
    ("knowledge", "What gas do plants absorb from the air during photosynthesis?"),
    ("knowledge", "Name the largest planet in our solar system."),
    ("format", "Answer with a single letter: Which comes first alphabetically, B or D?"),
    ("format", "Answer with a short phrase: What color do you get by mixing red and white?"),
]

STEERING_TEXTS: List[str] = [
    "Explain why the ocean has tides.",
    "Describe how to make a cup of tea.",
    "Give one reason exercise can improve health.",
    "Explain what a library is used for.",
    "Describe what happens when water freezes.",
    "Give a short answer about why sleep matters.",
]

EVAL_PROMPTS: List[str] = [
    "Explain why the moon appears to change shape over a month.",
    "Give practical advice for staying focused while studying.",
]

RANKING_ALIGNMENT_ROWS: List[Dict[str, Any]] = [
    {"pool": "CAA contrastive", "prefill_rho": -0.370, "decode_rho": 0.700, "delta": 1.070},
    {"pool": "Instruction", "prefill_rho": 0.172, "decode_rho": 0.767, "delta": 0.595},
    {"pool": "SAE features", "prefill_rho": -0.064, "decode_rho": 0.594, "delta": 0.659},
    {"pool": "Diagnostic", "prefill_rho": 0.065, "decode_rho": 0.700, "delta": 0.635},
]

DEPLOYMENT_SELECTION_ROWS: List[Dict[str, Any]] = [
    {
        "proxy": "Prefill-aligned",
        "real_mean": -0.002,
        "real_worst": -0.003,
        "flip_rate": 0.750,
        "regret_at_1": 0.016,
    },
    {
        "proxy": "Mixed stages",
        "real_mean": -0.002,
        "real_worst": -0.003,
        "flip_rate": 0.750,
        "regret_at_1": 0.016,
    },
    {
        "proxy": "Decode-aligned",
        "real_mean": 0.011,
        "real_worst": 0.010,
        "flip_rate": 0.083,
        "regret_at_1": 0.003,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--layer", type=int, default=28)
    parser.add_argument("--basis_k", type=int, default=32)
    parser.add_argument("--calib_max_new_tokens", type=int, default=8)
    parser.add_argument("--steer_max_new_tokens", type=int, default=8)
    parser.add_argument("--eval_max_new_tokens", type=int, default=80)
    parser.add_argument("--max_prompt_tokens", type=int, default=384)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--inject_first_n", type=int, default=20)
    parser.add_argument("--demo_vector_mode", choices=["caa", "caa_plus_shared"], default="caa_plus_shared")
    parser.add_argument("--shared_component_scale", type=float, default=4.0)
    parser.add_argument("--preserve_residual_norm", action="store_true")
    parser.add_argument("--system", default="You are a helpful assistant.")
    parser.add_argument("--positive_style", default="Reply in a vivid pirate voice.")
    parser.add_argument("--negative_style", default="Reply in a concise neutral voice.")
    parser.add_argument("--out_dir", default="outputs/demo_steering_projection")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def load_runtime_dependencies() -> None:
    global torch, AutoModelForCausalLM, AutoTokenizer
    import torch as _torch
    from transformers import AutoModelForCausalLM as _AutoModelForCausalLM
    from transformers import AutoTokenizer as _AutoTokenizer

    torch = _torch
    AutoModelForCausalLM = _AutoModelForCausalLM
    AutoTokenizer = _AutoTokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_dtype(name: str):
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


def load_model_and_tokenizer(args: argparse.Namespace):
    dtype = torch_dtype(args.dtype)
    kwargs = {
        "local_files_only": bool(args.local_files_only),
        "trust_remote_code": bool(args.trust_remote_code),
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=bool(args.local_files_only),
        trust_remote_code=bool(args.trust_remote_code),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.to(args.device)
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    return model, tokenizer


def get_layers(model) -> Sequence[Any]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return list(model.gpt_neox.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    if hasattr(model, "transformer") and hasattr(model.transformer, "blocks"):
        return list(model.transformer.blocks)
    raise RuntimeError(f"Cannot locate transformer layers for {type(model)}")


def get_layer(model, layer_idx: int):
    layers = get_layers(model)
    if layer_idx < 0 or layer_idx >= len(layers):
        raise ValueError(f"--layer {layer_idx} out of range for model with {len(layers)} layers")
    return layers[layer_idx]


def model_device(model):
    return next(model.parameters()).device


def format_prompt(tokenizer, user_text: str, system_text: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        messages = []
        if system_text:
            messages.append({"role": "system", "content": system_text})
        messages.append({"role": "user", "content": user_text})
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": user_text}],
                tokenize=False,
                add_generation_prompt=True,
            )
    if system_text:
        return f"{system_text}\n\nUser: {user_text}\nAssistant:"
    return f"User: {user_text}\nAssistant:"


def tokenize_prompt(tokenizer, text: str, max_prompt_tokens: int):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_prompt_tokens)
    return enc["input_ids"]


class DecodeStateCollector:
    def __init__(self) -> None:
        self.records: List[Any] = []
        self.prefill_calls = 0
        self.decode_calls = 0

    def __call__(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if not hasattr(hidden, "ndim") or hidden.ndim != 3:
            return output
        if int(hidden.shape[1]) == 1:
            self.decode_calls += 1
            self.records.append(hidden[:, -1, :].detach().float().cpu())
        else:
            self.prefill_calls += 1
        return output


class AddDecodeVectorHook:
    def __init__(self, vector, alpha: float, inject_first_n: int) -> None:
        self.vector = vector.detach().float().cpu()
        self.alpha = float(alpha)
        self.inject_first_n = int(inject_first_n)
        self.decode_calls = 0
        self.applied = 0

    def __call__(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if not hasattr(hidden, "ndim") or hidden.ndim != 3 or int(hidden.shape[1]) != 1:
            return output
        self.decode_calls += 1
        if self.inject_first_n > 0 and self.applied >= self.inject_first_n:
            return output

        vector = self.vector.to(device=hidden.device, dtype=torch.float32)
        patched = hidden.clone()
        patched[:, -1, :] = (patched[:, -1, :].float() + self.alpha * vector).to(hidden.dtype)
        self.applied += 1
        if isinstance(output, tuple):
            return (patched,) + output[1:]
        return patched


def decode_rollout(
    model,
    tokenizer,
    prompt_text: str,
    *,
    max_prompt_tokens: int,
    max_new_tokens: int,
) -> str:
    device = model_device(model)
    input_ids = tokenize_prompt(tokenizer, prompt_text, max_prompt_tokens).to(device)
    if input_ids.shape[1] < 1:
        raise RuntimeError("Tokenizer produced an empty prompt.")

    generated: List[int] = []
    with torch.inference_mode():
        if input_ids.shape[1] > 1:
            prefill = model(input_ids=input_ids[:, :-1], use_cache=True)
            past = prefill.past_key_values
            current = input_ids[:, -1:]
        else:
            past = None
            current = input_ids

        for _ in range(int(max_new_tokens)):
            out = model(input_ids=current, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_id = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            token_id = int(next_id.item())
            if tokenizer.eos_token_id is not None and token_id == int(tokenizer.eos_token_id):
                break
            generated.append(token_id)
            current = next_id

    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def decode_rollout_stream(
    model,
    tokenizer,
    prompt_text: str,
    *,
    max_prompt_tokens: int,
    max_new_tokens: int,
):
    device = model_device(model)
    input_ids = tokenize_prompt(tokenizer, prompt_text, max_prompt_tokens).to(device)
    if input_ids.shape[1] < 1:
        raise RuntimeError("Tokenizer produced an empty prompt.")

    generated: List[int] = []
    with torch.inference_mode():
        if input_ids.shape[1] > 1:
            prefill = model(input_ids=input_ids[:, :-1], use_cache=True)
            past = prefill.past_key_values
            current = input_ids[:, -1:]
        else:
            past = None
            current = input_ids

        for _ in range(int(max_new_tokens)):
            out = model(input_ids=current, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_id = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            token_id = int(next_id.item())
            if tokenizer.eos_token_id is not None and token_id == int(tokenizer.eos_token_id):
                break
            generated.append(token_id)
            current = next_id
            text = tokenizer.decode(generated, skip_special_tokens=True).strip()
            yield {
                "text": text,
                "token_count": len(generated),
            }


def collect_decode_states(
    model,
    tokenizer,
    prompts: Iterable[str],
    *,
    layer: int,
    max_prompt_tokens: int,
    max_new_tokens: int,
) -> Tuple[Any, Dict[str, int]]:
    block = get_layer(model, layer)
    collector = DecodeStateCollector()
    handle = block.register_forward_hook(collector)
    try:
        for prompt in prompts:
            _ = decode_rollout(
                model,
                tokenizer,
                prompt,
                max_prompt_tokens=max_prompt_tokens,
                max_new_tokens=max_new_tokens,
            )
    finally:
        handle.remove()
    if not collector.records:
        raise RuntimeError("No decode-time states were collected. Try --calib_max_new_tokens >= 2.")
    states = torch.cat(collector.records, dim=0).float()
    return states, {"prefill_calls": collector.prefill_calls, "decode_calls": collector.decode_calls}


def make_calibration_prompts(tokenizer, system_text: str) -> List[Tuple[str, str]]:
    return [(task, format_prompt(tokenizer, prompt, system_text)) for task, prompt in CALIBRATION_PROMPTS]


def make_steering_prompts(tokenizer, args: argparse.Namespace) -> Tuple[List[str], List[str]]:
    positive, negative = [], []
    for text in STEERING_TEXTS:
        positive.append(format_prompt(tokenizer, f"{args.positive_style}\n\n{text}", args.system))
        negative.append(format_prompt(tokenizer, f"{args.negative_style}\n\n{text}", args.system))
    return positive, negative


def estimate_shared_basis(
    model,
    tokenizer,
    args: argparse.Namespace,
) -> Tuple[Any, Dict[str, Any]]:
    grouped_states = []
    group_sizes: Dict[str, int] = {}
    for task, prompt in make_calibration_prompts(tokenizer, args.system):
        states, _ = collect_decode_states(
            model,
            tokenizer,
            [prompt],
            layer=args.layer,
            max_prompt_tokens=args.max_prompt_tokens,
            max_new_tokens=args.calib_max_new_tokens,
        )
        states = states - states.mean(dim=0, keepdim=True)
        grouped_states.append(states)
        group_sizes[task] = group_sizes.get(task, 0) + int(states.shape[0])

    X = torch.cat(grouped_states, dim=0).float()
    X = X - X.mean(dim=0, keepdim=True)
    q = int(min(args.basis_k, X.shape[0] - 1, X.shape[1]))
    if q < 1:
        raise RuntimeError(f"Not enough states for basis estimation: shape={tuple(X.shape)}")

    _, singular_values, vh = torch.linalg.svd(X, full_matrices=False)
    basis = vh[:q].T.contiguous()
    basis, _ = torch.linalg.qr(basis, mode="reduced")
    var = singular_values.pow(2)
    explained = float((var[:q].sum() / (var.sum() + 1e-12)).item())
    info = {
        "n_states": int(X.shape[0]),
        "hidden_dim": int(X.shape[1]),
        "basis_dim": int(q),
        "explained_variance_demo": explained,
        "group_sizes": group_sizes,
    }
    return basis.cpu().float(), info


def estimate_steering_vector(model, tokenizer, args: argparse.Namespace) -> Tuple[Any, Dict[str, Any]]:
    positive, negative = make_steering_prompts(tokenizer, args)
    pos_states, pos_stats = collect_decode_states(
        model,
        tokenizer,
        positive,
        layer=args.layer,
        max_prompt_tokens=args.max_prompt_tokens,
        max_new_tokens=args.steer_max_new_tokens,
    )
    neg_states, neg_stats = collect_decode_states(
        model,
        tokenizer,
        negative,
        layer=args.layer,
        max_prompt_tokens=args.max_prompt_tokens,
        max_new_tokens=args.steer_max_new_tokens,
    )
    vector = pos_states.mean(dim=0) - neg_states.mean(dim=0)
    norm = float(vector.norm().item())
    if not math.isfinite(norm) or norm <= 1e-8:
        raise RuntimeError("Estimated steering vector has near-zero norm.")
    info = {
        "source": "CAA-style contrastive mean-difference",
        "positive_states": int(pos_states.shape[0]),
        "negative_states": int(neg_states.shape[0]),
        "positive_decode_calls": int(pos_stats["decode_calls"]),
        "negative_decode_calls": int(neg_stats["decode_calls"]),
        "vector_norm": norm,
    }
    return vector.cpu().float(), info


def project_vector(basis, vector) -> Dict[str, Any]:
    basis = basis.float()
    vector = vector.float()
    coeff = basis.T @ vector
    shared = basis @ coeff
    residual = vector - shared
    residual_preserve_norm = residual / (residual.norm() + 1e-12) * (vector.norm() + 1e-12)
    residual_overlap = float((basis.T @ residual).norm().item() / (residual.norm().item() + 1e-12))
    residual_preserve_overlap = float(
        (basis.T @ residual_preserve_norm).norm().item()
        / (residual_preserve_norm.norm().item() + 1e-12)
    )
    return {
        "shared": shared.cpu().float(),
        "residual": residual.cpu().float(),
        "residual_preserve_norm": residual_preserve_norm.cpu().float(),
        "overlap_original": float(shared.norm().item() / (vector.norm().item() + 1e-12)),
        "overlap_residual": residual_overlap,
        "overlap_residual_preserve_norm": residual_preserve_overlap,
        "shared_norm": float(shared.norm().item()),
        "residual_norm": float(residual.norm().item()),
        "residual_preserve_norm_value": float(residual_preserve_norm.norm().item()),
    }


def build_demo_vector(basis, caa_vector, args: argparse.Namespace) -> Tuple[Any, Any, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    caa_projection = project_vector(basis, caa_vector)
    if args.demo_vector_mode == "caa":
        demo_vector = caa_vector.float()
        description = "raw CAA-style contrastive mean-difference vector"
    else:


        demo_vector = caa_projection["residual"] + float(args.shared_component_scale) * caa_projection["shared"]
        description = (
            "CAA-style vector with its shared component amplified "
            f"{float(args.shared_component_scale):.2f}x for visual contrast"
        )

    demo_projection = project_vector(basis, demo_vector)
    if bool(args.preserve_residual_norm):
        projected_vector = demo_projection["residual_preserve_norm"]
        norm_policy = "preserve original demo-vector norm"
    else:
        projected_vector = demo_projection["residual"]
        norm_policy = "raw residual norm"

    info = {
        "mode": args.demo_vector_mode,
        "description": description,
        "shared_component_scale": float(args.shared_component_scale),
        "projected_norm_policy": norm_policy,
        "base_caa_overlap": float(caa_projection["overlap_original"]),
        "demo_vector_norm": float(demo_vector.norm().item()),
        "projected_vector_norm": float(projected_vector.norm().item()),
    }
    return demo_vector.cpu().float(), projected_vector.cpu().float(), demo_projection, caa_projection, info


def generate_with_optional_vector(
    model,
    tokenizer,
    user_prompt: str,
    args: argparse.Namespace,
    vector=None,
    alpha: Optional[float] = None,
) -> Dict[str, Any]:
    prompt = format_prompt(tokenizer, user_prompt, args.system)
    hook = None
    handle = None
    if vector is not None:
        hook = AddDecodeVectorHook(vector, alpha=float(args.alpha if alpha is None else alpha), inject_first_n=args.inject_first_n)
        handle = get_layer(model, args.layer).register_forward_hook(hook)
    try:
        text = decode_rollout(
            model,
            tokenizer,
            prompt,
            max_prompt_tokens=args.max_prompt_tokens,
            max_new_tokens=args.eval_max_new_tokens,
        )
    finally:
        if handle is not None:
            handle.remove()
    return {
        "text": text,
        "decode_calls": int(hook.decode_calls) if hook is not None else 0,
        "hook_applications": int(hook.applied) if hook is not None else 0,
    }


def generate_with_optional_vector_stream(
    model,
    tokenizer,
    user_prompt: str,
    args: argparse.Namespace,
    vector=None,
    alpha: Optional[float] = None,
):
    prompt = format_prompt(tokenizer, user_prompt, args.system)
    hook = None
    handle = None
    if vector is not None:
        hook = AddDecodeVectorHook(vector, alpha=float(args.alpha if alpha is None else alpha), inject_first_n=args.inject_first_n)
        handle = get_layer(model, args.layer).register_forward_hook(hook)
    seen = False
    try:
        for step in decode_rollout_stream(
            model,
            tokenizer,
            prompt,
            max_prompt_tokens=args.max_prompt_tokens,
            max_new_tokens=args.eval_max_new_tokens,
        ):
            seen = True
            yield {
                "text": step["text"],
                "token_count": int(step["token_count"]),
                "decode_calls": int(hook.decode_calls) if hook is not None else 0,
                "hook_applications": int(hook.applied) if hook is not None else 0,
            }
    finally:
        if handle is not None:
            handle.remove()
    if not seen:
        yield {
            "text": "",
            "token_count": 0,
            "decode_calls": int(hook.decode_calls) if hook is not None else 0,
            "hook_applications": int(hook.applied) if hook is not None else 0,
        }


def one_step_logits(model, tokenizer, prompt_text: str, args: argparse.Namespace, vector=None) -> Any:
    device = model_device(model)
    input_ids = tokenize_prompt(tokenizer, prompt_text, args.max_prompt_tokens).to(device)
    hook = None
    handle = None
    if vector is not None:
        hook = AddDecodeVectorHook(vector, alpha=args.alpha, inject_first_n=1)
        handle = get_layer(model, args.layer).register_forward_hook(hook)
    try:
        with torch.inference_mode():
            if input_ids.shape[1] > 1:
                prefill = model(input_ids=input_ids[:, :-1], use_cache=True)
                past = prefill.past_key_values
                current = input_ids[:, -1:]
            else:
                past = None
                current = input_ids
            out = model(input_ids=current, past_key_values=past, use_cache=True)
            return out.logits[0, -1, :].detach().float().cpu()
    finally:
        if handle is not None:
            handle.remove()


def top_logit_deltas(model, tokenizer, args: argparse.Namespace, vector, label: str, top_k: int = 12) -> Dict[str, Any]:
    prompt = format_prompt(tokenizer, EVAL_PROMPTS[0], args.system)
    base = one_step_logits(model, tokenizer, prompt, args, vector=None)
    steered = one_step_logits(model, tokenizer, prompt, args, vector=vector)
    delta = steered - base
    values, ids = torch.topk(delta, k=min(top_k, int(delta.numel())))
    rows = []
    for value, token_id in zip(values.tolist(), ids.tolist()):
        token = tokenizer.decode([int(token_id)]).replace("\n", "\\n")
        rows.append({"token": token, "token_id": int(token_id), "delta_logit": float(value)})
    return {"label": label, "rows": rows}


def html_table(rows: List[Dict[str, Any]], columns: List[Tuple[str, str]]) -> str:
    head = "".join(f"<th>{html.escape(title)}</th>" for _, title in columns)
    body_rows = []
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                text = f"{value:.4f}"
            else:
                text = str(value)
            cells.append(f"<td>{html.escape(text)}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def signed_cell(value: float, digits: int = 3) -> str:
    return f"{float(value):+.{digits}f}"


def rank_flip_table(rows: List[Dict[str, Any]]) -> str:
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{html.escape(row['pool'])}</td>"
            f"<td>{float(row['prefill_rho']):.3f}</td>"
            f"<td class='emph'>{float(row['decode_rho']):.3f}</td>"
            f"<td>{signed_cell(float(row['delta']))}</td>"
            "</tr>"
        )
    return (
        "<table>"
        "<thead><tr><th>Pool</th><th>Prefill rho</th><th>Decode rho</th><th>Delta</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def deployment_table(rows: List[Dict[str, Any]]) -> str:
    body = []
    for row in rows:
        is_decode = row["proxy"] == "Decode-aligned"
        cls = " class='emph-row'" if is_decode else ""
        body.append(
            f"<tr{cls}>"
            f"<td>{html.escape(row['proxy'])}</td>"
            f"<td>{signed_cell(float(row['real_mean']))}</td>"
            f"<td>{signed_cell(float(row['real_worst']))}</td>"
            f"<td>{float(row['flip_rate']):.3f}</td>"
            f"<td>{float(row['regret_at_1']):.3f}</td>"
            "</tr>"
        )
    return (
        "<table>"
        "<thead><tr><th>Proxy</th><th>REAL mean</th><th>REAL worst</th>"
        "<th>Flip rate</th><th>Regret@1</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def projection_svg(overlap: float) -> str:
    overlap_pct = max(0.0, min(1.0, float(overlap))) * 100.0
    residual_pct = 100.0 - overlap_pct
    shared_w = 520.0 * overlap_pct / 100.0
    residual_w = 520.0 - shared_w
    return f"""
<svg viewBox="0 0 640 120" role="img" aria-label="steering vector split">
  <rect x="60" y="38" width="520" height="34" rx="5" fill="#e8edf3"/>
  <rect x="60" y="38" width="{shared_w:.1f}" height="34" rx="5" fill="#d85b50"/>
  <rect x="{60 + shared_w:.1f}" y="38" width="{residual_w:.1f}" height="34" rx="5" fill="#3577b8"/>
  <text x="60" y="28" font-size="14" fill="#25313d">Original steering vector</text>
  <text x="60" y="92" font-size="13" fill="#25313d">shared component: {overlap_pct:.1f}%</text>
  <text x="365" y="92" font-size="13" fill="#25313d">residual component: {residual_pct:.1f}%</text>
</svg>
"""


def write_report(out_dir: Path, summary: Dict[str, Any]) -> None:
    metrics = summary["projection_metrics"]
    metric_rows = [
        {"metric": "demo vector mode", "value": summary["demo_vector"]["mode"]},
        {"metric": "projection norm policy", "value": summary["demo_vector"]["projected_norm_policy"]},
        {"metric": "base CAA shared overlap", "value": summary["demo_vector"]["base_caa_overlap"]},
        {"metric": "basis dimension", "value": summary["basis"]["basis_dim"]},
        {"metric": "decode states for basis", "value": summary["basis"]["n_states"]},
        {"metric": "demo PCA variance", "value": summary["basis"]["explained_variance_demo"]},
        {"metric": "demo vector shared overlap", "value": metrics["overlap_original"]},
        {"metric": "projected residual overlap", "value": metrics["overlap_residual"]},
        {"metric": "steering vector norm", "value": summary["steering_vector"]["vector_norm"]},
    ]

    generation_rows = []
    for prompt, results in summary["generations"].items():
        for label, result in results.items():
            generation_rows.append(
                {
                    "prompt": prompt,
                    "method": label,
                    "hook_applications": result.get("hook_applications", 0),
                    "text": result.get("text", ""),
                }
            )

    logit_sections = []
    for block in summary["top_logit_deltas"]:
        logit_sections.append(
            f"<h3>{html.escape(block['label'])}</h3>"
            + html_table(block["rows"], [("token", "Token"), ("token_id", "ID"), ("delta_logit", "Delta logit")])
        )

    css = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1e252c; }
main { max-width: 1040px; margin: 0 auto; }
h1, h2, h3 { color: #17212b; }
.subtle { color: #52616f; }
.panel { border: 1px solid #d7dee7; border-radius: 8px; padding: 18px; margin: 18px 0; background: #fbfcfe; }
table { width: 100%; border-collapse: collapse; margin: 10px 0 18px; }
th, td { border-bottom: 1px solid #e1e7ef; padding: 8px 10px; vertical-align: top; text-align: left; }
th { background: #f3f6fa; font-weight: 650; }
td:nth-child(3), td:nth-child(4) { font-variant-numeric: tabular-nums; }
pre { white-space: pre-wrap; margin: 0; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.method { font-weight: 650; }
.emph, .emph-row td { font-weight: 700; color: #164f86; }
.grid { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 18px; }
@media (max-width: 780px) { .grid { grid-template-columns: 1fr; } }
"""

    gen_html = []
    for row in generation_rows:
        gen_html.append(
            "<tr>"
            f"<td>{html.escape(row['prompt'])}</td>"
            f"<td class='method'>{html.escape(row['method'])}</td>"
            f"<td>{html.escape(str(row['hook_applications']))}</td>"
            f"<td><pre>{html.escape(row['text'])}</pre></td>"
            "</tr>"
        )

    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DecodeShare Steering Projection Demo</title>
  <style>{css}</style>
</head>
<body>
<main>
  <h1>DecodeShare Steering Projection Demo</h1>
  <p class="subtle">
    Model: <strong>{html.escape(summary['config']['model'])}</strong>,
    layer {summary['config']['layer']}, alpha {summary['config']['alpha']}.
  </p>

  <section class="panel">
    <h2>Rank Flip Context</h2>
    <p>
      The projection demo below shows how one steering vector is split. The
      paper-level rank-flip result explains why DecodeShare evaluates steering
      vectors at decode time: prefill-aligned proxies can rank vectors
      differently from held-out KV-cached deployment.
    </p>
    <div class="grid">
      <div>
        <h3>Ranking alignment</h3>
        {rank_flip_table(summary['rank_flip_snapshot']['ranking_alignment'])}
      </div>
      <div>
        <h3>Deployment selection</h3>
        {deployment_table(summary['rank_flip_snapshot']['deployment_selection'])}
      </div>
    </div>
    <p class="subtle">
      Snapshot values are from the DecodeShare paper tables. Reproduce them with
      <code>bash scripts/reproduce_steering_flip_tables.sh</code>.
    </p>
  </section>

  <section class="panel">
    <h2>Vector Split</h2>
    <p>
      The demo vector starts from a CAA-style contrastive steering vector. By
      default, its shared-channel component is amplified so the before/after
      behavior is visible in a short run; use <code>--demo_vector_mode caa</code>
      to show the untouched CAA-style vector.
    </p>
    {projection_svg(metrics['overlap_original'])}
    {html_table(metric_rows, [("metric", "Metric"), ("value", "Value")])}
  </section>

  <section class="panel">
    <h2>Generations</h2>
    <table>
      <thead><tr><th>Prompt</th><th>Method</th><th>Hook apps</th><th>Output</th></tr></thead>
      <tbody>{''.join(gen_html)}</tbody>
    </table>
  </section>

  <section class="panel">
    <h2>Top One-Step Logit Changes</h2>
    <p class="subtle">Positive values show tokens most increased at the first decode decision.</p>
    {''.join(logit_sections)}
  </section>
</main>
</body>
</html>
"""
    (out_dir / "steering_projection_report.html").write_text(doc, encoding="utf-8")


def tensor_to_list(x, limit: int = 12) -> List[float]:
    return [float(v) for v in x.detach().cpu().flatten()[:limit].tolist()]


def clip_for_console(text: str, limit: int = 900) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def print_console_summary(summary: Dict[str, Any], out_dir: Path) -> None:
    metrics = summary["projection_metrics"]
    print("\n" + "=" * 88)
    print("DecodeShare steering demo: rank flip + example vector before/after")
    print("=" * 88)
    print("\n[Rank flip snapshot: paper-level steering result]")
    print(f"{'Pool':<18} {'Prefill rho':>12} {'Decode rho':>12} {'Delta':>10}")
    for row in summary["rank_flip_snapshot"]["ranking_alignment"]:
        print(
            f"{row['pool']:<18} "
            f"{float(row['prefill_rho']):>12.3f} "
            f"{float(row['decode_rho']):>12.3f} "
            f"{float(row['delta']):>+10.3f}"
        )

    print("\n[Deployment selection]")
    print(f"{'Proxy':<18} {'REAL mean':>10} {'REAL worst':>11} {'Flip rate':>10} {'Regret@1':>10}")
    for row in summary["rank_flip_snapshot"]["deployment_selection"]:
        print(
            f"{row['proxy']:<18} "
            f"{float(row['real_mean']):>+10.3f} "
            f"{float(row['real_worst']):>+11.3f} "
            f"{float(row['flip_rate']):>10.3f} "
            f"{float(row['regret_at_1']):>10.3f}"
        )

    print("\n[Example steering-vector projection]")
    print("Protocol role: inspect and remove overlap with the decode-shared channel.")
    print("Base vector source:", summary["steering_vector"]["source"])
    print("Demo vector:", summary["demo_vector"]["description"])
    print(f"Base vector shared overlap: {float(summary['demo_vector']['base_caa_overlap']):.3f}")
    print(f"Demo vector shared overlap: {float(metrics['overlap_original']):.3f}")
    print(f"After projection overlap: {float(metrics['overlap_residual']):.6f}")
    print("Projected norm policy:", summary["demo_vector"]["projected_norm_policy"])
    print(
        "Interpretation: DecodeShare removes the component of the demo vector "
        "that lies in the decode-shared channel, then reuses the residual vector."
    )

    print("\n[Top one-step logit increases]")
    for block in summary["top_logit_deltas"]:
        print(f"\n{block['label']}:")
        for row in block["rows"][:6]:
            print(f"  {float(row['delta_logit']):+7.3f}  {str(row['token'])!r}")

    print("\n[Before/after example generations]")
    for prompt, results in summary["generations"].items():
        print("\nPrompt:", prompt)
        for label, result in results.items():
            print(f"\n--- {label} ---")
            print(clip_for_console(result.get("text", "")))

    print("\n[Report files]")
    print(out_dir / "steering_projection_report.html")
    print(out_dir / "projection_summary.json")
    print("=" * 88 + "\n")


def main() -> None:
    args = parse_args()
    if args.dry_run:
        print("DecodeShare steering projection demo")
        print(f"  model: {args.model}")
        print(f"  device: {args.device}")
        print(f"  layer: {args.layer}")
        print(f"  demo_vector_mode: {args.demo_vector_mode}")
        print(f"  out_dir: {args.out_dir}")
        print("  dry_run: no model will be loaded")
        return

    load_runtime_dependencies()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Load] model={args.model} device={args.device} dtype={args.dtype}")
    model, tokenizer = load_model_and_tokenizer(args)

    print("[Basis] collecting decode-time states and estimating demo shared basis")
    basis, basis_info = estimate_shared_basis(model, tokenizer, args)
    print(f"[Basis] states={basis_info['n_states']} dim={basis_info['basis_dim']} var={basis_info['explained_variance_demo']:.3f}")

    print("[Vector] estimating contrastive steering vector")
    caa_vector, steering_info = estimate_steering_vector(model, tokenizer, args)
    demo_vector, residual, projection, caa_projection, demo_info = build_demo_vector(basis, caa_vector, args)
    print(f"[Projection] base CAA overlap={caa_projection['overlap_original']:.3f}")
    print(f"[Projection] demo vector overlap={projection['overlap_original']:.3f}")
    print(f"[Projection] residual overlap={projection['overlap_residual']:.6f}")

    print("[Generate] baseline/original/projected examples")
    generations: Dict[str, Dict[str, Any]] = {}
    for prompt in EVAL_PROMPTS:
        generations[prompt] = {
            "baseline (no steering)": generate_with_optional_vector(model, tokenizer, prompt, args, vector=None),
            "before DecodeShare projection (demo vector)": generate_with_optional_vector(
                model, tokenizer, prompt, args, vector=demo_vector
            ),
            "removed shared component only": generate_with_optional_vector(
                model, tokenizer, prompt, args, vector=projection["shared"]
            ),
            "after DecodeShare projection (shared removed)": generate_with_optional_vector(
                model, tokenizer, prompt, args, vector=residual
            ),
        }

    print("[Probe] one-step logit deltas")
    top_delta_blocks = [
        top_logit_deltas(model, tokenizer, args, demo_vector, "Before DecodeShare projection"),
        top_logit_deltas(model, tokenizer, args, projection["shared"], "Removed shared component only"),
        top_logit_deltas(model, tokenizer, args, residual, "After DecodeShare projection"),
    ]

    summary = {
        "config": {
            "model": args.model,
            "device": args.device,
            "dtype": args.dtype,
            "layer": int(args.layer),
            "alpha": float(args.alpha),
            "inject_first_n": int(args.inject_first_n),
            "demo_vector_mode": args.demo_vector_mode,
            "shared_component_scale": float(args.shared_component_scale),
            "preserve_residual_norm": bool(args.preserve_residual_norm),
            "positive_style": args.positive_style,
            "negative_style": args.negative_style,
        },
        "basis": basis_info,
        "steering_vector": steering_info,
        "demo_vector": demo_info,
        "base_caa_projection_metrics": {k: v for k, v in caa_projection.items() if not hasattr(v, "shape")},
        "projection_metrics": {k: v for k, v in projection.items() if not hasattr(v, "shape")},
        "vector_preview": {
            "base_caa_first_values": tensor_to_list(caa_vector),
            "demo_vector_first_values": tensor_to_list(demo_vector),
            "shared_first_values": tensor_to_list(projection["shared"]),
            "residual_first_values": tensor_to_list(residual),
        },
        "generations": generations,
        "top_logit_deltas": top_delta_blocks,
        "rank_flip_snapshot": {
            "ranking_alignment": RANKING_ALIGNMENT_ROWS,
            "deployment_selection": DEPLOYMENT_SELECTION_ROWS,
        },
    }
    (out_dir / "projection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(out_dir, summary)
    print_console_summary(summary, out_dir)
    print(f"[Done] wrote {out_dir / 'steering_projection_report.html'}")
    print(f"[Done] wrote {out_dir / 'projection_summary.json'}")


if __name__ == "__main__":
    main()
