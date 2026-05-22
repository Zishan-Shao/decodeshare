#!/usr/bin/env python3
"""Interactive Gradio chat for the DecodeShare steering demo."""

from __future__ import annotations

import argparse
import html
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHAT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_CHAT_CACHE = REPO_ROOT / "demo" / "assets" / "interactive_tinyllama_chat_cache.pt"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

VECTOR_PRESETS: Dict[str, Dict[str, str]] = {
    "Pirate": {
        "positive": "Reply in a vivid pirate voice.",
        "negative": "Reply in a concise neutral voice.",
        "description": "Style vector for visible tone changes.",
    },
    "Concise": {
        "positive": "Reply with a short direct answer.",
        "negative": "Reply with a long detailed explanation and extra context.",
        "description": "Length/control vector for shorter answers.",
    },
    "Step-by-step": {
        "positive": "Reply with clear numbered step-by-step reasoning.",
        "negative": "Reply with a terse answer and no explanation.",
        "description": "Reasoning-scaffold vector.",
    },
    "Confident": {
        "positive": "Reply with confident decisive wording.",
        "negative": "Reply with cautious hedged wording.",
        "description": "Tone vector for confidence and decisiveness.",
    },
}

VECTOR_MODES = [
    "original vector",
    "DecodeShare residual",
    "shared component only",
    "partial removal",
]

CHAT_SESSIONS: Dict[str, Dict[str, Any]] = {}

EXAMPLE_PROMPTS: Dict[str, str] = {
    "SVD pirate metaphor": "Explain the concept of 'Singular Value Decomposition' to a 5-year-old using a pirate metaphor.",
    "30-minute study plan": "I keep getting distracted when studying. Give me a plan for the next 30 minutes.",
    "Debugging checklist": "Give me a step-by-step checklist for debugging a Python script that suddenly became slow.",
    "Concise exam advice": "I have an exam tomorrow and only two hours to prepare. What should I do?",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server_name", default="127.0.0.1")
    parser.add_argument("--server_port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def resolve_cache_path(path: str) -> Path:
    p = Path(path or DEFAULT_CHAT_CACHE).expanduser()
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p


def default_device_choice() -> str:
    try:
        import torch
    except Exception:
        return "cuda"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cuda_available = torch.cuda.is_available()
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if cuda_available:
        return "cuda"
    if mps_available:
        return "mps"
    return "cpu"


def default_dtype_choice(device: str) -> str:
    return "fp32" if device == "cpu" else "fp16"


def default_max_new_tokens(device: str) -> int:
    return 40 if device == "cpu" else 80


def update_progress(progress: Optional[Any], value: float, desc: str) -> None:
    if progress is not None:
        progress(float(value), desc=desc)


def demo_runtime():
    from demo import run_steering_projection_demo as runtime

    runtime.load_runtime_dependencies()
    return runtime


def make_runtime_args(
    *,
    model: str,
    device: str,
    dtype: str,
    layer: int,
    basis_k: int,
    calib_max_new_tokens: int,
    steer_max_new_tokens: int,
    eval_max_new_tokens: int,
    max_prompt_tokens: int,
    alpha: float,
    inject_first_n: int,
    system: str,
    positive_style: str,
    negative_style: str,
    seed: int,
    local_files_only: bool,
    trust_remote_code: bool,
) -> argparse.Namespace:
    return argparse.Namespace(
        model=model,
        device=device,
        dtype=dtype,
        layer=int(layer),
        basis_k=int(basis_k),
        calib_max_new_tokens=int(calib_max_new_tokens),
        steer_max_new_tokens=int(steer_max_new_tokens),
        eval_max_new_tokens=int(eval_max_new_tokens),
        max_prompt_tokens=int(max_prompt_tokens),
        alpha=float(alpha),
        inject_first_n=int(inject_first_n),
        demo_vector_mode="caa",
        shared_component_scale=1.0,
        preserve_residual_norm=False,
        system=system,
        positive_style=positive_style,
        negative_style=negative_style,
        out_dir="",
        seed=int(seed),
        local_files_only=bool(local_files_only),
        trust_remote_code=bool(trust_remote_code),
        dry_run=False,
    )


def config_for_cache(
    *,
    model: str,
    layer: int,
    basis_k: int,
    calib_max_new_tokens: int,
    steer_max_new_tokens: int,
    max_prompt_tokens: int,
    system: str,
    seed: int,
) -> Dict[str, Any]:
    return {
        "model": model,
        "layer": int(layer),
        "basis_k": int(basis_k),
        "calib_max_new_tokens": int(calib_max_new_tokens),
        "steer_max_new_tokens": int(steer_max_new_tokens),
        "max_prompt_tokens": int(max_prompt_tokens),
        "system": system,
        "seed": int(seed),
        "vector_presets": VECTOR_PRESETS,
    }


def tensor_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "info": record["info"],
        "original": record["original"].detach().cpu().float(),
        "shared": record["shared"].detach().cpu().float(),
        "residual": record["residual"].detach().cpu().float(),
        "overlap_original": float(record["overlap_original"]),
        "overlap_residual": float(record["overlap_residual"]),
        "shared_norm": float(record["shared_norm"]),
        "residual_norm": float(record["residual_norm"]),
    }


def cache_payload_from_core(core: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "format_version": 1,
        "config": core["config"],
        "basis": core["basis"].detach().cpu().float(),
        "basis_info": core["basis_info"],
        "vectors": {
            name: {
                "description": record["description"],
                "positive": record["positive"],
                "negative": record["negative"],
                "decode": tensor_record(record["decode"]),
                "prefill": tensor_record(record["prefill"]),
            }
            for name, record in core["vectors"].items()
        },
    }


def save_chat_cache(runtime, cache_path: Path, core: Dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    runtime.torch.save(cache_payload_from_core(core), cache_path)


def load_chat_cache(runtime, cache_path: Path, expected_config: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    try:
        payload = runtime.torch.load(cache_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = runtime.torch.load(cache_path, map_location="cpu")

    if payload.get("format_version") != 1:
        raise ValueError(f"Unsupported cache format: {payload.get('format_version')}")

    cached_config = payload.get("config", {})
    mismatches = []
    for key in [
        "model",
        "layer",
        "basis_k",
        "calib_max_new_tokens",
        "steer_max_new_tokens",
        "max_prompt_tokens",
        "system",
        "seed",
    ]:
        if cached_config.get(key) != expected_config.get(key):
            mismatches.append(f"{key}: cache={cached_config.get(key)!r}, requested={expected_config.get(key)!r}")
    if cached_config.get("vector_presets") != expected_config.get("vector_presets"):
        mismatches.append("vector_presets changed")
    if mismatches:
        raise ValueError("Cache does not match current setup (" + "; ".join(mismatches) + ")")

    core = {
        "basis": payload["basis"].detach().cpu().float(),
        "basis_info": payload["basis_info"],
        "vectors": payload["vectors"],
        "config": cached_config,
    }
    return core, f"Loaded cached basis/vectors from {cache_path}"


def render_chat_setup(state: Optional[Dict[str, Any]]) -> str:
    if not state:
        return """
<div class="panel intro-panel">
  <div class="title-row">
    <h2>DecodeShare Steering Chat</h2>
    <span class="channel-chip">prefill vs decode</span>
  </div>
</div>
"""
    basis = state["basis_info"]
    return f"""
<div class="panel intro-panel">
  <div class="title-row">
    <h2>DecodeShare Steering Chat</h2>
    <span class="channel-chip">ready</span>
  </div>
  <div class="metric-grid">
    <div class="metric"><span>model</span><strong>{esc(state["config"]["model"])}</strong></div>
    <div class="metric"><span>layer</span><strong>{esc(state["config"]["layer"])}</strong></div>
    <div class="metric"><span>basis dim</span><strong>{esc(basis.get("basis_dim", ""))}</strong></div>
    <div class="metric"><span>decode states</span><strong>{esc(basis.get("n_states", ""))}</strong></div>
  </div>
</div>
"""


def collect_prefill_last_states(runtime, model_obj, tokenizer, prompts: List[str], args: argparse.Namespace):
    torch = runtime.torch
    records = []
    calls = 0
    block = runtime.get_layer(model_obj, args.layer)

    def hook(_module, _inputs, output):
        nonlocal calls
        hidden = output[0] if isinstance(output, tuple) else output
        if hasattr(hidden, "ndim") and hidden.ndim == 3:
            calls += 1
            records.append(hidden[:, -1, :].detach().float().cpu())
        return output

    handle = block.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            for prompt in prompts:
                input_ids = runtime.tokenize_prompt(tokenizer, prompt, args.max_prompt_tokens).to(runtime.model_device(model_obj))
                _ = model_obj(input_ids=input_ids, use_cache=False)
    finally:
        handle.remove()

    if not records:
        raise RuntimeError("No prefill states were collected.")
    return torch.cat(records, dim=0).float(), {"prefill_calls": int(calls)}


def estimate_prefill_steering_vector(runtime, model_obj, tokenizer, args: argparse.Namespace):
    positive, negative = runtime.make_steering_prompts(tokenizer, args)
    pos_states, pos_stats = collect_prefill_last_states(runtime, model_obj, tokenizer, positive, args)
    neg_states, neg_stats = collect_prefill_last_states(runtime, model_obj, tokenizer, negative, args)
    vector = pos_states.mean(dim=0) - neg_states.mean(dim=0)
    norm = float(vector.norm().item())
    if not norm > 1e-8:
        raise RuntimeError("Estimated prefill steering vector has near-zero norm.")
    info = {
        "source": "prefill contrastive mean-difference",
        "positive_states": int(pos_states.shape[0]),
        "negative_states": int(neg_states.shape[0]),
        "positive_prefill_calls": int(pos_stats["prefill_calls"]),
        "negative_prefill_calls": int(neg_stats["prefill_calls"]),
        "vector_norm": norm,
    }
    return vector.cpu().float(), info


def make_vector_record(runtime, basis, vector, info: Dict[str, Any]) -> Dict[str, Any]:
    projection = runtime.project_vector(basis, vector)
    return {
        "info": {k: v for k, v in info.items() if not hasattr(v, "shape")},
        "original": vector.cpu().float(),
        "shared": projection["shared"].cpu().float(),
        "residual": projection["residual"].cpu().float(),
        "overlap_original": float(projection["overlap_original"]),
        "overlap_residual": float(projection["overlap_residual"]),
        "shared_norm": float(projection["shared_norm"]),
        "residual_norm": float(projection["residual_norm"]),
    }


def estimate_chat_core(
    runtime,
    model_obj,
    tokenizer,
    *,
    model: str,
    layer: int,
    basis_k: int,
    calib_max_new_tokens: int,
    steer_max_new_tokens: int,
    max_prompt_tokens: int,
    system: str,
    seed: int,
    local_files_only: bool,
    trust_remote_code: bool,
    progress: Optional[Any] = None,
) -> Dict[str, Any]:
    update_progress(progress, 0.58, "Estimating decode-shared basis")
    base_args = make_runtime_args(
        model=model,
        device=str(runtime.model_device(model_obj)),
        dtype="fp32",
        layer=layer,
        basis_k=basis_k,
        calib_max_new_tokens=calib_max_new_tokens,
        steer_max_new_tokens=steer_max_new_tokens,
        eval_max_new_tokens=80,
        max_prompt_tokens=max_prompt_tokens,
        alpha=1.0,
        inject_first_n=20,
        system=system,
        positive_style="",
        negative_style="",
        seed=seed,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    basis, basis_info = runtime.estimate_shared_basis(model_obj, tokenizer, base_args)

    vectors: Dict[str, Dict[str, Any]] = {}
    preset_items = list(VECTOR_PRESETS.items())
    for idx, (name, preset) in enumerate(preset_items, start=1):
        update_progress(progress, 0.62 + 0.28 * (idx - 1) / max(1, len(preset_items)), f"Estimating {name} vector")
        vec_args = make_runtime_args(
            model=model,
            device=str(runtime.model_device(model_obj)),
            dtype="fp32",
            layer=layer,
            basis_k=basis_k,
            calib_max_new_tokens=calib_max_new_tokens,
            steer_max_new_tokens=steer_max_new_tokens,
            eval_max_new_tokens=80,
            max_prompt_tokens=max_prompt_tokens,
            alpha=1.0,
            inject_first_n=20,
            system=system,
            positive_style=preset["positive"],
            negative_style=preset["negative"],
            seed=seed,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        decode_vector, decode_info = runtime.estimate_steering_vector(model_obj, tokenizer, vec_args)
        prefill_vector, prefill_info = estimate_prefill_steering_vector(runtime, model_obj, tokenizer, vec_args)
        vectors[name] = {
            "description": preset["description"],
            "positive": preset["positive"],
            "negative": preset["negative"],
            "decode": make_vector_record(runtime, basis, decode_vector, decode_info),
            "prefill": make_vector_record(runtime, basis, prefill_vector, prefill_info),
        }

    return {
        "basis": basis,
        "basis_info": basis_info,
        "vectors": vectors,
        "config": config_for_cache(
            model=model,
            layer=layer,
            basis_k=basis_k,
            calib_max_new_tokens=calib_max_new_tokens,
            steer_max_new_tokens=steer_max_new_tokens,
            max_prompt_tokens=max_prompt_tokens,
            system=system,
            seed=seed,
        ),
    }


def prepare_chat_state(
    *,
    model: str,
    device: str,
    dtype: str,
    layer: int,
    basis_k: int,
    calib_max_new_tokens: int,
    steer_max_new_tokens: int,
    max_prompt_tokens: int,
    system: str,
    seed: int,
    local_files_only: bool,
    trust_remote_code: bool,
    cache_path: str,
    use_cache: bool,
    save_cache: bool,
    progress: Optional[Any] = None,
) -> Tuple[Dict[str, Any], str]:
    update_progress(progress, 0.02, "Loading runtime dependencies")
    runtime = demo_runtime()
    update_progress(progress, 0.08, "Checking device")
    if device == "cuda" and not runtime.torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this Python environment. Select cpu/mps or use a CUDA-enabled env.")
    expected_config = config_for_cache(
        model=model,
        layer=layer,
        basis_k=basis_k,
        calib_max_new_tokens=calib_max_new_tokens,
        steer_max_new_tokens=steer_max_new_tokens,
        max_prompt_tokens=max_prompt_tokens,
        system=system,
        seed=seed,
    )
    resolved_cache = resolve_cache_path(cache_path)
    messages: List[str] = []

    update_progress(progress, 0.12, "Preparing model arguments")
    runtime.set_seed(int(seed))
    model_args = make_runtime_args(
        model=model,
        device=device,
        dtype=dtype,
        layer=layer,
        basis_k=basis_k,
        calib_max_new_tokens=calib_max_new_tokens,
        steer_max_new_tokens=steer_max_new_tokens,
        eval_max_new_tokens=80,
        max_prompt_tokens=max_prompt_tokens,
        alpha=1.0,
        inject_first_n=20,
        system=system,
        positive_style="",
        negative_style="",
        seed=seed,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    update_progress(progress, 0.18, "Loading model and tokenizer")
    model_obj, tokenizer = runtime.load_model_and_tokenizer(model_args)

    core = None
    if use_cache and resolved_cache.exists():
        update_progress(progress, 0.72, "Loading cached basis and vectors")
        try:
            core, msg = load_chat_cache(runtime, resolved_cache, expected_config)
            messages.append(msg)
        except Exception as exc:
            messages.append(f"Skipped cache: {exc}")
    elif use_cache:
        messages.append(f"No cache found at {resolved_cache}; estimating basis/vectors.")

    if core is None:
        core = estimate_chat_core(
            runtime,
            model_obj,
            tokenizer,
            model=model,
            layer=layer,
            basis_k=basis_k,
            calib_max_new_tokens=calib_max_new_tokens,
            steer_max_new_tokens=steer_max_new_tokens,
            max_prompt_tokens=max_prompt_tokens,
            system=system,
            seed=seed,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            progress=progress,
        )
        messages.append("Estimated decode-shared basis and preset vectors.")
        if save_cache:
            update_progress(progress, 0.92, "Saving basis/vector cache")
            save_chat_cache(runtime, resolved_cache, core)
            messages.append(f"Saved cache to {resolved_cache}.")

    update_progress(progress, 0.96, "Finalizing demo state")
    state = {
        "runtime": runtime,
        "model": model_obj,
        "tokenizer": tokenizer,
        "basis": core["basis"],
        "basis_info": core["basis_info"],
        "vectors": core["vectors"],
        "config": {
            **core["config"],
            "device": device,
            "dtype": dtype,
        },
    }
    update_progress(progress, 1.0, "Ready")
    return state, " ".join(messages)


def initialize_chat_state(
    model: str,
    device: str,
    dtype: str,
    layer: int,
    basis_k: int,
    calib_max_new_tokens: int,
    steer_max_new_tokens: int,
    max_prompt_tokens: int,
    system: str,
    seed: int,
    local_files_only: bool,
    trust_remote_code: bool,
    cache_path: str,
    use_cache: bool,
    save_cache: bool,
    progress: Optional[Any] = None,
) -> Tuple[str, str, str]:
    state, status = prepare_chat_state(
        model=model,
        device=device,
        dtype=dtype,
        layer=layer,
        basis_k=basis_k,
        calib_max_new_tokens=calib_max_new_tokens,
        steer_max_new_tokens=steer_max_new_tokens,
        max_prompt_tokens=max_prompt_tokens,
        system=system,
        seed=seed,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        cache_path=cache_path,
        use_cache=use_cache,
        save_cache=save_cache,
        progress=progress,
    )
    session_id = uuid4().hex
    CHAT_SESSIONS[session_id] = state
    return session_id, render_chat_setup(state), status


def selected_vector(record: Dict[str, Any], mode: str, beta: float):
    original = record["original"]
    shared = record["shared"]
    if mode == "original vector":
        return original
    if mode == "DecodeShare residual":
        return record["residual"]
    if mode == "shared component only":
        return shared
    if mode == "partial removal":
        return original - float(beta) * shared
    raise ValueError(f"Unknown vector mode: {mode}")


def vector_status(
    preset: str,
    estimator: str,
    mode: str,
    beta: float,
    alpha: float,
    record: Optional[Dict[str, Any]],
) -> str:
    if record is None:
        return f"{estimator}: no steering vector applied."
    return (
        f"{estimator} {preset} | {mode} | alpha={float(alpha):.2f} | beta={float(beta):.2f} | "
        f"shared overlap={float(record['overlap_original']):.3f} | "
        f"residual overlap={float(record['overlap_residual']):.6f}"
    )


def chat_prompt_from_history(history: List[Any], message: str) -> str:
    recent = history[-6:] if history else []
    if not recent:
        return message
    parts = []
    for item in recent:
        if isinstance(item, dict):
            role = item.get("role", "assistant")
            content = item.get("content", "")
            parts.append(f"Previous {role}: {content}")
        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            parts.append(f"Previous user: {item[0]}\nPrevious assistant: {item[1]}")
    parts.append(f"Current user: {message}")
    return "\n\n".join(parts)


def chat_once(
    session_id: str,
    message: str,
    baseline_history: Optional[List[Dict[str, str]]],
    prefill_history: Optional[List[Dict[str, str]]],
    decode_history: Optional[List[Dict[str, str]]],
    preset: str,
    mode: str,
    alpha: float,
    beta: float,
    inject_first_n: int,
    max_new_tokens: int,
    progress: Optional[Any] = None,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]], str, str]:
    baseline_history = list(baseline_history or [])
    prefill_history = list(prefill_history or [])
    decode_history = list(decode_history or [])
    message = (message or "").strip()
    if not message:
        return baseline_history, prefill_history, decode_history, "", "Enter a prompt first."
    state = CHAT_SESSIONS.get(session_id or "")
    if state is None:
        return baseline_history, prefill_history, decode_history, message, "Initialize the model before chatting."

    update_progress(progress, 0.02, "Preparing prompts")
    runtime = state["runtime"]
    config = state["config"]
    args = make_runtime_args(
        model=config["model"],
        device=config["device"],
        dtype=config["dtype"],
        layer=config["layer"],
        basis_k=config["basis_k"],
        calib_max_new_tokens=config["calib_max_new_tokens"],
        steer_max_new_tokens=config["steer_max_new_tokens"],
        eval_max_new_tokens=max_new_tokens,
        max_prompt_tokens=config["max_prompt_tokens"],
        alpha=alpha,
        inject_first_n=inject_first_n,
        system=config["system"],
        positive_style="",
        negative_style="",
        seed=config["seed"],
        local_files_only=False,
        trust_remote_code=False,
    )

    baseline_prompt = chat_prompt_from_history(baseline_history, message)
    update_progress(progress, 0.10, "Generating baseline response (1/3)")
    baseline = runtime.generate_with_optional_vector(
        state["model"], state["tokenizer"], baseline_prompt, args, vector=None
    )

    preset_record = None if preset == "None" else state["vectors"].get(preset)
    prefill_record = None if preset_record is None else preset_record["prefill"]
    decode_record = None if preset_record is None else preset_record["decode"]

    prefill_vector = None if prefill_record is None else selected_vector(prefill_record, mode, beta)
    prefill_prompt = chat_prompt_from_history(prefill_history, message)
    update_progress(progress, 0.40, "Generating prefill-estimated response (2/3)")
    prefill = runtime.generate_with_optional_vector(
        state["model"], state["tokenizer"], prefill_prompt, args, vector=prefill_vector, alpha=alpha
    )

    decode_vector = None if decode_record is None else selected_vector(decode_record, mode, beta)
    decode_prompt = chat_prompt_from_history(decode_history, message)
    update_progress(progress, 0.70, "Generating decode-estimated response (3/3)")
    decode = runtime.generate_with_optional_vector(
        state["model"], state["tokenizer"], decode_prompt, args, vector=decode_vector, alpha=alpha
    )

    update_progress(progress, 0.94, "Updating chat")
    baseline_history.extend(
        [{"role": "user", "content": message}, {"role": "assistant", "content": baseline.get("text", "")}]
    )
    prefill_history.extend(
        [{"role": "user", "content": message}, {"role": "assistant", "content": prefill.get("text", "")}]
    )
    decode_history.extend(
        [{"role": "user", "content": message}, {"role": "assistant", "content": decode.get("text", "")}]
    )
    status = (
        vector_status(preset, "prefill-est", mode, beta, alpha, prefill_record)
        + f" | prefill hook apps={int(prefill.get('hook_applications', 0))}\n"
        + vector_status(preset, "decode-est", mode, beta, alpha, decode_record)
        + f" | decode hook apps={int(decode.get('hook_applications', 0))}"
    )
    update_progress(progress, 1.0, "Done")
    return baseline_history, prefill_history, decode_history, "", status


def clear_chat() -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]], str]:
    return [], [], [], "Cleared chat history."


def load_example_prompt(label: str) -> str:
    return EXAMPLE_PROMPTS.get(label, "")


CSS = """
.gradio-container {
  max-width: 1280px !important;
  color: #182233;
  background:
    linear-gradient(180deg, #f4fbff 0%, #ffffff 38%, #f8fafc 100%) !important;
}
.gradio-container a { color: #1667a8; }
.gradio-container button.primary {
  color: #ffffff !important;
  border: 1px solid #008ea6 !important;
  background: linear-gradient(135deg, #0fbad0 0%, #2466b4 100%) !important;
  box-shadow: 0 9px 22px rgba(19, 128, 177, .23) !important;
}
.gradio-container button.secondary,
.gradio-container button:not(.primary) {
  border: 1px solid #cdddeb !important;
  color: #1f3550 !important;
  background: linear-gradient(180deg, #ffffff 0%, #f3f8fc 100%) !important;
}
.gradio-container label,
.gradio-container .block-title {
  color: #26374a !important;
  font-weight: 650 !important;
}
.gradio-container input,
.gradio-container textarea,
.gradio-container select {
  border-color: #c7d8e8 !important;
}
.panel {
  border: 1px solid #c7dce9;
  border-radius: 8px;
  background: rgba(255,255,255,.94);
  padding: 18px;
  margin: 14px 0;
  box-shadow: 0 14px 36px rgba(33, 83, 138, .10);
}
.intro-panel {
  position: relative;
  overflow: hidden;
  border-color: rgba(0, 188, 212, .34);
  background:
    linear-gradient(135deg, rgba(0, 201, 216, .18) 0%, rgba(255,255,255,.96) 46%, rgba(255, 212, 91, .18) 100%);
}
.intro-panel::before {
  content: "";
  position: absolute;
  inset: 0 0 auto;
  height: 5px;
  background: linear-gradient(90deg, #03bfd2 0%, #2b65b1 54%, #ffd35b 100%);
}
.title-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}
.channel-chip {
  flex: 0 0 auto;
  border: 1px solid rgba(0, 170, 192, .36);
  border-radius: 999px;
  padding: 6px 10px;
  color: #095b74;
  background: rgba(231, 254, 255, .78);
  font-size: 12px;
  font-weight: 750;
  text-transform: uppercase;
}
.panel h2 { margin: 0 0 12px; font-size: 22px; color: #10233f; letter-spacing: 0; }
.title-row h2 { margin: 0; }
.chat-note { color: #415160; margin: 0; line-height: 1.55; }
.metric-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin: 16px 0;
}
.metric {
  border: 1px solid #d5e5f0;
  border-radius: 8px;
  background: linear-gradient(180deg, rgba(255,255,255,.92) 0%, rgba(239,249,252,.86) 100%);
  padding: 13px 14px;
}
.metric span { display: block; color: #667585; font-size: 12px; text-transform: uppercase; margin-bottom: 7px; }
.metric strong { font-size: 18px; color: #10233f; overflow-wrap: anywhere; }
.baseline-card,
.prefill-card,
.decode-card {
  border-radius: 8px !important;
  overflow: hidden;
  border: 1px solid #d7e4ef !important;
  background: #ffffff !important;
}
.baseline-card { border-top: 4px solid #2f77b4 !important; }
.prefill-card { border-top: 4px solid #c53e72 !important; }
.decode-card { border-top: 4px solid #08bfd0 !important; }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th, td { border-bottom: 1px solid #e2edf5; padding: 8px 9px; text-align: left; vertical-align: top; }
th { color: #405162; background: #f4f9fc; font-weight: 650; }
td { color: #1e2935; }
@media (max-width: 900px) {
  .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .title-row { align-items: flex-start; flex-direction: column; }
}
"""


def build_app():
    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Install demo dependencies first: pip install -r demo/requirements-demo.txt") from exc

    def initialize_chat_state_ui(*args, progress=gr.Progress(track_tqdm=True)):
        return initialize_chat_state(*args, progress=progress)

    def chat_once_ui(*args, progress=gr.Progress(track_tqdm=True)):
        return chat_once(*args, progress=progress)

    with gr.Blocks(css=CSS, title="DecodeShare Interactive Steering Chat") as app:
        chat_session = gr.State("")
        chat_intro = gr.HTML(render_chat_setup(None))
        default_device = default_device_choice()

        with gr.Row():
            init_button = gr.Button("Initialize Demo", variant="primary", scale=1)
            chat_status = gr.Textbox(label="Status", interactive=False, scale=4)

        with gr.Row():
            preset = gr.Dropdown(["None"] + list(VECTOR_PRESETS.keys()), value="Step-by-step", label="Steering preset")
            alpha = gr.Slider(-8.0, 8.0, value=3.0, step=0.25, label="Alpha")
            max_new_tokens = gr.Slider(
                8,
                192,
                value=default_max_new_tokens(default_device),
                step=4,
                label="Max new tokens",
            )

        user_message = gr.Textbox(
            label="Prompt",
            placeholder="Ask a question, request a style, or test a reasoning prompt.",
            lines=3,
        )
        with gr.Row():
            example_prompt = gr.Dropdown(
                list(EXAMPLE_PROMPTS.keys()),
                value=None,
                label="Example prompt",
                scale=4,
            )
            load_example_button = gr.Button("Use Example", scale=1)
        with gr.Row():
            send_button = gr.Button("Send", variant="primary")
            clear_button = gr.Button("Clear")
        with gr.Row():
            baseline_chat = gr.Chatbot(
                label="Baseline",
                height=420,
                type="messages",
                elem_classes=["baseline-card"],
            )
            prefill_chat = gr.Chatbot(
                label="Prefill-estimated vector",
                height=420,
                type="messages",
                elem_classes=["prefill-card"],
            )
            decode_chat = gr.Chatbot(
                label="Decode-estimated vector",
                height=420,
                type="messages",
                elem_classes=["decode-card"],
            )

        with gr.Accordion("Advanced Settings", open=False):
            with gr.Row():
                chat_model = gr.Textbox(value=DEFAULT_CHAT_MODEL, label="Model")
                chat_system = gr.Textbox(value="You are a helpful assistant.", label="System prompt")
            with gr.Row():
                cache_path = gr.Textbox(value=str(DEFAULT_CHAT_CACHE.relative_to(REPO_ROOT)), label="Basis/vector cache")
                use_cache = gr.Checkbox(value=True, label="Use cache if available")
                save_cache = gr.Checkbox(value=True, label="Save cache after estimation")
            with gr.Row():
                chat_device = gr.Dropdown(["cuda", "cpu", "mps"], value=default_device, label="Device")
                chat_dtype = gr.Dropdown(["fp16", "bf16", "fp32"], value=default_dtype_choice(default_device), label="Dtype")
                chat_layer = gr.Number(value=16, precision=0, label="Layer")
                chat_basis_k = gr.Number(value=24, precision=0, label="Basis dim")
            with gr.Row():
                chat_calib_tokens = gr.Number(value=6, precision=0, label="Basis decode tokens")
                chat_steer_tokens = gr.Number(value=6, precision=0, label="Vector decode tokens")
                chat_max_prompt = gr.Number(value=384, precision=0, label="Max prompt tokens")
                chat_seed = gr.Number(value=7, precision=0, label="Seed")
            with gr.Row():
                chat_local_files = gr.Checkbox(value=False, label="Local files only")
                chat_trust_remote = gr.Checkbox(value=False, label="Trust remote code")
            with gr.Row():
                vector_mode = gr.Dropdown(VECTOR_MODES, value="original vector", label="Vector mode")
                beta = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="Beta for partial removal")
                inject_first_n = gr.Slider(1, 128, value=20, step=1, label="Inject first N decode steps")

        init_button.click(
            initialize_chat_state_ui,
            inputs=[
                chat_model,
                chat_device,
                chat_dtype,
                chat_layer,
                chat_basis_k,
                chat_calib_tokens,
                chat_steer_tokens,
                chat_max_prompt,
                chat_system,
                chat_seed,
                chat_local_files,
                chat_trust_remote,
                cache_path,
                use_cache,
                save_cache,
            ],
            outputs=[chat_session, chat_intro, chat_status],
            show_progress="full",
        )
        send_button.click(
            chat_once_ui,
            inputs=[
                chat_session,
                user_message,
                baseline_chat,
                prefill_chat,
                decode_chat,
                preset,
                vector_mode,
                alpha,
                beta,
                inject_first_n,
                max_new_tokens,
            ],
            outputs=[baseline_chat, prefill_chat, decode_chat, user_message, chat_status],
            show_progress="full",
        )
        user_message.submit(
            chat_once_ui,
            inputs=[
                chat_session,
                user_message,
                baseline_chat,
                prefill_chat,
                decode_chat,
                preset,
                vector_mode,
                alpha,
                beta,
                inject_first_n,
                max_new_tokens,
            ],
            outputs=[baseline_chat, prefill_chat, decode_chat, user_message, chat_status],
            show_progress="full",
        )
        load_example_button.click(load_example_prompt, inputs=[example_prompt], outputs=[user_message])
        clear_button.click(clear_chat, outputs=[baseline_chat, prefill_chat, decode_chat, chat_status])
    return app.queue()


def main() -> None:
    args = parse_args()
    app = build_app()
    app.launch(server_name=args.server_name, server_port=args.server_port, share=args.share)


if __name__ == "__main__":
    main()
