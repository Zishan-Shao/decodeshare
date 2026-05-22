#!/usr/bin/env python3
"""Gradio interface for the DecodeShare steering projection demo."""

from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from uuid import uuid4


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = REPO_ROOT / "outputs" / "demo_steering_projection" / "projection_summary.json"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY), help="Default projection_summary.json to load")
    parser.add_argument("--server_name", default="127.0.0.1")
    parser.add_argument("--server_port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def fmt(value: Any, digits: int = 3, signed: bool = False) -> str:
    try:
        v = float(value)
    except Exception:
        return esc(value)
    if signed:
        return f"{v:+.{digits}f}"
    return f"{v:.{digits}f}"


def load_summary(path: str) -> Tuple[Dict[str, Any], Path]:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"Summary not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f), p


def table(headers: Iterable[str], rows: Iterable[Iterable[Any]], *, cls: str = "") -> str:
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>")
    cls_attr = f" class='{cls}'" if cls else ""
    return f"<table{cls_attr}><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def metric_cards(summary: Dict[str, Any]) -> str:
    config = summary.get("config", {})
    basis = summary.get("basis", {})
    demo = summary.get("demo_vector", {})
    projection = summary.get("projection_metrics", {})
    steering = summary.get("steering_vector", {})
    items = [
        ("model", config.get("model", "")),
        ("layer", config.get("layer", "")),
        ("basis dim", basis.get("basis_dim", "")),
        ("decode states", basis.get("n_states", "")),
        ("shared overlap", fmt(projection.get("overlap_original", 0.0), 3)),
        ("residual overlap", fmt(projection.get("overlap_residual", 0.0), 6)),
        ("vector norm", fmt(steering.get("vector_norm", 0.0), 3)),
        ("mode", demo.get("mode", "")),
    ]
    return "<div class='metric-grid'>" + "".join(
        f"<div class='metric'><span>{esc(label)}</span><strong>{esc(value)}</strong></div>"
        for label, value in items
    ) + "</div>"


def projection_bar(summary: Dict[str, Any]) -> str:
    overlap = float(summary.get("projection_metrics", {}).get("overlap_original", 0.0) or 0.0)
    pct = max(0.0, min(100.0, overlap * 100.0))
    residual = 100.0 - pct
    return f"""
<div class="split">
  <div class="split-track">
    <div class="split-shared" style="width:{pct:.1f}%"></div>
    <div class="split-residual" style="width:{residual:.1f}%"></div>
  </div>
  <div class="split-labels">
    <span>shared {pct:.1f}%</span>
    <span>residual {residual:.1f}%</span>
  </div>
</div>
"""


def render_rank_tables(summary: Dict[str, Any]) -> str:
    snap = summary.get("rank_flip_snapshot", {})
    ranking = snap.get("ranking_alignment", [])
    deployment = snap.get("deployment_selection", [])
    ranking_rows = [
        [
            esc(r.get("pool", "")),
            fmt(r.get("prefill_rho", 0.0), 3),
            f"<strong>{fmt(r.get('decode_rho', 0.0), 3)}</strong>",
            fmt(r.get("delta", 0.0), 3, signed=True),
        ]
        for r in ranking
    ]
    deploy_rows = [
        [
            esc(r.get("proxy", "")),
            fmt(r.get("real_mean", 0.0), 3, signed=True),
            fmt(r.get("real_worst", 0.0), 3, signed=True),
            fmt(r.get("flip_rate", 0.0), 3),
            fmt(r.get("regret_at_1", 0.0), 3),
        ]
        for r in deployment
    ]
    return f"""
<div class="two-col">
  <section class="panel">
    <h2>Ranking Alignment</h2>
    {table(["Pool", "Prefill rho", "Decode rho", "Delta"], ranking_rows)}
  </section>
  <section class="panel">
    <h2>Deployment Selection</h2>
    {table(["Proxy", "REAL mean", "REAL worst", "Flip rate", "Regret@1"], deploy_rows)}
  </section>
</div>
"""


def render_generations(summary: Dict[str, Any]) -> str:
    generations = summary.get("generations", {})
    blocks: List[str] = []
    for prompt, result_map in generations.items():
        cards = []
        ordered = [
            "baseline (no steering)",
            "before DecodeShare projection (demo vector)",
            "removed shared component only",
            "after DecodeShare projection (shared removed)",
        ]
        keys = [k for k in ordered if k in result_map] + [k for k in result_map if k not in ordered]
        for key in keys:
            result = result_map.get(key, {})
            text = result.get("text", "")
            apps = result.get("hook_applications", 0)
            cards.append(
                f"<div class='generation-card'><div class='method'>{esc(key)}</div>"
                f"<div class='apps'>{esc(apps)} hook apps</div><pre>{esc(text)}</pre></div>"
            )
        blocks.append(
            f"<section class='panel'><h2>{esc(prompt)}</h2><div class='generation-grid'>{''.join(cards)}</div></section>"
        )
    return "".join(blocks)


def render_logits(summary: Dict[str, Any]) -> str:
    chunks = []
    for block in summary.get("top_logit_deltas", []):
        rows = [
            [f"<code>{esc(row.get('token', ''))}</code>", esc(row.get("token_id", "")), fmt(row.get("delta_logit", 0.0), 3, signed=True)]
            for row in block.get("rows", [])[:8]
        ]
        chunks.append(
            f"<section class='panel'><h2>{esc(block.get('label', ''))}</h2>"
            f"{table(['Token', 'ID', 'Delta logit'], rows)}</section>"
        )
    return "<div class='three-col'>" + "".join(chunks) + "</div>"


def render_summary(summary: Dict[str, Any], summary_path: Path) -> str:
    config = summary.get("config", {})
    report_path = summary_path.with_name("steering_projection_report.html")
    title = "DecodeShare Steering Projection"
    subtitle = f"{config.get('model', '')} | layer {config.get('layer', '')}"
    report_line = (
        f"<a href='file://{report_path}'>{esc(report_path)}</a>"
        if report_path.exists()
        else esc(report_path)
    )
    return f"""
<div class="dash">
  <header class="hero">
    <div>
      <p class="eyebrow">Decode-time steering repair</p>
      <h1>{esc(title)}</h1>
      <p class="subtitle">{esc(subtitle)}</p>
    </div>
    <div class="pathbox">
      <span>summary</span><code>{esc(summary_path)}</code>
      <span>report</span><code>{report_line}</code>
    </div>
  </header>
  {metric_cards(summary)}
  <section class="panel vector-panel">
    <h2>Vector Split</h2>
    {projection_bar(summary)}
    <p>{esc(summary.get("demo_vector", {}).get("description", ""))}</p>
  </section>
  {render_rank_tables(summary)}
  {render_generations(summary)}
  {render_logits(summary)}
</div>
"""


def load_summary_for_ui(path: str) -> Tuple[str, str]:
    try:
        summary, resolved = load_summary(path)
        return render_summary(summary, resolved), f"Loaded {resolved}"
    except Exception as exc:
        return empty_state(str(exc)), str(exc)


def empty_state(message: str = "") -> str:
    msg = message or f"No summary loaded. Run the CLI demo or load {DEFAULT_SUMMARY}."
    return f"""
<div class="dash">
  <header class="hero">
    <div>
      <p class="eyebrow">Decode-time steering repair</p>
      <h1>DecodeShare Steering Projection</h1>
      <p class="subtitle">{esc(msg)}</p>
    </div>
  </header>
</div>
"""


def run_live_demo(
    model: str,
    device: str,
    dtype: str,
    layer: int,
    out_dir: str,
    demo_vector_mode: str,
    shared_component_scale: float,
    eval_max_new_tokens: int,
    local_files_only: bool,
    trust_remote_code: bool,
) -> Tuple[str, str, str]:
    out = Path(out_dir).expanduser()
    if not out.is_absolute():
        out = REPO_ROOT / out
    cmd = [
        sys.executable,
        str(REPO_ROOT / "demo" / "run_steering_projection_demo.py"),
        "--model",
        str(model),
        "--device",
        str(device),
        "--dtype",
        str(dtype),
        "--layer",
        str(int(layer)),
        "--out_dir",
        str(out),
        "--demo_vector_mode",
        str(demo_vector_mode),
        "--shared_component_scale",
        str(float(shared_component_scale)),
        "--eval_max_new_tokens",
        str(int(eval_max_new_tokens)),
    ]
    if local_files_only:
        cmd.append("--local_files_only")
    if trust_remote_code:
        cmd.append("--trust_remote_code")

    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    summary_path = out / "projection_summary.json"
    if proc.returncode != 0:
        return empty_state("Live run failed."), proc.stdout[-8000:], str(summary_path)
    html_out, status = load_summary_for_ui(str(summary_path))
    return html_out, proc.stdout[-8000:] + "\n" + status, str(summary_path)


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


def render_chat_setup(state: Optional[Dict[str, Any]]) -> str:
    if not state:
        return """
<div class="panel">
  <h2>Interactive Steering Chat</h2>
  <p class="chat-note">
    Initialize a small model to estimate a demo decode-shared basis and a few
    preset steering vectors. This is a qualitative protocol demo, not a claim
    that every preset is repaired or improved by projection.
  </p>
</div>
"""
    rows = []
    for name, record in state["vectors"].items():
        pre = record["prefill"]
        dec = record["decode"]
        rows.append(
            [
                esc(name),
                esc(record["description"]),
                fmt(pre["overlap_original"], 3),
                fmt(dec["overlap_original"], 3),
                fmt(pre["overlap_residual"], 6),
                fmt(dec["overlap_residual"], 6),
            ]
        )
    basis = state["basis_info"]
    return f"""
<div class="panel">
  <h2>Interactive Steering Chat</h2>
  <p class="chat-note">
    Model is initialized. Compare baseline, prefill-estimated steering, and
    decode-estimated steering side by side. Both steering vectors are deployed
    during KV-cached decoding; only the estimation source differs. Treat the
    chat output as an inspection surface; the paper-level claim is the
    rank-flip/validation result, not that every preset visibly improves here.
  </p>
  <div class="metric-grid">
    <div class="metric"><span>model</span><strong>{esc(state["config"]["model"])}</strong></div>
    <div class="metric"><span>layer</span><strong>{esc(state["config"]["layer"])}</strong></div>
    <div class="metric"><span>basis dim</span><strong>{esc(basis.get("basis_dim", ""))}</strong></div>
    <div class="metric"><span>decode states</span><strong>{esc(basis.get("n_states", ""))}</strong></div>
  </div>
  {table(["Preset", "Role", "Prefill-vector shared overlap", "Decode-vector shared overlap", "Prefill residual overlap", "Decode residual overlap"], rows)}
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
) -> Tuple[str, str, str]:
    runtime = demo_runtime()
    args = make_runtime_args(
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
    runtime.set_seed(int(seed))
    model_obj, tokenizer = runtime.load_model_and_tokenizer(args)
    basis, basis_info = runtime.estimate_shared_basis(model_obj, tokenizer, args)

    vectors: Dict[str, Dict[str, Any]] = {}
    for name, preset in VECTOR_PRESETS.items():
        vec_args = make_runtime_args(
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

    state = {
        "runtime": runtime,
        "model": model_obj,
        "tokenizer": tokenizer,
        "basis": basis,
        "basis_info": basis_info,
        "vectors": vectors,
        "config": {
            "model": model,
            "device": device,
            "dtype": dtype,
            "layer": int(layer),
            "basis_k": int(basis_k),
            "max_prompt_tokens": int(max_prompt_tokens),
            "system": system,
            "seed": int(seed),
        },
    }
    session_id = uuid4().hex
    CHAT_SESSIONS[session_id] = state
    return session_id, render_chat_setup(state), f"Initialized {model} with {len(vectors)} steering presets."


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


def chat_prompt_from_history(history: List[Tuple[str, str]], message: str) -> str:
    recent = history[-3:] if history else []
    if not recent:
        return message
    parts = []
    for user_msg, assistant_msg in recent:
        parts.append(f"Previous user: {user_msg}\nPrevious assistant: {assistant_msg}")
    parts.append(f"Current user: {message}")
    return "\n\n".join(parts)


def chat_once(
    session_id: str,
    message: str,
    baseline_history: Optional[List[Tuple[str, str]]],
    prefill_history: Optional[List[Tuple[str, str]]],
    decode_history: Optional[List[Tuple[str, str]]],
    preset: str,
    mode: str,
    alpha: float,
    beta: float,
    inject_first_n: int,
    max_new_tokens: int,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]], str, str]:
    baseline_history = list(baseline_history or [])
    prefill_history = list(prefill_history or [])
    decode_history = list(decode_history or [])
    message = (message or "").strip()
    if not message:
        return baseline_history, prefill_history, decode_history, "", "Enter a prompt first."
    state = CHAT_SESSIONS.get(session_id or "")
    if state is None:
        return baseline_history, prefill_history, decode_history, message, "Initialize the model before chatting."

    runtime = state["runtime"]
    config = state["config"]
    args = make_runtime_args(
        model=config["model"],
        device=config["device"],
        dtype=config["dtype"],
        layer=config["layer"],
        basis_k=config["basis_k"],
        calib_max_new_tokens=8,
        steer_max_new_tokens=8,
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
    baseline = runtime.generate_with_optional_vector(
        state["model"], state["tokenizer"], baseline_prompt, args, vector=None
    )

    preset_record = None if preset == "None" else state["vectors"].get(preset)
    prefill_record = None if preset_record is None else preset_record["prefill"]
    decode_record = None if preset_record is None else preset_record["decode"]

    prefill_vector = None if prefill_record is None else selected_vector(prefill_record, mode, beta)
    prefill_prompt = chat_prompt_from_history(prefill_history, message)
    prefill = runtime.generate_with_optional_vector(
        state["model"], state["tokenizer"], prefill_prompt, args, vector=prefill_vector, alpha=alpha
    )

    decode_vector = None if decode_record is None else selected_vector(decode_record, mode, beta)
    decode_prompt = chat_prompt_from_history(decode_history, message)
    decode = runtime.generate_with_optional_vector(
        state["model"], state["tokenizer"], decode_prompt, args, vector=decode_vector, alpha=alpha
    )

    baseline_history.append((message, baseline.get("text", "")))
    prefill_history.append((message, prefill.get("text", "")))
    decode_history.append((message, decode.get("text", "")))
    status = (
        vector_status(preset, "prefill-est", mode, beta, alpha, prefill_record)
        + f" | prefill hook apps={int(prefill.get('hook_applications', 0))}\n"
        + vector_status(preset, "decode-est", mode, beta, alpha, decode_record)
        + f" | decode hook apps={int(decode.get('hook_applications', 0))}"
    )
    return baseline_history, prefill_history, decode_history, "", status


def clear_chat() -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]], str]:
    return [], [], [], "Cleared chat history."


CSS = """
.gradio-container { max-width: 1280px !important; }
.dash { color: #18212c; }
.hero {
  display: grid;
  grid-template-columns: minmax(0, 1.2fr) minmax(280px, 0.8fr);
  gap: 18px;
  align-items: end;
  padding: 26px;
  border: 1px solid #d9e2ec;
  border-radius: 8px;
  background: linear-gradient(135deg, #f7fafc 0%, #eef5f8 48%, #f8f3ed 100%);
}
.eyebrow { margin: 0 0 8px; color: #526170; font-size: 13px; text-transform: uppercase; letter-spacing: .08em; }
.hero h1 { margin: 0; font-size: 34px; line-height: 1.1; color: #121b24; }
.subtitle { color: #415160; margin: 10px 0 0; }
.pathbox {
  background: rgba(255,255,255,.76);
  border: 1px solid #dce3ea;
  border-radius: 8px;
  padding: 12px;
  display: grid;
  gap: 6px;
}
.pathbox span { color: #627180; font-size: 12px; text-transform: uppercase; }
.pathbox code { white-space: pre-wrap; font-size: 12px; color: #22313f; }
.metric-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin: 16px 0;
}
.metric {
  border: 1px solid #dce3ea;
  border-radius: 8px;
  background: #ffffff;
  padding: 13px 14px;
}
.metric span { display: block; color: #667585; font-size: 12px; text-transform: uppercase; margin-bottom: 7px; }
.metric strong { font-size: 18px; color: #162330; overflow-wrap: anywhere; }
.panel {
  border: 1px solid #dce3ea;
  border-radius: 8px;
  background: #ffffff;
  padding: 18px;
  margin: 14px 0;
}
.panel h2 { margin: 0 0 12px; font-size: 19px; color: #162330; }
.two-col { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
.three-col { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th, td { border-bottom: 1px solid #e6edf3; padding: 8px 9px; text-align: left; vertical-align: top; }
th { color: #405162; background: #f6f8fa; font-weight: 650; }
td { color: #1e2935; }
.split-track { display: flex; height: 32px; background: #e8eef4; border-radius: 8px; overflow: hidden; border: 1px solid #d4dee8; }
.split-shared { background: #c8584f; }
.split-residual { background: #2f78a8; }
.split-labels { display: flex; justify-content: space-between; margin-top: 7px; color: #526170; font-size: 13px; }
.generation-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
.generation-card { border: 1px solid #e1e7ee; border-radius: 8px; padding: 13px; background: #fbfcfe; min-height: 170px; }
.method { font-weight: 700; color: #172432; margin-bottom: 4px; }
.apps { color: #627180; font-size: 12px; margin-bottom: 9px; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 13px; margin: 0; }
@media (max-width: 900px) {
  .hero, .two-col, .three-col, .generation-grid { grid-template-columns: 1fr; }
  .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
"""


def build_app(summary_path: str):
    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Install demo dependencies first: pip install -r demo/requirements-demo.txt") from exc

    initial_html, initial_status = load_summary_for_ui(summary_path)
    with gr.Blocks(css=CSS, title="DecodeShare Steering Demo") as app:
        with gr.Tabs():
            with gr.Tab("Report Dashboard"):
                dashboard = gr.HTML(initial_html, elem_id="dashboard")
                with gr.Row():
                    summary_input = gr.Textbox(value=summary_path, label="projection_summary.json", scale=4)
                    load_button = gr.Button("Load", variant="primary", scale=1)
                status = gr.Textbox(value=initial_status, label="Status", interactive=False)

                with gr.Accordion("Run static report demo", open=False):
                    with gr.Row():
                        model = gr.Textbox(value="TinyLlama/TinyLlama-1.1B-Chat-v1.0", label="Model")
                        out_dir = gr.Textbox(value="outputs/demo_steering_projection_gradio", label="Output dir")
                    with gr.Row():
                        device = gr.Dropdown(["cuda", "cpu", "mps"], value="cuda", label="Device")
                        dtype = gr.Dropdown(["fp16", "bf16", "fp32"], value="fp16", label="Dtype")
                        layer = gr.Number(value=16, precision=0, label="Layer")
                        eval_tokens = gr.Number(value=80, precision=0, label="Eval tokens")
                    with gr.Row():
                        mode = gr.Dropdown(["caa_plus_shared", "caa"], value="caa_plus_shared", label="Vector mode")
                        scale = gr.Slider(1.0, 8.0, value=4.0, step=0.25, label="Shared scale")
                        local_files_only = gr.Checkbox(value=False, label="Local files only")
                        trust_remote_code = gr.Checkbox(value=False, label="Trust remote code")
                    run_button = gr.Button("Run Static Demo", variant="primary")
                    logs = gr.Textbox(label="Run log", lines=16, interactive=False)

            with gr.Tab("Interactive Steering Chat"):
                chat_session = gr.State("")
                chat_intro = gr.HTML(render_chat_setup(None))
                with gr.Accordion("Initialize model and vectors", open=True):
                    with gr.Row():
                        chat_model = gr.Textbox(value="TinyLlama/TinyLlama-1.1B-Chat-v1.0", label="Model")
                        chat_system = gr.Textbox(value="You are a helpful assistant.", label="System prompt")
                    with gr.Row():
                        chat_device = gr.Dropdown(["cuda", "cpu", "mps"], value="cuda", label="Device")
                        chat_dtype = gr.Dropdown(["fp16", "bf16", "fp32"], value="fp16", label="Dtype")
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
                        init_button = gr.Button("Initialize Chat Demo", variant="primary")
                chat_status = gr.Textbox(label="Chat status", interactive=False)

                with gr.Row():
                    preset = gr.Dropdown(["None"] + list(VECTOR_PRESETS.keys()), value="Step-by-step", label="Steering preset")
                    vector_mode = gr.Dropdown(VECTOR_MODES, value="original vector", label="Vector mode")
                    alpha = gr.Slider(-8.0, 8.0, value=1.0, step=0.25, label="Alpha")
                    beta = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="Beta for partial removal")
                with gr.Row():
                    inject_first_n = gr.Slider(1, 128, value=20, step=1, label="Inject first N decode steps")
                    max_new_tokens = gr.Slider(8, 192, value=80, step=4, label="Max new tokens")
                user_message = gr.Textbox(
                    label="Prompt",
                    placeholder="Ask a question, request a style, or test a reasoning prompt.",
                    lines=3,
                )
                with gr.Row():
                    send_button = gr.Button("Send", variant="primary")
                    clear_button = gr.Button("Clear")
                with gr.Row():
                    baseline_chat = gr.Chatbot(label="Baseline", height=420)
                    prefill_chat = gr.Chatbot(label="Prefill-estimated vector", height=420)
                    decode_chat = gr.Chatbot(label="Decode-estimated vector", height=420)

        load_button.click(load_summary_for_ui, inputs=[summary_input], outputs=[dashboard, status])
        run_button.click(
            run_live_demo,
            inputs=[model, device, dtype, layer, out_dir, mode, scale, eval_tokens, local_files_only, trust_remote_code],
            outputs=[dashboard, logs, summary_input],
        ).then(load_summary_for_ui, inputs=[summary_input], outputs=[dashboard, status])

        init_button.click(
            initialize_chat_state,
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
            ],
            outputs=[chat_session, chat_intro, chat_status],
        )
        send_button.click(
            chat_once,
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
        )
        user_message.submit(
            chat_once,
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
        )
        clear_button.click(clear_chat, outputs=[baseline_chat, prefill_chat, decode_chat, chat_status])
    return app


def main() -> None:
    args = parse_args()
    app = build_app(args.summary)
    app.launch(server_name=args.server_name, server_port=args.server_port, share=args.share)


if __name__ == "__main__":
    main()
