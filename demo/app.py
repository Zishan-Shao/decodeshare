#!/usr/bin/env python3
"""Gradio interface for the DecodeShare steering projection demo."""

from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = REPO_ROOT / "outputs" / "demo_steering_projection" / "projection_summary.json"


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
        dashboard = gr.HTML(initial_html, elem_id="dashboard")
        with gr.Row():
            summary_input = gr.Textbox(value=summary_path, label="projection_summary.json", scale=4)
            load_button = gr.Button("Load", variant="primary", scale=1)
        status = gr.Textbox(value=initial_status, label="Status", interactive=False)

        with gr.Accordion("Run live demo", open=False):
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
            run_button = gr.Button("Run Demo", variant="primary")
            logs = gr.Textbox(label="Run log", lines=16, interactive=False)

        load_button.click(load_summary_for_ui, inputs=[summary_input], outputs=[dashboard, status])
        run_button.click(
            run_live_demo,
            inputs=[model, device, dtype, layer, out_dir, mode, scale, eval_tokens, local_files_only, trust_remote_code],
            outputs=[dashboard, logs, summary_input],
        ).then(load_summary_for_ui, inputs=[summary_input], outputs=[dashboard, status])
    return app


def main() -> None:
    args = parse_args()
    app = build_app(args.summary)
    app.launch(server_name=args.server_name, server_port=args.server_port, share=args.share)


if __name__ == "__main__":
    main()
