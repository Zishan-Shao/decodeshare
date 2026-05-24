# DecodeShare

DecodeShare is a protocol for finding task-general decision channels in
KV-cached decode-time hidden states. It estimates low-dimensional subspaces
shared across tasks, then tests their causal role by removing them only during
decoding under matched intervention budgets.

The artifact reproduces the paper's sharedness, decode ablation, patchback,
prefill/decode mismatch, and steering-ranking experiments, including the result
that decode-time validation is a stronger proxy for held-out steering utility
than prefill-based validation.

<p align="center">
  <img src="paper_artifacts/figures/decodeshare_pipeline.jpg" alt="DecodeShare pipeline" width="96%">
</p>

## What This Repo Contains

- Canonical experiment entry points for the paper's main hypotheses.
- Public shell wrappers for smoke checks and full GPU reruns.
- Lightweight paper figures under `paper_artifacts/figures/`.
- Downstream bundles for patchback, steering ranking, and deployment checks.

Large raw model outputs are intentionally not committed. Full reruns write to
`outputs/` by default, and long-running jobs can be inspected with `DRY_RUN=1`
before launching model inference.

<p align="center">
  <img src="paper_artifacts/figures/decodeshare_steering_demo.gif" alt="DecodeShare GPU steering demo comparing baseline, prefill steering, and decode steering" width="96%">
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/Zishan-Shao/decodeshare-demo">Try the free CPU Space</a>
</p>

## Core Results

### Leave-One-Task-Out Decode Ablation

The main causal check estimates the shared decode-time subspace from all but
one task, then evaluates whether removing that subspace affects the held-out
task. This keeps the intervention from reusing the evaluation task's own
activations.

<p align="center">
  <img src="paper_artifacts/figures/2_loto.png" alt="Leave-one-task-out decode ablation results" width="96%">
</p>

Reproduce this section with:

```bash
bash scripts/reproduce_ablation_tables.sh
```

### Steering Ranking and Deployment

Decode-aligned validation is the steering utility check: it asks whether the
ranking signal used to select vectors matches held-out decode-time behavior.

Ranking alignment across vector pools:

<div align="center">

<table>
  <thead>
    <tr>
      <th align="left">Pool</th>
      <th align="right">Prefill rho</th>
      <th align="right">Decode rho</th>
      <th align="right">Delta</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>CAA contrastive</td>
      <td align="right">-0.370</td>
      <td align="right"><strong>0.700</strong></td>
      <td align="right">+1.070</td>
    </tr>
    <tr>
      <td>Instruction</td>
      <td align="right">0.172</td>
      <td align="right"><strong>0.767</strong></td>
      <td align="right">+0.595</td>
    </tr>
    <tr>
      <td>SAE features</td>
      <td align="right">-0.064</td>
      <td align="right"><strong>0.594</strong></td>
      <td align="right">+0.659</td>
    </tr>
    <tr>
      <td>Diagnostic</td>
      <td align="right">0.065</td>
      <td align="right"><strong>0.700</strong></td>
      <td align="right">+0.635</td>
    </tr>
  </tbody>
</table>

</div>

Held-out REAL utility after selecting vectors by each proxy:

<div align="center">

<table>
  <thead>
    <tr>
      <th align="left">Proxy</th>
      <th align="right">REAL mean</th>
      <th align="right">REAL worst</th>
      <th align="right">Flip rate</th>
      <th align="right">Regret@1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Prefill-aligned</td>
      <td align="right">-0.002</td>
      <td align="right">-0.003</td>
      <td align="right">0.750</td>
      <td align="right">0.016</td>
    </tr>
    <tr>
      <td>Mixed stages</td>
      <td align="right">-0.002</td>
      <td align="right">-0.003</td>
      <td align="right">0.750</td>
      <td align="right">0.016</td>
    </tr>
    <tr>
      <td><strong>Decode-aligned</strong></td>
      <td align="right"><strong>+0.011</strong></td>
      <td align="right"><strong>+0.010</strong></td>
      <td align="right"><strong>0.083</strong></td>
      <td align="right"><strong>0.003</strong></td>
    </tr>
  </tbody>
</table>

</div>

The pools contain 32 CAA contrastive vectors, 64 instruction-derived vectors,
64 SAE feature directions, and 100 diagnostic directions. Higher is better for
rho, REAL mean, and REAL worst; lower is better for flip rate and regret.

```bash
bash scripts/reproduce_steering_flip_tables.sh
```

Implementation and command records:

- `downstream/steering_rank_flip/`
- `downstream/steering_controls/`
- `scripts/05_steering_rank_flip/COMMANDS.md`

### Shared Channel Contents

A 32D Llama-2-7B layer-10 workspace splits into a 3D readout slice
`Q_out` and a 29D residual core `Q_core`.

| Ablated subspace | Dim. | Acc. | Delta Acc. |
|---|---:|---:|---:|
| Full shared | 32 | 28.5 | -15.6 |
| `Q_out` | 3 | 44.9 | +0.8 |
| `Q_core` | 29 | 27.3 | -16.8 |

Forced-choice baseline accuracy is 44.1%.

| Probe tag | `Q_core` AP | `Q_out` AP | Delta |
|---|---:|---:|---:|
| Reasoning markers | 0.564 | 0.041 | +0.523 |
| Step markers | 0.169 | 0.029 | +0.140 |
| Digits | 0.966 | 0.132 | +0.834 |
| Equation symbols | 0.673 | 0.055 | +0.618 |

Layer-28 vocab-alignment mean overlap:

| Token family | Shared | Nonshared | Prefill shared |
|---|---:|---:|---:|
| Answer scaffold | 0.283 | 0.213 | 0.208 |
| Correctness markers | 0.207 | 0.189 | 0.182 |
| Confidence markers | 0.234 | 0.195 | 0.221 |
| Newline | 0.374 | 0.204 | 0.190 |
| Digits | 0.314 | 0.261 | 0.159 |
| Sentiment markers | 0.171 | 0.176 | 0.182 |

## Quick Start

```bash
conda env create -f environment.yml
conda activate decodeshare
bash scripts/run_all_smoke_tests.sh
```

The smoke suite checks imports, command-line wiring, and lightweight summary
helpers. It does not download models or run long GPU experiments.
The conda environment installs the local package in editable mode through
`pyproject.toml`, so a separate `pip install -e .` step is not needed.

## Demo

For a compact visual example of how DecodeShare changes a steering vector, run
the Llama steering projection demo:

```bash
python demo/run_steering_projection_demo.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --device cuda \
  --layer 28 \
  --demo_vector_mode caa_plus_shared \
  --out_dir outputs/demo_steering_projection
```

The demo estimates a small decode-time shared basis, decomposes a contrastive
steering vector into shared and residual components, includes the steering
rank-flip snapshot, prints an example steering-vector before/after, and writes
an HTML report. The default demo amplifies the vector's shared component for visual
contrast; use `--demo_vector_mode caa` for the untouched CAA-style vector. See
`demo/README.md` for a smaller TinyLlama smoke-run option.

An optional Gradio app keeps only the interactive steering chat. It compares
baseline, prefill-estimated steering, and decode-estimated steering side by
side:

```bash
pip install -r demo/requirements-demo.txt
python demo/app.py --server_port 7860
```

The app still has to load the model. To skip repeated basis/vector estimation,
precompute the optional cache once:

```bash
python demo/build_interactive_cache.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --device cuda \
  --layer 16 \
  --cache demo/assets/interactive_tinyllama_chat_cache.pt
```

The cached TinyLlama demo uses example-specific defaults: the pirate prompt uses
`alpha=1.5` to avoid repetitive text, while more structural prompts use stronger
values for visible differences under greedy decoding.

Hosted CPU Space: https://huggingface.co/spaces/Zishan-Shao/decodeshare-demo

The hosted Space uses Hugging Face's free CPU hardware by default. It may be
sleeping when opened and can take time to cold-start. The same Gradio demo can
also run locally on CPU with the bundled cache, though generation is much slower
than on GPU. Responses stream sequentially so CPU runs remain trackable.

## Reproducing Experiments

Full rerun wrappers live in `scripts/`. They share common overrides such as
`GPU_ID`, `MODEL`, `LAYER`, `N_EVAL`, `OUT_DIR`, and `DRY_RUN`.

```bash
# Print the exact commands without running model inference.
DRY_RUN=1 bash scripts/reproduce_ablation_tables.sh

# Run section-level reproductions.
bash scripts/reproduce_h1_tables.sh
bash scripts/reproduce_ablation_tables.sh
bash scripts/reproduce_table_1_patchback.sh
bash scripts/reproduce_steering_flip_tables.sh
bash scripts/reproduce_table_3_h3.sh
```

Section command notes:

- `scripts/01_h1_sharedness/COMMANDS.md`
- `scripts/02_h2_decode_ablation/COMMANDS.md`
- `scripts/03_h2_patchback/COMMANDS.md`
- `scripts/04_h3_prefill_decode/COMMANDS.md`
- `scripts/05_steering_rank_flip/COMMANDS.md`

## Repository Layout

```text
decodeshare/
  demo/                     # Compact steering vector projection demo
  decodeshare/              # Shared method/library code
  experiments/              # Paper-section experiment code
  scripts/                  # Smoke checks and full reproduction wrappers
  downstream/               # Downstream experiments and legacy provenance bundles
  paper_artifacts/figures/  # Lightweight paper figures for browsing
  docs/                     # Setup and reproduction notes
  tests/                    # Lightweight local checks
```

## Experiment Map

| Section | Purpose | Main entry points |
|---|---|---|
| H1 sharedness | Estimate and test shared decode-time structure | `experiments/01_sharedness/` |
| H2 decode ablation | Remove shared decode components with LOTO controls | `experiments/02_decode_ablation/` |
| H2 patchback | Patch shared subspaces back into corrupted decisions | `experiments/03_patchback/` |
| H3 prefill/decode | Compare estimator and intervention timing | `experiments/04_prefill_decode/` |
| Steering rank flip | Compare prefill/decode steering validation against held-out deployment | `downstream/steering_rank_flip/`, `scripts/05_steering_rank_flip/` |
| Steering controls | Projection controls and robustness checks for steering vectors | `downstream/steering_controls/`, `experiments/05_steering_controls/` |

## Notes

- GPU reruns are expensive; start with `DRY_RUN=1` and smoke checks.
- Raw JSON/PT/NPY artifacts should stay outside git unless intentionally
  curated.
- The public tree is organized for reproduction first; historical exploratory
  files were removed or moved out of the main workflow.

## License

This repository is released under the MIT License. See [LICENSE](LICENSE).
