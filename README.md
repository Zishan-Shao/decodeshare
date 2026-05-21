# DecodeShare

DecodeShare is a reproducibility package for studying shared decode-time
subspaces in language models. The code estimates cross-task activation
subspaces, tests whether they are causally involved in reasoning behavior, and
reuses them for decode ablations, patchback, prefill/decode mismatch analysis,
and steering repair.

<p align="center">
  <img src="paper_artifacts/figures/decodeshare_pipeline.jpg" alt="DecodeShare pipeline" width="96%">
</p>

## What This Repo Contains

- Canonical experiment entry points for the paper's main hypotheses.
- Public shell wrappers for smoke checks and full GPU reruns.
- Lightweight paper figures under `paper_artifacts/figures/`.
- Downstream bundles for patchback, steering repair, and rebuttal-style checks.

Large raw model outputs are intentionally not committed. Full reruns write to
`outputs/` by default, and long-running jobs can be inspected with `DRY_RUN=1`
before launching model inference.

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

### Steering Repair

The steering experiments test whether DecodeShare-style shared subspaces can
stabilize downstream steering behavior across tasks and prompt templates. They
are the main downstream robustness check in the public release.

Layer-28 Llama-2-7B-chat-hf ranking protocol:

| Ranking signal | Spearman with real decode |
|---|---:|
| Traditional prefill | -0.22 |
| Decode-aligned | 0.60 |

Layer-28 Llama-2-7B-chat-hf repair controls:

| Method | Mean delta | Worst-template delta | Template std |
|---|---:|---:|---:|
| Original steering | -0.005 | -0.017 | 0.010 |
| Shared repair | -0.004 | -0.013 | 0.006 |
| Random control | -0.005 | -0.013 | 0.007 |
| PCA control | -0.004 | -0.013 | 0.006 |
| Prefill-PCA control | -0.008 | -0.017 | 0.009 |
| Norm-matched shrink | -0.004 | -0.013 | 0.008 |

| Worst-template comparison | Shared repair win rate |
|---|---:|
| vs. random control | 60% |
| vs. PCA control | 20% |
| vs. prefill-PCA control | 100% |
| vs. norm-matched shrink | 80% |

Higher is better for Spearman and deltas; lower is better for template std.
Source summaries:
`downstream/rebuttal/results/rebuttal_20260208_212945/ranking_flip.json` and
`downstream/rebuttal/results/rebuttal_20260208_212945/repair_controls.json`.

```bash
bash scripts/reproduce_table_2_steering.sh
```

Implementation and command records:

- `experiments/05_steering_repair/`
- `scripts/05_steering_repair/COMMANDS.md`

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
bash scripts/reproduce_table_2_steering.sh
bash scripts/reproduce_table_3_h3.sh
```

Section command notes:

- `scripts/01_h1_sharedness/COMMANDS.md`
- `scripts/02_h2_decode_ablation/COMMANDS.md`
- `scripts/03_h2_patchback/COMMANDS.md`
- `scripts/04_h3_prefill_decode/COMMANDS.md`
- `scripts/05_steering_repair/COMMANDS.md`

## Repository Layout

```text
decodeshare/
  experiments/              # Paper-section experiment code
  scripts/                  # Smoke checks and full reproduction wrappers
  src/decodeshare/          # Shared package namespace
  downstream/               # Patchback, steering, and rebuttal bundles
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
| Steering repair | Test downstream robustness and steering repair | `experiments/05_steering_repair/` |

## Notes

- GPU reruns are expensive; start with `DRY_RUN=1` and smoke checks.
- Raw JSON/PT/NPY artifacts should stay outside git unless intentionally
  curated.
- The public tree is organized for reproduction first; historical exploratory
  files were removed or moved out of the main workflow.
