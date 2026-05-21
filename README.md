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

Compact local summary from the layer-28 Llama-2-7B-chat-hf steering runs:

| Steering check | Summary |
|---|---|
| Ranking protocol | Spearman with real decode: traditional prefill `-0.22`; decode-aligned `0.60` |
| Worst-template repair | Mean worst-template delta improves from original `-0.017` to shared repair `-0.013` |
| Template stability | Mean template std drops from original `0.010` to shared repair `0.006` |
| Control comparison | Shared repair wins by worst-template delta against norm-matched shrink on `80%` of vectors and prefill-PCA on `100%`; random/PCA controls are included in the source JSON |

Higher is better for Spearman and deltas; lower is better for template std.
The source summaries are
`downstream/rebuttal/results/rebuttal_20260208_212945/ranking_flip.json` and
`downstream/rebuttal/results/rebuttal_20260208_212945/repair_controls.json`.

```bash
bash scripts/reproduce_table_2_steering.sh
```

Implementation and command records:

- `experiments/05_steering_repair/`
- `scripts/05_steering_repair/COMMANDS.md`

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
