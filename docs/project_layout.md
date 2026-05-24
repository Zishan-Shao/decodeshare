# Project Layout and File Contract

This document defines the public camera-ready layout. The goal is that a new
reader can find the code for each paper result without seeing exploratory
folders or review-phase scratch work as first-class project structure.

## Top-Level Directories

- `decodeshare/`: reusable method/library code shared across experiments.
- `experiments/`: one folder per paper experiment block, ordered by paper body.
- `downstream/`: downstream steering, patchback, mismatch diagnostics, and
  legacy provenance grouped outside the main paper-section flow.
- `scripts/`: command-line wrappers for smoke tests and reproducibility entry
  points.
- `paper_artifacts/`: compact table/figure artifacts and summaries.
- `docs/`: setup, data/model, reproduction, and troubleshooting notes.
- `tests/`: lightweight tests that do not require long GPU runs.

Historical top-level directories such as `Hype1/`, `patch_back/`,
`brittleness/`, `reasoning/`, `lateruse/`, and `results/` are not
part of the public root layout. Their current camera-ready homes are
`experiments/`, `downstream/`, `paper_artifacts/`, and `decodeshare/`.

## DecodeShare Package Format

The public package namespace is intentionally flat. Shared helpers should live
directly under `decodeshare/` as focused modules such as `subspace.py`,
`sharedness.py`, `benchmark_dataloaders.py`, `eval_perf.py`, and
`decode_loto.py`. Historical compatibility shims may stay at the package root
when needed, but exploratory subpackages should not be first-class public
structure.

## Experiment Folder Format

Each `experiments/NN_name/` folder should contain:

- `README.md`: paper outputs covered, source provenance, and commands.
- `configs/`: YAML or JSON configs for model, layer, seed, dataset, and method
  settings.
- `run_*.py`: experiment implementations; executable shell wrappers live under `scripts/`.
- `summarize_*.py`: scripts that convert raw outputs into compact paper
  summaries.

Config filenames should encode the main axes when useful:

```text
model=<model_slug>__layer=<layer>__k=<rank>__seed=<seed>.yaml
```

Raw result filenames should include enough metadata to audit a result without
opening the file:

```text
<experiment>__model=<model_slug>__layer=<layer>__k=<rank>__seed=<seed>.<ext>
```

## Artifact Folder Format

Use `paper_artifacts/` for small, reviewable files:

- `paper_artifacts/tables/*.tex`
- `paper_artifacts/tables/*.csv`
- `paper_artifacts/figures/*.pdf`
- `paper_artifacts/figures/*.png`
- `paper_artifacts/summaries/*.md`

Large raw artifacts and copied paper PDFs should stay outside git. Record raw
artifact paths, checksums, and generation commands in `MANIFEST.md`.

## Command Policy

Every paper result should have two command levels:

- A smoke command that runs quickly and checks local wiring.
- A full command record with model, dataset, layer, rank, seed, and output path.

The current smoke command entry point is:

```bash
bash scripts/run_all_smoke_tests.sh
```

Full commands are recorded under `scripts/*/COMMANDS.md`; executable rerun
wrappers live in `scripts/full_runs/`.
