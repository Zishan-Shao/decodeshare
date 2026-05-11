# Project Layout and File Contract

This document defines the public camera-ready layout. The goal is that a new
reader can find the code for each paper result without seeing exploratory
folders or rebuttal-only scratch work as first-class project structure.

## Top-Level Directories

- `src/decodeshare/`: reusable package code shared across experiments.
- `experiments/`: one folder per paper experiment block, ordered by paper body.
- `downstream/`: patchback, brittleness/steering, and rebuttal bundles grouped
  outside the main paper-section flow.
- `scripts/`: command-line wrappers for smoke tests and reproducibility entry
  points.
- `paper_artifacts/`: final PDF, compact table/figure artifacts, and summaries.
- `docs/`: setup, data/model, reproduction, and troubleshooting notes.
- `tests/`: lightweight tests that do not require long GPU runs.
- `camera_ready/`: temporary migration records and mock-test command logs.

Historical top-level directories such as `Hype1/`, `patch_back/`,
`brittleness/`, `reasoning/`, `lateruse/`, `rebuttal/`, and `results/` are not
part of the public root layout. Their current camera-ready homes are
`experiments/`, `downstream/`, `paper_artifacts/`, and `src/`.

## Experiment Folder Format

Each `experiments/NN_name/` folder should contain:

- `README.md`: paper outputs covered, source provenance, and commands.
- `configs/`: YAML or JSON configs for model, layer, seed, dataset, and method
  settings.
- `run_*.py` or `run_*.sh`: canonical full-run entry points.
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

- `paper_artifacts/DecodeShare_camera_ready.pdf`
- `paper_artifacts/tables/*.tex`
- `paper_artifacts/tables/*.csv`
- `paper_artifacts/figures/*.pdf`
- `paper_artifacts/figures/*.png`
- `paper_artifacts/summaries/*.md`

Large raw artifacts should stay outside git. Record their path, checksum, and
generation command in `MANIFEST.md`.

## Command Policy

Every paper result should have two command levels:

- A smoke command that runs quickly and checks local wiring.
- A full command record with model, dataset, layer, rank, seed, and output path.

The current smoke command entry point is:

```bash
bash scripts/run_all_smoke_tests.sh
```

Full commands are currently recorded under `camera_ready/*/COMMANDS.md` while
the code is being promoted into `experiments/`.
