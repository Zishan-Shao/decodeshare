# DecodeShare

This repository contains the camera-ready artifact and reproducibility package for
DecodeShare.

The camera-ready branch is organized as a clean public project. Historical
experiment directories are retained only as migration sources until their
canonical scripts have been moved into the public layout.

## Quick Start

```bash
conda env create -f environment.yml
conda activate flashsvd
pip install -e .
bash scripts/run_all_smoke_tests.sh
```

The smoke tests check that the curated paper artifacts and lightweight
summarizers are wired correctly. They do not launch long GPU jobs.

Current cluster constraint: only `Node0` and `Node1` are available for reruns
until this note is updated.

## Project Layout

```text
decodeshare/
  README.md
  environment.yml
  pyproject.toml
  src/decodeshare/              # public Python package namespace
  experiments/                  # paper-section experiment entry points
  scripts/                      # smoke and reproduction command wrappers
  downstream/                   # downstream/rebuttal/patchback bundles
  paper_artifacts/              # PDF, tables, figures, compact summaries
  docs/                         # setup, data/model, reproduction notes
  tests/                        # lightweight local tests
  camera_ready/                 # migration notes and mock-test command records
  MANIFEST.md                   # canonical artifact manifest
  CAMERA_READY_CODE_INVENTORY.md
```

The public experiment order follows the paper body:

1. `experiments/01_sharedness/`: H1 shared decode-time structure.
2. `experiments/02_decode_ablation/`: H2 decode ablation, LOTO, and controls.
3. `experiments/03_patchback/`: H2 sufficiency and patchback transfer.
4. `experiments/04_prefill_decode/`: H3 prefill/decode deployment mismatch.
5. `experiments/05_steering_repair/`: steering repair and robustness checks.

`downstream/` contains the consolidated downstream bundles requested for the
camera-ready cleanup:

- `downstream/patch_back/`
- `downstream/brittleness/`
- `downstream/rebuttal/`

See `docs/project_layout.md` for the file-format contract and naming rules.

## Reproduction Levels

- Smoke checks: `bash scripts/run_all_smoke_tests.sh`
- Paper-section wrappers: `scripts/reproduce_*.sh`
- Full command records: `camera_ready/*/COMMANDS.md`
- Artifact inventory: `MANIFEST.md`

Large raw outputs are not committed by default. Commit compact `.md`, `.csv`,
`.tex`, and final figure/table assets; keep multi-GB raw JSON/PT/NPY artifacts in
an external artifact store and record their path, checksum, model, layer, and
seed in `MANIFEST.md`.

## Working Rule

Do not bulk-copy the old experiment workspace into this branch. Bring over only
canonical scripts, compact summaries, paper tables/figures, and artifact
manifests. Historical top-level directories have been promoted into
`experiments/`, moved under `downstream/`, or moved into `paper_artifacts/`.
