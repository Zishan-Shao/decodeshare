# Migration Plan

The camera-ready branch currently keeps historical directories so we can verify
provenance while building the public artifact. They should not remain as the
final project surface.

## Keep as Public Structure

- `src/decodeshare/`
- `experiments/`
- `scripts/`
- `paper_artifacts/`
- `docs/`
- `tests/`

## Promoted From Historical Sources

- `Hype1/` -> `experiments/01_sharedness/` and `paper_artifacts/h1_results/`
- `reasoning/`, root `src/*.py` -> `experiments/02_decode_ablation/` and
  `experiments/04_prefill_decode/`
- `patch_back/` -> `downstream/patch_back/`
- `brittleness/` -> `downstream/brittleness/`
- `rebuttal/` and rebuttal-only `results/` -> `downstream/rebuttal/`
- `joint_subspace_large/` -> `src/joint_subspace_large/`
- `lateruse/` -> `experiments/04_prefill_decode/legacy_lateruse/`

## Exclude Unless Needed

- ad hoc run folders, exploratory notebooks, and obsolete result dumps

## Release Blockers

- Choose and add a repository license.
- Add `CITATION.cff` after the final title, author list, and venue metadata are
  settled.
- Decide whether `downstream/rebuttal/` should remain in the final public
  artifact or be referenced externally only.

Before release, every retained file should either be in the public structure or
be explicitly justified in `MANIFEST.md`.
