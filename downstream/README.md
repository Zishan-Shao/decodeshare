# Downstream Experiments

This directory keeps paper-facing downstream experiments plus a small amount of
legacy provenance code that still feeds those runs.

- `steering_rank_flip/`: decode-aligned steering validation, cross-method
  candidate pools, diagnostic rank flip, and deployment-selection scripts.
- `steering_controls/`: projection/repair controls for steering vectors.
- `prefill_decode_mismatch/`: PCA mismatch diagnostics for prefill vs decode
  hidden-state distributions.
- `patchback_provenance/`: organized `exp_*.py` provenance scripts for
  patchback, open-answer patching, transfer controls, and uniqueness checks.
  Use `experiments/03_patchback/` for canonical paper-facing patchback
  reproduction.
- `brittleness/`: organized `exp_*.py` provenance scripts for steering
  robustness, template-sensitivity, pirate projection checks, and reasoning
  LOTO brittleness checks. The canonical paper-facing steering-control copies
  live in `experiments/05_steering_controls/`.

Generated `results/` directories should stay out of git. The public
reproduction wrappers write fresh outputs under repository-level `outputs/`.
