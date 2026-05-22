# Downstream Experiments

This directory keeps paper-facing downstream experiments plus a small amount of
legacy provenance code that still feeds those runs.

- `steering_rank_flip/`: decode-aligned steering validation, cross-method
  candidate pools, diagnostic rank flip, and deployment-selection scripts.
- `steering_controls/`: projection/repair controls for steering vectors.
- `prefill_decode_mismatch/`: PCA mismatch diagnostics for prefill vs decode
  hidden-state distributions.
- `patchback_legacy/`: historical patchback, open-answer patching, and
  transfer-control bundle kept for provenance. Canonical paper-facing patchback
  entry points live in `experiments/03_patchback/`.
- `brittleness/`: legacy steering robustness and template-sensitivity scripts.
  This remains as provenance for older vector-generation workflows; it is not
  the main public entry point for steering rank-flip reproduction.

Generated `results/` directories should stay out of git. The public
reproduction wrappers write fresh outputs under repository-level `outputs/`.
