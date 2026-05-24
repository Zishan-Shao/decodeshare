# Patchback Provenance

This folder keeps downstream patchback provenance scripts in the same
`exp_*.py` style used by the other downstream folders.

Current entry points:

- `exp_patchback_loto.py`: local decode-stage LOTO/helper run used by the
  provenance patchback scripts.
- `exp_subspace_patching_transfer.py`: multiple-choice subspace patchback on
  flip sets.
- `exp_subspace_patching_transfer_controls.py`: enhanced transfer controls,
  including extra shared-subspace donor controls.
- `exp_openanswer_patching.py`: open-answer patchback for math/code-style
  tasks.
- `exp_flipset_alpha_transfer.py`: flip-set alpha sweep and transfer-donor
  patching.
- `exp_unique_vs_ih_tc.py`: uniqueness checks against induction-head and sparse
  head-subset baselines.
- `summarize_patchback_results.py`: JSON-to-summary aggregation for patchback
  outputs.

The canonical paper-facing patchback scripts live in `experiments/03_patchback/`.
This folder is retained for historical/provenance workflows.
