# DecodeShare Package Namespace

`decodeshare/` contains reusable library code shared by the paper experiments.
Paper-specific runners and provenance scripts live under `experiments/` and
`downstream/`.

## Current Structure

```text
decodeshare/
  __init__.py
  activations.py                  # H1 activation collection helpers
  benchmark_dataloaders.py        # Benchmark loading, prompt building, answer checks
  decode_loto.py                  # Shared decode-stage LOTO utilities
  eval_perf.py                    # Forced-choice and generation evaluation helpers
  sharedness.py                   # H1 sharedness estimation and null tests
  subspace.py                     # Shared basis construction and intervention helpers
  disturb_cross_task_all_shared.py # Legacy compatibility shim for old imports
```

## Public Modules

- `benchmark_dataloaders`: shared HF benchmark loading, prompt construction,
  answer parsing, correctness checks, and deterministic seeding.
- `subspace`: shared subspace construction, model-layer discovery, shared-basis
  scoring, and subspace removal hooks.
- `sharedness`: H1 sharedness prompt loading, decode-state collection,
  shared-component scoring, null tests, and model loading helpers.
- `activations`: activation-collection helper namespace used by H1 diagnostics.
- `eval_perf`: forced-choice evaluation, decode/prefill shared-basis helpers,
  and shared evaluation utilities used by H2/H3 scripts.
- `decode_loto`: shared decode-stage LOTO utilities used by patchback and
  brittleness experiments.

`disturb_cross_task_all_shared.py` is intentionally only a compatibility shim;
new code should import from `decodeshare.subspace`.
