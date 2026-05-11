# DecodeShare Package Namespace

This package is the target home for reusable code promoted from the historical
experiment scripts.

Planned module boundaries:

- `data`: dataset adapters and prompt/task loading.
- `models`: model/tokenizer loading and device placement helpers.
- `subspace`: basis construction, projection, and sharedness utilities.
- `collect`: activation collection entry points.
- `interventions`: ablation and patching primitives.
- `evaluation`: task metrics and aggregation helpers.
- `stats`: bootstrap, confidence interval, and multiple-comparison utilities.
- `plotting`: table and figure formatting helpers.

Paper-specific entry points live under `experiments/`. Compatibility helpers
that still need their historical import path can live as sibling packages under
`src/`, for example `src/joint_subspace_large/`.
