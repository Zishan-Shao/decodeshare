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

Paper-specific entry points live under `experiments/`. Historical compatibility
helpers that are still shared by several experiment scripts live under
`decodeshare/joint_subspace_large/`.
