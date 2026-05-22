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

Current public modules:

- `benchmark_dataloaders`: shared HF benchmark loading, prompt construction,
  answer parsing, correctness checks, and deterministic seeding.
- `eval_perf`: forced-choice evaluation, decode/prefill shared-basis helpers,
  and shared evaluation utilities used by H2/H3 scripts.
- `joint_subspace_large`: shared subspace construction/intervention helpers
  used by the paper experiments.
