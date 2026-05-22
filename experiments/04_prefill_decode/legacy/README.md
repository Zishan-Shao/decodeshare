# H3 Legacy Variants

This folder keeps older or exploratory H3 prefill/decode scripts for provenance.
The public reproduction path uses `../run_h3_grid_reasoning.py` and
`../run_prefill_decode_reasoning_sweeps.py`.

- `run_h3_grid_reasoning_src.py`: earlier decode-intervention H3 grid.
- `run_h3_grid_reasoning_legacy.py`: refactored forced-choice-only variant.
- `run_h3_grid_generation.py`: generation-only H3 grid prototype.
- `run_prefill_decode_generation.py`: generation-capable prefill/decode alignment runner.
- `run_prefill_decode_reasoning.py`: older reasoning alignment runner.
- `prefill_vs_decode_alignment_experiment_reasoning*.py`: historical alignment scripts.
- `cross_layer_shared_workspace_scan.py`: cross-layer diagnostic prototype.

Some legacy files reference historical script names in comments, but runnable
imports point back to `../h3_decode_subspace_helpers.py` where needed.
