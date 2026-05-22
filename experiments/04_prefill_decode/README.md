# 04 Prefill Decode

Paper role: H3, estimator-deployment mismatch between prefill and decode
subspaces.

Primary outputs:

- Main H3 table.
- Prefill/decode sweep summaries.
- Appendix mismatch and contrast tables.

Current command record:

- `scripts/04_h3_prefill_decode/COMMANDS.md`

Canonical scripts in this folder:

- `run_h3_grid_reasoning.py`: main H3 2x2 estimator/intervention grid.
- `run_prefill_decode_reasoning_sweeps.py`
- `h3_decode_subspace_helpers.py`: local decode-shared subspace helper used by
  the H3 grid runner.
- `analysis/summarize_h3_grid.py`: JSON-to-table summary for H3 grid runs.

Smoke check:

```bash
bash scripts/reproduce_table_3_h3.sh
```
