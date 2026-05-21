# 04 Prefill Decode

Paper role: H3, estimator-deployment mismatch between prefill and decode
subspaces.

Primary outputs:

- Main H3 table.
- Prefill/decode sweep summaries.
- Appendix mismatch and contrast tables.

Current command record:

- `scripts/04_h3_prefill_decode/COMMANDS.md`

Canonical scripts now in this folder:

- `run_h3_grid_generation.py`
- `run_h3_grid_reasoning_v2.py`
- `run_h3_grid_reasoning_src.py`
- `run_prefill_decode_generation.py`
- `run_prefill_decode_reasoning.py`
- `run_prefill_decode_reasoning_sweeps.py`
- `summarize_h3_grid.py`

Smoke check:

```bash
bash scripts/reproduce_table_3_h3.sh
```
