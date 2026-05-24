# 05 Steering Controls

Paper role: downstream steering-vector controls and template robustness checks.

Primary outputs:

- Steering projection-control table.
- Multi-benchmark repair summary pack.
- Pirate/template sanity checks and appendix robustness tables.

Current command records:

- `scripts/05_steering_rank_flip/COMMANDS.md` for the public rank-flip tables.
- `scripts/full_runs/run_steering_repair.sh` for this legacy multibench bundle.

Canonical scripts in this folder:

- `steering_vector_reliability_multibench_patch.py`: multibench steering controls.
- `summarize_multibench_full.py`: paper-ready repair tables.
- `mvp_projection_patch_pirate.py`: pirate/template sanity check.

The organized downstream provenance bundle remains in `downstream/brittleness/`;
this folder keeps only the paper-facing entry points.

Smoke check:

```bash
bash scripts/reproduce_table_2_steering.sh
```
