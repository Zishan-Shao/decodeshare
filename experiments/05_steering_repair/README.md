# 05 Steering Repair

Paper role: downstream steering repair and template robustness checks.

Primary outputs:

- Main steering repair table.
- Multi-benchmark repair summary pack.
- Pirate/template sanity checks and appendix robustness tables.

Current command record:

- `camera_ready/05_steering_repair/COMMANDS.md`

Canonical scripts in this folder:

- `steering_vector_reliability_multibench_patch_v3.py`: multibench steering repair.
- `summarize_multibench_v3_full.py`: paper-ready repair tables.
- `mvp_projection_patch_pirate_v5.py`: pirate/template sanity check.

The complete historical bundle remains in `downstream/brittleness/`; this
folder keeps only the paper-facing entry points.

Smoke check:

```bash
bash scripts/reproduce_table_2_steering.sh
```
