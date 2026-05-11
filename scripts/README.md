# Scripts

This directory contains stable command wrappers for the camera-ready artifact.

- `run_all_smoke_tests.sh`: runs every lightweight check.
- `reproduce_h1_tables.sh`: H1 sharedness smoke entry point.
- `reproduce_ablation_tables.sh`: H2 ablation smoke entry point.
- `reproduce_table_1_patchback.sh`: patchback smoke entry point.
- `reproduce_table_3_h3.sh`: H3 smoke entry point.
- `reproduce_table_2_steering.sh`: steering repair smoke entry point.
- `full_runs/`: longer historical full-run wrappers, relocated out of the
  repository root.

Full GPU command records currently live in `camera_ready/*/COMMANDS.md`.

## Full-Run Wrapper Notes

`full_runs/run_disturb_cot_loto8_fc_reason.sh` is intentionally kept as the H2 LOTO forced-choice wrapper and now calls `experiments/02_decode_ablation/run_loto_reasoning.py`. This was updated because the paper-facing `fc_eval2048` summaries require the forced-choice path; older generation-only LOTO wrappers remain historical command records rather than canonical reproduction entry points.
