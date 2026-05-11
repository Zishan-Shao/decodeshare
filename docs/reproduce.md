# Reproduction Guide

There are two reproduction levels.

## Smoke Checks

Smoke checks validate paths, local summaries, and lightweight artifact readers:

```bash
bash scripts/run_all_smoke_tests.sh
```

Section-specific smoke wrappers:

```bash
bash scripts/reproduce_h1_tables.sh
bash scripts/reproduce_ablation_tables.sh
bash scripts/reproduce_table_1_patchback.sh
bash scripts/reproduce_table_3_h3.sh
bash scripts/reproduce_table_2_steering.sh
```

## Full Reruns

Full GPU reruns are recorded in:

- `camera_ready/01_h1_sharedness/COMMANDS.md`
- `camera_ready/02_h2_decode_ablation/COMMANDS.md`
- `camera_ready/03_h2_patchback/COMMANDS.md`
- `camera_ready/04_h3_prefill_decode/COMMANDS.md`
- `camera_ready/05_steering_repair/COMMANDS.md`

As migration continues, those commands should move into the corresponding
`experiments/NN_name/README.md` files and canonical `run_*.py` entry points.
