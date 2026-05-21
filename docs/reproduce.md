# Reproduction Guide

There are two reproduction levels.

## Smoke Checks

Smoke checks validate imports, CLI wiring, and lightweight summary helpers:

```bash
bash scripts/run_all_smoke_tests.sh
```

Section-specific smoke wrappers:

```bash
bash scripts/01_h1_sharedness/run_mock.sh
bash scripts/02_h2_decode_ablation/run_mock.sh
bash scripts/03_h2_patchback/run_mock.sh
bash scripts/04_h3_prefill_decode/run_mock.sh
bash scripts/05_steering_repair/run_mock.sh
```

## Full Reruns

Full GPU reruns are executable from:

- `scripts/reproduce_h1_tables.sh`
- `scripts/reproduce_ablation_tables.sh`
- `scripts/reproduce_table_1_patchback.sh`
- `scripts/reproduce_table_2_steering.sh`
- `scripts/reproduce_table_3_h3.sh`

Use `DRY_RUN=1` to print any command without running model inference:

```bash
DRY_RUN=1 bash scripts/reproduce_ablation_tables.sh
```

Section command notes live in `scripts/*/COMMANDS.md`; lower-level wrappers live
in `scripts/full_runs/`.
