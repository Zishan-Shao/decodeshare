# Reproduction Scripts

This directory contains the public reproduction entry points. Section folders
hold lightweight smoke checks and command notes; `full_runs/` contains GPU
rerun wrappers.

Current cluster constraint: use only `Node0` and `Node1`.

## Environment

Run from an activated environment, or set `PYTHON_CMD` explicitly:

```bash
conda activate flashsvd
# or:
PYTHON_CMD="conda run -n flashsvd python" bash scripts/run_all_smoke_tests.sh
```

Common overrides:

- `GPU_ID=0`
- `MODEL=meta-llama/Llama-2-7b-chat-hf`
- `LAYER=10`
- `N_EVAL=2048`
- `OUT_DIR=/path/to/output`
- `DRY_RUN=1` to print commands without executing them

## Smoke Checks

```bash
bash scripts/run_all_smoke_tests.sh
```

Section-level smoke scripts:

- `scripts/01_h1_sharedness/run_mock.sh`
- `scripts/02_h2_decode_ablation/run_mock.sh`
- `scripts/03_h2_patchback/run_mock.sh`
- `scripts/04_h3_prefill_decode/run_mock.sh`
- `scripts/05_steering_repair/run_mock.sh`

## Full Reruns

- `scripts/reproduce_h1_tables.sh`
- `scripts/reproduce_ablation_tables.sh`
- `scripts/reproduce_table_1_patchback.sh`
- `scripts/reproduce_table_2_steering.sh`
- `scripts/reproduce_table_3_h3.sh`

The lower-level wrappers in `scripts/full_runs/` can also be called directly.
