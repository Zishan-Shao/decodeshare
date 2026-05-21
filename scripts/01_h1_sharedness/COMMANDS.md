# 01 H1 Shared Decode-Time Structure

Paper outputs:

- Main: Figures 2-4; Table 6.
- Appendix: Figures 8, 11-14; Tables 7-13.

## Smoke Check

```bash
bash scripts/01_h1_sharedness/run_mock.sh
```

## Full Run

The public entry point runs `experiments/01_sharedness/run_full_benchmark.py`
and then regenerates summary tables from the produced JSON/TXT records.

```bash
bash scripts/reproduce_h1_tables.sh
```

Common overrides:

```bash
GPU_ID=0 \
MODEL=Qwen/Qwen2.5-7B-Instruct \
LAYER=10 \
N_PROMPTS=128 \
CALIB_MAX_NEW_TOKENS=128 \
N_EVAL=2048 \
bash scripts/reproduce_h1_tables.sh
```

Lower-level wrapper:

```bash
bash scripts/full_runs/run_h1_full_benchmark.sh
```

Outputs default to:

```text
outputs/01_sharedness/full_benchmark/
```

Use `DRY_RUN=1` to print the command without executing it.
