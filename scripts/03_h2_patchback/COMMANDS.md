# 03 H2 Patchback and Transfer

Paper outputs:

- Main: Table 1; Figures 5-6.
- Appendix: Tables 14-15, 20; Figures 16-17.

## Smoke Check

```bash
bash scripts/03_h2_patchback/run_mock.sh
```

## Full Run

```bash
bash scripts/reproduce_table_1_patchback.sh
```

Lower-level wrapper:

```bash
bash scripts/full_runs/run_patchback_table1.sh
```

Default outputs:

```text
outputs/03_patchback/table1/
```

Common overrides:

```bash
GPU_ID=0 \
MODEL=meta-llama/Llama-2-7b-chat-hf \
LAYER=10 \
TASK=aqua \
N_EVAL=254 \
MAX_FLIPS=128 \
bash scripts/full_runs/run_patchback_table1.sh
```

Use `COMPUTE_QS=0 QS_PATH=/path/to/Q_shared.npy` to reuse a precomputed basis.
Use `DRY_RUN=1` to print the command without executing it.
