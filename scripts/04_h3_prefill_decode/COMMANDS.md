# 04 H3 Prefill-Decode Mismatch

Paper outputs:

- Main: Table 3.
- Appendix: Tables 16-19; Figure 14.

## Smoke Check

```bash
bash scripts/04_h3_prefill_decode/run_mock.sh
```

## Full Run

```bash
bash scripts/reproduce_table_3_h3.sh
```

Lower-level wrappers:

```bash
bash scripts/full_runs/run_h3_grid.sh
bash scripts/full_runs/run_prefill_decode_nextsteps.sh
```

Default outputs:

```text
outputs/04_prefill_decode/h3_grid/
outputs/04_prefill_decode/prefill_decode_sweeps/
```

Common overrides:

```bash
GPU_ID=0 \
MODEL=meta-llama/Llama-2-7b-chat-hf \
LAYER=10 \
N_SUBSPACE=128 \
N_EVAL=2048 \
bash scripts/full_runs/run_h3_grid.sh
```

Use `DRY_RUN=1` to print commands without executing them.
