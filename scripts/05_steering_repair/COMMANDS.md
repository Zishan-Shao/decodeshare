# 05 Steering Repair and Template Robustness

Paper outputs:

- Main: Table 2.
- Appendix: Figure 15; Tables 21-25, 29.

## Smoke Check

```bash
bash scripts/05_steering_repair/run_mock.sh
```

## Full Run

```bash
bash scripts/reproduce_table_2_steering.sh
```

Lower-level wrapper:

```bash
bash scripts/full_runs/run_steering_repair.sh
```

Default outputs:

```text
outputs/05_steering_repair/multibench/
```

Common overrides:

```bash
GPU_ID=0 \
MODEL=meta-llama/Llama-2-7b-chat-hf \
LAYER=10 \
TASKS=boolq,rte,sst2 \
CALIB_PER_CLASS=256 \
EVAL_PER_CLASS=128 \
bash scripts/full_runs/run_steering_repair.sh
```

Use `DRY_RUN=1` to print the command without executing it.
