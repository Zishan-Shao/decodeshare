# 05 Steering Flip Tables

Paper outputs:

- Main README: steering ranking-alignment table.
- Main README: held-out REAL deployment-selection table.
- Legacy steering-repair multibench scripts remain under
  `scripts/full_runs/run_steering_repair.sh`, but they are not the source of
  the README flip tables.

## Smoke Check

```bash
bash scripts/05_steering_repair/run_mock.sh
```

## Full Run

```bash
bash scripts/reproduce_steering_flip_tables.sh
```

This wrapper runs the two public pipelines behind the README steering tables:

```text
downstream/rebuttal/orthogonal_steer/exp_A1_cross_method_rankflip.py
downstream/rebuttal/exp_ranking_flip_steering_layer28_rand100_full.py
downstream/rebuttal/exp_ranking_flip_trad_family.py
```

Default outputs:

```text
outputs/05_steering_flip_tables/ranking_alignment/
outputs/05_steering_flip_tables/deployment_selection/
```

Common overrides:

```bash
GPU_ID=0 \
MODEL=meta-llama/Llama-2-7b-chat-hf \
LAYER=28 \
TASKS=commonsenseqa,arc_challenge,openbookqa,qasc,logiqa \
N_EVAL=128 \
N_VEC_CAA=32 \
N_VEC_INSTR=64 \
N_VEC_SAE=64 \
N_DIAG=100 \
bash scripts/reproduce_steering_flip_tables.sh
```

Use `DRY_RUN=1` to print the exact commands without executing model inference.
