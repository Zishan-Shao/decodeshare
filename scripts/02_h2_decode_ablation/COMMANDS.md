# 02 H2 Decode-Only Ablation and Energy Controls

Paper outputs:

- Main: Figure 7.
- Appendix: Figures 9-10; Tables 5, 26-28.

## Smoke Check

```bash
bash scripts/02_h2_decode_ablation/run_mock.sh
```

## Full Runs

The section-level wrapper runs the forced-choice LOTO experiment and the
alpha/k-match energy-control sweep:

```bash
bash scripts/reproduce_ablation_tables.sh
```

Run only one part:

```bash
RUN_ENERGY=0 bash scripts/reproduce_ablation_tables.sh
RUN_LOTO=0 bash scripts/reproduce_ablation_tables.sh
```

Lower-level wrappers:

```bash
bash scripts/full_runs/run_disturb_cot_loto8_fc_reason.sh
bash scripts/full_runs/run_alpha_kmatch_sweep.sh
```

Default outputs:

```text
outputs/02_decode_ablation/loto/
outputs/02_decode_ablation/energy_kmatch_alpha_sweep/
```

Common overrides:

```bash
GPU_ID=0 \
MODEL=meta-llama/Llama-2-7b-chat-hf \
LAYER=10 \
N_SUBSPACE=128 \
N_EVAL=2048 \
bash scripts/full_runs/run_disturb_cot_loto8_fc_reason.sh
```

Use `DRY_RUN=1` to print commands without executing them.
