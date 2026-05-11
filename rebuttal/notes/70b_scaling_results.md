# 70B Scaling Spot-Check

Model: `meta-llama/Llama-2-70b-chat-hf`

Infrastructure note:
- Multi-GPU sharded loading is now supported via `device_map=auto` plus CPU offload.
- All 70B runs below used 4 A100-80GB GPUs with offload.

## Main result to cite

We ran a 70B forced-choice spot-check by first constructing a decode-shared basis at layer 25 on six tasks, then reusing that basis for a larger `commonsenseqa` evaluation (`n=64`).

| Task | n | Baseline | Shared | Delta Shared | Ctrl(E) | Delta Ctrl(E) | Rand(E) | Delta Rand(E) |
|---|---|---|---|---|---|---|---|---|
| commonsenseqa | 64 | 68.8 [57.8, 79.7] | 53.1 [42.2, 65.6] | -15.6 [-25.0, -4.7] (p=0.012) | 64.1 [52.3, 75.8] | -4.7 [-10.9, +0.0] (p=0.232) | 65.6 [54.7, 76.6] | -3.1 [-7.8, +0.0] (p=0.487) |

Interpretation:
- Shared-subspace removal remains harmful at 70B.
- Energy-matched and random energy-matched controls are much smaller and not significant.
- This is consistent with the 7B finding that the decode-shared subspace scales beyond 7B.

## Basis-construction smoke run

The initial 70B spot-check used six subspace tasks:
`commonsenseqa, arc_challenge, openbookqa, qasc, logiqa, boolq`

Layer-25 smoke results on `n=8` eval examples per task:

| Task | n | Baseline | Shared | Delta Shared | Ctrl(E) | Delta Ctrl(E) | Rand(E) | Delta Rand(E) |
|---|---|---|---|---|---|---|---|---|
| commonsenseqa | 8 | 50.0 [12.5, 87.5] | 37.5 [12.5, 75.0] | -12.5 [-50.0, +31.6] (p=1) | 50.0 [12.5, 87.5] | +0.0 [+0.0, +0.0] (p=1) | 50.0 [12.5, 87.5] | +0.0 [+0.0, +0.0] (p=1) |
| arc_challenge | 8 | 87.5 [75.0, 100.0] | 87.5 [75.0, 100.0] | +0.0 [+0.0, +0.0] (p=1) | 87.5 [75.0, 100.0] | +0.0 [+0.0, +0.0] (p=1) | 87.5 [75.0, 100.0] | +0.0 [+0.0, +0.0] (p=1) |

This smoke run was mainly to validate the 70B pipeline end-to-end and save the reusable basis used in the larger CSQA evaluation above.

## Files

- Main 70B spot-check basis construction:
  `results/rebuttal_scaling/llama2_70b_a3_smoke_v3/exp_A3_causal_layer25_llama2_70b_l25_smoke_v3.md`
- Main reusable basis:
  `results/rebuttal_scaling/llama2_70b_a3_smoke_v3/exp_A3_bases_layer25_llama2_70b_l25_smoke_v3.npz`
- Larger 70B evaluation on saved basis:
  `results/rebuttal_scaling/llama2_70b_eval_saved_basis_v1/exp_A3_eval_saved_basis_layer25_csqa64.md`
