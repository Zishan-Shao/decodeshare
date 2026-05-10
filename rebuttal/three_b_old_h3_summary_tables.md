# 3B summary tables using the older 8-task H3 basis

Important scope note:

- The strong **old** `3B H3` result comes from the completed 8-task run:
  `reasoning/h3_grid_v3_meta-llama_Llama-3.2-3B-Instruct_layer27_k57_W0_seed0.json`
- This lets us report a clean 4-task slice (`CSQA/OBQA/QASC/Arc`) from a broader 8-task basis.
- However, the current `3B H2` result is still the existing 4-task run:
  `rebuttal/after_review/exp_4_scale_extension/llama32_3b_h2_l27_fc_loto_n128.md`
- So the safest phrasing is:
  `3B H3 uses the broader 8-task basis; H2 is reported on the 4-task MC slice.`

## Table 1. Small-scale evidence (3B): existence and causal effect

| Model | Layer | `|S|` | Ratio | `H1 (p_perm)` | `H2` mean `Δ(shared-baseline)` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Llama-3.2-3B-Instruct` | 27 | 129 | 7.2% | 0.0154 | `-21.9 pts` (`3/4` significant) |

Source:

- `H1`: `rebuttal/after_review/exp_4_scale_extension/llama32_3b_h1_l27_8tasks_lite.txt`
- `H2`: `rebuttal/after_review/exp_4_scale_extension/llama32_3b_h2_l27_fc_loto_n128.md`

## Table 2. Prefill-decode mismatch at 3B (`H3`), reported on the 4-task slice from the older 8-task basis run

Old-basis `H3` source:

- `reasoning/h3_grid_v3_meta-llama_Llama-3.2-3B-Instruct_layer27_k57_W0_seed0.json`

Basis task pool in that run:

- `commonsenseqa`
- `strategyqa`
- `piqa`
- `arc_challenge`
- `openbookqa`
- `qasc`
- `logiqa`
- `boolq`

| Task | Baseline | `decode-est@decode` | `prefill-est@decode` | `rand@decode` | `Δ(decode-baseline)` | `Δ(prefill-baseline)` | `Δ(rand-baseline)` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `CSQA` | 69.5 | 36.1 | 65.8 | 69.1 | `-33.4` | `-3.7` | `-0.4` |
| `OBQA` | 68.4 | 52.8 | 64.6 | 69.0 | `-15.6` | `-3.8` | `+0.6` |
| `QASC` | 70.1 | 43.8 | 66.0 | 70.9 | `-26.4` | `-4.1` | `+0.8` |
| `Arc` | 68.9 | 52.5 | 60.7 | 67.8 | `-16.4` | `-8.2` | `-1.2` |
| **Mean (care-4)** | 69.2 | 46.3 | 64.3 | 69.2 | `-22.9` | `-5.0` | `-0.0` |

## Table 3. Approximate paired significance for the same 3B old-basis `H3` slice

These `p` values were recomputed from the saved per-example correctness arrays using the same sign-flip paired-test family as the `H2` summaries.

| Task | `decode-est@decode` vs baseline | `prefill-est@decode` vs baseline | `decode-est@decode` vs `prefill-est@decode` |
| --- | ---: | ---: | ---: |
| `CSQA` | `p=0.0002` | `p=0.0110` | `p=0.0002` |
| `OBQA` | `p=0.0002` | `p=0.0274` | `p=0.0002` |
| `QASC` | `p=0.0002` | `p=0.0446` | `p=0.0002` |
| `Arc` | `p=0.0002` | `p=0.0002` | `p=0.0082` |

## Short interpretation

- The older 8-task-basis `3B H3` result is much cleaner than the newer aligned-4 rerun.
- On the four reported MC tasks, decode-estimated interventions consistently degrade performance, with a mean drop of `-22.9 pts`.
- Prefill-estimated interventions are not zero, but they are materially weaker than decode-estimated interventions on all four tasks.
- This supports a robust prefill-decode mismatch story at `3B`, especially when the shared subspace is estimated from the broader 8-task pool.
