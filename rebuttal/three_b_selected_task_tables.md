# `Llama-3.2-3B-Instruct`: selected-task summary for `H1/H2/H3`

Requested focus tasks:

- `commonsenseqa`
- `qasc`
- `arc_challenge`
- `openbookqa`

Important caveat for `H1`:

- `H1` is an existence test on a calibration task pool, not a per-task held-out metric.
- The existing `3B` `H1` runs use the 8-task pool
  `gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa`.
- So these `H1` runs include `commonsenseqa`, `openbookqa`, and `qasc`, but **do not include `arc_challenge`**.

Sources:

- `H1`: [llama32_3b_h1_l27_8tasks_lite.json](rebuttal/after_review/exp_4_scale_extension/llama32_3b_h1_l27_8tasks_lite.json), [llama32_3b_h1_l24_8tasks_lite.json](rebuttal/after_review/exp_4_scale_extension/llama32_3b_h1_l24_8tasks_lite.json), and the earlier `layer 4/10/20/24` existence runs in [src/results/exists](src/results/exists)
- `H2`: [llama32_3b_h2_l24_fc_loto_n64.md](rebuttal/after_review/exp_4_scale_extension/llama32_3b_h2_l24_fc_loto_n64.md), [llama32_3b_h2_l27_fc_loto_n64.md](rebuttal/after_review/exp_4_scale_extension/llama32_3b_h2_l27_fc_loto_n64.md), [llama32_3b_h2_l27_fc_loto_n128.md](rebuttal/after_review/exp_4_scale_extension/llama32_3b_h2_l27_fc_loto_n128.md)
- `H3`: [h3_grid_v3_meta-llama_Llama-3.2-3B-Instruct_layer27_k57_W0_seed0.json](reasoning/h3_grid_v3_meta-llama_Llama-3.2-3B-Instruct_layer27_k57_W0_seed0.json)

## Table 3B-H1. Layer-wise existence support

| Layer | Run setting | Calibration pool | `n_prompts` | `cross_dim` | `|S|` | `|S|/cross_dim` | `p_perm` | `p_scramble` | `H1` |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 4 | full | 8-task pool (`csqa/open/qasc`, no `arc`) | 128 | 2069 | 119 | 5.75% | 0.00050 | 0.0196 | PASS |
| 10 | full | 8-task pool (`csqa/open/qasc`, no `arc`) | 128 | 2092 | 124 | 5.93% | 0.00050 | 0.0196 | PASS |
| 20 | full | 8-task pool (`csqa/open/qasc`, no `arc`) | 128 | 2129 | 135 | 6.34% | 0.00050 | 0.0196 | PASS |
| 24 | full | 8-task pool (`csqa/open/qasc`, no `arc`) | 128 | 2188 | 137 | 6.26% | 0.00050 | 0.0196 | PASS |
| 27 | lite | 8-task pool (`csqa/open/qasc`, no `arc`) | 32 | 1792 | 129 | 7.20% | 0.0154 | 0.0476 | PASS |

Short analysis:

- `3B` shows stable `H1` support across all available layers.
- The shared subset stays compact but clearly nonzero, with ratios around `5.8%` to `7.2%`.
- The only real caveat is coverage: current `3B H1` artifacts do **not** include `arc_challenge` in the calibration pool.

## Table 3B-H2. Selected-task causal effect across available layers

| Task | `L24, n=64` `Δ(shared-baseline)` | `p` | `L27, n=64` `Δ(shared-baseline)` | `p` | `L27, n=128` `Δ(shared-baseline)` | `p` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `commonsenseqa` | `+4.7 pts` | 0.348 | `-12.5 pts` | 0.093 | `-25.0 pts` | 0.0005 |
| `qasc` | `-3.1 pts` | 0.703 | `-26.6 pts` | 0.0005 | `-42.2 pts` | 0.0005 |
| `arc_challenge` | `+3.1 pts` | 0.627 | `-14.1 pts` | 0.067 | `-9.4 pts` | 0.0635 |
| `openbookqa` | `-1.6 pts` | 1.000 | `-14.1 pts` | 0.062 | `-10.9 pts` | 0.0315 |
| **Mean over 4 tasks** | `+0.8 pts` |  | `-16.8 pts` |  | `-21.9 pts` |  |

Short analysis:

- `layer 24` is weak: the mean effect is near zero and two tasks are slightly positive.
- `layer 27` is the first clearly causal layer for this `3B` model.
- Increasing `n_eval` from `64` to `128` makes the story much cleaner, especially on `commonsenseqa`, `qasc`, and `openbookqa`.

## Table 3B-H3. Prefill-decode mismatch on selected tasks (`layer=27`)

Run-level note:

- `k_match=57`
- mean principal angle between decode- and prefill-estimated bases: `72.09°`

| Task | Baseline | `decode-est@decode` | `prefill-est@decode` | `rand@decode` | `Δ(decode-est@decode)` | `Δ(prefill-est@decode)` | `Δ(rand@decode)` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `commonsenseqa` | `69.5` | `36.1` | `65.8` | `69.1` | `-33.4 pts` | `-3.7 pts` | `-0.4 pts` |
| `qasc` | `70.1` | `43.8` | `66.0` | `70.9` | `-26.4 pts` | `-4.1 pts` | `+0.8 pts` |
| `arc_challenge` | `68.9` | `52.5` | `60.7` | `67.8` | `-16.4 pts` | `-8.2 pts` | `-1.2 pts` |
| `openbookqa` | `68.4` | `52.8` | `64.6` | `69.0` | `-15.6 pts` | `-3.8 pts` | `+0.6 pts` |
| **Mean over 4 tasks** | `69.2` | `46.3` | `64.3` | `69.2` | `-22.9 pts` | `-5.0 pts` | `0.0 pts` |

Short analysis:

- `H3` is clean on this four-task slice at `layer 27`.
- Decode-estimated interventions hurt all four tasks substantially.
- Prefill-estimated interventions are much weaker, and the random decode control stays near zero.
- So for this `3B` model, the strongest mismatch signal is already visible at `layer 27`.
