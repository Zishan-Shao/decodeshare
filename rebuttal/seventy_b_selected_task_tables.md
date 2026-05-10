# `Llama-2-70b-chat-hf`: selected-task summary for `H1/H2/H3`

Requested focus tasks:

- `commonsenseqa`
- `qasc`
- `arc_challenge`
- `openbookqa`

Important caveat for `H1`:

- `H1` is an existence test on a calibration task pool, not a per-task held-out metric.
- The existing `70B` `H1` runs use the 8-task pool
  `gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa`.
- So these `H1` runs include `commonsenseqa`, `openbookqa`, and `qasc`, but **do not include `arc_challenge`**.

Sources:

- `H1`: [llama2_70b_h1_l32_8tasks_lite.json](rebuttal/after_review/exp_8_scale_llama/llama2_70b_h1_l32_8tasks_lite.json), [llama2_70b_h1_l48_8tasks_lite.json](rebuttal/after_review/exp_8_scale_llama/llama2_70b_h1_l48_8tasks_lite.json), [llama2_70b_h1_l56_8tasks_lite.json](rebuttal/after_review/exp_8_scale_llama/llama2_70b_h1_l56_8tasks_lite.json)
- `H2`: the per-task `layer 25/32/40/48/56` summaries in [rebuttal/after_review/exp_8_scale_llama](rebuttal/after_review/exp_8_scale_llama)
- `H3`: [h3_grid_v3_meta-llama_Llama-2-70b-chat-hf_layer56_k49_W0_seed0.json](reasoning/h3_grid_v3_meta-llama_Llama-2-70b-chat-hf_layer56_k49_W0_seed0.json)

## Table 70B-H1. Layer-wise existence support

| Layer | Calibration pool | `n_prompts` | `cross_dim` | `|S|` | `|S|/cross_dim` | `p_perm` | `p_scramble` | `H1` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 32 | 8-task pool (`csqa/open/qasc`, no `arc`) | 32 | 3522 | 21 | 0.60% | 0.0154 | 0.0476 | PASS |
| 48 | 8-task pool (`csqa/open/qasc`, no `arc`) | 32 | 4018 | 2 | 0.05% | 0.0154 | 0.0476 | PASS |
| 56 | 8-task pool (`csqa/open/qasc`, no `arc`) | 32 | 4096 | 13 | 0.32% | 0.0154 | 0.0476 | PASS |

Short analysis:

- `70B` passes `H1` at all three tested late layers.
- The shared subset is much more compact than in `3B/13B`, especially at `48/56`.
- So the `70B` story is not “large shared fraction,” but rather “very small yet still significant shared subset.”

## Table 70B-H2. Selected-task causal effect across available late layers

| Layer | `commonsenseqa` | `openbookqa` | `qasc` | `arc_challenge` | Mean `Δ` | Significant tasks |
| --- | --- | --- | --- | --- | ---: | ---: |
| 25 | `-14.1` (`p=0.039`) | `-1.6` (`p=1`) | `-10.9` (`p=0.048`) | `-7.8` (`p=0.161`) | `-8.6 pts` | `2/4` |
| 32 | `-26.6` (`p=0.002`) | `-14.1` (`p=0.046`) | `-7.8` (`p=0.180`) | `-12.5` (`p=0.00999`) | `-15.2 pts` | `3/4` |
| 40 | `-25.0` (`p=0.000999`) | `-10.9` (`p=0.108`) | `-15.6` (`p=0.00899`) | `-21.9` (`p=0.005`) | `-18.4 pts` | `3/4` |
| 48 | `-28.1` (`p=0.000999`) | `-21.9` (`p=0.004`) | `-7.8` (`p=0.114`) | `-23.4` (`p=0.002`) | `-20.3 pts` | `3/4` |
| 56 | `-21.9` (`p=0.000999`) | `-15.6` (`p=0.038`) | `-17.2` (`p=0.000999`) | `-26.6` (`p=0.000999`) | `-20.3 pts` | `4/4` |

Short analysis:

- The `70B` `H2` story is strong and gets cleaner as depth increases.
- `layer 25` already shows signal, but `40/48/56` are the real hits.
- `layer 56` is the cleanest single layer: all four tasks drop, and all four are significant.

## Table 70B-H3. Prefill-decode mismatch on selected tasks (`layer=56`)

Run-level note:

- `k_match=49`
- mean principal angle between decode- and prefill-estimated bases: `81.92°`

| Task | Baseline | `decode-est@decode` | `prefill-est@decode` | `rand@decode` | `Δ(decode-est@decode)` | `Δ(prefill-est@decode)` | `Δ(rand@decode)` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `commonsenseqa` | `84.4` | `43.8` | `79.7` | `82.8` | `-40.6 pts` | `-4.7 pts` | `-1.6 pts` |
| `qasc` | `60.9` | `29.7` | `59.4` | `60.9` | `-31.2 pts` | `-1.6 pts` | `0.0 pts` |
| `arc_challenge` | `84.4` | `54.7` | `85.9` | `84.4` | `-29.7 pts` | `+1.6 pts` | `0.0 pts` |
| `openbookqa` | `78.1` | `40.6` | `78.1` | `76.6` | `-37.5 pts` | `0.0 pts` | `-1.6 pts` |
| **Mean over 4 tasks** | `77.0` | `42.2` | `75.8` | `76.2` | `-34.8 pts` | `-1.2 pts` | `-0.8 pts` |

Short analysis:

- `H3` is extremely clean for `70B` at `layer 56`.
- Decode-estimated interventions sharply hurt all four tasks.
- Prefill-estimated interventions are near zero overall, and the random control is also near zero.
- This is the strongest prefilling-vs-decoding mismatch result among the current `3B/13B/70B` scale runs.
