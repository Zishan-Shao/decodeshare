# 13B H2 aligned4 table (`n=64`)

Source files:

- `rebuttal/after_review/exp_4_scale_extension/llama2_13b_h2_l10_fc_loto_n64_aligned4_rerun.md`
- `rebuttal/after_review/exp_4_scale_extension/llama2_13b_h2_l18_fc_loto_n64_aligned4.md`
- `rebuttal/after_review/exp_4_scale_extension/llama2_13b_h2_l28_fc_loto_n64_aligned4.md`

All rows use the aligned 4-task set:

- `commonsenseqa`
- `openbookqa`
- `qasc`
- `arc_challenge`

## Table

**Table. 13B task-level** `Δ(shared - baseline)` **under H2-style forced-choice evaluation on the aligned 4-task set;** `*` denotes `p < 0.05`. All layers use `n=64`.

| Layer | CSQA | OBQA | QASC | Arc | Sig |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10 | +1.6 | -7.8 | -4.7 | -6.2 | 0/4 |
| 18 | -10.9 | -1.6 | -1.6 | -4.7 | 0/4 |
| 28 | -9.4 | -12.5* | -14.1 | -6.2 | 1/4 |

## Short interpretation

- `L10` and `L18` are weak.
- `L28` is the clearest late-layer hit in the aligned 4-task setting.
- The late-layer effect is still weaker than the older non-aligned `L26, n=128` result, but this table is the clean one to use for cross-scale comparison against `70B`.
