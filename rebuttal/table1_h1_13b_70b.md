# Table 1: H1 existence at larger scales

Source files:

- `rebuttal/after_review/exp_4_scale_extension/llama2_13b_h1_l10_8tasks_lite.txt`
- `rebuttal/after_review/exp_4_scale_extension/llama2_13b_h1_l18_8tasks_lite.txt`
- `rebuttal/after_review/exp_4_scale_extension/llama2_13b_h1_l28_8tasks_lite.txt`
- `rebuttal/after_review/exp_8_scale_llama/llama2_70b_h1_l32_8tasks_lite.txt`
- `rebuttal/after_review/exp_8_scale_llama/llama2_70b_h1_l48_8tasks_lite.txt`
- `rebuttal/after_review/exp_8_scale_llama/llama2_70b_h1_l56_8tasks_lite.txt`

Shared-task pool in all runs:

- `gsm8k`
- `commonsenseqa`
- `strategyqa`
- `aqua`
- `openbookqa`
- `qasc`
- `boolq`
- `piqa`

All runs use:

- `tau=0.001`
- `m_shared=all`
- `n_prompts=32`
- `calib_max_new_tokens=64`
- `per_task_max_states=8000`

## Rebuttal-ready prose

First, on both `Llama-2-13B-Chat` and `Llama-2-70B-Chat`, we confirmed `H1` at multiple layers under the same strict setting (`tau=10^{-3}`, `8` shared tasks). For `13B`, the recovered shared decode-time subspace remains compact but clearly nontrivial at layers `10/18/28`, with shared ratios `3.6% / 4.3% / 4.8%`. For `70B`, the shared subset is even more concentrated at layers `32/48/56`, with shared ratios `0.60% / 0.05% / 0.32%`. All six runs pass both null tests (`p_perm=0.0154`, `p_scramble=0.0476`). This shows that the shared decode-time structure is not a `7B`-only artifact.

## Table

**Table 1.** Multi-layer `H1` confirmation at larger scales under the same strict 8-task setting.

| Model | Layer | `cross_dim` | `|S|` | `|S| / cross_dim` | `p_perm` | `p_scramble` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Llama-2-13B-Chat` | 10 | 2427 | 88 | 3.63% | 0.0154 | 0.0476 |
| `Llama-2-13B-Chat` | 18 | 2539 | 108 | 4.25% | 0.0154 | 0.0476 |
| `Llama-2-13B-Chat` | 28 | 2746 | 132 | 4.81% | 0.0154 | 0.0476 |
| `Llama-2-70B-Chat` | 32 | 3522 | 21 | 0.60% | 0.0154 | 0.0476 |
| `Llama-2-70B-Chat` | 48 | 4018 | 2 | 0.05% | 0.0154 | 0.0476 |
| `Llama-2-70B-Chat` | 56 | 4096 | 13 | 0.32% | 0.0154 | 0.0476 |

## Short interpretation

- `13B` shows a compact but stable shared subset across all tested layers, with the shared ratio increasing toward later layers.
- `70B` also passes `H1` at all tested layers, but with a much tighter shared subset.
- The clean takeaway is not that the shared ratio is constant across scales, but that a statistically significant shared decode-time subset persists beyond the original `7B` setting.
