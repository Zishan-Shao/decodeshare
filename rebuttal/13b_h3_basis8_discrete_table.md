## 13B H3 at Layer 28 on the discrete 8-task pool

Source:

- `reasoning/h3_grid_v3_meta-llama_Llama-2-13b-chat-hf_layer28_k51_W0_seed0.json`

Task pool:

- `commonsenseqa, strategyqa, piqa, arc_challenge, openbookqa, qasc, logiqa, boolq`

This is the same discrete 8-task pool later reused for the stronger `13B` `H2` reruns.

| Task | Baseline | Decode-est@Decode | Prefill-est@Decode | Rand@Decode | `Δ(decode-baseline)` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `commonsenseqa` | 59.2 | 54.3 | 60.9 | 59.8 | -4.9 |
| `strategyqa` | 49.0 | 55.1 | 49.2 | 48.8 | +6.1 |
| `piqa` | 67.4 | 69.1 | 71.3 | 69.7 | +1.8 |
| `arc_challenge` | 60.0 | 61.3 | 62.3 | 60.9 | +1.4 |
| `openbookqa` | 61.8 | 56.2 | 60.4 | 62.4 | -5.6 |
| `qasc` | 58.6 | 48.0 | 57.2 | 58.0 | -10.5 |
| `logiqa` | 32.8 | 29.7 | 33.6 | 33.0 | -3.1 |
| `boolq` | 0.0 | 0.0 | 0.0 | 0.0 | +0.0 |

Short read:

- The clearest negative effects are on `qasc`, `openbookqa`, and `commonsenseqa`.
- `arc_challenge` is slightly positive in this older `H3` run, so it is the main mismatch with the newer `H2` table.
- `strategyqa` and `piqa` are also positive, so this 8-task `H3` should be treated as a mixed, supportive result rather than a clean all-task confirmation.
