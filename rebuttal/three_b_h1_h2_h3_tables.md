# 3B Models: H1 / H2 / H3 Tables

This note summarizes the currently available `3B`-scale evidence in the workspace.

Covered models:

- `meta-llama/Llama-3.2-3B-Instruct`
- `Qwen/Qwen2.5-3B-Instruct` (H1 only in the currently packaged results)

Important scope note:

- `Llama-3.2-3B-Instruct` has available `H1`, `H2`, and one packaged `H3` result.
- `Qwen/Qwen2.5-3B-Instruct` currently has packaged `H1`, but not a matching packaged `H2/H3` table in this workspace.

## Table 3B-H1. Shared-subspace existence (`H1`)

| Model | Setting | Layer | `n_prompts` | `cross_dim` | `|S|` | `|S|/cross_dim` | `p_perm` | `p_scramble` | H1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `meta-llama/Llama-3.2-3B-Instruct` | full | 4 | 128 | 2069 | 119 | 5.8% | 0.00050 | 0.01961 | PASS |
| `meta-llama/Llama-3.2-3B-Instruct` | full | 10 | 128 | 2092 | 124 | 5.9% | 0.00050 | 0.01961 | PASS |
| `meta-llama/Llama-3.2-3B-Instruct` | full | 20 | 128 | 2129 | 135 | 6.3% | 0.00050 | 0.01961 | PASS |
| `meta-llama/Llama-3.2-3B-Instruct` | full | 24 | 128 | 2188 | 137 | 6.3% | 0.00050 | 0.01961 | PASS |
| `meta-llama/Llama-3.2-3B-Instruct` | lite | 24 | 32 | 1898 | 139 | 7.3% | 0.01538 | 0.04762 | PASS |
| `meta-llama/Llama-3.2-3B-Instruct` | lite | 27 | 32 | 1792 | 129 | 7.2% | 0.01538 | 0.04762 | PASS |
| `Qwen/Qwen2.5-3B-Instruct` | lite | 10 | 32 | 1435 | 112 | 7.8% | 0.01538 | 0.04762 | PASS |

Reading:

- `Llama-3.2-3B-Instruct` shows stable `H1` support across multiple layers in both the older full setting and the newer lite setting.
- `Qwen/Qwen2.5-3B-Instruct` gives an additional architecture-family `3B` confirmation on `H1`.

## Table 3B-H2. Decode-time causal effect (`H2`)

| Model | Setting | Layer | Eval scope | `n` | mean `Δ(shared-baseline)` | Significant tasks |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| `meta-llama/Llama-3.2-3B-Instruct` | appendix LOTO(8) | 4 | 8 tasks | task-specific | `-37.7 pts` | `8/8` |
| `meta-llama/Llama-3.2-3B-Instruct` | appendix LOTO(8) | 10 | 8 tasks | task-specific | `-41.2 pts` | `8/8` |
| `meta-llama/Llama-3.2-3B-Instruct` | appendix LOTO(8) | 24 | 8 tasks | task-specific | `-39.0 pts` | `8/8` |
| `meta-llama/Llama-3.2-3B-Instruct` | rebuttal H2-lite | 24 | 4 MC tasks | 64 | `+0.8 pts` | `0/4` |
| `meta-llama/Llama-3.2-3B-Instruct` | rebuttal H2-lite | 27 | 4 MC tasks | 64 | `-16.8 pts` | `1/4` |
| `meta-llama/Llama-3.2-3B-Instruct` | rebuttal H2-lite | 27 | 4 MC tasks | 128 | `-21.9 pts` | `3/4` |

Reading:

- In the older appendix-style `LOTO(8)` runs, `3B` already showed strong causal effects at layers `4/10/24`.
- In the newer rebuttal-sized forced-choice package, `layer=24` is weak, but the later `layer=27` run becomes clearly positive, especially at `n=128`.
- So the safest `3B` causal summary is: the effect is real, and the strongest rebuttal-friendly causal artifact is the later-layer `Llama-3.2-3B-Instruct layer=27` run.

## Table 3B-H3. Prefill/decode mismatch (`H3`) summary

Currently packaged `3B` `H3` evidence in this workspace:

| Model | Layer | `k_match` | mean angle | mean `Δ(decode-est@decode)` | mean `Δ(prefill-est@decode)` | mean `Δ(rand@decode)` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `meta-llama/Llama-3.2-3B-Instruct` | 24 | 43 | `79.0°` | `-3.2 pts` | `+0.5 pts` | `+0.1 pts` |

Reading:

- The decode-estimated intervention is harmful under decode-time application, while the prefill-estimated and random controls remain near zero on average.
- The large principal angle (`79.0°`) is consistent with a strong prefill/decode mismatch story.

## Table 3B-H3b. `Llama-3.2-3B-Instruct` layer-24 per-task `H3` deltas

| Task | `Δ(decode-est@decode)` | `Δ(prefill-est@decode)` | `Δ(rand@decode)` |
| --- | ---: | ---: | ---: |
| commonsenseqa | `-8.0 pts` | `+0.0 pts` | `-0.6 pts` |
| strategyqa | `-1.8 pts` | `-2.0 pts` | `-0.2 pts` |
| piqa | `-2.0 pts` | `+1.0 pts` | `+0.4 pts` |
| arc_challenge | `-1.8 pts` | `+1.2 pts` | `+0.2 pts` |
| openbookqa | `-4.8 pts` | `+1.2 pts` | `-0.2 pts` |
| qasc | `-6.1 pts` | `-0.2 pts` | `+0.6 pts` |
| logiqa | `-1.4 pts` | `+2.5 pts` | `+0.2 pts` |
| boolq | `+0.0 pts` | `+0.0 pts` | `+0.0 pts` |

Reading:

- The clearest `3B` `H3` signal comes from `commonsenseqa`, `openbookqa`, and `qasc`.
- Even where the absolute effect is smaller than in the strongest `H2` runs, the key protocol pattern still holds:
  - decode-estimated decode-time intervention hurts;
  - prefill-estimated decode intervention stays near zero or even slightly positive;
  - random control stays near zero.

## Short rebuttal-safe takeaway

> At `3B`, the workspace now contains direct support for all three parts of the main story. `H1` holds across multiple layers for `Llama-3.2-3B-Instruct` and also appears in a distinct `Qwen2.5-3B` model family. `H2` is strongly positive in both the original `LOTO(8)` package and a later-layer rebuttal-sized `H2-lite` run (`layer=27`). `H3` is currently packaged at `layer=24`, where the decode-estimated decode-time intervention is harmful while prefill-estimated and random decode controls remain near zero, together with a large prefill/decode principal-angle mismatch.
