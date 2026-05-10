# W1 Scale Tables

This note fills the W1 scale-extension table using only results that already exist in the current workspace.

Important honesty constraint:

- There is no packaged `1B` result in the current workspace.
- I therefore do **not** recommend claiming `1B–70B` or "three orders of magnitude".
- The safe scale claim is:
  - `3B -> 70B` causal support
  - plus additional `8B` / `3B` existence support
  - therefore the phenomenon is **not confined to 7B**

## Recommended replacement text

> We agree that testing beyond the original 7B setting is important. Using existing and newly packaged scale-extension runs, we now find support for DecodeShare across a broader scale range. On the larger side, exact `Llama-2-13B-chat` shows multi-layer `H1` support together with late-layer `H2-lite` causal confirmation, and `Llama-2-70B-chat` shows a particularly strong late-layer causal effect at `layer=56`. On the smaller side, `Llama-3.2-3B-Instruct` shows both significant `H1` structure and a strong `H2-lite` effect, while additional `3B/8B` runs further confirm that the shared decode-time structure is not specific to 7B.

## Table W1-A. Existing causal scale evidence (`H1` + `H2-lite`)

| Model | Layer | `|S|` | Ratio | `H1` (`p_perm`) | `H2` mean `Δ(shared-baseline)` |
| :--- | :---: | ---: | ---: | ---: | ---: |
| `meta-llama/Llama-3.2-3B-Instruct` | 27 | 129 | 7.2% | 0.0154 | `-21.9 pts` (`n=128`, `3/4` significant) |
| `meta-llama/Llama-2-13b-chat-hf` | 26 | 116 | 3.9% | 0.0154 | `-15.6 pts` (`n=128`, `3/4` significant) |
| `meta-llama/Llama-2-70b-chat-hf` | 56 | 13 | 0.32% | 0.0154 | `-20.3 pts` (`n=64`, `4/4` significant) |

Notes:

- `3B` row uses [llama32_3b_h1_l27_8tasks_lite.json](rebuttal/after_review/exp_4_scale_extension/llama32_3b_h1_l27_8tasks_lite.json) and [llama32_3b_h2_l27_fc_loto_n128.md](rebuttal/after_review/exp_4_scale_extension/llama32_3b_h2_l27_fc_loto_n128.md).
- `13B` row uses [exp_4_scale_extension_summary.md](rebuttal/after_review/exp_4_scale_extension/exp_4_scale_extension_summary.md).
- `70B` row uses [llama2_70b_h1_l56_8tasks_lite.json](rebuttal/after_review/exp_8_scale_llama/llama2_70b_h1_l56_8tasks_lite.json) and the four held-out `L56` files under [exp_8_scale_llama](rebuttal/after_review/exp_8_scale_llama).

## Table W1-B. Additional non-7B support already in the workspace

| Model | Layer | `|S|` | Ratio | `H1` (`p_perm`) | Additional note |
| :--- | :---: | ---: | ---: | ---: | :--- |
| `Qwen/Qwen2.5-3B-Instruct` | 10 | 112 | 7.8% | 0.0154 | `H1` pass on a distinct 3B family |
| `meta-llama/Llama-3.1-8B-Instruct` | 10 | 103 | 4.1% | 0.0154 | fresh `8B` confirmatory `H1` pass |
| `meta-llama/Llama-3.1-8B-Instruct` | 4 / 10 / 24 | - | - | - | `H3` decode-vs-prefill separation present at all three tested layers |
| `meta-llama/Llama-2-13b-chat-hf` | 10 / 28 | 77 / 128 | 2.8% / 4.1% | 0.0154 | multi-layer `H1` support, not a single-layer artifact |

## If you want a minimal table for the rebuttal body

If space is tight, I would use only Table W1-A in the main rebuttal and keep Table W1-B in backup notes.

## Suggested text under the table

> These results support a cautious but clear multi-scale claim: the shared decode-time subspace is not confined to the original 7B setting. We now observe direct causal support at `3B`, `13B`, and `70B`, with especially strong late-layer effects at `3B layer=27`, `13B layer=26`, and `70B layer=56`. Additional `3B` and `8B` existence runs further indicate that the shared structure persists beyond a single architecture or parameter regime.

## Things I would not claim

- Do **not** say `1B–70B`.
- Do **not** say "three orders of magnitude".
- Do **not** say "scales robustly across the full parameter spectrum".
- Do **not** imply that the current workspace already contains a full `H1/H2/H3` sweep for every scale.
