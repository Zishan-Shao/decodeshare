# Residual Characterization After Factoring Out Probe-Derived Format Directions

## Main causal result

Layer-10 probe-split causal evaluation on 4 forced-choice tasks (`commonsenseqa, arc_challenge, openbookqa, logiqa`; `n=256` pooled):

| Condition | k | Pooled accuracy delta | p-value |
|---|---:|---:|---:|
| `shared_full` | 32 | `-15.6 pts` | `0.0010` |
| `fmt_only` | 3 | `+0.8 pts` | `0.748` |
| `resid_only` | 29 | `-16.8 pts` | `0.0005` |

Interpretation: the probe-identifiable format/readout component explains little of the full shared-subspace causal effect, while the residual component preserves essentially all of it.

Source: `results/rebuttal_mechanism/a5_probe_split_l10/exp_A5_probe_split_layer10_main_4tasks_n64.md`

## Focused unembedding / logit-lens on `Q_fmt` vs `Q_resid`

Scores below are RMS-normalized unembedding projection strengths, so `Q_fmt (k=3)` and `Q_resid (k=29)` are directly comparable.

| Basis | option_letter max | yes_no max | reasoning_marker max | digit max | newline max | top reasoning token |
|---|---:|---:|---:|---:|---:|---|
| `Q_fmt` | `0.0225` | `0.0233` | `0.0221` | `0.0343` | `0.0109` | `because` |
| `Q_resid` | `0.0161` | `0.0169` | `0.0177` | `0.0223` | `0.0279` | `therefore` |

Top-tag examples:

| Basis | tag | top tokens |
|---|---|---|
| `Q_fmt` | `option_letter` | `A, C, C, E, B` |
| `Q_fmt` | `reasoning_marker` | `because, thus, hence, because, Hence` |
| `Q_resid` | `yes_no` | `no, YES, yes, YES, No` |
| `Q_resid` | `reasoning_marker` | `therefore, because, Therefore, hence, Because` |
| `Q_resid` | `newline` | `\\n` |

Interpretation: the two components are not cleanly separable by vocabulary signature alone; both retain mixed readout/decision-like tokens. The causal split above is therefore the stronger result.

Sources:
- `results/rebuttal_mechanism/focus_vocab_a5_l10/exp_1e_saved_basis_focus_vocab_Q_fmt_layer10_main_rms.md`
- `results/rebuttal_mechanism/focus_vocab_a5_l10/exp_1e_saved_basis_focus_vocab_Q_resid_layer10_main_rms.md`

## Open-answer residual reasoning-style probes

Open-answer decode states from `gsm8k, strategyqa, aqua` (`n=17181` tokens pooled), with prompts that explicitly request reasoning and no extra appended answer prefix.

| Tag | n_pos | `Q_resid` AP | `Q_fmt` AP | `Q_rand_resid` AP |
|---|---:|---:|---:|---:|
| `reasoning_marker` | 120 | `0.564` | `0.041` | `0.726` |
| `step_marker` | 98 | `0.169` | `0.029` | `0.100` |
| `digit` | 1813 | `0.966` | `0.132` | `0.963` |
| `equation_symbol` | 544 | `0.673` | `0.055` | `0.590` |

Interpretation:
- `Q_resid` strongly supports linear readout of reasoning-style surface features (`reasoning_marker`, `step_marker`) and arithmetic-format features (`digit`, `equation_symbol`).
- `Q_fmt` is consistently much worse on all four tags.
- Same-width random residual partitions can also linearly decode several of these surface tags well, so this probe should be used as a characterization result, not as the main causal argument.

Source: `results/rebuttal_mechanism/reasoning_probe_l10/exp_1d_open_answer_reasoning_probe_layer10_noapfx_m128.md`

## Recommended rebuttal wording

> To further characterize the non-format residual, we decomposed the layer-10 shared subspace into a probe-derived format/readout component (`k=3`) and its orthogonal residual (`k=29`). On four forced-choice tasks (`n=256` pooled), ablating the full shared subspace reduced accuracy by `15.6` points (`p=0.001`), ablating the format/readout component alone had essentially no effect (`+0.8` points, `p=0.748`), while ablating the residual component preserved the full effect (`-16.8` points, `p=5e-4`). We also characterized the residual with open-answer probes on `gsm8k`, `strategyqa`, and `aqua`: reasoning-style token families were much more linearly decodable from the residual than from the format/readout component (e.g., `reasoning_marker` AP `0.564` vs `0.041`; `step_marker` AP `0.169` vs `0.029`). These results support a mixed interpretation: formatting/readout structure is present, but the dominant causal effect lies in a non-format residual that also carries reasoning-like and arithmetic/decomposition signals.
