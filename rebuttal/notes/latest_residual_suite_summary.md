# Latest Residual-Suite Summary

This note summarizes the latest experiments aimed at characterizing the probe-derived
format/readout component `Q_fmt` and the orthogonal residual `Q_resid` inside the
layer-10 shared decode subspace.

Primary artifacts:

- `results/rebuttal_mechanism/a5_probe_split_l10/exp_A5_probe_split_layer10_main_4tasks_n64.md`
- `results/rebuttal_mechanism/focus_vocab_a5_l10/exp_1e_saved_basis_focus_vocab_Q_fmt_layer10_main_rms.md`
- `results/rebuttal_mechanism/focus_vocab_a5_l10/exp_1e_saved_basis_focus_vocab_Q_resid_layer10_main_rms.md`
- `results/rebuttal_mechanism/reasoning_probe_l10/exp_1d_open_answer_reasoning_probe_layer10_noapfx_m128.md`

## Executive summary

1. The strongest result remains causal: after splitting the layer-10 shared subspace into a probe-derived format/readout part (`k=3`) and an orthogonal residual (`k=29`), ablating the format/readout part alone has essentially no effect, while ablating the residual preserves the full shared-subspace drop.
2. Vocabulary-level characterization is mixed rather than cleanly separated. `Q_fmt` is enriched for answer/readout tokens, but `Q_resid` still retains option letters, yes/no, reasoning markers, digits, and newline.
3. On open-answer decoding (`gsm8k`, `strategyqa`, `aqua`), reasoning-style and arithmetic/decomposition tags are much more linearly decodable from `Q_resid` than from `Q_fmt`. However, same-width random residual partitions also decode several such surface tags well, so these probes should be treated as characterization rather than the main causal evidence.

## Table 1. Probe fit used to define `Q_fmt`

The probe-derived split was learned inside the layer-10 shared coordinates using held-out logistic probes on decode-time states.

| Probe tag | n_pos | split | ROC-AUC | AP | BalAcc |
|---|---:|---|---:|---:|---:|
| `answer_readout` | 896 | group | 0.985 | 0.898 | 0.946 |
| `option_letter` | 521 | group | 0.996 | 0.957 | 0.974 |
| `newline` | 2563 | group | 1.000 | 0.999 | 0.997 |

Interpretation:

- These probes show that answer-readout structure is strongly linearly recoverable from the original shared subspace.
- `Q_fmt` is therefore not an arbitrary lexical subset; it is a data-driven, probe-derived subspace targeted at answer/readout signals.
- No probe tag was skipped in this stage.

## Table 2. Decomposition summary

| Quantity | Value |
|---|---:|
| Ambient dim `D` | 4096 |
| Shared dim `k_shared` | 32 |
| Format/readout dim `k_fmt` | 3 |
| Residual dim `k_resid` | 29 |
| `max_overlap_fmt_vs_resid` | `8.20e-08` |
| `fmt_vs_full_drop_share` | `-0.05` |
| `resid_vs_full_drop_share` | `1.075` |
| `rand_fmt_vs_full_drop_share` | `-0.075` |
| `rand_resid_vs_full_drop_share` | `0.775` |

Interpretation:

- The orthogonality between `Q_fmt` and `Q_resid` is numerically clean.
- The drop-share numbers already preview the main conclusion: the probe-identifiable format/readout component explains essentially none of the full shared-subspace effect, while the residual explains essentially all of it.
- Values above `1.0` should not be overinterpreted literally; they reflect finite-sample noise and non-additivity of causal effects. The correct qualitative interpretation is that the residual preserves the dominant effect.

## Table 3. Main causal results, pooled

Layer-10 forced-choice evaluation on `commonsenseqa`, `arc_challenge`, `openbookqa`, and `logiqa` (`n=256` pooled).

| Condition | k | Accuracy | Delta vs baseline | p-value |
|---|---:|---:|---:|---:|
| `Baseline` | - | 44.1 | - | - |
| `SharedFull` | 32 | 28.5 | `-15.6` | `0.0010` |
| `FmtOnly` | 3 | 44.9 | `+0.8` | `0.748` |
| `ResidOnly` | 29 | 27.3 | `-16.8` | `0.0005` |
| `RandFmt` | 3 | 45.3 | `+1.2` | `0.370` |
| `RandResid` | 29 | 32.0 | `-12.1` | `0.0035` |

Interpretation:

- This is the single strongest new result.
- Removing `Q_fmt` alone does not hurt performance at all.
- Removing `Q_resid` alone reproduces the full shared-subspace effect.
- A same-width random residual partition also hurts accuracy, which suggests that a large fraction of the causal mass lives in the broad residual/shared region rather than in a tiny, uniquely probe-identified lexical slice.
- The sharp asymmetry is the key rebuttal point: the dominant causal contribution is not carried by the probe-identifiable format/readout directions.

## Table 4. Main causal results, per task

| Task | n | Baseline | `ΔFull` | `ΔFmt` | `ΔResid` | `ΔRandFmt` | `ΔRandResid` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `commonsenseqa` | 64 | 48.4 | `-15.6` | `+1.6` | `-15.6` | `+3.1` | `-10.9` |
| `arc_challenge` | 64 | 59.4 | `-25.0` | `-3.1` | `-31.2` | `+0.0` | `-23.4` |
| `openbookqa` | 64 | 43.8 | `-17.2` | `+1.6` | `-15.6` | `+1.6` | `-9.4` |
| `logiqa` | 64 | 25.0 | `-4.7` | `+3.1` | `-4.7` | `+0.0` | `-4.7` |

Interpretation:

- The strongest single-task signal comes from `arc_challenge`, where `ΔResid` is even larger than `ΔFull`.
- `commonsenseqa` and `openbookqa` show the same qualitative pattern: `FmtOnly` stays near baseline while `ResidOnly` tracks the full drop.
- `logiqa` is the weakest task here, with small drops and no clean separation. It does not contradict the pooled result, but it shows the effect is not equally strong on every benchmark.

## Table 5. Focused unembedding / logit-lens family summary

The scores below are RMS-normalized unembedding projection strengths, so `Q_fmt (k=3)` and `Q_resid (k=29)` are directly comparable.

| Tag family | `Q_fmt` max | `Q_resid` max | `Q_resid / Q_fmt` |
|---|---:|---:|---:|
| `option_letter` | 0.0225 | 0.0161 | 0.715 |
| `yes_no` | 0.0233 | 0.0169 | 0.728 |
| `reasoning_marker` | 0.0221 | 0.0177 | 0.802 |
| `digit` | 0.0343 | 0.0223 | 0.650 |
| `newline` | 0.0109 | 0.0279 | 2.558 |
| `punct` | 0.0418 | 0.0317 | 0.757 |
| `whitespace` | 0.0247 | 0.0279 | 1.130 |

Interpretation:

- This table is important because it shows that the lexical story is mixed, not cleanly separated.
- `Q_fmt` is somewhat stronger on option letters, yes/no, reasoning markers, and digits, but not by a huge margin.
- `Q_resid` is actually stronger on newline and whitespace.
- Therefore, the causal split cannot be reduced to a trivial lexical separation where `Q_fmt` is “all the formatting tokens” and `Q_resid` is “everything semantic.” The two components remain lexically entangled.

## Table 6. Representative top tokens by family

| Basis | Family | Top tokens |
|---|---|---|
| `Q_fmt` | `option_letter` | `A`, `C`, `C`, `E`, `B` |
| `Q_fmt` | `yes_no` | `YES`, `YES`, `Yes`, `Yes`, `NO` |
| `Q_fmt` | `reasoning_marker` | `because`, `thus`, `hence`, `because`, `Hence` |
| `Q_resid` | `option_letter` | `C`, `C`, `B`, `A`, `A` |
| `Q_resid` | `yes_no` | `no`, `YES`, `yes`, `YES`, `No` |
| `Q_resid` | `reasoning_marker` | `therefore`, `because`, `Therefore`, `hence`, `Because` |
| `Q_resid` | `newline` | `\\n` |

Interpretation:

- `Q_fmt` clearly carries readout-style tokens, but it is not pure formatting: it also activates explicit discourse markers like `because` and `thus`.
- `Q_resid` still contains option letters and yes/no tokens, and it also carries discourse markers such as `therefore`, `because`, and `Therefore`.
- This is exactly why the causal experiment matters more than token inspection alone: the vocabulary signatures are mixed, but the causal contribution is highly asymmetric.

## Table 7. Open-answer reasoning-style probes

This probe uses open-answer decode states from `gsm8k`, `strategyqa`, and `aqua` with prompts that explicitly request reasoning and no extra appended answer prefix. Total pooled decode-token sample size is `17181`.

| Tag | n_pos | `Q_resid` AP | `Q_fmt` AP | `Q_rand_resid` AP | `Q_resid - Q_fmt` AP gap |
|---|---:|---:|---:|---:|---:|
| `reasoning_marker` | 120 | 0.564 | 0.041 | 0.726 | 0.522 |
| `step_marker` | 98 | 0.169 | 0.029 | 0.100 | 0.139 |
| `digit` | 1813 | 0.966 | 0.132 | 0.963 | 0.834 |
| `equation_symbol` | 544 | 0.673 | 0.055 | 0.590 | 0.617 |

Interpretation:

- The residual clearly supports much stronger linear readout of reasoning-like and arithmetic/decomposition tags than the small `Q_fmt` subspace.
- The gap is especially large for `digit` and `equation_symbol`, which is consistent with the residual carrying task-relevant answer-content / computational state.
- `reasoning_marker` and `step_marker` are rarer, but even there `Q_resid` is much better than `Q_fmt`.

## Table 8. Open-answer reasoning-style probes, full metrics

| Tag | Basis | ROC-AUC | AP | BalAcc |
|---|---|---:|---:|---:|
| `reasoning_marker` | `Q_resid` | 0.995 | 0.564 | 0.951 |
| `reasoning_marker` | `Q_fmt` | 0.887 | 0.041 | 0.774 |
| `reasoning_marker` | `Q_rand_resid` | 0.993 | 0.726 | 0.953 |
| `step_marker` | `Q_resid` | 0.963 | 0.169 | 0.917 |
| `step_marker` | `Q_fmt` | 0.916 | 0.029 | 0.879 |
| `step_marker` | `Q_rand_resid` | 0.967 | 0.100 | 0.953 |
| `digit` | `Q_resid` | 0.994 | 0.966 | 0.964 |
| `digit` | `Q_fmt` | 0.645 | 0.132 | 0.638 |
| `digit` | `Q_rand_resid` | 0.993 | 0.963 | 0.963 |
| `equation_symbol` | `Q_resid` | 0.971 | 0.673 | 0.906 |
| `equation_symbol` | `Q_fmt` | 0.695 | 0.055 | 0.673 |
| `equation_symbol` | `Q_rand_resid` | 0.957 | 0.590 | 0.887 |

Interpretation:

- `Q_resid` dominates `Q_fmt` on every tag.
- However, same-width `Q_rand_resid` is often competitive with `Q_resid`, and even better on `reasoning_marker`.
- Therefore, this probe should not be framed as “the residual is uniquely special among all residual partitions.” The safer interpretation is narrower: the small probe-derived format/readout slice is insufficient, and a broad non-format residual region carries reasoning-like and arithmetic/decomposition signals.

## Important caveats

1. The causal split is the main result. The classifier and logit-lens analyses are characterization layers on top of it.
2. `Q_fmt` is not purely formatting and `Q_resid` is not purely semantic. Both remain lexically mixed.
3. The `answer_marker` family did not show whole-token hits in the focus-vocab analysis because the current token tagger only catches whole tokens matching patterns like `Final answer` or `Answer:`, while Llama tokenization often splits these strings across multiple pieces.
4. The earlier open-answer probe run with an appended answer prefix produced too few `reasoning_marker` / `step_marker` positives. The latest `noapfx_m128` run is the correct one to cite.

## Recommended bottom-line phrasing

> The new results support a mixed interpretation. The shared decode subspace does contain probe-identifiable answer-readout / formatting structure, but this structure does not explain the observed causal effect. When we factor out a small probe-derived format/readout component from the layer-10 shared subspace, ablating that component alone has essentially no effect, whereas ablating the orthogonal residual preserves the full shared-subspace drop. The residual is still lexically mixed, but it also supports strong linear readout of reasoning-like and arithmetic/decomposition token families in open-answer settings.
