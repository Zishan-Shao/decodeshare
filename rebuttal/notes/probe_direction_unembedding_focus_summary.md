# Probe-Direction Unembedding: Focus-Family Summary

This note summarizes a classifier-aligned unembedding post-processing analysis on
the saved `exp_1g_probe_direction_unembedding_layer10_main.json` results.

Key point: the raw whole-vocabulary top tokens for layer-10 probe directions are
still noisy, so they should **not** be used directly in the rebuttal. However,
when we restrict the unembedding view to the probe-relevant token family, the
unembedding becomes much more interpretable and agrees well with the linear-probe
story.

## Format/readout directions

These directions come from the saved A5 linear probes on `Q_shared`.

| Direction | AP | Focus-family top tokens |
|---|---:|---|
| `answer_readout` | `0.898` | `C, B, E, D, No, no, No, B, E, no` |
| `option_letter` | `0.957` | `B, D, E, C, C, A, B, D, E, A` |
| `newline` | `0.999` | `\\n` |

Interpretation:

- The `option_letter` and `newline` directions are especially clean.
- The broader `answer_readout` direction resolves to a mixture of option letters and yes/no tokens, which is exactly what the probe target represents.
- This is much more consistent with the linear-probe interpretation than the noisy whole-vocab top-token dump.

## Residual reasoning-style directions

These directions come from fresh open-answer probes fitted inside `Q_resid` on
`gsm8k`, `strategyqa`, and `aqua`.

| Direction | AP | Focus-family top tokens |
|---|---:|---|
| `reasoning_marker` | `0.466` | `Therefore, SO, therefore, SO, because, Since, since, Since, So, So` |
| `step_marker` | `0.043` | `next, Next, Finally, next, Next, First, finally, first, step, Step` |
| `digit` | `0.965` | `0, ₀, 9, 8, 5, ₇, ₉, ₆, 6, 7` |
| `equation_symbol` | `0.659` | `<-, =., ================, ={{, =\\{, }}%, }%, =(, {%, {%` |

Interpretation:

- The residual `reasoning_marker` direction cleanly resolves to discourse-connective tokens such as `Therefore`, `because`, `Since`, and `So`.
- The residual `step_marker` direction resolves to explicit organizational markers like `next`, `Finally`, `First`, and `step`.
- The residual `digit` and `equation_symbol` directions resolve to arithmetic-format tokens as expected.

## Writing recommendation

This analysis is strong enough to justify a **soft** claim:

> A classifier-aligned unembedding analysis is consistent with the linear-probe picture: probe directions associated with answer readout resolve to option-letter / yes-no / newline families, while probe directions fit inside the residual resolve to reasoning-connective, step-organization, digit, and equation-symbol families.

But it is still safer **not** to say that the unembedding and linear-probe results are “fully identical” or “completely consistent” in a strong sense, because:

1. the whole-vocabulary unembedding at layer 10 remains noisy;
2. the clean interpretation only appears after restricting to the relevant token family; and
3. the unembedding is best used as supportive characterization, whereas the causal split remains the main evidence.

## Recommendation for rebuttal positioning

- Keep unembedding as a secondary sentence, not the main axis.
- Say that classifier-aligned unembedding is directionally consistent with the linear probes.
- Do **not** make the unembedding carry the core argument.
