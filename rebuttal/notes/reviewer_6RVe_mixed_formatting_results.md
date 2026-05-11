# Reviewer 6RVe: Mixed Formatting + Non-Formatting Evidence

## Short claim

Our evidence supports a mixed interpretation: the decode-shared subspace contains a stable answer-readout / formatting component, but a purely formatting-only account is insufficient. Its causal effect persists after removing the original answer-label format, and in open-ended generation it often preserves extractable answer format while changing the answer itself.

## Rebuttal-ready paragraph

We agree that the shared decode subspace is not a purely semantic factor with formatting fully removed. Our current evidence instead supports a mixed interpretation: it contains a stable answer-readout / formatting component, but also task-relevant non-format decision information. First, probing through the unembedding shows consistent enrichment of answer-format / readout tokens, including option letters, answer markers, yes/no tokens, and reasoning connectors. Second, however, a purely formatting-only account is insufficient: when we rewrite answer labels from A/B/C/D to 1/2/3/4, shared-subspace ablation still causes a substantial accuracy drop, while matched control subspaces do not. Third, in open-ended GSM8K generation, accuracy drops sharply under shared ablation even though extraction rates remain high, and we observe cases where the model still produces a normal extractable final answer format but with the wrong numeric answer. Taken together, these results suggest that the shared subspace bundles formatting/readout features with causally important non-format decision content.

## Table 1. Unembedding probe: what is visibly encoded?

| Probe | Main enriched tokens | Stability across template seeds | Interpretation |
|---|---|---|---|
| Layer 28 focus tokens | `A/B/C/D/E`, `Answer`, `Final`, `Yes/No`, `Thus/Therefore/Hence/Because` | `focus_topk_overlap = 1.00`; `delta_rms_topk_overlap = 0.832` | Strong answer-readout / reasoning-scaffold signature |
| Layer 24 focus tokens | `A/B/C/D/E`, `Answer`, `Final`, `Yes/No`, `Thus/Therefore/Hence/Because` | `focus_topk_overlap = 1.00`; `delta_rms_topk_overlap = 0.753` | Same qualitative pattern one layer earlier |

## Table 2. Main causal test against the "pure formatting" hypothesis

Experiment: rewrite answer labels from `A/B/C/D` to `1/2/3/4`, then score forced-choice accuracy over the rewritten labels.

| Task | n | Baseline | Shared | Delta Shared | Ctrl(E) | Delta Ctrl(E) | Rand(E) | Delta Rand(E) |
|---|---|---|---|---|---|---|---|---|
| commonsenseqa | 64 | 57.8 [45.3, 68.8] | 45.3 [32.8, 57.8] | -12.5 [-23.4, -1.6] (p=0.051) | 57.8 [45.3, 70.3] | +0.0 [+0.0, +0.0] (p=1) | 57.8 [46.9, 70.3] | +0.0 [+0.0, +0.0] (p=1) |
| arc_challenge | 64 | 48.4 [35.9, 59.4] | 37.5 [26.6, 50.0] | -10.9 [-21.9, -1.6] (p=0.092) | 48.4 [35.9, 59.4] | +0.0 [+0.0, +0.0] (p=1) | 48.4 [35.9, 60.9] | +0.0 [+0.0, +0.0] (p=1) |
| Pooled | 128 | 53.1 [44.5, 61.7] | 41.4 [33.6, 50.0] | -11.7 [-19.5, -4.7] (p=0.006) | 53.1 [44.5, 61.7] | +0.0 [+0.0, +0.0] (p=1) | 53.1 [44.5, 60.9] | +0.0 [+0.0, +0.0] (p=1) |

Interpretation: removing the original letter format weakens the "pure formatting" explanation, yet the shared-subspace effect remains large. This supports a mixed formatting + non-format account.

## Table 3. Open-ended generation: accuracy can collapse without extraction collapsing

Held-out GSM8K generation, `n=64`.

| Layer / Condition | Accuracy | Extraction rate | EOS rate | Avg new tokens | Interpretation |
|---|---|---|---|---|---|
| L28 baseline | 25.0 | 85.9 | 73.4 | 196.3 | Baseline open-answer performance |
| L28 shared_full | 7.8 | 89.1 | 53.1 | 171.5 | Accuracy drops sharply, but extraction remains high |
| L28 rand_full | 26.6 | 92.2 | 78.1 | 197.4 | Random control does not reproduce the accuracy drop |
| L26 baseline | 25.0 | 85.9 | 73.4 | 196.3 | Baseline replication |
| L26 shared_full | 9.4 | 90.6 | 64.1 | 165.6 | Same pattern: low accuracy, high extraction |
| L26 rand_full | 23.4 | 92.2 | 76.6 | 190.3 | Random control stays near baseline |

Interpretation: this argues against a simple "the model can no longer emit or extract the answer format" explanation.

## Example to mention in prose

Representative GSM8K example: `gsm8k-test-56` (Jon's car tune-ups). Under `shared_full`, the model still emits an extractable final-answer string, but the number is wrong: `Final Answer 9 tune-ups` instead of the gold answer `3`. This is useful as a concrete example that formatting can remain intact while answer content changes.

## Short version if space is tight

Unembedding shows a stable answer-readout signature (`A/B/C/D/E`, `Answer`, `Final`) together with `Yes/No` and reasoning markers (`Thus`, `Therefore`, `Because`). However, a purely formatting-only account is insufficient: after relabeling answers from `A/B/C/D` to `1/2/3/4`, shared-subspace ablation still causes a large pooled accuracy drop (`53.1 -> 41.4`, `Δ=-11.7`, `p=0.006`), while matched controls remain at baseline. In open-ended GSM8K, shared ablation also lowers accuracy without lowering extraction rate, indicating that the effect is not just broken answer formatting.
