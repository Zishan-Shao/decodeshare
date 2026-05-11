# Q1 Rebuttal Draft: Feature Disentanglement Framing

## Recommended framing

The cleanest way to answer this question is **not** to center the discussion on
formatting tokens themselves, but to frame the new result as a **feature
disentanglement experiment**:

- We use linear probes to identify a small, explicitly readable answer-readout /
  formatting slice inside the shared decode subspace.
- We then causally remove that slice and compare it against removing the
  orthogonal remainder.
- The result is that the readout/format slice is present but causally minor,
  whereas the orthogonal remainder carries essentially the full DecodeShare
  effect.

This lets the rebuttal answer the reviewer’s question directly while keeping the
story aligned with the paper’s main claim.

## Naming recommendation

I would **not** rename `Q_resid` all the way to `Q_reasoning`, because the
current evidence does not justify claiming a pure reasoning module.

The safer options are:

- keep the symbol `Q_resid`, but describe it in prose as the
  **decision/computation residual**
- or write `Q_resid` (decision/computation residual) on first mention

This gives you the rhetorical benefit of not calling the main object “residual”
in prose, without overclaiming semantic purity.

## Rebuttal text

```text
This is an important question. To characterize what DecodeShare encodes more directly, we performed a feature-disentanglement analysis of the layer-10 shared decode subspace using held-out linear probes. Concretely, we first identified a small probe-readable answer-readout / formatting slice inside the 32D shared basis, using probes for `answer_readout`, `option_letter`, and `newline` (AP `0.898 / 0.957 / 0.999`). This yields a 3D readout/format subspace, `Q_fmt`, and a 29D orthogonal complement, `Q_resid`, which we interpret as the decision/computation residual.

The key question is then causal: does the main DecodeShare effect come from this probe-identifiable readout/format slice, or from the orthogonal residual? Table 1 shows a clear answer. On four forced-choice tasks (`n=256` pooled), ablating the full shared subspace reduces accuracy by `15.6` points (`p=0.001`), ablating the 3D readout/format slice alone has essentially no effect (`+0.8` points, `p=0.748`), while ablating the orthogonal 29D residual nearly reproduces the full effect (`-16.8` points, `p=5e-4`). Thus, answer-readout / formatting information is certainly present in the shared subspace, but it is not what explains the main DecodeShare effect.

We then asked what is readable from the decision/computation residual itself. In open-answer settings (`gsm8k`, `strategyqa`, `aqua`), held-out probes show that the residual supports much stronger reasoning-style and intermediate-computation signals than the small readout/format slice: for explicit reasoning-connective tokens (`because`, `therefore`, `thus`, `hence`, etc.), AP is `0.564` on `Q_resid` versus `0.041` on `Q_fmt`; for `step_marker`, `0.169` versus `0.029`; for `digit`, `0.966` versus `0.132`; and for `equation_symbol`, `0.673` versus `0.055` (Table 2). We therefore do not interpret DecodeShare as a trivial output-formatting channel. The more supported conclusion is that the shared decode subspace contains a probe-identifiable readout/format component, but its dominant causal mass lies in an orthogonal decision/computation residual that also carries reasoning-style and intermediate-computation structure.
```

## Table 1. Probe-identified readout/format slice and causal split

| Quantity / Condition | Value |
|---|---:|
| Shared basis dimension | `32` |
| Probe-identified readout/format dimension `Q_fmt` | `3` |
| Orthogonal residual dimension `Q_resid` | `29` |
| Probe AP: `answer_readout` | `0.898` |
| Probe AP: `option_letter` | `0.957` |
| Probe AP: `newline` | `0.999` |
| Pooled delta: `SharedFull` | `-15.6 pts` (`p=0.001`) |
| Pooled delta: `FmtOnly` | `+0.8 pts` (`p=0.748`) |
| Pooled delta: `ResidOnly` | `-16.8 pts` (`p=5e-4`) |

Why this table is strong:

- It first establishes that answer-readout / formatting structure is truly present and linearly recoverable.
- It then immediately shows that this slice is causally minor.
- It keeps the reader focused on the main argumentative move: **presence of formatting is not the same as causal dominance of formatting**.

## Table 2. What is readable from the decision/computation residual?

Open-answer probe results on `gsm8k`, `strategyqa`, `aqua` (`n=17181` decode-time tokens pooled).

| Tag | `Q_resid` AP | `Q_fmt` AP | Gap |
|---|---:|---:|---:|
| `reasoning_marker` | `0.564` | `0.041` | `+0.522` |
| `step_marker` | `0.169` | `0.029` | `+0.139` |
| `digit` | `0.966` | `0.132` | `+0.834` |
| `equation_symbol` | `0.673` | `0.055` | `+0.617` |

Why this table is useful:

- It gives the reader a positive description of what is encoded beyond readout/formatting.
- It supports a stronger narrative than “not formatting”: the residual is also rich in
  reasoning-like and intermediate-computation signals.
- It stays appropriately conservative. These are probe-readable features, not proof of
  a perfectly pure reasoning module.

## Optional one-sentence unembedding add-on

If you want to keep unembedding at all, I would reduce it to one sentence:

```text
A classifier-aligned unembedding analysis is directionally consistent with this picture: probe directions associated with readout resolve to option-letter / yes-no / newline families, while residual probe directions resolve to reasoning-connective, step-organization, digit, and equation-symbol families; we therefore treat unembedding as supportive characterization rather than as the main evidence.
```

## Why this version is better than the current draft

- It answers the reviewer’s “unembedding or linear classifiers” question directly, but keeps the cleanest evidence first.
- It turns the story from a defensive “we ruled out formatting” response into a positive
  **disentanglement** result.
- It avoids overclaiming that the residual is a pure reasoning subspace.
- It avoids making unembedding carry more weight than the current evidence supports.
