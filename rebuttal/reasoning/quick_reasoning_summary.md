# Quick Reasoning Rebuttal Check

This is a quick-turn held-out-task check for reasoning-heavy tasks.

- Model: `meta-llama/Llama-2-7b-chat-hf` dtype=fp16 device=cuda
- Tasks used for basis/eval: `gsm8k,commonsenseqa,strategyqa,arc_challenge,openbookqa,qasc,logiqa`
- Held-out tasks run: `gsm8k,logiqa`
- Per-task eval size: n_eval=32, n_subspace=64, layer=10
- Protocol: LOTO heldout, forced_choice=True, do_sample=False

## Per-task results

| Held-out | Type                              | n  | Baseline           | Decode-shared | Prefill-shared | Random | D-P delta           | p     |
|----------|-----------------------------------|----|--------------------|---------------|----------------|--------|---------------------|-------|
| gsm8k    | Open-ended numeric reasoning      | 32 | 0.0                | 0.0           | 0.0            | 0.0    | +0.0 [+0.0, +0.0]   | 1     |
| logiqa   | Logical reasoning multiple choice | 32 | 31.2 (chance 25.0) | 15.6          | 34.4           | 34.4   | -18.8 [-31.2, -6.2] | 0.036 |

## Aggregate

- Mean accuracy: baseline=15.6, decode_shared=7.8, prefill_shared=17.2, random=17.2
- Mean deltas vs baseline: decode=-7.8, prefill=+1.6, random=+1.6
- Mean decode-minus-prefill delta: -9.4
- Informative held-out tasks: `logiqa`
- Inconclusive due to baseline floor/chance: `gsm8k`

## Interpretation

- `gsm8k` is currently inconclusive: baseline is at or near floor/chance, so this fold does not say much about decode-vs-prefill selectivity.
- `logiqa` is informative: decode-shared changes accuracy by -15.6 vs baseline and -18.8 vs prefill-shared.
- Use informative folds as rebuttal evidence that the decode-shared phenomenon is not confined to short classification tasks.
