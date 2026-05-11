# Rebuttal Reasoning Check

This folder is for a quick-turn reasoning generalization check that extends the
current rebuttal beyond short classification benchmarks.

Files:

- `quick_reasoning_sweep.py`: thin wrapper around `reasoning/disturb_CoT_shared_loto_reasoning.py`
- `raw/`: per-heldout-task JSON and Markdown emitted by the underlying evaluator
- `quick_reasoning_summary.json`: compact machine-readable summary across heldout tasks
- `quick_reasoning_summary.md`: compact human-readable summary for rebuttal drafting

Why this setup:

- It reuses the existing decode-aligned LOTO evaluator instead of introducing a second implementation.
- It targets the most rebuttal-relevant reasoning cases first:
  - `gsm8k`: open-ended numeric generation
  - `logiqa`: harder logical multiple-choice reasoning
- It keeps the run small enough to finish quickly, then leaves a clean path to scale up.

Suggested command:

```bash
CUDA_VISIBLE_DEVICES=3 /home/zs89/miniconda3/envs/flashsvd/bin/python rebuttal/reasoning/quick_reasoning_sweep.py
```

Notes:

- The current repo loader already supports `gsm8k`, `strategyqa`, `commonsenseqa`,
  `arc_challenge`, `openbookqa`, `qasc`, `logiqa`, `boolq`, `piqa`, and `aqua`.
- `mmlu` and `MATH` are not wired into the local loader yet, so they are not part of
  this quick-turn check.












