# 02 Decode Ablation

Paper role: H2, decode-only removal, leave-one-task-out ablations, and energy-matched controls.

This folder contains the canonical code for the H2 decode-only ablation results.
New camera-ready reruns should use the top-level scripts listed here.

## What Changed And Why

The original import copied three LOTO runners into this folder:

- `run_loto_reasoning.py`: older monolithic LOTO/generation runner.
- `run_loto_reasoning_src.py`: reasoning LOTO runner with the forced-choice fix used for `fc_eval2048` outputs.
- `run_loto_reasoning_refactored.py`: shorter refactor that delegates to `eval_perf.py`.

For camera-ready reproducibility, we collapsed these to one top-level
`run_loto_reasoning.py`. The canonical file is the former
`run_loto_reasoning_src.py`, because it supports `--use_forced_choice 1`,
`--fc_warmup_tokens`, `--fc_prefix_mode`, and `--fc_answer_prefix`, which match
the paper-facing `energy_balance_loto8_reasoning_fc_eval2048` results.

The canonical H2 runners also avoid using `torch.cuda.is_available()` while building argparse defaults. That CUDA probe can block on a busy cluster before `--help` is printed; the scripts now default to `--device cuda`, and users can pass `--device cpu` explicitly for CPU-only dry runs.

## Structure

- `run_loto_reasoning.py`: canonical H2 LOTO/all-task decode-only removal runner with forced-choice support.
- `run_energy_kmatch_reasoning.py`: alpha-match and k-match energy controls for reasoning tasks.
- `run_energy_kmatch_generation.py`: generation-side energy-control runner.
- `benchmark_dataloaders.py`: benchmark loading, prompt construction, and answer parsing helpers.
- `eval_perf.py`: shared evaluation utilities used by some control code.
- `summarize_disturb_cot_results.py`: aggregates LOTO/all-task result JSONs.
- `summarize_disturb_cot_diagnostics.py`: diagnostic summary tables.
- `summarize_energy_kmatch_outputs.py`: energy-control summary aggregation.
- `configs/`: paper parameter records.

## Reproduction Notes

- Use only `Node0` and `Node1` for camera-ready reruns unless the cluster availability changes.
- Keep `PYTHONPATH` pointed at the repo root when running direct Python commands.
- The paper LOTO command uses `n_eval=2048` and `--use_forced_choice 1` for MC/Yes-No tasks; `gsm8k` remains generation-based.

```bash
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
bash scripts/02_h2_decode_ablation/run_mock.sh
```

See `scripts/02_h2_decode_ablation/COMMANDS.md` for exact paper-facing commands.
