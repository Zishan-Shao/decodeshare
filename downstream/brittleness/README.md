# Brittleness Provenance

This folder keeps downstream steering-brittleness and robustness provenance
scripts in the same `exp_*.py` style used by other downstream folders.

Current entry points:

- `exp_multibench_steering_repair.py`: multi-benchmark steering projection
  repair and template-robustness experiment.
- `summarize_multibench_full.py`: summary tables, plots, and Markdown report
  for the multibench steering repair outputs.
- `exp_pirate_projection_patch.py`: pirate-style steering projection sanity
  check with fail-fast diagnostics.
- `exp_reasoning_loto.py`: reasoning LOTO provenance run for brittleness
  checks.

The paper-facing steering-control copies live in `experiments/05_steering_controls/`.
This folder is retained for downstream/provenance workflows.
