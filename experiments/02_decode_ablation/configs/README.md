# Decode Ablation Configs

These files record the paper-scale H2 ablation settings. The scripts currently use argparse, so configs here are human-readable run records rather than directly consumed YAML.

The LOTO settings should point to the canonical `experiments/02_decode_ablation/run_loto_reasoning.py`. This matters because the paper-facing LOTO summaries use the forced-choice evaluation path (`--use_forced_choice 1`) added in that canonical runner.

- `loto_forced_choice.yaml`: Table/Figure H2 LOTO rerun settings.
