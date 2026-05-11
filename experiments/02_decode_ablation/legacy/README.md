# Legacy Decode-Ablation Scripts

This directory keeps non-canonical LOTO variants from the original workspace.

- `run_loto_generation_legacy.py`: older monolithic LOTO/generation runner. It is retained because it records the earlier generation-only experiment path, but it does not expose the forced-choice options used by the `fc_eval2048` paper summaries.
- `run_loto_reasoning_refactored_experimental.py`: cleaner refactor that delegates to `eval_perf.py`. It is retained as reference code, but it does not preserve every behavior of the paper runner, especially the staged reasoning-token removal noted in its docstring.

Use the top-level `../run_loto_reasoning.py` for camera-ready reruns.
