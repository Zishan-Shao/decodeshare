# H1 Diagnostic Analysis

These scripts consume activation dumps from
`experiments/01_sharedness/collect_activations.py` and generate the H1
diagnostic figures/tables.

The reusable method code is imported from `decodeshare.sharedness`; this folder
contains paper-facing analysis entry points only.

- `analyze_within_vs_mixed.py`: within-category vs mixed-category sharedness.
- `analyze_task_count_convergence.py`: task-count convergence.
- `analyze_phase_convergence.py`: decode/prefill/decode-step convergence.
- `analyze_tau_sensitivity.py`: PCA-retention and tau sensitivity.
