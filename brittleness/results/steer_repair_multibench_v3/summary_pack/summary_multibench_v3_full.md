# Multibench v3 summary (full)

## Run config

- **model**: `meta-llama/Llama-2-7b-chat-hf`
- **dtype**: `fp32`
- **device**: `cuda`
- **layer**: `10`
- **basis_source**: `neutral`
- **basis_k**: `512`
- **basis_max_states**: `1024`
- **v_est_templates**: `0`
- **betas**: `0,0.25,0.5,0.75,1.0`
- **lambdas**: `0,0.5,1.0`
- **n_rand**: `5`

## Tasks

boolq, rte, sst2

## Candidate calibration

Each task chooses a forced-choice candidate pair by maximizing baseline forced-choice accuracy on a small balanced subset.

| Task | Chosen | Baseline acc | sh(v) | Top-3 candidates (acc) |
|---|---|---:|---:|---|
| boolq | Yes/No (Yes/No) | 0.583 | 0.405 | Yes/No(0.583), True/False(0.552) |
| rte | True/False (True/False) | 0.615 | 0.386 | True/False(0.615), Yes/No(0.562) |
| sst2 | Good/Bad (Good/Bad) | 0.688 | 0.391 | Good/Bad(0.688), Yes/No(0.599), True/False(0.599) |

## Main results (template robustness)

Metrics are computed on correct-signed margin shift aggregated across templates at the final $\lambda$: mean $\mu$, template std $\sigma_{tmpl}$, worst-case template mean, and worst-case anti-steer rate (lower is better).

| Task | Method | beta | mu | sigma_tmpl | worst | anti_worst | worst template id |
|---|---|---:|---:|---:|---:|---:|---:|
| boolq | decode_est | 0 | 0.0383 | 0.0019 | 0.0357 | 0.3633 | 2 |
| boolq | decode_beta0.5 | 0.5 | 0.0357 | 0.0026 | 0.0321 | 0.3594 | 2 |
| boolq | decode_fixed | 1 | 0.0312 | 0.0034 | 0.0267 | 0.3633 | 2 |
| boolq | rand_matched |  | -0.0100 | 0.0015 | -0.0118 | 0.5695 | 2 |
| rte | decode_est | 0 | 0.0173 | 0.0095 | 0.0104 | 0.4844 | 0 |
| rte | decode_beta0.5 | 0.5 | 0.0171 | 0.0092 | 0.0101 | 0.4766 | 2 |
| rte | decode_fixed | 1 | 0.0162 | 0.0085 | 0.0089 | 0.4648 | 2 |
| rte | rand_matched |  | -0.0005 | 0.0020 | -0.0033 | 0.5195 | 1 |
| sst2 | decode_est | 0 | 0.0099 | 0.0126 | -0.0007 | 0.4336 | 2 |
| sst2 | decode_beta0.5 | 0.5 | 0.0130 | 0.0132 | 0.0011 | 0.4492 | 2 |
| sst2 | decode_fixed | 1 | 0.0154 | 0.0134 | 0.0028 | 0.4648 | 2 |
| sst2 | rand_matched |  | 0.0012 | 0.0114 | -0.0082 | 0.5594 | 0 |

## Recommended beta (simple robust heuristic)

We recommend a beta per task by maximizing worst-case mean shift while preserving at least 90\% of the baseline mean shift (beta=0). This is a reporting aid (not used to tune results).

| Task | recommended beta | recommended method |
|---|---:|---|
| boolq | 0.00 | decode_est |
| rte | 0.25 | decode_beta0.25 |
| sst2 | 1.00 | decode_fixed |

## Per-template breakdown (selected methods)

### boolq

| template | decode_est mean | decode_beta0.5 mean | decode_fixed mean |
|---|---:|---:|---:|
| T0 | 0.0401 | 0.0372 | 0.0323 |
| T1 | 0.0393 | 0.0380 | 0.0347 |
| T2 | 0.0357 | 0.0321 | 0.0267 |

| template | decode_est anti | decode_beta0.5 anti | decode_fixed anti |
|---|---:|---:|---:|
| T0 | 0.3633 | 0.3594 | 0.3633 |
| T1 | 0.2969 | 0.2812 | 0.2656 |
| T2 | 0.3594 | 0.3594 | 0.3516 |

### rte

| template | decode_est mean | decode_beta0.5 mean | decode_fixed mean |
|---|---:|---:|---:|
| T0 | 0.0104 | 0.0112 | 0.0116 |
| T1 | 0.0307 | 0.0301 | 0.0281 |
| T2 | 0.0107 | 0.0101 | 0.0089 |

| template | decode_est anti | decode_beta0.5 anti | decode_fixed anti |
|---|---:|---:|---:|
| T0 | 0.4844 | 0.4766 | 0.4648 |
| T1 | 0.3320 | 0.3242 | 0.3203 |
| T2 | 0.3789 | 0.3828 | 0.3750 |

### sst2

| template | decode_est mean | decode_beta0.5 mean | decode_fixed mean |
|---|---:|---:|---:|
| T0 | 0.0275 | 0.0314 | 0.0339 |
| T1 | 0.0030 | 0.0063 | 0.0095 |
| T2 | -0.0007 | 0.0011 | 0.0028 |

| template | decode_est anti | decode_beta0.5 anti | decode_fixed anti |
|---|---:|---:|---:|
| T0 | 0.3555 | 0.3086 | 0.3008 |
| T1 | 0.4297 | 0.4492 | 0.4648 |
| T2 | 0.4336 | 0.4492 | 0.4219 |

## LaTeX tables

Written to `tables_multibench_v3_full.tex` (requires `booktabs` and `multirow`).

## Figures

- `fig_beta_vs_mean_of_means.pdf`
- `fig_beta_vs_worst_case_mean.pdf`
- `fig_beta_vs_anti_worst.pdf`
- `fig_beta_vs_sigma_tmpl.pdf`
- `fig_beta_vs_slope.pdf`
- `fig_beta_vs_sharedness.pdf`
- `fig_sharedness_vs_worst.pdf`
- `fig_sharedness_vs_anti_worst.pdf`
- `fig_candidate_acc_boolq.pdf`
- `fig_heatmap_mean_boolq.pdf`
- `fig_heatmap_anti_boolq.pdf`
- `fig_candidate_acc_rte.pdf`
- `fig_heatmap_mean_rte.pdf`
- `fig_heatmap_anti_rte.pdf`
- `fig_candidate_acc_sst2.pdf`
- `fig_heatmap_mean_sst2.pdf`
- `fig_heatmap_anti_sst2.pdf`
