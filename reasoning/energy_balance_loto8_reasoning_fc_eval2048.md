# Energy-balance + LOTO Summary (refactored)

- Model: `meta-llama/Llama-2-7b-chat-hf` dtype=fp32 device=cuda

- Tasks: ['gsm8k', 'commonsenseqa', 'strategyqa', 'aqua', 'arc_challenge', 'openbookqa', 'qasc', 'logiqa']

- Mode: loto (loto_eval_mode=heldout)

- Template randomization: True (seed=1234), shuffle_choices=True

- Sharedness: pca_var=0.95, tau=0.001, m_shared=all, k_eval=auto

- Calibration: calib_decode_max_new_tokens=128, per_task_max_states=20000

- Evaluation: forced_choice=True warmup_tokens=0


## LOTO held-out performance

| Held-out      | n    | Protocol      | Baseline          | Decode-shared     | Prefill-shared    | Random            | Δ(Decode-Prefill) [CI] | p      |
|---------------|------|---------------|-------------------|-------------------|-------------------|-------------------|------------------------|--------|
| gsm8k         | 1319 | generation    | 4.9 [3.7, 6.1]    | 2.9 [2.0, 3.9]    | 5.5 [4.3, 6.7]    | 5.2 [4.1, 6.5]    | -2.7 [-4.1, -1.1]      | 0.0006 |
| commonsenseqa | 1221 | forced_choice | 54.1 [51.3, 56.8] | 45.9 [43.2, 48.7] | 54.5 [51.7, 57.5] | 53.9 [51.0, 56.6] | -8.7 [-11.5, -6.1]     | 0.0001 |
| strategyqa    | 687  | forced_choice | 55.5 [51.8, 59.1] | 53.6 [49.8, 57.5] | 52.8 [49.1, 56.6] | 55.2 [51.5, 59.0] | +0.7 [-1.3, +2.8]      | 0.412  |
| aqua          | 254  | forced_choice | 24.0 [18.9, 29.5] | 19.7 [15.0, 24.8] | 22.0 [16.9, 27.2] | 21.7 [16.9, 27.2] | -2.4 [-8.3, +3.5]      | 0.516  |
| arc_challenge | 1172 | forced_choice | 50.9 [48.0, 53.7] | 38.6 [35.7, 41.5] | 49.3 [46.4, 52.2] | 50.6 [47.8, 53.5] | -10.8 [-13.6, -7.8]    | 0.0001 |
| openbookqa    | 500  | forced_choice | 50.2 [45.8, 54.6] | 44.2 [39.8, 48.8] | 49.0 [44.8, 53.2] | 51.6 [47.2, 55.8] | -4.8 [-9.0, -0.6]      | 0.0231 |
| qasc          | 926  | forced_choice | 48.5 [45.2, 51.6] | 42.5 [39.5, 45.8] | 48.6 [45.5, 51.7] | 48.5 [45.2, 51.6] | -6.0 [-8.9, -3.1]      | 0.0002 |
| logiqa        | 651  | forced_choice | 32.7 [29.2, 36.4] | 28.1 [24.6, 31.5] | 33.0 [29.5, 36.7] | 33.2 [29.5, 36.7] | -4.9 [-8.4, -1.4]      | 0.0079 |

## Basis diagnostics (per fold)

### Holdout: gsm8k

- k_decode=134, k_prefill=70, k_eval=70

- Similarity(k): max_cos=0.697, mean_cos=0.210, min_cos=0.005


### Holdout: commonsenseqa

- k_decode=153, k_prefill=70, k_eval=70

- Similarity(k): max_cos=0.686, mean_cos=0.213, min_cos=0.004


### Holdout: strategyqa

- k_decode=155, k_prefill=69, k_eval=69

- Similarity(k): max_cos=0.675, mean_cos=0.207, min_cos=0.004


### Holdout: aqua

- k_decode=147, k_prefill=73, k_eval=73

- Similarity(k): max_cos=0.693, mean_cos=0.207, min_cos=0.000


### Holdout: arc_challenge

- k_decode=151, k_prefill=67, k_eval=67

- Similarity(k): max_cos=0.682, mean_cos=0.207, min_cos=0.003


### Holdout: openbookqa

- k_decode=147, k_prefill=75, k_eval=75

- Similarity(k): max_cos=0.703, mean_cos=0.217, min_cos=0.003


### Holdout: qasc

- k_decode=147, k_prefill=73, k_eval=73

- Similarity(k): max_cos=0.686, mean_cos=0.214, min_cos=0.004


### Holdout: logiqa

- k_decode=144, k_prefill=68, k_eval=68

- Similarity(k): max_cos=0.666, mean_cos=0.204, min_cos=0.003
