# Energy-balance + LOTO Summary (refactored)

- Model: `meta-llama/Llama-2-7b-chat-hf` dtype=fp16 device=cuda

- Tasks: ['gsm8k', 'commonsenseqa', 'strategyqa', 'arc_challenge', 'openbookqa', 'qasc', 'logiqa']

- Mode: loto (loto_eval_mode=heldout)

- Template randomization: True (seed=1234), shuffle_choices=True

- Sharedness: pca_var=0.95, tau=0.001, m_shared=all, k_eval=auto

- Calibration: calib_decode_max_new_tokens=48, per_task_max_states=2048

- Evaluation: forced_choice=True warmup_tokens=0


## LOTO held-out performance

| Held-out | n  | Protocol      | Baseline          | Decode-shared    | Prefill-shared    | Random            | Δ(Decode-Prefill) [CI] | p     |
|----------|----|---------------|-------------------|------------------|-------------------|-------------------|------------------------|-------|
| logiqa   | 32 | forced_choice | 31.2 [15.6, 43.8] | 15.6 [3.1, 28.1] | 34.4 [15.6, 50.0] | 34.4 [18.8, 53.1] | -18.8 [-31.2, -6.2]    | 0.036 |

## Basis diagnostics (per fold)

### Holdout: logiqa

- k_decode=172, k_prefill=70, k_eval=70

- Similarity(k): max_cos=0.650, mean_cos=0.196, min_cos=0.001

