# Energy-balance + LOTO Summary (refactored)

- Model: `meta-llama/Llama-2-7b-chat-hf` dtype=fp16 device=cuda

- Tasks: ['gsm8k', 'commonsenseqa', 'strategyqa', 'arc_challenge', 'openbookqa', 'qasc', 'logiqa']

- Mode: loto (loto_eval_mode=heldout)

- Template randomization: True (seed=1234), shuffle_choices=True

- Sharedness: pca_var=0.95, tau=0.001, m_shared=all, k_eval=auto

- Calibration: calib_decode_max_new_tokens=48, per_task_max_states=2048

- Evaluation: forced_choice=True warmup_tokens=0


## LOTO held-out performance

| Held-out | n  | Protocol   | Baseline       | Decode-shared  | Prefill-shared | Random         | Δ(Decode-Prefill) [CI] | p |
|----------|----|------------|----------------|----------------|----------------|----------------|------------------------|---|
| gsm8k    | 32 | generation | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | +0.0 [+0.0, +0.0]      | 1 |

## Basis diagnostics (per fold)

### Holdout: gsm8k

- k_decode=136, k_prefill=77, k_eval=77

- Similarity(k): max_cos=0.676, mean_cos=0.209, min_cos=0.001

