# Energy-balance + LOTO(8) Summary

- Model: `meta-llama/Llama-2-7b-chat-hf` dtype=fp16 device=cuda

- Tasks: ['gsm8k', 'commonsenseqa', 'strategyqa', 'arc_challenge', 'openbookqa', 'qasc', 'logiqa']

- Mode: loto

- Template randomization: True (seed=1234), shuffle_choices=True

- Sharedness: pca_var=0.95, tau=0.001, m_shared=all

- Calibration decode max_new_tokens=64, per_task_max_states=2048

- Evaluation: forced_choice=True (MC tasks only)


## LOTO held-out performance

| Held-out | n  | Protocol   | Baseline          | Shared(full)     | Rand(full)        | Δ(shared-baseline)  | p(shared-baseline) |
|----------|----|------------|-------------------|------------------|-------------------|---------------------|--------------------|
| gsm8k    | 32 | generation | 31.2 [15.6, 46.9] | 12.5 [3.1, 21.9] | 31.2 [15.6, 46.9] | -18.8 [-34.4, -3.1] | 0.0749             |
