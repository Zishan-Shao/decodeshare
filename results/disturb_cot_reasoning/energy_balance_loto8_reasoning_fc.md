# Energy-balance + LOTO(8) Summary

- Model: `meta-llama/Llama-2-7b-chat-hf` dtype=fp32 device=cuda

- Tasks: ['gsm8k', 'commonsenseqa', 'strategyqa', 'aqua', 'arc_challenge', 'openbookqa', 'qasc', 'logiqa']

- Mode: loto

- Template randomization: True (seed=1234), shuffle_choices=True

- Sharedness: pca_var=0.95, tau=0.001, m_shared=all

- Calibration decode max_new_tokens=128, per_task_max_states=20000

- Evaluation: forced_choice=True (MC tasks only)


## LOTO held-out performance

| Held-out      | n   | Protocol      | Baseline          | Shared(full)      | Rand(full)        | Δ(shared-baseline)  | p(shared-baseline) |
|---------------|-----|---------------|-------------------|-------------------|-------------------|---------------------|--------------------|
| gsm8k         | 256 | generation    | 4.7 [2.3, 7.4]    | 3.5 [1.6, 5.9]    | 5.1 [2.7, 7.8]    | -1.2 [-4.3, +2.0]   | 0.631              |
| commonsenseqa | 256 | forced_choice | 51.6 [45.7, 57.8] | 47.7 [41.4, 53.9] | 50.8 [44.5, 57.0] | -3.9 [-9.8, +1.6]   | 0.219              |
| strategyqa    | 256 | forced_choice | 57.0 [51.2, 62.9] | 53.5 [47.7, 59.4] | 57.8 [52.0, 63.7] | -3.5 [-7.4, +0.0]   | 0.0947             |
| aqua          | 254 | forced_choice | 24.0 [18.9, 29.5] | 17.3 [12.6, 22.0] | 22.0 [16.9, 27.2] | -6.7 [-13.4, -0.4]  | 0.0547             |
| arc_challenge | 256 | forced_choice | 51.6 [45.3, 57.4] | 42.6 [36.7, 48.4] | 51.6 [45.7, 57.8] | -9.0 [-14.8, -2.7]  | 0.0099             |
| openbookqa    | 256 | forced_choice | 52.7 [46.5, 59.0] | 41.4 [35.2, 47.3] | 54.7 [48.4, 60.5] | -11.3 [-18.4, -4.3] | 0.0027             |
| qasc          | 256 | forced_choice | 50.4 [44.5, 56.6] | 41.8 [35.5, 47.7] | 51.2 [44.9, 57.4] | -8.6 [-14.5, -2.7]  | 0.006              |
| logiqa        | 256 | forced_choice | 35.5 [29.7, 41.8] | 24.2 [19.1, 29.7] | 37.1 [31.2, 43.0] | -11.3 [-19.1, -3.5] | 0.0062             |
