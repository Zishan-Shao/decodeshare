# Energy-balance + LOTO(8) Summary

- Model: `meta-llama/Llama-2-7b-chat-hf` dtype=fp32 device=cuda

- Tasks: ['gsm8k', 'commonsenseqa', 'strategyqa', 'aqua', 'arc_challenge', 'openbookqa', 'qasc', 'logiqa']

- Mode: loto

- Template randomization: True (seed=1234), shuffle_choices=True

- Sharedness: pca_var=0.95, tau=0.001, m_shared=all

- Calibration decode max_new_tokens=128, per_task_max_states=20000

- Evaluation: forced_choice=True (MC tasks only)


## LOTO held-out performance

| Held-out      | n    | Protocol      | Baseline          | Shared(full)      | Rand(full)        | Δ(shared-baseline)  | p(shared-baseline) |
|---------------|------|---------------|-------------------|-------------------|-------------------|---------------------|--------------------|
| gsm8k         | 1319 | generation    | 4.9 [3.8, 6.0]    | 2.3 [1.5, 3.1]    | 4.7 [3.6, 5.8]    | -2.6 [-3.9, -1.3]   | 0.0001             |
| commonsenseqa | 1221 | forced_choice | 54.1 [51.3, 56.8] | 50.0 [47.3, 52.9] | 54.5 [51.6, 57.2] | -4.0 [-6.7, -1.2]   | 0.0043             |
| strategyqa    | 687  | forced_choice | 55.5 [51.7, 59.1] | 52.3 [48.5, 55.9] | 56.3 [52.5, 60.0] | -3.2 [-5.8, -0.7]   | 0.0182             |
| aqua          | 254  | forced_choice | 24.0 [18.9, 29.5] | 17.3 [12.6, 22.0] | 22.0 [16.9, 27.2] | -6.7 [-13.4, -0.4]  | 0.0547             |
| arc_challenge | 1172 | forced_choice | 50.9 [48.1, 53.8] | 40.4 [37.5, 43.2] | 50.6 [47.8, 53.4] | -10.5 [-13.3, -7.7] | 0.0001             |
| openbookqa    | 500  | forced_choice | 50.2 [45.8, 54.6] | 41.6 [37.4, 45.8] | 52.4 [48.0, 56.8] | -8.6 [-13.6, -3.6]  | 0.0005             |
| qasc          | 926  | forced_choice | 48.5 [45.2, 51.7] | 40.7 [37.6, 43.8] | 49.0 [45.8, 52.2] | -7.8 [-10.8, -4.8]  | 0.0001             |
| logiqa        | 651  | forced_choice | 32.7 [29.0, 36.4] | 26.9 [23.5, 30.3] | 33.2 [29.6, 36.9] | -5.8 [-10.8, -0.9]  | 0.0147             |
