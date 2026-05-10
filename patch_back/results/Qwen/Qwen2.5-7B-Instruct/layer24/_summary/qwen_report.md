# Qwen subspace patching + flipset report

Generated from `summary.csv` and `alpha_sweep.csv`.

## Overview

- Runs: 15 total JSON summaries

- MC runs: 7; Open-answer runs: 4; Flipset runs: 4

## Key plots (PDF, dpi=300)

- `mc_patched0_rescue.pdf`
- `mc_controls_gap.pdf`
- `alpha_sweep_fliprate.pdf`
- `alpha_sweep_deltam.pdf`
- `openanswer_patchedself_rescue.pdf`


## Multiple-choice patchback (subspace_mc)

| task          |   eval_mode |   seed |   base_acc_scan |   ablt_acc_scan |   flips_scan |   patched_0_rescued_pct |   patched_full_rescued_pct |   control_time_shuffled_rescued_pct |   control_shared_randvec_rescued_pct |   control_rand_subspace_rescued_pct |   control_patch_nonshared_rescued_pct |
|:--------------|------------:|-------:|----------------:|----------------:|-------------:|------------------------:|---------------------------:|------------------------------------:|-------------------------------------:|------------------------------------:|--------------------------------------:|
| aqua          |         nan |    123 |        0.413386 |        0.23622  |           61 |                     100 |                        100 |                             27.8689 |                              32.7869 |                             31.1475 |                                     0 |
| arc_challenge |         nan |    123 |        0.905882 |        0.607843 |           83 |                     100 |                        100 |                             59.0361 |                              27.7108 |                             33.7349 |                                     0 |
| commonsenseqa |         nan |    123 |        0.867188 |        0.644531 |           64 |                     100 |                        100 |                             46.875  |                              35.9375 |                             26.5625 |                                     0 |
| logiqa        |         nan |    123 |        0.472656 |        0.417969 |           34 |                     100 |                        100 |                             47.0588 |                              29.4118 |                             38.2353 |                                     0 |
| openbookqa    |         nan |    123 |        0.859375 |        0.613281 |           69 |                     100 |                        100 |                             44.9275 |                              31.8841 |                             40.5797 |                                     0 |
| piqa          |         nan |    123 |        0.871094 |        0.707031 |           53 |                     100 |                        100 |                             92.4528 |                              32.0755 |                             26.4151 |                                     0 |
| qasc          |         nan |    123 |        0.808594 |        0.492188 |           89 |                     100 |                        100 |                             51.6854 |                              61.7978 |                             66.2921 |                                     0 |


## Open-answer patchback (openanswer)

| task      | eval_mode        |   seed |   base_acc_scan |   ablt_acc_scan |   flips_scan |   patched_self_rescued_pct |   control_time_shuffled_rescued_pct |   control_shared_randvec_rescued_pct |   control_rand_subspace_rescued_pct |   control_patch_nonshared_rescued_pct |
|:----------|:-----------------|-------:|----------------:|----------------:|-------------:|---------------------------:|------------------------------------:|-------------------------------------:|------------------------------------:|--------------------------------------:|
| gsm8k     | gen_math         |    123 |       0.0351562 |       0.0390625 |            8 |                    0       |                            12.5     |                             12.5     |                              0      |                               0       |
| gsm8k     | pair_logprob     |    123 |       0.800781  |       0.761719  |           15 |                   93.3333  |                            73.3333  |                             20       |                             13.3333 |                               6.66667 |
| humaneval | gen_code_compile |    123 |     nan         |     nan         |            0 |                    5.26316 |                             5.26316 |                              5.26316 |                              0      |                               0       |
| humaneval | pair_logprob     |    123 |       0.682927  |       0.664634  |           12 |                   83.3333  |                            50       |                              0       |                              0      |                               0       |


## Flipset transfer patching (flipset)

| file                                                |   seed | task   |   base_acc_scan |   ablt_acc_scan |   flips_scan |   patched_self_rescued_pct |   patched_transfer_rescued_pct |
|:----------------------------------------------------|-------:|:-------|----------------:|----------------:|-------------:|---------------------------:|-------------------------------:|
| aqua_alpha_sweep_seed123.json                       |    123 | aqua   |        0.413386 |        0.23622  |           61 |                        nan |                       nan      |
| aqua_transfer_cross_mc_baselinecorrect_seed123.json |    123 | aqua   |        0.413386 |        0.23622  |           61 |                        100 |                        18.0328 |
| aqua_transfer_same_task_seed123.json                |    123 | aqua   |        0.413386 |        0.23622  |           61 |                        100 |                        31.1475 |
| aqua_alpha_sweep_seed456.json                       |    456 | aqua   |        0.401575 |        0.299213 |           45 |                        nan |                       nan      |


## Alpha sweep (flip-set)

| file                          |   seed |   alpha |   n |   flip_rate |   ablated_acc |   mean_delta_margin_vs_baseline |
|:------------------------------|-------:|--------:|----:|------------:|--------------:|--------------------------------:|
| aqua_alpha_sweep_seed123.json |    123 |    0    |  61 |    0        |      1        |                         0       |
| aqua_alpha_sweep_seed123.json |    123 |    0.5  |  61 |    0.163934 |      0.836066 |                        -1.14309 |
| aqua_alpha_sweep_seed123.json |    123 |    0.75 |  61 |    0.409836 |      0.590164 |                        -2.33868 |
| aqua_alpha_sweep_seed123.json |    123 |    1    |  61 |    1        |      0        |                        -4.62629 |
| aqua_alpha_sweep_seed456.json |    456 |    0    |  45 |    0        |      1        |                         0       |
| aqua_alpha_sweep_seed456.json |    456 |    0.5  |  45 |    0.222222 |      0.777778 |                        -1.00896 |
| aqua_alpha_sweep_seed456.json |    456 |    0.75 |  45 |    0.444444 |      0.555556 |                        -1.98602 |
| aqua_alpha_sweep_seed456.json |    456 |    1    |  45 |    1        |      0        |                        -3.93168 |
