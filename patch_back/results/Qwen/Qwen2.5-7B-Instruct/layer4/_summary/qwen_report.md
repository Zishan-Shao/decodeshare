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
| aqua          |         nan |    123 |        0.413386 |        0.287402 |           58 |                     100 |                        100 |                             98.2759 |                              15.5172 |                             24.1379 |                                     0 |
| arc_challenge |         nan |    123 |        0.905882 |        0.729412 |           50 |                     100 |                        100 |                            100      |                              42      |                             38      |                                     0 |
| commonsenseqa |         nan |    123 |        0.867188 |        0.804688 |           27 |                     100 |                        100 |                            100      |                              51.8519 |                             51.8519 |                                     0 |
| logiqa        |         nan |    123 |        0.472656 |        0.390625 |           53 |                     100 |                        100 |                             98.1132 |                              33.9623 |                             24.5283 |                                     0 |
| openbookqa    |         nan |    123 |        0.859375 |        0.664062 |           55 |                     100 |                        100 |                            100      |                              27.2727 |                             36.3636 |                                     0 |
| piqa          |         nan |    123 |        0.871094 |        0.742188 |           49 |                     100 |                        100 |                            100      |                              40.8163 |                             30.6122 |                                     0 |
| qasc          |         nan |    123 |        0.808594 |        0.695312 |           33 |                     100 |                        100 |                            100      |                              27.2727 |                             36.3636 |                                     0 |


## Open-answer patchback (openanswer)

| task      | eval_mode        |   seed |   base_acc_scan |   ablt_acc_scan |   flips_scan |   patched_self_rescued_pct |   control_time_shuffled_rescued_pct |   control_shared_randvec_rescued_pct |   control_rand_subspace_rescued_pct |   control_patch_nonshared_rescued_pct |
|:----------|:-----------------|-------:|----------------:|----------------:|-------------:|---------------------------:|------------------------------------:|-------------------------------------:|------------------------------------:|--------------------------------------:|
| gsm8k     | gen_math         |    123 |       0.0351562 |        0.03125  |            9 |                    22.2222 |                             11.1111 |                              22.2222 |                             33.3333 |                               44.4444 |
| gsm8k     | pair_logprob     |    123 |       0.800781  |        0.503906 |           83 |                    98.4375 |                             93.75   |                              12.5    |                             14.0625 |                                4.6875 |
| humaneval | gen_code_compile |    123 |     nan         |      nan        |            0 |                    16.6667 |                             16.6667 |                              33.3333 |                             16.6667 |                                0      |
| humaneval | pair_logprob     |    123 |       0.682927  |        0.707317 |            3 |                    66.6667 |                             66.6667 |                              66.6667 |                             33.3333 |                                0      |


## Flipset transfer patching (flipset)

| file                                                |   seed | task   |   base_acc_scan |   ablt_acc_scan |   flips_scan |   patched_self_rescued_pct |   patched_transfer_rescued_pct |
|:----------------------------------------------------|-------:|:-------|----------------:|----------------:|-------------:|---------------------------:|-------------------------------:|
| aqua_alpha_sweep_seed123.json                       |    123 | aqua   |        0.413386 |        0.287402 |           58 |                        nan |                       nan      |
| aqua_transfer_cross_mc_baselinecorrect_seed123.json |    123 | aqua   |        0.413386 |        0.287402 |           58 |                        100 |                        96.5517 |
| aqua_transfer_same_task_seed123.json                |    123 | aqua   |        0.413386 |        0.287402 |           58 |                        100 |                        96.5517 |
| aqua_alpha_sweep_seed456.json                       |    456 | aqua   |        0.401575 |        0.311024 |           63 |                        nan |                       nan      |


## Alpha sweep (flip-set)

| file                          |   seed |   alpha |   n |   flip_rate |   ablated_acc |   mean_delta_margin_vs_baseline |
|:------------------------------|-------:|--------:|----:|------------:|--------------:|--------------------------------:|
| aqua_alpha_sweep_seed123.json |    123 |    0    |  58 |    0        |      1        |                        0        |
| aqua_alpha_sweep_seed123.json |    123 |    0.5  |  58 |    0.103448 |      0.896552 |                       -0.291348 |
| aqua_alpha_sweep_seed123.json |    123 |    0.75 |  58 |    0.5      |      0.5      |                       -2.70827  |
| aqua_alpha_sweep_seed123.json |    123 |    1    |  58 |    1        |      0        |                       -3.81376  |
| aqua_alpha_sweep_seed456.json |    456 |    0    |  63 |    0        |      1        |                        0        |
| aqua_alpha_sweep_seed456.json |    456 |    0.5  |  63 |    0.047619 |      0.952381 |                       -0.358774 |
| aqua_alpha_sweep_seed456.json |    456 |    0.75 |  63 |    0.349206 |      0.650794 |                       -2.1342   |
| aqua_alpha_sweep_seed456.json |    456 |    1    |  63 |    1        |      0        |                       -4.09522  |
