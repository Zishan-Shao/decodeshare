# meta-llama/Llama-2-7b-chat-hf subspace patching + flipset report

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
| aqua          |         nan |    123 |        0.208661 |        0.228346 |           33 |                 57.5758 |                        100 |                             45.4545 |                              6.06061 |                             0       |                                     0 |
| arc_challenge |         nan |    123 |        0.513725 |        0.470588 |           36 |                 69.4444 |                        100 |                             61.1111 |                              8.33333 |                             2.77778 |                                     0 |
| commonsenseqa |         nan |    123 |        0.609375 |        0.472656 |           59 |                 67.7966 |                        100 |                             55.9322 |                              8.47458 |                             6.77966 |                                     0 |
| logiqa        |         nan |    123 |        0.308594 |        0.3125   |           51 |                 56.8627 |                        100 |                             56.8627 |                              3.92157 |                             3.92157 |                                     0 |
| openbookqa    |         nan |    123 |        0.53125  |        0.429688 |           47 |                 68.0851 |                        100 |                             53.1915 |                             14.8936  |                             6.38298 |                                     0 |
| piqa          |         nan |    123 |        0.675781 |        0.65625  |           20 |                 80      |                        100 |                             80      |                             15       |                            25       |                                     0 |
| qasc          |         nan |    123 |        0.46875  |        0.359375 |           51 |                 62.7451 |                        100 |                             49.0196 |                              7.84314 |                             5.88235 |                                     0 |


## Open-answer patchback (openanswer)

| task      | eval_mode        |   seed |   base_acc_scan |   ablt_acc_scan |   flips_scan |   patched_self_rescued_pct |   control_time_shuffled_rescued_pct |   control_shared_randvec_rescued_pct |   control_rand_subspace_rescued_pct |   control_patch_nonshared_rescued_pct |
|:----------|:-----------------|-------:|----------------:|----------------:|-------------:|---------------------------:|------------------------------------:|-------------------------------------:|------------------------------------:|--------------------------------------:|
| gsm8k     | gen_math         |    123 |       0.0390625 |        0        |           10 |                    80      |                             50      |                             10       |                            10       |                                     0 |
| gsm8k     | pair_logprob     |    123 |       0.625     |        0.597656 |           20 |                    85      |                             70      |                              0       |                             0       |                                     5 |
| humaneval | gen_code_compile |    123 |     nan         |      nan        |            0 |                    20      |                             25.7143 |                              5.71429 |                             2.85714 |                                     0 |
| humaneval | pair_logprob     |    123 |       0.658537  |        0.640244 |            6 |                    83.3333 |                             33.3333 |                              0       |                             0       |                                     0 |


## Flipset transfer patching (flipset)

| file                                                |   seed | task   |   base_acc_scan |   ablt_acc_scan |   flips_scan |   patched_self_rescued_pct |   patched_transfer_rescued_pct |
|:----------------------------------------------------|-------:|:-------|----------------:|----------------:|-------------:|---------------------------:|-------------------------------:|
| aqua_alpha_sweep_seed123.json                       |    123 | aqua   |        0.208661 |        0.228346 |           33 |                   nan      |                       nan      |
| aqua_transfer_cross_mc_baselinecorrect_seed123.json |    123 | aqua   |        0.208661 |        0.228346 |           33 |                    57.5758 |                        30.303  |
| aqua_transfer_same_task_seed123.json                |    123 | aqua   |        0.208661 |        0.228346 |           33 |                    57.5758 |                        42.4242 |
| aqua_alpha_sweep_seed456.json                       |    456 | aqua   |        0.208661 |        0.204724 |           38 |                   nan      |                       nan      |


## Alpha sweep (flip-set)

| file                          |   seed |   alpha |   n |   flip_rate |   ablated_acc |   mean_delta_margin_vs_baseline |
|:------------------------------|-------:|--------:|----:|------------:|--------------:|--------------------------------:|
| aqua_alpha_sweep_seed123.json |    123 |    0    |  33 |    0        |      1        |                        0        |
| aqua_alpha_sweep_seed123.json |    123 |    0.5  |  33 |    0.515152 |      0.484848 |                       -0.628207 |
| aqua_alpha_sweep_seed123.json |    123 |    0.75 |  33 |    0.69697  |      0.30303  |                       -1.54052  |
| aqua_alpha_sweep_seed123.json |    123 |    1    |  33 |    1        |      0        |                       -2.93819  |
| aqua_alpha_sweep_seed456.json |    456 |    0    |  38 |    0        |      1        |                        0        |
| aqua_alpha_sweep_seed456.json |    456 |    0.5  |  38 |    0.631579 |      0.368421 |                       -1.12215  |
| aqua_alpha_sweep_seed456.json |    456 |    0.75 |  38 |    0.868421 |      0.131579 |                       -2.39807  |
| aqua_alpha_sweep_seed456.json |    456 |    1    |  38 |    1        |      0        |                       -3.72873  |
