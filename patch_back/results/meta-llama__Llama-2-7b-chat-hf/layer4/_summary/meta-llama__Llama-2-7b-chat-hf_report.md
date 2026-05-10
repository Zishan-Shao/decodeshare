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
| aqua          |         nan |    123 |        0.208661 |        0.19685  |           20 |                 75      |                        100 |                             75      |                             25       |                             30      |                                     0 |
| arc_challenge |         nan |    123 |        0.513725 |        0.505882 |           20 |                 85      |                        100 |                             90      |                             25       |                              5      |                                     0 |
| commonsenseqa |         nan |    123 |        0.609375 |        0.511719 |           36 |                 83.3333 |                        100 |                             86.1111 |                             30.5556  |                             19.4444 |                                     0 |
| logiqa        |         nan |    123 |        0.308594 |        0.296875 |           36 |                 83.3333 |                        100 |                             86.1111 |                              8.33333 |                             11.1111 |                                     0 |
| openbookqa    |         nan |    123 |        0.53125  |        0.460938 |           26 |                 76.9231 |                        100 |                             76.9231 |                             15.3846  |                             19.2308 |                                     0 |
| piqa          |         nan |    123 |        0.675781 |        0.597656 |           46 |                 91.3043 |                        100 |                             93.4783 |                             17.3913  |                             13.0435 |                                     0 |
| qasc          |         nan |    123 |        0.46875  |        0.441406 |           18 |                 88.8889 |                        100 |                             88.8889 |                             16.6667  |                             22.2222 |                                     0 |


## Open-answer patchback (openanswer)

| task      | eval_mode        |   seed |   base_acc_scan |   ablt_acc_scan |   flips_scan |   patched_self_rescued_pct |   control_time_shuffled_rescued_pct |   control_shared_randvec_rescued_pct |   control_rand_subspace_rescued_pct |   control_patch_nonshared_rescued_pct |
|:----------|:-----------------|-------:|----------------:|----------------:|-------------:|---------------------------:|------------------------------------:|-------------------------------------:|------------------------------------:|--------------------------------------:|
| gsm8k     | gen_math         |    123 |       0.0390625 |       0.0351562 |            8 |                   100      |                             87.5    |                              12.5    |                              0      |                                     0 |
| gsm8k     | pair_logprob     |    123 |       0.625     |       0.628906  |           11 |                   100      |                             90.9091 |                              27.2727 |                             18.1818 |                                     0 |
| humaneval | gen_code_compile |    123 |     nan         |     nan         |            0 |                    41.6667 |                             33.3333 |                              16.6667 |                             13.8889 |                                     0 |
| humaneval | pair_logprob     |    123 |       0.658537  |       0.646341  |            5 |                    60      |                             60      |                              40      |                              0      |                                     0 |


## Flipset transfer patching (flipset)

| file                                                |   seed | task   |   base_acc_scan |   ablt_acc_scan |   flips_scan |   patched_self_rescued_pct |   patched_transfer_rescued_pct |
|:----------------------------------------------------|-------:|:-------|----------------:|----------------:|-------------:|---------------------------:|-------------------------------:|
| aqua_alpha_sweep_seed123.json                       |    123 | aqua   |        0.208661 |        0.19685  |           20 |                        nan |                            nan |
| aqua_transfer_cross_mc_baselinecorrect_seed123.json |    123 | aqua   |        0.208661 |        0.19685  |           20 |                         75 |                             75 |
| aqua_transfer_same_task_seed123.json                |    123 | aqua   |        0.208661 |        0.19685  |           20 |                         75 |                             75 |
| aqua_alpha_sweep_seed456.json                       |    456 | aqua   |        0.208661 |        0.181102 |           21 |                        nan |                            nan |


## Alpha sweep (flip-set)

| file                          |   seed |   alpha |   n |   flip_rate |   ablated_acc |   mean_delta_margin_vs_baseline |
|:------------------------------|-------:|--------:|----:|------------:|--------------:|--------------------------------:|
| aqua_alpha_sweep_seed123.json |    123 |    0    |  20 |   0         |      1        |                        0        |
| aqua_alpha_sweep_seed123.json |    123 |    0.5  |  20 |   0.15      |      0.85     |                       -0.164675 |
| aqua_alpha_sweep_seed123.json |    123 |    0.75 |  20 |   0.4       |      0.6      |                       -0.313424 |
| aqua_alpha_sweep_seed123.json |    123 |    1    |  20 |   1         |      0        |                       -0.909532 |
| aqua_alpha_sweep_seed456.json |    456 |    0    |  21 |   0         |      1        |                        0        |
| aqua_alpha_sweep_seed456.json |    456 |    0.5  |  21 |   0.0952381 |      0.904762 |                       -0.174735 |
| aqua_alpha_sweep_seed456.json |    456 |    0.75 |  21 |   0.428571  |      0.571429 |                       -0.478114 |
| aqua_alpha_sweep_seed456.json |    456 |    1    |  21 |   1         |      0        |                       -1.38985  |
