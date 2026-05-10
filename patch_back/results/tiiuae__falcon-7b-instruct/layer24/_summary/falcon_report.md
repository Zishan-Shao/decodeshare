# Falcon subspace patching + flipset report

Generated from `summary.csv` and `alpha_sweep.csv`.

## Overview

- Runs: 15 total JSON summaries

- MC runs: 7; Open-answer runs: 4; Flipset runs: 4

## Key plots (PDF)

- `mc_patched0_rescue.pdf`
- `alpha_sweep_fliprate.pdf`

## Multiple-choice patchback (subspace_mc)

| task          |   eval_mode |   seed |   base_acc_scan |   ablt_acc_scan |   flips_scan |   patched_0_rescued_pct |   patched_full_rescued_pct |   control_time_shuffled_rescued_pct |   control_shared_randvec_rescued_pct |   control_rand_subspace_rescued_pct |   control_patch_nonshared_rescued_pct |
|:--------------|------------:|-------:|----------------:|----------------:|-------------:|------------------------:|---------------------------:|------------------------------------:|-------------------------------------:|------------------------------------:|--------------------------------------:|
| aqua          |         nan |    123 |        0.251969 |        0.212598 |           44 |                     100 |                        100 |                             95.4545 |                              15.9091 |                             9.09091 |                                     0 |
| arc_challenge |         nan |    123 |        0.261719 |        0.210938 |           32 |                     100 |                        100 |                             71.875  |                               9.375  |                             6.25    |                                     0 |
| commonsenseqa |         nan |    123 |        0.1875   |        0.226562 |           29 |                     100 |                        100 |                             89.6552 |                              17.2414 |                            17.2414  |                                     0 |
| logiqa        |         nan |    123 |        0.167969 |        0.265625 |            7 |                     100 |                        100 |                             57.1429 |                              28.5714 |                            28.5714  |                                     0 |
| openbookqa    |         nan |    123 |        0.246094 |        0.265625 |           25 |                     100 |                        100 |                             84      |                              32      |                            24       |                                     0 |
| piqa          |         nan |    123 |        0.539062 |        0.574219 |           38 |                     100 |                        100 |                             84.2105 |                              21.0526 |                            44.7368  |                                     0 |
| qasc          |         nan |    123 |        0.121094 |        0.121094 |           17 |                     100 |                        100 |                             82.3529 |                              23.5294 |                            11.7647  |                                     0 |

## Open-answer patchback (openanswer)

| task      | eval_mode        |   seed |   base_acc_scan |   ablt_acc_scan |   flips_scan |   patched_self_rescued_pct |   control_time_shuffled_rescued_pct |   control_shared_randvec_rescued_pct |   control_rand_subspace_rescued_pct |   control_patch_nonshared_rescued_pct |
|:----------|:-----------------|-------:|----------------:|----------------:|-------------:|---------------------------:|------------------------------------:|-------------------------------------:|------------------------------------:|--------------------------------------:|
| gsm8k     | gen_math         |    123 |       0.0351562 |       0.0234375 |            7 |                    14.2857 |                             28.5714 |                               0      |                                   0 |                               0       |
| gsm8k     | pair_logprob     |    123 |       0.636719  |       0.648438  |           17 |                   100      |                             70.5882 |                               0      |                                   0 |                               0       |
| humaneval | gen_code_compile |    123 |     nan         |     nan         |            0 |                    20      |                             18.1818 |                              10.9091 |                                   0 |                               1.81818 |
| humaneval | pair_logprob     |    123 |       0.640244  |       0.640244  |            1 |                   100      |                            100      |                               0      |                                   0 |                               0       |

## Flipset transfer patching (flipset)

| file                                                |   seed | task   |   base_acc_scan |   ablt_acc_scan |   flips_scan |   patched_self_rescued_pct |   patched_transfer_rescued_pct |
|:----------------------------------------------------|-------:|:-------|----------------:|----------------:|-------------:|---------------------------:|-------------------------------:|
| aqua_alpha_sweep_seed123.json                       |    123 | aqua   |        0.251969 |        0.212598 |           44 |                        nan |                       nan      |
| aqua_transfer_cross_mc_baselinecorrect_seed123.json |    123 | aqua   |        0.251969 |        0.212598 |           44 |                        100 |                        90.9091 |
| aqua_transfer_same_task_seed123.json                |    123 | aqua   |        0.251969 |        0.212598 |           44 |                        100 |                        93.1818 |
| aqua_alpha_sweep_seed456.json                       |    456 | aqua   |        0.251969 |        0.216535 |           21 |                        nan |                       nan      |
