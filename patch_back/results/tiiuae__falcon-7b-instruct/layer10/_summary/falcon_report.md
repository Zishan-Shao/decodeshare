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
| aqua          |         nan |    123 |        0.251969 |        0.248031 |            3 |                     100 |                        100 |                             66.6667 |                                    0 |                             0       |                                     0 |
| arc_challenge |         nan |    123 |        0.261719 |        0.257812 |           20 |                     100 |                        100 |                             90      |                                    0 |                             0       |                                     0 |
| commonsenseqa |         nan |    123 |        0.1875   |        0.199219 |            8 |                     100 |                        100 |                            100      |                                    0 |                             0       |                                     0 |
| logiqa        |         nan |    123 |        0.167969 |        0.164062 |            3 |                     100 |                        100 |                            100      |                                    0 |                             0       |                                     0 |
| openbookqa    |         nan |    123 |        0.246094 |        0.273438 |           14 |                     100 |                        100 |                            100      |                                    0 |                             0       |                                     0 |
| piqa          |         nan |    123 |        0.539062 |        0.554688 |           46 |                     100 |                        100 |                             95.6522 |                                    0 |                             2.17391 |                                     0 |
| qasc          |         nan |    123 |        0.121094 |        0.128906 |            3 |                     100 |                        100 |                            100      |                                    0 |                             0       |                                     0 |

## Open-answer patchback (openanswer)

| task      | eval_mode        |   seed |   base_acc_scan |   ablt_acc_scan |   flips_scan |   patched_self_rescued_pct |   control_time_shuffled_rescued_pct |   control_shared_randvec_rescued_pct |   control_rand_subspace_rescued_pct |   control_patch_nonshared_rescued_pct |
|:----------|:-----------------|-------:|----------------:|----------------:|-------------:|---------------------------:|------------------------------------:|-------------------------------------:|------------------------------------:|--------------------------------------:|
| gsm8k     | gen_math         |    123 |       0.0351562 |       0.0117188 |            8 |                    75      |                            37.5     |                              0       |                             0       |                               0       |
| gsm8k     | pair_logprob     |    123 |       0.636719  |       0.511719  |           40 |                   100      |                            95       |                              5       |                             7.5     |                               0       |
| humaneval | gen_code_compile |    123 |     nan         |     nan         |            0 |                    11.5385 |                             9.61538 |                              7.69231 |                             1.92308 |                               1.92308 |
| humaneval | pair_logprob     |    123 |       0.640244  |       0.609756  |            5 |                   100      |                           100       |                              0       |                             0       |                               0       |

## Flipset transfer patching (flipset)

| file                                                |   seed | task   |   base_acc_scan |   ablt_acc_scan |   flips_scan |   patched_self_rescued_pct |   patched_transfer_rescued_pct |
|:----------------------------------------------------|-------:|:-------|----------------:|----------------:|-------------:|---------------------------:|-------------------------------:|
| aqua_alpha_sweep_seed123.json                       |    123 | aqua   |        0.251969 |        0.248031 |            3 |                        nan |                       nan      |
| aqua_transfer_cross_mc_baselinecorrect_seed123.json |    123 | aqua   |        0.251969 |        0.248031 |            3 |                        100 |                        66.6667 |
| aqua_transfer_same_task_seed123.json                |    123 | aqua   |        0.251969 |        0.248031 |            3 |                        100 |                       100      |
| aqua_alpha_sweep_seed456.json                       |    456 | aqua   |        0.251969 |        0.248031 |            3 |                        nan |                       nan      |
