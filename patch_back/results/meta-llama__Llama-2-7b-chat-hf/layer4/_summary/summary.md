| kind | task | eval_mode | file | base_acc_scan | ablt_acc_scan | flips_scan | patched_primary_method | patched_primary_rescued_pct | control_time_shuffled_rescued_pct | control_shared_randvec_rescued_pct | control_rand_subspace_rescued_pct | control_patch_nonshared_rescued_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| flipset | aqua |  | aqua_alpha_sweep_seed123.json | 0.209 | 0.197 | 20 |  |  |  |  |  |  |
| flipset | aqua |  | aqua_alpha_sweep_seed456.json | 0.209 | 0.181 | 21 |  |  |  |  |  |  |
| flipset | aqua |  | aqua_transfer_cross_mc_baselinecorrect_seed123.json | 0.209 | 0.197 | 20 | patched_self | 75.0 |  |  |  |  |
| flipset | aqua |  | aqua_transfer_same_task_seed123.json | 0.209 | 0.197 | 20 | patched_self | 75.0 |  |  |  |  |
| openanswer | gsm8k | gen_math | gsm8k_genmath.json | 0.039 | 0.035 | 8 | patched_self | 100.0 | 87.5 | 12.5 | 0.0 | 0.0 |
| openanswer | gsm8k | pair_logprob | gsm8k_pairlogprob.json | 0.625 | 0.629 | 11 | patched_self | 100.0 | 90.9 | 27.3 | 18.2 | 0.0 |
| openanswer | humaneval | gen_code_compile | humaneval_gencode_compile.json |  |  | 0 | patched_self | 41.7 | 33.3 | 16.7 | 13.9 | 0.0 |
| openanswer | humaneval | pair_logprob | humaneval_pairlogprob.json | 0.659 | 0.646 | 5 | patched_self | 60.0 | 60.0 | 40.0 | 0.0 | 0.0 |
| subspace_mc | aqua |  | aqua.json | 0.209 | 0.197 | 20 | patched_0 | 75.0 | 75.0 | 25.0 | 30.0 | 0.0 |
| subspace_mc | arc_challenge |  | arc_challenge.json | 0.514 | 0.506 | 20 | patched_0 | 85.0 | 90.0 | 25.0 | 5.0 | 0.0 |
| subspace_mc | commonsenseqa |  | commonsenseqa.json | 0.609 | 0.512 | 36 | patched_0 | 83.3 | 86.1 | 30.6 | 19.4 | 0.0 |
| subspace_mc | logiqa |  | logiqa.json | 0.309 | 0.297 | 36 | patched_0 | 83.3 | 86.1 | 8.3 | 11.1 | 0.0 |
| subspace_mc | openbookqa |  | openbookqa.json | 0.531 | 0.461 | 26 | patched_0 | 76.9 | 76.9 | 15.4 | 19.2 | 0.0 |
| subspace_mc | piqa |  | piqa.json | 0.676 | 0.598 | 46 | patched_0 | 91.3 | 93.5 | 17.4 | 13.0 | 0.0 |
| subspace_mc | qasc |  | qasc.json | 0.469 | 0.441 | 18 | patched_0 | 88.9 | 88.9 | 16.7 | 22.2 | 0.0 |
