| kind | task | eval_mode | file | base_acc_scan | ablt_acc_scan | flips_scan | patched_primary_method | patched_primary_rescued_pct | control_time_shuffled_rescued_pct | control_shared_randvec_rescued_pct | control_rand_subspace_rescued_pct | control_patch_nonshared_rescued_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| flipset | aqua |  | aqua_alpha_sweep_seed123.json | 0.209 | 0.228 | 33 |  |  |  |  |  |  |
| flipset | aqua |  | aqua_alpha_sweep_seed456.json | 0.209 | 0.205 | 38 |  |  |  |  |  |  |
| flipset | aqua |  | aqua_transfer_cross_mc_baselinecorrect_seed123.json | 0.209 | 0.228 | 33 | patched_self | 57.6 |  |  |  |  |
| flipset | aqua |  | aqua_transfer_same_task_seed123.json | 0.209 | 0.228 | 33 | patched_self | 57.6 |  |  |  |  |
| openanswer | gsm8k | gen_math | gsm8k_genmath.json | 0.039 | 0.000 | 10 | patched_self | 80.0 | 50.0 | 10.0 | 10.0 | 0.0 |
| openanswer | gsm8k | pair_logprob | gsm8k_pairlogprob.json | 0.625 | 0.598 | 20 | patched_self | 85.0 | 70.0 | 0.0 | 0.0 | 5.0 |
| openanswer | humaneval | gen_code_compile | humaneval_gencode_compile.json |  |  | 0 | patched_self | 20.0 | 25.7 | 5.7 | 2.9 | 0.0 |
| openanswer | humaneval | pair_logprob | humaneval_pairlogprob.json | 0.659 | 0.640 | 6 | patched_self | 83.3 | 33.3 | 0.0 | 0.0 | 0.0 |
| subspace_mc | aqua |  | aqua.json | 0.209 | 0.228 | 33 | patched_0 | 57.6 | 45.5 | 6.1 | 0.0 | 0.0 |
| subspace_mc | arc_challenge |  | arc_challenge.json | 0.514 | 0.471 | 36 | patched_0 | 69.4 | 61.1 | 8.3 | 2.8 | 0.0 |
| subspace_mc | commonsenseqa |  | commonsenseqa.json | 0.609 | 0.473 | 59 | patched_0 | 67.8 | 55.9 | 8.5 | 6.8 | 0.0 |
| subspace_mc | logiqa |  | logiqa.json | 0.309 | 0.312 | 51 | patched_0 | 56.9 | 56.9 | 3.9 | 3.9 | 0.0 |
| subspace_mc | openbookqa |  | openbookqa.json | 0.531 | 0.430 | 47 | patched_0 | 68.1 | 53.2 | 14.9 | 6.4 | 0.0 |
| subspace_mc | piqa |  | piqa.json | 0.676 | 0.656 | 20 | patched_0 | 80.0 | 80.0 | 15.0 | 25.0 | 0.0 |
| subspace_mc | qasc |  | qasc.json | 0.469 | 0.359 | 51 | patched_0 | 62.7 | 49.0 | 7.8 | 5.9 | 0.0 |
