| kind | task | eval_mode | file | base_acc_scan | ablt_acc_scan | flips_scan | patched_primary_method | patched_primary_rescued_pct | control_time_shuffled_rescued_pct | control_shared_randvec_rescued_pct | control_rand_subspace_rescued_pct | control_patch_nonshared_rescued_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| flipset | aqua |  | aqua_alpha_sweep_seed123.json | 0.413 | 0.287 | 58 |  |  |  |  |  |  |
| flipset | aqua |  | aqua_alpha_sweep_seed456.json | 0.402 | 0.311 | 63 |  |  |  |  |  |  |
| flipset | aqua |  | aqua_transfer_cross_mc_baselinecorrect_seed123.json | 0.413 | 0.287 | 58 | patched_self | 100.0 |  |  |  |  |
| flipset | aqua |  | aqua_transfer_same_task_seed123.json | 0.413 | 0.287 | 58 | patched_self | 100.0 |  |  |  |  |
| openanswer | gsm8k | gen_math | gsm8k_genmath.json | 0.035 | 0.031 | 9 | patched_self | 22.2 | 11.1 | 22.2 | 33.3 | 44.4 |
| openanswer | gsm8k | pair_logprob | gsm8k_pairlogprob.json | 0.801 | 0.504 | 83 | patched_self | 98.4 | 93.8 | 12.5 | 14.1 | 4.7 |
| openanswer | humaneval | gen_code_compile | humaneval_gencode_compile.json |  |  | 0 | patched_self | 16.7 | 16.7 | 33.3 | 16.7 | 0.0 |
| openanswer | humaneval | pair_logprob | humaneval_pairlogprob.json | 0.683 | 0.707 | 3 | patched_self | 66.7 | 66.7 | 66.7 | 33.3 | 0.0 |
| subspace_mc | aqua |  | aqua.json | 0.413 | 0.287 | 58 | patched_0 | 100.0 | 98.3 | 15.5 | 24.1 | 0.0 |
| subspace_mc | arc_challenge |  | arc_challenge.json | 0.906 | 0.729 | 50 | patched_0 | 100.0 | 100.0 | 42.0 | 38.0 | 0.0 |
| subspace_mc | commonsenseqa |  | commonsenseqa.json | 0.867 | 0.805 | 27 | patched_0 | 100.0 | 100.0 | 51.9 | 51.9 | 0.0 |
| subspace_mc | logiqa |  | logiqa.json | 0.473 | 0.391 | 53 | patched_0 | 100.0 | 98.1 | 34.0 | 24.5 | 0.0 |
| subspace_mc | openbookqa |  | openbookqa.json | 0.859 | 0.664 | 55 | patched_0 | 100.0 | 100.0 | 27.3 | 36.4 | 0.0 |
| subspace_mc | piqa |  | piqa.json | 0.871 | 0.742 | 49 | patched_0 | 100.0 | 100.0 | 40.8 | 30.6 | 0.0 |
| subspace_mc | qasc |  | qasc.json | 0.809 | 0.695 | 33 | patched_0 | 100.0 | 100.0 | 27.3 | 36.4 | 0.0 |
