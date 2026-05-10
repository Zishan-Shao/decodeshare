| kind | task | eval_mode | file | base_acc_scan | ablt_acc_scan | flips_scan | patched_primary_method | patched_primary_rescued_pct | control_time_shuffled_rescued_pct | control_shared_randvec_rescued_pct | control_rand_subspace_rescued_pct | control_patch_nonshared_rescued_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| flipset | aqua |  | aqua_flipset_alpha_sweep_seed123.json | 0.209 | 0.220 | 42 |  |  |  |  |  |  |
| flipset | aqua |  | aqua_flipset_alpha_sweep_seed456.json | 0.209 | 0.181 | 43 |  |  |  |  |  |  |
| flipset | aqua |  | aqua_flipset_transfer_cross_generative_seed123.json | 0.209 | 0.220 | 42 | patched_self | 73.8 |  |  |  |  |
| flipset | aqua |  | aqua_flipset_transfer_cross_mc_baselinecorrect_seed123.json | 0.209 | 0.220 | 42 | patched_self | 73.8 |  |  |  |  |
| flipset | aqua |  | aqua_flipset_transfer_cross_mc_baselinecorrect_seed456.json | 0.209 | 0.181 | 43 | patched_self | 79.1 |  |  |  |  |
| flipset | aqua |  | aqua_flipset_transfer_same_task_seed123.json | 0.209 | 0.220 | 42 | patched_self | 73.8 |  |  |  |  |
| flipset | aqua |  | flipset_sweep_transfer.json | 0.209 | 0.220 | 42 | patched_self | 73.8 |  |  |  |  |
| openanswer | gsm8k | gen_math | gsm8k_genmath.json | 0.039 | 0.031 | 9 | patched_self | 88.9 | 77.8 | 0.0 | 0.0 | 0.0 |
| openanswer | gsm8k | pair_logprob | gsm8k_pairlogprob.json | 0.625 | 0.594 | 31 | patched_self | 35.5 | 35.5 | 3.2 | 0.0 | 0.0 |
| openanswer | humaneval | gen_code_compile | humaneval_gencode_compile.json |  |  | 0 | patched_self | 21.1 | 21.1 | 26.3 | 18.4 | 5.3 |
| openanswer | humaneval | pair_logprob | humaneval_pairlogprob.json | 0.659 | 0.640 | 8 | patched_self | 75.0 | 62.5 | 0.0 | 12.5 | 12.5 |
| subspace_mc | aqua |  | aqua.json | 0.209 | 0.220 | 42 | patched_0 | 73.8 | 76.2 | 4.8 | 4.8 | 0.0 |
| subspace_mc | arc_challenge |  | arc_challenge.json | 0.514 | 0.416 | 58 | patched_0 | 89.7 | 89.7 | 10.3 | 5.2 | 0.0 |
| subspace_mc | commonsenseqa |  | commonsenseqa.json | 0.609 | 0.387 | 75 | patched_0 | 90.7 | 89.3 | 13.3 | 8.0 | 0.0 |
| subspace_mc | logiqa |  | logiqa.json | 0.309 | 0.281 | 62 | patched_0 | 80.6 | 82.3 | 3.2 | 3.2 | 0.0 |
| subspace_mc | openbookqa |  | openbookqa.json | 0.531 | 0.387 | 66 | patched_0 | 86.4 | 86.4 | 10.6 | 4.5 | 0.0 |
| subspace_mc | piqa |  | piqa.json | 0.676 | 0.535 | 87 | patched_0 | 83.9 | 82.8 | 1.1 | 2.3 | 0.0 |
| subspace_mc | qasc |  | qasc.json | 0.469 | 0.293 | 59 | patched_0 | 91.5 | 89.8 | 10.2 | 10.2 | 0.0 |
