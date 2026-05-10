| kind | task | eval_mode | file | base_acc_scan | ablt_acc_scan | flips_scan | patched_primary_method | patched_primary_rescued_pct | control_time_shuffled_rescued_pct | control_shared_randvec_rescued_pct | control_rand_subspace_rescued_pct | control_patch_nonshared_rescued_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| flipset | aqua |  | aqua_alpha_sweep_seed123.json | 0.413 | 0.276 | 64 |  |  |  |  |  |  |
| flipset | aqua |  | aqua_alpha_sweep_seed456.json | 0.402 | 0.291 | 65 |  |  |  |  |  |  |
| flipset | aqua |  | aqua_transfer_cross_mc_baselinecorrect_seed123.json | 0.413 | 0.276 | 64 | patched_self | 100.0 |  |  |  |  |
| flipset | aqua |  | aqua_transfer_same_task_seed123.json | 0.413 | 0.276 | 64 | patched_self | 100.0 |  |  |  |  |
| openanswer | gsm8k | gen_math | gsm8k_genmath.json | 0.035 | 0.059 | 7 | patched_self | 14.3 | 14.3 | 14.3 | 0.0 | 42.9 |
| openanswer | gsm8k | pair_logprob | gsm8k_pairlogprob.json | 0.801 | 0.625 | 53 | patched_self | 98.1 | 94.3 | 20.8 | 18.9 | 5.7 |
| openanswer | humaneval | gen_code_compile | humaneval_gencode_compile.json |  |  | 0 | patched_self | 10.5 | 10.5 | 15.8 | 26.3 | 0.0 |
| openanswer | humaneval | pair_logprob | humaneval_pairlogprob.json | 0.683 | 0.695 | 5 | patched_self | 80.0 | 80.0 | 0.0 | 20.0 | 0.0 |
| subspace_mc | aqua |  | aqua.json | 0.413 | 0.276 | 64 | patched_0 | 100.0 | 98.4 | 20.3 | 35.9 | 0.0 |
| subspace_mc | arc_challenge |  | arc_challenge.json | 0.906 | 0.514 | 108 | patched_0 | 100.0 | 100.0 | 36.1 | 60.2 | 0.0 |
| subspace_mc | commonsenseqa |  | commonsenseqa.json | 0.867 | 0.598 | 77 | patched_0 | 100.0 | 100.0 | 39.0 | 55.8 | 0.0 |
| subspace_mc | logiqa |  | logiqa.json | 0.473 | 0.395 | 45 | patched_0 | 100.0 | 100.0 | 35.6 | 40.0 | 0.0 |
| subspace_mc | openbookqa |  | openbookqa.json | 0.859 | 0.457 | 107 | patched_0 | 100.0 | 99.1 | 33.6 | 47.7 | 0.0 |
| subspace_mc | piqa |  | piqa.json | 0.871 | 0.773 | 37 | patched_0 | 100.0 | 100.0 | 56.8 | 48.6 | 0.0 |
| subspace_mc | qasc |  | qasc.json | 0.809 | 0.402 | 110 | patched_0 | 100.0 | 100.0 | 39.1 | 50.0 | 0.0 |
