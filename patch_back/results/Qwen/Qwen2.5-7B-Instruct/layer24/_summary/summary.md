| kind | task | eval_mode | file | base_acc_scan | ablt_acc_scan | flips_scan | patched_primary_method | patched_primary_rescued_pct | control_time_shuffled_rescued_pct | control_shared_randvec_rescued_pct | control_rand_subspace_rescued_pct | control_patch_nonshared_rescued_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| flipset | aqua |  | aqua_alpha_sweep_seed123.json | 0.413 | 0.236 | 61 |  |  |  |  |  |  |
| flipset | aqua |  | aqua_alpha_sweep_seed456.json | 0.402 | 0.299 | 45 |  |  |  |  |  |  |
| flipset | aqua |  | aqua_transfer_cross_mc_baselinecorrect_seed123.json | 0.413 | 0.236 | 61 | patched_self | 100.0 |  |  |  |  |
| flipset | aqua |  | aqua_transfer_same_task_seed123.json | 0.413 | 0.236 | 61 | patched_self | 100.0 |  |  |  |  |
| openanswer | gsm8k | gen_math | gsm8k_genmath.json | 0.035 | 0.039 | 8 | patched_self | 0.0 | 12.5 | 12.5 | 0.0 | 0.0 |
| openanswer | gsm8k | pair_logprob | gsm8k_pairlogprob.json | 0.801 | 0.762 | 15 | patched_self | 93.3 | 73.3 | 20.0 | 13.3 | 6.7 |
| openanswer | humaneval | gen_code_compile | humaneval_gencode_compile.json |  |  | 0 | patched_self | 5.3 | 5.3 | 5.3 | 0.0 | 0.0 |
| openanswer | humaneval | pair_logprob | humaneval_pairlogprob.json | 0.683 | 0.665 | 12 | patched_self | 83.3 | 50.0 | 0.0 | 0.0 | 0.0 |
| subspace_mc | aqua |  | aqua.json | 0.413 | 0.236 | 61 | patched_0 | 100.0 | 27.9 | 32.8 | 31.1 | 0.0 |
| subspace_mc | arc_challenge |  | arc_challenge.json | 0.906 | 0.608 | 83 | patched_0 | 100.0 | 59.0 | 27.7 | 33.7 | 0.0 |
| subspace_mc | commonsenseqa |  | commonsenseqa.json | 0.867 | 0.645 | 64 | patched_0 | 100.0 | 46.9 | 35.9 | 26.6 | 0.0 |
| subspace_mc | logiqa |  | logiqa.json | 0.473 | 0.418 | 34 | patched_0 | 100.0 | 47.1 | 29.4 | 38.2 | 0.0 |
| subspace_mc | openbookqa |  | openbookqa.json | 0.859 | 0.613 | 69 | patched_0 | 100.0 | 44.9 | 31.9 | 40.6 | 0.0 |
| subspace_mc | piqa |  | piqa.json | 0.871 | 0.707 | 53 | patched_0 | 100.0 | 92.5 | 32.1 | 26.4 | 0.0 |
| subspace_mc | qasc |  | qasc.json | 0.809 | 0.492 | 89 | patched_0 | 100.0 | 51.7 | 61.8 | 66.3 | 0.0 |
