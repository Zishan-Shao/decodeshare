| kind | task | eval_mode | file | base_acc_scan | ablt_acc_scan | flips_scan | patched_primary_method | patched_primary_rescued_pct | control_time_shuffled_rescued_pct | control_shared_randvec_rescued_pct | control_rand_subspace_rescued_pct | control_patch_nonshared_rescued_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| flipset | aqua |  | aqua_alpha_sweep_seed123.json | 0.252 | 0.248 | 3 |  |  |  |  |  |  |
| flipset | aqua |  | aqua_alpha_sweep_seed456.json | 0.252 | 0.248 | 3 |  |  |  |  |  |  |
| flipset | aqua |  | aqua_transfer_cross_mc_baselinecorrect_seed123.json | 0.252 | 0.248 | 3 | patched_self | 100.0 |  |  |  |  |
| flipset | aqua |  | aqua_transfer_same_task_seed123.json | 0.252 | 0.248 | 3 | patched_self | 100.0 |  |  |  |  |
| openanswer | gsm8k | gen_math | gsm8k_genmath.json | 0.035 | 0.027 | 9 | patched_self | 55.6 | 44.4 | 22.2 | 0.0 | 0.0 |
| openanswer | gsm8k | pair_logprob | gsm8k_pairlogprob.json | 0.637 | 0.539 | 42 | patched_self | 100.0 | 97.6 | 9.5 | 2.4 | 0.0 |
| openanswer | humaneval | gen_code_compile | humaneval_gencode_compile.json |  |  | 0 | patched_self | 8.3 | 4.2 | 6.2 | 2.1 | 2.1 |
| openanswer | humaneval | pair_logprob | humaneval_pairlogprob.json | 0.640 | 0.610 | 8 | patched_self | 100.0 | 62.5 | 0.0 | 0.0 | 0.0 |
| subspace_mc | aqua |  | aqua.json | 0.252 | 0.248 | 3 | patched_0 | 100.0 | 100.0 | 0.0 | 0.0 | 0.0 |
| subspace_mc | arc_challenge |  | arc_challenge.json | 0.262 | 0.254 | 20 | patched_0 | 100.0 | 100.0 | 15.0 | 0.0 | 0.0 |
| subspace_mc | commonsenseqa |  | commonsenseqa.json | 0.188 | 0.199 | 8 | patched_0 | 100.0 | 100.0 | 0.0 | 0.0 | 0.0 |
| subspace_mc | logiqa |  | logiqa.json | 0.168 | 0.164 | 3 | patched_0 | 100.0 | 100.0 | 0.0 | 0.0 | 0.0 |
| subspace_mc | openbookqa |  | openbookqa.json | 0.246 | 0.281 | 13 | patched_0 | 100.0 | 100.0 | 15.4 | 0.0 | 0.0 |
| subspace_mc | piqa |  | piqa.json | 0.539 | 0.570 | 41 | patched_0 | 100.0 | 100.0 | 9.8 | 0.0 | 0.0 |
| subspace_mc | qasc |  | qasc.json | 0.121 | 0.125 | 3 | patched_0 | 100.0 | 100.0 | 0.0 | 0.0 | 0.0 |
