| kind | task | eval_mode | file | base_acc_scan | ablt_acc_scan | flips_scan | patched_primary_method | patched_primary_rescued_pct | control_time_shuffled_rescued_pct | control_shared_randvec_rescued_pct | control_rand_subspace_rescued_pct | control_patch_nonshared_rescued_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| flipset | aqua |  | aqua_alpha_sweep_seed123.json | 0.252 | 0.213 | 44 |  |  |  |  |  |  |
| flipset | aqua |  | aqua_alpha_sweep_seed456.json | 0.252 | 0.217 | 21 |  |  |  |  |  |  |
| flipset | aqua |  | aqua_transfer_cross_mc_baselinecorrect_seed123.json | 0.252 | 0.213 | 44 | patched_self | 100.0 |  |  |  |  |
| flipset | aqua |  | aqua_transfer_same_task_seed123.json | 0.252 | 0.213 | 44 | patched_self | 100.0 |  |  |  |  |
| openanswer | gsm8k | gen_math | gsm8k_genmath.json | 0.035 | 0.023 | 7 | patched_self | 14.3 | 28.6 | 0.0 | 0.0 | 0.0 |
| openanswer | gsm8k | pair_logprob | gsm8k_pairlogprob.json | 0.637 | 0.648 | 17 | patched_self | 100.0 | 70.6 | 0.0 | 0.0 | 0.0 |
| openanswer | humaneval | gen_code_compile | humaneval_gencode_compile.json |  |  | 0 | patched_self | 20.0 | 18.2 | 10.9 | 0.0 | 1.8 |
| openanswer | humaneval | pair_logprob | humaneval_pairlogprob.json | 0.640 | 0.640 | 1 | patched_self | 100.0 | 100.0 | 0.0 | 0.0 | 0.0 |
| subspace_mc | aqua |  | aqua.json | 0.252 | 0.213 | 44 | patched_0 | 100.0 | 95.5 | 15.9 | 9.1 | 0.0 |
| subspace_mc | arc_challenge |  | arc_challenge.json | 0.262 | 0.211 | 32 | patched_0 | 100.0 | 71.9 | 9.4 | 6.2 | 0.0 |
| subspace_mc | commonsenseqa |  | commonsenseqa.json | 0.188 | 0.227 | 29 | patched_0 | 100.0 | 89.7 | 17.2 | 17.2 | 0.0 |
| subspace_mc | logiqa |  | logiqa.json | 0.168 | 0.266 | 7 | patched_0 | 100.0 | 57.1 | 28.6 | 28.6 | 0.0 |
| subspace_mc | openbookqa |  | openbookqa.json | 0.246 | 0.266 | 25 | patched_0 | 100.0 | 84.0 | 32.0 | 24.0 | 0.0 |
| subspace_mc | piqa |  | piqa.json | 0.539 | 0.574 | 38 | patched_0 | 100.0 | 84.2 | 21.1 | 44.7 | 0.0 |
| subspace_mc | qasc |  | qasc.json | 0.121 | 0.121 | 17 | patched_0 | 100.0 | 82.4 | 23.5 | 11.8 | 0.0 |
