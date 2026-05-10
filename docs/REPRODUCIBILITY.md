# Reproducibility Map

This file maps the main DecodeShare claims to the code and compact outputs
tracked on the `Halo` branch.

## H1: Shared Decode-Time Workspace

Purpose: test whether a compact subspace is consistently shared across tasks in
KV-cached decode-time hidden states.

Code:

- `Hype1/collect_decode_acts.py`
- `Hype1/prove_sharedness_decode_fair.py`
- `Hype1/prove_sharedness_decode_full.py`
- `src/prove_sharedness_decode_fair.py`
- `src/prove_sharedness_decode_full.py`

Scripts:

- `Hype1/run_00_collect_acts.sh`
- `Hype1/run_01_exp1_within_vs_mixed.sh`
- `Hype1/run_02_exp2_convergence.sh`
- `run_exists.sh`

Compact outputs:

- `Hype1/results/full_benchmark/H1_full_benchmark_summary.md`
- `Hype1/results/full_benchmark/H1_full_benchmark_summary.csv`
- `Hype1/results/exp1/`
- `Hype1/results/exp2/`
- `Hype1/results/exp2.5/`
- `Hype1/results/exp2.75/`
- `Hype1/results/exp3/`

## H2: Decode-Time Causal Role

Purpose: remove the discovered shared subspace during decode and compare against
matched controls.

Code:

- `src/disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py`
- `src/disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8_reasoning.py`
- `src/disturb_energy_matched_sharedness_kmatch.py`
- `reasoning/disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py`
- `reasoning/disturb_energy_matched_sharedness_kmatch.py`

Scripts:

- `run_disturb_cot_loto8.sh`
- `run_disturb_cot_loto8_fc_reason.sh`
- `run_disturb_cot_loto8_main.sh`
- `run_disturb_cot_loto8_main_qwen.sh`
- `run_disturb_cot_loto8_main_falcon.sh`

Compact outputs:

- `results/disturb_cot_reasoning/energy_balance_loto8_reasoning_fc.md`
- `results/disturb_cot_reasoning/energy_balance_loto8_reasoning_fc_eval2048.md`
- `reasoning/energy_balance_loto8_reasoning_fc_eval2048.md`

## H2 Patchback

Purpose: test whether patching shared decode directions rescues flip-set
failures while nonshared controls do not.

Code:

- `patch_back/subspace_patching_transfer.py`
- `patch_back/subspace_patching_transfer_enhanced.py`
- `patch_back/openanswer_subspace_patching.py`
- `patch_back/openanswer_subspace_patching_qwen.py`
- `patch_back/openanswer_subspace_patching_falcon.py`
- `patch_back/summarize_patching_jsons.py`

Scripts:

- `patch_back/run_decodeshare_suite.sh`
- `patch_back/run_all_flip_experiments.sh`
- `patch_back/run_qwen_suite_and_report.sh`
- `patch_back/run_falcon_suite_and_report.sh`
- `patch_back/run_llama_suite_and_report.sh`

Compact outputs:

- `patch_back/paper/patchback_tables_all_models_all_layers.tex`
- `patch_back/paper/patchback_discussion_all_models_all_layers.tex`
- `patch_back/results/summary/`

## H3: Prefill-vs-Decode Mismatch

Purpose: compare prefill-estimated and decode-estimated bases, then evaluate
decode-time interventions using the aligned decode protocol.

Code:

- `src/prefill_vs_decode_alignment_experiment_generation.py`
- `src/prefill_vs_decode_alignment_experiment_reasoning_fixed_sweeps_metrics.py`
- `reasoning/prefill_vs_decode_alignment_experiment_reasoning.py`
- `reasoning/h3_killer_counterfactual_grid_reasoning.py`
- `reasoning/h3_killer_counterfactual_grid_reasoning_v2.py`

Scripts:

- `reasoning/run_h3_grid.sh`

Compact outputs:

- `results/h3_grid/h3_grid_reasoning.md`
- `results/h3_grid/h3_grid_reasoning.json`
- `reasoning/h3_grid_v3_*.json`

## Steering Robustness

Purpose: evaluate whether overlap with the decode-shared subspace helps explain
steering brittleness and whether projection-style repair changes robustness.

Code:

- `brittleness/steering_vector_reliability_multibench_patch_v3.py`
- `brittleness/steering_vector_reliability_multibench_patch_qwen.py`
- `brittleness/steering_vector_reliability_multibench_patch_falcon.py`
- `brittleness/steering_decodeshare_full.py`
- `rebuttal/exp_ranking_flip_steering.py`
- `rebuttal/exp_ranking_flip_trad_family.py`
- `rebuttal/orthogonal_steer/exp_A1_cross_method_rankflip.py`

Compact outputs:

- `brittleness/results/*/summary*`
- `rebuttal/important_results_summary.md`
- `rebuttal/DecodeShare_experiments_summary.md`
- `rebuttal/orthogonal_steer/results/A1_cross_method/summary.md`

## Downstream Compression

Purpose: compare downstream compression / whitening variants that use
DecodeShare-style bases.

Code:

- `downstream/run_svdllm_whitening_only.py`
- `downstream/export_layer_pca_basis.py`
- `downstream/make_calib_mix_jsonl.py`
- `downstream/svdllm_vendor/`

Script:

- `downstream/run_compare.sh`

Large `.pt` outputs are not tracked in GitHub; see
`docs/artifact_manifest.tsv`.
