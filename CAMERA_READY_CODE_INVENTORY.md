# DecodeShare Camera-Ready Code Inventory

This note maps the experimental claims in `paper_artifacts/DecodeShare_camera_ready.pdf` to the code and result artifacts in the camera-ready worktree.

Cluster constraint for reruns: use only `Node0` and `Node1` unless this changes.

## High-Level Status

The camera-ready branch now uses the public layout below:

- `experiments/01_sharedness/`: H1 shared-subspace existence, sharedness diagnostics, convergence, threshold sensitivity.
- `experiments/02_decode_ablation/`: H2 decode-only ablations, LOTO, and energy-matched controls.
- `downstream/patch_back/`: H2 patchback, open-answer patchback, transfer, and alpha-sweep controls.
- `experiments/04_prefill_decode/`: H3 prefill/decode mismatch and table extraction.
- `downstream/brittleness/`: steering repair, multibench robustness, pirate style sanity checks.
- `downstream/rebuttal/`: rank-flip / deployment-facing experiments plus after-review additions.
- `scripts/full_runs/`: longer full-run shell wrappers moved out of the repository root.
- `paper_artifacts/`: PDF and compact paper-facing artifacts.

Absolute `/home/zs89/decodeshare/...` paths below refer to the original
experiment workspace used as the artifact source. They are provenance pointers,
not paths in the cleaned camera-ready tree.

## Required Code Bundles

### 1. H1 Shared Decode-Time Structure

Paper outputs:

- Main: Figures 2-4; Table 6.
- Appendix: Figures 8, 11-14; Tables 7-13.

Primary code:

- `experiments/01_sharedness/run_full_benchmark.py`: canonical full benchmark H1 runner, including additional tasks/models.
- `experiments/01_sharedness/sharedness_base.py`: original fair-task H1 runner and sharedness utilities.
- `experiments/01_sharedness/collect_activations.py`: saves per-task decode activations for diagnostics.
- `experiments/01_sharedness/analyze_within_vs_mixed.py`: within-category vs mixed-category sharedness.
- `experiments/01_sharedness/analyze_task_count_convergence.py`: pooled-subspace convergence vs number of tasks.
- `experiments/01_sharedness/analyze_phase_convergence.py`: decode/prefill/decode-step convergence diagnostics.
- `experiments/01_sharedness/analyze_tau_sensitivity.py`: tau/PCA-retention sensitivity.
- `experiments/01_sharedness/summarize_full_benchmark.py`: aggregates H1 JSON/TXT into paper-ready CSV/Markdown/LaTeX.
- `experiments/01_sharedness/PAPER_RESULTS.md`: maps H1 paper outputs to commands and artifacts.
- `plot/plot_pca_specturm.py`: PCA spectrum plots with shared components marked.

Current artifacts:

- `paper_artifacts/h1_results/results/full_benchmark/H1_full_benchmark_summary.md`
- `paper_artifacts/h1_results/results/full_benchmark/H1_full_benchmark_summary.tex`
- `paper_artifacts/h1_results/results/full_benchmark/*_exist*.json`
- `paper_artifacts/h1_results/results/full_benchmark/*_exist*.txt`
- `paper_artifacts/h1_results/results/exp1/*.csv`
- `paper_artifacts/h1_results/results/exp2/*.csv`
- `paper_artifacts/h1_results/results/exp2.75/*.csv`
- `paper_artifacts/h1_results/results/exp2.75/*.csv`
- `paper_artifacts/h1_results/results/exp3/*.csv`
- `plot/*.pdf`

Camera-ready action:

- Keep `run_full_benchmark.py` as the canonical H1 entry point.
- Regenerate Table 6 / Tables 7-13 from checked-in `paper_artifacts/h1_results/results/full_benchmark` records.
- Keep old H1 wrappers under `experiments/01_sharedness/legacy/` only.

### 2. H2 Decode-Only Ablation, LOTO, and Energy-Matched Controls

Paper outputs:

- Main: Figure 7.
- Appendix: Figures 9-10; Tables 5, 26-28.

Primary code:

- `experiments/02_decode_ablation/run_loto_reasoning.py`: canonical LOTO/all-task decode-only shared removal with forced-choice and generation metrics.
- `experiments/02_decode_ablation/run_energy_kmatch_reasoning.py`: alpha-match and k-match energy controls.
- `experiments/02_decode_ablation/summarize_disturb_cot_results.py`
- `experiments/02_decode_ablation/summarize_disturb_cot_diagnostics.py`
- `experiments/02_decode_ablation/summarize_energy_kmatch_outputs.py`
- `experiments/02_decode_ablation/configs/loto_forced_choice.yaml`
- Root runners:
  - `scripts/full_runs/run_disturb_cot_loto8_fc_reason.sh`
  - `scripts/full_runs/run_alpha_kmatch_sweep.sh`

Current artifacts:

- `results/disturb_cot_reasoning/energy_balance_loto8_reasoning_fc_eval2048.md`
- `results/disturb_cot_reasoning/energy_balance_loto8_reasoning_fc.md`
- `results/energy_kmatch_alpha_sweep/*.json`
- `results/energy_kmatch_alpha_sweep/*.tex`
- `reasoning/energy_balance_loto8_reasoning_fc_eval2048.md`

Camera-ready action:

- Treat `results/disturb_cot_reasoning/*.md` and `results/energy_kmatch_alpha_sweep/*.tex` as the paper-facing summaries.
- Do not commit the multi-GB raw JSONs unless the artifact policy requires it; put them in an artifact manifest with checksums instead.
- Keep only the forced-choice capable LOTO runner at top level; retain older LOTO variants under `experiments/02_decode_ablation/legacy/` with README rationale.

### 3. H2 Patchback and Transfer

Paper outputs:

- Main: Table 1; Figures 5-6.
- Appendix: Tables 14-15, 20; Figures 16-17.

Primary code:

- `experiments/03_patchback/subspace_patching_transfer.py`: multiple-choice patchback on flip sets.
- `experiments/03_patchback/openanswer_subspace_patching.py`: GSM8K/HumanEval open-answer patchback.
- `experiments/03_patchback/flipset_alpha_sweep_and_transfer.py`: AQuA alpha-sweep and transfer-donor patching.
- `experiments/03_patchback/summarize_patching_jsons.py`: aggregates patchback JSONs.
- Run wrappers:
  - `downstream/patch_back/run_decodeshare_suite.sh`
  - `downstream/patch_back/run_qwen_suite_and_report.sh`
  - richer model/layer wrappers remain in the original workspace unless copied later.

Current artifacts:

- `/home/zs89/decodeshare/patch_back/paper/patchback_tables_all_models_all_layers.tex`
- `/home/zs89/decodeshare/patch_back/paper/patchback_discussion_all_models_all_layers.tex`
- `/home/zs89/decodeshare/patch_back/results/**/_summary/*.md`
- `/home/zs89/decodeshare/patch_back/results/**/_summary/*.tex`
- `/home/zs89/decodeshare/patch_back/results/openanswer/*.json`
- `/home/zs89/decodeshare/patch_back/results/runs_flip_supplement/**/*.json`

Camera-ready action:

- Keep `/home/zs89/decodeshare/patch_back/paper/patchback_tables_all_models_all_layers.tex` as the authoritative table source until copied into `paper_artifacts/tables/`.
- Preserve the legacy note for Llama layer 10, because the current table explicitly mixes legacy and newer outputs.
- Add a manifest listing which JSON directories feed each row in Tables 1, 14, 15, and 20.

### 4. H3 Prefill-Decode Mismatch

Paper outputs:

- Main: Table 3.
- Appendix: Tables 16-19; Figure 14.

Primary code:

- `experiments/04_prefill_decode/run_h3_grid_reasoning_v2.py`: despite the header saying v3, this is the 2x2 H3 estimation/intervention grid runner.
- `experiments/04_prefill_decode/run_prefill_decode_reasoning.py`: dimension-matched prefill-vs-decode alignment experiments.
- `experiments/04_prefill_decode/summarize_h3_grid.py`: H3 table extraction.
- Root runner moved to `scripts/full_runs/run_prefill_decode_nextsteps.sh`

Current artifacts:

- `/home/zs89/decodeshare/results/h3_grid/h3_grid_reasoning.md`
- `/home/zs89/decodeshare/results/h3_grid/out.tex`
- `/home/zs89/decodeshare/results/h3_grid/h3_grid_v3_*_layer*_k*_W0_seed0.json`
- `/home/zs89/decodeshare/results/prefill_decode_nextsteps/k_16.md`
- `/home/zs89/decodeshare/results/prefill_decode_nextsteps/k_126.md`
- `/home/zs89/decodeshare/results/prefill_decode_nextsteps/alpha_*.md`
- `downstream/rebuttal/pca_prefill_decode_mismatch_layer28.json`

Camera-ready action:

- Rename or wrap the H3 runner to avoid the `v2.py` filename vs `v3` docstring mismatch.
- Use `experiments/04_prefill_decode/summarize_h3_grid.py` as the canonical table generator.
- Keep separate paper tables for native rank, k-matched rank, and small-k sanity, because they answer different reviewer concerns.

### 5. Steering Repair, Multibench Robustness, and Pirate Sanity Checks

Paper outputs:

- Main: Table 2.
- Appendix: Figure 15; Tables 21-25, 29.

Primary code:

- `experiments/05_steering_repair/steering_vector_reliability_multibench_patch_v3.py`: beta sweep, candidate calibration, multibench repair.
- `experiments/05_steering_repair/summarize_multibench_v3_full.py`: paper-ready multibench tables.
- `experiments/05_steering_repair/mvp_projection_patch_pirate_v5.py`: pirate style projection sanity check.
- `run_mvp_pirate_qwen.sh`

Current artifacts:

- `/home/zs89/decodeshare/brittleness/results/steer_repair_multibench_v3/summary_pack/summary_multibench_v3_full.md`
- `/home/zs89/decodeshare/brittleness/results/steer_repair_multibench_v3/summary_pack/tables_multibench_v3_full.tex`
- `/home/zs89/decodeshare/brittleness/results/sharedspace_solid_llama2_7b_chat/summary.md`
- `/home/zs89/decodeshare/brittleness/results/sharedspace_solid_llama2_7b_chat/summary.tex`
- `/home/zs89/decodeshare/brittleness/results/mvp_pirate_v5_story_clean/summary.md`

Camera-ready action:

- Use `summary_pack/tables_multibench_v3_full.tex` for Table 2.
- Keep the claim conservative: the evidence supports a robustness/variance knob with task-dependent trade-offs, not a universal steering improvement.
- Include candidate calibration details because Tables 22-23 depend on them.

### 6. Rank-Flip and Deployment-Facing Steering Selection

Paper status:

- This is represented in `downstream/rebuttal/`; it may or may not be in the current camera-ready PDF body.

Primary code:

- `downstream/rebuttal/exp_ranking_flip_steering.py`
- `downstream/rebuttal/exp_ranking_flip_steering_full.py`
- `downstream/rebuttal/exp_ranking_flip_trad_family.py`
- `downstream/rebuttal/orthogonal_steer/exp_A1_cross_method_rankflip.py`

Current artifacts:

- `downstream/rebuttal/results/rebuttal_rankflip_layer28_rand100_full/*.json`
- `downstream/rebuttal/orthogonal_steer/results/A1_cross_method/*`

Camera-ready action:

- If rank-flip remains in camera-ready, package it as a separate optional bundle because it is deployment-facing rather than core H1/H2/H3.
- Keep `TRAD-A`, `TRAD-B`, `DECODE`, and `REAL` definitions explicit.

### 7. After-Review Additions

Paper status:

- Useful if added to appendix/rebuttal-derived sections; not all are core to the current PDF.

Primary code:

- Not currently copied into this camera-ready worktree. Keep these external
  unless the camera-ready PDF cites them directly.

Current artifacts:

- External in the original workspace if needed.

Camera-ready action:

- Include only the additions that are actually cited in the camera-ready manuscript.
- The scale-extension summary currently supports `13B H1 + H2-lite`; it explicitly does not package 30B/70B causal results.

## Shared Utilities and Environment

Utilities used across many experiments:

- `src/joint_subspace_large/disturb_cross_task_all_shared.py`: shared subspace computation utilities.
- `experiments/02_decode_ablation/benchmark_dataloaders.py`
- `experiments/04_prefill_decode/benchmark_dataloaders.py`
- `downstream/patch_back/benchmark_dataloaders.py`
- `downstream/brittleness/benchmark_dataloaders.py`
- `environment.yml`

Camera-ready action:

- Consolidate the duplicate `benchmark_dataloaders.py` copies or document which copy each experiment bundle imports.
- Pin package versions from `environment.yml` and mention Hugging Face dataset/model access requirements.
- Add a small smoke-test mode for each bundle (`n_eval` / `n_subspace` small) so artifact reviewers can validate code paths without rerunning full jobs.

## Immediate Risks / Cleanup Items

1. The repository contains only the submitted PDF, not the LaTeX source for `17010_DecodeShare` in the current root. If the paper source lives elsewhere, link each table label to the code bundle above.
2. Some raw JSONs are multi-GB, especially `/home/zs89/decodeshare/results/disturb_cot_reasoning/*.json`; these should be referenced as artifacts, not casually committed.
3. Rebuttal-only result vectors are currently grouped under `downstream/rebuttal/`; decide whether they should stay in git or become external artifacts.
4. Several experiments have legacy/new runner mismatch, especially H3 (`v2.py` filename vs v3 docstring) and patchback layer-10 legacy rows.
5. The current PDF text extraction shows an extra reviewer-instruction-like string at the end of the document. Remove it from camera-ready output if it is present in the source/PDF.
6. Claims about 30B/70B should remain conservative unless new packaged runs are produced.

## Recommended Camera-Ready Organization

The cleaned branch now exposes these stable smoke wrappers:

- `scripts/run_all_smoke_tests.sh`
- `scripts/reproduce_h1_tables.sh`
- `scripts/reproduce_ablation_tables.sh`
- `scripts/reproduce_table_1_patchback.sh`
- `scripts/reproduce_table_3_h3.sh`
- `scripts/reproduce_table_2_steering.sh`

Each full rerun should:

- Set `CUDA_VISIBLE_DEVICES` explicitly.
- Write to a timestamped output directory.
- Save the exact command, git commit, environment, model ID, dataset split, seed, and node name.
- Emit a compact `.md`/`.tex` summary separate from raw JSON.
