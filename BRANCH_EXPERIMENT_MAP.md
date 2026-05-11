# DecodeShare Branch Experiment Map

This note maps the experiments in the current paper draft
`17010_DecodeShare_Tracing_the.pdf` to the branches that currently exist in the
repository.

The goal is practical: if you move to a new cluster and want the most complete
open-source starting point for the paper, this file should tell you which
branch to use and where each experiment family lives.

## Short Recommendation

- Use `Halo` as the base branch for the main paper release.
- Use `lighthouse` for rebuttal-only and post-paper extensions.
- Do not use `main` as the primary open-source base; it is an in-between branch
  and is less complete than `Halo` for the paper package.

Why:

- `Halo` is the curated release branch with the best paper-facing packaging:
  `LICENSE`, `CITATION.cff`, `docs/REPRODUCIBILITY.md`,
  `docs/MODEL_AND_DATA.md`, `docs/RELEASE_NOTES.md`,
  `docs/HUGGINGFACE_UPLOAD.md`, and `docs/artifact_manifest.tsv`.
- `lighthouse` adds many useful rebuttal scripts, notes, and infra updates, but
  it does not carry the same complete release packaging for the original paper.
- `Halo` and `lighthouse` should be treated as parallel lines, not as a simple
  linear history.

## Branch Roles

| Branch | Current role | Best use |
|---|---|---|
| `Halo` | Curated ICML-style public release branch | Main paper open source |
| `lighthouse` | Rebuttal and post-paper development branch | Mechanism follow-ups, rebuttal code, cluster-friendly evaluator updates |
| `main` | Intermediate branch between the older repo state and `lighthouse` | Not recommended as the main public branch |

## Paper Experiment Map

### 1. `H1`: Shared Decode-Time Workspace Exists

Paper coverage:

- Abstract
- `§2.3` DECODESHARE shared-subspace construction
- `§3.1` H1 shared structure
- Figures `2`, `3`, `4`

Best branch:

- `Halo`

Where to find it on `Halo`:

- Core code:
  - `Hype1/collect_decode_acts.py`
  - `Hype1/prove_sharedness_decode_fair.py`
  - `Hype1/prove_sharedness_decode_full.py`
  - `src/prove_sharedness_decode_fair.py`
  - `src/prove_sharedness_decode_full.py`
- Run scripts:
  - `Hype1/run_00_collect_acts.sh`
  - `Hype1/run_01_exp1_within_vs_mixed.sh`
  - `Hype1/run_02_exp2_convergence.sh`
  - `run_exists.sh`
- Compact outputs:
  - `Hype1/results/full_benchmark/H1_full_benchmark_summary.md`
  - `Hype1/results/full_benchmark/H1_full_benchmark_summary.csv`
  - `Hype1/results/exp1/`
  - `Hype1/results/exp2/`
  - `Hype1/results/exp2.5/`
  - `Hype1/results/exp2.75/`
  - `Hype1/results/exp3/`
- Release documentation:
  - `docs/REPRODUCIBILITY.md`

What exists on the other branches:

- `main` and `lighthouse` still contain the core `Hype1` scripts.
- `main` and `lighthouse` do not contain the curated `H1` full-benchmark result
  package from `Halo` such as
  `Hype1/results/full_benchmark/H1_full_benchmark_summary.md`.

Conclusion:

- If you want the paper-ready `H1` package, use `Halo`.

### 2. `H2`: Decode-Time Causal Removal of the Shared Subspace

Paper coverage:

- `§2.4` decode-only causal tests with matched controls
- `§3.2.1` and `§3.2.2`
- Figure `5`

Best branch:

- `Halo`

Where to find it on `Halo`:

- Core code:
  - `src/disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py`
  - `src/disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8_reasoning.py`
  - `src/disturb_energy_matched_sharedness_kmatch.py`
  - `reasoning/disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py`
  - `reasoning/disturb_energy_matched_sharedness_kmatch.py`
- Run scripts:
  - `run_disturb_cot_loto8.sh`
  - `run_disturb_cot_loto8_fc_reason.sh`
  - `run_disturb_cot_loto8_main.sh`
  - `run_disturb_cot_loto8_main_qwen.sh`
  - `run_disturb_cot_loto8_main_falcon.sh`
- Compact outputs:
  - `results/disturb_cot_reasoning/energy_balance_loto8_reasoning_fc.md`
  - `results/disturb_cot_reasoning/energy_balance_loto8_reasoning_fc_eval2048.md`
  - `reasoning/energy_balance_loto8_reasoning_fc_eval2048.md`

What exists on the other branches:

- `main` and `lighthouse` still contain the core disturbance code in `src/` and
  `reasoning/`.
- `lighthouse` also contains evaluator upgrades that are useful on a new
  cluster:
  - `reasoning/eval_perf.py`
  - `reasoning/disturb_CoT_shared_loto_reasoning.py`
  - `src/disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8_reasoning.py`
- `main` and `lighthouse` do not contain the curated compact output directory
  `results/disturb_cot_reasoning/` from `Halo`.

Conclusion:

- Use `Halo` for the paper-facing `H2` package.
- Cherry-pick the evaluator/runtime improvements from `lighthouse` if you want a
  better multi-GPU reproduction environment.

### 3. `H2` Patchback

Paper coverage:

- `§3.2.3`
- Figure `6`
- Appendix patchback diagnostics

Best branch:

- `Halo`

Where to find it on `Halo`:

- Core code:
  - `patch_back/subspace_patching_transfer.py`
  - `patch_back/subspace_patching_transfer_enhanced.py`
  - `patch_back/openanswer_subspace_patching.py`
  - `patch_back/openanswer_subspace_patching_qwen.py`
  - `patch_back/openanswer_subspace_patching_falcon.py`
  - `patch_back/summarize_patching_jsons.py`
- Run scripts:
  - `patch_back/run_decodeshare_suite.sh`
  - `patch_back/run_all_flip_experiments.sh`
  - `patch_back/run_qwen_suite_and_report.sh`
  - `patch_back/run_falcon_suite_and_report.sh`
  - `patch_back/run_llama_suite_and_report.sh`
- Compact outputs:
  - `patch_back/paper/patchback_tables_all_models_all_layers.tex`
  - `patch_back/paper/patchback_discussion_all_models_all_layers.tex`
  - `patch_back/results/summary/`

What exists on the other branches:

- `main` and `lighthouse` still contain the core patchback scripts.
- The paper-ready patchback tables and curated summary pack are only on `Halo`.

Conclusion:

- For the patchback part of the paper, `Halo` is the authoritative branch.

### 4. `H3`: Prefill-vs-Decode Mismatch

Paper coverage:

- `§3.3`
- Table `3`
- geometric mismatch and decode-time causal mismatch story

Best branch:

- `Halo`

Where to find it on `Halo`:

- Core code:
  - `src/prefill_vs_decode_alignment_experiment_generation.py`
  - `src/prefill_vs_decode_alignment_experiment_reasoning_fixed_sweeps_metrics.py`
  - `reasoning/prefill_vs_decode_alignment_experiment_reasoning.py`
  - `reasoning/h3_killer_counterfactual_grid_reasoning.py`
  - `reasoning/h3_killer_counterfactual_grid_reasoning_v2.py`
- Run script:
  - `reasoning/run_h3_grid.sh`
- Compact outputs:
  - `results/h3_grid/h3_grid_reasoning.md`
  - `results/h3_grid/h3_grid_reasoning.json`
  - `reasoning/h3_grid_v3_*.json`

What exists on the other branches:

- `main` and `lighthouse` keep the core `H3` code.
- The curated `results/h3_grid/` release outputs and the nicer release README
  framing are only on `Halo`.

Conclusion:

- `Halo` is the branch that most directly matches the paper's `H3` package.

### 5. Section `4`: Downstream Utility / Steering Robustness

Paper coverage:

- `§4` Downstream Utility
- steering-vector overlap and robustness / template-sensitivity story

Best branch for the original paper-facing package:

- `Halo`

Where to find it on `Halo`:

- Core code:
  - `brittleness/steering_vector_reliability_multibench_patch_v3.py`
  - `brittleness/steering_vector_reliability_multibench_patch_qwen.py`
  - `brittleness/steering_vector_reliability_multibench_patch_falcon.py`
  - `brittleness/steering_decodeshare_full.py`
  - `rebuttal/exp_ranking_flip_steering.py`
  - `rebuttal/exp_ranking_flip_trad_family.py`
  - `rebuttal/orthogonal_steer/exp_A1_cross_method_rankflip.py`
- Compact outputs:
  - `brittleness/results/*/summary*`
  - `rebuttal/important_results_summary.md`
  - `rebuttal/DecodeShare_experiments_summary.md`

What exists on the other branches:

- `lighthouse` contains a much larger rebuttal-era expansion of this whole area:
  - `rebuttal/mechanism/*.py`
  - `rebuttal/notes/*.md`
  - `rebuttal/reasoning/*`
  - `rebuttal/steer_robustness/*`
- `lighthouse` is therefore stronger for follow-up analysis, but not as clean as
  `Halo` for the original paper release.

Conclusion:

- For the original paper's downstream-utility story, start from `Halo`.
- For rebuttal and follow-up steering analysis, use `lighthouse`.

## Release-Adjacent Material That Is Useful But Not Central To The Core Paper

### Downstream Compression / Whitening Utilities

Best branch:

- `Halo`

Where:

- `downstream/run_compare.sh`
- `downstream/run_svdllm_whitening_only.py`
- `downstream/export_layer_pca_basis.py`
- `downstream/make_calib_mix_jsonl.py`
- `downstream/svdllm_vendor/`
- `docs/HUGGINGFACE_UPLOAD.md`
- `docs/artifact_manifest.tsv`

Why it matters:

- This material is important for the public release and HF artifact workflow,
  even though it is not the central evidence chain in the paper's `H1/H2/H3`
  narrative.

## Rebuttal-Only / Post-Paper Extensions

These are not the best reason to choose `Halo`, but they are the best reason to
keep `lighthouse` around.

Best branch:

- `lighthouse`

Where:

- Mechanism suite:
  - `rebuttal/mechanism/`
  - `rebuttal/mechanism/PartA/`
- Rebuttal notes:
  - `rebuttal/notes/`
- Quick reasoning follow-ups:
  - `rebuttal/reasoning/`
- Steering robustness follow-ups:
  - `rebuttal/steer_robustness/`
- Additional orthogonal steering scripts:
  - `rebuttal/orthogonal_steer/exp_A1_cross_method_rankflip.py`

Important caveat:

- Large rebuttal result bundles are intentionally not stored in GitHub on
  `lighthouse`.
- They were pushed to Hugging Face instead, especially:
  - `artifacts/results/rebuttal_mechanism/`
  - `artifacts/results/rebuttal_scaling/`
  - `artifacts/rebuttal/`

Practical interpretation:

- `lighthouse` is the better branch for rebuttal and follow-up work.
- `Halo` is the better branch for the main paper release.

## Final Recommendation For A New Cluster

If the goal is:

- **Open-source the main paper cleanly**: start from `Halo`.
- **Re-run or extend rebuttal experiments**: use `lighthouse`.
- **Build a stronger final public branch**: start from `Halo`, then selectively
  bring in these `lighthouse` improvements:
  - `reasoning/eval_perf.py`
  - `reasoning/disturb_CoT_shared_loto_reasoning.py`
  - `src/disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8_reasoning.py`
  - `README.md` artifact-policy language
  - chosen subsets of `rebuttal/mechanism/`, `rebuttal/notes/`,
    `rebuttal/reasoning/`, and `rebuttal/steer_robustness/`

In one sentence:

- `Halo` is the most complete branch for the experiments that are actually in
  the current paper.
- `lighthouse` is the most complete branch for the rebuttal and post-paper
  extensions.
