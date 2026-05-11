# Mock-Test Results

Date: 2026-05-11

Scope: lightweight reproducibility checks only. These tests verify that canonical scripts can be invoked, that required helper scripts are present, and that compact local result artifacts exist. They do not run long GPU experiments or compare full numeric outputs.

Default Python used by mock scripts:

```bash
conda run -n flashsvd python
```

This matters because the base Python in this environment does not have `torch`.

## Results

| Section | Command | Result |
|---|---|---|
| Paper PDF | `bash camera_ready/00_paper/run_mock.sh` | PASS |
| H1 sharedness | `bash camera_ready/01_h1_sharedness/run_mock.sh` | PASS |
| H2 ablation/energy | `bash camera_ready/02_h2_decode_ablation/run_mock.sh` | PASS |
| H2 patchback | `bash camera_ready/03_h2_patchback/run_mock.sh` | PASS |
| H3 prefill/decode | `bash camera_ready/04_h3_prefill_decode/run_mock.sh` | PASS |
| Steering repair | `bash camera_ready/05_steering_repair/run_mock.sh` | PASS |
| Table aggregation artifacts | `bash camera_ready/90_make_tables/run_mock.sh` | PASS |

## Artifact-Based Regeneration Checks

These commands regenerate compact paper-style summaries from existing local results:

| Check | Command | Output | Result |
|---|---|---|---|
| H1 summary regeneration | `conda run -n flashsvd python experiments/01_sharedness/summarize_full_benchmark.py --results_dir paper_artifacts/h1_results/results/full_benchmark --out_dir /tmp/decodeshare_camera_ready_mock_h1 --alpha 0.05` | `/tmp/decodeshare_camera_ready_mock_h1/H1_full_benchmark_summary.{csv,md,tex}` and `H1_evidence_chain.tex` | PASS |
| H3 table regeneration | `conda run -n flashsvd python experiments/04_prefill_decode/summarize_h3_grid.py --inputs /home/zs89/decodeshare/results/h3_grid/h3_grid_v3_meta-llama_Llama-2-7b-chat-hf_layer10_k48_W0_seed0.json --out_csv /tmp/decodeshare_camera_ready_mock_h3/out.csv --out_latex /tmp/decodeshare_camera_ready_mock_h3/out.tex --latex_mode acc` | `/tmp/decodeshare_camera_ready_mock_h3/out.{csv,tex}` | PASS |

H3 regenerated key sanity numbers:

- model: `meta-llama/Llama-2-7b-chat-hf`, layer `10`, k `48`
- mean principal angle: `79.74 deg`
- macro `Delta Dec-est/Dec-int`: `-17.1 pp`
- macro `Delta Pre-est/Dec-int`: `+0.2 pp`
- H3 contrast: `-17.3 pp`

## What Was Validated

- The paper PDF exists in the camera-ready worktree.
- Canonical experiment scripts respond to `--help` under the `flashsvd` environment.
- The H1 summarizer and H3 table analyzer are present in the clean branch:
  - `experiments/01_sharedness/summarize_full_benchmark.py`
  - `experiments/04_prefill_decode/summarize_h3_grid.py`
- Compact H1 full-benchmark records and generated summaries are checked in under `paper_artifacts/h1_results/results/full_benchmark`.
- Large non-H1 raw artifacts remain external unless listed in `MANIFEST.md`.

## Not Validated

- Full GPU reruns.
- Dataset/model download access.
- Numeric equality between regenerated raw outputs and paper tables.
- Multi-node scheduling behavior. Current constraint remains: use only `Node0` and `Node1`.

## Next Mock-Test Upgrade

Add a tiny smoke mode per section, for example `n_prompts=2`, `n_eval=4`, and a small local output directory. That would test actual model/data plumbing without attempting full paper-scale replication.
