# Mock-Test Results

Date: 2026-05-11

Scope: lightweight reproducibility checks only. These tests verify that
canonical scripts can be invoked and that required helper code imports. They do
not run long GPU experiments or compare full numeric outputs.

Default Python used by mock scripts:

```bash
python
```

Set `PYTHON_CMD="conda run -n flashsvd python"` if you are not already inside
the project environment.

## Results

| Section | Command | Result |
|---|---|---|
| H1 sharedness | `bash scripts/01_h1_sharedness/run_mock.sh` | PASS |
| H2 ablation/energy | `bash scripts/02_h2_decode_ablation/run_mock.sh` | PASS |
| H2 patchback | `bash scripts/03_h2_patchback/run_mock.sh` | PASS |
| H3 prefill/decode | `bash scripts/04_h3_prefill_decode/run_mock.sh` | PASS |
| Steering repair | `bash scripts/05_steering_repair/run_mock.sh` | PASS |

## What Was Validated

- Canonical experiment scripts respond to `--help`.
- The H1 summarizer and H3 table analyzer are present in the clean branch:
  - `experiments/01_sharedness/summarize_full_benchmark.py`
  - `experiments/04_prefill_decode/summarize_h3_grid.py`
- Experiment entry points can be imported and print CLI help from the public tree.
- Mock checks do not depend on private local artifact paths.

## Not Validated

- Full GPU reruns.
- Dataset/model download access.
- Numeric equality between regenerated raw outputs and paper tables.
- Multi-node scheduling behavior. Current constraint remains: use only `Node0` and `Node1`.

## Next Mock-Test Upgrade

Add a tiny smoke mode per section, for example `n_prompts=2`, `n_eval=4`, and a small local output directory. That would test actual model/data plumbing without attempting full paper-scale replication.
