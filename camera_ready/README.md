# Camera-Ready Rerun Wrappers

This directory is ordered to follow the paper's experimental organization, excluding rebuttal-only material.

Current cluster constraint: use only `Node0` and `Node1`.

## Order

1. `01_h1_sharedness/`: H1, shared decode-time structure.
2. `02_h2_decode_ablation/`: H2, decode-only removal, LOTO, and energy controls.
3. `03_h2_patchback/`: H2 sufficiency/patchback and transfer controls.
4. `04_h3_prefill_decode/`: H3, prefill/decode estimator-deployment mismatch.
5. `05_steering_repair/`: downstream steering repair and template robustness.
6. `90_make_tables/`: table/figure aggregation wrappers.

Each folder contains a `COMMANDS.md` with the local artifacts found and the corresponding full-run commands. `run_mock.sh` files are lightweight checks only; they do not start long GPU jobs.
