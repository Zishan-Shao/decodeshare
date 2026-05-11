# 03 Patchback

Paper role: H2 sufficiency, patchback, and transfer controls.

Primary outputs:

- Main patchback table.
- Open-answer and transfer-control appendix summaries.
- Flipset and alpha-sweep artifacts used for paper tables and figures.

Canonical scripts in this folder:

- `subspace_patching_transfer.py`: multiple-choice patchback on flip sets.
- `openanswer_subspace_patching.py`: GSM8K/HumanEval open-answer patchback.
- `flipset_alpha_sweep_and_transfer.py`: AQuA alpha sweep and transfer-donor patching.
- `summarize_patching_jsons.py`: JSON-to-summary aggregation.
- `benchmark_dataloaders.py`: local task loader dependency.
- `disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py`: local LOTO/helper dependency used by the patchback scripts.

The complete historical bundle remains in `downstream/patch_back/`; this folder
keeps only the paper-facing entry points.

Smoke check:

```bash
bash scripts/reproduce_table_1_patchback.sh
```

Full command records:

- `camera_ready/03_h2_patchback/COMMANDS.md`
