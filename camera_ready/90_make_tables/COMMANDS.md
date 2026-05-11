# 90 Make Tables and Figures

This folder should eventually become the single entry point for regenerating compact paper artifacts from selected result summaries.

## Inputs by Section

- H1: `/home/zs89/decodeshare/Hype1/results/full_benchmark/H1_full_benchmark_summary.tex`
- H2 ablation: `/home/zs89/decodeshare/results/disturb_cot_reasoning/*.md`
- H2 energy controls: `/home/zs89/decodeshare/results/energy_kmatch_alpha_sweep/*.tex`
- H2 patchback: `/home/zs89/decodeshare/patch_back/paper/patchback_tables_all_models_all_layers.tex`
- H3: `/home/zs89/decodeshare/results/h3_grid/out.tex`, `/home/zs89/decodeshare/results/prefill_decode_nextsteps/*.tex`
- Steering repair: `/home/zs89/decodeshare/brittleness/results/steer_repair_multibench_v3/summary_pack/tables_multibench_v3_full.tex`

## Mock Command

```bash
bash camera_ready/90_make_tables/run_mock.sh
```

The mock checks presence of compact artifacts only.
