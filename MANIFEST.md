# Camera-Ready Artifact Manifest

This manifest tracks which files in the camera-ready branch are canonical and which large files remain external artifacts.

## Cluster

- Available nodes: `Node0`, `Node1`
- Avoid scheduling camera-ready reruns on other nodes unless the cluster status changes.

## Canonical Bundles

| Bundle | Paper outputs | Code entry point | Summary artifacts | Raw artifacts |
|---|---|---|---|---|
| H1 sharedness | Figures 2-4, 8, 11-14; Tables 6-13 | `camera_ready/01_h1_sharedness/COMMANDS.md` | `paper_artifacts/h1_results/results/full_benchmark/H1_full_benchmark_summary.*` | `paper_artifacts/h1_results/results/full_benchmark/*_exist*.(json\|txt)` |
| H2 ablation/LOTO | Figure 7; Tables 5, 26-28 | `camera_ready/02_h2_decode_ablation/COMMANDS.md` | `/home/zs89/decodeshare/results/disturb_cot_reasoning/*.md`, `/home/zs89/decodeshare/results/energy_kmatch_alpha_sweep/*.tex` | external/checksum TODO |
| H2 patchback | Table 1; Tables 14-15, 20; Figures 5-6, 16-17 | `camera_ready/03_h2_patchback/COMMANDS.md` | `/home/zs89/decodeshare/patch_back/paper/*.tex` | external/checksum TODO |
| H3 prefill/decode | Table 3; Tables 16-19; Figure 14 | `camera_ready/04_h3_prefill_decode/COMMANDS.md` | `/home/zs89/decodeshare/results/h3_grid/out.tex`, `/home/zs89/decodeshare/results/prefill_decode_nextsteps/*.md` | external/checksum TODO |
| Steering repair | Table 2; Tables 21-25, 29; Figure 15 | `camera_ready/05_steering_repair/COMMANDS.md` | `/home/zs89/decodeshare/brittleness/results/steer_repair_multibench_v3/summary_pack/*` | external/checksum TODO |
| Optional rank-flip | deployment/rebuttal evidence | `downstream/rebuttal/` | `downstream/rebuttal/results/rebuttal_*` | external/checksum TODO |

## Artifact Policy

- Commit compact `.md`, `.csv`, `.tex`, and final figure/table assets.
- Do not commit paper PDFs, internal verification notes, or migration scratch
  inventories.
- Do not commit multi-GB raw JSONs by default.
- For each external raw artifact, record:
  - absolute path in the original workspace or artifact store
  - generation command
  - model/dataset/layer/seed
  - file size
  - checksum

## Curation Checklist

- [x] Copy only canonical H1 experiment scripts.
- [ ] Add smoke-test commands for each bundle.
- [ ] Generate paper table/figure summaries from checked-in summaries or manifest-listed raw artifacts.
- [ ] Record environment and package versions.
