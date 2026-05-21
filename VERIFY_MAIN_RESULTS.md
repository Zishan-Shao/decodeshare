# Main Result Verification

This file records the artifact-replay checks used to validate the main
camera-ready results. The standard here is deliberately narrower than a full GPU
rerun: given existing raw/local/Hugging Face artifacts, rerun the public
summarizers and confirm that the main paper numbers are reproduced.

Rank-flip and rebuttal-only experiments are out of scope for this pass. They are
kept under `downstream/rebuttal/` and should be curated in a later artifact pass.

## Verification Scope

| Section | Paper outputs | Source artifacts | Status |
|---|---|---|---|
| H1 sharedness | Figures 2-4, 8, 11-14; Tables 6-13 | `paper_artifacts/h1_results/results/full_benchmark/*_exist*.(json|txt)` | PASS |
| H2 decode ablation and LOTO | Figure 7; Tables 5, 26-28 | `/home/zs89/decodeshare/results/disturb_cot_reasoning/*.json`, `/home/zs89/decodeshare/results/energy_kmatch_alpha_sweep/*.json` | PASS |
| H2 patchback | Table 1; Tables 14-15, 20; Figures 5-6, 16-17 | `/home/zs89/decodeshare/patch_back/results/**/*.json`, HF `artifacts/patch_back/results/` | PASS |
| H3 prefill/decode | Table 3; Tables 16-19; Figure 14 | `/home/zs89/decodeshare/results/h3_grid/*.json` | PASS |
| Steering repair | Table 2; Tables 21-25, 29; Figure 15 | `/home/zs89/decodeshare/brittleness/results/steer_repair_multibench_v3/` | PASS |

## Commands Replayed

### H1 Sharedness

```bash
conda run -n flashsvd python experiments/01_sharedness/summarize_full_benchmark.py \
  --results_dir paper_artifacts/h1_results/results/full_benchmark \
  --out_dir /tmp/decodeshare_verify_h1_latest \
  --alpha 0.05
```

Checked regenerated summary artifacts against the checked-in H1 summary CSV and
LaTeX outputs. Result: clean match.

### H2 LOTO

```bash
conda run -n flashsvd python experiments/02_decode_ablation/summarize_disturb_cot_results.py \
  --results_dir /home/zs89/decodeshare/results/disturb_cot_reasoning \
  --pattern energy_balance_loto8_reasoning_fc_eval2048.json \
  --no_recursive \
  --output /tmp/decodeshare_verify_loto_official_auto.md
```

The replayed table matches the paper LOTO values. The current summarizer uses
`--decoding auto`, so GSM8K is read from the `greedy` branch and multiple-choice
tasks are read from `forced_choice`.

Key replayed rows:

| Task | Protocol | Baseline | Shared full | p-value |
|---|---|---:|---:|---:|
| gsm8k | greedy | 4.9 | 2.3 | 0.0001 |
| commonsenseqa | forced_choice | 54.1 | 50.0 | 0.0043 |
| strategyqa | forced_choice | 55.5 | 52.3 | 0.0182 |
| aqua | forced_choice | 24.0 | 17.3 | 0.0547 |
| arc_challenge | forced_choice | 50.9 | 40.4 | 0.0001 |
| openbookqa | forced_choice | 50.2 | 41.6 | 0.0005 |
| qasc | forced_choice | 48.5 | 40.7 | 0.0001 |
| logiqa | forced_choice | 32.7 | 26.9 | 0.0147 |

### H2 Energy Controls

```bash
conda run -n flashsvd python experiments/02_decode_ablation/summarize_energy_kmatch_outputs.py \
  --results_dir /home/zs89/decodeshare/results/energy_kmatch_alpha_sweep \
  --pattern meta-llama_Llama-2-7b-chat-hf_L10_seed42_ts20260110_080440.json \
  --output /tmp/decodeshare_verify_energy.md \
  --write_csv 1 \
  --csv_prefix /tmp/decodeshare_verify_energy
```

Verified the raw JSON against the paper LaTeX table for the same run. Result:
8 alpha tables and 64 task rows matched.

### H2 Patchback

```bash
mkdir -p /tmp/decodeshare_verify_patchback
conda run -n flashsvd python experiments/03_patchback/summarize_patching_jsons.py \
  --dir /home/zs89/decodeshare/patch_back/results/Qwen/Qwen2.5-7B-Instruct/layer10 \
  --pattern "**/*.json" \
  --recursive \
  --no_dedupe \
  --out_csv /tmp/decodeshare_verify_patchback/summary.csv \
  --out_md /tmp/decodeshare_verify_patchback/summary.md \
  --out_paper_md /tmp/decodeshare_verify_patchback/paper_table.md \
  --out_alpha_csv /tmp/decodeshare_verify_patchback/alpha_sweep.csv \
  --out_alpha_md /tmp/decodeshare_verify_patchback/alpha_sweep.md
```

Verified Qwen layer-10 summaries against the local patchback summary CSV, paper
table, and alpha-sweep outputs. Result: clean match.

Note: paper summaries compute scan accuracies using effective, non-skipped rows.
The run script now records `n_effective` and `n_skipped` in raw JSON metadata so
future raw outputs use the same denominator as the summarizer.

### H3 Prefill/Decode

```bash
conda run -n flashsvd python experiments/04_prefill_decode/summarize_h3_grid.py \
  --inputs /home/zs89/decodeshare/results/h3_grid/h3_grid_v3_meta-llama_Llama-2-7b-chat-hf_layer10_k48_W0_seed0.json \
  --out_csv /tmp/decodeshare_verify_h3/out.csv \
  --out_latex /tmp/decodeshare_verify_h3/out.tex \
  --latex_mode acc
```

Checked regenerated `out.csv` and `out.tex` against
`/home/zs89/decodeshare/results/h3_grid/out.csv` and `out.tex`. Result: clean
match.

### Steering Repair

```bash
conda run -n flashsvd python experiments/05_steering_repair/summarize_multibench_v3_full.py \
  --root_dir /home/zs89/decodeshare/brittleness/results/steer_repair_multibench_v3 \
  --out_dir /tmp/decodeshare_verify_steer_latest
```

Checked regenerated summary and LaTeX tables against the local
`summary_pack/`. Result: clean match.

## Code Audit Fixes Applied

- `experiments/02_decode_ablation/summarize_disturb_cot_results.py`: default
  `--decoding auto` handles mixed `greedy` and `forced_choice` LOTO outputs.
- `experiments/02_decode_ablation/summarize_energy_kmatch_outputs.py`: current
  `alpha_runs` schema is parsed directly, and both `accuracy` and `acc` fields
  are accepted.
- `experiments/03_patchback/subspace_patching_transfer.py`: scan accuracy
  metadata now uses the effective denominator after skipped rows are removed.
- `experiments/01_sharedness/summarize_full_benchmark.py`: TXT fallback parsing
  now accepts scientific-notation p-values.

## Lightweight Checks

```bash
find experiments -maxdepth 2 -type f -name '*.py' ! -path '*/legacy/*' -print0 | \
  xargs -0 python -m py_compile

bash scripts/run_all_smoke_tests.sh
```

Both checks passed. The smoke test output ended with:

```text
all_smoke_tests_ok
```

## Remaining Artifact Notes

- The Hugging Face repository `Zishan-Shao/decodeshare` currently contains the
  patchback artifact tree and selected large artifacts, but not every H2/H3 or
  steering raw artifact required for standalone external replay.
- `MANIFEST.md` still marks several raw artifact entries as
  `external/checksum TODO`. This is not a blocker for the current local
  verification standard, but it should be completed before claiming that HF plus
  GitHub alone is a complete artifact bundle.
