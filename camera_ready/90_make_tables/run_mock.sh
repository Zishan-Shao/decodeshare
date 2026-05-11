#!/usr/bin/env bash
set -euo pipefail

ORIG="/home/zs89/decodeshare"

test -s "${ORIG}/Hype1/results/full_benchmark/H1_full_benchmark_summary.tex"
test -s "${ORIG}/results/energy_kmatch_alpha_sweep/meta-llama_Llama-2-7b-chat-hf_L10_seed42_ts20260110_080440.tex"
test -s "${ORIG}/patch_back/paper/patchback_tables_all_models_all_layers.tex"
test -s "${ORIG}/results/h3_grid/out.tex"
test -s "${ORIG}/brittleness/results/steer_repair_multibench_v3/summary_pack/tables_multibench_v3_full.tex"
echo "make_tables_mock_ok"
