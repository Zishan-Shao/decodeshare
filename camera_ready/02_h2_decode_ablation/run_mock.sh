#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ORIG="/home/zs89/decodeshare"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
read -r -a PYTHON_ARR <<< "${PYTHON_CMD:-conda run -n flashsvd python}"

(cd "${ROOT}/experiments/02_decode_ablation" && CUDA_VISIBLE_DEVICES="" "${PYTHON_ARR[@]}" run_loto_reasoning.py --help >/dev/null)
(cd "${ROOT}/experiments/02_decode_ablation" && CUDA_VISIBLE_DEVICES="" "${PYTHON_ARR[@]}" run_energy_kmatch_reasoning.py --help >/dev/null)
test -s "${ORIG}/results/disturb_cot_reasoning/energy_balance_loto8_reasoning_fc_eval2048.md"
test -s "${ORIG}/results/energy_kmatch_alpha_sweep/meta-llama_Llama-2-7b-chat-hf_L10_seed42_ts20260110_080440.tex"
echo "h2_ablation_mock_ok"
