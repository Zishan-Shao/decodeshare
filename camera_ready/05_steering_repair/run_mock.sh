#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ORIG="/home/zs89/decodeshare"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
read -r -a PYTHON_ARR <<< "${PYTHON_CMD:-conda run -n flashsvd python}"

(cd "${ROOT}/experiments/05_steering_repair" && "${PYTHON_ARR[@]}" steering_vector_reliability_multibench_patch_v3.py --help >/dev/null)
"${PYTHON_ARR[@]}" "${ROOT}/experiments/05_steering_repair/summarize_multibench_v3_full.py" --help >/dev/null
test -s "${ORIG}/brittleness/results/steer_repair_multibench_v3/summary_pack/tables_multibench_v3_full.tex"
echo "steering_repair_mock_ok"
