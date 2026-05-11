#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ORIG="/home/zs89/decodeshare"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
read -r -a PYTHON_ARR <<< "${PYTHON_CMD:-conda run -n flashsvd python}"

(cd "${ROOT}/experiments/04_prefill_decode" && "${PYTHON_ARR[@]}" run_h3_grid_reasoning_v2.py --help >/dev/null)
"${PYTHON_ARR[@]}" "${ROOT}/experiments/04_prefill_decode/summarize_h3_grid.py" --help >/dev/null
test -s "${ORIG}/results/h3_grid/out.tex"
test -s "${ORIG}/results/prefill_decode_nextsteps/k_16.md"
echo "h3_mock_ok"
