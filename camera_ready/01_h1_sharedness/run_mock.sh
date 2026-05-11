#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
read -r -a PYTHON_ARR <<< "${PYTHON_CMD:-conda run -n flashsvd python}"

(cd "${ROOT}/experiments/01_sharedness" && "${PYTHON_ARR[@]}" run_full_benchmark.py --help >/dev/null)
(cd "${ROOT}/experiments/01_sharedness" && "${PYTHON_ARR[@]}" sharedness_base.py --help >/dev/null)
"${PYTHON_ARR[@]}" "${ROOT}/experiments/01_sharedness/summarize_full_benchmark.py" --help >/dev/null
test -s "${ROOT}/paper_artifacts/h1_results/results/full_benchmark/H1_full_benchmark_summary.md"
"${PYTHON_ARR[@]}" "${ROOT}/experiments/01_sharedness/summarize_full_benchmark.py" \
  --results_dir "${ROOT}/paper_artifacts/h1_results/results/full_benchmark" \
  --out_dir /tmp/decodeshare_camera_ready_mock_h1 \
  --alpha 0.05 >/dev/null
test -s /tmp/decodeshare_camera_ready_mock_h1/H1_full_benchmark_summary.md
echo "h1_mock_ok"
