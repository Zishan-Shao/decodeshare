#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"

(cd "${REPO_ROOT}/experiments/04_prefill_decode" && run_python run_h3_grid_reasoning.py --help >/dev/null)
run_python "${REPO_ROOT}/experiments/04_prefill_decode/analysis/summarize_h3_grid.py" --help >/dev/null
echo "h3_mock_ok"
