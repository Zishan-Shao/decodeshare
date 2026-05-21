#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"

(cd "${REPO_ROOT}/experiments/01_sharedness" && run_python run_full_benchmark.py --help >/dev/null)
(cd "${REPO_ROOT}/experiments/01_sharedness" && run_python sharedness_base.py --help >/dev/null)
run_python "${REPO_ROOT}/experiments/01_sharedness/summarize_full_benchmark.py" --help >/dev/null
echo "h1_mock_ok"
