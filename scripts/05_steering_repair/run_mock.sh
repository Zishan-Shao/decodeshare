#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"

(cd "${REPO_ROOT}/experiments/05_steering_repair" && run_python steering_vector_reliability_multibench_patch_v3.py --help >/dev/null)
run_python "${REPO_ROOT}/experiments/05_steering_repair/summarize_multibench_v3_full.py" --help >/dev/null
echo "steering_repair_mock_ok"
