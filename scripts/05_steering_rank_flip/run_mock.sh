#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"

run_python "${REPO_ROOT}/downstream/steering_rank_flip/exp_cross_method_rank_flip.py" --help >/dev/null
run_python "${REPO_ROOT}/downstream/steering_rank_flip/exp_diagnostic_rank_flip.py" --help >/dev/null
run_python "${REPO_ROOT}/downstream/steering_rank_flip/exp_trad_family_rank_flip.py" --help >/dev/null
PYTHON_CMD="${PYTHON_CMD}" DRY_RUN=1 bash "${REPO_ROOT}/scripts/reproduce_steering_flip_tables.sh" >/dev/null 2>&1
echo "steering_flip_mock_ok"
