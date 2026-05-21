#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"

run_python "${REPO_ROOT}/downstream/rebuttal/orthogonal_steer/exp_A1_cross_method_rankflip.py" --help >/dev/null
run_python "${REPO_ROOT}/downstream/rebuttal/exp_ranking_flip_steering_layer28_rand100_full.py" --help >/dev/null
run_python "${REPO_ROOT}/downstream/rebuttal/exp_ranking_flip_trad_family.py" --help >/dev/null
PYTHON_CMD="${PYTHON_CMD}" DRY_RUN=1 bash "${REPO_ROOT}/scripts/reproduce_steering_flip_tables.sh" >/dev/null 2>&1
echo "steering_flip_mock_ok"
