#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ORIG="/home/zs89/decodeshare"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
read -r -a PYTHON_ARR <<< "${PYTHON_CMD:-conda run -n flashsvd python}"

(cd "${ROOT}/experiments/03_patchback" && "${PYTHON_ARR[@]}" subspace_patching_transfer.py --help >/dev/null)
(cd "${ROOT}/experiments/03_patchback" && "${PYTHON_ARR[@]}" openanswer_subspace_patching.py --help >/dev/null)
test -s "${ORIG}/patch_back/paper/patchback_tables_all_models_all_layers.tex"
echo "h2_patchback_mock_ok"
