#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ORIG="/home/zs89/decodeshare"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
read -r -a PYTHON_ARR <<< "${PYTHON_CMD:-conda run -n flashsvd python}"

(cd "${ROOT}/experiments/03_patchback" && "${PYTHON_ARR[@]}" subspace_patching_transfer.py --help >/dev/null)
(cd "${ROOT}/experiments/03_patchback" && "${PYTHON_ARR[@]}" openanswer_subspace_patching.py --help >/dev/null)
(cd "${ROOT}/experiments/03_patchback" && "${PYTHON_ARR[@]}" - <<'PY'
import math
from subspace_patching_transfer import summarize_scan_accuracy_counts

rows = [
    {"baseline": {"correct": True}, "ablated": {"correct": False}},
    {"baseline": {"correct": False}, "ablated": {"correct": False}},
    {"baseline": {"correct": False}, "ablated": {"correct": False}, "skipped_reason": "gold_not_in_candidates"},
]
out = summarize_scan_accuracy_counts(rows)
assert out["n_scanned"] == 3
assert out["n_effective"] == 2
assert out["n_skipped"] == 1
assert math.isclose(out["baseline_acc"], 0.5)
assert math.isclose(out["ablated_acc"], 0.0)
PY
)
test -s "${ORIG}/patch_back/paper/patchback_tables_all_models_all_layers.tex"
echo "h2_patchback_mock_ok"
