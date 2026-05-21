#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"

(cd "${REPO_ROOT}/experiments/03_patchback" && run_python subspace_patching_transfer.py --help >/dev/null)
(cd "${REPO_ROOT}/experiments/03_patchback" && run_python openanswer_subspace_patching.py --help >/dev/null)
(cd "${REPO_ROOT}/experiments/03_patchback" && run_python - <<'PY'
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
echo "h2_patchback_mock_ok"
