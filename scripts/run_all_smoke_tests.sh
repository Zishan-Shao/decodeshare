#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

tests=(
  "scripts/01_h1_sharedness/run_mock.sh"
  "scripts/02_h2_decode_ablation/run_mock.sh"
  "scripts/03_h2_patchback/run_mock.sh"
  "scripts/04_h3_prefill_decode/run_mock.sh"
  "scripts/05_steering_repair/run_mock.sh"
)

for test_script in "${tests[@]}"; do
  echo "==> ${test_script}"
  bash "$test_script"
done

echo "all_smoke_tests_ok"
