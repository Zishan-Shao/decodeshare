#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ "${RUN_LOTO:-1}" == "1" ]]; then
  bash scripts/full_runs/run_disturb_cot_loto8_fc_reason.sh
fi

if [[ "${RUN_ENERGY:-1}" == "1" ]]; then
  bash scripts/full_runs/run_alpha_kmatch_sweep.sh
fi
