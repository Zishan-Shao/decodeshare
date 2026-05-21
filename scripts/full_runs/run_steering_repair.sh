#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"

cd "${REPO_ROOT}/experiments/05_steering_repair"

OUT_DIR="${OUT_DIR:-${REPO_ROOT}/outputs/05_steering_repair/multibench}"
mkdir -p "${OUT_DIR}"

run_python_gpu steering_vector_reliability_multibench_patch_v3.py \
  --model "${MODEL:-meta-llama/Llama-2-7b-chat-hf}" \
  --device "${DEVICE}" \
  --dtype "${MODEL_DTYPE:-fp32}" \
  --layer "${LAYER:-10}" \
  --seed "${SEED:-0}" \
  --tasks "${TASKS:-boolq,rte,sst2}" \
  --calib_per_class "${CALIB_PER_CLASS:-256}" \
  --eval_per_class "${EVAL_PER_CLASS:-128}" \
  --basis_source "${BASIS_SOURCE:-neutral}" \
  --basis_k "${BASIS_K:-512}" \
  --basis_max_states "${BASIS_MAX_STATES:-1024}" \
  --betas "${BETAS:-0,0.25,0.5,0.75,1.0}" \
  --lambdas "${LAMBDAS:-0,0.5,1.0}" \
  --n_rand "${N_RAND:-5}" \
  --cand_calib_per_class "${CAND_CALIB_PER_CLASS:-32}" \
  --cand_calib_templates "${CAND_CALIB_TEMPLATES:-all}" \
  --out_dir "${OUT_DIR}" \
  --show_per_template "${SHOW_PER_TEMPLATE:-1}"

if [[ "${SUMMARIZE:-1}" == "1" ]]; then
  run_python summarize_multibench_v3_full.py \
    --root_dir "${OUT_DIR}" \
    --out_dir "${OUT_DIR}/summary_pack"
fi
