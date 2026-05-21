#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"

cd "${REPO_ROOT}/experiments/01_sharedness"

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
MODEL_TAG="${MODEL_TAG:-$(echo "${MODEL}" | tr '/:' '__')}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/outputs/01_sharedness/full_benchmark}"
mkdir -p "${OUT_DIR}"

RUN_TAG="${RUN_TAG:-${MODEL_TAG}_layer${LAYER:-10}_seed${SEED:-42}}"

run_python_gpu run_full_benchmark.py \
  --model "${MODEL}" \
  --device "${DEVICE}" \
  --model_dtype "${MODEL_DTYPE:-fp32}" \
  --layer "${LAYER:-10}" \
  --n_prompts "${N_PROMPTS:-128}" \
  --calib_max_new_tokens "${CALIB_MAX_NEW_TOKENS:-128}" \
  --max_prompt_len "${MAX_PROMPT_LEN:-512}" \
  --per_task_max_states "${PER_TASK_MAX_STATES:-20000}" \
  --tau "${TAU:-0.001}" \
  --m_shared "${M_SHARED:-all}" \
  --null_perm_trials "${NULL_PERM_TRIALS:-2000}" \
  --null_scramble_trials "${NULL_SCRAMBLE_TRIALS:-100}" \
  --seed "${SEED:-42}" \
  --tasks "${TASKS:-all}" \
  --out_json "${OUT_DIR}/${RUN_TAG}.json" \
  --out_txt "${OUT_DIR}/${RUN_TAG}.txt"

if [[ "${SUMMARIZE:-1}" == "1" ]]; then
  run_python summarize_full_benchmark.py \
    --results_dir "${OUT_DIR}" \
    --out_dir "${OUT_DIR}" \
    --alpha "${SUMMARY_ALPHA:-0.05}"
fi
