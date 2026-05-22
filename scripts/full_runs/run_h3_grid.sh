#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"

cd "${REPO_ROOT}/experiments/04_prefill_decode"

MODEL="${MODEL:-meta-llama/Llama-2-7b-chat-hf}"
MODEL_TAG="${MODEL_TAG:-$(echo "${MODEL}" | tr '/:' '__')}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/outputs/04_prefill_decode/h3_grid}"
mkdir -p "${OUT_DIR}"

RUN_TAG="${RUN_TAG:-h3_grid_${MODEL_TAG}_layer${LAYER:-10}_seed${SEED:-0}}"

run_python_gpu run_h3_grid_reasoning.py \
  --model "${MODEL}" \
  --device "${DEVICE}" \
  --model_dtype "${MODEL_DTYPE:-fp32}" \
  --tasks "${TASKS:-gsm8k,commonsenseqa,strategyqa,piqa,arc_challenge,openbookqa,qasc,logiqa,boolq}" \
  --layer "${LAYER:-10}" \
  --n_subspace "${N_SUBSPACE:-128}" \
  --n_eval "${N_EVAL:-2048}" \
  --calib_decode_max_new_tokens "${CALIB_DECODE_MAX_NEW_TOKENS:-512}" \
  --per_task_max_states "${PER_TASK_MAX_STATES:-20000}" \
  --max_prompt_len "${MAX_PROMPT_LEN:-1024}" \
  --batch_size "${BATCH_SIZE:-8}" \
  --pca_var "${PCA_VAR:-0.95}" \
  --tau "${TAU:-0.001}" \
  --m_shared "${M_SHARED:-all}" \
  --answer_prefix "${ANSWER_PREFIX:-${FINAL_ANSWER_PREFIX_DEFAULT}}" \
  --warmup_tokens "${WARMUP_TOKENS:-0}" \
  --template_randomization "${TEMPLATE_RANDOMIZATION:-1}" \
  --shuffle_choices "${SHUFFLE_CHOICES:-1}" \
  --seed "${SEED:-0}" \
  --alpha_remove "${ALPHA_REMOVE:-1.0}" \
  --run_prefill_intervene "${RUN_PREFILL_INTERVENE:-1}" \
  --run_decode_intervene "${RUN_DECODE_INTERVENE:-1}" \
  --out_json "${OUT_DIR}/${RUN_TAG}.json"

if [[ "${SUMMARIZE:-1}" == "1" ]]; then
  run_python analysis/summarize_h3_grid.py \
    --inputs "${OUT_DIR}/${RUN_TAG}.json" \
    --out_csv "${OUT_DIR}/${RUN_TAG}.csv" \
    --out_latex "${OUT_DIR}/${RUN_TAG}.tex" \
    --latex_mode "${LATEX_MODE:-acc}"
fi
