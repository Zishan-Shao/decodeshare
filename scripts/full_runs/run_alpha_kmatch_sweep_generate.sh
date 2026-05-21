#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"

cd "${REPO_ROOT}/experiments/02_decode_ablation"

MODEL="${MODEL:-meta-llama/Llama-2-7b-chat-hf}"
MODEL_TAG="${MODEL_TAG:-$(echo "${MODEL}" | tr '/:' '__')}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/outputs/02_decode_ablation/energy_kmatch_generation}"
mkdir -p "${OUT_DIR}"

for ALPHA in ${ALPHAS:-0.25 0.5 0.75 1.0 1.25}; do
  ALPHA_TAG="${ALPHA/./p}"
  run_python_gpu run_energy_kmatch_generation.py \
    --model "${MODEL}" \
    --device "${DEVICE}" \
    --dtype "${MODEL_DTYPE:-fp32}" \
    --layer "${LAYER:-10}" \
    --n_prompts "${N_PROMPTS:-128}" \
    --calib_max_new_tokens "${CALIB_MAX_NEW_TOKENS:-128}" \
    --per_task_max_states "${PER_TASK_MAX_STATES:-20000}" \
    --pca_var "${PCA_VAR:-0.95}" \
    --tau "${TAU:-0.001}" \
    --m_shared "${M_SHARED:-all}" \
    --control_basis "${CONTROL_BASIS:-joint_nonshared_topk}" \
    --eval_n "${N_EVAL:-2048}" \
    --max_prompt_len "${MAX_PROMPT_LEN:-512}" \
    --batch_size "${BATCH_SIZE:-8}" \
    --bootstrap_iters "${BOOTSTRAP_ITERS:-5000}" \
    --perm_iters "${PERM_ITERS:-10000}" \
    --ci_alpha "${CI_ALPHA:-0.05}" \
    --seed "${SEED:-42}" \
    --use_chat_template "${USE_CHAT_TEMPLATE:-1}" \
    --alpha_shared_base "${ALPHA}" \
    --eval_tasks "${EVAL_TASKS:-commonsenseqa,strategyqa,aqua,arc_challenge,openbookqa,qasc,logiqa,boolq}" \
    --out_json "${OUT_DIR}/${MODEL_TAG}_layer${LAYER:-10}_alpha${ALPHA_TAG}.json" \
    --out_txt "${OUT_DIR}/${MODEL_TAG}_layer${LAYER:-10}_alpha${ALPHA_TAG}.txt"
done
