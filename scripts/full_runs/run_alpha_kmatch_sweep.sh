#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"

cd "${REPO_ROOT}"

MODEL="${MODEL:-meta-llama/Llama-2-7b-chat-hf}"
MODEL_TAG="${MODEL_TAG:-$(echo "${MODEL}" | tr '/:' '__')}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/outputs/02_decode_ablation/energy_kmatch_alpha_sweep}"
mkdir -p "${OUT_DIR}"

RUN_TAG="${RUN_TAG:-${MODEL_TAG}_layer${LAYER:-10}_seed${SEED:-42}_alpha_kmatch}"

run_python_gpu experiments/02_decode_ablation/run_energy_kmatch_reasoning.py \
  --model "${MODEL}" \
  --device "${DEVICE}" \
  --dtype "${MODEL_DTYPE:-fp32}" \
  --layer "${LAYER:-10}" \
  --use_chat_template "${USE_CHAT_TEMPLATE:-0}" \
  --tasks "${TASKS:-gsm8k,commonsenseqa,strategyqa,aqua,arc_challenge,openbookqa,qasc,logiqa,boolq}" \
  --n_prompts "${N_PROMPTS:-128}" \
  --eval_n "${N_EVAL:-2048}" \
  --max_prompt_len "${MAX_PROMPT_LEN:-512}" \
  --batch_size "${BATCH_SIZE:-4}" \
  --calib_max_new_tokens "${CALIB_MAX_NEW_TOKENS:-128}" \
  --per_task_max_states "${PER_TASK_MAX_STATES:-20000}" \
  --pca_var "${PCA_VAR:-0.95}" \
  --tau "${TAU:-0.001}" \
  --m_shared "${M_SHARED:-all}" \
  --k_eval "${K_EVAL:-0}" \
  --answer_prefix "${ANSWER_PREFIX:-${FINAL_ANSWER_PREFIX_DEFAULT}}" \
  --fc_prefix_mode "${FC_PREFIX_MODE:-auto}" \
  --warmup_tokens "${WARMUP_TOKENS:-128}" \
  --warmup_phrase "${WARMUP_PHRASE:- Let us think step by step.}" \
  --template_randomization "${TEMPLATE_RANDOMIZATION:-1}" \
  --shuffle_choices "${SHUFFLE_CHOICES:-1}" \
  --template_seed "${TEMPLATE_SEED:-1234}" \
  --save_scores "${SAVE_SCORES:-0}" \
  --alphas "${ALPHAS:-0,0.25,0.5,0.75,1.0,1.25,1.5,2.0}" \
  --kmatch_per_alpha "${KMATCH_PER_ALPHA:-1}" \
  --bootstrap_iters "${BOOTSTRAP_ITERS:-5000}" \
  --perm_iters "${PERM_ITERS:-10000}" \
  --ci_alpha "${CI_ALPHA:-0.05}" \
  --seed "${SEED:-42}" \
  --out_json "${OUT_DIR}/${RUN_TAG}.json" \
  --out_txt "${OUT_DIR}/${RUN_TAG}.txt" \
  --out_md "${OUT_DIR}/${RUN_TAG}.md" \
  --out_tex "${OUT_DIR}/${RUN_TAG}.tex"
