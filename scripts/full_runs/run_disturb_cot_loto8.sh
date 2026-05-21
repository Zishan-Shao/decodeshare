#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"

cd "${REPO_ROOT}/experiments/02_decode_ablation"

MODEL="${MODEL:-meta-llama/Llama-2-7b-chat-hf}"
MODEL_TAG="${MODEL_TAG:-$(echo "${MODEL}" | tr '/:' '__')}"
TASKS="${TASKS:-gsm8k,commonsenseqa,strategyqa,aqua,arc_challenge,openbookqa,qasc,logiqa}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/outputs/02_decode_ablation/loto}"
mkdir -p "${OUT_DIR}"

RUN_TAG="${RUN_TAG:-energy_balance_loto8_generation_eval${N_EVAL:-2048}_${MODEL_TAG}_layer${LAYER:-10}_seed${SEED:-42}}"

run_python_gpu run_loto_reasoning.py \
  --model "${MODEL}" \
  --device "${DEVICE}" \
  --model_dtype "${MODEL_DTYPE:-fp32}" \
  --mode "${MODE:-loto}" \
  --loto_eval_mode "${LOTO_EVAL_MODE:-heldout}" \
  --loto_only "${LOTO_ONLY:-}" \
  --tasks "${TASKS}" \
  --layer "${LAYER:-10}" \
  --n_subspace "${N_SUBSPACE:-128}" \
  --n_eval "${N_EVAL:-2048}" \
  --calib_decode_max_new_tokens "${CALIB_DECODE_MAX_NEW_TOKENS:-128}" \
  --per_task_max_states "${PER_TASK_MAX_STATES:-20000}" \
  --alpha_remove "${ALPHA_REMOVE:-1.0}" \
  --reasoning_tokens "${REASONING_TOKENS:-128}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-256}" \
  --batch_size "${BATCH_SIZE:-4}" \
  --max_prompt_len "${MAX_PROMPT_LEN:-512}" \
  --template_randomization "${TEMPLATE_RANDOMIZATION:-1}" \
  --shuffle_choices "${SHUFFLE_CHOICES:-1}" \
  --add_answer_prefix "${ADD_ANSWER_PREFIX:-1}" \
  --answer_prefix "${ANSWER_PREFIX:-${FINAL_ANSWER_PREFIX_DEFAULT}}" \
  --use_forced_choice "${USE_FORCED_CHOICE:-0}" \
  --fc_warmup_tokens "${FC_WARMUP_TOKENS:-0}" \
  --fc_prefix_mode "${FC_PREFIX_MODE:-auto}" \
  --fc_answer_prefix "${FC_ANSWER_PREFIX:-${FINAL_ANSWER_PREFIX_DEFAULT}}" \
  --do_sample "${DO_SAMPLE:-0}" \
  --bootstrap_iters "${BOOTSTRAP_ITERS:-5000}" \
  --perm_iters "${PERM_ITERS:-10000}" \
  --ci_alpha "${CI_ALPHA:-0.05}" \
  --seed "${SEED:-42}" \
  --out_json "${OUT_DIR}/${RUN_TAG}.json" \
  --out_md "${OUT_DIR}/${RUN_TAG}.md"
