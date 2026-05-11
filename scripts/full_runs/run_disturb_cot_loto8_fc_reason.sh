#!/usr/bin/env bash
set -euo pipefail

# Camera-ready LOTO forced-choice runner.
# Run this on Node0 or Node1 only unless the cluster availability changes.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKDIR="${ROOT}/experiments/02_decode_ablation"
cd "${WORKDIR}"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-flashsvd}"

export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

GPU_ID="${GPU_ID:-0}"
MODEL="${MODEL:-meta-llama/Llama-2-7b-chat-hf}"
TASKS="${TASKS:-gsm8k,commonsenseqa,strategyqa,aqua,arc_challenge,openbookqa,qasc,logiqa}"
OUT_DIR="${OUT_DIR:-${ROOT}/outputs/disturb_cot_reasoning}"
mkdir -p "${OUT_DIR}"

MODEL_TAG="$(echo "${MODEL}" | tr '/:' '__')"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python run_loto_reasoning.py \
  --model "${MODEL}" \
  --device cuda \
  --model_dtype fp32 \
  --mode loto \
  --loto_eval_mode heldout \
  --tasks "${TASKS}" \
  --layer "${LAYER:-10}" \
  --n_subspace "${N_SUBSPACE:-128}" \
  --n_eval "${N_EVAL:-2048}" \
  --calib_decode_max_new_tokens "${CALIB_DECODE_MAX_NEW_TOKENS:-128}" \
  --per_task_max_states "${PER_TASK_MAX_STATES:-20000}" \
  --reasoning_tokens "${REASONING_TOKENS:-128}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-256}" \
  --template_randomization 1 \
  --shuffle_choices 1 \
  --add_answer_prefix 1 \
  --answer_prefix $'\nFinal answer:' \
  --use_forced_choice 1 \
  --fc_warmup_tokens 0 \
  --fc_prefix_mode auto \
  --fc_answer_prefix $'\nFinal answer:' \
  --do_sample 0 \
  --out_json "${OUT_DIR}/energy_balance_loto8_reasoning_fc_eval2048_${MODEL_TAG}.json" \
  --out_md "${OUT_DIR}/energy_balance_loto8_reasoning_fc_eval2048_${MODEL_TAG}.md"
