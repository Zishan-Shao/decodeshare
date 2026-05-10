#!/usr/bin/env bash
set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate flashsvd
cd Hype1

export TOKENIZERS_PARALLELISM=false

GPU_ID="${GPU_ID:-1}"
export CUDA_VISIBLE_DEVICES=0

MODEL_NAMES=(
  "meta-llama/Llama-2-7b-chat-hf"
  "Qwen/Qwen2.5-7B-Instruct"
)

LAYER=10
N_PROMPTS=128
CALIB_MAX_NEW_TOKENS=256
MAX_PROMPT_LEN=512
PER_TASK_MAX_STATES=20000
BATCH_SIZE=4
SEED=42

SAVE_DTYPE="fp32"   # fp16 更省盘；如需完全对齐可改 fp32

mkdir -p results/acts

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
  MODEL_TAG="$(echo "$MODEL_NAME" | tr '/:' '__')"
  OUT_DIR="results/acts/${MODEL_TAG}/layer${LAYER}_n${N_PROMPTS}_new${CALIB_MAX_NEW_TOKENS}_maxlen${MAX_PROMPT_LEN}_states${PER_TASK_MAX_STATES}_seed${SEED}"

  mkdir -p "${OUT_DIR}"

  echo "[Collect] model=${MODEL_NAME} -> ${OUT_DIR}"

  python collect_decode_acts.py \
    --model "${MODEL_NAME}" \
    --device cuda \
    --model_dtype fp32 \
    --layer "${LAYER}" \
    --n_prompts "${N_PROMPTS}" \
    --calib_max_new_tokens "${CALIB_MAX_NEW_TOKENS}" \
    --max_prompt_len "${MAX_PROMPT_LEN}" \
    --per_task_max_states "${PER_TASK_MAX_STATES}" \
    --batch_size "${BATCH_SIZE}" \
    --seed "${SEED}" \
    --out_dir "${OUT_DIR}" \
    --save_dtype "${SAVE_DTYPE}" \
    --out_txt "${OUT_DIR}/collect.txt" \
    --overwrite
done
