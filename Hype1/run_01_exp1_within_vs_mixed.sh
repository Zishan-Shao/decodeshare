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
SEED=42

PCA_VAR=0.95
TAU=0.001
N_MIXED=50

mkdir -p results/exp1

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
  MODEL_TAG="$(echo "$MODEL_NAME" | tr '/:' '__')"
  ACTS_DIR="results/acts/${MODEL_TAG}/layer${LAYER}_n${N_PROMPTS}_new${CALIB_MAX_NEW_TOKENS}_maxlen${MAX_PROMPT_LEN}_states${PER_TASK_MAX_STATES}_seed${SEED}"

  OUT_CSV="results/exp1/${MODEL_TAG}_within_vs_mixed_pv${PCA_VAR}_tau${TAU}.csv"
  OUT_PNG="results/exp1/${MODEL_TAG}_within_vs_mixed_pv${PCA_VAR}_tau${TAU}.png"

  echo "[Exp1] model=${MODEL_NAME}"

  python exp1_within_vs_mixed.py \
    --acts_dir "${ACTS_DIR}" \
    --pca_var "${PCA_VAR}" \
    --tau "${TAU}" \
    --n_mixed "${N_MIXED}" \
    --seed 123 \
    --out_csv "${OUT_CSV}" \
    --out_png "${OUT_PNG}"
done
