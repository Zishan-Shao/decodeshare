#!/usr/bin/env bash
set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate flashsvd
cd /home/zs89/decodeshare/Hype1

export TOKENIZERS_PARALLELISM=false

GPU_ID="${GPU_ID:-1}"
export CUDA_VISIBLE_DEVICES=2

LLAMA_MODEL="meta-llama/Llama-2-7b-chat-hf"

LAYER=10
N_PROMPTS=128
CALIB_MAX_NEW_TOKENS=256
MAX_PROMPT_LEN=512
PER_TASK_MAX_STATES=20000
SEED=42

PCA_VARS="0.8,0.9,0.95,0.97,0.99"
TAUS="1e-4,2e-4,5e-4,1e-3,2e-3,5e-3,1e-2"

MODEL_TAG="$(echo "$LLAMA_MODEL" | tr '/:' '__')"
ACTS_DIR="results/acts/${MODEL_TAG}/layer${LAYER}_n${N_PROMPTS}_new${CALIB_MAX_NEW_TOKENS}_maxlen${MAX_PROMPT_LEN}_states${PER_TASK_MAX_STATES}_seed${SEED}"

mkdir -p results/exp3

OUT_CSV="results/exp3/${MODEL_TAG}_sens_pv_${PCA_VARS}_tau_${TAUS}.csv"
OUT_PNG="results/exp3/${MODEL_TAG}_sens_pv_${PCA_VARS}_tau_${TAUS}.png"

echo "[Exp3] llama=${LLAMA_MODEL}"

python exp3_sensitivity_sweep.py \
  --acts_dir "${ACTS_DIR}" \
  --pca_vars "${PCA_VARS}" \
  --taus "${TAUS}" \
  --out_csv "${OUT_CSV}" \
  --out_png "${OUT_PNG}" \
  --seed 123
