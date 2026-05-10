

#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Env / paths
# -----------------------------
WORKDIR="src"
cd "${WORKDIR}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate flashsvd

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# -----------------------------
# Output dirs
# -----------------------------
#mkdir -p results/alpha_sweep_decode_aligned
mkdir -p results/kmatch

# -----------------------------
# Config
# -----------------------------
MODEL_NAME="meta-llama/Llama-2-7b-chat-hf"
MODEL_TAG="${MODEL_NAME//\//_}"
LAYER=10

BATCH_SIZE=8
MAX_PROMPT_LEN=512
N_PROMPTS=128
CALIB_MAX_NEW_TOKENS=128
PER_TASK_MAX_STATES=20000
EVAL_N=2048

PCA_VAR=0.95
TAU=0.001
M_SHARED=all

BOOTSTRAP_ITERS=5000
PERM_ITERS=10000
CI_ALPHA=0.05

SEED=42
USE_CHAT_TEMPLATE=1



# -----------------------------
# (1) Alpha sweep (decode-aligned alpha-scaling + forced-choice)
# Note: each run ALSO computes ctrl_energy (K-match) internally.
# -----------------------------
ALPHAS=(0.25 0.5 0.75 1.0 1.25)

for a in "${ALPHAS[@]}"; do
  tag="${a/./p}"

  CUDA_VISIBLE_DEVICES=3 python disturb_energy_matched_sharedness_kmatch.py \
    --model "${MODEL_NAME}" --device cuda --dtype fp32 \
    --layer "${LAYER}" \
    --n_prompts "${N_PROMPTS}" \
    --calib_max_new_tokens "${CALIB_MAX_NEW_TOKENS}" \
    --per_task_max_states "${PER_TASK_MAX_STATES}" \
    --pca_var "${PCA_VAR}" --tau "${TAU}" --m_shared "${M_SHARED}" \
    --eval_n "${EVAL_N}" --max_prompt_len "${MAX_PROMPT_LEN}" \
    --control_basis joint_nonshared_topk \
    --batch_size "${BATCH_SIZE}" \
    --bootstrap_iters "${BOOTSTRAP_ITERS}" --perm_iters "${PERM_ITERS}" --ci_alpha "${CI_ALPHA}" \
    --seed "${SEED}" \
    --use_chat_template "${USE_CHAT_TEMPLATE}" \
    --alpha_shared_base "${a}" \
    --out_json "results/alpha_sweep_decode_aligned/${MODEL_TAG}_layer${LAYER}_alphaShared_${tag}.json" \
    --out_txt  "results/alpha_sweep_decode_aligned/${MODEL_TAG}_layer${LAYER}_alphaShared_${tag}.txt"
done

# -----------------------------
# (2) K-match single run (alpha=1)
# (Optional but nice for a clean headline K-match result)
# -----------------------------
CUDA_VISIBLE_DEVICES=3 python disturb_energy_matched_sharedness_kmatch.py \
  --model "${MODEL_NAME}" --device cuda --dtype fp32 \
  --layer "${LAYER}" \
  --n_prompts "${N_PROMPTS}" --calib_max_new_tokens "${CALIB_MAX_NEW_TOKENS}" --per_task_max_states "${PER_TASK_MAX_STATES}" \
  --pca_var "${PCA_VAR}" --tau "${TAU}" --m_shared "${M_SHARED}" \
  --eval_n "${EVAL_N}" --max_prompt_len "${MAX_PROMPT_LEN}" \
  --control_basis joint_nonshared_topk \
  --batch_size "${BATCH_SIZE}" \
  --bootstrap_iters "${BOOTSTRAP_ITERS}" --perm_iters "${PERM_ITERS}" --ci_alpha "${CI_ALPHA}" \
  --seed "${SEED}" \
  --use_chat_template "${USE_CHAT_TEMPLATE}" \
  --alpha_shared_base 1.0 \
  --out_json "results/kmatch/${MODEL_TAG}_layer${LAYER}_kmatch.json" \
  --out_txt  "results/kmatch/${MODEL_TAG}_layer${LAYER}_kmatch.txt"
