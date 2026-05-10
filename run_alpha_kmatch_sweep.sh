#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Env / paths
# -----------------------------
WORKDIR=""
cd "${WORKDIR}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate flashsvd

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# -----------------------------
# Output dirs
# -----------------------------
mkdir -p results/energy_kmatch_alpha_sweep

# -----------------------------
# Config
# -----------------------------
SCRIPT="reasoning/disturb_energy_matched_sharedness_kmatch.py"

MODEL_NAME="meta-llama/Llama-2-7b-chat-hf"
MODEL_TAG="${MODEL_NAME//\//_}"
LAYER=10

# data / compute knobs
TASKS="gsm8k,commonsenseqa,strategyqa,aqua,arc_challenge,openbookqa,qasc,logiqa,boolq"

N_SUBSPACE=128          # --n_prompts
N_EVAL=2048             # --eval_n

# NOTE: the new script uses --calib_max_new_tokens (not calib_decode_max_new_tokens)
CALIB_DECODE_STEPS=128  # --calib_max_new_tokens
PER_TASK_MAX_STATES=20000

PCA_VAR=0.95
TAU=0.001
M_SHARED="all"
K_EVAL=0                # 0 => use all shared comps; set e.g. 64 to pin k

MAX_PROMPT_LEN=512
BATCH=4
DTYPE="fp32"
SEED=42

# forced-choice probing details (new script arg names)
FC_PREFIX_MODE="auto"
ANSWER_PREFIX=$'\nFinal answer:'     # used for both dataset templates + forced-choice anchor in this script
WARMUP_TOKENS=128                      # new arg: --warmup_tokens
WARMUP_PHRASE=" Let's think step by step."

# alpha sweep list (shared vs ctrl_struct with same k)
ALPHAS="0,0.25,0.5,0.75,1.0,1.25,1.5,2.0"

# If 1: per-alpha k-match (k_c(alpha)) is computed (otherwise reuse k_c(alpha=1))
KMATCH_PER_ALPHA=1

# -----------------------------
# Run
# -----------------------------
TS="$(date +%Y%m%d_%H%M%S)"
OUT_PREFIX="results/energy_kmatch_alpha_sweep/${MODEL_TAG}_L${LAYER}_seed${SEED}_ts${TS}"

OUT_JSON="${OUT_PREFIX}.json"
OUT_TXT="${OUT_PREFIX}.txt"
OUT_MD="${OUT_PREFIX}.md"
OUT_TEX="${OUT_PREFIX}.tex"

echo "[Run] OUT_PREFIX=${OUT_PREFIX}"
echo "[Run] MODEL=${MODEL_NAME} LAYER=${LAYER} DTYPE=${DTYPE}" #CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[Run] TASKS=${TASKS}"
echo "[Run] ALPHAS=${ALPHAS} KMATCH_PER_ALPHA=${KMATCH_PER_ALPHA}"

CUDA_VISIBLE_DEVICES=0 python "${SCRIPT}" \
  --model "${MODEL_NAME}" \
  --device cuda \
  --dtype "${DTYPE}" \
  --layer "${LAYER}" \
  --use_chat_template 0 \
  --tasks "${TASKS}" \
  --n_prompts "${N_SUBSPACE}" \
  --eval_n "${N_EVAL}" \
  --max_prompt_len "${MAX_PROMPT_LEN}" \
  --batch_size "${BATCH}" \
  --calib_max_new_tokens "${CALIB_DECODE_STEPS}" \
  --per_task_max_states "${PER_TASK_MAX_STATES}" \
  --pca_var "${PCA_VAR}" \
  --tau "${TAU}" \
  --m_shared "${M_SHARED}" \
  --k_eval "${K_EVAL}" \
  --answer_prefix "${ANSWER_PREFIX}" \
  --fc_prefix_mode "${FC_PREFIX_MODE}" \
  --warmup_tokens "${WARMUP_TOKENS}" \
  --warmup_phrase "${WARMUP_PHRASE}" \
  --template_randomization 1 \
  --shuffle_choices 1 \
  --template_seed 1234 \
  --save_scores 0 \
  --alphas "${ALPHAS}" \
  --kmatch_per_alpha "${KMATCH_PER_ALPHA}" \
  --bootstrap_iters 5000 \
  --perm_iters 10000 \
  --ci_alpha 0.05 \
  --seed "${SEED}" \
  --out_json "${OUT_JSON}" \
  --out_txt "${OUT_TXT}" \
  --out_md "${OUT_MD}" \
  --out_tex "${OUT_TEX}"

echo ""
echo "[Done] Wrote:"
echo "  ${OUT_JSON}"
echo "  ${OUT_TXT}"
echo "  ${OUT_MD}"
echo "  ${OUT_TEX}"











# #!/usr/bin/env bash
# set -euo pipefail

# # -----------------------------
# # Env / paths
# # -----------------------------
# WORKDIR="src"
# cd "${WORKDIR}"

# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate flashsvd

# export TOKENIZERS_PARALLELISM=false
# export PYTHONUNBUFFERED=1
# export CUDA_VISIBLE_DEVICES=0

# # -----------------------------
# # Output dirs
# # -----------------------------
# mkdir -p results/alpha_sweep_decode_aligned
# mkdir -p results/kmatch

# # -----------------------------
# # Config
# # -----------------------------
# MODEL_NAME="meta-llama/Llama-2-7b-chat-hf"
# MODEL_TAG="${MODEL_NAME//\//_}"
# LAYER=10

# BATCH_SIZE=4
# MAX_PROMPT_LEN=512
# N_PROMPTS=128
# CALIB_MAX_NEW_TOKENS=128
# PER_TASK_MAX_STATES=20000
# EVAL_N=2048

# PCA_VAR=0.95
# TAU=0.001
# M_SHARED=all

# BOOTSTRAP_ITERS=5000
# PERM_ITERS=10000
# CI_ALPHA=0.05

# SEED=42
# USE_CHAT_TEMPLATE=1



# # -----------------------------
# # (1) Alpha sweep (decode-aligned alpha-scaling + forced-choice)
# # Note: each run ALSO computes ctrl_energy (K-match) internally.
# # -----------------------------
# ALPHAS=(0 0.25 0.5 0.75 1.0 1.25)

# for a in "${ALPHAS[@]}"; do
#   tag="${a/./p}"

#   CUDA_VISIBLE_DEVICES=1 python disturb_energy_matched_sharedness_kmatch.py \
#     --model "${MODEL_NAME}" --device cuda --dtype fp32 \
#     --layer "${LAYER}" \
#     --n_prompts "${N_PROMPTS}" \
#     --calib_max_new_tokens "${CALIB_MAX_NEW_TOKENS}" \
#     --per_task_max_states "${PER_TASK_MAX_STATES}" \
#     --pca_var "${PCA_VAR}" --tau "${TAU}" --m_shared "${M_SHARED}" \
#     --eval_n "${EVAL_N}" --max_prompt_len "${MAX_PROMPT_LEN}" \
#     --control_basis joint_nonshared_topk \
#     --batch_size "${BATCH_SIZE}" \
#     --bootstrap_iters "${BOOTSTRAP_ITERS}" --perm_iters "${PERM_ITERS}" --ci_alpha "${CI_ALPHA}" \
#     --seed "${SEED}" \
#     --use_chat_template "${USE_CHAT_TEMPLATE}" \
#     --alpha_shared_base "${a}" \
#     --out_json "results/alpha_sweep_decode_aligned/${MODEL_TAG}_layer${LAYER}_alphaShared_${tag}.json" \
#     --out_txt  "results/alpha_sweep_decode_aligned/${MODEL_TAG}_layer${LAYER}_alphaShared_${tag}.txt"
# done

# # -----------------------------
# # (2) K-match single run (alpha=1)
# # (Optional but nice for a clean headline K-match result)
# # -----------------------------
# CUDA_VISIBLE_DEVICES=1 python disturb_energy_matched_sharedness_kmatch.py \
#   --model "${MODEL_NAME}" --device cuda --dtype fp32 \
#   --layer "${LAYER}" \
#   --n_prompts "${N_PROMPTS}" --calib_max_new_tokens "${CALIB_MAX_NEW_TOKENS}" --per_task_max_states "${PER_TASK_MAX_STATES}" \
#   --pca_var "${PCA_VAR}" --tau "${TAU}" --m_shared "${M_SHARED}" \
#   --eval_n "${EVAL_N}" --max_prompt_len "${MAX_PROMPT_LEN}" \
#   --control_basis joint_nonshared_topk \
#   --batch_size "${BATCH_SIZE}" \
#   --bootstrap_iters "${BOOTSTRAP_ITERS}" --perm_iters "${PERM_ITERS}" --ci_alpha "${CI_ALPHA}" \
#   --seed "${SEED}" \
#   --use_chat_template "${USE_CHAT_TEMPLATE}" \
#   --alpha_shared_base 1.0 \
#   --out_json "results/kmatch/${MODEL_TAG}_layer${LAYER}_kmatch.json" \
#   --out_txt  "results/kmatch/${MODEL_TAG}_layer${LAYER}_kmatch.txt"
