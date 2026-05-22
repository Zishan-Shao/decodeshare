#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

cd "${REPO_ROOT}"

OUT_DIR="${OUT_DIR:-${REPO_ROOT}/outputs/05_steering_flip_tables}"
MODEL_NAME="${MODEL:-meta-llama/Llama-2-7b-chat-hf}"
MODEL_DTYPE="${MODEL_DTYPE:-fp32}"
LAYER="${LAYER:-28}"
TASKS="${TASKS:-commonsenseqa,arc_challenge,openbookqa,qasc,logiqa}"
N_EVAL="${N_EVAL:-128}"
TEMPLATE_SEEDS_RANK="${TEMPLATE_SEEDS_RANK:-1234,2345,3456}"
TEMPLATE_SEEDS_REAL="${TEMPLATE_SEEDS_REAL:-4567,5678,6789}"
SEED="${SEED:-42}"
DECODING="${DECODING:-greedy}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
REASONING_TOKENS="${REASONING_TOKENS:-128}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-512}"
SAMPLE_SEED="${SAMPLE_SEED:-12345}"
STAGED="${STAGED:-1}"
AGG="${AGG:-mean}"

N_VEC_CAA="${N_VEC_CAA:-32}"
N_VEC_INSTR="${N_VEC_INSTR:-64}"
N_VEC_SAE="${N_VEC_SAE:-64}"
SUBSET_SIZE="${SUBSET_SIZE:-96}"
N_CORPUS="${N_CORPUS:-512}"
SAE_TRAIN_SAMPLES="${SAE_TRAIN_SAMPLES:-20000}"
SAE_LATENT_DIM="${SAE_LATENT_DIM:-8192}"
SAE_STEPS="${SAE_STEPS:-3000}"

N_DIAG="${N_DIAG:-100}"
VECTOR_SEED="${VECTOR_SEED:-0}"
K_LIST="${K_LIST:-1}"
DO_TRAD_NOCACHE="${DO_TRAD_NOCACHE:-0}"

RUN_CROSS_METHOD="${RUN_CROSS_METHOD:-1}"
RUN_DIAGNOSTIC="${RUN_DIAGNOSTIC:-1}"

RANKING_DIR="${OUT_DIR}/ranking_alignment"
DEPLOY_DIR="${OUT_DIR}/deployment_selection"
DIAG_JSON="${DEPLOY_DIR}/ranking_flip_layer${LAYER}_rand${N_DIAG}.json"
DIAG_MANIFEST="${DEPLOY_DIR}/steering_vectors_layer${LAYER}_rand${N_DIAG}.jsonl"
TRAD_FAMILY_JSON="${DEPLOY_DIR}/ranking_flip_trad_family.json"

mkdir -p "${RANKING_DIR}" "${DEPLOY_DIR}"

if [[ "${RUN_CROSS_METHOD}" == "1" ]]; then
  run_python_gpu downstream/steering_rank_flip/exp_cross_method_rank_flip.py \
    --model "${MODEL_NAME}" \
    --device "${DEVICE}" \
    --model_dtype "${MODEL_DTYPE}" \
    --layer "${LAYER}" \
    --tasks_eval "${TASKS}" \
    --n_eval "${N_EVAL}" \
    --template_seeds_rank "${TEMPLATE_SEEDS_RANK}" \
    --template_seeds_real "${TEMPLATE_SEEDS_REAL}" \
    --decoding "${DECODING}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --reasoning_tokens "${REASONING_TOKENS}" \
    --batch_size "${BATCH_SIZE}" \
    --max_prompt_len "${MAX_PROMPT_LEN}" \
    --sample_seed "${SAMPLE_SEED}" \
    --staged "${STAGED}" \
    --agg "${AGG}" \
    --n_vec_caa "${N_VEC_CAA}" \
    --n_vec_instr "${N_VEC_INSTR}" \
    --n_vec_sae "${N_VEC_SAE}" \
    --subset_size "${SUBSET_SIZE}" \
    --n_corpus "${N_CORPUS}" \
    --sae_train_samples "${SAE_TRAIN_SAMPLES}" \
    --sae_latent_dim "${SAE_LATENT_DIM}" \
    --sae_steps "${SAE_STEPS}" \
    --out_dir "${RANKING_DIR}" \
    --seed "${SEED}"
fi

if [[ "${RUN_DIAGNOSTIC}" == "1" ]]; then
  run_python_gpu downstream/steering_rank_flip/exp_diagnostic_rank_flip.py \
    --model "${MODEL_NAME}" \
    --device "${DEVICE}" \
    --model_dtype "${MODEL_DTYPE}" \
    --layer "${LAYER}" \
    --n_vectors "${N_DIAG}" \
    --vector_seed "${VECTOR_SEED}" \
    --tasks "${TASKS}" \
    --n_eval "${N_EVAL}" \
    --template_seeds_rank "${TEMPLATE_SEEDS_RANK}" \
    --template_seeds_real "${TEMPLATE_SEEDS_REAL}" \
    --decoding "${DECODING}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --reasoning_tokens "${REASONING_TOKENS}" \
    --batch_size "${BATCH_SIZE}" \
    --max_prompt_len "${MAX_PROMPT_LEN}" \
    --sample_seed "${SAMPLE_SEED}" \
    --staged "${STAGED}" \
    --agg "${AGG}" \
    --seed "${SEED}" \
    --out_dir "${DEPLOY_DIR}" \
    --out_json "${DIAG_JSON}"

  run_python_gpu downstream/steering_rank_flip/exp_trad_family_rank_flip.py \
    --base_json "${DIAG_JSON}" \
    --vectors_manifest "${DIAG_MANIFEST}" \
    --model "${MODEL_NAME}" \
    --device "${DEVICE}" \
    --model_dtype "${MODEL_DTYPE}" \
    --max_vectors 0 \
    --tasks "${TASKS}" \
    --n_eval "${N_EVAL}" \
    --template_seeds_rank "${TEMPLATE_SEEDS_RANK}" \
    --template_seeds_real "${TEMPLATE_SEEDS_REAL}" \
    --decoding "${DECODING}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --reasoning_tokens "${REASONING_TOKENS}" \
    --batch_size "${BATCH_SIZE}" \
    --max_prompt_len "${MAX_PROMPT_LEN}" \
    --sample_seed "${SAMPLE_SEED}" \
    --staged "${STAGED}" \
    --decode_mode decode \
    --agg "${AGG}" \
    --do_trad_both 1 \
    --do_trad_nocache "${DO_TRAD_NOCACHE}" \
    --k_list "${K_LIST}" \
    --seed "${SEED}" \
    --out_json "${TRAD_FAMILY_JSON}"
fi
