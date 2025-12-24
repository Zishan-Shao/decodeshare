#!/usr/bin/env bash
set -euo pipefail

conda activate flashsvd
cd /home/zs89/FlashSVDTrain/shared_space/causal_analysis

# 建议关掉 tokenizer 并行提示
export TOKENIZERS_PARALLELISM=false

GPU_ID="${GPU_ID:-1}"

# models（用数组，不要反复覆盖 MODEL_NAME）
MODEL_NAMES=(
  "meta-llama/Llama-2-7b-chat-hf"
  "meta-llama/Llama-2-7b-hf"
  "facebook/opt-6.7b"
  "Qwen/Qwen2.5-7B"
  "Qwen/Qwen2.5-7B-Instruct"
  "google/gemma-3-12b-it"
  "google/gemma-2-2b-it"
)

# sweeps（bash 数组写法）
LAYERS=(2 8 10 18 20 24)
TAUS=(0.001 0.01 0.1)
N_PROMPTS_LIST=(128 256 512)
CALIB_MAX_NEW_TOKENS_LIST=(128 256 512)
MAX_PROMPT_LEN_LIST=(512)
PER_TASK_MAX_STATES_LIST=(20000)

NULL_PERM_TRIALS=2000
NULL_SCRAMBLE_TRIALS=100   # 你原来的 TRIALS=100 实际对应这个

mkdir -p results/exists

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
  # 文件名里不要出现 / ，否则会变成目录
  MODEL_TAG="$(echo "$MODEL_NAME" | tr '/:' '__')"

  for LAYER in "${LAYERS[@]}"; do
    for TAU in "${TAUS[@]}"; do
      for N_PROMPTS in "${N_PROMPTS_LIST[@]}"; do
        for CALIB_MAX_NEW_TOKENS in "${CALIB_MAX_NEW_TOKENS_LIST[@]}"; do
          for MAX_PROMPT_LEN in "${MAX_PROMPT_LEN_LIST[@]}"; do
            for PER_TASK_MAX_STATES in "${PER_TASK_MAX_STATES_LIST[@]}"; do

              OUT_JSON="results/exists/prove_existence_${MODEL_TAG}_fp32_layer${LAYER}_n${N_PROMPTS}_new${CALIB_MAX_NEW_TOKENS}_maxlen${MAX_PROMPT_LEN}_states${PER_TASK_MAX_STATES}_tau${TAU}_msharedall_perm${NULL_PERM_TRIALS}_scr${NULL_SCRAMBLE_TRIALS}.json"
              OUT_TXT="${OUT_JSON%.json}.txt"

              echo "[Run] model=${MODEL_NAME} layer=${LAYER} tau=${TAU} n=${N_PROMPTS} new=${CALIB_MAX_NEW_TOKENS} maxlen=${MAX_PROMPT_LEN} states=${PER_TASK_MAX_STATES}"

                python prove_sharedness_decode_fair.py \
                --model "${MODEL_NAME}" \
                --device cuda \
                --model_dtype fp32 \
                --layer "${LAYER}" \
                --n_prompts "${N_PROMPTS}" \
                --calib_max_new_tokens "${CALIB_MAX_NEW_TOKENS}" \
                --max_prompt_len "${MAX_PROMPT_LEN}" \
                --per_task_max_states "${PER_TASK_MAX_STATES}" \
                --tau "${TAU}" \
                --m_shared all \
                --null_perm_trials "${NULL_PERM_TRIALS}" \
                --null_scramble_trials "${NULL_SCRAMBLE_TRIALS}" \
                --out_json "${OUT_JSON}" \
                --out_txt "${OUT_TXT}"

            done
          done
        done
      done
    done
  done
done














# #!/bin/bash

# cd /home/zs89/FlashSVDTrain/shared_space/causal_analysis

# MODEL_NAME="meta-llama/Llama-2-7b-chat-hf"
# MODEL_NAME="meta-llama/Llama-2-7b-hf"
# MODEL_NAME="facebook/opt-6.7b"
# MODEL_NAME="Qwen/Qwen2.5-7B"
# MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
# MODEL_NAME="google/gemma-3-12b-it"
# MODEL_NAME="google/gemma-2-2b-it"

# TRIALS=100
# LAYER=[2,8,10,18,20,24]
# TAU=[0.001,0.01,0.1]
# N_PROMPTS=[128,256,512]
# CALIB_MAX_NEW_TOKENS=[128,256,512]
# MAX_PROMPT_LEN=[512,1024,2048]
# PER_TASK_MAX_STATES=[20000,40000,60000]


# OUT_JSON="results/exists/prove_existence_${MODEL_NAME}_fp32_${LAYER}_128_128_512_20000_0.001_all_2000_${TRIALS}.json"

# CUDA_VISIBLE_DEVICES=1 python prove_sharedness_decode_fair.py \
#     --model $MODEL_NAME \
#     --device cuda \
#     --model_dtype fp32 \
#     --layer ${LAYER} \
#     --n_prompts ${N_PROMPTS} \
#     --calib_max_new_tokens ${CALIB_MAX_NEW_TOKENS} \
#     --max_prompt_len ${MAX_PROMPT_LEN} \
#     --per_task_max_states ${PER_TASK_MAX_STATES} \
#     --tau ${TAU} \
#     --m_shared all \
#     --null_perm_trials 2000 \
#     --null_scramble_trials 100
#     --out_json $OUT_JSON



