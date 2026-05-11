#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/zs89/decodeshare"
RESULT_DIR="${ROOT_DIR}/results/rebuttal_scaling/llama2_70b_eval_sampled_t07_v1"
LOG_DIR="${RESULT_DIR}/logs"
LOG_PATH="${LOG_DIR}/exp_A3_eval_saved_basis_layer25_csqa32_sample_t07.log"

VISIBLE_GPU="${VISIBLE_GPU:-1}"
WAIT_GPU_INDEX="${WAIT_GPU_INDEX:-1}"
WAIT_MAX_USED_MIB="${WAIT_MAX_USED_MIB:-1024}"
POLL_SECONDS="${POLL_SECONDS:-60}"

mkdir -p "${LOG_DIR}"

wait_for_gpu() {
  while true; do
    local used
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${WAIT_GPU_INDEX}" | tr -d ' ')"
    if [[ -n "${used}" ]] && [[ "${used}" =~ ^[0-9]+$ ]] && (( used <= WAIT_MAX_USED_MIB )); then
      break
    fi
    echo "[Wait] GPU ${WAIT_GPU_INDEX} still busy: used=${used} MiB threshold=${WAIT_MAX_USED_MIB} MiB" | tee -a "${LOG_PATH}"
    sleep "${POLL_SECONDS}"
  done
}

echo "[Start] $(date '+%Y-%m-%d %H:%M:%S %Z')" | tee -a "${LOG_PATH}"
echo "[Config] visible_gpu=${VISIBLE_GPU} wait_gpu_index=${WAIT_GPU_INDEX} threshold_mib=${WAIT_MAX_USED_MIB}" | tee -a "${LOG_PATH}"
wait_for_gpu
echo "[Launch] $(date '+%Y-%m-%d %H:%M:%S %Z')" | tee -a "${LOG_PATH}"

export CUDA_VISIBLE_DEVICES="${VISIBLE_GPU}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

python "${ROOT_DIR}/rebuttal/mechanism/PartA/exp_A3_eval_saved_basis.py" \
  --basis_npz "${ROOT_DIR}/results/rebuttal_scaling/llama2_70b_a3_smoke_v3/exp_A3_bases_layer25_llama2_70b_l25_smoke_v3.npz" \
  --model "meta-llama/Llama-2-70b-chat-hf" \
  --device cuda \
  --dtype fp16 \
  --device_map auto \
  --max_memory_map "0:76,cpu:320" \
  --cpu_offload_gb 320 \
  --layer 25 \
  --tasks_eval "commonsenseqa" \
  --conditions "baseline,shared,ctrl_energy,rand_energy" \
  --protocol generation \
  --eval_n 32 \
  --batch_size 1 \
  --max_prompt_len 512 \
  --template_randomization 1 \
  --template_seed 1234 \
  --shuffle_choices 0 \
  --add_answer_prefix 1 \
  --answer_prefix $'\nFinal answer:' \
  --decoding sample \
  --temperature 0.7 \
  --top_p 0.9 \
  --top_k 0 \
  --max_new_tokens 32 \
  --bootstrap_iters 500 \
  --perm_iters 1000 \
  --out_dir "${RESULT_DIR}" \
  --tag "layer25_csqa32_sample_t07" \
  2>&1 | tee -a "${LOG_PATH}"
