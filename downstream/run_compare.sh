#!/usr/bin/env bash
set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"

conda activate svdllm
cd downstream

# 建议关掉 tokenizer 并行提示
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HOME/.cache/hf_datasets_clean}"


# 1) Baseline whitening
python svdllm_vendor/SVDLLM.py \
  --step 1 \
  --model meta-llama/Llama-2-7b-chat-hf \
  --ratio 0.2 \
  --dataset ./calib_mix.jsonl \
  --seed 0 \
  --whitening_nsamples 128 \
  --model_seq_len 2048 \
  --save_path ./outputs \
  --whitening_style baseline


# 2) DecodeShare whitening（同一个 calib_mix.jsonl）
python svdllm_vendor/SVDLLM.py \
  --step 1 \
  --model meta-llama/Llama-2-7b-chat-hf \
  --ratio 0.2 \
  --dataset ./calib_mix.jsonl \
  --seed 0 \
  --whitening_nsamples 128 \
  --model_seq_len 2048 \
  --save_path ./outputs \
  --whitening_style decodeshare \
  --decodeshare_basis_dir ./decodeshare_basis \
  --decodeshare_alpha 2.0 \
  --decodeshare_scale_mode mean_diag
