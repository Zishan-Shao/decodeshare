#!/usr/bin/env bash
set -euo pipefail

WORKDIR="/home/zs89/decodeshare"
cd "${WORKDIR}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate flashsvd

SEEDS=1234,2345,3456,4567,5678,6789,7890,8901
MODEL=meta-llama/Llama-2-7b-chat-hf
AP=$'\nFinal answer:'
BASIS_DIR="results/rebuttal_mechanism/logit_lens_l26"
BASIS_NPZ="${BASIS_DIR}/basis_layer26_tseed1234.npz"

# 1) L26 basis (M1)
CUDA_VISIBLE_DEVICES=6 python rebuttal/mechanism/exp_1_logit_lens_vocab_signature.py \
  --model $MODEL --device cuda --dtype fp16 \
  --layer 26 --tasks commonsenseqa,arc_challenge,openbookqa,qasc,logiqa,boolq \
  --n_prompts 128 --seed 42 --template_seed 1234 --template_randomization 1 \
  --shuffle_choices 0 --add_answer_prefix 1 --answer_prefix "$AP" \
  --batch_size 4 --k_analyze 32 --topk 40 \
  --out_dir "${BASIS_DIR}"

test -f "${BASIS_NPZ}"


# 2) Exp-5 @L26 with Q26 (主实验)
CUDA_VISIBLE_DEVICES=6 python rebuttal/mechanism/exp_5_residual_path_attribution.py \
  --model $MODEL --device cuda --dtype fp16 \
  --layer 26 --alpha_remove 1.0 \
  --basis_npz "${BASIS_NPZ}" --k_basis 32 \
  --tasks commonsenseqa,arc_challenge,openbookqa,qasc,logiqa,boolq \
  --eval_n 256 --template_seeds $SEEDS --seed 42 \
  --fc_warmup_tokens 32 --fc_warmup_decoding greedy --fc_prefix_mode auto \
  --answer_prefix "$AP" --fc_answer_prefix "$AP" \
  --window_n 4 --add_mid_window 1 --exclude_final_step 1 \
  --include_residual_in 1 --include_residual_out 1 --include_sum_delta 1 --add_random_control 1 \
  --batch_size 6 --out_dir results/rebuttal_mechanism/m5_residual_path_l26 --tag n256_s8_q26



# 3) Exp-4 @L26 with Q26
CUDA_VISIBLE_DEVICES=6 python rebuttal/mechanism/exp_4_delta_attribution_attn_vs_mlp.py \
  --model $MODEL --device cuda --dtype fp16 \
  --layer 26 --alpha_remove 1.0 \
  --basis_npz "${BASIS_NPZ}" --k_basis 32 \
  --tasks commonsenseqa,arc_challenge,openbookqa,qasc,logiqa,boolq \
  --eval_n 256 --template_seeds $SEEDS --seed 42 \
  --fc_warmup_tokens 32 --fc_prefix_mode auto --answer_prefix "$AP" --fc_answer_prefix "$AP" \
  --window_n 4 --add_mid_window 1 --exclude_final_step 1 \
  --out_dir results/rebuttal_mechanism/m4_delta_attn_vs_mlp_l26 --tag n256_s8_q26



# 4) Exp-2 @L26 with Q26
CUDA_VISIBLE_DEVICES=6 python rebuttal/mechanism/exp_2_time_window_early_vs_late_enhanced.py \
  --model $MODEL --device cuda --dtype fp16 \
  --layer 26 --alpha_remove 1.0 \
  --basis_npz "${BASIS_NPZ}" --k_basis 32 \
  --tasks commonsenseqa,arc_challenge,openbookqa,qasc,logiqa,boolq \
  --eval_n 256 --template_seeds $SEEDS --seed 42 \
  --fc_warmup_tokens 32 --fc_prefix_mode auto --fc_answer_prefix "$AP" \
  --window_n 4 --add_mid_window 1 --exclude_final_step 1 \
  --out_dir results/rebuttal_mechanism/m2_time_window_l26 --tag eval256_s8_q26
