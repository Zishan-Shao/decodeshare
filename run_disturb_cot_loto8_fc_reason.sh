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







# mistralai/Mistral-7B-Instruct-v0.3
# run the with-aqua version
# Single (all-tasks) basis estimation + evaluation:
CUDA_VISIBLE_DEVICES=3 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
  --model mistralai/Mistral-7B-Instruct-v0.3 --device cuda --model_dtype fp32 \
  --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
  --mode all --batch_size 16 \
  --n_subspace 128 --n_eval 2048 --layer 4 \
  --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
  --reasoning_tokens 128 --max_new_tokens 256 \
  --template_randomization 1 --shuffle_choices 1 \
  --do_sample 0 --out_json results/disturb_cot_reason_appendix/all_tasks_energy_balance_greedy_results_Mistral-7B-Instruct-v0.3_layer4.json --out_md results/disturb_cot_reason_appendix/all_tasks_energy_balance_greedy_summary_Mistral-7B-Instruct-v0.3_layer4.md

# LOTO (estimate basis on N-1 tasks, evaluate held-out only):
CUDA_VISIBLE_DEVICES=3 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
  --model mistralai/Mistral-7B-Instruct-v0.3 --device cuda --model_dtype fp32 \
  --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
  --mode loto --batch_size 16 \
  --loto_eval_mode heldout \
  --n_subspace 128 --n_eval 2048 --layer 4 \
  --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
  --reasoning_tokens 128 --max_new_tokens 256 \
  --template_randomization 1 --shuffle_choices 1 \
  --do_sample 0 --out_json results/disturb_cot_reason_appendix/energy_balance_loto8_results_Mistral-7B-Instruct-v0.3_layer4.json --out_md results/disturb_cot_reason_appendix/energy_balance_loto8_summary_Mistral-7B-Instruct-v0.3_layer4.md

# Single (all-tasks) basis estimation + evaluation:
CUDA_VISIBLE_DEVICES=3 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
  --model mistralai/Mistral-7B-Instruct-v0.3 --device cuda --model_dtype fp32 \
  --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
  --mode all --batch_size 16 \
  --n_subspace 128 --n_eval 2048 --layer 10 \
  --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
  --reasoning_tokens 128 --max_new_tokens 256 \
  --template_randomization 1 --shuffle_choices 1 \
  --do_sample 0 --out_json results/disturb_cot_reason_appendix/all_tasks_energy_balance_greedy_results_Mistral-7B-Instruct-v0.3_layer10.json --out_md results/disturb_cot_reason_appendix/all_tasks_energy_balance_greedy_summary_Mistral-7B-Instruct-v0.3_layer10.md

# LOTO (estimate basis on N-1 tasks, evaluate held-out only):
CUDA_VISIBLE_DEVICES=3 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
  --model mistralai/Mistral-7B-Instruct-v0.3 --device cuda --model_dtype fp32 \
  --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
  --mode loto --batch_size 16 \
  --loto_eval_mode heldout \
  --n_subspace 128 --n_eval 2048 --layer 10 \
  --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
  --reasoning_tokens 128 --max_new_tokens 256 \
  --template_randomization 1 --shuffle_choices 1 \
  --do_sample 0 --out_json results/disturb_cot_reason_appendix/energy_balance_loto8_results_Mistral-7B-Instruct-v0.3_layer10.json --out_md results/disturb_cot_reason_appendix/energy_balance_loto8_summary_Mistral-7B-Instruct-v0.3_layer10.md


# Single (all-tasks) basis estimation + evaluation:
CUDA_VISIBLE_DEVICES=3 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
  --model mistralai/Mistral-7B-Instruct-v0.3 --device cuda --model_dtype fp32 \
  --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
  --mode all --batch_size 16 \
  --n_subspace 128 --n_eval 2048 --layer 24 \
  --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
  --reasoning_tokens 128 --max_new_tokens 256 \
  --template_randomization 1 --shuffle_choices 1 \
  --do_sample 0 --out_json results/disturb_cot_reason_appendix/all_tasks_energy_balance_greedy_results_Mistral-7B-Instruct-v0.3_layer24.json --out_md results/disturb_cot_reason_appendix/all_tasks_energy_balance_greedy_summary_Mistral-7B-Instruct-v0.3_layer24.md

# LOTO (estimate basis on N-1 tasks, evaluate held-out only):
CUDA_VISIBLE_DEVICES=3 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
  --model mistralai/Mistral-7B-Instruct-v0.3 --device cuda --model_dtype fp32 \
  --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
  --mode loto --batch_size 16 \
  --loto_eval_mode heldout \
  --n_subspace 128 --n_eval 2048 --layer 24 \
  --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
  --reasoning_tokens 128 --max_new_tokens 256 \
  --template_randomization 1 --shuffle_choices 1 \
  --do_sample 0 --out_json results/disturb_cot_reason_appendix/energy_balance_loto8_results_Mistral-7B-Instruct-v0.3_layer24.json --out_md results/disturb_cot_reason_appendix/energy_balance_loto8_summary_Mistral-7B-Instruct-v0.3_layer24.md




# tiiuae/falcon-7b-instruct
# run the with-aqua version
# Single (all-tasks) basis estimation + evaluation:
CUDA_VISIBLE_DEVICES=3 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
  --model tiiuae/falcon-7b-instruct --device cuda --model_dtype fp32 \
  --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
  --mode all --batch_size 16 \
  --n_subspace 128 --n_eval 2048 --layer 4 \
  --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
  --reasoning_tokens 128 --max_new_tokens 256 \
  --template_randomization 1 --shuffle_choices 1 \
  --do_sample 0 --out_json results/disturb_cot_reason_appendix/all_tasks_energy_balance_greedy_results_falcon-7b-instruct_layer4.json --out_md results/disturb_cot_reason_appendix/all_tasks_energy_balance_greedy_summary_falcon-7b-instruct_layer4.md

# LOTO (estimate basis on N-1 tasks, evaluate held-out only):
CUDA_VISIBLE_DEVICES=3 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
  --model tiiuae/falcon-7b-instruct --device cuda --model_dtype fp32 \
  --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
  --mode loto --batch_size 16 \
  --loto_eval_mode heldout \
  --n_subspace 128 --n_eval 2048 --layer 4 \
  --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
  --reasoning_tokens 128 --max_new_tokens 256 \
  --template_randomization 1 --shuffle_choices 1 \
  --do_sample 0 --out_json results/disturb_cot_reason_appendix/energy_balance_loto8_results_falcon-7b-instruct_layer4.json --out_md results/disturb_cot_reason_appendix/energy_balance_loto8_summary_falcon-7b-instruct_layer4.md

# Single (all-tasks) basis estimation + evaluation:
CUDA_VISIBLE_DEVICES=3 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
  --model tiiuae/falcon-7b-instruct --device cuda --model_dtype fp32 \
  --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
  --mode all --batch_size 16 \
  --n_subspace 128 --n_eval 2048 --layer 10 \
  --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
  --reasoning_tokens 128 --max_new_tokens 256 \
  --template_randomization 1 --shuffle_choices 1 \
  --do_sample 0 --out_json results/disturb_cot_reason_appendix/all_tasks_energy_balance_greedy_results_falcon-7b-instruct_layer10.json --out_md results/disturb_cot_reason_appendix/all_tasks_energy_balance_greedy_summary_falcon-7b-instruct_layer10.md

# LOTO (estimate basis on N-1 tasks, evaluate held-out only):
CUDA_VISIBLE_DEVICES=3 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
  --model tiiuae/falcon-7b-instruct --device cuda --model_dtype fp32 \
  --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
  --mode loto --batch_size 16 \
  --loto_eval_mode heldout \
  --n_subspace 128 --n_eval 2048 --layer 10 \
  --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
  --reasoning_tokens 128 --max_new_tokens 256 \
  --template_randomization 1 --shuffle_choices 1 \
  --do_sample 0 --out_json results/disturb_cot_reason_appendix/energy_balance_loto8_results_falcon-7b-instruct_layer10.json --out_md results/disturb_cot_reason_appendix/energy_balance_loto8_summary_falcon-7b-instruct_layer10.md


# Single (all-tasks) basis estimation + evaluation:
CUDA_VISIBLE_DEVICES=3 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
  --model tiiuae/falcon-7b-instruct --device cuda --model_dtype fp32 \
  --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
  --mode all --batch_size 16 \
  --n_subspace 128 --n_eval 2048 --layer 24 \
  --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
  --reasoning_tokens 128 --max_new_tokens 256 \
  --template_randomization 1 --shuffle_choices 1 \
  --do_sample 0 --out_json results/disturb_cot_reason_appendix/all_tasks_energy_balance_greedy_results_falcon-7b-instruct_layer24.json --out_md results/disturb_cot_reason_appendix/all_tasks_energy_balance_greedy_summary_falcon-7b-instruct_layer24.md

# LOTO (estimate basis on N-1 tasks, evaluate held-out only):
CUDA_VISIBLE_DEVICES=3 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
  --model tiiuae/falcon-7b-instruct --device cuda --model_dtype fp32 \
  --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
  --mode loto --batch_size 16 \
  --loto_eval_mode heldout \
  --n_subspace 128 --n_eval 2048 --layer 24 \
  --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
  --reasoning_tokens 128 --max_new_tokens 256 \
  --template_randomization 1 --shuffle_choices 1 \
  --do_sample 0 --out_json results/disturb_cot_reason_appendix/energy_balance_loto8_results_falcon-7b-instruct_layer24.json --out_md results/disturb_cot_reason_appendix/energy_balance_loto8_summary_falcon-7b-instruct_layer24.md
