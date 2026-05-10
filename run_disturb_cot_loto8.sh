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





# meta-llama/Llama-3.1-8B-Instruct
# run the with-aqua version
# Single (all-tasks) basis estimation + evaluation:
CUDA_VISIBLE_DEVICES=2 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
  --model meta-llama/Llama-3.1-8B-Instruct --device cuda --model_dtype fp32 \
  --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
  --mode all --batch_size 16 \
  --n_subspace 128 --n_eval 2048 --layer 4 \
  --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
  --reasoning_tokens 128 --max_new_tokens 256 \
  --template_randomization 1 --shuffle_choices 1 \
  --do_sample 0 --out_json results/disturb_cot_reason_appendix/all_tasks_energy_balance_greedy_results_Llama-3.1-8B-Instruct_layer4.json --out_md results/disturb_cot_reason_appendix/all_tasks_energy_balance_greedy_summary_Llama-3.1-8B-Instruct_layer4.md

# LOTO (estimate basis on N-1 tasks, evaluate held-out only):
CUDA_VISIBLE_DEVICES=2 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
  --model meta-llama/Llama-3.1-8B-Instruct --device cuda --model_dtype fp32 \
  --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
  --mode loto --batch_size 16 \
  --loto_eval_mode heldout \
  --n_subspace 128 --n_eval 2048 --layer 4 \
  --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
  --reasoning_tokens 128 --max_new_tokens 256 \
  --template_randomization 1 --shuffle_choices 1 \
  --do_sample 0 --out_json results/disturb_cot_reason_appendix/energy_balance_loto8_results_Llama-3.1-8B-Instruct_layer4.json --out_md results/disturb_cot_reason_appendix/energy_balance_loto8_summary_Llama-3.1-8B-Instruct_layer4.md


# run the with-aqua version
# Single (all-tasks) basis estimation + evaluation:
CUDA_VISIBLE_DEVICES=2 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
  --model meta-llama/Llama-3.1-8B-Instruct --device cuda --model_dtype fp32 \
  --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
  --mode all --batch_size 16 \
  --n_subspace 128 --n_eval 2048 --layer 10 \
  --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
  --reasoning_tokens 128 --max_new_tokens 256 \
  --template_randomization 1 --shuffle_choices 1 \
  --do_sample 0 --out_json results/disturb_cot_reason_appendix/all_tasks_energy_balance_greedy_results_Llama-3.1-8B-Instruct_layer10.json --out_md results/disturb_cot_reason_appendix/all_tasks_energy_balance_greedy_summary_Llama-3.1-8B-Instruct_layer10.md

# LOTO (estimate basis on N-1 tasks, evaluate held-out only):
CUDA_VISIBLE_DEVICES=2 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
  --model meta-llama/Llama-3.1-8B-Instruct --device cuda --model_dtype fp32 \
  --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
  --mode loto --batch_size 16 \
  --loto_eval_mode heldout \
  --n_subspace 128 --n_eval 2048 --layer 10 \
  --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
  --reasoning_tokens 128 --max_new_tokens 256 \
  --template_randomization 1 --shuffle_choices 1 \
  --do_sample 0 --out_json results/disturb_cot_reason_appendix/energy_balance_loto8_results_Llama-3.1-8B-Instruct_layer10.json --out_md results/disturb_cot_reason_appendix/energy_balance_loto8_summary_Llama-3.1-8B-Instruct_layer10.md


# run the with-aqua version
# Single (all-tasks) basis estimation + evaluation:
CUDA_VISIBLE_DEVICES=2 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
  --model meta-llama/Llama-3.1-8B-Instruct --device cuda --model_dtype fp32 \
  --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
  --mode all --batch_size 16 \
  --n_subspace 128 --n_eval 2048 --layer 24 \
  --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
  --reasoning_tokens 128 --max_new_tokens 256 \
  --template_randomization 1 --shuffle_choices 1 \
  --do_sample 0 --out_json results/disturb_cot_reason_appendix/all_tasks_energy_balance_greedy_results_Llama-3.1-8B-Instruct_layer24.json --out_md results/disturb_cot_reason_appendix/all_tasks_energy_balance_greedy_summary_Llama-3.1-8B-Instruct_layer24.md

# LOTO (estimate basis on N-1 tasks, evaluate held-out only):
CUDA_VISIBLE_DEVICES=2 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
  --model meta-llama/Llama-3.1-8B-Instruct --device cuda --model_dtype fp32 \
  --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
  --mode loto --batch_size 16 \
  --loto_eval_mode heldout \
  --n_subspace 128 --n_eval 2048 --layer 24 \
  --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
  --reasoning_tokens 128 --max_new_tokens 256 \
  --template_randomization 1 --shuffle_choices 1 \
  --do_sample 0 --out_json results/disturb_cot_reason_appendix/energy_balance_loto8_results_Llama-3.1-8B-Instruct_layer24.json --out_md results/disturb_cot_reason_appendix/energy_balance_loto8_summary_Llama-3.1-8B-Instruct_layer24.md






# Qwen/Qwen2.5-7B-Instruct
# run the with-aqua version
# Single (all-tasks) basis estimation + evaluation:
CUDA_VISIBLE_DEVICES=2 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
  --model Qwen/Qwen2.5-7B-Instruct --device cuda --model_dtype fp32 \
  --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
  --mode all --batch_size 16 \
  --n_subspace 128 --n_eval 2048 --layer 4 \
  --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
  --reasoning_tokens 128 --max_new_tokens 256 \
  --template_randomization 1 --shuffle_choices 1 \
  --do_sample 0 --out_json results/disturb_cot_reason_appendix/all_tasks_energy_balance_greedy_results_Qwen2.5-7B-Instruct_layer4.json --out_md results/disturb_cot_reason_appendix/all_tasks_energy_balance_greedy_summary_Qwen2.5-7B-Instruct_layer4.md

# LOTO (estimate basis on N-1 tasks, evaluate held-out only):
CUDA_VISIBLE_DEVICES=2 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
  --model Qwen/Qwen2.5-7B-Instruct --device cuda --model_dtype fp32 \
  --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
  --mode loto --batch_size 16 \
  --loto_eval_mode heldout \
  --n_subspace 128 --n_eval 2048 --layer 4 \
  --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
  --reasoning_tokens 128 --max_new_tokens 256 \
  --template_randomization 1 --shuffle_choices 1 \
  --do_sample 0 --out_json results/disturb_cot_reason_appendix/energy_balance_loto8_results_Qwen2.5-7B-Instruct_layer4.json --out_md results/disturb_cot_reason_appendix/energy_balance_loto8_summary_Qwen2.5-7B-Instruct_layer4.md


  CUDA_VISIBLE_DEVICES=2 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
    --model Qwen/Qwen2.5-7B-Instruct --device cuda --model_dtype fp32 \
    --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
    --mode all --batch_size 16 \
    --n_subspace 128 --n_eval 2048 --layer 10 \
    --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
    --reasoning_tokens 128 --max_new_tokens 256 \
    --template_randomization 1 --shuffle_choices 1 \
    --do_sample 0 --out_json results/disturb_cot_reason_appendix/all_tasks_energy_balance_greedy_results_Qwen2.5-7B-Instruct_layer10.json --out_md results/disturb_cot_reason_appendix/all_tasks_energy_balance_greedy_summary_Qwen2.5-7B-Instruct_layer10.md

  # LOTO (estimate basis on N-1 tasks, evaluate held-out only):
  CUDA_VISIBLE_DEVICES=2 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
    --model Qwen/Qwen2.5-7B-Instruct --device cuda --model_dtype fp32 \
    --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
    --mode loto --batch_size 16 \
    --loto_eval_mode heldout \
    --n_subspace 128 --n_eval 2048 --layer 10 \
    --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
    --reasoning_tokens 128 --max_new_tokens 256 \
    --template_randomization 1 --shuffle_choices 1 \
    --do_sample 0 --out_json results/disturb_cot_reason_appendix/energy_balance_loto8_results_Qwen2.5-7B-Instruct_layer10.json --out_md results/disturb_cot_reason_appendix/energy_balance_loto8_summary_Qwen2.5-7B-Instruct_layer10.md



  CUDA_VISIBLE_DEVICES=2 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
    --model Qwen/Qwen2.5-7B-Instruct --device cuda --model_dtype fp32 \
    --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
    --mode all --batch_size 16 \
    --n_subspace 128 --n_eval 2048 --layer 24 \
    --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
    --reasoning_tokens 128 --max_new_tokens 256 \
    --template_randomization 1 --shuffle_choices 1 \
    --do_sample 0 --out_json results/disturb_cot_reason_appendix/all_tasks_energy_balance_greedy_results_Qwen2.5-7B-Instruct_layer24.json --out_md results/disturb_cot_reason_appendix/all_tasks_energy_balance_greedy_summary_Qwen2.5-7B-Instruct_layer24.md

  # LOTO (estimate basis on N-1 tasks, evaluate held-out only):
  CUDA_VISIBLE_DEVICES=2 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
    --model Qwen/Qwen2.5-7B-Instruct --device cuda --model_dtype fp32 \
    --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
    --mode loto --batch_size 16 \
    --loto_eval_mode heldout \
    --n_subspace 128 --n_eval 2048 --layer 24 \
    --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
    --reasoning_tokens 128 --max_new_tokens 256 \
    --template_randomization 1 --shuffle_choices 1 \
    --do_sample 0 --out_json results/disturb_cot_reason_appendix/energy_balance_loto8_results_Qwen2.5-7B-Instruct_layer24.json --out_md results/disturb_cot_reason_appendix/energy_balance_loto8_summary_Qwen2.5-7B-Instruct_layer24.md







# meta-llama/Llama-3.2-3B-Instruct
# run the with-aqua version
# Single (all-tasks) basis estimation + evaluation:
CUDA_VISIBLE_DEVICES=2 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
  --model meta-llama/Llama-3.2-3B-Instruct --device cuda --model_dtype fp32 \
  --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
  --mode all --batch_size 16 \
  --n_subspace 128 --n_eval 2048 --layer 4 \
  --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
  --reasoning_tokens 128 --max_new_tokens 256 \
  --template_randomization 1 --shuffle_choices 1 \
  --do_sample 0 --out_json results/disturb_cot_reason_appendix/all_tasks_energy_balance_greedy_results_Llama-3.2-3B-Instruct_layer4.json --out_md results/disturb_cot_reason_appendix/all_tasks_energy_balance_greedy_summary_Llama-3.2-3B-Instruct_layer4.md

# LOTO (estimate basis on N-1 tasks, evaluate held-out only):
CUDA_VISIBLE_DEVICES=2 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
  --model meta-llama/Llama-3.2-3B-Instruct --device cuda --model_dtype fp32 \
  --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
  --mode loto --batch_size 16 \
  --loto_eval_mode heldout \
  --n_subspace 128 --n_eval 2048 --layer 4 \
  --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
  --reasoning_tokens 128 --max_new_tokens 256 \
  --template_randomization 1 --shuffle_choices 1 \
  --do_sample 0 --out_json results/disturb_cot_reason_appendix/energy_balance_loto8_results_Llama-3.2-3B-Instruct_layer4.json --out_md results/disturb_cot_reason_appendix/energy_balance_loto8_summary_Llama-3.2-3B-Instruct_layer4.md


  CUDA_VISIBLE_DEVICES=2 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
    --model meta-llama/Llama-3.2-3B-Instruct --device cuda --model_dtype fp32 \
    --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
    --mode all --batch_size 16 \
    --n_subspace 128 --n_eval 2048 --layer 10 \
    --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
    --reasoning_tokens 128 --max_new_tokens 256 \
    --template_randomization 1 --shuffle_choices 1 \
    --do_sample 0 --out_json results/disturb_cot_reason_appendix/all_tasks_energy_balance_greedy_results_Llama-3.2-3B-Instruct_layer10.json --out_md results/disturb_cot_reason_appendix/all_tasks_energy_balance_greedy_summary_Llama-3.2-3B-Instruct_layer10.md

  # LOTO (estimate basis on N-1 tasks, evaluate held-out only):
  CUDA_VISIBLE_DEVICES=2 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
    --model meta-llama/Llama-3.2-3B-Instruct --device cuda --model_dtype fp32 \
    --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
    --mode loto --batch_size 16 \
    --loto_eval_mode heldout \
    --n_subspace 128 --n_eval 2048 --layer 10 \
    --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
    --reasoning_tokens 128 --max_new_tokens 256 \
    --template_randomization 1 --shuffle_choices 1 \
    --do_sample 0 --out_json results/disturb_cot_reason_appendix/energy_balance_loto8_results_Llama-3.2-3B-Instruct_layer10.json --out_md results/disturb_cot_reason_appendix/energy_balance_loto8_summary_Llama-3.2-3B-Instruct_layer10.md


  CUDA_VISIBLE_DEVICES=2 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
    --model meta-llama/Llama-3.2-3B-Instruct --device cuda --model_dtype fp32 \
    --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
    --mode all --batch_size 16 \
    --n_subspace 128 --n_eval 2048 --layer 24 \
    --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
    --reasoning_tokens 128 --max_new_tokens 256 \
    --template_randomization 1 --shuffle_choices 1 \
    --do_sample 0 --out_json results/disturb_cot_reason_appendix/all_tasks_energy_balance_greedy_results_Llama-3.2-3B-Instruct_layer24.json --out_md results/disturb_cot_reason_appendix/all_tasks_energy_balance_greedy_summary_Llama-3.2-3B-Instruct_layer24.md

  # LOTO (estimate basis on N-1 tasks, evaluate held-out only):
  CUDA_VISIBLE_DEVICES=2 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
    --model meta-llama/Llama-3.2-3B-Instruct --device cuda --model_dtype fp32 \
    --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
    --mode loto --batch_size 16 \
    --loto_eval_mode heldout \
    --n_subspace 128 --n_eval 2048 --layer 24 \
    --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
    --reasoning_tokens 128 --max_new_tokens 256 \
    --template_randomization 1 --shuffle_choices 1 \
    --do_sample 0 --out_json results/disturb_cot_reason_appendix/energy_balance_loto8_results_Llama-3.2-3B-Instruct_layer24.json --out_md results/disturb_cot_reason_appendix/energy_balance_loto8_summary_Llama-3.2-3B-Instruct_layer24.md





# # mistralai/Mistral-7B-Instruct-v0.3
# # run the with-aqua version
# # Single (all-tasks) basis estimation + evaluation:
# CUDA_VISIBLE_DEVICES=2 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
#   --model mistralai/Mistral-7B-Instruct-v0.3 --device cuda --model_dtype fp32 \
#   --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
#   --mode all --batch_size 16 \
#   --n_subspace 128 --n_eval 2048 --layer 4 \
#   --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
#   --reasoning_tokens 128 --max_new_tokens 256 \
#   --template_randomization 1 --shuffle_choices 1 \
#   --do_sample 0 --out_json results/disturb_cot_reason_appendix/all_tasks_energy_balance_greedy_results_Mistral-7B-Instruct-v0.3_layer4.json --out_md results/disturb_cot_reason_appendix/all_tasks_energy_balance_greedy_summary_Mistral-7B-Instruct-v0.3_layer4.md

# # LOTO (estimate basis on N-1 tasks, evaluate held-out only):
# CUDA_VISIBLE_DEVICES=2 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
#   --model mistralai/Mistral-7B-Instruct-v0.3 --device cuda --model_dtype fp32 \
#   --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
#   --mode loto --batch_size 16 \
#   --loto_eval_mode heldout \
#   --n_subspace 128 --n_eval 2048 --layer 4 \
#   --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
#   --reasoning_tokens 128 --max_new_tokens 256 \
#   --template_randomization 1 --shuffle_choices 1 \
#   --do_sample 0 --out_json results/disturb_cot_reason_appendix/energy_balance_loto8_results_Mistral-7B-Instruct-v0.3_layer4.json --out_md results/disturb_cot_reason_appendix/energy_balance_loto8_summary_Mistral-7B-Instruct-v0.3_layer4.md

# # Single (all-tasks) basis estimation + evaluation:
# CUDA_VISIBLE_DEVICES=2 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
#   --model mistralai/Mistral-7B-Instruct-v0.3 --device cuda --model_dtype fp32 \
#   --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
#   --mode all --batch_size 16 \
#   --n_subspace 128 --n_eval 2048 --layer 10 \
#   --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
#   --reasoning_tokens 128 --max_new_tokens 256 \
#   --template_randomization 1 --shuffle_choices 1 \
#   --do_sample 0 --out_json results/disturb_cot_reason_appendix/all_tasks_energy_balance_greedy_results_Mistral-7B-Instruct-v0.3_layer10.json --out_md results/disturb_cot_reason_appendix/all_tasks_energy_balance_greedy_summary_Mistral-7B-Instruct-v0.3_layer10.md

# # LOTO (estimate basis on N-1 tasks, evaluate held-out only):
# CUDA_VISIBLE_DEVICES=2 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
#   --model mistralai/Mistral-7B-Instruct-v0.3 --device cuda --model_dtype fp32 \
#   --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
#   --mode loto --batch_size 16 \
#   --loto_eval_mode heldout \
#   --n_subspace 128 --n_eval 2048 --layer 10 \
#   --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
#   --reasoning_tokens 128 --max_new_tokens 256 \
#   --template_randomization 1 --shuffle_choices 1 \
#   --do_sample 0 --out_json results/disturb_cot_reason_appendix/energy_balance_loto8_results_Mistral-7B-Instruct-v0.3_layer10.json --out_md results/disturb_cot_reason_appendix/energy_balance_loto8_summary_Mistral-7B-Instruct-v0.3_layer10.md


# # Single (all-tasks) basis estimation + evaluation:
# CUDA_VISIBLE_DEVICES=2 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
#   --model mistralai/Mistral-7B-Instruct-v0.3 --device cuda --model_dtype fp32 \
#   --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
#   --mode all --batch_size 16 \
#   --n_subspace 128 --n_eval 2048 --layer 24 \
#   --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
#   --reasoning_tokens 128 --max_new_tokens 256 \
#   --template_randomization 1 --shuffle_choices 1 \
#   --do_sample 0 --out_json results/disturb_cot_reason_appendix/all_tasks_energy_balance_greedy_results_Mistral-7B-Instruct-v0.3_layer24.json --out_md results/disturb_cot_reason_appendix/all_tasks_energy_balance_greedy_summary_Mistral-7B-Instruct-v0.3_layer24.md

# # LOTO (estimate basis on N-1 tasks, evaluate held-out only):
# CUDA_VISIBLE_DEVICES=2 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
#   --model mistralai/Mistral-7B-Instruct-v0.3 --device cuda --model_dtype fp32 \
#   --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
#   --mode loto --batch_size 16 \
#   --loto_eval_mode heldout \
#   --n_subspace 128 --n_eval 2048 --layer 24 \
#   --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
#   --reasoning_tokens 128 --max_new_tokens 256 \
#   --template_randomization 1 --shuffle_choices 1 \
#   --do_sample 0 --out_json results/disturb_cot_reason_appendix/energy_balance_loto8_results_Mistral-7B-Instruct-v0.3_layer24.json --out_md results/disturb_cot_reason_appendix/energy_balance_loto8_summary_Mistral-7B-Instruct-v0.3_layer24.md




# # tiiuae/falcon-7b-instruct
# # run the with-aqua version
# # Single (all-tasks) basis estimation + evaluation:
# CUDA_VISIBLE_DEVICES=2 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
#   --model tiiuae/falcon-7b-instruct --device cuda --model_dtype fp32 \
#   --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
#   --mode all --batch_size 16 \
#   --n_subspace 128 --n_eval 2048 --layer 4 \
#   --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
#   --reasoning_tokens 128 --max_new_tokens 256 \
#   --template_randomization 1 --shuffle_choices 1 \
#   --do_sample 0 --out_json results/disturb_cot_reason_appendix/all_tasks_energy_balance_greedy_results_falcon-7b-instruct_layer4.json --out_md results/disturb_cot_reason_appendix/all_tasks_energy_balance_greedy_summary_falcon-7b-instruct_layer4.md

# # LOTO (estimate basis on N-1 tasks, evaluate held-out only):
# CUDA_VISIBLE_DEVICES=2 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
#   --model tiiuae/falcon-7b-instruct --device cuda --model_dtype fp32 \
#   --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
#   --mode loto --batch_size 16 \
#   --loto_eval_mode heldout \
#   --n_subspace 128 --n_eval 2048 --layer 4 \
#   --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
#   --reasoning_tokens 128 --max_new_tokens 256 \
#   --template_randomization 1 --shuffle_choices 1 \
#   --do_sample 0 --out_json results/disturb_cot_reason_appendix/energy_balance_loto8_results_falcon-7b-instruct_layer4.json --out_md results/disturb_cot_reason_appendix/energy_balance_loto8_summary_falcon-7b-instruct_layer4.md

# # Single (all-tasks) basis estimation + evaluation:
# CUDA_VISIBLE_DEVICES=2 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
#   --model tiiuae/falcon-7b-instruct --device cuda --model_dtype fp32 \
#   --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
#   --mode all --batch_size 16 \
#   --n_subspace 128 --n_eval 2048 --layer 10 \
#   --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
#   --reasoning_tokens 128 --max_new_tokens 256 \
#   --template_randomization 1 --shuffle_choices 1 \
#   --do_sample 0 --out_json results/disturb_cot_reason_appendix/all_tasks_energy_balance_greedy_results_falcon-7b-instruct_layer10.json --out_md results/disturb_cot_reason_appendix/all_tasks_energy_balance_greedy_summary_falcon-7b-instruct_layer10.md

# # LOTO (estimate basis on N-1 tasks, evaluate held-out only):
# CUDA_VISIBLE_DEVICES=2 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
#   --model tiiuae/falcon-7b-instruct --device cuda --model_dtype fp32 \
#   --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
#   --mode loto --batch_size 16 \
#   --loto_eval_mode heldout \
#   --n_subspace 128 --n_eval 2048 --layer 10 \
#   --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
#   --reasoning_tokens 128 --max_new_tokens 256 \
#   --template_randomization 1 --shuffle_choices 1 \
#   --do_sample 0 --out_json results/disturb_cot_reason_appendix/energy_balance_loto8_results_falcon-7b-instruct_layer10.json --out_md results/disturb_cot_reason_appendix/energy_balance_loto8_summary_falcon-7b-instruct_layer10.md


# # Single (all-tasks) basis estimation + evaluation:
# CUDA_VISIBLE_DEVICES=2 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
#   --model tiiuae/falcon-7b-instruct --device cuda --model_dtype fp32 \
#   --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
#   --mode all --batch_size 16 \
#   --n_subspace 128 --n_eval 2048 --layer 24 \
#   --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
#   --reasoning_tokens 128 --max_new_tokens 256 \
#   --template_randomization 1 --shuffle_choices 1 \
#   --do_sample 0 --out_json results/disturb_cot_reason_appendix/all_tasks_energy_balance_greedy_results_falcon-7b-instruct_layer24.json --out_md results/disturb_cot_reason_appendix/all_tasks_energy_balance_greedy_summary_falcon-7b-instruct_layer24.md

# # LOTO (estimate basis on N-1 tasks, evaluate held-out only):
# CUDA_VISIBLE_DEVICES=2 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
#   --model tiiuae/falcon-7b-instruct --device cuda --model_dtype fp32 \
#   --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
#   --mode loto --batch_size 16 \
#   --loto_eval_mode heldout \
#   --n_subspace 128 --n_eval 2048 --layer 24 \
#   --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
#   --reasoning_tokens 128 --max_new_tokens 256 \
#   --template_randomization 1 --shuffle_choices 1 \
#   --do_sample 0 --out_json results/disturb_cot_reason_appendix/energy_balance_loto8_results_falcon-7b-instruct_layer24.json --out_md results/disturb_cot_reason_appendix/energy_balance_loto8_summary_falcon-7b-instruct_layer24.md




# # run the with-aqua version
#   # Single (all-tasks) basis estimation + evaluation:
#   CUDA_VISIBLE_DEVICES=3 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
#     --model meta-llama/Llama-2-7b-chat-hf --device cuda --model_dtype fp32 \
#     --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
#     --mode all --batch_size 16 \
#     --n_subspace 128 --n_eval 2048 --layer 10 \
#     --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
#     --reasoning_tokens 128 --max_new_tokens 256 \
#     --template_randomization 1 --shuffle_choices 1 \
#     --do_sample 0 --out_json results/disturb_cot_full/all_tasks_energy_balance_greedy_results_Llama-2-7b-chat-hf.json --out_md results/disturb_cot_full/all_tasks_energy_balance_greedy_summary_Llama-2-7b-chat-hf.md

#   # CUDA_VISIBLE_DEVICES=3 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
#   #   --model meta-llama/Llama-2-7b-chat-hf --device cuda --model_dtype fp32 \
#   #   --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
#   #   --mode all --batch_size 16 \
#   #   --n_subspace 128 --n_eval 2048 --layer 10 \
#   #   --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
#   #   --reasoning_tokens 128 --max_new_tokens 256 \
#   #   --template_randomization 1 --shuffle_choices 1 \
#   #   --do_sample 1 --out_json results/disturb_cot_full/all_tasks_energy_balance_sample_results_Llama-2-7b-chat-hf.json --out_md results/disturb_cot_full/all_tasks_energy_balance_sample_summary_Llama-2-7b-chat-hf.md

#   # LOTO (estimate basis on N-1 tasks, evaluate held-out only):
#   CUDA_VISIBLE_DEVICES=3 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
#     --model meta-llama/Llama-2-7b-chat-hf --device cuda --model_dtype fp32 \
#     --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
#     --mode loto --batch_size 16 \
#     --loto_eval_mode heldout \
#     --n_subspace 128 --n_eval 2048 --layer 10 \
#     --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
#     --reasoning_tokens 128 --max_new_tokens 256 \
#     --template_randomization 1 --shuffle_choices 1 \
#     --do_sample 0 --out_json results/disturb_cot_full/energy_balance_loto8_results_Llama-2-7b-chat-hf.json --out_md results/disturb_cot_full/energy_balance_loto8_summary_Llama-2-7b-chat-hf.md







# # run the with-aqua version
#   # Single (all-tasks) basis estimation + evaluation:
#   CUDA_VISIBLE_DEVICES=3 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
#     --model Qwen/Qwen2.5-7B-Instruct --device cuda --model_dtype fp32 \
#     --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
#     --mode all --batch_size 16 \
#     --n_subspace 128 --n_eval 2048 --layer 10 \
#     --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
#     --reasoning_tokens 128 --max_new_tokens 256 \
#     --template_randomization 1 --shuffle_choices 1 \
#     --do_sample 0 --out_json results/disturb_cot_full/all_tasks_energy_balance_greedy_results_Qwen2.5-7B-Instruct.json --out_md results/disturb_cot_full/all_tasks_energy_balance_greedy_summary_Qwen2.5-7B-Instruct.md

#   # CUDA_VISIBLE_DEVICES=3 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
#   #   --model Qwen/Qwen2.5-7B-Instruct --device cuda --model_dtype fp32 \
#   #   --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
#   #   --mode all \
#   #   --n_subspace 128 --n_eval 2048 --layer 10 \
#   #   --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
#   #   --reasoning_tokens 128 --max_new_tokens 256 \
#   #   --template_randomization 1 --shuffle_choices 1 \
#   #   --do_sample 1 --out_json results/disturb_cot_full/all_tasks_energy_balance_sample_results_Qwen2.5-7B-Instruct.json --out_md results/disturb_cot_full/all_tasks_energy_balance_sample_summary_Qwen2.5-7B-Instruct.md

#   # LOTO (estimate basis on N-1 tasks, evaluate held-out only):
#   CUDA_VISIBLE_DEVICES=3 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
#     --model Qwen/Qwen2.5-7B-Instruct --device cuda --model_dtype fp32 \
#     --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
#     --mode loto --batch_size 16 \
#     --loto_eval_mode heldout \
#     --n_subspace 128 --n_eval 2048 --layer 10 \
#     --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
#     --reasoning_tokens 128 --max_new_tokens 256 \
#     --template_randomization 1 --shuffle_choices 1 \
#     --do_sample 0 --out_json results/disturb_cot_full/energy_balance_loto8_results_Qwen2.5-7B-Instruct.json --out_md results/disturb_cot_full/energy_balance_loto8_summary_Qwen2.5-7B-Instruct.md






# # run the with-aqua version
#   # Single (all-tasks) basis estimation + evaluation:
#   CUDA_VISIBLE_DEVICES=3 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
#     --model facebook/opt-6.7b --device cuda --model_dtype fp32 \
#     --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
#     --mode all --batch_size 16 \
#     --n_subspace 128 --n_eval 2048 --layer 10 \
#     --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
#     --reasoning_tokens 128 --max_new_tokens 256 \
#     --template_randomization 1 --shuffle_choices 1 \
#     --do_sample 0 --out_json results/disturb_cot_full/all_tasks_energy_balance_greedy_results_opt-6.7b.json --out_md results/disturb_cot_full/all_tasks_energy_balance_greedy_summary_opt-6.7b.md

#   # CUDA_VISIBLE_DEVICES=3 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
#   #   --model google/gemma-3-12b-it --device cuda --model_dtype fp32 \
#   #   --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
#   #   --mode all \
#   #   --n_subspace 128 --n_eval 2048 --layer 10 \
#   #   --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
#   #   --reasoning_tokens 128 --max_new_tokens 256 \
#   #   --template_randomization 1 --shuffle_choices 1 \
#   #   --do_sample 1 --out_json results/disturb_cot_full/all_tasks_energy_balance_sample_results_gemma-3-12b-it.json --out_md results/disturb_cot_full/all_tasks_energy_balance_sample_summary_gemma-3-12b-it.md

#   # LOTO (estimate basis on N-1 tasks, evaluate held-out only):
#   CUDA_VISIBLE_DEVICES=3 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
#     --model facebook/opt-6.7b --device cuda --model_dtype fp32 \
#     --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
#     --mode loto --batch_size 16 \
#     --loto_eval_mode heldout \
#     --n_subspace 128 --n_eval 2048 --layer 10 \
#     --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
#     --reasoning_tokens 128 --max_new_tokens 256 \
#     --template_randomization 1 --shuffle_choices 1 \
#     --do_sample 0 --out_json results/disturb_cot_full/energy_balance_loto8_results_opt-6.7b.json --out_md results/disturb_cot_full/energy_balance_loto8_summary_opt-6.7b.md






# # run the with-aqua version
#   # Single (all-tasks) basis estimation + evaluation:
#   CUDA_VISIBLE_DEVICES=3 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
#     --model google/gemma-3-12b-it --device cuda --model_dtype fp32 \
#     --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
#     --mode all --batch_size 16 \
#     --n_subspace 128 --n_eval 2048 --layer 10 \
#     --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
#     --reasoning_tokens 128 --max_new_tokens 256 \
#     --template_randomization 1 --shuffle_choices 1 \
#     --do_sample 0 --out_json results/disturb_cot_full/all_tasks_energy_balance_greedy_results_gemma-3-12b-it.json --out_md results/disturb_cot_full/all_tasks_energy_balance_greedy_summary_gemma-3-12b-it.md

#   # CUDA_VISIBLE_DEVICES=3 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
#   #   --model google/gemma-3-12b-it --device cuda --model_dtype fp32 \
#   #   --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
#   #   --mode all \
#   #   --n_subspace 128 --n_eval 2048 --layer 10 \
#   #   --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
#   #   --reasoning_tokens 128 --max_new_tokens 256 \
#   #   --template_randomization 1 --shuffle_choices 1 \
#   #   --do_sample 1 --out_json results/disturb_cot_full/all_tasks_energy_balance_sample_results_gemma-3-12b-it.json --out_md results/disturb_cot_full/all_tasks_energy_balance_sample_summary_gemma-3-12b-it.md

#   # LOTO (estimate basis on N-1 tasks, evaluate held-out only):
#   CUDA_VISIBLE_DEVICES=3 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
#     --model google/gemma-3-12b-it --device cuda --model_dtype fp32 \
#     --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
#     --mode loto --batch_size 16 \
#     --loto_eval_mode heldout \
#     --n_subspace 128 --n_eval 2048 --layer 10 \
#     --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
#     --reasoning_tokens 128 --max_new_tokens 256 \
#     --template_randomization 1 --shuffle_choices 1 \
#     --do_sample 0 --out_json results/disturb_cot_full/energy_balance_loto8_results_gemma-3-12b-it.json --out_md results/disturb_cot_full/energy_balance_loto8_summary_gemma-3-12b-it.md









# # 你现在这套 benchmark，和哪些模型“天然匹配”（不改代码）
# # 你当前任务集：gsm8k, commonsenseqa, strategyqa, aqua, arc_challenge, openbookqa, qasc
# # 共同点：都用“请 step-by-step + 最后一行 Final answer”这种 CoT 风格指令。
# # ✅ 最匹配（建议作为主报告/主对比）
# # 这些模型本来就更像“会听话”的 assistant，能稳定输出你要的最后一行格式：
# # meta-llama/Llama-2-7b-chat-hf
# # Qwen/Qwen2.5-7B-Instruct
# # google/gemma-2-2b-it
# # google/gemma-3-12b-it（如果它确实是 it / instruction-tuned）
# # 对这组，用你现有 7 个任务集做横向对比是合理的。


# # -----------------------------
# # Env / paths
# # -----------------------------
# WORKDIR="src"
# cd "${WORKDIR}"

# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate flashsvd

# export TOKENIZERS_PARALLELISM=false
# export PYTHONUNBUFFERED=1

# GPU_ID="${GPU_ID:-1}"
# MODEL_DTYPE="${MODEL_DTYPE:-fp32}"          # fp32 / fp16
# DO_SAMPLE="${DO_SAMPLE:-0}"                # 0/1

# # Energy-balance script (LOTO8 version)
# SCRIPT_PATH="${SCRIPT_PATH:-${WORKDIR}/disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py}"

# # Tasks used by the loto8 script
# TASKS="${TASKS:-gsm8k,commonsenseqa,strategyqa,aqua,arc_challenge,openbookqa,qasc}"

# # all: one basis on all tasks
# # loto: estimate on N-1 tasks, eval held-out (or all)
# MODE="${MODE:-all}"                         # all / loto
# LOTO_EVAL_MODE="${LOTO_EVAL_MODE:-heldout}" # heldout / all (only used when MODE=loto)

# # Template randomization knobs (optional but recommended)
# TEMPLATE_RANDOMIZATION="${TEMPLATE_RANDOMIZATION:-0}"  # 0/1
# TEMPLATE_SEED="${TEMPLATE_SEED:-1234}"
# SHUFFLE_CHOICES="${SHUFFLE_CHOICES:-0}"               # 0/1
# ADD_ANSWER_PREFIX="${ADD_ANSWER_PREFIX:-1}"           # 0/1
# ANSWER_PREFIX="${ANSWER_PREFIX:-$'\nFinal answer:'}"

# RESULTS_DIR="${RESULTS_DIR:-${WORKDIR}/results/energy_balance}"
# mkdir -p "${RESULTS_DIR}"

# # -----------------------------
# # Models / sweeps
# # -----------------------------
# MODEL_NAMES=(
#   # "meta-llama/Llama-2-7b-chat-hf"
#   #"meta-llama/Llama-2-7b-hf"
#   #"facebook/opt-6.7b"
#   #"Qwen/Qwen2.5-7B"
#   "Qwen/Qwen2.5-7B-Instruct"
#   # "google/gemma-3-12b-it"
#   # "google/gemma-2-2b-it"
# )

# LAYERS=(2 8 10 18 20 24)

# # sharedness params (supported by loto8 script)
# TAUS=(0.001 0.01)
# M_SHARED="all"  # or e.g. 6

# N_SUBSPACE_LIST=(128)
# N_EVAL_LIST=(256)
# CALIB_DECODE_MAX_NEW_TOKENS_LIST=(128)
# MAX_PROMPT_LEN_LIST=(512)
# PER_TASK_MAX_STATES_LIST=(20000)

# # Random controls (supported by loto8 script)
# RAND_TYPES=(
#   "joint_nonshared_varmatch"
#   "joint_nonshared_topk"
# )

# # fixed decode intervention params
# REASONING_TOKENS=128
# MAX_NEW_TOKENS=256

# # -----------------------------
# # Run
# # -----------------------------
# for MODEL_NAME in "${MODEL_NAMES[@]}"; do
#   MODEL_TAG="$(echo "${MODEL_NAME}" | tr '/:' '__')"

#   for LAYER in "${LAYERS[@]}"; do
#     for TAU in "${TAUS[@]}"; do
#       for N_SUBSPACE in "${N_SUBSPACE_LIST[@]}"; do
#         for N_EVAL in "${N_EVAL_LIST[@]}"; do
#           for CALIB_NEW in "${CALIB_DECODE_MAX_NEW_TOKENS_LIST[@]}"; do
#             for MAX_PROMPT_LEN in "${MAX_PROMPT_LEN_LIST[@]}"; do
#               for PER_TASK_MAX_STATES in "${PER_TASK_MAX_STATES_LIST[@]}"; do
#                 for RAND_TYPE in "${RAND_TYPES[@]}"; do

#                   OUT_BASE="${RESULTS_DIR}/energy_${MODEL_TAG}_${MODEL_DTYPE}_layer${LAYER}_sub${N_SUBSPACE}_eval${N_EVAL}_calib${CALIB_NEW}_maxlen${MAX_PROMPT_LEN}_states${PER_TASK_MAX_STATES}_tau${TAU}_m${M_SHARED}_rand${RAND_TYPE}_mode${MODE}"
#                   OUT_JSON="${OUT_BASE}.json"
#                   OUT_MD="${OUT_BASE}.md"
#                   OUT_LOG="${OUT_BASE}.txt"

#                   echo ""
#                   echo "================================================================================"
#                   echo "[Run] model=${MODEL_NAME} dtype=${MODEL_DTYPE} gpu=${GPU_ID}"
#                   echo "      layer=${LAYER} tau=${TAU} m_shared=${M_SHARED} rand_type=${RAND_TYPE}"
#                   echo "      n_subspace=${N_SUBSPACE} n_eval=${N_EVAL} calib_new=${CALIB_NEW}"
#                   echo "      max_prompt_len=${MAX_PROMPT_LEN} per_task_max_states=${PER_TASK_MAX_STATES}"
#                   echo "      tasks=${TASKS}"
#                   echo "      mode=${MODE} loto_eval_mode=${LOTO_EVAL_MODE}"
#                   echo "      template_randomization=${TEMPLATE_RANDOMIZATION} shuffle_choices=${SHUFFLE_CHOICES}"
#                   echo "================================================================================"
#                   echo "[Out] ${OUT_LOG}"
#                   echo ""

#                   #CUDA_VISIBLE_DEVICES="${GPU_ID}" python "${SCRIPT_PATH}" \
#                   CUDA_VISIBLE_DEVICES=5 python "${SCRIPT_PATH}" \
#                     --model "${MODEL_NAME}" \
#                     --device cuda \
#                     --model_dtype "${MODEL_DTYPE}" \
#                     --layer "${LAYER}" \
#                     --tasks "${TASKS}" \
#                     --mode "${MODE}" \
#                     --loto_eval_mode "${LOTO_EVAL_MODE}" \
#                     --n_subspace "${N_SUBSPACE}" \
#                     --n_eval "${N_EVAL}" \
#                     --pca_var 0.95 \
#                     --tau "${TAU}" \
#                     --m_shared "${M_SHARED}" \
#                     --calib_decode_max_new_tokens "${CALIB_NEW}" \
#                     --per_task_max_states "${PER_TASK_MAX_STATES}" \
#                     --reasoning_tokens "${REASONING_TOKENS}" \
#                     --max_new_tokens "${MAX_NEW_TOKENS}" \
#                     --batch_size 4 \
#                     --max_prompt_len "${MAX_PROMPT_LEN}" \
#                     --rand_type "${RAND_TYPE}" \
#                     --template_randomization "${TEMPLATE_RANDOMIZATION}" \
#                     --template_seed "${TEMPLATE_SEED}" \
#                     --shuffle_choices "${SHUFFLE_CHOICES}" \
#                     --add_answer_prefix "${ADD_ANSWER_PREFIX}" \
#                     --answer_prefix "${ANSWER_PREFIX}" \
#                     --do_sample "${DO_SAMPLE}" \
#                     --out_json "${OUT_JSON}" \
#                     --out_md "${OUT_MD}" \
#                     2>&1 | tee "${OUT_LOG}"

#                 done
#               done
#             done
#           done
#         done
#       done
#     done
#   done
# done
