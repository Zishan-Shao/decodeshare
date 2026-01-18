#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Env / paths
# -----------------------------
WORKDIR="/home/zs89/decodeshare/src"
cd "${WORKDIR}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate flashsvd

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1







# run the with-aqua version
  # Single (all-tasks) basis estimation + evaluation:
  CUDA_VISIBLE_DEVICES=2 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
    --model google/gemma-3-12b-it --device cuda --model_dtype fp32 \
    --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
    --mode all --batch_size 16 \
    --n_subspace 128 --n_eval 2048 --layer 10 \
    --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
    --reasoning_tokens 128 --max_new_tokens 256 \
    --template_randomization 1 --shuffle_choices 1 \
    --do_sample 0 --out_json results/disturb_cot_reason/all_tasks_energy_balance_greedy_results_gemma-3-12b-it_improved.json --out_md results/disturb_cot_reason/all_tasks_energy_balance_greedy_summary_gemma-3-12b-it_improved.md

  # LOTO (estimate basis on N-1 tasks, evaluate held-out only):
  CUDA_VISIBLE_DEVICES=2 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
    --model google/gemma-3-12b-it --device cuda --model_dtype fp32 \
    --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
    --mode loto --batch_size 16 \
    --loto_eval_mode heldout \
    --n_subspace 128 --n_eval 2048 --layer 10 \
    --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
    --reasoning_tokens 128 --max_new_tokens 256 \
    --template_randomization 1 --shuffle_choices 1 \
    --do_sample 0 --out_json results/disturb_cot_reason/energy_balance_loto8_results_gemma-3-12b-it_improved.json --out_md results/disturb_cot_reason/energy_balance_loto8_summary_gemma-3-12b-it_improved.md 




# # run the with-aqua version
#   # Single (all-tasks) basis estimation + evaluation:
#   CUDA_VISIBLE_DEVICES=0 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
#     --model facebook/opt-6.7b --device cuda --model_dtype fp32 \
#     --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
#     --mode all --batch_size 16 \
#     --n_subspace 128 --n_eval 2048 --layer 10 \
#     --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
#     --reasoning_tokens 128 --max_new_tokens 256 \
#     --template_randomization 1 --shuffle_choices 1 \
#     --do_sample 0 --out_json results/disturb_cot_reason/all_tasks_energy_balance_greedy_results_opt-6.7b.json --out_md results/disturb_cot_reason/all_tasks_energy_balance_greedy_summary_opt-6.7b.md

#   # LOTO (estimate basis on N-1 tasks, evaluate held-out only):
#   CUDA_VISIBLE_DEVICES=0 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
#     --model facebook/opt-6.7b --device cuda --model_dtype fp32 \
#     --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
#     --mode loto --batch_size 16 \
#     --loto_eval_mode heldout \
#     --n_subspace 128 --n_eval 2048 --layer 10 \
#     --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
#     --reasoning_tokens 128 --max_new_tokens 256 \
#     --template_randomization 1 --shuffle_choices 1 \
#     --do_sample 0 --out_json results/disturb_cot_reason/energy_balance_loto8_results_opt-6.7b.json --out_md results/disturb_cot_reason/energy_balance_loto8_summary_opt-6.7b.md 



# # run the with-aqua version
#   # Single (all-tasks) basis estimation + evaluation:
#   CUDA_VISIBLE_DEVICES=2 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
#     --model meta-llama/Llama-2-7b-chat-hf --device cuda --model_dtype fp32 \
#     --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
#     --mode all --batch_size 16 \
#     --n_subspace 128 --n_eval 2048 --layer 10 \
#     --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
#     --reasoning_tokens 128 --max_new_tokens 256 \
#     --template_randomization 1 --shuffle_choices 1 \
#     --do_sample 0 --out_json results/disturb_cot_reason/all_tasks_energy_balance_greedy_results_Llama-2-7b-chat-hf.json --out_md results/disturb_cot_reason/all_tasks_energy_balance_greedy_summary_Llama-2-7b-chat-hf.md

#   # LOTO (estimate basis on N-1 tasks, evaluate held-out only):
#   CUDA_VISIBLE_DEVICES=2 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
#     --model meta-llama/Llama-2-7b-chat-hf --device cuda --model_dtype fp32 \
#     --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
#     --mode loto --batch_size 16 \
#     --loto_eval_mode heldout \
#     --n_subspace 128 --n_eval 2048 --layer 10 \
#     --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
#     --reasoning_tokens 128 --max_new_tokens 256 \
#     --template_randomization 1 --shuffle_choices 1 \
#     --do_sample 0 --out_json results/disturb_cot_reason/energy_balance_loto8_results_Llama-2-7b-chat-hf.json --out_md results/disturb_cot_reason/energy_balance_loto8_summary_Llama-2-7b-chat-hf.md 









# # run the with-aqua version
#   # Single (all-tasks) basis estimation + evaluation:
#   CUDA_VISIBLE_DEVICES=2 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
#     --model Qwen/Qwen2.5-7B-Instruct --device cuda --model_dtype fp32 \
#     --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
#     --mode all --batch_size 16 \
#     --n_subspace 128 --n_eval 2048 --layer 10 \
#     --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
#     --reasoning_tokens 128 --max_new_tokens 256 \
#     --template_randomization 1 --shuffle_choices 1 \
#     --do_sample 0 --out_json results/disturb_cot_reason/all_tasks_energy_balance_greedy_results_Qwen2.5-7B-Instruct.json --out_md results/disturb_cot_reason/all_tasks_energy_balance_greedy_summary_Qwen2.5-7B-Instruct.md

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
    --do_sample 0 --out_json results/disturb_cot_reason/energy_balance_loto8_results_Qwen2.5-7B-Instruct.json --out_md results/disturb_cot_reason/energy_balance_loto8_summary_Qwen2.5-7B-Instruct.md 






# run the with-aqua version
  # Single (all-tasks) basis estimation + evaluation:
  CUDA_VISIBLE_DEVICES=2 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
    --model google/gemma-3-12b-it --device cuda --model_dtype fp32 \
    --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
    --mode all --batch_size 16 \
    --n_subspace 128 --n_eval 2048 --layer 10 \
    --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
    --reasoning_tokens 128 --max_new_tokens 256 \
    --template_randomization 1 --shuffle_choices 1 \
    --do_sample 0 --out_json results/disturb_cot_reason/all_tasks_energy_balance_greedy_results_gemma-3-12b-it.json --out_md results/disturb_cot_reason/all_tasks_energy_balance_greedy_summary_gemma-3-12b-it.md

  # LOTO (estimate basis on N-1 tasks, evaluate held-out only):
  CUDA_VISIBLE_DEVICES=2 python disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py \
    --model google/gemma-3-12b-it --device cuda --model_dtype fp32 \
    --tasks gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,qasc,boolq,piqa \
    --mode loto --batch_size 16 \
    --loto_eval_mode heldout \
    --n_subspace 128 --n_eval 2048 --layer 10 \
    --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \
    --reasoning_tokens 128 --max_new_tokens 256 \
    --template_randomization 1 --shuffle_choices 1 \
    --do_sample 0 --out_json results/disturb_cot_reason/energy_balance_loto8_results_gemma-3-12b-it.json --out_md results/disturb_cot_reason/energy_balance_loto8_summary_gemma-3-12b-it.md 






