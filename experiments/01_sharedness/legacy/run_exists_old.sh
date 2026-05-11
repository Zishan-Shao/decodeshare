#!/usr/bin/env bash
set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"

conda activate flashsvd
cd /home/zs89/decodeshare/Hype1

# 建议关掉 tokenizer 并行提示
export TOKENIZERS_PARALLELISM=false

GPU_ID="${GPU_ID:-1}"

# mistralai/Mistral-7B-Instruct-v0.3
# tiiuae/falcon-7b-instruct

#CUDA_VISIBLE_DEVICES=0 python prove_sharedness_decode_full.py   --model mistralai/Mistral-7B-Instruct-v0.3   --device cuda   --model_dtype fp32   --layer 10   --n_prompts 128   --calib_max_new_tokens 128   --max_prompt_len 512   --per_task_max_states 20000   --tau 0.001   --m_shared all   --null_perm_trials 2000   --null_scramble_trials 100   --out_json results/full_benchmark/Mistral-7B-Instruct-v0.3_exist.json   --out_txt  results/full_benchmark/Mistral-7B-Instruct-v0.3_exist.txt

CUDA_VISIBLE_DEVICES=0 python prove_sharedness_decode_full.py   --model tiiuae/falcon-7b-instruct   --device cuda   --model_dtype fp32   --layer 10   --n_prompts 128   --calib_max_new_tokens 128   --max_prompt_len 512   --per_task_max_states 20000   --tau 0.001   --m_shared all   --null_perm_trials 2000   --null_scramble_trials 100   --out_json results/full_benchmark/falcon-7b-instruct_exist.json   --out_txt  results/full_benchmark/falcon-7b-instruct_exist.txt

CUDA_VISIBLE_DEVICES=0 python prove_sharedness_decode_full.py   --model meta-llama/Llama-3.1-8B-Instruct   --device cuda   --model_dtype fp32   --layer 10   --n_prompts 128   --calib_max_new_tokens 128   --max_prompt_len 512   --per_task_max_states 20000   --tau 0.001   --m_shared all   --null_perm_trials 2000   --null_scramble_trials 100   --out_json results/full_benchmark/Llama-3.1-8B-Instruct_exist.json   --out_txt  results/full_benchmark/Llama-3.1-8B-Instruct_exist.txt

CUDA_VISIBLE_DEVICES=0 python prove_sharedness_decode_full.py   --model Qwen/Qwen2.5-1.5B-Instruct   --device cuda   --model_dtype fp32   --layer 10   --n_prompts 128   --calib_max_new_tokens 128   --max_prompt_len 512   --per_task_max_states 20000   --tau 0.001   --m_shared all   --null_perm_trials 2000   --null_scramble_trials 100   --out_json results/full_benchmark/Qwen2.5-1.5B-Instruct_exist.json   --out_txt  results/full_benchmark/Qwen2.5-1.5B-Instruct_exist.txt




CUDA_VISIBLE_DEVICES=0 python prove_sharedness_decode_full.py   --model mistralai/Mistral-7B-Instruct-v0.3   --device cuda   --model_dtype fp32   --layer 10   --n_prompts 128   --calib_max_new_tokens 128   --max_prompt_len 512   --per_task_max_states 20000   --tau 0.001   --m_shared 8   --null_perm_trials 2000   --null_scramble_trials 100   --out_json results/full_benchmark/Mistral-7B-Instruct-v0.3_exist_pooled.json   --out_txt  results/full_benchmark/Mistral-7B-Instruct-v0.3_exist_pooled.txt

CUDA_VISIBLE_DEVICES=0 python prove_sharedness_decode_full.py   --model tiiuae/falcon-7b-instruct   --device cuda   --model_dtype fp32   --layer 10   --n_prompts 128   --calib_max_new_tokens 128   --max_prompt_len 512   --per_task_max_states 20000   --tau 0.001   --m_shared 8   --null_perm_trials 2000   --null_scramble_trials 100   --out_json results/full_benchmark/falcon-7b-instruct_exist_pooled.json   --out_txt  results/full_benchmark/falcon-7b-instruct_exist_pooled.txt

CUDA_VISIBLE_DEVICES=0 python prove_sharedness_decode_full.py   --model meta-llama/Llama-3.1-8B-Instruct   --device cuda   --model_dtype fp32   --layer 10   --n_prompts 128   --calib_max_new_tokens 128   --max_prompt_len 512   --per_task_max_states 20000   --tau 0.001   --m_shared 8   --null_perm_trials 2000   --null_scramble_trials 100   --out_json results/full_benchmark/Llama-3.1-8B-Instruct_exist_pooled.json   --out_txt  results/full_benchmark/Llama-3.1-8B-Instruct_exist_pooled.txt

CUDA_VISIBLE_DEVICES=0 python prove_sharedness_decode_full.py   --model Qwen/Qwen2.5-1.5B-Instruct   --device cuda   --model_dtype fp32   --layer 10   --n_prompts 128   --calib_max_new_tokens 128   --max_prompt_len 512   --per_task_max_states 20000   --tau 0.001   --m_shared 8   --null_perm_trials 2000   --null_scramble_trials 100   --out_json results/full_benchmark/Qwen2.5-1.5B-Instruct_exist_pooled.json   --out_txt  results/full_benchmark/Qwen2.5-1.5B-Instruct_exist_pooled.txt


# run with loosen threshold
CUDA_VISIBLE_DEVICES=0 python prove_sharedness_decode_full.py   --model mistralai/Mistral-7B-Instruct-v0.3   --device cuda   --model_dtype fp32   --layer 10   --n_prompts 128   --calib_max_new_tokens 128   --max_prompt_len 512   --per_task_max_states 20000   --tau 0.0003   --m_shared 8   --null_perm_trials 2000   --null_scramble_trials 100   --out_json results/full_benchmark/Mistral-7B-Instruct-v0.3_exist_loosened.json   --out_txt  results/full_benchmark/Mistral-7B-Instruct-v0.3_exist_loosened.txt

CUDA_VISIBLE_DEVICES=0 python prove_sharedness_decode_full.py   --model tiiuae/falcon-7b-instruct   --device cuda   --model_dtype fp32   --layer 10   --n_prompts 128   --calib_max_new_tokens 128   --max_prompt_len 512   --per_task_max_states 20000   --tau 0.0003   --m_shared 8   --null_perm_trials 2000   --null_scramble_trials 100   --out_json results/full_benchmark/falcon-7b-instruct_exist_loosened.json   --out_txt  results/full_benchmark/falcon-7b-instruct_exist_loosened.txt

CUDA_VISIBLE_DEVICES=0 python prove_sharedness_decode_full.py   --model meta-llama/Llama-3.1-8B-Instruct   --device cuda   --model_dtype fp32   --layer 10   --n_prompts 128   --calib_max_new_tokens 128   --max_prompt_len 512   --per_task_max_states 20000   --tau 0.0003   --m_shared 8   --null_perm_trials 2000   --null_scramble_trials 100   --out_json results/full_benchmark/Llama-3.1-8B-Instruct_exist_loosened.json   --out_txt  results/full_benchmark/Llama-3.1-8B-Instruct_exist_loosened.txt

CUDA_VISIBLE_DEVICES=0 python prove_sharedness_decode_full.py   --model Qwen/Qwen2.5-1.5B-Instruct   --device cuda   --model_dtype fp32   --layer 10   --n_prompts 128   --calib_max_new_tokens 128   --max_prompt_len 512   --per_task_max_states 20000   --tau 0.0003   --m_shared 8   --null_perm_trials 2000   --null_scramble_trials 100   --out_json results/full_benchmark/Qwen2.5-1.5B-Instruct_exist_loosened.json   --out_txt  results/full_benchmark/Qwen2.5-1.5B-Instruct_exist_loosened.txt





# CUDA_VISIBLE_DEVICES=1 python prove_sharedness_decode_full.py   --model google/gemma-3-12b-it   --device cuda   --model_dtype fp32   --layer 10   --n_prompts 128   --calib_max_new_tokens 128   --max_prompt_len 512   --per_task_max_states 20000   --tau 0.001   --m_shared all   --null_perm_trials 2000   --null_scramble_trials 100   --out_json results/full_benchmark/gemma3_exist.json   --out_txt  results/full_benchmark/gemma3_exist.txt

# CUDA_VISIBLE_DEVICES=1 python prove_sharedness_decode_full.py   --model meta-llama/Llama-2-7b-chat-hf   --device cuda   --model_dtype fp32   --layer 10   --n_prompts 128   --calib_max_new_tokens 128   --max_prompt_len 512   --per_task_max_states 20000   --tau 0.001   --m_shared all   --null_perm_trials 2000   --null_scramble_trials 100   --out_json results/full_benchmark/llama2-7b-chat-hf_exist.json   --out_txt  results/full_benchmark/llama2-7b-chat-hf_exist.txt

# CUDA_VISIBLE_DEVICES=1 python prove_sharedness_decode_full.py   --model Qwen/Qwen2.5-7B-Instruct   --device cuda   --model_dtype fp32   --layer 10   --n_prompts 128   --calib_max_new_tokens 128   --max_prompt_len 512   --per_task_max_states 20000   --tau 0.001   --m_shared all   --null_perm_trials 2000   --null_scramble_trials 100   --out_json results/full_benchmark/Qwen2.5-7B-Instruct_exist.json   --out_txt  results/full_benchmark/Qwen2.5-7B-Instruct_exist.txt



# CUDA_VISIBLE_DEVICES=1 python prove_sharedness_decode_full.py   --model google/gemma-3-12b-it   --device cuda   --model_dtype fp32   --layer 10   --n_prompts 128   --calib_max_new_tokens 128   --max_prompt_len 512   --per_task_max_states 20000   --tau 0.001   --m_shared 8   --null_perm_trials 2000   --null_scramble_trials 100   --out_json results/full_benchmark/gemma3_exist.json   --out_txt  results/full_benchmark/gemma3_exist_pooled.txt

# CUDA_VISIBLE_DEVICES=1 python prove_sharedness_decode_full.py   --model meta-llama/Llama-2-7b-chat-hf   --device cuda   --model_dtype fp32   --layer 10   --n_prompts 128   --calib_max_new_tokens 128   --max_prompt_len 512   --per_task_max_states 20000   --tau 0.001   --m_shared 8   --null_perm_trials 2000   --null_scramble_trials 100   --out_json results/full_benchmark/llama2-7b-chat-hf_exist.json   --out_txt  results/full_benchmark/llama2-7b-chat-hf_exist_pooled.txt

# CUDA_VISIBLE_DEVICES=1 python prove_sharedness_decode_full.py   --model Qwen/Qwen2.5-7B-Instruct   --device cuda   --model_dtype fp32   --layer 10   --n_prompts 128   --calib_max_new_tokens 128   --max_prompt_len 512   --per_task_max_states 20000   --tau 0.001   --m_shared 8   --null_perm_trials 2000   --null_scramble_trials 100   --out_json results/full_benchmark/Qwen2.5-7B-Instruct_exist.json   --out_txt  results/full_benchmark/Qwen2.5-7B-Instruct_exist_pooled.txt


# # run with loosen threshold
# CUDA_VISIBLE_DEVICES=1 python prove_sharedness_decode_full.py   --model google/gemma-3-12b-it   --device cuda   --model_dtype fp32   --layer 10   --n_prompts 128   --calib_max_new_tokens 128   --max_prompt_len 512   --per_task_max_states 20000   --tau 0.0003   --m_shared 8   --null_perm_trials 2000   --null_scramble_trials 100   --out_json results/full_benchmark/gemma3_exist.json   --out_txt  results/full_benchmark/gemma3_exist_loosened.txt

# CUDA_VISIBLE_DEVICES=1 python prove_sharedness_decode_full.py   --model meta-llama/Llama-2-7b-chat-hf   --device cuda   --model_dtype fp32   --layer 10   --n_prompts 128   --calib_max_new_tokens 128   --max_prompt_len 512   --per_task_max_states 20000   --tau 0.0003   --m_shared 8   --null_perm_trials 2000   --null_scramble_trials 100   --out_json results/full_benchmark/llama2-7b-chat-hf_exist.json   --out_txt  results/full_benchmark/llama2-7b-chat-hf_exist_loosened.txt

# CUDA_VISIBLE_DEVICES=1 python prove_sharedness_decode_full.py   --model Qwen/Qwen2.5-7B-Instruct   --device cuda   --model_dtype fp32   --layer 10   --n_prompts 128   --calib_max_new_tokens 128   --max_prompt_len 512   --per_task_max_states 20000   --tau 0.0003   --m_shared 8   --null_perm_trials 2000   --null_scramble_trials 100   --out_json results/full_benchmark/Qwen2.5-7B-Instruct_exist.json   --out_txt  results/full_benchmark/Qwen2.5-7B-Instruct_exist_loosened.txt
