#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Env / paths
# -----------------------------
WORKDIR="reasoning"
cd "${WORKDIR}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate flashsvd

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1


MODEL_NAMES=(
  "tiiuae/falcon-7b-instruct"
  "mistralai/Mistral-7B-Instruct-v0.3"
  "meta-llama/Llama-3.1-8B-Instruct"
  "meta-llama/Llama-3.2-3B-Instruct"
  "Qwen/Qwen2.5-1.5B-Instruct"
)

LAYERS=(4 10 24)


# # llama-2-7b-chat-hf
# CUDA_VISIBLE_DEVICES=0 python h3_killer_counterfactual_grid_reasoning_v2.py   --model meta-llama/Llama-2-7b-chat-hf  --device cuda --model_dtype fp32   --tasks gsm8k,commonsenseqa,strategyqa,piqa,arc_challenge,openbookqa,qasc,logiqa,boolq   --layer 4 --n_subspace 128 --n_eval 2048   --calib_decode_max_new_tokens 512 --per_task_max_states 20000   --answer_prefix $'\nFinal answer:'   --warmup_tokens 0   --template_randomization 1 --shuffle_choices 1

# CUDA_VISIBLE_DEVICES=0 python h3_killer_counterfactual_grid_reasoning_v2.py   --model meta-llama/Llama-2-7b-chat-hf  --device cuda --model_dtype fp32   --tasks gsm8k,commonsenseqa,strategyqa,piqa,arc_challenge,openbookqa,qasc,logiqa,boolq   --layer 24 --n_subspace 128 --n_eval 2048   --calib_decode_max_new_tokens 512 --per_task_max_states 20000   --answer_prefix $'\nFinal answer:'   --warmup_tokens 0   --template_randomization 1 --shuffle_choices 1



# # Qwen/Qwen2.5-7B-Instruct
# CUDA_VISIBLE_DEVICES=0 python h3_killer_counterfactual_grid_reasoning_v2.py   --model Qwen/Qwen2.5-7B-Instruct  --device cuda --model_dtype fp32   --tasks gsm8k,commonsenseqa,strategyqa,piqa,arc_challenge,openbookqa,qasc,logiqa,boolq   --layer 4 --n_subspace 128 --n_eval 2048   --calib_decode_max_new_tokens 512 --per_task_max_states 20000   --answer_prefix $'\nFinal answer:'   --warmup_tokens 0   --template_randomization 1 --shuffle_choices 1

# CUDA_VISIBLE_DEVICES=0 python h3_killer_counterfactual_grid_reasoning_v2.py   --model Qwen/Qwen2.5-7B-Instruct  --device cuda --model_dtype fp32   --tasks gsm8k,commonsenseqa,strategyqa,piqa,arc_challenge,openbookqa,qasc,logiqa,boolq   --layer 24 --n_subspace 128 --n_eval 2048   --calib_decode_max_new_tokens 512 --per_task_max_states 20000   --answer_prefix $'\nFinal answer:'   --warmup_tokens 0   --template_randomization 1 --shuffle_choices 1



# tiiuae/falcon-7b-instruct
CUDA_VISIBLE_DEVICES=2 python h3_killer_counterfactual_grid_reasoning_v2_falcon.py   --model tiiuae/falcon-7b-instruct  --device cuda --model_dtype fp32   --tasks gsm8k,commonsenseqa,strategyqa,piqa,arc_challenge,openbookqa,qasc,logiqa,boolq   --layer 4 --n_subspace 128 --n_eval 2048   --calib_decode_max_new_tokens 512 --per_task_max_states 20000   --answer_prefix $'\nFinal answer:'   --warmup_tokens 0   --template_randomization 1 --shuffle_choices 1

CUDA_VISIBLE_DEVICES=2 python h3_killer_counterfactual_grid_reasoning_v2_falcon.py   --model tiiuae/falcon-7b-instruct  --device cuda --model_dtype fp32   --tasks gsm8k,commonsenseqa,strategyqa,piqa,arc_challenge,openbookqa,qasc,logiqa,boolq   --layer 10 --n_subspace 128 --n_eval 2048   --calib_decode_max_new_tokens 512 --per_task_max_states 20000   --answer_prefix $'\nFinal answer:'   --warmup_tokens 0   --template_randomization 1 --shuffle_choices 1

CUDA_VISIBLE_DEVICES=2 python h3_killer_counterfactual_grid_reasoning_v2_falcon.py   --model tiiuae/falcon-7b-instruct  --device cuda --model_dtype fp32   --tasks gsm8k,commonsenseqa,strategyqa,piqa,arc_challenge,openbookqa,qasc,logiqa,boolq   --layer 24 --n_subspace 128 --n_eval 2048   --calib_decode_max_new_tokens 512 --per_task_max_states 20000   --answer_prefix $'\nFinal answer:'   --warmup_tokens 0   --template_randomization 1 --shuffle_choices 1



# llama-3.1-8b-instruct
CUDA_VISIBLE_DEVICES=2 python h3_killer_counterfactual_grid_reasoning_v2.py   --model meta-llama/Llama-3.1-8B-Instruct  --device cuda --model_dtype fp32   --tasks gsm8k,commonsenseqa,strategyqa,piqa,arc_challenge,openbookqa,qasc,logiqa,boolq   --layer 4 --n_subspace 128 --n_eval 2048   --calib_decode_max_new_tokens 512 --per_task_max_states 20000   --answer_prefix $'\nFinal answer:'   --warmup_tokens 0   --template_randomization 1 --shuffle_choices 1

CUDA_VISIBLE_DEVICES=2 python h3_killer_counterfactual_grid_reasoning_v2.py   --model meta-llama/Llama-3.1-8B-Instruct  --device cuda --model_dtype fp32   --tasks gsm8k,commonsenseqa,strategyqa,piqa,arc_challenge,openbookqa,qasc,logiqa,boolq   --layer 10 --n_subspace 128 --n_eval 2048   --calib_decode_max_new_tokens 512 --per_task_max_states 20000   --answer_prefix $'\nFinal answer:'   --warmup_tokens 0   --template_randomization 1 --shuffle_choices 1

CUDA_VISIBLE_DEVICES=2 python h3_killer_counterfactual_grid_reasoning_v2.py   --model meta-llama/Llama-3.1-8B-Instruct  --device cuda --model_dtype fp32   --tasks gsm8k,commonsenseqa,strategyqa,piqa,arc_challenge,openbookqa,qasc,logiqa,boolq   --layer 24 --n_subspace 128 --n_eval 2048   --calib_decode_max_new_tokens 512 --per_task_max_states 20000   --answer_prefix $'\nFinal answer:'   --warmup_tokens 0   --template_randomization 1 --shuffle_choices 1



# # mistralai/Mistral-7B-Instruct-v0.3
# CUDA_VISIBLE_DEVICES=0 python h3_killer_counterfactual_grid_reasoning_v2.py   --model mistralai/Mistral-7B-Instruct-v0.3  --device cuda --model_dtype fp32   --tasks gsm8k,commonsenseqa,strategyqa,piqa,arc_challenge,openbookqa,qasc,logiqa,boolq   --layer 4 --n_subspace 128 --n_eval 2048   --calib_decode_max_new_tokens 512 --per_task_max_states 20000   --answer_prefix $'\nFinal answer:'   --warmup_tokens 0   --template_randomization 1 --shuffle_choices 1

# CUDA_VISIBLE_DEVICES=0 python h3_killer_counterfactual_grid_reasoning_v2.py   --model mistralai/Mistral-7B-Instruct-v0.3  --device cuda --model_dtype fp32   --tasks gsm8k,commonsenseqa,strategyqa,piqa,arc_challenge,openbookqa,qasc,logiqa,boolq   --layer 10 --n_subspace 128 --n_eval 2048   --calib_decode_max_new_tokens 512 --per_task_max_states 20000   --answer_prefix $'\nFinal answer:'   --warmup_tokens 0   --template_randomization 1 --shuffle_choices 1

# CUDA_VISIBLE_DEVICES=0 python h3_killer_counterfactual_grid_reasoning_v2.py   --model mistralai/Mistral-7B-Instruct-v0.3  --device cuda --model_dtype fp32   --tasks gsm8k,commonsenseqa,strategyqa,piqa,arc_challenge,openbookqa,qasc,logiqa,boolq   --layer 24 --n_subspace 128 --n_eval 2048   --calib_decode_max_new_tokens 512 --per_task_max_states 20000   --answer_prefix $'\nFinal answer:'   --warmup_tokens 0   --template_randomization 1 --shuffle_choices 1




# # llama-3.2-3b-instruct
# CUDA_VISIBLE_DEVICES=0 python h3_killer_counterfactual_grid_reasoning_v2.py   --model meta-llama/Llama-3.2-3B-Instruct  --device cuda --model_dtype fp32   --tasks gsm8k,commonsenseqa,strategyqa,piqa,arc_challenge,openbookqa,qasc,logiqa,boolq   --layer 4 --n_subspace 128 --n_eval 2048   --calib_decode_max_new_tokens 512 --per_task_max_states 20000   --answer_prefix $'\nFinal answer:'   --warmup_tokens 0   --template_randomization 1 --shuffle_choices 1

# CUDA_VISIBLE_DEVICES=0 python h3_killer_counterfactual_grid_reasoning_v2.py   --model meta-llama/Llama-3.2-3B-Instruct  --device cuda --model_dtype fp32   --tasks gsm8k,commonsenseqa,strategyqa,piqa,arc_challenge,openbookqa,qasc,logiqa,boolq   --layer 10 --n_subspace 128 --n_eval 2048   --calib_decode_max_new_tokens 512 --per_task_max_states 20000   --answer_prefix $'\nFinal answer:'   --warmup_tokens 0   --template_randomization 1 --shuffle_choices 1

# CUDA_VISIBLE_DEVICES=0 python h3_killer_counterfactual_grid_reasoning_v2.py   --model meta-llama/Llama-3.2-3B-Instruct  --device cuda --model_dtype fp32   --tasks gsm8k,commonsenseqa,strategyqa,piqa,arc_challenge,openbookqa,qasc,logiqa,boolq   --layer 24 --n_subspace 128 --n_eval 2048   --calib_decode_max_new_tokens 512 --per_task_max_states 20000   --answer_prefix $'\nFinal answer:'   --warmup_tokens 0   --template_randomization 1 --shuffle_choices 1



# # Qwen/Qwen2.5-1.5B-Instruct
# CUDA_VISIBLE_DEVICES=0 python h3_killer_counterfactual_grid_reasoning_v2.py   --model Qwen/Qwen2.5-1.5B-Instruct  --device cuda --model_dtype fp32   --tasks gsm8k,commonsenseqa,strategyqa,piqa,arc_challenge,openbookqa,qasc,logiqa,boolq   --layer 4 --n_subspace 128 --n_eval 2048   --calib_decode_max_new_tokens 512 --per_task_max_states 20000   --answer_prefix $'\nFinal answer:'   --warmup_tokens 0   --template_randomization 1 --shuffle_choices 1

# CUDA_VISIBLE_DEVICES=0 python h3_killer_counterfactual_grid_reasoning_v2.py   --model Qwen/Qwen2.5-1.5B-Instruct  --device cuda --model_dtype fp32   --tasks gsm8k,commonsenseqa,strategyqa,piqa,arc_challenge,openbookqa,qasc,logiqa,boolq   --layer 10 --n_subspace 128 --n_eval 2048   --calib_decode_max_new_tokens 512 --per_task_max_states 20000   --answer_prefix $'\nFinal answer:'   --warmup_tokens 0   --template_randomization 1 --shuffle_choices 1

# CUDA_VISIBLE_DEVICES=0 python h3_killer_counterfactual_grid_reasoning_v2.py   --model Qwen/Qwen2.5-1.5B-Instruct  --device cuda --model_dtype fp32   --tasks gsm8k,commonsenseqa,strategyqa,piqa,arc_challenge,openbookqa,qasc,logiqa,boolq   --layer 24 --n_subspace 128 --n_eval 2048   --calib_decode_max_new_tokens 512 --per_task_max_states 20000   --answer_prefix $'\nFinal answer:'   --warmup_tokens 0   --template_randomization 1 --shuffle_choices 1



cd analysis

python analyze_h3_grid_json.py --inputs "results/h3_grid/h3_grid_v3_meta-llama_Llama-2-7b-chat-hf_layer4_k48_W0_seed0.json" --out_csv results/llama2-7b-chat-hf/llama2-7b-chat-hf_layer4_out.csv --out_latex results/llama2-7b-chat-hf/llama2-7b-chat-hf_layer4_out.tex --latex_mode acc

python analyze_h3_grid_json.py --inputs "results/h3_grid/h3_grid_v3_meta-llama_Llama-2-7b-chat-hf_layer24_k48_W0_seed0.json" --out_csv results/llama2-7b-chat-hf/llama2-7b-chat-hf_layer24_out.csv --out_latex results/llama2-7b-chat-hf/llama2-7b-chat-hf_layer24_out.tex --latex_mode acc


python analyze_h3_grid_json.py --inputs "results/h3_grid/h3_grid_v3_Qwen_Qwen2.5-7B-Instruct_layer4_k20_W0_seed0.json" --out_csv results/Qwen2.5-7B-Instruct/Qwen2.5-7B-Instruct_layer4_out.csv --out_latex results/Qwen2.5-7B-Instruct/Qwen2.5-7B-Instruct_layer4_out.tex --latex_mode acc

python analyze_h3_grid_json.py --inputs "results/h3_grid/h3_grid_v3_Qwen_Qwen2.5-7B-Instruct_layer24_k20_W0_seed0.json" --out_csv results/Qwen2.5-7B-Instruct/Qwen2.5-7B-Instruct_layer24_out.csv --out_latex results/Qwen2.5-7B-Instruct/Qwen2.5-7B-Instruct_layer24_out.tex --latex_mode acc



python analyze_h3_grid_json.py --inputs "results/h3_grid/h3_grid_v3_tiiuae_falcon-7b-instruct_layer4_k48_W0_seed0.json" --out_csv results/falcon-7b-instruct/falcon-7b-instruct_layer4_out.csv --out_latex results/falcon-7b-instruct/falcon-7b-instruct_layer4_out.tex --latex_mode acc

python analyze_h3_grid_json.py --inputs "results/h3_grid/h3_grid_v3_tiiuae_falcon-7b-instruct_layer10_k48_W0_seed0.json" --out_csv results/falcon-7b-instruct/falcon-7b-instruct_layer10_out.csv --out_latex results/falcon-7b-instruct/falcon-7b-instruct_layer10_out.tex --latex_mode acc

python analyze_h3_grid_json.py --inputs "results/h3_grid/h3_grid_v3_tiiuae_falcon-7b-instruct_layer24_k48_W0_seed0.json" --out_csv results/falcon-7b-instruct/falcon-7b-instruct_layer24_out.csv --out_latex results/falcon-7b-instruct/falcon-7b-instruct_layer24_out.tex --latex_mode acc



python analyze_h3_grid_json.py --inputs "results/h3_grid/h3_grid_v3_mistralai_Mistral-7B-Instruct-v0.3_layer4_k20_W0_seed0.json" --out_csv results/Mistral-7B-Instruct-v0.3/Mistral-7B-Instruct-v0.3_layer4_out.csv --out_latex results/Mistral-7B-Instruct-v0.3/Mistral-7B-Instruct-v0.3_layer4_out.tex --latex_mode acc

python analyze_h3_grid_json.py --inputs "results/h3_grid/h3_grid_v3_mistralai_Mistral-7B-Instruct-v0.3_layer10_k20_W0_seed0.json" --out_csv results/Mistral-7B-Instruct-v0.3/Mistral-7B-Instruct-v0.3_layer10_out.csv --out_latex results/Mistral-7B-Instruct-v0.3/Mistral-7B-Instruct-v0.3_layer10_out.tex --latex_mode acc

python analyze_h3_grid_json.py --inputs "results/h3_grid/h3_grid_v3_mistralai_Mistral-7B-Instruct-v0.3_layer24_k20_W0_seed0.json" --out_csv results/Mistral-7B-Instruct-v0.3/Mistral-7B-Instruct-v0.3_layer24_out.csv --out_latex results/Mistral-7B-Instruct-v0.3/Mistral-7B-Instruct-v0.3_layer24_out.tex --latex_mode acc



python analyze_h3_grid_json.py --inputs "results/h3_grid/h3_grid_v3_meta-llama_Llama-3.1-8B-Instruct_layer4_k48_W0_seed0.json" --out_csv results/Llama-3.1-8B-Instruct/Llama-3.1-8B-Instruct_layer4_out.csv --out_latex results/Llama-3.1-8B-Instruct/Llama-3.1-8B-Instruct_layer4_out.tex --latex_mode acc

python analyze_h3_grid_json.py --inputs "results/h3_grid/h3_grid_v3_meta-llama_Llama-3.1-8B-Instruct_layer10_k48_W0_seed0.json" --out_csv results/Llama-3.1-8B-Instruct/Llama-3.1-8B-Instruct_layer10_out.csv --out_latex results/Llama-3.1-8B-Instruct/Llama-3.1-8B-Instruct_layer10_out.tex --latex_mode acc

python analyze_h3_grid_json.py --inputs "results/h3_grid/h3_grid_v3_meta-llama_Llama-3.1-8B-Instruct_layer24_k48_W0_seed0.json" --out_csv results/Llama-3.1-8B-Instruct/Llama-3.1-8B-Instruct_layer24_out.csv --out_latex results/Llama-3.1-8B-Instruct/Llama-3.1-8B-Instruct_layer24_out.tex --latex_mode acc



python analyze_h3_grid_json.py --inputs "results/h3_grid/h3_grid_v3_meta-llama_Llama-3.2-3B-Instruct_layer4_k48_W0_seed0.json" --out_csv results/Llama-3.2-3B-Instruct/Llama-3.2-3B-Instruct_layer4_out.csv --out_latex results/Llama-3.2-3B-Instruct/Llama-3.2-3B-Instruct_layer4_out.tex --latex_mode acc

python analyze_h3_grid_json.py --inputs "results/h3_grid/h3_grid_v3_meta-llama_Llama-3.2-3B-Instruct_layer10_k48_W0_seed0.json" --out_csv results/Llama-3.2-3B-Instruct/Llama-3.2-3B-Instruct_layer10_out.csv --out_latex results/Llama-3.2-3B-Instruct/Llama-3.2-3B-Instruct_layer10_out.tex --latex_mode acc

python analyze_h3_grid_json.py --inputs "results/h3_grid/h3_grid_v3_meta-llama_Llama-3.2-3B-Instruct_layer24_k48_W0_seed0.json" --out_csv results/Llama-3.2-3B-Instruct/Llama-3.2-3B-Instruct_layer24_out.csv --out_latex results/Llama-3.2-3B-Instruct/Llama-3.2-3B-Instruct_layer24_out.tex --latex_mode acc
