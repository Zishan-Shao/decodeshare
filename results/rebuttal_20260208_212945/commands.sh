# Auto-generated: 2026-02-08T21:29:45-05:00

# OUT_DIR
/home/zs89/decodeshare/results/rebuttal_20260208_212945

# Ranking flip
python /home/zs89/decodeshare/rebuttal/exp_ranking_flip_steering.py --model meta-llama/Llama-2-7b-chat-hf --device cuda --model_dtype fp32 --vectors_manifest /home/zs89/decodeshare/rebuttal/steering_vectors_layer28.jsonl --max_vectors 0 --tasks commonsenseqa\,arc_challenge\,openbookqa\,qasc\,logiqa --n_eval 128 --template_seeds_rank 1234\,2345\,3456 --template_seeds_real 4567\,5678\,6789 --decoding greedy --max_new_tokens 256 --reasoning_tokens 128 --batch_size 4 --max_prompt_len 512 --sample_seed 12345 --trad_mode prefill --decode_mode decode --staged 1 --agg mean --seed 42 --out_json /home/zs89/decodeshare/results/rebuttal_20260208_212945/ranking_flip.json 

# Repair controls
python /home/zs89/decodeshare/rebuttal/exp_repair_controls_steering.py --model meta-llama/Llama-2-7b-chat-hf --device cuda --model_dtype fp32 --vectors_manifest /home/zs89/decodeshare/rebuttal/steering_vectors_layer28.jsonl --max_vectors 0 --tasks_subspace gsm8k\,commonsenseqa\,strategyqa\,aqua\,arc_challenge\,openbookqa\,qasc\,logiqa --n_subspace 128 --tasks_eval commonsenseqa\,arc_challenge\,openbookqa\,qasc\,logiqa --n_eval 128 --template_seeds 1234\,2345\,3456\,4567\,5678 --decoding greedy --max_new_tokens 256 --reasoning_tokens 128 --batch_size 4 --max_prompt_len 512 --sample_seed 12345 --staged 1 --alpha_proj 1.0 --norm_match 1 --tau 0.001 --m_shared all --shared_dim 1023 --pca_var 0.95 --pca_max_rows 200000 --pca_max_dim 4096 --per_task_max_states 20000 --calib_decode_max_new_tokens -1 --include_pca_prefill 1 --seed 42 --out_json /home/zs89/decodeshare/results/rebuttal_20260208_212945/repair_controls.json 

