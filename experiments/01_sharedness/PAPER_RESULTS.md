# H1 Paper Result Map

This file maps paper-facing H1 outputs to code and checked-in artifacts.

## Main And Appendix Tables

Paper outputs: Table 6 and appendix Tables 7-13.

Canonical generation path:

```bash
cd /home/zs89/decodeshare-camera-ready
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
python experiments/01_sharedness/summarize_full_benchmark.py \
  --results_dir paper_artifacts/h1_results/results/full_benchmark \
  --out_dir paper_artifacts/h1_results/results/full_benchmark \
  --alpha 0.05
```

Inputs:

- `paper_artifacts/h1_results/results/full_benchmark/*_exist*.json`
- `paper_artifacts/h1_results/results/full_benchmark/*_exist*.txt`

Outputs:

- `paper_artifacts/h1_results/results/full_benchmark/H1_full_benchmark_summary.csv`
- `paper_artifacts/h1_results/results/full_benchmark/H1_full_benchmark_summary.md`
- `paper_artifacts/h1_results/results/full_benchmark/H1_full_benchmark_summary.tex`
- `paper_artifacts/h1_results/results/full_benchmark/H1_evidence_chain.tex`

## Full Benchmark Rerun Template

Run on `Node0` or `Node1` only.

```bash
cd /home/zs89/decodeshare-camera-ready/experiments/01_sharedness
export PYTHONPATH=/home/zs89/decodeshare-camera-ready/src:${PYTHONPATH:-}
CUDA_VISIBLE_DEVICES=0 python run_full_benchmark.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --device cuda \
  --model_dtype fp32 \
  --layer 10 \
  --n_prompts 128 \
  --calib_max_new_tokens 128 \
  --max_prompt_len 512 \
  --per_task_max_states 20000 \
  --tau 0.001 \
  --m_shared all \
  --null_perm_trials 2000 \
  --null_scramble_trials 100 \
  --out_json ../../outputs/01_sharedness/full_benchmark/Qwen2.5-7B-Instruct_exist.json \
  --out_txt ../../outputs/01_sharedness/full_benchmark/Qwen2.5-7B-Instruct_exist.txt
```

The paper result records were produced with the parameter families documented in `configs/full_benchmark.yaml`.

## Diagnostic Figures

Paper outputs: main Figures 2-4 and appendix Figures 8, 11-14.

The checked-in diagnostic artifacts are:

- `paper_artifacts/h1_results/results/exp1/*.csv` and `*.png`
- `paper_artifacts/h1_results/results/exp2/*.csv` and `*.png`
- `paper_artifacts/h1_results/results/exp2.75/*.csv` and `*.png`
- `paper_artifacts/h1_results/results/exp3/*.csv` and `*.png`

Canonical diagnostic pipeline:

```bash
cd /home/zs89/decodeshare-camera-ready/experiments/01_sharedness
export PYTHONPATH=/home/zs89/decodeshare-camera-ready/src:${PYTHONPATH:-}

CUDA_VISIBLE_DEVICES=0 python collect_activations.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --device cuda \
  --model_dtype fp32 \
  --layer 10 \
  --n_prompts 128 \
  --calib_max_new_tokens 256 \
  --max_prompt_len 512 \
  --per_task_max_states 20000 \
  --seed 42 \
  --out_dir ../../outputs/01_sharedness/acts/llama2_layer10_seed42 \
  --save_dtype fp16

python analyze_within_vs_mixed.py \
  --acts_dir ../../outputs/01_sharedness/acts/llama2_layer10_seed42 \
  --pca_var 0.95 \
  --tau 0.001 \
  --n_mixed 50 \
  --seed 123 \
  --out_csv ../../outputs/01_sharedness/exp1/llama_within_vs_mixed.csv \
  --out_png ../../outputs/01_sharedness/exp1/llama_within_vs_mixed.png

python analyze_task_count_convergence.py \
  --acts_dir ../../outputs/01_sharedness/acts/llama2_layer10_seed42 \
  --pca_var 0.95 \
  --repeats 20 \
  --seed 123 \
  --out_csv ../../outputs/01_sharedness/exp2/llama_convergence.csv \
  --out_png ../../outputs/01_sharedness/exp2/llama_convergence.png

python analyze_tau_sensitivity.py \
  --acts_dir ../../outputs/01_sharedness/acts/llama2_layer10_seed42 \
  --pca_vars 0.8,0.9,0.95,0.97,0.99 \
  --taus 1e-4,2e-4,5e-4,1e-3,2e-3,5e-3,1e-2 \
  --seed 123 \
  --out_csv ../../outputs/01_sharedness/exp3/llama_sensitivity.csv \
  --out_png ../../outputs/01_sharedness/exp3/llama_sensitivity.png
```

`analyze_phase_convergence.py` is the canonical script for the exp2.75 decode/prefill/decode-step diagnostic.
