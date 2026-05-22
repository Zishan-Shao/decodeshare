# H1 Paper Result Map

This file maps paper-facing H1 outputs to code and public rerun commands.

## Main And Appendix Tables

Paper outputs: Table 6 and appendix Tables 7-13.

Canonical rerun:

```bash
bash scripts/reproduce_h1_tables.sh
```

Lower-level command:

```bash
bash scripts/full_runs/run_h1_full_benchmark.sh
```

Default outputs:

```text
outputs/01_sharedness/full_benchmark/
```

The paper result parameter families are documented in
`configs/full_benchmark.yaml`.

## Diagnostic Figures

Paper outputs: main Figures 2-4 and appendix Figures 8, 11-14.

Generated diagnostic artifacts should be written under:

```text
outputs/01_sharedness/
```

Canonical diagnostic pipeline:

```bash
cd experiments/01_sharedness
export PYTHONPATH="../..:${PYTHONPATH:-}"

CUDA_VISIBLE_DEVICES="${GPU_ID:-0}" python collect_activations.py \
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

`analyze_phase_convergence.py` is the canonical script for the exp2.75
decode/prefill/decode-step diagnostic.
