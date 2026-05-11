# 04 H3 Prefill-Decode Mismatch

Paper outputs:

- Main: Table 3.
- Appendix: Tables 16-19; Figure 14.

## Canonical Scripts

- `experiments/04_prefill_decode/run_h3_grid_reasoning_v2.py`
- `experiments/04_prefill_decode/run_prefill_decode_reasoning.py`
- `experiments/04_prefill_decode/summarize_h3_grid.py`
- `scripts/full_runs/run_prefill_decode_nextsteps.sh`

Note: `run_h3_grid_reasoning_v2.py` has a docstring that describes the v3 2x2 grid. Keep this mismatch documented or add a wrapper before final artifact release.

## Local Results Found

Compact outputs in the original workspace:

- `/home/zs89/decodeshare/results/h3_grid/h3_grid_reasoning.md`
- `/home/zs89/decodeshare/results/h3_grid/out.tex`
- `/home/zs89/decodeshare/results/h3_grid/h3_grid_v3_meta-llama_Llama-2-7b-chat-hf_layer10_k48_W0_seed0.json`
- `/home/zs89/decodeshare/results/h3_grid/h3_grid_v3_Qwen_Qwen2.5-7B-Instruct_layer10_k20_W0_seed0.json`
- `/home/zs89/decodeshare/results/prefill_decode_nextsteps/k_16.md`
- `/home/zs89/decodeshare/results/prefill_decode_nextsteps/k_126.md`
- `/home/zs89/decodeshare/results/prefill_decode_nextsteps/alpha_*.md`

## Full-Run Command

2x2 grid:

```bash
cd /home/zs89/decodeshare-camera-ready/experiments/04_prefill_decode
CUDA_VISIBLE_DEVICES=0 python run_h3_grid_reasoning_v2.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --device cuda \
  --model_dtype fp32 \
  --tasks gsm8k,commonsenseqa,strategyqa,piqa,arc_challenge,openbookqa,qasc,logiqa,boolq \
  --layer 10 \
  --n_subspace 128 \
  --n_eval 2048 \
  --calib_decode_max_new_tokens 512 \
  --per_task_max_states 20000 \
  --answer_prefix $'\nFinal answer:' \
  --warmup_tokens 0 \
  --template_randomization 1 \
  --shuffle_choices 1 \
  --out_json ../../outputs/h3_grid/h3_grid_v3_meta-llama_Llama-2-7b-chat-hf_layer10_k48_W0_seed0.json
```

K/alpha sweeps:

```bash
cd /home/zs89/decodeshare-camera-ready
bash scripts/full_runs/run_prefill_decode_nextsteps.sh
```

Table extraction:

```bash
python experiments/04_prefill_decode/summarize_h3_grid.py \
  --inputs /home/zs89/decodeshare/results/h3_grid/h3_grid_v3_meta-llama_Llama-2-7b-chat-hf_layer10_k48_W0_seed0.json \
  --out_csv outputs/h3_grid/out.csv \
  --out_latex outputs/h3_grid/out.tex \
  --latex_mode acc
```

## Mock-Test Scope

`run_mock.sh` checks CLI availability, table extraction help, and local result-summary presence. It does not run model inference.
