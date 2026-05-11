# 02 H2 Decode-Only Ablation and Energy Controls

Paper outputs:

- Main: Figure 7.
- Appendix: Figures 9-10; Tables 5, 26-28.

## Canonical Scripts

- `experiments/02_decode_ablation/run_loto_reasoning.py`
- `experiments/02_decode_ablation/run_energy_kmatch_reasoning.py`
- `experiments/02_decode_ablation/summarize_disturb_cot_results.py`
- `experiments/02_decode_ablation/summarize_disturb_cot_diagnostics.py`
- `experiments/02_decode_ablation/summarize_energy_kmatch_outputs.py`
- `scripts/full_runs/run_disturb_cot_loto8_fc_reason.sh`
- `scripts/full_runs/run_alpha_kmatch_sweep.sh`

The canonical LOTO runner is the forced-choice capable version. Older generation-only and experimental refactored variants are retained under `experiments/02_decode_ablation/legacy/` and are not paper reproduction entry points.

The matching parameter record is `experiments/02_decode_ablation/configs/loto_forced_choice.yaml`.

## Local Results Found

Compact summaries in the original workspace:

- `/home/zs89/decodeshare/results/disturb_cot_reasoning/energy_balance_loto8_reasoning_fc_eval2048.md`
- `/home/zs89/decodeshare/results/disturb_cot_reasoning/energy_balance_loto8_reasoning_fc.md`
- `/home/zs89/decodeshare/results/disturb_cot_reasoning/DIAGNOSTIC_SUMMARY.md`
- `/home/zs89/decodeshare/results/energy_kmatch_alpha_sweep/meta-llama_Llama-2-7b-chat-hf_L10_seed42_ts20260110_080440.md`
- `/home/zs89/decodeshare/results/energy_kmatch_alpha_sweep/meta-llama_Llama-2-7b-chat-hf_L10_seed42_ts20260110_080440.tex`

Raw JSONs under `results/disturb_cot_reasoning/` are multi-GB and should stay external unless an artifact policy requires them.

## LOTO Full-Run Command

```bash
cd /home/zs89/decodeshare-camera-ready/experiments/02_decode_ablation
export PYTHONPATH=/home/zs89/decodeshare-camera-ready/src:${PYTHONPATH:-}
CUDA_VISIBLE_DEVICES=0 python run_loto_reasoning.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --device cuda \
  --model_dtype fp32 \
  --mode loto \
  --loto_eval_mode heldout \
  --tasks gsm8k,commonsenseqa,strategyqa,aqua,arc_challenge,openbookqa,qasc,logiqa \
  --layer 10 \
  --n_subspace 128 \
  --n_eval 2048 \
  --calib_decode_max_new_tokens 128 \
  --per_task_max_states 20000 \
  --reasoning_tokens 128 \
  --max_new_tokens 256 \
  --template_randomization 1 \
  --shuffle_choices 1 \
  --add_answer_prefix 1 \
  --answer_prefix $'\nFinal answer:' \
  --use_forced_choice 1 \
  --fc_warmup_tokens 0 \
  --fc_prefix_mode auto \
  --fc_answer_prefix $'\nFinal answer:' \
  --do_sample 0 \
  --out_json ../../outputs/disturb_cot_reasoning/energy_balance_loto8_reasoning_fc_eval2048.json \
  --out_md ../../outputs/disturb_cot_reasoning/energy_balance_loto8_reasoning_fc_eval2048.md
```

## Energy-Control Full-Run Command

The tracked root wrapper has the exact alpha/k-match sweep:

```bash
cd /home/zs89/decodeshare-camera-ready
bash scripts/full_runs/run_alpha_kmatch_sweep.sh
```

Key settings from that wrapper:

- model: `meta-llama/Llama-2-7b-chat-hf`
- layer: `10`
- tasks: `gsm8k,commonsenseqa,strategyqa,aqua,arc_challenge,openbookqa,qasc,logiqa,boolq`
- `n_prompts=128`, `eval_n=2048`, `tau=0.001`, `m_shared=all`
- alphas: `0,0.25,0.5,0.75,1.0,1.25,1.5,2.0`

## Mock-Test Scope

`run_mock.sh` checks CLI availability and compact result summaries. It does not run model inference.
