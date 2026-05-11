# 01 H1 Shared Decode-Time Structure

Paper outputs:

- Main: Figures 2-4; Table 6.
- Appendix: Figures 8, 11-14; Tables 7-13.

## Canonical Scripts

- `experiments/01_sharedness/run_full_benchmark.py`
- `experiments/01_sharedness/sharedness_base.py`
- `experiments/01_sharedness/collect_activations.py`
- `experiments/01_sharedness/analyze_within_vs_mixed.py`
- `experiments/01_sharedness/analyze_task_count_convergence.py`
- `experiments/01_sharedness/analyze_phase_convergence.py`
- `experiments/01_sharedness/analyze_tau_sensitivity.py`
- `experiments/01_sharedness/summarize_full_benchmark.py`

Historical wrappers and exploratory variants are under `experiments/01_sharedness/legacy/`.

## Checked-In Results

- `paper_artifacts/h1_results/results/full_benchmark/H1_full_benchmark_summary.*`
- `paper_artifacts/h1_results/results/full_benchmark/H1_evidence_chain.tex`
- `paper_artifacts/h1_results/results/full_benchmark/*_exist*.json`
- `paper_artifacts/h1_results/results/full_benchmark/*_exist*.txt`
- `paper_artifacts/h1_results/results/exp1/*`
- `paper_artifacts/h1_results/results/exp2/*`
- `paper_artifacts/h1_results/results/exp2.75/*`
- `paper_artifacts/h1_results/results/exp3/*`

## Summary Regeneration

```bash
cd /home/zs89/decodeshare-camera-ready
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
python experiments/01_sharedness/summarize_full_benchmark.py \
  --results_dir paper_artifacts/h1_results/results/full_benchmark \
  --out_dir paper_artifacts/h1_results/results/full_benchmark \
  --alpha 0.05
```

## Full-Run Command Template

Use only `Node0` or `Node1` for camera-ready reruns.

Example for the Qwen layer-10 full benchmark row:

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

## Diagnostics

Activation-based diagnostics use this sequence:

1. `collect_activations.py`
2. `analyze_within_vs_mixed.py`
3. `analyze_task_count_convergence.py`
4. `analyze_phase_convergence.py`
5. `analyze_tau_sensitivity.py`

The paper parameter records are in:

- `experiments/01_sharedness/configs/full_benchmark.yaml`
- `experiments/01_sharedness/configs/diagnostics.yaml`

## Mock-Test Scope

`run_mock.sh` checks script CLI/import validity, verifies checked-in H1 summaries, and regenerates a temporary summary from checked-in full-benchmark records. It does not load a model.
