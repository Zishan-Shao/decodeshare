# 05 Steering Repair and Template Robustness

Paper outputs:

- Main: Table 2.
- Appendix: Figure 15; Tables 21-25, 29.

## Canonical Scripts

- `experiments/05_steering_repair/steering_vector_reliability_multibench_patch_v3.py`
- `experiments/05_steering_repair/summarize_multibench_v3_full.py`
- `experiments/05_steering_repair/mvp_projection_patch_pirate_v5.py`

## Local Results Found

Compact summaries:

- `/home/zs89/decodeshare/brittleness/results/steer_repair_multibench_v3/summary_pack/summary_multibench_v3_full.md`
- `/home/zs89/decodeshare/brittleness/results/steer_repair_multibench_v3/summary_pack/tables_multibench_v3_full.tex`
- `/home/zs89/decodeshare/brittleness/results/sharedspace_solid_llama2_7b_chat/summary.md`
- `/home/zs89/decodeshare/brittleness/results/sharedspace_solid_llama2_7b_chat/summary.tex`
- `/home/zs89/decodeshare/brittleness/results/mvp_pirate_v5_story_clean/summary.md`

## Full-Run Command

Multibench repair:

```bash
cd /home/zs89/decodeshare-camera-ready/experiments/05_steering_repair
CUDA_VISIBLE_DEVICES=0 python steering_vector_reliability_multibench_patch_v3.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --device cuda \
  --dtype fp32 \
  --layer 10 \
  --tasks boolq,rte,sst2 \
  --calib_per_class 256 \
  --eval_per_class 128 \
  --basis_source neutral \
  --basis_k 512 \
  --basis_max_states 1024 \
  --betas 0,0.25,0.5,0.75,1.0 \
  --lambdas 0,0.5,1.0 \
  --n_rand 5 \
  --cand_calib_per_class 32 \
  --cand_calib_templates all \
  --out_dir results/steer_repair_multibench_v3 \
  --show_per_template 1
```

Summary regeneration:

```bash
cd /home/zs89/decodeshare-camera-ready/experiments/05_steering_repair
python summarize_multibench_v3_full.py \
  --root_dir /home/zs89/decodeshare/brittleness/results/steer_repair_multibench_v3 \
  --out_dir /home/zs89/decodeshare/brittleness/results/steer_repair_multibench_v3/summary_pack
```

Pirate sanity check:

```bash
cd /home/zs89/decodeshare-camera-ready
python experiments/05_steering_repair/mvp_projection_patch_pirate_v5.py --help
```

## Mock-Test Scope

`run_mock.sh` checks CLI availability and compact result-summary presence. It does not run steering evaluation.
