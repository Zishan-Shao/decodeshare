# 03 H2 Patchback and Transfer

Paper outputs:

- Main: Table 1; Figures 5-6.
- Appendix: Tables 14-15, 20; Figures 16-17.

## Canonical Scripts

- `experiments/03_patchback/subspace_patching_transfer.py`
- `experiments/03_patchback/openanswer_subspace_patching.py`
- `experiments/03_patchback/flipset_alpha_sweep_and_transfer.py`
- `experiments/03_patchback/summarize_patching_jsons.py`
- `downstream/patch_back/run_decodeshare_suite.sh`
- `downstream/patch_back/run_qwen_suite_and_report.sh`

Some richer layer/model wrappers exist locally in the original workspace but are not currently tracked in this branch:

- `/home/zs89/decodeshare/patch_back/run_llama_suite_and_report.sh`
- `/home/zs89/decodeshare/patch_back/run_falcon_suite_and_report.sh`
- `/home/zs89/decodeshare/patch_back/run_llama_suite_and_report_l24.sh`
- `/home/zs89/decodeshare/patch_back/run_falcon_suite_and_report_l24.sh`

## Local Results Found

Compact paper-ready artifacts:

- `/home/zs89/decodeshare/patch_back/paper/patchback_tables_all_models_all_layers.tex`
- `/home/zs89/decodeshare/patch_back/paper/patchback_discussion_all_models_all_layers.tex`
- `/home/zs89/decodeshare/patch_back/results/**/_summary/*.md`
- `/home/zs89/decodeshare/patch_back/results/**/_summary/*.tex`

## Full-Run Command Template

Multiple-choice patchback, computing the shared basis:

```bash
cd /home/zs89/decodeshare-camera-ready/experiments/03_patchback
CUDA_VISIBLE_DEVICES=0 python subspace_patching_transfer.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --device cuda \
  --dtype fp32 \
  --layer 10 \
  --seed 123 \
  --compute_Qs 1 \
  --Qs_out results/subspace_patching_transfer/runs_layer10_seed123/Q_shared_layer10.npy \
  --basis_tasks gsm8k,commonsenseqa,strategyqa,openbookqa,qasc,boolq,piqa \
  --basis_n_subspace 128 \
  --task aqua \
  --candidate_labels ABCDE \
  --n_eval 254 \
  --max_flips 128 \
  --out_json results/subspace_patching_transfer/runs_layer10_seed123/aqua_computeQ.json
```

Open-answer patchback example:

```bash
cd /home/zs89/decodeshare-camera-ready/experiments/03_patchback
CUDA_VISIBLE_DEVICES=0 python openanswer_subspace_patching.py \
  --base_script_path subspace_patching_transfer.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --device cuda \
  --dtype fp32 \
  --layer 10 \
  --seed 123 \
  --task gsm8k \
  --n_eval 256 \
  --max_flips 64 \
  --eval_mode pair_logprob \
  --Qs_path Q_shared_layer10.npy \
  --patch_n_steps 4 \
  --out_json results/openanswer/gsm8k_pairlogprob.json
```

Summary aggregation:

```bash
cd /home/zs89/decodeshare-camera-ready/experiments/03_patchback
python summarize_patching_jsons.py --help
```

## Mock-Test Scope

`run_mock.sh` checks CLI availability and that the paper table artifact exists locally. It does not run patchback.
