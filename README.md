# DecodeShare

Code and lightweight release artifacts for **DecodeShare: Tracing the Shared
Pathways of LLM Decode-Time Decisions**.

DecodeShare studies low-dimensional subspaces in KV-cached decode-time hidden
states. The repository focuses on three checks:

- `H1`: shared decode-time structure exists beyond chance.
- `H2`: removing the shared decode subspace during decode causally affects
  decisions more than matched controls.
- `H3`: prefill-estimated and decode-estimated subspaces can differ
  substantially, so decode-aligned evaluation matters.

This branch is a curated ICML release branch. It intentionally keeps GitHub
small: source code, scripts, paper-level tables, summaries, and compact JSON/CSV
outputs are tracked here; raw activations, checkpoints, basis dumps, and large
downstream artifacts are documented in `docs/artifact_manifest.tsv` for upload
to Hugging Face.

## Repository layout

- `src/`: core shared-subspace, decode-time disturbance, prefill/decode
  mismatch, and energy-matched experiments.
- `Hype1/`: H1 sharedness experiments and compact H1 result summaries.
- `reasoning/`: H2/H3 reasoning and forced-choice disturbance experiments.
- `patch_back/`: subspace patchback and transfer-patching experiments.
- `brittleness/`: steering robustness and repair-style experiments.
- `downstream/`: downstream compression / whitening utility experiments.
- `joint_subspace_large/`: larger joint-subspace probes.
- `lateruse/`: exploratory follow-up analyses kept separate from the main
  protocol path.
- `plot/`: plotting scripts and small final figure assets.
- `docs/`: release notes, reproducibility map, data/model notes, and Hugging
  Face artifact upload instructions.

## Environment

The original experiments were run with CUDA-capable GPUs and Hugging Face model
checkpoints. Create the environment with:

```bash
conda env create -f environment.yml
conda activate decodeshare
```

Some experiments require access to gated model repositories, for example
Llama-family checkpoints. Authenticate with Hugging Face before running those
scripts.

## Quick commands

H1 shared decode workspace:

```bash
bash Hype1/run_00_collect_acts.sh
bash Hype1/run_01_exp1_within_vs_mixed.sh
bash Hype1/run_02_exp2_convergence.sh
```

Decode-time disturbance / H2:

```bash
bash run_disturb_cot_loto8_main.sh
bash run_disturb_cot_loto8_main_qwen.sh
bash run_disturb_cot_loto8_main_falcon.sh
```

Prefill-vs-decode mismatch / H3:

```bash
bash reasoning/run_h3_grid.sh
```

Patchback:

```bash
bash patch_back/run_decodeshare_suite.sh
```

Downstream compression comparison:

```bash
bash downstream/run_compare.sh
```

## Results

Start with these compact summaries:

- `Hype1/results/full_benchmark/H1_full_benchmark_summary.md`
- `results/disturb_cot_reasoning/energy_balance_loto8_reasoning_fc.md`
- `results/h3_grid/h3_grid_reasoning.md`
- `patch_back/paper/patchback_tables_all_models_all_layers.tex`
- `rebuttal/important_results_summary.md`

See `docs/REPRODUCIBILITY.md` for a map from claims to scripts and outputs.

## Large artifacts

Large files are not committed to this branch. The manifest at
`docs/artifact_manifest.tsv` lists raw activations, bases, checkpoints, and
downstream outputs that can be uploaded to the Hugging Face repository
`Zishan-Shao/decodeshare`.

See `docs/HUGGINGFACE_UPLOAD.md` for the upload workflow.

## License

MIT. See `LICENSE`.
