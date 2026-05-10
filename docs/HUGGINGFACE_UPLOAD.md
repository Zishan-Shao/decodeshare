# Hugging Face Artifact Upload

Large DecodeShare artifacts should live outside Git history. The intended
repository is:

```text
Zishan-Shao/decodeshare
```

The file `docs/artifact_manifest.tsv` lists large local files and suggested
paths under `artifacts/` in the Hugging Face repository.

## Suggested upload pattern

Install and authenticate the Hugging Face CLI:

```bash
pip install -U huggingface_hub[hf_transfer]
hf auth login
```

For a full upload from the original workspace:

```bash
cd /path/to/decodeshare
hf upload Zishan-Shao/decodeshare Hype1/results/acts artifacts/Hype1/results/acts
hf upload Zishan-Shao/decodeshare patch_back/results artifacts/patch_back/results
```

## Downstream outputs

`downstream/outputs` contains several `.pt` files. The profiling caches can be
larger than Hugging Face Hub's 50 GB single-file limit, so upload them as
ordered 10 GiB parts rather than as raw files.

The uploaded layout keeps the original run directories. Files ending in
`.pt.part-000`, `.pt.part-001`, and so on are split chunks that should be
concatenated in lexical order to recover the original `.pt` file.

Example staging flow:

```bash
SRC_ROOT=/path/to/decodeshare
STAGE=/path/to/decodeshare_hf_downstream_split
rm -rf "$STAGE"
mkdir -p "$STAGE/artifacts/downstream/outputs"

mkdir -p "$STAGE/artifacts/downstream/outputs/llama2_r0.2_baseline"
ln "$SRC_ROOT/downstream/outputs/llama2_r0.2_baseline/meta_llama_Llama_2_7b_chat_hf_whitening_only_keep0p8_baseline.pt" \
  "$STAGE/artifacts/downstream/outputs/llama2_r0.2_baseline/"
split -b 10G -d -a 3 \
  "$SRC_ROOT/downstream/outputs/llama2_r0.2_baseline/meta-llama_Llama-2-7b-chat-hf_profiling___calib_mix_jsonl_128_0.pt" \
  "$STAGE/artifacts/downstream/outputs/llama2_r0.2_baseline/meta-llama_Llama-2-7b-chat-hf_profiling___calib_mix_jsonl_128_0.pt.part-"

mkdir -p "$STAGE/artifacts/downstream/outputs/llama2_r0.2_decodeshare_a2"
ln "$SRC_ROOT/downstream/outputs/llama2_r0.2_decodeshare_a2/meta_llama_Llama_2_7b_chat_hf_whitening_only_keep0p8_decodeshare_a2p0.pt" \
  "$STAGE/artifacts/downstream/outputs/llama2_r0.2_decodeshare_a2/"
split -b 10G -d -a 3 \
  "$SRC_ROOT/downstream/outputs/llama2_r0.2_decodeshare_a2/meta-llama_Llama-2-7b-chat-hf_profiling___calib_mix_jsonl_128_0.pt" \
  "$STAGE/artifacts/downstream/outputs/llama2_r0.2_decodeshare_a2/meta-llama_Llama-2-7b-chat-hf_profiling___calib_mix_jsonl_128_0.pt.part-"

mkdir -p "$STAGE/artifacts/downstream/outputs/svdllm_whiten_r0.2"
ln "$SRC_ROOT/downstream/outputs/svdllm_whiten_r0.2/meta_llama_Llama_2_7b_chat_hf_whitening_only_0.8.pt" \
  "$STAGE/artifacts/downstream/outputs/svdllm_whiten_r0.2/"
split -b 10G -d -a 3 \
  "$SRC_ROOT/downstream/outputs/svdllm_whiten_r0.2/meta_llama_Llama_2_7b_chat_hf_profiling_wikitext2_128_0.pt" \
  "$STAGE/artifacts/downstream/outputs/svdllm_whiten_r0.2/meta_llama_Llama_2_7b_chat_hf_profiling_wikitext2_128_0.pt.part-"

find "$STAGE" -type f -size +50G -print
hf upload-large-folder Zishan-Shao/decodeshare "$STAGE" \
  --repo-type model \
  --num-workers 4 \
  --no-bars
```

Reassemble a split file after download:

```bash
cat artifacts/downstream/outputs/llama2_r0.2_baseline/meta-llama_Llama-2-7b-chat-hf_profiling___calib_mix_jsonl_128_0.pt.part-* \
  > artifacts/downstream/outputs/llama2_r0.2_baseline/meta-llama_Llama-2-7b-chat-hf_profiling___calib_mix_jsonl_128_0.pt
```

## Notes

- The current GitHub branch excludes `.npy`, `.npz`, `.pt`, `.bin`, and related
  large binary formats.
- The largest local files are downstream `.pt` profiling and whitening outputs;
  the profiling outputs need the split workflow above.
- The manifest is intentionally broad and includes local experimental archives.
  Inspect it before uploading everything.
- If you want this to be a dataset repository rather than a model repository,
  create or switch to a Hugging Face Dataset repo and add `--repo-type dataset`
  to the `hf upload` commands.
