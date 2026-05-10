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
cd
hf upload Zishan-Shao/decodeshare Hype1/results/acts artifacts/Hype1/results/acts
hf upload Zishan-Shao/decodeshare downstream/outputs artifacts/downstream/outputs
hf upload Zishan-Shao/decodeshare patch_back/results artifacts/patch_back/results
```

For a smaller first release, upload only the most reusable artifacts:

```bash
hf upload Zishan-Shao/decodeshare Hype1/results/acts artifacts/Hype1/results/acts
hf upload Zishan-Shao/decodeshare patch_back/results artifacts/patch_back/results
```

## Notes

- The current GitHub branch excludes `.npy`, `.npz`, `.pt`, `.bin`, and related
  large binary formats.
- The largest local files are downstream `.pt` profiling and whitening outputs;
  decide whether those are necessary before uploading the full manifest.
- The manifest is intentionally broad and includes local experimental archives.
  Inspect it before uploading everything.
- If you want this to be a dataset repository rather than a model repository,
  create or switch to a Hugging Face Dataset repo and add `--repo-type dataset`
  to the `hf upload` commands.
