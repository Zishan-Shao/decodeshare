---
license: mit
tags:
  - interpretability
  - mechanistic-interpretability
  - activation-steering
  - llm
  - kv-cache
  - icml-2026
---

# DecodeShare

Large artifacts for **DecodeShare: Tracing the Shared Pathways of LLM
Decode-Time Decisions**.

The corresponding GitHub release branch is:

```text
https://github.com/Zishan-Shao/decodeshare/tree/Halo
```

This Hugging Face repository is intended for files that should not live in Git
history:

- decode-time activation caches
- shared-subspace bases
- patchback result archives
- downstream compression outputs, with oversized profiling caches stored as
  `.pt.part-*` chunks
- selected steering vectors and cached candidate pools

The GitHub branch tracks compact code, scripts, summaries, and the full artifact
manifest at:

```text
docs/artifact_manifest.tsv
```

Suggested layout:

```text
artifacts/
  Hype1/results/acts/
  patch_back/results/
  downstream/outputs/
```

Install the Hugging Face CLI and upload from the original workspace:

```bash
pip install -U huggingface_hub[hf_transfer]
hf auth login
cd /path/to/decodeshare
hf upload Zishan-Shao/decodeshare Hype1/results/acts artifacts/Hype1/results/acts
hf upload Zishan-Shao/decodeshare patch_back/results artifacts/patch_back/results
```

For downstream profiling caches, use the split-file workflow in
`docs/HUGGINGFACE_UPLOAD.md`. Reassembly notes are included under
`artifacts/downstream/outputs/SPLIT_FILES.md` after upload.

Model and dataset licenses remain governed by their upstream providers.
