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
- downstream compression outputs
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
cd /home/zs89/decodeshare
hf upload Zishan-Shao/decodeshare Hype1/results/acts artifacts/Hype1/results/acts
hf upload Zishan-Shao/decodeshare patch_back/results artifacts/patch_back/results
hf upload Zishan-Shao/decodeshare downstream/outputs artifacts/downstream/outputs
```

Model and dataset licenses remain governed by their upstream providers.
