# DecodeShare

Code and compact summaries for decode-time shared-subspace experiments.

## Layout

- `Hype1/`: early sharedness experiments and small result summaries
- `patch_back/`: patch-back and transfer experiments
- `reasoning/` and `src/`: core evaluation and intervention code
- `rebuttal/`: protocol-focused rebuttal scripts, notes, and compact summaries

## Artifact policy

This GitHub repo keeps code, manifests, and small human-readable summaries.
Large binaries and bulky result bundles are published separately on Hugging Face:

- `https://huggingface.co/Zishan-Shao/decodeshare`

In practice that means directories such as `results/rebuttal_mechanism/` and
`results/rebuttal_scaling/` stay off GitHub, while their scripts and notes stay
in this repo.
