# Halo Branch Release Notes

This branch is a curated public release for the ICML version of DecodeShare.

Included:

- Core Python implementations and run scripts for H1, H2, H3, patchback,
  steering robustness, and downstream compression experiments.
- Compact tables, CSV/JSON summaries, markdown summaries, and small figures.
- A manifest of large artifacts for Hugging Face upload.

Excluded:

- Raw decode activations.
- Cached bases and large `.npy` / `.npz` arrays.
- Model checkpoints and downstream `.pt` outputs.
- Temporary logs and Python caches.
- The local anonymous/preliminary PDF copy.

The local PDF was not included because it is not a clean camera-ready public
artifact. Add the final public PDF later if desired.
