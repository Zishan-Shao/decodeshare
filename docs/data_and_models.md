# Data and Models

Model weights and datasets are not committed to this repository.

Expected local setup:

- Hugging Face credentials are configured outside the repo when gated models are
  used.
- Model caches live in the user's normal Hugging Face cache or a cluster-local
  cache path.
- Large raw experiment outputs live outside git and are recorded in
  `MANIFEST.md`.

Cluster rerun constraint for the current camera-ready pass:

- Available nodes: `Node0`, `Node1`
- Do not schedule full reruns on other nodes unless this status changes.

For public release, each full-run command should state the exact model name,
dataset/task, layer, subspace rank, seed, and output directory.
