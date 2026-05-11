# Paper Artifact Manifest

Small camera-ready artifacts live in this directory.

## Included

- `DecodeShare_camera_ready.pdf`: copied paper PDF for the camera-ready staging
  branch.
- `tables/`: final or regenerated paper tables.
- `figures/`: final or regenerated paper figures.
- `summaries/`: compact human-readable summaries used to audit paper results.

## External

Large raw experiment outputs should not be committed by default. Record them in
the repository-level `MANIFEST.md` with:

- original path or artifact-store URI;
- checksum;
- generation command;
- model, dataset/task, layer, rank, seed;
- file size.
