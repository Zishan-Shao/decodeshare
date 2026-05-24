# Paper Artifact Manifest

Small camera-ready artifacts live in this directory.

## Included

- `tables/`: final or regenerated paper tables.
- `figures/`: final or regenerated paper figures.
- `summaries/`: compact human-readable summaries used to audit paper results.
- `figures/decodeshare_steering_demo.gif`: README-embedded GPU demo preview.
- `figures/decodeshare_steering_demo.mp4`: compact source clip corresponding
  to the README demo preview.

## External

Large raw experiment outputs should not be committed by default. Record them in
the repository-level `MANIFEST.md` with:

- original path or artifact-store URI;
- checksum;
- generation command;
- model, dataset/task, layer, rank, seed;
- file size.
