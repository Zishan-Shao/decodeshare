# 00 Paper

## Local Files

- Branch copy: `paper_artifacts/DecodeShare_camera_ready.pdf`
- Original source copy: `/home/zs89/decodeshare/17010_DecodeShare_Tracing_the_ (2).pdf`

## Camera-Ready Checks

The current repository root does not contain the LaTeX paper source. Once the source is added or linked, check:

- table and figure labels match the folders below;
- the final PDF does not include stray review-instruction text;
- the PDF title and anonymity/camera-ready metadata are updated.

## Mock Command

```bash
python - <<'PY'
from pathlib import Path
p = Path("paper_artifacts/DecodeShare_camera_ready.pdf")
print(p.exists(), p.stat().st_size if p.exists() else 0)
PY
```
