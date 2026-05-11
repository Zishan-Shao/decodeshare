#!/usr/bin/env bash
set -euo pipefail

test -s "paper_artifacts/DecodeShare_camera_ready.pdf"
python - <<'PY'
from pathlib import Path
p = Path("paper_artifacts/DecodeShare_camera_ready.pdf")
print(f"paper_pdf_ok size_bytes={p.stat().st_size}")
PY
