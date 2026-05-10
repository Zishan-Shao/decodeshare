#!/usr/bin/env bash
set -euo pipefail

REPO_ID="${REPO_ID:-Zishan-Shao/decodeshare}"
SRC_ROOT="${SRC_ROOT:-$PWD}"
HF_NUM_WORKERS="${HF_NUM_WORKERS:-4}"

if ! command -v hf >/dev/null 2>&1; then
  echo "Missing Hugging Face CLI. Install with: pip install -U huggingface_hub[hf_transfer]" >&2
  exit 1
fi

cd "$SRC_ROOT"

hf upload "$REPO_ID" Hype1/results/acts artifacts/Hype1/results/acts
hf upload "$REPO_ID" patch_back/results artifacts/patch_back/results

if [[ -n "${DOWNSTREAM_SPLIT_ROOT:-}" ]]; then
  hf upload-large-folder "$REPO_ID" "$DOWNSTREAM_SPLIT_ROOT" \
    --repo-type model \
    --num-workers "$HF_NUM_WORKERS" \
    --no-bars
else
  echo "Skipping downstream/outputs. Create a split staging directory and set DOWNSTREAM_SPLIT_ROOT; see docs/HUGGINGFACE_UPLOAD.md." >&2
fi

echo "Uploaded selected DecodeShare artifacts to $REPO_ID"
