#!/usr/bin/env bash
set -euo pipefail

REPO_ID="${REPO_ID:-Zishan-Shao/decodeshare}"
SRC_ROOT="${SRC_ROOT:-/home/zs89/decodeshare}"

if ! command -v hf >/dev/null 2>&1; then
  echo "Missing Hugging Face CLI. Install with: pip install -U huggingface_hub[hf_transfer]" >&2
  exit 1
fi

cd "$SRC_ROOT"

hf upload "$REPO_ID" Hype1/results/acts artifacts/Hype1/results/acts
hf upload "$REPO_ID" patch_back/results artifacts/patch_back/results
hf upload "$REPO_ID" downstream/outputs artifacts/downstream/outputs

echo "Uploaded core DecodeShare artifacts to $REPO_ID"
