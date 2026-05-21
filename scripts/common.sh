#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "scripts/common.sh is meant to be sourced, not executed directly." >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_CMD="${PYTHON_CMD:-python}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda}"
FINAL_ANSWER_PREFIX_DEFAULT=$'\nFinal answer:'

read -r -a PYTHON_ARR <<< "${PYTHON_CMD}"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

run_logged() {
  if [[ "${DRY_RUN:-0}" == "1" || "${TRACE_COMMANDS:-0}" == "1" ]]; then
    {
      printf '+'
      printf ' %q' "$@"
      printf '\n'
    } >&2
  fi
  if [[ "${DRY_RUN:-0}" != "1" ]]; then
    "$@"
  fi
}

run_python() {
  run_logged "${PYTHON_ARR[@]}" "$@"
}

run_python_gpu() {
  run_logged env CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_ARR[@]}" "$@"
}
