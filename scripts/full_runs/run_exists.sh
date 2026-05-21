#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible H1 entry point.
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_h1_full_benchmark.sh" "$@"
