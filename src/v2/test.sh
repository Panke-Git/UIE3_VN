#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash src/v2/test.sh <run-directory> [config-path]" >&2
  exit 2
fi

cd "$(dirname "$0")/../.."

python -m src.v2.test_v2 \
  --config "${2:-configs/configV2.yaml}" \
  --run-dir "$1"
