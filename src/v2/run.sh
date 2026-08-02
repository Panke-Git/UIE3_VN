#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: bash src/v2/run.sh <variant> <seed>" >&2
  exit 2
fi

cd "$(dirname "$0")/../.."

python -m src.v2.train_v2 \
  --config configs/configV2.yaml \
  --variant "$1" \
  --seed "$2"
