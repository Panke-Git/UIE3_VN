#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python -m src.v1.test_v1 \
  --config configs/configV1.yaml
