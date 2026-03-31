#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT="${1:-hanger}"

echo "Syncing $ROOT_DIR to nlbutts@${UNIT}:hangermon..."
rsync -avz --delete \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '.ruff_cache' \
    --exclude '.pytest_cache' \
    --exclude '.venv' \
    --exclude 'videos' \
    "$ROOT_DIR/" "nlbutts@${UNIT}:hangermon/"

echo "Sync to ${UNIT} complete!"
