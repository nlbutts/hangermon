#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}

# Start YOLO server in background using uv
cd "$ROOT_DIR"
python3 app.py

