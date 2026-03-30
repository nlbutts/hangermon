#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}

# Ensure uv is in PATH
export PATH="$HOME/.local/bin:$PATH"

# Start YOLO server in background using uv
cd "$ROOT_DIR"
uv run yolo_det_rpi/yolo_server.py \
    -m yolo_det_rpi/yolov8n.onnx \
    -p 5555 \
    -c 0.5 \
    -n 0.45 \
    &
YOLO_PID=$!

# Wait for server to initialize
sleep 3

# Cleanup function
cleanup() {
    echo "Shutting down YOLO server (PID: $YOLO_PID)..."
    kill $YOLO_PID 2>/dev/null || true
    wait $YOLO_PID 2>/dev/null || true
}
trap cleanup EXIT

exec uv run python3 app.py

