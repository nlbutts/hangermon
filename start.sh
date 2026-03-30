#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate optional virtual environment when present so dependencies resolve.
if [[ -d "$ROOT_DIR/.venv" && -f "$ROOT_DIR/.venv/bin/activate" ]]; then
	# shellcheck disable=SC1091
	source "$ROOT_DIR/.venv/bin/activate"
fi

export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}

# Start YOLO server in background
python3 "$ROOT_DIR/yolo_det_rpi/yolo_server.py" \
    -m "$ROOT_DIR/yolo_det_rpi/yolov8n.onnx" \
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

exec python3 "$ROOT_DIR/app.py"

