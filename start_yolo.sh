#!/usr/bin/env bash
set -euo pipefail

cd hangermon/yolo_det_rpi
uv run yolo_server.py -m yolo_det_rpi/yolov8n.onnx -p 5555 -c 0.5 -n 0.45
