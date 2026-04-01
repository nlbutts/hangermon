#!/usr/bin/env bash
set -euo pipefail

cd yolo_det_rpi
python3 yolo_server.py -m yolov8n_integer_quant.tflite -p 5555 -c 0.5 -n 0.45
