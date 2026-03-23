# Hanger Monitor (hangermon)

A Raspberry Pi-based person detection service that records video clips when movement is detected. Supports both the Sony IMX500 AI Camera (on-device NPU) and the Pi Camera Module v3 (via YOLOv8n TFLite on CPU).

## Key Features

- **Dual Sensor Support**: Select between `imx500` and `picamera3` via configuration.
- **YOLOv8 Detection**: High-performance CPU inference for the Pi Camera v3.
- **Automatic Recording**: Buffers 1-2 seconds of pre-motion video and saves clips upon human detection.
- **Web Dashboard**: View real-time status and browse recorded clips at `http://<pi-hostname>:8000`.
- **Systemd Integration**: Runs as a background service with automatic restarts.

## Getting Started

1.  **Clone the repository** (locally or on the Pi):
    ```bash
    git clone https://github.com/nlbutts/hangermon.git
    cd hangermon
    ```

2.  **Run the installation script** on the Raspberry Pi:
    ```bash
    sudo ./scripts/install_systemd.sh
    ```
    This script installs all necessary `apt` dependencies (including `python3-picamera2`, `python3-opencv`, `ffmpeg`), installs TensorFlow for YOLOv8 support, and sets up the systemd service.

3.  **Configure**: Edit `config.yaml` to specify your sensor and recording preferences.
    ```yaml
    sensor: picamera3
    camera:
       width: 1280
       height: 720
    ```

4.  **Deploy from your local machine** (optional):
    ```bash
    ./scripts/deploy_to_pi.sh
    ```
    This syncs local changes to the Pi and restarts the service.

## Configuration Reference (config.yaml)

| Key | Default | Description |
| --- | --- | --- |
| `sensor` | `picamera3` | `picamera3` (YOLOv8) or `imx500` (NPU). |
| `camera.width` | 1280 | Capture resolution width. |
| `camera.height`| 720 | Capture resolution height. |
| `yolov8.model_path`| `models/yolov8n_int8.tflite` | Path to the TFLite model. |
| `yolov8.min_confidence`| 0.45 | Detection confidence threshold. |
| `recording.video_dir`| `videos/` | Where to save `.mp4` clips. |

## Development

Use `scripts/run_on_pi.sh` to run the application in the foreground for debugging:
```bash
./scripts/run_on_pi.sh
```

## Logs & Status

Check service health:
```bash
sudo systemctl status hangermon.service
journalctl -u hangermon.service -f
```
