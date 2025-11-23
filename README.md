# Hanger Monitor

Real-time Raspberry Pi monitoring stack that uses an AI camera to detect humans, record short video clips, and expose everything through a Flask web dashboard.

## Features

- Live MJPEG stream with IMX500 overlays and confidence readouts
- Automatic clip recording (pre/post buffer + retention pruning)
- Flask REST API for status + recorded video catalog
- Responsive web UI with live telemetry and clip playback links
- Fully configurable via environment variables for camera, detection, recording, and web settings

## System Architecture

| Layer | Responsibilities | Key Modules |
| --- | --- | --- |
| Camera Pipeline | Capture frames from the Raspberry Pi AI camera (Picamera2 or OpenCV fallback), regulate FPS, push frames to downstream consumers. | `hangermon/camera/streamer.py` |
| Detection Engine | Read IMX500 inference metadata emitted by the Raspberry Pi AI Camera, filter for people, and overlay bounding boxes directly on captured frames. | `hangermon/detection/detector.py` |
| Recording Manager | Start/stop MP4 writers when humans appear/disappear, embed metadata, and maintain a rolling catalog of clips on disk. | `hangermon/recording/writer.py`, `hangermon/storage/catalog.py` |
| Backend Service | Coordinate capture/detect/record loops, keep latest status, and expose REST + MJPEG streaming endpoints via Flask. | `hangermon/service.py`, `app.py` |
| Web UI | Responsive dashboard (Vanilla JS + HTMX) showing live stream, detection status, and playable clip list. | `templates/index.html`, `static/` |

### Data Flow

1. `CameraStreamer` opens the camera device (Picamera2 preferred, OpenCV fallback) and publishes frames to a bounded queue.
2. `Imx500Detector` inspects the IMX500 metadata attached to each frame, converts detections to bounding boxes, and returns an annotated image.
3. `MonitorService` updates a status snapshot (human present flag, confidence, FPS) and hands frames to `ClipRecorder`.
4. `ClipRecorder` spools annotated frames to disk while a detection is active and closes the clip once detections stop for a configurable cooldown period.
5. `ClipCatalog` indexes the `videos/YYYY/MM/DD` hierarchy, exposing metadata via `/api/clips` so the web UI can present and play recordings.

## Raspberry Pi Considerations

- Install the Raspberry Pi AI Camera firmware + IMX500 plugin (`sudo apt install -y python3-picamera2 pi-imx500-tools`).
- Use `imx500-runner` (ships with the tools package) to deploy a person-detection task blob to the sensor before launching this app.
- When Picamera2 is active the IMX500 metadata is attached to each frame; ensure `USE_PICAMERA2=true` for the AI camera.
- Keep capture FPS/resolution modest (≤1280x720 @ 15fps) so the sensor and ISP do not thermal throttle.

## Getting Started

> These steps were validated on Raspberry Pi OS Bookworm (64-bit). Adjust package names for other distros.

### 1. Install system dependencies

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-dev libatlas-base-dev libopenjp2-7 libtiff6 libilmbase-dev libopenexr-dev libgstreamer1.0-dev
# Optional but recommended for Pi Camera users
sudo apt install -y python3-picamera2 pi-imx500-tools ffmpeg
```

### 2. Create a Python environment and install packages

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure environment variables (optional)

You can override any defaults from `hangermon/config.py`. Example `.env`:

```
CAMERA_WIDTH=1280
CAMERA_HEIGHT=720
CAMERA_FPS=15
IMX500_METADATA_PATH=imx500.results
IMX500_TARGET_LABELS=person
IMX500_MIN_CONFIDENCE=0.4
VIDEO_RETENTION_DAYS=7
SERVER_PORT=8000
```

Before launching the app, use `imx500-runner` (part of `pi-imx500-tools`) to load and start a person-detection task on the sensor. The metadata keys in the settings above must align with the task's output schema.

### 4. Run the web server

```bash
export FLASK_APP=app.py
flask run --host=0.0.0.0 --port=8000
```

or use `python app.py` to rely on the built-in runner.

Point your browser to `http://<raspberrypi>:8000` to open the dashboard.

### 5. Run tests

```bash
pytest
```

## Configuration Reference

| Variable | Default | Description |
| --- | --- | --- |
| `CAMERA_DEVICE` | `0` | `/dev/video` index for OpenCV capture |
| `CAMERA_WIDTH`/`HEIGHT` | `1280x720` | Capture resolution |
| `CAMERA_FPS` | `15` | Target capture FPS |
| `USE_PICAMERA2` | `false` | Switch to Picamera2 pipeline |
| `IMX500_METADATA_PATH` | `imx500.results` | Dot-path inside Picamera2 metadata that contains detections |
| `IMX500_LABEL_FIELD` | `label` | Key within each detection entry that stores the class label |
| `IMX500_SCORE_FIELD` | `score` | Key for the confidence score |
| `IMX500_BBOX_FIELD` | `bbox` | Key for the bounding box array |
| `IMX500_BBOX_FORMAT` | `xywh` | Format of the bounding box (`xywh` or `xyxy`) |
| `IMX500_BBOX_NORMALIZED` | `true` | Treat box coordinates as normalized 0‑1 values |
| `IMX500_TARGET_LABELS` | `person` | Comma-separated labels that count as “human present” |
| `IMX500_MIN_CONFIDENCE` | `0.45` | Minimum confidence required to treat a detection as valid |
| `IMX500_DRAW_OVERLAY` | `true` | Toggle bounding-box rendering on the live stream |
| `IMX500_LATENCY_FIELD` | _unset_ | Optional dot-path to a latency metric for telemetry |
| `VIDEO_DIR` | `videos` | Root directory for MP4 + metadata |
| `VIDEO_PREBUFFER` | `1.0` | Seconds to include before detection |
| `VIDEO_GRACE_PERIOD` | `2.5` | Seconds to keep recording after subject leaves |
| `VIDEO_RETENTION_DAYS` | `14` | Automatic deletion window |
| `SERVER_HOST` | `0.0.0.0` | Flask bind address |
| `SERVER_PORT` | `8000` | Flask port |

## Directory Layout

```
hangermon/
├── app.py                # Flask entrypoint
├── hangermon/
│   ├── camera/streamer.py    # Camera capture helpers
│   ├── detection/detector.py # IMX500 metadata parsing + overlay
│   ├── recording/writer.py   # Detection-driven clip recorder
│   ├── service.py            # MonitorService orchestration loop
│   └── storage/catalog.py    # Clip metadata + retention
├── static/ + templates/  # Web UI assets
├── videos/               # Recorded clips + metadata (created at runtime)
├── requirements.txt
└── README.md
```

## Deployment Notes

- Use `systemd` to manage the Flask process (`ExecStart=/home/pi/hangermon/.venv/bin/gunicorn -w 1 -b 0.0.0.0:8000 app:app`).
- Pre-load your preferred IMX500 task (e.g., person detection) using `imx500-runner --task person-detect.json` before launching the service.
- Serve the app behind nginx if you need TLS for remote access.

## Troubleshooting

- **Camera feed black**: verify `libcamera-hello` works; set `USE_PICAMERA2=true` when using Picamera2.
- **No detections**: confirm `imx500-runner status` shows an active task and that the metadata keys in `IMX500_*` settings match what the firmware emits.
- **Clips missing**: ensure `videos/` is writable and check logs for codec issues (`VIDEO_CODEC=mp4v` works broadly).
