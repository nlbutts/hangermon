#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: sudo ./scripts/install_systemd.sh [--path /opt/hangermon] [--user hangermon]

Options:
  -p, --path PATH     Absolute path to run the service from. Defaults to the current
                      repository path. When the path differs from the repo, the
                      contents (excluding videos/) are synced there.
  -u, --user USER     System user to run the service as. Defaults to $SUDO_USER if set,
                      otherwise "hangermon". A system user is created when missing.
  -h, --help          Show this message and exit.

The script installs the systemd unit, ensures the runtime directory exists,
reloads systemd, and enables/starts the service.
USAGE
}

if [[ ${EUID} -ne 0 ]]; then
  echo "Please run this script with sudo or as root" >&2
  exit 1
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL_DIR=""
SERVICE_USER=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--path)
      INSTALL_DIR="$2"
      shift 2
      ;;
    -u|--user)
      SERVICE_USER="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$INSTALL_DIR" ]]; then
  INSTALL_DIR="$REPO_DIR"
fi

if [[ ! "$INSTALL_DIR" = /* ]]; then
  echo "Please supply an absolute path for --path" >&2
  exit 1
fi

SERVICE_USER="${SERVICE_USER:-${SUDO_USER:-hangermon}}"
SERVICE_TEMPLATE="$REPO_DIR/systemd/hangermon.service"
SERVICE_TARGET="/etc/systemd/system/hangermon.service"
ENV_FILE="/etc/default/hangermon"

if [[ ! -f "$SERVICE_TEMPLATE" ]]; then
  echo "Missing systemd template at $SERVICE_TEMPLATE" >&2
  exit 1
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "rsync is required for deployment. Install it and re-run this script." >&2
  exit 1
fi

mkdir -p "$INSTALL_DIR"

# 2. Install APT dependencies
echo "Installing APT dependencies..."
apt-get update
# Install core system packages for Python 3.13 (Trixie)
apt-get install -y \
  python3 \
  python3-dev \
  python3-pip \
  python3-flask \
  python3-opencv \
  python3-numpy \
  python3-yaml \
  python3-dotenv \
  python3-picamera2 \
  ffmpeg \
  rsync \
  curl \
  libatlas-base-dev \
  libopenjp2-7 \
  libtiff6 \
  libgstreamer1.0-dev

# 3. Install TensorFlow for YOLOv8 (Python 3.13 support)
# We use --break-system-packages because Pi OS manages its own python environment.
echo "Installing TensorFlow (this may take a while)..."
pip3 install --break-system-packages "tensorflow>=2.21.0" || \
  echo "Warning: TensorFlow installation failed. YOLOv8 detection will be unavailable."

if [[ "$(realpath "$INSTALL_DIR")" != "$(realpath "$REPO_DIR")" ]]; then
  echo "Syncing repository to $INSTALL_DIR"
  rsync -a --delete \
    --exclude '.git' \
    --exclude '.venv' \
    --exclude '__pycache__' \
    --exclude '.pytest_cache' \
    --exclude 'videos' \
    --exclude 'videos/' \
    "$REPO_DIR"/ "$INSTALL_DIR"/
fi

if ! id -u "$SERVICE_USER" >/dev/null 2>&1; then
  echo "Creating system user $SERVICE_USER"
  useradd --system --home "$INSTALL_DIR" --shell /usr/sbin/nologin "$SERVICE_USER"
fi

mkdir -p "$INSTALL_DIR/videos"
chown -R "$SERVICE_USER":"$SERVICE_USER" "$INSTALL_DIR/videos"
chmod +x "$INSTALL_DIR/start.sh"

sed -e "s|__INSTALL_DIR__|$INSTALL_DIR|g" \
    -e "s|__HANGERMON_USER__|$SERVICE_USER|g" \
    "$SERVICE_TEMPLATE" > "$SERVICE_TARGET"
chmod 644 "$SERVICE_TARGET"

if [[ ! -f "$ENV_FILE" ]]; then
  cat <<'ENV' > "$ENV_FILE"
# Example environment overrides for hangermon.service
# VIDEO_DIR=/opt/hangermon/videos
ENV
fi

systemctl daemon-reload
systemctl enable --now hangermon.service

systemctl --no-pager status hangermon.service

echo "hangermon.service installed. Update $ENV_FILE for overrides and restart with: sudo systemctl restart hangermon.service"
