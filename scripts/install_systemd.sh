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

if [[ "$(realpath "$INSTALL_DIR")" != "$(realpath "$REPO_DIR")" ]]; then
  echo "Syncing repository to $INSTALL_DIR"
  rsync -a --delete \
    --exclude '.git' \
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
# Uncomment and adjust as needed.
# IMX500_MODEL_PATH=/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk
# VIDEO_DIR=/opt/hangermon/videos
ENV
fi

systemctl daemon-reload
systemctl enable --now hangermon.service

systemctl --no-pager status hangermon.service

echo "hangermon.service installed. Update $ENV_FILE for additional overrides and restart with:\n  sudo systemctl restart hangermon.service"
