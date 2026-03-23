#!/usr/bin/env bash
set -euo pipefail

# Configuration
PI_HOST="${1:-hanger.local}"
PI_USER="${2:-nlbutts}"
INSTALL_DIR="${3:-/home/$PI_USER/hangermon}"

echo "🔄 Starting hangermon in foreground on the Pi (System Python)..."
ssh "$PI_USER@$PI_HOST" "cd \"$INSTALL_DIR\" && python3 app.py"
