#!/usr/bin/env bash
set -euo pipefail

# Configuration
PI_HOST="${1:-hanger.local}"
PI_USER="${2:-nlbutts}"
INSTALL_DIR="${3:-/home/$PI_USER/hangermon}"

echo "🚀 Deploying hangermon to $PI_USER@$PI_HOST:$INSTALL_DIR"

# 1. Sync the codebase to the Pi
# Excluding .git, venv, and videos
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "📦 Syncing files..."
rsync -avz --delete \
    --exclude '.git' \
    --exclude '.venv' \
    --exclude '.pytest_cache' \
    --exclude '__pycache__' \
    --exclude 'videos' \
    --exclude 'videos/' \
    "$REPO_DIR/" "$PI_USER@$PI_HOST:$INSTALL_DIR"

# 2. Restart the service on the Pi
echo "🔄 Restarting hangermon.service on the Pi..."
ssh "$PI_USER@$PI_HOST" "sudo systemctl restart hangermon.service"

