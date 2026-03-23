#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}

# Use the system Python. Dependencies are installed via install_systemd.sh using apt.
exec python3 "$ROOT_DIR/app.py"
