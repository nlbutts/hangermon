#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate optional virtual environment when present so dependencies resolve.
if [[ -d "$ROOT_DIR/.venv" && -f "$ROOT_DIR/.venv/bin/activate" ]]; then
	# shellcheck disable=SC1091
	source "$ROOT_DIR/.venv/bin/activate"
fi

export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}

exec python3 "$ROOT_DIR/app.py"

