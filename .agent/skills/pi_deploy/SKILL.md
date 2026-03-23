---
name: raspberry_pi_deployment
description: Pushes the code base to a remote Raspberry Pi and runs it.
---

# Raspberry Pi Deployment Skill

This skill allows Antigravity to deploy the `hangermon` codebase to a Raspberry Pi. It handles:
- Code synchronization via `rsync`.
- APT dependency installation (`picamera2`, `ffmpeg`, etc.).
- `uv` installation and usage for managed Python environments.
- Installation as a systemd service for automatic startup.

## Prerequisites

- SSH access to the target Raspberry Pi.
- `rsync` installed on both the local machine and the Pi.
- Python 3.11+ on the Pi.
- Dedicated user for the service (defaults to the SSH user).

## Usage

Use the `deploy_to_pi.sh` script to trigger a full sync and deployment.

### 1. Basic Deployment
Reach the Pi at `raspberrypi.local` with the username `pi`.

```bash
./scripts/deploy_to_pi.sh
```

### 2. Custom Host / User
If your Pi is at a different address or has a custom username:

```bash
./scripts/deploy_to_pi.sh username@hostname.local
```

### 3. Verification
Once deployed, check the status of the service on the Pi:

```bash
ssh <user>@<host> "sudo systemctl status hangermon.service"
```

The web interface should be available at `http://<host>:8000`.

## Key Files

- `scripts/deploy_to_pi.sh`: The local script that pushes code.
