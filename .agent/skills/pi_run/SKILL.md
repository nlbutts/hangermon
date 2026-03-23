---
name: raspberry_pi_run
description: Runs the code base on a remote Raspberry Pi.
---

# Raspberry Pi Run Skill

This skill allows Antigravity to run the `hangermon` codebase on a Raspberry Pi. It handles:

## Prerequisites

- SSH access to the target Raspberry Pi.
- Python 3.11+ on the Pi.
- Dedicated user for the service (defaults to the SSH user).

## Usage

Use the `run_on_pi.sh` script to trigger a full sync and deployment.

### 1. Basic Deployment
Reach the Pi at `hanger.local` with the username `nlbutts`.

```bash
./scripts/run_on_pi.sh
```

### 2. Custom Host / User
If your Pi is at a different address or has a custom username:

```bash
./scripts/run_on_pi.sh username@hostname.local
```

### 3. Verification
It should print to stdout and should be able to see what is running.

## Key Files

- `scripts/run_on_pi.sh`: The local script that runs the code.
