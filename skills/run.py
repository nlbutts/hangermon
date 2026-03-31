#!/usr/bin/env python3
"""Skill to run hangermon on a remote unit."""

import subprocess
import sys


def run(unit: str = "hanger") -> None:
    """Run hangermon on the specified unit."""
    cmd = f"ssh nlbutts@{unit} 'cd /home/nlbutts/hangermon && ./start.sh'"
    subprocess.run(cmd, shell=True, check=True)


if __name__ == "__main__":
    unit = sys.argv[1] if len(sys.argv) > 1 else "hanger"
    run(unit)
