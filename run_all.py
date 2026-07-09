"""
run_all.py — Execute the complete RiceEWS pipeline.

This script runs all three modules in sequence using the project's
default configuration.  It is the standard entry point for normal users.

Usage:
  python run_all.py
"""

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    engine = Path(__file__).resolve().parent / "run_pipeline.py"
    result = subprocess.run(
        [sys.executable, str(engine), "--all"] + sys.argv[1:],
        check=False,
    )
    sys.exit(result.returncode)
