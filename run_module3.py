"""
run_module3.py — Execute Module 3 only (Network Recovery).

Prerequisites: Module 2 outputs must exist before running this script.
               Run `python run_module2.py` first if they are missing.

Usage:
  python run_module3.py
"""

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    engine = Path(__file__).resolve().parent / "run_pipeline.py"
    result = subprocess.run(
        [sys.executable, str(engine), "--module", "3"] + sys.argv[1:],
        check=False,
    )
    sys.exit(result.returncode)
