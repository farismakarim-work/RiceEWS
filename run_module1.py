"""
run_module1.py — Execute Module 1 only (Data Preprocessing).

Usage:
  python run_module1.py
"""

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    engine = Path(__file__).resolve().parent / "run_pipeline.py"
    result = subprocess.run(
        [sys.executable, str(engine), "--module", "1"] + sys.argv[1:],
        check=False,
    )
    sys.exit(result.returncode)
