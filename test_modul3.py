"""
Test Script untuk MODUL 3 - Network Inference
============================================

Script ini menguji MODUL 3 dengan input dari output MODUL 2:
- Membaca granger_results.json
- Membangun edge list per grade
- Mengecek properti DAG/polytree sederhana
- Menghasilkan output terpisah di data/processed/module_03

Cara menjalankan:
    python test_modul3.py
"""

import sys
from pathlib import Path


# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from modules.module_03_network_inference.network_builder import run_module3_network_inference


def _check_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"✗ Missing {label}: {path}")
    print(f"✓ Found {label}: {path}")


def _check_non_empty(path: Path, label: str) -> None:
    _check_exists(path, label)
    if path.stat().st_size == 0:
        raise ValueError(f"✗ Empty {label}: {path}")
    print(f"✓ Non-empty {label} ({path.stat().st_size} bytes)")


def main() -> int:
    print("\n" + "=" * 70)
    print("MODUL 3 - NETWORK INFERENCE TEST")
    print("=" * 70)

    root = Path(__file__).parent
    granger_json = root / "data" / "processed" / "granger_results.json"
    output_dir = root / "data" / "processed" / "module_03"

    _check_non_empty(granger_json, "MODUL 2 output (granger_results.json)")

    print("\nRunning MODUL 3...")
    results = run_module3_network_inference(
        granger_json_path=str(granger_json),
        output_dir=str(output_dir),
    )

    print("\nMODUL 3 selesai. Verifikasi output...")

    # Core outputs
    _check_non_empty(output_dir / "network_inference_results.json", "network_inference_results.json")
    _check_non_empty(output_dir / "network_edges.csv", "network_edges.csv")
    _check_non_empty(output_dir / "network_summary.csv", "network_summary.csv")
    _check_non_empty(output_dir / "network_summary.md", "network_summary.md")

    # Per-grade leader outputs
    for grade in results.get("grades", []):
        _check_non_empty(output_dir / f"market_leaders_{grade}.csv", f"market_leaders_{grade}.csv")

    print("\nSummary:")
    for row in results.get("summary", []):
        print(
            f"  - Grade={row['grade']}, Nodes={row['nodes']}, Edges={row['edges']}, "
            f"DAG={row['is_dag']}, Polytree={row['is_polytree']}, "
            f"TopLeader={row['top_leader']}"
        )

    print("\n" + "=" * 70)
    print("✓ TEST MODUL 3 BERHASIL")
    print("=" * 70)
    print(f"Output directory: {output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
