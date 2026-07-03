"""
Test Script untuk MODUL 4 - Shock Propagation Simulation
=========================================================

Script ini menguji MODUL 4 dengan input dari output MODUL 3:
- Membaca network_inference_results.json dan network_edges.csv
- Mensimulasikan penyebaran shock per grade
- Memverifikasi semua output wajib tersimpan dan non-empty

Cara menjalankan:
    python test_modul4.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def _check_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"✗ Missing {label}: {path}")
    print(f"✓ Found {label}: {path}")


def _check_non_empty(path: Path, label: str) -> None:
    _check_exists(path, label)
    if path.stat().st_size == 0:
        raise ValueError(f"✗ Empty {label}: {path}")
    print(f"✓ Non-empty {label} ({path.stat().st_size} bytes)")


def _run_upstream_modules(root: Path) -> None:
    """Jalankan MODUL 1–3 jika outputnya belum tersedia."""
    processed_dir = root / "data" / "processed"
    module03_dir = processed_dir / "module_03"

    network_json = module03_dir / "network_inference_results.json"
    if network_json.exists() and network_json.stat().st_size > 0:
        print("✓ MODUL 3 output sudah tersedia, skip upstream run.")
        return

    print("\nOutput MODUL 3 belum ada, menjalankan MODUL 1–3...\n")

    # MODUL 1
    preprocessed_csv = processed_dir / "preprocessed_pilot_data.csv"
    if not (preprocessed_csv.exists() and preprocessed_csv.stat().st_size > 0):
        print("Running MODUL 1...")
        from modules.preprocessing.data_preprocessor import run_full_preprocessing_pipeline

        raw_dir = root / "data" / "raw"
        xlsx_files = list(raw_dir.glob("*.xlsx"))
        if not xlsx_files:
            raise FileNotFoundError(f"Tidak ada file .xlsx di {raw_dir}")
        run_full_preprocessing_pipeline(
            input_file=str(xlsx_files[0]),
            output_file=str(preprocessed_csv),
        )
    else:
        print("✓ MODUL 1 output sudah tersedia.")

    # MODUL 2
    granger_json = processed_dir / "granger_results.json"
    if not (granger_json.exists() and granger_json.stat().st_size > 0):
        print("Running MODUL 2...")
        from modules.causality_testing.granger_tester import run_full_granger_analysis

        run_full_granger_analysis(
            input_file=str(preprocessed_csv),
            output_dir=str(processed_dir),
            config={
                "lag_order": 4,
                "price_col": "price_diff",
                "significance_level": 0.05,
            },
        )
    else:
        print("✓ MODUL 2 output sudah tersedia.")

    # MODUL 3
    print("Running MODUL 3...")
    from modules.module_03_network_inference.network_builder import run_module3_network_inference

    run_module3_network_inference(
        granger_json_path=str(granger_json),
        output_dir=str(module03_dir),
    )
    print("✓ MODUL 3 selesai.")


def main() -> int:
    print("\n" + "=" * 70)
    print("MODUL 4 - SHOCK PROPAGATION SIMULATION TEST")
    print("=" * 70)

    root = Path(__file__).parent
    module03_dir = root / "data" / "processed" / "module_03"
    module04_dir = root / "data" / "processed" / "module_04"

    # Ensure upstream outputs exist
    _run_upstream_modules(root)

    network_json = module03_dir / "network_inference_results.json"
    network_edges_csv = module03_dir / "network_edges.csv"

    _check_non_empty(network_json, "MODUL 3 output (network_inference_results.json)")
    _check_non_empty(network_edges_csv, "MODUL 3 output (network_edges.csv)")

    # Run MODUL 4
    from modules.module_04_shock_propagation.shock_simulator import (
        run_module4_shock_propagation,
    )

    print("\nRunning MODUL 4...")
    results = run_module4_shock_propagation(
        network_results_path=str(network_json),
        network_edges_path=str(network_edges_csv),
        output_dir=str(module04_dir),
        config={
            "num_steps": 5,
            "shock_magnitude": 1.0,
            "top_n": 5,
        },
    )

    print("\nMODUL 4 selesai. Verifikasi output...")

    # Core outputs
    _check_non_empty(module04_dir / "shock_propagation_results.json", "shock_propagation_results.json")
    _check_non_empty(module04_dir / "shock_summary.csv", "shock_summary.csv")
    _check_non_empty(module04_dir / "shock_summary.md", "shock_summary.md")

    # Per-grade visualization
    grades = results.get("grades", [])
    for grade in grades:
        _check_non_empty(module04_dir / f"shock_graph_{grade}.html", f"shock_graph_{grade}.html")

    # Verify visualization paths in results dict
    visualizations = results.get("visualizations", {})
    for grade in grades:
        if grade not in visualizations:
            raise KeyError(f"✗ Missing visualization path in results for grade: {grade}")

    # Schema checks on the JSON result
    per_grade = results.get("per_grade", {})
    for grade in grades:
        gr = per_grade.get(grade)
        if gr is None:
            raise KeyError(f"✗ Missing per_grade entry for grade: {grade}")

        for key in ("nodes", "edges_count", "most_influential_sources", "most_vulnerable_nodes"):
            if key not in gr:
                raise KeyError(f"✗ Missing key '{key}' in per_grade['{grade}']")

        # If edges exist, we expect at least some metrics
        if gr["edges_count"] > 0:
            if not isinstance(gr["most_influential_sources"], list):
                raise TypeError(f"✗ most_influential_sources should be a list for grade {grade}")
            if not isinstance(gr["most_vulnerable_nodes"], list):
                raise TypeError(f"✗ most_vulnerable_nodes should be a list for grade {grade}")

    # Print summary
    print("\nSummary:")
    for grade in grades:
        gr = per_grade.get(grade, {})
        top_src = gr.get("most_influential_sources", [])
        top_vuln = gr.get("most_vulnerable_nodes", [])
        top_src_str = top_src[0]["node"] if top_src else "-"
        top_vuln_str = top_vuln[0]["node"] if top_vuln else "-"
        print(
            f"  Grade={grade}: nodes={len(gr.get('nodes', []))}, "
            f"edges={gr.get('edges_count', 0)}, "
            f"top_source={top_src_str}, "
            f"top_vulnerable={top_vuln_str}"
        )

    print("\n" + "=" * 70)
    print("✓ TEST MODUL 4 BERHASIL")
    print("=" * 70)
    print(f"Output directory: {module04_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
