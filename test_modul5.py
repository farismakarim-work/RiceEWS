"""
Test Script untuk MODUL 5 - Intervention Analysis
==================================================
Script ini menguji MODUL 5 dengan input dari output MODUL 4:
- Membaca shock_propagation_results.json dan network_edges.csv
- Mensimulasikan skenario intervensi per grade
- Memverifikasi semua output wajib tersimpan dan non-empty
Cara menjalankan:
    python test_modul5.py
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
    """Jalankan MODUL 1–4 jika outputnya belum tersedia."""
    processed_dir = root / "data" / "processed"
    module04_dir = processed_dir / "module_04"
    shock_json = module04_dir / "shock_propagation_results.json"

    if shock_json.exists() and shock_json.stat().st_size > 0:
        print("✓ MODUL 4 output sudah tersedia, skip upstream run.")
        return

    print("\nOutput MODUL 4 belum ada, menjalankan MODUL 1–4...\n")

    module03_dir = processed_dir / "module_03"
    network_json = module03_dir / "network_inference_results.json"

    if not (network_json.exists() and network_json.stat().st_size > 0):
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

    else:
        print("✓ MODUL 3 output sudah tersedia.")

    # MODUL 4
    print("Running MODUL 4...")
    network_edges_csv = module03_dir / "network_edges.csv"
    from modules.module_04_shock_propagation.shock_simulator import run_module4_shock_propagation

    run_module4_shock_propagation(
        network_results_path=str(network_json),
        network_edges_path=str(network_edges_csv),
        output_dir=str(module04_dir),
        config={
            "num_steps": 5,
            "shock_magnitude": 1.0,
            "top_n": 5,
        },
    )
    print("✓ MODUL 4 selesai.")


def main() -> int:
    print("\n" + "=" * 70)
    print("MODUL 5 - INTERVENTION ANALYSIS TEST")
    print("=" * 70)

    root = Path(__file__).parent
    module03_dir = root / "data" / "processed" / "module_03"
    module04_dir = root / "data" / "processed" / "module_04"
    module05_dir = root / "data" / "processed" / "module_05"

    # Ensure upstream outputs exist
    _run_upstream_modules(root)

    shock_json = module04_dir / "shock_propagation_results.json"
    network_edges_csv = module03_dir / "network_edges.csv"

    _check_non_empty(shock_json, "MODUL 4 output (shock_propagation_results.json)")

    # Run MODUL 5
    from modules.module_05_intervention_analysis.intervention_analyzer import (
        run_module5_intervention_analysis,
    )

    print("\nRunning MODUL 5...")
    results = run_module5_intervention_analysis(
        shock_results_path=str(shock_json),
        network_edges_path=str(network_edges_csv) if network_edges_csv.exists() else None,
        output_dir=str(module05_dir),
        config={
            "num_steps": 5,
            "shock_magnitude": 1.0,
            "top_n": 5,
            "node_attenuation_factor": 0.0,
            "edge_attenuation_factor": 0.0,
            "topk_control": 1,
        },
    )

    print("\nMODUL 5 selesai. Verifikasi output...")

    # Core outputs
    _check_non_empty(module05_dir / "intervention_results.json", "intervention_results.json")
    _check_non_empty(module05_dir / "intervention_summary.csv", "intervention_summary.csv")
    _check_non_empty(module05_dir / "intervention_summary.md", "intervention_summary.md")

    # Per-grade visualization
    grades = results.get("grades", [])
    for grade in grades:
        _check_non_empty(
            module05_dir / f"intervention_graph_{grade}.html",
            f"intervention_graph_{grade}.html",
        )

    # Verify visualization paths in results dict
    visualizations = results.get("visualizations", {})
    for grade in grades:
        if grade not in visualizations:
            raise KeyError(f"✗ Missing visualization path in results for grade: {grade}")

    # Schema checks on the JSON result
    per_grade = results.get("per_grade", {})
    for grade in grades:
        gd = per_grade.get(grade)
        if gd is None:
            raise KeyError(f"✗ Missing per_grade entry for grade: {grade}")

        for key in ("baseline", "scenarios", "ranking"):
            if key not in gd:
                raise KeyError(f"✗ Missing key '{key}' in per_grade['{grade}']")

        baseline = gd["baseline"]
        for b_key in (
            "total_impact_per_source",
            "affected_nodes_count",
            "most_influential_sources",
            "most_vulnerable_nodes",
            "cumulative_impact_total",
        ):
            if b_key not in baseline:
                raise KeyError(
                    f"✗ Missing key '{b_key}' in per_grade['{grade}']['baseline']"
                )

        scenarios = gd["scenarios"]
        for sc_name, sc in scenarios.items():
            for sc_key in ("description", "simulation", "comparison"):
                if sc_key not in sc:
                    raise KeyError(
                        f"✗ Missing key '{sc_key}' in scenario '{sc_name}' for grade '{grade}'"
                    )
            cmp = sc["comparison"]
            for cmp_key in (
                "impact_reduction_pct",
                "affected_nodes_reduction",
                "baseline_cumulative_impact",
                "intervention_cumulative_impact",
            ):
                if cmp_key not in cmp:
                    raise KeyError(
                        f"✗ Missing comparison key '{cmp_key}' for scenario '{sc_name}', grade '{grade}'"
                    )

        # Ranking should be sorted descending by impact_reduction_pct
        ranking = gd["ranking"]
        if len(ranking) > 1:
            for i in range(len(ranking) - 1):
                if ranking[i]["impact_reduction_pct"] < ranking[i + 1]["impact_reduction_pct"]:
                    raise ValueError(
                        f"✗ Ranking not sorted correctly for grade {grade}: "
                        f"{ranking[i]['scenario']} ({ranking[i]['impact_reduction_pct']}) "
                        f"< {ranking[i+1]['scenario']} ({ranking[i+1]['impact_reduction_pct']})"
                    )
        print(f"✓ Schema válido para grade {grade}")

    # Print summary
    print("\nSummary:")
    for grade in grades:
        gd = per_grade.get(grade, {})
        ranking = gd.get("ranking", [])
        best = ranking[0] if ranking else {}
        print(
            f"  Grade={grade}: best_scenario={best.get('scenario', '-')}, "
            f"impact_reduction={best.get('impact_reduction_pct', 0.0):.2f}%"
        )

    print("\n" + "=" * 70)
    print("✓ TEST MODUL 5 BERHASIL")
    print("=" * 70)
    print(f"Output directory: {module05_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
