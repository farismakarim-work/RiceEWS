"""
Test Script untuk MODUL 6 - Validation & Robustness
=====================================================
Script ini menguji MODUL 6 dengan input dari output MODUL 3, 4, dan 5:
- Membaca network_inference_results.json, shock_propagation_results.json,
  dan intervention_results.json
- Menjalankan sensitivity analysis dan stability metrics per grade
- Memverifikasi semua output wajib tersimpan dan non-empty
Cara menjalankan:
    python test_modul6.py
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
    """Jalankan MODUL 1–5 jika outputnya belum tersedia."""
    processed_dir = root / "data" / "processed"
    module05_dir = processed_dir / "module_05"
    intervention_json = module05_dir / "intervention_results.json"

    if intervention_json.exists() and intervention_json.stat().st_size > 0:
        print("✓ MODUL 5 output sudah tersedia, skip upstream run.")
        return

    print("\nOutput MODUL 5 belum ada, menjalankan MODUL 1–5...\n")

    module03_dir = processed_dir / "module_03"
    module04_dir = processed_dir / "module_04"
    network_json = module03_dir / "network_inference_results.json"
    shock_json = module04_dir / "shock_propagation_results.json"

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
                config={"lag_order": 4, "price_col": "price_diff", "significance_level": 0.05},
            )
        else:
            print("✓ MODUL 2 output sudah tersedia.")

        # MODUL 3
        print("Running MODUL 3...")
        granger_json = processed_dir / "granger_results.json"
        from modules.module_03_network_inference.network_builder import run_module3_network_inference

        run_module3_network_inference(
            granger_json_path=str(granger_json),
            output_dir=str(module03_dir),
        )
        print("✓ MODUL 3 selesai.")
    else:
        print("✓ MODUL 3 output sudah tersedia.")

    if not (shock_json.exists() and shock_json.stat().st_size > 0):
        print("Running MODUL 4...")
        network_edges_csv = module03_dir / "network_edges.csv"
        from modules.module_04_shock_propagation.shock_simulator import run_module4_shock_propagation

        run_module4_shock_propagation(
            network_results_path=str(network_json),
            network_edges_path=str(network_edges_csv),
            output_dir=str(module04_dir),
            config={"num_steps": 5, "shock_magnitude": 1.0, "top_n": 5},
        )
        print("✓ MODUL 4 selesai.")
    else:
        print("✓ MODUL 4 output sudah tersedia.")

    print("Running MODUL 5...")
    network_edges_csv = module03_dir / "network_edges.csv"
    from modules.module_05_intervention_analysis.intervention_analyzer import (
        run_module5_intervention_analysis,
    )

    run_module5_intervention_analysis(
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
    print("✓ MODUL 5 selesai.")


def main() -> int:
    print("\n" + "=" * 70)
    print("MODUL 6 - VALIDATION & ROBUSTNESS TEST")
    print("=" * 70)

    root = Path(__file__).parent
    module03_dir = root / "data" / "processed" / "module_03"
    module04_dir = root / "data" / "processed" / "module_04"
    module05_dir = root / "data" / "processed" / "module_05"
    module06_dir = root / "data" / "processed" / "module_06"

    # Ensure upstream outputs exist
    _run_upstream_modules(root)

    network_json = module03_dir / "network_inference_results.json"
    shock_json = module04_dir / "shock_propagation_results.json"
    intervention_json = module05_dir / "intervention_results.json"

    _check_non_empty(network_json, "MODUL 3 output (network_inference_results.json)")
    _check_non_empty(shock_json, "MODUL 4 output (shock_propagation_results.json)")
    _check_non_empty(intervention_json, "MODUL 5 output (intervention_results.json)")

    # Run MODUL 6
    from modules.module_06_validation_robustness.robustness_validator import (
        run_module6_validation_robustness,
    )

    print("\nRunning MODUL 6...")
    results = run_module6_validation_robustness(
        network_results_path=str(network_json),
        shock_results_path=str(shock_json),
        intervention_results_path=str(intervention_json),
        output_dir=str(module06_dir),
        config={
            "propagation_steps_range": [3, 5, 7],
            "attenuation_factors_range": [0.3, 0.5, 0.7],
            "top_n": 5,
            "high_stability_threshold": 0.70,
            "low_stability_threshold": 0.40,
        },
    )

    print("\nMODUL 6 selesai. Verifikasi output...")

    # Core outputs
    _check_non_empty(module06_dir / "robustness_results.json", "robustness_results.json")
    _check_non_empty(module06_dir / "robustness_summary.csv", "robustness_summary.csv")
    _check_non_empty(module06_dir / "robustness_summary.md", "robustness_summary.md")

    # Per-grade visualization
    grades = results.get("grades", [])
    for grade in grades:
        _check_non_empty(
            module06_dir / f"robustness_graph_{grade}.html",
            f"robustness_graph_{grade}.html",
        )

    # Verify visualization paths in results dict
    visualizations = results.get("visualizations", {})
    for grade in grades:
        if grade not in visualizations:
            raise KeyError(f"✗ Missing visualization path in results for grade: {grade}")

    # Schema checks
    per_grade = results.get("per_grade", {})
    valid_confidence = {"HIGH", "MEDIUM", "LOW"}
    for grade in grades:
        gd = per_grade.get(grade)
        if gd is None:
            raise KeyError(f"✗ Missing per_grade entry for grade: {grade}")

        for key in ("nodes", "edges_count", "sweep_results", "stability_metrics", "confidence_level",
                    "cross_module_consistency"):
            if key not in gd:
                raise KeyError(f"✗ Missing key '{key}' in per_grade['{grade}']")

        sm = gd["stability_metrics"]
        for sm_key in ("influential_consistency", "vulnerable_consistency",
                       "impact_cov", "stability_score"):
            if sm_key not in sm:
                raise KeyError(f"✗ Missing stability metric '{sm_key}' for grade '{grade}'")
            val = sm[sm_key]
            if not isinstance(val, (int, float)):
                raise TypeError(f"✗ stability metric '{sm_key}' should be numeric for grade '{grade}'")

        # Consistency scores must be in [0, 1]
        for cons_key in ("influential_consistency", "vulnerable_consistency", "stability_score"):
            val = sm[cons_key]
            if not (0.0 <= val <= 1.0):
                raise ValueError(
                    f"✗ {cons_key}={val} out of [0,1] range for grade '{grade}'"
                )

        conf = gd["confidence_level"]
        if conf not in valid_confidence:
            raise ValueError(f"✗ Invalid confidence_level='{conf}' for grade '{grade}'")

        cross = gd["cross_module_consistency"]
        for cross_key in ("m4_m5_influential_jaccard", "best_intervention_impact_reduction_pct",
                          "cross_module_consistent"):
            if cross_key not in cross:
                raise KeyError(
                    f"✗ Missing cross_module key '{cross_key}' for grade '{grade}'"
                )

        # Sweep results: expect at least propagation_steps and attenuation_factor entries
        sweep = gd["sweep_results"]
        if gd.get("edges_count", 0) > 0:
            param_types = {v.get("param_type") for v in sweep.values()}
            if "propagation_steps" not in param_types:
                raise ValueError(f"✗ Missing propagation_steps sweep for grade '{grade}'")
            if "attenuation_factor" not in param_types:
                raise ValueError(f"✗ Missing attenuation_factor sweep for grade '{grade}'")

        print(f"✓ Schema valid untuk grade {grade}: "
              f"stability={sm['stability_score']:.4f}, "
              f"confidence={conf}")

    # Print summary
    print("\nSummary:")
    for grade in grades:
        gd = per_grade.get(grade, {})
        sm = gd.get("stability_metrics", {})
        print(
            f"  Grade={grade}: stability_score={sm.get('stability_score', 0.0):.4f}, "
            f"confidence={gd.get('confidence_level', 'LOW')}, "
            f"inf_consistency={sm.get('influential_consistency', 0.0):.4f}, "
            f"vul_consistency={sm.get('vulnerable_consistency', 0.0):.4f}"
        )

    print("\n" + "=" * 70)
    print("✓ TEST MODUL 6 BERHASIL")
    print("=" * 70)
    print(f"Output directory: {module06_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
