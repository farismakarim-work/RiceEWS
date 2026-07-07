"""
Test Script untuk MODUL 7 - Policy Recommendation & EWS Dashboard
==================================================================
Script ini menguji MODUL 7 dengan input dari output MODUL 5 dan MODUL 6:
- Membaca intervention_results.json dan robustness_results.json
- Menjalankan sintesis rekomendasi per grade
- Memverifikasi semua output wajib tersimpan dan non-empty
- Validasi trigger level hanya GREEN/YELLOW/RED
- Cek ews_dashboard.html ada

Cara menjalankan:
    python test_modul7.py
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
    """Jalankan MODUL 1–6 jika outputnya belum tersedia."""
    processed_dir = root / "data" / "processed"
    module05_dir = processed_dir / "module_05"
    module06_dir = processed_dir / "module_06"
    intervention_json = module05_dir / "intervention_results.json"
    robustness_json = module06_dir / "robustness_results.json"

    if (
        robustness_json.exists() and robustness_json.stat().st_size > 0
        and intervention_json.exists() and intervention_json.stat().st_size > 0
    ):
        print("✓ MODUL 5 & 6 output sudah tersedia, skip upstream run.")
        return

    print("\nOutput MODUL 5/6 belum ada, menjalankan MODUL 1–6...\n")

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

    if not (intervention_json.exists() and intervention_json.stat().st_size > 0):
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
    else:
        print("✓ MODUL 5 output sudah tersedia.")

    if not (robustness_json.exists() and robustness_json.stat().st_size > 0):
        print("Running MODUL 6...")
        from modules.module_06_validation_robustness.robustness_validator import (
            run_module6_validation_robustness,
        )

        run_module6_validation_robustness(
            shock_results_path=str(shock_json),
            intervention_results_path=str(intervention_json),
            network_results_path=str(network_json),
            output_dir=str(module06_dir),
            config={"num_perturb": 5, "noise_std": 0.05, "top_n": 5},
        )
        print("✓ MODUL 6 selesai.")
    else:
        print("✓ MODUL 6 output sudah tersedia.")


def main() -> int:
    print("\n" + "=" * 70)
    print("MODUL 7 - POLICY RECOMMENDATION & EWS DASHBOARD TEST")
    print("=" * 70)

    root = Path(__file__).parent
    module05_dir = root / "data" / "processed" / "module_05"
    module06_dir = root / "data" / "processed" / "module_06"
    module07_dir = root / "data" / "processed" / "module_07"

    # --- 0) Jalankan upstream modules jika diperlukan ---
    _run_upstream_modules(root)

    # --- 1) Cek input Modul 5 & 6 ---
    print("\n[1] Memeriksa input MODUL 5 & 6...")
    intervention_json = module05_dir / "intervention_results.json"
    robustness_json = module06_dir / "robustness_results.json"

    _check_non_empty(intervention_json, "MODUL 5 output (intervention_results.json)")
    _check_non_empty(robustness_json, "MODUL 6 output (robustness_results.json)")

    # --- 2) Jalankan Modul 7 ---
    print("\n[2] Menjalankan MODUL 7...")
    from modules.module_07_policy_recommendation.policy_recommender import (
        run_module7_policy_recommendation,
    )

    results = run_module7_policy_recommendation(
        intervention_results_path=str(intervention_json),
        robustness_results_path=str(robustness_json),
        output_dir=str(module07_dir),
        config={
            "red_impact_threshold": 4.0,
            "yellow_impact_threshold": 2.0,
            "red_vulnerability_threshold": 0.50,
            "yellow_vulnerability_threshold": 0.30,
            "min_impact_reduction_pct": 5.0,
            "rule_version": "v1.0",
        },
    )

    # --- 3) Cek output wajib ---
    print("\n[3] Verifikasi output wajib...")
    _check_non_empty(module07_dir / "policy_recommendation_results.json", "policy_recommendation_results.json")
    _check_non_empty(module07_dir / "policy_recommendation_summary.csv", "policy_recommendation_summary.csv")
    _check_non_empty(module07_dir / "policy_recommendation_summary.md", "policy_recommendation_summary.md")
    _check_non_empty(module07_dir / "ews_dashboard.html", "ews_dashboard.html")

    # --- 4) Validasi trigger level hanya GREEN/YELLOW/RED ---
    print("\n[4] Validasi trigger level per grade...")
    valid_triggers = {"GREEN", "YELLOW", "RED"}
    grades = results.get("grades", [])
    per_grade = results.get("per_grade", {})

    if not grades:
        raise ValueError("✗ Tidak ada grade dalam results")

    for grade in grades:
        rec = per_grade.get(grade)
        if rec is None:
            raise KeyError(f"✗ Missing per_grade entry untuk grade: {grade}")

        trigger = rec.get("trigger_level")
        if trigger not in valid_triggers:
            raise ValueError(
                f"✗ Invalid trigger_level='{trigger}' untuk grade '{grade}'. "
                f"Harus salah satu dari: {valid_triggers}"
            )

        confidence = rec.get("confidence_level")
        if confidence not in {"HIGH", "MEDIUM", "LOW"}:
            raise ValueError(
                f"✗ Invalid confidence_level='{confidence}' untuk grade '{grade}'"
            )

        # Cek required keys (termasuk stability_score dan cumulative_impact_index)
        for key in (
            "priority_markets",
            "priority_interventions",
            "expected_impact",
            "confidence_level",
            "short_term_actions",
            "medium_term_actions",
            "weekly_monitoring_indicators",
            "stability_score",
            "cumulative_impact_index",
        ):
            if key not in rec:
                raise KeyError(f"✗ Missing key '{key}' dalam per_grade['{grade}']")

        # priority_markets dan interventions harus list
        if not isinstance(rec["priority_markets"], list):
            raise TypeError(f"✗ priority_markets harus list untuk grade '{grade}'")
        if not isinstance(rec["priority_interventions"], list):
            raise TypeError(f"✗ priority_interventions harus list untuk grade '{grade}'")

        # stability_score harus float dalam [0, 1]
        stability = rec["stability_score"]
        if not isinstance(stability, (int, float)) or not (0.0 <= float(stability) <= 1.0):
            raise ValueError(
                f"✗ stability_score={stability!r} harus float dalam [0, 1] untuk grade '{grade}'"
            )

        # cumulative_impact_index harus non-negative numeric
        impact_idx = rec["cumulative_impact_index"]
        if not isinstance(impact_idx, (int, float)) or float(impact_idx) < 0:
            raise ValueError(
                f"✗ cumulative_impact_index={impact_idx!r} harus >= 0 untuk grade '{grade}'"
            )

        print(
            f"  ✓ Grade {grade}: trigger={trigger}, confidence={confidence}, "
            f"stability={stability:.4f}, impact_index={impact_idx:.4f}"
        )

    # --- 5) Cek ews_dashboard.html mengandung konten penting ---
    print("\n[5] Validasi konten ews_dashboard.html...")
    html_content = (module07_dir / "ews_dashboard.html").read_text(encoding="utf-8")

    required_strings = [
        "RiceEWS",
        "generated_at",
        "grade_count",
        "rule_version",
        "GREEN",
        "YELLOW",
        "RED",
    ]
    for s in required_strings:
        if s not in html_content:
            raise ValueError(f"✗ String '{s}' tidak ditemukan dalam ews_dashboard.html")
    print(f"  ✓ ews_dashboard.html valid ({len(html_content)} karakter)")

    # Per-grade section in dashboard
    for grade in grades:
        if f"grade-{grade}" not in html_content:
            raise ValueError(f"✗ Section grade '{grade}' tidak ditemukan dalam dashboard HTML")
        print(f"  ✓ Section grade {grade} ditemukan di dashboard")

    # --- 6) Cek CSV struktur ---
    print("\n[6] Validasi struktur CSV...")
    import csv

    csv_path = module07_dir / "policy_recommendation_summary.csv"
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    expected_cols = {
        "grade", "trigger_level", "confidence_level", "stability_score",
        "cumulative_impact_index", "vulnerability_concentration",
        "priority_markets", "priority_interventions", "expected_impact",
    }
    actual_cols = set(rows[0].keys()) if rows else set()
    missing_cols = expected_cols - actual_cols
    if missing_cols:
        raise ValueError(f"✗ Kolom CSV yang hilang: {missing_cols}")

    print(f"  ✓ CSV valid: {len(rows)} baris, {len(actual_cols)} kolom")

    for row in rows:
        if row["trigger_level"] not in valid_triggers:
            raise ValueError(
                f"✗ trigger_level='{row['trigger_level']}' tidak valid dalam CSV (grade={row['grade']})"
            )

    # --- Summary ---
    print("\n" + "=" * 70)
    print("✓ TEST MODUL 7 BERHASIL")
    print("=" * 70)
    print(f"Output directory: {module07_dir}")
    print(f"\nRingkasan per grade:")
    for grade in grades:
        rec = per_grade[grade]
        print(
            f"  Grade={grade}: trigger={rec['trigger_level']}, "
            f"confidence={rec['confidence_level']}, "
            f"stability={rec['stability_score']:.4f}, "
            f"impact_index={rec['cumulative_impact_index']:.4f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

