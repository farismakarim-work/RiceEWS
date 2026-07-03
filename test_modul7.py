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


def main() -> int:
    print("\n" + "=" * 70)
    print("MODUL 7 - POLICY RECOMMENDATION & EWS DASHBOARD TEST")
    print("=" * 70)

    root = Path(__file__).parent
    module05_dir = root / "data" / "processed" / "module_05"
    module06_dir = root / "data" / "processed" / "module_06"
    module07_dir = root / "data" / "processed" / "module_07"

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

        # Cek required keys
        for key in (
            "priority_markets",
            "priority_interventions",
            "expected_impact",
            "confidence_level",
            "short_term_actions",
            "medium_term_actions",
            "weekly_monitoring_indicators",
        ):
            if key not in rec:
                raise KeyError(f"✗ Missing key '{key}' dalam per_grade['{grade}']")

        # priority_markets dan interventions harus list
        if not isinstance(rec["priority_markets"], list):
            raise TypeError(f"✗ priority_markets harus list untuk grade '{grade}'")
        if not isinstance(rec["priority_interventions"], list):
            raise TypeError(f"✗ priority_interventions harus list untuk grade '{grade}'")

        print(
            f"  ✓ Grade {grade}: trigger={trigger}, confidence={confidence}, "
            f"markets={rec['priority_markets'][:3]}"
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
