import json
from pathlib import Path


def _assert_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"✗ Missing {label}: {path}")
    print(f"✓ Found {label}: {path}")


def _assert_non_empty(path: Path, label: str) -> None:
    _assert_exists(path, label)
    if path.stat().st_size == 0:
        raise ValueError(f"✗ Empty {label}: {path}")
    print(f"✓ Non-empty {label} ({path.stat().st_size} bytes)")


def _is_plotly_installed() -> bool:
    try:
        import plotly  # noqa: F401
        return True
    except ImportError:
        return False


def main() -> None:
    print("=" * 70)
    print("MODUL 2 - GRANGER TEST (ALL OUTPUTS + VISUALIZATION) TEST")
    print("=" * 70)

    root = Path(__file__).resolve().parent
    processed_dir = root / "data" / "processed"

    # Required input from MODUL 1
    preprocessed_csv = processed_dir / "preprocessed_pilot_data.csv"
    _assert_exists(preprocessed_csv, "MODUL 1 output CSV")

    # Run MODUL 2 pipeline
    from src.modules.causality_testing.granger_tester import run_full_granger_analysis

    print("\nRunning MODUL 2 pipeline...")
    run_full_granger_analysis(
        input_file=str(preprocessed_csv),
        output_dir=str(processed_dir),
        config={
            "lag_order": 4,
            "price_col": "price_diff",
            "significance_level": 0.05,
        },
    )

    print("\nPipeline finished. Verifying outputs...")

    # Core outputs
    json_file = processed_dir / "granger_results.json"
    excel_file = processed_dir / "granger_results.xlsx"
    npz_file = processed_dir / "granger_matrices.npz"
    md_file = processed_dir / "granger_results.md"

    _assert_non_empty(json_file, "JSON output")
    _assert_non_empty(excel_file, "Excel output")
    _assert_non_empty(npz_file, "NPZ output")
    _assert_non_empty(md_file, "Markdown output")

    # Per-grade CSV outputs
    grades = ["low1", "low2", "med1", "med2"]
    for g in grades:
        csv_file = processed_dir / f"granger_pairwise_{g}.csv"
        _assert_non_empty(csv_file, f"CSV output ({g})")

    # Visualization outputs from current generator (HTML)
    if not _is_plotly_installed():
        print("\n⚠ Plotly not installed; skipping visualization file checks.")
    else:
        expected_plots = [
            "network_graph_low1.html",
            "network_graph_low2.html",
            "network_graph_med1.html",
            "network_graph_med2.html",
            "heatmap_low1.html",
            "heatmap_low2.html",
            "heatmap_med1.html",
            "heatmap_med2.html",
            "ranking_low1.html",
            "ranking_low2.html",
            "ranking_med1.html",
            "ranking_med2.html",
        ]

        missing_plots = []
        for plot_name in expected_plots:
            plot_path = processed_dir / plot_name
            if plot_path.exists() and plot_path.stat().st_size > 0:
                print(f"✓ Plot: {plot_path}")
            else:
                missing_plots.append(plot_name)

        if missing_plots:
            print("\n⚠ Some expected visualization files were not found:")
            for p in missing_plots:
                print(f"  - {p}")
            print("\nEnsure plotly is installed and visualization generation is enabled in modul 2.")
        else:
            print("\n✓ All expected visualization files found.")

    # Minimal schema check for JSON
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise TypeError("granger_results.json must contain a JSON object")

    print("\n✓ JSON schema sanity check passed (top-level object)")

    print("\n" + "=" * 70)
    print("✓ MODUL 2 TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
