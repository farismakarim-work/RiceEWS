import sys
import subprocess
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent / "src"))

from modules.preprocessing import data_preprocessor
from modules.preprocessing.data_preprocessor import DataPreprocessor, run_full_preprocessing_pipeline


REPO_ROOT = Path(__file__).resolve().parent
PILOT_DATASET = REPO_ROOT / "data" / "raw" / "Pilot Dataset.xlsx"


def _sample_raw_dataset(price_offset: int = 0) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=12, freq="D")
    rows = []
    for market_id in [101, 102]:
        for grade in ["low1", "med1"]:
            for index, date in enumerate(dates):
                rows.append(
                    {
                        "date": date,
                        "market_id": market_id,
                        "grade": grade,
                        "price": 100 + price_offset + market_id + index,
                    }
                )
    return pd.DataFrame(rows)


def test_module1_pilot_pipeline_writes_module01_outputs(tmp_path):
    output_file = tmp_path / "module_01" / "preprocessed_pilot_data.csv"

    df_processed = run_full_preprocessing_pipeline(
        input_file=PILOT_DATASET,
        output_file=output_file,
        config={"detrend_method": "none", "duplicate_strategy": "error"},
    )

    assert not df_processed.empty
    assert output_file.exists()
    assert (output_file.parent / "preprocessed_pilot_data_report.json").exists()
    assert output_file.parent.name == "module_01"
    assert {"date", "market_id", "grade", "price_diff"}.issubset(df_processed.columns)


def test_module1_discovers_multiple_xlsx_files_when_input_is_none(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    first = _sample_raw_dataset()
    second = _sample_raw_dataset(price_offset=1000)
    first.to_excel(raw_dir / "first.xlsx", index=False)
    second.iloc[:0].to_excel(raw_dir / "second.xlsx", index=False)

    monkeypatch.setattr(data_preprocessor, "RAW_DATA_DIR", raw_dir)

    output_file = tmp_path / "module_01" / "detected.csv"
    df_processed = run_full_preprocessing_pipeline(
        input_file=None,
        output_file=output_file,
        config={"detrend_method": "none", "duplicate_strategy": "error"},
    )

    assert len(df_processed) > 0
    assert output_file.exists()


def test_module1_duplicate_handling_strategies(tmp_path):
    first_path = tmp_path / "first.xlsx"
    second_path = tmp_path / "second.xlsx"

    first = _sample_raw_dataset()
    second = first.copy()
    second.loc[:, "price"] = second["price"] + 500

    first.to_excel(first_path, index=False)
    second.to_excel(second_path, index=False)

    preprocessor = DataPreprocessor(verbose=False)

    with pytest.raises(ValueError, match="Duplicate"):
        preprocessor.load_pilot_data([first_path, second_path], duplicate_strategy="error")

    keep_first = preprocessor.load_pilot_data(
        [first_path, second_path],
        duplicate_strategy="keep_first",
    )
    keep_last = preprocessor.load_pilot_data(
        [first_path, second_path],
        duplicate_strategy="keep_last",
    )

    merged_key = keep_first.iloc[0][["date", "market_id", "grade"]].to_dict()
    first_price = keep_first.loc[
        (keep_first["date"] == merged_key["date"])
        & (keep_first["market_id"] == merged_key["market_id"])
        & (keep_first["grade"] == merged_key["grade"]),
        "price",
    ].iloc[0]
    last_price = keep_last.loc[
        (keep_last["date"] == merged_key["date"])
        & (keep_last["market_id"] == merged_key["market_id"])
        & (keep_last["grade"] == merged_key["grade"]),
        "price",
    ].iloc[0]

    assert first_price != last_price
    assert first_price == first.iloc[0]["price"]
    assert last_price == second.iloc[0]["price"]


def test_module1_filters_applied_before_preprocessing(tmp_path):
    dataset_path = tmp_path / "filter_input.xlsx"
    _sample_raw_dataset().to_excel(dataset_path, index=False)

    output_file = tmp_path / "module_01" / "filtered.csv"
    filtered = run_full_preprocessing_pipeline(
        input_file=dataset_path,
        output_file=output_file,
        config={
            "duplicate_strategy": "error",
            "markets": [101],
            "grades": ["low1"],
            "date_range": (pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-06")),
        },
    )

    assert set(filtered["market_id"]) == {101}
    assert set(filtered["grade"]) == {"low1"}
    assert filtered["date"].min() >= pd.Timestamp("2024-01-03")
    assert filtered["date"].max() <= pd.Timestamp("2024-01-06")


def test_run_pipeline_module1_reruns_and_overwrites_by_default(tmp_path):
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    raw_dir.mkdir()

    dataset_path = raw_dir / "input.xlsx"
    _sample_raw_dataset().to_excel(dataset_path, index=False)

    command = [
        sys.executable,
        str(REPO_ROOT / "run_pipeline.py"),
        "--module",
        "1",
        "--input",
        str(raw_dir),
        "--output",
        str(output_dir),
    ]

    first = subprocess.run(command, cwd=REPO_ROOT, check=False, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr

    output_file = output_dir / "module_01" / "preprocessed_pilot_data.csv"
    assert output_file.exists()
    first_mtime = output_file.stat().st_mtime_ns
    first_df = pd.read_csv(output_file)

    second = subprocess.run(command, cwd=REPO_ROOT, check=False, capture_output=True, text=True)
    assert second.returncode == 0, second.stderr
    second_mtime = output_file.stat().st_mtime_ns
    second_df = pd.read_csv(output_file)

    assert second_mtime > first_mtime
    assert len(second_df) == len(first_df)
    assert "Skipping" not in second.stdout
