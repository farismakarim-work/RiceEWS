import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent / "src"))

from modules.causality_testing.granger_tester import run_full_granger_analysis
from modules.module_03_network_inference.network_builder import run_module3_network_inference
from modules.preprocessing.data_preprocessor import run_full_preprocessing_pipeline


REPO_ROOT = Path(__file__).resolve().parent
PILOT_DATASET = REPO_ROOT / "data" / "raw" / "Pilot Dataset.xlsx"


@pytest.fixture(scope="module")
def module3_outputs(tmp_path_factory):
    base_dir = tmp_path_factory.mktemp("module3")
    module1_dir = base_dir / "module_01"
    module2_dir = base_dir / "module_02"
    module3_dir = base_dir / "module_03"

    preprocessed_csv = module1_dir / "preprocessed_pilot_data.csv"
    run_full_preprocessing_pipeline(
        input_file=PILOT_DATASET,
        output_file=preprocessed_csv,
        config={"detrend_method": "none", "duplicate_strategy": "error"},
    )
    run_full_granger_analysis(
        input_file=str(preprocessed_csv),
        output_dir=str(module2_dir),
        config={"lag_order": 4, "price_col": "price_diff", "significance_level": 0.05},
    )
    results = run_module3_network_inference(
        granger_json_path=str(module2_dir / "granger_results.json"),
        output_dir=str(module3_dir),
    )
    return {"module3_dir": module3_dir, "results": results}


def test_module3_recovers_integrated_graph_with_metadata(module3_outputs):
    module3_dir = module3_outputs["module3_dir"]
    results = module3_outputs["results"]

    edges_path = module3_dir / "network_edges.csv"
    summary_path = module3_dir / "network_summary.csv"

    assert results["analysis_type"] == "integrated"
    assert edges_path.exists()
    assert summary_path.exists()
    assert (module3_dir / "network_inference_results.json").exists()
    assert (module3_dir / "network_summary.md").exists()
    assert (module3_dir / "network_graph_integrated.html").exists()
    assert (module3_dir / "market_leaders_integrated.csv").exists()

    edges_df = pd.read_csv(edges_path)
    summary_df = pd.read_csv(summary_path)

    required_columns = {
        "source",
        "target",
        "grade_source",
        "grade_target",
        "lag",
        "p_value",
        "adjusted_p_value",
        "test_statistic",
        "relationship_type",
        "direction",
    }
    assert required_columns.issubset(edges_df.columns)
    assert not edges_df.empty
    assert edges_df[["p_value", "adjusted_p_value", "test_statistic"]].notna().all().all()

    expected_grades = {"low1", "low2", "med1", "med2"}
    assert set(summary_df["grade"]) == expected_grades
    assert (summary_df["nodes"] > 0).all()
    assert summary_df["top_leader"].notna().all()

    for grade in expected_grades:
        assert (module3_dir / f"market_leaders_{grade}.csv").exists()
        assert (module3_dir / f"network_graph_{grade}.html").exists()


def test_module3_parses_legacy_string_booleans_and_recovers_cross_grade_edges(tmp_path):
    module2_json = tmp_path / "module_02" / "granger_results.json"
    module2_json.parent.mkdir(parents=True)

    synthetic_results = {
        "analysis_type": "integrated",
        "nodes": [
            {"node_id": "M101_low1", "market_id": 101, "grade": "low1"},
            {"node_id": "M102_low1", "market_id": 102, "grade": "low1"},
            {"node_id": "M103_low1", "market_id": 103, "grade": "low1"},
            {"node_id": "M101_med1", "market_id": 101, "grade": "med1"},
        ],
        "pairwise_tests": {
            "M101_low1→M102_low1": {
                "granger_causes": "True",
                "f_statistic": 9.0,
                "p_value": 0.01,
                "p_value_bh": 0.02,
                "lag_order": 2,
            },
            "M102_low1→M103_low1": {
                "granger_causes": "True",
                "f_statistic": 8.0,
                "p_value": 0.02,
                "p_value_bh": 0.03,
                "lag_order": 2,
            },
            "M101_low1→M103_low1": {
                "granger_causes": "True",
                "f_statistic": 7.0,
                "p_value": 0.03,
                "p_value_bh": 0.04,
                "lag_order": 2,
            },
            "M101_low1→M101_med1": {
                "granger_causes": "True",
                "f_statistic": 6.0,
                "p_value": 0.04,
                "p_value_bh": 0.05,
                "lag_order": 1,
            },
        },
    }
    module2_json.write_text(json.dumps(synthetic_results), encoding="utf-8")

    output_dir = tmp_path / "module_03"
    run_module3_network_inference(str(module2_json), str(output_dir))

    edges_df = pd.read_csv(output_dir / "network_edges.csv")
    edge_pairs = set(zip(edges_df["source_node"], edges_df["target_node"]))

    assert ("M101_low1", "M103_low1") not in edge_pairs
    assert ("M101_low1", "M101_med1") in edge_pairs
    assert ((edges_df["grade_source"] != edges_df["grade_target"]).any())


def test_module3_supports_market_grade_filters(tmp_path):
    module2_json = tmp_path / "module_02" / "granger_results.json"
    module2_json.parent.mkdir(parents=True)

    synthetic_results = {
        "analysis_type": "integrated",
        "nodes": [
            {"node_id": "M101_low1", "market_id": 101, "grade": "low1"},
            {"node_id": "M102_low1", "market_id": 102, "grade": "low1"},
            {"node_id": "M101_med1", "market_id": 101, "grade": "med1"},
        ],
        "pairwise_tests": {
            "M101_low1→M102_low1": {"granger_causes": True, "f_statistic": 4.0, "p_value": 0.01, "p_value_bh": 0.01, "lag_order": 1},
            "M101_low1→M101_med1": {"granger_causes": True, "f_statistic": 4.0, "p_value": 0.01, "p_value_bh": 0.01, "lag_order": 1},
        },
    }
    module2_json.write_text(json.dumps(synthetic_results), encoding="utf-8")

    output_dir = tmp_path / "module_03"
    result = run_module3_network_inference(
        str(module2_json),
        str(output_dir),
        markets=[101],
        grades=["low1"],
    )

    kept_nodes = {node["node_id"] for node in result["nodes"]}
    assert kept_nodes == {"M101_low1"}
