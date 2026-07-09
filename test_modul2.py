import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent / "src"))

from modules.causality_testing.granger_tester import run_full_granger_analysis
from modules.preprocessing.data_preprocessor import run_full_preprocessing_pipeline


REPO_ROOT = Path(__file__).resolve().parent
PILOT_DATASET = REPO_ROOT / "data" / "raw" / "Pilot Dataset.xlsx"


@pytest.fixture(scope="module")
def module2_outputs(tmp_path_factory):
    base_dir = tmp_path_factory.mktemp("module2")
    module1_dir = base_dir / "module_01"
    module2_dir = base_dir / "module_02"

    preprocessed_csv = module1_dir / "preprocessed_pilot_data.csv"
    run_full_preprocessing_pipeline(
        input_file=PILOT_DATASET,
        output_file=preprocessed_csv,
        config={"detrend_method": "none", "duplicate_strategy": "error"},
    )
    results = run_full_granger_analysis(
        input_file=str(preprocessed_csv),
        output_dir=str(module2_dir),
        config={"lag_order": 4, "price_col": "price_diff", "significance_level": 0.05},
    )
    return {"base_dir": base_dir, "module2_dir": module2_dir, "results": results}


def test_module2_produces_integrated_outputs_and_module02_paths(module2_outputs):
    module2_dir = module2_outputs["module2_dir"]
    results = module2_outputs["results"]

    assert results["analysis_type"] == "integrated"
    assert module2_dir.name == "module_02"
    assert (module2_dir / "granger_results.json").exists()
    assert (module2_dir / "granger_pairwise.csv").exists()
    assert (module2_dir / "granger_results.xlsx").exists()
    assert (module2_dir / "granger_results.md").exists()
    assert (module2_dir / "granger_matrices.npz").exists()
    assert (module2_dir / "network_graph_integrated.html").exists()
    assert (module2_dir / "heatmap_integrated.html").exists()
    assert (module2_dir / "ranking_integrated.html").exists()

    node_count = len(results["nodes"])
    matrix = np.array(results["pairwise_ancestor_matrix"])
    assert matrix.shape == (node_count, node_count)


def test_module2_serializes_granger_flags_as_booleans(module2_outputs):
    json_path = module2_outputs["module2_dir"] / "granger_results.json"
    pairwise_csv = module2_outputs["module2_dir"] / "granger_pairwise.csv"

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    flags = [result["granger_causes"] for result in data["pairwise_tests"].values()]
    assert flags
    assert all(isinstance(flag, bool) for flag in flags)

    df = pd.read_csv(pairwise_csv)
    assert {"grade_source", "grade_target", "source_node", "target_node"}.issubset(df.columns)
    assert ((df["grade_source"] != df["grade_target"]).any())
