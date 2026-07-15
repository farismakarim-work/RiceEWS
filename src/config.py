"""
RiceEWS centralized configuration.
All configurable runtime parameters should be defined in this file.
"""

from __future__ import annotations

from pathlib import Path


# =============================================================================
# Project
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

for _directory in (PROCESSED_DATA_DIR, RESULTS_DIR):
    _directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Input
# =============================================================================

RAW_INPUT_FILE_PATTERN = "*.xlsx"
DEFAULT_DATE_FORMAT = "%Y-%m-%d"
DEFAULT_PRICE_COL = "price"
DEFAULT_MARKET_COL = "market_id"
DEFAULT_DATE_COL = "date"
DEFAULT_GRADE_COL = "grade"

EXPECTED_COLUMNS = {
    "date": "Date column",
    "market_id": "Market identifier",
    "grade": "Rice grade",
    "price": "Rice price",
    "volume": "Trade volume (optional)",
}


# =============================================================================
# Output
# =============================================================================

MODULE_01_DIRNAME = "module_01"
MODULE_02_DIRNAME = "module_02"
MODULE_03_DIRNAME = "module_03"

MODULE_01_OUTPUT_FILENAME = "preprocessed_pilot_data.csv"
MODULE_01_REPORT_FILENAME = "preprocessed_pilot_data_report.json"
MODULE_02_MAIN_OUTPUT_FILENAME = "granger_results.json"
MODULE_03_MAIN_OUTPUT_FILENAME = "network_inference_results.json"

MODULE_01_OUTPUT_DIR = PROCESSED_DATA_DIR / MODULE_01_DIRNAME
MODULE_02_OUTPUT_DIR = PROCESSED_DATA_DIR / MODULE_02_DIRNAME
MODULE_03_OUTPUT_DIR = PROCESSED_DATA_DIR / MODULE_03_DIRNAME

MODULE_01_OUTPUT_PATH = MODULE_01_OUTPUT_DIR / MODULE_01_OUTPUT_FILENAME
MODULE_02_OUTPUT_PATH = MODULE_02_OUTPUT_DIR / MODULE_02_MAIN_OUTPUT_FILENAME
MODULE_03_OUTPUT_PATH = MODULE_03_OUTPUT_DIR / MODULE_03_MAIN_OUTPUT_FILENAME


# =============================================================================
# Execution
# =============================================================================

OVERWRITE_OUTPUTS = True
SKIP_EXISTING_OUTPUTS = False


# =============================================================================
# Module 1 (Preprocessing)
# =============================================================================

M1_VERBOSE = True

# Duplicate handling
M1_DUPLICATE_STRATEGY = "error"
M1_DUPLICATE_STRATEGY_OPTIONS = ("keep_first", "keep_last", "error")

# Missing values
M1_MISSING_VALUE_MODE = "disabled"
M1_MISSING_VALUE_OPTIONS = (
    "disabled",
    "drop",
    "forward_fill",
    "backward_fill",
    "linear_interpolation",
    "mean",
)

# Outliers
M1_OUTLIER_MODE = "iqr"
M1_OUTLIER_MODE_OPTIONS = ("disabled", "iqr", "zscore")
M1_OUTLIER_THRESHOLD = 1.5

# Transformations
M1_LOG_TRANSFORM = True
M1_DETREND_METHOD = "none"
M1_DETREND_METHOD_OPTIONS = ("none", "linear", "polynomial")

# Stationarity-driven workflow
M1_STATIONARITY_TEST = "ADF"
M1_STATIONARITY_TEST_OPTIONS = ("ADF", "KPSS", "ADF_KPSS")
M1_STATIONARITY_SIGNIFICANCE_LEVEL = 0.05
M1_REQUIRE_STATIONARITY = True
M1_MAX_DIFFERENCING_ORDER = 1
M1_DIFFERENCING_MODE = "AUTO"  # AUTO or MANUAL
M1_DIFFERENCING_MODE_OPTIONS = ("AUTO", "MANUAL")
M1_MANUAL_DIFFERENCING_ORDER = 1

# Standardization
M1_STANDARDIZATION_ENABLED = True
M1_STANDARDIZATION_METHOD = "zscore"
M1_STANDARDIZATION_METHOD_OPTIONS = ("none", "zscore", "minmax", "robust")

# Downstream column consumed by Module 2
M1_MODULE2_PRICE_COLUMN = "price_diff"


# =============================================================================
# Module 2 (Pairwise Granger)
# =============================================================================

M2_VERBOSE = True
M2_PRICE_COL = M1_MODULE2_PRICE_COLUMN

# Lag control: AUTO uses existing lag-selection implementation, MANUAL uses M2_LAG_ORDER
M2_LAG_SELECTION_MODE = "MANUAL"
M2_LAG_SELECTION_MODE_OPTIONS = ("AUTO", "MANUAL")
M2_LAG_ORDER = 4
M2_MAX_LAG = 8
M2_LAG_CRITERION = "bic"

# Significance control
M2_SIGNIFICANCE_MODE = "MANUAL"  # AUTO or MANUAL
M2_SIGNIFICANCE_MODE_OPTIONS = ("AUTO", "MANUAL")
M2_SIGNIFICANCE_LEVEL = 0.05
M2_AUTO_SIGNIFICANCE_LEVEL = 0.05

# Benjamini-Hochberg / FDR
M2_APPLY_BH_CORRECTION = True


# =============================================================================
# Module 3 (Network Recovery)
# =============================================================================

M3_ENFORCE_DAG = True


# =============================================================================
# Visualization
# =============================================================================

ENABLE_VISUALIZATIONS = True


# =============================================================================
# Logging
# =============================================================================

LOG_LEVEL_DEFAULT = "WARNING"
LOG_LEVEL_VERBOSE = "INFO"
LOG_LEVEL_DEBUG = "DEBUG"
LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
LOG_DATEFMT = "%H:%M:%S"


# =============================================================================
# CLI
# =============================================================================

DEFAULT_GRADES = None
DEFAULT_MARKETS = None
DEFAULT_DATE_RANGE = None
ALLOW_MARKET_FILTER = True
ALLOW_GRADE_FILTER = True
ALLOW_DATE_FILTER = True

CLI_DEFAULT_DUPLICATE_STRATEGY = M1_DUPLICATE_STRATEGY
CLI_DEFAULT_INPUT_DIR = RAW_DATA_DIR
CLI_DEFAULT_OUTPUT_DIR = PROCESSED_DATA_DIR


# =============================================================================
# Backward compatibility aliases
# =============================================================================

DEFAULT_MAX_LAG = M2_LAG_ORDER
DEFAULT_SIGNIFICANCE_LEVEL = M2_SIGNIFICANCE_LEVEL
DEFAULT_DIFFERENCING_ORDER = M1_MANUAL_DIFFERENCING_ORDER

PREPROCESSING_DEFAULTS = {
    "missing_method": M1_MISSING_VALUE_MODE,
    "outlier_method": M1_OUTLIER_MODE,
    "outlier_threshold": M1_OUTLIER_THRESHOLD,
    "detrend_method": M1_DETREND_METHOD,
    "differencing_order": M1_MANUAL_DIFFERENCING_ORDER,
    "standardize_method": M1_STANDARDIZATION_METHOD,
}
