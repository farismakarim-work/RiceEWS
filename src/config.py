"""
RiceEWS - Rice Early Warning System
Main configuration and constants
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
for directory in [PROCESSED_DATA_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Constants
DEFAULT_DATE_FORMAT = "%Y-%m-%d"
DEFAULT_PRICE_COL = "price"
DEFAULT_MARKET_COL = "market"
DEFAULT_DATE_COL = "date"

# Analysis parameters
DEFAULT_MAX_LAG = 4
DEFAULT_SIGNIFICANCE_LEVEL = 0.05
DEFAULT_DIFFERENCING_ORDER = 1

# Column names expected in raw data
EXPECTED_COLUMNS = {
    'date': 'Date column',
    'market': 'Market identifier',
    'price': 'Rice price',
    'volume': 'Trade volume (optional)',
}

# Preprocessing defaults
PREPROCESSING_DEFAULTS = {
    'missing_method': 'interpolate',
    'outlier_method': 'iqr',
    'outlier_threshold': 1.5,
    'detrend_method': 'linear',
    'differencing_order': 1,
    'standardize_method': 'zscore',
}
