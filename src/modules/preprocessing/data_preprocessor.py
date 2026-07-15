"""
MODUL 1: Data Preprocessing, Stationarity & Detrending
========================================================

Complete data preprocessing pipeline untuk pilot dataset rice market.
Includes: loading, cleaning, outlier removal, log transformation, detrending, 
differencing, stationarity testing.

Dataset structure: Date, Market_id, Grade, Price (41,016 records)
- 6 markets (101-106)
- 4 grades (low1, low2, med1, med2)
- Date range: 2019-04-01 to 2026-01-08

Features:
- Load and validate pilot dataset
- Handle missing values and outliers
- Log transformation untuk stabilize variance
- Detrending untuk remove trend component
- Differencing untuk achieve stationarity
- Standardization untuk normalize
- Stationarity testing (ADF, KPSS)
- Separate processing per grade
- Output: ready untuk Granger causality testing
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Sequence, Union
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
import logging
import warnings
import json

from config import (
    MODULE_01_STATIONARITY_REPORT_FILENAME,
    RAW_DATA_DIR,
    RAW_INPUT_FILE_PATTERN,
    M1_DIFFERENCING_MODE,
    M1_DIFFERENCING_MODE_OPTIONS,
    M1_DUPLICATE_STRATEGY,
    M1_LOG_TRANSFORM,
    M1_MAX_DIFFERENCING_ORDER,
    M1_MIN_OBSERVATIONS,
    M1_MANUAL_DIFFERENCING_ORDER,
    M1_MISSING_VALUE_MODE,
    M1_MODULE2_PRICE_COLUMN,
    M1_OUTLIER_MODE,
    M1_OUTLIER_THRESHOLD,
    M1_REQUIRE_STATIONARITY,
    M1_SKIP_ALL_NAN_SERIES,
    M1_SKIP_CONSTANT_SERIES,
    M1_SKIP_SHORT_SERIES,
    M1_SKIP_ZERO_VARIANCE_SERIES,
    M1_STANDARDIZATION_ENABLED,
    M1_STANDARDIZATION_METHOD,
    M1_STATIONARITY_SIGNIFICANCE_LEVEL,
    M1_STATIONARITY_TEST,
    M1_DETREND_METHOD,
    M1_VERBOSE,
    M1_VERBOSE_STATIONARITY,
)

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_NOT_TESTABLE = "NOT TESTABLE"


def _normalize_input_files(
    input_file: Optional[Union[str, Path, Sequence[Union[str, Path]]]]
) -> List[Path]:
    """Normalize supported input_file values into a list of Excel paths."""

    if input_file is None:
        files = sorted(RAW_DATA_DIR.glob(RAW_INPUT_FILE_PATTERN))
        if not files:
            raise FileNotFoundError(
                f"No Excel datasets found in raw data directory: {RAW_DATA_DIR}"
            )
        return files

    if isinstance(input_file, (str, Path)):
        return [Path(input_file)]

    if isinstance(input_file, Sequence):
        files = [Path(path) for path in input_file]
        if not files:
            raise ValueError("input_file list is empty")
        return files

    raise TypeError(
        "input_file must be None, a path-like value, or a list of path-like values"
    )


class DataPreprocessor:
    """
    Complete preprocessing pipeline untuk rice market time series data.
    Designed untuk pilot dataset structure with multiple grades and markets.
    """
    
    def __init__(self, verbose: bool = M1_VERBOSE):
        """
        Initialize preprocessor.
        
        Parameters:
        -----------
        verbose : bool
            Print detailed logs
        """
        self.verbose = verbose
        self.logger = logger
        self.processing_report = {}
        self.min_observations = int(M1_MIN_OBSERVATIONS)
        self.skip_short_series = bool(M1_SKIP_SHORT_SERIES)
        self.skip_constant_series = bool(M1_SKIP_CONSTANT_SERIES)
        self.skip_zero_variance_series = bool(M1_SKIP_ZERO_VARIANCE_SERIES)
        self.skip_all_nan_series = bool(M1_SKIP_ALL_NAN_SERIES)
        self.verbose_stationarity = bool(M1_VERBOSE_STATIONARITY)
    
    def load_pilot_data(
        self,
        filepath: Optional[Union[str, Path, Sequence[Union[str, Path]]]],
        duplicate_strategy: str = 'error',
    ) -> pd.DataFrame:
        """
        Load satu atau banyak dataset Excel mentah.
        
        Expected format:
        - Date (datetime)
        - Market_id (int: 101-106)
        - Grade (str: low1, low2, med1, med2)
        - Price (int)
        
        Parameters:
        -----------
        filepath : None | str | Path | list[str] | list[Path]
            Jika None, semua file ``*.xlsx`` di ``data/raw`` akan dimuat.
        duplicate_strategy : {'keep_first', 'keep_last', 'error'}
            Strategi saat ditemukan duplikasi ``(date, market_id, grade)``
            lintas file input.
            
        Returns:
        --------
        pd.DataFrame
            Loaded data with standardized column names
        """
        
        try:
            if duplicate_strategy not in {'keep_first', 'keep_last', 'error'}:
                raise ValueError(
                    "duplicate_strategy must be one of: 'keep_first', 'keep_last', 'error'"
                )

            filepaths = _normalize_input_files(filepath)
            frames: List[pd.DataFrame] = []
            expected_columns: Optional[List[str]] = None

            for file_index, path in enumerate(filepaths):
                if not path.exists():
                    raise FileNotFoundError(f"File not found: {path}")

                frame = pd.read_excel(path)
                raw_columns = frame.columns.tolist()
                frame.columns = frame.columns.str.lower()

                if expected_columns is None:
                    expected_columns = frame.columns.tolist()
                elif frame.columns.tolist() != expected_columns:
                    raise ValueError(
                        "Input datasets must have identical schemas. "
                        f"Expected columns {expected_columns} from {filepaths[0].name}, "
                        f"but found {frame.columns.tolist()} in {path.name}."
                    )

                required_cols = ['date', 'market_id', 'grade', 'price']
                missing_cols = [col for col in required_cols if col not in frame.columns]
                if missing_cols:
                    raise ValueError(
                        f"Missing required columns in {path.name}: {missing_cols}\n"
                        f"Found: {frame.columns.tolist()}"
                    )

                frame['__source_file'] = path.name
                frame['__source_file_order'] = file_index
                frame['__source_row_order'] = np.arange(len(frame))
                frames.append(frame)

                if self.verbose:
                    print(f"\n{'='*70}")
                    print("DATA LOADING")
                    print(f"{'='*70}")
                    print(f"File: {path.name}")
                    print(f"Shape: {frame.shape}")
                    print(f"Columns (raw): {raw_columns}")

            df = pd.concat(frames, ignore_index=True)

            if self.verbose:
                input_summary = ", ".join(path.name for path in filepaths)
                print(f"Shape: {df.shape}")
                print(f"Files: {input_summary}")

            df['date'] = pd.to_datetime(df['date'])
            df['market_id'] = pd.to_numeric(df['market_id'])
            df['price'] = pd.to_numeric(df['price'])
            df['grade'] = df['grade'].astype(str)

            duplicate_keys = ['date', 'market_id', 'grade']
            duplicate_mask = df.duplicated(subset=duplicate_keys, keep=False)
            duplicate_count = int(duplicate_mask.sum())
            duplicate_records = int(df.loc[duplicate_mask, duplicate_keys].drop_duplicates().shape[0])
            if duplicate_count:
                duplicate_preview = (
                    df.loc[duplicate_mask, duplicate_keys + ['__source_file']]
                    .sort_values(duplicate_keys + ['__source_file'])
                    .head(10)
                    .to_dict(orient='records')
                )
                if duplicate_strategy == 'error':
                    raise ValueError(
                        "Duplicate (date, market_id, grade) rows detected across input datasets. "
                        f"Found {duplicate_records} duplicated keys. "
                        f"Set duplicate_strategy to 'keep_first' or 'keep_last' to resolve. "
                        f"Examples: {duplicate_preview}"
                    )
                keep = 'first' if duplicate_strategy == 'keep_first' else 'last'
                df = df.drop_duplicates(subset=duplicate_keys, keep=keep).copy()
                if self.verbose:
                    print(
                        f"Resolved {duplicate_records} duplicate keys using duplicate_strategy='{duplicate_strategy}'."
                    )

            df = df.sort_values(['date', 'market_id', 'grade']).reset_index(drop=True)
            df = df.drop(columns=['__source_file', '__source_file_order', '__source_row_order'])
            
            if self.verbose:
                print(f"\nData Info:")
                print(f"  Records: {len(df):,}")
                print(f"  Markets: {sorted(df['market_id'].unique().tolist())}")
                print(f"  Grades: {sorted(df['grade'].unique().tolist())}")
                print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
                print(f"  Price range: {df['price'].min():,} to {df['price'].max():,}")
                print(f"\n  Records per grade:")
                for grade in sorted(df['grade'].unique()):
                    count = len(df[df['grade'] == grade])
                    print(f"    {grade}: {count:,}")
            
            self.processing_report['loaded'] = {
                'records': len(df),
                'input_files': [str(path) for path in filepaths],
                'markets': sorted(df['market_id'].unique().tolist()),
                'grades': sorted(df['grade'].unique().tolist()),
                'date_range': (str(df['date'].min().date()), str(df['date'].max().date())),
                'price_range': (int(df['price'].min()), int(df['price'].max())),
                'duplicate_strategy': duplicate_strategy,
                'duplicate_key_count': duplicate_records
            }
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def handle_missing_values(self,
                             df: pd.DataFrame,
                             price_col: str = 'price',
                             market_col: str = 'market_id',
                             grade_col: str = 'grade',
                             date_col: str = 'date',
                             method: str = M1_MISSING_VALUE_MODE) -> pd.DataFrame:
        """
        Handle missing values dalam time series per market per grade.
        
        Parameters:
        -----------
        df : pd.DataFrame
        price_col : str
        market_col : str
        grade_col : str
        date_col : str
        method : str
            'drop' - Remove rows with missing price
            'interpolate' - Linear interpolation (default)
            'forward_fill' - Forward fill
            'mean' - Fill dengan market-grade mean
            
        Returns:
        --------
        pd.DataFrame
        """
        
        df = df.copy()
        missing_before = df[price_col].isnull().sum()
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"HANDLING MISSING VALUES (method: {method})")
            print(f"{'='*70}")
            print(f"Missing values before: {missing_before:,}")
        
        if method == 'disabled':
            if self.verbose:
                print("Missing value handling is disabled by configuration")
            self.processing_report['missing_values'] = {
                'before': int(missing_before),
                'after': int(missing_before),
                'method': method
            }
            return df
        elif method == 'drop':
            df = df.dropna(subset=[price_col])

        elif method in {'interpolate', 'linear_interpolation'}:
            # Interpolate per market-grade combination
            for market in df[market_col].unique():
                for grade in df[grade_col].unique():
                    mask = (df[market_col] == market) & (df[grade_col] == grade)
                    if mask.any():
                        df.loc[mask, price_col] = (
                            df.loc[mask, price_col]
                            .interpolate(method='linear', limit_direction='both')
                        )
        
        elif method == 'forward_fill':
            for market in df[market_col].unique():
                for grade in df[grade_col].unique():
                    mask = (df[market_col] == market) & (df[grade_col] == grade)
                    if mask.any():
                        df.loc[mask, price_col] = (
                            df.loc[mask, price_col]
                            .ffill()
                            .bfill()
                        )
        elif method == 'backward_fill':
            for market in df[market_col].unique():
                for grade in df[grade_col].unique():
                    mask = (df[market_col] == market) & (df[grade_col] == grade)
                    if mask.any():
                        df.loc[mask, price_col] = (
                            df.loc[mask, price_col]
                            .bfill()
                            .ffill()
                        )

        elif method == 'mean':
            for market in df[market_col].unique():
                for grade in df[grade_col].unique():
                    mask = (df[market_col] == market) & (df[grade_col] == grade)
                    if mask.any():
                        market_grade_mean = df.loc[mask, price_col].mean()
                        df.loc[mask, price_col] = (
                            df.loc[mask, price_col].fillna(market_grade_mean)
                        )
        else:
            raise ValueError(f"Unsupported missing-value mode: {method}")
        
        missing_after = df[price_col].isnull().sum()
        
        if self.verbose:
            print(f"Missing values after: {missing_after:,}")
            print(f"✓ Handled {missing_before - missing_after:,} missing values")
        
        self.processing_report['missing_values'] = {
            'before': int(missing_before),
            'after': int(missing_after),
            'method': method
        }
        
        return df
    
    def remove_outliers(self,
                       df: pd.DataFrame,
                       price_col: str = 'price',
                       market_col: str = 'market_id',
                       grade_col: str = 'grade',
                       method: str = M1_OUTLIER_MODE,
                       threshold: float = M1_OUTLIER_THRESHOLD) -> pd.DataFrame:
        """
        Remove outliers per market-grade combination.
        
        Parameters:
        -----------
        df : pd.DataFrame
        price_col : str
        market_col : str
        grade_col : str
        method : str
            'iqr' - Interquartile range (default)
            'zscore' - Z-score method
        threshold : float
            IQR threshold (default 1.5) or Z-score threshold
            
        Returns:
        --------
        pd.DataFrame
        """
        
        df = df.copy()
        outliers_replaced = 0
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"OUTLIER DETECTION & REMOVAL (method: {method}, threshold: {threshold})")
            print(f"{'='*70}")

        if method == 'disabled':
            self.processing_report['outliers'] = {
                'method': method,
                'threshold': threshold,
                'total_replaced': 0
            }
            return df
        
        for market in df[market_col].unique():
            for grade in df[grade_col].unique():
                mask = (df[market_col] == market) & (df[grade_col] == grade)
                if not mask.any():
                    continue
                
                prices = df.loc[mask, price_col].copy()
                
                if method == 'iqr':
                    Q1 = prices.quantile(0.25)
                    Q3 = prices.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    
                    outlier_mask = (prices < lower_bound) | (prices > upper_bound)
                
                elif method == 'zscore':
                    valid_prices = prices.dropna()
                    if len(valid_prices) > 1:
                        z_scores = np.abs(stats.zscore(valid_prices))
                        outlier_indices = np.where(z_scores > threshold)[0]
                        outlier_mask = pd.Series(False, index=prices.index)
                        for idx in outlier_indices:
                            outlier_mask.iloc[idx] = True
                    else:
                        outlier_mask = pd.Series(False, index=prices.index)
                
                # Replace outliers with market-grade median
                if outlier_mask.any():
                    median = prices.median()
                    count = outlier_mask.sum()
                    df.loc[mask, price_col] = (
                        df.loc[mask, price_col].mask(outlier_mask, median)
                    )
                    outliers_replaced += count
        
        if self.verbose:
            print(f"✓ Total outliers replaced: {outliers_replaced:,}")
        
        self.processing_report['outliers'] = {
            'method': method,
            'threshold': threshold,
            'total_replaced': int(outliers_replaced)
        }
        
        return df
    
    def apply_log_transformation(self,
                                df: pd.DataFrame,
                                price_col: str = 'price',
                                market_col: str = 'market_id',
                                grade_col: str = 'grade',
                                apply: bool = M1_LOG_TRANSFORM) -> pd.DataFrame:
        """
        Apply natural log transformation untuk stabilize variance.
        
        Log transformation converts multiplicative errors ke additive errors,
        important untuk financial/commodity price data.
        
        Formula: price_log = ln(price)
        
        After differencing: log_return = ln(price_t) - ln(price_t-1)
        
        Parameters:
        -----------
        df : pd.DataFrame
        price_col : str
            Original price column
        market_col : str
        grade_col : str
        apply : bool
            If False, skip log transformation
            
        Returns:
        --------
        pd.DataFrame dengan kolom 'price_log'
        """
        
        df = df.copy()
        
        if not apply:
            df['price_log'] = df[price_col].copy()
            if self.verbose:
                print(f"\n{'='*70}")
                print(f"LOG TRANSFORMATION (SKIPPED)")
                print(f"{'='*70}")
            return df
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"LOG TRANSFORMATION")
            print(f"{'='*70}")
            print(f"Applying: price_log = ln(price)")
        
        # Check for non-positive values
        if (df[price_col] <= 0).any():
            self.logger.warning("Found non-positive prices, cannot apply log transformation")
            df['price_log'] = df[price_col].copy()
            return df
        
        # Apply log transformation
        df['price_log'] = np.log(df[price_col])
        
        if self.verbose:
            print(f"✓ Log transformation applied to all records")
            print(f"  Original range: {df[price_col].min():.2f} to {df[price_col].max():.2f}")
            print(f"  Log range: {df['price_log'].min():.4f} to {df['price_log'].max():.4f}")
        
        self.processing_report['log_transformation'] = {
            'applied': True,
            'original_range': (float(df[price_col].min()), float(df[price_col].max())),
            'log_range': (float(df['price_log'].min()), float(df['price_log'].max()))
        }
        
        return df
    
    def detrend_data(self,
                    df: pd.DataFrame,
                    price_col: str = 'price_log',
                    market_col: str = 'market_id',
                    grade_col: str = 'grade',
                    date_col: str = 'date',
                    method: str = M1_DETREND_METHOD) -> pd.DataFrame:
        """
        Remove trend component per market-grade combination.
        
        Parameters:
        -----------
        df : pd.DataFrame
        price_col : str
            Column to detrend (usually 'price_log' after log transformation)
        market_col : str
        grade_col : str
        date_col : str
        method : str
            'linear' - Linear detrending (default)
            'polynomial' - Polynomial (order 3)
            'none' - No detrending
            
        Returns:
        --------
        pd.DataFrame dengan kolom 'price_detrended'
        """
        
        df = df.copy()
        
        if method == 'none':
            df['price_detrended'] = df[price_col]
            return df
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"DETRENDING (method: {method})")
            print(f"{'='*70}")
        
        df['price_detrended'] = np.nan
        
        for market in df[market_col].unique():
            for grade in df[grade_col].unique():
                mask = (df[market_col] == market) & (df[grade_col] == grade)
                if not mask.any():
                    continue
                
                market_grade_data = df.loc[mask].copy()
                
                if len(market_grade_data) < 3:
                    df.loc[mask, 'price_detrended'] = market_grade_data[price_col]
                    continue
                
                indices = np.arange(len(market_grade_data))
                prices = market_grade_data[price_col].values
                
                if method == 'linear':
                    # Fit linear trend: y = a*t + b
                    z = np.polyfit(indices, prices, 1)
                    p = np.poly1d(z)
                    trend = p(indices)
                    detrended = prices - trend
                
                elif method == 'polynomial':
                    # Fit polynomial trend (order 3)
                    z = np.polyfit(indices, prices, 3)
                    p = np.poly1d(z)
                    trend = p(indices)
                    detrended = prices - trend
                
                df.loc[mask, 'price_detrended'] = detrended
        
        if self.verbose:
            num_combinations = len(df[[market_col, grade_col]].drop_duplicates())
            print(f"✓ Detrended {num_combinations} market-grade combinations using {method} method")
        
        self.processing_report['detrending'] = {
            'method': method,
            'combinations_processed': len(df[[market_col, grade_col]].drop_duplicates())
        }
        
        return df
    
    def apply_differencing(self,
                          df: pd.DataFrame,
                          price_col: str = 'price_detrended',
                          market_col: str = 'market_id',
                          grade_col: str = 'grade',
                          order: int = 1) -> pd.DataFrame:
        """
        Apply differencing untuk achieve stationarity.
        
        After log transformation, differencing creates log returns:
            log_return = log_price(t) - log_price(t-1)
        
        Parameters:
        -----------
        df : pd.DataFrame
        price_col : str
            Column to difference
        market_col : str
        grade_col : str
        order : int
            Order of differencing (1 or 2)
            
        Returns:
        --------
        pd.DataFrame dengan kolom 'price_diff' (dan optional 'price_diff2')
        """
        
        df = df.copy()
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"DIFFERENCING (order: {order})")
            print(f"{'='*70}")
        
        df['price_diff'] = np.nan
        if order >= 2:
            df['price_diff2'] = np.nan
        
        for market in df[market_col].unique():
            for grade in df[grade_col].unique():
                mask = (df[market_col] == market) & (df[grade_col] == grade)
                if not mask.any():
                    continue
                
                prices = df.loc[mask, price_col].values
                
                # First differencing
                if order >= 1:
                    diff1 = np.diff(prices, n=1)
                    diff1_padded = np.concatenate([[np.nan], diff1])
                    df.loc[mask, 'price_diff'] = diff1_padded[:len(prices)]
                
                # Second differencing
                if order >= 2 and len(prices) > 2:
                    diff2 = np.diff(prices, n=2)
                    diff2_padded = np.concatenate([[np.nan, np.nan], diff2])
                    df.loc[mask, 'price_diff2'] = diff2_padded[:len(prices)]
        
        if self.verbose:
            print(f"✓ Applied {order}-order differencing")
            print(f"  Note: First {order} observations per market-grade will be NaN")
        
        self.processing_report['differencing'] = {
            'order': order,
            'output_columns': ['price_diff'] if order == 1 else ['price_diff', 'price_diff2']
        }
        
        return df
    
    def standardize_data(self,
                        df: pd.DataFrame,
                        price_col: str = 'price_diff',
                        market_col: str = 'market_id',
                        grade_col: str = 'grade',
                        method: str = 'zscore') -> pd.DataFrame:
        """
        Standardize/normalize data per market-grade.
        
        Parameters:
        -----------
        df : pd.DataFrame
        price_col : str
        market_col : str
        grade_col : str
        method : str
            'zscore' - Z-score normalization
            'minmax' - Min-Max scaling (0-1)
            'robust' - Robust scaling (median, IQR)
            'none' - No standardization
            
        Returns:
        --------
        pd.DataFrame dengan kolom 'price_standardized'
        """
        
        df = df.copy()
        
        if method == 'none':
            df['price_standardized'] = df[price_col]
            return df
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"STANDARDIZATION (method: {method})")
            print(f"{'='*70}")
        
        df['price_standardized'] = np.nan
        
        for market in df[market_col].unique():
            for grade in df[grade_col].unique():
                mask = (df[market_col] == market) & (df[grade_col] == grade)
                if not mask.any():
                    continue
                
                prices = df.loc[mask, price_col].dropna().values
                
                if len(prices) < 2:
                    continue
                
                if method == 'zscore':
                    mean = np.nanmean(prices)
                    std = np.nanstd(prices)
                    standardized = (prices - mean) / std if std != 0 else prices
                
                elif method == 'minmax':
                    min_val = np.nanmin(prices)
                    max_val = np.nanmax(prices)
                    standardized = (prices - min_val) / (max_val - min_val) \
                                  if max_val != min_val else prices
                
                elif method == 'robust':
                    median = np.nanmedian(prices)
                    Q1 = np.nanpercentile(prices, 25)
                    Q3 = np.nanpercentile(prices, 75)
                    IQR = Q3 - Q1
                    standardized = (prices - median) / IQR if IQR != 0 else prices
                
                # Align with original indices
                original_indices = df.loc[mask].index
                valid_mask = df.loc[mask, price_col].notna()
                df.loc[original_indices[valid_mask], 'price_standardized'] = standardized
        
        if self.verbose:
            print(f"✓ Standardized using {method} method")
        
        self.processing_report['standardization'] = {
            'method': method
        }
        
        return df
    
    def filter_data(self,
                    df: pd.DataFrame,
                    markets: Optional[List[int]] = None,
                    grades: Optional[List[str]] = None,
                    date_range: Optional[Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]] = None) -> pd.DataFrame:
        """Filter loaded data before downstream preprocessing."""
        filtered = df.copy()
        initial_count = len(filtered)
        start_date = pd.Timestamp(date_range[0]) if date_range and date_range[0] is not None else None
        end_date = pd.Timestamp(date_range[1]) if date_range and date_range[1] is not None else None

        if markets:
            market_set = {int(market) for market in markets}
            filtered = filtered[filtered['market_id'].isin(market_set)]
        if grades:
            grade_set = {str(grade) for grade in grades}
            filtered = filtered[filtered['grade'].isin(grade_set)]
        if start_date is not None:
            filtered = filtered[filtered['date'] >= start_date]
        if end_date is not None:
            filtered = filtered[filtered['date'] <= end_date]

        filtered = filtered.sort_values(['date', 'market_id', 'grade']).reset_index(drop=True)
        self.processing_report['filters'] = {
            'markets': markets or 'all',
            'grades': grades or 'all',
            'date_range': (
                str(start_date.date()) if start_date is not None else None,
                str(end_date.date()) if end_date is not None else None,
            ),
            'records_before': int(initial_count),
            'records_after': int(len(filtered)),
        }
        return filtered

    def _validate_stationarity_input(self, prices: np.ndarray) -> Dict:
        """Validate one series before stationarity tests.

        Returns keys:
        ``clean_values``, ``observations``, ``unique_values``, ``variance``,
        ``is_testable``, and ``reason``.
        """

        clean_values = np.asarray(prices, dtype=float)
        clean_values = clean_values[~np.isnan(clean_values)]
        observations = int(clean_values.size)
        unique_values = int(np.unique(clean_values).size) if observations > 0 else 0
        variance = float(np.var(clean_values)) if observations > 0 else np.nan

        reason: Optional[str] = None
        if observations == 0 and self.skip_all_nan_series:
            reason = "All NaN"
        elif observations < int(self.min_observations) and self.skip_short_series:
            reason = "Insufficient observations"
        elif unique_values <= 1 and self.skip_constant_series:
            reason = "Constant series"
        elif variance <= 0 and self.skip_zero_variance_series:
            reason = "Zero variance"

        return {
            'clean_values': clean_values,
            'observations': observations,
            'unique_values': unique_values,
            'variance': variance,
            'is_testable': reason is None,
            'reason': reason,
        }

    def _evaluate_stationarity_series(self,
                                      prices: np.ndarray,
                                      test: str = M1_STATIONARITY_TEST,
                                      significance_level: float = M1_STATIONARITY_SIGNIFICANCE_LEVEL) -> Dict:
        test_name = str(test).upper()
        if test_name not in {'ADF', 'KPSS', 'ADF_KPSS'}:
            raise ValueError(f"Unsupported stationarity test: {test}")

        validation = self._validate_stationarity_input(prices)
        outcome: Dict = {
            'test': test_name,
            'observations': int(validation['observations']),
            'unique_values': int(validation['unique_values']),
            'variance': float(validation['variance']) if validation['observations'] > 0 else np.nan,
            'adf_p_value': None,
            'p_value': None,
            'reason': None,
        }

        if not validation['is_testable']:
            outcome['status'] = STATUS_NOT_TESTABLE
            outcome['stationary'] = False
            outcome['eligible_for_pwgc'] = False
            outcome['reason'] = validation['reason']
            outcome['significance_level'] = float(significance_level)
            return outcome

        clean_values = validation['clean_values']
        try:
            if test_name in {'ADF', 'ADF_KPSS'}:
                adf_stat, adf_p, *_ = adfuller(clean_values, autolag='AIC')
                outcome['adf_statistic'] = float(adf_stat)
                outcome['adf_p_value'] = float(adf_p)
                outcome['p_value'] = float(adf_p)
                adf_stationary = adf_p < significance_level
            else:
                adf_stationary = True

            if test_name in {'KPSS', 'ADF_KPSS'}:
                kpss_stat, kpss_p, *_ = kpss(clean_values, regression='c', nlags='auto')
                outcome['kpss_statistic'] = float(kpss_stat)
                outcome['kpss_p_value'] = float(kpss_p)
                if outcome['p_value'] is None:
                    outcome['p_value'] = float(kpss_p)
                kpss_stationary = kpss_p > significance_level
            else:
                kpss_stationary = True
        except Exception as exc:  # noqa: BLE001
            self.logger.debug("Stationarity estimation failed for %s", test_name, exc_info=True)
            outcome['status'] = STATUS_NOT_TESTABLE
            outcome['stationary'] = False
            outcome['eligible_for_pwgc'] = False
            outcome['reason'] = f"{test_name} could not be estimated: {type(exc).__name__}: {exc}"
            outcome['significance_level'] = float(significance_level)
            return outcome

        stationary = bool(adf_stationary and kpss_stationary)
        outcome['stationary'] = stationary
        outcome['status'] = STATUS_PASS if stationary else STATUS_FAIL
        outcome['eligible_for_pwgc'] = True
        outcome['reason'] = ""
        outcome['significance_level'] = float(significance_level)
        return outcome

    def check_stationarity(self,
                          df: pd.DataFrame,
                          price_col: str = 'price_diff',
                          market_col: str = 'market_id',
                          grade_col: str = 'grade',
                          test: str = 'ADF',
                          significance_level: float = 0.05) -> Dict:
        """
        Check stationarity per market-grade menggunakan ADF atau KPSS test.
        
        Parameters:
        -----------
        df : pd.DataFrame
        price_col : str
        market_col : str
        grade_col : str
        test : str
            'adf' - Augmented Dickey-Fuller
            'kpss' - KPSS test
            
        Returns:
        --------
        Dict dengan hasil per market-grade
        """
        
        test_name = str(test).upper()
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"STATIONARITY TESTING ({test_name} test)")
            print(f"{'='*70}")
        
        results = {}
        status_counts = {
            STATUS_PASS: 0,
            STATUS_FAIL: 0,
            STATUS_NOT_TESTABLE: 0,
        }

        for market in sorted(df[market_col].unique()):
            for grade in sorted(df[grade_col].unique()):
                mask = (df[market_col] == market) & (df[grade_col] == grade)
                prices = df.loc[mask, price_col].dropna().values
                
                key = f"M{market}_{grade}"
                
                stationarity = self._evaluate_stationarity_series(
                    prices,
                    test=test_name,
                    significance_level=significance_level,
                )
                results[key] = {
                    'series': key,
                    'market_id': int(market),
                    'grade': str(grade),
                    **stationarity,
                }
                status = stationarity.get('status', STATUS_FAIL)
                status_counts[status] = status_counts.get(status, 0) + 1

                if self.verbose and self.verbose_stationarity:
                    p_value = stationarity.get('adf_p_value')
                    reason = stationarity.get('reason') or "-"
                    if p_value is None or pd.isna(p_value):
                        print(f"  {key}: {status} | eligible={stationarity.get('eligible_for_pwgc')} | reason={reason}")
                    else:
                        print(
                            f"  {key}: {status} | eligible={stationarity.get('eligible_for_pwgc')} "
                            f"| adf_p={p_value:.4f} | reason={reason}"
                        )

        if self.verbose:
            total_tests = len(results)
            print(
                f"\nPASS: {status_counts[STATUS_PASS]}/{total_tests} | "
                f"FAIL: {status_counts[STATUS_FAIL]} | "
                f"NOT TESTABLE: {status_counts[STATUS_NOT_TESTABLE]}"
            )

        self.processing_report['stationarity_test'] = {
            'test': test_name,
            'pass_count': int(status_counts[STATUS_PASS]),
            'fail_count': int(status_counts[STATUS_FAIL]),
            'not_testable_count': int(status_counts[STATUS_NOT_TESTABLE]),
            'total_tests': len(results),
            'significance_level': float(significance_level),
        }
        
        return results
    
    def get_preprocessing_report(self, df: pd.DataFrame) -> Dict:
        """Generate complete preprocessing summary report."""
        
        report = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'data_types': {str(k): str(v) for k, v in df.dtypes.to_dict().items()},
            'null_counts': df.isnull().sum().to_dict(),
            'processing_steps': self.processing_report
        }
        
        return report
    
    def save_processed_data(self, 
                           df: pd.DataFrame, 
                           output_path: str,
                           output_format: str = 'csv',
                           include_report: bool = True) -> Path:
        """
        Save processed data dan report.
        
        Parameters:
        -----------
        df : pd.DataFrame
        output_path : str
            Path untuk output file
        output_format : str
            'csv' atau 'excel'
        include_report : bool
            Save processing report as JSON
            
        Returns:
        --------
        Path
        """
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save data
        if output_format == 'csv':
            df.to_csv(output_path, index=False)
            ext = '.csv'
        elif output_format == 'excel':
            df.to_excel(output_path, index=False)
            ext = '.xlsx'
        
        if self.verbose:
            print(f"\n✓ Saved processed data to: {output_path}")
        
        # Save report
        if include_report:
            report_path = output_path.parent / f"{output_path.stem}_report.json"
            report = self.get_preprocessing_report(df)
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            if self.verbose:
                print(f"✓ Saved processing report to: {report_path}")
        
        return output_path


def _format_summary_table(rows: List[Tuple[str, Union[int, str]]]) -> str:
    """Format key-value rows into an ASCII table."""
    if not rows:
        rows = [("No data available", "N/A")]
    label_width = max(len(label) for label, _ in rows)
    value_width = max(len(str(value)) for _, value in rows)
    border = f"+-{'-' * label_width}-+-{'-' * value_width}-+"
    lines = [border]
    for label, value in rows:
        lines.append(f"| {label:<{label_width}} | {str(value):>{value_width}} |")
    lines.append(border)
    return "\n".join(lines)


def _build_stationarity_report_dataframe(stationarity_results: Dict[str, Dict]) -> pd.DataFrame:
    """Build the stationarity report DataFrame from per-series result dictionaries."""
    rows: List[Dict] = []
    for series_key in sorted(stationarity_results.keys()):
        result = stationarity_results[series_key]
        rows.append(
            {
                'Series': series_key,
                'Market': int(result.get('market_id')),
                'Grade': str(result.get('grade')),
                'Observations': int(result.get('observations', 0)),
                'Unique Values': int(result.get('unique_values', 0)),
                'Variance': float(result.get('variance')) if result.get('observations', 0) > 0 else np.nan,
                'Status': result.get('status', STATUS_NOT_TESTABLE),
                'Eligible for PWGC': 'YES' if result.get('eligible_for_pwgc') else 'NO',
                'ADF p-value': result.get('adf_p_value'),
                'Reason': result.get('reason', ''),
            }
        )
    return pd.DataFrame(rows)


def _print_module1_summary(stationarity_report_df: pd.DataFrame) -> None:
    """Print the Module 1 stationarity and PWGC eligibility summary tables."""
    total_series = int(len(stationarity_report_df))
    eligible_count = int((stationarity_report_df['Eligible for PWGC'] == 'YES').sum())
    not_eligible_count = int((stationarity_report_df['Eligible for PWGC'] == 'NO').sum())
    status_counts = stationarity_report_df['Status'].value_counts()
    not_testable_reasons = (
        stationarity_report_df[stationarity_report_df['Status'] == STATUS_NOT_TESTABLE]['Reason']
        .replace('', 'No reason provided')
        .value_counts()
    )

    print("\n" + "=" * 70)
    print("MODULE 1 SUMMARY")
    print("=" * 70)
    print(
        _format_summary_table(
            [
                ("Total Time Series", total_series),
                ("Successfully Processed", eligible_count),
                ("Eligible for PWGC", eligible_count),
                ("Not Eligible", not_eligible_count),
            ]
        )
    )
    print("\nStationarity Results")
    print(
        _format_summary_table(
            [
                (STATUS_PASS, int(status_counts.get(STATUS_PASS, 0))),
                (STATUS_FAIL, int(status_counts.get(STATUS_FAIL, 0))),
                (STATUS_NOT_TESTABLE, int(status_counts.get(STATUS_NOT_TESTABLE, 0))),
            ]
        )
    )
    if not not_testable_reasons.empty:
        print("\nReasons for NOT TESTABLE")
        print(_format_summary_table([(reason, int(count)) for reason, count in not_testable_reasons.items()]))
    print("=" * 70)


def _filter_pwgc_eligible_data(
    df: pd.DataFrame,
    stationarity_report_df: pd.DataFrame,
    market_col: str = 'market_id',
    grade_col: str = 'grade',
) -> pd.DataFrame:
    """Keep only market-grade series marked as PWGC-eligible in the report."""
    eligible_pairs = stationarity_report_df[stationarity_report_df['Eligible for PWGC'] == 'YES'][['Market', 'Grade']]
    if eligible_pairs.empty:
        return df.head(0).copy()
    eligible_pairs = eligible_pairs.rename(columns={'Market': market_col, 'Grade': grade_col})
    eligible_pairs[market_col] = eligible_pairs[market_col].astype(int)
    eligible_pairs[grade_col] = eligible_pairs[grade_col].astype(str)
    filtered = df.merge(eligible_pairs.drop_duplicates(), on=[market_col, grade_col], how='inner')
    return filtered.sort_values(['date', market_col, grade_col]).reset_index(drop=True)


def _apply_stationarity_driven_differencing(
    df: pd.DataFrame,
    preprocessor: DataPreprocessor,
    stationarity_test: str,
    stationarity_alpha: float,
    differencing_mode: str,
    manual_differencing_order: int,
    max_differencing_order: int,
    require_stationarity: bool,
    source_col: str = 'price_detrended',
    market_col: str = 'market_id',
    grade_col: str = 'grade',
) -> Tuple[pd.DataFrame, Dict]:
    """Apply assumption-driven differencing per market-grade series."""

    mode = str(differencing_mode).upper()
    if mode not in set(M1_DIFFERENCING_MODE_OPTIONS):
        raise ValueError(
            f"differencing_mode must be one of {M1_DIFFERENCING_MODE_OPTIONS}, got: {differencing_mode}"
        )

    transformed = df.copy()
    transformed[M1_MODULE2_PRICE_COLUMN] = np.nan
    transformed['differencing_order_used'] = 0

    stationarity_report: Dict[str, Dict] = {}

    for market in sorted(transformed[market_col].unique()):
        for grade in sorted(transformed[grade_col].unique()):
            mask = (transformed[market_col] == market) & (transformed[grade_col] == grade)
            if not mask.any():
                continue

            prices = transformed.loc[mask, source_col].astype(float).values
            key = f"M{market}_{grade}"
            test_history: List[Dict] = []
            order_used = 0

            def _record_state(series_values: np.ndarray, current_order: int) -> Dict:
                result = preprocessor._evaluate_stationarity_series(
                    series_values,
                    test=stationarity_test,
                    significance_level=stationarity_alpha,
                )
                result['differencing_order'] = int(current_order)
                test_history.append(result)
                return result

            final_series = prices.copy()

            if mode == 'MANUAL':
                baseline_result = _record_state(final_series, order_used)
                if baseline_result.get('status') == STATUS_NOT_TESTABLE:
                    order_used = 0
                    final_series = prices.copy()
                else:
                    order_used = max(0, int(manual_differencing_order))
                    if order_used > 0:
                        differenced = np.diff(prices, n=order_used)
                        final_series = np.concatenate([np.full(order_used, np.nan), differenced])
                    _record_state(final_series, order_used)
            else:
                result: Dict = _record_state(final_series, order_used)
                while (
                    result.get('status') == STATUS_FAIL
                    and order_used < int(max_differencing_order)
                ):
                    order_used += 1
                    differenced = np.diff(prices, n=order_used)
                    final_series = np.concatenate([np.full(order_used, np.nan), differenced])
                    result = _record_state(final_series, order_used)

                if result.get('status') == STATUS_NOT_TESTABLE:
                    order_used = 0
                    final_series = prices.copy()

                if require_stationarity and result.get('status') == STATUS_FAIL:
                    raise ValueError(
                        f"Series {key} remains non-stationary at differencing order {order_used} "
                        f"(max allowed: {max_differencing_order}). "
                        "Increase M1_MAX_DIFFERENCING_ORDER or set M1_REQUIRE_STATIONARITY=False "
                        "if you want to continue with partially non-stationary series."
                    )

            expected_length = int(mask.sum())
            if len(final_series) != expected_length:
                if len(final_series) < expected_length:
                    final_series = np.concatenate(
                        [final_series, np.full(expected_length - len(final_series), np.nan)]
                    )
                else:
                    final_series = final_series[:expected_length]
            transformed.loc[mask, M1_MODULE2_PRICE_COLUMN] = final_series
            transformed.loc[mask, 'differencing_order_used'] = int(order_used)
            final_result = test_history[-1] if test_history else {}
            stationarity_report[key] = {
                'market_id': int(market),
                'grade': str(grade),
                'differencing_order_used': int(order_used),
                'status': final_result.get('status', STATUS_NOT_TESTABLE),
                'eligible_for_pwgc': bool(final_result.get('eligible_for_pwgc', False)),
                'adf_p_value': final_result.get('adf_p_value'),
                'reason': final_result.get('reason', ''),
                'final_stationary': bool(final_result.get('stationary', False)),
                'history': test_history,
            }

    preprocessor.processing_report['differencing'] = {
        'mode': mode,
        'manual_differencing_order': int(manual_differencing_order),
        'max_differencing_order': int(max_differencing_order),
    }
    preprocessor.processing_report['stationarity_workflow'] = {
        'test': str(stationarity_test).upper(),
        'significance_level': float(stationarity_alpha),
        'require_stationarity': bool(require_stationarity),
    }
    return transformed, stationarity_report


def run_full_preprocessing_pipeline(input_file: Optional[Union[str, Path, Sequence[Union[str, Path]]]], 
                                   output_file: Union[str, Path],
                                   config: Dict = None) -> pd.DataFrame:
    """
    Run complete preprocessing pipeline untuk pilot dataset.
    
    Pipeline stages:
    1. Load data
    2. Handle missing values
    3. Remove outliers
    4. Log transformation (untuk stabilize variance)
    5. Detrending
    6. Differencing
    7. Standardization
    8. Stationarity testing
    
    Parameters:
    -----------
    input_file : None | str | Path | list[str] | list[Path]
        Raw Excel input(s). Jika None, semua ``*.xlsx`` di ``data/raw`` diproses.
    output_file : str | Path
        Path to save processed data
    config : Dict, optional
        Configuration untuk setiap step
        
    Returns:
    --------
    pd.DataFrame
        Fully processed data ready untuk Granger causality testing
    """
    
    # Default config (single source of truth in src/config.py)
    defaults = {
        'missing_value_mode': M1_MISSING_VALUE_MODE,
        'outlier_mode': M1_OUTLIER_MODE,
        'outlier_threshold': M1_OUTLIER_THRESHOLD,
        'log_transform': M1_LOG_TRANSFORM,
        'detrend_method': M1_DETREND_METHOD,
        'stationarity_test': M1_STATIONARITY_TEST,
        'stationarity_significance_level': M1_STATIONARITY_SIGNIFICANCE_LEVEL,
        'require_stationarity': M1_REQUIRE_STATIONARITY,
        'differencing_mode': M1_DIFFERENCING_MODE,
        'manual_differencing_order': M1_MANUAL_DIFFERENCING_ORDER,
        'max_differencing_order': M1_MAX_DIFFERENCING_ORDER,
        'min_observations': M1_MIN_OBSERVATIONS,
        'skip_short_series': M1_SKIP_SHORT_SERIES,
        'skip_constant_series': M1_SKIP_CONSTANT_SERIES,
        'skip_zero_variance_series': M1_SKIP_ZERO_VARIANCE_SERIES,
        'skip_all_nan_series': M1_SKIP_ALL_NAN_SERIES,
        'verbose_stationarity': M1_VERBOSE_STATIONARITY,
        'standardization_enabled': M1_STANDARDIZATION_ENABLED,
        'standardization_method': M1_STANDARDIZATION_METHOD,
        'duplicate_strategy': M1_DUPLICATE_STRATEGY,
        'markets': None,
        'grades': None,
        'date_range': None,
    }
    config = {**defaults, **(config or {})}

    preprocessor = DataPreprocessor(verbose=M1_VERBOSE)
    preprocessor.min_observations = int(config['min_observations'])
    preprocessor.skip_short_series = bool(config['skip_short_series'])
    preprocessor.skip_constant_series = bool(config['skip_constant_series'])
    preprocessor.skip_zero_variance_series = bool(config['skip_zero_variance_series'])
    preprocessor.skip_all_nan_series = bool(config['skip_all_nan_series'])
    preprocessor.verbose_stationarity = bool(config['verbose_stationarity'])
    
    # Load
    df = preprocessor.load_pilot_data(
        input_file,
        duplicate_strategy=config['duplicate_strategy'],
    )

    # Filter first (before downstream processing)
    df = preprocessor.filter_data(
        df,
        markets=config.get('markets'),
        grades=config.get('grades'),
        date_range=config.get('date_range'),
    )
    if df.empty:
        raise ValueError("No data remaining after applying filters.")

    # Clean / transform controls from centralized config
    df = preprocessor.handle_missing_values(df, method=config['missing_value_mode'])
    df = preprocessor.remove_outliers(
        df, 
        method=config['outlier_mode'],
        threshold=config['outlier_threshold']
    )

    # Transform
    df = preprocessor.apply_log_transformation(df, apply=config.get('log_transform', True))
    df = preprocessor.detrend_data(df, method=config['detrend_method'])

    # Assumption-driven stationarity workflow
    df, stationarity_results = _apply_stationarity_driven_differencing(
        df=df,
        preprocessor=preprocessor,
        stationarity_test=config['stationarity_test'],
        stationarity_alpha=float(config['stationarity_significance_level']),
        differencing_mode=config['differencing_mode'],
        manual_differencing_order=int(config['manual_differencing_order']),
        max_differencing_order=int(config['max_differencing_order']),
        require_stationarity=bool(config['require_stationarity']),
    )

    if config.get('standardization_enabled', True):
        df = preprocessor.standardize_data(df, method=config['standardization_method'])
    else:
        df['price_standardized'] = df[M1_MODULE2_PRICE_COLUMN]

    # Final stationarity audit for downstream series
    final_stationarity_results = preprocessor.check_stationarity(
        df,
        price_col=M1_MODULE2_PRICE_COLUMN,
        test=config['stationarity_test'],
        significance_level=float(config['stationarity_significance_level']),
    )
    preprocessor.processing_report['stationarity_results'] = stationarity_results

    stationarity_report_df = _build_stationarity_report_dataframe(final_stationarity_results)
    stationarity_report_path = Path(output_file).parent / MODULE_01_STATIONARITY_REPORT_FILENAME
    stationarity_report_path.parent.mkdir(parents=True, exist_ok=True)
    stationarity_report_df.to_csv(stationarity_report_path, index=False)
    preprocessor.processing_report['stationarity_report_csv'] = str(stationarity_report_path)

    df_for_module2 = _filter_pwgc_eligible_data(df, stationarity_report_df)
    preprocessor.processing_report['pwgc_eligibility'] = {
        'eligible_series': int((stationarity_report_df['Eligible for PWGC'] == 'YES').sum()),
        'not_eligible_series': int((stationarity_report_df['Eligible for PWGC'] == 'NO').sum()),
        'total_series': int(len(stationarity_report_df)),
        'output_records': int(len(df_for_module2)),
    }

    _print_module1_summary(stationarity_report_df)
    
    # Save
    output_path = Path(output_file)
    output_format = 'excel' if output_path.suffix.lower() == '.xlsx' else 'csv'
    preprocessor.save_processed_data(df_for_module2, str(output_path), output_format=output_format)
    
    print(f"\n{'='*70}")
    print("PREPROCESSING COMPLETE - Ready for Granger Causality Testing")
    print(f"{'='*70}")
    
    return df_for_module2


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    input_file = "data/raw/Pilot Dataset.xlsx"
    output_file = "data/processed/module_01/preprocessed_pilot_data.csv"
    
    try:
        df_processed = run_full_preprocessing_pipeline(input_file, output_file)
        print(f"\nProcessed data shape: {df_processed.shape}")
        print(f"Columns: {df_processed.columns.tolist()}")
        print(f"\nFirst few rows:\n{df_processed.head(15)}")
        print(f"\nReady untuk Granger causality testing!")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure pilot dataset exists at: data/raw/Pilot Dataset.xlsx")
