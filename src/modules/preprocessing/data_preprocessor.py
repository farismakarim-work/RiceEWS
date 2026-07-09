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

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)
RAW_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"


def _normalize_input_files(
    input_file: Optional[Union[str, Path, Sequence[Union[str, Path]]]]
) -> List[Path]:
    """Normalize supported input_file values into a list of Excel paths."""

    if input_file is None:
        files = sorted(RAW_DATA_DIR.glob("*.xlsx"))
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
    
    def __init__(self, verbose: bool = True):
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
                             method: str = 'interpolate') -> pd.DataFrame:
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
        
        if method == 'drop':
            df = df.dropna(subset=[price_col])
        
        elif method == 'interpolate':
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
                            .fillna(method='ffill')
                            .fillna(method='bfill')
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
                       method: str = 'iqr',
                       threshold: float = 1.5) -> pd.DataFrame:
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
                                apply: bool = True) -> pd.DataFrame:
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
                    method: str = 'linear') -> pd.DataFrame:
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
    
    def check_stationarity(self,
                          df: pd.DataFrame,
                          price_col: str = 'price_diff',
                          market_col: str = 'market_id',
                          grade_col: str = 'grade',
                          test: str = 'adf') -> Dict:
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
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"STATIONARITY TESTING ({test.upper()} test)")
            print(f"{'='*70}")
        
        results = {}
        stationary_count = 0
        
        for market in sorted(df[market_col].unique()):
            for grade in sorted(df[grade_col].unique()):
                mask = (df[market_col] == market) & (df[grade_col] == grade)
                prices = df.loc[mask, price_col].dropna().values
                
                key = f"M{market}_{grade}"
                
                if len(prices) < 10:
                    results[key] = {
                        'market_id': market,
                        'grade': grade,
                        'stationary': False,
                        'reason': 'insufficient_data',
                        'p_value': None
                    }
                    continue
                
                try:
                    if test == 'adf':
                        # ADF: H0 = non-stationary, reject if p < 0.05
                        stat, p_value, _, _, _, _ = adfuller(prices, autolag='AIC')
                        stationary = p_value < 0.05
                        test_stat = stat
                    
                    elif test == 'kpss':
                        # KPSS: H0 = stationary, reject if p > 0.05
                        stat, p_value, _, _ = kpss(prices, regression='c', nlags='auto')
                        stationary = p_value > 0.05
                        test_stat = stat
                    
                    results[key] = {
                        'market_id': market,
                        'grade': grade,
                        'stationary': stationary,
                        'test': test,
                        'test_statistic': float(test_stat),
                        'p_value': float(p_value)
                    }
                    
                    if stationary:
                        stationary_count += 1
                    
                    if self.verbose:
                        status = "✓" if stationary else "✗"
                        print(f"  {key}: {status} (p={p_value:.4f})")
                
                except Exception as e:
                    results[key] = {
                        'market_id': market,
                        'grade': grade,
                        'stationary': False,
                        'error': str(e)
                    }
        
        if self.verbose:
            total_tests = len(results)
            print(f"\nStationary: {stationary_count}/{total_tests}")
        
        self.processing_report['stationarity_test'] = {
            'test': test,
            'stationary_count': stationary_count,
            'total_tests': len(results)
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
    
    # Default config
    # NOTE (Fix #1 – Kinnear & Mazumdar alignment): Linear detrending followed by
    # order-1 differencing is redundant — differencing already removes a linear trend.
    # Applying both is "over-differencing" and can distort the autocorrelation structure
    # that Granger tests rely on.  The default is therefore 'none' for detrend_method
    # when differencing_order >= 1.  If you need detrending without differencing, set
    # detrend_method to 'linear' or 'polynomial' AND differencing_order to 0.
    defaults = {
        'missing_method': 'interpolate',
        'outlier_method': 'iqr',
        'outlier_threshold': 1.5,
        'log_transform': True,
        'detrend_method': 'none',
        'differencing_order': 1,
        'standardize_method': 'zscore',
        'stationarity_test': 'adf',
        'duplicate_strategy': 'error',
    }
    config = {**defaults, **(config or {})}
    
    preprocessor = DataPreprocessor(verbose=True)
    
    # Load
    df = preprocessor.load_pilot_data(
        input_file,
        duplicate_strategy=config['duplicate_strategy'],
    )
    
    # Clean
    df = preprocessor.handle_missing_values(df, method=config['missing_method'])
    df = preprocessor.remove_outliers(
        df, 
        method=config['outlier_method'],
        threshold=config['outlier_threshold']
    )
    
    # Warn about redundant double-detrending (Fix #1)
    detrend_method = config.get('detrend_method', 'none')
    differencing_order = config.get('differencing_order', 1)
    if detrend_method not in ('none', None) and differencing_order >= 1:
        import warnings as _warnings
        _warnings.warn(
            "Both detrend_method='{}' and differencing_order={} are active.  "
            "Differencing order-1 already removes a linear trend, so applying "
            "linear detrending beforehand is redundant and may cause "
            "over-differencing that weakens Granger test power.  "
            "Consider setting detrend_method='none'.".format(
                detrend_method, differencing_order),
            UserWarning, stacklevel=2,
        )
    
    # Transform
    df = preprocessor.apply_log_transformation(df, apply=config.get('log_transform', True))
    df = preprocessor.detrend_data(df, method=config['detrend_method'])
    df = preprocessor.apply_differencing(df, order=config['differencing_order'])
    df = preprocessor.standardize_data(df, method=config['standardize_method'])
    
    # Test stationarity
    stationarity_results = preprocessor.check_stationarity(
        df,
        test=config['stationarity_test']
    )
    
    # Save
    output_path = Path(output_file)
    output_format = 'excel' if output_path.suffix.lower() == '.xlsx' else 'csv'
    preprocessor.save_processed_data(df, str(output_path), output_format=output_format)
    
    print(f"\n{'='*70}")
    print("PREPROCESSING COMPLETE - Ready for Granger Causality Testing")
    print(f"{'='*70}")
    
    return df


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
