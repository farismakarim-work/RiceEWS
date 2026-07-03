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
from typing import Dict, List, Tuple, Optional
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
import logging
import warnings
import json

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


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
    
    def load_pilot_data(self, filepath: str) -> pd.DataFrame:
        """
        Load pilot dataset dari Excel file.
        
        Expected format:
        - Date (datetime)
        - Market_id (int: 101-106)
        - Grade (str: low1, low2, med1, med2)
        - Price (int)
        
        Parameters:
        -----------
        filepath : str
            Path to Excel file (e.g., 'data/raw/Pilot Dataset.xlsx')
            
        Returns:
        --------
        pd.DataFrame
            Loaded data with standardized column names
        """
        
        try:
            filepath = Path(filepath)
            if not filepath.exists():
                raise FileNotFoundError(f"File not found: {filepath}")
            
            # Load from Excel
            df = pd.read_excel(filepath)
            
            if self.verbose:
                print(f"\n{'='*70}")
                print("DATA LOADING")
                print(f"{'='*70}")
                print(f"File: {filepath.name}")
                print(f"Shape: {df.shape}")
                print(f"Columns (raw): {df.columns.tolist()}")
            
            # Standardize column names to lowercase
            df.columns = df.columns.str.lower()
            
            # Validate required columns
            required_cols = ['date', 'market_id', 'grade', 'price']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}\nFound: {df.columns.tolist()}")
            
            # Convert date to datetime if not already
            df['date'] = pd.to_datetime(df['date'])
            
            # Sort by date, market, grade
            df = df.sort_values(['date', 'market_id', 'grade']).reset_index(drop=True)
            
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
                'markets': sorted(df['market_id'].unique().tolist()),
                'grades': sorted(df['grade'].unique().tolist()),
                'date_range': (str(df['date'].min().date()), str(df['date'].max().date())),
                'price_range': (int(df['price'].min()), int(df['price'].max()))
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


def run_full_preprocessing_pipeline(input_file: str, 
                                   output_file: str,
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
    input_file : str
        Path to raw Excel file
    output_file : str
        Path to save processed data
    config : Dict, optional
        Configuration untuk setiap step
        
    Returns:
    --------
    pd.DataFrame
        Fully processed data ready untuk Granger causality testing
    """
    
    # Default config
    if config is None:
        config = {
            'missing_method': 'interpolate',
            'outlier_method': 'iqr',
            'outlier_threshold': 1.5,
            'log_transform': True,
            'detrend_method': 'linear',
            'differencing_order': 1,
            'standardize_method': 'zscore',
            'stationarity_test': 'adf'
        }
    
    preprocessor = DataPreprocessor(verbose=True)
    
    # Load
    df = preprocessor.load_pilot_data(input_file)
    
    # Clean
    df = preprocessor.handle_missing_values(df, method=config['missing_method'])
    df = preprocessor.remove_outliers(
        df, 
        method=config['outlier_method'],
        threshold=config['outlier_threshold']
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
    output_format = 'excel' if output_file.endswith('.xlsx') else 'csv'
    preprocessor.save_processed_data(df, output_file, output_format=output_format)
    
    print(f"\n{'='*70}")
    print("PREPROCESSING COMPLETE - Ready for Granger Causality Testing")
    print(f"{'='*70}")
    
    return df


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    input_file = "data/raw/Pilot Dataset.xlsx"
    output_file = "data/processed/preprocessed_pilot_data.csv"
    
    try:
        df_processed = run_full_preprocessing_pipeline(input_file, output_file)
        print(f"\nProcessed data shape: {df_processed.shape}")
        print(f"Columns: {df_processed.columns.tolist()}")
        print(f"\nFirst few rows:\n{df_processed.head(15)}")
        print(f"\nReady untuk Granger causality testing!")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure pilot dataset exists at: data/raw/Pilot Dataset.xlsx")
