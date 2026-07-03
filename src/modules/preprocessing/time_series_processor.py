"""
Time Series Preprocessing Module
==================================

Preprocessing untuk time series data harga beras sebelum Granger causality testing.
Includes: normalization, detrending, stationarity checking, missing data handling.

Key Steps:
1. Handle missing values
2. Outlier detection and treatment
3. Detrending
4. Differencing for stationarity
5. Normalization/Standardization
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
import logging
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class TimeSeriesProcessor:
    """
    Processor untuk preprocessing time series data pasar beras.
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize processor.
        
        Parameters:
        -----------
        verbose : bool
            Print detailed logs
        """
        self.verbose = verbose
        self.logger = logger
        self.preprocessing_history = {}

    def process_pipeline(self,
                        df: pd.DataFrame,
                        market_col: str = 'market',
                        price_col: str = 'price',
                        date_col: str = 'date',
                        volume_col: Optional[str] = 'volume',
                        steps: List[str] = None) -> pd.DataFrame:
        """
        Execute full preprocessing pipeline.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Raw market data
        market_col : str
            Column name untuk market identifier
        price_col : str
            Column name untuk price
        date_col : str
            Column name untuk date
        volume_col : str, optional
            Column name untuk volume
        steps : List[str], optional
            List of preprocessing steps to apply
            Default: ['handle_missing', 'remove_outliers', 'detrend', 
                     'differencing', 'standardize']
            
        Returns:
        --------
        pd.DataFrame
            Preprocessed data
        """
        
        if steps is None:
            steps = ['handle_missing', 'remove_outliers', 'detrend', 
                    'differencing', 'standardize']
        
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        if self.verbose:
            print(f"\n{'='*70}")
            print("TIME SERIES PREPROCESSING PIPELINE")
            print(f"{'='*70}")
            print(f"Input shape: {df.shape}")
            print(f"Markets: {df[market_col].nunique()}")
            print(f"Date range: {df[date_col].min()} to {df[date_col].max()}")
            print(f"Steps: {', '.join(steps)}\n")
        
        # Apply each step
        for step in steps:
            if self.verbose:
                print(f"→ Applying: {step}")
            
            if step == 'handle_missing':
                df = self.handle_missing_values(df, market_col, price_col, date_col)
            elif step == 'remove_outliers':
                df = self.remove_outliers(df, market_col, price_col)
            elif step == 'detrend':
                df = self.detrend_data(df, market_col, price_col)
            elif step == 'differencing':
                df = self.apply_differencing(df, market_col, price_col)
            elif step == 'standardize':
                df = self.standardize_data(df, market_col, price_col)
            else:
                self.logger.warning(f"Unknown step: {step}")
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Output shape: {df.shape}")
            print(f"{'='*70}\n")
        
        return df

    def handle_missing_values(self,
                             df: pd.DataFrame,
                             market_col: str,
                             price_col: str,
                             date_col: str,
                             method: str = 'interpolate') -> pd.DataFrame:
        """
        Handle missing values dalam time series.
        
        Parameters:
        -----------
        df : pd.DataFrame
        market_col, price_col, date_col : str
        method : str
            'drop' - Remove rows with missing values
            'interpolate' - Linear interpolation (default)
            'forward_fill' - Forward fill
            'mean' - Fill dengan market mean
            
        Returns:
        --------
        pd.DataFrame
        """
        
        df = df.copy()
        missing_before = df[price_col].isnull().sum()
        
        if method == 'drop':
            df = df.dropna(subset=[price_col])
        
        elif method == 'interpolate':
            # Interpolate per market
            for market in df[market_col].unique():
                market_mask = df[market_col] == market
                df.loc[market_mask, price_col] = (
                    df.loc[market_mask, price_col]
                    .interpolate(method='linear', limit_direction='both')
                )
        
        elif method == 'forward_fill':
            for market in df[market_col].unique():
                market_mask = df[market_col] == market
                df.loc[market_mask, price_col] = (
                    df.loc[market_mask, price_col].fillna(method='ffill')
                )
        
        elif method == 'mean':
            for market in df[market_col].unique():
                market_mean = df[df[market_col] == market][price_col].mean()
                market_mask = df[market_col] == market
                df.loc[market_mask, price_col] = (
                    df.loc[market_mask, price_col].fillna(market_mean)
                )
        
        missing_after = df[price_col].isnull().sum()
        
        if self.verbose:
            print(f"  Missing values: {missing_before} → {missing_after}")
        
        return df

    def remove_outliers(self,
                       df: pd.DataFrame,
                       market_col: str,
                       price_col: str,
                       method: str = 'iqr',
                       threshold: float = 1.5) -> pd.DataFrame:
        """
        Remove outliers menggunakan IQR atau Z-score.
        
        Parameters:
        -----------
        df : pd.DataFrame
        market_col, price_col : str
        method : str
            'iqr' - Interquartile range (default)
            'zscore' - Z-score method
        threshold : float
            IQR threshold (default 1.5) or Z-score threshold (default 3.0)
            
        Returns:
        --------
        pd.DataFrame
        """
        
        df = df.copy()
        records_before = len(df)
        
        for market in df[market_col].unique():
            market_mask = df[market_col] == market
            prices = df.loc[market_mask, price_col]
            
            if method == 'iqr':
                Q1 = prices.quantile(0.25)
                Q3 = prices.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outlier_mask = (prices < lower_bound) | (prices > upper_bound)
            
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(prices.dropna()))
                outlier_mask = z_scores > threshold
            
            # Replace outliers dengan market median
            if outlier_mask.any():
                median = prices.median()
                df.loc[market_mask, price_col] = (
                    df.loc[market_mask, price_col].mask(outlier_mask, median)
                )
        
        records_after = len(df)
        
        if self.verbose:
            print(f"  Records: {records_before} → {records_after} "
                  f"({records_before - records_after} outliers removed)")
        
        return df

    def detrend_data(self,
                    df: pd.DataFrame,
                    market_col: str,
                    price_col: str,
                    method: str = 'linear') -> pd.DataFrame:
        """
        Remove trend dari time series menggunakan detrending.
        
        Parameters:
        -----------
        df : pd.DataFrame
        market_col, price_col : str
        method : str
            'linear' - Linear detrending (default)
            'polynomial' - Polynomial detrending
            'seasonal' - Seasonal decomposition
            
        Returns:
        --------
        pd.DataFrame dengan kolom baru 'price_detrended'
        """
        
        df = df.copy()
        df['price_detrended'] = df[price_col].copy()
        
        for market in df[market_col].unique():
            market_mask = df[market_col] == market
            indices = np.where(market_mask)[0]
            prices = df.loc[market_mask, price_col].values
            
            if method == 'linear':
                # Fit linear trend
                z = np.polyfit(indices, prices, 1)
                p = np.poly1d(z)
                trend = p(indices)
                detrended = prices - trend
            
            elif method == 'polynomial':
                # Fit polynomial trend
                z = np.polyfit(indices, prices, 3)
                p = np.poly1d(z)
                trend = p(indices)
                detrended = prices - trend
            
            df.loc[market_mask, 'price_detrended'] = detrended
        
        if self.verbose:
            print(f"  Detrended using {method} method")
        
        return df

    def apply_differencing(self,
                          df: pd.DataFrame,
                          market_col: str,
                          price_col: str,
                          lag: int = 1,
                          order: int = 1) -> pd.DataFrame:
        """
        Apply differencing untuk achieve stationarity.
        
        Parameters:
        -----------
        df : pd.DataFrame
        market_col, price_col : str
        lag : int
            Lag untuk differencing (default 1)
        order : int
            Order of differencing (1 or 2)
            
        Returns:
        --------
        pd.DataFrame dengan kolom 'price_diff'
        """
        
        df = df.copy()
        df['price_diff'] = np.nan
        
        for market in df[market_col].unique():
            market_mask = df[market_col] == market
            prices = df.loc[market_mask, price_col].values
            
            # First differencing
            diff1 = np.diff(prices, n=1) if order >= 1 else prices
            
            # Second differencing jika needed
            diff_final = np.diff(diff1, n=1) if order >= 2 else diff1
            
            # Pad dengan NaN untuk maintain index alignment
            diff_padded = np.concatenate([np.full(lag * order, np.nan), diff_final])
            
            df.loc[market_mask, 'price_diff'] = diff_padded[:len(prices)]
        
        if self.verbose:
            print(f"  Applied {order}-order differencing (lag={lag})")
        
        return df

    def standardize_data(self,
                        df: pd.DataFrame,
                        market_col: str,
                        price_col: str,
                        method: str = 'zscore') -> pd.DataFrame:
        """
        Standardize/normalize data.
        
        Parameters:
        -----------
        df : pd.DataFrame
        market_col, price_col : str
        method : str
            'zscore' - Z-score normalization (mean=0, std=1)
            'minmax' - Min-Max scaling (0-1 range)
            'robust' - Robust scaling (median, IQR)
            
        Returns:
        --------
        pd.DataFrame dengan kolom 'price_standardized'
        """
        
        df = df.copy()
        df['price_standardized'] = np.nan
        
        for market in df[market_col].unique():
            market_mask = df[market_col] == market
            prices = df.loc[market_mask, price_col].values
            
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
            
            df.loc[market_mask, 'price_standardized'] = standardized
        
        if self.verbose:
            print(f"  Standardized using {method} method")
        
        return df

    def check_stationarity(self,
                          df: pd.DataFrame,
                          market_col: str,
                          price_col: str,
                          test: str = 'adf') -> Dict:
        """
        Check stationarity menggunakan ADF atau KPSS test.
        
        Parameters:
        -----------
        df : pd.DataFrame
        market_col, price_col : str
        test : str
            'adf' - Augmented Dickey-Fuller (default)
            'kpss' - KPSS test
            
        Returns:
        --------
        Dict dengan hasil test per market
        """
        
        results = {}
        
        for market in df[market_col].unique():
            prices = df[df[market_col] == market][price_col].dropna().values
            
            if len(prices) < 10:
                results[market] = {'stationary': False, 'reason': 'insufficient_data'}
                continue
            
            try:
                if test == 'adf':
                    stat, p_value, _, _, _, _ = adfuller(prices)
                    stationary = p_value < 0.05
                
                elif test == 'kpss':
                    stat, p_value, _, _ = kpss(prices)
                    stationary = p_value > 0.05
                
                results[market] = {
                    'stationary': stationary,
                    'test': test,
                    'statistic': stat,
                    'p_value': p_value
                }
            
            except Exception as e:
                results[market] = {'stationary': False, 'error': str(e)}
        
        return results

    def get_preprocessing_report(self, df: pd.DataFrame) -> Dict:
        """Generate preprocessing summary report."""
        
        report = {
            'shape': df.shape,
            'columns': list(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'null_counts': df.isnull().sum().to_dict(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist()
        }
        
        return report


class DataQualityChecker:
    """
    Checker untuk data quality metrics.
    """
    
    @staticmethod
    def check_completeness(df: pd.DataFrame, 
                          market_col: str,
                          price_col: str,
                          min_coverage: float = 0.8) -> Dict:
        """
        Check data completeness per market.
        
        Parameters:
        -----------
        min_coverage : float
            Minimum coverage threshold (0-1)
            
        Returns:
        --------
        Dict dengan completeness metrics
        """
        
        completeness = {}
        
        for market in df[market_col].unique():
            market_data = df[df[market_col] == market]
            total = len(market_data)
            valid = market_data[price_col].notna().sum()
            coverage = valid / total if total > 0 else 0
            
            completeness[market] = {
                'total': total,
                'valid': valid,
                'missing': total - valid,
                'coverage': coverage,
                'meets_threshold': coverage >= min_coverage
            }
        
        return completeness
    
    @staticmethod
    def check_consistency(df: pd.DataFrame,
                         market_col: str,
                         price_col: str) -> Dict:
        """
        Check data consistency (price ranges, logical values).
        
        Returns:
        --------
        Dict dengan consistency checks
        """
        
        consistency = {
            'price_ranges': {},
            'logical_errors': []
        }
        
        for market in df[market_col].unique():
            prices = df[df[market_col] == market][price_col]
            consistency['price_ranges'][market] = {
                'min': float(prices.min()),
                'max': float(prices.max()),
                'mean': float(prices.mean()),
                'std': float(prices.std())
            }
        
        # Check for logical errors
        if (df[price_col] <= 0).any():
            consistency['logical_errors'].append('Negative or zero prices detected')
        
        if df[price_col].max() > 1000000:
            consistency['logical_errors'].append('Unusually high prices detected')
        
        return consistency


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Import fetcher untuk contoh
    from rice_market_data_fetcher import RiceMarketDataFetcher
    
    print("Generating sample data...")
    fetcher = RiceMarketDataFetcher()
    df = fetcher.fetch_synthetic_timeseries(
        start_date="2023-01-01",
        end_date="2023-12-31"
    )
    
    print("\nApplying preprocessing pipeline...")
    processor = TimeSeriesProcessor(verbose=True)
    
    # Full pipeline
    df_processed = processor.process_pipeline(
        df,
        market_col='market',
        price_col='price',
        date_col='date',
        steps=['handle_missing', 'remove_outliers', 'detrend', 'differencing', 'standardize']
    )
    
    # Check stationarity
    print("\nChecking stationarity...")
    stationarity = processor.check_stationarity(
        df_processed,
        market_col='market',
        price_col='price_diff'
    )
    
    # Data quality check
    print("\nData quality check...")
    quality_checker = DataQualityChecker()
    completeness = quality_checker.check_completeness(df_processed, 'market', 'price')
    consistency = quality_checker.check_consistency(df_processed, 'market', 'price')
    
    print(f"\nProcessed data shape: {df_processed.shape}")
    print(df_processed.head(10))
