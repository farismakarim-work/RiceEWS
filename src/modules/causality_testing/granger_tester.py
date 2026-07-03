"""
MODUL 2: Granger Causality Testing
===================================

Pairwise Granger causality testing untuk mengidentifikasi market leaders.
Reference: Kinnear & Mazumdar (2023) - Unconditional pairwise Granger tests

Implementation WITHOUT statsmodels - custom OLS regression untuk flexibility.

Features:
- Pairwise Granger causality testing (6 markets = 30 tests)
- Separate analysis per grade (low1, low2, med1, med2)
- Configurable lag order (default: 4)
- F-test untuk significance testing
- Output: Causal relationships per grade
- Ready untuk network inference

Pipeline:
1. Load preprocessed data (dari MODUL 1)
2. Extract time series per market-grade
3. Perform pairwise Granger tests
4. Calculate F-statistics and p-values
5. Identify significant causal relationships
6. Generate causal matrix per grade
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import json
from scipy import stats

logger = logging.getLogger(__name__)


class GrangerCausalityTester:
    """
    Granger causality testing untuk rice market price relationships.
    Tests pairwise causal relationships between markets per grade.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize Granger tester.
        
        Parameters:
        -----------
        verbose : bool
            Print detailed logs
        """
        self.verbose = verbose
        self.logger = logger
        self.testing_report = {}
    
    def load_preprocessed_data(self, filepath: str) -> pd.DataFrame:
        """
        Load preprocessed data dari MODUL 1.
        
        Expected columns: date, market_id, grade, price, price_log, 
                         price_detrended, price_diff, price_standardized
        
        Parameters:
        -----------
        filepath : str
            Path to preprocessed CSV file
            
        Returns:
        --------
        pd.DataFrame
            Preprocessed time series data
        """
        
        try:
            filepath = Path(filepath)
            if not filepath.exists():
                raise FileNotFoundError(f"File not found: {filepath}")
            
            df = pd.read_csv(filepath)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values(['date', 'market_id', 'grade']).reset_index(drop=True)
            
            if self.verbose:
                print(f"\n{'='*70}")
                print("LOADING PREPROCESSED DATA")
                print(f"{'='*70}")
                print(f"File: {filepath.name}")
                print(f"Shape: {df.shape}")
                print(f"Columns: {df.columns.tolist()}")
                print(f"Markets: {sorted(df['market_id'].unique().tolist())}")
                print(f"Grades: {sorted(df['grade'].unique().tolist())}")
            
            self.testing_report['data_loaded'] = {
                'records': len(df),
                'markets': sorted(df['market_id'].unique().tolist()),
                'grades': sorted(df['grade'].unique().tolist())
            }
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def prepare_time_series(self,
                           df: pd.DataFrame,
                           market_id: int,
                           grade: str,
                           price_col: str = 'price_diff') -> Optional[np.ndarray]:
        """
        Extract dan prepare time series untuk single market-grade.
        
        Parameters:
        -----------
        df : pd.DataFrame
        market_id : int
            Market identifier
        grade : str
            Rice grade (low1, low2, med1, med2)
        price_col : str
            Column to use (default: price_diff untuk Granger)
            
        Returns:
        --------
        np.ndarray atau None
            Time series data (cleaned, NaN removed)
        """
        
        mask = (df['market_id'] == market_id) & (df['grade'] == grade)
        ts = df.loc[mask, price_col].dropna().values
        
        if len(ts) < 10:
            return None
        
        return ts
    
    def _ols_regression(self,
                       X: np.ndarray,
                       y: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Custom OLS regression (tanpa statsmodels).
        
        Parameters:
        -----------
        X : np.ndarray
            Design matrix (n_samples, n_features)
        y : np.ndarray
            Target variable (n_samples,)
            
        Returns:
        --------
        Tuple[coefficients, r_squared, residual_ss]
            Beta coefficients, R-squared, Sum of squared residuals
        """
        
        # Add intercept column
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        
        try:
            # OLS: beta = (X'X)^-1 X'y
            XtX = X_with_intercept.T @ X_with_intercept
            XtX_inv = np.linalg.inv(XtX)
            beta = XtX_inv @ X_with_intercept.T @ y
            
            # Predictions dan residuals
            y_pred = X_with_intercept @ beta
            residuals = y - y_pred
            rss = np.sum(residuals ** 2)
            
            # R-squared
            tss = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (rss / tss) if tss != 0 else 0
            
            return beta, r_squared, rss
            
        except np.linalg.LinAlgError:
            return None, None, None
    
    def _build_lagged_matrix(self,
                            ts: np.ndarray,
                            lag_order: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build lagged regression matrix untuk single variable.
        
        Formula: y_t = c + a1*y_{t-1} + a2*y_{t-2} + ... + ap*y_{t-p} + e_t
        
        Parameters:
        -----------
        ts : np.ndarray
            Time series (1D array)
        lag_order : int
            Number of lags
            
        Returns:
        --------
        Tuple[X, y]
            Design matrix (X), target (y)
        """
        
        n = len(ts)
        X = np.zeros((n - lag_order, lag_order))
        
        for i in range(lag_order):
            X[:, i] = ts[lag_order - i - 1:n - i - 1]
        
        y = ts[lag_order:]
        
        return X, y
    
    def _build_granger_matrix(self,
                             y_ts: np.ndarray,
                             x_ts: np.ndarray,
                             lag_order: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build augmented regression matrix untuk Granger causality test.
        
        Formula: y_t = c + sum(a_i * y_{t-i}) + sum(b_i * x_{t-i}) + e_t
        
        Parameters:
        -----------
        y_ts : np.ndarray
            Dependent variable time series
        x_ts : np.ndarray
            Potential Granger cause time series
        lag_order : int
            Number of lags
            
        Returns:
        --------
        Tuple[X_augmented, y]
            Augmented design matrix, target variable
        """
        
        min_len = min(len(y_ts), len(x_ts))
        y_ts = y_ts[:min_len]
        x_ts = x_ts[:min_len]
        
        n = min_len
        X_aug = np.zeros((n - lag_order, 2 * lag_order))
        
        # Lags of y
        for i in range(lag_order):
            X_aug[:, i] = y_ts[lag_order - i - 1:n - i - 1]
        
        # Lags of x
        for i in range(lag_order):
            X_aug[:, lag_order + i] = x_ts[lag_order - i - 1:n - i - 1]
        
        y = y_ts[lag_order:]
        
        return X_aug, y
    
    def granger_causality_test(self,
                              y_ts: np.ndarray,
                              x_ts: np.ndarray,
                              lag_order: int = 4,
                              significance_level: float = 0.05) -> Dict:
        """
        Perform Granger causality test.
        
        H0: x does NOT Granger-cause y
        H1: x DOES Granger-cause y
        
        Test statistic: F = (RSS_restricted - RSS_unrestricted) / lag_order
                            ÷ RSS_unrestricted / (n - 2*lag_order - 1)
        
        Parameters:
        -----------
        y_ts : np.ndarray
            Dependent variable (target)
        x_ts : np.ndarray
            Potential Granger cause
        lag_order : int
            Number of lags (default: 4)
        significance_level : float
            Alpha for significance testing (default: 0.05)
            
        Returns:
        --------
        Dict
            Test results including F-stat, p-value, causality indicator
        """
        
        # Build restricted model (only y lags)
        X_restricted, y = self._build_lagged_matrix(y_ts, lag_order)
        
        if X_restricted.shape[0] < lag_order + 1:
            return {
                'f_statistic': None,
                'p_value': None,
                'granger_causes': False,
                'reason': 'insufficient_data'
            }
        
        # Fit restricted model
        beta_r, r2_r, rss_r = self._ols_regression(X_restricted, y)
        
        if beta_r is None:
            return {
                'f_statistic': None,
                'p_value': None,
                'granger_causes': False,
                'reason': 'regression_failed'
            }
        
        # Build augmented model (y lags + x lags)
        X_augmented, y = self._build_granger_matrix(y_ts, x_ts, lag_order)
        
        if X_augmented.shape[0] < 2 * lag_order + 1:
            return {
                'f_statistic': None,
                'p_value': None,
                'granger_causes': False,
                'reason': 'insufficient_data'
            }
        
        # Fit augmented model
        beta_u, r2_u, rss_u = self._ols_regression(X_augmented, y)
        
        if beta_u is None:
            return {
                'f_statistic': None,
                'p_value': None,
                'granger_causes': False,
                'reason': 'regression_failed'
            }
        
        # Calculate F-statistic
        n = len(y)
        numerator = (rss_r - rss_u) / lag_order
        denominator = rss_u / (n - 2 * lag_order - 1)
        
        if denominator == 0:
            f_stat = np.inf
        else:
            f_stat = numerator / denominator
        
        # Calculate p-value from F-distribution
        # F ~ F(lag_order, n - 2*lag_order - 1)
        df1 = lag_order
        df2 = n - 2 * lag_order - 1
        
        p_value = 1 - stats.f.cdf(f_stat, df1, df2)
        
        # Determine causality
        granger_causes = p_value < significance_level
        
        return {
            'f_statistic': f_stat,
            'p_value': p_value,
            'granger_causes': granger_causes,
            'lag_order': lag_order,
            'n_observations': n,
            'r2_restricted': r2_r,
            'r2_augmented': r2_u
        }
    
    def test_all_pairwise_relationships(self,
                                       df: pd.DataFrame,
                                       grade: str,
                                       lag_order: int = 4,
                                       price_col: str = 'price_diff') -> Dict:
        """
        Test ALL pairwise Granger relationships untuk satu grade.
        
        6 markets → 6*5 = 30 pairwise tests
        
        Parameters:
        -----------
        df : pd.DataFrame
        grade : str
            Rice grade (low1, low2, med1, med2)
        lag_order : int
            Number of lags
        price_col : str
            Which preprocessed column to use
            
        Returns:
        --------
        Dict
            Results untuk semua pairwise combinations
        """
        
        markets = sorted(df[df['grade'] == grade]['market_id'].unique())
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"GRANGER CAUSALITY TESTING - Grade: {grade}")
            print(f"{'='*70}")
            print(f"Markets: {markets}")
            print(f"Pairwise tests: {len(markets) * (len(markets) - 1)}")
            print(f"Lag order: {lag_order}\n")
        
        results = {}
        causal_count = 0
        
        # Test: does x_market Granger-cause y_market?
        for y_market in markets:
            for x_market in markets:
                if y_market == x_market:
                    continue
                
                # Prepare time series
                y_ts = self.prepare_time_series(df, y_market, grade, price_col)
                x_ts = self.prepare_time_series(df, x_market, grade, price_col)
                
                if y_ts is None or x_ts is None:
                    results[f"M{x_market}→M{y_market}"] = {
                        'granger_causes': False,
                        'reason': 'insufficient_data'
                    }
                    continue
                
                # Perform Granger test
                test_result = self.granger_causality_test(y_ts, x_ts, lag_order)
                results[f"M{x_market}→M{y_market}"] = test_result
                
                if test_result['granger_causes']:
                    causal_count += 1
                    if self.verbose:
                        print(f"  ✓ M{x_market}→M{y_market}: F={test_result['f_statistic']:.4f}, "
                              f"p={test_result['p_value']:.4f}")
        
        if self.verbose:
            print(f"\nSignificant causal relationships: {causal_count}/{len(results)}")
        
        return results
    
    def build_causal_matrix(self,
                           grade_results: Dict,
                           markets: List[int]) -> np.ndarray:
        """
        Build causal adjacency matrix dari test results.
        
        Matrix[i,j] = 1 jika market j Granger-causes market i, 0 otherwise
        
        Parameters:
        -----------
        grade_results : Dict
            Granger test results untuk satu grade
        markets : List[int]
            List of market IDs
            
        Returns:
        --------
        np.ndarray
            Causal adjacency matrix (n_markets x n_markets)
        """
        
        n = len(markets)
        matrix = np.zeros((n, n))
        
        market_idx = {m: i for i, m in enumerate(markets)}
        
        for key, result in grade_results.items():
            if result.get('granger_causes', False):
                # Parse "M{x}→M{y}" format
                parts = key.split('→')
                x_market = int(parts[0][1:])  # Remove 'M' prefix
                y_market = int(parts[1][1:])
                
                i = market_idx[y_market]
                j = market_idx[x_market]
                matrix[i, j] = 1
        
        return matrix
    
    def build_adjacency_matrix(self,
                              grade_results: Dict,
                              markets: List[int],
                              strength: bool = False) -> np.ndarray:
        """
        Build adjacency matrix dengan optional strength weights.
        
        Parameters:
        -----------
        grade_results : Dict
        markets : List[int]
        strength : bool
            If True, use F-statistic as weight; if False, binary
            
        Returns:
        --------
        np.ndarray
            Adjacency matrix (n_markets x n_markets)
        """
        
        n = len(markets)
        matrix = np.zeros((n, n))
        
        market_idx = {m: i for i, m in enumerate(markets)}
        
        for key, result in grade_results.items():
            if result.get('granger_causes', False):
                parts = key.split('→')
                x_market = int(parts[0][1:])
                y_market = int(parts[1][1:])
                
                i = market_idx[y_market]
                j = market_idx[x_market]
                
                if strength:
                    matrix[i, j] = result.get('f_statistic', 1)
                else:
                    matrix[i, j] = 1
        
        return matrix
    
    def get_market_out_degree(self,
                             causal_matrix: np.ndarray,
                             markets: List[int]) -> Dict:
        """
        Calculate out-degree (influence) untuk setiap market.
        
        Out-degree = jumlah markets yang di-influence (causes)
        
        Parameters:
        -----------
        causal_matrix : np.ndarray
            Causal adjacency matrix
        markets : List[int]
        
        Returns:
        --------
        Dict
            Market IDs dengan out-degree scores
        """
        
        out_degrees = {}
        for i, market in enumerate(markets):
            out_degrees[market] = int(np.sum(causal_matrix[:, i]))
        
        return out_degrees
    
    def get_market_in_degree(self,
                            causal_matrix: np.ndarray,
                            markets: List[int]) -> Dict:
        """
        Calculate in-degree (being influenced) untuk setiap market.
        
        In-degree = jumlah markets yang influence ini (causes this one)
        
        Parameters:
        -----------
        causal_matrix : np.ndarray
        markets : List[int]
        
        Returns:
        --------
        Dict
            Market IDs dengan in-degree scores
        """
        
        in_degrees = {}
        for i, market in enumerate(markets):
            in_degrees[market] = int(np.sum(causal_matrix[i, :]))
        
        return in_degrees
    
    def identify_market_leaders(self,
                               causal_matrix: np.ndarray,
                               markets: List[int]) -> List[int]:
        """
        Identify market leaders based on out-degree.
        
        Market leader = market dengan highest out-degree (most influential)
        
        Parameters:
        -----------
        causal_matrix : np.ndarray
        markets : List[int]
        
        Returns:
        --------
        List[int]
            Ranked market IDs (highest to lowest influence)
        """
        
        out_degrees = self.get_market_out_degree(causal_matrix, markets)
        ranked_markets = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)
        
        return [m for m, _ in ranked_markets]
    
    def get_testing_report(self) -> Dict:
        """Generate testing summary report."""
        return self.testing_report
    
    def save_results(self,
                    all_results: Dict,
                    output_path: str,
                    include_report: bool = True) -> Path:
        """
        Save Granger testing results.
        
        Parameters:
        -----------
        all_results : Dict
            Complete results dictionary (per grade)
        output_path : str
            Path untuk output file
        include_report : bool
            Save detailed report
            
        Returns:
        --------
        Path
        """
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save results as JSON
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        if self.verbose:
            print(f"\n✓ Saved Granger results to: {output_path}")
        
        return output_path


def run_full_granger_analysis(input_file: str,
                             output_file: str,
                             config: Dict = None) -> Dict:
    """
    Run complete Granger causality analysis.
    
    Parameters:
    -----------
    input_file : str
        Path to preprocessed CSV (dari MODUL 1)
    output_file : str
        Path to save results JSON
    config : Dict, optional
        Configuration (lag_order, etc.)
        
    Returns:
    --------
    Dict
        Complete analysis results per grade
    """
    
    if config is None:
        config = {
            'lag_order': 4,
            'price_col': 'price_diff',
            'significance_level': 0.05
        }
    
    tester = GrangerCausalityTester(verbose=True)
    
    # Load preprocessed data
    df = tester.load_preprocessed_data(input_file)
    
    # Test per grade
    all_results = {}
    markets = sorted(df['market_id'].unique().tolist())
    
    for grade in sorted(df['grade'].unique()):
        print(f"\n{'='*70}")
        print(f"PROCESSING GRADE: {grade}")
        print(f"{'='*70}")
        
        # Test pairwise relationships
        grade_results = tester.test_all_pairwise_relationships(
            df,
            grade,
            lag_order=config['lag_order'],
            price_col=config['price_col']
        )
        
        # Build causal matrix
        causal_matrix = tester.build_causal_matrix(grade_results, markets)
        
        # Calculate centrality measures
        out_degrees = tester.get_market_out_degree(causal_matrix, markets)
        in_degrees = tester.get_market_in_degree(causal_matrix, markets)
        market_leaders = tester.identify_market_leaders(causal_matrix, markets)
        
        # Store results
        all_results[grade] = {
            'pairwise_tests': grade_results,
            'causal_matrix': causal_matrix.tolist(),
            'out_degrees': out_degrees,
            'in_degrees': in_degrees,
            'market_leaders': market_leaders,
            'config': config
        }
        
        if tester.verbose:
            print(f"\nMarket Leaders (by influence):")
            for i, market in enumerate(market_leaders, 1):
                print(f"  {i}. Market {market} (out-degree: {out_degrees[market]})")
    
    # Save results
    tester.save_results(all_results, output_file)
    
    print(f"\n{'='*70}")
    print("GRANGER CAUSALITY ANALYSIS COMPLETE")
    print(f"{'='*70}")
    
    return all_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    input_file = "data/processed/preprocessed_pilot_data.csv"
    output_file = "data/processed/granger_results.json"
    
    try:
        results = run_full_granger_analysis(
            input_file,
            output_file,
            config={
                'lag_order': 4,
                'price_col': 'price_diff',
                'significance_level': 0.05
            }
        )
        
        print(f"\nAnalysis complete!")
        print(f"Results saved to: {output_file}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure preprocessed data exists at: data/processed/preprocessed_pilot_data.csv")
