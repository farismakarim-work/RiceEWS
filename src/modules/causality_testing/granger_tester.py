"""
MODUL 2: Granger Causality Testing dengan Visualization
========================================================

Pairwise Granger causality testing untuk mengidentifikasi market leaders.
Reference: Kinnear & Mazumdar (2023) - Unconditional pairwise Granger tests

Implementation WITHOUT statsmodels - custom OLS regression untuk flexibility.

Features:
- Pairwise Granger causality testing (6 markets = 30 tests)
- Separate analysis per grade (low1, low2, med1, med2)
- Configurable lag order (default: 4)
- F-test untuk significance testing
- Output: Multiple formats (JSON, CSV, Excel, NPZ, Markdown)
- VISUALIZATION: Network graphs, heatmaps, bar charts (PNG, SVG, HTML)
- Ready untuk network inference

Pipeline:
1. Load preprocessed data (dari MODUL 1)
2. Extract time series per market-grade
3. Perform pairwise Granger tests
4. Calculate F-statistics and p-values
5. Identify significant causal relationships
6. Generate causal matrix per grade
7. Create visualizations (network, heatmap, ranking charts)
8. Save in multiple output formats
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import json
from scipy import stats

logger = logging.getLogger(__name__)


def _benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    Apply Benjamini-Hochberg (BH) procedure to control the False Discovery Rate.

    Reference: Benjamini & Hochberg (1995), as cited in Kinnear & Mazumdar (2023)
    Section 3: the pairwise testing heuristic "essentially controls the false
    discovery rate substantially below what it would be with a threshold-based
    pairwise scheme."

    Parameters
    ----------
    p_values : np.ndarray
        Raw p-values from individual tests.
    alpha : float
        Target FDR level (default 0.05).

    Returns
    -------
    np.ndarray
        BH-adjusted p-values (same order as input).  Compare against ``alpha``
        to determine significance: ``adjusted_p < alpha`` ⟺ reject H0.
    """
    n = len(p_values)
    if n == 0:
        return np.array([])

    order = np.argsort(p_values)
    sorted_p = p_values[order]

    # BH critical values: p_i * n / rank
    adjusted = sorted_p * n / np.arange(1, n + 1)

    # Enforce monotonicity (cumulative minimum from the right)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]

    # Clip to [0, 1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    # Restore original order
    result = np.empty(n)
    result[order] = adjusted
    return result


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
        """

        mask = (df['market_id'] == market_id) & (df['grade'] == grade)
        ts = df.loc[mask, price_col].dropna().values

        if len(ts) < 10:
            return None

        return ts

    def _ols_regression(self,
                        X: np.ndarray,
                        y: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Custom OLS regression (tanpa statsmodels)."""

        X_with_intercept = np.column_stack([np.ones(len(X)), X])

        try:
            XtX = X_with_intercept.T @ X_with_intercept
            XtX_inv = np.linalg.inv(XtX)
            beta = XtX_inv @ X_with_intercept.T @ y

            y_pred = X_with_intercept @ beta
            residuals = y - y_pred
            rss = np.sum(residuals ** 2)

            tss = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (rss / tss) if tss != 0 else 0

            return beta, r_squared, rss

        except np.linalg.LinAlgError:
            return None, None, None

    def _build_lagged_matrix(self,
                             ts: np.ndarray,
                             lag_order: int) -> Tuple[np.ndarray, np.ndarray]:
        """Build lagged regression matrix untuk single variable."""

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
        """Build augmented regression matrix untuk Granger causality test."""

        min_len = min(len(y_ts), len(x_ts))
        y_ts = y_ts[:min_len]
        x_ts = x_ts[:min_len]

        n = min_len
        X_aug = np.zeros((n - lag_order, 2 * lag_order))

        for i in range(lag_order):
            X_aug[:, i] = y_ts[lag_order - i - 1:n - i - 1]

        for i in range(lag_order):
            X_aug[:, lag_order + i] = x_ts[lag_order - i - 1:n - i - 1]

        y = y_ts[lag_order:]
        return X_aug, y

    def select_optimal_lag(self,
                           y_ts: np.ndarray,
                           x_ts: Optional[np.ndarray] = None,
                           max_lag: int = 8,
                           criterion: str = 'bic') -> int:
        """
        Pilih lag optimal menggunakan information criterion (BIC atau AIC).

        Untuk setiap lag k dari 1 hingga max_lag, model restricted (y pada
        lag-nya sendiri) di-fit dan information criterion dihitung.
        Lag dengan nilai IC terendah dipilih.

        Parameters
        ----------
        y_ts : np.ndarray
            Time series target.
        x_ts : np.ndarray, optional
            Time series prediktor (jika None, hanya gunakan model restricted).
        max_lag : int
            Lag maksimum yang dicoba (default 8).
        criterion : str
            'bic' (Bayesian Information Criterion, default) atau 'aic'.

        Returns
        -------
        int
            Lag optimal (minimal 1).
        """
        best_lag = 1
        best_ic = float('inf')

        n_total = len(y_ts)

        for k in range(1, max_lag + 1):
            X, y = self._build_lagged_matrix(y_ts, k)
            n = len(y)

            if n < k + 2:
                # Tidak cukup observasi untuk lag ini
                break

            _, _, rss = self._ols_regression(X, y)
            if rss is None or rss <= 0:
                continue

            # Jumlah parameter: k lag + intercept
            num_params = k + 1

            sigma2 = rss / n
            # Hindari log(0)
            if sigma2 <= 0:
                continue

            log_lik = -n / 2.0 * np.log(sigma2)

            if criterion == 'bic':
                ic = -2 * log_lik + num_params * np.log(n)
            else:  # aic
                ic = -2 * log_lik + 2 * num_params

            if ic < best_ic:
                best_ic = ic
                best_lag = k

        return best_lag

    def granger_causality_test(self,
                               y_ts: np.ndarray,
                               x_ts: np.ndarray,
                               lag_order: int = 4,
                               significance_level: float = 0.05) -> Dict:
        """
        Perform pairwise Granger causality test.

        H0: x_ts does NOT Granger-cause y_ts.
        H1: x_ts DOES Granger-cause y_ts (lags of x improve prediction of y).

        Parameters
        ----------
        y_ts, x_ts : np.ndarray
            Time series untuk target dan prediktor.
        lag_order : int
            Jumlah lag yang digunakan.
        significance_level : float
            Threshold p-value untuk menolak H0 (default 0.05).

        Returns
        -------
        dict
            f_statistic, p_value, granger_causes, dan metrik pendukung.
        """

        X_restricted, y = self._build_lagged_matrix(y_ts, lag_order)

        if X_restricted.shape[0] < lag_order + 1:
            return {
                'f_statistic': None,
                'p_value': None,
                'granger_causes': False,
                'reason': 'insufficient_data'
            }

        beta_r, r2_r, rss_r = self._ols_regression(X_restricted, y)

        if beta_r is None:
            return {
                'f_statistic': None,
                'p_value': None,
                'granger_causes': False,
                'reason': 'regression_failed'
            }

        X_augmented, y = self._build_granger_matrix(y_ts, x_ts, lag_order)

        if X_augmented.shape[0] < 2 * lag_order + 1:
            return {
                'f_statistic': None,
                'p_value': None,
                'granger_causes': False,
                'reason': 'insufficient_data'
            }

        beta_u, r2_u, rss_u = self._ols_regression(X_augmented, y)

        if beta_u is None:
            return {
                'f_statistic': None,
                'p_value': None,
                'granger_causes': False,
                'reason': 'regression_failed'
            }

        n = len(y)
        numerator = (rss_r - rss_u) / lag_order
        denominator = rss_u / (n - 2 * lag_order - 1)

        if denominator == 0:
            f_stat = np.inf
        else:
            f_stat = numerator / denominator

        df1 = lag_order
        df2 = n - 2 * lag_order - 1
        p_value = 1 - stats.f.cdf(f_stat, df1, df2)
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
                                        price_col: str = 'price_diff',
                                        auto_lag: bool = False,
                                        max_lag: int = 8,
                                        lag_criterion: str = 'bic',
                                        significance_level: float = 0.05,
                                        apply_fdr: bool = True) -> Dict:
        """
        Test ALL pairwise Granger relationships untuk satu grade.

        Parameters
        ----------
        df : pd.DataFrame
            Preprocessed data dari MODUL 1.
        grade : str
            Grade yang diproses (misal 'low1').
        lag_order : int
            Jumlah lag yang digunakan jika ``auto_lag=False``.
        price_col : str
            Kolom harga yang digunakan (default 'price_diff').
        auto_lag : bool
            Jika True, lag optimal dipilih otomatis berbasis BIC/AIC
            untuk setiap target market menggunakan model restricted.
            Mengabaikan parameter ``lag_order``.
        max_lag : int
            Lag maksimum yang dicoba saat ``auto_lag=True`` (default 8).
        lag_criterion : str
            'bic' atau 'aic', digunakan saat ``auto_lag=True``.
        significance_level : float
            Threshold p-value nominal (default 0.05).
        apply_fdr : bool
            Jika True (default), terapkan koreksi Benjamini-Hochberg (BH)
            untuk mengendalikan False Discovery Rate (FDR) atas seluruh
            N*(N-1) pengujian pairwise sekaligus, sesuai Section 3 paper
            Kinnear & Mazumdar (2023) yang menekankan pentingnya
            pengendalian FDR.  Keputusan ``granger_causes`` didasarkan
            pada p-value terkoreksi (``p_value_bh``).

        Returns
        -------
        dict
            Hasil pairwise test dengan kunci format "M{x}→M{y}".
        """

        markets = sorted(df[df['grade'] == grade]['market_id'].unique())

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"GRANGER CAUSALITY TESTING - Grade: {grade}")
            print(f"{'='*70}")
            print(f"Markets: {markets}")
            print(f"Pairwise tests: {len(markets) * (len(markets) - 1)}")
            lag_info = f"auto ({lag_criterion.upper()}, max={max_lag})" if auto_lag else str(lag_order)
            print(f"Lag order: {lag_info}")
            if apply_fdr:
                print("FDR correction: Benjamini-Hochberg (BH)\n")
            else:
                print("FDR correction: NONE (per-test alpha, may inflate false discoveries)\n")

        results = {}

        for y_market in markets:
            # Jika auto_lag, tentukan lag optimal untuk y_market ini
            if auto_lag:
                y_ts_for_lag = self.prepare_time_series(df, y_market, grade, price_col)
                effective_lag = (
                    self.select_optimal_lag(y_ts_for_lag, max_lag=max_lag, criterion=lag_criterion)
                    if y_ts_for_lag is not None
                    else lag_order
                )
            else:
                effective_lag = lag_order

            for x_market in markets:
                if y_market == x_market:
                    continue

                y_ts = self.prepare_time_series(df, y_market, grade, price_col)
                x_ts = self.prepare_time_series(df, x_market, grade, price_col)

                if y_ts is None or x_ts is None:
                    results[f"M{x_market}→M{y_market}"] = {
                        'granger_causes': False,
                        'reason': 'insufficient_data'
                    }
                    continue

                test_result = self.granger_causality_test(
                    y_ts, x_ts, effective_lag, significance_level
                )
                results[f"M{x_market}→M{y_market}"] = test_result

        # --- Benjamini-Hochberg FDR correction (Fix #3) ---
        # Section 3, Kinnear & Mazumdar (2023): the heuristic "essentially controls
        # the false discovery rate substantially below what it would be with a
        # threshold-based pairwise scheme."  Applying BH across all N*(N-1) tests
        # follows Benjamini & Hochberg (1995), as referenced in the paper.
        if apply_fdr:
            keys_with_pval = [
                k for k, v in results.items()
                if v.get('p_value') is not None
            ]
            if keys_with_pval:
                p_values = np.array([results[k]['p_value'] for k in keys_with_pval])
                p_values_bh = _benjamini_hochberg(p_values, alpha=significance_level)

                for k, p_bh in zip(keys_with_pval, p_values_bh):
                    results[k]['p_value_bh'] = float(p_bh)
                    results[k]['granger_causes'] = bool(p_bh < significance_level)

        causal_count = sum(1 for v in results.values() if v.get('granger_causes', False))
        if self.verbose:
            for k, v in results.items():
                if v.get('granger_causes'):
                    p_display = v.get('p_value_bh', v.get('p_value'))
                    print(
                        f"  ✓ {k}: F={v.get('f_statistic', float('nan')):.4f}, "
                        f"p_raw={v.get('p_value', float('nan')):.4f}"
                        + (f", p_BH={p_display:.4f}" if apply_fdr and 'p_value_bh' in v else "")
                        + f", lag={v.get('lag_order', '?')}"
                    )
            print(f"\nSignificant causal relationships: {causal_count}/{len(results)}")

        return results

    def build_pairwise_ancestor_matrix(self,
                                       grade_results: Dict,
                                       markets: List[int]) -> np.ndarray:
        """
        Build pairwise ancestor adjacency matrix W dari test results.

        This matrix represents the set W of Kinnear & Mazumdar (2023): each
        entry W[i,j]=1 means market j is a pairwise Granger ancestor of market i
        (i.e., j pairwise Granger-causes i).

        NOTE: W may contain transitive (false positive) edges.  The final
        direct-edge graph E is obtained only after running Algorithm 1
        (transitive reduction) in Module 3.  Do NOT use this matrix as the
        definitive causal graph.

        Parameters
        ----------
        grade_results : dict
            Pairwise test results from ``test_all_pairwise_relationships``.
        markets : list of int
            Ordered list of market IDs.

        Returns
        -------
        np.ndarray
            N×N binary ancestor matrix W.
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
                matrix[i, j] = 1

        return matrix

    def build_causal_matrix(self,
                            grade_results: Dict,
                            markets: List[int]) -> np.ndarray:
        """
        .. deprecated::
            Use ``build_pairwise_ancestor_matrix`` instead.  This method
            returns the same matrix W (pairwise ancestor set), not the
            final direct-edge graph E.  The name "causal_matrix" is
            misleading because W contains transitive edges.
        """
        return self.build_pairwise_ancestor_matrix(grade_results, markets)

    def build_strength_matrix(self,
                              grade_results: Dict,
                              markets: List[int]) -> np.ndarray:
        """Build strength matrix menggunakan F-statistics."""

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
                f_stat = result.get('f_statistic', 1)
                matrix[i, j] = f_stat if f_stat != np.inf else 100

        return matrix

    def get_market_out_degree(self,
                              causal_matrix: np.ndarray,
                              markets: List[int]) -> Dict:
        """Calculate out-degree (influence) untuk setiap market."""

        out_degrees = {}
        for i, market in enumerate(markets):
            out_degrees[market] = int(np.sum(causal_matrix[:, i]))

        return out_degrees

    def get_market_in_degree(self,
                             causal_matrix: np.ndarray,
                             markets: List[int]) -> Dict:
        """Calculate in-degree (being influenced) untuk setiap market."""

        in_degrees = {}
        for i, market in enumerate(markets):
            in_degrees[market] = int(np.sum(causal_matrix[i, :]))

        return in_degrees

    def identify_market_leaders(self,
                                causal_matrix: np.ndarray,
                                markets: List[int]) -> List[int]:
        """Identify market leaders based on out-degree."""

        out_degrees = self.get_market_out_degree(causal_matrix, markets)
        ranked_markets = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)
        return [m for m, _ in ranked_markets]

    def save_as_json(self,
                     all_results: Dict,
                     output_path: str) -> Path:
        """Save results as JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, default=str)

        if self.verbose:
            print(f"✓ Saved JSON to: {output_path}")

        return output_path

    def save_as_csv(self,
                    all_results: Dict,
                    output_dir: str) -> List[Path]:
        """Save results as CSV files (one per grade)."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_files = []

        for grade, grade_data in all_results.items():
            pairwise_data = []
            for relationship, test_result in grade_data['pairwise_tests'].items():
                pairwise_data.append({
                    'Grade': grade,
                    'Relationship': relationship,
                    'F_Statistic': test_result.get('f_statistic'),
                    'P_Value': test_result.get('p_value'),
                    'Granger_Causes': test_result.get('granger_causes', False),
                    'Lag_Order': test_result.get('lag_order'),
                    'N_Observations': test_result.get('n_observations'),
                    'R2_Restricted': test_result.get('r2_restricted'),
                    'R2_Augmented': test_result.get('r2_augmented')
                })

            df_pairwise = pd.DataFrame(pairwise_data)
            pairwise_path = output_dir / f"granger_pairwise_{grade}.csv"
            df_pairwise.to_csv(pairwise_path, index=False)
            saved_files.append(pairwise_path)

            if self.verbose:
                print(f"✓ Saved pairwise CSV to: {pairwise_path}")

        return saved_files

    def save_as_excel(self,
                      all_results: Dict,
                      output_path: str) -> Path:
        """Save results as Excel with multiple sheets."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import openpyxl  # noqa: F401
        except ImportError:
            if self.verbose:
                print("Warning: openpyxl not installed, skipping Excel export")
            return None

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            summary_data = []
            for grade, grade_data in all_results.items():
                out_degrees = grade_data.get('preliminary_out_degrees', grade_data.get('out_degrees', {}))
                in_degrees = grade_data.get('preliminary_in_degrees', grade_data.get('in_degrees', {}))
                leaders = grade_data.get('preliminary_market_leaders', grade_data.get('market_leaders', []))

                for market in sorted(out_degrees.keys()):
                    summary_data.append({
                        'Grade': grade,
                        'Market': market,
                        'W_Out_Degree': out_degrees[market],
                        'W_In_Degree': in_degrees[market],
                        'Preliminary_Leader_Rank': leaders.index(market) + 1 if market in leaders else None
                    })

            df_summary = pd.DataFrame(summary_data)
            df_summary.to_excel(writer, sheet_name='Summary', index=False)

            for grade, grade_data in all_results.items():
                pairwise_data = []
                for relationship, test_result in grade_data['pairwise_tests'].items():
                    pairwise_data.append({
                        'Relationship': relationship,
                        'F_Statistic': test_result.get('f_statistic'),
                        'P_Value': test_result.get('p_value'),
                        'Granger_Causes': test_result.get('granger_causes', False),
                        'Lag_Order': test_result.get('lag_order'),
                        'N_Obs': test_result.get('n_observations')
                    })

                df_pairwise = pd.DataFrame(pairwise_data)
                sheet_name = f"Pairwise_{grade}"[:31]
                df_pairwise.to_excel(writer, sheet_name=sheet_name, index=False)

            for grade, grade_data in all_results.items():
                # Fix #4: use pairwise_ancestor_matrix, fall back to causal_matrix for compat
                ancestor_matrix = np.array(
                    grade_data.get('pairwise_ancestor_matrix', grade_data.get('causal_matrix', []))
                )
                df_matrix = pd.DataFrame(ancestor_matrix)
                sheet_name = f"AncestorMatrix_{grade}"[:31]
                df_matrix.to_excel(writer, sheet_name=sheet_name, index=True)

        if self.verbose:
            print(f"✓ Saved Excel to: {output_path}")

        return output_path

    def save_as_npz(self,
                    all_results: Dict,
                    output_path: str) -> Path:
        """Save causal matrices as NumPy binary format."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        matrices_dict = {}
        for grade, grade_data in all_results.items():
            # Fix #4: use pairwise_ancestor_matrix, fall back to causal_matrix for compat
            mat = grade_data.get('pairwise_ancestor_matrix', grade_data.get('causal_matrix', []))
            matrices_dict[f"{grade}_pairwise_ancestor_matrix"] = np.array(mat)

        np.savez(output_path, **matrices_dict)

        if self.verbose:
            print(f"✓ Saved NPZ to: {output_path}")

        return output_path

    def save_as_markdown(self,
                         all_results: Dict,
                         output_path: str) -> Path:
        """Save results as Markdown report."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Granger Causality Testing Results\n\n")
            f.write(
                "> **Note**: Market rankings below are based on the **pairwise ancestor set W** "
                "(all significant pairwise Granger relations).  W may contain transitive false "
                "positives.  Authoritative market leaders based on the direct-edge graph E "
                "are produced by Module 3 (Algorithm 1, transitive reduction).\n\n"
            )
            f.write("## Summary\n\n")

            for grade, grade_data in all_results.items():
                f.write(f"### Grade: {grade}\n\n")

                leaders = grade_data.get('preliminary_market_leaders', grade_data.get('market_leaders', []))
                out_degrees = grade_data.get('preliminary_out_degrees', grade_data.get('out_degrees', {}))
                in_degrees = grade_data.get('preliminary_in_degrees', grade_data.get('in_degrees', {}))

                f.write("#### Preliminary Market Rankings (based on W, see Note above)\n\n")
                f.write("| Rank | Market | W-Out-Degree | W-In-Degree |\n")
                f.write("|------|--------|------------|----------|\n")

                for i, market in enumerate(leaders, 1):
                    f.write(f"| {i} | M{market} | {out_degrees.get(market, 0)} | {in_degrees.get(market, 0)} |\n")

                f.write("\n#### Significant Causal Relationships\n\n")
                significant = [
                    (k, v) for k, v in grade_data['pairwise_tests'].items()
                    if v.get('granger_causes', False)
                ]
                significant = sorted(significant, key=lambda x: x[1].get('f_statistic', 0), reverse=True)

                if significant:
                    f.write("| Relationship | F-Statistic | P-Value (raw) | P-Value (BH) | Lag |\n")
                    f.write("|---|---|---|---|---|\n")
                    for relationship, test_result in significant:
                        f_stat = test_result.get('f_statistic', '-')
                        p_val = test_result.get('p_value', '-')
                        p_bh = test_result.get('p_value_bh', '-')
                        lag = test_result.get('lag_order', '-')
                        p_val_str = f"{p_val:.4f}" if isinstance(p_val, float) else str(p_val)
                        p_bh_str = f"{p_bh:.4f}" if isinstance(p_bh, float) else str(p_bh)
                        f_str = f"{f_stat:.4f}" if isinstance(f_stat, float) else str(f_stat)
                        f.write(f"| {relationship} | {f_str} | {p_val_str} | {p_bh_str} | {lag} |\n")
                else:
                    f.write("No significant relationships found.\n")

                f.write("\n")

        if self.verbose:
            print(f"✓ Saved Markdown to: {output_path}")

        return output_path

    def visualize_network_graph(self,
                                all_results: Dict,
                                output_dir: str,
                                markets: List[int],
                                format: str = 'html') -> List[Path]:
        """Create interactive network graph visualization per grade."""

        try:
            import plotly.graph_objects as go
        except ImportError:
            if self.verbose:
                print("Warning: plotly not installed, skipping network visualization")
            return []

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_files = []

        for grade, grade_data in all_results.items():
            # Fix #4: use pairwise_ancestor_matrix, fall back to causal_matrix for compat
            causal_matrix = np.array(
                grade_data.get('pairwise_ancestor_matrix', grade_data.get('causal_matrix', []))
            )
            out_degrees = grade_data.get('preliminary_out_degrees', grade_data.get('out_degrees', {}))

            n_markets = len(markets)
            pos_x = [np.cos(2 * np.pi * i / n_markets) for i in range(n_markets)]
            pos_y = [np.sin(2 * np.pi * i / n_markets) for i in range(n_markets)]

            edge_trace_x = []
            edge_trace_y = []

            for i, _ in enumerate(markets):
                for j, _ in enumerate(markets):
                    if causal_matrix[i, j] > 0:
                        edge_trace_x.extend([pos_x[j], pos_x[i], None])
                        edge_trace_y.extend([pos_y[j], pos_y[i], None])

            edge_trace = go.Scatter(
                x=edge_trace_x,
                y=edge_trace_y,
                mode='lines',
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                showlegend=False,
            )

            node_trace = go.Scatter(
                x=pos_x,
                y=pos_y,
                mode='markers+text',
                text=[f"M{m}" for m in markets],
                textposition="top center",
                hoverinfo='text',
                hovertext=[f"Market {m}<br>Influence: {out_degrees[m]}" for m in markets],
                marker=dict(
                    showscale=True,
                    color=[out_degrees[m] for m in markets],
                    size=[30 + out_degrees[m] * 10 for m in markets],
                    colorscale='YlOrRd',
                    line_width=2,
                ),
            )

            fig = go.Figure(
                data=[edge_trace, node_trace],
                layout=go.Layout(
                    title=f"Causal Network - Grade: {grade}",
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                ),
            )

            html_path = output_dir / f"network_graph_{grade}.html"
            fig.write_html(str(html_path))
            saved_files.append(html_path)

            if self.verbose:
                print(f"✓ Saved network graph (HTML) to: {html_path}")

        return saved_files

    def visualize_heatmap(self,
                          all_results: Dict,
                          output_dir: str,
                          markets: List[int]) -> List[Path]:
        """Create heatmap visualization per grade."""

        try:
            import plotly.graph_objects as go
        except ImportError:
            if self.verbose:
                print("Warning: plotly not installed, skipping heatmap visualization")
            return []

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_files = []

        for grade, grade_data in all_results.items():
            # Fix #4: use pairwise_ancestor_matrix, fall back to causal_matrix for compat
            causal_matrix = np.array(
                grade_data.get('pairwise_ancestor_matrix', grade_data.get('causal_matrix', []))
            )

            fig = go.Figure(
                data=go.Heatmap(
                    z=causal_matrix,
                    x=[f"M{m}" for m in markets],
                    y=[f"M{m}" for m in markets],
                    colorscale='Blues',
                    text=causal_matrix,
                    texttemplate='%{text}',
                    textfont={"size": 12},
                    colorbar=dict(title="Causal<br>Relationship"),
                )
            )

            fig.update_layout(
                title=f"Causal Adjacency Matrix - Grade: {grade}",
                xaxis_title="Source Market (causes →)",
                yaxis_title="Target Market (← affected)",
                height=600,
                width=700,
            )

            html_path = output_dir / f"heatmap_{grade}.html"
            fig.write_html(str(html_path))
            saved_files.append(html_path)

            if self.verbose:
                print(f"✓ Saved heatmap (HTML) to: {html_path}")

        return saved_files

    def visualize_rankings(self,
                           all_results: Dict,
                           output_dir: str) -> List[Path]:
        """Create bar chart for market leaders ranking per grade."""

        try:
            import plotly.graph_objects as go
        except ImportError:
            if self.verbose:
                print("Warning: plotly not installed, skipping ranking visualization")
            return []

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_files = []

        for grade, grade_data in all_results.items():
            out_degrees = grade_data['out_degrees']
            in_degrees = grade_data['in_degrees']
            markets = sorted(out_degrees.keys())

            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=[f"M{m}" for m in markets],
                    y=[out_degrees[m] for m in markets],
                    name='Out-Degree (Influence)',
                    marker_color='lightblue',
                )
            )
            fig.add_trace(
                go.Bar(
                    x=[f"M{m}" for m in markets],
                    y=[in_degrees[m] for m in markets],
                    name='In-Degree (Susceptibility)',
                    marker_color='lightcoral',
                )
            )

            fig.update_layout(
                title=f"Market Leaders Ranking - Grade: {grade}",
                xaxis_title="Market",
                yaxis_title="Degree Score",
                barmode='group',
                height=500,
                width=900,
            )

            html_path = output_dir / f"ranking_{grade}.html"
            fig.write_html(str(html_path))
            saved_files.append(html_path)

            if self.verbose:
                print(f"✓ Saved ranking chart (HTML) to: {html_path}")

        return saved_files

    def save_results_all_formats(self,
                                 all_results: Dict,
                                 output_dir: str,
                                 markets: List[int]) -> Dict[str, Path]:
        """Save results in ALL formats (JSON, CSV, Excel, NPZ, Markdown, Visualizations)."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = {}

        json_path = output_dir / "granger_results.json"
        self.save_as_json(all_results, str(json_path))
        saved_paths['json'] = json_path

        csv_paths = self.save_as_csv(all_results, str(output_dir))
        saved_paths['csv'] = csv_paths

        excel_path = output_dir / "granger_results.xlsx"
        self.save_as_excel(all_results, str(excel_path))
        saved_paths['excel'] = excel_path

        npz_path = output_dir / "granger_matrices.npz"
        self.save_as_npz(all_results, str(npz_path))
        saved_paths['npz'] = npz_path

        md_path = output_dir / "granger_results.md"
        self.save_as_markdown(all_results, str(md_path))
        saved_paths['markdown'] = md_path

        print(f"\n{'='*70}")
        print("GENERATING VISUALIZATIONS")
        print(f"{'='*70}")

        network_graphs = self.visualize_network_graph(all_results, str(output_dir), markets, format='html')
        saved_paths['network_graphs'] = network_graphs

        heatmaps = self.visualize_heatmap(all_results, str(output_dir), markets)
        saved_paths['heatmaps'] = heatmaps

        rankings = self.visualize_rankings(all_results, str(output_dir))
        saved_paths['rankings'] = rankings

        return saved_paths


def run_full_granger_analysis(input_file: str,
                              output_dir: str,
                              config: Dict = None) -> Dict:
    """
    Run complete Granger causality analysis dengan visualizations.

    Parameters
    ----------
    input_file : str
        Path ke preprocessed_pilot_data.csv dari MODUL 1.
    output_dir : str
        Direktori output untuk semua format hasil.
    config : dict, optional
        Konfigurasi. Kunci yang didukung:

        - ``lag_order`` (int, default 4): jumlah lag yang digunakan.
        - ``price_col`` (str, default 'price_diff'): kolom harga.
        - ``significance_level`` (float, default 0.05): threshold p-value.
        - ``auto_lag`` (bool, default False): aktifkan pemilihan lag
          otomatis berbasis BIC. Jika True, ``lag_order`` diabaikan.
        - ``max_lag`` (int, default 8): lag maksimum saat ``auto_lag=True``.
        - ``lag_criterion`` (str, default 'bic'): 'bic' atau 'aic'.
        - ``apply_fdr`` (bool, default True): terapkan Benjamini-Hochberg
          FDR correction lintas semua N*(N-1) pengujian pairwise per grade,
          sesuai Section 3 Kinnear & Mazumdar (2023).

    Returns
    -------
    dict
        Hasil lengkap per grade.  Setiap grade entry berisi:

        - ``pairwise_tests``: hasil uji pairwise, masing-masing meliputi
          ``granger_causes`` (bool, berdasarkan p-value terkoreksi BH jika
          ``apply_fdr=True``), ``f_statistic``, ``p_value`` (raw),
          dan ``p_value_bh`` (BH-adjusted, jika apply_fdr=True).
        - ``pairwise_ancestor_matrix``: matriks W (N×N) — pairwise ancestor
          set.  **Bukan** graf kausal langsung; berisi false positive transitif.
          Graf kausal langsung E diperoleh di MODUL 3 via Algorithm 1.
        - ``causal_matrix``: alias deprecated dari ``pairwise_ancestor_matrix``.
        - ``preliminary_out_degrees``, ``preliminary_in_degrees``,
          ``preliminary_market_leaders``: metrik berbasis W (preliminary only;
          lihat MODUL 3 untuk market leaders final berbasis E).
    """
    if config is None:
        config = {
            'lag_order': 4,
            'price_col': 'price_diff',
            'significance_level': 0.05,
            'auto_lag': False,
            'max_lag': 8,
            'lag_criterion': 'bic',
            'apply_fdr': True,
        }

    # Berikan nilai default untuk kunci opsional agar tidak KeyError
    lag_order = int(config.get('lag_order', 4))
    price_col = str(config.get('price_col', 'price_diff'))
    significance_level = float(config.get('significance_level', 0.05))
    auto_lag = bool(config.get('auto_lag', False))
    max_lag = int(config.get('max_lag', 8))
    lag_criterion = str(config.get('lag_criterion', 'bic'))
    apply_fdr = bool(config.get('apply_fdr', True))

    tester = GrangerCausalityTester(verbose=True)
    df = tester.load_preprocessed_data(input_file)

    all_results = {}
    markets = sorted(df['market_id'].unique().tolist())

    for grade in sorted(df['grade'].unique()):
        print(f"\n{'='*70}")
        print(f"PROCESSING GRADE: {grade}")
        print(f"{'='*70}")

        grade_results = tester.test_all_pairwise_relationships(
            df,
            grade,
            lag_order=lag_order,
            price_col=price_col,
            auto_lag=auto_lag,
            max_lag=max_lag,
            lag_criterion=lag_criterion,
            significance_level=significance_level,
            apply_fdr=apply_fdr,
        )

        # Build pairwise ancestor matrix W (Fix #4: renamed from causal_matrix)
        pairwise_ancestor_matrix = tester.build_pairwise_ancestor_matrix(grade_results, markets)
        out_degrees = tester.get_market_out_degree(pairwise_ancestor_matrix, markets)
        in_degrees = tester.get_market_in_degree(pairwise_ancestor_matrix, markets)
        market_leaders = tester.identify_market_leaders(pairwise_ancestor_matrix, markets)

        all_results[grade] = {
            'pairwise_tests': grade_results,
            # Fix #4: renamed key; causal_matrix kept as deprecated alias for backward compat
            'pairwise_ancestor_matrix': pairwise_ancestor_matrix.tolist(),
            'causal_matrix': pairwise_ancestor_matrix.tolist(),  # deprecated alias
            # Preliminary metrics based on W (ancestor set), NOT the final graph E.
            # Use Module 3 output for authoritative market leaders based on E.
            'preliminary_out_degrees': out_degrees,
            'preliminary_in_degrees': in_degrees,
            'preliminary_market_leaders': market_leaders,
            # Kept for backward compat with downstream consumers
            'out_degrees': out_degrees,
            'in_degrees': in_degrees,
            'market_leaders': market_leaders,
            'config': config,
        }

        if tester.verbose:
            print(f"\nPreliminary market rankings (from pairwise ancestor set W; "
                  f"final rankings require Module 3 transitive reduction):")
            for i, market in enumerate(market_leaders, 1):
                print(f"  {i}. Market {market} (W-out-degree: {out_degrees[market]})")

    print(f"\n{'='*70}")
    print("SAVING RESULTS IN MULTIPLE FORMATS + VISUALIZATIONS")
    print(f"{'='*70}")

    tester.save_results_all_formats(all_results, output_dir, markets)

    print(f"\n{'='*70}")
    print("GRANGER CAUSALITY ANALYSIS COMPLETE")
    print(f"{'='*70}")

    return all_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    input_file = "data/processed/preprocessed_pilot_data.csv"
    output_dir = "data/processed"

    try:
        run_full_granger_analysis(
            input_file,
            output_dir,
            config={
                'lag_order': 4,
                'price_col': 'price_diff',
                'significance_level': 0.05,
            },
        )

        print("\nAnalysis complete!")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure preprocessed data exists at: data/processed/preprocessed_pilot_data.csv")
