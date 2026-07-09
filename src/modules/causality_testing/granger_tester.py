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
import warnings
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


def _parse_granger_flag(value) -> bool:
    """Parse native or legacy serialized Granger boolean values safely."""

    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "false"}:
            return normalized == "true"
    raise ValueError(f"Invalid granger_causes value: {value!r}")


def _make_node_id(market_id: int, grade: str) -> str:
    return f"M{int(market_id)}_{grade}"


def _split_node_id(node_id: str) -> Tuple[int, str]:
    if not str(node_id).startswith("M") or "_" not in str(node_id):
        raise ValueError(f"Invalid node identifier: {node_id}")
    market_part, grade = str(node_id)[1:].split("_", 1)
    return int(market_part), grade


def _relation_key(source_node: str, target_node: str) -> str:
    return f"{source_node}→{target_node}"


def _parse_relation_nodes(relation: str) -> Tuple[str, str]:
    rel = str(relation).replace("->", "→")
    if "→" not in rel:
        raise ValueError(f"Invalid relation format: {relation}")
    source, target = rel.split("→", 1)
    return source.strip(), target.strip()


def _to_json_compatible(value):
    if isinstance(value, dict):
        return {str(k): _to_json_compatible(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_json_compatible(v) for v in value]
    if isinstance(value, tuple):
        return [_to_json_compatible(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


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
                                       nodes: List[str]) -> np.ndarray:
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
        nodes : list of str
            Ordered list of integrated node identifiers.

        Returns
        -------
        np.ndarray
            N×N binary ancestor matrix W.
        """
        n = len(nodes)
        matrix = np.zeros((n, n))
        node_idx = {node: i for i, node in enumerate(nodes)}

        for key, result in grade_results.items():
            if _parse_granger_flag(result.get('granger_causes', False)):
                source_node, target_node = _parse_relation_nodes(key)
                i = node_idx[target_node]
                j = node_idx[source_node]
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

            This alias will be removed in a future release.
        """
        warnings.warn(
            "build_causal_matrix() is deprecated and will be removed in a future release. "
            "Use build_pairwise_ancestor_matrix() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
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
                                markets: List[str]) -> List[str]:
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
            json.dump(_to_json_compatible(all_results), f, indent=2)

        if self.verbose:
            print(f"✓ Saved JSON to: {output_path}")

        return output_path

    def save_as_csv(self,
                    all_results: Dict,
                    output_dir: str) -> List[Path]:
        """Save integrated pairwise Granger results as CSV."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pairwise_data = []

        for relationship, test_result in all_results['pairwise_tests'].items():
            source_node, target_node = _parse_relation_nodes(relationship)
            source_market, source_grade = _split_node_id(source_node)
            target_market, target_grade = _split_node_id(target_node)
            pairwise_data.append({
                'source': source_market,
                'target': target_market,
                'grade_source': source_grade,
                'grade_target': target_grade,
                'source_node': source_node,
                'target_node': target_node,
                'granger_causes': _parse_granger_flag(test_result.get('granger_causes', False)),
                'lag': test_result.get('lag_order'),
                'p_value': test_result.get('p_value'),
                'adjusted_p_value': test_result.get('p_value_bh'),
                'test_statistic': test_result.get('f_statistic'),
                'n_observations': test_result.get('n_observations'),
                'r2_restricted': test_result.get('r2_restricted'),
                'r2_augmented': test_result.get('r2_augmented'),
            })

        df_pairwise = pd.DataFrame(pairwise_data)
        pairwise_path = output_dir / "granger_pairwise.csv"
        df_pairwise.to_csv(pairwise_path, index=False)

        if self.verbose:
            print(f"✓ Saved pairwise CSV to: {pairwise_path}")

        return [pairwise_path]

    def save_as_excel(self,
                      all_results: Dict,
                      output_path: str) -> Path:
        """Save integrated results as Excel with multiple sheets."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import openpyxl  # noqa: F401
        except ImportError:
            if self.verbose:
                print("Warning: openpyxl not installed, skipping Excel export")
            return None

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            node_rows = []
            leaders = all_results.get('preliminary_market_leaders', [])
            out_degrees = all_results.get('preliminary_out_degrees', {})
            in_degrees = all_results.get('preliminary_in_degrees', {})
            for rank, node in enumerate(leaders, start=1):
                market_id, grade = _split_node_id(node)
                node_rows.append({
                    'node': node,
                    'market_id': market_id,
                    'grade': grade,
                    'W_out_degree': out_degrees.get(node, 0),
                    'W_in_degree': in_degrees.get(node, 0),
                    'preliminary_leader_rank': rank,
                })

            pd.DataFrame(node_rows).to_excel(writer, sheet_name='Summary', index=False)
            self.save_as_csv(all_results, str(output_path.parent))
            pd.read_csv(output_path.parent / "granger_pairwise.csv").to_excel(
                writer,
                sheet_name='PairwiseTests',
                index=False,
            )

            node_labels = [node['node_id'] for node in all_results['nodes']]
            ancestor_matrix = np.array(
                all_results.get('pairwise_ancestor_matrix', all_results.get('causal_matrix', []))
            )
            df_matrix = pd.DataFrame(ancestor_matrix, index=node_labels, columns=node_labels)
            df_matrix.to_excel(writer, sheet_name='AncestorMatrix', index=True)

        if self.verbose:
            print(f"✓ Saved Excel to: {output_path}")

        return output_path

    def save_as_npz(self,
                    all_results: Dict,
                    output_path: str) -> Path:
        """Save integrated ancestor matrix as NumPy binary format."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        matrices_dict = {
            "integrated_pairwise_ancestor_matrix": np.array(
                all_results.get('pairwise_ancestor_matrix', all_results.get('causal_matrix', []))
            ),
            "node_labels": np.array([node['node_id'] for node in all_results['nodes']], dtype=object),
        }

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
                "> **Integrated workflow**: Module 2 produces one integrated ancestor set W "
                "over nodes `(market_id, grade)`. Final direct edges are recovered only in "
                "Module 3 via Algorithm 1.\n\n"
            )
            f.write("## Summary\n\n")
            f.write("### Preliminary Node Rankings (based on W)\n\n")
            f.write("| Rank | Node | Market | Grade | W-Out-Degree | W-In-Degree |\n")
            f.write("|---|---|---:|---|---:|---:|\n")
            leaders = all_results.get('preliminary_market_leaders', [])
            out_degrees = all_results.get('preliminary_out_degrees', {})
            in_degrees = all_results.get('preliminary_in_degrees', {})
            for rank, node in enumerate(leaders, start=1):
                market_id, grade = _split_node_id(node)
                f.write(
                    f"| {rank} | {node} | {market_id} | {grade} | "
                    f"{out_degrees.get(node, 0)} | {in_degrees.get(node, 0)} |\n"
                )

            f.write("\n### Significant Pairwise Relationships\n\n")
            significant = [
                (k, v) for k, v in all_results['pairwise_tests'].items()
                if _parse_granger_flag(v.get('granger_causes', False))
            ]
            significant = sorted(significant, key=lambda x: x[1].get('f_statistic', 0), reverse=True)

            if significant:
                f.write("| Relationship | F-Statistic | P-Value (raw) | P-Value (BH) | Lag |\n")
                f.write("|---|---:|---:|---:|---:|\n")
                for relationship, test_result in significant:
                    f.write(
                        f"| {relationship} | {float(test_result.get('f_statistic', 0.0)):.4f} | "
                        f"{float(test_result.get('p_value', 1.0)):.6f} | "
                        f"{float(test_result.get('p_value_bh', 1.0)):.6f} | "
                        f"{int(test_result.get('lag_order', 0) or 0)} |\n"
                    )
            else:
                f.write("No significant relationships found.\n")

        if self.verbose:
            print(f"✓ Saved Markdown to: {output_path}")

        return output_path

    def visualize_network_graph(self,
                                all_results: Dict,
                                output_dir: str,
                                markets: List[str],
                                format: str = 'html') -> List[Path]:
        """Create integrated ancestor-network visualization."""

        try:
            import plotly.graph_objects as go
        except ImportError:
            if self.verbose:
                print("Warning: plotly not installed, skipping network visualization")
            return []

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_files = []

        causal_matrix = np.array(
            all_results.get('pairwise_ancestor_matrix', all_results.get('causal_matrix', []))
        )
        out_degrees = all_results.get('preliminary_out_degrees', all_results.get('out_degrees', {}))
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
            text=markets,
            textposition="top center",
            hoverinfo='text',
            hovertext=[f"{node}<br>Influence: {out_degrees.get(node, 0)}" for node in markets],
            marker=dict(
                showscale=True,
                color=[out_degrees.get(node, 0) for node in markets],
                size=[20 + out_degrees.get(node, 0) * 4 for node in markets],
                colorscale='YlOrRd',
                line_width=2,
            ),
        )

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title="Integrated Pairwise Ancestor Set W",
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            ),
        )

        html_path = output_dir / "network_graph_integrated.html"
        fig.write_html(str(html_path))
        saved_files.append(html_path)

        if self.verbose:
            print(f"✓ Saved network graph (HTML) to: {html_path}")

        return saved_files

    def visualize_heatmap(self,
                          all_results: Dict,
                          output_dir: str,
                          markets: List[str]) -> List[Path]:
        """Create integrated ancestor-matrix heatmap."""

        try:
            import plotly.graph_objects as go
        except ImportError:
            if self.verbose:
                print("Warning: plotly not installed, skipping heatmap visualization")
            return []

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_files = []

        causal_matrix = np.array(
            all_results.get('pairwise_ancestor_matrix', all_results.get('causal_matrix', []))
        )

        fig = go.Figure(
            data=go.Heatmap(
                z=causal_matrix,
                x=markets,
                y=markets,
                colorscale='Blues',
                text=causal_matrix,
                texttemplate='%{text}',
                textfont={"size": 12},
                colorbar=dict(title="Ancestor<br>Relation"),
            )
        )

        fig.update_layout(
            title="Integrated Pairwise Ancestor Matrix W",
            xaxis_title="Source node",
            yaxis_title="Target node",
            height=800,
            width=900,
        )

        html_path = output_dir / "heatmap_integrated.html"
        fig.write_html(str(html_path))
        saved_files.append(html_path)

        if self.verbose:
            print(f"✓ Saved heatmap (HTML) to: {html_path}")

        return saved_files

    def visualize_rankings(self,
                           all_results: Dict,
                           output_dir: str) -> List[Path]:
        """Create integrated ranking chart."""

        try:
            import plotly.graph_objects as go
        except ImportError:
            if self.verbose:
                print("Warning: plotly not installed, skipping ranking visualization")
            return []

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_files = []

        out_degrees = all_results['out_degrees']
        in_degrees = all_results['in_degrees']
        markets = list(all_results['preliminary_market_leaders'])

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=markets,
                y=[out_degrees[m] for m in markets],
                name='Out-Degree (W)',
                marker_color='lightblue',
            )
        )
        fig.add_trace(
            go.Bar(
                x=markets,
                y=[in_degrees[m] for m in markets],
                name='In-Degree (W)',
                marker_color='lightcoral',
            )
        )

        fig.update_layout(
            title="Integrated Preliminary Node Ranking",
            xaxis_title="Node",
            yaxis_title="Degree",
            barmode='group',
            height=500,
            width=1200,
        )

        html_path = output_dir / "ranking_integrated.html"
        fig.write_html(str(html_path))
        saved_files.append(html_path)

        if self.verbose:
            print(f"✓ Saved ranking chart (HTML) to: {html_path}")

        return saved_files

    def save_results_all_formats(self,
                                 all_results: Dict,
                                 output_dir: str,
                                 markets: List[str]) -> Dict[str, Path]:
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
        Hasil lengkap terintegrasi untuk seluruh node ``(market_id, grade)``.
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

    node_frame = (
        df[['market_id', 'grade']]
        .drop_duplicates()
        .sort_values(['market_id', 'grade'])
        .reset_index(drop=True)
    )
    node_records = [
        {
            'node_id': _make_node_id(row.market_id, row.grade),
            'market_id': int(row.market_id),
            'grade': row.grade,
        }
        for row in node_frame.itertuples(index=False)
    ]
    node_labels = [record['node_id'] for record in node_records]
    pairwise_results: Dict[str, Dict] = {}

    print(f"\n{'='*70}")
    print("PROCESSING INTEGRATED NODE GRAPH")
    print(f"{'='*70}")
    print(f"Nodes: {len(node_labels)}")
    print(f"Pairwise tests: {len(node_labels) * (len(node_labels) - 1)}")

    node_series = {
        node['node_id']: tester.prepare_time_series(df, node['market_id'], node['grade'], price_col)
        for node in node_records
    }

    for target in node_records:
        target_node = target['node_id']
        target_ts = node_series[target_node]
        if auto_lag and target_ts is not None:
            effective_lag = tester.select_optimal_lag(
                target_ts,
                max_lag=max_lag,
                criterion=lag_criterion,
            )
        else:
            effective_lag = lag_order

        for source in node_records:
            source_node = source['node_id']
            if source_node == target_node:
                continue

            source_ts = node_series[source_node]
            relation = _relation_key(source_node, target_node)
            if target_ts is None or source_ts is None:
                pairwise_results[relation] = {
                    'granger_causes': False,
                    'reason': 'insufficient_data',
                }
                continue

            pairwise_results[relation] = tester.granger_causality_test(
                target_ts,
                source_ts,
                effective_lag,
                significance_level,
            )

    if apply_fdr:
        keys_with_pval = [
            key for key, value in pairwise_results.items()
            if value.get('p_value') is not None
        ]
        if keys_with_pval:
            p_values = np.array([pairwise_results[key]['p_value'] for key in keys_with_pval])
            p_values_bh = _benjamini_hochberg(p_values, alpha=significance_level)
            for key, adjusted in zip(keys_with_pval, p_values_bh):
                pairwise_results[key]['p_value_bh'] = float(adjusted)
                pairwise_results[key]['granger_causes'] = bool(adjusted < significance_level)

    pairwise_ancestor_matrix = tester.build_pairwise_ancestor_matrix(pairwise_results, node_labels)
    out_degrees = tester.get_market_out_degree(pairwise_ancestor_matrix, node_labels)
    in_degrees = tester.get_market_in_degree(pairwise_ancestor_matrix, node_labels)
    market_leaders = tester.identify_market_leaders(pairwise_ancestor_matrix, node_labels)

    all_results = {
        'analysis_type': 'integrated',
        'nodes': node_records,
        'pairwise_tests': pairwise_results,
        'pairwise_ancestor_matrix': pairwise_ancestor_matrix.tolist(),
        'causal_matrix': pairwise_ancestor_matrix.tolist(),
        'preliminary_out_degrees': out_degrees,
        'preliminary_in_degrees': in_degrees,
        'preliminary_market_leaders': market_leaders,
        'out_degrees': out_degrees,
        'in_degrees': in_degrees,
        'market_leaders': market_leaders,
        'config': config,
    }

    if tester.verbose:
        print("\nPreliminary node rankings (from integrated pairwise ancestor set W):")
        for i, node in enumerate(market_leaders[:10], 1):
            print(f"  {i}. {node} (W-out-degree: {out_degrees[node]})")

    print(f"\n{'='*70}")
    print("SAVING RESULTS IN MULTIPLE FORMATS + VISUALIZATIONS")
    print(f"{'='*70}")

    tester.save_results_all_formats(all_results, output_dir, node_labels)

    print(f"\n{'='*70}")
    print("GRANGER CAUSALITY ANALYSIS COMPLETE")
    print(f"{'='*70}")

    return all_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    input_file = "data/processed/module_01/preprocessed_pilot_data.csv"
    output_dir = "data/processed/module_02"

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
        print("Please ensure preprocessed data exists at: data/processed/module_01/preprocessed_pilot_data.csv")
