"""
Cointegration Analyzer

Performs cointegration tests and analysis for pairs trading.
Cointegration is stronger than correlation for pairs trading because
it implies a long-term equilibrium relationship.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from scipy import stats

# Statistical tests for cointegration
try:
    from statsmodels.tsa.stattools import coint, adfuller
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logging.warning("statsmodels not available. Install with: pip install statsmodels")

import sys
sys.path.append(str(__file__).rsplit('\\', 3)[0])

from config.settings import Settings


logger = logging.getLogger(__name__)


@dataclass
class CointegrationResult:
    """Container for cointegration test results."""
    pair: Tuple[str, str]
    is_cointegrated: bool
    test_statistic: float
    p_value: float
    critical_values: Dict[str, float]
    hedge_ratio: float
    half_life: Optional[float]
    adf_statistic: Optional[float]
    adf_p_value: Optional[float]


class CointegrationAnalyzer:
    """
    Analyzes cointegration between pairs of instruments.
    
    Features:
    - Engle-Granger two-step cointegration test
    - Johansen cointegration test
    - Half-life calculation for mean reversion
    - ADF test on spread residuals
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize the Cointegration Analyzer.
        
        Args:
            settings: System settings
        """
        self.settings = settings
        self.params = settings.spread_params
        
        if not STATSMODELS_AVAILABLE:
            raise ImportError(
                "statsmodels is required for cointegration analysis. "
                "Install with: pip install statsmodels"
            )
    
    def engle_granger_test(
        self,
        series_a: pd.Series,
        series_b: pd.Series,
        significance_level: float = 0.05
    ) -> CointegrationResult:
        """
        Perform Engle-Granger two-step cointegration test.
        
        Step 1: Estimate cointegrating regression
        Step 2: Test residuals for stationarity
        
        Args:
            series_a: First price series
            series_b: Second price series
            significance_level: Significance level for the test
            
        Returns:
            CointegrationResult with test details
        """
        # Align the series
        aligned = pd.concat([series_a, series_b], axis=1).dropna()
        if len(aligned) < 100:
            logger.warning("Insufficient data for cointegration test")
            return CointegrationResult(
                pair=(series_a.name or 'A', series_b.name or 'B'),
                is_cointegrated=False,
                test_statistic=0.0,
                p_value=1.0,
                critical_values={},
                hedge_ratio=1.0,
                half_life=None,
                adf_statistic=None,
                adf_p_value=None
            )
        
        y = aligned.iloc[:, 0].values
        x = aligned.iloc[:, 1].values
        
        # Perform cointegration test
        coint_stat, p_value, crit_values = coint(y, x)
        
        # Estimate hedge ratio via OLS
        X = np.column_stack([np.ones(len(x)), x])
        model = OLS(y, X).fit()
        hedge_ratio = model.params[1]
        
        # Calculate spread
        spread = y - hedge_ratio * x
        
        # Test spread stationarity
        adf_result = adfuller(spread, maxlag=1)
        adf_stat = adf_result[0]
        adf_p_value = adf_result[1]
        
        # Calculate half-life of mean reversion
        half_life = self._calculate_half_life(spread)
        
        # Determine if cointegrated
        is_cointegrated = p_value < significance_level
        
        return CointegrationResult(
            pair=(series_a.name or 'A', series_b.name or 'B'),
            is_cointegrated=is_cointegrated,
            test_statistic=coint_stat,
            p_value=p_value,
            critical_values={
                '1%': crit_values[0],
                '5%': crit_values[1],
                '10%': crit_values[2]
            },
            hedge_ratio=hedge_ratio,
            half_life=half_life,
            adf_statistic=adf_stat,
            adf_p_value=adf_p_value
        )
    
    def johansen_test(
        self,
        prices: pd.DataFrame,
        det_order: int = 0,
        k_ar_diff: int = 1
    ) -> Dict:
        """
        Perform Johansen cointegration test.
        
        This test can identify multiple cointegrating relationships
        and provides eigenvectors for the cointegrating vectors.
        
        Args:
            prices: DataFrame with price series
            det_order: Deterministic term order (-1: no constant, 0: constant, 1: trend)
            k_ar_diff: Number of lagged differences
            
        Returns:
            Dictionary with test results
        """
        # Ensure we have at least 2 columns
        if prices.shape[1] < 2:
            raise ValueError("Need at least 2 price series for Johansen test")
        
        # Perform test
        result = coint_johansen(prices.values, det_order, k_ar_diff)
        
        # Extract results
        trace_stats = result.lr1  # Trace statistics
        trace_crit = result.cvt   # Critical values for trace
        eigen_stats = result.lr2  # Eigenvalue statistics
        eigen_crit = result.cvm   # Critical values for eigenvalue
        
        # Number of cointegrating vectors
        # Count how many trace stats exceed critical value at 5%
        n_coint = sum(trace_stats > trace_crit[:, 1])
        
        return {
            'n_cointegrating_relations': n_coint,
            'trace_statistics': trace_stats.tolist(),
            'trace_critical_values_5pct': trace_crit[:, 1].tolist(),
            'eigenvalue_statistics': eigen_stats.tolist(),
            'eigenvalue_critical_values_5pct': eigen_crit[:, 1].tolist(),
            'eigenvectors': result.evec.tolist(),
            'eigenvalues': result.eig.tolist()
        }
    
    def _calculate_half_life(self, spread: np.ndarray) -> Optional[float]:
        """
        Calculate half-life of mean reversion using OLS.
        
        The half-life indicates how quickly the spread reverts to its mean.
        Shorter half-life = faster mean reversion = better for trading.
        
        Args:
            spread: Spread time series
            
        Returns:
            Half-life in bars, or None if not mean-reverting
        """
        # Lag the spread
        spread_lag = np.roll(spread, 1)
        spread_lag[0] = spread_lag[1]
        
        # Calculate the change
        spread_diff = spread - spread_lag
        
        # Remove first element (affected by roll)
        spread_lag = spread_lag[1:]
        spread_diff = spread_diff[1:]
        
        # Regress change on lagged level
        # spread_diff = alpha + beta * spread_lag + epsilon
        X = np.column_stack([np.ones(len(spread_lag)), spread_lag])
        model = OLS(spread_diff, X).fit()
        
        beta = model.params[1]
        
        # Half-life = -ln(2) / beta
        if beta >= 0:
            # Not mean-reverting
            return None
        
        half_life = -np.log(2) / beta
        
        return half_life
    
    def analyze_pair(
        self,
        aligned_prices: pd.DataFrame,
        significance_level: float = 0.05
    ) -> CointegrationResult:
        """
        Comprehensive cointegration analysis on a pair.
        
        Args:
            aligned_prices: DataFrame with aligned prices
            significance_level: Significance level for tests
            
        Returns:
            CointegrationResult with analysis
        """
        if aligned_prices.empty or len(aligned_prices.columns) < 2:
            raise ValueError("Need at least 2 columns of price data")
        
        col_a = aligned_prices.columns[0]
        col_b = aligned_prices.columns[1]
        
        series_a = aligned_prices[col_a]
        series_b = aligned_prices[col_b]
        
        # Set names for the series
        series_a.name = col_a
        series_b.name = col_b
        
        return self.engle_granger_test(series_a, series_b, significance_level)
    
    def screen_pairs(
        self,
        all_pairs_data: Dict[Tuple[str, str], pd.DataFrame],
        significance_level: float = 0.05,
        max_half_life: Optional[int] = None
    ) -> List[CointegrationResult]:
        """
        Screen pairs based on cointegration criteria.
        
        Args:
            all_pairs_data: Dictionary of aligned prices for each pair
            significance_level: Maximum p-value for cointegration
            max_half_life: Maximum acceptable half-life
            
        Returns:
            List of cointegrated pairs, sorted by half-life
        """
        max_half_life = max_half_life or self.params.max_half_life
        
        valid_pairs = []
        
        for pair, aligned_prices in all_pairs_data.items():
            try:
                result = self.analyze_pair(aligned_prices, significance_level)
                
                if result.is_cointegrated:
                    # Check half-life
                    if result.half_life is not None and result.half_life <= max_half_life:
                        valid_pairs.append(result)
                        logger.info(
                            f"Pair {pair} is cointegrated: "
                            f"p={result.p_value:.4f}, "
                            f"half_life={result.half_life:.1f}"
                        )
                    else:
                        logger.debug(
                            f"Pair {pair} has poor half-life: {result.half_life}"
                        )
                else:
                    logger.debug(
                        f"Pair {pair} not cointegrated: p={result.p_value:.4f}"
                    )
            except Exception as e:
                logger.error(f"Error analyzing pair {pair}: {e}")
        
        # Sort by half-life (shorter is better)
        valid_pairs.sort(
            key=lambda x: x.half_life if x.half_life else float('inf')
        )
        
        return valid_pairs
    
    def rolling_cointegration_test(
        self,
        series_a: pd.Series,
        series_b: pd.Series,
        window: int = 252
    ) -> pd.DataFrame:
        """
        Perform rolling cointegration tests.
        
        Useful for monitoring if cointegration relationship holds over time.
        
        Args:
            series_a: First price series
            series_b: Second price series
            window: Rolling window size
            
        Returns:
            DataFrame with rolling test statistics
        """
        results = []
        
        aligned = pd.concat([series_a, series_b], axis=1).dropna()
        
        for i in range(window, len(aligned)):
            window_data = aligned.iloc[i-window:i]
            
            y = window_data.iloc[:, 0].values
            x = window_data.iloc[:, 1].values
            
            try:
                coint_stat, p_value, _ = coint(y, x)
                
                results.append({
                    'time': aligned.index[i],
                    'coint_statistic': coint_stat,
                    'p_value': p_value,
                    'is_cointegrated': p_value < 0.05
                })
            except Exception:
                continue
        
        return pd.DataFrame(results).set_index('time')
    
    def check_stationarity(
        self,
        series: pd.Series,
        significance_level: float = 0.05
    ) -> Dict:
        """
        Check if a series is stationary using ADF test.
        
        Args:
            series: Time series to test
            significance_level: Significance level
            
        Returns:
            Dictionary with test results
        """
        result = adfuller(series.dropna(), maxlag=1)
        
        return {
            'is_stationary': result[1] < significance_level,
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': {
                '1%': result[4]['1%'],
                '5%': result[4]['5%'],
                '10%': result[4]['10%']
            },
            'n_lags': result[2],
            'n_observations': result[3]
        }
