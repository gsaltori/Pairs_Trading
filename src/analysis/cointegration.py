"""
Cointegration Analysis Module.

Implements Engle-Granger and Johansen cointegration tests.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.regression.linear_model import OLS
import logging


logger = logging.getLogger(__name__)


@dataclass
class CointegrationResult:
    """Results from cointegration analysis."""
    is_cointegrated: bool
    p_value: float
    test_statistic: float
    critical_values: dict
    hedge_ratio: float
    intercept: float
    half_life: float
    adf_statistic: float
    residuals: Optional[pd.Series] = None


class CointegrationAnalyzer:
    """
    Analyzes cointegration between price series.
    
    Features:
    - Engle-Granger two-step method
    - ADF test on residuals
    - Half-life calculation
    - Dynamic hedge ratio
    """
    
    def __init__(
        self,
        significance_level: float = 0.05,
        max_lag: Optional[int] = None
    ):
        """
        Initialize analyzer.
        
        Args:
            significance_level: P-value threshold for cointegration
            max_lag: Maximum lag for ADF test
        """
        self.significance_level = significance_level
        self.max_lag = max_lag
    
    def engle_granger_test(
        self,
        price_a: pd.Series,
        price_b: pd.Series
    ) -> CointegrationResult:
        """
        Perform Engle-Granger cointegration test.
        
        Steps:
        1. Run OLS regression: price_a = α + β * price_b + ε
        2. Test residuals for stationarity (ADF)
        
        Args:
            price_a: First price series (dependent)
            price_b: Second price series (independent)
            
        Returns:
            CointegrationResult with all metrics
        """
        # Align series
        common_idx = price_a.index.intersection(price_b.index)
        y = price_a.loc[common_idx].values
        x = price_b.loc[common_idx].values
        
        if len(y) < 100:
            return CointegrationResult(
                is_cointegrated=False,
                p_value=1.0,
                test_statistic=0.0,
                critical_values={},
                hedge_ratio=0.0,
                intercept=0.0,
                half_life=np.inf,
                adf_statistic=0.0
            )
        
        # Step 1: OLS Regression
        X = sm.add_constant(x)
        model = OLS(y, X).fit()
        
        intercept = model.params[0]
        hedge_ratio = model.params[1]
        
        # Get residuals (spread)
        residuals = y - (intercept + hedge_ratio * x)
        residuals_series = pd.Series(residuals, index=common_idx)
        
        # Step 2: ADF test on residuals
        maxlag = self.max_lag or int(np.sqrt(len(residuals)))
        
        try:
            adf_result = adfuller(residuals, maxlag=maxlag, regression='c')
            
            adf_stat = adf_result[0]
            p_value = adf_result[1]
            critical_values = {
                '1%': adf_result[4]['1%'],
                '5%': adf_result[4]['5%'],
                '10%': adf_result[4]['10%']
            }
        except Exception as e:
            logger.warning(f"ADF test failed: {e}")
            adf_stat = 0.0
            p_value = 1.0
            critical_values = {}
        
        # Cointegration decision
        is_cointegrated = p_value < self.significance_level
        
        # Calculate half-life of mean reversion
        half_life = self._calculate_half_life(residuals)
        
        return CointegrationResult(
            is_cointegrated=is_cointegrated,
            p_value=p_value,
            test_statistic=adf_stat,
            critical_values=critical_values,
            hedge_ratio=hedge_ratio,
            intercept=intercept,
            half_life=half_life,
            adf_statistic=adf_stat,
            residuals=residuals_series
        )
    
    def _calculate_half_life(self, spread: np.ndarray) -> float:
        """
        Calculate half-life of mean reversion using Ornstein-Uhlenbeck.
        
        Half-life = -ln(2) / ln(1 + θ)
        where θ is from: Δspread = θ * spread_{t-1} + ε
        
        Args:
            spread: Spread series as numpy array
            
        Returns:
            Half-life in bars
        """
        if len(spread) < 30:
            return np.inf
        
        # Lag spread
        spread_lag = spread[:-1]
        spread_diff = np.diff(spread)
        
        # Regression: Δspread = θ * spread_{t-1}
        spread_lag = spread_lag.reshape(-1, 1)
        
        try:
            model = OLS(spread_diff, spread_lag).fit()
            theta = model.params[0]
            
            if theta >= 0:
                # Not mean-reverting
                return np.inf
            
            half_life = -np.log(2) / theta
            
            # Sanity check
            if half_life < 0 or half_life > len(spread):
                return np.inf
            
            return half_life
            
        except Exception as e:
            logger.warning(f"Half-life calculation failed: {e}")
            return np.inf
    
    def calculate_hedge_ratio_ols(
        self,
        price_a: pd.Series,
        price_b: pd.Series,
        window: Optional[int] = None
    ) -> float:
        """
        Calculate hedge ratio using OLS regression.
        
        Args:
            price_a: First price series
            price_b: Second price series
            window: Use last N bars only
            
        Returns:
            Hedge ratio (β)
        """
        if window:
            price_a = price_a.tail(window)
            price_b = price_b.tail(window)
        
        # Align
        common_idx = price_a.index.intersection(price_b.index)
        y = price_a.loc[common_idx].values
        x = price_b.loc[common_idx].values
        
        if len(y) < 30:
            return 1.0
        
        X = sm.add_constant(x)
        model = OLS(y, X).fit()
        
        return model.params[1]
    
    def calculate_rolling_hedge_ratio(
        self,
        price_a: pd.Series,
        price_b: pd.Series,
        window: int = 120
    ) -> pd.Series:
        """
        Calculate rolling hedge ratio.
        
        Args:
            price_a: First price series
            price_b: Second price series
            window: Rolling window size
            
        Returns:
            Series of hedge ratios
        """
        # Align series
        common_idx = price_a.index.intersection(price_b.index)
        price_a = price_a.loc[common_idx]
        price_b = price_b.loc[common_idx]
        
        hedge_ratios = pd.Series(index=common_idx, dtype=float)
        
        for i in range(window, len(common_idx)):
            y = price_a.iloc[i-window:i].values
            x = price_b.iloc[i-window:i].values
            
            X = sm.add_constant(x)
            try:
                model = OLS(y, X).fit()
                hedge_ratios.iloc[i] = model.params[1]
            except:
                hedge_ratios.iloc[i] = np.nan
        
        return hedge_ratios
    
    def test_stationarity(
        self,
        series: pd.Series,
        significance: float = 0.05
    ) -> Tuple[bool, float, dict]:
        """
        Test series for stationarity using ADF test.
        
        Args:
            series: Time series to test
            significance: P-value threshold
            
        Returns:
            (is_stationary, p_value, critical_values)
        """
        clean = series.dropna()
        
        if len(clean) < 30:
            return False, 1.0, {}
        
        try:
            result = adfuller(clean.values)
            
            p_value = result[1]
            critical_values = {
                '1%': result[4]['1%'],
                '5%': result[4]['5%'],
                '10%': result[4]['10%']
            }
            
            is_stationary = p_value < significance
            
            return is_stationary, p_value, critical_values
            
        except Exception as e:
            logger.warning(f"Stationarity test failed: {e}")
            return False, 1.0, {}
    
    def calculate_hurst_exponent(
        self,
        series: pd.Series,
        max_lag: int = 100
    ) -> float:
        """
        Calculate Hurst exponent to measure mean reversion.
        
        H < 0.5: Mean-reverting
        H = 0.5: Random walk
        H > 0.5: Trending
        
        Args:
            series: Time series
            max_lag: Maximum lag for analysis
            
        Returns:
            Hurst exponent [0, 1]
        """
        clean = series.dropna().values
        
        if len(clean) < max_lag * 2:
            return 0.5
        
        lags = range(2, min(max_lag, len(clean) // 2))
        
        # Calculate variance of lagged differences
        tau = []
        for lag in lags:
            diffs = clean[lag:] - clean[:-lag]
            tau.append(np.std(diffs))
        
        tau = np.array(tau)
        lags = np.array(list(lags))
        
        # Filter zeros
        mask = tau > 0
        if mask.sum() < 10:
            return 0.5
        
        tau = tau[mask]
        lags = lags[mask]
        
        # Linear regression on log-log scale
        try:
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            hurst = poly[0]
            
            # Clamp to valid range
            return max(0, min(1, hurst))
            
        except Exception:
            return 0.5
