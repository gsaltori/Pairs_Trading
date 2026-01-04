"""
Spread Builder Module.

Constructs and normalizes the trading spread between two assets.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
import logging


logger = logging.getLogger(__name__)


@dataclass
class SpreadMetrics:
    """Metrics for the constructed spread."""
    mean: float
    std: float
    zscore: float
    half_life: float
    hurst_exponent: float
    hedge_ratio: float
    is_mean_reverting: bool
    adf_pvalue: float


class SpreadBuilder:
    """
    Builds and analyzes the spread between two price series.
    
    Spread = Price_A - β × Price_B
    
    Where β (hedge ratio) is calculated via OLS regression.
    """
    
    def __init__(
        self,
        regression_window: int = 120,
        zscore_window: int = 60,
        recalculate_beta: bool = True
    ):
        """
        Initialize spread builder.
        
        Args:
            regression_window: Window for hedge ratio calculation
            zscore_window: Window for z-score normalization
            recalculate_beta: Whether to use rolling hedge ratio
        """
        self.regression_window = regression_window
        self.zscore_window = zscore_window
        self.recalculate_beta = recalculate_beta
    
    def build_spread(
        self,
        price_a: pd.Series,
        price_b: pd.Series,
        hedge_ratio: Optional[float] = None
    ) -> pd.Series:
        """
        Build spread series.
        
        Args:
            price_a: First price series
            price_b: Second price series
            hedge_ratio: Fixed hedge ratio (calculated if None)
            
        Returns:
            Spread series
        """
        # Align series
        common_idx = price_a.index.intersection(price_b.index)
        price_a = price_a.loc[common_idx]
        price_b = price_b.loc[common_idx]
        
        if hedge_ratio is None:
            hedge_ratio = self._calculate_hedge_ratio(price_a, price_b)
        
        spread = price_a - hedge_ratio * price_b
        
        return spread
    
    def build_spread_with_zscore(
        self,
        price_a: pd.Series,
        price_b: pd.Series
    ) -> pd.DataFrame:
        """
        Build spread with rolling hedge ratio and z-score.
        
        Args:
            price_a: First price series
            price_b: Second price series
            
        Returns:
            DataFrame with spread, zscore, hedge_ratio columns
        """
        # Align series
        common_idx = price_a.index.intersection(price_b.index)
        price_a = price_a.loc[common_idx]
        price_b = price_b.loc[common_idx]
        
        n = len(common_idx)
        
        # Initialize arrays
        spread = np.full(n, np.nan)
        zscore = np.full(n, np.nan)
        hedge_ratios = np.full(n, np.nan)
        
        # Build spread with rolling parameters
        for i in range(self.regression_window, n):
            # Calculate hedge ratio
            if self.recalculate_beta or i == self.regression_window:
                y = price_a.iloc[i-self.regression_window:i].values
                x = price_b.iloc[i-self.regression_window:i].values
                
                X = sm.add_constant(x)
                try:
                    model = OLS(y, X).fit()
                    beta = model.params[1]
                except:
                    beta = hedge_ratios[i-1] if i > self.regression_window else 1.0
                
                hedge_ratios[i] = beta
            else:
                beta = hedge_ratios[i-1]
                hedge_ratios[i] = beta
            
            # Calculate spread
            spread[i] = price_a.iloc[i] - beta * price_b.iloc[i]
        
        # Calculate z-score
        for i in range(self.regression_window + self.zscore_window, n):
            window_spread = spread[i-self.zscore_window:i]
            
            if not np.all(np.isnan(window_spread)):
                mean = np.nanmean(window_spread)
                std = np.nanstd(window_spread)
                
                if std > 0:
                    zscore[i] = (spread[i] - mean) / std
        
        result = pd.DataFrame({
            'spread': spread,
            'zscore': zscore,
            'hedge_ratio': hedge_ratios
        }, index=common_idx)
        
        return result
    
    def _calculate_hedge_ratio(
        self,
        price_a: pd.Series,
        price_b: pd.Series,
        window: Optional[int] = None
    ) -> float:
        """
        Calculate hedge ratio via OLS regression.
        
        Args:
            price_a: First price series
            price_b: Second price series
            window: Window for calculation
            
        Returns:
            Hedge ratio (β)
        """
        window = window or self.regression_window
        
        # Use last N bars
        y = price_a.tail(window).values
        x = price_b.tail(window).values
        
        if len(y) < 30:
            return 1.0
        
        X = sm.add_constant(x)
        
        try:
            model = OLS(y, X).fit()
            return model.params[1]
        except:
            return 1.0
    
    def calculate_zscore(
        self,
        spread: pd.Series,
        window: Optional[int] = None
    ) -> pd.Series:
        """
        Calculate rolling z-score of spread.
        
        Args:
            spread: Spread series
            window: Rolling window
            
        Returns:
            Z-score series
        """
        window = window or self.zscore_window
        
        rolling_mean = spread.rolling(window=window).mean()
        rolling_std = spread.rolling(window=window).std()
        
        zscore = (spread - rolling_mean) / rolling_std
        
        return zscore
    
    def get_spread_metrics(
        self,
        price_a: pd.Series,
        price_b: pd.Series
    ) -> Optional[SpreadMetrics]:
        """
        Calculate comprehensive spread metrics.
        
        Args:
            price_a: First price series
            price_b: Second price series
            
        Returns:
            SpreadMetrics or None if calculation fails
        """
        try:
            # Build spread with zscore
            spread_data = self.build_spread_with_zscore(price_a, price_b)
            
            spread = spread_data['spread'].dropna()
            zscore = spread_data['zscore'].dropna()
            hedge_ratio = spread_data['hedge_ratio'].dropna().iloc[-1]
            
            if len(spread) < 100:
                return None
            
            # Basic stats
            mean = spread.mean()
            std = spread.std()
            current_zscore = zscore.iloc[-1] if len(zscore) > 0 else 0
            
            # Half-life
            half_life = self._calculate_half_life(spread.values)
            
            # Hurst exponent
            hurst = self._calculate_hurst(spread.values)
            
            # ADF test for stationarity
            from statsmodels.tsa.stattools import adfuller
            adf_result = adfuller(spread.values)
            adf_pvalue = adf_result[1]
            
            # Determine if mean-reverting
            is_mean_reverting = (
                adf_pvalue < 0.1 and
                hurst < 0.5 and
                half_life < 100
            )
            
            return SpreadMetrics(
                mean=mean,
                std=std,
                zscore=current_zscore,
                half_life=half_life,
                hurst_exponent=hurst,
                hedge_ratio=hedge_ratio,
                is_mean_reverting=is_mean_reverting,
                adf_pvalue=adf_pvalue
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate spread metrics: {e}")
            return None
    
    def _calculate_half_life(self, spread: np.ndarray) -> float:
        """Calculate half-life of mean reversion."""
        if len(spread) < 30:
            return np.inf
        
        spread_lag = spread[:-1].reshape(-1, 1)
        spread_diff = np.diff(spread)
        
        try:
            model = OLS(spread_diff, spread_lag).fit()
            theta = model.params[0]
            
            if theta >= 0:
                return np.inf
            
            half_life = -np.log(2) / theta
            
            return max(1, min(half_life, len(spread)))
            
        except:
            return np.inf
    
    def _calculate_hurst(self, series: np.ndarray, max_lag: int = 100) -> float:
        """Calculate Hurst exponent."""
        if len(series) < max_lag * 2:
            return 0.5
        
        lags = range(2, min(max_lag, len(series) // 2))
        
        tau = []
        for lag in lags:
            diffs = series[lag:] - series[:-lag]
            tau.append(np.std(diffs))
        
        tau = np.array(tau)
        lags = np.array(list(lags))
        
        mask = tau > 0
        if mask.sum() < 10:
            return 0.5
        
        try:
            poly = np.polyfit(np.log(lags[mask]), np.log(tau[mask]), 1)
            return max(0, min(1, poly[0]))
        except:
            return 0.5
    
    def get_trading_signal(
        self,
        zscore: float,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.2
    ) -> str:
        """
        Get trading signal based on z-score.
        
        Args:
            zscore: Current z-score
            entry_threshold: Entry threshold
            exit_threshold: Exit threshold
            
        Returns:
            Signal string: 'long', 'short', 'exit_long', 'exit_short', 'hold'
        """
        if zscore <= -entry_threshold:
            return 'long'  # Spread is cheap, buy spread
        elif zscore >= entry_threshold:
            return 'short'  # Spread is expensive, sell spread
        elif abs(zscore) <= exit_threshold:
            return 'exit'  # Mean reversion complete
        else:
            return 'hold'
    
    def validate_spread(
        self,
        price_a: pd.Series,
        price_b: pd.Series,
        max_half_life: int = 50,
        min_half_life: int = 5
    ) -> Tuple[bool, str]:
        """
        Validate if spread is tradeable.
        
        Args:
            price_a: First price series
            price_b: Second price series
            max_half_life: Maximum acceptable half-life
            min_half_life: Minimum acceptable half-life
            
        Returns:
            (is_valid, reason) tuple
        """
        metrics = self.get_spread_metrics(price_a, price_b)
        
        if metrics is None:
            return False, "Failed to calculate spread metrics"
        
        if not metrics.is_mean_reverting:
            return False, "Spread is not mean-reverting"
        
        if metrics.half_life > max_half_life:
            return False, f"Half-life too long: {metrics.half_life:.1f}"
        
        if metrics.half_life < min_half_life:
            return False, f"Half-life too short: {metrics.half_life:.1f}"
        
        if metrics.hurst_exponent >= 0.5:
            return False, f"Hurst exponent suggests trending: {metrics.hurst_exponent:.2f}"
        
        return True, "Spread is valid for trading"
