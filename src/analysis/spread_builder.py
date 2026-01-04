"""
Spread Builder

Constructs and analyzes the spread between two instruments
for pairs trading. Includes rolling regression for dynamic
hedge ratio calculation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import logging

try:
    from statsmodels.regression.linear_model import OLS
    from statsmodels.regression.rolling import RollingOLS
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

import sys
sys.path.append(str(__file__).rsplit('\\', 3)[0])

from config.settings import Settings, SpreadParameters


logger = logging.getLogger(__name__)


@dataclass
class SpreadMetrics:
    """Container for spread statistics."""
    mean: float
    std: float
    current_value: float
    zscore: float
    half_life: Optional[float]
    hedge_ratio: float
    hurst_exponent: Optional[float]


class SpreadBuilder:
    """
    Builds and analyzes spread between two instruments.
    
    Features:
    - Static and dynamic hedge ratio calculation
    - Rolling regression for adaptive hedge ratios
    - Z-score calculation
    - Spread statistics and metrics
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize the Spread Builder.
        
        Args:
            settings: System settings
        """
        self.settings = settings
        self.params = settings.spread_params
    
    def calculate_hedge_ratio_ols(
        self,
        series_a: pd.Series,
        series_b: pd.Series
    ) -> float:
        """
        Calculate hedge ratio using Ordinary Least Squares.
        
        Spread = Price_A - beta * Price_B
        
        Args:
            series_a: Dependent variable (Price A)
            series_b: Independent variable (Price B)
            
        Returns:
            Hedge ratio (beta)
        """
        # Align the series
        aligned = pd.concat([series_a, series_b], axis=1).dropna()
        
        if len(aligned) < 10:
            return 1.0  # Default to equal weighting
        
        y = aligned.iloc[:, 0].values
        x = aligned.iloc[:, 1].values
        
        # Add constant for intercept
        X = np.column_stack([np.ones(len(x)), x])
        
        model = OLS(y, X).fit()
        
        return model.params[1]  # Return the beta coefficient
    
    def calculate_rolling_hedge_ratio(
        self,
        series_a: pd.Series,
        series_b: pd.Series,
        window: Optional[int] = None
    ) -> pd.Series:
        """
        Calculate rolling hedge ratio using rolling OLS.
        
        Args:
            series_a: Dependent variable
            series_b: Independent variable
            window: Rolling window size
            
        Returns:
            Series of rolling hedge ratios
        """
        window = window or self.params.regression_window
        
        aligned = pd.concat([series_a, series_b], axis=1).dropna()
        
        if len(aligned) < window:
            logger.warning(f"Insufficient data for rolling regression (need {window}, got {len(aligned)})")
            return pd.Series(dtype=float)
        
        y = aligned.iloc[:, 0]
        x = aligned.iloc[:, 1]
        
        # Add constant
        X = pd.concat([pd.Series(1.0, index=x.index, name='const'), x], axis=1)
        
        if STATSMODELS_AVAILABLE:
            # Use statsmodels RollingOLS
            model = RollingOLS(y, X, window=window)
            results = model.fit()
            
            # Extract beta (second parameter after constant)
            hedge_ratios = results.params.iloc[:, 1]
        else:
            # Manual rolling calculation
            hedge_ratios = pd.Series(index=aligned.index, dtype=float)
            
            for i in range(window, len(aligned) + 1):
                window_y = y.iloc[i-window:i].values
                window_x = X.iloc[i-window:i].values
                
                try:
                    beta = np.linalg.lstsq(window_x, window_y, rcond=None)[0][1]
                    hedge_ratios.iloc[i-1] = beta
                except Exception:
                    hedge_ratios.iloc[i-1] = np.nan
        
        return hedge_ratios.dropna()
    
    def build_spread(
        self,
        series_a: pd.Series,
        series_b: pd.Series,
        hedge_ratio: Optional[float] = None,
        use_rolling: bool = True
    ) -> pd.DataFrame:
        """
        Build the spread between two instruments.
        
        Spread = Price_A - hedge_ratio * Price_B
        
        Args:
            series_a: First price series
            series_b: Second price series
            hedge_ratio: Static hedge ratio (if not using rolling)
            use_rolling: Whether to use rolling hedge ratio
            
        Returns:
            DataFrame with spread and hedge ratio
        """
        aligned = pd.concat([series_a, series_b], axis=1).dropna()
        
        if use_rolling:
            hedge_ratios = self.calculate_rolling_hedge_ratio(
                aligned.iloc[:, 0],
                aligned.iloc[:, 1]
            )
            
            # Align with original data
            result = pd.DataFrame(index=hedge_ratios.index)
            result['hedge_ratio'] = hedge_ratios
            result['price_a'] = aligned.iloc[:, 0].loc[hedge_ratios.index]
            result['price_b'] = aligned.iloc[:, 1].loc[hedge_ratios.index]
            result['spread'] = result['price_a'] - result['hedge_ratio'] * result['price_b']
        else:
            if hedge_ratio is None:
                hedge_ratio = self.calculate_hedge_ratio_ols(
                    aligned.iloc[:, 0],
                    aligned.iloc[:, 1]
                )
            
            result = pd.DataFrame(index=aligned.index)
            result['hedge_ratio'] = hedge_ratio
            result['price_a'] = aligned.iloc[:, 0]
            result['price_b'] = aligned.iloc[:, 1]
            result['spread'] = result['price_a'] - hedge_ratio * result['price_b']
        
        return result
    
    def calculate_zscore(
        self,
        spread: pd.Series,
        window: Optional[int] = None
    ) -> pd.Series:
        """
        Calculate rolling Z-score of the spread.
        
        Z = (spread - mean) / std
        
        Args:
            spread: Spread series
            window: Window for rolling mean/std
            
        Returns:
            Z-score series
        """
        window = window or self.params.zscore_window
        
        rolling_mean = spread.rolling(window=window).mean()
        rolling_std = spread.rolling(window=window).std()
        
        zscore = (spread - rolling_mean) / rolling_std
        
        return zscore
    
    def build_spread_with_zscore(
        self,
        series_a: pd.Series,
        series_b: pd.Series,
        use_rolling: bool = True
    ) -> pd.DataFrame:
        """
        Build spread with Z-score and all metrics.
        
        Args:
            series_a: First price series
            series_b: Second price series
            use_rolling: Whether to use rolling hedge ratio
            
        Returns:
            DataFrame with spread, zscore, and metrics
        """
        # Build spread
        result = self.build_spread(series_a, series_b, use_rolling=use_rolling)
        
        if result.empty:
            return result
        
        # Calculate Z-score
        result['zscore'] = self.calculate_zscore(result['spread'])
        
        # Rolling mean and std
        window = self.params.zscore_window
        result['spread_mean'] = result['spread'].rolling(window=window).mean()
        result['spread_std'] = result['spread'].rolling(window=window).std()
        
        return result.dropna()
    
    def get_spread_metrics(
        self,
        spread_data: pd.DataFrame
    ) -> SpreadMetrics:
        """
        Calculate comprehensive spread metrics.
        
        Args:
            spread_data: DataFrame from build_spread_with_zscore
            
        Returns:
            SpreadMetrics object
        """
        if spread_data.empty:
            raise ValueError("Empty spread data")
        
        spread = spread_data['spread']
        
        # Calculate half-life
        half_life = self._calculate_half_life(spread.values)
        
        # Calculate Hurst exponent (measure of mean reversion)
        hurst = self._calculate_hurst_exponent(spread.values)
        
        return SpreadMetrics(
            mean=spread_data['spread_mean'].iloc[-1] if 'spread_mean' in spread_data else spread.mean(),
            std=spread_data['spread_std'].iloc[-1] if 'spread_std' in spread_data else spread.std(),
            current_value=spread.iloc[-1],
            zscore=spread_data['zscore'].iloc[-1] if 'zscore' in spread_data else 0.0,
            half_life=half_life,
            hedge_ratio=spread_data['hedge_ratio'].iloc[-1],
            hurst_exponent=hurst
        )
    
    def _calculate_half_life(self, spread: np.ndarray) -> Optional[float]:
        """Calculate half-life of mean reversion."""
        spread_lag = np.roll(spread, 1)
        spread_lag[0] = spread_lag[1]
        
        spread_diff = spread - spread_lag
        
        spread_lag = spread_lag[1:]
        spread_diff = spread_diff[1:]
        
        # Regress
        X = np.column_stack([np.ones(len(spread_lag)), spread_lag])
        
        try:
            if STATSMODELS_AVAILABLE:
                model = OLS(spread_diff, X).fit()
                beta = model.params[1]
            else:
                beta = np.linalg.lstsq(X, spread_diff, rcond=None)[0][1]
            
            if beta >= 0:
                return None
            
            return -np.log(2) / beta
        except Exception:
            return None
    
    def _calculate_hurst_exponent(
        self,
        series: np.ndarray,
        max_lag: int = 100
    ) -> Optional[float]:
        """
        Calculate Hurst exponent using R/S analysis.
        
        H < 0.5: Mean reverting
        H = 0.5: Random walk
        H > 0.5: Trending
        
        Args:
            series: Time series data
            max_lag: Maximum lag for calculation
            
        Returns:
            Hurst exponent or None
        """
        if len(series) < max_lag:
            return None
        
        try:
            lags = range(2, min(max_lag, len(series) // 2))
            
            # Calculate R/S for each lag
            rs_values = []
            
            for lag in lags:
                # Split series into sub-series
                n_subseries = len(series) // lag
                rs_subseries = []
                
                for i in range(n_subseries):
                    subseries = series[i * lag:(i + 1) * lag]
                    
                    # Mean-adjusted cumulative sum
                    mean_adj = subseries - np.mean(subseries)
                    cumsum = np.cumsum(mean_adj)
                    
                    # Range
                    R = np.max(cumsum) - np.min(cumsum)
                    
                    # Standard deviation
                    S = np.std(subseries)
                    
                    if S > 0:
                        rs_subseries.append(R / S)
                
                if rs_subseries:
                    rs_values.append(np.mean(rs_subseries))
            
            if len(rs_values) < 2:
                return None
            
            # Fit log-log regression
            log_lags = np.log(list(lags)[:len(rs_values)])
            log_rs = np.log(rs_values)
            
            # Linear regression
            slope, _ = np.polyfit(log_lags, log_rs, 1)
            
            return slope
            
        except Exception as e:
            logger.debug(f"Error calculating Hurst exponent: {e}")
            return None
    
    def update_spread(
        self,
        current_spread_data: pd.DataFrame,
        new_price_a: float,
        new_price_b: float,
        timestamp: pd.Timestamp
    ) -> pd.DataFrame:
        """
        Update spread with new price data.
        
        Args:
            current_spread_data: Existing spread data
            new_price_a: New price for instrument A
            new_price_b: New price for instrument B
            timestamp: Timestamp for new data
            
        Returns:
            Updated spread DataFrame
        """
        # Get current hedge ratio
        current_hedge = current_spread_data['hedge_ratio'].iloc[-1]
        
        # Calculate new spread
        new_spread = new_price_a - current_hedge * new_price_b
        
        # Calculate new z-score
        window = self.params.zscore_window
        recent_spreads = current_spread_data['spread'].iloc[-window:]
        mean = recent_spreads.mean()
        std = recent_spreads.std()
        new_zscore = (new_spread - mean) / std if std > 0 else 0.0
        
        # Create new row
        new_row = pd.DataFrame({
            'hedge_ratio': [current_hedge],
            'price_a': [new_price_a],
            'price_b': [new_price_b],
            'spread': [new_spread],
            'zscore': [new_zscore],
            'spread_mean': [mean],
            'spread_std': [std]
        }, index=[timestamp])
        
        # Append
        result = pd.concat([current_spread_data, new_row])
        
        return result
    
    def recalculate_hedge_ratio(
        self,
        spread_data: pd.DataFrame,
        window: Optional[int] = None
    ) -> float:
        """
        Recalculate hedge ratio based on recent data.
        
        Args:
            spread_data: Spread data with prices
            window: Window for recalculation
            
        Returns:
            New hedge ratio
        """
        window = window or self.params.regression_window
        
        recent_data = spread_data.iloc[-window:]
        
        return self.calculate_hedge_ratio_ols(
            recent_data['price_a'],
            recent_data['price_b']
        )
    
    def is_spread_tradeable(
        self,
        metrics: SpreadMetrics
    ) -> Tuple[bool, str]:
        """
        Check if spread is currently tradeable.
        
        Args:
            metrics: SpreadMetrics object
            
        Returns:
            Tuple of (is_tradeable, reason)
        """
        # Check half-life
        if metrics.half_life is not None:
            if metrics.half_life > self.params.max_half_life:
                return False, f"Half-life too long: {metrics.half_life:.1f}"
        
        # Check Hurst exponent
        if metrics.hurst_exponent is not None:
            if metrics.hurst_exponent > 0.5:
                return False, f"Spread is trending (Hurst={metrics.hurst_exponent:.3f})"
        
        # Check for extreme values
        if abs(metrics.zscore) > self.params.stop_loss_zscore:
            return False, f"Z-score too extreme: {metrics.zscore:.2f}"
        
        return True, "Spread is tradeable"
