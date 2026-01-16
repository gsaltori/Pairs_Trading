"""
Spread calculation and Z-score computation.

This module computes the spread between two FX pairs and normalizes it.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple
import numpy as np

try:
    from .config import CONFIG
    from .observations import MarketObservation
except ImportError:
    from config import CONFIG
    from observations import MarketObservation


@dataclass(frozen=True)
class SpreadObservation:
    """
    Spread observation for a structural pair.
    
    IMMUTABLE after creation.
    Contains all spread metrics at a single point in time.
    """
    # Identity
    observation_id: str
    timestamp: datetime
    timeframe: str
    
    # Pair
    pair: Tuple[str, str]
    
    # Raw prices (close)
    price_a: float
    price_b: float
    
    # Computed values
    hedge_ratio: float
    spread_value: float
    spread_mean: float
    spread_std: float
    zscore: float
    
    # Correlation (for invalidation checks)
    correlation: float
    
    # Validity
    is_valid: bool
    invalidity_reason: Optional[str] = None


class SpreadCalculator:
    """
    Calculates spread and Z-score for a pair of FX symbols.
    
    FIXED PARAMETERS:
    - Window size: 60 bars (from CONFIG)
    - Hedge ratio: Rolling OLS
    - Z-score: (spread - mean) / std
    
    NO TUNING PERMITTED.
    """
    
    def __init__(
        self,
        symbol_a: str = CONFIG.SYMBOL_A,
        symbol_b: str = CONFIG.SYMBOL_B,
        window: int = CONFIG.ZSCORE_WINDOW,
    ):
        self.symbol_a = symbol_a
        self.symbol_b = symbol_b
        self.pair = (symbol_a, symbol_b)
        self.window = window
        
        # Price history buffers
        self._prices_a: List[float] = []
        self._prices_b: List[float] = []
        self._timestamps: List[datetime] = []
    
    def update(self, observation: MarketObservation) -> Optional[SpreadObservation]:
        """
        Update calculator with new observation and compute spread.
        
        Returns SpreadObservation if sufficient data, None otherwise.
        """
        # Extract close prices
        price_a = observation.bar_a.close
        price_b = observation.bar_b.close
        
        # Add to history
        self._prices_a.append(price_a)
        self._prices_b.append(price_b)
        self._timestamps.append(observation.timestamp)
        
        # Trim to window size + buffer
        max_history = self.window + 10
        if len(self._prices_a) > max_history:
            self._prices_a = self._prices_a[-max_history:]
            self._prices_b = self._prices_b[-max_history:]
            self._timestamps = self._timestamps[-max_history:]
        
        # Check if we have enough data
        if len(self._prices_a) < self.window:
            return SpreadObservation(
                observation_id=observation.observation_id,
                timestamp=observation.timestamp,
                timeframe=observation.timeframe,
                pair=self.pair,
                price_a=price_a,
                price_b=price_b,
                hedge_ratio=0.0,
                spread_value=0.0,
                spread_mean=0.0,
                spread_std=0.0,
                zscore=0.0,
                correlation=0.0,
                is_valid=False,
                invalidity_reason=f"Insufficient data: need {self.window}, have {len(self._prices_a)}",
            )
        
        # Compute spread metrics
        return self._compute_spread(
            observation_id=observation.observation_id,
            timestamp=observation.timestamp,
            timeframe=observation.timeframe,
            price_a=price_a,
            price_b=price_b,
        )
    
    def _compute_spread(
        self,
        observation_id: str,
        timestamp: datetime,
        timeframe: str,
        price_a: float,
        price_b: float,
    ) -> SpreadObservation:
        """
        Compute spread, hedge ratio, Z-score, and correlation.
        """
        # Get rolling window
        window_a = np.array(self._prices_a[-self.window:])
        window_b = np.array(self._prices_b[-self.window:])
        
        # Compute correlation
        correlation = np.corrcoef(window_a, window_b)[0, 1]
        
        # Check for valid correlation
        if np.isnan(correlation):
            return SpreadObservation(
                observation_id=observation_id,
                timestamp=timestamp,
                timeframe=timeframe,
                pair=self.pair,
                price_a=price_a,
                price_b=price_b,
                hedge_ratio=0.0,
                spread_value=0.0,
                spread_mean=0.0,
                spread_std=0.0,
                zscore=0.0,
                correlation=0.0,
                is_valid=False,
                invalidity_reason="Correlation is NaN",
            )
        
        # Compute hedge ratio using OLS: price_a = hedge_ratio * price_b + residual
        # hedge_ratio = Cov(a, b) / Var(b)
        cov = np.cov(window_a, window_b)[0, 1]
        var_b = np.var(window_b)
        
        if var_b < 1e-10:
            return SpreadObservation(
                observation_id=observation_id,
                timestamp=timestamp,
                timeframe=timeframe,
                pair=self.pair,
                price_a=price_a,
                price_b=price_b,
                hedge_ratio=0.0,
                spread_value=0.0,
                spread_mean=0.0,
                spread_std=0.0,
                zscore=0.0,
                correlation=correlation,
                is_valid=False,
                invalidity_reason="Zero variance in price_b",
            )
        
        hedge_ratio = cov / var_b
        
        # Clamp hedge ratio to reasonable range
        hedge_ratio = float(np.clip(hedge_ratio, 0.1, 10.0))
        
        # Compute spread series
        spread_series = window_a - hedge_ratio * window_b
        
        # Compute statistics
        spread_mean = float(np.mean(spread_series))
        spread_std = float(np.std(spread_series))
        
        if spread_std < 1e-10:
            return SpreadObservation(
                observation_id=observation_id,
                timestamp=timestamp,
                timeframe=timeframe,
                pair=self.pair,
                price_a=price_a,
                price_b=price_b,
                hedge_ratio=hedge_ratio,
                spread_value=float(spread_series[-1]),
                spread_mean=spread_mean,
                spread_std=spread_std,
                zscore=0.0,
                correlation=correlation,
                is_valid=False,
                invalidity_reason="Zero standard deviation in spread",
            )
        
        # Current spread value
        spread_value = price_a - hedge_ratio * price_b
        
        # Z-score
        zscore = (spread_value - spread_mean) / spread_std
        zscore = float(np.clip(zscore, -10, 10))  # Clamp extreme values
        
        return SpreadObservation(
            observation_id=observation_id,
            timestamp=timestamp,
            timeframe=timeframe,
            pair=self.pair,
            price_a=price_a,
            price_b=price_b,
            hedge_ratio=hedge_ratio,
            spread_value=spread_value,
            spread_mean=spread_mean,
            spread_std=spread_std,
            zscore=zscore,
            correlation=correlation,
            is_valid=True,
        )
    
    def get_current_correlation(self) -> float:
        """Get the most recent correlation value."""
        if len(self._prices_a) < self.window:
            return 0.0
        
        window_a = np.array(self._prices_a[-self.window:])
        window_b = np.array(self._prices_b[-self.window:])
        
        corr = np.corrcoef(window_a, window_b)[0, 1]
        return float(corr) if not np.isnan(corr) else 0.0
    
    def reset(self) -> None:
        """Reset calculator state."""
        self._prices_a = []
        self._prices_b = []
        self._timestamps = []
