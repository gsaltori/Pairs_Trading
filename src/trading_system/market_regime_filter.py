"""
Market Regime Filter (MRF)

Pre-signal filter that blocks ALL trades when market conditions
are unfavorable for trend-following.

CONDITIONS (ALL must pass):
1. ADX(14) > 22 — Sufficient trend strength
2. ATR(14) / ATR(100) > 1.1 — Volatility expanding (not contracting)
3. |EMA200 slope| > epsilon — Clear directional bias

If ANY condition fails → BLOCK all trades.

This filter runs BEFORE signal generation to avoid
generating signals in hostile environments.
"""

import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List
from enum import Enum


class MRFBlockReason(Enum):
    """Reason for MRF block."""
    WEAK_TREND = "WEAK_TREND"           # ADX <= 22
    VOLATILITY_CONTRACTING = "VOL_CONTRACTING"  # ATR ratio <= 1.1
    NO_DIRECTIONAL_BIAS = "NO_DIRECTION"  # EMA slope too flat


@dataclass(frozen=True)
class MRFDecision:
    """
    Immutable MRF decision.
    
    Contains decision and full observables for audit.
    """
    allowed: bool
    reasons: tuple  # Tuple of MRFBlockReason
    timestamp: datetime
    
    # Observables at decision time
    adx: float
    atr_ratio: float
    ema_slope: float
    
    # Thresholds used
    adx_threshold: float
    atr_ratio_threshold: float
    slope_epsilon: float
    
    @property
    def is_blocked(self) -> bool:
        return not self.allowed
    
    def to_dict(self) -> dict:
        return {
            "allowed": self.allowed,
            "reasons": [r.value for r in self.reasons],
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "adx": self.adx,
            "atr_ratio": self.atr_ratio,
            "ema_slope": self.ema_slope,
        }


class MarketRegimeFilter:
    """
    Market Regime Filter for trend-following strategies.
    
    Blocks trades in unfavorable market conditions:
    - Low trend strength (ADX)
    - Contracting volatility
    - No clear directional bias
    
    PARAMETERS (LOCKED - DO NOT MODIFY):
    - ADX period: 14
    - ADX threshold: 22
    - ATR short period: 14
    - ATR long period: 100
    - ATR ratio threshold: 1.1
    - EMA period: 200
    - Slope lookback: 10 bars
    - Slope epsilon: 0.00001 (1 pip per 100 bars)
    """
    
    # LOCKED PARAMETERS
    ADX_PERIOD = 14
    ADX_THRESHOLD = 22
    
    ATR_SHORT_PERIOD = 14
    ATR_LONG_PERIOD = 100
    ATR_RATIO_THRESHOLD = 1.1
    
    EMA_PERIOD = 200
    SLOPE_LOOKBACK = 10
    SLOPE_EPSILON = 0.00001  # Minimum slope magnitude
    
    def __init__(self):
        # Price history
        self._timestamps: List[datetime] = []
        self._highs: List[float] = []
        self._lows: List[float] = []
        self._closes: List[float] = []
        
        # Indicator values
        self._adx: float = 0.0
        self._atr_short: float = 0.0
        self._atr_long: float = 0.0
        self._ema200: float = 0.0
        self._ema200_prev: float = 0.0
        
        # EMA history for slope calculation
        self._ema_history: List[float] = []
        
        # State
        self._bar_count = 0
    
    @property
    def min_bars_required(self) -> int:
        """Minimum bars needed for valid filter."""
        return max(self.ADX_PERIOD + 20, self.ATR_LONG_PERIOD, self.EMA_PERIOD) + self.SLOPE_LOOKBACK
    
    @property
    def is_ready(self) -> bool:
        return self._bar_count >= self.min_bars_required
    
    @property
    def current_adx(self) -> float:
        return self._adx
    
    @property
    def current_atr_ratio(self) -> float:
        if self._atr_long > 0:
            return self._atr_short / self._atr_long
        return 0.0
    
    @property
    def current_ema_slope(self) -> float:
        if len(self._ema_history) >= self.SLOPE_LOOKBACK:
            return self._ema_history[-1] - self._ema_history[-self.SLOPE_LOOKBACK]
        return 0.0
    
    def update(
        self,
        timestamp: datetime,
        high: float,
        low: float,
        close: float,
    ) -> None:
        """
        Update filter with new bar data.
        
        Call this BEFORE checking evaluate().
        """
        self._timestamps.append(timestamp)
        self._highs.append(high)
        self._lows.append(low)
        self._closes.append(close)
        self._bar_count += 1
        
        # Trim history
        max_history = self.ATR_LONG_PERIOD + 50
        if len(self._closes) > max_history:
            self._timestamps = self._timestamps[-max_history:]
            self._highs = self._highs[-max_history:]
            self._lows = self._lows[-max_history:]
            self._closes = self._closes[-max_history:]
        
        if len(self._ema_history) > self.SLOPE_LOOKBACK + 10:
            self._ema_history = self._ema_history[-(self.SLOPE_LOOKBACK + 10):]
        
        # Calculate indicators
        if self._bar_count > self.ADX_PERIOD:
            self._calculate_indicators()
    
    def _calculate_indicators(self) -> None:
        """Calculate all MRF indicators."""
        highs = np.array(self._highs)
        lows = np.array(self._lows)
        closes = np.array(self._closes)
        
        # ADX
        self._adx = self._calc_adx(highs, lows, closes, self.ADX_PERIOD)
        
        # ATR short and long
        self._atr_short = self._calc_atr(highs, lows, closes, self.ATR_SHORT_PERIOD)
        self._atr_long = self._calc_atr(highs, lows, closes, self.ATR_LONG_PERIOD)
        
        # EMA200
        self._ema200_prev = self._ema200
        self._ema200 = self._calc_ema(closes, self.EMA_PERIOD)
        
        if not np.isnan(self._ema200):
            self._ema_history.append(self._ema200)
    
    def _calc_ema(self, data: np.ndarray, period: int) -> float:
        """Calculate current EMA value."""
        if len(data) < period:
            return float('nan')
        
        multiplier = 2.0 / (period + 1)
        ema = np.mean(data[:period])
        
        for i in range(period, len(data)):
            ema = (data[i] - ema) * multiplier + ema
        
        return ema
    
    def _calc_atr(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int,
    ) -> float:
        """Calculate current ATR value."""
        if len(closes) < period + 1:
            return float('nan')
        
        tr = np.zeros(len(closes))
        tr[0] = highs[0] - lows[0]
        
        for i in range(1, len(closes)):
            tr[i] = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
        
        # Wilder's smoothing
        atr = np.mean(tr[:period])
        for i in range(period, len(tr)):
            atr = (atr * (period - 1) + tr[i]) / period
        
        return atr
    
    def _calc_adx(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int,
    ) -> float:
        """Calculate current ADX value."""
        if len(closes) < period * 2:
            return 0.0
        
        n = len(closes)
        
        # True Range
        tr = np.zeros(n)
        tr[0] = highs[0] - lows[0]
        for i in range(1, n):
            tr[i] = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
        
        # Directional Movement
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)
        
        for i in range(1, n):
            up_move = highs[i] - highs[i-1]
            down_move = lows[i-1] - lows[i]
            
            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move
        
        # Smooth TR, +DM, -DM using Wilder's method
        atr = np.mean(tr[:period])
        plus_dm_smooth = np.mean(plus_dm[:period])
        minus_dm_smooth = np.mean(minus_dm[:period])
        
        for i in range(period, n):
            atr = (atr * (period - 1) + tr[i]) / period
            plus_dm_smooth = (plus_dm_smooth * (period - 1) + plus_dm[i]) / period
            minus_dm_smooth = (minus_dm_smooth * (period - 1) + minus_dm[i]) / period
        
        # Directional Indicators
        if atr > 0:
            plus_di = 100 * plus_dm_smooth / atr
            minus_di = 100 * minus_dm_smooth / atr
        else:
            return 0.0
        
        # DX
        di_sum = plus_di + minus_di
        if di_sum > 0:
            dx = 100 * abs(plus_di - minus_di) / di_sum
        else:
            return 0.0
        
        # For simplicity, return current DX as ADX approximation
        # In production, would use full Wilder smoothing of DX
        return dx
    
    def evaluate(self) -> MRFDecision:
        """
        Evaluate current market regime.
        
        Returns MRFDecision with allowed status and reasons.
        """
        reasons: List[MRFBlockReason] = []
        
        if not self.is_ready:
            # Allow by default if not enough data
            return MRFDecision(
                allowed=True,
                reasons=tuple(),
                timestamp=datetime.utcnow(),
                adx=self._adx,
                atr_ratio=self.current_atr_ratio,
                ema_slope=self.current_ema_slope,
                adx_threshold=self.ADX_THRESHOLD,
                atr_ratio_threshold=self.ATR_RATIO_THRESHOLD,
                slope_epsilon=self.SLOPE_EPSILON,
            )
        
        # Condition 1: ADX > 22
        if self._adx <= self.ADX_THRESHOLD:
            reasons.append(MRFBlockReason.WEAK_TREND)
        
        # Condition 2: ATR(14) / ATR(100) > 1.1
        atr_ratio = self.current_atr_ratio
        if atr_ratio <= self.ATR_RATIO_THRESHOLD:
            reasons.append(MRFBlockReason.VOLATILITY_CONTRACTING)
        
        # Condition 3: |EMA200 slope| > epsilon
        ema_slope = self.current_ema_slope
        if abs(ema_slope) <= self.SLOPE_EPSILON:
            reasons.append(MRFBlockReason.NO_DIRECTIONAL_BIAS)
        
        allowed = len(reasons) == 0
        
        return MRFDecision(
            allowed=allowed,
            reasons=tuple(reasons),
            timestamp=self._timestamps[-1] if self._timestamps else datetime.utcnow(),
            adx=self._adx,
            atr_ratio=atr_ratio,
            ema_slope=ema_slope,
            adx_threshold=self.ADX_THRESHOLD,
            atr_ratio_threshold=self.ATR_RATIO_THRESHOLD,
            slope_epsilon=self.SLOPE_EPSILON,
        )
    
    def reset(self) -> None:
        """Reset filter state."""
        self._timestamps.clear()
        self._highs.clear()
        self._lows.clear()
        self._closes.clear()
        self._ema_history.clear()
        self._bar_count = 0
        self._adx = 0.0
        self._atr_short = 0.0
        self._atr_long = 0.0
        self._ema200 = 0.0
        self._ema200_prev = 0.0
