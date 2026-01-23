"""
Volatility Expansion Strategy Engine

Captures impulsive breakout moves after volatility compression.

STRATEGY RULES (LOCKED):
- Timeframe: H4
- Compression: ATR(14) < 20th percentile of ATR(100)
- Breakout Long: Close above 20-bar high
- Breakout Short: Close below 20-bar low
- Entry: Confirmed bar close only
- Stop: ATR(14) × 1.5
- Take Profit: RR = 1.8
- No counter-trend logic
- No mean reversion
"""

import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List
from enum import Enum


class VolExpDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass(frozen=True)
class VolExpSignal:
    """Immutable volatility expansion trade signal."""
    bar_index: int
    timestamp: datetime
    direction: VolExpDirection
    entry_price: float
    stop_loss: float
    take_profit: float
    atr_value: float
    atr_percentile: float
    breakout_level: float
    
    strategy_name: str = "VOL_EXPANSION"
    
    @property
    def sl_distance(self) -> float:
        return abs(self.entry_price - self.stop_loss)
    
    @property
    def tp_distance(self) -> float:
        return abs(self.take_profit - self.entry_price)
    
    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy_name,
            "bar_index": self.bar_index,
            "timestamp": self.timestamp.isoformat(),
            "direction": self.direction.value,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "atr_value": self.atr_value,
            "atr_percentile": self.atr_percentile,
            "breakout_level": self.breakout_level,
        }


class VolatilityExpansionEngine:
    """
    Volatility Expansion Breakout Strategy Engine.
    
    Generates signals when price breaks out of consolidation
    during low volatility periods.
    
    Parameters (LOCKED - DO NOT MODIFY):
    - ATR period: 14
    - ATR lookback for percentile: 100
    - Compression threshold: 20th percentile
    - Breakout lookback: 20 bars
    - ATR multiplier: 1.5
    - Risk/Reward: 1.8
    """
    
    # LOCKED PARAMETERS
    ATR_PERIOD = 14
    ATR_LOOKBACK = 100
    COMPRESSION_PERCENTILE = 20
    BREAKOUT_LOOKBACK = 20
    ATR_MULTIPLIER = 1.5
    RISK_REWARD = 1.8
    
    def __init__(self):
        # Price history
        self._timestamps: List[datetime] = []
        self._opens: List[float] = []
        self._highs: List[float] = []
        self._lows: List[float] = []
        self._closes: List[float] = []
        
        # ATR history for percentile calculation
        self._atr_history: List[float] = []
        
        # Current values
        self._current_atr: Optional[float] = None
        self._atr_threshold: Optional[float] = None
        self._is_compressed: bool = False
        
        # State
        self._bar_count = 0
    
    @property
    def min_bars_required(self) -> int:
        """Minimum bars needed before signals can be generated."""
        return max(self.ATR_LOOKBACK, self.BREAKOUT_LOOKBACK, self.ATR_PERIOD) + 10
    
    @property
    def is_ready(self) -> bool:
        return self._bar_count >= self.min_bars_required
    
    @property
    def current_atr(self) -> Optional[float]:
        return self._current_atr
    
    @property
    def volatility_compressed(self) -> bool:
        return self._is_compressed
    
    def update(
        self,
        timestamp: datetime,
        open_: float,
        high: float,
        low: float,
        close: float,
    ) -> Optional[VolExpSignal]:
        """
        Process new bar and potentially generate signal.
        
        Returns VolExpSignal if conditions met, None otherwise.
        """
        # Store price data
        self._timestamps.append(timestamp)
        self._opens.append(open_)
        self._highs.append(high)
        self._lows.append(low)
        self._closes.append(close)
        self._bar_count += 1
        
        # Trim history
        max_history = self.ATR_LOOKBACK + 50
        if len(self._closes) > max_history:
            self._timestamps = self._timestamps[-max_history:]
            self._opens = self._opens[-max_history:]
            self._highs = self._highs[-max_history:]
            self._lows = self._lows[-max_history:]
            self._closes = self._closes[-max_history:]
        
        if len(self._atr_history) > self.ATR_LOOKBACK:
            self._atr_history = self._atr_history[-self.ATR_LOOKBACK:]
        
        if not self.is_ready:
            # Still accumulate ATR history
            if len(self._closes) > self.ATR_PERIOD:
                atr = self._calc_atr(
                    np.array(self._highs),
                    np.array(self._lows),
                    np.array(self._closes),
                    self.ATR_PERIOD
                )
                if not np.isnan(atr):
                    self._atr_history.append(atr)
            return None
        
        # Calculate indicators
        self._calculate_indicators()
        
        # Check for signal
        return self._check_signal(timestamp, close, high, low)
    
    def _calculate_indicators(self) -> None:
        """Calculate all required indicators."""
        highs = np.array(self._highs)
        lows = np.array(self._lows)
        closes = np.array(self._closes)
        
        # Current ATR
        self._current_atr = self._calc_atr(highs, lows, closes, self.ATR_PERIOD)
        
        # Update ATR history
        if not np.isnan(self._current_atr):
            self._atr_history.append(self._current_atr)
        
        # Calculate ATR percentile threshold
        if len(self._atr_history) >= self.ATR_LOOKBACK:
            recent_atrs = self._atr_history[-self.ATR_LOOKBACK:]
            self._atr_threshold = np.percentile(recent_atrs, self.COMPRESSION_PERCENTILE)
            self._is_compressed = self._current_atr < self._atr_threshold
        else:
            self._is_compressed = False
    
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
    
    def _get_channel_high(self) -> float:
        """Get 20-bar high for breakout detection."""
        lookback = min(self.BREAKOUT_LOOKBACK, len(self._highs) - 1)
        # Exclude current bar
        return max(self._highs[-lookback-1:-1]) if lookback > 0 else self._highs[-2]
    
    def _get_channel_low(self) -> float:
        """Get 20-bar low for breakout detection."""
        lookback = min(self.BREAKOUT_LOOKBACK, len(self._lows) - 1)
        # Exclude current bar
        return min(self._lows[-lookback-1:-1]) if lookback > 0 else self._lows[-2]
    
    def _check_signal(
        self,
        timestamp: datetime,
        close: float,
        high: float,
        low: float,
    ) -> Optional[VolExpSignal]:
        """Check for volatility expansion breakout signal."""
        if np.isnan(self._current_atr) or self._current_atr <= 0:
            return None
        
        # Must be in compression state
        if not self._is_compressed:
            return None
        
        channel_high = self._get_channel_high()
        channel_low = self._get_channel_low()
        
        signal = None
        
        # LONG: Close above channel high
        if close > channel_high:
            signal = self._create_long_signal(timestamp, close, channel_high)
        
        # SHORT: Close below channel low
        elif close < channel_low:
            signal = self._create_short_signal(timestamp, close, channel_low)
        
        return signal
    
    def _create_long_signal(
        self,
        timestamp: datetime,
        close: float,
        breakout_level: float,
    ) -> VolExpSignal:
        """Create long volatility expansion signal."""
        stop_loss = close - self._current_atr * self.ATR_MULTIPLIER
        sl_distance = close - stop_loss
        take_profit = close + sl_distance * self.RISK_REWARD
        
        # Calculate current ATR percentile for logging
        atr_percentile = 0.0
        if len(self._atr_history) >= self.ATR_LOOKBACK:
            recent_atrs = sorted(self._atr_history[-self.ATR_LOOKBACK:])
            atr_percentile = (sum(1 for a in recent_atrs if a < self._current_atr) / 
                            len(recent_atrs)) * 100
        
        return VolExpSignal(
            bar_index=self._bar_count - 1,
            timestamp=timestamp,
            direction=VolExpDirection.LONG,
            entry_price=close,
            stop_loss=stop_loss,
            take_profit=take_profit,
            atr_value=self._current_atr,
            atr_percentile=atr_percentile,
            breakout_level=breakout_level,
        )
    
    def _create_short_signal(
        self,
        timestamp: datetime,
        close: float,
        breakout_level: float,
    ) -> VolExpSignal:
        """Create short volatility expansion signal."""
        stop_loss = close + self._current_atr * self.ATR_MULTIPLIER
        sl_distance = stop_loss - close
        take_profit = close - sl_distance * self.RISK_REWARD
        
        # Calculate current ATR percentile for logging
        atr_percentile = 0.0
        if len(self._atr_history) >= self.ATR_LOOKBACK:
            recent_atrs = sorted(self._atr_history[-self.ATR_LOOKBACK:])
            atr_percentile = (sum(1 for a in recent_atrs if a < self._current_atr) / 
                            len(recent_atrs)) * 100
        
        return VolExpSignal(
            bar_index=self._bar_count - 1,
            timestamp=timestamp,
            direction=VolExpDirection.SHORT,
            entry_price=close,
            stop_loss=stop_loss,
            take_profit=take_profit,
            atr_value=self._current_atr,
            atr_percentile=atr_percentile,
            breakout_level=breakout_level,
        )
    
    def reset(self) -> None:
        """Reset engine state."""
        self._timestamps.clear()
        self._opens.clear()
        self._highs.clear()
        self._lows.clear()
        self._closes.clear()
        self._atr_history.clear()
        self._bar_count = 0
        self._current_atr = None
        self._atr_threshold = None
        self._is_compressed = False
