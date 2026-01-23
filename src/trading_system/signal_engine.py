"""
Signal Engine - Trend Following Signal Generation

Generates trading signals based on the validated trend-following strategy.
NO STATE LEAKAGE - Each signal is independent.
ONE SIGNAL PER BAR MAXIMUM.

Strategy (LOCKED):
- Long only if Close > EMA200
- Short only if Close < EMA200
- Entry: Pullback to EMA50
- SL: ATR(14) × 1.5
- TP: RR = 2.0
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime
from enum import Enum

from .config import StrategyConfig


class SignalDirection(Enum):
    """Trade direction."""
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass(frozen=True)
class TradeSignal:
    """
    Immutable trade signal.
    
    Contains all information needed to execute a trade.
    """
    bar_index: int
    timestamp: datetime
    direction: SignalDirection
    entry_price: float
    stop_loss: float
    take_profit: float
    atr_value: float
    
    @property
    def sl_distance(self) -> float:
        """Distance to stop loss."""
        return abs(self.entry_price - self.stop_loss)
    
    @property
    def tp_distance(self) -> float:
        """Distance to take profit."""
        return abs(self.take_profit - self.entry_price)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "bar_index": self.bar_index,
            "timestamp": self.timestamp.isoformat(),
            "direction": self.direction.value,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "atr_value": self.atr_value,
        }


class SignalEngine:
    """
    Trend-following signal generator.
    
    Generates at most ONE signal per bar.
    No internal state that affects signal generation.
    """
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        
        # Price history (rolling window)
        self._timestamps: List[datetime] = []
        self._opens: List[float] = []
        self._highs: List[float] = []
        self._lows: List[float] = []
        self._closes: List[float] = []
        
        # Indicator caches (recalculated each bar)
        self._ema_fast: Optional[float] = None
        self._ema_slow: Optional[float] = None
        self._atr: Optional[float] = None
        
        # State for pullback detection (minimal)
        self._prev_close_above_ema50: Optional[bool] = None
        
        # Bar counter
        self._bar_count = 0
    
    @property
    def min_bars_required(self) -> int:
        """Minimum bars needed before signals can be generated."""
        return max(self.config.ema_slow, self.config.atr_period) + 10
    
    @property
    def is_ready(self) -> bool:
        """Check if engine has enough data."""
        return self._bar_count >= self.min_bars_required
    
    @property
    def current_atr(self) -> Optional[float]:
        """Get current ATR value."""
        return self._atr
    
    def update(
        self,
        timestamp: datetime,
        open_: float,
        high: float,
        low: float,
        close: float,
    ) -> Optional[TradeSignal]:
        """
        Process a new bar and potentially generate a signal.
        
        Args:
            timestamp: Bar timestamp
            open_: Open price
            high: High price
            low: Low price
            close: Close price
        
        Returns:
            TradeSignal if conditions met, None otherwise
        """
        # Store price data
        self._timestamps.append(timestamp)
        self._opens.append(open_)
        self._highs.append(high)
        self._lows.append(low)
        self._closes.append(close)
        self._bar_count += 1
        
        # Trim history to save memory (keep 2x ema_slow)
        max_history = self.config.ema_slow * 2
        if len(self._closes) > max_history:
            self._timestamps = self._timestamps[-max_history:]
            self._opens = self._opens[-max_history:]
            self._highs = self._highs[-max_history:]
            self._lows = self._lows[-max_history:]
            self._closes = self._closes[-max_history:]
        
        # Not enough data yet
        if not self.is_ready:
            return None
        
        # Calculate indicators
        self._calculate_indicators()
        
        # Generate signal (if any)
        signal = self._check_signal(timestamp, close, high, low)
        
        # Update state for next bar
        self._prev_close_above_ema50 = close > self._ema_fast
        
        return signal
    
    def _calculate_indicators(self) -> None:
        """Calculate all required indicators."""
        closes = np.array(self._closes)
        highs = np.array(self._highs)
        lows = np.array(self._lows)
        
        # EMA calculations
        self._ema_fast = self._calc_ema(closes, self.config.ema_fast)
        self._ema_slow = self._calc_ema(closes, self.config.ema_slow)
        
        # ATR calculation
        self._atr = self._calc_atr(highs, lows, closes, self.config.atr_period)
    
    def _calc_ema(self, data: np.ndarray, period: int) -> float:
        """Calculate current EMA value."""
        if len(data) < period:
            return float('nan')
        
        multiplier = 2.0 / (period + 1)
        ema = np.mean(data[:period])  # Start with SMA
        
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
        
        # True Range
        tr = np.zeros(len(closes))
        tr[0] = highs[0] - lows[0]
        
        for i in range(1, len(closes)):
            tr[i] = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
        
        # ATR (Wilder's smoothing)
        atr = np.mean(tr[:period])
        for i in range(period, len(tr)):
            atr = (atr * (period - 1) + tr[i]) / period
        
        return atr
    
    def _check_signal(
        self,
        timestamp: datetime,
        close: float,
        high: float,
        low: float,
    ) -> Optional[TradeSignal]:
        """Check if current bar generates a signal."""
        # Validate indicators
        if np.isnan(self._ema_fast) or np.isnan(self._ema_slow) or np.isnan(self._atr):
            return None
        
        if self._atr <= 0:
            return None
        
        # Need previous bar state for pullback detection
        if self._prev_close_above_ema50 is None:
            return None
        
        # Determine trend direction
        if close > self._ema_slow:
            trend = SignalDirection.LONG
        elif close < self._ema_slow:
            trend = SignalDirection.SHORT
        else:
            return None  # No clear trend
        
        # Check for pullback entry
        current_above_ema50 = close > self._ema_fast
        signal = None
        
        # LONG: Price was below EMA50, now above, and we're in uptrend
        if (trend == SignalDirection.LONG and 
            not self._prev_close_above_ema50 and 
            current_above_ema50):
            # Verify bar touched EMA50
            if low <= self._ema_fast:
                signal = self._create_signal(
                    timestamp=timestamp,
                    direction=SignalDirection.LONG,
                    entry_price=close,
                )
        
        # SHORT: Price was above EMA50, now below, and we're in downtrend
        elif (trend == SignalDirection.SHORT and 
              self._prev_close_above_ema50 and 
              not current_above_ema50):
            # Verify bar touched EMA50
            if high >= self._ema_fast:
                signal = self._create_signal(
                    timestamp=timestamp,
                    direction=SignalDirection.SHORT,
                    entry_price=close,
                )
        
        return signal
    
    def _create_signal(
        self,
        timestamp: datetime,
        direction: SignalDirection,
        entry_price: float,
    ) -> TradeSignal:
        """Create a complete trade signal."""
        sl_distance = self._atr * self.config.atr_multiplier
        tp_distance = sl_distance * self.config.risk_reward
        
        if direction == SignalDirection.LONG:
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + tp_distance
        else:
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - tp_distance
        
        return TradeSignal(
            bar_index=self._bar_count - 1,
            timestamp=timestamp,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            atr_value=self._atr,
        )
    
    def reset(self) -> None:
        """Reset engine state (use with caution in production)."""
        self._timestamps.clear()
        self._opens.clear()
        self._highs.clear()
        self._lows.clear()
        self._closes.clear()
        self._bar_count = 0
        self._prev_close_above_ema50 = None
        self._ema_fast = None
        self._ema_slow = None
        self._atr = None
