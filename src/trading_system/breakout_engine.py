"""
Range Breakout Strategy Engine

A volatility contraction → expansion breakout strategy
with asymmetric payoff (R ≥ 2.5).

EDGE HYPOTHESIS:
Markets alternate between contraction and expansion phases.
After sufficient contraction, the subsequent expansion move
often exceeds the contraction range by a multiple.

SIGNAL LOGIC:
1. Identify consolidation: N-bar range where width < ATR threshold
2. Wait for breakout: Close outside the range boundaries
3. Entry: On confirmed close (no intrabar triggers)
4. Direction: Long if close > range_high, Short if close < range_low

TRADE LOGIC:
- SL: Opposite side of range + buffer
- TP: 2.5 × risk distance
- Max 1 trade per breakout (no re-entry)
- Time stop: 20 bars max hold (optional)

PARAMETERS (LOCKED - NO OPTIMIZATION):
- Range lookback: 6 bars (24h on H4)
- ATR period: 14
- Compression threshold: Range width < 0.8 × ATR(14)
- Risk/Reward: 2.5
- SL buffer: 0.1 × ATR
"""

import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Tuple
from enum import Enum


class BreakoutDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass(frozen=True)
class BreakoutSignal:
    """Immutable breakout trade signal."""
    bar_index: int
    timestamp: datetime
    direction: BreakoutDirection
    entry_price: float
    stop_loss: float
    take_profit: float
    
    # Range info for audit
    range_high: float
    range_low: float
    range_width: float
    atr_value: float
    compression_ratio: float  # range_width / ATR
    
    strategy_name: str = "RANGE_BREAKOUT"
    
    @property
    def sl_distance(self) -> float:
        return abs(self.entry_price - self.stop_loss)
    
    @property
    def tp_distance(self) -> float:
        return abs(self.take_profit - self.entry_price)
    
    @property
    def risk_reward(self) -> float:
        if self.sl_distance > 0:
            return self.tp_distance / self.sl_distance
        return 0
    
    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy_name,
            "bar_index": self.bar_index,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "direction": self.direction.value,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "range_high": self.range_high,
            "range_low": self.range_low,
            "range_width": self.range_width,
            "atr_value": self.atr_value,
            "compression_ratio": self.compression_ratio,
            "risk_reward": self.risk_reward,
        }


class RangeBreakoutEngine:
    """
    Range Breakout Strategy Engine.
    
    Generates signals when price breaks out of a compressed range.
    
    PARAMETERS (LOCKED - DO NOT MODIFY):
    - Range lookback: 6 bars
    - ATR period: 14
    - Compression threshold: 0.8 (range < 0.8 × ATR)
    - Risk/Reward: 2.5
    - SL buffer: 0.1 × ATR
    - Cooldown: 3 bars after breakout (prevent re-entry)
    """
    
    # LOCKED PARAMETERS
    RANGE_LOOKBACK = 6          # 6 bars = 24h on H4
    ATR_PERIOD = 14
    COMPRESSION_THRESHOLD = 0.8  # Range must be < 0.8 × ATR
    RISK_REWARD = 2.5
    SL_BUFFER_MULT = 0.1        # Buffer as fraction of ATR
    COOLDOWN_BARS = 3           # Bars to wait after signal
    
    def __init__(self):
        # Price history
        self._timestamps: List[datetime] = []
        self._opens: List[float] = []
        self._highs: List[float] = []
        self._lows: List[float] = []
        self._closes: List[float] = []
        
        # Indicator
        self._atr: float = 0.0
        
        # State
        self._bar_count = 0
        self._last_signal_bar = -100  # Cooldown tracking
        self._in_range = False
        self._range_high = 0.0
        self._range_low = 0.0
    
    @property
    def min_bars_required(self) -> int:
        """Minimum bars needed before signals possible."""
        return max(self.RANGE_LOOKBACK, self.ATR_PERIOD) + 5
    
    @property
    def is_ready(self) -> bool:
        return self._bar_count >= self.min_bars_required
    
    @property
    def current_atr(self) -> float:
        return self._atr
    
    def update(
        self,
        timestamp: datetime,
        open_: float,
        high: float,
        low: float,
        close: float,
    ) -> Optional[BreakoutSignal]:
        """
        Process new bar and potentially generate signal.
        
        Returns BreakoutSignal if breakout conditions met, None otherwise.
        """
        # Store price data
        self._timestamps.append(timestamp)
        self._opens.append(open_)
        self._highs.append(high)
        self._lows.append(low)
        self._closes.append(close)
        self._bar_count += 1
        
        # Trim history
        max_history = max(self.RANGE_LOOKBACK, self.ATR_PERIOD) + 20
        if len(self._closes) > max_history:
            self._timestamps = self._timestamps[-max_history:]
            self._opens = self._opens[-max_history:]
            self._highs = self._highs[-max_history:]
            self._lows = self._lows[-max_history:]
            self._closes = self._closes[-max_history:]
        
        if not self.is_ready:
            return None
        
        # Calculate ATR
        self._atr = self._calc_atr()
        
        if self._atr <= 0:
            return None
        
        # Check cooldown
        if self._bar_count - self._last_signal_bar < self.COOLDOWN_BARS:
            return None
        
        # Check for breakout
        return self._check_breakout(timestamp, close)
    
    def _calc_atr(self) -> float:
        """Calculate current ATR value."""
        highs = np.array(self._highs)
        lows = np.array(self._lows)
        closes = np.array(self._closes)
        
        if len(closes) < self.ATR_PERIOD + 1:
            return 0.0
        
        tr = np.zeros(len(closes))
        tr[0] = highs[0] - lows[0]
        
        for i in range(1, len(closes)):
            tr[i] = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
        
        # Wilder's smoothing
        atr = np.mean(tr[:self.ATR_PERIOD])
        for i in range(self.ATR_PERIOD, len(tr)):
            atr = (atr * (self.ATR_PERIOD - 1) + tr[i]) / self.ATR_PERIOD
        
        return atr
    
    def _get_range(self) -> Tuple[float, float, float]:
        """
        Get the N-bar range (excluding current bar).
        
        Returns: (range_high, range_low, range_width)
        """
        # Use previous N bars (not including current)
        lookback = min(self.RANGE_LOOKBACK, len(self._highs) - 1)
        
        if lookback < 2:
            return 0, 0, 0
        
        range_highs = self._highs[-lookback-1:-1]
        range_lows = self._lows[-lookback-1:-1]
        
        range_high = max(range_highs)
        range_low = min(range_lows)
        range_width = range_high - range_low
        
        return range_high, range_low, range_width
    
    def _check_breakout(self, timestamp: datetime, close: float) -> Optional[BreakoutSignal]:
        """Check for valid breakout signal."""
        range_high, range_low, range_width = self._get_range()
        
        if range_width <= 0:
            return None
        
        # Check compression: range must be < threshold × ATR
        compression_ratio = range_width / self._atr
        
        if compression_ratio >= self.COMPRESSION_THRESHOLD:
            # Range is not compressed enough
            return None
        
        # Check for breakout
        signal = None
        sl_buffer = self._atr * self.SL_BUFFER_MULT
        
        # LONG: Close above range high
        if close > range_high:
            stop_loss = range_low - sl_buffer
            sl_distance = close - stop_loss
            take_profit = close + sl_distance * self.RISK_REWARD
            
            signal = BreakoutSignal(
                bar_index=self._bar_count - 1,
                timestamp=timestamp,
                direction=BreakoutDirection.LONG,
                entry_price=close,
                stop_loss=stop_loss,
                take_profit=take_profit,
                range_high=range_high,
                range_low=range_low,
                range_width=range_width,
                atr_value=self._atr,
                compression_ratio=compression_ratio,
            )
        
        # SHORT: Close below range low
        elif close < range_low:
            stop_loss = range_high + sl_buffer
            sl_distance = stop_loss - close
            take_profit = close - sl_distance * self.RISK_REWARD
            
            signal = BreakoutSignal(
                bar_index=self._bar_count - 1,
                timestamp=timestamp,
                direction=BreakoutDirection.SHORT,
                entry_price=close,
                stop_loss=stop_loss,
                take_profit=take_profit,
                range_high=range_high,
                range_low=range_low,
                range_width=range_width,
                atr_value=self._atr,
                compression_ratio=compression_ratio,
            )
        
        if signal is not None:
            self._last_signal_bar = self._bar_count
        
        return signal
    
    def reset(self) -> None:
        """Reset engine state."""
        self._timestamps.clear()
        self._opens.clear()
        self._highs.clear()
        self._lows.clear()
        self._closes.clear()
        self._bar_count = 0
        self._last_signal_bar = -100
        self._atr = 0.0
        self._in_range = False
        self._range_high = 0.0
        self._range_low = 0.0
