"""
Pullback Strategy Engine

RSI-based trend pullback entries within established trends.

STRATEGY RULES (LOCKED):
- Timeframe: H4
- Trend filter: Close > EMA200 (long), Close < EMA200 (short)
- Entry: RSI(14) crosses back above 40 (uptrend) / below 60 (downtrend)
- Stop: max(ATR(14) × 1.2, recent swing)
- Take Profit: RR = 1.5
- One signal per bar
- One open position at a time
"""

import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List
from enum import Enum


class PullbackDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass(frozen=True)
class PullbackSignal:
    """Immutable pullback trade signal."""
    bar_index: int
    timestamp: datetime
    direction: PullbackDirection
    entry_price: float
    stop_loss: float
    take_profit: float
    atr_value: float
    rsi_value: float
    swing_level: float
    
    strategy_name: str = "PULLBACK"
    
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
            "rsi_value": self.rsi_value,
            "swing_level": self.swing_level,
        }


class PullbackEngine:
    """
    RSI Pullback Strategy Engine.
    
    Generates signals when RSI recovers from oversold/overbought
    within an established trend.
    
    Parameters (LOCKED - DO NOT MODIFY):
    - EMA period: 200 (trend filter)
    - RSI period: 14
    - RSI long threshold: 40
    - RSI short threshold: 60
    - ATR period: 14
    - ATR multiplier: 1.2
    - Risk/Reward: 1.5
    - Swing lookback: 10 bars
    """
    
    # LOCKED PARAMETERS
    EMA_PERIOD = 200
    RSI_PERIOD = 14
    RSI_LONG_THRESHOLD = 40
    RSI_SHORT_THRESHOLD = 60
    ATR_PERIOD = 14
    ATR_MULTIPLIER = 1.2
    RISK_REWARD = 1.5
    SWING_LOOKBACK = 10
    
    def __init__(self):
        # Price history
        self._timestamps: List[datetime] = []
        self._opens: List[float] = []
        self._highs: List[float] = []
        self._lows: List[float] = []
        self._closes: List[float] = []
        
        # Indicator values
        self._ema200: Optional[float] = None
        self._rsi: Optional[float] = None
        self._atr: Optional[float] = None
        self._prev_rsi: Optional[float] = None
        
        # State
        self._bar_count = 0
    
    @property
    def min_bars_required(self) -> int:
        """Minimum bars needed before signals can be generated."""
        return max(self.EMA_PERIOD, self.RSI_PERIOD, self.ATR_PERIOD) + 10
    
    @property
    def is_ready(self) -> bool:
        return self._bar_count >= self.min_bars_required
    
    @property
    def current_rsi(self) -> Optional[float]:
        return self._rsi
    
    def update(
        self,
        timestamp: datetime,
        open_: float,
        high: float,
        low: float,
        close: float,
    ) -> Optional[PullbackSignal]:
        """
        Process new bar and potentially generate signal.
        
        Returns PullbackSignal if conditions met, None otherwise.
        """
        # Store previous RSI for crossover detection
        self._prev_rsi = self._rsi
        
        # Store price data
        self._timestamps.append(timestamp)
        self._opens.append(open_)
        self._highs.append(high)
        self._lows.append(low)
        self._closes.append(close)
        self._bar_count += 1
        
        # Trim history
        max_history = self.EMA_PERIOD * 2
        if len(self._closes) > max_history:
            self._timestamps = self._timestamps[-max_history:]
            self._opens = self._opens[-max_history:]
            self._highs = self._highs[-max_history:]
            self._lows = self._lows[-max_history:]
            self._closes = self._closes[-max_history:]
        
        if not self.is_ready:
            return None
        
        # Calculate indicators
        self._calculate_indicators()
        
        # Check for signal
        return self._check_signal(timestamp, close)
    
    def _calculate_indicators(self) -> None:
        """Calculate all required indicators."""
        closes = np.array(self._closes)
        highs = np.array(self._highs)
        lows = np.array(self._lows)
        
        self._ema200 = self._calc_ema(closes, self.EMA_PERIOD)
        self._rsi = self._calc_rsi(closes, self.RSI_PERIOD)
        self._atr = self._calc_atr(highs, lows, closes, self.ATR_PERIOD)
    
    def _calc_ema(self, data: np.ndarray, period: int) -> float:
        """Calculate current EMA value."""
        if len(data) < period:
            return float('nan')
        
        multiplier = 2.0 / (period + 1)
        ema = np.mean(data[:period])
        
        for i in range(period, len(data)):
            ema = (data[i] - ema) * multiplier + ema
        
        return ema
    
    def _calc_rsi(self, data: np.ndarray, period: int) -> float:
        """Calculate current RSI value."""
        if len(data) < period + 1:
            return 50.0  # Neutral default
        
        deltas = np.diff(data)
        
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Use Wilder's smoothing
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        if avg_loss < 1e-10:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        
        return rsi
    
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
    
    def _get_recent_swing_low(self) -> float:
        """Get recent swing low for stop placement."""
        lookback = min(self.SWING_LOOKBACK, len(self._lows))
        return min(self._lows[-lookback:])
    
    def _get_recent_swing_high(self) -> float:
        """Get recent swing high for stop placement."""
        lookback = min(self.SWING_LOOKBACK, len(self._highs))
        return max(self._highs[-lookback:])
    
    def _check_signal(self, timestamp: datetime, close: float) -> Optional[PullbackSignal]:
        """Check for pullback entry signal."""
        if np.isnan(self._ema200) or np.isnan(self._rsi) or np.isnan(self._atr):
            return None
        
        if self._prev_rsi is None:
            return None
        
        if self._atr <= 0:
            return None
        
        signal = None
        
        # LONG: Uptrend + RSI crosses above 40
        if close > self._ema200:
            if self._prev_rsi < self.RSI_LONG_THRESHOLD and self._rsi >= self.RSI_LONG_THRESHOLD:
                signal = self._create_long_signal(timestamp, close)
        
        # SHORT: Downtrend + RSI crosses below 60
        elif close < self._ema200:
            if self._prev_rsi > self.RSI_SHORT_THRESHOLD and self._rsi <= self.RSI_SHORT_THRESHOLD:
                signal = self._create_short_signal(timestamp, close)
        
        return signal
    
    def _create_long_signal(self, timestamp: datetime, close: float) -> PullbackSignal:
        """Create long pullback signal."""
        # Stop: max(ATR × 1.2, recent swing low)
        atr_stop = close - self._atr * self.ATR_MULTIPLIER
        swing_stop = self._get_recent_swing_low() - self._atr * 0.1  # Small buffer
        stop_loss = min(atr_stop, swing_stop)  # Use wider stop
        
        sl_distance = close - stop_loss
        take_profit = close + sl_distance * self.RISK_REWARD
        
        return PullbackSignal(
            bar_index=self._bar_count - 1,
            timestamp=timestamp,
            direction=PullbackDirection.LONG,
            entry_price=close,
            stop_loss=stop_loss,
            take_profit=take_profit,
            atr_value=self._atr,
            rsi_value=self._rsi,
            swing_level=self._get_recent_swing_low(),
        )
    
    def _create_short_signal(self, timestamp: datetime, close: float) -> PullbackSignal:
        """Create short pullback signal."""
        # Stop: max(ATR × 1.2, recent swing high)
        atr_stop = close + self._atr * self.ATR_MULTIPLIER
        swing_stop = self._get_recent_swing_high() + self._atr * 0.1  # Small buffer
        stop_loss = max(atr_stop, swing_stop)  # Use wider stop
        
        sl_distance = stop_loss - close
        take_profit = close - sl_distance * self.RISK_REWARD
        
        return PullbackSignal(
            bar_index=self._bar_count - 1,
            timestamp=timestamp,
            direction=PullbackDirection.SHORT,
            entry_price=close,
            stop_loss=stop_loss,
            take_profit=take_profit,
            atr_value=self._atr,
            rsi_value=self._rsi,
            swing_level=self._get_recent_swing_high(),
        )
    
    def reset(self) -> None:
        """Reset engine state."""
        self._timestamps.clear()
        self._opens.clear()
        self._highs.clear()
        self._lows.clear()
        self._closes.clear()
        self._bar_count = 0
        self._prev_rsi = None
        self._ema200 = None
        self._rsi = None
        self._atr = None
