"""
Session-Based FX Strategy Engine

Asia Range → London Session Expansion Bias

EDGE HYPOTHESIS:
- Asian session (low liquidity) establishes a range
- London session (high liquidity) expands this range directionally
- Asia close position relative to midpoint predicts expansion direction

LIQUIDITY MECHANICS:
- Asia: Banks in Tokyo/Sydney set initial range, volume ~15% of daily
- London: Institutional order flow (~35% of daily volume) breaks range
- Direction bias: Where Asian session "settled" indicates pending order flow

SESSION DEFINITIONS (UTC):
- Asia:   00:00 - 06:00 UTC
- London: 07:00 - 11:00 UTC

ENTRY LOGIC:
- Calculate Asia range: High, Low, Midpoint
- Determine bias: Close > Mid = BULLISH, Close < Mid = BEARISH
- Entry: Break of Asia high (bullish) or low (bearish) during London
- No counter-trend trades (bias must match breakout direction)

TRADE PARAMETERS:
- SL: Opposite side of Asia range + buffer
- TP: 2.5 × risk distance
- Time stop: End of London session (11:00 UTC)
- Max 1 trade per day
"""

import numpy as np
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from typing import Optional, List, Tuple
from enum import Enum


class SessionBias(Enum):
    BULLISH = "BULLISH"   # Asia close > midpoint
    BEARISH = "BEARISH"   # Asia close < midpoint
    NEUTRAL = "NEUTRAL"   # Asia close ≈ midpoint (no trade)


class SessionDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass(frozen=True)
class AsiaRange:
    """Immutable Asia session range data."""
    date: datetime  # Date of the session
    high: float
    low: float
    open: float
    close: float
    
    @property
    def midpoint(self) -> float:
        return (self.high + self.low) / 2
    
    @property
    def range_pips(self) -> float:
        """Range size in pips (for EURUSD, 1 pip = 0.0001)."""
        return (self.high - self.low) / 0.0001
    
    @property
    def bias(self) -> SessionBias:
        """Determine directional bias from close position."""
        mid = self.midpoint
        range_size = self.high - self.low
        
        # Require meaningful deviation from midpoint (>20% of range)
        threshold = range_size * 0.2
        
        if self.close > mid + threshold:
            return SessionBias.BULLISH
        elif self.close < mid - threshold:
            return SessionBias.BEARISH
        else:
            return SessionBias.NEUTRAL
    
    @property
    def is_valid(self) -> bool:
        """Check if range is tradeable."""
        # Minimum range: 15 pips (avoid noise)
        # Maximum range: 80 pips (avoid excessive risk)
        return 15 <= self.range_pips <= 80


@dataclass(frozen=True)
class SessionSignal:
    """Immutable session-based trade signal."""
    date: datetime
    timestamp: datetime  # Entry timestamp
    direction: SessionDirection
    entry_price: float
    stop_loss: float
    take_profit: float
    
    # Session data for audit
    asia_high: float
    asia_low: float
    asia_close: float
    asia_bias: SessionBias
    asia_range_pips: float
    
    strategy_name: str = "ASIA_LONDON_EXPANSION"
    
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
            "date": self.date.isoformat() if self.date else None,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "direction": self.direction.value,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "asia_high": self.asia_high,
            "asia_low": self.asia_low,
            "asia_close": self.asia_close,
            "asia_bias": self.asia_bias.value,
            "asia_range_pips": self.asia_range_pips,
            "risk_reward": self.risk_reward,
        }


class SessionEngine:
    """
    Asia Range → London Expansion Strategy Engine.
    
    PARAMETERS (LOCKED - DO NOT MODIFY):
    - Asia session: 00:00 - 06:00 UTC
    - London session: 07:00 - 11:00 UTC
    - Min Asia range: 15 pips
    - Max Asia range: 80 pips
    - Bias threshold: 20% of range from midpoint
    - Risk/Reward: 2.5
    - SL buffer: 3 pips
    """
    
    # LOCKED SESSION TIMES (UTC)
    ASIA_START = time(0, 0)    # 00:00 UTC
    ASIA_END = time(6, 0)      # 06:00 UTC
    LONDON_START = time(7, 0)  # 07:00 UTC
    LONDON_END = time(11, 0)   # 11:00 UTC
    
    # LOCKED PARAMETERS
    MIN_RANGE_PIPS = 15
    MAX_RANGE_PIPS = 80
    BIAS_THRESHOLD = 0.2  # 20% of range
    RISK_REWARD = 2.5
    SL_BUFFER_PIPS = 3
    
    def __init__(self):
        # Session tracking
        self._current_asia: Optional[AsiaRange] = None
        self._asia_bars: List[dict] = []  # Bars within current Asia session
        
        # Trade tracking
        self._last_trade_date: Optional[datetime] = None
        self._pending_signal: Optional[SessionSignal] = None
        self._in_london: bool = False
        
        # Statistics
        self._sessions_analyzed = 0
        self._valid_setups = 0
        self._neutral_biases = 0
    
    @property
    def sessions_analyzed(self) -> int:
        return self._sessions_analyzed
    
    @property
    def valid_setups(self) -> int:
        return self._valid_setups
    
    def _is_asia_session(self, ts: datetime) -> bool:
        """Check if timestamp is within Asia session."""
        t = ts.time()
        return self.ASIA_START <= t < self.ASIA_END
    
    def _is_london_session(self, ts: datetime) -> bool:
        """Check if timestamp is within London session."""
        t = ts.time()
        return self.LONDON_START <= t < self.LONDON_END
    
    def _get_session_date(self, ts: datetime) -> datetime:
        """Get the trading date for a timestamp."""
        # For times before Asia end, it's the same calendar date
        # For times after, still same date
        return ts.replace(hour=0, minute=0, second=0, microsecond=0)
    
    def update(
        self,
        timestamp: datetime,
        open_: float,
        high: float,
        low: float,
        close: float,
    ) -> Optional[SessionSignal]:
        """
        Process new bar and potentially generate signal.
        
        Call for each bar in chronological order.
        Returns SessionSignal if London breakout conditions met.
        """
        bar = {
            'timestamp': timestamp,
            'open': open_,
            'high': high,
            'low': low,
            'close': close,
        }
        
        session_date = self._get_session_date(timestamp)
        
        # Check if we're in Asia session - accumulate bars
        if self._is_asia_session(timestamp):
            # If new day, reset Asia accumulation
            if self._asia_bars and self._get_session_date(self._asia_bars[0]['timestamp']) != session_date:
                self._asia_bars = []
            
            self._asia_bars.append(bar)
            self._in_london = False
            return None
        
        # After Asia, before London - process Asia range
        if not self._is_london_session(timestamp) and timestamp.time() >= self.ASIA_END and timestamp.time() < self.LONDON_START:
            if self._asia_bars:
                self._finalize_asia_range(session_date)
            return None
        
        # London session - check for breakout entry
        if self._is_london_session(timestamp):
            self._in_london = True
            
            # Finalize Asia if not done
            if self._asia_bars and self._current_asia is None:
                self._finalize_asia_range(session_date)
            
            # Check for entry
            return self._check_london_entry(bar, session_date)
        
        # After London - reset for next day
        if timestamp.time() >= self.LONDON_END:
            self._in_london = False
            self._pending_signal = None
            return None
        
        return None
    
    def _finalize_asia_range(self, session_date: datetime) -> None:
        """Calculate Asia range from accumulated bars."""
        if not self._asia_bars:
            return
        
        highs = [b['high'] for b in self._asia_bars]
        lows = [b['low'] for b in self._asia_bars]
        
        asia_high = max(highs)
        asia_low = min(lows)
        asia_open = self._asia_bars[0]['open']
        asia_close = self._asia_bars[-1]['close']
        
        self._current_asia = AsiaRange(
            date=session_date,
            high=asia_high,
            low=asia_low,
            open=asia_open,
            close=asia_close,
        )
        
        self._sessions_analyzed += 1
        self._asia_bars = []  # Clear for next day
        
        if self._current_asia.is_valid:
            if self._current_asia.bias != SessionBias.NEUTRAL:
                self._valid_setups += 1
            else:
                self._neutral_biases += 1
    
    def _check_london_entry(self, bar: dict, session_date: datetime) -> Optional[SessionSignal]:
        """Check for London session breakout entry."""
        if self._current_asia is None:
            return None
        
        # Only 1 trade per day
        if self._last_trade_date == session_date:
            return None
        
        # Check if Asia range is valid
        if not self._current_asia.is_valid:
            return None
        
        # Get bias
        bias = self._current_asia.bias
        if bias == SessionBias.NEUTRAL:
            return None
        
        asia = self._current_asia
        high = bar['high']
        low = bar['low']
        close = bar['close']
        timestamp = bar['timestamp']
        
        # SL buffer in price terms
        sl_buffer = self.SL_BUFFER_PIPS * 0.0001
        
        signal = None
        
        # BULLISH bias: Look for break above Asia high
        if bias == SessionBias.BULLISH:
            if high > asia.high and close > asia.high:
                entry_price = close
                stop_loss = asia.low - sl_buffer
                sl_distance = entry_price - stop_loss
                take_profit = entry_price + sl_distance * self.RISK_REWARD
                
                signal = SessionSignal(
                    date=session_date,
                    timestamp=timestamp,
                    direction=SessionDirection.LONG,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    asia_high=asia.high,
                    asia_low=asia.low,
                    asia_close=asia.close,
                    asia_bias=bias,
                    asia_range_pips=asia.range_pips,
                )
        
        # BEARISH bias: Look for break below Asia low
        elif bias == SessionBias.BEARISH:
            if low < asia.low and close < asia.low:
                entry_price = close
                stop_loss = asia.high + sl_buffer
                sl_distance = stop_loss - entry_price
                take_profit = entry_price - sl_distance * self.RISK_REWARD
                
                signal = SessionSignal(
                    date=session_date,
                    timestamp=timestamp,
                    direction=SessionDirection.SHORT,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    asia_high=asia.high,
                    asia_low=asia.low,
                    asia_close=asia.close,
                    asia_bias=bias,
                    asia_range_pips=asia.range_pips,
                )
        
        if signal is not None:
            self._last_trade_date = session_date
        
        return signal
    
    def get_current_asia_range(self) -> Optional[AsiaRange]:
        """Get current Asia range (for debugging/display)."""
        return self._current_asia
    
    def reset(self) -> None:
        """Reset engine state."""
        self._current_asia = None
        self._asia_bars = []
        self._last_trade_date = None
        self._pending_signal = None
        self._in_london = False
        self._sessions_analyzed = 0
        self._valid_setups = 0
        self._neutral_biases = 0
