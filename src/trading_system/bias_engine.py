"""
Session Directional Bias Engine

Converts the session strategy from a TRADE GENERATOR to a DIRECTIONAL FILTER.

KEY INSIGHT:
- The Asia → London strategy has ~53% directional accuracy
- 0.03R expectancy is too thin for standalone trading
- BUT: This directional accuracy is VALUABLE as a filter for other strategies

OUTPUT:
- BULL / BEAR / NEUTRAL bias for each session
- Confidence score (0-1) based on signal strength
- Used by complementary strategies to filter trades

This engine does NOT generate trades directly.
It provides context for other engines to use.
"""

import numpy as np
from dataclasses import dataclass
from datetime import datetime, time as dt_time, timedelta
from typing import Optional, List, Dict, Tuple
from enum import Enum


class DirectionalBias(Enum):
    BULL = "BULL"
    BEAR = "BEAR"
    NEUTRAL = "NEUTRAL"


class TradingSession(Enum):
    ASIA = "ASIA"           # 00:00 - 06:00 UTC
    LONDON = "LONDON"       # 07:00 - 11:00 UTC
    NY_OVERLAP = "NY_OVERLAP"  # 12:00 - 16:00 UTC
    NY_LATE = "NY_LATE"     # 16:00 - 21:00 UTC


@dataclass
class SessionBiasOutput:
    """Output from the bias engine for a specific session."""
    session: TradingSession
    bias: DirectionalBias
    confidence: float  # 0.0 to 1.0
    
    # Supporting data
    asia_high: float
    asia_low: float
    asia_mid: float
    asia_close: float
    asia_range_pips: float
    
    # Derived levels
    bull_trigger: float  # Price above this confirms bull
    bear_trigger: float  # Price below this confirms bear
    pivot: float         # Key level (Asia mid)
    
    # Timing
    valid_from: datetime
    valid_until: datetime
    
    def is_valid_at(self, timestamp: datetime) -> bool:
        """Check if bias is still valid at given time."""
        return self.valid_from <= timestamp < self.valid_until
    
    def allows_long(self) -> bool:
        """Returns True if bias allows long trades."""
        return self.bias == DirectionalBias.BULL
    
    def allows_short(self) -> bool:
        """Returns True if bias allows short trades."""
        return self.bias == DirectionalBias.BEAR
    
    def to_dict(self) -> dict:
        return {
            'session': self.session.value,
            'bias': self.bias.value,
            'confidence': self.confidence,
            'asia_range_pips': self.asia_range_pips,
            'pivot': self.pivot,
            'valid_from': self.valid_from.isoformat() if self.valid_from else None,
            'valid_until': self.valid_until.isoformat() if self.valid_until else None,
        }


@dataclass
class BiasState:
    """Current state of all session biases."""
    date: datetime
    london_bias: Optional[SessionBiasOutput] = None
    ny_bias: Optional[SessionBiasOutput] = None
    
    # Intraday updates
    london_confirmed: bool = False  # Bias confirmed by price action
    london_invalidated: bool = False  # Bias failed
    
    def get_active_bias(self, timestamp: datetime) -> Optional[SessionBiasOutput]:
        """Get the currently active bias for a timestamp."""
        t = timestamp.time()
        
        # London session
        if dt_time(7, 0) <= t < dt_time(12, 0):
            return self.london_bias
        
        # NY session
        if dt_time(12, 0) <= t < dt_time(21, 0):
            return self.ny_bias
        
        return None


class SessionBiasEngine:
    """
    Generates directional bias for trading sessions.
    
    This is a FILTER, not a trade generator.
    Other strategies use this bias to filter their entries.
    
    BIAS DETERMINATION:
    
    1. Asia Session (00:00-06:00 UTC):
       - Establish range: High, Low, Midpoint
       - Determine bias from close position:
         * Close > Mid + 25% range → BULL (strong)
         * Close > Mid + 10% range → BULL (moderate)
         * Close < Mid - 25% range → BEAR (strong)
         * Close < Mid - 10% range → BEAR (moderate)
         * Otherwise → NEUTRAL
    
    2. Confidence Score:
       - Based on how far close is from midpoint
       - Scaled 0.5 to 1.0 (never below 0.5 for non-neutral)
       - NEUTRAL always has 0.0 confidence
    
    3. London Bias:
       - Inherits Asia bias
       - Valid 07:00 - 12:00 UTC
       - Can be confirmed/invalidated by price action
    
    4. NY Bias:
       - Starts with Asia bias
       - Modified by London outcome:
         * London confirmed → NY continues
         * London failed → NY may reverse
    """
    
    # Session times (UTC)
    ASIA_START = dt_time(0, 0)
    ASIA_END = dt_time(6, 0)
    LONDON_START = dt_time(7, 0)
    LONDON_END = dt_time(12, 0)
    NY_START = dt_time(12, 0)
    NY_END = dt_time(21, 0)
    
    # Bias thresholds
    STRONG_THRESHOLD = 0.25  # 25% of range from mid = strong bias
    MODERATE_THRESHOLD = 0.10  # 10% of range from mid = moderate bias
    
    # Range filters
    MIN_RANGE_PIPS = 15
    MAX_RANGE_PIPS = 80
    
    def __init__(self):
        self._asia_bars: List[dict] = []
        self._current_state: Optional[BiasState] = None
        self._bar_count = 0
    
    @property
    def current_state(self) -> Optional[BiasState]:
        return self._current_state
    
    def update(
        self,
        timestamp: datetime,
        open_: float,
        high: float,
        low: float,
        close: float,
    ) -> Optional[BiasState]:
        """
        Process new bar and update bias state.
        
        Returns BiasState when a new session bias is established.
        """
        self._bar_count += 1
        bar = {'timestamp': timestamp, 'open': open_, 'high': high, 'low': low, 'close': close}
        
        t = timestamp.time()
        session_date = self._get_session_date(timestamp)
        
        # Reset for new day
        if self._current_state is None or self._current_state.date.date() != session_date.date():
            self._current_state = BiasState(date=session_date)
            self._asia_bars = []
        
        # Accumulate Asia bars
        if self._is_asia_session(t):
            self._asia_bars.append(bar)
            return None
        
        # After Asia, before London - calculate bias
        if self.ASIA_END <= t < self.LONDON_START:
            if self._asia_bars and self._current_state.london_bias is None:
                self._calculate_session_biases(session_date)
            return self._current_state
        
        # During London - update confirmation status
        if self._is_london_session(t):
            if self._current_state.london_bias:
                self._update_london_confirmation(high, low, close)
            return self._current_state
        
        # During NY - update NY bias based on London outcome
        if self._is_ny_session(t):
            if self._current_state.ny_bias is None:
                self._update_ny_bias(session_date)
            return self._current_state
        
        return self._current_state
    
    def _get_session_date(self, ts: datetime) -> datetime:
        """Get trading date (Asia session defines the day)."""
        return ts.replace(hour=0, minute=0, second=0, microsecond=0)
    
    def _is_asia_session(self, t: dt_time) -> bool:
        return self.ASIA_START <= t < self.ASIA_END
    
    def _is_london_session(self, t: dt_time) -> bool:
        return self.LONDON_START <= t < self.LONDON_END
    
    def _is_ny_session(self, t: dt_time) -> bool:
        return self.NY_START <= t < self.NY_END
    
    def _calculate_session_biases(self, session_date: datetime) -> None:
        """Calculate bias from Asia range."""
        if not self._asia_bars:
            return
        
        # Calculate Asia range
        highs = [b['high'] for b in self._asia_bars]
        lows = [b['low'] for b in self._asia_bars]
        
        asia_high = max(highs)
        asia_low = min(lows)
        asia_mid = (asia_high + asia_low) / 2
        asia_close = self._asia_bars[-1]['close']
        asia_range = asia_high - asia_low
        asia_range_pips = asia_range / 0.0001
        
        # Check range validity
        if not (self.MIN_RANGE_PIPS <= asia_range_pips <= self.MAX_RANGE_PIPS):
            # Invalid range - set neutral
            self._set_neutral_bias(session_date, asia_high, asia_low, asia_mid, asia_close, asia_range_pips)
            return
        
        # Determine bias and confidence
        close_position = (asia_close - asia_mid) / asia_range if asia_range > 0 else 0
        
        if close_position > self.STRONG_THRESHOLD:
            bias = DirectionalBias.BULL
            confidence = min(1.0, 0.7 + abs(close_position))
        elif close_position > self.MODERATE_THRESHOLD:
            bias = DirectionalBias.BULL
            confidence = 0.5 + abs(close_position)
        elif close_position < -self.STRONG_THRESHOLD:
            bias = DirectionalBias.BEAR
            confidence = min(1.0, 0.7 + abs(close_position))
        elif close_position < -self.MODERATE_THRESHOLD:
            bias = DirectionalBias.BEAR
            confidence = 0.5 + abs(close_position)
        else:
            bias = DirectionalBias.NEUTRAL
            confidence = 0.0
        
        # Create London bias
        self._current_state.london_bias = SessionBiasOutput(
            session=TradingSession.LONDON,
            bias=bias,
            confidence=confidence,
            asia_high=asia_high,
            asia_low=asia_low,
            asia_mid=asia_mid,
            asia_close=asia_close,
            asia_range_pips=asia_range_pips,
            bull_trigger=asia_high,
            bear_trigger=asia_low,
            pivot=asia_mid,
            valid_from=session_date.replace(hour=7, minute=0),
            valid_until=session_date.replace(hour=12, minute=0),
        )
        
        # Initial NY bias (same as London, will be updated)
        self._current_state.ny_bias = SessionBiasOutput(
            session=TradingSession.NY_OVERLAP,
            bias=bias,
            confidence=confidence * 0.8,  # Slightly less confident
            asia_high=asia_high,
            asia_low=asia_low,
            asia_mid=asia_mid,
            asia_close=asia_close,
            asia_range_pips=asia_range_pips,
            bull_trigger=asia_high,
            bear_trigger=asia_low,
            pivot=asia_mid,
            valid_from=session_date.replace(hour=12, minute=0),
            valid_until=session_date.replace(hour=21, minute=0),
        )
    
    def _set_neutral_bias(
        self,
        session_date: datetime,
        asia_high: float,
        asia_low: float,
        asia_mid: float,
        asia_close: float,
        asia_range_pips: float,
    ) -> None:
        """Set neutral bias for invalid conditions."""
        self._current_state.london_bias = SessionBiasOutput(
            session=TradingSession.LONDON,
            bias=DirectionalBias.NEUTRAL,
            confidence=0.0,
            asia_high=asia_high,
            asia_low=asia_low,
            asia_mid=asia_mid,
            asia_close=asia_close,
            asia_range_pips=asia_range_pips,
            bull_trigger=asia_high,
            bear_trigger=asia_low,
            pivot=asia_mid,
            valid_from=session_date.replace(hour=7, minute=0),
            valid_until=session_date.replace(hour=12, minute=0),
        )
        
        self._current_state.ny_bias = SessionBiasOutput(
            session=TradingSession.NY_OVERLAP,
            bias=DirectionalBias.NEUTRAL,
            confidence=0.0,
            asia_high=asia_high,
            asia_low=asia_low,
            asia_mid=asia_mid,
            asia_close=asia_close,
            asia_range_pips=asia_range_pips,
            bull_trigger=asia_high,
            bear_trigger=asia_low,
            pivot=asia_mid,
            valid_from=session_date.replace(hour=12, minute=0),
            valid_until=session_date.replace(hour=21, minute=0),
        )
    
    def _update_london_confirmation(self, high: float, low: float, close: float) -> None:
        """Update London bias confirmation based on price action."""
        bias = self._current_state.london_bias
        if bias is None or bias.bias == DirectionalBias.NEUTRAL:
            return
        
        if bias.bias == DirectionalBias.BULL:
            # Confirmed if price broke above Asia high
            if high > bias.bull_trigger:
                self._current_state.london_confirmed = True
            # Invalidated if price broke below Asia low
            if low < bias.bear_trigger:
                self._current_state.london_invalidated = True
        
        elif bias.bias == DirectionalBias.BEAR:
            # Confirmed if price broke below Asia low
            if low < bias.bear_trigger:
                self._current_state.london_confirmed = True
            # Invalidated if price broke above Asia high
            if high > bias.bull_trigger:
                self._current_state.london_invalidated = True
    
    def _update_ny_bias(self, session_date: datetime) -> None:
        """Update NY bias based on London outcome."""
        london = self._current_state.london_bias
        if london is None:
            return
        
        # If London confirmed, NY continues with higher confidence
        if self._current_state.london_confirmed and not self._current_state.london_invalidated:
            new_confidence = min(1.0, london.confidence + 0.2)
            bias = london.bias
        # If London invalidated, NY reverses
        elif self._current_state.london_invalidated and not self._current_state.london_confirmed:
            new_confidence = 0.5  # Lower confidence on reversal
            if london.bias == DirectionalBias.BULL:
                bias = DirectionalBias.BEAR
            elif london.bias == DirectionalBias.BEAR:
                bias = DirectionalBias.BULL
            else:
                bias = DirectionalBias.NEUTRAL
        # Mixed signals - go neutral
        elif self._current_state.london_confirmed and self._current_state.london_invalidated:
            bias = DirectionalBias.NEUTRAL
            new_confidence = 0.0
        else:
            # No confirmation either way - slight reduction in confidence
            bias = london.bias
            new_confidence = london.confidence * 0.7
        
        self._current_state.ny_bias = SessionBiasOutput(
            session=TradingSession.NY_OVERLAP,
            bias=bias,
            confidence=new_confidence,
            asia_high=london.asia_high,
            asia_low=london.asia_low,
            asia_mid=london.asia_mid,
            asia_close=london.asia_close,
            asia_range_pips=london.asia_range_pips,
            bull_trigger=london.bull_trigger,
            bear_trigger=london.bear_trigger,
            pivot=london.pivot,
            valid_from=session_date.replace(hour=12, minute=0),
            valid_until=session_date.replace(hour=21, minute=0),
        )
    
    def get_bias_at(self, timestamp: datetime) -> Optional[SessionBiasOutput]:
        """Get active bias for a specific timestamp."""
        if self._current_state is None:
            return None
        return self._current_state.get_active_bias(timestamp)
    
    def reset(self) -> None:
        """Reset engine state."""
        self._asia_bars = []
        self._current_state = None
        self._bar_count = 0
