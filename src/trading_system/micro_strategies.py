"""
Complementary Micro-Edge Strategies

Three high-frequency strategies that use the Session Bias Engine as a filter.
Each targets small R (0.3-0.8R) with high frequency.

STRATEGY 1: London Pullback Scalper
- Trade pullbacks during London aligned with Asia bias
- Entry: Price pulls back to key level, then shows momentum in bias direction
- Target: 0.5R, Stop: At Asia range opposite side

STRATEGY 2: Momentum Burst Strategy  
- Catch quick momentum moves aligned with bias
- Entry: Strong candle close in bias direction
- Target: 0.4R, Stop: 1.5 × ATR

STRATEGY 3: Pivot Bounce Strategy
- Trade bounces off Asia midpoint (pivot) aligned with bias
- Entry: Price touches pivot, reverses in bias direction
- Target: 0.6R to Asia range boundary
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, time as dt_time
from typing import Optional, List, Tuple
from enum import Enum

from .bias_engine import SessionBiasEngine, SessionBiasOutput, DirectionalBias, BiasState


class MicroEdgeDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class MicroSignal:
    """Signal from a micro-edge strategy."""
    strategy_name: str
    timestamp: datetime
    direction: MicroEdgeDirection
    entry_price: float
    stop_loss: float
    take_profit: float
    
    # Context
    bias: DirectionalBias
    bias_confidence: float
    
    # Strategy-specific
    trigger_type: str = ""  # "PULLBACK", "MOMENTUM", "PIVOT_BOUNCE"
    
    @property
    def sl_distance(self) -> float:
        return abs(self.entry_price - self.stop_loss)
    
    @property
    def tp_distance(self) -> float:
        return abs(self.take_profit - self.entry_price)
    
    @property
    def risk_reward(self) -> float:
        return self.tp_distance / self.sl_distance if self.sl_distance > 0 else 0


class BaseMicroStrategy(ABC):
    """Base class for micro-edge strategies."""
    
    def __init__(self, name: str, target_r: float):
        self.name = name
        self.target_r = target_r
        self._signal_count = 0
    
    @abstractmethod
    def check_entry(
        self,
        timestamp: datetime,
        open_: float,
        high: float,
        low: float,
        close: float,
        atr: float,
        bias: SessionBiasOutput,
    ) -> Optional[MicroSignal]:
        """Check for entry signal. Returns MicroSignal if conditions met."""
        pass
    
    @property
    def signal_count(self) -> int:
        return self._signal_count


# =============================================================================
# STRATEGY 1: London Pullback Scalper
# =============================================================================

class LondonPullbackScalper(BaseMicroStrategy):
    """
    Trade pullbacks during London session aligned with Asia bias.
    
    LOGIC:
    - If BULL bias: Wait for price to pull back toward Asia mid, then buy
    - If BEAR bias: Wait for price to rally toward Asia mid, then sell
    
    ENTRY CONDITIONS:
    1. Active London session (07:00-11:00 UTC)
    2. Non-neutral bias with confidence > 0.5
    3. Price in "pullback zone" (between Asia mid and Asia boundary)
    4. Current candle shows reversal in bias direction
    
    EXIT:
    - TP: 0.5R (50% of risk distance)
    - SL: Opposite Asia boundary + small buffer
    
    TARGET: 0.5R
    EXPECTED FREQUENCY: 1-3 per London session on active days
    """
    
    LONDON_START = dt_time(7, 0)
    LONDON_END = dt_time(11, 0)
    
    MIN_CONFIDENCE = 0.5
    TARGET_R = 0.5
    
    def __init__(self):
        super().__init__("PULLBACK_SCALPER", self.TARGET_R)
        self._last_signal_time: Optional[datetime] = None
        self._cooldown_minutes = 30  # Min time between signals
    
    def check_entry(
        self,
        timestamp: datetime,
        open_: float,
        high: float,
        low: float,
        close: float,
        atr: float,
        bias: SessionBiasOutput,
    ) -> Optional[MicroSignal]:
        """Check for pullback entry."""
        # Check session
        t = timestamp.time()
        if not (self.LONDON_START <= t < self.LONDON_END):
            return None
        
        # Check bias
        if bias.bias == DirectionalBias.NEUTRAL:
            return None
        
        if bias.confidence < self.MIN_CONFIDENCE:
            return None
        
        # Check cooldown
        if self._last_signal_time:
            delta = (timestamp - self._last_signal_time).total_seconds() / 60
            if delta < self._cooldown_minutes:
                return None
        
        signal = None
        buffer = atr * 0.2  # Small buffer for SL
        
        if bias.bias == DirectionalBias.BULL:
            # Pullback zone: price between mid and low
            if bias.asia_low < low <= bias.pivot:
                # Reversal candle: Close > Open (bullish)
                if close > open_ and (close - open_) > atr * 0.3:
                    entry = close
                    sl = bias.asia_low - buffer
                    sl_dist = entry - sl
                    tp = entry + sl_dist * self.TARGET_R
                    
                    signal = MicroSignal(
                        strategy_name=self.name,
                        timestamp=timestamp,
                        direction=MicroEdgeDirection.LONG,
                        entry_price=entry,
                        stop_loss=sl,
                        take_profit=tp,
                        bias=bias.bias,
                        bias_confidence=bias.confidence,
                        trigger_type="PULLBACK",
                    )
        
        elif bias.bias == DirectionalBias.BEAR:
            # Pullback zone: price between mid and high
            if bias.pivot <= high < bias.asia_high:
                # Reversal candle: Close < Open (bearish)
                if close < open_ and (open_ - close) > atr * 0.3:
                    entry = close
                    sl = bias.asia_high + buffer
                    sl_dist = sl - entry
                    tp = entry - sl_dist * self.TARGET_R
                    
                    signal = MicroSignal(
                        strategy_name=self.name,
                        timestamp=timestamp,
                        direction=MicroEdgeDirection.SHORT,
                        entry_price=entry,
                        stop_loss=sl,
                        take_profit=tp,
                        bias=bias.bias,
                        bias_confidence=bias.confidence,
                        trigger_type="PULLBACK",
                    )
        
        if signal:
            self._signal_count += 1
            self._last_signal_time = timestamp
        
        return signal


# =============================================================================
# STRATEGY 2: Momentum Burst Strategy
# =============================================================================

class MomentumBurstStrategy(BaseMicroStrategy):
    """
    Catch quick momentum moves aligned with bias.
    
    LOGIC:
    - Look for strong momentum candles in bias direction
    - Entry on close of momentum candle
    - Quick exit at small target
    
    ENTRY CONDITIONS:
    1. Active session (London or NY overlap)
    2. Non-neutral bias
    3. Strong candle: Body > 1.2 × ATR
    4. Candle direction matches bias
    5. Volume/momentum confirmation (close near candle extreme)
    
    EXIT:
    - TP: 0.4R
    - SL: 1.5 × ATR from entry
    
    TARGET: 0.4R
    EXPECTED FREQUENCY: 2-4 per day on volatile days
    """
    
    ACTIVE_START = dt_time(7, 0)
    ACTIVE_END = dt_time(17, 0)
    
    BODY_ATR_MULT = 1.2  # Candle body must be > 1.2 × ATR
    SL_ATR_MULT = 1.5
    TARGET_R = 0.4
    
    def __init__(self):
        super().__init__("MOMENTUM_BURST", self.TARGET_R)
        self._last_signal_time: Optional[datetime] = None
        self._cooldown_minutes = 20
    
    def check_entry(
        self,
        timestamp: datetime,
        open_: float,
        high: float,
        low: float,
        close: float,
        atr: float,
        bias: SessionBiasOutput,
    ) -> Optional[MicroSignal]:
        """Check for momentum entry."""
        # Check session
        t = timestamp.time()
        if not (self.ACTIVE_START <= t < self.ACTIVE_END):
            return None
        
        # Check bias
        if bias.bias == DirectionalBias.NEUTRAL:
            return None
        
        # Check cooldown
        if self._last_signal_time:
            delta = (timestamp - self._last_signal_time).total_seconds() / 60
            if delta < self._cooldown_minutes:
                return None
        
        # Check candle size
        body = abs(close - open_)
        if body < atr * self.BODY_ATR_MULT:
            return None
        
        signal = None
        
        if bias.bias == DirectionalBias.BULL:
            # Bullish momentum: Close > Open, close near high
            if close > open_:
                upper_wick = high - close
                if upper_wick < body * 0.3:  # Close near high
                    entry = close
                    sl = entry - atr * self.SL_ATR_MULT
                    sl_dist = entry - sl
                    tp = entry + sl_dist * self.TARGET_R
                    
                    signal = MicroSignal(
                        strategy_name=self.name,
                        timestamp=timestamp,
                        direction=MicroEdgeDirection.LONG,
                        entry_price=entry,
                        stop_loss=sl,
                        take_profit=tp,
                        bias=bias.bias,
                        bias_confidence=bias.confidence,
                        trigger_type="MOMENTUM",
                    )
        
        elif bias.bias == DirectionalBias.BEAR:
            # Bearish momentum: Close < Open, close near low
            if close < open_:
                lower_wick = close - low
                if lower_wick < body * 0.3:  # Close near low
                    entry = close
                    sl = entry + atr * self.SL_ATR_MULT
                    sl_dist = sl - entry
                    tp = entry - sl_dist * self.TARGET_R
                    
                    signal = MicroSignal(
                        strategy_name=self.name,
                        timestamp=timestamp,
                        direction=MicroEdgeDirection.SHORT,
                        entry_price=entry,
                        stop_loss=sl,
                        take_profit=tp,
                        bias=bias.bias,
                        bias_confidence=bias.confidence,
                        trigger_type="MOMENTUM",
                    )
        
        if signal:
            self._signal_count += 1
            self._last_signal_time = timestamp
        
        return signal


# =============================================================================
# STRATEGY 3: Pivot Bounce Strategy
# =============================================================================

class PivotBounceStrategy(BaseMicroStrategy):
    """
    Trade bounces off Asia midpoint (pivot) aligned with bias.
    
    LOGIC:
    - Asia midpoint is a key level
    - If BULL bias: Buy bounces off pivot from above
    - If BEAR bias: Sell bounces off pivot from below
    
    ENTRY CONDITIONS:
    1. Active London or early NY session
    2. Non-neutral bias with confidence > 0.6
    3. Price touches pivot zone (within 0.3 × ATR)
    4. Candle shows rejection (wick > body in correct direction)
    5. Bias-aligned reversal
    
    EXIT:
    - TP: 0.6R (to Asia boundary)
    - SL: Through pivot by 0.5 × Asia range
    
    TARGET: 0.6R
    EXPECTED FREQUENCY: 0-2 per day
    """
    
    ACTIVE_START = dt_time(7, 0)
    ACTIVE_END = dt_time(16, 0)
    
    MIN_CONFIDENCE = 0.6
    PIVOT_ZONE_ATR_MULT = 0.3  # How close to pivot to trigger
    TARGET_R = 0.6
    
    def __init__(self):
        super().__init__("PIVOT_BOUNCE", self.TARGET_R)
        self._last_signal_time: Optional[datetime] = None
        self._cooldown_minutes = 60  # Longer cooldown for pivot trades
    
    def check_entry(
        self,
        timestamp: datetime,
        open_: float,
        high: float,
        low: float,
        close: float,
        atr: float,
        bias: SessionBiasOutput,
    ) -> Optional[MicroSignal]:
        """Check for pivot bounce entry."""
        # Check session
        t = timestamp.time()
        if not (self.ACTIVE_START <= t < self.ACTIVE_END):
            return None
        
        # Check bias
        if bias.bias == DirectionalBias.NEUTRAL:
            return None
        
        if bias.confidence < self.MIN_CONFIDENCE:
            return None
        
        # Check cooldown
        if self._last_signal_time:
            delta = (timestamp - self._last_signal_time).total_seconds() / 60
            if delta < self._cooldown_minutes:
                return None
        
        pivot = bias.pivot
        pivot_zone = atr * self.PIVOT_ZONE_ATR_MULT
        asia_range = bias.asia_high - bias.asia_low
        
        signal = None
        
        if bias.bias == DirectionalBias.BULL:
            # Touched pivot from above and bounced up
            if low <= pivot + pivot_zone and close > pivot:
                lower_wick = min(open_, close) - low
                body = abs(close - open_)
                
                # Rejection: Lower wick > body (buying pressure)
                if lower_wick > body and close > open_:
                    entry = close
                    sl = pivot - asia_range * 0.5
                    sl_dist = entry - sl
                    tp = entry + sl_dist * self.TARGET_R
                    
                    signal = MicroSignal(
                        strategy_name=self.name,
                        timestamp=timestamp,
                        direction=MicroEdgeDirection.LONG,
                        entry_price=entry,
                        stop_loss=sl,
                        take_profit=tp,
                        bias=bias.bias,
                        bias_confidence=bias.confidence,
                        trigger_type="PIVOT_BOUNCE",
                    )
        
        elif bias.bias == DirectionalBias.BEAR:
            # Touched pivot from below and bounced down
            if high >= pivot - pivot_zone and close < pivot:
                upper_wick = high - max(open_, close)
                body = abs(close - open_)
                
                # Rejection: Upper wick > body (selling pressure)
                if upper_wick > body and close < open_:
                    entry = close
                    sl = pivot + asia_range * 0.5
                    sl_dist = sl - entry
                    tp = entry - sl_dist * self.TARGET_R
                    
                    signal = MicroSignal(
                        strategy_name=self.name,
                        timestamp=timestamp,
                        direction=MicroEdgeDirection.SHORT,
                        entry_price=entry,
                        stop_loss=sl,
                        take_profit=tp,
                        bias=bias.bias,
                        bias_confidence=bias.confidence,
                        trigger_type="PIVOT_BOUNCE",
                    )
        
        if signal:
            self._signal_count += 1
            self._last_signal_time = timestamp
        
        return signal


# =============================================================================
# PORTFOLIO COORDINATOR
# =============================================================================

@dataclass
class PortfolioSignal:
    """A signal from any strategy, with portfolio context."""
    signal: MicroSignal
    strategy_priority: int  # Lower = higher priority
    allocated_risk: float   # Fraction of daily risk budget


class MicroEdgePortfolio:
    """
    Coordinates multiple micro-edge strategies.
    
    RESPONSIBILITIES:
    - Maintain bias engine state
    - Collect signals from all strategies
    - Apply risk allocation rules
    - Enforce daily limits
    
    CAPITAL ALLOCATION:
    - Max 2% daily risk
    - Each strategy gets equal base allocation
    - Higher confidence bias → larger allocation
    - Correlation control: Max 2 concurrent trades same direction
    """
    
    MAX_DAILY_RISK = 0.02  # 2% max daily risk
    MAX_CONCURRENT_SAME_DIR = 2  # Max positions same direction
    BASE_RISK_PER_TRADE = 0.003  # 0.3% per trade
    
    def __init__(self):
        self.bias_engine = SessionBiasEngine()
        
        self.strategies = [
            LondonPullbackScalper(),
            MomentumBurstStrategy(),
            PivotBounceStrategy(),
        ]
        
        # State
        self._daily_risk_used = 0.0
        self._current_date: Optional[datetime] = None
        self._long_count = 0
        self._short_count = 0
        self._atr_history: List[float] = []
    
    def update(
        self,
        timestamp: datetime,
        open_: float,
        high: float,
        low: float,
        close: float,
    ) -> List[PortfolioSignal]:
        """
        Process new bar and collect signals from all strategies.
        
        Returns list of valid signals with risk allocation.
        """
        # Reset daily counters
        session_date = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        if self._current_date != session_date:
            self._current_date = session_date
            self._daily_risk_used = 0.0
            self._long_count = 0
            self._short_count = 0
        
        # Update bias engine
        self.bias_engine.update(timestamp, open_, high, low, close)
        
        # Get current bias
        bias = self.bias_engine.get_bias_at(timestamp)
        if bias is None:
            return []
        
        # Calculate ATR (simplified)
        self._update_atr(high, low, close)
        atr = self._get_atr()
        if atr <= 0:
            return []
        
        # Check daily risk limit
        if self._daily_risk_used >= self.MAX_DAILY_RISK:
            return []
        
        # Collect signals from all strategies
        signals = []
        
        for priority, strategy in enumerate(self.strategies):
            signal = strategy.check_entry(
                timestamp=timestamp,
                open_=open_,
                high=high,
                low=low,
                close=close,
                atr=atr,
                bias=bias,
            )
            
            if signal:
                # Check correlation limit
                if signal.direction == MicroEdgeDirection.LONG:
                    if self._long_count >= self.MAX_CONCURRENT_SAME_DIR:
                        continue
                else:
                    if self._short_count >= self.MAX_CONCURRENT_SAME_DIR:
                        continue
                
                # Calculate risk allocation
                remaining_risk = self.MAX_DAILY_RISK - self._daily_risk_used
                base_risk = min(self.BASE_RISK_PER_TRADE, remaining_risk)
                
                # Adjust by confidence
                adjusted_risk = base_risk * (0.5 + 0.5 * bias.confidence)
                
                if adjusted_risk > 0:
                    portfolio_signal = PortfolioSignal(
                        signal=signal,
                        strategy_priority=priority,
                        allocated_risk=adjusted_risk,
                    )
                    signals.append(portfolio_signal)
        
        return signals
    
    def register_trade_opened(self, direction: MicroEdgeDirection, risk: float) -> None:
        """Register that a trade was opened."""
        self._daily_risk_used += risk
        if direction == MicroEdgeDirection.LONG:
            self._long_count += 1
        else:
            self._short_count += 1
    
    def register_trade_closed(self, direction: MicroEdgeDirection) -> None:
        """Register that a trade was closed."""
        if direction == MicroEdgeDirection.LONG:
            self._long_count = max(0, self._long_count - 1)
        else:
            self._short_count = max(0, self._short_count - 1)
    
    def _update_atr(self, high: float, low: float, close: float) -> None:
        """Update ATR calculation."""
        tr = high - low
        if self._atr_history:
            prev_close = self._atr_history[-1]
            tr = max(tr, abs(high - prev_close), abs(low - prev_close))
        
        self._atr_history.append(tr)
        
        # Keep last 14 bars
        if len(self._atr_history) > 20:
            self._atr_history = self._atr_history[-20:]
    
    def _get_atr(self, period: int = 14) -> float:
        """Get current ATR value."""
        if len(self._atr_history) < period:
            return np.mean(self._atr_history) if self._atr_history else 0
        return np.mean(self._atr_history[-period:])
    
    def get_status(self) -> dict:
        """Get portfolio status."""
        bias_state = self.bias_engine.current_state
        
        return {
            'daily_risk_used': self._daily_risk_used,
            'daily_risk_remaining': self.MAX_DAILY_RISK - self._daily_risk_used,
            'long_positions': self._long_count,
            'short_positions': self._short_count,
            'london_bias': bias_state.london_bias.bias.value if bias_state and bias_state.london_bias else None,
            'ny_bias': bias_state.ny_bias.bias.value if bias_state and bias_state.ny_bias else None,
            'strategies': {
                s.name: {'signal_count': s.signal_count}
                for s in self.strategies
            },
        }
