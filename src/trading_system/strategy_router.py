"""
Strategy Router - Multi-Strategy Signal Selection

Coordinates multiple strategy engines and applies priority-based
signal selection with shared gatekeeper and risk management.

PRIORITY ORDER (LOCKED):
1. Trend Continuation (highest)
2. Trend Pullback
3. Volatility Expansion (lowest)

Only ONE trade may be opened per bar.
Gatekeeper applied BEFORE any trade is allowed.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Union, Tuple
from enum import Enum

from .signal_engine import SignalEngine, TradeSignal, SignalDirection
from .pullback_engine import PullbackEngine, PullbackSignal, PullbackDirection
from .volatility_expansion_engine import VolatilityExpansionEngine, VolExpSignal, VolExpDirection
from .gatekeeper_engine import GatekeeperEngine, GatekeeperDecision
from .config import StrategyConfig, GatekeeperConfig


class StrategyType(Enum):
    """Strategy identification."""
    TREND_CONTINUATION = "TREND_CONTINUATION"
    TREND_PULLBACK = "TREND_PULLBACK"
    VOLATILITY_EXPANSION = "VOLATILITY_EXPANSION"


# Type alias for any signal type
AnySignal = Union[TradeSignal, PullbackSignal, VolExpSignal]


@dataclass
class RouterDecision:
    """
    Complete routing decision with audit trail.
    
    Contains selected signal, gatekeeper decision, and
    full details of all signals considered.
    """
    bar_index: int
    timestamp: datetime
    
    # Selected signal (if any)
    selected_signal: Optional[AnySignal]
    selected_strategy: Optional[StrategyType]
    
    # Gatekeeper
    gatekeeper_decision: Optional[GatekeeperDecision]
    was_blocked: bool
    
    # All signals generated this bar (for audit)
    signals_generated: Dict[StrategyType, AnySignal]
    
    # Risk allocation
    risk_allocation: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "bar_index": self.bar_index,
            "timestamp": self.timestamp.isoformat(),
            "selected_strategy": self.selected_strategy.value if self.selected_strategy else None,
            "was_blocked": self.was_blocked,
            "block_reasons": [r.value for r in self.gatekeeper_decision.reasons] if self.gatekeeper_decision else [],
            "signals_generated": list(self.signals_generated.keys()),
            "risk_allocation": self.risk_allocation,
        }


class StrategyRouter:
    """
    Multi-strategy signal router.
    
    Manages multiple strategy engines, applies priority-based
    signal selection, and coordinates with gatekeeper.
    
    ARCHITECTURE:
    - Each strategy engine runs independently
    - All signals collected each bar
    - Priority determines selection
    - Gatekeeper has final veto
    - Risk allocation is per-strategy
    
    PRIORITY (LOCKED):
    1. Trend Continuation: 0.30% risk
    2. Trend Pullback: 0.25% risk
    3. Volatility Expansion: 0.20% risk
    
    Total max risk: 0.75%
    """
    
    # LOCKED RISK ALLOCATIONS
    RISK_ALLOCATIONS = {
        StrategyType.TREND_CONTINUATION: 0.0030,    # 0.30%
        StrategyType.TREND_PULLBACK: 0.0025,        # 0.25%
        StrategyType.VOLATILITY_EXPANSION: 0.0020,  # 0.20%
    }
    
    # PRIORITY ORDER (lower index = higher priority)
    PRIORITY_ORDER = [
        StrategyType.TREND_CONTINUATION,
        StrategyType.TREND_PULLBACK,
        StrategyType.VOLATILITY_EXPANSION,
    ]
    
    def __init__(
        self,
        strategy_config: StrategyConfig = None,
        gatekeeper_config: GatekeeperConfig = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize strategy engines
        self.trend_engine = SignalEngine(strategy_config or StrategyConfig())
        self.pullback_engine = PullbackEngine()
        self.volexp_engine = VolatilityExpansionEngine()
        
        # Initialize gatekeeper
        self.gatekeeper = GatekeeperEngine(
            gatekeeper_config or GatekeeperConfig(),
            self.logger,
        )
        
        # State
        self._bar_count = 0
        self._current_open_strategy: Optional[StrategyType] = None
        
        # Statistics
        self._signals_by_strategy: Dict[StrategyType, int] = {s: 0 for s in StrategyType}
        self._blocks_by_strategy: Dict[StrategyType, int] = {s: 0 for s in StrategyType}
        self._trades_by_strategy: Dict[StrategyType, int] = {s: 0 for s in StrategyType}
    
    @property
    def is_ready(self) -> bool:
        """Check if all engines have enough data."""
        return (
            self.trend_engine.is_ready and
            self.pullback_engine.is_ready and
            self.volexp_engine.is_ready and
            self.gatekeeper.is_ready
        )
    
    @property
    def has_open_position(self) -> bool:
        """Check if any strategy has an open position."""
        return self._current_open_strategy is not None
    
    def update(
        self,
        timestamp: datetime,
        eurusd_open: float,
        eurusd_high: float,
        eurusd_low: float,
        eurusd_close: float,
        gbpusd_close: float,
    ) -> RouterDecision:
        """
        Process new bar through all strategies.
        
        Args:
            timestamp: Bar timestamp
            eurusd_*: EURUSD OHLC
            gbpusd_close: GBPUSD close (for gatekeeper)
        
        Returns:
            RouterDecision with selected signal or None
        """
        self._bar_count += 1
        
        # Update gatekeeper observables
        self.gatekeeper.update(eurusd_close, gbpusd_close)
        
        # Collect signals from all engines
        signals: Dict[StrategyType, AnySignal] = {}
        
        # Strategy 1: Trend Continuation
        trend_signal = self.trend_engine.update(
            timestamp, eurusd_open, eurusd_high, eurusd_low, eurusd_close
        )
        if trend_signal is not None:
            signals[StrategyType.TREND_CONTINUATION] = trend_signal
            self._signals_by_strategy[StrategyType.TREND_CONTINUATION] += 1
        
        # Strategy 2: Trend Pullback
        pullback_signal = self.pullback_engine.update(
            timestamp, eurusd_open, eurusd_high, eurusd_low, eurusd_close
        )
        if pullback_signal is not None:
            signals[StrategyType.TREND_PULLBACK] = pullback_signal
            self._signals_by_strategy[StrategyType.TREND_PULLBACK] += 1
        
        # Strategy 3: Volatility Expansion
        volexp_signal = self.volexp_engine.update(
            timestamp, eurusd_open, eurusd_high, eurusd_low, eurusd_close
        )
        if volexp_signal is not None:
            signals[StrategyType.VOLATILITY_EXPANSION] = volexp_signal
            self._signals_by_strategy[StrategyType.VOLATILITY_EXPANSION] += 1
        
        # If already in a position, no new signals
        if self.has_open_position:
            return RouterDecision(
                bar_index=self._bar_count - 1,
                timestamp=timestamp,
                selected_signal=None,
                selected_strategy=None,
                gatekeeper_decision=None,
                was_blocked=False,
                signals_generated=signals,
                risk_allocation=0.0,
            )
        
        # Select signal by priority
        selected_signal = None
        selected_strategy = None
        
        for strategy_type in self.PRIORITY_ORDER:
            if strategy_type in signals:
                selected_signal = signals[strategy_type]
                selected_strategy = strategy_type
                break
        
        # No signal selected
        if selected_signal is None:
            return RouterDecision(
                bar_index=self._bar_count - 1,
                timestamp=timestamp,
                selected_signal=None,
                selected_strategy=None,
                gatekeeper_decision=None,
                was_blocked=False,
                signals_generated=signals,
                risk_allocation=0.0,
            )
        
        # Apply gatekeeper
        gate_decision = self.gatekeeper.evaluate()
        
        if not gate_decision.allowed:
            self._blocks_by_strategy[selected_strategy] += 1
            self.logger.info(
                f"BLOCKED: {selected_strategy.value} by gatekeeper: "
                f"{[r.value for r in gate_decision.reasons]}"
            )
            
            return RouterDecision(
                bar_index=self._bar_count - 1,
                timestamp=timestamp,
                selected_signal=selected_signal,
                selected_strategy=selected_strategy,
                gatekeeper_decision=gate_decision,
                was_blocked=True,
                signals_generated=signals,
                risk_allocation=0.0,
            )
        
        # Signal allowed
        risk_allocation = self.RISK_ALLOCATIONS[selected_strategy]
        self._trades_by_strategy[selected_strategy] += 1
        
        self.logger.info(
            f"ALLOWED: {selected_strategy.value} @ {self._get_entry_price(selected_signal):.5f}, "
            f"risk: {risk_allocation:.2%}"
        )
        
        return RouterDecision(
            bar_index=self._bar_count - 1,
            timestamp=timestamp,
            selected_signal=selected_signal,
            selected_strategy=selected_strategy,
            gatekeeper_decision=gate_decision,
            was_blocked=False,
            signals_generated=signals,
            risk_allocation=risk_allocation,
        )
    
    def register_position_opened(self, strategy: StrategyType) -> None:
        """Register that a position was opened."""
        self._current_open_strategy = strategy
    
    def register_position_closed(self) -> None:
        """Register that the position was closed."""
        self._current_open_strategy = None
    
    def _get_entry_price(self, signal: AnySignal) -> float:
        """Extract entry price from any signal type."""
        return signal.entry_price
    
    def _get_direction(self, signal: AnySignal) -> str:
        """Extract direction from any signal type."""
        if isinstance(signal, TradeSignal):
            return signal.direction.value
        elif isinstance(signal, PullbackSignal):
            return signal.direction.value
        elif isinstance(signal, VolExpSignal):
            return signal.direction.value
        return "UNKNOWN"
    
    def get_statistics(self) -> Dict:
        """Get router statistics."""
        return {
            "total_bars": self._bar_count,
            "signals_by_strategy": {s.value: c for s, c in self._signals_by_strategy.items()},
            "blocks_by_strategy": {s.value: c for s, c in self._blocks_by_strategy.items()},
            "trades_by_strategy": {s.value: c for s, c in self._trades_by_strategy.items()},
            "gatekeeper_observables": self.gatekeeper.current_observables,
        }
    
    def reset(self) -> None:
        """Reset all engines."""
        self.trend_engine.reset()
        self.pullback_engine.reset()
        self.volexp_engine.reset()
        self.gatekeeper.reset()
        self._bar_count = 0
        self._current_open_strategy = None
        self._signals_by_strategy = {s: 0 for s in StrategyType}
        self._blocks_by_strategy = {s: 0 for s in StrategyType}
        self._trades_by_strategy = {s: 0 for s in StrategyType}


def normalize_signal_direction(signal: AnySignal) -> str:
    """Normalize direction across signal types."""
    if isinstance(signal, TradeSignal):
        return signal.direction.value
    elif isinstance(signal, PullbackSignal):
        return signal.direction.value
    elif isinstance(signal, VolExpSignal):
        return signal.direction.value
    return "UNKNOWN"


def extract_signal_params(signal: AnySignal) -> Tuple[float, float, float]:
    """Extract (entry, sl, tp) from any signal type."""
    return (signal.entry_price, signal.stop_loss, signal.take_profit)
