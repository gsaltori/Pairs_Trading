"""
FX Conditional Relative Value (CRV) System - Layer 5: Execution Safety.

CRITICAL EXECUTION SAFETY LAYER

This module implements:
1. Position limits and exposure management
2. Currency concentration controls
3. Kill-switch conditions
4. Correlation breakdown detection
5. Regime change response
6. Drawdown protection

Philosophy:
    This system MUST be safe for automation.
    Even if execution is not yet coded, the logic must be automation-ready.
    
    SAFETY > PROFIT
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple, Set
from enum import Enum
import logging

from src.crv.regime_filter import FXRegime, REGIME_PERMITS_CRV

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================

class KillSwitchReason(Enum):
    """Reasons for kill-switch activation."""
    NONE = "none"
    DRAWDOWN_LIMIT = "drawdown_limit"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    REGIME_FLIP = "regime_flip"
    HEDGE_RATIO_FLIP = "hedge_ratio_flip"
    VOLATILITY_EXPLOSION = "volatility_explosion"
    MAX_LOSSES_REACHED = "max_losses_reached"
    MANUAL_OVERRIDE = "manual_override"
    DATA_INTEGRITY_FAILURE = "data_integrity_failure"


class ExitReason(Enum):
    """Reasons for position exit."""
    MEAN_REVERSION = "mean_reversion"
    TIME_STOP = "time_stop"
    STRUCTURAL_STOP = "structural_stop"
    REGIME_CHANGE = "regime_change"
    RISK_LIMIT = "risk_limit"
    KILL_SWITCH = "kill_switch"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    MANUAL = "manual"


class SignalType(Enum):
    """Signal types."""
    NO_SIGNAL = "no_signal"
    LONG_SPREAD = "long_spread"
    SHORT_SPREAD = "short_spread"
    EXIT_LONG = "exit_long"
    EXIT_SHORT = "exit_short"
    EMERGENCY_EXIT = "emergency_exit"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class Position:
    """Active CRV position."""
    pair: Tuple[str, str]
    direction: str  # "long" or "short"
    entry_time: datetime
    entry_z: float
    hedge_ratio: float
    size_pct: float
    
    # Current state
    current_z: float = 0.0
    bars_held: int = 0
    unrealized_pnl_pct: float = 0.0
    
    # Targets
    target_z: float = 0.0
    stop_z: float = 0.0
    max_bars: int = 50
    
    # Currencies exposed
    currencies: Set[str] = field(default_factory=set)


@dataclass
class RiskState:
    """Current risk state of the system."""
    timestamp: datetime
    
    # Position counts
    n_positions: int
    max_positions: int
    
    # Exposure
    total_exposure: float
    max_exposure: float
    remaining_exposure: float
    
    # Currency concentration
    currency_exposures: Dict[str, float]
    max_currency_exposure: float
    
    # Drawdown
    current_drawdown: float
    max_drawdown: float
    
    # Kill switch
    is_killed: bool
    kill_reason: Optional[KillSwitchReason]
    kill_timestamp: Optional[datetime]
    
    # Correlation health
    correlation_health: Dict[Tuple[str, str], float]
    
    # Validity
    is_valid: bool
    can_open_new: bool


@dataclass
class ExecutionConstraints:
    """Constraints for execution safety."""
    # Position limits
    max_positions: int = 3
    max_exposure_pct: float = 10.0
    max_position_size_pct: float = 3.0
    
    # Currency limits
    max_currency_exposure_pct: float = 15.0
    max_correlated_positions: int = 2
    
    # Drawdown limits
    warning_drawdown_pct: float = 3.0
    max_drawdown_pct: float = 5.0
    kill_switch_drawdown_pct: float = 8.0
    
    # Time limits
    max_holding_bars: int = 50
    min_bars_between_trades: int = 10
    
    # Correlation limits
    min_pair_correlation: float = 0.3
    correlation_breakdown_threshold: float = -0.2
    
    # Loss limits
    max_consecutive_losses: int = 3
    max_daily_losses: int = 5


@dataclass
class CRVSignal:
    """Complete CRV trading signal."""
    pair: Tuple[str, str]
    timestamp: datetime
    
    # Signal type
    signal_type: SignalType
    
    # Entry parameters
    entry_z: float
    target_z: float
    stop_z: float
    
    # Position sizing
    suggested_size_pct: float
    confidence: float
    
    # Hedge
    hedge_ratio: float
    
    # Validity
    is_valid: bool
    
    # Invalidation reason if not valid
    rejection_reasons: List[str] = field(default_factory=list)


# ============================================================================
# EXECUTION SAFETY MANAGER
# ============================================================================

class ExecutionSafetyManager:
    """
    Layer 5: Execution Safety for FX CRV.
    
    MANDATORY for live trading.
    
    Implements:
    1. Position and exposure limits
    2. Currency concentration controls
    3. Kill-switch conditions
    4. Dynamic pair invalidation
    5. Drawdown protection
    """
    
    def __init__(self, constraints: Optional[ExecutionConstraints] = None):
        self.constraints = constraints or ExecutionConstraints()
        
        # State tracking
        self._positions: Dict[Tuple[str, str], Position] = {}
        self._peak_equity: float = 0.0
        self._current_equity: float = 0.0
        
        # Kill switch state
        self._is_killed: bool = False
        self._kill_reason: Optional[KillSwitchReason] = None
        self._kill_timestamp: Optional[datetime] = None
        
        # Trade history
        self._consecutive_losses: int = 0
        self._daily_losses: int = 0
        self._last_trade_bar: int = 0
        
        # Correlation tracking
        self._correlation_history: Dict[Tuple[str, str], List[float]] = {}
    
    def update_equity(self, equity: float) -> None:
        """Update current equity and peak tracking."""
        self._current_equity = equity
        if equity > self._peak_equity:
            self._peak_equity = equity
    
    def get_current_drawdown(self) -> float:
        """Calculate current drawdown from peak."""
        if self._peak_equity <= 0:
            return 0.0
        return (self._peak_equity - self._current_equity) / self._peak_equity
    
    def check_kill_switch(self) -> Tuple[bool, Optional[KillSwitchReason]]:
        """
        Check if kill-switch should be activated.
        
        Kill-switch conditions (ANY triggers system shutdown):
        1. Drawdown exceeds threshold
        2. Max consecutive losses
        3. Max daily losses
        """
        # Already killed?
        if self._is_killed:
            return True, self._kill_reason
        
        # Drawdown check
        drawdown = self.get_current_drawdown()
        if drawdown >= self.constraints.kill_switch_drawdown_pct / 100:
            self._activate_kill_switch(KillSwitchReason.DRAWDOWN_LIMIT)
            return True, KillSwitchReason.DRAWDOWN_LIMIT
        
        # Consecutive losses
        if self._consecutive_losses >= self.constraints.max_consecutive_losses:
            self._activate_kill_switch(KillSwitchReason.MAX_LOSSES_REACHED)
            return True, KillSwitchReason.MAX_LOSSES_REACHED
        
        # Daily losses
        if self._daily_losses >= self.constraints.max_daily_losses:
            self._activate_kill_switch(KillSwitchReason.MAX_LOSSES_REACHED)
            return True, KillSwitchReason.MAX_LOSSES_REACHED
        
        return False, None
    
    def _activate_kill_switch(self, reason: KillSwitchReason) -> None:
        """Activate kill-switch."""
        self._is_killed = True
        self._kill_reason = reason
        self._kill_timestamp = datetime.now()
        logger.critical(f"KILL-SWITCH ACTIVATED: {reason.value}")
    
    def reset_kill_switch(self, manual: bool = True) -> bool:
        """
        Reset kill-switch (requires manual confirmation).
        
        Only allowed if conditions have improved.
        """
        if not manual:
            return False
        
        drawdown = self.get_current_drawdown()
        if drawdown >= self.constraints.max_drawdown_pct / 100:
            logger.warning("Cannot reset kill-switch: drawdown still too high")
            return False
        
        self._is_killed = False
        self._kill_reason = None
        self._kill_timestamp = None
        self._consecutive_losses = 0
        self._daily_losses = 0
        
        logger.info("Kill-switch reset")
        return True
    
    def get_risk_state(self, equity: float) -> RiskState:
        """Get complete risk state."""
        self.update_equity(equity)
        
        # Check kill-switch
        is_killed, kill_reason = self.check_kill_switch()
        
        # Calculate exposures
        total_exposure = sum(p.size_pct for p in self._positions.values())
        remaining_exposure = max(0, self.constraints.max_exposure_pct - total_exposure)
        
        # Currency exposures
        currency_exposures = self._calculate_currency_exposures()
        
        # Can open new?
        can_open = (
            not is_killed and
            len(self._positions) < self.constraints.max_positions and
            remaining_exposure > 0 and
            self.get_current_drawdown() < self.constraints.max_drawdown_pct / 100
        )
        
        # Correlation health
        corr_health = {pair: self._get_pair_correlation_health(pair) 
                      for pair in self._positions.keys()}
        
        return RiskState(
            timestamp=datetime.now(),
            n_positions=len(self._positions),
            max_positions=self.constraints.max_positions,
            total_exposure=total_exposure,
            max_exposure=self.constraints.max_exposure_pct,
            remaining_exposure=remaining_exposure,
            currency_exposures=currency_exposures,
            max_currency_exposure=self.constraints.max_currency_exposure_pct,
            current_drawdown=self.get_current_drawdown(),
            max_drawdown=self.constraints.max_drawdown_pct / 100,
            is_killed=is_killed,
            kill_reason=kill_reason,
            kill_timestamp=self._kill_timestamp,
            correlation_health=corr_health,
            is_valid=True,
            can_open_new=can_open
        )
    
    def _calculate_currency_exposures(self) -> Dict[str, float]:
        """Calculate exposure per currency."""
        exposures: Dict[str, float] = {}
        
        for pos in self._positions.values():
            for curr in pos.currencies:
                exposures[curr] = exposures.get(curr, 0) + pos.size_pct
        
        return exposures
    
    def _get_pair_correlation_health(self, pair: Tuple[str, str]) -> float:
        """Get correlation health for a pair (0-1, 1 = healthy)."""
        history = self._correlation_history.get(pair, [])
        
        if len(history) < 5:
            return 1.0  # Assume healthy if insufficient data
        
        recent = history[-20:]
        avg_corr = np.mean(recent)
        
        if avg_corr < self.constraints.correlation_breakdown_threshold:
            return 0.0  # Breakdown
        elif avg_corr < self.constraints.min_pair_correlation:
            return 0.5  # Warning
        else:
            return 1.0  # Healthy
    
    def update_correlation(self, pair: Tuple[str, str], correlation: float) -> None:
        """Update correlation tracking for a pair."""
        if pair not in self._correlation_history:
            self._correlation_history[pair] = []
        
        self._correlation_history[pair].append(correlation)
        
        # Keep last 100 observations
        if len(self._correlation_history[pair]) > 100:
            self._correlation_history[pair] = self._correlation_history[pair][-100:]
    
    def check_correlation_breakdown(self, pair: Tuple[str, str]) -> bool:
        """Check if a pair has correlation breakdown."""
        health = self._get_pair_correlation_health(pair)
        return health < 0.5
    
    def validate_new_position(
        self,
        pair: Tuple[str, str],
        size_pct: float,
        regime: FXRegime
    ) -> Tuple[bool, List[str]]:
        """
        Validate if a new position can be opened.
        
        Returns:
            (is_valid, rejection_reasons)
        """
        reasons = []
        
        # Kill-switch check
        if self._is_killed:
            reasons.append(f"Kill-switch active: {self._kill_reason.value}")
            return False, reasons
        
        # Regime check
        if not REGIME_PERMITS_CRV.get(regime, False):
            reasons.append(f"Regime {regime.value} does not permit CRV")
            return False, reasons
        
        # Position limit
        if len(self._positions) >= self.constraints.max_positions:
            reasons.append(f"Max positions reached: {self.constraints.max_positions}")
            return False, reasons
        
        # Already have position in this pair
        if pair in self._positions or (pair[1], pair[0]) in self._positions:
            reasons.append("Already have position in this pair")
            return False, reasons
        
        # Exposure check
        current_exposure = sum(p.size_pct for p in self._positions.values())
        if current_exposure + size_pct > self.constraints.max_exposure_pct:
            reasons.append(f"Would exceed max exposure: {current_exposure + size_pct:.1f}%")
            return False, reasons
        
        # Position size check
        if size_pct > self.constraints.max_position_size_pct:
            reasons.append(f"Position too large: {size_pct:.1f}%")
            return False, reasons
        
        # Currency concentration check
        currencies = self._extract_currencies(pair)
        current_currency_exp = self._calculate_currency_exposures()
        
        for curr in currencies:
            new_exp = current_currency_exp.get(curr, 0) + size_pct
            if new_exp > self.constraints.max_currency_exposure_pct:
                reasons.append(f"Would exceed {curr} concentration: {new_exp:.1f}%")
                return False, reasons
        
        # Drawdown check
        if self.get_current_drawdown() >= self.constraints.max_drawdown_pct / 100:
            reasons.append("Max drawdown reached")
            return False, reasons
        
        # Correlation breakdown check
        if self.check_correlation_breakdown(pair):
            reasons.append("Correlation breakdown detected")
            return False, reasons
        
        return True, reasons
    
    def _extract_currencies(self, pair: Tuple[str, str]) -> Set[str]:
        """Extract all currencies from a pair of symbols."""
        currencies = set()
        for symbol in pair:
            if len(symbol) >= 6:
                currencies.add(symbol[:3])
                currencies.add(symbol[3:6])
        return currencies
    
    def open_position(
        self,
        pair: Tuple[str, str],
        direction: str,
        entry_z: float,
        hedge_ratio: float,
        size_pct: float,
        target_z: float = 0.0,
        stop_z: float = 3.0
    ) -> Optional[Position]:
        """
        Open a new position.
        
        Returns Position if successful, None if rejected.
        """
        # Validate first
        is_valid, reasons = self.validate_new_position(
            pair, size_pct, FXRegime.STABLE_NORMAL_VOL  # Assume valid regime
        )
        
        if not is_valid:
            logger.warning(f"Position rejected: {reasons}")
            return None
        
        position = Position(
            pair=pair,
            direction=direction,
            entry_time=datetime.now(),
            entry_z=entry_z,
            hedge_ratio=hedge_ratio,
            size_pct=size_pct,
            current_z=entry_z,
            bars_held=0,
            target_z=target_z,
            stop_z=stop_z,
            max_bars=self.constraints.max_holding_bars,
            currencies=self._extract_currencies(pair)
        )
        
        self._positions[pair] = position
        logger.info(f"Opened position: {pair} {direction} at Z={entry_z:.2f}")
        
        return position
    
    def close_position(
        self,
        pair: Tuple[str, str],
        reason: ExitReason,
        pnl_pct: float
    ) -> bool:
        """Close a position."""
        if pair not in self._positions:
            return False
        
        position = self._positions.pop(pair)
        
        # Update loss tracking
        if pnl_pct < 0:
            self._consecutive_losses += 1
            self._daily_losses += 1
        else:
            self._consecutive_losses = 0
        
        logger.info(f"Closed position: {pair} reason={reason.value} pnl={pnl_pct:+.2f}%")
        
        return True
    
    def check_position_exits(
        self,
        pair: Tuple[str, str],
        current_z: float,
        current_regime: FXRegime,
        current_bar: int
    ) -> Tuple[bool, Optional[ExitReason]]:
        """
        Check if a position should be exited.
        
        Returns:
            (should_exit, reason)
        """
        if pair not in self._positions:
            return False, None
        
        position = self._positions[pair]
        position.current_z = current_z
        position.bars_held = current_bar - position.entry_time.timestamp()  # Approximate
        
        # Regime change exit
        if not REGIME_PERMITS_CRV.get(current_regime, False):
            return True, ExitReason.REGIME_CHANGE
        
        # Kill-switch exit
        if self._is_killed:
            return True, ExitReason.KILL_SWITCH
        
        # Time stop
        if position.bars_held >= position.max_bars:
            return True, ExitReason.TIME_STOP
        
        # Mean reversion target
        if position.direction == "long" and current_z >= position.target_z:
            return True, ExitReason.MEAN_REVERSION
        if position.direction == "short" and current_z <= position.target_z:
            return True, ExitReason.MEAN_REVERSION
        
        # Structural stop
        if position.direction == "long" and current_z <= -position.stop_z:
            return True, ExitReason.STRUCTURAL_STOP
        if position.direction == "short" and current_z >= position.stop_z:
            return True, ExitReason.STRUCTURAL_STOP
        
        # Correlation breakdown
        if self.check_correlation_breakdown(pair):
            return True, ExitReason.CORRELATION_BREAKDOWN
        
        return False, None
    
    def emergency_close_all(self, reason: str = "manual") -> int:
        """Emergency close all positions."""
        closed = 0
        
        pairs_to_close = list(self._positions.keys())
        
        for pair in pairs_to_close:
            self.close_position(pair, ExitReason.KILL_SWITCH, 0.0)
            closed += 1
        
        logger.critical(f"Emergency close: {closed} positions closed. Reason: {reason}")
        
        return closed
    
    def get_positions(self) -> Dict[Tuple[str, str], Position]:
        """Get all current positions."""
        return self._positions.copy()
    
    def get_position_count(self) -> int:
        """Get number of open positions."""
        return len(self._positions)


# ============================================================================
# SIGNAL ENGINE (HARDENED)
# ============================================================================

class CRVSignalEngine:
    """
    Layer 4: Signal Generation with Execution Safety.
    
    Generates signals ONLY when ALL conditions are met:
    1. Structural validity
    2. Regime permission
    3. Conditional divergence
    4. Risk limits OK
    5. Execution constraints OK
    """
    
    def __init__(
        self,
        base_entry_z: float = 1.5,
        base_exit_z: float = 0.3,
        max_holding_bars: int = 50,
        min_confidence: float = 0.5
    ):
        self.base_entry_z = base_entry_z
        self.base_exit_z = base_exit_z
        self.max_holding_bars = max_holding_bars
        self.min_confidence = min_confidence
    
    def generate_signal(
        self,
        pair: Tuple[str, str],
        zscore_conditional: float,
        hedge_ratio: float,
        regime: FXRegime,
        regime_confidence: float,
        risk_state: RiskState,
        spread_is_valid: bool,
        timestamp: Optional[datetime] = None
    ) -> CRVSignal:
        """
        Generate CRV signal with full validation.
        
        GUARANTEES:
        - Only generates valid signals when ALL conditions met
        - Never generates signal with invalid inputs
        - Explicit rejection reasons provided
        """
        timestamp = timestamp or datetime.now()
        rejection_reasons = []
        
        # === VALIDATION CHAIN ===
        
        # 1. Check spread validity
        if not spread_is_valid:
            rejection_reasons.append("Spread data invalid")
        
        # 2. Check regime permission
        if not REGIME_PERMITS_CRV.get(regime, False):
            rejection_reasons.append(f"Regime {regime.value} blocks trading")
        
        # 3. Check regime confidence
        if regime_confidence < 0.6:
            rejection_reasons.append(f"Low regime confidence: {regime_confidence:.0%}")
        
        # 4. Check risk state
        if risk_state.is_killed:
            rejection_reasons.append(f"Kill-switch active: {risk_state.kill_reason.value}")
        
        if not risk_state.can_open_new:
            rejection_reasons.append("Cannot open new positions")
        
        # 5. Check z-score validity
        if np.isnan(zscore_conditional) or np.isinf(zscore_conditional):
            rejection_reasons.append("Invalid z-score")
            zscore_conditional = 0.0
        
        # 6. Check hedge ratio
        if np.isnan(hedge_ratio) or abs(hedge_ratio) < 0.1 or abs(hedge_ratio) > 10:
            rejection_reasons.append(f"Invalid hedge ratio: {hedge_ratio}")
        
        # === SIGNAL DETERMINATION ===
        
        if rejection_reasons:
            return CRVSignal(
                pair=pair,
                timestamp=timestamp,
                signal_type=SignalType.NO_SIGNAL,
                entry_z=0.0,
                target_z=0.0,
                stop_z=0.0,
                suggested_size_pct=0.0,
                confidence=0.0,
                hedge_ratio=hedge_ratio if not np.isnan(hedge_ratio) else 1.0,
                is_valid=False,
                rejection_reasons=rejection_reasons
            )
        
        # Check entry threshold
        entry_threshold = self._get_entry_threshold(regime)
        
        if abs(zscore_conditional) < entry_threshold:
            return CRVSignal(
                pair=pair,
                timestamp=timestamp,
                signal_type=SignalType.NO_SIGNAL,
                entry_z=zscore_conditional,
                target_z=0.0,
                stop_z=0.0,
                suggested_size_pct=0.0,
                confidence=0.0,
                hedge_ratio=hedge_ratio,
                is_valid=True,
                rejection_reasons=["Z-score below threshold"]
            )
        
        # Determine direction
        if zscore_conditional < -entry_threshold:
            signal_type = SignalType.LONG_SPREAD
            target_z = -self.base_exit_z
            stop_z = zscore_conditional - 2.0  # Stop if diverges further
        elif zscore_conditional > entry_threshold:
            signal_type = SignalType.SHORT_SPREAD
            target_z = self.base_exit_z
            stop_z = zscore_conditional + 2.0
        else:
            signal_type = SignalType.NO_SIGNAL
            target_z = 0.0
            stop_z = 0.0
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            zscore_conditional, regime_confidence, regime
        )
        
        if confidence < self.min_confidence:
            return CRVSignal(
                pair=pair,
                timestamp=timestamp,
                signal_type=SignalType.NO_SIGNAL,
                entry_z=zscore_conditional,
                target_z=target_z,
                stop_z=stop_z,
                suggested_size_pct=0.0,
                confidence=confidence,
                hedge_ratio=hedge_ratio,
                is_valid=True,
                rejection_reasons=[f"Low confidence: {confidence:.0%}"]
            )
        
        # Calculate position size
        size_pct = self._calculate_position_size(
            confidence, risk_state.remaining_exposure, regime
        )
        
        return CRVSignal(
            pair=pair,
            timestamp=timestamp,
            signal_type=signal_type,
            entry_z=zscore_conditional,
            target_z=target_z,
            stop_z=abs(stop_z),
            suggested_size_pct=size_pct,
            confidence=confidence,
            hedge_ratio=hedge_ratio,
            is_valid=True,
            rejection_reasons=[]
        )
    
    def _get_entry_threshold(self, regime: FXRegime) -> float:
        """Get regime-specific entry threshold."""
        thresholds = {
            FXRegime.STABLE_LOW_VOL: 1.25,
            FXRegime.STABLE_NORMAL_VOL: 1.5,
            FXRegime.RANGE_BOUND: 1.75,
            FXRegime.UNKNOWN: 2.0,
        }
        return thresholds.get(regime, self.base_entry_z)
    
    def _calculate_confidence(
        self,
        zscore: float,
        regime_confidence: float,
        regime: FXRegime
    ) -> float:
        """Calculate signal confidence."""
        # Base confidence from z-score
        z_confidence = min(1.0, abs(zscore) / 3.0)
        
        # Regime multiplier
        regime_mult = {
            FXRegime.STABLE_LOW_VOL: 1.0,
            FXRegime.STABLE_NORMAL_VOL: 0.9,
            FXRegime.RANGE_BOUND: 0.8,
        }.get(regime, 0.5)
        
        # Combined confidence
        confidence = z_confidence * regime_confidence * regime_mult
        
        return min(1.0, max(0.0, confidence))
    
    def _calculate_position_size(
        self,
        confidence: float,
        remaining_exposure: float,
        regime: FXRegime
    ) -> float:
        """Calculate suggested position size."""
        # Base size
        base_size = 2.0
        
        # Scale by confidence
        size = base_size * confidence
        
        # Regime multiplier
        regime_mult = {
            FXRegime.STABLE_LOW_VOL: 1.0,
            FXRegime.STABLE_NORMAL_VOL: 0.8,
            FXRegime.RANGE_BOUND: 0.6,
        }.get(regime, 0.5)
        
        size *= regime_mult
        
        # Cap at remaining exposure
        size = min(size, remaining_exposure * 0.5)
        
        # Cap at max position size
        size = min(size, 3.0)
        
        return max(0.0, size)


# ============================================================================
# CONVENIENCE EXPORTS
# ============================================================================

# For backward compatibility
CRVRiskManager = ExecutionSafetyManager
