"""
Conditional Pair Manager - Part 2.

Integrates regime detection, cointegration validation, and spread health
into a unified decision system.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Tuple
import logging

from src.strategy.conditional_statarb import (
    PairState, MarketRegime, TradingSession,
    VolatilityMetrics, TrendMetrics, SpreadHealth, CointegrationStatus,
    RegimeAnalysis, PairStatus,
    MarketRegimeDetector, DynamicCointegrationValidator, SpreadHealthMonitor,
    REGIME_TRADEABLE
)

logger = logging.getLogger(__name__)


# ============================================================================
# CONDITIONAL SIGNAL GENERATOR
# ============================================================================

@dataclass
class ConditionalSignal:
    """Signal with all validation context."""
    pair: Tuple[str, str]
    timestamp: datetime
    
    # Signal
    direction: Optional[str]  # "long", "short", None
    zscore: float
    strength: float  # 0-1
    
    # Context validation
    is_valid: bool
    
    # Components
    spread_stationary: bool
    half_life_valid: bool
    volatility_stable: bool
    regime_favorable: bool
    no_structural_break: bool
    
    # Risk parameters
    suggested_entry_z: float
    suggested_exit_z: float
    suggested_stop_z: float
    max_holding_bars: int
    
    # Invalidation conditions
    invalidation_reasons: List[str] = field(default_factory=list)


class ConditionalSignalGenerator:
    """
    Generates trading signals ONLY when all conditions are met.
    
    Z-Score is NOT used raw. Signal validity requires:
    1. Spread is stationary in current window
    2. Half-life is in optimal range
    3. Spread volatility is stable
    4. No structural break detected
    5. Regime is favorable
    """
    
    def __init__(
        self,
        entry_zscore: float = 2.0,
        exit_zscore: float = 0.3,
        stop_zscore: float = 3.5,
        min_signal_strength: float = 0.5
    ):
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.stop_zscore = stop_zscore
        self.min_signal_strength = min_signal_strength
    
    def generate(
        self,
        pair: Tuple[str, str],
        zscore: float,
        spread_health: SpreadHealth,
        cointegration: CointegrationStatus,
        regime: MarketRegime,
        timestamp: datetime
    ) -> ConditionalSignal:
        """
        Generate a conditional signal.
        
        The signal is only valid if ALL conditions pass.
        """
        invalidation_reasons = []
        
        # Check all conditions
        spread_stationary = spread_health.is_stationary
        if not spread_stationary:
            invalidation_reasons.append(f"Spread not stationary (ADF p={spread_health.adf_pvalue:.3f})")
        
        half_life_valid = spread_health.half_life_stable
        if not half_life_valid:
            invalidation_reasons.append(f"Half-life out of range: {spread_health.half_life:.0f}")
        
        volatility_stable = spread_health.spread_vol_stable
        if not volatility_stable:
            invalidation_reasons.append("Spread volatility unstable")
        
        no_structural_break = (
            cointegration.is_stable and 
            spread_health.hedge_ratio_stable
        )
        if not no_structural_break:
            if not cointegration.is_stable:
                invalidation_reasons.append(f"Cointegration unstable ({cointegration.confidence})")
            if not spread_health.hedge_ratio_stable:
                invalidation_reasons.append(f"Hedge ratio drifting (Z={spread_health.hedge_ratio_zscore:.1f})")
        
        regime_favorable = REGIME_TRADEABLE.get(regime, False)
        if not regime_favorable:
            invalidation_reasons.append(f"Regime unfavorable: {regime.value}")
        
        # Check if all conditions pass
        is_valid = (
            spread_stationary and
            half_life_valid and
            volatility_stable and
            no_structural_break and
            regime_favorable
        )
        
        # Determine direction (only if potentially valid)
        direction = None
        strength = 0.0
        
        if abs(zscore) >= self.entry_zscore:
            if zscore < -self.entry_zscore:
                direction = "long"
                strength = min(1.0, (abs(zscore) - self.entry_zscore) / 1.5 + 0.5)
            elif zscore > self.entry_zscore:
                direction = "short"
                strength = min(1.0, (abs(zscore) - self.entry_zscore) / 1.5 + 0.5)
        
        # If conditions don't pass, signal is not valid
        if not is_valid:
            direction = None
            strength = 0.0
        
        # Calculate adaptive parameters based on half-life
        half_life = spread_health.half_life
        if half_life < 20:
            suggested_entry = 2.0
            suggested_exit = 0.2
            suggested_stop = 3.0
            max_holding = int(half_life * 3)
        elif half_life < 40:
            suggested_entry = 2.0
            suggested_exit = 0.3
            suggested_stop = 3.5
            max_holding = int(half_life * 2.5)
        else:
            suggested_entry = 2.2
            suggested_exit = 0.5
            suggested_stop = 4.0
            max_holding = int(half_life * 2)
        
        return ConditionalSignal(
            pair=pair,
            timestamp=timestamp,
            direction=direction,
            zscore=zscore,
            strength=strength,
            is_valid=is_valid,
            spread_stationary=spread_stationary,
            half_life_valid=half_life_valid,
            volatility_stable=volatility_stable,
            regime_favorable=regime_favorable,
            no_structural_break=no_structural_break,
            suggested_entry_z=suggested_entry,
            suggested_exit_z=suggested_exit,
            suggested_stop_z=suggested_stop,
            max_holding_bars=max_holding,
            invalidation_reasons=invalidation_reasons
        )


# ============================================================================
# CONDITIONAL PAIR MANAGER
# ============================================================================

class ConditionalPairManager:
    """
    Manages pair states and determines tradability.
    
    Core philosophy:
    - A pair can be statistically valid but NOT tradeable
    - DORMANT is a valid state (regime unfavorable)
    - The system knows when NOT to trade
    - Inactivity is a feature, not a bug
    
    States:
    - ACTIVE: Valid and tradeable now
    - DORMANT: Valid but regime unfavorable
    - INVALIDATED: Failed statistical tests
    - WARMING_UP: Insufficient data
    """
    
    def __init__(
        self,
        regime_detector: MarketRegimeDetector,
        coint_validator: DynamicCointegrationValidator,
        spread_monitor: SpreadHealthMonitor,
        signal_generator: ConditionalSignalGenerator
    ):
        self.regime_detector = regime_detector
        self.coint_validator = coint_validator
        self.spread_monitor = spread_monitor
        self.signal_generator = signal_generator
        
        # State tracking
        self.pair_states: Dict[Tuple[str, str], PairStatus] = {}
        self.state_history: Dict[Tuple[str, str], List[Tuple[datetime, PairState]]] = {}
    
    def update_pair_status(
        self,
        pair: Tuple[str, str],
        price_a: pd.Series,
        price_b: pd.Series,
        ohlc_a: Optional[pd.DataFrame] = None,
        timestamp: Optional[datetime] = None
    ) -> PairStatus:
        """
        Update the complete status of a pair.
        
        This is the main entry point for pair analysis.
        """
        timestamp = timestamp or datetime.now()
        state_reasons = []
        
        # Minimum data check
        if len(price_a) < 500:
            return PairStatus(
                pair=pair,
                state=PairState.WARMING_UP,
                is_statistically_valid=False,
                regime_analysis=None,
                is_regime_favorable=False,
                is_tradeable_now=False,
                current_zscore=0.0,
                signal=None,
                signal_strength=0.0,
                state_reasons=["Insufficient data for analysis"]
            )
        
        # 1. Validate cointegration
        coint_status = self.coint_validator.validate(price_a, price_b)
        
        if not coint_status.is_stable:
            state_reasons.append(f"Cointegration unstable (confidence: {coint_status.confidence})")
        
        # 2. Analyze spread health
        spread_health = self.spread_monitor.analyze(price_a, price_b)
        
        if not spread_health.is_healthy:
            if not spread_health.is_stationary:
                state_reasons.append("Spread not stationary")
            if not spread_health.half_life_stable:
                state_reasons.append(f"Half-life: {spread_health.half_life:.0f} bars (out of range)")
            if not spread_health.is_mean_reverting:
                state_reasons.append(f"Hurst: {spread_health.hurst_exponent:.3f} (not mean-reverting)")
            if not spread_health.hedge_ratio_stable:
                state_reasons.append("Hedge ratio drifting")
        
        # 3. Detect market regime (requires OHLC)
        if ohlc_a is not None and len(ohlc_a) > 250:
            regime, regime_details = self.regime_detector.detect_regime(ohlc_a)
            session = self.regime_detector.get_trading_session(timestamp)
            
            is_regime_favorable = REGIME_TRADEABLE.get(regime, False)
            
            if not is_regime_favorable:
                state_reasons.append(f"Regime: {regime.value}")
            
            # Build regime analysis
            regime_analysis = RegimeAnalysis(
                timestamp=timestamp,
                volatility=regime_details.get("volatility"),
                trend=regime_details.get("trend"),
                session=session,
                spread_health=spread_health,
                cointegration=coint_status,
                market_regime=regime,
                is_tradeable=is_regime_favorable,
                blocking_reasons=[] if is_regime_favorable else [f"Regime: {regime.value}"]
            )
        else:
            regime = MarketRegime.UNKNOWN
            is_regime_favorable = False
            regime_analysis = None
            state_reasons.append("No OHLC data for regime detection")
        
        # 4. Calculate current Z-score
        spread = price_a - spread_health.hedge_ratio * price_b
        zscore_window = 60
        spread_mean = spread.rolling(zscore_window).mean()
        spread_std = spread.rolling(zscore_window).std()
        zscore = (spread - spread_mean) / spread_std
        current_zscore = float(zscore.iloc[-1]) if not pd.isna(zscore.iloc[-1]) else 0.0
        
        # 5. Determine state
        is_statistically_valid = coint_status.is_stable and spread_health.is_healthy
        
        if not is_statistically_valid:
            state = PairState.INVALIDATED
            is_tradeable = False
        elif not is_regime_favorable:
            state = PairState.DORMANT
            is_tradeable = False
        else:
            state = PairState.ACTIVE
            is_tradeable = True
        
        # 6. Generate signal (only if active)
        signal = None
        signal_strength = 0.0
        
        if state == PairState.ACTIVE:
            conditional_signal = self.signal_generator.generate(
                pair=pair,
                zscore=current_zscore,
                spread_health=spread_health,
                cointegration=coint_status,
                regime=regime,
                timestamp=timestamp
            )
            
            if conditional_signal.is_valid and conditional_signal.direction:
                signal = conditional_signal.direction
                signal_strength = conditional_signal.strength
        
        # 7. Build status
        status = PairStatus(
            pair=pair,
            state=state,
            is_statistically_valid=is_statistically_valid,
            regime_analysis=regime_analysis,
            is_regime_favorable=is_regime_favorable,
            is_tradeable_now=is_tradeable,
            current_zscore=current_zscore,
            signal=signal,
            signal_strength=signal_strength,
            state_reasons=state_reasons
        )
        
        # Update state tracking
        self._update_state_tracking(pair, state, timestamp, status)
        
        return status
    
    def _update_state_tracking(
        self,
        pair: Tuple[str, str],
        new_state: PairState,
        timestamp: datetime,
        status: PairStatus
    ):
        """Track state changes."""
        # Update current state
        old_status = self.pair_states.get(pair)
        
        if old_status:
            # Preserve timestamps
            if new_state == PairState.ACTIVE and old_status.state != PairState.ACTIVE:
                status.last_active = timestamp
            elif old_status.last_active:
                status.last_active = old_status.last_active
            
            if new_state == PairState.DORMANT and old_status.state != PairState.DORMANT:
                status.dormant_since = timestamp
            elif old_status.dormant_since and new_state == PairState.DORMANT:
                status.dormant_since = old_status.dormant_since
            
            if new_state == PairState.INVALIDATED and old_status.state != PairState.INVALIDATED:
                status.invalidated_since = timestamp
            elif old_status.invalidated_since and new_state == PairState.INVALIDATED:
                status.invalidated_since = old_status.invalidated_since
        
        self.pair_states[pair] = status
        
        # Update history
        if pair not in self.state_history:
            self.state_history[pair] = []
        
        self.state_history[pair].append((timestamp, new_state))
        
        # Keep only last 100 state changes
        if len(self.state_history[pair]) > 100:
            self.state_history[pair] = self.state_history[pair][-100:]
    
    def get_active_pairs(self) -> List[PairStatus]:
        """Get all pairs in ACTIVE state."""
        return [s for s in self.pair_states.values() if s.state == PairState.ACTIVE]
    
    def get_dormant_pairs(self) -> List[PairStatus]:
        """Get all pairs in DORMANT state."""
        return [s for s in self.pair_states.values() if s.state == PairState.DORMANT]
    
    def get_tradeable_signals(self) -> List[PairStatus]:
        """Get pairs with valid trading signals."""
        return [
            s for s in self.pair_states.values() 
            if s.state == PairState.ACTIVE and s.signal is not None
        ]
    
    def get_state_summary(self) -> Dict[str, int]:
        """Get count of pairs in each state."""
        summary = {state.value: 0 for state in PairState}
        
        for status in self.pair_states.values():
            summary[status.state.value] += 1
        
        return summary
    
    def should_trade(self, pair: Tuple[str, str]) -> Tuple[bool, List[str]]:
        """
        Determine if a pair should be traded NOW.
        
        Returns:
            (should_trade, reasons_if_not)
        """
        status = self.pair_states.get(pair)
        
        if not status:
            return False, ["Pair not analyzed"]
        
        if status.state == PairState.WARMING_UP:
            return False, ["Insufficient data"]
        
        if status.state == PairState.INVALIDATED:
            return False, status.state_reasons
        
        if status.state == PairState.DORMANT:
            return False, status.state_reasons + ["Regime unfavorable - wait for change"]
        
        if status.state == PairState.ACTIVE:
            if status.signal:
                return True, []
            else:
                return False, ["No signal - Z-score not at entry level"]
        
        return False, ["Unknown state"]


# ============================================================================
# CONDITIONAL STATARB SYSTEM
# ============================================================================

class ConditionalStatArbSystem:
    """
    Main system integrating all components.
    
    This is the top-level class that:
    1. Manages multiple pairs
    2. Coordinates regime detection
    3. Generates trading decisions
    4. Tracks system state
    """
    
    def __init__(
        self,
        # Regime detection
        atr_period: int = 14,
        adx_period: int = 14,
        trend_threshold: float = 25.0,
        
        # Cointegration
        coint_windows: List[int] = [250, 500, 750],
        coint_pvalue: float = 0.05,
        
        # Spread health
        half_life_max: int = 60,
        half_life_min: int = 5,
        hurst_max: float = 0.55,
        
        # Signals
        entry_zscore: float = 2.0,
        exit_zscore: float = 0.3,
        stop_zscore: float = 3.5
    ):
        # Initialize components
        self.regime_detector = MarketRegimeDetector(
            atr_period=atr_period,
            adx_period=adx_period,
            trend_threshold=trend_threshold
        )
        
        self.coint_validator = DynamicCointegrationValidator(
            windows=coint_windows,
            pvalue_threshold=coint_pvalue
        )
        
        self.spread_monitor = SpreadHealthMonitor(
            half_life_max=half_life_max,
            half_life_min=half_life_min,
            hurst_max=hurst_max
        )
        
        self.signal_generator = ConditionalSignalGenerator(
            entry_zscore=entry_zscore,
            exit_zscore=exit_zscore,
            stop_zscore=stop_zscore
        )
        
        self.pair_manager = ConditionalPairManager(
            regime_detector=self.regime_detector,
            coint_validator=self.coint_validator,
            spread_monitor=self.spread_monitor,
            signal_generator=self.signal_generator
        )
        
        # System state
        self.is_active = False
        self.last_update = None
        self.global_regime = MarketRegime.UNKNOWN
    
    def update(
        self,
        pairs_data: Dict[Tuple[str, str], Tuple[pd.Series, pd.Series, Optional[pd.DataFrame]]],
        timestamp: Optional[datetime] = None
    ) -> Dict[str, any]:
        """
        Update all pairs and return system status.
        
        Args:
            pairs_data: {pair: (price_a, price_b, ohlc_a)}
            timestamp: Current timestamp
            
        Returns:
            System status dict
        """
        timestamp = timestamp or datetime.now()
        self.last_update = timestamp
        
        # Update each pair
        for pair, (price_a, price_b, ohlc_a) in pairs_data.items():
            self.pair_manager.update_pair_status(
                pair=pair,
                price_a=price_a,
                price_b=price_b,
                ohlc_a=ohlc_a,
                timestamp=timestamp
            )
        
        # Get summary
        state_summary = self.pair_manager.get_state_summary()
        active_pairs = self.pair_manager.get_active_pairs()
        signals = self.pair_manager.get_tradeable_signals()
        
        # Determine if system is active
        self.is_active = len(active_pairs) > 0
        
        return {
            "timestamp": timestamp,
            "is_active": self.is_active,
            "state_summary": state_summary,
            "active_pairs": len(active_pairs),
            "dormant_pairs": state_summary.get("dormant", 0),
            "invalidated_pairs": state_summary.get("invalidated", 0),
            "signals": [
                {
                    "pair": s.pair,
                    "direction": s.signal,
                    "zscore": s.current_zscore,
                    "strength": s.signal_strength
                }
                for s in signals
            ]
        }
    
    def get_trading_decisions(self) -> List[Dict]:
        """Get all current trading decisions."""
        signals = self.pair_manager.get_tradeable_signals()
        
        decisions = []
        for status in signals:
            decisions.append({
                "pair": status.pair,
                "action": status.signal,
                "zscore": status.current_zscore,
                "strength": status.signal_strength,
                "regime": status.regime_analysis.market_regime.value if status.regime_analysis else "unknown"
            })
        
        return decisions
    
    def should_system_trade(self) -> Tuple[bool, List[str]]:
        """
        Determine if the system should trade at all.
        
        Returns:
            (should_trade, reasons_if_not)
        """
        if not self.is_active:
            return False, ["No active pairs"]
        
        signals = self.pair_manager.get_tradeable_signals()
        
        if not signals:
            return False, ["No valid signals"]
        
        return True, []
    
    def generate_status_report(self) -> str:
        """Generate a human-readable status report."""
        lines = []
        lines.append("=" * 70)
        lines.append("CONDITIONAL STATARB SYSTEM STATUS")
        lines.append("=" * 70)
        
        lines.append(f"\nLast Update: {self.last_update}")
        lines.append(f"System Active: {'YES' if self.is_active else 'NO'}")
        
        # State summary
        summary = self.pair_manager.get_state_summary()
        lines.append(f"\nPair States:")
        lines.append(f"  ACTIVE:      {summary.get('active', 0)}")
        lines.append(f"  DORMANT:     {summary.get('dormant', 0)}")
        lines.append(f"  INVALIDATED: {summary.get('invalidated', 0)}")
        lines.append(f"  WARMING UP:  {summary.get('warming_up', 0)}")
        
        # Active pairs
        active = self.pair_manager.get_active_pairs()
        if active:
            lines.append(f"\n" + "-" * 70)
            lines.append("ACTIVE PAIRS")
            lines.append("-" * 70)
            
            for status in active:
                signal = f" → {status.signal.upper()} (Z={status.current_zscore:+.2f})" if status.signal else ""
                lines.append(f"\n  {status.pair[0]}/{status.pair[1]}{signal}")
                
                if status.regime_analysis:
                    ra = status.regime_analysis
                    lines.append(f"    Regime: {ra.market_regime.value}")
                    lines.append(f"    Session: {ra.session.value}")
                    lines.append(f"    Spread Health: {ra.spread_health.health_score:.0f}/100")
        
        # Dormant pairs
        dormant = self.pair_manager.get_dormant_pairs()
        if dormant:
            lines.append(f"\n" + "-" * 70)
            lines.append("DORMANT PAIRS (waiting for favorable regime)")
            lines.append("-" * 70)
            
            for status in dormant:
                lines.append(f"\n  {status.pair[0]}/{status.pair[1]}")
                lines.append(f"    Reasons: {', '.join(status.state_reasons[:2])}")
                if status.dormant_since:
                    lines.append(f"    Dormant since: {status.dormant_since}")
        
        # Signals
        signals = self.pair_manager.get_tradeable_signals()
        if signals:
            lines.append(f"\n" + "-" * 70)
            lines.append("🚨 TRADING SIGNALS")
            lines.append("-" * 70)
            
            for status in signals:
                direction = "LONG" if status.signal == "long" else "SHORT"
                lines.append(f"\n  {status.pair[0]}/{status.pair[1]}: {direction} SPREAD")
                lines.append(f"    Z-Score: {status.current_zscore:+.2f}")
                lines.append(f"    Strength: {status.signal_strength:.0%}")
        else:
            lines.append(f"\n  No trading signals - system waiting")
        
        lines.append("\n" + "=" * 70)
        
        return "\n".join(lines)
