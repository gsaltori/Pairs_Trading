"""
FX Conditional Relative Value (CRV) System - Integrated System (HARDENED).

INSTITUTIONAL GRADE - VERSION 2.0

This is the main system that integrates all CRV layers:
- Layer 0: Data Integrity (MANDATORY)
- Layer 1: Structural Pair Selection
- Layer 2: Regime Filter
- Layer 3: Conditional Spread Analysis
- Layer 4: Signal Generation
- Layer 5: Execution Safety

CRITICAL FIXES IN THIS VERSION:
1. Explicit NaN handling throughout
2. pct_change(fill_method=None) applied
3. Kill-switch integration
4. Dynamic pair invalidation
5. Execution-safe constraints

Key Philosophy:
    This is NOT Statistical Arbitrage. FX does not exhibit permanent
    mean reversion. We trade CONDITIONAL relative value - exploiting
    temporary divergences only when market regime permits.

    The system is designed to NOT TRADE most of the time.
    Inactivity is the correct behavior when edge is absent.
    
    SAFETY > PROFIT
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
import logging
import json
from pathlib import Path

# Layer 0: Data Integrity
from src.crv.data_integrity import (
    FXDataValidator, AlignedDataset, safe_returns
)

# Layer 1: Pair Selection
from src.crv.pair_selector import (
    FXStructuralPairSelector, StructuralPairAssessment,
    PairRelationship, has_macro_coherence, get_macro_coherence
)

# Layer 2: Regime Filter
from src.crv.regime_filter import (
    FXRegimeFilter, FXRegimeAssessment,
    FXRegime, REGIME_PERMITS_CRV
)

# Layer 3: Conditional Spread
from src.crv.conditional_spread import (
    ConditionalSpreadAnalyzer, ConditionalSpreadData,
    RegimeStatistics
)

# Layer 4-5: Signal + Execution Safety
from src.crv.execution_safety import (
    ExecutionSafetyManager, CRVSignalEngine,
    CRVSignal, SignalType, ExitReason, RiskState,
    ExecutionConstraints, KillSwitchReason
)

# Import FSM (optional - for FSM-aware operation)
try:
    from src.crv.state_machine import CRVStateMachine, SystemMode
    FSM_AVAILABLE = True
except ImportError:
    FSM_AVAILABLE = False
    CRVStateMachine = None
    SystemMode = None

logger = logging.getLogger(__name__)


# ============================================================================
# CRV SYSTEM STATE
# ============================================================================

@dataclass
class CRVSystemState:
    """Complete state of the CRV system."""
    timestamp: datetime
    
    # Layer 0: Data
    data_valid: bool
    n_symbols_loaded: int
    n_symbols_rejected: int
    
    # Layer 1: Structural pairs
    n_structural_pairs: int
    structural_pairs: List[Tuple[str, str]]
    tier_distribution: Dict[str, int]
    
    # Layer 2: Regime
    current_regime: FXRegime
    regime_permits_trading: bool
    regime_confidence: float
    
    # Layer 3-4: Signals
    n_signals: int
    active_signals: List[CRVSignal]
    
    # Layer 5: Risk
    risk_state: RiskState
    n_positions: int
    
    # System status
    is_active: bool
    is_trading: bool
    inactivity_reason: Optional[str] = None
    
    # Health
    system_health: str = "unknown"  # "healthy", "warning", "critical"


@dataclass
class CRVAnalysisResult:
    """Result of CRV analysis for a single pair."""
    pair: Tuple[str, str]
    timestamp: datetime
    
    # Structural assessment
    structural: StructuralPairAssessment
    
    # Regime context
    regime: FXRegime
    regime_permits: bool
    
    # Spread analysis
    spread_data: Optional[ConditionalSpreadData]
    
    # Signal
    signal: Optional[CRVSignal]
    
    # Overall status
    is_tradeable: bool
    is_valid: bool  # NEW: Explicit validity
    status_notes: List[str] = field(default_factory=list)


# ============================================================================
# MAIN CRV SYSTEM (HARDENED)
# ============================================================================

class FXConditionalRelativeValueSystem:
    """
    FX Conditional Relative Value Trading System (HARDENED).
    
    INSTITUTIONAL GRADE - VERSION 2.0
    
    This system explicitly REJECTS classical StatArb assumptions:
    1. NO permanent cointegration assumption
    2. NO universal mean reversion
    3. NO static hedge ratios
    
    Instead, we:
    1. Require STRUCTURAL stability (not cointegration)
    2. Trade ONLY in favorable regimes
    3. Use CONDITIONAL z-scores (regime-dependent)
    4. Accept long periods of inactivity
    5. NEVER propagate NaN values
    6. Have kill-switch for safety
    
    The system is designed to:
    - Trade rarely but with edge
    - Know when NOT to trade
    - Survive regime changes
    - Avoid false signals
    - BE SAFE FOR AUTOMATION
    """
    
    def __init__(
        self,
        # Layer 1: Pair selection (FX-native parameters)
        min_macro_score: float = 0.50,
        min_median_correlation: float = 0.20,
        
        # Layer 2: Regime filter
        adx_threshold: float = 25.0,
        vol_high_percentile: float = 75.0,
        
        # Layer 3: Spread analysis
        conditional_entry_z: float = 1.5,
        conditional_exit_z: float = 0.3,
        
        # Layer 4-5: Signal & Risk
        max_positions: int = 3,
        max_exposure_pct: float = 10.0,
        max_holding_bars: int = 50,
        
        # Execution constraints
        constraints: Optional[ExecutionConstraints] = None,
    ):
        # Layer 0: Data Validator
        # Note: FX has weekend gaps of ~72-76 hours (Fri close to Mon open)
        # So max_gap_hours must be > 76 to allow weekend gaps
        self.data_validator = FXDataValidator(
            max_nan_percentage=2.0,
            max_gap_hours=80.0,  # Allow weekend gaps
            min_bars_required=500
        )
        
        # Layer 1: Pair Selector
        self.pair_selector = FXStructuralPairSelector(
            min_macro_score=min_macro_score,
            min_median_correlation=min_median_correlation,
        )
        
        # Layer 2: Regime Filter
        self.regime_filter = FXRegimeFilter(
            adx_strong_threshold=adx_threshold,
            vol_high_percentile=vol_high_percentile
        )
        
        # Layer 3: Spread Analyzer
        self.spread_analyzer = ConditionalSpreadAnalyzer(
            zscore_entry_threshold=conditional_entry_z,
        )
        
        # Layer 4: Signal Engine
        self.signal_engine = CRVSignalEngine(
            base_entry_z=conditional_entry_z,
            base_exit_z=conditional_exit_z,
            max_holding_bars=max_holding_bars
        )
        
        # Layer 5: Execution Safety
        self.execution_manager = ExecutionSafetyManager(
            constraints=constraints or ExecutionConstraints(
                max_positions=max_positions,
                max_exposure_pct=max_exposure_pct,
                max_holding_bars=max_holding_bars
            )
        )
        
        # State
        self._aligned_data: Optional[AlignedDataset] = None
        self._structural_pairs: List[StructuralPairAssessment] = []
        self._current_regime: FXRegime = FXRegime.UNKNOWN
        self._regime_assessment: Optional[FXRegimeAssessment] = None
        self._last_update: Optional[datetime] = None
        
        logger.info("FX CRV System initialized (HARDENED v2.0)")
    
    # ========================================================================
    # LAYER 0: DATA INTEGRITY
    # ========================================================================
    
    def validate_and_align_data(
        self,
        price_data: Dict[str, pd.Series],
        ohlc_data: Optional[Dict[str, pd.DataFrame]] = None,
        timeframe: str = "H4"
    ) -> AlignedDataset:
        """
        Layer 0: Validate and align all data.
        
        MANDATORY before any analysis.
        
        Returns:
            AlignedDataset with only valid, aligned data
        """
        logger.info("Layer 0: Validating data integrity...")
        
        self._aligned_data = self.data_validator.align_and_validate(
            price_data=price_data,
            ohlc_data=ohlc_data,
            timeframe=timeframe
        )
        
        logger.info(f"Data validation complete: {len(self._aligned_data.symbols)} valid symbols, "
                   f"{len(self._aligned_data.rejected_symbols)} rejected")
        
        return self._aligned_data
    
    # ========================================================================
    # LAYER 1: STRUCTURAL PAIR SELECTION
    # ========================================================================
    
    def update_structural_pairs(
        self,
        price_data: Optional[Dict[str, pd.Series]] = None,
        ohlc_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> List[StructuralPairAssessment]:
        """
        Layer 1: Update structural pair selection.
        
        Uses aligned data if available, otherwise validates input.
        
        This should be called MONTHLY or when structure changes.
        """
        logger.info("Layer 1: Updating structural pair selection...")
        
        # Use aligned data if available
        if price_data is None and self._aligned_data:
            price_data = self._aligned_data.prices
            ohlc_data = self._aligned_data.ohlc
        
        if price_data is None:
            logger.error("No price data available")
            return []
        
        self._structural_pairs = self.pair_selector.select_valid_pairs(
            price_data=price_data,
            ohlc_data=ohlc_data
        )
        
        # Log tier distribution
        tier_dist = {"A": 0, "B": 0, "C": 0}
        for sp in self._structural_pairs:
            tier_dist[sp.tier] = tier_dist.get(sp.tier, 0) + 1
        
        logger.info(f"Structural pairs: {len(self._structural_pairs)} "
                   f"(A={tier_dist['A']}, B={tier_dist['B']}, C={tier_dist['C']})")
        
        return self._structural_pairs
    
    # ========================================================================
    # LAYER 2: REGIME FILTER
    # ========================================================================
    
    def update_regime(
        self,
        ohlc_data: pd.DataFrame,
        usdjpy: Optional[pd.Series] = None,
        audjpy: Optional[pd.Series] = None,
        upcoming_events: Optional[List[Dict]] = None
    ) -> FXRegimeAssessment:
        """
        Layer 2: Update regime assessment.
        
        This should be called DAILY or when regime may have changed.
        """
        logger.info("Layer 2: Assessing market regime...")
        
        self._regime_assessment = self.regime_filter.assess_regime(
            ohlc=ohlc_data,
            usdjpy=usdjpy,
            audjpy=audjpy,
            upcoming_events=upcoming_events
        )
        
        self._current_regime = self._regime_assessment.regime
        
        permits = REGIME_PERMITS_CRV.get(self._current_regime, False)
        
        logger.info(f"Regime: {self._current_regime.value}, "
                   f"Permits CRV: {'YES' if permits else 'NO'}, "
                   f"Confidence: {self._regime_assessment.confidence:.0%}")
        
        return self._regime_assessment
    
    # ========================================================================
    # LAYER 3-4: SPREAD ANALYSIS + SIGNAL GENERATION
    # ========================================================================
    
    def analyze_pair(
        self,
        pair: Tuple[str, str],
        price_a: pd.Series,
        price_b: pd.Series,
        regime_history: pd.Series,
        current_equity: float,
        timestamp: Optional[datetime] = None
    ) -> CRVAnalysisResult:
        """
        Complete analysis for a single pair (Layers 3-4).
        
        Integrates all layers to produce trading decision.
        
        GUARANTEES:
        - Never returns NaN in critical fields
        - Explicit is_valid flag
        - Clear rejection reasons
        """
        timestamp = timestamp or datetime.now()
        status_notes = []
        is_valid = True
        
        # === LAYER 1: Check structural validity ===
        structural = None
        for sp in self._structural_pairs:
            if sp.pair == pair or sp.pair == (pair[1], pair[0]):
                structural = sp
                break
        
        if structural is None:
            # Run structural assessment
            structural = self.pair_selector.assess_pair(
                pair, price_a, price_b
            )
        
        if not structural.is_structurally_valid:
            status_notes.append(f"Structural: INVALID - {structural.rejection_reasons[:1]}")
            return CRVAnalysisResult(
                pair=pair,
                timestamp=timestamp,
                structural=structural,
                regime=self._current_regime,
                regime_permits=False,
                spread_data=None,
                signal=None,
                is_tradeable=False,
                is_valid=True,
                status_notes=status_notes
            )
        
        status_notes.append(f"Structural: TIER {structural.tier} ({structural.structural_score:.0f})")
        
        # === LAYER 2: Check regime ===
        regime_permits = REGIME_PERMITS_CRV.get(self._current_regime, False)
        regime_confidence = self._regime_assessment.confidence if self._regime_assessment else 0.5
        
        if not regime_permits:
            status_notes.append(f"Regime: BLOCKED ({self._current_regime.value})")
        else:
            status_notes.append(f"Regime: OK ({self._current_regime.value})")
        
        # === LAYER 3: Analyze spread conditionally ===
        spread_data = self.spread_analyzer.analyze(
            price_a=price_a,
            price_b=price_b,
            regime_history=regime_history,
            current_regime=self._current_regime,
            timestamp=timestamp
        )
        
        if not spread_data.is_valid:
            status_notes.append(f"Spread: INVALID - {spread_data.invalidity_reason}")
            is_valid = False
        else:
            status_notes.append(f"Z-score: {spread_data.zscore_conditional:+.2f} (conditional)")
        
        # === LAYER 4: Generate signal ===
        risk_state = self.execution_manager.get_risk_state(current_equity)
        
        signal = self.signal_engine.generate_signal(
            pair=pair,
            zscore_conditional=spread_data.zscore_conditional,
            hedge_ratio=spread_data.hedge_ratio,
            regime=self._current_regime,
            regime_confidence=regime_confidence,
            risk_state=risk_state,
            spread_is_valid=spread_data.is_valid,
            timestamp=timestamp
        )
        
        if signal.signal_type != SignalType.NO_SIGNAL:
            status_notes.append(f"Signal: {signal.signal_type.value.upper()}")
        
        # === DETERMINE TRADEABILITY ===
        is_tradeable = (
            structural.is_structurally_valid and
            spread_data.is_valid and
            regime_permits and
            signal.is_valid and
            signal.signal_type in [SignalType.LONG_SPREAD, SignalType.SHORT_SPREAD]
        )
        
        return CRVAnalysisResult(
            pair=pair,
            timestamp=timestamp,
            structural=structural,
            regime=self._current_regime,
            regime_permits=regime_permits,
            spread_data=spread_data,
            signal=signal,
            is_tradeable=is_tradeable,
            is_valid=is_valid,
            status_notes=status_notes
        )
    
    def analyze_all_pairs(
        self,
        price_data: Dict[str, pd.Series],
        ohlc_data: Dict[str, pd.DataFrame],
        regime_history: pd.Series,
        current_equity: float,
        timestamp: Optional[datetime] = None
    ) -> List[CRVAnalysisResult]:
        """
        Analyze all structural pairs.
        
        Returns list of analysis results sorted by signal strength.
        """
        timestamp = timestamp or datetime.now()
        results = []
        
        logger.info(f"Analyzing {len(self._structural_pairs)} structural pairs...")
        
        for structural in self._structural_pairs:
            pair = structural.pair
            
            if pair[0] not in price_data or pair[1] not in price_data:
                continue
            
            result = self.analyze_pair(
                pair=pair,
                price_a=price_data[pair[0]],
                price_b=price_data[pair[1]],
                regime_history=regime_history,
                current_equity=current_equity,
                timestamp=timestamp
            )
            
            results.append(result)
        
        # Sort by tradeability, then signal strength
        results.sort(
            key=lambda x: (
                x.is_tradeable,
                x.signal.confidence if x.signal else 0,
                abs(x.spread_data.zscore_conditional) if x.spread_data and x.spread_data.is_valid else 0
            ),
            reverse=True
        )
        
        tradeable_count = sum(1 for r in results if r.is_tradeable)
        logger.info(f"Analysis complete: {tradeable_count} tradeable signals")
        
        return results
    
    # ========================================================================
    # LAYER 5: EXECUTION SAFETY
    # ========================================================================
    
    def check_system_safety(self, current_equity: float) -> Tuple[bool, List[str]]:
        """
        Check overall system safety.
        
        Returns:
            (is_safe, issues)
        """
        issues = []
        
        # Check kill-switch
        is_killed, kill_reason = self.execution_manager.check_kill_switch()
        if is_killed:
            issues.append(f"KILL-SWITCH ACTIVE: {kill_reason.value}")
        
        # Check drawdown
        drawdown = self.execution_manager.get_current_drawdown()
        if drawdown >= 0.05:
            issues.append(f"High drawdown: {drawdown:.1%}")
        
        # Check regime
        if not REGIME_PERMITS_CRV.get(self._current_regime, False):
            issues.append(f"Regime blocked: {self._current_regime.value}")
        
        # Check structural pairs
        if len(self._structural_pairs) == 0:
            issues.append("No structural pairs available")
        
        is_safe = len(issues) == 0
        
        return is_safe, issues
    
    # ========================================================================
    # STATE & REPORTING
    # ========================================================================
    
    def get_system_state(self, current_equity: float) -> CRVSystemState:
        """Get complete system state."""
        risk_state = self.execution_manager.get_risk_state(current_equity)
        
        # Determine activity status
        regime_permits = REGIME_PERMITS_CRV.get(self._current_regime, False)
        
        if risk_state.is_killed:
            is_active = False
            inactivity_reason = f"Kill-switch: {risk_state.kill_reason.value}"
            system_health = "critical"
        elif not regime_permits:
            is_active = False
            inactivity_reason = f"Regime {self._current_regime.value} blocks CRV"
            system_health = "warning"
        elif len(self._structural_pairs) == 0:
            is_active = False
            inactivity_reason = "No structural pairs available"
            system_health = "warning"
        else:
            is_active = True
            inactivity_reason = None
            system_health = "healthy"
        
        # Tier distribution
        tier_dist = {"A": 0, "B": 0, "C": 0}
        for sp in self._structural_pairs:
            tier_dist[sp.tier] = tier_dist.get(sp.tier, 0) + 1
        
        # Data stats
        data_valid = self._aligned_data is not None and len(self._aligned_data.symbols) > 0
        n_symbols = len(self._aligned_data.symbols) if self._aligned_data else 0
        n_rejected = len(self._aligned_data.rejected_symbols) if self._aligned_data else 0
        
        return CRVSystemState(
            timestamp=datetime.now(),
            data_valid=data_valid,
            n_symbols_loaded=n_symbols,
            n_symbols_rejected=n_rejected,
            n_structural_pairs=len(self._structural_pairs),
            structural_pairs=[p.pair for p in self._structural_pairs],
            tier_distribution=tier_dist,
            current_regime=self._current_regime,
            regime_permits_trading=regime_permits,
            regime_confidence=self._regime_assessment.confidence if self._regime_assessment else 0.0,
            n_signals=0,
            active_signals=[],
            risk_state=risk_state,
            n_positions=risk_state.n_positions,
            is_active=is_active,
            is_trading=risk_state.n_positions > 0,
            inactivity_reason=inactivity_reason,
            system_health=system_health
        )
    
    def generate_report(
        self,
        results: List[CRVAnalysisResult],
        current_equity: float
    ) -> str:
        """Generate comprehensive CRV analysis report."""
        lines = []
        lines.append("=" * 80)
        lines.append("FX CONDITIONAL RELATIVE VALUE (CRV) SYSTEM REPORT")
        lines.append("VERSION 2.0 - HARDENED")
        lines.append("=" * 80)
        lines.append(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # System state
        state = self.get_system_state(current_equity)
        
        # Health indicator
        health_icon = {"healthy": "🟢", "warning": "🟡", "critical": "🔴"}.get(state.system_health, "⚪")
        
        lines.append(f"\n{'-' * 80}")
        lines.append(f"SYSTEM HEALTH: {health_icon} {state.system_health.upper()}")
        lines.append(f"{'-' * 80}")
        
        # Data status
        lines.append(f"\n📊 DATA STATUS")
        lines.append(f"  Symbols loaded: {state.n_symbols_loaded}")
        lines.append(f"  Symbols rejected: {state.n_symbols_rejected}")
        
        # Structural pairs
        lines.append(f"\n🔧 STRUCTURAL PAIRS: {state.n_structural_pairs}")
        lines.append(f"  Tier A: {state.tier_distribution.get('A', 0)}")
        lines.append(f"  Tier B: {state.tier_distribution.get('B', 0)}")
        lines.append(f"  Tier C: {state.tier_distribution.get('C', 0)}")
        
        # Regime
        lines.append(f"\n📈 REGIME")
        lines.append(f"  Current: {state.current_regime.value}")
        lines.append(f"  Permits CRV: {'YES' if state.regime_permits_trading else 'NO'}")
        lines.append(f"  Confidence: {state.regime_confidence:.0%}")
        
        # Risk
        lines.append(f"\n⚠️ RISK STATUS")
        lines.append(f"  Kill-Switch: {'🔴 ACTIVE' if state.risk_state.is_killed else '🟢 OFF'}")
        lines.append(f"  Drawdown: {state.risk_state.current_drawdown:.1%}")
        lines.append(f"  Exposure: {state.risk_state.total_exposure:.1f}%")
        lines.append(f"  Positions: {state.n_positions}/{state.risk_state.max_positions}")
        
        if state.inactivity_reason:
            lines.append(f"\n  ⏸️ Inactivity: {state.inactivity_reason}")
        
        # Tradeable signals
        tradeable = [r for r in results if r.is_tradeable]
        
        if tradeable:
            lines.append(f"\n{'-' * 80}")
            lines.append(f"🚨 TRADEABLE SIGNALS ({len(tradeable)})")
            lines.append(f"{'-' * 80}")
            
            for r in tradeable:
                direction = r.signal.signal_type.value if r.signal else "unknown"
                lines.append(f"\n  → {r.pair[0]}/{r.pair[1]}: {direction.upper()}")
                lines.append(f"    Z-Score: {r.spread_data.zscore_conditional:+.2f}")
                lines.append(f"    Confidence: {r.signal.confidence:.0%}")
                lines.append(f"    Size: {r.signal.suggested_size_pct:.1f}%")
                lines.append(f"    Hedge: {r.spread_data.hedge_ratio:.4f}")
        else:
            lines.append(f"\n⏸️ NO TRADEABLE SIGNALS")
            lines.append("  This is expected - CRV requires confluence of conditions.")
        
        # Watchlist
        watchlist = [r for r in results if r.structural.is_structurally_valid and not r.is_tradeable]
        
        if watchlist[:5]:
            lines.append(f"\n{'-' * 80}")
            lines.append("WATCHLIST (Valid but Not Tradeable)")
            lines.append(f"{'-' * 80}")
            
            for r in watchlist[:5]:
                z = r.spread_data.zscore_conditional if r.spread_data and r.spread_data.is_valid else 0
                lines.append(f"\n  • {r.pair[0]}/{r.pair[1]}: Z={z:+.2f}")
                lines.append(f"    {', '.join(r.status_notes[:2])}")
        
        # Philosophy reminder
        lines.append(f"\n{'=' * 80}")
        lines.append("CRV PHILOSOPHY")
        lines.append(f"{'=' * 80}")
        lines.append("• This is NOT Statistical Arbitrage")
        lines.append("• FX does NOT exhibit permanent mean reversion")
        lines.append("• Inactivity is CORRECT behavior")
        lines.append("• SAFETY > PROFIT")
        
        lines.append("\n" + "=" * 80)
        
        return "\n".join(lines)
    
    def get_live_safety_checklist(
        self,
        fsm: Optional['CRVStateMachine'] = None
    ) -> Dict[str, any]:
        """
        Get checklist for live trading safety.
        
        FSM-AWARE: Skips checks that FSM forbids.
        
        Returns dict with:
        - bool values for actual checks
        - "SKIPPED" string for FSM-disabled checks
        """
        state = self.get_system_state(self._aligned_data.n_bars if self._aligned_data else 0)
        
        checklist = {}
        
        # Always-evaluated checks (data integrity, structure, regime)
        checklist["data_integrity_valid"] = state.data_valid
        checklist["structural_pairs_available"] = state.n_structural_pairs > 0
        checklist["regime_permits_trading"] = state.regime_permits_trading
        checklist["exposure_within_limits"] = state.risk_state.total_exposure < state.risk_state.max_exposure
        
        # FSM-GATED CHECKS
        if fsm is not None and FSM_AVAILABLE:
            # Drawdown check (FSM-GATED)
            if fsm.can_evaluate_drawdown():
                checklist["drawdown_acceptable"] = state.risk_state.current_drawdown < 0.05
            else:
                checklist["drawdown_acceptable"] = "SKIPPED"  # FSM forbids evaluation
                logger.debug("[SKIPPED] drawdown_acceptable — FSM forbids evaluation")
            
            # Kill-switch check (FSM-GATED)
            if fsm.can_place_orders():
                checklist["kill_switch_off"] = not state.risk_state.is_killed
            else:
                checklist["kill_switch_off"] = "SKIPPED"  # Execution disabled
                logger.debug("[SKIPPED] kill_switch_off — FSM execution disabled")
            
            # System health (MODE-AWARE)
            if fsm.mode == SystemMode.MODE_LIVE_CHECK:
                # In observational mode, health cannot be CRITICAL
                checklist["system_health_ok"] = True  # Neutral in observational mode
            else:
                checklist["system_health_ok"] = state.system_health in ["healthy", "warning"]
        else:
            # Legacy behavior (no FSM) - evaluate all
            checklist["drawdown_acceptable"] = state.risk_state.current_drawdown < 0.05
            checklist["kill_switch_off"] = not state.risk_state.is_killed
            checklist["system_health_ok"] = state.system_health in ["healthy", "warning"]
        
        return checklist
