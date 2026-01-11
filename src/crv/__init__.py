"""
FX Conditional Relative Value (CRV) System - INSTITUTIONAL GRADE.

VERSION 2.1 - INSTITUTIONAL HARDENING

ARCHITECTURE:
    Layer 0: Data Integrity (MANDATORY)
    Layer 1: FX-Native Structural Pair Selection
    Layer 2: Regime Filter
    Layer 3: Conditional Spread Analysis
    Layer 4: Signal Generation
    Layer 5: Execution Safety
    
INSTITUTIONAL MODULES:
    FSM: Finite State Machine for mode control
    Kill-Switch: Auditable kill-switch with explicit causes
    Execution Adapter: MT5 integration ready

PHILOSOPHY:
    - FX does NOT exhibit permanent mean reversion
    - Cointegration is NOT used
    - The system is designed to NOT trade most of the time
    - Inactivity is CORRECT behavior
    - Safety > Profit
    - Zero trades is a SUCCESS state
"""

# ============================================================================
# FINITE STATE MACHINE (FSM)
# ============================================================================

from src.crv.state_machine import (
    # Main FSM
    CRVStateMachine,
    SystemMode,
    ModeCapabilities,
    MODE_CAPABILITIES,
    VALID_TRANSITIONS,
    
    # Errors
    FSMError,
    InvalidModeError,
    InvalidTransitionError,
    ModeCapabilityError,
    ModeNotInitializedError,
    
    # Decorators
    requires_signal_capability,
    requires_drawdown_capability,
    requires_order_capability,
    
    # Convenience
    create_fsm,
)

# ============================================================================
# INSTITUTIONAL KILL-SWITCH
# ============================================================================

from src.crv.kill_switch import (
    # Kill-switch manager
    InstitutionalKillSwitch,
    KillSwitchPrimaryReason,
    KillSwitchState,
    
    # Equity state
    EquityState,
    
    # Exceptions
    SystemLogicError,
)

# ============================================================================
# EXECUTION ADAPTER
# ============================================================================

from src.crv.execution_adapter import (
    # Adapter interface
    ExecutionAdapter,
    MT5ExecutionAdapter,
    MT5ConnectionManager,
    
    # Order types
    CRVOrder,
    OrderSide,
    OrderType,
    OrderStatus,
    MT5Position,
)

# ============================================================================
# Layer 0: Data Integrity
# ============================================================================

from src.crv.data_integrity import (
    FXDataValidator,
    DataQualityReport,
    AlignedDataset,
    safe_returns,
    safe_rolling_correlation,
    safe_zscore,
    check_price_sanity,
    verify_spread_integrity,
)

# ============================================================================
# Layer 1: Pair Selection (FX-Native)
# ============================================================================

from src.crv.pair_selector import (
    FXStructuralPairSelector,
    StructuralPairSelector,  # Alias
    StructuralPairAssessment,
    ConditionalCorrelation,
    OperationalViability,
    
    # Macro Framework
    MacroDriver,
    PairRelationship,
    FX_PAIR_RELATIONSHIPS,
    FX_RELATIONSHIPS,
    CURRENCY_MACRO_PROFILE,
    get_macro_coherence,
    has_macro_coherence,
    has_economic_coherence,
    get_pair_relationship,
    
    # Assessment Functions
    assess_conditional_correlation,
    assess_operational_viability,
)

# ============================================================================
# Layer 2: Regime Filter
# ============================================================================

from src.crv.regime_filter import (
    FXRegimeFilter,
    FXRegimeAssessment,
    FXRegime,
    RiskSentiment,
    VolatilityRegimeData,
    TrendRegimeData,
    RiskSentimentData,
    MacroEventData,
    REGIME_PERMITS_CRV,
    HIGH_IMPACT_EVENTS,
)

# ============================================================================
# Layer 3: Conditional Spread
# ============================================================================

from src.crv.conditional_spread import (
    ConditionalSpreadAnalyzer,
    ConditionalSpreadData,
    RegimeStatistics,
    SpreadDecomposition,
    safe_pct_change,
    safe_statistics,
    MIN_STD_THRESHOLD,
    INVALID_ZSCORE,
)

# ============================================================================
# Layer 4-5: Signal Engine & Execution Safety
# ============================================================================

from src.crv.execution_safety import (
    # Signal Engine
    CRVSignalEngine,
    CRVSignal,
    SignalType,
    
    # Execution Safety Manager (legacy)
    ExecutionSafetyManager,
    ExecutionConstraints,
    
    # Risk Management
    CRVRiskManager,  # Alias for ExecutionSafetyManager
    RiskState,
    Position,
    
    # Enums
    ExitReason,
    KillSwitchReason,
)

# ============================================================================
# Main System
# ============================================================================

from src.crv.crv_system import (
    FXConditionalRelativeValueSystem,
    CRVSystemState,
    CRVAnalysisResult,
)

__all__ = [
    # FSM
    'CRVStateMachine',
    'SystemMode',
    'ModeCapabilities',
    'MODE_CAPABILITIES',
    'VALID_TRANSITIONS',
    'FSMError',
    'InvalidModeError',
    'InvalidTransitionError',
    'ModeCapabilityError',
    'ModeNotInitializedError',
    'requires_signal_capability',
    'requires_drawdown_capability',
    'requires_order_capability',
    'create_fsm',
    
    # Kill-Switch
    'InstitutionalKillSwitch',
    'KillSwitchPrimaryReason',
    'KillSwitchState',
    'EquityState',
    'SystemLogicError',
    
    # Execution Adapter
    'ExecutionAdapter',
    'MT5ExecutionAdapter',
    'MT5ConnectionManager',
    'CRVOrder',
    'OrderSide',
    'OrderType',
    'OrderStatus',
    'MT5Position',
    
    # Layer 0: Data Integrity
    'FXDataValidator',
    'DataQualityReport',
    'AlignedDataset',
    'safe_returns',
    'safe_rolling_correlation',
    'safe_zscore',
    'check_price_sanity',
    'verify_spread_integrity',
    
    # Layer 1: Pair Selection
    'FXStructuralPairSelector',
    'StructuralPairSelector',
    'StructuralPairAssessment',
    'ConditionalCorrelation',
    'OperationalViability',
    'MacroDriver',
    'PairRelationship',
    'FX_PAIR_RELATIONSHIPS',
    'FX_RELATIONSHIPS',
    'CURRENCY_MACRO_PROFILE',
    'get_macro_coherence',
    'has_macro_coherence',
    'has_economic_coherence',
    'get_pair_relationship',
    'assess_conditional_correlation',
    'assess_operational_viability',
    
    # Layer 2: Regime Filter
    'FXRegimeFilter',
    'FXRegimeAssessment',
    'FXRegime',
    'RiskSentiment',
    'VolatilityRegimeData',
    'TrendRegimeData',
    'RiskSentimentData',
    'MacroEventData',
    'REGIME_PERMITS_CRV',
    'HIGH_IMPACT_EVENTS',
    
    # Layer 3: Conditional Spread
    'ConditionalSpreadAnalyzer',
    'ConditionalSpreadData',
    'RegimeStatistics',
    'SpreadDecomposition',
    'safe_pct_change',
    'safe_statistics',
    'MIN_STD_THRESHOLD',
    'INVALID_ZSCORE',
    
    # Layer 4-5: Signals & Execution
    'CRVSignalEngine',
    'CRVSignal',
    'SignalType',
    'ExecutionSafetyManager',
    'ExecutionConstraints',
    'CRVRiskManager',
    'RiskState',
    'Position',
    'ExitReason',
    'KillSwitchReason',
    
    # Main System
    'FXConditionalRelativeValueSystem',
    'CRVSystemState',
    'CRVAnalysisResult',
]
