"""
FX Conditional Relative Value Research Engine
Minimum Executable Vertical Slice

This package implements:
- P1: Spread Reversion predictions
- P4: Structural Stability gating
- P5: Regime Memory learning
"""

__version__ = "0.3.0"
__author__ = "CRV Research Team"

from .config import CONFIG
from .observations import OHLCBar, MarketObservation, ObservationStream
from .spread import SpreadCalculator, SpreadObservation
from .predictions import (
    P1_SpreadReversionPrediction,
    PredictionGenerator,
    GatedPredictionGenerator,
    HypothesisContext,
    BlockedPrediction,
)
from .resolution import ResolutionState, P1_Resolver, ResolutionEngine
from .statistics import ResolutionStatistics, StatisticsAccumulator
from .structural import (
    StructuralValidity,
    StructuralState,
    StructuralConfig,
    StructuralStabilityEvaluator,
    StructuralGate,
    STRUCTURAL_CONFIG,
)
from .regime import (
    CorrelationStability,
    CorrelationTrend,
    VolatilityStability,
    SpreadDynamics,
    OutcomeType,
    RegimeConfig,
    RegimeSignature,
    RegimeOutcome,
    RegimeStats,
    RegimeMemory,
    RegimeEvaluation,
    RegimeEvaluator,
    RegimeGatedPredictionGenerator,
    OutcomeRecorder,
    create_regime_signature,
    convert_resolution_to_outcome,
    REGIME_CONFIG,
)

__all__ = [
    # Config
    'CONFIG',
    
    # Observations
    'OHLCBar',
    'MarketObservation',
    'ObservationStream',
    
    # Spread
    'SpreadCalculator',
    'SpreadObservation',
    
    # Predictions
    'P1_SpreadReversionPrediction',
    'PredictionGenerator',
    'GatedPredictionGenerator',
    'HypothesisContext',
    'BlockedPrediction',
    
    # Resolution
    'ResolutionState',
    'P1_Resolver',
    'ResolutionEngine',
    
    # Statistics
    'ResolutionStatistics',
    'StatisticsAccumulator',
    
    # Structural (P4)
    'StructuralValidity',
    'StructuralState',
    'StructuralConfig',
    'StructuralStabilityEvaluator',
    'StructuralGate',
    'STRUCTURAL_CONFIG',
    
    # Regime (P5)
    'CorrelationStability',
    'CorrelationTrend',
    'VolatilityStability',
    'SpreadDynamics',
    'OutcomeType',
    'RegimeConfig',
    'RegimeSignature',
    'RegimeOutcome',
    'RegimeStats',
    'RegimeMemory',
    'RegimeEvaluation',
    'RegimeEvaluator',
    'RegimeGatedPredictionGenerator',
    'OutcomeRecorder',
    'create_regime_signature',
    'convert_resolution_to_outcome',
    'REGIME_CONFIG',
]
