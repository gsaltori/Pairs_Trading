"""
FX Conditional Relative Value Research Engine
Minimum Executable Vertical Slice

This package implements P1 (Spread Reversion) predictions with
optional P4 (Structural Stability) gating.
"""

__version__ = "0.2.0"
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
    
    # Predictions (base and gated)
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
]
