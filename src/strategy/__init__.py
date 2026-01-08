"""Strategy module - Signal generation, pairs strategy, and conditional StatArb."""

from src.strategy.signals import SignalGenerator, SignalType, Signal
from src.strategy.pairs_strategy import PairsStrategy, PairAnalysis
from src.strategy.adaptive_params import (
    ParameterAdapter, 
    AdaptiveParameters, 
    format_parameters_report
)
from src.strategy.conditional_statarb import (
    PairState, MarketRegime, TradingSession,
    VolatilityMetrics, TrendMetrics, SpreadHealth, CointegrationStatus,
    RegimeAnalysis, PairStatus,
    MarketRegimeDetector, DynamicCointegrationValidator, SpreadHealthMonitor,
    REGIME_TRADEABLE
)
from src.strategy.conditional_manager import (
    ConditionalSignal, ConditionalSignalGenerator,
    ConditionalPairManager, ConditionalStatArbSystem
)

__all__ = [
    # Original
    'SignalGenerator', 'SignalType', 'Signal',
    'PairsStrategy', 'PairAnalysis',
    'ParameterAdapter', 'AdaptiveParameters', 'format_parameters_report',
    
    # Conditional StatArb
    'PairState', 'MarketRegime', 'TradingSession',
    'VolatilityMetrics', 'TrendMetrics', 'SpreadHealth', 'CointegrationStatus',
    'RegimeAnalysis', 'PairStatus',
    'MarketRegimeDetector', 'DynamicCointegrationValidator', 'SpreadHealthMonitor',
    'REGIME_TRADEABLE',
    'ConditionalSignal', 'ConditionalSignalGenerator',
    'ConditionalPairManager', 'ConditionalStatArbSystem'
]
