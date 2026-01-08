"""Analysis module - Statistical analysis for pairs trading."""

from src.analysis.correlation import CorrelationAnalyzer, CorrelationResult
from src.analysis.cointegration import CointegrationAnalyzer, CointegrationResult
from src.analysis.spread_builder import SpreadBuilder, SpreadMetrics
from src.analysis.pair_screener import PairScreener, PairScore, ScreeningResult
from src.analysis.institutional_selector import (
    InstitutionalPairSelector, 
    PairStatistics, 
    PipelineResult
)
from src.analysis.strict_selector import (
    StrictForexPairSelector,
    StrictPairAnalysis,
    StrictPipelineResult,
    VALID_PAIR_COMBINATIONS,
    TIMEFRAME_CONFIG,
    is_valid_forex_combination
)

__all__ = [
    'CorrelationAnalyzer', 'CorrelationResult',
    'CointegrationAnalyzer', 'CointegrationResult',
    'SpreadBuilder', 'SpreadMetrics',
    'PairScreener', 'PairScore', 'ScreeningResult',
    'InstitutionalPairSelector', 'PairStatistics', 'PipelineResult',
    'StrictForexPairSelector', 'StrictPairAnalysis', 'StrictPipelineResult',
    'VALID_PAIR_COMBINATIONS', 'TIMEFRAME_CONFIG', 'is_valid_forex_combination'
]
