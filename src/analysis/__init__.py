"""
Analysis module for the Pairs Trading System.
"""

from .correlation import CorrelationAnalyzer
from .cointegration import CointegrationAnalyzer
from .spread_builder import SpreadBuilder

__all__ = ['CorrelationAnalyzer', 'CointegrationAnalyzer', 'SpreadBuilder']
