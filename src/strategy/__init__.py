"""
Strategy module for the Pairs Trading System.
"""

from .signals import SignalGenerator, Signal, SignalType
from .pairs_strategy import PairsStrategy

__all__ = ['SignalGenerator', 'Signal', 'SignalType', 'PairsStrategy']
