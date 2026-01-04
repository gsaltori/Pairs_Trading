"""
Data module for the Pairs Trading System.
"""

from .broker_client import OandaClient
from .data_manager import DataManager

__all__ = ['OandaClient', 'DataManager']
