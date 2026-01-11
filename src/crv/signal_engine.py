"""
FX Conditional Relative Value (CRV) System - Signal Engine (DEPRECATED).

This module is DEPRECATED. All functionality has been moved to:
- execution_safety.py

This file exists for backward compatibility only.
"""

# Re-export everything from execution_safety for backward compatibility
from src.crv.execution_safety import (
    # Main classes
    CRVSignalEngine,
    ExecutionSafetyManager as CRVRiskManager,
    
    # Data classes
    CRVSignal,
    Position,
    RiskState,
    
    # Enums
    SignalType,
    ExitReason,
    KillSwitchReason,
)

import warnings

warnings.warn(
    "signal_engine module is deprecated. "
    "Import from execution_safety instead.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = [
    'CRVSignalEngine',
    'CRVRiskManager',
    'CRVSignal',
    'Position',
    'RiskState',
    'SignalType',
    'ExitReason',
]
