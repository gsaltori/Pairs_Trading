"""
Configuration constants for CRV Engine.

ALL VALUES ARE FIXED. NO TUNING PERMITTED.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class CRVConfig:
    """
    Immutable configuration for CRV Research Engine.
    
    These values are FIXED by specification and must not be changed.
    """
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PAIR CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════════
    SYMBOL_A: str = "EURUSD"
    SYMBOL_B: str = "GBPUSD"
    PAIR: Tuple[str, str] = (SYMBOL_A, SYMBOL_B)
    TIMEFRAME: str = "H4"
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SPREAD & Z-SCORE PARAMETERS (FIXED - NO TUNING)
    # ═══════════════════════════════════════════════════════════════════════════
    ZSCORE_WINDOW: int = 60           # Bars for rolling Z-score calculation
    HEDGE_RATIO_WINDOW: int = 60      # Bars for hedge ratio calculation
    
    # ═══════════════════════════════════════════════════════════════════════════
    # P1 PREDICTION THRESHOLDS (FIXED - NO TUNING)
    # ═══════════════════════════════════════════════════════════════════════════
    TRIGGER_THRESHOLD: float = 1.5    # |Z| > 1.5 triggers P1 generation
    CONFIRMATION_THRESHOLD: float = 0.3  # |Z| < 0.3 = CONFIRMED
    REFUTATION_THRESHOLD: float = 3.0    # |Z| > 3.0 = REFUTED
    MAX_HOLDING_BARS: int = 50        # Maximum bars before TIMEOUT
    
    # ═══════════════════════════════════════════════════════════════════════════
    # INVALIDATION THRESHOLDS (FIXED - NO TUNING)
    # ═══════════════════════════════════════════════════════════════════════════
    MIN_CORRELATION: float = 0.20     # Correlation below this = INVALIDATED
    MAX_CORRELATION_DROP: float = 0.30  # Drop > 0.30 from initial = INVALIDATED
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SYNTHETIC DATA PARAMETERS (FOR DEMO ONLY)
    # ═══════════════════════════════════════════════════════════════════════════
    DEMO_BARS: int = 1000             # Number of bars to generate
    DEMO_SEED: int = 42               # Random seed for reproducibility


# Global configuration instance
CONFIG = CRVConfig()
