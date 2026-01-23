"""
Gatekeeper Engine - Structural Market Filter

Wraps the existing validated TradeGatekeeper for production use.
Computes observables from live EURUSD/GBPUSD data.

BLOCKING RULES (LOCKED - EMPIRICALLY VALIDATED):
1. |Z-score| > 3.0 → BLOCK
2. Correlation trend < -0.05 → BLOCK  
3. Volatility ratio < 0.7 → BLOCK

Any single condition triggers a block.
"""

import logging
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict
from enum import Enum

from .config import GatekeeperConfig


class BlockReason(Enum):
    """Reason for blocking a trade."""
    EXTREME_SPREAD = "EXTREME_SPREAD"
    DETERIORATING_CORRELATION = "DETERIORATING_CORRELATION"
    COMPRESSED_VOLATILITY = "COMPRESSED_VOLATILITY"


@dataclass(frozen=True)
class GatekeeperDecision:
    """
    Immutable gatekeeper decision.
    
    Contains decision and full observables for audit.
    """
    allowed: bool
    reasons: tuple  # Tuple of BlockReason
    timestamp: datetime
    
    # Observables at decision time
    zscore: float
    correlation: float
    correlation_trend: float
    volatility_ratio: float
    
    @property
    def is_blocked(self) -> bool:
        return not self.allowed
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "allowed": self.allowed,
            "reasons": [r.value for r in self.reasons],
            "timestamp": self.timestamp.isoformat(),
            "zscore": self.zscore,
            "correlation": self.correlation,
            "correlation_trend": self.correlation_trend,
            "volatility_ratio": self.volatility_ratio,
        }


class GatekeeperEngine:
    """
    Production gatekeeper engine.
    
    Computes observables from EURUSD/GBPUSD price streams
    and makes ALLOW/BLOCK decisions.
    """
    
    def __init__(
        self,
        config: GatekeeperConfig,
        logger: Optional[logging.Logger] = None,
    ):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Price history
        self._prices_a: List[float] = []  # EURUSD
        self._prices_b: List[float] = []  # GBPUSD
        
        # Correlation history for trend
        self._correlation_history: List[float] = []
        
        # Current observables (updated each bar)
        self._zscore: float = 0.0
        self._correlation: float = 0.0
        self._correlation_trend: float = 0.0
        self._volatility_ratio: float = 1.0
    
    @property
    def is_ready(self) -> bool:
        """Check if enough data for valid observables."""
        return len(self._prices_a) >= self.config.zscore_window
    
    @property
    def current_observables(self) -> Dict:
        """Get current observable values."""
        return {
            "zscore": self._zscore,
            "correlation": self._correlation,
            "correlation_trend": self._correlation_trend,
            "volatility_ratio": self._volatility_ratio,
        }
    
    def update(self, price_eurusd: float, price_gbpusd: float) -> None:
        """
        Update with new price data.
        
        Call this on every new bar BEFORE requesting a decision.
        """
        self._prices_a.append(price_eurusd)
        self._prices_b.append(price_gbpusd)
        
        # Compute observables if enough data
        if self.is_ready:
            self._compute_observables()
        
        # Trim history to save memory
        max_history = max(
            self.config.zscore_window,
            self.config.correlation_window,
            self.config.volatility_window
        ) + 50
        
        if len(self._prices_a) > max_history:
            self._prices_a = self._prices_a[-max_history:]
            self._prices_b = self._prices_b[-max_history:]
        
        if len(self._correlation_history) > 50:
            self._correlation_history = self._correlation_history[-50:]
    
    def _compute_observables(self) -> None:
        """Compute all observables from current price history."""
        # Z-score
        self._zscore = self._compute_zscore()
        
        # Correlation
        self._correlation = self._compute_correlation()
        self._correlation_history.append(self._correlation)
        
        # Correlation trend
        self._correlation_trend = self._compute_correlation_trend()
        
        # Volatility ratio
        self._volatility_ratio = self._compute_volatility_ratio()
    
    def _compute_zscore(self) -> float:
        """Compute spread Z-score."""
        window = self.config.zscore_window
        
        if len(self._prices_a) < window:
            return 0.0
        
        a = np.array(self._prices_a[-window:])
        b = np.array(self._prices_b[-window:])
        
        # Hedge ratio (OLS)
        cov = np.cov(a, b)[0, 1]
        var_b = np.var(b)
        
        if var_b < 1e-10:
            return 0.0
        
        hedge_ratio = np.clip(cov / var_b, 0.1, 10.0)
        
        # Spread
        spread = a - hedge_ratio * b
        spread_mean = np.mean(spread)
        spread_std = np.std(spread)
        
        if spread_std < 1e-10:
            return 0.0
        
        current_spread = self._prices_a[-1] - hedge_ratio * self._prices_b[-1]
        zscore = (current_spread - spread_mean) / spread_std
        
        return float(np.clip(zscore, -10, 10))
    
    def _compute_correlation(self) -> float:
        """Compute rolling correlation."""
        window = self.config.correlation_window
        
        if len(self._prices_a) < window:
            return 0.0
        
        a = np.array(self._prices_a[-window:])
        b = np.array(self._prices_b[-window:])
        
        corr = np.corrcoef(a, b)[0, 1]
        return float(corr) if not np.isnan(corr) else 0.0
    
    def _compute_correlation_trend(self) -> float:
        """Compute correlation trend (change over last 10 samples)."""
        if len(self._correlation_history) < 10:
            return 0.0
        
        recent = self._correlation_history[-10:]
        return recent[-1] - recent[0]
    
    def _compute_volatility_ratio(self) -> float:
        """Compute volatility ratio (vol_A / vol_B)."""
        window = self.config.volatility_window
        
        if len(self._prices_a) < window:
            return 1.0
        
        a = np.array(self._prices_a[-window:])
        b = np.array(self._prices_b[-window:])
        
        returns_a = np.diff(a) / a[:-1]
        returns_b = np.diff(b) / b[:-1]
        
        vol_a = np.std(returns_a) if len(returns_a) > 0 else 0.01
        vol_b = np.std(returns_b) if len(returns_b) > 0 else 0.01
        
        if vol_b < 1e-10:
            return 1.0
        
        return float(vol_a / vol_b)
    
    def evaluate(self) -> GatekeeperDecision:
        """
        Evaluate current conditions and return ALLOW/BLOCK decision.
        
        Returns:
            GatekeeperDecision with allowed status and reasons
        """
        reasons: List[BlockReason] = []
        
        if not self.is_ready:
            # Allow by default if not enough data (conservative: could block instead)
            return GatekeeperDecision(
                allowed=True,
                reasons=tuple(),
                timestamp=datetime.utcnow(),
                zscore=self._zscore,
                correlation=self._correlation,
                correlation_trend=self._correlation_trend,
                volatility_ratio=self._volatility_ratio,
            )
        
        # Rule 1: Extreme spread (|Z| > 3.0)
        if abs(self._zscore) > self.config.zscore_extreme_threshold:
            reasons.append(BlockReason.EXTREME_SPREAD)
        
        # Rule 2: Deteriorating correlation (trend < -0.05)
        if self._correlation_trend < self.config.correlation_deteriorating_threshold:
            reasons.append(BlockReason.DETERIORATING_CORRELATION)
        
        # Rule 3: Compressed volatility (ratio < 0.7)
        if self._volatility_ratio < self.config.volatility_compressed_threshold:
            reasons.append(BlockReason.COMPRESSED_VOLATILITY)
        
        allowed = len(reasons) == 0
        
        decision = GatekeeperDecision(
            allowed=allowed,
            reasons=tuple(reasons),
            timestamp=datetime.utcnow(),
            zscore=self._zscore,
            correlation=self._correlation,
            correlation_trend=self._correlation_trend,
            volatility_ratio=self._volatility_ratio,
        )
        
        if not allowed:
            self.logger.info(f"GATEKEEPER BLOCK: {[r.value for r in reasons]}")
        
        return decision
    
    def reset(self) -> None:
        """Reset engine state (use with caution)."""
        self._prices_a.clear()
        self._prices_b.clear()
        self._correlation_history.clear()
        self._zscore = 0.0
        self._correlation = 0.0
        self._correlation_trend = 0.0
        self._volatility_ratio = 1.0
