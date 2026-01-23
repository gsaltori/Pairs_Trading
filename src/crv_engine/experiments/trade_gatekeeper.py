"""
Trade Gatekeeper Module

A risk firewall that blocks trade execution in empirically proven failure regimes.
Based on Edge Boundary Analysis findings from canonical EURUSD/GBPUSD experiment.

BLOCKING RULES (empirically derived, not tunable):
1. Block if |Z-score| > 3.0 (EXTREME spread state → 0% CRR)
2. Block if correlation_trend == DETERIORATING (→ 34.1% CRR)
3. Block if volatility_ratio == COMPRESSED (→ 42.3% CRR)

This module does NOT:
- Generate signals
- Optimize thresholds
- Score or weight conditions
- Modify CRV logic

This module DOES:
- Provide binary veto for external strategies
- Make it impossible to trade in proven failure regimes
- Remain completely strategy-agnostic
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple
from datetime import datetime, timezone

# Import edge boundary observables
from .edge_boundary import (
    PredictionObservables,
    CorrelationTrendBucket,
    VolatilityRatioBucket,
    classify_volatility_ratio,
)


# =============================================================================
# BLOCKING REASONS
# =============================================================================

class BlockReason(Enum):
    """
    Explicit semantic labels for blocking conditions.
    
    Each reason corresponds to an empirically validated failure mode
    from the Edge Boundary Analysis.
    """
    
    EXTREME_SPREAD = "EXTREME_SPREAD"
    # |Z-score| > 3.0 at prediction time
    # Empirical result: CRR = 0.0% (n=55)
    # Interpretation: Complete edge collapse
    
    DETERIORATING_CORRELATION = "DETERIORATING_CORRELATION"
    # Correlation trend is negative (declining relationship)
    # Empirical result: CRR = 34.1% (n=44)
    # Interpretation: Structural instability
    
    COMPRESSED_VOLATILITY = "COMPRESSED_VOLATILITY"
    # Volatility ratio < 0.7 (symbol A much less volatile than B)
    # Empirical result: CRR = 42.3% (n=26)
    # Interpretation: Regime imbalance


# =============================================================================
# TRADE PERMISSION
# =============================================================================

@dataclass(frozen=True)
class TradePermission:
    """
    Binary trade permission with audit trail.
    
    Attributes:
        allowed: True if trade is permitted, False if blocked
        reasons: List of blocking reasons (empty if allowed)
        timestamp: When permission was evaluated
        observables_snapshot: Key values at evaluation time
    """
    allowed: bool
    reasons: Tuple[BlockReason, ...]
    timestamp: datetime
    observables_snapshot: dict
    
    @property
    def is_blocked(self) -> bool:
        """Convenience property for blocked state."""
        return not self.allowed
    
    @property
    def reason_labels(self) -> List[str]:
        """Human-readable reason labels."""
        return [r.value for r in self.reasons]
    
    def __str__(self) -> str:
        if self.allowed:
            return "ALLOWED"
        else:
            return f"BLOCKED: {', '.join(self.reason_labels)}"


# =============================================================================
# THRESHOLD CONSTANTS (FIXED - Empirically derived)
# =============================================================================

# Z-score threshold for EXTREME spread state
ZSCORE_EXTREME_THRESHOLD = 3.0

# Correlation trend threshold for DETERIORATING
# correlation_trend < -0.05 means declining correlation
CORRELATION_TREND_DETERIORATING_THRESHOLD = -0.05

# Volatility ratio threshold for COMPRESSED
# volatility_ratio < 0.7 means symbol A much less volatile than B
VOLATILITY_RATIO_COMPRESSED_THRESHOLD = 0.7


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _classify_correlation_trend_from_delta(trend_delta: float) -> CorrelationTrendBucket:
    """
    Classify correlation trend from pre-computed delta value.
    
    The PredictionObservables already contains the trend (delta),
    so we classify directly without needing prior/current values.
    
    Args:
        trend_delta: Correlation change (current - prior)
    
    Returns:
        CorrelationTrendBucket classification
    """
    if trend_delta < CORRELATION_TREND_DETERIORATING_THRESHOLD:
        return CorrelationTrendBucket.DETERIORATING
    elif trend_delta > 0.05:  # IMPROVING threshold
        return CorrelationTrendBucket.IMPROVING
    else:
        return CorrelationTrendBucket.STABLE


# =============================================================================
# TRADE GATEKEEPER
# =============================================================================

class TradeGatekeeper:
    """
    Risk firewall that blocks trades in empirically proven failure regimes.
    
    USAGE:
        gatekeeper = TradeGatekeeper()
        permission = gatekeeper.evaluate(observables)
        
        if permission.allowed:
            strategy.execute_trade()
        else:
            log(f"Trade blocked: {permission.reasons}")
    
    BLOCKING THRESHOLDS (FIXED - DO NOT MODIFY):
        - Z-score: |Z| > 3.0
        - Correlation trend: < -0.05 (DETERIORATING bucket)
        - Volatility ratio: < 0.7 (COMPRESSED bucket)
    
    These thresholds are derived from empirical failure boundaries,
    not optimized for performance.
    """
    
    def __init__(self):
        """
        Initialize gatekeeper.
        
        No configuration parameters. All thresholds are fixed
        based on empirical failure analysis.
        """
        self._evaluation_count = 0
        self._block_count = 0
        self._block_reason_counts = {reason: 0 for reason in BlockReason}
    
    def evaluate(
        self,
        observables: PredictionObservables,
    ) -> TradePermission:
        """
        Evaluate trade permission based on current observables.
        
        Args:
            observables: Current market state observables
        
        Returns:
            TradePermission with allowed/blocked status and reasons
        """
        self._evaluation_count += 1
        
        reasons: List[BlockReason] = []
        
        # RULE 1: Block if |Z-score| > 3.0 (EXTREME spread)
        if abs(observables.zscore) > ZSCORE_EXTREME_THRESHOLD:
            reasons.append(BlockReason.EXTREME_SPREAD)
        
        # RULE 2: Block if correlation trend is DETERIORATING
        corr_trend_bucket = _classify_correlation_trend_from_delta(
            observables.correlation_trend
        )
        if corr_trend_bucket == CorrelationTrendBucket.DETERIORATING:
            reasons.append(BlockReason.DETERIORATING_CORRELATION)
        
        # RULE 3: Block if volatility ratio is COMPRESSED
        vol_bucket = classify_volatility_ratio(observables.volatility_ratio)
        if vol_bucket == VolatilityRatioBucket.COMPRESSED:
            reasons.append(BlockReason.COMPRESSED_VOLATILITY)
        
        # Build permission
        allowed = len(reasons) == 0
        
        # Update statistics
        if not allowed:
            self._block_count += 1
            for reason in reasons:
                self._block_reason_counts[reason] += 1
        
        # Create snapshot for audit trail
        snapshot = {
            'zscore': observables.zscore,
            'correlation_trend': observables.correlation_trend,
            'volatility_ratio': observables.volatility_ratio,
            'correlation': observables.correlation,
        }
        
        return TradePermission(
            allowed=allowed,
            reasons=tuple(reasons),
            timestamp=datetime.now(timezone.utc),
            observables_snapshot=snapshot,
        )
    
    def evaluate_raw(
        self,
        zscore: float,
        correlation_trend: float,
        volatility_ratio: float,
        correlation: float = 0.8,
    ) -> TradePermission:
        """
        Evaluate trade permission from raw values.
        
        Convenience method for strategies that don't use PredictionObservables.
        
        Args:
            zscore: Current Z-score
            correlation_trend: Recent correlation change
            volatility_ratio: Vol(A) / Vol(B)
            correlation: Current correlation (optional, for audit)
        
        Returns:
            TradePermission with allowed/blocked status
        """
        # Create temporary observables
        obs = PredictionObservables(
            prediction_id="raw_eval",
            outcome="PENDING",
            bars_to_resolution=0,
            correlation=correlation,
            correlation_trend=correlation_trend,
            volatility_ratio=volatility_ratio,
            zscore=zscore,
            spread_velocity=0.0,
        )
        return self.evaluate(obs)
    
    # =========================================================================
    # STATISTICS
    # =========================================================================
    
    @property
    def evaluation_count(self) -> int:
        """Total number of evaluations."""
        return self._evaluation_count
    
    @property
    def block_count(self) -> int:
        """Total number of blocked trades."""
        return self._block_count
    
    @property
    def block_rate(self) -> float:
        """Fraction of evaluations that were blocked."""
        if self._evaluation_count == 0:
            return 0.0
        return self._block_count / self._evaluation_count
    
    @property
    def allow_count(self) -> int:
        """Total number of allowed trades."""
        return self._evaluation_count - self._block_count
    
    def get_block_reason_distribution(self) -> dict:
        """
        Get distribution of blocking reasons.
        
        Returns dict mapping BlockReason to count.
        Note: A single blocked trade may have multiple reasons.
        """
        return {
            reason.value: count 
            for reason, count in self._block_reason_counts.items()
        }
    
    def reset_statistics(self) -> None:
        """Reset all statistics counters."""
        self._evaluation_count = 0
        self._block_count = 0
        self._block_reason_counts = {reason: 0 for reason in BlockReason}
    
    def get_summary(self) -> dict:
        """Get summary of gatekeeper activity."""
        return {
            'total_evaluations': self._evaluation_count,
            'blocked': self._block_count,
            'allowed': self.allow_count,
            'block_rate': self.block_rate,
            'reason_distribution': self.get_block_reason_distribution(),
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_gatekeeper() -> TradeGatekeeper:
    """
    Factory function for TradeGatekeeper.
    
    Returns a configured gatekeeper with all empirical thresholds.
    """
    return TradeGatekeeper()


def check_trade_permission(
    zscore: float,
    correlation_trend: float,
    volatility_ratio: float,
) -> bool:
    """
    Quick check if trade would be allowed.
    
    Stateless convenience function for simple integrations.
    
    Args:
        zscore: Current Z-score
        correlation_trend: Recent correlation change  
        volatility_ratio: Vol(A) / Vol(B)
    
    Returns:
        True if trade is allowed, False if blocked
    """
    # Rule 1: Extreme spread
    if abs(zscore) > ZSCORE_EXTREME_THRESHOLD:
        return False
    
    # Rule 2: Deteriorating correlation
    if correlation_trend < CORRELATION_TREND_DETERIORATING_THRESHOLD:
        return False
    
    # Rule 3: Compressed volatility
    if volatility_ratio < VOLATILITY_RATIO_COMPRESSED_THRESHOLD:
        return False
    
    return True
