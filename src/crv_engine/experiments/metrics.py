"""
Extended Metrics Module

Computes additional metrics from EXISTING engine output data.

CRITICAL CONSTRAINT:
These metrics use ONLY data already produced by the engine:
- Prediction outcomes (CONFIRMED, REFUTED, INVALIDATED, TIMEOUT)
- Bars elapsed to resolution
- Resolution timestamps

NO NEW SIGNALS. NO NEW INDICATORS. NO NEW DATA SOURCES.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import math


# Local implementations to avoid import conflicts with local statistics.py
def _median(data: List[float]) -> float:
    """Compute median of a list."""
    n = len(data)
    if n == 0:
        raise ValueError("No median for empty data")
    sorted_data = sorted(data)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_data[mid - 1] + sorted_data[mid]) / 2
    return sorted_data[mid]


def _mean(data: List[float]) -> float:
    """Compute mean of a list."""
    if not data:
        raise ValueError("No mean for empty data")
    return sum(data) / len(data)


def _stdev(data: List[float]) -> float:
    """Compute sample standard deviation."""
    n = len(data)
    if n < 2:
        raise ValueError("Need at least 2 data points for stdev")
    m = _mean(data)
    ss = sum((x - m) ** 2 for x in data)
    return math.sqrt(ss / (n - 1))


# =============================================================================
# METRIC DEFINITIONS
# =============================================================================

@dataclass
class ExtendedMetrics:
    """
    Extended metrics computed from existing engine output.
    
    ALL metrics are derived from resolution outcomes and timing data
    that the engine ALREADY produces.
    """
    # Primary metrics (from engine statistics)
    crr: float                          # Confirmation Rate Ratio
    invalidation_rate: float            # INVALIDATED / resolved
    timeout_rate: float                 # TIMEOUT / resolved
    
    # Sample sizes
    total_predictions: int
    resolved_count: int
    testable_count: int                 # CONFIRMED + REFUTED
    confirmed_count: int
    refuted_count: int
    invalidated_count: int
    timeout_count: int
    
    # ─────────────────────────────────────────────────────────────────────────
    # EXTENDED METRIC 1: Expected Resolution Value (ERV)
    # ─────────────────────────────────────────────────────────────────────────
    # WHY IT MATTERS:
    #   ERV weights outcomes by their informational value.
    #   - CONFIRMED: +1 (prediction correct)
    #   - REFUTED: -1 (prediction wrong)
    #   - INVALIDATED: 0 (structural failure, no signal quality info)
    #   - TIMEOUT: -0.5 (weak negative - reversion didn't occur)
    #
    # HOW IT REDUCES FALSE CONFIDENCE:
    #   CRR ignores invalidations and timeouts. A strategy with 60% CRR
    #   but 40% invalidation rate may have negative ERV when timeouts
    #   are considered. ERV provides a fuller picture.
    #
    # HOW IT AVOIDS OVERFITTING:
    #   ERV uses fixed weights (not tuned). The weights reflect logical
    #   information content, not optimized outcomes.
    # ─────────────────────────────────────────────────────────────────────────
    erv: float                          # Expected Resolution Value
    erv_per_prediction: float           # ERV / total predictions
    
    # ─────────────────────────────────────────────────────────────────────────
    # EXTENDED METRIC 2: Resolution Timing Asymmetry
    # ─────────────────────────────────────────────────────────────────────────
    # WHY IT MATTERS:
    #   If confirmations are slow but refutations are fast, the "edge"
    #   may be illusory - we're catching random walks that eventually
    #   revert, while genuine adverse moves are captured quickly.
    #
    # HOW IT REDUCES FALSE CONFIDENCE:
    #   A healthy edge should show FASTER confirmations than refutations.
    #   Symmetric or inverted timing suggests random behavior.
    #
    # HOW IT AVOIDS OVERFITTING:
    #   Median is robust to outliers. No parameters are tuned.
    # ─────────────────────────────────────────────────────────────────────────
    median_bars_to_confirmation: Optional[float]
    median_bars_to_refutation: Optional[float]
    timing_asymmetry_ratio: Optional[float]  # confirm_median / refute_median
    
    mean_bars_to_confirmation: Optional[float]
    mean_bars_to_refutation: Optional[float]
    
    # ─────────────────────────────────────────────────────────────────────────
    # EXTENDED METRIC 3: Conditional CRR
    # ─────────────────────────────────────────────────────────────────────────
    # WHY IT MATTERS:
    #   CRR may vary based on how long structural stability persisted
    #   before the prediction. This metric segments CRR by stability duration.
    #
    # HOW IT REDUCES FALSE CONFIDENCE:
    #   If CRR is only high for short stability durations, the edge may
    #   depend on specific market conditions, not a general phenomenon.
    #
    # HOW IT AVOIDS OVERFITTING:
    #   Uses simple duration buckets (short/medium/long), not optimized bins.
    # ─────────────────────────────────────────────────────────────────────────
    conditional_crr_short_stability: Optional[float]   # < 20 bars stable
    conditional_crr_medium_stability: Optional[float]  # 20-40 bars stable
    conditional_crr_long_stability: Optional[float]    # > 40 bars stable
    
    conditional_n_short: int = 0
    conditional_n_medium: int = 0
    conditional_n_long: int = 0
    
    # Distribution statistics
    bars_to_resolution_std: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            # Primary
            'crr': self.crr,
            'invalidation_rate': self.invalidation_rate,
            'timeout_rate': self.timeout_rate,
            
            # Sample sizes
            'total_predictions': self.total_predictions,
            'resolved_count': self.resolved_count,
            'testable_count': self.testable_count,
            'confirmed_count': self.confirmed_count,
            'refuted_count': self.refuted_count,
            'invalidated_count': self.invalidated_count,
            'timeout_count': self.timeout_count,
            
            # ERV
            'erv': self.erv,
            'erv_per_prediction': self.erv_per_prediction,
            
            # Timing
            'median_bars_to_confirmation': self.median_bars_to_confirmation,
            'median_bars_to_refutation': self.median_bars_to_refutation,
            'timing_asymmetry_ratio': self.timing_asymmetry_ratio,
            'mean_bars_to_confirmation': self.mean_bars_to_confirmation,
            'mean_bars_to_refutation': self.mean_bars_to_refutation,
            
            # Conditional CRR
            'conditional_crr_short_stability': self.conditional_crr_short_stability,
            'conditional_crr_medium_stability': self.conditional_crr_medium_stability,
            'conditional_crr_long_stability': self.conditional_crr_long_stability,
            'conditional_n_short': self.conditional_n_short,
            'conditional_n_medium': self.conditional_n_medium,
            'conditional_n_long': self.conditional_n_long,
            
            # Distribution
            'bars_to_resolution_std': self.bars_to_resolution_std,
        }


# =============================================================================
# METRIC COMPUTATION
# =============================================================================

@dataclass
class ResolutionRecord:
    """
    A single prediction resolution record.
    
    This is the INPUT data format - extracted from engine output.
    """
    prediction_id: str
    outcome: str  # "CONFIRMED", "REFUTED", "INVALIDATED", "TIMEOUT"
    bars_to_resolution: int
    stability_duration_at_creation: Optional[int] = None  # Bars of prior stability


# ERV weights (FIXED - NOT TUNABLE)
ERV_WEIGHTS = {
    'CONFIRMED': 1.0,
    'REFUTED': -1.0,
    'INVALIDATED': 0.0,  # No signal quality information
    'TIMEOUT': -0.5,     # Weak negative - reversion did not occur
}


def compute_extended_metrics(
    records: List[ResolutionRecord],
) -> ExtendedMetrics:
    """
    Compute extended metrics from resolution records.
    
    Args:
        records: List of resolution records from engine output
    
    Returns:
        ExtendedMetrics with all computed values
    """
    if not records:
        return _empty_metrics()
    
    # Count by outcome
    confirmed = [r for r in records if r.outcome == 'CONFIRMED']
    refuted = [r for r in records if r.outcome == 'REFUTED']
    invalidated = [r for r in records if r.outcome == 'INVALIDATED']
    timeout = [r for r in records if r.outcome == 'TIMEOUT']
    
    total = len(records)
    n_confirmed = len(confirmed)
    n_refuted = len(refuted)
    n_invalidated = len(invalidated)
    n_timeout = len(timeout)
    
    resolved = total  # All records are resolved
    testable = n_confirmed + n_refuted
    
    # Primary metrics
    crr = n_confirmed / testable if testable > 0 else 0.0
    invalidation_rate = n_invalidated / resolved if resolved > 0 else 0.0
    timeout_rate = n_timeout / resolved if resolved > 0 else 0.0
    
    # ─────────────────────────────────────────────────────────────────────────
    # ERV Computation
    # ─────────────────────────────────────────────────────────────────────────
    erv = sum(ERV_WEIGHTS.get(r.outcome, 0) for r in records)
    erv_per_prediction = erv / total if total > 0 else 0.0
    
    # ─────────────────────────────────────────────────────────────────────────
    # Timing Metrics
    # ─────────────────────────────────────────────────────────────────────────
    confirm_bars = [r.bars_to_resolution for r in confirmed]
    refute_bars = [r.bars_to_resolution for r in refuted]
    all_bars = [r.bars_to_resolution for r in records]
    
    median_confirm = _median(confirm_bars) if confirm_bars else None
    median_refute = _median(refute_bars) if refute_bars else None
    
    mean_confirm = _mean(confirm_bars) if confirm_bars else None
    mean_refute = _mean(refute_bars) if refute_bars else None
    
    # Timing asymmetry: < 1.0 means confirms are faster (good)
    if median_confirm is not None and median_refute is not None and median_refute > 0:
        timing_asymmetry = median_confirm / median_refute
    else:
        timing_asymmetry = None
    
    bars_std = _stdev(all_bars) if len(all_bars) > 1 else None
    
    # ─────────────────────────────────────────────────────────────────────────
    # Conditional CRR
    # ─────────────────────────────────────────────────────────────────────────
    # Bucket by stability duration at creation
    short_bucket = [r for r in records if r.stability_duration_at_creation is not None 
                    and r.stability_duration_at_creation < 20]
    medium_bucket = [r for r in records if r.stability_duration_at_creation is not None 
                     and 20 <= r.stability_duration_at_creation <= 40]
    long_bucket = [r for r in records if r.stability_duration_at_creation is not None 
                   and r.stability_duration_at_creation > 40]
    
    def bucket_crr(bucket: List[ResolutionRecord]) -> Optional[float]:
        testable = [r for r in bucket if r.outcome in ('CONFIRMED', 'REFUTED')]
        if len(testable) < 5:  # Minimum for meaningful CRR
            return None
        confirmed = sum(1 for r in testable if r.outcome == 'CONFIRMED')
        return confirmed / len(testable)
    
    cond_crr_short = bucket_crr(short_bucket)
    cond_crr_medium = bucket_crr(medium_bucket)
    cond_crr_long = bucket_crr(long_bucket)
    
    return ExtendedMetrics(
        # Primary
        crr=crr,
        invalidation_rate=invalidation_rate,
        timeout_rate=timeout_rate,
        
        # Sample sizes
        total_predictions=total,
        resolved_count=resolved,
        testable_count=testable,
        confirmed_count=n_confirmed,
        refuted_count=n_refuted,
        invalidated_count=n_invalidated,
        timeout_count=n_timeout,
        
        # ERV
        erv=erv,
        erv_per_prediction=erv_per_prediction,
        
        # Timing
        median_bars_to_confirmation=median_confirm,
        median_bars_to_refutation=median_refute,
        timing_asymmetry_ratio=timing_asymmetry,
        mean_bars_to_confirmation=mean_confirm,
        mean_bars_to_refutation=mean_refute,
        
        # Conditional CRR
        conditional_crr_short_stability=cond_crr_short,
        conditional_crr_medium_stability=cond_crr_medium,
        conditional_crr_long_stability=cond_crr_long,
        conditional_n_short=len([r for r in short_bucket if r.outcome in ('CONFIRMED', 'REFUTED')]),
        conditional_n_medium=len([r for r in medium_bucket if r.outcome in ('CONFIRMED', 'REFUTED')]),
        conditional_n_long=len([r for r in long_bucket if r.outcome in ('CONFIRMED', 'REFUTED')]),
        
        # Distribution
        bars_to_resolution_std=bars_std,
    )


def _empty_metrics() -> ExtendedMetrics:
    """Return metrics structure with zero/None values."""
    return ExtendedMetrics(
        crr=0.0,
        invalidation_rate=0.0,
        timeout_rate=0.0,
        total_predictions=0,
        resolved_count=0,
        testable_count=0,
        confirmed_count=0,
        refuted_count=0,
        invalidated_count=0,
        timeout_count=0,
        erv=0.0,
        erv_per_prediction=0.0,
        median_bars_to_confirmation=None,
        median_bars_to_refutation=None,
        timing_asymmetry_ratio=None,
        mean_bars_to_confirmation=None,
        mean_bars_to_refutation=None,
        conditional_crr_short_stability=None,
        conditional_crr_medium_stability=None,
        conditional_crr_long_stability=None,
    )


# =============================================================================
# METRIC INTERPRETATION HELPERS
# =============================================================================

def interpret_erv(erv_per_prediction: float) -> str:
    """
    Provide interpretation of ERV per prediction.
    
    This is for DOCUMENTATION only, not decision-making.
    """
    if erv_per_prediction > 0.3:
        return "STRONG_POSITIVE: Predictions have clear positive value"
    elif erv_per_prediction > 0.1:
        return "POSITIVE: Predictions have net positive value"
    elif erv_per_prediction > -0.1:
        return "NEUTRAL: No clear edge detected"
    elif erv_per_prediction > -0.3:
        return "NEGATIVE: Predictions have net negative value"
    else:
        return "STRONG_NEGATIVE: Predictions are consistently wrong"


def interpret_timing_asymmetry(ratio: Optional[float]) -> str:
    """
    Provide interpretation of timing asymmetry.
    
    ratio = median_confirm / median_refute
    < 1.0 means confirmations are faster (healthy)
    > 1.0 means refutations are faster (concerning)
    """
    if ratio is None:
        return "INSUFFICIENT_DATA"
    elif ratio < 0.7:
        return "HEALTHY: Confirmations significantly faster than refutations"
    elif ratio < 1.0:
        return "ACCEPTABLE: Confirmations somewhat faster"
    elif ratio < 1.3:
        return "WARNING: Timing nearly symmetric"
    else:
        return "CONCERNING: Refutations faster than confirmations"
