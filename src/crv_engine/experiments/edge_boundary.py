"""
Edge Boundary Analysis Layer

Identifies failure boundaries and collapse conditions for CRV hypotheses.

DESIGN PRINCIPLES:
1. Operates on TOP of walk-forward results (no raw data reprocessing)
2. Uses ONLY already-available observables
3. Identifies NECESSARY (not sufficient) failure conditions
4. Failure is a feature, not a bug

THIS IS NOT:
- A filter
- An optimizer
- A strategy improvement

THIS IS:
- Failure cartography
- Edge death documentation
- State space boundary mapping
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import json
import math
from datetime import datetime, timezone


# =============================================================================
# OBSERVABLE BUCKETS
# =============================================================================

class CorrelationBucket(Enum):
    """Correlation level buckets."""
    VERY_LOW = "VERY_LOW"        # < 0.3
    LOW = "LOW"                  # 0.3 - 0.5
    MODERATE = "MODERATE"        # 0.5 - 0.7
    HIGH = "HIGH"                # 0.7 - 0.85
    VERY_HIGH = "VERY_HIGH"      # > 0.85


class CorrelationTrendBucket(Enum):
    """Correlation trend direction."""
    DETERIORATING = "DETERIORATING"  # Correlation dropping
    STABLE = "STABLE"                # Correlation flat
    IMPROVING = "IMPROVING"          # Correlation rising


class VolatilityRatioBucket(Enum):
    """Relative volatility between symbols."""
    COMPRESSED = "COMPRESSED"    # Vol ratio < 0.7
    BALANCED = "BALANCED"        # Vol ratio 0.7 - 1.3
    IMBALANCED = "IMBALANCED"    # Vol ratio 1.3 - 2.0
    EXTREME = "EXTREME"          # Vol ratio > 2.0


class SpreadBucket(Enum):
    """Spread state bucket."""
    CONTRACTED = "CONTRACTED"    # |Z| < 1.0
    NORMAL = "NORMAL"            # |Z| 1.0 - 2.0
    EXTENDED = "EXTENDED"        # |Z| 2.0 - 3.0
    EXTREME = "EXTREME"          # |Z| > 3.0


class RegimeTransitionBucket(Enum):
    """Regime signature transition type."""
    STABLE = "STABLE"            # No regime change
    MINOR = "MINOR"              # Small regime shift
    MAJOR = "MAJOR"              # Large regime flip


def classify_correlation_level(correlation: float) -> CorrelationBucket:
    """Classify correlation into bucket."""
    if correlation < 0.3:
        return CorrelationBucket.VERY_LOW
    elif correlation < 0.5:
        return CorrelationBucket.LOW
    elif correlation < 0.7:
        return CorrelationBucket.MODERATE
    elif correlation < 0.85:
        return CorrelationBucket.HIGH
    else:
        return CorrelationBucket.VERY_HIGH


def classify_correlation_trend(
    current_corr: float,
    prior_corr: float,
    threshold: float = 0.05,
) -> CorrelationTrendBucket:
    """Classify correlation trend direction."""
    delta = current_corr - prior_corr
    if delta < -threshold:
        return CorrelationTrendBucket.DETERIORATING
    elif delta > threshold:
        return CorrelationTrendBucket.IMPROVING
    else:
        return CorrelationTrendBucket.STABLE


def classify_volatility_ratio(vol_ratio: float) -> VolatilityRatioBucket:
    """Classify volatility ratio into bucket."""
    if vol_ratio < 0.7:
        return VolatilityRatioBucket.COMPRESSED
    elif vol_ratio < 1.3:
        return VolatilityRatioBucket.BALANCED
    elif vol_ratio < 2.0:
        return VolatilityRatioBucket.IMBALANCED
    else:
        return VolatilityRatioBucket.EXTREME


def classify_spread_state(zscore: float) -> SpreadBucket:
    """Classify spread state into bucket."""
    abs_z = abs(zscore)
    if abs_z < 1.0:
        return SpreadBucket.CONTRACTED
    elif abs_z < 2.0:
        return SpreadBucket.NORMAL
    elif abs_z < 3.0:
        return SpreadBucket.EXTENDED
    else:
        return SpreadBucket.EXTREME


# =============================================================================
# PREDICTION OBSERVABLE SNAPSHOT
# =============================================================================

@dataclass
class PredictionObservables:
    """
    Observable state at prediction creation time.
    
    These are extracted from EXISTING engine data, not computed fresh.
    Each prediction already knows its context.
    """
    prediction_id: str
    outcome: str  # CONFIRMED, REFUTED, INVALIDATED, TIMEOUT
    bars_to_resolution: int
    
    # Observables at prediction time
    correlation: float
    correlation_trend: float  # Change from prior window
    volatility_ratio: float   # vol_a / vol_b
    zscore: float
    spread_velocity: float    # Z-score change rate
    
    # Regime info (if available)
    regime_signature: Optional[str] = None
    prior_regime_signature: Optional[str] = None
    
    # Derived buckets
    @property
    def correlation_bucket(self) -> CorrelationBucket:
        return classify_correlation_level(self.correlation)
    
    @property
    def correlation_trend_bucket(self) -> CorrelationTrendBucket:
        return classify_correlation_trend(self.correlation, self.correlation - self.correlation_trend)
    
    @property
    def volatility_bucket(self) -> VolatilityRatioBucket:
        return classify_volatility_ratio(self.volatility_ratio)
    
    @property
    def spread_bucket(self) -> SpreadBucket:
        return classify_spread_state(self.zscore)
    
    @property
    def regime_transition_bucket(self) -> RegimeTransitionBucket:
        if self.regime_signature is None or self.prior_regime_signature is None:
            return RegimeTransitionBucket.STABLE
        if self.regime_signature == self.prior_regime_signature:
            return RegimeTransitionBucket.STABLE
        # Simple heuristic: different signatures = major transition
        return RegimeTransitionBucket.MAJOR
    
    @property
    def is_testable(self) -> bool:
        return self.outcome in ('CONFIRMED', 'REFUTED')
    
    @property
    def is_confirmed(self) -> bool:
        return self.outcome == 'CONFIRMED'
    
    @property
    def is_invalidated(self) -> bool:
        return self.outcome == 'INVALIDATED'


# =============================================================================
# CONDITIONAL METRICS
# =============================================================================

@dataclass
class BucketMetrics:
    """Metrics for a single condition bucket."""
    bucket_name: str
    bucket_value: str
    
    # Sample sizes
    total_predictions: int
    testable_count: int
    confirmed_count: int
    refuted_count: int
    invalidated_count: int
    timeout_count: int
    
    # Rates
    crr: float                # confirmed / testable
    invalidation_rate: float  # invalidated / total
    
    # Block-level metrics (for verdict persistence)
    blocks_with_data: int
    blocks_supported: int
    verdict_persistence: float  # blocks_supported / blocks_with_data
    
    # Edge quality indicators
    mean_bars_to_confirm: Optional[float]
    mean_bars_to_refute: Optional[float]
    
    def to_dict(self) -> Dict:
        return {
            'bucket_name': self.bucket_name,
            'bucket_value': self.bucket_value,
            'total_predictions': self.total_predictions,
            'testable_count': self.testable_count,
            'confirmed_count': self.confirmed_count,
            'refuted_count': self.refuted_count,
            'invalidated_count': self.invalidated_count,
            'timeout_count': self.timeout_count,
            'crr': self.crr,
            'invalidation_rate': self.invalidation_rate,
            'blocks_with_data': self.blocks_with_data,
            'blocks_supported': self.blocks_supported,
            'verdict_persistence': self.verdict_persistence,
            'mean_bars_to_confirm': self.mean_bars_to_confirm,
            'mean_bars_to_refute': self.mean_bars_to_refute,
        }


def compute_bucket_metrics(
    bucket_name: str,
    bucket_value: str,
    predictions: List[PredictionObservables],
    crr_threshold: float = 0.55,
    min_testable_for_verdict: int = 5,
) -> BucketMetrics:
    """Compute metrics for a single bucket of predictions."""
    if not predictions:
        return BucketMetrics(
            bucket_name=bucket_name,
            bucket_value=bucket_value,
            total_predictions=0,
            testable_count=0,
            confirmed_count=0,
            refuted_count=0,
            invalidated_count=0,
            timeout_count=0,
            crr=0.0,
            invalidation_rate=0.0,
            blocks_with_data=0,
            blocks_supported=0,
            verdict_persistence=0.0,
            mean_bars_to_confirm=None,
            mean_bars_to_refute=None,
        )
    
    total = len(predictions)
    confirmed = [p for p in predictions if p.outcome == 'CONFIRMED']
    refuted = [p for p in predictions if p.outcome == 'REFUTED']
    invalidated = [p for p in predictions if p.outcome == 'INVALIDATED']
    timeout = [p for p in predictions if p.outcome == 'TIMEOUT']
    testable = confirmed + refuted
    
    n_confirmed = len(confirmed)
    n_refuted = len(refuted)
    n_invalidated = len(invalidated)
    n_timeout = len(timeout)
    n_testable = len(testable)
    
    crr = n_confirmed / n_testable if n_testable > 0 else 0.0
    inv_rate = n_invalidated / total if total > 0 else 0.0
    
    # For verdict persistence, we'd need block information
    # Simplified: consider this bucket as "supported" if CRR >= threshold
    blocks_with_data = 1 if n_testable >= min_testable_for_verdict else 0
    blocks_supported = 1 if (blocks_with_data and crr >= crr_threshold) else 0
    verdict_persistence = blocks_supported / blocks_with_data if blocks_with_data > 0 else 0.0
    
    # Timing
    confirm_bars = [p.bars_to_resolution for p in confirmed]
    refute_bars = [p.bars_to_resolution for p in refuted]
    
    mean_confirm = sum(confirm_bars) / len(confirm_bars) if confirm_bars else None
    mean_refute = sum(refute_bars) / len(refute_bars) if refute_bars else None
    
    return BucketMetrics(
        bucket_name=bucket_name,
        bucket_value=bucket_value,
        total_predictions=total,
        testable_count=n_testable,
        confirmed_count=n_confirmed,
        refuted_count=n_refuted,
        invalidated_count=n_invalidated,
        timeout_count=n_timeout,
        crr=crr,
        invalidation_rate=inv_rate,
        blocks_with_data=blocks_with_data,
        blocks_supported=blocks_supported,
        verdict_persistence=verdict_persistence,
        mean_bars_to_confirm=mean_confirm,
        mean_bars_to_refute=mean_refute,
    )


# =============================================================================
# FAILURE SURFACE
# =============================================================================

@dataclass
class FailureSurface:
    """
    Empirically derived failure boundary.
    
    A failure surface defines: "When observable X crosses threshold T,
    the edge ALWAYS fails (CRR < threshold, or invalidation > threshold)."
    
    This is a NECESSARY condition for failure, not SUFFICIENT for success.
    """
    observable_name: str
    threshold_value: float
    threshold_direction: str  # "above" or "below"
    failure_crr: float        # CRR in failure region
    failure_inv_rate: float   # Invalidation rate in failure region
    sample_size: int          # Predictions in failure region
    confidence: float         # Empirical confidence (based on sample size)
    description: str
    
    def to_dict(self) -> Dict:
        return {
            'observable_name': self.observable_name,
            'threshold_value': self.threshold_value,
            'threshold_direction': self.threshold_direction,
            'failure_crr': self.failure_crr,
            'failure_inv_rate': self.failure_inv_rate,
            'sample_size': self.sample_size,
            'confidence': self.confidence,
            'description': self.description,
        }


def derive_failure_threshold(
    observable_values: List[float],
    outcomes: List[str],
    crr_threshold: float = 0.55,
    min_samples: int = 10,
    search_direction: str = "above",  # or "below"
) -> Optional[Tuple[float, float, float, int]]:
    """
    Empirically derive failure threshold for an observable.
    
    Finds the threshold where CRR drops below crr_threshold.
    
    Returns: (threshold, failure_crr, failure_inv_rate, sample_size) or None
    """
    if len(observable_values) < min_samples:
        return None
    
    # Pair values with outcomes
    paired = list(zip(observable_values, outcomes))
    
    # Sort by observable value
    paired.sort(key=lambda x: x[0], reverse=(search_direction == "above"))
    
    # Slide through sorted values looking for failure region
    for i in range(min_samples, len(paired)):
        if search_direction == "above":
            # Looking for threshold ABOVE which edge fails
            region = paired[:i]
            threshold = paired[i-1][0]
        else:
            # Looking for threshold BELOW which edge fails
            region = paired[-i:]
            threshold = paired[-i][0]
        
        # Compute CRR in this region
        testable = [p for p in region if p[1] in ('CONFIRMED', 'REFUTED')]
        if len(testable) < min_samples:
            continue
        
        confirmed = sum(1 for p in testable if p[1] == 'CONFIRMED')
        crr = confirmed / len(testable)
        
        # Compute invalidation rate
        invalidated = sum(1 for p in region if p[1] == 'INVALIDATED')
        inv_rate = invalidated / len(region)
        
        # Check if this is a failure region
        if crr < crr_threshold:
            return (threshold, crr, inv_rate, len(region))
    
    return None


# =============================================================================
# COLLAPSE CLASSIFICATION
# =============================================================================

class CollapseType(Enum):
    """Classification of edge collapse pattern."""
    GRADUAL_DECAY = "GRADUAL_DECAY"
    SUDDEN_REGIME_FLIP = "SUDDEN_REGIME_FLIP"
    NOISE_INDUCED_INSTABILITY = "NOISE_INDUCED_INSTABILITY"
    STRUCTURAL_INVALIDATION = "STRUCTURAL_INVALIDATION"
    NO_COLLAPSE = "NO_COLLAPSE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass
class CollapseAnalysis:
    """Analysis of collapse pattern."""
    collapse_type: CollapseType
    confidence: float
    evidence: Dict[str, Any]
    description: str
    
    def to_dict(self) -> Dict:
        return {
            'collapse_type': self.collapse_type.value,
            'confidence': self.confidence,
            'evidence': self.evidence,
            'description': self.description,
        }


def classify_collapse(
    crr_series: List[float],
    invalidation_series: List[float],
    regime_change_rate: float,
    crr_variance: float,
) -> CollapseAnalysis:
    """
    Classify the type of edge collapse.
    
    Uses pattern recognition on already-computed metrics.
    """
    if len(crr_series) < 3:
        return CollapseAnalysis(
            collapse_type=CollapseType.INSUFFICIENT_DATA,
            confidence=0.0,
            evidence={'crr_series_length': len(crr_series)},
            description="Insufficient data for collapse classification",
        )
    
    # Compute decay metrics
    crr_trend = _compute_linear_slope(crr_series)
    inv_trend = _compute_linear_slope(invalidation_series)
    crr_std = math.sqrt(crr_variance) if crr_variance > 0 else 0.0
    
    # Check for no collapse (stable edge)
    if crr_trend > -0.01 and crr_std < 0.10:
        return CollapseAnalysis(
            collapse_type=CollapseType.NO_COLLAPSE,
            confidence=min(1.0, 1 - abs(crr_trend) * 10),
            evidence={
                'crr_trend': crr_trend,
                'crr_std': crr_std,
            },
            description="Edge appears stable, no collapse detected",
        )
    
    # Check for structural invalidation (high invalidation rate increase)
    if inv_trend > 0.05:
        return CollapseAnalysis(
            collapse_type=CollapseType.STRUCTURAL_INVALIDATION,
            confidence=min(1.0, inv_trend * 10),
            evidence={
                'invalidation_trend': inv_trend,
                'final_inv_rate': invalidation_series[-1] if invalidation_series else 0,
            },
            description="Edge failing due to structural breakdown (rising invalidations)",
        )
    
    # Check for sudden regime flip
    if regime_change_rate > 0.3:  # More than 30% of blocks had regime changes
        max_crr_drop = max(
            crr_series[i] - crr_series[i+1]
            for i in range(len(crr_series) - 1)
        ) if len(crr_series) > 1 else 0
        
        if max_crr_drop > 0.15:  # Single drop > 15%
            return CollapseAnalysis(
                collapse_type=CollapseType.SUDDEN_REGIME_FLIP,
                confidence=min(1.0, max_crr_drop * 3),
                evidence={
                    'regime_change_rate': regime_change_rate,
                    'max_crr_drop': max_crr_drop,
                },
                description="Edge collapsed due to sudden regime change",
            )
    
    # Check for noise-induced instability
    if crr_std > 0.15 and abs(crr_trend) < 0.02:
        return CollapseAnalysis(
            collapse_type=CollapseType.NOISE_INDUCED_INSTABILITY,
            confidence=min(1.0, crr_std * 5),
            evidence={
                'crr_std': crr_std,
                'crr_trend': crr_trend,
            },
            description="Edge unstable due to high variance without clear trend",
        )
    
    # Default: gradual decay
    if crr_trend < -0.01:
        return CollapseAnalysis(
            collapse_type=CollapseType.GRADUAL_DECAY,
            confidence=min(1.0, abs(crr_trend) * 20),
            evidence={
                'crr_trend': crr_trend,
                'decay_rate_per_block': abs(crr_trend),
            },
            description="Edge decaying gradually over time",
        )
    
    # Unclear pattern
    return CollapseAnalysis(
        collapse_type=CollapseType.NOISE_INDUCED_INSTABILITY,
        confidence=0.3,
        evidence={
            'crr_trend': crr_trend,
            'crr_std': crr_std,
            'inv_trend': inv_trend,
        },
        description="Collapse pattern unclear, defaulting to noise-induced",
    )


def _compute_linear_slope(values: List[float]) -> float:
    """Compute linear regression slope."""
    n = len(values)
    if n < 2:
        return 0.0
    
    x = list(range(n))
    x_mean = sum(x) / n
    y_mean = sum(values) / n
    
    numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, values))
    denominator = sum((xi - x_mean) ** 2 for xi in x)
    
    return numerator / denominator if denominator > 0 else 0.0


# =============================================================================
# EDGE SAFE ZONE
# =============================================================================

@dataclass
class EdgeSafeZone:
    """
    Observable ranges where edge appears to hold.
    
    This is NOT a recommendation to trade.
    This is a documentation of where edge exists (if anywhere).
    """
    correlation_range: Tuple[float, float]  # (min, max)
    volatility_ratio_range: Tuple[float, float]
    spread_range: Tuple[float, float]  # Z-score range
    correlation_trend_requirement: str  # "any", "stable_or_improving", "stable"
    
    # Metrics in safe zone
    safe_zone_crr: float
    safe_zone_inv_rate: float
    safe_zone_sample_size: int
    
    # Fraction of total data in safe zone
    coverage_fraction: float
    
    def to_dict(self) -> Dict:
        return {
            'correlation_range': list(self.correlation_range),
            'volatility_ratio_range': list(self.volatility_ratio_range),
            'spread_range': list(self.spread_range),
            'correlation_trend_requirement': self.correlation_trend_requirement,
            'safe_zone_crr': self.safe_zone_crr,
            'safe_zone_inv_rate': self.safe_zone_inv_rate,
            'safe_zone_sample_size': self.safe_zone_sample_size,
            'coverage_fraction': self.coverage_fraction,
        }


def identify_safe_zone(
    predictions: List[PredictionObservables],
    crr_threshold: float = 0.55,
    max_inv_rate: float = 0.40,
    min_samples: int = 20,
) -> Optional[EdgeSafeZone]:
    """
    Identify the observable region where edge holds.
    
    This is derived empirically, not configured.
    """
    if len(predictions) < min_samples:
        return None
    
    # Find correlation range where CRR holds
    corr_values = [p.correlation for p in predictions]
    corr_min, corr_max = min(corr_values), max(corr_values)
    
    # Find volatility ratio range
    vol_values = [p.volatility_ratio for p in predictions]
    vol_min, vol_max = min(vol_values), max(vol_values)
    
    # Find spread range
    spread_values = [abs(p.zscore) for p in predictions]
    spread_min, spread_max = min(spread_values), max(spread_values)
    
    # Try to narrow ranges to where edge holds
    best_zone = None
    best_crr = 0.0
    
    # Grid search over correlation bins
    corr_bins = [(corr_min + i * (corr_max - corr_min) / 10,
                  corr_min + (i + 5) * (corr_max - corr_min) / 10)
                 for i in range(6)]
    
    for c_low, c_high in corr_bins:
        filtered = [p for p in predictions 
                   if c_low <= p.correlation <= c_high]
        
        if len(filtered) < min_samples:
            continue
        
        testable = [p for p in filtered if p.is_testable]
        if len(testable) < min_samples // 2:
            continue
        
        confirmed = sum(1 for p in testable if p.is_confirmed)
        crr = confirmed / len(testable)
        
        invalidated = sum(1 for p in filtered if p.is_invalidated)
        inv_rate = invalidated / len(filtered)
        
        if crr >= crr_threshold and inv_rate <= max_inv_rate and crr > best_crr:
            best_crr = crr
            best_zone = {
                'corr_range': (c_low, c_high),
                'crr': crr,
                'inv_rate': inv_rate,
                'sample_size': len(filtered),
            }
    
    if best_zone is None:
        return None
    
    # Get vol and spread ranges for the safe correlation zone
    safe_preds = [p for p in predictions 
                  if best_zone['corr_range'][0] <= p.correlation <= best_zone['corr_range'][1]]
    
    safe_vol = [p.volatility_ratio for p in safe_preds]
    safe_spread = [abs(p.zscore) for p in safe_preds]
    
    return EdgeSafeZone(
        correlation_range=best_zone['corr_range'],
        volatility_ratio_range=(min(safe_vol), max(safe_vol)),
        spread_range=(min(safe_spread), max(safe_spread)),
        correlation_trend_requirement="any",  # Simplified
        safe_zone_crr=best_zone['crr'],
        safe_zone_inv_rate=best_zone['inv_rate'],
        safe_zone_sample_size=best_zone['sample_size'],
        coverage_fraction=best_zone['sample_size'] / len(predictions),
    )


# =============================================================================
# EDGE BOUNDARY OUTPUT
# =============================================================================

EDGE_BOUNDARY_SCHEMA_VERSION = "1.0.0"


@dataclass
class EdgeBoundaryOutput:
    """
    Complete edge boundary analysis output.
    
    Single deterministic artifact documenting all failure modes.
    """
    schema_version: str
    analysis_timestamp: str
    
    # Source data summary
    total_predictions: int
    total_blocks: int
    overall_crr: float
    overall_inv_rate: float
    
    # Boundary metrics by condition
    correlation_level_metrics: List[Dict]
    correlation_trend_metrics: List[Dict]
    volatility_ratio_metrics: List[Dict]
    spread_state_metrics: List[Dict]
    regime_transition_metrics: List[Dict]
    
    # Failure surfaces
    failure_surfaces: List[Dict]
    
    # Necessary conditions for failure
    necessary_failure_conditions: Dict[str, Any]
    
    # Non-failure regions
    non_failure_regions: Dict[str, Any]
    
    # Edge safe zone
    edge_safe_zone: Optional[Dict]
    
    # Collapse classification
    collapse_analysis: Dict
    
    # Summary flags
    flags: Dict[str, bool]
    
    def to_dict(self) -> Dict:
        return {
            'schema_version': self.schema_version,
            'analysis_timestamp': self.analysis_timestamp,
            'total_predictions': self.total_predictions,
            'total_blocks': self.total_blocks,
            'overall_crr': self.overall_crr,
            'overall_inv_rate': self.overall_inv_rate,
            'correlation_level_metrics': self.correlation_level_metrics,
            'correlation_trend_metrics': self.correlation_trend_metrics,
            'volatility_ratio_metrics': self.volatility_ratio_metrics,
            'spread_state_metrics': self.spread_state_metrics,
            'regime_transition_metrics': self.regime_transition_metrics,
            'failure_surfaces': self.failure_surfaces,
            'necessary_failure_conditions': self.necessary_failure_conditions,
            'non_failure_regions': self.non_failure_regions,
            'edge_safe_zone': self.edge_safe_zone,
            'collapse_analysis': self.collapse_analysis,
            'flags': self.flags,
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
