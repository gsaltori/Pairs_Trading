"""
Walk-Forward Falsification Engine

Evaluates temporal persistence vs decay of CRV hypotheses by running
the existing experiment pipeline across sequential, non-overlapping time blocks.

DESIGN PRINCIPLES:
1. Each block is treated as out-of-sample
2. No leakage between blocks (fresh state per block)
3. Regime memory RESET per block (not persisted)
4. Uses existing experiment infrastructure unchanged
5. Produces machine-readable stability analysis

THIS IS NOT:
- Parameter optimization
- Strategy improvement
- Trading system development

THIS IS:
- Scientific falsification of temporal stability claims
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
from datetime import datetime
import json
import hashlib
import math
from pathlib import Path


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class WalkForwardConfig:
    """
    Immutable configuration for walk-forward analysis.
    
    BLOCK STRATEGY: Fixed-bar, non-overlapping blocks.
    
    Parameters:
        block_size: Number of bars per block (must accommodate warm-up + predictions)
        min_bars_per_block: Minimum bars required for valid block (after warm-up)
        warmup_bars: Bars consumed by spread/correlation initialization
        min_testable_per_block: Minimum testable predictions for valid verdict
        min_blocks_for_stability: Minimum blocks required for stability analysis
    """
    block_size: int = 500
    min_bars_per_block: int = 300  # After warmup, need at least this many
    warmup_bars: int = 60          # Correlation/spread warmup requirement
    min_testable_per_block: int = 10  # Need at least this many testable per block
    min_blocks_for_stability: int = 3  # Need at least 3 blocks for trend analysis
    
    def __post_init__(self):
        if self.block_size < self.warmup_bars + self.min_bars_per_block:
            raise ValueError(
                f"block_size ({self.block_size}) must be >= "
                f"warmup_bars ({self.warmup_bars}) + min_bars_per_block ({self.min_bars_per_block})"
            )
        if self.min_testable_per_block < 5:
            raise ValueError("min_testable_per_block must be >= 5 for statistical validity")
    
    def compute_blocks(self, total_bars: int) -> List[Tuple[int, int]]:
        """
        Compute non-overlapping block boundaries.
        
        Returns:
            List of (start_bar, end_bar) tuples, exclusive end
        """
        blocks = []
        start = 0
        while start + self.block_size <= total_bars:
            blocks.append((start, start + self.block_size))
            start += self.block_size
        
        # Handle remainder: only include if meets minimum
        if start < total_bars:
            remaining = total_bars - start
            if remaining >= self.warmup_bars + self.min_bars_per_block:
                blocks.append((start, total_bars))
        
        return blocks
    
    @property
    def config_hash(self) -> str:
        """Deterministic hash for reproducibility."""
        config_str = json.dumps({
            'block_size': self.block_size,
            'min_bars_per_block': self.min_bars_per_block,
            'warmup_bars': self.warmup_bars,
            'min_testable_per_block': self.min_testable_per_block,
        }, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:12]


# =============================================================================
# BLOCK RESULT
# =============================================================================

class BlockVerdict(Enum):
    """Verdict for a single time block."""
    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass
class RegimeDistribution:
    """Distribution of predictions across regime signatures."""
    regime_counts: Dict[str, int]  # regime_code -> count
    dominant_regime: Optional[str]
    regime_entropy: float  # Higher = more diverse
    
    def to_dict(self) -> Dict:
        return {
            'regime_counts': self.regime_counts,
            'dominant_regime': self.dominant_regime,
            'regime_entropy': self.regime_entropy,
        }


@dataclass
class BlockResult:
    """
    Complete result from a single time block.
    
    Contains all metrics needed for temporal stability analysis.
    """
    # Block identification
    block_index: int
    start_bar: int
    end_bar: int
    bar_count: int
    
    # Primary metrics
    total_predictions: int
    testable_count: int
    confirmed_count: int
    refuted_count: int
    invalidated_count: int
    timeout_count: int
    
    crr: float  # Confirmation Rate Ratio
    erv: float  # Expected Resolution Value (total)
    erv_per_prediction: float
    invalidation_rate: float
    
    # Timing metrics
    median_bars_to_confirmation: Optional[float]
    median_bars_to_refutation: Optional[float]
    timing_asymmetry_ratio: Optional[float]
    
    # Regime distribution (if regime tracking enabled)
    regime_distribution: Optional[RegimeDistribution]
    
    # Verdict
    verdict: BlockVerdict
    verdict_reason: str
    
    # Hypothesis thresholds used
    crr_threshold: float
    max_invalidation_threshold: float
    
    def to_dict(self) -> Dict:
        return {
            'block_index': self.block_index,
            'start_bar': self.start_bar,
            'end_bar': self.end_bar,
            'bar_count': self.bar_count,
            'total_predictions': self.total_predictions,
            'testable_count': self.testable_count,
            'confirmed_count': self.confirmed_count,
            'refuted_count': self.refuted_count,
            'invalidated_count': self.invalidated_count,
            'timeout_count': self.timeout_count,
            'crr': self.crr,
            'erv': self.erv,
            'erv_per_prediction': self.erv_per_prediction,
            'invalidation_rate': self.invalidation_rate,
            'median_bars_to_confirmation': self.median_bars_to_confirmation,
            'median_bars_to_refutation': self.median_bars_to_refutation,
            'timing_asymmetry_ratio': self.timing_asymmetry_ratio,
            'regime_distribution': self.regime_distribution.to_dict() if self.regime_distribution else None,
            'verdict': self.verdict.value,
            'verdict_reason': self.verdict_reason,
            'crr_threshold': self.crr_threshold,
            'max_invalidation_threshold': self.max_invalidation_threshold,
        }
    
    @property
    def is_sufficient(self) -> bool:
        """Does this block have sufficient data for analysis?"""
        return self.verdict != BlockVerdict.INSUFFICIENT_DATA


# =============================================================================
# TEMPORAL STABILITY METRICS
# =============================================================================

class EdgeStabilityClass(Enum):
    """Classification of edge temporal behavior."""
    PERSISTENT = "PERSISTENT"           # Edge holds across blocks
    DECAYING = "DECAYING"               # Edge degrades monotonically
    REGIME_DEPENDENT = "REGIME_DEPENDENT"  # Edge varies with regime distribution
    UNSTABLE = "UNSTABLE"               # No consistent pattern
    UNFALSIFIABLE = "UNFALSIFIABLE"     # Insufficient data to classify


@dataclass
class TemporalStabilityMetrics:
    """
    Metrics quantifying temporal persistence of the hypothesis.
    
    ALL metrics are explicitly defined and computable.
    """
    # Sample counts
    total_blocks: int
    sufficient_blocks: int  # Blocks with enough data
    
    # CRR Drift Analysis
    # Linear regression: CRR = a + b * block_index
    # crr_drift_slope = b (negative = decay)
    crr_values: List[float]
    crr_drift_slope: Optional[float]
    crr_drift_intercept: Optional[float]
    crr_drift_r_squared: Optional[float]  # Goodness of fit
    
    # ERV Drift Analysis
    erv_values: List[float]
    erv_drift_slope: Optional[float]
    erv_drift_intercept: Optional[float]
    erv_drift_r_squared: Optional[float]
    
    # Verdict Persistence
    # = (blocks with SUPPORTED) / (blocks with sufficient data)
    supported_count: int
    refuted_count: int
    verdict_persistence_ratio: float
    
    # Structural Decay Indicator
    # Detects if CRR is monotonically decreasing
    # Uses runs test: count direction changes
    crr_direction_changes: int  # Number of times CRR changes direction
    is_monotonic_decay: bool    # True if CRR never increases
    decay_consistency: float    # 1.0 = pure decay, 0.0 = random
    
    # Invalidation trend
    invalidation_rates: List[float]
    invalidation_trend_slope: Optional[float]
    
    # Classification
    stability_class: EdgeStabilityClass
    classification_confidence: float  # 0.0 to 1.0
    classification_reason: str
    
    def to_dict(self) -> Dict:
        return {
            'total_blocks': self.total_blocks,
            'sufficient_blocks': self.sufficient_blocks,
            'crr_values': self.crr_values,
            'crr_drift_slope': self.crr_drift_slope,
            'crr_drift_intercept': self.crr_drift_intercept,
            'crr_drift_r_squared': self.crr_drift_r_squared,
            'erv_values': self.erv_values,
            'erv_drift_slope': self.erv_drift_slope,
            'erv_drift_intercept': self.erv_drift_intercept,
            'erv_drift_r_squared': self.erv_drift_r_squared,
            'supported_count': self.supported_count,
            'refuted_count': self.refuted_count,
            'verdict_persistence_ratio': self.verdict_persistence_ratio,
            'crr_direction_changes': self.crr_direction_changes,
            'is_monotonic_decay': self.is_monotonic_decay,
            'decay_consistency': self.decay_consistency,
            'invalidation_rates': self.invalidation_rates,
            'invalidation_trend_slope': self.invalidation_trend_slope,
            'stability_class': self.stability_class.value,
            'classification_confidence': self.classification_confidence,
            'classification_reason': self.classification_reason,
        }


def _linear_regression(x: List[float], y: List[float]) -> Tuple[float, float, float]:
    """
    Simple linear regression: y = a + b*x
    
    Returns: (intercept, slope, r_squared)
    """
    n = len(x)
    if n < 2:
        return (0.0, 0.0, 0.0)
    
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))
    sum_x2 = sum(xi ** 2 for xi in x)
    sum_y2 = sum(yi ** 2 for yi in y)
    
    denom = n * sum_x2 - sum_x ** 2
    if abs(denom) < 1e-10:
        return (sum_y / n if n > 0 else 0.0, 0.0, 0.0)
    
    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n
    
    # R-squared
    y_mean = sum_y / n
    ss_tot = sum((yi - y_mean) ** 2 for yi in y)
    ss_res = sum((yi - (intercept + slope * xi)) ** 2 for xi, yi in zip(x, y))
    
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0.0
    r_squared = max(0.0, min(1.0, r_squared))  # Clamp to [0, 1]
    
    return (intercept, slope, r_squared)


def _count_direction_changes(values: List[float]) -> int:
    """Count number of times the series changes direction."""
    if len(values) < 3:
        return 0
    
    changes = 0
    prev_direction = None
    
    for i in range(1, len(values)):
        diff = values[i] - values[i-1]
        if abs(diff) < 1e-10:
            continue
        
        direction = 1 if diff > 0 else -1
        
        if prev_direction is not None and direction != prev_direction:
            changes += 1
        
        prev_direction = direction
    
    return changes


def _is_monotonic_decreasing(values: List[float], tolerance: float = 0.001) -> bool:
    """Check if values are monotonically decreasing (with small tolerance)."""
    if len(values) < 2:
        return False
    
    for i in range(1, len(values)):
        if values[i] > values[i-1] + tolerance:
            return False
    
    return True


def _compute_decay_consistency(values: List[float]) -> float:
    """
    Compute decay consistency score.
    
    1.0 = pure monotonic decay
    0.0 = completely random
    
    Uses ratio of decreases to total comparisons.
    """
    if len(values) < 2:
        return 0.0
    
    decreases = 0
    total = 0
    
    for i in range(1, len(values)):
        total += 1
        if values[i] < values[i-1]:
            decreases += 1
    
    return decreases / total if total > 0 else 0.0


def compute_temporal_stability(
    block_results: List[BlockResult],
    config: WalkForwardConfig,
) -> TemporalStabilityMetrics:
    """
    Compute temporal stability metrics from block results.
    
    This is the core analysis function that determines if edge persists.
    """
    total_blocks = len(block_results)
    sufficient_blocks = [b for b in block_results if b.is_sufficient]
    n_sufficient = len(sufficient_blocks)
    
    # Extract metric series from sufficient blocks
    crr_values = [b.crr for b in sufficient_blocks]
    erv_values = [b.erv_per_prediction for b in sufficient_blocks]
    inv_rates = [b.invalidation_rate for b in sufficient_blocks]
    
    # Count verdicts
    supported_count = sum(1 for b in sufficient_blocks if b.verdict == BlockVerdict.SUPPORTED)
    refuted_count = sum(1 for b in sufficient_blocks if b.verdict == BlockVerdict.REFUTED)
    
    # Verdict persistence ratio
    verdict_persistence = supported_count / n_sufficient if n_sufficient > 0 else 0.0
    
    # Check if we have enough data for stability analysis
    if n_sufficient < config.min_blocks_for_stability:
        return TemporalStabilityMetrics(
            total_blocks=total_blocks,
            sufficient_blocks=n_sufficient,
            crr_values=crr_values,
            crr_drift_slope=None,
            crr_drift_intercept=None,
            crr_drift_r_squared=None,
            erv_values=erv_values,
            erv_drift_slope=None,
            erv_drift_intercept=None,
            erv_drift_r_squared=None,
            supported_count=supported_count,
            refuted_count=refuted_count,
            verdict_persistence_ratio=verdict_persistence,
            crr_direction_changes=0,
            is_monotonic_decay=False,
            decay_consistency=0.0,
            invalidation_rates=inv_rates,
            invalidation_trend_slope=None,
            stability_class=EdgeStabilityClass.UNFALSIFIABLE,
            classification_confidence=0.0,
            classification_reason=f"Insufficient blocks: {n_sufficient} < {config.min_blocks_for_stability}",
        )
    
    # Linear regression analysis
    block_indices = list(range(n_sufficient))
    
    crr_intercept, crr_slope, crr_r2 = _linear_regression(block_indices, crr_values)
    erv_intercept, erv_slope, erv_r2 = _linear_regression(block_indices, erv_values)
    inv_intercept, inv_slope, inv_r2 = _linear_regression(block_indices, inv_rates)
    
    # Direction change analysis
    crr_changes = _count_direction_changes(crr_values)
    is_mono_decay = _is_monotonic_decreasing(crr_values)
    decay_cons = _compute_decay_consistency(crr_values)
    
    # Classification logic
    stability_class, confidence, reason = _classify_stability(
        crr_values=crr_values,
        crr_slope=crr_slope,
        crr_r2=crr_r2,
        verdict_persistence=verdict_persistence,
        is_mono_decay=is_mono_decay,
        decay_consistency=decay_cons,
        n_blocks=n_sufficient,
    )
    
    return TemporalStabilityMetrics(
        total_blocks=total_blocks,
        sufficient_blocks=n_sufficient,
        crr_values=crr_values,
        crr_drift_slope=crr_slope,
        crr_drift_intercept=crr_intercept,
        crr_drift_r_squared=crr_r2,
        erv_values=erv_values,
        erv_drift_slope=erv_slope,
        erv_drift_intercept=erv_intercept,
        erv_drift_r_squared=erv_r2,
        supported_count=supported_count,
        refuted_count=refuted_count,
        verdict_persistence_ratio=verdict_persistence,
        crr_direction_changes=crr_changes,
        is_monotonic_decay=is_mono_decay,
        decay_consistency=decay_cons,
        invalidation_rates=inv_rates,
        invalidation_trend_slope=inv_slope,
        stability_class=stability_class,
        classification_confidence=confidence,
        classification_reason=reason,
    )


def _classify_stability(
    crr_values: List[float],
    crr_slope: float,
    crr_r2: float,
    verdict_persistence: float,
    is_mono_decay: bool,
    decay_consistency: float,
    n_blocks: int,
) -> Tuple[EdgeStabilityClass, float, str]:
    """
    Classify edge stability based on temporal metrics.
    
    Returns: (class, confidence, reason)
    """
    # Thresholds (FIXED, not tunable)
    PERSISTENCE_THRESHOLD = 0.7      # 70% of blocks must be SUPPORTED
    DECAY_SLOPE_THRESHOLD = -0.02    # CRR drops >2% per block
    DECAY_R2_THRESHOLD = 0.5         # Trend must explain 50% of variance
    VARIANCE_THRESHOLD = 0.15        # CRR std dev for instability
    
    crr_mean = sum(crr_values) / len(crr_values) if crr_values else 0.0
    crr_std = math.sqrt(sum((x - crr_mean)**2 for x in crr_values) / len(crr_values)) if len(crr_values) > 1 else 0.0
    
    # Decision tree (explicit, not ML)
    
    # 1. Check for persistent edge
    if verdict_persistence >= PERSISTENCE_THRESHOLD and crr_slope > DECAY_SLOPE_THRESHOLD:
        confidence = min(1.0, verdict_persistence * (1 - abs(crr_slope) / 0.1))
        return (
            EdgeStabilityClass.PERSISTENT,
            confidence,
            f"Verdict persistence {verdict_persistence:.0%} >= {PERSISTENCE_THRESHOLD:.0%}, "
            f"CRR slope {crr_slope:.4f} > {DECAY_SLOPE_THRESHOLD}"
        )
    
    # 2. Check for monotonic decay
    if is_mono_decay and crr_slope < DECAY_SLOPE_THRESHOLD:
        confidence = min(1.0, decay_consistency * crr_r2)
        return (
            EdgeStabilityClass.DECAYING,
            confidence,
            f"Monotonic CRR decay detected, slope={crr_slope:.4f}, R²={crr_r2:.2f}"
        )
    
    # 3. Check for significant decay trend (not necessarily monotonic)
    if crr_slope < DECAY_SLOPE_THRESHOLD and crr_r2 > DECAY_R2_THRESHOLD:
        confidence = crr_r2
        return (
            EdgeStabilityClass.DECAYING,
            confidence,
            f"Significant decay trend, slope={crr_slope:.4f}, R²={crr_r2:.2f}"
        )
    
    # 4. Check for high variance (unstable)
    if crr_std > VARIANCE_THRESHOLD:
        confidence = min(1.0, crr_std / 0.3)
        return (
            EdgeStabilityClass.UNSTABLE,
            confidence,
            f"High CRR variance: std={crr_std:.3f} > {VARIANCE_THRESHOLD}"
        )
    
    # 5. Check for regime dependence (varying verdicts without clear trend)
    if verdict_persistence < PERSISTENCE_THRESHOLD and abs(crr_slope) < abs(DECAY_SLOPE_THRESHOLD):
        confidence = 1 - verdict_persistence
        return (
            EdgeStabilityClass.REGIME_DEPENDENT,
            confidence,
            f"Variable verdicts ({verdict_persistence:.0%} supported) without decay trend"
        )
    
    # 6. Default: unstable
    return (
        EdgeStabilityClass.UNSTABLE,
        0.5,
        "No clear stability pattern detected"
    )


# =============================================================================
# WALK-FORWARD OUTPUT
# =============================================================================

@dataclass
class WalkForwardOutput:
    """
    Complete output from walk-forward analysis.
    
    Machine-readable, deterministic, and reproducible.
    """
    # Metadata
    schema_version: str
    run_timestamp: str
    config_hash: str
    
    # Configuration used
    walk_forward_config: Dict[str, Any]
    experiment_config: Dict[str, Any]
    hypothesis_config: Dict[str, Any]
    
    # Per-block results
    block_results: List[Dict[str, Any]]
    
    # Temporal stability analysis
    stability_metrics: Dict[str, Any]
    
    # Summary flags
    flags: Dict[str, bool]
    
    # Warnings and errors
    warnings: List[str]
    errors: List[str]
    
    def to_dict(self) -> Dict:
        return {
            'schema_version': self.schema_version,
            'run_timestamp': self.run_timestamp,
            'config_hash': self.config_hash,
            'walk_forward_config': self.walk_forward_config,
            'experiment_config': self.experiment_config,
            'hypothesis_config': self.hypothesis_config,
            'block_results': self.block_results,
            'stability_metrics': self.stability_metrics,
            'flags': self.flags,
            'warnings': self.warnings,
            'errors': self.errors,
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
    
    def save(self, output_dir: Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = self.run_timestamp.replace(':', '-').replace('.', '-')
        filename = f"walkforward_{self.config_hash}_{timestamp}.json"
        filepath = output_dir / filename
        
        with open(filepath, 'w') as f:
            f.write(self.to_json())
        
        return filepath


WALK_FORWARD_SCHEMA_VERSION = "1.0.0"
