"""
P5 Regime Memory Module

This module observes historical P4 states and P1 outcomes to learn which
structural contexts tend to produce CONFIRMED vs REFUTED outcomes.

DESIGN PRINCIPLES:
1. Pattern recognition, NOT prediction
2. Forward-only (no lookahead bias)
3. Bounded memory (FIFO eviction)
4. Discretization for generalization
5. Conservative confidence estimation
6. Cold start permissive (allow when uncertain)

REGIME DIMENSIONS:
1. Correlation stability (STABLE / MODERATE / UNSTABLE)
2. Correlation trend (DECLINING / NEUTRAL / IMPROVING)
3. Volatility stability (STABLE / MODERATE / UNSTABLE)
4. Spread dynamics (CONTRACTING / NORMAL / EXPANDING)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple, Dict
from enum import Enum
from collections import deque
import math

try:
    from .config import CONFIG
except ImportError:
    from config import CONFIG


# =============================================================================
# REGIME DISCRETIZATION ENUMS
# =============================================================================

class CorrelationStability(Enum):
    """Discretized correlation stability level."""
    STABLE = "STABLE"           # Low variance in correlation
    MODERATE = "MODERATE"       # Medium variance
    UNSTABLE = "UNSTABLE"       # High variance


class CorrelationTrend(Enum):
    """Discretized correlation trend direction."""
    DECLINING = "DECLINING"     # Negative slope
    NEUTRAL = "NEUTRAL"         # Near-zero slope
    IMPROVING = "IMPROVING"     # Positive slope


class VolatilityStability(Enum):
    """Discretized volatility ratio stability."""
    STABLE = "STABLE"           # Consistent vol ratio
    MODERATE = "MODERATE"       # Some variation
    UNSTABLE = "UNSTABLE"       # High variation


class SpreadDynamics(Enum):
    """Discretized spread variance dynamics."""
    CONTRACTING = "CONTRACTING"  # Variance decreasing
    NORMAL = "NORMAL"            # Variance stable
    EXPANDING = "EXPANDING"      # Variance increasing


class OutcomeType(Enum):
    """Simplified outcome categories for regime learning."""
    CONFIRMED = "CONFIRMED"      # Prediction was correct
    REFUTED = "REFUTED"          # Prediction was wrong
    UNTESTABLE = "UNTESTABLE"    # INVALIDATED or TIMEOUT


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class RegimeConfig:
    """
    Fixed configuration for regime memory.
    
    ALL VALUES ARE FIXED. NO TUNING PERMITTED.
    """
    # ═══════════════════════════════════════════════════════════════════════════
    # DISCRETIZATION THRESHOLDS
    # ═══════════════════════════════════════════════════════════════════════════
    # Correlation stability bins (based on std of rolling correlations)
    CORR_STABILITY_LOW: float = 0.06    # Below = STABLE
    CORR_STABILITY_HIGH: float = 0.12   # Above = UNSTABLE
    
    # Correlation trend bins (slope per bar)
    CORR_TREND_DECLINING: float = -0.004  # Below = DECLINING
    CORR_TREND_IMPROVING: float = 0.004   # Above = IMPROVING
    
    # Volatility ratio stability bins
    VOL_STABILITY_LOW: float = 0.08    # Below = STABLE
    VOL_STABILITY_HIGH: float = 0.20   # Above = UNSTABLE
    
    # Spread variance ratio bins
    SPREAD_CONTRACTING: float = 0.8    # Below = CONTRACTING
    SPREAD_EXPANDING: float = 1.5      # Above = EXPANDING
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MEMORY PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════════
    MAX_MEMORY_SIZE: int = 500         # Maximum outcomes to remember
    MIN_SAMPLES_FOR_CONFIDENCE: int = 10  # Minimum N before applying filter
    
    # ═══════════════════════════════════════════════════════════════════════════
    # GATING THRESHOLDS
    # ═══════════════════════════════════════════════════════════════════════════
    MIN_CONFIDENCE_THRESHOLD: float = 0.40  # Block if confidence below this
    WILSON_Z: float = 1.96                  # 95% confidence interval


REGIME_CONFIG = RegimeConfig()


# =============================================================================
# REGIME SIGNATURE
# =============================================================================

@dataclass(frozen=True)
class RegimeSignature:
    """
    Immutable signature capturing the structural regime at prediction creation.
    
    This is a DISCRETIZED representation of the continuous P4 metrics.
    Discretization allows matching across similar (not identical) contexts.
    
    CRITICAL: Contains NO prices, NO future data.
    """
    correlation_stability: CorrelationStability
    correlation_trend: CorrelationTrend
    volatility_stability: VolatilityStability
    spread_dynamics: SpreadDynamics
    
    def __hash__(self):
        return hash((
            self.correlation_stability,
            self.correlation_trend,
            self.volatility_stability,
            self.spread_dynamics,
        ))
    
    def __eq__(self, other):
        if not isinstance(other, RegimeSignature):
            return False
        return (
            self.correlation_stability == other.correlation_stability and
            self.correlation_trend == other.correlation_trend and
            self.volatility_stability == other.volatility_stability and
            self.spread_dynamics == other.spread_dynamics
        )
    
    def to_tuple(self) -> Tuple[str, str, str, str]:
        """Convert to tuple for display."""
        return (
            self.correlation_stability.value,
            self.correlation_trend.value,
            self.volatility_stability.value,
            self.spread_dynamics.value,
        )
    
    def short_code(self) -> str:
        """Short code for logging."""
        cs = {'STABLE': 'S', 'MODERATE': 'M', 'UNSTABLE': 'U'}[self.correlation_stability.value]
        ct = {'DECLINING': 'D', 'NEUTRAL': 'N', 'IMPROVING': 'I'}[self.correlation_trend.value]
        vs = {'STABLE': 'S', 'MODERATE': 'M', 'UNSTABLE': 'U'}[self.volatility_stability.value]
        sd = {'CONTRACTING': 'C', 'NORMAL': 'N', 'EXPANDING': 'E'}[self.spread_dynamics.value]
        return f"{cs}{ct}{vs}{sd}"


def create_regime_signature(
    correlation_stability: float,
    correlation_trend: float,
    volatility_ratio_stability: float,
    spread_variance_ratio: float,
    config: RegimeConfig = REGIME_CONFIG,
) -> RegimeSignature:
    """
    Create a RegimeSignature by discretizing continuous P4 metrics.
    
    Args:
        correlation_stability: Std of rolling correlations (from P4)
        correlation_trend: Slope of correlation trend (from P4)
        volatility_ratio_stability: Std of vol ratio (from P4)
        spread_variance_ratio: Current/baseline spread variance (from P4)
        config: Discretization thresholds
    
    Returns:
        Discretized RegimeSignature
    """
    # Discretize correlation stability
    if correlation_stability < config.CORR_STABILITY_LOW:
        cs = CorrelationStability.STABLE
    elif correlation_stability < config.CORR_STABILITY_HIGH:
        cs = CorrelationStability.MODERATE
    else:
        cs = CorrelationStability.UNSTABLE
    
    # Discretize correlation trend
    if correlation_trend < config.CORR_TREND_DECLINING:
        ct = CorrelationTrend.DECLINING
    elif correlation_trend > config.CORR_TREND_IMPROVING:
        ct = CorrelationTrend.IMPROVING
    else:
        ct = CorrelationTrend.NEUTRAL
    
    # Discretize volatility stability
    if volatility_ratio_stability < config.VOL_STABILITY_LOW:
        vs = VolatilityStability.STABLE
    elif volatility_ratio_stability < config.VOL_STABILITY_HIGH:
        vs = VolatilityStability.MODERATE
    else:
        vs = VolatilityStability.UNSTABLE
    
    # Discretize spread dynamics
    if spread_variance_ratio < config.SPREAD_CONTRACTING:
        sd = SpreadDynamics.CONTRACTING
    elif spread_variance_ratio > config.SPREAD_EXPANDING:
        sd = SpreadDynamics.EXPANDING
    else:
        sd = SpreadDynamics.NORMAL
    
    return RegimeSignature(
        correlation_stability=cs,
        correlation_trend=ct,
        volatility_stability=vs,
        spread_dynamics=sd,
    )


# =============================================================================
# REGIME OUTCOME RECORD
# =============================================================================

@dataclass(frozen=True)
class RegimeOutcome:
    """
    Record of a prediction outcome in a specific regime.
    
    IMMUTABLE: Captures the regime at creation and outcome at resolution.
    """
    prediction_id: str
    creation_timestamp: datetime
    resolution_timestamp: datetime
    regime: RegimeSignature
    outcome: OutcomeType
    bars_to_resolution: int


# =============================================================================
# REGIME STATISTICS
# =============================================================================

@dataclass
class RegimeStats:
    """
    Statistics for a specific regime.
    """
    regime: RegimeSignature
    total_count: int = 0
    confirmed_count: int = 0
    refuted_count: int = 0
    untestable_count: int = 0
    
    @property
    def testable_count(self) -> int:
        """Number of testable outcomes (CONFIRMED + REFUTED)."""
        return self.confirmed_count + self.refuted_count
    
    @property
    def raw_confirmation_rate(self) -> float:
        """Raw confirmation rate (no confidence adjustment)."""
        if self.testable_count == 0:
            return 0.5  # No evidence, assume null
        return self.confirmed_count / self.testable_count
    
    def wilson_confidence_lower(self, z: float = REGIME_CONFIG.WILSON_Z) -> float:
        """
        Wilson score lower bound for confirmation rate.
        
        This provides a conservative estimate that naturally
        penalizes low sample sizes.
        
        Args:
            z: Z-score for confidence interval (1.96 = 95%)
        
        Returns:
            Lower bound of confidence interval
        """
        n = self.testable_count
        if n == 0:
            return 0.0
        
        p = self.raw_confirmation_rate
        
        # Wilson score interval formula
        denominator = 1 + z * z / n
        center = p + z * z / (2 * n)
        spread = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
        
        lower = (center - spread) / denominator
        return max(0.0, lower)
    
    @property
    def confidence_score(self) -> float:
        """
        Confidence score for this regime.
        
        Uses Wilson lower bound, which naturally penalizes low N.
        Returns 0.5 if insufficient samples (cold start).
        """
        if self.testable_count < REGIME_CONFIG.MIN_SAMPLES_FOR_CONFIDENCE:
            return 0.5  # Insufficient data, return neutral
        
        return self.wilson_confidence_lower()
    
    @property
    def has_sufficient_history(self) -> bool:
        """Whether we have enough samples for confident gating."""
        return self.testable_count >= REGIME_CONFIG.MIN_SAMPLES_FOR_CONFIDENCE


# =============================================================================
# REGIME MEMORY
# =============================================================================

class RegimeMemory:
    """
    Rolling, bounded memory of regime outcomes.
    
    FIFO eviction when full - oldest observations naturally age out.
    No explicit decay parameters.
    """
    
    def __init__(self, max_size: int = REGIME_CONFIG.MAX_MEMORY_SIZE):
        self.max_size = max_size
        self._outcomes: deque = deque(maxlen=max_size)
        self._stats_cache: Dict[RegimeSignature, RegimeStats] = {}
        self._cache_valid: bool = False
    
    def record(self, outcome: RegimeOutcome) -> None:
        """
        Record a new outcome.
        
        Args:
            outcome: The regime outcome to record
        """
        self._outcomes.append(outcome)
        self._cache_valid = False  # Invalidate cache
    
    def _rebuild_cache(self) -> None:
        """Rebuild statistics cache from outcomes."""
        self._stats_cache = {}
        
        for outcome in self._outcomes:
            regime = outcome.regime
            
            if regime not in self._stats_cache:
                self._stats_cache[regime] = RegimeStats(regime=regime)
            
            stats = self._stats_cache[regime]
            stats.total_count += 1
            
            if outcome.outcome == OutcomeType.CONFIRMED:
                stats.confirmed_count += 1
            elif outcome.outcome == OutcomeType.REFUTED:
                stats.refuted_count += 1
            else:
                stats.untestable_count += 1
        
        self._cache_valid = True
    
    def get_stats(self, regime: RegimeSignature) -> RegimeStats:
        """
        Get statistics for a specific regime.
        
        Args:
            regime: The regime signature to query
        
        Returns:
            RegimeStats for the regime (empty if no history)
        """
        if not self._cache_valid:
            self._rebuild_cache()
        
        if regime in self._stats_cache:
            return self._stats_cache[regime]
        
        return RegimeStats(regime=regime)
    
    def get_all_stats(self) -> Dict[RegimeSignature, RegimeStats]:
        """Get statistics for all observed regimes."""
        if not self._cache_valid:
            self._rebuild_cache()
        
        return self._stats_cache.copy()
    
    @property
    def total_outcomes(self) -> int:
        """Total number of outcomes in memory."""
        return len(self._outcomes)
    
    @property
    def unique_regimes(self) -> int:
        """Number of unique regimes observed."""
        if not self._cache_valid:
            self._rebuild_cache()
        return len(self._stats_cache)
    
    def clear(self) -> None:
        """Clear all memory."""
        self._outcomes.clear()
        self._stats_cache.clear()
        self._cache_valid = True


# =============================================================================
# REGIME EVALUATOR
# =============================================================================

@dataclass
class RegimeEvaluation:
    """
    Result of evaluating a regime for P1 generation.
    """
    regime: RegimeSignature
    is_allowed: bool
    confidence_score: float
    sample_size: int
    raw_confirmation_rate: float
    reason: str
    
    def summary(self) -> str:
        """Human-readable summary."""
        status = "ALLOWED" if self.is_allowed else "BLOCKED"
        return (
            f"Regime {self.regime.short_code()}: {status}\n"
            f"  Confidence: {self.confidence_score:.1%}\n"
            f"  Raw CRR: {self.raw_confirmation_rate:.1%}\n"
            f"  Samples: {self.sample_size}\n"
            f"  Reason: {self.reason}"
        )


class RegimeEvaluator:
    """
    Evaluates whether a regime should allow P1 generation.
    
    GATING LOGIC:
    1. If insufficient history (N < min): ALLOW (cold start)
    2. If confidence >= threshold: ALLOW
    3. If confidence < threshold: BLOCK
    """
    
    def __init__(
        self,
        memory: RegimeMemory,
        config: RegimeConfig = REGIME_CONFIG,
    ):
        self.memory = memory
        self.config = config
    
    def evaluate(self, regime: RegimeSignature) -> RegimeEvaluation:
        """
        Evaluate whether to allow P1 generation in this regime.
        
        Args:
            regime: The current regime signature
        
        Returns:
            RegimeEvaluation with decision and reasoning
        """
        stats = self.memory.get_stats(regime)
        
        # Cold start: insufficient history
        if not stats.has_sufficient_history:
            return RegimeEvaluation(
                regime=regime,
                is_allowed=True,
                confidence_score=0.5,
                sample_size=stats.testable_count,
                raw_confirmation_rate=stats.raw_confirmation_rate,
                reason=f"cold_start (N={stats.testable_count} < {self.config.MIN_SAMPLES_FOR_CONFIDENCE})",
            )
        
        confidence = stats.confidence_score
        
        # Sufficient history: apply confidence threshold
        if confidence >= self.config.MIN_CONFIDENCE_THRESHOLD:
            return RegimeEvaluation(
                regime=regime,
                is_allowed=True,
                confidence_score=confidence,
                sample_size=stats.testable_count,
                raw_confirmation_rate=stats.raw_confirmation_rate,
                reason=f"confidence_sufficient ({confidence:.1%} >= {self.config.MIN_CONFIDENCE_THRESHOLD:.0%})",
            )
        else:
            return RegimeEvaluation(
                regime=regime,
                is_allowed=False,
                confidence_score=confidence,
                sample_size=stats.testable_count,
                raw_confirmation_rate=stats.raw_confirmation_rate,
                reason=f"confidence_insufficient ({confidence:.1%} < {self.config.MIN_CONFIDENCE_THRESHOLD:.0%})",
            )


# =============================================================================
# REGIME-GATED PREDICTION GENERATOR
# =============================================================================

class RegimeGatedPredictionGenerator:
    """
    P1 Prediction Generator with P4 Structural + P5 Regime Gating.
    
    GATING ORDER:
    1. P4: Structural stability (must be VALID)
    2. P5: Regime memory (must have sufficient confidence)
    
    This is the most conservative generator - requires both gates to pass.
    """
    
    def __init__(
        self,
        structural_evaluator,  # StructuralStabilityEvaluator from structural.py
        regime_memory: RegimeMemory,
        config: RegimeConfig = REGIME_CONFIG,
    ):
        # Import here to avoid circular imports
        try:
            from .predictions import PredictionGenerator
            from .structural import StructuralValidity
        except ImportError:
            from predictions import PredictionGenerator
            from structural import StructuralValidity
        
        self._base_generator = PredictionGenerator()
        self._structural = structural_evaluator
        self._regime_memory = regime_memory
        self._regime_evaluator = RegimeEvaluator(regime_memory, config)
        self._StructuralValidity = StructuralValidity
        
        # Tracking
        self._blocked_by_structural: int = 0
        self._blocked_by_regime: int = 0
        self._allowed_cold_start: int = 0
        self._last_regime_eval: Optional[RegimeEvaluation] = None
        self._last_structural_state = None
    
    def should_generate(self, spread_obs) -> bool:
        """
        Determine if P1 prediction should be generated.
        
        GATED CHECK ORDER:
        1. Base conditions (spread valid, Z > threshold, no pending)
        2. Structural stability (P4)
        3. Regime memory confidence (P5)
        
        Returns True only if ALL pass.
        """
        # Check base conditions
        if not self._base_generator.should_generate(spread_obs):
            return False
        
        # Check structural stability (P4)
        structural_state = self._structural.evaluate(spread_obs.timestamp)
        self._last_structural_state = structural_state
        
        if not structural_state.is_valid:
            self._blocked_by_structural += 1
            return False
        
        # Create regime signature from structural state
        regime = create_regime_signature(
            correlation_stability=structural_state.correlation_stability,
            correlation_trend=structural_state.correlation_trend,
            volatility_ratio_stability=structural_state.volatility_ratio_stability,
            spread_variance_ratio=structural_state.spread_variance_ratio,
        )
        
        # Check regime memory (P5)
        regime_eval = self._regime_evaluator.evaluate(regime)
        self._last_regime_eval = regime_eval
        
        if not regime_eval.is_allowed:
            self._blocked_by_regime += 1
            return False
        
        # Track cold start allowances
        if "cold_start" in regime_eval.reason:
            self._allowed_cold_start += 1
        
        return True
    
    def generate(self, spread_obs, bar_index: int):
        """
        Generate P1 prediction (delegates to base generator).
        
        Should only be called after should_generate() returns True.
        """
        return self._base_generator.generate(spread_obs, bar_index)
    
    def mark_resolved(self, pair) -> None:
        """Mark a pair as no longer having a pending prediction."""
        self._base_generator.mark_resolved(pair)
    
    @property
    def pending_count(self) -> int:
        return self._base_generator.pending_count
    
    @property
    def total_generated(self) -> int:
        return self._base_generator.total_generated
    
    @property
    def blocked_by_structural(self) -> int:
        return self._blocked_by_structural
    
    @property
    def blocked_by_regime(self) -> int:
        return self._blocked_by_regime
    
    @property
    def allowed_cold_start(self) -> int:
        return self._allowed_cold_start
    
    @property
    def last_regime_evaluation(self) -> Optional[RegimeEvaluation]:
        return self._last_regime_eval
    
    @property
    def last_structural_state(self):
        return self._last_structural_state


# =============================================================================
# OUTCOME RECORDER
# =============================================================================

def convert_resolution_to_outcome(resolution_state) -> OutcomeType:
    """
    Convert P1 ResolutionState to simplified OutcomeType.
    
    Args:
        resolution_state: ResolutionState from resolution.py
    
    Returns:
        Simplified OutcomeType for regime learning
    """
    try:
        from .resolution import ResolutionState
    except ImportError:
        from resolution import ResolutionState
    
    if resolution_state == ResolutionState.CONFIRMED:
        return OutcomeType.CONFIRMED
    elif resolution_state == ResolutionState.REFUTED:
        return OutcomeType.REFUTED
    else:
        return OutcomeType.UNTESTABLE


class OutcomeRecorder:
    """
    Records prediction outcomes to regime memory.
    
    Tracks predictions from creation to resolution and records
    the regime-outcome pair when resolved.
    """
    
    def __init__(self, regime_memory: RegimeMemory):
        self._memory = regime_memory
        self._pending_regimes: Dict[str, Tuple[RegimeSignature, datetime]] = {}
    
    def track_prediction(
        self,
        prediction_id: str,
        regime: RegimeSignature,
        creation_timestamp: datetime,
    ) -> None:
        """
        Start tracking a prediction.
        
        Call this when a P1 prediction is created.
        """
        self._pending_regimes[prediction_id] = (regime, creation_timestamp)
    
    def record_resolution(
        self,
        prediction_id: str,
        resolution_state,
        resolution_timestamp: datetime,
        bars_elapsed: int,
    ) -> Optional[RegimeOutcome]:
        """
        Record the resolution of a tracked prediction.
        
        Call this when a P1 prediction is resolved.
        
        Returns:
            RegimeOutcome if prediction was tracked, None otherwise
        """
        if prediction_id not in self._pending_regimes:
            return None
        
        regime, creation_timestamp = self._pending_regimes.pop(prediction_id)
        outcome_type = convert_resolution_to_outcome(resolution_state)
        
        outcome = RegimeOutcome(
            prediction_id=prediction_id,
            creation_timestamp=creation_timestamp,
            resolution_timestamp=resolution_timestamp,
            regime=regime,
            outcome=outcome_type,
            bars_to_resolution=bars_elapsed,
        )
        
        self._memory.record(outcome)
        return outcome
    
    @property
    def pending_count(self) -> int:
        """Number of predictions being tracked."""
        return len(self._pending_regimes)
