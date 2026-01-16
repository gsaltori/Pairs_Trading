"""
P4 Structural Stability Module

This module determines whether a pair relationship is currently stable
BEFORE allowing P1 prediction generation.

DESIGN PRINCIPLES:
1. Stability over level - we care about variance, not absolute values
2. Forward-only - all metrics use only past data
3. Multi-dimensional - multiple stability criteria must be satisfied
4. Conservative - better to miss opportunities than generate doomed P1s
5. Explainable - every decision has clear reasoning

STABILITY DIMENSIONS:
1. Correlation stability (variance of rolling correlations)
2. Correlation trend (is correlation drifting down?)
3. Volatility ratio stability (are relative volatilities consistent?)
4. Spread variance ratio (is spread becoming explosive?)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple
from enum import Enum
import numpy as np

try:
    from .config import CONFIG
except ImportError:
    from config import CONFIG


class StructuralValidity(Enum):
    """Binary structural state."""
    STRUCTURALLY_VALID = "STRUCTURALLY_VALID"
    STRUCTURALLY_INVALID = "STRUCTURALLY_INVALID"


@dataclass(frozen=True)
class StructuralState:
    """
    Complete structural state assessment at a point in time.
    
    IMMUTABLE after creation.
    """
    # Overall validity
    validity: StructuralValidity
    
    # Timestamp
    timestamp: datetime
    
    # Individual metrics
    correlation_stability: float      # Std of rolling correlations (lower = more stable)
    correlation_trend: float          # Slope of correlation (negative = declining)
    correlation_current: float        # Current correlation level
    volatility_ratio_stability: float # Std of vol_a/vol_b ratio (lower = more stable)
    spread_variance_ratio: float      # Current spread var / historical spread var
    
    # Diagnostic
    invalidity_reasons: Tuple[str, ...] = field(default=())
    
    @property
    def is_valid(self) -> bool:
        """Convenience property for validity check."""
        return self.validity == StructuralValidity.STRUCTURALLY_VALID
    
    def summary(self) -> str:
        """Human-readable summary."""
        status = "VALID" if self.is_valid else "INVALID"
        lines = [
            f"Structural State: {status}",
            f"  Correlation Stability:    {self.correlation_stability:.4f}",
            f"  Correlation Trend:        {self.correlation_trend:+.4f}",
            f"  Correlation Current:      {self.correlation_current:.3f}",
            f"  Vol Ratio Stability:      {self.volatility_ratio_stability:.4f}",
            f"  Spread Variance Ratio:    {self.spread_variance_ratio:.2f}x",
        ]
        if self.invalidity_reasons:
            lines.append("  Reasons for invalidity:")
            for reason in self.invalidity_reasons:
                lines.append(f"    - {reason}")
        return "\n".join(lines)


@dataclass(frozen=True)
class StructuralConfig:
    """
    Configuration for structural stability evaluation.
    
    ALL VALUES ARE FIXED. NO TUNING PERMITTED.
    """
    # Window sizes
    CORRELATION_WINDOW: int = 20          # Bars to assess correlation stability
    VOLATILITY_WINDOW: int = 20           # Bars to assess volatility ratio
    SPREAD_VARIANCE_WINDOW: int = 30      # Bars for spread variance comparison
    SPREAD_VARIANCE_BASELINE: int = 60    # Historical baseline for spread variance
    
    # Stability thresholds
    MAX_CORRELATION_STD: float = 0.12     # Max std of rolling correlations
    MIN_CORRELATION_TREND: float = -0.008 # Min slope (more negative = declining faster)
    MAX_VOLATILITY_RATIO_STD: float = 0.25  # Max std of vol ratio
    MAX_SPREAD_VARIANCE_RATIO: float = 2.5  # Max current/historical spread variance
    
    # Minimum data requirements
    MIN_HISTORY_BARS: int = 60            # Minimum bars before evaluation is valid


# Global structural config
STRUCTURAL_CONFIG = StructuralConfig()


class StructuralStabilityEvaluator:
    """
    Evaluates structural stability of a pair relationship.
    
    Uses rolling windows to assess whether the pair relationship
    is currently stable enough to support P1 predictions.
    
    FORWARD-ONLY: All computations use only past data.
    """
    
    def __init__(
        self,
        correlation_window: int = STRUCTURAL_CONFIG.CORRELATION_WINDOW,
        volatility_window: int = STRUCTURAL_CONFIG.VOLATILITY_WINDOW,
        spread_var_window: int = STRUCTURAL_CONFIG.SPREAD_VARIANCE_WINDOW,
        spread_var_baseline: int = STRUCTURAL_CONFIG.SPREAD_VARIANCE_BASELINE,
    ):
        self.correlation_window = correlation_window
        self.volatility_window = volatility_window
        self.spread_var_window = spread_var_window
        self.spread_var_baseline = spread_var_baseline
        
        # Historical buffers
        self._correlations: List[float] = []
        self._volatility_ratios: List[float] = []
        self._spread_values: List[float] = []
        self._timestamps: List[datetime] = []
        
        # Blocked count for statistics
        self._blocked_count: int = 0
    
    def update(
        self,
        timestamp: datetime,
        correlation: float,
        volatility_a: float,
        volatility_b: float,
        spread_value: float,
    ) -> None:
        """
        Update evaluator with new observation data.
        
        Args:
            timestamp: Current timestamp
            correlation: Rolling correlation between prices
            volatility_a: Volatility of symbol A (e.g., std of returns)
            volatility_b: Volatility of symbol B
            spread_value: Current spread value
        """
        self._timestamps.append(timestamp)
        self._correlations.append(correlation)
        
        # Compute volatility ratio (avoid division by zero)
        if volatility_b > 1e-10:
            vol_ratio = volatility_a / volatility_b
        else:
            vol_ratio = 1.0
        self._volatility_ratios.append(vol_ratio)
        
        self._spread_values.append(spread_value)
        
        # Trim buffers to reasonable size
        max_history = max(
            self.spread_var_baseline + 20,
            self.correlation_window + 20,
            self.volatility_window + 20,
        )
        if len(self._correlations) > max_history:
            self._correlations = self._correlations[-max_history:]
            self._volatility_ratios = self._volatility_ratios[-max_history:]
            self._spread_values = self._spread_values[-max_history:]
            self._timestamps = self._timestamps[-max_history:]
    
    def evaluate(self, timestamp: datetime) -> StructuralState:
        """
        Evaluate current structural stability.
        
        Returns StructuralState with validity assessment and metrics.
        """
        invalidity_reasons = []
        
        # Check minimum data requirement
        if len(self._correlations) < STRUCTURAL_CONFIG.MIN_HISTORY_BARS:
            return StructuralState(
                validity=StructuralValidity.STRUCTURALLY_INVALID,
                timestamp=timestamp,
                correlation_stability=1.0,
                correlation_trend=0.0,
                correlation_current=0.0,
                volatility_ratio_stability=1.0,
                spread_variance_ratio=1.0,
                invalidity_reasons=(f"Insufficient history: {len(self._correlations)} < {STRUCTURAL_CONFIG.MIN_HISTORY_BARS}",),
            )
        
        # ═══════════════════════════════════════════════════════════════════════
        # METRIC 1: Correlation Stability
        # ═══════════════════════════════════════════════════════════════════════
        recent_correlations = self._correlations[-self.correlation_window:]
        correlation_stability = float(np.std(recent_correlations))
        correlation_current = float(recent_correlations[-1])
        
        if correlation_stability > STRUCTURAL_CONFIG.MAX_CORRELATION_STD:
            invalidity_reasons.append(
                f"Correlation unstable: std={correlation_stability:.4f} > {STRUCTURAL_CONFIG.MAX_CORRELATION_STD}"
            )
        
        # ═══════════════════════════════════════════════════════════════════════
        # METRIC 2: Correlation Trend
        # ═══════════════════════════════════════════════════════════════════════
        correlation_trend = self._compute_slope(recent_correlations)
        
        if correlation_trend < STRUCTURAL_CONFIG.MIN_CORRELATION_TREND:
            invalidity_reasons.append(
                f"Correlation declining: trend={correlation_trend:+.4f} < {STRUCTURAL_CONFIG.MIN_CORRELATION_TREND}"
            )
        
        # ═══════════════════════════════════════════════════════════════════════
        # METRIC 3: Volatility Ratio Stability
        # ═══════════════════════════════════════════════════════════════════════
        recent_vol_ratios = self._volatility_ratios[-self.volatility_window:]
        volatility_ratio_stability = float(np.std(recent_vol_ratios))
        
        if volatility_ratio_stability > STRUCTURAL_CONFIG.MAX_VOLATILITY_RATIO_STD:
            invalidity_reasons.append(
                f"Volatility ratio unstable: std={volatility_ratio_stability:.4f} > {STRUCTURAL_CONFIG.MAX_VOLATILITY_RATIO_STD}"
            )
        
        # ═══════════════════════════════════════════════════════════════════════
        # METRIC 4: Spread Variance Ratio
        # ═══════════════════════════════════════════════════════════════════════
        spread_variance_ratio = self._compute_spread_variance_ratio()
        
        if spread_variance_ratio > STRUCTURAL_CONFIG.MAX_SPREAD_VARIANCE_RATIO:
            invalidity_reasons.append(
                f"Spread variance exploding: ratio={spread_variance_ratio:.2f}x > {STRUCTURAL_CONFIG.MAX_SPREAD_VARIANCE_RATIO}x"
            )
        
        # ═══════════════════════════════════════════════════════════════════════
        # FINAL VALIDITY DETERMINATION
        # ═══════════════════════════════════════════════════════════════════════
        is_valid = len(invalidity_reasons) == 0
        validity = StructuralValidity.STRUCTURALLY_VALID if is_valid else StructuralValidity.STRUCTURALLY_INVALID
        
        if not is_valid:
            self._blocked_count += 1
        
        return StructuralState(
            validity=validity,
            timestamp=timestamp,
            correlation_stability=correlation_stability,
            correlation_trend=correlation_trend,
            correlation_current=correlation_current,
            volatility_ratio_stability=volatility_ratio_stability,
            spread_variance_ratio=spread_variance_ratio,
            invalidity_reasons=tuple(invalidity_reasons),
        )
    
    def _compute_slope(self, values: List[float]) -> float:
        """
        Compute linear regression slope of values.
        
        Positive slope = increasing trend
        Negative slope = decreasing trend
        """
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x = np.arange(n)
        y = np.array(values)
        
        # Simple linear regression: slope = Cov(x,y) / Var(x)
        x_mean = x.mean()
        y_mean = y.mean()
        
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        if abs(denominator) < 1e-10:
            return 0.0
        
        return float(numerator / denominator)
    
    def _compute_spread_variance_ratio(self) -> float:
        """
        Compute ratio of recent spread variance to historical baseline.
        
        Ratio > 1.0 means spread is more volatile than usual.
        Ratio > 2.0 suggests regime change / structural instability.
        """
        if len(self._spread_values) < self.spread_var_baseline:
            return 1.0
        
        # Recent variance
        recent_spreads = self._spread_values[-self.spread_var_window:]
        recent_var = np.var(recent_spreads)
        
        # Historical baseline variance (excluding recent period)
        baseline_end = -self.spread_var_window
        baseline_start = baseline_end - self.spread_var_baseline
        if baseline_start < 0:
            baseline_start = 0
        
        baseline_spreads = self._spread_values[baseline_start:baseline_end]
        if len(baseline_spreads) < 10:
            return 1.0
        
        baseline_var = np.var(baseline_spreads)
        
        if baseline_var < 1e-15:
            return 1.0
        
        return float(recent_var / baseline_var)
    
    @property
    def blocked_count(self) -> int:
        """Number of times evaluation returned INVALID."""
        return self._blocked_count
    
    def reset(self) -> None:
        """Reset evaluator state."""
        self._correlations = []
        self._volatility_ratios = []
        self._spread_values = []
        self._timestamps = []
        self._blocked_count = 0


class StructuralGate:
    """
    High-level interface for structural gating of P1 predictions.
    
    Wraps StructuralStabilityEvaluator with convenient methods
    for integration into the prediction pipeline.
    """
    
    def __init__(self):
        self.evaluator = StructuralStabilityEvaluator()
        self._last_state: Optional[StructuralState] = None
        self._total_checks: int = 0
        self._passed_checks: int = 0
        self._blocked_checks: int = 0
    
    def update_from_spread(
        self,
        spread_obs,  # SpreadObservation - avoiding circular import
        volatility_a: float,
        volatility_b: float,
    ) -> None:
        """
        Update structural evaluator from a spread observation.
        
        Args:
            spread_obs: SpreadObservation containing correlation and spread
            volatility_a: Rolling volatility of symbol A
            volatility_b: Rolling volatility of symbol B
        """
        self.evaluator.update(
            timestamp=spread_obs.timestamp,
            correlation=spread_obs.correlation,
            volatility_a=volatility_a,
            volatility_b=volatility_b,
            spread_value=spread_obs.spread_value,
        )
    
    def check(self, timestamp: datetime) -> StructuralState:
        """
        Check current structural validity.
        
        Returns StructuralState and updates internal statistics.
        """
        self._total_checks += 1
        state = self.evaluator.evaluate(timestamp)
        self._last_state = state
        
        if state.is_valid:
            self._passed_checks += 1
        else:
            self._blocked_checks += 1
        
        return state
    
    def should_allow_p1(self, timestamp: datetime) -> bool:
        """
        Simplified check: should P1 generation be allowed?
        
        Returns True if structurally valid, False otherwise.
        """
        state = self.check(timestamp)
        return state.is_valid
    
    @property
    def last_state(self) -> Optional[StructuralState]:
        """Most recent structural state."""
        return self._last_state
    
    @property
    def total_checks(self) -> int:
        """Total number of validity checks performed."""
        return self._total_checks
    
    @property
    def passed_checks(self) -> int:
        """Number of checks that returned VALID."""
        return self._passed_checks
    
    @property
    def blocked_checks(self) -> int:
        """Number of checks that returned INVALID."""
        return self._blocked_checks
    
    @property
    def pass_rate(self) -> float:
        """Percentage of checks that passed."""
        if self._total_checks == 0:
            return 0.0
        return self._passed_checks / self._total_checks
    
    def reset(self) -> None:
        """Reset gate state and statistics."""
        self.evaluator.reset()
        self._last_state = None
        self._total_checks = 0
        self._passed_checks = 0
        self._blocked_checks = 0
    
    def summary(self) -> str:
        """Generate summary statistics."""
        lines = [
            "Structural Gate Statistics:",
            f"  Total Checks:    {self._total_checks}",
            f"  Passed:          {self._passed_checks} ({self.pass_rate:.1%})",
            f"  Blocked:         {self._blocked_checks} ({1 - self.pass_rate:.1%})",
        ]
        if self._last_state:
            lines.append("")
            lines.append("Last State:")
            lines.append(self._last_state.summary())
        return "\n".join(lines)
