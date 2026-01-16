"""
P1 Resolution Logic - EXACTLY AS SPECIFIED.

Resolution order (evaluated in sequence):
1. INVALIDATED - Structural breakdown
2. CONFIRMED - Spread reverted
3. REFUTED - Spread continued
4. TIMEOUT - Time expired
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Tuple
from enum import Enum

try:
    from .config import CONFIG
    from .predictions import P1_SpreadReversionPrediction
    from .spread import SpreadObservation
except ImportError:
    from config import CONFIG
    from predictions import P1_SpreadReversionPrediction
    from spread import SpreadObservation


class ResolutionState(Enum):
    """
    Resolution states - mutually exclusive.
    
    Every prediction resolves to exactly ONE of these states.
    """
    PENDING = "PENDING"           # Not yet resolved
    CONFIRMED = "CONFIRMED"       # Spread reverted (prediction correct)
    REFUTED = "REFUTED"           # Spread continued (prediction wrong)
    TIMEOUT = "TIMEOUT"           # Time expired, unresolvable
    INVALIDATED = "INVALIDATED"   # Structural change, untestable


class P1_Resolver:
    """
    P1 Resolution Logic - DETERMINISTIC.
    
    RESOLUTION ORDER (evaluated in sequence):
    1. INVALIDATED - Structural conditions changed
    2. CONFIRMED - |Z| < 0.3 (spread reverted)
    3. REFUTED - |Z| > 3.0 in same direction (spread continued)
    4. TIMEOUT - bars_elapsed >= 50
    5. PENDING - None of the above
    
    NO DISCRETION ALLOWED. Same inputs always produce same output.
    """
    
    def __init__(self):
        self.min_correlation = CONFIG.MIN_CORRELATION
        self.max_correlation_drop = CONFIG.MAX_CORRELATION_DROP
        self.confirmation_threshold = CONFIG.CONFIRMATION_THRESHOLD
        self.refutation_threshold = CONFIG.REFUTATION_THRESHOLD
        self.max_holding_bars = CONFIG.MAX_HOLDING_BARS
    
    def evaluate(
        self,
        prediction: P1_SpreadReversionPrediction,
        current_spread: SpreadObservation,
        bars_elapsed: int,
    ) -> Optional[ResolutionState]:
        """
        Evaluate whether prediction should be resolved.
        
        DETERMINISTIC: Same inputs always produce same output.
        
        Args:
            prediction: The P1 prediction to evaluate
            current_spread: Current spread observation
            bars_elapsed: Bars since prediction creation
        
        Returns:
            ResolutionState if resolved, None if still pending
        """
        if prediction.is_resolved:
            return prediction.resolution_state
        
        if not current_spread.is_valid:
            return None  # Cannot resolve with invalid spread data
        
        initial_z = prediction.context.zscore_at_creation
        initial_corr = prediction.context.correlation_at_creation
        current_z = current_spread.zscore
        current_corr = current_spread.correlation
        
        # ═══════════════════════════════════════════════════════════════════════
        # CONDITION 1: INVALIDATED
        # ═══════════════════════════════════════════════════════════════════════
        # Structural conditions have changed, making the prediction meaningless.
        # This is NOT the same as the prediction being wrong.
        
        # Invalidation trigger 1: Correlation collapsed below minimum
        if current_corr < self.min_correlation:
            return ResolutionState.INVALIDATED
        
        # Invalidation trigger 2: Correlation sign changed
        if initial_corr > 0.30 and current_corr < 0:
            return ResolutionState.INVALIDATED
        
        # Invalidation trigger 3: Correlation dropped by more than max_drop
        if (initial_corr - current_corr) > self.max_correlation_drop:
            return ResolutionState.INVALIDATED
        
        # ═══════════════════════════════════════════════════════════════════════
        # CONDITION 2: CONFIRMED
        # ═══════════════════════════════════════════════════════════════════════
        # Spread reverted toward mean: |Z| went from > threshold to < 0.3
        
        if abs(current_z) < self.confirmation_threshold:
            return ResolutionState.CONFIRMED
        
        # ═══════════════════════════════════════════════════════════════════════
        # CONDITION 3: REFUTED
        # ═══════════════════════════════════════════════════════════════════════
        # Spread continued in the SAME direction: |Z| exceeded refutation threshold
        
        if prediction.context.zscore_sign > 0:
            # Initial Z was positive (spread above mean)
            # Refuted if current Z is even more positive (continued up)
            if current_z > self.refutation_threshold:
                return ResolutionState.REFUTED
        else:
            # Initial Z was negative (spread below mean)
            # Refuted if current Z is even more negative (continued down)
            if current_z < -self.refutation_threshold:
                return ResolutionState.REFUTED
        
        # ═══════════════════════════════════════════════════════════════════════
        # CONDITION 4: TIMEOUT
        # ═══════════════════════════════════════════════════════════════════════
        # Time horizon reached without clear resolution
        
        if bars_elapsed >= self.max_holding_bars:
            return ResolutionState.TIMEOUT
        
        # ═══════════════════════════════════════════════════════════════════════
        # CONDITION 5: PENDING
        # ═══════════════════════════════════════════════════════════════════════
        # None of the above conditions met - prediction still pending
        
        return None


@dataclass
class ResolutionResult:
    """Result of attempting to resolve a prediction."""
    prediction_id: str
    state: Optional[ResolutionState]
    bars_elapsed: int
    current_zscore: float
    current_correlation: float


class ResolutionEngine:
    """
    Engine that processes pending predictions and resolves them.
    """
    
    def __init__(self):
        self.resolver = P1_Resolver()
        self._pending: List[P1_SpreadReversionPrediction] = []
        self._resolved: List[P1_SpreadReversionPrediction] = []
    
    def add_pending(self, prediction: P1_SpreadReversionPrediction) -> None:
        """Add a prediction to the pending queue."""
        self._pending.append(prediction)
    
    def process(
        self,
        current_spread: SpreadObservation,
        current_bar_index: int,
    ) -> List[ResolutionResult]:
        """
        Process all pending predictions against current spread.
        
        Returns list of resolution results (for logging/auditing).
        """
        results = []
        still_pending = []
        
        for prediction in self._pending:
            bars_elapsed = current_bar_index - prediction.creation_bar_index
            
            state = self.resolver.evaluate(
                prediction=prediction,
                current_spread=current_spread,
                bars_elapsed=bars_elapsed,
            )
            
            result = ResolutionResult(
                prediction_id=prediction.prediction_id,
                state=state,
                bars_elapsed=bars_elapsed,
                current_zscore=current_spread.zscore,
                current_correlation=current_spread.correlation,
            )
            results.append(result)
            
            if state is not None:
                # Resolve the prediction
                prediction.resolve(
                    state=state,
                    timestamp=current_spread.timestamp,
                    observation_id=current_spread.observation_id,
                    bar_index=current_bar_index,
                    bars_elapsed=bars_elapsed,
                    zscore_final=current_spread.zscore,
                    spread_final=current_spread.spread_value,
                    correlation_final=current_spread.correlation,
                )
                self._resolved.append(prediction)
            else:
                still_pending.append(prediction)
        
        self._pending = still_pending
        return results
    
    @property
    def pending_count(self) -> int:
        """Number of pending predictions."""
        return len(self._pending)
    
    @property
    def resolved_count(self) -> int:
        """Number of resolved predictions."""
        return len(self._resolved)
    
    def get_resolved(self) -> List[P1_SpreadReversionPrediction]:
        """Get all resolved predictions."""
        return self._resolved.copy()
    
    def get_pending(self) -> List[P1_SpreadReversionPrediction]:
        """Get all pending predictions."""
        return self._pending.copy()
