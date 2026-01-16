"""
Resolution statistics accumulator.

Computes CRR, timeout rate, invalidation rate, and related metrics.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime

try:
    from .predictions import P1_SpreadReversionPrediction
    from .resolution import ResolutionState
except ImportError:
    from predictions import P1_SpreadReversionPrediction
    from resolution import ResolutionState


@dataclass
class ResolutionStatistics:
    """
    Aggregate resolution statistics.
    
    Tracks counts by resolution state and computes derived metrics.
    """
    # Counts by state
    total_predictions: int = 0
    confirmed_count: int = 0
    refuted_count: int = 0
    timeout_count: int = 0
    invalidated_count: int = 0
    pending_count: int = 0
    
    # Timing metrics
    avg_bars_to_resolution: float = 0.0
    avg_bars_to_confirmation: float = 0.0
    avg_bars_to_refutation: float = 0.0
    
    # Z-score metrics at creation
    avg_zscore_at_creation: float = 0.0
    avg_zscore_confirmed: float = 0.0
    avg_zscore_refuted: float = 0.0
    
    @property
    def testable_count(self) -> int:
        """
        Predictions that were actually tested (CONFIRMED + REFUTED).
        
        TIMEOUT and INVALIDATED are excluded from CRR calculation.
        """
        return self.confirmed_count + self.refuted_count
    
    @property
    def resolved_count(self) -> int:
        """Predictions that reached a final state."""
        return self.total_predictions - self.pending_count
    
    @property
    def crr(self) -> float:
        """
        Conditional Reversion Rate: CONFIRMED / (CONFIRMED + REFUTED).
        
        This is the PRIMARY edge metric.
        CRR > 0.50 indicates potential edge.
        """
        if self.testable_count == 0:
            return 0.5  # No evidence - assume null hypothesis
        return self.confirmed_count / self.testable_count
    
    @property
    def timeout_rate(self) -> float:
        """Timeout rate among resolved predictions."""
        if self.resolved_count == 0:
            return 0.0
        return self.timeout_count / self.resolved_count
    
    @property
    def invalidation_rate(self) -> float:
        """Invalidation rate among resolved predictions."""
        if self.resolved_count == 0:
            return 0.0
        return self.invalidated_count / self.resolved_count
    
    @property
    def confirmation_rate(self) -> float:
        """Confirmation rate among resolved predictions."""
        if self.resolved_count == 0:
            return 0.0
        return self.confirmed_count / self.resolved_count
    
    @property
    def refutation_rate(self) -> float:
        """Refutation rate among resolved predictions."""
        if self.resolved_count == 0:
            return 0.0
        return self.refuted_count / self.resolved_count
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "",
            "=" * 70,
            "                    RESOLUTION STATISTICS",
            "=" * 70,
            "",
            f"  Total Predictions:     {self.total_predictions}",
            f"  Resolved:              {self.resolved_count}",
            f"  Pending:               {self.pending_count}",
            "",
            "  RESOLUTION BREAKDOWN:",
            f"    CONFIRMED:           {self.confirmed_count:4d}  ({self.confirmation_rate:6.1%})",
            f"    REFUTED:             {self.refuted_count:4d}  ({self.refutation_rate:6.1%})",
            f"    TIMEOUT:             {self.timeout_count:4d}  ({self.timeout_rate:6.1%})",
            f"    INVALIDATED:         {self.invalidated_count:4d}  ({self.invalidation_rate:6.1%})",
            "",
            "  KEY METRICS:",
            f"    Testable Count:      {self.testable_count}",
            f"    CRR (Edge Metric):   {self.crr:.1%}",
            "",
            f"    Avg Z at Creation:   {self.avg_zscore_at_creation:.2f}",
            f"    Avg Bars to Resolve: {self.avg_bars_to_resolution:.1f}",
            "",
            "=" * 70,
        ]
        return "\n".join(lines)


class StatisticsAccumulator:
    """
    Accumulates predictions and computes statistics.
    """
    
    def __init__(self):
        self._predictions: List[P1_SpreadReversionPrediction] = []
    
    def add(self, prediction: P1_SpreadReversionPrediction) -> None:
        """Add a prediction to the accumulator."""
        self._predictions.append(prediction)
    
    def compute(self) -> ResolutionStatistics:
        """
        Compute aggregate statistics from accumulated predictions.
        """
        stats = ResolutionStatistics()
        
        stats.total_predictions = len(self._predictions)
        
        # Accumulators for averages
        total_bars = 0
        total_bars_confirmed = 0
        total_bars_refuted = 0
        total_zscore = 0.0
        total_zscore_confirmed = 0.0
        total_zscore_refuted = 0.0
        
        for pred in self._predictions:
            # Count by state
            if pred.resolution_state is None:
                stats.pending_count += 1
            elif pred.resolution_state == ResolutionState.CONFIRMED:
                stats.confirmed_count += 1
                if pred.resolution_bars_elapsed:
                    total_bars_confirmed += pred.resolution_bars_elapsed
                total_zscore_confirmed += abs(pred.context.zscore_at_creation)
            elif pred.resolution_state == ResolutionState.REFUTED:
                stats.refuted_count += 1
                if pred.resolution_bars_elapsed:
                    total_bars_refuted += pred.resolution_bars_elapsed
                total_zscore_refuted += abs(pred.context.zscore_at_creation)
            elif pred.resolution_state == ResolutionState.TIMEOUT:
                stats.timeout_count += 1
            elif pred.resolution_state == ResolutionState.INVALIDATED:
                stats.invalidated_count += 1
            
            # Accumulate for averages
            total_zscore += abs(pred.context.zscore_at_creation)
            if pred.resolution_bars_elapsed:
                total_bars += pred.resolution_bars_elapsed
        
        # Compute averages
        if stats.total_predictions > 0:
            stats.avg_zscore_at_creation = total_zscore / stats.total_predictions
        
        if stats.resolved_count > 0:
            stats.avg_bars_to_resolution = total_bars / stats.resolved_count
        
        if stats.confirmed_count > 0:
            stats.avg_bars_to_confirmation = total_bars_confirmed / stats.confirmed_count
            stats.avg_zscore_confirmed = total_zscore_confirmed / stats.confirmed_count
        
        if stats.refuted_count > 0:
            stats.avg_bars_to_refutation = total_bars_refuted / stats.refuted_count
            stats.avg_zscore_refuted = total_zscore_refuted / stats.refuted_count
        
        return stats
    
    def get_predictions_by_state(
        self,
        state: ResolutionState,
    ) -> List[P1_SpreadReversionPrediction]:
        """Get all predictions with a specific resolution state."""
        return [p for p in self._predictions if p.resolution_state == state]
    
    def clear(self) -> None:
        """Clear all accumulated predictions."""
        self._predictions = []
    
    @property
    def count(self) -> int:
        """Number of accumulated predictions."""
        return len(self._predictions)
