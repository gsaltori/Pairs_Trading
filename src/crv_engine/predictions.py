"""
P1 Spread Reversion Prediction implementation.

This is the ONLY prediction type implemented in this vertical slice.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Tuple, Optional, List, TYPE_CHECKING
from enum import Enum
import hashlib
import json
import numpy as np

try:
    from .config import CONFIG
    from .spread import SpreadObservation
except ImportError:
    from config import CONFIG
    from spread import SpreadObservation

if TYPE_CHECKING:
    from .resolution import ResolutionState


class PredictionType(Enum):
    """Prediction type enumeration."""
    P1_SPREAD_REVERSION = "P1_SPREAD_REVERSION"


class PredictionDirection(Enum):
    """Predicted direction."""
    REVERT_TOWARD_MEAN = "REVERT_TOWARD_MEAN"


@dataclass(frozen=True)
class HypothesisContext:
    """
    Complete context at hypothesis creation time.
    
    FROZEN: Cannot be modified after creation.
    PURPOSE: Capture ALL information available at prediction time.
    """
    # Spread state at creation
    zscore_at_creation: float
    spread_at_creation: float
    hedge_ratio_at_creation: float
    
    # Prices at creation
    price_a_at_creation: float
    price_b_at_creation: float
    
    # Statistics at creation
    spread_mean_at_creation: float
    spread_std_at_creation: float
    
    # Correlation at creation
    correlation_at_creation: float
    
    # Direction sign (+1 or -1)
    zscore_sign: int


@dataclass
class P1_SpreadReversionPrediction:
    """
    P1: Spread Reversion Prediction
    
    CLAIM: Given |Z₀| > threshold, spread is more likely to revert
    toward mean (|Z| < 0.3) than to continue (|Z| > 3.0).
    
    IMMUTABILITY RULES:
    1. Creation fields are set once at creation
    2. Resolution fields can be set exactly ONCE
    3. _is_finalized flag prevents any further changes
    """
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CREATION FIELDS (Immutable after creation)
    # ═══════════════════════════════════════════════════════════════════════════
    prediction_id: str
    prediction_type: PredictionType
    
    creation_timestamp: datetime
    creation_observation_id: str
    creation_bar_index: int
    
    # Pair and timeframe
    pair: Tuple[str, str]
    timeframe: str
    
    # Prediction direction
    prediction: PredictionDirection
    
    # Context at creation (immutable snapshot)
    context: HypothesisContext
    
    # Resolution parameters (fixed at creation from CONFIG)
    trigger_threshold: float = CONFIG.TRIGGER_THRESHOLD
    confirmation_threshold: float = CONFIG.CONFIRMATION_THRESHOLD
    refutation_threshold: float = CONFIG.REFUTATION_THRESHOLD
    max_holding_bars: int = CONFIG.MAX_HOLDING_BARS
    
    # Integrity hash
    creation_hash: str = field(default='')
    
    # ═══════════════════════════════════════════════════════════════════════════
    # RESOLUTION FIELDS (Set exactly once)
    # ═══════════════════════════════════════════════════════════════════════════
    resolution_state: Optional['ResolutionState'] = None
    resolution_timestamp: Optional[datetime] = None
    resolution_observation_id: Optional[str] = None
    resolution_bar_index: Optional[int] = None
    resolution_bars_elapsed: Optional[int] = None
    
    # Values at resolution
    zscore_at_resolution: Optional[float] = None
    spread_at_resolution: Optional[float] = None
    correlation_at_resolution: Optional[float] = None
    
    # Finalization flag
    _is_finalized: bool = field(default=False, repr=False)
    
    def __post_init__(self):
        """Compute creation hash if not provided."""
        if not self.creation_hash:
            object.__setattr__(self, 'creation_hash', self._compute_creation_hash())
    
    def _compute_creation_hash(self) -> str:
        """Hash of creation fields for integrity verification."""
        data = {
            'id': self.prediction_id,
            'timestamp': self.creation_timestamp.isoformat(),
            'pair': list(self.pair),
            'zscore': self.context.zscore_at_creation,
            'bar_index': self.creation_bar_index,
        }
        content = json.dumps(data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:32]
    
    def resolve(
        self,
        state: 'ResolutionState',
        timestamp: datetime,
        observation_id: str,
        bar_index: int,
        bars_elapsed: int,
        zscore_final: float,
        spread_final: float,
        correlation_final: float,
    ) -> None:
        """
        Resolve the prediction.
        
        CRITICAL: Can only be called ONCE.
        """
        if self._is_finalized:
            raise RuntimeError(f"Prediction {self.prediction_id} is already finalized")
        
        try:
            from .resolution import ResolutionState
        except ImportError:
            from resolution import ResolutionState
        
        if state == ResolutionState.PENDING:
            raise ValueError("Cannot resolve to PENDING state")
        
        self.resolution_state = state
        self.resolution_timestamp = timestamp
        self.resolution_observation_id = observation_id
        self.resolution_bar_index = bar_index
        self.resolution_bars_elapsed = bars_elapsed
        self.zscore_at_resolution = zscore_final
        self.spread_at_resolution = spread_final
        self.correlation_at_resolution = correlation_final
        self._is_finalized = True
    
    @property
    def is_pending(self) -> bool:
        """Check if prediction is still pending."""
        try:
            from .resolution import ResolutionState
        except ImportError:
            from resolution import ResolutionState
        return self.resolution_state is None or self.resolution_state == ResolutionState.PENDING
    
    @property
    def is_resolved(self) -> bool:
        """Check if prediction has been resolved."""
        return not self.is_pending


def create_prediction_id(
    timestamp: datetime,
    pair: Tuple[str, str],
    bar_index: int,
) -> str:
    """
    Create unique prediction ID.
    
    Format: P1-{YYYYMMDD}-{HHMM}-{PAIR}-{INDEX}
    """
    pair_str = f"{pair[0]}_{pair[1]}"
    return f"P1-{timestamp.strftime('%Y%m%d')}-{timestamp.strftime('%H%M')}-{pair_str}-{bar_index:05d}"


class PredictionGenerator:
    """
    Generates P1 predictions based on Z-score threshold breaches.
    
    RULES:
    1. |Z-score| must exceed threshold (1.5)
    2. No existing pending prediction for the same pair
    3. Spread observation must be valid
    
    NOTE: This is the UNGATED version. Use GatedPredictionGenerator
    for structural stability filtering.
    """
    
    def __init__(self):
        self.trigger_threshold = CONFIG.TRIGGER_THRESHOLD
        self._pending_pairs: set = set()
        self._prediction_count = 0
    
    def should_generate(
        self,
        spread_obs: SpreadObservation,
    ) -> bool:
        """
        Determine if P1 prediction should be generated.
        
        Returns True if:
        1. Spread observation is valid
        2. |Z-score| > threshold
        3. No pending prediction for this pair
        """
        if not spread_obs.is_valid:
            return False
        
        if spread_obs.pair in self._pending_pairs:
            return False
        
        if abs(spread_obs.zscore) <= self.trigger_threshold:
            return False
        
        return True
    
    def generate(
        self,
        spread_obs: SpreadObservation,
        bar_index: int,
    ) -> P1_SpreadReversionPrediction:
        """
        Generate P1 prediction from spread observation.
        
        CRITICAL: All context is captured here, BEFORE outcome is known.
        """
        self._prediction_count += 1
        
        # Create context (immutable snapshot)
        context = HypothesisContext(
            zscore_at_creation=spread_obs.zscore,
            spread_at_creation=spread_obs.spread_value,
            hedge_ratio_at_creation=spread_obs.hedge_ratio,
            price_a_at_creation=spread_obs.price_a,
            price_b_at_creation=spread_obs.price_b,
            spread_mean_at_creation=spread_obs.spread_mean,
            spread_std_at_creation=spread_obs.spread_std,
            correlation_at_creation=spread_obs.correlation,
            zscore_sign=1 if spread_obs.zscore > 0 else -1,
        )
        
        # Generate prediction ID
        prediction_id = create_prediction_id(
            spread_obs.timestamp,
            spread_obs.pair,
            bar_index,
        )
        
        # Create prediction
        prediction = P1_SpreadReversionPrediction(
            prediction_id=prediction_id,
            prediction_type=PredictionType.P1_SPREAD_REVERSION,
            creation_timestamp=spread_obs.timestamp,
            creation_observation_id=spread_obs.observation_id,
            creation_bar_index=bar_index,
            pair=spread_obs.pair,
            timeframe=spread_obs.timeframe,
            prediction=PredictionDirection.REVERT_TOWARD_MEAN,
            context=context,
        )
        
        # Track pending
        self._pending_pairs.add(spread_obs.pair)
        
        return prediction
    
    def mark_resolved(self, pair: Tuple[str, str]) -> None:
        """Mark a pair as no longer having a pending prediction."""
        self._pending_pairs.discard(pair)
    
    @property
    def pending_count(self) -> int:
        """Number of pairs with pending predictions."""
        return len(self._pending_pairs)
    
    @property
    def total_generated(self) -> int:
        """Total predictions generated."""
        return self._prediction_count


# =============================================================================
# GATED PREDICTION GENERATOR (P4 Structural Filtering)
# =============================================================================

@dataclass
class BlockedPrediction:
    """
    Record of a prediction that was blocked by structural gate.
    
    Used for analysis and transparency.
    """
    timestamp: datetime
    bar_index: int
    zscore: float
    correlation: float
    block_reasons: Tuple[str, ...]


class GatedPredictionGenerator:
    """
    P1 Prediction Generator with P4 Structural Gating.
    
    This generator wraps the base PredictionGenerator and adds
    structural stability checks before allowing P1 generation.
    
    P1 predictions are ONLY generated when:
    1. Base conditions are met (Z > threshold, no pending, valid spread)
    2. Structural state is VALID
    
    This should reduce INVALIDATED outcomes by preemptively filtering
    predictions that would have been made in unstable regimes.
    """
    
    def __init__(self):
        self._base_generator = PredictionGenerator()
        
        try:
            from .structural import StructuralGate
        except ImportError:
            from structural import StructuralGate
        
        self._structural_gate = StructuralGate()
        
        # Track blocked predictions for analysis
        self._blocked_predictions: List[BlockedPrediction] = []
        
        # Price history for volatility calculation
        self._prices_a: List[float] = []
        self._prices_b: List[float] = []
        self._volatility_window = 20
    
    def update_prices(self, price_a: float, price_b: float) -> None:
        """
        Update price history for volatility calculation.
        
        Must be called for each new bar BEFORE should_generate().
        """
        self._prices_a.append(price_a)
        self._prices_b.append(price_b)
        
        # Trim to reasonable size
        max_history = 100
        if len(self._prices_a) > max_history:
            self._prices_a = self._prices_a[-max_history:]
            self._prices_b = self._prices_b[-max_history:]
    
    def _compute_volatility(self, prices: List[float]) -> float:
        """Compute realized volatility (std of returns)."""
        if len(prices) < self._volatility_window:
            return 0.01  # Default volatility
        
        recent = np.array(prices[-self._volatility_window:])
        returns = np.diff(recent) / recent[:-1]
        return float(np.std(returns)) if len(returns) > 0 else 0.01
    
    def should_generate(
        self,
        spread_obs: SpreadObservation,
        bar_index: int,
    ) -> bool:
        """
        Determine if P1 prediction should be generated.
        
        GATED: Checks both base conditions AND structural stability.
        
        Returns True only if:
        1. Base conditions pass (Z > threshold, no pending, valid)
        2. Structural state is VALID
        """
        # First check base conditions
        base_would_generate = self._base_generator.should_generate(spread_obs)
        
        if not base_would_generate:
            return False
        
        # Update structural gate with current data
        vol_a = self._compute_volatility(self._prices_a)
        vol_b = self._compute_volatility(self._prices_b)
        
        self._structural_gate.update_from_spread(
            spread_obs=spread_obs,
            volatility_a=vol_a,
            volatility_b=vol_b,
        )
        
        # Check structural validity
        structural_state = self._structural_gate.check(spread_obs.timestamp)
        
        if not structural_state.is_valid:
            # Record blocked prediction for analysis
            self._blocked_predictions.append(BlockedPrediction(
                timestamp=spread_obs.timestamp,
                bar_index=bar_index,
                zscore=spread_obs.zscore,
                correlation=spread_obs.correlation,
                block_reasons=structural_state.invalidity_reasons,
            ))
            return False
        
        return True
    
    def generate(
        self,
        spread_obs: SpreadObservation,
        bar_index: int,
    ) -> P1_SpreadReversionPrediction:
        """
        Generate P1 prediction.
        
        Should only be called after should_generate() returns True.
        """
        return self._base_generator.generate(spread_obs, bar_index)
    
    def mark_resolved(self, pair: Tuple[str, str]) -> None:
        """Mark a pair as no longer having a pending prediction."""
        self._base_generator.mark_resolved(pair)
    
    @property
    def pending_count(self) -> int:
        """Number of pairs with pending predictions."""
        return self._base_generator.pending_count
    
    @property
    def total_generated(self) -> int:
        """Total predictions generated (that passed gate)."""
        return self._base_generator.total_generated
    
    @property
    def total_blocked(self) -> int:
        """Total predictions blocked by structural gate."""
        return len(self._blocked_predictions)
    
    @property
    def blocked_predictions(self) -> List[BlockedPrediction]:
        """List of all blocked predictions."""
        return self._blocked_predictions.copy()
    
    @property
    def structural_gate(self):
        """Access to structural gate for statistics."""
        return self._structural_gate
    
    def get_block_reasons_summary(self) -> dict:
        """
        Get summary of block reasons.
        
        Returns dict mapping reason string to count.
        """
        reasons = {}
        for blocked in self._blocked_predictions:
            for reason in blocked.block_reasons:
                # Simplify reason for counting
                if "Correlation unstable" in reason:
                    key = "correlation_unstable"
                elif "Correlation declining" in reason:
                    key = "correlation_declining"
                elif "Volatility ratio unstable" in reason:
                    key = "volatility_ratio_unstable"
                elif "Spread variance exploding" in reason:
                    key = "spread_variance_exploding"
                elif "Insufficient history" in reason:
                    key = "insufficient_history"
                else:
                    key = "other"
                reasons[key] = reasons.get(key, 0) + 1
        return reasons
