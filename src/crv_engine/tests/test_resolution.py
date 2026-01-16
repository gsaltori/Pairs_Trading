"""
Unit tests for P1 resolution logic.

Verifies that resolution rules are implemented EXACTLY as specified.
"""

import pytest
from datetime import datetime, timezone
from dataclasses import dataclass

import sys
sys.path.insert(0, '..')

from config import CONFIG
from predictions import (
    P1_SpreadReversionPrediction,
    PredictionType,
    PredictionDirection,
    HypothesisContext,
)
from resolution import ResolutionState, P1_Resolver
from spread import SpreadObservation


def make_prediction(
    zscore: float = 2.0,
    correlation: float = 0.70,
) -> P1_SpreadReversionPrediction:
    """Helper to create test predictions."""
    context = HypothesisContext(
        zscore_at_creation=zscore,
        spread_at_creation=0.001,
        hedge_ratio_at_creation=0.88,
        price_a_at_creation=1.1000,
        price_b_at_creation=1.2500,
        spread_mean_at_creation=0.0,
        spread_std_at_creation=0.0005,
        correlation_at_creation=correlation,
        zscore_sign=1 if zscore > 0 else -1,
    )
    
    return P1_SpreadReversionPrediction(
        prediction_id="TEST-001",
        prediction_type=PredictionType.P1_SPREAD_REVERSION,
        creation_timestamp=datetime.now(timezone.utc),
        creation_observation_id="OBS-TEST-001",
        creation_bar_index=100,
        pair=("EURUSD", "GBPUSD"),
        timeframe="H4",
        prediction=PredictionDirection.REVERT_TOWARD_MEAN,
        context=context,
    )


def make_spread_obs(
    zscore: float = 0.0,
    correlation: float = 0.70,
) -> SpreadObservation:
    """Helper to create test spread observations."""
    return SpreadObservation(
        observation_id="OBS-CURRENT",
        timestamp=datetime.now(timezone.utc),
        timeframe="H4",
        pair=("EURUSD", "GBPUSD"),
        price_a=1.1000,
        price_b=1.2500,
        hedge_ratio=0.88,
        spread_value=0.001,
        spread_mean=0.0,
        spread_std=0.0005,
        zscore=zscore,
        correlation=correlation,
        is_valid=True,
    )


class TestP1Resolution:
    """Test P1 resolution logic."""
    
    def setup_method(self):
        """Setup for each test."""
        self.resolver = P1_Resolver()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CONFIRMATION TESTS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def test_confirmed_when_zscore_below_threshold(self):
        """CONFIRMED when |Z| < 0.3"""
        pred = make_prediction(zscore=2.0)
        spread = make_spread_obs(zscore=0.2)
        
        result = self.resolver.evaluate(pred, spread, bars_elapsed=10)
        
        assert result == ResolutionState.CONFIRMED
    
    def test_confirmed_at_zero(self):
        """CONFIRMED when Z = 0"""
        pred = make_prediction(zscore=2.0)
        spread = make_spread_obs(zscore=0.0)
        
        result = self.resolver.evaluate(pred, spread, bars_elapsed=10)
        
        assert result == ResolutionState.CONFIRMED
    
    def test_confirmed_negative_zscore_reverting(self):
        """CONFIRMED when negative Z reverts toward zero"""
        pred = make_prediction(zscore=-2.0)
        spread = make_spread_obs(zscore=-0.1)
        
        result = self.resolver.evaluate(pred, spread, bars_elapsed=10)
        
        assert result == ResolutionState.CONFIRMED
    
    # ═══════════════════════════════════════════════════════════════════════════
    # REFUTATION TESTS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def test_refuted_positive_zscore_continues(self):
        """REFUTED when positive Z continues beyond 3.0"""
        pred = make_prediction(zscore=2.0)  # Initial Z = +2.0
        spread = make_spread_obs(zscore=3.5)  # Current Z = +3.5
        
        result = self.resolver.evaluate(pred, spread, bars_elapsed=10)
        
        assert result == ResolutionState.REFUTED
    
    def test_refuted_negative_zscore_continues(self):
        """REFUTED when negative Z continues beyond -3.0"""
        pred = make_prediction(zscore=-2.0)  # Initial Z = -2.0
        spread = make_spread_obs(zscore=-3.5)  # Current Z = -3.5
        
        result = self.resolver.evaluate(pred, spread, bars_elapsed=10)
        
        assert result == ResolutionState.REFUTED
    
    def test_not_refuted_opposite_direction(self):
        """NOT REFUTED when Z moves in opposite direction beyond threshold"""
        pred = make_prediction(zscore=2.0)  # Initial Z = +2.0
        spread = make_spread_obs(zscore=-3.5)  # Current Z = -3.5 (opposite)
        
        result = self.resolver.evaluate(pred, spread, bars_elapsed=10)
        
        # This should NOT be REFUTED because Z moved in opposite direction
        # It's not confirmed because |Z| = 3.5 > 0.3
        # And it's not refuted because direction changed
        # So it's pending
        assert result is None
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TIMEOUT TESTS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def test_timeout_at_max_bars(self):
        """TIMEOUT when bars_elapsed >= max_holding_bars"""
        pred = make_prediction(zscore=2.0)
        spread = make_spread_obs(zscore=1.0)  # Neither confirmed nor refuted
        
        result = self.resolver.evaluate(pred, spread, bars_elapsed=50)
        
        assert result == ResolutionState.TIMEOUT
    
    def test_no_timeout_before_max(self):
        """No TIMEOUT before max_holding_bars"""
        pred = make_prediction(zscore=2.0)
        spread = make_spread_obs(zscore=1.0)
        
        result = self.resolver.evaluate(pred, spread, bars_elapsed=49)
        
        assert result is None  # Still pending
    
    # ═══════════════════════════════════════════════════════════════════════════
    # INVALIDATION TESTS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def test_invalidated_correlation_collapsed(self):
        """INVALIDATED when correlation < 0.20"""
        pred = make_prediction(zscore=2.0, correlation=0.70)
        spread = make_spread_obs(zscore=1.0, correlation=0.15)
        
        result = self.resolver.evaluate(pred, spread, bars_elapsed=10)
        
        assert result == ResolutionState.INVALIDATED
    
    def test_invalidated_correlation_sign_change(self):
        """INVALIDATED when correlation changes sign"""
        pred = make_prediction(zscore=2.0, correlation=0.70)
        spread = make_spread_obs(zscore=1.0, correlation=-0.10)
        
        result = self.resolver.evaluate(pred, spread, bars_elapsed=10)
        
        assert result == ResolutionState.INVALIDATED
    
    def test_invalidated_correlation_large_drop(self):
        """INVALIDATED when correlation drops > 0.30 from initial"""
        pred = make_prediction(zscore=2.0, correlation=0.70)
        spread = make_spread_obs(zscore=1.0, correlation=0.35)  # Drop of 0.35 > 0.30
        
        result = self.resolver.evaluate(pred, spread, bars_elapsed=10)
        
        assert result == ResolutionState.INVALIDATED
    
    def test_invalidation_takes_priority(self):
        """INVALIDATED takes priority over CONFIRMED"""
        pred = make_prediction(zscore=2.0, correlation=0.70)
        # Z-score is in confirmation range, but correlation collapsed
        spread = make_spread_obs(zscore=0.1, correlation=0.15)
        
        result = self.resolver.evaluate(pred, spread, bars_elapsed=10)
        
        assert result == ResolutionState.INVALIDATED
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PENDING TESTS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def test_pending_when_no_conditions_met(self):
        """PENDING when no resolution conditions are met"""
        pred = make_prediction(zscore=2.0, correlation=0.70)
        spread = make_spread_obs(zscore=1.5, correlation=0.65)  # Still dislocated
        
        result = self.resolver.evaluate(pred, spread, bars_elapsed=10)
        
        assert result is None  # Still pending
    
    # ═══════════════════════════════════════════════════════════════════════════
    # RESOLUTION ORDER TESTS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def test_invalidation_before_confirmation(self):
        """INVALIDATED is checked before CONFIRMED"""
        pred = make_prediction(zscore=2.0, correlation=0.70)
        # Both conditions met: Z is in confirmation range AND correlation collapsed
        spread = make_spread_obs(zscore=0.1, correlation=0.10)
        
        result = self.resolver.evaluate(pred, spread, bars_elapsed=10)
        
        # INVALIDATED should win because it's checked first
        assert result == ResolutionState.INVALIDATED
    
    def test_confirmation_before_refutation(self):
        """CONFIRMED is checked before REFUTED (but they're mutually exclusive)"""
        # This test verifies the logic: if |Z| < 0.3, we CONFIRM
        # Refutation requires |Z| > 3.0 in same direction
        # These conditions cannot both be true
        pred = make_prediction(zscore=2.0)
        spread = make_spread_obs(zscore=0.2)  # |Z| < 0.3
        
        result = self.resolver.evaluate(pred, spread, bars_elapsed=10)
        
        assert result == ResolutionState.CONFIRMED
    
    def test_confirmation_before_timeout(self):
        """CONFIRMED is checked before TIMEOUT"""
        pred = make_prediction(zscore=2.0)
        spread = make_spread_obs(zscore=0.2)  # Confirmation condition
        
        # Even at max bars, confirmation should happen
        result = self.resolver.evaluate(pred, spread, bars_elapsed=50)
        
        assert result == ResolutionState.CONFIRMED


class TestResolutionImmutability:
    """Test that resolved predictions cannot be re-resolved."""
    
    def test_cannot_resolve_twice(self):
        """Once resolved, prediction cannot be resolved again."""
        pred = make_prediction(zscore=2.0)
        spread = make_spread_obs(zscore=0.2)
        
        # First resolution
        pred.resolve(
            state=ResolutionState.CONFIRMED,
            timestamp=spread.timestamp,
            observation_id=spread.observation_id,
            bar_index=110,
            bars_elapsed=10,
            zscore_final=0.2,
            spread_final=0.001,
            correlation_final=0.70,
        )
        
        # Attempt second resolution
        with pytest.raises(RuntimeError):
            pred.resolve(
                state=ResolutionState.REFUTED,
                timestamp=spread.timestamp,
                observation_id=spread.observation_id,
                bar_index=120,
                bars_elapsed=20,
                zscore_final=3.5,
                spread_final=0.002,
                correlation_final=0.65,
            )


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def setup_method(self):
        """Setup for each test."""
        self.resolver = P1_Resolver()
    
    def test_zscore_exactly_at_confirmation_threshold(self):
        """Z-score exactly at confirmation threshold (0.3)"""
        pred = make_prediction(zscore=2.0)
        spread = make_spread_obs(zscore=0.3)  # Exactly at threshold
        
        result = self.resolver.evaluate(pred, spread, bars_elapsed=10)
        
        # |Z| = 0.3 is NOT < 0.3, so should NOT be confirmed
        assert result is None
    
    def test_zscore_just_below_confirmation_threshold(self):
        """Z-score just below confirmation threshold"""
        pred = make_prediction(zscore=2.0)
        spread = make_spread_obs(zscore=0.29)  # Just below threshold
        
        result = self.resolver.evaluate(pred, spread, bars_elapsed=10)
        
        assert result == ResolutionState.CONFIRMED
    
    def test_zscore_exactly_at_refutation_threshold(self):
        """Z-score exactly at refutation threshold (3.0)"""
        pred = make_prediction(zscore=2.0)
        spread = make_spread_obs(zscore=3.0)  # Exactly at threshold
        
        result = self.resolver.evaluate(pred, spread, bars_elapsed=10)
        
        # Z = 3.0 is NOT > 3.0, so should NOT be refuted
        assert result is None
    
    def test_zscore_just_above_refutation_threshold(self):
        """Z-score just above refutation threshold"""
        pred = make_prediction(zscore=2.0)
        spread = make_spread_obs(zscore=3.01)  # Just above threshold
        
        result = self.resolver.evaluate(pred, spread, bars_elapsed=10)
        
        assert result == ResolutionState.REFUTED
    
    def test_correlation_exactly_at_minimum(self):
        """Correlation exactly at minimum threshold (0.20)"""
        # Use initial correlation of 0.40 so the drop is only 0.20 (< 0.30 max drop)
        pred = make_prediction(zscore=2.0, correlation=0.40)
        spread = make_spread_obs(zscore=1.0, correlation=0.20)  # Exactly at threshold
        
        result = self.resolver.evaluate(pred, spread, bars_elapsed=10)
        
        # Correlation = 0.20 is NOT < 0.20, and drop of 0.20 is NOT > 0.30
        # So should NOT be invalidated
        assert result is None
    
    def test_correlation_just_below_minimum(self):
        """Correlation just below minimum threshold"""
        pred = make_prediction(zscore=2.0, correlation=0.70)
        spread = make_spread_obs(zscore=1.0, correlation=0.19)  # Just below threshold
        
        result = self.resolver.evaluate(pred, spread, bars_elapsed=10)
        
        assert result == ResolutionState.INVALIDATED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
