"""
Unit tests for P5 Regime Memory Module.

Tests cover:
1. Cold start behavior
2. Regime learning
3. Regime decay / forgetting (FIFO)
4. False confidence prevention
5. Wilson score calculation
6. Outcome recording
"""

import pytest
from datetime import datetime, timezone, timedelta
import math

import sys
sys.path.insert(0, '..')

from regime import (
    CorrelationStability,
    CorrelationTrend,
    VolatilityStability,
    SpreadDynamics,
    OutcomeType,
    RegimeConfig,
    RegimeSignature,
    RegimeOutcome,
    RegimeStats,
    RegimeMemory,
    RegimeEvaluation,
    RegimeEvaluator,
    OutcomeRecorder,
    create_regime_signature,
    REGIME_CONFIG,
)


def make_timestamp(bar_index: int) -> datetime:
    """Create timestamp for a given bar index."""
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return base + timedelta(hours=4 * bar_index)


def make_regime(
    cs: str = "STABLE",
    ct: str = "NEUTRAL",
    vs: str = "STABLE",
    sd: str = "NORMAL",
) -> RegimeSignature:
    """Create a regime signature from string codes."""
    return RegimeSignature(
        correlation_stability=CorrelationStability[cs],
        correlation_trend=CorrelationTrend[ct],
        volatility_stability=VolatilityStability[vs],
        spread_dynamics=SpreadDynamics[sd],
    )


def make_outcome(
    regime: RegimeSignature,
    outcome: OutcomeType,
    prediction_id: str = "test",
    bar_index: int = 0,
) -> RegimeOutcome:
    """Create a regime outcome."""
    return RegimeOutcome(
        prediction_id=prediction_id,
        creation_timestamp=make_timestamp(bar_index),
        resolution_timestamp=make_timestamp(bar_index + 10),
        regime=regime,
        outcome=outcome,
        bars_to_resolution=10,
    )


class TestRegimeSignature:
    """Test RegimeSignature creation and properties."""
    
    def test_signature_is_hashable(self):
        """Signatures should be hashable for use as dict keys."""
        regime = make_regime()
        d = {regime: "test"}
        assert d[regime] == "test"
    
    def test_equal_signatures_match(self):
        """Equal signatures should be equal."""
        r1 = make_regime("STABLE", "NEUTRAL", "STABLE", "NORMAL")
        r2 = make_regime("STABLE", "NEUTRAL", "STABLE", "NORMAL")
        assert r1 == r2
        assert hash(r1) == hash(r2)
    
    def test_different_signatures_dont_match(self):
        """Different signatures should not be equal."""
        r1 = make_regime("STABLE", "NEUTRAL", "STABLE", "NORMAL")
        r2 = make_regime("UNSTABLE", "NEUTRAL", "STABLE", "NORMAL")
        assert r1 != r2
    
    def test_short_code_format(self):
        """Short code should be 4 characters."""
        regime = make_regime("STABLE", "DECLINING", "MODERATE", "EXPANDING")
        code = regime.short_code()
        assert len(code) == 4
        assert code == "SDME"


class TestCreateRegimeSignature:
    """Test regime signature creation from continuous values."""
    
    def test_stable_correlation(self):
        """Low correlation std should map to STABLE."""
        regime = create_regime_signature(0.03, 0.0, 0.05, 1.0)
        assert regime.correlation_stability == CorrelationStability.STABLE
    
    def test_unstable_correlation(self):
        """High correlation std should map to UNSTABLE."""
        regime = create_regime_signature(0.15, 0.0, 0.05, 1.0)
        assert regime.correlation_stability == CorrelationStability.UNSTABLE
    
    def test_declining_trend(self):
        """Negative trend should map to DECLINING."""
        regime = create_regime_signature(0.05, -0.01, 0.05, 1.0)
        assert regime.correlation_trend == CorrelationTrend.DECLINING
    
    def test_improving_trend(self):
        """Positive trend should map to IMPROVING."""
        regime = create_regime_signature(0.05, 0.01, 0.05, 1.0)
        assert regime.correlation_trend == CorrelationTrend.IMPROVING
    
    def test_expanding_spread(self):
        """High variance ratio should map to EXPANDING."""
        regime = create_regime_signature(0.05, 0.0, 0.05, 2.0)
        assert regime.spread_dynamics == SpreadDynamics.EXPANDING


class TestRegimeStats:
    """Test regime statistics calculations."""
    
    def test_empty_stats_returns_neutral_crr(self):
        """Empty stats should return 0.5 CRR."""
        stats = RegimeStats(regime=make_regime())
        assert stats.raw_confirmation_rate == 0.5
    
    def test_testable_count(self):
        """Testable count should be CONFIRMED + REFUTED."""
        stats = RegimeStats(
            regime=make_regime(),
            confirmed_count=10,
            refuted_count=5,
            untestable_count=3,
        )
        assert stats.testable_count == 15
    
    def test_raw_confirmation_rate(self):
        """CRR should be CONFIRMED / testable."""
        stats = RegimeStats(
            regime=make_regime(),
            confirmed_count=6,
            refuted_count=4,
        )
        assert stats.raw_confirmation_rate == 0.6
    
    def test_wilson_lower_bound_penalizes_low_n(self):
        """Wilson score should be lower for small samples."""
        # Same ratio, different N
        stats_small = RegimeStats(regime=make_regime(), confirmed_count=3, refuted_count=2)
        stats_large = RegimeStats(regime=make_regime(), confirmed_count=30, refuted_count=20)
        
        assert stats_small.raw_confirmation_rate == stats_large.raw_confirmation_rate
        assert stats_small.wilson_confidence_lower() < stats_large.wilson_confidence_lower()
    
    def test_has_sufficient_history(self):
        """Sufficient history check."""
        stats_insufficient = RegimeStats(regime=make_regime(), confirmed_count=3, refuted_count=2)
        stats_sufficient = RegimeStats(regime=make_regime(), confirmed_count=6, refuted_count=5)
        
        assert not stats_insufficient.has_sufficient_history
        assert stats_sufficient.has_sufficient_history


class TestRegimeMemory:
    """Test regime memory operations."""
    
    def test_empty_memory(self):
        """Empty memory should have zero outcomes."""
        memory = RegimeMemory()
        assert memory.total_outcomes == 0
        assert memory.unique_regimes == 0
    
    def test_record_outcome(self):
        """Recording should increase count."""
        memory = RegimeMemory()
        regime = make_regime()
        outcome = make_outcome(regime, OutcomeType.CONFIRMED)
        
        memory.record(outcome)
        
        assert memory.total_outcomes == 1
    
    def test_get_stats_for_regime(self):
        """Should return stats for specific regime."""
        memory = RegimeMemory()
        regime = make_regime()
        
        memory.record(make_outcome(regime, OutcomeType.CONFIRMED, "p1"))
        memory.record(make_outcome(regime, OutcomeType.CONFIRMED, "p2"))
        memory.record(make_outcome(regime, OutcomeType.REFUTED, "p3"))
        
        stats = memory.get_stats(regime)
        
        assert stats.confirmed_count == 2
        assert stats.refuted_count == 1
    
    def test_fifo_eviction(self):
        """Memory should evict oldest when full."""
        memory = RegimeMemory(max_size=5)
        regime = make_regime()
        
        for i in range(10):
            outcome = make_outcome(regime, OutcomeType.CONFIRMED, f"p{i}", i)
            memory.record(outcome)
        
        assert memory.total_outcomes == 5  # Only kept last 5
    
    def test_multiple_regimes(self):
        """Should track multiple regimes separately."""
        memory = RegimeMemory()
        
        r1 = make_regime("STABLE", "NEUTRAL", "STABLE", "NORMAL")
        r2 = make_regime("UNSTABLE", "DECLINING", "UNSTABLE", "EXPANDING")
        
        memory.record(make_outcome(r1, OutcomeType.CONFIRMED, "p1"))
        memory.record(make_outcome(r1, OutcomeType.CONFIRMED, "p2"))
        memory.record(make_outcome(r2, OutcomeType.REFUTED, "p3"))
        
        assert memory.unique_regimes == 2
        assert memory.get_stats(r1).confirmed_count == 2
        assert memory.get_stats(r2).refuted_count == 1
    
    def test_clear(self):
        """Clear should remove all data."""
        memory = RegimeMemory()
        regime = make_regime()
        memory.record(make_outcome(regime, OutcomeType.CONFIRMED))
        
        memory.clear()
        
        assert memory.total_outcomes == 0


class TestRegimeEvaluator:
    """Test regime evaluator gating logic."""
    
    def test_cold_start_allows(self):
        """Cold start (insufficient data) should ALLOW."""
        memory = RegimeMemory()
        evaluator = RegimeEvaluator(memory)
        regime = make_regime()
        
        # No data at all
        eval_result = evaluator.evaluate(regime)
        
        assert eval_result.is_allowed
        assert "cold_start" in eval_result.reason
    
    def test_cold_start_with_some_data(self):
        """Cold start with N < min should still ALLOW."""
        memory = RegimeMemory()
        evaluator = RegimeEvaluator(memory)
        regime = make_regime()
        
        # Add 5 outcomes (less than MIN_SAMPLES_FOR_CONFIDENCE=10)
        for i in range(5):
            memory.record(make_outcome(regime, OutcomeType.CONFIRMED, f"p{i}"))
        
        eval_result = evaluator.evaluate(regime)
        
        assert eval_result.is_allowed
        assert "cold_start" in eval_result.reason
    
    def test_high_confidence_allows(self):
        """High confidence regime should be ALLOWED."""
        memory = RegimeMemory()
        evaluator = RegimeEvaluator(memory)
        regime = make_regime()
        
        # Add 15 outcomes with 80% confirmation rate
        for i in range(12):
            memory.record(make_outcome(regime, OutcomeType.CONFIRMED, f"c{i}"))
        for i in range(3):
            memory.record(make_outcome(regime, OutcomeType.REFUTED, f"r{i}"))
        
        eval_result = evaluator.evaluate(regime)
        
        assert eval_result.is_allowed
        assert "confidence_sufficient" in eval_result.reason
    
    def test_low_confidence_blocks(self):
        """Low confidence regime should be BLOCKED."""
        memory = RegimeMemory()
        evaluator = RegimeEvaluator(memory)
        regime = make_regime()
        
        # Add 15 outcomes with 20% confirmation rate
        for i in range(3):
            memory.record(make_outcome(regime, OutcomeType.CONFIRMED, f"c{i}"))
        for i in range(12):
            memory.record(make_outcome(regime, OutcomeType.REFUTED, f"r{i}"))
        
        eval_result = evaluator.evaluate(regime)
        
        assert not eval_result.is_allowed
        assert "confidence_insufficient" in eval_result.reason


class TestOutcomeRecorder:
    """Test outcome recorder tracking."""
    
    def test_track_and_record(self):
        """Should track prediction and record outcome."""
        memory = RegimeMemory()
        recorder = OutcomeRecorder(memory)
        regime = make_regime()
        
        # Track
        recorder.track_prediction("pred1", regime, make_timestamp(0))
        assert recorder.pending_count == 1
        
        # Record resolution
        from resolution import ResolutionState
        outcome = recorder.record_resolution(
            "pred1",
            ResolutionState.CONFIRMED,
            make_timestamp(10),
            10,
        )
        
        assert outcome is not None
        assert outcome.outcome == OutcomeType.CONFIRMED
        assert recorder.pending_count == 0
        assert memory.total_outcomes == 1
    
    def test_untracked_prediction_returns_none(self):
        """Recording untracked prediction should return None."""
        memory = RegimeMemory()
        recorder = OutcomeRecorder(memory)
        
        from resolution import ResolutionState
        outcome = recorder.record_resolution(
            "unknown",
            ResolutionState.CONFIRMED,
            make_timestamp(10),
            10,
        )
        
        assert outcome is None


class TestFalseConfidencePrevention:
    """Test that the system doesn't develop false confidence."""
    
    def test_untestable_not_counted(self):
        """UNTESTABLE outcomes should not affect confidence."""
        memory = RegimeMemory()
        evaluator = RegimeEvaluator(memory)
        regime = make_regime()
        
        # Add many untestable outcomes
        for i in range(20):
            memory.record(make_outcome(regime, OutcomeType.UNTESTABLE, f"u{i}"))
        
        stats = memory.get_stats(regime)
        
        assert stats.total_count == 20
        assert stats.testable_count == 0
        assert not stats.has_sufficient_history
    
    def test_mixed_regimes_isolated(self):
        """Different regimes should not cross-contaminate."""
        memory = RegimeMemory()
        evaluator = RegimeEvaluator(memory)
        
        good_regime = make_regime("STABLE", "NEUTRAL", "STABLE", "NORMAL")
        bad_regime = make_regime("UNSTABLE", "DECLINING", "UNSTABLE", "EXPANDING")
        
        # Good regime: 90% CRR
        for i in range(18):
            memory.record(make_outcome(good_regime, OutcomeType.CONFIRMED, f"gc{i}"))
        for i in range(2):
            memory.record(make_outcome(good_regime, OutcomeType.REFUTED, f"gr{i}"))
        
        # Bad regime: 10% CRR
        for i in range(2):
            memory.record(make_outcome(bad_regime, OutcomeType.CONFIRMED, f"bc{i}"))
        for i in range(18):
            memory.record(make_outcome(bad_regime, OutcomeType.REFUTED, f"br{i}"))
        
        good_eval = evaluator.evaluate(good_regime)
        bad_eval = evaluator.evaluate(bad_regime)
        
        assert good_eval.is_allowed
        assert not bad_eval.is_allowed


class TestRegimeDecay:
    """Test that old observations naturally age out."""
    
    def test_old_data_evicted(self):
        """Old data should be evicted as new data arrives."""
        memory = RegimeMemory(max_size=10)
        regime = make_regime()
        
        # Fill with CONFIRMED
        for i in range(10):
            memory.record(make_outcome(regime, OutcomeType.CONFIRMED, f"c{i}", i))
        
        stats_before = memory.get_stats(regime)
        assert stats_before.confirmed_count == 10
        
        # Add REFUTED (will evict old CONFIRMED)
        for i in range(10):
            memory.record(make_outcome(regime, OutcomeType.REFUTED, f"r{i}", 20+i))
        
        stats_after = memory.get_stats(regime)
        assert stats_after.refuted_count == 10
        assert stats_after.confirmed_count == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
