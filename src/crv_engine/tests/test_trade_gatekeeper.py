"""
Unit Tests for Trade Gatekeeper

Tests verify that the gatekeeper correctly blocks trades in
empirically proven failure regimes.
"""

import pytest
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.trade_gatekeeper import (
    TradeGatekeeper,
    TradePermission,
    BlockReason,
    check_trade_permission,
    create_gatekeeper,
)
from experiments.edge_boundary import PredictionObservables


# =============================================================================
# HELPER FACTORIES
# =============================================================================

def make_observables(
    zscore: float = 1.5,
    correlation_trend: float = 0.0,
    volatility_ratio: float = 1.0,
    correlation: float = 0.8,
) -> PredictionObservables:
    """Create observables with specified values, defaults are safe."""
    return PredictionObservables(
        prediction_id="test",
        outcome="PENDING",
        bars_to_resolution=0,
        correlation=correlation,
        correlation_trend=correlation_trend,
        volatility_ratio=volatility_ratio,
        zscore=zscore,
        spread_velocity=0.0,
    )


# =============================================================================
# TEST: INDIVIDUAL BLOCKING CONDITIONS
# =============================================================================

class TestExtremeSpreadBlocking:
    """Tests for EXTREME_SPREAD blocking rule (|Z| > 3.0)."""
    
    def test_blocks_positive_extreme_zscore(self):
        gk = TradeGatekeeper()
        obs = make_observables(zscore=3.5)
        permission = gk.evaluate(obs)
        assert permission.is_blocked
        assert BlockReason.EXTREME_SPREAD in permission.reasons
    
    def test_blocks_negative_extreme_zscore(self):
        gk = TradeGatekeeper()
        obs = make_observables(zscore=-3.5)
        permission = gk.evaluate(obs)
        assert permission.is_blocked
        assert BlockReason.EXTREME_SPREAD in permission.reasons
    
    def test_allows_zscore_at_threshold(self):
        gk = TradeGatekeeper()
        obs = make_observables(zscore=3.0)
        permission = gk.evaluate(obs)
        assert BlockReason.EXTREME_SPREAD not in permission.reasons
    
    def test_allows_zscore_below_threshold(self):
        gk = TradeGatekeeper()
        obs = make_observables(zscore=2.9)
        permission = gk.evaluate(obs)
        assert BlockReason.EXTREME_SPREAD not in permission.reasons
    
    def test_blocks_very_extreme_zscore(self):
        gk = TradeGatekeeper()
        obs = make_observables(zscore=5.0)
        permission = gk.evaluate(obs)
        assert permission.is_blocked
        assert BlockReason.EXTREME_SPREAD in permission.reasons


class TestDeterioratingCorrelationBlocking:
    """Tests for DETERIORATING_CORRELATION blocking rule."""
    
    def test_blocks_deteriorating_correlation(self):
        gk = TradeGatekeeper()
        obs = make_observables(correlation_trend=-0.10)
        permission = gk.evaluate(obs)
        assert permission.is_blocked
        assert BlockReason.DETERIORATING_CORRELATION in permission.reasons
    
    def test_allows_stable_correlation(self):
        gk = TradeGatekeeper()
        obs = make_observables(correlation_trend=0.0)
        permission = gk.evaluate(obs)
        assert BlockReason.DETERIORATING_CORRELATION not in permission.reasons
    
    def test_allows_improving_correlation(self):
        gk = TradeGatekeeper()
        obs = make_observables(correlation_trend=0.10)
        permission = gk.evaluate(obs)
        assert BlockReason.DETERIORATING_CORRELATION not in permission.reasons
    
    def test_allows_slightly_negative_trend(self):
        gk = TradeGatekeeper()
        obs = make_observables(correlation_trend=-0.03)
        permission = gk.evaluate(obs)
        assert BlockReason.DETERIORATING_CORRELATION not in permission.reasons
    
    def test_blocks_at_deteriorating_threshold(self):
        gk = TradeGatekeeper()
        obs = make_observables(correlation_trend=-0.06)
        permission = gk.evaluate(obs)
        assert BlockReason.DETERIORATING_CORRELATION in permission.reasons


class TestCompressedVolatilityBlocking:
    """Tests for COMPRESSED_VOLATILITY blocking rule."""
    
    def test_blocks_compressed_volatility(self):
        gk = TradeGatekeeper()
        obs = make_observables(volatility_ratio=0.5)
        permission = gk.evaluate(obs)
        assert permission.is_blocked
        assert BlockReason.COMPRESSED_VOLATILITY in permission.reasons
    
    def test_allows_balanced_volatility(self):
        gk = TradeGatekeeper()
        obs = make_observables(volatility_ratio=1.0)
        permission = gk.evaluate(obs)
        assert BlockReason.COMPRESSED_VOLATILITY not in permission.reasons
    
    def test_allows_imbalanced_volatility(self):
        gk = TradeGatekeeper()
        obs = make_observables(volatility_ratio=1.5)
        permission = gk.evaluate(obs)
        assert BlockReason.COMPRESSED_VOLATILITY not in permission.reasons
    
    def test_allows_volatility_at_threshold(self):
        gk = TradeGatekeeper()
        obs = make_observables(volatility_ratio=0.7)
        permission = gk.evaluate(obs)
        assert BlockReason.COMPRESSED_VOLATILITY not in permission.reasons
    
    def test_blocks_just_below_threshold(self):
        gk = TradeGatekeeper()
        obs = make_observables(volatility_ratio=0.69)
        permission = gk.evaluate(obs)
        assert BlockReason.COMPRESSED_VOLATILITY in permission.reasons


# =============================================================================
# TEST: MULTIPLE CONDITIONS
# =============================================================================

class TestMultipleBlockingConditions:
    """Tests for multiple simultaneous blocking conditions."""
    
    def test_two_conditions_both_reported(self):
        gk = TradeGatekeeper()
        obs = make_observables(zscore=4.0, correlation_trend=-0.15)
        permission = gk.evaluate(obs)
        assert permission.is_blocked
        assert len(permission.reasons) == 2
        assert BlockReason.EXTREME_SPREAD in permission.reasons
        assert BlockReason.DETERIORATING_CORRELATION in permission.reasons
    
    def test_all_three_conditions(self):
        gk = TradeGatekeeper()
        obs = make_observables(zscore=3.5, correlation_trend=-0.10, volatility_ratio=0.5)
        permission = gk.evaluate(obs)
        assert permission.is_blocked
        assert len(permission.reasons) == 3
    
    def test_extreme_and_compressed(self):
        gk = TradeGatekeeper()
        obs = make_observables(zscore=-3.5, volatility_ratio=0.4)
        permission = gk.evaluate(obs)
        assert permission.is_blocked
        assert BlockReason.EXTREME_SPREAD in permission.reasons
        assert BlockReason.COMPRESSED_VOLATILITY in permission.reasons
    
    def test_deteriorating_and_compressed(self):
        gk = TradeGatekeeper()
        obs = make_observables(correlation_trend=-0.20, volatility_ratio=0.3)
        permission = gk.evaluate(obs)
        assert permission.is_blocked
        assert BlockReason.DETERIORATING_CORRELATION in permission.reasons
        assert BlockReason.COMPRESSED_VOLATILITY in permission.reasons


# =============================================================================
# TEST: FULL ALLOW CASE
# =============================================================================

class TestAllowCases:
    """Tests for fully allowed trade scenarios."""
    
    def test_all_conditions_safe(self):
        gk = TradeGatekeeper()
        obs = make_observables(zscore=1.5, correlation_trend=0.02, volatility_ratio=1.0)
        permission = gk.evaluate(obs)
        assert permission.allowed
        assert len(permission.reasons) == 0
    
    def test_borderline_safe(self):
        gk = TradeGatekeeper()
        obs = make_observables(zscore=2.99, correlation_trend=-0.04, volatility_ratio=0.71)
        permission = gk.evaluate(obs)
        assert permission.allowed
    
    def test_very_safe_values(self):
        gk = TradeGatekeeper()
        obs = make_observables(zscore=0.5, correlation_trend=0.10, volatility_ratio=1.2)
        permission = gk.evaluate(obs)
        assert permission.allowed
        assert str(permission) == "ALLOWED"


# =============================================================================
# TEST: PERMISSION OBJECT
# =============================================================================

class TestTradePermission:
    """Tests for TradePermission object behavior."""
    
    def test_allowed_permission_properties(self):
        gk = TradeGatekeeper()
        obs = make_observables()
        permission = gk.evaluate(obs)
        assert permission.allowed
        assert not permission.is_blocked
        assert permission.reasons == ()
        assert permission.reason_labels == []
    
    def test_blocked_permission_properties(self):
        gk = TradeGatekeeper()
        obs = make_observables(zscore=4.0)
        permission = gk.evaluate(obs)
        assert not permission.allowed
        assert permission.is_blocked
        assert "EXTREME_SPREAD" in permission.reason_labels
    
    def test_permission_snapshot(self):
        gk = TradeGatekeeper()
        obs = make_observables(zscore=2.5, volatility_ratio=1.1)
        permission = gk.evaluate(obs)
        assert permission.observables_snapshot['zscore'] == 2.5
        assert permission.observables_snapshot['volatility_ratio'] == 1.1
    
    def test_permission_string_blocked(self):
        gk = TradeGatekeeper()
        obs = make_observables(zscore=4.0)
        permission = gk.evaluate(obs)
        assert "BLOCKED" in str(permission)
        assert "EXTREME_SPREAD" in str(permission)


# =============================================================================
# TEST: STATISTICS
# =============================================================================

class TestGatekeeperStatistics:
    """Tests for gatekeeper statistics tracking."""
    
    def test_evaluation_count(self):
        gk = TradeGatekeeper()
        gk.evaluate(make_observables())
        gk.evaluate(make_observables())
        gk.evaluate(make_observables())
        assert gk.evaluation_count == 3
    
    def test_block_count(self):
        gk = TradeGatekeeper()
        gk.evaluate(make_observables(zscore=1.0))
        gk.evaluate(make_observables(zscore=4.0))
        gk.evaluate(make_observables(zscore=1.0))
        gk.evaluate(make_observables(zscore=-4.0))
        assert gk.block_count == 2
        assert gk.allow_count == 2
    
    def test_block_rate(self):
        gk = TradeGatekeeper()
        gk.evaluate(make_observables(zscore=1.0))
        gk.evaluate(make_observables(zscore=4.0))
        gk.evaluate(make_observables(zscore=1.0))
        gk.evaluate(make_observables(zscore=4.0))
        assert gk.block_rate == 0.5
    
    def test_reason_distribution(self):
        gk = TradeGatekeeper()
        gk.evaluate(make_observables(zscore=4.0))
        gk.evaluate(make_observables(zscore=4.0))
        gk.evaluate(make_observables(correlation_trend=-0.10))
        dist = gk.get_block_reason_distribution()
        assert dist['EXTREME_SPREAD'] == 2
        assert dist['DETERIORATING_CORRELATION'] == 1
        assert dist['COMPRESSED_VOLATILITY'] == 0
    
    def test_reset_statistics(self):
        gk = TradeGatekeeper()
        gk.evaluate(make_observables(zscore=4.0))
        gk.evaluate(make_observables(zscore=4.0))
        gk.reset_statistics()
        assert gk.evaluation_count == 0
        assert gk.block_count == 0
    
    def test_get_summary(self):
        gk = TradeGatekeeper()
        gk.evaluate(make_observables(zscore=1.0))
        gk.evaluate(make_observables(zscore=4.0))
        summary = gk.get_summary()
        assert summary['total_evaluations'] == 2
        assert summary['blocked'] == 1
        assert summary['allowed'] == 1


# =============================================================================
# TEST: RAW EVALUATION
# =============================================================================

class TestRawEvaluation:
    """Tests for evaluate_raw convenience method."""
    
    def test_raw_evaluation_allows(self):
        gk = TradeGatekeeper()
        permission = gk.evaluate_raw(zscore=1.5, correlation_trend=0.0, volatility_ratio=1.0)
        assert permission.allowed
    
    def test_raw_evaluation_blocks(self):
        gk = TradeGatekeeper()
        permission = gk.evaluate_raw(zscore=4.0, correlation_trend=0.0, volatility_ratio=1.0)
        assert permission.is_blocked


# =============================================================================
# TEST: CONVENIENCE FUNCTIONS
# =============================================================================

class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""
    
    def test_check_trade_permission_allows(self):
        assert check_trade_permission(zscore=1.5, correlation_trend=0.0, volatility_ratio=1.0) is True
    
    def test_check_trade_permission_blocks_extreme(self):
        assert check_trade_permission(zscore=4.0, correlation_trend=0.0, volatility_ratio=1.0) is False
    
    def test_check_trade_permission_blocks_deteriorating(self):
        assert check_trade_permission(zscore=1.5, correlation_trend=-0.10, volatility_ratio=1.0) is False
    
    def test_check_trade_permission_blocks_compressed(self):
        assert check_trade_permission(zscore=1.5, correlation_trend=0.0, volatility_ratio=0.5) is False
    
    def test_create_gatekeeper(self):
        gk = create_gatekeeper()
        permission = gk.evaluate(make_observables())
        assert permission.allowed


# =============================================================================
# TEST: EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_zero_zscore(self):
        gk = TradeGatekeeper()
        obs = make_observables(zscore=0.0)
        permission = gk.evaluate(obs)
        assert permission.allowed
    
    def test_negative_volatility_ratio(self):
        gk = TradeGatekeeper()
        obs = make_observables(volatility_ratio=-0.5)
        permission = gk.evaluate(obs)
        assert BlockReason.COMPRESSED_VOLATILITY in permission.reasons
    
    def test_very_large_positive_trend(self):
        gk = TradeGatekeeper()
        obs = make_observables(correlation_trend=0.50)
        permission = gk.evaluate(obs)
        assert BlockReason.DETERIORATING_CORRELATION not in permission.reasons
    
    def test_multiple_evaluations_independent(self):
        gk = TradeGatekeeper()
        p1 = gk.evaluate(make_observables(zscore=4.0))
        p2 = gk.evaluate(make_observables(zscore=1.0))
        p3 = gk.evaluate(make_observables(zscore=4.0))
        assert p1.is_blocked
        assert p2.allowed
        assert p3.is_blocked
