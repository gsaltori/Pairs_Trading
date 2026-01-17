"""
Unit Tests for Walk-Forward Falsification Engine

Tests cover:
1. Block boundary computation
2. Block isolation guarantees
3. Deterministic segmentation
4. Temporal stability metrics
5. False persistence prevention

NO STRATEGY LOGIC TESTED. ONLY INFRASTRUCTURE VALIDATION.
"""

import pytest
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.walk_forward import (
    WalkForwardConfig,
    BlockResult,
    BlockVerdict,
    RegimeDistribution,
    TemporalStabilityMetrics,
    EdgeStabilityClass,
    compute_temporal_stability,
    _linear_regression,
    _count_direction_changes,
    _is_monotonic_decreasing,
    _compute_decay_consistency,
)


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================

class TestWalkForwardConfig:
    """Test walk-forward configuration."""
    
    def test_valid_config_creation(self):
        """Valid config should be created without error."""
        config = WalkForwardConfig(
            block_size=500,
            min_bars_per_block=300,
            warmup_bars=60,
        )
        assert config.block_size == 500
        assert config.warmup_bars == 60
    
    def test_invalid_block_size_raises(self):
        """Block size smaller than warmup + min_bars should raise."""
        with pytest.raises(ValueError, match="block_size"):
            WalkForwardConfig(
                block_size=100,
                min_bars_per_block=300,
                warmup_bars=60,
            )
    
    def test_invalid_min_testable_raises(self):
        """min_testable_per_block < 5 should raise."""
        with pytest.raises(ValueError, match="min_testable_per_block"):
            WalkForwardConfig(
                block_size=500,
                min_testable_per_block=3,
            )
    
    def test_config_hash_deterministic(self):
        """Same config should produce same hash."""
        config1 = WalkForwardConfig(block_size=500)
        config2 = WalkForwardConfig(block_size=500)
        assert config1.config_hash == config2.config_hash
    
    def test_config_hash_changes_with_params(self):
        """Different configs should produce different hashes."""
        config1 = WalkForwardConfig(block_size=500)
        config2 = WalkForwardConfig(block_size=600)
        assert config1.config_hash != config2.config_hash


class TestBlockComputation:
    """Test block boundary computation."""
    
    def test_exact_division(self):
        """Total bars exactly divisible by block size."""
        config = WalkForwardConfig(block_size=500)
        blocks = config.compute_blocks(total_bars=1500)
        
        assert len(blocks) == 3
        assert blocks[0] == (0, 500)
        assert blocks[1] == (500, 1000)
        assert blocks[2] == (1000, 1500)
    
    def test_remainder_above_minimum(self):
        """Remainder above minimum should create partial block."""
        config = WalkForwardConfig(
            block_size=500,
            min_bars_per_block=300,
            warmup_bars=60,
        )
        # 1400 bars = 2 full blocks + 400 remainder (> 360 min)
        blocks = config.compute_blocks(total_bars=1400)
        
        assert len(blocks) == 3
        assert blocks[2] == (1000, 1400)
    
    def test_remainder_below_minimum_discarded(self):
        """Remainder below minimum should be discarded."""
        config = WalkForwardConfig(
            block_size=500,
            min_bars_per_block=300,
            warmup_bars=60,
        )
        # 1100 bars = 2 full blocks + 100 remainder (< 360 min)
        blocks = config.compute_blocks(total_bars=1100)
        
        assert len(blocks) == 2
        assert blocks[1] == (500, 1000)
    
    def test_non_overlapping_blocks(self):
        """Blocks must be non-overlapping."""
        config = WalkForwardConfig(block_size=500)
        blocks = config.compute_blocks(total_bars=2000)
        
        for i in range(len(blocks) - 1):
            assert blocks[i][1] == blocks[i+1][0], "Blocks must be contiguous"
            assert blocks[i][1] <= blocks[i+1][0], "Blocks must not overlap"
    
    def test_too_few_bars_for_one_block(self):
        """Insufficient bars for even one block should return empty."""
        config = WalkForwardConfig(block_size=500)
        blocks = config.compute_blocks(total_bars=200)
        
        assert len(blocks) == 0


# =============================================================================
# TEMPORAL STABILITY METRIC TESTS
# =============================================================================

class TestLinearRegression:
    """Test linear regression helper."""
    
    def test_perfect_positive_trend(self):
        """Perfect positive correlation."""
        x = [0.0, 1.0, 2.0, 3.0]
        y = [0.0, 1.0, 2.0, 3.0]
        
        intercept, slope, r2 = _linear_regression(x, y)
        
        assert abs(slope - 1.0) < 0.01
        assert abs(intercept) < 0.01
        assert abs(r2 - 1.0) < 0.01
    
    def test_perfect_negative_trend(self):
        """Perfect negative correlation."""
        x = [0.0, 1.0, 2.0, 3.0]
        y = [3.0, 2.0, 1.0, 0.0]
        
        intercept, slope, r2 = _linear_regression(x, y)
        
        assert abs(slope - (-1.0)) < 0.01
        assert abs(r2 - 1.0) < 0.01
    
    def test_no_trend(self):
        """Flat line should have zero slope."""
        x = [0.0, 1.0, 2.0, 3.0]
        y = [5.0, 5.0, 5.0, 5.0]
        
        intercept, slope, r2 = _linear_regression(x, y)
        
        assert abs(slope) < 0.01
        assert abs(intercept - 5.0) < 0.01
    
    def test_insufficient_data(self):
        """Single point should return zeros."""
        x = [0.0]
        y = [5.0]
        
        intercept, slope, r2 = _linear_regression(x, y)
        
        assert slope == 0.0
        assert r2 == 0.0


class TestDirectionChanges:
    """Test direction change detection."""
    
    def test_monotonic_increase(self):
        """No direction changes in monotonic increase."""
        values = [1.0, 2.0, 3.0, 4.0]
        assert _count_direction_changes(values) == 0
    
    def test_monotonic_decrease(self):
        """No direction changes in monotonic decrease."""
        values = [4.0, 3.0, 2.0, 1.0]
        assert _count_direction_changes(values) == 0
    
    def test_one_reversal(self):
        """Single reversal should count as 1."""
        values = [1.0, 2.0, 3.0, 2.0]  # Up then down
        assert _count_direction_changes(values) == 1
    
    def test_multiple_reversals(self):
        """Multiple reversals should count correctly."""
        values = [1.0, 2.0, 1.0, 2.0, 1.0]  # Up, down, up, down
        assert _count_direction_changes(values) == 3
    
    def test_insufficient_points(self):
        """Too few points should return 0."""
        assert _count_direction_changes([1.0]) == 0
        assert _count_direction_changes([1.0, 2.0]) == 0


class TestMonotonicDecay:
    """Test monotonic decay detection."""
    
    def test_strict_decrease(self):
        """Strictly decreasing should be monotonic decay."""
        values = [0.7, 0.6, 0.5, 0.4]
        assert _is_monotonic_decreasing(values) is True
    
    def test_increase_breaks_monotonicity(self):
        """Any increase should break monotonicity."""
        values = [0.7, 0.6, 0.65, 0.4]  # 0.65 > 0.6
        assert _is_monotonic_decreasing(values) is False
    
    def test_flat_allowed_within_tolerance(self):
        """Small increases within tolerance allowed."""
        values = [0.7, 0.6, 0.6005, 0.5]  # 0.6005 > 0.6 but within tolerance
        assert _is_monotonic_decreasing(values, tolerance=0.001) is True
    
    def test_insufficient_data(self):
        """Single point cannot be monotonic."""
        assert _is_monotonic_decreasing([0.5]) is False


class TestDecayConsistency:
    """Test decay consistency metric."""
    
    def test_pure_decay(self):
        """All decreases should give consistency = 1.0."""
        values = [0.8, 0.7, 0.6, 0.5]
        assert _compute_decay_consistency(values) == 1.0
    
    def test_pure_increase(self):
        """All increases should give consistency = 0.0."""
        values = [0.5, 0.6, 0.7, 0.8]
        assert _compute_decay_consistency(values) == 0.0
    
    def test_mixed(self):
        """Mixed should give intermediate value."""
        values = [0.6, 0.5, 0.7, 0.4]  # 2 decreases, 1 increase
        consistency = _compute_decay_consistency(values)
        assert 0.0 < consistency < 1.0
        assert abs(consistency - 2/3) < 0.01  # 2 decreases out of 3 comparisons


# =============================================================================
# BLOCK RESULT TESTS
# =============================================================================

class TestBlockResult:
    """Test BlockResult data structure."""
    
    def test_serialization(self):
        """BlockResult should serialize to dict correctly."""
        result = BlockResult(
            block_index=0,
            start_bar=0,
            end_bar=500,
            bar_count=500,
            total_predictions=50,
            testable_count=40,
            confirmed_count=24,
            refuted_count=16,
            invalidated_count=8,
            timeout_count=2,
            crr=0.60,
            erv=8.0,
            erv_per_prediction=0.16,
            invalidation_rate=0.16,
            median_bars_to_confirmation=12.0,
            median_bars_to_refutation=18.0,
            timing_asymmetry_ratio=0.67,
            regime_distribution=None,
            verdict=BlockVerdict.SUPPORTED,
            verdict_reason="CRR 60% >= 55%",
            crr_threshold=0.55,
            max_invalidation_threshold=0.40,
        )
        
        d = result.to_dict()
        
        assert d['block_index'] == 0
        assert d['crr'] == 0.60
        assert d['verdict'] == 'SUPPORTED'
    
    def test_is_sufficient_property(self):
        """is_sufficient should reflect verdict."""
        supported = BlockResult(
            block_index=0, start_bar=0, end_bar=500, bar_count=500,
            total_predictions=50, testable_count=40, confirmed_count=24,
            refuted_count=16, invalidated_count=8, timeout_count=2,
            crr=0.60, erv=8.0, erv_per_prediction=0.16, invalidation_rate=0.16,
            median_bars_to_confirmation=12.0, median_bars_to_refutation=18.0,
            timing_asymmetry_ratio=0.67, regime_distribution=None,
            verdict=BlockVerdict.SUPPORTED, verdict_reason="",
            crr_threshold=0.55, max_invalidation_threshold=0.40,
        )
        assert supported.is_sufficient is True
        
        insufficient = BlockResult(
            block_index=1, start_bar=500, end_bar=1000, bar_count=500,
            total_predictions=5, testable_count=3, confirmed_count=2,
            refuted_count=1, invalidated_count=2, timeout_count=0,
            crr=0.67, erv=1.0, erv_per_prediction=0.20, invalidation_rate=0.40,
            median_bars_to_confirmation=None, median_bars_to_refutation=None,
            timing_asymmetry_ratio=None, regime_distribution=None,
            verdict=BlockVerdict.INSUFFICIENT_DATA, verdict_reason="",
            crr_threshold=0.55, max_invalidation_threshold=0.40,
        )
        assert insufficient.is_sufficient is False


# =============================================================================
# TEMPORAL STABILITY ANALYSIS TESTS
# =============================================================================

class TestTemporalStabilityComputation:
    """Test temporal stability analysis."""
    
    def _create_block(
        self,
        index: int,
        crr: float,
        erv: float,
        inv_rate: float,
        testable: int = 20,
        verdict: BlockVerdict = BlockVerdict.SUPPORTED,
    ) -> BlockResult:
        """Helper to create mock block results."""
        confirmed = int(testable * crr)
        refuted = testable - confirmed
        return BlockResult(
            block_index=index,
            start_bar=index * 500,
            end_bar=(index + 1) * 500,
            bar_count=500,
            total_predictions=testable + 10,
            testable_count=testable,
            confirmed_count=confirmed,
            refuted_count=refuted,
            invalidated_count=int((testable + 10) * inv_rate),
            timeout_count=0,
            crr=crr,
            erv=erv * testable,
            erv_per_prediction=erv,
            invalidation_rate=inv_rate,
            median_bars_to_confirmation=12.0,
            median_bars_to_refutation=18.0,
            timing_asymmetry_ratio=0.67,
            regime_distribution=None,
            verdict=verdict,
            verdict_reason="",
            crr_threshold=0.55,
            max_invalidation_threshold=0.40,
        )
    
    def test_persistent_edge_detection(self):
        """Stable CRR across blocks should classify as PERSISTENT."""
        config = WalkForwardConfig(block_size=500)
        
        blocks = [
            self._create_block(0, crr=0.60, erv=0.10, inv_rate=0.20),
            self._create_block(1, crr=0.58, erv=0.08, inv_rate=0.22),
            self._create_block(2, crr=0.62, erv=0.12, inv_rate=0.18),
            self._create_block(3, crr=0.59, erv=0.09, inv_rate=0.21),
        ]
        
        stability = compute_temporal_stability(blocks, config)
        
        assert stability.stability_class == EdgeStabilityClass.PERSISTENT
        assert stability.verdict_persistence_ratio == 1.0
    
    def test_decaying_edge_detection(self):
        """Monotonically decreasing CRR should classify as DECAYING."""
        config = WalkForwardConfig(block_size=500)
        
        blocks = [
            self._create_block(0, crr=0.70, erv=0.20, inv_rate=0.15),
            self._create_block(1, crr=0.60, erv=0.10, inv_rate=0.25),
            self._create_block(2, crr=0.50, erv=0.00, inv_rate=0.35, verdict=BlockVerdict.REFUTED),
            self._create_block(3, crr=0.40, erv=-0.10, inv_rate=0.45, verdict=BlockVerdict.REFUTED),
        ]
        
        stability = compute_temporal_stability(blocks, config)
        
        assert stability.stability_class == EdgeStabilityClass.DECAYING
        assert stability.is_monotonic_decay is True
        assert stability.crr_drift_slope < 0
    
    def test_insufficient_blocks_unfalsifiable(self):
        """Too few blocks should classify as UNFALSIFIABLE."""
        config = WalkForwardConfig(block_size=500, min_blocks_for_stability=3)
        
        blocks = [
            self._create_block(0, crr=0.60, erv=0.10, inv_rate=0.20),
            self._create_block(1, crr=0.58, erv=0.08, inv_rate=0.22),
        ]
        
        stability = compute_temporal_stability(blocks, config)
        
        assert stability.stability_class == EdgeStabilityClass.UNFALSIFIABLE
        assert "Insufficient blocks" in stability.classification_reason
    
    def test_verdict_persistence_calculation(self):
        """Verdict persistence should be correctly calculated."""
        config = WalkForwardConfig(block_size=500)
        
        blocks = [
            self._create_block(0, crr=0.60, erv=0.10, inv_rate=0.20, verdict=BlockVerdict.SUPPORTED),
            self._create_block(1, crr=0.55, erv=0.05, inv_rate=0.25, verdict=BlockVerdict.SUPPORTED),
            self._create_block(2, crr=0.50, erv=0.00, inv_rate=0.35, verdict=BlockVerdict.REFUTED),
            self._create_block(3, crr=0.58, erv=0.08, inv_rate=0.22, verdict=BlockVerdict.SUPPORTED),
        ]
        
        stability = compute_temporal_stability(blocks, config)
        
        # 3 supported out of 4
        assert stability.verdict_persistence_ratio == 0.75
        assert stability.supported_count == 3
        assert stability.refuted_count == 1


# =============================================================================
# FALSE PERSISTENCE PREVENTION TESTS
# =============================================================================

class TestFalsePersistencePrevention:
    """Tests to ensure low sample sizes don't create false persistence."""
    
    def test_low_sample_blocks_excluded_from_verdict_count(self):
        """Blocks with insufficient data should not count toward persistence."""
        config = WalkForwardConfig(block_size=500, min_testable_per_block=10)
        
        # Create helper function
        def create_block(index, crr, testable, verdict):
            confirmed = int(testable * crr)
            return BlockResult(
                block_index=index,
                start_bar=index * 500,
                end_bar=(index + 1) * 500,
                bar_count=500,
                total_predictions=testable + 5,
                testable_count=testable,
                confirmed_count=confirmed,
                refuted_count=testable - confirmed,
                invalidated_count=5,
                timeout_count=0,
                crr=crr,
                erv=confirmed - (testable - confirmed),
                erv_per_prediction=(confirmed - (testable - confirmed)) / (testable + 5) if testable > 0 else 0,
                invalidation_rate=5 / (testable + 5),
                median_bars_to_confirmation=12.0,
                median_bars_to_refutation=18.0,
                timing_asymmetry_ratio=0.67,
                regime_distribution=None,
                verdict=verdict,
                verdict_reason="",
                crr_threshold=0.55,
                max_invalidation_threshold=0.40,
            )
        
        blocks = [
            create_block(0, 0.60, 20, BlockVerdict.SUPPORTED),
            create_block(1, 0.90, 3, BlockVerdict.INSUFFICIENT_DATA),  # High CRR but low N
            create_block(2, 0.55, 20, BlockVerdict.SUPPORTED),
            create_block(3, 0.95, 2, BlockVerdict.INSUFFICIENT_DATA),  # Very high CRR but very low N
        ]
        
        stability = compute_temporal_stability(blocks, config)
        
        # Only 2 sufficient blocks
        assert stability.sufficient_blocks == 2
        # Both sufficient blocks were SUPPORTED
        assert stability.verdict_persistence_ratio == 1.0
        # CRR values should only include sufficient blocks
        assert len(stability.crr_values) == 2
        assert 0.90 not in stability.crr_values  # High CRR from low-N block excluded
    
    def test_unfalsifiable_when_all_blocks_insufficient(self):
        """If all blocks have insufficient data, result is UNFALSIFIABLE."""
        config = WalkForwardConfig(block_size=500, min_testable_per_block=10)
        
        def create_insufficient_block(index):
            return BlockResult(
                block_index=index,
                start_bar=index * 500,
                end_bar=(index + 1) * 500,
                bar_count=500,
                total_predictions=8,
                testable_count=5,  # Below minimum
                confirmed_count=4,
                refuted_count=1,
                invalidated_count=3,
                timeout_count=0,
                crr=0.80,  # Looks great but not trustworthy
                erv=3.0,
                erv_per_prediction=0.375,
                invalidation_rate=0.375,
                median_bars_to_confirmation=10.0,
                median_bars_to_refutation=15.0,
                timing_asymmetry_ratio=0.67,
                regime_distribution=None,
                verdict=BlockVerdict.INSUFFICIENT_DATA,
                verdict_reason="Testable < 10",
                crr_threshold=0.55,
                max_invalidation_threshold=0.40,
            )
        
        blocks = [create_insufficient_block(i) for i in range(5)]
        
        stability = compute_temporal_stability(blocks, config)
        
        assert stability.stability_class == EdgeStabilityClass.UNFALSIFIABLE
        assert stability.sufficient_blocks == 0


# =============================================================================
# BLOCK ISOLATION TESTS
# =============================================================================

class TestBlockIsolation:
    """Tests to verify block isolation guarantees."""
    
    def test_block_boundaries_are_exclusive(self):
        """Block end should be exclusive (no overlap)."""
        config = WalkForwardConfig(block_size=500)
        blocks = config.compute_blocks(1500)
        
        # Check no bar belongs to multiple blocks
        seen_bars = set()
        for start, end in blocks:
            for bar in range(start, end):
                assert bar not in seen_bars, f"Bar {bar} in multiple blocks"
                seen_bars.add(bar)
    
    def test_all_bars_covered(self):
        """All bars should be covered (no gaps)."""
        config = WalkForwardConfig(block_size=500)
        blocks = config.compute_blocks(1500)
        
        covered_bars = set()
        for start, end in blocks:
            covered_bars.update(range(start, end))
        
        expected_bars = set(range(1500))
        assert covered_bars == expected_bars


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
