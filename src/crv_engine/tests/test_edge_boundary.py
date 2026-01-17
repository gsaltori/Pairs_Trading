"""
Unit Tests for Edge Boundary Analysis

Tests cover:
1. Observable bucket classification
2. Conditional metrics computation
3. Failure surface derivation
4. Collapse classification
5. Safe zone identification

NO STRATEGY LOGIC TESTED. ONLY FAILURE CARTOGRAPHY VALIDATION.
"""

import pytest
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.edge_boundary import (
    PredictionObservables,
    BucketMetrics,
    FailureSurface,
    CollapseAnalysis,
    CollapseType,
    EdgeSafeZone,
    CorrelationBucket,
    CorrelationTrendBucket,
    VolatilityRatioBucket,
    SpreadBucket,
    RegimeTransitionBucket,
    classify_correlation_level,
    classify_correlation_trend,
    classify_volatility_ratio,
    classify_spread_state,
    compute_bucket_metrics,
    derive_failure_threshold,
    classify_collapse,
    identify_safe_zone,
)
from experiments.edge_boundary_analyzer import (
    EdgeBoundaryAnalyzer,
    analyze_edge_boundaries,
)


# =============================================================================
# BUCKET CLASSIFICATION TESTS
# =============================================================================

class TestCorrelationBuckets:
    """Test correlation level classification."""
    
    def test_very_low_correlation(self):
        assert classify_correlation_level(0.1) == CorrelationBucket.VERY_LOW
        assert classify_correlation_level(0.29) == CorrelationBucket.VERY_LOW
    
    def test_low_correlation(self):
        assert classify_correlation_level(0.3) == CorrelationBucket.LOW
        assert classify_correlation_level(0.49) == CorrelationBucket.LOW
    
    def test_moderate_correlation(self):
        assert classify_correlation_level(0.5) == CorrelationBucket.MODERATE
        assert classify_correlation_level(0.69) == CorrelationBucket.MODERATE
    
    def test_high_correlation(self):
        assert classify_correlation_level(0.7) == CorrelationBucket.HIGH
        assert classify_correlation_level(0.84) == CorrelationBucket.HIGH
    
    def test_very_high_correlation(self):
        assert classify_correlation_level(0.85) == CorrelationBucket.VERY_HIGH
        assert classify_correlation_level(0.95) == CorrelationBucket.VERY_HIGH


class TestCorrelationTrendBuckets:
    """Test correlation trend classification."""
    
    def test_deteriorating_trend(self):
        # Current 0.5, prior 0.6 → delta = -0.1 < -0.05
        assert classify_correlation_trend(0.5, 0.6) == CorrelationTrendBucket.DETERIORATING
    
    def test_stable_trend(self):
        # Current 0.5, prior 0.52 → delta = -0.02, within threshold
        assert classify_correlation_trend(0.5, 0.52) == CorrelationTrendBucket.STABLE
    
    def test_improving_trend(self):
        # Current 0.6, prior 0.5 → delta = +0.1 > 0.05
        assert classify_correlation_trend(0.6, 0.5) == CorrelationTrendBucket.IMPROVING


class TestVolatilityRatioBuckets:
    """Test volatility ratio classification."""
    
    def test_compressed_volatility(self):
        assert classify_volatility_ratio(0.5) == VolatilityRatioBucket.COMPRESSED
        assert classify_volatility_ratio(0.69) == VolatilityRatioBucket.COMPRESSED
    
    def test_balanced_volatility(self):
        assert classify_volatility_ratio(0.7) == VolatilityRatioBucket.BALANCED
        assert classify_volatility_ratio(1.0) == VolatilityRatioBucket.BALANCED
        assert classify_volatility_ratio(1.29) == VolatilityRatioBucket.BALANCED
    
    def test_imbalanced_volatility(self):
        assert classify_volatility_ratio(1.3) == VolatilityRatioBucket.IMBALANCED
        assert classify_volatility_ratio(1.99) == VolatilityRatioBucket.IMBALANCED
    
    def test_extreme_volatility(self):
        assert classify_volatility_ratio(2.0) == VolatilityRatioBucket.EXTREME
        assert classify_volatility_ratio(3.0) == VolatilityRatioBucket.EXTREME


class TestSpreadBuckets:
    """Test spread state classification."""
    
    def test_contracted_spread(self):
        assert classify_spread_state(0.5) == SpreadBucket.CONTRACTED
        assert classify_spread_state(-0.9) == SpreadBucket.CONTRACTED
    
    def test_normal_spread(self):
        assert classify_spread_state(1.5) == SpreadBucket.NORMAL
        assert classify_spread_state(-1.9) == SpreadBucket.NORMAL
    
    def test_extended_spread(self):
        assert classify_spread_state(2.5) == SpreadBucket.EXTENDED
        assert classify_spread_state(-2.9) == SpreadBucket.EXTENDED
    
    def test_extreme_spread(self):
        assert classify_spread_state(3.5) == SpreadBucket.EXTREME
        assert classify_spread_state(-4.0) == SpreadBucket.EXTREME


# =============================================================================
# PREDICTION OBSERVABLES TESTS
# =============================================================================

class TestPredictionObservables:
    """Test PredictionObservables derived properties."""
    
    def test_bucket_derivation(self):
        pred = PredictionObservables(
            prediction_id="p1",
            outcome="CONFIRMED",
            bars_to_resolution=10,
            correlation=0.65,
            correlation_trend=-0.08,
            volatility_ratio=1.1,
            zscore=1.8,
            spread_velocity=0.1,
        )
        
        assert pred.correlation_bucket == CorrelationBucket.MODERATE
        assert pred.correlation_trend_bucket == CorrelationTrendBucket.DETERIORATING
        assert pred.volatility_bucket == VolatilityRatioBucket.BALANCED
        assert pred.spread_bucket == SpreadBucket.NORMAL
    
    def test_is_testable(self):
        confirmed = PredictionObservables(
            prediction_id="p1", outcome="CONFIRMED", bars_to_resolution=10,
            correlation=0.7, correlation_trend=0.0, volatility_ratio=1.0,
            zscore=1.5, spread_velocity=0.0,
        )
        refuted = PredictionObservables(
            prediction_id="p2", outcome="REFUTED", bars_to_resolution=15,
            correlation=0.7, correlation_trend=0.0, volatility_ratio=1.0,
            zscore=1.5, spread_velocity=0.0,
        )
        invalidated = PredictionObservables(
            prediction_id="p3", outcome="INVALIDATED", bars_to_resolution=5,
            correlation=0.7, correlation_trend=0.0, volatility_ratio=1.0,
            zscore=1.5, spread_velocity=0.0,
        )
        
        assert confirmed.is_testable is True
        assert refuted.is_testable is True
        assert invalidated.is_testable is False


# =============================================================================
# BUCKET METRICS TESTS
# =============================================================================

class TestBucketMetrics:
    """Test bucket metrics computation."""
    
    def _create_prediction(
        self,
        outcome: str,
        correlation: float = 0.7,
    ) -> PredictionObservables:
        return PredictionObservables(
            prediction_id=f"p_{outcome}_{correlation}",
            outcome=outcome,
            bars_to_resolution=10,
            correlation=correlation,
            correlation_trend=0.0,
            volatility_ratio=1.0,
            zscore=1.5,
            spread_velocity=0.0,
        )
    
    def test_crr_calculation(self):
        predictions = [
            self._create_prediction("CONFIRMED"),
            self._create_prediction("CONFIRMED"),
            self._create_prediction("CONFIRMED"),
            self._create_prediction("REFUTED"),
            self._create_prediction("REFUTED"),
        ]
        
        metrics = compute_bucket_metrics(
            bucket_name="test",
            bucket_value="TEST",
            predictions=predictions,
        )
        
        assert metrics.testable_count == 5
        assert metrics.confirmed_count == 3
        assert metrics.crr == 0.6
    
    def test_invalidation_rate(self):
        predictions = [
            self._create_prediction("CONFIRMED"),
            self._create_prediction("REFUTED"),
            self._create_prediction("INVALIDATED"),
            self._create_prediction("INVALIDATED"),
        ]
        
        metrics = compute_bucket_metrics(
            bucket_name="test",
            bucket_value="TEST",
            predictions=predictions,
        )
        
        assert metrics.invalidation_rate == 0.5  # 2/4
    
    def test_empty_bucket(self):
        metrics = compute_bucket_metrics(
            bucket_name="test",
            bucket_value="TEST",
            predictions=[],
        )
        
        assert metrics.total_predictions == 0
        assert metrics.crr == 0.0
        assert metrics.invalidation_rate == 0.0


# =============================================================================
# FAILURE SURFACE TESTS
# =============================================================================

class TestFailureSurfaceDerivation:
    """Test empirical failure threshold derivation."""
    
    def test_finds_correlation_failure_threshold(self):
        # Create predictions where low correlation = failure
        outcomes = []
        correlations = []
        
        # Low correlation: mostly refuted
        for _ in range(20):
            correlations.append(0.3)
            outcomes.append("REFUTED")
        for _ in range(5):
            correlations.append(0.35)
            outcomes.append("CONFIRMED")
        
        # High correlation: mostly confirmed
        for _ in range(30):
            correlations.append(0.7)
            outcomes.append("CONFIRMED")
        for _ in range(5):
            correlations.append(0.75)
            outcomes.append("REFUTED")
        
        result = derive_failure_threshold(
            observable_values=correlations,
            outcomes=outcomes,
            crr_threshold=0.55,
            min_samples=10,
            search_direction="below",
        )
        
        assert result is not None
        threshold, crr, inv_rate, n = result
        assert crr < 0.55  # Should be in failure region
    
    def test_no_threshold_when_insufficient_data(self):
        result = derive_failure_threshold(
            observable_values=[0.5, 0.6],
            outcomes=["CONFIRMED", "REFUTED"],
            crr_threshold=0.55,
            min_samples=10,
            search_direction="below",
        )
        
        assert result is None


# =============================================================================
# COLLAPSE CLASSIFICATION TESTS
# =============================================================================

class TestCollapseClassification:
    """Test collapse pattern classification."""
    
    def test_gradual_decay_detection(self):
        crr_series = [0.70, 0.65, 0.60, 0.55, 0.50]
        inv_series = [0.15, 0.18, 0.20, 0.22, 0.25]
        
        result = classify_collapse(
            crr_series=crr_series,
            invalidation_series=inv_series,
            regime_change_rate=0.1,
            crr_variance=0.005,
        )
        
        assert result.collapse_type == CollapseType.GRADUAL_DECAY
    
    def test_structural_invalidation_detection(self):
        crr_series = [0.60, 0.58, 0.55, 0.52]
        inv_series = [0.15, 0.30, 0.45, 0.60]  # Rapidly rising invalidation
        
        result = classify_collapse(
            crr_series=crr_series,
            invalidation_series=inv_series,
            regime_change_rate=0.1,
            crr_variance=0.002,
        )
        
        assert result.collapse_type == CollapseType.STRUCTURAL_INVALIDATION
    
    def test_no_collapse_stable_edge(self):
        crr_series = [0.60, 0.62, 0.58, 0.61, 0.59]
        inv_series = [0.15, 0.14, 0.16, 0.15, 0.15]
        
        result = classify_collapse(
            crr_series=crr_series,
            invalidation_series=inv_series,
            regime_change_rate=0.05,
            crr_variance=0.002,
        )
        
        assert result.collapse_type == CollapseType.NO_COLLAPSE
    
    def test_insufficient_data(self):
        result = classify_collapse(
            crr_series=[0.6, 0.55],
            invalidation_series=[0.15, 0.18],
            regime_change_rate=0.0,
            crr_variance=0.001,
        )
        
        assert result.collapse_type == CollapseType.INSUFFICIENT_DATA


# =============================================================================
# SAFE ZONE TESTS
# =============================================================================

class TestSafeZoneIdentification:
    """Test safe zone identification."""
    
    def _create_predictions_with_correlation(
        self,
        correlations: List[float],
        success_rate: float,
    ) -> List[PredictionObservables]:
        preds = []
        for i, corr in enumerate(correlations):
            outcome = "CONFIRMED" if (i / len(correlations)) < success_rate else "REFUTED"
            preds.append(PredictionObservables(
                prediction_id=f"p{i}",
                outcome=outcome,
                bars_to_resolution=10,
                correlation=corr,
                correlation_trend=0.0,
                volatility_ratio=1.0,
                zscore=1.5,
                spread_velocity=0.0,
            ))
        return preds
    
    def test_identifies_safe_zone_with_high_correlation(self):
        # Low correlation predictions: 40% success
        low_corr_preds = self._create_predictions_with_correlation(
            correlations=[0.3 + 0.01 * i for i in range(30)],
            success_rate=0.4,
        )
        
        # High correlation predictions: 70% success
        high_corr_preds = self._create_predictions_with_correlation(
            correlations=[0.7 + 0.01 * i for i in range(30)],
            success_rate=0.7,
        )
        
        all_preds = low_corr_preds + high_corr_preds
        
        safe_zone = identify_safe_zone(
            predictions=all_preds,
            crr_threshold=0.55,
            min_samples=10,
        )
        
        # Should find safe zone in high correlation region
        if safe_zone:
            assert safe_zone.safe_zone_crr >= 0.55
    
    def test_no_safe_zone_when_universally_failing(self):
        # All predictions fail
        preds = [
            PredictionObservables(
                prediction_id=f"p{i}",
                outcome="REFUTED",
                bars_to_resolution=10,
                correlation=0.3 + 0.05 * i,
                correlation_trend=0.0,
                volatility_ratio=1.0,
                zscore=1.5,
                spread_velocity=0.0,
            )
            for i in range(30)
        ]
        
        safe_zone = identify_safe_zone(
            predictions=preds,
            crr_threshold=0.55,
            min_samples=10,
        )
        
        assert safe_zone is None


# =============================================================================
# ANALYZER INTEGRATION TESTS
# =============================================================================

class TestEdgeBoundaryAnalyzer:
    """Test full analyzer integration."""
    
    def _create_diverse_predictions(self, n: int = 100) -> List[PredictionObservables]:
        """Create diverse prediction set for testing."""
        import random
        random.seed(42)
        
        preds = []
        for i in range(n):
            # Vary correlation
            corr = random.uniform(0.3, 0.9)
            
            # Success depends on correlation
            success_prob = 0.3 + 0.5 * corr  # Higher correlation = more success
            outcome = "CONFIRMED" if random.random() < success_prob else "REFUTED"
            
            # Some invalidations at extreme volatility
            vol_ratio = random.uniform(0.5, 2.5)
            if vol_ratio > 2.0 and random.random() < 0.5:
                outcome = "INVALIDATED"
            
            preds.append(PredictionObservables(
                prediction_id=f"p{i}",
                outcome=outcome,
                bars_to_resolution=random.randint(5, 30),
                correlation=corr,
                correlation_trend=random.uniform(-0.1, 0.1),
                volatility_ratio=vol_ratio,
                zscore=random.uniform(-3, 3),
                spread_velocity=random.uniform(-0.5, 0.5),
            ))
        
        return preds
    
    def test_analyzer_produces_output(self):
        preds = self._create_diverse_predictions(100)
        
        output = analyze_edge_boundaries(
            predictions=preds,
            crr_threshold=0.55,
            verbose=False,
        )
        
        assert output.total_predictions == 100
        assert len(output.correlation_level_metrics) == 5  # 5 buckets
        assert output.collapse_analysis is not None
    
    def test_analyzer_identifies_correlation_dependency(self):
        # Create predictions where success strongly depends on correlation
        preds = []
        for i in range(50):
            # Low correlation: fail
            preds.append(PredictionObservables(
                prediction_id=f"low_{i}",
                outcome="REFUTED",
                bars_to_resolution=10,
                correlation=0.35,
                correlation_trend=0.0,
                volatility_ratio=1.0,
                zscore=1.5,
                spread_velocity=0.0,
            ))
        
        for i in range(50):
            # High correlation: succeed
            preds.append(PredictionObservables(
                prediction_id=f"high_{i}",
                outcome="CONFIRMED",
                bars_to_resolution=10,
                correlation=0.75,
                correlation_trend=0.0,
                volatility_ratio=1.0,
                zscore=1.5,
                spread_velocity=0.0,
            ))
        
        output = analyze_edge_boundaries(
            predictions=preds,
            crr_threshold=0.55,
            verbose=False,
        )
        
        # Should flag correlation dependency
        assert output.flags['has_correlation_dependency'] is True
    
    def test_output_serialization(self):
        preds = self._create_diverse_predictions(50)
        
        output = analyze_edge_boundaries(
            predictions=preds,
            crr_threshold=0.55,
            verbose=False,
        )
        
        # Should serialize to JSON without error
        json_str = output.to_json()
        assert len(json_str) > 0
        
        # Should deserialize back
        import json
        data = json.loads(json_str)
        assert data['schema_version'] == '1.0.0'
        assert 'failure_surfaces' in data
        assert 'edge_safe_zone' in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
