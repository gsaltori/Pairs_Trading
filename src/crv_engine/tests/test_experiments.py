"""
Unit tests for CRV Experiment Harness.

Tests cover:
1. Configuration validation (single-dimension constraint)
2. Hypothesis evaluation logic
3. Metrics computation
4. Output serialization
5. Batch aggregation

NO STRATEGY LOGIC TESTED. ONLY HARNESS INFRASTRUCTURE.
"""

import pytest
import json
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.config import (
    ExperimentConfig,
    ExperimentDimension,
    ExperimentBatch,
    Timeframe,
    BASELINE,
    create_baseline_experiment,
    create_symbol_pair_experiment,
    create_timeframe_experiment,
    create_zscore_window_experiment,
    create_seed_experiment,
)
from experiments.hypothesis import (
    HypothesisSpec,
    HypothesisVerdict,
    HypothesisResult,
    FalsificationCriterion,
    SampleSizeRequirement,
    ComparisonOperator,
    create_edge_exists_hypothesis,
    create_confirmation_speed_hypothesis,
)
from experiments.metrics import (
    ExtendedMetrics,
    ResolutionRecord,
    compute_extended_metrics,
    interpret_erv,
    interpret_timing_asymmetry,
)
from experiments.output import (
    ExperimentOutput,
    ExperimentOutputBuilder,
    AggregatedResults,
    aggregate_experiments,
    SCHEMA_VERSION,
)


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================

class TestExperimentConfig:
    """Test experiment configuration validation."""
    
    def test_baseline_experiment_creation(self):
        """Baseline experiment should use all default values."""
        config = create_baseline_experiment("test_001")
        
        assert config.symbol_a == BASELINE.SYMBOL_A
        assert config.symbol_b == BASELINE.SYMBOL_B
        assert config.timeframe == BASELINE.TIMEFRAME
        assert config.zscore_window == BASELINE.ZSCORE_WINDOW
        assert config.correlation_window == BASELINE.CORRELATION_WINDOW
    
    def test_config_hash_deterministic(self):
        """Same config should produce same hash."""
        config1 = create_baseline_experiment("test_001")
        config2 = create_baseline_experiment("test_001")
        
        assert config1.config_hash == config2.config_hash
    
    def test_config_hash_changes_with_params(self):
        """Different params should produce different hash."""
        config1 = create_baseline_experiment("test_001")
        config2 = create_seed_experiment("test_002", random_seed=999)
        
        assert config1.config_hash != config2.config_hash
    
    def test_single_dimension_constraint_symbol(self):
        """Symbol pair experiment should only vary symbols."""
        config = create_symbol_pair_experiment("test", "USDJPY", "EURJPY")
        
        # Should validate without error
        assert config.validate_single_dimension()
        assert config.dimension_varied == ExperimentDimension.SYMBOL_PAIR
    
    def test_single_dimension_constraint_timeframe(self):
        """Timeframe experiment should only vary timeframe."""
        config = create_timeframe_experiment("test", Timeframe.D1)
        
        assert config.validate_single_dimension()
        assert config.dimension_varied == ExperimentDimension.TIMEFRAME
    
    def test_single_dimension_constraint_zscore(self):
        """Z-score window experiment should only vary zscore_window."""
        config = create_zscore_window_experiment("test", zscore_window=30)
        
        assert config.validate_single_dimension()
        assert config.dimension_varied == ExperimentDimension.ZSCORE_WINDOW
    
    def test_multiple_dimensions_raises_error(self):
        """Varying multiple dimensions should raise ValueError."""
        # Manually create config with multiple changes
        config = ExperimentConfig(
            experiment_id="invalid",
            dimension_varied=ExperimentDimension.TIMEFRAME,  # Declared
            symbol_a="USDJPY",  # Changed from baseline
            symbol_b="EURJPY",  # Changed from baseline
            timeframe=Timeframe.D1,  # Changed from baseline
            zscore_window=BASELINE.ZSCORE_WINDOW,
            correlation_window=BASELINE.CORRELATION_WINDOW,
        )
        
        with pytest.raises(ValueError, match="Multiple dimensions varied"):
            config.validate_single_dimension()


class TestExperimentBatch:
    """Test experiment batch validation."""
    
    def test_batch_same_dimension(self):
        """All experiments in batch must vary same dimension."""
        experiments = (
            create_seed_experiment("seed_42", random_seed=42),
            create_seed_experiment("seed_43", random_seed=43),
            create_seed_experiment("seed_44", random_seed=44),
        )
        
        batch = ExperimentBatch(
            batch_id="seed_batch",
            dimension=ExperimentDimension.RANDOM_SEED,
            experiments=experiments,
        )
        
        assert len(batch.experiments) == 3
    
    def test_batch_mixed_dimensions_raises_error(self):
        """Mixed dimensions in batch should raise ValueError."""
        experiments = (
            create_seed_experiment("seed_42", random_seed=42),
            create_timeframe_experiment("tf_d1", Timeframe.D1),
        )
        
        with pytest.raises(ValueError):
            ExperimentBatch(
                batch_id="invalid_batch",
                dimension=ExperimentDimension.RANDOM_SEED,
                experiments=experiments,
            )


# =============================================================================
# HYPOTHESIS TESTS
# =============================================================================

class TestFalsificationCriterion:
    """Test falsification criterion evaluation."""
    
    def test_greater_than(self):
        """Test > operator."""
        criterion = FalsificationCriterion(
            metric_name="crr",
            operator=ComparisonOperator.GREATER_THAN,
            threshold=0.5,
            description="CRR > 50%",
        )
        
        assert criterion.evaluate(0.6) == True
        assert criterion.evaluate(0.5) == False
        assert criterion.evaluate(0.4) == False
    
    def test_greater_than_or_equal(self):
        """Test >= operator."""
        criterion = FalsificationCriterion(
            metric_name="crr",
            operator=ComparisonOperator.GREATER_THAN_OR_EQUAL,
            threshold=0.55,
            description="CRR >= 55%",
        )
        
        assert criterion.evaluate(0.55) == True
        assert criterion.evaluate(0.56) == True
        assert criterion.evaluate(0.54) == False
    
    def test_less_than_or_equal(self):
        """Test <= operator."""
        criterion = FalsificationCriterion(
            metric_name="invalidation_rate",
            operator=ComparisonOperator.LESS_THAN_OR_EQUAL,
            threshold=0.40,
            description="Invalidation <= 40%",
        )
        
        assert criterion.evaluate(0.35) == True
        assert criterion.evaluate(0.40) == True
        assert criterion.evaluate(0.45) == False


class TestHypothesisEvaluation:
    """Test hypothesis evaluation logic."""
    
    def test_supported_when_all_criteria_pass(self):
        """Hypothesis is SUPPORTED when all criteria pass."""
        hypothesis = create_edge_exists_hypothesis(
            "h_test",
            min_crr=0.55,
            min_testable=20,
            max_invalidation_rate=0.40,
        )
        
        metrics = {'crr': 0.60, 'invalidation_rate': 0.30}
        sample_sizes = {'testable_count': 50}
        
        result = hypothesis.evaluate(metrics, sample_sizes)
        
        assert result.verdict == HypothesisVerdict.SUPPORTED
        assert result.is_supported
    
    def test_refuted_when_criterion_fails(self):
        """Hypothesis is REFUTED when any criterion fails."""
        hypothesis = create_edge_exists_hypothesis(
            "h_test",
            min_crr=0.55,
            min_testable=20,
            max_invalidation_rate=0.40,
        )
        
        metrics = {'crr': 0.50, 'invalidation_rate': 0.30}  # CRR too low
        sample_sizes = {'testable_count': 50}
        
        result = hypothesis.evaluate(metrics, sample_sizes)
        
        assert result.verdict == HypothesisVerdict.REFUTED
        assert result.is_refuted
    
    def test_insufficient_data_when_sample_too_small(self):
        """Hypothesis is INSUFFICIENT_DATA when sample size fails."""
        hypothesis = create_edge_exists_hypothesis(
            "h_test",
            min_crr=0.55,
            min_testable=30,
            max_invalidation_rate=0.40,
        )
        
        metrics = {'crr': 0.60, 'invalidation_rate': 0.30}
        sample_sizes = {'testable_count': 15}  # Too small
        
        result = hypothesis.evaluate(metrics, sample_sizes)
        
        assert result.verdict == HypothesisVerdict.INSUFFICIENT_DATA
        assert result.has_insufficient_data
    
    def test_confirmation_speed_hypothesis(self):
        """Test confirmation speed hypothesis."""
        hypothesis = create_confirmation_speed_hypothesis(
            "h_speed",
            max_median_bars_to_confirm=20,
            min_confirmed=10,
        )
        
        metrics = {'median_bars_to_confirmation': 15.0}
        sample_sizes = {'confirmed_count': 25}
        
        result = hypothesis.evaluate(metrics, sample_sizes)
        
        assert result.verdict == HypothesisVerdict.SUPPORTED


# =============================================================================
# METRICS TESTS
# =============================================================================

class TestExtendedMetrics:
    """Test extended metrics computation."""
    
    def test_empty_records(self):
        """Empty records should return zero metrics."""
        metrics = compute_extended_metrics([])
        
        assert metrics.total_predictions == 0
        assert metrics.crr == 0.0
        assert metrics.erv == 0.0
    
    def test_crr_calculation(self):
        """CRR should be CONFIRMED / testable."""
        records = [
            ResolutionRecord("p1", "CONFIRMED", 10),
            ResolutionRecord("p2", "CONFIRMED", 8),
            ResolutionRecord("p3", "REFUTED", 15),
            ResolutionRecord("p4", "CONFIRMED", 12),
        ]
        
        metrics = compute_extended_metrics(records)
        
        assert metrics.testable_count == 4
        assert metrics.confirmed_count == 3
        assert metrics.crr == 0.75
    
    def test_erv_calculation(self):
        """ERV should weight outcomes correctly."""
        records = [
            ResolutionRecord("p1", "CONFIRMED", 10),   # +1
            ResolutionRecord("p2", "REFUTED", 15),     # -1
            ResolutionRecord("p3", "INVALIDATED", 5),  # 0
            ResolutionRecord("p4", "TIMEOUT", 50),     # -0.5
        ]
        
        metrics = compute_extended_metrics(records)
        
        expected_erv = 1.0 - 1.0 + 0.0 - 0.5  # -0.5
        assert metrics.erv == expected_erv
        assert metrics.erv_per_prediction == expected_erv / 4
    
    def test_timing_asymmetry(self):
        """Timing asymmetry should be confirm_median / refute_median."""
        records = [
            ResolutionRecord("p1", "CONFIRMED", 10),
            ResolutionRecord("p2", "CONFIRMED", 12),
            ResolutionRecord("p3", "REFUTED", 20),
            ResolutionRecord("p4", "REFUTED", 24),
        ]
        
        metrics = compute_extended_metrics(records)
        
        # Median confirm = 11, Median refute = 22
        assert metrics.median_bars_to_confirmation == 11.0
        assert metrics.median_bars_to_refutation == 22.0
        assert metrics.timing_asymmetry_ratio == 0.5
    
    def test_conditional_crr_buckets(self):
        """Conditional CRR should segment by stability duration."""
        records = [
            # Short stability (< 20)
            ResolutionRecord("p1", "CONFIRMED", 10, stability_duration_at_creation=10),
            ResolutionRecord("p2", "CONFIRMED", 10, stability_duration_at_creation=15),
            ResolutionRecord("p3", "REFUTED", 10, stability_duration_at_creation=10),
            ResolutionRecord("p4", "REFUTED", 10, stability_duration_at_creation=10),
            ResolutionRecord("p5", "REFUTED", 10, stability_duration_at_creation=10),
            # Medium stability (20-40)
            ResolutionRecord("p6", "CONFIRMED", 10, stability_duration_at_creation=30),
            ResolutionRecord("p7", "CONFIRMED", 10, stability_duration_at_creation=30),
            ResolutionRecord("p8", "CONFIRMED", 10, stability_duration_at_creation=30),
            ResolutionRecord("p9", "REFUTED", 10, stability_duration_at_creation=30),
            ResolutionRecord("p10", "REFUTED", 10, stability_duration_at_creation=30),
        ]
        
        metrics = compute_extended_metrics(records)
        
        # Short: 2/5 = 40%
        assert metrics.conditional_crr_short_stability == 0.4
        assert metrics.conditional_n_short == 5
        
        # Medium: 3/5 = 60%
        assert metrics.conditional_crr_medium_stability == 0.6
        assert metrics.conditional_n_medium == 5


class TestMetricInterpretation:
    """Test metric interpretation helpers."""
    
    def test_interpret_erv(self):
        """ERV interpretation should categorize correctly."""
        assert "STRONG_POSITIVE" in interpret_erv(0.4)
        assert "POSITIVE" in interpret_erv(0.2)
        assert "NEUTRAL" in interpret_erv(0.0)
        assert "NEGATIVE" in interpret_erv(-0.2)
        assert "STRONG_NEGATIVE" in interpret_erv(-0.4)
    
    def test_interpret_timing_asymmetry(self):
        """Timing asymmetry interpretation should categorize correctly."""
        assert "HEALTHY" in interpret_timing_asymmetry(0.5)
        assert "ACCEPTABLE" in interpret_timing_asymmetry(0.8)
        assert "WARNING" in interpret_timing_asymmetry(1.1)
        assert "CONCERNING" in interpret_timing_asymmetry(1.5)
        assert "INSUFFICIENT" in interpret_timing_asymmetry(None)


# =============================================================================
# OUTPUT TESTS
# =============================================================================

class TestExperimentOutput:
    """Test experiment output serialization."""
    
    def test_output_to_json(self):
        """Output should serialize to valid JSON."""
        output = ExperimentOutput(
            schema_version=SCHEMA_VERSION,
            experiment_id="test_001",
            config_hash="abc123",
            run_timestamp="2026-01-16T12:00:00",
            config={'symbol_a': 'EURUSD', 'symbol_b': 'GBPUSD'},
            metrics={'crr': 0.55, 'erv': 10.0},
            hypothesis_results=[{'verdict': 'SUPPORTED'}],
            execution_time_seconds=1.5,
            engine_version="0.3.0",
            data_source="synthetic",
            warnings=[],
            errors=[],
        )
        
        json_str = output.to_json()
        parsed = json.loads(json_str)
        
        assert parsed['experiment_id'] == "test_001"
        assert parsed['metrics']['crr'] == 0.55
    
    def test_primary_verdict(self):
        """Primary verdict should return first hypothesis verdict."""
        output = ExperimentOutput(
            schema_version=SCHEMA_VERSION,
            experiment_id="test_001",
            config_hash="abc123",
            run_timestamp="2026-01-16T12:00:00",
            config={},
            metrics={},
            hypothesis_results=[
                {'verdict': 'REFUTED', 'hypothesis_id': 'h1'},
                {'verdict': 'SUPPORTED', 'hypothesis_id': 'h2'},
            ],
            execution_time_seconds=1.0,
            engine_version="0.3.0",
            data_source="synthetic",
            warnings=[],
            errors=[],
        )
        
        assert output.primary_verdict == 'REFUTED'


class TestOutputBuilder:
    """Test experiment output builder."""
    
    def test_builder_requires_metrics(self):
        """Builder should raise error if metrics not set."""
        config = create_baseline_experiment("test")
        builder = ExperimentOutputBuilder(config)
        
        with pytest.raises(ValueError, match="Metrics must be set"):
            builder.build()
    
    def test_builder_creates_valid_output(self):
        """Builder should create complete output."""
        config = create_baseline_experiment("test")
        metrics = compute_extended_metrics([
            ResolutionRecord("p1", "CONFIRMED", 10),
        ])
        
        builder = ExperimentOutputBuilder(config)
        builder.start()
        builder.set_metrics(metrics)
        
        output = builder.build()
        
        assert output.experiment_id == "test"
        assert output.metrics['confirmed_count'] == 1


class TestAggregation:
    """Test experiment aggregation."""
    
    def test_aggregate_experiments(self):
        """Aggregation should compute correct statistics."""
        outputs = [
            ExperimentOutput(
                schema_version=SCHEMA_VERSION,
                experiment_id=f"exp_{i}",
                config_hash="hash",
                run_timestamp="2026-01-16T12:00:00",
                config={'dimension_varied': 'random_seed'},
                metrics={'crr': crr, 'erv_per_prediction': erv, 'invalidation_rate': 0.3},
                hypothesis_results=[{'verdict': 'SUPPORTED'}],
                execution_time_seconds=1.0,
                engine_version="0.3.0",
                data_source="synthetic",
                warnings=[],
                errors=[],
            )
            for i, (crr, erv) in enumerate([(0.55, 0.1), (0.60, 0.15), (0.50, 0.05)])
        ]
        
        aggregated = aggregate_experiments("batch_001", outputs)
        
        assert aggregated.experiment_count == 3
        assert aggregated.mean_crr == pytest.approx(0.55, abs=0.01)
        assert aggregated.min_crr == 0.50
        assert aggregated.max_crr == 0.60
        assert aggregated.hypotheses_supported == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
