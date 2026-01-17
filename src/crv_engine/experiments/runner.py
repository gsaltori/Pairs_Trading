"""
Experiment Runner Module

Executes the CRV engine UNCHANGED under controlled variation.

CRITICAL CONSTRAINT:
This module does NOT modify the engine. It only:
1. Configures the engine with experiment parameters
2. Runs the engine
3. Collects output data
4. Computes extended metrics
5. Evaluates hypotheses
6. Produces structured output

NO STRATEGY LOGIC. NO SIGNAL CHANGES. EXECUTION ONLY.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Callable
from datetime import datetime
from pathlib import Path
import time
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from .config import (
    ExperimentConfig,
    ExperimentBatch,
    BASELINE,
    create_baseline_experiment,
)
from .hypothesis import HypothesisSpec, HypothesisResult
from .metrics import ExtendedMetrics, ResolutionRecord, compute_extended_metrics
from .output import ExperimentOutput, ExperimentOutputBuilder, aggregate_experiments


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

class ExperimentRunner:
    """
    Runs the CRV engine unchanged under controlled configuration.
    
    RESPONSIBILITIES:
    1. Configure engine with experiment parameters
    2. Execute engine on data
    3. Collect resolution records
    4. Compute extended metrics
    5. Evaluate hypotheses
    6. Produce structured output
    
    DOES NOT:
    - Modify engine logic
    - Add signals or filters
    - Tune parameters
    """
    
    def __init__(
        self,
        output_dir: Optional[Path] = None,
        verbose: bool = False,
    ):
        self.output_dir = Path(output_dir) if output_dir else Path("experiments_output")
        self.verbose = verbose
    
    def run(
        self,
        config: ExperimentConfig,
        hypotheses: List[HypothesisSpec] = None,
    ) -> ExperimentOutput:
        """
        Run a single experiment.
        
        Args:
            config: Experiment configuration
            hypotheses: List of hypotheses to evaluate (optional)
        
        Returns:
            ExperimentOutput with complete results
        """
        hypotheses = hypotheses or []
        
        # Validate configuration
        config.validate_single_dimension()
        
        # Build output
        builder = ExperimentOutputBuilder(config)
        builder.start()
        
        if self.verbose:
            print(f"Running experiment: {config.experiment_id}")
            print(f"  Dimension varied: {config.dimension_varied.value}")
            print(f"  Config hash: {config.config_hash}")
        
        try:
            # Run engine and collect records
            records = self._execute_engine(config, builder)
            
            # Compute extended metrics
            metrics = compute_extended_metrics(records)
            builder.set_metrics(metrics)
            
            if self.verbose:
                print(f"  Predictions: {metrics.total_predictions}")
                print(f"  CRR: {metrics.crr:.1%}")
                print(f"  ERV/pred: {metrics.erv_per_prediction:.3f}")
            
            # Evaluate hypotheses
            metrics_dict = metrics.to_dict()
            sample_sizes = {
                'testable_count': metrics.testable_count,
                'confirmed_count': metrics.confirmed_count,
                'refuted_count': metrics.refuted_count,
                'total_predictions': metrics.total_predictions,
            }
            
            for hypothesis in hypotheses:
                result = hypothesis.evaluate(metrics_dict, sample_sizes)
                builder.add_hypothesis_result(result)
                
                if self.verbose:
                    print(f"  Hypothesis '{hypothesis.name}': {result.verdict.value}")
        
        except Exception as e:
            builder.add_error(str(e))
            # Set empty metrics to allow output generation
            builder.set_metrics(compute_extended_metrics([]))
            raise
        
        # Build and save output
        output = builder.build()
        
        if self.output_dir:
            filepath = output.save(self.output_dir)
            if self.verbose:
                print(f"  Output saved: {filepath}")
        
        return output
    
    def run_batch(
        self,
        batch: ExperimentBatch,
        hypotheses: List[HypothesisSpec] = None,
    ) -> List[ExperimentOutput]:
        """
        Run a batch of experiments.
        
        Args:
            batch: Experiment batch configuration
            hypotheses: Hypotheses to evaluate for each experiment
        
        Returns:
            List of ExperimentOutput objects
        """
        if self.verbose:
            print(f"Running batch: {batch.batch_id}")
            print(f"  Dimension: {batch.dimension.value}")
            print(f"  Experiments: {len(batch.experiments)}")
            print()
        
        outputs = []
        for config in batch.experiments:
            output = self.run(config, hypotheses)
            outputs.append(output)
            if self.verbose:
                print()
        
        return outputs
    
    def _execute_engine(
        self,
        config: ExperimentConfig,
        builder: ExperimentOutputBuilder,
    ) -> List[ResolutionRecord]:
        """
        Execute the CRV engine with given configuration.
        
        Returns list of resolution records for metric computation.
        
        IMPORTANT: This method runs the EXISTING engine unchanged.
        """
        # Import engine components
        try:
            from observations import generate_observation_stream
            from spread import SpreadCalculator
            from predictions import PredictionGenerator
            from resolution import ResolutionEngine, ResolutionState
        except ImportError:
            from ..observations import generate_observation_stream
            from ..spread import SpreadCalculator
            from ..predictions import PredictionGenerator
            from ..resolution import ResolutionEngine, ResolutionState
        
        # Generate data
        if config.use_synthetic:
            observation_stream = generate_observation_stream(
                n_bars=config.n_bars,
                seed=config.random_seed,
            )
        else:
            builder.add_error("Live MT5 data not implemented in experiment harness")
            return []
        
        # Initialize engine components (UNCHANGED)
        spread_calc = SpreadCalculator(
            pair=config.pair,
            zscore_window=config.zscore_window,
            hedge_ratio_window=config.correlation_window,
        )
        
        pred_generator = PredictionGenerator(
            trigger_threshold=config.trigger_threshold,
            confirmation_threshold=config.confirmation_threshold,
            refutation_threshold=config.refutation_threshold,
        )
        
        resolution_engine = ResolutionEngine(
            confirmation_threshold=config.confirmation_threshold,
            refutation_threshold=config.refutation_threshold,
            max_holding_bars=config.max_holding_bars,
            min_correlation=config.min_correlation,
            max_correlation_drop=config.max_correlation_drop,
        )
        
        # Track stability duration at prediction creation
        stability_tracker = _StabilityTracker()
        
        # Run engine
        records: List[ResolutionRecord] = []
        
        for bar_index, observation in enumerate(observation_stream):
            # Update spread calculator
            spread_obs = spread_calc.update(observation)
            
            if spread_obs is None or not spread_obs.is_valid:
                stability_tracker.reset()
                continue
            
            # Track stability
            stability_tracker.update(spread_obs)
            
            # Check for prediction
            if pred_generator.should_generate(spread_obs):
                prediction = pred_generator.generate(spread_obs, bar_index)
                resolution_engine.add_pending(prediction)
                
                # Record stability duration at creation
                stability_tracker.record_prediction(
                    prediction.prediction_id,
                    stability_tracker.current_duration,
                )
            
            # Process resolutions
            results = resolution_engine.process(spread_obs, bar_index)
            
            for result in results:
                if result.state is not None:
                    pred_generator.mark_resolved(config.pair)
                    
                    # Create resolution record
                    stability_duration = stability_tracker.get_prediction_stability(
                        result.prediction_id
                    )
                    
                    record = ResolutionRecord(
                        prediction_id=result.prediction_id,
                        outcome=result.state.value,
                        bars_to_resolution=result.bars_elapsed,
                        stability_duration_at_creation=stability_duration,
                    )
                    records.append(record)
        
        return records


# =============================================================================
# STABILITY TRACKER (Internal utility)
# =============================================================================

class _StabilityTracker:
    """
    Tracks how long structural conditions have been stable.
    
    This is used to compute conditional CRR based on stability duration
    at prediction creation time.
    
    USES ONLY EXISTING ENGINE DATA.
    """
    
    def __init__(self):
        self.current_duration = 0
        self._prediction_stability: Dict[str, int] = {}
    
    def update(self, spread_obs) -> None:
        """Update stability count based on spread validity."""
        if spread_obs.is_valid:
            self.current_duration += 1
        else:
            self.current_duration = 0
    
    def reset(self) -> None:
        """Reset stability count."""
        self.current_duration = 0
    
    def record_prediction(self, prediction_id: str, duration: int) -> None:
        """Record stability duration at prediction creation."""
        self._prediction_stability[prediction_id] = duration
    
    def get_prediction_stability(self, prediction_id: str) -> Optional[int]:
        """Get stability duration for a prediction."""
        return self._prediction_stability.get(prediction_id)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def run_single_experiment(
    config: ExperimentConfig,
    hypotheses: List[HypothesisSpec] = None,
    output_dir: Optional[Path] = None,
    verbose: bool = True,
) -> ExperimentOutput:
    """
    Convenience function to run a single experiment.
    
    Entry point for simple experiment execution.
    """
    runner = ExperimentRunner(output_dir=output_dir, verbose=verbose)
    return runner.run(config, hypotheses)


def run_batch_experiment(
    batch: ExperimentBatch,
    hypotheses: List[HypothesisSpec] = None,
    output_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Tuple[List[ExperimentOutput], Dict]:
    """
    Convenience function to run a batch of experiments.
    
    Returns:
        Tuple of (outputs, aggregated_results_dict)
    """
    runner = ExperimentRunner(output_dir=output_dir, verbose=verbose)
    outputs = runner.run_batch(batch, hypotheses)
    
    aggregated = aggregate_experiments(batch.batch_id, outputs)
    
    return outputs, aggregated.to_dict()
