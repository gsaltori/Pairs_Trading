"""
Walk-Forward Runner

Executes the CRV experiment pipeline across sequential time blocks
with guaranteed isolation between blocks.

KEY ISOLATION PRINCIPLES:
1. Fresh engine state per block (spread calculator, predictor, resolver)
2. Regime memory RESET per block (no learning leakage)
3. No warm-up data from previous blocks
4. Each block is fully out-of-sample with respect to other blocks
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timezone
from pathlib import Path
import sys
import math

sys.path.insert(0, str(Path(__file__).parent.parent))

from .walk_forward import (
    WalkForwardConfig,
    BlockResult,
    BlockVerdict,
    RegimeDistribution,
    TemporalStabilityMetrics,
    WalkForwardOutput,
    EdgeStabilityClass,
    compute_temporal_stability,
    WALK_FORWARD_SCHEMA_VERSION,
)
from .config import ExperimentConfig, BASELINE
from .metrics import ResolutionRecord, compute_extended_metrics
from .edge_boundary import PredictionObservables


# =============================================================================
# OBSERVATION SLICING
# =============================================================================

def slice_observations(
    observation_stream: List[Any],
    start_bar: int,
    end_bar: int,
) -> List[Any]:
    """
    Extract a slice of observations for a specific block.
    
    This is the ONLY place where data flows from the full stream to a block.
    """
    return observation_stream[start_bar:end_bar]


# =============================================================================
# BLOCK EXECUTOR
# =============================================================================

@dataclass
class BlockExecutionResult:
    """Result from block execution including predictions with observables."""
    block_result: BlockResult
    prediction_observables: List[PredictionObservables]


class BlockExecutor:
    """
    Executes a single time block with complete isolation.
    
    Creates FRESH instances of all stateful components:
    - SpreadCalculator (fresh rolling windows)
    - PredictionGenerator (no pending predictions)
    - P1_Resolver (stateless by design)
    
    Regime memory is NOT used (reset per block = don't instantiate).
    """
    
    def __init__(
        self,
        config: ExperimentConfig,
        crr_threshold: float = 0.55,
        max_invalidation_threshold: float = 0.40,
        min_testable: int = 10,
    ):
        self.config = config
        self.crr_threshold = crr_threshold
        self.max_invalidation_threshold = max_invalidation_threshold
        self.min_testable = min_testable
    
    def execute(
        self,
        observations: List[Any],
        block_index: int,
        start_bar: int,
        end_bar: int,
    ) -> BlockExecutionResult:
        """
        Execute experiment on a single block.
        
        ALL state is created fresh within this method.
        Nothing persists beyond the method call.
        """
        # Import engine components (deferred to avoid circular imports)
        try:
            from spread import SpreadCalculator
            from predictions import PredictionGenerator
            from resolution import P1_Resolver, ResolutionState
        except ImportError:
            from ..spread import SpreadCalculator
            from ..predictions import PredictionGenerator
            from ..resolution import P1_Resolver, ResolutionState
        
        bar_count = len(observations)
        
        # Create FRESH instances for this block
        spread_calc = SpreadCalculator(
            symbol_a=self.config.symbol_a,
            symbol_b=self.config.symbol_b,
            window=self.config.zscore_window,
        )
        
        pred_generator = PredictionGenerator()
        resolver = P1_Resolver()
        
        # Track regime distribution (simplified - just count by observation state)
        regime_counts: Dict[str, int] = {}
        
        # Track pending predictions and their observables
        pending_predictions: Dict[str, Any] = {}  # prediction_id -> prediction
        pending_observables: Dict[str, Dict] = {}  # prediction_id -> observable data at creation
        
        # Run engine on this block
        records: List[ResolutionRecord] = []
        prediction_observables: List[PredictionObservables] = []
        
        # Track correlation history for trend calculation
        correlation_history: List[float] = []
        
        for local_bar, observation in enumerate(observations):
            # Update spread calculator
            spread_obs = spread_calc.update(observation)
            
            if spread_obs is None or not spread_obs.is_valid:
                continue
            
            # Track correlation history
            correlation_history.append(spread_obs.correlation)
            
            # Track "regime" as simplified spread state
            if spread_obs.zscore is not None:
                regime_code = _classify_spread_regime(spread_obs.zscore)
                regime_counts[regime_code] = regime_counts.get(regime_code, 0) + 1
            
            # Compute volatility ratio (using price changes)
            vol_ratio = self._compute_volatility_ratio(observations, local_bar)
            
            # Compute correlation trend
            corr_trend = self._compute_correlation_trend(correlation_history)
            
            # Compute spread velocity
            spread_velocity = self._compute_spread_velocity(spread_calc, spread_obs)
            
            # Check for prediction
            if pred_generator.should_generate(spread_obs):
                prediction = pred_generator.generate(spread_obs, local_bar)
                pending_predictions[prediction.prediction_id] = prediction
                
                # Capture observables at creation time
                pending_observables[prediction.prediction_id] = {
                    'correlation': spread_obs.correlation,
                    'correlation_trend': corr_trend,
                    'volatility_ratio': vol_ratio,
                    'zscore': spread_obs.zscore,
                    'spread_velocity': spread_velocity,
                }
            
            # Process resolutions for pending predictions
            resolved_ids = []
            for pred_id, prediction in pending_predictions.items():
                bars_elapsed = local_bar - prediction.creation_bar_index
                
                result = resolver.evaluate(prediction, spread_obs, bars_elapsed)
                
                if result is not None and result != ResolutionState.PENDING:
                    # Resolve the prediction
                    prediction.resolve(
                        state=result,
                        timestamp=spread_obs.timestamp,
                        observation_id=spread_obs.observation_id,
                        bar_index=local_bar,
                        bars_elapsed=bars_elapsed,
                        zscore_final=spread_obs.zscore,
                        spread_final=spread_obs.spread_value,
                        correlation_final=spread_obs.correlation,
                    )
                    
                    pred_generator.mark_resolved(self.config.pair)
                    resolved_ids.append(pred_id)
                    
                    # Create resolution record
                    record = ResolutionRecord(
                        prediction_id=pred_id,
                        outcome=result.value,
                        bars_to_resolution=bars_elapsed,
                        stability_duration_at_creation=None,
                    )
                    records.append(record)
                    
                    # Create prediction observable
                    obs_data = pending_observables[pred_id]
                    pred_obs = PredictionObservables(
                        prediction_id=pred_id,
                        outcome=result.value,
                        bars_to_resolution=bars_elapsed,
                        correlation=obs_data['correlation'],
                        correlation_trend=obs_data['correlation_trend'],
                        volatility_ratio=obs_data['volatility_ratio'],
                        zscore=obs_data['zscore'],
                        spread_velocity=obs_data['spread_velocity'],
                    )
                    prediction_observables.append(pred_obs)
            
            # Remove resolved predictions
            for pred_id in resolved_ids:
                del pending_predictions[pred_id]
                del pending_observables[pred_id]
        
        # Compute metrics from records
        metrics = compute_extended_metrics(records)
        
        # Compute regime distribution
        regime_dist = _compute_regime_distribution(regime_counts)
        
        # Determine verdict
        verdict, reason = self._determine_verdict(metrics)
        
        block_result = BlockResult(
            block_index=block_index,
            start_bar=start_bar,
            end_bar=end_bar,
            bar_count=bar_count,
            total_predictions=metrics.total_predictions,
            testable_count=metrics.testable_count,
            confirmed_count=metrics.confirmed_count,
            refuted_count=metrics.refuted_count,
            invalidated_count=metrics.invalidated_count,
            timeout_count=metrics.timeout_count,
            crr=metrics.crr,
            erv=metrics.erv,
            erv_per_prediction=metrics.erv_per_prediction,
            invalidation_rate=metrics.invalidation_rate,
            median_bars_to_confirmation=metrics.median_bars_to_confirmation,
            median_bars_to_refutation=metrics.median_bars_to_refutation,
            timing_asymmetry_ratio=metrics.timing_asymmetry_ratio,
            regime_distribution=regime_dist,
            verdict=verdict,
            verdict_reason=reason,
            crr_threshold=self.crr_threshold,
            max_invalidation_threshold=self.max_invalidation_threshold,
        )
        
        return BlockExecutionResult(
            block_result=block_result,
            prediction_observables=prediction_observables,
        )
    
    def _compute_volatility_ratio(
        self,
        observations: List[Any],
        current_bar: int,
        window: int = 20,
    ) -> float:
        """Compute volatility ratio between the two symbols."""
        if current_bar < window:
            return 1.0
        
        import numpy as np
        
        prices_a = [obs.bar_a.close for obs in observations[current_bar-window:current_bar]]
        prices_b = [obs.bar_b.close for obs in observations[current_bar-window:current_bar]]
        
        returns_a = np.diff(prices_a) / np.array(prices_a[:-1])
        returns_b = np.diff(prices_b) / np.array(prices_b[:-1])
        
        vol_a = np.std(returns_a) if len(returns_a) > 0 else 0.01
        vol_b = np.std(returns_b) if len(returns_b) > 0 else 0.01
        
        if vol_b < 1e-10:
            return 1.0
        
        return float(vol_a / vol_b)
    
    def _compute_correlation_trend(
        self,
        correlation_history: List[float],
        window: int = 10,
    ) -> float:
        """Compute correlation trend (change over recent window)."""
        if len(correlation_history) < window:
            return 0.0
        
        recent = correlation_history[-window:]
        return recent[-1] - recent[0]
    
    def _compute_spread_velocity(
        self,
        spread_calc,
        spread_obs,
        window: int = 5,
    ) -> float:
        """Compute spread velocity (rate of Z-score change)."""
        # Simplified: just use current zscore normalized
        return spread_obs.zscore / 3.0 if spread_obs.zscore else 0.0
    
    def _determine_verdict(self, metrics) -> Tuple[BlockVerdict, str]:
        """Determine block verdict based on hypothesis thresholds."""
        if metrics.testable_count < self.min_testable:
            return (
                BlockVerdict.INSUFFICIENT_DATA,
                f"Testable count {metrics.testable_count} < {self.min_testable}"
            )
        
        crr_ok = metrics.crr >= self.crr_threshold
        inv_ok = metrics.invalidation_rate <= self.max_invalidation_threshold
        
        if crr_ok and inv_ok:
            return (
                BlockVerdict.SUPPORTED,
                f"CRR {metrics.crr:.1%} >= {self.crr_threshold:.0%}, "
                f"Invalidation {metrics.invalidation_rate:.1%} <= {self.max_invalidation_threshold:.0%}"
            )
        else:
            reasons = []
            if not crr_ok:
                reasons.append(f"CRR {metrics.crr:.1%} < {self.crr_threshold:.0%}")
            if not inv_ok:
                reasons.append(f"Invalidation {metrics.invalidation_rate:.1%} > {self.max_invalidation_threshold:.0%}")
            return (BlockVerdict.REFUTED, ", ".join(reasons))


def _classify_spread_regime(zscore: float) -> str:
    """
    Simple spread regime classification based on Z-score magnitude.
    
    This is NOT a new indicator - just bucketing existing data.
    """
    abs_z = abs(zscore)
    if abs_z < 1.0:
        return "NEUTRAL"
    elif abs_z < 2.0:
        return "MODERATE"
    else:
        return "EXTREME"


def _compute_regime_distribution(regime_counts: Dict[str, int]) -> RegimeDistribution:
    """Compute regime distribution statistics."""
    if not regime_counts:
        return RegimeDistribution(
            regime_counts={},
            dominant_regime=None,
            regime_entropy=0.0,
        )
    
    total = sum(regime_counts.values())
    
    # Dominant regime
    dominant = max(regime_counts, key=regime_counts.get)
    
    # Entropy (Shannon)
    entropy = 0.0
    for count in regime_counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    
    return RegimeDistribution(
        regime_counts=regime_counts,
        dominant_regime=dominant,
        regime_entropy=entropy,
    )


# =============================================================================
# WALK-FORWARD RUNNER
# =============================================================================

@dataclass
class WalkForwardResult:
    """Complete walk-forward result including predictions for edge boundary analysis."""
    output: WalkForwardOutput
    all_predictions: List[PredictionObservables]


class WalkForwardRunner:
    """
    Orchestrates walk-forward analysis across time blocks.
    
    RESPONSIBILITIES:
    1. Compute block boundaries
    2. Iterate over blocks
    3. Execute each block with isolation
    4. Aggregate results
    5. Compute stability metrics
    6. Produce final output
    
    DOES NOT:
    - Modify engine logic
    - Persist state between blocks
    - Optimize anything
    """
    
    def __init__(
        self,
        walk_forward_config: WalkForwardConfig,
        experiment_config: ExperimentConfig,
        crr_threshold: float = 0.55,
        max_invalidation_threshold: float = 0.40,
        output_dir: Optional[Path] = None,
        verbose: bool = False,
    ):
        self.wf_config = walk_forward_config
        self.exp_config = experiment_config
        self.crr_threshold = crr_threshold
        self.max_invalidation_threshold = max_invalidation_threshold
        self.output_dir = Path(output_dir) if output_dir else None
        self.verbose = verbose
    
    def run(self, observation_stream: List[Any]) -> WalkForwardResult:
        """
        Execute walk-forward analysis on observation stream.
        
        Args:
            observation_stream: Complete list of observations (bars)
        
        Returns:
            WalkForwardResult with output and all prediction observables
        """
        warnings = []
        errors = []
        
        total_bars = len(observation_stream)
        
        if self.verbose:
            print(f"Walk-Forward Analysis")
            print(f"  Total bars: {total_bars}")
            print(f"  Block size: {self.wf_config.block_size}")
        
        # Compute block boundaries
        blocks = self.wf_config.compute_blocks(total_bars)
        
        if self.verbose:
            print(f"  Blocks: {len(blocks)}")
        
        if len(blocks) < self.wf_config.min_blocks_for_stability:
            warnings.append(
                f"Only {len(blocks)} blocks < {self.wf_config.min_blocks_for_stability} minimum. "
                "Stability analysis will be limited."
            )
        
        # Execute each block
        block_results: List[BlockResult] = []
        all_predictions: List[PredictionObservables] = []
        
        executor = BlockExecutor(
            config=self.exp_config,
            crr_threshold=self.crr_threshold,
            max_invalidation_threshold=self.max_invalidation_threshold,
            min_testable=self.wf_config.min_testable_per_block,
        )
        
        for block_idx, (start_bar, end_bar) in enumerate(blocks):
            if self.verbose:
                print(f"  Block {block_idx}: bars [{start_bar}:{end_bar}]", end=" ")
            
            # Slice observations for this block
            block_obs = slice_observations(observation_stream, start_bar, end_bar)
            
            # Execute with complete isolation
            exec_result = executor.execute(
                observations=block_obs,
                block_index=block_idx,
                start_bar=start_bar,
                end_bar=end_bar,
            )
            
            block_results.append(exec_result.block_result)
            all_predictions.extend(exec_result.prediction_observables)
            
            if self.verbose:
                print(f"→ {exec_result.block_result.verdict.value} "
                      f"(CRR={exec_result.block_result.crr:.1%}, "
                      f"n={exec_result.block_result.testable_count})")
        
        # Compute temporal stability metrics
        stability_metrics = compute_temporal_stability(block_results, self.wf_config)
        
        if self.verbose:
            print()
            print(f"Stability Analysis:")
            print(f"  Classification: {stability_metrics.stability_class.value}")
            print(f"  Confidence: {stability_metrics.classification_confidence:.1%}")
            print(f"  Verdict Persistence: {stability_metrics.verdict_persistence_ratio:.1%}")
            if stability_metrics.crr_drift_slope is not None:
                print(f"  CRR Drift Slope: {stability_metrics.crr_drift_slope:.4f}")
        
        # Build flags
        flags = self._compute_flags(stability_metrics, block_results)
        
        # Build output
        output = WalkForwardOutput(
            schema_version=WALK_FORWARD_SCHEMA_VERSION,
            run_timestamp=datetime.now(timezone.utc).isoformat(),
            config_hash=self._compute_combined_hash(),
            walk_forward_config={
                'block_size': self.wf_config.block_size,
                'min_bars_per_block': self.wf_config.min_bars_per_block,
                'warmup_bars': self.wf_config.warmup_bars,
                'min_testable_per_block': self.wf_config.min_testable_per_block,
                'min_blocks_for_stability': self.wf_config.min_blocks_for_stability,
            },
            experiment_config={
                'experiment_id': self.exp_config.experiment_id,
                'symbol_a': self.exp_config.symbol_a,
                'symbol_b': self.exp_config.symbol_b,
                'timeframe': self.exp_config.timeframe.value,
                'zscore_window': self.exp_config.zscore_window,
                'correlation_window': self.exp_config.correlation_window,
            },
            hypothesis_config={
                'crr_threshold': self.crr_threshold,
                'max_invalidation_threshold': self.max_invalidation_threshold,
            },
            block_results=[b.to_dict() for b in block_results],
            stability_metrics=stability_metrics.to_dict(),
            flags=flags,
            warnings=warnings,
            errors=errors,
        )
        
        # Save if output directory specified
        if self.output_dir:
            filepath = output.save(self.output_dir)
            if self.verbose:
                print(f"\nOutput saved: {filepath}")
        
        return WalkForwardResult(output=output, all_predictions=all_predictions)
    
    def _compute_combined_hash(self) -> str:
        """Compute combined hash of all configurations."""
        import hashlib
        import json
        
        combined = {
            'wf': self.wf_config.config_hash,
            'exp': self.exp_config.config_hash,
            'crr': self.crr_threshold,
            'inv': self.max_invalidation_threshold,
        }
        config_str = json.dumps(combined, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:12]
    
    def _compute_flags(
        self,
        stability: TemporalStabilityMetrics,
        blocks: List[BlockResult],
    ) -> Dict[str, bool]:
        """Compute summary flags for output."""
        return {
            'has_persistent_edge': stability.stability_class == EdgeStabilityClass.PERSISTENT,
            'has_decaying_edge': stability.stability_class == EdgeStabilityClass.DECAYING,
            'is_regime_dependent': stability.stability_class == EdgeStabilityClass.REGIME_DEPENDENT,
            'is_unstable': stability.stability_class == EdgeStabilityClass.UNSTABLE,
            'is_unfalsifiable': stability.stability_class == EdgeStabilityClass.UNFALSIFIABLE,
            'all_blocks_sufficient': all(b.is_sufficient for b in blocks),
            'majority_supported': stability.verdict_persistence_ratio >= 0.5,
            'no_monotonic_decay': not stability.is_monotonic_decay,
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def run_walk_forward(
    observation_stream: List[Any],
    experiment_config: ExperimentConfig,
    block_size: int = 500,
    crr_threshold: float = 0.55,
    max_invalidation_threshold: float = 0.40,
    output_dir: Optional[Path] = None,
    verbose: bool = True,
) -> WalkForwardResult:
    """
    Convenience function to run walk-forward analysis.
    
    Args:
        observation_stream: Complete historical data
        experiment_config: Experiment configuration
        block_size: Number of bars per block
        crr_threshold: Hypothesis CRR threshold
        max_invalidation_threshold: Max acceptable invalidation rate
        output_dir: Directory for output files
        verbose: Print progress
    
    Returns:
        WalkForwardResult with output and all prediction observables
    """
    wf_config = WalkForwardConfig(block_size=block_size)
    
    runner = WalkForwardRunner(
        walk_forward_config=wf_config,
        experiment_config=experiment_config,
        crr_threshold=crr_threshold,
        max_invalidation_threshold=max_invalidation_threshold,
        output_dir=output_dir,
        verbose=verbose,
    )
    
    return runner.run(observation_stream)
