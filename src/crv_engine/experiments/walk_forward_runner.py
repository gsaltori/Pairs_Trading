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

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple, Generator
from datetime import datetime, timezone
from pathlib import Path
import sys

# Ensure imports work
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
import math


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

class BlockExecutor:
    """
    Executes a single time block with complete isolation.
    
    Creates FRESH instances of all stateful components:
    - SpreadCalculator (fresh rolling windows)
    - PredictionGenerator (no pending predictions)
    - ResolutionEngine (no pending resolutions)
    - Stability tracker (fresh)
    
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
    ) -> BlockResult:
        """
        Execute experiment on a single block.
        
        ALL state is created fresh within this method.
        Nothing persists beyond the method call.
        """
        # Import engine components (deferred to avoid circular imports)
        try:
            from spread import SpreadCalculator
            from predictions import PredictionGenerator
            from resolution import ResolutionEngine, ResolutionState
        except ImportError:
            from ..spread import SpreadCalculator
            from ..predictions import PredictionGenerator
            from ..resolution import ResolutionEngine, ResolutionState
        
        bar_count = len(observations)
        
        # Create FRESH instances for this block
        spread_calc = SpreadCalculator(
            pair=self.config.pair,
            zscore_window=self.config.zscore_window,
            hedge_ratio_window=self.config.correlation_window,
        )
        
        pred_generator = PredictionGenerator(
            trigger_threshold=self.config.trigger_threshold,
            confirmation_threshold=self.config.confirmation_threshold,
            refutation_threshold=self.config.refutation_threshold,
        )
        
        resolution_engine = ResolutionEngine(
            confirmation_threshold=self.config.confirmation_threshold,
            refutation_threshold=self.config.refutation_threshold,
            max_holding_bars=self.config.max_holding_bars,
            min_correlation=self.config.min_correlation,
            max_correlation_drop=self.config.max_correlation_drop,
        )
        
        # Track regime distribution (simplified - just count by observation state)
        regime_counts: Dict[str, int] = {}
        
        # Run engine on this block
        records: List[ResolutionRecord] = []
        
        for local_bar, observation in enumerate(observations):
            # Update spread calculator
            spread_obs = spread_calc.update(observation)
            
            if spread_obs is None or not spread_obs.is_valid:
                continue
            
            # Track "regime" as simplified spread state
            if spread_obs.zscore is not None:
                regime_code = _classify_spread_regime(spread_obs.zscore)
                regime_counts[regime_code] = regime_counts.get(regime_code, 0) + 1
            
            # Check for prediction
            if pred_generator.should_generate(spread_obs):
                prediction = pred_generator.generate(spread_obs, local_bar)
                resolution_engine.add_pending(prediction)
            
            # Process resolutions
            results = resolution_engine.process(spread_obs, local_bar)
            
            for result in results:
                if result.state is not None:
                    pred_generator.mark_resolved(self.config.pair)
                    
                    record = ResolutionRecord(
                        prediction_id=result.prediction_id,
                        outcome=result.state.value,
                        bars_to_resolution=result.bars_elapsed,
                        stability_duration_at_creation=None,
                    )
                    records.append(record)
        
        # Compute metrics from records
        metrics = compute_extended_metrics(records)
        
        # Compute regime distribution
        regime_dist = _compute_regime_distribution(regime_counts)
        
        # Determine verdict
        verdict, reason = self._determine_verdict(metrics)
        
        return BlockResult(
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
    
    def run(self, observation_stream: List[Any]) -> WalkForwardOutput:
        """
        Execute walk-forward analysis on observation stream.
        
        Args:
            observation_stream: Complete list of observations (bars)
        
        Returns:
            WalkForwardOutput with all results and stability analysis
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
            result = executor.execute(
                observations=block_obs,
                block_index=block_idx,
                start_bar=start_bar,
                end_bar=end_bar,
            )
            
            block_results.append(result)
            
            if self.verbose:
                print(f"→ {result.verdict.value} (CRR={result.crr:.1%}, n={result.testable_count})")
        
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
        
        return output
    
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
) -> WalkForwardOutput:
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
        WalkForwardOutput with complete analysis
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
