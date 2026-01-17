#!/usr/bin/env python3
"""
CRV Engine Demo - P4 Structural + P5 Regime Memory Comparison

This script demonstrates:
1. UNGATED P1 predictions (original behavior)
2. P4 GATED predictions (structural filtering only)
3. P4+P5 GATED predictions (structural + regime memory)

MOCK DATA - NOT REAL MARKET DATA
"""

import sys
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Tuple, Optional, Dict

try:
    from .config import CONFIG
    from .observations import generate_observation_stream, ObservationStream
    from .spread import SpreadCalculator, SpreadObservation
    from .predictions import PredictionGenerator, GatedPredictionGenerator
    from .resolution import ResolutionEngine, ResolutionState
    from .statistics import StatisticsAccumulator, ResolutionStatistics
    from .structural import StructuralStabilityEvaluator, StructuralValidity, STRUCTURAL_CONFIG
    from .regime import (
        RegimeMemory, RegimeEvaluator, OutcomeRecorder,
        create_regime_signature, REGIME_CONFIG, RegimeSignature,
    )
except ImportError:
    from config import CONFIG
    from observations import generate_observation_stream, ObservationStream
    from spread import SpreadCalculator, SpreadObservation
    from predictions import PredictionGenerator, GatedPredictionGenerator
    from resolution import ResolutionEngine, ResolutionState
    from statistics import StatisticsAccumulator, ResolutionStatistics
    from structural import StructuralStabilityEvaluator, StructuralValidity, STRUCTURAL_CONFIG
    from regime import (
        RegimeMemory, RegimeEvaluator, OutcomeRecorder,
        create_regime_signature, REGIME_CONFIG, RegimeSignature,
    )


@dataclass
class RunResult:
    """Results from a single demo run."""
    mode: str
    stats: ResolutionStatistics
    total_generated: int
    blocked_structural: int
    blocked_regime: int
    cold_start_allowed: int
    regime_stats: Optional[Dict] = None


def run_ungated(observation_stream: ObservationStream) -> RunResult:
    """Run engine without any gating."""
    spread_calc = SpreadCalculator()
    pred_generator = PredictionGenerator()
    resolution_engine = ResolutionEngine()
    stats_accumulator = StatisticsAccumulator()
    
    for bar_index, observation in enumerate(observation_stream):
        spread_obs = spread_calc.update(observation)
        
        if spread_obs is None or not spread_obs.is_valid:
            continue
        
        if pred_generator.should_generate(spread_obs):
            prediction = pred_generator.generate(spread_obs, bar_index)
            resolution_engine.add_pending(prediction)
            stats_accumulator.add(prediction)
        
        results = resolution_engine.process(spread_obs, bar_index)
        for result in results:
            if result.state is not None:
                pred_generator.mark_resolved(CONFIG.PAIR)
    
    return RunResult(
        mode="UNGATED",
        stats=stats_accumulator.compute(),
        total_generated=pred_generator.total_generated,
        blocked_structural=0,
        blocked_regime=0,
        cold_start_allowed=0,
    )


def run_p4_gated(observation_stream: ObservationStream) -> RunResult:
    """Run engine with P4 structural gating only."""
    spread_calc = SpreadCalculator()
    pred_generator = GatedPredictionGenerator()  # Uses internal StructuralGate
    resolution_engine = ResolutionEngine()
    stats_accumulator = StatisticsAccumulator()
    
    for bar_index, observation in enumerate(observation_stream):
        spread_obs = spread_calc.update(observation)
        
        if spread_obs is None or not spread_obs.is_valid:
            continue
        
        if pred_generator.should_generate(spread_obs, bar_index):
            prediction = pred_generator.generate(spread_obs, bar_index)
            resolution_engine.add_pending(prediction)
            stats_accumulator.add(prediction)
        
        results = resolution_engine.process(spread_obs, bar_index)
        for result in results:
            if result.state is not None:
                pred_generator.mark_resolved(CONFIG.PAIR)
    
    return RunResult(
        mode="P4_GATED",
        stats=stats_accumulator.compute(),
        total_generated=pred_generator.total_generated,
        blocked_structural=pred_generator.total_blocked,
        blocked_regime=0,
        cold_start_allowed=0,
    )


def run_p4_p5_gated(observation_stream: ObservationStream) -> RunResult:
    """Run engine with P4 structural + P5 regime memory gating."""
    spread_calc = SpreadCalculator()
    structural_eval = StructuralStabilityEvaluator()
    regime_memory = RegimeMemory()
    regime_evaluator = RegimeEvaluator(regime_memory)
    outcome_recorder = OutcomeRecorder(regime_memory)
    pred_generator = PredictionGenerator()  # Base generator
    resolution_engine = ResolutionEngine()
    stats_accumulator = StatisticsAccumulator()
    
    blocked_structural = 0
    blocked_regime = 0
    cold_start_allowed = 0
    
    # Track predictions with their regimes
    prediction_regimes: Dict[str, RegimeSignature] = {}
    
    for bar_index, observation in enumerate(observation_stream):
        # Update structural evaluator
        corr = spread_calc.get_current_correlation() if len(spread_calc._prices_a) >= CONFIG.ZSCORE_WINDOW else 0.5
        structural_eval.update(observation.timestamp, corr, 0.01, 0.01, 0.001)
        
        spread_obs = spread_calc.update(observation)
        
        if spread_obs is None or not spread_obs.is_valid:
            continue
        
        # Check base conditions
        if pred_generator.should_generate(spread_obs):
            # P4: Check structural stability
            structural_state = structural_eval.evaluate(observation.timestamp)
            
            if not structural_state.is_valid:
                blocked_structural += 1
                continue
            
            # P5: Create regime signature and check
            regime = create_regime_signature(
                structural_state.correlation_stability,
                structural_state.correlation_trend,
                structural_state.volatility_ratio_stability,
                structural_state.spread_variance_ratio,
            )
            
            regime_eval = regime_evaluator.evaluate(regime)
            
            if not regime_eval.is_allowed:
                blocked_regime += 1
                continue
            
            if "cold_start" in regime_eval.reason:
                cold_start_allowed += 1
            
            # Generate prediction
            prediction = pred_generator.generate(spread_obs, bar_index)
            resolution_engine.add_pending(prediction)
            stats_accumulator.add(prediction)
            
            # Track for regime learning
            prediction_regimes[prediction.prediction_id] = regime
            outcome_recorder.track_prediction(
                prediction.prediction_id, regime, prediction.creation_timestamp
            )
        
        # Process resolutions
        results = resolution_engine.process(spread_obs, bar_index)
        for result in results:
            if result.state is not None:
                pred_generator.mark_resolved(CONFIG.PAIR)
                
                # Record outcome
                for pred in resolution_engine.get_resolved():
                    if pred.prediction_id == result.prediction_id:
                        outcome_recorder.record_resolution(
                            pred.prediction_id,
                            pred.resolution_state,
                            pred.resolution_timestamp,
                            pred.resolution_bars_elapsed,
                        )
    
    # Get regime statistics
    regime_stats = {}
    for regime, stats in regime_memory.get_all_stats().items():
        regime_stats[regime.short_code()] = {
            'testable': stats.testable_count,
            'confirmed': stats.confirmed_count,
            'refuted': stats.refuted_count,
            'crr': stats.raw_confirmation_rate,
            'confidence': stats.confidence_score,
        }
    
    return RunResult(
        mode="P4+P5_GATED",
        stats=stats_accumulator.compute(),
        total_generated=pred_generator.total_generated,
        blocked_structural=blocked_structural,
        blocked_regime=blocked_regime,
        cold_start_allowed=cold_start_allowed,
        regime_stats=regime_stats,
    )


def print_comparison(ungated: RunResult, p4: RunResult, p4p5: RunResult) -> None:
    """Print comparison of all three modes."""
    print()
    print("=" * 90)
    print("                    P4 + P5 GATING COMPARISON")
    print("=" * 90)
    print()
    print(f"{'Metric':<35} {'UNGATED':>15} {'P4 ONLY':>15} {'P4+P5':>15}")
    print("-" * 90)
    
    print(f"{'Predictions Generated':<35} {ungated.total_generated:>15} {p4.total_generated:>15} {p4p5.total_generated:>15}")
    print(f"{'Blocked by P4 (Structural)':<35} {ungated.blocked_structural:>15} {p4.blocked_structural:>15} {p4p5.blocked_structural:>15}")
    print(f"{'Blocked by P5 (Regime)':<35} {'-':>15} {'-':>15} {p4p5.blocked_regime:>15}")
    print(f"{'Allowed (Cold Start)':<35} {'-':>15} {'-':>15} {p4p5.cold_start_allowed:>15}")
    
    print()
    print("-" * 90)
    
    u, p, pp = ungated.stats, p4.stats, p4p5.stats
    
    print(f"{'Resolved':<35} {u.resolved_count:>15} {p.resolved_count:>15} {pp.resolved_count:>15}")
    print(f"{'  CONFIRMED':<35} {u.confirmed_count:>15} {p.confirmed_count:>15} {pp.confirmed_count:>15}")
    print(f"{'  REFUTED':<35} {u.refuted_count:>15} {p.refuted_count:>15} {pp.refuted_count:>15}")
    print(f"{'  TIMEOUT':<35} {u.timeout_count:>15} {p.timeout_count:>15} {pp.timeout_count:>15}")
    print(f"{'  INVALIDATED':<35} {u.invalidated_count:>15} {p.invalidated_count:>15} {pp.invalidated_count:>15}")
    
    print()
    print("-" * 90)
    
    print(f"{'Testable Count':<35} {u.testable_count:>15} {p.testable_count:>15} {pp.testable_count:>15}")
    print(f"{'CRR (Edge Metric)':<35} {u.crr:>14.1%} {p.crr:>14.1%} {pp.crr:>14.1%}")
    print(f"{'Invalidation Rate':<35} {u.invalidation_rate:>14.1%} {p.invalidation_rate:>14.1%} {pp.invalidation_rate:>14.1%}")
    
    print()
    print("=" * 90)
    
    # Regime stats
    if p4p5.regime_stats:
        print()
        print("P5 REGIME MEMORY STATISTICS:")
        print("-" * 60)
        print(f"{'Regime':<10} {'Testable':>10} {'Confirmed':>10} {'CRR':>10} {'Conf':>10}")
        print("-" * 60)
        
        for code, stats in sorted(p4p5.regime_stats.items(), key=lambda x: -x[1]['testable']):
            if stats['testable'] >= 3:
                print(f"{code:<10} {stats['testable']:>10} {stats['confirmed']:>10} "
                      f"{stats['crr']:>9.1%} {stats['confidence']:>9.1%}")
        
        # Show regimes that would be blocked
        print()
        print("REGIMES THAT WOULD BE BLOCKED (with sufficient samples):")
        for code, stats in sorted(p4p5.regime_stats.items(), key=lambda x: x[1]['confidence']):
            if stats['testable'] >= 5 and stats['confidence'] < REGIME_CONFIG.MIN_CONFIDENCE_THRESHOLD:
                print(f"  {code}: CRR={stats['crr']:.1%}, Conf={stats['confidence']:.1%}, N={stats['testable']}")
    
    # Summary
    print()
    print("=" * 90)
    print("INTERPRETATION:")
    print("-" * 90)
    
    p4_inv_reduction = u.invalidation_rate - p.invalidation_rate
    total_inv_reduction = u.invalidation_rate - pp.invalidation_rate
    
    print(f"  P4 Invalidation Reduction: {p4_inv_reduction:+.1%}")
    print(f"  P5 Regime Blocks: {p4p5.blocked_regime}")
    print(f"  TOTAL Invalidation Reduction: {total_inv_reduction:+.1%}")
    
    if p4p5.blocked_regime > 0:
        print("  P5 VERDICT: Regime memory is ACTIVE and blocking unproductive contexts")
    elif p4p5.cold_start_allowed > 10:
        print("  P5 VERDICT: Regime memory in LEARNING phase")
    else:
        print("  P5 VERDICT: Insufficient history for regime learning")
    
    print()
    print("  NOTE: This is MOCK data. P5 needs extended run (5000+ bars) to fully learn.")
    print("=" * 90)


def run_demo(n_bars: int = 2000, seed: int = 42) -> None:
    """Run the comparison demo."""
    print("=" * 90)
    print("        FX CONDITIONAL RELATIVE VALUE - RESEARCH ENGINE DEMO")
    print("               P4 Structural + P5 Regime Memory Analysis")
    print("=" * 90)
    print()
    print(f"Configuration:")
    print(f"  Pair:               {CONFIG.PAIR}")
    print(f"  Demo Bars:          {n_bars}")
    print(f"  Random Seed:        {seed}")
    print()
    print("P5 Regime Config:")
    print(f"  Max Memory:         {REGIME_CONFIG.MAX_MEMORY_SIZE}")
    print(f"  Min Samples:        {REGIME_CONFIG.MIN_SAMPLES_FOR_CONFIDENCE}")
    print(f"  Min Confidence:     {REGIME_CONFIG.MIN_CONFIDENCE_THRESHOLD:.0%}")
    print()
    
    print("Generating synthetic data...")
    obs_ungated = generate_observation_stream(n_bars=n_bars, seed=seed)
    obs_p4 = generate_observation_stream(n_bars=n_bars, seed=seed)
    obs_p4p5 = generate_observation_stream(n_bars=n_bars, seed=seed)
    
    print("Running UNGATED mode...")
    ungated = run_ungated(obs_ungated)
    print(f"  Generated: {ungated.total_generated}")
    
    print("Running P4 GATED mode...")
    p4 = run_p4_gated(obs_p4)
    print(f"  Generated: {p4.total_generated}, Blocked: {p4.blocked_structural}")
    
    print("Running P4+P5 GATED mode...")
    p4p5 = run_p4_p5_gated(obs_p4p5)
    print(f"  Generated: {p4p5.total_generated}, Struct: {p4p5.blocked_structural}, Regime: {p4p5.blocked_regime}")
    
    print_comparison(ungated, p4, p4p5)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CRV Engine Demo with P5")
    parser.add_argument("--bars", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    run_demo(n_bars=args.bars, seed=args.seed)
