#!/usr/bin/env python3
"""
CRV Engine Demo - P4 Structural Gating Impact Analysis

This script demonstrates:
1. UNGATED P1 predictions (original behavior)
2. GATED P1 predictions (with P4 structural filtering)
3. Side-by-side comparison of CRR and invalidation rates

KEY QUESTION: Does P4 gating reduce invalidation without inflating CRR?

MOCK DATA - NOT REAL MARKET DATA
"""

import sys
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Tuple, Dict

try:
    from .config import CONFIG
    from .observations import generate_observation_stream, ObservationStream
    from .spread import SpreadCalculator, SpreadObservation
    from .predictions import PredictionGenerator, GatedPredictionGenerator
    from .resolution import ResolutionEngine, ResolutionState
    from .statistics import StatisticsAccumulator, ResolutionStatistics
    from .structural import STRUCTURAL_CONFIG
except ImportError:
    from config import CONFIG
    from observations import generate_observation_stream, ObservationStream
    from spread import SpreadCalculator, SpreadObservation
    from predictions import PredictionGenerator, GatedPredictionGenerator
    from resolution import ResolutionEngine, ResolutionState
    from statistics import StatisticsAccumulator, ResolutionStatistics
    from structural import STRUCTURAL_CONFIG


@dataclass
class DemoResult:
    """Results from a single demo run."""
    mode: str
    stats: ResolutionStatistics
    total_generated: int
    total_blocked: int
    block_reasons: Dict[str, int]


def run_ungated(
    observation_stream: ObservationStream,
) -> DemoResult:
    """
    Run P1 predictions WITHOUT structural gating.
    
    This is the baseline for comparison.
    """
    spread_calc = SpreadCalculator()
    pred_gen = PredictionGenerator()
    resolution_engine = ResolutionEngine()
    stats_acc = StatisticsAccumulator()
    
    for bar_index, obs in enumerate(observation_stream):
        spread_obs = spread_calc.update(obs)
        
        if spread_obs is None or not spread_obs.is_valid:
            continue
        
        # Check for P1 generation (no structural gate)
        if pred_gen.should_generate(spread_obs):
            prediction = pred_gen.generate(spread_obs, bar_index)
            resolution_engine.add_pending(prediction)
            stats_acc.add(prediction)
        
        # Process resolutions
        results = resolution_engine.process(spread_obs, bar_index)
        for result in results:
            if result.state is not None:
                pred_gen.mark_resolved(CONFIG.PAIR)
    
    return DemoResult(
        mode="UNGATED",
        stats=stats_acc.compute(),
        total_generated=pred_gen.total_generated,
        total_blocked=0,
        block_reasons={},
    )


def run_gated(
    observation_stream: ObservationStream,
) -> DemoResult:
    """
    Run P1 predictions WITH P4 structural gating.
    """
    spread_calc = SpreadCalculator()
    pred_gen = GatedPredictionGenerator()
    resolution_engine = ResolutionEngine()
    stats_acc = StatisticsAccumulator()
    
    for bar_index, obs in enumerate(observation_stream):
        # Update price history for volatility calculation
        pred_gen.update_prices(obs.bar_a.close, obs.bar_b.close)
        
        spread_obs = spread_calc.update(obs)
        
        if spread_obs is None or not spread_obs.is_valid:
            continue
        
        # Check for P1 generation (WITH structural gate)
        if pred_gen.should_generate(spread_obs, bar_index):
            prediction = pred_gen.generate(spread_obs, bar_index)
            resolution_engine.add_pending(prediction)
            stats_acc.add(prediction)
        
        # Process resolutions
        results = resolution_engine.process(spread_obs, bar_index)
        for result in results:
            if result.state is not None:
                pred_gen.mark_resolved(CONFIG.PAIR)
    
    return DemoResult(
        mode="GATED",
        stats=stats_acc.compute(),
        total_generated=pred_gen.total_generated,
        total_blocked=pred_gen.total_blocked,
        block_reasons=pred_gen.get_block_reasons_summary(),
    )


def print_comparison(ungated: DemoResult, gated: DemoResult) -> None:
    """Print side-by-side comparison of results."""
    u = ungated.stats
    g = gated.stats
    
    print()
    print("=" * 78)
    print("              P4 STRUCTURAL GATING IMPACT ANALYSIS")
    print("=" * 78)
    print()
    
    # Header
    print(f"{'METRIC':<35} {'UNGATED':>15} {'GATED':>15} {'Δ':>10}")
    print("-" * 78)
    
    # Generation
    print(f"{'Total Predictions Generated':<35} {ungated.total_generated:>15} {gated.total_generated:>15} {gated.total_generated - ungated.total_generated:>+10}")
    print(f"{'Predictions Blocked by P4':<35} {ungated.total_blocked:>15} {gated.total_blocked:>15} {gated.total_blocked:>+10}")
    
    if gated.total_blocked > 0:
        potential = gated.total_generated + gated.total_blocked
        block_rate = gated.total_blocked / potential
        print(f"{'P4 Block Rate':<35} {'N/A':>15} {block_rate:>14.1%} {'-':>10}")
    
    print()
    print("-" * 78)
    
    # Resolution breakdown
    print(f"{'Resolved':<35} {u.resolved_count:>15} {g.resolved_count:>15} {g.resolved_count - u.resolved_count:>+10}")
    print(f"{'  CONFIRMED':<35} {u.confirmed_count:>15} {g.confirmed_count:>15} {g.confirmed_count - u.confirmed_count:>+10}")
    print(f"{'  REFUTED':<35} {u.refuted_count:>15} {g.refuted_count:>15} {g.refuted_count - u.refuted_count:>+10}")
    print(f"{'  TIMEOUT':<35} {u.timeout_count:>15} {g.timeout_count:>15} {g.timeout_count - u.timeout_count:>+10}")
    print(f"{'  INVALIDATED':<35} {u.invalidated_count:>15} {g.invalidated_count:>15} {g.invalidated_count - u.invalidated_count:>+10}")
    
    print()
    print("-" * 78)
    
    # Key metrics
    print(f"{'Testable Count':<35} {u.testable_count:>15} {g.testable_count:>15} {g.testable_count - u.testable_count:>+10}")
    
    crr_delta = g.crr - u.crr
    print(f"{'CRR (Edge Metric)':<35} {u.crr:>14.1%} {g.crr:>14.1%} {crr_delta:>+9.1%}")
    
    inv_delta = g.invalidation_rate - u.invalidation_rate
    print(f"{'Invalidation Rate':<35} {u.invalidation_rate:>14.1%} {g.invalidation_rate:>14.1%} {inv_delta:>+9.1%}")
    
    to_delta = g.timeout_rate - u.timeout_rate
    print(f"{'Timeout Rate':<35} {u.timeout_rate:>14.1%} {g.timeout_rate:>14.1%} {to_delta:>+9.1%}")
    
    print()
    print("=" * 78)
    
    # Block reasons breakdown
    if gated.block_reasons:
        print()
        print("P4 BLOCK REASONS:")
        print("-" * 40)
        total_reasons = sum(gated.block_reasons.values())
        for reason, count in sorted(gated.block_reasons.items(), key=lambda x: -x[1]):
            pct = count / total_reasons if total_reasons > 0 else 0
            print(f"  {reason:<28} {count:>5} ({pct:>5.1%})")
    
    # Interpretation
    print()
    print("=" * 78)
    print("INTERPRETATION:")
    print("-" * 78)
    
    # Check invalidation reduction
    if u.invalidation_rate > 0:
        inv_reduction_pct = (u.invalidation_rate - g.invalidation_rate) / u.invalidation_rate
    else:
        inv_reduction_pct = 0
    
    if g.invalidation_rate < u.invalidation_rate:
        print(f"  [✓] P4 REDUCED invalidation by {abs(inv_delta):.1%} (relative: {inv_reduction_pct:.0%})")
    else:
        print(f"  [X] P4 did NOT reduce invalidation rate")
    
    # Check CRR stability
    if abs(crr_delta) < 0.05:
        print(f"  [✓] CRR remained stable (Δ={crr_delta:+.1%}) - NOT artificially inflated")
    elif crr_delta > 0.05:
        print(f"  [?] CRR increased by {crr_delta:.1%} - may indicate survivor bias")
    else:
        print(f"  [~] CRR decreased by {abs(crr_delta):.1%}")
    
    # Final verdict
    print()
    effectiveness = False
    if g.invalidation_rate < u.invalidation_rate * 0.70:  # 30%+ reduction
        if abs(crr_delta) < 0.10:  # CRR stable within 10%
            effectiveness = True
            print("  VERDICT: P4 gating is EFFECTIVE")
            print("    - Materially reduces invalidation")
            print("    - Does not artificially inflate CRR")
        else:
            print("  VERDICT: P4 gating shows PROMISE but CRR change needs investigation")
    elif g.invalidation_rate < u.invalidation_rate:
        print("  VERDICT: P4 gating shows MODERATE improvement")
    else:
        print("  VERDICT: P4 gating NOT effective in this sample")
    
    print()
    print("-" * 78)
    print("  NOTE: This is MOCK data. Validate with live MT5 data.")
    print("=" * 78)


def run_demo(
    n_bars: int = 2000,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[DemoResult, DemoResult]:
    """
    Run the complete demo with both gated and ungated modes.
    """
    print("=" * 78)
    print("        FX CONDITIONAL RELATIVE VALUE - RESEARCH ENGINE DEMO")
    print("                   P4 Structural Gating Analysis")
    print("=" * 78)
    print()
    
    print(f"Configuration:")
    print(f"  Pair:                  {CONFIG.PAIR}")
    print(f"  Timeframe:             {CONFIG.TIMEFRAME}")
    print(f"  Z-Score Window:        {CONFIG.ZSCORE_WINDOW} bars")
    print(f"  Trigger Threshold:     |Z| > {CONFIG.TRIGGER_THRESHOLD}")
    print(f"  Confirmation:          |Z| < {CONFIG.CONFIRMATION_THRESHOLD}")
    print(f"  Refutation:            |Z| > {CONFIG.REFUTATION_THRESHOLD}")
    print(f"  Max Holding:           {CONFIG.MAX_HOLDING_BARS} bars")
    print(f"  Demo Bars:             {n_bars}")
    print(f"  Random Seed:           {seed}")
    print()
    
    print("P4 Structural Thresholds:")
    print(f"  Max Correlation Std:   {STRUCTURAL_CONFIG.MAX_CORRELATION_STD}")
    print(f"  Min Correlation Trend: {STRUCTURAL_CONFIG.MIN_CORRELATION_TREND}")
    print(f"  Max Vol Ratio Std:     {STRUCTURAL_CONFIG.MAX_VOLATILITY_RATIO_STD}")
    print(f"  Max Spread Var Ratio:  {STRUCTURAL_CONFIG.MAX_SPREAD_VARIANCE_RATIO}")
    print()
    
    print("Generating synthetic data...")
    print("  [MOCK: This is NOT real market data]")
    print()
    
    # Run UNGATED
    print("Running UNGATED mode...")
    obs_stream_1 = generate_observation_stream(n_bars=n_bars, seed=seed)
    ungated_result = run_ungated(obs_stream_1)
    print(f"  Generated: {ungated_result.total_generated} predictions")
    print(f"  Resolved:  {ungated_result.stats.resolved_count}")
    print(f"  Invalidated: {ungated_result.stats.invalidated_count}")
    print()
    
    # Run GATED
    print("Running GATED mode (P4 structural filtering)...")
    obs_stream_2 = generate_observation_stream(n_bars=n_bars, seed=seed)
    gated_result = run_gated(obs_stream_2)
    print(f"  Generated: {gated_result.total_generated} predictions")
    print(f"  Blocked:   {gated_result.total_blocked} predictions")
    print(f"  Resolved:  {gated_result.stats.resolved_count}")
    print(f"  Invalidated: {gated_result.stats.invalidated_count}")
    print()
    
    # Print comparison
    print_comparison(ungated_result, gated_result)
    
    return ungated_result, gated_result


def run_multiple_seeds(n_seeds: int = 5, n_bars: int = 2000) -> None:
    """
    Run demo with multiple seeds to assess robustness.
    """
    print()
    print("=" * 78)
    print("              MULTI-SEED ROBUSTNESS ANALYSIS")
    print("=" * 78)
    print()
    
    results = []
    
    for seed in range(42, 42 + n_seeds):
        print(f"Processing seed {seed}...", end=" ")
        
        obs_1 = generate_observation_stream(n_bars=n_bars, seed=seed)
        obs_2 = generate_observation_stream(n_bars=n_bars, seed=seed)
        
        ungated = run_ungated(obs_1)
        gated = run_gated(obs_2)
        
        results.append({
            'seed': seed,
            'u_crr': ungated.stats.crr,
            'g_crr': gated.stats.crr,
            'u_inv': ungated.stats.invalidation_rate,
            'g_inv': gated.stats.invalidation_rate,
            'blocked': gated.total_blocked,
            'u_testable': ungated.stats.testable_count,
            'g_testable': gated.stats.testable_count,
        })
        
        print(f"CRR: {ungated.stats.crr:.1%} → {gated.stats.crr:.1%}, "
              f"Inv: {ungated.stats.invalidation_rate:.1%} → {gated.stats.invalidation_rate:.1%}")
    
    print()
    print("=" * 78)
    print("SUMMARY ACROSS SEEDS:")
    print("-" * 78)
    print(f"{'Seed':<6} {'CRR(U)':<10} {'CRR(G)':<10} {'Inv(U)':<10} {'Inv(G)':<10} {'Blocked':<10}")
    print("-" * 78)
    
    for r in results:
        print(f"{r['seed']:<6} {r['u_crr']:.1%}     {r['g_crr']:.1%}     "
              f"{r['u_inv']:.1%}     {r['g_inv']:.1%}     {r['blocked']:<10}")
    
    # Compute averages
    avg_u_crr = sum(r['u_crr'] for r in results) / len(results)
    avg_g_crr = sum(r['g_crr'] for r in results) / len(results)
    avg_u_inv = sum(r['u_inv'] for r in results) / len(results)
    avg_g_inv = sum(r['g_inv'] for r in results) / len(results)
    avg_blocked = sum(r['blocked'] for r in results) / len(results)
    
    print("-" * 78)
    print(f"{'AVG':<6} {avg_u_crr:.1%}     {avg_g_crr:.1%}     "
          f"{avg_u_inv:.1%}     {avg_g_inv:.1%}     {avg_blocked:.0f}")
    
    print()
    print("AGGREGATE ANALYSIS:")
    crr_change = avg_g_crr - avg_u_crr
    inv_reduction = avg_u_inv - avg_g_inv
    
    print(f"  Average CRR Change:             {crr_change:+.1%}")
    print(f"  Average Invalidation Reduction: {inv_reduction:+.1%}")
    print()
    
    # Final verdict
    if inv_reduction > 0.05 and abs(crr_change) < 0.10:
        print("  VERDICT: P4 gating CONSISTENTLY effective across seeds")
    elif inv_reduction > 0:
        print("  VERDICT: P4 gating shows MODERATE improvement")
    else:
        print("  VERDICT: P4 gating effect INCONSISTENT")
    
    print("=" * 78)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CRV Engine Demo with P4 Gating")
    parser.add_argument("--bars", type=int, default=2000, help="Number of bars")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--multi", action="store_true", help="Run with multiple seeds")
    parser.add_argument("--quiet", action="store_true", help="Less output")
    
    args = parser.parse_args()
    
    if args.multi:
        run_multiple_seeds(n_seeds=5, n_bars=args.bars)
    else:
        run_demo(n_bars=args.bars, seed=args.seed, verbose=not args.quiet)
