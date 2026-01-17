#!/usr/bin/env python3
"""
Example Experiment Script

Demonstrates how to use the experiment harness to run
reproducible, falsifiable experiments on the CRV engine.

THIS SCRIPT DOES NOT MODIFY THE ENGINE.
It only executes experiments and evaluates hypotheses.

USAGE:
    python run_example_experiment.py
    python run_example_experiment.py --batch
    python run_example_experiment.py --seed-variation
"""

import os
import sys
import argparse
from datetime import datetime

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments import (
    run_experiment,
    run_batch,
    create_seed_variation_experiments,
    create_zscore_window_experiments,
    BASELINE_CONFIG,
    STANDARD_HYPOTHESES,
    HypothesisVerdict,
    aggregate_experiments,
)


def run_single_baseline():
    """Run a single baseline experiment."""
    print("=" * 80)
    print("SINGLE BASELINE EXPERIMENT")
    print("=" * 80)
    print()
    
    print(f"Configuration:")
    print(f"  Experiment ID: {BASELINE_CONFIG.experiment_id}")
    print(f"  Dimension Varied: {BASELINE_CONFIG.dimension_varied.value}")
    print(f"  Pair: {BASELINE_CONFIG.engine_params.pair}")
    print(f"  Timeframe: {BASELINE_CONFIG.engine_params.timeframe.value}")
    print(f"  Bars: {BASELINE_CONFIG.data_params.n_bars}")
    print(f"  Seed: {BASELINE_CONFIG.data_params.random_seed}")
    print(f"  Structural Gating: {BASELINE_CONFIG.structural_gating_enabled}")
    print()
    
    print("Running experiment...")
    result = run_experiment(
        BASELINE_CONFIG,
        list(STANDARD_HYPOTHESES.values()),
        output_dir="experiment_results",
    )
    
    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    
    # Metrics
    m = result.metrics
    print("METRICS:")
    print(f"  Predictions Generated:  {m['n_predictions']}")
    print(f"  Resolved:               {m['n_resolved']}")
    print(f"  Testable:               {m['n_testable']}")
    print()
    print(f"  CRR:                    {m['crr']:.1%} (95% CI: [{m['crr_95_ci_lower']:.1%}, {m['crr_95_ci_upper']:.1%}])")
    print(f"  ERV:                    {m['erv']:.2f}")
    print(f"  ERV per testable:       {m['erv_per_testable']:.3f}")
    print()
    print(f"  Invalidation Rate:      {m['invalidation_rate']:.1%}")
    print(f"  Timeout Rate:           {m['timeout_rate']:.1%}")
    print()
    print(f"  Median Bars (Confirm):  {m['median_bars_confirmed']:.1f}")
    print(f"  Median Bars (Refute):   {m['median_bars_refuted']:.1f}")
    print(f"  Bars Ratio (C/R):       {m['bars_ratio_conf_to_ref']:.2f}")
    print()
    print("  Conditional CRR by Stability Duration:")
    print(f"    Short (<20 bars):     {m['crr_short_stability']:.1%} (N={m['n_short_stability']})")
    print(f"    Medium (20-50 bars):  {m['crr_medium_stability']:.1%} (N={m['n_medium_stability']})")
    print(f"    Long (>50 bars):      {m['crr_long_stability']:.1%} (N={m['n_long_stability']})")
    print()
    
    # Hypothesis verdicts
    print("=" * 80)
    print("HYPOTHESIS VERDICTS")
    print("=" * 80)
    print()
    
    for h in result.hypotheses_evaluated:
        verdict_symbol = {
            "SUPPORTED": "✓",
            "REFUTED": "✗",
            "INSUFFICIENT_DATA": "?",
        }.get(h["verdict"], "?")
        
        print(f"  [{verdict_symbol}] {h['hypothesis_id']}: {h['verdict']}")
        print(f"      Reason: {h['reason']}")
        print()
    
    # Output file
    print("=" * 80)
    print(f"Config Hash: {result.config_hash}")
    print(f"Output saved to: experiment_results/")
    print("=" * 80)
    
    return result


def run_seed_variation_batch():
    """Run batch of experiments varying only the random seed."""
    print("=" * 80)
    print("SEED VARIATION BATCH")
    print("=" * 80)
    print()
    print("This tests robustness across different synthetic data realizations.")
    print("ALL parameters are fixed except the random seed.")
    print()
    
    experiments = create_seed_variation_experiments(
        BASELINE_CONFIG,
        seeds=[42, 43, 44, 45, 46],
    )
    
    print(f"Running {len(experiments)} experiments...")
    print()
    
    results = run_batch(
        experiments,
        list(STANDARD_HYPOTHESES.values()),
        output_dir="experiment_results",
    )
    
    print()
    print("=" * 80)
    print("AGGREGATED RESULTS")
    print("=" * 80)
    print()
    
    agg = aggregate_experiments(results)
    print(f"Experiments: {agg['n_experiments']}")
    print()
    print(f"CRR:              Mean={agg['crr']['mean']:.1%}, Std={agg['crr']['std']:.1%}, Range=[{agg['crr']['min']:.1%}, {agg['crr']['max']:.1%}]")
    print(f"ERV:              Mean={agg['erv']['mean']:.2f}, Std={agg['erv']['std']:.2f}")
    print(f"Invalidation:     Mean={agg['invalidation_rate']['mean']:.1%}, Std={agg['invalidation_rate']['std']:.1%}")
    print()
    
    # Hypothesis consistency
    print("HYPOTHESIS CONSISTENCY:")
    for h_id in STANDARD_HYPOTHESES.keys():
        verdicts = []
        for r in results:
            for h in r.hypotheses_evaluated:
                if h["hypothesis_id"] == h_id:
                    verdicts.append(h["verdict"])
        
        supported = verdicts.count("SUPPORTED")
        refuted = verdicts.count("REFUTED")
        insufficient = verdicts.count("INSUFFICIENT_DATA")
        
        print(f"  {h_id}:")
        print(f"    SUPPORTED: {supported}/{len(verdicts)}, REFUTED: {refuted}/{len(verdicts)}, INSUFFICIENT: {insufficient}/{len(verdicts)}")
    
    return results


def run_zscore_window_batch():
    """Run batch of experiments varying only the Z-score window."""
    print("=" * 80)
    print("Z-SCORE WINDOW VARIATION BATCH")
    print("=" * 80)
    print()
    print("This tests sensitivity to the Z-score lookback window.")
    print("ALL parameters are fixed except zscore_window.")
    print()
    
    experiments = create_zscore_window_experiments(
        BASELINE_CONFIG,
        windows=[30, 45, 60, 90, 120],
    )
    
    print(f"Running {len(experiments)} experiments...")
    print()
    
    results = run_batch(
        experiments,
        list(STANDARD_HYPOTHESES.values()),
        output_dir="experiment_results",
    )
    
    print()
    print("=" * 80)
    print("WINDOW SENSITIVITY ANALYSIS")
    print("=" * 80)
    print()
    
    print(f"{'Window':<10} {'CRR':>10} {'ERV':>10} {'Inv Rate':>10} {'H1':>10}")
    print("-" * 50)
    
    for r in results:
        window = r.configuration["engine_params"]["zscore_window"]
        crr = r.metrics["crr"]
        erv = r.metrics["erv"]
        inv = r.metrics["invalidation_rate"]
        h1 = next(h for h in r.hypotheses_evaluated if h["hypothesis_id"] == "H1_CRR_ABOVE_RANDOM")
        
        print(f"{window:<10} {crr:>9.1%} {erv:>10.2f} {inv:>9.1%} {h1['verdict']:>10}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="CRV Engine Experiment Harness Example"
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Run seed variation batch"
    )
    parser.add_argument(
        "--zscore", action="store_true",
        help="Run Z-score window variation batch"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs("experiment_results", exist_ok=True)
    
    if args.batch:
        run_seed_variation_batch()
    elif args.zscore:
        run_zscore_window_batch()
    else:
        run_single_baseline()


if __name__ == "__main__":
    main()
