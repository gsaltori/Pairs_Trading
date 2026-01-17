#!/usr/bin/env python3
"""
Example Walk-Forward Analysis

Demonstrates the complete walk-forward falsification workflow using synthetic data.

This script:
1. Generates synthetic observation stream
2. Configures walk-forward analysis
3. Runs the engine across sequential blocks
4. Analyzes temporal stability
5. Produces structured output

NO STRATEGY MODIFICATION. PURE EVALUATION.
"""

import sys
from pathlib import Path

# Ensure imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.config import create_baseline_experiment
from experiments.walk_forward import WalkForwardConfig
from experiments.walk_forward_runner import WalkForwardRunner, run_walk_forward


def main():
    """Run example walk-forward analysis."""
    print("=" * 70)
    print("WALK-FORWARD FALSIFICATION ENGINE")
    print("=" * 70)
    print()
    
    # ─────────────────────────────────────────────────────────────────────────
    # 1. GENERATE SYNTHETIC DATA
    # ─────────────────────────────────────────────────────────────────────────
    print("1. Generating synthetic data...")
    
    try:
        from observations import generate_observation_stream
    except ImportError:
        from ..observations import generate_observation_stream
    
    # Generate 3000 bars (enough for ~6 blocks of 500)
    n_bars = 3000
    seed = 42
    
    observation_stream = list(generate_observation_stream(n_bars=n_bars, seed=seed))
    
    print(f"   Generated {len(observation_stream)} observations")
    print()
    
    # ─────────────────────────────────────────────────────────────────────────
    # 2. CONFIGURE EXPERIMENT
    # ─────────────────────────────────────────────────────────────────────────
    print("2. Configuring experiment...")
    
    exp_config = create_baseline_experiment(
        experiment_id="wf_demo_001",
        n_bars=n_bars,
    )
    
    wf_config = WalkForwardConfig(
        block_size=500,
        min_bars_per_block=300,
        warmup_bars=60,
        min_testable_per_block=10,
        min_blocks_for_stability=3,
    )
    
    # Hypothesis thresholds
    crr_threshold = 0.55
    max_invalidation_threshold = 0.40
    
    print(f"   Experiment ID: {exp_config.experiment_id}")
    print(f"   Block size: {wf_config.block_size} bars")
    print(f"   Expected blocks: {len(wf_config.compute_blocks(n_bars))}")
    print(f"   CRR threshold: {crr_threshold:.0%}")
    print(f"   Max invalidation: {max_invalidation_threshold:.0%}")
    print()
    
    # ─────────────────────────────────────────────────────────────────────────
    # 3. RUN WALK-FORWARD ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────
    print("3. Running walk-forward analysis...")
    print("-" * 70)
    
    output_dir = Path(__file__).parent.parent / "experiments_output"
    
    runner = WalkForwardRunner(
        walk_forward_config=wf_config,
        experiment_config=exp_config,
        crr_threshold=crr_threshold,
        max_invalidation_threshold=max_invalidation_threshold,
        output_dir=output_dir,
        verbose=True,
    )
    
    output = runner.run(observation_stream)
    
    print("-" * 70)
    print()
    
    # ─────────────────────────────────────────────────────────────────────────
    # 4. DISPLAY RESULTS
    # ─────────────────────────────────────────────────────────────────────────
    print("4. Results Summary")
    print("=" * 70)
    print()
    
    # Block summary table
    print("BLOCK RESULTS:")
    print("-" * 70)
    print(f"{'Block':<8} {'Bars':<12} {'Testable':<10} {'CRR':<10} {'ERV/pred':<10} {'Verdict':<12}")
    print("-" * 70)
    
    for block in output.block_results:
        print(
            f"{block['block_index']:<8} "
            f"[{block['start_bar']}:{block['end_bar']}]".ljust(12) + " "
            f"{block['testable_count']:<10} "
            f"{block['crr']:.1%}".ljust(10) + " "
            f"{block['erv_per_prediction']:.3f}".ljust(10) + " "
            f"{block['verdict']:<12}"
        )
    
    print("-" * 70)
    print()
    
    # Stability metrics
    sm = output.stability_metrics
    print("TEMPORAL STABILITY ANALYSIS:")
    print("-" * 70)
    print(f"  Total blocks:           {sm['total_blocks']}")
    print(f"  Sufficient blocks:      {sm['sufficient_blocks']}")
    print(f"  Verdict persistence:    {sm['verdict_persistence_ratio']:.1%}")
    print(f"  Supported count:        {sm['supported_count']}")
    print(f"  Refuted count:          {sm['refuted_count']}")
    print()
    
    if sm['crr_drift_slope'] is not None:
        print(f"  CRR drift slope:        {sm['crr_drift_slope']:.4f}")
        print(f"  CRR drift R²:           {sm['crr_drift_r_squared']:.3f}")
        print(f"  Is monotonic decay:     {sm['is_monotonic_decay']}")
        print(f"  Decay consistency:      {sm['decay_consistency']:.2f}")
    print()
    
    print(f"  STABILITY CLASS:        {sm['stability_class']}")
    print(f"  Confidence:             {sm['classification_confidence']:.1%}")
    print(f"  Reason:                 {sm['classification_reason']}")
    print()
    
    # Flags
    print("FLAGS:")
    print("-" * 70)
    for flag_name, flag_value in output.flags.items():
        indicator = "✓" if flag_value else "✗"
        print(f"  [{indicator}] {flag_name}")
    print()
    
    # ─────────────────────────────────────────────────────────────────────────
    # 5. INTERPRETATION
    # ─────────────────────────────────────────────────────────────────────────
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print()
    
    stability_class = sm['stability_class']
    
    if stability_class == "PERSISTENT":
        print("CONCLUSION: Hypothesis shows PERSISTENT edge.")
        print("  - CRR remains above threshold across time blocks")
        print("  - No significant decay trend detected")
        print("  - Verdict consistency is high")
        print()
        print("CAUTION: This does NOT prove profitable trading.")
        print("         It only shows temporal stability of the measured metric.")
    
    elif stability_class == "DECAYING":
        print("CONCLUSION: Hypothesis shows DECAYING edge.")
        print("  - CRR degrades over time")
        print("  - Edge may have been real but is deteriorating")
        print("  - Possible causes: regime shift, overfitting, market adaptation")
        print()
        print("RECOMMENDATION: Do not assume forward validity.")
    
    elif stability_class == "REGIME_DEPENDENT":
        print("CONCLUSION: Hypothesis is REGIME-DEPENDENT.")
        print("  - Edge varies significantly across blocks")
        print("  - No clear decay trend, but also no consistent support")
        print("  - Edge may only exist under specific market conditions")
        print()
        print("RECOMMENDATION: Investigate regime characteristics of supporting blocks.")
    
    elif stability_class == "UNSTABLE":
        print("CONCLUSION: Hypothesis is UNSTABLE.")
        print("  - High variance in CRR across blocks")
        print("  - No consistent pattern detected")
        print("  - Cannot reliably classify temporal behavior")
        print()
        print("RECOMMENDATION: Consider if hypothesis is fundamentally flawed.")
    
    elif stability_class == "UNFALSIFIABLE":
        print("CONCLUSION: Hypothesis is UNFALSIFIABLE with current data.")
        print("  - Insufficient blocks or testable predictions")
        print("  - Cannot draw temporal stability conclusions")
        print()
        print("RECOMMENDATION: Acquire more data before making claims.")
    
    print()
    print("=" * 70)
    print(f"Output saved to: {output_dir}")
    print("=" * 70)
    
    return output


if __name__ == "__main__":
    main()
