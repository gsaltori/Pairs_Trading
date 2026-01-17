#!/usr/bin/env python3
"""
Example Experiment: Baseline Edge Evaluation

This is a REFERENCE IMPLEMENTATION demonstrating proper use of
the CRV Experiment Harness.

EXPERIMENT DESIGN:
- Configuration: All baseline parameters
- Hypothesis: Edge exists (CRR > 55%, invalidation < 40%)
- Data: Synthetic (seed=42, n_bars=2000)

NO STRATEGY MODIFICATIONS. EVALUATION ONLY.
"""

import sys
from pathlib import Path

# Ensure imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments import (
    create_baseline_experiment,
    create_seed_experiment,
    create_edge_exists_hypothesis,
    create_confirmation_speed_hypothesis,
    ExperimentRunner,
    ExperimentBatch,
    ExperimentDimension,
)


def run_baseline_experiment():
    """
    Run a single baseline experiment with edge hypothesis.
    
    This is the simplest possible experiment.
    """
    print("=" * 70)
    print("EXAMPLE EXPERIMENT: Baseline Edge Evaluation")
    print("=" * 70)
    print()
    
    # ─────────────────────────────────────────────────────────────────────────
    # 1. CREATE EXPERIMENT CONFIGURATION
    # ─────────────────────────────────────────────────────────────────────────
    config = create_baseline_experiment(
        experiment_id="baseline_001",
        n_bars=2000,
    )
    
    print("Configuration:")
    print(f"  Experiment ID:      {config.experiment_id}")
    print(f"  Config Hash:        {config.config_hash}")
    print(f"  Pair:               {config.pair}")
    print(f"  Timeframe:          {config.timeframe.value}")
    print(f"  Z-Score Window:     {config.zscore_window}")
    print(f"  Correlation Window: {config.correlation_window}")
    print(f"  Data:               {'Synthetic' if config.use_synthetic else 'Live'}")
    print(f"  Seed:               {config.random_seed}")
    print()
    
    # ─────────────────────────────────────────────────────────────────────────
    # 2. DEFINE HYPOTHESES
    # ─────────────────────────────────────────────────────────────────────────
    hypotheses = [
        create_edge_exists_hypothesis(
            hypothesis_id="h_edge_baseline",
            min_crr=0.55,
            min_testable=30,
            max_invalidation_rate=0.40,
        ),
        create_confirmation_speed_hypothesis(
            hypothesis_id="h_speed_baseline",
            max_median_bars_to_confirm=25,
            min_confirmed=15,
        ),
    ]
    
    print("Hypotheses:")
    for h in hypotheses:
        print(f"  {h.hypothesis_id}: {h.name}")
        print(f"    Null: {h.null_hypothesis}")
        print(f"    Alt:  {h.alternative_hypothesis}")
    print()
    
    # ─────────────────────────────────────────────────────────────────────────
    # 3. RUN EXPERIMENT
    # ─────────────────────────────────────────────────────────────────────────
    output_dir = Path(__file__).parent.parent / "experiments_output"
    
    runner = ExperimentRunner(output_dir=output_dir, verbose=True)
    output = runner.run(config, hypotheses)
    
    # ─────────────────────────────────────────────────────────────────────────
    # 4. DISPLAY RESULTS
    # ─────────────────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    
    m = output.metrics
    
    print("Primary Metrics:")
    print(f"  Total Predictions:  {m['total_predictions']}")
    print(f"  Testable Count:     {m['testable_count']}")
    print(f"  CRR:                {m['crr']:.1%}")
    print(f"  Invalidation Rate:  {m['invalidation_rate']:.1%}")
    print(f"  Timeout Rate:       {m['timeout_rate']:.1%}")
    print()
    
    print("Extended Metrics:")
    print(f"  ERV Total:          {m['erv']:.2f}")
    print(f"  ERV/Prediction:     {m['erv_per_prediction']:.3f}")
    if m['median_bars_to_confirmation']:
        print(f"  Median Bars→Confirm:{m['median_bars_to_confirmation']:.1f}")
    if m['median_bars_to_refutation']:
        print(f"  Median Bars→Refute: {m['median_bars_to_refutation']:.1f}")
    if m['timing_asymmetry_ratio']:
        print(f"  Timing Asymmetry:   {m['timing_asymmetry_ratio']:.2f}")
    print()
    
    print("Hypothesis Verdicts:")
    for h_result in output.hypothesis_results:
        print(f"  {h_result['hypothesis_id']}: {h_result['verdict']}")
        print(f"    {h_result['summary']}")
    print()
    
    print("=" * 70)
    print(f"Output saved to: {output_dir}")
    print("=" * 70)
    
    return output


def run_seed_robustness_batch():
    """
    Run a batch experiment varying only the random seed.
    
    PURPOSE: Verify that results are stable across different
    synthetic data realizations.
    """
    print()
    print("=" * 70)
    print("BATCH EXPERIMENT: Seed Robustness Analysis")
    print("=" * 70)
    print()
    
    # Create experiments with different seeds
    seeds = [42, 43, 44, 45, 46]
    experiments = tuple(
        create_seed_experiment(
            experiment_id=f"seed_{seed}",
            random_seed=seed,
            n_bars=2000,
        )
        for seed in seeds
    )
    
    batch = ExperimentBatch(
        batch_id="seed_robustness",
        dimension=ExperimentDimension.RANDOM_SEED,
        experiments=experiments,
        description="Evaluate CRR stability across random seeds",
    )
    
    # Define hypothesis
    hypothesis = create_edge_exists_hypothesis(
        hypothesis_id="h_edge_seed",
        min_crr=0.52,  # Slightly lower threshold for robustness
        min_testable=25,
        max_invalidation_rate=0.45,
    )
    
    # Run batch
    output_dir = Path(__file__).parent.parent / "experiments_output"
    runner = ExperimentRunner(output_dir=output_dir, verbose=False)
    outputs = runner.run_batch(batch, [hypothesis])
    
    # Summarize
    print()
    print("BATCH SUMMARY:")
    print("-" * 70)
    print(f"{'Seed':<10} {'CRR':>10} {'ERV/pred':>12} {'Verdict':>15}")
    print("-" * 70)
    
    crrs = []
    for output in outputs:
        seed = output.config['random_seed']
        crr = output.metrics['crr']
        erv = output.metrics['erv_per_prediction']
        verdict = output.hypothesis_results[0]['verdict'] if output.hypothesis_results else 'N/A'
        
        crrs.append(crr)
        print(f"{seed:<10} {crr:>9.1%} {erv:>12.3f} {verdict:>15}")
    
    print("-" * 70)
    
    from statistics import mean, stdev
    print(f"{'Mean':>10} {mean(crrs):>9.1%}")
    print(f"{'StdDev':>10} {stdev(crrs):>9.1%}")
    print()
    
    # Verdict summary
    supported = sum(1 for o in outputs if o.primary_verdict == 'SUPPORTED')
    refuted = sum(1 for o in outputs if o.primary_verdict == 'REFUTED')
    insufficient = sum(1 for o in outputs if o.primary_verdict == 'INSUFFICIENT_DATA')
    
    print(f"Hypotheses SUPPORTED: {supported}/{len(outputs)}")
    print(f"Hypotheses REFUTED:   {refuted}/{len(outputs)}")
    print(f"INSUFFICIENT_DATA:    {insufficient}/{len(outputs)}")
    print()
    
    return outputs


if __name__ == "__main__":
    # Run baseline experiment
    baseline_output = run_baseline_experiment()
    
    # Run seed robustness batch
    batch_outputs = run_seed_robustness_batch()
    
    print()
    print("=" * 70)
    print("EXPERIMENT HARNESS DEMONSTRATION COMPLETE")
    print("=" * 70)
