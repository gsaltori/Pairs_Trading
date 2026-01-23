#!/usr/bin/env python3
"""
Canonical Real-Data Experiment

LOCKED PARAMETERS - DO NOT MODIFY:
- EURUSD / GBPUSD
- H4
- 5,000 bars
- Block size: 500
- CRR threshold: 0.55
- Max invalidation: 0.40

This is a ONE-SHOT epistemic execution.
No retries for "better" results.
No parameter adjustments.

Produces:
1. walkforward_canonical_*.json
2. edge_boundary_canonical_*.json

These outputs enable Edge Worthiness judgment.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from crv_engine.mt5_loader import (
    initialize_mt5,
    shutdown_mt5,
    load_pair_observations,
    validate_data,
    print_data_summary,
)
from crv_engine.experiments import (
    create_baseline_experiment,
    WalkForwardConfig,
    WalkForwardRunner,
    analyze_edge_boundaries,
)


# ==============================================================================
# LOCKED PARAMETERS (DO NOT EDIT)
# ==============================================================================
SYMBOL_A = "EURUSD"
SYMBOL_B = "GBPUSD"
TIMEFRAME = "H4"
N_BARS = 5000
BLOCK_SIZE = 500
WARMUP_BARS = 60
CRR_THRESHOLD = 0.55
MAX_INVALIDATION = 0.40
MIN_TESTABLE_PER_BLOCK = 10
MIN_BLOCKS_FOR_STABILITY = 3


def main():
    """Execute the canonical experiment."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "crv_engine" / "experiment_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("CANONICAL REAL-DATA EXPERIMENT")
    print("=" * 70)
    print(f"Timestamp: {timestamp}")
    print(f"Symbols: {SYMBOL_A} / {SYMBOL_B}")
    print(f"Timeframe: {TIMEFRAME}")
    print(f"Bars: {N_BARS}")
    print(f"Block size: {BLOCK_SIZE}")
    print(f"CRR threshold: {CRR_THRESHOLD:.0%}")
    print(f"Max invalidation: {MAX_INVALIDATION:.0%}")
    print()
    
    # =========================================================================
    # STEP 1: Initialize MT5
    # =========================================================================
    print("=" * 70)
    print("STEP 1: Initialize MT5 Connection")
    print("=" * 70)
    
    if not initialize_mt5():
        print("❌ ABORT: MT5 initialization failed")
        return 1
    
    print("✅ MT5 initialized")
    print()
    
    try:
        # =====================================================================
        # STEP 2: Load Data
        # =====================================================================
        print("=" * 70)
        print("STEP 2: Load Market Data")
        print("=" * 70)
        
        observations = load_pair_observations(
            symbol_a=SYMBOL_A,
            symbol_b=SYMBOL_B,
            timeframe=TIMEFRAME,
            n_bars=N_BARS,
        )
        
        if observations is None:
            print("❌ ABORT: Data loading failed")
            return 1
        
        print(f"✅ Loaded {len(observations)} observations")
        print(f"   First: {observations[0].timestamp}")
        print(f"   Last:  {observations[-1].timestamp}")
        
        # Print data summary
        print_data_summary(observations, TIMEFRAME)
        print()
        
        # =====================================================================
        # STEP 3: Validate Data
        # =====================================================================
        print("=" * 70)
        print("STEP 3: Validate Data")
        print("=" * 70)
        
        is_valid, issues = validate_data(observations, N_BARS, TIMEFRAME)
        
        if not is_valid:
            print("❌ ABORT: Data validation failed")
            for issue in issues:
                print(f"   - {issue}")
            return 1
        
        print("✅ Data validation passed")
        print()
        
        # =====================================================================
        # STEP 4: Create Experiment Configuration
        # =====================================================================
        print("=" * 70)
        print("STEP 4: Configure Experiment")
        print("=" * 70)
        
        exp_config = create_baseline_experiment(
            experiment_id=f"canonical_{timestamp}",
            n_bars=N_BARS,
        )
        
        wf_config = WalkForwardConfig(
            block_size=BLOCK_SIZE,
            min_bars_per_block=300,
            warmup_bars=WARMUP_BARS,
            min_testable_per_block=MIN_TESTABLE_PER_BLOCK,
            min_blocks_for_stability=MIN_BLOCKS_FOR_STABILITY,
        )
        
        print(f"   Experiment ID: {exp_config.experiment_id}")
        print(f"   Config hash: {exp_config.config_hash}")
        print(f"   Expected blocks: {len(wf_config.compute_blocks(N_BARS))}")
        print()
        
        # =====================================================================
        # STEP 5: Run Walk-Forward Analysis
        # =====================================================================
        print("=" * 70)
        print("STEP 5: Walk-Forward Falsification")
        print("=" * 70)
        
        runner = WalkForwardRunner(
            walk_forward_config=wf_config,
            experiment_config=exp_config,
            crr_threshold=CRR_THRESHOLD,
            max_invalidation_threshold=MAX_INVALIDATION,
            output_dir=None,  # We'll save manually
            verbose=True,
        )
        
        wf_result = runner.run(observations)
        
        # Save walk-forward output
        wf_filename = f"walkforward_canonical_{timestamp}.json"
        wf_filepath = output_dir / wf_filename
        
        with open(wf_filepath, 'w') as f:
            f.write(wf_result.output.to_json(indent=2))
        
        print()
        print(f"✅ Walk-forward saved: {wf_filepath}")
        print()
        
        # =====================================================================
        # STEP 6: Run Edge Boundary Analysis
        # =====================================================================
        print("=" * 70)
        print("STEP 6: Edge Boundary Analysis")
        print("=" * 70)
        
        if len(wf_result.all_predictions) == 0:
            print("❌ ABORT: No predictions generated")
            return 1
        
        print(f"   Total predictions: {len(wf_result.all_predictions)}")
        
        eb_output = analyze_edge_boundaries(
            predictions=wf_result.all_predictions,
            walk_forward_output=wf_result.output,
            crr_threshold=CRR_THRESHOLD,
            verbose=True,
        )
        
        # Save edge boundary output
        eb_filename = f"edge_boundary_canonical_{timestamp}.json"
        eb_filepath = output_dir / eb_filename
        
        with open(eb_filepath, 'w') as f:
            f.write(eb_output.to_json(indent=2))
        
        print()
        print(f"✅ Edge boundary saved: {eb_filepath}")
        print()
        
        # =====================================================================
        # STEP 7: Summary
        # =====================================================================
        print("=" * 70)
        print("CANONICAL EXPERIMENT COMPLETE")
        print("=" * 70)
        print()
        print("OUTPUT FILES:")
        print(f"   1. {wf_filepath}")
        print(f"   2. {eb_filepath}")
        print()
        print("KEY RESULTS:")
        print(f"   Stability Class: {wf_result.output.stability_metrics['stability_class']}")
        print(f"   Verdict Persistence: {wf_result.output.stability_metrics['verdict_persistence_ratio']:.1%}")
        print(f"   Total Blocks: {wf_result.output.stability_metrics['total_blocks']}")
        print(f"   Sufficient Blocks: {wf_result.output.stability_metrics['sufficient_blocks']}")
        
        if wf_result.output.stability_metrics['crr_drift_slope'] is not None:
            print(f"   CRR Drift Slope: {wf_result.output.stability_metrics['crr_drift_slope']:.4f}")
        
        print()
        print("WALK-FORWARD FLAGS:")
        for flag, value in wf_result.output.flags.items():
            indicator = "✓" if value else "✗"
            print(f"   [{indicator}] {flag}")
        
        print()
        print("EDGE BOUNDARY FLAGS:")
        for flag, value in eb_output.flags.items():
            indicator = "✓" if value else "✗"
            print(f"   [{indicator}] {flag}")
        
        print()
        print("=" * 70)
        print("JUDGMENT CAN NOW PROCEED")
        print("=" * 70)
        
        return 0
        
    finally:
        shutdown_mt5()
        print("\nMT5 connection closed")


if __name__ == "__main__":
    sys.exit(main())
