"""
Run Exit Model Comparison Backtest

Compares different exit strategies on the session-based strategy:
- BASELINE: 2.5R single target (current - 94% time stops)
- MULTI_TARGET: 50% at 1.0R, 30% at 2.0R, 20% runner with trailing
- REDUCED_TP: 1.5R single target
- AGGRESSIVE: 1.0R single target

GOAL: Improve expectancy without changing entry logic.
"""

import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from crv_engine.mt5_loader import initialize_mt5, shutdown_mt5, load_ohlc
from trading_system.exit_models import ExitModel
from trading_system.exit_comparison_backtest import (
    ExitComparisonBacktest,
    print_exit_comparison,
    print_detailed_exit_analysis,
    determine_best_exit,
    print_recommendation,
)


INITIAL_CAPITAL = 100.0


def load_data_h1(n_bars: int = 8000):
    """Load H1 data for session-level resolution."""
    print("=" * 95)
    print("LOADING MT5 DATA (H1)")
    print("=" * 95)
    
    if not initialize_mt5():
        raise RuntimeError("MT5 init failed")
    
    try:
        print(f"Loading EURUSD H1 ({n_bars} bars)...")
        eu_bars = load_ohlc("EURUSD", "H1", n_bars)
        
        print(f"Loading GBPUSD H1 ({n_bars} bars)...")
        gb_bars = load_ohlc("GBPUSD", "H1", n_bars)
        
        if eu_bars is None or gb_bars is None:
            raise RuntimeError("Data load failed")
        
        df_eu = pd.DataFrame([{
            'timestamp': b.timestamp,
            'open': b.open,
            'high': b.high,
            'low': b.low,
            'close': b.close,
        } for b in eu_bars])
        
        df_gb = pd.DataFrame([{
            'timestamp': b.timestamp,
            'open': b.open,
            'high': b.high,
            'low': b.low,
            'close': b.close,
        } for b in gb_bars])
        
        # Align
        df_eu = df_eu.set_index('timestamp')
        df_gb = df_gb.set_index('timestamp')
        
        common = df_eu.index.intersection(df_gb.index)
        df_eu = df_eu.loc[common].reset_index()
        df_gb = df_gb.loc[common].reset_index()
        
        print(f"✅ Loaded {len(df_eu)} aligned bars")
        print(f"   Range: {df_eu['timestamp'].iloc[0]} to {df_eu['timestamp'].iloc[-1]}")
        
        start = df_eu['timestamp'].iloc[0]
        end = df_eu['timestamp'].iloc[-1]
        days = (end - start).days
        print(f"   Span: {days} days (~{days/365:.1f} years)")
        
        return df_eu, df_gb
        
    finally:
        shutdown_mt5()


def main():
    print()
    print("╔" + "═" * 93 + "╗")
    print("║" + " EXIT MODEL COMPARISON BACKTEST ".center(93) + "║")
    print("║" + " Session Strategy: Asia Range → London Expansion ".center(93) + "║")
    print("║" + " Focus: Improve expectancy via exit logic only ".center(93) + "║")
    print("╚" + "═" * 93 + "╝")
    print()
    
    # Current problem summary
    print("CURRENT ISSUE:")
    print("-" * 50)
    print("  - 94% of trades exit via TIME stop")
    print("  - 0% TP hits at R=2.5")
    print("  - Expectancy is barely positive (+0.03R)")
    print("  - Target is clearly too aggressive")
    print()
    
    # Exit models being tested
    print("EXIT MODELS TO TEST:")
    print("-" * 50)
    print("  1. BASELINE:     2.5R single target (current)")
    print("  2. MULTI_TARGET: 50% at 1.0R, 30% at 2.0R, 20% runner")
    print("                   + Breakeven stop after TP1")
    print("                   + Trailing stop for runner")
    print("  3. REDUCED_TP:   1.5R single target")
    print("  4. AGGRESSIVE:   1.0R single target")
    print()
    
    # Load data
    df_eu, df_gb = load_data_h1(n_bars=8000)
    print()
    
    # Run comparison
    print("Running EXIT COMPARISON...")
    backtest = ExitComparisonBacktest(
        initial_capital=INITIAL_CAPITAL,
        risk_per_trade=0.005,
    )
    results = backtest.run(df_eu, df_gb)
    print()
    
    # Print comparison
    print_exit_comparison(results)
    print_detailed_exit_analysis(results)
    
    # Determine best
    baseline_dd = results[ExitModel.BASELINE].max_drawdown_pct
    best_model, verdict, explanation = determine_best_exit(results, baseline_dd)
    
    # Recommendation
    print_recommendation(best_model, verdict, explanation, results)
    
    # Summary
    print()
    print("╔" + "═" * 93 + "╗")
    print("║" + " SUMMARY ".center(93) + "║")
    print("╚" + "═" * 93 + "╝")
    print()
    
    baseline = results[ExitModel.BASELINE]
    best = results[best_model]
    
    print(f"  Entry Logic: Unchanged (Asia → London breakout)")
    print(f"  Gatekeeper:  Unchanged (structural filter)")
    print()
    print(f"  Baseline Exit:   2.5R target, {baseline.expectancy_r:.2f}R expectancy, {baseline.pct_tp_hits:.1%} TP hits")
    print(f"  Best Exit:       {best_model.value}, {best.expectancy_r:.2f}R expectancy, {best.pct_tp_hits:.1%} TP hits")
    print()
    
    if best.expectancy_r > baseline.expectancy_r:
        improvement = (best.expectancy_r - baseline.expectancy_r) / abs(baseline.expectancy_r) * 100
        print(f"  Expectancy Improvement: +{improvement:.0f}%")
    
    # Kill check
    if best.expectancy_r <= 0:
        print()
        print("  ⛔ STRATEGY STILL NOT VIABLE")
        print("     Even with optimized exits, expectancy ≤ 0")
        print("     Recommendation: ABANDON this edge")
    elif best.expectancy_r < 0.10:
        print()
        print("  ⚠️  STRATEGY MARGINAL")
        print("     Expectancy is positive but thin")
        print("     Proceed with caution, extended paper testing required")
    else:
        print()
        print("  ✅ STRATEGY IMPROVED")
        print("     Exit optimization successful")
        print("     Proceed to paper trading with new exit model")
    
    print()
    
    return verdict


if __name__ == "__main__":
    verdict = main()
