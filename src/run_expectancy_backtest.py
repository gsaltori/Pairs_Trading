"""
Run Single-Strategy Expectancy Backtest

Executes the 3-way comparison:
1. Baseline Trend (no filters)
2. Trend + Gatekeeper
3. Trend + Gatekeeper + MRF

With counterfactual analysis and viability assessment.

GOAL: Maximize expectancy, not win rate.
"""

import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from crv_engine.mt5_loader import initialize_mt5, shutdown_mt5, load_ohlc
from trading_system.single_strategy_backtest import (
    run_single_strategy_comparison,
    print_comparison_table,
    print_mrf_counterfactual,
    determine_viability,
)


INITIAL_CAPITAL = 100.0  # $100 micro account


def load_data(n_bars: int = 5000):
    """Load and align data from MT5."""
    print("=" * 85)
    print("LOADING MT5 DATA")
    print("=" * 85)
    
    if not initialize_mt5():
        raise RuntimeError("MT5 init failed")
    
    try:
        print(f"Loading EURUSD H4 ({n_bars} bars)...")
        eu_bars = load_ohlc("EURUSD", "H4", n_bars)
        
        print(f"Loading GBPUSD H4 ({n_bars} bars)...")
        gb_bars = load_ohlc("GBPUSD", "H4", n_bars)
        
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
        
        # Align by timestamp
        df_eu = df_eu.set_index('timestamp')
        df_gb = df_gb.set_index('timestamp')
        
        common = df_eu.index.intersection(df_gb.index)
        df_eu = df_eu.loc[common].reset_index()
        df_gb = df_gb.loc[common].reset_index()
        
        print(f"✅ Loaded {len(df_eu)} aligned bars")
        print(f"   Range: {df_eu['timestamp'].iloc[0]} to {df_eu['timestamp'].iloc[-1]}")
        
        # Calculate data span
        start = df_eu['timestamp'].iloc[0]
        end = df_eu['timestamp'].iloc[-1]
        days = (end - start).days
        print(f"   Span: {days} days (~{days/365:.1f} years)")
        
        return df_eu, df_gb
        
    finally:
        shutdown_mt5()


def main():
    print()
    print("╔" + "═" * 83 + "╗")
    print("║" + " SINGLE-STRATEGY EXPECTANCY BACKTEST ".center(83) + "║")
    print("║" + " Focus: EXPECTANCY > Win Rate ".center(83) + "║")
    print("║" + f" Initial Capital: ${INITIAL_CAPITAL:.2f} ".center(83) + "║")
    print("╚" + "═" * 83 + "╝")
    print()
    
    # Load data
    df_eu, df_gb = load_data(n_bars=5000)
    print()
    
    # Run comparison
    print("Running BACKTEST COMPARISON...")
    baseline, gate, full, mrf_cf = run_single_strategy_comparison(
        df_eu, df_gb, initial_capital=INITIAL_CAPITAL
    )
    print()
    
    # Print results
    print_comparison_table(baseline, gate, full)
    print_mrf_counterfactual(mrf_cf)
    
    # Viability assessment
    verdict, explanation = determine_viability(full)
    
    # Summary
    print()
    print("╔" + "═" * 83 + "╗")
    print("║" + " SUMMARY ".center(83) + "║")
    print("╚" + "═" * 83 + "╝")
    print()
    
    print(f"  Best Configuration: Trend + Gatekeeper + MRF")
    print(f"  Final Expectancy:   ${full.expectancy:.2f} per trade")
    print(f"  Final Expectancy:   {full.expectancy_r:.2f}R per trade")
    print(f"  Final Profit Factor: {full.profit_factor:.2f}")
    print(f"  Final Max Drawdown:  {full.max_drawdown_pct:.1%}")
    print(f"  Final Net PnL:       ${full.total_pnl:.2f}")
    print()
    
    # Clear recommendation
    if verdict == "NOT VIABLE":
        print("  ⛔ RECOMMENDATION: DO NOT DEPLOY")
        print("     This strategy is not suitable for real capital.")
        print("     Expectancy is insufficient to overcome costs and variance.")
        print()
        print("     OPTIONS:")
        print("     1. Abandon this edge entirely")
        print("     2. Research alternative entry/exit logic")
        print("     3. Test on different pairs/timeframes")
        print()
    elif verdict == "MARGINAL":
        print("  ⚠️  RECOMMENDATION: EXTENDED PAPER TRADING ONLY")
        print("     Strategy shows marginal viability.")
        print("     Do not risk real capital until forward-tested.")
        print()
    elif verdict == "CAUTIOUS PROCEED":
        print("  🟡 RECOMMENDATION: CAUTIOUS PAPER TRADING")
        print("     Strategy shows promise but needs validation.")
        print("     Paper trade for 3+ months before any live capital.")
        print()
    else:
        print("  ✅ RECOMMENDATION: PROCEED TO PAPER TRADING")
        print("     Strategy meets viability criteria.")
        print("     Begin with extended paper trading, then micro capital.")
        print()
    
    return verdict


if __name__ == "__main__":
    verdict = main()
