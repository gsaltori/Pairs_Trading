"""
Run Breakout Strategy Backtest

New edge evaluation:
- Range Breakout with R=2.5 asymmetric payoff
- Baseline vs +Gatekeeper comparison
- Explicit kill criteria evaluation

EDGE HYPOTHESIS:
Volatility contraction → expansion breakouts
with wide stops and asymmetric targets.
"""

import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from crv_engine.mt5_loader import initialize_mt5, shutdown_mt5, load_ohlc
from trading_system.breakout_backtest import (
    run_breakout_comparison,
    print_breakout_comparison,
    print_counterfactual,
    determine_breakout_viability,
)


INITIAL_CAPITAL = 100.0  # $100 micro account


def load_data(n_bars: int = 5000):
    """Load and align data from MT5."""
    print("=" * 75)
    print("LOADING MT5 DATA")
    print("=" * 75)
    
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
        
        # Data span
        start = df_eu['timestamp'].iloc[0]
        end = df_eu['timestamp'].iloc[-1]
        days = (end - start).days
        print(f"   Span: {days} days (~{days/365:.1f} years)")
        
        return df_eu, df_gb
        
    finally:
        shutdown_mt5()


def main():
    print()
    print("╔" + "═" * 73 + "╗")
    print("║" + " BREAKOUT STRATEGY EVALUATION ".center(73) + "║")
    print("║" + " Range Compression → Expansion Breakout ".center(73) + "║")
    print("║" + " Target R/R: 2.5 (Asymmetric Payoff) ".center(73) + "║")
    print("║" + f" Initial Capital: ${INITIAL_CAPITAL:.2f} ".center(73) + "║")
    print("╚" + "═" * 73 + "╝")
    print()
    
    # Strategy parameters
    print("STRATEGY PARAMETERS (LOCKED)")
    print("-" * 40)
    print("  Range Lookback:        6 bars (24h on H4)")
    print("  ATR Period:            14")
    print("  Compression Threshold: Range < 0.8 × ATR")
    print("  Risk/Reward:           2.5")
    print("  SL Buffer:             0.1 × ATR")
    print("  Cooldown:              3 bars")
    print()
    
    # Load data
    df_eu, df_gb = load_data(n_bars=5000)
    print()
    
    # Run comparison
    print("Running BACKTEST COMPARISON...")
    baseline, gated, cf = run_breakout_comparison(
        df_eu, df_gb, initial_capital=INITIAL_CAPITAL
    )
    print()
    
    # Print results
    print_breakout_comparison(baseline, gated)
    print_counterfactual(cf)
    
    # Viability assessment
    # Use gated result for final verdict
    best_result = gated if gated.expectancy >= baseline.expectancy else baseline
    best_name = "Breakout+Gate" if best_result == gated else "Baseline"
    
    print(f"Evaluating best configuration: {best_name}")
    print()
    
    verdict, explanation = determine_breakout_viability(best_result)
    
    # Summary
    print()
    print("╔" + "═" * 73 + "╗")
    print("║" + " SUMMARY ".center(73) + "║")
    print("╚" + "═" * 73 + "╝")
    print()
    
    print(f"  Strategy:           Range Breakout (R=2.5)")
    print(f"  Best Config:        {best_name}")
    print(f"  Trades:             {best_result.total_trades}")
    print(f"  Win Rate:           {best_result.win_rate:.1%}")
    print(f"  Profit Factor:      {best_result.profit_factor:.2f}")
    print(f"  Expectancy ($):     ${best_result.expectancy:.2f}")
    print(f"  Expectancy (R):     {best_result.expectancy_r:.2f}R")
    print(f"  Max Drawdown:       {best_result.max_drawdown_pct:.1%}")
    print(f"  Net PnL:            ${best_result.total_pnl:.2f}")
    print()
    
    # Breakeven analysis
    breakeven_wr = 1 / (1 + 2.5)  # 28.6% for R=2.5
    print(f"  Mathematical Breakeven WR (R=2.5): {breakeven_wr:.1%}")
    print(f"  Actual WR:                         {best_result.win_rate:.1%}")
    print(f"  WR vs Breakeven:                   {best_result.win_rate - breakeven_wr:+.1%}")
    print()
    
    # Clear recommendation
    if verdict == "NOT VIABLE":
        print("  ⛔ VERDICT: NOT VIABLE")
        print()
        print("     The breakout edge is INVALID.")
        print("     Expectancy is insufficient for profitability.")
        print()
        print("     RECOMMENDATION: ABANDON THIS EDGE")
        print()
        print("     Options:")
        print("     1. Test different timeframe (D1)")
        print("     2. Test different pair")
        print("     3. Research completely different edge type")
        print()
    elif verdict == "MARGINAL":
        print("  ⚠️  VERDICT: MARGINAL")
        print()
        print("     Strategy shows weak potential.")
        print("     Extended paper trading only.")
        print()
    elif verdict == "CAUTIOUS PROCEED":
        print("  🟡 VERDICT: CAUTIOUS PROCEED")
        print()
        print("     Strategy shows promise but needs validation.")
        print("     Paper trade for 3+ months.")
        print()
    else:
        print("  ✅ VERDICT: VIABLE")
        print()
        print("     Strategy meets viability criteria.")
        print("     Proceed to paper trading, then micro capital.")
        print()
    
    # Trade frequency analysis
    days = (df_eu['timestamp'].iloc[-1] - df_eu['timestamp'].iloc[0]).days
    trades_per_month = best_result.total_trades / (days / 30) if days > 0 else 0
    print(f"  Trade Frequency: {trades_per_month:.1f} trades/month")
    
    if trades_per_month < 2:
        print("  ⚠️  Low frequency - expect long periods without trades")
    
    print()
    
    return verdict


if __name__ == "__main__":
    verdict = main()
