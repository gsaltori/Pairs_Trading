"""
Run Session Strategy Backtest

Asia Range → London Expansion edge evaluation.

EDGE HYPOTHESIS:
Asian session establishes a range with low liquidity.
London session brings institutional order flow that expands this range.
Asia close position (vs midpoint) predicts expansion direction.

LIQUIDITY MECHANICS:
- Asia: ~15% daily FX volume
- London: ~35% daily FX volume
- Direction bias from Asian "settlement" position
"""

import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from crv_engine.mt5_loader import initialize_mt5, shutdown_mt5, load_ohlc
from trading_system.session_backtest import (
    run_session_comparison,
    print_session_comparison,
    print_session_counterfactual,
    determine_session_viability,
)


INITIAL_CAPITAL = 100.0  # $100 micro account


def load_data_m30(n_bars: int = 10000):
    """Load M30 data for better session resolution."""
    print("=" * 75)
    print("LOADING MT5 DATA (M30)")
    print("=" * 75)
    
    if not initialize_mt5():
        raise RuntimeError("MT5 init failed")
    
    try:
        print(f"Loading EURUSD M30 ({n_bars} bars)...")
        eu_bars = load_ohlc("EURUSD", "M30", n_bars)
        
        print(f"Loading GBPUSD M30 ({n_bars} bars)...")
        gb_bars = load_ohlc("GBPUSD", "M30", n_bars)
        
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
        
        start = df_eu['timestamp'].iloc[0]
        end = df_eu['timestamp'].iloc[-1]
        days = (end - start).days
        print(f"   Span: {days} days (~{days/365:.1f} years)")
        
        return df_eu, df_gb
        
    finally:
        shutdown_mt5()


def load_data_h1(n_bars: int = 5000):
    """Load H1 data as alternative."""
    print("=" * 75)
    print("LOADING MT5 DATA (H1)")
    print("=" * 75)
    
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
    print("╔" + "═" * 73 + "╗")
    print("║" + " SESSION-BASED FX STRATEGY EVALUATION ".center(73) + "║")
    print("║" + " Asia Range → London Session Expansion ".center(73) + "║")
    print("║" + " Edge: Liquidity-Driven Directional Bias ".center(73) + "║")
    print("║" + f" Initial Capital: ${INITIAL_CAPITAL:.2f} ".center(73) + "║")
    print("╚" + "═" * 73 + "╝")
    print()
    
    # Strategy parameters
    print("STRATEGY PARAMETERS (LOCKED)")
    print("-" * 50)
    print("  Sessions (UTC):")
    print("    Asia:   00:00 - 06:00 (range establishment)")
    print("    London: 07:00 - 11:00 (expansion phase)")
    print()
    print("  Entry Logic:")
    print("    - Calculate Asia range (high, low, mid)")
    print("    - Bias: Close > mid+20% → BULLISH")
    print("    - Bias: Close < mid-20% → BEARISH")
    print("    - Entry: Break of Asia high (bull) / low (bear)")
    print("    - Max 1 trade per day")
    print()
    print("  Trade Management:")
    print("    - SL: Opposite side of Asia range + 3 pips")
    print("    - TP: 2.5 × risk distance")
    print("    - Time stop: End of London session (11:00 UTC)")
    print()
    print("  Filters:")
    print("    - Min Asia range: 15 pips")
    print("    - Max Asia range: 80 pips")
    print("    - Gatekeeper: Structural regime filter")
    print()
    
    # Load data - try H1 first (better session resolution than H4)
    try:
        df_eu, df_gb = load_data_h1(n_bars=8000)
    except Exception as e:
        print(f"H1 failed: {e}, trying M30...")
        try:
            df_eu, df_gb = load_data_m30(n_bars=15000)
        except Exception as e2:
            print(f"M30 also failed: {e2}")
            raise
    
    print()
    
    # Run comparison
    print("Running BACKTEST COMPARISON...")
    baseline, gated, cf = run_session_comparison(
        df_eu, df_gb, initial_capital=INITIAL_CAPITAL
    )
    print()
    
    # Print results
    print_session_comparison(baseline, gated)
    print_session_counterfactual(cf)
    
    # Select best result
    best_result = gated if gated.expectancy >= baseline.expectancy else baseline
    best_name = "Session+Gate" if best_result == gated else "Baseline"
    
    print(f"Evaluating best configuration: {best_name}")
    print()
    
    # Viability assessment
    verdict, explanation = determine_session_viability(best_result)
    
    # Summary
    print()
    print("╔" + "═" * 73 + "╗")
    print("║" + " SUMMARY ".center(73) + "║")
    print("╚" + "═" * 73 + "╝")
    print()
    
    print(f"  Strategy:           Asia → London Expansion (R=2.5)")
    print(f"  Best Config:        {best_name}")
    print(f"  Sessions Analyzed:  {best_result.sessions_analyzed}")
    print(f"  Valid Setups:       {best_result.valid_setups}")
    print(f"  Trades:             {best_result.total_trades}")
    print(f"  Trades/Month:       {best_result.trades_per_month:.1f}")
    print(f"  Win Rate:           {best_result.win_rate:.1%}")
    print(f"  Profit Factor:      {best_result.profit_factor:.2f}")
    print(f"  Expectancy ($):     ${best_result.expectancy:.2f}")
    print(f"  Expectancy (R):     {best_result.expectancy_r:.2f}R")
    print(f"  Max Drawdown:       {best_result.max_drawdown_pct:.1%}")
    print(f"  Net PnL:            ${best_result.total_pnl:.2f}")
    print()
    
    # Breakeven analysis
    breakeven_wr = 1 / (1 + 2.5)
    print(f"  Mathematical Breakeven WR (R=2.5): {breakeven_wr:.1%}")
    print(f"  Actual WR:                         {best_result.win_rate:.1%}")
    print(f"  WR vs Breakeven:                   {best_result.win_rate - breakeven_wr:+.1%}")
    print()
    
    # Exit analysis
    total_exits = best_result.exits_by_sl + best_result.exits_by_tp + best_result.exits_by_time
    if total_exits > 0:
        print("  Exit Distribution:")
        print(f"    SL hits:    {best_result.exits_by_sl} ({100*best_result.exits_by_sl/total_exits:.0f}%)")
        print(f"    TP hits:    {best_result.exits_by_tp} ({100*best_result.exits_by_tp/total_exits:.0f}%)")
        print(f"    Time stops: {best_result.exits_by_time} ({100*best_result.exits_by_time/total_exits:.0f}%)")
        print()
    
    # Clear recommendation
    if verdict == "NOT VIABLE":
        print("  ⛔ VERDICT: NOT VIABLE")
        print()
        print("     The session edge is INVALID.")
        print("     Either expectancy is negative or frequency is too low.")
        print()
        print("     RECOMMENDATION: ABANDON THIS EDGE")
        print()
        print("     Options:")
        print("     1. Test different session definitions")
        print("     2. Test different pairs (GBP-based)")
        print("     3. Research different liquidity-based edges")
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
    
    # Gatekeeper impact
    if gated.blocked_by_gatekeeper > 0:
        print("  Gatekeeper Impact:")
        print(f"    Trades blocked: {gated.blocked_by_gatekeeper}")
        print(f"    {cf['effectiveness']}")
        
        if "HARMFUL" in cf['effectiveness']:
            print("    ⚠️  WARNING: Gatekeeper may be blocking profitable trades")
            print("       Consider reviewing gatekeeper thresholds")
        print()
    
    return verdict


if __name__ == "__main__":
    verdict = main()
