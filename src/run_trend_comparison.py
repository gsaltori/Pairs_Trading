"""
Run Trend Strategy Comparison: Baseline vs Gated

Executes the same trend-following strategy with and without
the structural gatekeeper, producing a side-by-side comparison.
"""

import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from trend_strategy_backtest import (
    TrendFollowingStrategy,
    analyze_blocked_trades,
    get_block_reason_distribution,
)
from crv_engine.mt5_loader import (
    initialize_mt5,
    shutdown_mt5,
    load_ohlc,
)


def load_data(n_bars: int = 5000):
    """Load EURUSD and GBPUSD H4 data from MT5."""
    print("=" * 70)
    print("DATA LOADING")
    print("=" * 70)
    
    if not initialize_mt5():
        raise RuntimeError("Failed to initialize MT5")
    
    try:
        print(f"Loading EURUSD H4 ({n_bars} bars)...")
        eurusd_bars = load_ohlc("EURUSD", "H4", n_bars)
        
        print(f"Loading GBPUSD H4 ({n_bars} bars)...")
        gbpusd_bars = load_ohlc("GBPUSD", "H4", n_bars)
        
        if eurusd_bars is None or gbpusd_bars is None:
            raise RuntimeError("Failed to load data")
        
        # Convert to DataFrames
        df_eurusd = pd.DataFrame([{
            'timestamp': bar.timestamp,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
        } for bar in eurusd_bars])
        
        df_gbpusd = pd.DataFrame([{
            'timestamp': bar.timestamp,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
        } for bar in gbpusd_bars])
        
        # Align by timestamp
        df_eurusd = df_eurusd.set_index('timestamp')
        df_gbpusd = df_gbpusd.set_index('timestamp')
        
        common_idx = df_eurusd.index.intersection(df_gbpusd.index)
        df_eurusd = df_eurusd.loc[common_idx].reset_index()
        df_gbpusd = df_gbpusd.loc[common_idx].reset_index()
        
        print(f"✅ Loaded {len(df_eurusd)} aligned bars")
        print(f"   Date range: {df_eurusd['timestamp'].iloc[0]} to {df_eurusd['timestamp'].iloc[-1]}")
        
        return df_eurusd, df_gbpusd
        
    finally:
        shutdown_mt5()
        print("   MT5 connection closed\n")


def print_results_table(baseline: dict, gated: dict):
    """Print comparison table."""
    print("=" * 70)
    print("BACKTEST COMPARISON: BASELINE vs GATED")
    print("=" * 70)
    print()
    
    def fmt_pct(v): return f"{v:.2%}"
    def fmt_num(v): return f"{v:.2f}"
    def fmt_int(v): return f"{int(v)}"
    def fmt_dollar(v): return f"${v:,.2f}"
    
    def delta(b, g, fmt_func, higher_better=True):
        diff = g - b
        if fmt_func == fmt_pct:
            s = f"{diff:+.2%}"
        elif fmt_func == fmt_dollar:
            s = f"${diff:+,.2f}"
        else:
            s = f"{diff:+.2f}"
        return s
    
    metrics = [
        ("Total Trades", 'total_trades', fmt_int),
        ("Wins", 'win_count', fmt_int),
        ("Losses", 'loss_count', fmt_int),
        ("Win Rate", 'win_rate', fmt_pct),
        ("Profit Factor", 'profit_factor', fmt_num),
        ("Expectancy", 'expectancy', fmt_dollar),
        ("Max Drawdown", 'max_drawdown', fmt_pct),
        ("Net Return", 'net_return', fmt_pct),
        ("Total PnL", 'total_pnl', fmt_dollar),
        ("Avg Trade", 'avg_trade', fmt_dollar),
        ("Avg Duration (bars)", 'avg_duration_bars', fmt_num),
    ]
    
    print(f"{'Metric':<22} {'Baseline':>14} {'Gated':>14} {'Δ':>14}")
    print("-" * 70)
    
    for label, key, fmt in metrics:
        b_val = baseline.get(key, 0)
        g_val = gated.get(key, 0)
        d = delta(b_val, g_val, fmt)
        print(f"{label:<22} {fmt(b_val):>14} {fmt(g_val):>14} {d:>14}")
    
    print()


def print_gatekeeper_diagnostics(gated: dict, counterfactual: dict, df_eurusd):
    """Print gatekeeper-specific diagnostics."""
    print("=" * 70)
    print("GATEKEEPER DIAGNOSTICS")
    print("=" * 70)
    print()
    
    total_signals = gated['total_trades'] + len(gated['blocked_trades'])
    blocked = len(gated['blocked_trades'])
    block_rate = blocked / total_signals if total_signals > 0 else 0
    
    print(f"Total Signals Generated:     {total_signals}")
    print(f"Trades Executed:             {gated['total_trades']}")
    print(f"Trades Blocked:              {blocked}")
    print(f"Block Rate:                  {block_rate:.2%}")
    print()
    
    # Block reason distribution
    reasons = get_block_reason_distribution(gated['blocked_trades'])
    print("Block Reason Distribution:")
    for reason, count in reasons.items():
        pct = count / blocked if blocked > 0 else 0
        print(f"  {reason:<28} {count:>4} ({pct:>6.1%})")
    print()
    
    # Counterfactual analysis
    print("COUNTERFACTUAL ANALYSIS (What if blocked trades executed?):")
    print(f"  Blocked trades resolved:   {counterfactual['resolved_blocked']}")
    print(f"  Would have won:            {counterfactual['would_have_won']}")
    print(f"  Would have lost:           {counterfactual['would_have_lost']}")
    print(f"  Counterfactual win rate:   {counterfactual['counterfactual_win_rate']:.2%}")
    print(f"  Would have PnL:            ${counterfactual['would_have_pnl']:,.2f}")
    print()
    
    # Compare win rates
    allowed_wr = gated['win_rate']
    blocked_wr = counterfactual['counterfactual_win_rate']
    print(f"  Win Rate Comparison:")
    print(f"    Allowed trades:          {allowed_wr:.2%}")
    print(f"    Blocked trades:          {blocked_wr:.2%}")
    print(f"    Difference:              {allowed_wr - blocked_wr:+.2%}")
    print()


def render_conclusion(baseline: dict, gated: dict, counterfactual: dict) -> str:
    """
    Render final conclusion.
    
    Returns one of:
    - VALID
    - FILTER TOO AGGRESSIVE
    - NO VALUE ADD
    """
    print("=" * 70)
    print("FINAL CONCLUSION")
    print("=" * 70)
    print()
    
    # Key metrics
    b_wr = baseline['win_rate']
    g_wr = gated['win_rate']
    wr_delta = g_wr - b_wr
    
    b_dd = baseline['max_drawdown']
    g_dd = gated['max_drawdown']
    dd_improvement = b_dd - g_dd  # Positive = better
    
    b_exp = baseline['expectancy']
    g_exp = gated['expectancy']
    exp_ratio = g_exp / b_exp if b_exp != 0 else 0
    
    b_pf = baseline['profit_factor']
    g_pf = gated['profit_factor']
    pf_delta = g_pf - b_pf
    
    blocked_wr = counterfactual['counterfactual_win_rate']
    allowed_wr = gated['win_rate']
    
    print(f"Win Rate Improvement:        {wr_delta:+.2%}")
    print(f"Drawdown Improvement:        {dd_improvement:+.2%}")
    print(f"Profit Factor Change:        {pf_delta:+.2f}")
    print(f"Expectancy Ratio:            {exp_ratio:.2%}")
    print(f"Allowed WR vs Blocked WR:    {allowed_wr:.2%} vs {blocked_wr:.2%}")
    print()
    
    # Determine robustness improvement
    robustness_improved = (
        dd_improvement > 0.005 or  # Drawdown reduced
        wr_delta > 0.02 or         # Win rate improved 2%+
        pf_delta > 0.1             # Profit factor improved
    )
    
    # Determine if expectancy survived
    expectancy_survived = exp_ratio > 0.7 and g_exp > 0  # At least 70% of original
    expectancy_killed = exp_ratio < 0.4 or g_exp <= 0
    
    # Additional check: blocked trades had lower win rate
    gatekeeper_effective = allowed_wr > blocked_wr
    
    print("-" * 70)
    
    if robustness_improved and expectancy_survived:
        conclusion = "VALID"
        print("VERDICT: VALID")
        print()
        print("The gatekeeper improves robustness without destroying expectancy.")
        if gatekeeper_effective:
            print(f"Blocked trades had {allowed_wr - blocked_wr:.1%} lower win rate,")
            print("confirming the gatekeeper correctly identifies failure conditions.")
    elif robustness_improved and expectancy_killed:
        conclusion = "FILTER TOO AGGRESSIVE"
        print("VERDICT: FILTER TOO AGGRESSIVE")
        print()
        print("The gatekeeper improves some metrics but kills expectancy.")
        print("Filter blocks too many winning trades.")
    else:
        conclusion = "NO VALUE ADD"
        print("VERDICT: NO VALUE ADD")
        print()
        print("The gatekeeper does not materially improve robustness.")
        if not gatekeeper_effective:
            print("Blocked trades had similar or better win rate than allowed trades.")
    
    print()
    print("=" * 70)
    
    return conclusion


def main():
    """Main execution."""
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " TREND STRATEGY BACKTEST: BASELINE vs GATED ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    # Load data
    df_eurusd, df_gbpusd = load_data(n_bars=5000)
    
    # Run BASELINE (no gatekeeper)
    print("=" * 70)
    print("RUNNING BASELINE BACKTEST (No Gatekeeper)")
    print("=" * 70)
    baseline_strategy = TrendFollowingStrategy(use_gatekeeper=False)
    baseline_results = baseline_strategy.run_backtest(df_eurusd, df_gbpusd)
    print(f"✅ Completed: {baseline_results['total_trades']} trades")
    print()
    
    # Run GATED (with gatekeeper)
    print("=" * 70)
    print("RUNNING GATED BACKTEST (With Gatekeeper)")
    print("=" * 70)
    gated_strategy = TrendFollowingStrategy(use_gatekeeper=True)
    gated_results = gated_strategy.run_backtest(df_eurusd, df_gbpusd)
    print(f"✅ Completed: {gated_results['total_trades']} trades")
    print(f"   Blocked:   {len(gated_results['blocked_trades'])} trades")
    print()
    
    # Counterfactual analysis
    counterfactual = analyze_blocked_trades(
        gated_results['blocked_trades'],
        df_eurusd,
    )
    
    # Print results
    print_results_table(baseline_results, gated_results)
    print_gatekeeper_diagnostics(gated_results, counterfactual, df_eurusd)
    
    # Render conclusion
    conclusion = render_conclusion(baseline_results, gated_results, counterfactual)
    
    return conclusion


if __name__ == "__main__":
    conclusion = main()
