"""
Run Multi-Strategy Portfolio Backtest

Executes:
1. Individual strategy backtests (with gatekeeper)
2. Combined portfolio backtest
3. Baseline comparison (no gatekeeper)
4. Counterfactual analysis

Initial capital: $100 (micro account)
"""

import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from crv_engine.mt5_loader import initialize_mt5, shutdown_mt5, load_ohlc
from trading_system.portfolio_backtest import (
    PortfolioBacktest,
    run_individual_backtests,
    counterfactual_analysis,
    StrategyResult,
    PortfolioResult,
)


INITIAL_CAPITAL = 100.0  # $100 micro account


def load_data(n_bars: int = 5000):
    """Load and align data."""
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
        
        df_eu = df_eu.set_index('timestamp')
        df_gb = df_gb.set_index('timestamp')
        
        common = df_eu.index.intersection(df_gb.index)
        df_eu = df_eu.loc[common].reset_index()
        df_gb = df_gb.loc[common].reset_index()
        
        print(f"✅ Loaded {len(df_eu)} aligned bars")
        print(f"   Range: {df_eu['timestamp'].iloc[0]} to {df_eu['timestamp'].iloc[-1]}")
        
        return df_eu, df_gb
        
    finally:
        shutdown_mt5()


def print_individual_results(results: dict):
    """Print individual strategy results."""
    print()
    print("=" * 75)
    print("INDIVIDUAL STRATEGY RESULTS (With Gatekeeper)")
    print("=" * 75)
    print()
    
    header = f"{'Strategy':<25} {'Trades':>8} {'Win%':>8} {'PF':>8} {'PnL':>12} {'MaxDD':>8} {'Blocked':>8}"
    print(header)
    print("-" * 75)
    
    for name, r in results.items():
        print(
            f"{name:<25} "
            f"{r.total_trades:>8} "
            f"{r.win_rate:>7.1%} "
            f"{r.profit_factor:>8.2f} "
            f"${r.total_pnl:>10.2f} "
            f"{r.max_drawdown_pct:>7.1%} "
            f"{r.blocked_count:>8}"
        )
    
    print()


def print_portfolio_comparison(baseline: PortfolioResult, gated: PortfolioResult):
    """Print portfolio comparison."""
    print()
    print("=" * 75)
    print("PORTFOLIO COMPARISON: BASELINE vs GATED")
    print("=" * 75)
    print()
    
    metrics = [
        ("Total Trades", baseline.total_trades, gated.total_trades),
        ("Wins", baseline.wins, gated.wins),
        ("Losses", baseline.losses, gated.losses),
        ("Win Rate", f"{baseline.win_rate:.1%}", f"{gated.win_rate:.1%}"),
        ("Profit Factor", f"{baseline.profit_factor:.2f}", f"{gated.profit_factor:.2f}"),
        ("Expectancy", f"${baseline.expectancy:.2f}", f"${gated.expectancy:.2f}"),
        ("Total PnL", f"${baseline.total_pnl:.2f}", f"${gated.total_pnl:.2f}"),
        ("Max Drawdown", f"{baseline.max_drawdown_pct:.1%}", f"{gated.max_drawdown_pct:.1%}"),
        ("Trades Blocked", "N/A", f"{gated.blocked_count}"),
    ]
    
    print(f"{'Metric':<20} {'Baseline':>15} {'Gated':>15}")
    print("-" * 55)
    
    for name, b, g in metrics:
        print(f"{name:<20} {str(b):>15} {str(g):>15}")
    
    print()
    
    # By strategy breakdown
    print("Trades by Strategy:")
    for strat, count in gated.trades_by_strategy.items():
        blocked = gated.blocks_by_strategy.get(strat, 0)
        print(f"  {strat:<25} Executed: {count:>4}, Blocked: {blocked:>4}")
    
    print()


def print_counterfactual(cf: dict):
    """Print counterfactual analysis."""
    print("=" * 75)
    print("COUNTERFACTUAL ANALYSIS (Blocked Trades)")
    print("=" * 75)
    print()
    
    print(f"Total Blocked:           {cf['total']}")
    print(f"Resolved (SL/TP hit):    {cf['resolved']}")
    print(f"Would Have Won:          {cf['wins']}")
    print(f"Would Have Lost:         {cf['losses']}")
    print(f"Counterfactual WR:       {cf['win_rate']:.1%}")
    print(f"Counterfactual PnL:      ${cf['pnl']:.2f}")
    print()


def determine_verdict(baseline: PortfolioResult, gated: PortfolioResult, cf: dict) -> str:
    """Determine final verdict."""
    # Robustness improved?
    dd_improved = baseline.max_drawdown_pct - gated.max_drawdown_pct > 0.005
    wr_improved = gated.win_rate - baseline.win_rate > 0.02
    pf_improved = gated.profit_factor - baseline.profit_factor > 0.1
    
    robustness_improved = dd_improved or wr_improved or pf_improved
    
    # Expectancy survived?
    if baseline.expectancy != 0:
        exp_ratio = gated.expectancy / baseline.expectancy
    else:
        exp_ratio = 1.0 if gated.expectancy >= 0 else 0
    
    expectancy_survived = exp_ratio > 0.7 and gated.expectancy > 0
    
    # Gatekeeper effective?
    gatekeeper_effective = gated.win_rate > cf['win_rate'] if cf['win_rate'] > 0 else True
    
    print("=" * 75)
    print("VERDICT")
    print("=" * 75)
    print()
    print(f"Drawdown Improved:      {dd_improved} ({baseline.max_drawdown_pct:.1%} → {gated.max_drawdown_pct:.1%})")
    print(f"Win Rate Improved:      {wr_improved} ({baseline.win_rate:.1%} → {gated.win_rate:.1%})")
    print(f"Profit Factor Improved: {pf_improved} ({baseline.profit_factor:.2f} → {gated.profit_factor:.2f})")
    print(f"Expectancy Ratio:       {exp_ratio:.1%}")
    print(f"Gatekeeper Effective:   {gatekeeper_effective}")
    print()
    
    if robustness_improved and expectancy_survived:
        verdict = "VALID"
        msg = "Portfolio improves robustness without destroying expectancy"
    elif robustness_improved and not expectancy_survived:
        verdict = "FILTER TOO AGGRESSIVE"
        msg = "Gatekeeper improves robustness but kills expectancy"
    else:
        verdict = "NO VALUE ADD"
        msg = "Gatekeeper does not materially improve robustness"
    
    print(f"FINAL VERDICT: {verdict}")
    print(f"               {msg}")
    print()
    print("=" * 75)
    
    return verdict


def main():
    print()
    print("╔" + "═" * 73 + "╗")
    print("║" + " MULTI-STRATEGY PORTFOLIO BACKTEST ".center(73) + "║")
    print("║" + f" Initial Capital: ${INITIAL_CAPITAL:.2f} (Micro Account) ".center(73) + "║")
    print("╚" + "═" * 73 + "╝")
    print()
    
    # Load data
    df_eu, df_gb = load_data(n_bars=5000)
    print()
    
    # 1. Individual backtests
    print("Running INDIVIDUAL STRATEGY BACKTESTS...")
    individual_results = run_individual_backtests(
        df_eu.copy(), df_gb.copy(), initial_capital=INITIAL_CAPITAL
    )
    print_individual_results(individual_results)
    
    # 2. Baseline portfolio (no gatekeeper)
    print("Running BASELINE PORTFOLIO (No Gatekeeper)...")
    baseline_bt = PortfolioBacktest(
        use_gatekeeper=False,
        initial_capital=INITIAL_CAPITAL,
    )
    baseline = baseline_bt.run(df_eu.copy(), df_gb.copy())
    print(f"  Completed: {baseline.total_trades} trades, PnL: ${baseline.total_pnl:.2f}")
    print()
    
    # 3. Gated portfolio
    print("Running GATED PORTFOLIO (With Gatekeeper)...")
    gated_bt = PortfolioBacktest(
        use_gatekeeper=True,
        initial_capital=INITIAL_CAPITAL,
    )
    gated = gated_bt.run(df_eu.copy(), df_gb.copy())
    print(f"  Completed: {gated.total_trades} trades, {gated.blocked_count} blocked, PnL: ${gated.total_pnl:.2f}")
    print()
    
    # 4. Counterfactual
    cf = counterfactual_analysis(gated_bt.blocked_trades, df_eu)
    
    # 5. Print results
    print_portfolio_comparison(baseline, gated)
    print_counterfactual(cf)
    
    # 6. Verdict
    verdict = determine_verdict(baseline, gated, cf)
    
    # 7. Safety notes
    print()
    print("╔" + "═" * 73 + "╗")
    print("║" + " SAFETY NOTES FOR LIVE DEPLOYMENT ".center(73) + "║")
    print("╚" + "═" * 73 + "╝")
    print("""
    1. INITIAL CAPITAL: $100 micro account
       - Per-trade risk capped at fractions of a dollar
       - Position sizes will be 0.01 lots minimum
       - Commission/spread impact is significant at this scale
    
    2. DRAWDOWN GOVERNORS:
       - DD ≥ 5%  → Risk reduced by 50% ($5 on $100)
       - DD ≥ 8%  → SYSTEM HALT ($8 on $100)
       - 3 consecutive losses → 24h cooling off
    
    3. RISK LIMITS (per strategy):
       - Trend Continuation: 0.30% = $0.30
       - Trend Pullback:     0.25% = $0.25
       - Volatility Exp:     0.20% = $0.20
       - Total max:          0.75% = $0.75
    
    4. BEFORE LIVE:
       - Run in DRY_RUN mode for 2+ weeks
       - Verify logs are being created
       - Test MT5 connection stability
       - Confirm broker accepts micro lots (0.01)
    
    5. EXPECTATIONS:
       - At $100, expect slow growth
       - Each trade risks ~$0.25-0.30
       - A 10-trade losing streak = $2.50-3.00 loss
       - System will halt at $92 equity (-8%)
    """)
    
    return verdict


if __name__ == "__main__":
    verdict = main()
