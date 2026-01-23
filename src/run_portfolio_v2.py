"""
Run Portfolio Backtest

Tests the micro-edge portfolio system:
- Session Directional Bias Engine (filter)
- Three complementary strategies
- Portfolio-level risk management
- Gatekeeper integration
"""

import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from crv_engine.mt5_loader import initialize_mt5, shutdown_mt5, load_ohlc
from trading_system.portfolio_backtest_v2 import (
    PortfolioBacktest,
    print_portfolio_results,
    determine_portfolio_viability,
)


INITIAL_CAPITAL = 100.0


def load_data_m30(n_bars: int = 15000):
    """Load M30 data for intraday strategies."""
    print("=" * 80)
    print("LOADING MT5 DATA (M30)")
    print("=" * 80)
    
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
        print(f"   Span: {days} days (~{days/30:.1f} months)")
        
        return df_eu, df_gb
        
    finally:
        shutdown_mt5()


def main():
    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " MICRO-EDGE PORTFOLIO BACKTEST ".center(78) + "║")
    print("║" + " Session Bias Filter + Complementary Strategies ".center(78) + "║")
    print("║" + f" Initial Capital: ${INITIAL_CAPITAL:.2f} ".center(78) + "║")
    print("╚" + "═" * 78 + "╝")
    print()
    
    # System architecture
    print("SYSTEM ARCHITECTURE")
    print("-" * 60)
    print()
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │          SESSION DIRECTIONAL BIAS ENGINE            │")
    print("  │                                                     │")
    print("  │   Asia Range → BULL / BEAR / NEUTRAL bias          │")
    print("  │   (This is a FILTER, not a trade generator)        │")
    print("  └───────────────────────┬─────────────────────────────┘")
    print("                          │")
    print("                          ▼")
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │              MICRO-EDGE STRATEGIES                  │")
    print("  │                                                     │")
    print("  │   1. PULLBACK_SCALPER   → 0.5R target              │")
    print("  │   2. MOMENTUM_BURST     → 0.4R target              │")
    print("  │   3. PIVOT_BOUNCE       → 0.6R target              │")
    print("  │                                                     │")
    print("  │   Only trade WHEN bias is non-neutral AND          │")
    print("  │   DIRECTION matches bias                           │")
    print("  └───────────────────────┬─────────────────────────────┘")
    print("                          │")
    print("                          ▼")
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │                GATEKEEPER FILTER                    │")
    print("  │                                                     │")
    print("  │   • |Z-score| ≤ 3.0                                │")
    print("  │   • Correlation trend ≥ -0.05                      │")
    print("  │   • Volatility ratio ≥ 0.7                         │")
    print("  └───────────────────────┬─────────────────────────────┘")
    print("                          │")
    print("                          ▼")
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │             PORTFOLIO RISK MANAGER                  │")
    print("  │                                                     │")
    print("  │   • Max 2% daily risk budget                       │")
    print("  │   • Max 1% daily loss → halt                       │")
    print("  │   • Max 3% weekly loss → halt                      │")
    print("  │   • Max 1.5% directional exposure                  │")
    print("  │   • 5 consecutive losses → 4h cooldown             │")
    print("  └─────────────────────────────────────────────────────┘")
    print()
    
    # Strategy details
    print("STRATEGY SPECIFICATIONS")
    print("-" * 60)
    print()
    print("  1. PULLBACK_SCALPER (London session)")
    print("     - Entry: Pullback to Asia mid, reversal in bias direction")
    print("     - Target: 0.5R")
    print("     - SL: Opposite Asia boundary")
    print("     - Expected: 1-3 per session")
    print()
    print("  2. MOMENTUM_BURST (London + NY)")
    print("     - Entry: Strong momentum candle (body > 1.2 × ATR)")
    print("     - Target: 0.4R")
    print("     - SL: 1.5 × ATR")
    print("     - Expected: 2-4 per day")
    print()
    print("  3. PIVOT_BOUNCE (London + early NY)")
    print("     - Entry: Touch Asia pivot, rejection candle")
    print("     - Target: 0.6R")
    print("     - SL: Through pivot by 0.5 × range")
    print("     - Expected: 0-2 per day")
    print()
    
    # Load data
    df_eu, df_gb = load_data_m30(n_bars=15000)
    print()
    
    # Run backtest
    print("Running PORTFOLIO BACKTEST...")
    print()
    
    backtest = PortfolioBacktest(
        initial_capital=INITIAL_CAPITAL,
        use_gatekeeper=True,
    )
    
    metrics = backtest.run(df_eu, df_gb)
    
    # Print results
    print_portfolio_results(metrics, INITIAL_CAPITAL)
    
    # Viability assessment
    verdict, explanation, tradeable = determine_portfolio_viability(metrics)
    
    # Summary and roadmap
    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " DEPLOYMENT ROADMAP ".center(78) + "║")
    print("╚" + "═" * 78 + "╝")
    print()
    
    if verdict == "VIABLE":
        print("  ✅ PORTFOLIO IS VIABLE FOR PAPER TRADING")
        print()
        print("  PHASE 1: Paper Trading (4-6 weeks)")
        print("    - Run in dry-run mode")
        print("    - Validate signal frequency matches backtest")
        print("    - Confirm risk management triggers work")
        print("    - Target: 50+ trades")
        print()
        print("  PHASE 2: Micro Capital ($20-50)")
        print("    - Trade with minimal capital")
        print("    - Focus on execution quality")
        print("    - Validate slippage assumptions")
        print("    - Target: 100+ trades")
        print()
        print("  PHASE 3: Scale to $100")
        print("    - Increase to target capital")
        print("    - Monitor drawdowns closely")
        print("    - Evaluate quarterly performance")
        print()
        print(f"  TRADEABLE STRATEGIES: {', '.join(tradeable)}")
        
    elif verdict == "MARGINAL":
        print("  ⚠️  PORTFOLIO IS MARGINAL")
        print()
        print("  RECOMMENDED ACTIONS:")
        print("    1. Extended paper trading (8+ weeks)")
        print("    2. Analyze which strategies drag performance")
        print("    3. Consider running only viable strategies")
        print()
        if tradeable:
            print(f"  CONSIDER DEPLOYING ONLY: {', '.join(tradeable)}")
        
    else:
        print("  ⛔ PORTFOLIO IS NOT VIABLE")
        print()
        print("  RECOMMENDED ACTIONS:")
        print("    1. Analyze why strategies underperform")
        print("    2. Consider different strategy types")
        print("    3. Test on different instruments")
        print("    4. Accept that retail FX may lack accessible edge")
        print()
        print("  DO NOT DEPLOY WITH REAL CAPITAL")
    
    print()
    print("=" * 80)
    
    return verdict, tradeable


if __name__ == "__main__":
    verdict, tradeable = main()
