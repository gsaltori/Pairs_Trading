"""
Conditional StatArb Screening Script.

This script runs the conditional statistical arbitrage system which:
1. Only trades when regime is favorable
2. Uses dynamic cointegration validation
3. Implements pair dormancy states
4. Knows when NOT to trade

Zero trades is a VALID outcome when conditions aren't met.

Usage:
    python scripts/conditional_screen.py
    python scripts/conditional_screen.py --timeframe H4
    python scripts/conditional_screen.py --save
"""

import sys
from pathlib import Path
from datetime import datetime
import argparse
import logging
import pandas as pd
import numpy as np
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.strict_selector import VALID_PAIR_COMBINATIONS, is_valid_forex_combination
from src.strategy.conditional_statarb import (
    PairState, MarketRegime,
    MarketRegimeDetector, DynamicCointegrationValidator, SpreadHealthMonitor
)
from src.strategy.conditional_manager import (
    ConditionalSignalGenerator, ConditionalPairManager, ConditionalStatArbSystem
)


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Complete Forex Universe
FOREX_UNIVERSE = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "NZDUSD", "USDCAD",
    "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY",
    "EURGBP", "EURAUD", "EURNZD", "EURCHF", "EURCAD",
    "GBPAUD", "GBPNZD", "GBPCAD", "GBPCHF",
    "AUDNZD", "AUDCAD", "AUDCHF", "NZDCAD", "NZDCHF", "CADCHF"
]


def load_ohlc_data(symbol: str, data_dir: Path, timeframe: str = "H1") -> pd.DataFrame:
    """Load OHLC data from local parquet file."""
    pattern = f"{symbol}_{timeframe}_*.parquet"
    files = list(data_dir.glob(pattern))
    
    if not files:
        return None
    
    filepath = max(files, key=lambda f: f.stat().st_mtime)
    df = pd.read_parquet(filepath)
    
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'time' in df.columns:
            df.set_index('time', inplace=True)
        df.index = pd.to_datetime(df.index)
    
    # Ensure column names
    df.columns = df.columns.str.lower()
    
    return df


def get_valid_pairs(symbols: list) -> list:
    """Get all valid pair combinations from available symbols."""
    valid_pairs = []
    
    for i, sym_a in enumerate(symbols):
        for sym_b in symbols[i+1:]:
            if is_valid_forex_combination(sym_a, sym_b):
                valid_pairs.append((sym_a, sym_b))
    
    return valid_pairs


def main():
    parser = argparse.ArgumentParser(
        description='Conditional StatArb screening with regime awareness'
    )
    
    parser.add_argument(
        '--symbols', type=str,
        help='Comma-separated symbols'
    )
    parser.add_argument(
        '--data-dir', type=str, default='data/historical',
        help='Data directory'
    )
    parser.add_argument(
        '--timeframe', type=str, default='H1',
        choices=['M15', 'M30', 'H1', 'H4', 'D1'],
        help='Timeframe'
    )
    parser.add_argument(
        '--half-life-max', type=int, default=60,
        help='Maximum half-life in bars'
    )
    parser.add_argument(
        '--save', action='store_true',
        help='Save results'
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        return 1
    
    # Header
    print("=" * 80)
    print("CONDITIONAL STATISTICAL ARBITRAGE SYSTEM")
    print("=" * 80)
    print("\n📊 This system implements REGIME-AWARE pair trading.")
    print("   Pairs can be valid but DORMANT when regime is unfavorable.")
    print("   Zero trades is a VALID outcome - the system knows when NOT to trade.")
    
    # Load data
    print(f"\n" + "-" * 80)
    print("LOADING DATA")
    print("-" * 80)
    
    symbols = args.symbols.split(',') if args.symbols else FOREX_UNIVERSE
    symbols = [s.strip().upper() for s in symbols]
    
    ohlc_data = {}
    price_data = {}
    
    for symbol in symbols:
        df = load_ohlc_data(symbol, data_dir, args.timeframe)
        if df is not None and len(df) >= 500:
            ohlc_data[symbol] = df
            price_data[symbol] = df['close']
            print(f"  ✓ {symbol}: {len(df)} bars")
        else:
            print(f"  ✗ {symbol}: No data")
    
    print(f"\nLoaded: {len(price_data)} symbols")
    
    # Get valid pairs
    valid_pairs = get_valid_pairs(list(price_data.keys()))
    print(f"Valid pair combinations: {len(valid_pairs)}")
    
    if not valid_pairs:
        print("\nNo valid pair combinations found.")
        return 1
    
    # Initialize system
    print(f"\n" + "-" * 80)
    print("INITIALIZING CONDITIONAL STATARB SYSTEM")
    print("-" * 80)
    
    # Configure based on timeframe
    half_life_config = {
        'M15': 40, 'M30': 50, 'H1': 60, 'H4': 120, 'D1': 40
    }
    
    system = ConditionalStatArbSystem(
        half_life_max=half_life_config.get(args.timeframe, 60),
        half_life_min=5,
        hurst_max=0.55,
        entry_zscore=2.0,
        exit_zscore=0.3,
        stop_zscore=3.5,
        coint_pvalue=0.05,
        trend_threshold=25.0
    )
    
    print(f"  Half-life max: {half_life_config.get(args.timeframe, 60)} bars")
    print(f"  Hurst max: 0.55")
    print(f"  Entry Z: ±2.0")
    print(f"  Trend threshold (ADX): 25")
    
    # Analyze pairs
    print(f"\n" + "-" * 80)
    print("ANALYZING PAIRS")
    print("-" * 80)
    
    pairs_data = {}
    
    for pair in valid_pairs:
        sym_a, sym_b = pair
        
        if sym_a in price_data and sym_b in price_data:
            pairs_data[pair] = (
                price_data[sym_a],
                price_data[sym_b],
                ohlc_data.get(sym_a)  # OHLC for regime detection
            )
    
    print(f"\nAnalyzing {len(pairs_data)} valid pairs...")
    
    # Update system
    timestamp = datetime.now()
    status = system.update(pairs_data, timestamp)
    
    # Generate report
    print("\n" + system.generate_status_report())
    
    # Detailed analysis for each pair
    print(f"\n" + "-" * 80)
    print("DETAILED PAIR ANALYSIS")
    print("-" * 80)
    
    for pair, pair_status in system.pair_manager.pair_states.items():
        state_icon = {
            PairState.ACTIVE: "✅",
            PairState.DORMANT: "⏸️",
            PairState.INVALIDATED: "❌",
            PairState.WARMING_UP: "⏳"
        }.get(pair_status.state, "?")
        
        print(f"\n{state_icon} {pair[0]}/{pair[1]} - {pair_status.state.value.upper()}")
        
        if pair_status.regime_analysis:
            ra = pair_status.regime_analysis
            
            print(f"    Market Regime: {ra.market_regime.value}")
            
            if ra.volatility:
                print(f"    Volatility: {ra.volatility.vol_regime} (ATR percentile: {ra.volatility.atr_percentile:.0f})")
            
            if ra.trend:
                print(f"    Trend: ADX={ra.trend.adx:.1f} ({ra.trend.trend_strength})")
            
            print(f"    Session: {ra.session.value}")
            
            if ra.spread_health:
                sh = ra.spread_health
                print(f"    Spread Health: {sh.health_score:.0f}/100")
                print(f"      Stationary: {'✓' if sh.is_stationary else '✗'} (ADF p={sh.adf_pvalue:.3f})")
                print(f"      Half-life: {sh.half_life:.0f} bars {'✓' if sh.half_life_stable else '✗'}")
                print(f"      Hurst: {sh.hurst_exponent:.3f} {'✓' if sh.is_mean_reverting else '✗'}")
                print(f"      Hedge stable: {'✓' if sh.hedge_ratio_stable else '✗'}")
            
            if ra.cointegration:
                ci = ra.cointegration
                print(f"    Cointegration: {ci.confidence} confidence")
                print(f"      Current: {'✓' if ci.is_cointegrated_current else '✗'} (p={ci.eg_pvalue_current:.4f})")
                print(f"      Consistency: {ci.cointegration_consistency:.0%}")
                print(f"      Breakdowns: {ci.breakdown_frequency:.1%}")
        
        if pair_status.state_reasons:
            print(f"    Reasons: {', '.join(pair_status.state_reasons[:3])}")
        
        print(f"    Z-Score: {pair_status.current_zscore:+.2f}")
        
        if pair_status.signal:
            print(f"    🚨 SIGNAL: {pair_status.signal.upper()} (strength: {pair_status.signal_strength:.0%})")
    
    # Trading decision summary
    print(f"\n" + "=" * 80)
    print("TRADING DECISION")
    print("=" * 80)
    
    should_trade, reasons = system.should_system_trade()
    
    if should_trade:
        decisions = system.get_trading_decisions()
        print(f"\n✅ SYSTEM IS READY TO TRADE")
        print(f"\nTrading Signals:")
        
        for d in decisions:
            direction = "LONG" if d['action'] == 'long' else 'SHORT'
            print(f"\n  → {d['pair'][0]}/{d['pair'][1]}: {direction} SPREAD")
            print(f"    Z-Score: {d['zscore']:+.2f}")
            print(f"    Strength: {d['strength']:.0%}")
            print(f"    Regime: {d['regime']}")
    else:
        print(f"\n⏸️ SYSTEM IS WAITING - No trades")
        print(f"\nReasons:")
        for reason in reasons:
            print(f"  • {reason}")
        
        # Show what would need to change
        dormant = system.pair_manager.get_dormant_pairs()
        if dormant:
            print(f"\n📋 {len(dormant)} pair(s) are DORMANT (valid but regime unfavorable):")
            for s in dormant[:3]:
                print(f"  • {s.pair[0]}/{s.pair[1]}: waiting for {', '.join(s.state_reasons[:2])}")
    
    # Summary stats
    summary = status['state_summary']
    total = sum(summary.values())
    
    print(f"\n" + "-" * 80)
    print("SUMMARY")
    print("-" * 80)
    print(f"\n  Total pairs analyzed: {total}")
    print(f"  Active:       {summary.get('active', 0)} ({summary.get('active', 0)/total*100:.0f}%)")
    print(f"  Dormant:      {summary.get('dormant', 0)} ({summary.get('dormant', 0)/total*100:.0f}%)")
    print(f"  Invalidated:  {summary.get('invalidated', 0)} ({summary.get('invalidated', 0)/total*100:.0f}%)")
    
    rejection_rate = (summary.get('invalidated', 0) + summary.get('dormant', 0)) / total * 100
    print(f"\n  Rejection/Dormancy rate: {rejection_rate:.0f}%")
    print(f"  (This is EXPECTED - the system knows when NOT to trade)")
    
    # Save results
    if args.save:
        results_dir = Path("results/screening")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = results_dir / f"conditional_{args.timeframe}_{datetime.now():%Y%m%d_%H%M%S}.json"
        
        save_data = {
            "timestamp": status['timestamp'].isoformat(),
            "timeframe": args.timeframe,
            "is_active": status['is_active'],
            "state_summary": status['state_summary'],
            "signals": status['signals'],
            "pairs": [
                {
                    "pair": list(s.pair),
                    "state": s.state.value,
                    "zscore": float(s.current_zscore),
                    "signal": s.signal,
                    "reasons": s.state_reasons
                }
                for s in system.pair_manager.pair_states.values()
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"\n✓ Results saved to: {filepath}")
    
    print(f"\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
