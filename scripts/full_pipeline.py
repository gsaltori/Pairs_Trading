"""
Full Pipeline Script.

Complete workflow: Download → Screen → Backtest → Optimize

Usage:
    python scripts/full_pipeline.py --days 365
    python scripts/full_pipeline.py --days 730 --skip-download
"""

import sys
from pathlib import Path
from datetime import datetime
import argparse
import logging
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import Settings
from config.broker_config import MT5Config
from src.data.broker_client import MT5Client, Timeframe as MT5Timeframe
from src.analysis.pair_screener import PairScreener
from src.backtest.backtest_engine import BacktestEngine
from src.optimization.optimizer import WalkForwardOptimizer


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Symbols for screening
SCREENING_SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
    "AUDUSD", "NZDUSD", "USDCAD",
    "EURJPY", "GBPJPY", "AUDJPY",
    "EURGBP", "EURAUD", "EURCHF",
    "GBPAUD", "AUDNZD"
]


def download_data(symbols, days, timeframe, output_dir, client):
    """Download data for all symbols."""
    print("\n" + "=" * 60)
    print("STEP 1: DOWNLOADING DATA")
    print("=" * 60)
    
    bars_per_day = {'M1': 1440, 'M5': 288, 'M15': 96, 'M30': 48, 'H1': 24, 'H4': 6, 'D1': 1}
    count = days * bars_per_day.get(timeframe, 24)
    
    success = 0
    failed = 0
    
    for symbol in symbols:
        try:
            data = client.get_candles(symbol, MT5Timeframe.from_string(timeframe), count)
            
            if data.empty:
                print(f"  ✗ {symbol}: No data")
                failed += 1
                continue
            
            filepath = output_dir / f"{symbol}_{timeframe}_{days}d.parquet"
            data.to_parquet(filepath)
            print(f"  ✓ {symbol}: {len(data)} bars")
            success += 1
            
        except Exception as e:
            print(f"  ✗ {symbol}: {e}")
            failed += 1
    
    print(f"\nDownloaded: {success} | Failed: {failed}")
    return success > 0


def load_local_data(symbol, data_dir, timeframe):
    """Load data from local file."""
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
    
    return df['close']


def screen_pairs(data_dir, timeframe, top_n=10):
    """Screen pairs from local data."""
    print("\n" + "=" * 60)
    print("STEP 2: SCREENING PAIRS")
    print("=" * 60)
    
    price_data = {}
    
    for symbol in SCREENING_SYMBOLS:
        data = load_local_data(symbol, data_dir, timeframe)
        if data is not None and len(data) >= 500:
            price_data[symbol] = data
    
    print(f"Loaded {len(price_data)} symbols")
    
    if len(price_data) < 2:
        return None
    
    screener = PairScreener(min_correlation=0.70, max_half_life=50)
    result = screener.screen_pairs(price_data, top_n=top_n)
    
    print(f"\nAnalyzed {result.pairs_analyzed} pairs")
    print(f"Passed criteria: {result.pairs_passed} pairs")
    
    if result.top_pairs:
        print("\nTop 5 pairs:")
        for i, p in enumerate(result.top_pairs[:5], 1):
            signal = ""
            if p.current_zscore < -2:
                signal = " 🟢 LONG"
            elif p.current_zscore > 2:
                signal = " 🔴 SHORT"
            print(f"  {i}. {p.pair[0]}/{p.pair[1]}: Score={p.total_score:.1f}, Z={p.current_zscore:+.2f}{signal}")
    
    return result


def backtest_top_pairs(screening_result, price_data, settings, top_n=5):
    """Backtest top pairs."""
    print("\n" + "=" * 60)
    print("STEP 3: BACKTESTING TOP PAIRS")
    print("=" * 60)
    
    results = []
    engine = BacktestEngine(settings)
    
    for p in screening_result.top_pairs[:top_n]:
        print(f"\n  Backtesting {p.pair[0]}/{p.pair[1]}...")
        
        result = engine.run_backtest(
            p.pair,
            price_data[p.pair[0]],
            price_data[p.pair[1]]
        )
        
        if result:
            results.append({
                'pair': p.pair,
                'score': p.total_score,
                'sharpe': result.sharpe_ratio,
                'return': result.total_return,
                'trades': result.total_trades,
                'win_rate': result.win_rate,
                'max_dd': result.max_drawdown,
                'half_life': p.half_life,
                'current_zscore': p.current_zscore
            })
            
            status = "✓" if result.sharpe_ratio > 0 else "✗"
            print(f"    {status} Sharpe: {result.sharpe_ratio:.2f} | Return: {result.total_return:.1%} | Trades: {result.total_trades}")
    
    return results


def optimize_best_pair(pair, price_data, settings):
    """Run walk-forward optimization on best pair."""
    print("\n" + "=" * 60)
    print(f"STEP 4: OPTIMIZING {pair[0]}/{pair[1]}")
    print("=" * 60)
    
    optimizer = WalkForwardOptimizer(settings)
    optimizer.set_fast_mode()  # Use fast mode for pipeline
    
    result = optimizer.optimize(
        pair,
        price_data[pair[0]],
        price_data[pair[1]]
    )
    
    print(f"\nOptimization Results:")
    print(f"  Periods analyzed: {len(result.periods)}")
    print(f"  IS Sharpe: {result.is_sharpe:.2f}")
    print(f"  OOS Sharpe: {result.oos_sharpe:.2f}")
    print(f"  Efficiency: {result.efficiency_ratio:.1%}")
    print(f"  Robustness: {result.robustness_score:.1%}")
    
    if result.best_params:
        print(f"\nBest Parameters:")
        print(f"  Entry Z: {result.best_params.entry_zscore}")
        print(f"  Exit Z: {result.best_params.exit_zscore}")
        print(f"  Regression window: {result.best_params.regression_window}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Full pairs trading pipeline')
    
    parser.add_argument('--days', type=int, default=365, help='Days of data')
    parser.add_argument('--timeframe', type=str, default='H1', help='Timeframe')
    parser.add_argument('--skip-download', action='store_true', help='Skip download step')
    parser.add_argument('--skip-optimize', action='store_true', help='Skip optimization step')
    parser.add_argument('--top', type=int, default=5, help='Top N pairs to backtest')
    
    args = parser.parse_args()
    
    data_dir = Path("data/historical")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("PAIRS TRADING - FULL PIPELINE")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Days: {args.days}")
    print(f"  Timeframe: {args.timeframe}")
    print(f"  Skip download: {args.skip_download}")
    
    # Step 1: Download
    if not args.skip_download:
        try:
            config = MT5Config.from_env()
            client = MT5Client(config)
            
            if not client.connect():
                print("\nERROR: Could not connect to MT5")
                print("Use --skip-download if data already exists")
                return 1
            
            download_data(SCREENING_SYMBOLS, args.days, args.timeframe, data_dir, client)
            client.disconnect()
            
        except Exception as e:
            print(f"\nDownload failed: {e}")
            print("Continuing with existing data...")
    
    # Load price data
    price_data = {}
    for symbol in SCREENING_SYMBOLS:
        data = load_local_data(symbol, data_dir, args.timeframe)
        if data is not None and len(data) >= 500:
            price_data[symbol] = data
    
    if len(price_data) < 2:
        print("\nERROR: Insufficient data")
        return 1
    
    # Step 2: Screen
    screening_result = screen_pairs(data_dir, args.timeframe, top_n=args.top)
    
    if not screening_result or not screening_result.top_pairs:
        print("\nNo viable pairs found")
        return 1
    
    # Step 3: Backtest
    settings = Settings()
    backtest_results = backtest_top_pairs(screening_result, price_data, settings, args.top)
    
    if not backtest_results:
        print("\nAll backtests failed")
        return 1
    
    # Find best pair
    profitable = [r for r in backtest_results if r['sharpe'] > 0]
    
    if profitable:
        best = max(profitable, key=lambda x: x['sharpe'])
    else:
        best = max(backtest_results, key=lambda x: x['sharpe'])
    
    # Step 4: Optimize best pair
    if not args.skip_optimize:
        opt_result = optimize_best_pair(best['pair'], price_data, settings)
    
    # Final Summary
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    
    print(f"\nBest Pair: {best['pair'][0]}/{best['pair'][1]}")
    print(f"  Screening Score: {best['score']:.1f}")
    print(f"  Backtest Sharpe: {best['sharpe']:.2f}")
    print(f"  Backtest Return: {best['return']:.1%}")
    print(f"  Half-life: {best['half_life']:.1f} bars")
    print(f"  Current Z-score: {best['current_zscore']:+.2f}")
    
    # Trading signal
    if abs(best['current_zscore']) >= 2.0:
        direction = "LONG spread" if best['current_zscore'] < 0 else "SHORT spread"
        print(f"\n🚨 ACTIVE SIGNAL: {direction}")
        print(f"   Entry Z: {best['current_zscore']:+.2f}")
    else:
        print(f"\n⏳ No active signal. Wait for |Z| >= 2.0")
    
    # All backtest results
    print("\n" + "-" * 60)
    print("ALL BACKTEST RESULTS")
    print("-" * 60)
    print(f"\n{'Pair':<15} {'Score':>8} {'Sharpe':>8} {'Return':>10} {'Trades':>8}")
    print("-" * 55)
    
    for r in sorted(backtest_results, key=lambda x: x['sharpe'], reverse=True):
        pair_str = f"{r['pair'][0]}/{r['pair'][1]}"
        print(f"{pair_str:<15} {r['score']:>8.1f} {r['sharpe']:>8.2f} {r['return']:>9.1%} {r['trades']:>8}")
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
