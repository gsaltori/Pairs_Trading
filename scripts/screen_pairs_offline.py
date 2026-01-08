"""
Offline Pair Screening Script.

Analyzes all possible pairs from local data to find the best trading candidates.

Usage:
    python scripts/screen_pairs_offline.py
    python scripts/screen_pairs_offline.py --symbols EURUSD,GBPUSD,USDJPY,AUDUSD
    python scripts/screen_pairs_offline.py --top 20 --backtest
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

from config.settings import Settings
from src.analysis.pair_screener import PairScreener, ScreeningResult
from src.backtest.backtest_engine import BacktestEngine


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Default symbols for screening (common forex pairs)
DEFAULT_SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
    "AUDUSD", "NZDUSD", "USDCAD",
    "EURJPY", "GBPJPY", "AUDJPY",
    "EURGBP", "EURAUD", "EURCHF",
    "GBPAUD", "AUDNZD"
]


def load_local_data(
    symbol: str,
    data_dir: Path,
    timeframe: str = "H1"
) -> pd.Series:
    """Load data from local parquet file."""
    pattern = f"{symbol}_{timeframe}_*.parquet"
    files = list(data_dir.glob(pattern))
    
    if not files:
        pattern = f"{symbol}_{timeframe}.parquet"
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


def run_quick_backtest(pair, price_a, price_b, settings):
    """Run a quick backtest to validate pair."""
    engine = BacktestEngine(settings)
    result = engine.run_backtest(pair, price_a, price_b)
    return result


def convert_to_json_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, (np.bool_, np.bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(i) for i in obj]
    return obj


def main():
    parser = argparse.ArgumentParser(
        description='Screen pairs from local data to find best trading candidates'
    )
    
    parser.add_argument(
        '--symbols', type=str,
        help='Comma-separated list of symbols (default: major forex pairs)'
    )
    parser.add_argument(
        '--data-dir', type=str, default='data/historical',
        help='Directory containing historical data'
    )
    parser.add_argument(
        '--timeframe', type=str, default='H1',
        help='Timeframe (default: H1)'
    )
    parser.add_argument(
        '--top', type=int, default=10,
        help='Number of top pairs to show (default: 10)'
    )
    parser.add_argument(
        '--min-corr', type=float, default=0.60,
        help='Minimum price correlation (default: 0.60)'
    )
    parser.add_argument(
        '--max-half-life', type=int, default=500,
        help='Maximum half-life in bars (default: 500)'
    )
    parser.add_argument(
        '--backtest', action='store_true',
        help='Run quick backtest on top pairs'
    )
    parser.add_argument(
        '--save', action='store_true',
        help='Save results to file'
    )
    
    args = parser.parse_args()
    
    # Parse symbols
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(',')]
    else:
        symbols = DEFAULT_SYMBOLS
    
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        print("\nPlease download data first:")
        print("  python scripts/download_data.py --forex --days 730")
        return 1
    
    print("=" * 70)
    print("PAIR SCREENING - FIND BEST TRADING PAIRS")
    print("=" * 70)
    print(f"\nSymbols to analyze: {len(symbols)}")
    print(f"Data directory: {data_dir}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Min correlation: {args.min_corr}")
    print(f"Max half-life: {args.max_half_life} bars")
    
    # Load data
    print("\n" + "-" * 70)
    print("LOADING DATA...")
    print("-" * 70)
    
    price_data = {}
    loaded = 0
    failed = 0
    
    for symbol in symbols:
        data = load_local_data(symbol, data_dir, args.timeframe)
        if data is not None and len(data) >= 500:
            price_data[symbol] = data
            loaded += 1
            print(f"  ✓ {symbol}: {len(data)} bars")
        else:
            failed += 1
            print(f"  ✗ {symbol}: No data or insufficient bars")
    
    print(f"\nLoaded: {loaded} symbols | Failed: {failed} symbols")
    
    if loaded < 2:
        print("\nERROR: Need at least 2 symbols with data")
        print("\nDownload data first:")
        print("  python scripts/download_data.py --forex --days 730")
        return 1
    
    # Calculate number of pairs
    n_pairs = loaded * (loaded - 1) // 2
    print(f"Pairs to analyze: {n_pairs}")
    
    # Run screening
    print("\n" + "-" * 70)
    print("SCREENING PAIRS...")
    print("-" * 70)
    
    screener = PairScreener(
        min_correlation=args.min_corr,
        max_half_life=args.max_half_life
    )
    
    result = screener.screen_pairs(price_data, top_n=args.top)
    
    # Display results
    print("\n" + screener.generate_report(result))
    
    # Quick backtest on top pairs
    if args.backtest and result.top_pairs:
        print("\n" + "-" * 70)
        print("QUICK BACKTEST ON TOP PAIRS")
        print("-" * 70)
        
        settings = Settings()
        backtest_results = []
        
        # Backtest top 5 tradeable or highest scored
        pairs_to_test = result.top_pairs[:5]
        
        for p in pairs_to_test:
            print(f"\n  Backtesting {p.pair[0]}/{p.pair[1]}...")
            
            bt_result = run_quick_backtest(
                p.pair,
                price_data[p.pair[0]],
                price_data[p.pair[1]],
                settings
            )
            
            if bt_result:
                backtest_results.append({
                    'pair': p.pair,
                    'score': p.total_score,
                    'cointegrated': bool(p.is_cointegrated),
                    'sharpe': bt_result.sharpe_ratio,
                    'return': bt_result.total_return,
                    'trades': bt_result.total_trades,
                    'win_rate': bt_result.win_rate,
                    'max_dd': bt_result.max_drawdown,
                    'half_life': p.half_life,
                    'zscore': p.current_zscore
                })
                
                status = "✓" if bt_result.sharpe_ratio > 0 else "✗"
                print(f"    {status} Sharpe: {bt_result.sharpe_ratio:.2f} | Return: {bt_result.total_return:.1%} | Trades: {bt_result.total_trades} | Win: {bt_result.win_rate:.1%}")
            else:
                print(f"    ✗ Backtest failed")
        
        # Summary table
        if backtest_results:
            print("\n" + "-" * 70)
            print("BACKTEST SUMMARY")
            print("-" * 70)
            print(f"\n{'Pair':<15} {'Score':>7} {'Coint':>6} {'Sharpe':>8} {'Return':>9} {'Trades':>7} {'Win%':>7}")
            print("-" * 70)
            
            for r in sorted(backtest_results, key=lambda x: x['sharpe'], reverse=True):
                pair_str = f"{r['pair'][0]}/{r['pair'][1]}"
                coint = "Yes" if r['cointegrated'] else "No"
                print(f"{pair_str:<15} {r['score']:>7.1f} {coint:>6} {r['sharpe']:>8.2f} {r['return']:>8.1%} {r['trades']:>7} {r['win_rate']:>6.1%}")
            
            # Best pair recommendation
            profitable = [r for r in backtest_results if r['sharpe'] > 0]
            if profitable:
                best = max(profitable, key=lambda x: x['sharpe'])
                print(f"\n✓ BEST PERFORMING: {best['pair'][0]}/{best['pair'][1]}")
                print(f"  Sharpe: {best['sharpe']:.2f} | Cointegrated: {'Yes' if best['cointegrated'] else 'No'}")
    
    # Save results
    if args.save:
        results_dir = Path("results/screening")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to JSON
        filepath = results_dir / f"screening_{datetime.now():%Y%m%d_%H%M%S}.json"
        
        data = {
            'timestamp': result.timestamp.isoformat(),
            'symbols_analyzed': int(result.symbols_analyzed),
            'pairs_analyzed': int(result.pairs_analyzed),
            'pairs_passed': int(result.pairs_passed),
            'avg_correlation': float(result.avg_correlation),
            'cointegration_rate': float(result.cointegration_rate),
            'top_pairs': [
                {
                    'pair': list(p.pair),
                    'total_score': float(p.total_score),
                    'price_correlation': float(p.price_correlation),
                    'returns_correlation': float(p.returns_correlation),
                    'is_cointegrated': bool(p.is_cointegrated),
                    'coint_pvalue': float(p.coint_pvalue),
                    'adf_pvalue': float(p.adf_pvalue),
                    'half_life': float(p.half_life),
                    'hurst': float(p.hurst_exponent),
                    'current_zscore': float(p.current_zscore),
                    'hedge_ratio': float(p.hedge_ratio),
                    'is_tradeable': bool(p.is_tradeable),
                    'scores': {
                        'correlation': float(p.correlation_score),
                        'cointegration': float(p.cointegration_score),
                        'mean_reversion': float(p.mean_reversion_score),
                        'tradability': float(p.tradability_score)
                    }
                }
                for p in result.top_pairs
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n✓ Results saved to: {filepath}")
    
    # Trading suggestions
    print("\n" + "=" * 70)
    print("TRADING SUGGESTIONS")
    print("=" * 70)
    
    # Best cointegrated pairs
    cointegrated = [p for p in result.top_pairs if p.is_cointegrated]
    if cointegrated:
        print(f"\nBest COINTEGRATED pairs (statistically robust):")
        for p in cointegrated[:3]:
            print(f"  • {p.pair[0]}/{p.pair[1]}: Score={p.total_score:.1f}, HL={p.half_life:.0f}, Z={p.current_zscore:+.2f}")
    
    # Pairs with active signals
    signals = [p for p in result.top_pairs if abs(p.current_zscore) >= 2.0]
    
    if signals:
        print("\nPairs with ACTIVE ENTRY SIGNALS:")
        for p in signals:
            direction = "LONG spread (buy A, sell B)" if p.current_zscore < 0 else "SHORT spread (sell A, buy B)"
            coint_str = " [COINTEGRATED]" if p.is_cointegrated else ""
            print(f"\n  → {p.pair[0]}/{p.pair[1]}: {direction}{coint_str}")
            print(f"    Z-score: {p.current_zscore:+.2f} | Half-life: {p.half_life:.0f} bars")
            print(f"    Hedge ratio: {p.hedge_ratio:.4f}")
    else:
        print("\nNo pairs with active entry signals (|Z| >= 2.0)")
    
    # Approaching signals
    approaching = [p for p in result.top_pairs if 1.5 <= abs(p.current_zscore) < 2.0]
    if approaching:
        print("\nPairs APPROACHING entry threshold:")
        for p in approaching[:5]:
            direction = "LONG" if p.current_zscore < 0 else "SHORT"
            print(f"  • {p.pair[0]}/{p.pair[1]}: Z={p.current_zscore:+.2f} ({direction} when |Z| >= 2.0)")
    
    print("\n" + "=" * 70)
    print("SCREENING COMPLETE")
    print("=" * 70)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
