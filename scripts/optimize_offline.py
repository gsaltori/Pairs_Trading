"""
Offline Walk-Forward Optimization Script.

Run optimization using locally saved historical data (no MT5 connection required).

Usage:
    python scripts/optimize_offline.py --pair EURUSD,GBPUSD
    python scripts/optimize_offline.py --pair EURUSD,GBPUSD --fast
    python scripts/optimize_offline.py --pair EURUSD,GBPUSD --data-dir data/historical
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

from config.settings import Settings, Timeframe
from src.optimization.optimizer import WalkForwardOptimizer


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
        raise FileNotFoundError(f"No data file found for {symbol} {timeframe} in {data_dir}")
    
    filepath = max(files, key=lambda f: f.stat().st_mtime)
    
    logger.info(f"Loading {filepath}")
    
    df = pd.read_parquet(filepath)
    
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'time' in df.columns:
            df.set_index('time', inplace=True)
        df.index = pd.to_datetime(df.index)
    
    return df['close']


def main():
    parser = argparse.ArgumentParser(
        description='Run offline walk-forward optimization'
    )
    
    parser.add_argument(
        '--pair', type=str, required=True,
        help='Pair to optimize (e.g., EURUSD,GBPUSD)'
    )
    parser.add_argument(
        '--data-dir', type=str, default='data/historical',
        help='Directory containing historical data (default: data/historical)'
    )
    parser.add_argument(
        '--timeframe', type=str, default='H1',
        help='Timeframe: M15, M30, H1, H4, D1 (default: H1)'
    )
    parser.add_argument(
        '--is-bars', type=int, default=504,
        help='In-sample period in bars (default: 504 = ~3 weeks H1)'
    )
    parser.add_argument(
        '--oos-bars', type=int, default=168,
        help='Out-of-sample period in bars (default: 168 = ~1 week H1)'
    )
    parser.add_argument(
        '--fast', action='store_true',
        help='Use fast mode with reduced parameter grid'
    )
    parser.add_argument(
        '--max-periods', type=int, default=50,
        help='Maximum number of WF periods to analyze (default: 50)'
    )
    
    args = parser.parse_args()
    
    # Parse pair
    symbols = args.pair.split(',')
    if len(symbols) != 2:
        print("ERROR: Please specify exactly 2 symbols separated by comma")
        return 1
    
    pair = (symbols[0].strip(), symbols[1].strip())
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        print("\nPlease download data first:")
        print(f"  python scripts/download_data.py --symbols {args.pair} --days 730")
        return 1
    
    print("="*60)
    print("OFFLINE WALK-FORWARD OPTIMIZATION")
    print("="*60)
    print(f"\nPair: {pair[0]}/{pair[1]}")
    print(f"Data directory: {data_dir}")
    print(f"Timeframe: {args.timeframe}")
    print(f"IS period: {args.is_bars} bars")
    print(f"OOS period: {args.oos_bars} bars")
    print(f"Fast mode: {'Yes' if args.fast else 'No'}")
    
    # Load data
    print("\n" + "-"*60)
    print("LOADING DATA...")
    print("-"*60)
    
    try:
        price_a = load_local_data(pair[0], data_dir, args.timeframe)
        print(f"✓ {pair[0]}: {len(price_a)} bars")
        
        price_b = load_local_data(pair[1], data_dir, args.timeframe)
        print(f"✓ {pair[1]}: {len(price_b)} bars")
        
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nPlease download data first:")
        print(f"  python scripts/download_data.py --symbols {args.pair} --days 730")
        return 1
    
    # Align data
    common_idx = price_a.index.intersection(price_b.index)
    price_a = price_a.loc[common_idx]
    price_b = price_b.loc[common_idx]
    
    print(f"\n✓ Aligned data: {len(common_idx)} bars")
    print(f"  Period: {common_idx[0]} to {common_idx[-1]}")
    
    # Calculate periods
    total_periods = (len(common_idx) - args.is_bars) // args.oos_bars
    actual_periods = min(total_periods, args.max_periods)
    print(f"  Possible WF periods: {total_periods}")
    print(f"  Will analyze: {actual_periods} periods")
    
    if len(common_idx) < args.is_bars + args.oos_bars:
        print(f"\nERROR: Insufficient data for optimization")
        print(f"  Need: {args.is_bars + args.oos_bars} bars")
        print(f"  Have: {len(common_idx)} bars")
        return 1
    
    # Run optimization
    print("\n" + "-"*60)
    print("RUNNING WALK-FORWARD OPTIMIZATION...")
    if args.fast:
        print("(Fast mode - reduced parameter grid)")
    print("This may take several minutes...")
    print("-"*60)
    
    settings = Settings()
    settings.optimization.in_sample_bars = args.is_bars
    settings.optimization.out_of_sample_bars = args.oos_bars
    settings.optimization.max_iterations = args.max_periods
    
    optimizer = WalkForwardOptimizer(settings)
    
    if args.fast:
        optimizer.set_fast_mode()
    
    result = optimizer.optimize(pair, price_a, price_b)
    
    # Display results
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    
    print(f"\nPeriods analyzed: {len(result.periods)}")
    print(f"Total OOS trades: {result.total_trades}")
    
    print(f"\nPerformance:")
    print(f"  In-Sample Sharpe:     {result.is_sharpe:>8.2f}")
    print(f"  Out-of-Sample Sharpe: {result.oos_sharpe:>8.2f}")
    print(f"  Efficiency Ratio:     {result.efficiency_ratio:>8.2%}")
    print(f"  Robustness Score:     {result.robustness_score:>8.1%}")
    
    # Interpretation
    print("\nInterpretation:")
    if result.efficiency_ratio >= 0.5:
        print("  ✓ Good parameter stability (efficiency ≥ 50%)")
    elif result.efficiency_ratio >= 0.3:
        print("  ⚠ Moderate parameter stability")
    else:
        print("  ✗ Low efficiency ratio - potential overfitting")
    
    if result.robustness_score >= 0.6:
        print(f"  ✓ {result.robustness_score:.0%} of periods profitable OOS")
    elif result.robustness_score >= 0.4:
        print(f"  ⚠ {result.robustness_score:.0%} of periods profitable OOS")
    else:
        print(f"  ✗ Only {result.robustness_score:.0%} of periods profitable OOS")
    
    if result.best_params:
        print(f"\nBest Parameters:")
        print(f"  Entry Z-score:      {result.best_params.entry_zscore}")
        print(f"  Exit Z-score:       {result.best_params.exit_zscore}")
        print(f"  Stop-loss Z-score:  {result.best_params.stop_loss_zscore}")
        print(f"  Regression window:  {result.best_params.regression_window}")
        print(f"  Z-score window:     {result.best_params.zscore_window}")
        print(f"  Min correlation:    {result.best_params.min_correlation}")
    
    # Period breakdown (show first 10 and last 5)
    print("\n" + "-"*60)
    print("PERIOD BREAKDOWN")
    print("-"*60)
    
    show_periods = result.periods[:10] + (result.periods[-5:] if len(result.periods) > 15 else [])
    shown_indices = set()
    
    for p in result.periods[:10]:
        shown_indices.add(p.period_index)
        status = "✓" if p.oos_sharpe > 0 else "✗"
        print(f"\n{status} Period {p.period_index + 1}:")
        print(f"  IS:  {p.is_start.strftime('%Y-%m-%d')} to {p.is_end.strftime('%Y-%m-%d')}")
        print(f"  OOS: {p.oos_start.strftime('%Y-%m-%d')} to {p.oos_end.strftime('%Y-%m-%d')}")
        print(f"  IS Sharpe: {p.is_sharpe:.2f} | OOS Sharpe: {p.oos_sharpe:.2f} | Trades: {p.oos_trades}")
    
    if len(result.periods) > 15:
        print(f"\n... ({len(result.periods) - 15} periods omitted) ...")
        
        for p in result.periods[-5:]:
            if p.period_index not in shown_indices:
                status = "✓" if p.oos_sharpe > 0 else "✗"
                print(f"\n{status} Period {p.period_index + 1}:")
                print(f"  IS:  {p.is_start.strftime('%Y-%m-%d')} to {p.is_end.strftime('%Y-%m-%d')}")
                print(f"  OOS: {p.oos_start.strftime('%Y-%m-%d')} to {p.oos_end.strftime('%Y-%m-%d')}")
                print(f"  IS Sharpe: {p.is_sharpe:.2f} | OOS Sharpe: {p.oos_sharpe:.2f} | Trades: {p.oos_trades}")
    
    # Save results
    results_dir = Path(settings.paths.optimization_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = results_dir / f"opt_{pair[0]}_{pair[1]}_{datetime.now():%Y%m%d_%H%M%S}.json"
    optimizer.save_results(result, str(filepath))
    print(f"\n✓ Results saved to: {filepath}")
    
    # Recommendations
    print("\n" + "-"*60)
    print("RECOMMENDATIONS")
    print("-"*60)
    
    if result.efficiency_ratio >= 0.5 and result.oos_sharpe > 0.5:
        print("\n✓ Strategy shows robust out-of-sample performance")
        print("  Recommended for paper trading validation")
        print(f"\n  Suggested parameters:")
        if result.best_params:
            print(f"    entry_zscore: {result.best_params.entry_zscore}")
            print(f"    exit_zscore: {result.best_params.exit_zscore}")
            print(f"    regression_window: {result.best_params.regression_window}")
            print(f"    zscore_window: {result.best_params.zscore_window}")
    elif result.efficiency_ratio < 0.3:
        print("\n⚠ WARNING: Very low efficiency ratio suggests overfitting")
        print("  Strategy parameters are not robust")
        print("  Consider:")
        print("    - Using more conservative parameters")
        print("    - Increasing OOS period length")
        print("    - Reducing parameter search space")
    else:
        print("\n⚠ Strategy shows moderate performance")
        print("  Further validation recommended before live trading")
    
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
