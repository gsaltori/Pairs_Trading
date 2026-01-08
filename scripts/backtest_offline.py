"""
Offline Backtesting Script.

Run backtests using locally saved historical data (no MT5 connection required).

Usage:
    python scripts/backtest_offline.py --pair EURUSD,GBPUSD
    python scripts/backtest_offline.py --pair EURUSD,GBPUSD --data-dir data/historical
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
from src.backtest.backtest_engine import BacktestEngine


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
    """
    Load data from local parquet file.
    
    Args:
        symbol: Symbol name
        data_dir: Directory containing data files
        timeframe: Timeframe string
        
    Returns:
        Close price series
    """
    # Find matching file
    pattern = f"{symbol}_{timeframe}_*.parquet"
    files = list(data_dir.glob(pattern))
    
    if not files:
        # Try without days suffix
        pattern = f"{symbol}_{timeframe}.parquet"
        files = list(data_dir.glob(pattern))
    
    if not files:
        raise FileNotFoundError(f"No data file found for {symbol} {timeframe} in {data_dir}")
    
    # Use the most recent file
    filepath = max(files, key=lambda f: f.stat().st_mtime)
    
    logger.info(f"Loading {filepath}")
    
    df = pd.read_parquet(filepath)
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'time' in df.columns:
            df.set_index('time', inplace=True)
        df.index = pd.to_datetime(df.index)
    
    return df['close']


def main():
    parser = argparse.ArgumentParser(
        description='Run offline backtest from local data'
    )
    
    parser.add_argument(
        '--pair', type=str, required=True,
        help='Pair to backtest (e.g., EURUSD,GBPUSD)'
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
        '--capital', type=float, default=10000,
        help='Initial capital (default: 10000)'
    )
    parser.add_argument(
        '--save', action='store_true',
        help='Save results to file'
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
    print("OFFLINE BACKTEST")
    print("="*60)
    print(f"\nPair: {pair[0]}/{pair[1]}")
    print(f"Data directory: {data_dir}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Initial capital: ${args.capital:,.2f}")
    
    # Load data
    print("\n" + "-"*60)
    print("LOADING DATA...")
    print("-"*60)
    
    try:
        price_a = load_local_data(pair[0], data_dir, args.timeframe)
        print(f"✓ {pair[0]}: {len(price_a)} bars")
        print(f"  Period: {price_a.index[0]} to {price_a.index[-1]}")
        
        price_b = load_local_data(pair[1], data_dir, args.timeframe)
        print(f"✓ {pair[1]}: {len(price_b)} bars")
        print(f"  Period: {price_b.index[0]} to {price_b.index[-1]}")
        
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
    
    # Run backtest
    print("\n" + "-"*60)
    print("RUNNING BACKTEST...")
    print("-"*60)
    
    settings = Settings()
    settings.backtest.initial_capital = args.capital
    
    engine = BacktestEngine(settings)
    result = engine.run_backtest(pair, price_a, price_b, args.capital)
    
    if result is None:
        print("\nERROR: Backtest failed")
        return 1
    
    # Display results
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    
    print(f"\nPeriod: {result.start_date.strftime('%Y-%m-%d')} to {result.end_date.strftime('%Y-%m-%d')}")
    
    print(f"\nPerformance:")
    print(f"  Total Return:    {result.total_return:>10.2%}")
    print(f"  Annual Return:   {result.annual_return:>10.2%}")
    print(f"  Sharpe Ratio:    {result.sharpe_ratio:>10.2f}")
    print(f"  Sortino Ratio:   {result.sortino_ratio:>10.2f}")
    print(f"  Calmar Ratio:    {result.calmar_ratio:>10.2f}")
    
    print(f"\nRisk Metrics:")
    print(f"  Max Drawdown:    {result.max_drawdown:>10.2%}")
    print(f"  Volatility:      {result.volatility:>10.2%}")
    
    print(f"\nTrade Statistics:")
    print(f"  Total Trades:    {result.total_trades:>10}")
    print(f"  Win Rate:        {result.win_rate:>10.1%}")
    print(f"  Profit Factor:   {result.profit_factor:>10.2f}")
    print(f"  Avg Trade:       ${result.avg_trade:>9.2f}")
    print(f"  Avg Win:         ${result.avg_win:>9.2f}")
    print(f"  Avg Loss:        ${result.avg_loss:>9.2f}")
    print(f"  Max Win:         ${result.max_win:>9.2f}")
    print(f"  Max Loss:        ${result.max_loss:>9.2f}")
    print(f"  Avg Holding:     {result.avg_holding_period:>9.1f} hours")
    
    print(f"\nCapital:")
    print(f"  Initial:         ${result.initial_capital:>9,.2f}")
    print(f"  Final:           ${result.final_capital:>9,.2f}")
    print(f"  Net Profit:      ${result.final_capital - result.initial_capital:>9,.2f}")
    print(f"  Total Costs:     ${result.total_costs:>9,.2f}")
    
    # Save results
    if args.save:
        results_dir = Path(settings.paths.backtest_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = results_dir / f"{pair[0]}_{pair[1]}_{datetime.now():%Y%m%d_%H%M%S}.json"
        engine.save_results(result, str(filepath))
        print(f"\n✓ Results saved to: {filepath}")
    
    # Trade summary
    if result.trades:
        print("\n" + "-"*60)
        print("TRADE LOG (Last 10)")
        print("-"*60)
        
        for trade in result.trades[-10:]:
            direction = "LONG" if trade.direction == "long_spread" else "SHORT"
            status = "✓" if trade.net_pnl > 0 else "✗"
            print(f"\n{status} {direction} @ Z={trade.entry_zscore:+.2f}")
            print(f"  Entry: {trade.entry_time.strftime('%Y-%m-%d %H:%M')}")
            print(f"  Exit:  {trade.exit_time.strftime('%Y-%m-%d %H:%M')} ({trade.exit_reason})")
            print(f"  P&L:   ${trade.net_pnl:.2f}")
    
    print("\n" + "="*60)
    print("BACKTEST COMPLETE")
    print("="*60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
