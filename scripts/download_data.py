"""
Download and save historical data from MT5.

This script downloads historical data for multiple symbols and saves it locally
for offline backtesting and analysis.

Usage:
    # Single timeframe
    python scripts/download_data.py --symbols EURUSD,GBPUSD --days 730 --timeframe H1
    
    # Multiple timeframes
    python scripts/download_data.py --forex --days 730 --timeframes H1,H4,D1
    
    # All pairs, all major timeframes
    python scripts/download_data.py --all --days 365 --timeframes H1,H4,D1
    
    # Just majors
    python scripts/download_data.py --majors --days 730 --timeframes H1,H4,D1
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import Settings, Timeframe
from config.broker_config import MT5Config, IC_MARKETS_FOREX_PAIRS
from src.data.broker_client import MT5Client, Timeframe as MT5Timeframe


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Symbol groups
MAJOR_PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
    "AUDUSD", "NZDUSD", "USDCAD"
]

CROSS_PAIRS = [
    "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY",
    "EURGBP", "EURAUD", "EURCHF", "EURCAD", "EURNZD",
    "GBPAUD", "GBPNZD", "GBPCAD", "GBPCHF",
    "AUDNZD", "AUDCAD", "AUDCHF",
    "NZDCAD", "NZDCHF", "CADCHF"
]

SCREENING_PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
    "AUDUSD", "NZDUSD", "USDCAD",
    "EURJPY", "GBPJPY", "AUDJPY",
    "EURGBP", "EURAUD", "EURCHF",
    "GBPAUD", "AUDNZD"
]


def download_symbol_data(
    client: MT5Client,
    symbol: str,
    timeframe: MT5Timeframe,
    days: int,
    output_dir: Path
) -> bool:
    """
    Download and save data for a single symbol.
    
    Args:
        client: MT5 client
        symbol: Symbol to download
        timeframe: Timeframe
        days: Number of days
        output_dir: Output directory
        
    Returns:
        True if successful
    """
    try:
        bars_per_day = {
            MT5Timeframe.M1: 1440,
            MT5Timeframe.M5: 288,
            MT5Timeframe.M15: 96,
            MT5Timeframe.M30: 48,
            MT5Timeframe.H1: 24,
            MT5Timeframe.H4: 6,
            MT5Timeframe.D1: 1,
        }
        
        count = days * bars_per_day.get(timeframe, 24)
        
        # Get data
        data = client.get_candles(symbol, timeframe, count)
        
        if data.empty:
            return False
        
        # Save to parquet
        filename = f"{symbol}_{timeframe.name}_{days}d.parquet"
        filepath = output_dir / filename
        
        data.to_parquet(filepath)
        
        # Also save to CSV for easy inspection
        csv_filepath = output_dir / f"{symbol}_{timeframe.name}_{days}d.csv"
        data.to_csv(csv_filepath)
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to download {symbol}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Download historical data from MT5'
    )
    
    parser.add_argument(
        '--symbols', type=str,
        help='Comma-separated list of symbols (e.g., EURUSD,GBPUSD)'
    )
    parser.add_argument(
        '--all', action='store_true',
        help='Download all IC Markets Forex pairs'
    )
    parser.add_argument(
        '--forex', action='store_true',
        help='Download screening set (15 key pairs)'
    )
    parser.add_argument(
        '--majors', action='store_true',
        help='Download major pairs only (7 pairs)'
    )
    parser.add_argument(
        '--days', type=int, default=365,
        help='Number of days to download (default: 365)'
    )
    parser.add_argument(
        '--timeframe', type=str, default='H1',
        help='Timeframe: M1, M5, M15, M30, H1, H4, D1 (default: H1)'
    )
    parser.add_argument(
        '--timeframes', type=str,
        help='Multiple timeframes: e.g., H1,H4,D1 (overrides --timeframe)'
    )
    parser.add_argument(
        '--output', type=str, default='data/historical',
        help='Output directory (default: data/historical)'
    )
    
    args = parser.parse_args()
    
    # Determine symbols to download
    if args.all:
        symbols = IC_MARKETS_FOREX_PAIRS
    elif args.forex:
        symbols = SCREENING_PAIRS
    elif args.majors:
        symbols = MAJOR_PAIRS
    elif args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(',')]
    else:
        # Default: screening pairs
        symbols = SCREENING_PAIRS
    
    # Parse timeframe(s)
    tf_mapping = {
        'M1': MT5Timeframe.M1,
        'M5': MT5Timeframe.M5,
        'M15': MT5Timeframe.M15,
        'M30': MT5Timeframe.M30,
        'H1': MT5Timeframe.H1,
        'H4': MT5Timeframe.H4,
        'D1': MT5Timeframe.D1,
    }
    
    # Support multiple timeframes
    if args.timeframes:
        tf_list = [s.strip().upper() for s in args.timeframes.split(',')]
        timeframes = [tf_mapping.get(tf, MT5Timeframe.H1) for tf in tf_list]
        timeframe_names = tf_list
    else:
        timeframes = [tf_mapping.get(args.timeframe.upper(), MT5Timeframe.H1)]
        timeframe_names = [args.timeframe.upper()]
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("MT5 DATA DOWNLOADER")
    print("=" * 60)
    print(f"\nSymbols: {len(symbols)}")
    print(f"Days: {args.days}")
    print(f"Timeframes: {', '.join(timeframe_names)}")
    print(f"Output: {output_dir}")
    
    # Connect to MT5
    try:
        config = MT5Config.from_env()
        client = MT5Client(config)
        
        if not client.connect():
            print("\nERROR: Could not connect to MT5")
            print("Make sure MT5 terminal is running and credentials are correct.")
            return 1
        
        print("\n✓ Connected to MT5")
        
        account = client.get_account_info()
        print(f"✓ Account: {account.get('login')}")
        print(f"✓ Server: {account.get('server')}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        return 1
    
    # Download data
    print("\n" + "-" * 60)
    print("DOWNLOADING DATA...")
    print("-" * 60 + "\n")
    
    success_count = 0
    fail_count = 0
    total_downloads = len(symbols) * len(timeframes)
    current = 0
    
    for timeframe, tf_name in zip(timeframes, timeframe_names):
        print(f"\n📊 Timeframe: {tf_name}")
        print("-" * 40)
        
        for symbol in symbols:
            current += 1
            result = download_symbol_data(client, symbol, timeframe, args.days, output_dir)
            
            if result:
                success_count += 1
                print(f"[{current:3}/{total_downloads}] ✓ {symbol}_{tf_name}")
            else:
                fail_count += 1
                print(f"[{current:3}/{total_downloads}] ✗ {symbol}_{tf_name} (no data or error)")
    
    # Disconnect
    client.disconnect()
    
    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"\nSuccessful: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"\nData saved to: {output_dir.absolute()}")
    
    # Calculate total size
    total_size = sum(f.stat().st_size for f in output_dir.glob("*.parquet"))
    print(f"Total size: {total_size / (1024 * 1024):.1f} MB")
    
    # Next steps
    print("\n" + "-" * 60)
    print("NEXT STEPS")
    print("-" * 60)
    print("\n1. Screen pairs for a specific timeframe:")
    print("   python scripts/conditional_screen.py --timeframe H1 --save")
    print("   python scripts/conditional_screen.py --timeframe H4 --save")
    print("   python scripts/conditional_screen.py --timeframe D1 --save")
    print("\n2. Or use strict screening:")
    print("   python scripts/strict_screen.py --timeframe H1 --backtest")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
