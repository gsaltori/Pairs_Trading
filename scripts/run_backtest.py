"""
Example: Run a complete backtest on EUR_USD/GBP_USD pair.

This script demonstrates how to:
1. Initialize the trading system
2. Run a backtest
3. Analyze results
4. Generate a report
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import Settings
from config.broker_config import BrokerConfig
from src.data.broker_client import OandaClient
from src.data.data_manager import DataManager
from src.backtest.backtest_engine import BacktestEngine


def main():
    """Run backtest example."""
    print("="*60)
    print("PAIRS TRADING BACKTEST EXAMPLE")
    print("="*60)
    
    # Configuration
    pair = ('EUR_USD', 'GBP_USD')
    days_of_history = 365
    
    # Initialize settings
    settings = Settings()
    
    # Optionally customize parameters
    settings.spread.entry_zscore = 2.0
    settings.spread.exit_zscore = 0.2
    settings.spread.stop_loss_zscore = 3.0
    settings.backtest.initial_capital = 10000
    
    print(f"\nPair: {pair[0]}/{pair[1]}")
    print(f"History: {days_of_history} days")
    print(f"Initial Capital: ${settings.backtest.initial_capital:,.0f}")
    print(f"Entry Z-Score: ±{settings.spread.entry_zscore}")
    print(f"Exit Z-Score: ±{settings.spread.exit_zscore}")
    
    # Load broker config
    try:
        broker_config = BrokerConfig.from_env()
        print("\n✓ Loaded OANDA credentials from environment")
    except Exception as e:
        print(f"\n⚠ Could not load OANDA credentials: {e}")
        print("Using cached data only (if available)")
        broker_config = None
    
    # Initialize data manager
    if broker_config:
        client = OandaClient(broker_config)
        data_manager = DataManager(client, settings.paths.cache_dir)
    else:
        data_manager = None
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_of_history)
    
    print(f"\nDate Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Fetch data
    print("\nFetching data...")
    
    if data_manager:
        data_a = data_manager.fetch_data(
            instrument=pair[0],
            timeframe=settings.timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        data_b = data_manager.fetch_data(
            instrument=pair[1],
            timeframe=settings.timeframe,
            start_date=start_date,
            end_date=end_date
        )
    else:
        print("ERROR: No data manager available. Set up OANDA credentials.")
        return
    
    if data_a is None or data_b is None:
        print("ERROR: Could not fetch data")
        return
    
    print(f"✓ Loaded {len(data_a)} bars for {pair[0]}")
    print(f"✓ Loaded {len(data_b)} bars for {pair[1]}")
    
    # Run backtest
    print("\nRunning backtest...")
    
    engine = BacktestEngine(settings, data_manager)
    result = engine.run_backtest(pair, data_a, data_b)
    
    # Print results
    print(result.summary())
    
    # Save results
    result_file = settings.paths.backtest_results / f"backtest_{pair[0]}_{pair[1]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    engine.save_results(result, str(result_file))
    print(f"\n✓ Results saved to: {result_file}")
    
    # Print trade log
    if result.trades:
        print("\n" + "-"*60)
        print("TRADE LOG (First 10 trades)")
        print("-"*60)
        
        trades_df = engine.get_trades_dataframe()
        print(trades_df.head(10).to_string())
    
    print("\n" + "="*60)
    print("BACKTEST COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
