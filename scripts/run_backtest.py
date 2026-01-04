"""
Example: Run backtest for a currency pair.

This script demonstrates how to:
1. Connect to MT5
2. Load historical data
3. Run a backtest
4. Display results
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import Settings, Timeframe
from config.broker_config import MT5Config
from src.data.broker_client import MT5Client, Timeframe as MT5Timeframe
from src.data.data_manager import DataManager
from src.backtest.backtest_engine import BacktestEngine


def main():
    """Run backtest example."""
    print("="*60)
    print("PAIRS TRADING BACKTEST")
    print("="*60)
    
    # Configuration
    pair = ("EURUSD", "GBPUSD")
    days = 365
    timeframe = Timeframe.H1
    
    print(f"\nPair: {pair[0]}/{pair[1]}")
    print(f"Period: {days} days")
    print(f"Timeframe: {timeframe.value}")
    
    # Initialize
    settings = Settings()
    
    try:
        # Connect to MT5
        config = MT5Config.from_env()
        client = MT5Client(config)
        
        if not client.connect():
            print("\nERROR: Could not connect to MT5")
            print("Make sure MT5 terminal is running and credentials are correct.")
            return
        
        print("\n✓ Connected to MT5")
        
        # Get account info
        account = client.get_account_info()
        print(f"✓ Account: {account.get('login')}")
        print(f"✓ Balance: ${account.get('balance', 0):,.2f}")
        
        # Load data
        data_manager = DataManager(client, settings.paths.cache_dir)
        
        bars_per_day = 24  # H1
        count = days * bars_per_day
        
        print(f"\nLoading {count} bars of data...")
        
        mt5_tf = MT5Timeframe.from_string(timeframe.value)
        price_a, price_b = data_manager.get_pair_data(
            pair[0], pair[1], mt5_tf, count
        )
        
        if len(price_a) == 0:
            print("ERROR: No data received")
            return
        
        print(f"✓ Loaded {len(price_a)} bars")
        print(f"  Period: {price_a.index[0]} to {price_a.index[-1]}")
        
        # Run backtest
        print("\n" + "-"*60)
        print("RUNNING BACKTEST...")
        print("-"*60)
        
        engine = BacktestEngine(settings)
        result = engine.run_backtest(pair, price_a, price_b)
        
        if result is None:
            print("ERROR: Backtest failed")
            return
        
        # Display results
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        
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
        
        print(f"\nCapital:")
        print(f"  Initial:         ${result.initial_capital:>9,.2f}")
        print(f"  Final:           ${result.final_capital:>9,.2f}")
        print(f"  Net Profit:      ${result.final_capital - result.initial_capital:>9,.2f}")
        
        # Save results
        results_path = Path(settings.paths.backtest_dir) / f"{pair[0]}_{pair[1]}_{datetime.now():%Y%m%d_%H%M%S}.json"
        engine.save_results(result, str(results_path))
        print(f"\n✓ Results saved to: {results_path}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        raise
    finally:
        if 'client' in locals():
            client.disconnect()
    
    print("\n" + "="*60)
    print("BACKTEST COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
