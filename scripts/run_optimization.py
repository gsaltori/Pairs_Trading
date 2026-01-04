"""
Example: Run walk-forward optimization.

This script demonstrates how to:
1. Load historical data
2. Run walk-forward optimization
3. Analyze parameter stability
4. Get optimal parameters
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
from src.optimization.optimizer import WalkForwardOptimizer


def main():
    """Run optimization example."""
    print("="*60)
    print("PAIRS TRADING - WALK-FORWARD OPTIMIZATION")
    print("="*60)
    
    # Configuration
    pair = ("EURUSD", "GBPUSD")
    days = 730  # 2 years
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
            return
        
        print("\n✓ Connected to MT5")
        
        # Load data
        data_manager = DataManager(client, settings.paths.cache_dir)
        
        bars_per_day = 24
        count = days * bars_per_day
        
        mt5_tf = MT5Timeframe.from_string(timeframe.value)
        
        print(f"\nLoading {count} bars...")
        
        price_a, price_b = data_manager.get_pair_data(
            pair[0], pair[1], mt5_tf, count
        )
        
        if len(price_a) < 1000:
            print(f"ERROR: Insufficient data ({len(price_a)} bars)")
            return
        
        print(f"✓ Loaded {len(price_a)} bars")
        
        # Run optimization
        print("\n" + "-"*60)
        print("RUNNING WALK-FORWARD OPTIMIZATION...")
        print("-"*60)
        print("\nThis may take several minutes...")
        
        optimizer = WalkForwardOptimizer(settings)
        result = optimizer.optimize(pair, price_a, price_b)
        
        # Display results
        print("\n" + "="*60)
        print("OPTIMIZATION RESULTS")
        print("="*60)
        
        print(f"\nPeriods analyzed: {len(result.periods)}")
        print(f"Total trades: {result.total_trades}")
        
        print(f"\nPerformance:")
        print(f"  In-Sample Sharpe:  {result.is_sharpe:>8.2f}")
        print(f"  Out-of-Sample Sharpe: {result.oos_sharpe:>8.2f}")
        print(f"  Efficiency Ratio:  {result.efficiency_ratio:>8.2f}")
        
        if result.efficiency_ratio >= 0.5:
            print("  → Good parameter stability (efficiency ≥ 0.5)")
        else:
            print("  → Warning: Low efficiency ratio (potential overfitting)")
        
        if result.best_params:
            print(f"\nBest Parameters:")
            print(f"  Entry Z-score:      {result.best_params.entry_zscore}")
            print(f"  Exit Z-score:       {result.best_params.exit_zscore}")
            print(f"  Stop-loss Z-score:  {result.best_params.stop_loss_zscore}")
            print(f"  Regression window:  {result.best_params.regression_window}")
            print(f"  Z-score window:     {result.best_params.zscore_window}")
            print(f"  Min correlation:    {result.best_params.min_correlation}")
        
        # Period details
        print("\nPeriod-by-Period Results:")
        print("-" * 40)
        
        for i, period in enumerate(result.periods, 1):
            print(f"\nPeriod {i}:")
            print(f"  IS Sharpe: {period.is_sharpe:.2f} | OOS Sharpe: {period.oos_sharpe:.2f}")
            if period.best_params:
                print(f"  Best entry Z: {period.best_params.entry_zscore}")
        
        # Save results
        results_path = Path(settings.paths.optimization_dir) / f"opt_{pair[0]}_{pair[1]}_{datetime.now():%Y%m%d}.json"
        optimizer.save_results(result, str(results_path))
        print(f"\n✓ Results saved to: {results_path}")
        
        # Recommendations
        print("\n" + "-"*60)
        print("RECOMMENDATIONS")
        print("-"*60)
        
        if result.efficiency_ratio >= 0.5 and result.oos_sharpe > 0.5:
            print("\n✓ Strategy shows robust out-of-sample performance")
            print("  Recommended for paper trading validation")
        elif result.efficiency_ratio < 0.5:
            print("\n⚠ Warning: Low efficiency ratio suggests overfitting")
            print("  Consider using more conservative parameters")
        elif result.oos_sharpe < 0.5:
            print("\n⚠ Warning: Poor out-of-sample performance")
            print("  Strategy may not be viable for this pair")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        raise
    finally:
        if 'client' in locals():
            client.disconnect()
    
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
