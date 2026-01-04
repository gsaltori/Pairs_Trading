"""
Example: Run walk-forward optimization on a currency pair.

This script demonstrates how to:
1. Set up parameter ranges for optimization
2. Run walk-forward analysis
3. Analyze robustness of parameters
4. Get recommended settings
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
from src.optimization.optimizer import WalkForwardOptimizer, GridSearchOptimizer


def main():
    """Run walk-forward optimization example."""
    print("="*60)
    print("PAIRS TRADING - WALK-FORWARD OPTIMIZATION")
    print("="*60)
    
    # Configuration
    pair = ('EUR_USD', 'GBP_USD')
    days_of_history = 730  # 2 years for proper walk-forward
    
    print(f"\nPair: {pair[0]}/{pair[1]}")
    print(f"History: {days_of_history} days")
    
    # Initialize settings
    settings = Settings()
    
    # Walk-forward configuration
    settings.optimization.in_sample_bars = 504   # ~3 weeks of H1 bars
    settings.optimization.out_sample_bars = 168  # ~1 week
    
    print(f"\nWalk-Forward Configuration:")
    print(f"  In-Sample:  {settings.optimization.in_sample_bars} bars")
    print(f"  Out-Sample: {settings.optimization.out_sample_bars} bars")
    
    # Load broker config
    try:
        broker_config = BrokerConfig.from_env()
        client = OandaClient(broker_config)
        data_manager = DataManager(client, settings.paths.cache_dir)
        print("\n✓ Connected to OANDA")
    except Exception as e:
        print(f"\nERROR: Could not connect to OANDA: {e}")
        return
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_of_history)
    
    # Fetch data
    print("\nFetching data...")
    
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
    
    if data_a is None or data_b is None:
        print("ERROR: Could not fetch data")
        return
    
    print(f"✓ Loaded {len(data_a)} bars for {pair[0]}")
    print(f"✓ Loaded {len(data_b)} bars for {pair[1]}")
    
    # Initialize optimizer
    optimizer = WalkForwardOptimizer(settings, objective='sharpe')
    
    # Define parameter grid
    param_grid = {
        'entry_zscore': [1.5, 2.0, 2.5],
        'exit_zscore': [0.0, 0.25, 0.5],
        'stop_loss_zscore': [2.5, 3.0, 3.5],
        'regression_window': [90, 120, 150],
        'zscore_window': [30, 60],
        'min_correlation': [0.65, 0.70, 0.75]
    }
    
    optimizer.set_param_grid(param_grid)
    
    # Calculate number of combinations
    n_combos = 1
    for values in param_grid.values():
        n_combos *= len(values)
    
    print(f"\nParameter Grid:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
    print(f"\nTotal combinations: {n_combos}")
    
    # Estimate walk-forward periods
    total_bars = len(data_a)
    is_bars = settings.optimization.in_sample_bars
    oos_bars = settings.optimization.out_sample_bars
    n_periods = (total_bars - is_bars) // oos_bars
    
    print(f"Expected walk-forward periods: {n_periods}")
    print(f"Total optimizations: {n_periods * n_combos:,}")
    
    # Confirm
    print("\n" + "-"*40)
    confirm = input("Start optimization? (y/n): ")
    if confirm.lower() != 'y':
        print("Optimization cancelled.")
        return
    
    # Run optimization
    print("\nRunning walk-forward optimization...")
    print("This may take a while...\n")
    
    result = optimizer.optimize(pair, data_a, data_b)
    
    # Print results
    print(result.summary())
    
    # Save results
    result_file = settings.paths.optimization_results / f"wfo_{pair[0]}_{pair[1]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    optimizer.save_results(result, str(result_file))
    print(f"\n✓ Results saved to: {result_file}")
    
    # Recommendations
    print("\n" + "="*60)
    print("RECOMMENDED SETTINGS")
    print("="*60)
    
    if result.most_robust_params:
        params = result.most_robust_params
        print(f"""
Based on out-of-sample performance, the recommended parameters are:

settings.spread.entry_zscore = {params.entry_zscore}
settings.spread.exit_zscore = {params.exit_zscore}
settings.spread.stop_loss_zscore = {params.stop_loss_zscore}
settings.spread.regression_window = {params.regression_window}
settings.spread.zscore_window = {params.zscore_window}
settings.spread.min_correlation = {params.min_correlation}

Expected Out-of-Sample Performance:
  - Sharpe Ratio: {result.combined_sharpe:.2f}
  - Total Return: {result.combined_return:.2%}
  - Max Drawdown: {result.combined_max_drawdown:.2%}
  - Efficiency Ratio: {result.avg_efficiency_ratio:.2%}
""")
        
        if result.avg_efficiency_ratio < 0.5:
            print("⚠ WARNING: Low efficiency ratio (<50%) suggests overfitting.")
            print("  Consider using simpler parameters or more data.")
        elif result.avg_efficiency_ratio > 0.8:
            print("✓ Good efficiency ratio (>80%) suggests robust parameters.")
    
    print("="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
