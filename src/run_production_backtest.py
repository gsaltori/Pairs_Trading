"""
Run Production Backtest Comparison

Loads MT5 data and runs the production backtest harness
comparing Baseline vs Gated strategies.
"""

import sys
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent))

from crv_engine.mt5_loader import initialize_mt5, shutdown_mt5, load_ohlc
from trading_system.backtest_harness import run_production_backtest


def load_data(n_bars: int = 5000):
    """Load and align EURUSD/GBPUSD data."""
    print("=" * 70)
    print("LOADING MT5 DATA")
    print("=" * 70)
    
    if not initialize_mt5():
        raise RuntimeError("Failed to initialize MT5")
    
    try:
        print(f"Loading EURUSD H4 ({n_bars} bars)...")
        eurusd_bars = load_ohlc("EURUSD", "H4", n_bars)
        
        print(f"Loading GBPUSD H4 ({n_bars} bars)...")
        gbpusd_bars = load_ohlc("GBPUSD", "H4", n_bars)
        
        if eurusd_bars is None or gbpusd_bars is None:
            raise RuntimeError("Failed to load data")
        
        # Convert to DataFrames
        df_eurusd = pd.DataFrame([{
            'timestamp': bar.timestamp,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
        } for bar in eurusd_bars])
        
        df_gbpusd = pd.DataFrame([{
            'timestamp': bar.timestamp,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
        } for bar in gbpusd_bars])
        
        # Align by timestamp
        df_eurusd = df_eurusd.set_index('timestamp')
        df_gbpusd = df_gbpusd.set_index('timestamp')
        
        common_idx = df_eurusd.index.intersection(df_gbpusd.index)
        df_eurusd = df_eurusd.loc[common_idx].reset_index()
        df_gbpusd = df_gbpusd.loc[common_idx].reset_index()
        
        print(f"✅ Loaded {len(df_eurusd)} aligned bars")
        print(f"   Date range: {df_eurusd['timestamp'].iloc[0]} to {df_eurusd['timestamp'].iloc[-1]}")
        
        return df_eurusd, df_gbpusd
        
    finally:
        shutdown_mt5()


def main():
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " PRODUCTION BACKTEST: BASELINE vs GATED ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    # Load data
    df_eurusd, df_gbpusd = load_data(n_bars=5000)
    print()
    
    # Run backtest
    baseline, gated, verdict = run_production_backtest(df_eurusd, df_gbpusd)
    
    return verdict


if __name__ == "__main__":
    verdict = main()
