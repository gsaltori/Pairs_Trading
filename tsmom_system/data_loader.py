"""
Time-Series Momentum System - Data Loader
ETF data acquisition with validation.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

try:
    import yfinance as yf
except ImportError:
    raise ImportError("yfinance required: pip install yfinance")

from config import CONFIG


class DataValidationError(Exception):
    """Data validation failed."""
    pass


class DataLoader:
    """Load and validate ETF data with caching."""
    
    def __init__(self, cache_dir: str = "data"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def load_universe(
        self,
        start_date: str = None,
        end_date: str = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Load adjusted close prices for universe.
        
        Returns:
            DataFrame: dates as index, symbols as columns
        """
        if start_date is None:
            start_date = CONFIG.BACKTEST_START
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        cache_file = self.cache_dir / f"tsmom_data_{start_date}_{end_date}.parquet"
        
        if use_cache and cache_file.exists():
            print(f"Loading from cache: {cache_file}")
            df = pd.read_parquet(cache_file)
            return df
        
        print(f"Downloading {len(CONFIG.UNIVERSE)} ETFs...")
        
        prices = {}
        for symbol in CONFIG.UNIVERSE:
            try:
                print(f"  {symbol}...", end=" ")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(
                        start=start_date,
                        end=end_date,
                        auto_adjust=True,
                    )
                
                if len(hist) > 0:
                    prices[symbol] = hist['Close']
                    print(f"{len(hist)} bars")
                else:
                    print("NO DATA")
            except Exception as e:
                print(f"ERROR: {e}")
        
        df = pd.DataFrame(prices)
        df.index = df.index.tz_localize(None)
        df = df.sort_index()
        
        # Forward fill small gaps
        df = df.fillna(method='ffill', limit=5)
        
        # Drop rows with any remaining NaN
        df = df.dropna()
        
        if use_cache:
            df.to_parquet(cache_file)
            print(f"Cached to: {cache_file}")
        
        return df
    
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate data quality."""
        issues = []
        
        # Minimum history required
        warmup = CONFIG.TREND_LOOKBACK + CONFIG.VOL_LOOKBACK + 50
        if len(df) < warmup:
            issues.append(f"Insufficient data: {len(df)} < {warmup} required")
        
        # All symbols present
        missing = set(CONFIG.UNIVERSE) - set(df.columns)
        if missing:
            issues.append(f"Missing symbols: {missing}")
        
        # No negative/zero prices
        if (df <= 0).any().any():
            issues.append("Non-positive prices detected")
        
        # No NaN after warmup
        after_warmup = df.iloc[warmup:]
        nan_count = after_warmup.isna().sum().sum()
        if nan_count > 0:
            issues.append(f"NaN values after warmup: {nan_count}")
        
        # Years of data
        years = (df.index[-1] - df.index[0]).days / 365.25
        if years < 5:
            issues.append(f"Only {years:.1f} years of data")
        
        return len(issues) == 0, issues
    
    def get_monthly_rebalance_dates(self, df: pd.DataFrame) -> List[pd.Timestamp]:
        """Get last trading day of each month."""
        monthly = df.groupby(pd.Grouper(freq='ME')).tail(1)
        return list(monthly.index)
    
    def split_in_out_sample(
        self,
        df: pd.DataFrame,
        oos_pct: float = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split into in-sample (70%) and out-of-sample (30%)."""
        if oos_pct is None:
            oos_pct = CONFIG.OUT_OF_SAMPLE_PCT
        
        split_idx = int(len(df) * (1 - oos_pct))
        return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


if __name__ == "__main__":
    loader = DataLoader()
    df = loader.load_universe()
    
    valid, issues = loader.validate_data(df)
    
    print(f"\nData Summary:")
    print(f"  Period: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"  Days: {len(df)}")
    print(f"  Assets: {len(df.columns)}")
    print(f"  Valid: {valid}")
    
    if issues:
        for i in issues:
            print(f"  Issue: {i}")
    
    rebal_dates = loader.get_monthly_rebalance_dates(df)
    print(f"  Rebalance dates: {len(rebal_dates)}")
