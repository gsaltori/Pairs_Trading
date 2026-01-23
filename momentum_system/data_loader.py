"""
Cross-Sectional Momentum System - Data Loader
Handles ETF data acquisition, validation, and preprocessing.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import warnings

try:
    import yfinance as yf
except ImportError:
    raise ImportError("yfinance required: pip install yfinance")

from config import CONFIG, EXEC_CONFIG


class DataLoader:
    """
    Loads and validates ETF data for momentum system.
    
    Features:
    - Downloads adjusted close prices (handles splits/dividends)
    - Validates data quality
    - Aligns all assets to common dates
    - Caches to disk
    """
    
    def __init__(self, cache_dir: str = "data"):
        """Initialize data loader."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._price_data: Optional[pd.DataFrame] = None
    
    def load_universe(
        self,
        start_date: str = None,
        end_date: str = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Load price data for entire universe.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with dates as index, symbols as columns (adjusted close)
        """
        if start_date is None:
            start_date = CONFIG.BACKTEST_START
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        cache_file = self.cache_dir / f"universe_{start_date}_{end_date}.parquet"
        
        # Try cache
        if use_cache and cache_file.exists():
            print(f"Loading from cache: {cache_file}")
            df = pd.read_parquet(cache_file)
            self._price_data = df
            return df
        
        # Download fresh
        print(f"Downloading data for {len(CONFIG.UNIVERSE)} assets...")
        print(f"Period: {start_date} to {end_date}")
        
        prices = {}
        
        for symbol in CONFIG.UNIVERSE:
            try:
                print(f"  Downloading {symbol}...", end=" ")
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(
                        start=start_date,
                        end=end_date,
                        auto_adjust=True,  # Use adjusted prices
                    )
                
                if len(hist) == 0:
                    print("NO DATA")
                    continue
                
                prices[symbol] = hist['Close']
                print(f"{len(hist)} bars")
                
            except Exception as e:
                print(f"ERROR: {e}")
        
        # Combine into DataFrame
        df = pd.DataFrame(prices)
        
        # Remove timezone
        df.index = df.index.tz_localize(None)
        
        # Sort
        df = df.sort_index()
        
        # Forward fill small gaps (max 5 days)
        df = df.fillna(method='ffill', limit=5)
        
        # Drop rows with any remaining NaN
        df = df.dropna()
        
        # Cache
        if use_cache:
            df.to_parquet(cache_file)
            print(f"Cached to: {cache_file}")
        
        self._price_data = df
        
        return df
    
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate data quality.
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check minimum history
        min_required = CONFIG.MOMENTUM_LOOKBACK + CONFIG.TREND_FILTER_PERIOD + 100
        if len(df) < min_required:
            issues.append(f"Insufficient history: {len(df)} < {min_required} required")
        
        # Check all symbols present
        missing = set(CONFIG.UNIVERSE) - set(df.columns)
        if missing:
            issues.append(f"Missing symbols: {missing}")
        
        # Check for excessive gaps
        for col in df.columns:
            pct_change = df[col].pct_change().abs()
            extreme = pct_change > 0.50  # 50% daily move is suspicious
            if extreme.sum() > 5:
                issues.append(f"{col}: {extreme.sum()} extreme moves (>50%)")
        
        # Check date range
        years = (df.index[-1] - df.index[0]).days / 365
        if years < 10:
            issues.append(f"Only {years:.1f} years of data (need 10+)")
        
        return len(issues) == 0, issues
    
    def get_monthly_dates(self, df: pd.DataFrame) -> List[pd.Timestamp]:
        """
        Get last trading day of each month.
        
        Args:
            df: Price DataFrame
            
        Returns:
            List of month-end dates
        """
        # Group by year-month and get last date in each group
        monthly = df.groupby(pd.Grouper(freq='M')).tail(1)
        return list(monthly.index)
    
    def split_in_out_sample(
        self,
        df: pd.DataFrame,
        oos_pct: float = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into in-sample and out-of-sample.
        
        Args:
            df: Full price DataFrame
            oos_pct: Percentage for out-of-sample (default from config)
            
        Returns:
            Tuple of (in_sample_df, out_of_sample_df)
        """
        if oos_pct is None:
            oos_pct = CONFIG.OUT_OF_SAMPLE_PCT
        
        split_idx = int(len(df) * (1 - oos_pct))
        
        in_sample = df.iloc[:split_idx]
        out_of_sample = df.iloc[split_idx:]
        
        return in_sample, out_of_sample
    
    def get_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate daily returns."""
        return df.pct_change().dropna()
    
    def get_momentum(
        self,
        df: pd.DataFrame,
        lookback: int = None,
    ) -> pd.DataFrame:
        """
        Calculate momentum (total return over lookback).
        
        momentum = (Price[t] / Price[t-lookback]) - 1
        
        Args:
            df: Price DataFrame
            lookback: Lookback period in days
            
        Returns:
            DataFrame of momentum values
        """
        if lookback is None:
            lookback = CONFIG.MOMENTUM_LOOKBACK
        
        return df / df.shift(lookback) - 1
    
    def get_ema(
        self,
        df: pd.DataFrame,
        period: int = None,
    ) -> pd.DataFrame:
        """
        Calculate EMA for trend filter.
        
        Args:
            df: Price DataFrame
            period: EMA period
            
        Returns:
            DataFrame of EMA values
        """
        if period is None:
            period = CONFIG.TREND_FILTER_PERIOD
        
        return df.ewm(span=period, adjust=False).mean()


def load_data() -> pd.DataFrame:
    """Convenience function to load data."""
    loader = DataLoader()
    df = loader.load_universe()
    
    valid, issues = loader.validate_data(df)
    
    if not valid:
        print("\n⚠️ Data validation issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✅ Data validation passed")
    
    print(f"\nData Summary:")
    print(f"  Period: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"  Days: {len(df)}")
    print(f"  Years: {(df.index[-1] - df.index[0]).days / 365:.1f}")
    print(f"  Assets: {len(df.columns)}")
    
    return df


if __name__ == "__main__":
    df = load_data()
    print("\nSample (last 5 rows):")
    print(df.tail())
