"""
Trend Following System - Data Loader
ETF data ingestion with validation and caching.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, List
import warnings

try:
    import yfinance as yf
except ImportError:
    raise ImportError("yfinance required: pip install yfinance")

from config import TRADING_CONFIG, EXECUTION_CONFIG


class DataLoader:
    """
    Loads and validates ETF data for backtesting and live trading.
    
    Features:
    - Downloads from Yahoo Finance
    - Validates data quality
    - Caches to disk for performance
    - Handles corporate actions (adjusted close)
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize data loader with optional cache directory."""
        self.cache_dir = Path(cache_dir) if cache_dir else Path(EXECUTION_CONFIG.DATA_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._data_cache: Dict[str, pd.DataFrame] = {}
    
    def load_symbol(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Load daily OHLCV data for a single symbol.
        
        Args:
            symbol: ETF ticker symbol
            start_date: Start date for data
            end_date: End date for data
            use_cache: Whether to use disk cache
            
        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume, Adj Close
        """
        # Default dates
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=365 * (TRADING_CONFIG.MIN_HISTORY_YEARS + 1))
        
        # Check memory cache first
        cache_key = f"{symbol}_{start_date.date()}_{end_date.date()}"
        if cache_key in self._data_cache:
            return self._data_cache[cache_key].copy()
        
        # Check disk cache
        cache_file = self.cache_dir / f"{symbol}_daily.parquet"
        
        if use_cache and cache_file.exists():
            df = pd.read_parquet(cache_file)
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            
            # Validate we have enough data
            if len(df) > 0:
                days_available = (df.index[-1] - df.index[0]).days
                if days_available >= 365 * TRADING_CONFIG.MIN_HISTORY_YEARS * 0.9:
                    self._data_cache[cache_key] = df
                    return df.copy()
        
        # Download fresh data
        df = self._download_data(symbol, start_date, end_date)
        
        # Cache to disk
        if use_cache and df is not None and len(df) > 0:
            df.to_parquet(cache_file)
        
        self._data_cache[cache_key] = df
        return df.copy()
    
    def _download_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Download data from Yahoo Finance."""
        print(f"  Downloading {symbol}...")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval="1d",
                auto_adjust=False,
            )
        
        if df.empty:
            raise ValueError(f"No data returned for {symbol}")
        
        # Standardize column names
        df.columns = [c.title().replace(" ", "_") for c in df.columns]
        
        # Ensure we have required columns
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Use adjusted close for backtesting (handles splits/dividends)
        if 'Adj_Close' in df.columns:
            # Calculate adjustment factor
            adj_factor = df['Adj_Close'] / df['Close']
            
            # Adjust OHLC prices
            df['Open'] = df['Open'] * adj_factor
            df['High'] = df['High'] * adj_factor
            df['Low'] = df['Low'] * adj_factor
            df['Close'] = df['Adj_Close']
        
        # Remove timezone info for consistency
        df.index = df.index.tz_localize(None)
        
        # Sort by date
        df = df.sort_index()
        
        # Remove any duplicate dates
        df = df[~df.index.duplicated(keep='last')]
        
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    def load_universe(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Load data for all symbols in universe.
        
        Args:
            symbols: List of symbols (defaults to config)
            start_date: Start date
            end_date: End date
            
        Returns:
            Dict mapping symbol to DataFrame
        """
        if symbols is None:
            symbols = list(TRADING_CONFIG.SYMBOLS)
        
        print(f"Loading data for {len(symbols)} symbols...")
        
        data = {}
        for symbol in symbols:
            try:
                df = self.load_symbol(symbol, start_date, end_date)
                data[symbol] = df
                print(f"  {symbol}: {len(df)} bars, {df.index[0].date()} to {df.index[-1].date()}")
            except Exception as e:
                print(f"  {symbol}: ERROR - {e}")
        
        return data
    
    def validate_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
        """
        Validate data quality for all symbols.
        
        Returns:
            Dict mapping symbol to list of issues (empty if valid)
        """
        issues = {}
        
        for symbol, df in data.items():
            symbol_issues = []
            
            # Check minimum history
            if len(df) < 252 * TRADING_CONFIG.MIN_HISTORY_YEARS:
                symbol_issues.append(
                    f"Insufficient history: {len(df)} bars "
                    f"(need {252 * TRADING_CONFIG.MIN_HISTORY_YEARS})"
                )
            
            # Check for gaps
            date_diffs = df.index.to_series().diff().dropna()
            large_gaps = date_diffs[date_diffs > timedelta(days=5)]
            if len(large_gaps) > 0:
                symbol_issues.append(f"Data gaps detected: {len(large_gaps)} gaps > 5 days")
            
            # Check for invalid prices
            if (df['Close'] <= 0).any():
                symbol_issues.append("Invalid prices: Close <= 0")
            
            if (df['Volume'] < 0).any():
                symbol_issues.append("Invalid volume: Volume < 0")
            
            # Check OHLC consistency
            ohlc_invalid = (
                (df['Low'] > df['High']) |
                (df['Open'] > df['High']) |
                (df['Open'] < df['Low']) |
                (df['Close'] > df['High']) |
                (df['Close'] < df['Low'])
            )
            if ohlc_invalid.any():
                symbol_issues.append(f"OHLC consistency issues: {ohlc_invalid.sum()} bars")
            
            issues[symbol] = symbol_issues
        
        return issues
    
    def get_aligned_data(
        self,
        data: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Create a single DataFrame with aligned dates for all symbols.
        
        Returns:
            MultiIndex DataFrame with (date, symbol) as index
        """
        frames = []
        
        for symbol, df in data.items():
            df_copy = df.copy()
            df_copy['Symbol'] = symbol
            frames.append(df_copy)
        
        combined = pd.concat(frames)
        combined = combined.reset_index().rename(columns={'index': 'Date'})
        combined = combined.set_index(['Date', 'Symbol']).sort_index()
        
        return combined
    
    def clear_cache(self):
        """Clear all cached data."""
        self._data_cache.clear()
        for f in self.cache_dir.glob("*.parquet"):
            f.unlink()
        print("Cache cleared.")


def load_backtest_data() -> Dict[str, pd.DataFrame]:
    """
    Convenience function to load data for backtesting.
    
    Returns validated data for all symbols in config.
    """
    loader = DataLoader()
    data = loader.load_universe()
    
    # Validate
    issues = loader.validate_data(data)
    
    has_issues = False
    for symbol, symbol_issues in issues.items():
        if symbol_issues:
            has_issues = True
            print(f"\n⚠️  {symbol} data issues:")
            for issue in symbol_issues:
                print(f"    - {issue}")
    
    if has_issues:
        print("\n⚠️  Data quality issues detected. Review before proceeding.")
    else:
        print("\n✅ All data validated successfully.")
    
    return data


if __name__ == "__main__":
    # Test data loading
    data = load_backtest_data()
    
    print("\nData Summary:")
    print("-" * 60)
    for symbol, df in data.items():
        years = (df.index[-1] - df.index[0]).days / 365
        print(f"{symbol}: {len(df):,} bars over {years:.1f} years")
