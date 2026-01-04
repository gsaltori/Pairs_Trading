"""
Data Manager for Pairs Trading System.

Handles:
- Data retrieval from MT5
- Caching (Parquet format)
- Preprocessing and alignment
- Multiple pair synchronization
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import logging
import hashlib

from src.data.broker_client import MT5Client, Timeframe


logger = logging.getLogger(__name__)


class DataManager:
    """
    Manages data retrieval, caching, and preprocessing for pairs trading.
    
    Features:
    - Automatic caching with Parquet format
    - Data alignment between pairs
    - Missing data handling
    - Timezone normalization
    """
    
    def __init__(
        self,
        client: MT5Client,
        cache_dir: str = "data/cache",
        cache_expiry_hours: int = 1
    ):
        """
        Initialize DataManager.
        
        Args:
            client: MT5Client instance
            cache_dir: Directory for cached data
            cache_expiry_hours: Cache expiry time in hours
        """
        self.client = client
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_expiry = timedelta(hours=cache_expiry_hours)
        
        # In-memory cache for current session
        self._memory_cache: Dict[str, pd.DataFrame] = {}
    
    def _get_cache_path(self, symbol: str, timeframe: Timeframe) -> Path:
        """Generate cache file path."""
        return self.cache_dir / f"{symbol}_{timeframe.name}.parquet"
    
    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cache file is still valid."""
        if not cache_path.exists():
            return False
        
        modified_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        return datetime.now() - modified_time < self.cache_expiry
    
    def _load_from_cache(self, cache_path: Path) -> Optional[pd.DataFrame]:
        """Load data from cache file."""
        try:
            if cache_path.exists():
                return pd.read_parquet(cache_path)
        except Exception as e:
            logger.warning(f"Failed to load cache {cache_path}: {e}")
        return None
    
    def _save_to_cache(self, data: pd.DataFrame, cache_path: Path):
        """Save data to cache file."""
        try:
            data.to_parquet(cache_path)
            logger.debug(f"Cached data to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_path}: {e}")
    
    def get_candles(
        self,
        symbol: str,
        timeframe: Timeframe,
        count: int = 500,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Get OHLC data for a symbol.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe enum
            count: Number of candles
            use_cache: Whether to use caching
            
        Returns:
            DataFrame with OHLC data
        """
        cache_key = f"{symbol}_{timeframe.name}_{count}"
        
        # Check memory cache first
        if use_cache and cache_key in self._memory_cache:
            logger.debug(f"Memory cache hit: {cache_key}")
            return self._memory_cache[cache_key].copy()
        
        # Check file cache
        cache_path = self._get_cache_path(symbol, timeframe)
        if use_cache and self._is_cache_valid(cache_path):
            data = self._load_from_cache(cache_path)
            if data is not None and len(data) >= count:
                data = data.tail(count)
                self._memory_cache[cache_key] = data
                logger.debug(f"File cache hit: {cache_path}")
                return data.copy()
        
        # Fetch from MT5
        logger.info(f"Fetching {count} candles for {symbol} {timeframe.name}")
        data = self.client.get_candles(symbol, timeframe, count)
        
        if data.empty:
            logger.warning(f"No data received for {symbol}")
            return pd.DataFrame()
        
        # Cache the data
        if use_cache:
            self._save_to_cache(data, cache_path)
            self._memory_cache[cache_key] = data
        
        return data.copy()
    
    def get_close_prices(
        self,
        symbol: str,
        timeframe: Timeframe,
        count: int = 500,
        use_cache: bool = True
    ) -> pd.Series:
        """Get close prices as Series."""
        data = self.get_candles(symbol, timeframe, count, use_cache)
        if data.empty:
            return pd.Series(dtype=float)
        return data['close']
    
    def get_pair_data(
        self,
        symbol_a: str,
        symbol_b: str,
        timeframe: Timeframe,
        count: int = 500,
        align: bool = True
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Get aligned price data for a pair.
        
        Args:
            symbol_a: First symbol
            symbol_b: Second symbol
            timeframe: Timeframe enum
            count: Number of candles
            align: Whether to align timestamps
            
        Returns:
            Tuple of (price_a, price_b) Series
        """
        # Get data for both symbols
        data_a = self.get_candles(symbol_a, timeframe, count)
        data_b = self.get_candles(symbol_b, timeframe, count)
        
        if data_a.empty or data_b.empty:
            logger.warning(f"Missing data for {symbol_a}/{symbol_b}")
            return pd.Series(dtype=float), pd.Series(dtype=float)
        
        price_a = data_a['close']
        price_b = data_b['close']
        
        if align:
            # Align on common index
            common_index = price_a.index.intersection(price_b.index)
            price_a = price_a.loc[common_index]
            price_b = price_b.loc[common_index]
            
            logger.debug(f"Aligned {len(common_index)} bars for {symbol_a}/{symbol_b}")
        
        return price_a, price_b
    
    def get_multiple_pairs_data(
        self,
        pairs: List[Tuple[str, str]],
        timeframe: Timeframe,
        count: int = 500
    ) -> Dict[Tuple[str, str], Tuple[pd.Series, pd.Series]]:
        """
        Get data for multiple pairs efficiently.
        
        Args:
            pairs: List of (symbol_a, symbol_b) tuples
            timeframe: Timeframe enum
            count: Number of candles
            
        Returns:
            Dictionary mapping pairs to (price_a, price_b) tuples
        """
        # Collect unique symbols
        symbols = set()
        for a, b in pairs:
            symbols.add(a)
            symbols.add(b)
        
        # Fetch all symbols
        symbol_data = {}
        for symbol in symbols:
            data = self.get_candles(symbol, timeframe, count)
            if not data.empty:
                symbol_data[symbol] = data['close']
        
        # Build pair data
        result = {}
        for pair in pairs:
            symbol_a, symbol_b = pair
            
            if symbol_a not in symbol_data or symbol_b not in symbol_data:
                logger.warning(f"Missing data for pair {symbol_a}/{symbol_b}")
                continue
            
            price_a = symbol_data[symbol_a]
            price_b = symbol_data[symbol_b]
            
            # Align timestamps
            common_index = price_a.index.intersection(price_b.index)
            result[pair] = (
                price_a.loc[common_index],
                price_b.loc[common_index]
            )
        
        return result
    
    def get_historical_data(
        self,
        symbol: str,
        timeframe: Timeframe,
        start_date: datetime,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get historical data for a specific date range.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe enum
            start_date: Start datetime
            end_date: End datetime (None = now)
            
        Returns:
            DataFrame with OHLC data
        """
        if end_date is None:
            end_date = datetime.now()
        
        return self.client.get_candles_range(symbol, timeframe, start_date, end_date)
    
    def preprocess_data(
        self,
        data: pd.DataFrame,
        fill_method: str = 'ffill',
        remove_outliers: bool = False,
        outlier_std: float = 4.0
    ) -> pd.DataFrame:
        """
        Preprocess OHLC data.
        
        Args:
            data: Input DataFrame
            fill_method: Method for filling NaN ('ffill', 'interpolate')
            remove_outliers: Whether to remove outliers
            outlier_std: Standard deviations for outlier detection
            
        Returns:
            Preprocessed DataFrame
        """
        df = data.copy()
        
        # Handle missing values
        if fill_method == 'ffill':
            df = df.ffill()
        elif fill_method == 'interpolate':
            df = df.interpolate(method='time')
        
        # Remove remaining NaN
        df = df.dropna()
        
        # Remove outliers if requested
        if remove_outliers and 'close' in df.columns:
            returns = df['close'].pct_change()
            mean = returns.mean()
            std = returns.std()
            
            mask = abs(returns - mean) < outlier_std * std
            df = df[mask]
        
        return df
    
    def calculate_returns(
        self,
        prices: pd.Series,
        method: str = 'log'
    ) -> pd.Series:
        """
        Calculate returns from prices.
        
        Args:
            prices: Price series
            method: 'log' or 'simple'
            
        Returns:
            Returns series
        """
        if method == 'log':
            return np.log(prices / prices.shift(1))
        else:
            return prices.pct_change()
    
    def synchronize_series(
        self,
        series_list: List[pd.Series],
        method: str = 'inner'
    ) -> List[pd.Series]:
        """
        Synchronize multiple series on common timestamps.
        
        Args:
            series_list: List of Series to synchronize
            method: 'inner' (common only) or 'outer' (all timestamps)
            
        Returns:
            List of synchronized Series
        """
        if not series_list:
            return []
        
        # Find common index
        if method == 'inner':
            common_index = series_list[0].index
            for s in series_list[1:]:
                common_index = common_index.intersection(s.index)
        else:
            common_index = series_list[0].index
            for s in series_list[1:]:
                common_index = common_index.union(s.index)
        
        return [s.reindex(common_index) for s in series_list]
    
    def clear_cache(self, symbol: Optional[str] = None):
        """Clear cached data."""
        if symbol:
            # Clear specific symbol
            patterns = [f"{symbol}_*.parquet"]
            keys_to_remove = [k for k in self._memory_cache if symbol in k]
        else:
            # Clear all
            patterns = ["*.parquet"]
            keys_to_remove = list(self._memory_cache.keys())
        
        # Clear memory cache
        for key in keys_to_remove:
            del self._memory_cache[key]
        
        # Clear file cache
        for pattern in patterns:
            for f in self.cache_dir.glob(pattern):
                f.unlink()
                logger.debug(f"Deleted cache file: {f}")
    
    def get_spread_info(self, symbol: str) -> Optional[float]:
        """Get current spread for symbol in pips."""
        tick = self.client.get_tick(symbol)
        if tick is None:
            return None
        
        info = self.client.get_symbol_info(symbol)
        if info is None:
            return None
        
        spread_points = tick['ask'] - tick['bid']
        spread_pips = spread_points / info.point / 10  # Convert to pips
        
        return spread_pips
    
    def validate_pair_data(
        self,
        price_a: pd.Series,
        price_b: pd.Series,
        min_bars: int = 100
    ) -> Tuple[bool, str]:
        """
        Validate pair data quality.
        
        Args:
            price_a: First price series
            price_b: Second price series
            min_bars: Minimum required bars
            
        Returns:
            (is_valid, message)
        """
        if len(price_a) < min_bars:
            return False, f"Insufficient data for first symbol: {len(price_a)} < {min_bars}"
        
        if len(price_b) < min_bars:
            return False, f"Insufficient data for second symbol: {len(price_b)} < {min_bars}"
        
        if len(price_a) != len(price_b):
            return False, f"Length mismatch: {len(price_a)} vs {len(price_b)}"
        
        # Check for excessive NaN
        nan_pct_a = price_a.isna().sum() / len(price_a)
        nan_pct_b = price_b.isna().sum() / len(price_b)
        
        if nan_pct_a > 0.05:
            return False, f"Too many NaN in first symbol: {nan_pct_a:.1%}"
        
        if nan_pct_b > 0.05:
            return False, f"Too many NaN in second symbol: {nan_pct_b:.1%}"
        
        return True, "Data validation passed"
