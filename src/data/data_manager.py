"""
Data Manager

Handles data fetching, caching, and preprocessing for the Pairs Trading System.
Provides a unified interface for accessing market data from any source.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import pickle
import hashlib
import logging
import json

import sys
sys.path.append(str(__file__).rsplit('\\', 3)[0])

from config.settings import Settings, Timeframe
from config.broker_config import BrokerConfig
from src.data.broker_client import OandaClient


logger = logging.getLogger(__name__)


class DataManager:
    """
    Manages market data for the Pairs Trading System.
    
    Responsibilities:
    - Fetching data from broker
    - Caching data to disk
    - Data validation and preprocessing
    - Providing unified access to data
    """
    
    def __init__(self, settings: Settings, broker_config: Optional[BrokerConfig] = None):
        """
        Initialize the Data Manager.
        
        Args:
            settings: System settings
            broker_config: Broker configuration (uses env vars if None)
        """
        self.settings = settings
        self.broker_config = broker_config or BrokerConfig.from_env()
        self._client: Optional[OandaClient] = None
        self._data_cache: Dict[str, pd.DataFrame] = {}
        
        # Ensure directories exist
        self.settings.historical_dir.mkdir(parents=True, exist_ok=True)
        self.settings.cache_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def client(self) -> OandaClient:
        """Get or create OANDA client."""
        if self._client is None:
            self._client = OandaClient(self.broker_config)
        return self._client
    
    def _get_cache_key(self, instrument: str, timeframe: Timeframe, start: datetime, end: datetime) -> str:
        """Generate unique cache key for data request."""
        key_str = f"{instrument}_{timeframe.value}_{start.date()}_{end.date()}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cache_path(self, instrument: str, timeframe: Timeframe) -> Path:
        """Get path to cached data file."""
        return self.settings.historical_dir / f"{instrument}_{timeframe.value}.parquet"
    
    def fetch_data(
        self,
        instrument: str,
        timeframe: Optional[Timeframe] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical data for an instrument.
        
        Args:
            instrument: Instrument name (e.g., 'EUR_USD')
            timeframe: Timeframe (default from settings)
            start_date: Start date (default: lookback_days ago)
            end_date: End date (default: now)
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with OHLCV data
        """
        timeframe = timeframe or self.settings.timeframe
        end_date = end_date or datetime.utcnow()
        start_date = start_date or (end_date - timedelta(days=self.settings.lookback_days))
        
        # Check memory cache first
        cache_key = f"{instrument}_{timeframe.value}"
        if use_cache and cache_key in self._data_cache:
            cached_df = self._data_cache[cache_key]
            if not cached_df.empty:
                # Check if cached data covers the requested range
                if cached_df.index[0] <= start_date and cached_df.index[-1] >= end_date - timedelta(hours=1):
                    return cached_df[(cached_df.index >= start_date) & (cached_df.index <= end_date)].copy()
        
        # Check disk cache
        cache_path = self._get_cache_path(instrument, timeframe)
        if use_cache and cache_path.exists():
            logger.info(f"Loading cached data for {instrument}")
            cached_df = pd.read_parquet(cache_path)
            
            # Update if needed
            if not cached_df.empty and cached_df.index[-1] < end_date - timedelta(hours=1):
                logger.info(f"Updating cached data for {instrument}")
                new_data = self.client.get_historical_data(
                    instrument=instrument,
                    timeframe=timeframe,
                    start_date=cached_df.index[-1],
                    end_date=end_date
                )
                
                if not new_data.empty:
                    cached_df = pd.concat([cached_df, new_data])
                    cached_df = cached_df[~cached_df.index.duplicated(keep='last')]
                    cached_df.sort_index(inplace=True)
                    cached_df.to_parquet(cache_path)
            
            self._data_cache[cache_key] = cached_df
            return cached_df[(cached_df.index >= start_date) & (cached_df.index <= end_date)].copy()
        
        # Fetch from broker
        logger.info(f"Fetching {instrument} data from broker")
        df = self.client.get_historical_data(
            instrument=instrument,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        if not df.empty:
            # Cache to disk
            df.to_parquet(cache_path)
            # Cache in memory
            self._data_cache[cache_key] = df
        
        return df
    
    def fetch_pair_data(
        self,
        pair: Tuple[str, str],
        timeframe: Optional[Timeframe] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        use_cache: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch data for a pair of instruments.
        
        Args:
            pair: Tuple of instrument names
            timeframe: Timeframe
            start_date: Start date
            end_date: End date
            use_cache: Whether to use cache
            
        Returns:
            Tuple of DataFrames for each instrument
        """
        instrument_a, instrument_b = pair
        
        df_a = self.fetch_data(instrument_a, timeframe, start_date, end_date, use_cache)
        df_b = self.fetch_data(instrument_b, timeframe, start_date, end_date, use_cache)
        
        return df_a, df_b
    
    def get_aligned_prices(
        self,
        pair: Tuple[str, str],
        timeframe: Optional[Timeframe] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        price_col: str = 'close'
    ) -> pd.DataFrame:
        """
        Get aligned prices for a pair of instruments.
        
        Args:
            pair: Tuple of instrument names
            timeframe: Timeframe
            start_date: Start date
            end_date: End date
            price_col: Column to use ('open', 'high', 'low', 'close')
            
        Returns:
            DataFrame with aligned prices for both instruments
        """
        df_a, df_b = self.fetch_pair_data(pair, timeframe, start_date, end_date)
        
        # Align indices
        aligned = pd.DataFrame({
            pair[0]: df_a[price_col],
            pair[1]: df_b[price_col]
        })
        
        # Drop any rows with missing data
        aligned.dropna(inplace=True)
        
        return aligned
    
    def get_all_pairs_data(
        self,
        timeframe: Optional[Timeframe] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[Tuple[str, str], pd.DataFrame]:
        """
        Fetch aligned data for all pairs in the universe.
        
        Returns:
            Dictionary mapping pair tuple to aligned price DataFrame
        """
        result = {}
        
        for pair in self.settings.pairs_universe:
            logger.info(f"Fetching data for pair: {pair}")
            try:
                aligned = self.get_aligned_prices(pair, timeframe, start_date, end_date)
                if len(aligned) > 0:
                    result[pair] = aligned
                else:
                    logger.warning(f"No data for pair: {pair}")
            except Exception as e:
                logger.error(f"Error fetching data for {pair}: {e}")
        
        return result
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess raw OHLCV data.
        
        Steps:
        - Handle missing values
        - Remove outliers
        - Calculate returns
        
        Args:
            df: Raw OHLCV DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        df = df.copy()
        
        # Forward fill small gaps (up to 3 bars)
        df = df.ffill(limit=3)
        
        # Drop remaining NaN
        df.dropna(inplace=True)
        
        # Remove obvious outliers (price changes > 5% in one bar)
        if 'close' in df.columns:
            returns = df['close'].pct_change()
            df = df[returns.abs() < 0.05]
        
        # Calculate log returns
        if 'close' in df.columns:
            df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        return df
    
    def validate_data(self, df: pd.DataFrame, min_bars: int = 100) -> bool:
        """
        Validate data quality.
        
        Args:
            df: DataFrame to validate
            min_bars: Minimum required bars
            
        Returns:
            True if data passes validation
        """
        if df.empty:
            logger.warning("Empty DataFrame")
            return False
        
        if len(df) < min_bars:
            logger.warning(f"Insufficient data: {len(df)} bars (min: {min_bars})")
            return False
        
        # Check for gaps
        if isinstance(df.index, pd.DatetimeIndex):
            time_diffs = df.index.to_series().diff()
            # More than 10% missing data is suspicious
            expected_interval = time_diffs.mode().iloc[0]
            gaps = (time_diffs > expected_interval * 2).sum()
            gap_ratio = gaps / len(df)
            
            if gap_ratio > 0.1:
                logger.warning(f"High gap ratio: {gap_ratio:.2%}")
                return False
        
        # Check for zero/negative prices
        if 'close' in df.columns:
            if (df['close'] <= 0).any():
                logger.warning("Zero or negative prices found")
                return False
        
        return True
    
    def get_current_prices(self, instruments: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Get current market prices.
        
        Args:
            instruments: List of instrument names
            
        Returns:
            Dictionary mapping instrument to price data
        """
        return self.client.get_current_prices(instruments)
    
    def clear_cache(self, instrument: Optional[str] = None) -> None:
        """
        Clear cached data.
        
        Args:
            instrument: Specific instrument to clear (None = all)
        """
        if instrument:
            # Clear memory cache
            keys_to_delete = [k for k in self._data_cache if k.startswith(instrument)]
            for k in keys_to_delete:
                del self._data_cache[k]
            
            # Clear disk cache
            for f in self.settings.historical_dir.glob(f"{instrument}_*.parquet"):
                f.unlink()
            
            logger.info(f"Cleared cache for {instrument}")
        else:
            self._data_cache.clear()
            for f in self.settings.historical_dir.glob("*.parquet"):
                f.unlink()
            logger.info("Cleared all cache")
    
    def save_metadata(self, pair: Tuple[str, str], metadata: Dict) -> None:
        """
        Save metadata for a pair (e.g., correlation stats, optimal parameters).
        
        Args:
            pair: Instrument pair
            metadata: Metadata dictionary
        """
        filename = f"{pair[0]}_{pair[1]}_metadata.json"
        filepath = self.settings.cache_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    def load_metadata(self, pair: Tuple[str, str]) -> Optional[Dict]:
        """
        Load metadata for a pair.
        
        Args:
            pair: Instrument pair
            
        Returns:
            Metadata dictionary or None
        """
        filename = f"{pair[0]}_{pair[1]}_metadata.json"
        filepath = self.settings.cache_dir / filename
        
        if filepath.exists():
            with open(filepath, 'r') as f:
                return json.load(f)
        
        return None
    
    def close(self) -> None:
        """Close broker connection."""
        if self._client:
            self._client.close()
            self._client = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
