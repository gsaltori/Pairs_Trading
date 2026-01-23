"""
Trend Following System - Technical Indicators
EMA, Donchian Channels, ATR with vectorized calculations.
"""

import pandas as pd
import numpy as np
from typing import Tuple

from config import TRADING_CONFIG


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """
    Calculate Exponential Moving Average.
    
    Args:
        series: Price series
        period: EMA period
        
    Returns:
        EMA series
    """
    return series.ewm(span=period, adjust=False).mean()


def calculate_donchian_channel(
    high: pd.Series,
    low: pd.Series,
    period: int,
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Donchian Channel (highest high, lowest low).
    
    Args:
        high: High price series
        low: Low price series
        period: Lookback period
        
    Returns:
        Tuple of (upper_band, lower_band)
    """
    upper = high.rolling(window=period).max()
    lower = low.rolling(window=period).min()
    
    return upper, lower


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int,
) -> pd.Series:
    """
    Calculate Average True Range.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: ATR period
        
    Returns:
        ATR series
    """
    # True Range components
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    # True Range is the maximum of the three
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # ATR is the EMA of True Range
    atr = tr.ewm(span=period, adjust=False).mean()
    
    return atr


def calculate_atr_median(atr: pd.Series, lookback: int) -> pd.Series:
    """
    Calculate rolling median of ATR.
    
    Used to filter out low-volatility environments.
    
    Args:
        atr: ATR series
        lookback: Rolling window for median
        
    Returns:
        Median ATR series
    """
    return atr.rolling(window=lookback).median()


class IndicatorEngine:
    """
    Calculates all required indicators for trend following system.
    
    Indicators:
    - EMA(200): Long-term trend filter
    - Donchian(55): Entry breakout channel
    - Donchian(20): Trailing stop channel
    - ATR(20): Volatility for position sizing
    - ATR Median(252): Volatility filter
    """
    
    def __init__(self):
        """Initialize with configuration parameters."""
        self.ema_period = TRADING_CONFIG.EMA_PERIOD
        self.donchian_entry = TRADING_CONFIG.DONCHIAN_ENTRY
        self.donchian_exit = TRADING_CONFIG.DONCHIAN_EXIT
        self.atr_period = TRADING_CONFIG.ATR_PERIOD
        self.atr_lookback = TRADING_CONFIG.ATR_LOOKBACK
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all indicators for a single symbol.
        
        Args:
            df: DataFrame with OHLCV columns
            
        Returns:
            DataFrame with added indicator columns
        """
        df = df.copy()
        
        # EMA for trend filter
        df['EMA_200'] = calculate_ema(df['Close'], self.ema_period)
        
        # Donchian channels for entry (55-day)
        df['Donchian_Entry_High'], _ = calculate_donchian_channel(
            df['High'], df['Low'], self.donchian_entry
        )
        
        # Use PREVIOUS day's highest high for breakout detection
        # This avoids lookahead bias
        df['Donchian_Entry_High'] = df['Donchian_Entry_High'].shift(1)
        
        # Donchian channels for exit (20-day trailing stop)
        _, df['Donchian_Exit_Low'] = calculate_donchian_channel(
            df['High'], df['Low'], self.donchian_exit
        )
        
        # Use PREVIOUS day's lowest low for trailing stop
        df['Trailing_Stop'] = df['Donchian_Exit_Low'].shift(1)
        
        # ATR for position sizing
        df['ATR'] = calculate_atr(
            df['High'], df['Low'], df['Close'], self.atr_period
        )
        
        # ATR median for volatility filter
        df['ATR_Median'] = calculate_atr_median(df['ATR'], self.atr_lookback)
        
        # Derived signals
        df['Above_EMA'] = df['Close'] > df['EMA_200']
        df['Breakout'] = df['Close'] > df['Donchian_Entry_High']
        df['Volatility_OK'] = df['ATR'] >= df['ATR_Median']
        
        return df
    
    def calculate_universe(
        self,
        data: dict,
    ) -> dict:
        """
        Calculate indicators for all symbols in universe.
        
        Args:
            data: Dict mapping symbol to DataFrame
            
        Returns:
            Dict mapping symbol to DataFrame with indicators
        """
        result = {}
        
        for symbol, df in data.items():
            result[symbol] = self.calculate_all(df)
        
        return result
    
    def get_warmup_period(self) -> int:
        """
        Get minimum bars required before indicators are valid.
        
        Returns:
            Number of bars needed for warmup
        """
        return max(
            self.ema_period,
            self.donchian_entry,
            self.atr_lookback,
        )


def add_indicators(data: dict) -> dict:
    """
    Convenience function to add indicators to all symbols.
    
    Args:
        data: Dict mapping symbol to raw OHLCV DataFrame
        
    Returns:
        Dict mapping symbol to DataFrame with indicators
    """
    engine = IndicatorEngine()
    return engine.calculate_universe(data)


if __name__ == "__main__":
    # Test indicators
    from data_loader import load_backtest_data
    
    print("Loading data...")
    data = load_backtest_data()
    
    print("\nCalculating indicators...")
    engine = IndicatorEngine()
    data_with_indicators = engine.calculate_universe(data)
    
    print(f"\nWarmup period: {engine.get_warmup_period()} bars")
    
    # Show sample
    for symbol, df in data_with_indicators.items():
        print(f"\n{symbol} - Last 5 rows:")
        cols = ['Close', 'EMA_200', 'Donchian_Entry_High', 'Trailing_Stop', 
                'ATR', 'Above_EMA', 'Breakout', 'Volatility_OK']
        print(df[cols].tail())
        break
