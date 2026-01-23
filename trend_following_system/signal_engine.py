"""
Trend Following System - Signal Engine
Entry and exit signal generation with strict rules.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List
from enum import Enum

from config import TRADING_CONFIG


class SignalType(Enum):
    """Signal types."""
    ENTRY = "ENTRY"
    EXIT = "EXIT"
    NONE = "NONE"


@dataclass
class Signal:
    """Trading signal with all required information."""
    date: pd.Timestamp
    symbol: str
    signal_type: SignalType
    price: float                    # Signal price (close)
    entry_price: Optional[float]    # For exits: original entry price
    stop_price: float               # Current trailing stop
    atr: float                      # For position sizing
    
    def __repr__(self):
        return f"Signal({self.signal_type.value}, {self.symbol}, {self.date.date()}, ${self.price:.2f})"


class SignalEngine:
    """
    Generates entry and exit signals based on trend following rules.
    
    ENTRY CONDITIONS (ALL required):
    1. Close > EMA(200)
    2. Close > Highest High of last 55 days
    3. ATR >= Median ATR of last 252 days
    4. No existing position
    
    EXIT CONDITIONS (ANY triggers exit):
    1. Close < Lowest Low of last 20 days (trailing stop)
    """
    
    def __init__(self):
        """Initialize signal engine."""
        self._positions: Dict[str, dict] = {}  # Current positions
    
    def check_entry(
        self,
        date: pd.Timestamp,
        symbol: str,
        row: pd.Series,
    ) -> Optional[Signal]:
        """
        Check if entry conditions are met.
        
        Args:
            date: Current date
            symbol: Symbol to check
            row: DataFrame row with indicators
            
        Returns:
            Entry Signal if conditions met, None otherwise
        """
        # Already in position
        if symbol in self._positions:
            return None
        
        # Check all entry conditions
        above_ema = row.get('Above_EMA', False)
        breakout = row.get('Breakout', False)
        volatility_ok = row.get('Volatility_OK', False)
        
        # All conditions must be True
        if not (above_ema and breakout and volatility_ok):
            return None
        
        # Valid entry signal
        return Signal(
            date=date,
            symbol=symbol,
            signal_type=SignalType.ENTRY,
            price=row['Close'],
            entry_price=None,
            stop_price=row['Trailing_Stop'],
            atr=row['ATR'],
        )
    
    def check_exit(
        self,
        date: pd.Timestamp,
        symbol: str,
        row: pd.Series,
    ) -> Optional[Signal]:
        """
        Check if exit conditions are met.
        
        Args:
            date: Current date
            symbol: Symbol to check
            row: DataFrame row with indicators
            
        Returns:
            Exit Signal if conditions met, None otherwise
        """
        # Not in position
        if symbol not in self._positions:
            return None
        
        position = self._positions[symbol]
        trailing_stop = row['Trailing_Stop']
        
        # Exit if close breaches trailing stop
        if row['Close'] < trailing_stop:
            return Signal(
                date=date,
                symbol=symbol,
                signal_type=SignalType.EXIT,
                price=row['Close'],
                entry_price=position['entry_price'],
                stop_price=trailing_stop,
                atr=row['ATR'],
            )
        
        return None
    
    def process_bar(
        self,
        date: pd.Timestamp,
        symbol: str,
        row: pd.Series,
    ) -> Optional[Signal]:
        """
        Process a single bar and return signal if any.
        
        Priority: Exit signals checked before entry signals.
        
        Args:
            date: Current date
            symbol: Symbol
            row: DataFrame row with OHLCV and indicators
            
        Returns:
            Signal if generated, None otherwise
        """
        # Check for missing indicator values
        if pd.isna(row.get('EMA_200')) or pd.isna(row.get('ATR')):
            return None
        
        # Exit check first (always)
        exit_signal = self.check_exit(date, symbol, row)
        if exit_signal:
            return exit_signal
        
        # Then entry check
        entry_signal = self.check_entry(date, symbol, row)
        if entry_signal:
            return entry_signal
        
        return None
    
    def register_entry(
        self,
        symbol: str,
        entry_price: float,
        entry_date: pd.Timestamp,
        shares: int,
        stop_price: float,
    ):
        """Register a new position after entry execution."""
        self._positions[symbol] = {
            'entry_price': entry_price,
            'entry_date': entry_date,
            'shares': shares,
            'stop_price': stop_price,
        }
    
    def register_exit(self, symbol: str):
        """Remove position after exit execution."""
        if symbol in self._positions:
            del self._positions[symbol]
    
    def get_position(self, symbol: str) -> Optional[dict]:
        """Get current position for symbol."""
        return self._positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, dict]:
        """Get all current positions."""
        return self._positions.copy()
    
    def position_count(self) -> int:
        """Get number of open positions."""
        return len(self._positions)
    
    def has_position(self, symbol: str) -> bool:
        """Check if symbol has open position."""
        return symbol in self._positions
    
    def reset(self):
        """Clear all positions (for backtesting)."""
        self._positions.clear()


def generate_signals_vectorized(
    data: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Generate all signals vectorized (for analysis, not live trading).
    
    This is useful for understanding signal distribution but NOT for
    backtesting (which must be done sequentially due to position limits).
    
    Args:
        data: Dict mapping symbol to DataFrame with indicators
        
    Returns:
        DataFrame with all potential entry signals
    """
    signals = []
    
    for symbol, df in data.items():
        # Entry signals (all conditions met)
        entries = df[
            df['Above_EMA'] & 
            df['Breakout'] & 
            df['Volatility_OK']
        ].copy()
        
        if len(entries) > 0:
            entries['Symbol'] = symbol
            entries['Signal'] = 'ENTRY'
            signals.append(entries)
    
    if not signals:
        return pd.DataFrame()
    
    result = pd.concat(signals)
    result = result.sort_index()
    
    return result


if __name__ == "__main__":
    # Test signal engine
    from data_loader import load_backtest_data
    from indicators import add_indicators
    
    print("Loading data...")
    data = load_backtest_data()
    
    print("Calculating indicators...")
    data = add_indicators(data)
    
    print("\nAnalyzing potential entry signals...")
    signals_df = generate_signals_vectorized(data)
    
    print(f"\nTotal potential entry signals: {len(signals_df)}")
    
    # Distribution by symbol
    if len(signals_df) > 0:
        print("\nSignals by symbol:")
        print(signals_df.groupby('Symbol').size())
        
        # Distribution by year
        signals_df['Year'] = signals_df.index.year
        print("\nSignals by year:")
        print(signals_df.groupby('Year').size())
