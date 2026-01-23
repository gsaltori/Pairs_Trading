"""
Cross-Sectional Momentum System - Momentum Engine
Handles momentum calculation, ranking, and asset selection.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from config import CONFIG


@dataclass
class MomentumSignal:
    """Signal for a single rebalance date."""
    date: pd.Timestamp
    rankings: Dict[str, float]       # symbol -> momentum score
    selected: List[str]              # Top N assets passing filter
    weights: Dict[str, float]        # symbol -> target weight
    trend_filter: Dict[str, bool]    # symbol -> passes EMA filter
    cash_weight: float               # Weight to hold in cash
    
    def __repr__(self):
        selected_str = ", ".join(self.selected) if self.selected else "NONE"
        return f"MomentumSignal({self.date.date()}, [{selected_str}], cash={self.cash_weight:.1%})"


class MomentumEngine:
    """
    Cross-sectional momentum ranking and selection engine.
    
    Logic:
    1. Calculate 12-month momentum for all assets
    2. Rank assets by momentum (highest = best)
    3. Apply trend filter: Close > EMA(200)
    4. Select top N assets that pass filter
    5. Equal weight among selected
    6. Remaining capital in cash if fewer than N qualify
    """
    
    def __init__(self):
        """Initialize momentum engine."""
        self.lookback = CONFIG.MOMENTUM_LOOKBACK
        self.trend_period = CONFIG.TREND_FILTER_PERIOD
        self.top_n = CONFIG.TOP_N_ASSETS
        self.weight_per_position = CONFIG.WEIGHT_PER_POSITION
    
    def calculate_momentum(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate 12-month momentum for all assets.
        
        momentum = (Price[t] / Price[t-252]) - 1
        """
        return prices / prices.shift(self.lookback) - 1
    
    def calculate_ema(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Calculate EMA(200) for trend filter."""
        return prices.ewm(span=self.trend_period, adjust=False).mean()
    
    def get_trend_filter(
        self,
        prices: pd.DataFrame,
        ema: pd.DataFrame,
        date: pd.Timestamp,
    ) -> Dict[str, bool]:
        """
        Check if each asset passes trend filter (Close > EMA).
        
        Returns:
            Dict mapping symbol to pass/fail
        """
        result = {}
        
        for symbol in prices.columns:
            price = prices.loc[date, symbol]
            ema_val = ema.loc[date, symbol]
            
            if pd.isna(price) or pd.isna(ema_val):
                result[symbol] = False
            else:
                result[symbol] = price > ema_val
        
        return result
    
    def rank_assets(
        self,
        momentum: pd.DataFrame,
        date: pd.Timestamp,
    ) -> Dict[str, float]:
        """
        Rank all assets by momentum on given date.
        
        Returns:
            Dict mapping symbol to momentum value, sorted descending
        """
        mom_values = momentum.loc[date].dropna()
        sorted_mom = mom_values.sort_values(ascending=False)
        
        return dict(sorted_mom)
    
    def select_assets(
        self,
        rankings: Dict[str, float],
        trend_filter: Dict[str, bool],
    ) -> List[str]:
        """
        Select top N assets that pass trend filter.
        
        Args:
            rankings: Momentum rankings (sorted descending)
            trend_filter: Dict of symbol -> passes trend filter
            
        Returns:
            List of selected symbols (up to top_n)
        """
        selected = []
        
        for symbol in rankings.keys():
            if trend_filter.get(symbol, False):
                selected.append(symbol)
                
                if len(selected) >= self.top_n:
                    break
        
        return selected
    
    def calculate_weights(
        self,
        selected: List[str],
    ) -> Tuple[Dict[str, float], float]:
        """
        Calculate target weights for selected assets.
        
        Args:
            selected: List of selected symbols
            
        Returns:
            Tuple of (weights dict, cash weight)
        """
        if not selected:
            return {}, 1.0
        
        weight = 1.0 / self.top_n  # Equal weight based on max positions
        
        weights = {symbol: weight for symbol in selected}
        cash_weight = 1.0 - sum(weights.values())
        
        return weights, cash_weight
    
    def generate_signal(
        self,
        prices: pd.DataFrame,
        momentum: pd.DataFrame,
        ema: pd.DataFrame,
        date: pd.Timestamp,
    ) -> MomentumSignal:
        """
        Generate momentum signal for a single rebalance date.
        
        Args:
            prices: Price DataFrame
            momentum: Momentum DataFrame
            ema: EMA DataFrame
            date: Rebalance date
            
        Returns:
            MomentumSignal with all selection details
        """
        # Get rankings
        rankings = self.rank_assets(momentum, date)
        
        # Get trend filter status
        trend_filter = self.get_trend_filter(prices, ema, date)
        
        # Select assets
        selected = self.select_assets(rankings, trend_filter)
        
        # Calculate weights
        weights, cash_weight = self.calculate_weights(selected)
        
        return MomentumSignal(
            date=date,
            rankings=rankings,
            selected=selected,
            weights=weights,
            trend_filter=trend_filter,
            cash_weight=cash_weight,
        )
    
    def generate_all_signals(
        self,
        prices: pd.DataFrame,
        rebalance_dates: List[pd.Timestamp],
    ) -> List[MomentumSignal]:
        """
        Generate signals for all rebalance dates.
        
        Args:
            prices: Price DataFrame
            rebalance_dates: List of month-end dates
            
        Returns:
            List of MomentumSignal objects
        """
        # Pre-calculate indicators
        momentum = self.calculate_momentum(prices)
        ema = self.calculate_ema(prices)
        
        # Skip dates before we have enough history
        warmup = max(self.lookback, self.trend_period)
        valid_dates = [d for d in rebalance_dates if d >= prices.index[warmup]]
        
        signals = []
        
        for date in valid_dates:
            signal = self.generate_signal(prices, momentum, ema, date)
            signals.append(signal)
        
        return signals


def print_signal_summary(signals: List[MomentumSignal]):
    """Print summary of signals."""
    print("\nSignal Summary:")
    print("-" * 60)
    
    # Count selections
    selection_counts = {}
    for sig in signals:
        for symbol in sig.selected:
            selection_counts[symbol] = selection_counts.get(symbol, 0) + 1
    
    print(f"Total rebalance dates: {len(signals)}")
    print(f"\nSelection frequency:")
    
    for symbol, count in sorted(selection_counts.items(), key=lambda x: -x[1]):
        pct = count / len(signals) * 100
        print(f"  {symbol}: {count} times ({pct:.1f}%)")
    
    # Cash allocation
    avg_cash = np.mean([s.cash_weight for s in signals])
    print(f"\nAverage cash allocation: {avg_cash:.1%}")


if __name__ == "__main__":
    from data_loader import load_data, DataLoader
    
    # Load data
    df = load_data()
    
    # Get rebalance dates
    loader = DataLoader()
    rebalance_dates = loader.get_monthly_dates(df)
    
    print(f"\nRebalance dates: {len(rebalance_dates)}")
    
    # Generate signals
    engine = MomentumEngine()
    signals = engine.generate_all_signals(df, rebalance_dates)
    
    print_signal_summary(signals)
    
    # Show last few signals
    print("\nLast 5 signals:")
    for sig in signals[-5:]:
        print(f"  {sig}")
