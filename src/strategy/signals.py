"""
Signal Generator

Generates trading signals based on spread Z-score and
correlation conditions for pairs trading.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging

import sys
sys.path.append(str(__file__).rsplit('\\', 3)[0])

from config.settings import Settings, SpreadParameters


logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Trading signal types."""
    LONG_SPREAD = "long_spread"    # Long A, Short B
    SHORT_SPREAD = "short_spread"  # Short A, Long B
    EXIT = "exit"                   # Close positions
    STOP_LOSS = "stop_loss"        # Emergency exit
    NO_SIGNAL = "no_signal"        # No action


@dataclass
class Signal:
    """Trading signal container."""
    signal_type: SignalType
    pair: Tuple[str, str]
    timestamp: datetime
    zscore: float
    correlation: float
    hedge_ratio: float
    confidence: float = 1.0
    reason: str = ""
    metadata: Dict = field(default_factory=dict)


class SignalGenerator:
    """
    Generates trading signals for pairs trading.
    
    Entry conditions:
    - Long spread: Z-score <= -2.0, correlation > 0.7
    - Short spread: Z-score >= +2.0, correlation > 0.7
    
    Exit conditions:
    - Z-score returns to [-0.2, +0.2]
    - Stop loss: Z-score >= ±3.0
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize the Signal Generator.
        
        Args:
            settings: System settings
        """
        self.settings = settings
        self.params = settings.spread_params
        
        # Custom thresholds (can be overridden per pair)
        self._custom_thresholds: Dict[Tuple[str, str], Dict] = {}
    
    def set_custom_thresholds(
        self,
        pair: Tuple[str, str],
        entry_long: Optional[float] = None,
        entry_short: Optional[float] = None,
        exit_lower: Optional[float] = None,
        exit_upper: Optional[float] = None,
        stop_loss: Optional[float] = None
    ) -> None:
        """
        Set custom signal thresholds for a specific pair.
        
        Args:
            pair: Instrument pair
            entry_long: Long entry threshold
            entry_short: Short entry threshold
            exit_lower: Lower exit threshold
            exit_upper: Upper exit threshold
            stop_loss: Stop loss threshold
        """
        self._custom_thresholds[pair] = {
            'entry_long': entry_long or self.params.entry_zscore_long,
            'entry_short': entry_short or self.params.entry_zscore_short,
            'exit_lower': exit_lower or self.params.exit_zscore_lower,
            'exit_upper': exit_upper or self.params.exit_zscore_upper,
            'stop_loss': stop_loss or self.params.stop_loss_zscore
        }
    
    def get_thresholds(self, pair: Tuple[str, str]) -> Dict:
        """Get thresholds for a pair (custom or default)."""
        if pair in self._custom_thresholds:
            return self._custom_thresholds[pair]
        
        return {
            'entry_long': self.params.entry_zscore_long,
            'entry_short': self.params.entry_zscore_short,
            'exit_lower': self.params.exit_zscore_lower,
            'exit_upper': self.params.exit_zscore_upper,
            'stop_loss': self.params.stop_loss_zscore
        }
    
    def generate_entry_signal(
        self,
        pair: Tuple[str, str],
        zscore: float,
        correlation: float,
        hedge_ratio: float,
        timestamp: datetime
    ) -> Signal:
        """
        Generate entry signal based on current conditions.
        
        Args:
            pair: Instrument pair
            zscore: Current Z-score
            correlation: Current correlation
            hedge_ratio: Current hedge ratio
            timestamp: Current timestamp
            
        Returns:
            Signal object
        """
        thresholds = self.get_thresholds(pair)
        
        # Check correlation condition
        if correlation < self.params.min_correlation:
            return Signal(
                signal_type=SignalType.NO_SIGNAL,
                pair=pair,
                timestamp=timestamp,
                zscore=zscore,
                correlation=correlation,
                hedge_ratio=hedge_ratio,
                reason=f"Correlation below threshold: {correlation:.3f} < {self.params.min_correlation:.3f}"
            )
        
        # Check for long spread entry (Z-score very negative)
        if zscore <= thresholds['entry_long']:
            confidence = min(1.0, abs(zscore) / 3.0)  # Higher z-score = higher confidence
            return Signal(
                signal_type=SignalType.LONG_SPREAD,
                pair=pair,
                timestamp=timestamp,
                zscore=zscore,
                correlation=correlation,
                hedge_ratio=hedge_ratio,
                confidence=confidence,
                reason=f"Long spread entry: Z={zscore:.2f} <= {thresholds['entry_long']:.2f}"
            )
        
        # Check for short spread entry (Z-score very positive)
        if zscore >= thresholds['entry_short']:
            confidence = min(1.0, abs(zscore) / 3.0)
            return Signal(
                signal_type=SignalType.SHORT_SPREAD,
                pair=pair,
                timestamp=timestamp,
                zscore=zscore,
                correlation=correlation,
                hedge_ratio=hedge_ratio,
                confidence=confidence,
                reason=f"Short spread entry: Z={zscore:.2f} >= {thresholds['entry_short']:.2f}"
            )
        
        return Signal(
            signal_type=SignalType.NO_SIGNAL,
            pair=pair,
            timestamp=timestamp,
            zscore=zscore,
            correlation=correlation,
            hedge_ratio=hedge_ratio,
            reason=f"Z-score in neutral zone: {zscore:.2f}"
        )
    
    def generate_exit_signal(
        self,
        pair: Tuple[str, str],
        position_type: SignalType,
        zscore: float,
        correlation: float,
        hedge_ratio: float,
        timestamp: datetime,
        entry_zscore: Optional[float] = None
    ) -> Signal:
        """
        Generate exit signal for existing position.
        
        Args:
            pair: Instrument pair
            position_type: Current position type (LONG_SPREAD or SHORT_SPREAD)
            zscore: Current Z-score
            correlation: Current correlation
            hedge_ratio: Current hedge ratio
            timestamp: Current timestamp
            entry_zscore: Z-score at entry
            
        Returns:
            Signal object (EXIT, STOP_LOSS, or NO_SIGNAL)
        """
        thresholds = self.get_thresholds(pair)
        
        # Check stop loss
        if abs(zscore) >= thresholds['stop_loss']:
            return Signal(
                signal_type=SignalType.STOP_LOSS,
                pair=pair,
                timestamp=timestamp,
                zscore=zscore,
                correlation=correlation,
                hedge_ratio=hedge_ratio,
                reason=f"Stop loss triggered: |Z|={abs(zscore):.2f} >= {thresholds['stop_loss']:.2f}"
            )
        
        # Check correlation breakdown
        if correlation < self.params.min_correlation - 0.1:
            return Signal(
                signal_type=SignalType.STOP_LOSS,
                pair=pair,
                timestamp=timestamp,
                zscore=zscore,
                correlation=correlation,
                hedge_ratio=hedge_ratio,
                reason=f"Correlation breakdown: {correlation:.3f}"
            )
        
        # Check take profit (mean reversion)
        exit_lower = thresholds['exit_lower']
        exit_upper = thresholds['exit_upper']
        
        if exit_lower <= zscore <= exit_upper:
            return Signal(
                signal_type=SignalType.EXIT,
                pair=pair,
                timestamp=timestamp,
                zscore=zscore,
                correlation=correlation,
                hedge_ratio=hedge_ratio,
                reason=f"Take profit: Z={zscore:.2f} in [{exit_lower:.2f}, {exit_upper:.2f}]"
            )
        
        # Check for position-specific exits
        if position_type == SignalType.LONG_SPREAD and entry_zscore is not None:
            # If we're long spread and z-score crosses zero significantly
            if zscore > 0 and entry_zscore < 0:
                return Signal(
                    signal_type=SignalType.EXIT,
                    pair=pair,
                    timestamp=timestamp,
                    zscore=zscore,
                    correlation=correlation,
                    hedge_ratio=hedge_ratio,
                    reason=f"Z-score crossed zero: {entry_zscore:.2f} -> {zscore:.2f}"
                )
        
        elif position_type == SignalType.SHORT_SPREAD and entry_zscore is not None:
            # If we're short spread and z-score crosses zero significantly
            if zscore < 0 and entry_zscore > 0:
                return Signal(
                    signal_type=SignalType.EXIT,
                    pair=pair,
                    timestamp=timestamp,
                    zscore=zscore,
                    correlation=correlation,
                    hedge_ratio=hedge_ratio,
                    reason=f"Z-score crossed zero: {entry_zscore:.2f} -> {zscore:.2f}"
                )
        
        return Signal(
            signal_type=SignalType.NO_SIGNAL,
            pair=pair,
            timestamp=timestamp,
            zscore=zscore,
            correlation=correlation,
            hedge_ratio=hedge_ratio,
            reason="No exit condition met"
        )
    
    def generate_signals_batch(
        self,
        spread_data: pd.DataFrame,
        correlation_data: pd.Series,
        pair: Tuple[str, str]
    ) -> pd.DataFrame:
        """
        Generate signals for a batch of data (for backtesting).
        
        Args:
            spread_data: DataFrame with 'zscore' and 'hedge_ratio'
            correlation_data: Series of rolling correlations
            pair: Instrument pair
            
        Returns:
            DataFrame with signals
        """
        thresholds = self.get_thresholds(pair)
        
        # Align data
        aligned = spread_data.copy()
        aligned['correlation'] = correlation_data.reindex(aligned.index).ffill()
        
        # Initialize signal column
        aligned['signal'] = SignalType.NO_SIGNAL.value
        aligned['signal_reason'] = ''
        
        # Generate signals based on conditions
        mask_corr_ok = aligned['correlation'] >= self.params.min_correlation
        
        # Entry conditions
        mask_long = (aligned['zscore'] <= thresholds['entry_long']) & mask_corr_ok
        mask_short = (aligned['zscore'] >= thresholds['entry_short']) & mask_corr_ok
        
        # Exit conditions
        mask_exit = (
            (aligned['zscore'] >= thresholds['exit_lower']) & 
            (aligned['zscore'] <= thresholds['exit_upper'])
        )
        
        # Stop loss
        mask_stop = abs(aligned['zscore']) >= thresholds['stop_loss']
        
        # Apply signals (priority: stop_loss > exit > entry)
        aligned.loc[mask_long, 'signal'] = SignalType.LONG_SPREAD.value
        aligned.loc[mask_long, 'signal_reason'] = 'Long spread entry'
        
        aligned.loc[mask_short, 'signal'] = SignalType.SHORT_SPREAD.value
        aligned.loc[mask_short, 'signal_reason'] = 'Short spread entry'
        
        aligned.loc[mask_exit, 'signal'] = SignalType.EXIT.value
        aligned.loc[mask_exit, 'signal_reason'] = 'Take profit'
        
        aligned.loc[mask_stop, 'signal'] = SignalType.STOP_LOSS.value
        aligned.loc[mask_stop, 'signal_reason'] = 'Stop loss'
        
        return aligned
    
    def filter_signals(
        self,
        signals_df: pd.DataFrame,
        min_bars_between_trades: int = 1
    ) -> pd.DataFrame:
        """
        Filter signals to prevent overtrading.
        
        Args:
            signals_df: DataFrame with signals
            min_bars_between_trades: Minimum bars between trades
            
        Returns:
            Filtered DataFrame
        """
        filtered = signals_df.copy()
        
        # Track position state
        in_position = False
        position_type = None
        last_signal_idx = -min_bars_between_trades
        
        for i, (idx, row) in enumerate(filtered.iterrows()):
            signal = row['signal']
            
            if signal in [SignalType.LONG_SPREAD.value, SignalType.SHORT_SPREAD.value]:
                if in_position:
                    # Already in position, ignore entry signal
                    filtered.loc[idx, 'signal'] = SignalType.NO_SIGNAL.value
                elif i - last_signal_idx < min_bars_between_trades:
                    # Too soon after last trade
                    filtered.loc[idx, 'signal'] = SignalType.NO_SIGNAL.value
                else:
                    in_position = True
                    position_type = signal
                    last_signal_idx = i
            
            elif signal in [SignalType.EXIT.value, SignalType.STOP_LOSS.value]:
                if not in_position:
                    # Not in position, ignore exit signal
                    filtered.loc[idx, 'signal'] = SignalType.NO_SIGNAL.value
                else:
                    in_position = False
                    position_type = None
                    last_signal_idx = i
        
        return filtered
    
    def calculate_signal_strength(
        self,
        zscore: float,
        correlation: float,
        half_life: Optional[float] = None
    ) -> float:
        """
        Calculate overall signal strength (0-1).
        
        Higher strength indicates better trading opportunity.
        
        Args:
            zscore: Current Z-score
            correlation: Current correlation
            half_life: Spread half-life
            
        Returns:
            Signal strength between 0 and 1
        """
        # Z-score component (higher abs z-score = stronger)
        zscore_strength = min(1.0, abs(zscore) / 3.0)
        
        # Correlation component
        corr_strength = min(1.0, (correlation - 0.5) / 0.5)
        
        # Half-life component (shorter = better)
        if half_life and half_life > 0:
            hl_strength = min(1.0, self.params.max_half_life / half_life)
        else:
            hl_strength = 0.5  # Neutral if unknown
        
        # Weighted combination
        strength = (
            0.5 * zscore_strength +
            0.3 * corr_strength +
            0.2 * hl_strength
        )
        
        return strength
