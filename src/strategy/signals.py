"""
Signal Generation Module.

Generates trading signals based on z-score and other indicators.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, Tuple, List
import logging

from config.settings import Settings


logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Types of trading signals."""
    LONG_SPREAD = "long_spread"      # Buy A, Sell B
    SHORT_SPREAD = "short_spread"    # Sell A, Buy B
    EXIT = "exit"                    # Close position
    STOP_LOSS = "stop_loss"          # Emergency exit
    NO_SIGNAL = "no_signal"


@dataclass
class Signal:
    """Trading signal with metadata."""
    type: SignalType
    pair: Tuple[str, str]
    timestamp: datetime
    zscore: float
    strength: float  # 0-1, confidence in signal
    reason: str
    hedge_ratio: Optional[float] = None
    correlation: Optional[float] = None
    half_life: Optional[float] = None


class SignalGenerator:
    """
    Generates trading signals for pairs trading.
    
    Entry signals based on:
    - Z-score thresholds
    - Correlation filter
    - Half-life validation
    
    Exit signals based on:
    - Mean reversion (z-score near zero)
    - Stop loss (extreme z-score)
    - Correlation breakdown
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize signal generator.
        
        Args:
            settings: Trading settings
        """
        self.settings = settings
        self.spread_settings = settings.spread
    
    def generate_entry_signal(
        self,
        pair: Tuple[str, str],
        zscore: float,
        correlation: float,
        hedge_ratio: float,
        half_life: float
    ) -> Optional[Signal]:
        """
        Generate entry signal if conditions are met.
        
        Args:
            pair: (symbol_a, symbol_b) tuple
            zscore: Current z-score
            correlation: Current correlation
            hedge_ratio: Current hedge ratio
            half_life: Spread half-life
            
        Returns:
            Signal or None
        """
        timestamp = datetime.now()
        
        # Correlation filter
        if correlation < self.spread_settings.min_correlation:
            logger.debug(f"Signal rejected: correlation {correlation:.2f} < {self.spread_settings.min_correlation}")
            return None
        
        # Half-life filter
        if half_life > self.spread_settings.max_half_life:
            logger.debug(f"Signal rejected: half-life {half_life:.1f} > {self.spread_settings.max_half_life}")
            return None
        
        if half_life < self.spread_settings.min_half_life:
            logger.debug(f"Signal rejected: half-life {half_life:.1f} < {self.spread_settings.min_half_life}")
            return None
        
        # Entry thresholds
        entry_threshold = self.spread_settings.entry_zscore
        
        # Long spread: z-score is very negative (spread undervalued)
        if zscore <= -entry_threshold:
            strength = min(1.0, abs(zscore) / 3.0)
            
            return Signal(
                type=SignalType.LONG_SPREAD,
                pair=pair,
                timestamp=timestamp,
                zscore=zscore,
                strength=strength,
                reason=f"Z-score {zscore:.2f} <= -{entry_threshold}",
                hedge_ratio=hedge_ratio,
                correlation=correlation,
                half_life=half_life
            )
        
        # Short spread: z-score is very positive (spread overvalued)
        if zscore >= entry_threshold:
            strength = min(1.0, abs(zscore) / 3.0)
            
            return Signal(
                type=SignalType.SHORT_SPREAD,
                pair=pair,
                timestamp=timestamp,
                zscore=zscore,
                strength=strength,
                reason=f"Z-score {zscore:.2f} >= +{entry_threshold}",
                hedge_ratio=hedge_ratio,
                correlation=correlation,
                half_life=half_life
            )
        
        return None
    
    def generate_exit_signal(
        self,
        pair: Tuple[str, str],
        zscore: float,
        correlation: float,
        current_direction: str  # 'long_spread' or 'short_spread'
    ) -> Optional[Signal]:
        """
        Generate exit signal if conditions are met.
        
        Args:
            pair: (symbol_a, symbol_b) tuple
            zscore: Current z-score
            correlation: Current correlation
            current_direction: Current position direction
            
        Returns:
            Signal or None
        """
        timestamp = datetime.now()
        
        exit_threshold = self.spread_settings.exit_zscore
        stop_threshold = self.spread_settings.stop_loss_zscore
        
        # Stop loss: extreme z-score
        if abs(zscore) >= stop_threshold:
            return Signal(
                type=SignalType.STOP_LOSS,
                pair=pair,
                timestamp=timestamp,
                zscore=zscore,
                strength=1.0,
                reason=f"Stop loss: |Z| = {abs(zscore):.2f} >= {stop_threshold}"
            )
        
        # Correlation breakdown
        if correlation < self.spread_settings.min_correlation - 0.1:
            return Signal(
                type=SignalType.EXIT,
                pair=pair,
                timestamp=timestamp,
                zscore=zscore,
                strength=0.9,
                reason=f"Correlation breakdown: {correlation:.2f}"
            )
        
        # Mean reversion exit
        if current_direction == 'long_spread':
            # Long spread profits when z-score increases toward 0
            if zscore >= -exit_threshold:
                return Signal(
                    type=SignalType.EXIT,
                    pair=pair,
                    timestamp=timestamp,
                    zscore=zscore,
                    strength=0.8,
                    reason=f"Mean reversion: Z = {zscore:.2f}"
                )
        
        elif current_direction == 'short_spread':
            # Short spread profits when z-score decreases toward 0
            if zscore <= exit_threshold:
                return Signal(
                    type=SignalType.EXIT,
                    pair=pair,
                    timestamp=timestamp,
                    zscore=zscore,
                    strength=0.8,
                    reason=f"Mean reversion: Z = {zscore:.2f}"
                )
        
        return None
    
    def evaluate_signal_strength(
        self,
        zscore: float,
        correlation: float,
        half_life: float,
        correlation_stability: float
    ) -> float:
        """
        Calculate overall signal strength (0-1).
        
        Args:
            zscore: Current z-score
            correlation: Current correlation
            half_life: Spread half-life
            correlation_stability: Correlation stability score
            
        Returns:
            Strength score [0, 1]
        """
        # Z-score component (how far from entry threshold)
        entry = self.spread_settings.entry_zscore
        zscore_score = min(1.0, (abs(zscore) - entry) / entry) if abs(zscore) >= entry else 0
        
        # Correlation component
        min_corr = self.spread_settings.min_correlation
        corr_score = (correlation - min_corr) / (1 - min_corr) if correlation >= min_corr else 0
        
        # Half-life component (prefer shorter)
        max_hl = self.spread_settings.max_half_life
        hl_score = max(0, 1 - half_life / max_hl)
        
        # Stability component
        stability_score = correlation_stability
        
        # Weighted average
        strength = (
            0.35 * zscore_score +
            0.25 * corr_score +
            0.25 * hl_score +
            0.15 * stability_score
        )
        
        return min(1.0, max(0.0, strength))
    
    def should_enter_position(
        self,
        zscore: float,
        correlation: float,
        half_life: float,
        existing_positions: int,
        max_positions: int
    ) -> Tuple[bool, str]:
        """
        Determine if a new position should be entered.
        
        Args:
            zscore: Current z-score
            correlation: Current correlation
            half_life: Spread half-life
            existing_positions: Number of open positions
            max_positions: Maximum allowed positions
            
        Returns:
            (should_enter, reason) tuple
        """
        # Position limit check
        if existing_positions >= max_positions:
            return False, f"Position limit reached: {existing_positions}/{max_positions}"
        
        # Z-score threshold
        if abs(zscore) < self.spread_settings.entry_zscore:
            return False, f"Z-score {zscore:.2f} below threshold"
        
        # Correlation check
        if correlation < self.spread_settings.min_correlation:
            return False, f"Correlation {correlation:.2f} too low"
        
        # Half-life check
        if half_life > self.spread_settings.max_half_life:
            return False, f"Half-life {half_life:.1f} too long"
        
        if half_life < self.spread_settings.min_half_life:
            return False, f"Half-life {half_life:.1f} too short"
        
        return True, "All conditions met"
    
    def generate_composite_signal(
        self,
        pair: Tuple[str, str],
        zscore: float,
        correlation: float,
        hedge_ratio: float,
        half_life: float,
        current_position: Optional[str] = None
    ) -> Signal:
        """
        Generate comprehensive signal considering all factors.
        
        Args:
            pair: (symbol_a, symbol_b) tuple
            zscore: Current z-score
            correlation: Current correlation
            hedge_ratio: Current hedge ratio
            half_life: Spread half-life
            current_position: Current position direction if any
            
        Returns:
            Signal (may be NO_SIGNAL)
        """
        timestamp = datetime.now()
        
        # If in position, check for exit
        if current_position:
            exit_signal = self.generate_exit_signal(
                pair, zscore, correlation, current_position
            )
            if exit_signal:
                return exit_signal
            
            # Stay in position
            return Signal(
                type=SignalType.NO_SIGNAL,
                pair=pair,
                timestamp=timestamp,
                zscore=zscore,
                strength=0.0,
                reason="Maintaining position"
            )
        
        # Not in position, check for entry
        entry_signal = self.generate_entry_signal(
            pair, zscore, correlation, hedge_ratio, half_life
        )
        
        if entry_signal:
            return entry_signal
        
        # No signal
        return Signal(
            type=SignalType.NO_SIGNAL,
            pair=pair,
            timestamp=timestamp,
            zscore=zscore,
            strength=0.0,
            reason="No conditions met"
        )
