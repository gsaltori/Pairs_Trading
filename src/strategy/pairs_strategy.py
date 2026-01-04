"""
Pairs Trading Strategy

Main strategy class that orchestrates all components
for pairs trading execution.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging

import sys
sys.path.append(str(__file__).rsplit('\\', 3)[0])

from config.settings import Settings
from src.analysis.correlation import CorrelationAnalyzer, CorrelationResult
from src.analysis.cointegration import CointegrationAnalyzer, CointegrationResult
from src.analysis.spread_builder import SpreadBuilder, SpreadMetrics
from src.strategy.signals import SignalGenerator, Signal, SignalType


logger = logging.getLogger(__name__)


@dataclass
class PairState:
    """Current state for a tradeable pair."""
    pair: Tuple[str, str]
    spread_data: pd.DataFrame
    correlation_result: CorrelationResult
    cointegration_result: Optional[CointegrationResult]
    spread_metrics: SpreadMetrics
    current_signal: Signal
    position: Optional[SignalType] = None
    entry_zscore: Optional[float] = None
    entry_time: Optional[datetime] = None
    entry_hedge_ratio: Optional[float] = None
    is_tradeable: bool = True
    tradeable_reason: str = ""


@dataclass
class StrategyState:
    """Overall strategy state."""
    timestamp: datetime
    pairs: Dict[Tuple[str, str], PairState] = field(default_factory=dict)
    open_positions: int = 0
    total_exposure: float = 0.0


class PairsStrategy:
    """
    Main Pairs Trading Strategy.
    
    Orchestrates:
    - Pair analysis and screening
    - Signal generation
    - Position tracking
    - Strategy state management
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize the Pairs Strategy.
        
        Args:
            settings: System settings
        """
        self.settings = settings
        
        # Initialize components
        self.correlation_analyzer = CorrelationAnalyzer(settings)
        self.cointegration_analyzer = CointegrationAnalyzer(settings)
        self.spread_builder = SpreadBuilder(settings)
        self.signal_generator = SignalGenerator(settings)
        
        # State
        self._state: Optional[StrategyState] = None
        self._pair_states: Dict[Tuple[str, str], PairState] = {}
    
    def analyze_pair(
        self,
        pair: Tuple[str, str],
        aligned_prices: pd.DataFrame
    ) -> PairState:
        """
        Perform complete analysis on a pair.
        
        Args:
            pair: Instrument pair tuple
            aligned_prices: DataFrame with aligned prices
            
        Returns:
            PairState with all analysis results
        """
        logger.info(f"Analyzing pair: {pair}")
        
        # Correlation analysis
        corr_result = self.correlation_analyzer.analyze_pair(aligned_prices)
        
        # Cointegration analysis
        try:
            coint_result = self.cointegration_analyzer.analyze_pair(aligned_prices)
        except Exception as e:
            logger.warning(f"Cointegration analysis failed for {pair}: {e}")
            coint_result = None
        
        # Build spread
        spread_data = self.spread_builder.build_spread_with_zscore(
            aligned_prices.iloc[:, 0],
            aligned_prices.iloc[:, 1]
        )
        
        # Get spread metrics
        if not spread_data.empty:
            spread_metrics = self.spread_builder.get_spread_metrics(spread_data)
        else:
            spread_metrics = SpreadMetrics(
                mean=0.0, std=1.0, current_value=0.0, zscore=0.0,
                half_life=None, hedge_ratio=1.0, hurst_exponent=None
            )
        
        # Check if tradeable
        is_tradeable, tradeable_reason = self._check_pair_tradeable(
            corr_result, coint_result, spread_metrics
        )
        
        # Generate signal
        if is_tradeable and not spread_data.empty:
            current_signal = self.signal_generator.generate_entry_signal(
                pair=pair,
                zscore=spread_metrics.zscore,
                correlation=corr_result.current_correlation,
                hedge_ratio=spread_metrics.hedge_ratio,
                timestamp=spread_data.index[-1]
            )
        else:
            current_signal = Signal(
                signal_type=SignalType.NO_SIGNAL,
                pair=pair,
                timestamp=datetime.now(),
                zscore=spread_metrics.zscore,
                correlation=corr_result.current_correlation,
                hedge_ratio=spread_metrics.hedge_ratio,
                reason=tradeable_reason
            )
        
        return PairState(
            pair=pair,
            spread_data=spread_data,
            correlation_result=corr_result,
            cointegration_result=coint_result,
            spread_metrics=spread_metrics,
            current_signal=current_signal,
            is_tradeable=is_tradeable,
            tradeable_reason=tradeable_reason
        )
    
    def _check_pair_tradeable(
        self,
        corr_result: CorrelationResult,
        coint_result: Optional[CointegrationResult],
        spread_metrics: SpreadMetrics
    ) -> Tuple[bool, str]:
        """
        Check if a pair meets all trading criteria.
        
        Returns:
            Tuple of (is_tradeable, reason)
        """
        # Check correlation
        if corr_result.current_correlation < self.settings.spread_params.min_correlation:
            return False, f"Low correlation: {corr_result.current_correlation:.3f}"
        
        if not corr_result.is_stable:
            return False, "Unstable correlation"
        
        # Check cointegration
        if coint_result and not coint_result.is_cointegrated:
            return False, f"Not cointegrated (p={coint_result.p_value:.4f})"
        
        # Check half-life
        if spread_metrics.half_life is not None:
            if spread_metrics.half_life > self.settings.spread_params.max_half_life:
                return False, f"Half-life too long: {spread_metrics.half_life:.1f}"
        
        # Check Hurst exponent
        if spread_metrics.hurst_exponent is not None:
            if spread_metrics.hurst_exponent > 0.5:
                return False, f"Spread is trending (H={spread_metrics.hurst_exponent:.3f})"
        
        return True, "Pair is tradeable"
    
    def screen_pairs(
        self,
        all_pairs_data: Dict[Tuple[str, str], pd.DataFrame]
    ) -> List[PairState]:
        """
        Screen all pairs and return tradeable ones.
        
        Args:
            all_pairs_data: Dictionary of aligned prices for each pair
            
        Returns:
            List of PairState for tradeable pairs
        """
        tradeable_pairs = []
        
        for pair, aligned_prices in all_pairs_data.items():
            try:
                state = self.analyze_pair(pair, aligned_prices)
                self._pair_states[pair] = state
                
                if state.is_tradeable:
                    tradeable_pairs.append(state)
                    logger.info(f"Pair {pair} is tradeable")
                else:
                    logger.debug(f"Pair {pair} not tradeable: {state.tradeable_reason}")
                    
            except Exception as e:
                logger.error(f"Error analyzing pair {pair}: {e}")
        
        # Sort by signal strength
        tradeable_pairs.sort(
            key=lambda x: self.signal_generator.calculate_signal_strength(
                x.spread_metrics.zscore,
                x.correlation_result.current_correlation,
                x.spread_metrics.half_life
            ),
            reverse=True
        )
        
        return tradeable_pairs
    
    def update_pair(
        self,
        pair: Tuple[str, str],
        new_price_a: float,
        new_price_b: float,
        timestamp: datetime
    ) -> PairState:
        """
        Update pair state with new prices.
        
        Args:
            pair: Instrument pair
            new_price_a: New price for instrument A
            new_price_b: New price for instrument B
            timestamp: Current timestamp
            
        Returns:
            Updated PairState
        """
        if pair not in self._pair_states:
            raise ValueError(f"Pair {pair} not initialized")
        
        state = self._pair_states[pair]
        
        # Update spread data
        state.spread_data = self.spread_builder.update_spread(
            state.spread_data,
            new_price_a,
            new_price_b,
            timestamp
        )
        
        # Update metrics
        state.spread_metrics = self.spread_builder.get_spread_metrics(state.spread_data)
        
        # Update correlation (periodically)
        # For real-time, you might update less frequently
        
        # Generate new signal
        if state.position is None:
            # No position - check for entry
            state.current_signal = self.signal_generator.generate_entry_signal(
                pair=pair,
                zscore=state.spread_metrics.zscore,
                correlation=state.correlation_result.current_correlation,
                hedge_ratio=state.spread_metrics.hedge_ratio,
                timestamp=timestamp
            )
        else:
            # In position - check for exit
            state.current_signal = self.signal_generator.generate_exit_signal(
                pair=pair,
                position_type=state.position,
                zscore=state.spread_metrics.zscore,
                correlation=state.correlation_result.current_correlation,
                hedge_ratio=state.spread_metrics.hedge_ratio,
                timestamp=timestamp,
                entry_zscore=state.entry_zscore
            )
        
        return state
    
    def enter_position(
        self,
        pair: Tuple[str, str],
        signal: Signal
    ) -> None:
        """
        Record position entry.
        
        Args:
            pair: Instrument pair
            signal: Entry signal
        """
        if pair not in self._pair_states:
            raise ValueError(f"Pair {pair} not initialized")
        
        state = self._pair_states[pair]
        state.position = signal.signal_type
        state.entry_zscore = signal.zscore
        state.entry_time = signal.timestamp
        state.entry_hedge_ratio = signal.hedge_ratio
        
        logger.info(
            f"Entered position for {pair}: {signal.signal_type.value} "
            f"at Z={signal.zscore:.2f}, hedge_ratio={signal.hedge_ratio:.4f}"
        )
    
    def exit_position(
        self,
        pair: Tuple[str, str],
        signal: Signal
    ) -> Dict[str, Any]:
        """
        Record position exit and return trade info.
        
        Args:
            pair: Instrument pair
            signal: Exit signal
            
        Returns:
            Trade information dictionary
        """
        if pair not in self._pair_states:
            raise ValueError(f"Pair {pair} not initialized")
        
        state = self._pair_states[pair]
        
        trade_info = {
            'pair': pair,
            'entry_type': state.position.value if state.position else None,
            'entry_zscore': state.entry_zscore,
            'entry_time': state.entry_time,
            'entry_hedge_ratio': state.entry_hedge_ratio,
            'exit_type': signal.signal_type.value,
            'exit_zscore': signal.zscore,
            'exit_time': signal.timestamp,
            'exit_reason': signal.reason
        }
        
        # Reset position state
        state.position = None
        state.entry_zscore = None
        state.entry_time = None
        state.entry_hedge_ratio = None
        
        logger.info(
            f"Exited position for {pair}: {signal.signal_type.value} "
            f"at Z={signal.zscore:.2f}, reason: {signal.reason}"
        )
        
        return trade_info
    
    def get_state(self) -> StrategyState:
        """
        Get current strategy state.
        
        Returns:
            StrategyState object
        """
        open_positions = sum(
            1 for state in self._pair_states.values()
            if state.position is not None
        )
        
        return StrategyState(
            timestamp=datetime.now(),
            pairs=self._pair_states.copy(),
            open_positions=open_positions
        )
    
    def get_active_signals(self) -> List[Signal]:
        """
        Get all current active signals.
        
        Returns:
            List of non-NO_SIGNAL signals
        """
        return [
            state.current_signal
            for state in self._pair_states.values()
            if state.current_signal.signal_type != SignalType.NO_SIGNAL
        ]
    
    def get_pair_summary(self, pair: Tuple[str, str]) -> Dict:
        """
        Get summary information for a pair.
        
        Args:
            pair: Instrument pair
            
        Returns:
            Summary dictionary
        """
        if pair not in self._pair_states:
            raise ValueError(f"Pair {pair} not initialized")
        
        state = self._pair_states[pair]
        
        return {
            'pair': pair,
            'is_tradeable': state.is_tradeable,
            'tradeable_reason': state.tradeable_reason,
            'correlation': state.correlation_result.current_correlation,
            'correlation_stable': state.correlation_result.is_stable,
            'is_cointegrated': state.cointegration_result.is_cointegrated if state.cointegration_result else None,
            'half_life': state.spread_metrics.half_life,
            'hurst_exponent': state.spread_metrics.hurst_exponent,
            'current_zscore': state.spread_metrics.zscore,
            'hedge_ratio': state.spread_metrics.hedge_ratio,
            'current_signal': state.current_signal.signal_type.value,
            'in_position': state.position is not None,
            'position_type': state.position.value if state.position else None,
            'entry_zscore': state.entry_zscore
        }
    
    def reset(self) -> None:
        """Reset strategy state."""
        self._pair_states.clear()
        self._state = None
        logger.info("Strategy state reset")
