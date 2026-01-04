"""
Pairs Trading Strategy Module.

Orchestrates the complete trading strategy.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Tuple
import logging

from config.settings import Settings, Timeframe
from src.data.broker_client import Timeframe as MT5Timeframe
from src.data.data_manager import DataManager
from src.analysis.correlation import CorrelationAnalyzer, CorrelationResult
from src.analysis.cointegration import CointegrationAnalyzer, CointegrationResult
from src.analysis.spread_builder import SpreadBuilder, SpreadMetrics
from src.strategy.signals import SignalGenerator, Signal, SignalType


logger = logging.getLogger(__name__)


@dataclass
class PairAnalysis:
    """Complete analysis of a trading pair."""
    pair: Tuple[str, str]
    timestamp: datetime
    correlation_result: Optional[CorrelationResult]
    cointegration_result: Optional[CointegrationResult]
    spread_metrics: Optional[SpreadMetrics]
    current_signal: Optional[Signal]
    is_tradeable: bool
    rejection_reason: Optional[str] = None


class PairsStrategy:
    """
    Main pairs trading strategy implementation.
    
    Combines:
    - Correlation analysis
    - Cointegration testing
    - Spread construction
    - Signal generation
    """
    
    def __init__(
        self,
        settings: Settings,
        data_manager: DataManager
    ):
        """
        Initialize strategy.
        
        Args:
            settings: Trading settings
            data_manager: Data manager instance
        """
        self.settings = settings
        self.data_manager = data_manager
        
        # Initialize analyzers
        self.correlation_analyzer = CorrelationAnalyzer(
            window=settings.spread.correlation_window,
            min_correlation=settings.spread.min_correlation
        )
        
        self.cointegration_analyzer = CointegrationAnalyzer()
        
        self.spread_builder = SpreadBuilder(
            regression_window=settings.spread.regression_window,
            zscore_window=settings.spread.zscore_window,
            recalculate_beta=settings.spread.recalc_hedge_ratio
        )
        
        self.signal_generator = SignalGenerator(settings)
        
        # State tracking
        self._positions: Dict[Tuple[str, str], str] = {}  # pair -> direction
        self._last_analysis: Dict[Tuple[str, str], PairAnalysis] = {}
    
    def analyze_pair(
        self,
        pair: Tuple[str, str],
        timeframe: Optional[Timeframe] = None,
        bars: int = 500
    ) -> Optional[PairAnalysis]:
        """
        Perform complete analysis on a pair.
        
        Args:
            pair: (symbol_a, symbol_b) tuple
            timeframe: Analysis timeframe
            bars: Number of bars to analyze
            
        Returns:
            PairAnalysis or None
        """
        timeframe = timeframe or self.settings.timeframe
        mt5_tf = MT5Timeframe.from_string(timeframe.value)
        
        try:
            # Get data
            price_a, price_b = self.data_manager.get_pair_data(
                pair[0], pair[1], mt5_tf, bars
            )
            
            if len(price_a) < self.settings.backtest.min_bars_required:
                return PairAnalysis(
                    pair=pair,
                    timestamp=datetime.now(),
                    correlation_result=None,
                    cointegration_result=None,
                    spread_metrics=None,
                    current_signal=None,
                    is_tradeable=False,
                    rejection_reason=f"Insufficient data: {len(price_a)} bars"
                )
            
            # Correlation analysis
            corr_result = self.correlation_analyzer.analyze_pair(price_a, price_b)
            
            if corr_result.current_correlation < self.settings.spread.min_correlation:
                return PairAnalysis(
                    pair=pair,
                    timestamp=datetime.now(),
                    correlation_result=corr_result,
                    cointegration_result=None,
                    spread_metrics=None,
                    current_signal=None,
                    is_tradeable=False,
                    rejection_reason=f"Low correlation: {corr_result.current_correlation:.3f}"
                )
            
            # Cointegration test
            coint_result = self.cointegration_analyzer.engle_granger_test(price_a, price_b)
            
            # Spread metrics
            spread_metrics = self.spread_builder.get_spread_metrics(price_a, price_b)
            
            if spread_metrics is None:
                return PairAnalysis(
                    pair=pair,
                    timestamp=datetime.now(),
                    correlation_result=corr_result,
                    cointegration_result=coint_result,
                    spread_metrics=None,
                    current_signal=None,
                    is_tradeable=False,
                    rejection_reason="Failed to calculate spread metrics"
                )
            
            # Check tradeability
            is_tradeable = True
            rejection_reason = None
            
            if not coint_result.is_cointegrated:
                is_tradeable = False
                rejection_reason = f"Not cointegrated (p={coint_result.p_value:.4f})"
            
            elif spread_metrics.half_life > self.settings.spread.max_half_life:
                is_tradeable = False
                rejection_reason = f"Half-life too long: {spread_metrics.half_life:.1f}"
            
            elif spread_metrics.half_life < self.settings.spread.min_half_life:
                is_tradeable = False
                rejection_reason = f"Half-life too short: {spread_metrics.half_life:.1f}"
            
            # Generate signal
            current_position = self._positions.get(pair)
            
            signal = self.signal_generator.generate_composite_signal(
                pair=pair,
                zscore=spread_metrics.zscore,
                correlation=corr_result.current_correlation,
                hedge_ratio=spread_metrics.hedge_ratio,
                half_life=spread_metrics.half_life,
                current_position=current_position
            )
            
            analysis = PairAnalysis(
                pair=pair,
                timestamp=datetime.now(),
                correlation_result=corr_result,
                cointegration_result=coint_result,
                spread_metrics=spread_metrics,
                current_signal=signal,
                is_tradeable=is_tradeable,
                rejection_reason=rejection_reason
            )
            
            # Cache analysis
            self._last_analysis[pair] = analysis
            
            return analysis
            
        except Exception as e:
            logger.error(f"Analysis failed for {pair}: {e}")
            return None
    
    def screen_pairs(
        self,
        symbols: List[str],
        timeframe: Optional[Timeframe] = None,
        bars: int = 500
    ) -> List[PairAnalysis]:
        """
        Screen multiple symbols for tradeable pairs.
        
        Args:
            symbols: List of symbols to analyze
            timeframe: Analysis timeframe
            bars: Number of bars
            
        Returns:
            List of PairAnalysis for tradeable pairs
        """
        timeframe = timeframe or self.settings.timeframe
        mt5_tf = MT5Timeframe.from_string(timeframe.value)
        
        results = []
        
        # Load all data
        symbol_data = {}
        for symbol in symbols:
            try:
                data = self.data_manager.get_close_prices(symbol, mt5_tf, bars)
                if len(data) >= self.settings.backtest.min_bars_required:
                    symbol_data[symbol] = data
            except Exception as e:
                logger.warning(f"Failed to load {symbol}: {e}")
        
        # Analyze all pairs
        symbols_list = list(symbol_data.keys())
        
        for i, symbol_a in enumerate(symbols_list):
            for symbol_b in symbols_list[i+1:]:
                analysis = self.analyze_pair((symbol_a, symbol_b), timeframe, bars)
                
                if analysis and analysis.is_tradeable:
                    results.append(analysis)
        
        # Sort by signal strength
        results.sort(
            key=lambda x: abs(x.spread_metrics.zscore) if x.spread_metrics else 0,
            reverse=True
        )
        
        return results
    
    def get_signals(
        self,
        pairs: List[Tuple[str, str]],
        timeframe: Optional[Timeframe] = None
    ) -> List[Signal]:
        """
        Get current signals for multiple pairs.
        
        Args:
            pairs: List of pairs to analyze
            timeframe: Analysis timeframe
            
        Returns:
            List of active signals
        """
        signals = []
        
        for pair in pairs:
            analysis = self.analyze_pair(pair, timeframe)
            
            if analysis and analysis.current_signal:
                if analysis.current_signal.type != SignalType.NO_SIGNAL:
                    signals.append(analysis.current_signal)
        
        return signals
    
    def update_position(
        self,
        pair: Tuple[str, str],
        direction: Optional[str]
    ):
        """
        Update position tracking.
        
        Args:
            pair: Pair tuple
            direction: 'long_spread', 'short_spread', or None (closed)
        """
        if direction:
            self._positions[pair] = direction
        elif pair in self._positions:
            del self._positions[pair]
    
    def get_position(self, pair: Tuple[str, str]) -> Optional[str]:
        """Get current position direction for pair."""
        return self._positions.get(pair)
    
    def get_last_analysis(self, pair: Tuple[str, str]) -> Optional[PairAnalysis]:
        """Get cached analysis for pair."""
        return self._last_analysis.get(pair)
    
    def calculate_entry_parameters(
        self,
        pair: Tuple[str, str],
        signal: Signal,
        capital: float
    ) -> Dict:
        """
        Calculate entry parameters for a trade.
        
        Args:
            pair: Pair tuple
            signal: Entry signal
            capital: Available capital
            
        Returns:
            Dictionary with trade parameters
        """
        risk_amount = capital * self.settings.risk.max_risk_per_trade
        hedge_ratio = signal.hedge_ratio or 1.0
        
        # For pairs trading, split risk between legs
        risk_per_leg = risk_amount / 2
        
        return {
            'pair': pair,
            'direction': signal.type.value,
            'hedge_ratio': hedge_ratio,
            'risk_amount': risk_amount,
            'risk_per_leg': risk_per_leg,
            'entry_zscore': signal.zscore,
            'signal_strength': signal.strength
        }
    
    def get_status_summary(self) -> Dict:
        """
        Get strategy status summary.
        
        Returns:
            Dictionary with strategy status
        """
        return {
            'open_positions': len(self._positions),
            'positions': dict(self._positions),
            'last_analyses': len(self._last_analysis),
            'settings': {
                'entry_zscore': self.settings.spread.entry_zscore,
                'exit_zscore': self.settings.spread.exit_zscore,
                'min_correlation': self.settings.spread.min_correlation,
                'max_half_life': self.settings.spread.max_half_life
            }
        }
