"""
Backtesting Engine for the Pairs Trading System.

Provides realistic backtesting with:
- Transaction costs (spreads, commissions)
- Slippage simulation
- Position tracking
- Performance analytics
- Detailed trade logs
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from enum import Enum
import logging

from ..strategy.pairs_strategy import PairsStrategy
from ..strategy.signals import SignalType, Signal
from ..risk.risk_manager import RiskManager, PositionSize
from ..analysis.spread_builder import SpreadBuilder
from ..analysis.correlation import CorrelationAnalyzer
from ..analysis.cointegration import CointegrationAnalyzer
from ..data.data_manager import DataManager
from config.settings import Settings, BacktestParameters


logger = logging.getLogger(__name__)


class TradeDirection(Enum):
    """Direction of the spread trade."""
    LONG_SPREAD = "long_spread"   # Long A, Short B
    SHORT_SPREAD = "short_spread" # Short A, Long B


@dataclass
class Trade:
    """Represents a completed trade."""
    trade_id: int
    pair: Tuple[str, str]
    direction: TradeDirection
    entry_time: datetime
    exit_time: datetime
    entry_zscore: float
    exit_zscore: float
    entry_price_a: float
    entry_price_b: float
    exit_price_a: float
    exit_price_b: float
    units_a: float
    units_b: float
    hedge_ratio: float
    gross_pnl: float
    transaction_costs: float
    slippage_cost: float
    net_pnl: float
    exit_reason: str
    holding_bars: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'trade_id': self.trade_id,
            'pair': f"{self.pair[0]}/{self.pair[1]}",
            'direction': self.direction.value,
            'entry_time': self.entry_time,
            'exit_time': self.exit_time,
            'entry_zscore': self.entry_zscore,
            'exit_zscore': self.exit_zscore,
            'entry_price_a': self.entry_price_a,
            'entry_price_b': self.entry_price_b,
            'exit_price_a': self.exit_price_a,
            'exit_price_b': self.exit_price_b,
            'units_a': self.units_a,
            'units_b': self.units_b,
            'hedge_ratio': self.hedge_ratio,
            'gross_pnl': self.gross_pnl,
            'transaction_costs': self.transaction_costs,
            'slippage_cost': self.slippage_cost,
            'net_pnl': self.net_pnl,
            'exit_reason': self.exit_reason,
            'holding_bars': self.holding_bars
        }


@dataclass
class OpenPosition:
    """Represents an open position during backtest."""
    pair: Tuple[str, str]
    direction: TradeDirection
    entry_time: datetime
    entry_bar: int
    entry_zscore: float
    entry_price_a: float
    entry_price_b: float
    units_a: float
    units_b: float
    hedge_ratio: float
    entry_transaction_cost: float
    entry_slippage: float


@dataclass
class BacktestResult:
    """Complete backtest results."""
    # Basic info
    pair: Tuple[str, str]
    start_date: datetime
    end_date: datetime
    total_bars: int
    
    # Capital
    initial_capital: float
    final_capital: float
    
    # Performance
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int  # in bars
    calmar_ratio: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_trade_pnl: float
    avg_holding_period: float
    
    # Costs
    total_transaction_costs: float
    total_slippage_costs: float
    
    # Data
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    drawdown_series: pd.Series = field(default_factory=pd.Series)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding large data)."""
        return {
            'pair': f"{self.pair[0]}/{self.pair[1]}",
            'start_date': self.start_date,
            'end_date': self.end_date,
            'total_bars': self.total_bars,
            'initial_capital': self.initial_capital,
            'final_capital': self.final_capital,
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_duration': self.max_drawdown_duration,
            'calmar_ratio': self.calmar_ratio,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'largest_win': self.largest_win,
            'largest_loss': self.largest_loss,
            'avg_trade_pnl': self.avg_trade_pnl,
            'avg_holding_period': self.avg_holding_period,
            'total_transaction_costs': self.total_transaction_costs,
            'total_slippage_costs': self.total_slippage_costs
        }
    
    def summary(self) -> str:
        """Generate text summary."""
        return f"""
================================================================================
BACKTEST RESULTS: {self.pair[0]}/{self.pair[1]}
================================================================================
Period: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}
Total Bars: {self.total_bars:,}

CAPITAL
-------
Initial Capital: ${self.initial_capital:,.2f}
Final Capital:   ${self.final_capital:,.2f}
Net P/L:         ${self.final_capital - self.initial_capital:,.2f}

PERFORMANCE
-----------
Total Return:      {self.total_return:.2%}
Annualized Return: {self.annualized_return:.2%}
Sharpe Ratio:      {self.sharpe_ratio:.3f}
Sortino Ratio:     {self.sortino_ratio:.3f}
Max Drawdown:      {self.max_drawdown:.2%}
Calmar Ratio:      {self.calmar_ratio:.3f}

TRADES
------
Total Trades:      {self.total_trades}
Winning Trades:    {self.winning_trades}
Losing Trades:     {self.losing_trades}
Win Rate:          {self.win_rate:.2%}
Profit Factor:     {self.profit_factor:.2f}
Average Win:       ${self.avg_win:,.2f}
Average Loss:      ${self.avg_loss:,.2f}
Largest Win:       ${self.largest_win:,.2f}
Largest Loss:      ${self.largest_loss:,.2f}
Avg Trade P/L:     ${self.avg_trade_pnl:,.2f}
Avg Holding Period: {self.avg_holding_period:.1f} bars

COSTS
-----
Transaction Costs: ${self.total_transaction_costs:,.2f}
Slippage Costs:    ${self.total_slippage_costs:,.2f}
Total Costs:       ${self.total_transaction_costs + self.total_slippage_costs:,.2f}
================================================================================
"""


class BacktestEngine:
    """
    Backtesting engine for pairs trading strategies.
    
    Features:
    - Realistic transaction costs
    - Slippage simulation
    - Position tracking
    - Equity curve generation
    - Comprehensive performance metrics
    """
    
    def __init__(
        self,
        settings: Settings,
        data_manager: Optional[DataManager] = None
    ):
        """
        Initialize the backtest engine.
        
        Args:
            settings: Trading system settings
            data_manager: Optional data manager (creates one if not provided)
        """
        self.settings = settings
        self.backtest_params = settings.backtest
        self.spread_params = settings.spread
        self.risk_params = settings.risk
        
        self.data_manager = data_manager
        
        # Analysis components
        self.spread_builder = SpreadBuilder(
            regression_window=self.spread_params.regression_window,
            zscore_window=self.spread_params.zscore_window
        )
        self.correlation_analyzer = CorrelationAnalyzer(
            window=self.spread_params.regression_window
        )
        self.cointegration_analyzer = CointegrationAnalyzer()
        
        # State
        self.trades: List[Trade] = []
        self.open_positions: Dict[Tuple[str, str], OpenPosition] = {}
        self.equity_curve: List[float] = []
        self.trade_counter: int = 0
        
        logger.info("BacktestEngine initialized")
    
    def run_backtest(
        self,
        pair: Tuple[str, str],
        data_a: pd.DataFrame,
        data_b: pd.DataFrame,
        initial_capital: Optional[float] = None
    ) -> BacktestResult:
        """
        Run backtest on a single pair.
        
        Args:
            pair: Tuple of (instrument_a, instrument_b)
            data_a: OHLCV data for instrument A
            data_b: OHLCV data for instrument B
            initial_capital: Starting capital (uses settings default if None)
            
        Returns:
            BacktestResult with complete metrics
        """
        capital = initial_capital or self.backtest_params.initial_capital
        
        # Reset state
        self._reset_state(capital)
        
        # Align data
        aligned = self._align_data(data_a, data_b)
        if aligned is None or len(aligned) < self.spread_params.regression_window + 100:
            logger.warning(f"Insufficient data for pair {pair}")
            return self._create_empty_result(pair, capital)
        
        logger.info(f"Running backtest for {pair[0]}/{pair[1]} with {len(aligned)} bars")
        
        # Extract price series
        price_a = aligned['close_a']
        price_b = aligned['close_b']
        timestamps = aligned.index
        
        # Pre-calculate spread and z-scores
        spread_data = self.spread_builder.build_spread_with_zscore(price_a, price_b)
        if spread_data is None:
            logger.warning(f"Could not build spread for {pair}")
            return self._create_empty_result(pair, capital)
        
        # Get rolling correlation
        returns_a = price_a.pct_change()
        returns_b = price_b.pct_change()
        rolling_corr = returns_a.rolling(
            window=self.spread_params.regression_window
        ).corr(returns_b)
        
        # Main backtest loop
        warmup = max(self.spread_params.regression_window, self.spread_params.zscore_window) + 10
        
        for i in range(warmup, len(aligned)):
            timestamp = timestamps[i]
            current_price_a = price_a.iloc[i]
            current_price_b = price_b.iloc[i]
            current_zscore = spread_data['zscore'].iloc[i]
            current_hedge_ratio = spread_data['hedge_ratio'].iloc[i]
            current_corr = rolling_corr.iloc[i]
            
            # Skip if any value is NaN
            if pd.isna(current_zscore) or pd.isna(current_hedge_ratio) or pd.isna(current_corr):
                continue
            
            # Update equity with mark-to-market
            self._update_equity(pair, current_price_a, current_price_b, i)
            
            # Check for exits first
            if pair in self.open_positions:
                exit_signal = self._check_exit_conditions(
                    pair, current_zscore, current_corr, i
                )
                if exit_signal:
                    self._close_position(
                        pair, timestamp, i, current_price_a, current_price_b,
                        current_zscore, exit_signal
                    )
            
            # Check for entries (only if not in position)
            if pair not in self.open_positions:
                entry_signal = self._check_entry_conditions(
                    current_zscore, current_corr
                )
                if entry_signal:
                    self._open_position(
                        pair, entry_signal, timestamp, i,
                        current_price_a, current_price_b,
                        current_zscore, current_hedge_ratio
                    )
        
        # Close any remaining positions at end
        if pair in self.open_positions:
            self._close_position(
                pair, timestamps[-1], len(aligned) - 1,
                price_a.iloc[-1], price_b.iloc[-1],
                spread_data['zscore'].iloc[-1], "end_of_data"
            )
        
        # Calculate results
        result = self._calculate_results(
            pair, timestamps[0], timestamps[-1], len(aligned), capital
        )
        
        logger.info(f"Backtest complete: {result.total_trades} trades, "
                   f"Return: {result.total_return:.2%}, Sharpe: {result.sharpe_ratio:.2f}")
        
        return result
    
    def run_multi_pair_backtest(
        self,
        pairs_data: Dict[Tuple[str, str], Tuple[pd.DataFrame, pd.DataFrame]],
        initial_capital: Optional[float] = None
    ) -> Dict[Tuple[str, str], BacktestResult]:
        """
        Run backtest on multiple pairs.
        
        Args:
            pairs_data: Dict mapping pairs to (data_a, data_b) tuples
            initial_capital: Starting capital
            
        Returns:
            Dict mapping pairs to their BacktestResults
        """
        results = {}
        
        for pair, (data_a, data_b) in pairs_data.items():
            logger.info(f"Backtesting pair: {pair[0]}/{pair[1]}")
            results[pair] = self.run_backtest(pair, data_a, data_b, initial_capital)
        
        return results
    
    def _reset_state(self, capital: float) -> None:
        """Reset backtest state."""
        self.trades = []
        self.open_positions = {}
        self.equity_curve = [capital]
        self.trade_counter = 0
        self.current_capital = capital
    
    def _align_data(
        self,
        data_a: pd.DataFrame,
        data_b: pd.DataFrame
    ) -> Optional[pd.DataFrame]:
        """Align two dataframes by timestamp."""
        try:
            # Get common index
            common_idx = data_a.index.intersection(data_b.index)
            
            if len(common_idx) == 0:
                return None
            
            # Create aligned dataframe
            aligned = pd.DataFrame(index=common_idx)
            aligned['open_a'] = data_a.loc[common_idx, 'open']
            aligned['high_a'] = data_a.loc[common_idx, 'high']
            aligned['low_a'] = data_a.loc[common_idx, 'low']
            aligned['close_a'] = data_a.loc[common_idx, 'close']
            aligned['open_b'] = data_b.loc[common_idx, 'open']
            aligned['high_b'] = data_b.loc[common_idx, 'high']
            aligned['low_b'] = data_b.loc[common_idx, 'low']
            aligned['close_b'] = data_b.loc[common_idx, 'close']
            
            return aligned.dropna()
            
        except Exception as e:
            logger.error(f"Error aligning data: {e}")
            return None
    
    def _check_entry_conditions(
        self,
        zscore: float,
        correlation: float
    ) -> Optional[TradeDirection]:
        """Check if entry conditions are met."""
        # Minimum correlation required
        if correlation < self.spread_params.min_correlation:
            return None
        
        # Maximum open positions
        if len(self.open_positions) >= self.risk_params.max_open_pairs:
            return None
        
        # Entry signals
        if zscore <= -self.spread_params.entry_zscore:
            return TradeDirection.LONG_SPREAD
        elif zscore >= self.spread_params.entry_zscore:
            return TradeDirection.SHORT_SPREAD
        
        return None
    
    def _check_exit_conditions(
        self,
        pair: Tuple[str, str],
        zscore: float,
        correlation: float,
        current_bar: int
    ) -> Optional[str]:
        """Check if exit conditions are met."""
        if pair not in self.open_positions:
            return None
        
        position = self.open_positions[pair]
        
        # Stop loss on extreme z-score
        if abs(zscore) >= self.spread_params.stop_loss_zscore:
            return "stop_loss"
        
        # Correlation breakdown
        if correlation < self.spread_params.min_correlation - 0.1:
            return "correlation_breakdown"
        
        # Mean reversion exit
        if position.direction == TradeDirection.LONG_SPREAD:
            # Exit long when zscore rises above exit threshold
            if zscore >= -self.spread_params.exit_zscore:
                return "mean_reversion"
        else:
            # Exit short when zscore falls below exit threshold
            if zscore <= self.spread_params.exit_zscore:
                return "mean_reversion"
        
        return None
    
    def _calculate_position_size(
        self,
        price_a: float,
        price_b: float,
        hedge_ratio: float
    ) -> Tuple[float, float]:
        """Calculate position sizes for both legs."""
        # Risk-based position sizing
        risk_amount = self.current_capital * self.risk_params.max_risk_per_trade
        
        # Calculate notional for equal risk
        # Units_A * Price_A ≈ Units_B * Price_B * hedge_ratio
        notional_a = risk_amount / 2
        notional_b = risk_amount / 2
        
        units_a = notional_a / price_a
        units_b = (notional_b / price_b) * abs(hedge_ratio)
        
        return units_a, units_b
    
    def _calculate_transaction_cost(
        self,
        instrument: str,
        units: float,
        price: float
    ) -> float:
        """Calculate transaction cost for a trade."""
        # Spread cost in pips
        spread_pips = self.backtest_params.spread_cost_pips
        
        # Convert pips to price
        # For JPY pairs, pip = 0.01, for others pip = 0.0001
        if 'JPY' in instrument:
            pip_value = 0.01
        else:
            pip_value = 0.0001
        
        spread_cost = units * spread_pips * pip_value
        
        return abs(spread_cost)
    
    def _calculate_slippage(
        self,
        instrument: str,
        units: float,
        price: float
    ) -> float:
        """Calculate slippage cost."""
        slippage_pips = self.backtest_params.slippage_pips
        
        if 'JPY' in instrument:
            pip_value = 0.01
        else:
            pip_value = 0.0001
        
        slippage_cost = units * slippage_pips * pip_value
        
        return abs(slippage_cost)
    
    def _open_position(
        self,
        pair: Tuple[str, str],
        direction: TradeDirection,
        timestamp: datetime,
        bar: int,
        price_a: float,
        price_b: float,
        zscore: float,
        hedge_ratio: float
    ) -> None:
        """Open a new position."""
        units_a, units_b = self._calculate_position_size(price_a, price_b, hedge_ratio)
        
        # Calculate entry costs
        cost_a = self._calculate_transaction_cost(pair[0], units_a, price_a)
        cost_b = self._calculate_transaction_cost(pair[1], units_b, price_b)
        slip_a = self._calculate_slippage(pair[0], units_a, price_a)
        slip_b = self._calculate_slippage(pair[1], units_b, price_b)
        
        total_cost = cost_a + cost_b
        total_slip = slip_a + slip_b
        
        # Deduct costs from capital
        self.current_capital -= (total_cost + total_slip)
        
        # Create position
        position = OpenPosition(
            pair=pair,
            direction=direction,
            entry_time=timestamp,
            entry_bar=bar,
            entry_zscore=zscore,
            entry_price_a=price_a,
            entry_price_b=price_b,
            units_a=units_a,
            units_b=units_b,
            hedge_ratio=hedge_ratio,
            entry_transaction_cost=total_cost,
            entry_slippage=total_slip
        )
        
        self.open_positions[pair] = position
        
        logger.debug(f"Opened {direction.value} position on {pair} at z={zscore:.2f}")
    
    def _close_position(
        self,
        pair: Tuple[str, str],
        timestamp: datetime,
        bar: int,
        price_a: float,
        price_b: float,
        zscore: float,
        exit_reason: str
    ) -> None:
        """Close an open position."""
        if pair not in self.open_positions:
            return
        
        position = self.open_positions[pair]
        
        # Calculate exit costs
        cost_a = self._calculate_transaction_cost(pair[0], position.units_a, price_a)
        cost_b = self._calculate_transaction_cost(pair[1], position.units_b, price_b)
        slip_a = self._calculate_slippage(pair[0], position.units_a, price_a)
        slip_b = self._calculate_slippage(pair[1], position.units_b, price_b)
        
        exit_cost = cost_a + cost_b
        exit_slip = slip_a + slip_b
        
        # Calculate P/L
        if position.direction == TradeDirection.LONG_SPREAD:
            # Long A, Short B
            pnl_a = position.units_a * (price_a - position.entry_price_a)
            pnl_b = position.units_b * (position.entry_price_b - price_b)
        else:
            # Short A, Long B
            pnl_a = position.units_a * (position.entry_price_a - price_a)
            pnl_b = position.units_b * (price_b - position.entry_price_b)
        
        gross_pnl = pnl_a + pnl_b
        total_costs = position.entry_transaction_cost + exit_cost
        total_slip = position.entry_slippage + exit_slip
        net_pnl = gross_pnl - exit_cost - exit_slip
        
        # Update capital
        self.current_capital += net_pnl
        
        # Record trade
        self.trade_counter += 1
        trade = Trade(
            trade_id=self.trade_counter,
            pair=pair,
            direction=position.direction,
            entry_time=position.entry_time,
            exit_time=timestamp,
            entry_zscore=position.entry_zscore,
            exit_zscore=zscore,
            entry_price_a=position.entry_price_a,
            entry_price_b=position.entry_price_b,
            exit_price_a=price_a,
            exit_price_b=price_b,
            units_a=position.units_a,
            units_b=position.units_b,
            hedge_ratio=position.hedge_ratio,
            gross_pnl=gross_pnl,
            transaction_costs=total_costs,
            slippage_cost=total_slip,
            net_pnl=net_pnl,
            exit_reason=exit_reason,
            holding_bars=bar - position.entry_bar
        )
        
        self.trades.append(trade)
        
        # Remove from open positions
        del self.open_positions[pair]
        
        logger.debug(f"Closed {position.direction.value} on {pair}: "
                    f"P/L=${net_pnl:.2f}, reason={exit_reason}")
    
    def _update_equity(
        self,
        pair: Tuple[str, str],
        price_a: float,
        price_b: float,
        bar: int
    ) -> None:
        """Update equity curve with mark-to-market."""
        equity = self.current_capital
        
        # Add unrealized P/L from open positions
        for p, position in self.open_positions.items():
            if p == pair:
                if position.direction == TradeDirection.LONG_SPREAD:
                    pnl_a = position.units_a * (price_a - position.entry_price_a)
                    pnl_b = position.units_b * (position.entry_price_b - price_b)
                else:
                    pnl_a = position.units_a * (position.entry_price_a - price_a)
                    pnl_b = position.units_b * (price_b - position.entry_price_b)
                equity += pnl_a + pnl_b
        
        self.equity_curve.append(equity)
    
    def _calculate_results(
        self,
        pair: Tuple[str, str],
        start_date: datetime,
        end_date: datetime,
        total_bars: int,
        initial_capital: float
    ) -> BacktestResult:
        """Calculate comprehensive backtest results."""
        equity = pd.Series(self.equity_curve)
        
        # Basic returns
        final_capital = equity.iloc[-1]
        total_return = (final_capital - initial_capital) / initial_capital
        
        # Annualized return (assuming hourly bars, 24*365 = 8760 bars/year)
        bars_per_year = 8760  # For H1
        years = total_bars / bars_per_year
        if years > 0 and total_return > -1:
            annualized_return = (1 + total_return) ** (1 / years) - 1
        else:
            annualized_return = 0.0
        
        # Returns for ratios
        returns = equity.pct_change().dropna()
        
        # Sharpe Ratio (annualized)
        if len(returns) > 1 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(bars_per_year)
        else:
            sharpe = 0.0
        
        # Sortino Ratio (downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 1:
            downside_std = downside_returns.std()
            if downside_std > 0:
                sortino = (returns.mean() / downside_std) * np.sqrt(bars_per_year)
            else:
                sortino = 0.0
        else:
            sortino = 0.0
        
        # Drawdown
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min())
        
        # Max drawdown duration
        in_drawdown = drawdown < 0
        drawdown_groups = (~in_drawdown).cumsum()
        if in_drawdown.any():
            dd_lengths = in_drawdown.groupby(drawdown_groups).sum()
            max_dd_duration = dd_lengths.max()
        else:
            max_dd_duration = 0
        
        # Calmar Ratio
        if max_drawdown > 0:
            calmar = annualized_return / max_drawdown
        else:
            calmar = 0.0
        
        # Trade statistics
        total_trades = len(self.trades)
        
        if total_trades > 0:
            pnls = [t.net_pnl for t in self.trades]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p <= 0]
            
            winning_trades = len(wins)
            losing_trades = len(losses)
            win_rate = winning_trades / total_trades
            
            avg_win = np.mean(wins) if wins else 0.0
            avg_loss = abs(np.mean(losses)) if losses else 0.0
            
            gross_profit = sum(wins) if wins else 0.0
            gross_loss = abs(sum(losses)) if losses else 0.0
            
            if gross_loss > 0:
                profit_factor = gross_profit / gross_loss
            else:
                profit_factor = float('inf') if gross_profit > 0 else 0.0
            
            largest_win = max(pnls) if pnls else 0.0
            largest_loss = min(pnls) if pnls else 0.0
            avg_trade_pnl = np.mean(pnls)
            
            holding_periods = [t.holding_bars for t in self.trades]
            avg_holding_period = np.mean(holding_periods)
            
            total_transaction_costs = sum(t.transaction_costs for t in self.trades)
            total_slippage_costs = sum(t.slippage_cost for t in self.trades)
        else:
            winning_trades = losing_trades = 0
            win_rate = 0.0
            profit_factor = 0.0
            avg_win = avg_loss = 0.0
            largest_win = largest_loss = 0.0
            avg_trade_pnl = 0.0
            avg_holding_period = 0.0
            total_transaction_costs = total_slippage_costs = 0.0
        
        return BacktestResult(
            pair=pair,
            start_date=start_date,
            end_date=end_date,
            total_bars=total_bars,
            initial_capital=initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_dd_duration,
            calmar_ratio=calmar,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            avg_trade_pnl=avg_trade_pnl,
            avg_holding_period=avg_holding_period,
            total_transaction_costs=total_transaction_costs,
            total_slippage_costs=total_slippage_costs,
            trades=self.trades.copy(),
            equity_curve=equity,
            drawdown_series=drawdown
        )
    
    def _create_empty_result(
        self,
        pair: Tuple[str, str],
        capital: float
    ) -> BacktestResult:
        """Create empty result for failed backtests."""
        return BacktestResult(
            pair=pair,
            start_date=datetime.now(),
            end_date=datetime.now(),
            total_bars=0,
            initial_capital=capital,
            final_capital=capital,
            total_return=0.0,
            annualized_return=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            max_drawdown_duration=0,
            calmar_ratio=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            largest_win=0.0,
            largest_loss=0.0,
            avg_trade_pnl=0.0,
            avg_holding_period=0.0,
            total_transaction_costs=0.0,
            total_slippage_costs=0.0
        )
    
    def get_trades_dataframe(self) -> pd.DataFrame:
        """Get all trades as a DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        
        return pd.DataFrame([t.to_dict() for t in self.trades])
    
    def save_results(
        self,
        result: BacktestResult,
        filepath: str
    ) -> None:
        """Save backtest results to file."""
        import json
        
        # Prepare data
        data = result.to_dict()
        data['trades'] = [t.to_dict() for t in result.trades]
        
        # Convert datetime objects
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        
        for trade in data['trades']:
            for key, value in trade.items():
                if isinstance(value, datetime):
                    trade[key] = value.isoformat()
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filepath}")
