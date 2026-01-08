"""
Backtesting Engine Module.

Professional backtester with transaction costs and realistic execution.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Tuple
import json
import logging

from config.settings import Settings
from src.analysis.spread_builder import SpreadBuilder
from src.analysis.correlation import CorrelationAnalyzer


logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Single trade record."""
    entry_time: datetime
    exit_time: datetime
    pair: Tuple[str, str]
    direction: str  # 'long_spread' or 'short_spread'
    entry_zscore: float
    exit_zscore: float
    hedge_ratio: float
    size_a: float
    size_b: float
    entry_price_a: float
    entry_price_b: float
    exit_price_a: float
    exit_price_b: float
    gross_pnl: float
    costs: float
    net_pnl: float
    exit_reason: str


@dataclass
class BacktestResult:
    """Complete backtest results."""
    pair: Tuple[str, str]
    start_date: datetime
    end_date: datetime
    
    # Capital
    initial_capital: float
    final_capital: float
    
    # Returns
    total_return: float
    annual_return: float
    
    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    volatility: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    avg_trade: float
    avg_win: float
    avg_loss: float
    max_win: float
    max_loss: float
    avg_holding_period: float
    
    # Costs
    total_costs: float
    
    # Equity curve
    equity_curve: Optional[pd.Series] = None
    trades: List[Trade] = field(default_factory=list)


class BacktestEngine:
    """
    Backtesting engine for pairs trading.
    
    Features:
    - Realistic transaction costs
    - Slippage simulation
    - Rolling parameters
    - Comprehensive metrics
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize backtest engine.
        
        Args:
            settings: Trading settings
        """
        self.settings = settings
        self.bt_settings = settings.backtest
        self.spread_settings = settings.spread
        self.risk_settings = settings.risk
        
        self.spread_builder = SpreadBuilder(
            regression_window=settings.spread.regression_window,
            zscore_window=settings.spread.zscore_window,
            recalculate_beta=settings.spread.recalc_hedge_ratio
        )
        
        self.correlation_analyzer = CorrelationAnalyzer(
            window=settings.spread.correlation_window
        )
    
    def run_backtest(
        self,
        pair: Tuple[str, str],
        price_a: pd.Series,
        price_b: pd.Series,
        initial_capital: Optional[float] = None
    ) -> Optional[BacktestResult]:
        """
        Run backtest on historical data.
        
        Args:
            pair: (symbol_a, symbol_b) tuple
            price_a: Price series for symbol A
            price_b: Price series for symbol B
            initial_capital: Starting capital
            
        Returns:
            BacktestResult or None
        """
        initial_capital = initial_capital or self.bt_settings.initial_capital
        
        # Align data
        common_idx = price_a.index.intersection(price_b.index)
        price_a = price_a.loc[common_idx]
        price_b = price_b.loc[common_idx]
        
        # Dynamic min_bars based on settings
        min_required = max(
            self.spread_settings.regression_window + self.spread_settings.zscore_window + 50,
            self.bt_settings.min_bars_required
        )
        
        if len(price_a) < min_required:
            logger.warning(f"Insufficient data: {len(price_a)} bars (need {min_required})")
            return None
        
        # Build spread with z-score
        spread_data = self.spread_builder.build_spread_with_zscore(price_a, price_b)
        
        # Calculate rolling correlation
        rolling_corr = self.correlation_analyzer.calculate_rolling_correlation(price_a, price_b)
        
        # Initialize tracking
        capital = initial_capital
        equity_curve = []
        trades = []
        
        position = None  # Current position
        
        # Calculate start index (ensure we have valid z-scores)
        start_idx = self.spread_settings.regression_window + self.spread_settings.zscore_window
        
        # Find first valid z-score
        for i in range(start_idx, len(common_idx)):
            if not pd.isna(spread_data['zscore'].iloc[i]):
                start_idx = i
                break
        
        if start_idx >= len(common_idx) - 10:
            logger.warning("Not enough valid data points after warmup")
            return None
        
        # Iterate through data
        for i in range(start_idx, len(common_idx)):
            idx = common_idx[i]
            
            zscore = spread_data['zscore'].iloc[i]
            hedge_ratio = spread_data['hedge_ratio'].iloc[i]
            correlation = rolling_corr.iloc[i] if i < len(rolling_corr) and not pd.isna(rolling_corr.iloc[i]) else 0.7
            
            current_price_a = price_a.iloc[i]
            current_price_b = price_b.iloc[i]
            
            # Skip if NaN
            if pd.isna(zscore) or pd.isna(hedge_ratio):
                equity_curve.append(capital)
                continue
            
            # If in position, check for exit
            if position is not None:
                should_exit = False
                exit_reason = ""
                
                # Mean reversion exit
                if position['direction'] == 'long_spread' and zscore >= -self.spread_settings.exit_zscore:
                    should_exit = True
                    exit_reason = "mean_reversion"
                
                elif position['direction'] == 'short_spread' and zscore <= self.spread_settings.exit_zscore:
                    should_exit = True
                    exit_reason = "mean_reversion"
                
                # Stop loss
                if abs(zscore) >= self.spread_settings.stop_loss_zscore:
                    should_exit = True
                    exit_reason = "stop_loss"
                
                # Correlation breakdown
                if correlation < self.spread_settings.min_correlation - 0.1:
                    should_exit = True
                    exit_reason = "correlation_breakdown"
                
                if should_exit:
                    # Close position
                    trade = self._close_position(
                        position=position,
                        exit_idx=idx,
                        exit_price_a=current_price_a,
                        exit_price_b=current_price_b,
                        exit_zscore=zscore,
                        exit_reason=exit_reason
                    )
                    
                    trades.append(trade)
                    capital += trade.net_pnl
                    position = None
            
            # If not in position, check for entry
            if position is None:
                # Correlation filter
                if correlation < self.spread_settings.min_correlation:
                    equity_curve.append(capital)
                    continue
                
                should_enter = False
                direction = None
                
                # Long spread entry
                if zscore <= -self.spread_settings.entry_zscore:
                    should_enter = True
                    direction = 'long_spread'
                
                # Short spread entry
                elif zscore >= self.spread_settings.entry_zscore:
                    should_enter = True
                    direction = 'short_spread'
                
                if should_enter:
                    # Calculate position size
                    risk_amount = capital * self.risk_settings.max_risk_per_trade
                    size_a, size_b = self._calculate_size(risk_amount, hedge_ratio)
                    
                    position = {
                        'pair': pair,
                        'direction': direction,
                        'entry_idx': idx,
                        'entry_price_a': current_price_a,
                        'entry_price_b': current_price_b,
                        'entry_zscore': zscore,
                        'hedge_ratio': hedge_ratio,
                        'size_a': size_a,
                        'size_b': size_b
                    }
            
            equity_curve.append(capital)
        
        # Close any remaining position
        if position is not None:
            trade = self._close_position(
                position=position,
                exit_idx=common_idx[-1],
                exit_price_a=price_a.iloc[-1],
                exit_price_b=price_b.iloc[-1],
                exit_zscore=spread_data['zscore'].iloc[-1] if not pd.isna(spread_data['zscore'].iloc[-1]) else 0,
                exit_reason="end_of_data"
            )
            trades.append(trade)
            capital += trade.net_pnl
            if equity_curve:
                equity_curve[-1] = capital
        
        # Create equity series
        if len(equity_curve) > 0:
            equity_series = pd.Series(equity_curve, index=common_idx[start_idx:start_idx + len(equity_curve)])
        else:
            equity_series = pd.Series([initial_capital])
        
        # Calculate metrics
        result = self._calculate_metrics(
            pair=pair,
            initial_capital=initial_capital,
            final_capital=capital,
            equity_curve=equity_series,
            trades=trades,
            start_date=common_idx[start_idx],
            end_date=common_idx[-1]
        )
        
        return result
    
    def _close_position(
        self,
        position: dict,
        exit_idx: datetime,
        exit_price_a: float,
        exit_price_b: float,
        exit_zscore: float,
        exit_reason: str
    ) -> Trade:
        """Close a position and create trade record."""
        # Calculate P&L
        if position['direction'] == 'long_spread':
            # Long A, Short B
            pnl_a = (exit_price_a - position['entry_price_a']) * position['size_a'] * 100000
            pnl_b = (position['entry_price_b'] - exit_price_b) * position['size_b'] * 100000
        else:
            # Short A, Long B
            pnl_a = (position['entry_price_a'] - exit_price_a) * position['size_a'] * 100000
            pnl_b = (exit_price_b - position['entry_price_b']) * position['size_b'] * 100000
        
        gross_pnl = pnl_a + pnl_b
        
        # Calculate costs
        spread_cost = self.bt_settings.spread_cost * (position['size_a'] + position['size_b']) * 10
        slippage = self.bt_settings.slippage * (position['size_a'] + position['size_b']) * 10 * 2  # Entry + exit
        commission = self.bt_settings.commission_per_lot * (position['size_a'] + position['size_b']) * 2
        
        total_costs = spread_cost + slippage + commission
        net_pnl = gross_pnl - total_costs
        
        return Trade(
            entry_time=position['entry_idx'],
            exit_time=exit_idx,
            pair=position['pair'],
            direction=position['direction'],
            entry_zscore=position['entry_zscore'],
            exit_zscore=exit_zscore,
            hedge_ratio=position['hedge_ratio'],
            size_a=position['size_a'],
            size_b=position['size_b'],
            entry_price_a=position['entry_price_a'],
            entry_price_b=position['entry_price_b'],
            exit_price_a=exit_price_a,
            exit_price_b=exit_price_b,
            gross_pnl=gross_pnl,
            costs=total_costs,
            net_pnl=net_pnl,
            exit_reason=exit_reason
        )
    
    def _calculate_size(
        self,
        risk_amount: float,
        hedge_ratio: float
    ) -> Tuple[float, float]:
        """Calculate position sizes."""
        # Simple sizing: divide risk by 2 for each leg
        risk_per_leg = risk_amount / 2
        
        # Assume 30 pip stop = $300 per lot
        base_size = risk_per_leg / 300
        
        size_a = round(base_size, 2)
        size_b = round(base_size * abs(hedge_ratio), 2)
        
        return max(0.01, size_a), max(0.01, size_b)
    
    def _calculate_metrics(
        self,
        pair: Tuple[str, str],
        initial_capital: float,
        final_capital: float,
        equity_curve: pd.Series,
        trades: List[Trade],
        start_date: datetime,
        end_date: datetime
    ) -> BacktestResult:
        """Calculate comprehensive backtest metrics."""
        # Basic returns
        total_return = (final_capital - initial_capital) / initial_capital
        
        # Annualized return
        days = (end_date - start_date).days
        years = days / 365.25 if days > 0 else 1
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Returns series
        returns = equity_curve.pct_change().dropna()
        
        # Volatility
        if len(returns) > 0:
            volatility = returns.std() * np.sqrt(252 * 24)  # Hourly data
        else:
            volatility = 0
        
        # Sharpe Ratio
        risk_free_rate = 0.02
        if volatility > 0:
            excess_returns = returns.mean() * 252 * 24 - risk_free_rate
            sharpe_ratio = excess_returns / volatility
        else:
            sharpe_ratio = 0
        
        # Sortino Ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_std = downside_returns.std() * np.sqrt(252 * 24)
            sortino_ratio = (returns.mean() * 252 * 24 - risk_free_rate) / downside_std if downside_std > 0 else 0
        else:
            sortino_ratio = 0
        
        # Max Drawdown
        if len(equity_curve) > 0:
            cummax = equity_curve.cummax()
            drawdown = (equity_curve - cummax) / cummax
            max_drawdown = drawdown.min()
        else:
            max_drawdown = 0
        
        # Calmar Ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Trade statistics
        total_trades = len(trades)
        
        if total_trades > 0:
            pnls = [t.net_pnl for t in trades]
            winning = [p for p in pnls if p > 0]
            losing = [p for p in pnls if p <= 0]
            
            winning_trades = len(winning)
            losing_trades = len(losing)
            win_rate = winning_trades / total_trades
            
            gross_profit = sum(winning)
            gross_loss = abs(sum(losing))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
            
            avg_trade = np.mean(pnls)
            avg_win = np.mean(winning) if winning else 0
            avg_loss = np.mean(losing) if losing else 0
            max_win = max(pnls) if pnls else 0
            max_loss = min(pnls) if pnls else 0
            
            # Holding period
            holding_periods = [(t.exit_time - t.entry_time).total_seconds() / 3600 for t in trades]
            avg_holding_period = np.mean(holding_periods)
            
            total_costs = sum(t.costs for t in trades)
        else:
            winning_trades = losing_trades = 0
            win_rate = profit_factor = 0
            avg_trade = avg_win = avg_loss = max_win = max_loss = 0
            avg_holding_period = 0
            total_costs = 0
        
        return BacktestResult(
            pair=pair,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            volatility=volatility,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_trade=avg_trade,
            avg_win=avg_win,
            avg_loss=avg_loss,
            max_win=max_win,
            max_loss=max_loss,
            avg_holding_period=avg_holding_period,
            total_costs=total_costs,
            equity_curve=equity_curve,
            trades=trades
        )
    
    def save_results(self, result: BacktestResult, filepath: str):
        """Save backtest results to JSON."""
        data = {
            'pair': result.pair,
            'start_date': result.start_date.isoformat(),
            'end_date': result.end_date.isoformat(),
            'initial_capital': result.initial_capital,
            'final_capital': result.final_capital,
            'total_return': result.total_return,
            'annual_return': result.annual_return,
            'sharpe_ratio': result.sharpe_ratio,
            'sortino_ratio': result.sortino_ratio,
            'calmar_ratio': result.calmar_ratio,
            'max_drawdown': result.max_drawdown,
            'volatility': result.volatility,
            'total_trades': result.total_trades,
            'win_rate': result.win_rate,
            'profit_factor': result.profit_factor,
            'avg_trade': result.avg_trade,
            'total_costs': result.total_costs,
            'trades': [
                {
                    'entry_time': t.entry_time.isoformat(),
                    'exit_time': t.exit_time.isoformat(),
                    'direction': t.direction,
                    'net_pnl': t.net_pnl,
                    'exit_reason': t.exit_reason
                }
                for t in result.trades
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
