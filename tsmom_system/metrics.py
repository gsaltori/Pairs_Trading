"""
Time-Series Momentum System - Metrics Calculator
Performance metrics and reporting.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    
    # Period
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    years: float
    
    # Returns
    total_return: float
    cagr: float
    
    # Risk
    volatility: float
    max_drawdown: float
    avg_drawdown: float
    max_drawdown_duration_days: int
    
    # Risk-Adjusted
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Trade Statistics
    total_trades: int
    buy_trades: int
    sell_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    # P&L
    gross_profit: float
    gross_loss: float
    net_profit: float
    profit_factor: float
    expectancy: float
    expectancy_r: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    
    # Costs
    total_commissions: float
    total_slippage: float
    total_costs: float
    
    # Final State
    final_equity: float


class MetricsCalculator:
    """Calculate all performance metrics."""
    
    @staticmethod
    def calculate(
        equity_df: pd.DataFrame,
        trades_df: pd.DataFrame,
        initial_capital: float,
        risk_free_rate: float = 0.04,
    ) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            equity_df: DataFrame with Equity, Drawdown columns
            trades_df: DataFrame with trade history
            initial_capital: Starting capital
            risk_free_rate: Annual risk-free rate (default 4%)
        """
        # Period
        start_date = equity_df.index[0]
        end_date = equity_df.index[-1]
        years = (end_date - start_date).days / 365.25
        
        # Final values
        final_equity = equity_df['Equity'].iloc[-1]
        
        # Returns
        total_return = (final_equity / initial_capital) - 1
        cagr = (final_equity / initial_capital) ** (1 / years) - 1 if years > 0 else 0
        
        # Volatility from daily returns
        daily_returns = equity_df['Equity'].pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252)
        
        # Drawdown
        max_drawdown = equity_df['Drawdown'].max()
        avg_drawdown = equity_df['Drawdown'].mean()
        
        # Max drawdown duration
        in_drawdown = equity_df['Drawdown'] > 0.001
        if in_drawdown.any():
            drawdown_groups = (~in_drawdown).cumsum()
            dd_lengths = in_drawdown.groupby(drawdown_groups).sum()
            max_dd_duration = int(dd_lengths.max())
        else:
            max_dd_duration = 0
        
        # Risk-adjusted ratios
        excess_return = cagr - risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0
        
        # Sortino (downside deviation)
        negative_returns = daily_returns[daily_returns < 0]
        if len(negative_returns) > 0:
            downside_std = negative_returns.std() * np.sqrt(252)
            sortino_ratio = excess_return / downside_std if downside_std > 0 else 0
        else:
            sortino_ratio = 0
        
        # Calmar
        calmar_ratio = cagr / max_drawdown if max_drawdown > 0 else 0
        
        # Trade statistics
        total_trades = len(trades_df)
        
        if total_trades > 0:
            buy_trades = len(trades_df[trades_df['Side'] == 'BUY'])
            sell_trades = len(trades_df[trades_df['Side'] == 'SELL'])
            
            # P&L from sell trades
            sell_df = trades_df[trades_df['Side'] == 'SELL'].copy()
            
            if len(sell_df) > 0 and 'Realized_PnL' in sell_df.columns:
                pnl_series = sell_df['Realized_PnL'].dropna()
                
                winners = pnl_series[pnl_series > 0]
                losers = pnl_series[pnl_series < 0]
                
                winning_trades = len(winners)
                losing_trades = len(losers)
                win_rate = winning_trades / len(pnl_series) if len(pnl_series) > 0 else 0
                
                gross_profit = winners.sum() if len(winners) > 0 else 0
                gross_loss = abs(losers.sum()) if len(losers) > 0 else 0
                net_profit = gross_profit - gross_loss
                
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
                
                expectancy = pnl_series.mean() if len(pnl_series) > 0 else 0
                
                avg_win = winners.mean() if len(winners) > 0 else 0
                avg_loss = losers.mean() if len(losers) > 0 else 0
                
                # Expectancy in R (risk units)
                avg_trade_size = sell_df['Gross_Value'].mean()
                risk_per_trade = avg_trade_size * 0.10  # Assume 10% risk per trade
                expectancy_r = expectancy / risk_per_trade if risk_per_trade > 0 else 0
                
                largest_win = winners.max() if len(winners) > 0 else 0
                largest_loss = losers.min() if len(losers) > 0 else 0
            else:
                winning_trades = losing_trades = 0
                win_rate = 0
                gross_profit = gross_loss = net_profit = 0
                profit_factor = 0
                expectancy = expectancy_r = 0
                avg_win = avg_loss = largest_win = largest_loss = 0
            
            # Costs
            total_commissions = trades_df['Commission'].sum()
            total_slippage = trades_df['Slippage'].sum()
        else:
            buy_trades = sell_trades = 0
            winning_trades = losing_trades = 0
            win_rate = 0
            gross_profit = gross_loss = net_profit = 0
            profit_factor = 0
            expectancy = expectancy_r = 0
            avg_win = avg_loss = largest_win = largest_loss = 0
            total_commissions = total_slippage = 0
        
        return PerformanceMetrics(
            start_date=start_date,
            end_date=end_date,
            years=years,
            total_return=total_return,
            cagr=cagr,
            volatility=volatility,
            max_drawdown=max_drawdown,
            avg_drawdown=avg_drawdown,
            max_drawdown_duration_days=max_dd_duration,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            total_trades=total_trades,
            buy_trades=buy_trades,
            sell_trades=sell_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            net_profit=net_profit,
            profit_factor=profit_factor,
            expectancy=expectancy,
            expectancy_r=expectancy_r,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            total_commissions=total_commissions,
            total_slippage=total_slippage,
            total_costs=total_commissions + total_slippage,
            final_equity=final_equity,
        )


def print_metrics(m: PerformanceMetrics, title: str = "PERFORMANCE METRICS"):
    """Print formatted metrics."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    
    print(f"\nPeriod: {m.start_date.strftime('%Y-%m-%d')} to {m.end_date.strftime('%Y-%m-%d')} ({m.years:.1f} years)")
    
    print("\n--- RETURNS ---")
    print(f"  Total Return:      {m.total_return:>10.1%}")
    print(f"  CAGR:              {m.cagr:>10.1%}")
    print(f"  Final Equity:      ${m.final_equity:>12,.2f}")
    
    print("\n--- RISK ---")
    print(f"  Volatility:        {m.volatility:>10.1%}")
    print(f"  Max Drawdown:      {m.max_drawdown:>10.1%}")
    print(f"  Avg Drawdown:      {m.avg_drawdown:>10.1%}")
    print(f"  Max DD Duration:   {m.max_drawdown_duration_days:>10} days")
    
    print("\n--- RISK-ADJUSTED ---")
    print(f"  Sharpe Ratio:      {m.sharpe_ratio:>10.2f}")
    print(f"  Sortino Ratio:     {m.sortino_ratio:>10.2f}")
    print(f"  Calmar Ratio:      {m.calmar_ratio:>10.2f}")
    
    print("\n--- TRADE STATISTICS ---")
    print(f"  Total Trades:      {m.total_trades:>10}")
    print(f"  Win Rate:          {m.win_rate:>10.1%}")
    print(f"  Profit Factor:     {m.profit_factor:>10.2f}")
    print(f"  Expectancy ($):    ${m.expectancy:>10.2f}")
    print(f"  Expectancy (R):    {m.expectancy_r:>10.2f}R")
    
    print("\n--- P&L ---")
    print(f"  Gross Profit:      ${m.gross_profit:>12,.2f}")
    print(f"  Gross Loss:        ${m.gross_loss:>12,.2f}")
    print(f"  Net Profit:        ${m.net_profit:>12,.2f}")
    print(f"  Avg Win:           ${m.avg_win:>12,.2f}")
    print(f"  Avg Loss:          ${m.avg_loss:>12,.2f}")
    
    print("\n--- COSTS ---")
    print(f"  Commissions:       ${m.total_commissions:>12,.2f}")
    print(f"  Slippage:          ${m.total_slippage:>12,.2f}")
    print(f"  Total Costs:       ${m.total_costs:>12,.2f}")
    
    print("=" * 60)
