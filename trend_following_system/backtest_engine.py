"""
Trend Following System - Backtest Engine
Sequential event-driven backtesting with realistic execution.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from config import TRADING_CONFIG
from data_loader import DataLoader
from indicators import IndicatorEngine
from signal_engine import SignalEngine, Signal, SignalType
from position_sizer import PositionSizer
from portfolio_engine import PortfolioEngine, Trade


@dataclass
class BacktestMetrics:
    """Comprehensive backtest performance metrics."""
    # Returns
    total_return: float
    cagr: float
    
    # Risk
    max_drawdown: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    # Profit metrics
    profit_factor: float
    expectancy: float          # Average profit per trade
    expectancy_r: float        # Expectancy in R-multiples
    avg_win: float
    avg_loss: float
    avg_win_r: float
    avg_loss_r: float
    largest_win: float
    largest_loss: float
    
    # Efficiency
    avg_holding_days: float
    trades_per_year: float
    exposure_pct: float
    
    # Costs
    total_costs: float
    cost_per_trade: float
    
    # Time period
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    years: float


class BacktestEngine:
    """
    Event-driven backtesting engine.
    
    Execution model:
    - Signals generated at end of day (Close)
    - Orders executed at next day's Open
    - Costs applied realistically
    
    Order of operations each day:
    1. Execute pending orders at Open
    2. Update trailing stops
    3. Check exit conditions
    4. Check entry conditions
    5. Generate new signals
    6. Record equity
    """
    
    def __init__(self, initial_capital: float = None):
        """Initialize backtest engine."""
        if initial_capital is None:
            initial_capital = TRADING_CONFIG.INITIAL_CAPITAL
        
        self.initial_capital = initial_capital
        
        # Components
        self.data_loader = DataLoader()
        self.indicator_engine = IndicatorEngine()
        self.signal_engine = SignalEngine()
        self.position_sizer = PositionSizer()
        self.portfolio = PortfolioEngine(initial_capital)
        
        # State
        self._data: Dict[str, pd.DataFrame] = {}
        self._pending_signals: List[Signal] = []
        self._all_dates: List[pd.Timestamp] = []
    
    def run(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> BacktestMetrics:
        """
        Run backtest over specified period.
        
        Args:
            start_date: Start of backtest period
            end_date: End of backtest period
            
        Returns:
            BacktestMetrics with performance statistics
        """
        print("=" * 60)
        print("BACKTEST EXECUTION")
        print("=" * 60)
        
        # Load and prepare data
        print("\n1. Loading data...")
        raw_data = self.data_loader.load_universe()
        
        print("\n2. Calculating indicators...")
        self._data = self.indicator_engine.calculate_universe(raw_data)
        
        # Get aligned dates
        self._all_dates = self._get_aligned_dates(start_date, end_date)
        print(f"\n3. Backtest period: {self._all_dates[0].date()} to {self._all_dates[-1].date()}")
        print(f"   Total bars: {len(self._all_dates)}")
        
        # Reset state
        self.portfolio.reset()
        self.signal_engine.reset()
        self._pending_signals = []
        
        # Run simulation
        print("\n4. Running simulation...")
        self._run_simulation()
        
        # Calculate metrics
        print("\n5. Calculating metrics...")
        metrics = self._calculate_metrics()
        
        return metrics
    
    def _get_aligned_dates(
        self,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
    ) -> List[pd.Timestamp]:
        """Get trading dates present in all symbols."""
        # Get intersection of all date indices
        date_sets = [set(df.index) for df in self._data.values()]
        common_dates = set.intersection(*date_sets)
        
        # Sort and filter
        dates = sorted(common_dates)
        
        # Apply warmup period
        warmup = self.indicator_engine.get_warmup_period()
        dates = dates[warmup:]
        
        # Apply date filters
        if start_date:
            dates = [d for d in dates if d >= pd.Timestamp(start_date)]
        if end_date:
            dates = [d for d in dates if d <= pd.Timestamp(end_date)]
        
        return dates
    
    def _run_simulation(self):
        """Execute simulation day by day."""
        total_days = len(self._all_dates)
        
        for i, date in enumerate(self._all_dates):
            # Progress indicator
            if i % 500 == 0:
                print(f"   Processing day {i+1}/{total_days}...")
            
            # Get current prices
            prices = {
                symbol: df.loc[date, 'Close']
                for symbol, df in self._data.items()
                if date in df.index
            }
            
            opens = {
                symbol: df.loc[date, 'Open']
                for symbol, df in self._data.items()
                if date in df.index
            }
            
            # 1. Execute pending orders at Open
            self._execute_pending_orders(date, opens)
            
            # 2. Update trailing stops
            self.portfolio.update_stops(date, self._data)
            
            # 3. Check exits (at Close)
            self._check_exits(date, prices)
            
            # 4. Check entries and generate signals (at Close)
            self._check_entries(date)
            
            # 5. Record equity
            self.portfolio.update_equity_curve(date, prices)
    
    def _execute_pending_orders(
        self,
        date: pd.Timestamp,
        opens: Dict[str, float],
    ):
        """Execute pending entry orders at market open."""
        executed = []
        
        for signal in self._pending_signals:
            if signal.signal_type != SignalType.ENTRY:
                continue
            
            if signal.symbol not in opens:
                continue
            
            # Check if we can still enter (position limits)
            if not self.portfolio.can_add_position:
                continue
            
            # Calculate execution price with slippage
            open_price = opens[signal.symbol]
            slippage = open_price * TRADING_CONFIG.SLIPPAGE_PCT
            execution_price = open_price + slippage  # Buy at higher price
            
            # Get current equity for position sizing
            equity = self.portfolio.get_equity(opens)
            
            # Calculate position size
            try:
                size = self.position_sizer.calculate(
                    equity=equity,
                    entry_price=execution_price,
                    stop_price=signal.stop_price,
                    available_capital=self.portfolio.cash,
                )
            except ValueError:
                continue
            
            if size.shares < 1:
                continue
            
            # Calculate costs
            commission = size.shares * TRADING_CONFIG.COMMISSION_PER_SHARE
            entry_cost = commission + slippage * size.shares
            
            # Execute entry
            success = self.portfolio.process_entry(
                signal=signal,
                shares=size.shares,
                execution_price=execution_price,
                entry_cost=entry_cost,
            )
            
            if success:
                # Register with signal engine for exit tracking
                self.signal_engine.register_entry(
                    symbol=signal.symbol,
                    entry_price=execution_price,
                    entry_date=date,
                    shares=size.shares,
                    stop_price=signal.stop_price,
                )
                executed.append(signal)
        
        # Clear executed signals
        for signal in executed:
            self._pending_signals.remove(signal)
    
    def _check_exits(self, date: pd.Timestamp, prices: Dict[str, float]):
        """Check for exit signals and execute immediately."""
        for symbol in list(self.portfolio.positions.keys()):
            if symbol not in self._data:
                continue
            
            df = self._data[symbol]
            if date not in df.index:
                continue
            
            row = df.loc[date]
            
            # Check exit conditions
            signal = self.signal_engine.check_exit(date, symbol, row)
            
            if signal:
                # Calculate execution price (at close, with slippage)
                close_price = prices[symbol]
                slippage = close_price * TRADING_CONFIG.SLIPPAGE_PCT
                execution_price = close_price - slippage  # Sell at lower price
                
                # Calculate costs
                position = self.portfolio.positions[symbol]
                commission = position.shares * TRADING_CONFIG.COMMISSION_PER_SHARE
                exit_cost = commission + slippage * position.shares
                
                # Execute exit
                trade = self.portfolio.process_exit(
                    signal=signal,
                    execution_price=execution_price,
                    exit_cost=exit_cost,
                )
                
                if trade:
                    self.signal_engine.register_exit(symbol)
    
    def _check_entries(self, date: pd.Timestamp):
        """Check for entry signals (to be executed next day)."""
        if not self.portfolio.can_add_position:
            return
        
        signals_today = []
        
        for symbol in TRADING_CONFIG.SYMBOLS:
            if symbol not in self._data:
                continue
            
            # Skip if already have position
            if symbol in self.portfolio.positions:
                continue
            
            df = self._data[symbol]
            if date not in df.index:
                continue
            
            row = df.loc[date]
            
            # Check entry conditions
            signal = self.signal_engine.check_entry(date, symbol, row)
            
            if signal:
                signals_today.append(signal)
        
        # If multiple signals, prioritize by ATR (higher volatility = more potential)
        if len(signals_today) > 0:
            signals_today.sort(key=lambda s: s.atr, reverse=True)
            
            # Only queue signals up to position limit
            available_slots = TRADING_CONFIG.MAX_POSITIONS - self.portfolio.position_count
            for signal in signals_today[:available_slots]:
                self._pending_signals.append(signal)
    
    def _calculate_metrics(self) -> BacktestMetrics:
        """Calculate comprehensive performance metrics."""
        trades_df = self.portfolio.get_trades_df()
        equity_df = self.portfolio.get_equity_df()
        
        if len(trades_df) == 0 or len(equity_df) == 0:
            raise ValueError("No trades or equity data to analyze")
        
        # Time period
        start_date = equity_df.index[0]
        end_date = equity_df.index[-1]
        years = (end_date - start_date).days / 365.25
        
        # Returns
        final_equity = equity_df['Equity'].iloc[-1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        cagr = (final_equity / self.initial_capital) ** (1 / years) - 1
        
        # Risk metrics
        max_drawdown = equity_df['Drawdown'].max()
        
        # Daily returns for volatility calculation
        equity_df['Return'] = equity_df['Equity'].pct_change()
        daily_returns = equity_df['Return'].dropna()
        
        volatility = daily_returns.std() * np.sqrt(252)
        
        # Sharpe ratio (assuming 4% risk-free rate)
        risk_free = 0.04 / 252  # Daily
        excess_returns = daily_returns - risk_free
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = daily_returns[daily_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0.0001
        sortino_ratio = (cagr - 0.04) / downside_std if downside_std > 0 else 0
        
        # Calmar ratio
        calmar_ratio = cagr / max_drawdown if max_drawdown > 0 else 0
        
        # Trade statistics
        total_trades = len(trades_df)
        winning_trades = trades_df['Won'].sum()
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # PnL analysis
        winners = trades_df[trades_df['Won']]
        losers = trades_df[~trades_df['Won']]
        
        gross_profit = winners['PnL'].sum() if len(winners) > 0 else 0
        gross_loss = abs(losers['PnL'].sum()) if len(losers) > 0 else 0.0001
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        avg_win = winners['PnL'].mean() if len(winners) > 0 else 0
        avg_loss = losers['PnL'].mean() if len(losers) > 0 else 0  # Negative
        
        avg_win_r = winners['PnL_R'].mean() if len(winners) > 0 else 0
        avg_loss_r = losers['PnL_R'].mean() if len(losers) > 0 else 0
        
        largest_win = trades_df['PnL'].max()
        largest_loss = trades_df['PnL'].min()
        
        # Expectancy
        expectancy = trades_df['PnL'].mean()
        expectancy_r = trades_df['PnL_R'].mean()
        
        # Efficiency
        avg_holding_days = trades_df['Holding_Days'].mean()
        trades_per_year = total_trades / years
        
        # Exposure calculation
        # Rough estimate: average position count / max positions
        equity_df['Has_Position'] = (equity_df['Equity'] != equity_df['Equity'].shift(1)).cumsum() % 2
        exposure_pct = (1 - (equity_df['Equity'] == self.initial_capital).mean())
        
        # Costs
        total_costs = trades_df['Entry_Cost'].sum() + trades_df['Exit_Cost'].sum() if 'Entry_Cost' in trades_df.columns else 0
        cost_per_trade = total_costs / total_trades if total_trades > 0 else 0
        
        return BacktestMetrics(
            total_return=total_return,
            cagr=cagr,
            max_drawdown=max_drawdown,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            expectancy=expectancy,
            expectancy_r=expectancy_r,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_win_r=avg_win_r,
            avg_loss_r=avg_loss_r,
            largest_win=largest_win,
            largest_loss=largest_loss,
            avg_holding_days=avg_holding_days,
            trades_per_year=trades_per_year,
            exposure_pct=exposure_pct,
            total_costs=total_costs,
            cost_per_trade=cost_per_trade,
            start_date=start_date,
            end_date=end_date,
            years=years,
        )
    
    def get_trades(self) -> pd.DataFrame:
        """Get trade history."""
        return self.portfolio.get_trades_df()
    
    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve."""
        return self.portfolio.get_equity_df()


def print_metrics(metrics: BacktestMetrics):
    """Print formatted backtest metrics."""
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    
    print(f"\nPeriod: {metrics.start_date.date()} to {metrics.end_date.date()} ({metrics.years:.1f} years)")
    
    print("\n--- RETURNS ---")
    print(f"Total Return:      {metrics.total_return:>10.1%}")
    print(f"CAGR:              {metrics.cagr:>10.1%}")
    
    print("\n--- RISK ---")
    print(f"Max Drawdown:      {metrics.max_drawdown:>10.1%}")
    print(f"Volatility:        {metrics.volatility:>10.1%}")
    print(f"Sharpe Ratio:      {metrics.sharpe_ratio:>10.2f}")
    print(f"Sortino Ratio:     {metrics.sortino_ratio:>10.2f}")
    print(f"Calmar Ratio:      {metrics.calmar_ratio:>10.2f}")
    
    print("\n--- TRADE STATISTICS ---")
    print(f"Total Trades:      {metrics.total_trades:>10}")
    print(f"Win Rate:          {metrics.win_rate:>10.1%}")
    print(f"Profit Factor:     {metrics.profit_factor:>10.2f}")
    print(f"Expectancy ($):    {metrics.expectancy:>10.2f}")
    print(f"Expectancy (R):    {metrics.expectancy_r:>10.2f}")
    
    print("\n--- WIN/LOSS ANALYSIS ---")
    print(f"Avg Win ($):       {metrics.avg_win:>10.2f}")
    print(f"Avg Loss ($):      {metrics.avg_loss:>10.2f}")
    print(f"Avg Win (R):       {metrics.avg_win_r:>10.2f}")
    print(f"Avg Loss (R):      {metrics.avg_loss_r:>10.2f}")
    print(f"Largest Win:       {metrics.largest_win:>10.2f}")
    print(f"Largest Loss:      {metrics.largest_loss:>10.2f}")
    
    print("\n--- EFFICIENCY ---")
    print(f"Trades/Year:       {metrics.trades_per_year:>10.1f}")
    print(f"Avg Holding Days:  {metrics.avg_holding_days:>10.1f}")
    print(f"Exposure:          {metrics.exposure_pct:>10.1%}")
    
    print("\n--- COSTS ---")
    print(f"Total Costs:       ${metrics.total_costs:>9.2f}")
    print(f"Cost/Trade:        ${metrics.cost_per_trade:>9.2f}")
    
    print("=" * 60)


if __name__ == "__main__":
    # Quick test
    engine = BacktestEngine()
    metrics = engine.run()
    print_metrics(metrics)
