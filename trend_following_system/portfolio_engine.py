"""
Trend Following System - Portfolio Engine
Multi-asset coordination with position limits.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from config import TRADING_CONFIG
from signal_engine import Signal, SignalType


@dataclass
class Position:
    """Active position tracking."""
    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: int
    stop_price: float
    entry_cost: float = 0.0
    
    @property
    def position_value(self) -> float:
        return self.shares * self.entry_price
    
    def current_value(self, price: float) -> float:
        return self.shares * price
    
    def unrealized_pnl(self, price: float) -> float:
        return (price - self.entry_price) * self.shares - self.entry_cost
    
    def pnl_r(self, price: float) -> float:
        """PnL in R-multiples."""
        risk = (self.entry_price - self.stop_price) * self.shares
        if risk <= 0:
            return 0.0
        return self.unrealized_pnl(price) / risk


@dataclass
class Trade:
    """Completed trade record."""
    symbol: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    shares: int
    initial_stop: float
    pnl: float
    pnl_pct: float
    pnl_r: float
    holding_days: int
    entry_cost: float
    exit_cost: float
    
    @property
    def won(self) -> bool:
        return self.pnl > 0


@dataclass
class PortfolioState:
    """Current portfolio state."""
    cash: float
    equity: float
    positions: Dict[str, Position]
    position_count: int
    total_value: float
    unrealized_pnl: float
    
    @property
    def exposure(self) -> float:
        """Percentage of equity in positions."""
        if self.equity <= 0:
            return 0.0
        return (self.total_value - self.cash) / self.equity


class PortfolioEngine:
    """
    Manages portfolio of positions across multiple assets.
    
    Responsibilities:
    - Track cash and positions
    - Enforce position limits
    - Process signals in priority order
    - Record trade history
    - Calculate portfolio metrics
    """
    
    def __init__(self, initial_capital: float = None):
        """Initialize portfolio with starting capital."""
        if initial_capital is None:
            initial_capital = TRADING_CONFIG.INITIAL_CAPITAL
        
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[pd.Timestamp, float]] = []
        
        self._peak_equity = initial_capital
        self._max_drawdown = 0.0
    
    @property
    def position_count(self) -> int:
        return len(self.positions)
    
    @property
    def can_add_position(self) -> bool:
        return self.position_count < TRADING_CONFIG.MAX_POSITIONS
    
    def get_equity(self, prices: Dict[str, float] = None) -> float:
        """Calculate current equity value."""
        if prices is None:
            prices = {}
        
        position_value = sum(
            pos.current_value(prices.get(symbol, pos.entry_price))
            for symbol, pos in self.positions.items()
        )
        
        return self.cash + position_value
    
    def get_state(self, prices: Dict[str, float] = None) -> PortfolioState:
        """Get current portfolio state."""
        if prices is None:
            prices = {}
        
        equity = self.get_equity(prices)
        position_value = equity - self.cash
        
        unrealized = sum(
            pos.unrealized_pnl(prices.get(symbol, pos.entry_price))
            for symbol, pos in self.positions.items()
        )
        
        return PortfolioState(
            cash=self.cash,
            equity=equity,
            positions=self.positions.copy(),
            position_count=self.position_count,
            total_value=equity,
            unrealized_pnl=unrealized,
        )
    
    def process_entry(
        self,
        signal: Signal,
        shares: int,
        execution_price: float,
        entry_cost: float,
    ) -> bool:
        """
        Execute an entry signal.
        
        Args:
            signal: Entry signal
            shares: Number of shares to buy
            execution_price: Actual execution price
            entry_cost: Transaction costs
            
        Returns:
            True if entry executed, False otherwise
        """
        if not self.can_add_position:
            return False
        
        if signal.symbol in self.positions:
            return False
        
        total_cost = (shares * execution_price) + entry_cost
        
        if total_cost > self.cash:
            return False
        
        # Execute entry
        self.cash -= total_cost
        
        self.positions[signal.symbol] = Position(
            symbol=signal.symbol,
            entry_date=signal.date,
            entry_price=execution_price,
            shares=shares,
            stop_price=signal.stop_price,
            entry_cost=entry_cost,
        )
        
        return True
    
    def process_exit(
        self,
        signal: Signal,
        execution_price: float,
        exit_cost: float,
    ) -> Optional[Trade]:
        """
        Execute an exit signal.
        
        Args:
            signal: Exit signal
            execution_price: Actual execution price
            exit_cost: Transaction costs
            
        Returns:
            Completed Trade record, or None if no position
        """
        if signal.symbol not in self.positions:
            return None
        
        position = self.positions[signal.symbol]
        
        # Calculate proceeds
        gross_proceeds = position.shares * execution_price
        net_proceeds = gross_proceeds - exit_cost
        
        # Calculate PnL
        total_cost = position.position_value + position.entry_cost
        pnl = net_proceeds - total_cost
        pnl_pct = pnl / total_cost
        
        # Calculate R-multiple
        initial_risk = (position.entry_price - position.stop_price) * position.shares
        pnl_r = pnl / initial_risk if initial_risk > 0 else 0.0
        
        # Holding period
        holding_days = (signal.date - position.entry_date).days
        
        # Create trade record
        trade = Trade(
            symbol=signal.symbol,
            entry_date=position.entry_date,
            exit_date=signal.date,
            entry_price=position.entry_price,
            exit_price=execution_price,
            shares=position.shares,
            initial_stop=position.stop_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            pnl_r=pnl_r,
            holding_days=holding_days,
            entry_cost=position.entry_cost,
            exit_cost=exit_cost,
        )
        
        # Execute exit
        self.cash += net_proceeds
        del self.positions[signal.symbol]
        self.trades.append(trade)
        
        return trade
    
    def update_equity_curve(
        self,
        date: pd.Timestamp,
        prices: Dict[str, float],
    ):
        """Record equity value for date."""
        equity = self.get_equity(prices)
        self.equity_curve.append((date, equity))
        
        # Update drawdown tracking
        if equity > self._peak_equity:
            self._peak_equity = equity
        
        dd = (self._peak_equity - equity) / self._peak_equity
        self._max_drawdown = max(self._max_drawdown, dd)
    
    def update_stops(self, date: pd.Timestamp, data: Dict[str, pd.DataFrame]):
        """Update trailing stops for all positions."""
        for symbol, position in self.positions.items():
            if symbol in data:
                df = data[symbol]
                if date in df.index:
                    new_stop = df.loc[date, 'Trailing_Stop']
                    if not pd.isna(new_stop) and new_stop > position.stop_price:
                        position.stop_price = new_stop
    
    def get_max_drawdown(self) -> float:
        """Get maximum drawdown experienced."""
        return self._max_drawdown
    
    def get_equity_df(self) -> pd.DataFrame:
        """Get equity curve as DataFrame."""
        if not self.equity_curve:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.equity_curve, columns=['Date', 'Equity'])
        df = df.set_index('Date')
        
        # Calculate drawdown series
        df['Peak'] = df['Equity'].cummax()
        df['Drawdown'] = (df['Peak'] - df['Equity']) / df['Peak']
        
        return df
    
    def get_trades_df(self) -> pd.DataFrame:
        """Get trade history as DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        
        records = []
        for t in self.trades:
            records.append({
                'Symbol': t.symbol,
                'Entry_Date': t.entry_date,
                'Exit_Date': t.exit_date,
                'Entry_Price': t.entry_price,
                'Exit_Price': t.exit_price,
                'Shares': t.shares,
                'Initial_Stop': t.initial_stop,
                'PnL': t.pnl,
                'PnL_Pct': t.pnl_pct,
                'PnL_R': t.pnl_r,
                'Holding_Days': t.holding_days,
                'Won': t.won,
            })
        
        return pd.DataFrame(records)
    
    def reset(self):
        """Reset portfolio to initial state."""
        self.cash = self.initial_capital
        self.positions.clear()
        self.trades.clear()
        self.equity_curve.clear()
        self._peak_equity = self.initial_capital
        self._max_drawdown = 0.0


if __name__ == "__main__":
    # Test portfolio engine
    print("Portfolio Engine Test")
    print("=" * 60)
    
    portfolio = PortfolioEngine(100_000)
    
    print(f"Initial capital: ${portfolio.cash:,.2f}")
    print(f"Max positions: {TRADING_CONFIG.MAX_POSITIONS}")
    
    # Simulate entry
    from signal_engine import Signal, SignalType
    
    signal = Signal(
        date=pd.Timestamp('2024-01-15'),
        symbol='SPY',
        signal_type=SignalType.ENTRY,
        price=450.0,
        entry_price=None,
        stop_price=430.0,
        atr=5.0,
    )
    
    print(f"\nProcessing entry: {signal}")
    
    success = portfolio.process_entry(
        signal=signal,
        shares=25,
        execution_price=450.50,
        entry_cost=5.0,
    )
    
    print(f"Entry executed: {success}")
    
    state = portfolio.get_state({'SPY': 455.0})
    print(f"\nPortfolio State:")
    print(f"  Cash: ${state.cash:,.2f}")
    print(f"  Equity: ${state.equity:,.2f}")
    print(f"  Positions: {state.position_count}")
    print(f"  Unrealized P&L: ${state.unrealized_pnl:,.2f}")
    print(f"  Exposure: {state.exposure:.1%}")
