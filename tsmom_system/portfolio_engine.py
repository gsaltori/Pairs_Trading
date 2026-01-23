"""
Time-Series Momentum System - Portfolio Engine
Position tracking and rebalancing with MANDATORY sanity checks.

SANITY CHECKS ENFORCED:
1. Equity must NEVER be <= 0
2. Sum of weights must ALWAYS be <= 1.0 (no leverage)
3. Cash must ALWAYS be >= 0
4. Trades ONLY on rebalance dates
5. Cash debited on buys, credited on sells
6. Position integrity maintained

If ANY check fails: ABORT with explicit error.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from config import CONFIG
from signal_engine import TSMOMSignal, SanityCheckError


@dataclass
class Position:
    """Single position with cost tracking."""
    symbol: str
    shares: float
    avg_cost: float      # Average cost per share
    cost_basis: float    # Total cost basis
    
    def market_value(self, price: float) -> float:
        """Current market value."""
        return self.shares * price
    
    def unrealized_pnl(self, price: float) -> float:
        """Unrealized P&L."""
        return self.market_value(price) - self.cost_basis


@dataclass
class Trade:
    """Completed trade record."""
    date: pd.Timestamp
    symbol: str
    side: str            # "BUY" or "SELL"
    shares: float
    price: float
    gross_value: float
    commission: float
    slippage_cost: float
    net_value: float
    realized_pnl: Optional[float] = None  # For sells


class PortfolioEngine:
    """
    Portfolio management with mandatory sanity checks.
    
    ALL operations validate state integrity.
    Execution ABORTS if any check fails.
    """
    
    def __init__(self, initial_capital: float = None):
        if initial_capital is None:
            initial_capital = CONFIG.INITIAL_CAPITAL
        
        if initial_capital <= 0:
            raise SanityCheckError(f"Initial capital must be > 0: {initial_capital}")
        
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[pd.Timestamp, float, float, Dict[str, float]]] = []
        
        self._prev_equity = initial_capital
        self._rebalance_dates: set = set()
        self._total_commissions = 0.0
        self._total_slippage = 0.0
    
    # =========================================================================
    # SANITY CHECKS - MANDATORY
    # =========================================================================
    
    def _check_equity_positive(self, prices: Dict[str, float], context: str):
        """SANITY: Equity must be > 0."""
        equity = self.get_equity(prices)
        if equity <= 0:
            raise SanityCheckError(
                f"[{context}] EQUITY <= 0: ${equity:.2f}\n"
                f"  Cash: ${self.cash:.2f}\n"
                f"  Positions: {list(self.positions.keys())}"
            )
    
    def _check_no_leverage(self, prices: Dict[str, float], context: str):
        """SANITY: No leverage (invested <= equity)."""
        equity = self.get_equity(prices)
        if equity <= 0:
            return
        
        invested = sum(
            pos.market_value(prices.get(sym, pos.avg_cost))
            for sym, pos in self.positions.items()
        )
        
        if invested > equity * 1.001:  # 0.1% tolerance
            raise SanityCheckError(
                f"[{context}] LEVERAGE DETECTED:\n"
                f"  Invested: ${invested:.2f}\n"
                f"  Equity: ${equity:.2f}\n"
                f"  Ratio: {invested/equity:.2%}"
            )
    
    def _check_cash_non_negative(self, context: str):
        """SANITY: Cash must be >= 0."""
        if self.cash < -1.0:  # $1 tolerance for float errors
            raise SanityCheckError(
                f"[{context}] NEGATIVE CASH: ${self.cash:.2f}"
            )
    
    def _check_positions_valid(self, context: str):
        """SANITY: All positions must have positive shares."""
        for sym, pos in self.positions.items():
            if pos.shares <= 0:
                raise SanityCheckError(
                    f"[{context}] Invalid position {sym}: {pos.shares} shares"
                )
            if pos.cost_basis <= 0:
                raise SanityCheckError(
                    f"[{context}] Invalid cost basis for {sym}: ${pos.cost_basis:.2f}"
                )
    
    def run_all_sanity_checks(self, prices: Dict[str, float], context: str = ""):
        """Run ALL sanity checks."""
        self._check_equity_positive(prices, context)
        self._check_no_leverage(prices, context)
        self._check_cash_non_negative(context)
        self._check_positions_valid(context)
    
    # =========================================================================
    # CORE METHODS
    # =========================================================================
    
    def get_equity(self, prices: Dict[str, float]) -> float:
        """Calculate current equity = cash + positions."""
        position_value = sum(
            pos.market_value(prices.get(sym, pos.avg_cost))
            for sym, pos in self.positions.items()
        )
        return self.cash + position_value
    
    def get_position_weights(self, prices: Dict[str, float]) -> Dict[str, float]:
        """Get current weight of each position."""
        equity = self.get_equity(prices)
        if equity <= 0:
            return {}
        
        weights = {}
        for sym, pos in self.positions.items():
            weights[sym] = pos.market_value(prices.get(sym, pos.avg_cost)) / equity
        
        return weights
    
    def rebalance_to_weights(
        self,
        target_weights: Dict[str, float],
        prices: Dict[str, float],
        date: pd.Timestamp,
    ) -> List[Trade]:
        """
        Rebalance portfolio to target weights.
        
        Order of operations:
        1. Pre-rebalance sanity checks
        2. Calculate target values
        3. Execute SELLS first (free up cash)
        4. Execute BUYS
        5. Post-rebalance sanity checks
        
        Returns list of executed trades.
        """
        # Record this as a rebalance date
        self._rebalance_dates.add(date)
        
        # Pre-rebalance sanity checks
        self.run_all_sanity_checks(prices, f"pre-rebalance {date.date()}")
        
        equity = self.get_equity(prices)
        executed_trades = []
        
        # Calculate target values for each asset
        target_values = {}
        for sym, weight in target_weights.items():
            if weight > 0.001:  # Ignore tiny weights
                target_values[sym] = equity * weight
        
        # Current values
        current_values = {}
        for sym, pos in self.positions.items():
            current_values[sym] = pos.market_value(prices.get(sym, pos.avg_cost))
        
        # =====================================================================
        # STEP 1: CLOSE positions not in target
        # =====================================================================
        for sym in list(self.positions.keys()):
            if sym not in target_values:
                trade = self._close_position(sym, prices, date)
                if trade:
                    executed_trades.append(trade)
        
        # =====================================================================
        # STEP 2: REDUCE oversized positions
        # =====================================================================
        for sym, pos in list(self.positions.items()):
            if sym in target_values:
                target_val = target_values[sym]
                current_val = current_values.get(sym, 0)
                
                # If current > target by more than 5%, reduce
                if current_val > target_val * 1.05:
                    reduce_value = current_val - target_val
                    price = prices.get(sym, pos.avg_cost)
                    shares_to_sell = reduce_value / price
                    
                    trade = self._sell_shares(sym, shares_to_sell, prices, date)
                    if trade:
                        executed_trades.append(trade)
        
        # =====================================================================
        # STEP 3: INCREASE undersized positions and OPEN new positions
        # =====================================================================
        for sym, target_val in target_values.items():
            current_val = 0
            if sym in self.positions:
                current_val = self.positions[sym].market_value(
                    prices.get(sym, self.positions[sym].avg_cost)
                )
            
            # If target > current by more than 5%, buy
            if target_val > current_val * 1.05:
                buy_value = target_val - current_val
                
                # Don't exceed available cash (leave $1 buffer)
                buy_value = min(buy_value, self.cash - 1)
                
                if buy_value > 100:  # Minimum trade size
                    price = prices.get(sym)
                    if price and price > 0:
                        shares_to_buy = buy_value / price
                        trade = self._buy_shares(sym, shares_to_buy, prices, date)
                        if trade:
                            executed_trades.append(trade)
        
        # Post-rebalance sanity checks
        self.run_all_sanity_checks(prices, f"post-rebalance {date.date()}")
        
        return executed_trades
    
    def _buy_shares(
        self,
        symbol: str,
        shares: float,
        prices: Dict[str, float],
        date: pd.Timestamp,
    ) -> Optional[Trade]:
        """Execute buy order with costs."""
        if shares <= 0:
            return None
        
        price = prices.get(symbol)
        if not price or price <= 0:
            return None
        
        # Apply slippage (buy at higher price)
        slippage = price * CONFIG.SLIPPAGE_PCT
        exec_price = price + slippage
        
        # Calculate costs
        gross_value = shares * exec_price
        commission = gross_value * CONFIG.COMMISSION_PCT
        total_cost = gross_value + commission
        
        # Check if we have enough cash
        if total_cost > self.cash:
            # Reduce shares to fit
            available = self.cash - 1  # Leave $1 buffer
            if available <= commission:
                return None
            
            shares = (available - commission) / exec_price
            gross_value = shares * exec_price
            commission = gross_value * CONFIG.COMMISSION_PCT
            total_cost = gross_value + commission
        
        if shares <= 0:
            return None
        
        # DEBIT cash
        self.cash -= total_cost
        self._total_commissions += commission
        self._total_slippage += shares * slippage
        
        # Update or create position
        if symbol in self.positions:
            old = self.positions[symbol]
            new_shares = old.shares + shares
            new_cost_basis = old.cost_basis + gross_value
            
            self.positions[symbol] = Position(
                symbol=symbol,
                shares=new_shares,
                avg_cost=new_cost_basis / new_shares,
                cost_basis=new_cost_basis,
            )
        else:
            self.positions[symbol] = Position(
                symbol=symbol,
                shares=shares,
                avg_cost=exec_price,
                cost_basis=gross_value,
            )
        
        # Record trade
        trade = Trade(
            date=date,
            symbol=symbol,
            side="BUY",
            shares=shares,
            price=exec_price,
            gross_value=gross_value,
            commission=commission,
            slippage_cost=shares * slippage,
            net_value=total_cost,
        )
        self.trades.append(trade)
        
        return trade
    
    def _sell_shares(
        self,
        symbol: str,
        shares: float,
        prices: Dict[str, float],
        date: pd.Timestamp,
    ) -> Optional[Trade]:
        """Execute partial sell with costs."""
        if symbol not in self.positions:
            return None
        
        pos = self.positions[symbol]
        shares = min(shares, pos.shares)  # Can't sell more than we have
        
        if shares <= 0:
            return None
        
        price = prices.get(symbol, pos.avg_cost)
        
        # Apply slippage (sell at lower price)
        slippage = price * CONFIG.SLIPPAGE_PCT
        exec_price = price - slippage
        
        # Calculate proceeds
        gross_proceeds = shares * exec_price
        commission = gross_proceeds * CONFIG.COMMISSION_PCT
        net_proceeds = gross_proceeds - commission
        
        # Calculate proportional cost basis for P&L
        proportion_sold = shares / pos.shares
        cost_basis_sold = pos.cost_basis * proportion_sold
        realized_pnl = net_proceeds - cost_basis_sold
        
        # CREDIT cash
        self.cash += net_proceeds
        self._total_commissions += commission
        self._total_slippage += shares * slippage
        
        # Update position
        remaining_shares = pos.shares - shares
        
        if remaining_shares < 0.001:  # Fully closed
            del self.positions[symbol]
        else:
            # Reduce position proportionally
            self.positions[symbol] = Position(
                symbol=symbol,
                shares=remaining_shares,
                avg_cost=pos.avg_cost,
                cost_basis=pos.cost_basis * (1 - proportion_sold),
            )
        
        # Record trade
        trade = Trade(
            date=date,
            symbol=symbol,
            side="SELL",
            shares=shares,
            price=exec_price,
            gross_value=gross_proceeds,
            commission=commission,
            slippage_cost=shares * slippage,
            net_value=net_proceeds,
            realized_pnl=realized_pnl,
        )
        self.trades.append(trade)
        
        return trade
    
    def _close_position(
        self,
        symbol: str,
        prices: Dict[str, float],
        date: pd.Timestamp,
    ) -> Optional[Trade]:
        """Close entire position."""
        if symbol not in self.positions:
            return None
        
        return self._sell_shares(symbol, self.positions[symbol].shares, prices, date)
    
    def update_equity_curve(self, date: pd.Timestamp, prices: Dict[str, float]):
        """Record equity and validate."""
        equity = self.get_equity(prices)
        
        if equity <= 0:
            raise SanityCheckError(
                f"Equity dropped to ${equity:.2f} on {date}"
            )
        
        weights = self.get_position_weights(prices)
        cash_pct = self.cash / equity if equity > 0 else 1.0
        
        self.equity_curve.append((date, equity, cash_pct, weights.copy()))
        self._prev_equity = equity
    
    def get_equity_df(self) -> pd.DataFrame:
        """Get equity curve as DataFrame."""
        if not self.equity_curve:
            return pd.DataFrame()
        
        data = []
        for date, equity, cash_pct, weights in self.equity_curve:
            data.append({
                'Date': date,
                'Equity': equity,
                'Cash_Pct': cash_pct,
            })
        
        df = pd.DataFrame(data)
        df = df.set_index('Date')
        
        df['Peak'] = df['Equity'].cummax()
        df['Drawdown'] = (df['Peak'] - df['Equity']) / df['Peak']
        df['Return'] = df['Equity'].pct_change()
        
        return df
    
    def get_trades_df(self) -> pd.DataFrame:
        """Get trades as DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        
        records = []
        for t in self.trades:
            records.append({
                'Date': t.date,
                'Symbol': t.symbol,
                'Side': t.side,
                'Shares': t.shares,
                'Price': t.price,
                'Gross_Value': t.gross_value,
                'Commission': t.commission,
                'Slippage': t.slippage_cost,
                'Net_Value': t.net_value,
                'Realized_PnL': t.realized_pnl,
            })
        
        return pd.DataFrame(records)
    
    def get_monthly_returns(self) -> pd.Series:
        """Calculate monthly returns from equity curve."""
        df = self.get_equity_df()
        if len(df) == 0:
            return pd.Series()
        
        monthly = df['Equity'].resample('ME').last()
        returns = monthly.pct_change().dropna()
        return returns
    
    def get_cost_summary(self) -> Dict:
        """Get summary of transaction costs."""
        return {
            'total_commissions': self._total_commissions,
            'total_slippage': self._total_slippage,
            'total_costs': self._total_commissions + self._total_slippage,
            'trade_count': len(self.trades),
        }
    
    def reset(self):
        """Reset to initial state."""
        self.cash = self.initial_capital
        self.positions.clear()
        self.trades.clear()
        self.equity_curve.clear()
        self._prev_equity = self.initial_capital
        self._rebalance_dates.clear()
        self._total_commissions = 0.0
        self._total_slippage = 0.0


def validate_portfolio_state(
    portfolio: PortfolioEngine,
    prices: Dict[str, float],
) -> Tuple[bool, List[str]]:
    """Final validation of portfolio state."""
    issues = []
    
    # Equity positive
    equity = portfolio.get_equity(prices)
    if equity <= 0:
        issues.append(f"Final equity <= 0: ${equity:.2f}")
    
    # Cash non-negative
    if portfolio.cash < -1:
        issues.append(f"Negative cash: ${portfolio.cash:.2f}")
    
    # Check equity curve
    df = portfolio.get_equity_df()
    if len(df) > 0:
        if (df['Equity'] <= 0).any():
            min_eq = df['Equity'].min()
            min_date = df['Equity'].idxmin()
            issues.append(f"Equity went to ${min_eq:.2f} on {min_date}")
        
        if (df['Drawdown'] >= 1.0).any():
            issues.append("Drawdown reached 100%")
    
    return len(issues) == 0, issues


if __name__ == "__main__":
    print("Portfolio Engine Test")
    print("=" * 60)
    
    portfolio = PortfolioEngine(100_000)
    prices = {'SPY': 450, 'QQQ': 380, 'GLD': 180, 'TLT': 100}
    date = pd.Timestamp('2024-01-31')
    
    # Test rebalance
    target = {'SPY': 0.25, 'QQQ': 0.25, 'GLD': 0.20, 'TLT': 0.20}
    
    print(f"\nInitial: ${portfolio.get_equity(prices):,.2f}")
    
    trades = portfolio.rebalance_to_weights(target, prices, date)
    portfolio.update_equity_curve(date, prices)
    
    print(f"\nExecuted {len(trades)} trades")
    print(f"Equity: ${portfolio.get_equity(prices):,.2f}")
    print(f"Cash: ${portfolio.cash:,.2f}")
    print(f"Weights: {portfolio.get_position_weights(prices)}")
    
    # Validate
    valid, issues = validate_portfolio_state(portfolio, prices)
    print(f"\nValid: {valid}")
    if issues:
        for i in issues:
            print(f"  - {i}")
