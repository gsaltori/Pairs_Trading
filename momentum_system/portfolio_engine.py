"""
Cross-Sectional Momentum System - Portfolio Engine (CORRECTED)
Handles position tracking, rebalancing, and trade execution.

FIXES APPLIED:
1. Partial sells now correctly reduce position instead of deleting
2. PnL calculation is proportional to shares sold
3. Full closes vs partial sells handled separately
4. Comprehensive sanity checks throughout

SANITY CHECKS ENFORCED:
- Equity must never be <= 0
- Sum of position weights must be <= 1.0
- No implicit leverage
- Cash debited on entry, credited on exit
- EMA200 NaNs block trading (enforced in momentum_engine)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from config import CONFIG
from momentum_engine import MomentumSignal


class SanityCheckError(Exception):
    """Raised when a sanity check fails."""
    pass


@dataclass
class Position:
    """Single position."""
    symbol: str
    shares: float
    entry_price: float
    entry_date: pd.Timestamp
    entry_value: float  # Total cost basis
    
    def current_value(self, price: float) -> float:
        return self.shares * price
    
    def cost_per_share(self) -> float:
        """Average cost per share."""
        return self.entry_value / self.shares if self.shares > 0 else 0
    
    def pnl(self, price: float) -> float:
        return self.current_value(price) - self.entry_value
    
    def pnl_pct(self, price: float) -> float:
        return self.pnl(price) / self.entry_value if self.entry_value > 0 else 0


@dataclass
class Trade:
    """Completed trade record."""
    symbol: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    shares: float
    pnl: float
    pnl_pct: float
    holding_days: int
    costs: float
    
    @property
    def won(self) -> bool:
        return self.pnl > 0


@dataclass
class RebalanceAction:
    """Actions to take during rebalance."""
    date: pd.Timestamp
    sells: Dict[str, float]      # symbol -> shares to sell
    buys: Dict[str, float]       # symbol -> shares to buy
    new_positions: List[str]     # Symbols entering
    closed_positions: List[str]  # Symbols exiting (FULL close)
    turnover: float              # Fraction of portfolio traded


class PortfolioEngine:
    """
    Portfolio management for momentum system.
    
    CRITICAL: All operations include sanity checks.
    Execution halts if any check fails.
    """
    
    def __init__(self, initial_capital: float = None):
        """Initialize portfolio."""
        if initial_capital is None:
            initial_capital = CONFIG.INITIAL_CAPITAL
        
        if initial_capital <= 0:
            raise SanityCheckError(f"Initial capital must be > 0, got {initial_capital}")
        
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[pd.Timestamp, float]] = []
        self.daily_returns: List[Tuple[pd.Timestamp, float]] = []
        
        self._prev_equity = initial_capital
        self._total_costs = 0.0
        self._rebalance_dates: List[pd.Timestamp] = []
    
    # =========================================================================
    # SANITY CHECKS
    # =========================================================================
    
    def _check_equity_positive(self, prices: Dict[str, float], context: str = ""):
        """SANITY: Equity must be > 0."""
        equity = self.get_equity(prices)
        if equity <= 0:
            raise SanityCheckError(
                f"[{context}] Equity is {equity:.2f} <= 0. "
                f"Cash={self.cash:.2f}, Positions={list(self.positions.keys())}"
            )
    
    def _check_cash_non_negative(self, context: str = ""):
        """SANITY: Cash must be >= 0."""
        if self.cash < -0.01:  # Allow tiny float errors
            raise SanityCheckError(
                f"[{context}] Cash is negative: {self.cash:.2f}"
            )
    
    def _check_no_leverage(self, prices: Dict[str, float], context: str = ""):
        """SANITY: Position value must not exceed equity (no leverage)."""
        equity = self.get_equity(prices)
        position_value = sum(
            pos.current_value(prices.get(sym, pos.entry_price))
            for sym, pos in self.positions.items()
        )
        
        if position_value > equity * 1.001:  # 0.1% tolerance for float errors
            raise SanityCheckError(
                f"[{context}] Implicit leverage detected. "
                f"Position value {position_value:.2f} > equity {equity:.2f}"
            )
    
    def _check_weights_valid(self, weights: Dict[str, float], context: str = ""):
        """SANITY: Sum of weights must be <= 1.0."""
        total_weight = sum(weights.values())
        if total_weight > 1.001:  # 0.1% tolerance
            raise SanityCheckError(
                f"[{context}] Weights sum to {total_weight:.4f} > 1.0. "
                f"Weights: {weights}"
            )
        
        for symbol, weight in weights.items():
            if weight < 0:
                raise SanityCheckError(
                    f"[{context}] Negative weight for {symbol}: {weight}"
                )
    
    def _check_position_integrity(self, context: str = ""):
        """SANITY: All positions must have positive shares and entry_value."""
        for symbol, pos in self.positions.items():
            if pos.shares <= 0:
                raise SanityCheckError(
                    f"[{context}] Position {symbol} has non-positive shares: {pos.shares}"
                )
            if pos.entry_value <= 0:
                raise SanityCheckError(
                    f"[{context}] Position {symbol} has non-positive entry_value: {pos.entry_value}"
                )
    
    def run_all_sanity_checks(self, prices: Dict[str, float], context: str = ""):
        """Run all sanity checks."""
        self._check_equity_positive(prices, context)
        self._check_cash_non_negative(context)
        self._check_no_leverage(prices, context)
        self._check_position_integrity(context)
    
    # =========================================================================
    # CORE METHODS
    # =========================================================================
    
    def get_equity(self, prices: Dict[str, float]) -> float:
        """Calculate current equity."""
        position_value = sum(
            pos.current_value(prices.get(sym, pos.entry_price))
            for sym, pos in self.positions.items()
        )
        return self.cash + position_value
    
    def get_position_weights(self, prices: Dict[str, float]) -> Dict[str, float]:
        """Get current weight of each position."""
        equity = self.get_equity(prices)
        if equity <= 0:
            return {}
        
        return {
            sym: pos.current_value(prices.get(sym, pos.entry_price)) / equity
            for sym, pos in self.positions.items()
        }
    
    def calculate_rebalance_actions(
        self,
        signal: MomentumSignal,
        prices: Dict[str, float],
    ) -> RebalanceAction:
        """
        Calculate what trades need to happen for rebalancing.
        
        SANITY: Validates weights before calculating actions.
        """
        # Validate target weights
        self._check_weights_valid(signal.weights, "calculate_rebalance_actions")
        
        equity = self.get_equity(prices)
        
        if equity <= 0:
            raise SanityCheckError(
                f"Cannot rebalance with equity {equity:.2f} <= 0"
            )
        
        # Current positions
        current_symbols = set(self.positions.keys())
        target_symbols = set(signal.weights.keys())
        
        # Determine changes
        to_close = current_symbols - target_symbols  # Full close
        to_open = target_symbols - current_symbols   # New positions
        to_rebalance = current_symbols & target_symbols  # Adjust existing
        
        sells = {}
        buys = {}
        
        # FULL CLOSE: positions no longer in selection
        for symbol in to_close:
            if symbol in self.positions:
                sells[symbol] = self.positions[symbol].shares
        
        # NEW POSITIONS: open with target weight
        for symbol in to_open:
            target_value = equity * signal.weights[symbol]
            price = prices.get(symbol)
            if price and price > 0:
                shares = target_value / price
                if shares > 0:
                    buys[symbol] = shares
        
        # REBALANCE: adjust existing positions
        for symbol in to_rebalance:
            target_value = equity * signal.weights[symbol]
            price = prices.get(symbol)
            if price and price > 0:
                target_shares = target_value / price
                current_shares = self.positions[symbol].shares
                
                diff = target_shares - current_shares
                
                # Only rebalance if difference is significant (>5%)
                if current_shares > 0 and abs(diff / current_shares) > 0.05:
                    if diff > 0:
                        buys[symbol] = diff
                    elif diff < 0:
                        # Partial sell - NOT a full close
                        sells[symbol] = abs(diff)
        
        # Calculate turnover
        sell_value = sum(
            shares * prices.get(sym, 0)
            for sym, shares in sells.items()
        )
        buy_value = sum(
            shares * prices.get(sym, 0)
            for sym, shares in buys.items()
        )
        turnover = (sell_value + buy_value) / (2 * equity) if equity > 0 else 0
        
        return RebalanceAction(
            date=signal.date,
            sells=sells,
            buys=buys,
            new_positions=list(to_open),
            closed_positions=list(to_close),  # Only FULL closes
            turnover=turnover,
        )
    
    def execute_rebalance(
        self,
        actions: RebalanceAction,
        prices: Dict[str, float],
    ) -> List[Trade]:
        """
        Execute rebalance actions.
        
        Order: Sells first (to free cash), then buys.
        SANITY: Checks run before and after.
        """
        # Pre-rebalance check
        self.run_all_sanity_checks(prices, "pre-rebalance")
        
        completed_trades = []
        
        # Execute sells first (frees up cash for buys)
        for symbol, shares_to_sell in actions.sells.items():
            is_full_close = symbol in actions.closed_positions
            
            trade = self._execute_sell(
                symbol=symbol,
                shares=shares_to_sell,
                prices=prices,
                date=actions.date,
                is_full_close=is_full_close,
            )
            
            if trade:
                completed_trades.append(trade)
        
        # Execute buys
        for symbol, shares_to_buy in actions.buys.items():
            self._execute_buy(symbol, shares_to_buy, prices, actions.date)
        
        # Post-rebalance check
        self.run_all_sanity_checks(prices, "post-rebalance")
        
        # Record rebalance date
        self._rebalance_dates.append(actions.date)
        
        return completed_trades
    
    def _execute_sell(
        self,
        symbol: str,
        shares: float,
        prices: Dict[str, float],
        date: pd.Timestamp,
        is_full_close: bool,
    ) -> Optional[Trade]:
        """
        Execute a sell order.
        
        CRITICAL FIX: Handles partial sells correctly.
        - Full close: Delete position, record trade
        - Partial sell: Reduce shares/entry_value proportionally, record trade
        """
        if symbol not in self.positions:
            return None
        
        pos = self.positions[symbol]
        
        # Validate shares to sell
        if shares <= 0:
            return None
        
        if shares > pos.shares * 1.001:  # Allow tiny float error
            raise SanityCheckError(
                f"Cannot sell {shares} shares of {symbol}, only have {pos.shares}"
            )
        
        # Cap at actual shares
        shares = min(shares, pos.shares)
        
        price = prices.get(symbol, pos.entry_price)
        
        # Apply slippage (sell at lower price)
        exec_price = price * (1 - CONFIG.SLIPPAGE_PCT)
        
        # Calculate proceeds
        gross_proceeds = shares * exec_price
        commission = gross_proceeds * CONFIG.COMMISSION_PCT
        net_proceeds = gross_proceeds - commission
        
        # Calculate PROPORTIONAL cost basis for shares sold
        proportion_sold = shares / pos.shares
        cost_basis_sold = pos.entry_value * proportion_sold
        
        # PnL for this specific trade
        trade_pnl = net_proceeds - cost_basis_sold
        
        # Update cash (CREDIT on sale)
        self.cash += net_proceeds
        self._total_costs += commission
        
        # SANITY: Cash should not go negative from a sale
        if self.cash < -0.01:
            raise SanityCheckError(
                f"Cash went negative after selling {symbol}: {self.cash:.2f}"
            )
        
        # Create trade record
        trade = Trade(
            symbol=symbol,
            entry_date=pos.entry_date,
            exit_date=date,
            entry_price=pos.cost_per_share(),
            exit_price=exec_price,
            shares=shares,
            pnl=trade_pnl,
            pnl_pct=(exec_price / pos.cost_per_share()) - 1 if pos.cost_per_share() > 0 else 0,
            holding_days=(date - pos.entry_date).days,
            costs=commission,
        )
        
        self.trades.append(trade)
        
        # Update or remove position
        if is_full_close or abs(pos.shares - shares) < 0.001:
            # FULL CLOSE: Remove position entirely
            del self.positions[symbol]
        else:
            # PARTIAL SELL: Reduce position proportionally
            remaining_shares = pos.shares - shares
            remaining_value = pos.entry_value * (1 - proportion_sold)
            
            self.positions[symbol] = Position(
                symbol=symbol,
                shares=remaining_shares,
                entry_price=pos.entry_price,  # Keep original entry price for tracking
                entry_date=pos.entry_date,
                entry_value=remaining_value,
            )
        
        return trade
    
    def _execute_buy(
        self,
        symbol: str,
        shares: float,
        prices: Dict[str, float],
        date: pd.Timestamp,
    ):
        """
        Execute a buy order.
        
        SANITY: Cash is debited, cannot go significantly negative.
        """
        if shares <= 0:
            return
        
        price = prices.get(symbol)
        if not price or price <= 0:
            return
        
        # Apply slippage (buy at higher price)
        exec_price = price * (1 + CONFIG.SLIPPAGE_PCT)
        
        # Calculate cost
        gross_cost = shares * exec_price
        commission = gross_cost * CONFIG.COMMISSION_PCT
        total_cost = gross_cost + commission
        
        # Check if we have enough cash
        if total_cost > self.cash + 0.01:  # Allow tiny float error
            # Reduce shares to fit available cash
            available_for_shares = self.cash - commission
            if available_for_shares <= 0:
                return  # Can't buy anything
            
            shares = available_for_shares / exec_price
            gross_cost = shares * exec_price
            commission = gross_cost * CONFIG.COMMISSION_PCT
            total_cost = gross_cost + commission
        
        if shares <= 0:
            return
        
        # DEBIT cash
        self.cash -= total_cost
        self._total_costs += commission
        
        # SANITY: Cash should not go significantly negative
        if self.cash < -1.0:  # Allow $1 float error
            raise SanityCheckError(
                f"Cash went significantly negative after buying {symbol}: {self.cash:.2f}"
            )
        
        # Create or update position
        if symbol in self.positions:
            # Add to existing position (average in)
            old = self.positions[symbol]
            new_shares = old.shares + shares
            new_value = old.entry_value + gross_cost
            new_avg_price = new_value / new_shares
            
            self.positions[symbol] = Position(
                symbol=symbol,
                shares=new_shares,
                entry_price=new_avg_price,
                entry_date=old.entry_date,  # Keep original entry date
                entry_value=new_value,
            )
        else:
            # New position
            self.positions[symbol] = Position(
                symbol=symbol,
                shares=shares,
                entry_price=exec_price,
                entry_date=date,
                entry_value=gross_cost,
            )
    
    def update_equity_curve(self, date: pd.Timestamp, prices: Dict[str, float]):
        """
        Record equity for this date.
        
        SANITY: Equity must remain positive.
        """
        equity = self.get_equity(prices)
        
        # SANITY CHECK
        if equity <= 0:
            raise SanityCheckError(
                f"Equity dropped to {equity:.2f} on {date}. "
                f"Cash={self.cash:.2f}, Positions={self.positions}"
            )
        
        self.equity_curve.append((date, equity))
        
        # Calculate daily return
        if self._prev_equity > 0:
            daily_return = (equity / self._prev_equity) - 1
        else:
            daily_return = 0
        
        self.daily_returns.append((date, daily_return))
        self._prev_equity = equity
    
    def get_equity_df(self) -> pd.DataFrame:
        """Get equity curve as DataFrame."""
        if not self.equity_curve:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.equity_curve, columns=['Date', 'Equity'])
        df = df.set_index('Date')
        
        df['Peak'] = df['Equity'].cummax()
        df['Drawdown'] = (df['Peak'] - df['Equity']) / df['Peak']
        
        return df
    
    def get_trades_df(self) -> pd.DataFrame:
        """Get trades as DataFrame."""
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
                'PnL': t.pnl,
                'PnL_Pct': t.pnl_pct,
                'Holding_Days': t.holding_days,
                'Costs': t.costs,
                'Won': t.won,
            })
        
        return pd.DataFrame(records)
    
    def get_monthly_returns(self) -> pd.Series:
        """Calculate monthly returns."""
        df = self.get_equity_df()
        if len(df) == 0:
            return pd.Series()
        
        monthly = df['Equity'].resample('ME').last()
        returns = monthly.pct_change().dropna()
        return returns
    
    def validate_final_state(self, prices: Dict[str, float]) -> Tuple[bool, List[str]]:
        """
        Validate final portfolio state.
        
        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []
        
        # Check equity
        equity = self.get_equity(prices)
        if equity <= 0:
            issues.append(f"Final equity is {equity:.2f} <= 0")
        
        # Check cash
        if self.cash < -0.01:
            issues.append(f"Final cash is negative: {self.cash:.2f}")
        
        # Check positions
        for symbol, pos in self.positions.items():
            if pos.shares <= 0:
                issues.append(f"Position {symbol} has non-positive shares: {pos.shares}")
            if pos.entry_value <= 0:
                issues.append(f"Position {symbol} has non-positive entry_value: {pos.entry_value}")
        
        # Check no leverage
        position_value = sum(
            pos.current_value(prices.get(sym, pos.entry_price))
            for sym, pos in self.positions.items()
        )
        if position_value > equity * 1.01:
            issues.append(f"Implicit leverage: position value {position_value:.2f} > equity {equity:.2f}")
        
        # Check equity curve is always positive
        df = self.get_equity_df()
        if len(df) > 0:
            min_equity = df['Equity'].min()
            if min_equity <= 0:
                issues.append(f"Equity curve went to {min_equity:.2f} at some point")
        
        return len(issues) == 0, issues
    
    def reset(self):
        """Reset portfolio to initial state."""
        self.cash = self.initial_capital
        self.positions.clear()
        self.trades.clear()
        self.equity_curve.clear()
        self.daily_returns.clear()
        self._prev_equity = self.initial_capital
        self._total_costs = 0.0
        self._rebalance_dates.clear()


# =============================================================================
# STANDALONE VALIDATION
# =============================================================================

def validate_backtest_results(
    portfolio: PortfolioEngine,
    prices: Dict[str, float],
) -> Tuple[bool, List[str]]:
    """
    Comprehensive validation of backtest results.
    
    Returns:
        Tuple of (is_valid, list of issues)
    """
    issues = []
    
    # 1. Final state validation
    state_valid, state_issues = portfolio.validate_final_state(prices)
    issues.extend(state_issues)
    
    # 2. Equity curve validation
    df = portfolio.get_equity_df()
    if len(df) > 0:
        # Never negative
        if (df['Equity'] <= 0).any():
            min_eq = df['Equity'].min()
            min_date = df['Equity'].idxmin()
            issues.append(f"Equity went to {min_eq:.2f} on {min_date}")
        
        # Reasonable returns (sanity check for extreme values)
        total_return = (df['Equity'].iloc[-1] / portfolio.initial_capital) - 1
        years = (df.index[-1] - df.index[0]).days / 365.25
        
        if years > 0:
            cagr = (1 + total_return) ** (1/years) - 1
            if cagr > 1.0:  # >100% CAGR is suspicious
                issues.append(f"Suspiciously high CAGR: {cagr:.1%}")
            if cagr < -0.99:  # Lost 99%+ is suspicious for long-only ETFs
                issues.append(f"Suspiciously low CAGR: {cagr:.1%}")
    
    # 3. Trade validation
    trades_df = portfolio.get_trades_df()
    if len(trades_df) > 0:
        # No infinite or NaN values
        if trades_df['PnL'].isna().any():
            issues.append("Some trades have NaN PnL")
        if np.isinf(trades_df['PnL']).any():
            issues.append("Some trades have infinite PnL")
        
        # Shares must be positive
        if (trades_df['Shares'] <= 0).any():
            issues.append("Some trades have non-positive shares")
    
    # 4. Cash accounting validation
    # Final cash + position value should equal final equity
    final_equity = portfolio.get_equity(prices)
    position_value = sum(
        pos.current_value(prices.get(sym, pos.entry_price))
        for sym, pos in portfolio.positions.items()
    )
    computed_equity = portfolio.cash + position_value
    
    if abs(final_equity - computed_equity) > 1.0:
        issues.append(
            f"Equity mismatch: get_equity={final_equity:.2f}, "
            f"cash+positions={computed_equity:.2f}"
        )
    
    return len(issues) == 0, issues


if __name__ == "__main__":
    print("Portfolio Engine Test (CORRECTED VERSION)")
    print("=" * 60)
    
    portfolio = PortfolioEngine(100_000)
    
    prices = {'SPY': 450.0, 'QQQ': 380.0}
    date = pd.Timestamp('2024-01-15')
    
    # Test buy
    print("\n1. Testing buy...")
    portfolio._execute_buy('SPY', 100, prices, date)
    portfolio._execute_buy('QQQ', 100, prices, date)
    
    print(f"  Cash: ${portfolio.cash:,.2f}")
    print(f"  Positions: {list(portfolio.positions.keys())}")
    print(f"  Equity: ${portfolio.get_equity(prices):,.2f}")
    
    # Test partial sell
    print("\n2. Testing PARTIAL sell (50 shares of SPY)...")
    trade = portfolio._execute_sell('SPY', 50, prices, date, is_full_close=False)
    
    print(f"  Trade PnL: ${trade.pnl:.2f}")
    print(f"  Remaining SPY shares: {portfolio.positions.get('SPY', Position('', 0, 0, date, 0)).shares}")
    print(f"  Cash: ${portfolio.cash:,.2f}")
    print(f"  Equity: ${portfolio.get_equity(prices):,.2f}")
    
    # Test full close
    print("\n3. Testing FULL close (remaining SPY)...")
    trade = portfolio._execute_sell('SPY', 50, prices, date, is_full_close=True)
    
    print(f"  Trade PnL: ${trade.pnl:.2f}")
    print(f"  SPY in positions: {'SPY' in portfolio.positions}")
    print(f"  Cash: ${portfolio.cash:,.2f}")
    print(f"  Equity: ${portfolio.get_equity(prices):,.2f}")
    
    # Run sanity checks
    print("\n4. Running sanity checks...")
    portfolio.run_all_sanity_checks(prices, "test")
    print("  All sanity checks passed!")
    
    # Validate
    print("\n5. Validating results...")
    valid, issues = validate_backtest_results(portfolio, prices)
    print(f"  Valid: {valid}")
    if issues:
        for issue in issues:
            print(f"  Issue: {issue}")
