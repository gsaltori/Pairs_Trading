"""
Trend Following System - Position Sizer
Risk-based position sizing with hard limits.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple

from config import TRADING_CONFIG


@dataclass
class PositionSize:
    """Position sizing result."""
    shares: int
    dollar_risk: float
    position_value: float
    risk_per_share: float
    stop_distance_pct: float
    
    def __repr__(self):
        return (
            f"PositionSize(shares={self.shares}, "
            f"risk=${self.dollar_risk:.2f}, "
            f"value=${self.position_value:.2f})"
        )


class PositionSizer:
    """
    Risk-based position sizing.
    
    Formula:
        Position Size = (Equity × Risk%) / Stop Distance
        
    Where:
        - Risk% = 0.5% of equity (fixed)
        - Stop Distance = Entry Price - Trailing Stop
    
    Constraints:
        - Minimum 1 share
        - Maximum based on available capital
        - Position value cannot exceed equity
    """
    
    def __init__(self):
        """Initialize position sizer with config parameters."""
        self.risk_per_trade = TRADING_CONFIG.RISK_PER_TRADE
    
    def calculate(
        self,
        equity: float,
        entry_price: float,
        stop_price: float,
        available_capital: float = None,
    ) -> PositionSize:
        """
        Calculate position size based on risk.
        
        Args:
            equity: Current portfolio equity
            entry_price: Expected entry price
            stop_price: Initial stop loss price
            available_capital: Capital available for new positions
            
        Returns:
            PositionSize with shares and risk details
        """
        if available_capital is None:
            available_capital = equity
        
        # Calculate risk per share (stop distance)
        stop_distance = entry_price - stop_price
        
        # Validate stop distance
        if stop_distance <= 0:
            raise ValueError(
                f"Invalid stop distance: entry={entry_price}, stop={stop_price}"
            )
        
        stop_distance_pct = stop_distance / entry_price
        
        # Calculate dollar risk (fixed % of equity)
        dollar_risk = equity * self.risk_per_trade
        
        # Calculate raw shares
        raw_shares = dollar_risk / stop_distance
        
        # Round down to whole shares
        shares = int(np.floor(raw_shares))
        
        # Ensure minimum 1 share
        shares = max(1, shares)
        
        # Check against available capital
        position_value = shares * entry_price
        if position_value > available_capital:
            shares = int(np.floor(available_capital / entry_price))
            position_value = shares * entry_price
        
        # Recalculate actual risk
        actual_risk = shares * stop_distance
        
        return PositionSize(
            shares=shares,
            dollar_risk=actual_risk,
            position_value=position_value,
            risk_per_share=stop_distance,
            stop_distance_pct=stop_distance_pct,
        )
    
    def calculate_with_costs(
        self,
        equity: float,
        entry_price: float,
        stop_price: float,
        available_capital: float = None,
    ) -> Tuple[PositionSize, float, float]:
        """
        Calculate position size accounting for transaction costs.
        
        Args:
            equity: Current portfolio equity
            entry_price: Expected entry price
            stop_price: Initial stop loss price
            available_capital: Capital available for new positions
            
        Returns:
            Tuple of (PositionSize, entry_cost, estimated_exit_cost)
        """
        size = self.calculate(equity, entry_price, stop_price, available_capital)
        
        # Calculate costs
        commission = TRADING_CONFIG.COMMISSION_PER_SHARE
        slippage = TRADING_CONFIG.SLIPPAGE_PCT
        
        # Entry costs
        entry_slippage = size.position_value * slippage
        entry_commission = size.shares * commission
        entry_cost = entry_slippage + entry_commission
        
        # Estimated exit costs (same structure)
        exit_cost = entry_cost  # Symmetric assumption
        
        return size, entry_cost, exit_cost


def calculate_shares(
    equity: float,
    entry_price: float,
    stop_price: float,
) -> int:
    """
    Convenience function to get just the share count.
    
    Args:
        equity: Current portfolio equity
        entry_price: Expected entry price
        stop_price: Initial stop loss price
        
    Returns:
        Number of shares to trade
    """
    sizer = PositionSizer()
    size = sizer.calculate(equity, entry_price, stop_price)
    return size.shares


if __name__ == "__main__":
    # Test position sizing
    sizer = PositionSizer()
    
    print("Position Sizing Examples")
    print("=" * 60)
    
    # Example 1: SPY
    print("\nExample 1: SPY")
    print("-" * 40)
    equity = 100_000
    entry_price = 450.0
    stop_price = 430.0  # 20 point stop
    
    size = sizer.calculate(equity, entry_price, stop_price)
    print(f"Equity: ${equity:,.2f}")
    print(f"Entry: ${entry_price:.2f}")
    print(f"Stop: ${stop_price:.2f}")
    print(f"Stop Distance: ${entry_price - stop_price:.2f} ({size.stop_distance_pct:.1%})")
    print(f"Risk Target: ${equity * TRADING_CONFIG.RISK_PER_TRADE:.2f}")
    print(f"Result: {size}")
    
    # Example 2: GLD (smaller price)
    print("\nExample 2: GLD")
    print("-" * 40)
    entry_price = 180.0
    stop_price = 172.0  # 8 point stop
    
    size = sizer.calculate(equity, entry_price, stop_price)
    print(f"Equity: ${equity:,.2f}")
    print(f"Entry: ${entry_price:.2f}")
    print(f"Stop: ${stop_price:.2f}")
    print(f"Stop Distance: ${entry_price - stop_price:.2f} ({size.stop_distance_pct:.1%})")
    print(f"Result: {size}")
    
    # Example 3: With costs
    print("\nExample 3: With Transaction Costs")
    print("-" * 40)
    entry_price = 450.0
    stop_price = 430.0
    
    size, entry_cost, exit_cost = sizer.calculate_with_costs(
        equity, entry_price, stop_price
    )
    print(f"Position: {size}")
    print(f"Entry Cost: ${entry_cost:.2f}")
    print(f"Exit Cost: ${exit_cost:.2f}")
    print(f"Total Cost: ${entry_cost + exit_cost:.2f}")
    print(f"Cost as % of Risk: {(entry_cost + exit_cost) / size.dollar_risk:.1%}")
