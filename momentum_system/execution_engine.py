"""
Cross-Sectional Momentum System - Execution Engine
Handles live and paper trading execution.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, date
from typing import Dict, List, Optional
from pathlib import Path
import json

from config import CONFIG, EXEC_CONFIG


@dataclass
class OrderRequest:
    """Order request for execution."""
    symbol: str
    side: str  # "BUY" or "SELL"
    quantity: float
    order_type: str = "MARKET"
    limit_price: Optional[float] = None


@dataclass
class OrderFill:
    """Order fill confirmation."""
    symbol: str
    side: str
    quantity: float
    price: float
    commission: float
    timestamp: datetime
    order_id: str


class ExecutionInterface(ABC):
    """Abstract execution interface."""
    
    @abstractmethod
    def connect(self) -> bool:
        pass
    
    @abstractmethod
    def get_positions(self) -> Dict[str, float]:
        pass
    
    @abstractmethod
    def get_cash(self) -> float:
        pass
    
    @abstractmethod
    def get_price(self, symbol: str) -> float:
        pass
    
    @abstractmethod
    def submit_order(self, order: OrderRequest) -> Optional[OrderFill]:
        pass


class PaperExecution(ExecutionInterface):
    """
    Paper trading execution for testing.
    
    Simulates order execution with realistic fills.
    """
    
    def __init__(self, initial_capital: float = None):
        if initial_capital is None:
            initial_capital = CONFIG.INITIAL_CAPITAL
        
        self.cash = initial_capital
        self.positions: Dict[str, float] = {}  # symbol -> shares
        self.avg_prices: Dict[str, float] = {}  # symbol -> avg entry price
        self._prices: Dict[str, float] = {}
        self._fills: List[OrderFill] = []
        self._order_id = 0
        self._connected = False
    
    def connect(self) -> bool:
        self._connected = True
        return True
    
    def update_prices(self, prices: Dict[str, float]):
        """Update current prices (for simulation)."""
        self._prices = prices.copy()
    
    def get_positions(self) -> Dict[str, float]:
        return self.positions.copy()
    
    def get_cash(self) -> float:
        return self.cash
    
    def get_price(self, symbol: str) -> float:
        return self._prices.get(symbol, 0.0)
    
    def get_equity(self) -> float:
        position_value = sum(
            shares * self._prices.get(sym, 0)
            for sym, shares in self.positions.items()
        )
        return self.cash + position_value
    
    def submit_order(self, order: OrderRequest) -> Optional[OrderFill]:
        """Execute order with slippage and commission."""
        if not self._connected:
            return None
        
        price = self._prices.get(order.symbol)
        if not price:
            return None
        
        self._order_id += 1
        
        # Apply slippage
        if order.side == "BUY":
            exec_price = price * (1 + CONFIG.SLIPPAGE_PCT)
        else:
            exec_price = price * (1 - CONFIG.SLIPPAGE_PCT)
        
        # Calculate values
        value = order.quantity * exec_price
        commission = value * CONFIG.COMMISSION_PCT
        
        if order.side == "BUY":
            total_cost = value + commission
            if total_cost > self.cash:
                return None
            
            self.cash -= total_cost
            
            # Update position
            old_shares = self.positions.get(order.symbol, 0)
            old_value = old_shares * self.avg_prices.get(order.symbol, 0)
            
            new_shares = old_shares + order.quantity
            new_value = old_value + value
            
            self.positions[order.symbol] = new_shares
            self.avg_prices[order.symbol] = new_value / new_shares if new_shares > 0 else 0
        
        else:  # SELL
            old_shares = self.positions.get(order.symbol, 0)
            if order.quantity > old_shares:
                return None
            
            self.cash += value - commission
            
            new_shares = old_shares - order.quantity
            if new_shares > 0:
                self.positions[order.symbol] = new_shares
            else:
                del self.positions[order.symbol]
                if order.symbol in self.avg_prices:
                    del self.avg_prices[order.symbol]
        
        fill = OrderFill(
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=exec_price,
            commission=commission,
            timestamp=datetime.now(),
            order_id=f"PAPER-{self._order_id}",
        )
        
        self._fills.append(fill)
        return fill


class ExecutionEngine:
    """
    High-level execution coordinator.
    
    Handles rebalancing logic and order generation.
    """
    
    def __init__(self, execution: ExecutionInterface, dry_run: bool = True):
        """Initialize execution engine."""
        self.execution = execution
        self.dry_run = dry_run
        self._log: List[dict] = []
    
    def connect(self) -> bool:
        """Connect to execution interface."""
        return self.execution.connect()
    
    def get_current_state(self) -> dict:
        """Get current portfolio state."""
        return {
            'cash': self.execution.get_cash(),
            'positions': self.execution.get_positions(),
            'equity': sum(
                shares * self.execution.get_price(sym)
                for sym, shares in self.execution.get_positions().items()
            ) + self.execution.get_cash(),
        }
    
    def calculate_target_positions(
        self,
        target_weights: Dict[str, float],
        equity: float,
        prices: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Calculate target shares for each position.
        
        Args:
            target_weights: Target weight per symbol
            equity: Total equity
            prices: Current prices
            
        Returns:
            Dict of symbol -> target shares
        """
        targets = {}
        
        for symbol, weight in target_weights.items():
            target_value = equity * weight
            price = prices.get(symbol, 0)
            
            if price > 0:
                targets[symbol] = target_value / price
        
        return targets
    
    def generate_orders(
        self,
        target_shares: Dict[str, float],
    ) -> List[OrderRequest]:
        """
        Generate orders to reach target positions.
        
        Args:
            target_shares: Target shares per symbol
            
        Returns:
            List of orders to execute
        """
        current = self.execution.get_positions()
        orders = []
        
        # Sells first
        for symbol, current_shares in current.items():
            target = target_shares.get(symbol, 0)
            
            if current_shares > target:
                sell_qty = current_shares - target
                orders.append(OrderRequest(
                    symbol=symbol,
                    side="SELL",
                    quantity=sell_qty,
                ))
        
        # Then buys
        for symbol, target in target_shares.items():
            current_shares = current.get(symbol, 0)
            
            if target > current_shares:
                buy_qty = target - current_shares
                orders.append(OrderRequest(
                    symbol=symbol,
                    side="BUY",
                    quantity=buy_qty,
                ))
        
        return orders
    
    def execute_rebalance(
        self,
        target_weights: Dict[str, float],
        prices: Dict[str, float],
    ) -> List[OrderFill]:
        """
        Execute a full rebalance to target weights.
        
        Args:
            target_weights: Target weight per symbol
            prices: Current prices
            
        Returns:
            List of fills
        """
        # Update prices in execution interface
        if hasattr(self.execution, 'update_prices'):
            self.execution.update_prices(prices)
        
        # Get current equity
        cash = self.execution.get_cash()
        positions = self.execution.get_positions()
        position_value = sum(
            shares * prices.get(sym, 0)
            for sym, shares in positions.items()
        )
        equity = cash + position_value
        
        # Calculate targets
        target_shares = self.calculate_target_positions(
            target_weights, equity, prices
        )
        
        # Generate orders
        orders = self.generate_orders(target_shares)
        
        if self.dry_run:
            print(f"\n[DRY RUN] Would execute {len(orders)} orders:")
            for order in orders:
                print(f"  {order.side} {order.quantity:.2f} {order.symbol}")
            return []
        
        # Execute orders
        fills = []
        for order in orders:
            fill = self.execution.submit_order(order)
            if fill:
                fills.append(fill)
                self._log_fill(fill)
        
        return fills
    
    def _log_fill(self, fill: OrderFill):
        """Log fill to audit trail."""
        self._log.append({
            'timestamp': fill.timestamp.isoformat(),
            'symbol': fill.symbol,
            'side': fill.side,
            'quantity': fill.quantity,
            'price': fill.price,
            'commission': fill.commission,
            'order_id': fill.order_id,
        })
    
    def save_log(self, filepath: str):
        """Save execution log to file."""
        with open(filepath, 'w') as f:
            json.dump(self._log, f, indent=2)


if __name__ == "__main__":
    print("Execution Engine Test")
    print("=" * 60)
    
    # Create paper execution
    paper = PaperExecution(100_000)
    engine = ExecutionEngine(paper, dry_run=False)
    
    engine.connect()
    
    # Set prices
    prices = {'SPY': 450.0, 'QQQ': 380.0, 'GLD': 180.0}
    paper.update_prices(prices)
    
    # Execute rebalance
    target_weights = {'SPY': 0.333, 'QQQ': 0.333, 'GLD': 0.333}
    
    print(f"Initial state: {engine.get_current_state()}")
    
    fills = engine.execute_rebalance(target_weights, prices)
    
    print(f"\nFills: {len(fills)}")
    for fill in fills:
        print(f"  {fill.side} {fill.quantity:.2f} {fill.symbol} @ ${fill.price:.2f}")
    
    print(f"\nFinal state: {engine.get_current_state()}")
