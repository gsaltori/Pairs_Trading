"""
Trend Following System - Execution Engine
Broker-agnostic order execution abstraction.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, List, Callable
import json
from pathlib import Path

from config import TRADING_CONFIG, EXECUTION_CONFIG


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class Order:
    """Order representation."""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    
    status: OrderStatus = OrderStatus.PENDING
    filled_price: Optional[float] = None
    filled_quantity: int = 0
    filled_time: Optional[datetime] = None
    
    created_time: datetime = None
    
    def __post_init__(self):
        if self.created_time is None:
            self.created_time = datetime.now()
    
    def to_dict(self) -> dict:
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'order_type': self.order_type.value,
            'limit_price': self.limit_price,
            'stop_price': self.stop_price,
            'status': self.status.value,
            'filled_price': self.filled_price,
            'filled_quantity': self.filled_quantity,
            'filled_time': self.filled_time.isoformat() if self.filled_time else None,
            'created_time': self.created_time.isoformat() if self.created_time else None,
        }


@dataclass
class Fill:
    """Order fill details."""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    price: float
    commission: float
    fill_time: datetime


class BrokerInterface(ABC):
    """Abstract broker interface."""
    
    @abstractmethod
    def connect(self) -> bool:
        """Connect to broker."""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from broker."""
        pass
    
    @abstractmethod
    def submit_order(self, order: Order) -> bool:
        """Submit order to broker."""
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order."""
        pass
    
    @abstractmethod
    def get_positions(self) -> Dict[str, int]:
        """Get current positions."""
        pass
    
    @abstractmethod
    def get_account_value(self) -> float:
        """Get account equity."""
        pass
    
    @abstractmethod
    def get_buying_power(self) -> float:
        """Get available buying power."""
        pass


class PaperBroker(BrokerInterface):
    """
    Paper trading broker for simulation.
    
    Simulates order execution with realistic fills.
    """
    
    def __init__(
        self,
        initial_capital: float = None,
        data_source: Callable[[str], float] = None,
    ):
        """Initialize paper broker."""
        if initial_capital is None:
            initial_capital = TRADING_CONFIG.INITIAL_CAPITAL
        
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, int] = {}
        self.orders: Dict[str, Order] = {}
        self.fills: List[Fill] = []
        
        # Price source for simulated fills
        self._get_price = data_source
        
        self._connected = False
        self._order_counter = 0
    
    def connect(self) -> bool:
        """Connect (always succeeds for paper)."""
        self._connected = True
        return True
    
    def disconnect(self) -> None:
        """Disconnect."""
        self._connected = False
    
    def submit_order(self, order: Order) -> bool:
        """
        Submit and immediately fill market orders.
        
        For paper trading, we simulate immediate fills
        with slippage applied.
        """
        if not self._connected:
            return False
        
        # Generate order ID if needed
        if not order.order_id:
            self._order_counter += 1
            order.order_id = f"PAPER-{self._order_counter}"
        
        order.status = OrderStatus.SUBMITTED
        self.orders[order.order_id] = order
        
        # For market orders, fill immediately
        if order.order_type == OrderType.MARKET:
            return self._fill_order(order)
        
        return True
    
    def _fill_order(self, order: Order) -> bool:
        """Simulate order fill with costs."""
        # Get current price
        if self._get_price:
            price = self._get_price(order.symbol)
        else:
            # Default: use limit price or reject
            if order.limit_price:
                price = order.limit_price
            else:
                order.status = OrderStatus.REJECTED
                return False
        
        # Apply slippage
        slippage = price * TRADING_CONFIG.SLIPPAGE_PCT
        if order.side == OrderSide.BUY:
            fill_price = price + slippage
        else:
            fill_price = price - slippage
        
        # Calculate commission
        commission = order.quantity * TRADING_CONFIG.COMMISSION_PER_SHARE
        
        # Check buying power for buys
        if order.side == OrderSide.BUY:
            total_cost = order.quantity * fill_price + commission
            if total_cost > self.cash:
                order.status = OrderStatus.REJECTED
                return False
            
            self.cash -= total_cost
            self.positions[order.symbol] = self.positions.get(order.symbol, 0) + order.quantity
        
        else:  # SELL
            current_qty = self.positions.get(order.symbol, 0)
            if order.quantity > current_qty:
                order.status = OrderStatus.REJECTED
                return False
            
            proceeds = order.quantity * fill_price - commission
            self.cash += proceeds
            self.positions[order.symbol] = current_qty - order.quantity
            
            if self.positions[order.symbol] == 0:
                del self.positions[order.symbol]
        
        # Record fill
        order.status = OrderStatus.FILLED
        order.filled_price = fill_price
        order.filled_quantity = order.quantity
        order.filled_time = datetime.now()
        
        fill = Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            commission=commission,
            fill_time=datetime.now(),
        )
        self.fills.append(fill)
        
        return True
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order."""
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
            return False
        
        order.status = OrderStatus.CANCELLED
        return True
    
    def get_positions(self) -> Dict[str, int]:
        """Get current positions."""
        return self.positions.copy()
    
    def get_account_value(self) -> float:
        """Get account equity (cash + position value)."""
        if not self._get_price:
            return self.cash
        
        position_value = sum(
            qty * self._get_price(symbol)
            for symbol, qty in self.positions.items()
        )
        
        return self.cash + position_value
    
    def get_buying_power(self) -> float:
        """Get available cash."""
        return self.cash


class ExecutionEngine:
    """
    High-level execution interface.
    
    Manages order flow, position tracking, and broker communication.
    Designed to be broker-agnostic.
    """
    
    def __init__(self, broker: BrokerInterface):
        """Initialize with broker."""
        self.broker = broker
        self._pending_orders: Dict[str, Order] = {}
    
    def connect(self) -> bool:
        """Connect to broker."""
        return self.broker.connect()
    
    def disconnect(self) -> None:
        """Disconnect from broker."""
        self.broker.disconnect()
    
    def buy(
        self,
        symbol: str,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
    ) -> Optional[Order]:
        """
        Place buy order.
        
        Args:
            symbol: Symbol to buy
            quantity: Number of shares
            order_type: MARKET or LIMIT
            limit_price: Price for limit orders
            
        Returns:
            Order object if submitted, None otherwise
        """
        order = Order(
            order_id="",  # Will be assigned by broker
            symbol=symbol,
            side=OrderSide.BUY,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
        )
        
        if self.broker.submit_order(order):
            return order
        return None
    
    def sell(
        self,
        symbol: str,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
    ) -> Optional[Order]:
        """
        Place sell order.
        
        Args:
            symbol: Symbol to sell
            quantity: Number of shares
            order_type: MARKET or LIMIT
            limit_price: Price for limit orders
            
        Returns:
            Order object if submitted, None otherwise
        """
        order = Order(
            order_id="",
            symbol=symbol,
            side=OrderSide.SELL,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
        )
        
        if self.broker.submit_order(order):
            return order
        return None
    
    def close_position(self, symbol: str) -> Optional[Order]:
        """Close entire position in symbol."""
        positions = self.broker.get_positions()
        
        if symbol not in positions or positions[symbol] == 0:
            return None
        
        return self.sell(symbol, positions[symbol])
    
    def close_all_positions(self) -> List[Order]:
        """Close all open positions."""
        orders = []
        positions = self.broker.get_positions()
        
        for symbol, qty in positions.items():
            if qty > 0:
                order = self.sell(symbol, qty)
                if order:
                    orders.append(order)
        
        return orders
    
    def get_positions(self) -> Dict[str, int]:
        """Get current positions."""
        return self.broker.get_positions()
    
    def get_equity(self) -> float:
        """Get account equity."""
        return self.broker.get_account_value()
    
    def get_cash(self) -> float:
        """Get available cash."""
        return self.broker.get_buying_power()
    
    def has_position(self, symbol: str) -> bool:
        """Check if symbol has open position."""
        positions = self.broker.get_positions()
        return symbol in positions and positions[symbol] > 0
    
    def position_count(self) -> int:
        """Get number of open positions."""
        return len([p for p in self.broker.get_positions().values() if p > 0])


class OrderLogger:
    """Logs all orders and fills for audit trail."""
    
    def __init__(self, log_dir: str = None):
        """Initialize logger."""
        if log_dir is None:
            log_dir = EXECUTION_CONFIG.LOGS_DIR
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self._orders: List[dict] = []
        self._fills: List[dict] = []
    
    def log_order(self, order: Order) -> None:
        """Log order submission."""
        self._orders.append(order.to_dict())
        self._save_orders()
    
    def log_fill(self, fill: Fill) -> None:
        """Log order fill."""
        self._fills.append({
            'order_id': fill.order_id,
            'symbol': fill.symbol,
            'side': fill.side.value,
            'quantity': fill.quantity,
            'price': fill.price,
            'commission': fill.commission,
            'fill_time': fill.fill_time.isoformat(),
        })
        self._save_fills()
    
    def _save_orders(self) -> None:
        """Save orders to file."""
        with open(self.log_dir / 'orders.json', 'w') as f:
            json.dump(self._orders, f, indent=2)
    
    def _save_fills(self) -> None:
        """Save fills to file."""
        with open(self.log_dir / 'fills.json', 'w') as f:
            json.dump(self._fills, f, indent=2)


if __name__ == "__main__":
    # Test paper broker
    print("Paper Broker Test")
    print("=" * 60)
    
    # Create mock price source
    prices = {'SPY': 450.0, 'QQQ': 380.0}
    
    def get_price(symbol):
        return prices.get(symbol, 100.0)
    
    broker = PaperBroker(initial_capital=100_000, data_source=get_price)
    engine = ExecutionEngine(broker)
    
    print(f"Initial equity: ${engine.get_equity():,.2f}")
    print(f"Initial cash: ${engine.get_cash():,.2f}")
    
    # Connect
    engine.connect()
    
    # Buy SPY
    print("\nBuying 50 shares of SPY...")
    order = engine.buy('SPY', 50)
    print(f"Order status: {order.status.value}")
    print(f"Filled at: ${order.filled_price:.2f}")
    
    print(f"\nEquity: ${engine.get_equity():,.2f}")
    print(f"Cash: ${engine.get_cash():,.2f}")
    print(f"Positions: {engine.get_positions()}")
    
    # Sell SPY
    print("\nSelling SPY position...")
    order = engine.close_position('SPY')
    print(f"Order status: {order.status.value}")
    
    print(f"\nFinal equity: ${engine.get_equity():,.2f}")
    print(f"Final cash: ${engine.get_cash():,.2f}")
    print(f"Positions: {engine.get_positions()}")
