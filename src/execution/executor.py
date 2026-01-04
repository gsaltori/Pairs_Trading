"""
Live Execution Module for the Pairs Trading System.

Provides:
- Real-time order execution via OANDA
- Position management
- Risk checks before execution
- Execution logging and monitoring
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from enum import Enum
import logging
import time
import threading
from queue import Queue

from ..data.broker_client import OandaClient
from ..strategy.pairs_strategy import PairsStrategy
from ..strategy.signals import Signal, SignalType
from ..risk.risk_manager import RiskManager, PositionSize
from ..analysis.spread_builder import SpreadBuilder
from config.settings import Settings, TradingMode
from config.broker_config import BrokerConfig, OANDA_INSTRUMENTS


logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderSide(Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


@dataclass
class OrderResult:
    """Result of an order execution."""
    order_id: str
    instrument: str
    side: OrderSide
    units: float
    requested_price: float
    fill_price: float
    status: OrderStatus
    timestamp: datetime
    commission: float = 0.0
    slippage: float = 0.0
    error_message: Optional[str] = None
    
    @property
    def is_success(self) -> bool:
        """Check if order was successful."""
        return self.status == OrderStatus.FILLED


@dataclass
class PairPosition:
    """Active position in a pair."""
    pair: Tuple[str, str]
    direction: str  # 'long_spread' or 'short_spread'
    entry_time: datetime
    entry_zscore: float
    
    # Leg A
    units_a: float
    entry_price_a: float
    order_id_a: str
    
    # Leg B
    units_b: float
    entry_price_b: float
    order_id_b: str
    
    # Hedge ratio at entry
    hedge_ratio: float
    
    # Current state
    unrealized_pnl: float = 0.0
    current_zscore: float = 0.0


@dataclass
class ExecutionState:
    """Current state of the executor."""
    mode: TradingMode
    is_running: bool
    open_positions: Dict[Tuple[str, str], PairPosition]
    pending_orders: List[str]
    daily_trades: int
    daily_pnl: float
    last_update: datetime
    errors: List[str] = field(default_factory=list)


class LiveExecutor:
    """
    Live trading executor for pairs trading.
    
    Handles:
    - Signal reception and validation
    - Order execution via OANDA API
    - Position tracking
    - Risk checks before execution
    - Execution reporting
    """
    
    def __init__(
        self,
        settings: Settings,
        broker_config: BrokerConfig,
        risk_manager: RiskManager
    ):
        """
        Initialize the live executor.
        
        Args:
            settings: Trading system settings
            broker_config: OANDA broker configuration
            risk_manager: Risk manager instance
        """
        self.settings = settings
        self.broker_config = broker_config
        self.risk_manager = risk_manager
        
        # Initialize broker client
        self.client = OandaClient(broker_config)
        
        # State
        self.positions: Dict[Tuple[str, str], PairPosition] = {}
        self.pending_orders: List[str] = []
        self.trade_history: List[Dict] = []
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.last_daily_reset = datetime.now().date()
        
        # Threading
        self.is_running = False
        self.order_queue: Queue = Queue()
        self._lock = threading.Lock()
        
        logger.info(f"LiveExecutor initialized in {settings.mode.value} mode")
    
    def start(self) -> None:
        """Start the executor."""
        self.is_running = True
        logger.info("LiveExecutor started")
    
    def stop(self) -> None:
        """Stop the executor."""
        self.is_running = False
        logger.info("LiveExecutor stopped")
    
    def execute_signal(self, signal: Signal) -> Tuple[bool, str]:
        """
        Execute a trading signal.
        
        Args:
            signal: Signal to execute
            
        Returns:
            Tuple of (success, message)
        """
        if not self.is_running:
            return False, "Executor not running"
        
        # Reset daily counters if new day
        self._check_daily_reset()
        
        pair = signal.pair
        
        try:
            if signal.type == SignalType.LONG_SPREAD:
                return self._open_long_spread(signal)
            elif signal.type == SignalType.SHORT_SPREAD:
                return self._open_short_spread(signal)
            elif signal.type in [SignalType.EXIT, SignalType.STOP_LOSS]:
                return self._close_position(signal)
            else:
                return False, f"Unknown signal type: {signal.type}"
                
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
            return False, str(e)
    
    def _open_long_spread(self, signal: Signal) -> Tuple[bool, str]:
        """Open a long spread position (Long A, Short B)."""
        pair = signal.pair
        
        # Check if already in position
        if pair in self.positions:
            return False, f"Already in position for {pair}"
        
        # Get current prices
        prices = self._get_current_prices(pair)
        if prices is None:
            return False, "Could not get current prices"
        
        price_a, price_b = prices
        
        # Calculate position size
        position_size = self.risk_manager.calculate_position_size(
            pair=pair,
            price_a=price_a,
            price_b=price_b,
            hedge_ratio=signal.hedge_ratio,
            method='risk_based'
        )
        
        # Risk checks
        allowed, reason = self.risk_manager.check_trade_allowed(
            position_size.notional_value
        )
        if not allowed:
            return False, f"Risk check failed: {reason}"
        
        # Execute leg A (BUY)
        result_a = self._execute_order(
            instrument=pair[0],
            units=position_size.units_a,
            side=OrderSide.BUY
        )
        
        if not result_a.is_success:
            return False, f"Failed to execute leg A: {result_a.error_message}"
        
        # Execute leg B (SELL)
        result_b = self._execute_order(
            instrument=pair[1],
            units=position_size.units_b,
            side=OrderSide.SELL
        )
        
        if not result_b.is_success:
            # Rollback leg A
            logger.warning("Leg B failed, rolling back leg A")
            self._execute_order(
                instrument=pair[0],
                units=position_size.units_a,
                side=OrderSide.SELL
            )
            return False, f"Failed to execute leg B: {result_b.error_message}"
        
        # Record position
        position = PairPosition(
            pair=pair,
            direction='long_spread',
            entry_time=datetime.now(),
            entry_zscore=signal.zscore,
            units_a=position_size.units_a,
            entry_price_a=result_a.fill_price,
            order_id_a=result_a.order_id,
            units_b=position_size.units_b,
            entry_price_b=result_b.fill_price,
            order_id_b=result_b.order_id,
            hedge_ratio=signal.hedge_ratio
        )
        
        with self._lock:
            self.positions[pair] = position
            self.daily_trades += 1
        
        # Update risk manager
        self.risk_manager.register_open_position(
            pair=pair,
            notional=position_size.notional_value
        )
        
        logger.info(f"Opened LONG SPREAD on {pair[0]}/{pair[1]} "
                   f"at z={signal.zscore:.2f}")
        
        return True, "Long spread opened successfully"
    
    def _open_short_spread(self, signal: Signal) -> Tuple[bool, str]:
        """Open a short spread position (Short A, Long B)."""
        pair = signal.pair
        
        # Check if already in position
        if pair in self.positions:
            return False, f"Already in position for {pair}"
        
        # Get current prices
        prices = self._get_current_prices(pair)
        if prices is None:
            return False, "Could not get current prices"
        
        price_a, price_b = prices
        
        # Calculate position size
        position_size = self.risk_manager.calculate_position_size(
            pair=pair,
            price_a=price_a,
            price_b=price_b,
            hedge_ratio=signal.hedge_ratio,
            method='risk_based'
        )
        
        # Risk checks
        allowed, reason = self.risk_manager.check_trade_allowed(
            position_size.notional_value
        )
        if not allowed:
            return False, f"Risk check failed: {reason}"
        
        # Execute leg A (SELL)
        result_a = self._execute_order(
            instrument=pair[0],
            units=position_size.units_a,
            side=OrderSide.SELL
        )
        
        if not result_a.is_success:
            return False, f"Failed to execute leg A: {result_a.error_message}"
        
        # Execute leg B (BUY)
        result_b = self._execute_order(
            instrument=pair[1],
            units=position_size.units_b,
            side=OrderSide.BUY
        )
        
        if not result_b.is_success:
            # Rollback leg A
            logger.warning("Leg B failed, rolling back leg A")
            self._execute_order(
                instrument=pair[0],
                units=position_size.units_a,
                side=OrderSide.BUY
            )
            return False, f"Failed to execute leg B: {result_b.error_message}"
        
        # Record position
        position = PairPosition(
            pair=pair,
            direction='short_spread',
            entry_time=datetime.now(),
            entry_zscore=signal.zscore,
            units_a=position_size.units_a,
            entry_price_a=result_a.fill_price,
            order_id_a=result_a.order_id,
            units_b=position_size.units_b,
            entry_price_b=result_b.fill_price,
            order_id_b=result_b.order_id,
            hedge_ratio=signal.hedge_ratio
        )
        
        with self._lock:
            self.positions[pair] = position
            self.daily_trades += 1
        
        # Update risk manager
        self.risk_manager.register_open_position(
            pair=pair,
            notional=position_size.notional_value
        )
        
        logger.info(f"Opened SHORT SPREAD on {pair[0]}/{pair[1]} "
                   f"at z={signal.zscore:.2f}")
        
        return True, "Short spread opened successfully"
    
    def _close_position(self, signal: Signal) -> Tuple[bool, str]:
        """Close an existing position."""
        pair = signal.pair
        
        if pair not in self.positions:
            return False, f"No position to close for {pair}"
        
        position = self.positions[pair]
        
        # Determine close sides based on position direction
        if position.direction == 'long_spread':
            # Was: Long A, Short B
            # Close: Sell A, Buy B
            side_a = OrderSide.SELL
            side_b = OrderSide.BUY
        else:
            # Was: Short A, Long B
            # Close: Buy A, Sell B
            side_a = OrderSide.BUY
            side_b = OrderSide.SELL
        
        # Execute close leg A
        result_a = self._execute_order(
            instrument=pair[0],
            units=position.units_a,
            side=side_a
        )
        
        if not result_a.is_success:
            return False, f"Failed to close leg A: {result_a.error_message}"
        
        # Execute close leg B
        result_b = self._execute_order(
            instrument=pair[1],
            units=position.units_b,
            side=side_b
        )
        
        if not result_b.is_success:
            # Try to re-open leg A (emergency)
            logger.error("Failed to close leg B, attempting recovery")
            opposite_side = OrderSide.BUY if side_a == OrderSide.SELL else OrderSide.SELL
            self._execute_order(
                instrument=pair[0],
                units=position.units_a,
                side=opposite_side
            )
            return False, f"Failed to close leg B: {result_b.error_message}"
        
        # Calculate P/L
        if position.direction == 'long_spread':
            pnl_a = position.units_a * (result_a.fill_price - position.entry_price_a)
            pnl_b = position.units_b * (position.entry_price_b - result_b.fill_price)
        else:
            pnl_a = position.units_a * (position.entry_price_a - result_a.fill_price)
            pnl_b = position.units_b * (result_b.fill_price - position.entry_price_b)
        
        total_pnl = pnl_a + pnl_b
        
        # Record trade
        trade_record = {
            'pair': pair,
            'direction': position.direction,
            'entry_time': position.entry_time,
            'exit_time': datetime.now(),
            'entry_zscore': position.entry_zscore,
            'exit_zscore': signal.zscore,
            'entry_price_a': position.entry_price_a,
            'exit_price_a': result_a.fill_price,
            'entry_price_b': position.entry_price_b,
            'exit_price_b': result_b.fill_price,
            'units_a': position.units_a,
            'units_b': position.units_b,
            'pnl': total_pnl,
            'exit_reason': signal.reason
        }
        
        with self._lock:
            self.trade_history.append(trade_record)
            self.daily_pnl += total_pnl
            del self.positions[pair]
        
        # Update risk manager
        self.risk_manager.close_position(pair, total_pnl)
        
        exit_type = "STOP LOSS" if signal.type == SignalType.STOP_LOSS else "EXIT"
        logger.info(f"Closed {position.direction} on {pair[0]}/{pair[1]} "
                   f"({exit_type}): P/L=${total_pnl:.2f}")
        
        return True, f"Position closed: P/L=${total_pnl:.2f}"
    
    def _execute_order(
        self,
        instrument: str,
        units: float,
        side: OrderSide
    ) -> OrderResult:
        """Execute a single order via OANDA."""
        # Adjust units sign based on side
        if side == OrderSide.SELL:
            units = -abs(units)
        else:
            units = abs(units)
        
        # Round units to integer (OANDA requirement)
        units = int(round(units))
        
        if units == 0:
            return OrderResult(
                order_id="",
                instrument=instrument,
                side=side,
                units=0,
                requested_price=0,
                fill_price=0,
                status=OrderStatus.REJECTED,
                timestamp=datetime.now(),
                error_message="Units rounded to zero"
            )
        
        # Get current price for reference
        try:
            price_data = self.client.get_current_price(instrument)
            if side == OrderSide.BUY:
                requested_price = price_data.get('ask', 0)
            else:
                requested_price = price_data.get('bid', 0)
        except Exception as e:
            requested_price = 0
        
        # Execute based on mode
        if self.settings.mode == TradingMode.BACKTEST:
            # Simulated execution
            return self._simulate_execution(instrument, units, side, requested_price)
        elif self.settings.mode == TradingMode.PAPER:
            # Paper trading (still real API but practice account)
            return self._real_execution(instrument, units, side, requested_price)
        else:
            # Live trading
            return self._real_execution(instrument, units, side, requested_price)
    
    def _simulate_execution(
        self,
        instrument: str,
        units: int,
        side: OrderSide,
        price: float
    ) -> OrderResult:
        """Simulate order execution for backtesting."""
        # Add simulated slippage
        slippage_pips = self.settings.backtest.slippage_pips
        pip_value = 0.01 if 'JPY' in instrument else 0.0001
        
        if side == OrderSide.BUY:
            fill_price = price + slippage_pips * pip_value
        else:
            fill_price = price - slippage_pips * pip_value
        
        slippage = abs(fill_price - price)
        
        return OrderResult(
            order_id=f"SIM-{datetime.now().timestamp()}",
            instrument=instrument,
            side=side,
            units=abs(units),
            requested_price=price,
            fill_price=fill_price,
            status=OrderStatus.FILLED,
            timestamp=datetime.now(),
            slippage=slippage
        )
    
    def _real_execution(
        self,
        instrument: str,
        units: int,
        side: OrderSide,
        requested_price: float
    ) -> OrderResult:
        """Execute real order via OANDA API."""
        try:
            # Place market order
            response = self.client.place_market_order(
                instrument=instrument,
                units=units  # Already signed correctly
            )
            
            # Parse response
            if 'orderFillTransaction' in response:
                fill = response['orderFillTransaction']
                fill_price = float(fill.get('price', requested_price))
                order_id = fill.get('orderID', '')
                
                slippage = abs(fill_price - requested_price) if requested_price > 0 else 0
                
                return OrderResult(
                    order_id=order_id,
                    instrument=instrument,
                    side=side,
                    units=abs(units),
                    requested_price=requested_price,
                    fill_price=fill_price,
                    status=OrderStatus.FILLED,
                    timestamp=datetime.now(),
                    slippage=slippage
                )
            else:
                # Order rejected
                error_msg = response.get('errorMessage', 'Unknown error')
                return OrderResult(
                    order_id="",
                    instrument=instrument,
                    side=side,
                    units=abs(units),
                    requested_price=requested_price,
                    fill_price=0,
                    status=OrderStatus.REJECTED,
                    timestamp=datetime.now(),
                    error_message=error_msg
                )
                
        except Exception as e:
            logger.error(f"Order execution error: {e}")
            return OrderResult(
                order_id="",
                instrument=instrument,
                side=side,
                units=abs(units),
                requested_price=requested_price,
                fill_price=0,
                status=OrderStatus.REJECTED,
                timestamp=datetime.now(),
                error_message=str(e)
            )
    
    def _get_current_prices(
        self,
        pair: Tuple[str, str]
    ) -> Optional[Tuple[float, float]]:
        """Get current mid prices for both instruments."""
        try:
            prices = self.client.get_current_prices([pair[0], pair[1]])
            
            price_a = prices.get(pair[0], {}).get('mid')
            price_b = prices.get(pair[1], {}).get('mid')
            
            if price_a is None or price_b is None:
                return None
            
            return price_a, price_b
            
        except Exception as e:
            logger.error(f"Error getting prices: {e}")
            return None
    
    def _check_daily_reset(self) -> None:
        """Check if daily counters need reset."""
        today = datetime.now().date()
        if today != self.last_daily_reset:
            with self._lock:
                self.daily_trades = 0
                self.daily_pnl = 0.0
                self.last_daily_reset = today
            logger.info("Daily counters reset")
    
    def update_positions(self) -> None:
        """Update unrealized P/L for all positions."""
        for pair, position in self.positions.items():
            prices = self._get_current_prices(pair)
            if prices is None:
                continue
            
            price_a, price_b = prices
            
            if position.direction == 'long_spread':
                pnl_a = position.units_a * (price_a - position.entry_price_a)
                pnl_b = position.units_b * (position.entry_price_b - price_b)
            else:
                pnl_a = position.units_a * (position.entry_price_a - price_a)
                pnl_b = position.units_b * (price_b - position.entry_price_b)
            
            position.unrealized_pnl = pnl_a + pnl_b
    
    def get_state(self) -> ExecutionState:
        """Get current executor state."""
        return ExecutionState(
            mode=self.settings.mode,
            is_running=self.is_running,
            open_positions=self.positions.copy(),
            pending_orders=self.pending_orders.copy(),
            daily_trades=self.daily_trades,
            daily_pnl=self.daily_pnl,
            last_update=datetime.now()
        )
    
    def get_position_summary(self) -> pd.DataFrame:
        """Get summary of all positions."""
        if not self.positions:
            return pd.DataFrame()
        
        data = []
        for pair, pos in self.positions.items():
            data.append({
                'pair': f"{pair[0]}/{pair[1]}",
                'direction': pos.direction,
                'entry_time': pos.entry_time,
                'entry_zscore': pos.entry_zscore,
                'current_zscore': pos.current_zscore,
                'units_a': pos.units_a,
                'units_b': pos.units_b,
                'unrealized_pnl': pos.unrealized_pnl
            })
        
        return pd.DataFrame(data)
    
    def get_trade_history(self) -> pd.DataFrame:
        """Get trade history as DataFrame."""
        if not self.trade_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.trade_history)
    
    def close_all_positions(self) -> Dict[Tuple[str, str], Tuple[bool, str]]:
        """Emergency close all positions."""
        results = {}
        
        for pair in list(self.positions.keys()):
            # Create exit signal
            signal = Signal(
                type=SignalType.EXIT,
                pair=pair,
                timestamp=datetime.now(),
                zscore=0,
                correlation=0,
                hedge_ratio=self.positions[pair].hedge_ratio,
                confidence=1.0,
                reason="emergency_close"
            )
            
            success, msg = self._close_position(signal)
            results[pair] = (success, msg)
        
        return results
