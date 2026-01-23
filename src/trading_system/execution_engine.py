"""
Execution Engine - MT5 Order Management

Handles all order placement, modification, and tracking.
Enforces execution safety rules.

RULES:
- Every order MUST have SL and TP
- No pyramiding
- No averaging down
- No revenge trades (enforced by risk engine)
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, List, Dict, Tuple
from enum import Enum

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

from .config import SystemConfig, ExecutionConfig
from .signal_engine import TradeSignal, SignalDirection


class OrderResult(Enum):
    """Order execution result."""
    SUCCESS = "SUCCESS"
    FAILED_NO_CONNECTION = "FAILED_NO_CONNECTION"
    FAILED_INVALID_PARAMS = "FAILED_INVALID_PARAMS"
    FAILED_REJECTED = "FAILED_REJECTED"
    FAILED_TIMEOUT = "FAILED_TIMEOUT"
    FAILED_SLIPPAGE = "FAILED_SLIPPAGE"
    FAILED_DRY_RUN = "DRY_RUN"


@dataclass
class OrderExecution:
    """Order execution result details."""
    result: OrderResult
    ticket: Optional[int] = None
    symbol: str = ""
    direction: str = ""
    volume: float = 0.0
    price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    timestamp: datetime = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
    
    @property
    def is_success(self) -> bool:
        return self.result == OrderResult.SUCCESS
    
    def to_dict(self) -> dict:
        return {
            "result": self.result.value,
            "ticket": self.ticket,
            "symbol": self.symbol,
            "direction": self.direction,
            "volume": self.volume,
            "price": self.price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "error": self.error_message,
        }


@dataclass
class Position:
    """Current open position."""
    ticket: int
    symbol: str
    direction: str
    volume: float
    entry_price: float
    stop_loss: float
    take_profit: float
    current_price: float
    profit: float
    open_time: datetime


class ExecutionEngine:
    """
    MT5 execution engine.
    
    Handles order placement with safety checks.
    Supports dry-run mode for testing.
    """
    
    def __init__(
        self,
        config: SystemConfig,
        logger: Optional[logging.Logger] = None,
    ):
        self.config = config
        self.exec_config = config.execution
        self.logger = logger or logging.getLogger(__name__)
        
        self._connected = False
        self._dry_run = config.dry_run
        self._dry_run_ticket_counter = 10000
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    def connect(self) -> bool:
        """
        Connect to MT5 terminal.
        
        Returns True if connected successfully.
        """
        if not MT5_AVAILABLE:
            self.logger.error("MetaTrader5 package not installed")
            return False
        
        if self._dry_run:
            self.logger.info("DRY RUN MODE - No real connection")
            self._connected = True
            return True
        
        if not mt5.initialize():
            self.logger.error(f"MT5 initialization failed: {mt5.last_error()}")
            return False
        
        terminal_info = mt5.terminal_info()
        if terminal_info is None:
            self.logger.error("Failed to get terminal info")
            return False
        
        self.logger.info(f"Connected to MT5: {terminal_info.name}")
        self._connected = True
        return True
    
    def disconnect(self) -> None:
        """Disconnect from MT5."""
        if MT5_AVAILABLE and not self._dry_run:
            mt5.shutdown()
        self._connected = False
        self.logger.info("Disconnected from MT5")
    
    def get_account_equity(self) -> Optional[float]:
        """Get current account equity."""
        if self._dry_run:
            return 100000.0  # Simulated equity
        
        if not MT5_AVAILABLE or not self._connected:
            return None
        
        account_info = mt5.account_info()
        if account_info is None:
            return None
        
        return account_info.equity
    
    def get_account_balance(self) -> Optional[float]:
        """Get current account balance."""
        if self._dry_run:
            return 100000.0
        
        if not MT5_AVAILABLE or not self._connected:
            return None
        
        account_info = mt5.account_info()
        if account_info is None:
            return None
        
        return account_info.balance
    
    def get_open_positions(self) -> List[Position]:
        """Get all open positions for our magic number."""
        if self._dry_run:
            return []  # No positions in dry run
        
        if not MT5_AVAILABLE or not self._connected:
            return []
        
        positions = mt5.positions_get()
        if positions is None:
            return []
        
        our_positions = []
        for pos in positions:
            if pos.magic == self.exec_config.magic_number:
                direction = "LONG" if pos.type == mt5.POSITION_TYPE_BUY else "SHORT"
                our_positions.append(Position(
                    ticket=pos.ticket,
                    symbol=pos.symbol,
                    direction=direction,
                    volume=pos.volume,
                    entry_price=pos.price_open,
                    stop_loss=pos.sl,
                    take_profit=pos.tp,
                    current_price=pos.price_current,
                    profit=pos.profit,
                    open_time=datetime.fromtimestamp(pos.time, tz=timezone.utc),
                ))
        
        return our_positions
    
    def execute_signal(
        self,
        signal: TradeSignal,
        position_size: float,
        symbol: str,
    ) -> OrderExecution:
        """
        Execute a trade signal.
        
        Args:
            signal: Trade signal to execute
            position_size: Position size in lots
            symbol: Symbol to trade
        
        Returns:
            OrderExecution with result details
        """
        # Validate inputs
        if position_size <= 0:
            return OrderExecution(
                result=OrderResult.FAILED_INVALID_PARAMS,
                error_message="Invalid position size",
            )
        
        if signal.stop_loss <= 0 or signal.take_profit <= 0:
            return OrderExecution(
                result=OrderResult.FAILED_INVALID_PARAMS,
                error_message="SL/TP must be set",
            )
        
        # Enforce SL/TP requirement
        if self.exec_config.require_sl and signal.stop_loss == 0:
            return OrderExecution(
                result=OrderResult.FAILED_INVALID_PARAMS,
                error_message="Stop loss required but not set",
            )
        
        if self.exec_config.require_tp and signal.take_profit == 0:
            return OrderExecution(
                result=OrderResult.FAILED_INVALID_PARAMS,
                error_message="Take profit required but not set",
            )
        
        # DRY RUN mode
        if self._dry_run:
            self._dry_run_ticket_counter += 1
            self.logger.info(f"DRY RUN: Would execute {signal.direction.value} {symbol} "
                           f"@ {signal.entry_price:.5f}, SL: {signal.stop_loss:.5f}, "
                           f"TP: {signal.take_profit:.5f}, Size: {position_size}")
            
            return OrderExecution(
                result=OrderResult.FAILED_DRY_RUN,
                ticket=self._dry_run_ticket_counter,
                symbol=symbol,
                direction=signal.direction.value,
                volume=position_size,
                price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
            )
        
        # Live execution
        return self._execute_live(signal, position_size, symbol)
    
    def _execute_live(
        self,
        signal: TradeSignal,
        position_size: float,
        symbol: str,
    ) -> OrderExecution:
        """Execute order on live MT5."""
        if not MT5_AVAILABLE or not self._connected:
            return OrderExecution(
                result=OrderResult.FAILED_NO_CONNECTION,
                error_message="Not connected to MT5",
            )
        
        # Get symbol info
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return OrderExecution(
                result=OrderResult.FAILED_INVALID_PARAMS,
                error_message=f"Symbol {symbol} not found",
            )
        
        if not symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                return OrderExecution(
                    result=OrderResult.FAILED_INVALID_PARAMS,
                    error_message=f"Failed to select symbol {symbol}",
                )
        
        # Determine order type and price
        if signal.direction == SignalDirection.LONG:
            order_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(symbol).ask
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(symbol).bid
        
        # Build request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": position_size,
            "type": order_type,
            "price": price,
            "sl": signal.stop_loss,
            "tp": signal.take_profit,
            "deviation": self.exec_config.slippage_points,
            "magic": self.exec_config.magic_number,
            "comment": self.exec_config.order_comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Execute with retry
        for attempt in range(self.exec_config.max_order_retries):
            result = mt5.order_send(request)
            
            if result is None:
                self.logger.error(f"Order send returned None (attempt {attempt + 1})")
                time.sleep(self.exec_config.retry_delay_seconds)
                continue
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                self.logger.info(f"Order executed: {result.order}")
                return OrderExecution(
                    result=OrderResult.SUCCESS,
                    ticket=result.order,
                    symbol=symbol,
                    direction=signal.direction.value,
                    volume=result.volume,
                    price=result.price,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                )
            
            # Check for requote
            if result.retcode == mt5.TRADE_RETCODE_REQUOTE:
                self.logger.warning(f"Requote received (attempt {attempt + 1})")
                # Update price and retry
                if signal.direction == SignalDirection.LONG:
                    request["price"] = mt5.symbol_info_tick(symbol).ask
                else:
                    request["price"] = mt5.symbol_info_tick(symbol).bid
                time.sleep(self.exec_config.retry_delay_seconds)
                continue
            
            # Other error
            self.logger.error(f"Order failed: {result.retcode}, {result.comment}")
            break
        
        return OrderExecution(
            result=OrderResult.FAILED_REJECTED,
            symbol=symbol,
            direction=signal.direction.value,
            error_message=f"Retcode: {result.retcode if result else 'None'}",
        )
    
    def close_position(self, ticket: int) -> OrderExecution:
        """Close an open position by ticket."""
        if self._dry_run:
            self.logger.info(f"DRY RUN: Would close position {ticket}")
            return OrderExecution(
                result=OrderResult.FAILED_DRY_RUN,
                ticket=ticket,
            )
        
        if not MT5_AVAILABLE or not self._connected:
            return OrderExecution(
                result=OrderResult.FAILED_NO_CONNECTION,
                error_message="Not connected to MT5",
            )
        
        # Get position
        position = mt5.positions_get(ticket=ticket)
        if position is None or len(position) == 0:
            return OrderExecution(
                result=OrderResult.FAILED_INVALID_PARAMS,
                error_message=f"Position {ticket} not found",
            )
        
        pos = position[0]
        
        # Close order
        if pos.type == mt5.POSITION_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(pos.symbol).bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(pos.symbol).ask
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": self.exec_config.slippage_points,
            "magic": self.exec_config.magic_number,
            "comment": "CLOSE",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            return OrderExecution(
                result=OrderResult.FAILED_REJECTED,
                ticket=ticket,
                error_message=f"Close failed: {result.retcode if result else 'None'}",
            )
        
        return OrderExecution(
            result=OrderResult.SUCCESS,
            ticket=ticket,
            symbol=pos.symbol,
            price=result.price,
        )
    
    def modify_sl_tp(
        self,
        ticket: int,
        new_sl: Optional[float] = None,
        new_tp: Optional[float] = None,
    ) -> bool:
        """Modify SL/TP of an open position."""
        if self._dry_run:
            self.logger.info(f"DRY RUN: Would modify position {ticket}")
            return True
        
        if not MT5_AVAILABLE or not self._connected:
            return False
        
        position = mt5.positions_get(ticket=ticket)
        if position is None or len(position) == 0:
            return False
        
        pos = position[0]
        
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": pos.symbol,
            "position": ticket,
            "sl": new_sl if new_sl is not None else pos.sl,
            "tp": new_tp if new_tp is not None else pos.tp,
        }
        
        result = mt5.order_send(request)
        return result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
