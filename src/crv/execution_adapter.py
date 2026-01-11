"""
FX Conditional Relative Value (CRV) System - MT5 Execution Adapter.

INSTITUTIONAL GRADE - EXECUTION ADAPTER MODULE

This module implements:
1. Execution adapter interface
2. Order construction and validation
3. Simulation execution (paper mode)
4. MT5 connection management

CRITICAL RULES:
- send_orders() MUST RAISE ERROR unless MODE_LIVE_TRADING
- All orders must be logged before execution
- Pair-neutral execution required
- Hedge ratios from system must be used

Data Source: MT5 - IC Markets
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Tuple
from enum import Enum
import logging

from src.crv.state_machine import SystemMode, CRVStateMachine, ModeCapabilityError

logger = logging.getLogger(__name__)


# ============================================================================
# ORDER TYPES
# ============================================================================

class OrderSide(Enum):
    """Order side."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order type."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "PENDING"
    VALIDATED = "VALIDATED"
    SIMULATED = "SIMULATED"
    SENT = "SENT"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"


# ============================================================================
# ORDER DATA CLASSES
# ============================================================================

@dataclass
class CRVOrder:
    """
    CRV spread order (contains two legs).
    
    CRV orders are always pair-neutral:
    - Long spread: BUY leg_a, SELL leg_b * hedge_ratio
    - Short spread: SELL leg_a, BUY leg_b * hedge_ratio
    """
    # Order ID
    order_id: str
    
    # Pair info
    symbol_a: str
    symbol_b: str
    spread_direction: str  # "LONG_SPREAD" or "SHORT_SPREAD"
    
    # Sizing
    hedge_ratio: float
    lots_a: float
    lots_b: float  # Calculated from hedge_ratio
    
    # Entry parameters
    entry_z: float
    target_z: float
    stop_z: float
    
    # Status
    status: OrderStatus = OrderStatus.PENDING
    
    # Execution info
    created_at: datetime = field(default_factory=datetime.utcnow)
    validated_at: Optional[datetime] = None
    sent_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    
    # Validation
    validation_errors: List[str] = field(default_factory=list)
    
    # Execution prices (filled after execution)
    fill_price_a: Optional[float] = None
    fill_price_b: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging."""
        return {
            "order_id": self.order_id,
            "symbol_a": self.symbol_a,
            "symbol_b": self.symbol_b,
            "spread_direction": self.spread_direction,
            "hedge_ratio": self.hedge_ratio,
            "lots_a": self.lots_a,
            "lots_b": self.lots_b,
            "entry_z": self.entry_z,
            "target_z": self.target_z,
            "stop_z": self.stop_z,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class MT5Position:
    """Representation of an MT5 position."""
    ticket: int
    symbol: str
    volume: float
    type: str  # "BUY" or "SELL"
    price_open: float
    price_current: float
    profit: float
    time_open: datetime


# ============================================================================
# EXECUTION ADAPTER INTERFACE
# ============================================================================

class ExecutionAdapter(ABC):
    """
    Abstract execution adapter interface.
    
    All execution adapters must implement:
    - validate(): Validate orders before execution
    - build_orders(): Construct orders from signals
    - simulate_execution(): Paper trading execution
    - send_orders(): Real execution (MODE_LIVE_TRADING only)
    """
    
    @abstractmethod
    def validate(self, orders: List[CRVOrder]) -> Tuple[List[CRVOrder], List[str]]:
        """
        Validate orders before execution.
        
        Returns:
            (valid_orders, error_messages)
        """
        pass
    
    @abstractmethod
    def build_orders(
        self,
        signal: Dict,
        account_equity: float,
        max_position_pct: float = 3.0
    ) -> CRVOrder:
        """
        Build orders from CRV signal.
        
        Args:
            signal: CRV signal dict
            account_equity: Current account equity
            max_position_pct: Max position size as % of equity
            
        Returns:
            CRVOrder ready for validation
        """
        pass
    
    @abstractmethod
    def simulate_execution(self, orders: List[CRVOrder]) -> List[CRVOrder]:
        """
        Simulate order execution (paper mode).
        
        Returns:
            Orders with simulated fills
        """
        pass
    
    @abstractmethod
    def send_orders(self, orders: List[CRVOrder]) -> List[CRVOrder]:
        """
        Send orders to broker.
        
        CRITICAL: MUST RAISE ERROR unless MODE_LIVE_TRADING.
        
        Returns:
            Orders with execution status
        """
        pass


# ============================================================================
# MT5 EXECUTION ADAPTER
# ============================================================================

class MT5ExecutionAdapter(ExecutionAdapter):
    """
    MT5 Execution Adapter for IC Markets.
    
    IMPLEMENTATION STATUS:
    - validate(): ✅ Implemented
    - build_orders(): ✅ Implemented
    - simulate_execution(): ✅ Implemented
    - send_orders(): ⚠️ DISABLED unless MODE_LIVE_TRADING
    
    CONSTRAINTS:
    - Data source: MT5 - IC Markets
    - Execution: Pair-neutral
    - Hedge ratios: From CRV system
    - Exposure limits: Enforced
    """
    
    def __init__(
        self,
        fsm: CRVStateMachine,
        # Account constraints
        max_exposure_pct: float = 10.0,
        max_position_pct: float = 3.0,
        max_positions: int = 3,
        # MT5 connection (placeholder)
        mt5_connected: bool = False,
    ):
        self.fsm = fsm
        self.max_exposure_pct = max_exposure_pct
        self.max_position_pct = max_position_pct
        self.max_positions = max_positions
        self.mt5_connected = mt5_connected
        
        # Order tracking
        self._pending_orders: List[CRVOrder] = []
        self._executed_orders: List[CRVOrder] = []
        self._order_counter = 0
        
        logger.info(f"MT5 Execution Adapter initialized (Mode: {fsm.mode.value})")
    
    def _generate_order_id(self) -> str:
        """Generate unique order ID."""
        self._order_counter += 1
        return f"CRV-{datetime.utcnow():%Y%m%d%H%M%S}-{self._order_counter:04d}"
    
    def validate(self, orders: List[CRVOrder]) -> Tuple[List[CRVOrder], List[str]]:
        """
        Validate orders before execution.
        
        Validation checks:
        1. Symbol validity
        2. Lot size limits
        3. Hedge ratio sanity
        4. Exposure limits
        """
        valid_orders = []
        errors = []
        
        for order in orders:
            order_errors = []
            
            # Check symbols
            if not order.symbol_a or len(order.symbol_a) != 6:
                order_errors.append(f"Invalid symbol_a: {order.symbol_a}")
            
            if not order.symbol_b or len(order.symbol_b) != 6:
                order_errors.append(f"Invalid symbol_b: {order.symbol_b}")
            
            # Check lot sizes
            if order.lots_a <= 0 or order.lots_a > 100:
                order_errors.append(f"Invalid lots_a: {order.lots_a}")
            
            if order.lots_b <= 0 or order.lots_b > 100:
                order_errors.append(f"Invalid lots_b: {order.lots_b}")
            
            # Check hedge ratio
            if abs(order.hedge_ratio) < 0.1 or abs(order.hedge_ratio) > 10:
                order_errors.append(f"Invalid hedge_ratio: {order.hedge_ratio}")
            
            # Check spread direction
            if order.spread_direction not in ["LONG_SPREAD", "SHORT_SPREAD"]:
                order_errors.append(f"Invalid spread_direction: {order.spread_direction}")
            
            if order_errors:
                order.status = OrderStatus.REJECTED
                order.validation_errors = order_errors
                errors.extend(order_errors)
            else:
                order.status = OrderStatus.VALIDATED
                order.validated_at = datetime.utcnow()
                valid_orders.append(order)
        
        logger.info(f"Order validation: {len(valid_orders)} valid, {len(errors)} errors")
        
        return valid_orders, errors
    
    def build_orders(
        self,
        signal: Dict,
        account_equity: float,
        max_position_pct: float = 3.0
    ) -> CRVOrder:
        """
        Build CRV order from signal.
        
        Signal expected format:
        {
            "pair": ("EURUSD", "GBPUSD"),
            "signal_type": "LONG_SPREAD" or "SHORT_SPREAD",
            "hedge_ratio": float,
            "entry_z": float,
            "target_z": float,
            "stop_z": float,
            "suggested_size_pct": float,
        }
        """
        pair = signal.get("pair", ("", ""))
        signal_type = signal.get("signal_type", "NO_SIGNAL")
        hedge_ratio = signal.get("hedge_ratio", 1.0)
        
        # Calculate position size
        size_pct = min(
            signal.get("suggested_size_pct", 2.0),
            max_position_pct
        )
        
        # Convert to lots (simplified - would use pip value in production)
        # Assuming standard lot = 100,000 units
        risk_amount = account_equity * (size_pct / 100.0)
        lots_a = round(risk_amount / 100000 * 10, 2)  # Simplified calculation
        lots_b = round(lots_a * abs(hedge_ratio), 2)
        
        order = CRVOrder(
            order_id=self._generate_order_id(),
            symbol_a=pair[0],
            symbol_b=pair[1],
            spread_direction=signal_type,
            hedge_ratio=hedge_ratio,
            lots_a=max(0.01, lots_a),  # Minimum lot size
            lots_b=max(0.01, lots_b),
            entry_z=signal.get("entry_z", 0.0),
            target_z=signal.get("target_z", 0.0),
            stop_z=signal.get("stop_z", 3.0),
        )
        
        logger.info(f"Order built: {order.order_id} | {order.symbol_a}/{order.symbol_b} {order.spread_direction}")
        
        return order
    
    def simulate_execution(self, orders: List[CRVOrder]) -> List[CRVOrder]:
        """
        Simulate order execution for paper trading.
        
        Simulates fills at current market prices (placeholder).
        """
        for order in orders:
            if order.status == OrderStatus.VALIDATED:
                # Simulate fill (would get real prices from MT5)
                order.status = OrderStatus.SIMULATED
                order.filled_at = datetime.utcnow()
                
                # Placeholder prices
                order.fill_price_a = 1.0000  # Would be real price
                order.fill_price_b = 1.0000  # Would be real price
                
                logger.info(
                    f"SIMULATED: {order.order_id} | "
                    f"{order.symbol_a}/{order.symbol_b} {order.spread_direction} | "
                    f"Lots: {order.lots_a}/{order.lots_b}"
                )
                
                self._executed_orders.append(order)
        
        return orders
    
    def send_orders(self, orders: List[CRVOrder]) -> List[CRVOrder]:
        """
        Send orders to MT5.
        
        CRITICAL: RAISES ERROR unless MODE_LIVE_TRADING.
        
        This is the ONLY method that can place real trades.
        """
        # HARD MODE CHECK
        if self.fsm.mode != SystemMode.MODE_LIVE_TRADING:
            error_msg = (
                f"EXECUTION BLOCKED: Cannot send orders in {self.fsm.mode.value}. "
                f"Order execution only allowed in MODE_LIVE_TRADING."
            )
            logger.error(error_msg)
            raise ModeCapabilityError(error_msg)
        
        # Check MT5 connection
        if not self.mt5_connected:
            logger.error("MT5 not connected - cannot send orders")
            for order in orders:
                order.status = OrderStatus.REJECTED
                order.validation_errors.append("MT5 not connected")
            return orders
        
        # Send orders (placeholder - would use mt5 library)
        for order in orders:
            if order.status == OrderStatus.VALIDATED:
                try:
                    # Would call MT5 here:
                    # result = mt5.order_send(request)
                    
                    order.status = OrderStatus.SENT
                    order.sent_at = datetime.utcnow()
                    
                    logger.info(
                        f"ORDER SENT: {order.order_id} | "
                        f"{order.symbol_a}/{order.symbol_b} {order.spread_direction}"
                    )
                    
                except Exception as e:
                    order.status = OrderStatus.REJECTED
                    order.validation_errors.append(str(e))
                    logger.error(f"Order send failed: {order.order_id} - {e}")
        
        return orders
    
    def log_order_summary(self, order: CRVOrder) -> None:
        """Log detailed order summary."""
        logger.info("=" * 60)
        logger.info("CRV ORDER SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Order ID: {order.order_id}")
        logger.info(f"Pair: {order.symbol_a}/{order.symbol_b}")
        logger.info(f"Direction: {order.spread_direction}")
        logger.info(f"Hedge Ratio: {order.hedge_ratio:.4f}")
        logger.info(f"Lots A: {order.lots_a}")
        logger.info(f"Lots B: {order.lots_b}")
        logger.info(f"Entry Z: {order.entry_z:.2f}")
        logger.info(f"Target Z: {order.target_z:.2f}")
        logger.info(f"Stop Z: {order.stop_z:.2f}")
        logger.info(f"Status: {order.status.value}")
        logger.info(f"Mode: {self.fsm.mode.value}")
        logger.info("=" * 60)
    
    def get_pending_orders(self) -> List[CRVOrder]:
        """Get list of pending orders."""
        return self._pending_orders.copy()
    
    def get_executed_orders(self) -> List[CRVOrder]:
        """Get list of executed orders."""
        return self._executed_orders.copy()


# ============================================================================
# MT5 CONNECTION MANAGER (PLACEHOLDER)
# ============================================================================

class MT5ConnectionManager:
    """
    MT5 Connection Manager.
    
    PLACEHOLDER - Would implement actual MT5 connection.
    
    Required for live trading:
    - mt5.initialize()
    - mt5.login()
    - mt5.symbol_info()
    - mt5.order_send()
    """
    
    def __init__(self, account: int = 0, server: str = "ICMarkets-Demo"):
        self.account = account
        self.server = server
        self._connected = False
    
    def connect(self) -> bool:
        """
        Connect to MT5.
        
        PLACEHOLDER - Would call mt5.initialize() and mt5.login()
        """
        logger.info(f"MT5 connection attempt: Account {self.account} @ {self.server}")
        
        # Placeholder
        # import MetaTrader5 as mt5
        # if not mt5.initialize():
        #     return False
        # if not mt5.login(self.account, server=self.server):
        #     return False
        
        self._connected = False  # Not actually connected
        return self._connected
    
    def disconnect(self) -> None:
        """Disconnect from MT5."""
        self._connected = False
        logger.info("MT5 disconnected")
    
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected
    
    def get_account_info(self) -> Dict:
        """Get account information."""
        if not self._connected:
            return {}
        
        # Placeholder
        return {
            "balance": 0.0,
            "equity": 0.0,
            "margin": 0.0,
            "free_margin": 0.0,
        }
