"""
Live Executor for Pairs Trading System.

Handles real-time order execution via MetaTrader 5 for IC Markets Global.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
from enum import Enum
import logging
import threading
import time

from config.settings import Settings, TradingMode
from config.broker_config import MT5Config
from src.data.broker_client import MT5Client, OrderType, OrderResult
from src.risk.risk_manager import RiskManager
from src.strategy.signals import Signal, SignalType


logger = logging.getLogger(__name__)


class OrderSide(Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order execution status."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


@dataclass
class PairLeg:
    """Single leg of a pairs trade."""
    symbol: str
    side: OrderSide
    volume: float
    ticket: int = 0
    entry_price: float = 0.0
    current_price: float = 0.0
    pnl: float = 0.0


@dataclass
class PairPosition:
    """Open pairs position."""
    pair: Tuple[str, str]
    direction: str  # 'long_spread' or 'short_spread'
    leg_a: PairLeg
    leg_b: PairLeg
    hedge_ratio: float
    entry_zscore: float
    entry_time: datetime
    unrealized_pnl: float = 0.0
    
    def update_pnl(self):
        """Update unrealized P&L."""
        self.leg_a.pnl = self._calculate_leg_pnl(self.leg_a)
        self.leg_b.pnl = self._calculate_leg_pnl(self.leg_b)
        self.unrealized_pnl = self.leg_a.pnl + self.leg_b.pnl
    
    def _calculate_leg_pnl(self, leg: PairLeg) -> float:
        """Calculate P&L for a leg."""
        if leg.entry_price == 0:
            return 0.0
        
        if leg.side == OrderSide.BUY:
            return (leg.current_price - leg.entry_price) * leg.volume * 100000
        else:
            return (leg.entry_price - leg.current_price) * leg.volume * 100000


@dataclass
class ExecutionState:
    """Current execution state."""
    open_positions: Dict[Tuple[str, str], PairPosition] = field(default_factory=dict)
    daily_trades: int = 0
    daily_pnl: float = 0.0
    last_reset: datetime = field(default_factory=datetime.now)
    is_halted: bool = False
    halt_reason: str = ""


class LiveExecutor:
    """
    Live trading executor for pairs trading via MT5.
    
    Handles:
    - Order execution for both legs
    - Position tracking
    - Risk checks before execution
    - Atomic execution with rollback on failure
    """
    
    def __init__(
        self,
        settings: Settings,
        mt5_config: MT5Config,
        risk_manager: RiskManager
    ):
        """
        Initialize executor.
        
        Args:
            settings: Trading settings
            mt5_config: MT5 configuration
            risk_manager: Risk manager instance
        """
        self.settings = settings
        self.config = mt5_config
        self.risk_manager = risk_manager
        
        self.client: Optional[MT5Client] = None
        self.positions: Dict[Tuple[str, str], PairPosition] = {}
        self.state = ExecutionState()
        
        self._lock = threading.Lock()
        self._running = False
        self._trade_history: List[Dict] = []
    
    def start(self):
        """Start executor and connect to MT5."""
        self.client = MT5Client(self.config)
        
        if not self.client.connect():
            raise ConnectionError("Failed to connect to MT5")
        
        self._running = True
        logger.info("LiveExecutor started")
        
        # Log account info
        info = self.client.get_account_info()
        logger.info(f"Account: {info.get('login')} | Balance: ${info.get('balance', 0):,.2f}")
    
    def stop(self):
        """Stop executor and disconnect."""
        self._running = False
        
        if self.client:
            self.client.disconnect()
        
        logger.info("LiveExecutor stopped")
    
    def execute_signal(self, signal: Signal) -> Tuple[bool, str]:
        """
        Execute a trading signal.
        
        Args:
            signal: Signal to execute
            
        Returns:
            (success, message) tuple
        """
        with self._lock:
            # Check if halted
            if self.state.is_halted:
                return False, f"Trading halted: {self.state.halt_reason}"
            
            # Reset daily stats if new day
            self._check_daily_reset()
            
            # Route signal
            if signal.type == SignalType.LONG_SPREAD:
                return self._open_long_spread(signal)
            
            elif signal.type == SignalType.SHORT_SPREAD:
                return self._open_short_spread(signal)
            
            elif signal.type in [SignalType.EXIT, SignalType.STOP_LOSS]:
                return self._close_position(signal)
            
            elif signal.type == SignalType.NO_SIGNAL:
                return True, "No action required"
            
            return False, f"Unknown signal type: {signal.type}"
    
    def _open_long_spread(self, signal: Signal) -> Tuple[bool, str]:
        """
        Open long spread position.
        Long spread = Buy A, Sell B
        """
        pair = signal.pair
        
        # Check if already in position
        if pair in self.positions:
            return False, f"Already in position for {pair[0]}/{pair[1]}"
        
        # Risk checks
        if not self.risk_manager.can_open_position(pair):
            return False, "Risk check failed"
        
        if self.state.daily_trades >= self.settings.risk.max_daily_trades:
            return False, "Daily trade limit reached"
        
        # Calculate position sizes
        balance = self.client.get_balance()
        risk_amount = balance * self.settings.risk.max_risk_per_trade
        
        # Get current prices
        tick_a = self.client.get_tick(pair[0])
        tick_b = self.client.get_tick(pair[1])
        
        if not tick_a or not tick_b:
            return False, "Failed to get current prices"
        
        # Calculate lot sizes (balanced by hedge ratio)
        hedge_ratio = signal.hedge_ratio if signal.hedge_ratio else 1.0
        base_volume = self._calculate_volume(pair[0], risk_amount / 2)
        
        volume_a = base_volume
        volume_b = base_volume * hedge_ratio
        
        # Round to lot step
        volume_a = self._round_volume(pair[0], volume_a)
        volume_b = self._round_volume(pair[1], volume_b)
        
        # Execute leg A (BUY)
        result_a = self.client.market_order(
            symbol=pair[0],
            order_type=OrderType.BUY,
            volume=volume_a,
            comment=f"PT Long A {pair[1]}"
        )
        
        if not result_a.success:
            return False, f"Failed to open leg A: {result_a.error_message}"
        
        # Execute leg B (SELL)
        result_b = self.client.market_order(
            symbol=pair[1],
            order_type=OrderType.SELL,
            volume=volume_b,
            comment=f"PT Long B {pair[0]}"
        )
        
        if not result_b.success:
            # Rollback leg A
            self.client.close_position(result_a.ticket, comment="Rollback")
            return False, f"Failed to open leg B, rolled back: {result_b.error_message}"
        
        # Create position
        position = PairPosition(
            pair=pair,
            direction='long_spread',
            leg_a=PairLeg(
                symbol=pair[0],
                side=OrderSide.BUY,
                volume=volume_a,
                ticket=result_a.ticket,
                entry_price=result_a.price
            ),
            leg_b=PairLeg(
                symbol=pair[1],
                side=OrderSide.SELL,
                volume=volume_b,
                ticket=result_b.ticket,
                entry_price=result_b.price
            ),
            hedge_ratio=hedge_ratio,
            entry_zscore=signal.zscore,
            entry_time=datetime.now()
        )
        
        self.positions[pair] = position
        self.state.daily_trades += 1
        
        # Record in risk manager
        self.risk_manager.record_entry(
            pair=pair,
            direction='long_spread',
            size_a=volume_a,
            size_b=volume_b
        )
        
        logger.info(f"Opened LONG SPREAD: {pair[0]}/{pair[1]} @ Z={signal.zscore:.2f}")
        
        return True, f"Long spread opened: {pair[0]} +{volume_a} / {pair[1]} -{volume_b}"
    
    def _open_short_spread(self, signal: Signal) -> Tuple[bool, str]:
        """
        Open short spread position.
        Short spread = Sell A, Buy B
        """
        pair = signal.pair
        
        # Check if already in position
        if pair in self.positions:
            return False, f"Already in position for {pair[0]}/{pair[1]}"
        
        # Risk checks
        if not self.risk_manager.can_open_position(pair):
            return False, "Risk check failed"
        
        if self.state.daily_trades >= self.settings.risk.max_daily_trades:
            return False, "Daily trade limit reached"
        
        # Calculate position sizes
        balance = self.client.get_balance()
        risk_amount = balance * self.settings.risk.max_risk_per_trade
        
        # Get current prices
        tick_a = self.client.get_tick(pair[0])
        tick_b = self.client.get_tick(pair[1])
        
        if not tick_a or not tick_b:
            return False, "Failed to get current prices"
        
        # Calculate lot sizes
        hedge_ratio = signal.hedge_ratio if signal.hedge_ratio else 1.0
        base_volume = self._calculate_volume(pair[0], risk_amount / 2)
        
        volume_a = base_volume
        volume_b = base_volume * hedge_ratio
        
        volume_a = self._round_volume(pair[0], volume_a)
        volume_b = self._round_volume(pair[1], volume_b)
        
        # Execute leg A (SELL)
        result_a = self.client.market_order(
            symbol=pair[0],
            order_type=OrderType.SELL,
            volume=volume_a,
            comment=f"PT Short A {pair[1]}"
        )
        
        if not result_a.success:
            return False, f"Failed to open leg A: {result_a.error_message}"
        
        # Execute leg B (BUY)
        result_b = self.client.market_order(
            symbol=pair[1],
            order_type=OrderType.BUY,
            volume=volume_b,
            comment=f"PT Short B {pair[0]}"
        )
        
        if not result_b.success:
            # Rollback leg A
            self.client.close_position(result_a.ticket, comment="Rollback")
            return False, f"Failed to open leg B, rolled back: {result_b.error_message}"
        
        # Create position
        position = PairPosition(
            pair=pair,
            direction='short_spread',
            leg_a=PairLeg(
                symbol=pair[0],
                side=OrderSide.SELL,
                volume=volume_a,
                ticket=result_a.ticket,
                entry_price=result_a.price
            ),
            leg_b=PairLeg(
                symbol=pair[1],
                side=OrderSide.BUY,
                volume=volume_b,
                ticket=result_b.ticket,
                entry_price=result_b.price
            ),
            hedge_ratio=hedge_ratio,
            entry_zscore=signal.zscore,
            entry_time=datetime.now()
        )
        
        self.positions[pair] = position
        self.state.daily_trades += 1
        
        # Record in risk manager
        self.risk_manager.record_entry(
            pair=pair,
            direction='short_spread',
            size_a=volume_a,
            size_b=volume_b
        )
        
        logger.info(f"Opened SHORT SPREAD: {pair[0]}/{pair[1]} @ Z={signal.zscore:.2f}")
        
        return True, f"Short spread opened: {pair[0]} -{volume_a} / {pair[1]} +{volume_b}"
    
    def _close_position(self, signal: Signal) -> Tuple[bool, str]:
        """Close an existing pairs position."""
        pair = signal.pair
        
        if pair not in self.positions:
            return False, f"No position found for {pair[0]}/{pair[1]}"
        
        position = self.positions[pair]
        
        # Close leg A
        result_a = self.client.close_position(
            ticket=position.leg_a.ticket,
            comment=f"PT Close {signal.type.value}"
        )
        
        # Close leg B
        result_b = self.client.close_position(
            ticket=position.leg_b.ticket,
            comment=f"PT Close {signal.type.value}"
        )
        
        # Calculate final P&L
        total_pnl = 0.0
        
        if result_a.success:
            pnl_a = self._calculate_close_pnl(position.leg_a, result_a.price)
            total_pnl += pnl_a
        
        if result_b.success:
            pnl_b = self._calculate_close_pnl(position.leg_b, result_b.price)
            total_pnl += pnl_b
        
        # Record trade
        self._record_trade(position, total_pnl, signal.type.value)
        
        # Update state
        self.state.daily_pnl += total_pnl
        
        # Record in risk manager
        self.risk_manager.record_exit(pair=pair, pnl=total_pnl)
        
        # Remove position
        del self.positions[pair]
        
        close_type = "Stop Loss" if signal.type == SignalType.STOP_LOSS else "Exit"
        logger.info(f"Closed {position.direction}: {pair[0]}/{pair[1]} | P&L: ${total_pnl:.2f} ({close_type})")
        
        return True, f"Position closed: P&L ${total_pnl:.2f}"
    
    def _calculate_volume(self, symbol: str, risk_amount: float) -> float:
        """Calculate lot size based on risk amount."""
        # Simplified calculation - 1 pip risk = 10 USD per lot for most pairs
        pip_value = self.client.get_pip_value(symbol, 1.0)
        
        if pip_value <= 0:
            return 0.01  # Minimum lot
        
        # Assume 30 pip stop loss equivalent
        stop_loss_pips = 30
        volume = risk_amount / (stop_loss_pips * pip_value)
        
        return max(0.01, volume)
    
    def _round_volume(self, symbol: str, volume: float) -> float:
        """Round volume to symbol's lot step."""
        info = self.client.get_symbol_info(symbol)
        if info is None:
            return round(volume, 2)
        
        step = info.volume_step
        volume = round(volume / step) * step
        
        return max(info.volume_min, min(volume, info.volume_max))
    
    def _calculate_close_pnl(self, leg: PairLeg, close_price: float) -> float:
        """Calculate P&L for closing a leg."""
        if leg.entry_price == 0:
            return 0.0
        
        # Contract size is typically 100,000 for forex
        contract_size = 100000
        
        if leg.side == OrderSide.BUY:
            return (close_price - leg.entry_price) * leg.volume * contract_size
        else:
            return (leg.entry_price - close_price) * leg.volume * contract_size
    
    def _record_trade(self, position: PairPosition, pnl: float, exit_reason: str):
        """Record completed trade."""
        trade = {
            'pair': f"{position.pair[0]}/{position.pair[1]}",
            'direction': position.direction,
            'entry_time': position.entry_time,
            'exit_time': datetime.now(),
            'entry_zscore': position.entry_zscore,
            'hedge_ratio': position.hedge_ratio,
            'leg_a_symbol': position.leg_a.symbol,
            'leg_a_side': position.leg_a.side.value,
            'leg_a_volume': position.leg_a.volume,
            'leg_a_entry': position.leg_a.entry_price,
            'leg_b_symbol': position.leg_b.symbol,
            'leg_b_side': position.leg_b.side.value,
            'leg_b_volume': position.leg_b.volume,
            'leg_b_entry': position.leg_b.entry_price,
            'pnl': pnl,
            'exit_reason': exit_reason
        }
        
        self._trade_history.append(trade)
    
    def _check_daily_reset(self):
        """Reset daily stats if new trading day."""
        now = datetime.now()
        
        if now.date() > self.state.last_reset.date():
            self.state.daily_trades = 0
            self.state.daily_pnl = 0.0
            self.state.last_reset = now
            logger.info("Daily stats reset")
    
    def update_positions(self):
        """Update all position prices and P&L."""
        for pair, position in self.positions.items():
            # Update leg A
            tick_a = self.client.get_tick(pair[0])
            if tick_a:
                if position.leg_a.side == OrderSide.BUY:
                    position.leg_a.current_price = tick_a['bid']
                else:
                    position.leg_a.current_price = tick_a['ask']
            
            # Update leg B
            tick_b = self.client.get_tick(pair[1])
            if tick_b:
                if position.leg_b.side == OrderSide.BUY:
                    position.leg_b.current_price = tick_b['bid']
                else:
                    position.leg_b.current_price = tick_b['ask']
            
            # Recalculate P&L
            position.update_pnl()
        
        # Check daily loss limit
        if self.state.daily_pnl < -self.client.get_balance() * self.settings.risk.max_daily_loss:
            self.state.is_halted = True
            self.state.halt_reason = "Daily loss limit reached"
            logger.warning(f"Trading halted: {self.state.halt_reason}")
    
    def close_all_positions(self, reason: str = "Manual close") -> Dict[Tuple[str, str], Tuple[bool, str]]:
        """Close all open positions."""
        results = {}
        
        for pair in list(self.positions.keys()):
            signal = Signal(
                type=SignalType.EXIT,
                pair=pair,
                timestamp=datetime.now(),
                zscore=0.0,
                strength=1.0,
                reason=reason
            )
            
            success, msg = self._close_position(signal)
            results[pair] = (success, msg)
        
        return results
    
    def get_state(self) -> ExecutionState:
        """Get current execution state."""
        self.state.open_positions = self.positions.copy()
        return self.state
    
    def get_trade_history(self) -> pd.DataFrame:
        """Get trade history as DataFrame."""
        if not self._trade_history:
            return pd.DataFrame()
        return pd.DataFrame(self._trade_history)
    
    def halt_trading(self, reason: str):
        """Halt all trading."""
        self.state.is_halted = True
        self.state.halt_reason = reason
        logger.warning(f"Trading halted: {reason}")
    
    def resume_trading(self):
        """Resume trading."""
        self.state.is_halted = False
        self.state.halt_reason = ""
        logger.info("Trading resumed")
