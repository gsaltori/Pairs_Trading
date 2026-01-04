"""
MetaTrader 5 Client for IC Markets Global.

Handles all MT5 API interactions:
- Connection management
- Historical data retrieval
- Order execution
- Account information
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
import time

from config.broker_config import MT5Config, SymbolInfo


logger = logging.getLogger(__name__)


class OrderType(Enum):
    """MT5 Order types."""
    BUY = mt5.ORDER_TYPE_BUY
    SELL = mt5.ORDER_TYPE_SELL
    BUY_LIMIT = mt5.ORDER_TYPE_BUY_LIMIT
    SELL_LIMIT = mt5.ORDER_TYPE_SELL_LIMIT
    BUY_STOP = mt5.ORDER_TYPE_BUY_STOP
    SELL_STOP = mt5.ORDER_TYPE_SELL_STOP


class Timeframe(Enum):
    """MT5 Timeframes."""
    M1 = mt5.TIMEFRAME_M1
    M5 = mt5.TIMEFRAME_M5
    M15 = mt5.TIMEFRAME_M15
    M30 = mt5.TIMEFRAME_M30
    H1 = mt5.TIMEFRAME_H1
    H4 = mt5.TIMEFRAME_H4
    D1 = mt5.TIMEFRAME_D1
    W1 = mt5.TIMEFRAME_W1
    MN1 = mt5.TIMEFRAME_MN1
    
    @classmethod
    def from_string(cls, tf_str: str) -> 'Timeframe':
        """Convert string to Timeframe."""
        mapping = {
            'M1': cls.M1, 'M5': cls.M5, 'M15': cls.M15, 'M30': cls.M30,
            'H1': cls.H1, 'H4': cls.H4, 'D1': cls.D1, 'W1': cls.W1, 'MN1': cls.MN1
        }
        return mapping.get(tf_str.upper(), cls.H1)


@dataclass
class OrderResult:
    """Result of an order execution."""
    success: bool
    ticket: int
    symbol: str
    volume: float
    price: float
    order_type: str
    comment: str
    error_code: int = 0
    error_message: str = ""


@dataclass
class Position:
    """Open position information."""
    ticket: int
    symbol: str
    volume: float
    price_open: float
    price_current: float
    profit: float
    swap: float
    type: str  # 'buy' or 'sell'
    magic: int
    comment: str
    time: datetime


class MT5Client:
    """
    MetaTrader 5 Client for IC Markets Global.
    
    Provides unified interface for:
    - Historical data retrieval
    - Real-time quotes
    - Order execution
    - Position management
    """
    
    def __init__(self, config: MT5Config):
        """Initialize MT5 client."""
        self.config = config
        self._connected = False
        self._symbol_cache: Dict[str, SymbolInfo] = {}
    
    def connect(self) -> bool:
        """
        Initialize and connect to MT5 terminal.
        
        Returns:
            True if connected successfully
        """
        if self._connected:
            return True
        
        # Initialize MT5
        init_params = {}
        if self.config.terminal_path:
            init_params['path'] = self.config.terminal_path
        
        if not mt5.initialize(**init_params):
            error = mt5.last_error()
            logger.error(f"MT5 initialization failed: {error}")
            return False
        
        # Login to account
        if self.config.login > 0:
            authorized = mt5.login(
                login=self.config.login,
                password=self.config.password,
                server=self.config.server,
                timeout=self.config.timeout
            )
            
            if not authorized:
                error = mt5.last_error()
                logger.error(f"MT5 login failed: {error}")
                mt5.shutdown()
                return False
        
        self._connected = True
        logger.info(f"Connected to MT5: {self.config.server}")
        return True
    
    def disconnect(self):
        """Disconnect from MT5 terminal."""
        if self._connected:
            mt5.shutdown()
            self._connected = False
            logger.info("Disconnected from MT5")
    
    def ensure_connected(self):
        """Ensure connection is active, reconnect if needed."""
        if not self._connected:
            if not self.connect():
                raise ConnectionError("Failed to connect to MT5")
    
    # ==================== Account Information ====================
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get account information."""
        self.ensure_connected()
        
        info = mt5.account_info()
        if info is None:
            return {}
        
        return {
            'login': info.login,
            'server': info.server,
            'balance': info.balance,
            'equity': info.equity,
            'margin': info.margin,
            'free_margin': info.margin_free,
            'margin_level': info.margin_level,
            'profit': info.profit,
            'currency': info.currency,
            'leverage': info.leverage,
            'name': info.name,
            'company': info.company
        }
    
    def get_balance(self) -> float:
        """Get account balance."""
        info = self.get_account_info()
        return info.get('balance', 0.0)
    
    def get_equity(self) -> float:
        """Get account equity."""
        info = self.get_account_info()
        return info.get('equity', 0.0)
    
    # ==================== Symbol Information ====================
    
    def get_symbol_info(self, symbol: str) -> Optional[SymbolInfo]:
        """Get symbol information."""
        self.ensure_connected()
        
        full_symbol = self.config.get_symbol(symbol)
        
        # Check cache
        if full_symbol in self._symbol_cache:
            return self._symbol_cache[full_symbol]
        
        info = mt5.symbol_info(full_symbol)
        if info is None:
            logger.warning(f"Symbol not found: {full_symbol}")
            return None
        
        symbol_info = SymbolInfo(
            name=info.name,
            digits=info.digits,
            point=info.point,
            trade_tick_size=info.trade_tick_size,
            trade_tick_value=info.trade_tick_value,
            volume_min=info.volume_min,
            volume_max=info.volume_max,
            volume_step=info.volume_step,
            trade_contract_size=info.trade_contract_size,
            spread=info.spread,
            swap_long=info.swap_long,
            swap_short=info.swap_short,
            margin_initial=info.margin_initial,
            currency_base=info.currency_base,
            currency_profit=info.currency_profit,
            description=info.description
        )
        
        # Cache it
        self._symbol_cache[full_symbol] = symbol_info
        return symbol_info
    
    def select_symbol(self, symbol: str) -> bool:
        """Enable symbol in Market Watch."""
        self.ensure_connected()
        full_symbol = self.config.get_symbol(symbol)
        return mt5.symbol_select(full_symbol, True)
    
    def get_tick(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current tick data for symbol."""
        self.ensure_connected()
        full_symbol = self.config.get_symbol(symbol)
        
        tick = mt5.symbol_info_tick(full_symbol)
        if tick is None:
            return None
        
        return {
            'bid': tick.bid,
            'ask': tick.ask,
            'last': tick.last,
            'volume': tick.volume,
            'time': datetime.fromtimestamp(tick.time)
        }
    
    # ==================== Historical Data ====================
    
    def get_candles(
        self,
        symbol: str,
        timeframe: Timeframe,
        count: int = 500,
        start_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get historical OHLC data.
        
        Args:
            symbol: Trading symbol (without suffix)
            timeframe: Timeframe enum
            count: Number of candles to retrieve
            start_time: Start time for data (None = from current time backwards)
            
        Returns:
            DataFrame with OHLC data
        """
        self.ensure_connected()
        full_symbol = self.config.get_symbol(symbol)
        
        # Ensure symbol is selected
        self.select_symbol(symbol)
        
        if start_time:
            rates = mt5.copy_rates_from(
                full_symbol,
                timeframe.value,
                start_time,
                count
            )
        else:
            rates = mt5.copy_rates_from_pos(
                full_symbol,
                timeframe.value,
                0,  # Start from current bar
                count
            )
        
        if rates is None or len(rates) == 0:
            logger.warning(f"No data received for {full_symbol}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df.rename(columns={
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'tick_volume': 'volume',
            'spread': 'spread',
            'real_volume': 'real_volume'
        }, inplace=True)
        
        return df[['open', 'high', 'low', 'close', 'volume', 'spread']]
    
    def get_candles_range(
        self,
        symbol: str,
        timeframe: Timeframe,
        start_time: datetime,
        end_time: datetime
    ) -> pd.DataFrame:
        """
        Get historical data for a specific date range.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe enum
            start_time: Start datetime
            end_time: End datetime
            
        Returns:
            DataFrame with OHLC data
        """
        self.ensure_connected()
        full_symbol = self.config.get_symbol(symbol)
        
        self.select_symbol(symbol)
        
        rates = mt5.copy_rates_range(
            full_symbol,
            timeframe.value,
            start_time,
            end_time
        )
        
        if rates is None or len(rates) == 0:
            logger.warning(f"No data for {full_symbol} in range")
            return pd.DataFrame()
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        return df[['open', 'high', 'low', 'close', 'tick_volume', 'spread']]
    
    def get_close_prices(
        self,
        symbol: str,
        timeframe: Timeframe,
        count: int = 500
    ) -> pd.Series:
        """Get close prices as Series."""
        df = self.get_candles(symbol, timeframe, count)
        if df.empty:
            return pd.Series(dtype=float)
        return df['close']
    
    # ==================== Order Execution ====================
    
    def market_order(
        self,
        symbol: str,
        order_type: OrderType,
        volume: float,
        sl: float = 0.0,
        tp: float = 0.0,
        comment: str = ""
    ) -> OrderResult:
        """
        Execute a market order.
        
        Args:
            symbol: Trading symbol
            order_type: BUY or SELL
            volume: Lot size
            sl: Stop loss price (0 = no SL)
            tp: Take profit price (0 = no TP)
            comment: Order comment
            
        Returns:
            OrderResult with execution details
        """
        self.ensure_connected()
        full_symbol = self.config.get_symbol(symbol)
        
        # Get current price
        tick = mt5.symbol_info_tick(full_symbol)
        if tick is None:
            return OrderResult(
                success=False,
                ticket=0,
                symbol=symbol,
                volume=volume,
                price=0,
                order_type=order_type.name,
                comment=comment,
                error_code=-1,
                error_message="Failed to get tick data"
            )
        
        price = tick.ask if order_type == OrderType.BUY else tick.bid
        
        # Build request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": full_symbol,
            "volume": volume,
            "type": order_type.value,
            "price": price,
            "deviation": self.config.deviation,
            "magic": self.config.magic_number,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        if sl > 0:
            request["sl"] = sl
        if tp > 0:
            request["tp"] = tp
        
        # Execute order
        result = mt5.order_send(request)
        
        if result is None:
            error = mt5.last_error()
            return OrderResult(
                success=False,
                ticket=0,
                symbol=symbol,
                volume=volume,
                price=price,
                order_type=order_type.name,
                comment=comment,
                error_code=error[0],
                error_message=error[1]
            )
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return OrderResult(
                success=False,
                ticket=result.order,
                symbol=symbol,
                volume=volume,
                price=result.price,
                order_type=order_type.name,
                comment=comment,
                error_code=result.retcode,
                error_message=result.comment
            )
        
        logger.info(f"Order executed: {order_type.name} {volume} {symbol} @ {result.price}")
        
        return OrderResult(
            success=True,
            ticket=result.order,
            symbol=symbol,
            volume=result.volume,
            price=result.price,
            order_type=order_type.name,
            comment=comment
        )
    
    def close_position(
        self,
        ticket: int,
        volume: Optional[float] = None,
        comment: str = "Close position"
    ) -> OrderResult:
        """
        Close an open position.
        
        Args:
            ticket: Position ticket
            volume: Volume to close (None = full position)
            comment: Order comment
            
        Returns:
            OrderResult with execution details
        """
        self.ensure_connected()
        
        # Get position info
        position = mt5.positions_get(ticket=ticket)
        if not position:
            return OrderResult(
                success=False,
                ticket=ticket,
                symbol="",
                volume=0,
                price=0,
                order_type="CLOSE",
                comment=comment,
                error_code=-1,
                error_message="Position not found"
            )
        
        pos = position[0]
        symbol = pos.symbol
        pos_volume = volume if volume else pos.volume
        
        # Determine close direction
        close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        
        # Get current price
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return OrderResult(
                success=False,
                ticket=ticket,
                symbol=symbol,
                volume=pos_volume,
                price=0,
                order_type="CLOSE",
                comment=comment,
                error_code=-1,
                error_message="Failed to get tick data"
            )
        
        price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
        
        # Build close request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": pos_volume,
            "type": close_type,
            "position": ticket,
            "price": price,
            "deviation": self.config.deviation,
            "magic": self.config.magic_number,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            error = mt5.last_error() if result is None else (result.retcode, result.comment)
            return OrderResult(
                success=False,
                ticket=ticket,
                symbol=symbol,
                volume=pos_volume,
                price=price,
                order_type="CLOSE",
                comment=comment,
                error_code=error[0],
                error_message=str(error[1])
            )
        
        logger.info(f"Position closed: {ticket} @ {result.price}")
        
        return OrderResult(
            success=True,
            ticket=result.order,
            symbol=symbol,
            volume=result.volume,
            price=result.price,
            order_type="CLOSE",
            comment=comment
        )
    
    def close_all_positions(self, symbol: Optional[str] = None) -> List[OrderResult]:
        """Close all positions, optionally filtered by symbol."""
        self.ensure_connected()
        
        if symbol:
            full_symbol = self.config.get_symbol(symbol)
            positions = mt5.positions_get(symbol=full_symbol)
        else:
            positions = mt5.positions_get()
        
        results = []
        if positions:
            for pos in positions:
                if pos.magic == self.config.magic_number:
                    result = self.close_position(pos.ticket)
                    results.append(result)
        
        return results
    
    # ==================== Position Management ====================
    
    def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """Get open positions."""
        self.ensure_connected()
        
        if symbol:
            full_symbol = self.config.get_symbol(symbol)
            positions = mt5.positions_get(symbol=full_symbol)
        else:
            positions = mt5.positions_get()
        
        if not positions:
            return []
        
        result = []
        for pos in positions:
            # Filter by magic number
            if pos.magic != self.config.magic_number:
                continue
            
            result.append(Position(
                ticket=pos.ticket,
                symbol=pos.symbol,
                volume=pos.volume,
                price_open=pos.price_open,
                price_current=pos.price_current,
                profit=pos.profit,
                swap=pos.swap,
                type='buy' if pos.type == mt5.ORDER_TYPE_BUY else 'sell',
                magic=pos.magic,
                comment=pos.comment,
                time=datetime.fromtimestamp(pos.time)
            ))
        
        return result
    
    def get_position_by_ticket(self, ticket: int) -> Optional[Position]:
        """Get specific position by ticket."""
        positions = self.get_positions()
        for pos in positions:
            if pos.ticket == ticket:
                return pos
        return None
    
    def get_total_profit(self) -> float:
        """Get total profit/loss of all open positions."""
        positions = self.get_positions()
        return sum(pos.profit for pos in positions)
    
    # ==================== Order History ====================
    
    def get_history_orders(
        self,
        start_time: datetime,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get historical orders."""
        self.ensure_connected()
        
        if end_time is None:
            end_time = datetime.now()
        
        orders = mt5.history_orders_get(start_time, end_time)
        
        if orders is None:
            return []
        
        return [
            {
                'ticket': order.ticket,
                'symbol': order.symbol,
                'type': 'buy' if order.type == mt5.ORDER_TYPE_BUY else 'sell',
                'volume': order.volume_current,
                'price': order.price_current,
                'state': order.state,
                'time': datetime.fromtimestamp(order.time_setup),
                'comment': order.comment
            }
            for order in orders
            if order.magic == self.config.magic_number
        ]
    
    def get_history_deals(
        self,
        start_time: datetime,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Get historical deals as DataFrame."""
        self.ensure_connected()
        
        if end_time is None:
            end_time = datetime.now()
        
        deals = mt5.history_deals_get(start_time, end_time)
        
        if deals is None or len(deals) == 0:
            return pd.DataFrame()
        
        data = []
        for deal in deals:
            if deal.magic == self.config.magic_number:
                data.append({
                    'ticket': deal.ticket,
                    'order': deal.order,
                    'symbol': deal.symbol,
                    'type': 'buy' if deal.type == mt5.DEAL_TYPE_BUY else 'sell',
                    'volume': deal.volume,
                    'price': deal.price,
                    'profit': deal.profit,
                    'swap': deal.swap,
                    'commission': deal.commission,
                    'time': datetime.fromtimestamp(deal.time),
                    'comment': deal.comment
                })
        
        return pd.DataFrame(data)
    
    # ==================== Utility Methods ====================
    
    def calculate_lot_size(
        self,
        symbol: str,
        risk_amount: float,
        stop_loss_pips: float
    ) -> float:
        """
        Calculate lot size based on risk.
        
        Args:
            symbol: Trading symbol
            risk_amount: Amount to risk in account currency
            stop_loss_pips: Stop loss distance in pips
            
        Returns:
            Calculated lot size (rounded to volume step)
        """
        info = self.get_symbol_info(symbol)
        if info is None:
            return 0.0
        
        # Pip value calculation
        pip_value = info.trade_tick_value * (info.trade_tick_size / info.point)
        
        if stop_loss_pips <= 0:
            return info.volume_min
        
        # Calculate lot size
        lot_size = risk_amount / (stop_loss_pips * pip_value)
        
        # Round to volume step
        lot_size = round(lot_size / info.volume_step) * info.volume_step
        
        # Clamp to min/max
        lot_size = max(info.volume_min, min(lot_size, info.volume_max))
        
        return round(lot_size, 2)
    
    def get_pip_value(self, symbol: str, volume: float = 1.0) -> float:
        """Get pip value in account currency for given volume."""
        info = self.get_symbol_info(symbol)
        if info is None:
            return 0.0
        
        return info.trade_tick_value * (info.trade_tick_size / info.point) * volume
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
