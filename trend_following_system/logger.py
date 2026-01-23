"""
Trend Following System - Logger
Comprehensive audit trail and logging.
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import sys

from config import EXECUTION_CONFIG


class TradingLogger:
    """
    Structured logging for trading system.
    
    Provides:
    - Console output for real-time monitoring
    - File logging for audit trail
    - JSON event logging for analysis
    """
    
    def __init__(
        self,
        name: str = "TrendFollowing",
        log_dir: str = None,
        level: str = None,
    ):
        """Initialize logger."""
        if log_dir is None:
            log_dir = EXECUTION_CONFIG.LOGS_DIR
        if level is None:
            level = EXECUTION_CONFIG.LOG_LEVEL
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level))
        
        # Clear existing handlers
        self.logger.handlers = []
        
        # Console handler
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter(EXECUTION_CONFIG.LOG_FORMAT))
        self.logger.addHandler(console)
        
        # File handler
        log_file = self.log_dir / f"trading_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(EXECUTION_CONFIG.LOG_FORMAT))
        self.logger.addHandler(file_handler)
        
        # Event log (JSON)
        self._events_file = self.log_dir / f"events_{datetime.now().strftime('%Y%m%d')}.jsonl"
        self._events = []
    
    def info(self, msg: str, **kwargs):
        """Log info message."""
        self.logger.info(msg)
        if kwargs:
            self._log_event('INFO', msg, kwargs)
    
    def warning(self, msg: str, **kwargs):
        """Log warning message."""
        self.logger.warning(msg)
        self._log_event('WARNING', msg, kwargs)
    
    def error(self, msg: str, **kwargs):
        """Log error message."""
        self.logger.error(msg)
        self._log_event('ERROR', msg, kwargs)
    
    def debug(self, msg: str, **kwargs):
        """Log debug message."""
        self.logger.debug(msg)
    
    def _log_event(self, level: str, msg: str, data: Dict[str, Any]):
        """Log structured event to JSON file."""
        event = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': msg,
            'data': data,
        }
        
        with open(self._events_file, 'a') as f:
            f.write(json.dumps(event) + '\n')
    
    # Trading-specific logging methods
    
    def log_signal(
        self,
        signal_type: str,
        symbol: str,
        price: float,
        stop_price: float = None,
        **kwargs
    ):
        """Log trading signal."""
        msg = f"SIGNAL | {signal_type} | {symbol} | Price: ${price:.2f}"
        if stop_price:
            msg += f" | Stop: ${stop_price:.2f}"
        
        self.info(msg)
        self._log_event('SIGNAL', msg, {
            'signal_type': signal_type,
            'symbol': symbol,
            'price': price,
            'stop_price': stop_price,
            **kwargs
        })
    
    def log_order(
        self,
        order_type: str,
        symbol: str,
        side: str,
        quantity: int,
        price: float = None,
        **kwargs
    ):
        """Log order submission."""
        msg = f"ORDER | {side} {quantity} {symbol}"
        if price:
            msg += f" @ ${price:.2f}"
        
        self.info(msg)
        self._log_event('ORDER', msg, {
            'order_type': order_type,
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            **kwargs
        })
    
    def log_fill(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        commission: float,
        **kwargs
    ):
        """Log order fill."""
        msg = f"FILL | {side} {quantity} {symbol} @ ${price:.2f} | Commission: ${commission:.2f}"
        
        self.info(msg)
        self._log_event('FILL', msg, {
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'commission': commission,
            **kwargs
        })
    
    def log_position_update(
        self,
        symbol: str,
        shares: int,
        avg_price: float,
        unrealized_pnl: float,
        **kwargs
    ):
        """Log position update."""
        msg = f"POSITION | {symbol} | {shares} shares @ ${avg_price:.2f} | P&L: ${unrealized_pnl:.2f}"
        
        self.debug(msg)
        self._log_event('POSITION', msg, {
            'symbol': symbol,
            'shares': shares,
            'avg_price': avg_price,
            'unrealized_pnl': unrealized_pnl,
            **kwargs
        })
    
    def log_portfolio_update(
        self,
        equity: float,
        cash: float,
        positions: int,
        drawdown: float = 0,
        **kwargs
    ):
        """Log portfolio state."""
        msg = f"PORTFOLIO | Equity: ${equity:,.2f} | Cash: ${cash:,.2f} | Positions: {positions}"
        if drawdown > 0:
            msg += f" | DD: {drawdown:.1%}"
        
        self.info(msg)
        self._log_event('PORTFOLIO', msg, {
            'equity': equity,
            'cash': cash,
            'positions': positions,
            'drawdown': drawdown,
            **kwargs
        })
    
    def log_trade_closed(
        self,
        symbol: str,
        pnl: float,
        pnl_pct: float,
        holding_days: int,
        **kwargs
    ):
        """Log completed trade."""
        msg = f"TRADE CLOSED | {symbol} | P&L: ${pnl:.2f} ({pnl_pct:.1%}) | Days: {holding_days}"
        
        self.info(msg)
        self._log_event('TRADE_CLOSED', msg, {
            'symbol': symbol,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'holding_days': holding_days,
            **kwargs
        })
    
    def log_error(self, error: Exception, context: str = ""):
        """Log error with context."""
        msg = f"ERROR | {context} | {type(error).__name__}: {str(error)}"
        self.error(msg)
        self._log_event('ERROR', msg, {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
        })


# Global logger instance
_logger: Optional[TradingLogger] = None


def get_logger() -> TradingLogger:
    """Get or create global logger instance."""
    global _logger
    if _logger is None:
        _logger = TradingLogger()
    return _logger


def setup_logger(
    name: str = "TrendFollowing",
    log_dir: str = None,
    level: str = None,
) -> TradingLogger:
    """Setup and return logger."""
    global _logger
    _logger = TradingLogger(name, log_dir, level)
    return _logger


if __name__ == "__main__":
    # Test logger
    logger = setup_logger()
    
    logger.info("Trading system starting...")
    logger.log_signal("ENTRY", "SPY", 450.0, 430.0)
    logger.log_order("MARKET", "SPY", "BUY", 50, 450.50)
    logger.log_fill("SPY", "BUY", 50, 450.55, 0.50)
    logger.log_position_update("SPY", 50, 450.55, 25.0)
    logger.log_portfolio_update(100_025, 77_472.5, 1, 0.0)
    logger.log_trade_closed("SPY", 500.0, 0.022, 15)
    
    print(f"\nLogs written to: {logger.log_dir}")
