"""
Trend Following System - Live Trading Runner
Paper and live execution with safety controls.

Usage:
    python run_live.py --paper    # Paper trading mode
    python run_live.py --live     # Live trading (requires confirmation)
    python run_live.py --status   # Check system status

IMPORTANT: Always run paper trading first to validate behavior.
"""

import argparse
import sys
import time
import schedule
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Dict, Optional
import json

from config import TRADING_CONFIG, EXECUTION_CONFIG, print_config
from data_loader import DataLoader
from indicators import IndicatorEngine
from signal_engine import SignalEngine, Signal, SignalType
from position_sizer import PositionSizer
from execution_engine import (
    ExecutionEngine, 
    PaperBroker, 
    OrderSide, 
    OrderType,
)
from logger import setup_logger, get_logger


class LiveTradingSystem:
    """
    Live trading system coordinator.
    
    Handles:
    - Daily signal evaluation
    - Order generation and execution
    - Position tracking
    - Risk monitoring
    """
    
    def __init__(
        self,
        broker: 'BrokerInterface',
        paper_mode: bool = True,
    ):
        """Initialize live trading system."""
        self.broker = broker
        self.paper_mode = paper_mode
        
        self.data_loader = DataLoader()
        self.indicator_engine = IndicatorEngine()
        self.signal_engine = SignalEngine()
        self.position_sizer = PositionSizer()
        self.execution = ExecutionEngine(broker)
        self.logger = get_logger()
        
        self._running = False
        self._last_update = None
        
        # State file for persistence
        self.state_file = Path("state.json")
    
    def start(self):
        """Start the trading system."""
        self.logger.info("=" * 60)
        self.logger.info("TREND FOLLOWING SYSTEM - STARTING")
        self.logger.info(f"Mode: {'PAPER' if self.paper_mode else 'LIVE'}")
        self.logger.info("=" * 60)
        
        # Connect to broker
        if not self.execution.connect():
            self.logger.error("Failed to connect to broker")
            return False
        
        # Load saved state
        self._load_state()
        
        # Log initial status
        self._log_status()
        
        self._running = True
        return True
    
    def stop(self):
        """Stop the trading system."""
        self.logger.info("Stopping trading system...")
        self._running = False
        self._save_state()
        self.execution.disconnect()
        self.logger.info("System stopped.")
    
    def run_daily_update(self):
        """
        Run daily trading logic.
        
        Called at market close to:
        1. Update data
        2. Check exit conditions
        3. Generate entry signals
        4. Queue orders for next open
        """
        try:
            self.logger.info("-" * 40)
            self.logger.info("DAILY UPDATE")
            self.logger.info(f"Time: {datetime.now()}")
            self.logger.info("-" * 40)
            
            # Get current data
            data = self._get_current_data()
            
            # Get prices for equity calculation
            prices = self._get_current_prices(data)
            
            # Log portfolio status
            equity = self.execution.get_equity()
            positions = self.execution.get_positions()
            
            self.logger.log_portfolio_update(
                equity=equity,
                cash=self.execution.get_cash(),
                positions=len(positions),
            )
            
            # Check exits for current positions
            self._check_exits(data, prices)
            
            # Check for new entries
            self._check_entries(data, prices)
            
            self._last_update = datetime.now()
            self._save_state()
            
        except Exception as e:
            self.logger.log_error(e, "Daily update")
    
    def _get_current_data(self) -> Dict[str, 'pd.DataFrame']:
        """Load and process current market data."""
        # Load recent data (enough for indicators)
        data = self.data_loader.load_universe()
        
        # Calculate indicators
        data = self.indicator_engine.calculate_universe(data)
        
        return data
    
    def _get_current_prices(self, data: Dict[str, 'pd.DataFrame']) -> Dict[str, float]:
        """Get current close prices."""
        prices = {}
        for symbol, df in data.items():
            if len(df) > 0:
                prices[symbol] = df['Close'].iloc[-1]
        return prices
    
    def _check_exits(self, data: Dict[str, 'pd.DataFrame'], prices: Dict[str, float]):
        """Check and execute exit signals."""
        positions = self.execution.get_positions()
        
        for symbol, shares in positions.items():
            if shares <= 0:
                continue
            
            if symbol not in data:
                continue
            
            df = data[symbol]
            if len(df) == 0:
                continue
            
            row = df.iloc[-1]
            date = df.index[-1]
            
            # Check exit signal
            signal = self.signal_engine.check_exit(date, symbol, row)
            
            if signal:
                self.logger.log_signal(
                    signal_type="EXIT",
                    symbol=symbol,
                    price=row['Close'],
                    stop_price=row['Trailing_Stop'],
                )
                
                # Execute exit at market
                order = self.execution.sell(symbol, shares, OrderType.MARKET)
                
                if order and order.filled_quantity > 0:
                    self.logger.log_fill(
                        symbol=symbol,
                        side="SELL",
                        quantity=order.filled_quantity,
                        price=order.filled_price,
                        commission=shares * TRADING_CONFIG.COMMISSION_PER_SHARE,
                    )
                    
                    self.signal_engine.register_exit(symbol)
    
    def _check_entries(self, data: Dict[str, 'pd.DataFrame'], prices: Dict[str, float]):
        """Check and execute entry signals."""
        # Check position limit
        current_positions = len(self.execution.get_positions())
        if current_positions >= TRADING_CONFIG.MAX_POSITIONS:
            self.logger.debug(f"Position limit reached ({current_positions})")
            return
        
        signals = []
        
        for symbol in TRADING_CONFIG.SYMBOLS:
            if self.execution.has_position(symbol):
                continue
            
            if symbol not in data:
                continue
            
            df = data[symbol]
            if len(df) == 0:
                continue
            
            row = df.iloc[-1]
            date = df.index[-1]
            
            # Check entry signal
            signal = self.signal_engine.check_entry(date, symbol, row)
            
            if signal:
                signals.append(signal)
        
        # Sort by ATR (highest volatility = most potential)
        signals.sort(key=lambda s: s.atr, reverse=True)
        
        # Execute up to position limit
        available_slots = TRADING_CONFIG.MAX_POSITIONS - current_positions
        
        for signal in signals[:available_slots]:
            self._execute_entry(signal, prices)
    
    def _execute_entry(self, signal: Signal, prices: Dict[str, float]):
        """Execute entry signal."""
        self.logger.log_signal(
            signal_type="ENTRY",
            symbol=signal.symbol,
            price=signal.price,
            stop_price=signal.stop_price,
        )
        
        # Calculate position size
        equity = self.execution.get_equity()
        cash = self.execution.get_cash()
        
        try:
            size = self.position_sizer.calculate(
                equity=equity,
                entry_price=signal.price,
                stop_price=signal.stop_price,
                available_capital=cash,
            )
        except ValueError as e:
            self.logger.warning(f"Position sizing failed: {e}")
            return
        
        if size.shares < 1:
            self.logger.warning(f"Position size too small for {signal.symbol}")
            return
        
        self.logger.log_order(
            order_type="MARKET",
            symbol=signal.symbol,
            side="BUY",
            quantity=size.shares,
        )
        
        # Execute
        order = self.execution.buy(
            signal.symbol,
            size.shares,
            OrderType.MARKET,
        )
        
        if order and order.filled_quantity > 0:
            self.logger.log_fill(
                symbol=signal.symbol,
                side="BUY",
                quantity=order.filled_quantity,
                price=order.filled_price,
                commission=size.shares * TRADING_CONFIG.COMMISSION_PER_SHARE,
            )
            
            self.signal_engine.register_entry(
                symbol=signal.symbol,
                entry_price=order.filled_price,
                entry_date=datetime.now(),
                shares=order.filled_quantity,
                stop_price=signal.stop_price,
            )
    
    def _log_status(self):
        """Log current system status."""
        equity = self.execution.get_equity()
        cash = self.execution.get_cash()
        positions = self.execution.get_positions()
        
        self.logger.info("\nSystem Status:")
        self.logger.info(f"  Equity: ${equity:,.2f}")
        self.logger.info(f"  Cash: ${cash:,.2f}")
        self.logger.info(f"  Open Positions: {len(positions)}")
        
        for symbol, shares in positions.items():
            self.logger.info(f"    {symbol}: {shares} shares")
    
    def _save_state(self):
        """Save system state to file."""
        state = {
            'last_update': self._last_update.isoformat() if self._last_update else None,
            'positions': self.signal_engine.get_all_positions(),
        }
        
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    def _load_state(self):
        """Load system state from file."""
        if not self.state_file.exists():
            return
        
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            if state.get('last_update'):
                self._last_update = datetime.fromisoformat(state['last_update'])
            
            self.logger.info(f"Loaded state from {self.state_file}")
            
        except Exception as e:
            self.logger.warning(f"Could not load state: {e}")
    
    def get_status(self) -> dict:
        """Get current system status."""
        return {
            'running': self._running,
            'mode': 'PAPER' if self.paper_mode else 'LIVE',
            'last_update': self._last_update.isoformat() if self._last_update else None,
            'equity': self.execution.get_equity(),
            'cash': self.execution.get_cash(),
            'positions': self.execution.get_positions(),
            'position_count': self.execution.position_count(),
        }


def run_paper_trading():
    """Run paper trading mode."""
    logger = setup_logger()
    print_config()
    
    # Create paper broker
    broker = PaperBroker(
        initial_capital=TRADING_CONFIG.INITIAL_CAPITAL,
    )
    
    # Create system
    system = LiveTradingSystem(broker, paper_mode=True)
    
    if not system.start():
        return 1
    
    print("\n" + "=" * 60)
    print("PAPER TRADING MODE")
    print("=" * 60)
    print("\nSystem will run daily updates.")
    print("Press Ctrl+C to stop.\n")
    
    try:
        # Schedule daily update at market close
        schedule.every().day.at("16:00").do(system.run_daily_update)
        
        # Run initial update
        system.run_daily_update()
        
        # Keep running
        while True:
            schedule.run_pending()
            time.sleep(60)
            
    except KeyboardInterrupt:
        print("\nShutdown requested...")
    finally:
        system.stop()
    
    return 0


def run_live_trading():
    """Run live trading mode (requires confirmation)."""
    print("\n" + "=" * 60)
    print("⚠️  LIVE TRADING MODE")
    print("=" * 60)
    print("\nThis will execute REAL trades with REAL money.")
    print("Make sure you have:")
    print("  1. Validated the system with paper trading")
    print("  2. Reviewed the RESULTS.md report")
    print("  3. Connected a real broker implementation")
    print("\n")
    
    confirm = input("Type 'CONFIRM LIVE TRADING' to proceed: ")
    
    if confirm != "CONFIRM LIVE TRADING":
        print("\nLive trading cancelled.")
        return 1
    
    print("\n⚠️  Live trading not implemented yet.")
    print("Please implement a live broker connection first.")
    print("See execution_engine.py for the BrokerInterface.")
    
    return 1


def show_status():
    """Show current system status."""
    state_file = Path("state.json")
    
    print("\n" + "=" * 60)
    print("SYSTEM STATUS")
    print("=" * 60)
    
    if not state_file.exists():
        print("\nNo state file found. System has not been run.")
        return
    
    try:
        with open(state_file, 'r') as f:
            state = json.load(f)
        
        print(f"\nLast Update: {state.get('last_update', 'Never')}")
        print(f"Positions: {state.get('positions', {})}")
        
    except Exception as e:
        print(f"\nError reading state: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Trend Following Trading System"
    )
    parser.add_argument(
        '--paper',
        action='store_true',
        help='Run in paper trading mode'
    )
    parser.add_argument(
        '--live',
        action='store_true',
        help='Run in live trading mode (CAUTION)'
    )
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show current system status'
    )
    
    args = parser.parse_args()
    
    if args.status:
        show_status()
        return 0
    
    if args.live:
        return run_live_trading()
    
    if args.paper:
        return run_paper_trading()
    
    # Default: show help
    parser.print_help()
    print("\n\nQuick start:")
    print("  python run_live.py --paper    # Paper trading")
    print("  python run_live.py --status   # Check status")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
