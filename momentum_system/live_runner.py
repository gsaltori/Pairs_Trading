"""
Cross-Sectional Momentum System - Live Runner
Paper and live execution with monthly rebalancing.

Usage:
    python live_runner.py --paper    # Paper trading (default)
    python live_runner.py --live     # Live trading (requires confirmation)
    python live_runner.py --status   # Show current status
    python live_runner.py --signal   # Generate today's signal
"""

import argparse
import sys
import json
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Optional
import calendar

from config import CONFIG, EXEC_CONFIG, print_config
from data_loader import DataLoader
from momentum_engine import MomentumEngine, MomentumSignal
from execution_engine import ExecutionEngine, PaperExecution, OrderRequest


class LiveRunner:
    """
    Live/paper trading runner for momentum system.
    
    Handles:
    - Monthly rebalance scheduling
    - Signal generation
    - Order execution
    - State persistence
    """
    
    STATE_FILE = Path("live_state.json")
    
    def __init__(self, paper_mode: bool = True):
        """Initialize live runner."""
        self.paper_mode = paper_mode
        
        self.data_loader = DataLoader()
        self.momentum_engine = MomentumEngine()
        
        # Execution
        if paper_mode:
            self.execution = PaperExecution(CONFIG.INITIAL_CAPITAL)
        else:
            raise NotImplementedError("Live execution not implemented")
        
        self.engine = ExecutionEngine(self.execution, dry_run=False)
        
        # State
        self._last_rebalance: Optional[date] = None
        self._load_state()
    
    def _load_state(self):
        """Load persisted state."""
        if self.STATE_FILE.exists():
            try:
                with open(self.STATE_FILE, 'r') as f:
                    state = json.load(f)
                
                if state.get('last_rebalance'):
                    self._last_rebalance = date.fromisoformat(state['last_rebalance'])
                
                # Restore positions for paper trading
                if self.paper_mode and 'positions' in state:
                    self.execution.positions = state['positions']
                    self.execution.cash = state.get('cash', CONFIG.INITIAL_CAPITAL)
                
                print(f"Loaded state from {self.STATE_FILE}")
                
            except Exception as e:
                print(f"Warning: Could not load state: {e}")
    
    def _save_state(self):
        """Save state to file."""
        state = {
            'last_rebalance': self._last_rebalance.isoformat() if self._last_rebalance else None,
            'positions': self.execution.positions if self.paper_mode else {},
            'cash': self.execution.cash if self.paper_mode else 0,
            'timestamp': datetime.now().isoformat(),
        }
        
        with open(self.STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
    
    def is_last_trading_day_of_month(self, check_date: date = None) -> bool:
        """Check if date is the last trading day of the month."""
        if check_date is None:
            check_date = date.today()
        
        # Get last day of month
        _, last_day = calendar.monthrange(check_date.year, check_date.month)
        month_end = date(check_date.year, check_date.month, last_day)
        
        # If last day is weekend, find Friday
        while month_end.weekday() > 4:  # Saturday = 5, Sunday = 6
            month_end = month_end.replace(day=month_end.day - 1)
        
        return check_date == month_end
    
    def should_rebalance(self) -> bool:
        """Check if we should rebalance today."""
        today = date.today()
        
        # Already rebalanced this month?
        if self._last_rebalance:
            if (self._last_rebalance.year == today.year and 
                self._last_rebalance.month == today.month):
                return False
        
        # Is it the last trading day?
        return self.is_last_trading_day_of_month(today)
    
    def generate_signal(self) -> Optional[MomentumSignal]:
        """Generate momentum signal for today."""
        print("\nGenerating momentum signal...")
        
        # Load recent data
        prices = self.data_loader.load_universe()
        
        if len(prices) == 0:
            print("ERROR: No price data available")
            return None
        
        # Get today's date (or last available)
        latest_date = prices.index[-1]
        print(f"Latest data date: {latest_date.date()}")
        
        # Generate signal
        rebalance_dates = [latest_date]
        signals = self.momentum_engine.generate_all_signals(prices, rebalance_dates)
        
        if not signals:
            print("No valid signal generated (insufficient history)")
            return None
        
        return signals[0]
    
    def execute_rebalance(self, signal: MomentumSignal) -> bool:
        """Execute rebalance based on signal."""
        print(f"\nExecuting rebalance...")
        print(f"  Date: {signal.date.date()}")
        print(f"  Selected: {signal.selected}")
        print(f"  Cash weight: {signal.cash_weight:.1%}")
        
        # Get current prices
        prices = self.data_loader.load_universe()
        current_prices = prices.iloc[-1].to_dict()
        
        # Update execution interface prices
        self.execution.update_prices(current_prices)
        
        # Connect
        self.engine.connect()
        
        # Execute
        fills = self.engine.execute_rebalance(signal.weights, current_prices)
        
        print(f"\nExecuted {len(fills)} orders:")
        for fill in fills:
            print(f"  {fill.side} {fill.quantity:.2f} {fill.symbol} @ ${fill.price:.2f}")
        
        # Update state
        self._last_rebalance = date.today()
        self._save_state()
        
        return True
    
    def show_status(self):
        """Show current system status."""
        print("\n" + "=" * 60)
        print("MOMENTUM SYSTEM STATUS")
        print("=" * 60)
        
        print(f"\nMode: {'PAPER' if self.paper_mode else 'LIVE'}")
        print(f"Last rebalance: {self._last_rebalance or 'Never'}")
        print(f"Should rebalance today: {self.should_rebalance()}")
        
        # Load state
        if self.paper_mode:
            equity = self.execution.get_equity()
            print(f"\nPortfolio:")
            print(f"  Cash: ${self.execution.cash:,.2f}")
            print(f"  Equity: ${equity:,.2f}")
            print(f"  Positions: {len(self.execution.positions)}")
            
            for symbol, shares in self.execution.positions.items():
                price = self.execution.get_price(symbol)
                value = shares * price if price else 0
                print(f"    {symbol}: {shares:.2f} shares (${value:,.2f})")
        
        print("=" * 60)
    
    def show_signal(self):
        """Generate and display today's signal."""
        signal = self.generate_signal()
        
        if signal is None:
            return
        
        print("\n" + "=" * 60)
        print("MOMENTUM SIGNAL")
        print("=" * 60)
        
        print(f"\nDate: {signal.date.date()}")
        
        print("\nMomentum Rankings:")
        for i, (symbol, mom) in enumerate(signal.rankings.items(), 1):
            filter_status = "✓" if signal.trend_filter.get(symbol, False) else "✗"
            selected_marker = " ★" if symbol in signal.selected else ""
            print(f"  {i:2}. {symbol}: {mom:>7.1%} [{filter_status}]{selected_marker}")
        
        print(f"\nSelected Assets: {', '.join(signal.selected) if signal.selected else 'NONE'}")
        
        print("\nTarget Weights:")
        for symbol, weight in signal.weights.items():
            print(f"  {symbol}: {weight:.1%}")
        print(f"  CASH: {signal.cash_weight:.1%}")
        
        print("=" * 60)
    
    def run_rebalance_check(self):
        """Check and execute rebalance if needed."""
        if not self.should_rebalance():
            print("\nNo rebalance needed today.")
            self.show_status()
            return
        
        print("\n⚡ REBALANCE DAY ⚡")
        
        signal = self.generate_signal()
        
        if signal is None:
            print("Could not generate signal.")
            return
        
        self.show_signal()
        
        if self.paper_mode:
            # Execute automatically in paper mode
            self.execute_rebalance(signal)
        else:
            # Require confirmation in live mode
            print("\n⚠️ LIVE MODE - Confirm execution?")
            confirm = input("Type 'EXECUTE' to proceed: ")
            
            if confirm == "EXECUTE":
                self.execute_rebalance(signal)
            else:
                print("Execution cancelled.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Cross-Sectional Momentum Live Trading System"
    )
    parser.add_argument(
        '--paper',
        action='store_true',
        default=True,
        help='Run in paper trading mode (default)'
    )
    parser.add_argument(
        '--live',
        action='store_true',
        help='Run in live trading mode'
    )
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show current status'
    )
    parser.add_argument(
        '--signal',
        action='store_true',
        help='Generate and show today\'s signal'
    )
    parser.add_argument(
        '--rebalance',
        action='store_true',
        help='Force rebalance check and execution'
    )
    
    args = parser.parse_args()
    
    paper_mode = not args.live
    
    if args.live:
        print("\n⚠️ LIVE MODE NOT IMPLEMENTED")
        print("Please implement a real broker connection first.")
        return 1
    
    runner = LiveRunner(paper_mode=paper_mode)
    
    if args.status:
        # Update prices first
        prices = runner.data_loader.load_universe()
        runner.execution.update_prices(prices.iloc[-1].to_dict())
        runner.show_status()
        return 0
    
    if args.signal:
        runner.show_signal()
        return 0
    
    if args.rebalance:
        signal = runner.generate_signal()
        if signal:
            runner.execute_rebalance(signal)
        return 0
    
    # Default: check and execute if rebalance day
    print("\n" + "=" * 60)
    print("CROSS-SECTIONAL MOMENTUM SYSTEM")
    print("=" * 60)
    print_config()
    
    runner.run_rebalance_check()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
