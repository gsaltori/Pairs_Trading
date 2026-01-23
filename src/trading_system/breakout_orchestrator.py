"""
Breakout Strategy Orchestrator

Production trading system for Range Breakout strategy:
- Volatility compression → expansion breakouts
- Asymmetric payoff (R=2.5)
- Integrated Gatekeeper
- Risk governance

FILTER ORDER:
1. Breakout engine generates signal (compression → breakout)
2. Gatekeeper evaluates structural conditions
3. Risk engine validates position limits
4. Execution engine places order

SAFETY: System defaults to DRY RUN mode.
"""

import time
import signal as os_signal
import sys
from datetime import datetime, timezone
from typing import Optional, Tuple
from pathlib import Path

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

from .config import SystemConfig, DEFAULT_CONFIG
from .breakout_engine import RangeBreakoutEngine, BreakoutSignal
from .gatekeeper_engine import GatekeeperEngine
from .risk_engine import RiskEngine, TradePermission
from .execution_engine import ExecutionEngine, OrderResult
from .signal_engine import TradeSignal, SignalDirection
from .logging_module import (
    setup_logging,
    TradeLogger,
    BlockLogger,
    RiskLogger,
)


class BreakoutTradingSystem:
    """
    Breakout strategy trading system.
    
    Coordinates:
    - Breakout Signal Engine
    - Gatekeeper (structural filter)
    - Risk Engine
    - Execution Engine
    """
    
    def __init__(self, config: SystemConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.config.ensure_directories()
        
        # Logging
        self.logger = setup_logging(self.config.paths, self.config.verbose)
        
        # Components
        self.breakout_engine = RangeBreakoutEngine()
        self.gatekeeper = GatekeeperEngine(self.config.gatekeeper, self.logger)
        self.risk_engine = RiskEngine(self.config, self.logger)
        self.execution = ExecutionEngine(self.config, self.logger)
        
        # CSV loggers
        self.trade_logger = TradeLogger(self.config.paths.trade_log)
        self.block_logger = BlockLogger(self.config.paths.block_log)
        self.risk_logger = RiskLogger(self.config.paths.risk_log)
        
        # State
        self._running = False
        self._bar_count = 0
        self._last_bar_time: Optional[datetime] = None
        
        # Statistics
        self._signals_generated = 0
        self._gate_blocks = 0
        self._trades_executed = 0
        
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self) -> None:
        """Setup graceful shutdown."""
        def handler(signum, frame):
            self.logger.info("Shutdown signal received...")
            self._running = False
        
        os_signal.signal(os_signal.SIGINT, handler)
        os_signal.signal(os_signal.SIGTERM, handler)
    
    def initialize(self) -> bool:
        """Initialize system."""
        self.logger.info("=" * 60)
        self.logger.info("BREAKOUT TRADING SYSTEM")
        self.logger.info("Strategy: Range Compression Breakout (R=2.5)")
        self.logger.info("=" * 60)
        
        if not self.execution.connect():
            self.logger.error("MT5 connection failed")
            return False
        
        equity = self.execution.get_account_equity()
        if equity is None:
            self.logger.error("Failed to get equity")
            return False
        
        self.logger.info(f"Account equity: ${equity:,.2f}")
        
        self.risk_engine.initialize(equity)
        self.risk_logger.log_state(self.risk_engine.state)
        
        if self.risk_engine.state.is_halted:
            self.logger.warning(f"System HALTED: {self.risk_engine.state.halt_reason}")
            return False
        
        self.logger.info(f"Risk level: {self.risk_engine.state.risk_level.value}")
        self.logger.info(f"Dry run: {self.config.dry_run}")
        self.logger.info("Initialization complete")
        
        return True
    
    def run(self) -> None:
        """Main trading loop."""
        if not self.initialize():
            self.logger.error("Init failed - aborting")
            return
        
        self.logger.info("Starting main loop...")
        self._running = True
        
        while self._running:
            try:
                self._process_cycle()
                time.sleep(60)
            except Exception as e:
                self.logger.exception(f"Error: {e}")
                time.sleep(60)
        
        self._shutdown()
    
    def _process_cycle(self) -> None:
        """Process one cycle."""
        if not self._check_new_bar():
            return
        
        self._bar_count += 1
        
        # Update equity
        equity = self.execution.get_account_equity()
        if equity:
            self.risk_engine.update_equity(equity)
        
        # Check closed positions
        self._check_closed_positions()
        
        # Check halted
        if self.risk_engine.state.is_halted:
            self.logger.warning("System halted")
            return
        
        # Get bar data
        bar_data = self._get_current_bars()
        if bar_data is None:
            return
        
        eu_bar, gb_bar, timestamp = bar_data
        
        # Update gatekeeper
        self.gatekeeper.update(eu_bar['close'], gb_bar['close'])
        
        # Generate signal
        signal = self.breakout_engine.update(
            timestamp=timestamp,
            open_=eu_bar['open'],
            high=eu_bar['high'],
            low=eu_bar['low'],
            close=eu_bar['close'],
        )
        
        # Process signal
        if signal is not None:
            self._signals_generated += 1
            self._process_signal(signal)
        
        # Periodic risk logging
        if self._bar_count % 6 == 0:
            self.risk_logger.log_state(self.risk_engine.state)
    
    def _check_new_bar(self) -> bool:
        """Check for new bar."""
        if not MT5_AVAILABLE or self.config.dry_run:
            return True
        
        rates = mt5.copy_rates_from_pos(
            self.config.strategy.primary_symbol,
            mt5.TIMEFRAME_H4,
            0, 1
        )
        
        if rates is None or len(rates) == 0:
            return False
        
        bar_time = datetime.fromtimestamp(rates[0]['time'], tz=timezone.utc)
        
        if self._last_bar_time is None or bar_time > self._last_bar_time:
            self._last_bar_time = bar_time
            return True
        
        return False
    
    def _get_current_bars(self):
        """Get current bar data."""
        if not MT5_AVAILABLE:
            return None
        
        if self.config.dry_run:
            now = datetime.now(timezone.utc)
            return (
                {'open': 1.0850, 'high': 1.0860, 'low': 1.0840, 'close': 1.0855},
                {'open': 1.2650, 'high': 1.2660, 'low': 1.2640, 'close': 1.2655},
                now,
            )
        
        eu_rates = mt5.copy_rates_from_pos(
            self.config.strategy.primary_symbol,
            mt5.TIMEFRAME_H4, 0, 1
        )
        gb_rates = mt5.copy_rates_from_pos(
            self.config.strategy.secondary_symbol,
            mt5.TIMEFRAME_H4, 0, 1
        )
        
        if eu_rates is None or gb_rates is None:
            return None
        
        timestamp = datetime.fromtimestamp(eu_rates[0]['time'], tz=timezone.utc)
        
        return (
            {
                'open': eu_rates[0]['open'],
                'high': eu_rates[0]['high'],
                'low': eu_rates[0]['low'],
                'close': eu_rates[0]['close'],
            },
            {
                'open': gb_rates[0]['open'],
                'high': gb_rates[0]['high'],
                'low': gb_rates[0]['low'],
                'close': gb_rates[0]['close'],
            },
            timestamp,
        )
    
    def _process_signal(self, signal: BreakoutSignal) -> None:
        """Process breakout signal."""
        symbol = self.config.strategy.primary_symbol
        
        self.logger.info(
            f"Breakout Signal: {signal.direction.value} @ {signal.entry_price:.5f}, "
            f"Range: {signal.range_width:.5f}, Compression: {signal.compression_ratio:.2f}"
        )
        
        # Check risk engine
        risk_permission = self.risk_engine.can_trade(self._bar_count)
        if risk_permission != TradePermission.ALLOWED:
            self.logger.info(f"Risk blocked: {risk_permission.value}")
            return
        
        # Check gatekeeper
        gate_decision = self.gatekeeper.evaluate()
        
        if not gate_decision.allowed:
            self._gate_blocks += 1
            self.logger.info(f"Gatekeeper blocked: {[r.value for r in gate_decision.reasons]}")
            # Log as blocked
            self.block_logger.log_decision(gate_decision, self._create_trade_signal(signal))
            return
        
        # Calculate position size
        position_size = self.risk_engine.calculate_position_size(
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            symbol=symbol,
        )
        
        if position_size < self.config.risk.min_position_size:
            self.logger.warning(f"Position too small: {position_size}")
            return
        
        # Execute
        self.logger.info(f"Executing: {signal.direction.value} {position_size:.2f} lots")
        
        exec_signal = self._create_trade_signal(signal)
        execution = self.execution.execute_signal(exec_signal, position_size, symbol)
        
        if execution.is_success:
            self._trades_executed += 1
            self.risk_engine.register_trade_opened(
                ticket=execution.ticket,
                symbol=symbol,
                direction=signal.direction.value,
                entry_price=execution.price,
                stop_loss=signal.stop_loss,
                position_size=position_size,
                current_bar=self._bar_count,
            )
            self.logger.info(f"Trade opened: {execution.ticket}")
        elif execution.result == OrderResult.FAILED_DRY_RUN:
            self._trades_executed += 1
            self.logger.info("Dry run - trade simulated")
        else:
            self.logger.error(f"Trade failed: {execution.error_message}")
    
    def _create_trade_signal(self, breakout_signal: BreakoutSignal) -> TradeSignal:
        """Convert BreakoutSignal to TradeSignal for execution engine."""
        direction = SignalDirection.LONG if breakout_signal.direction.value == "LONG" else SignalDirection.SHORT
        
        return TradeSignal(
            bar_index=breakout_signal.bar_index,
            timestamp=breakout_signal.timestamp,
            direction=direction,
            entry_price=breakout_signal.entry_price,
            stop_loss=breakout_signal.stop_loss,
            take_profit=breakout_signal.take_profit,
            atr_value=breakout_signal.atr_value,
        )
    
    def _check_closed_positions(self) -> None:
        """Check for closed positions."""
        if self.config.dry_run:
            return
        
        current = self.execution.get_open_positions()
        current_tickets = {p.ticket for p in current}
        
        for pos in list(self.risk_engine.state.open_positions):
            if pos.ticket not in current_tickets:
                self.logger.info(f"Position {pos.ticket} closed")
                self.risk_engine.register_trade_closed(pos.ticket, 0, 0)
    
    def _shutdown(self) -> None:
        """Shutdown system."""
        self.logger.info("=" * 60)
        self.logger.info("SYSTEM SHUTDOWN")
        self.logger.info("=" * 60)
        
        self.logger.info(f"Bars processed: {self._bar_count}")
        self.logger.info(f"Signals generated: {self._signals_generated}")
        self.logger.info(f"Gatekeeper blocks: {self._gate_blocks}")
        self.logger.info(f"Trades executed: {self._trades_executed}")
        
        if self.risk_engine.state:
            self.risk_logger.log_state(self.risk_engine.state)
            self.logger.info(f"Final equity: ${self.risk_engine.state.current_equity:.2f}")
        
        self.execution.disconnect()
        self.logger.info("Shutdown complete")
    
    def get_status(self) -> dict:
        """Get system status."""
        return {
            "running": self._running,
            "strategy": "Range Breakout (R=2.5)",
            "bar_count": self._bar_count,
            "signals_generated": self._signals_generated,
            "gate_blocks": self._gate_blocks,
            "trades_executed": self._trades_executed,
            "breakout_ready": self.breakout_engine.is_ready,
            "breakout_atr": self.breakout_engine.current_atr,
            "gatekeeper_ready": self.gatekeeper.is_ready,
            "risk_status": self.risk_engine.get_status_summary(),
            "connected": self.execution.is_connected,
            "dry_run": self.config.dry_run,
        }


def main():
    """Entry point."""
    print("=" * 60)
    print("BREAKOUT TRADING SYSTEM")
    print("Strategy: Range Compression Breakout (R=2.5)")
    print("=" * 60)
    print()
    
    config = SystemConfig(dry_run=True)  # SAFETY: Default dry run
    
    system = BreakoutTradingSystem(config)
    
    try:
        system.run()
    except KeyboardInterrupt:
        print("\nShutdown requested...")
    except Exception as e:
        print(f"Fatal: {e}")
        raise


if __name__ == "__main__":
    main()
