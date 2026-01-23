"""
Single-Strategy Trading System Orchestrator

Production trading system with:
- Trend Continuation strategy
- Market Regime Filter (MRF) - BEFORE signal generation
- Structural Gatekeeper - AFTER signal generation
- Risk engine with drawdown governors

FILTER ORDER:
1. MRF evaluates market regime
2. If MRF blocks → skip signal generation entirely
3. Signal engine generates signal
4. Gatekeeper evaluates structural conditions
5. Risk engine validates position limits
6. Execution engine places order

SAFETY: System defaults to DRY RUN mode.
"""

import time
import signal as os_signal
import sys
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple
from pathlib import Path

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

from .config import SystemConfig, DEFAULT_CONFIG
from .signal_engine import SignalEngine, TradeSignal
from .market_regime_filter import MarketRegimeFilter, MRFDecision
from .gatekeeper_engine import GatekeeperEngine, GatekeeperDecision
from .risk_engine import RiskEngine, TradePermission
from .execution_engine import ExecutionEngine, OrderResult
from .logging_module import (
    setup_logging,
    TradeLogger,
    BlockLogger,
    RiskLogger,
)


class TradingSystem:
    """
    Single-strategy trading system orchestrator.
    
    Coordinates:
    - Market Regime Filter (pre-signal)
    - Signal Engine
    - Gatekeeper (post-signal)
    - Risk Engine
    - Execution Engine
    """
    
    def __init__(self, config: SystemConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.config.ensure_directories()
        
        # Logging
        self.logger = setup_logging(self.config.paths, self.config.verbose)
        
        # Components
        self.mrf = MarketRegimeFilter()
        self.signal_engine = SignalEngine(self.config.strategy)
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
        self._mrf_blocks = 0
        self._gate_blocks = 0
        self._trades_executed = 0
        
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self) -> None:
        """Setup graceful shutdown handlers."""
        def shutdown_handler(signum, frame):
            self.logger.info("Shutdown signal received...")
            self._running = False
        
        os_signal.signal(os_signal.SIGINT, shutdown_handler)
        os_signal.signal(os_signal.SIGTERM, shutdown_handler)
    
    def initialize(self) -> bool:
        """Initialize the trading system."""
        self.logger.info("=" * 60)
        self.logger.info("SINGLE-STRATEGY TRADING SYSTEM")
        self.logger.info("=" * 60)
        
        # Connect to MT5
        if not self.execution.connect():
            self.logger.error("Failed to connect to MT5")
            return False
        
        # Get initial equity
        equity = self.execution.get_account_equity()
        if equity is None:
            self.logger.error("Failed to get account equity")
            return False
        
        self.logger.info(f"Account equity: ${equity:,.2f}")
        
        # Initialize risk engine
        self.risk_engine.initialize(equity)
        
        # Log initial state
        self.risk_logger.log_state(self.risk_engine.state)
        
        # Check if system should be halted
        if self.risk_engine.state.is_halted:
            self.logger.warning(f"System is HALTED: {self.risk_engine.state.halt_reason}")
            return False
        
        self.logger.info(f"Risk level: {self.risk_engine.state.risk_level.value}")
        self.logger.info(f"Dry run mode: {self.config.dry_run}")
        self.logger.info("Components: MRF + Signal + Gatekeeper + Risk")
        self.logger.info("Initialization complete")
        
        return True
    
    def run(self) -> None:
        """Main trading loop."""
        if not self.initialize():
            self.logger.error("Initialization failed - aborting")
            return
        
        self.logger.info("Starting main loop...")
        self._running = True
        
        while self._running:
            try:
                self._process_cycle()
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.exception(f"Error in main loop: {e}")
                time.sleep(60)
        
        self._shutdown()
    
    def _process_cycle(self) -> None:
        """Process one cycle of the trading loop."""
        # Check for new bar
        new_bar = self._check_new_bar()
        if not new_bar:
            return
        
        self._bar_count += 1
        
        # Update equity
        equity = self.execution.get_account_equity()
        if equity is not None:
            self.risk_engine.update_equity(equity)
        
        # Check for closed positions
        self._check_closed_positions()
        
        # Check if halted
        if self.risk_engine.state.is_halted:
            self.logger.warning("System halted - skipping signal processing")
            return
        
        # Get bar data
        bar_data = self._get_current_bars()
        if bar_data is None:
            return
        
        eurusd_bar, gbpusd_bar, timestamp = bar_data
        
        # Step 1: Update and check MRF FIRST
        self.mrf.update(
            timestamp=timestamp,
            high=eurusd_bar['high'],
            low=eurusd_bar['low'],
            close=eurusd_bar['close'],
        )
        
        mrf_decision = self.mrf.evaluate()
        
        if not mrf_decision.allowed:
            self._mrf_blocks += 1
            self.logger.debug(
                f"MRF BLOCK: {[r.value for r in mrf_decision.reasons]} "
                f"ADX={mrf_decision.adx:.1f}, ATR_ratio={mrf_decision.atr_ratio:.2f}"
            )
            # Still update other components for state consistency
            self.gatekeeper.update(eurusd_bar['close'], gbpusd_bar['close'])
            self.signal_engine.update(
                timestamp=timestamp,
                open_=eurusd_bar['open'],
                high=eurusd_bar['high'],
                low=eurusd_bar['low'],
                close=eurusd_bar['close'],
            )
            return
        
        # Step 2: Update gatekeeper
        self.gatekeeper.update(eurusd_bar['close'], gbpusd_bar['close'])
        
        # Step 3: Generate signal
        signal = self.signal_engine.update(
            timestamp=timestamp,
            open_=eurusd_bar['open'],
            high=eurusd_bar['high'],
            low=eurusd_bar['low'],
            close=eurusd_bar['close'],
        )
        
        # Step 4: Process signal if generated
        if signal is not None:
            self._process_signal(signal, mrf_decision)
        
        # Log risk state periodically
        if self._bar_count % 6 == 0:  # Every ~24 hours for H4
            self.risk_logger.log_state(self.risk_engine.state)
    
    def _check_new_bar(self) -> bool:
        """Check if a new bar has formed."""
        if not MT5_AVAILABLE or self.config.dry_run:
            return True
        
        rates = mt5.copy_rates_from_pos(
            self.config.strategy.primary_symbol,
            mt5.TIMEFRAME_H4,
            0,
            1
        )
        
        if rates is None or len(rates) == 0:
            return False
        
        bar_time = datetime.fromtimestamp(rates[0]['time'], tz=timezone.utc)
        
        if self._last_bar_time is None or bar_time > self._last_bar_time:
            self._last_bar_time = bar_time
            return True
        
        return False
    
    def _get_current_bars(self) -> Optional[Tuple[dict, dict, datetime]]:
        """Get current bar data for both symbols."""
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
            mt5.TIMEFRAME_H4,
            0, 1
        )
        
        gb_rates = mt5.copy_rates_from_pos(
            self.config.strategy.secondary_symbol,
            mt5.TIMEFRAME_H4,
            0, 1
        )
        
        if eu_rates is None or gb_rates is None:
            return None
        
        if len(eu_rates) == 0 or len(gb_rates) == 0:
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
    
    def _process_signal(self, signal: TradeSignal, mrf_decision: MRFDecision) -> None:
        """Process a generated trade signal."""
        symbol = self.config.strategy.primary_symbol
        
        self.logger.info(f"Signal: {signal.direction.value} @ {signal.entry_price:.5f}")
        self.trade_logger.log_signal(signal, symbol)
        
        # Step 1: Check risk engine permission
        risk_permission = self.risk_engine.can_trade(self._bar_count)
        
        if risk_permission != TradePermission.ALLOWED:
            self.logger.info(f"Risk engine blocked: {risk_permission.value}")
            return
        
        # Step 2: Check gatekeeper permission
        gate_decision = self.gatekeeper.evaluate()
        self.block_logger.log_decision(gate_decision, signal)
        
        if not gate_decision.allowed:
            self._gate_blocks += 1
            self.logger.info(f"Gatekeeper blocked: {[r.value for r in gate_decision.reasons]}")
            return
        
        # Step 3: Calculate position size
        position_size = self.risk_engine.calculate_position_size(
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            symbol=symbol,
        )
        
        if position_size < self.config.risk.min_position_size:
            self.logger.warning(f"Position size too small: {position_size}")
            return
        
        # Step 4: Execute trade
        self.logger.info(f"Executing: {signal.direction.value} {position_size:.2f} lots")
        
        execution = self.execution.execute_signal(signal, position_size, symbol)
        self.trade_logger.log_execution(execution)
        
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
    
    def _check_closed_positions(self) -> None:
        """Check for positions that have been closed."""
        if self.config.dry_run:
            return
        
        current_positions = self.execution.get_open_positions()
        current_tickets = {p.ticket for p in current_positions}
        
        for pos in list(self.risk_engine.state.open_positions):
            if pos.ticket not in current_tickets:
                self.logger.info(f"Position {pos.ticket} closed externally")
                self.risk_engine.register_trade_closed(
                    ticket=pos.ticket,
                    exit_price=0,
                    pnl=0,
                )
    
    def _shutdown(self) -> None:
        """Graceful system shutdown."""
        self.logger.info("=" * 60)
        self.logger.info("SYSTEM SHUTDOWN")
        self.logger.info("=" * 60)
        
        # Log statistics
        self.logger.info(f"Bars processed: {self._bar_count}")
        self.logger.info(f"MRF blocks: {self._mrf_blocks}")
        self.logger.info(f"Gatekeeper blocks: {self._gate_blocks}")
        self.logger.info(f"Trades executed: {self._trades_executed}")
        
        # Log final risk state
        if self.risk_engine.state is not None:
            self.risk_logger.log_state(self.risk_engine.state)
            self.logger.info(f"Final equity: ${self.risk_engine.state.current_equity:,.2f}")
            self.logger.info(f"Open positions: {len(self.risk_engine.state.open_positions)}")
        
        self.execution.disconnect()
        self.logger.info("Shutdown complete")
    
    def get_status(self) -> dict:
        """Get current system status."""
        return {
            "running": self._running,
            "bar_count": self._bar_count,
            "mrf_blocks": self._mrf_blocks,
            "gate_blocks": self._gate_blocks,
            "trades_executed": self._trades_executed,
            "mrf_ready": self.mrf.is_ready,
            "mrf_adx": self.mrf.current_adx,
            "mrf_atr_ratio": self.mrf.current_atr_ratio,
            "signal_engine_ready": self.signal_engine.is_ready,
            "gatekeeper_ready": self.gatekeeper.is_ready,
            "risk_status": self.risk_engine.get_status_summary(),
            "connected": self.execution.is_connected,
            "dry_run": self.config.dry_run,
        }
    
    def emergency_close_all(self) -> None:
        """Emergency: Close all open positions."""
        self.logger.critical("EMERGENCY CLOSE ALL POSITIONS")
        
        positions = self.execution.get_open_positions()
        
        for pos in positions:
            result = self.execution.close_position(pos.ticket)
            if result.is_success:
                self.logger.info(f"Closed {pos.ticket}")
            else:
                self.logger.error(f"Failed to close {pos.ticket}: {result.error_message}")


def main():
    """Entry point for trading system."""
    print("=" * 60)
    print("SINGLE-STRATEGY TRADING SYSTEM")
    print("Filters: MRF + Gatekeeper")
    print("=" * 60)
    print()
    
    config = SystemConfig(dry_run=True)  # SAFETY: Default to dry run
    
    system = TradingSystem(config)
    
    try:
        system.run()
    except KeyboardInterrupt:
        print("\nShutdown requested...")
    except Exception as e:
        print(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()
