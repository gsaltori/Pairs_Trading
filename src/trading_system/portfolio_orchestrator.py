"""
Multi-Strategy Portfolio Orchestrator

Production trading system for multi-strategy portfolio:
- Trend Continuation
- Trend Pullback
- Volatility Expansion

With shared gatekeeper and portfolio risk management.
"""

import time
import signal as os_signal
import sys
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

from .config import SystemConfig, DEFAULT_CONFIG
from .strategy_router import StrategyRouter, StrategyType, normalize_signal_direction, extract_signal_params
from .portfolio_risk_engine import PortfolioRiskEngine
from .execution_engine import ExecutionEngine, OrderResult
from .logging_module import (
    setup_logging,
    TradeLogger,
    BlockLogger,
    RiskLogger,
)


class PortfolioOrchestrator:
    """
    Multi-strategy portfolio trading system.
    
    Coordinates:
    - Strategy Router (3 strategies + gatekeeper)
    - Portfolio Risk Engine
    - Execution Engine
    - Logging
    """
    
    def __init__(self, config: SystemConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.config.ensure_directories()
        
        # Logging
        self.logger = setup_logging(self.config.paths, self.config.verbose)
        
        # Components
        self.router = StrategyRouter(
            self.config.strategy,
            self.config.gatekeeper,
            self.logger,
        )
        self.risk_engine = PortfolioRiskEngine(self.config, self.logger)
        self.execution = ExecutionEngine(self.config, self.logger)
        
        # CSV loggers
        self.trade_logger = TradeLogger(self.config.paths.trade_log)
        self.block_logger = BlockLogger(self.config.paths.block_log)
        self.risk_logger = RiskLogger(self.config.paths.risk_log)
        
        # State
        self._running = False
        self._bar_count = 0
        self._last_bar_time: Optional[datetime] = None
        
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
        self.logger.info("MULTI-STRATEGY PORTFOLIO INITIALIZATION")
        self.logger.info("=" * 60)
        
        if not self.execution.connect():
            self.logger.error("MT5 connection failed")
            return False
        
        equity = self.execution.get_account_equity()
        if equity is None:
            self.logger.error("Failed to get equity")
            return False
        
        self.logger.info(f"Account equity: ${equity:.2f}")
        
        self.risk_engine.initialize(equity)
        
        if self.risk_engine.state.is_halted:
            self.logger.warning(f"System HALTED: {self.risk_engine.state.halt_reason}")
            return False
        
        self.logger.info(f"Risk level: {self.risk_engine.state.risk_level.value}")
        self.logger.info(f"Risk multiplier: {self.risk_engine.state.risk_multiplier:.1f}x")
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
                time.sleep(60)  # Check every minute
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
        
        # Update router and get decision
        decision = self.router.update(
            timestamp=timestamp,
            eurusd_open=eu_bar['open'],
            eurusd_high=eu_bar['high'],
            eurusd_low=eu_bar['low'],
            eurusd_close=eu_bar['close'],
            gbpusd_close=gb_bar['close'],
        )
        
        # Process decision
        if decision.selected_signal is not None:
            self._process_decision(decision)
        
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
    
    def _process_decision(self, decision) -> None:
        """Process router decision."""
        signal = decision.selected_signal
        strategy = decision.selected_strategy
        symbol = self.config.strategy.primary_symbol
        
        # Log blocked
        if decision.was_blocked:
            self.block_logger.log_decision(decision.gatekeeper_decision, signal)
            return
        
        # Check risk engine
        allowed, reason = self.risk_engine.can_trade(strategy, self._bar_count)
        if not allowed:
            self.logger.info(f"Risk blocked: {reason}")
            return
        
        # Calculate position size
        entry, sl, tp = extract_signal_params(signal)
        position_size = self.risk_engine.calculate_position_size(
            strategy, entry, sl
        )
        
        if position_size < 0.01:
            self.logger.warning("Position too small")
            return
        
        # Create execution signal (adapt to execution engine format)
        from .signal_engine import TradeSignal, SignalDirection
        
        direction = normalize_signal_direction(signal)
        exec_signal = TradeSignal(
            bar_index=decision.bar_index,
            timestamp=decision.timestamp,
            direction=SignalDirection.LONG if direction == "LONG" else SignalDirection.SHORT,
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            atr_value=getattr(signal, 'atr_value', 0),
        )
        
        # Execute
        self.logger.info(f"Executing: {strategy.value} {direction} {position_size:.2f} lots")
        
        execution = self.execution.execute_signal(exec_signal, position_size, symbol)
        self.trade_logger.log_execution(execution)
        
        if execution.is_success or execution.result == OrderResult.FAILED_DRY_RUN:
            self.risk_engine.register_trade_opened(
                ticket=execution.ticket or 0,
                strategy=strategy,
                symbol=symbol,
                direction=direction,
                entry_price=entry,
                stop_loss=sl,
                take_profit=tp,
                position_size=position_size,
                current_bar=self._bar_count,
            )
            self.router.register_position_opened(strategy)
            self.logger.info(f"Trade opened: {strategy.value}")
        else:
            self.logger.error(f"Trade failed: {execution.error_message}")
    
    def _check_closed_positions(self) -> None:
        """Check for closed positions."""
        if self.config.dry_run:
            return
        
        current = self.execution.get_open_positions()
        current_tickets = {p.ticket for p in current}
        
        for strat_name, pos in list(self.risk_engine.state.positions.items()):
            if pos.ticket not in current_tickets:
                strategy = StrategyType(strat_name)
                self.logger.info(f"Position {pos.ticket} closed")
                self.risk_engine.register_trade_closed(strategy, 0, 0)
                self.router.register_position_closed()
    
    def _shutdown(self) -> None:
        """Shutdown system."""
        self.logger.info("=" * 60)
        self.logger.info("SYSTEM SHUTDOWN")
        self.logger.info("=" * 60)
        
        if self.risk_engine.state:
            self.risk_logger.log_state(self.risk_engine.state)
            self.logger.info(f"Final equity: ${self.risk_engine.state.current_equity:.2f}")
        
        self.execution.disconnect()
        self.logger.info("Shutdown complete")
    
    def get_status(self) -> dict:
        """Get system status."""
        return {
            "running": self._running,
            "bar_count": self._bar_count,
            "risk_status": self.risk_engine.get_status_summary(),
            "router_stats": self.router.get_statistics(),
            "connected": self.execution.is_connected,
            "dry_run": self.config.dry_run,
        }


def main():
    """Entry point."""
    print("=" * 60)
    print("MULTI-STRATEGY PORTFOLIO TRADING SYSTEM")
    print("=" * 60)
    print()
    
    config = SystemConfig(dry_run=True)  # SAFETY: Default dry run
    
    system = PortfolioOrchestrator(config)
    
    try:
        system.run()
    except KeyboardInterrupt:
        print("\nShutdown requested...")
    except Exception as e:
        print(f"Fatal: {e}")
        raise


if __name__ == "__main__":
    main()
