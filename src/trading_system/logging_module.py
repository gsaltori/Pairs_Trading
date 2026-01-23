"""
Logging & Monitoring Module

Comprehensive logging for audit trail:
- Trade log (CSV)
- Block log (CSV)
- Risk state log (CSV)
- System log (structured)
- Daily summary

All actions must be logged.
"""

import csv
import json
import logging
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass

from .config import PathConfig
from .signal_engine import TradeSignal
from .gatekeeper_engine import GatekeeperDecision
from .execution_engine import OrderExecution
from .risk_engine import RiskState


def setup_logging(paths: PathConfig, verbose: bool = True) -> logging.Logger:
    """
    Configure system logging.
    
    Returns configured logger.
    """
    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("TradingSystem")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    
    # File handler
    file_handler = logging.FileHandler(paths.system_log)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_format)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


class TradeLogger:
    """
    CSV trade logger for audit trail.
    
    Logs all trade attempts, executions, and closures.
    """
    
    HEADERS = [
        "timestamp",
        "action",  # SIGNAL, EXECUTE, CLOSE
        "ticket",
        "symbol",
        "direction",
        "entry_price",
        "exit_price",
        "stop_loss",
        "take_profit",
        "volume",
        "pnl",
        "result",
        "error",
    ]
    
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self._ensure_headers()
    
    def _ensure_headers(self) -> None:
        """Ensure CSV has headers."""
        if not self.log_path.exists():
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.HEADERS)
    
    def log_signal(self, signal: TradeSignal, symbol: str) -> None:
        """Log a generated signal."""
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "SIGNAL",
            "ticket": "",
            "symbol": symbol,
            "direction": signal.direction.value,
            "entry_price": signal.entry_price,
            "exit_price": "",
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
            "volume": "",
            "pnl": "",
            "result": "",
            "error": "",
        }
        self._write_row(row)
    
    def log_execution(self, execution: OrderExecution) -> None:
        """Log an order execution."""
        row = {
            "timestamp": execution.timestamp.isoformat() if execution.timestamp else "",
            "action": "EXECUTE",
            "ticket": execution.ticket or "",
            "symbol": execution.symbol,
            "direction": execution.direction,
            "entry_price": execution.price,
            "exit_price": "",
            "stop_loss": execution.stop_loss,
            "take_profit": execution.take_profit,
            "volume": execution.volume,
            "pnl": "",
            "result": execution.result.value,
            "error": execution.error_message or "",
        }
        self._write_row(row)
    
    def log_close(
        self,
        ticket: int,
        symbol: str,
        exit_price: float,
        pnl: float,
    ) -> None:
        """Log a trade closure."""
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "CLOSE",
            "ticket": ticket,
            "symbol": symbol,
            "direction": "",
            "entry_price": "",
            "exit_price": exit_price,
            "stop_loss": "",
            "take_profit": "",
            "volume": "",
            "pnl": pnl,
            "result": "CLOSED",
            "error": "",
        }
        self._write_row(row)
    
    def _write_row(self, row: Dict) -> None:
        """Write a row to CSV."""
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.HEADERS)
            writer.writerow(row)


class BlockLogger:
    """
    CSV logger for gatekeeper blocks.
    """
    
    HEADERS = [
        "timestamp",
        "signal_direction",
        "signal_price",
        "allowed",
        "reasons",
        "zscore",
        "correlation",
        "correlation_trend",
        "volatility_ratio",
    ]
    
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self._ensure_headers()
    
    def _ensure_headers(self) -> None:
        if not self.log_path.exists():
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.HEADERS)
    
    def log_decision(
        self,
        decision: GatekeeperDecision,
        signal: Optional[TradeSignal] = None,
    ) -> None:
        """Log a gatekeeper decision."""
        row = {
            "timestamp": decision.timestamp.isoformat(),
            "signal_direction": signal.direction.value if signal else "",
            "signal_price": signal.entry_price if signal else "",
            "allowed": decision.allowed,
            "reasons": "|".join(r.value for r in decision.reasons),
            "zscore": f"{decision.zscore:.4f}",
            "correlation": f"{decision.correlation:.4f}",
            "correlation_trend": f"{decision.correlation_trend:.4f}",
            "volatility_ratio": f"{decision.volatility_ratio:.4f}",
        }
        
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.HEADERS)
            writer.writerow(row)


class RiskLogger:
    """
    CSV logger for risk state snapshots.
    """
    
    HEADERS = [
        "timestamp",
        "equity",
        "high_water_mark",
        "drawdown_pct",
        "risk_level",
        "risk_per_trade",
        "open_positions",
        "total_open_risk",
        "is_halted",
    ]
    
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self._ensure_headers()
    
    def _ensure_headers(self) -> None:
        if not self.log_path.exists():
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.HEADERS)
    
    def log_state(self, state: RiskState) -> None:
        """Log current risk state."""
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "equity": f"{state.current_equity:.2f}",
            "high_water_mark": f"{state.high_water_mark:.2f}",
            "drawdown_pct": f"{state.current_drawdown_pct:.4f}",
            "risk_level": state.risk_level.value,
            "risk_per_trade": f"{state.risk_per_trade:.4f}",
            "open_positions": len(state.open_positions),
            "total_open_risk": f"{state.total_open_risk:.2f}",
            "is_halted": state.is_halted,
        }
        
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.HEADERS)
            writer.writerow(row)


@dataclass
class DailySummary:
    """Daily trading summary."""
    date: date
    starting_equity: float
    ending_equity: float
    high_water_mark: float
    max_drawdown: float
    trades_taken: int
    trades_won: int
    trades_lost: int
    trades_blocked: int
    gross_pnl: float
    risk_level_changes: int
    was_halted: bool
    
    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(),
            "starting_equity": self.starting_equity,
            "ending_equity": self.ending_equity,
            "daily_return_pct": (self.ending_equity - self.starting_equity) / self.starting_equity,
            "high_water_mark": self.high_water_mark,
            "max_drawdown": self.max_drawdown,
            "trades_taken": self.trades_taken,
            "trades_won": self.trades_won,
            "trades_lost": self.trades_lost,
            "win_rate": self.trades_won / self.trades_taken if self.trades_taken > 0 else 0,
            "trades_blocked": self.trades_blocked,
            "gross_pnl": self.gross_pnl,
            "was_halted": self.was_halted,
        }


class SummaryLogger:
    """
    JSON logger for daily summaries.
    """
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def save_daily_summary(self, summary: DailySummary) -> None:
        """Save daily summary to JSON."""
        filename = f"summary_{summary.date.isoformat()}.json"
        filepath = self.log_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(summary.to_dict(), f, indent=2)
    
    def load_daily_summary(self, target_date: date) -> Optional[DailySummary]:
        """Load daily summary from JSON."""
        filename = f"summary_{target_date.isoformat()}.json"
        filepath = self.log_dir / filename
        
        if not filepath.exists():
            return None
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return DailySummary(
            date=date.fromisoformat(data["date"]),
            starting_equity=data["starting_equity"],
            ending_equity=data["ending_equity"],
            high_water_mark=data["high_water_mark"],
            max_drawdown=data["max_drawdown"],
            trades_taken=data["trades_taken"],
            trades_won=data["trades_won"],
            trades_lost=data["trades_lost"],
            trades_blocked=data["trades_blocked"],
            gross_pnl=data["gross_pnl"],
            risk_level_changes=0,
            was_halted=data["was_halted"],
        )
