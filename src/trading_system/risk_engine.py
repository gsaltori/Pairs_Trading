"""
Risk Engine - Capital Preservation Governor

CRITICAL COMPONENT - Handles all risk decisions.
This module can ONLY reduce risk, never increase it beyond baseline.

Priority: Capital preservation > Everything else
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, date, timezone
from enum import Enum
from typing import Optional, List, Dict
from pathlib import Path

from .config import SystemConfig, RiskConfig


class RiskLevel(Enum):
    """System risk level states."""
    NORMAL = "NORMAL"           # Full trading allowed
    REDUCED = "REDUCED"         # -3% DD: 50% risk, max 2 trades
    SINGLE_TRADE = "SINGLE"     # -6% DD: max 1 trade
    HALTED = "HALTED"           # -8% DD: no new trades
    MANUAL_REVIEW = "MANUAL"    # -10% DD: requires human intervention


class TradePermission(Enum):
    """Trade permission result."""
    ALLOWED = "ALLOWED"
    BLOCKED_MAX_TRADES = "BLOCKED_MAX_TRADES"
    BLOCKED_MAX_RISK = "BLOCKED_MAX_RISK"
    BLOCKED_DRAWDOWN = "BLOCKED_DRAWDOWN"
    BLOCKED_HALTED = "BLOCKED_HALTED"
    BLOCKED_DAILY_LOSSES = "BLOCKED_DAILY_LOSSES"
    BLOCKED_TOO_SOON = "BLOCKED_TOO_SOON"


@dataclass
class OpenPosition:
    """Track an open position for risk calculation."""
    ticket: int
    symbol: str
    direction: str  # "LONG" or "SHORT"
    entry_price: float
    stop_loss: float
    position_size: float
    entry_time: datetime
    
    @property
    def risk_amount(self) -> float:
        """Risk in price terms × position size."""
        return abs(self.entry_price - self.stop_loss) * self.position_size * 100000


@dataclass
class DailyStats:
    """Daily trading statistics."""
    date: date
    trades_taken: int = 0
    trades_won: int = 0
    trades_lost: int = 0
    consecutive_losses: int = 0
    pnl: float = 0.0


@dataclass
class RiskState:
    """Complete risk state - persisted to disk."""
    # Equity tracking
    current_equity: float
    high_water_mark: float
    initial_equity: float
    
    # Drawdown
    current_drawdown: float = 0.0
    current_drawdown_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    
    # Risk level
    risk_level: RiskLevel = RiskLevel.NORMAL
    risk_per_trade: float = 0.005
    
    # Open positions
    open_positions: List[OpenPosition] = field(default_factory=list)
    total_open_risk: float = 0.0
    
    # Daily tracking
    daily_stats: DailyStats = field(default_factory=lambda: DailyStats(date=date.today()))
    
    # Timestamps
    last_trade_bar: int = -100
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Halt state
    is_halted: bool = False
    halt_reason: Optional[str] = None
    halt_time: Optional[datetime] = None
    requires_manual_review: bool = False


class RiskEngine:
    """
    Central risk management engine.
    
    ALL trade decisions must pass through this engine.
    This engine can ONLY block or reduce - never allow more risk.
    """
    
    def __init__(self, config: SystemConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.risk_config = config.risk
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize state (will be loaded from disk if exists)
        self.state: Optional[RiskState] = None
    
    def initialize(self, current_equity: float) -> None:
        """Initialize risk engine with current equity."""
        state_file = self.config.paths.state_file
        
        if state_file.exists():
            self.logger.info("Loading existing risk state...")
            self._load_state()
            # Update equity
            self.state.current_equity = current_equity
            self._recalculate_drawdown()
        else:
            self.logger.info("Initializing new risk state...")
            self.state = RiskState(
                current_equity=current_equity,
                high_water_mark=current_equity,
                initial_equity=current_equity,
                risk_per_trade=self.risk_config.risk_per_trade_normal,
            )
        
        self._update_risk_level()
        self._save_state()
        
        self.logger.info(f"Risk engine initialized: {self.state.risk_level.value}, "
                        f"DD: {self.state.current_drawdown_pct:.2%}")
    
    def update_equity(self, new_equity: float) -> None:
        """
        Update current equity and recalculate risk state.
        
        Called on every bar or after position changes.
        """
        if self.state is None:
            raise RuntimeError("Risk engine not initialized")
        
        self.state.current_equity = new_equity
        self.state.last_update = datetime.now(timezone.utc)
        
        # Update high water mark (only goes up)
        if new_equity > self.state.high_water_mark:
            self.state.high_water_mark = new_equity
            self.logger.info(f"New high water mark: ${new_equity:,.2f}")
        
        self._recalculate_drawdown()
        self._update_risk_level()
        self._save_state()
    
    def _recalculate_drawdown(self) -> None:
        """Recalculate drawdown from high water mark."""
        hwm = self.state.high_water_mark
        equity = self.state.current_equity
        
        self.state.current_drawdown = hwm - equity
        self.state.current_drawdown_pct = self.state.current_drawdown / hwm if hwm > 0 else 0
        
        # Track max drawdown
        if self.state.current_drawdown_pct > self.state.max_drawdown_pct:
            self.state.max_drawdown_pct = self.state.current_drawdown_pct
    
    def _update_risk_level(self) -> None:
        """Update risk level based on drawdown."""
        dd = self.state.current_drawdown_pct
        cfg = self.risk_config
        
        prev_level = self.state.risk_level
        
        # Determine new level (strictest rule wins)
        if dd >= cfg.dd_manual_review_threshold:
            self.state.risk_level = RiskLevel.MANUAL_REVIEW
            self.state.requires_manual_review = True
            self._halt_system(f"Drawdown {dd:.2%} >= {cfg.dd_manual_review_threshold:.2%} - MANUAL REVIEW REQUIRED")
        
        elif dd >= cfg.dd_halt_threshold:
            self.state.risk_level = RiskLevel.HALTED
            self._halt_system(f"Drawdown {dd:.2%} >= {cfg.dd_halt_threshold:.2%}")
        
        elif dd >= cfg.dd_single_trade_threshold:
            self.state.risk_level = RiskLevel.SINGLE_TRADE
            self.state.risk_per_trade = cfg.risk_per_trade_reduced
        
        elif dd >= cfg.dd_reduce_risk_threshold:
            self.state.risk_level = RiskLevel.REDUCED
            self.state.risk_per_trade = cfg.risk_per_trade_reduced
        
        else:
            # Check if we can resume normal (need recovery buffer)
            if prev_level != RiskLevel.NORMAL:
                # Must recover beyond threshold + buffer to resume
                recovery_threshold = cfg.dd_reduce_risk_threshold - cfg.dd_recovery_buffer
                if dd < recovery_threshold:
                    self.state.risk_level = RiskLevel.NORMAL
                    self.state.risk_per_trade = cfg.risk_per_trade_normal
                    self.logger.info("Risk level returned to NORMAL")
            else:
                self.state.risk_level = RiskLevel.NORMAL
                self.state.risk_per_trade = cfg.risk_per_trade_normal
        
        # Log level changes
        if prev_level != self.state.risk_level:
            self.logger.warning(f"Risk level changed: {prev_level.value} → {self.state.risk_level.value}")
    
    def _halt_system(self, reason: str) -> None:
        """Halt the trading system."""
        self.state.is_halted = True
        self.state.halt_reason = reason
        self.state.halt_time = datetime.now(timezone.utc)
        self.logger.critical(f"SYSTEM HALTED: {reason}")
    
    def can_trade(self, current_bar: int) -> TradePermission:
        """
        Check if a new trade is allowed.
        
        Returns permission status with reason if blocked.
        """
        if self.state is None:
            return TradePermission.BLOCKED_HALTED
        
        # Check 1: System halted?
        if self.state.is_halted:
            return TradePermission.BLOCKED_HALTED
        
        # Check 2: Risk level allows trading?
        if self.state.risk_level in (RiskLevel.HALTED, RiskLevel.MANUAL_REVIEW):
            return TradePermission.BLOCKED_DRAWDOWN
        
        # Check 3: Max concurrent trades
        max_trades = self._get_max_concurrent_trades()
        if len(self.state.open_positions) >= max_trades:
            return TradePermission.BLOCKED_MAX_TRADES
        
        # Check 4: Max concurrent risk
        if self.state.total_open_risk >= self.state.current_equity * self.risk_config.max_concurrent_risk:
            return TradePermission.BLOCKED_MAX_RISK
        
        # Check 5: Too soon after last trade
        bars_since_last = current_bar - self.state.last_trade_bar
        if bars_since_last < self.risk_config.min_bars_between_trades:
            return TradePermission.BLOCKED_TOO_SOON
        
        # Check 6: Daily loss limit
        self._ensure_daily_stats()
        if self.state.daily_stats.consecutive_losses >= self.risk_config.max_losses_per_day:
            return TradePermission.BLOCKED_DAILY_LOSSES
        
        return TradePermission.ALLOWED
    
    def _get_max_concurrent_trades(self) -> int:
        """Get max trades based on risk level."""
        if self.state.risk_level == RiskLevel.SINGLE_TRADE:
            return 1
        return self.risk_config.max_concurrent_trades
    
    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        symbol: str,
    ) -> float:
        """
        Calculate position size based on current risk parameters.
        
        Returns position size in lots.
        """
        if self.state is None:
            return 0.0
        
        # Risk amount in account currency
        risk_amount = self.state.current_equity * self.state.risk_per_trade
        
        # Price distance to stop
        sl_distance = abs(entry_price - stop_loss)
        if sl_distance < 1e-6:
            return 0.0
        
        # Get symbol config
        symbol_config = self.config.symbols.get(symbol)
        if symbol_config is None:
            self.logger.error(f"Unknown symbol: {symbol}")
            return 0.0
        
        # Calculate lots
        # For forex: risk_amount = sl_pips × pip_value × lots
        # sl_pips = sl_distance / 0.0001 (for 4-digit pairs)
        point_value = 0.0001  # Standard forex point
        sl_points = sl_distance / point_value
        
        # Position size = risk_amount / (sl_points × point_value_per_lot)
        # point_value_per_lot ≈ 10 for EURUSD/GBPUSD per standard lot
        lot_size = risk_amount / (sl_points * symbol_config.pip_value)
        
        # Apply limits
        lot_size = max(symbol_config.min_lot, lot_size)
        lot_size = min(symbol_config.max_lot, lot_size)
        lot_size = min(self.risk_config.max_position_size, lot_size)
        
        # Round to lot step
        lot_size = round(lot_size / symbol_config.lot_step) * symbol_config.lot_step
        
        return lot_size
    
    def register_trade_opened(
        self,
        ticket: int,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        position_size: float,
        current_bar: int,
    ) -> None:
        """Register a new opened trade."""
        position = OpenPosition(
            ticket=ticket,
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            position_size=position_size,
            entry_time=datetime.now(timezone.utc),
        )
        
        self.state.open_positions.append(position)
        self.state.last_trade_bar = current_bar
        self._recalculate_open_risk()
        
        self._ensure_daily_stats()
        self.state.daily_stats.trades_taken += 1
        
        self._save_state()
        
        self.logger.info(f"Trade registered: {ticket} {direction} {symbol} @ {entry_price}")
    
    def register_trade_closed(
        self,
        ticket: int,
        exit_price: float,
        pnl: float,
    ) -> None:
        """Register a closed trade."""
        # Find and remove position
        for i, pos in enumerate(self.state.open_positions):
            if pos.ticket == ticket:
                self.state.open_positions.pop(i)
                break
        
        self._recalculate_open_risk()
        
        # Update daily stats
        self._ensure_daily_stats()
        if pnl > 0:
            self.state.daily_stats.trades_won += 1
            self.state.daily_stats.consecutive_losses = 0
        else:
            self.state.daily_stats.trades_lost += 1
            self.state.daily_stats.consecutive_losses += 1
        
        self.state.daily_stats.pnl += pnl
        
        self._save_state()
        
        self.logger.info(f"Trade closed: {ticket} PnL: ${pnl:.2f}")
    
    def _recalculate_open_risk(self) -> None:
        """Recalculate total open risk."""
        self.state.total_open_risk = sum(pos.risk_amount for pos in self.state.open_positions)
    
    def _ensure_daily_stats(self) -> None:
        """Ensure daily stats are for today."""
        today = date.today()
        if self.state.daily_stats.date != today:
            # New day - reset stats
            self.state.daily_stats = DailyStats(date=today)
    
    def manual_resume(self, new_hwm: Optional[float] = None) -> bool:
        """
        Manually resume system after halt.
        
        REQUIRES: Manual review confirmation
        
        Args:
            new_hwm: Optional new high water mark to reset to
        
        Returns:
            True if resumed successfully
        """
        if not self.state.is_halted:
            return True
        
        if self.state.requires_manual_review:
            self.logger.warning("System requires manual review before resuming")
            # In production, this would require external confirmation
            # For now, we just log the warning
        
        self.state.is_halted = False
        self.state.halt_reason = None
        self.state.halt_time = None
        self.state.requires_manual_review = False
        
        if new_hwm is not None:
            self.state.high_water_mark = new_hwm
        
        self._recalculate_drawdown()
        self._update_risk_level()
        self._save_state()
        
        self.logger.info("System manually resumed")
        return True
    
    def get_status_summary(self) -> Dict:
        """Get current risk status summary."""
        if self.state is None:
            return {"error": "Not initialized"}
        
        return {
            "risk_level": self.state.risk_level.value,
            "is_halted": self.state.is_halted,
            "halt_reason": self.state.halt_reason,
            "current_equity": self.state.current_equity,
            "high_water_mark": self.state.high_water_mark,
            "drawdown_pct": self.state.current_drawdown_pct,
            "max_drawdown_pct": self.state.max_drawdown_pct,
            "risk_per_trade": self.state.risk_per_trade,
            "open_positions": len(self.state.open_positions),
            "total_open_risk": self.state.total_open_risk,
            "daily_trades": self.state.daily_stats.trades_taken,
            "daily_losses": self.state.daily_stats.consecutive_losses,
        }
    
    def _save_state(self) -> None:
        """Persist state to disk."""
        state_file = self.config.paths.state_file
        state_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable dict
        state_dict = {
            "current_equity": self.state.current_equity,
            "high_water_mark": self.state.high_water_mark,
            "initial_equity": self.state.initial_equity,
            "current_drawdown": self.state.current_drawdown,
            "current_drawdown_pct": self.state.current_drawdown_pct,
            "max_drawdown_pct": self.state.max_drawdown_pct,
            "risk_level": self.state.risk_level.value,
            "risk_per_trade": self.state.risk_per_trade,
            "total_open_risk": self.state.total_open_risk,
            "last_trade_bar": self.state.last_trade_bar,
            "last_update": self.state.last_update.isoformat(),
            "is_halted": self.state.is_halted,
            "halt_reason": self.state.halt_reason,
            "halt_time": self.state.halt_time.isoformat() if self.state.halt_time else None,
            "requires_manual_review": self.state.requires_manual_review,
            "open_positions": [
                {
                    "ticket": p.ticket,
                    "symbol": p.symbol,
                    "direction": p.direction,
                    "entry_price": p.entry_price,
                    "stop_loss": p.stop_loss,
                    "position_size": p.position_size,
                    "entry_time": p.entry_time.isoformat(),
                }
                for p in self.state.open_positions
            ],
            "daily_stats": {
                "date": self.state.daily_stats.date.isoformat(),
                "trades_taken": self.state.daily_stats.trades_taken,
                "trades_won": self.state.daily_stats.trades_won,
                "trades_lost": self.state.daily_stats.trades_lost,
                "consecutive_losses": self.state.daily_stats.consecutive_losses,
                "pnl": self.state.daily_stats.pnl,
            },
        }
        
        with open(state_file, "w") as f:
            json.dump(state_dict, f, indent=2)
    
    def _load_state(self) -> None:
        """Load state from disk."""
        state_file = self.config.paths.state_file
        
        with open(state_file, "r") as f:
            data = json.load(f)
        
        # Reconstruct state
        self.state = RiskState(
            current_equity=data["current_equity"],
            high_water_mark=data["high_water_mark"],
            initial_equity=data["initial_equity"],
            current_drawdown=data["current_drawdown"],
            current_drawdown_pct=data["current_drawdown_pct"],
            max_drawdown_pct=data["max_drawdown_pct"],
            risk_level=RiskLevel(data["risk_level"]),
            risk_per_trade=data["risk_per_trade"],
            total_open_risk=data["total_open_risk"],
            last_trade_bar=data["last_trade_bar"],
            last_update=datetime.fromisoformat(data["last_update"]),
            is_halted=data["is_halted"],
            halt_reason=data["halt_reason"],
            halt_time=datetime.fromisoformat(data["halt_time"]) if data["halt_time"] else None,
            requires_manual_review=data["requires_manual_review"],
        )
        
        # Reconstruct positions
        self.state.open_positions = [
            OpenPosition(
                ticket=p["ticket"],
                symbol=p["symbol"],
                direction=p["direction"],
                entry_price=p["entry_price"],
                stop_loss=p["stop_loss"],
                position_size=p["position_size"],
                entry_time=datetime.fromisoformat(p["entry_time"]),
            )
            for p in data["open_positions"]
        ]
        
        # Reconstruct daily stats
        ds = data["daily_stats"]
        self.state.daily_stats = DailyStats(
            date=date.fromisoformat(ds["date"]),
            trades_taken=ds["trades_taken"],
            trades_won=ds["trades_won"],
            trades_lost=ds["trades_lost"],
            consecutive_losses=ds["consecutive_losses"],
            pnl=ds["pnl"],
        )
