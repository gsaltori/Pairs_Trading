"""
Multi-Strategy Risk Engine

Extended risk engine supporting per-strategy risk limits
and the portfolio-level drawdown governors.

RISK LIMITS (LOCKED):
- Trend Continuation: 0.30%
- Trend Pullback: 0.25%
- Volatility Expansion: 0.20%
- Total max: 0.75%

DRAWDOWN GOVERNORS (LOCKED):
- DD ≥ 5% → reduce all risk by 50%
- DD ≥ 8% → SYSTEM HALT
- 3 consecutive losses → stop for 24h

Initial capital: USD 100 (micro account)
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, date, timezone, timedelta
from enum import Enum
from typing import Optional, List, Dict
from pathlib import Path

from .config import SystemConfig
from .strategy_router import StrategyType


class PortfolioRiskLevel(Enum):
    """Portfolio risk level states."""
    NORMAL = "NORMAL"           # Full trading
    REDUCED = "REDUCED"         # -5% DD: 50% risk
    HALTED = "HALTED"           # -8% DD: no trading
    COOLING_OFF = "COOLING_OFF" # 3 consecutive losses


@dataclass
class StrategyPosition:
    """Track position per strategy."""
    ticket: int
    strategy: StrategyType
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    risk_amount: float
    entry_time: datetime
    entry_bar: int


@dataclass
class PortfolioDailyStats:
    """Daily portfolio statistics."""
    date: date
    trades_taken: int = 0
    trades_won: int = 0
    trades_lost: int = 0
    consecutive_losses: int = 0
    pnl: float = 0.0
    cooling_off_until: Optional[datetime] = None


@dataclass
class PortfolioRiskState:
    """Complete portfolio risk state."""
    # Equity tracking
    current_equity: float
    high_water_mark: float
    initial_equity: float
    
    # Drawdown
    current_drawdown: float = 0.0
    current_drawdown_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    
    # Risk level
    risk_level: PortfolioRiskLevel = PortfolioRiskLevel.NORMAL
    risk_multiplier: float = 1.0  # 1.0 = normal, 0.5 = reduced
    
    # Positions by strategy
    positions: Dict[str, StrategyPosition] = field(default_factory=dict)
    
    # Daily tracking
    daily_stats: PortfolioDailyStats = field(default_factory=lambda: PortfolioDailyStats(date=date.today()))
    
    # State
    last_trade_bar: int = -100
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Halt state
    is_halted: bool = False
    halt_reason: Optional[str] = None
    halt_time: Optional[datetime] = None


class PortfolioRiskEngine:
    """
    Portfolio-level risk management engine.
    
    Manages risk across multiple strategies with:
    - Per-strategy risk limits
    - Global drawdown governors
    - Consecutive loss protection
    """
    
    # LOCKED PARAMETERS
    STRATEGY_RISK_LIMITS = {
        StrategyType.TREND_CONTINUATION: 0.0030,
        StrategyType.TREND_PULLBACK: 0.0025,
        StrategyType.VOLATILITY_EXPANSION: 0.0020,
    }
    
    TOTAL_MAX_RISK = 0.0075  # 0.75%
    
    DD_REDUCE_THRESHOLD = 0.05      # -5% → reduce risk 50%
    DD_HALT_THRESHOLD = 0.08        # -8% → halt
    
    CONSECUTIVE_LOSS_LIMIT = 3
    COOLING_OFF_HOURS = 24
    
    MIN_BARS_BETWEEN_TRADES = 2
    
    def __init__(
        self,
        config: SystemConfig,
        logger: Optional[logging.Logger] = None,
    ):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        self.state: Optional[PortfolioRiskState] = None
    
    def initialize(self, current_equity: float) -> None:
        """Initialize risk engine with current equity."""
        state_file = self.config.paths.state_file
        
        if state_file.exists():
            self.logger.info("Loading existing portfolio risk state...")
            self._load_state()
            self.state.current_equity = current_equity
            self._recalculate_drawdown()
        else:
            self.logger.info(f"Initializing new portfolio risk state (equity: ${current_equity:.2f})...")
            self.state = PortfolioRiskState(
                current_equity=current_equity,
                high_water_mark=current_equity,
                initial_equity=current_equity,
            )
        
        self._update_risk_level()
        self._save_state()
        
        self.logger.info(
            f"Portfolio risk initialized: {self.state.risk_level.value}, "
            f"DD: {self.state.current_drawdown_pct:.2%}, "
            f"Multiplier: {self.state.risk_multiplier:.1f}x"
        )
    
    def update_equity(self, new_equity: float) -> None:
        """Update current equity and recalculate risk state."""
        if self.state is None:
            raise RuntimeError("Risk engine not initialized")
        
        self.state.current_equity = new_equity
        self.state.last_update = datetime.now(timezone.utc)
        
        if new_equity > self.state.high_water_mark:
            self.state.high_water_mark = new_equity
            self.logger.info(f"New HWM: ${new_equity:.2f}")
        
        self._recalculate_drawdown()
        self._update_risk_level()
        self._save_state()
    
    def _recalculate_drawdown(self) -> None:
        """Recalculate drawdown from HWM."""
        hwm = self.state.high_water_mark
        equity = self.state.current_equity
        
        self.state.current_drawdown = hwm - equity
        self.state.current_drawdown_pct = self.state.current_drawdown / hwm if hwm > 0 else 0
        
        if self.state.current_drawdown_pct > self.state.max_drawdown_pct:
            self.state.max_drawdown_pct = self.state.current_drawdown_pct
    
    def _update_risk_level(self) -> None:
        """Update risk level based on drawdown and losses."""
        dd = self.state.current_drawdown_pct
        prev_level = self.state.risk_level
        
        # Check cooling off
        self._check_cooling_off()
        
        if self.state.risk_level == PortfolioRiskLevel.COOLING_OFF:
            return  # Don't override cooling off
        
        # DD ≥ 8% → HALT
        if dd >= self.DD_HALT_THRESHOLD:
            self.state.risk_level = PortfolioRiskLevel.HALTED
            self.state.risk_multiplier = 0.0
            self._halt_system(f"Drawdown {dd:.2%} >= {self.DD_HALT_THRESHOLD:.2%}")
        
        # DD ≥ 5% → REDUCED
        elif dd >= self.DD_REDUCE_THRESHOLD:
            self.state.risk_level = PortfolioRiskLevel.REDUCED
            self.state.risk_multiplier = 0.5
        
        # Normal
        else:
            if prev_level == PortfolioRiskLevel.REDUCED:
                # Need recovery buffer before returning to normal
                recovery_threshold = self.DD_REDUCE_THRESHOLD - 0.01
                if dd < recovery_threshold:
                    self.state.risk_level = PortfolioRiskLevel.NORMAL
                    self.state.risk_multiplier = 1.0
            else:
                self.state.risk_level = PortfolioRiskLevel.NORMAL
                self.state.risk_multiplier = 1.0
        
        if prev_level != self.state.risk_level:
            self.logger.warning(f"Risk level: {prev_level.value} → {self.state.risk_level.value}")
    
    def _check_cooling_off(self) -> None:
        """Check if cooling off period has ended."""
        if self.state.risk_level != PortfolioRiskLevel.COOLING_OFF:
            return
        
        self._ensure_daily_stats()
        
        if self.state.daily_stats.cooling_off_until:
            now = datetime.now(timezone.utc)
            if now >= self.state.daily_stats.cooling_off_until:
                self.logger.info("Cooling off period ended")
                self.state.risk_level = PortfolioRiskLevel.NORMAL
                self.state.risk_multiplier = 1.0
                self.state.daily_stats.cooling_off_until = None
                self.state.daily_stats.consecutive_losses = 0
    
    def _halt_system(self, reason: str) -> None:
        """Halt trading system."""
        self.state.is_halted = True
        self.state.halt_reason = reason
        self.state.halt_time = datetime.now(timezone.utc)
        self.logger.critical(f"SYSTEM HALTED: {reason}")
    
    def can_trade(
        self,
        strategy: StrategyType,
        current_bar: int,
    ) -> tuple:
        """
        Check if a trade is allowed for the given strategy.
        
        Returns: (allowed: bool, reason: str)
        """
        if self.state is None:
            return False, "NOT_INITIALIZED"
        
        # System halted
        if self.state.is_halted:
            return False, "SYSTEM_HALTED"
        
        # Check cooling off
        self._check_cooling_off()
        if self.state.risk_level == PortfolioRiskLevel.COOLING_OFF:
            return False, "COOLING_OFF"
        
        # Risk level prevents trading
        if self.state.risk_level == PortfolioRiskLevel.HALTED:
            return False, "HALTED_DRAWDOWN"
        
        # Already have a position for this strategy
        if strategy.value in self.state.positions:
            return False, "POSITION_OPEN"
        
        # Too soon after last trade
        if current_bar - self.state.last_trade_bar < self.MIN_BARS_BETWEEN_TRADES:
            return False, "TOO_SOON"
        
        # Check total risk
        current_risk = sum(p.risk_amount for p in self.state.positions.values())
        max_new_risk = self.get_risk_amount(strategy)
        
        if (current_risk + max_new_risk) / self.state.current_equity > self.TOTAL_MAX_RISK:
            return False, "MAX_TOTAL_RISK"
        
        return True, "ALLOWED"
    
    def get_risk_amount(self, strategy: StrategyType) -> float:
        """Get risk amount for strategy (adjusted by risk multiplier)."""
        base_risk = self.STRATEGY_RISK_LIMITS.get(strategy, 0.0025)
        return self.state.current_equity * base_risk * self.state.risk_multiplier
    
    def get_risk_percent(self, strategy: StrategyType) -> float:
        """Get risk percent for strategy (adjusted by multiplier)."""
        base_risk = self.STRATEGY_RISK_LIMITS.get(strategy, 0.0025)
        return base_risk * self.state.risk_multiplier
    
    def calculate_position_size(
        self,
        strategy: StrategyType,
        entry_price: float,
        stop_loss: float,
        pip_value: float = 10.0,
    ) -> float:
        """
        Calculate position size for given strategy.
        
        Args:
            strategy: Strategy type
            entry_price: Entry price
            stop_loss: Stop loss price
            pip_value: Value per pip per lot
        
        Returns:
            Position size in lots
        """
        risk_amount = self.get_risk_amount(strategy)
        sl_distance = abs(entry_price - stop_loss)
        
        if sl_distance < 1e-6:
            return 0.0
        
        # For forex: sl_pips = sl_distance / point
        point = 0.0001
        sl_pips = sl_distance / point
        
        # lots = risk_amount / (sl_pips × pip_value)
        lots = risk_amount / (sl_pips * pip_value)
        
        # Apply limits
        lots = max(0.01, lots)  # Min 0.01 lots
        lots = min(1.0, lots)   # Max 1.0 lots
        lots = round(lots, 2)   # Round to 0.01
        
        return lots
    
    def register_trade_opened(
        self,
        ticket: int,
        strategy: StrategyType,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        position_size: float,
        current_bar: int,
    ) -> None:
        """Register a new opened trade."""
        risk_amount = abs(entry_price - stop_loss) * position_size * 100000
        
        position = StrategyPosition(
            ticket=ticket,
            strategy=strategy,
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            risk_amount=risk_amount,
            entry_time=datetime.now(timezone.utc),
            entry_bar=current_bar,
        )
        
        self.state.positions[strategy.value] = position
        self.state.last_trade_bar = current_bar
        
        self._ensure_daily_stats()
        self.state.daily_stats.trades_taken += 1
        
        self._save_state()
        
        self.logger.info(
            f"Position opened: {strategy.value} {direction} {symbol} "
            f"@ {entry_price:.5f}, risk: ${risk_amount:.2f}"
        )
    
    def register_trade_closed(
        self,
        strategy: StrategyType,
        exit_price: float,
        pnl: float,
    ) -> None:
        """Register a closed trade."""
        if strategy.value not in self.state.positions:
            return
        
        del self.state.positions[strategy.value]
        
        self._ensure_daily_stats()
        
        if pnl > 0:
            self.state.daily_stats.trades_won += 1
            self.state.daily_stats.consecutive_losses = 0
        else:
            self.state.daily_stats.trades_lost += 1
            self.state.daily_stats.consecutive_losses += 1
            
            # Check consecutive loss limit
            if self.state.daily_stats.consecutive_losses >= self.CONSECUTIVE_LOSS_LIMIT:
                self._enter_cooling_off()
        
        self.state.daily_stats.pnl += pnl
        
        self._save_state()
        
        self.logger.info(f"Position closed: {strategy.value} PnL: ${pnl:.2f}")
    
    def _enter_cooling_off(self) -> None:
        """Enter cooling off period after consecutive losses."""
        self.state.risk_level = PortfolioRiskLevel.COOLING_OFF
        self.state.risk_multiplier = 0.0
        self.state.daily_stats.cooling_off_until = (
            datetime.now(timezone.utc) + timedelta(hours=self.COOLING_OFF_HOURS)
        )
        self.logger.warning(
            f"COOLING OFF: {self.CONSECUTIVE_LOSS_LIMIT} consecutive losses. "
            f"Trading resumes at {self.state.daily_stats.cooling_off_until}"
        )
    
    def _ensure_daily_stats(self) -> None:
        """Ensure daily stats are for today."""
        today = date.today()
        if self.state.daily_stats.date != today:
            # Check if we were in cooling off
            was_cooling = self.state.risk_level == PortfolioRiskLevel.COOLING_OFF
            cooling_until = self.state.daily_stats.cooling_off_until
            
            self.state.daily_stats = PortfolioDailyStats(date=today)
            
            # Preserve cooling off if not expired
            if was_cooling and cooling_until:
                if datetime.now(timezone.utc) < cooling_until:
                    self.state.daily_stats.cooling_off_until = cooling_until
    
    def get_status_summary(self) -> Dict:
        """Get current risk status summary."""
        if self.state is None:
            return {"error": "Not initialized"}
        
        return {
            "risk_level": self.state.risk_level.value,
            "is_halted": self.state.is_halted,
            "halt_reason": self.state.halt_reason,
            "equity": self.state.current_equity,
            "hwm": self.state.high_water_mark,
            "drawdown_pct": self.state.current_drawdown_pct,
            "max_drawdown_pct": self.state.max_drawdown_pct,
            "risk_multiplier": self.state.risk_multiplier,
            "open_positions": list(self.state.positions.keys()),
            "daily_trades": self.state.daily_stats.trades_taken,
            "consecutive_losses": self.state.daily_stats.consecutive_losses,
        }
    
    def _save_state(self) -> None:
        """Persist state to disk."""
        state_file = self.config.paths.state_file
        state_file.parent.mkdir(parents=True, exist_ok=True)
        
        state_dict = {
            "current_equity": self.state.current_equity,
            "high_water_mark": self.state.high_water_mark,
            "initial_equity": self.state.initial_equity,
            "current_drawdown": self.state.current_drawdown,
            "current_drawdown_pct": self.state.current_drawdown_pct,
            "max_drawdown_pct": self.state.max_drawdown_pct,
            "risk_level": self.state.risk_level.value,
            "risk_multiplier": self.state.risk_multiplier,
            "last_trade_bar": self.state.last_trade_bar,
            "last_update": self.state.last_update.isoformat(),
            "is_halted": self.state.is_halted,
            "halt_reason": self.state.halt_reason,
            "halt_time": self.state.halt_time.isoformat() if self.state.halt_time else None,
            "positions": {
                k: {
                    "ticket": p.ticket,
                    "strategy": p.strategy.value,
                    "symbol": p.symbol,
                    "direction": p.direction,
                    "entry_price": p.entry_price,
                    "stop_loss": p.stop_loss,
                    "take_profit": p.take_profit,
                    "position_size": p.position_size,
                    "risk_amount": p.risk_amount,
                    "entry_time": p.entry_time.isoformat(),
                    "entry_bar": p.entry_bar,
                }
                for k, p in self.state.positions.items()
            },
            "daily_stats": {
                "date": self.state.daily_stats.date.isoformat(),
                "trades_taken": self.state.daily_stats.trades_taken,
                "trades_won": self.state.daily_stats.trades_won,
                "trades_lost": self.state.daily_stats.trades_lost,
                "consecutive_losses": self.state.daily_stats.consecutive_losses,
                "pnl": self.state.daily_stats.pnl,
                "cooling_off_until": (
                    self.state.daily_stats.cooling_off_until.isoformat()
                    if self.state.daily_stats.cooling_off_until else None
                ),
            },
        }
        
        with open(state_file, "w") as f:
            json.dump(state_dict, f, indent=2)
    
    def _load_state(self) -> None:
        """Load state from disk."""
        state_file = self.config.paths.state_file
        
        with open(state_file, "r") as f:
            data = json.load(f)
        
        self.state = PortfolioRiskState(
            current_equity=data["current_equity"],
            high_water_mark=data["high_water_mark"],
            initial_equity=data["initial_equity"],
            current_drawdown=data["current_drawdown"],
            current_drawdown_pct=data["current_drawdown_pct"],
            max_drawdown_pct=data["max_drawdown_pct"],
            risk_level=PortfolioRiskLevel(data["risk_level"]),
            risk_multiplier=data["risk_multiplier"],
            last_trade_bar=data["last_trade_bar"],
            last_update=datetime.fromisoformat(data["last_update"]),
            is_halted=data["is_halted"],
            halt_reason=data["halt_reason"],
            halt_time=datetime.fromisoformat(data["halt_time"]) if data["halt_time"] else None,
        )
        
        # Reconstruct positions
        self.state.positions = {}
        for k, p in data["positions"].items():
            self.state.positions[k] = StrategyPosition(
                ticket=p["ticket"],
                strategy=StrategyType(p["strategy"]),
                symbol=p["symbol"],
                direction=p["direction"],
                entry_price=p["entry_price"],
                stop_loss=p["stop_loss"],
                take_profit=p["take_profit"],
                position_size=p["position_size"],
                risk_amount=p["risk_amount"],
                entry_time=datetime.fromisoformat(p["entry_time"]),
                entry_bar=p["entry_bar"],
            )
        
        # Reconstruct daily stats
        ds = data["daily_stats"]
        self.state.daily_stats = PortfolioDailyStats(
            date=date.fromisoformat(ds["date"]),
            trades_taken=ds["trades_taken"],
            trades_won=ds["trades_won"],
            trades_lost=ds["trades_lost"],
            consecutive_losses=ds["consecutive_losses"],
            pnl=ds["pnl"],
            cooling_off_until=(
                datetime.fromisoformat(ds["cooling_off_until"])
                if ds["cooling_off_until"] else None
            ),
        )
