"""
Portfolio Risk Management for Micro-Edge System

Handles:
- Daily risk budget allocation
- Correlation control
- Kill-switch logic
- Position sizing across strategies
"""

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
from enum import Enum


class KillSwitchReason(Enum):
    DAILY_LOSS_LIMIT = "DAILY_LOSS_LIMIT"
    WEEKLY_LOSS_LIMIT = "WEEKLY_LOSS_LIMIT"
    CONSECUTIVE_LOSSES = "CONSECUTIVE_LOSSES"
    CORRELATION_BREACH = "CORRELATION_BREACH"
    MANUAL = "MANUAL"


@dataclass
class DailyPortfolioStats:
    """Daily statistics for portfolio risk tracking."""
    date: datetime
    trades_opened: int = 0
    trades_closed: int = 0
    wins: int = 0
    losses: int = 0
    gross_pnl: float = 0.0
    risk_used: float = 0.0
    max_concurrent: int = 0
    
    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0


@dataclass
class PortfolioRiskState:
    """Current state of portfolio risk management."""
    # Capital
    initial_capital: float
    current_equity: float
    
    # Daily tracking
    daily_pnl: float = 0.0
    daily_risk_used: float = 0.0
    daily_trades: int = 0
    
    # Weekly tracking
    weekly_pnl: float = 0.0
    weekly_trades: int = 0
    
    # Streaks
    consecutive_losses: int = 0
    consecutive_wins: int = 0
    
    # Correlation
    current_long_exposure: float = 0.0  # Total risk in longs
    current_short_exposure: float = 0.0  # Total risk in shorts
    
    # Kill switch
    is_halted: bool = False
    halt_reason: Optional[KillSwitchReason] = None
    halt_until: Optional[datetime] = None
    
    # History
    daily_history: List[DailyPortfolioStats] = field(default_factory=list)
    
    @property
    def drawdown_pct(self) -> float:
        if self.initial_capital <= 0:
            return 0
        return (self.initial_capital - self.current_equity) / self.initial_capital
    
    @property
    def net_exposure(self) -> float:
        return self.current_long_exposure - self.current_short_exposure
    
    @property
    def gross_exposure(self) -> float:
        return self.current_long_exposure + self.current_short_exposure


class PortfolioRiskManager:
    """
    Portfolio-level risk management for micro-edge system.
    
    RISK LIMITS:
    - Max 2% daily risk budget
    - Max 1% daily loss (hard stop)
    - Max 3% weekly loss (weekly halt)
    - Max 5 consecutive losses (cooldown)
    - Max 1.5% directional exposure
    
    POSITION SIZING:
    - Base: 0.3% risk per trade
    - Adjusted by bias confidence (0.5x to 1.5x)
    - Reduced after losses
    
    KILL SWITCHES:
    - Daily loss > 1% → Stop trading for day
    - Weekly loss > 3% → Stop trading for week
    - 5 consecutive losses → 4-hour cooldown
    - Correlation breach → Reduce exposure
    """
    
    # Risk limits (as fraction of capital)
    MAX_DAILY_RISK = 0.02       # 2% max daily risk budget
    MAX_DAILY_LOSS = 0.01       # 1% daily loss limit
    MAX_WEEKLY_LOSS = 0.03      # 3% weekly loss limit
    MAX_DIRECTIONAL = 0.015     # 1.5% max in one direction
    MAX_CONSECUTIVE_LOSS = 5     # Consecutive loss limit
    
    # Position sizing
    BASE_RISK = 0.003           # 0.3% base risk per trade
    MIN_RISK = 0.001            # 0.1% minimum risk
    MAX_RISK = 0.005            # 0.5% maximum risk
    
    # Trade limits
    MAX_DAILY_TRADES = 10
    MAX_CONCURRENT = 3
    
    def __init__(self, initial_capital: float = 100.0):
        self.state = PortfolioRiskState(
            initial_capital=initial_capital,
            current_equity=initial_capital,
        )
        self._open_positions: Dict[str, dict] = {}
        self._current_date: Optional[datetime] = None
        self._week_start: Optional[datetime] = None
    
    def new_day(self, date: datetime) -> None:
        """Reset daily counters."""
        # Save previous day stats
        if self._current_date:
            daily_stat = DailyPortfolioStats(
                date=self._current_date,
                trades_opened=self.state.daily_trades,
                gross_pnl=self.state.daily_pnl,
                risk_used=self.state.daily_risk_used,
            )
            self.state.daily_history.append(daily_stat)
        
        self._current_date = date
        self.state.daily_pnl = 0.0
        self.state.daily_risk_used = 0.0
        self.state.daily_trades = 0
        
        # Check weekly reset (Monday)
        if self._week_start is None or (date - self._week_start).days >= 7:
            self._week_start = date
            self.state.weekly_pnl = 0.0
            self.state.weekly_trades = 0
        
        # Clear daily halt
        if self.state.is_halted and self.state.halt_reason == KillSwitchReason.DAILY_LOSS_LIMIT:
            self.state.is_halted = False
            self.state.halt_reason = None
    
    def can_trade(self, timestamp: datetime) -> Tuple[bool, Optional[str]]:
        """Check if trading is allowed."""
        # Check halt
        if self.state.is_halted:
            if self.state.halt_until and timestamp < self.state.halt_until:
                return False, f"Halted: {self.state.halt_reason.value}"
            else:
                # Halt expired
                self.state.is_halted = False
                self.state.halt_reason = None
        
        # Check daily loss
        if self.state.daily_pnl <= -self.state.current_equity * self.MAX_DAILY_LOSS:
            self._trigger_halt(KillSwitchReason.DAILY_LOSS_LIMIT, None)
            return False, "Daily loss limit reached"
        
        # Check weekly loss
        if self.state.weekly_pnl <= -self.state.initial_capital * self.MAX_WEEKLY_LOSS:
            # Calculate next Monday
            days_until_monday = (7 - timestamp.weekday()) % 7
            if days_until_monday == 0:
                days_until_monday = 7
            halt_until = timestamp + timedelta(days=days_until_monday)
            self._trigger_halt(KillSwitchReason.WEEKLY_LOSS_LIMIT, halt_until)
            return False, "Weekly loss limit reached"
        
        # Check consecutive losses
        if self.state.consecutive_losses >= self.MAX_CONSECUTIVE_LOSS:
            halt_until = timestamp + timedelta(hours=4)
            self._trigger_halt(KillSwitchReason.CONSECUTIVE_LOSSES, halt_until)
            self.state.consecutive_losses = 0  # Reset after halt
            return False, "Consecutive loss limit"
        
        # Check daily trade limit
        if self.state.daily_trades >= self.MAX_DAILY_TRADES:
            return False, "Daily trade limit reached"
        
        # Check daily risk budget
        if self.state.daily_risk_used >= self.MAX_DAILY_RISK:
            return False, "Daily risk budget exhausted"
        
        # Check concurrent positions
        if len(self._open_positions) >= self.MAX_CONCURRENT:
            return False, "Max concurrent positions"
        
        return True, None
    
    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        direction: str,  # "LONG" or "SHORT"
        bias_confidence: float = 0.5,
    ) -> Tuple[float, float]:
        """
        Calculate position size for a trade.
        
        Returns: (lot_size, risk_fraction)
        """
        # Base risk
        risk_frac = self.BASE_RISK
        
        # Adjust for confidence
        confidence_mult = 0.5 + bias_confidence  # 0.5 to 1.5
        risk_frac *= confidence_mult
        
        # Adjust for consecutive losses
        if self.state.consecutive_losses >= 2:
            risk_frac *= 0.5
        
        # Clamp
        risk_frac = max(self.MIN_RISK, min(self.MAX_RISK, risk_frac))
        
        # Check daily budget
        remaining = (self.MAX_DAILY_RISK - self.state.daily_risk_used)
        risk_frac = min(risk_frac, remaining)
        
        # Check directional limit
        if direction == "LONG":
            remaining_directional = self.MAX_DIRECTIONAL - self.state.current_long_exposure
        else:
            remaining_directional = self.MAX_DIRECTIONAL - self.state.current_short_exposure
        
        risk_frac = min(risk_frac, max(0, remaining_directional))
        
        if risk_frac <= 0:
            return 0.0, 0.0
        
        # Convert to lot size
        risk_amount = self.state.current_equity * risk_frac
        sl_distance = abs(entry_price - stop_loss)
        
        if sl_distance <= 0:
            return 0.0, 0.0
        
        lot_size = risk_amount / (sl_distance * 100000)  # For EURUSD
        lot_size = max(0.01, min(1.0, round(lot_size, 2)))
        
        return lot_size, risk_frac
    
    def register_trade_opened(
        self,
        trade_id: str,
        direction: str,
        risk_frac: float,
        entry_price: float,
        stop_loss: float,
        lot_size: float,
    ) -> None:
        """Register a new trade."""
        self._open_positions[trade_id] = {
            'direction': direction,
            'risk': risk_frac,
            'entry': entry_price,
            'sl': stop_loss,
            'size': lot_size,
        }
        
        self.state.daily_trades += 1
        self.state.daily_risk_used += risk_frac
        self.state.weekly_trades += 1
        
        if direction == "LONG":
            self.state.current_long_exposure += risk_frac
        else:
            self.state.current_short_exposure += risk_frac
    
    def register_trade_closed(
        self,
        trade_id: str,
        pnl: float,
        won: bool,
    ) -> None:
        """Register a closed trade."""
        if trade_id not in self._open_positions:
            return
        
        pos = self._open_positions.pop(trade_id)
        direction = pos['direction']
        risk_frac = pos['risk']
        
        # Update exposure
        if direction == "LONG":
            self.state.current_long_exposure -= risk_frac
        else:
            self.state.current_short_exposure -= risk_frac
        
        # Update PnL
        self.state.daily_pnl += pnl
        self.state.weekly_pnl += pnl
        self.state.current_equity += pnl
        
        # Update streaks
        if won:
            self.state.consecutive_wins += 1
            self.state.consecutive_losses = 0
        else:
            self.state.consecutive_losses += 1
            self.state.consecutive_wins = 0
    
    def _trigger_halt(self, reason: KillSwitchReason, halt_until: Optional[datetime]) -> None:
        """Trigger kill switch."""
        self.state.is_halted = True
        self.state.halt_reason = reason
        self.state.halt_until = halt_until
    
    def get_status(self) -> dict:
        """Get current risk status."""
        return {
            'equity': self.state.current_equity,
            'drawdown_pct': self.state.drawdown_pct,
            'daily_pnl': self.state.daily_pnl,
            'daily_risk_used': self.state.daily_risk_used,
            'daily_risk_remaining': self.MAX_DAILY_RISK - self.state.daily_risk_used,
            'weekly_pnl': self.state.weekly_pnl,
            'consecutive_losses': self.state.consecutive_losses,
            'consecutive_wins': self.state.consecutive_wins,
            'open_positions': len(self._open_positions),
            'long_exposure': self.state.current_long_exposure,
            'short_exposure': self.state.current_short_exposure,
            'is_halted': self.state.is_halted,
            'halt_reason': self.state.halt_reason.value if self.state.halt_reason else None,
        }

