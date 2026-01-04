"""
Risk Management Module.

Handles position sizing, exposure limits, and drawdown control.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import logging

from config.settings import Settings


logger = logging.getLogger(__name__)


@dataclass
class PositionInfo:
    """Information about an open position."""
    pair: Tuple[str, str]
    direction: str  # 'long_spread' or 'short_spread'
    size_a: float
    size_b: float
    entry_time: datetime
    entry_pnl: float = 0.0
    current_pnl: float = 0.0


@dataclass
class RiskState:
    """Current risk state."""
    total_exposure: float
    open_pairs: int
    daily_pnl: float
    daily_trades: int
    current_drawdown: float
    peak_equity: float
    is_halted: bool = False
    halt_reason: str = ""


class RiskManager:
    """
    Manages risk for the pairs trading system.
    
    Features:
    - Position sizing with hedge ratio balancing
    - Exposure limits
    - Drawdown monitoring
    - Daily loss limits
    """
    
    def __init__(
        self,
        settings: Settings,
        initial_capital: float
    ):
        """
        Initialize risk manager.
        
        Args:
            settings: Trading settings
            initial_capital: Starting capital
        """
        self.settings = settings
        self.risk_settings = settings.risk
        
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        
        # Position tracking
        self.positions: Dict[Tuple[str, str], PositionInfo] = {}
        
        # Daily tracking
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.last_reset = datetime.now()
        
        # State
        self.is_halted = False
        self.halt_reason = ""
    
    def can_open_position(
        self,
        pair: Tuple[str, str],
        size_usd: Optional[float] = None
    ) -> bool:
        """
        Check if a new position can be opened.
        
        Args:
            pair: Pair to trade
            size_usd: Proposed position size in USD
            
        Returns:
            True if position can be opened
        """
        # Check if trading is halted
        if self.is_halted:
            logger.warning(f"Trading halted: {self.halt_reason}")
            return False
        
        # Check pair limit
        if len(self.positions) >= self.risk_settings.max_open_pairs:
            logger.warning(f"Max pairs reached: {len(self.positions)}")
            return False
        
        # Check if already in position
        if pair in self.positions:
            logger.warning(f"Already in position for {pair}")
            return False
        
        # Check daily trade limit
        self._check_daily_reset()
        if self.daily_trades >= self.risk_settings.max_daily_trades:
            logger.warning("Daily trade limit reached")
            return False
        
        # Check daily loss limit
        daily_loss_limit = self.current_capital * self.risk_settings.max_daily_loss
        if self.daily_pnl < -daily_loss_limit:
            logger.warning(f"Daily loss limit reached: {self.daily_pnl:.2f}")
            return False
        
        # Check total exposure
        if size_usd:
            current_exposure = self._calculate_total_exposure()
            max_exposure = self.current_capital * self.risk_settings.max_total_exposure
            
            if current_exposure + size_usd > max_exposure:
                logger.warning(f"Exposure limit exceeded")
                return False
        
        return True
    
    def calculate_position_size(
        self,
        pair: Tuple[str, str],
        hedge_ratio: float,
        stop_loss_pct: float = 0.02
    ) -> Tuple[float, float]:
        """
        Calculate position sizes for both legs.
        
        Args:
            pair: Pair to trade
            hedge_ratio: Hedge ratio (β)
            stop_loss_pct: Expected stop loss percentage
            
        Returns:
            (size_a, size_b) in lots
        """
        # Risk amount per trade
        risk_amount = self.current_capital * self.risk_settings.max_risk_per_trade
        
        # Split risk between legs if balancing
        if self.risk_settings.balance_legs:
            risk_per_leg = risk_amount / 2
        else:
            risk_per_leg = risk_amount
        
        # Calculate notional size based on risk
        # Assuming ~$10 per pip per standard lot for majors
        pip_value = 10.0  # Approximate
        stop_pips = stop_loss_pct * 100 * 100  # Convert to pips
        
        if stop_pips <= 0:
            stop_pips = 30  # Default 30 pips
        
        base_size = risk_per_leg / (stop_pips * pip_value)
        
        # Apply hedge ratio
        size_a = base_size
        size_b = base_size * hedge_ratio
        
        # Round to lot step (0.01)
        size_a = round(size_a, 2)
        size_b = round(size_b, 2)
        
        # Minimum lot
        size_a = max(0.01, size_a)
        size_b = max(0.01, size_b)
        
        return size_a, size_b
    
    def record_entry(
        self,
        pair: Tuple[str, str],
        direction: str,
        size_a: float,
        size_b: float
    ):
        """
        Record a new position entry.
        
        Args:
            pair: Pair traded
            direction: 'long_spread' or 'short_spread'
            size_a: Size of leg A
            size_b: Size of leg B
        """
        self.positions[pair] = PositionInfo(
            pair=pair,
            direction=direction,
            size_a=size_a,
            size_b=size_b,
            entry_time=datetime.now()
        )
        
        self.daily_trades += 1
        
        logger.info(f"Position recorded: {pair} {direction}")
    
    def record_exit(
        self,
        pair: Tuple[str, str],
        pnl: float
    ):
        """
        Record position exit.
        
        Args:
            pair: Pair closed
            pnl: Realized P&L
        """
        if pair in self.positions:
            del self.positions[pair]
        
        self.daily_pnl += pnl
        self.current_capital += pnl
        
        # Update peak
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        
        # Check drawdown
        self._check_drawdown()
        
        logger.info(f"Position closed: {pair}, P&L: {pnl:.2f}")
    
    def update_pnl(
        self,
        pair: Tuple[str, str],
        unrealized_pnl: float
    ):
        """
        Update unrealized P&L for a position.
        
        Args:
            pair: Pair to update
            unrealized_pnl: Current unrealized P&L
        """
        if pair in self.positions:
            self.positions[pair].current_pnl = unrealized_pnl
    
    def _calculate_total_exposure(self) -> float:
        """Calculate total exposure across all positions."""
        # Simplified: sum of position sizes * approximate contract value
        exposure = 0.0
        
        for pos in self.positions.values():
            # Assume 100k contract size
            exposure += pos.size_a * 100000
            exposure += pos.size_b * 100000
        
        return exposure
    
    def _check_drawdown(self):
        """Check and enforce drawdown limits."""
        if self.peak_capital <= 0:
            return
        
        drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        
        if drawdown >= self.risk_settings.max_drawdown:
            if self.risk_settings.drawdown_halt:
                self.halt_trading(f"Max drawdown reached: {drawdown:.1%}")
    
    def _check_daily_reset(self):
        """Reset daily counters if new day."""
        now = datetime.now()
        
        if now.date() > self.last_reset.date():
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.last_reset = now
            
            # Resume trading if halted for daily loss
            if "daily" in self.halt_reason.lower():
                self.resume_trading()
            
            logger.info("Daily counters reset")
    
    def halt_trading(self, reason: str):
        """Halt all trading."""
        self.is_halted = True
        self.halt_reason = reason
        logger.warning(f"Trading halted: {reason}")
    
    def resume_trading(self):
        """Resume trading."""
        self.is_halted = False
        self.halt_reason = ""
        logger.info("Trading resumed")
    
    def get_state(self) -> RiskState:
        """Get current risk state."""
        self._check_daily_reset()
        
        drawdown = 0.0
        if self.peak_capital > 0:
            drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        
        return RiskState(
            total_exposure=self._calculate_total_exposure(),
            open_pairs=len(self.positions),
            daily_pnl=self.daily_pnl,
            daily_trades=self.daily_trades,
            current_drawdown=drawdown,
            peak_equity=self.peak_capital,
            is_halted=self.is_halted,
            halt_reason=self.halt_reason
        )
    
    def get_risk_summary(self) -> str:
        """Get human-readable risk summary."""
        state = self.get_state()
        
        return f"""
Risk Summary
============
Capital: ${self.current_capital:,.2f} (Peak: ${self.peak_capital:,.2f})
Drawdown: {state.current_drawdown:.1%}
Exposure: ${state.total_exposure:,.0f}
Open Pairs: {state.open_pairs} / {self.risk_settings.max_open_pairs}
Daily P&L: ${state.daily_pnl:.2f}
Daily Trades: {state.daily_trades} / {self.risk_settings.max_daily_trades}
Status: {"HALTED - " + state.halt_reason if state.is_halted else "Active"}
"""
    
    def check_emergency_exit(
        self,
        pair: Tuple[str, str],
        unrealized_pnl: float
    ) -> bool:
        """
        Check if emergency exit is needed for a position.
        
        Args:
            pair: Pair to check
            unrealized_pnl: Current unrealized P&L
            
        Returns:
            True if emergency exit needed
        """
        # Check position-level stop
        max_loss = self.current_capital * self.risk_settings.max_risk_per_trade * 2
        
        if unrealized_pnl < -max_loss:
            logger.warning(f"Emergency exit triggered for {pair}: {unrealized_pnl:.2f}")
            return True
        
        return False
    
    def validate_trade(
        self,
        pair: Tuple[str, str],
        size_a: float,
        size_b: float
    ) -> Tuple[bool, str]:
        """
        Validate a proposed trade.
        
        Args:
            pair: Pair to trade
            size_a: Proposed size for leg A
            size_b: Proposed size for leg B
            
        Returns:
            (is_valid, reason) tuple
        """
        # Check basic permissions
        if not self.can_open_position(pair):
            return False, "Cannot open position"
        
        # Check minimum size
        if size_a < 0.01 or size_b < 0.01:
            return False, "Position size too small"
        
        # Check maximum size
        max_lot = 10.0  # Arbitrary max
        if size_a > max_lot or size_b > max_lot:
            return False, "Position size too large"
        
        return True, "Trade validated"
