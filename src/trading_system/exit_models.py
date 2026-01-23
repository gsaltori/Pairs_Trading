"""
Session Strategy Exit Models

Multiple exit strategies for Asia → London expansion:
1. BASELINE: Single TP at 2.5R, time stop at London end
2. MULTI_TARGET: Scaled exit (50% at 1.0R, 30% at 2.0R, 20% runner)
3. REDUCED_TP: Single TP at 1.5R
4. AGGRESSIVE: Single TP at 1.0R (high probability)

The problem: 94% time stops, 0% TP hits means R=2.5 is unrealistic
for typical London expansion magnitude.
"""

from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, time as dt_time
from typing import Optional, List, Tuple


class ExitModel(Enum):
    BASELINE = "BASELINE"           # 2.5R single target
    MULTI_TARGET = "MULTI_TARGET"   # 1.0R/2.0R/runner scaled
    REDUCED_TP = "REDUCED_TP"       # 1.5R single target
    AGGRESSIVE = "AGGRESSIVE"       # 1.0R single target


@dataclass
class PositionSlice:
    """A portion of a position with its own target."""
    slice_id: int
    size_pct: float  # Percentage of total position
    target_r: float  # Target in R-multiples
    is_runner: bool = False  # Runner uses trailing or time exit
    
    # State
    is_open: bool = True
    exit_price: Optional[float] = None
    exit_reason: str = ""
    pnl_r: float = 0.0


@dataclass
class ExitConfig:
    """Configuration for an exit model."""
    model: ExitModel
    slices: List[PositionSlice]
    use_breakeven_stop: bool = False  # Move SL to entry after first TP
    breakeven_buffer_r: float = 0.1   # Buffer above entry for BE stop
    runner_trail_r: float = 0.5       # Trail distance for runner (in R)
    
    @classmethod
    def baseline(cls) -> 'ExitConfig':
        """Original 2.5R single target."""
        return cls(
            model=ExitModel.BASELINE,
            slices=[
                PositionSlice(slice_id=0, size_pct=1.0, target_r=2.5),
            ],
        )
    
    @classmethod
    def multi_target(cls) -> 'ExitConfig':
        """Scaled exit: 50% at 1.0R, 30% at 2.0R, 20% runner."""
        return cls(
            model=ExitModel.MULTI_TARGET,
            slices=[
                PositionSlice(slice_id=0, size_pct=0.50, target_r=1.0),
                PositionSlice(slice_id=1, size_pct=0.30, target_r=2.0),
                PositionSlice(slice_id=2, size_pct=0.20, target_r=999.0, is_runner=True),
            ],
            use_breakeven_stop=True,
            breakeven_buffer_r=0.1,
            runner_trail_r=0.5,
        )
    
    @classmethod
    def reduced_tp(cls) -> 'ExitConfig':
        """Single 1.5R target."""
        return cls(
            model=ExitModel.REDUCED_TP,
            slices=[
                PositionSlice(slice_id=0, size_pct=1.0, target_r=1.5),
            ],
        )
    
    @classmethod
    def aggressive(cls) -> 'ExitConfig':
        """Single 1.0R target (high probability)."""
        return cls(
            model=ExitModel.AGGRESSIVE,
            slices=[
                PositionSlice(slice_id=0, size_pct=1.0, target_r=1.0),
            ],
        )


class ExitManager:
    """
    Manages position exits with configurable models.
    
    Tracks multiple position slices, handles:
    - Multiple take profit levels
    - Breakeven stop movement
    - Runner trailing stop
    - Time stop at session end
    """
    
    LONDON_END = dt_time(11, 0)
    
    def __init__(self, config: ExitConfig):
        self.config = config
        self.reset()
    
    def reset(self):
        """Reset for new trade."""
        self._entry_price: float = 0.0
        self._original_sl: float = 0.0
        self._current_sl: float = 0.0
        self._direction: str = ""
        self._sl_distance: float = 0.0
        self._is_active: bool = False
        self._highest_r: float = 0.0  # Highest R reached (for trailing)
        self._slices: List[PositionSlice] = []
        self._breakeven_activated: bool = False
    
    def open_position(
        self,
        entry_price: float,
        stop_loss: float,
        direction: str,  # "LONG" or "SHORT"
    ) -> None:
        """Initialize position with entry parameters."""
        self._entry_price = entry_price
        self._original_sl = stop_loss
        self._current_sl = stop_loss
        self._direction = direction
        self._sl_distance = abs(entry_price - stop_loss)
        self._is_active = True
        self._highest_r = 0.0
        self._breakeven_activated = False
        
        # Create fresh slices
        self._slices = []
        for s in self.config.slices:
            self._slices.append(PositionSlice(
                slice_id=s.slice_id,
                size_pct=s.size_pct,
                target_r=s.target_r,
                is_runner=s.is_runner,
                is_open=True,
            ))
    
    @property
    def is_active(self) -> bool:
        return self._is_active
    
    @property
    def current_sl(self) -> float:
        return self._current_sl
    
    @property
    def slices(self) -> List[PositionSlice]:
        return self._slices
    
    def _price_to_r(self, price: float) -> float:
        """Convert price to R-multiple from entry."""
        if self._sl_distance <= 0:
            return 0.0
        
        if self._direction == "LONG":
            return (price - self._entry_price) / self._sl_distance
        else:
            return (self._entry_price - price) / self._sl_distance
    
    def _r_to_price(self, r: float) -> float:
        """Convert R-multiple to price."""
        if self._direction == "LONG":
            return self._entry_price + r * self._sl_distance
        else:
            return self._entry_price - r * self._sl_distance
    
    def update(
        self,
        high: float,
        low: float,
        close: float,
        timestamp: datetime,
    ) -> Tuple[bool, List[PositionSlice], str]:
        """
        Update position with new bar data.
        
        Returns:
            (position_closed, closed_slices, final_reason)
        """
        if not self._is_active:
            return True, [], "NOT_ACTIVE"
        
        closed_slices = []
        
        # Calculate current R excursion
        if self._direction == "LONG":
            best_price = high
            worst_price = low
        else:
            best_price = low
            worst_price = high
        
        current_r = self._price_to_r(best_price)
        
        # Update highest R for trailing
        if current_r > self._highest_r:
            self._highest_r = current_r
        
        # Check stop loss FIRST (worst case)
        if self._hit_stop(worst_price):
            # Close all remaining slices at SL
            for s in self._slices:
                if s.is_open:
                    s.is_open = False
                    s.exit_price = self._current_sl
                    s.exit_reason = "SL"
                    s.pnl_r = self._price_to_r(self._current_sl)
                    closed_slices.append(s)
            
            self._is_active = False
            return True, closed_slices, "SL"
        
        # Check take profits for each slice
        for s in self._slices:
            if not s.is_open:
                continue
            
            if s.is_runner:
                # Runner uses trailing stop
                self._update_trailing_stop()
                continue
            
            # Check if target hit
            target_price = self._r_to_price(s.target_r)
            
            if self._direction == "LONG" and high >= target_price:
                s.is_open = False
                s.exit_price = target_price
                s.exit_reason = f"TP{s.slice_id + 1}"
                s.pnl_r = s.target_r
                closed_slices.append(s)
                
                # Activate breakeven after first TP
                if self.config.use_breakeven_stop and not self._breakeven_activated:
                    self._activate_breakeven()
            
            elif self._direction == "SHORT" and low <= target_price:
                s.is_open = False
                s.exit_price = target_price
                s.exit_reason = f"TP{s.slice_id + 1}"
                s.pnl_r = s.target_r
                closed_slices.append(s)
                
                if self.config.use_breakeven_stop and not self._breakeven_activated:
                    self._activate_breakeven()
        
        # Check time stop
        if timestamp.time() >= self.LONDON_END:
            # Close all remaining slices at close price
            for s in self._slices:
                if s.is_open:
                    s.is_open = False
                    s.exit_price = close
                    s.exit_reason = "TIME"
                    s.pnl_r = self._price_to_r(close)
                    closed_slices.append(s)
            
            self._is_active = False
            return True, closed_slices, "TIME"
        
        # Check if all slices closed
        all_closed = all(not s.is_open for s in self._slices)
        if all_closed:
            self._is_active = False
            return True, closed_slices, "ALL_TP"
        
        return False, closed_slices, ""
    
    def _hit_stop(self, worst_price: float) -> bool:
        """Check if stop loss was hit."""
        if self._direction == "LONG":
            return worst_price <= self._current_sl
        else:
            return worst_price >= self._current_sl
    
    def _activate_breakeven(self) -> None:
        """Move stop to breakeven + buffer after first TP."""
        if self._breakeven_activated:
            return
        
        buffer = self.config.breakeven_buffer_r * self._sl_distance
        
        if self._direction == "LONG":
            new_sl = self._entry_price + buffer
            # Only move if better than current
            if new_sl > self._current_sl:
                self._current_sl = new_sl
        else:
            new_sl = self._entry_price - buffer
            if new_sl < self._current_sl:
                self._current_sl = new_sl
        
        self._breakeven_activated = True
    
    def _update_trailing_stop(self) -> None:
        """Update trailing stop for runner slice."""
        if not self._breakeven_activated:
            return
        
        trail_distance = self.config.runner_trail_r * self._sl_distance
        
        if self._direction == "LONG":
            trail_price = self._r_to_price(self._highest_r) - trail_distance
            if trail_price > self._current_sl:
                self._current_sl = trail_price
        else:
            trail_price = self._r_to_price(self._highest_r) + trail_distance
            if trail_price < self._current_sl:
                self._current_sl = trail_price
    
    def force_close(self, price: float, reason: str) -> List[PositionSlice]:
        """Force close all remaining slices."""
        closed = []
        for s in self._slices:
            if s.is_open:
                s.is_open = False
                s.exit_price = price
                s.exit_reason = reason
                s.pnl_r = self._price_to_r(price)
                closed.append(s)
        
        self._is_active = False
        return closed
    
    def get_total_pnl_r(self) -> float:
        """Calculate weighted total PnL in R."""
        total = 0.0
        for s in self._slices:
            if not s.is_open:
                total += s.pnl_r * s.size_pct
        return total
    
    def get_exit_summary(self) -> dict:
        """Get summary of all slice exits."""
        return {
            'model': self.config.model.value,
            'slices': [
                {
                    'id': s.slice_id,
                    'size_pct': s.size_pct,
                    'target_r': s.target_r,
                    'exit_reason': s.exit_reason,
                    'pnl_r': s.pnl_r,
                }
                for s in self._slices
            ],
            'total_pnl_r': self.get_total_pnl_r(),
            'breakeven_activated': self._breakeven_activated,
            'highest_r_reached': self._highest_r,
        }
