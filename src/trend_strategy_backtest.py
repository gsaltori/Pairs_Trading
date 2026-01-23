"""
Trend-Following Strategy with Gatekeeper Integration

STRATEGY SPECIFICATION (LOCKED):
- Instrument: EURUSD H4
- Direction: EMA200 filter (long if Close > EMA200, short if Close < EMA200)
- Entry: Pullback to EMA50, entry on close in trend direction
- Stop Loss: ATR(14) × 1.5
- Take Profit: RR = 2.0
- Risk: Fixed 1% per trade

GATEKEEPER (LOCKED):
- Block if |Z-score| > 3.0
- Block if correlation_trend == DETERIORATING
- Block if volatility_ratio == COMPRESSED
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from crv_engine.experiments.trade_gatekeeper import (
    TradeGatekeeper,
    TradePermission,
    BlockReason,
    check_trade_permission,
)


# =============================================================================
# CONSTANTS (LOCKED - DO NOT MODIFY)
# =============================================================================

# Strategy parameters
EMA_FAST = 50
EMA_SLOW = 200
ATR_PERIOD = 14
ATR_MULTIPLIER = 1.5
RISK_REWARD = 2.0
RISK_PER_TRADE = 0.01  # 1%

# Gatekeeper observables calculation
ZSCORE_WINDOW = 60
CORRELATION_WINDOW = 60
VOLATILITY_WINDOW = 20


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class TradeDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class TradeStatus(Enum):
    OPEN = "OPEN"
    WIN = "WIN"
    LOSS = "LOSS"


@dataclass
class Trade:
    """Single trade record."""
    entry_bar: int
    entry_time: datetime
    direction: TradeDirection
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    
    # Exit info (filled on close)
    exit_bar: Optional[int] = None
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    status: TradeStatus = TradeStatus.OPEN
    pnl: float = 0.0
    pnl_pct: float = 0.0
    
    # Gatekeeper info
    was_blocked: bool = False
    block_reasons: Tuple[str, ...] = field(default_factory=tuple)
    
    # Observables at entry
    zscore_at_entry: Optional[float] = None
    correlation_trend_at_entry: Optional[float] = None
    volatility_ratio_at_entry: Optional[float] = None
    
    @property
    def bars_held(self) -> int:
        if self.exit_bar is None:
            return 0
        return self.exit_bar - self.entry_bar
    
    @property
    def risk_amount(self) -> float:
        return abs(self.entry_price - self.stop_loss) * self.position_size


# =============================================================================
# INDICATOR CALCULATIONS
# =============================================================================

def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """Calculate Exponential Moving Average."""
    ema = np.zeros_like(prices)
    ema[:period] = np.nan
    
    if len(prices) < period:
        return ema
    
    # Initial SMA
    ema[period - 1] = np.mean(prices[:period])
    
    # EMA calculation
    multiplier = 2.0 / (period + 1)
    for i in range(period, len(prices)):
        ema[i] = (prices[i] - ema[i-1]) * multiplier + ema[i-1]
    
    return ema


def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Calculate Average True Range."""
    n = len(close)
    tr = np.zeros(n)
    atr = np.zeros(n)
    atr[:period] = np.nan
    
    # True Range
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    # Initial ATR (SMA of TR)
    if n >= period:
        atr[period - 1] = np.mean(tr[:period])
        
        # Smoothed ATR
        for i in range(period, n):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
    return atr


# =============================================================================
# GATEKEEPER OBSERVABLES
# =============================================================================

class GatekeeperObservables:
    """
    Computes observables required by the gatekeeper from EURUSD/GBPUSD data.
    
    Observables:
    - Z-score of spread
    - Correlation trend
    - Volatility ratio
    """
    
    def __init__(
        self,
        zscore_window: int = ZSCORE_WINDOW,
        correlation_window: int = CORRELATION_WINDOW,
        volatility_window: int = VOLATILITY_WINDOW,
    ):
        self.zscore_window = zscore_window
        self.correlation_window = correlation_window
        self.volatility_window = volatility_window
        
        # Price history
        self._prices_a: List[float] = []  # EURUSD
        self._prices_b: List[float] = []  # GBPUSD
        
        # Correlation history for trend
        self._correlation_history: List[float] = []
    
    def update(self, price_a: float, price_b: float) -> None:
        """Update with new prices."""
        self._prices_a.append(price_a)
        self._prices_b.append(price_b)
        
        # Compute and store correlation
        if len(self._prices_a) >= self.correlation_window:
            corr = self._compute_correlation()
            self._correlation_history.append(corr)
        
        # Trim history
        max_history = max(self.zscore_window, self.correlation_window, self.volatility_window) + 50
        if len(self._prices_a) > max_history:
            self._prices_a = self._prices_a[-max_history:]
            self._prices_b = self._prices_b[-max_history:]
        if len(self._correlation_history) > 50:
            self._correlation_history = self._correlation_history[-50:]
    
    def _compute_correlation(self) -> float:
        """Compute rolling correlation."""
        if len(self._prices_a) < self.correlation_window:
            return 0.0
        
        a = np.array(self._prices_a[-self.correlation_window:])
        b = np.array(self._prices_b[-self.correlation_window:])
        
        corr = np.corrcoef(a, b)[0, 1]
        return float(corr) if not np.isnan(corr) else 0.0
    
    def get_zscore(self) -> float:
        """Get current spread Z-score."""
        if len(self._prices_a) < self.zscore_window:
            return 0.0
        
        a = np.array(self._prices_a[-self.zscore_window:])
        b = np.array(self._prices_b[-self.zscore_window:])
        
        # Compute hedge ratio
        cov = np.cov(a, b)[0, 1]
        var_b = np.var(b)
        if var_b < 1e-10:
            return 0.0
        hedge_ratio = cov / var_b
        hedge_ratio = np.clip(hedge_ratio, 0.1, 10.0)
        
        # Compute spread
        spread = a - hedge_ratio * b
        spread_mean = np.mean(spread)
        spread_std = np.std(spread)
        
        if spread_std < 1e-10:
            return 0.0
        
        current_spread = self._prices_a[-1] - hedge_ratio * self._prices_b[-1]
        zscore = (current_spread - spread_mean) / spread_std
        
        return float(np.clip(zscore, -10, 10))
    
    def get_correlation_trend(self) -> float:
        """Get correlation trend (change over recent period)."""
        if len(self._correlation_history) < 10:
            return 0.0
        
        recent = self._correlation_history[-10:]
        return recent[-1] - recent[0]
    
    def get_volatility_ratio(self) -> float:
        """Get volatility ratio (vol_A / vol_B)."""
        if len(self._prices_a) < self.volatility_window:
            return 1.0
        
        a = np.array(self._prices_a[-self.volatility_window:])
        b = np.array(self._prices_b[-self.volatility_window:])
        
        returns_a = np.diff(a) / a[:-1]
        returns_b = np.diff(b) / b[:-1]
        
        vol_a = np.std(returns_a) if len(returns_a) > 0 else 0.01
        vol_b = np.std(returns_b) if len(returns_b) > 0 else 0.01
        
        if vol_b < 1e-10:
            return 1.0
        
        return float(vol_a / vol_b)
    
    def get_correlation(self) -> float:
        """Get current correlation level."""
        return self._compute_correlation()
    
    def is_ready(self) -> bool:
        """Check if enough data for valid observables."""
        return len(self._prices_a) >= self.zscore_window


# =============================================================================
# STRATEGY ENGINE
# =============================================================================

class TrendFollowingStrategy:
    """
    Simple trend-following strategy on EURUSD H4.
    
    Rules:
    - Long only if Close > EMA200
    - Short only if Close < EMA200
    - Entry: Pullback to EMA50
    - SL: ATR(14) × 1.5
    - TP: RR = 2.0
    """
    
    def __init__(
        self,
        use_gatekeeper: bool = False,
        initial_capital: float = 100000.0,
    ):
        self.use_gatekeeper = use_gatekeeper
        self.initial_capital = initial_capital
        self.equity = initial_capital
        
        # Components
        self.gatekeeper = TradeGatekeeper() if use_gatekeeper else None
        self.observables = GatekeeperObservables()
        
        # State
        self.trades: List[Trade] = []
        self.current_trade: Optional[Trade] = None
        self.equity_curve: List[float] = []
        
        # Blocked trades (for counterfactual)
        self.blocked_trades: List[Trade] = []
    
    def run_backtest(
        self,
        df_eurusd: pd.DataFrame,
        df_gbpusd: pd.DataFrame,
    ) -> Dict:
        """
        Run backtest on EURUSD with GBPUSD for gatekeeper observables.
        
        DataFrames must have: timestamp, open, high, low, close
        """
        # Validate alignment
        assert len(df_eurusd) == len(df_gbpusd), "Data length mismatch"
        
        n = len(df_eurusd)
        
        # Calculate indicators
        close_eu = df_eurusd['close'].values
        high_eu = df_eurusd['high'].values
        low_eu = df_eurusd['low'].values
        close_gb = df_gbpusd['close'].values
        
        ema50 = calculate_ema(close_eu, EMA_FAST)
        ema200 = calculate_ema(close_eu, EMA_SLOW)
        atr = calculate_atr(high_eu, low_eu, close_eu, ATR_PERIOD)
        
        # Track previous bar state for pullback detection
        prev_above_ema50 = None
        
        # Main loop
        for i in range(EMA_SLOW, n):
            # Update gatekeeper observables
            self.observables.update(close_eu[i], close_gb[i])
            
            # Get current values
            price = close_eu[i]
            current_ema50 = ema50[i]
            current_ema200 = ema200[i]
            current_atr = atr[i]
            timestamp = df_eurusd.iloc[i]['timestamp']
            
            # Update equity curve
            self.equity_curve.append(self.equity)
            
            # Check for exit if in trade
            if self.current_trade is not None:
                self._check_exit(i, high_eu[i], low_eu[i], price, timestamp)
            
            # Skip if already in trade or ATR invalid
            if self.current_trade is not None or np.isnan(current_atr) or current_atr <= 0:
                prev_above_ema50 = price > current_ema50
                continue
            
            # Determine trend direction
            if price > current_ema200:
                trend = TradeDirection.LONG
            elif price < current_ema200:
                trend = TradeDirection.SHORT
            else:
                prev_above_ema50 = price > current_ema50
                continue
            
            # Check for pullback to EMA50
            current_above_ema50 = price > current_ema50
            
            if prev_above_ema50 is not None:
                # Long entry: was below EMA50, now above, in uptrend
                long_signal = (
                    trend == TradeDirection.LONG and
                    not prev_above_ema50 and
                    current_above_ema50
                )
                
                # Short entry: was above EMA50, now below, in downtrend
                short_signal = (
                    trend == TradeDirection.SHORT and
                    prev_above_ema50 and
                    not current_above_ema50
                )
                
                if long_signal or short_signal:
                    self._attempt_entry(
                        bar_index=i,
                        timestamp=timestamp,
                        direction=trend,
                        price=price,
                        atr=current_atr,
                    )
            
            prev_above_ema50 = current_above_ema50
        
        # Close any open trade at end
        if self.current_trade is not None:
            self._force_close(n - 1, close_eu[-1], df_eurusd.iloc[-1]['timestamp'])
        
        return self._compute_results()
    
    def _attempt_entry(
        self,
        bar_index: int,
        timestamp: datetime,
        direction: TradeDirection,
        price: float,
        atr: float,
    ) -> None:
        """Attempt to enter a trade, applying gatekeeper if enabled."""
        
        # Calculate SL/TP
        sl_distance = atr * ATR_MULTIPLIER
        tp_distance = sl_distance * RISK_REWARD
        
        if direction == TradeDirection.LONG:
            stop_loss = price - sl_distance
            take_profit = price + tp_distance
        else:
            stop_loss = price + sl_distance
            take_profit = price - tp_distance
        
        # Calculate position size (fixed 1% risk)
        risk_amount = self.equity * RISK_PER_TRADE
        position_size = risk_amount / sl_distance
        
        # Get gatekeeper observables
        zscore = self.observables.get_zscore()
        corr_trend = self.observables.get_correlation_trend()
        vol_ratio = self.observables.get_volatility_ratio()
        
        # Create trade object
        trade = Trade(
            entry_bar=bar_index,
            entry_time=timestamp,
            direction=direction,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            zscore_at_entry=zscore,
            correlation_trend_at_entry=corr_trend,
            volatility_ratio_at_entry=vol_ratio,
        )
        
        # Apply gatekeeper
        if self.use_gatekeeper and self.observables.is_ready():
            allowed = check_trade_permission(
                zscore=zscore,
                correlation_trend=corr_trend,
                volatility_ratio=vol_ratio,
            )
            
            if not allowed:
                # Determine block reasons
                reasons = []
                if abs(zscore) > 3.0:
                    reasons.append("EXTREME_SPREAD")
                if corr_trend < -0.05:
                    reasons.append("DETERIORATING_CORRELATION")
                if vol_ratio < 0.7:
                    reasons.append("COMPRESSED_VOLATILITY")
                
                trade.was_blocked = True
                trade.block_reasons = tuple(reasons)
                self.blocked_trades.append(trade)
                return
        
        # Execute trade
        self.current_trade = trade
        self.trades.append(trade)
    
    def _check_exit(
        self,
        bar_index: int,
        high: float,
        low: float,
        close: float,
        timestamp: datetime,
    ) -> None:
        """Check for SL/TP hit."""
        trade = self.current_trade
        
        if trade.direction == TradeDirection.LONG:
            # Check SL first (conservative)
            if low <= trade.stop_loss:
                self._close_trade(bar_index, trade.stop_loss, timestamp, TradeStatus.LOSS)
            elif high >= trade.take_profit:
                self._close_trade(bar_index, trade.take_profit, timestamp, TradeStatus.WIN)
        else:  # SHORT
            if high >= trade.stop_loss:
                self._close_trade(bar_index, trade.stop_loss, timestamp, TradeStatus.LOSS)
            elif low <= trade.take_profit:
                self._close_trade(bar_index, trade.take_profit, timestamp, TradeStatus.WIN)
    
    def _close_trade(
        self,
        bar_index: int,
        exit_price: float,
        timestamp: datetime,
        status: TradeStatus,
    ) -> None:
        """Close the current trade."""
        trade = self.current_trade
        trade.exit_bar = bar_index
        trade.exit_time = timestamp
        trade.exit_price = exit_price
        trade.status = status
        
        # Calculate PnL
        if trade.direction == TradeDirection.LONG:
            trade.pnl = (exit_price - trade.entry_price) * trade.position_size
        else:
            trade.pnl = (trade.entry_price - exit_price) * trade.position_size
        
        trade.pnl_pct = trade.pnl / self.equity
        
        # Update equity
        self.equity += trade.pnl
        
        self.current_trade = None
    
    def _force_close(self, bar_index: int, price: float, timestamp: datetime) -> None:
        """Force close at end of backtest."""
        trade = self.current_trade
        
        if trade.direction == TradeDirection.LONG:
            pnl = (price - trade.entry_price) * trade.position_size
        else:
            pnl = (trade.entry_price - price) * trade.position_size
        
        status = TradeStatus.WIN if pnl > 0 else TradeStatus.LOSS
        self._close_trade(bar_index, price, timestamp, status)
    
    def _compute_results(self) -> Dict:
        """Compute backtest results."""
        if not self.trades:
            return self._empty_results()
        
        # Basic stats
        total_trades = len(self.trades)
        wins = [t for t in self.trades if t.status == TradeStatus.WIN]
        losses = [t for t in self.trades if t.status == TradeStatus.LOSS]
        
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = win_count / total_trades if total_trades > 0 else 0
        
        # PnL stats
        total_pnl = sum(t.pnl for t in self.trades)
        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0.0001
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Expectancy
        avg_win = np.mean([t.pnl for t in wins]) if wins else 0
        avg_loss = abs(np.mean([t.pnl for t in losses])) if losses else 0.0001
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        # Drawdown
        equity_array = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (running_max - equity_array) / running_max
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        
        # Trade duration
        durations = [t.bars_held for t in self.trades if t.exit_bar is not None]
        avg_duration = np.mean(durations) if durations else 0
        
        # Net return
        net_return = (self.equity - self.initial_capital) / self.initial_capital
        
        return {
            'total_trades': total_trades,
            'win_count': win_count,
            'loss_count': loss_count,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'max_drawdown': max_drawdown,
            'net_return': net_return,
            'total_pnl': total_pnl,
            'avg_trade': total_pnl / total_trades if total_trades > 0 else 0,
            'avg_duration_bars': avg_duration,
            'final_equity': self.equity,
            'equity_curve': self.equity_curve,
            'trades': self.trades,
            'blocked_trades': self.blocked_trades,
        }
    
    def _empty_results(self) -> Dict:
        """Return empty results structure."""
        return {
            'total_trades': 0,
            'win_count': 0,
            'loss_count': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'expectancy': 0,
            'max_drawdown': 0,
            'net_return': 0,
            'total_pnl': 0,
            'avg_trade': 0,
            'avg_duration_bars': 0,
            'final_equity': self.initial_capital,
            'equity_curve': self.equity_curve,
            'trades': [],
            'blocked_trades': [],
        }


# =============================================================================
# COUNTERFACTUAL ANALYSIS
# =============================================================================

def analyze_blocked_trades(
    blocked_trades: List[Trade],
    df_eurusd: pd.DataFrame,
) -> Dict:
    """
    Counterfactual analysis: what would have happened if blocked trades executed?
    """
    if not blocked_trades:
        return {
            'total_blocked': 0,
            'would_have_won': 0,
            'would_have_lost': 0,
            'would_have_pnl': 0,
            'counterfactual_win_rate': 0,
        }
    
    high = df_eurusd['high'].values
    low = df_eurusd['low'].values
    close = df_eurusd['close'].values
    n = len(close)
    
    wins = 0
    losses = 0
    total_pnl = 0.0
    
    for trade in blocked_trades:
        entry_bar = trade.entry_bar
        sl = trade.stop_loss
        tp = trade.take_profit
        direction = trade.direction
        
        # Simulate trade outcome
        for i in range(entry_bar + 1, min(entry_bar + 100, n)):  # Max 100 bars
            if direction == TradeDirection.LONG:
                if low[i] <= sl:
                    losses += 1
                    total_pnl -= trade.risk_amount
                    break
                elif high[i] >= tp:
                    wins += 1
                    total_pnl += trade.risk_amount * RISK_REWARD
                    break
            else:
                if high[i] >= sl:
                    losses += 1
                    total_pnl -= trade.risk_amount
                    break
                elif low[i] <= tp:
                    wins += 1
                    total_pnl += trade.risk_amount * RISK_REWARD
                    break
    
    total = wins + losses
    
    return {
        'total_blocked': len(blocked_trades),
        'resolved_blocked': total,
        'would_have_won': wins,
        'would_have_lost': losses,
        'would_have_pnl': total_pnl,
        'counterfactual_win_rate': wins / total if total > 0 else 0,
    }


def get_block_reason_distribution(blocked_trades: List[Trade]) -> Dict[str, int]:
    """Get distribution of block reasons."""
    reasons = {
        'EXTREME_SPREAD': 0,
        'DETERIORATING_CORRELATION': 0,
        'COMPRESSED_VOLATILITY': 0,
    }
    
    for trade in blocked_trades:
        for reason in trade.block_reasons:
            if reason in reasons:
                reasons[reason] += 1
    
    return reasons
