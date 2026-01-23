"""
Production Backtest Harness

Clean comparative backtest using the production system components:
- Baseline (no gatekeeper)
- Gated (with gatekeeper)

Identical execution rules, identical data, identical risk sizing.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from trading_system.config import StrategyConfig, GatekeeperConfig, RiskConfig
from trading_system.signal_engine import SignalEngine, SignalDirection
from trading_system.gatekeeper_engine import GatekeeperEngine


@dataclass
class BacktestTrade:
    """Trade record for backtest."""
    entry_bar: int
    entry_time: datetime
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    
    exit_bar: Optional[int] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    won: bool = False
    
    # Gatekeeper info
    was_blocked: bool = False
    block_reasons: List[str] = field(default_factory=list)


@dataclass 
class BacktestResult:
    """Complete backtest results."""
    name: str
    trades: List[BacktestTrade]
    equity_curve: List[float]
    
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_trade: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    
    # Gatekeeper specific
    blocked_count: int = 0
    block_reasons: Dict[str, int] = field(default_factory=dict)


class ProductionBacktest:
    """
    Backtest engine using production components.
    """
    
    def __init__(
        self,
        use_gatekeeper: bool = False,
        initial_capital: float = 100000.0,
        risk_per_trade: float = 0.005,
    ):
        self.use_gatekeeper = use_gatekeeper
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        
        # Components
        self.signal_engine = SignalEngine(StrategyConfig())
        self.gatekeeper = GatekeeperEngine(GatekeeperConfig()) if use_gatekeeper else None
        
        # State
        self.equity = initial_capital
        self.trades: List[BacktestTrade] = []
        self.blocked_trades: List[BacktestTrade] = []
        self.current_trade: Optional[BacktestTrade] = None
        self.equity_curve: List[float] = []
        
        self._bar_index = 0
    
    def run(
        self,
        df_eurusd: pd.DataFrame,
        df_gbpusd: pd.DataFrame,
    ) -> BacktestResult:
        """
        Run backtest on historical data.
        
        DataFrames must have: timestamp, open, high, low, close
        """
        assert len(df_eurusd) == len(df_gbpusd), "Data length mismatch"
        
        n = len(df_eurusd)
        
        for i in range(n):
            self._bar_index = i
            
            # Get bar data
            eu_bar = df_eurusd.iloc[i]
            gb_bar = df_gbpusd.iloc[i]
            timestamp = eu_bar['timestamp']
            
            # Update gatekeeper
            if self.gatekeeper:
                self.gatekeeper.update(eu_bar['close'], gb_bar['close'])
            
            # Check for exit
            if self.current_trade is not None:
                self._check_exit(eu_bar['high'], eu_bar['low'], timestamp)
            
            # Get signal
            signal = self.signal_engine.update(
                timestamp=timestamp,
                open_=eu_bar['open'],
                high=eu_bar['high'],
                low=eu_bar['low'],
                close=eu_bar['close'],
            )
            
            # Process signal
            if signal is not None and self.current_trade is None:
                self._process_signal(signal, timestamp)
            
            # Record equity
            self.equity_curve.append(self.equity)
        
        # Force close any open trade
        if self.current_trade is not None:
            last_bar = df_eurusd.iloc[-1]
            self._force_close(last_bar['close'], last_bar['timestamp'])
        
        return self._compute_results()
    
    def _process_signal(self, signal, timestamp: datetime) -> None:
        """Process a trade signal."""
        # Check gatekeeper
        if self.gatekeeper:
            decision = self.gatekeeper.evaluate()
            
            if not decision.allowed:
                # Record blocked trade
                trade = BacktestTrade(
                    entry_bar=self._bar_index,
                    entry_time=timestamp,
                    direction=signal.direction.value,
                    entry_price=signal.entry_price,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    position_size=0,
                    was_blocked=True,
                    block_reasons=[r.value for r in decision.reasons],
                )
                self.blocked_trades.append(trade)
                return
        
        # Calculate position size
        sl_distance = abs(signal.entry_price - signal.stop_loss)
        risk_amount = self.equity * self.risk_per_trade
        position_size = risk_amount / sl_distance if sl_distance > 0 else 0
        
        # Create trade
        trade = BacktestTrade(
            entry_bar=self._bar_index,
            entry_time=timestamp,
            direction=signal.direction.value,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            position_size=position_size,
        )
        
        self.current_trade = trade
        self.trades.append(trade)
    
    def _check_exit(self, high: float, low: float, timestamp: datetime) -> None:
        """Check for SL/TP exit."""
        trade = self.current_trade
        
        if trade.direction == "LONG":
            if low <= trade.stop_loss:
                self._close_trade(trade.stop_loss, timestamp, won=False)
            elif high >= trade.take_profit:
                self._close_trade(trade.take_profit, timestamp, won=True)
        else:  # SHORT
            if high >= trade.stop_loss:
                self._close_trade(trade.stop_loss, timestamp, won=False)
            elif low <= trade.take_profit:
                self._close_trade(trade.take_profit, timestamp, won=True)
    
    def _close_trade(self, exit_price: float, timestamp: datetime, won: bool) -> None:
        """Close current trade."""
        trade = self.current_trade
        trade.exit_bar = self._bar_index
        trade.exit_price = exit_price
        trade.won = won
        
        if trade.direction == "LONG":
            trade.pnl = (exit_price - trade.entry_price) * trade.position_size
        else:
            trade.pnl = (trade.entry_price - exit_price) * trade.position_size
        
        self.equity += trade.pnl
        self.current_trade = None
    
    def _force_close(self, price: float, timestamp: datetime) -> None:
        """Force close at end of backtest."""
        trade = self.current_trade
        trade.exit_bar = self._bar_index
        trade.exit_price = price
        
        if trade.direction == "LONG":
            trade.pnl = (price - trade.entry_price) * trade.position_size
        else:
            trade.pnl = (trade.entry_price - price) * trade.position_size
        
        trade.won = trade.pnl > 0
        self.equity += trade.pnl
        self.current_trade = None
    
    def _compute_results(self) -> BacktestResult:
        """Compute backtest statistics."""
        name = "Gated" if self.use_gatekeeper else "Baseline"
        
        if not self.trades:
            return BacktestResult(name=name, trades=[], equity_curve=self.equity_curve)
        
        wins = [t for t in self.trades if t.won]
        losses = [t for t in self.trades if not t.won]
        
        total_trades = len(self.trades)
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = win_count / total_trades if total_trades > 0 else 0
        
        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0.0001
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        total_pnl = sum(t.pnl for t in self.trades)
        avg_trade = total_pnl / total_trades if total_trades > 0 else 0
        avg_win = gross_profit / win_count if win_count > 0 else 0
        avg_loss = gross_loss / loss_count if loss_count > 0 else 0
        
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        # Drawdown
        equity_array = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (running_max - equity_array) / running_max
        max_dd = np.max(drawdown) if len(drawdown) > 0 else 0
        max_dd_abs = np.max(running_max - equity_array) if len(drawdown) > 0 else 0
        
        # Block reasons
        block_reasons = {}
        for trade in self.blocked_trades:
            for reason in trade.block_reasons:
                block_reasons[reason] = block_reasons.get(reason, 0) + 1
        
        return BacktestResult(
            name=name,
            trades=self.trades,
            equity_curve=self.equity_curve,
            total_trades=total_trades,
            wins=win_count,
            losses=loss_count,
            win_rate=win_rate,
            profit_factor=profit_factor,
            expectancy=expectancy,
            total_pnl=total_pnl,
            max_drawdown=max_dd_abs,
            max_drawdown_pct=max_dd,
            avg_trade=avg_trade,
            avg_win=avg_win,
            avg_loss=avg_loss,
            blocked_count=len(self.blocked_trades),
            block_reasons=block_reasons,
        )


def simulate_blocked_trades(
    blocked_trades: List[BacktestTrade],
    df_eurusd: pd.DataFrame,
) -> Tuple[int, int, float]:
    """
    Counterfactual: what would blocked trades have done?
    
    Returns: (wins, losses, pnl)
    """
    high = df_eurusd['high'].values
    low = df_eurusd['low'].values
    n = len(high)
    
    wins = 0
    losses = 0
    pnl = 0.0
    
    for trade in blocked_trades:
        entry_bar = trade.entry_bar
        sl = trade.stop_loss
        tp = trade.take_profit
        
        # Simulate using 1% risk equivalent
        risk_amount = 1000  # $1000 fixed for comparison
        sl_dist = abs(trade.entry_price - sl)
        pos_size = risk_amount / sl_dist if sl_dist > 0 else 0
        
        for i in range(entry_bar + 1, min(entry_bar + 100, n)):
            if trade.direction == "LONG":
                if low[i] <= sl:
                    losses += 1
                    pnl -= risk_amount
                    break
                elif high[i] >= tp:
                    wins += 1
                    pnl += risk_amount * 2  # RR = 2
                    break
            else:
                if high[i] >= sl:
                    losses += 1
                    pnl -= risk_amount
                    break
                elif low[i] <= tp:
                    wins += 1
                    pnl += risk_amount * 2
                    break
    
    return wins, losses, pnl


def print_comparison(baseline: BacktestResult, gated: BacktestResult, counterfactual: dict):
    """Print side-by-side comparison."""
    print()
    print("=" * 75)
    print("PRODUCTION BACKTEST COMPARISON: BASELINE vs GATED")
    print("=" * 75)
    print()
    
    def fmt(v, t='f'):
        if t == 'p': return f"{v:.2%}"
        if t == 'd': return f"${v:,.2f}"
        if t == 'i': return f"{int(v)}"
        return f"{v:.2f}"
    
    metrics = [
        ("Total Trades", baseline.total_trades, gated.total_trades, 'i'),
        ("Wins", baseline.wins, gated.wins, 'i'),
        ("Losses", baseline.losses, gated.losses, 'i'),
        ("Win Rate", baseline.win_rate, gated.win_rate, 'p'),
        ("Profit Factor", baseline.profit_factor, gated.profit_factor, 'f'),
        ("Expectancy", baseline.expectancy, gated.expectancy, 'd'),
        ("Max Drawdown", baseline.max_drawdown_pct, gated.max_drawdown_pct, 'p'),
        ("Total PnL", baseline.total_pnl, gated.total_pnl, 'd'),
        ("Avg Trade", baseline.avg_trade, gated.avg_trade, 'd'),
        ("Avg Win", baseline.avg_win, gated.avg_win, 'd'),
        ("Avg Loss", baseline.avg_loss, gated.avg_loss, 'd'),
    ]
    
    print(f"{'Metric':<20} {'Baseline':>15} {'Gated':>15} {'Delta':>15}")
    print("-" * 75)
    
    for label, b_val, g_val, t in metrics:
        delta = g_val - b_val
        if t == 'p':
            delta_str = f"{delta:+.2%}"
        elif t == 'd':
            delta_str = f"${delta:+,.2f}"
        else:
            delta_str = f"{delta:+.2f}"
        
        print(f"{label:<20} {fmt(b_val, t):>15} {fmt(g_val, t):>15} {delta_str:>15}")
    
    print()
    print("-" * 75)
    print("GATEKEEPER ANALYSIS")
    print("-" * 75)
    
    total_signals = gated.total_trades + gated.blocked_count
    block_rate = gated.blocked_count / total_signals if total_signals > 0 else 0
    
    print(f"Total Signals:       {total_signals}")
    print(f"Trades Executed:     {gated.total_trades}")
    print(f"Trades Blocked:      {gated.blocked_count}")
    print(f"Block Rate:          {block_rate:.2%}")
    print()
    print("Block Reasons:")
    for reason, count in gated.block_reasons.items():
        pct = count / gated.blocked_count if gated.blocked_count > 0 else 0
        print(f"  {reason:<30} {count:>4} ({pct:.1%})")
    
    print()
    print("-" * 75)
    print("COUNTERFACTUAL (Blocked Trades)")
    print("-" * 75)
    print(f"Would have won:      {counterfactual['wins']}")
    print(f"Would have lost:     {counterfactual['losses']}")
    
    cf_total = counterfactual['wins'] + counterfactual['losses']
    cf_wr = counterfactual['wins'] / cf_total if cf_total > 0 else 0
    print(f"Counterfactual WR:   {cf_wr:.2%}")
    print(f"Counterfactual PnL:  ${counterfactual['pnl']:,.2f}")
    
    print()
    wr_diff = gated.win_rate - cf_wr
    print(f"WR Difference (Allowed - Blocked): {wr_diff:+.2%}")
    
    print()
    print("=" * 75)
    
    # Verdict
    robustness_improved = (
        (baseline.max_drawdown_pct - gated.max_drawdown_pct) > 0.005 or
        (gated.win_rate - baseline.win_rate) > 0.02 or
        (gated.profit_factor - baseline.profit_factor) > 0.1
    )
    
    expectancy_survived = gated.expectancy > baseline.expectancy * 0.7 and gated.expectancy > 0
    
    if robustness_improved and expectancy_survived:
        verdict = "VALID"
        msg = "Gatekeeper improves robustness without destroying expectancy"
    elif robustness_improved and not expectancy_survived:
        verdict = "FILTER TOO AGGRESSIVE"
        msg = "Gatekeeper improves robustness but kills expectancy"
    else:
        verdict = "NO VALUE ADD"
        msg = "Gatekeeper does not materially improve robustness"
    
    print(f"VERDICT: {verdict}")
    print(f"         {msg}")
    print("=" * 75)
    
    return verdict


def run_production_backtest(df_eurusd: pd.DataFrame, df_gbpusd: pd.DataFrame):
    """Run complete production backtest comparison."""
    print("Running BASELINE backtest...")
    baseline_bt = ProductionBacktest(use_gatekeeper=False)
    baseline = baseline_bt.run(df_eurusd.copy(), df_gbpusd.copy())
    print(f"  Completed: {baseline.total_trades} trades")
    
    print("Running GATED backtest...")
    gated_bt = ProductionBacktest(use_gatekeeper=True)
    gated = gated_bt.run(df_eurusd.copy(), df_gbpusd.copy())
    print(f"  Completed: {gated.total_trades} trades, {gated.blocked_count} blocked")
    
    # Counterfactual
    cf_wins, cf_losses, cf_pnl = simulate_blocked_trades(
        gated_bt.blocked_trades, df_eurusd
    )
    counterfactual = {'wins': cf_wins, 'losses': cf_losses, 'pnl': cf_pnl}
    
    verdict = print_comparison(baseline, gated, counterfactual)
    
    return baseline, gated, verdict


if __name__ == "__main__":
    print("Run with: from backtest_harness import run_production_backtest")
