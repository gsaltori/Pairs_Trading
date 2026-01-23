"""
Breakout Strategy Backtest Harness

Evaluates the Range Breakout strategy with:
1. Baseline (no filter)
2. Baseline + Gatekeeper

Focus on EXPECTANCY and asymmetric payoff validation.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_system.config import GatekeeperConfig
from trading_system.breakout_engine import RangeBreakoutEngine, BreakoutSignal, BreakoutDirection
from trading_system.gatekeeper_engine import GatekeeperEngine, GatekeeperDecision


@dataclass
class BreakoutTrade:
    """Trade record for breakout backtest."""
    entry_bar: int
    entry_time: datetime
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    
    # Range info
    range_high: float = 0.0
    range_low: float = 0.0
    range_width: float = 0.0
    compression_ratio: float = 0.0
    
    # Exit info
    exit_bar: Optional[int] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""  # "SL", "TP", "TIME", "END"
    bars_held: int = 0
    
    # Results
    pnl: float = 0.0
    pnl_r: float = 0.0
    won: bool = False
    
    # Filter info
    blocked_by_gatekeeper: bool = False
    gatekeeper_reasons: List[str] = field(default_factory=list)


@dataclass
class BreakoutResult:
    """Complete backtest results."""
    name: str
    trades: List[BreakoutTrade]
    blocked_trades: List[BreakoutTrade]
    equity_curve: List[float]
    
    # Core metrics
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    
    # Expectancy (KEY METRICS)
    profit_factor: float = 0.0
    expectancy: float = 0.0
    expectancy_r: float = 0.0
    
    # Win/Loss analysis
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_win_r: float = 0.0
    avg_loss_r: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    
    # Risk metrics
    total_pnl: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_abs: float = 0.0
    
    # Trade characteristics
    avg_bars_held: float = 0.0
    avg_compression_ratio: float = 0.0
    
    # Filter stats
    signals_generated: int = 0
    blocked_by_gatekeeper: int = 0
    
    # Exit analysis
    exits_by_sl: int = 0
    exits_by_tp: int = 0
    exits_by_time: int = 0
    exits_by_end: int = 0


class BreakoutBacktest:
    """
    Breakout strategy backtest engine.
    
    Modes:
    - baseline: No filters
    - gated: With Gatekeeper
    """
    
    TIME_STOP_BARS = 20  # Max bars to hold (optional)
    USE_TIME_STOP = False  # Set to True to enable
    
    def __init__(
        self,
        use_gatekeeper: bool = False,
        initial_capital: float = 100.0,
        risk_per_trade: float = 0.005,  # 0.5%
    ):
        self.use_gatekeeper = use_gatekeeper
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        
        # Components
        self.breakout_engine = RangeBreakoutEngine()
        self.gatekeeper = GatekeeperEngine(GatekeeperConfig()) if use_gatekeeper else None
        
        # State
        self.equity = initial_capital
        self.trades: List[BreakoutTrade] = []
        self.blocked_trades: List[BreakoutTrade] = []
        self.current_trade: Optional[BreakoutTrade] = None
        self.equity_curve: List[float] = []
        
        self._bar_index = 0
        self._signals_generated = 0
    
    def run(
        self,
        df_eurusd: pd.DataFrame,
        df_gbpusd: pd.DataFrame,
    ) -> BreakoutResult:
        """Run backtest."""
        assert len(df_eurusd) == len(df_gbpusd), "Data length mismatch"
        
        n = len(df_eurusd)
        
        for i in range(n):
            self._bar_index = i
            
            eu_bar = df_eurusd.iloc[i]
            gb_bar = df_gbpusd.iloc[i]
            timestamp = eu_bar['timestamp']
            
            # Update gatekeeper
            if self.gatekeeper:
                self.gatekeeper.update(eu_bar['close'], gb_bar['close'])
            
            # Check exit first
            if self.current_trade is not None:
                self._check_exit(eu_bar['high'], eu_bar['low'], timestamp)
            
            # Generate signal
            signal = self.breakout_engine.update(
                timestamp=timestamp,
                open_=eu_bar['open'],
                high=eu_bar['high'],
                low=eu_bar['low'],
                close=eu_bar['close'],
            )
            
            # Process signal
            if signal is not None:
                self._signals_generated += 1
                
                if self.current_trade is None:
                    self._process_signal(signal, timestamp)
            
            self.equity_curve.append(self.equity)
        
        # Force close at end
        if self.current_trade is not None:
            last = df_eurusd.iloc[-1]
            self._force_close(last['close'], last['timestamp'], "END")
        
        return self._compute_results()
    
    def _process_signal(self, signal: BreakoutSignal, timestamp: datetime) -> None:
        """Process a breakout signal."""
        direction = signal.direction.value
        
        # Check gatekeeper
        if self.gatekeeper:
            gate_decision = self.gatekeeper.evaluate()
            
            if not gate_decision.allowed:
                trade = BreakoutTrade(
                    entry_bar=self._bar_index,
                    entry_time=timestamp,
                    direction=direction,
                    entry_price=signal.entry_price,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    position_size=0,
                    range_high=signal.range_high,
                    range_low=signal.range_low,
                    range_width=signal.range_width,
                    compression_ratio=signal.compression_ratio,
                    blocked_by_gatekeeper=True,
                    gatekeeper_reasons=[r.value for r in gate_decision.reasons],
                )
                self.blocked_trades.append(trade)
                return
        
        # Calculate position size
        sl_distance = abs(signal.entry_price - signal.stop_loss)
        risk_amount = self.equity * self.risk_per_trade
        
        if sl_distance > 0:
            position_size = risk_amount / (sl_distance * 100000)
            position_size = max(0.01, min(1.0, round(position_size, 2)))
        else:
            position_size = 0.01
        
        # Create trade
        trade = BreakoutTrade(
            entry_bar=self._bar_index,
            entry_time=timestamp,
            direction=direction,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            position_size=position_size,
            range_high=signal.range_high,
            range_low=signal.range_low,
            range_width=signal.range_width,
            compression_ratio=signal.compression_ratio,
        )
        
        self.current_trade = trade
        self.trades.append(trade)
    
    def _check_exit(self, high: float, low: float, timestamp: datetime) -> None:
        """Check SL/TP/Time exits."""
        trade = self.current_trade
        trade.bars_held = self._bar_index - trade.entry_bar
        
        # Time stop (optional)
        if self.USE_TIME_STOP and trade.bars_held >= self.TIME_STOP_BARS:
            mid_price = (high + low) / 2
            self._close_trade(mid_price, timestamp, "TIME")
            return
        
        if trade.direction == "LONG":
            # Check SL first (conservative)
            if low <= trade.stop_loss:
                self._close_trade(trade.stop_loss, timestamp, "SL")
            elif high >= trade.take_profit:
                self._close_trade(trade.take_profit, timestamp, "TP")
        else:
            if high >= trade.stop_loss:
                self._close_trade(trade.stop_loss, timestamp, "SL")
            elif low <= trade.take_profit:
                self._close_trade(trade.take_profit, timestamp, "TP")
    
    def _close_trade(self, exit_price: float, timestamp: datetime, reason: str) -> None:
        """Close trade with specified exit price and reason."""
        trade = self.current_trade
        trade.exit_bar = self._bar_index
        trade.exit_price = exit_price
        trade.exit_reason = reason
        trade.bars_held = self._bar_index - trade.entry_bar
        
        # PnL calculation
        if trade.direction == "LONG":
            price_diff = exit_price - trade.entry_price
        else:
            price_diff = trade.entry_price - exit_price
        
        trade.pnl = price_diff * trade.position_size * 100000
        
        # R-multiple
        sl_distance = abs(trade.entry_price - trade.stop_loss)
        if sl_distance > 0:
            trade.pnl_r = price_diff / sl_distance
        else:
            trade.pnl_r = 0
        
        trade.won = trade.pnl > 0
        
        self.equity += trade.pnl
        self.current_trade = None
    
    def _force_close(self, price: float, timestamp: datetime, reason: str) -> None:
        """Force close at end of data."""
        self._close_trade(price, timestamp, reason)
    
    def _compute_results(self) -> BreakoutResult:
        """Compute comprehensive backtest statistics."""
        name = "Breakout+Gate" if self.use_gatekeeper else "Breakout"
        
        if not self.trades:
            return BreakoutResult(
                name=name,
                trades=self.trades,
                blocked_trades=self.blocked_trades,
                equity_curve=self.equity_curve,
                signals_generated=self._signals_generated,
                blocked_by_gatekeeper=len(self.blocked_trades),
            )
        
        # Win/Loss separation
        wins = [t for t in self.trades if t.won]
        losses = [t for t in self.trades if not t.won]
        
        total = len(self.trades)
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = win_count / total if total > 0 else 0
        
        # Dollar metrics
        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0.0001
        pf = gross_profit / gross_loss if gross_loss > 0 else 0
        
        total_pnl = sum(t.pnl for t in self.trades)
        avg_win = gross_profit / win_count if win_count > 0 else 0
        avg_loss = gross_loss / loss_count if loss_count > 0 else 0
        largest_win = max(t.pnl for t in wins) if wins else 0
        largest_loss = min(t.pnl for t in losses) if losses else 0
        
        # EXPECTANCY
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        # R-multiple metrics
        avg_win_r = np.mean([t.pnl_r for t in wins]) if wins else 0
        avg_loss_r = abs(np.mean([t.pnl_r for t in losses])) if losses else 0
        expectancy_r = (win_rate * avg_win_r) - ((1 - win_rate) * avg_loss_r)
        
        # Drawdown
        eq = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(eq)
        dd_abs = running_max - eq
        dd_pct = dd_abs / running_max
        max_dd_pct = np.max(dd_pct) if len(dd_pct) > 0 else 0
        max_dd_abs = np.max(dd_abs) if len(dd_abs) > 0 else 0
        
        # Trade characteristics
        avg_bars = np.mean([t.bars_held for t in self.trades])
        avg_compression = np.mean([t.compression_ratio for t in self.trades])
        
        # Exit analysis
        exits_sl = len([t for t in self.trades if t.exit_reason == "SL"])
        exits_tp = len([t for t in self.trades if t.exit_reason == "TP"])
        exits_time = len([t for t in self.trades if t.exit_reason == "TIME"])
        exits_end = len([t for t in self.trades if t.exit_reason == "END"])
        
        return BreakoutResult(
            name=name,
            trades=self.trades,
            blocked_trades=self.blocked_trades,
            equity_curve=self.equity_curve,
            total_trades=total,
            wins=win_count,
            losses=loss_count,
            win_rate=win_rate,
            profit_factor=pf,
            expectancy=expectancy,
            expectancy_r=expectancy_r,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_win_r=avg_win_r,
            avg_loss_r=avg_loss_r,
            largest_win=largest_win,
            largest_loss=largest_loss,
            total_pnl=total_pnl,
            max_drawdown_pct=max_dd_pct,
            max_drawdown_abs=max_dd_abs,
            avg_bars_held=avg_bars,
            avg_compression_ratio=avg_compression,
            signals_generated=self._signals_generated,
            blocked_by_gatekeeper=len(self.blocked_trades),
            exits_by_sl=exits_sl,
            exits_by_tp=exits_tp,
            exits_by_time=exits_time,
            exits_by_end=exits_end,
        )


def counterfactual_blocked_trades(
    blocked_trades: List[BreakoutTrade],
    df_eurusd: pd.DataFrame,
) -> Dict:
    """
    Analyze what gatekeeper-blocked trades would have done.
    """
    if not blocked_trades:
        return {
            'total': 0,
            'resolved': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'expectancy': 0,
            'effectiveness': 'N/A (no blocks)',
        }
    
    high = df_eurusd['high'].values
    low = df_eurusd['low'].values
    n = len(high)
    
    wins = 0
    losses = 0
    total_pnl = 0.0
    resolved = 0
    
    for trade in blocked_trades:
        entry_bar = trade.entry_bar
        sl = trade.stop_loss
        tp = trade.take_profit
        entry = trade.entry_price
        
        # Simulate fixed $1 risk
        risk = 1.0
        sl_dist = abs(entry - sl)
        tp_dist = abs(tp - entry)
        rr = tp_dist / sl_dist if sl_dist > 0 else 2.5
        
        for i in range(entry_bar + 1, min(entry_bar + 50, n)):
            if trade.direction == "LONG":
                if low[i] <= sl:
                    losses += 1
                    total_pnl -= risk
                    resolved += 1
                    break
                elif high[i] >= tp:
                    wins += 1
                    total_pnl += risk * rr
                    resolved += 1
                    break
            else:
                if high[i] >= sl:
                    losses += 1
                    total_pnl -= risk
                    resolved += 1
                    break
                elif low[i] <= tp:
                    wins += 1
                    total_pnl += risk * rr
                    resolved += 1
                    break
    
    total = wins + losses
    win_rate = wins / total if total > 0 else 0
    
    avg_win = (total_pnl / wins) if wins > 0 and total_pnl > 0 else 0
    avg_loss = 1.0  # Fixed $1 risk
    expectancy = (win_rate * avg_win * 2.5) - ((1 - win_rate) * avg_loss) if total > 0 else 0
    
    # Effectiveness assessment
    if expectancy < 0:
        effectiveness = f"EFFECTIVE (blocked trades expectancy: ${expectancy:.2f})"
    elif expectancy > 0:
        effectiveness = f"HARMFUL (blocked trades expectancy: ${expectancy:.2f})"
    else:
        effectiveness = "NEUTRAL"
    
    return {
        'total': len(blocked_trades),
        'resolved': resolved,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'expectancy': expectancy,
        'effectiveness': effectiveness,
    }


def run_breakout_comparison(
    df_eurusd: pd.DataFrame,
    df_gbpusd: pd.DataFrame,
    initial_capital: float = 100.0,
) -> Tuple[BreakoutResult, BreakoutResult, Dict]:
    """
    Run baseline vs gated comparison.
    """
    # 1. Baseline
    print("  Running BASELINE (no gatekeeper)...")
    baseline_bt = BreakoutBacktest(
        use_gatekeeper=False,
        initial_capital=initial_capital,
    )
    baseline = baseline_bt.run(df_eurusd.copy(), df_gbpusd.copy())
    print(f"    {baseline.total_trades} trades, Exp: ${baseline.expectancy:.2f}, WR: {baseline.win_rate:.1%}")
    
    # 2. With Gatekeeper
    print("  Running BREAKOUT + GATEKEEPER...")
    gated_bt = BreakoutBacktest(
        use_gatekeeper=True,
        initial_capital=initial_capital,
    )
    gated = gated_bt.run(df_eurusd.copy(), df_gbpusd.copy())
    print(f"    {gated.total_trades} trades, Exp: ${gated.expectancy:.2f}, Blocked: {gated.blocked_by_gatekeeper}")
    
    # 3. Counterfactual
    print("  Running COUNTERFACTUAL...")
    cf = counterfactual_blocked_trades(gated.blocked_trades, df_eurusd)
    print(f"    Blocked: {cf['total']}, CF Win Rate: {cf['win_rate']:.1%}")
    
    return baseline, gated, cf


def print_breakout_comparison(baseline: BreakoutResult, gated: BreakoutResult):
    """Print comparison table."""
    print()
    print("=" * 75)
    print("BREAKOUT STRATEGY BACKTEST COMPARISON")
    print("=" * 75)
    print()
    
    header = f"{'Metric':<30} {'Baseline':>20} {'Breakout+Gate':>20}"
    print(header)
    print("-" * 75)
    
    def fmt(v, t='f'):
        if t == 'p': return f"{v:.1%}"
        if t == 'd': return f"${v:.2f}"
        if t == 'i': return f"{int(v)}"
        if t == 'r': return f"{v:.2f}R"
        return f"{v:.2f}"
    
    rows = [
        ("Trades", baseline.total_trades, gated.total_trades, 'i'),
        ("Wins", baseline.wins, gated.wins, 'i'),
        ("Losses", baseline.losses, gated.losses, 'i'),
        ("Win Rate", baseline.win_rate, gated.win_rate, 'p'),
        ("Profit Factor", baseline.profit_factor, gated.profit_factor, 'f'),
        ("", "", "", ''),
        ("EXPECTANCY ($)", baseline.expectancy, gated.expectancy, 'd'),
        ("Expectancy (R)", baseline.expectancy_r, gated.expectancy_r, 'r'),
        ("", "", "", ''),
        ("Avg Win ($)", baseline.avg_win, gated.avg_win, 'd'),
        ("Avg Loss ($)", baseline.avg_loss, gated.avg_loss, 'd'),
        ("Avg Win (R)", baseline.avg_win_r, gated.avg_win_r, 'r'),
        ("Avg Loss (R)", baseline.avg_loss_r, gated.avg_loss_r, 'r'),
        ("Largest Win", baseline.largest_win, gated.largest_win, 'd'),
        ("Largest Loss", baseline.largest_loss, gated.largest_loss, 'd'),
        ("", "", "", ''),
        ("Max Drawdown", baseline.max_drawdown_pct, gated.max_drawdown_pct, 'p'),
        ("Max DD ($)", baseline.max_drawdown_abs, gated.max_drawdown_abs, 'd'),
        ("Net PnL", baseline.total_pnl, gated.total_pnl, 'd'),
        ("", "", "", ''),
        ("Avg Bars Held", baseline.avg_bars_held, gated.avg_bars_held, 'f'),
        ("Avg Compression", baseline.avg_compression_ratio, gated.avg_compression_ratio, 'f'),
        ("", "", "", ''),
        ("Exits by SL", baseline.exits_by_sl, gated.exits_by_sl, 'i'),
        ("Exits by TP", baseline.exits_by_tp, gated.exits_by_tp, 'i'),
        ("Blocked (Gate)", 0, gated.blocked_by_gatekeeper, 'i'),
    ]
    
    for label, b, g, t in rows:
        if label == "":
            print()
            continue
        print(f"{label:<30} {fmt(b, t):>20} {fmt(g, t):>20}")
    
    print()


def print_counterfactual(cf: Dict):
    """Print counterfactual analysis."""
    print("=" * 75)
    print("GATEKEEPER COUNTERFACTUAL ANALYSIS")
    print("=" * 75)
    print()
    print("What would blocked trades have done?")
    print()
    print(f"  Blocked by Gatekeeper:    {cf['total']}")
    print(f"  Resolved (SL/TP hit):     {cf['resolved']}")
    print(f"  Would Have Won:           {cf['wins']}")
    print(f"  Would Have Lost:          {cf['losses']}")
    print(f"  Counterfactual Win Rate:  {cf['win_rate']:.1%}")
    print(f"  Counterfactual PnL:       ${cf['total_pnl']:.2f}")
    print()
    print(f"  Gatekeeper: {cf['effectiveness']}")
    print()


def determine_breakout_viability(result: BreakoutResult) -> Tuple[str, str]:
    """
    Determine if breakout strategy is viable.
    
    Kill Criteria:
    - Expectancy ≤ 0 → NOT VIABLE
    - Profit Factor < 1.0 → NOT VIABLE
    - Win Rate < 15% with R=2.5 → mathematically unprofitable
    """
    exp = result.expectancy
    exp_r = result.expectancy_r
    pf = result.profit_factor
    wr = result.win_rate
    dd = result.max_drawdown_pct
    trades = result.total_trades
    
    print("=" * 75)
    print("VIABILITY ASSESSMENT")
    print("=" * 75)
    print()
    
    issues = []
    
    # Kill criterion: Expectancy ≤ 0
    if exp <= 0:
        issues.append(f"KILL: Expectancy is ${exp:.2f} (≤ $0)")
    elif exp < 0.10:
        issues.append(f"WARNING: Expectancy is low (${exp:.2f})")
    
    # Kill criterion: R-expectancy ≤ 0
    if exp_r <= 0:
        issues.append(f"KILL: R-Expectancy is {exp_r:.2f}R (≤ 0R)")
    elif exp_r < 0.15:
        issues.append(f"WARNING: R-Expectancy is low ({exp_r:.2f}R)")
    
    # Kill criterion: PF < 1.0
    if pf < 1.0:
        issues.append(f"KILL: Profit Factor is {pf:.2f} (< 1.0)")
    elif pf < 1.2:
        issues.append(f"WARNING: Profit Factor is marginal ({pf:.2f})")
    
    # Check win rate vs RR math
    # For R=2.5: breakeven WR = 1/(1+2.5) = 28.6%
    breakeven_wr = 1 / (1 + 2.5)
    if wr < breakeven_wr:
        issues.append(f"KILL: Win rate {wr:.1%} below breakeven {breakeven_wr:.1%} for R=2.5")
    
    # Drawdown check
    if dd > 0.25:
        issues.append(f"WARNING: Max Drawdown is high ({dd:.1%})")
    
    # Sample size
    if trades < 20:
        issues.append(f"WARNING: Small sample size ({trades} trades)")
    
    for issue in issues:
        print(f"  • {issue}")
    
    print()
    
    # Verdict
    kill_count = sum(1 for i in issues if i.startswith("KILL"))
    warning_count = sum(1 for i in issues if i.startswith("WARNING"))
    
    if kill_count > 0:
        verdict = "NOT VIABLE"
        explanation = (
            "Strategy FAILS kill criteria.\n"
            "  Edge is mathematically invalid.\n"
            "  DO NOT deploy. Abandon or redesign."
        )
    elif warning_count >= 2:
        verdict = "MARGINAL"
        explanation = (
            "Strategy has multiple concerns.\n"
            "  Extended paper testing required.\n"
            "  Do not risk real capital yet."
        )
    elif warning_count == 1:
        verdict = "CAUTIOUS PROCEED"
        explanation = (
            "Strategy shows potential with concerns.\n"
            "  Forward test for 3+ months before live."
        )
    else:
        verdict = "VIABLE"
        explanation = (
            "Strategy meets viability criteria.\n"
            "  Proceed to paper trading, then micro capital."
        )
    
    print(f"VERDICT: {verdict}")
    print()
    for line in explanation.split('\n'):
        print(f"  {line}")
    print()
    print("=" * 75)
    
    return verdict, explanation
