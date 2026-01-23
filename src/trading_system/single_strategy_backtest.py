"""
Single-Strategy Backtest Harness

Focused backtest comparing:
1. Baseline Trend (no filters)
2. Trend + Gatekeeper
3. Trend + Gatekeeper + MRF

With counterfactual analysis and expectancy focus.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_system.config import StrategyConfig, GatekeeperConfig
from trading_system.signal_engine import SignalEngine, TradeSignal, SignalDirection
from trading_system.gatekeeper_engine import GatekeeperEngine, GatekeeperDecision
from trading_system.market_regime_filter import MarketRegimeFilter, MRFDecision, MRFBlockReason


@dataclass
class BacktestTrade:
    """Trade record."""
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
    pnl_r: float = 0.0  # PnL in R-multiples
    won: bool = False
    
    # Filter info
    blocked_by_gatekeeper: bool = False
    blocked_by_mrf: bool = False
    gatekeeper_reasons: List[str] = field(default_factory=list)
    mrf_reasons: List[str] = field(default_factory=list)


@dataclass
class BacktestResult:
    """Complete backtest results."""
    name: str
    trades: List[BacktestTrade]
    blocked_trades: List[BacktestTrade]
    equity_curve: List[float]
    
    # Core metrics
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    
    # Expectancy focus
    profit_factor: float = 0.0
    expectancy: float = 0.0
    expectancy_r: float = 0.0  # In R-multiples
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_win_r: float = 0.0
    avg_loss_r: float = 0.0
    
    # Risk
    total_pnl: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_abs: float = 0.0
    
    # Filter stats
    signals_generated: int = 0
    blocked_by_gatekeeper: int = 0
    blocked_by_mrf: int = 0


class SingleStrategyBacktest:
    """
    Single strategy backtest engine.
    
    Modes:
    - baseline: No filters
    - gatekeeper: Gatekeeper only
    - full: Gatekeeper + MRF
    """
    
    def __init__(
        self,
        mode: str = "full",  # "baseline", "gatekeeper", "full"
        initial_capital: float = 100.0,
        risk_per_trade: float = 0.005,  # 0.5%
    ):
        assert mode in ("baseline", "gatekeeper", "full")
        
        self.mode = mode
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        
        # Components
        self.signal_engine = SignalEngine(StrategyConfig())
        self.gatekeeper = GatekeeperEngine(GatekeeperConfig()) if mode in ("gatekeeper", "full") else None
        self.mrf = MarketRegimeFilter() if mode == "full" else None
        
        # State
        self.equity = initial_capital
        self.trades: List[BacktestTrade] = []
        self.blocked_trades: List[BacktestTrade] = []
        self.current_trade: Optional[BacktestTrade] = None
        self.equity_curve: List[float] = []
        
        self._bar_index = 0
        self._signals_generated = 0
    
    def run(
        self,
        df_eurusd: pd.DataFrame,
        df_gbpusd: pd.DataFrame,
    ) -> BacktestResult:
        """Run backtest."""
        assert len(df_eurusd) == len(df_gbpusd), "Data mismatch"
        
        n = len(df_eurusd)
        
        for i in range(n):
            self._bar_index = i
            
            eu_bar = df_eurusd.iloc[i]
            gb_bar = df_gbpusd.iloc[i]
            timestamp = eu_bar['timestamp']
            
            # Update filters
            if self.gatekeeper:
                self.gatekeeper.update(eu_bar['close'], gb_bar['close'])
            
            if self.mrf:
                self.mrf.update(
                    timestamp=timestamp,
                    high=eu_bar['high'],
                    low=eu_bar['low'],
                    close=eu_bar['close'],
                )
            
            # Check exit first
            if self.current_trade is not None:
                self._check_exit(eu_bar['high'], eu_bar['low'], timestamp)
            
            # Check MRF BEFORE signal generation
            mrf_allowed = True
            mrf_decision = None
            if self.mrf:
                mrf_decision = self.mrf.evaluate()
                mrf_allowed = mrf_decision.allowed
            
            # Only generate signal if MRF allows (or no MRF)
            signal = None
            if mrf_allowed:
                signal = self.signal_engine.update(
                    timestamp=timestamp,
                    open_=eu_bar['open'],
                    high=eu_bar['high'],
                    low=eu_bar['low'],
                    close=eu_bar['close'],
                )
            else:
                # Still update signal engine state but don't get signal
                self.signal_engine.update(
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
                    self._process_signal(signal, timestamp, mrf_decision)
            elif not mrf_allowed and self.current_trade is None:
                # Check if signal engine WOULD have generated a signal
                # by temporarily checking conditions
                # (For counterfactual - we record MRF blocks)
                pass
            
            self.equity_curve.append(self.equity)
        
        # Force close
        if self.current_trade is not None:
            last = df_eurusd.iloc[-1]
            self._force_close(last['close'], last['timestamp'])
        
        return self._compute_results()
    
    def _process_signal(
        self,
        signal: TradeSignal,
        timestamp: datetime,
        mrf_decision: Optional[MRFDecision],
    ) -> None:
        """Process a trade signal."""
        direction = signal.direction.value
        
        # Check gatekeeper
        if self.gatekeeper:
            gate_decision = self.gatekeeper.evaluate()
            
            if not gate_decision.allowed:
                trade = BacktestTrade(
                    entry_bar=self._bar_index,
                    entry_time=timestamp,
                    direction=direction,
                    entry_price=signal.entry_price,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    position_size=0,
                    blocked_by_gatekeeper=True,
                    gatekeeper_reasons=[r.value for r in gate_decision.reasons],
                )
                self.blocked_trades.append(trade)
                return
        
        # Execute trade
        sl_distance = abs(signal.entry_price - signal.stop_loss)
        risk_amount = self.equity * self.risk_per_trade
        
        if sl_distance > 0:
            position_size = risk_amount / (sl_distance * 100000)
            position_size = max(0.01, min(1.0, round(position_size, 2)))
        else:
            position_size = 0.01
        
        trade = BacktestTrade(
            entry_bar=self._bar_index,
            entry_time=timestamp,
            direction=direction,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            position_size=position_size,
        )
        
        self.current_trade = trade
        self.trades.append(trade)
    
    def _check_exit(self, high: float, low: float, timestamp: datetime) -> None:
        """Check SL/TP."""
        trade = self.current_trade
        
        if trade.direction == "LONG":
            # Check SL first (conservative)
            if low <= trade.stop_loss:
                self._close_trade(trade.stop_loss, timestamp, won=False)
            elif high >= trade.take_profit:
                self._close_trade(trade.take_profit, timestamp, won=True)
        else:
            if high >= trade.stop_loss:
                self._close_trade(trade.stop_loss, timestamp, won=False)
            elif low <= trade.take_profit:
                self._close_trade(trade.take_profit, timestamp, won=True)
    
    def _close_trade(self, exit_price: float, timestamp: datetime, won: bool) -> None:
        """Close trade."""
        trade = self.current_trade
        trade.exit_bar = self._bar_index
        trade.exit_price = exit_price
        trade.won = won
        
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
        
        self.equity += trade.pnl
        self.current_trade = None
    
    def _force_close(self, price: float, timestamp: datetime) -> None:
        """Force close at end."""
        trade = self.current_trade
        trade.exit_bar = self._bar_index
        trade.exit_price = price
        
        if trade.direction == "LONG":
            price_diff = price - trade.entry_price
        else:
            price_diff = trade.entry_price - price
        
        trade.pnl = price_diff * trade.position_size * 100000
        
        sl_distance = abs(trade.entry_price - trade.stop_loss)
        if sl_distance > 0:
            trade.pnl_r = price_diff / sl_distance
        
        trade.won = trade.pnl > 0
        self.equity += trade.pnl
        self.current_trade = None
    
    def _compute_results(self) -> BacktestResult:
        """Compute backtest statistics with expectancy focus."""
        name = {
            "baseline": "Baseline",
            "gatekeeper": "Trend+Gate",
            "full": "Trend+Gate+MRF",
        }[self.mode]
        
        if not self.trades:
            return BacktestResult(
                name=name,
                trades=self.trades,
                blocked_trades=self.blocked_trades,
                equity_curve=self.equity_curve,
                signals_generated=self._signals_generated,
            )
        
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
        
        # EXPECTANCY (key metric)
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
        
        # Block stats
        blocked_gate = len([t for t in self.blocked_trades if t.blocked_by_gatekeeper])
        blocked_mrf = len([t for t in self.blocked_trades if t.blocked_by_mrf])
        
        return BacktestResult(
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
            total_pnl=total_pnl,
            max_drawdown_pct=max_dd_pct,
            max_drawdown_abs=max_dd_abs,
            signals_generated=self._signals_generated,
            blocked_by_gatekeeper=blocked_gate,
            blocked_by_mrf=blocked_mrf,
        )


class MRFCounterfactualAnalyzer:
    """
    Analyze what MRF-blocked signals would have done.
    """
    
    def __init__(
        self,
        initial_capital: float = 100.0,
        risk_per_trade: float = 0.005,
    ):
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
    
    def run(
        self,
        df_eurusd: pd.DataFrame,
        df_gbpusd: pd.DataFrame,
    ) -> Dict:
        """
        Run MRF counterfactual analysis.
        
        Runs baseline, identifies signals that would be blocked by MRF,
        and analyzes their outcomes.
        """
        # Run baseline to get all signals
        baseline = SingleStrategyBacktest(
            mode="baseline",
            initial_capital=self.initial_capital,
            risk_per_trade=self.risk_per_trade,
        )
        baseline_result = baseline.run(df_eurusd.copy(), df_gbpusd.copy())
        
        # Run with MRF to identify blocks
        full = SingleStrategyBacktest(
            mode="full",
            initial_capital=self.initial_capital,
            risk_per_trade=self.risk_per_trade,
        )
        full_result = full.run(df_eurusd.copy(), df_gbpusd.copy())
        
        # Identify MRF-blocked trades by comparing entry bars
        full_entry_bars = {t.entry_bar for t in full_result.trades}
        full_blocked_bars = {t.entry_bar for t in full_result.blocked_trades}
        
        # Trades in baseline but not in full (and not blocked by gatekeeper)
        # These are MRF blocks
        mrf_blocked_trades = []
        for trade in baseline_result.trades:
            if trade.entry_bar not in full_entry_bars and trade.entry_bar not in full_blocked_bars:
                mrf_blocked_trades.append(trade)
        
        # Also add gatekeeper-blocked trades from full (they would have been blocked anyway)
        # But we want to isolate MRF-only blocks
        
        # Analyze MRF-blocked outcomes
        if not mrf_blocked_trades:
            return {
                'total_blocked': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0,
                'pnl': 0,
                'expectancy': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'mrf_effectiveness': 'N/A (no blocks)',
            }
        
        wins = [t for t in mrf_blocked_trades if t.won]
        losses = [t for t in mrf_blocked_trades if not t.won]
        
        total = len(mrf_blocked_trades)
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = win_count / total if total > 0 else 0
        
        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0.0001
        
        pnl = sum(t.pnl for t in mrf_blocked_trades)
        avg_win = gross_profit / win_count if win_count > 0 else 0
        avg_loss = gross_loss / loss_count if loss_count > 0 else 0
        
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        # MRF effectiveness: if blocked trades have negative expectancy, MRF is working
        if expectancy < 0:
            effectiveness = f"EFFECTIVE (blocked trades had ${expectancy:.2f} expectancy)"
        elif expectancy > 0:
            effectiveness = f"HARMFUL (blocked trades had ${expectancy:.2f} positive expectancy)"
        else:
            effectiveness = "NEUTRAL"
        
        return {
            'total_blocked': total,
            'wins': win_count,
            'losses': loss_count,
            'win_rate': win_rate,
            'pnl': pnl,
            'expectancy': expectancy,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'mrf_effectiveness': effectiveness,
        }


def run_single_strategy_comparison(
    df_eurusd: pd.DataFrame,
    df_gbpusd: pd.DataFrame,
    initial_capital: float = 100.0,
) -> Tuple[BacktestResult, BacktestResult, BacktestResult, Dict]:
    """
    Run 3-way comparison:
    1. Baseline
    2. Trend + Gatekeeper
    3. Trend + Gatekeeper + MRF
    
    Plus MRF counterfactual analysis.
    """
    # 1. Baseline
    print("  Running BASELINE (no filters)...")
    baseline_bt = SingleStrategyBacktest(
        mode="baseline",
        initial_capital=initial_capital,
    )
    baseline = baseline_bt.run(df_eurusd.copy(), df_gbpusd.copy())
    print(f"    {baseline.total_trades} trades, Exp: ${baseline.expectancy:.2f}")
    
    # 2. Gatekeeper only
    print("  Running TREND + GATEKEEPER...")
    gate_bt = SingleStrategyBacktest(
        mode="gatekeeper",
        initial_capital=initial_capital,
    )
    gate = gate_bt.run(df_eurusd.copy(), df_gbpusd.copy())
    print(f"    {gate.total_trades} trades, Exp: ${gate.expectancy:.2f}, Blocked: {gate.blocked_by_gatekeeper}")
    
    # 3. Full (Gatekeeper + MRF)
    print("  Running TREND + GATEKEEPER + MRF...")
    full_bt = SingleStrategyBacktest(
        mode="full",
        initial_capital=initial_capital,
    )
    full = full_bt.run(df_eurusd.copy(), df_gbpusd.copy())
    print(f"    {full.total_trades} trades, Exp: ${full.expectancy:.2f}")
    
    # 4. MRF Counterfactual
    print("  Running MRF COUNTERFACTUAL...")
    cf_analyzer = MRFCounterfactualAnalyzer(initial_capital=initial_capital)
    mrf_cf = cf_analyzer.run(df_eurusd.copy(), df_gbpusd.copy())
    print(f"    MRF blocked: {mrf_cf['total_blocked']}, Their WR: {mrf_cf['win_rate']:.1%}")
    
    return baseline, gate, full, mrf_cf


def print_comparison_table(
    baseline: BacktestResult,
    gate: BacktestResult,
    full: BacktestResult,
):
    """Print comparison table."""
    print()
    print("=" * 85)
    print("SINGLE-STRATEGY BACKTEST COMPARISON")
    print("=" * 85)
    print()
    
    header = f"{'Metric':<25} {'Baseline':>15} {'Trend+Gate':>15} {'Trend+Gate+MRF':>15}"
    print(header)
    print("-" * 85)
    
    def fmt(v, t='f'):
        if t == 'p': return f"{v:.1%}"
        if t == 'd': return f"${v:.2f}"
        if t == 'i': return f"{int(v)}"
        if t == 'r': return f"{v:.2f}R"
        return f"{v:.2f}"
    
    rows = [
        ("Trades", baseline.total_trades, gate.total_trades, full.total_trades, 'i'),
        ("Wins", baseline.wins, gate.wins, full.wins, 'i'),
        ("Losses", baseline.losses, gate.losses, full.losses, 'i'),
        ("Win Rate", baseline.win_rate, gate.win_rate, full.win_rate, 'p'),
        ("Profit Factor", baseline.profit_factor, gate.profit_factor, full.profit_factor, 'f'),
        ("", "", "", "", ''),  # Separator
        ("EXPECTANCY ($)", baseline.expectancy, gate.expectancy, full.expectancy, 'd'),
        ("Expectancy (R)", baseline.expectancy_r, gate.expectancy_r, full.expectancy_r, 'r'),
        ("", "", "", "", ''),
        ("Avg Win ($)", baseline.avg_win, gate.avg_win, full.avg_win, 'd'),
        ("Avg Loss ($)", baseline.avg_loss, gate.avg_loss, full.avg_loss, 'd'),
        ("Avg Win (R)", baseline.avg_win_r, gate.avg_win_r, full.avg_win_r, 'r'),
        ("Avg Loss (R)", baseline.avg_loss_r, gate.avg_loss_r, full.avg_loss_r, 'r'),
        ("", "", "", "", ''),
        ("Max Drawdown", baseline.max_drawdown_pct, gate.max_drawdown_pct, full.max_drawdown_pct, 'p'),
        ("Max DD ($)", baseline.max_drawdown_abs, gate.max_drawdown_abs, full.max_drawdown_abs, 'd'),
        ("Net PnL", baseline.total_pnl, gate.total_pnl, full.total_pnl, 'd'),
        ("", "", "", "", ''),
        ("Blocked (Gate)", "-", gate.blocked_by_gatekeeper, full.blocked_by_gatekeeper, 'i'),
        ("Blocked (MRF)", "-", "-", full.blocked_by_mrf, 'i'),
    ]
    
    for label, b, g, f, t in rows:
        if label == "":
            print()
            continue
        
        if t == '':
            print()
            continue
        
        if b == "-":
            b_str = "-"
        else:
            b_str = fmt(b, t)
        
        if g == "-":
            g_str = "-"
        else:
            g_str = fmt(g, t)
        
        if f == "-":
            f_str = "-"
        else:
            f_str = fmt(f, t)
        
        print(f"{label:<25} {b_str:>15} {g_str:>15} {f_str:>15}")
    
    print()


def print_mrf_counterfactual(cf: Dict):
    """Print MRF counterfactual analysis."""
    print("=" * 85)
    print("MRF COUNTERFACTUAL ANALYSIS")
    print("=" * 85)
    print()
    print("What would MRF-blocked trades have done if executed?")
    print()
    print(f"  Total Blocked by MRF:    {cf['total_blocked']}")
    print(f"  Would Have Won:          {cf['wins']}")
    print(f"  Would Have Lost:         {cf['losses']}")
    print(f"  Counterfactual Win Rate: {cf['win_rate']:.1%}")
    print(f"  Counterfactual PnL:      ${cf['pnl']:.2f}")
    print(f"  Counterfactual Expectancy: ${cf['expectancy']:.2f}")
    print()
    print(f"  MRF Effectiveness: {cf['mrf_effectiveness']}")
    print()


def determine_viability(full: BacktestResult) -> Tuple[str, str]:
    """
    Determine if strategy is viable for real capital.
    
    Returns: (verdict, explanation)
    """
    exp = full.expectancy
    exp_r = full.expectancy_r
    pf = full.profit_factor
    wr = full.win_rate
    dd = full.max_drawdown_pct
    trades = full.total_trades
    
    print("=" * 85)
    print("VIABILITY ASSESSMENT")
    print("=" * 85)
    print()
    
    issues = []
    
    # Check expectancy
    if exp <= 0:
        issues.append(f"CRITICAL: Expectancy is ${exp:.2f} (≤ $0)")
    elif exp < 0.10:
        issues.append(f"WARNING: Expectancy is low (${exp:.2f})")
    
    # Check R-expectancy
    if exp_r <= 0:
        issues.append(f"CRITICAL: R-Expectancy is {exp_r:.2f}R (≤ 0R)")
    elif exp_r < 0.1:
        issues.append(f"WARNING: R-Expectancy is low ({exp_r:.2f}R)")
    
    # Check profit factor
    if pf < 1.0:
        issues.append(f"CRITICAL: Profit Factor is {pf:.2f} (< 1.0)")
    elif pf < 1.2:
        issues.append(f"WARNING: Profit Factor is marginal ({pf:.2f})")
    
    # Check drawdown
    if dd > 0.20:
        issues.append(f"CRITICAL: Max Drawdown is {dd:.1%} (> 20%)")
    elif dd > 0.10:
        issues.append(f"WARNING: Max Drawdown is elevated ({dd:.1%})")
    
    # Check sample size
    if trades < 30:
        issues.append(f"WARNING: Sample size is small ({trades} trades)")
    
    # Print issues
    for issue in issues:
        print(f"  • {issue}")
    
    print()
    
    # Determine verdict
    critical_count = sum(1 for i in issues if i.startswith("CRITICAL"))
    warning_count = sum(1 for i in issues if i.startswith("WARNING"))
    
    if critical_count > 0:
        verdict = "NOT VIABLE"
        explanation = (
            "Strategy has CRITICAL issues and is NOT recommended for real capital.\n"
            "DO NOT deploy this system live. Further research or abandonment is required."
        )
    elif warning_count >= 2:
        verdict = "MARGINAL"
        explanation = (
            "Strategy has multiple warnings. Proceed with EXTREME CAUTION.\n"
            "If deployed, use minimal capital and strict risk limits."
        )
    elif warning_count == 1:
        verdict = "CAUTIOUS PROCEED"
        explanation = (
            "Strategy shows promise but has concerns. Proceed with caution.\n"
            "Extended forward testing recommended before scaling."
        )
    else:
        verdict = "VIABLE"
        explanation = (
            "Strategy meets basic viability criteria.\n"
            "Proceed with paper trading, then small live capital."
        )
    
    print(f"VERDICT: {verdict}")
    print()
    print(f"  {explanation}")
    print()
    print("=" * 85)
    
    return verdict, explanation
