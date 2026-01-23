"""
Session Strategy Exit Comparison Backtest

Compares multiple exit models on the same entry signals:
1. BASELINE: 2.5R single target (current)
2. MULTI_TARGET: 50% at 1.0R, 30% at 2.0R, 20% runner
3. REDUCED_TP: 1.5R single target
4. AGGRESSIVE: 1.0R single target

Focus: Improve expectancy without changing entry logic.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone, time as dt_time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_system.config import GatekeeperConfig
from trading_system.session_engine import (
    SessionEngine,
    SessionSignal,
    SessionDirection,
)
from trading_system.gatekeeper_engine import GatekeeperEngine
from trading_system.exit_models import (
    ExitModel,
    ExitConfig,
    ExitManager,
    PositionSlice,
)


@dataclass
class ExitTrade:
    """Trade record with detailed exit info."""
    trade_date: datetime
    entry_bar: int
    entry_time: datetime
    direction: str
    entry_price: float
    stop_loss: float
    position_size: float
    
    # Session data
    asia_high: float = 0.0
    asia_low: float = 0.0
    asia_range_pips: float = 0.0
    
    # Exit info
    exit_model: str = ""
    exit_bar: Optional[int] = None
    exit_time: Optional[datetime] = None
    primary_exit_reason: str = ""
    
    # Slice-level results
    slices_summary: List[dict] = field(default_factory=list)
    
    # Aggregate results
    total_pnl_r: float = 0.0
    total_pnl_usd: float = 0.0
    won: bool = False
    
    # Tracking
    highest_r_reached: float = 0.0
    breakeven_activated: bool = False
    
    # Filter info
    blocked_by_gatekeeper: bool = False


@dataclass
class ExitModelResult:
    """Results for a single exit model."""
    model: ExitModel
    trades: List[ExitTrade]
    equity_curve: List[float]
    
    # Core metrics
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    
    # Expectancy
    profit_factor: float = 0.0
    expectancy_r: float = 0.0
    expectancy_usd: float = 0.0
    avg_r_per_trade: float = 0.0
    
    # Exit analysis
    exits_by_tp: Dict[str, int] = field(default_factory=dict)
    exits_by_sl: int = 0
    exits_by_time: int = 0
    pct_tp_hits: float = 0.0
    
    # Risk
    total_pnl_r: float = 0.0
    total_pnl_usd: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_usd: float = 0.0
    
    # Trade characteristics
    avg_highest_r: float = 0.0
    pct_breakeven_activated: float = 0.0


class ExitComparisonBacktest:
    """
    Runs session strategy with multiple exit models on same signals.
    
    This ensures fair comparison - same entries, different exits.
    """
    
    def __init__(
        self,
        initial_capital: float = 100.0,
        risk_per_trade: float = 0.005,
    ):
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        
        # Shared components
        self.session_engine = SessionEngine()
        self.gatekeeper = GatekeeperEngine(GatekeeperConfig())
        
        # Exit models to test
        self.exit_configs = {
            ExitModel.BASELINE: ExitConfig.baseline(),
            ExitModel.MULTI_TARGET: ExitConfig.multi_target(),
            ExitModel.REDUCED_TP: ExitConfig.reduced_tp(),
            ExitModel.AGGRESSIVE: ExitConfig.aggressive(),
        }
    
    def run(
        self,
        df_eurusd: pd.DataFrame,
        df_gbpusd: pd.DataFrame,
    ) -> Dict[ExitModel, ExitModelResult]:
        """
        Run backtest with all exit models.
        
        First pass: Collect all valid entry signals
        Second pass: Simulate each exit model on same signals
        """
        # Pass 1: Collect signals
        signals = self._collect_signals(df_eurusd, df_gbpusd)
        print(f"  Collected {len(signals)} valid entry signals")
        
        # Pass 2: Run each exit model
        results = {}
        for model, config in self.exit_configs.items():
            print(f"  Testing {model.value}...")
            result = self._run_exit_model(signals, df_eurusd, config)
            results[model] = result
            print(f"    Trades: {result.total_trades}, Exp: {result.expectancy_r:.2f}R, TP%: {result.pct_tp_hits:.1%}")
        
        return results
    
    def _collect_signals(
        self,
        df_eurusd: pd.DataFrame,
        df_gbpusd: pd.DataFrame,
    ) -> List[Tuple[int, SessionSignal, bool]]:
        """
        Collect all valid entry signals with gatekeeper status.
        
        Returns: List of (bar_index, signal, gatekeeper_passed)
        """
        signals = []
        n = len(df_eurusd)
        
        for i in range(n):
            eu_bar = df_eurusd.iloc[i]
            gb_bar = df_gbpusd.iloc[i]
            timestamp = eu_bar['timestamp']
            
            # Update gatekeeper
            self.gatekeeper.update(eu_bar['close'], gb_bar['close'])
            
            # Generate signal
            signal = self.session_engine.update(
                timestamp=timestamp,
                open_=eu_bar['open'],
                high=eu_bar['high'],
                low=eu_bar['low'],
                close=eu_bar['close'],
            )
            
            if signal is not None:
                gate_decision = self.gatekeeper.evaluate()
                signals.append((i, signal, gate_decision.allowed))
        
        return signals
    
    def _run_exit_model(
        self,
        signals: List[Tuple[int, SessionSignal, bool]],
        df_eurusd: pd.DataFrame,
        config: ExitConfig,
    ) -> ExitModelResult:
        """Run a single exit model on collected signals."""
        trades: List[ExitTrade] = []
        equity = self.initial_capital
        equity_curve = [equity]
        
        exit_manager = ExitManager(config)
        current_trade: Optional[ExitTrade] = None
        
        n = len(df_eurusd)
        signal_idx = 0
        
        for i in range(n):
            eu_bar = df_eurusd.iloc[i]
            timestamp = eu_bar['timestamp']
            high = eu_bar['high']
            low = eu_bar['low']
            close = eu_bar['close']
            
            # Check for exit on current trade
            if current_trade is not None and exit_manager.is_active:
                closed, closed_slices, reason = exit_manager.update(
                    high=high,
                    low=low,
                    close=close,
                    timestamp=timestamp,
                )
                
                if closed:
                    # Finalize trade
                    current_trade.exit_bar = i
                    current_trade.exit_time = timestamp
                    current_trade.primary_exit_reason = reason
                    
                    summary = exit_manager.get_exit_summary()
                    current_trade.slices_summary = summary['slices']
                    current_trade.total_pnl_r = summary['total_pnl_r']
                    current_trade.highest_r_reached = summary['highest_r_reached']
                    current_trade.breakeven_activated = summary['breakeven_activated']
                    
                    # Calculate USD PnL
                    sl_distance = abs(current_trade.entry_price - current_trade.stop_loss)
                    risk_usd = equity * self.risk_per_trade
                    current_trade.total_pnl_usd = current_trade.total_pnl_r * risk_usd
                    current_trade.won = current_trade.total_pnl_r > 0
                    
                    equity += current_trade.total_pnl_usd
                    trades.append(current_trade)
                    current_trade = None
                    exit_manager.reset()
            
            # Check for new signal at this bar
            while signal_idx < len(signals) and signals[signal_idx][0] == i:
                bar_idx, signal, gate_passed = signals[signal_idx]
                signal_idx += 1
                
                # Skip if position already open
                if current_trade is not None:
                    continue
                
                # Skip if gatekeeper blocked
                if not gate_passed:
                    blocked_trade = ExitTrade(
                        trade_date=signal.date,
                        entry_bar=i,
                        entry_time=timestamp,
                        direction=signal.direction.value,
                        entry_price=signal.entry_price,
                        stop_loss=signal.stop_loss,
                        position_size=0,
                        asia_high=signal.asia_high,
                        asia_low=signal.asia_low,
                        asia_range_pips=signal.asia_range_pips,
                        exit_model=config.model.value,
                        blocked_by_gatekeeper=True,
                    )
                    # Don't add blocked trades to main list
                    continue
                
                # Open new trade
                sl_distance = abs(signal.entry_price - signal.stop_loss)
                risk_usd = equity * self.risk_per_trade
                position_size = risk_usd / (sl_distance * 100000)
                position_size = max(0.01, min(1.0, round(position_size, 2)))
                
                current_trade = ExitTrade(
                    trade_date=signal.date,
                    entry_bar=i,
                    entry_time=timestamp,
                    direction=signal.direction.value,
                    entry_price=signal.entry_price,
                    stop_loss=signal.stop_loss,
                    position_size=position_size,
                    asia_high=signal.asia_high,
                    asia_low=signal.asia_low,
                    asia_range_pips=signal.asia_range_pips,
                    exit_model=config.model.value,
                )
                
                exit_manager.open_position(
                    entry_price=signal.entry_price,
                    stop_loss=signal.stop_loss,
                    direction=signal.direction.value,
                )
            
            equity_curve.append(equity)
        
        # Force close if still open
        if current_trade is not None and exit_manager.is_active:
            last = df_eurusd.iloc[-1]
            closed_slices = exit_manager.force_close(last['close'], "END")
            
            current_trade.exit_bar = n - 1
            current_trade.exit_time = last['timestamp']
            current_trade.primary_exit_reason = "END"
            
            summary = exit_manager.get_exit_summary()
            current_trade.slices_summary = summary['slices']
            current_trade.total_pnl_r = summary['total_pnl_r']
            current_trade.highest_r_reached = summary['highest_r_reached']
            current_trade.breakeven_activated = summary['breakeven_activated']
            
            sl_distance = abs(current_trade.entry_price - current_trade.stop_loss)
            risk_usd = equity * self.risk_per_trade
            current_trade.total_pnl_usd = current_trade.total_pnl_r * risk_usd
            current_trade.won = current_trade.total_pnl_r > 0
            
            equity += current_trade.total_pnl_usd
            trades.append(current_trade)
        
        return self._compute_result(config.model, trades, equity_curve)
    
    def _compute_result(
        self,
        model: ExitModel,
        trades: List[ExitTrade],
        equity_curve: List[float],
    ) -> ExitModelResult:
        """Compute statistics for an exit model."""
        if not trades:
            return ExitModelResult(
                model=model,
                trades=trades,
                equity_curve=equity_curve,
            )
        
        # Basic counts
        total = len(trades)
        wins = [t for t in trades if t.won]
        losses = [t for t in trades if not t.won]
        
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = win_count / total if total > 0 else 0
        
        # PnL analysis
        total_pnl_r = sum(t.total_pnl_r for t in trades)
        total_pnl_usd = sum(t.total_pnl_usd for t in trades)
        
        gross_profit_r = sum(t.total_pnl_r for t in wins) if wins else 0
        gross_loss_r = abs(sum(t.total_pnl_r for t in losses)) if losses else 0.0001
        
        pf = gross_profit_r / gross_loss_r if gross_loss_r > 0 else 0
        
        # Expectancy
        avg_r = total_pnl_r / total if total > 0 else 0
        avg_usd = total_pnl_usd / total if total > 0 else 0
        
        # Exit analysis
        exits_by_tp = {}
        exits_by_sl = 0
        exits_by_time = 0
        tp_hits = 0
        
        for t in trades:
            reason = t.primary_exit_reason
            if reason == "SL":
                exits_by_sl += 1
            elif reason == "TIME":
                exits_by_time += 1
            elif reason == "ALL_TP" or reason.startswith("TP"):
                tp_hits += 1
                exits_by_tp[reason] = exits_by_tp.get(reason, 0) + 1
            
            # Also count partial TPs in multi-target
            for s in t.slices_summary:
                if s['exit_reason'].startswith("TP"):
                    key = s['exit_reason']
                    exits_by_tp[key] = exits_by_tp.get(key, 0) + 1
        
        pct_tp = tp_hits / total if total > 0 else 0
        
        # Drawdown
        eq = np.array(equity_curve)
        running_max = np.maximum.accumulate(eq)
        dd_abs = running_max - eq
        dd_pct = np.where(running_max > 0, dd_abs / running_max, 0)
        max_dd_pct = np.max(dd_pct) if len(dd_pct) > 0 else 0
        max_dd_usd = np.max(dd_abs) if len(dd_abs) > 0 else 0
        
        # Trade characteristics
        avg_highest_r = np.mean([t.highest_r_reached for t in trades])
        pct_be = sum(1 for t in trades if t.breakeven_activated) / total if total > 0 else 0
        
        return ExitModelResult(
            model=model,
            trades=trades,
            equity_curve=equity_curve,
            total_trades=total,
            wins=win_count,
            losses=loss_count,
            win_rate=win_rate,
            profit_factor=pf,
            expectancy_r=avg_r,
            expectancy_usd=avg_usd,
            avg_r_per_trade=avg_r,
            exits_by_tp=exits_by_tp,
            exits_by_sl=exits_by_sl,
            exits_by_time=exits_by_time,
            pct_tp_hits=pct_tp,
            total_pnl_r=total_pnl_r,
            total_pnl_usd=total_pnl_usd,
            max_drawdown_pct=max_dd_pct,
            max_drawdown_usd=max_dd_usd,
            avg_highest_r=avg_highest_r,
            pct_breakeven_activated=pct_be,
        )


def print_exit_comparison(results: Dict[ExitModel, ExitModelResult]):
    """Print comparison table of all exit models."""
    print()
    print("=" * 95)
    print("EXIT MODEL COMPARISON")
    print("Same entry signals, different exit logic")
    print("=" * 95)
    print()
    
    models = [ExitModel.BASELINE, ExitModel.MULTI_TARGET, ExitModel.REDUCED_TP, ExitModel.AGGRESSIVE]
    
    header = f"{'Metric':<25}"
    for m in models:
        header += f" {m.value:>15}"
    print(header)
    print("-" * 95)
    
    def fmt(v, t='f'):
        if t == 'p': return f"{v:.1%}"
        if t == 'd': return f"${v:.2f}"
        if t == 'r': return f"{v:.2f}R"
        if t == 'i': return f"{int(v)}"
        return f"{v:.2f}"
    
    def row(label, key, t='f'):
        line = f"{label:<25}"
        for m in models:
            r = results[m]
            v = getattr(r, key)
            line += f" {fmt(v, t):>15}"
        return line
    
    print(row("Trades", "total_trades", 'i'))
    print(row("Wins", "wins", 'i'))
    print(row("Losses", "losses", 'i'))
    print(row("Win Rate", "win_rate", 'p'))
    print()
    print(row("EXPECTANCY (R)", "expectancy_r", 'r'))
    print(row("Expectancy ($)", "expectancy_usd", 'd'))
    print(row("Profit Factor", "profit_factor", 'f'))
    print()
    print(row("Total PnL (R)", "total_pnl_r", 'r'))
    print(row("Total PnL ($)", "total_pnl_usd", 'd'))
    print()
    print(row("Exits by SL", "exits_by_sl", 'i'))
    print(row("Exits by TIME", "exits_by_time", 'i'))
    print(row("% TP Hits", "pct_tp_hits", 'p'))
    print()
    print(row("Max Drawdown", "max_drawdown_pct", 'p'))
    print(row("Max DD ($)", "max_drawdown_usd", 'd'))
    print()
    print(row("Avg Highest R Reached", "avg_highest_r", 'r'))
    print(row("% Breakeven Activated", "pct_breakeven_activated", 'p'))
    print()


def print_detailed_exit_analysis(results: Dict[ExitModel, ExitModelResult]):
    """Print detailed exit breakdown for multi-target."""
    print("=" * 95)
    print("DETAILED EXIT ANALYSIS")
    print("=" * 95)
    print()
    
    for model in [ExitModel.MULTI_TARGET]:
        r = results[model]
        print(f"{model.value}:")
        print(f"  TP breakdown:")
        for tp, count in sorted(r.exits_by_tp.items()):
            pct = count / r.total_trades if r.total_trades > 0 else 0
            print(f"    {tp}: {count} ({pct:.1%})")
        print(f"  SL exits: {r.exits_by_sl} ({r.exits_by_sl/r.total_trades:.1%})")
        print(f"  TIME exits: {r.exits_by_time} ({r.exits_by_time/r.total_trades:.1%})")
        print()
    
    # Price excursion analysis
    print("PRICE EXCURSION ANALYSIS (How far did price go in our favor?)")
    print()
    
    for model in [ExitModel.BASELINE, ExitModel.MULTI_TARGET]:
        r = results[model]
        excursions = [t.highest_r_reached for t in r.trades]
        
        if excursions:
            pct_above_1r = sum(1 for e in excursions if e >= 1.0) / len(excursions)
            pct_above_1_5r = sum(1 for e in excursions if e >= 1.5) / len(excursions)
            pct_above_2r = sum(1 for e in excursions if e >= 2.0) / len(excursions)
            pct_above_2_5r = sum(1 for e in excursions if e >= 2.5) / len(excursions)
            
            print(f"  {model.value}:")
            print(f"    Avg highest R:   {np.mean(excursions):.2f}R")
            print(f"    Reached ≥1.0R:   {pct_above_1r:.1%}")
            print(f"    Reached ≥1.5R:   {pct_above_1_5r:.1%}")
            print(f"    Reached ≥2.0R:   {pct_above_2r:.1%}")
            print(f"    Reached ≥2.5R:   {pct_above_2_5r:.1%}")
            print()


def determine_best_exit(
    results: Dict[ExitModel, ExitModelResult],
    baseline_dd: float,
) -> Tuple[ExitModel, str, str]:
    """
    Determine best exit model based on criteria.
    
    Returns: (best_model, verdict, explanation)
    """
    print("=" * 95)
    print("EXIT MODEL EVALUATION")
    print("=" * 95)
    print()
    
    # Evaluation criteria
    # 1. Expectancy must be > 0
    # 2. DD must not increase > 20% vs baseline
    # 3. Higher expectancy is better
    
    baseline = results[ExitModel.BASELINE]
    
    candidates = []
    
    for model, r in results.items():
        issues = []
        
        # Kill: Expectancy <= 0
        if r.expectancy_r <= 0:
            issues.append(f"KILL: Expectancy {r.expectancy_r:.2f}R ≤ 0")
        
        # Kill: DD increased > 20%
        dd_increase = (r.max_drawdown_pct - baseline_dd) / baseline_dd if baseline_dd > 0 else 0
        if dd_increase > 0.20:
            issues.append(f"KILL: DD increased {dd_increase:.1%} vs baseline")
        
        # Score: Higher expectancy is better
        score = r.expectancy_r
        
        print(f"{model.value}:")
        print(f"  Expectancy: {r.expectancy_r:.2f}R")
        print(f"  DD: {r.max_drawdown_pct:.1%} (vs baseline {baseline_dd:.1%})")
        print(f"  TP Hit Rate: {r.pct_tp_hits:.1%}")
        
        if issues:
            for issue in issues:
                print(f"  ❌ {issue}")
            print(f"  VERDICT: REJECT")
        else:
            exp_improvement = (r.expectancy_r - baseline.expectancy_r) / abs(baseline.expectancy_r) if baseline.expectancy_r != 0 else 0
            print(f"  ✓ Valid candidate")
            print(f"  Expectancy vs baseline: {exp_improvement:+.1%}")
            candidates.append((model, score, r))
        
        print()
    
    # Select best
    if not candidates:
        return ExitModel.BASELINE, "NO_IMPROVEMENT", "All alternatives failed kill criteria"
    
    candidates.sort(key=lambda x: x[1], reverse=True)
    best_model, best_score, best_result = candidates[0]
    
    # Check if best is actually better than baseline
    if best_model == ExitModel.BASELINE:
        return best_model, "KEEP_BASELINE", "Baseline is the best option"
    
    improvement = (best_score - baseline.expectancy_r) / abs(baseline.expectancy_r) if baseline.expectancy_r != 0 else 0
    
    if improvement > 0.10:  # >10% improvement
        verdict = "ADOPT"
        explanation = f"{best_model.value} improves expectancy by {improvement:.1%}"
    elif improvement > 0:
        verdict = "MARGINAL"
        explanation = f"{best_model.value} shows minor improvement ({improvement:.1%})"
    else:
        verdict = "KEEP_BASELINE"
        explanation = "No meaningful improvement over baseline"
    
    return best_model, verdict, explanation


def print_recommendation(
    best_model: ExitModel,
    verdict: str,
    explanation: str,
    results: Dict[ExitModel, ExitModelResult],
):
    """Print final recommendation."""
    print("=" * 95)
    print("RECOMMENDATION")
    print("=" * 95)
    print()
    
    baseline = results[ExitModel.BASELINE]
    best = results[best_model]
    
    print(f"  Verdict: {verdict}")
    print(f"  Best Exit Model: {best_model.value}")
    print(f"  Explanation: {explanation}")
    print()
    
    if best_model != ExitModel.BASELINE:
        print("  Comparison vs Baseline:")
        print(f"    Expectancy: {baseline.expectancy_r:.2f}R → {best.expectancy_r:.2f}R ({(best.expectancy_r/baseline.expectancy_r - 1)*100:+.0f}%)")
        print(f"    Win Rate:   {baseline.win_rate:.1%} → {best.win_rate:.1%}")
        print(f"    TP Hits:    {baseline.pct_tp_hits:.1%} → {best.pct_tp_hits:.1%}")
        print(f"    Max DD:     {baseline.max_drawdown_pct:.1%} → {best.max_drawdown_pct:.1%}")
        print()
    
    # Next steps
    print("  NEXT STEPS:")
    if verdict == "ADOPT":
        print(f"    1. Implement {best_model.value} exit logic in production")
        print("    2. Paper trade for 1 month to validate")
        print("    3. Monitor TP hit rates match backtest")
    elif verdict == "MARGINAL":
        print(f"    1. Consider {best_model.value} for further testing")
        print("    2. Extended paper trading recommended")
        print("    3. May not be worth the complexity")
    elif verdict == "KEEP_BASELINE":
        print("    1. Keep current exit logic")
        print("    2. Focus on other improvements (entry timing, filters)")
        print("    3. Consider different strategy type")
    else:
        print("    1. Exit logic changes do not fix fundamental edge issue")
        print("    2. Review entry logic or abandon strategy")
    
    print()
    print("=" * 95)
