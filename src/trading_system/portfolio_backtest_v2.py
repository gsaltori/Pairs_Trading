"""
Portfolio Backtest Harness

Tests the micro-edge portfolio system:
1. Individual strategy performance
2. Combined portfolio performance
3. Correlation analysis
4. Risk metrics
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone, time as dt_time, timedelta
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_system.config import GatekeeperConfig
from trading_system.gatekeeper_engine import GatekeeperEngine
from trading_system.bias_engine import SessionBiasEngine, DirectionalBias
from trading_system.micro_strategies import (
    MicroEdgePortfolio,
    MicroSignal,
    MicroEdgeDirection,
    LondonPullbackScalper,
    MomentumBurstStrategy,
    PivotBounceStrategy,
)
from trading_system.portfolio_risk import PortfolioRiskManager


@dataclass
class PortfolioTrade:
    """Trade record for portfolio backtest."""
    trade_id: str
    strategy: str
    timestamp: datetime
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    risk_fraction: float
    
    # Bias context
    bias: str
    bias_confidence: float
    
    # Exit
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""
    
    # Results
    pnl: float = 0.0
    pnl_r: float = 0.0
    won: bool = False


@dataclass
class StrategyMetrics:
    """Metrics for a single strategy."""
    name: str
    trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_pnl_r: float = 0.0
    expectancy_r: float = 0.0
    profit_factor: float = 0.0
    avg_win_r: float = 0.0
    avg_loss_r: float = 0.0


@dataclass 
class PortfolioMetrics:
    """Comprehensive portfolio metrics."""
    # Trade stats
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    
    # PnL
    total_pnl: float = 0.0
    total_pnl_r: float = 0.0
    expectancy_r: float = 0.0
    profit_factor: float = 0.0
    
    # Risk
    max_drawdown_pct: float = 0.0
    max_drawdown_abs: float = 0.0
    sharpe_ratio: float = 0.0
    
    # Frequency
    trades_per_month: float = 0.0
    avg_trades_per_day: float = 0.0
    
    # Strategy breakdown
    strategy_metrics: Dict[str, StrategyMetrics] = field(default_factory=dict)
    
    # Correlation
    long_short_ratio: float = 0.0
    max_concurrent: int = 0
    
    # Kill switch stats
    daily_halts: int = 0
    weekly_halts: int = 0
    consecutive_loss_halts: int = 0


class PortfolioBacktest:
    """
    Backtest the micro-edge portfolio system.
    
    Runs all strategies with shared bias engine and gatekeeper.
    Applies portfolio-level risk management.
    """
    
    def __init__(
        self,
        initial_capital: float = 100.0,
        use_gatekeeper: bool = True,
    ):
        self.initial_capital = initial_capital
        self.use_gatekeeper = use_gatekeeper
        
        # Components
        self.portfolio = MicroEdgePortfolio()
        self.risk_manager = PortfolioRiskManager(initial_capital)
        self.gatekeeper = GatekeeperEngine(GatekeeperConfig()) if use_gatekeeper else None
        
        # State
        self.trades: List[PortfolioTrade] = []
        self.open_trades: Dict[str, PortfolioTrade] = {}
        self.equity_curve: List[float] = []
        
        self._trade_counter = 0
        self._current_date: Optional[datetime] = None
        
        # Stats
        self._daily_halts = 0
        self._weekly_halts = 0
        self._consec_halts = 0
        self._max_concurrent = 0
    
    def run(
        self,
        df_eurusd: pd.DataFrame,
        df_gbpusd: pd.DataFrame,
    ) -> PortfolioMetrics:
        """Run portfolio backtest."""
        n = len(df_eurusd)
        
        for i in range(n):
            eu_bar = df_eurusd.iloc[i]
            gb_bar = df_gbpusd.iloc[i]
            timestamp = eu_bar['timestamp']
            
            # New day check
            session_date = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
            if self._current_date != session_date:
                self._current_date = session_date
                self.risk_manager.new_day(session_date)
            
            # Update gatekeeper
            if self.gatekeeper:
                self.gatekeeper.update(eu_bar['close'], gb_bar['close'])
            
            # Check exits for open trades
            self._check_exits(
                high=eu_bar['high'],
                low=eu_bar['low'],
                close=eu_bar['close'],
                timestamp=timestamp,
            )
            
            # Track concurrent positions
            self._max_concurrent = max(self._max_concurrent, len(self.open_trades))
            
            # Can we trade?
            can_trade, reason = self.risk_manager.can_trade(timestamp)
            
            if reason:
                if "Daily loss" in reason:
                    self._daily_halts += 1
                elif "Weekly loss" in reason:
                    self._weekly_halts += 1
                elif "Consecutive" in reason:
                    self._consec_halts += 1
            
            if not can_trade:
                self.equity_curve.append(self.risk_manager.state.current_equity)
                continue
            
            # Check gatekeeper
            if self.gatekeeper:
                gate_decision = self.gatekeeper.evaluate()
                if not gate_decision.allowed:
                    self.equity_curve.append(self.risk_manager.state.current_equity)
                    continue
            
            # Get signals from portfolio
            signals = self.portfolio.update(
                timestamp=timestamp,
                open_=eu_bar['open'],
                high=eu_bar['high'],
                low=eu_bar['low'],
                close=eu_bar['close'],
            )
            
            # Process signals
            for port_signal in signals:
                self._process_signal(port_signal.signal, timestamp)
            
            self.equity_curve.append(self.risk_manager.state.current_equity)
        
        # Force close remaining
        if self.open_trades:
            last = df_eurusd.iloc[-1]
            for trade_id in list(self.open_trades.keys()):
                self._close_trade(trade_id, last['close'], last['timestamp'], "END")
        
        return self._compute_metrics(df_eurusd)
    
    def _process_signal(self, signal: MicroSignal, timestamp: datetime) -> None:
        """Process a micro signal."""
        direction = "LONG" if signal.direction == MicroEdgeDirection.LONG else "SHORT"
        
        # Get position size from risk manager
        lot_size, risk_frac = self.risk_manager.calculate_position_size(
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            direction=direction,
            bias_confidence=signal.bias_confidence,
        )
        
        if lot_size <= 0:
            return
        
        # Create trade
        self._trade_counter += 1
        trade_id = f"T{self._trade_counter}"
        
        trade = PortfolioTrade(
            trade_id=trade_id,
            strategy=signal.strategy_name,
            timestamp=timestamp,
            direction=direction,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            position_size=lot_size,
            risk_fraction=risk_frac,
            bias=signal.bias.value,
            bias_confidence=signal.bias_confidence,
        )
        
        self.open_trades[trade_id] = trade
        self.trades.append(trade)
        
        # Register with risk manager
        self.risk_manager.register_trade_opened(
            trade_id=trade_id,
            direction=direction,
            risk_frac=risk_frac,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            lot_size=lot_size,
        )
        
        # Register with portfolio
        self.portfolio.register_trade_opened(signal.direction, risk_frac)
    
    def _check_exits(
        self,
        high: float,
        low: float,
        close: float,
        timestamp: datetime,
    ) -> None:
        """Check for trade exits."""
        for trade_id in list(self.open_trades.keys()):
            trade = self.open_trades[trade_id]
            
            if trade.direction == "LONG":
                # Check SL
                if low <= trade.stop_loss:
                    self._close_trade(trade_id, trade.stop_loss, timestamp, "SL")
                # Check TP
                elif high >= trade.take_profit:
                    self._close_trade(trade_id, trade.take_profit, timestamp, "TP")
            else:
                if high >= trade.stop_loss:
                    self._close_trade(trade_id, trade.stop_loss, timestamp, "SL")
                elif low <= trade.take_profit:
                    self._close_trade(trade_id, trade.take_profit, timestamp, "TP")
    
    def _close_trade(
        self,
        trade_id: str,
        exit_price: float,
        timestamp: datetime,
        reason: str,
    ) -> None:
        """Close a trade."""
        if trade_id not in self.open_trades:
            return
        
        trade = self.open_trades.pop(trade_id)
        trade.exit_price = exit_price
        trade.exit_time = timestamp
        trade.exit_reason = reason
        
        # Calculate PnL
        if trade.direction == "LONG":
            price_diff = exit_price - trade.entry_price
        else:
            price_diff = trade.entry_price - exit_price
        
        trade.pnl = price_diff * trade.position_size * 100000
        
        # R-multiple
        sl_dist = abs(trade.entry_price - trade.stop_loss)
        trade.pnl_r = price_diff / sl_dist if sl_dist > 0 else 0
        trade.won = trade.pnl > 0
        
        # Update risk manager
        self.risk_manager.register_trade_closed(trade_id, trade.pnl, trade.won)
        
        # Update portfolio
        direction = MicroEdgeDirection.LONG if trade.direction == "LONG" else MicroEdgeDirection.SHORT
        self.portfolio.register_trade_closed(direction)
    
    def _compute_metrics(self, df: pd.DataFrame) -> PortfolioMetrics:
        """Compute comprehensive portfolio metrics."""
        if not self.trades:
            return PortfolioMetrics()
        
        # Basic stats
        total = len(self.trades)
        wins = [t for t in self.trades if t.won]
        losses = [t for t in self.trades if not t.won]
        
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = win_count / total if total > 0 else 0
        
        # PnL
        total_pnl = sum(t.pnl for t in self.trades)
        total_pnl_r = sum(t.pnl_r for t in self.trades)
        
        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0.0001
        
        pf = gross_profit / gross_loss if gross_loss > 0 else 0
        expectancy_r = total_pnl_r / total if total > 0 else 0
        
        # Drawdown
        eq = np.array(self.equity_curve)
        if len(eq) > 0:
            running_max = np.maximum.accumulate(eq)
            dd_abs = running_max - eq
            dd_pct = np.where(running_max > 0, dd_abs / running_max, 0)
            max_dd_pct = np.max(dd_pct)
            max_dd_abs = np.max(dd_abs)
        else:
            max_dd_pct = 0
            max_dd_abs = 0
        
        # Frequency
        start = df['timestamp'].iloc[0]
        end = df['timestamp'].iloc[-1]
        days = (end - start).days
        months = days / 30.0
        
        trades_per_month = total / months if months > 0 else 0
        avg_trades_per_day = total / days if days > 0 else 0
        
        # Strategy breakdown
        strategy_metrics = {}
        by_strategy = defaultdict(list)
        for t in self.trades:
            by_strategy[t.strategy].append(t)
        
        for name, strat_trades in by_strategy.items():
            strat_wins = [t for t in strat_trades if t.won]
            strat_losses = [t for t in strat_trades if not t.won]
            
            strat_total = len(strat_trades)
            strat_win_count = len(strat_wins)
            
            strat_pnl = sum(t.pnl for t in strat_trades)
            strat_pnl_r = sum(t.pnl_r for t in strat_trades)
            
            strat_gross_profit = sum(t.pnl for t in strat_wins) if strat_wins else 0
            strat_gross_loss = abs(sum(t.pnl for t in strat_losses)) if strat_losses else 0.0001
            
            avg_win_r = np.mean([t.pnl_r for t in strat_wins]) if strat_wins else 0
            avg_loss_r = abs(np.mean([t.pnl_r for t in strat_losses])) if strat_losses else 0
            
            strategy_metrics[name] = StrategyMetrics(
                name=name,
                trades=strat_total,
                wins=strat_win_count,
                losses=len(strat_losses),
                win_rate=strat_win_count / strat_total if strat_total > 0 else 0,
                total_pnl=strat_pnl,
                total_pnl_r=strat_pnl_r,
                expectancy_r=strat_pnl_r / strat_total if strat_total > 0 else 0,
                profit_factor=strat_gross_profit / strat_gross_loss if strat_gross_loss > 0 else 0,
                avg_win_r=avg_win_r,
                avg_loss_r=avg_loss_r,
            )
        
        # Long/short ratio
        longs = [t for t in self.trades if t.direction == "LONG"]
        shorts = [t for t in self.trades if t.direction == "SHORT"]
        long_short_ratio = len(longs) / len(shorts) if shorts else float('inf')
        
        return PortfolioMetrics(
            total_trades=total,
            wins=win_count,
            losses=loss_count,
            win_rate=win_rate,
            total_pnl=total_pnl,
            total_pnl_r=total_pnl_r,
            expectancy_r=expectancy_r,
            profit_factor=pf,
            max_drawdown_pct=max_dd_pct,
            max_drawdown_abs=max_dd_abs,
            trades_per_month=trades_per_month,
            avg_trades_per_day=avg_trades_per_day,
            strategy_metrics=strategy_metrics,
            long_short_ratio=long_short_ratio,
            max_concurrent=self._max_concurrent,
            daily_halts=self._daily_halts,
            weekly_halts=self._weekly_halts,
            consecutive_loss_halts=self._consec_halts,
        )


def print_portfolio_results(metrics: PortfolioMetrics, initial_capital: float):
    """Print comprehensive portfolio results."""
    print()
    print("=" * 80)
    print("PORTFOLIO BACKTEST RESULTS")
    print("=" * 80)
    print()
    
    # Overall
    print("OVERALL PERFORMANCE")
    print("-" * 40)
    print(f"  Total Trades:       {metrics.total_trades}")
    print(f"  Wins:               {metrics.wins}")
    print(f"  Losses:             {metrics.losses}")
    print(f"  Win Rate:           {metrics.win_rate:.1%}")
    print()
    print(f"  Total PnL ($):      ${metrics.total_pnl:.2f}")
    print(f"  Total PnL (R):      {metrics.total_pnl_r:.2f}R")
    print(f"  EXPECTANCY (R):     {metrics.expectancy_r:.3f}R")
    print(f"  Profit Factor:      {metrics.profit_factor:.2f}")
    print()
    print(f"  Max Drawdown:       {metrics.max_drawdown_pct:.1%}")
    print(f"  Max DD ($):         ${metrics.max_drawdown_abs:.2f}")
    print(f"  Final Equity:       ${initial_capital + metrics.total_pnl:.2f}")
    print(f"  Return:             {100 * metrics.total_pnl / initial_capital:.1f}%")
    print()
    
    # Frequency
    print("TRADE FREQUENCY")
    print("-" * 40)
    print(f"  Trades/Month:       {metrics.trades_per_month:.1f}")
    print(f"  Trades/Day (avg):   {metrics.avg_trades_per_day:.2f}")
    print(f"  Long/Short Ratio:   {metrics.long_short_ratio:.2f}")
    print(f"  Max Concurrent:     {metrics.max_concurrent}")
    print()
    
    # Strategy breakdown
    print("STRATEGY BREAKDOWN")
    print("-" * 40)
    print(f"{'Strategy':<25} {'Trades':>8} {'WR':>8} {'Exp(R)':>10} {'PF':>8}")
    print("-" * 60)
    
    for name, sm in sorted(metrics.strategy_metrics.items()):
        print(f"{name:<25} {sm.trades:>8} {sm.win_rate:>7.1%} {sm.expectancy_r:>9.3f}R {sm.profit_factor:>7.2f}")
    
    print()
    
    # Kill switch stats
    print("RISK MANAGEMENT")
    print("-" * 40)
    print(f"  Daily Halts:        {metrics.daily_halts}")
    print(f"  Weekly Halts:       {metrics.weekly_halts}")
    print(f"  Consec Loss Halts:  {metrics.consecutive_loss_halts}")
    print()


def determine_portfolio_viability(metrics: PortfolioMetrics) -> Tuple[str, str, List[str]]:
    """
    Determine if portfolio is viable.
    
    Returns: (verdict, explanation, tradeable_strategies)
    """
    print("=" * 80)
    print("VIABILITY ASSESSMENT")
    print("=" * 80)
    print()
    
    issues = []
    tradeable = []
    
    # Portfolio-level checks
    if metrics.expectancy_r <= 0:
        issues.append(f"KILL: Portfolio expectancy {metrics.expectancy_r:.3f}R ≤ 0")
    elif metrics.expectancy_r < 0.05:
        issues.append(f"WARNING: Portfolio expectancy low ({metrics.expectancy_r:.3f}R)")
    
    if metrics.profit_factor < 1.0:
        issues.append(f"KILL: Profit factor {metrics.profit_factor:.2f} < 1.0")
    elif metrics.profit_factor < 1.2:
        issues.append(f"WARNING: Profit factor marginal ({metrics.profit_factor:.2f})")
    
    if metrics.max_drawdown_pct > 0.15:
        issues.append(f"KILL: Max drawdown {metrics.max_drawdown_pct:.1%} > 15%")
    elif metrics.max_drawdown_pct > 0.10:
        issues.append(f"WARNING: Max drawdown elevated ({metrics.max_drawdown_pct:.1%})")
    
    if metrics.trades_per_month < 5:
        issues.append(f"WARNING: Low frequency ({metrics.trades_per_month:.1f}/month)")
    
    # Strategy-level checks
    print("Strategy-Level Assessment:")
    print()
    
    for name, sm in metrics.strategy_metrics.items():
        strat_issues = []
        
        if sm.expectancy_r <= 0:
            strat_issues.append(f"Expectancy ≤ 0")
        if sm.profit_factor < 1.0:
            strat_issues.append(f"PF < 1.0")
        if sm.trades < 10:
            strat_issues.append(f"Low sample ({sm.trades})")
        
        if strat_issues:
            print(f"  {name}: ❌ NOT VIABLE - {', '.join(strat_issues)}")
        else:
            print(f"  {name}: ✓ VIABLE (Exp: {sm.expectancy_r:.3f}R, PF: {sm.profit_factor:.2f})")
            tradeable.append(name)
    
    print()
    
    # Print issues
    if issues:
        print("Portfolio Issues:")
        for issue in issues:
            print(f"  • {issue}")
        print()
    
    # Verdict
    kill_count = sum(1 for i in issues if i.startswith("KILL"))
    warning_count = sum(1 for i in issues if i.startswith("WARNING"))
    
    if kill_count > 0:
        verdict = "NOT VIABLE"
        explanation = "Portfolio fails kill criteria. Do not deploy."
    elif len(tradeable) == 0:
        verdict = "NOT VIABLE"
        explanation = "No individual strategies are viable."
    elif warning_count >= 2:
        verdict = "MARGINAL"
        explanation = "Multiple concerns. Extended paper testing only."
    elif warning_count == 1:
        verdict = "CAUTIOUS"
        explanation = "Minor concerns. Proceed with caution."
    else:
        verdict = "VIABLE"
        explanation = "Portfolio meets criteria. Proceed to paper trading."
    
    print(f"VERDICT: {verdict}")
    print(f"  {explanation}")
    print()
    
    if tradeable:
        print(f"TRADEABLE STRATEGIES: {', '.join(tradeable)}")
    else:
        print("TRADEABLE STRATEGIES: None")
    
    print()
    print("=" * 80)
    
    return verdict, explanation, tradeable
