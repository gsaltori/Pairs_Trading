"""
Multi-Strategy Portfolio Backtest Harness

Runs individual and combined portfolio backtests with:
- Individual strategy backtests
- Combined portfolio backtest
- Baseline vs Gated vs Portfolio comparison
- Counterfactual analysis for blocked trades
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Union
from enum import Enum

sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_system.config import StrategyConfig, GatekeeperConfig
from trading_system.signal_engine import SignalEngine, TradeSignal, SignalDirection
from trading_system.pullback_engine import PullbackEngine, PullbackSignal, PullbackDirection
from trading_system.volatility_expansion_engine import VolatilityExpansionEngine, VolExpSignal, VolExpDirection
from trading_system.gatekeeper_engine import GatekeeperEngine
from trading_system.strategy_router import StrategyRouter, StrategyType, AnySignal


@dataclass
class PortfolioTrade:
    """Trade record for portfolio backtest."""
    strategy: StrategyType
    entry_bar: int
    entry_time: datetime
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    risk_amount: float
    
    exit_bar: Optional[int] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    won: bool = False
    
    # Gatekeeper
    was_blocked: bool = False
    block_reasons: List[str] = field(default_factory=list)


@dataclass
class StrategyResult:
    """Results for a single strategy."""
    name: str
    trades: List[PortfolioTrade]
    equity_curve: List[float]
    
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    total_pnl: float = 0.0
    max_drawdown_pct: float = 0.0
    
    blocked_count: int = 0


@dataclass
class PortfolioResult:
    """Combined portfolio results."""
    strategy_results: Dict[str, StrategyResult]
    combined_equity: List[float]
    
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    total_pnl: float = 0.0
    max_drawdown_pct: float = 0.0
    
    blocked_count: int = 0
    trades_by_strategy: Dict[str, int] = field(default_factory=dict)
    blocks_by_strategy: Dict[str, int] = field(default_factory=dict)


class PortfolioBacktest:
    """
    Multi-strategy portfolio backtest engine.
    """
    
    RISK_ALLOCATIONS = {
        StrategyType.TREND_CONTINUATION: 0.0030,
        StrategyType.TREND_PULLBACK: 0.0025,
        StrategyType.VOLATILITY_EXPANSION: 0.0020,
    }
    
    def __init__(
        self,
        use_gatekeeper: bool = True,
        initial_capital: float = 100.0,  # $100 micro account
    ):
        self.use_gatekeeper = use_gatekeeper
        self.initial_capital = initial_capital
        
        # Strategy engines
        self.trend_engine = SignalEngine(StrategyConfig())
        self.pullback_engine = PullbackEngine()
        self.volexp_engine = VolatilityExpansionEngine()
        
        # Gatekeeper
        self.gatekeeper = GatekeeperEngine(GatekeeperConfig()) if use_gatekeeper else None
        
        # State
        self.equity = initial_capital
        self.all_trades: List[PortfolioTrade] = []
        self.blocked_trades: List[PortfolioTrade] = []
        self.current_position: Optional[PortfolioTrade] = None
        self.equity_curve: List[float] = []
        
        self._bar_index = 0
    
    def run(
        self,
        df_eurusd: pd.DataFrame,
        df_gbpusd: pd.DataFrame,
    ) -> PortfolioResult:
        """Run portfolio backtest."""
        assert len(df_eurusd) == len(df_gbpusd), "Data mismatch"
        
        n = len(df_eurusd)
        
        for i in range(n):
            self._bar_index = i
            
            eu_bar = df_eurusd.iloc[i]
            gb_bar = df_gbpusd.iloc[i]
            timestamp = eu_bar['timestamp']
            
            # Update gatekeeper
            if self.gatekeeper:
                self.gatekeeper.update(eu_bar['close'], gb_bar['close'])
            
            # Check exit
            if self.current_position is not None:
                self._check_exit(eu_bar['high'], eu_bar['low'], timestamp)
            
            # Collect signals
            signals = self._collect_signals(eu_bar, timestamp)
            
            # Process signals if no position
            if signals and self.current_position is None:
                self._process_signals(signals, timestamp)
            
            self.equity_curve.append(self.equity)
        
        # Force close
        if self.current_position is not None:
            last = df_eurusd.iloc[-1]
            self._force_close(last['close'], last['timestamp'])
        
        return self._compute_results()
    
    def _collect_signals(self, bar: pd.Series, timestamp: datetime) -> Dict[StrategyType, AnySignal]:
        """Collect signals from all strategies."""
        signals = {}
        
        # Trend Continuation
        trend_sig = self.trend_engine.update(
            timestamp, bar['open'], bar['high'], bar['low'], bar['close']
        )
        if trend_sig:
            signals[StrategyType.TREND_CONTINUATION] = trend_sig
        
        # Pullback
        pullback_sig = self.pullback_engine.update(
            timestamp, bar['open'], bar['high'], bar['low'], bar['close']
        )
        if pullback_sig:
            signals[StrategyType.TREND_PULLBACK] = pullback_sig
        
        # Vol Expansion
        volexp_sig = self.volexp_engine.update(
            timestamp, bar['open'], bar['high'], bar['low'], bar['close']
        )
        if volexp_sig:
            signals[StrategyType.VOLATILITY_EXPANSION] = volexp_sig
        
        return signals
    
    def _process_signals(
        self,
        signals: Dict[StrategyType, AnySignal],
        timestamp: datetime,
    ) -> None:
        """Process signals by priority."""
        priority = [
            StrategyType.TREND_CONTINUATION,
            StrategyType.TREND_PULLBACK,
            StrategyType.VOLATILITY_EXPANSION,
        ]
        
        for strategy in priority:
            if strategy not in signals:
                continue
            
            signal = signals[strategy]
            direction = self._get_direction(signal)
            
            # Check gatekeeper
            if self.gatekeeper:
                decision = self.gatekeeper.evaluate()
                if not decision.allowed:
                    trade = PortfolioTrade(
                        strategy=strategy,
                        entry_bar=self._bar_index,
                        entry_time=timestamp,
                        direction=direction,
                        entry_price=signal.entry_price,
                        stop_loss=signal.stop_loss,
                        take_profit=signal.take_profit,
                        position_size=0,
                        risk_amount=0,
                        was_blocked=True,
                        block_reasons=[r.value for r in decision.reasons],
                    )
                    self.blocked_trades.append(trade)
                    continue  # Try next priority
            
            # Execute trade
            risk_pct = self.RISK_ALLOCATIONS[strategy]
            risk_amount = self.equity * risk_pct
            sl_dist = abs(signal.entry_price - signal.stop_loss)
            
            if sl_dist > 0:
                position_size = risk_amount / (sl_dist * 100000)  # Simplified
                position_size = max(0.01, min(1.0, position_size))
            else:
                position_size = 0.01
            
            trade = PortfolioTrade(
                strategy=strategy,
                entry_bar=self._bar_index,
                entry_time=timestamp,
                direction=direction,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                position_size=position_size,
                risk_amount=risk_amount,
            )
            
            self.current_position = trade
            self.all_trades.append(trade)
            return  # Only one trade per bar
    
    def _check_exit(self, high: float, low: float, timestamp: datetime) -> None:
        """Check SL/TP."""
        trade = self.current_position
        
        if trade.direction == "LONG":
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
        trade = self.current_position
        trade.exit_bar = self._bar_index
        trade.exit_price = exit_price
        trade.won = won
        
        if trade.direction == "LONG":
            trade.pnl = (exit_price - trade.entry_price) * trade.position_size * 100000
        else:
            trade.pnl = (trade.entry_price - exit_price) * trade.position_size * 100000
        
        self.equity += trade.pnl
        self.current_position = None
    
    def _force_close(self, price: float, timestamp: datetime) -> None:
        """Force close at end."""
        trade = self.current_position
        trade.exit_bar = self._bar_index
        trade.exit_price = price
        
        if trade.direction == "LONG":
            trade.pnl = (price - trade.entry_price) * trade.position_size * 100000
        else:
            trade.pnl = (trade.entry_price - price) * trade.position_size * 100000
        
        trade.won = trade.pnl > 0
        self.equity += trade.pnl
        self.current_position = None
    
    def _get_direction(self, signal: AnySignal) -> str:
        """Get direction from signal."""
        if hasattr(signal, 'direction'):
            return signal.direction.value
        return "UNKNOWN"
    
    def _compute_results(self) -> PortfolioResult:
        """Compute portfolio results."""
        # Group by strategy
        by_strategy: Dict[StrategyType, List[PortfolioTrade]] = {s: [] for s in StrategyType}
        for trade in self.all_trades:
            by_strategy[trade.strategy].append(trade)
        
        # Strategy results
        strategy_results = {}
        for strategy, trades in by_strategy.items():
            strategy_results[strategy.value] = self._compute_strategy_result(
                strategy.value, trades
            )
        
        # Portfolio totals
        all_trades = self.all_trades
        if not all_trades:
            return PortfolioResult(
                strategy_results=strategy_results,
                combined_equity=self.equity_curve,
            )
        
        wins = [t for t in all_trades if t.won]
        losses = [t for t in all_trades if not t.won]
        
        total = len(all_trades)
        win_count = len(wins)
        win_rate = win_count / total if total > 0 else 0
        
        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0.0001
        pf = gross_profit / gross_loss if gross_loss > 0 else 0
        
        total_pnl = sum(t.pnl for t in all_trades)
        avg_win = gross_profit / win_count if win_count > 0 else 0
        avg_loss = gross_loss / len(losses) if losses else 0
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        # Drawdown
        eq = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(eq)
        dd = (running_max - eq) / running_max
        max_dd = np.max(dd) if len(dd) > 0 else 0
        
        # Blocked
        blocked_by_strategy = {s.value: 0 for s in StrategyType}
        for t in self.blocked_trades:
            blocked_by_strategy[t.strategy.value] += 1
        
        trades_by_strategy = {s.value: len(trades) for s, trades in by_strategy.items()}
        
        return PortfolioResult(
            strategy_results=strategy_results,
            combined_equity=self.equity_curve,
            total_trades=total,
            wins=win_count,
            losses=len(losses),
            win_rate=win_rate,
            profit_factor=pf,
            expectancy=expectancy,
            total_pnl=total_pnl,
            max_drawdown_pct=max_dd,
            blocked_count=len(self.blocked_trades),
            trades_by_strategy=trades_by_strategy,
            blocks_by_strategy=blocked_by_strategy,
        )
    
    def _compute_strategy_result(
        self,
        name: str,
        trades: List[PortfolioTrade],
    ) -> StrategyResult:
        """Compute results for single strategy."""
        if not trades:
            return StrategyResult(name=name, trades=trades, equity_curve=[])
        
        wins = [t for t in trades if t.won]
        losses = [t for t in trades if not t.won]
        
        total = len(trades)
        win_count = len(wins)
        win_rate = win_count / total if total > 0 else 0
        
        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0.0001
        pf = gross_profit / gross_loss if gross_loss > 0 else 0
        
        total_pnl = sum(t.pnl for t in trades)
        avg_win = gross_profit / win_count if win_count > 0 else 0
        avg_loss = gross_loss / len(losses) if losses else 0
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        return StrategyResult(
            name=name,
            trades=trades,
            equity_curve=[],
            total_trades=total,
            wins=win_count,
            losses=len(losses),
            win_rate=win_rate,
            profit_factor=pf,
            expectancy=expectancy,
            total_pnl=total_pnl,
        )


def run_individual_backtests(
    df_eurusd: pd.DataFrame,
    df_gbpusd: pd.DataFrame,
    initial_capital: float = 100.0,
) -> Dict[str, StrategyResult]:
    """Run individual backtests for each strategy."""
    results = {}
    
    # Trend Continuation only
    print("  Running TREND CONTINUATION...")
    trend_bt = SingleStrategyBacktest(
        strategy_type=StrategyType.TREND_CONTINUATION,
        use_gatekeeper=True,
        initial_capital=initial_capital,
    )
    results["TREND_CONTINUATION"] = trend_bt.run(df_eurusd.copy(), df_gbpusd.copy())
    
    # Pullback only
    print("  Running TREND PULLBACK...")
    pullback_bt = SingleStrategyBacktest(
        strategy_type=StrategyType.TREND_PULLBACK,
        use_gatekeeper=True,
        initial_capital=initial_capital,
    )
    results["TREND_PULLBACK"] = pullback_bt.run(df_eurusd.copy(), df_gbpusd.copy())
    
    # VolExp only
    print("  Running VOLATILITY EXPANSION...")
    volexp_bt = SingleStrategyBacktest(
        strategy_type=StrategyType.VOLATILITY_EXPANSION,
        use_gatekeeper=True,
        initial_capital=initial_capital,
    )
    results["VOLATILITY_EXPANSION"] = volexp_bt.run(df_eurusd.copy(), df_gbpusd.copy())
    
    return results


class SingleStrategyBacktest:
    """Backtest for a single strategy."""
    
    RISK_ALLOCATIONS = {
        StrategyType.TREND_CONTINUATION: 0.0030,
        StrategyType.TREND_PULLBACK: 0.0025,
        StrategyType.VOLATILITY_EXPANSION: 0.0020,
    }
    
    def __init__(
        self,
        strategy_type: StrategyType,
        use_gatekeeper: bool = True,
        initial_capital: float = 100.0,
    ):
        self.strategy_type = strategy_type
        self.use_gatekeeper = use_gatekeeper
        self.initial_capital = initial_capital
        
        # Engine
        if strategy_type == StrategyType.TREND_CONTINUATION:
            self.engine = SignalEngine(StrategyConfig())
        elif strategy_type == StrategyType.TREND_PULLBACK:
            self.engine = PullbackEngine()
        else:
            self.engine = VolatilityExpansionEngine()
        
        self.gatekeeper = GatekeeperEngine(GatekeeperConfig()) if use_gatekeeper else None
        
        self.equity = initial_capital
        self.trades: List[PortfolioTrade] = []
        self.blocked_trades: List[PortfolioTrade] = []
        self.current_position: Optional[PortfolioTrade] = None
        self.equity_curve: List[float] = []
        self._bar_index = 0
    
    def run(
        self,
        df_eurusd: pd.DataFrame,
        df_gbpusd: pd.DataFrame,
    ) -> StrategyResult:
        """Run single strategy backtest."""
        n = len(df_eurusd)
        
        for i in range(n):
            self._bar_index = i
            
            eu_bar = df_eurusd.iloc[i]
            gb_bar = df_gbpusd.iloc[i]
            timestamp = eu_bar['timestamp']
            
            if self.gatekeeper:
                self.gatekeeper.update(eu_bar['close'], gb_bar['close'])
            
            if self.current_position is not None:
                self._check_exit(eu_bar['high'], eu_bar['low'], timestamp)
            
            signal = self.engine.update(
                timestamp, eu_bar['open'], eu_bar['high'], eu_bar['low'], eu_bar['close']
            )
            
            if signal and self.current_position is None:
                self._process_signal(signal, timestamp)
            
            self.equity_curve.append(self.equity)
        
        if self.current_position is not None:
            last = df_eurusd.iloc[-1]
            self._force_close(last['close'], last['timestamp'])
        
        return self._compute_result()
    
    def _process_signal(self, signal: AnySignal, timestamp: datetime) -> None:
        """Process signal."""
        direction = signal.direction.value
        
        if self.gatekeeper:
            decision = self.gatekeeper.evaluate()
            if not decision.allowed:
                trade = PortfolioTrade(
                    strategy=self.strategy_type,
                    entry_bar=self._bar_index,
                    entry_time=timestamp,
                    direction=direction,
                    entry_price=signal.entry_price,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    position_size=0,
                    risk_amount=0,
                    was_blocked=True,
                    block_reasons=[r.value for r in decision.reasons],
                )
                self.blocked_trades.append(trade)
                return
        
        risk_pct = self.RISK_ALLOCATIONS[self.strategy_type]
        risk_amount = self.equity * risk_pct
        sl_dist = abs(signal.entry_price - signal.stop_loss)
        
        if sl_dist > 0:
            position_size = risk_amount / (sl_dist * 100000)
            position_size = max(0.01, min(1.0, position_size))
        else:
            position_size = 0.01
        
        trade = PortfolioTrade(
            strategy=self.strategy_type,
            entry_bar=self._bar_index,
            entry_time=timestamp,
            direction=direction,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            position_size=position_size,
            risk_amount=risk_amount,
        )
        
        self.current_position = trade
        self.trades.append(trade)
    
    def _check_exit(self, high: float, low: float, timestamp: datetime) -> None:
        """Check SL/TP."""
        trade = self.current_position
        
        if trade.direction == "LONG":
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
        trade = self.current_position
        trade.exit_bar = self._bar_index
        trade.exit_price = exit_price
        trade.won = won
        
        if trade.direction == "LONG":
            trade.pnl = (exit_price - trade.entry_price) * trade.position_size * 100000
        else:
            trade.pnl = (trade.entry_price - exit_price) * trade.position_size * 100000
        
        self.equity += trade.pnl
        self.current_position = None
    
    def _force_close(self, price: float, timestamp: datetime) -> None:
        """Force close."""
        trade = self.current_position
        trade.exit_bar = self._bar_index
        trade.exit_price = price
        
        if trade.direction == "LONG":
            trade.pnl = (price - trade.entry_price) * trade.position_size * 100000
        else:
            trade.pnl = (trade.entry_price - price) * trade.position_size * 100000
        
        trade.won = trade.pnl > 0
        self.equity += trade.pnl
        self.current_position = None
    
    def _compute_result(self) -> StrategyResult:
        """Compute results."""
        if not self.trades:
            return StrategyResult(
                name=self.strategy_type.value,
                trades=self.trades,
                equity_curve=self.equity_curve,
                blocked_count=len(self.blocked_trades),
            )
        
        wins = [t for t in self.trades if t.won]
        losses = [t for t in self.trades if not t.won]
        
        total = len(self.trades)
        win_count = len(wins)
        win_rate = win_count / total if total > 0 else 0
        
        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0.0001
        pf = gross_profit / gross_loss if gross_loss > 0 else 0
        
        total_pnl = sum(t.pnl for t in self.trades)
        avg_win = gross_profit / win_count if win_count > 0 else 0
        avg_loss = gross_loss / len(losses) if losses else 0
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        eq = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(eq)
        dd = (running_max - eq) / running_max
        max_dd = np.max(dd) if len(dd) > 0 else 0
        
        return StrategyResult(
            name=self.strategy_type.value,
            trades=self.trades,
            equity_curve=self.equity_curve,
            total_trades=total,
            wins=win_count,
            losses=len(losses),
            win_rate=win_rate,
            profit_factor=pf,
            expectancy=expectancy,
            total_pnl=total_pnl,
            max_drawdown_pct=max_dd,
            blocked_count=len(self.blocked_trades),
        )


def counterfactual_analysis(
    blocked_trades: List[PortfolioTrade],
    df_eurusd: pd.DataFrame,
) -> Dict:
    """Analyze what blocked trades would have done."""
    if not blocked_trades:
        return {
            'total': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0,
            'pnl': 0,
        }
    
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
        
        # Use fixed risk for comparison
        risk = 1.0  # $1 per blocked trade
        
        for i in range(entry_bar + 1, min(entry_bar + 100, n)):
            if trade.direction == "LONG":
                if low[i] <= sl:
                    losses += 1
                    pnl -= risk
                    break
                elif high[i] >= tp:
                    wins += 1
                    # Use actual RR from signal
                    sl_dist = trade.entry_price - sl
                    tp_dist = tp - trade.entry_price
                    rr = tp_dist / sl_dist if sl_dist > 0 else 2.0
                    pnl += risk * rr
                    break
            else:
                if high[i] >= sl:
                    losses += 1
                    pnl -= risk
                    break
                elif low[i] <= tp:
                    wins += 1
                    sl_dist = sl - trade.entry_price
                    tp_dist = trade.entry_price - tp
                    rr = tp_dist / sl_dist if sl_dist > 0 else 2.0
                    pnl += risk * rr
                    break
    
    total = wins + losses
    
    return {
        'total': len(blocked_trades),
        'resolved': total,
        'wins': wins,
        'losses': losses,
        'win_rate': wins / total if total > 0 else 0,
        'pnl': pnl,
    }
