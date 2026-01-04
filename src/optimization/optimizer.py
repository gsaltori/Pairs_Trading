"""
Walk-Forward Optimization Module.

Implements walk-forward optimization to find robust parameters.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Tuple
from itertools import product
import json
import logging
from copy import deepcopy

from config.settings import Settings
from src.backtest.backtest_engine import BacktestEngine, BacktestResult


logger = logging.getLogger(__name__)


@dataclass
class OptimizationParams:
    """Optimizable parameters."""
    entry_zscore: float = 2.0
    exit_zscore: float = 0.2
    stop_loss_zscore: float = 3.0
    regression_window: int = 120
    zscore_window: int = 60
    min_correlation: float = 0.70


@dataclass
class WalkForwardPeriod:
    """Results for a single walk-forward period."""
    period_index: int
    is_start: datetime
    is_end: datetime
    oos_start: datetime
    oos_end: datetime
    best_params: Optional[OptimizationParams]
    is_sharpe: float  # In-sample Sharpe
    oos_sharpe: float  # Out-of-sample Sharpe
    is_return: float
    oos_return: float
    oos_trades: int


@dataclass
class WalkForwardResult:
    """Complete walk-forward optimization result."""
    pair: Tuple[str, str]
    periods: List[WalkForwardPeriod]
    best_params: Optional[OptimizationParams]
    efficiency_ratio: float  # OOS/IS performance ratio
    is_sharpe: float  # Average IS Sharpe
    oos_sharpe: float  # Average OOS Sharpe
    total_trades: int
    robustness_score: float


class WalkForwardOptimizer:
    """
    Walk-forward optimization for pairs trading.
    
    Process:
    1. Divide data into IS (in-sample) and OOS (out-of-sample) periods
    2. Optimize parameters on IS period
    3. Validate on OOS period
    4. Roll forward and repeat
    5. Calculate efficiency ratio
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize optimizer.
        
        Args:
            settings: Trading settings
        """
        self.settings = settings
        self.opt_settings = settings.optimization
        
        # Parameter search space
        self.param_grid = {
            'entry_zscore': [1.5, 2.0, 2.5],
            'exit_zscore': [0.1, 0.2, 0.3],
            'stop_loss_zscore': [2.5, 3.0, 3.5],
            'regression_window': [60, 120, 180],
            'zscore_window': [30, 60, 90],
            'min_correlation': [0.65, 0.70, 0.75]
        }
    
    def optimize(
        self,
        pair: Tuple[str, str],
        price_a: pd.Series,
        price_b: pd.Series
    ) -> WalkForwardResult:
        """
        Run walk-forward optimization.
        
        Args:
            pair: (symbol_a, symbol_b) tuple
            price_a: Price series for symbol A
            price_b: Price series for symbol B
            
        Returns:
            WalkForwardResult
        """
        # Align data
        common_idx = price_a.index.intersection(price_b.index)
        price_a = price_a.loc[common_idx]
        price_b = price_b.loc[common_idx]
        
        total_bars = len(common_idx)
        is_bars = self.opt_settings.in_sample_bars
        oos_bars = self.opt_settings.out_of_sample_bars
        step = oos_bars
        
        # Calculate number of periods
        periods = []
        period_idx = 0
        
        start = 0
        while start + is_bars + oos_bars <= total_bars:
            is_start_idx = start
            is_end_idx = start + is_bars
            oos_start_idx = is_end_idx
            oos_end_idx = is_end_idx + oos_bars
            
            # Extract data slices
            is_price_a = price_a.iloc[is_start_idx:is_end_idx]
            is_price_b = price_b.iloc[is_start_idx:is_end_idx]
            
            oos_price_a = price_a.iloc[oos_start_idx:oos_end_idx]
            oos_price_b = price_b.iloc[oos_start_idx:oos_end_idx]
            
            # Optimize on IS
            logger.info(f"Optimizing period {period_idx + 1}...")
            best_params, is_result = self._optimize_period(
                pair, is_price_a, is_price_b
            )
            
            is_sharpe = is_result.sharpe_ratio if is_result else 0
            is_return = is_result.total_return if is_result else 0
            
            # Validate on OOS
            oos_result = self._validate_period(
                pair, oos_price_a, oos_price_b, best_params
            )
            
            oos_sharpe = oos_result.sharpe_ratio if oos_result else 0
            oos_return = oos_result.total_return if oos_result else 0
            oos_trades = oos_result.total_trades if oos_result else 0
            
            periods.append(WalkForwardPeriod(
                period_index=period_idx,
                is_start=common_idx[is_start_idx],
                is_end=common_idx[is_end_idx - 1],
                oos_start=common_idx[oos_start_idx],
                oos_end=common_idx[oos_end_idx - 1],
                best_params=best_params,
                is_sharpe=is_sharpe,
                oos_sharpe=oos_sharpe,
                is_return=is_return,
                oos_return=oos_return,
                oos_trades=oos_trades
            ))
            
            period_idx += 1
            start += step
            
            if period_idx >= self.opt_settings.max_iterations:
                break
        
        # Calculate overall metrics
        avg_is_sharpe = np.mean([p.is_sharpe for p in periods]) if periods else 0
        avg_oos_sharpe = np.mean([p.oos_sharpe for p in periods]) if periods else 0
        efficiency_ratio = avg_oos_sharpe / avg_is_sharpe if avg_is_sharpe > 0 else 0
        total_trades = sum(p.oos_trades for p in periods)
        
        # Select overall best params (from period with highest OOS Sharpe)
        if periods:
            best_period = max(periods, key=lambda p: p.oos_sharpe)
            best_params = best_period.best_params
        else:
            best_params = None
        
        # Robustness score
        positive_oos = sum(1 for p in periods if p.oos_sharpe > 0)
        robustness_score = positive_oos / len(periods) if periods else 0
        
        return WalkForwardResult(
            pair=pair,
            periods=periods,
            best_params=best_params,
            efficiency_ratio=efficiency_ratio,
            is_sharpe=avg_is_sharpe,
            oos_sharpe=avg_oos_sharpe,
            total_trades=total_trades,
            robustness_score=robustness_score
        )
    
    def _optimize_period(
        self,
        pair: Tuple[str, str],
        price_a: pd.Series,
        price_b: pd.Series
    ) -> Tuple[OptimizationParams, Optional[BacktestResult]]:
        """
        Optimize parameters on in-sample period.
        
        Returns:
            (best_params, best_result)
        """
        best_sharpe = -np.inf
        best_params = None
        best_result = None
        
        # Generate parameter combinations
        param_combinations = list(product(*self.param_grid.values()))
        param_names = list(self.param_grid.keys())
        
        for combo in param_combinations:
            params = OptimizationParams(**dict(zip(param_names, combo)))
            
            # Create settings with these params
            test_settings = self._create_test_settings(params)
            
            # Run backtest
            engine = BacktestEngine(test_settings)
            result = engine.run_backtest(pair, price_a, price_b)
            
            if result and result.sharpe_ratio > best_sharpe:
                best_sharpe = result.sharpe_ratio
                best_params = params
                best_result = result
        
        if best_params is None:
            best_params = OptimizationParams()
        
        return best_params, best_result
    
    def _validate_period(
        self,
        pair: Tuple[str, str],
        price_a: pd.Series,
        price_b: pd.Series,
        params: OptimizationParams
    ) -> Optional[BacktestResult]:
        """
        Validate parameters on out-of-sample period.
        
        Returns:
            BacktestResult
        """
        test_settings = self._create_test_settings(params)
        
        engine = BacktestEngine(test_settings)
        result = engine.run_backtest(pair, price_a, price_b)
        
        return result
    
    def _create_test_settings(self, params: OptimizationParams) -> Settings:
        """Create settings with test parameters."""
        settings = deepcopy(self.settings)
        
        settings.spread.entry_zscore = params.entry_zscore
        settings.spread.exit_zscore = params.exit_zscore
        settings.spread.stop_loss_zscore = params.stop_loss_zscore
        settings.spread.regression_window = params.regression_window
        settings.spread.zscore_window = params.zscore_window
        settings.spread.min_correlation = params.min_correlation
        
        return settings
    
    def set_param_grid(
        self,
        entry_zscore: Optional[List[float]] = None,
        exit_zscore: Optional[List[float]] = None,
        stop_loss_zscore: Optional[List[float]] = None,
        regression_window: Optional[List[int]] = None,
        zscore_window: Optional[List[int]] = None,
        min_correlation: Optional[List[float]] = None
    ):
        """
        Customize parameter search space.
        
        Args:
            entry_zscore: Entry threshold values
            exit_zscore: Exit threshold values
            stop_loss_zscore: Stop loss threshold values
            regression_window: Regression window sizes
            zscore_window: Z-score window sizes
            min_correlation: Minimum correlation values
        """
        if entry_zscore:
            self.param_grid['entry_zscore'] = entry_zscore
        if exit_zscore:
            self.param_grid['exit_zscore'] = exit_zscore
        if stop_loss_zscore:
            self.param_grid['stop_loss_zscore'] = stop_loss_zscore
        if regression_window:
            self.param_grid['regression_window'] = regression_window
        if zscore_window:
            self.param_grid['zscore_window'] = zscore_window
        if min_correlation:
            self.param_grid['min_correlation'] = min_correlation
    
    def save_results(self, result: WalkForwardResult, filepath: str):
        """Save optimization results to JSON."""
        data = {
            'pair': result.pair,
            'efficiency_ratio': result.efficiency_ratio,
            'is_sharpe': result.is_sharpe,
            'oos_sharpe': result.oos_sharpe,
            'total_trades': result.total_trades,
            'robustness_score': result.robustness_score,
            'best_params': {
                'entry_zscore': result.best_params.entry_zscore,
                'exit_zscore': result.best_params.exit_zscore,
                'stop_loss_zscore': result.best_params.stop_loss_zscore,
                'regression_window': result.best_params.regression_window,
                'zscore_window': result.best_params.zscore_window,
                'min_correlation': result.best_params.min_correlation
            } if result.best_params else None,
            'periods': [
                {
                    'period': p.period_index,
                    'is_start': p.is_start.isoformat(),
                    'is_end': p.is_end.isoformat(),
                    'oos_start': p.oos_start.isoformat(),
                    'oos_end': p.oos_end.isoformat(),
                    'is_sharpe': p.is_sharpe,
                    'oos_sharpe': p.oos_sharpe,
                    'oos_trades': p.oos_trades,
                    'params': {
                        'entry_zscore': p.best_params.entry_zscore,
                        'exit_zscore': p.best_params.exit_zscore
                    } if p.best_params else None
                }
                for p in result.periods
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Optimization results saved to {filepath}")
    
    def generate_report(self, result: WalkForwardResult) -> str:
        """Generate human-readable optimization report."""
        report = []
        report.append("="*60)
        report.append("WALK-FORWARD OPTIMIZATION REPORT")
        report.append("="*60)
        report.append(f"\nPair: {result.pair[0]}/{result.pair[1]}")
        report.append(f"Periods analyzed: {len(result.periods)}")
        
        report.append("\nPERFORMANCE SUMMARY")
        report.append("-"*40)
        report.append(f"Average IS Sharpe:   {result.is_sharpe:.2f}")
        report.append(f"Average OOS Sharpe:  {result.oos_sharpe:.2f}")
        report.append(f"Efficiency Ratio:    {result.efficiency_ratio:.2%}")
        report.append(f"Robustness Score:    {result.robustness_score:.1%}")
        report.append(f"Total OOS Trades:    {result.total_trades}")
        
        if result.best_params:
            report.append("\nBEST PARAMETERS")
            report.append("-"*40)
            report.append(f"Entry Z-score:      {result.best_params.entry_zscore}")
            report.append(f"Exit Z-score:       {result.best_params.exit_zscore}")
            report.append(f"Stop-loss Z-score:  {result.best_params.stop_loss_zscore}")
            report.append(f"Regression window:  {result.best_params.regression_window}")
            report.append(f"Z-score window:     {result.best_params.zscore_window}")
            report.append(f"Min correlation:    {result.best_params.min_correlation}")
        
        report.append("\nPERIOD BREAKDOWN")
        report.append("-"*40)
        
        for p in result.periods:
            status = "✓" if p.oos_sharpe > 0 else "✗"
            report.append(f"\nPeriod {p.period_index + 1}: {status}")
            report.append(f"  IS: {p.is_start.strftime('%Y-%m-%d')} to {p.is_end.strftime('%Y-%m-%d')}")
            report.append(f"  OOS: {p.oos_start.strftime('%Y-%m-%d')} to {p.oos_end.strftime('%Y-%m-%d')}")
            report.append(f"  IS Sharpe: {p.is_sharpe:.2f} | OOS Sharpe: {p.oos_sharpe:.2f}")
            report.append(f"  OOS Trades: {p.oos_trades}")
        
        report.append("\n" + "="*60)
        
        # Recommendation
        if result.efficiency_ratio >= 0.5 and result.robustness_score >= 0.6:
            report.append("RECOMMENDATION: Strategy shows robust performance.")
            report.append("Consider forward testing with optimal parameters.")
        elif result.efficiency_ratio >= 0.3:
            report.append("RECOMMENDATION: Strategy shows moderate performance.")
            report.append("Consider further validation before live trading.")
        else:
            report.append("WARNING: Low efficiency ratio suggests overfitting.")
            report.append("Strategy may not perform well out-of-sample.")
        
        report.append("="*60)
        
        return "\n".join(report)
