"""
Walk-Forward Optimization for the Pairs Trading System.

Provides:
- Walk-forward analysis with rolling windows
- Parameter grid search
- Out-of-sample validation
- Robustness metrics
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime
from itertools import product
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import json

from ..backtest.backtest_engine import BacktestEngine, BacktestResult
from config.settings import Settings, SpreadParameters, RiskParameters


logger = logging.getLogger(__name__)


@dataclass
class ParameterSet:
    """A set of parameters to optimize."""
    entry_zscore: float
    exit_zscore: float
    stop_loss_zscore: float
    regression_window: int
    zscore_window: int
    min_correlation: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'entry_zscore': self.entry_zscore,
            'exit_zscore': self.exit_zscore,
            'stop_loss_zscore': self.stop_loss_zscore,
            'regression_window': self.regression_window,
            'zscore_window': self.zscore_window,
            'min_correlation': self.min_correlation
        }


@dataclass
class WalkForwardPeriod:
    """Results for a single walk-forward period."""
    period_num: int
    in_sample_start: datetime
    in_sample_end: datetime
    out_sample_start: datetime
    out_sample_end: datetime
    best_params: ParameterSet
    in_sample_result: BacktestResult
    out_sample_result: BacktestResult
    
    def efficiency_ratio(self) -> float:
        """Calculate walk-forward efficiency ratio."""
        if self.in_sample_result.sharpe_ratio == 0:
            return 0.0
        return self.out_sample_result.sharpe_ratio / self.in_sample_result.sharpe_ratio


@dataclass
class OptimizationResult:
    """Complete walk-forward optimization results."""
    pair: Tuple[str, str]
    total_periods: int
    parameter_combinations_tested: int
    
    # Aggregated out-of-sample metrics
    combined_sharpe: float
    combined_return: float
    combined_max_drawdown: float
    combined_win_rate: float
    avg_efficiency_ratio: float
    
    # Individual periods
    periods: List[WalkForwardPeriod] = field(default_factory=list)
    
    # Best parameters across all periods
    most_robust_params: Optional[ParameterSet] = None
    
    # Combined equity curve
    combined_equity: pd.Series = field(default_factory=pd.Series)
    
    def summary(self) -> str:
        """Generate text summary."""
        return f"""
================================================================================
WALK-FORWARD OPTIMIZATION RESULTS: {self.pair[0]}/{self.pair[1]}
================================================================================
Total Periods: {self.total_periods}
Parameter Combinations Tested: {self.parameter_combinations_tested}

COMBINED OUT-OF-SAMPLE PERFORMANCE
----------------------------------
Sharpe Ratio:    {self.combined_sharpe:.3f}
Total Return:    {self.combined_return:.2%}
Max Drawdown:    {self.combined_max_drawdown:.2%}
Win Rate:        {self.combined_win_rate:.2%}
Efficiency Ratio: {self.avg_efficiency_ratio:.2%}

MOST ROBUST PARAMETERS
----------------------
Entry Z-Score:     {self.most_robust_params.entry_zscore if self.most_robust_params else 'N/A'}
Exit Z-Score:      {self.most_robust_params.exit_zscore if self.most_robust_params else 'N/A'}
Stop Loss Z-Score: {self.most_robust_params.stop_loss_zscore if self.most_robust_params else 'N/A'}
Regression Window: {self.most_robust_params.regression_window if self.most_robust_params else 'N/A'}
Z-Score Window:    {self.most_robust_params.zscore_window if self.most_robust_params else 'N/A'}
Min Correlation:   {self.most_robust_params.min_correlation if self.most_robust_params else 'N/A'}

PERIOD-BY-PERIOD BREAKDOWN
--------------------------"""
        + "\n".join([
            f"Period {p.period_num}: IS Sharpe={p.in_sample_result.sharpe_ratio:.2f}, "
            f"OOS Sharpe={p.out_sample_result.sharpe_ratio:.2f}, "
            f"Efficiency={p.efficiency_ratio():.2%}"
            for p in self.periods
        ]) + "\n" + "=" * 80


class WalkForwardOptimizer:
    """
    Walk-forward optimizer for pairs trading strategies.
    
    Walk-forward analysis:
    1. Split data into in-sample (IS) and out-of-sample (OOS) periods
    2. Optimize parameters on IS period
    3. Test on OOS period
    4. Roll forward and repeat
    
    This provides more realistic performance expectations than
    single-period backtesting.
    """
    
    def __init__(
        self,
        settings: Settings,
        objective: str = 'sharpe'
    ):
        """
        Initialize optimizer.
        
        Args:
            settings: Trading system settings
            objective: Optimization objective ('sharpe', 'return', 'calmar', 'sortino')
        """
        self.settings = settings
        self.opt_params = settings.optimization
        self.objective = objective
        
        # Default parameter grid
        self.param_grid = {
            'entry_zscore': [1.5, 2.0, 2.5],
            'exit_zscore': [0.0, 0.25, 0.5],
            'stop_loss_zscore': [2.5, 3.0, 3.5],
            'regression_window': [90, 120, 150],
            'zscore_window': [30, 60, 90],
            'min_correlation': [0.65, 0.70, 0.75]
        }
        
        logger.info(f"WalkForwardOptimizer initialized with objective: {objective}")
    
    def set_param_grid(self, param_grid: Dict[str, List]) -> None:
        """Set custom parameter grid."""
        self.param_grid = param_grid
        logger.info(f"Parameter grid updated: {len(self._generate_param_combinations())} combinations")
    
    def _generate_param_combinations(self) -> List[ParameterSet]:
        """Generate all parameter combinations."""
        keys = list(self.param_grid.keys())
        values = [self.param_grid[k] for k in keys]
        
        combinations = []
        for combo in product(*values):
            params = dict(zip(keys, combo))
            combinations.append(ParameterSet(**params))
        
        return combinations
    
    def optimize(
        self,
        pair: Tuple[str, str],
        data_a: pd.DataFrame,
        data_b: pd.DataFrame,
        n_periods: Optional[int] = None,
        parallel: bool = False
    ) -> OptimizationResult:
        """
        Run walk-forward optimization.
        
        Args:
            pair: Tuple of (instrument_a, instrument_b)
            data_a: OHLCV data for instrument A
            data_b: OHLCV data for instrument B
            n_periods: Number of walk-forward periods (auto-calculated if None)
            parallel: Whether to use parallel processing
            
        Returns:
            OptimizationResult with all periods and metrics
        """
        logger.info(f"Starting walk-forward optimization for {pair[0]}/{pair[1]}")
        
        # Calculate periods
        is_bars = self.opt_params.in_sample_bars
        oos_bars = self.opt_params.out_sample_bars
        total_bars = len(data_a)
        
        if n_periods is None:
            # Calculate how many periods fit
            period_size = is_bars + oos_bars
            n_periods = max(1, (total_bars - is_bars) // oos_bars)
        
        logger.info(f"Running {n_periods} walk-forward periods "
                   f"(IS={is_bars} bars, OOS={oos_bars} bars)")
        
        # Generate parameter combinations
        param_combos = self._generate_param_combinations()
        n_combos = len(param_combos)
        logger.info(f"Testing {n_combos} parameter combinations per period")
        
        # Walk-forward loop
        periods = []
        all_oos_equity = []
        
        for period in range(n_periods):
            # Calculate period boundaries
            is_start_idx = period * oos_bars
            is_end_idx = is_start_idx + is_bars
            oos_start_idx = is_end_idx
            oos_end_idx = oos_start_idx + oos_bars
            
            # Check bounds
            if oos_end_idx > total_bars:
                logger.info(f"Not enough data for period {period + 1}, stopping")
                break
            
            # Extract period data
            is_data_a = data_a.iloc[is_start_idx:is_end_idx].copy()
            is_data_b = data_b.iloc[is_start_idx:is_end_idx].copy()
            oos_data_a = data_a.iloc[oos_start_idx:oos_end_idx].copy()
            oos_data_b = data_b.iloc[oos_start_idx:oos_end_idx].copy()
            
            logger.info(f"Period {period + 1}: IS {is_data_a.index[0]} to {is_data_a.index[-1]}, "
                       f"OOS {oos_data_a.index[0]} to {oos_data_a.index[-1]}")
            
            # Optimize on in-sample
            best_params, best_is_result = self._optimize_in_sample(
                pair, is_data_a, is_data_b, param_combos, parallel
            )
            
            # Test on out-of-sample
            oos_result = self._run_with_params(
                pair, oos_data_a, oos_data_b, best_params
            )
            
            # Record period results
            wf_period = WalkForwardPeriod(
                period_num=period + 1,
                in_sample_start=is_data_a.index[0],
                in_sample_end=is_data_a.index[-1],
                out_sample_start=oos_data_a.index[0],
                out_sample_end=oos_data_a.index[-1],
                best_params=best_params,
                in_sample_result=best_is_result,
                out_sample_result=oos_result
            )
            
            periods.append(wf_period)
            
            # Collect OOS equity for combined curve
            if len(oos_result.equity_curve) > 0:
                all_oos_equity.append(oos_result.equity_curve)
            
            logger.info(f"Period {period + 1} complete: "
                       f"IS Sharpe={best_is_result.sharpe_ratio:.2f}, "
                       f"OOS Sharpe={oos_result.sharpe_ratio:.2f}")
        
        # Calculate combined metrics
        result = self._calculate_combined_results(pair, periods, n_combos, all_oos_equity)
        
        logger.info(f"Optimization complete: Combined Sharpe={result.combined_sharpe:.2f}")
        
        return result
    
    def _optimize_in_sample(
        self,
        pair: Tuple[str, str],
        data_a: pd.DataFrame,
        data_b: pd.DataFrame,
        param_combos: List[ParameterSet],
        parallel: bool = False
    ) -> Tuple[ParameterSet, BacktestResult]:
        """Optimize parameters on in-sample data."""
        best_score = float('-inf')
        best_params = param_combos[0]
        best_result = None
        
        if parallel:
            # Parallel optimization (requires multiprocessing-safe code)
            results = self._parallel_optimize(pair, data_a, data_b, param_combos)
        else:
            results = []
            for params in param_combos:
                result = self._run_with_params(pair, data_a, data_b, params)
                results.append((params, result))
        
        # Find best
        for params, result in results:
            score = self._calculate_objective(result)
            if score > best_score:
                best_score = score
                best_params = params
                best_result = result
        
        return best_params, best_result
    
    def _run_with_params(
        self,
        pair: Tuple[str, str],
        data_a: pd.DataFrame,
        data_b: pd.DataFrame,
        params: ParameterSet
    ) -> BacktestResult:
        """Run backtest with specific parameters."""
        # Create settings copy with modified parameters
        settings_copy = Settings()
        settings_copy.spread.entry_zscore = params.entry_zscore
        settings_copy.spread.exit_zscore = params.exit_zscore
        settings_copy.spread.stop_loss_zscore = params.stop_loss_zscore
        settings_copy.spread.regression_window = params.regression_window
        settings_copy.spread.zscore_window = params.zscore_window
        settings_copy.spread.min_correlation = params.min_correlation
        
        # Copy backtest settings
        settings_copy.backtest = self.settings.backtest
        settings_copy.risk = self.settings.risk
        
        # Run backtest
        engine = BacktestEngine(settings_copy)
        result = engine.run_backtest(pair, data_a, data_b)
        
        return result
    
    def _parallel_optimize(
        self,
        pair: Tuple[str, str],
        data_a: pd.DataFrame,
        data_b: pd.DataFrame,
        param_combos: List[ParameterSet]
    ) -> List[Tuple[ParameterSet, BacktestResult]]:
        """Run optimization in parallel."""
        # Note: This is a simplified version. In practice, you'd need
        # to handle serialization carefully for multiprocessing.
        results = []
        
        # Fall back to sequential for now (parallel requires more setup)
        for params in param_combos:
            result = self._run_with_params(pair, data_a, data_b, params)
            results.append((params, result))
        
        return results
    
    def _calculate_objective(self, result: BacktestResult) -> float:
        """Calculate optimization objective score."""
        if self.objective == 'sharpe':
            return result.sharpe_ratio
        elif self.objective == 'return':
            return result.total_return
        elif self.objective == 'calmar':
            return result.calmar_ratio
        elif self.objective == 'sortino':
            return result.sortino_ratio
        else:
            return result.sharpe_ratio
    
    def _calculate_combined_results(
        self,
        pair: Tuple[str, str],
        periods: List[WalkForwardPeriod],
        n_combos: int,
        all_oos_equity: List[pd.Series]
    ) -> OptimizationResult:
        """Calculate combined walk-forward results."""
        if not periods:
            return OptimizationResult(
                pair=pair,
                total_periods=0,
                parameter_combinations_tested=n_combos,
                combined_sharpe=0.0,
                combined_return=0.0,
                combined_max_drawdown=0.0,
                combined_win_rate=0.0,
                avg_efficiency_ratio=0.0
            )
        
        # Combine OOS equity curves
        if all_oos_equity:
            # Chain equity curves together
            combined_equity = self._chain_equity_curves(all_oos_equity)
        else:
            combined_equity = pd.Series()
        
        # Combined metrics from OOS results
        oos_returns = [p.out_sample_result.total_return for p in periods]
        oos_sharpes = [p.out_sample_result.sharpe_ratio for p in periods]
        oos_drawdowns = [p.out_sample_result.max_drawdown for p in periods]
        oos_win_rates = [p.out_sample_result.win_rate for p in periods]
        
        # Efficiency ratios
        efficiency_ratios = [p.efficiency_ratio() for p in periods]
        
        # Calculate combined Sharpe from combined equity
        if len(combined_equity) > 1:
            returns = combined_equity.pct_change().dropna()
            if len(returns) > 0 and returns.std() > 0:
                combined_sharpe = (returns.mean() / returns.std()) * np.sqrt(8760)
            else:
                combined_sharpe = 0.0
            
            # Combined drawdown
            rolling_max = combined_equity.expanding().max()
            drawdown = (combined_equity - rolling_max) / rolling_max
            combined_max_dd = abs(drawdown.min())
        else:
            combined_sharpe = np.mean(oos_sharpes) if oos_sharpes else 0.0
            combined_max_dd = np.max(oos_drawdowns) if oos_drawdowns else 0.0
        
        # Combined return
        combined_return = np.prod([1 + r for r in oos_returns]) - 1 if oos_returns else 0.0
        
        # Find most robust parameters (most frequently selected)
        param_counts = {}
        for p in periods:
            key = str(p.best_params.to_dict())
            param_counts[key] = param_counts.get(key, 0) + 1
        
        most_common = max(param_counts, key=param_counts.get) if param_counts else None
        most_robust_params = periods[0].best_params if periods else None
        
        # Find the parameters with best OOS performance
        best_oos_idx = np.argmax(oos_sharpes) if oos_sharpes else 0
        most_robust_params = periods[best_oos_idx].best_params if periods else None
        
        return OptimizationResult(
            pair=pair,
            total_periods=len(periods),
            parameter_combinations_tested=n_combos,
            combined_sharpe=combined_sharpe,
            combined_return=combined_return,
            combined_max_drawdown=combined_max_dd,
            combined_win_rate=np.mean(oos_win_rates) if oos_win_rates else 0.0,
            avg_efficiency_ratio=np.mean(efficiency_ratios) if efficiency_ratios else 0.0,
            periods=periods,
            most_robust_params=most_robust_params,
            combined_equity=combined_equity
        )
    
    def _chain_equity_curves(self, equity_curves: List[pd.Series]) -> pd.Series:
        """Chain multiple equity curves together."""
        if not equity_curves:
            return pd.Series()
        
        chained = []
        last_value = equity_curves[0].iloc[0]  # Starting capital
        
        for curve in equity_curves:
            if len(curve) == 0:
                continue
            
            # Scale curve to continue from last value
            scale_factor = last_value / curve.iloc[0]
            scaled = curve * scale_factor
            
            chained.append(scaled)
            last_value = scaled.iloc[-1]
        
        if not chained:
            return pd.Series()
        
        return pd.concat(chained, ignore_index=True)
    
    def save_results(
        self,
        result: OptimizationResult,
        filepath: str
    ) -> None:
        """Save optimization results to file."""
        data = {
            'pair': f"{result.pair[0]}/{result.pair[1]}",
            'total_periods': result.total_periods,
            'parameter_combinations_tested': result.parameter_combinations_tested,
            'combined_sharpe': result.combined_sharpe,
            'combined_return': result.combined_return,
            'combined_max_drawdown': result.combined_max_drawdown,
            'combined_win_rate': result.combined_win_rate,
            'avg_efficiency_ratio': result.avg_efficiency_ratio,
            'most_robust_params': result.most_robust_params.to_dict() if result.most_robust_params else None,
            'periods': [
                {
                    'period_num': p.period_num,
                    'in_sample_start': p.in_sample_start.isoformat(),
                    'in_sample_end': p.in_sample_end.isoformat(),
                    'out_sample_start': p.out_sample_start.isoformat(),
                    'out_sample_end': p.out_sample_end.isoformat(),
                    'best_params': p.best_params.to_dict(),
                    'in_sample_sharpe': p.in_sample_result.sharpe_ratio,
                    'in_sample_return': p.in_sample_result.total_return,
                    'out_sample_sharpe': p.out_sample_result.sharpe_ratio,
                    'out_sample_return': p.out_sample_result.total_return,
                    'efficiency_ratio': p.efficiency_ratio()
                }
                for p in result.periods
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Optimization results saved to {filepath}")


class GridSearchOptimizer:
    """
    Simple grid search optimizer for single-period optimization.
    
    Use this when walk-forward is not needed (e.g., for quick parameter tuning).
    """
    
    def __init__(self, settings: Settings, objective: str = 'sharpe'):
        """Initialize grid search optimizer."""
        self.settings = settings
        self.objective = objective
        self.param_grid = {}
    
    def set_param_grid(self, param_grid: Dict[str, List]) -> None:
        """Set parameter grid to search."""
        self.param_grid = param_grid
    
    def optimize(
        self,
        pair: Tuple[str, str],
        data_a: pd.DataFrame,
        data_b: pd.DataFrame
    ) -> Tuple[Dict[str, Any], BacktestResult]:
        """
        Run grid search optimization.
        
        Returns:
            Tuple of (best_params_dict, best_backtest_result)
        """
        if not self.param_grid:
            raise ValueError("Parameter grid not set. Call set_param_grid first.")
        
        best_score = float('-inf')
        best_params = None
        best_result = None
        
        # Generate combinations
        keys = list(self.param_grid.keys())
        values = [self.param_grid[k] for k in keys]
        
        total_combos = 1
        for v in values:
            total_combos *= len(v)
        
        logger.info(f"Grid search: testing {total_combos} combinations")
        
        for i, combo in enumerate(product(*values)):
            params = dict(zip(keys, combo))
            
            # Create settings with these parameters
            settings_copy = Settings()
            
            # Apply spread parameters
            if 'entry_zscore' in params:
                settings_copy.spread.entry_zscore = params['entry_zscore']
            if 'exit_zscore' in params:
                settings_copy.spread.exit_zscore = params['exit_zscore']
            if 'stop_loss_zscore' in params:
                settings_copy.spread.stop_loss_zscore = params['stop_loss_zscore']
            if 'regression_window' in params:
                settings_copy.spread.regression_window = params['regression_window']
            if 'zscore_window' in params:
                settings_copy.spread.zscore_window = params['zscore_window']
            if 'min_correlation' in params:
                settings_copy.spread.min_correlation = params['min_correlation']
            
            settings_copy.backtest = self.settings.backtest
            settings_copy.risk = self.settings.risk
            
            # Run backtest
            engine = BacktestEngine(settings_copy)
            result = engine.run_backtest(pair, data_a, data_b)
            
            # Calculate objective
            if self.objective == 'sharpe':
                score = result.sharpe_ratio
            elif self.objective == 'return':
                score = result.total_return
            elif self.objective == 'calmar':
                score = result.calmar_ratio
            else:
                score = result.sharpe_ratio
            
            if score > best_score:
                best_score = score
                best_params = params
                best_result = result
            
            if (i + 1) % 10 == 0:
                logger.debug(f"Tested {i + 1}/{total_combos} combinations")
        
        logger.info(f"Best parameters found: {best_params}, "
                   f"Objective ({self.objective}): {best_score:.4f}")
        
        return best_params, best_result
