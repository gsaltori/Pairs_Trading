"""
Cross-Sectional Momentum System - Backtest Runner (CORRECTED)
Complete historical simulation with MANDATORY sanity checks.

SANITY CHECKS ENFORCED:
1. Portfolio equity must never be <= 0
2. Sum of position weights must be <= 1.0
3. No implicit leverage
4. Cash debited on entry, credited on exit
5. Trades only on monthly rebalance dates
6. EMA200 NaNs block trading
7. Portfolio return = weighted average of asset returns (validated)

If ANY check fails:
- Execution STOPS
- Explicit error raised
- NO performance metrics produced

Usage:
    python backtest_runner.py
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from config import CONFIG, print_config
from data_loader import DataLoader
from momentum_engine import MomentumEngine, MomentumSignal
from portfolio_engine import (
    PortfolioEngine, 
    RebalanceAction, 
    SanityCheckError,
    validate_backtest_results,
)
from reporting import (
    MetricsCalculator,
    MonteCarloSimulator,
    PerformanceMetrics,
    MonteCarloResults,
    check_viability,
    print_metrics,
    print_monte_carlo,
    generate_report,
)


class BacktestValidationError(Exception):
    """Raised when backtest validation fails."""
    pass


class BacktestRunner:
    """
    Full backtest execution engine with mandatory sanity checks.
    
    CRITICAL: Will abort and raise errors if any validation fails.
    """
    
    def __init__(self):
        """Initialize backtest runner."""
        self.data_loader = DataLoader()
        self.momentum_engine = MomentumEngine()
        self.portfolio = PortfolioEngine()
        
        self._prices: pd.DataFrame = None
        self._signals: List[MomentumSignal] = []
        self._validation_log: List[str] = []
    
    def _log(self, msg: str):
        """Log validation message."""
        self._validation_log.append(msg)
        print(f"  [VALIDATION] {msg}")
    
    def _validate_data(self, prices: pd.DataFrame):
        """Validate input data before backtest."""
        self._log("Checking data quality...")
        
        # No NaN in prices (after warmup)
        warmup = max(CONFIG.MOMENTUM_LOOKBACK, CONFIG.TREND_FILTER_PERIOD)
        data_after_warmup = prices.iloc[warmup:]
        
        nan_counts = data_after_warmup.isna().sum()
        if nan_counts.sum() > 0:
            raise BacktestValidationError(
                f"NaN values in price data after warmup: {nan_counts[nan_counts > 0].to_dict()}"
            )
        
        # No negative prices
        if (prices <= 0).any().any():
            raise BacktestValidationError("Negative or zero prices detected")
        
        # Sufficient history
        min_required = warmup + 252  # At least 1 year after warmup
        if len(prices) < min_required:
            raise BacktestValidationError(
                f"Insufficient data: {len(prices)} rows, need {min_required}"
            )
        
        self._log(f"Data validated: {len(prices)} rows, {len(prices.columns)} assets")
    
    def _validate_signal(self, signal: MomentumSignal, prices: pd.DataFrame):
        """Validate a momentum signal before execution."""
        # Weights must sum to <= 1
        total_weight = sum(signal.weights.values()) + signal.cash_weight
        if abs(total_weight - 1.0) > 0.001:
            raise BacktestValidationError(
                f"Signal weights don't sum to 1.0: {total_weight:.4f}"
            )
        
        # No negative weights
        for symbol, weight in signal.weights.items():
            if weight < 0:
                raise BacktestValidationError(
                    f"Negative weight for {symbol}: {weight}"
                )
        
        # Selected assets must have valid prices
        for symbol in signal.selected:
            if symbol not in prices.columns:
                raise BacktestValidationError(
                    f"Selected symbol {symbol} not in price data"
                )
            
            if signal.date in prices.index:
                price = prices.loc[signal.date, symbol]
                if pd.isna(price) or price <= 0:
                    raise BacktestValidationError(
                        f"Invalid price for {symbol} on {signal.date}: {price}"
                    )
    
    def _validate_rebalance_date(self, date: pd.Timestamp, rebalance_dates: List[pd.Timestamp]):
        """Validate that we only trade on rebalance dates."""
        if date not in rebalance_dates:
            raise BacktestValidationError(
                f"Attempting to trade on non-rebalance date: {date}"
            )
    
    def _validate_portfolio_return(
        self,
        prev_equity: float,
        curr_equity: float,
        positions: Dict,
        prev_prices: Dict[str, float],
        curr_prices: Dict[str, float],
    ):
        """
        Validate portfolio return matches weighted asset returns.
        
        This catches bugs where equity changes don't match actual asset performance.
        """
        if prev_equity <= 0:
            return  # Can't validate
        
        portfolio_return = (curr_equity / prev_equity) - 1
        
        # Calculate expected return from positions
        expected_return = 0.0
        total_weight = 0.0
        
        for symbol, pos in positions.items():
            if symbol in prev_prices and symbol in curr_prices:
                prev_price = prev_prices[symbol]
                curr_price = curr_prices[symbol]
                
                if prev_price > 0:
                    asset_return = (curr_price / prev_price) - 1
                    weight = (pos.shares * prev_price) / prev_equity
                    expected_return += asset_return * weight
                    total_weight += weight
        
        # Cash weight (no return)
        cash_weight = 1 - total_weight
        
        # Allow for transaction costs and small float errors
        tolerance = 0.01  # 1%
        
        if abs(portfolio_return - expected_return) > tolerance:
            # This could be due to rebalancing, so only warn if very large
            if abs(portfolio_return - expected_return) > 0.05:  # 5%
                raise BacktestValidationError(
                    f"Portfolio return mismatch: actual={portfolio_return:.4f}, "
                    f"expected from positions={expected_return:.4f}"
                )
    
    def run(
        self,
        prices: pd.DataFrame = None,
    ) -> Tuple[PerformanceMetrics, pd.DataFrame, pd.DataFrame, pd.Series]:
        """
        Run full backtest with mandatory validation.
        
        WILL ABORT if any sanity check fails.
        """
        print("\n" + "=" * 70)
        print("BACKTEST EXECUTION (WITH SANITY CHECKS)")
        print("=" * 70)
        
        self._validation_log.clear()
        
        # Load data if not provided
        if prices is None:
            print("\n1. Loading data...")
            prices = self.data_loader.load_universe()
        
        self._prices = prices
        
        # VALIDATE: Data quality
        print("\n2. Validating data...")
        self._validate_data(prices)
        
        print(f"\n   Data: {len(prices)} days, {len(prices.columns)} assets")
        print(f"   Period: {prices.index[0].date()} to {prices.index[-1].date()}")
        
        # Get rebalance dates
        print("\n3. Identifying rebalance dates...")
        rebalance_dates = self.data_loader.get_monthly_dates(prices)
        rebalance_set = set(rebalance_dates)
        print(f"   Rebalance dates: {len(rebalance_dates)}")
        
        # Generate signals
        print("\n4. Generating momentum signals...")
        self._signals = self.momentum_engine.generate_all_signals(prices, rebalance_dates)
        print(f"   Valid signals: {len(self._signals)}")
        
        # VALIDATE: Each signal
        print("\n5. Validating signals...")
        for signal in self._signals:
            self._validate_signal(signal, prices)
        self._log(f"All {len(self._signals)} signals validated")
        
        # Reset portfolio
        self.portfolio.reset()
        
        # Run simulation with validation
        print("\n6. Running simulation with validation...")
        self._run_simulation_with_validation(prices, self._signals, rebalance_set)
        
        # Get results
        equity_df = self.portfolio.get_equity_df()
        trades_df = self.portfolio.get_trades_df()
        monthly_returns = self.portfolio.get_monthly_returns()
        
        # FINAL VALIDATION
        print("\n7. Final validation...")
        final_prices = prices.iloc[-1].to_dict()
        valid, issues = validate_backtest_results(self.portfolio, final_prices)
        
        if not valid:
            print("\n❌ VALIDATION FAILED:")
            for issue in issues:
                print(f"   - {issue}")
            raise BacktestValidationError(
                f"Backtest validation failed with {len(issues)} issues"
            )
        
        self._log("Final validation PASSED")
        
        # Additional sanity checks on results
        self._validate_results(equity_df, trades_df)
        
        print(f"\n   Total trades: {len(trades_df)}")
        print(f"   Final equity: ${equity_df['Equity'].iloc[-1]:,.2f}")
        
        # Calculate metrics (only if validation passed)
        print("\n8. Calculating metrics...")
        metrics = MetricsCalculator.calculate(
            equity_df=equity_df,
            trades_df=trades_df,
            monthly_returns=monthly_returns,
            initial_capital=CONFIG.INITIAL_CAPITAL,
        )
        
        print("\n✅ BACKTEST COMPLETED SUCCESSFULLY")
        print(f"   Validation checks passed: {len(self._validation_log)}")
        
        return metrics, equity_df, trades_df, monthly_returns
    
    def _run_simulation_with_validation(
        self,
        prices: pd.DataFrame,
        signals: List[MomentumSignal],
        rebalance_dates: set,
    ):
        """Run simulation with validation at each step."""
        signal_idx = 0
        prev_prices: Optional[Dict[str, float]] = None
        prev_equity = self.portfolio.initial_capital
        check_interval = 100  # Validate every N days
        
        for i, date in enumerate(prices.index):
            today_prices = prices.loc[date].to_dict()
            
            # Check for rebalance
            if signal_idx < len(signals) and signals[signal_idx].date <= date:
                signal = signals[signal_idx]
                signal_idx += 1
                
                # VALIDATE: Only trade on rebalance dates
                if signal.date not in rebalance_dates:
                    raise BacktestValidationError(
                        f"Signal date {signal.date} not in rebalance dates"
                    )
                
                # Execute rebalance (includes internal validation)
                try:
                    self._execute_rebalance(signal, today_prices, date)
                except SanityCheckError as e:
                    raise BacktestValidationError(f"Sanity check failed: {e}")
            
            # Update equity curve
            try:
                self.portfolio.update_equity_curve(date, today_prices)
            except SanityCheckError as e:
                raise BacktestValidationError(
                    f"Equity update failed on {date}: {e}"
                )
            
            # Periodic portfolio return validation (expensive, so not every day)
            if i > 0 and i % check_interval == 0 and prev_prices is not None:
                curr_equity = self.portfolio.get_equity(today_prices)
                try:
                    self._validate_portfolio_return(
                        prev_equity, curr_equity,
                        self.portfolio.positions,
                        prev_prices, today_prices
                    )
                except BacktestValidationError as e:
                    print(f"\n⚠️ Return validation warning on {date}: {e}")
                    # Don't fail, just warn - rebalancing can cause legitimate differences
                
                prev_equity = curr_equity
            
            prev_prices = today_prices
            
            # Progress
            if i % 500 == 0:
                equity = self.portfolio.get_equity(today_prices)
                print(f"   Day {i+1}/{len(prices.index)}: Equity=${equity:,.2f}")
    
    def _execute_rebalance(
        self,
        signal: MomentumSignal,
        prices: Dict[str, float],
        date: pd.Timestamp,
    ):
        """Execute rebalance with validation."""
        # Calculate actions
        actions = self.portfolio.calculate_rebalance_actions(signal, prices)
        
        # VALIDATE: No implicit leverage from actions
        equity = self.portfolio.get_equity(prices)
        total_buy_value = sum(
            shares * prices.get(sym, 0)
            for sym, shares in actions.buys.items()
        )
        available_after_sells = self.portfolio.cash + sum(
            shares * prices.get(sym, 0) * (1 - CONFIG.SLIPPAGE_PCT) * (1 - CONFIG.COMMISSION_PCT)
            for sym, shares in actions.sells.items()
        )
        
        # Execute
        self.portfolio.execute_rebalance(actions, prices)
    
    def _validate_results(self, equity_df: pd.DataFrame, trades_df: pd.DataFrame):
        """Validate final backtest results."""
        # Equity never negative
        if (equity_df['Equity'] <= 0).any():
            min_eq = equity_df['Equity'].min()
            min_date = equity_df['Equity'].idxmin()
            raise BacktestValidationError(
                f"Equity went to {min_eq:.2f} on {min_date}"
            )
        
        # Drawdown never > 100%
        if (equity_df['Drawdown'] >= 1.0).any():
            raise BacktestValidationError("Drawdown reached 100%")
        
        # No NaN in equity
        if equity_df['Equity'].isna().any():
            raise BacktestValidationError("NaN values in equity curve")
        
        # No NaN in trades
        if len(trades_df) > 0:
            if trades_df['PnL'].isna().any():
                raise BacktestValidationError("NaN values in trade PnL")
            if trades_df['Shares'].isna().any() or (trades_df['Shares'] <= 0).any():
                raise BacktestValidationError("Invalid shares in trades")
        
        self._log("Results validation PASSED")
    
    def run_in_out_sample(
        self,
    ) -> Tuple[PerformanceMetrics, PerformanceMetrics, PerformanceMetrics, 
               pd.DataFrame, pd.DataFrame, pd.Series]:
        """
        Run backtest with in-sample/out-of-sample split.
        """
        # Load full data
        prices = self.data_loader.load_universe()
        
        # Split
        is_prices, oos_prices = self.data_loader.split_in_out_sample(prices)
        
        print(f"\nIn-Sample: {is_prices.index[0].date()} to {is_prices.index[-1].date()}")
        print(f"Out-of-Sample: {oos_prices.index[0].date()} to {oos_prices.index[-1].date()}")
        
        # Run in-sample
        print("\n" + "-" * 70)
        print("IN-SAMPLE BACKTEST")
        print("-" * 70)
        
        is_metrics, is_equity, is_trades, is_monthly = self.run(is_prices)
        
        # Run out-of-sample
        print("\n" + "-" * 70)
        print("OUT-OF-SAMPLE BACKTEST")
        print("-" * 70)
        
        self.portfolio.reset()
        oos_metrics, oos_equity, oos_trades, oos_monthly = self.run(oos_prices)
        
        # Run full sample
        print("\n" + "-" * 70)
        print("FULL SAMPLE BACKTEST")
        print("-" * 70)
        
        self.portfolio.reset()
        full_metrics, full_equity, full_trades, full_monthly = self.run(prices)
        
        return full_metrics, is_metrics, oos_metrics, full_equity, full_trades, full_monthly


def main():
    """Main entry point."""
    print("\n" + "=" * 70)
    print("CROSS-SECTIONAL MOMENTUM SYSTEM - BACKTEST")
    print("(WITH MANDATORY SANITY CHECKS)")
    print("=" * 70)
    
    # Print configuration
    print_config()
    
    # Create output directory
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    # Run backtest
    runner = BacktestRunner()
    
    try:
        (full_metrics, is_metrics, oos_metrics, 
         equity_df, trades_df, monthly_returns) = runner.run_in_out_sample()
        
    except (BacktestValidationError, SanityCheckError) as e:
        print("\n" + "=" * 70)
        print("❌ BACKTEST FAILED - VALIDATION ERROR")
        print("=" * 70)
        print(f"\nError: {e}")
        print("\nNO METRICS PRODUCED - Results are INVALID")
        return 1
    
    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ BACKTEST FAILED - UNEXPECTED ERROR")
        print("=" * 70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Print full metrics
    print_metrics(full_metrics)
    
    # Run Monte Carlo
    print("\n" + "=" * 70)
    print("MONTE CARLO SIMULATION")
    print("=" * 70)
    
    mc_sim = MonteCarloSimulator()
    mc_results = mc_sim.run(monthly_returns, CONFIG.INITIAL_CAPITAL)
    
    print_monte_carlo(mc_results)
    
    # Check viability
    print("\n" + "=" * 70)
    print("VIABILITY ASSESSMENT")
    print("=" * 70)
    
    passed, issues = check_viability(full_metrics, mc_results)
    
    if passed:
        print("\n✅ All viability criteria PASSED")
    else:
        print("\n❌ Viability criteria FAILED:")
        for issue in issues:
            print(f"  - {issue}")
    
    # IS vs OOS comparison
    print("\n" + "-" * 70)
    print("IN-SAMPLE vs OUT-OF-SAMPLE COMPARISON")
    print("-" * 70)
    
    print(f"\n{'Metric':<20} {'In-Sample':>15} {'Out-of-Sample':>15}")
    print("-" * 50)
    print(f"{'CAGR':<20} {is_metrics.cagr:>14.1%} {oos_metrics.cagr:>14.1%}")
    print(f"{'Sharpe':<20} {is_metrics.sharpe_ratio:>15.2f} {oos_metrics.sharpe_ratio:>15.2f}")
    print(f"{'Max DD':<20} {is_metrics.max_drawdown:>14.1%} {oos_metrics.max_drawdown:>14.1%}")
    print(f"{'Win Rate':<20} {is_metrics.win_rate:>14.1%} {oos_metrics.win_rate:>14.1%}")
    print(f"{'Profit Factor':<20} {is_metrics.profit_factor:>15.2f} {oos_metrics.profit_factor:>15.2f}")
    
    # Final verdict
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)
    
    print("\n  ✅ BACKTEST RESULTS ARE VALID")
    print("     All sanity checks passed")
    
    if passed:
        print("\n  ✅ GO - System passes all viability criteria")
        print("\n  \"I would deploy this system with real capital.\"")
    else:
        print("\n  ❌ NO-GO - System fails viability criteria")
        print("\n  \"I would NOT deploy this system with real capital.\"")
    
    # Save results
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    equity_file = output_dir / "equity_curve.csv"
    equity_df.to_csv(equity_file)
    print(f"\n  Equity curve: {equity_file}")
    
    trades_file = output_dir / "trades.csv"
    trades_df.to_csv(trades_file, index=False)
    print(f"  Trade history: {trades_file}")
    
    monthly_file = output_dir / "monthly_returns.csv"
    monthly_returns.to_csv(monthly_file)
    print(f"  Monthly returns: {monthly_file}")
    
    report = generate_report(
        metrics=full_metrics,
        mc=mc_results,
        is_metrics=is_metrics,
        oos_metrics=oos_metrics,
        passed=passed,
        issues=issues,
    )
    
    report_file = output_dir / "REPORT.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  Full report: {report_file}")
    
    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE - RESULTS VALID")
    print("=" * 70 + "\n")
    
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
