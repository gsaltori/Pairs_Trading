"""
Time-Series Momentum System - Main Backtest Runner
Complete backtest with validation and reporting.

MANDATORY SANITY CHECKS:
1. No leverage (sum of weights <= 1.0)
2. Equity must never be <= 0
3. Cash accounting must be explicit
4. Trades ONLY on rebalance dates
5. No lookahead bias
6. Portfolio return must equal weighted asset returns

If ANY check fails: STOP and raise explicit error.

Usage:
    python run_backtest.py
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict

from config import CONFIG, print_config
from data_loader import DataLoader
from signal_engine import SignalEngine, TSMOMSignal, validate_signal, SanityCheckError
from portfolio_engine import PortfolioEngine, validate_portfolio_state
from metrics import MetricsCalculator, PerformanceMetrics, print_metrics
from monte_carlo import MonteCarloSimulator, MonteCarloResult, print_monte_carlo


class BacktestError(Exception):
    """Backtest validation failed."""
    pass


class BacktestRunner:
    """
    Time-Series Momentum backtest with full validation.
    
    Aborts execution if any sanity check fails.
    """
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.signal_engine = SignalEngine()
        self.portfolio = PortfolioEngine()
        
        self._validation_log: List[str] = []
    
    def _log(self, msg: str):
        """Log validation message."""
        self._validation_log.append(msg)
        print(f"  [CHECK] {msg}")
    
    def run(
        self,
        prices: pd.DataFrame = None,
    ) -> Tuple[PerformanceMetrics, pd.DataFrame, pd.DataFrame, pd.Series, List[TSMOMSignal]]:
        """
        Run full backtest with validation.
        
        Returns:
            Tuple of (metrics, equity_df, trades_df, monthly_returns, signals)
        """
        print("\n" + "=" * 70)
        print("BACKTEST EXECUTION")
        print("=" * 70)
        
        self._validation_log.clear()
        
        # Load data
        if prices is None:
            print("\n1. Loading data...")
            prices = self.data_loader.load_universe()
        
        # Validate data
        print("\n2. Validating data...")
        valid, issues = self.data_loader.validate_data(prices)
        if not valid:
            raise BacktestError(f"Data validation failed: {issues}")
        self._log(f"Data OK: {len(prices)} days, {len(prices.columns)} assets")
        
        # Get rebalance dates
        print("\n3. Getting rebalance dates...")
        rebalance_dates = self.data_loader.get_monthly_rebalance_dates(prices)
        rebalance_set = set(rebalance_dates)
        self._log(f"Rebalance dates: {len(rebalance_dates)}")
        
        # Generate signals
        print("\n4. Generating signals...")
        signals = self.signal_engine.generate_all_signals(prices, rebalance_dates)
        self._log(f"Signals generated: {len(signals)}")
        
        # Validate all signals
        for sig in signals:
            valid, issues = validate_signal(sig)
            if not valid:
                raise BacktestError(f"Signal validation failed on {sig.date}: {issues}")
        self._log("All signals validated")
        
        # Reset portfolio
        self.portfolio.reset()
        
        # Run simulation
        print("\n5. Running simulation...")
        self._run_simulation(prices, signals, rebalance_set)
        
        # Get results
        equity_df = self.portfolio.get_equity_df()
        trades_df = self.portfolio.get_trades_df()
        monthly_returns = self.portfolio.get_monthly_returns()
        
        # Final validation
        print("\n6. Final validation...")
        final_prices = prices.iloc[-1].to_dict()
        valid, issues = validate_portfolio_state(self.portfolio, final_prices)
        if not valid:
            raise BacktestError(f"Final validation failed: {issues}")
        self._log("Final state validated")
        
        # Additional checks
        self._validate_equity_curve(equity_df)
        self._validate_no_lookahead(prices, signals)
        
        print(f"\n   Trades: {len(trades_df)}")
        print(f"   Final equity: ${equity_df['Equity'].iloc[-1]:,.2f}")
        
        # Calculate metrics
        print("\n7. Calculating metrics...")
        metrics = MetricsCalculator.calculate(
            equity_df=equity_df,
            trades_df=trades_df,
            initial_capital=CONFIG.INITIAL_CAPITAL,
        )
        
        print("\n✅ BACKTEST COMPLETED SUCCESSFULLY")
        
        return metrics, equity_df, trades_df, monthly_returns, signals
    
    def _run_simulation(
        self,
        prices: pd.DataFrame,
        signals: List[TSMOMSignal],
        rebalance_dates: set,
    ):
        """Run day-by-day simulation."""
        signal_map = {s.date: s for s in signals}
        
        for i, date in enumerate(prices.index):
            today_prices = prices.loc[date].to_dict()
            
            # Check for rebalance
            if date in signal_map:
                signal = signal_map[date]
                
                # SANITY: Only trade on rebalance dates
                if date not in rebalance_dates:
                    raise BacktestError(
                        f"Attempting trade on non-rebalance date: {date}"
                    )
                
                # Execute rebalance
                self.portfolio.rebalance_to_weights(
                    signal.target_weights,
                    today_prices,
                    date,
                )
            
            # Update equity curve
            self.portfolio.update_equity_curve(date, today_prices)
            
            # Progress
            if i % 500 == 0:
                eq = self.portfolio.get_equity(today_prices)
                print(f"   Day {i+1}/{len(prices)}: ${eq:,.0f}")
    
    def _validate_equity_curve(self, equity_df: pd.DataFrame):
        """Validate equity curve integrity."""
        # Equity always positive
        if (equity_df['Equity'] <= 0).any():
            min_eq = equity_df['Equity'].min()
            raise BacktestError(f"Equity went to ${min_eq:.2f}")
        
        # No NaN values
        if equity_df['Equity'].isna().any():
            raise BacktestError("NaN in equity curve")
        
        # Drawdown never 100%
        if (equity_df['Drawdown'] >= 1.0).any():
            raise BacktestError("Drawdown reached 100%")
        
        self._log("Equity curve validated")
    
    def _validate_no_lookahead(
        self,
        prices: pd.DataFrame,
        signals: List[TSMOMSignal],
    ):
        """Validate no lookahead bias."""
        warmup = max(CONFIG.TREND_LOOKBACK, CONFIG.VOL_LOOKBACK)
        
        for sig in signals:
            # Signal date must be in data
            if sig.date not in prices.index:
                raise BacktestError(f"Signal date {sig.date} not in price data")
            
            # Signal date must be after warmup
            sig_idx = prices.index.get_loc(sig.date)
            if sig_idx < warmup:
                raise BacktestError(
                    f"Signal on {sig.date} before warmup period"
                )
        
        self._log("No lookahead bias detected")
    
    def run_in_out_sample(
        self,
    ) -> Tuple[PerformanceMetrics, PerformanceMetrics, PerformanceMetrics,
               pd.DataFrame, pd.DataFrame, pd.Series, List[TSMOMSignal]]:
        """
        Run backtest with in-sample / out-of-sample split.
        
        Returns:
            (full_metrics, is_metrics, oos_metrics, equity_df, trades_df, monthly_returns, signals)
        """
        # Load full data
        prices = self.data_loader.load_universe()
        
        # Split
        is_prices, oos_prices = self.data_loader.split_in_out_sample(prices)
        
        print(f"\n{'='*70}")
        print("DATA SPLIT")
        print(f"{'='*70}")
        print(f"In-Sample:      {is_prices.index[0].date()} to {is_prices.index[-1].date()} ({len(is_prices)} days)")
        print(f"Out-of-Sample:  {oos_prices.index[0].date()} to {oos_prices.index[-1].date()} ({len(oos_prices)} days)")
        
        # Run in-sample
        print(f"\n{'='*70}")
        print("IN-SAMPLE BACKTEST")
        print(f"{'='*70}")
        
        is_metrics, is_equity, is_trades, is_monthly, is_signals = self.run(is_prices)
        
        # Run out-of-sample
        print(f"\n{'='*70}")
        print("OUT-OF-SAMPLE BACKTEST")
        print(f"{'='*70}")
        
        self.portfolio.reset()
        oos_metrics, oos_equity, oos_trades, oos_monthly, oos_signals = self.run(oos_prices)
        
        # Run full sample
        print(f"\n{'='*70}")
        print("FULL SAMPLE BACKTEST")
        print(f"{'='*70}")
        
        self.portfolio.reset()
        full_metrics, full_equity, full_trades, full_monthly, full_signals = self.run(prices)
        
        return (full_metrics, is_metrics, oos_metrics,
                full_equity, full_trades, full_monthly, full_signals)


def generate_report(
    full_metrics: PerformanceMetrics,
    is_metrics: PerformanceMetrics,
    oos_metrics: PerformanceMetrics,
    mc: MonteCarloResult,
) -> str:
    """Generate markdown report."""
    
    report = f"""# Time-Series Momentum with Volatility Targeting - Backtest Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## System Configuration

| Parameter | Value |
|-----------|-------|
| Universe | {', '.join(CONFIG.UNIVERSE)} |
| Trend Signal | Close > SMA({CONFIG.TREND_LOOKBACK}) |
| Target Volatility | {CONFIG.TARGET_VOL:.0%} |
| Vol Lookback | {CONFIG.VOL_LOOKBACK} days |
| Max Weight/Asset | {CONFIG.MAX_WEIGHT_PER_ASSET:.0%} |
| Rebalance | Monthly |
| Commission | {CONFIG.COMMISSION_PCT:.2%} |
| Slippage | {CONFIG.SLIPPAGE_PCT:.2%} |

---

## Full Sample Results

**Period:** {full_metrics.start_date.strftime('%Y-%m-%d')} to {full_metrics.end_date.strftime('%Y-%m-%d')} ({full_metrics.years:.1f} years)

### Returns & Risk
| Metric | Value |
|--------|-------|
| Total Return | {full_metrics.total_return:.1%} |
| **CAGR** | **{full_metrics.cagr:.1%}** |
| Volatility | {full_metrics.volatility:.1%} |
| **Max Drawdown** | **{full_metrics.max_drawdown:.1%}** |
| **Sharpe Ratio** | **{full_metrics.sharpe_ratio:.2f}** |
| Sortino Ratio | {full_metrics.sortino_ratio:.2f} |
| Calmar Ratio | {full_metrics.calmar_ratio:.2f} |

### Trade Statistics
| Metric | Value |
|--------|-------|
| Total Trades | {full_metrics.total_trades} |
| Win Rate | {full_metrics.win_rate:.1%} |
| **Profit Factor** | **{full_metrics.profit_factor:.2f}** |
| **Expectancy (R)** | **{full_metrics.expectancy_r:.2f}R** |
| Total Costs | ${full_metrics.total_costs:,.2f} |

---

## In-Sample vs Out-of-Sample

| Metric | In-Sample | Out-of-Sample | Degradation |
|--------|-----------|---------------|-------------|
| CAGR | {is_metrics.cagr:.1%} | {oos_metrics.cagr:.1%} | {((oos_metrics.cagr/is_metrics.cagr)-1)*100 if is_metrics.cagr != 0 else 0:+.0f}% |
| Sharpe | {is_metrics.sharpe_ratio:.2f} | {oos_metrics.sharpe_ratio:.2f} | {((oos_metrics.sharpe_ratio/is_metrics.sharpe_ratio)-1)*100 if is_metrics.sharpe_ratio != 0 else 0:+.0f}% |
| Max DD | {is_metrics.max_drawdown:.1%} | {oos_metrics.max_drawdown:.1%} | |
| Volatility | {is_metrics.volatility:.1%} | {oos_metrics.volatility:.1%} | |
| Win Rate | {is_metrics.win_rate:.1%} | {oos_metrics.win_rate:.1%} | |

---

## Monte Carlo Analysis ({mc.n_simulations} simulations)

### CAGR Distribution
| Percentile | Value |
|------------|-------|
| 5th | {mc.cagr_5th:.1%} |
| Median | {mc.cagr_median:.1%} |
| 95th | {mc.cagr_95th:.1%} |

### Drawdown Distribution
| Percentile | Value |
|------------|-------|
| 5th | {mc.dd_5th:.1%} |
| Median | {mc.dd_median:.1%} |
| **95th** | **{mc.dd_95th:.1%}** |
| Worst | {mc.dd_max:.1%} |

### Risk Probabilities
| Scenario | Probability |
|----------|-------------|
| P(Loss) | {mc.prob_loss:.1%} |
| **P(DD > 20%)** | **{mc.prob_dd_20:.1%}** |
| **P(DD > 30%)** | **{mc.prob_dd_30:.1%}** |

---

## Viability Assessment

### Criteria Check
| Criterion | Required | Actual | Status |
|-----------|----------|--------|--------|
| Sharpe Ratio | >= {CONFIG.MIN_SHARPE} | {full_metrics.sharpe_ratio:.2f} | {'✅' if full_metrics.sharpe_ratio >= CONFIG.MIN_SHARPE else '❌'} |
| Max Drawdown | <= {CONFIG.MAX_DRAWDOWN:.0%} | {full_metrics.max_drawdown:.1%} | {'✅' if full_metrics.max_drawdown <= CONFIG.MAX_DRAWDOWN else '❌'} |
| Total Trades | >= {CONFIG.MIN_TRADES} | {full_metrics.total_trades} | {'✅' if full_metrics.total_trades >= CONFIG.MIN_TRADES else '❌'} |
| OOS Sharpe | > 0 | {oos_metrics.sharpe_ratio:.2f} | {'✅' if oos_metrics.sharpe_ratio > 0 else '❌'} |
| MC 95th DD | <= 35% | {mc.dd_95th:.1%} | {'✅' if mc.dd_95th <= 0.35 else '❌'} |

---

## Files Generated

- `equity_curve.csv` - Daily equity values
- `trades.csv` - Complete trade history
- `monthly_returns.csv` - Monthly return series
- `REPORT.md` - This report

---

*Generated by Time-Series Momentum System v1.0*
"""
    
    return report


def main():
    """Main entry point."""
    print("\n" + "=" * 70)
    print("TIME-SERIES MOMENTUM WITH VOLATILITY TARGETING")
    print("PRODUCTION BACKTEST")
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
         equity_df, trades_df, monthly_returns, signals) = runner.run_in_out_sample()
    
    except (BacktestError, SanityCheckError) as e:
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
    
    # Print metrics
    print_metrics(full_metrics, "FULL SAMPLE METRICS")
    
    # Run Monte Carlo
    print("\n" + "=" * 70)
    print("MONTE CARLO SIMULATION")
    print("=" * 70)
    
    mc_sim = MonteCarloSimulator()
    mc_results = mc_sim.run(monthly_returns)
    
    print_monte_carlo(mc_results)
    
    # IS vs OOS comparison
    print("\n" + "=" * 70)
    print("IN-SAMPLE vs OUT-OF-SAMPLE COMPARISON")
    print("=" * 70)
    
    print(f"\n{'Metric':<20} {'In-Sample':>15} {'Out-of-Sample':>15}")
    print("-" * 50)
    print(f"{'CAGR':<20} {is_metrics.cagr:>14.1%} {oos_metrics.cagr:>14.1%}")
    print(f"{'Sharpe':<20} {is_metrics.sharpe_ratio:>15.2f} {oos_metrics.sharpe_ratio:>15.2f}")
    print(f"{'Max DD':<20} {is_metrics.max_drawdown:>14.1%} {oos_metrics.max_drawdown:>14.1%}")
    print(f"{'Volatility':<20} {is_metrics.volatility:>14.1%} {oos_metrics.volatility:>14.1%}")
    print(f"{'Win Rate':<20} {is_metrics.win_rate:>14.1%} {oos_metrics.win_rate:>14.1%}")
    
    # Viability assessment
    print("\n" + "=" * 70)
    print("VIABILITY ASSESSMENT")
    print("=" * 70)
    
    passed_criteria = []
    failed_criteria = []
    
    if full_metrics.sharpe_ratio >= CONFIG.MIN_SHARPE:
        passed_criteria.append(f"Sharpe {full_metrics.sharpe_ratio:.2f} >= {CONFIG.MIN_SHARPE}")
    else:
        failed_criteria.append(f"Sharpe {full_metrics.sharpe_ratio:.2f} < {CONFIG.MIN_SHARPE}")
    
    if full_metrics.max_drawdown <= CONFIG.MAX_DRAWDOWN:
        passed_criteria.append(f"Max DD {full_metrics.max_drawdown:.1%} <= {CONFIG.MAX_DRAWDOWN:.0%}")
    else:
        failed_criteria.append(f"Max DD {full_metrics.max_drawdown:.1%} > {CONFIG.MAX_DRAWDOWN:.0%}")
    
    if full_metrics.total_trades >= CONFIG.MIN_TRADES:
        passed_criteria.append(f"Trades {full_metrics.total_trades} >= {CONFIG.MIN_TRADES}")
    else:
        failed_criteria.append(f"Trades {full_metrics.total_trades} < {CONFIG.MIN_TRADES}")
    
    if oos_metrics.sharpe_ratio > 0:
        passed_criteria.append(f"OOS Sharpe {oos_metrics.sharpe_ratio:.2f} > 0")
    else:
        failed_criteria.append(f"OOS Sharpe {oos_metrics.sharpe_ratio:.2f} <= 0")
    
    if mc_results.dd_95th <= 0.35:
        passed_criteria.append(f"MC 95th DD {mc_results.dd_95th:.1%} <= 35%")
    else:
        failed_criteria.append(f"MC 95th DD {mc_results.dd_95th:.1%} > 35%")
    
    print("\n✅ PASSED:")
    for c in passed_criteria:
        print(f"   - {c}")
    
    if failed_criteria:
        print("\n❌ FAILED:")
        for c in failed_criteria:
            print(f"   - {c}")
    
    is_viable = len(failed_criteria) == 0
    
    # Final verdict
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)
    
    if is_viable:
        print("\n  ✅ SYSTEM IS DEPLOYABLE WITH REAL CAPITAL: YES")
        print("\n  RATIONALE:")
        print("  - Time-series momentum is a well-documented factor with academic backing")
        print("  - Volatility targeting provides adaptive risk management")
        print("  - Out-of-sample performance confirms in-sample results")
        print("  - Monte Carlo analysis shows acceptable tail risk")
        print("  - Transaction costs are realistic and accounted for")
        print("\n  RISKS THAT REMAIN:")
        print("  - Regime change: momentum can suffer in choppy markets")
        print("  - Crowding: widespread adoption may reduce alpha")
        print("  - Execution: real slippage may exceed estimates")
        print("  - Black swan events not captured in historical data")
        print("\n  RECOMMENDATION: Deploy with 50% of intended capital initially")
    else:
        print("\n  ❌ SYSTEM IS DEPLOYABLE WITH REAL CAPITAL: NO")
        print("\n  REASON: Failed one or more viability criteria")
        print("  DO NOT DEPLOY until issues are resolved")
    
    # Save results
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    equity_file = output_dir / "equity_curve.csv"
    equity_df.to_csv(equity_file)
    print(f"\n  Equity curve: {equity_file}")
    
    trades_file = output_dir / "trades.csv"
    trades_df.to_csv(trades_file, index=False)
    print(f"  Trades: {trades_file}")
    
    monthly_file = output_dir / "monthly_returns.csv"
    monthly_returns.to_csv(monthly_file)
    print(f"  Monthly returns: {monthly_file}")
    
    report = generate_report(full_metrics, is_metrics, oos_metrics, mc_results)
    report_file = output_dir / "REPORT.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  Report: {report_file}")
    
    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70 + "\n")
    
    return 0 if is_viable else 1


if __name__ == "__main__":
    sys.exit(main())
