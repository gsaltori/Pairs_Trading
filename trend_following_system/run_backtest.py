"""
Trend Following System - Backtest Runner
One-shot historical backtest with full validation.

Usage:
    python run_backtest.py

Output:
    - Console metrics
    - RESULTS.md report
    - Equity curve CSV
    - Trade history CSV
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

from config import TRADING_CONFIG, print_config
from backtest_engine import BacktestEngine, BacktestMetrics, print_metrics
from monte_carlo import MonteCarloEngine, print_monte_carlo_results, check_monte_carlo_viability
from logger import setup_logger


def check_viability(metrics: BacktestMetrics) -> tuple:
    """
    Check if backtest results pass viability criteria.
    
    Returns:
        Tuple of (passed: bool, issues: list)
    """
    issues = []
    
    # Expectancy check
    if metrics.expectancy_r < TRADING_CONFIG.MIN_EXPECTANCY_R:
        issues.append(
            f"Expectancy ({metrics.expectancy_r:.2f}R) < "
            f"minimum ({TRADING_CONFIG.MIN_EXPECTANCY_R}R)"
        )
    
    # Profit factor check
    if metrics.profit_factor < TRADING_CONFIG.MIN_PROFIT_FACTOR:
        issues.append(
            f"Profit Factor ({metrics.profit_factor:.2f}) < "
            f"minimum ({TRADING_CONFIG.MIN_PROFIT_FACTOR})"
        )
    
    # Max drawdown check
    if metrics.max_drawdown > TRADING_CONFIG.MAX_ALLOWED_DD:
        issues.append(
            f"Max Drawdown ({metrics.max_drawdown:.1%}) > "
            f"limit ({TRADING_CONFIG.MAX_ALLOWED_DD:.0%})"
        )
    
    # Trade count check
    if metrics.total_trades < TRADING_CONFIG.MIN_TOTAL_TRADES:
        issues.append(
            f"Total Trades ({metrics.total_trades}) < "
            f"minimum ({TRADING_CONFIG.MIN_TOTAL_TRADES})"
        )
    
    return len(issues) == 0, issues


def generate_results_report(
    metrics: BacktestMetrics,
    mc_results,
    viability_passed: bool,
    viability_issues: list,
    mc_passed: bool,
    mc_issues: list,
    output_dir: Path,
) -> str:
    """Generate RESULTS.md report."""
    
    verdict = "✅ GO" if (viability_passed and mc_passed) else "❌ NO-GO"
    
    report = f"""# Trend Following System - Backtest Results

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## System Configuration

| Parameter | Value |
|-----------|-------|
| Universe | {', '.join(TRADING_CONFIG.SYMBOLS)} |
| Timeframe | Daily (D1) |
| Initial Capital | ${TRADING_CONFIG.INITIAL_CAPITAL:,.2f} |
| Risk per Trade | {TRADING_CONFIG.RISK_PER_TRADE:.1%} |
| Max Positions | {TRADING_CONFIG.MAX_POSITIONS} |

### Entry Rules
1. Close > EMA({TRADING_CONFIG.EMA_PERIOD})
2. Close > Highest High of last {TRADING_CONFIG.DONCHIAN_ENTRY} days
3. ATR({TRADING_CONFIG.ATR_PERIOD}) >= Median ATR of last {TRADING_CONFIG.ATR_LOOKBACK} days

### Exit Rules
1. Trailing stop = Lowest Low of last {TRADING_CONFIG.DONCHIAN_EXIT} days
2. No fixed take profit

---

## Backtest Summary

**Period:** {metrics.start_date.strftime('%Y-%m-%d')} to {metrics.end_date.strftime('%Y-%m-%d')} ({metrics.years:.1f} years)

### Returns
| Metric | Value |
|--------|-------|
| Total Return | {metrics.total_return:.1%} |
| CAGR | {metrics.cagr:.1%} |
| Sharpe Ratio | {metrics.sharpe_ratio:.2f} |
| Sortino Ratio | {metrics.sortino_ratio:.2f} |
| Calmar Ratio | {metrics.calmar_ratio:.2f} |

### Risk
| Metric | Value |
|--------|-------|
| Max Drawdown | {metrics.max_drawdown:.1%} |
| Volatility | {metrics.volatility:.1%} |

### Trade Statistics
| Metric | Value |
|--------|-------|
| Total Trades | {metrics.total_trades} |
| Win Rate | {metrics.win_rate:.1%} |
| **Profit Factor** | **{metrics.profit_factor:.2f}** |
| **Expectancy (R)** | **{metrics.expectancy_r:.2f}R** |
| Avg Win | ${metrics.avg_win:.2f} ({metrics.avg_win_r:.2f}R) |
| Avg Loss | ${metrics.avg_loss:.2f} ({metrics.avg_loss_r:.2f}R) |
| Largest Win | ${metrics.largest_win:.2f} |
| Largest Loss | ${metrics.largest_loss:.2f} |

### Efficiency
| Metric | Value |
|--------|-------|
| Trades per Year | {metrics.trades_per_year:.1f} |
| Avg Holding Days | {metrics.avg_holding_days:.1f} |
| Exposure | {metrics.exposure_pct:.1%} |
| Total Costs | ${metrics.total_costs:.2f} |

---

## Monte Carlo Analysis

**Simulations:** {mc_results.n_simulations}

### Drawdown Distribution
| Percentile | Max Drawdown |
|------------|--------------|
| 5th | {mc_results.dd_5th_percentile:.1%} |
| Median | {mc_results.median_max_dd:.1%} |
| 95th | {mc_results.dd_95th_percentile:.1%} |
| Worst | {mc_results.dd_worst:.1%} |

### Return Distribution
| Percentile | CAGR |
|------------|------|
| 5th | {mc_results.cagr_5th_percentile:.1%} |
| Median | {mc_results.median_cagr:.1%} |
| 95th | {mc_results.cagr_95th_percentile:.1%} |

### Risk of Ruin
| Threshold | Probability |
|-----------|-------------|
| DD > 20% | {mc_results.prob_ruin_20pct:.1%} |
| DD > 30% | {mc_results.prob_ruin_30pct:.1%} |
| DD > 50% | {mc_results.prob_ruin_50pct:.1%} |

---

## Viability Assessment

### Kill Criteria
| Criterion | Threshold | Actual | Status |
|-----------|-----------|--------|--------|
| Expectancy | ≥ {TRADING_CONFIG.MIN_EXPECTANCY_R}R | {metrics.expectancy_r:.2f}R | {'✅' if metrics.expectancy_r >= TRADING_CONFIG.MIN_EXPECTANCY_R else '❌'} |
| Profit Factor | ≥ {TRADING_CONFIG.MIN_PROFIT_FACTOR} | {metrics.profit_factor:.2f} | {'✅' if metrics.profit_factor >= TRADING_CONFIG.MIN_PROFIT_FACTOR else '❌'} |
| Max Drawdown | ≤ {TRADING_CONFIG.MAX_ALLOWED_DD:.0%} | {metrics.max_drawdown:.1%} | {'✅' if metrics.max_drawdown <= TRADING_CONFIG.MAX_ALLOWED_DD else '❌'} |
| Total Trades | ≥ {TRADING_CONFIG.MIN_TOTAL_TRADES} | {metrics.total_trades} | {'✅' if metrics.total_trades >= TRADING_CONFIG.MIN_TOTAL_TRADES else '❌'} |
| MC 95th DD | ≤ {TRADING_CONFIG.MONTE_CARLO_MAX_DD:.0%} | {mc_results.dd_95th_percentile:.1%} | {'✅' if mc_results.dd_95th_percentile <= TRADING_CONFIG.MONTE_CARLO_MAX_DD else '❌'} |

"""
    
    if viability_issues or mc_issues:
        report += "### Issues Found\n\n"
        for issue in viability_issues + mc_issues:
            report += f"- ❌ {issue}\n"
        report += "\n"
    
    report += f"""---

## Final Verdict

# {verdict}

"""
    
    if viability_passed and mc_passed:
        report += """### Recommended Next Steps

1. **Paper Trading Phase** (2-4 weeks)
   - Run `python run_live.py --paper`
   - Validate signal generation matches backtest
   - Confirm execution timing

2. **Micro Live Phase** (1-3 months)
   - Start with 10-25% of intended capital
   - Monitor for live vs backtest discrepancies
   - Document any issues

3. **Full Live Phase**
   - Scale to full capital if micro phase successful
   - Maintain daily monitoring
   - Monthly performance review

### Operating Procedures

- Evaluate signals daily at market close
- Execute orders at next day's open
- Update trailing stops daily
- Review performance weekly
"""
    else:
        report += """### System Rejected

The system does not meet viability criteria and should NOT be deployed with real capital.

Review the issues listed above and consider:
- Parameter adjustments (with fresh validation)
- Universe changes
- Alternative strategies
"""
    
    report += f"""
---

## Files Generated

- `RESULTS.md` - This report
- `equity_curve.csv` - Daily equity values
- `trades.csv` - Complete trade history

---

*Generated by Trend Following System v1.0*
"""
    
    return report


def main():
    """Run full backtest with validation."""
    print("\n" + "=" * 70)
    print("TREND FOLLOWING SYSTEM - BACKTEST")
    print("=" * 70)
    
    # Setup
    logger = setup_logger()
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    # Print configuration
    print_config()
    
    # Run backtest
    print("\n" + "=" * 70)
    engine = BacktestEngine()
    
    try:
        metrics = engine.run()
    except Exception as e:
        logger.log_error(e, "Backtest execution")
        print(f"\n❌ Backtest failed: {e}")
        return 1
    
    # Print metrics
    print_metrics(metrics)
    
    # Run Monte Carlo
    print("\n" + "=" * 70)
    print("MONTE CARLO VALIDATION")
    print("=" * 70)
    
    trades_df = engine.get_trades()
    mc_engine = MonteCarloEngine()
    
    try:
        mc_results = mc_engine.run(
            trades_df=trades_df,
            initial_capital=TRADING_CONFIG.INITIAL_CAPITAL,
            years=metrics.years,
        )
    except Exception as e:
        logger.log_error(e, "Monte Carlo simulation")
        print(f"\n❌ Monte Carlo failed: {e}")
        return 1
    
    print_monte_carlo_results(mc_results)
    
    # Check viability
    print("\n" + "=" * 70)
    print("VIABILITY ASSESSMENT")
    print("=" * 70)
    
    viability_passed, viability_issues = check_viability(metrics)
    mc_passed, mc_issues = check_monte_carlo_viability(mc_results)
    
    print("\nBacktest Criteria:")
    if viability_passed:
        print("  ✅ All criteria PASSED")
    else:
        print("  ❌ Some criteria FAILED:")
        for issue in viability_issues:
            print(f"     - {issue}")
    
    print("\nMonte Carlo Criteria:")
    if mc_passed:
        print("  ✅ All criteria PASSED")
    else:
        print("  ❌ Some criteria FAILED:")
        for issue in mc_issues:
            print(f"     - {issue}")
    
    # Final verdict
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)
    
    if viability_passed and mc_passed:
        print("\n  ✅ GO - System passes all viability criteria")
        print("\n  Proceed to paper trading phase.")
    else:
        print("\n  ❌ NO-GO - System fails viability criteria")
        print("\n  DO NOT deploy with real capital.")
    
    # Save results
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    # Equity curve
    equity_df = engine.get_equity_curve()
    equity_file = output_dir / "equity_curve.csv"
    equity_df.to_csv(equity_file)
    print(f"\n  Equity curve saved to: {equity_file}")
    
    # Trades
    trades_file = output_dir / "trades.csv"
    trades_df.to_csv(trades_file, index=False)
    print(f"  Trade history saved to: {trades_file}")
    
    # Report
    report = generate_results_report(
        metrics=metrics,
        mc_results=mc_results,
        viability_passed=viability_passed,
        viability_issues=viability_issues,
        mc_passed=mc_passed,
        mc_issues=mc_issues,
        output_dir=output_dir,
    )
    
    results_file = output_dir / "RESULTS.md"
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  Full report saved to: {results_file}")
    
    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70 + "\n")
    
    return 0 if (viability_passed and mc_passed) else 1


if __name__ == "__main__":
    sys.exit(main())
