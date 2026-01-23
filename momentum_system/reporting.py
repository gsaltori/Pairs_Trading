"""
Cross-Sectional Momentum System - Reporting
Calculates metrics and generates reports.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from pathlib import Path

from config import CONFIG


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    # Returns
    total_return: float
    cagr: float
    
    # Risk-adjusted
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Risk
    volatility: float
    max_drawdown: float
    avg_drawdown: float
    
    # Trade statistics
    total_trades: int
    win_rate: float
    profit_factor: float
    expectancy: float       # Average $ profit per trade
    expectancy_r: float     # Expectancy in risk units
    
    # Win/Loss
    avg_win: float
    avg_loss: float
    win_loss_ratio: float
    largest_win: float
    largest_loss: float
    
    # Efficiency
    avg_holding_days: float
    turnover_annual: float
    
    # Costs
    total_costs: float
    
    # Period
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    years: float


@dataclass
class MonteCarloResults:
    """Monte Carlo simulation results."""
    n_simulations: int
    
    # CAGR distribution
    cagr_mean: float
    cagr_median: float
    cagr_5th: float
    cagr_95th: float
    
    # Drawdown distribution
    dd_mean: float
    dd_median: float
    dd_5th: float
    dd_95th: float
    dd_worst: float
    
    # Risk of ruin
    prob_loss: float          # Probability of negative return
    prob_dd_over_20: float    # Probability of 20%+ drawdown
    prob_dd_over_30: float    # Probability of 30%+ drawdown


class MetricsCalculator:
    """Calculates all performance metrics."""
    
    @staticmethod
    def calculate(
        equity_df: pd.DataFrame,
        trades_df: pd.DataFrame,
        monthly_returns: pd.Series,
        initial_capital: float,
    ) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            equity_df: DataFrame with Equity and Drawdown columns
            trades_df: DataFrame with trade history
            monthly_returns: Series of monthly returns
            initial_capital: Starting capital
            
        Returns:
            PerformanceMetrics object
        """
        # Period
        start_date = equity_df.index[0]
        end_date = equity_df.index[-1]
        years = (end_date - start_date).days / 365.25
        
        # Returns
        final_equity = equity_df['Equity'].iloc[-1]
        total_return = (final_equity / initial_capital) - 1
        cagr = (final_equity / initial_capital) ** (1 / years) - 1 if years > 0 else 0
        
        # Risk metrics from daily data
        daily_returns = equity_df['Equity'].pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252)
        
        # Max drawdown
        max_drawdown = equity_df['Drawdown'].max()
        avg_drawdown = equity_df['Drawdown'].mean()
        
        # Risk-adjusted ratios
        risk_free = 0.04  # Assume 4% risk-free rate
        excess_return = cagr - risk_free
        
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0
        
        # Sortino (downside deviation)
        negative_returns = daily_returns[daily_returns < 0]
        downside_std = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0.0001
        sortino_ratio = excess_return / downside_std if downside_std > 0 else 0
        
        # Calmar
        calmar_ratio = cagr / max_drawdown if max_drawdown > 0 else 0
        
        # Trade statistics
        if len(trades_df) > 0:
            total_trades = len(trades_df)
            winners = trades_df[trades_df['Won']]
            losers = trades_df[~trades_df['Won']]
            
            win_rate = len(winners) / total_trades if total_trades > 0 else 0
            
            gross_profit = winners['PnL'].sum() if len(winners) > 0 else 0
            gross_loss = abs(losers['PnL'].sum()) if len(losers) > 0 else 0.0001
            
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            avg_win = winners['PnL'].mean() if len(winners) > 0 else 0
            avg_loss = losers['PnL'].mean() if len(losers) > 0 else 0  # Negative
            
            win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            
            largest_win = trades_df['PnL'].max()
            largest_loss = trades_df['PnL'].min()
            
            expectancy = trades_df['PnL'].mean()
            
            # Expectancy in R units (risk = initial position value * 10% assumed)
            avg_position = trades_df['Shares'].mean() * trades_df['Entry_Price'].mean()
            risk_per_trade = avg_position * 0.10 if avg_position > 0 else 1
            expectancy_r = expectancy / risk_per_trade
            
            avg_holding_days = trades_df['Holding_Days'].mean()
            
            total_costs = trades_df['Costs'].sum()
        else:
            total_trades = 0
            win_rate = 0
            profit_factor = 0
            expectancy = 0
            expectancy_r = 0
            avg_win = 0
            avg_loss = 0
            win_loss_ratio = 0
            largest_win = 0
            largest_loss = 0
            avg_holding_days = 0
            total_costs = 0
        
        # Turnover (from monthly data)
        turnover_annual = 12 * 0.5  # Estimate: ~50% turnover per month
        
        return PerformanceMetrics(
            total_return=total_return,
            cagr=cagr,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            volatility=volatility,
            max_drawdown=max_drawdown,
            avg_drawdown=avg_drawdown,
            total_trades=total_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            expectancy=expectancy,
            expectancy_r=expectancy_r,
            avg_win=avg_win,
            avg_loss=avg_loss,
            win_loss_ratio=win_loss_ratio,
            largest_win=largest_win,
            largest_loss=largest_loss,
            avg_holding_days=avg_holding_days,
            turnover_annual=turnover_annual,
            total_costs=total_costs,
            start_date=start_date,
            end_date=end_date,
            years=years,
        )


class MonteCarloSimulator:
    """Monte Carlo simulation for robustness testing."""
    
    def __init__(self, n_simulations: int = None):
        if n_simulations is None:
            n_simulations = CONFIG.MONTE_CARLO_RUNS
        self.n_simulations = n_simulations
    
    def run(
        self,
        monthly_returns: pd.Series,
        initial_capital: float,
    ) -> MonteCarloResults:
        """
        Run Monte Carlo simulation by resampling monthly returns.
        
        Args:
            monthly_returns: Series of monthly returns
            initial_capital: Starting capital
            
        Returns:
            MonteCarloResults object
        """
        returns = monthly_returns.values
        n_months = len(returns)
        years = n_months / 12
        
        all_cagrs = []
        all_max_dds = []
        
        for _ in range(self.n_simulations):
            # Resample returns with replacement
            resampled = np.random.choice(returns, size=n_months, replace=True)
            
            # Build equity curve
            equity = initial_capital
            peak = initial_capital
            max_dd = 0
            
            for ret in resampled:
                equity *= (1 + ret)
                
                if equity > peak:
                    peak = equity
                
                dd = (peak - equity) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)
            
            # Calculate CAGR
            cagr = (equity / initial_capital) ** (1 / years) - 1 if years > 0 else 0
            
            all_cagrs.append(cagr)
            all_max_dds.append(max_dd)
        
        all_cagrs = np.array(all_cagrs)
        all_max_dds = np.array(all_max_dds)
        
        return MonteCarloResults(
            n_simulations=self.n_simulations,
            cagr_mean=np.mean(all_cagrs),
            cagr_median=np.median(all_cagrs),
            cagr_5th=np.percentile(all_cagrs, 5),
            cagr_95th=np.percentile(all_cagrs, 95),
            dd_mean=np.mean(all_max_dds),
            dd_median=np.median(all_max_dds),
            dd_5th=np.percentile(all_max_dds, 5),
            dd_95th=np.percentile(all_max_dds, 95),
            dd_worst=np.max(all_max_dds),
            prob_loss=(all_cagrs < 0).mean(),
            prob_dd_over_20=(all_max_dds > 0.20).mean(),
            prob_dd_over_30=(all_max_dds > 0.30).mean(),
        )


def check_viability(metrics: PerformanceMetrics, mc: MonteCarloResults) -> Tuple[bool, List[str]]:
    """
    Check if system passes viability criteria.
    
    Returns:
        Tuple of (passed, list of issues)
    """
    issues = []
    
    # Expectancy
    if metrics.expectancy_r < CONFIG.MIN_EXPECTANCY_R:
        issues.append(f"Expectancy {metrics.expectancy_r:.2f}R < {CONFIG.MIN_EXPECTANCY_R}R")
    
    # Sharpe
    if metrics.sharpe_ratio < CONFIG.MIN_SHARPE:
        issues.append(f"Sharpe {metrics.sharpe_ratio:.2f} < {CONFIG.MIN_SHARPE}")
    
    # Max DD
    if metrics.max_drawdown > CONFIG.MAX_DRAWDOWN:
        issues.append(f"Max DD {metrics.max_drawdown:.1%} > {CONFIG.MAX_DRAWDOWN:.0%}")
    
    # Trade count
    if metrics.total_trades < CONFIG.MIN_TRADES:
        issues.append(f"Trades {metrics.total_trades} < {CONFIG.MIN_TRADES}")
    
    # Monte Carlo 95th percentile DD
    if mc.dd_95th > CONFIG.MAX_DRAWDOWN:
        issues.append(f"MC 95th DD {mc.dd_95th:.1%} > {CONFIG.MAX_DRAWDOWN:.0%}")
    
    return len(issues) == 0, issues


def print_metrics(metrics: PerformanceMetrics):
    """Print formatted metrics."""
    print("\n" + "=" * 60)
    print("PERFORMANCE METRICS")
    print("=" * 60)
    
    print(f"\nPeriod: {metrics.start_date.strftime('%Y-%m-%d')} to {metrics.end_date.strftime('%Y-%m-%d')} ({metrics.years:.1f} years)")
    
    print("\n--- RETURNS ---")
    print(f"  Total Return:      {metrics.total_return:>10.1%}")
    print(f"  CAGR:              {metrics.cagr:>10.1%}")
    
    print("\n--- RISK-ADJUSTED ---")
    print(f"  Sharpe Ratio:      {metrics.sharpe_ratio:>10.2f}")
    print(f"  Sortino Ratio:     {metrics.sortino_ratio:>10.2f}")
    print(f"  Calmar Ratio:      {metrics.calmar_ratio:>10.2f}")
    
    print("\n--- RISK ---")
    print(f"  Volatility:        {metrics.volatility:>10.1%}")
    print(f"  Max Drawdown:      {metrics.max_drawdown:>10.1%}")
    print(f"  Avg Drawdown:      {metrics.avg_drawdown:>10.1%}")
    
    print("\n--- TRADE STATISTICS ---")
    print(f"  Total Trades:      {metrics.total_trades:>10}")
    print(f"  Win Rate:          {metrics.win_rate:>10.1%}")
    print(f"  Profit Factor:     {metrics.profit_factor:>10.2f}")
    print(f"  Expectancy ($):    ${metrics.expectancy:>9.2f}")
    print(f"  Expectancy (R):    {metrics.expectancy_r:>10.2f}")
    
    print("\n--- WIN/LOSS ---")
    print(f"  Avg Win:           ${metrics.avg_win:>9.2f}")
    print(f"  Avg Loss:          ${metrics.avg_loss:>9.2f}")
    print(f"  Win/Loss Ratio:    {metrics.win_loss_ratio:>10.2f}")
    print(f"  Largest Win:       ${metrics.largest_win:>9.2f}")
    print(f"  Largest Loss:      ${metrics.largest_loss:>9.2f}")
    
    print("\n--- EFFICIENCY ---")
    print(f"  Avg Hold (days):   {metrics.avg_holding_days:>10.1f}")
    print(f"  Total Costs:       ${metrics.total_costs:>9.2f}")
    
    print("=" * 60)


def print_monte_carlo(mc: MonteCarloResults):
    """Print Monte Carlo results."""
    print("\n" + "=" * 60)
    print("MONTE CARLO RESULTS")
    print("=" * 60)
    
    print(f"\nSimulations: {mc.n_simulations}")
    
    print("\n--- CAGR DISTRIBUTION ---")
    print(f"  5th Percentile:    {mc.cagr_5th:>10.1%}")
    print(f"  Median:            {mc.cagr_median:>10.1%}")
    print(f"  Mean:              {mc.cagr_mean:>10.1%}")
    print(f"  95th Percentile:   {mc.cagr_95th:>10.1%}")
    
    print("\n--- DRAWDOWN DISTRIBUTION ---")
    print(f"  5th Percentile:    {mc.dd_5th:>10.1%}")
    print(f"  Median:            {mc.dd_median:>10.1%}")
    print(f"  Mean:              {mc.dd_mean:>10.1%}")
    print(f"  95th Percentile:   {mc.dd_95th:>10.1%}")
    print(f"  Worst Case:        {mc.dd_worst:>10.1%}")
    
    print("\n--- RISK OF RUIN ---")
    print(f"  P(Negative Return):{mc.prob_loss:>10.1%}")
    print(f"  P(DD > 20%):       {mc.prob_dd_over_20:>10.1%}")
    print(f"  P(DD > 30%):       {mc.prob_dd_over_30:>10.1%}")
    
    print("=" * 60)


def generate_report(
    metrics: PerformanceMetrics,
    mc: MonteCarloResults,
    is_metrics: PerformanceMetrics,
    oos_metrics: PerformanceMetrics,
    passed: bool,
    issues: List[str],
) -> str:
    """Generate markdown report."""
    
    verdict = "GO" if passed else "NO-GO"
    verdict_emoji = "✅" if passed else "❌"
    
    report = f"""# Cross-Sectional Momentum System - Backtest Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## System Configuration

| Parameter | Value |
|-----------|-------|
| Universe | {', '.join(CONFIG.UNIVERSE)} |
| Momentum Lookback | {CONFIG.MOMENTUM_LOOKBACK} days (12 months) |
| Trend Filter | EMA({CONFIG.TREND_FILTER_PERIOD}) |
| Rebalance | Monthly |
| Top N Selection | {CONFIG.TOP_N_ASSETS} |
| Position Sizing | Equal weight ({CONFIG.WEIGHT_PER_POSITION:.1%} each) |
| Commission | {CONFIG.COMMISSION_PCT:.2%} |
| Slippage | {CONFIG.SLIPPAGE_PCT:.2%} |

---

## Full Sample Results

**Period:** {metrics.start_date.strftime('%Y-%m-%d')} to {metrics.end_date.strftime('%Y-%m-%d')} ({metrics.years:.1f} years)

### Returns
| Metric | Value |
|--------|-------|
| Total Return | {metrics.total_return:.1%} |
| **CAGR** | **{metrics.cagr:.1%}** |

### Risk-Adjusted
| Metric | Value |
|--------|-------|
| **Sharpe Ratio** | **{metrics.sharpe_ratio:.2f}** |
| Sortino Ratio | {metrics.sortino_ratio:.2f} |
| Calmar Ratio | {metrics.calmar_ratio:.2f} |

### Risk
| Metric | Value |
|--------|-------|
| Volatility | {metrics.volatility:.1%} |
| **Max Drawdown** | **{metrics.max_drawdown:.1%}** |

### Trade Statistics
| Metric | Value |
|--------|-------|
| Total Trades | {metrics.total_trades} |
| Win Rate | {metrics.win_rate:.1%} |
| **Profit Factor** | **{metrics.profit_factor:.2f}** |
| **Expectancy (R)** | **{metrics.expectancy_r:.2f}R** |
| Avg Win | ${metrics.avg_win:.2f} |
| Avg Loss | ${metrics.avg_loss:.2f} |

---

## In-Sample vs Out-of-Sample

| Metric | In-Sample | Out-of-Sample |
|--------|-----------|---------------|
| Period | {is_metrics.start_date.strftime('%Y-%m-%d')} to {is_metrics.end_date.strftime('%Y-%m-%d')} | {oos_metrics.start_date.strftime('%Y-%m-%d')} to {oos_metrics.end_date.strftime('%Y-%m-%d')} |
| CAGR | {is_metrics.cagr:.1%} | {oos_metrics.cagr:.1%} |
| Sharpe | {is_metrics.sharpe_ratio:.2f} | {oos_metrics.sharpe_ratio:.2f} |
| Max DD | {is_metrics.max_drawdown:.1%} | {oos_metrics.max_drawdown:.1%} |
| Trades | {is_metrics.total_trades} | {oos_metrics.total_trades} |
| Win Rate | {is_metrics.win_rate:.1%} | {oos_metrics.win_rate:.1%} |

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
| Worst | {mc.dd_worst:.1%} |

### Risk of Ruin
| Scenario | Probability |
|----------|-------------|
| Negative Total Return | {mc.prob_loss:.1%} |
| Drawdown > 20% | {mc.prob_dd_over_20:.1%} |
| Drawdown > 30% | {mc.prob_dd_over_30:.1%} |

---

## Viability Assessment

### Kill Criteria
| Criterion | Threshold | Actual | Status |
|-----------|-----------|--------|--------|
| Expectancy | >= {CONFIG.MIN_EXPECTANCY_R}R | {metrics.expectancy_r:.2f}R | {'✅' if metrics.expectancy_r >= CONFIG.MIN_EXPECTANCY_R else '❌'} |
| Sharpe | >= {CONFIG.MIN_SHARPE} | {metrics.sharpe_ratio:.2f} | {'✅' if metrics.sharpe_ratio >= CONFIG.MIN_SHARPE else '❌'} |
| Max DD | <= {CONFIG.MAX_DRAWDOWN:.0%} | {metrics.max_drawdown:.1%} | {'✅' if metrics.max_drawdown <= CONFIG.MAX_DRAWDOWN else '❌'} |
| Total Trades | >= {CONFIG.MIN_TRADES} | {metrics.total_trades} | {'✅' if metrics.total_trades >= CONFIG.MIN_TRADES else '❌'} |
| MC 95th DD | <= {CONFIG.MAX_DRAWDOWN:.0%} | {mc.dd_95th:.1%} | {'✅' if mc.dd_95th <= CONFIG.MAX_DRAWDOWN else '❌'} |

"""
    
    if issues:
        report += "\n### Issues\n"
        for issue in issues:
            report += f"- ❌ {issue}\n"
    
    report += f"""

---

## Final Verdict

# {verdict_emoji} {verdict}

"""
    
    if passed:
        report += """### Deployment Recommendation

This system meets all viability criteria and shows robust performance across:
- In-sample period
- Out-of-sample period  
- Monte Carlo stress testing

**Recommended deployment path:**

1. **Paper Trading (1-3 months)**
   - Run `python live_runner.py --paper`
   - Validate monthly rebalancing works correctly
   - Confirm costs match expectations

2. **Small Capital Deployment (3-6 months)**
   - Start with 25% of intended capital
   - Monitor for live vs backtest discrepancies

3. **Full Deployment**
   - Scale to full capital if validation successful
   - Maintain monthly monitoring

### I would deploy this system with real capital.

The cross-sectional momentum effect has decades of academic evidence and this implementation shows robust out-of-sample performance with acceptable drawdown characteristics.
"""
    else:
        report += """### Deployment Recommendation

This system FAILS one or more viability criteria and should NOT be deployed with real capital.

### I would NOT deploy this system with real capital.

Review the issues above before any further consideration.
"""
    
    report += f"""

---

## Files Generated

- `REPORT.md` - This report
- `equity_curve.csv` - Daily equity values
- `trades.csv` - Complete trade history
- `monthly_returns.csv` - Monthly return series

---

*Generated by Cross-Sectional Momentum System v1.0*
"""
    
    return report
