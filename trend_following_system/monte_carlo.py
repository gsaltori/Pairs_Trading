"""
Trend Following System - Monte Carlo Validation
Robustness testing via equity curve resampling.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor
import warnings

from config import TRADING_CONFIG


@dataclass
class MonteCarloResults:
    """Monte Carlo simulation results."""
    n_simulations: int
    
    # Drawdown statistics
    median_max_dd: float
    mean_max_dd: float
    dd_5th_percentile: float
    dd_95th_percentile: float
    dd_worst: float
    
    # Return statistics
    median_cagr: float
    mean_cagr: float
    cagr_5th_percentile: float
    cagr_95th_percentile: float
    
    # Risk of ruin
    prob_ruin_20pct: float    # Probability of 20% drawdown
    prob_ruin_30pct: float    # Probability of 30% drawdown
    prob_ruin_50pct: float    # Probability of 50% drawdown
    
    # Confidence intervals
    cagr_ci_lower: float
    cagr_ci_upper: float
    dd_ci_lower: float
    dd_ci_upper: float
    
    # All simulation results (for plotting)
    all_max_dds: np.ndarray
    all_cagrs: np.ndarray


class MonteCarloEngine:
    """
    Monte Carlo simulation for strategy robustness.
    
    Method:
    - Resample trade returns with replacement
    - Reconstruct equity curves
    - Measure drawdown distribution
    - Estimate probability of ruin
    
    This tests whether the strategy's performance is robust
    to different orderings of the same trades.
    """
    
    def __init__(
        self,
        n_simulations: int = None,
        confidence_level: float = None,
    ):
        """Initialize Monte Carlo engine."""
        if n_simulations is None:
            n_simulations = TRADING_CONFIG.MONTE_CARLO_SIMULATIONS
        if confidence_level is None:
            confidence_level = TRADING_CONFIG.MONTE_CARLO_CONFIDENCE
        
        self.n_simulations = n_simulations
        self.confidence_level = confidence_level
    
    def run(
        self,
        trades_df: pd.DataFrame,
        initial_capital: float,
        years: float,
    ) -> MonteCarloResults:
        """
        Run Monte Carlo simulation on trade results.
        
        Args:
            trades_df: DataFrame with trade history (needs 'PnL' column)
            initial_capital: Starting capital
            years: Number of years in backtest
            
        Returns:
            MonteCarloResults with statistics
        """
        if len(trades_df) == 0:
            raise ValueError("No trades to simulate")
        
        # Extract trade PnLs
        trade_pnls = trades_df['PnL'].values
        n_trades = len(trade_pnls)
        
        print(f"\nMonte Carlo Simulation")
        print(f"  Trades: {n_trades}")
        print(f"  Simulations: {self.n_simulations}")
        
        # Run simulations
        all_max_dds = []
        all_final_equities = []
        
        for i in range(self.n_simulations):
            if i % 200 == 0:
                print(f"  Running simulation {i+1}/{self.n_simulations}...")
            
            # Resample trades with replacement
            resampled_pnls = np.random.choice(trade_pnls, size=n_trades, replace=True)
            
            # Build equity curve
            equity = initial_capital
            peak = initial_capital
            max_dd = 0.0
            
            for pnl in resampled_pnls:
                equity += pnl
                
                if equity > peak:
                    peak = equity
                
                dd = (peak - equity) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)
            
            all_max_dds.append(max_dd)
            all_final_equities.append(equity)
        
        all_max_dds = np.array(all_max_dds)
        all_final_equities = np.array(all_final_equities)
        
        # Calculate CAGRs
        all_cagrs = (all_final_equities / initial_capital) ** (1 / years) - 1
        
        # Drawdown statistics
        median_max_dd = np.median(all_max_dds)
        mean_max_dd = np.mean(all_max_dds)
        dd_5th = np.percentile(all_max_dds, 5)
        dd_95th = np.percentile(all_max_dds, 95)
        dd_worst = np.max(all_max_dds)
        
        # Return statistics
        median_cagr = np.median(all_cagrs)
        mean_cagr = np.mean(all_cagrs)
        cagr_5th = np.percentile(all_cagrs, 5)
        cagr_95th = np.percentile(all_cagrs, 95)
        
        # Risk of ruin (probability of hitting certain DD levels)
        prob_20 = (all_max_dds >= 0.20).mean()
        prob_30 = (all_max_dds >= 0.30).mean()
        prob_50 = (all_max_dds >= 0.50).mean()
        
        # Confidence intervals
        alpha = 1 - self.confidence_level
        cagr_ci_lower = np.percentile(all_cagrs, alpha / 2 * 100)
        cagr_ci_upper = np.percentile(all_cagrs, (1 - alpha / 2) * 100)
        dd_ci_lower = np.percentile(all_max_dds, alpha / 2 * 100)
        dd_ci_upper = np.percentile(all_max_dds, (1 - alpha / 2) * 100)
        
        return MonteCarloResults(
            n_simulations=self.n_simulations,
            median_max_dd=median_max_dd,
            mean_max_dd=mean_max_dd,
            dd_5th_percentile=dd_5th,
            dd_95th_percentile=dd_95th,
            dd_worst=dd_worst,
            median_cagr=median_cagr,
            mean_cagr=mean_cagr,
            cagr_5th_percentile=cagr_5th,
            cagr_95th_percentile=cagr_95th,
            prob_ruin_20pct=prob_20,
            prob_ruin_30pct=prob_30,
            prob_ruin_50pct=prob_50,
            cagr_ci_lower=cagr_ci_lower,
            cagr_ci_upper=cagr_ci_upper,
            dd_ci_lower=dd_ci_lower,
            dd_ci_upper=dd_ci_upper,
            all_max_dds=all_max_dds,
            all_cagrs=all_cagrs,
        )
    
    def run_bootstrap_returns(
        self,
        equity_df: pd.DataFrame,
        initial_capital: float,
    ) -> MonteCarloResults:
        """
        Alternative: Bootstrap daily returns instead of trades.
        
        More conservative as it breaks trade autocorrelation.
        """
        if len(equity_df) == 0:
            raise ValueError("No equity data to simulate")
        
        # Get daily returns
        returns = equity_df['Equity'].pct_change().dropna().values
        n_days = len(returns)
        years = n_days / 252
        
        print(f"\nMonte Carlo (Bootstrap Returns)")
        print(f"  Days: {n_days}")
        print(f"  Simulations: {self.n_simulations}")
        
        all_max_dds = []
        all_final_equities = []
        
        for i in range(self.n_simulations):
            if i % 200 == 0:
                print(f"  Running simulation {i+1}/{self.n_simulations}...")
            
            # Resample returns
            resampled_returns = np.random.choice(returns, size=n_days, replace=True)
            
            # Build equity curve
            equity = initial_capital
            peak = initial_capital
            max_dd = 0.0
            
            for ret in resampled_returns:
                equity *= (1 + ret)
                
                if equity > peak:
                    peak = equity
                
                dd = (peak - equity) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)
            
            all_max_dds.append(max_dd)
            all_final_equities.append(equity)
        
        all_max_dds = np.array(all_max_dds)
        all_final_equities = np.array(all_final_equities)
        all_cagrs = (all_final_equities / initial_capital) ** (1 / years) - 1
        
        # Same statistics as trade-based method
        return MonteCarloResults(
            n_simulations=self.n_simulations,
            median_max_dd=np.median(all_max_dds),
            mean_max_dd=np.mean(all_max_dds),
            dd_5th_percentile=np.percentile(all_max_dds, 5),
            dd_95th_percentile=np.percentile(all_max_dds, 95),
            dd_worst=np.max(all_max_dds),
            median_cagr=np.median(all_cagrs),
            mean_cagr=np.mean(all_cagrs),
            cagr_5th_percentile=np.percentile(all_cagrs, 5),
            cagr_95th_percentile=np.percentile(all_cagrs, 95),
            prob_ruin_20pct=(all_max_dds >= 0.20).mean(),
            prob_ruin_30pct=(all_max_dds >= 0.30).mean(),
            prob_ruin_50pct=(all_max_dds >= 0.50).mean(),
            cagr_ci_lower=np.percentile(all_cagrs, 2.5),
            cagr_ci_upper=np.percentile(all_cagrs, 97.5),
            dd_ci_lower=np.percentile(all_max_dds, 2.5),
            dd_ci_upper=np.percentile(all_max_dds, 97.5),
            all_max_dds=all_max_dds,
            all_cagrs=all_cagrs,
        )


def print_monte_carlo_results(results: MonteCarloResults):
    """Print formatted Monte Carlo results."""
    print("\n" + "=" * 60)
    print("MONTE CARLO RESULTS")
    print("=" * 60)
    
    print(f"\nSimulations: {results.n_simulations}")
    
    print("\n--- DRAWDOWN DISTRIBUTION ---")
    print(f"Median Max DD:     {results.median_max_dd:>10.1%}")
    print(f"Mean Max DD:       {results.mean_max_dd:>10.1%}")
    print(f"5th Percentile:    {results.dd_5th_percentile:>10.1%}")
    print(f"95th Percentile:   {results.dd_95th_percentile:>10.1%}")
    print(f"Worst Case:        {results.dd_worst:>10.1%}")
    
    print("\n--- RETURN DISTRIBUTION ---")
    print(f"Median CAGR:       {results.median_cagr:>10.1%}")
    print(f"Mean CAGR:         {results.mean_cagr:>10.1%}")
    print(f"5th Percentile:    {results.cagr_5th_percentile:>10.1%}")
    print(f"95th Percentile:   {results.cagr_95th_percentile:>10.1%}")
    
    print("\n--- RISK OF RUIN ---")
    print(f"P(DD > 20%):       {results.prob_ruin_20pct:>10.1%}")
    print(f"P(DD > 30%):       {results.prob_ruin_30pct:>10.1%}")
    print(f"P(DD > 50%):       {results.prob_ruin_50pct:>10.1%}")
    
    print("\n--- CONFIDENCE INTERVALS (95%) ---")
    print(f"CAGR:              [{results.cagr_ci_lower:>6.1%}, {results.cagr_ci_upper:>6.1%}]")
    print(f"Max DD:            [{results.dd_ci_lower:>6.1%}, {results.dd_ci_upper:>6.1%}]")
    
    print("=" * 60)


def check_monte_carlo_viability(results: MonteCarloResults) -> Tuple[bool, List[str]]:
    """
    Check if Monte Carlo results pass viability criteria.
    
    Returns:
        Tuple of (passed, list of issues)
    """
    issues = []
    
    # 95th percentile DD must be < 30%
    if results.dd_95th_percentile > TRADING_CONFIG.MONTE_CARLO_MAX_DD:
        issues.append(
            f"95th percentile DD ({results.dd_95th_percentile:.1%}) "
            f"> {TRADING_CONFIG.MONTE_CARLO_MAX_DD:.0%} limit"
        )
    
    # Probability of 30% DD should be low
    if results.prob_ruin_30pct > 0.10:
        issues.append(
            f"P(DD > 30%) = {results.prob_ruin_30pct:.1%} is too high (>10%)"
        )
    
    # Median CAGR should be positive
    if results.median_cagr < 0:
        issues.append(
            f"Median CAGR ({results.median_cagr:.1%}) is negative"
        )
    
    # 5th percentile CAGR should not be severely negative
    if results.cagr_5th_percentile < -0.10:
        issues.append(
            f"5th percentile CAGR ({results.cagr_5th_percentile:.1%}) < -10%"
        )
    
    passed = len(issues) == 0
    return passed, issues


if __name__ == "__main__":
    # Test with synthetic trades
    print("Monte Carlo Test with Synthetic Trades")
    print("=" * 60)
    
    # Create synthetic trade distribution
    np.random.seed(42)
    
    # Simulate trend following style: many small losses, few big wins
    n_trades = 200
    win_rate = 0.40
    
    wins = np.random.uniform(500, 5000, int(n_trades * win_rate))
    losses = np.random.uniform(-500, -100, n_trades - len(wins))
    
    pnls = np.concatenate([wins, losses])
    np.random.shuffle(pnls)
    
    trades_df = pd.DataFrame({'PnL': pnls})
    
    print(f"Synthetic trades: {n_trades}")
    print(f"Win rate: {win_rate:.0%}")
    print(f"Total PnL: ${pnls.sum():,.2f}")
    
    # Run Monte Carlo
    mc = MonteCarloEngine(n_simulations=1000)
    results = mc.run(trades_df, initial_capital=100_000, years=10)
    
    print_monte_carlo_results(results)
    
    # Check viability
    passed, issues = check_monte_carlo_viability(results)
    print(f"\nViability Check: {'PASS' if passed else 'FAIL'}")
    for issue in issues:
        print(f"  - {issue}")
