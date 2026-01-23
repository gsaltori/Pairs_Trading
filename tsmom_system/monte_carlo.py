"""
Time-Series Momentum System - Monte Carlo Analysis
Block bootstrap simulation for robustness testing.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List

from config import CONFIG


@dataclass
class MonteCarloResult:
    """Monte Carlo simulation results."""
    n_simulations: int
    block_size: int
    
    # CAGR distribution
    cagr_mean: float
    cagr_median: float
    cagr_5th: float
    cagr_95th: float
    cagr_min: float
    cagr_max: float
    
    # Drawdown distribution
    dd_mean: float
    dd_median: float
    dd_5th: float
    dd_95th: float
    dd_max: float
    
    # Risk probabilities
    prob_loss: float        # P(CAGR < 0)
    prob_dd_20: float       # P(DD > 20%)
    prob_dd_30: float       # P(DD > 30%)
    
    # Raw arrays for plotting
    all_cagrs: np.ndarray
    all_max_dds: np.ndarray


class MonteCarloSimulator:
    """
    Block bootstrap Monte Carlo simulation.
    
    Uses overlapping blocks to preserve autocorrelation
    structure in monthly returns.
    """
    
    def __init__(
        self,
        n_simulations: int = None,
        block_size: int = None,
    ):
        if n_simulations is None:
            n_simulations = CONFIG.MONTE_CARLO_RUNS
        if block_size is None:
            block_size = CONFIG.BLOCK_SIZE
        
        self.n_simulations = n_simulations
        self.block_size = block_size
    
    def run(
        self,
        monthly_returns: pd.Series,
        initial_capital: float = None,
    ) -> MonteCarloResult:
        """
        Run block bootstrap Monte Carlo.
        
        Args:
            monthly_returns: Series of monthly returns
            initial_capital: Starting capital
            
        Returns:
            MonteCarloResult with all statistics
        """
        if initial_capital is None:
            initial_capital = CONFIG.INITIAL_CAPITAL
        
        returns = monthly_returns.values
        n_months = len(returns)
        years = n_months / 12
        
        if n_months < self.block_size * 2:
            raise ValueError(f"Need at least {self.block_size * 2} months of data")
        
        print(f"\nMonte Carlo Simulation")
        print(f"  Simulations: {self.n_simulations}")
        print(f"  Block size: {self.block_size} months")
        print(f"  Data: {n_months} months ({years:.1f} years)")
        
        all_cagrs = []
        all_max_dds = []
        
        for i in range(self.n_simulations):
            if (i + 1) % 200 == 0:
                print(f"  Progress: {i+1}/{self.n_simulations}")
            
            # Generate resampled returns using block bootstrap
            resampled = self._block_bootstrap(returns, n_months)
            
            # Build equity curve
            equity = initial_capital
            peak = initial_capital
            max_dd = 0.0
            
            for ret in resampled:
                equity *= (1 + ret)
                
                if equity > peak:
                    peak = equity
                
                if peak > 0:
                    dd = (peak - equity) / peak
                    max_dd = max(max_dd, dd)
            
            # Calculate CAGR
            if equity > 0 and years > 0:
                cagr = (equity / initial_capital) ** (1 / years) - 1
            else:
                cagr = -1.0
            
            all_cagrs.append(cagr)
            all_max_dds.append(max_dd)
        
        all_cagrs = np.array(all_cagrs)
        all_max_dds = np.array(all_max_dds)
        
        return MonteCarloResult(
            n_simulations=self.n_simulations,
            block_size=self.block_size,
            cagr_mean=np.mean(all_cagrs),
            cagr_median=np.median(all_cagrs),
            cagr_5th=np.percentile(all_cagrs, 5),
            cagr_95th=np.percentile(all_cagrs, 95),
            cagr_min=np.min(all_cagrs),
            cagr_max=np.max(all_cagrs),
            dd_mean=np.mean(all_max_dds),
            dd_median=np.median(all_max_dds),
            dd_5th=np.percentile(all_max_dds, 5),
            dd_95th=np.percentile(all_max_dds, 95),
            dd_max=np.max(all_max_dds),
            prob_loss=(all_cagrs < 0).mean(),
            prob_dd_20=(all_max_dds > 0.20).mean(),
            prob_dd_30=(all_max_dds > 0.30).mean(),
            all_cagrs=all_cagrs,
            all_max_dds=all_max_dds,
        )
    
    def _block_bootstrap(
        self,
        returns: np.ndarray,
        target_length: int,
    ) -> np.ndarray:
        """
        Block bootstrap resampling.
        
        Randomly selects overlapping blocks of consecutive
        returns to preserve autocorrelation.
        """
        n = len(returns)
        result = []
        
        while len(result) < target_length:
            # Random starting position
            max_start = n - self.block_size
            if max_start < 0:
                max_start = 0
            
            start = np.random.randint(0, max_start + 1)
            
            # Extract block
            end = min(start + self.block_size, n)
            block = returns[start:end]
            result.extend(block)
        
        return np.array(result[:target_length])


def print_monte_carlo(mc: MonteCarloResult):
    """Print Monte Carlo results."""
    print("\n" + "=" * 60)
    print("MONTE CARLO ANALYSIS")
    print("=" * 60)
    
    print(f"\nSimulations: {mc.n_simulations}")
    print(f"Block Size: {mc.block_size} months")
    
    print("\n--- CAGR DISTRIBUTION ---")
    print(f"  5th Percentile:    {mc.cagr_5th:>10.1%}")
    print(f"  Median:            {mc.cagr_median:>10.1%}")
    print(f"  Mean:              {mc.cagr_mean:>10.1%}")
    print(f"  95th Percentile:   {mc.cagr_95th:>10.1%}")
    print(f"  Range:             {mc.cagr_min:>10.1%} to {mc.cagr_max:.1%}")
    
    print("\n--- DRAWDOWN DISTRIBUTION ---")
    print(f"  5th Percentile:    {mc.dd_5th:>10.1%}")
    print(f"  Median:            {mc.dd_median:>10.1%}")
    print(f"  Mean:              {mc.dd_mean:>10.1%}")
    print(f"  95th Percentile:   {mc.dd_95th:>10.1%}")
    print(f"  Worst Case:        {mc.dd_max:>10.1%}")
    
    print("\n--- RISK PROBABILITIES ---")
    print(f"  P(Loss):           {mc.prob_loss:>10.1%}")
    print(f"  P(DD > 20%):       {mc.prob_dd_20:>10.1%}")
    print(f"  P(DD > 30%):       {mc.prob_dd_30:>10.1%}")
    
    print("=" * 60)


if __name__ == "__main__":
    # Test with synthetic returns
    np.random.seed(42)
    
    # Simulate realistic monthly returns
    monthly = pd.Series(np.random.normal(0.007, 0.035, 200))
    
    mc = MonteCarloSimulator(n_simulations=500)
    result = mc.run(monthly, 100_000)
    
    print_monte_carlo(result)
