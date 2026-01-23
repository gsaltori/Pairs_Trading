"""
Cross-Sectional Momentum System - Configuration
SINGLE SOURCE OF TRUTH - ALL PARAMETERS LOCKED

Academic foundation: Jegadeesh & Titman (1993), Asness et al. (2013)
This is a well-documented factor with decades of out-of-sample evidence.
"""

from dataclasses import dataclass
from typing import Tuple
from datetime import datetime


@dataclass(frozen=True)
class MomentumConfig:
    """Immutable momentum system configuration."""
    
    # ==========================================================================
    # UNIVERSE (LOCKED - Diversified across asset classes)
    # ==========================================================================
    UNIVERSE: Tuple[str, ...] = (
        "SPY",   # US Large Cap
        "QQQ",   # US Tech/Growth
        "IWM",   # US Small Cap
        "EFA",   # Developed International
        "EEM",   # Emerging Markets
        "TLT",   # Long-Term Treasuries
        "IEF",   # Intermediate Treasuries
        "GLD",   # Gold
        "DBC",   # Commodities
        "VNQ",   # Real Estate
    )
    
    # ==========================================================================
    # MOMENTUM PARAMETERS (CANONICAL - DO NOT OPTIMIZE)
    # ==========================================================================
    MOMENTUM_LOOKBACK: int = 252      # 12 months (~252 trading days)
    TREND_FILTER_PERIOD: int = 200    # EMA(200) trend filter
    TOP_N_ASSETS: int = 3             # Select top 3 by momentum
    
    # ==========================================================================
    # REBALANCE SCHEDULE
    # ==========================================================================
    REBALANCE_FREQUENCY: str = "MONTHLY"  # Last trading day of month
    
    # ==========================================================================
    # POSITION SIZING (EQUAL WEIGHT)
    # ==========================================================================
    MAX_POSITIONS: int = 3
    WEIGHT_PER_POSITION: float = 1.0 / 3  # Equal weight ~33.3%
    
    # ==========================================================================
    # TRANSACTION COSTS (REALISTIC)
    # ==========================================================================
    COMMISSION_PCT: float = 0.0002    # 0.02% commission
    SLIPPAGE_PCT: float = 0.0001      # 0.01% slippage
    
    # ==========================================================================
    # CAPITAL
    # ==========================================================================
    INITIAL_CAPITAL: float = 100_000.0
    
    # ==========================================================================
    # BACKTEST PARAMETERS
    # ==========================================================================
    BACKTEST_START: str = "2005-01-01"
    OUT_OF_SAMPLE_PCT: float = 0.30    # Last 30% for OOS
    
    # ==========================================================================
    # VIABILITY KILL CRITERIA
    # ==========================================================================
    MIN_EXPECTANCY_R: float = 0.25
    MIN_SHARPE: float = 0.70
    MAX_DRAWDOWN: float = 0.30
    MIN_TRADES: int = 300
    
    # ==========================================================================
    # MONTE CARLO
    # ==========================================================================
    MONTE_CARLO_RUNS: int = 1000
    MONTE_CARLO_CONFIDENCE: float = 0.95


@dataclass(frozen=True)
class ExecutionConfig:
    """Execution configuration."""
    DATA_SOURCE: str = "yfinance"
    LOG_DIR: str = "logs"
    RESULTS_DIR: str = "results"
    DRY_RUN: bool = True  # Default to paper trading


# Global singletons
CONFIG = MomentumConfig()
EXEC_CONFIG = ExecutionConfig()


def print_config():
    """Print configuration for audit."""
    print("=" * 70)
    print("CROSS-SECTIONAL MOMENTUM SYSTEM - CONFIGURATION")
    print("=" * 70)
    print(f"\nUniverse ({len(CONFIG.UNIVERSE)} assets):")
    for i, sym in enumerate(CONFIG.UNIVERSE):
        print(f"  {i+1}. {sym}")
    print(f"\nMomentum Parameters:")
    print(f"  Lookback:          {CONFIG.MOMENTUM_LOOKBACK} days (12 months)")
    print(f"  Trend Filter:      EMA({CONFIG.TREND_FILTER_PERIOD})")
    print(f"  Top N Selection:   {CONFIG.TOP_N_ASSETS}")
    print(f"\nRebalance:           {CONFIG.REBALANCE_FREQUENCY}")
    print(f"\nPosition Sizing:")
    print(f"  Max Positions:     {CONFIG.MAX_POSITIONS}")
    print(f"  Weight Each:       {CONFIG.WEIGHT_PER_POSITION:.1%}")
    print(f"\nTransaction Costs:")
    print(f"  Commission:        {CONFIG.COMMISSION_PCT:.2%}")
    print(f"  Slippage:          {CONFIG.SLIPPAGE_PCT:.2%}")
    print(f"\nKill Criteria:")
    print(f"  Min Expectancy:    {CONFIG.MIN_EXPECTANCY_R}R")
    print(f"  Min Sharpe:        {CONFIG.MIN_SHARPE}")
    print(f"  Max Drawdown:      {CONFIG.MAX_DRAWDOWN:.0%}")
    print(f"  Min Trades:        {CONFIG.MIN_TRADES}")
    print("=" * 70)


if __name__ == "__main__":
    print_config()
