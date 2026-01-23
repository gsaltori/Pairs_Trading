"""
Time-Series Momentum System with Volatility Targeting
Configuration - SINGLE SOURCE OF TRUTH - ALL PARAMETERS LOCKED

Academic Foundation:
- Moskowitz, Ooi, Pedersen (2012): "Time Series Momentum"
- Volatility targeting: Moreira & Muir (2017)

DO NOT OPTIMIZE THESE PARAMETERS
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class TSMOMConfig:
    """Immutable system configuration."""
    
    # ==========================================================================
    # UNIVERSE (LOCKED)
    # ==========================================================================
    UNIVERSE: Tuple[str, ...] = (
        "SPY",   # US Large Cap
        "QQQ",   # US Tech
        "IWM",   # US Small Cap
        "EFA",   # Developed International
        "EEM",   # Emerging Markets
        "TLT",   # Long-Term Treasuries
        "GLD",   # Gold
        "DBC",   # Commodities
    )
    
    # ==========================================================================
    # SIGNAL PARAMETERS (CANONICAL)
    # ==========================================================================
    TREND_LOOKBACK: int = 252         # 12-month SMA for trend signal
    
    # ==========================================================================
    # VOLATILITY TARGETING
    # ==========================================================================
    TARGET_VOL: float = 0.10          # 10% annualized target
    VOL_LOOKBACK: int = 20            # 20-day realized volatility
    MAX_WEIGHT_PER_ASSET: float = 0.30  # 30% cap per asset
    
    # ==========================================================================
    # REBALANCE
    # ==========================================================================
    REBALANCE_FREQUENCY: str = "MONTHLY"
    
    # ==========================================================================
    # TRANSACTION COSTS
    # ==========================================================================
    COMMISSION_PCT: float = 0.0002    # 0.02%
    SLIPPAGE_PCT: float = 0.0001      # 0.01%
    
    # ==========================================================================
    # CAPITAL
    # ==========================================================================
    INITIAL_CAPITAL: float = 100_000.0
    
    # ==========================================================================
    # BACKTEST
    # ==========================================================================
    BACKTEST_START: str = "2006-01-01"
    OUT_OF_SAMPLE_PCT: float = 0.30
    
    # ==========================================================================
    # VIABILITY CRITERIA
    # ==========================================================================
    MIN_SHARPE: float = 0.50
    MAX_DRAWDOWN: float = 0.25
    MIN_TRADES: int = 100
    
    # ==========================================================================
    # MONTE CARLO
    # ==========================================================================
    MONTE_CARLO_RUNS: int = 1000
    BLOCK_SIZE: int = 3  # 3-month blocks for bootstrap


CONFIG = TSMOMConfig()


def print_config():
    """Print configuration for audit."""
    print("=" * 70)
    print("TIME-SERIES MOMENTUM WITH VOLATILITY TARGETING - CONFIGURATION")
    print("=" * 70)
    print(f"\nUniverse ({len(CONFIG.UNIVERSE)} assets):")
    for sym in CONFIG.UNIVERSE:
        print(f"  - {sym}")
    print(f"\nSignal Logic:")
    print(f"  If Close > SMA({CONFIG.TREND_LOOKBACK}) → LONG")
    print(f"  Else → CASH (no shorts)")
    print(f"\nVolatility Targeting:")
    print(f"  Target Vol:        {CONFIG.TARGET_VOL:.0%} annualized")
    print(f"  Vol Lookback:      {CONFIG.VOL_LOOKBACK} days")
    print(f"  Max Weight/Asset:  {CONFIG.MAX_WEIGHT_PER_ASSET:.0%}")
    print(f"\nRebalance: {CONFIG.REBALANCE_FREQUENCY} (last trading day)")
    print(f"\nCosts:")
    print(f"  Commission:        {CONFIG.COMMISSION_PCT:.2%}")
    print(f"  Slippage:          {CONFIG.SLIPPAGE_PCT:.2%}")
    print("=" * 70)


if __name__ == "__main__":
    print_config()
