"""
Trend Following System - Configuration
SINGLE SOURCE OF TRUTH - FROZEN PARAMETERS

DO NOT MODIFY THESE VALUES.
All parameters are canonical and have been locked to prevent optimization bias.
"""

from dataclasses import dataclass
from typing import List, Tuple
from datetime import datetime


@dataclass(frozen=True)
class TradingConfig:
    """Immutable trading configuration."""
    
    # ==========================================================================
    # UNIVERSE (LOCKED)
    # ==========================================================================
    SYMBOLS: Tuple[str, ...] = ("SPY", "QQQ", "IWM", "GLD")
    
    # ==========================================================================
    # INDICATOR PARAMETERS (CANONICAL - DO NOT OPTIMIZE)
    # ==========================================================================
    EMA_PERIOD: int = 200           # Long-term trend filter
    DONCHIAN_ENTRY: int = 55        # Breakout entry channel
    DONCHIAN_EXIT: int = 20         # Trailing stop channel
    ATR_PERIOD: int = 20            # Volatility measurement
    ATR_LOOKBACK: int = 252         # Median ATR reference period
    
    # ==========================================================================
    # RISK MANAGEMENT (HARD LIMITS)
    # ==========================================================================
    RISK_PER_TRADE: float = 0.005   # 0.5% of equity per trade
    MAX_POSITIONS: int = 3          # Maximum concurrent positions
    MAX_DRAWDOWN: float = 0.20      # 20% max drawdown tolerance
    
    # ==========================================================================
    # COST MODEL (REALISTIC)
    # ==========================================================================
    COMMISSION_PER_SHARE: float = 0.01  # $0.01 per share
    SLIPPAGE_PCT: float = 0.0001        # 0.01% slippage
    
    # ==========================================================================
    # BACKTEST PARAMETERS
    # ==========================================================================
    INITIAL_CAPITAL: float = 100_000.0
    MIN_HISTORY_YEARS: int = 10
    
    # ==========================================================================
    # VIABILITY KILL CRITERIA (SYSTEM REJECTED IF ANY FAIL)
    # ==========================================================================
    MIN_EXPECTANCY_R: float = 0.25
    MIN_PROFIT_FACTOR: float = 1.5
    MAX_ALLOWED_DD: float = 0.20
    MIN_TOTAL_TRADES: int = 200
    MONTE_CARLO_MAX_DD: float = 0.30  # 95th percentile
    
    # ==========================================================================
    # MONTE CARLO PARAMETERS
    # ==========================================================================
    MONTE_CARLO_SIMULATIONS: int = 1000
    MONTE_CARLO_CONFIDENCE: float = 0.95


@dataclass(frozen=True)
class ExecutionConfig:
    """Execution and operational configuration."""
    
    # Timing
    SIGNAL_TIME: str = "16:00"      # Evaluate signals at market close
    EXECUTION_TIME: str = "09:31"   # Execute at next open + 1 min
    
    # Data source
    DATA_SOURCE: str = "yfinance"   # Primary data source
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s | %(levelname)s | %(message)s"
    
    # Paths
    DATA_DIR: str = "data"
    LOGS_DIR: str = "logs"
    RESULTS_DIR: str = "results"


# Global singleton instances
TRADING_CONFIG = TradingConfig()
EXECUTION_CONFIG = ExecutionConfig()


def print_config():
    """Print current configuration for audit trail."""
    print("=" * 60)
    print("TREND FOLLOWING SYSTEM - CONFIGURATION")
    print("=" * 60)
    print(f"\nUniverse: {TRADING_CONFIG.SYMBOLS}")
    print(f"\nIndicators:")
    print(f"  EMA Period:        {TRADING_CONFIG.EMA_PERIOD}")
    print(f"  Donchian Entry:    {TRADING_CONFIG.DONCHIAN_ENTRY}")
    print(f"  Donchian Exit:     {TRADING_CONFIG.DONCHIAN_EXIT}")
    print(f"  ATR Period:        {TRADING_CONFIG.ATR_PERIOD}")
    print(f"\nRisk Management:")
    print(f"  Risk per Trade:    {TRADING_CONFIG.RISK_PER_TRADE:.1%}")
    print(f"  Max Positions:     {TRADING_CONFIG.MAX_POSITIONS}")
    print(f"  Max Drawdown:      {TRADING_CONFIG.MAX_DRAWDOWN:.0%}")
    print(f"\nCost Model:")
    print(f"  Commission:        ${TRADING_CONFIG.COMMISSION_PER_SHARE}/share")
    print(f"  Slippage:          {TRADING_CONFIG.SLIPPAGE_PCT:.2%}")
    print(f"\nKill Criteria:")
    print(f"  Min Expectancy:    {TRADING_CONFIG.MIN_EXPECTANCY_R}R")
    print(f"  Min Profit Factor: {TRADING_CONFIG.MIN_PROFIT_FACTOR}")
    print(f"  Max Drawdown:      {TRADING_CONFIG.MAX_ALLOWED_DD:.0%}")
    print(f"  Min Trades:        {TRADING_CONFIG.MIN_TOTAL_TRADES}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
