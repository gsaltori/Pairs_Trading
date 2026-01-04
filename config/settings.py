"""
Settings and Configuration for Pairs Trading System.

Centralized configuration management using dataclasses and YAML support.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
from pathlib import Path
import yaml


class TradingMode(Enum):
    """Trading mode enumeration."""
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


class Timeframe(Enum):
    """Supported timeframes."""
    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    M30 = "M30"
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"
    
    def to_minutes(self) -> int:
        """Convert timeframe to minutes."""
        mapping = {
            'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30,
            'H1': 60, 'H4': 240, 'D1': 1440
        }
        return mapping.get(self.value, 60)
    
    def to_mt5(self) -> str:
        """Convert to MT5 timeframe string."""
        return self.value


@dataclass
class SpreadSettings:
    """Spread construction and signal parameters."""
    
    # Entry/Exit thresholds
    entry_zscore: float = 2.0        # |Z| >= 2.0 to enter
    exit_zscore: float = 0.2         # |Z| <= 0.2 to exit (mean reversion)
    stop_loss_zscore: float = 3.0    # |Z| >= 3.0 emergency stop
    
    # Regression settings
    regression_window: int = 120     # Bars for hedge ratio calculation
    zscore_window: int = 60          # Bars for z-score normalization
    
    # Correlation filter
    min_correlation: float = 0.70    # Minimum correlation to trade
    correlation_window: int = 60     # Bars for rolling correlation
    
    # Spread validation
    max_half_life: int = 50          # Maximum half-life (bars)
    min_half_life: int = 5           # Minimum half-life (bars)
    
    # Recalculation
    recalc_hedge_ratio: bool = True  # Rolling hedge ratio
    recalc_interval: int = 20        # Bars between recalculations


@dataclass
class RiskSettings:
    """Risk management parameters."""
    
    # Per-trade risk
    max_risk_per_trade: float = 0.01     # 1% of capital
    
    # Exposure limits
    max_total_exposure: float = 0.10      # 10% total exposure
    max_open_pairs: int = 3               # Maximum concurrent pairs
    
    # Drawdown control
    max_drawdown: float = 0.15            # 15% maximum drawdown
    drawdown_halt: bool = True            # Stop trading at max DD
    
    # Position sizing
    balance_legs: bool = True             # Equal $ risk per leg
    use_atr_sizing: bool = False          # ATR-based sizing
    
    # Daily limits
    max_daily_trades: int = 10            # Maximum trades per day
    max_daily_loss: float = 0.03          # 3% max daily loss


@dataclass 
class BacktestSettings:
    """Backtesting parameters."""
    
    # Capital
    initial_capital: float = 10000.0
    
    # Transaction costs (in pips)
    spread_cost: float = 1.5              # Average spread cost
    slippage: float = 0.5                 # Execution slippage
    commission_per_lot: float = 7.0       # Commission in USD per lot
    
    # Execution simulation
    fill_ratio: float = 1.0               # Percentage of orders filled
    
    # Data requirements
    min_bars_required: int = 200          # Minimum bars for backtest


@dataclass
class OptimizationSettings:
    """Walk-forward optimization parameters."""
    
    # Window sizes (in bars)
    in_sample_bars: int = 504             # ~3 weeks H1
    out_of_sample_bars: int = 168         # ~1 week H1
    
    # Iteration control
    max_iterations: int = 100
    min_efficiency_ratio: float = 0.5     # OOS/IS Sharpe ratio threshold
    
    # Parameter search
    optimize_entry_zscore: bool = True
    optimize_regression_window: bool = True
    optimize_zscore_window: bool = True


@dataclass
class PathSettings:
    """File paths configuration."""
    
    cache_dir: str = "data/cache"
    historical_dir: str = "data/historical"
    results_dir: str = "results"
    logs_dir: str = "logs"
    backtest_dir: str = "results/backtests"
    optimization_dir: str = "results/optimization"
    
    def create_directories(self):
        """Create all required directories."""
        for path in [
            self.cache_dir, self.historical_dir, self.results_dir,
            self.logs_dir, self.backtest_dir, self.optimization_dir
        ]:
            Path(path).mkdir(parents=True, exist_ok=True)


@dataclass
class Settings:
    """Main settings container."""
    
    # Sub-settings
    spread: SpreadSettings = field(default_factory=SpreadSettings)
    risk: RiskSettings = field(default_factory=RiskSettings)
    backtest: BacktestSettings = field(default_factory=BacktestSettings)
    optimization: OptimizationSettings = field(default_factory=OptimizationSettings)
    paths: PathSettings = field(default_factory=PathSettings)
    
    # Trading configuration
    mode: TradingMode = TradingMode.BACKTEST
    timeframe: Timeframe = Timeframe.H1
    
    # Default pairs to analyze
    default_pairs: List[Tuple[str, str]] = field(default_factory=lambda: [
        ("EURUSD", "GBPUSD"),
        ("AUDUSD", "NZDUSD"),
        ("EURJPY", "USDJPY"),
        ("EURCHF", "USDCHF"),
    ])
    
    # Universe of symbols for screening
    symbol_universe: List[str] = field(default_factory=lambda: [
        "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
        "AUDUSD", "NZDUSD", "USDCAD",
        "EURJPY", "GBPJPY", "AUDJPY",
        "EURGBP", "EURAUD", "EURCHF",
    ])
    
    def __post_init__(self):
        """Initialize paths after creation."""
        self.paths.create_directories()
    
    @classmethod
    def from_yaml(cls, path: str) -> 'Settings':
        """Load settings from YAML file."""
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        
        return cls(
            spread=SpreadSettings(**config.get('spread', {})),
            risk=RiskSettings(**config.get('risk', {})),
            backtest=BacktestSettings(**config.get('backtest', {})),
            optimization=OptimizationSettings(**config.get('optimization', {})),
            paths=PathSettings(**config.get('paths', {})),
            mode=TradingMode(config.get('mode', 'backtest')),
            timeframe=Timeframe(config.get('timeframe', 'H1')),
            default_pairs=[tuple(p) for p in config.get('default_pairs', [])],
            symbol_universe=config.get('symbol_universe', [])
        )
    
    def to_yaml(self, path: str):
        """Save settings to YAML file."""
        config = {
            'spread': {
                'entry_zscore': self.spread.entry_zscore,
                'exit_zscore': self.spread.exit_zscore,
                'stop_loss_zscore': self.spread.stop_loss_zscore,
                'regression_window': self.spread.regression_window,
                'zscore_window': self.spread.zscore_window,
                'min_correlation': self.spread.min_correlation,
                'correlation_window': self.spread.correlation_window,
                'max_half_life': self.spread.max_half_life,
                'min_half_life': self.spread.min_half_life,
            },
            'risk': {
                'max_risk_per_trade': self.risk.max_risk_per_trade,
                'max_total_exposure': self.risk.max_total_exposure,
                'max_open_pairs': self.risk.max_open_pairs,
                'max_drawdown': self.risk.max_drawdown,
                'max_daily_trades': self.risk.max_daily_trades,
                'max_daily_loss': self.risk.max_daily_loss,
            },
            'backtest': {
                'initial_capital': self.backtest.initial_capital,
                'spread_cost': self.backtest.spread_cost,
                'slippage': self.backtest.slippage,
                'commission_per_lot': self.backtest.commission_per_lot,
            },
            'optimization': {
                'in_sample_bars': self.optimization.in_sample_bars,
                'out_of_sample_bars': self.optimization.out_of_sample_bars,
                'max_iterations': self.optimization.max_iterations,
                'min_efficiency_ratio': self.optimization.min_efficiency_ratio,
            },
            'mode': self.mode.value,
            'timeframe': self.timeframe.value,
            'default_pairs': [list(p) for p in self.default_pairs],
            'symbol_universe': self.symbol_universe,
        }
        
        with open(path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate settings configuration."""
        errors = []
        
        # Spread validation
        if self.spread.entry_zscore <= self.spread.exit_zscore:
            errors.append("entry_zscore must be > exit_zscore")
        
        if self.spread.stop_loss_zscore <= self.spread.entry_zscore:
            errors.append("stop_loss_zscore must be > entry_zscore")
        
        if self.spread.min_correlation < 0 or self.spread.min_correlation > 1:
            errors.append("min_correlation must be between 0 and 1")
        
        # Risk validation
        if self.risk.max_risk_per_trade <= 0 or self.risk.max_risk_per_trade > 0.1:
            errors.append("max_risk_per_trade should be between 0 and 10%")
        
        if self.risk.max_open_pairs < 1:
            errors.append("max_open_pairs must be at least 1")
        
        # Backtest validation
        if self.backtest.initial_capital <= 0:
            errors.append("initial_capital must be positive")
        
        return len(errors) == 0, errors
    
    def get_summary(self) -> str:
        """Get settings summary string."""
        return f"""
Pairs Trading Settings Summary
==============================
Mode: {self.mode.value}
Timeframe: {self.timeframe.value}

Spread Parameters:
  Entry Z-score: ±{self.spread.entry_zscore}
  Exit Z-score: ±{self.spread.exit_zscore}
  Stop-loss Z-score: ±{self.spread.stop_loss_zscore}
  Min Correlation: {self.spread.min_correlation}
  Regression Window: {self.spread.regression_window} bars
  Z-score Window: {self.spread.zscore_window} bars

Risk Parameters:
  Max Risk/Trade: {self.risk.max_risk_per_trade:.1%}
  Max Exposure: {self.risk.max_total_exposure:.1%}
  Max Open Pairs: {self.risk.max_open_pairs}
  Max Drawdown: {self.risk.max_drawdown:.1%}

Backtest Parameters:
  Initial Capital: ${self.backtest.initial_capital:,.0f}
  Spread Cost: {self.backtest.spread_cost} pips
  Slippage: {self.backtest.slippage} pips
  Commission: ${self.backtest.commission_per_lot}/lot
"""


# Default configuration template
DEFAULT_CONFIG_YAML = """
# Pairs Trading System Configuration
# IC Markets Global via MetaTrader 5

spread:
  entry_zscore: 2.0
  exit_zscore: 0.2
  stop_loss_zscore: 3.0
  regression_window: 120
  zscore_window: 60
  min_correlation: 0.70
  correlation_window: 60
  max_half_life: 50
  min_half_life: 5

risk:
  max_risk_per_trade: 0.01
  max_total_exposure: 0.10
  max_open_pairs: 3
  max_drawdown: 0.15
  max_daily_trades: 10
  max_daily_loss: 0.03

backtest:
  initial_capital: 10000
  spread_cost: 1.5
  slippage: 0.5
  commission_per_lot: 7.0

optimization:
  in_sample_bars: 504
  out_of_sample_bars: 168
  max_iterations: 100
  min_efficiency_ratio: 0.5

mode: backtest
timeframe: H1

default_pairs:
  - [EURUSD, GBPUSD]
  - [AUDUSD, NZDUSD]
  - [EURJPY, USDJPY]
  - [EURCHF, USDCHF]

symbol_universe:
  - EURUSD
  - GBPUSD
  - USDJPY
  - USDCHF
  - AUDUSD
  - NZDUSD
  - USDCAD
  - EURJPY
  - GBPJPY
  - EURGBP
  - EURCHF
"""
