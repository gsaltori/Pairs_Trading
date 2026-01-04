"""
General settings for the Pairs Trading System.

This module contains all configurable parameters for the trading system.
Parameters are organized by functionality for clarity and maintainability.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from pathlib import Path
from enum import Enum
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
    D = "D"
    
    @property
    def minutes(self) -> int:
        """Convert timeframe to minutes."""
        mapping = {
            "M1": 1, "M5": 5, "M15": 15, "M30": 30,
            "H1": 60, "H4": 240, "D": 1440
        }
        return mapping[self.value]
    
    @property
    def oanda_granularity(self) -> str:
        """Convert to OANDA granularity format."""
        mapping = {
            "M1": "M1", "M5": "M5", "M15": "M15", "M30": "M30",
            "H1": "H1", "H4": "H4", "D": "D"
        }
        return mapping[self.value]


@dataclass
class SpreadParameters:
    """Parameters for spread calculation and signals."""
    # Regression window for hedge ratio calculation
    regression_window: int = 120
    # Z-score calculation window
    zscore_window: int = 60
    # Entry thresholds
    entry_zscore_long: float = -2.0
    entry_zscore_short: float = 2.0
    # Exit thresholds
    exit_zscore_lower: float = -0.2
    exit_zscore_upper: float = 0.2
    # Stop loss threshold
    stop_loss_zscore: float = 3.0
    # Minimum correlation required
    min_correlation: float = 0.70
    # Rolling correlation window
    correlation_window: int = 252  # ~1 year for daily, ~1 month for H1
    # Half-life threshold (bars) - pairs with higher half-life are less mean-reverting
    max_half_life: int = 50


@dataclass
class RiskParameters:
    """Risk management parameters."""
    # Maximum risk per pair trade (% of capital)
    max_risk_per_trade: float = 0.01  # 1%
    # Maximum total exposure (% of capital)
    max_total_exposure: float = 0.10  # 10%
    # Maximum number of pairs open simultaneously
    max_open_pairs: int = 3
    # Maximum drawdown allowed before system stops
    max_drawdown: float = 0.15  # 15%
    # Maximum monetary loss per trade (absolute USD)
    max_loss_per_trade: Optional[float] = None
    # Position sizing method: 'equal' or 'volatility_adjusted'
    sizing_method: str = 'volatility_adjusted'


@dataclass
class BacktestParameters:
    """Backtesting configuration."""
    # Transaction costs (in pips)
    spread_cost: float = 1.5
    # Slippage (in pips)
    slippage: float = 0.5
    # Commission per lot (if applicable)
    commission_per_lot: float = 0.0
    # Initial capital (USD)
    initial_capital: float = 10000.0
    # Use OHLC or close only
    use_ohlc: bool = True


@dataclass
class OptimizationParameters:
    """Walk-forward optimization configuration."""
    # In-sample period (bars)
    in_sample_period: int = 504  # ~6 months for H1
    # Out-of-sample period (bars)
    out_sample_period: int = 168  # ~1 month for H1
    # Number of walk-forward windows
    num_windows: int = 6
    # Parameter ranges for optimization
    regression_window_range: Tuple[int, int] = (60, 180)
    zscore_window_range: Tuple[int, int] = (30, 90)
    entry_zscore_range: Tuple[float, float] = (1.5, 2.5)
    exit_zscore_range: Tuple[float, float] = (0.1, 0.5)
    # Optimization metric: 'sharpe', 'profit_factor', 'expectancy'
    optimization_metric: str = 'sharpe'
    # Number of iterations for optimization
    n_iterations: int = 100
    # Re-optimization frequency (in bars)
    reoptimization_frequency: int = 168  # Weekly for H1


@dataclass
class Settings:
    """Main settings container for the Pairs Trading System."""
    
    # Project paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    
    # Trading mode
    mode: TradingMode = TradingMode.BACKTEST
    
    # Timeframe
    timeframe: Timeframe = Timeframe.H1
    
    # Forex pairs to analyze
    pairs_universe: List[Tuple[str, str]] = field(default_factory=lambda: [
        ("EUR_USD", "GBP_USD"),
        ("AUD_USD", "NZD_USD"),
        ("EUR_JPY", "USD_JPY"),
        ("EUR_CHF", "USD_CHF"),
        ("EUR_GBP", "EUR_CHF"),
        ("GBP_JPY", "EUR_JPY"),
        ("AUD_NZD", "AUD_USD"),
        ("USD_CAD", "USD_CHF"),
    ])
    
    # Module parameters
    spread_params: SpreadParameters = field(default_factory=SpreadParameters)
    risk_params: RiskParameters = field(default_factory=RiskParameters)
    backtest_params: BacktestParameters = field(default_factory=BacktestParameters)
    optimization_params: OptimizationParameters = field(default_factory=OptimizationParameters)
    
    # Data settings
    lookback_days: int = 365 * 2  # 2 years of historical data
    
    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    
    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"
    
    @property
    def historical_dir(self) -> Path:
        return self.data_dir / "historical"
    
    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"
    
    @property
    def results_dir(self) -> Path:
        return self.project_root / "results"
    
    @property
    def logs_dir(self) -> Path:
        return self.project_root / "logs"
    
    def save_to_yaml(self, filepath: Path) -> None:
        """Save settings to YAML file."""
        config_dict = {
            'mode': self.mode.value,
            'timeframe': self.timeframe.value,
            'pairs_universe': self.pairs_universe,
            'spread_params': {
                'regression_window': self.spread_params.regression_window,
                'zscore_window': self.spread_params.zscore_window,
                'entry_zscore_long': self.spread_params.entry_zscore_long,
                'entry_zscore_short': self.spread_params.entry_zscore_short,
                'exit_zscore_lower': self.spread_params.exit_zscore_lower,
                'exit_zscore_upper': self.spread_params.exit_zscore_upper,
                'stop_loss_zscore': self.spread_params.stop_loss_zscore,
                'min_correlation': self.spread_params.min_correlation,
                'correlation_window': self.spread_params.correlation_window,
                'max_half_life': self.spread_params.max_half_life,
            },
            'risk_params': {
                'max_risk_per_trade': self.risk_params.max_risk_per_trade,
                'max_total_exposure': self.risk_params.max_total_exposure,
                'max_open_pairs': self.risk_params.max_open_pairs,
                'max_drawdown': self.risk_params.max_drawdown,
                'sizing_method': self.risk_params.sizing_method,
            },
            'backtest_params': {
                'spread_cost': self.backtest_params.spread_cost,
                'slippage': self.backtest_params.slippage,
                'commission_per_lot': self.backtest_params.commission_per_lot,
                'initial_capital': self.backtest_params.initial_capital,
            },
            'optimization_params': {
                'in_sample_period': self.optimization_params.in_sample_period,
                'out_sample_period': self.optimization_params.out_sample_period,
                'num_windows': self.optimization_params.num_windows,
                'optimization_metric': self.optimization_params.optimization_metric,
                'n_iterations': self.optimization_params.n_iterations,
                'reoptimization_frequency': self.optimization_params.reoptimization_frequency,
            },
            'lookback_days': self.lookback_days,
            'log_level': self.log_level,
        }
        
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    @classmethod
    def load_from_yaml(cls, filepath: Path) -> 'Settings':
        """Load settings from YAML file."""
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
        
        settings = cls()
        settings.mode = TradingMode(config.get('mode', 'backtest'))
        settings.timeframe = Timeframe(config.get('timeframe', 'H1'))
        settings.pairs_universe = [tuple(p) for p in config.get('pairs_universe', settings.pairs_universe)]
        
        if 'spread_params' in config:
            sp = config['spread_params']
            settings.spread_params = SpreadParameters(**sp)
        
        if 'risk_params' in config:
            rp = config['risk_params']
            settings.risk_params = RiskParameters(**rp)
        
        if 'backtest_params' in config:
            bp = config['backtest_params']
            settings.backtest_params = BacktestParameters(**bp)
        
        if 'optimization_params' in config:
            op = config['optimization_params']
            settings.optimization_params.in_sample_period = op.get('in_sample_period', settings.optimization_params.in_sample_period)
            settings.optimization_params.out_sample_period = op.get('out_sample_period', settings.optimization_params.out_sample_period)
            settings.optimization_params.num_windows = op.get('num_windows', settings.optimization_params.num_windows)
            settings.optimization_params.optimization_metric = op.get('optimization_metric', settings.optimization_params.optimization_metric)
            settings.optimization_params.n_iterations = op.get('n_iterations', settings.optimization_params.n_iterations)
            settings.optimization_params.reoptimization_frequency = op.get('reoptimization_frequency', settings.optimization_params.reoptimization_frequency)
        
        settings.lookback_days = config.get('lookback_days', settings.lookback_days)
        settings.log_level = config.get('log_level', settings.log_level)
        
        return settings


# Default settings instance
DEFAULT_SETTINGS = Settings()
