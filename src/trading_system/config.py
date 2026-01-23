"""
Production Trading System Configuration

Single source of truth for all system parameters.
DO NOT modify thresholds - they are empirically validated.
"""

from dataclasses import dataclass, field
from typing import Dict
from pathlib import Path


@dataclass(frozen=True)
class SymbolConfig:
    """Symbol-specific configuration."""
    name: str
    pip_value: float  # Value per pip per standard lot
    min_lot: float
    max_lot: float
    lot_step: float


@dataclass(frozen=True)
class StrategyConfig:
    """Strategy parameters - LOCKED, DO NOT MODIFY."""
    # Trend filter
    ema_fast: int = 50
    ema_slow: int = 200
    
    # Entry/Exit
    atr_period: int = 14
    atr_multiplier: float = 1.5
    risk_reward: float = 2.0
    
    # Timeframe
    timeframe: str = "H4"
    
    # Symbols
    primary_symbol: str = "EURUSD"
    secondary_symbol: str = "GBPUSD"  # For gatekeeper observables


@dataclass(frozen=True)
class GatekeeperConfig:
    """Gatekeeper thresholds - LOCKED, EMPIRICALLY VALIDATED."""
    # Block if |zscore| > this
    zscore_extreme_threshold: float = 3.0
    
    # Block if correlation_trend < this
    correlation_deteriorating_threshold: float = -0.05
    
    # Block if volatility_ratio < this
    volatility_compressed_threshold: float = 0.7
    
    # Observable windows
    zscore_window: int = 60
    correlation_window: int = 60
    volatility_window: int = 20


@dataclass(frozen=True)
class RiskConfig:
    """Risk management parameters - CONSERVATIVE."""
    # Per-trade risk
    risk_per_trade_normal: float = 0.005      # 0.5% normal
    risk_per_trade_reduced: float = 0.0025    # 0.25% reduced (after -3% DD)
    
    # Position limits
    max_concurrent_trades: int = 2
    max_concurrent_risk: float = 0.01         # 1% max total exposure
    
    # Drawdown governors (from equity high water mark)
    dd_reduce_risk_threshold: float = 0.03    # -3% → reduce risk 50%
    dd_single_trade_threshold: float = 0.06   # -6% → max 1 trade
    dd_halt_threshold: float = 0.08           # -8% → system halt
    dd_manual_review_threshold: float = 0.10  # -10% → manual review required
    
    # Recovery
    dd_recovery_buffer: float = 0.01          # Must recover 1% before resuming normal
    
    # Anti-revenge trading
    min_bars_between_trades: int = 2          # No immediate re-entry
    max_losses_per_day: int = 3               # Stop trading after 3 losses in a day
    
    # Position sizing
    max_position_size: float = 1.0            # Max 1 lot per trade
    min_position_size: float = 0.01           # Min 0.01 lot


@dataclass(frozen=True)
class ExecutionConfig:
    """Execution parameters."""
    # Order settings
    slippage_points: int = 30                 # Max acceptable slippage
    magic_number: int = 20260117              # Unique identifier for our orders
    order_comment: str = "TREND_GATED_SYS"
    
    # Retry logic
    max_order_retries: int = 3
    retry_delay_seconds: float = 1.0
    
    # Safety
    require_sl: bool = True                   # Never place order without SL
    require_tp: bool = True                   # Never place order without TP


@dataclass(frozen=True)
class PathConfig:
    """File paths for persistence and logging."""
    base_dir: Path = Path("trading_system_data")
    
    @property
    def logs_dir(self) -> Path:
        return self.base_dir / "logs"
    
    @property
    def state_dir(self) -> Path:
        return self.base_dir / "state"
    
    @property
    def data_dir(self) -> Path:
        return self.base_dir / "data"
    
    @property
    def trade_log(self) -> Path:
        return self.logs_dir / "trades.csv"
    
    @property
    def block_log(self) -> Path:
        return self.logs_dir / "blocks.csv"
    
    @property
    def risk_log(self) -> Path:
        return self.logs_dir / "risk_state.csv"
    
    @property
    def system_log(self) -> Path:
        return self.logs_dir / "system.log"
    
    @property
    def state_file(self) -> Path:
        return self.state_dir / "system_state.json"
    
    @property
    def equity_history(self) -> Path:
        return self.state_dir / "equity_history.json"


@dataclass
class SystemConfig:
    """Master configuration - aggregates all configs."""
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    gatekeeper: GatekeeperConfig = field(default_factory=GatekeeperConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    
    # Symbol definitions
    symbols: Dict[str, SymbolConfig] = field(default_factory=lambda: {
        "EURUSD": SymbolConfig(
            name="EURUSD",
            pip_value=10.0,  # $10 per pip per standard lot
            min_lot=0.01,
            max_lot=100.0,
            lot_step=0.01,
        ),
        "GBPUSD": SymbolConfig(
            name="GBPUSD",
            pip_value=10.0,
            min_lot=0.01,
            max_lot=100.0,
            lot_step=0.01,
        ),
    })
    
    # System behavior
    dry_run: bool = True                      # SAFETY: Start in dry-run mode
    verbose: bool = True
    
    def ensure_directories(self) -> None:
        """Create required directories."""
        self.paths.logs_dir.mkdir(parents=True, exist_ok=True)
        self.paths.state_dir.mkdir(parents=True, exist_ok=True)
        self.paths.data_dir.mkdir(parents=True, exist_ok=True)


# Default configuration instance
DEFAULT_CONFIG = SystemConfig()
