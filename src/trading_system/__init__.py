"""
Trading System - Production FX Trading Framework

CURRENT SYSTEM: Micro-Edge Portfolio
=====================================
Session Directional Bias Engine (filter)
+ Complementary high-frequency strategies
+ Portfolio-level risk management
+ Gatekeeper integration

COMPONENTS:
1. Bias Engine - Converts session analysis to directional filter
2. Micro Strategies - High-frequency, small-target strategies
3. Portfolio Risk - Correlation control, kill switches
4. Gatekeeper - Structural failure filter (unchanged)

SAFETY: System defaults to DRY RUN mode.
"""

from .config import (
    SystemConfig,
    StrategyConfig,
    GatekeeperConfig,
    RiskConfig,
    ExecutionConfig,
    PathConfig,
    SymbolConfig,
    DEFAULT_CONFIG,
)

# Session Directional Bias Engine
from .bias_engine import (
    SessionBiasEngine,
    SessionBiasOutput,
    DirectionalBias,
    BiasState,
    TradingSession,
)

# Micro-Edge Strategies
from .micro_strategies import (
    MicroEdgePortfolio,
    MicroSignal,
    MicroEdgeDirection,
    LondonPullbackScalper,
    MomentumBurstStrategy,
    PivotBounceStrategy,
    BaseMicroStrategy,
    PortfolioSignal,
)

# Portfolio Risk Management
from .portfolio_risk import (
    PortfolioRiskManager,
    PortfolioRiskState,
    DailyPortfolioStats,
    KillSwitchReason,
)

# Exit Models (for potential future use)
from .exit_models import (
    ExitModel,
    ExitConfig,
    ExitManager,
    PositionSlice,
)

# Gatekeeper (shared across all strategies)
from .gatekeeper_engine import (
    GatekeeperEngine,
    GatekeeperDecision,
    BlockReason,
)

# Trade-Level Risk Management
from .risk_engine import (
    RiskEngine,
    RiskState,
    RiskLevel,
    TradePermission,
    OpenPosition,
    DailyStats,
)

# Execution
from .execution_engine import (
    ExecutionEngine,
    OrderExecution,
    OrderResult,
    Position,
)


__all__ = [
    # Config
    'SystemConfig',
    'StrategyConfig',
    'GatekeeperConfig',
    'RiskConfig',
    'ExecutionConfig',
    'PathConfig',
    'SymbolConfig',
    'DEFAULT_CONFIG',
    
    # Bias Engine
    'SessionBiasEngine',
    'SessionBiasOutput',
    'DirectionalBias',
    'BiasState',
    'TradingSession',
    
    # Micro Strategies
    'MicroEdgePortfolio',
    'MicroSignal',
    'MicroEdgeDirection',
    'LondonPullbackScalper',
    'MomentumBurstStrategy',
    'PivotBounceStrategy',
    'BaseMicroStrategy',
    'PortfolioSignal',
    
    # Portfolio Risk
    'PortfolioRiskManager',
    'PortfolioRiskState',
    'DailyPortfolioStats',
    'KillSwitchReason',
    
    # Exit Models
    'ExitModel',
    'ExitConfig',
    'ExitManager',
    'PositionSlice',
    
    # Gatekeeper
    'GatekeeperEngine',
    'GatekeeperDecision',
    'BlockReason',
    
    # Risk
    'RiskEngine',
    'RiskState',
    'RiskLevel',
    'TradePermission',
    'OpenPosition',
    'DailyStats',
    
    # Execution
    'ExecutionEngine',
    'OrderExecution',
    'OrderResult',
    'Position',
]
