"""
FX Conditional Relative Value (CRV) System - Formal State Machine.

INSTITUTIONAL GRADE - FSM MODULE

This module implements the Finite State Machine that controls
all system behavior and gates functionality based on mode.

Modes:
    MODE_BACKTEST      - Historical analysis, no execution
    MODE_PAPER         - Paper trading with full risk checks
    MODE_LIVE_CHECK    - Pre-live validation, no risk evaluation
    MODE_LIVE_TRADING  - Full live execution enabled

FSM Rules:
    - Mode must be known at all times
    - Mode gates risk checks, drawdown, and execution
    - Invalid transitions raise HARD ERRORS
    - All transitions are logged
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Set, Callable
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# SYSTEM MODES (EXACT NAMES - DO NOT MODIFY)
# ============================================================================

class SystemMode(Enum):
    """
    Finite State Machine modes for CRV system.
    
    EXACT NAMES - DO NOT EXTEND OR MODIFY.
    """
    MODE_BACKTEST = "MODE_BACKTEST"
    MODE_PAPER = "MODE_PAPER"
    MODE_LIVE_CHECK = "MODE_LIVE_CHECK"
    MODE_LIVE_TRADING = "MODE_LIVE_TRADING"


# ============================================================================
# MODE CAPABILITIES MATRIX
# ============================================================================

@dataclass(frozen=True)
class ModeCapabilities:
    """Defines what each mode can and cannot do."""
    can_generate_signals: bool
    can_evaluate_drawdown: bool
    can_place_orders: bool
    can_modify_positions: bool
    requires_equity_init: bool


# Capability matrix - IMMUTABLE
MODE_CAPABILITIES: Dict[SystemMode, ModeCapabilities] = {
    SystemMode.MODE_BACKTEST: ModeCapabilities(
        can_generate_signals=True,
        can_evaluate_drawdown=True,
        can_place_orders=False,
        can_modify_positions=False,
        requires_equity_init=False,
    ),
    SystemMode.MODE_PAPER: ModeCapabilities(
        can_generate_signals=True,
        can_evaluate_drawdown=True,
        can_place_orders=False,
        can_modify_positions=False,
        requires_equity_init=True,
    ),
    SystemMode.MODE_LIVE_CHECK: ModeCapabilities(
        can_generate_signals=True,
        can_evaluate_drawdown=False,  # CRITICAL: No drawdown eval in live check
        can_place_orders=False,
        can_modify_positions=False,
        requires_equity_init=False,
    ),
    SystemMode.MODE_LIVE_TRADING: ModeCapabilities(
        can_generate_signals=True,
        can_evaluate_drawdown=True,
        can_place_orders=True,
        can_modify_positions=True,
        requires_equity_init=True,
    ),
}


# Valid mode transitions
VALID_TRANSITIONS: Dict[SystemMode, Set[SystemMode]] = {
    SystemMode.MODE_BACKTEST: {
        SystemMode.MODE_PAPER,
        SystemMode.MODE_LIVE_CHECK,
    },
    SystemMode.MODE_PAPER: {
        SystemMode.MODE_BACKTEST,
        SystemMode.MODE_LIVE_CHECK,
    },
    SystemMode.MODE_LIVE_CHECK: {
        SystemMode.MODE_PAPER,
        SystemMode.MODE_LIVE_TRADING,  # Only from live check to live trading
        SystemMode.MODE_BACKTEST,
    },
    SystemMode.MODE_LIVE_TRADING: {
        SystemMode.MODE_PAPER,  # Can go back to paper
        SystemMode.MODE_LIVE_CHECK,  # Can go back to check
    },
}


# ============================================================================
# FSM ERRORS
# ============================================================================

class FSMError(Exception):
    """Base exception for FSM errors."""
    pass


class InvalidModeError(FSMError):
    """Raised when mode is invalid or unknown."""
    pass


class InvalidTransitionError(FSMError):
    """Raised when attempting an invalid mode transition."""
    pass


class ModeCapabilityError(FSMError):
    """Raised when attempting an action not allowed in current mode."""
    pass


class ModeNotInitializedError(FSMError):
    """Raised when system is used before mode is set."""
    pass


# ============================================================================
# FSM STATE MACHINE
# ============================================================================

@dataclass
class FSMTransition:
    """Record of a mode transition."""
    from_mode: Optional[SystemMode]
    to_mode: SystemMode
    timestamp: datetime
    reason: str


class CRVStateMachine:
    """
    Finite State Machine for CRV System.
    
    Controls all mode-dependent behavior and ensures
    invalid operations are blocked.
    
    CRITICAL RULES:
    1. Mode MUST be set before any operation
    2. Invalid transitions raise HARD errors
    3. All transitions are logged
    4. Capabilities are strictly enforced
    """
    
    def __init__(self, initial_mode: Optional[SystemMode] = None):
        self._mode: Optional[SystemMode] = None
        self._transition_history: List[FSMTransition] = []
        self._initialized: bool = False
        
        if initial_mode is not None:
            self.set_mode(initial_mode, reason="Initial mode set")
    
    @property
    def mode(self) -> SystemMode:
        """Get current mode. Raises if not initialized."""
        if self._mode is None:
            raise ModeNotInitializedError(
                "System mode not initialized. Call set_mode() first."
            )
        return self._mode
    
    @property
    def is_initialized(self) -> bool:
        """Check if mode has been set."""
        return self._mode is not None
    
    @property
    def capabilities(self) -> ModeCapabilities:
        """Get current mode capabilities."""
        return MODE_CAPABILITIES[self.mode]
    
    def set_mode(self, new_mode: SystemMode, reason: str = "Mode change") -> None:
        """
        Set or transition to a new mode.
        
        Args:
            new_mode: Target mode
            reason: Reason for transition (logged)
            
        Raises:
            InvalidModeError: If mode is not valid
            InvalidTransitionError: If transition is not allowed
        """
        # Validate mode
        if not isinstance(new_mode, SystemMode):
            raise InvalidModeError(f"Invalid mode: {new_mode}")
        
        # Check transition validity
        if self._mode is not None:
            if new_mode not in VALID_TRANSITIONS.get(self._mode, set()):
                raise InvalidTransitionError(
                    f"Invalid transition: {self._mode.value} → {new_mode.value}. "
                    f"Valid transitions: {[m.value for m in VALID_TRANSITIONS.get(self._mode, set())]}"
                )
        
        # Record transition
        transition = FSMTransition(
            from_mode=self._mode,
            to_mode=new_mode,
            timestamp=datetime.utcnow(),
            reason=reason
        )
        self._transition_history.append(transition)
        
        # Log transition
        if self._mode is None:
            logger.info(f"FSM INITIALIZED: {new_mode.value} | Reason: {reason}")
        else:
            logger.info(f"FSM TRANSITION: {self._mode.value} → {new_mode.value} | Reason: {reason}")
        
        self._mode = new_mode
        self._initialized = True
    
    def can_generate_signals(self) -> bool:
        """Check if current mode allows signal generation."""
        return self.capabilities.can_generate_signals
    
    def can_evaluate_drawdown(self) -> bool:
        """Check if current mode allows drawdown evaluation."""
        return self.capabilities.can_evaluate_drawdown
    
    def can_place_orders(self) -> bool:
        """Check if current mode allows order placement."""
        return self.capabilities.can_place_orders
    
    def can_modify_positions(self) -> bool:
        """Check if current mode allows position modification."""
        return self.capabilities.can_modify_positions
    
    def requires_equity_init(self) -> bool:
        """Check if current mode requires equity initialization."""
        return self.capabilities.requires_equity_init
    
    def assert_can_generate_signals(self) -> None:
        """Assert signal generation is allowed. Raises if not."""
        if not self.can_generate_signals():
            raise ModeCapabilityError(
                f"Signal generation not allowed in {self.mode.value}"
            )
    
    def assert_can_evaluate_drawdown(self) -> None:
        """Assert drawdown evaluation is allowed. Raises if not."""
        if not self.can_evaluate_drawdown():
            raise ModeCapabilityError(
                f"Drawdown evaluation not allowed in {self.mode.value}"
            )
    
    def assert_can_place_orders(self) -> None:
        """Assert order placement is allowed. Raises if not."""
        if not self.can_place_orders():
            raise ModeCapabilityError(
                f"Order placement not allowed in {self.mode.value}. "
                f"Only allowed in MODE_LIVE_TRADING."
            )
    
    def get_mode_status(self) -> Dict:
        """Get current mode status for logging/display."""
        return {
            "mode": self.mode.value,
            "initialized": self._initialized,
            "can_generate_signals": self.can_generate_signals(),
            "can_evaluate_drawdown": self.can_evaluate_drawdown(),
            "can_place_orders": self.can_place_orders(),
            "can_modify_positions": self.can_modify_positions(),
            "requires_equity_init": self.requires_equity_init(),
            "transition_count": len(self._transition_history),
        }
    
    def get_transition_history(self) -> List[FSMTransition]:
        """Get history of mode transitions."""
        return self._transition_history.copy()
    
    def log_startup(self) -> None:
        """Log startup information. Called once at system init."""
        status = self.get_mode_status()
        
        logger.info("=" * 60)
        logger.info("FX CRV SYSTEM - FINITE STATE MACHINE STATUS")
        logger.info("=" * 60)
        logger.info(f"  Mode: {status['mode']}")
        logger.info(f"  Can Generate Signals: {status['can_generate_signals']}")
        logger.info(f"  Can Evaluate Drawdown: {status['can_evaluate_drawdown']}")
        logger.info(f"  Can Place Orders: {status['can_place_orders']}")
        logger.info(f"  Requires Equity Init: {status['requires_equity_init']}")
        logger.info("=" * 60)


# ============================================================================
# FSM DECORATORS FOR CAPABILITY ENFORCEMENT
# ============================================================================

def requires_signal_capability(func: Callable) -> Callable:
    """Decorator to enforce signal generation capability."""
    def wrapper(self, *args, **kwargs):
        if hasattr(self, 'fsm'):
            self.fsm.assert_can_generate_signals()
        return func(self, *args, **kwargs)
    return wrapper


def requires_drawdown_capability(func: Callable) -> Callable:
    """Decorator to enforce drawdown evaluation capability."""
    def wrapper(self, *args, **kwargs):
        if hasattr(self, 'fsm'):
            self.fsm.assert_can_evaluate_drawdown()
        return func(self, *args, **kwargs)
    return wrapper


def requires_order_capability(func: Callable) -> Callable:
    """Decorator to enforce order placement capability."""
    def wrapper(self, *args, **kwargs):
        if hasattr(self, 'fsm'):
            self.fsm.assert_can_place_orders()
        return func(self, *args, **kwargs)
    return wrapper


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_fsm(mode: str) -> CRVStateMachine:
    """
    Create FSM from mode string.
    
    Args:
        mode: Mode string (e.g., "backtest", "paper", "live_check", "live")
        
    Returns:
        Initialized CRVStateMachine
    """
    mode_map = {
        "backtest": SystemMode.MODE_BACKTEST,
        "paper": SystemMode.MODE_PAPER,
        "live_check": SystemMode.MODE_LIVE_CHECK,
        "live": SystemMode.MODE_LIVE_TRADING,
        "live_trading": SystemMode.MODE_LIVE_TRADING,
    }
    
    mode_lower = mode.lower().replace("-", "_")
    
    if mode_lower not in mode_map:
        raise InvalidModeError(
            f"Unknown mode: {mode}. Valid modes: {list(mode_map.keys())}"
        )
    
    fsm = CRVStateMachine(initial_mode=mode_map[mode_lower])
    fsm.log_startup()
    
    return fsm
