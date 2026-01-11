"""
FX Conditional Relative Value (CRV) System - Institutional Kill-Switch.

INSTITUTIONAL GRADE - AUDITABLE KILL-SWITCH MODULE

This module implements:
1. Explicit kill-switch with primary/secondary reasons
2. Safe equity initialization guards
3. Mode-aware drawdown evaluation
4. Full auditability

EVERY kill-switch activation MUST include:
- Primary Reason (ENUM)
- Secondary Reason (explicit condition)
- Mode (current system mode)
- Timestamp (UTC)

NO generic kill-switches allowed.
"""

from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Tuple
import logging

from src.crv.state_machine import SystemMode, CRVStateMachine

logger = logging.getLogger(__name__)


# ============================================================================
# EXCEPTIONS
# ============================================================================

class SystemLogicError(Exception):
    """
    Raised when system logic is violated.
    
    This indicates a BUG, not a runtime condition.
    Examples:
    - Kill-switch activated on invalid drawdown
    - FSM capability violation
    """
    pass


# ============================================================================
# KILL-SWITCH REASONS (ENUM - DO NOT EXTEND)
# ============================================================================

class KillSwitchPrimaryReason(Enum):
    """
    Primary reasons for kill-switch activation.
    
    ENUM - DO NOT EXTEND without institutional approval.
    """
    NONE = "none"
    DRAWDOWN_LIMIT = "drawdown_limit"
    DATA_INTEGRITY_FAILURE = "data_integrity_failure"
    EXPOSURE_LIMIT = "exposure_limit"
    SYSTEM_HEALTH_FAILURE = "system_health_failure"
    PRE_TRADE_VALIDATION = "pre_trade_validation"
    MANUAL_OVERRIDE = "manual_override"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    REGIME_VIOLATION = "regime_violation"


# ============================================================================
# KILL-SWITCH STATE
# ============================================================================

@dataclass
class KillSwitchState:
    """
    Complete kill-switch state with auditability.
    
    EVERY activation must have:
    - primary_reason: Enum value
    - secondary_reason: Explicit condition string
    - mode: System mode at activation
    - timestamp: UTC timestamp
    """
    is_active: bool
    primary_reason: KillSwitchPrimaryReason
    secondary_reason: str
    mode: SystemMode
    timestamp: datetime
    
    # Additional context
    drawdown_at_trigger: Optional[float] = None
    equity_at_trigger: Optional[float] = None
    
    def to_audit_string(self) -> str:
        """Generate audit-compliant string representation."""
        return (
            f"Kill-Switch: {'ACTIVE' if self.is_active else 'INACTIVE'}\n"
            f"Primary Reason: {self.primary_reason.value}\n"
            f"Secondary Reason: {self.secondary_reason}\n"
            f"Mode: {self.mode.value}\n"
            f"Timestamp: {self.timestamp.isoformat()}Z"
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "is_active": self.is_active,
            "primary_reason": self.primary_reason.value,
            "secondary_reason": self.secondary_reason,
            "mode": self.mode.value,
            "timestamp": self.timestamp.isoformat(),
            "drawdown_at_trigger": self.drawdown_at_trigger,
            "equity_at_trigger": self.equity_at_trigger,
        }


# ============================================================================
# EQUITY STATE WITH SAFE INITIALIZATION
# ============================================================================

@dataclass
class EquityState:
    """
    Equity state with institutional-safe initialization.
    
    CRITICAL: Drawdown is only valid when equity is properly initialized.
    """
    current_equity: float
    equity_peak: float
    
    # Validity flags
    equity_initialized: bool = False
    drawdown_valid: bool = False
    
    # Calculated values
    drawdown: float = 0.0
    drawdown_pct: float = 0.0
    
    # History
    last_update: Optional[datetime] = None
    
    @classmethod
    def create_uninitialized(cls) -> 'EquityState':
        """Create uninitialized equity state."""
        return cls(
            current_equity=0.0,
            equity_peak=0.0,
            equity_initialized=False,
            drawdown_valid=False,
            drawdown=0.0,
            drawdown_pct=0.0,
        )
    
    @classmethod
    def create_safe(cls, equity: float) -> 'EquityState':
        """
        Create equity state with safe initialization.
        
        SAFE INIT GUARD:
        - If equity <= 0, state is invalid
        - Drawdown is 0.0 but marked as invalid
        """
        if equity is None or equity <= 0:
            return cls(
                current_equity=0.0,
                equity_peak=0.0,
                equity_initialized=False,
                drawdown_valid=False,
                drawdown=0.0,
                drawdown_pct=0.0,
                last_update=datetime.utcnow(),
            )
        
        return cls(
            current_equity=equity,
            equity_peak=equity,
            equity_initialized=True,
            drawdown_valid=True,
            drawdown=0.0,
            drawdown_pct=0.0,
            last_update=datetime.utcnow(),
        )
    
    def update(self, new_equity: float) -> None:
        """
        Update equity and recalculate drawdown.
        
        SAFE UPDATE:
        - If not initialized, initialize now
        - If equity <= 0, mark as invalid
        """
        self.last_update = datetime.utcnow()
        
        if new_equity is None or new_equity <= 0:
            # Invalid equity - mark drawdown as invalid
            self.drawdown_valid = False
            return
        
        # Initialize if needed
        if not self.equity_initialized:
            self.current_equity = new_equity
            self.equity_peak = new_equity
            self.equity_initialized = True
            self.drawdown_valid = True
            self.drawdown = 0.0
            self.drawdown_pct = 0.0
            return
        
        # Update equity
        self.current_equity = new_equity
        
        # Update peak if new high
        if new_equity > self.equity_peak:
            self.equity_peak = new_equity
        
        # Calculate drawdown
        if self.equity_peak > 0:
            self.drawdown = self.equity_peak - self.current_equity
            self.drawdown_pct = self.drawdown / self.equity_peak
            self.drawdown_valid = True
        else:
            self.drawdown = 0.0
            self.drawdown_pct = 0.0
            self.drawdown_valid = False


# ============================================================================
# INSTITUTIONAL KILL-SWITCH MANAGER
# ============================================================================

class InstitutionalKillSwitch:
    """
    Institutional-grade kill-switch manager.
    
    CRITICAL FEATURES:
    1. Every activation has explicit primary + secondary reason
    2. Mode-aware - respects FSM capabilities
    3. Safe equity initialization
    4. Full audit trail
    
    RULES:
    - Kill-switch MUST NOT trigger on invalid drawdown
    - Kill-switch MUST NOT trigger in MODE_LIVE_CHECK for drawdown
    - Every activation MUST be logged with full context
    """
    
    def __init__(
        self,
        fsm: CRVStateMachine,
        # Thresholds
        warning_drawdown_pct: float = 3.0,
        max_drawdown_pct: float = 5.0,
        kill_switch_drawdown_pct: float = 8.0,
        max_exposure_pct: float = 15.0,
        max_consecutive_losses: int = 3,
    ):
        self.fsm = fsm
        
        # Thresholds (as fractions, not percentages)
        self.warning_drawdown = warning_drawdown_pct / 100.0
        self.max_drawdown = max_drawdown_pct / 100.0
        self.kill_switch_drawdown = kill_switch_drawdown_pct / 100.0
        self.max_exposure = max_exposure_pct / 100.0
        self.max_consecutive_losses = max_consecutive_losses
        
        # State
        self._equity_state = EquityState.create_uninitialized()
        self._kill_switch_state = self._create_inactive_state()
        self._activation_history: List[KillSwitchState] = []
        
        # Counters
        self._consecutive_losses = 0
        self._current_exposure = 0.0
    
    def _create_inactive_state(self) -> KillSwitchState:
        """Create inactive kill-switch state."""
        return KillSwitchState(
            is_active=False,
            primary_reason=KillSwitchPrimaryReason.NONE,
            secondary_reason="Kill-switch inactive",
            mode=self.fsm.mode if self.fsm.is_initialized else SystemMode.MODE_BACKTEST,
            timestamp=datetime.utcnow(),
        )
    
    @property
    def is_active(self) -> bool:
        """Check if kill-switch is active."""
        return self._kill_switch_state.is_active
    
    @property
    def equity_state(self) -> EquityState:
        """Get current equity state."""
        return self._equity_state
    
    @property
    def kill_switch_state(self) -> KillSwitchState:
        """Get current kill-switch state."""
        return self._kill_switch_state
    
    def initialize_equity(self, equity: float) -> bool:
        """
        Initialize equity tracking.
        
        Returns:
            True if successfully initialized, False if equity invalid
        """
        self._equity_state = EquityState.create_safe(equity)
        
        if self._equity_state.equity_initialized:
            logger.info(f"Equity initialized: {equity:.2f}")
            return True
        else:
            logger.warning(f"Equity initialization failed: {equity}")
            return False
    
    def update_equity(self, equity: float) -> None:
        """Update current equity."""
        self._equity_state.update(equity)
    
    def check_and_update(
        self,
        equity: Optional[float] = None,
        exposure: Optional[float] = None,
        consecutive_losses: Optional[int] = None,
    ) -> KillSwitchState:
        """
        Check all kill-switch conditions and update state.
        
        CRITICAL: FSM-GATED EVALUATION.
        
        FSM AUTHORITY:
        - If can_evaluate_drawdown() == False → SKIP drawdown completely
        - If can_place_orders() == False → Kill-switch cannot activate
        
        Returns:
            Current kill-switch state
        """
        # Update values if provided
        if equity is not None:
            # Only update equity if FSM permits
            if self.fsm.can_evaluate_drawdown():
                self.update_equity(equity)
            else:
                logger.debug("[SKIPPED] equity_update — FSM forbids drawdown evaluation")
        
        if exposure is not None:
            self._current_exposure = exposure
        
        if consecutive_losses is not None:
            self._consecutive_losses = consecutive_losses
        
        # If already killed, stay killed
        if self._kill_switch_state.is_active:
            return self._kill_switch_state
        
        # =====================================================================
        # FSM AUTHORITY CHECK - CRITICAL
        # =====================================================================
        # In observational modes (LIVE_CHECK), kill-switch CANNOT activate
        # based on drawdown or execution-related conditions
        # =====================================================================
        
        if not self.fsm.can_place_orders():
            # In non-execution modes, kill-switch is DISABLED for most conditions
            logger.debug("[FSM] Kill-switch evaluation limited — execution disabled")
            
            # Only data integrity failures can trigger in observational modes
            # All other conditions are SKIPPED
            return self._kill_switch_state
        
        # =====================================================================
        # FULL EVALUATION (Only in execution-enabled modes)
        # =====================================================================
        
        # 1. Drawdown check (FSM-GATED)
        if self.fsm.can_evaluate_drawdown():
            triggered, reason = self._check_drawdown()
            if triggered:
                self._activate(
                    primary=KillSwitchPrimaryReason.DRAWDOWN_LIMIT,
                    secondary=reason
                )
                return self._kill_switch_state
        else:
            logger.debug("[SKIPPED] drawdown_check — FSM forbids evaluation")
        
        # 2. Exposure check
        triggered, reason = self._check_exposure()
        if triggered:
            self._activate(
                primary=KillSwitchPrimaryReason.EXPOSURE_LIMIT,
                secondary=reason
            )
            return self._kill_switch_state
        
        # 3. Consecutive losses check
        triggered, reason = self._check_consecutive_losses()
        if triggered:
            self._activate(
                primary=KillSwitchPrimaryReason.SYSTEM_HEALTH_FAILURE,
                secondary=reason
            )
            return self._kill_switch_state
        
        return self._kill_switch_state
    
    def _check_drawdown(self) -> Tuple[bool, str]:
        """
        Check drawdown condition.
        
        CRITICAL SAFE INIT GUARD:
        - If drawdown_valid is False, CANNOT trigger
        - If equity not initialized, CANNOT trigger
        """
        # Safe init guard
        if not self._equity_state.equity_initialized:
            return False, "equity_not_initialized"
        
        if not self._equity_state.drawdown_valid:
            return False, "drawdown_invalid"
        
        # Check threshold
        if self._equity_state.drawdown_pct >= self.kill_switch_drawdown:
            return True, f"drawdown_{self._equity_state.drawdown_pct*100:.1f}pct_exceeds_{self.kill_switch_drawdown*100:.1f}pct_limit"
        
        return False, "drawdown_within_limits"
    
    def _check_exposure(self) -> Tuple[bool, str]:
        """Check exposure limit."""
        if self._current_exposure >= self.max_exposure:
            return True, f"exposure_{self._current_exposure*100:.1f}pct_exceeds_{self.max_exposure*100:.1f}pct_limit"
        return False, "exposure_within_limits"
    
    def _check_consecutive_losses(self) -> Tuple[bool, str]:
        """Check consecutive losses."""
        if self._consecutive_losses >= self.max_consecutive_losses:
            return True, f"{self._consecutive_losses}_consecutive_losses_exceeds_{self.max_consecutive_losses}_limit"
        return False, "consecutive_losses_within_limits"
    
    def _activate(
        self,
        primary: KillSwitchPrimaryReason,
        secondary: str,
    ) -> None:
        """
        Activate kill-switch with full audit context.
        
        MANDATORY: Primary + Secondary reason required.
        
        GUARDRAIL: Validates that activation is legal per FSM.
        """
        # =====================================================================
        # ILLEGAL ACTIVATION GUARDRAIL
        # =====================================================================
        # If activating for drawdown_limit but drawdown is invalid,
        # this is an ILLEGAL activation and indicates a system logic error.
        # =====================================================================
        
        if primary == KillSwitchPrimaryReason.DRAWDOWN_LIMIT:
            if not self._equity_state.drawdown_valid:
                error_msg = (
                    f"ILLEGAL KILL-SWITCH ACTIVATION: "
                    f"Attempted to activate for {primary.value} "
                    f"but drawdown_valid={self._equity_state.drawdown_valid}. "
                    f"FSM Mode: {self.fsm.mode.value}"
                )
                logger.critical(error_msg)
                raise SystemLogicError(error_msg)
            
            if not self.fsm.can_evaluate_drawdown():
                error_msg = (
                    f"ILLEGAL KILL-SWITCH ACTIVATION: "
                    f"Attempted to activate for {primary.value} "
                    f"but FSM forbids drawdown evaluation. "
                    f"FSM Mode: {self.fsm.mode.value}"
                )
                logger.critical(error_msg)
                raise SystemLogicError(error_msg)
        
        # =====================================================================
        # EXECUTION-RELATED ACTIVATION GUARDRAIL
        # =====================================================================
        
        if primary in [
            KillSwitchPrimaryReason.EXPOSURE_LIMIT,
            KillSwitchPrimaryReason.SYSTEM_HEALTH_FAILURE,
        ]:
            if not self.fsm.can_place_orders():
                error_msg = (
                    f"ILLEGAL KILL-SWITCH ACTIVATION: "
                    f"Attempted to activate for {primary.value} "
                    f"but FSM forbids order placement (observational mode). "
                    f"FSM Mode: {self.fsm.mode.value}"
                )
                logger.critical(error_msg)
                raise SystemLogicError(error_msg)
        
        # =====================================================================
        # LEGAL ACTIVATION - PROCEED
        # =====================================================================
        
        self._kill_switch_state = KillSwitchState(
            is_active=True,
            primary_reason=primary,
            secondary_reason=secondary,
            mode=self.fsm.mode,
            timestamp=datetime.utcnow(),
            drawdown_at_trigger=self._equity_state.drawdown_pct if self._equity_state.drawdown_valid else None,
            equity_at_trigger=self._equity_state.current_equity if self._equity_state.equity_initialized else None,
        )
        
        # Log activation with full context
        logger.critical("=" * 60)
        logger.critical("KILL-SWITCH ACTIVATED")
        logger.critical("=" * 60)
        logger.critical(self._kill_switch_state.to_audit_string())
        logger.critical("=" * 60)
        
        # Record in history
        self._activation_history.append(self._kill_switch_state)
    
    def activate_manual(self, reason: str) -> None:
        """Activate kill-switch manually."""
        self._activate(
            primary=KillSwitchPrimaryReason.MANUAL_OVERRIDE,
            secondary=f"manual_override: {reason}"
        )
    
    def activate_data_integrity(self, reason: str) -> None:
        """Activate kill-switch for data integrity failure."""
        self._activate(
            primary=KillSwitchPrimaryReason.DATA_INTEGRITY_FAILURE,
            secondary=f"data_integrity: {reason}"
        )
    
    def activate_pre_trade_validation(self, reason: str) -> None:
        """Activate kill-switch for pre-trade validation failure."""
        self._activate(
            primary=KillSwitchPrimaryReason.PRE_TRADE_VALIDATION,
            secondary=f"pre_trade_validation: {reason}"
        )
    
    def activate_correlation_breakdown(self, pair: str, correlation: float) -> None:
        """Activate kill-switch for correlation breakdown."""
        self._activate(
            primary=KillSwitchPrimaryReason.CORRELATION_BREAKDOWN,
            secondary=f"pair_{pair}_correlation_{correlation:.2f}_breakdown"
        )
    
    def activate_regime_violation(self, regime: str) -> None:
        """Activate kill-switch for regime violation."""
        self._activate(
            primary=KillSwitchPrimaryReason.REGIME_VIOLATION,
            secondary=f"regime_{regime}_violates_crv_conditions"
        )
    
    def reset(self, manual_confirmation: bool = True) -> bool:
        """
        Reset kill-switch.
        
        REQUIRES manual confirmation.
        Only allowed if conditions have improved.
        """
        if not manual_confirmation:
            logger.warning("Kill-switch reset requires manual confirmation")
            return False
        
        # Check conditions before reset
        if self.fsm.can_evaluate_drawdown():
            if self._equity_state.drawdown_valid:
                if self._equity_state.drawdown_pct >= self.max_drawdown:
                    logger.warning(
                        f"Cannot reset kill-switch: drawdown still at "
                        f"{self._equity_state.drawdown_pct*100:.1f}%"
                    )
                    return False
        
        # Reset
        self._kill_switch_state = self._create_inactive_state()
        self._consecutive_losses = 0
        
        logger.info("Kill-switch reset successfully")
        return True
    
    def get_status(self) -> Dict:
        """Get complete status for display/logging."""
        return {
            "kill_switch": {
                "is_active": self._kill_switch_state.is_active,
                "primary_reason": self._kill_switch_state.primary_reason.value,
                "secondary_reason": self._kill_switch_state.secondary_reason,
                "mode": self._kill_switch_state.mode.value,
                "timestamp": self._kill_switch_state.timestamp.isoformat(),
            },
            "equity": {
                "current": self._equity_state.current_equity,
                "peak": self._equity_state.equity_peak,
                "drawdown_pct": self._equity_state.drawdown_pct * 100,
                "initialized": self._equity_state.equity_initialized,
                "drawdown_valid": self._equity_state.drawdown_valid,
            },
            "thresholds": {
                "warning_drawdown_pct": self.warning_drawdown * 100,
                "max_drawdown_pct": self.max_drawdown * 100,
                "kill_switch_drawdown_pct": self.kill_switch_drawdown * 100,
                "max_exposure_pct": self.max_exposure * 100,
            },
            "fsm": {
                "mode": self.fsm.mode.value,
                "can_evaluate_drawdown": self.fsm.can_evaluate_drawdown(),
            },
        }
    
    def get_activation_history(self) -> List[KillSwitchState]:
        """Get history of kill-switch activations."""
        return self._activation_history.copy()
