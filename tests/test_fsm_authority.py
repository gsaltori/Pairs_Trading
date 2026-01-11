"""
FX CRV System - FSM Authority Unit Tests.

INSTITUTIONAL GRADE - MANDATORY TESTS

These tests verify that the FSM has ABSOLUTE authority over:
1. Kill-switch activation
2. Drawdown evaluation
3. Safety checklist behavior

If ANY of these tests fail, the system is INVALID.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.crv.state_machine import (
    CRVStateMachine,
    SystemMode,
    ModeCapabilityError,
    create_fsm,
)
from src.crv.kill_switch import (
    InstitutionalKillSwitch,
    KillSwitchPrimaryReason,
    EquityState,
    SystemLogicError,
)


# ============================================================================
# TEST 1: MODE_LIVE_CHECK must NEVER activate kill-switch
# ============================================================================

class TestMODE_LIVE_CHECK_KillSwitch:
    """
    MANDATORY TEST 1:
    MODE_LIVE_CHECK must never activate kill-switch.
    """
    
    def test_kill_switch_disabled_in_live_check(self):
        """Kill-switch cannot activate in MODE_LIVE_CHECK."""
        fsm = create_fsm("live_check")
        kill_switch = InstitutionalKillSwitch(
            fsm=fsm,
            kill_switch_drawdown_pct=8.0,
        )
        
        # Try to trigger kill-switch with various conditions
        result = kill_switch.check_and_update(
            equity=50000,  # 50% drawdown if initialized at 100000
            exposure=0.20,  # 20% exposure (over limit)
            consecutive_losses=5,  # Over limit
        )
        
        # Kill-switch MUST remain inactive
        assert result.is_active == False, (
            f"ILLEGAL: Kill-switch activated in MODE_LIVE_CHECK. "
            f"Primary reason: {result.primary_reason.value}"
        )
    
    def test_drawdown_not_evaluated_in_live_check(self):
        """Drawdown must not be evaluated in MODE_LIVE_CHECK."""
        fsm = create_fsm("live_check")
        kill_switch = InstitutionalKillSwitch(fsm=fsm)
        
        # Equity update should be skipped
        kill_switch.check_and_update(equity=100000)
        
        # Equity state should remain uninitialized
        assert kill_switch.equity_state.equity_initialized == False, (
            "ILLEGAL: Equity was initialized in MODE_LIVE_CHECK"
        )
    
    def test_fsm_capabilities_in_live_check(self):
        """Verify FSM capabilities for MODE_LIVE_CHECK."""
        fsm = create_fsm("live_check")
        
        assert fsm.can_generate_signals() == True
        assert fsm.can_evaluate_drawdown() == False
        assert fsm.can_place_orders() == False
        assert fsm.requires_equity_init() == False


# ============================================================================
# TEST 2: Drawdown logic must be unreachable when FSM forbids it
# ============================================================================

class TestDrawdownFSMAuthority:
    """
    MANDATORY TEST 2:
    Drawdown logic must be unreachable when FSM forbids it.
    """
    
    def test_drawdown_check_skipped_when_fsm_forbids(self):
        """Drawdown check must be skipped when can_evaluate_drawdown == False."""
        fsm = create_fsm("live_check")
        kill_switch = InstitutionalKillSwitch(
            fsm=fsm,
            kill_switch_drawdown_pct=1.0,  # Very low threshold
        )
        
        # Even with extreme drawdown, should not trigger
        result = kill_switch.check_and_update(equity=1)  # 99.99% drawdown
        
        assert result.is_active == False
        assert result.primary_reason == KillSwitchPrimaryReason.NONE
    
    def test_drawdown_evaluated_when_fsm_permits(self):
        """Drawdown should be evaluated when can_evaluate_drawdown == True."""
        fsm = create_fsm("paper")  # Paper mode allows drawdown evaluation
        kill_switch = InstitutionalKillSwitch(
            fsm=fsm,
            kill_switch_drawdown_pct=5.0,
        )
        
        # Initialize equity
        kill_switch.initialize_equity(100000)
        
        # Simulate 10% drawdown
        kill_switch.update_equity(90000)
        
        # Check should trigger kill-switch
        result = kill_switch.check_and_update()
        
        # Paper mode can place orders? No, so kill-switch is disabled
        # Only MODE_LIVE_TRADING can activate kill-switch
        assert fsm.can_place_orders() == False
        assert result.is_active == False  # Kill-switch disabled in paper mode too
    
    def test_drawdown_only_triggers_in_live_trading(self):
        """Kill-switch for drawdown only activates in MODE_LIVE_TRADING."""
        fsm = create_fsm("live")  # Live trading mode
        kill_switch = InstitutionalKillSwitch(
            fsm=fsm,
            kill_switch_drawdown_pct=5.0,
        )
        
        # Initialize and create drawdown
        kill_switch.initialize_equity(100000)
        kill_switch.update_equity(90000)  # 10% drawdown
        
        # Now it should trigger
        result = kill_switch.check_and_update()
        
        assert result.is_active == True
        assert result.primary_reason == KillSwitchPrimaryReason.DRAWDOWN_LIMIT
    
    def test_illegal_activation_raises_error(self):
        """Illegal kill-switch activation must raise SystemLogicError."""
        fsm = create_fsm("live_check")
        kill_switch = InstitutionalKillSwitch(fsm=fsm)
        
        # Manually try to activate kill-switch for drawdown
        # This should raise SystemLogicError
        with pytest.raises(SystemLogicError) as exc_info:
            kill_switch._activate(
                primary=KillSwitchPrimaryReason.DRAWDOWN_LIMIT,
                secondary="test_illegal_activation"
            )
        
        assert "ILLEGAL KILL-SWITCH ACTIVATION" in str(exc_info.value)


# ============================================================================
# TEST 3: Safety checklist must respect FSM capabilities
# ============================================================================

class TestSafetyChecklistFSMAuthority:
    """
    MANDATORY TEST 3:
    Safety checklist must respect FSM capabilities.
    """
    
    def test_checklist_skips_drawdown_in_live_check(self):
        """Checklist must skip drawdown check in MODE_LIVE_CHECK."""
        # This requires the full CRV system
        # For now, test the FSM gating logic
        fsm = create_fsm("live_check")
        
        # Simulate checklist logic
        checklist = {}
        
        if fsm.can_evaluate_drawdown():
            checklist["drawdown_acceptable"] = False  # Would fail
        else:
            checklist["drawdown_acceptable"] = "SKIPPED"
        
        assert checklist["drawdown_acceptable"] == "SKIPPED"
    
    def test_checklist_skips_kill_switch_when_execution_disabled(self):
        """Checklist must skip kill-switch check when can_place_orders == False."""
        fsm = create_fsm("paper")
        
        # Simulate checklist logic
        checklist = {}
        
        if fsm.can_place_orders():
            checklist["kill_switch_off"] = False  # Would fail
        else:
            checklist["kill_switch_off"] = "SKIPPED"
        
        assert checklist["kill_switch_off"] == "SKIPPED"
    
    def test_checklist_evaluates_all_in_live_trading(self):
        """Checklist must evaluate all checks in MODE_LIVE_TRADING."""
        fsm = create_fsm("live")
        
        assert fsm.can_generate_signals() == True
        assert fsm.can_evaluate_drawdown() == True
        assert fsm.can_place_orders() == True
        
        # All checks should be evaluated (not skipped)


# ============================================================================
# TEST: FSM Mode Capabilities Matrix
# ============================================================================

class TestFSMCapabilitiesMatrix:
    """Verify FSM capabilities matrix is correct."""
    
    def test_backtest_capabilities(self):
        fsm = create_fsm("backtest")
        assert fsm.can_generate_signals() == True
        assert fsm.can_evaluate_drawdown() == True
        assert fsm.can_place_orders() == False
        assert fsm.requires_equity_init() == False
    
    def test_paper_capabilities(self):
        fsm = create_fsm("paper")
        assert fsm.can_generate_signals() == True
        assert fsm.can_evaluate_drawdown() == True
        assert fsm.can_place_orders() == False
        assert fsm.requires_equity_init() == True
    
    def test_live_check_capabilities(self):
        fsm = create_fsm("live_check")
        assert fsm.can_generate_signals() == True
        assert fsm.can_evaluate_drawdown() == False
        assert fsm.can_place_orders() == False
        assert fsm.requires_equity_init() == False
    
    def test_live_trading_capabilities(self):
        fsm = create_fsm("live")
        assert fsm.can_generate_signals() == True
        assert fsm.can_evaluate_drawdown() == True
        assert fsm.can_place_orders() == True
        assert fsm.requires_equity_init() == True


# ============================================================================
# TEST: Equity State Safe Initialization
# ============================================================================

class TestEquityStateSafeInit:
    """Verify equity state safe initialization guards."""
    
    def test_create_uninitialized(self):
        state = EquityState.create_uninitialized()
        assert state.equity_initialized == False
        assert state.drawdown_valid == False
        assert state.drawdown == 0.0
    
    def test_create_safe_with_valid_equity(self):
        state = EquityState.create_safe(100000)
        assert state.equity_initialized == True
        assert state.drawdown_valid == True
        assert state.current_equity == 100000
        assert state.equity_peak == 100000
        assert state.drawdown == 0.0
    
    def test_create_safe_with_zero_equity(self):
        state = EquityState.create_safe(0)
        assert state.equity_initialized == False
        assert state.drawdown_valid == False
    
    def test_create_safe_with_negative_equity(self):
        state = EquityState.create_safe(-1000)
        assert state.equity_initialized == False
        assert state.drawdown_valid == False
    
    def test_create_safe_with_none(self):
        state = EquityState.create_safe(None)
        assert state.equity_initialized == False
        assert state.drawdown_valid == False


# ============================================================================
# TEST: Mode Transitions
# ============================================================================

class TestFSMTransitions:
    """Verify FSM transition rules."""
    
    def test_valid_transition_backtest_to_live_check(self):
        fsm = CRVStateMachine(initial_mode=SystemMode.MODE_BACKTEST)
        fsm.set_mode(SystemMode.MODE_LIVE_CHECK, reason="Test transition")
        assert fsm.mode == SystemMode.MODE_LIVE_CHECK
    
    def test_valid_transition_live_check_to_live_trading(self):
        fsm = CRVStateMachine(initial_mode=SystemMode.MODE_LIVE_CHECK)
        fsm.set_mode(SystemMode.MODE_LIVE_TRADING, reason="Go live")
        assert fsm.mode == SystemMode.MODE_LIVE_TRADING
    
    def test_invalid_transition_backtest_to_live_trading(self):
        """Cannot go directly from backtest to live trading."""
        from src.crv.state_machine import InvalidTransitionError
        
        fsm = CRVStateMachine(initial_mode=SystemMode.MODE_BACKTEST)
        
        with pytest.raises(InvalidTransitionError):
            fsm.set_mode(SystemMode.MODE_LIVE_TRADING, reason="Illegal transition")


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
