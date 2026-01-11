#!/usr/bin/env python
"""
Verify FSM Authority Fix - Quick Test.

This script verifies the FSM fix is working correctly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.crv import (
    CRVStateMachine,
    SystemMode,
    create_fsm,
    InstitutionalKillSwitch,
    KillSwitchPrimaryReason,
    SystemLogicError,
)


def test_mode_live_check():
    """Test MODE_LIVE_CHECK behavior."""
    print("=" * 60)
    print("TEST: MODE_LIVE_CHECK")
    print("=" * 60)
    
    fsm = create_fsm("live_check")
    
    print(f"\n✓ FSM Mode: {fsm.mode.value}")
    print(f"  Can Generate Signals: {fsm.can_generate_signals()}")
    print(f"  Can Evaluate Drawdown: {fsm.can_evaluate_drawdown()}")
    print(f"  Can Place Orders: {fsm.can_place_orders()}")
    print(f"  Requires Equity Init: {fsm.requires_equity_init()}")
    
    # Create kill-switch
    kill_switch = InstitutionalKillSwitch(
        fsm=fsm,
        kill_switch_drawdown_pct=8.0,
        max_exposure_pct=15.0,
        max_consecutive_losses=3,
    )
    
    # Try to trigger kill-switch with extreme conditions
    print("\n📊 Attempting to trigger kill-switch with extreme conditions...")
    print("   equity=50000 (50% drawdown if initialized at 100k)")
    print("   exposure=0.20 (20%)")
    print("   consecutive_losses=5")
    
    result = kill_switch.check_and_update(
        equity=50000,
        exposure=0.20,
        consecutive_losses=5,
    )
    
    # Verify kill-switch is NOT active
    if result.is_active:
        print(f"\n❌ FAIL: Kill-switch ILLEGALLY activated!")
        print(f"   Primary Reason: {result.primary_reason.value}")
        print(f"   Secondary Reason: {result.secondary_reason}")
        return False
    else:
        print(f"\n✓ PASS: Kill-switch correctly remains INACTIVE")
        print(f"   Primary Reason: {result.primary_reason.value}")
    
    # Verify equity was not initialized
    if kill_switch.equity_state.equity_initialized:
        print(f"\n❌ FAIL: Equity was ILLEGALLY initialized!")
        return False
    else:
        print(f"✓ PASS: Equity correctly NOT initialized")
    
    # Verify drawdown is not valid
    if kill_switch.equity_state.drawdown_valid:
        print(f"\n❌ FAIL: Drawdown marked as valid!")
        return False
    else:
        print(f"✓ PASS: Drawdown correctly marked as INVALID")
    
    return True


def test_mode_live_trading():
    """Test MODE_LIVE_TRADING behavior."""
    print("\n" + "=" * 60)
    print("TEST: MODE_LIVE_TRADING")
    print("=" * 60)
    
    fsm = create_fsm("live")
    
    print(f"\n✓ FSM Mode: {fsm.mode.value}")
    print(f"  Can Generate Signals: {fsm.can_generate_signals()}")
    print(f"  Can Evaluate Drawdown: {fsm.can_evaluate_drawdown()}")
    print(f"  Can Place Orders: {fsm.can_place_orders()}")
    
    # Create kill-switch
    kill_switch = InstitutionalKillSwitch(
        fsm=fsm,
        kill_switch_drawdown_pct=8.0,
    )
    
    # Initialize equity
    kill_switch.initialize_equity(100000)
    
    # Create 10% drawdown
    kill_switch.update_equity(90000)
    
    # Check kill-switch
    result = kill_switch.check_and_update()
    
    # Verify kill-switch IS active (10% > 8% threshold)
    if not result.is_active:
        print(f"\n❌ FAIL: Kill-switch should be active!")
        return False
    else:
        print(f"\n✓ PASS: Kill-switch correctly activated")
        print(f"   Primary Reason: {result.primary_reason.value}")
        print(f"   Secondary Reason: {result.secondary_reason}")
    
    return True


def test_illegal_activation_guard():
    """Test that illegal activation raises error."""
    print("\n" + "=" * 60)
    print("TEST: ILLEGAL ACTIVATION GUARDRAIL")
    print("=" * 60)
    
    fsm = create_fsm("live_check")
    kill_switch = InstitutionalKillSwitch(fsm=fsm)
    
    print("\n📊 Attempting illegal activation...")
    
    try:
        kill_switch._activate(
            primary=KillSwitchPrimaryReason.DRAWDOWN_LIMIT,
            secondary="test_illegal"
        )
        print(f"\n❌ FAIL: Illegal activation was NOT blocked!")
        return False
    except SystemLogicError as e:
        print(f"\n✓ PASS: Illegal activation correctly blocked")
        print(f"   Error: {str(e)[:80]}...")
        return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("FX CRV SYSTEM - FSM AUTHORITY VERIFICATION")
    print("=" * 60)
    
    results = []
    
    results.append(("MODE_LIVE_CHECK", test_mode_live_check()))
    results.append(("MODE_LIVE_TRADING", test_mode_live_trading()))
    results.append(("ILLEGAL_ACTIVATION_GUARD", test_illegal_activation_guard()))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_pass = True
    for name, passed in results:
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_pass = False
    
    if all_pass:
        print("\n🟢 ALL TESTS PASSED - FSM Authority is ABSOLUTE")
        return 0
    else:
        print("\n🔴 SOME TESTS FAILED - System is INVALID")
        return 1


if __name__ == "__main__":
    sys.exit(main())
