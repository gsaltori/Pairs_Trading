"""
Quick validation of portfolio engine fixes.
Run this BEFORE the full backtest to verify corrections work.
"""

import pandas as pd
import numpy as np
from portfolio_engine import PortfolioEngine, SanityCheckError, validate_backtest_results
from momentum_engine import MomentumSignal

def test_partial_sell_bug_fix():
    """Test that partial sells don't delete entire position."""
    print("=" * 60)
    print("TEST 1: Partial Sell Bug Fix")
    print("=" * 60)
    
    portfolio = PortfolioEngine(100_000)
    prices = {'SPY': 450.0, 'QQQ': 380.0, 'GLD': 180.0}
    date = pd.Timestamp('2024-01-15')
    
    # Buy positions
    portfolio._execute_buy('SPY', 100, prices, date)  # 100 shares @ ~$450
    portfolio._execute_buy('QQQ', 100, prices, date)
    
    initial_spy_shares = portfolio.positions['SPY'].shares
    initial_equity = portfolio.get_equity(prices)
    
    print(f"Initial SPY shares: {initial_spy_shares:.2f}")
    print(f"Initial equity: ${initial_equity:,.2f}")
    
    # Partial sell - only 20 shares
    trade = portfolio._execute_sell('SPY', 20, prices, date, is_full_close=False)
    
    # Verify position still exists with reduced shares
    assert 'SPY' in portfolio.positions, "FAIL: SPY position was deleted!"
    remaining_shares = portfolio.positions['SPY'].shares
    expected_remaining = initial_spy_shares - 20
    
    print(f"\nAfter selling 20 shares:")
    print(f"  Remaining SPY shares: {remaining_shares:.2f} (expected: {expected_remaining:.2f})")
    print(f"  Trade PnL: ${trade.pnl:.2f}")
    print(f"  Current equity: ${portfolio.get_equity(prices):,.2f}")
    
    assert abs(remaining_shares - expected_remaining) < 0.01, f"FAIL: Expected {expected_remaining}, got {remaining_shares}"
    assert trade.pnl is not None and not np.isnan(trade.pnl), "FAIL: Trade PnL is NaN"
    
    print("\n✅ TEST 1 PASSED: Partial sells work correctly")
    return True


def test_pnl_calculation():
    """Test that PnL is calculated proportionally for partial sells."""
    print("\n" + "=" * 60)
    print("TEST 2: PnL Calculation")
    print("=" * 60)
    
    portfolio = PortfolioEngine(100_000)
    
    # Buy at $100
    buy_prices = {'TEST': 100.0}
    date1 = pd.Timestamp('2024-01-01')
    portfolio._execute_buy('TEST', 100, buy_prices, date1)  # 100 shares @ $100 = $10,000
    
    entry_value = portfolio.positions['TEST'].entry_value
    print(f"Bought 100 shares at ~$100")
    print(f"Entry value: ${entry_value:,.2f}")
    
    # Price goes up to $110
    sell_prices = {'TEST': 110.0}
    date2 = pd.Timestamp('2024-02-01')
    
    # Sell 50 shares (half the position)
    trade = portfolio._execute_sell('TEST', 50, sell_prices, date2, is_full_close=False)
    
    # Expected PnL for 50 shares:
    # Cost basis for 50 shares ≈ $5,000 (half of $10,000)
    # Proceeds ≈ 50 * $110 * (1 - slippage) - commission ≈ $5,495
    # Expected PnL ≈ $5,495 - $5,000 = ~$495 (positive!)
    
    print(f"\nSold 50 shares at ~$110:")
    print(f"  Trade PnL: ${trade.pnl:,.2f}")
    print(f"  Trade PnL %: {trade.pnl_pct:.2%}")
    
    # The OLD buggy code would have calculated:
    # PnL = $5,495 - $10,000 = -$4,505 (MASSIVELY WRONG!)
    
    # PnL should be positive since we sold at a higher price
    assert trade.pnl > 0, f"FAIL: PnL should be positive, got {trade.pnl:.2f}"
    assert trade.pnl < 600, f"FAIL: PnL too high, got {trade.pnl:.2f}"  # Sanity check
    
    print("\n✅ TEST 2 PASSED: PnL calculated correctly for partial sells")
    return True


def test_equity_never_negative():
    """Test that equity stays positive through rebalancing."""
    print("\n" + "=" * 60)
    print("TEST 3: Equity Stays Positive")
    print("=" * 60)
    
    portfolio = PortfolioEngine(100_000)
    prices = {'SPY': 450.0, 'QQQ': 380.0, 'GLD': 180.0}
    
    # Simulate multiple rebalances
    for i in range(5):
        date = pd.Timestamp(f'2024-{i+1:02d}-28')
        
        # Create a signal that changes positions
        if i % 2 == 0:
            weights = {'SPY': 0.33, 'QQQ': 0.33, 'GLD': 0.33}
        else:
            weights = {'SPY': 0.50, 'QQQ': 0.50}
        
        from momentum_engine import MomentumSignal
        signal = MomentumSignal(
            date=date,
            rankings={'SPY': 0.1, 'QQQ': 0.05, 'GLD': 0.02},
            selected=list(weights.keys()),
            weights=weights,
            trend_filter={'SPY': True, 'QQQ': True, 'GLD': True},
            cash_weight=1.0 - sum(weights.values()),
        )
        
        # Execute rebalance
        actions = portfolio.calculate_rebalance_actions(signal, prices)
        portfolio.execute_rebalance(actions, prices)
        portfolio.update_equity_curve(date, prices)
        
        equity = portfolio.get_equity(prices)
        print(f"  Rebalance {i+1}: Equity = ${equity:,.2f}, Positions = {list(portfolio.positions.keys())}")
        
        assert equity > 0, f"FAIL: Equity went negative: {equity}"
    
    print("\n✅ TEST 3 PASSED: Equity stayed positive through all rebalances")
    return True


def test_sanity_checks_catch_errors():
    """Test that sanity checks catch invalid states."""
    print("\n" + "=" * 60)
    print("TEST 4: Sanity Checks Work")
    print("=" * 60)
    
    portfolio = PortfolioEngine(100_000)
    prices = {'SPY': 450.0}
    
    # This should work
    portfolio.run_all_sanity_checks(prices, "initial")
    print("  Initial state: ✅")
    
    # Try to sell more shares than we have (should fail)
    portfolio._execute_buy('SPY', 10, prices, pd.Timestamp('2024-01-01'))
    
    try:
        portfolio._execute_sell('SPY', 100, prices, pd.Timestamp('2024-01-15'), is_full_close=False)
        print("  FAIL: Should have caught selling too many shares!")
        return False
    except SanityCheckError as e:
        print(f"  Caught invalid sell: ✅ ({str(e)[:50]}...)")
    
    print("\n✅ TEST 4 PASSED: Sanity checks catch errors")
    return True


def test_full_rebalance_cycle():
    """Test a complete rebalance cycle with validation."""
    print("\n" + "=" * 60)
    print("TEST 5: Full Rebalance Cycle")
    print("=" * 60)
    
    portfolio = PortfolioEngine(100_000)
    
    # Month 1: Enter SPY, QQQ, GLD
    prices1 = {'SPY': 450.0, 'QQQ': 380.0, 'GLD': 180.0}
    date1 = pd.Timestamp('2024-01-31')
    
    signal1 = MomentumSignal(
        date=date1,
        rankings={'SPY': 0.15, 'QQQ': 0.10, 'GLD': 0.05},
        selected=['SPY', 'QQQ', 'GLD'],
        weights={'SPY': 0.333, 'QQQ': 0.333, 'GLD': 0.333},
        trend_filter={'SPY': True, 'QQQ': True, 'GLD': True},
        cash_weight=0.001,
    )
    
    actions1 = portfolio.calculate_rebalance_actions(signal1, prices1)
    portfolio.execute_rebalance(actions1, prices1)
    portfolio.update_equity_curve(date1, prices1)
    
    eq1 = portfolio.get_equity(prices1)
    print(f"Month 1: Equity = ${eq1:,.2f}")
    print(f"  Positions: {list(portfolio.positions.keys())}")
    
    # Month 2: GLD drops out, IWM enters
    prices2 = {'SPY': 460.0, 'QQQ': 390.0, 'GLD': 170.0, 'IWM': 220.0}
    date2 = pd.Timestamp('2024-02-29')
    
    signal2 = MomentumSignal(
        date=date2,
        rankings={'SPY': 0.18, 'QQQ': 0.12, 'IWM': 0.08, 'GLD': 0.02},
        selected=['SPY', 'QQQ', 'IWM'],
        weights={'SPY': 0.333, 'QQQ': 0.333, 'IWM': 0.333},
        trend_filter={'SPY': True, 'QQQ': True, 'IWM': True, 'GLD': False},
        cash_weight=0.001,
    )
    
    actions2 = portfolio.calculate_rebalance_actions(signal2, prices2)
    portfolio.execute_rebalance(actions2, prices2)
    portfolio.update_equity_curve(date2, prices2)
    
    eq2 = portfolio.get_equity(prices2)
    print(f"Month 2: Equity = ${eq2:,.2f}")
    print(f"  Positions: {list(portfolio.positions.keys())}")
    print(f"  Trades: {len(portfolio.trades)}")
    
    # Validate final state
    valid, issues = validate_backtest_results(portfolio, prices2)
    
    if not valid:
        print(f"\n❌ Validation failed: {issues}")
        return False
    
    # Check that GLD was closed and IWM was opened
    assert 'GLD' not in portfolio.positions, "GLD should have been closed"
    assert 'IWM' in portfolio.positions, "IWM should have been opened"
    
    print("\n✅ TEST 5 PASSED: Full rebalance cycle works correctly")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("PORTFOLIO ENGINE VALIDATION TESTS")
    print("=" * 70)
    
    all_passed = True
    
    try:
        all_passed &= test_partial_sell_bug_fix()
        all_passed &= test_pnl_calculation()
        all_passed &= test_equity_never_negative()
        all_passed &= test_sanity_checks_catch_errors()
        all_passed &= test_full_rebalance_cycle()
    except Exception as e:
        print(f"\n❌ TEST FAILED WITH EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ ALL TESTS PASSED - Portfolio engine is CORRECTED")
        print("   You can now run: python backtest_runner.py")
    else:
        print("❌ SOME TESTS FAILED - Do not run backtest until fixed")
    print("=" * 70 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
