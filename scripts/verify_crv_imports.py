#!/usr/bin/env python
"""
CRV System v2.0 - Import Verification Script.

Run this script to verify all modules are correctly installed.

Usage:
    python scripts/verify_crv_imports.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_import(module_name: str, items: list) -> bool:
    """Test importing items from a module."""
    try:
        module = __import__(module_name, fromlist=items)
        for item in items:
            getattr(module, item)
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def main():
    print("=" * 60)
    print("FX CRV SYSTEM v2.0 - IMPORT VERIFICATION")
    print("=" * 60)
    
    all_ok = True
    
    # Layer 0: Data Integrity
    print("\n[Layer 0] Data Integrity...")
    if test_import('src.crv.data_integrity', [
        'FXDataValidator', 'AlignedDataset', 'DataQualityReport',
        'safe_returns', 'safe_rolling_correlation', 'safe_zscore'
    ]):
        print("  ✓ PASS")
    else:
        print("  ✗ FAIL")
        all_ok = False
    
    # Layer 1: Pair Selector
    print("\n[Layer 1] Pair Selector...")
    if test_import('src.crv.pair_selector', [
        'FXStructuralPairSelector', 'StructuralPairAssessment',
        'FX_PAIR_RELATIONSHIPS', 'PairRelationship',
        'assess_conditional_correlation', 'assess_operational_viability'
    ]):
        print("  ✓ PASS")
    else:
        print("  ✗ FAIL")
        all_ok = False
    
    # Layer 2: Regime Filter
    print("\n[Layer 2] Regime Filter...")
    if test_import('src.crv.regime_filter', [
        'FXRegimeFilter', 'FXRegimeAssessment', 'FXRegime',
        'REGIME_PERMITS_CRV', 'RiskSentiment'
    ]):
        print("  ✓ PASS")
    else:
        print("  ✗ FAIL")
        all_ok = False
    
    # Layer 3: Conditional Spread
    print("\n[Layer 3] Conditional Spread...")
    if test_import('src.crv.conditional_spread', [
        'ConditionalSpreadAnalyzer', 'ConditionalSpreadData',
        'RegimeStatistics', 'safe_pct_change'
    ]):
        print("  ✓ PASS")
    else:
        print("  ✗ FAIL")
        all_ok = False
    
    # Layer 4-5: Execution Safety
    print("\n[Layer 4-5] Execution Safety...")
    if test_import('src.crv.execution_safety', [
        'ExecutionSafetyManager', 'CRVSignalEngine',
        'CRVSignal', 'SignalType', 'RiskState',
        'ExecutionConstraints', 'KillSwitchReason'
    ]):
        print("  ✓ PASS")
    else:
        print("  ✗ FAIL")
        all_ok = False
    
    # Main System
    print("\n[Main] CRV System...")
    if test_import('src.crv.crv_system', [
        'FXConditionalRelativeValueSystem',
        'CRVSystemState', 'CRVAnalysisResult'
    ]):
        print("  ✓ PASS")
    else:
        print("  ✗ FAIL")
        all_ok = False
    
    # Package __init__
    print("\n[Package] src.crv exports...")
    if test_import('src.crv', [
        'FXDataValidator',
        'FXStructuralPairSelector',
        'FXRegimeFilter',
        'ConditionalSpreadAnalyzer',
        'CRVSignalEngine',
        'ExecutionSafetyManager',
        'FXConditionalRelativeValueSystem',
    ]):
        print("  ✓ PASS")
    else:
        print("  ✗ FAIL")
        all_ok = False
    
    # Summary
    print("\n" + "=" * 60)
    if all_ok:
        print("✓ ALL IMPORTS SUCCESSFUL")
        print("  System is ready to use.")
    else:
        print("✗ SOME IMPORTS FAILED")
        print("  Please check the errors above.")
    print("=" * 60)
    
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
