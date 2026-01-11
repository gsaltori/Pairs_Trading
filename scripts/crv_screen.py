#!/usr/bin/env python
"""
FX Conditional Relative Value (CRV) Screening Script - v2.1 INSTITUTIONAL.

INSTITUTIONAL GRADE SCANNER with FSM

This script runs the complete CRV analysis with proper state management:
1. FSM: Finite State Machine mode control
2. Layer 0: Data validation and integrity
3. Layer 1: FX-native structural pair selection
4. Layer 2: Regime assessment
5. Layer 3: Conditional spread analysis
6. Layer 4: Signal generation
7. Layer 5: Execution safety validation with institutional kill-switch

MODES:
    --mode backtest     : Historical analysis (default)
    --mode paper        : Paper trading simulation
    --mode live_check   : Pre-live validation (NO drawdown eval)
    --mode live         : Live trading (NOT RECOMMENDED without MT5)

Philosophy:
    - This is NOT Statistical Arbitrage
    - FX does NOT exhibit permanent mean reversion
    - We trade CONDITIONAL relative value
    - Inactivity is correct when edge is absent
    - SAFETY > PROFIT
    - Zero trades is a SUCCESS state

Usage:
    python scripts/crv_screen.py --timeframe H4 --mode backtest
    python scripts/crv_screen.py --timeframe H4 --mode live_check --save
"""

import sys
from pathlib import Path
from datetime import datetime
import argparse
import logging
import pandas as pd
import numpy as np
import json
import warnings

# Suppress FutureWarnings (we've fixed them but some libs may still emit)
warnings.filterwarnings('ignore', category=FutureWarning)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.crv import (
    # FSM
    CRVStateMachine,
    SystemMode,
    create_fsm,
    
    # Institutional Kill-Switch
    InstitutionalKillSwitch,
    KillSwitchPrimaryReason,
    EquityState,
    SystemLogicError,
    
    # Main System
    FXConditionalRelativeValueSystem,
    
    # Data Integrity
    FXDataValidator,
    safe_returns,
    
    # Layer 1
    FX_PAIR_RELATIONSHIPS,
    PairRelationship,
    
    # Layer 2
    FXRegime,
    REGIME_PERMITS_CRV,
    
    # Layer 5
    ExecutionConstraints,
)


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# FX Universe - 28 Major Pairs
FX_UNIVERSE = [
    # Majors
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "NZDUSD", "USDCAD",
    # Yen crosses
    "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY",
    # Euro crosses
    "EURGBP", "EURAUD", "EURNZD", "EURCHF", "EURCAD",
    # GBP crosses
    "GBPAUD", "GBPNZD", "GBPCAD", "GBPCHF",
    # Commodity crosses
    "AUDNZD", "AUDCAD", "AUDCHF", "NZDCAD", "NZDCHF", "CADCHF"
]


def load_ohlc_data(symbol: str, data_dir: Path, timeframe: str = "H4") -> pd.DataFrame:
    """Load OHLC data from local parquet file."""
    pattern = f"{symbol}_{timeframe}_*.parquet"
    files = list(data_dir.glob(pattern))
    
    if not files:
        return None
    
    filepath = max(files, key=lambda f: f.stat().st_mtime)
    df = pd.read_parquet(filepath)
    
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'time' in df.columns:
            df.set_index('time', inplace=True)
        df.index = pd.to_datetime(df.index)
    
    df.columns = df.columns.str.lower()
    
    return df


def create_dummy_regime_history(price_series: pd.Series) -> pd.Series:
    """
    Create dummy regime history for backtesting.
    
    In production, this would come from actual regime classification
    stored historically.
    """
    # Use safe_returns to avoid FutureWarning
    returns = safe_returns(price_series)
    
    if len(returns) == 0:
        return pd.Series(dtype=str)
    
    # Rolling volatility (on aligned index)
    rolling_vol = returns.rolling(20, min_periods=10).std()
    
    # Fill early NaN with first valid value
    rolling_vol = rolling_vol.fillna(method='bfill')
    
    # Percentile rank
    vol_percentile = rolling_vol.rank(pct=True)
    
    # Create regime series
    regime = pd.Series(index=returns.index, dtype=str)
    regime[vol_percentile <= 0.25] = FXRegime.STABLE_LOW_VOL.value
    regime[(vol_percentile > 0.25) & (vol_percentile <= 0.50)] = FXRegime.STABLE_NORMAL_VOL.value
    regime[(vol_percentile > 0.50) & (vol_percentile <= 0.75)] = FXRegime.RANGE_BOUND.value
    regime[vol_percentile > 0.75] = FXRegime.HIGH_VOLATILITY.value
    
    regime = regime.fillna(FXRegime.UNKNOWN.value)
    
    # Reindex to match original price series
    regime = regime.reindex(price_series.index, method='ffill')
    regime = regime.fillna(FXRegime.UNKNOWN.value)
    
    return regime


def main():
    parser = argparse.ArgumentParser(
        description='FX Conditional Relative Value (CRV) Screening - v2.1 INSTITUTIONAL'
    )
    
    parser.add_argument(
        '--mode', type=str, default='backtest',
        choices=['backtest', 'paper', 'live_check', 'live'],
        help='System mode (backtest, paper, live_check, live)'
    )
    parser.add_argument(
        '--symbols', type=str,
        help='Comma-separated symbols'
    )
    parser.add_argument(
        '--data-dir', type=str, default='data/historical',
        help='Data directory'
    )
    parser.add_argument(
        '--timeframe', type=str, default='H4',
        choices=['H1', 'H4', 'D1'],
        help='Timeframe (H4 or D1 recommended for CRV)'
    )
    parser.add_argument(
        '--equity', type=float, default=100000,
        help='Account equity for position sizing'
    )
    parser.add_argument(
        '--structural-only', action='store_true',
        help='Only show structural pair analysis'
    )
    parser.add_argument(
        '--save', action='store_true',
        help='Save results to file'
    )
    
    args = parser.parse_args()
    
    # =========================================================================
    # FSM INITIALIZATION
    # =========================================================================
    print("=" * 80)
    print("FX CONDITIONAL RELATIVE VALUE (CRV) SYSTEM - v2.1 INSTITUTIONAL")
    print("=" * 80)
    
    # Create FSM with specified mode
    try:
        fsm = create_fsm(args.mode)
    except Exception as e:
        print(f"\nFATAL: Failed to initialize FSM: {e}")
        return 1
    
    print(f"\n🔧 FSM MODE: {fsm.mode.value}")
    print(f"   Can Generate Signals: {fsm.can_generate_signals()}")
    print(f"   Can Evaluate Drawdown: {fsm.can_evaluate_drawdown()}")
    print(f"   Can Place Orders: {fsm.can_place_orders()}")
    
    # =========================================================================
    # INSTITUTIONAL KILL-SWITCH INITIALIZATION
    # =========================================================================
    kill_switch = InstitutionalKillSwitch(
        fsm=fsm,
        warning_drawdown_pct=3.0,
        max_drawdown_pct=5.0,
        kill_switch_drawdown_pct=8.0,
        max_exposure_pct=15.0,
        max_consecutive_losses=3,
    )
    
    # Initialize equity (MODE-AWARE)
    if fsm.requires_equity_init():
        if not kill_switch.initialize_equity(args.equity):
            print(f"\n⚠️ EQUITY INITIALIZATION FAILED")
            kill_switch.activate_pre_trade_validation("equity_not_initialized")
    else:
        print(f"\n📊 Mode {fsm.mode.value} does not require equity initialization")
    
    # Check kill-switch status
    ks_status = kill_switch.get_status()
    if ks_status['kill_switch']['is_active']:
        print(f"\n{'=' * 80}")
        print("🚨 KILL-SWITCH ACTIVE")
        print(f"   Primary Reason: {ks_status['kill_switch']['primary_reason']}")
        print(f"   Secondary Reason: {ks_status['kill_switch']['secondary_reason']}")
        print(f"={'=' * 80}")
        if args.mode != 'live_check':
            return 1
    
    print("\n⚠️  THIS IS NOT STATISTICAL ARBITRAGE")
    print("    FX does NOT exhibit permanent mean reversion.")
    print("    We trade CONDITIONAL relative value only.")
    print("    The system is designed to NOT TRADE most of the time.")
    print("    SAFETY > PROFIT")
    print("    Zero trades is a SUCCESS state.")
    
    # Check data directory
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        print(f"\nERROR: Data directory not found: {data_dir}")
        print("\nDownload data first:")
        print("  python scripts/download_data.py --forex --timeframes H1,H4,D1 --days 730")
        return 1

    # Show macro relationships summary
    print(f"\n{'─' * 80}")
    print("FX MACRO COHERENT RELATIONSHIPS")
    print(f"{'─' * 80}")
    
    # Group by relationship type
    by_type = {}
    for pair, (rel, score) in FX_PAIR_RELATIONSHIPS.items():
        if rel.value not in by_type:
            by_type[rel.value] = []
        by_type[rel.value].append((pair, score))
    
    for rel_type, pairs in sorted(by_type.items()):
        count = len(pairs)
        top_pairs = sorted(pairs, key=lambda x: -x[1])[:3]
        top_str = ", ".join([f"{p[0]}/{p[1]}" for p in top_pairs])
        print(f"  {rel_type}: {count} pairs (top: {top_str})")
    
    # Load data
    print(f"\n{'─' * 80}")
    print("LAYER 0: DATA LOADING & VALIDATION")
    print(f"{'─' * 80}")
    
    symbols = args.symbols.split(',') if args.symbols else FX_UNIVERSE
    symbols = [s.strip().upper() for s in symbols]
    
    ohlc_data = {}
    price_data = {}
    loaded = 0
    failed = 0
    
    for symbol in symbols:
        df = load_ohlc_data(symbol, data_dir, args.timeframe)
        if df is not None and len(df) >= 500:
            ohlc_data[symbol] = df
            price_data[symbol] = df['close']
            loaded += 1
        else:
            failed += 1
    
    print(f"  ✓ Loaded: {loaded} symbols")
    print(f"  ✗ Failed: {failed} symbols")
    
    if len(price_data) < 2:
        print("\nERROR: Need at least 2 symbols")
        return 1
    
    # Initialize CRV system with execution constraints
    print(f"\n{'─' * 80}")
    print("INITIALIZING CRV SYSTEM (HARDENED)")
    print(f"{'─' * 80}")
    
    constraints = ExecutionConstraints(
        max_positions=3,
        max_exposure_pct=10.0,
        max_position_size_pct=3.0,
        max_currency_exposure_pct=15.0,
        warning_drawdown_pct=3.0,
        max_drawdown_pct=5.0,
        kill_switch_drawdown_pct=8.0,
        max_holding_bars=50,
        max_consecutive_losses=3,
    )
    
    crv_system = FXConditionalRelativeValueSystem(
        # Layer 1: FX-native pair selection
        min_macro_score=0.50,
        min_median_correlation=0.20,
        
        # Layer 2: Regime filter
        adx_threshold=25.0,
        vol_high_percentile=75.0,
        
        # Layer 3-4: Signals
        conditional_entry_z=1.5,
        conditional_exit_z=0.3,
        
        # Layer 5: Execution safety
        constraints=constraints,
    )
    
    print("  ✓ CRV System initialized")
    print(f"    Macro coherence threshold: 0.50")
    print(f"    Min median correlation: 0.20")
    print(f"    Entry Z: ±1.5 (conditional)")
    print(f"    Max positions: {constraints.max_positions}")
    print(f"    Kill-switch drawdown: {constraints.kill_switch_drawdown_pct}%")
    
    # Layer 0: Validate and align data
    print(f"\n{'─' * 80}")
    print("LAYER 0: DATA INTEGRITY VALIDATION")
    print(f"{'─' * 80}")
    
    aligned = crv_system.validate_and_align_data(
        price_data=price_data,
        ohlc_data=ohlc_data,
        timeframe=args.timeframe
    )
    
    print(f"  Valid symbols: {len(aligned.symbols)}")
    print(f"  Rejected symbols: {len(aligned.rejected_symbols)}")
    print(f"  Common bars: {aligned.n_bars}")
    
    if aligned.rejected_symbols:
        print(f"  Rejected: {', '.join(aligned.rejected_symbols[:5])}")
    
    # Use aligned data
    price_data = aligned.prices
    ohlc_data = aligned.ohlc or {}
    
    if len(price_data) < 2:
        print(f"\n{'=' * 80}")
        print("ERROR: Insufficient valid symbols after data validation.")
        print("All symbols were rejected due to data quality issues.")
        print("\nPossible causes:")
        print("  • Weekend gaps in FX data (expected, should be handled)")
        print("  • Missing data or corrupted files")
        print("  • Data too old or gaps too large")
        print("\nSolution: Re-download data with:")
        print("  python scripts/download_data.py --forex --timeframes H4 --days 730")
        print("=" * 80)
        return 1
    
    # Layer 1: Structural pair selection
    print(f"\n{'─' * 80}")
    print("LAYER 1: FX-NATIVE STRUCTURAL PAIR SELECTION")
    print(f"{'─' * 80}")
    print("\n  Selection criteria:")
    print("  • Macro coherence (50%)")
    print("  • Conditional correlation (30%)")
    print("  • Operational viability (20%)")
    print("  NO cointegration, NO stationarity tests")
    
    structural_pairs = crv_system.update_structural_pairs(price_data, ohlc_data)
    
    print(f"\n  Found {len(structural_pairs)} structurally valid pairs")
    
    # Group by tier
    for tier in ["A", "B", "C"]:
        tier_pairs = [sp for sp in structural_pairs if sp.tier == tier]
        if tier_pairs:
            print(f"\n  TIER {tier} ({len(tier_pairs)} pairs):")
            for sp in tier_pairs[:5]:
                rel = sp.relationship.value if sp.relationship else "derived"
                print(f"    ✓ {sp.pair[0]}/{sp.pair[1]}")
                print(f"      Macro: {rel} ({sp.macro_coherence_score:.2f})")
                print(f"      Corr: median={sp.correlation.median_correlation:.2f}, "
                      f"low_vol={sp.correlation.correlation_low_vol:.2f}")
                print(f"      Score: {sp.structural_score:.0f}/100")
    
    if args.structural_only:
        print(f"\n{'=' * 80}")
        return 0
    
    # Layer 2: Regime assessment
    print(f"\n{'─' * 80}")
    print("LAYER 2: REGIME ASSESSMENT")
    print(f"{'─' * 80}")
    
    # Use first available OHLC for regime detection
    reference_symbol = list(ohlc_data.keys())[0] if ohlc_data else list(price_data.keys())[0]
    reference_ohlc = ohlc_data.get(reference_symbol)
    
    if reference_ohlc is None:
        # Create dummy OHLC from prices
        reference_prices = price_data[reference_symbol]
        reference_ohlc = pd.DataFrame({
            'open': reference_prices,
            'high': reference_prices * 1.001,
            'low': reference_prices * 0.999,
            'close': reference_prices
        })
    
    # Get USDJPY and AUDJPY for sentiment
    usdjpy = price_data.get("USDJPY")
    audjpy = price_data.get("AUDJPY")
    
    regime_assessment = crv_system.update_regime(
        ohlc_data=reference_ohlc,
        usdjpy=usdjpy,
        audjpy=audjpy
    )
    
    regime_icon = "🟢" if regime_assessment.permits_crv else "🔴"
    print(f"\n  {regime_icon} Current Regime: {regime_assessment.regime.value}")
    print(f"  Permits CRV: {'YES' if regime_assessment.permits_crv else 'NO'}")
    print(f"  Confidence: {regime_assessment.confidence:.0%}")
    
    print(f"\n  Volatility:")
    print(f"    ATR Percentile: {regime_assessment.volatility.atr_percentile:.0f}")
    print(f"    Classification: {regime_assessment.volatility.vol_regime}")
    print(f"    Expanding: {'YES' if regime_assessment.volatility.is_expanding else 'NO'}")
    
    print(f"\n  Trend:")
    print(f"    ADX: {regime_assessment.trend.adx_value:.1f}")
    print(f"    Strength: {regime_assessment.trend.trend_strength}")
    print(f"    Bias: {regime_assessment.trend.directional_bias}")
    
    print(f"\n  Sentiment:")
    print(f"    Classification: {regime_assessment.sentiment.sentiment.value}")
    print(f"    Score: {regime_assessment.sentiment.sentiment_score:.0f}")
    
    if regime_assessment.blocking_reasons:
        print(f"\n  ⚠️ Blocking Reasons:")
        for reason in regime_assessment.blocking_reasons:
            print(f"    • {reason}")
    
    # Layer 3-4: Spread analysis and signals
    print(f"\n{'─' * 80}")
    print("LAYER 3-4: CONDITIONAL SPREAD ANALYSIS & SIGNALS")
    print(f"{'─' * 80}")
    
    # Create regime history
    regime_history = create_dummy_regime_history(price_data[reference_symbol])
    
    results = crv_system.analyze_all_pairs(
        price_data=price_data,
        ohlc_data=ohlc_data,
        regime_history=regime_history,
        current_equity=args.equity
    )
    
    # Tradeable signals
    tradeable = [r for r in results if r.is_tradeable]
    
    if tradeable:
        print(f"\n🚨 TRADEABLE SIGNALS: {len(tradeable)}")
        
        for r in tradeable:
            direction = r.signal.signal_type.value if r.signal else "unknown"
            print(f"\n  → {r.pair[0]}/{r.pair[1]}: {direction.upper()}")
            print(f"    Conditional Z: {r.spread_data.zscore_conditional:+.2f}")
            print(f"    Unconditional Z: {r.spread_data.zscore_unconditional:+.2f}")
            print(f"    Confidence: {r.signal.confidence:.0%}")
            print(f"    Size: {r.signal.suggested_size_pct:.1f}% equity")
            print(f"    Hedge Ratio: {r.spread_data.hedge_ratio:.4f}")
            print(f"    Entry Z: ±{r.signal.entry_z:.1f}")
            print(f"    Target Z: ±{r.signal.target_z:.1f}")
            print(f"    Stop Z: ±{r.signal.stop_z:.1f}")
    else:
        print("\n  ⏸️ NO TRADEABLE SIGNALS")
        print("    This is expected - CRV requires confluence of all conditions.")
    
    # Watchlist
    watchlist = [r for r in results if r.structural.is_structurally_valid and not r.is_tradeable and r.spread_data and r.spread_data.is_valid]
    
    if watchlist:
        print(f"\n{'─' * 80}")
        print("WATCHLIST (Approaching Entry)")
        print(f"{'─' * 80}")
        
        # Sort by absolute z-score
        watchlist.sort(key=lambda x: abs(x.spread_data.zscore_conditional), reverse=True)
        
        for r in watchlist[:5]:
            z = r.spread_data.zscore_conditional
            print(f"\n  • {r.pair[0]}/{r.pair[1]}")
            print(f"    Z: {z:+.2f} (need ±1.5 for entry)")
            print(f"    Status: {', '.join(r.status_notes[:2])}")
    
    # Layer 5: Safety check (MODE-AWARE)
    if args.mode == 'live_check':
        print(f"\n{'─' * 80}")
        print("LAYER 5: LIVE TRADING SAFETY CHECKLIST (FSM-AWARE)")
        print(f"{'─' * 80}")
        
        # Pass FSM to checklist for proper mode-aware evaluation
        checklist = crv_system.get_live_safety_checklist(fsm=fsm)
        
        # Count actual passes (not SKIPPED)
        actual_checks = {k: v for k, v in checklist.items() if v != "SKIPPED"}
        skipped_checks = {k: v for k, v in checklist.items() if v == "SKIPPED"}
        
        all_pass = all(v == True for v in actual_checks.values())
        
        # Display evaluated checks
        for check, result in checklist.items():
            if result == "SKIPPED":
                print(f"  [SKIP] {check}: SKIPPED (FSM)")
            elif result == True:
                print(f"  [✓] {check}: PASS")
            else:
                print(f"  [✗] {check}: FAIL")
        
        # Summary
        print(f"\n  FSM Mode: {fsm.mode.value}")
        print(f"  Can Evaluate Drawdown: {fsm.can_evaluate_drawdown()}")
        print(f"  Can Place Orders: {fsm.can_place_orders()}")
        
        if skipped_checks:
            print(f"\n  ⚠️ {len(skipped_checks)} checks SKIPPED by FSM (observational mode)")
        
        # Final verdict
        if all_pass:
            print(f"\n  🟢 SYSTEM READY (evaluated checks passed)")
        else:
            print(f"\n  🟡 SYSTEM OBSERVATIONAL — some checks failed or skipped")
    
    # System state summary (MODE-AWARE)
    state = crv_system.get_system_state(args.equity)
    
    print(f"\n{'─' * 80}")
    print("SYSTEM STATE SUMMARY")
    print(f"{'─' * 80}")
    
    # Health indicator (MODE-AWARE)
    if fsm.mode == SystemMode.MODE_LIVE_CHECK:
        # In observational mode, show OBSERVATIONAL instead of health
        print(f"  Health: 🟡 OBSERVATIONAL (Live Check Mode)")
        print(f"  Active: NO (observation only)")
    else:
        health_icon = {"healthy": "🟢", "warning": "🟡", "critical": "🔴"}.get(state.system_health, "⚪")
        print(f"  Health: {health_icon} {state.system_health.upper()}")
        print(f"  Active: {'YES' if state.is_active else 'NO'}")
    
    if state.inactivity_reason and fsm.mode != SystemMode.MODE_LIVE_CHECK:
        print(f"  Inactivity: {state.inactivity_reason}")
    
    print(f"  Structural Pairs: {state.n_structural_pairs}")
    print(f"  Regime: {state.current_regime.value}")
    print(f"  Permits CRV: {'YES' if state.regime_permits_trading else 'NO'}")
    print(f"  Positions: {state.n_positions}")
    print(f"  Exposure: {state.risk_state.total_exposure:.1f}%")
    
    # Drawdown (MODE-AWARE)
    if fsm.can_evaluate_drawdown():
        print(f"  Drawdown: {state.risk_state.current_drawdown:.1%}")
    else:
        print(f"  Drawdown: [SKIPPED - FSM forbids evaluation]")
    
    # Kill-Switch (MODE-AWARE)
    if fsm.can_place_orders():
        print(f"  Kill-Switch: {'🔴 ACTIVE' if state.risk_state.is_killed else '🟢 OFF'}")
    else:
        print(f"  Kill-Switch: [N/A - Execution disabled by FSM]")
    
    # Save results
    if args.save:
        results_dir = Path("results/crv")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = results_dir / f"crv_{args.timeframe}_{datetime.now():%Y%m%d_%H%M%S}.json"
        
        save_data = {
            "timestamp": datetime.now().isoformat(),
            "version": "2.1-institutional",
            "timeframe": args.timeframe,
            "fsm": {
                "mode": fsm.mode.value,
                "can_generate_signals": fsm.can_generate_signals(),
                "can_evaluate_drawdown": fsm.can_evaluate_drawdown(),
                "can_place_orders": fsm.can_place_orders(),
            },
            "kill_switch": kill_switch.get_status()['kill_switch'],
            "equity": kill_switch.get_status()['equity'],
            "data": {
                "symbols_loaded": len(aligned.symbols),
                "symbols_rejected": len(aligned.rejected_symbols),
                "common_bars": aligned.n_bars
            },
            "regime": {
                "current": regime_assessment.regime.value,
                "permits_crv": regime_assessment.permits_crv,
                "confidence": regime_assessment.confidence,
                "volatility": regime_assessment.volatility.vol_regime,
                "trend_strength": regime_assessment.trend.trend_strength,
                "sentiment": regime_assessment.sentiment.sentiment.value
            },
            "structural_pairs": [
                {
                    "pair": list(sp.pair),
                    "tier": sp.tier,
                    "relationship": sp.relationship.value if sp.relationship else None,
                    "macro_coherence": float(sp.macro_coherence_score),
                    "correlation_median": float(sp.correlation.median_correlation),
                    "correlation_low_vol": float(sp.correlation.correlation_low_vol),
                    "hedge_ratio": float(sp.operations.current_hedge_ratio),
                    "structural_score": float(sp.structural_score)
                }
                for sp in structural_pairs
            ],
            "signals": [
                {
                    "pair": list(r.pair),
                    "is_tradeable": r.is_tradeable,
                    "is_valid": r.is_valid,
                    "zscore_conditional": float(r.spread_data.zscore_conditional) if r.spread_data and r.spread_data.is_valid else None,
                    "signal_type": r.signal.signal_type.value if r.signal else None,
                    "confidence": float(r.signal.confidence) if r.signal else None
                }
                for r in results
            ],
            "system_state": {
                "health": state.system_health,
                "active": state.is_active,
                "inactivity_reason": state.inactivity_reason,
                "kill_switch": state.risk_state.is_killed,
                "drawdown": state.risk_state.current_drawdown,
                "exposure": state.risk_state.total_exposure
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"\n✓ Results saved to: {filepath}")
    
    # Final summary
    print(f"\n{'=' * 80}")
    print("CRV ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nRemember:")
    print("• FX Relative Value is CONDITIONAL")
    print("• No signal is a VALID signal")
    print("• SAFETY > PROFIT")
    print("• If system trades frequently, something is WRONG")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
