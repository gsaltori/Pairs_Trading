# FX Conditional Relative Value (CRV) System - HARDENED v2.0

## Summary of Changes

This document summarizes the complete refactoring and hardening of the FX CRV system.

---

## 🔧 CRITICAL FIXES APPLIED

### 1. NaN Propagation Eliminated

**Problem:** Z-scores were returning `NaN` when:
- Insufficient data in regime windows
- Division by near-zero standard deviation
- Misaligned time indices

**Solution:**
```python
# OLD (Broken)
zscore = (current - mean) / std  # Returns NaN if std ≈ 0

# NEW (Safe)
def safe_zscore(value, mean, std, max_zscore=10.0):
    if np.isnan(value) or np.isnan(mean) or np.isnan(std):
        return INVALID_ZSCORE, False
    if std < MIN_STD_THRESHOLD:
        return INVALID_ZSCORE, False
    zscore = (value - mean) / std
    return np.clip(zscore, -max_zscore, max_zscore), True
```

### 2. FutureWarning Fixed

**Problem:** 
```
FutureWarning: Series.pct_change(fill_method='pad') is deprecated
```

**Solution:**
```python
# OLD (Deprecated)
returns = series.pct_change()

# NEW (Correct)
returns = series.pct_change(fill_method=None).dropna()
```

**Files Updated:**
- `conditional_spread.py`
- `data_integrity.py`
- `crv_screen.py`

### 3. Explicit Validity Flags

**Problem:** Functions returned `None` or incomplete data without indicating invalidity.

**Solution:** All data classes now have explicit `is_valid` flags:

```python
@dataclass
class ConditionalSpreadData:
    # ... fields ...
    is_valid: bool              # EXPLICIT validity
    invalidity_reason: Optional[str]  # WHY it's invalid
```

### 4. Kill-Switch Implementation

**New Feature:** Automatic system shutdown on risk events:

```python
class KillSwitchReason(Enum):
    DRAWDOWN_LIMIT = "drawdown_limit"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    REGIME_FLIP = "regime_flip"
    MAX_LOSSES_REACHED = "max_losses_reached"
```

Triggers:
- Drawdown > 8%
- 3 consecutive losses
- 5 daily losses
- Correlation breakdown detected

---

## 📁 NEW/UPDATED FILES

### Layer 0: Data Integrity (NEW)
**File:** `src/crv/data_integrity.py`

Implements:
- `FXDataValidator` - Validates and aligns data
- `AlignedDataset` - Container for clean, aligned data
- `safe_returns()` - Safe return calculation
- `safe_zscore()` - Safe z-score calculation
- `check_price_sanity()` - Detects bad price data
- `verify_spread_integrity()` - Validates spread construction

### Layer 1: Pair Selection (UPDATED)
**File:** `src/crv/pair_selector.py`

Changes:
- Added `assess_conditional_correlation()` with safe NaN handling
- Added `assess_operational_viability()` with hedge ratio checks
- Tiering system (A/B/C) for pair quality

### Layer 3: Conditional Spread (REWRITTEN)
**File:** `src/crv/conditional_spread.py`

Complete rewrite with:
- `safe_pct_change()` function
- `safe_zscore()` function
- `safe_statistics()` function
- `_invalid_result()` method - returns valid object, not NaN
- All calculations wrapped in try/except

### Layer 5: Execution Safety (NEW)
**File:** `src/crv/execution_safety.py`

New module containing:
- `ExecutionSafetyManager` - Position and exposure management
- `ExecutionConstraints` - Configurable safety limits
- `CRVSignalEngine` - Signal generation with validation
- Kill-switch logic
- Currency concentration controls

### System Integration (UPDATED)
**File:** `src/crv/crv_system.py`

Updates:
- Integrated Layer 0 validation
- Added `check_system_safety()` method
- Added `get_live_safety_checklist()` method
- Health monitoring (healthy/warning/critical)

---

## 🛡️ SAFETY CONSTRAINTS

### Position Limits
```python
max_positions: 3
max_exposure_pct: 10.0
max_position_size_pct: 3.0
```

### Currency Limits
```python
max_currency_exposure_pct: 15.0
max_correlated_positions: 2
```

### Drawdown Limits
```python
warning_drawdown_pct: 3.0
max_drawdown_pct: 5.0
kill_switch_drawdown_pct: 8.0
```

### Time Limits
```python
max_holding_bars: 50
min_bars_between_trades: 10
```

---

## ✅ LIVE TRADING CHECKLIST

The system provides an automated checklist for live trading safety:

```python
checklist = system.get_live_safety_checklist()
# Returns:
{
    "data_integrity_valid": True/False,
    "structural_pairs_available": True/False,
    "regime_permits_trading": True/False,
    "kill_switch_off": True/False,
    "drawdown_acceptable": True/False,
    "exposure_within_limits": True/False,
    "system_health_ok": True/False,
}
```

**ALL must be TRUE for live trading.**

---

## 🚀 USAGE

### Run Screening
```bash
python scripts/crv_screen.py --timeframe H4 --save
```

### Expected Output
```
================================================================================
FX CONDITIONAL RELATIVE VALUE (CRV) SYSTEM
VERSION 2.0 - HARDENED
================================================================================

⚠️  THIS IS NOT STATISTICAL ARBITRAGE
    FX does NOT exhibit permanent mean reversion.
    We trade CONDITIONAL relative value only.
    The system is designed to NOT TRADE most of the time.
    SAFETY > PROFIT

...

LAYER 5: EXECUTION SAFETY CHECK
────────────────────────────────────────────────────────────────────────────────

  🟢 SYSTEM SAFE FOR TRADING

  LIVE TRADING CHECKLIST:
    [✓] Data Integrity Valid
    [✓] Structural Pairs Available
    [✓] Regime Permits Trading
    [✓] Kill Switch Off
    [✓] Drawdown Acceptable
    [✓] Exposure Within Limits
    [✓] System Health Ok

  ✓ ALL CHECKS PASSED - System is live-safe
```

---

## 📊 SYSTEM FLOW SUMMARY

```
┌─────────────────────────────────────────────────────────────────┐
│                    FX CRV SYSTEM v2.0                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ LAYER 0: DATA INTEGRITY (MANDATORY)                       │   │
│  │   • Validate all symbols                                  │   │
│  │   • Remove NaN before calculations                        │   │
│  │   • Align time indices                                    │   │
│  │   • Reject bad data EARLY                                 │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ LAYER 1: STRUCTURAL PAIR SELECTION                        │   │
│  │   • Macro coherence (50%)                                 │   │
│  │   • Conditional correlation (30%)                         │   │
│  │   • Operational viability (20%)                           │   │
│  │   → Output: 3-10 pairs (Tier A/B/C)                       │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ LAYER 2: REGIME FILTER                                    │   │
│  │   • Volatility analysis                                   │   │
│  │   • Trend strength (ADX)                                  │   │
│  │   • Risk sentiment                                        │   │
│  │   • Macro events                                          │   │
│  │   → BLOCKS if unfavorable                                 │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ LAYER 3: CONDITIONAL SPREAD ANALYSIS                      │   │
│  │   • Regime-specific z-score                               │   │
│  │   • Safe calculations (no NaN)                            │   │
│  │   • Explicit validity flags                               │   │
│  │   → Returns valid spread_data OR marked invalid           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ LAYER 4: SIGNAL GENERATION                                │   │
│  │   • Requires ALL conditions met                           │   │
│  │   • Explicit rejection reasons                            │   │
│  │   • Confidence scoring                                    │   │
│  │   → LONG_SPREAD / SHORT_SPREAD / NO_SIGNAL               │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ LAYER 5: EXECUTION SAFETY                                 │   │
│  │   • Position limits                                       │   │
│  │   • Currency concentration                                │   │
│  │   • Drawdown protection                                   │   │
│  │   • Kill-switch                                           │   │
│  │   → SAFE TO EXECUTE / BLOCKED                            │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## ⚠️ PHILOSOPHY REMINDERS

1. **FX does NOT exhibit permanent mean reversion**
2. **Reversion is CONDITIONAL to regime**
3. **Inactivity is CORRECT behavior when edge is absent**
4. **Zero trades > Statistically invalid trades**
5. **SAFETY > PROFIT**

If the system trades frequently, something is wrong.

---

## 🔴 WHAT NOT TO DO

- ❌ Do NOT relax filters to generate more trades
- ❌ Do NOT use `pct_change()` without `fill_method=None`
- ❌ Do NOT ignore NaN in z-score calculations
- ❌ Do NOT assume cointegration exists
- ❌ Do NOT override kill-switch without investigation
- ❌ Do NOT trade in blocked regimes

---

## Version Information

```
Version: 2.0.0-hardened
Date: January 2025
Status: Production-Ready
```
