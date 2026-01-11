# FX Conditional Relative Value (CRV) System - v2.0 HARDENED

## INSTITUTIONAL GRADE DOCUMENTATION

---

## 🎯 SYSTEM PHILOSOPHY

```
THIS IS NOT STATISTICAL ARBITRAGE
FX DOES NOT EXHIBIT PERMANENT MEAN REVERSION
THE SYSTEM IS DESIGNED TO NOT TRADE MOST OF THE TIME
INACTIVITY IS CORRECT BEHAVIOR
SAFETY > PROFIT
```

---

## 📊 ARCHITECTURE OVERVIEW

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FX CRV SYSTEM v2.0 HARDENED                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  LAYER 0: DATA INTEGRITY (MANDATORY)                               │
│  ├── NaN validation                                                │
│  ├── Time alignment                                                │
│  ├── Gap detection                                                 │
│  └── Symbol rejection                                              │
│                                                                     │
│  LAYER 1: STRUCTURAL PAIR SELECTION (FX-NATIVE)                    │
│  ├── Macro coherence (50%)                                         │
│  ├── Conditional correlation (30%)                                 │
│  └── Operational viability (20%)                                   │
│                                                                     │
│  LAYER 2: REGIME FILTER (NON-NEGOTIABLE)                           │
│  ├── Volatility analysis                                           │
│  ├── Trend strength (ADX)                                          │
│  ├── Risk sentiment                                                │
│  └── Macro event blocking                                          │
│                                                                     │
│  LAYER 3: CONDITIONAL SPREAD ANALYSIS                              │
│  ├── Regime-specific statistics                                    │
│  ├── Conditional Z-score (NOT universal)                           │
│  └── Never returns NaN                                             │
│                                                                     │
│  LAYER 4: SIGNAL GENERATION                                        │
│  ├── Confluence requirement                                        │
│  ├── Confidence scoring                                            │
│  └── Position sizing                                               │
│                                                                     │
│  LAYER 5: EXECUTION SAFETY                                         │
│  ├── Kill-switch conditions                                        │
│  ├── Currency concentration limits                                 │
│  ├── Drawdown protection                                           │
│  └── Correlation breakdown detection                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🔧 CRITICAL FIXES IN v2.0

### 1. FutureWarning Elimination

**Problem:**
```python
# OLD - Triggers FutureWarning
series.pct_change()  # Uses deprecated fill_method='pad'
```

**Solution:**
```python
# NEW - Clean implementation
series.pct_change(fill_method=None).dropna()

# Or use safe_returns() from data_integrity module
from src.crv.data_integrity import safe_returns
returns = safe_returns(prices)
```

### 2. NaN Prevention

**Problem:**
- Z-scores returning NaN
- NaN propagating through calculations
- Invalid hedge ratios

**Solution:**
```python
# All functions now have explicit validity checks
def safe_zscore(value, mean, std):
    if np.isnan(value) or np.isnan(mean) or np.isnan(std):
        return INVALID_ZSCORE, False  # Never NaN!
    if std < MIN_STD_THRESHOLD:
        return INVALID_ZSCORE, False
    return (value - mean) / std, True
```

### 3. Layer 0 Data Integrity

**New Mandatory Layer:**
```python
validator = FXDataValidator(
    max_nan_percentage=2.0,
    max_gap_hours=48.0,
    min_bars_required=500
)

aligned = validator.align_and_validate(price_data, ohlc_data)
# Returns AlignedDataset with only valid, aligned data
```

### 4. Execution Safety Layer

**New Kill-Switch Conditions:**
```python
class KillSwitchReason(Enum):
    DRAWDOWN_LIMIT = "drawdown_limit"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    REGIME_FLIP = "regime_flip"
    HEDGE_RATIO_FLIP = "hedge_ratio_flip"
    VOLATILITY_EXPLOSION = "volatility_explosion"
    MAX_LOSSES_REACHED = "max_losses_reached"
```

---

## 📋 LIVE TRADING SAFETY CHECKLIST

**ALL must be TRUE before live trading:**

```
[ ] data_integrity_valid        - Data passed Layer 0 validation
[ ] structural_pairs_available  - At least 1 pair passed Layer 1
[ ] regime_permits_trading      - Current regime allows CRV
[ ] kill_switch_off             - No kill-switch condition active
[ ] drawdown_acceptable         - Current drawdown < 5%
[ ] exposure_within_limits      - Total exposure < max limit
[ ] system_health_ok            - Health is "healthy" or "warning"
```

**Run checklist:**
```bash
python scripts/crv_screen.py --timeframe H4 --live-check
```

---

## 🚫 WHAT NOT TO DO

### DO NOT USE:
- Cointegration tests
- Stationarity assumptions (ADF, KPSS)
- Static hedge ratios
- Universal mean reversion
- Forward-filled NaN values

### DO NOT TRADE WHEN:
- Regime blocks CRV
- Kill-switch is active
- Drawdown exceeds threshold
- Correlation breakdown detected
- Macro event within 24h

### DO NOT ASSUME:
- FX pairs are cointegrated
- Spreads always mean-revert
- Divergences are temporary
- More trades = more profit

---

## 📈 EXPECTED BEHAVIOR

### Structural Pairs
- **Normal conditions:** 3-10 valid pairs
- **Stressed conditions:** 0-3 pairs (correct!)
- **Tier distribution:** Mix of A, B, C

### Signals
- **Most of the time:** NO SIGNAL
- **When signal exists:** High confidence required
- **Trading frequency:** RARE (by design)

### System States
- **Healthy:** All layers OK, can trade
- **Warning:** Some concerns, trade cautiously
- **Critical:** Kill-switch active, NO trading

---

## 🔄 REGIME PERMISSIONS

| Regime | Permits CRV | Notes |
|--------|-------------|-------|
| STABLE_LOW_VOL | ✅ YES | Best for CRV |
| STABLE_NORMAL_VOL | ✅ YES | Good for CRV |
| RANGE_BOUND | ✅ YES | Acceptable |
| TRENDING_STRONG | ❌ NO | Divergences are real |
| HIGH_VOLATILITY | ❌ NO | Too much noise |
| RISK_OFF_EXTREME | ❌ NO | Macro-driven |
| RISK_ON_EXTREME | ❌ NO | Macro-driven |
| MACRO_EVENT | ❌ NO | Event risk |
| UNKNOWN | ❌ NO | Play it safe |

---

## 📦 MODULE STRUCTURE

```
src/crv/
├── __init__.py           # All exports
├── data_integrity.py     # Layer 0: Data validation
├── pair_selector.py      # Layer 1: FX-native selection
├── regime_filter.py      # Layer 2: Regime classification
├── conditional_spread.py # Layer 3: Spread analysis
├── execution_safety.py   # Layer 4-5: Signals + Safety
├── signal_engine.py      # Deprecated - use execution_safety
└── crv_system.py         # Main integrated system
```

---

## 🚀 USAGE

### Basic Screening
```bash
python scripts/crv_screen.py --timeframe H4
```

### With Safety Check
```bash
python scripts/crv_screen.py --timeframe H4 --live-check --save
```

### Structural Only
```bash
python scripts/crv_screen.py --timeframe H4 --structural-only
```

### Programmatic Usage
```python
from src.crv import FXConditionalRelativeValueSystem

system = FXConditionalRelativeValueSystem(
    min_macro_score=0.50,
    min_median_correlation=0.20,
    conditional_entry_z=1.5,
    max_positions=3,
)

# Layer 0: Validate data
aligned = system.validate_and_align_data(price_data, ohlc_data)

# Layer 1: Update structural pairs
pairs = system.update_structural_pairs()

# Layer 2: Update regime
regime = system.update_regime(ohlc_data)

# Layers 3-4: Analyze all pairs
results = system.analyze_all_pairs(
    price_data, ohlc_data, regime_history, equity
)

# Layer 5: Safety check
checklist = system.get_live_safety_checklist()
```

---

## ⚠️ CRITICAL REMINDERS

1. **If the system trades frequently, something is WRONG**
2. **No signal is a VALID signal**
3. **FX Relative Value is CONDITIONAL**
4. **Correlation is contextual, not constant**
5. **SAFETY > PROFIT always**

---

## 📊 FLOW SUMMARY

```
Data → [Layer 0: Validate] → Clean Data
                ↓
Clean Data → [Layer 1: Select Pairs] → Structural Pairs
                ↓
Market Data → [Layer 2: Regime Filter] → Regime OK?
                ↓                              ↓
              YES                              NO → STAND DOWN
                ↓
Structural Pairs → [Layer 3: Spread Analysis] → Conditional Z-Score
                ↓
Conditional Z → [Layer 4: Signal Engine] → Signal?
                ↓                              ↓
              YES                              NO → NO TRADE
                ↓
Signal → [Layer 5: Safety Check] → Safe to Execute?
                ↓                              ↓
              YES                              NO → BLOCK
                ↓
         EXECUTE TRADE
```

---

## 📝 VERSION HISTORY

- **v1.0:** Initial StatArb-derived implementation (BROKEN)
- **v1.5:** FX-native Layer 1 redesign
- **v2.0:** HARDENED - Full refactor with safety layers

---

*Last updated: January 2026*
*Author: FX CRV Development Team*
