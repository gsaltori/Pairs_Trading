# FX CRV System v2.1 - INSTITUTIONAL HARDENING

## 📐 FSM DESIGN

```
┌─────────────────────────────────────────────────────────────────┐
│                    FINITE STATE MACHINE (FSM)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐         ┌──────────────┐                      │
│  │  MODE_       │◄───────►│  MODE_PAPER  │                      │
│  │  BACKTEST    │         │              │                      │
│  └──────┬───────┘         └──────┬───────┘                      │
│         │                        │                               │
│         │                        │                               │
│         │                        ▼                               │
│         │              ┌──────────────┐                          │
│         └─────────────►│ MODE_LIVE_   │                          │
│                        │ CHECK        │                          │
│                        └──────┬───────┘                          │
│                               │                                  │
│                               │ ONLY valid transition            │
│                               │ to live trading                  │
│                               ▼                                  │
│                        ┌──────────────┐                          │
│                        │ MODE_LIVE_   │                          │
│                        │ TRADING      │                          │
│                        └──────────────┘                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

CAPABILITY MATRIX:
┌───────────────────┬────────────┬────────────┬────────────┐
│       Mode        │  Signals   │  Drawdown  │   Orders   │
├───────────────────┼────────────┼────────────┼────────────┤
│ MODE_BACKTEST     │     ✅      │     ✅      │     ❌      │
│ MODE_PAPER        │     ✅      │     ✅      │     ❌      │
│ MODE_LIVE_CHECK   │     ✅      │     ❌      │     ❌      │
│ MODE_LIVE_TRADING │     ✅      │     ✅      │     ✅      │
└───────────────────┴────────────┴────────────┴────────────┘
```

---

## 🧠 DRAWDOWN & EQUITY LOGIC (WITH GUARDS)

```python
# SAFE EQUITY INITIALIZATION

class EquityState:
    @classmethod
    def create_safe(cls, equity: float) -> 'EquityState':
        """
        SAFE INIT GUARD:
        - If equity is None or <= 0, state is INVALID
        - Drawdown marked as invalid
        - Kill-switch CANNOT trigger on invalid drawdown
        """
        if equity is None or equity <= 0:
            return cls(
                current_equity=0.0,
                equity_peak=0.0,
                equity_initialized=False,  # CRITICAL
                drawdown_valid=False,       # CRITICAL
                drawdown=0.0,
                drawdown_pct=0.0,
            )
        
        return cls(
            current_equity=equity,
            equity_peak=equity,
            equity_initialized=True,
            drawdown_valid=True,
            drawdown=0.0,
            drawdown_pct=0.0,
        )
```

```python
# DRAWDOWN CHECK WITH GUARDS

def _check_drawdown(self) -> Tuple[bool, str]:
    """
    CRITICAL SAFE INIT GUARD:
    - If drawdown_valid is False → CANNOT trigger
    - If equity not initialized → CANNOT trigger
    """
    # Guard 1: Equity must be initialized
    if not self._equity_state.equity_initialized:
        return False, "equity_not_initialized"
    
    # Guard 2: Drawdown must be valid
    if not self._equity_state.drawdown_valid:
        return False, "drawdown_invalid"
    
    # Only now check threshold
    if self._equity_state.drawdown_pct >= self.kill_switch_drawdown:
        return True, f"drawdown_{self._equity_state.drawdown_pct*100:.1f}pct_exceeds_limit"
    
    return False, "drawdown_within_limits"
```

---

## 🚨 KILL-SWITCH WITH EXPLICIT CAUSES

### Primary Reasons (ENUM - DO NOT EXTEND)

```python
class KillSwitchPrimaryReason(Enum):
    NONE = "none"
    DRAWDOWN_LIMIT = "drawdown_limit"
    DATA_INTEGRITY_FAILURE = "data_integrity_failure"
    EXPOSURE_LIMIT = "exposure_limit"
    SYSTEM_HEALTH_FAILURE = "system_health_failure"
    PRE_TRADE_VALIDATION = "pre_trade_validation"
    MANUAL_OVERRIDE = "manual_override"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    REGIME_VIOLATION = "regime_violation"
```

### Audit Format (MANDATORY)

```
Kill-Switch: ACTIVE
Primary Reason: pre_trade_validation
Secondary Reason: equity_not_initialized
Mode: MODE_LIVE_CHECK
Timestamp: 2026-01-09T04:30:00Z
```

### Examples

✅ **VALID:**
```
Kill-Switch: ACTIVE
Primary Reason: drawdown_limit
Secondary Reason: drawdown_8.5pct_exceeds_8.0pct_limit
Mode: MODE_PAPER
Timestamp: 2026-01-09T04:30:00Z
```

❌ **INVALID:**
```
Kill-Switch: ACTIVE
Reason: drawdown
```

---

## ⚙️ MT5 EXECUTION ADAPTER

### Interface

```python
class ExecutionAdapter(ABC):
    @abstractmethod
    def validate(self, orders: List[CRVOrder]) -> Tuple[List[CRVOrder], List[str]]:
        """Validate orders before execution."""
        pass
    
    @abstractmethod
    def build_orders(self, signal: Dict, account_equity: float) -> CRVOrder:
        """Build orders from CRV signal."""
        pass
    
    @abstractmethod
    def simulate_execution(self, orders: List[CRVOrder]) -> List[CRVOrder]:
        """Simulate order execution (paper mode)."""
        pass
    
    @abstractmethod
    def send_orders(self, orders: List[CRVOrder]) -> List[CRVOrder]:
        """
        Send orders to broker.
        
        CRITICAL: MUST RAISE ERROR unless MODE_LIVE_TRADING.
        """
        pass
```

### Mode Guard (HARD RULE)

```python
def send_orders(self, orders: List[CRVOrder]) -> List[CRVOrder]:
    # HARD MODE CHECK
    if self.fsm.mode != SystemMode.MODE_LIVE_TRADING:
        error_msg = (
            f"EXECUTION BLOCKED: Cannot send orders in {self.fsm.mode.value}. "
            f"Order execution only allowed in MODE_LIVE_TRADING."
        )
        raise ModeCapabilityError(error_msg)
    
    # ... actual execution
```

---

## 🧪 EXAMPLE LOGS FOR EACH MODE

### MODE_BACKTEST
```
================================================================================
FX CONDITIONAL RELATIVE VALUE (CRV) SYSTEM - v2.1 INSTITUTIONAL
================================================================================
2026-01-09 04:30:00 - INFO - FSM INITIALIZED: MODE_BACKTEST | Reason: Initial mode set
============================================================
FX CRV SYSTEM - FINITE STATE MACHINE STATUS
============================================================
  Mode: MODE_BACKTEST
  Can Generate Signals: True
  Can Evaluate Drawdown: True
  Can Place Orders: False
  Requires Equity Init: False
============================================================

📊 Mode MODE_BACKTEST does not require equity initialization
```

### MODE_PAPER
```
================================================================================
FX CONDITIONAL RELATIVE VALUE (CRV) SYSTEM - v2.1 INSTITUTIONAL
================================================================================
2026-01-09 04:30:00 - INFO - FSM INITIALIZED: MODE_PAPER | Reason: Initial mode set
2026-01-09 04:30:00 - INFO - Equity initialized: 100000.00

🔧 FSM MODE: MODE_PAPER
   Can Generate Signals: True
   Can Evaluate Drawdown: True
   Can Place Orders: False
```

### MODE_LIVE_CHECK (CORRECTED)
```
================================================================================
FX CONDITIONAL RELATIVE VALUE (CRV) SYSTEM - v2.1 INSTITUTIONAL
================================================================================
2026-01-09 04:30:00 - INFO - FSM INITIALIZED: MODE_LIVE_CHECK | Reason: Initial mode set

🔧 FSM MODE: MODE_LIVE_CHECK
   Can Generate Signals: True
   Can Evaluate Drawdown: False  ← CRITICAL: No drawdown evaluation
   Can Place Orders: False

📊 Mode MODE_LIVE_CHECK does not require equity initialization

────────────────────────────────────────────────────────────────────────────────
LAYER 5: LIVE TRADING SAFETY CHECKLIST (FSM-AWARE)
────────────────────────────────────────────────────────────────────────────────
  [✓] data_integrity_valid: PASS
  [✓] structural_pairs_available: PASS
  [✓] regime_permits_trading: PASS
  [✓] exposure_within_limits: PASS
  [SKIP] drawdown_acceptable: SKIPPED (FSM)
  [SKIP] kill_switch_off: SKIPPED (FSM)
  [✓] system_health_ok: PASS

  FSM Mode: MODE_LIVE_CHECK
  Can Evaluate Drawdown: False
  Can Place Orders: False

  ⚠️ 2 checks SKIPPED by FSM (observational mode)

  🟢 SYSTEM READY (evaluated checks passed)

────────────────────────────────────────────────────────────────────────────────
SYSTEM STATE SUMMARY
────────────────────────────────────────────────────────────────────────────────
  Health: 🟡 OBSERVATIONAL (Live Check Mode)
  Active: NO (observation only)
  Structural Pairs: 8
  Regime: STABLE_NORMAL_VOL
  Permits CRV: YES
  Positions: 0
  Exposure: 0.0%
  Drawdown: [SKIPPED - FSM forbids evaluation]
  Kill-Switch: [N/A - Execution disabled by FSM]
```

### MODE_LIVE_TRADING (Blocked Example)
```
================================================================================
FX CONDITIONAL RELATIVE VALUE (CRV) SYSTEM - v2.1 INSTITUTIONAL
================================================================================
2026-01-09 04:30:00 - ERROR - EXECUTION BLOCKED: Cannot send orders in MODE_PAPER. 
                              Order execution only allowed in MODE_LIVE_TRADING.
```

---

## ❌ THINGS NOT TO DO

### DO NOT:
1. **Relax filters** - Keep all safety layers
2. **Increase trading frequency** - Zero trades is success
3. **Remove safety logic** - Safety > Profit
4. **Trade without regime permission** - Stand down if blocked
5. **Trade during LIVE_CHECK** - No order execution allowed
6. **Trigger kill-switch on invalid drawdown** - Check guards first
7. **Use generic kill-switch reasons** - Always primary + secondary
8. **Send orders in non-live modes** - Hard error if attempted
9. **Extend KillSwitchPrimaryReason enum** - Requires approval
10. **Assume mean reversion** - FX is NOT StatArb

### SYSTEM IS WRONG IF:
- Trades frequently
- Trades without regime permission
- Trades without structural pairs
- Trades during LIVE_CHECK
- Kill-switch triggers at 0% drawdown
- Orders sent in non-live modes

---

## 📦 FILES CREATED/MODIFIED

| File | Status | Description |
|------|--------|-------------|
| `src/crv/state_machine.py` | **NEW** | FSM implementation |
| `src/crv/kill_switch.py` | **NEW** | Institutional kill-switch |
| `src/crv/execution_adapter.py` | **NEW** | MT5 execution adapter |
| `src/crv/__init__.py` | Modified | Added new exports |
| `scripts/crv_screen.py` | Modified | FSM + kill-switch integration |
| `docs/INSTITUTIONAL_HARDENING.md` | **NEW** | This documentation |

---

## 🚀 USAGE

```bash
# Backtest mode (default)
python scripts/crv_screen.py --timeframe H4 --mode backtest

# Paper trading
python scripts/crv_screen.py --timeframe H4 --mode paper --equity 100000

# Live check (pre-live validation)
python scripts/crv_screen.py --timeframe H4 --mode live_check --save

# Live trading (NOT RECOMMENDED without MT5)
python scripts/crv_screen.py --timeframe H4 --mode live --equity 100000
```

---

## ⚠️ FINAL WARNING

```
IF THE SYSTEM:
- Trades frequently           → WRONG
- Trades without regime       → WRONG
- Trades without pairs        → WRONG
- Trades during LIVE_CHECK    → WRONG

INACTIVITY IS A SUCCESS STATE.
```

---

*Generated: January 2026*
*Version: 2.1 INSTITUTIONAL*
