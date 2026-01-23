# Exit Optimization Documentation

## Problem Statement

The session strategy (Asia → London Expansion) showed:
- **94% of trades exit via TIME stop** (end of London session)
- **0% TP hits** at the R=2.5 target
- **Expectancy: +0.03R** (barely positive)

This indicates the **target is too aggressive** for the actual price movement during London session expansion.

## Solution: Multiple Exit Models

Instead of changing entry logic, we optimize exits only.

### Exit Models Implemented

#### 1. BASELINE (Current)
```
Single TP at 2.5R
Time stop at London end (11:00 UTC)
```

**Problem:** Price rarely reaches 2.5R during London session. Most trades time out.

#### 2. MULTI_TARGET (Scaled Exit)
```
TP1: 1.0R → Close 50% of position
TP2: 2.0R → Close 30% of position
Runner: Remaining 20% with trailing stop

After TP1:
- Move SL to breakeven + 0.1R buffer
- Trail runner stop at 0.5R behind highest price
```

**Logic:**
- Capture quick moves with TP1 (high probability)
- Let winners run with TP2 and runner
- Protect profits with breakeven stop
- Trail runner for extended moves

#### 3. REDUCED_TP
```
Single TP at 1.5R
Time stop at London end
```

**Logic:** More realistic target based on typical London expansion.

#### 4. AGGRESSIVE
```
Single TP at 1.0R
Time stop at London end
```

**Logic:** High probability, lower reward. Tests if frequent small wins beat rare large wins.

---

## Mathematical Analysis

### Expected Outcomes by Model

For a strategy with ~53% baseline win rate and typical London expansion of 1.0-1.5R:

| Model | Expected TP Hit Rate | Expected Avg R | Trade-off |
|-------|---------------------|----------------|-----------|
| BASELINE (2.5R) | ~0-5% | Low (time stops) | High target, rarely hit |
| MULTI_TARGET | TP1: ~40-50% | Medium | Complexity, but captures movement |
| REDUCED_TP (1.5R) | ~20-30% | Medium | Balanced |
| AGGRESSIVE (1.0R) | ~40-60% | Lower per trade | High hit rate |

### Breakeven Analysis

For each model to be profitable:

| Model | Target R | Required Win Rate |
|-------|----------|-------------------|
| BASELINE | 2.5R | 28.6% |
| REDUCED_TP | 1.5R | 40.0% |
| AGGRESSIVE | 1.0R | 50.0% |
| MULTI_TARGET | Weighted | ~35-40% |

---

## Signal Flow with Exit Management

```
ENTRY SIGNAL (unchanged)
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  POSITION OPENED                                                     │
│                                                                     │
│  Slices created based on ExitConfig:                                │
│  - MULTI_TARGET: 3 slices (50%, 30%, 20%)                          │
│  - Others: 1 slice (100%)                                           │
└─────────────────┬───────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  ON EACH BAR UPDATE                                                  │
│                                                                     │
│  1. Check SL (worst_price)                                          │
│     - If hit → Close ALL remaining slices at SL                     │
│                                                                     │
│  2. Check TP for each open slice                                    │
│     - If target reached → Close that slice at target                │
│     - If first TP → Activate breakeven stop                         │
│                                                                     │
│  3. Update trailing stop for runner (if breakeven active)           │
│                                                                     │
│  4. Check TIME stop (London end)                                    │
│     - If 11:00 UTC → Close ALL remaining slices at close price     │
└─────────────────────────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  POSITION CLOSED                                                     │
│                                                                     │
│  Calculate weighted PnL:                                            │
│  total_pnl_r = Σ (slice.pnl_r × slice.size_pct)                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Breakeven Stop Mechanics

After TP1 is hit in MULTI_TARGET:

```
Before TP1:
  Entry: 1.0850
  SL:    1.0820 (30 pips below)
  TP1:   1.0880 (1.0R = 30 pips above)

After TP1 hit:
  New SL: 1.0853 (entry + 0.1R buffer = 3 pips above entry)
  
This protects:
  - Locks in small profit on remaining 50%
  - Runner and TP2 now risk-free
```

---

## Trailing Stop Mechanics (Runner)

For the 20% runner slice:

```
Trail distance: 0.5R (half the original SL distance)

Example (LONG):
  Entry: 1.0850
  SL distance: 30 pips
  Trail: 15 pips behind highest price

As price moves:
  Highest: 1.0895 → Trail SL: 1.0880
  Highest: 1.0920 → Trail SL: 1.0905
  Price pulls back to 1.0905 → Runner closed
  
Runner PnL: (1.0905 - 1.0850) / 0.0030 = 1.83R
```

---

## Kill Criteria

The exit optimization is valid ONLY if:

| Criterion | Threshold | Action if Failed |
|-----------|-----------|------------------|
| Expectancy | > 0 | REJECT model |
| Drawdown increase | ≤ 20% vs baseline | REJECT model |
| Trade frequency | Unchanged | N/A (exits don't affect entry) |

If ALL models fail kill criteria, the strategy itself is invalid.

---

## Expected Output

```
╔═══════════════════════════════════════════════════════════════════════════════════════════╗
║                              EXIT MODEL COMPARISON BACKTEST                                ║
╚═══════════════════════════════════════════════════════════════════════════════════════════╝

EXIT MODEL COMPARISON
Same entry signals, different exit logic
===============================================================================================

Metric                          BASELINE      MULTI_TARGET       REDUCED_TP       AGGRESSIVE
-----------------------------------------------------------------------------------------------
Trades                                XX               XX               XX               XX
Win Rate                           XX.X%            XX.X%            XX.X%            XX.X%

EXPECTANCY (R)                    X.XXR            X.XXR            X.XXR            X.XXR
Profit Factor                      X.XX             X.XX             X.XX             X.XX

Exits by SL                           X                X                X                X
Exits by TIME                        XX               XX               XX               XX
% TP Hits                          X.X%            XX.X%            XX.X%            XX.X%

Max Drawdown                       X.X%             X.X%             X.X%             X.X%

===============================================================================================
RECOMMENDATION
===============================================================================================

  Verdict: ADOPT / KEEP_BASELINE / REJECT_ALL
  Best Exit Model: [MODEL_NAME]
  Explanation: [Why this model is best]
```

---

## To Run

```bash
cd C:\Users\giova\OneDrive\Escritorio\personal\Pairs_Trading\src
python run_exit_comparison.py
```

---

## Decision Matrix

| Verdict | Meaning | Action |
|---------|---------|--------|
| **ADOPT** | New model significantly better | Implement in production |
| **MARGINAL** | Minor improvement | Consider for extended testing |
| **KEEP_BASELINE** | No improvement | Keep current exits |
| **REJECT_ALL** | All fail kill criteria | Strategy is invalid |

---

## Implementation Notes

### For Production Use

If MULTI_TARGET is adopted:

1. **ExitManager Integration**
   - Replace simple SL/TP logic with ExitManager
   - Track multiple position slices
   - Handle partial closes correctly

2. **MT5 Execution**
   - Partial close using `mt5.Close()` with reduced volume
   - Update SL for remaining position after partial close
   - Track each slice separately

3. **Logging**
   - Log each slice close separately
   - Track which TP level was hit
   - Record breakeven activation and trailing updates

4. **Risk Accounting**
   - Position size calculated on TOTAL risk
   - Each slice inherits proportional size
   - Risk-free after breakeven (no risk attribution to remaining slices)
