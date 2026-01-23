# Session-Based FX Strategy Documentation

## Edge Hypothesis

### Liquidity-Driven Directional Bias

Unlike technical pattern-based strategies, this edge exploits **institutional order flow mechanics**:

```
┌─────────────────────────────────────────────────────────────────────┐
│  GLOBAL FX MARKET VOLUME DISTRIBUTION                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Sydney/Tokyo (Asia):  ~15% of daily volume                        │
│  ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
│                                                                     │
│  London:               ~35% of daily volume                        │
│  ██████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
│                                                                     │
│  New York:             ~25% of daily volume                        │
│  ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
│                                                                     │
│  London/NY overlap:    ~20% of daily volume                        │
│  ██████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### The Mechanism

1. **Asian Session (00:00-06:00 UTC)**
   - Low liquidity environment (~15% volume)
   - Major banks in Tokyo/Sydney set initial range
   - Price "settles" at a level reflecting overnight order flow
   - **The close position relative to the range midpoint reveals pending institutional bias**

2. **London Session (07:00-11:00 UTC)**
   - High liquidity environment (~35% volume)
   - European institutional order flow enters
   - **If Asia closed bullish** (above mid): Buy-side flow tends to dominate
   - **If Asia closed bearish** (below mid): Sell-side flow tends to dominate
   - Range expansion occurs in the biased direction

### Why This Works (Theoretically)

- Asian session is a "discovery phase" with limited participation
- Where price settles indicates the balance of overnight orders
- London brings fresh capital that confirms or denies the overnight bias
- Confirmation (expansion in bias direction) happens more often than reversal

---

## Strategy Rules (LOCKED)

### Session Definitions (UTC)

| Session | Time (UTC) | Purpose |
|---------|------------|---------|
| Asia | 00:00 - 06:00 | Range establishment |
| London | 07:00 - 11:00 | Expansion phase |

### Asia Range Analysis

```python
# Calculate from Asia session bars
asia_high = max(highs during 00:00-06:00 UTC)
asia_low = min(lows during 00:00-06:00 UTC)
asia_mid = (asia_high + asia_low) / 2
asia_close = close of last Asia bar (05:30 or 06:00)

# Determine bias
if asia_close > asia_mid + 0.2 * range:
    bias = BULLISH
elif asia_close < asia_mid - 0.2 * range:
    bias = BEARISH
else:
    bias = NEUTRAL (no trade)
```

### Entry Logic

| Condition | Action |
|-----------|--------|
| Bias = BULLISH | Wait for break above Asia high during London |
| Bias = BEARISH | Wait for break below Asia low during London |
| Bias = NEUTRAL | No trade (bias too weak) |

**Entry Requirements:**
- Close outside Asia range (not just wick)
- Must be during London session (07:00-11:00 UTC)
- Max 1 trade per day
- No counter-trend trades (bias must match direction)

### Trade Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Stop Loss | Opposite side of Asia range + 3 pips | Full range protection |
| Take Profit | 2.5 × SL distance | Asymmetric payoff |
| Time Stop | End of London (11:00 UTC) | Don't hold through NY overlap |
| Max Hold | ~4 hours | London session only |

### Filters

| Filter | Threshold | Rationale |
|--------|-----------|-----------|
| Min Asia Range | 15 pips | Avoid noise-dominated days |
| Max Asia Range | 80 pips | Avoid excessive risk/event days |
| Bias Threshold | 20% of range from mid | Require meaningful bias |

---

## Signal Flow

```
00:00 UTC - Asia Session Opens
    │
    ▼ [Accumulate bars until 06:00]
    
06:00 UTC - Asia Session Closes
    │
    ▼
┌─────────────────────────────────────────┐
│  CALCULATE ASIA RANGE                   │
│  - High, Low, Midpoint                  │
│  - Range size (pips)                    │
│  - Close position vs midpoint           │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  VALIDATE RANGE                         │
│  - 15 ≤ range ≤ 80 pips?               │
│                                         │
│  NO → Skip day (no trade possible)      │
└─────────────────┬───────────────────────┘
                  │ YES
                  ▼
┌─────────────────────────────────────────┐
│  DETERMINE BIAS                         │
│  - Close > mid+20% → BULLISH           │
│  - Close < mid-20% → BEARISH           │
│  - Otherwise → NEUTRAL                  │
│                                         │
│  NEUTRAL → Skip day                     │
└─────────────────┬───────────────────────┘
                  │ BULLISH or BEARISH
                  ▼
07:00 UTC - London Session Opens
                  │
                  ▼ [Monitor each bar]
                  
┌─────────────────────────────────────────┐
│  CHECK FOR BREAKOUT                     │
│                                         │
│  BULLISH bias:                          │
│    Close > Asia high? → LONG            │
│                                         │
│  BEARISH bias:                          │
│    Close < Asia low? → SHORT            │
│                                         │
│  No breakout → Wait                     │
└─────────────────┬───────────────────────┘
                  │ BREAKOUT
                  ▼
┌─────────────────────────────────────────┐
│  GATEKEEPER CHECK (unchanged)           │
│  - |Z-score| ≤ 3.0                      │
│  - Correlation trend ≥ -0.05            │
│  - Volatility ratio ≥ 0.7               │
│                                         │
│  ANY FAIL → BLOCK                       │
└─────────────────┬───────────────────────┘
                  │ ALL PASS
                  ▼
┌─────────────────────────────────────────┐
│  EXECUTE TRADE                          │
│  - Entry: Close price                   │
│  - SL: Opposite side + 3 pips          │
│  - TP: Entry ± 2.5 × SL distance       │
└─────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  TRADE MANAGEMENT                       │
│  - Monitor for SL/TP hit               │
│  - Time stop at 11:00 UTC              │
└─────────────────────────────────────────┘
```

---

## Kill Criteria

| Criterion | Threshold | Action |
|-----------|-----------|--------|
| Expectancy ($) | ≤ $0 | **ABANDON** |
| Expectancy (R) | ≤ 0R | **ABANDON** |
| Profit Factor | < 1.0 | **ABANDON** |
| Win Rate | < 28.6% (breakeven for R=2.5) | **ABANDON** |
| Trade Frequency | < 2/month | **ABANDON** |

If ANY kill criterion is met, the strategy is NOT VIABLE.

---

## Expected Output Format

```
╔═════════════════════════════════════════════════════════════════════════╗
║                 SESSION-BASED FX STRATEGY EVALUATION                     ║
║                  Asia Range → London Session Expansion                   ║
╚═════════════════════════════════════════════════════════════════════════╝

===========================================================================
SESSION STRATEGY BACKTEST COMPARISON
===========================================================================

Metric                              Baseline       Session+Gate
---------------------------------------------------------------------------
Sessions Analyzed                         XXX                XXX
Valid Setups                              XX                 XX
Trades Executed                           XX                 XX
Trades/Month                             X.X                X.X

Win Rate                               XX.X%              XX.X%
Profit Factor                           X.XX               X.XX

EXPECTANCY ($)                         $X.XX              $X.XX
Expectancy (R)                        X.XXR              X.XXR

Max Drawdown                            X.X%               X.X%
Net PnL                                $X.XX              $X.XX

Exits by SL                               XX                 XX
Exits by TP                               XX                 XX
Exits by TIME                             XX                 XX

===========================================================================
VIABILITY ASSESSMENT
===========================================================================

VERDICT: VIABLE / NOT VIABLE

[Clear recommendation]
```

---

## To Run

```bash
cd C:\Users\giova\OneDrive\Escritorio\personal\Pairs_Trading\src
python run_session_backtest.py
```

---

## Why This Differs From Previous Edges

| Previous Edge | Basis | Why Failed |
|---------------|-------|------------|
| Trend Continuation | EMA crossover | Technical signal, no structural advantage |
| Range Breakout | ATR compression | Pattern recognition, noise-dominated |

| This Edge | Basis | Potential Advantage |
|-----------|-------|---------------------|
| Session | Institutional liquidity mechanics | Structural market microstructure |

The key difference: **This edge is based on HOW markets work (liquidity dynamics), not WHAT patterns form (technical analysis).**

---

## Risk of Failure

Even if the liquidity hypothesis is correct, this strategy can still fail due to:

1. **Insufficient edge magnitude** - The bias may exist but be too weak to profit after costs
2. **Low frequency** - May not generate enough trades to be practical
3. **Regime dependence** - May work in trending markets but not ranging
4. **Gatekeeper interference** - May block the few profitable setups

The backtest will reveal whether the edge is **viable** or should be **abandoned**.

---

## If Strategy Fails

If this session-based approach is invalidated, options include:

1. **Different session pairs** (e.g., London → NY handoff)
2. **Different instruments** (e.g., GBPUSD more London-centric)
3. **Different liquidity edges** (e.g., news-driven flow)
4. **Accept that retail FX may have no accessible edge**

The goal is to find truth, not to force profitability.
