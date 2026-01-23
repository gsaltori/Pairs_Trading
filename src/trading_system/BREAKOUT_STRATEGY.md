# Breakout Strategy Documentation

## Edge Hypothesis

**Market Behavior:**
Markets alternate between contraction (consolidation) and expansion (trending) phases.
After sufficient contraction, the subsequent expansion move often exceeds the
contraction range by a multiple, creating an asymmetric payoff opportunity.

**Edge Mechanics:**
- Enter on confirmed breakout from compressed range
- Wide stop (opposite side of range) survives noise
- Target 2.5× risk captures expansion moves
- Low win rate offset by high R/R

## Strategy Rules (LOCKED)

### Entry Conditions

| Condition | Parameter | Rationale |
|-----------|-----------|-----------|
| Range lookback | 6 bars | 24 hours on H4 captures session range |
| Compression | Range < 0.8 × ATR(14) | Identifies genuine contraction |
| Breakout | Close > range_high (long) | Confirmed close, not intrabar |
| Breakout | Close < range_low (short) | Confirmed close, not intrabar |

### Trade Management

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Stop Loss | Opposite side + 0.1×ATR | Survives retest of range |
| Take Profit | 2.5 × SL distance | Asymmetric payoff |
| Max hold | 50 bars (optional) | Prevents zombie trades |
| Cooldown | 3 bars | No re-entry on same breakout |

### Mathematical Breakeven

For R/R = 2.5:
```
Breakeven Win Rate = 1 / (1 + R/R) = 1 / 3.5 = 28.6%
```

If actual win rate > 28.6%, strategy is profitable.

## Signal Flow

```
New H4 Bar
    │
    ▼
┌─────────────────────────────────────┐
│  Calculate Range (last 6 bars)      │
│  - range_high = max(highs)          │
│  - range_low = min(lows)            │
│  - range_width = high - low         │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Check Compression                  │
│  range_width < 0.8 × ATR(14)?       │
│                                     │
│  NO → Skip (range not compressed)   │
└─────────────┬───────────────────────┘
              │ YES
              ▼
┌─────────────────────────────────────┐
│  Check Breakout                     │
│  - Close > range_high → LONG        │
│  - Close < range_low → SHORT        │
│                                     │
│  NO breakout → Wait                 │
└─────────────┬───────────────────────┘
              │ BREAKOUT
              ▼
┌─────────────────────────────────────┐
│  GATEKEEPER CHECK                   │
│  - |Z-score| ≤ 3.0?                 │
│  - Correlation trend ≥ -0.05?       │
│  - Volatility ratio ≥ 0.7?          │
│                                     │
│  ANY FAIL → BLOCK                   │
└─────────────┬───────────────────────┘
              │ ALL PASS
              ▼
┌─────────────────────────────────────┐
│  EXECUTE TRADE                      │
│  - Entry: Close price               │
│  - SL: Opposite side + buffer       │
│  - TP: Entry ± 2.5 × SL distance    │
└─────────────────────────────────────┘
```

## Backtest Evaluation Criteria

### Kill Criteria (Any → NOT VIABLE)

| Criterion | Threshold | Meaning |
|-----------|-----------|---------|
| Expectancy | ≤ $0 | Losing money on average |
| R-Expectancy | ≤ 0R | Losing risk units on average |
| Profit Factor | < 1.0 | Gross loss > gross profit |
| Win Rate | < 28.6% | Below mathematical breakeven |

### Warning Criteria

| Criterion | Threshold | Concern |
|-----------|-----------|---------|
| Expectancy | < $0.10 | Marginal profitability |
| R-Expectancy | < 0.15R | Thin edge |
| Profit Factor | < 1.2 | Weak edge |
| Max Drawdown | > 25% | High risk |
| Sample Size | < 20 | Insufficient data |

## Gatekeeper Integration

The structural gatekeeper is applied AFTER signal generation but BEFORE execution:

### Gatekeeper Conditions (LOCKED)

| Condition | Threshold | Block Reason |
|-----------|-----------|--------------|
| |Z-score| | > 3.0 | EXTREME_SPREAD |
| Correlation trend | < -0.05 | DETERIORATING_CORRELATION |
| Volatility ratio | < 0.7 | COMPRESSED_VOLATILITY |

### Why Gatekeeper Matters for Breakout

1. **EXTREME_SPREAD**: Wide spread during breakout = immediate slippage loss
2. **DETERIORATING_CORRELATION**: Pair decoupling = unpredictable behavior
3. **COMPRESSED_VOLATILITY**: Low vol = breakout may fail to extend

## Expected Output Format

```
╔═════════════════════════════════════════════════════════════════════════╗
║                     BREAKOUT STRATEGY EVALUATION                         ║
║               Range Compression → Expansion Breakout                     ║
║                   Target R/R: 2.5 (Asymmetric Payoff)                   ║
║                        Initial Capital: $100.00                          ║
╚═════════════════════════════════════════════════════════════════════════╝

STRATEGY PARAMETERS (LOCKED)
----------------------------------------
  Range Lookback:        6 bars (24h on H4)
  ATR Period:            14
  Compression Threshold: Range < 0.8 × ATR
  Risk/Reward:           2.5
  SL Buffer:             0.1 × ATR
  Cooldown:              3 bars

===========================================================================
BREAKOUT STRATEGY BACKTEST COMPARISON
===========================================================================

Metric                              Baseline       Breakout+Gate
---------------------------------------------------------------------------
Trades                                    XX                  XX
Wins                                      XX                  XX
Losses                                    XX                  XX
Win Rate                               XX.X%               XX.X%
Profit Factor                           X.XX                X.XX

EXPECTANCY ($)                         $X.XX               $X.XX
Expectancy (R)                        X.XXR               X.XXR

Avg Win ($)                            $X.XX               $X.XX
Avg Loss ($)                           $X.XX               $X.XX
Avg Win (R)                           X.XXR               X.XXR
Avg Loss (R)                          X.XXR               X.XXR
Largest Win                            $X.XX               $X.XX
Largest Loss                           $X.XX               $X.XX

Max Drawdown                            X.X%                X.X%
Max DD ($)                             $X.XX               $X.XX
Net PnL                                $X.XX               $X.XX

Avg Bars Held                           X.XX                X.XX
Avg Compression                         X.XX                X.XX

Exits by SL                               XX                  XX
Exits by TP                               XX                  XX
Blocked (Gate)                             0                  XX

===========================================================================
GATEKEEPER COUNTERFACTUAL ANALYSIS
===========================================================================

What would blocked trades have done?

  Blocked by Gatekeeper:    XX
  Resolved (SL/TP hit):     XX
  Would Have Won:           XX
  Would Have Lost:          XX
  Counterfactual Win Rate:  XX.X%
  Counterfactual PnL:       $X.XX

  Gatekeeper: EFFECTIVE/HARMFUL/NEUTRAL (...)

===========================================================================
VIABILITY ASSESSMENT
===========================================================================

  • [KILL/WARNING]: ...

VERDICT: VIABLE / MARGINAL / NOT VIABLE

  [Explanation]

===========================================================================

╔═════════════════════════════════════════════════════════════════════════╗
║                              SUMMARY                                     ║
╚═════════════════════════════════════════════════════════════════════════╝

  Strategy:           Range Breakout (R=2.5)
  Best Config:        [Name]
  Trades:             XX
  Win Rate:           XX.X%
  Profit Factor:      X.XX
  Expectancy ($):     $X.XX
  Expectancy (R):     X.XXR
  Max Drawdown:       X.X%
  Net PnL:            $X.XX

  Mathematical Breakeven WR (R=2.5): 28.6%
  Actual WR:                         XX.X%
  WR vs Breakeven:                   +/-X.X%

  [RECOMMENDATION]
```

## To Run

```bash
cd C:\Users\giova\OneDrive\Escritorio\personal\Pairs_Trading\src
python run_breakout_backtest.py
```

## Decision Matrix

| Verdict | Action |
|---------|--------|
| NOT VIABLE | Abandon edge, research alternatives |
| MARGINAL | Paper trade only, no real capital |
| CAUTIOUS PROCEED | Extended forward test (3+ months) |
| VIABLE | Paper trade → micro capital |

## Risk of Ruin

At $100 with 0.5% risk per trade:
- Risk per trade: $0.50
- With R=2.5 and 30% WR: ~10% probability of 10-trade losing streak
- 10-trade loss: $5.00 = 5% drawdown
- System halt at 8% = $8 loss

The asymmetric payoff protects against ruin while waiting for winners.
