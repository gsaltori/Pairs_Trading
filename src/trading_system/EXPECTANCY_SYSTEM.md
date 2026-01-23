# Single-Strategy Expectancy System

## Implementation Complete

### Files Added/Modified

**NEW:**
- `market_regime_filter.py` - Pre-signal market regime filter
- `single_strategy_backtest.py` - 3-way comparison backtest
- `run_expectancy_backtest.py` - Main runner script

**MODIFIED:**
- `orchestrator.py` - Integrated MRF before signal generation
- `__init__.py` - Updated exports (removed multi-strategy)

**KEPT (unchanged):**
- `signal_engine.py` - Trend Continuation strategy
- `gatekeeper_engine.py` - Structural filter
- `risk_engine.py` - Drawdown governors
- `execution_engine.py` - MT5 execution

**DEPRECATED (still exist but not exported):**
- `pullback_engine.py`
- `volatility_expansion_engine.py`
- `strategy_router.py`
- `portfolio_*.py`

---

## Market Regime Filter (MRF)

### Conditions (ALL must pass)

| Condition | Threshold | Rationale |
|-----------|-----------|-----------|
| ADX(14) > 22 | Trend strength | Avoid ranging markets |
| ATR(14)/ATR(100) > 1.1 | Volatility expanding | Avoid compression |
| \|EMA200 slope\| > ε | Directional bias | Avoid flat markets |

### Filter Order

```
New Bar
    │
    ▼
┌─────────────────────┐
│   MRF EVALUATE      │  ◄── BEFORE signal generation
│                     │
│   ADX > 22?         │
│   ATR ratio > 1.1?  │
│   EMA slope > ε?    │
└─────────┬───────────┘
          │
          │ ALL PASS
          ▼
┌─────────────────────┐
│  SIGNAL ENGINE      │
│  (Generate signal)  │
└─────────┬───────────┘
          │
          │ Signal exists
          ▼
┌─────────────────────┐
│   GATEKEEPER        │  ◄── AFTER signal generation
│                     │
│   |Z-score| ≤ 3.0?  │
│   Corr trend ≥ -0.05?│
│   Vol ratio ≥ 0.7?  │
└─────────┬───────────┘
          │
          │ ALL PASS
          ▼
┌─────────────────────┐
│   RISK ENGINE       │
│   EXECUTION         │
└─────────────────────┘
```

---

## To Run Backtest

```bash
cd C:\Users\giova\OneDrive\Escritorio\personal\Pairs_Trading\src
python run_expectancy_backtest.py
```

---

## Expected Output Format

```
╔═══════════════════════════════════════════════════════════════════════════════════╗
║                    SINGLE-STRATEGY EXPECTANCY BACKTEST                             ║
║                        Focus: EXPECTANCY > Win Rate                                ║
║                         Initial Capital: $100.00                                   ║
╚═══════════════════════════════════════════════════════════════════════════════════╝

=====================================================================================
LOADING MT5 DATA
=====================================================================================
Loading EURUSD H4 (5000 bars)...
Loading GBPUSD H4 (5000 bars)...
✅ Loaded 4950 aligned bars
   Range: 2022-01-03 00:00:00 to 2025-12-31 00:00:00
   Span: 1093 days (~3.0 years)

Running BACKTEST COMPARISON...
  Running BASELINE (no filters)...
    XX trades, Exp: $X.XX
  Running TREND + GATEKEEPER...
    XX trades, Exp: $X.XX, Blocked: X
  Running TREND + GATEKEEPER + MRF...
    XX trades, Exp: $X.XX
  Running MRF COUNTERFACTUAL...
    MRF blocked: X, Their WR: XX.X%

=====================================================================================
SINGLE-STRATEGY BACKTEST COMPARISON
=====================================================================================

Metric                           Baseline   Trend+Gate  Trend+Gate+MRF
-------------------------------------------------------------------------------------
Trades                                 XX           XX              XX
Wins                                   XX           XX              XX
Losses                                 XX           XX              XX
Win Rate                            XX.X%        XX.X%           XX.X%
Profit Factor                        X.XX         X.XX            X.XX

EXPECTANCY ($)                      $X.XX        $X.XX           $X.XX
Expectancy (R)                      X.XXR        X.XXR           X.XXR

Avg Win ($)                         $X.XX        $X.XX           $X.XX
Avg Loss ($)                        $X.XX        $X.XX           $X.XX
Avg Win (R)                         X.XXR        X.XXR           X.XXR
Avg Loss (R)                        X.XXR        X.XXR           X.XXR

Max Drawdown                         X.X%         X.X%            X.X%
Max DD ($)                          $X.XX        $X.XX           $X.XX
Net PnL                             $X.XX        $X.XX           $X.XX

Blocked (Gate)                          -            X               X
Blocked (MRF)                           -            -               X

=====================================================================================
MRF COUNTERFACTUAL ANALYSIS
=====================================================================================

What would MRF-blocked trades have done if executed?

  Total Blocked by MRF:    X
  Would Have Won:          X
  Would Have Lost:         X
  Counterfactual Win Rate: XX.X%
  Counterfactual PnL:      $X.XX
  Counterfactual Expectancy: $X.XX

  MRF Effectiveness: EFFECTIVE/HARMFUL/NEUTRAL (blocked trades had $X.XX expectancy)

=====================================================================================
VIABILITY ASSESSMENT
=====================================================================================

  • [CRITICAL/WARNING]: ...

VERDICT: VIABLE / MARGINAL / NOT VIABLE

  [Explanation]

=====================================================================================

╔═══════════════════════════════════════════════════════════════════════════════════╗
║                                    SUMMARY                                         ║
╚═══════════════════════════════════════════════════════════════════════════════════╝

  Best Configuration: Trend + Gatekeeper + MRF
  Final Expectancy:   $X.XX per trade
  Final Expectancy:   X.XXR per trade
  Final Profit Factor: X.XX
  Final Max Drawdown:  X.X%
  Final Net PnL:       $X.XX

  [RECOMMENDATION based on verdict]
```

---

## Viability Criteria

| Metric | Critical Fail | Warning | Pass |
|--------|---------------|---------|------|
| Expectancy ($) | ≤ $0 | < $0.10 | ≥ $0.10 |
| Expectancy (R) | ≤ 0R | < 0.1R | ≥ 0.1R |
| Profit Factor | < 1.0 | < 1.2 | ≥ 1.2 |
| Max Drawdown | > 20% | > 10% | ≤ 10% |
| Sample Size | - | < 30 | ≥ 30 |

---

## Decision Tree

```
IF Expectancy ≤ 0:
    VERDICT: NOT VIABLE
    ACTION: DO NOT DEPLOY - Strategy is not profitable

ELIF Expectancy < $0.10 AND Profit Factor < 1.2:
    VERDICT: MARGINAL
    ACTION: Extended paper trading only

ELIF Drawdown > 10%:
    VERDICT: CAUTIOUS PROCEED
    ACTION: Paper trade 3+ months

ELSE:
    VERDICT: VIABLE
    ACTION: Paper trade, then micro capital
```

---

## Live System Runner

After backtest validation:

```bash
# Dry run (no real trades)
python run_trading_system.py

# Live (REAL MONEY)
python run_trading_system.py --live
```

---

## Safety Governors (unchanged)

| Condition | Action |
|-----------|--------|
| DD ≥ 3% | Risk reduced to 0.25% |
| DD ≥ 6% | Max 1 trade |
| DD ≥ 8% | **SYSTEM HALT** |
| DD ≥ 10% | **MANUAL REVIEW** |
| 3 daily losses | Stop for day |
