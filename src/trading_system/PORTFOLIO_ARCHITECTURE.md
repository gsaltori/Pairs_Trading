# Micro-Edge Portfolio System Architecture

## Executive Summary

This document describes a portfolio of micro-edge FX strategies that use a session-based directional bias as a **filter** rather than a standalone trading signal.

### The Core Insight

The original Asia → London session strategy showed:
- **53% directional accuracy** (slightly better than random)
- **0.03R expectancy** (too thin for standalone)
- **0.42R average favorable excursion** (price doesn't move enough)

**Key Realization:** This directional accuracy is valuable as a FILTER, not as a trade signal.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                     SESSION DIRECTIONAL BIAS ENGINE                         │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                        ASIA SESSION                                 │   │
│   │                       (00:00-06:00 UTC)                            │   │
│   │                                                                     │   │
│   │   • Calculate range: High, Low, Midpoint                           │   │
│   │   • Determine close position vs midpoint                           │   │
│   │   • Output: BULL / BEAR / NEUTRAL bias                             │   │
│   │   • Output: Confidence score (0.0 - 1.0)                           │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│                    ┌───────────────┴───────────────┐                       │
│                    │                               │                        │
│               LONDON BIAS                     NY BIAS                       │
│              (07:00-12:00)                  (12:00-21:00)                   │
│                    │                               │                        │
│                    │   Can be confirmed/           │   Updated based on     │
│                    │   invalidated by             │   London outcome       │
│                    │   price action               │                        │
│                    └───────────────┬───────────────┘                       │
│                                    │                                        │
└────────────────────────────────────┼────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                       MICRO-EDGE STRATEGY LAYER                             │
│                                                                             │
│   ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐               │
│   │   PULLBACK      │ │   MOMENTUM      │ │   PIVOT         │               │
│   │   SCALPER       │ │   BURST         │ │   BOUNCE        │               │
│   │                 │ │                 │ │                 │               │
│   │ • London only   │ │ • London + NY   │ │ • London + NY   │               │
│   │ • Pullback to   │ │ • Strong candle │ │ • Touch pivot   │               │
│   │   Asia mid      │ │   > 1.2 × ATR   │ │ • Rejection     │               │
│   │ • Target: 0.5R  │ │ • Target: 0.4R  │ │ • Target: 0.6R  │               │
│   │ • Freq: 1-3/day │ │ • Freq: 2-4/day │ │ • Freq: 0-2/day │               │
│   └────────┬────────┘ └────────┬────────┘ └────────┬────────┘               │
│            │                   │                   │                        │
│            └───────────────────┼───────────────────┘                        │
│                                │                                            │
│                    ONLY TRADE IF:                                           │
│                    • Bias is non-NEUTRAL                                    │
│                    • Signal direction matches bias                          │
│                                                                             │
└────────────────────────────────┼────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                          GATEKEEPER FILTER                                  │
│                                                                             │
│   Unchanged from previous implementation:                                   │
│   • |Z-score| ≤ 3.0 (spread not extreme)                                   │
│   • Correlation trend ≥ -0.05 (pair not decoupling)                        │
│   • Volatility ratio ≥ 0.7 (not compressed)                                │
│                                                                             │
└────────────────────────────────┼────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                      PORTFOLIO RISK MANAGER                                 │
│                                                                             │
│   CAPITAL ALLOCATION:                                                       │
│   • Base risk: 0.3% per trade                                              │
│   • Adjusted by bias confidence (0.5x to 1.5x)                             │
│   • Max 2% daily risk budget                                               │
│   • Max 1.5% directional exposure                                          │
│                                                                             │
│   KILL SWITCHES:                                                            │
│   • Daily loss > 1% → Stop for day                                         │
│   • Weekly loss > 3% → Stop for week                                       │
│   • 5 consecutive losses → 4-hour cooldown                                 │
│                                                                             │
│   CORRELATION CONTROL:                                                      │
│   • Max 3 concurrent positions                                             │
│   • Max 2 same-direction positions                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Strategy Details

### 1. London Pullback Scalper

**Concept:** During London session expansion, price often pulls back to key levels before continuing. Trade these pullbacks aligned with the session bias.

| Parameter | Value |
|-----------|-------|
| Session | London only (07:00-11:00 UTC) |
| Bias Required | BULL or BEAR (confidence > 0.5) |
| Entry | Price pulls back to Asia mid zone, shows reversal |
| Stop Loss | Opposite Asia boundary + buffer |
| Target | 0.5R |
| Cooldown | 30 minutes between signals |
| Expected Frequency | 1-3 per session |

**Why Bias Helps:** Without knowing the session bias, pullback trades are 50/50. With bias, we only take pullbacks in the expected direction, improving the odds.

### 2. Momentum Burst Strategy

**Concept:** Catch quick momentum moves that align with the session bias. These are high-probability, low-reward trades.

| Parameter | Value |
|-----------|-------|
| Session | London + NY overlap (07:00-17:00 UTC) |
| Bias Required | BULL or BEAR |
| Entry | Strong candle (body > 1.2 × ATR) in bias direction |
| Stop Loss | 1.5 × ATR from entry |
| Target | 0.4R |
| Cooldown | 20 minutes between signals |
| Expected Frequency | 2-4 per day |

**Why Bias Helps:** Momentum moves can go either direction. The bias helps filter which momentum signals to take.

### 3. Pivot Bounce Strategy

**Concept:** The Asia midpoint acts as a key intraday level. Trade bounces off this level when aligned with bias.

| Parameter | Value |
|-----------|-------|
| Session | London + early NY (07:00-16:00 UTC) |
| Bias Required | BULL or BEAR (confidence > 0.6) |
| Entry | Price touches Asia pivot, shows rejection candle |
| Stop Loss | Through pivot by 0.5 × Asia range |
| Target | 0.6R |
| Cooldown | 60 minutes between signals |
| Expected Frequency | 0-2 per day |

**Why Bias Helps:** Pivot bounces can fail if bias is wrong. The session bias acts as a quality filter.

---

## Risk Management Model

### Position Sizing

```python
base_risk = 0.003  # 0.3% of equity

# Adjust for bias confidence
if bias_confidence > 0.8:
    risk_mult = 1.5
elif bias_confidence > 0.6:
    risk_mult = 1.0
else:
    risk_mult = 0.5

# Adjust for recent losses
if consecutive_losses >= 2:
    risk_mult *= 0.5

# Final risk
trade_risk = base_risk * risk_mult
trade_risk = max(0.001, min(0.005, trade_risk))  # Clamp 0.1% to 0.5%
```

### Daily Risk Budget

| Metric | Limit |
|--------|-------|
| Max daily risk budget | 2% |
| Max daily loss | 1% (hard stop) |
| Max per-trade risk | 0.5% |
| Min per-trade risk | 0.1% |

### Directional Exposure

| Metric | Limit |
|--------|-------|
| Max long exposure | 1.5% |
| Max short exposure | 1.5% |
| Max concurrent positions | 3 |
| Max same-direction | 2 |

### Kill Switches

| Trigger | Action | Duration |
|---------|--------|----------|
| Daily loss > 1% | Stop trading | Until next day |
| Weekly loss > 3% | Stop trading | Until next Monday |
| 5 consecutive losses | Cooldown | 4 hours |

---

## Backtest Evaluation Criteria

### Portfolio-Level

| Metric | Kill Threshold | Warning Threshold | Target |
|--------|----------------|-------------------|--------|
| Expectancy (R) | ≤ 0 | < 0.05R | > 0.1R |
| Profit Factor | < 1.0 | < 1.2 | > 1.5 |
| Max Drawdown | > 15% | > 10% | < 8% |
| Trades/Month | < 5 | < 10 | > 20 |

### Strategy-Level

Each strategy is evaluated independently:

| Metric | Requirement |
|--------|-------------|
| Expectancy (R) | > 0 |
| Profit Factor | ≥ 1.0 |
| Sample Size | ≥ 10 trades |

**Tradeable Definition:** A strategy is "tradeable" if it passes all requirements individually.

---

## Decision Matrix

### Portfolio Verdict

| Condition | Verdict | Action |
|-----------|---------|--------|
| Any KILL criterion fails | NOT VIABLE | Abandon |
| 0 strategies tradeable | NOT VIABLE | Abandon |
| ≥2 WARNING criteria | MARGINAL | Extended paper test |
| 1 WARNING criterion | CAUTIOUS | Careful paper test |
| All clear | VIABLE | Proceed to deployment |

### Component Decisions

After backtest, each component gets explicit status:

```
COMPONENT STATUS
================
Session Bias Engine:    KEEP (provides 53% directional edge)
Gatekeeper:             KEEP (unchanged, validated)
Pullback Scalper:       TRADEABLE / NOT TRADEABLE
Momentum Burst:         TRADEABLE / NOT TRADEABLE
Pivot Bounce:           TRADEABLE / NOT TRADEABLE
Risk Manager:           KEEP (configurable)
```

---

## Deployment Roadmap

### Phase 0: Backtest Validation
- Run `python run_portfolio_v2.py`
- Analyze per-strategy metrics
- Identify tradeable components
- **Gate:** Expectancy > 0, at least 1 tradeable strategy

### Phase 1: Paper Trading (4-6 weeks)
- Run system in dry-run mode
- Log all signals and would-be trades
- Validate:
  - Signal frequency matches backtest
  - Risk management triggers work
  - No software bugs
- **Target:** 50+ paper trades
- **Gate:** Paper results within 20% of backtest

### Phase 2: Micro Capital ($20-50)
- Minimum account size
- Trade only verified strategies
- Focus on:
  - Execution quality
  - Slippage measurement
  - Emotional management
- **Target:** 100+ real trades
- **Gate:** Live expectancy > 0

### Phase 3: Scale to $100
- Increase to target capital
- Full risk management active
- Quarterly performance review
- **Target:** Sustained profitability

### Phase 4: Ongoing Operations
- Monthly performance review
- Quarterly strategy evaluation
- Annual edge validity check
- Continuous improvement

---

## File Structure

```
trading_system/
├── bias_engine.py           # Session Directional Bias Engine
├── micro_strategies.py      # Complementary micro-edge strategies
├── portfolio_risk.py        # Portfolio risk management
├── portfolio_backtest_v2.py # Backtest harness
├── gatekeeper_engine.py     # Structural filter (unchanged)
├── risk_engine.py           # Trade-level risk (unchanged)
└── PORTFOLIO_ARCHITECTURE.md # This document

src/
├── run_portfolio_v2.py      # Backtest runner
└── ...
```

---

## Key Principles

1. **Bias as Filter, Not Signal**
   - The session bias doesn't generate trades
   - It filters trades from other strategies
   - This leverages the 53% accuracy without needing large moves

2. **Small Targets, High Frequency**
   - 0.4R-0.6R targets match actual price excursion
   - More trades = law of large numbers
   - Expectancy compounds over many trades

3. **Strict Risk Control**
   - Multiple kill switches
   - Correlation limits
   - Daily/weekly caps
   - Can't blow up account in one bad day

4. **Component Independence**
   - Each strategy evaluated separately
   - Only deploy what works
   - System degrades gracefully

5. **Honest Evaluation**
   - Explicit kill criteria
   - No optimization allowed
   - If it doesn't work, we abandon it

---

## To Run

```bash
cd C:\Users\giova\OneDrive\Escritorio\personal\Pairs_Trading\src
python run_portfolio_v2.py
```

Expected output includes:
- Portfolio-level metrics
- Per-strategy breakdown
- Viability verdict
- Tradeable strategies list
- Deployment roadmap

---

## If Portfolio Fails

If the portfolio is NOT VIABLE:

1. **Analyze failure modes**
   - Which strategies drag performance?
   - Is the bias actually helping?
   - Are risk limits too tight/loose?

2. **Consider alternatives**
   - Different instruments (GBPUSD more responsive to London)
   - Different sessions (London → NY handoff)
   - Different strategy types entirely

3. **Accept reality**
   - Retail FX may not have accessible edges
   - The pursuit of truth matters more than forced profitability
   - Better to know than to lose money finding out
