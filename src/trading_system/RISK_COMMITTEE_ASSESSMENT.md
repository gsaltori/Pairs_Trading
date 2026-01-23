# RISK COMMITTEE ASSESSMENT
## Micro-Edge FX Portfolio - Final Evaluation

**Date:** January 2026  
**Capital at Risk:** $100  
**Assessment Type:** Go/No-Go Decision

---

## SECTION 1: OBSERVED RESULTS

### Portfolio Performance Summary

| Metric | Observed | Acceptable | Status |
|--------|----------|------------|--------|
| Max Drawdown | 22.3% | ≤10% | ❌ FAIL |
| Daily Halts | Excessive | Rare | ❌ FAIL |
| Edge Density | Low | Moderate | ❌ FAIL |

### Strategy-Level Results

| Strategy | Expectancy (R) | Status |
|----------|----------------|--------|
| PULLBACK_SCALPER | ≤ 0 | ❌ ABANDON |
| MOMENTUM_BURST | ≤ 0 | ❌ ABANDON |
| PIVOT_BOUNCE | +0.063R | ⚠️ MARGINAL |

**Only PIVOT_BOUNCE shows any positive expectancy.**

---

## SECTION 2: MATHEMATICAL REALITY CHECK

### Edge Magnitude Analysis

```
PIVOT_BOUNCE Expectancy: 0.063R

For $100 account with 0.5% risk per trade ($0.50):
  Expected profit per trade = 0.063 × $0.50 = $0.0315

For $100 account with 0.3% risk per trade ($0.30):
  Expected profit per trade = 0.063 × $0.30 = $0.019

For $100 account with 0.1% risk per trade ($0.10):
  Expected profit per trade = 0.063 × $0.10 = $0.0063
```

**Translation:** At conservative risk levels, expected profit is ~2 cents per trade.

### Transaction Cost Impact

```
Typical EURUSD spread: 0.8-1.2 pips
Round-trip cost on 0.01 lot: ~$0.10-0.15

If expected profit per trade = $0.02
And transaction cost = $0.10
Net expectancy = -$0.08 per trade

THE EDGE IS CONSUMED BY COSTS.
```

### Statistical Confidence

```
For 0.063R expectancy to be statistically significant at 95% confidence:

Assuming standard deviation of ~1R per trade (typical for FX):
  Standard Error = σ / √n
  
For 95% confidence, need: Expectancy > 1.96 × SE

If n = 20 trades:  SE = 1/√20 = 0.22   → Need exp > 0.44R
If n = 50 trades:  SE = 1/√50 = 0.14   → Need exp > 0.28R
If n = 100 trades: SE = 1/√100 = 0.10  → Need exp > 0.20R
If n = 400 trades: SE = 1/√400 = 0.05  → Need exp > 0.10R

0.063R expectancy requires ~1000+ trades for statistical significance.
At 0-2 trades/day, this requires 500-1000+ trading days (2-4 years).
```

**The observed expectancy is likely noise, not signal.**

### Drawdown Mathematics

```
Observed max DD: 22.3%

For $100 account:
  Max DD in dollars = $22.30
  
If using Kelly-optimal sizing on 0.063R edge:
  Kelly fraction = edge / odds ≈ 0.063 / 1 = 6.3%
  Half-Kelly (conservative) = 3.15%
  
But 22.3% DD on 3.15% risk means ~7 consecutive losses.
Probability of 7 losses in a row (assuming 40% win rate): 
  0.6^7 = 2.8%

This is within normal variance - the DD is expected, not anomalous.
```

---

## SECTION 3: SURVIVAL SYSTEM DESIGN (IF CONTINUING)

### Minimal PIVOT_BOUNCE System

**IF** the decision is to continue research, here is the minimal survivable configuration:

```
SYSTEM: Pivot Bounce - Survival Mode
=====================================

COMPONENTS:
- Session Bias Engine (filter only)
- Gatekeeper (unchanged)
- PIVOT_BOUNCE strategy (single strategy)

RISK PARAMETERS (Ultra-Conservative):
- Risk per trade: 0.1% ($0.10 on $100)
- Max daily risk: 0.3% ($0.30)
- Max weekly risk: 0.6% ($0.60)
- Max monthly drawdown: 2% ($2.00)

TRADE LIMITS:
- Max 1 trade per session
- Max 1 trade per day
- Max 5 trades per week

KILL SWITCHES:
- 2% monthly DD → HALT for month
- 5% account DD → HALT for 30 days
- 3% total DD → REDUCE to paper trading

EXPECTED PERFORMANCE:
- Trades per month: 10-15
- Expected monthly profit: $0.10-0.20 (before costs)
- Expected monthly profit after costs: NEGATIVE
```

### Why This System Still Fails

| Factor | Impact |
|--------|--------|
| Transaction costs | Consume 100%+ of edge |
| Low frequency | Insufficient for statistical learning |
| Thin edge | No margin for error |
| Variance | Will experience multi-month losing streaks |

---

## SECTION 4: KILL CRITERIA (If Research Continues)

### Immediate Abandonment Triggers

| Trigger | Threshold | Action |
|---------|-----------|--------|
| Account drawdown | > 5% ($5) | STOP permanently |
| Monthly drawdown | > 2% ($2) | HALT 30 days |
| Consecutive losses | > 5 | HALT 2 weeks |
| Win rate | < 30% over 20 trades | ABANDON |
| Net P&L after 50 trades | < $0 | ABANDON |

### Research Validation Requirements

To justify ANY real capital:

| Requirement | Threshold |
|-------------|-----------|
| Paper trade sample | ≥ 100 trades |
| Paper expectancy | > 0.10R (after simulated costs) |
| Paper max DD | < 10% |
| Paper win rate | > 35% |
| Paper profit factor | > 1.2 |

**Current system does not meet these requirements.**

---

## SECTION 5: SCALING CRITERIA (Theoretical)

### What Would Need to Be True for Scaling

To justify increasing from $100 to $500:

| Metric | Required | Currently |
|--------|----------|-----------|
| Live expectancy | > 0.15R | 0.063R (backtest) |
| Live trades | > 100 | 0 |
| Live DD | < 8% | 22.3% (backtest) |
| Live profit | > $5 | Unknown |
| Months profitable | 3 consecutive | 0 |

**None of these criteria are remotely close to being met.**

---

## SECTION 6: HONEST ASSESSMENT

### What the Data Shows

1. **The session bias provides ~53% directional accuracy**
   - This is real but too weak to profit from directly
   - As a filter, it adds marginal value

2. **PIVOT_BOUNCE shows 0.063R expectancy**
   - Statistically indistinguishable from zero
   - Likely noise, not edge
   - Transaction costs make it negative

3. **22.3% drawdown is unacceptable**
   - On $100, this is $22.30 at risk
   - For a ~$0.02/trade expected profit
   - Risk/reward is catastrophically poor

4. **The edge density is too low**
   - 0-2 trades per day (theoretical)
   - In practice, likely 2-5 per week with filters
   - Insufficient for meaningful compounding

### What Would Need to Change

For this approach to become viable:

| Change Needed | Current | Required |
|---------------|---------|----------|
| Expectancy | 0.063R | > 0.20R |
| Trade frequency | 2-5/week | > 20/week |
| Transaction costs | ~$0.10/trade | < $0.02/trade |
| Capital base | $100 | > $10,000 |

**None of these changes are achievable without fundamental strategy redesign.**

---

## SECTION 7: FINAL DECISION

### Risk Committee Verdict

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║                              DECISION: ABANDON                                ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

### Justification

| Factor | Assessment |
|--------|------------|
| Edge magnitude | Too small (0.063R) |
| Statistical significance | Insufficient (likely noise) |
| Transaction cost impact | Edge-destroying |
| Drawdown risk | Unacceptable (22.3%) |
| Risk/reward ratio | Catastrophically poor |
| Practical viability | None with $100 capital |

### Mathematical Summary

```
Expected value of continuing:

  Best case (edge is real):
    - 100 trades × $0.02/trade = $2.00 profit
    - Time required: 6-12 months
    - Risk exposure: $22.30 DD possible
    
  Expected case (edge is noise):
    - 100 trades × -$0.05/trade = -$5.00 loss
    - Time wasted: 6-12 months
    
  Worst case (bad variance):
    - Hit 5% DD limit = -$5.00 loss
    - Abandon anyway
    
Expected value = (0.1 × $2) + (0.6 × -$5) + (0.3 × -$5) = -$4.30

NEGATIVE EXPECTED VALUE. DO NOT CONTINUE.
```

---

## SECTION 8: RECOMMENDATIONS

### What to Do Instead

1. **Accept the Learning**
   - The systematic approach was correct
   - The edge validation framework worked
   - The result (no viable edge) is valuable information

2. **Consider Alternative Paths**

   | Path | Viability |
   |------|-----------|
   | Increase capital to $10K+ | Would reduce cost impact |
   | Different asset class | Crypto/stocks may have different dynamics |
   | Longer timeframes | Daily/weekly reduce transaction cost impact |
   | Different edge type | Mean reversion, carry, etc. |
   | Accept no retail edge | Index investing is mathematically superior |

3. **Preserve Capital**
   - $100 in index fund: ~7% annual expected return = $7/year
   - $100 in this system: -$4.30 expected value
   - Opportunity cost is real

### If You Absolutely Must Continue

```
PAPER TRADING ONLY

Requirements before ANY real money:
1. 100+ paper trades
2. Paper expectancy > 0.10R after costs
3. Paper max DD < 10%
4. 3+ months of data

Timeline: 4-6 months minimum
Real capital: $0 until validation complete
```

---

## SECTION 9: CLOSING STATEMENT

### The Uncomfortable Truth

> Retail FX trading on micro accounts does not have a positive expected value for the vast majority of participants. The combination of:
> - Thin edges (if any exist)
> - Transaction costs
> - Psychological challenges
> - Statistical noise
> 
> Makes profitability nearly impossible.
>
> The fact that this systematic approach found NO viable edge after extensive research is not a failure - it is the correct answer. Most edges in liquid markets have been arbitraged away.
>
> **The honest conclusion is that this capital and time would be better allocated elsewhere.**

### Final Recommendation

| Item | Recommendation |
|------|----------------|
| PIVOT_BOUNCE | ABANDON |
| Session Bias research | COMPLETE (useful learning) |
| Portfolio approach | ABANDON |
| $100 FX trading | NOT RECOMMENDED |
| Future research | Only with paper trading |

---

## APPENDIX: Decision Tree for Future Reference

```
START: New trading edge hypothesis
│
├─ Backtest expectancy > 0?
│  ├─ NO → ABANDON immediately
│  └─ YES → Continue
│
├─ Expectancy > 0.15R?
│  ├─ NO → Edge too thin, ABANDON
│  └─ YES → Continue
│
├─ Expectancy > transaction costs?
│  ├─ NO → Not viable, ABANDON
│  └─ YES → Continue
│
├─ Max DD < 10%?
│  ├─ NO → Risk too high, ABANDON or redesign
│  └─ YES → Continue
│
├─ Trade frequency > 20/month?
│  ├─ NO → Too slow to validate, consider ABANDON
│  └─ YES → Continue
│
├─ Statistical significance achievable in < 6 months?
│  ├─ NO → Too slow, ABANDON
│  └─ YES → Proceed to paper trading
│
├─ Paper trading confirms backtest?
│  ├─ NO → ABANDON
│  └─ YES → Proceed to micro capital
│
└─ Micro capital confirms paper?
   ├─ NO → ABANDON
   └─ YES → Scale cautiously
```

**Current system fails at step 2 (Expectancy > 0.15R).**

---

**Document Status:** FINAL  
**Decision:** ABANDON  
**Confidence:** HIGH  
**Basis:** Mathematical analysis, not opinion
