# Cross-Sectional Momentum System

A production-grade cross-sectional momentum trading system for ETFs.

## Academic Foundation

This system implements the well-documented momentum factor:
- Jegadeesh & Titman (1993): "Returns to Buying Winners and Selling Losers"
- Asness, Moskowitz & Pedersen (2013): "Value and Momentum Everywhere"

Cross-sectional momentum has shown persistent out-of-sample performance across decades and multiple asset classes.

## System Design

### Universe (10 Diversified ETFs)
- **US Equities**: SPY, QQQ, IWM
- **International**: EFA, EEM
- **Fixed Income**: TLT, IEF
- **Alternatives**: GLD, DBC, VNQ

### Strategy Logic
1. Calculate 12-month momentum for all assets
2. Rank assets by momentum (highest to lowest)
3. Apply trend filter: Price > EMA(200)
4. Select top 3 assets that pass filter
5. Equal weight (~33% each)
6. Rebalance monthly (last trading day)

### Risk Controls
- No leverage
- No stop losses (trend filter handles this)
- Monthly rebalancing only
- Cash allocation when fewer than 3 qualify

## Usage

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Backtest
```bash
python backtest_runner.py
```

This will:
1. Download 20+ years of ETF data
2. Run in-sample and out-of-sample backtests
3. Execute Monte Carlo simulation (1000 runs)
4. Check viability criteria
5. Generate comprehensive report

### Paper Trading
```bash
# Check current status
python live_runner.py --status

# Generate today's signal
python live_runner.py --signal

# Run rebalance check (auto-executes on month-end)
python live_runner.py
```

## Viability Criteria

System is REJECTED if ANY of:
- Expectancy < 0.25R
- Sharpe Ratio < 0.70
- Max Drawdown > 30%
- Total Trades < 300
- Monte Carlo 95th DD > 30%

## File Structure

```
momentum_system/
├── config.py           # Locked configuration
├── data_loader.py      # ETF data acquisition
├── momentum_engine.py  # Ranking and selection
├── portfolio_engine.py # Position management
├── execution_engine.py # Order execution
├── reporting.py        # Metrics and reports
├── backtest_runner.py  # Historical simulation
├── live_runner.py      # Paper/live trading
└── requirements.txt    # Dependencies
```

## Output Files

After backtest:
- `results/REPORT.md` - Full backtest report
- `results/equity_curve.csv` - Daily equity values
- `results/trades.csv` - Complete trade history
- `results/monthly_returns.csv` - Monthly return series

## Why This Strategy?

1. **Academic Evidence**: Momentum is one of the most robust factors in finance with 100+ years of data

2. **Simple Logic**: No complex rules, indicators, or optimization

3. **Diversified Universe**: Spans equities, bonds, commodities, real estate

4. **Trend Filter**: EMA(200) avoids buying into downtrends

5. **Monthly Rebalance**: Low turnover, low costs, practical for retail

6. **Scalable**: Works from $10K to $10M+ without market impact

## Expectations

Based on historical data (2005-present), expect:
- CAGR: 8-12%
- Sharpe: 0.6-1.0
- Max DD: 15-25%
- Monthly turnover: ~30-50%

**Note**: Past performance does not guarantee future results.

---

*Built with engineering discipline and quantitative rigor.*
