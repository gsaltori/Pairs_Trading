# Time-Series Momentum with Volatility Targeting

A production-grade trading system implementing time-series momentum with volatility targeting.

## Academic Foundation

- **Moskowitz, Ooi, Pedersen (2012)**: "Time Series Momentum"
- **Moreira & Muir (2017)**: Volatility-managed portfolios

## System Logic

### Signal Generation (Per Asset, Independent)
```
IF Close > SMA(252):
    Signal = LONG
ELSE:
    Signal = CASH (no shorts)
```

### Position Sizing (Volatility Targeting)
```
1. Raw Weight = TargetVol (10%) / AssetVol (20-day)
2. If Sum > 1.0: Normalize (no leverage)
3. Cap each asset at 30%
4. Remainder in cash
```

## Universe
- SPY (US Large Cap)
- QQQ (US Tech)
- IWM (US Small Cap)
- EFA (Developed International)
- EEM (Emerging Markets)
- TLT (Long-Term Treasuries)
- GLD (Gold)
- DBC (Commodities)

## Key Features

- **No Leverage**: Sum of weights always <= 1.0
- **Monthly Rebalancing**: Last trading day of month
- **Realistic Costs**: 0.02% commission + 0.01% slippage
- **Mandatory Sanity Checks**: Execution aborts if any check fails

## Sanity Checks

1. Equity > 0 always
2. Sum of weights <= 1.0 (no leverage)
3. Cash >= 0 always
4. Trades only on rebalance dates
5. No lookahead bias
6. Portfolio return = weighted asset returns

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run full backtest with IS/OOS split and Monte Carlo
python run_backtest.py
```

## Output Files

- `results/equity_curve.csv` - Daily equity
- `results/trades.csv` - Trade history
- `results/monthly_returns.csv` - Monthly returns
- `results/REPORT.md` - Full report

## File Structure

```
tsmom_system/
├── config.py           # Locked configuration
├── data_loader.py      # ETF data acquisition
├── signal_engine.py    # Trend signals + vol targeting
├── portfolio_engine.py # Position management + sanity checks
├── metrics.py          # Performance calculations
├── monte_carlo.py      # Block bootstrap simulation
├── run_backtest.py     # Main runner
└── requirements.txt    # Dependencies
```

## Viability Criteria

System is REJECTED if ANY:
- Sharpe < 0.50
- Max DD > 25%
- Trades < 100
- OOS Sharpe <= 0
- MC 95th DD > 35%

---

*Built with engineering discipline and quantitative rigor.*
