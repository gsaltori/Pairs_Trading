# Trend Following System - Results

*This file will be auto-generated after running the backtest.*

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run backtest
python run_backtest.py

# Paper trading (after backtest passes)
python run_live.py --paper

# Check status
python run_live.py --status
```

## System Overview

- **Universe**: SPY, QQQ, IWM, GLD
- **Timeframe**: Daily (D1)
- **Style**: Trend Following
- **Risk per Trade**: 0.5%
- **Max Positions**: 3

## Entry Rules

1. Close > EMA(200)
2. Close > 55-day highest high
3. ATR(20) >= median ATR of last 252 days

## Exit Rules

1. Trailing stop = 20-day lowest low
2. No fixed take profit

## Kill Criteria

The system is rejected if ANY of:
- Expectancy < 0.25R
- Profit Factor < 1.5
- Max Drawdown > 20%
- Total Trades < 200
- Monte Carlo 95th DD > 30%

---

*Run `python run_backtest.py` to generate full results.*
