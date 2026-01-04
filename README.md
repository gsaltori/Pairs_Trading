# Pairs Trading System

Professional statistical arbitrage system for Forex markets using **IC Markets Global** via **MetaTrader 5**.

## Overview

This system implements institutional-grade pairs trading with:
- Statistical pair selection (correlation + cointegration)
- Dynamic hedge ratio calculation (OLS regression)
- Z-score based entry/exit signals
- Professional risk management
- Walk-forward optimization
- Real-time execution via MT5

## Architecture

```
Pairs_Trading/
├── config/
│   ├── settings.py          # Configuration management
│   └── broker_config.py     # MT5 connection settings
├── src/
│   ├── data/
│   │   ├── broker_client.py # MT5 API client
│   │   └── data_manager.py  # Data caching & preprocessing
│   ├── analysis/
│   │   ├── correlation.py   # Correlation analysis
│   │   ├── cointegration.py # ADF & Johansen tests
│   │   └── spread_builder.py# Spread construction
│   ├── strategy/
│   │   ├── signals.py       # Signal generation
│   │   └── pairs_strategy.py# Strategy orchestration
│   ├── risk/
│   │   └── risk_manager.py  # Position sizing & limits
│   ├── backtest/
│   │   └── backtest_engine.py# Backtesting with costs
│   ├── optimization/
│   │   └── optimizer.py     # Walk-forward optimization
│   └── execution/
│       └── executor.py      # Live order execution
├── scripts/                  # Example scripts
├── tests/                    # Unit tests
└── main.py                   # CLI entry point
```

## Requirements

- Python 3.10+
- MetaTrader 5 terminal (IC Markets Global)
- Windows OS (MT5 Python API requirement)

## Installation

1. **Install MetaTrader 5** from IC Markets Global

2. **Clone and setup**:
```bash
git clone <repository>
cd Pairs_Trading
pip install -r requirements.txt
```

3. **Configure credentials**:
```bash
cp .env.example .env
# Edit .env with your MT5 credentials
```

## Configuration

Edit `.env`:
```ini
MT5_LOGIN=12345678
MT5_PASSWORD=your_password
MT5_SERVER=ICMarketsSC-Demo  # or ICMarketsSC-Live
MT5_MAGIC_NUMBER=123456
```

## Usage

### Screen for Pairs
```bash
python main.py screen --days 180
```

### Backtest
```bash
python main.py backtest --pair EURUSD,GBPUSD --days 365
```

### Optimize
```bash
python main.py optimize --pair EURUSD,GBPUSD --days 730
```

### Paper Trading
```bash
python main.py paper --pair EURUSD,GBPUSD --interval 60
```

### Live Trading
```bash
python main.py live --pair EURUSD,GBPUSD --interval 60
```

## Strategy Logic

### Pair Selection
1. **Correlation Filter**: Pearson correlation > 0.70
2. **Cointegration Test**: Engle-Granger ADF p-value < 0.05
3. **Mean Reversion**: Half-life < 50 bars

### Spread Construction
```
Spread = Price_A - β × Price_B
```
Where β is the hedge ratio from OLS regression.

### Entry Signals
- **Long Spread**: Z-score ≤ -2.0 (spread undervalued)
- **Short Spread**: Z-score ≥ +2.0 (spread overvalued)

### Exit Signals
- **Take Profit**: Z-score returns to ±0.2
- **Stop Loss**: Z-score reaches ±3.0 or correlation breakdown

## Risk Management

| Parameter | Default | Description |
|-----------|---------|-------------|
| Max Risk/Trade | 1% | Capital at risk per position |
| Max Exposure | 10% | Total portfolio exposure |
| Max Open Pairs | 3 | Concurrent positions |
| Max Drawdown | 15% | Halt threshold |
| Max Daily Loss | 3% | Daily loss limit |

## Default Pairs

| Pair | Rationale |
|------|-----------|
| EURUSD/GBPUSD | European majors |
| AUDUSD/NZDUSD | Oceanic currencies |
| EURJPY/USDJPY | Yen crosses |
| EURCHF/USDCHF | Swiss franc pairs |

## Backtest Costs

- **Spread**: 1.5 pips
- **Slippage**: 0.5 pips
- **Commission**: $7/lot

## API Reference

### MT5Client
```python
from src.data.broker_client import MT5Client, Timeframe

client = MT5Client(config)
client.connect()

# Get historical data
candles = client.get_candles("EURUSD", Timeframe.H1, 500)

# Execute order
result = client.market_order("EURUSD", OrderType.BUY, 0.1)
```

### PairsTradingSystem
```python
from main import PairsTradingSystem

system = PairsTradingSystem()

# Screen pairs
results = system.screen_pairs(days=180)

# Run backtest
backtest = system.run_backtest(("EURUSD", "GBPUSD"), days=365)
```

## Troubleshooting

### MT5 Connection Failed
- Verify MT5 terminal is running
- Check login credentials
- Ensure correct server name

### No Data Received
- Verify symbol names (check for suffix)
- Ensure symbol is in Market Watch
- Check internet connection

### Order Rejected
- Check account balance
- Verify symbol trading hours
- Check deviation settings

## Disclaimer

**This software is for educational purposes only.**

Trading forex involves substantial risk of loss. Past performance is not indicative of future results. Never trade with money you cannot afford to lose.

The authors are not responsible for any financial losses incurred through the use of this software.

## License

MIT License
