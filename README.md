# Pairs Trading System

A professional, modular pairs trading system for Forex markets using the OANDA API.

## Overview

This system implements statistical arbitrage through pairs trading - a market-neutral strategy that profits from the mean reversion of price spreads between correlated currency pairs.

### Key Features

- **Statistical Analysis**: Correlation analysis, cointegration testing (Engle-Granger, Johansen), Hurst exponent
- **Dynamic Spread Building**: Rolling hedge ratios, z-score calculation, mean reversion detection
- **Signal Generation**: Entry/exit signals based on z-score thresholds with correlation filters
- **Risk Management**: Position sizing, drawdown limits, exposure management
- **Walk-Forward Optimization**: Out-of-sample validated parameter optimization
- **Multiple Trading Modes**: Backtest, paper trading, and live trading
- **OANDA Integration**: Full API support for data fetching and order execution

## Project Structure

```
Pairs_Trading/
├── config/
│   ├── settings.py          # System configuration and parameters
│   └── broker_config.py     # OANDA API configuration
├── src/
│   ├── data/
│   │   ├── broker_client.py # OANDA REST API client
│   │   └── data_manager.py  # Data fetching and caching
│   ├── analysis/
│   │   ├── correlation.py   # Correlation analysis
│   │   ├── cointegration.py # Cointegration testing
│   │   └── spread_builder.py# Spread construction
│   ├── strategy/
│   │   ├── signals.py       # Signal generation
│   │   └── pairs_strategy.py# Main strategy logic
│   ├── risk/
│   │   └── risk_manager.py  # Risk management
│   ├── backtest/
│   │   └── backtest_engine.py# Backtesting engine
│   ├── optimization/
│   │   └── optimizer.py     # Walk-forward optimization
│   └── execution/
│       └── executor.py      # Live order execution
├── logs/
├── data/
│   ├── historical/
│   └── cache/
├── results/
│   ├── backtests/
│   └── optimization/
├── tests/
├── scripts/
├── main.py                   # Main entry point
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites

- Python 3.10 or higher
- OANDA account (practice or live)

### Setup

1. Clone or download the repository:
```bash
cd Pairs_Trading
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure OANDA credentials:

Create a `.env` file in the project root:
```env
OANDA_ACCOUNT_ID=your_account_id
OANDA_API_KEY=your_api_key
OANDA_ENVIRONMENT=practice  # or 'live'
```

Or create `config/broker_credentials.yaml`:
```yaml
account_id: "your_account_id"
api_key: "your_api_key"
environment: "practice"
```

## Quick Start

### 1. Screen for Tradeable Pairs

```bash
python main.py screen --days 180
```

This analyzes all major currency pairs and identifies those with:
- High correlation (>0.70)
- Statistical cointegration
- Suitable mean reversion characteristics (half-life < 50 bars)

### 2. Run a Backtest

```bash
python main.py backtest --pair EUR_USD,GBP_USD --days 365
```

Output includes:
- Total return and annualized return
- Sharpe ratio, Sortino ratio, Calmar ratio
- Maximum drawdown
- Win rate and profit factor
- Complete trade log

### 3. Walk-Forward Optimization

```bash
python main.py optimize --pair EUR_USD,GBP_USD --days 730
```

Performs rolling in-sample/out-of-sample optimization to find robust parameters.

### 4. Paper Trading

```bash
python main.py paper --pair EUR_USD,GBP_USD
```

Trades with real market data but simulated execution. Perfect for strategy validation.

### 5. Live Trading

```bash
python main.py live --pair EUR_USD,GBP_USD
```

**⚠️ WARNING: This executes real trades with real money!**

## Strategy Logic

### Entry Signals

- **Long Spread** (Buy A, Sell B): Z-score ≤ -2.0 AND correlation > 0.70
- **Short Spread** (Sell A, Buy B): Z-score ≥ +2.0 AND correlation > 0.70

### Exit Signals

- **Mean Reversion Exit**: Z-score returns to ±0.2 band
- **Stop Loss**: |Z-score| ≥ 3.0
- **Correlation Breakdown**: Correlation drops below 0.60

### Position Sizing

Risk-based sizing: each trade risks 1% of account equity, balanced across both legs using the hedge ratio.

## Configuration

Edit `config/settings.py` or create a YAML configuration file:

```yaml
# spread_params
entry_zscore: 2.0
exit_zscore: 0.2
stop_loss_zscore: 3.0
regression_window: 120
zscore_window: 60
min_correlation: 0.70
max_half_life: 50

# risk_params
max_risk_per_trade: 0.01
max_total_exposure: 0.10
max_open_pairs: 3
max_drawdown: 0.15

# backtest_params
initial_capital: 10000
spread_cost_pips: 1.5
slippage_pips: 0.5
```

## Default Pairs Universe

The system comes configured with these correlated pairs:

| Pair A    | Pair B    | Rationale                    |
|-----------|-----------|------------------------------|
| EUR_USD   | GBP_USD   | Dollar-based majors          |
| AUD_USD   | NZD_USD   | Commodity currencies         |
| EUR_JPY   | USD_JPY   | Yen crosses                  |
| EUR_CHF   | USD_CHF   | Franc crosses                |
| GBP_USD   | EUR_GBP   | Sterling relationships       |

## Performance Metrics

The system calculates comprehensive metrics:

- **Returns**: Total, annualized, risk-adjusted
- **Risk**: Sharpe ratio, Sortino ratio, Calmar ratio
- **Drawdown**: Maximum drawdown, duration
- **Trade Stats**: Win rate, profit factor, expectancy
- **Costs**: Transaction costs, slippage impact

## API Usage

```python
from config.settings import Settings
from config.broker_config import BrokerConfig
from main import PairsTradingSystem

# Initialize
settings = Settings()
broker_config = BrokerConfig.from_env()
system = PairsTradingSystem(settings, broker_config)

# Run backtest
from datetime import datetime, timedelta
end_date = datetime.now()
start_date = end_date - timedelta(days=365)

result = system.run_backtest(
    pair=('EUR_USD', 'GBP_USD'),
    start_date=start_date,
    end_date=end_date
)

print(result.summary())
```

## Testing

```bash
pytest tests/ -v --cov=src
```

## Logging

Logs are stored in `logs/` directory. Configure logging level:

```bash
python main.py backtest --pair EUR_USD,GBP_USD --log-level DEBUG --log-file logs/backtest.log
```

## Important Notes

1. **Paper Trade First**: Always validate strategies with paper trading before going live.

2. **Risk Management**: Never risk more than you can afford to lose. The default 1% risk per trade is conservative.

3. **Market Hours**: Forex markets are open 24/5. The system handles this automatically.

4. **Slippage**: Real execution may differ from backtest due to slippage and variable spreads.

5. **Cointegration Changes**: Pairs can lose their statistical relationships. Monitor regularly.

## Troubleshooting

### "Insufficient data" error
- Ensure OANDA credentials are correct
- Check if the instruments are available in your account type
- Increase `--days` parameter for more data

### "Connection refused" error
- Check internet connection
- Verify OANDA API status
- Ensure firewall allows outbound HTTPS

### Low Sharpe ratio in backtest
- Try different pairs from the screener
- Adjust z-score thresholds
- Run walk-forward optimization

## License

MIT License - See LICENSE file for details.

## Disclaimer

This software is for educational purposes only. Trading foreign exchange carries a high level of risk and may not be suitable for all investors. Past performance is not indicative of future results. The developers assume no responsibility for any trading losses.
