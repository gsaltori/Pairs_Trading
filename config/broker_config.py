"""
Broker configuration for OANDA API.

OANDA is chosen for its:
- Reliable REST API
- Good documentation
- Practice accounts available
- Reasonable spreads
- Python SDK available (oandapyV20)
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum
import os
from pathlib import Path


class OandaEnvironment(Enum):
    """OANDA API environments."""
    PRACTICE = "practice"
    LIVE = "live"


@dataclass
class BrokerConfig:
    """OANDA broker configuration."""
    
    # API credentials (loaded from environment variables for security)
    api_key: str = ""
    account_id: str = ""
    
    # Environment
    environment: OandaEnvironment = OandaEnvironment.PRACTICE
    
    # API URLs
    @property
    def api_url(self) -> str:
        """Get API URL based on environment."""
        if self.environment == OandaEnvironment.PRACTICE:
            return "https://api-fxpractice.oanda.com"
        return "https://api-fxtrade.oanda.com"
    
    @property
    def stream_url(self) -> str:
        """Get streaming URL based on environment."""
        if self.environment == OandaEnvironment.PRACTICE:
            return "https://stream-fxpractice.oanda.com"
        return "https://stream-fxtrade.oanda.com"
    
    # Request settings
    request_timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Rate limiting
    requests_per_second: int = 10
    
    # Data fetching
    max_candles_per_request: int = 5000
    
    @classmethod
    def from_env(cls) -> 'BrokerConfig':
        """
        Load configuration from environment variables.
        
        Required environment variables:
        - OANDA_API_KEY: Your OANDA API key
        - OANDA_ACCOUNT_ID: Your OANDA account ID
        - OANDA_ENVIRONMENT: 'practice' or 'live' (default: practice)
        """
        config = cls()
        config.api_key = os.getenv('OANDA_API_KEY', '')
        config.account_id = os.getenv('OANDA_ACCOUNT_ID', '')
        
        env_str = os.getenv('OANDA_ENVIRONMENT', 'practice').lower()
        config.environment = OandaEnvironment(env_str)
        
        return config
    
    @classmethod
    def from_file(cls, filepath: Path) -> 'BrokerConfig':
        """
        Load configuration from a credentials file.
        
        File format (JSON):
        {
            "api_key": "your-api-key",
            "account_id": "your-account-id",
            "environment": "practice"
        }
        """
        import json
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        config = cls()
        config.api_key = data.get('api_key', '')
        config.account_id = data.get('account_id', '')
        config.environment = OandaEnvironment(data.get('environment', 'practice'))
        
        return config
    
    def validate(self) -> bool:
        """Validate that required credentials are present."""
        if not self.api_key:
            raise ValueError("OANDA API key not configured. Set OANDA_API_KEY environment variable.")
        if not self.account_id:
            raise ValueError("OANDA account ID not configured. Set OANDA_ACCOUNT_ID environment variable.")
        return True
    
    def save_template(self, filepath: Path) -> None:
        """Save a template credentials file."""
        import json
        
        template = {
            "api_key": "YOUR_API_KEY_HERE",
            "account_id": "YOUR_ACCOUNT_ID_HERE",
            "environment": "practice"
        }
        
        with open(filepath, 'w') as f:
            json.dump(template, f, indent=2)
        
        print(f"Template saved to {filepath}")
        print("IMPORTANT: Add this file to .gitignore!")


# Forex instrument specifications for OANDA
OANDA_INSTRUMENTS = {
    "EUR_USD": {"pip_location": -4, "display_precision": 5, "trade_units_precision": 0},
    "GBP_USD": {"pip_location": -4, "display_precision": 5, "trade_units_precision": 0},
    "USD_JPY": {"pip_location": -2, "display_precision": 3, "trade_units_precision": 0},
    "USD_CHF": {"pip_location": -4, "display_precision": 5, "trade_units_precision": 0},
    "AUD_USD": {"pip_location": -4, "display_precision": 5, "trade_units_precision": 0},
    "NZD_USD": {"pip_location": -4, "display_precision": 5, "trade_units_precision": 0},
    "EUR_GBP": {"pip_location": -4, "display_precision": 5, "trade_units_precision": 0},
    "EUR_JPY": {"pip_location": -2, "display_precision": 3, "trade_units_precision": 0},
    "EUR_CHF": {"pip_location": -4, "display_precision": 5, "trade_units_precision": 0},
    "GBP_JPY": {"pip_location": -2, "display_precision": 3, "trade_units_precision": 0},
    "GBP_CHF": {"pip_location": -4, "display_precision": 5, "trade_units_precision": 0},
    "AUD_NZD": {"pip_location": -4, "display_precision": 5, "trade_units_precision": 0},
    "USD_CAD": {"pip_location": -4, "display_precision": 5, "trade_units_precision": 0},
}


def get_pip_value(instrument: str, units: float, current_price: float, account_currency: str = "USD") -> float:
    """
    Calculate pip value for a given instrument and position size.
    
    Args:
        instrument: OANDA instrument name (e.g., 'EUR_USD')
        units: Position size in units
        current_price: Current price of the instrument
        account_currency: Account base currency
    
    Returns:
        Pip value in account currency
    """
    if instrument not in OANDA_INSTRUMENTS:
        raise ValueError(f"Unknown instrument: {instrument}")
    
    spec = OANDA_INSTRUMENTS[instrument]
    pip_location = spec['pip_location']
    pip_size = 10 ** pip_location
    
    base_currency = instrument[:3]
    quote_currency = instrument[4:]
    
    # Standard pip value calculation
    pip_value = units * pip_size
    
    # Convert to account currency if needed
    if quote_currency != account_currency:
        # Would need current exchange rate for conversion
        # For simplicity, assuming USD account
        if account_currency == "USD":
            if quote_currency == "JPY":
                # Need USD/JPY rate
                pip_value = pip_value / current_price
            elif base_currency == "USD":
                pip_value = pip_value
            else:
                # More complex conversion needed
                pass
    
    return pip_value
