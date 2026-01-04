"""
Broker Configuration for MetaTrader 5 (IC Markets Global).

Handles MT5 connection settings and credentials.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class MT5Config:
    """MetaTrader 5 configuration for IC Markets Global."""
    
    # Connection settings
    login: int = 0
    password: str = ""
    server: str = "ICMarketsSC-Demo"  # or "ICMarketsSC-Live"
    
    # MT5 terminal path (optional, auto-detected if not specified)
    terminal_path: Optional[str] = None
    
    # Timeout settings
    timeout: int = 60000  # milliseconds
    
    # Trading settings
    magic_number: int = 123456  # Unique identifier for EA orders
    deviation: int = 20  # Maximum price deviation in points
    
    # Symbol suffix (some brokers add suffix like ".a" or "m")
    symbol_suffix: str = ""
    
    @classmethod
    def from_env(cls) -> 'MT5Config':
        """Load configuration from environment variables."""
        from dotenv import load_dotenv
        load_dotenv()
        
        login = os.getenv('MT5_LOGIN', '0')
        
        return cls(
            login=int(login) if login else 0,
            password=os.getenv('MT5_PASSWORD', ''),
            server=os.getenv('MT5_SERVER', 'ICMarketsSC-Demo'),
            terminal_path=os.getenv('MT5_TERMINAL_PATH'),
            timeout=int(os.getenv('MT5_TIMEOUT', '60000')),
            magic_number=int(os.getenv('MT5_MAGIC_NUMBER', '123456')),
            deviation=int(os.getenv('MT5_DEVIATION', '20')),
            symbol_suffix=os.getenv('MT5_SYMBOL_SUFFIX', '')
        )
    
    @classmethod
    def from_yaml(cls, path: str) -> 'MT5Config':
        """Load configuration from YAML file."""
        import yaml
        
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        
        mt5_config = config.get('mt5', {})
        
        return cls(
            login=mt5_config.get('login', 0),
            password=mt5_config.get('password', ''),
            server=mt5_config.get('server', 'ICMarketsSC-Demo'),
            terminal_path=mt5_config.get('terminal_path'),
            timeout=mt5_config.get('timeout', 60000),
            magic_number=mt5_config.get('magic_number', 123456),
            deviation=mt5_config.get('deviation', 20),
            symbol_suffix=mt5_config.get('symbol_suffix', '')
        )
    
    def get_symbol(self, base_symbol: str) -> str:
        """Get full symbol name with suffix if applicable."""
        return f"{base_symbol}{self.symbol_suffix}"
    
    def validate(self) -> bool:
        """Validate configuration."""
        if self.login <= 0:
            raise ValueError("MT5 login must be a positive integer")
        if not self.password:
            raise ValueError("MT5 password is required")
        if not self.server:
            raise ValueError("MT5 server is required")
        return True


@dataclass
class SymbolInfo:
    """Information about a trading symbol."""
    name: str
    digits: int
    point: float
    trade_tick_size: float
    trade_tick_value: float
    volume_min: float
    volume_max: float
    volume_step: float
    trade_contract_size: float
    spread: int
    swap_long: float
    swap_short: float
    margin_initial: float
    currency_base: str
    currency_profit: str
    description: str


# Default Forex pairs for IC Markets
IC_MARKETS_FOREX_PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
    "AUDUSD", "NZDUSD", "USDCAD",
    "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY",
    "EURGBP", "EURAUD", "EURNZD", "EURCHF",
    "GBPAUD", "GBPNZD", "GBPCHF", "GBPCAD",
    "AUDNZD", "AUDCAD", "AUDCHF",
    "NZDCAD", "NZDCHF", "CADCHF", "CADJPY"
]

# Default pairs for Pairs Trading analysis
DEFAULT_PAIR_CANDIDATES = [
    ("EURUSD", "GBPUSD"),    # European majors
    ("AUDUSD", "NZDUSD"),    # Oceanic
    ("EURJPY", "USDJPY"),    # Yen crosses
    ("EURCHF", "USDCHF"),    # Swiss crosses
    ("EURGBP", "GBPUSD"),    # GBP related
    ("AUDUSD", "USDCAD"),    # Commodity currencies
    ("EURUSD", "USDCHF"),    # Inverse relationship
    ("GBPJPY", "USDJPY"),    # Yen crosses
]
