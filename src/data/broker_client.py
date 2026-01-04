"""
OANDA Broker Client

Handles all communication with the OANDA REST API:
- Historical data fetching
- Real-time price streaming
- Order management
- Account information
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import time
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import sys
sys.path.append(str(__file__).rsplit('\\', 3)[0])

from config.broker_config import BrokerConfig, OandaEnvironment, OANDA_INSTRUMENTS
from config.settings import Timeframe


logger = logging.getLogger(__name__)


class OandaClient:
    """
    Client for interacting with OANDA's REST API.
    
    Provides methods for:
    - Fetching historical candlestick data
    - Placing and managing orders
    - Getting account information
    - Real-time price streaming
    """
    
    def __init__(self, config: Optional[BrokerConfig] = None):
        """
        Initialize the OANDA client.
        
        Args:
            config: BrokerConfig instance. If None, loads from environment.
        """
        self.config = config or BrokerConfig.from_env()
        self._session: Optional[requests.Session] = None
        self._last_request_time: float = 0.0
        
    @property
    def session(self) -> requests.Session:
        """Get or create HTTP session with retry logic."""
        if self._session is None:
            self._session = requests.Session()
            
            # Configure retries
            retry_strategy = Retry(
                total=self.config.max_retries,
                backoff_factor=self.config.retry_delay,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE"]
            )
            
            adapter = HTTPAdapter(max_retries=retry_strategy)
            self._session.mount("https://", adapter)
            self._session.mount("http://", adapter)
            
            # Set default headers
            self._session.headers.update({
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "Accept-Datetime-Format": "RFC3339"
            })
        
        return self._session
    
    def _rate_limit(self) -> None:
        """Implement rate limiting to avoid API throttling."""
        min_interval = 1.0 / self.config.requests_per_second
        elapsed = time.time() - self._last_request_time
        
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        
        self._last_request_time = time.time()
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        stream: bool = False
    ) -> Dict:
        """
        Make an API request with error handling.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            params: Query parameters
            data: Request body data
            stream: Whether to stream response
            
        Returns:
            Response data as dictionary
        """
        self._rate_limit()
        
        url = f"{self.config.api_url}{endpoint}"
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=data,
                timeout=self.config.request_timeout,
                stream=stream
            )
            
            response.raise_for_status()
            
            if stream:
                return response
            
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP Error: {e.response.status_code} - {e.response.text}")
            raise
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection Error: {e}")
            raise
        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout Error: {e}")
            raise
        except Exception as e:
            logger.error(f"Request Error: {e}")
            raise
    
    def get_account_info(self) -> Dict:
        """
        Get account information.
        
        Returns:
            Account details including balance, margin, positions
        """
        self.config.validate()
        endpoint = f"/v3/accounts/{self.config.account_id}"
        return self._make_request("GET", endpoint)
    
    def get_account_summary(self) -> Dict:
        """
        Get account summary.
        
        Returns:
            Account summary with balance, NAV, unrealized P&L
        """
        self.config.validate()
        endpoint = f"/v3/accounts/{self.config.account_id}/summary"
        return self._make_request("GET", endpoint)
    
    def get_instruments(self) -> List[Dict]:
        """
        Get available instruments.
        
        Returns:
            List of tradeable instruments with specifications
        """
        self.config.validate()
        endpoint = f"/v3/accounts/{self.config.account_id}/instruments"
        response = self._make_request("GET", endpoint)
        return response.get('instruments', [])
    
    def get_candles(
        self,
        instrument: str,
        granularity: str = "H1",
        count: Optional[int] = None,
        from_time: Optional[datetime] = None,
        to_time: Optional[datetime] = None,
        price: str = "M"  # M=mid, B=bid, A=ask
    ) -> pd.DataFrame:
        """
        Fetch historical candlestick data.
        
        Args:
            instrument: Instrument name (e.g., 'EUR_USD')
            granularity: Timeframe (M1, M5, M15, M30, H1, H4, D)
            count: Number of candles (max 5000)
            from_time: Start datetime
            to_time: End datetime
            price: Price type (M=mid, B=bid, A=ask)
            
        Returns:
            DataFrame with OHLCV data
        """
        self.config.validate()
        endpoint = f"/v3/instruments/{instrument}/candles"
        
        params = {
            "granularity": granularity,
            "price": price
        }
        
        if count:
            params["count"] = min(count, self.config.max_candles_per_request)
        
        if from_time:
            params["from"] = from_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        if to_time:
            params["to"] = to_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        response = self._make_request("GET", endpoint, params=params)
        
        candles = response.get('candles', [])
        
        if not candles:
            return pd.DataFrame()
        
        # Parse candles into DataFrame
        data = []
        for candle in candles:
            if candle.get('complete', True):  # Only complete candles
                mid = candle.get('mid', {})
                row = {
                    'time': pd.to_datetime(candle['time']),
                    'open': float(mid.get('o', 0)),
                    'high': float(mid.get('h', 0)),
                    'low': float(mid.get('l', 0)),
                    'close': float(mid.get('c', 0)),
                    'volume': int(candle.get('volume', 0))
                }
                data.append(row)
        
        df = pd.DataFrame(data)
        
        if not df.empty:
            df.set_index('time', inplace=True)
            df.sort_index(inplace=True)
        
        return df
    
    def get_historical_data(
        self,
        instrument: str,
        timeframe: Timeframe,
        start_date: datetime,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch historical data with automatic pagination.
        
        Args:
            instrument: Instrument name
            timeframe: Timeframe enum
            start_date: Start datetime
            end_date: End datetime (default: now)
            
        Returns:
            DataFrame with complete historical data
        """
        if end_date is None:
            end_date = datetime.utcnow()
        
        granularity = timeframe.oanda_granularity
        all_data = []
        current_start = start_date
        
        logger.info(f"Fetching {instrument} data from {start_date} to {end_date}")
        
        while current_start < end_date:
            df = self.get_candles(
                instrument=instrument,
                granularity=granularity,
                from_time=current_start,
                to_time=end_date,
                count=self.config.max_candles_per_request
            )
            
            if df.empty:
                break
            
            all_data.append(df)
            
            # Move start to last candle time
            last_time = df.index[-1].to_pydatetime()
            
            if last_time <= current_start:
                break
            
            current_start = last_time + timedelta(minutes=timeframe.minutes)
            
            logger.debug(f"Fetched {len(df)} candles, last: {last_time}")
        
        if not all_data:
            return pd.DataFrame()
        
        # Combine all data
        result = pd.concat(all_data)
        result = result[~result.index.duplicated(keep='first')]
        result.sort_index(inplace=True)
        
        logger.info(f"Total candles fetched: {len(result)}")
        
        return result
    
    def get_current_price(self, instrument: str) -> Dict[str, float]:
        """
        Get current bid/ask prices.
        
        Args:
            instrument: Instrument name
            
        Returns:
            Dictionary with bid, ask, and mid prices
        """
        self.config.validate()
        endpoint = f"/v3/accounts/{self.config.account_id}/pricing"
        params = {"instruments": instrument}
        
        response = self._make_request("GET", endpoint, params=params)
        
        prices = response.get('prices', [])
        if not prices:
            raise ValueError(f"No price data for {instrument}")
        
        price_data = prices[0]
        
        bid = float(price_data['bids'][0]['price'])
        ask = float(price_data['asks'][0]['price'])
        
        return {
            'bid': bid,
            'ask': ask,
            'mid': (bid + ask) / 2,
            'spread': ask - bid,
            'time': pd.to_datetime(price_data['time'])
        }
    
    def get_current_prices(self, instruments: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Get current prices for multiple instruments.
        
        Args:
            instruments: List of instrument names
            
        Returns:
            Dictionary mapping instrument to price data
        """
        self.config.validate()
        endpoint = f"/v3/accounts/{self.config.account_id}/pricing"
        params = {"instruments": ",".join(instruments)}
        
        response = self._make_request("GET", endpoint, params=params)
        
        result = {}
        for price_data in response.get('prices', []):
            instrument = price_data['instrument']
            bid = float(price_data['bids'][0]['price'])
            ask = float(price_data['asks'][0]['price'])
            
            result[instrument] = {
                'bid': bid,
                'ask': ask,
                'mid': (bid + ask) / 2,
                'spread': ask - bid,
                'time': pd.to_datetime(price_data['time'])
            }
        
        return result
    
    def place_market_order(
        self,
        instrument: str,
        units: int,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Dict:
        """
        Place a market order.
        
        Args:
            instrument: Instrument name
            units: Position size (positive for long, negative for short)
            stop_loss: Stop loss price
            take_profit: Take profit price
            
        Returns:
            Order response with execution details
        """
        self.config.validate()
        endpoint = f"/v3/accounts/{self.config.account_id}/orders"
        
        order_data = {
            "order": {
                "type": "MARKET",
                "instrument": instrument,
                "units": str(units),
                "timeInForce": "FOK",  # Fill or Kill
                "positionFill": "DEFAULT"
            }
        }
        
        if stop_loss:
            order_data["order"]["stopLossOnFill"] = {
                "price": f"{stop_loss:.5f}"
            }
        
        if take_profit:
            order_data["order"]["takeProfitOnFill"] = {
                "price": f"{take_profit:.5f}"
            }
        
        response = self._make_request("POST", endpoint, data=order_data)
        
        logger.info(f"Market order placed: {instrument} {units} units")
        
        return response
    
    def place_limit_order(
        self,
        instrument: str,
        units: int,
        price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        gtd_time: Optional[datetime] = None
    ) -> Dict:
        """
        Place a limit order.
        
        Args:
            instrument: Instrument name
            units: Position size
            price: Limit price
            stop_loss: Stop loss price
            take_profit: Take profit price
            gtd_time: Good till date
            
        Returns:
            Order response
        """
        self.config.validate()
        endpoint = f"/v3/accounts/{self.config.account_id}/orders"
        
        order_data = {
            "order": {
                "type": "LIMIT",
                "instrument": instrument,
                "units": str(units),
                "price": f"{price:.5f}",
                "timeInForce": "GTC" if not gtd_time else "GTD",
                "positionFill": "DEFAULT"
            }
        }
        
        if gtd_time:
            order_data["order"]["gtdTime"] = gtd_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        if stop_loss:
            order_data["order"]["stopLossOnFill"] = {
                "price": f"{stop_loss:.5f}"
            }
        
        if take_profit:
            order_data["order"]["takeProfitOnFill"] = {
                "price": f"{take_profit:.5f}"
            }
        
        return self._make_request("POST", endpoint, data=order_data)
    
    def close_position(self, instrument: str, units: Optional[int] = None) -> Dict:
        """
        Close a position (fully or partially).
        
        Args:
            instrument: Instrument name
            units: Units to close (None = close all)
            
        Returns:
            Close response
        """
        self.config.validate()
        endpoint = f"/v3/accounts/{self.config.account_id}/positions/{instrument}/close"
        
        data = {}
        if units:
            if units > 0:
                data["longUnits"] = str(units)
            else:
                data["shortUnits"] = str(abs(units))
        else:
            data["longUnits"] = "ALL"
            data["shortUnits"] = "ALL"
        
        response = self._make_request("PUT", endpoint, data=data)
        
        logger.info(f"Position closed: {instrument}")
        
        return response
    
    def get_open_positions(self) -> List[Dict]:
        """
        Get all open positions.
        
        Returns:
            List of open positions
        """
        self.config.validate()
        endpoint = f"/v3/accounts/{self.config.account_id}/openPositions"
        
        response = self._make_request("GET", endpoint)
        return response.get('positions', [])
    
    def get_position(self, instrument: str) -> Optional[Dict]:
        """
        Get position for a specific instrument.
        
        Args:
            instrument: Instrument name
            
        Returns:
            Position details or None
        """
        self.config.validate()
        endpoint = f"/v3/accounts/{self.config.account_id}/positions/{instrument}"
        
        try:
            response = self._make_request("GET", endpoint)
            return response.get('position')
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return None
            raise
    
    def get_pending_orders(self) -> List[Dict]:
        """
        Get all pending orders.
        
        Returns:
            List of pending orders
        """
        self.config.validate()
        endpoint = f"/v3/accounts/{self.config.account_id}/pendingOrders"
        
        response = self._make_request("GET", endpoint)
        return response.get('orders', [])
    
    def cancel_order(self, order_id: str) -> Dict:
        """
        Cancel a pending order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            Cancel response
        """
        self.config.validate()
        endpoint = f"/v3/accounts/{self.config.account_id}/orders/{order_id}/cancel"
        
        return self._make_request("PUT", endpoint)
    
    def get_trades(self, instrument: Optional[str] = None) -> List[Dict]:
        """
        Get open trades.
        
        Args:
            instrument: Filter by instrument (optional)
            
        Returns:
            List of open trades
        """
        self.config.validate()
        endpoint = f"/v3/accounts/{self.config.account_id}/openTrades"
        
        params = {}
        if instrument:
            params["instrument"] = instrument
        
        response = self._make_request("GET", endpoint, params=params)
        return response.get('trades', [])
    
    def close_trade(self, trade_id: str, units: Optional[int] = None) -> Dict:
        """
        Close a specific trade.
        
        Args:
            trade_id: Trade ID to close
            units: Units to close (None = all)
            
        Returns:
            Close response
        """
        self.config.validate()
        endpoint = f"/v3/accounts/{self.config.account_id}/trades/{trade_id}/close"
        
        data = {}
        if units:
            data["units"] = str(units)
        
        return self._make_request("PUT", endpoint, data=data if data else None)
    
    def get_transaction_history(
        self,
        from_time: Optional[datetime] = None,
        to_time: Optional[datetime] = None,
        page_size: int = 100
    ) -> List[Dict]:
        """
        Get transaction history.
        
        Args:
            from_time: Start datetime
            to_time: End datetime
            page_size: Number of transactions per page
            
        Returns:
            List of transactions
        """
        self.config.validate()
        endpoint = f"/v3/accounts/{self.config.account_id}/transactions"
        
        params = {"pageSize": page_size}
        
        if from_time:
            params["from"] = from_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        if to_time:
            params["to"] = to_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        response = self._make_request("GET", endpoint, params=params)
        return response.get('transactions', [])
    
    def close(self) -> None:
        """Close the HTTP session."""
        if self._session:
            self._session.close()
            self._session = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
