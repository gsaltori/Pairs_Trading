"""
Observation data structures and stream management.

This module handles raw market data ingestion.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Tuple, Generator
import numpy as np
import hashlib
import json


@dataclass(frozen=True)
class OHLCBar:
    """
    Single OHLC bar - immutable.
    
    Represents one candlestick of price data.
    """
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    
    def __post_init__(self):
        """Validate OHLC consistency."""
        if not (self.low <= min(self.open, self.close)):
            raise ValueError(f"Invalid OHLC: low={self.low} > min(open={self.open}, close={self.close})")
        if not (self.high >= max(self.open, self.close)):
            raise ValueError(f"Invalid OHLC: high={self.high} < max(open={self.open}, close={self.close})")


@dataclass(frozen=True)
class MarketObservation:
    """
    Complete market observation at a single timestamp.
    
    Contains OHLC data for both symbols in the pair.
    IMMUTABLE after creation.
    """
    observation_id: str
    timestamp: datetime
    timeframe: str
    
    # OHLC data for each symbol
    bar_a: OHLCBar
    bar_b: OHLCBar
    
    # Capture metadata
    capture_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Integrity hash
    integrity_hash: str = field(default='')
    
    def __post_init__(self):
        """Compute integrity hash if not provided."""
        if not self.integrity_hash:
            hash_val = self._compute_hash()
            object.__setattr__(self, 'integrity_hash', hash_val)
    
    def _compute_hash(self) -> str:
        """Compute SHA-256 hash of observation content."""
        data = {
            'id': self.observation_id,
            'timestamp': self.timestamp.isoformat(),
            'bar_a_close': self.bar_a.close,
            'bar_b_close': self.bar_b.close,
        }
        content = json.dumps(data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:32]
    
    def verify_integrity(self) -> bool:
        """Verify observation has not been tampered with."""
        return self.integrity_hash == self._compute_hash()


def create_observation_id(timestamp: datetime, timeframe: str) -> str:
    """Create unique observation ID."""
    return f"OBS-{timestamp.strftime('%Y%m%d')}-{timestamp.strftime('%H%M')}-{timeframe}"


class ObservationStream:
    """
    Stream of market observations.
    
    Manages sequential observation generation and retrieval.
    """
    
    def __init__(self, timeframe: str = "H4"):
        self.timeframe = timeframe
        self._observations: List[MarketObservation] = []
        self._index = 0
    
    def add_observation(self, obs: MarketObservation) -> None:
        """Add observation to stream."""
        self._observations.append(obs)
    
    def get_latest(self) -> Optional[MarketObservation]:
        """Get the most recent observation."""
        if not self._observations:
            return None
        return self._observations[-1]
    
    def get_history(self, n: int) -> List[MarketObservation]:
        """Get last n observations."""
        return self._observations[-n:] if self._observations else []
    
    def __len__(self) -> int:
        return len(self._observations)
    
    def __iter__(self):
        return iter(self._observations)
    
    def __getitem__(self, idx: int) -> MarketObservation:
        return self._observations[idx]


# =============================================================================
# MOCK: SYNTHETIC DATA GENERATOR
# =============================================================================

class SyntheticDataGenerator:
    """
    MOCK: Generates synthetic FX price data for testing.
    
    This is NOT real data. It is designed to produce:
    - Correlated price movements (for spread calculation)
    - Occasional Z-score breaches (for prediction generation)
    - Rare structural breaks (for invalidation testing)
    
    Uses a factor model approach:
    - Common factor drives correlated moves
    - Small idiosyncratic factors create spread
    - Mean-reverting spread dynamics
    """
    
    def __init__(
        self,
        symbol_a: str = "EURUSD",
        symbol_b: str = "GBPUSD",
        timeframe: str = "H4",
        seed: int = 42,
    ):
        self.symbol_a = symbol_a
        self.symbol_b = symbol_b
        self.timeframe = timeframe
        self.rng = np.random.default_rng(seed)
        
        # Initial prices
        self.price_a = 1.1000
        self.price_b = 1.2500
        
        # Common factor loading (high correlation)
        self.common_factor_loading = 0.95  # Both pairs move together ~95%
        
        # Volatility parameters
        self.common_volatility = 0.0025  # Common factor volatility
        self.idio_volatility = 0.0008    # Idiosyncratic volatility
        
        # Mean-reverting spread tracking
        self.cumulative_spread = 0.0
        self.spread_mean_reversion = 0.03  # 3% pull back per bar
        
        # Dislocation state
        self.in_dislocation = False
        self.dislocation_bars_remaining = 0
    
    def generate_bars(
        self,
        n_bars: int,
        start_time: Optional[datetime] = None,
    ) -> Generator[Tuple[OHLCBar, OHLCBar], None, None]:
        """
        Generate n_bars of synthetic OHLC data.
        
        MOCK DATA - NOT REAL MARKET DATA
        
        Yields: (bar_a, bar_b) tuples
        """
        if start_time is None:
            start_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        
        # Time increment based on timeframe
        if self.timeframe == "H4":
            delta = timedelta(hours=4)
        elif self.timeframe == "H1":
            delta = timedelta(hours=1)
        else:
            delta = timedelta(hours=4)
        
        current_time = start_time
        
        for i in range(n_bars):
            # Generate common factor return (drives correlation)
            common_return = self.rng.normal(0, self.common_volatility)
            
            # Generate idiosyncratic returns (small, uncorrelated)
            idio_a = self.rng.normal(0, self.idio_volatility)
            idio_b = self.rng.normal(0, self.idio_volatility)
            
            # Base returns (highly correlated)
            return_a = self.common_factor_loading * common_return + idio_a
            return_b = self.common_factor_loading * common_return + idio_b
            
            # Mean reversion pull on accumulated spread
            spread_reversion = -self.spread_mean_reversion * self.cumulative_spread
            return_a += spread_reversion * 0.5
            return_b -= spread_reversion * 0.5
            
            # Check for dislocation event (spread moves away from mean)
            if not self.in_dislocation and self.rng.random() < 0.04:  # 4% chance
                self.in_dislocation = True
                self.dislocation_bars_remaining = self.rng.integers(3, 20)
                dislocation_shock = self.rng.choice([-1, 1]) * self.rng.uniform(0.004, 0.010)
                return_a += dislocation_shock
            
            if self.in_dislocation:
                self.dislocation_bars_remaining -= 1
                if self.dislocation_bars_remaining <= 0:
                    self.in_dislocation = False
            
            # Rare structural break (large correlation change)
            if self.rng.random() < 0.005:  # 0.5% chance
                break_shock = self.rng.normal(0, 0.015)
                return_a += break_shock
                return_b -= break_shock * 0.3
            
            # Update prices
            new_price_a = self.price_a * (1 + return_a)
            new_price_b = self.price_b * (1 + return_b)
            
            # Track cumulative spread (for mean reversion dynamics)
            self.cumulative_spread += (return_a - return_b)
            
            # Generate OHLC from close-to-close
            bar_a = self._make_ohlc(
                symbol=self.symbol_a,
                timestamp=current_time,
                open_price=self.price_a,
                close_price=new_price_a,
            )
            
            bar_b = self._make_ohlc(
                symbol=self.symbol_b,
                timestamp=current_time,
                open_price=self.price_b,
                close_price=new_price_b,
            )
            
            # Update state
            self.price_a = new_price_a
            self.price_b = new_price_b
            current_time += delta
            
            yield bar_a, bar_b
    
    def _make_ohlc(
        self,
        symbol: str,
        timestamp: datetime,
        open_price: float,
        close_price: float,
    ) -> OHLCBar:
        """Create OHLC bar with synthetic high/low."""
        # Add intrabar volatility
        intrabar_vol = abs(close_price - open_price) * 0.5
        
        high = max(open_price, close_price) + self.rng.uniform(0, intrabar_vol + 0.0005)
        low = min(open_price, close_price) - self.rng.uniform(0, intrabar_vol + 0.0005)
        
        return OHLCBar(
            symbol=symbol,
            timestamp=timestamp,
            open=round(open_price, 5),
            high=round(high, 5),
            low=round(low, 5),
            close=round(close_price, 5),
        )


def generate_observation_stream(
    n_bars: int,
    symbol_a: str = "EURUSD",
    symbol_b: str = "GBPUSD",
    timeframe: str = "H4",
    seed: int = 42,
) -> ObservationStream:
    """
    MOCK: Generate a complete observation stream with synthetic data.
    
    This is a convenience function for testing.
    """
    generator = SyntheticDataGenerator(
        symbol_a=symbol_a,
        symbol_b=symbol_b,
        timeframe=timeframe,
        seed=seed,
    )
    
    stream = ObservationStream(timeframe=timeframe)
    
    for bar_a, bar_b in generator.generate_bars(n_bars):
        obs_id = create_observation_id(bar_a.timestamp, timeframe)
        
        obs = MarketObservation(
            observation_id=obs_id,
            timestamp=bar_a.timestamp,
            timeframe=timeframe,
            bar_a=bar_a,
            bar_b=bar_b,
        )
        
        stream.add_observation(obs)
    
    return stream
