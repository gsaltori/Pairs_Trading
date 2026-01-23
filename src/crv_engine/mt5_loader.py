"""
MT5 Data Loader for CRV Engine

Loads aligned OHLC data from MetaTrader 5 for two symbols.
"""

import MetaTrader5 as mt5
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Tuple
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from observations import OHLCBar, MarketObservation, create_observation_id


# Timeframe mapping
TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
    "W1": mt5.TIMEFRAME_W1,
}

# Expected bar duration in hours
TIMEFRAME_HOURS = {
    "M1": 1/60,
    "M5": 5/60,
    "M15": 15/60,
    "M30": 30/60,
    "H1": 1,
    "H4": 4,
    "D1": 24,
    "W1": 168,
}


def initialize_mt5(
    terminal_path: Optional[str] = None,
    login: Optional[int] = None,
    password: Optional[str] = None,
    server: Optional[str] = None,
) -> bool:
    """
    Initialize MT5 connection.
    
    Returns True if successful, False otherwise.
    """
    kwargs = {}
    if terminal_path:
        kwargs['path'] = terminal_path
    if login:
        kwargs['login'] = login
    if password:
        kwargs['password'] = password
    if server:
        kwargs['server'] = server
    
    if not mt5.initialize(**kwargs):
        print(f"MT5 initialization failed: {mt5.last_error()}")
        return False
    
    print(f"MT5 initialized: {mt5.terminal_info().name}")
    return True


def shutdown_mt5():
    """Shutdown MT5 connection."""
    mt5.shutdown()


def load_ohlc(
    symbol: str,
    timeframe: str,
    n_bars: int,
) -> Optional[List[OHLCBar]]:
    """
    Load OHLC data for a single symbol.
    
    Args:
        symbol: MT5 symbol name (e.g., "EURUSD")
        timeframe: Timeframe string (e.g., "H4")
        n_bars: Number of bars to load
    
    Returns:
        List of OHLCBar objects, or None if failed
    """
    tf = TIMEFRAME_MAP.get(timeframe)
    if tf is None:
        print(f"Unknown timeframe: {timeframe}")
        return None
    
    # Request bars from MT5
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, n_bars)
    
    if rates is None or len(rates) == 0:
        print(f"Failed to load {symbol} data: {mt5.last_error()}")
        return None
    
    bars = []
    for rate in rates:
        timestamp = datetime.fromtimestamp(rate['time'], tz=timezone.utc)
        bar = OHLCBar(
            symbol=symbol,
            timestamp=timestamp,
            open=float(rate['open']),
            high=float(rate['high']),
            low=float(rate['low']),
            close=float(rate['close']),
        )
        bars.append(bar)
    
    return bars


def load_pair_observations(
    symbol_a: str,
    symbol_b: str,
    timeframe: str,
    n_bars: int,
) -> Optional[List[MarketObservation]]:
    """
    Load aligned observations for a currency pair.
    
    Args:
        symbol_a: First symbol (e.g., "EURUSD")
        symbol_b: Second symbol (e.g., "GBPUSD")
        timeframe: Timeframe string (e.g., "H4")
        n_bars: Number of bars to load
    
    Returns:
        List of MarketObservation objects with aligned timestamps
    """
    print(f"Loading {symbol_a} {timeframe}...")
    bars_a = load_ohlc(symbol_a, timeframe, n_bars + 100)  # Extra buffer for alignment
    if bars_a is None:
        return None
    
    print(f"Loading {symbol_b} {timeframe}...")
    bars_b = load_ohlc(symbol_b, timeframe, n_bars + 100)
    if bars_b is None:
        return None
    
    # Create timestamp index for symbol B
    bars_b_by_time = {bar.timestamp: bar for bar in bars_b}
    
    # Align bars by timestamp
    observations = []
    for bar_a in bars_a:
        bar_b = bars_b_by_time.get(bar_a.timestamp)
        if bar_b is None:
            continue  # Skip unaligned bars
        
        obs_id = create_observation_id(bar_a.timestamp, timeframe)
        obs = MarketObservation(
            observation_id=obs_id,
            timestamp=bar_a.timestamp,
            timeframe=timeframe,
            bar_a=bar_a,
            bar_b=bar_b,
        )
        observations.append(obs)
    
    # Sort by timestamp and take most recent n_bars
    observations.sort(key=lambda x: x.timestamp)
    
    if len(observations) < n_bars:
        print(f"Warning: Only {len(observations)} aligned bars available (requested {n_bars})")
    
    return observations[-n_bars:] if len(observations) >= n_bars else observations


def validate_data(
    observations: List[MarketObservation],
    expected_bars: int,
    timeframe: str = "H4",
) -> Tuple[bool, List[str]]:
    """
    Validate loaded data meets requirements.
    
    FX markets have regular gaps for:
    - Weekends (~48-52 hours)
    - Holidays (can extend weekends to 72-100 hours)
    - Mid-week holidays (24-28 hours)
    
    We only flag ANOMALOUS gaps (>5 days) that indicate data problems.
    
    Returns:
        (is_valid, list of issues)
    """
    issues = []
    
    # Check count
    if len(observations) < expected_bars:
        issues.append(f"Insufficient bars: {len(observations)} < {expected_bars}")
    
    # Only flag gaps > 5 days (truly anomalous)
    max_acceptable_gap = timedelta(days=5)
    
    if len(observations) > 1:
        anomalous_gaps = []
        for i in range(1, len(observations)):
            delta = observations[i].timestamp - observations[i-1].timestamp
            
            if delta > max_acceptable_gap:
                anomalous_gaps.append((i, delta, observations[i-1].timestamp))
        
        for i, delta, ts in anomalous_gaps[:5]:
            issues.append(f"Anomalous gap at index {i} ({ts.date()}): {delta}")
        
        if len(anomalous_gaps) > 5:
            issues.append(f"... and {len(anomalous_gaps) - 5} more anomalous gaps")
    
    # Check price validity
    for i, obs in enumerate(observations):
        if obs.bar_a.close <= 0 or obs.bar_b.close <= 0:
            issues.append(f"Invalid price at index {i}")
            break
    
    # Check timestamp order
    for i in range(1, len(observations)):
        if observations[i].timestamp <= observations[i-1].timestamp:
            issues.append(f"Timestamp not strictly increasing at index {i}")
            break
    
    # Check for reasonable price ranges (EURUSD and GBPUSD)
    for i, obs in enumerate(observations[:10]):  # Check first 10
        if not (0.5 < obs.bar_a.close < 2.0):
            issues.append(f"EURUSD price out of range at index {i}: {obs.bar_a.close}")
            break
        if not (0.8 < obs.bar_b.close < 2.5):
            issues.append(f"GBPUSD price out of range at index {i}: {obs.bar_b.close}")
            break
    
    return len(issues) == 0, issues


def print_data_summary(observations: List[MarketObservation], timeframe: str = "H4"):
    """Print summary statistics about loaded data."""
    if not observations:
        print("No observations to summarize")
        return
    
    bar_hours = TIMEFRAME_HOURS.get(timeframe, 4)
    expected_delta = timedelta(hours=bar_hours)
    
    # Count weekend gaps
    weekend_gaps = 0
    max_gap = timedelta(0)
    
    for i in range(1, len(observations)):
        gap = observations[i].timestamp - observations[i-1].timestamp
        if gap > expected_delta * 1.5:
            weekend_gaps += 1
            if gap > max_gap:
                max_gap = gap
    
    # Price ranges
    prices_a = [obs.bar_a.close for obs in observations]
    prices_b = [obs.bar_b.close for obs in observations]
    
    print(f"\nData Summary:")
    print(f"  Total bars: {len(observations)}")
    print(f"  Date range: {observations[0].timestamp.date()} to {observations[-1].timestamp.date()}")
    print(f"  Calendar days: {(observations[-1].timestamp - observations[0].timestamp).days}")
    print(f"  Weekend/holiday gaps: {weekend_gaps}")
    print(f"  Max gap: {max_gap}")
    print(f"  {observations[0].bar_a.symbol} range: {min(prices_a):.5f} - {max(prices_a):.5f}")
    print(f"  {observations[0].bar_b.symbol} range: {min(prices_b):.5f} - {max(prices_b):.5f}")


if __name__ == "__main__":
    # Test the loader
    if initialize_mt5():
        obs = load_pair_observations("EURUSD", "GBPUSD", "H4", 100)
        if obs:
            print(f"Loaded {len(obs)} observations")
            print(f"First: {obs[0].timestamp}")
            print(f"Last: {obs[-1].timestamp}")
            valid, issues = validate_data(obs, 100, "H4")
            if valid:
                print("Data validation passed")
            else:
                print(f"Data validation failed: {issues}")
            print_data_summary(obs, "H4")
        shutdown_mt5()
