"""
FX Conditional Relative Value (CRV) System - Layer 0: Data Integrity.

MANDATORY data validation and cleaning layer.

This module ensures:
1. NO NaN propagation
2. Time alignment across symbols
3. Explicit rejection of bad data
4. Session gap handling

Philosophy:
    Bad data in = Bad signals out
    This layer prevents garbage from entering the system.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple, Set
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# DATA QUALITY METRICS
# ============================================================================

@dataclass
class DataQualityReport:
    """Report on data quality for a symbol."""
    symbol: str
    
    # Basic stats
    total_bars: int
    start_date: datetime
    end_date: datetime
    
    # Quality metrics
    nan_count: int
    nan_percentage: float
    gap_count: int
    max_gap_hours: float
    
    # Validation
    is_valid: bool
    rejection_reasons: List[str] = field(default_factory=list)


@dataclass 
class AlignedDataset:
    """Container for aligned, validated FX data."""
    symbols: List[str]
    
    # Aligned data
    prices: Dict[str, pd.Series]        # Close prices
    ohlc: Optional[Dict[str, pd.DataFrame]]  # Full OHLC if available
    
    # Common index
    common_index: pd.DatetimeIndex
    n_bars: int
    
    # Quality
    quality_reports: Dict[str, DataQualityReport]
    rejected_symbols: List[str]
    
    # Metadata
    timeframe: str
    alignment_timestamp: datetime


# ============================================================================
# DATA VALIDATOR
# ============================================================================

class FXDataValidator:
    """
    Layer 0: Data validation and integrity enforcement.
    
    MANDATORY before any analysis.
    
    Rules:
    1. No forward fills for NaN - drop or reject
    2. Session gaps must be explicit
    3. Time alignment is enforced
    4. Minimum data requirements checked
    """
    
    def __init__(
        self,
        # Quality thresholds
        max_nan_percentage: float = 2.0,
        max_gap_hours: float = 80.0,  # FX has weekend gaps of ~76h (Fri to Mon)
        min_bars_required: int = 500,
        
        # Time alignment
        require_time_alignment: bool = True,
        
        # Gap handling
        max_acceptable_gap_ratio: float = 0.05,  # 5% gaps max
    ):
        self.max_nan_percentage = max_nan_percentage
        self.max_gap_hours = max_gap_hours
        self.min_bars_required = min_bars_required
        self.require_time_alignment = require_time_alignment
        self.max_acceptable_gap_ratio = max_acceptable_gap_ratio
    
    def validate_symbol(
        self,
        symbol: str,
        data: pd.DataFrame,
        timeframe: str = "H4"
    ) -> DataQualityReport:
        """
        Validate data quality for a single symbol.
        
        Returns DataQualityReport with pass/fail status.
        """
        rejection_reasons = []
        
        # Ensure we have required columns
        required_cols = ['close']
        for col in required_cols:
            if col not in data.columns:
                rejection_reasons.append(f"Missing column: {col}")
                return DataQualityReport(
                    symbol=symbol,
                    total_bars=0,
                    start_date=datetime.now(),
                    end_date=datetime.now(),
                    nan_count=0,
                    nan_percentage=100.0,
                    gap_count=0,
                    max_gap_hours=0,
                    is_valid=False,
                    rejection_reasons=rejection_reasons
                )
        
        # Basic stats
        total_bars = len(data)
        
        if total_bars == 0:
            rejection_reasons.append("Empty dataset")
            return DataQualityReport(
                symbol=symbol,
                total_bars=0,
                start_date=datetime.now(),
                end_date=datetime.now(),
                nan_count=0,
                nan_percentage=100.0,
                gap_count=0,
                max_gap_hours=0,
                is_valid=False,
                rejection_reasons=rejection_reasons
            )
        
        start_date = data.index[0] if hasattr(data.index[0], 'to_pydatetime') else data.index[0]
        end_date = data.index[-1] if hasattr(data.index[-1], 'to_pydatetime') else data.index[-1]
        
        # NaN analysis
        nan_count = data['close'].isna().sum()
        nan_percentage = (nan_count / total_bars) * 100
        
        if nan_percentage > self.max_nan_percentage:
            rejection_reasons.append(f"NaN percentage too high: {nan_percentage:.1f}%")
        
        # Gap analysis
        if isinstance(data.index, pd.DatetimeIndex):
            time_diffs = data.index.to_series().diff()
            
            # Expected gap based on timeframe
            expected_gap = self._get_expected_gap(timeframe)
            
            # FX MARKET GAP HANDLING:
            # - Normal weekend gap: ~72-76 hours (Fri 5pm to Sun 5pm EST)
            # - Holiday weekend: ~100-120 hours (Fri to Tue if Mon is holiday)
            # - Truly abnormal: > 120 hours (indicates data issue)
            FX_WEEKEND_GAP = timedelta(hours=80)
            FX_HOLIDAY_GAP = timedelta(hours=120)
            
            # Calculate max gap
            if len(time_diffs.dropna()) > 0:
                max_gap = time_diffs.max()
                max_gap_hours = max_gap.total_seconds() / 3600 if pd.notna(max_gap) else 0
            else:
                max_gap_hours = 0
            
            # Count abnormal gaps (excluding normal weekend/holiday gaps)
            # A gap is abnormal if: > 2x expected timeframe BUT < weekend gap
            # (gaps >= weekend gap are assumed to be weekends)
            abnormal_gaps = (time_diffs > (expected_gap * 2)) & (time_diffs < FX_WEEKEND_GAP)
            gap_count = abnormal_gaps.sum()
            
            gap_ratio = gap_count / total_bars if total_bars > 0 else 0
            if gap_ratio > self.max_acceptable_gap_ratio:
                rejection_reasons.append(f"Too many mid-week gaps: {gap_count} ({gap_ratio:.1%})")
            
            # Only reject if max gap exceeds holiday threshold (truly abnormal)
            if max_gap_hours > FX_HOLIDAY_GAP.total_seconds() / 3600:
                rejection_reasons.append(f"Gap too large: {max_gap_hours:.0f}h (exceeds {FX_HOLIDAY_GAP.total_seconds()/3600:.0f}h limit)")
        else:
            gap_count = 0
            max_gap_hours = 0
        
        # Minimum bars check
        if total_bars < self.min_bars_required:
            rejection_reasons.append(f"Insufficient bars: {total_bars} < {self.min_bars_required}")
        
        is_valid = len(rejection_reasons) == 0
        
        return DataQualityReport(
            symbol=symbol,
            total_bars=total_bars,
            start_date=start_date,
            end_date=end_date,
            nan_count=nan_count,
            nan_percentage=nan_percentage,
            gap_count=gap_count,
            max_gap_hours=max_gap_hours,
            is_valid=is_valid,
            rejection_reasons=rejection_reasons
        )
    
    def align_and_validate(
        self,
        price_data: Dict[str, pd.Series],
        ohlc_data: Optional[Dict[str, pd.DataFrame]] = None,
        timeframe: str = "H4"
    ) -> AlignedDataset:
        """
        Align all symbols to common timeline and validate.
        
        Returns AlignedDataset with only valid, aligned data.
        """
        quality_reports = {}
        rejected_symbols = []
        valid_symbols = []
        
        logger.info(f"Validating {len(price_data)} symbols...")
        
        # Step 1: Validate each symbol
        for symbol, prices in price_data.items():
            # Create DataFrame for validation
            df = pd.DataFrame({'close': prices})
            df.index = prices.index
            
            report = self.validate_symbol(symbol, df, timeframe)
            quality_reports[symbol] = report
            
            if report.is_valid:
                valid_symbols.append(symbol)
            else:
                rejected_symbols.append(symbol)
                logger.warning(f"Rejected {symbol}: {report.rejection_reasons}")
        
        if len(valid_symbols) < 2:
            logger.error("Insufficient valid symbols for CRV analysis")
            return AlignedDataset(
                symbols=[],
                prices={},
                ohlc=None,
                common_index=pd.DatetimeIndex([]),
                n_bars=0,
                quality_reports=quality_reports,
                rejected_symbols=rejected_symbols,
                timeframe=timeframe,
                alignment_timestamp=datetime.now()
            )
        
        # Step 2: Find common index
        indices = [price_data[s].dropna().index for s in valid_symbols]
        common_index = indices[0]
        
        for idx in indices[1:]:
            common_index = common_index.intersection(idx)
        
        logger.info(f"Common index: {len(common_index)} bars")
        
        if len(common_index) < self.min_bars_required:
            logger.error(f"Common index too small: {len(common_index)}")
            return AlignedDataset(
                symbols=valid_symbols,
                prices={},
                ohlc=None,
                common_index=common_index,
                n_bars=len(common_index),
                quality_reports=quality_reports,
                rejected_symbols=rejected_symbols,
                timeframe=timeframe,
                alignment_timestamp=datetime.now()
            )
        
        # Step 3: Align all data to common index
        aligned_prices = {}
        for symbol in valid_symbols:
            aligned = price_data[symbol].loc[common_index].copy()
            
            # CRITICAL: No forward fill - drop NaN
            if aligned.isna().any():
                aligned = aligned.dropna()
                # Re-check if still valid
                if len(aligned) < self.min_bars_required * 0.9:
                    rejected_symbols.append(symbol)
                    logger.warning(f"Symbol {symbol} invalid after NaN removal")
                    continue
            
            aligned_prices[symbol] = aligned
        
        # Step 4: Align OHLC if provided
        aligned_ohlc = None
        if ohlc_data:
            aligned_ohlc = {}
            for symbol in aligned_prices.keys():
                if symbol in ohlc_data:
                    ohlc = ohlc_data[symbol]
                    common_ohlc_idx = ohlc.index.intersection(common_index)
                    aligned_ohlc[symbol] = ohlc.loc[common_ohlc_idx].copy()
        
        # Update valid symbols
        final_symbols = list(aligned_prices.keys())
        
        return AlignedDataset(
            symbols=final_symbols,
            prices=aligned_prices,
            ohlc=aligned_ohlc,
            common_index=common_index,
            n_bars=len(common_index),
            quality_reports=quality_reports,
            rejected_symbols=rejected_symbols,
            timeframe=timeframe,
            alignment_timestamp=datetime.now()
        )
    
    def _get_expected_gap(self, timeframe: str) -> timedelta:
        """Get expected time gap for timeframe."""
        gaps = {
            "M1": timedelta(minutes=1),
            "M5": timedelta(minutes=5),
            "M15": timedelta(minutes=15),
            "M30": timedelta(minutes=30),
            "H1": timedelta(hours=1),
            "H4": timedelta(hours=4),
            "D1": timedelta(days=1),
            "W1": timedelta(weeks=1),
        }
        return gaps.get(timeframe, timedelta(hours=4))


# ============================================================================
# SAFE RETURN CALCULATION
# ============================================================================

def safe_returns(
    prices: pd.Series,
    method: str = "simple"
) -> pd.Series:
    """
    Calculate returns with EXPLICIT NaN handling.
    
    FIXES FutureWarning: pct_change with fill_method is deprecated.
    
    Args:
        prices: Price series
        method: "simple" or "log"
        
    Returns:
        Clean return series with NaN explicitly dropped
    """
    # Remove any existing NaN BEFORE calculation
    clean_prices = prices.dropna()
    
    if len(clean_prices) < 2:
        return pd.Series(dtype=float)
    
    if method == "log":
        returns = np.log(clean_prices / clean_prices.shift(1))
    else:
        # CRITICAL: Use fill_method=None to avoid FutureWarning
        returns = clean_prices.pct_change(fill_method=None)
    
    # Drop NaN from result (first value will be NaN)
    return returns.dropna()


def safe_rolling_correlation(
    returns_a: pd.Series,
    returns_b: pd.Series,
    window: int = 40
) -> pd.Series:
    """
    Calculate rolling correlation with safe NaN handling.
    
    Returns Series with NaN for insufficient data periods.
    """
    # Align indices
    common_idx = returns_a.index.intersection(returns_b.index)
    
    if len(common_idx) < window:
        return pd.Series(dtype=float)
    
    a = returns_a.loc[common_idx]
    b = returns_b.loc[common_idx]
    
    # Rolling correlation
    corr = a.rolling(window, min_periods=window).corr(b)
    
    return corr


def safe_zscore(
    spread: pd.Series,
    window: int = 60,
    min_std: float = 1e-8
) -> Tuple[Optional[float], Optional[float], Optional[float], bool]:
    """
    Calculate Z-score with safety checks.
    
    Returns:
        (zscore, mean, std, is_valid)
        
    Returns (None, None, None, False) if calculation is invalid.
    """
    if len(spread) < window:
        return None, None, None, False
    
    # Use only clean data
    clean_spread = spread.dropna()
    
    if len(clean_spread) < window:
        return None, None, None, False
    
    # Calculate on recent window
    recent = clean_spread.tail(window)
    
    mean = float(recent.mean())
    std = float(recent.std())
    
    # Check for degenerate case
    if std < min_std or np.isnan(std) or np.isnan(mean):
        return None, None, None, False
    
    # Current value
    current = float(clean_spread.iloc[-1])
    
    if np.isnan(current):
        return None, None, None, False
    
    zscore = (current - mean) / std
    
    # Sanity check
    if np.isnan(zscore) or np.isinf(zscore):
        return None, None, None, False
    
    return zscore, mean, std, True


# ============================================================================
# DATA INTEGRITY CHECKS
# ============================================================================

def check_price_sanity(
    prices: pd.Series,
    symbol: str,
    max_daily_move: float = 0.10  # 10% max daily move
) -> Tuple[bool, List[str]]:
    """
    Check for obviously bad price data.
    
    Detects:
    - Zero prices
    - Negative prices
    - Extreme moves (likely bad data)
    """
    issues = []
    
    # Zero or negative prices
    if (prices <= 0).any():
        issues.append(f"Zero or negative prices found")
    
    # Extreme moves
    returns = safe_returns(prices)
    if len(returns) > 0:
        extreme_moves = (returns.abs() > max_daily_move).sum()
        if extreme_moves > 0:
            issues.append(f"{extreme_moves} extreme moves (>{max_daily_move:.0%})")
    
    # Constant price (dead data)
    if prices.std() == 0:
        issues.append("Constant price (dead data)")
    
    return len(issues) == 0, issues


def verify_spread_integrity(
    price_a: pd.Series,
    price_b: pd.Series,
    hedge_ratio: float
) -> Tuple[bool, List[str]]:
    """
    Verify spread construction is valid.
    """
    issues = []
    
    # Check alignment
    if len(price_a) != len(price_b):
        issues.append(f"Length mismatch: {len(price_a)} vs {len(price_b)}")
    
    if not price_a.index.equals(price_b.index):
        issues.append("Index mismatch")
    
    # Check hedge ratio
    if np.isnan(hedge_ratio) or np.isinf(hedge_ratio):
        issues.append(f"Invalid hedge ratio: {hedge_ratio}")
    
    if abs(hedge_ratio) < 0.01 or abs(hedge_ratio) > 100:
        issues.append(f"Extreme hedge ratio: {hedge_ratio}")
    
    # Check spread values
    spread = price_a - hedge_ratio * price_b
    
    if spread.isna().any():
        nan_pct = spread.isna().sum() / len(spread) * 100
        issues.append(f"Spread has {nan_pct:.1f}% NaN")
    
    if (spread == 0).all():
        issues.append("Spread is constant zero")
    
    return len(issues) == 0, issues
