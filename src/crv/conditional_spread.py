"""
FX Conditional Relative Value (CRV) System - Layer 3: Conditional Spread (HARDENED).

CRITICAL FIXES APPLIED:
1. Explicit NaN handling - NEVER return NaN z-scores
2. pct_change(fill_method=None) to fix FutureWarning
3. Minimum std threshold to prevent division by zero
4. Explicit NO_SIGNAL returns instead of NaN

Key Principle:
    The Z-score is NOT universal. A Z-score of 2.0 in a stable regime
    means something very different than 2.0 in a volatile regime.

    We construct spread statistics CONDITIONED on the current regime,
    not on the entire historical period.

PHILOSOPHY:
    If we cannot compute a valid z-score → return NO SIGNAL
    NaN propagation is a BUG, not a feature
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Tuple
from enum import Enum
import logging
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

from src.crv.regime_filter import FXRegime

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS - SAFETY THRESHOLDS
# ============================================================================

MIN_STD_THRESHOLD = 1e-8          # Minimum std to prevent division by zero
MIN_OBSERVATIONS = 30              # Minimum observations for valid statistics
MAX_ZSCORE = 10.0                  # Cap extreme z-scores
INVALID_ZSCORE = 0.0               # Return this instead of NaN
INVALID_HALF_LIFE = 9999.0         # Return this for non-mean-reverting series


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class RegimeStatistics:
    """Statistics computed within a specific regime."""
    regime: FXRegime
    
    # Sample info
    n_observations: int
    start_date: Optional[datetime]
    end_date: Optional[datetime]
    
    # Spread statistics
    spread_mean: float
    spread_std: float
    spread_median: float
    spread_q25: float
    spread_q75: float
    
    # Distribution metrics
    spread_skew: float
    spread_kurtosis: float
    
    # Mean reversion characteristics
    half_life: float
    mean_reversion_rate: float  # theta in OU process
    
    # Validity flag
    is_valid: bool = True


@dataclass
class ConditionalSpreadData:
    """Spread data with regime conditioning."""
    timestamp: datetime
    
    # Current spread
    raw_spread: float
    hedge_ratio: float
    
    # Unconditional z-score (for reference only)
    zscore_unconditional: float
    
    # CONDITIONAL z-score (THIS IS WHAT MATTERS)
    zscore_conditional: float
    current_regime: FXRegime
    
    # Regime-specific statistics
    regime_stats: Optional[RegimeStatistics]
    
    # Signal validity
    is_valid: bool              # NEW: Explicit validity flag
    is_extreme: bool
    signal_direction: Optional[str]  # "long", "short", None
    signal_confidence: float
    
    # Invalidity reason if applicable
    invalidity_reason: Optional[str] = None
    
    # Multi-regime context (optional, has default)
    all_regime_stats: Dict[str, RegimeStatistics] = field(default_factory=dict)


@dataclass  
class SpreadDecomposition:
    """Spread decomposed into components."""
    # Total spread
    spread: float
    
    # Components
    trend_component: float      # Long-term drift
    regime_component: float     # Regime-specific deviation
    residual_component: float   # True mean-reverting part
    
    # Interpretation
    is_trend_driven: bool
    is_regime_driven: bool
    is_mean_reverting: bool
    
    # Validity
    is_valid: bool = True


# ============================================================================
# SAFE CALCULATION FUNCTIONS
# ============================================================================

def safe_pct_change(series: pd.Series) -> pd.Series:
    """
    Calculate percentage change with EXPLICIT NaN handling.
    
    FIXES: FutureWarning about fill_method deprecation.
    """
    # Remove NaN BEFORE calculation
    clean = series.dropna()
    
    if len(clean) < 2:
        return pd.Series(dtype=float)
    
    # Use fill_method=None to avoid FutureWarning
    returns = clean.pct_change(fill_method=None)
    
    # Drop NaN from result
    return returns.dropna()


def safe_zscore(
    value: float,
    mean: float,
    std: float,
    max_zscore: float = MAX_ZSCORE
) -> Tuple[float, bool]:
    """
    Calculate z-score with safety checks.
    
    Returns:
        (zscore, is_valid)
        
    NEVER returns NaN - returns (0.0, False) if invalid.
    """
    # Check for NaN inputs
    if np.isnan(value) or np.isnan(mean) or np.isnan(std):
        return INVALID_ZSCORE, False
    
    # Check for degenerate std
    if std < MIN_STD_THRESHOLD:
        return INVALID_ZSCORE, False
    
    # Calculate z-score
    zscore = (value - mean) / std
    
    # Check for infinity
    if np.isinf(zscore):
        return INVALID_ZSCORE, False
    
    # Cap extreme values
    zscore = np.clip(zscore, -max_zscore, max_zscore)
    
    return float(zscore), True


def safe_statistics(series: pd.Series) -> Tuple[float, float, float, float, float, bool]:
    """
    Calculate statistics with safety checks.
    
    Returns:
        (mean, std, median, q25, q75, is_valid)
    """
    # Remove NaN
    clean = series.dropna()
    
    if len(clean) < MIN_OBSERVATIONS:
        return 0.0, 0.0, 0.0, 0.0, 0.0, False
    
    mean = float(clean.mean())
    std = float(clean.std())
    median = float(clean.median())
    q25 = float(clean.quantile(0.25))
    q75 = float(clean.quantile(0.75))
    
    # Check for NaN results
    if any(np.isnan(x) for x in [mean, std, median, q25, q75]):
        return 0.0, 0.0, 0.0, 0.0, 0.0, False
    
    return mean, std, median, q25, q75, True


# ============================================================================
# CONDITIONAL SPREAD ANALYZER (HARDENED)
# ============================================================================

class ConditionalSpreadAnalyzer:
    """
    Layer 3: Conditional Spread Analysis for FX CRV (HARDENED).
    
    CRITICAL SAFETY FEATURES:
    1. NEVER returns NaN z-scores
    2. Explicit validity flags on all outputs
    3. Safe division with std threshold
    4. pct_change(fill_method=None) fix applied
    
    Key innovations:
    1. Z-score is regime-specific, not universal
    2. Mean/std computed only from same-regime observations
    3. Spread decomposition separates trend vs mean-reversion
    4. Half-life is regime-dependent
    """
    
    def __init__(
        self,
        # Rolling windows
        hedge_ratio_window: int = 120,
        unconditional_window: int = 60,
        min_regime_observations: int = 50,
        
        # Z-score thresholds
        zscore_entry_threshold: float = 1.5,
        zscore_extreme_threshold: float = 2.5,
        
        # Safety thresholds
        min_std_threshold: float = MIN_STD_THRESHOLD,
        max_zscore: float = MAX_ZSCORE,
    ):
        self.hedge_ratio_window = hedge_ratio_window
        self.unconditional_window = unconditional_window
        self.min_regime_observations = min_regime_observations
        
        self.zscore_entry_threshold = zscore_entry_threshold
        self.zscore_extreme_threshold = zscore_extreme_threshold
        
        self.min_std_threshold = min_std_threshold
        self.max_zscore = max_zscore
    
    def analyze(
        self,
        price_a: pd.Series,
        price_b: pd.Series,
        regime_history: pd.Series,
        current_regime: FXRegime,
        timestamp: Optional[datetime] = None
    ) -> ConditionalSpreadData:
        """
        Analyze spread with regime conditioning (HARDENED).
        
        GUARANTEES:
        - Never returns NaN z-scores
        - Always returns valid is_valid flag
        - Explicit invalidity_reason if invalid
        
        Args:
            price_a: Close prices for leg A
            price_b: Close prices for leg B
            regime_history: Historical regime classification for each bar
            current_regime: Current regime classification
            timestamp: Analysis timestamp
            
        Returns:
            ConditionalSpreadData with regime-conditional z-score
        """
        timestamp = timestamp or datetime.now()
        
        # === STEP 0: Input validation ===
        if price_a is None or price_b is None:
            return self._invalid_result(timestamp, current_regime, "Null price input")
        
        if len(price_a) < self.unconditional_window or len(price_b) < self.unconditional_window:
            return self._invalid_result(timestamp, current_regime, "Insufficient price data")
        
        # Remove NaN from inputs
        price_a_clean = price_a.dropna()
        price_b_clean = price_b.dropna()
        
        # Align indices
        common_idx = price_a_clean.index.intersection(price_b_clean.index)
        
        if len(common_idx) < self.unconditional_window:
            return self._invalid_result(timestamp, current_regime, "Insufficient aligned data")
        
        price_a_aligned = price_a_clean.loc[common_idx]
        price_b_aligned = price_b_clean.loc[common_idx]
        
        # === STEP 1: Calculate dynamic hedge ratio ===
        hedge_ratio, hr_valid = self._calculate_dynamic_hedge_ratio_safe(
            price_a_aligned, price_b_aligned
        )
        
        if not hr_valid:
            return self._invalid_result(timestamp, current_regime, "Invalid hedge ratio")
        
        # === STEP 2: Construct spread ===
        spread_series = price_a_aligned - hedge_ratio * price_b_aligned
        spread_series = spread_series.dropna()
        
        if len(spread_series) < self.unconditional_window:
            return self._invalid_result(timestamp, current_regime, "Insufficient spread data")
        
        raw_spread = float(spread_series.iloc[-1])
        
        if np.isnan(raw_spread):
            return self._invalid_result(timestamp, current_regime, "Current spread is NaN")
        
        # === STEP 3: Calculate UNCONDITIONAL z-score (for reference) ===
        uncond_window = spread_series.tail(self.unconditional_window)
        uncond_mean, uncond_std, _, _, _, uncond_valid = safe_statistics(uncond_window)
        
        if uncond_valid:
            zscore_unconditional, _ = safe_zscore(raw_spread, uncond_mean, uncond_std)
        else:
            zscore_unconditional = INVALID_ZSCORE
        
        # === STEP 4: Calculate CONDITIONAL z-score (the key metric) ===
        regime_stats = self._calculate_regime_statistics_safe(
            spread_series, regime_history, current_regime
        )
        
        if regime_stats and regime_stats.is_valid and regime_stats.n_observations >= self.min_regime_observations:
            zscore_conditional, z_valid = safe_zscore(
                raw_spread, 
                regime_stats.spread_mean, 
                regime_stats.spread_std
            )
            
            if not z_valid:
                # Fall back to unconditional
                zscore_conditional = zscore_unconditional
                logger.warning("Conditional z-score invalid, using unconditional")
        else:
            # Fall back to unconditional if insufficient regime data
            zscore_conditional = zscore_unconditional
            logger.debug(f"Insufficient regime data for {current_regime.value}, using unconditional z-score")
        
        # === STEP 5: Calculate all regime statistics for context ===
        all_regime_stats = self._calculate_all_regime_statistics_safe(spread_series, regime_history)
        
        # === STEP 6: Determine signal ===
        is_extreme = abs(zscore_conditional) >= self.zscore_extreme_threshold
        
        if zscore_conditional <= -self.zscore_entry_threshold:
            signal_direction = "long"
            signal_confidence = min(1.0, abs(zscore_conditional) / 3.0)
        elif zscore_conditional >= self.zscore_entry_threshold:
            signal_direction = "short"
            signal_confidence = min(1.0, abs(zscore_conditional) / 3.0)
        else:
            signal_direction = None
            signal_confidence = 0.0
        
        return ConditionalSpreadData(
            timestamp=timestamp,
            raw_spread=raw_spread,
            hedge_ratio=hedge_ratio,
            zscore_unconditional=zscore_unconditional,
            zscore_conditional=zscore_conditional,
            current_regime=current_regime,
            regime_stats=regime_stats,
            all_regime_stats=all_regime_stats,
            is_valid=True,
            is_extreme=is_extreme,
            signal_direction=signal_direction,
            signal_confidence=signal_confidence,
            invalidity_reason=None
        )
    
    def _invalid_result(
        self,
        timestamp: datetime,
        current_regime: FXRegime,
        reason: str
    ) -> ConditionalSpreadData:
        """Return an explicitly invalid result - NEVER NaN."""
        logger.warning(f"Invalid spread analysis: {reason}")
        
        return ConditionalSpreadData(
            timestamp=timestamp,
            raw_spread=0.0,
            hedge_ratio=1.0,
            zscore_unconditional=INVALID_ZSCORE,
            zscore_conditional=INVALID_ZSCORE,
            current_regime=current_regime,
            regime_stats=None,
            all_regime_stats={},
            is_valid=False,
            is_extreme=False,
            signal_direction=None,
            signal_confidence=0.0,
            invalidity_reason=reason
        )
    
    def _calculate_dynamic_hedge_ratio_safe(
        self,
        price_a: pd.Series,
        price_b: pd.Series
    ) -> Tuple[float, bool]:
        """
        Calculate dynamic hedge ratio with safety checks.
        
        Returns:
            (hedge_ratio, is_valid)
        """
        try:
            if len(price_a) < 30 or len(price_b) < 30:
                return 1.0, False
            
            window = min(self.hedge_ratio_window, len(price_a))
            window_a = price_a.tail(window).values
            window_b = price_b.tail(window).values
            
            # Check for NaN
            if np.any(np.isnan(window_a)) or np.any(np.isnan(window_b)):
                return 1.0, False
            
            X = sm.add_constant(window_b)
            model = sm.OLS(window_a, X).fit()
            
            hedge_ratio = float(model.params[1])
            
            # Sanity checks
            if np.isnan(hedge_ratio) or np.isinf(hedge_ratio):
                return 1.0, False
            
            if abs(hedge_ratio) < 0.01 or abs(hedge_ratio) > 100:
                return 1.0, False
            
            return hedge_ratio, True
            
        except Exception as e:
            logger.error(f"Hedge ratio calculation failed: {e}")
            return 1.0, False
    
    def _calculate_regime_statistics_safe(
        self,
        spread_series: pd.Series,
        regime_history: pd.Series,
        target_regime: FXRegime
    ) -> Optional[RegimeStatistics]:
        """
        Calculate spread statistics for a specific regime (SAFE).
        
        NEVER returns statistics with NaN values.
        """
        try:
            # Align indices
            common_idx = spread_series.index.intersection(regime_history.index)
            
            if len(common_idx) < self.min_regime_observations:
                return None
            
            spread_aligned = spread_series.loc[common_idx]
            regime_aligned = regime_history.loc[common_idx]
            
            # Filter for target regime
            target_str = target_regime.value
            mask = regime_aligned.astype(str) == target_str
            
            regime_spread = spread_aligned[mask].dropna()
            
            if len(regime_spread) < self.min_regime_observations:
                return None
            
            # Calculate statistics safely
            mean, std, median, q25, q75, stats_valid = safe_statistics(regime_spread)
            
            if not stats_valid:
                return None
            
            # Distribution metrics
            try:
                spread_skew = float(regime_spread.skew())
                spread_kurtosis = float(regime_spread.kurtosis())
                
                if np.isnan(spread_skew):
                    spread_skew = 0.0
                if np.isnan(spread_kurtosis):
                    spread_kurtosis = 0.0
            except:
                spread_skew = 0.0
                spread_kurtosis = 0.0
            
            # Half-life calculation
            half_life = self._calculate_half_life_safe(regime_spread)
            mean_reversion_rate = np.log(2) / half_life if 0 < half_life < INVALID_HALF_LIFE else 0.0
            
            return RegimeStatistics(
                regime=target_regime,
                n_observations=len(regime_spread),
                start_date=regime_spread.index[0] if len(regime_spread) > 0 else None,
                end_date=regime_spread.index[-1] if len(regime_spread) > 0 else None,
                spread_mean=mean,
                spread_std=std,
                spread_median=median,
                spread_q25=q25,
                spread_q75=q75,
                spread_skew=spread_skew,
                spread_kurtosis=spread_kurtosis,
                half_life=half_life,
                mean_reversion_rate=mean_reversion_rate,
                is_valid=True
            )
            
        except Exception as e:
            logger.error(f"Regime statistics calculation failed: {e}")
            return None
    
    def _calculate_all_regime_statistics_safe(
        self,
        spread_series: pd.Series,
        regime_history: pd.Series
    ) -> Dict[str, RegimeStatistics]:
        """Calculate statistics for all regimes safely."""
        all_stats = {}
        
        for regime in FXRegime:
            try:
                stats = self._calculate_regime_statistics_safe(
                    spread_series, regime_history, regime
                )
                if stats and stats.is_valid:
                    all_stats[regime.value] = stats
            except:
                continue
        
        return all_stats
    
    def _calculate_half_life_safe(self, spread: pd.Series) -> float:
        """Calculate half-life with safety checks."""
        try:
            if len(spread) < 50:
                return INVALID_HALF_LIFE
            
            spread_clean = spread.dropna()
            
            if len(spread_clean) < 50:
                return INVALID_HALF_LIFE
            
            spread_lag = spread_clean.shift(1).dropna()
            spread_diff = spread_clean.diff().dropna()
            
            common = spread_lag.index.intersection(spread_diff.index)
            
            if len(common) < 30:
                return INVALID_HALF_LIFE
            
            y = spread_diff.loc[common].values
            X = spread_lag.loc[common].values.reshape(-1, 1)
            
            # Check for NaN
            if np.any(np.isnan(y)) or np.any(np.isnan(X)):
                return INVALID_HALF_LIFE
            
            model = LinearRegression()
            model.fit(X, y)
            theta = model.coef_[0]
            
            if theta >= 0:
                return INVALID_HALF_LIFE
            
            half_life = -np.log(2) / theta
            
            if np.isnan(half_life) or np.isinf(half_life):
                return INVALID_HALF_LIFE
            
            return min(float(half_life), INVALID_HALF_LIFE)
            
        except Exception as e:
            logger.debug(f"Half-life calculation failed: {e}")
            return INVALID_HALF_LIFE
    
    def decompose_spread(
        self,
        spread_series: pd.Series,
        regime_history: pd.Series
    ) -> SpreadDecomposition:
        """
        Decompose spread into trend, regime, and residual components (SAFE).
        """
        try:
            spread_clean = spread_series.dropna()
            
            if len(spread_clean) < self.unconditional_window:
                return SpreadDecomposition(
                    spread=0.0,
                    trend_component=0.0,
                    regime_component=0.0,
                    residual_component=0.0,
                    is_trend_driven=False,
                    is_regime_driven=False,
                    is_mean_reverting=False,
                    is_valid=False
                )
            
            spread = float(spread_clean.iloc[-1])
            
            # Trend component
            long_term_window = min(250, len(spread_clean))
            long_term_mean = float(spread_clean.tail(long_term_window).mean())
            short_term_mean = float(spread_clean.tail(20).mean())
            trend_component = short_term_mean - long_term_mean
            
            # Regime component
            regime_mean = float(spread_clean.tail(self.unconditional_window).mean())
            regime_component = regime_mean - long_term_mean
            
            # Residual
            residual_component = spread - regime_mean
            
            # Check for NaN
            if any(np.isnan(x) for x in [spread, trend_component, regime_component, residual_component]):
                return SpreadDecomposition(
                    spread=0.0,
                    trend_component=0.0,
                    regime_component=0.0,
                    residual_component=0.0,
                    is_trend_driven=False,
                    is_regime_driven=False,
                    is_mean_reverting=False,
                    is_valid=False
                )
            
            # Interpretation
            spread_std = float(spread_clean.std())
            
            is_trend_driven = abs(trend_component) > spread_std * 0.5 if spread_std > 0 else False
            is_regime_driven = abs(regime_component) > abs(residual_component)
            is_mean_reverting = abs(residual_component) > abs(trend_component)
            
            return SpreadDecomposition(
                spread=spread,
                trend_component=trend_component,
                regime_component=regime_component,
                residual_component=residual_component,
                is_trend_driven=is_trend_driven,
                is_regime_driven=is_regime_driven,
                is_mean_reverting=is_mean_reverting,
                is_valid=True
            )
            
        except Exception as e:
            logger.error(f"Spread decomposition failed: {e}")
            return SpreadDecomposition(
                spread=0.0,
                trend_component=0.0,
                regime_component=0.0,
                residual_component=0.0,
                is_trend_driven=False,
                is_regime_driven=False,
                is_mean_reverting=False,
                is_valid=False
            )
    
    def get_entry_threshold_for_regime(self, regime: FXRegime) -> float:
        """Get regime-specific entry threshold."""
        thresholds = {
            FXRegime.STABLE_LOW_VOL: 1.5,
            FXRegime.STABLE_NORMAL_VOL: 1.75,
            FXRegime.RANGE_BOUND: 2.0,
            FXRegime.TRENDING_STRONG: 3.0,
            FXRegime.HIGH_VOLATILITY: 3.0,
            FXRegime.RISK_OFF_EXTREME: 3.0,
            FXRegime.RISK_ON_EXTREME: 3.0,
            FXRegime.MACRO_EVENT: 99.0,
            FXRegime.UNKNOWN: 2.5,
        }
        return thresholds.get(regime, 2.0)
    
    def get_exit_threshold_for_regime(self, regime: FXRegime) -> float:
        """Get regime-specific exit threshold."""
        thresholds = {
            FXRegime.STABLE_LOW_VOL: 0.3,
            FXRegime.STABLE_NORMAL_VOL: 0.5,
            FXRegime.RANGE_BOUND: 0.5,
            FXRegime.TRENDING_STRONG: 0.0,
            FXRegime.HIGH_VOLATILITY: 0.0,
            FXRegime.RISK_OFF_EXTREME: 0.0,
            FXRegime.RISK_ON_EXTREME: 0.0,
            FXRegime.MACRO_EVENT: 0.0,
            FXRegime.UNKNOWN: 0.25,
        }
        return thresholds.get(regime, 0.5)
