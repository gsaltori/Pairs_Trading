"""
Conditional Statistical Arbitrage System for FX.

This module implements a regime-aware, conditional pair trading system.
The system only trades when statistical conditions are met AND market
regime is favorable. Inactivity is a VALID state.

Key Principles:
1. Cointegration is NOT permanent - validate dynamically
2. Market regime determines tradability
3. Pairs can be valid but DORMANT
4. Zero trades > Invalid trades
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Optional, List, Dict, Tuple, Literal
from enum import Enum
import logging

from statsmodels.tsa.stattools import coint, adfuller
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS & CONSTANTS
# ============================================================================

class PairState(Enum):
    """Pair trading states."""
    ACTIVE = "active"           # Valid and tradeable NOW
    DORMANT = "dormant"         # Valid but regime unfavorable
    INVALIDATED = "invalidated" # Failed statistical tests
    WARMING_UP = "warming_up"   # Insufficient data


class MarketRegime(Enum):
    """Market regime classification."""
    TRENDING_STRONG = "trending_strong"     # No StatArb
    TRENDING_WEAK = "trending_weak"         # Caution
    RANGING = "ranging"                     # Ideal for StatArb
    VOLATILE = "volatile"                   # No StatArb
    QUIET = "quiet"                         # Good for StatArb
    UNKNOWN = "unknown"


class TradingSession(Enum):
    """FX Trading sessions."""
    ASIA = "asia"           # 00:00-08:00 UTC
    LONDON = "london"       # 08:00-16:00 UTC
    NEW_YORK = "new_york"   # 13:00-21:00 UTC
    OVERLAP = "overlap"     # London-NY overlap
    OFF_HOURS = "off_hours"


# Regime trading rules
REGIME_TRADEABLE = {
    MarketRegime.RANGING: True,
    MarketRegime.QUIET: True,
    MarketRegime.TRENDING_WEAK: False,  # Caution - disabled by default
    MarketRegime.TRENDING_STRONG: False,
    MarketRegime.VOLATILE: False,
    MarketRegime.UNKNOWN: False,
}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class VolatilityMetrics:
    """Volatility analysis results."""
    atr_current: float
    atr_percentile: float  # Current ATR vs historical (0-100)
    parkinson_vol: float
    garman_klass_vol: float
    
    is_high_volatility: bool
    is_expanding: bool
    vol_regime: str  # "low", "normal", "high", "extreme"


@dataclass
class TrendMetrics:
    """Trend analysis results."""
    adx: float
    ema_slope: float  # Slope of EMA200
    price_vs_ema: float  # % distance from EMA200
    is_trending: bool
    trend_direction: str  # "up", "down", "neutral"
    trend_strength: str   # "none", "weak", "strong"
    adx_threshold: float = 25.0


@dataclass
class SpreadHealth:
    """Spread statistical health metrics."""
    # Stationarity
    adf_pvalue: float
    is_stationary: bool
    
    # Mean reversion
    half_life: float
    half_life_stable: bool
    hurst_exponent: float
    is_mean_reverting: bool
    
    # Hedge ratio
    hedge_ratio: float
    hedge_ratio_zscore: float  # How far from rolling mean
    hedge_ratio_stable: bool
    
    # Spread volatility
    spread_vol: float
    spread_vol_percentile: float
    spread_vol_stable: bool
    
    # Overall health
    is_healthy: bool
    health_score: float  # 0-100


@dataclass
class CointegrationStatus:
    """Dynamic cointegration status."""
    # Current window tests
    eg_pvalue_current: float
    is_cointegrated_current: bool
    
    # Multi-window validation
    windows_tested: List[int]
    windows_passed: int
    cointegration_consistency: float  # % of windows that pass
    
    # Temporal stability
    days_since_last_breakdown: int
    breakdown_frequency: float  # Breakdowns per 100 bars
    
    # Overall status
    is_stable: bool
    confidence: str  # "high", "medium", "low", "none"


@dataclass
class RegimeAnalysis:
    """Complete regime analysis for a pair."""
    timestamp: datetime
    
    # Components
    volatility: VolatilityMetrics
    trend: TrendMetrics
    session: TradingSession
    
    # Spread health
    spread_health: SpreadHealth
    cointegration: CointegrationStatus
    
    # Overall regime
    market_regime: MarketRegime
    is_tradeable: bool
    
    # Reasons
    blocking_reasons: List[str] = field(default_factory=list)


@dataclass
class PairStatus:
    """Complete status for a trading pair."""
    pair: Tuple[str, str]
    state: PairState
    
    # Statistical validity
    is_statistically_valid: bool
    
    # Regime status
    regime_analysis: Optional[RegimeAnalysis]
    is_regime_favorable: bool
    
    # Trading status
    is_tradeable_now: bool
    
    # Signal (only if tradeable)
    current_zscore: float
    signal: Optional[str]  # "long", "short", None
    signal_strength: float  # 0-1
    
    # Reasons for state
    state_reasons: List[str] = field(default_factory=list)
    
    # Timestamps
    last_active: Optional[datetime] = None
    dormant_since: Optional[datetime] = None
    invalidated_since: Optional[datetime] = None


# ============================================================================
# REGIME DETECTOR
# ============================================================================

class MarketRegimeDetector:
    """
    Detects market regime to determine if StatArb is viable.
    
    StatArb works best in:
    - Low volatility / ranging markets
    - Mean-reverting conditions
    - Stable correlations
    
    StatArb fails in:
    - Strong trends
    - High volatility / breakouts
    - Regime changes
    """
    
    def __init__(
        self,
        atr_period: int = 14,
        adx_period: int = 14,
        ema_period: int = 200,
        vol_lookback: int = 100,
        trend_threshold: float = 25.0,
        vol_high_percentile: float = 75.0,
        vol_extreme_percentile: float = 90.0
    ):
        self.atr_period = atr_period
        self.adx_period = adx_period
        self.ema_period = ema_period
        self.vol_lookback = vol_lookback
        self.trend_threshold = trend_threshold
        self.vol_high_percentile = vol_high_percentile
        self.vol_extreme_percentile = vol_extreme_percentile
    
    def detect_regime(
        self,
        df: pd.DataFrame,  # OHLC data
        spread: Optional[pd.Series] = None
    ) -> Tuple[MarketRegime, Dict]:
        """
        Detect current market regime.
        
        Args:
            df: OHLC DataFrame with columns: open, high, low, close
            spread: Optional spread series for pair analysis
            
        Returns:
            (MarketRegime, details_dict)
        """
        if len(df) < self.ema_period + 50:
            return MarketRegime.UNKNOWN, {"reason": "Insufficient data"}
        
        # Calculate indicators
        vol_metrics = self._analyze_volatility(df)
        trend_metrics = self._analyze_trend(df)
        
        # Determine regime
        details = {
            "volatility": vol_metrics,
            "trend": trend_metrics
        }
        
        # Decision tree for regime
        if vol_metrics.vol_regime == "extreme":
            regime = MarketRegime.VOLATILE
        elif vol_metrics.vol_regime == "high" and vol_metrics.is_expanding:
            regime = MarketRegime.VOLATILE
        elif trend_metrics.trend_strength == "strong":
            regime = MarketRegime.TRENDING_STRONG
        elif trend_metrics.trend_strength == "weak":
            regime = MarketRegime.TRENDING_WEAK
        elif vol_metrics.vol_regime == "low":
            regime = MarketRegime.QUIET
        else:
            regime = MarketRegime.RANGING
        
        details["regime"] = regime
        details["is_tradeable"] = REGIME_TRADEABLE.get(regime, False)
        
        return regime, details
    
    def _analyze_volatility(self, df: pd.DataFrame) -> VolatilityMetrics:
        """Analyze volatility using multiple methods."""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        open_ = df['open'].values
        
        # ATR
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        atr = pd.Series(tr).rolling(self.atr_period).mean().iloc[-1]
        
        # ATR percentile
        atr_history = pd.Series(tr).rolling(self.atr_period).mean().dropna()
        atr_percentile = (atr_history < atr).sum() / len(atr_history) * 100
        
        # Parkinson volatility
        log_hl = np.log(high / low)
        parkinson = np.sqrt(1 / (4 * np.log(2)) * (log_hl ** 2).mean())
        
        # Garman-Klass volatility
        log_hl_sq = (np.log(high / low)) ** 2
        log_co_sq = (np.log(close / open_)) ** 2
        gk_vol = np.sqrt(0.5 * log_hl_sq - (2 * np.log(2) - 1) * log_co_sq).mean()
        
        # Determine regime
        is_expanding = atr_percentile > 60 and atr_history.iloc[-1] > atr_history.iloc[-5]
        
        if atr_percentile >= self.vol_extreme_percentile:
            vol_regime = "extreme"
        elif atr_percentile >= self.vol_high_percentile:
            vol_regime = "high"
        elif atr_percentile <= 25:
            vol_regime = "low"
        else:
            vol_regime = "normal"
        
        return VolatilityMetrics(
            atr_current=float(atr),
            atr_percentile=float(atr_percentile),
            parkinson_vol=float(parkinson),
            garman_klass_vol=float(gk_vol),
            is_high_volatility=atr_percentile >= self.vol_high_percentile,
            is_expanding=is_expanding,
            vol_regime=vol_regime
        )
    
    def _analyze_trend(self, df: pd.DataFrame) -> TrendMetrics:
        """Analyze trend using ADX and EMA."""
        close = df['close']
        high = df['high']
        low = df['low']
        
        # EMA
        ema = close.ewm(span=self.ema_period).mean()
        ema_slope = (ema.iloc[-1] - ema.iloc[-20]) / ema.iloc[-20] * 100
        price_vs_ema = (close.iloc[-1] - ema.iloc[-1]) / ema.iloc[-1] * 100
        
        # ADX calculation
        adx = self._calculate_adx(high, low, close)
        
        # Determine trend
        is_trending = adx > self.trend_threshold
        
        if abs(ema_slope) < 0.5:
            trend_direction = "neutral"
        elif ema_slope > 0:
            trend_direction = "up"
        else:
            trend_direction = "down"
        
        if adx < 20:
            trend_strength = "none"
        elif adx < self.trend_threshold:
            trend_strength = "weak"
        else:
            trend_strength = "strong"
        
        return TrendMetrics(
            adx=float(adx),
            adx_threshold=self.trend_threshold,
            ema_slope=float(ema_slope),
            price_vs_ema=float(price_vs_ema),
            is_trending=is_trending,
            trend_direction=trend_direction,
            trend_strength=trend_strength
        )
    
    def _calculate_adx(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> float:
        """Calculate ADX indicator."""
        # True Range
        tr = pd.DataFrame({
            'hl': high - low,
            'hc': abs(high - close.shift(1)),
            'lc': abs(low - close.shift(1))
        }).max(axis=1)
        
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed averages
        atr = tr.rolling(period).mean()
        plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / atr
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        
        return float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else 0.0
    
    def get_trading_session(self, timestamp: datetime) -> TradingSession:
        """Determine current FX trading session."""
        hour = timestamp.hour
        
        # UTC times
        if 0 <= hour < 8:
            return TradingSession.ASIA
        elif 8 <= hour < 13:
            return TradingSession.LONDON
        elif 13 <= hour < 16:
            return TradingSession.OVERLAP  # London-NY
        elif 16 <= hour < 21:
            return TradingSession.NEW_YORK
        else:
            return TradingSession.OFF_HOURS


# ============================================================================
# DYNAMIC COINTEGRATION VALIDATOR
# ============================================================================

class DynamicCointegrationValidator:
    """
    Validates cointegration dynamically across multiple windows.
    
    Cointegration is NOT assumed permanent. This class:
    1. Tests multiple window sizes
    2. Tracks temporal consistency
    3. Detects breakdowns early
    4. Provides confidence levels
    """
    
    def __init__(
        self,
        windows: List[int] = [250, 500, 750],
        pvalue_threshold: float = 0.05,
        min_consistency: float = 0.67,  # 2/3 windows must pass
        breakdown_lookback: int = 100,
        breakdown_threshold: float = 0.10  # Max 10% of recent windows can fail
    ):
        self.windows = windows
        self.pvalue_threshold = pvalue_threshold
        self.min_consistency = min_consistency
        self.breakdown_lookback = breakdown_lookback
        self.breakdown_threshold = breakdown_threshold
    
    def validate(
        self,
        price_a: pd.Series,
        price_b: pd.Series
    ) -> CointegrationStatus:
        """
        Validate cointegration dynamically.
        
        Returns:
            CointegrationStatus with full analysis
        """
        n = len(price_a)
        
        # Test current window (full data)
        try:
            _, eg_pvalue_current, _ = coint(price_a.values, price_b.values)
            is_cointegrated_current = eg_pvalue_current < self.pvalue_threshold
        except:
            eg_pvalue_current = 1.0
            is_cointegrated_current = False
        
        # Test multiple windows
        windows_passed = 0
        tested_windows = []
        
        for window in self.windows:
            if n < window:
                continue
            
            tested_windows.append(window)
            
            try:
                _, pval, _ = coint(
                    price_a.iloc[-window:].values,
                    price_b.iloc[-window:].values
                )
                if pval < self.pvalue_threshold:
                    windows_passed += 1
            except:
                pass
        
        # Calculate consistency
        if tested_windows:
            consistency = windows_passed / len(tested_windows)
        else:
            consistency = 0.0
        
        # Check for recent breakdowns
        breakdown_count = 0
        rolling_window = min(self.windows)
        step = 20
        
        for i in range(0, min(self.breakdown_lookback, n - rolling_window), step):
            end_idx = n - i
            start_idx = end_idx - rolling_window
            
            if start_idx < 0:
                break
            
            try:
                _, pval, _ = coint(
                    price_a.iloc[start_idx:end_idx].values,
                    price_b.iloc[start_idx:end_idx].values
                )
                if pval > self.pvalue_threshold:
                    breakdown_count += 1
            except:
                pass
        
        total_checks = self.breakdown_lookback // step
        breakdown_frequency = breakdown_count / max(total_checks, 1)
        
        # Estimate days since last breakdown
        days_since_breakdown = 0
        for i in range(0, n - rolling_window, step):
            end_idx = n - i
            start_idx = end_idx - rolling_window
            
            try:
                _, pval, _ = coint(
                    price_a.iloc[start_idx:end_idx].values,
                    price_b.iloc[start_idx:end_idx].values
                )
                if pval > self.pvalue_threshold:
                    break
                days_since_breakdown += step
            except:
                break
        
        # Determine stability and confidence
        is_stable = (
            is_cointegrated_current and
            consistency >= self.min_consistency and
            breakdown_frequency <= self.breakdown_threshold
        )
        
        if consistency >= 0.90 and breakdown_frequency < 0.05:
            confidence = "high"
        elif consistency >= 0.67 and breakdown_frequency < 0.10:
            confidence = "medium"
        elif consistency >= 0.50:
            confidence = "low"
        else:
            confidence = "none"
        
        return CointegrationStatus(
            eg_pvalue_current=float(eg_pvalue_current),
            is_cointegrated_current=is_cointegrated_current,
            windows_tested=tested_windows,
            windows_passed=windows_passed,
            cointegration_consistency=consistency,
            days_since_last_breakdown=days_since_breakdown,
            breakdown_frequency=breakdown_frequency,
            is_stable=is_stable,
            confidence=confidence
        )


# ============================================================================
# SPREAD HEALTH MONITOR
# ============================================================================

class SpreadHealthMonitor:
    """
    Monitors the statistical health of a spread.
    
    A spread can be cointegrated but unhealthy for trading if:
    - Half-life is too long/unstable
    - Hurst indicates trending
    - Hedge ratio is drifting
    - Volatility is unstable
    """
    
    def __init__(
        self,
        half_life_max: int = 60,
        half_life_min: int = 5,
        hurst_max: float = 0.55,
        hedge_drift_threshold: float = 2.0,  # Z-score of hedge ratio
        vol_stability_min: float = 0.60
    ):
        self.half_life_max = half_life_max
        self.half_life_min = half_life_min
        self.hurst_max = hurst_max
        self.hedge_drift_threshold = hedge_drift_threshold
        self.vol_stability_min = vol_stability_min
    
    def analyze(
        self,
        price_a: pd.Series,
        price_b: pd.Series,
        window: int = 120
    ) -> SpreadHealth:
        """
        Analyze spread health.
        
        Returns:
            SpreadHealth with all metrics
        """
        # Calculate hedge ratio
        X = sm.add_constant(price_b.values)
        model = sm.OLS(price_a.values, X).fit()
        hedge_ratio = float(model.params[1])
        
        # Rolling hedge ratio
        rolling_hedge = []
        for i in range(window, len(price_a), 20):
            X_roll = sm.add_constant(price_b.iloc[i-window:i].values)
            model_roll = sm.OLS(price_a.iloc[i-window:i].values, X_roll).fit()
            rolling_hedge.append(model_roll.params[1])
        
        if rolling_hedge:
            hedge_mean = np.mean(rolling_hedge)
            hedge_std = np.std(rolling_hedge)
            hedge_zscore = (hedge_ratio - hedge_mean) / hedge_std if hedge_std > 0 else 0
            hedge_stable = abs(hedge_zscore) < self.hedge_drift_threshold
        else:
            hedge_zscore = 0
            hedge_stable = True
        
        # Construct spread
        spread = price_a - hedge_ratio * price_b
        spread_clean = spread.dropna()
        
        # ADF test
        try:
            adf_result = adfuller(spread_clean.values, maxlag=20)
            adf_pvalue = float(adf_result[1])
            is_stationary = adf_pvalue < 0.05
        except:
            adf_pvalue = 1.0
            is_stationary = False
        
        # Half-life
        half_life = self._calculate_half_life(spread_clean)
        half_life_stable = self.half_life_min <= half_life <= self.half_life_max
        
        # Hurst exponent
        hurst = self._calculate_hurst(spread_clean.values)
        is_mean_reverting = hurst < self.hurst_max
        
        # Spread volatility
        spread_returns = spread.pct_change().dropna()
        spread_vol = float(spread_returns.std())
        
        # Rolling volatility percentile
        rolling_vol = spread_returns.rolling(20).std()
        current_vol = rolling_vol.iloc[-1] if not pd.isna(rolling_vol.iloc[-1]) else spread_vol
        vol_percentile = (rolling_vol < current_vol).sum() / len(rolling_vol.dropna()) * 100
        
        # Volatility stability
        vol_cv = rolling_vol.std() / rolling_vol.mean() if rolling_vol.mean() > 0 else 1
        spread_vol_stable = vol_cv < 0.5  # Coefficient of variation < 50%
        
        # Overall health
        is_healthy = (
            is_stationary and
            half_life_stable and
            is_mean_reverting and
            hedge_stable
        )
        
        # Health score
        health_score = 0
        if is_stationary:
            health_score += 25
        if half_life_stable:
            health_score += 25
        if is_mean_reverting:
            health_score += 25
        if hedge_stable:
            health_score += 15
        if spread_vol_stable:
            health_score += 10
        
        return SpreadHealth(
            adf_pvalue=adf_pvalue,
            is_stationary=is_stationary,
            half_life=half_life,
            half_life_stable=half_life_stable,
            hurst_exponent=hurst,
            is_mean_reverting=is_mean_reverting,
            hedge_ratio=hedge_ratio,
            hedge_ratio_zscore=float(hedge_zscore),
            hedge_ratio_stable=hedge_stable,
            spread_vol=spread_vol,
            spread_vol_percentile=float(vol_percentile),
            spread_vol_stable=spread_vol_stable,
            is_healthy=is_healthy,
            health_score=health_score
        )
    
    def _calculate_half_life(self, spread: pd.Series) -> float:
        """Calculate half-life."""
        spread_lag = spread.shift(1).dropna()
        spread_diff = spread.diff().dropna()
        
        common = spread_lag.index.intersection(spread_diff.index)
        if len(common) < 100:
            return 9999.0
        
        y = spread_diff.loc[common].values
        X = spread_lag.loc[common].values.reshape(-1, 1)
        
        model = LinearRegression()
        model.fit(X, y)
        theta = model.coef_[0]
        
        if theta >= 0:
            return 9999.0
        
        half_life = -np.log(2) / theta
        return min(float(half_life), 9999.0)
    
    def _calculate_hurst(self, ts: np.ndarray, max_lag: int = 100) -> float:
        """Calculate Hurst exponent."""
        if len(ts) < max_lag * 2:
            max_lag = len(ts) // 4
        
        if max_lag < 10:
            return 0.5
        
        lags = range(10, max_lag)
        rs_values = []
        
        for lag in lags:
            rs_list = []
            for start in range(0, len(ts) - lag, lag):
                chunk = ts[start:start + lag]
                if len(chunk) < lag:
                    continue
                
                mean_chunk = np.mean(chunk)
                cumdev = np.cumsum(chunk - mean_chunk)
                R = np.max(cumdev) - np.min(cumdev)
                S = np.std(chunk)
                
                if S > 0:
                    rs_list.append(R / S)
            
            if rs_list:
                rs_values.append((lag, np.mean(rs_list)))
        
        if len(rs_values) < 5:
            return 0.5
        
        log_lags = np.log([x[0] for x in rs_values])
        log_rs = np.log([x[1] for x in rs_values])
        
        try:
            slope, _ = np.polyfit(log_lags, log_rs, 1)
            return min(max(float(slope), 0), 1)
        except:
            return 0.5
