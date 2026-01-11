"""
FX Conditional Relative Value (CRV) System - Layer 2: Regime Filter.

This module implements the FX REGIME FILTER that determines whether
market conditions permit Relative Value trading.

Key Principle:
    Relative Value only works when markets are STABLE.
    During directional macro moves or high volatility, 
    divergences are NOT temporary - they are regime shifts.

The system BLOCKS trading during:
- High volatility breakouts
- Strong macro directional moves  
- Major economic events (FOMC, ECB, NFP)
- Risk-on/off extremes
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, date, time
from typing import Optional, List, Dict, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# REGIME ENUMS
# ============================================================================

class FXRegime(Enum):
    """FX Market Regime Classification."""
    # Favorable for CRV
    STABLE_LOW_VOL = "stable_low_vol"       # Best for CRV
    STABLE_NORMAL_VOL = "stable_normal_vol" # Good for CRV
    RANGE_BOUND = "range_bound"             # Acceptable for CRV
    
    # Unfavorable - NO CRV
    TRENDING_STRONG = "trending_strong"     # NO CRV
    HIGH_VOLATILITY = "high_volatility"     # NO CRV
    RISK_OFF_EXTREME = "risk_off_extreme"   # NO CRV
    RISK_ON_EXTREME = "risk_on_extreme"     # NO CRV
    MACRO_EVENT = "macro_event"             # NO CRV
    UNKNOWN = "unknown"


class RiskSentiment(Enum):
    """Risk-on/Risk-off classification."""
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    NEUTRAL = "neutral"
    EXTREME_ON = "extreme_on"
    EXTREME_OFF = "extreme_off"


# Regime trading permissions
REGIME_PERMITS_CRV = {
    FXRegime.STABLE_LOW_VOL: True,
    FXRegime.STABLE_NORMAL_VOL: True,
    FXRegime.RANGE_BOUND: True,
    FXRegime.TRENDING_STRONG: False,
    FXRegime.HIGH_VOLATILITY: False,
    FXRegime.RISK_OFF_EXTREME: False,
    FXRegime.RISK_ON_EXTREME: False,
    FXRegime.MACRO_EVENT: False,
    FXRegime.UNKNOWN: False,
}


# ============================================================================
# ECONOMIC CALENDAR - High Impact Events
# ============================================================================

# These events should BLOCK CRV trading
HIGH_IMPACT_EVENTS = [
    "FOMC",
    "ECB",
    "BOE",
    "BOJ", 
    "RBA",
    "RBNZ",
    "SNB",
    "NFP",
    "CPI",
    "GDP",
    "Employment",
    "Retail Sales",
    "PMI"
]


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class VolatilityRegimeData:
    """Volatility regime analysis."""
    # ATR metrics
    atr_current: float
    atr_percentile: float  # vs last N periods
    atr_trend: str         # "expanding", "contracting", "stable"
    
    # Parkinson / Garman-Klass
    realized_vol: float
    realized_vol_percentile: float
    
    # Classification
    vol_regime: str  # "low", "normal", "high", "extreme"
    is_elevated: bool
    is_expanding: bool


@dataclass
class TrendRegimeData:
    """Trend strength analysis."""
    # ADX
    adx_value: float
    adx_trend: str  # "strengthening", "weakening", "stable"
    
    # Directional
    plus_di: float
    minus_di: float
    directional_bias: str  # "bullish", "bearish", "neutral"
    
    # EMA analysis
    ema_slope_short: float  # 20 EMA slope
    ema_slope_long: float   # 200 EMA slope
    price_vs_ema200: float  # % distance
    
    # Classification
    trend_strength: str  # "none", "weak", "moderate", "strong"
    is_trending: bool


@dataclass
class RiskSentimentData:
    """Risk-on/Risk-off analysis."""
    # Proxy indicators
    usdjpy_zscore: float   # JPY strength proxy
    audjpy_zscore: float   # Risk proxy
    gold_trend: Optional[str]
    vix_level: Optional[float]
    
    # Classification
    sentiment: RiskSentiment
    sentiment_score: float  # -100 (extreme risk-off) to +100 (extreme risk-on)
    is_extreme: bool


@dataclass
class MacroEventData:
    """Macro event proximity analysis."""
    # Event detection
    has_upcoming_event: bool
    hours_to_event: Optional[int]
    event_name: Optional[str]
    event_currency: Optional[str]
    
    # Block status
    is_blocked: bool
    block_reason: Optional[str]


@dataclass
class FXRegimeAssessment:
    """Complete FX regime assessment."""
    timestamp: datetime
    
    # Component analyses
    volatility: VolatilityRegimeData
    trend: TrendRegimeData
    sentiment: RiskSentimentData
    macro: MacroEventData
    
    # Overall regime
    regime: FXRegime
    permits_crv: bool
    
    # Confidence
    confidence: float  # 0-1
    
    # Blocking reasons
    blocking_reasons: List[str] = field(default_factory=list)


# ============================================================================
# REGIME FILTER
# ============================================================================

class FXRegimeFilter:
    """
    Layer 2: FX Regime Filter for Conditional Relative Value.
    
    Determines whether current market conditions permit CRV trading.
    
    CRV is BLOCKED when:
    1. Volatility is high or expanding rapidly
    2. Strong directional trend present (ADX > 25)
    3. Extreme risk-on or risk-off conditions
    4. Within 24h of major macro event
    
    CRV is PERMITTED when:
    1. Volatility is low to normal
    2. No strong trend (ADX < 20)
    3. Risk sentiment is neutral
    4. No imminent macro events
    """
    
    def __init__(
        self,
        # Volatility thresholds
        atr_period: int = 14,
        vol_high_percentile: float = 75.0,
        vol_extreme_percentile: float = 90.0,
        vol_lookback: int = 100,
        
        # Trend thresholds
        adx_period: int = 14,
        adx_weak_threshold: float = 20.0,
        adx_strong_threshold: float = 25.0,
        
        # Risk sentiment thresholds
        sentiment_extreme_threshold: float = 2.0,  # Z-score
        
        # Macro event thresholds
        event_block_hours: int = 24,
        
        # EMA periods
        ema_short: int = 20,
        ema_long: int = 200,
    ):
        self.atr_period = atr_period
        self.vol_high_percentile = vol_high_percentile
        self.vol_extreme_percentile = vol_extreme_percentile
        self.vol_lookback = vol_lookback
        
        self.adx_period = adx_period
        self.adx_weak_threshold = adx_weak_threshold
        self.adx_strong_threshold = adx_strong_threshold
        
        self.sentiment_extreme_threshold = sentiment_extreme_threshold
        self.event_block_hours = event_block_hours
        
        self.ema_short = ema_short
        self.ema_long = ema_long
    
    def assess_regime(
        self,
        ohlc: pd.DataFrame,
        usdjpy: Optional[pd.Series] = None,
        audjpy: Optional[pd.Series] = None,
        upcoming_events: Optional[List[Dict]] = None,
        timestamp: Optional[datetime] = None
    ) -> FXRegimeAssessment:
        """
        Assess current FX regime.
        
        Args:
            ohlc: OHLC data for primary analysis
            usdjpy: USDJPY close prices for risk-off proxy
            audjpy: AUDJPY close prices for risk-on proxy
            upcoming_events: List of upcoming macro events
            timestamp: Assessment timestamp
            
        Returns:
            FXRegimeAssessment
        """
        timestamp = timestamp or datetime.now()
        blocking_reasons = []
        
        # 1. Volatility analysis
        vol_data = self._analyze_volatility(ohlc)
        
        if vol_data.vol_regime == "extreme":
            blocking_reasons.append(f"Extreme volatility: {vol_data.atr_percentile:.0f} percentile")
        elif vol_data.vol_regime == "high" and vol_data.is_expanding:
            blocking_reasons.append(f"High expanding volatility")
        
        # 2. Trend analysis
        trend_data = self._analyze_trend(ohlc)
        
        if trend_data.trend_strength == "strong":
            blocking_reasons.append(f"Strong trend: ADX={trend_data.adx_value:.1f}")
        
        # 3. Risk sentiment analysis
        sentiment_data = self._analyze_sentiment(usdjpy, audjpy)
        
        if sentiment_data.is_extreme:
            blocking_reasons.append(f"Extreme sentiment: {sentiment_data.sentiment.value}")
        
        # 4. Macro event analysis
        macro_data = self._analyze_macro_events(upcoming_events, timestamp)
        
        if macro_data.is_blocked:
            blocking_reasons.append(macro_data.block_reason or "Macro event")
        
        # 5. Determine overall regime
        regime = self._classify_regime(vol_data, trend_data, sentiment_data, macro_data)
        permits_crv = REGIME_PERMITS_CRV.get(regime, False)
        
        # 6. Calculate confidence
        confidence = self._calculate_confidence(vol_data, trend_data, sentiment_data)
        
        return FXRegimeAssessment(
            timestamp=timestamp,
            volatility=vol_data,
            trend=trend_data,
            sentiment=sentiment_data,
            macro=macro_data,
            regime=regime,
            permits_crv=permits_crv,
            confidence=confidence,
            blocking_reasons=blocking_reasons
        )
    
    def _analyze_volatility(self, ohlc: pd.DataFrame) -> VolatilityRegimeData:
        """Analyze volatility regime."""
        high = ohlc['high'].values
        low = ohlc['low'].values
        close = ohlc['close'].values
        
        # ATR
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        atr_series = pd.Series(tr).rolling(self.atr_period).mean()
        atr_current = float(atr_series.iloc[-1])
        
        # ATR percentile
        atr_history = atr_series.dropna().tail(self.vol_lookback)
        atr_percentile = float((atr_history < atr_current).sum() / len(atr_history) * 100)
        
        # ATR trend
        atr_recent = atr_series.tail(5).mean()
        atr_previous = atr_series.tail(20).head(15).mean()
        
        if atr_recent > atr_previous * 1.1:
            atr_trend = "expanding"
        elif atr_recent < atr_previous * 0.9:
            atr_trend = "contracting"
        else:
            atr_trend = "stable"
        
        # Parkinson volatility
        log_hl = np.log(high / low)
        parkinson = np.sqrt(1 / (4 * np.log(2)) * np.mean(log_hl[-20:] ** 2)) * np.sqrt(252)
        
        # Realized vol percentile
        parkinson_history = []
        for i in range(20, min(len(ohlc), self.vol_lookback + 20)):
            pv = np.sqrt(1 / (4 * np.log(2)) * np.mean(log_hl[i-20:i] ** 2))
            parkinson_history.append(pv)
        
        realized_vol_pct = 50.0
        if parkinson_history:
            realized_vol_pct = float((np.array(parkinson_history) < parkinson).sum() / len(parkinson_history) * 100)
        
        # Classify
        if atr_percentile >= self.vol_extreme_percentile:
            vol_regime = "extreme"
        elif atr_percentile >= self.vol_high_percentile:
            vol_regime = "high"
        elif atr_percentile <= 25:
            vol_regime = "low"
        else:
            vol_regime = "normal"
        
        return VolatilityRegimeData(
            atr_current=atr_current,
            atr_percentile=atr_percentile,
            atr_trend=atr_trend,
            realized_vol=parkinson,
            realized_vol_percentile=realized_vol_pct,
            vol_regime=vol_regime,
            is_elevated=atr_percentile >= self.vol_high_percentile,
            is_expanding=atr_trend == "expanding"
        )
    
    def _analyze_trend(self, ohlc: pd.DataFrame) -> TrendRegimeData:
        """Analyze trend strength."""
        high = ohlc['high']
        low = ohlc['low']
        close = ohlc['close']
        
        # ADX calculation
        adx, plus_di, minus_di = self._calculate_adx(high, low, close)
        
        # ADX trend
        adx_series = pd.Series([adx])  # Simplified
        if len(ohlc) > self.adx_period * 2:
            adx_recent = adx
            # Would need full ADX history for proper trend detection
            adx_trend = "stable"
        else:
            adx_trend = "stable"
        
        # Directional bias
        if plus_di > minus_di * 1.2:
            directional_bias = "bullish"
        elif minus_di > plus_di * 1.2:
            directional_bias = "bearish"
        else:
            directional_bias = "neutral"
        
        # EMA analysis
        ema_short = close.ewm(span=self.ema_short).mean()
        ema_long = close.ewm(span=self.ema_long).mean()
        
        ema_slope_short = (ema_short.iloc[-1] - ema_short.iloc[-10]) / ema_short.iloc[-10] * 100
        ema_slope_long = (ema_long.iloc[-1] - ema_long.iloc[-50]) / ema_long.iloc[-50] * 100
        price_vs_ema200 = (close.iloc[-1] - ema_long.iloc[-1]) / ema_long.iloc[-1] * 100
        
        # Classify trend strength
        if adx >= self.adx_strong_threshold:
            trend_strength = "strong"
        elif adx >= self.adx_weak_threshold:
            trend_strength = "moderate"
        elif adx >= 15:
            trend_strength = "weak"
        else:
            trend_strength = "none"
        
        return TrendRegimeData(
            adx_value=adx,
            adx_trend=adx_trend,
            plus_di=plus_di,
            minus_di=minus_di,
            directional_bias=directional_bias,
            ema_slope_short=float(ema_slope_short),
            ema_slope_long=float(ema_slope_long),
            price_vs_ema200=float(price_vs_ema200),
            trend_strength=trend_strength,
            is_trending=adx >= self.adx_weak_threshold
        )
    
    def _analyze_sentiment(
        self,
        usdjpy: Optional[pd.Series],
        audjpy: Optional[pd.Series]
    ) -> RiskSentimentData:
        """Analyze risk sentiment using FX proxies."""
        
        usdjpy_zscore = 0.0
        audjpy_zscore = 0.0
        
        # USDJPY Z-score (JPY strength = risk-off)
        if usdjpy is not None and len(usdjpy) > 60:
            mean = usdjpy.tail(60).mean()
            std = usdjpy.tail(60).std()
            if std > 0:
                usdjpy_zscore = float((usdjpy.iloc[-1] - mean) / std)
        
        # AUDJPY Z-score (risk proxy)
        if audjpy is not None and len(audjpy) > 60:
            mean = audjpy.tail(60).mean()
            std = audjpy.tail(60).std()
            if std > 0:
                audjpy_zscore = float((audjpy.iloc[-1] - mean) / std)
        
        # Combined sentiment score
        # Higher USDJPY + Higher AUDJPY = risk-on
        # Lower USDJPY + Lower AUDJPY = risk-off
        sentiment_score = (usdjpy_zscore + audjpy_zscore) / 2 * 50  # Scale to -100 to +100
        
        # Classify
        if sentiment_score > self.sentiment_extreme_threshold * 50:
            sentiment = RiskSentiment.EXTREME_ON
            is_extreme = True
        elif sentiment_score < -self.sentiment_extreme_threshold * 50:
            sentiment = RiskSentiment.EXTREME_OFF
            is_extreme = True
        elif sentiment_score > 25:
            sentiment = RiskSentiment.RISK_ON
            is_extreme = False
        elif sentiment_score < -25:
            sentiment = RiskSentiment.RISK_OFF
            is_extreme = False
        else:
            sentiment = RiskSentiment.NEUTRAL
            is_extreme = False
        
        return RiskSentimentData(
            usdjpy_zscore=usdjpy_zscore,
            audjpy_zscore=audjpy_zscore,
            gold_trend=None,
            vix_level=None,
            sentiment=sentiment,
            sentiment_score=sentiment_score,
            is_extreme=is_extreme
        )
    
    def _analyze_macro_events(
        self,
        upcoming_events: Optional[List[Dict]],
        timestamp: datetime
    ) -> MacroEventData:
        """Analyze proximity to macro events."""
        
        if not upcoming_events:
            return MacroEventData(
                has_upcoming_event=False,
                hours_to_event=None,
                event_name=None,
                event_currency=None,
                is_blocked=False,
                block_reason=None
            )
        
        # Find nearest high-impact event
        nearest_event = None
        min_hours = float('inf')
        
        for event in upcoming_events:
            event_time = event.get('datetime')
            if event_time is None:
                continue
            
            if isinstance(event_time, str):
                event_time = datetime.fromisoformat(event_time)
            
            hours_diff = (event_time - timestamp).total_seconds() / 3600
            
            # Only consider future events within block window
            if 0 <= hours_diff < min_hours:
                event_name = event.get('name', '')
                
                # Check if high impact
                is_high_impact = any(
                    hi_event.lower() in event_name.lower() 
                    for hi_event in HIGH_IMPACT_EVENTS
                )
                
                if is_high_impact:
                    min_hours = hours_diff
                    nearest_event = event
        
        if nearest_event and min_hours <= self.event_block_hours:
            return MacroEventData(
                has_upcoming_event=True,
                hours_to_event=int(min_hours),
                event_name=nearest_event.get('name'),
                event_currency=nearest_event.get('currency'),
                is_blocked=True,
                block_reason=f"{nearest_event.get('name')} in {int(min_hours)}h"
            )
        
        return MacroEventData(
            has_upcoming_event=nearest_event is not None,
            hours_to_event=int(min_hours) if nearest_event else None,
            event_name=nearest_event.get('name') if nearest_event else None,
            event_currency=nearest_event.get('currency') if nearest_event else None,
            is_blocked=False,
            block_reason=None
        )
    
    def _classify_regime(
        self,
        vol: VolatilityRegimeData,
        trend: TrendRegimeData,
        sentiment: RiskSentimentData,
        macro: MacroEventData
    ) -> FXRegime:
        """Classify overall FX regime."""
        
        # Check blockers first
        if macro.is_blocked:
            return FXRegime.MACRO_EVENT
        
        if vol.vol_regime == "extreme":
            return FXRegime.HIGH_VOLATILITY
        
        if sentiment.sentiment == RiskSentiment.EXTREME_OFF:
            return FXRegime.RISK_OFF_EXTREME
        
        if sentiment.sentiment == RiskSentiment.EXTREME_ON:
            return FXRegime.RISK_ON_EXTREME
        
        if trend.trend_strength == "strong":
            return FXRegime.TRENDING_STRONG
        
        if vol.vol_regime == "high" and vol.is_expanding:
            return FXRegime.HIGH_VOLATILITY
        
        # Favorable regimes
        if vol.vol_regime == "low" and trend.trend_strength in ["none", "weak"]:
            return FXRegime.STABLE_LOW_VOL
        
        if vol.vol_regime == "normal" and trend.trend_strength in ["none", "weak"]:
            return FXRegime.STABLE_NORMAL_VOL
        
        if trend.trend_strength == "none":
            return FXRegime.RANGE_BOUND
        
        return FXRegime.UNKNOWN
    
    def _calculate_adx(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series
    ) -> Tuple[float, float, float]:
        """Calculate ADX and DI values."""
        period = self.adx_period
        
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
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
        adx = dx.rolling(period).mean()
        
        return (
            float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else 0.0,
            float(plus_di.iloc[-1]) if not pd.isna(plus_di.iloc[-1]) else 0.0,
            float(minus_di.iloc[-1]) if not pd.isna(minus_di.iloc[-1]) else 0.0
        )
    
    def _calculate_confidence(
        self,
        vol: VolatilityRegimeData,
        trend: TrendRegimeData,
        sentiment: RiskSentimentData
    ) -> float:
        """Calculate regime classification confidence."""
        confidence = 1.0
        
        # Reduce confidence near thresholds
        if vol.atr_percentile > 60 and vol.atr_percentile < 80:
            confidence *= 0.8  # Near high vol threshold
        
        if trend.adx_value > 18 and trend.adx_value < 22:
            confidence *= 0.8  # Near trend threshold
        
        if abs(sentiment.sentiment_score) > 70 and abs(sentiment.sentiment_score) < 100:
            confidence *= 0.85  # Near extreme sentiment
        
        return confidence
