"""
FX Conditional Relative Value (CRV) System - Layer 1: FX-Native Structural Selection.

COMPLETE REDESIGN - Abandoning StatArb-derived logic.

This module implements INSTITUTIONAL FX pair selection based on:
1. MACRO COHERENCE (primary) - shared economic drivers
2. CONDITIONAL CORRELATION - not stability, but absence of breakdowns
3. OPERATIONAL VIABILITY - can we hedge and execute this?

Key Philosophy:
    "Trade less, but not never"
    
    The structural layer answers: "Could this pair work under SOME conditions?"
    NOT: "Does this pair work ALL the time?"
    
    The fine selection happens in:
    - Regime filter (Layer 2)
    - Conditional spread (Layer 3)
    - Signal generation (Layer 4)

FX Reality Acknowledgments:
    - Correlations WILL vary 0.3 to 0.9 - this is NORMAL
    - Hedge ratios WILL drift - we adapt, not reject
    - Structural "breaks" are often regime rotations
    - We need 3-10 pairs in normal conditions, not 0
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Set
from enum import Enum
import logging
import statsmodels.api as sm

# Import safe data functions
from src.crv.data_integrity import safe_returns, safe_rolling_correlation

logger = logging.getLogger(__name__)


# ============================================================================
# PILAR 1: MACRO COHERENCE FRAMEWORK
# ============================================================================

class MacroDriver(Enum):
    """Primary macro drivers in FX."""
    RISK_SENTIMENT = "risk_sentiment"     # Risk-on/off flows
    COMMODITY = "commodity"               # Commodity price exposure
    CARRY = "carry"                       # Interest rate differential
    SAFE_HAVEN = "safe_haven"             # Flight to safety
    GROWTH_DIFFERENTIAL = "growth"        # Relative growth expectations
    MONETARY_POLICY = "monetary_policy"   # Central bank divergence


class PairRelationship(Enum):
    """Types of structural relationships between FX pairs."""
    SHARED_QUOTE = "shared_quote"         # Same quote currency (vs USD, vs JPY)
    SHARED_BASE = "shared_base"           # Same base currency
    COMMODITY_BLOCK = "commodity_block"   # AUD, NZD, CAD cluster
    EUROPEAN_BLOCK = "european_block"     # EUR, GBP, CHF cluster
    SAFE_HAVEN_PAIR = "safe_haven"        # JPY, CHF dynamics
    TRIANGULAR = "triangular"             # Triangular arbitrage relationship
    RISK_PROXY = "risk_proxy"             # Risk-on/off proxy pairs


# Currency macro profiles
CURRENCY_MACRO_PROFILE: Dict[str, Dict[str, float]] = {
    # Currency: {driver: sensitivity} where sensitivity is -1 to +1
    "EUR": {
        "risk_sentiment": 0.3,
        "growth": 0.4,
        "monetary_policy": 0.6
    },
    "GBP": {
        "risk_sentiment": 0.4,
        "growth": 0.5,
        "monetary_policy": 0.5
    },
    "USD": {
        "risk_sentiment": -0.3,  # Risk-off = USD strength
        "safe_haven": 0.5,
        "monetary_policy": 0.7
    },
    "JPY": {
        "risk_sentiment": -0.7,  # Strong risk-off currency
        "safe_haven": 0.8,
        "carry": -0.6           # Funding currency
    },
    "CHF": {
        "risk_sentiment": -0.5,
        "safe_haven": 0.7,
        "carry": -0.4
    },
    "AUD": {
        "risk_sentiment": 0.7,   # High beta to risk
        "commodity": 0.8,
        "carry": 0.5
    },
    "NZD": {
        "risk_sentiment": 0.6,
        "commodity": 0.6,
        "carry": 0.4
    },
    "CAD": {
        "risk_sentiment": 0.4,
        "commodity": 0.7,        # Oil exposure
        "growth": 0.4
    }
}


# EXPANDED FX Relationships - More inclusive than before
FX_PAIR_RELATIONSHIPS: Dict[Tuple[str, str], Tuple[PairRelationship, float]] = {
    # Format: (pair): (relationship_type, macro_coherence_score 0-1)
    
    # === TIER 1: STRONGEST STRUCTURAL RELATIONSHIPS (0.85-1.0) ===
    # Same quote currency (vs USD) - near-perfect for RV
    ("EURUSD", "GBPUSD"): (PairRelationship.SHARED_QUOTE, 0.90),
    ("AUDUSD", "NZDUSD"): (PairRelationship.SHARED_QUOTE, 0.95),  # Highest
    ("EURUSD", "AUDUSD"): (PairRelationship.SHARED_QUOTE, 0.75),
    ("GBPUSD", "AUDUSD"): (PairRelationship.SHARED_QUOTE, 0.70),
    ("EURUSD", "NZDUSD"): (PairRelationship.SHARED_QUOTE, 0.70),
    ("GBPUSD", "NZDUSD"): (PairRelationship.SHARED_QUOTE, 0.65),
    ("EURUSD", "USDCAD"): (PairRelationship.SHARED_QUOTE, 0.60),  # Inverse
    ("GBPUSD", "USDCAD"): (PairRelationship.SHARED_QUOTE, 0.55),
    
    # Same quote currency (vs JPY)
    ("EURJPY", "GBPJPY"): (PairRelationship.SHARED_QUOTE, 0.88),
    ("AUDJPY", "NZDJPY"): (PairRelationship.SHARED_QUOTE, 0.92),
    ("EURJPY", "AUDJPY"): (PairRelationship.SHARED_QUOTE, 0.72),
    ("GBPJPY", "AUDJPY"): (PairRelationship.SHARED_QUOTE, 0.70),
    ("USDJPY", "EURJPY"): (PairRelationship.SHARED_QUOTE, 0.75),
    ("USDJPY", "GBPJPY"): (PairRelationship.SHARED_QUOTE, 0.72),
    ("USDJPY", "AUDJPY"): (PairRelationship.SHARED_QUOTE, 0.68),
    ("CADJPY", "AUDJPY"): (PairRelationship.SHARED_QUOTE, 0.70),
    ("NZDJPY", "CADJPY"): (PairRelationship.SHARED_QUOTE, 0.65),
    
    # === TIER 2: COMMODITY BLOCK (0.70-0.90) ===
    ("AUDCAD", "NZDCAD"): (PairRelationship.COMMODITY_BLOCK, 0.85),
    ("AUDCHF", "NZDCHF"): (PairRelationship.COMMODITY_BLOCK, 0.82),
    ("AUDNZD", "AUDCAD"): (PairRelationship.COMMODITY_BLOCK, 0.65),
    
    # === TIER 3: EUROPEAN BLOCK (0.65-0.85) ===
    ("EURGBP", "EURCHF"): (PairRelationship.EUROPEAN_BLOCK, 0.70),
    ("GBPCHF", "EURCHF"): (PairRelationship.EUROPEAN_BLOCK, 0.68),
    ("EURAUD", "GBPAUD"): (PairRelationship.EUROPEAN_BLOCK, 0.75),
    ("EURNZD", "GBPNZD"): (PairRelationship.EUROPEAN_BLOCK, 0.72),
    ("EURCAD", "GBPCAD"): (PairRelationship.EUROPEAN_BLOCK, 0.70),
    
    # === TIER 4: SHARED BASE (0.60-0.80) ===
    ("EURUSD", "EURJPY"): (PairRelationship.SHARED_BASE, 0.70),
    ("EURUSD", "EURGBP"): (PairRelationship.SHARED_BASE, 0.65),
    ("GBPUSD", "GBPJPY"): (PairRelationship.SHARED_BASE, 0.68),
    ("AUDUSD", "AUDJPY"): (PairRelationship.SHARED_BASE, 0.72),
    ("NZDUSD", "NZDJPY"): (PairRelationship.SHARED_BASE, 0.70),
    ("USDCHF", "USDJPY"): (PairRelationship.SHARED_BASE, 0.60),
    
    # === TIER 5: SAFE HAVEN DYNAMICS (0.55-0.75) ===
    ("USDJPY", "USDCHF"): (PairRelationship.SAFE_HAVEN_PAIR, 0.62),
    ("EURJPY", "EURCHF"): (PairRelationship.SAFE_HAVEN_PAIR, 0.65),
    ("GBPJPY", "GBPCHF"): (PairRelationship.SAFE_HAVEN_PAIR, 0.60),
    ("CHFJPY", "USDJPY"): (PairRelationship.SAFE_HAVEN_PAIR, 0.55),
    
    # === TIER 6: RISK PROXIES (0.55-0.70) ===
    ("AUDJPY", "EURJPY"): (PairRelationship.RISK_PROXY, 0.65),
    ("AUDUSD", "USDJPY"): (PairRelationship.RISK_PROXY, 0.58),
    ("NZDJPY", "EURJPY"): (PairRelationship.RISK_PROXY, 0.62),
}


def get_macro_coherence(sym_a: str, sym_b: str) -> Tuple[Optional[PairRelationship], float]:
    """
    Get macro coherence between two FX pairs.
    
    Returns:
        (relationship_type, coherence_score 0-1)
    """
    key1 = (sym_a, sym_b)
    key2 = (sym_b, sym_a)
    
    result = FX_PAIR_RELATIONSHIPS.get(key1) or FX_PAIR_RELATIONSHIPS.get(key2)
    
    if result:
        return result
    
    # If not in predefined list, calculate from currency profiles
    score = _calculate_macro_similarity(sym_a, sym_b)
    if score >= 0.50:
        return (PairRelationship.RISK_PROXY, score)
    
    return (None, 0.0)


def _calculate_macro_similarity(sym_a: str, sym_b: str) -> float:
    """Calculate macro similarity from currency profiles."""
    # Extract currencies
    base_a, quote_a = sym_a[:3], sym_a[3:]
    base_b, quote_b = sym_b[:3], sym_b[3:]
    
    # Check for shared currencies (structural relationship)
    shared_currencies = set([base_a, quote_a]) & set([base_b, quote_b])
    
    if len(shared_currencies) >= 1:
        # Have structural link
        base_score = 0.5
    else:
        base_score = 0.2
    
    # Add macro profile similarity
    try:
        profile_a = CURRENCY_MACRO_PROFILE.get(base_a, {})
        profile_b = CURRENCY_MACRO_PROFILE.get(base_b, {})
        
        common_drivers = set(profile_a.keys()) & set(profile_b.keys())
        
        if common_drivers:
            similarity = 0
            for driver in common_drivers:
                # Same sign = similar exposure
                if profile_a[driver] * profile_b[driver] > 0:
                    similarity += 0.1
            
            base_score += min(0.3, similarity)
    except:
        pass
    
    return min(1.0, base_score)


def has_macro_coherence(sym_a: str, sym_b: str, min_score: float = 0.50) -> bool:
    """Check if pair has sufficient macro coherence."""
    _, score = get_macro_coherence(sym_a, sym_b)
    return score >= min_score


# ============================================================================
# PILAR 2: CONDITIONAL CORRELATION (NOT STABILITY)
# ============================================================================

@dataclass
class ConditionalCorrelation:
    """Correlation assessed conditionally, not for stability."""
    # Current state
    current_correlation: float
    
    # Distribution (not stability metrics!)
    median_correlation: float
    correlation_iqr: float  # Interquartile range (robust)
    
    # Breakdown detection
    has_breakdown: bool          # Sustained negative correlation
    breakdown_frequency: float   # % of periods with corr < 0
    
    # Regime-conditional (the important part)
    correlation_low_vol: float   # Correlation in low vol periods
    correlation_high_vol: float  # Correlation in high vol periods
    
    # Assessment
    is_viable: bool
    viability_score: float  # 0-100
    notes: List[str] = field(default_factory=list)


def assess_conditional_correlation(
    price_a: pd.Series,
    price_b: pd.Series,
    window: int = 40,           # Shorter window for FX
    breakdown_threshold: float = -0.20,
    max_breakdown_freq: float = 0.15  # Max 15% of time can be negative
) -> ConditionalCorrelation:
    """
    Assess correlation CONDITIONALLY, not for stability.
    
    FX Reality: Correlations vary 0.3 to 0.9 - this is NORMAL.
    We care about:
    1. Is it generally positive?
    2. Does it break down violently and persistently?
    3. How does it behave in different vol regimes?
    """
    notes = []
    
    # Calculate returns using SAFE method (no FutureWarning)
    returns_a = safe_returns(price_a)
    returns_b = safe_returns(price_b)
    
    # Align
    common_idx = returns_a.index.intersection(returns_b.index)
    returns_a = returns_a.loc[common_idx]
    returns_b = returns_b.loc[common_idx]
    
    if len(returns_a) < window * 3:
        return ConditionalCorrelation(
            current_correlation=0,
            median_correlation=0,
            correlation_iqr=1,
            has_breakdown=True,
            breakdown_frequency=1.0,
            correlation_low_vol=0,
            correlation_high_vol=0,
            is_viable=False,
            viability_score=0,
            notes=["Insufficient data"]
        )
    
    # Rolling correlation
    rolling_corr = returns_a.rolling(window).corr(returns_b).dropna()
    
    # Current and distribution
    current_corr = float(rolling_corr.iloc[-1])
    median_corr = float(rolling_corr.median())
    q25 = float(rolling_corr.quantile(0.25))
    q75 = float(rolling_corr.quantile(0.75))
    iqr = q75 - q25
    
    # Breakdown detection
    breakdown_periods = (rolling_corr < breakdown_threshold).sum()
    breakdown_freq = breakdown_periods / len(rolling_corr)
    
    # Sustained breakdown = more than 5 consecutive periods below threshold
    consecutive_below = 0
    max_consecutive = 0
    for c in rolling_corr:
        if c < breakdown_threshold:
            consecutive_below += 1
            max_consecutive = max(max_consecutive, consecutive_below)
        else:
            consecutive_below = 0
    
    has_breakdown = max_consecutive > 5 or breakdown_freq > max_breakdown_freq
    
    if has_breakdown:
        notes.append(f"Breakdown detected: {breakdown_freq:.0%} of time < {breakdown_threshold}")
    
    # Regime-conditional correlation
    # Use return volatility as proxy for regime
    combined_vol = (returns_a.abs() + returns_b.abs()).rolling(window).mean()
    vol_median = combined_vol.median()
    
    low_vol_mask = combined_vol < vol_median
    high_vol_mask = combined_vol >= vol_median
    
    # Correlation in each regime
    low_vol_idx = low_vol_mask[low_vol_mask].index
    high_vol_idx = high_vol_mask[high_vol_mask].index
    
    if len(low_vol_idx) > window:
        corr_low_vol = float(returns_a.loc[low_vol_idx].corr(returns_b.loc[low_vol_idx]))
    else:
        corr_low_vol = median_corr
    
    if len(high_vol_idx) > window:
        corr_high_vol = float(returns_a.loc[high_vol_idx].corr(returns_b.loc[high_vol_idx]))
    else:
        corr_high_vol = median_corr
    
    notes.append(f"Low vol corr: {corr_low_vol:.2f}, High vol corr: {corr_high_vol:.2f}")
    
    # Viability assessment
    # KEY CHANGE: We don't require stability, we require viability
    viability_score = 0.0
    
    # Median correlation (40 points)
    if median_corr >= 0.60:
        viability_score += 40
    elif median_corr >= 0.40:
        viability_score += 30
    elif median_corr >= 0.20:
        viability_score += 20
    elif median_corr >= 0:
        viability_score += 10
    
    # No breakdown (30 points)
    if not has_breakdown:
        viability_score += 30
    elif breakdown_freq < 0.10:
        viability_score += 20
    elif breakdown_freq < 0.20:
        viability_score += 10
    
    # Correlation in low vol regime (30 points) - THIS IS WHERE WE TRADE
    if corr_low_vol >= 0.50:
        viability_score += 30
    elif corr_low_vol >= 0.30:
        viability_score += 20
    elif corr_low_vol >= 0.10:
        viability_score += 10
    
    # Is viable?
    is_viable = (
        median_corr >= 0.20 and           # At least weakly positive
        not has_breakdown and              # No severe breakdowns
        corr_low_vol >= 0.20               # Positive in low vol (where we trade)
    )
    
    return ConditionalCorrelation(
        current_correlation=current_corr,
        median_correlation=median_corr,
        correlation_iqr=iqr,
        has_breakdown=has_breakdown,
        breakdown_frequency=breakdown_freq,
        correlation_low_vol=corr_low_vol,
        correlation_high_vol=corr_high_vol,
        is_viable=is_viable,
        viability_score=viability_score,
        notes=notes
    )


# ============================================================================
# PILAR 3: OPERATIONAL VIABILITY
# ============================================================================

@dataclass
class OperationalViability:
    """Operational assessment - can we actually trade this?"""
    # Hedge ratio
    current_hedge_ratio: float
    hedge_ratio_usable: bool    # Is it reasonable (0.2 to 5.0)?
    
    # Spread behavior
    spread_drift_annual: float  # Annualized drift of spread
    spread_has_explosive_drift: bool
    
    # Execution feasibility
    avg_daily_range_a: float
    avg_daily_range_b: float
    relative_liquidity: str     # "high", "medium", "low"
    
    # Assessment
    is_operable: bool
    operability_score: float  # 0-100
    notes: List[str] = field(default_factory=list)


def assess_operational_viability(
    price_a: pd.Series,
    price_b: pd.Series,
    ohlc_a: Optional[pd.DataFrame] = None,
    ohlc_b: Optional[pd.DataFrame] = None,
    window: int = 60,
    max_annual_drift_pct: float = 50.0  # Max 50% annual drift
) -> OperationalViability:
    """
    Assess if the pair is operationally viable.
    
    We care about:
    1. Hedge ratio is usable (not extreme)
    2. Spread doesn't have explosive drift
    3. Both legs are liquid enough
    """
    notes = []
    
    # Current hedge ratio
    X = sm.add_constant(price_b.values[-window:])
    model = sm.OLS(price_a.values[-window:], X).fit()
    hedge_ratio = float(model.params[1])
    
    # Hedge ratio usability
    hedge_usable = 0.2 <= abs(hedge_ratio) <= 5.0
    
    if not hedge_usable:
        notes.append(f"Hedge ratio extreme: {hedge_ratio:.2f}")
    
    # Spread drift analysis
    spread = price_a - hedge_ratio * price_b
    spread_returns = safe_returns(spread)
    
    # Annualized drift (assuming H4 = 6 bars/day * 252 days)
    bars_per_year = 6 * 252  # H4 assumption, adjust as needed
    mean_return = spread_returns.mean()
    annual_drift = mean_return * bars_per_year * 100  # As percentage
    
    explosive_drift = abs(annual_drift) > max_annual_drift_pct
    
    if explosive_drift:
        notes.append(f"Explosive drift: {annual_drift:.0f}% annual")
    
    # Daily range analysis (liquidity proxy)
    if ohlc_a is not None:
        adr_a = ((ohlc_a['high'] - ohlc_a['low']) / ohlc_a['close']).mean() * 100
    else:
        adr_a = (price_a.diff().abs() / price_a).mean() * 100 * 4  # Rough estimate
    
    if ohlc_b is not None:
        adr_b = ((ohlc_b['high'] - ohlc_b['low']) / ohlc_b['close']).mean() * 100
    else:
        adr_b = (price_b.diff().abs() / price_b).mean() * 100 * 4
    
    # Liquidity classification
    min_adr = min(adr_a, adr_b)
    if min_adr >= 0.5:
        liquidity = "high"
    elif min_adr >= 0.3:
        liquidity = "medium"
    else:
        liquidity = "low"
    
    notes.append(f"Liquidity: {liquidity}")
    
    # Operability score
    score = 0.0
    
    # Hedge ratio usable (40 points)
    if hedge_usable:
        if 0.5 <= abs(hedge_ratio) <= 2.0:
            score += 40  # Ideal range
        else:
            score += 25  # Usable but not ideal
    
    # No explosive drift (40 points)
    if not explosive_drift:
        if abs(annual_drift) < 20:
            score += 40
        else:
            score += 25
    
    # Liquidity (20 points)
    if liquidity == "high":
        score += 20
    elif liquidity == "medium":
        score += 12
    else:
        score += 5
    
    is_operable = hedge_usable and not explosive_drift
    
    return OperationalViability(
        current_hedge_ratio=hedge_ratio,
        hedge_ratio_usable=hedge_usable,
        spread_drift_annual=annual_drift,
        spread_has_explosive_drift=explosive_drift,
        avg_daily_range_a=adr_a,
        avg_daily_range_b=adr_b,
        relative_liquidity=liquidity,
        is_operable=is_operable,
        operability_score=score,
        notes=notes
    )


# ============================================================================
# DATA CLASSES FOR OUTPUT
# ============================================================================

@dataclass
class StructuralPairAssessment:
    """Complete FX-native structural assessment."""
    pair: Tuple[str, str]
    
    # Pilar 1: Macro Coherence
    relationship: Optional[PairRelationship]
    macro_coherence_score: float
    has_macro_coherence: bool
    
    # Pilar 2: Conditional Correlation
    correlation: ConditionalCorrelation
    
    # Pilar 3: Operational Viability
    operations: OperationalViability
    
    # Overall
    structural_score: float
    is_structurally_valid: bool
    tier: str  # "A", "B", "C" or "REJECTED"
    
    # Notes
    validation_notes: List[str] = field(default_factory=list)
    rejection_reasons: List[str] = field(default_factory=list)


# ============================================================================
# FX-NATIVE STRUCTURAL PAIR SELECTOR
# ============================================================================

class FXStructuralPairSelector:
    """
    Layer 1: FX-Native Structural Pair Selection.
    
    REDESIGNED from StatArb logic to FX Institutional logic.
    
    Selection criteria:
    1. MACRO COHERENCE (50% weight) - Do they share economic drivers?
    2. CONDITIONAL CORRELATION (30% weight) - Viable, not stable
    3. OPERATIONAL VIABILITY (20% weight) - Can we trade this?
    
    Expected output: 3-10 pairs in normal conditions
    Can be 0 only in extreme stress
    
    Philosophy: "Could work under SOME conditions" not "Works ALL the time"
    """
    
    def __init__(
        self,
        # Macro coherence
        min_macro_score: float = 0.50,
        
        # Correlation (relaxed!)
        min_median_correlation: float = 0.20,  # Much lower than StatArb!
        max_breakdown_frequency: float = 0.15,
        
        # Operations
        min_operability: float = 40.0,
        
        # Overall
        min_structural_score: float = 45.0,  # Lower threshold
        
        # Tiering thresholds
        tier_a_threshold: float = 70.0,
        tier_b_threshold: float = 55.0,
    ):
        self.min_macro_score = min_macro_score
        self.min_median_correlation = min_median_correlation
        self.max_breakdown_frequency = max_breakdown_frequency
        self.min_operability = min_operability
        self.min_structural_score = min_structural_score
        
        self.tier_a_threshold = tier_a_threshold
        self.tier_b_threshold = tier_b_threshold
    
    def assess_pair(
        self,
        pair: Tuple[str, str],
        price_a: pd.Series,
        price_b: pd.Series,
        ohlc_a: Optional[pd.DataFrame] = None,
        ohlc_b: Optional[pd.DataFrame] = None
    ) -> StructuralPairAssessment:
        """
        Perform FX-native structural assessment.
        
        Uses 3 pillars with weighted scoring.
        Designed to produce 3-10 valid pairs, not 0.
        """
        validation_notes = []
        rejection_reasons = []
        
        # === PILAR 1: MACRO COHERENCE (50% weight) ===
        relationship, macro_score = get_macro_coherence(pair[0], pair[1])
        has_macro = macro_score >= self.min_macro_score
        
        if has_macro:
            validation_notes.append(f"Macro: {relationship.value if relationship else 'derived'} ({macro_score:.2f})")
        else:
            rejection_reasons.append(f"Low macro coherence: {macro_score:.2f}")
        
        # === PILAR 2: CONDITIONAL CORRELATION (30% weight) ===
        corr = assess_conditional_correlation(
            price_a, price_b,
            window=40,
            breakdown_threshold=-0.20,
            max_breakdown_freq=self.max_breakdown_frequency
        )
        
        if corr.is_viable:
            validation_notes.append(f"Correlation viable: median={corr.median_correlation:.2f}")
        else:
            if corr.has_breakdown:
                rejection_reasons.append(f"Correlation breakdown: {corr.breakdown_frequency:.0%} negative")
            if corr.median_correlation < self.min_median_correlation:
                rejection_reasons.append(f"Correlation too low: {corr.median_correlation:.2f}")
        
        validation_notes.extend(corr.notes)
        
        # === PILAR 3: OPERATIONAL VIABILITY (20% weight) ===
        ops = assess_operational_viability(
            price_a, price_b,
            ohlc_a, ohlc_b,
            window=60
        )
        
        if ops.is_operable:
            validation_notes.append(f"Operable: HR={ops.current_hedge_ratio:.2f}")
        else:
            if not ops.hedge_ratio_usable:
                rejection_reasons.append(f"Hedge ratio unusable: {ops.current_hedge_ratio:.2f}")
            if ops.spread_has_explosive_drift:
                rejection_reasons.append(f"Explosive drift: {ops.spread_drift_annual:.0f}%/year")
        
        validation_notes.extend(ops.notes)
        
        # === CALCULATE WEIGHTED SCORE ===
        # Macro: 50%, Correlation: 30%, Operations: 20%
        structural_score = (
            macro_score * 100 * 0.50 +
            corr.viability_score * 0.30 +
            ops.operability_score * 0.20
        )
        
        # === VALIDATION LOGIC ===
        # Key change: OR logic for soft requirements, AND only for hard blockers
        
        # Hard blockers (must pass)
        hard_pass = (
            macro_score >= 0.40 and                    # Some macro relationship
            not corr.has_breakdown and                  # No severe breakdown
            ops.hedge_ratio_usable                      # Hedge ratio usable
        )
        
        # Soft requirements (score-based)
        soft_score = structural_score >= self.min_structural_score
        
        is_valid = hard_pass and soft_score
        
        # Tiering
        if is_valid:
            if structural_score >= self.tier_a_threshold:
                tier = "A"
            elif structural_score >= self.tier_b_threshold:
                tier = "B"
            else:
                tier = "C"
        else:
            tier = "REJECTED"
        
        return StructuralPairAssessment(
            pair=pair,
            relationship=relationship,
            macro_coherence_score=macro_score,
            has_macro_coherence=has_macro,
            correlation=corr,
            operations=ops,
            structural_score=structural_score,
            is_structurally_valid=is_valid,
            tier=tier,
            validation_notes=validation_notes,
            rejection_reasons=rejection_reasons
        )
    
    def select_valid_pairs(
        self,
        price_data: Dict[str, pd.Series],
        ohlc_data: Optional[Dict[str, pd.DataFrame]] = None,
        symbols: Optional[List[str]] = None,
        max_pairs: int = 15
    ) -> List[StructuralPairAssessment]:
        """
        Select structurally valid FX pairs.
        
        Expected to return 3-10 pairs in normal conditions.
        Sorted by tier then score.
        """
        from itertools import combinations
        
        if symbols is None:
            symbols = list(price_data.keys())
        
        all_pairs = list(combinations(symbols, 2))
        assessments = []
        
        logger.info(f"Assessing {len(all_pairs)} pair combinations...")
        
        # First pass: filter by macro coherence (fast)
        macro_valid = []
        for pair in all_pairs:
            sym_a, sym_b = pair
            if sym_a not in price_data or sym_b not in price_data:
                continue
            
            _, macro_score = get_macro_coherence(sym_a, sym_b)
            if macro_score >= 0.40:  # Minimal threshold for consideration
                macro_valid.append(pair)
        
        logger.info(f"Pairs with macro coherence: {len(macro_valid)}")
        
        # Second pass: full assessment
        for pair in macro_valid:
            sym_a, sym_b = pair
            
            ohlc_a = ohlc_data.get(sym_a) if ohlc_data else None
            ohlc_b = ohlc_data.get(sym_b) if ohlc_data else None
            
            assessment = self.assess_pair(
                pair=pair,
                price_a=price_data[sym_a],
                price_b=price_data[sym_b],
                ohlc_a=ohlc_a,
                ohlc_b=ohlc_b
            )
            
            assessments.append(assessment)
        
        # Filter valid and sort
        valid = [a for a in assessments if a.is_structurally_valid]
        
        # Sort by tier (A > B > C) then by score
        tier_order = {"A": 0, "B": 1, "C": 2}
        valid.sort(key=lambda x: (tier_order.get(x.tier, 3), -x.structural_score))
        
        # Limit to max pairs
        valid = valid[:max_pairs]
        
        logger.info(f"Structurally valid pairs: {len(valid)}")
        
        # Log tier distribution
        tier_counts = {"A": 0, "B": 0, "C": 0}
        for a in valid:
            tier_counts[a.tier] = tier_counts.get(a.tier, 0) + 1
        
        logger.info(f"Tier distribution: A={tier_counts['A']}, B={tier_counts['B']}, C={tier_counts['C']}")
        
        return valid
    
    def generate_report(self, assessments: List[StructuralPairAssessment]) -> str:
        """Generate structural selection report."""
        lines = []
        lines.append("=" * 80)
        lines.append("FX STRUCTURAL PAIR SELECTION - FX-NATIVE ASSESSMENT")
        lines.append("=" * 80)
        
        valid = [a for a in assessments if a.is_structurally_valid]
        rejected = [a for a in assessments if not a.is_structurally_valid]
        
        lines.append(f"\n📊 Summary: {len(valid)} valid, {len(rejected)} rejected")
        
        # Valid pairs by tier
        for tier in ["A", "B", "C"]:
            tier_pairs = [a for a in valid if a.tier == tier]
            if tier_pairs:
                lines.append(f"\n{'─' * 80}")
                lines.append(f"TIER {tier} PAIRS ({len(tier_pairs)})")
                lines.append(f"{'─' * 80}")
                
                for a in tier_pairs:
                    rel = a.relationship.value if a.relationship else "derived"
                    lines.append(f"\n  ✓ {a.pair[0]}/{a.pair[1]} - Score: {a.structural_score:.0f}")
                    lines.append(f"    Macro: {rel} ({a.macro_coherence_score:.2f})")
                    lines.append(f"    Correlation: median={a.correlation.median_correlation:.2f}, "
                               f"low_vol={a.correlation.correlation_low_vol:.2f}")
                    lines.append(f"    Operations: HR={a.operations.current_hedge_ratio:.2f}, "
                               f"drift={a.operations.spread_drift_annual:.0f}%/yr")
        
        # Top rejections
        if rejected:
            lines.append(f"\n{'─' * 80}")
            lines.append(f"TOP REJECTIONS (showing 5)")
            lines.append(f"{'─' * 80}")
            
            rejected.sort(key=lambda x: x.structural_score, reverse=True)
            
            for a in rejected[:5]:
                lines.append(f"\n  ✗ {a.pair[0]}/{a.pair[1]} - Score: {a.structural_score:.0f}")
                for reason in a.rejection_reasons[:2]:
                    lines.append(f"    - {reason}")
        
        lines.append("\n" + "=" * 80)
        
        return "\n".join(lines)


# ============================================================================
# BACKWARD COMPATIBILITY
# ============================================================================

# Aliases for backward compatibility
StructuralPairSelector = FXStructuralPairSelector

def has_economic_coherence(sym_a: str, sym_b: str) -> bool:
    """Backward compatible function."""
    return has_macro_coherence(sym_a, sym_b, min_score=0.40)

def get_pair_relationship(sym_a: str, sym_b: str) -> Optional[PairRelationship]:
    """Backward compatible function."""
    rel, _ = get_macro_coherence(sym_a, sym_b)
    return rel


# Export the relationships for external use
FX_RELATIONSHIPS = {k: v[0] for k, v in FX_PAIR_RELATIONSHIPS.items()}
