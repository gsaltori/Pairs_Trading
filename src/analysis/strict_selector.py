"""
Institutional Forex Pair Selection Pipeline - STRICT VERSION.

This pipeline implements aggressive filtering to ensure only statistically
valid pairs with real mean-reversion characteristics pass through.

Key Principles:
1. ZERO tolerance for non-mean-reverting spreads
2. Half-life is a HARD filter, not a score component
3. Hurst > 0.55 = automatic rejection
4. Only economically-related pairs allowed
5. Prefer zero pairs over statistically invalid pairs
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Set
from itertools import combinations
import logging
import warnings

from statsmodels.tsa.stattools import coint, adfuller, kpss
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.regression.rolling import RollingOLS
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


# ============================================================================
# VALID FOREX PAIR COMBINATIONS (Economically Related Only)
# ============================================================================

VALID_PAIR_COMBINATIONS: Set[Tuple[str, str]] = {
    # ==========================================================
    # TIER 1: STRONGEST ECONOMIC RELATIONSHIPS
    # ==========================================================
    
    # USD-based majors (same quote currency)
    ("EURUSD", "GBPUSD"),      # Both vs USD, European economies - CLASSIC
    ("AUDUSD", "NZDUSD"),      # Oceania commodity currencies vs USD - CLASSIC
    
    # JPY crosses (same quote currency)
    ("EURJPY", "GBPJPY"),      # European vs JPY - HIGH CORRELATION
    ("AUDJPY", "NZDJPY"),      # Oceania vs JPY - HIGH CORRELATION
    ("EURJPY", "CHFJPY"),      # European safe havens vs JPY
    
    # ==========================================================
    # TIER 2: STRONG ECONOMIC RELATIONSHIPS  
    # ==========================================================
    
    # Same base currency (EUR)
    ("EURUSD", "EURJPY"),      # EUR vs USD/JPY
    ("EURUSD", "EURGBP"),      # EUR triangular
    ("EURGBP", "EURCHF"),      # EUR vs European
    ("EURAUD", "EURNZD"),      # EUR vs Oceania
    
    # Same base currency (GBP)
    ("GBPUSD", "GBPJPY"),      # GBP vs USD/JPY
    ("GBPAUD", "GBPNZD"),      # GBP vs Oceania
    
    # Same base currency (AUD/NZD)
    ("AUDUSD", "AUDJPY"),      # AUD vs USD/JPY
    ("NZDUSD", "NZDJPY"),      # NZD vs USD/JPY
    
    # Same quote currency (AUD)
    ("EURAUD", "GBPAUD"),      # European vs AUD - SAME QUOTE
    
    # Same quote currency (NZD)
    ("EURNZD", "GBPNZD"),      # European vs NZD - SAME QUOTE
    
    # Same quote currency (CAD)
    ("EURCAD", "GBPCAD"),      # European vs CAD
    ("AUDCAD", "NZDCAD"),      # Oceania vs CAD - COMMODITY
    
    # Same quote currency (CHF)
    ("USDCHF", "EURCHF"),      # USD/EUR vs CHF - SAFE HAVEN
    ("AUDCHF", "NZDCHF"),      # Oceania vs CHF
    
    # ==========================================================
    # TIER 3: MODERATE RELATIONSHIPS (TRIANGULAR)
    # ==========================================================
    
    # Inverse relationships
    ("EURUSD", "USDCHF"),      # EUR/CHF via USD (inverse correlation)
    
    # Commodity currencies
    ("USDCAD", "AUDUSD"),      # Commodity vs USD
    ("CADJPY", "AUDJPY"),      # Commodity vs JPY
    
    # Cross triangular
    ("GBPUSD", "EURGBP"),      # GBP triangular
    ("AUDNZD", "AUDUSD"),      # AUD triangular
}

def is_valid_forex_combination(sym_a: str, sym_b: str) -> bool:
    """Check if pair combination has economic relationship."""
    pair = tuple(sorted([sym_a, sym_b]))
    reverse_pair = (pair[1], pair[0])
    
    # Check direct match
    if pair in VALID_PAIR_COMBINATIONS or reverse_pair in VALID_PAIR_COMBINATIONS:
        return True
    
    # Check if original order matches
    if (sym_a, sym_b) in VALID_PAIR_COMBINATIONS or (sym_b, sym_a) in VALID_PAIR_COMBINATIONS:
        return True
    
    return False


# ============================================================================
# STRICT THRESHOLDS BY TIMEFRAME
# ============================================================================

@dataclass
class TimeframeThresholds:
    """Hard thresholds for each timeframe."""
    max_half_life: int          # Bars - HARD rejection above this
    optimal_half_life: Tuple[int, int]  # Optimal range for scoring
    min_trades_per_year: int    # Minimum expected trades
    bars_per_day: int
    bars_per_year: int


TIMEFRAME_CONFIG = {
    "M15": TimeframeThresholds(
        max_half_life=40,           # ~10 hours max
        optimal_half_life=(5, 20),
        min_trades_per_year=20,
        bars_per_day=96,
        bars_per_year=24192
    ),
    "M30": TimeframeThresholds(
        max_half_life=50,           # ~25 hours max
        optimal_half_life=(8, 30),
        min_trades_per_year=15,
        bars_per_day=48,
        bars_per_year=12096
    ),
    "H1": TimeframeThresholds(
        max_half_life=60,           # 2.5 days max
        optimal_half_life=(10, 40),
        min_trades_per_year=12,
        bars_per_day=24,
        bars_per_year=6048
    ),
    "H4": TimeframeThresholds(
        max_half_life=120,          # 20 days max
        optimal_half_life=(15, 60),
        min_trades_per_year=8,
        bars_per_day=6,
        bars_per_year=1512
    ),
    "D1": TimeframeThresholds(
        max_half_life=40,           # 40 days max
        optimal_half_life=(5, 25),
        min_trades_per_year=5,
        bars_per_day=1,
        bars_per_year=252
    ),
}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class StrictPairAnalysis:
    """Complete analysis for a forex pair - strict version."""
    pair: Tuple[str, str]
    
    # Pre-filter results
    is_economically_valid: bool
    economic_relationship: str
    
    # Correlation (HARD FILTER)
    pearson_correlation: float
    spearman_correlation: float
    correlation_stability: float
    passes_correlation: bool
    
    # Cointegration - Engle-Granger (STRICT)
    eg_pvalue: float
    eg_is_cointegrated: bool  # p < 0.02
    
    # Cointegration - Johansen
    johansen_trace_stat: float
    johansen_trace_cv: float
    johansen_is_cointegrated: bool
    
    # Rolling cointegration stability
    rolling_coint_stable: bool
    rolling_coint_breakdown_pct: float
    
    # Stationarity (ADF must pass)
    adf_pvalue: float
    adf_is_stationary: bool
    
    # CRITICAL: Half-life (HARD FILTER)
    half_life: float
    half_life_days: float
    passes_half_life: bool
    
    # CRITICAL: Hurst exponent (HARD FILTER)
    hurst_exponent: float
    passes_hurst: bool  # Must be < 0.55
    
    # Spread characteristics
    hedge_ratio: float
    hedge_ratio_stability: float
    current_zscore: float
    
    # Tradability
    estimated_trades_per_year: float
    passes_trade_frequency: bool
    
    # Final verdict
    passes_all_filters: bool
    rejection_reasons: List[str] = field(default_factory=list)
    quality_score: float = 0.0  # Only calculated if passes all filters


@dataclass
class StrictPipelineResult:
    """Results from strict pipeline."""
    timestamp: datetime
    timeframe: str
    thresholds: TimeframeThresholds
    
    # Input stats
    symbols_analyzed: int
    pairs_analyzed: int
    
    # Rejection funnel
    rejected_economic: int
    rejected_correlation: int
    rejected_cointegration: int
    rejected_stationarity: int
    rejected_half_life: int
    rejected_hurst: int
    rejected_trade_frequency: int
    
    # Final output
    final_candidates: int
    
    # All analyses
    all_analyses: List[StrictPairAnalysis]
    
    # Selected pairs (passed ALL filters)
    selected_pairs: List[StrictPairAnalysis]


# ============================================================================
# STRICT PAIR SELECTOR
# ============================================================================

class StrictForexPairSelector:
    """
    Institutional-grade pair selector with ZERO tolerance for invalid pairs.
    
    Philosophy:
    - Better to have 0 trades than statistically invalid trades
    - Every filter is a HARD filter, not a score component
    - Only economically-related pairs are considered
    - Mean reversion must be PROVEN, not assumed
    """
    
    # HARD THRESHOLDS - NOT NEGOTIABLE
    MIN_PEARSON_CORR = 0.70
    MIN_SPEARMAN_CORR = 0.70
    MIN_CORR_STABILITY = 0.60
    
    EG_PVALUE_THRESHOLD = 0.02      # Very strict - was 0.05
    JOHANSEN_MARGIN = 1.10          # 10% above critical value
    
    ADF_PVALUE_THRESHOLD = 0.05
    
    MAX_HURST = 0.55                # HARD rejection above this
    TARGET_HURST = 0.45             # Ideal
    
    ROLLING_COINT_WINDOWS = [250, 500]
    MAX_COINT_BREAKDOWN_PCT = 0.20  # Max 20% of windows can show breakdown
    
    def __init__(self, timeframe: str = "H1"):
        """Initialize with timeframe-specific thresholds."""
        if timeframe not in TIMEFRAME_CONFIG:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        self.timeframe = timeframe
        self.thresholds = TIMEFRAME_CONFIG[timeframe]
        
        logger.info(f"Initialized StrictForexPairSelector for {timeframe}")
        logger.info(f"  Max half-life: {self.thresholds.max_half_life} bars")
        logger.info(f"  Max Hurst: {self.MAX_HURST}")
        logger.info(f"  Min correlation: {self.MIN_PEARSON_CORR}")
        logger.info(f"  EG p-value threshold: {self.EG_PVALUE_THRESHOLD}")
    
    def run_pipeline(
        self,
        price_data: Dict[str, pd.Series],
        top_n: int = 5
    ) -> StrictPipelineResult:
        """
        Run strict pair selection pipeline.
        
        Most pairs will be rejected. This is by design.
        """
        symbols = list(price_data.keys())
        all_combinations = list(combinations(symbols, 2))
        
        logger.info(f"=" * 60)
        logger.info(f"STRICT FOREX PAIR SELECTION PIPELINE")
        logger.info(f"=" * 60)
        logger.info(f"Timeframe: {self.timeframe}")
        logger.info(f"Symbols: {len(symbols)}")
        logger.info(f"Pairs to analyze: {len(all_combinations)}")
        
        # Counters for rejection funnel
        rejected_economic = 0
        rejected_correlation = 0
        rejected_cointegration = 0
        rejected_stationarity = 0
        rejected_half_life = 0
        rejected_hurst = 0
        rejected_trade_frequency = 0
        
        all_analyses = []
        
        for i, (sym_a, sym_b) in enumerate(all_combinations):
            if (i + 1) % 20 == 0:
                logger.info(f"  Processing {i+1}/{len(all_combinations)}...")
            
            try:
                analysis = self._analyze_pair(
                    pair=(sym_a, sym_b),
                    price_a=price_data[sym_a],
                    price_b=price_data[sym_b]
                )
                
                if analysis:
                    all_analyses.append(analysis)
                    
                    # Track rejections
                    if not analysis.is_economically_valid:
                        rejected_economic += 1
                    elif not analysis.passes_correlation:
                        rejected_correlation += 1
                    elif not (analysis.eg_is_cointegrated or analysis.johansen_is_cointegrated):
                        rejected_cointegration += 1
                    elif not analysis.adf_is_stationary:
                        rejected_stationarity += 1
                    elif not analysis.passes_half_life:
                        rejected_half_life += 1
                    elif not analysis.passes_hurst:
                        rejected_hurst += 1
                    elif not analysis.passes_trade_frequency:
                        rejected_trade_frequency += 1
                        
            except Exception as e:
                logger.warning(f"Error analyzing {sym_a}/{sym_b}: {e}")
        
        # Select pairs that passed ALL filters
        selected = [a for a in all_analyses if a.passes_all_filters]
        selected.sort(key=lambda x: x.quality_score, reverse=True)
        selected = selected[:top_n]
        
        final_candidates = len(selected)
        
        # Log results
        logger.info(f"\n{'='*60}")
        logger.info(f"PIPELINE RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Rejected - Not economically related: {rejected_economic}")
        logger.info(f"Rejected - Low correlation: {rejected_correlation}")
        logger.info(f"Rejected - Not cointegrated: {rejected_cointegration}")
        logger.info(f"Rejected - Spread not stationary: {rejected_stationarity}")
        logger.info(f"Rejected - Half-life too long: {rejected_half_life}")
        logger.info(f"Rejected - Hurst too high: {rejected_hurst}")
        logger.info(f"Rejected - Too few trades/year: {rejected_trade_frequency}")
        logger.info(f"FINAL CANDIDATES: {final_candidates}")
        
        return StrictPipelineResult(
            timestamp=datetime.now(),
            timeframe=self.timeframe,
            thresholds=self.thresholds,
            symbols_analyzed=len(symbols),
            pairs_analyzed=len(all_combinations),
            rejected_economic=rejected_economic,
            rejected_correlation=rejected_correlation,
            rejected_cointegration=rejected_cointegration,
            rejected_stationarity=rejected_stationarity,
            rejected_half_life=rejected_half_life,
            rejected_hurst=rejected_hurst,
            rejected_trade_frequency=rejected_trade_frequency,
            final_candidates=final_candidates,
            all_analyses=all_analyses,
            selected_pairs=selected
        )
    
    def _analyze_pair(
        self,
        pair: Tuple[str, str],
        price_a: pd.Series,
        price_b: pd.Series
    ) -> Optional[StrictPairAnalysis]:
        """
        Analyze a single pair with strict filtering.
        
        Returns None if data is insufficient.
        Analysis includes rejection tracking at each stage.
        """
        # Align data
        common_idx = price_a.index.intersection(price_b.index)
        price_a = price_a.loc[common_idx].astype(float)
        price_b = price_b.loc[common_idx].astype(float)
        
        if len(price_a) < 1000:  # Need substantial data
            return None
        
        rejection_reasons = []
        passes_all = True
        
        # ================================================================
        # STAGE 1: ECONOMIC RELATIONSHIP (HARD FILTER)
        # ================================================================
        is_economically_valid = is_valid_forex_combination(pair[0], pair[1])
        economic_relationship = self._get_economic_relationship(pair[0], pair[1])
        
        if not is_economically_valid:
            passes_all = False
            rejection_reasons.append(f"No economic relationship between {pair[0]} and {pair[1]}")
        
        # ================================================================
        # STAGE 2: CORRELATION (HARD FILTER)
        # ================================================================
        pearson_corr = float(price_a.corr(price_b))
        spearman_corr = float(price_a.corr(price_b, method='spearman'))
        
        # Rolling correlation stability
        returns_a = price_a.pct_change().dropna()
        returns_b = price_b.pct_change().dropna()
        rolling_corr = returns_a.rolling(60).corr(returns_b).dropna()
        corr_stability = 1 - min(float(rolling_corr.std()) / 0.3, 1)
        
        passes_correlation = (
            pearson_corr >= self.MIN_PEARSON_CORR and
            spearman_corr >= self.MIN_SPEARMAN_CORR and
            corr_stability >= self.MIN_CORR_STABILITY
        )
        
        if not passes_correlation:
            passes_all = False
            if pearson_corr < self.MIN_PEARSON_CORR:
                rejection_reasons.append(f"Pearson correlation {pearson_corr:.3f} < {self.MIN_PEARSON_CORR}")
            if spearman_corr < self.MIN_SPEARMAN_CORR:
                rejection_reasons.append(f"Spearman correlation {spearman_corr:.3f} < {self.MIN_SPEARMAN_CORR}")
            if corr_stability < self.MIN_CORR_STABILITY:
                rejection_reasons.append(f"Correlation stability {corr_stability:.2f} < {self.MIN_CORR_STABILITY}")
        
        # ================================================================
        # STAGE 3: COINTEGRATION (STRICT)
        # ================================================================
        # Engle-Granger with strict threshold
        try:
            _, eg_pvalue, _ = coint(price_a.values, price_b.values)
            eg_pvalue = float(eg_pvalue)
            eg_is_cointegrated = eg_pvalue < self.EG_PVALUE_THRESHOLD
        except:
            eg_pvalue = 1.0
            eg_is_cointegrated = False
        
        # Johansen test
        try:
            data = pd.DataFrame({'a': price_a.values, 'b': price_b.values})
            joh_result = coint_johansen(data, det_order=0, k_ar_diff=1)
            johansen_trace_stat = float(joh_result.lr1[0])
            johansen_trace_cv = float(joh_result.cvt[0, 1])
            # Require margin above critical value
            johansen_is_cointegrated = johansen_trace_stat > johansen_trace_cv * self.JOHANSEN_MARGIN
        except:
            johansen_trace_stat = 0
            johansen_trace_cv = 1
            johansen_is_cointegrated = False
        
        # Rolling cointegration stability
        rolling_coint_stable, breakdown_pct = self._check_rolling_cointegration(price_a, price_b)
        
        # Must pass at least one cointegration test AND be stable
        is_cointegrated = (eg_is_cointegrated or johansen_is_cointegrated) and rolling_coint_stable
        
        if not is_cointegrated:
            passes_all = False
            if not eg_is_cointegrated and not johansen_is_cointegrated:
                rejection_reasons.append(f"Not cointegrated (EG p={eg_pvalue:.4f}, Johansen={johansen_trace_stat:.1f}/{johansen_trace_cv:.1f})")
            elif not rolling_coint_stable:
                rejection_reasons.append(f"Cointegration unstable ({breakdown_pct:.0%} breakdown)")
        
        # ================================================================
        # STAGE 4: SPREAD CONSTRUCTION & STATIONARITY
        # ================================================================
        # OLS hedge ratio
        X = sm.add_constant(price_b.values)
        model = sm.OLS(price_a.values, X).fit()
        hedge_ratio = float(model.params[1])
        
        # Rolling hedge ratio stability
        try:
            rolling_model = RollingOLS(
                price_a, sm.add_constant(price_b), 
                window=120
            ).fit()
            rolling_beta = rolling_model.params.iloc[:, 1].dropna()
            hedge_ratio_stability = 1 - min(float(rolling_beta.std()) / abs(hedge_ratio), 1)
        except:
            hedge_ratio_stability = 0.5
        
        # Construct spread
        spread = price_a - hedge_ratio * price_b
        spread_clean = spread.dropna()
        
        # ADF test on spread
        try:
            adf_result = adfuller(spread_clean.values, maxlag=20, autolag='AIC')
            adf_pvalue = float(adf_result[1])
            adf_is_stationary = adf_pvalue < self.ADF_PVALUE_THRESHOLD
        except:
            adf_pvalue = 1.0
            adf_is_stationary = False
        
        if not adf_is_stationary:
            passes_all = False
            rejection_reasons.append(f"Spread not stationary (ADF p={adf_pvalue:.4f})")
        
        # Z-score
        zscore_window = 60
        spread_rolling_mean = spread.rolling(zscore_window).mean()
        spread_rolling_std = spread.rolling(zscore_window).std()
        zscore = (spread - spread_rolling_mean) / spread_rolling_std
        current_zscore = float(zscore.iloc[-1]) if not pd.isna(zscore.iloc[-1]) else 0
        
        # ================================================================
        # STAGE 5: HALF-LIFE (HARD FILTER)
        # ================================================================
        half_life = self._calculate_half_life(spread_clean)
        half_life_days = half_life / self.thresholds.bars_per_day
        
        passes_half_life = half_life <= self.thresholds.max_half_life
        
        if not passes_half_life:
            passes_all = False
            rejection_reasons.append(
                f"Half-life {half_life:.0f} bars ({half_life_days:.1f} days) > max {self.thresholds.max_half_life} bars"
            )
        
        # ================================================================
        # STAGE 6: HURST EXPONENT (HARD FILTER)
        # ================================================================
        hurst = self._calculate_hurst(spread_clean.values)
        passes_hurst = hurst < self.MAX_HURST
        
        if not passes_hurst:
            passes_all = False
            rejection_reasons.append(f"Hurst {hurst:.3f} >= {self.MAX_HURST} (not mean-reverting)")
        
        # ================================================================
        # STAGE 7: TRADE FREQUENCY
        # ================================================================
        estimated_trades = self._estimate_trades_per_year(half_life)
        passes_trade_frequency = estimated_trades >= self.thresholds.min_trades_per_year
        
        if not passes_trade_frequency:
            passes_all = False
            rejection_reasons.append(
                f"Estimated {estimated_trades:.0f} trades/year < min {self.thresholds.min_trades_per_year}"
            )
        
        # ================================================================
        # QUALITY SCORE (only if passes all filters)
        # ================================================================
        quality_score = 0.0
        if passes_all:
            quality_score = self._calculate_quality_score(
                pearson_corr, spearman_corr, corr_stability,
                eg_pvalue, johansen_is_cointegrated,
                adf_pvalue, half_life, hurst,
                current_zscore, hedge_ratio_stability
            )
        
        return StrictPairAnalysis(
            pair=pair,
            is_economically_valid=is_economically_valid,
            economic_relationship=economic_relationship,
            pearson_correlation=pearson_corr,
            spearman_correlation=spearman_corr,
            correlation_stability=corr_stability,
            passes_correlation=passes_correlation,
            eg_pvalue=eg_pvalue,
            eg_is_cointegrated=eg_is_cointegrated,
            johansen_trace_stat=johansen_trace_stat,
            johansen_trace_cv=johansen_trace_cv,
            johansen_is_cointegrated=johansen_is_cointegrated,
            rolling_coint_stable=rolling_coint_stable,
            rolling_coint_breakdown_pct=breakdown_pct,
            adf_pvalue=adf_pvalue,
            adf_is_stationary=adf_is_stationary,
            half_life=half_life,
            half_life_days=half_life_days,
            passes_half_life=passes_half_life,
            hurst_exponent=hurst,
            passes_hurst=passes_hurst,
            hedge_ratio=hedge_ratio,
            hedge_ratio_stability=hedge_ratio_stability,
            current_zscore=current_zscore,
            estimated_trades_per_year=estimated_trades,
            passes_trade_frequency=passes_trade_frequency,
            passes_all_filters=passes_all,
            rejection_reasons=rejection_reasons,
            quality_score=quality_score
        )
    
    def _get_economic_relationship(self, sym_a: str, sym_b: str) -> str:
        """Describe economic relationship between symbols."""
        relationships = {
            # Tier 1
            ("EURUSD", "GBPUSD"): "European currencies vs USD [TIER 1]",
            ("AUDUSD", "NZDUSD"): "Oceania commodity currencies vs USD [TIER 1]",
            ("EURJPY", "GBPJPY"): "European currencies vs JPY [TIER 1]",
            ("AUDJPY", "NZDJPY"): "Oceania vs JPY [TIER 1]",
            ("CHFJPY", "EURJPY"): "European safe havens vs JPY [TIER 1]",
            # Tier 2
            ("EURUSD", "EURJPY"): "EUR vs USD/JPY [TIER 2]",
            ("EURUSD", "EURGBP"): "EUR triangular [TIER 2]",
            ("EURGBP", "EURCHF"): "EUR vs European [TIER 2]",
            ("EURAUD", "EURNZD"): "EUR vs Oceania [TIER 2]",
            ("GBPUSD", "GBPJPY"): "GBP vs USD/JPY [TIER 2]",
            ("GBPAUD", "GBPNZD"): "GBP vs Oceania [TIER 2]",
            ("AUDUSD", "AUDJPY"): "AUD vs USD/JPY [TIER 2]",
            ("NZDUSD", "NZDJPY"): "NZD vs USD/JPY [TIER 2]",
            ("EURAUD", "GBPAUD"): "European vs AUD (same quote) [TIER 2]",
            ("EURNZD", "GBPNZD"): "European vs NZD (same quote) [TIER 2]",
            ("EURCAD", "GBPCAD"): "European vs CAD [TIER 2]",
            ("AUDCAD", "NZDCAD"): "Oceania vs CAD (commodity) [TIER 2]",
            ("USDCHF", "EURCHF"): "USD/EUR vs CHF (safe haven) [TIER 2]",
            ("AUDCHF", "NZDCHF"): "Oceania vs CHF [TIER 2]",
            # Tier 3
            ("EURUSD", "USDCHF"): "EUR/CHF via USD (inverse) [TIER 3]",
            ("AUDUSD", "USDCAD"): "Commodity vs USD [TIER 3]",
            ("AUDJPY", "CADJPY"): "Commodity vs JPY [TIER 3]",
            ("EURGBP", "GBPUSD"): "GBP triangular [TIER 3]",
            ("AUDNZD", "AUDUSD"): "AUD triangular [TIER 3]",
        }
        
        # Try both orderings
        pair1 = (sym_a, sym_b)
        pair2 = (sym_b, sym_a)
        pair_sorted = tuple(sorted([sym_a, sym_b]))
        
        for key in [pair1, pair2, pair_sorted]:
            if key in relationships:
                return relationships[key]
        
        return "Unknown relationship"
    
    def _check_rolling_cointegration(
        self,
        price_a: pd.Series,
        price_b: pd.Series
    ) -> Tuple[bool, float]:
        """
        Check cointegration stability across rolling windows.
        
        Returns:
            (is_stable, breakdown_percentage)
        """
        n = len(price_a)
        breakdown_count = 0
        total_windows = 0
        
        for window in self.ROLLING_COINT_WINDOWS:
            if n < window + 100:
                continue
            
            # Check last 5 non-overlapping windows
            for i in range(5):
                end_idx = n - (i * window // 2)
                start_idx = end_idx - window
                
                if start_idx < 0:
                    break
                
                try:
                    _, pval, _ = coint(
                        price_a.iloc[start_idx:end_idx].values,
                        price_b.iloc[start_idx:end_idx].values
                    )
                    total_windows += 1
                    if pval > 0.10:  # Cointegration breakdown
                        breakdown_count += 1
                except:
                    pass
        
        if total_windows == 0:
            return True, 0.0
        
        breakdown_pct = breakdown_count / total_windows
        is_stable = breakdown_pct <= self.MAX_COINT_BREAKDOWN_PCT
        
        return is_stable, breakdown_pct
    
    def _calculate_half_life(self, spread: pd.Series) -> float:
        """Calculate half-life using Ornstein-Uhlenbeck process."""
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
            return 9999.0  # Not mean-reverting
        
        half_life = -np.log(2) / theta
        return min(float(half_life), 9999.0)
    
    def _calculate_hurst(self, ts: np.ndarray, max_lag: int = 100) -> float:
        """Calculate Hurst exponent using R/S method."""
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
    
    def _estimate_trades_per_year(self, half_life: float) -> float:
        """Estimate trades per year based on half-life."""
        if half_life >= 9999:
            return 0
        
        # Average trade duration ~ 1.5 * half_life
        avg_trade_duration = half_life * 1.5
        
        # Entry opportunity frequency (Z reaches ±2)
        entry_frequency = 0.08  # ~8% of time
        
        # Max trades limited by duration
        bars_per_year = self.thresholds.bars_per_year
        max_trades = bars_per_year / avg_trade_duration
        
        # Actual trades
        estimated = min(max_trades * entry_frequency * 5, max_trades)
        
        return estimated
    
    def _calculate_quality_score(
        self,
        pearson_corr: float,
        spearman_corr: float,
        corr_stability: float,
        eg_pvalue: float,
        johansen_passed: bool,
        adf_pvalue: float,
        half_life: float,
        hurst: float,
        current_zscore: float,
        hedge_stability: float
    ) -> float:
        """
        Calculate quality score for pairs that passed ALL filters.
        
        Higher score = better pair.
        """
        score = 0.0
        
        # Correlation quality (max 20 points)
        corr_avg = (pearson_corr + spearman_corr) / 2
        score += (corr_avg - 0.70) / 0.30 * 15  # 0.70->0, 1.0->15
        score += corr_stability * 5
        
        # Cointegration strength (max 25 points)
        if eg_pvalue < 0.01:
            score += 20
        elif eg_pvalue < 0.02:
            score += 15
        else:
            score += 10
        
        if johansen_passed:
            score += 5
        
        # Stationarity (max 15 points)
        if adf_pvalue < 0.01:
            score += 15
        elif adf_pvalue < 0.05:
            score += 10
        else:
            score += 5
        
        # Half-life optimality (max 20 points)
        opt_min, opt_max = self.thresholds.optimal_half_life
        if opt_min <= half_life <= opt_max:
            score += 20
        elif half_life < opt_min:
            score += 15
        else:
            # Linear decay for longer half-life
            decay = (half_life - opt_max) / (self.thresholds.max_half_life - opt_max)
            score += max(5, 20 - decay * 15)
        
        # Hurst quality (max 15 points)
        if hurst < 0.40:
            score += 15
        elif hurst < 0.45:
            score += 12
        elif hurst < 0.50:
            score += 8
        else:
            score += 4
        
        # Current opportunity (max 5 points)
        abs_z = abs(current_zscore)
        if abs_z >= 2.0:
            score += 5
        elif abs_z >= 1.5:
            score += 3
        
        return min(100, score)
    
    def generate_report(self, result: StrictPipelineResult) -> str:
        """Generate institutional-grade report."""
        lines = []
        lines.append("=" * 80)
        lines.append("STRICT FOREX PAIR SELECTION PIPELINE - INSTITUTIONAL GRADE")
        lines.append("=" * 80)
        
        lines.append(f"\nTimestamp: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Timeframe: {result.timeframe}")
        
        lines.append(f"\nTHRESHOLDS APPLIED:")
        lines.append(f"  Max half-life: {result.thresholds.max_half_life} bars")
        lines.append(f"  Max Hurst: {self.MAX_HURST}")
        lines.append(f"  Min Pearson/Spearman correlation: {self.MIN_PEARSON_CORR}")
        lines.append(f"  EG cointegration p-value: < {self.EG_PVALUE_THRESHOLD}")
        lines.append(f"  Min trades/year: {result.thresholds.min_trades_per_year}")
        
        lines.append(f"\n" + "-" * 80)
        lines.append("REJECTION FUNNEL")
        lines.append("-" * 80)
        
        total = result.pairs_analyzed
        lines.append(f"  Total pairs analyzed:           {total:>4}")
        lines.append(f"  ├─ Rejected (no economic link): {result.rejected_economic:>4} ({result.rejected_economic/total*100:.1f}%)")
        lines.append(f"  ├─ Rejected (low correlation):  {result.rejected_correlation:>4}")
        lines.append(f"  ├─ Rejected (not cointegrated): {result.rejected_cointegration:>4}")
        lines.append(f"  ├─ Rejected (not stationary):   {result.rejected_stationarity:>4}")
        lines.append(f"  ├─ Rejected (half-life long):   {result.rejected_half_life:>4}")
        lines.append(f"  ├─ Rejected (Hurst too high):   {result.rejected_hurst:>4}")
        lines.append(f"  ├─ Rejected (too few trades):   {result.rejected_trade_frequency:>4}")
        lines.append(f"  └─ FINAL CANDIDATES:            {result.final_candidates:>4} ({result.final_candidates/total*100:.1f}%)")
        
        if result.selected_pairs:
            lines.append(f"\n" + "-" * 80)
            lines.append("SELECTED PAIRS (Passed ALL Filters)")
            lines.append("-" * 80)
            
            for i, p in enumerate(result.selected_pairs, 1):
                signal = ""
                if p.current_zscore < -2.0:
                    signal = " 🟢 LONG SIGNAL"
                elif p.current_zscore > 2.0:
                    signal = " 🔴 SHORT SIGNAL"
                
                lines.append(f"\n#{i} {p.pair[0]}/{p.pair[1]} - Quality Score: {p.quality_score:.1f}/100{signal}")
                lines.append(f"    Economic: {p.economic_relationship}")
                lines.append(f"    ")
                lines.append(f"    CORRELATION:")
                lines.append(f"      Pearson: {p.pearson_correlation:.3f} | Spearman: {p.spearman_correlation:.3f}")
                lines.append(f"      Stability: {p.correlation_stability:.1%}")
                lines.append(f"    ")
                lines.append(f"    COINTEGRATION:")
                eg = "✓" if p.eg_is_cointegrated else "✗"
                joh = "✓" if p.johansen_is_cointegrated else "✗"
                lines.append(f"      Engle-Granger: {eg} (p={p.eg_pvalue:.4f})")
                lines.append(f"      Johansen: {joh} (trace={p.johansen_trace_stat:.1f} vs cv={p.johansen_trace_cv:.1f})")
                lines.append(f"      Rolling stability: {1-p.rolling_coint_breakdown_pct:.0%}")
                lines.append(f"    ")
                lines.append(f"    MEAN REVERSION:")
                lines.append(f"      Half-life: {p.half_life:.1f} bars ({p.half_life_days:.1f} days)")
                lines.append(f"      Hurst: {p.hurst_exponent:.3f} {'✓' if p.hurst_exponent < 0.5 else '⚠️'}")
                lines.append(f"      ADF p-value: {p.adf_pvalue:.4f}")
                lines.append(f"    ")
                lines.append(f"    TRADING:")
                lines.append(f"      Hedge ratio: {p.hedge_ratio:.4f} (stability: {p.hedge_ratio_stability:.1%})")
                lines.append(f"      Current Z-score: {p.current_zscore:+.2f}")
                lines.append(f"      Est. trades/year: {p.estimated_trades_per_year:.0f}")
        else:
            lines.append(f"\n" + "-" * 80)
            lines.append("NO PAIRS PASSED ALL FILTERS")
            lines.append("-" * 80)
            lines.append("\nThis is expected behavior for a strict institutional pipeline.")
            lines.append("Consider:")
            lines.append("  1. Using a different timeframe (try H4 or D1)")
            lines.append("  2. Adding more symbols to the universe")
            lines.append("  3. Waiting for market regime change")
            lines.append("\nDO NOT relax the filters to force trades.")
        
        # Show near-misses if no pairs passed
        if not result.selected_pairs:
            near_misses = [a for a in result.all_analyses 
                          if len(a.rejection_reasons) <= 2 and a.is_economically_valid]
            near_misses.sort(key=lambda x: len(x.rejection_reasons))
            
            if near_misses[:3]:
                lines.append(f"\n" + "-" * 80)
                lines.append("NEAR MISSES (for analysis only, NOT tradeable)")
                lines.append("-" * 80)
                
                for p in near_misses[:3]:
                    lines.append(f"\n  {p.pair[0]}/{p.pair[1]}")
                    lines.append(f"    Failed: {', '.join(p.rejection_reasons)}")
                    lines.append(f"    Half-life: {p.half_life:.1f} | Hurst: {p.hurst_exponent:.3f}")
        
        lines.append("\n" + "=" * 80)
        
        return "\n".join(lines)
