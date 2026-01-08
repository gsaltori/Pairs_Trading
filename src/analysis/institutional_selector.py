"""
Institutional Pair Selection Pipeline.

Implements rigorous statistical testing for pair selection:
1. Pre-filters (correlation, volatility, liquidity proxy)
2. Cointegration testing (Engle-Granger + Johansen)
3. Stationarity testing (ADF + KPSS)
4. Half-life validation
5. Structural break detection
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Any
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


@dataclass
class PairStatistics:
    """Comprehensive statistical analysis for a pair."""
    pair: Tuple[str, str]
    
    # Correlation metrics
    pearson_correlation: float
    spearman_correlation: float
    rolling_corr_mean: float
    rolling_corr_std: float
    correlation_stability: float
    
    # Cointegration - Engle-Granger
    eg_coint_stat: float
    eg_coint_pvalue: float
    eg_is_cointegrated: bool
    
    # Cointegration - Johansen
    johansen_trace_stat: float
    johansen_trace_cv: float  # Critical value 5%
    johansen_max_eigen_stat: float
    johansen_max_eigen_cv: float
    johansen_is_cointegrated: bool
    
    # Stationarity of spread
    adf_stat: float
    adf_pvalue: float
    adf_is_stationary: bool
    kpss_stat: float
    kpss_pvalue: float
    kpss_is_stationary: bool
    
    # Spread characteristics
    hedge_ratio: float
    hedge_ratio_std: float  # Rolling stability
    half_life: float
    half_life_days: float
    hurst_exponent: float
    
    # Current state
    current_zscore: float
    spread_mean: float
    spread_std: float
    
    # Volatility
    vol_a: float
    vol_b: float
    spread_vol: float
    
    # Quality metrics
    liquidity_score: float  # Based on spread stability
    regime_stability: float
    
    # Overall assessment
    overall_score: float
    is_tradeable: bool
    rejection_reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass  
class PipelineResult:
    """Results from the pair selection pipeline."""
    timestamp: datetime
    timeframe: str
    symbols_analyzed: int
    pairs_analyzed: int
    
    # Filtering stages
    passed_correlation: int
    passed_cointegration: int
    passed_stationarity: int
    passed_half_life: int
    final_candidates: int
    
    # All analyzed pairs
    all_pairs: List[PairStatistics]
    
    # Selected pairs (sorted by score)
    selected_pairs: List[PairStatistics]
    
    # Summary statistics
    avg_half_life: float
    avg_correlation: float
    cointegration_rate: float


class InstitutionalPairSelector:
    """
    Institutional-grade pair selection pipeline.
    
    Implements multi-stage filtering:
    Stage 1: Correlation pre-filter
    Stage 2: Cointegration testing (EG + Johansen)
    Stage 3: Stationarity confirmation (ADF + KPSS)
    Stage 4: Half-life validation
    Stage 5: Regime stability check
    """
    
    def __init__(
        self,
        # Correlation thresholds
        min_pearson_corr: float = 0.70,
        max_pearson_corr: float = 0.995,
        min_corr_stability: float = 0.60,
        
        # Cointegration thresholds
        eg_significance: float = 0.10,
        johansen_significance: float = 0.05,
        
        # Stationarity thresholds
        adf_significance: float = 0.10,
        kpss_significance: float = 0.05,
        
        # Half-life bounds (in bars)
        min_half_life: float = 5,
        max_half_life: float = 300,  # ~12 days at H1
        optimal_half_life_range: Tuple[float, float] = (20, 100),
        
        # Window sizes
        correlation_window: int = 60,
        regression_window: int = 120,
        zscore_window: int = 60,
        
        # Volatility filters
        min_volatility: float = 0.0001,  # Annualized
        max_volatility: float = 0.50,
    ):
        self.min_pearson_corr = min_pearson_corr
        self.max_pearson_corr = max_pearson_corr
        self.min_corr_stability = min_corr_stability
        
        self.eg_significance = eg_significance
        self.johansen_significance = johansen_significance
        
        self.adf_significance = adf_significance
        self.kpss_significance = kpss_significance
        
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.optimal_half_life_range = optimal_half_life_range
        
        self.correlation_window = correlation_window
        self.regression_window = regression_window
        self.zscore_window = zscore_window
        
        self.min_volatility = min_volatility
        self.max_volatility = max_volatility
    
    def run_pipeline(
        self,
        price_data: Dict[str, pd.Series],
        timeframe: str = "H1",
        top_n: int = 10
    ) -> PipelineResult:
        """
        Run complete pair selection pipeline.
        
        Args:
            price_data: {symbol: close_prices}
            timeframe: Data timeframe
            top_n: Number of top pairs to select
            
        Returns:
            PipelineResult with all analysis
        """
        symbols = list(price_data.keys())
        all_combinations = list(combinations(symbols, 2))
        
        logger.info(f"Starting institutional pair selection pipeline")
        logger.info(f"  Symbols: {len(symbols)}")
        logger.info(f"  Pairs to analyze: {len(all_combinations)}")
        
        # Stage counters
        passed_correlation = 0
        passed_cointegration = 0
        passed_stationarity = 0
        passed_half_life = 0
        
        all_pairs = []
        
        for i, (sym_a, sym_b) in enumerate(all_combinations):
            if (i + 1) % 10 == 0:
                logger.info(f"  Processing pair {i+1}/{len(all_combinations)}...")
            
            try:
                stats = self._analyze_pair(
                    pair=(sym_a, sym_b),
                    price_a=price_data[sym_a],
                    price_b=price_data[sym_b],
                    timeframe=timeframe
                )
                
                if stats:
                    all_pairs.append(stats)
                    
                    # Track pipeline stages
                    if stats.pearson_correlation >= self.min_pearson_corr:
                        passed_correlation += 1
                    if stats.eg_is_cointegrated or stats.johansen_is_cointegrated:
                        passed_cointegration += 1
                    if stats.adf_is_stationary:
                        passed_stationarity += 1
                    if self.min_half_life <= stats.half_life <= self.max_half_life:
                        passed_half_life += 1
                        
            except Exception as e:
                logger.warning(f"Error analyzing {sym_a}/{sym_b}: {e}")
        
        # Sort by score
        all_pairs.sort(key=lambda x: x.overall_score, reverse=True)
        
        # Select tradeable pairs
        selected = [p for p in all_pairs if p.is_tradeable][:top_n]
        
        # If not enough tradeable, include best non-tradeable with warnings
        if len(selected) < top_n:
            remaining = [p for p in all_pairs if not p.is_tradeable]
            selected.extend(remaining[:top_n - len(selected)])
        
        # Calculate summary stats
        if selected:
            valid_hl = [p.half_life for p in selected if p.half_life < 9999]
            avg_half_life = np.mean(valid_hl) if valid_hl else 0
            avg_correlation = np.mean([p.pearson_correlation for p in selected])
        else:
            avg_half_life = 0
            avg_correlation = 0
        
        cointegration_rate = passed_cointegration / len(all_combinations) if all_combinations else 0
        
        return PipelineResult(
            timestamp=datetime.now(),
            timeframe=timeframe,
            symbols_analyzed=len(symbols),
            pairs_analyzed=len(all_combinations),
            passed_correlation=passed_correlation,
            passed_cointegration=passed_cointegration,
            passed_stationarity=passed_stationarity,
            passed_half_life=passed_half_life,
            final_candidates=len([p for p in all_pairs if p.is_tradeable]),
            all_pairs=all_pairs,
            selected_pairs=selected,
            avg_half_life=avg_half_life,
            avg_correlation=avg_correlation,
            cointegration_rate=cointegration_rate
        )
    
    def _analyze_pair(
        self,
        pair: Tuple[str, str],
        price_a: pd.Series,
        price_b: pd.Series,
        timeframe: str
    ) -> Optional[PairStatistics]:
        """Comprehensive pair analysis."""
        # Align data
        common_idx = price_a.index.intersection(price_b.index)
        price_a = price_a.loc[common_idx].astype(float)
        price_b = price_b.loc[common_idx].astype(float)
        
        if len(price_a) < 500:
            return None
        
        rejection_reasons = []
        warnings_list = []
        
        # ========== STAGE 1: CORRELATION ==========
        pearson_corr = float(price_a.corr(price_b))
        spearman_corr = float(price_a.corr(price_b, method='spearman'))
        
        # Rolling correlation
        returns_a = price_a.pct_change().dropna()
        returns_b = price_b.pct_change().dropna()
        rolling_corr = returns_a.rolling(self.correlation_window).corr(returns_b).dropna()
        
        rolling_corr_mean = float(rolling_corr.mean())
        rolling_corr_std = float(rolling_corr.std())
        correlation_stability = 1 - min(rolling_corr_std / 0.3, 1) if rolling_corr_std < 0.3 else 0
        
        # ========== STAGE 2: COINTEGRATION ==========
        # Engle-Granger
        try:
            eg_stat, eg_pvalue, _ = coint(price_a.values, price_b.values)
            eg_pvalue = float(eg_pvalue)
            eg_is_cointegrated = eg_pvalue < self.eg_significance
        except:
            eg_stat, eg_pvalue = 0, 1.0
            eg_is_cointegrated = False
        
        # Johansen
        try:
            data = pd.DataFrame({'a': price_a.values, 'b': price_b.values})
            joh_result = coint_johansen(data, det_order=0, k_ar_diff=1)
            
            # Trace statistic for r=0 (no cointegration)
            johansen_trace_stat = float(joh_result.lr1[0])
            johansen_trace_cv = float(joh_result.cvt[0, 1])  # 5% critical value
            
            # Max eigenvalue statistic
            johansen_max_eigen_stat = float(joh_result.lr2[0])
            johansen_max_eigen_cv = float(joh_result.cvm[0, 1])
            
            johansen_is_cointegrated = (
                johansen_trace_stat > johansen_trace_cv or
                johansen_max_eigen_stat > johansen_max_eigen_cv
            )
        except:
            johansen_trace_stat = johansen_trace_cv = 0
            johansen_max_eigen_stat = johansen_max_eigen_cv = 0
            johansen_is_cointegrated = False
        
        # ========== HEDGE RATIO & SPREAD ==========
        # OLS hedge ratio
        X = sm.add_constant(price_b.values)
        model = sm.OLS(price_a.values, X).fit()
        hedge_ratio = float(model.params[1])
        
        # Rolling hedge ratio for stability
        try:
            rolling_model = RollingOLS(
                price_a, sm.add_constant(price_b), 
                window=self.regression_window
            ).fit()
            rolling_beta = rolling_model.params.iloc[:, 1].dropna()
            hedge_ratio_std = float(rolling_beta.std())
        except:
            hedge_ratio_std = 0.1
        
        # Construct spread
        spread = price_a - hedge_ratio * price_b
        spread_mean_val = float(spread.mean())
        spread_std_val = float(spread.std())
        
        # Rolling z-score
        spread_rolling_mean = spread.rolling(self.zscore_window).mean()
        spread_rolling_std = spread.rolling(self.zscore_window).std()
        zscore = (spread - spread_rolling_mean) / spread_rolling_std
        current_zscore = float(zscore.iloc[-1]) if not pd.isna(zscore.iloc[-1]) else 0
        
        # ========== STAGE 3: STATIONARITY ==========
        spread_clean = spread.dropna()
        
        # ADF Test
        try:
            adf_result = adfuller(spread_clean.values, maxlag=20, autolag='AIC')
            adf_stat = float(adf_result[0])
            adf_pvalue = float(adf_result[1])
            adf_is_stationary = adf_pvalue < self.adf_significance
        except:
            adf_stat, adf_pvalue = 0, 1.0
            adf_is_stationary = False
        
        # KPSS Test (null = stationary, so we want to NOT reject)
        try:
            kpss_result = kpss(spread_clean.values, regression='c', nlags='auto')
            kpss_stat = float(kpss_result[0])
            kpss_pvalue = float(kpss_result[1])
            kpss_is_stationary = kpss_pvalue > self.kpss_significance
        except:
            kpss_stat, kpss_pvalue = 0, 0
            kpss_is_stationary = False
        
        # ========== STAGE 4: HALF-LIFE ==========
        half_life = self._calculate_half_life(spread_clean)
        
        # Convert to days based on timeframe
        bars_per_day = {'M1': 1440, 'M5': 288, 'M15': 96, 'M30': 48, 
                        'H1': 24, 'H4': 6, 'D1': 1}
        bpd = bars_per_day.get(timeframe, 24)
        half_life_days = half_life / bpd if half_life < 9999 else 9999
        
        # ========== HURST EXPONENT ==========
        hurst = self._calculate_hurst(spread_clean.values)
        
        # ========== VOLATILITY ==========
        vol_a = float(returns_a.std() * np.sqrt(252 * bpd))
        vol_b = float(returns_b.std() * np.sqrt(252 * bpd))
        spread_returns = spread.pct_change().dropna()
        spread_vol = float(spread_returns.std() * np.sqrt(252 * bpd)) if len(spread_returns) > 0 else 0
        
        # ========== LIQUIDITY SCORE ==========
        # Based on spread stability and hedge ratio stability
        liquidity_score = max(0, 1 - hedge_ratio_std / 0.5) * 100
        
        # ========== REGIME STABILITY ==========
        # Check for structural breaks in correlation
        corr_recent = rolling_corr.tail(60).mean() if len(rolling_corr) > 60 else rolling_corr.mean()
        corr_historical = rolling_corr.mean()
        regime_stability = max(0, 1 - abs(corr_recent - corr_historical) / 0.2) * 100
        
        # ========== REJECTION LOGIC ==========
        is_tradeable = True
        
        # Correlation check
        if pearson_corr < self.min_pearson_corr:
            is_tradeable = False
            rejection_reasons.append(f"Low correlation: {pearson_corr:.3f}")
        elif pearson_corr > self.max_pearson_corr:
            is_tradeable = False
            rejection_reasons.append(f"Correlation too high: {pearson_corr:.3f}")
        
        # Cointegration check - need at least one test to pass
        is_cointegrated = eg_is_cointegrated or johansen_is_cointegrated
        if not is_cointegrated:
            is_tradeable = False
            rejection_reasons.append(f"Not cointegrated (EG p={eg_pvalue:.3f})")
        
        # Stationarity check - ADF should reject unit root
        if not adf_is_stationary:
            # Allow if strongly cointegrated
            if not (eg_pvalue < 0.05 or johansen_is_cointegrated):
                is_tradeable = False
                rejection_reasons.append(f"Spread not stationary (ADF p={adf_pvalue:.3f})")
            else:
                warnings_list.append(f"Weak ADF (p={adf_pvalue:.3f}) but cointegrated")
        
        # Half-life check
        if half_life < self.min_half_life:
            warnings_list.append(f"Half-life very short: {half_life:.1f} bars")
        elif half_life > self.max_half_life:
            if half_life > self.max_half_life * 2:
                is_tradeable = False
                rejection_reasons.append(f"Half-life too long: {half_life:.0f} bars ({half_life_days:.1f} days)")
            else:
                warnings_list.append(f"Half-life long: {half_life:.0f} bars ({half_life_days:.1f} days)")
        
        # Hurst exponent check
        if hurst > 0.55:
            warnings_list.append(f"Hurst indicates trending: {hurst:.3f}")
        
        # ========== OVERALL SCORE ==========
        overall_score = self._calculate_score(
            pearson_corr, correlation_stability,
            eg_is_cointegrated, eg_pvalue,
            johansen_is_cointegrated,
            adf_is_stationary, adf_pvalue,
            half_life, hurst,
            current_zscore,
            regime_stability
        )
        
        return PairStatistics(
            pair=pair,
            pearson_correlation=pearson_corr,
            spearman_correlation=spearman_corr,
            rolling_corr_mean=rolling_corr_mean,
            rolling_corr_std=rolling_corr_std,
            correlation_stability=correlation_stability,
            eg_coint_stat=float(eg_stat),
            eg_coint_pvalue=eg_pvalue,
            eg_is_cointegrated=eg_is_cointegrated,
            johansen_trace_stat=johansen_trace_stat,
            johansen_trace_cv=johansen_trace_cv,
            johansen_max_eigen_stat=johansen_max_eigen_stat,
            johansen_max_eigen_cv=johansen_max_eigen_cv,
            johansen_is_cointegrated=johansen_is_cointegrated,
            adf_stat=adf_stat,
            adf_pvalue=adf_pvalue,
            adf_is_stationary=adf_is_stationary,
            kpss_stat=kpss_stat,
            kpss_pvalue=kpss_pvalue,
            kpss_is_stationary=kpss_is_stationary,
            hedge_ratio=hedge_ratio,
            hedge_ratio_std=hedge_ratio_std,
            half_life=half_life,
            half_life_days=half_life_days,
            hurst_exponent=hurst,
            current_zscore=current_zscore,
            spread_mean=spread_mean_val,
            spread_std=spread_std_val,
            vol_a=vol_a,
            vol_b=vol_b,
            spread_vol=spread_vol,
            liquidity_score=liquidity_score,
            regime_stability=regime_stability,
            overall_score=overall_score,
            is_tradeable=is_tradeable,
            rejection_reasons=rejection_reasons,
            warnings=warnings_list
        )
    
    def _calculate_half_life(self, spread: pd.Series) -> float:
        """Calculate half-life using Ornstein-Uhlenbeck process."""
        spread_lag = spread.shift(1).dropna()
        spread_diff = spread.diff().dropna()
        
        common = spread_lag.index.intersection(spread_diff.index)
        if len(common) < 50:
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
    
    def _calculate_score(
        self,
        pearson_corr: float,
        corr_stability: float,
        eg_cointegrated: bool,
        eg_pvalue: float,
        johansen_cointegrated: bool,
        adf_stationary: bool,
        adf_pvalue: float,
        half_life: float,
        hurst: float,
        current_zscore: float,
        regime_stability: float
    ) -> float:
        """Calculate overall quality score (0-100)."""
        
        # Correlation score (15%)
        if 0.80 <= pearson_corr <= 0.92:
            corr_score = 100
        elif pearson_corr >= 0.70:
            corr_score = 60 + (pearson_corr - 0.70) * 400
        else:
            corr_score = pearson_corr * 85
        corr_score = corr_score * 0.7 + corr_stability * 100 * 0.3
        
        # Cointegration score (30%)
        if eg_cointegrated and johansen_cointegrated:
            coint_score = 100
        elif eg_cointegrated:
            coint_score = 85 - eg_pvalue * 100
        elif johansen_cointegrated:
            coint_score = 80
        else:
            coint_score = max(0, 40 - eg_pvalue * 50)
        
        # Stationarity score (20%)
        if adf_stationary:
            stat_score = 100 - adf_pvalue * 100
        else:
            stat_score = max(0, 40 - adf_pvalue * 50)
        
        # Half-life score (20%)
        opt_min, opt_max = self.optimal_half_life_range
        if opt_min <= half_life <= opt_max:
            hl_score = 100
        elif half_life < opt_min:
            hl_score = max(50, 100 - (opt_min - half_life) * 5)
        elif half_life <= self.max_half_life:
            hl_score = max(40, 100 - (half_life - opt_max) * 0.3)
        else:
            hl_score = max(20, 40 - (half_life - self.max_half_life) * 0.05)
        
        # Hurst penalty
        if hurst < 0.45:
            hl_score = min(100, hl_score + 10)
        elif hurst > 0.55:
            hl_score = max(0, hl_score - 20)
        
        # Tradability score (10%)
        abs_z = abs(current_zscore)
        if abs_z >= 2.5:
            trade_score = 100
        elif abs_z >= 2.0:
            trade_score = 80
        elif abs_z >= 1.5:
            trade_score = 50
        else:
            trade_score = abs_z * 33
        
        # Regime stability (5%)
        regime_score = regime_stability
        
        # Weighted total
        total = (
            0.15 * corr_score +
            0.30 * coint_score +
            0.20 * stat_score +
            0.20 * hl_score +
            0.10 * trade_score +
            0.05 * regime_score
        )
        
        return min(100, max(0, total))
    
    def generate_report(self, result: PipelineResult) -> str:
        """Generate institutional-grade report."""
        lines = []
        lines.append("=" * 80)
        lines.append("INSTITUTIONAL PAIR SELECTION PIPELINE REPORT")
        lines.append("=" * 80)
        
        lines.append(f"\nTimestamp: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Timeframe: {result.timeframe}")
        lines.append(f"Symbols analyzed: {result.symbols_analyzed}")
        lines.append(f"Pairs analyzed: {result.pairs_analyzed}")
        
        lines.append("\n" + "-" * 80)
        lines.append("PIPELINE FUNNEL")
        lines.append("-" * 80)
        lines.append(f"  Stage 1 - Correlation filter:   {result.passed_correlation:>4} pairs ({result.passed_correlation/result.pairs_analyzed*100:.1f}%)")
        lines.append(f"  Stage 2 - Cointegration test:   {result.passed_cointegration:>4} pairs ({result.passed_cointegration/result.pairs_analyzed*100:.1f}%)")
        lines.append(f"  Stage 3 - Stationarity test:    {result.passed_stationarity:>4} pairs ({result.passed_stationarity/result.pairs_analyzed*100:.1f}%)")
        lines.append(f"  Stage 4 - Half-life validation: {result.passed_half_life:>4} pairs ({result.passed_half_life/result.pairs_analyzed*100:.1f}%)")
        lines.append(f"  FINAL CANDIDATES:               {result.final_candidates:>4} pairs ({result.final_candidates/result.pairs_analyzed*100:.1f}%)")
        
        lines.append("\n" + "-" * 80)
        lines.append("SELECTED PAIRS (Ranked by Score)")
        lines.append("-" * 80)
        
        for i, p in enumerate(result.selected_pairs[:10], 1):
            signal = ""
            if p.current_zscore < -2.0:
                signal = " 🟢 LONG SIGNAL"
            elif p.current_zscore > 2.0:
                signal = " 🔴 SHORT SIGNAL"
            
            tradeable = "✓" if p.is_tradeable else "✗"
            
            lines.append(f"\n#{i} {p.pair[0]}/{p.pair[1]} - Score: {p.overall_score:.1f}/100 [{tradeable}]{signal}")
            
            lines.append(f"    CORRELATION:")
            lines.append(f"      Pearson: {p.pearson_correlation:.3f} | Spearman: {p.spearman_correlation:.3f}")
            lines.append(f"      Rolling mean: {p.rolling_corr_mean:.3f} ± {p.rolling_corr_std:.3f}")
            lines.append(f"      Stability: {p.correlation_stability*100:.0f}%")
            
            lines.append(f"    COINTEGRATION:")
            eg_status = "✓" if p.eg_is_cointegrated else "✗"
            joh_status = "✓" if p.johansen_is_cointegrated else "✗"
            lines.append(f"      Engle-Granger: {eg_status} (p={p.eg_coint_pvalue:.4f})")
            lines.append(f"      Johansen Trace: {joh_status} (stat={p.johansen_trace_stat:.2f} vs cv={p.johansen_trace_cv:.2f})")
            
            lines.append(f"    STATIONARITY:")
            adf_status = "✓" if p.adf_is_stationary else "✗"
            kpss_status = "✓" if p.kpss_is_stationary else "✗"
            lines.append(f"      ADF: {adf_status} (stat={p.adf_stat:.2f}, p={p.adf_pvalue:.4f})")
            lines.append(f"      KPSS: {kpss_status} (stat={p.kpss_stat:.4f}, p={p.kpss_pvalue:.4f})")
            
            lines.append(f"    MEAN REVERSION:")
            lines.append(f"      Half-life: {p.half_life:.1f} bars ({p.half_life_days:.1f} days)")
            lines.append(f"      Hurst: {p.hurst_exponent:.3f} ({'mean-reverting' if p.hurst_exponent < 0.5 else 'trending'})")
            
            lines.append(f"    SPREAD:")
            lines.append(f"      Hedge ratio: {p.hedge_ratio:.4f} (std: {p.hedge_ratio_std:.4f})")
            lines.append(f"      Current Z-score: {p.current_zscore:+.2f}")
            
            if p.warnings:
                lines.append(f"    ⚠️ WARNINGS: {', '.join(p.warnings)}")
            if p.rejection_reasons:
                lines.append(f"    ❌ ISSUES: {', '.join(p.rejection_reasons)}")
        
        # Summary stats
        lines.append("\n" + "-" * 80)
        lines.append("SUMMARY STATISTICS")
        lines.append("-" * 80)
        lines.append(f"  Average correlation (selected): {result.avg_correlation:.3f}")
        lines.append(f"  Average half-life (selected): {result.avg_half_life:.1f} bars")
        lines.append(f"  Cointegration rate (all pairs): {result.cointegration_rate:.1%}")
        
        lines.append("\n" + "=" * 80)
        
        return "\n".join(lines)
