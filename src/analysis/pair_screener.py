"""
Pair Screening Module.

Analyzes multiple symbols to find the best trading pair candidates.
Optimized for Forex pairs trading.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Tuple
from itertools import combinations
import logging

from statsmodels.tsa.stattools import coint, adfuller
from sklearn.linear_model import LinearRegression


logger = logging.getLogger(__name__)


@dataclass
class PairScore:
    """Scoring metrics for a trading pair."""
    pair: Tuple[str, str]
    
    # Statistical metrics
    price_correlation: float       # Correlation of prices (levels)
    returns_correlation: float     # Correlation of returns
    is_cointegrated: bool
    coint_pvalue: float
    half_life: float
    hurst_exponent: float
    adf_pvalue: float
    hedge_ratio: float
    
    # Current state
    current_zscore: float
    spread_std: float
    
    # Quality scores (0-100)
    correlation_score: float
    cointegration_score: float
    mean_reversion_score: float
    tradability_score: float
    
    # Overall score
    total_score: float
    rank: int = 0
    
    # Flags
    is_tradeable: bool = True
    rejection_reasons: List[str] = field(default_factory=list)


@dataclass
class ScreeningResult:
    """Complete screening results."""
    timestamp: datetime
    symbols_analyzed: int
    pairs_analyzed: int
    pairs_passed: int
    
    # All pairs sorted by score
    all_pairs: List[PairScore]
    
    # Top candidates
    top_pairs: List[PairScore]
    
    # Statistics
    avg_correlation: float
    cointegration_rate: float


class PairScreener:
    """
    Screens multiple symbols to find optimal trading pairs.
    
    Criteria for Forex H1:
    1. Price correlation > 0.60 
    2. Cointegration (Engle-Granger p < 0.10)
    3. Half-life reasonable for trading
    """
    
    def __init__(
        self,
        min_correlation: float = 0.60,      # Price correlation threshold
        max_correlation: float = 0.995,     # Too high = same asset
        max_half_life: int = 500,           # Maximum half-life
        min_half_life: int = 3,             # Minimum for trading
        coint_pvalue: float = 0.10,         # Cointegration p-value
        regression_window: int = 120,
        zscore_window: int = 60
    ):
        """
        Initialize screener with relaxed defaults for forex.
        """
        self.min_correlation = min_correlation
        self.max_correlation = max_correlation
        self.max_half_life = max_half_life
        self.min_half_life = min_half_life
        self.coint_pvalue = coint_pvalue
        self.regression_window = regression_window
        self.zscore_window = zscore_window
    
    def screen_pairs(
        self,
        price_data: Dict[str, pd.Series],
        top_n: int = 10
    ) -> ScreeningResult:
        """
        Screen all possible pairs from given symbols.
        
        Args:
            price_data: Dictionary {symbol: close_prices}
            top_n: Number of top pairs to highlight
            
        Returns:
            ScreeningResult with ranked pairs
        """
        symbols = list(price_data.keys())
        all_pairs = list(combinations(symbols, 2))
        
        logger.info(f"Screening {len(all_pairs)} pairs from {len(symbols)} symbols...")
        
        pair_scores = []
        
        for i, (sym_a, sym_b) in enumerate(all_pairs):
            if (i + 1) % 10 == 0:
                logger.info(f"  Analyzing pair {i + 1}/{len(all_pairs)}...")
            
            try:
                score = self._analyze_pair(
                    pair=(sym_a, sym_b),
                    price_a=price_data[sym_a],
                    price_b=price_data[sym_b]
                )
                if score:
                    pair_scores.append(score)
            except Exception as e:
                logger.warning(f"Failed to analyze {sym_a}/{sym_b}: {e}")
        
        # Sort by total score
        pair_scores.sort(key=lambda x: x.total_score, reverse=True)
        
        # Assign ranks
        for i, score in enumerate(pair_scores):
            score.rank = i + 1
        
        # Filter tradeable pairs
        tradeable = [p for p in pair_scores if p.is_tradeable]
        
        # Calculate statistics (on all pairs)
        if pair_scores:
            avg_corr = float(np.mean([p.price_correlation for p in pair_scores]))
            coint_count = sum(1 for p in pair_scores if p.is_cointegrated)
            coint_rate = coint_count / len(pair_scores)
        else:
            avg_corr = 0
            coint_rate = 0
        
        return ScreeningResult(
            timestamp=datetime.now(),
            symbols_analyzed=len(symbols),
            pairs_analyzed=len(all_pairs),
            pairs_passed=len(tradeable),
            all_pairs=pair_scores,
            top_pairs=tradeable[:top_n] if tradeable else pair_scores[:top_n],
            avg_correlation=avg_corr,
            cointegration_rate=coint_rate
        )
    
    def _analyze_pair(
        self,
        pair: Tuple[str, str],
        price_a: pd.Series,
        price_b: pd.Series
    ) -> Optional[PairScore]:
        """Analyze a single pair with optimized calculations."""
        # Align data
        common_idx = price_a.index.intersection(price_b.index)
        price_a = price_a.loc[common_idx].astype(float)
        price_b = price_b.loc[common_idx].astype(float)
        
        if len(price_a) < 500:
            return None
        
        rejection_reasons = []
        
        # 1. Price Correlation (on levels)
        price_correlation = float(price_a.corr(price_b))
        
        # Returns correlation (for reference)
        returns_a = price_a.pct_change().dropna()
        returns_b = price_b.pct_change().dropna()
        returns_correlation = float(returns_a.corr(returns_b))
        
        # 2. Cointegration Test (Engle-Granger)
        try:
            _, coint_pvalue, _ = coint(price_a.values, price_b.values)
            coint_pvalue = float(coint_pvalue)
            is_cointegrated = coint_pvalue < self.coint_pvalue
        except:
            is_cointegrated = False
            coint_pvalue = 1.0
        
        # 3. Hedge Ratio via OLS
        X = price_b.values.reshape(-1, 1)
        y = price_a.values
        model = LinearRegression()
        model.fit(X, y)
        hedge_ratio = float(model.coef_[0])
        
        # 4. Spread and Z-score
        spread = price_a - hedge_ratio * price_b
        spread_mean = spread.rolling(self.zscore_window).mean()
        spread_std_rolling = spread.rolling(self.zscore_window).std()
        zscore = (spread - spread_mean) / spread_std_rolling
        
        current_zscore = float(zscore.iloc[-1]) if not pd.isna(zscore.iloc[-1]) else 0.0
        current_spread_std = float(spread_std_rolling.iloc[-1]) if not pd.isna(spread_std_rolling.iloc[-1]) else float(spread.std())
        
        # 5. ADF Test on spread
        try:
            adf_stat, adf_pvalue, _, _, _, _ = adfuller(spread.dropna().values, maxlag=20)
            adf_pvalue = float(adf_pvalue)
        except:
            adf_pvalue = 1.0
        
        # 6. Half-life calculation (Ornstein-Uhlenbeck)
        spread_clean = spread.dropna()
        spread_lag = spread_clean.shift(1).dropna()
        spread_ret = spread_clean.diff().dropna()
        
        # Align
        common = spread_lag.index.intersection(spread_ret.index)
        spread_lag = spread_lag.loc[common].values.reshape(-1, 1)
        spread_ret = spread_ret.loc[common].values
        
        if len(spread_lag) > 50:
            model_hl = LinearRegression()
            model_hl.fit(spread_lag, spread_ret)
            theta = float(model_hl.coef_[0])
            
            if theta < 0 and theta > -1:
                half_life = -np.log(2) / theta
                half_life = min(half_life, 9999)
            else:
                half_life = 9999.0
        else:
            half_life = 9999.0
        
        half_life = float(half_life)
        
        # 7. Hurst Exponent (simplified)
        hurst = self._calculate_hurst(spread_clean.values)
        hurst = float(hurst)
        
        # Rejection criteria
        is_tradeable = True
        
        # Primary criterion: Cointegration OR low ADF p-value
        if not is_cointegrated and adf_pvalue > 0.10:
            is_tradeable = False
            rejection_reasons.append(f"Not cointegrated (Coint p={coint_pvalue:.3f}, ADF p={adf_pvalue:.3f})")
        
        # Price correlation check
        if price_correlation < self.min_correlation:
            is_tradeable = False
            rejection_reasons.append(f"Low price correlation: {price_correlation:.2f}")
        
        if price_correlation > self.max_correlation:
            is_tradeable = False
            rejection_reasons.append(f"Correlation too high: {price_correlation:.3f}")
        
        # Half-life check
        if half_life > self.max_half_life:
            if not is_cointegrated:
                is_tradeable = False
            rejection_reasons.append(f"Half-life: {half_life:.0f} bars")
        
        # Calculate scores
        correlation_score = self._score_correlation(price_correlation, returns_correlation)
        cointegration_score = self._score_cointegration(is_cointegrated, coint_pvalue, adf_pvalue)
        mean_reversion_score = self._score_mean_reversion(half_life, hurst, is_cointegrated)
        tradability_score = self._score_tradability(current_zscore)
        
        # Total score (weighted)
        total_score = (
            0.20 * correlation_score +
            0.35 * cointegration_score +
            0.30 * mean_reversion_score +
            0.15 * tradability_score
        )
        
        return PairScore(
            pair=pair,
            price_correlation=price_correlation,
            returns_correlation=returns_correlation,
            is_cointegrated=is_cointegrated,
            coint_pvalue=coint_pvalue,
            half_life=half_life,
            hurst_exponent=hurst,
            adf_pvalue=adf_pvalue,
            hedge_ratio=hedge_ratio,
            current_zscore=current_zscore,
            spread_std=current_spread_std,
            correlation_score=float(correlation_score),
            cointegration_score=float(cointegration_score),
            mean_reversion_score=float(mean_reversion_score),
            tradability_score=float(tradability_score),
            total_score=float(total_score),
            is_tradeable=is_tradeable,
            rejection_reasons=rejection_reasons
        )
    
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
        
        # Linear regression in log-log space
        log_lags = np.log([x[0] for x in rs_values])
        log_rs = np.log([x[1] for x in rs_values])
        
        try:
            slope, _ = np.polyfit(log_lags, log_rs, 1)
            return min(max(slope, 0), 1)
        except:
            return 0.5
    
    def _score_correlation(self, price_corr: float, returns_corr: float) -> float:
        """Score based on price correlation."""
        # Optimal: 0.85-0.95
        if 0.85 <= price_corr <= 0.95:
            score = 100
        elif price_corr >= 0.95:
            score = 90 - (price_corr - 0.95) * 200
        elif price_corr >= 0.75:
            score = 70 + (price_corr - 0.75) * 300
        elif price_corr >= 0.60:
            score = 40 + (price_corr - 0.60) * 200
        else:
            score = price_corr * 66
        
        # Bonus for returns correlation
        if returns_corr > 0.5:
            score += 10
        elif returns_corr > 0.3:
            score += 5
        
        return min(100, max(0, score))
    
    def _score_cointegration(self, is_coint: bool, coint_pval: float, adf_pval: float) -> float:
        """Score cointegration strength."""
        if not is_coint and adf_pval > 0.10:
            return max(0, (0.20 - min(coint_pval, adf_pval)) / 0.20 * 40)
        
        # Base score from p-values
        best_pval = min(coint_pval, adf_pval)
        
        if best_pval < 0.01:
            base = 100
        elif best_pval < 0.05:
            base = 85
        elif best_pval < 0.10:
            base = 70
        else:
            base = 50
        
        return min(100, base)
    
    def _score_mean_reversion(self, half_life: float, hurst: float, is_coint: bool) -> float:
        """Score mean reversion quality."""
        # Half-life score (optimal: 20-100 bars for H1)
        if 20 <= half_life <= 100:
            hl_score = 100
        elif 10 <= half_life < 20:
            hl_score = 70 + (half_life - 10) * 3
        elif 100 < half_life <= 200:
            hl_score = 100 - (half_life - 100) * 0.5
        elif 5 <= half_life < 10:
            hl_score = 50 + (half_life - 5) * 4
        elif 200 < half_life <= 500:
            hl_score = 50 - (half_life - 200) * 0.1
        else:
            hl_score = max(20, 30 - (half_life - 500) * 0.01)
        
        # Hurst score
        if hurst < 0.4:
            hurst_score = 100
        elif hurst < 0.45:
            hurst_score = 80
        elif hurst < 0.5:
            hurst_score = 60
        elif hurst < 0.55:
            hurst_score = 40
        else:
            hurst_score = max(0, 30 - (hurst - 0.55) * 100)
        
        # If cointegrated, be more forgiving
        if is_coint:
            hl_score = max(hl_score, 40)
            hurst_score = max(hurst_score, 30)
        
        return hl_score * 0.6 + hurst_score * 0.4
    
    def _score_tradability(self, zscore: float) -> float:
        """Score current entry opportunity."""
        abs_z = abs(zscore) if not np.isnan(zscore) else 0
        
        if abs_z >= 2.5:
            return 100
        elif abs_z >= 2.0:
            return 80 + (abs_z - 2.0) * 40
        elif abs_z >= 1.5:
            return 50 + (abs_z - 1.5) * 60
        elif abs_z >= 1.0:
            return 20 + (abs_z - 1.0) * 60
        else:
            return abs_z * 20
    
    def generate_report(self, result: ScreeningResult) -> str:
        """Generate screening report."""
        lines = []
        lines.append("=" * 70)
        lines.append("PAIR SCREENING REPORT")
        lines.append("=" * 70)
        lines.append(f"\nTimestamp: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Symbols analyzed: {result.symbols_analyzed}")
        lines.append(f"Pairs analyzed: {result.pairs_analyzed}")
        lines.append(f"Pairs passed criteria: {result.pairs_passed}")
        
        # Calculate average correlation only for pairs with corr > 0.5
        high_corr_pairs = [p for p in result.all_pairs if p.price_correlation > 0.5]
        if high_corr_pairs:
            avg_high_corr = np.mean([p.price_correlation for p in high_corr_pairs])
            lines.append(f"Pairs with corr > 0.5: {len(high_corr_pairs)} (avg: {avg_high_corr:.2f})")
        
        lines.append(f"Cointegrated pairs: {sum(1 for p in result.all_pairs if p.is_cointegrated)} ({result.cointegration_rate:.1%})")
        
        lines.append("\n" + "-" * 70)
        lines.append("TOP TRADING PAIRS")
        lines.append("-" * 70)
        
        for i, p in enumerate(result.top_pairs[:10], 1):
            signal = ""
            if p.current_zscore < -2.0:
                signal = " 🟢 LONG SPREAD"
            elif p.current_zscore > 2.0:
                signal = " 🔴 SHORT SPREAD"
            elif abs(p.current_zscore) >= 1.5:
                signal = " ⏳ APPROACHING"
            
            coint_str = "✓" if p.is_cointegrated else "✗"
            tradeable_str = "✓" if p.is_tradeable else "✗"
            
            lines.append(f"\n#{i} {p.pair[0]}/{p.pair[1]} - Score: {p.total_score:.1f}/100 [{tradeable_str}]{signal}")
            lines.append(f"    Price Corr: {p.price_correlation:.3f} | Returns Corr: {p.returns_correlation:.3f}")
            lines.append(f"    Cointegrated: {coint_str} (p={p.coint_pvalue:.4f}) | ADF p={p.adf_pvalue:.4f}")
            lines.append(f"    Half-life: {p.half_life:.1f} bars ({p.half_life/24:.1f} days) | Hurst: {p.hurst_exponent:.3f}")
            lines.append(f"    Hedge ratio: {p.hedge_ratio:.4f}")
            lines.append(f"    Current Z-score: {p.current_zscore:+.2f}")
            lines.append(f"    Scores: Corr={p.correlation_score:.0f} Coint={p.cointegration_score:.0f} MR={p.mean_reversion_score:.0f} Trade={p.tradability_score:.0f}")
            
            if p.rejection_reasons:
                lines.append(f"    Notes: {', '.join(p.rejection_reasons)}")
        
        # Pairs with current signals
        signals = [p for p in result.top_pairs if abs(p.current_zscore) >= 2.0]
        if signals:
            lines.append("\n" + "-" * 70)
            lines.append("PAIRS WITH ACTIVE SIGNALS")
            lines.append("-" * 70)
            
            for p in signals:
                direction = "LONG" if p.current_zscore < 0 else "SHORT"
                lines.append(f"\n  {p.pair[0]}/{p.pair[1]}: {direction} spread (Z={p.current_zscore:+.2f})")
                lines.append(f"    Score: {p.total_score:.1f} | Half-life: {p.half_life:.1f} bars | Cointegrated: {'Yes' if p.is_cointegrated else 'No'}")
        
        # All cointegrated pairs
        cointegrated = [p for p in result.all_pairs if p.is_cointegrated]
        if cointegrated:
            lines.append("\n" + "-" * 70)
            lines.append(f"ALL COINTEGRATED PAIRS ({len(cointegrated)})")
            lines.append("-" * 70)
            
            for p in sorted(cointegrated, key=lambda x: x.coint_pvalue)[:15]:
                status = "✓" if p.is_tradeable else " "
                lines.append(f"  {status} {p.pair[0]}/{p.pair[1]}: Corr={p.price_correlation:.2f}, Coint p={p.coint_pvalue:.4f}, HL={p.half_life:.0f}, Z={p.current_zscore:+.2f}")
        
        lines.append("\n" + "=" * 70)
        
        return "\n".join(lines)
