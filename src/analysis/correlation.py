"""
Correlation Analysis Module.

Implements Pearson correlation with rolling windows and stability analysis.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import logging


logger = logging.getLogger(__name__)


@dataclass
class CorrelationResult:
    """Results from correlation analysis."""
    current_correlation: float
    mean_correlation: float
    min_correlation: float
    max_correlation: float
    std_correlation: float
    stability_score: float  # 0-1, higher = more stable
    rolling_correlation: Optional[pd.Series] = None
    is_stable: bool = True
    breakdown_detected: bool = False


class CorrelationAnalyzer:
    """
    Analyzes correlation between price series.
    
    Features:
    - Rolling correlation calculation
    - Stability assessment
    - Structural break detection
    """
    
    def __init__(
        self,
        window: int = 60,
        stability_threshold: float = 0.15,
        min_correlation: float = 0.70
    ):
        """
        Initialize analyzer.
        
        Args:
            window: Rolling window for correlation
            stability_threshold: Max std for "stable" correlation
            min_correlation: Minimum acceptable correlation
        """
        self.window = window
        self.stability_threshold = stability_threshold
        self.min_correlation = min_correlation
    
    def calculate_correlation(
        self,
        price_a: pd.Series,
        price_b: pd.Series
    ) -> float:
        """
        Calculate Pearson correlation coefficient.
        
        Args:
            price_a: First price series
            price_b: Second price series
            
        Returns:
            Correlation coefficient [-1, 1]
        """
        # Use returns for correlation (more stationary)
        returns_a = price_a.pct_change().dropna()
        returns_b = price_b.pct_change().dropna()
        
        # Align series
        common_idx = returns_a.index.intersection(returns_b.index)
        returns_a = returns_a.loc[common_idx]
        returns_b = returns_b.loc[common_idx]
        
        if len(returns_a) < 30:
            return 0.0
        
        return returns_a.corr(returns_b)
    
    def calculate_rolling_correlation(
        self,
        price_a: pd.Series,
        price_b: pd.Series,
        window: Optional[int] = None
    ) -> pd.Series:
        """
        Calculate rolling correlation.
        
        Args:
            price_a: First price series
            price_b: Second price series
            window: Rolling window (uses default if None)
            
        Returns:
            Series of rolling correlations
        """
        window = window or self.window
        
        # Use returns
        returns_a = price_a.pct_change()
        returns_b = price_b.pct_change()
        
        # Align
        common_idx = returns_a.index.intersection(returns_b.index)
        returns_a = returns_a.loc[common_idx]
        returns_b = returns_b.loc[common_idx]
        
        # Rolling correlation
        rolling_corr = returns_a.rolling(window=window).corr(returns_b)
        
        return rolling_corr
    
    def analyze_pair(
        self,
        price_a: pd.Series,
        price_b: pd.Series
    ) -> CorrelationResult:
        """
        Comprehensive correlation analysis.
        
        Args:
            price_a: First price series
            price_b: Second price series
            
        Returns:
            CorrelationResult with all metrics
        """
        # Calculate rolling correlation
        rolling_corr = self.calculate_rolling_correlation(price_a, price_b)
        
        # Remove NaN
        rolling_corr_clean = rolling_corr.dropna()
        
        if len(rolling_corr_clean) < 30:
            return CorrelationResult(
                current_correlation=0.0,
                mean_correlation=0.0,
                min_correlation=0.0,
                max_correlation=0.0,
                std_correlation=1.0,
                stability_score=0.0,
                is_stable=False,
                breakdown_detected=True
            )
        
        # Statistics
        current_corr = rolling_corr_clean.iloc[-1]
        mean_corr = rolling_corr_clean.mean()
        min_corr = rolling_corr_clean.min()
        max_corr = rolling_corr_clean.max()
        std_corr = rolling_corr_clean.std()
        
        # Stability score (inverse of coefficient of variation)
        if mean_corr != 0:
            cv = abs(std_corr / mean_corr)
            stability_score = max(0, 1 - cv)
        else:
            stability_score = 0.0
        
        # Check for structural breaks
        is_stable = std_corr < self.stability_threshold
        
        # Detect recent breakdown
        recent_corr = rolling_corr_clean.tail(20)
        breakdown_detected = (
            recent_corr.min() < self.min_correlation or
            (recent_corr.iloc[-1] - recent_corr.iloc[0]) < -0.2
        )
        
        return CorrelationResult(
            current_correlation=current_corr,
            mean_correlation=mean_corr,
            min_correlation=min_corr,
            max_correlation=max_corr,
            std_correlation=std_corr,
            stability_score=stability_score,
            rolling_correlation=rolling_corr,
            is_stable=is_stable,
            breakdown_detected=breakdown_detected
        )
    
    def is_pair_valid(
        self,
        price_a: pd.Series,
        price_b: pd.Series
    ) -> Tuple[bool, str]:
        """
        Check if pair meets correlation criteria.
        
        Args:
            price_a: First price series
            price_b: Second price series
            
        Returns:
            (is_valid, reason) tuple
        """
        result = self.analyze_pair(price_a, price_b)
        
        if result.current_correlation < self.min_correlation:
            return False, f"Correlation too low: {result.current_correlation:.3f}"
        
        if result.breakdown_detected:
            return False, "Correlation breakdown detected"
        
        if not result.is_stable:
            return False, f"Correlation unstable (std={result.std_correlation:.3f})"
        
        return True, "Pair meets correlation criteria"
    
    def get_correlation_regime(
        self,
        rolling_corr: pd.Series
    ) -> str:
        """
        Determine current correlation regime.
        
        Args:
            rolling_corr: Rolling correlation series
            
        Returns:
            Regime description string
        """
        clean = rolling_corr.dropna()
        
        if len(clean) < 20:
            return "insufficient_data"
        
        current = clean.iloc[-1]
        recent_trend = clean.iloc[-20:].mean() - clean.iloc[-40:-20].mean() if len(clean) >= 40 else 0
        
        if current >= 0.9:
            regime = "very_high"
        elif current >= 0.8:
            regime = "high"
        elif current >= 0.7:
            regime = "moderate"
        elif current >= 0.5:
            regime = "low"
        else:
            regime = "very_low"
        
        if recent_trend > 0.05:
            regime += "_increasing"
        elif recent_trend < -0.05:
            regime += "_decreasing"
        else:
            regime += "_stable"
        
        return regime
