"""
Correlation Analyzer

Performs rolling correlation analysis to identify and monitor
tradeable pairs for the Pairs Trading System.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

import sys
sys.path.append(str(__file__).rsplit('\\', 3)[0])

from config.settings import Settings, SpreadParameters


logger = logging.getLogger(__name__)


@dataclass
class CorrelationResult:
    """Container for correlation analysis results."""
    pair: Tuple[str, str]
    current_correlation: float
    mean_correlation: float
    std_correlation: float
    min_correlation: float
    max_correlation: float
    is_stable: bool
    stability_score: float  # Higher is better
    rolling_correlations: Optional[pd.Series] = None


class CorrelationAnalyzer:
    """
    Analyzes correlation between pairs of instruments.
    
    Features:
    - Pearson correlation calculation
    - Rolling correlation windows
    - Correlation stability assessment
    - Pair screening based on correlation criteria
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize the Correlation Analyzer.
        
        Args:
            settings: System settings
        """
        self.settings = settings
        self.params = settings.spread_params
    
    def calculate_correlation(
        self,
        series_a: pd.Series,
        series_b: pd.Series
    ) -> float:
        """
        Calculate Pearson correlation between two series.
        
        Args:
            series_a: First price series
            series_b: Second price series
            
        Returns:
            Pearson correlation coefficient
        """
        # Use returns instead of prices for correlation
        returns_a = series_a.pct_change().dropna()
        returns_b = series_b.pct_change().dropna()
        
        # Align the series
        aligned = pd.concat([returns_a, returns_b], axis=1).dropna()
        
        if len(aligned) < 2:
            return 0.0
        
        return aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
    
    def calculate_rolling_correlation(
        self,
        series_a: pd.Series,
        series_b: pd.Series,
        window: Optional[int] = None
    ) -> pd.Series:
        """
        Calculate rolling correlation between two series.
        
        Args:
            series_a: First price series
            series_b: Second price series
            window: Rolling window size
            
        Returns:
            Series of rolling correlations
        """
        window = window or self.params.correlation_window
        
        # Use returns
        returns_a = series_a.pct_change()
        returns_b = series_b.pct_change()
        
        rolling_corr = returns_a.rolling(window=window).corr(returns_b)
        
        return rolling_corr.dropna()
    
    def analyze_pair(
        self,
        aligned_prices: pd.DataFrame,
        window: Optional[int] = None
    ) -> CorrelationResult:
        """
        Perform comprehensive correlation analysis on a pair.
        
        Args:
            aligned_prices: DataFrame with aligned prices for both instruments
            window: Rolling window size
            
        Returns:
            CorrelationResult with analysis details
        """
        window = window or self.params.correlation_window
        
        if aligned_prices.empty or len(aligned_prices.columns) < 2:
            raise ValueError("Need at least 2 columns of price data")
        
        col_a = aligned_prices.columns[0]
        col_b = aligned_prices.columns[1]
        
        # Calculate rolling correlation
        rolling_corr = self.calculate_rolling_correlation(
            aligned_prices[col_a],
            aligned_prices[col_b],
            window
        )
        
        if rolling_corr.empty:
            return CorrelationResult(
                pair=(col_a, col_b),
                current_correlation=0.0,
                mean_correlation=0.0,
                std_correlation=1.0,
                min_correlation=-1.0,
                max_correlation=1.0,
                is_stable=False,
                stability_score=0.0
            )
        
        # Calculate statistics
        current_corr = rolling_corr.iloc[-1]
        mean_corr = rolling_corr.mean()
        std_corr = rolling_corr.std()
        min_corr = rolling_corr.min()
        max_corr = rolling_corr.max()
        
        # Stability assessment
        # A stable correlation should:
        # 1. Stay consistently above the minimum threshold
        # 2. Have low standard deviation
        # 3. Not drop below threshold too often
        
        threshold = self.params.min_correlation
        pct_above_threshold = (rolling_corr >= threshold).mean()
        
        # Stability score: weighted combination of factors
        stability_score = (
            0.4 * (mean_corr / 1.0) +  # Mean correlation (normalized)
            0.3 * (1 - std_corr) +      # Low volatility is good
            0.3 * pct_above_threshold    # Percentage above threshold
        )
        
        is_stable = (
            current_corr >= threshold and
            mean_corr >= threshold and
            pct_above_threshold >= 0.8  # At least 80% of time above threshold
        )
        
        return CorrelationResult(
            pair=(col_a, col_b),
            current_correlation=current_corr,
            mean_correlation=mean_corr,
            std_correlation=std_corr,
            min_correlation=min_corr,
            max_correlation=max_corr,
            is_stable=is_stable,
            stability_score=stability_score,
            rolling_correlations=rolling_corr
        )
    
    def screen_pairs(
        self,
        all_pairs_data: Dict[Tuple[str, str], pd.DataFrame],
        min_correlation: Optional[float] = None,
        min_stability_score: float = 0.5
    ) -> List[CorrelationResult]:
        """
        Screen pairs based on correlation criteria.
        
        Args:
            all_pairs_data: Dictionary of aligned prices for each pair
            min_correlation: Minimum required correlation
            min_stability_score: Minimum stability score
            
        Returns:
            List of pairs that pass screening, sorted by stability score
        """
        min_correlation = min_correlation or self.params.min_correlation
        
        valid_pairs = []
        
        for pair, aligned_prices in all_pairs_data.items():
            try:
                result = self.analyze_pair(aligned_prices)
                
                if (result.current_correlation >= min_correlation and
                    result.is_stable and
                    result.stability_score >= min_stability_score):
                    valid_pairs.append(result)
                    logger.info(
                        f"Pair {pair} passed screening: "
                        f"corr={result.current_correlation:.3f}, "
                        f"stability={result.stability_score:.3f}"
                    )
                else:
                    logger.debug(
                        f"Pair {pair} failed screening: "
                        f"corr={result.current_correlation:.3f}, "
                        f"stable={result.is_stable}, "
                        f"stability={result.stability_score:.3f}"
                    )
            except Exception as e:
                logger.error(f"Error analyzing pair {pair}: {e}")
        
        # Sort by stability score (descending)
        valid_pairs.sort(key=lambda x: x.stability_score, reverse=True)
        
        return valid_pairs
    
    def monitor_correlation(
        self,
        result: CorrelationResult,
        new_prices_a: pd.Series,
        new_prices_b: pd.Series
    ) -> CorrelationResult:
        """
        Update correlation analysis with new data.
        
        Args:
            result: Previous correlation result
            new_prices_a: New prices for first instrument
            new_prices_b: New prices for second instrument
            
        Returns:
            Updated CorrelationResult
        """
        # Calculate correlation on new data
        new_corr = self.calculate_correlation(new_prices_a, new_prices_b)
        
        # Update rolling correlations
        if result.rolling_correlations is not None:
            new_rolling = self.calculate_rolling_correlation(
                new_prices_a, new_prices_b
            )
            
            # Combine with existing
            combined = pd.concat([
                result.rolling_correlations,
                new_rolling[~new_rolling.index.isin(result.rolling_correlations.index)]
            ])
            
            # Keep only recent history
            max_history = self.params.correlation_window * 2
            if len(combined) > max_history:
                combined = combined.iloc[-max_history:]
        else:
            combined = None
        
        # Recalculate metrics
        return CorrelationResult(
            pair=result.pair,
            current_correlation=new_corr,
            mean_correlation=combined.mean() if combined is not None else new_corr,
            std_correlation=combined.std() if combined is not None else 0.0,
            min_correlation=combined.min() if combined is not None else new_corr,
            max_correlation=combined.max() if combined is not None else new_corr,
            is_stable=new_corr >= self.params.min_correlation,
            stability_score=result.stability_score,  # Would need full recalc
            rolling_correlations=combined
        )
    
    def correlation_breakdown_alert(
        self,
        result: CorrelationResult,
        threshold: Optional[float] = None
    ) -> bool:
        """
        Check if correlation has broken down below threshold.
        
        Args:
            result: Correlation analysis result
            threshold: Alert threshold (default: min_correlation - 0.1)
            
        Returns:
            True if correlation breakdown detected
        """
        threshold = threshold or (self.params.min_correlation - 0.1)
        
        if result.current_correlation < threshold:
            logger.warning(
                f"Correlation breakdown alert for {result.pair}: "
                f"{result.current_correlation:.3f} < {threshold:.3f}"
            )
            return True
        
        return False
    
    def get_correlation_matrix(
        self,
        prices_dict: Dict[str, pd.Series]
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix for multiple instruments.
        
        Args:
            prices_dict: Dictionary of price series by instrument
            
        Returns:
            Correlation matrix DataFrame
        """
        # Convert to returns
        returns = pd.DataFrame({
            name: series.pct_change() 
            for name, series in prices_dict.items()
        }).dropna()
        
        return returns.corr()
    
    def find_best_pairs(
        self,
        prices_dict: Dict[str, pd.Series],
        top_n: int = 10,
        min_correlation: Optional[float] = None
    ) -> List[Tuple[Tuple[str, str], float]]:
        """
        Find the best correlated pairs from a set of instruments.
        
        Args:
            prices_dict: Dictionary of price series by instrument
            top_n: Number of top pairs to return
            min_correlation: Minimum correlation threshold
            
        Returns:
            List of (pair, correlation) tuples
        """
        min_correlation = min_correlation or self.params.min_correlation
        
        corr_matrix = self.get_correlation_matrix(prices_dict)
        
        pairs = []
        instruments = list(corr_matrix.columns)
        
        for i, inst_a in enumerate(instruments):
            for j, inst_b in enumerate(instruments):
                if j <= i:  # Skip diagonal and duplicates
                    continue
                
                corr = corr_matrix.loc[inst_a, inst_b]
                
                if corr >= min_correlation:
                    pairs.append(((inst_a, inst_b), corr))
        
        # Sort by correlation (descending)
        pairs.sort(key=lambda x: x[1], reverse=True)
        
        return pairs[:top_n]
