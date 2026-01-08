"""
Adaptive Trading Parameters.

Dynamically adjusts trading parameters based on pair characteristics.
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class AdaptiveParameters:
    """Trading parameters adapted to pair characteristics."""
    
    # Entry/Exit thresholds
    entry_zscore: float
    exit_zscore: float
    stop_loss_zscore: float
    
    # Position management
    max_holding_bars: int
    time_decay_start: int  # Bars after which to start reducing position
    
    # Risk parameters
    position_size_factor: float  # Relative to base position
    
    # Re-estimation
    hedge_ratio_update_frequency: int
    
    # Description
    regime: str
    
    
class ParameterAdapter:
    """
    Adapts trading parameters based on pair statistics.
    
    Key factors:
    - Half-life: Longer HL = more patient exit, wider stops
    - Hurst: Lower = stronger mean reversion
    - Cointegration strength: Stronger = tighter parameters
    """
    
    @staticmethod
    def adapt(
        half_life: float,
        hurst: float,
        coint_pvalue: float,
        volatility: float = 0.15
    ) -> AdaptiveParameters:
        """
        Generate adaptive parameters.
        
        Args:
            half_life: Spread half-life in bars
            hurst: Hurst exponent
            coint_pvalue: Cointegration p-value
            volatility: Annualized volatility
            
        Returns:
            AdaptiveParameters
        """
        # Determine regime
        if half_life < 30:
            regime = "fast_mean_reversion"
        elif half_life < 100:
            regime = "moderate_mean_reversion"
        elif half_life < 300:
            regime = "slow_mean_reversion"
        else:
            regime = "very_slow_mean_reversion"
        
        # Base parameters by regime
        if regime == "fast_mean_reversion":
            entry_z = 2.0
            exit_z = 0.2
            stop_z = 3.0
            max_holding = int(half_life * 3)
            time_decay_start = int(half_life * 1.5)
            
        elif regime == "moderate_mean_reversion":
            entry_z = 2.0
            exit_z = 0.3
            stop_z = 3.5
            max_holding = int(half_life * 2.5)
            time_decay_start = int(half_life * 1.2)
            
        elif regime == "slow_mean_reversion":
            entry_z = 2.2
            exit_z = 0.5
            stop_z = 4.0
            max_holding = int(half_life * 2)
            time_decay_start = int(half_life)
            
        else:  # very_slow
            entry_z = 2.5
            exit_z = 0.8
            stop_z = 4.5
            max_holding = int(min(half_life * 1.5, 500))
            time_decay_start = int(half_life * 0.8)
        
        # Adjust for Hurst exponent
        if hurst < 0.4:  # Strong mean reversion
            entry_z -= 0.2
            exit_z -= 0.1
        elif hurst > 0.5:  # Trending tendency
            entry_z += 0.3
            stop_z -= 0.5
        
        # Adjust for cointegration strength
        if coint_pvalue < 0.01:  # Very strong cointegration
            exit_z -= 0.1
        elif coint_pvalue > 0.05:  # Weaker cointegration
            stop_z -= 0.3
            entry_z += 0.2
        
        # Position size factor based on regime reliability
        if regime == "fast_mean_reversion" and coint_pvalue < 0.05:
            size_factor = 1.0
        elif regime == "moderate_mean_reversion" and coint_pvalue < 0.05:
            size_factor = 0.8
        elif coint_pvalue < 0.10:
            size_factor = 0.6
        else:
            size_factor = 0.4
        
        # Hedge ratio update frequency
        if half_life < 50:
            hr_update = 20
        elif half_life < 150:
            hr_update = 40
        else:
            hr_update = 60
        
        return AdaptiveParameters(
            entry_zscore=round(entry_z, 1),
            exit_zscore=round(exit_z, 1),
            stop_loss_zscore=round(stop_z, 1),
            max_holding_bars=max_holding,
            time_decay_start=time_decay_start,
            position_size_factor=round(size_factor, 2),
            hedge_ratio_update_frequency=hr_update,
            regime=regime
        )
    
    @staticmethod
    def get_recommended_timeframe(half_life_h1: float) -> str:
        """
        Recommend optimal timeframe based on H1 half-life.
        
        Args:
            half_life_h1: Half-life measured on H1 data
            
        Returns:
            Recommended timeframe string
        """
        if half_life_h1 < 20:
            return "M15"  # Fast mean reversion - use lower TF
        elif half_life_h1 < 50:
            return "H1"   # Moderate - H1 is good
        elif half_life_h1 < 150:
            return "H4"   # Slower - use H4 for better signals
        else:
            return "D1"   # Very slow - use daily
    
    @staticmethod
    def estimate_expected_trade_duration(half_life: float) -> Tuple[float, float]:
        """
        Estimate expected trade duration based on half-life.
        
        Returns:
            (min_bars, max_bars) tuple
        """
        # Typical trade lasts 0.5x to 2x half-life
        return (half_life * 0.5, half_life * 2.0)
    
    @staticmethod
    def estimate_trades_per_year(half_life: float, timeframe: str) -> float:
        """
        Estimate number of trades per year.
        
        Assumes entries at |Z| >= 2 occur roughly 4.5% of the time
        in a normal distribution.
        """
        bars_per_year = {
            'M15': 252 * 24 * 4,
            'H1': 252 * 24,
            'H4': 252 * 6,
            'D1': 252
        }
        
        total_bars = bars_per_year.get(timeframe, 6048)
        
        # Average trade duration
        avg_duration = half_life * 1.2
        
        # Entry opportunities (rough estimate)
        entry_frequency = 0.10  # ~10% of time Z is in tradeable range
        
        # Maximum trades limited by duration
        max_trades = total_bars / avg_duration
        
        # Actual trades limited by entry opportunities
        estimated_trades = min(max_trades, total_bars * entry_frequency / avg_duration)
        
        return estimated_trades


def format_parameters_report(params: AdaptiveParameters, pair: str = "") -> str:
    """Generate formatted report for adaptive parameters."""
    lines = []
    lines.append(f"{'='*50}")
    lines.append(f"ADAPTIVE PARAMETERS{f' - {pair}' if pair else ''}")
    lines.append(f"{'='*50}")
    lines.append(f"\nRegime: {params.regime.upper()}")
    lines.append(f"\nEntry/Exit Thresholds:")
    lines.append(f"  Entry Z-score:     ±{params.entry_zscore}")
    lines.append(f"  Exit Z-score:      ±{params.exit_zscore}")
    lines.append(f"  Stop-loss Z-score: ±{params.stop_loss_zscore}")
    lines.append(f"\nPosition Management:")
    lines.append(f"  Max holding:       {params.max_holding_bars} bars")
    lines.append(f"  Time decay start:  {params.time_decay_start} bars")
    lines.append(f"  Size factor:       {params.position_size_factor:.0%}")
    lines.append(f"\nRe-estimation:")
    lines.append(f"  Hedge ratio update: every {params.hedge_ratio_update_frequency} bars")
    lines.append(f"{'='*50}")
    
    return "\n".join(lines)
