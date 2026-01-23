"""
Time-Series Momentum System - Signal Engine
Trend signals and volatility-targeted position sizing.

Signal Logic:
- Each asset evaluated INDEPENDENTLY (time-series, not cross-sectional)
- LONG if Close > SMA(252)
- CASH if Close <= SMA(252)
- NO SHORTS allowed

Volatility Targeting:
- Raw weight = TargetVol / AssetVol
- Normalize if sum > 1.0 (no leverage)
- Cap each asset at 30%
- Remainder in cash
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple

from config import CONFIG


class SanityCheckError(Exception):
    """Sanity check failed - abort execution."""
    pass


@dataclass
class TSMOMSignal:
    """Signal for a single rebalance date."""
    date: pd.Timestamp
    trend_signals: Dict[str, bool]      # symbol -> LONG (True) or CASH (False)
    asset_volatilities: Dict[str, float]  # 20-day realized vol per asset
    raw_weights: Dict[str, float]       # Before normalization
    target_weights: Dict[str, float]    # Final weights after capping
    cash_weight: float                  # Remainder in cash
    total_weight_check: float           # For validation
    
    def __repr__(self):
        active = [s for s, w in self.target_weights.items() if w > 0.01]
        return f"TSMOMSignal({self.date.date()}, active={len(active)}, cash={self.cash_weight:.1%})"


class SignalEngine:
    """
    Time-Series Momentum signal generation.
    
    Each asset is evaluated independently based on its own trend.
    Position sizes are inverse-volatility weighted.
    """
    
    def __init__(self):
        self.trend_lookback = CONFIG.TREND_LOOKBACK
        self.vol_lookback = CONFIG.VOL_LOOKBACK
        self.target_vol = CONFIG.TARGET_VOL
        self.max_weight = CONFIG.MAX_WEIGHT_PER_ASSET
    
    def calculate_sma(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Calculate SMA(252) for trend signal."""
        return prices.rolling(window=self.trend_lookback, min_periods=self.trend_lookback).mean()
    
    def calculate_realized_volatility(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate 20-day realized volatility (annualized).
        
        Using standard deviation of log returns.
        """
        log_returns = np.log(prices / prices.shift(1))
        vol = log_returns.rolling(window=self.vol_lookback, min_periods=self.vol_lookback).std()
        # Annualize
        vol_annualized = vol * np.sqrt(252)
        return vol_annualized
    
    def get_trend_signals(
        self,
        prices: pd.DataFrame,
        sma: pd.DataFrame,
        date: pd.Timestamp,
    ) -> Dict[str, bool]:
        """
        Generate trend signal for each asset.
        
        LONG if Close > SMA(252), else CASH.
        Each asset is independent (time-series momentum).
        """
        signals = {}
        
        for symbol in prices.columns:
            try:
                price = prices.loc[date, symbol]
                sma_val = sma.loc[date, symbol]
                
                if pd.isna(price) or pd.isna(sma_val):
                    signals[symbol] = False  # No valid data = CASH
                else:
                    signals[symbol] = price > sma_val  # LONG if above SMA
            except KeyError:
                signals[symbol] = False
        
        return signals
    
    def calculate_volatility_targeted_weights(
        self,
        trend_signals: Dict[str, bool],
        asset_vols: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, float], float]:
        """
        Calculate volatility-targeted position weights.
        
        Steps:
        1. For each LONG asset: raw_weight = target_vol / asset_vol
        2. If sum > 1.0: normalize to prevent leverage
        3. Cap each asset at max_weight (30%)
        4. Remainder goes to cash
        
        Returns:
            (raw_weights, target_weights, cash_weight)
        """
        raw_weights = {}
        
        # Step 1: Calculate raw inverse-volatility weights for LONG assets
        for symbol, is_long in trend_signals.items():
            if is_long:
                vol = asset_vols.get(symbol)
                
                if vol is None or pd.isna(vol) or vol <= 0:
                    vol = 0.20  # Default 20% if invalid
                
                # Floor volatility to prevent extreme weights
                vol = max(vol, 0.05)  # Min 5% vol
                
                raw_weights[symbol] = self.target_vol / vol
            else:
                raw_weights[symbol] = 0.0
        
        # Step 2: Normalize if sum > 1.0 (NO LEVERAGE)
        total_raw = sum(raw_weights.values())
        
        if total_raw > 1.0:
            # Scale down proportionally
            normalized = {s: w / total_raw for s, w in raw_weights.items()}
        elif total_raw > 0:
            # Keep weights as-is, remainder to cash
            normalized = raw_weights.copy()
        else:
            # No active positions
            normalized = {s: 0.0 for s in raw_weights}
        
        # Step 3: Cap each asset at max_weight
        target_weights = {}
        
        for symbol, weight in normalized.items():
            target_weights[symbol] = min(weight, self.max_weight)
        
        # Step 4: Calculate cash weight
        total_invested = sum(target_weights.values())
        cash_weight = 1.0 - total_invested
        
        # Sanity check
        if total_invested > 1.001:
            raise SanityCheckError(
                f"LEVERAGE DETECTED: weights sum to {total_invested:.4f}"
            )
        
        if cash_weight < -0.001:
            raise SanityCheckError(
                f"NEGATIVE CASH: {cash_weight:.4f}"
            )
        
        return raw_weights, target_weights, max(0.0, cash_weight)
    
    def generate_signal(
        self,
        prices: pd.DataFrame,
        sma: pd.DataFrame,
        vol: pd.DataFrame,
        date: pd.Timestamp,
    ) -> TSMOMSignal:
        """Generate signal for a single rebalance date."""
        # Get trend signals (LONG or CASH for each asset)
        trend_signals = self.get_trend_signals(prices, sma, date)
        
        # Get asset volatilities
        asset_vols = {}
        for symbol in prices.columns:
            try:
                v = vol.loc[date, symbol]
                asset_vols[symbol] = v if not pd.isna(v) else 0.20
            except KeyError:
                asset_vols[symbol] = 0.20
        
        # Calculate volatility-targeted weights
        raw_weights, target_weights, cash_weight = self.calculate_volatility_targeted_weights(
            trend_signals, asset_vols
        )
        
        # Validation
        total_check = sum(target_weights.values()) + cash_weight
        
        return TSMOMSignal(
            date=date,
            trend_signals=trend_signals,
            asset_volatilities=asset_vols,
            raw_weights=raw_weights,
            target_weights=target_weights,
            cash_weight=cash_weight,
            total_weight_check=total_check,
        )
    
    def generate_all_signals(
        self,
        prices: pd.DataFrame,
        rebalance_dates: List[pd.Timestamp],
    ) -> List[TSMOMSignal]:
        """Generate signals for all rebalance dates."""
        # Pre-calculate indicators (no lookahead - using historical data only)
        sma = self.calculate_sma(prices)
        vol = self.calculate_realized_volatility(prices)
        
        # Skip warmup period
        warmup = max(self.trend_lookback, self.vol_lookback)
        valid_dates = [d for d in rebalance_dates if d >= prices.index[warmup]]
        
        signals = []
        for date in valid_dates:
            try:
                sig = self.generate_signal(prices, sma, vol, date)
                
                # Validate signal
                if abs(sig.total_weight_check - 1.0) > 0.01:
                    raise SanityCheckError(
                        f"Weights don't sum to 1.0 on {date}: {sig.total_weight_check:.4f}"
                    )
                
                signals.append(sig)
            except Exception as e:
                raise SanityCheckError(f"Signal generation failed on {date}: {e}")
        
        return signals


def validate_signal(signal: TSMOMSignal) -> Tuple[bool, List[str]]:
    """Validate a signal before execution."""
    issues = []
    
    # Weights must sum to ~1.0
    total = sum(signal.target_weights.values()) + signal.cash_weight
    if abs(total - 1.0) > 0.01:
        issues.append(f"Weights sum to {total:.4f}, not 1.0")
    
    # No negative weights
    for sym, w in signal.target_weights.items():
        if w < -0.001:
            issues.append(f"Negative weight for {sym}: {w:.4f}")
    
    # No weight exceeds cap
    for sym, w in signal.target_weights.items():
        if w > CONFIG.MAX_WEIGHT_PER_ASSET + 0.001:
            issues.append(f"{sym} weight {w:.2%} exceeds {CONFIG.MAX_WEIGHT_PER_ASSET:.0%} cap")
    
    # Cash not negative
    if signal.cash_weight < -0.001:
        issues.append(f"Negative cash weight: {signal.cash_weight:.4f}")
    
    return len(issues) == 0, issues


if __name__ == "__main__":
    from data_loader import DataLoader
    
    print("Signal Engine Test")
    print("=" * 60)
    
    loader = DataLoader()
    prices = loader.load_universe()
    rebalance_dates = loader.get_monthly_rebalance_dates(prices)
    
    engine = SignalEngine()
    signals = engine.generate_all_signals(prices, rebalance_dates)
    
    print(f"\nGenerated {len(signals)} signals")
    
    # Validate all
    all_valid = True
    for sig in signals:
        valid, issues = validate_signal(sig)
        if not valid:
            print(f"Invalid signal on {sig.date}: {issues}")
            all_valid = False
    
    print(f"All signals valid: {all_valid}")
    
    # Summary statistics
    print("\nAsset selection frequency:")
    counts = {}
    for sig in signals:
        for sym, w in sig.target_weights.items():
            if w > 0.01:
                counts[sym] = counts.get(sym, 0) + 1
    
    for sym, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        pct = cnt / len(signals) * 100
        print(f"  {sym}: {cnt}/{len(signals)} ({pct:.1f}%)")
    
    # Average cash weight
    avg_cash = np.mean([s.cash_weight for s in signals])
    print(f"\nAverage cash weight: {avg_cash:.1%}")
    
    # Show last 3 signals
    print("\nLast 3 signals:")
    for sig in signals[-3:]:
        print(f"\n{sig.date.date()}:")
        for sym, w in sig.target_weights.items():
            if w > 0.01:
                trend = "LONG" if sig.trend_signals[sym] else "CASH"
                vol = sig.asset_volatilities[sym]
                print(f"  {sym}: {w:.1%} (vol={vol:.1%}, trend={trend})")
        print(f"  CASH: {sig.cash_weight:.1%}")
