"""
Diagnostic script to check pair analysis values.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_data(symbol, data_dir):
    """Load data from parquet."""
    pattern = f"{symbol}_H1_*.parquet"
    files = list(data_dir.glob(pattern))
    if not files:
        return None
    filepath = max(files, key=lambda f: f.stat().st_mtime)
    df = pd.read_parquet(filepath)
    if 'time' in df.columns:
        df.set_index('time', inplace=True)
    df.index = pd.to_datetime(df.index)
    return df['close']


def main():
    data_dir = Path("data/historical")
    
    # Load EURUSD and GBPUSD
    eurusd = load_data("EURUSD", data_dir)
    gbpusd = load_data("GBPUSD", data_dir)
    
    print("=" * 60)
    print("DIAGNOSTIC: EURUSD vs GBPUSD")
    print("=" * 60)
    
    # Align
    common = eurusd.index.intersection(gbpusd.index)
    eurusd = eurusd.loc[common]
    gbpusd = gbpusd.loc[common]
    
    print(f"\nData points: {len(eurusd)}")
    print(f"Date range: {eurusd.index[0]} to {eurusd.index[-1]}")
    
    # Raw price correlation
    raw_corr = eurusd.corr(gbpusd)
    print(f"\nRaw price correlation: {raw_corr:.4f}")
    
    # Returns correlation (this is what we should use)
    returns_a = eurusd.pct_change().dropna()
    returns_b = gbpusd.pct_change().dropna()
    returns_corr = returns_a.corr(returns_b)
    print(f"Returns correlation: {returns_corr:.4f}")
    
    # Log returns correlation
    log_returns_a = np.log(eurusd / eurusd.shift(1)).dropna()
    log_returns_b = np.log(gbpusd / gbpusd.shift(1)).dropna()
    log_returns_corr = log_returns_a.corr(log_returns_b)
    print(f"Log returns correlation: {log_returns_corr:.4f}")
    
    # Rolling correlation (60 bars)
    rolling_corr = returns_a.rolling(60).corr(returns_b)
    print(f"\nRolling correlation (60 bars):")
    print(f"  Current: {rolling_corr.iloc[-1]:.4f}")
    print(f"  Mean: {rolling_corr.mean():.4f}")
    print(f"  Std: {rolling_corr.std():.4f}")
    
    # Spread analysis
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tsa.stattools import adfuller
    
    # OLS for hedge ratio
    X = gbpusd.values.reshape(-1, 1)
    y = eurusd.values
    
    # Add constant
    X_with_const = np.column_stack([np.ones(len(X)), X])
    
    # OLS
    beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
    intercept, hedge_ratio = beta[0], beta[1]
    
    print(f"\nHedge Ratio (OLS): {hedge_ratio:.4f}")
    print(f"Intercept: {intercept:.4f}")
    
    # Spread
    spread = eurusd - hedge_ratio * gbpusd
    print(f"\nSpread stats:")
    print(f"  Mean: {spread.mean():.6f}")
    print(f"  Std: {spread.std():.6f}")
    
    # ADF test on spread
    adf_result = adfuller(spread.dropna(), maxlag=20, autolag='AIC')
    print(f"\nADF test on spread:")
    print(f"  Statistic: {adf_result[0]:.4f}")
    print(f"  P-value: {adf_result[1]:.4f}")
    print(f"  Critical values: 1%={adf_result[4]['1%']:.4f}, 5%={adf_result[4]['5%']:.4f}")
    
    # Half-life calculation
    spread_lag = spread.shift(1)
    spread_diff = spread - spread_lag
    
    # Remove NaN
    mask = ~(spread_lag.isna() | spread_diff.isna())
    spread_lag_clean = spread_lag[mask].values.reshape(-1, 1)
    spread_diff_clean = spread_diff[mask].values
    
    # OLS for mean reversion speed
    X_hl = np.column_stack([np.ones(len(spread_lag_clean)), spread_lag_clean])
    theta = np.linalg.lstsq(X_hl, spread_diff_clean, rcond=None)[0]
    
    if theta[1] < 0:
        half_life = -np.log(2) / theta[1]
    else:
        half_life = np.inf
    
    print(f"\nHalf-life: {half_life:.1f} bars ({half_life/24:.1f} days)")
    
    # Hurst exponent
    def hurst_exponent(ts, max_lag=100):
        lags = range(2, min(max_lag, len(ts) // 2))
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0
    
    hurst = hurst_exponent(spread.values)
    print(f"Hurst exponent: {hurst:.4f}")
    print(f"  < 0.5 = mean reverting, > 0.5 = trending")
    
    # Z-score
    zscore_window = 60
    spread_mean = spread.rolling(zscore_window).mean()
    spread_std = spread.rolling(zscore_window).std()
    zscore = (spread - spread_mean) / spread_std
    
    print(f"\nCurrent Z-score: {zscore.iloc[-1]:.2f}")
    
    # Test other pairs
    print("\n" + "=" * 60)
    print("TESTING OTHER COMMON PAIRS")
    print("=" * 60)
    
    test_pairs = [
        ("AUDUSD", "NZDUSD"),
        ("EURJPY", "USDJPY"),
        ("EURGBP", "EURUSD"),
        ("GBPUSD", "GBPJPY"),
    ]
    
    for sym_a, sym_b in test_pairs:
        price_a = load_data(sym_a, data_dir)
        price_b = load_data(sym_b, data_dir)
        
        if price_a is None or price_b is None:
            print(f"\n{sym_a}/{sym_b}: Missing data")
            continue
        
        common = price_a.index.intersection(price_b.index)
        price_a = price_a.loc[common]
        price_b = price_b.loc[common]
        
        # Returns correlation
        ret_a = price_a.pct_change().dropna()
        ret_b = price_b.pct_change().dropna()
        corr = ret_a.corr(ret_b)
        
        # Spread
        X = price_b.values.reshape(-1, 1)
        y = price_a.values
        X_c = np.column_stack([np.ones(len(X)), X])
        beta = np.linalg.lstsq(X_c, y, rcond=None)[0]
        hr = beta[1]
        spread = price_a - hr * price_b
        
        # ADF
        adf = adfuller(spread.dropna(), maxlag=20, autolag='AIC')
        
        # Half-life
        spread_lag = spread.shift(1)
        spread_diff = spread - spread_lag
        mask = ~(spread_lag.isna() | spread_diff.isna())
        X_hl = np.column_stack([np.ones(mask.sum()), spread_lag[mask].values.reshape(-1, 1)])
        theta = np.linalg.lstsq(X_hl, spread_diff[mask].values, rcond=None)[0]
        hl = -np.log(2) / theta[1] if theta[1] < 0 else np.inf
        
        # Hurst
        h = hurst_exponent(spread.values)
        
        print(f"\n{sym_a}/{sym_b}:")
        print(f"  Returns Corr: {corr:.3f}")
        print(f"  ADF p-value: {adf[1]:.4f} ({'cointegrated' if adf[1] < 0.05 else 'NOT cointegrated'})")
        print(f"  Half-life: {hl:.1f} bars")
        print(f"  Hurst: {h:.3f} ({'mean-reverting' if h < 0.5 else 'trending'})")


if __name__ == '__main__':
    main()
