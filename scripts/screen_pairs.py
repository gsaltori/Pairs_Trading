"""
Example: Screen currency pairs to find the best candidates for pairs trading.

This script demonstrates how to:
1. Analyze correlation between multiple currency pairs
2. Test for cointegration
3. Calculate spread metrics
4. Rank pairs by tradability
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import Settings
from config.broker_config import BrokerConfig
from src.data.broker_client import OandaClient
from src.data.data_manager import DataManager
from src.analysis.correlation import CorrelationAnalyzer
from src.analysis.cointegration import CointegrationAnalyzer
from src.analysis.spread_builder import SpreadBuilder


def main():
    """Run pair screening example."""
    print("="*60)
    print("PAIRS TRADING - PAIR SCREENING")
    print("="*60)
    
    # Instruments to screen
    instruments = [
        'EUR_USD', 'GBP_USD', 'USD_JPY', 'USD_CHF',
        'AUD_USD', 'NZD_USD', 'USD_CAD', 'EUR_GBP',
        'EUR_JPY', 'GBP_JPY', 'AUD_JPY', 'EUR_CHF'
    ]
    
    days_of_history = 180
    
    print(f"\nScreening {len(instruments)} instruments")
    print(f"History: {days_of_history} days")
    
    # Initialize
    settings = Settings()
    
    try:
        broker_config = BrokerConfig.from_env()
        client = OandaClient(broker_config)
        data_manager = DataManager(client, settings.paths.cache_dir)
        print("✓ Connected to OANDA")
    except Exception as e:
        print(f"ERROR: Could not connect to OANDA: {e}")
        return
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_of_history)
    
    # Fetch data for all instruments
    print("\nFetching data...")
    data = {}
    
    for inst in instruments:
        df = data_manager.fetch_data(
            instrument=inst,
            timeframe=settings.timeframe,
            start_date=start_date,
            end_date=end_date
        )
        if df is not None and len(df) > 100:
            data[inst] = df
            print(f"  ✓ {inst}: {len(df)} bars")
        else:
            print(f"  ✗ {inst}: insufficient data")
    
    print(f"\nLoaded {len(data)} instruments")
    
    # Initialize analyzers
    correlation_analyzer = CorrelationAnalyzer(
        window=settings.spread.regression_window
    )
    cointegration_analyzer = CointegrationAnalyzer()
    spread_builder = SpreadBuilder(
        regression_window=settings.spread.regression_window,
        zscore_window=settings.spread.zscore_window
    )
    
    # Analyze all pairs
    print("\nAnalyzing pairs...")
    results = []
    
    instruments_list = list(data.keys())
    total_pairs = len(instruments_list) * (len(instruments_list) - 1) // 2
    analyzed = 0
    
    for i, inst_a in enumerate(instruments_list):
        for inst_b in instruments_list[i+1:]:
            analyzed += 1
            
            try:
                # Get aligned prices
                price_a = data[inst_a]['close']
                price_b = data[inst_b]['close']
                
                # Align
                common_idx = price_a.index.intersection(price_b.index)
                if len(common_idx) < 200:
                    continue
                
                price_a = price_a.loc[common_idx]
                price_b = price_b.loc[common_idx]
                
                # Correlation
                corr_result = correlation_analyzer.analyze_pair(price_a, price_b)
                
                # Skip low correlation pairs early
                if corr_result.current_correlation < 0.5:
                    continue
                
                # Cointegration
                coint_result = cointegration_analyzer.engle_granger_test(
                    price_a, price_b
                )
                
                # Spread metrics
                spread_metrics = spread_builder.get_spread_metrics(price_a, price_b)
                
                results.append({
                    'Pair': f"{inst_a}/{inst_b}",
                    'Corr': corr_result.current_correlation,
                    'Corr_Stab': corr_result.stability_score,
                    'Coint_p': coint_result.p_value,
                    'Coint': '✓' if coint_result.is_cointegrated else '✗',
                    'Hedge': coint_result.hedge_ratio,
                    'Half_Life': coint_result.half_life,
                    'Hurst': spread_metrics.hurst_exponent if spread_metrics else None,
                    'Z-Score': spread_metrics.zscore if spread_metrics else None
                })
                
            except Exception as e:
                continue
    
    print(f"Analyzed {analyzed} pairs")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    if len(df) == 0:
        print("\nNo tradeable pairs found!")
        return
    
    # Sort by correlation stability
    df = df.sort_values('Corr_Stab', ascending=False)
    
    # Print all results
    print("\n" + "="*60)
    print("ALL ANALYZED PAIRS")
    print("="*60)
    
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', '{:.3f}'.format)
    
    print(df.to_string(index=False))
    
    # Filter tradeable pairs
    print("\n" + "="*60)
    print("TRADEABLE PAIRS (meeting all criteria)")
    print("="*60)
    
    tradeable = df[
        (df['Corr'] >= settings.spread.min_correlation) &
        (df['Coint'] == '✓') &
        (df['Half_Life'] <= settings.spread.max_half_life) &
        (df['Half_Life'] > 0)
    ].copy()
    
    if len(tradeable) > 0:
        print(f"\nCriteria:")
        print(f"  - Correlation >= {settings.spread.min_correlation}")
        print(f"  - Cointegrated (p < 0.05)")
        print(f"  - Half-life <= {settings.spread.max_half_life} bars")
        print(f"\nFound {len(tradeable)} tradeable pairs:\n")
        print(tradeable.to_string(index=False))
        
        # Best pair recommendation
        best = tradeable.iloc[0]
        print(f"\n{'='*60}")
        print("RECOMMENDED PAIR")
        print(f"{'='*60}")
        print(f"\n  {best['Pair']}")
        print(f"  Correlation: {best['Corr']:.3f}")
        print(f"  Stability Score: {best['Corr_Stab']:.3f}")
        print(f"  Half-Life: {best['Half_Life']:.1f} bars")
        print(f"  Current Z-Score: {best['Z-Score']:.2f}")
        
        # Trading opportunity
        zscore = best['Z-Score']
        if zscore <= -settings.spread.entry_zscore:
            print(f"\n  🟢 POTENTIAL LONG SPREAD ENTRY (z={zscore:.2f})")
        elif zscore >= settings.spread.entry_zscore:
            print(f"\n  🔴 POTENTIAL SHORT SPREAD ENTRY (z={zscore:.2f})")
        else:
            print(f"\n  ⏸ No immediate trading signal")
    else:
        print("\nNo pairs meet all criteria.")
        print("Try adjusting the screening parameters in settings.")
    
    print("\n" + "="*60)
    print("SCREENING COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
