"""
Example: Screen for tradeable pairs.

This script demonstrates how to:
1. Load data for multiple symbols
2. Calculate correlation and cointegration
3. Filter pairs by trading criteria
4. Display current trading opportunities
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import Settings, Timeframe
from config.broker_config import MT5Config
from src.data.broker_client import MT5Client, Timeframe as MT5Timeframe
from src.data.data_manager import DataManager
from src.analysis.correlation import CorrelationAnalyzer
from src.analysis.cointegration import CointegrationAnalyzer
from src.analysis.spread_builder import SpreadBuilder


def main():
    """Run pair screening example."""
    print("="*60)
    print("PAIRS TRADING - PAIR SCREENING")
    print("="*60)
    
    # Configuration
    symbols = [
        "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
        "AUDUSD", "NZDUSD", "USDCAD",
        "EURJPY", "GBPJPY", "EURGBP", "EURCHF"
    ]
    days = 180
    timeframe = Timeframe.H1
    
    print(f"\nSymbols: {len(symbols)}")
    print(f"Period: {days} days")
    print(f"Timeframe: {timeframe.value}")
    
    # Initialize
    settings = Settings()
    
    try:
        # Connect to MT5
        config = MT5Config.from_env()
        client = MT5Client(config)
        
        if not client.connect():
            print("\nERROR: Could not connect to MT5")
            return
        
        print("\n✓ Connected to MT5")
        
        # Load data for all symbols
        data_manager = DataManager(client, settings.paths.cache_dir)
        
        bars_per_day = 24
        count = days * bars_per_day
        
        mt5_tf = MT5Timeframe.from_string(timeframe.value)
        
        print(f"\nLoading data for {len(symbols)} symbols...")
        
        symbol_data = {}
        for symbol in symbols:
            try:
                data = data_manager.get_close_prices(symbol, mt5_tf, count)
                if len(data) >= 200:
                    symbol_data[symbol] = data
                    print(f"  ✓ {symbol}: {len(data)} bars")
                else:
                    print(f"  ✗ {symbol}: insufficient data")
            except Exception as e:
                print(f"  ✗ {symbol}: {e}")
        
        print(f"\n✓ Loaded {len(symbol_data)} symbols")
        
        # Analyze pairs
        print("\n" + "-"*60)
        print("ANALYZING PAIRS...")
        print("-"*60)
        
        corr_analyzer = CorrelationAnalyzer(window=settings.spread.correlation_window)
        coint_analyzer = CointegrationAnalyzer()
        spread_builder = SpreadBuilder(
            regression_window=settings.spread.regression_window,
            zscore_window=settings.spread.zscore_window
        )
        
        tradeable_pairs = []
        symbols_list = list(symbol_data.keys())
        
        total_pairs = len(symbols_list) * (len(symbols_list) - 1) // 2
        analyzed = 0
        
        for i, symbol_a in enumerate(symbols_list):
            for symbol_b in symbols_list[i+1:]:
                analyzed += 1
                
                price_a = symbol_data[symbol_a]
                price_b = symbol_data[symbol_b]
                
                # Align data
                common_idx = price_a.index.intersection(price_b.index)
                if len(common_idx) < 200:
                    continue
                
                price_a_aligned = price_a.loc[common_idx]
                price_b_aligned = price_b.loc[common_idx]
                
                # Correlation
                corr_result = corr_analyzer.analyze_pair(price_a_aligned, price_b_aligned)
                
                if corr_result.current_correlation < settings.spread.min_correlation:
                    continue
                
                # Cointegration
                coint_result = coint_analyzer.engle_granger_test(price_a_aligned, price_b_aligned)
                
                if not coint_result.is_cointegrated:
                    continue
                
                # Spread metrics
                metrics = spread_builder.get_spread_metrics(price_a_aligned, price_b_aligned)
                
                if metrics is None:
                    continue
                
                if metrics.half_life > settings.spread.max_half_life:
                    continue
                
                # Current z-score
                spread_data = spread_builder.build_spread_with_zscore(price_a_aligned, price_b_aligned)
                current_zscore = spread_data['zscore'].iloc[-1]
                
                tradeable_pairs.append({
                    'pair': (symbol_a, symbol_b),
                    'correlation': corr_result.current_correlation,
                    'stability': corr_result.stability_score,
                    'p_value': coint_result.p_value,
                    'hedge_ratio': coint_result.hedge_ratio,
                    'half_life': coint_result.half_life,
                    'hurst': metrics.hurst_exponent,
                    'zscore': current_zscore
                })
        
        print(f"\n✓ Analyzed {analyzed} pairs")
        print(f"✓ Found {len(tradeable_pairs)} tradeable pairs")
        
        # Display results
        print("\n" + "="*60)
        print("TRADEABLE PAIRS")
        print("="*60)
        
        # Sort by correlation
        tradeable_pairs.sort(key=lambda x: x['correlation'], reverse=True)
        
        for i, p in enumerate(tradeable_pairs, 1):
            symbol_a, symbol_b = p['pair']
            
            print(f"\n{i}. {symbol_a}/{symbol_b}")
            print(f"   Correlation: {p['correlation']:.3f} (stability: {p['stability']:.2f})")
            print(f"   Coint p-val: {p['p_value']:.4f}")
            print(f"   Hedge ratio: {p['hedge_ratio']:.4f}")
            print(f"   Half-life:   {p['half_life']:.1f} bars")
            print(f"   Hurst exp:   {p['hurst']:.3f}")
            print(f"   Z-score:     {p['zscore']:+.2f}")
            
            # Trading signal
            if abs(p['zscore']) >= settings.spread.entry_zscore:
                if p['zscore'] <= -settings.spread.entry_zscore:
                    print(f"   → SIGNAL: LONG SPREAD (buy {symbol_a}, sell {symbol_b})")
                else:
                    print(f"   → SIGNAL: SHORT SPREAD (sell {symbol_a}, buy {symbol_b})")
        
        # Summary
        print("\n" + "-"*60)
        print("TRADING OPPORTUNITIES")
        print("-"*60)
        
        signals = [p for p in tradeable_pairs if abs(p['zscore']) >= settings.spread.entry_zscore]
        
        if signals:
            print(f"\n{len(signals)} pairs with entry signals:")
            for p in signals:
                symbol_a, symbol_b = p['pair']
                direction = "LONG" if p['zscore'] < 0 else "SHORT"
                print(f"  {symbol_a}/{symbol_b}: {direction} SPREAD (Z={p['zscore']:+.2f})")
        else:
            print("\nNo entry signals at current levels.")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        raise
    finally:
        if 'client' in locals():
            client.disconnect()
    
    print("\n" + "="*60)
    print("SCREENING COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
