"""
Strict Institutional Forex Pair Screening.

This script implements ZERO-TOLERANCE filtering for pairs trading.
Most pairs will be rejected. This is by design.

Usage:
    python scripts/strict_screen.py
    python scripts/strict_screen.py --timeframe H4
    python scripts/strict_screen.py --backtest --save
"""

import sys
from pathlib import Path
from datetime import datetime
import argparse
import logging
import pandas as pd
import numpy as np
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import Settings
from src.analysis.strict_selector import (
    StrictForexPairSelector, 
    StrictPipelineResult,
    VALID_PAIR_COMBINATIONS,
    TIMEFRAME_CONFIG
)


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Complete Forex Universe
FOREX_UNIVERSE = [
    # Majors
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "NZDUSD", "USDCAD",
    # JPY Crosses  
    "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY",
    # EUR Crosses
    "EURGBP", "EURAUD", "EURNZD", "EURCHF", "EURCAD",
    # GBP Crosses
    "GBPAUD", "GBPNZD", "GBPCAD", "GBPCHF",
    # Other
    "AUDNZD", "AUDCAD", "AUDCHF", "NZDCAD", "NZDCHF", "CADCHF"
]


def load_local_data(symbol: str, data_dir: Path, timeframe: str = "H1") -> pd.Series:
    """Load data from local parquet file."""
    pattern = f"{symbol}_{timeframe}_*.parquet"
    files = list(data_dir.glob(pattern))
    
    if not files:
        pattern = f"{symbol}_{timeframe}.parquet"
        files = list(data_dir.glob(pattern))
    
    if not files:
        return None
    
    filepath = max(files, key=lambda f: f.stat().st_mtime)
    df = pd.read_parquet(filepath)
    
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'time' in df.columns:
            df.set_index('time', inplace=True)
        df.index = pd.to_datetime(df.index)
    
    return df['close']


def save_results(result: StrictPipelineResult, filepath: Path):
    """Save results to JSON."""
    data = {
        'timestamp': result.timestamp.isoformat(),
        'timeframe': result.timeframe,
        'thresholds': {
            'max_half_life': result.thresholds.max_half_life,
            'optimal_half_life': list(result.thresholds.optimal_half_life),
            'min_trades_per_year': result.thresholds.min_trades_per_year
        },
        'pipeline_stats': {
            'symbols_analyzed': result.symbols_analyzed,
            'pairs_analyzed': result.pairs_analyzed,
            'rejected_economic': result.rejected_economic,
            'rejected_correlation': result.rejected_correlation,
            'rejected_cointegration': result.rejected_cointegration,
            'rejected_stationarity': result.rejected_stationarity,
            'rejected_half_life': result.rejected_half_life,
            'rejected_hurst': result.rejected_hurst,
            'rejected_trade_frequency': result.rejected_trade_frequency,
            'final_candidates': result.final_candidates
        },
        'selected_pairs': [
            {
                'pair': list(p.pair),
                'quality_score': float(p.quality_score),
                'economic_relationship': p.economic_relationship,
                'pearson_correlation': float(p.pearson_correlation),
                'spearman_correlation': float(p.spearman_correlation),
                'correlation_stability': float(p.correlation_stability),
                'eg_pvalue': float(p.eg_pvalue),
                'eg_is_cointegrated': bool(p.eg_is_cointegrated),
                'johansen_is_cointegrated': bool(p.johansen_is_cointegrated),
                'rolling_coint_breakdown_pct': float(p.rolling_coint_breakdown_pct),
                'adf_pvalue': float(p.adf_pvalue),
                'half_life': float(p.half_life),
                'half_life_days': float(p.half_life_days),
                'hurst_exponent': float(p.hurst_exponent),
                'hedge_ratio': float(p.hedge_ratio),
                'hedge_ratio_stability': float(p.hedge_ratio_stability),
                'current_zscore': float(p.current_zscore),
                'estimated_trades_per_year': float(p.estimated_trades_per_year)
            }
            for p in result.selected_pairs
        ]
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def run_strict_backtest(pair, price_a, price_b, half_life, hurst, settings):
    """
    Run strict backtest with adaptive parameters.
    
    Key constraints:
    - Max trade duration = 5 * half_life
    - Position size scaled by mean reversion strength
    - No trades if Hurst > 0.55
    """
    from src.backtest.backtest_engine import BacktestEngine
    
    if hurst > 0.55:
        logger.warning(f"Skipping backtest - Hurst {hurst:.3f} > 0.55")
        return None
    
    # Adapt parameters to half-life
    if half_life < 20:
        settings.spread.entry_zscore = 2.0
        settings.spread.exit_zscore = 0.2
        settings.spread.stop_loss_zscore = 3.0
    elif half_life < 40:
        settings.spread.entry_zscore = 2.0
        settings.spread.exit_zscore = 0.3
        settings.spread.stop_loss_zscore = 3.5
    else:
        settings.spread.entry_zscore = 2.2
        settings.spread.exit_zscore = 0.5
        settings.spread.stop_loss_zscore = 4.0
    
    engine = BacktestEngine(settings)
    result = engine.run_backtest(pair, price_a, price_b)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Strict institutional forex pair screening'
    )
    
    parser.add_argument(
        '--symbols', type=str,
        help='Comma-separated symbols (default: full forex universe)'
    )
    parser.add_argument(
        '--data-dir', type=str, default='data/historical',
        help='Data directory'
    )
    parser.add_argument(
        '--timeframe', type=str, default='H1',
        choices=['M15', 'M30', 'H1', 'H4', 'D1'],
        help='Timeframe (default: H1)'
    )
    parser.add_argument(
        '--backtest', action='store_true',
        help='Run backtest on selected pairs'
    )
    parser.add_argument(
        '--save', action='store_true',
        help='Save results to file'
    )
    parser.add_argument(
        '--show-all', action='store_true',
        help='Show all analyzed pairs (not just selected)'
    )
    
    args = parser.parse_args()
    
    # Parse symbols
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(',')]
    else:
        symbols = FOREX_UNIVERSE
    
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        print("\nDownload data first:")
        print("  python scripts/download_data.py --all --days 730")
        return 1
    
    # Header
    print("=" * 80)
    print("STRICT INSTITUTIONAL FOREX PAIR SELECTION")
    print("=" * 80)
    print("\n⚠️  This pipeline uses ZERO-TOLERANCE filtering.")
    print("    Most pairs WILL be rejected. This is by design.")
    print("    Prefer zero trades over statistically invalid trades.")
    
    # Show thresholds
    thresholds = TIMEFRAME_CONFIG[args.timeframe]
    print(f"\n" + "-" * 80)
    print(f"STRICT THRESHOLDS FOR {args.timeframe}")
    print("-" * 80)
    print(f"  Max half-life:        {thresholds.max_half_life} bars")
    print(f"  Optimal half-life:    {thresholds.optimal_half_life[0]}-{thresholds.optimal_half_life[1]} bars")
    print(f"  Max Hurst exponent:   0.55 (HARD REJECTION above)")
    print(f"  Min correlation:      0.70 (Pearson AND Spearman)")
    print(f"  EG p-value:           < 0.02 (very strict)")
    print(f"  Min trades/year:      {thresholds.min_trades_per_year}")
    
    # Show valid pair combinations
    print(f"\n" + "-" * 80)
    print("VALID ECONOMIC PAIR COMBINATIONS")
    print("-" * 80)
    for pair in sorted(VALID_PAIR_COMBINATIONS):
        print(f"  • {pair[0]} / {pair[1]}")
    
    # Load data
    print(f"\n" + "-" * 80)
    print("LOADING DATA")
    print("-" * 80)
    
    price_data = {}
    
    for symbol in symbols:
        data = load_local_data(symbol, data_dir, args.timeframe)
        if data is not None and len(data) >= 1000:
            price_data[symbol] = data
            print(f"  ✓ {symbol}: {len(data)} bars")
        else:
            print(f"  ✗ {symbol}: No data or insufficient")
    
    print(f"\nLoaded: {len(price_data)} symbols")
    
    if len(price_data) < 2:
        print("\nERROR: Need at least 2 symbols")
        return 1
    
    # Run strict pipeline
    print(f"\n" + "-" * 80)
    print("RUNNING STRICT PIPELINE")
    print("-" * 80)
    
    selector = StrictForexPairSelector(timeframe=args.timeframe)
    result = selector.run_pipeline(price_data, top_n=5)
    
    # Display report
    print("\n" + selector.generate_report(result))
    
    # Show all pairs if requested
    if args.show_all:
        print(f"\n" + "-" * 80)
        print("ALL ANALYZED PAIRS")
        print("-" * 80)
        
        for p in result.all_analyses:
            status = "✓" if p.passes_all_filters else "✗"
            print(f"\n{status} {p.pair[0]}/{p.pair[1]}")
            print(f"    Economic: {'✓' if p.is_economically_valid else '✗'}")
            print(f"    Correlation: {p.pearson_correlation:.3f} / {p.spearman_correlation:.3f}")
            print(f"    Cointegrated: EG={'✓' if p.eg_is_cointegrated else '✗'} Joh={'✓' if p.johansen_is_cointegrated else '✗'}")
            print(f"    Half-life: {p.half_life:.0f} bars ({'✓' if p.passes_half_life else '✗'})")
            print(f"    Hurst: {p.hurst_exponent:.3f} ({'✓' if p.passes_hurst else '✗'})")
            if p.rejection_reasons:
                print(f"    Rejected: {p.rejection_reasons[0]}")
    
    # Backtest selected pairs
    if args.backtest and result.selected_pairs:
        print(f"\n" + "-" * 80)
        print("STRICT BACKTEST VALIDATION")
        print("-" * 80)
        
        settings = Settings()
        
        for p in result.selected_pairs:
            print(f"\n  Testing {p.pair[0]}/{p.pair[1]}...")
            print(f"    Half-life: {p.half_life:.0f} bars | Hurst: {p.hurst_exponent:.3f}")
            
            bt_result = run_strict_backtest(
                p.pair,
                price_data[p.pair[0]],
                price_data[p.pair[1]],
                p.half_life,
                p.hurst_exponent,
                settings
            )
            
            if bt_result and bt_result.total_trades > 0:
                status = "✓" if bt_result.sharpe_ratio > 0.5 else "⚠️" if bt_result.sharpe_ratio > 0 else "✗"
                print(f"    {status} Sharpe: {bt_result.sharpe_ratio:.2f}")
                print(f"      Return: {bt_result.total_return:.1%}")
                print(f"      Trades: {bt_result.total_trades}")
                print(f"      Win Rate: {bt_result.win_rate:.1%}")
                print(f"      Max DD: {bt_result.max_drawdown:.1%}")
                print(f"      Avg Trade: ${bt_result.avg_trade:.2f}")
                
                if bt_result.sharpe_ratio > 0.5:
                    print(f"\n    ✅ VALIDATED for paper trading")
                elif bt_result.sharpe_ratio > 0:
                    print(f"\n    ⚠️ Marginal - needs more validation")
                else:
                    print(f"\n    ❌ Failed backtest validation")
            else:
                print(f"    ✗ No trades generated or backtest failed")
    
    # Final recommendations
    print(f"\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    if result.selected_pairs:
        print(f"\n✅ {len(result.selected_pairs)} pair(s) passed ALL strict filters:")
        
        for p in result.selected_pairs:
            signal = ""
            if p.current_zscore < -2.0:
                signal = "🟢 LONG SIGNAL NOW"
            elif p.current_zscore > 2.0:
                signal = "🔴 SHORT SIGNAL NOW"
            else:
                signal = f"Wait for Z to reach ±2.0 (current: {p.current_zscore:+.2f})"
            
            print(f"\n  → {p.pair[0]}/{p.pair[1]}")
            print(f"    Quality Score: {p.quality_score:.0f}/100")
            print(f"    Half-life: {p.half_life:.0f} bars ({p.half_life_days:.1f} days)")
            print(f"    Hurst: {p.hurst_exponent:.3f}")
            print(f"    Signal: {signal}")
    else:
        print("\n❌ NO PAIRS PASSED ALL FILTERS")
        print("\n   This is the expected behavior for a strict institutional pipeline.")
        print("   The current Forex market may not offer statistically valid pairs.")
        print("\n   Options:")
        print("   1. Try a different timeframe (H4 or D1 may show different results)")
        print("   2. Wait for market regime change")
        print("   3. Add more symbols to the universe")
        print("\n   DO NOT relax filters to force trades.")
        print("   Zero trades > Statistically invalid trades")
    
    # Save results
    if args.save:
        results_dir = Path("results/screening")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = results_dir / f"strict_{args.timeframe}_{datetime.now():%Y%m%d_%H%M%S}.json"
        save_results(result, filepath)
        print(f"\n✓ Results saved to: {filepath}")
    
    print(f"\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
