"""
Institutional Pair Screening Script.

Professional-grade pair selection with multi-stage statistical validation.

Usage:
    python scripts/institutional_screen.py
    python scripts/institutional_screen.py --timeframe D1
    python scripts/institutional_screen.py --backtest --save
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
from src.analysis.institutional_selector import InstitutionalPairSelector, PipelineResult
from src.backtest.backtest_engine import BacktestEngine
from src.strategy.adaptive_params import ParameterAdapter, format_parameters_report


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Forex Universe
FOREX_MAJORS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "NZDUSD", "USDCAD"]
FOREX_MINORS = ["EURGBP", "EURJPY", "GBPJPY", "AUDJPY", "CHFJPY", "EURAUD", "EURNZD"]
FOREX_CROSSES = ["GBPAUD", "AUDNZD", "NZDJPY", "CADJPY", "EURCHF", "GBPCHF"]

DEFAULT_SYMBOLS = FOREX_MAJORS + FOREX_MINORS + ["GBPAUD", "AUDNZD", "EURCHF"]


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


def run_backtest_validation(pair, price_a, price_b, settings):
    """Run backtest to validate pair tradability."""
    engine = BacktestEngine(settings)
    result = engine.run_backtest(pair, price_a, price_b)
    return result


def save_results_json(result: PipelineResult, filepath: Path):
    """Save results to JSON with proper serialization."""
    data = {
        'timestamp': result.timestamp.isoformat(),
        'timeframe': result.timeframe,
        'symbols_analyzed': int(result.symbols_analyzed),
        'pairs_analyzed': int(result.pairs_analyzed),
        'pipeline_funnel': {
            'passed_correlation': int(result.passed_correlation),
            'passed_cointegration': int(result.passed_cointegration),
            'passed_stationarity': int(result.passed_stationarity),
            'passed_half_life': int(result.passed_half_life),
            'final_candidates': int(result.final_candidates)
        },
        'summary': {
            'avg_correlation': float(result.avg_correlation),
            'avg_half_life': float(result.avg_half_life),
            'cointegration_rate': float(result.cointegration_rate)
        },
        'selected_pairs': [
            {
                'pair': list(p.pair),
                'overall_score': float(p.overall_score),
                'is_tradeable': bool(p.is_tradeable),
                'correlation': {
                    'pearson': float(p.pearson_correlation),
                    'spearman': float(p.spearman_correlation),
                    'stability': float(p.correlation_stability)
                },
                'cointegration': {
                    'engle_granger_pvalue': float(p.eg_coint_pvalue),
                    'eg_is_cointegrated': bool(p.eg_is_cointegrated),
                    'johansen_is_cointegrated': bool(p.johansen_is_cointegrated)
                },
                'stationarity': {
                    'adf_pvalue': float(p.adf_pvalue),
                    'adf_is_stationary': bool(p.adf_is_stationary),
                    'kpss_pvalue': float(p.kpss_pvalue),
                    'kpss_is_stationary': bool(p.kpss_is_stationary)
                },
                'mean_reversion': {
                    'half_life_bars': float(p.half_life),
                    'half_life_days': float(p.half_life_days),
                    'hurst_exponent': float(p.hurst_exponent)
                },
                'spread': {
                    'hedge_ratio': float(p.hedge_ratio),
                    'hedge_ratio_std': float(p.hedge_ratio_std),
                    'current_zscore': float(p.current_zscore)
                },
                'warnings': p.warnings,
                'rejection_reasons': p.rejection_reasons
            }
            for p in result.selected_pairs
        ]
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='Institutional pair screening with multi-stage statistical validation'
    )
    
    parser.add_argument(
        '--symbols', type=str,
        help='Comma-separated list of symbols'
    )
    parser.add_argument(
        '--data-dir', type=str, default='data/historical',
        help='Directory containing historical data'
    )
    parser.add_argument(
        '--timeframe', type=str, default='H1',
        help='Timeframe: M15, M30, H1, H4, D1 (default: H1)'
    )
    parser.add_argument(
        '--top', type=int, default=10,
        help='Number of top pairs to show (default: 10)'
    )
    parser.add_argument(
        '--min-corr', type=float, default=0.70,
        help='Minimum correlation threshold (default: 0.70)'
    )
    parser.add_argument(
        '--max-half-life', type=int, default=300,
        help='Maximum half-life in bars (default: 300)'
    )
    parser.add_argument(
        '--backtest', action='store_true',
        help='Run backtest validation on selected pairs'
    )
    parser.add_argument(
        '--save', action='store_true',
        help='Save results to file'
    )
    
    args = parser.parse_args()
    
    # Parse symbols
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(',')]
    else:
        symbols = DEFAULT_SYMBOLS
    
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        print("\nDownload data first:")
        print("  python scripts/download_data.py --forex --days 730")
        return 1
    
    print("=" * 80)
    print("INSTITUTIONAL PAIR SELECTION PIPELINE")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Symbols: {len(symbols)}")
    print(f"  Data directory: {data_dir}")
    print(f"  Timeframe: {args.timeframe}")
    print(f"  Min correlation: {args.min_corr}")
    print(f"  Max half-life: {args.max_half_life} bars")
    
    # Load data
    print("\n" + "-" * 80)
    print("LOADING DATA")
    print("-" * 80)
    
    price_data = {}
    
    for symbol in symbols:
        data = load_local_data(symbol, data_dir, args.timeframe)
        if data is not None and len(data) >= 500:
            price_data[symbol] = data
            bars = len(data)
            days = bars / (24 if args.timeframe == 'H1' else 1)
            print(f"  ✓ {symbol}: {bars} bars ({days:.0f} days)")
        else:
            print(f"  ✗ {symbol}: No data or insufficient")
    
    print(f"\nLoaded: {len(price_data)} symbols")
    n_pairs = len(price_data) * (len(price_data) - 1) // 2
    print(f"Pairs to analyze: {n_pairs}")
    
    if len(price_data) < 2:
        print("\nERROR: Need at least 2 symbols")
        return 1
    
    # Run institutional pipeline
    print("\n" + "-" * 80)
    print("RUNNING INSTITUTIONAL SELECTION PIPELINE")
    print("-" * 80)
    
    selector = InstitutionalPairSelector(
        min_pearson_corr=args.min_corr,
        max_half_life=args.max_half_life
    )
    
    result = selector.run_pipeline(
        price_data=price_data,
        timeframe=args.timeframe,
        top_n=args.top
    )
    
    # Display report
    print("\n" + selector.generate_report(result))
    
    # Backtest validation
    if args.backtest and result.selected_pairs:
        print("\n" + "-" * 80)
        print("BACKTEST VALIDATION")
        print("-" * 80)
        
        settings = Settings()
        
        # Adjust settings based on half-life
        for p in result.selected_pairs[:5]:
            print(f"\n  Testing {p.pair[0]}/{p.pair[1]}...")
            
            # Adjust exit threshold based on half-life
            if p.half_life > 100:
                settings.spread.exit_zscore = 0.5  # More patient exit
                settings.spread.stop_loss_zscore = 4.0  # Wider stop
            
            bt_result = run_backtest_validation(
                p.pair,
                price_data[p.pair[0]],
                price_data[p.pair[1]],
                settings
            )
            
            if bt_result and bt_result.total_trades > 0:
                status = "✓" if bt_result.sharpe_ratio > 0 else "✗"
                print(f"    {status} Sharpe: {bt_result.sharpe_ratio:.2f}")
                print(f"      Return: {bt_result.total_return:.1%}")
                print(f"      Trades: {bt_result.total_trades}")
                print(f"      Win Rate: {bt_result.win_rate:.1%}")
                print(f"      Max DD: {bt_result.max_drawdown:.1%}")
                print(f"      Avg Trade: ${bt_result.avg_trade:.2f}")
            else:
                print(f"    ✗ No trades or backtest failed")
    
    # Trading recommendations
    print("\n" + "=" * 80)
    print("TRADING RECOMMENDATIONS")
    print("=" * 80)
    
    # Tradeable pairs with signals
    tradeable = [p for p in result.selected_pairs if p.is_tradeable]
    signals = [p for p in tradeable if abs(p.current_zscore) >= 2.0]
    
    if signals:
        print("\n🚨 PAIRS WITH ACTIVE ENTRY SIGNALS:")
        for p in signals:
            direction = "LONG spread" if p.current_zscore < 0 else "SHORT spread"
            print(f"\n  → {p.pair[0]}/{p.pair[1]}: {direction}")
            print(f"    Z-score: {p.current_zscore:+.2f}")
            print(f"    Hedge ratio: {p.hedge_ratio:.4f}")
            print(f"    Half-life: {p.half_life:.0f} bars ({p.half_life_days:.1f} days)")
            print(f"    Cointegration p-value: {p.eg_coint_pvalue:.4f}")
            
            # Get adaptive parameters
            adaptive = ParameterAdapter.adapt(
                half_life=p.half_life,
                hurst=p.hurst_exponent,
                coint_pvalue=p.eg_coint_pvalue
            )
            print(f"\n    RECOMMENDED PARAMETERS:")
            print(f"      Regime: {adaptive.regime}")
            print(f"      Entry: ±{adaptive.entry_zscore} | Exit: ±{adaptive.exit_zscore} | Stop: ±{adaptive.stop_loss_zscore}")
            print(f"      Max holding: {adaptive.max_holding_bars} bars")
            print(f"      Position size factor: {adaptive.position_size_factor:.0%}")
            
            # Recommended timeframe
            rec_tf = ParameterAdapter.get_recommended_timeframe(p.half_life)
            if rec_tf != args.timeframe:
                print(f"\n    💡 Consider using {rec_tf} timeframe for this pair")
    
    # Best candidates without signal
    no_signal = [p for p in tradeable if abs(p.current_zscore) < 2.0][:3]
    if no_signal:
        print("\n📋 WATCHLIST (waiting for signal):")
        for p in no_signal:
            adaptive = ParameterAdapter.adapt(
                half_life=p.half_life,
                hurst=p.hurst_exponent,
                coint_pvalue=p.eg_coint_pvalue
            )
            print(f"  • {p.pair[0]}/{p.pair[1]}: Z={p.current_zscore:+.2f}, Score={p.overall_score:.0f}")
            print(f"    Entry when Z reaches ±{adaptive.entry_zscore}")
            print(f"    Regime: {adaptive.regime}")
    
    # Pairs to avoid
    rejected = [p for p in result.selected_pairs if not p.is_tradeable][:3]
    if rejected:
        print("\n⚠️ PAIRS TO AVOID (failed validation):")
        for p in rejected:
            print(f"  • {p.pair[0]}/{p.pair[1]}: {', '.join(p.rejection_reasons[:2])}")
    
    # Timeframe recommendations
    if tradeable:
        print("\n📊 TIMEFRAME ANALYSIS:")
        for p in tradeable[:3]:
            rec_tf = ParameterAdapter.get_recommended_timeframe(p.half_life)
            trades_per_year = ParameterAdapter.estimate_trades_per_year(p.half_life, rec_tf)
            print(f"  • {p.pair[0]}/{p.pair[1]}:")
            print(f"    Optimal timeframe: {rec_tf}")
            print(f"    Estimated trades/year: ~{trades_per_year:.0f}")
            print(f"    Expected trade duration: {p.half_life*0.5:.0f} - {p.half_life*2:.0f} bars")
    
    # Save results
    if args.save:
        results_dir = Path("results/screening")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = results_dir / f"institutional_{datetime.now():%Y%m%d_%H%M%S}.json"
        save_results_json(result, filepath)
        print(f"\n✓ Results saved to: {filepath}")
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
