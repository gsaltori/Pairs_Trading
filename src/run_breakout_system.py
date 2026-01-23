"""
Run Breakout Trading System

Entry point for the production breakout system.
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from trading_system.breakout_orchestrator import BreakoutTradingSystem
from trading_system.config import SystemConfig, PathConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Breakout Trading System - Range Compression Breakout (R=2.5)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Strategy: Range Compression → Expansion Breakout
- Compression: Range < 0.8 × ATR(14)
- Breakout: Close outside 6-bar range
- R/R: 2.5 (asymmetric payoff)

Examples:
  python run_breakout_system.py                     # Dry run (default)
  python run_breakout_system.py --live              # LIVE TRADING
  python run_breakout_system.py --status            # Check status
        """
    )
    
    parser.add_argument(
        '--live',
        action='store_true',
        help='Enable LIVE trading (default: dry run)'
    )
    
    parser.add_argument(
        '--status',
        action='store_true',
        help='Print status and exit'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='breakout_system_data',
        help='Data directory'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    config = SystemConfig(
        dry_run=not args.live,
        verbose=True,
    )
    
    if args.data_dir:
        config.paths = PathConfig(base_dir=Path(args.data_dir))
    
    config.ensure_directories()
    
    system = BreakoutTradingSystem(config)
    
    if args.status:
        if system.initialize():
            import json
            print(json.dumps(system.get_status(), indent=2, default=str))
        return
    
    print()
    print("╔" + "═" * 60 + "╗")
    print("║" + " BREAKOUT TRADING SYSTEM ".center(60) + "║")
    print("║" + " Range Compression → Expansion ".center(60) + "║")
    print("║" + " Target R/R: 2.5 ".center(60) + "║")
    print("╚" + "═" * 60 + "╝")
    print()
    
    print("Strategy Parameters:")
    print("  Range Lookback:        6 bars (24h on H4)")
    print("  Compression Threshold: < 0.8 × ATR(14)")
    print("  Risk/Reward:           2.5")
    print("  Breakeven Win Rate:    28.6%")
    print()
    
    if args.live:
        print("⚠️  LIVE TRADING MODE - REAL MONEY AT RISK")
        print()
        print("   Risk: 0.5% per trade ($0.50 on $100)")
        print()
        print("   Governors:")
        print("   - DD ≥ 3%: Risk reduced")
        print("   - DD ≥ 8%: SYSTEM HALT")
        print()
        print("Starting in 5 seconds... Press Ctrl+C to cancel")
        import time
        time.sleep(5)
    else:
        print("🔵 DRY RUN MODE - No real trades")
    
    print()
    
    system.run()


if __name__ == "__main__":
    main()
