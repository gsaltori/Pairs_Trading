"""
Run Multi-Strategy Portfolio Trading System

Entry point for the production multi-strategy system.
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from trading_system import PortfolioOrchestrator, SystemConfig, PathConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-Strategy Portfolio Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_portfolio_system.py                    # Dry run (default)
  python run_portfolio_system.py --live             # LIVE TRADING
  python run_portfolio_system.py --status           # Check status
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
        default='portfolio_system_data',
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
    
    system = PortfolioOrchestrator(config)
    
    if args.status:
        if system.initialize():
            import json
            print(json.dumps(system.get_status(), indent=2))
        return
    
    print()
    print("╔" + "═" * 60 + "╗")
    print("║" + " MULTI-STRATEGY PORTFOLIO SYSTEM ".center(60) + "║")
    print("╚" + "═" * 60 + "╝")
    print()
    
    if args.live:
        print("⚠️  LIVE TRADING MODE - REAL MONEY AT RISK")
        print()
        print("   Risk Limits:")
        print("   - Trend Continuation: 0.30%")
        print("   - Trend Pullback:     0.25%")
        print("   - Volatility Exp:     0.20%")
        print("   - Total max:          0.75%")
        print()
        print("   Governors:")
        print("   - DD ≥ 5%: Risk reduced 50%")
        print("   - DD ≥ 8%: SYSTEM HALT")
        print("   - 3 losses: 24h cooling off")
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
