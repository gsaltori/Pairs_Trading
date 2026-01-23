"""
Trading System Runner

Entry point for the production trading system.
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_system import TradingSystem, SystemConfig, PathConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Production Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_trading_system.py                    # Dry run mode (default)
  python run_trading_system.py --live             # LIVE TRADING
  python run_trading_system.py --status           # Check system status
  python run_trading_system.py --emergency-close  # Close all positions
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
        help='Print system status and exit'
    )
    
    parser.add_argument(
        '--emergency-close',
        action='store_true',
        help='Emergency: Close all positions'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='trading_system_data',
        help='Data directory for logs and state'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Verbose logging'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Build config
    config = SystemConfig(
        dry_run=not args.live,
        verbose=args.verbose,
    )
    
    # Override data directory if specified
    if args.data_dir:
        config.paths = PathConfig(base_dir=Path(args.data_dir))
    
    config.ensure_directories()
    
    # Create system
    system = TradingSystem(config)
    
    # Handle special commands
    if args.status:
        if system.initialize():
            import json
            print(json.dumps(system.get_status(), indent=2))
        return
    
    if args.emergency_close:
        print("⚠️  EMERGENCY CLOSE ALL POSITIONS")
        print("    This will close ALL positions managed by this system.")
        confirm = input("    Type 'CONFIRM' to proceed: ")
        if confirm == 'CONFIRM':
            if system.initialize():
                system.emergency_close_all()
        else:
            print("    Aborted.")
        return
    
    # Normal run
    print()
    print("=" * 60)
    if args.live:
        print("⚠️  LIVE TRADING MODE - REAL MONEY AT RISK")
    else:
        print("🔵 DRY RUN MODE - No real trades will be placed")
    print("=" * 60)
    print()
    
    if args.live:
        print("Starting in 5 seconds... Press Ctrl+C to cancel")
        import time
        time.sleep(5)
    
    system.run()


if __name__ == "__main__":
    main()
