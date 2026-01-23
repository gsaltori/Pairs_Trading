"""
Run Session Trading System

Entry point for the production session-based system.
Asia Range → London Expansion strategy.
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from trading_system.session_orchestrator import SessionTradingSystem
from trading_system.config import SystemConfig, PathConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Session Trading System - Asia → London Expansion (R=2.5)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Strategy: Asia Range → London Expansion
- Asia session (00:00-06:00 UTC): Range establishment
- London session (07:00-11:00 UTC): Expansion phase
- Directional bias from Asia close position
- R/R: 2.5 (asymmetric payoff)

Examples:
  python run_session_system.py                     # Dry run (default)
  python run_session_system.py --live              # LIVE TRADING
  python run_session_system.py --status            # Check status
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
        default='session_system_data',
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
    
    system = SessionTradingSystem(config)
    
    if args.status:
        if system.initialize():
            import json
            status = system.get_status()
            print(json.dumps(status, indent=2, default=str))
        return
    
    print()
    print("╔" + "═" * 60 + "╗")
    print("║" + " SESSION TRADING SYSTEM ".center(60) + "║")
    print("║" + " Asia Range → London Expansion ".center(60) + "║")
    print("║" + " Liquidity-Based Directional Bias ".center(60) + "║")
    print("╚" + "═" * 60 + "╝")
    print()
    
    print("Session Timing (UTC):")
    print("  Asia:   00:00 - 06:00 (Range establishment)")
    print("  London: 07:00 - 11:00 (Expansion phase)")
    print()
    
    print("Trade Logic:")
    print("  - Bias from Asia close position vs midpoint")
    print("  - Entry: Break of Asia high (bull) / low (bear)")
    print("  - SL: Opposite side of Asia range")
    print("  - TP: 2.5 × risk (asymmetric payoff)")
    print("  - Time stop: End of London session")
    print()
    
    if args.live:
        print("⚠️  LIVE TRADING MODE - REAL MONEY AT RISK")
        print()
        print("   Risk: 0.5% per trade ($0.50 on $100)")
        print()
        print("   Expected frequency: ~2-6 trades/month")
        print("   (Most days have no valid setup)")
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
