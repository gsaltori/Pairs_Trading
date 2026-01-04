"""
Example: Run paper trading session.

This script demonstrates how to:
1. Set up a paper trading session
2. Monitor positions in real-time
3. Execute signals automatically
4. Track performance
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import time
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import Settings, TradingMode
from config.broker_config import BrokerConfig
from src.data.broker_client import OandaClient
from src.data.data_manager import DataManager
from src.strategy.pairs_strategy import PairsStrategy
from src.risk.risk_manager import RiskManager
from src.execution.executor import LiveExecutor


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/paper_trading.log')
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Run paper trading example."""
    print("="*60)
    print("PAIRS TRADING - PAPER TRADING")
    print("="*60)
    
    # Configuration
    pairs = [
        ('EUR_USD', 'GBP_USD'),
        ('AUD_USD', 'NZD_USD')
    ]
    check_interval = 60  # seconds
    
    print(f"\nPairs to trade:")
    for pair in pairs:
        print(f"  - {pair[0]}/{pair[1]}")
    print(f"Check interval: {check_interval} seconds")
    
    # Initialize settings
    settings = Settings()
    settings.mode = TradingMode.PAPER
    
    # Load broker config
    try:
        broker_config = BrokerConfig.from_env()
        client = OandaClient(broker_config)
        print("\n✓ Connected to OANDA")
        
        # Verify account
        account = client.get_account_summary()
        print(f"✓ Account: {account.get('id', 'N/A')}")
        print(f"✓ Balance: ${float(account.get('balance', 0)):,.2f}")
        
    except Exception as e:
        print(f"\nERROR: Could not connect to OANDA: {e}")
        return
    
    # Initialize components
    data_manager = DataManager(client, settings.paths.cache_dir)
    risk_manager = RiskManager(settings, float(account.get('balance', 10000)))
    strategy = PairsStrategy(settings, data_manager)
    executor = LiveExecutor(settings, broker_config, risk_manager)
    
    # Start executor
    executor.start()
    
    print("\n" + "-"*60)
    print("PAPER TRADING STARTED")
    print("Press Ctrl+C to stop")
    print("-"*60 + "\n")
    
    iteration = 0
    
    try:
        while True:
            iteration += 1
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            print(f"\n[{current_time}] Iteration {iteration}")
            print("-" * 40)
            
            for pair in pairs:
                try:
                    # Analyze pair
                    logger.info(f"Analyzing {pair[0]}/{pair[1]}...")
                    analysis = strategy.analyze_pair(pair)
                    
                    if analysis is None:
                        logger.warning(f"Could not analyze {pair[0]}/{pair[1]}")
                        continue
                    
                    # Get signal
                    signal = analysis.current_signal
                    
                    # Display status
                    spread_data = analysis.spread_metrics
                    zscore = spread_data.zscore if spread_data else 0
                    corr = analysis.correlation_result.current_correlation if analysis.correlation_result else 0
                    
                    status = f"{pair[0]}/{pair[1]}: Z={zscore:+.2f}, Corr={corr:.2f}"
                    
                    # Check if in position
                    in_position = pair in executor.positions
                    
                    if in_position:
                        pos = executor.positions[pair]
                        status += f" [POSITION: {pos.direction}, PnL=${pos.unrealized_pnl:.2f}]"
                    
                    print(status)
                    
                    # Execute signals
                    if signal and signal.type.value != 'no_signal':
                        logger.info(f"Signal detected: {signal.type.value} for {pair}")
                        
                        success, msg = executor.execute_signal(signal)
                        
                        if success:
                            logger.info(f"✓ Executed: {msg}")
                            print(f"  → {signal.type.value.upper()}: {msg}")
                        else:
                            logger.warning(f"✗ Failed: {msg}")
                            print(f"  → FAILED: {msg}")
                    
                except Exception as e:
                    logger.error(f"Error processing {pair}: {e}")
                    continue
            
            # Update positions
            executor.update_positions()
            
            # Display summary
            state = executor.get_state()
            print(f"\nPositions: {len(state.open_positions)}")
            print(f"Daily trades: {state.daily_trades}")
            print(f"Daily P/L: ${state.daily_pnl:.2f}")
            
            # Wait for next iteration
            print(f"\nNext check in {check_interval}s...")
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        print("\n\n" + "="*60)
        print("STOPPING PAPER TRADING")
        print("="*60)
        
        # Close positions?
        if executor.positions:
            print(f"\nOpen positions: {len(executor.positions)}")
            for pair, pos in executor.positions.items():
                print(f"  - {pair[0]}/{pair[1]}: {pos.direction}, PnL=${pos.unrealized_pnl:.2f}")
            
            close = input("\nClose all positions? (yes/no): ")
            if close.lower() == 'yes':
                results = executor.close_all_positions()
                for pair, (success, msg) in results.items():
                    print(f"  {pair[0]}/{pair[1]}: {msg}")
    
    finally:
        executor.stop()
        
        # Final summary
        print("\n" + "="*60)
        print("SESSION SUMMARY")
        print("="*60)
        
        history = executor.get_trade_history()
        
        if len(history) > 0:
            print(f"\nTotal trades: {len(history)}")
            wins = len(history[history['pnl'] > 0])
            print(f"Win rate: {wins/len(history):.1%}")
            print(f"Total P/L: ${history['pnl'].sum():.2f}")
            
            print("\nTrade Log:")
            print(history.to_string())
        else:
            print("\nNo trades executed this session.")
        
        print("\n" + "="*60)
        print("PAPER TRADING COMPLETE")
        print("="*60)


if __name__ == '__main__':
    main()
