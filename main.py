"""
Main entry point for the Pairs Trading System.

This module provides the main orchestration logic for:
- Backtesting mode
- Paper trading mode
- Live trading mode
- Walk-forward optimization
"""

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.settings import Settings, TradingMode, Timeframe
from config.broker_config import BrokerConfig

from src.data.broker_client import OandaClient
from src.data.data_manager import DataManager
from src.analysis.correlation import CorrelationAnalyzer
from src.analysis.cointegration import CointegrationAnalyzer
from src.analysis.spread_builder import SpreadBuilder
from src.strategy.signals import SignalGenerator
from src.strategy.pairs_strategy import PairsStrategy
from src.risk.risk_manager import RiskManager
from src.backtest.backtest_engine import BacktestEngine
from src.optimization.optimizer import WalkForwardOptimizer, GridSearchOptimizer
from src.execution.executor import LiveExecutor


# Configure logging
def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None):
    """Setup logging configuration."""
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


class PairsTradingSystem:
    """
    Main orchestration class for the Pairs Trading System.
    
    Supports multiple operation modes:
    - Backtesting: Test strategy on historical data
    - Paper trading: Test with live data but simulated execution
    - Live trading: Real money execution
    - Optimization: Walk-forward parameter optimization
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        broker_config: Optional[BrokerConfig] = None
    ):
        """
        Initialize the trading system.
        
        Args:
            settings: Trading settings (loads default if None)
            broker_config: Broker configuration (loads from env if None)
        """
        self.settings = settings or Settings()
        self.broker_config = broker_config
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize components (lazy loading)
        self._client: Optional[OandaClient] = None
        self._data_manager: Optional[DataManager] = None
        self._risk_manager: Optional[RiskManager] = None
        self._strategy: Optional[PairsStrategy] = None
        self._executor: Optional[LiveExecutor] = None
        
        self.logger.info("PairsTradingSystem initialized")
    
    @property
    def client(self) -> OandaClient:
        """Get or create broker client."""
        if self._client is None:
            if self.broker_config is None:
                self.broker_config = BrokerConfig.from_env()
            self._client = OandaClient(self.broker_config)
        return self._client
    
    @property
    def data_manager(self) -> DataManager:
        """Get or create data manager."""
        if self._data_manager is None:
            self._data_manager = DataManager(
                client=self.client,
                cache_dir=self.settings.paths.cache_dir
            )
        return self._data_manager
    
    @property
    def risk_manager(self) -> RiskManager:
        """Get or create risk manager."""
        if self._risk_manager is None:
            self._risk_manager = RiskManager(
                settings=self.settings,
                initial_balance=self.settings.backtest.initial_capital
            )
        return self._risk_manager
    
    @property
    def strategy(self) -> PairsStrategy:
        """Get or create strategy."""
        if self._strategy is None:
            self._strategy = PairsStrategy(
                settings=self.settings,
                data_manager=self.data_manager
            )
        return self._strategy
    
    @property
    def executor(self) -> LiveExecutor:
        """Get or create executor."""
        if self._executor is None:
            self._executor = LiveExecutor(
                settings=self.settings,
                broker_config=self.broker_config,
                risk_manager=self.risk_manager
            )
        return self._executor
    
    def run_backtest(
        self,
        pair: Tuple[str, str],
        start_date: datetime,
        end_date: datetime,
        timeframe: Optional[Timeframe] = None
    ):
        """
        Run backtest on a single pair.
        
        Args:
            pair: Tuple of (instrument_a, instrument_b)
            start_date: Backtest start date
            end_date: Backtest end date
            timeframe: Timeframe (uses settings default if None)
            
        Returns:
            BacktestResult
        """
        tf = timeframe or self.settings.timeframe
        
        self.logger.info(f"Running backtest for {pair[0]}/{pair[1]} "
                        f"from {start_date} to {end_date}")
        
        # Fetch data
        data_a = self.data_manager.fetch_data(
            instrument=pair[0],
            timeframe=tf,
            start_date=start_date,
            end_date=end_date
        )
        
        data_b = self.data_manager.fetch_data(
            instrument=pair[1],
            timeframe=tf,
            start_date=start_date,
            end_date=end_date
        )
        
        if data_a is None or data_b is None:
            self.logger.error("Failed to fetch data for backtest")
            return None
        
        # Run backtest
        engine = BacktestEngine(self.settings, self.data_manager)
        result = engine.run_backtest(pair, data_a, data_b)
        
        # Save results
        result_path = self.settings.paths.backtest_results / f"backtest_{pair[0]}_{pair[1]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        engine.save_results(result, str(result_path))
        
        # Print summary
        print(result.summary())
        
        return result
    
    def run_multi_pair_backtest(
        self,
        pairs: Optional[List[Tuple[str, str]]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[Tuple[str, str], any]:
        """
        Run backtest on multiple pairs.
        
        Args:
            pairs: List of pairs (uses settings default if None)
            start_date: Start date (1 year ago if None)
            end_date: End date (now if None)
            
        Returns:
            Dict mapping pairs to BacktestResults
        """
        pairs = pairs or self.settings.pairs
        end_date = end_date or datetime.now()
        start_date = start_date or (end_date - timedelta(days=365))
        
        results = {}
        
        for pair in pairs:
            self.logger.info(f"Backtesting {pair[0]}/{pair[1]}...")
            result = self.run_backtest(pair, start_date, end_date)
            if result:
                results[pair] = result
        
        # Summary
        self._print_multi_pair_summary(results)
        
        return results
    
    def run_optimization(
        self,
        pair: Tuple[str, str],
        start_date: datetime,
        end_date: datetime,
        objective: str = 'sharpe'
    ):
        """
        Run walk-forward optimization.
        
        Args:
            pair: Pair to optimize
            start_date: Data start date
            end_date: Data end date
            objective: Optimization objective
            
        Returns:
            OptimizationResult
        """
        self.logger.info(f"Running walk-forward optimization for {pair[0]}/{pair[1]}")
        
        # Fetch data
        data_a = self.data_manager.fetch_data(
            instrument=pair[0],
            timeframe=self.settings.timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        data_b = self.data_manager.fetch_data(
            instrument=pair[1],
            timeframe=self.settings.timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        if data_a is None or data_b is None:
            self.logger.error("Failed to fetch data for optimization")
            return None
        
        # Run optimization
        optimizer = WalkForwardOptimizer(self.settings, objective)
        result = optimizer.optimize(pair, data_a, data_b)
        
        # Save results
        result_path = self.settings.paths.optimization_results / f"opt_{pair[0]}_{pair[1]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        optimizer.save_results(result, str(result_path))
        
        # Print summary
        print(result.summary())
        
        return result
    
    def screen_pairs(
        self,
        instruments: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Screen instruments to find the best pairs.
        
        Args:
            instruments: List of instruments to screen
            start_date: Analysis start date
            end_date: Analysis end date
            
        Returns:
            DataFrame with pair analysis results
        """
        if instruments is None:
            instruments = [
                'EUR_USD', 'GBP_USD', 'USD_JPY', 'USD_CHF',
                'AUD_USD', 'NZD_USD', 'EUR_GBP', 'EUR_JPY'
            ]
        
        end_date = end_date or datetime.now()
        start_date = start_date or (end_date - timedelta(days=180))
        
        self.logger.info(f"Screening {len(instruments)} instruments for pairs")
        
        # Fetch data for all instruments
        data = {}
        for inst in instruments:
            df = self.data_manager.fetch_data(
                instrument=inst,
                timeframe=self.settings.timeframe,
                start_date=start_date,
                end_date=end_date
            )
            if df is not None and len(df) > 100:
                data[inst] = df
        
        self.logger.info(f"Loaded data for {len(data)} instruments")
        
        # Analyze all pairs
        results = []
        
        correlation_analyzer = CorrelationAnalyzer(
            window=self.settings.spread.regression_window
        )
        cointegration_analyzer = CointegrationAnalyzer()
        spread_builder = SpreadBuilder(
            regression_window=self.settings.spread.regression_window,
            zscore_window=self.settings.spread.zscore_window
        )
        
        instruments_list = list(data.keys())
        
        for i, inst_a in enumerate(instruments_list):
            for inst_b in instruments_list[i+1:]:
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
                    
                    # Cointegration
                    coint_result = cointegration_analyzer.engle_granger_test(
                        price_a, price_b
                    )
                    
                    # Spread metrics
                    spread_metrics = spread_builder.get_spread_metrics(price_a, price_b)
                    
                    results.append({
                        'pair': f"{inst_a}/{inst_b}",
                        'instrument_a': inst_a,
                        'instrument_b': inst_b,
                        'correlation': corr_result.current_correlation,
                        'correlation_stability': corr_result.stability_score,
                        'cointegration_pvalue': coint_result.p_value,
                        'is_cointegrated': coint_result.is_cointegrated,
                        'hedge_ratio': coint_result.hedge_ratio,
                        'half_life': coint_result.half_life,
                        'hurst_exponent': spread_metrics.hurst_exponent if spread_metrics else None,
                        'current_zscore': spread_metrics.zscore if spread_metrics else None
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Error analyzing {inst_a}/{inst_b}: {e}")
                    continue
        
        # Convert to DataFrame and sort
        df = pd.DataFrame(results)
        
        if len(df) > 0:
            # Filter and sort
            df = df[
                (df['correlation'] >= self.settings.spread.min_correlation) &
                (df['is_cointegrated'] == True) &
                (df['half_life'] <= self.settings.spread.max_half_life)
            ].copy()
            
            df = df.sort_values('correlation_stability', ascending=False)
        
        self.logger.info(f"Found {len(df)} tradeable pairs")
        
        return df
    
    def run_paper_trading(
        self,
        pairs: Optional[List[Tuple[str, str]]] = None,
        duration_hours: int = 24
    ):
        """
        Run paper trading session.
        
        Args:
            pairs: Pairs to trade
            duration_hours: How long to run
        """
        pairs = pairs or self.settings.pairs
        
        self.logger.info(f"Starting paper trading session for {duration_hours} hours")
        self.logger.info(f"Trading pairs: {pairs}")
        
        self.settings.mode = TradingMode.PAPER
        
        # Initialize components
        self.executor.start()
        
        end_time = datetime.now() + timedelta(hours=duration_hours)
        
        try:
            while datetime.now() < end_time:
                self._trading_loop_iteration(pairs)
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            self.logger.info("Paper trading stopped by user")
        finally:
            self.executor.stop()
            self._print_trading_summary()
    
    def run_live_trading(
        self,
        pairs: Optional[List[Tuple[str, str]]] = None
    ):
        """
        Run live trading.
        
        WARNING: This uses real money!
        
        Args:
            pairs: Pairs to trade
        """
        pairs = pairs or self.settings.pairs
        
        # Safety confirmation
        print("\n" + "="*60)
        print("WARNING: LIVE TRADING MODE")
        print("This will execute real trades with real money!")
        print("="*60)
        confirm = input("Type 'CONFIRM' to proceed: ")
        
        if confirm != 'CONFIRM':
            print("Live trading cancelled.")
            return
        
        self.logger.info("Starting LIVE trading session")
        self.settings.mode = TradingMode.LIVE
        
        # Verify account
        try:
            account = self.client.get_account_summary()
            self.logger.info(f"Account: {account.get('id')}")
            self.logger.info(f"Balance: {account.get('balance')}")
        except Exception as e:
            self.logger.error(f"Could not verify account: {e}")
            return
        
        self.executor.start()
        
        try:
            while True:
                self._trading_loop_iteration(pairs)
                time.sleep(60)
                
        except KeyboardInterrupt:
            self.logger.info("Live trading stopped by user")
        finally:
            # Emergency close option
            if self.executor.positions:
                close = input("Close all positions? (yes/no): ")
                if close.lower() == 'yes':
                    self.executor.close_all_positions()
            
            self.executor.stop()
            self._print_trading_summary()
    
    def _trading_loop_iteration(self, pairs: List[Tuple[str, str]]):
        """Single iteration of the trading loop."""
        for pair in pairs:
            try:
                # Analyze pair
                analysis = self.strategy.analyze_pair(pair)
                
                if analysis is None:
                    continue
                
                signal = analysis.current_signal
                
                if signal and signal.type != 'no_signal':
                    self.logger.info(f"Signal: {signal.type} for {pair}")
                    
                    # Execute signal
                    success, msg = self.executor.execute_signal(signal)
                    
                    if success:
                        self.logger.info(f"Executed: {msg}")
                    else:
                        self.logger.warning(f"Failed: {msg}")
                        
            except Exception as e:
                self.logger.error(f"Error in trading loop for {pair}: {e}")
        
        # Update positions
        self.executor.update_positions()
    
    def _print_multi_pair_summary(self, results: Dict):
        """Print summary of multi-pair backtest."""
        print("\n" + "="*60)
        print("MULTI-PAIR BACKTEST SUMMARY")
        print("="*60)
        
        for pair, result in results.items():
            print(f"\n{pair[0]}/{pair[1]}:")
            print(f"  Return: {result.total_return:.2%}")
            print(f"  Sharpe: {result.sharpe_ratio:.2f}")
            print(f"  Max DD: {result.max_drawdown:.2%}")
            print(f"  Trades: {result.total_trades}")
            print(f"  Win Rate: {result.win_rate:.2%}")
        
        # Aggregate
        if results:
            avg_sharpe = sum(r.sharpe_ratio for r in results.values()) / len(results)
            total_return = sum(r.total_return for r in results.values())
            
            print("\n" + "-"*40)
            print(f"Average Sharpe: {avg_sharpe:.2f}")
            print(f"Combined Return: {total_return:.2%}")
        
        print("="*60)
    
    def _print_trading_summary(self):
        """Print trading session summary."""
        state = self.executor.get_state()
        history = self.executor.get_trade_history()
        
        print("\n" + "="*60)
        print("TRADING SESSION SUMMARY")
        print("="*60)
        print(f"Mode: {state.mode.value}")
        print(f"Daily Trades: {state.daily_trades}")
        print(f"Daily P/L: ${state.daily_pnl:.2f}")
        print(f"Open Positions: {len(state.open_positions)}")
        
        if len(history) > 0:
            print(f"\nTotal Trades: {len(history)}")
            wins = len(history[history['pnl'] > 0])
            print(f"Win Rate: {wins/len(history):.2%}")
            print(f"Total P/L: ${history['pnl'].sum():.2f}")
        
        print("="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Pairs Trading System')
    
    parser.add_argument(
        'mode',
        choices=['backtest', 'optimize', 'screen', 'paper', 'live'],
        help='Operation mode'
    )
    
    parser.add_argument(
        '--pair',
        type=str,
        default='EUR_USD,GBP_USD',
        help='Trading pair (e.g., EUR_USD,GBP_USD)'
    )
    
    parser.add_argument(
        '--days',
        type=int,
        default=365,
        help='Number of days of history'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to settings YAML file'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        help='Log file path'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    # Load settings
    if args.config:
        settings = Settings.load(args.config)
    else:
        settings = Settings()
    
    # Parse pair
    pair_parts = args.pair.split(',')
    if len(pair_parts) == 2:
        pair = (pair_parts[0].strip(), pair_parts[1].strip())
    else:
        print("Invalid pair format. Use: INST_A,INST_B")
        return
    
    # Calculate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    # Initialize system
    try:
        broker_config = BrokerConfig.from_env()
    except Exception as e:
        logging.warning(f"Could not load broker config: {e}")
        broker_config = None
    
    system = PairsTradingSystem(settings, broker_config)
    
    # Execute mode
    if args.mode == 'backtest':
        system.run_backtest(pair, start_date, end_date)
        
    elif args.mode == 'optimize':
        system.run_optimization(pair, start_date, end_date)
        
    elif args.mode == 'screen':
        df = system.screen_pairs(start_date=start_date, end_date=end_date)
        print("\nTop Pairs:")
        print(df.to_string())
        
    elif args.mode == 'paper':
        system.run_paper_trading(pairs=[pair])
        
    elif args.mode == 'live':
        system.run_live_trading(pairs=[pair])


if __name__ == '__main__':
    main()
