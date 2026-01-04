"""
Pairs Trading System - Main Entry Point

Professional pairs trading system for Forex using IC Markets Global via MetaTrader 5.

Usage:
    python main.py screen --days 180
    python main.py backtest --pair EURUSD,GBPUSD --days 365
    python main.py optimize --pair EURUSD,GBPUSD --days 730
    python main.py paper --pair EURUSD,GBPUSD
    python main.py live --pair EURUSD,GBPUSD
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/pairs_trading.log')
    ]
)
logger = logging.getLogger(__name__)

# Imports
from config.settings import Settings, TradingMode, Timeframe
from config.broker_config import MT5Config
from src.data.broker_client import MT5Client, Timeframe as MT5Timeframe
from src.data.data_manager import DataManager
from src.analysis.correlation import CorrelationAnalyzer
from src.analysis.cointegration import CointegrationAnalyzer
from src.analysis.spread_builder import SpreadBuilder
from src.strategy.signals import SignalGenerator
from src.strategy.pairs_strategy import PairsStrategy
from src.risk.risk_manager import RiskManager
from src.backtest.backtest_engine import BacktestEngine
from src.optimization.optimizer import WalkForwardOptimizer
from src.execution.executor import LiveExecutor


class PairsTradingSystem:
    """
    Main orchestrator for the Pairs Trading System.
    
    Provides unified interface for:
    - Pair screening
    - Backtesting
    - Optimization
    - Paper trading
    - Live trading
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize the trading system."""
        self.settings = settings or Settings()
        
        # Lazy-loaded components
        self._client: Optional[MT5Client] = None
        self._data_manager: Optional[DataManager] = None
        self._risk_manager: Optional[RiskManager] = None
        self._strategy: Optional[PairsStrategy] = None
        self._executor: Optional[LiveExecutor] = None
    
    @property
    def client(self) -> MT5Client:
        """Get or create MT5 client."""
        if self._client is None:
            config = MT5Config.from_env()
            self._client = MT5Client(config)
            if not self._client.connect():
                raise ConnectionError("Failed to connect to MT5")
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
            balance = self.client.get_balance()
            self._risk_manager = RiskManager(self.settings, balance)
        return self._risk_manager
    
    @property
    def strategy(self) -> PairsStrategy:
        """Get or create strategy."""
        if self._strategy is None:
            self._strategy = PairsStrategy(self.settings, self.data_manager)
        return self._strategy
    
    def screen_pairs(
        self,
        symbols: Optional[List[str]] = None,
        timeframe: Optional[Timeframe] = None,
        days: int = 180
    ) -> List[Tuple[str, str, dict]]:
        """
        Screen for tradeable pairs.
        
        Args:
            symbols: List of symbols to analyze
            timeframe: Timeframe for analysis
            days: Days of history to analyze
            
        Returns:
            List of (symbol_a, symbol_b, metrics) tuples
        """
        symbols = symbols or self.settings.symbol_universe
        timeframe = timeframe or self.settings.timeframe
        
        logger.info(f"Screening {len(symbols)} symbols for pairs...")
        
        # Calculate bars needed
        bars_per_day = 24 if timeframe == Timeframe.H1 else (24 * 4 if timeframe == Timeframe.M15 else 24 * 2)
        count = days * bars_per_day
        
        # Convert timeframe
        mt5_tf = MT5Timeframe.from_string(timeframe.value)
        
        # Get data for all symbols
        symbol_data = {}
        for symbol in symbols:
            try:
                data = self.data_manager.get_close_prices(symbol, mt5_tf, count)
                if len(data) >= self.settings.backtest.min_bars_required:
                    symbol_data[symbol] = data
                    logger.debug(f"Loaded {len(data)} bars for {symbol}")
            except Exception as e:
                logger.warning(f"Failed to load {symbol}: {e}")
        
        logger.info(f"Loaded data for {len(symbol_data)} symbols")
        
        # Analyze all pairs
        corr_analyzer = CorrelationAnalyzer(window=self.settings.spread.correlation_window)
        coint_analyzer = CointegrationAnalyzer()
        spread_builder = SpreadBuilder(
            regression_window=self.settings.spread.regression_window,
            zscore_window=self.settings.spread.zscore_window
        )
        
        results = []
        symbols_list = list(symbol_data.keys())
        
        for i, symbol_a in enumerate(symbols_list):
            for symbol_b in symbols_list[i+1:]:
                price_a = symbol_data[symbol_a]
                price_b = symbol_data[symbol_b]
                
                # Align data
                common_idx = price_a.index.intersection(price_b.index)
                if len(common_idx) < self.settings.backtest.min_bars_required:
                    continue
                
                price_a = price_a.loc[common_idx]
                price_b = price_b.loc[common_idx]
                
                # Correlation analysis
                corr_result = corr_analyzer.analyze_pair(price_a, price_b)
                
                if corr_result.current_correlation < self.settings.spread.min_correlation:
                    continue
                
                # Cointegration test
                coint_result = coint_analyzer.engle_granger_test(price_a, price_b)
                
                if not coint_result.is_cointegrated:
                    continue
                
                # Spread metrics
                metrics = spread_builder.get_spread_metrics(price_a, price_b)
                
                if metrics is None:
                    continue
                
                if metrics.half_life > self.settings.spread.max_half_life:
                    continue
                
                # Calculate current z-score
                spread_data = spread_builder.build_spread_with_zscore(price_a, price_b)
                current_zscore = spread_data['zscore'].iloc[-1]
                
                results.append((symbol_a, symbol_b, {
                    'correlation': corr_result.current_correlation,
                    'stability': corr_result.stability_score,
                    'p_value': coint_result.p_value,
                    'hedge_ratio': coint_result.hedge_ratio,
                    'half_life': coint_result.half_life,
                    'hurst': metrics.hurst_exponent,
                    'current_zscore': current_zscore
                }))
        
        # Sort by correlation
        results.sort(key=lambda x: x[2]['correlation'], reverse=True)
        
        logger.info(f"Found {len(results)} tradeable pairs")
        return results
    
    def run_backtest(
        self,
        pair: Tuple[str, str],
        timeframe: Optional[Timeframe] = None,
        days: int = 365,
        save_results: bool = True
    ) -> dict:
        """
        Run backtest for a single pair.
        
        Args:
            pair: (symbol_a, symbol_b) tuple
            timeframe: Timeframe for analysis
            days: Days of history
            save_results: Whether to save results to file
            
        Returns:
            Backtest results dictionary
        """
        timeframe = timeframe or self.settings.timeframe
        
        logger.info(f"Running backtest for {pair[0]}/{pair[1]}...")
        
        # Calculate bars
        bars_per_day = 24 if timeframe == Timeframe.H1 else 96
        count = days * bars_per_day
        
        mt5_tf = MT5Timeframe.from_string(timeframe.value)
        
        # Get data
        price_a, price_b = self.data_manager.get_pair_data(
            pair[0], pair[1], mt5_tf, count
        )
        
        if len(price_a) < self.settings.backtest.min_bars_required:
            logger.error(f"Insufficient data: {len(price_a)} bars")
            return {}
        
        # Run backtest
        engine = BacktestEngine(self.settings)
        result = engine.run_backtest(pair, price_a, price_b)
        
        # Save results
        if save_results and result:
            filepath = Path(self.settings.paths.backtest_dir) / f"{pair[0]}_{pair[1]}_{datetime.now():%Y%m%d_%H%M%S}.json"
            engine.save_results(result, str(filepath))
            logger.info(f"Results saved to {filepath}")
        
        return result.__dict__ if result else {}
    
    def run_optimization(
        self,
        pair: Tuple[str, str],
        timeframe: Optional[Timeframe] = None,
        days: int = 730
    ) -> dict:
        """
        Run walk-forward optimization.
        
        Args:
            pair: (symbol_a, symbol_b) tuple
            timeframe: Timeframe for analysis
            days: Days of history
            
        Returns:
            Optimization results
        """
        timeframe = timeframe or self.settings.timeframe
        
        logger.info(f"Running optimization for {pair[0]}/{pair[1]}...")
        
        # Calculate bars
        bars_per_day = 24 if timeframe == Timeframe.H1 else 96
        count = days * bars_per_day
        
        mt5_tf = MT5Timeframe.from_string(timeframe.value)
        
        # Get data
        price_a, price_b = self.data_manager.get_pair_data(
            pair[0], pair[1], mt5_tf, count
        )
        
        if len(price_a) < 1000:
            logger.error("Insufficient data for optimization")
            return {}
        
        # Run optimization
        optimizer = WalkForwardOptimizer(self.settings)
        result = optimizer.optimize(pair, price_a, price_b)
        
        # Save results
        filepath = Path(self.settings.paths.optimization_dir) / f"opt_{pair[0]}_{pair[1]}_{datetime.now():%Y%m%d}.json"
        optimizer.save_results(result, str(filepath))
        logger.info(f"Optimization results saved to {filepath}")
        
        return {
            'best_params': result.best_params.__dict__ if result.best_params else {},
            'efficiency_ratio': result.efficiency_ratio,
            'is_sharpe': result.is_sharpe,
            'oos_sharpe': result.oos_sharpe,
            'total_trades': result.total_trades
        }
    
    def run_paper_trading(
        self,
        pairs: List[Tuple[str, str]],
        check_interval: int = 60
    ):
        """
        Run paper trading session.
        
        Args:
            pairs: List of pairs to trade
            check_interval: Seconds between checks
        """
        logger.info("Starting paper trading session...")
        
        self.settings.mode = TradingMode.PAPER
        
        config = MT5Config.from_env()
        executor = LiveExecutor(self.settings, config, self.risk_manager)
        
        try:
            executor.start()
            
            print("\n" + "="*60)
            print("PAPER TRADING STARTED")
            print("Press Ctrl+C to stop")
            print("="*60 + "\n")
            
            import time
            
            while True:
                self._trading_loop_iteration(pairs, executor)
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            executor.stop()
            
            # Print summary
            history = executor.get_trade_history()
            if not history.empty:
                print(f"\nTrades: {len(history)}")
                print(f"Total P&L: ${history['pnl'].sum():.2f}")
    
    def run_live_trading(
        self,
        pairs: List[Tuple[str, str]],
        check_interval: int = 60
    ):
        """
        Run live trading session.
        
        Args:
            pairs: List of pairs to trade
            check_interval: Seconds between checks
        """
        # Safety confirmation
        print("\n" + "!"*60)
        print("WARNING: LIVE TRADING MODE")
        print("This will execute REAL trades with REAL money!")
        print("!"*60)
        
        confirm = input("\nType 'CONFIRM' to proceed: ")
        if confirm != "CONFIRM":
            print("Cancelled.")
            return
        
        logger.info("Starting LIVE trading session...")
        
        self.settings.mode = TradingMode.LIVE
        
        config = MT5Config.from_env()
        executor = LiveExecutor(self.settings, config, self.risk_manager)
        
        try:
            executor.start()
            
            account = self.client.get_account_info()
            print(f"\nAccount: {account.get('login')}")
            print(f"Balance: ${account.get('balance', 0):,.2f}")
            print(f"Equity: ${account.get('equity', 0):,.2f}")
            
            print("\n" + "="*60)
            print("LIVE TRADING STARTED")
            print("Press Ctrl+C to stop")
            print("="*60 + "\n")
            
            import time
            
            while True:
                self._trading_loop_iteration(pairs, executor)
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            print("\nStopping...")
            
            # Offer to close positions
            if executor.positions:
                close = input("Close all positions? (yes/no): ")
                if close.lower() == 'yes':
                    executor.close_all_positions("Session end")
        finally:
            executor.stop()
    
    def _trading_loop_iteration(
        self,
        pairs: List[Tuple[str, str]],
        executor: LiveExecutor
    ):
        """Single iteration of the trading loop."""
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n[{current_time}]")
        print("-" * 40)
        
        for pair in pairs:
            try:
                # Analyze pair
                analysis = self.strategy.analyze_pair(pair)
                
                if analysis is None:
                    continue
                
                # Get current state
                zscore = analysis.spread_metrics.zscore if analysis.spread_metrics else 0
                corr = analysis.correlation_result.current_correlation if analysis.correlation_result else 0
                
                # Check position
                in_position = pair in executor.positions
                
                status = f"{pair[0]}/{pair[1]}: Z={zscore:+.2f}, Corr={corr:.2f}"
                
                if in_position:
                    pos = executor.positions[pair]
                    status += f" [POS: {pos.direction}, PnL=${pos.unrealized_pnl:.2f}]"
                
                print(status)
                
                # Execute signal
                signal = analysis.current_signal
                if signal and signal.type.value != 'no_signal':
                    success, msg = executor.execute_signal(signal)
                    action = "✓" if success else "✗"
                    print(f"  {action} {signal.type.value}: {msg}")
                
            except Exception as e:
                logger.error(f"Error processing {pair}: {e}")
        
        # Update positions
        executor.update_positions()
        
        # Show summary
        state = executor.get_state()
        print(f"\nPositions: {len(state.open_positions)} | Daily P&L: ${state.daily_pnl:.2f}")
    
    def shutdown(self):
        """Clean shutdown of all components."""
        if self._client:
            self._client.disconnect()
        logger.info("System shutdown complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Pairs Trading System for IC Markets Global via MT5'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Screen command
    screen_parser = subparsers.add_parser('screen', help='Screen for tradeable pairs')
    screen_parser.add_argument('--days', type=int, default=180, help='Days of history')
    screen_parser.add_argument('--timeframe', type=str, default='H1', help='Timeframe')
    
    # Backtest command
    bt_parser = subparsers.add_parser('backtest', help='Run backtest')
    bt_parser.add_argument('--pair', type=str, required=True, help='Pair (e.g., EURUSD,GBPUSD)')
    bt_parser.add_argument('--days', type=int, default=365, help='Days of history')
    bt_parser.add_argument('--timeframe', type=str, default='H1', help='Timeframe')
    
    # Optimize command
    opt_parser = subparsers.add_parser('optimize', help='Run optimization')
    opt_parser.add_argument('--pair', type=str, required=True, help='Pair')
    opt_parser.add_argument('--days', type=int, default=730, help='Days of history')
    
    # Paper trading command
    paper_parser = subparsers.add_parser('paper', help='Run paper trading')
    paper_parser.add_argument('--pair', type=str, required=True, help='Pairs to trade')
    paper_parser.add_argument('--interval', type=int, default=60, help='Check interval (seconds)')
    
    # Live trading command
    live_parser = subparsers.add_parser('live', help='Run live trading')
    live_parser.add_argument('--pair', type=str, required=True, help='Pairs to trade')
    live_parser.add_argument('--interval', type=int, default=60, help='Check interval (seconds)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Create system
    system = PairsTradingSystem()
    
    try:
        if args.command == 'screen':
            tf = Timeframe(args.timeframe) if args.timeframe else None
            results = system.screen_pairs(timeframe=tf, days=args.days)
            
            print("\n" + "="*80)
            print("PAIR SCREENING RESULTS")
            print("="*80)
            
            for symbol_a, symbol_b, metrics in results[:20]:
                print(f"\n{symbol_a}/{symbol_b}:")
                print(f"  Correlation: {metrics['correlation']:.3f} (stability: {metrics['stability']:.2f})")
                print(f"  Cointegration p-value: {metrics['p_value']:.4f}")
                print(f"  Hedge ratio: {metrics['hedge_ratio']:.4f}")
                print(f"  Half-life: {metrics['half_life']:.1f} bars")
                print(f"  Current Z-score: {metrics['current_zscore']:+.2f}")
        
        elif args.command == 'backtest':
            pair = tuple(args.pair.split(','))
            tf = Timeframe(args.timeframe) if args.timeframe else None
            
            result = system.run_backtest(pair, tf, args.days)
            
            if result:
                print("\n" + "="*60)
                print("BACKTEST RESULTS")
                print("="*60)
                print(f"Total Return: {result.get('total_return', 0):.2%}")
                print(f"Sharpe Ratio: {result.get('sharpe_ratio', 0):.2f}")
                print(f"Max Drawdown: {result.get('max_drawdown', 0):.2%}")
                print(f"Win Rate: {result.get('win_rate', 0):.1%}")
                print(f"Total Trades: {result.get('total_trades', 0)}")
        
        elif args.command == 'optimize':
            pair = tuple(args.pair.split(','))
            
            result = system.run_optimization(pair, days=args.days)
            
            if result:
                print("\n" + "="*60)
                print("OPTIMIZATION RESULTS")
                print("="*60)
                print(f"Best Parameters: {result.get('best_params', {})}")
                print(f"Efficiency Ratio: {result.get('efficiency_ratio', 0):.2f}")
                print(f"IS Sharpe: {result.get('is_sharpe', 0):.2f}")
                print(f"OOS Sharpe: {result.get('oos_sharpe', 0):.2f}")
        
        elif args.command == 'paper':
            pairs = [tuple(p.split(',')) for p in args.pair.split(';')]
            system.run_paper_trading(pairs, args.interval)
        
        elif args.command == 'live':
            pairs = [tuple(p.split(',')) for p in args.pair.split(';')]
            system.run_live_trading(pairs, args.interval)
    
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise
    finally:
        system.shutdown()


if __name__ == '__main__':
    # Create logs directory
    Path('logs').mkdir(exist_ok=True)
    main()
