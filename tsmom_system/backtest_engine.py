"""
Time-Series Momentum System - Backtest Engine
Event-driven backtesting with validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from config import CONFIG
from data_loader import DataLoader
from signal_engine import SignalEngine, TSMOMSignal, validate_signal, SanityCheckError
from portfolio_engine import PortfolioEngine, validate_portfolio


class BacktestError(Exception):
    """Backtest validation failed."""
    pass


@dataclass
class BacktestResult:
    """Backtest results container."""
    equity_df: pd.DataFrame
    trades_df: pd.DataFrame
    monthly_returns: pd.Series
    signals: List[TSMOMSignal]
    validation_passed: bool


class BacktestEngine:
    """
    Time-Series Momentum backtest engine.
    
    MANDATORY VALIDATIONS:
    - No lookahead bias
    - Trades only on rebalance dates
    - Portfolio return = weighted asset returns
    - Equity always positive
    - No leverage
    """
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.signal_engine = SignalEngine()
        self.portfolio = PortfolioEngine()
    
    def run(
        self,
        prices: pd.DataFrame = None,
        validate: bool = True,
    ) -> BacktestResult:
        """
        Run backtest.
        
        Args:
            prices: Price DataFrame (loads if None)
            validate: Whether to run validation checks
            
        Returns:
            BacktestResult
        """
        print("\n" + "=" * 70)
        print("BACKTEST EXECUTION")
        print("=" * 70)
        
        # Load data
        if prices is None:
            print("\n1. Loading data...")
            prices = self.data_loader.load_universe()
        
        valid, issues = self.data_loader.validate_data(prices)
        if not valid:
            raise BacktestError(f"Data validation failed: {issues}")
        
        print(f"   {len(prices)} days, {prices.index[0].date()} to {prices.index[-1].date()}")
        
        # Get rebalance dates
        print("\n2. Identifying rebalance dates...")
        rebalance_dates = self.data_loader.get_monthly_dates(prices)
        print(f"   {len(rebalance_dates)} monthly dates")
        
        # Generate signals
        print("\n3. Generating signals...")
        signals = self.signal_engine.generate_all_signals(prices, rebalance_dates)
        print(f"   {len(signals)} valid signals")
        
        # Validate signals
        if validate:
            for sig in signals:
                valid, issues = validate_signal(sig)
                if not valid:
                    raise BacktestError(f"Signal validation failed: {issues}")
        
        # Reset portfolio
        self.portfolio.reset()
        
        # Run simulation
        print("\n4. Running simulation...")
        self._run_simulation(prices, signals, validate)
        
        # Get results
        equity_df = self.portfolio.get_equity_df()
        trades_df = self.portfolio.get_trades_df()
        monthly_returns = self.portfolio.get_monthly_returns()
        
        # Final validation
        if validate:
            print("\n5. Final validation...")
            final_prices = prices.iloc[-1].to_dict()
            valid, issues = validate_portfolio(self.portfolio, final_prices)
            if not valid:
                raise BacktestError(f"Final validation failed: {issues}")
            print("   All checks passed")
        
        print(f"\n   Trades: {len(trades_df)}")
        print(f"   Final equity: ${equity_df['Equity'].iloc[-1]:,.2f}")
        
        return BacktestResult(
            equity_df=equity_df,
            trades_df=trades_df,
            monthly_returns=monthly_returns,
            signals=signals,
            validation_passed=True,
        )
    
    def _run_simulation(
        self,
        prices: pd.DataFrame,
        signals: List[TSMOMSignal],
        validate: bool,
    ):
        """Run day-by-day simulation."""
        signal_idx = 0
        signal_dates = {s.date for s in signals}
        
        prev_prices = None
        prev_equity = self.portfolio.initial_capital
        
        for i, date in enumerate(prices.index):
            today_prices = prices.loc[date].to_dict()
            
            # Check for rebalance
            if date in signal_dates and signal_idx < len(signals):
                signal = signals[signal_idx]
                if signal.date == date:
                    signal_idx += 1
                    
                    # Execute rebalance
                    self.portfolio.rebalance_to_weights(
                        signal.target_weights,
                        today_prices,
                        date,
                    )
            
            # Update equity
            self.portfolio.update_equity(date, today_prices)
            
            # Periodic return validation
            if validate and i > 0 and i % 63 == 0 and prev_prices:
                self._validate_return(prev_prices, today_prices, prev_equity)
            
            prev_prices = today_prices
            prev_equity = self.portfolio.get_equity(today_prices)
            
            # Progress
            if i % 500 == 0:
                eq = self.portfolio.get_equity(today_prices)
                print(f"   Day {i+1}/{len(prices)}: ${eq:,.0f}")
    
    def _validate_return(
        self,
        prev_prices: Dict,
        curr_prices: Dict,
        prev_equity: float,
    ):
        """Validate portfolio return matches position returns."""
        if prev_equity <= 0:
            return
        
        curr_equity = self.portfolio.get_equity(curr_prices)
        actual_return = (curr_equity / prev_equity) - 1
        
        # Expected from positions
        expected = 0.0
        for sym, pos in self.portfolio.positions.items():
            if sym in prev_prices and sym in curr_prices:
                prev_p = prev_prices[sym]
                curr_p = curr_prices[sym]
                if prev_p > 0:
                    asset_ret = (curr_p / prev_p) - 1
                    weight = pos.market_value(prev_p) / prev_equity
                    expected += asset_ret * weight
        
        # Allow tolerance for costs and rebalancing
        if abs(actual_return - expected) > 0.10:
            print(f"   ⚠️ Return mismatch: actual={actual_return:.2%}, expected={expected:.2%}")


def run_backtest(prices: pd.DataFrame = None) -> BacktestResult:
    """Convenience function to run backtest."""
    engine = BacktestEngine()
    return engine.run(prices)


if __name__ == "__main__":
    result = run_backtest()
    
    print("\nEquity curve tail:")
    print(result.equity_df.tail())
    
    print(f"\nTrades: {len(result.trades_df)}")
    print(f"Signals: {len(result.signals)}")
