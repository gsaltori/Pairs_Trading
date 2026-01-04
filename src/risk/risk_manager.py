"""
Risk Manager

Handles all risk management aspects:
- Position sizing
- Exposure limits
- Drawdown monitoring
- Risk-adjusted metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

import sys
sys.path.append(str(__file__).rsplit('\\', 3)[0])

from config.settings import Settings, RiskParameters
from config.broker_config import OANDA_INSTRUMENTS


logger = logging.getLogger(__name__)


@dataclass
class PositionSize:
    """Position sizing for a pair trade."""
    pair: Tuple[str, str]
    units_a: int  # Units for instrument A
    units_b: int  # Units for instrument B
    hedge_ratio: float
    notional_value: float
    risk_amount: float
    margin_required: float


@dataclass
class RiskState:
    """Current risk state."""
    timestamp: datetime
    account_balance: float
    account_equity: float
    open_positions: int
    total_exposure: float
    exposure_pct: float
    unrealized_pnl: float
    daily_pnl: float
    max_drawdown: float
    current_drawdown: float


class RiskManager:
    """
    Manages risk for the Pairs Trading System.
    
    Features:
    - Position sizing based on risk per trade
    - Hedge ratio-balanced position sizes
    - Exposure monitoring
    - Drawdown tracking
    - Risk limits enforcement
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize the Risk Manager.
        
        Args:
            settings: System settings
        """
        self.settings = settings
        self.params = settings.risk_params
        
        # Track state
        self._account_balance: float = settings.backtest_params.initial_capital
        self._peak_equity: float = self._account_balance
        self._current_equity: float = self._account_balance
        self._open_positions: Dict[Tuple[str, str], PositionSize] = {}
        self._daily_start_equity: float = self._account_balance
        self._trade_history: List[Dict] = []
    
    def set_account_balance(self, balance: float) -> None:
        """Update account balance."""
        self._account_balance = balance
        self._current_equity = balance
        if balance > self._peak_equity:
            self._peak_equity = balance
    
    def calculate_position_size(
        self,
        pair: Tuple[str, str],
        hedge_ratio: float,
        current_prices: Dict[str, float],
        atr_a: Optional[float] = None,
        atr_b: Optional[float] = None
    ) -> PositionSize:
        """
        Calculate position sizes for a pair trade.
        
        Uses risk-based position sizing balanced by hedge ratio.
        
        Args:
            pair: Instrument pair (A, B)
            hedge_ratio: Hedge ratio (beta) for the spread
            current_prices: Dictionary with current prices
            atr_a: ATR for instrument A (optional, for volatility sizing)
            atr_b: ATR for instrument B (optional)
            
        Returns:
            PositionSize with calculated units
        """
        instrument_a, instrument_b = pair
        price_a = current_prices.get(instrument_a, {}).get('mid', 0)
        price_b = current_prices.get(instrument_b, {}).get('mid', 0)
        
        if price_a == 0 or price_b == 0:
            raise ValueError(f"Invalid prices for {pair}")
        
        # Calculate risk amount
        risk_amount = self._account_balance * self.params.max_risk_per_trade
        
        if self.params.max_loss_per_trade:
            risk_amount = min(risk_amount, self.params.max_loss_per_trade)
        
        # Position sizing method
        if self.params.sizing_method == 'volatility_adjusted' and atr_a and atr_b:
            # Volatility-adjusted sizing
            # Units sized inversely to volatility
            vol_ratio = atr_b / atr_a if atr_a > 0 else 1.0
            base_units = risk_amount / (atr_a + atr_b * abs(hedge_ratio))
            units_a = int(base_units)
            units_b = int(base_units * abs(hedge_ratio) * vol_ratio)
        else:
            # Equal notional sizing adjusted for hedge ratio
            # Total notional = price_a * units_a + price_b * units_b * hedge_ratio
            # We want: price_a * units_a ≈ price_b * units_b * hedge_ratio
            
            total_notional = risk_amount * 10  # Use 10:1 leverage as baseline
            
            # Solve for units
            # units_a * price_a = units_b * price_b * hedge_ratio
            # total_notional = units_a * price_a + units_b * price_b * hedge_ratio
            # total_notional = 2 * units_a * price_a
            
            units_a = int(total_notional / (2 * price_a))
            units_b = int(units_a * price_a * abs(hedge_ratio) / price_b)
        
        # Ensure hedge ratio balance
        actual_ratio = (units_b * price_b) / (units_a * price_a) if units_a > 0 else 0
        
        # Calculate notional value
        notional_a = units_a * price_a
        notional_b = units_b * price_b
        total_notional = notional_a + notional_b
        
        # Estimate margin (typically 2-5% for major forex pairs)
        margin_rate = 0.02  # 2% margin = 50:1 leverage
        margin_required = total_notional * margin_rate
        
        return PositionSize(
            pair=pair,
            units_a=units_a,
            units_b=units_b,
            hedge_ratio=hedge_ratio,
            notional_value=total_notional,
            risk_amount=risk_amount,
            margin_required=margin_required
        )
    
    def check_trade_allowed(
        self,
        position_size: PositionSize
    ) -> Tuple[bool, str]:
        """
        Check if a new trade is allowed given current risk limits.
        
        Args:
            position_size: Proposed position size
            
        Returns:
            Tuple of (is_allowed, reason)
        """
        # Check max open pairs
        if len(self._open_positions) >= self.params.max_open_pairs:
            return False, f"Max open pairs reached: {len(self._open_positions)}"
        
        # Check if pair already open
        if position_size.pair in self._open_positions:
            return False, f"Pair {position_size.pair} already has open position"
        
        # Check total exposure
        current_exposure = sum(p.notional_value for p in self._open_positions.values())
        new_exposure = current_exposure + position_size.notional_value
        exposure_pct = new_exposure / self._account_balance
        
        if exposure_pct > self.params.max_total_exposure:
            return False, f"Max exposure exceeded: {exposure_pct:.1%} > {self.params.max_total_exposure:.1%}"
        
        # Check drawdown
        current_dd = self._calculate_current_drawdown()
        if current_dd >= self.params.max_drawdown:
            return False, f"Max drawdown reached: {current_dd:.1%}"
        
        # Check margin
        if position_size.margin_required > self._account_balance * 0.5:
            return False, f"Insufficient margin"
        
        return True, "Trade allowed"
    
    def register_open_position(self, position_size: PositionSize) -> None:
        """
        Register a newly opened position.
        
        Args:
            position_size: Position size details
        """
        self._open_positions[position_size.pair] = position_size
        
        logger.info(
            f"Registered position: {position_size.pair} - "
            f"A: {position_size.units_a}, B: {position_size.units_b}"
        )
    
    def close_position(
        self,
        pair: Tuple[str, str],
        pnl: float
    ) -> None:
        """
        Close a position and update risk state.
        
        Args:
            pair: Instrument pair
            pnl: Profit/loss from the trade
        """
        if pair not in self._open_positions:
            logger.warning(f"Position not found: {pair}")
            return
        
        position = self._open_positions.pop(pair)
        
        # Update balance
        self._account_balance += pnl
        self._current_equity = self._account_balance
        
        # Update peak if new high
        if self._current_equity > self._peak_equity:
            self._peak_equity = self._current_equity
        
        # Record trade
        self._trade_history.append({
            'pair': pair,
            'pnl': pnl,
            'timestamp': datetime.now(),
            'balance_after': self._account_balance
        })
        
        logger.info(
            f"Closed position: {pair} - PnL: {pnl:.2f}, "
            f"Balance: {self._account_balance:.2f}"
        )
    
    def _calculate_current_drawdown(self) -> float:
        """Calculate current drawdown from peak."""
        if self._peak_equity == 0:
            return 0.0
        return (self._peak_equity - self._current_equity) / self._peak_equity
    
    def update_equity(self, unrealized_pnl: float) -> None:
        """
        Update current equity with unrealized P&L.
        
        Args:
            unrealized_pnl: Total unrealized profit/loss
        """
        self._current_equity = self._account_balance + unrealized_pnl
    
    def get_risk_state(self) -> RiskState:
        """
        Get current risk state.
        
        Returns:
            RiskState object
        """
        current_exposure = sum(p.notional_value for p in self._open_positions.values())
        exposure_pct = current_exposure / self._account_balance if self._account_balance > 0 else 0
        unrealized = self._current_equity - self._account_balance
        daily_pnl = self._current_equity - self._daily_start_equity
        
        return RiskState(
            timestamp=datetime.now(),
            account_balance=self._account_balance,
            account_equity=self._current_equity,
            open_positions=len(self._open_positions),
            total_exposure=current_exposure,
            exposure_pct=exposure_pct,
            unrealized_pnl=unrealized,
            daily_pnl=daily_pnl,
            max_drawdown=self.params.max_drawdown,
            current_drawdown=self._calculate_current_drawdown()
        )
    
    def calculate_trade_pnl(
        self,
        position_size: PositionSize,
        entry_prices: Dict[str, float],
        exit_prices: Dict[str, float],
        position_type: str  # 'long_spread' or 'short_spread'
    ) -> float:
        """
        Calculate P&L for a pair trade.
        
        Args:
            position_size: Position size details
            entry_prices: Entry prices for both instruments
            exit_prices: Exit prices for both instruments
            position_type: Type of position
            
        Returns:
            Total P&L in account currency
        """
        instrument_a, instrument_b = position_size.pair
        
        entry_a = entry_prices.get(instrument_a, 0)
        entry_b = entry_prices.get(instrument_b, 0)
        exit_a = exit_prices.get(instrument_a, 0)
        exit_b = exit_prices.get(instrument_b, 0)
        
        # Calculate P&L for each leg
        if position_type == 'long_spread':
            # Long A, Short B
            pnl_a = (exit_a - entry_a) * position_size.units_a
            pnl_b = (entry_b - exit_b) * position_size.units_b
        else:  # short_spread
            # Short A, Long B
            pnl_a = (entry_a - exit_a) * position_size.units_a
            pnl_b = (exit_b - entry_b) * position_size.units_b
        
        # Convert to account currency if needed
        # For simplicity, assuming USD account and standard lot conversion
        pnl_a = self._convert_pnl_to_usd(instrument_a, pnl_a, exit_a)
        pnl_b = self._convert_pnl_to_usd(instrument_b, pnl_b, exit_b)
        
        return pnl_a + pnl_b
    
    def _convert_pnl_to_usd(
        self,
        instrument: str,
        pnl: float,
        current_price: float
    ) -> float:
        """
        Convert P&L to USD.
        
        Args:
            instrument: Instrument name
            pnl: P&L in quote currency
            current_price: Current price
            
        Returns:
            P&L in USD
        """
        # Extract quote currency
        quote_currency = instrument[4:]
        
        if quote_currency == 'USD':
            return pnl
        elif quote_currency == 'JPY':
            # Need USD/JPY rate - approximate
            return pnl / current_price
        else:
            # For other pairs, would need conversion rate
            # For simplicity, return as is
            return pnl
    
    def calculate_sharpe_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Sharpe ratio from returns.
        
        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading periods per year
            
        Returns:
            Annualized Sharpe ratio
        """
        if returns.empty or returns.std() == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / periods_per_year
        
        return np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()
    
    def calculate_sortino_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Sortino ratio (downside deviation only).
        
        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading periods per year
            
        Returns:
            Annualized Sortino ratio
        """
        if returns.empty:
            return 0.0
        
        excess_returns = returns - risk_free_rate / periods_per_year
        downside_returns = excess_returns[excess_returns < 0]
        
        if downside_returns.empty or downside_returns.std() == 0:
            return 0.0
        
        downside_std = np.sqrt((downside_returns ** 2).mean())
        
        return np.sqrt(periods_per_year) * excess_returns.mean() / downside_std
    
    def calculate_max_drawdown(self, equity_curve: pd.Series) -> Tuple[float, datetime, datetime]:
        """
        Calculate maximum drawdown from equity curve.
        
        Args:
            equity_curve: Series of equity values
            
        Returns:
            Tuple of (max_drawdown, peak_date, trough_date)
        """
        if equity_curve.empty:
            return 0.0, None, None
        
        # Calculate running maximum
        running_max = equity_curve.expanding().max()
        
        # Calculate drawdown
        drawdown = (running_max - equity_curve) / running_max
        
        # Find maximum drawdown
        max_dd = drawdown.max()
        max_dd_end = drawdown.idxmax()
        
        # Find peak before max drawdown
        peak_date = equity_curve[:max_dd_end].idxmax()
        
        return max_dd, peak_date, max_dd_end
    
    def get_trade_statistics(self) -> Dict:
        """
        Get statistics from trade history.
        
        Returns:
            Dictionary of trade statistics
        """
        if not self._trade_history:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'expectancy': 0.0
            }
        
        pnls = [t['pnl'] for t in self._trade_history]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        total_trades = len(pnls)
        winning_trades = len(wins)
        losing_trades = len(losses)
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'total_pnl': sum(pnls),
            'gross_profit': gross_profit,
            'gross_loss': gross_loss
        }
    
    def reset_daily(self) -> None:
        """Reset daily tracking (call at start of each day)."""
        self._daily_start_equity = self._current_equity
    
    def reset(self) -> None:
        """Reset all risk state."""
        self._account_balance = self.settings.backtest_params.initial_capital
        self._peak_equity = self._account_balance
        self._current_equity = self._account_balance
        self._open_positions.clear()
        self._daily_start_equity = self._account_balance
        self._trade_history.clear()
