#!/usr/bin/env python3
"""
Trade Gatekeeper Integration Example

Demonstrates how to integrate the TradeGatekeeper with an external
trading strategy. The gatekeeper acts as a risk firewall that
prevents execution in empirically proven failure regimes.

ARCHITECTURE:
    
    Market Data → Strategy (generates signals) → Gatekeeper (veto) → Execution
    
The strategy is agnostic to the gatekeeper. It simply generates signals.
The gatekeeper independently evaluates whether execution is permitted.

This example shows:
1. A mock primary strategy that generates trade signals
2. Gatekeeper integration as a veto layer
3. Audit trail of blocked trades
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime
from enum import Enum

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from crv_engine.experiments.trade_gatekeeper import (
    TradeGatekeeper,
    TradePermission,
    BlockReason,
)
from crv_engine.experiments.edge_boundary import PredictionObservables


# =============================================================================
# MOCK PRIMARY STRATEGY
# =============================================================================

class SignalDirection(Enum):
    """Trade signal direction."""
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


@dataclass
class TradeSignal:
    """Signal from primary strategy."""
    timestamp: datetime
    direction: SignalDirection
    zscore: float
    confidence: float
    
    @property
    def has_signal(self) -> bool:
        return self.direction != SignalDirection.FLAT


class MockMeanReversionStrategy:
    """
    Mock mean-reversion strategy for demonstration.
    
    Generates LONG when Z < -1.5, SHORT when Z > 1.5.
    This is a simplified example - real strategies are more complex.
    """
    
    def __init__(self, entry_threshold: float = 1.5):
        self.entry_threshold = entry_threshold
    
    def generate_signal(
        self,
        zscore: float,
        correlation: float,
        volatility_ratio: float,
    ) -> TradeSignal:
        """Generate trade signal from current market state."""
        
        if zscore > self.entry_threshold:
            direction = SignalDirection.SHORT  # Expect reversion down
            confidence = min(1.0, abs(zscore) / 3.0)
        elif zscore < -self.entry_threshold:
            direction = SignalDirection.LONG  # Expect reversion up
            confidence = min(1.0, abs(zscore) / 3.0)
        else:
            direction = SignalDirection.FLAT
            confidence = 0.0
        
        return TradeSignal(
            timestamp=datetime.utcnow(),
            direction=direction,
            zscore=zscore,
            confidence=confidence,
        )


# =============================================================================
# INTEGRATED TRADING SYSTEM
# =============================================================================

@dataclass
class ExecutionResult:
    """Result of trade execution attempt."""
    signal: TradeSignal
    permission: TradePermission
    executed: bool
    message: str


class GatedTradingSystem:
    """
    Trading system with integrated gatekeeper.
    
    The gatekeeper acts as a firewall between signal generation
    and trade execution. It cannot modify signals - only veto execution.
    """
    
    def __init__(
        self,
        strategy: MockMeanReversionStrategy,
        gatekeeper: TradeGatekeeper,
    ):
        self.strategy = strategy
        self.gatekeeper = gatekeeper
        self.execution_log: List[ExecutionResult] = []
    
    def process_market_update(
        self,
        zscore: float,
        correlation: float,
        correlation_trend: float,
        volatility_ratio: float,
    ) -> ExecutionResult:
        """
        Process market update through strategy and gatekeeper.
        
        Flow:
        1. Strategy generates signal
        2. If signal exists, gatekeeper evaluates permission
        3. If permitted, execute; otherwise log veto
        """
        
        # Step 1: Strategy generates signal
        signal = self.strategy.generate_signal(
            zscore=zscore,
            correlation=correlation,
            volatility_ratio=volatility_ratio,
        )
        
        # Step 2: If no signal, nothing to gate
        if not signal.has_signal:
            result = ExecutionResult(
                signal=signal,
                permission=None,
                executed=False,
                message="No signal generated",
            )
            self.execution_log.append(result)
            return result
        
        # Step 3: Gatekeeper evaluates permission
        observables = PredictionObservables(
            prediction_id=f"signal_{signal.timestamp.timestamp()}",
            outcome="PENDING",
            bars_to_resolution=0,
            correlation=correlation,
            correlation_trend=correlation_trend,
            volatility_ratio=volatility_ratio,
            zscore=zscore,
            spread_velocity=0.0,
        )
        
        permission = self.gatekeeper.evaluate(observables)
        
        # Step 4: Execute or veto
        if permission.allowed:
            executed = True
            message = f"EXECUTED: {signal.direction.value} at Z={zscore:.2f}"
        else:
            executed = False
            reasons = ", ".join(permission.reason_labels)
            message = f"BLOCKED: {signal.direction.value} vetoed ({reasons})"
        
        result = ExecutionResult(
            signal=signal,
            permission=permission,
            executed=executed,
            message=message,
        )
        
        self.execution_log.append(result)
        return result
    
    def get_execution_summary(self) -> dict:
        """Get summary of execution attempts."""
        total = len(self.execution_log)
        with_signal = sum(1 for r in self.execution_log if r.signal.has_signal)
        executed = sum(1 for r in self.execution_log if r.executed)
        blocked = with_signal - executed
        
        return {
            'total_updates': total,
            'signals_generated': with_signal,
            'trades_executed': executed,
            'trades_blocked': blocked,
            'block_rate': blocked / with_signal if with_signal > 0 else 0.0,
            'gatekeeper_summary': self.gatekeeper.get_summary(),
        }


# =============================================================================
# DEMONSTRATION
# =============================================================================

def run_demonstration():
    """
    Run demonstration of gatekeeper integration.
    
    Simulates a series of market updates and shows how the gatekeeper
    blocks trades in failure regimes while allowing trades in safe regimes.
    """
    
    print("=" * 70)
    print("TRADE GATEKEEPER INTEGRATION DEMONSTRATION")
    print("=" * 70)
    print()
    
    # Create components
    strategy = MockMeanReversionStrategy(entry_threshold=1.5)
    gatekeeper = TradeGatekeeper()
    system = GatedTradingSystem(strategy, gatekeeper)
    
    # Simulated market scenarios
    scenarios = [
        # (zscore, correlation, corr_trend, vol_ratio, description)
        (1.8, 0.85, 0.02, 1.0, "Normal entry signal - SHOULD EXECUTE"),
        (2.5, 0.80, 0.00, 1.1, "Stronger signal, safe conditions - SHOULD EXECUTE"),
        (3.5, 0.75, 0.01, 0.9, "EXTREME Z-score - SHOULD BLOCK"),
        (-2.0, 0.70, -0.12, 1.0, "Deteriorating correlation - SHOULD BLOCK"),
        (1.9, 0.82, 0.03, 0.5, "Compressed volatility - SHOULD BLOCK"),
        (-1.8, 0.88, 0.05, 1.2, "Safe conditions, long signal - SHOULD EXECUTE"),
        (4.0, 0.60, -0.15, 0.4, "Multiple failure modes - SHOULD BLOCK"),
        (0.5, 0.90, 0.01, 1.0, "No signal (Z too small) - NO ACTION"),
        (-2.2, 0.78, 0.00, 0.95, "Normal short signal - SHOULD EXECUTE"),
    ]
    
    print("Processing market updates...\n")
    print("-" * 70)
    
    for zscore, corr, corr_trend, vol_ratio, description in scenarios:
        result = system.process_market_update(
            zscore=zscore,
            correlation=corr,
            correlation_trend=corr_trend,
            volatility_ratio=vol_ratio,
        )
        
        print(f"Scenario: {description}")
        print(f"  Z={zscore:+.2f}, ρ={corr:.2f}, Δρ={corr_trend:+.2f}, σ_ratio={vol_ratio:.2f}")
        print(f"  Result: {result.message}")
        print()
    
    print("-" * 70)
    print("\nEXECUTION SUMMARY")
    print("-" * 70)
    
    summary = system.get_execution_summary()
    print(f"Total market updates:    {summary['total_updates']}")
    print(f"Signals generated:       {summary['signals_generated']}")
    print(f"Trades executed:         {summary['trades_executed']}")
    print(f"Trades blocked:          {summary['trades_blocked']}")
    print(f"Block rate:              {summary['block_rate']:.1%}")
    
    print("\nGATEKEEPER STATISTICS")
    print("-" * 70)
    gk_summary = summary['gatekeeper_summary']
    print(f"Total evaluations:       {gk_summary['total_evaluations']}")
    print(f"Block rate:              {gk_summary['block_rate']:.1%}")
    print("\nBlock reasons:")
    for reason, count in gk_summary['reason_distribution'].items():
        if count > 0:
            print(f"  {reason}: {count}")
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


# =============================================================================
# MINIMAL INTEGRATION EXAMPLE
# =============================================================================

def minimal_integration_example():
    """
    Minimal code example for documentation.
    
    Shows the simplest possible integration pattern.
    """
    
    print("\n" + "=" * 70)
    print("MINIMAL INTEGRATION PATTERN")
    print("=" * 70)
    
    code = '''
    from crv_engine.experiments.trade_gatekeeper import TradeGatekeeper
    from crv_engine.experiments.edge_boundary import PredictionObservables
    
    # Create gatekeeper (no configuration needed)
    gatekeeper = TradeGatekeeper()
    
    # When your strategy generates a signal, check with gatekeeper:
    def on_trade_signal(zscore, correlation, correlation_trend, volatility_ratio):
        # Create observables from current market state
        observables = PredictionObservables(
            prediction_id="signal_001",
            outcome="PENDING",
            bars_to_resolution=0,
            correlation=correlation,
            correlation_trend=correlation_trend,
            volatility_ratio=volatility_ratio,
            zscore=zscore,
            spread_velocity=0.0,
        )
        
        # Evaluate permission
        permission = gatekeeper.evaluate(observables)
        
        if permission.allowed:
            execute_trade()  # Your execution logic
        else:
            log_blocked_trade(permission.reasons)  # Audit trail
    
    # Or use the simple function for quick checks:
    from crv_engine.experiments.trade_gatekeeper import check_trade_permission
    
    if check_trade_permission(zscore=2.0, correlation_trend=0.0, volatility_ratio=1.0):
        execute_trade()
    '''
    
    print(code)


if __name__ == "__main__":
    run_demonstration()
    minimal_integration_example()
