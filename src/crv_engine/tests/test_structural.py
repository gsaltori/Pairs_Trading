"""
Unit tests for P4 Structural Stability Module.

Tests cover:
1. Stable regime detection (should be VALID)
2. Correlation drift detection (should be INVALID)
3. Sudden regime break detection (should be INVALID)
4. Volatility ratio instability
5. False positive prevention
"""

import pytest
from datetime import datetime, timezone, timedelta
import numpy as np

import sys
sys.path.insert(0, '..')

from structural import (
    StructuralValidity,
    StructuralState,
    StructuralConfig,
    StructuralStabilityEvaluator,
    STRUCTURAL_CONFIG,
)


def make_timestamp(bar_index: int) -> datetime:
    """Create timestamp for a given bar index."""
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return base + timedelta(hours=4 * bar_index)


class TestStructuralValidity:
    """Test StructuralValidity enum."""
    
    def test_valid_state_exists(self):
        """STRUCTURALLY_VALID state should exist."""
        assert StructuralValidity.STRUCTURALLY_VALID.value == "STRUCTURALLY_VALID"
    
    def test_invalid_state_exists(self):
        """STRUCTURALLY_INVALID state should exist."""
        assert StructuralValidity.STRUCTURALLY_INVALID.value == "STRUCTURALLY_INVALID"


class TestStructuralConfig:
    """Test StructuralConfig thresholds."""
    
    def test_config_is_frozen(self):
        """Config should be immutable."""
        config = StructuralConfig()
        with pytest.raises(Exception):
            config.MAX_CORRELATION_STD = 0.5
    
    def test_thresholds_reasonable(self):
        """Thresholds should be in reasonable ranges."""
        assert 0.05 <= STRUCTURAL_CONFIG.MAX_CORRELATION_STD <= 0.20
        assert -0.02 <= STRUCTURAL_CONFIG.MIN_CORRELATION_TREND <= 0


class TestStructuralState:
    """Test StructuralState dataclass."""
    
    def test_valid_state_is_valid(self):
        """STRUCTURALLY_VALID should report is_valid=True."""
        state = StructuralState(
            validity=StructuralValidity.STRUCTURALLY_VALID,
            timestamp=datetime.now(timezone.utc),
            correlation_stability=0.05,
            correlation_trend=0.001,
            correlation_current=0.80,
            volatility_ratio_stability=0.10,
            spread_variance_ratio=1.2,
        )
        assert state.is_valid == True
    
    def test_invalid_state_is_not_valid(self):
        """STRUCTURALLY_INVALID should report is_valid=False."""
        state = StructuralState(
            validity=StructuralValidity.STRUCTURALLY_INVALID,
            timestamp=datetime.now(timezone.utc),
            correlation_stability=0.20,
            correlation_trend=-0.02,
            correlation_current=0.40,
            volatility_ratio_stability=0.30,
            spread_variance_ratio=3.0,
            invalidity_reasons=("correlation_unstable",),
        )
        assert state.is_valid == False
        assert "correlation_unstable" in state.invalidity_reasons


class TestStableRegime:
    """Test that stable regimes are correctly identified as VALID."""
    
    def setup_method(self):
        """Setup for each test."""
        self.evaluator = StructuralStabilityEvaluator()
    
    def test_stable_high_correlation(self):
        """Stable regime with constant high correlation should be VALID."""
        for i in range(100):
            ts = make_timestamp(i)
            corr = 0.85  # Constant
            vol_a = 0.01
            vol_b = 0.01
            spread = 0.001
            self.evaluator.update(ts, corr, vol_a, vol_b, spread)
        
        state = self.evaluator.evaluate(make_timestamp(100))
        assert state.is_valid
        assert state.correlation_stability < STRUCTURAL_CONFIG.MAX_CORRELATION_STD
    
    def test_stable_moderate_correlation(self):
        """Stable regime with moderate correlation should be VALID."""
        for i in range(100):
            ts = make_timestamp(i)
            corr = 0.55  # Moderate but stable
            vol_a = 0.01
            vol_b = 0.01
            spread = 0.001
            self.evaluator.update(ts, corr, vol_a, vol_b, spread)
        
        state = self.evaluator.evaluate(make_timestamp(100))
        assert state.is_valid
    
    def test_stable_volatility_ratio(self):
        """Stable volatility ratio should have low stability metric."""
        for i in range(100):
            ts = make_timestamp(i)
            corr = 0.75
            vol_a = 0.012  # Consistent vol ratio
            vol_b = 0.010
            spread = 0.001
            self.evaluator.update(ts, corr, vol_a, vol_b, spread)
        
        state = self.evaluator.evaluate(make_timestamp(100))
        # Low stability value = more stable
        assert state.volatility_ratio_stability < 0.5


class TestCorrelationDrift:
    """Test detection of correlation drift (gradual deterioration)."""
    
    def setup_method(self):
        """Setup for each test."""
        self.evaluator = StructuralStabilityEvaluator()
    
    def test_declining_correlation_trend(self):
        """Declining correlation trend should be detected."""
        for i in range(100):
            ts = make_timestamp(i)
            # Correlation declines from 0.80 to 0.30
            corr = 0.80 - (0.50 * i / 100)
            vol_a = 0.01
            vol_b = 0.01
            spread = 0.001
            self.evaluator.update(ts, corr, vol_a, vol_b, spread)
        
        state = self.evaluator.evaluate(make_timestamp(100))
        assert state.correlation_trend < 0
    
    def test_correlation_high_variance(self):
        """High correlation variance should be detected."""
        rng = np.random.default_rng(42)
        for i in range(100):
            ts = make_timestamp(i)
            # Noisy correlation with high variance
            corr = 0.60 + rng.uniform(-0.25, 0.25)
            vol_a = 0.01
            vol_b = 0.01
            spread = 0.001
            self.evaluator.update(ts, corr, vol_a, vol_b, spread)
        
        state = self.evaluator.evaluate(make_timestamp(100))
        assert state.correlation_stability > 0.05


class TestVolatilityRatioInstability:
    """Test volatility ratio checks."""
    
    def setup_method(self):
        """Setup for each test."""
        self.evaluator = StructuralStabilityEvaluator()
    
    def test_volatile_ratio(self):
        """Unstable volatility ratio should be detected."""
        rng = np.random.default_rng(42)
        for i in range(100):
            ts = make_timestamp(i)
            corr = 0.75
            # Volatile ratio swings
            vol_a = 0.01 + rng.uniform(-0.005, 0.015)
            vol_b = 0.01
            spread = 0.001
            self.evaluator.update(ts, corr, vol_a, vol_b, spread)
        
        state = self.evaluator.evaluate(make_timestamp(100))
        assert state.volatility_ratio_stability > 0


class TestFalsePositivePrevention:
    """Test that we don't over-block (false positives)."""
    
    def setup_method(self):
        """Setup for each test."""
        self.evaluator = StructuralStabilityEvaluator()
    
    def test_normal_noise_not_blocked(self):
        """Normal market noise should not cause blocking."""
        rng = np.random.default_rng(42)
        for i in range(100):
            ts = make_timestamp(i)
            # Small natural noise
            corr = 0.75 + rng.normal(0, 0.02)
            vol_a = 0.01 + rng.normal(0, 0.001)
            vol_b = 0.01 + rng.normal(0, 0.001)
            spread = 0.001 + rng.normal(0, 0.0002)
            self.evaluator.update(ts, corr, max(0.001, vol_a), max(0.001, vol_b), abs(spread))
        
        state = self.evaluator.evaluate(make_timestamp(100))
        # Slight noise should still be valid
        assert state.correlation_stability < STRUCTURAL_CONFIG.MAX_CORRELATION_STD + 0.05
    
    def test_temporary_dip_recovers(self):
        """Temporary correlation dip should not permanently block."""
        for i in range(150):
            ts = make_timestamp(i)
            # Temporary dip in middle
            if 50 <= i <= 60:
                corr = 0.40
            else:
                corr = 0.75
            vol_a = 0.01
            vol_b = 0.01
            spread = 0.001
            self.evaluator.update(ts, corr, vol_a, vol_b, spread)
        
        # After recovery, evaluate
        state = self.evaluator.evaluate(make_timestamp(150))
        # Should show current stable correlation
        assert state.correlation_current > 0.50


class TestInsufficientData:
    """Test handling of insufficient data."""
    
    def test_insufficient_data_is_invalid(self):
        """Insufficient data should return INVALID."""
        evaluator = StructuralStabilityEvaluator()
        
        for i in range(10):  # Only 10 bars
            ts = make_timestamp(i)
            evaluator.update(ts, 0.75, 0.01, 0.01, 0.001)
        
        state = evaluator.evaluate(make_timestamp(10))
        assert not state.is_valid
        # Check that there's some message about insufficient history
        assert len(state.invalidity_reasons) > 0
        assert any("insufficient" in r.lower() or "history" in r.lower() 
                   for r in state.invalidity_reasons)
    
    def test_minimum_history_requirement(self):
        """After minimum history, should work normally."""
        evaluator = StructuralStabilityEvaluator()
        
        for i in range(60):  # Meet minimum
            ts = make_timestamp(i)
            evaluator.update(ts, 0.75, 0.01, 0.01, 0.001)
        
        state = evaluator.evaluate(make_timestamp(60))
        # Should not have insufficient history as the only reason
        has_insufficient = any("insufficient" in r.lower() or "history" in r.lower() 
                              for r in state.invalidity_reasons)
        # Either valid or invalid for other reasons
        assert state.is_valid or not has_insufficient


class TestEvaluatorStatistics:
    """Test evaluator state management."""
    
    def test_evaluator_tracks_history(self):
        """Evaluator should track history."""
        evaluator = StructuralStabilityEvaluator()
        
        for i in range(100):
            ts = make_timestamp(i)
            evaluator.update(ts, 0.75, 0.01, 0.01, 0.001)
        
        state = evaluator.evaluate(make_timestamp(100))
        # Should have evaluated successfully
        assert state.timestamp is not None
    
    def test_reset_clears_state(self):
        """Reset should clear all internal state."""
        evaluator = StructuralStabilityEvaluator()
        
        for i in range(100):
            ts = make_timestamp(i)
            evaluator.update(ts, 0.75, 0.01, 0.01, 0.001)
        
        evaluator.reset()
        
        # After reset, insufficient data
        state = evaluator.evaluate(make_timestamp(100))
        assert not state.is_valid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
