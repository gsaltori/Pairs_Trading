"""
Tests for the Pairs Trading System.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import Settings, Timeframe, TradingMode
from src.analysis.correlation import CorrelationAnalyzer
from src.analysis.cointegration import CointegrationAnalyzer
from src.analysis.spread_builder import SpreadBuilder
from src.strategy.signals import SignalGenerator, SignalType


# Fixtures
@pytest.fixture
def sample_price_data():
    """Create sample correlated price data."""
    np.random.seed(42)
    n = 500
    
    # Generate correlated random walks
    noise_a = np.random.randn(n)
    noise_b = 0.9 * noise_a + 0.1 * np.random.randn(n)  # High correlation
    
    price_a = 1.1000 + np.cumsum(noise_a * 0.0001)
    price_b = 1.3000 + np.cumsum(noise_b * 0.0001)
    
    dates = pd.date_range(start='2024-01-01', periods=n, freq='H')
    
    return pd.Series(price_a, index=dates), pd.Series(price_b, index=dates)


@pytest.fixture
def settings():
    """Create test settings."""
    return Settings()


# Correlation Tests
class TestCorrelationAnalyzer:
    
    def test_calculate_correlation(self, sample_price_data):
        """Test basic correlation calculation."""
        price_a, price_b = sample_price_data
        analyzer = CorrelationAnalyzer(window=60)
        
        corr = analyzer.calculate_correlation(price_a, price_b)
        
        assert -1 <= corr <= 1
        assert corr > 0.8  # Should be highly correlated
    
    def test_rolling_correlation(self, sample_price_data):
        """Test rolling correlation calculation."""
        price_a, price_b = sample_price_data
        analyzer = CorrelationAnalyzer(window=60)
        
        rolling_corr = analyzer.calculate_rolling_correlation(price_a, price_b)
        
        assert len(rolling_corr) == len(price_a)
        assert rolling_corr.iloc[-1] > 0.7
    
    def test_analyze_pair(self, sample_price_data):
        """Test comprehensive pair analysis."""
        price_a, price_b = sample_price_data
        analyzer = CorrelationAnalyzer(window=60)
        
        result = analyzer.analyze_pair(price_a, price_b)
        
        assert result.current_correlation > 0.7
        assert 0 <= result.stability_score <= 1


# Cointegration Tests
class TestCointegrationAnalyzer:
    
    def test_engle_granger(self, sample_price_data):
        """Test Engle-Granger cointegration test."""
        price_a, price_b = sample_price_data
        analyzer = CointegrationAnalyzer()
        
        result = analyzer.engle_granger_test(price_a, price_b)
        
        assert result.hedge_ratio != 0
        assert result.half_life > 0
        assert 0 <= result.p_value <= 1
    
    def test_hedge_ratio(self, sample_price_data):
        """Test hedge ratio calculation."""
        price_a, price_b = sample_price_data
        analyzer = CointegrationAnalyzer()
        
        result = analyzer.engle_granger_test(price_a, price_b)
        
        # Hedge ratio should be reasonable
        assert 0.5 < abs(result.hedge_ratio) < 2.0


# Spread Builder Tests
class TestSpreadBuilder:
    
    def test_build_spread(self, sample_price_data):
        """Test spread construction."""
        price_a, price_b = sample_price_data
        builder = SpreadBuilder(regression_window=120, zscore_window=60)
        
        spread = builder.build_spread(price_a, price_b)
        
        assert len(spread) == len(price_a)
        assert not spread.isna().all()
    
    def test_zscore_calculation(self, sample_price_data):
        """Test z-score calculation."""
        price_a, price_b = sample_price_data
        builder = SpreadBuilder(regression_window=120, zscore_window=60)
        
        spread_data = builder.build_spread_with_zscore(price_a, price_b)
        
        assert 'zscore' in spread_data.columns
        assert 'spread' in spread_data.columns
        assert 'hedge_ratio' in spread_data.columns
        
        # Z-scores should be standardized
        zscore = spread_data['zscore'].dropna()
        assert zscore.mean() < 1  # Should be close to 0
    
    def test_spread_metrics(self, sample_price_data):
        """Test spread metrics calculation."""
        price_a, price_b = sample_price_data
        builder = SpreadBuilder(regression_window=120, zscore_window=60)
        
        metrics = builder.get_spread_metrics(price_a, price_b)
        
        assert metrics is not None
        assert metrics.half_life > 0
        assert 0 <= metrics.hurst_exponent <= 1


# Signal Generator Tests
class TestSignalGenerator:
    
    def test_entry_signals(self, settings):
        """Test entry signal generation."""
        generator = SignalGenerator(settings)
        
        # Test long spread signal
        signal = generator.generate_entry_signal(
            pair=('EURUSD', 'GBPUSD'),
            zscore=-2.5,
            correlation=0.8,
            hedge_ratio=1.0,
            half_life=30
        )
        
        assert signal is not None
        assert signal.type == SignalType.LONG_SPREAD
        
        # Test short spread signal
        signal = generator.generate_entry_signal(
            pair=('EURUSD', 'GBPUSD'),
            zscore=2.5,
            correlation=0.8,
            hedge_ratio=1.0,
            half_life=30
        )
        
        assert signal is not None
        assert signal.type == SignalType.SHORT_SPREAD
    
    def test_no_signal_low_correlation(self, settings):
        """Test no signal when correlation is low."""
        generator = SignalGenerator(settings)
        
        signal = generator.generate_entry_signal(
            pair=('EURUSD', 'GBPUSD'),
            zscore=-2.5,
            correlation=0.5,  # Below threshold
            hedge_ratio=1.0,
            half_life=30
        )
        
        assert signal is None or signal.type == SignalType.NO_SIGNAL
    
    def test_exit_signals(self, settings):
        """Test exit signal generation."""
        generator = SignalGenerator(settings)
        
        # Test mean reversion exit
        signal = generator.generate_exit_signal(
            pair=('EURUSD', 'GBPUSD'),
            zscore=0.1,  # Near zero
            correlation=0.8,
            current_direction='long_spread'
        )
        
        assert signal is not None
        assert signal.type == SignalType.EXIT
    
    def test_stop_loss_signal(self, settings):
        """Test stop loss signal."""
        generator = SignalGenerator(settings)
        
        signal = generator.generate_exit_signal(
            pair=('EURUSD', 'GBPUSD'),
            zscore=-4.0,  # Extreme
            correlation=0.8,
            current_direction='long_spread'
        )
        
        assert signal is not None
        assert signal.type == SignalType.STOP_LOSS


# Settings Tests
class TestSettings:
    
    def test_default_settings(self):
        """Test default settings creation."""
        settings = Settings()
        
        assert settings.spread.entry_zscore == 2.0
        assert settings.spread.exit_zscore == 0.2
        assert settings.risk.max_risk_per_trade == 0.01
    
    def test_timeframe_conversion(self):
        """Test timeframe to MT5 string."""
        assert Timeframe.H1.to_mt5() == 'H1'
        assert Timeframe.M15.to_mt5() == 'M15'
        assert Timeframe.D1.to_mt5() == 'D1'


# Integration Tests
class TestIntegration:
    
    def test_full_analysis_pipeline(self, sample_price_data, settings):
        """Test complete analysis pipeline."""
        price_a, price_b = sample_price_data
        
        # Correlation
        corr_analyzer = CorrelationAnalyzer(window=60)
        corr_result = corr_analyzer.analyze_pair(price_a, price_b)
        
        # Cointegration
        coint_analyzer = CointegrationAnalyzer()
        coint_result = coint_analyzer.engle_granger_test(price_a, price_b)
        
        # Spread
        spread_builder = SpreadBuilder(regression_window=120, zscore_window=60)
        spread_data = spread_builder.build_spread_with_zscore(price_a, price_b)
        metrics = spread_builder.get_spread_metrics(price_a, price_b)
        
        # Signals
        signal_gen = SignalGenerator(settings)
        
        if not spread_data['zscore'].isna().all():
            current_zscore = spread_data['zscore'].iloc[-1]
            
            signal = signal_gen.generate_entry_signal(
                pair=('TEST_A', 'TEST_B'),
                zscore=current_zscore,
                correlation=corr_result.current_correlation,
                hedge_ratio=coint_result.hedge_ratio,
                half_life=coint_result.half_life
            )
            
            # Pipeline should complete without errors
            assert True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
