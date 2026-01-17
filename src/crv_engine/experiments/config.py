"""
Experiment Configuration Module

Defines immutable configuration dataclasses for reproducible experiments.
Each experiment varies EXACTLY ONE dimension while holding others at baseline.

NO STRATEGY LOGIC. NO TUNING. CONFIGURATION ONLY.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, List, FrozenSet
from enum import Enum
import hashlib
import json


class Timeframe(Enum):
    """Supported timeframes for experimentation."""
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"


class ExperimentDimension(Enum):
    """
    The single dimension being varied in an experiment.
    
    CRITICAL: Only ONE dimension may vary per experiment.
    All other parameters must remain at baseline.
    """
    SYMBOL_PAIR = "symbol_pair"
    TIMEFRAME = "timeframe"
    ZSCORE_WINDOW = "zscore_window"
    CORRELATION_WINDOW = "correlation_window"
    RANDOM_SEED = "random_seed"  # For synthetic data only


# =============================================================================
# BASELINE CONFIGURATION (FROZEN)
# =============================================================================

@dataclass(frozen=True)
class BaselineConfig:
    """
    Baseline configuration values.
    
    These are the DEFAULT values when a dimension is NOT being varied.
    THESE VALUES ARE FIXED AND MUST NOT BE TUNED.
    """
    # Symbol pair baseline
    SYMBOL_A: str = "EURUSD"
    SYMBOL_B: str = "GBPUSD"
    
    # Timeframe baseline
    TIMEFRAME: Timeframe = Timeframe.H4
    
    # Window lengths baseline (from original engine)
    ZSCORE_WINDOW: int = 60
    CORRELATION_WINDOW: int = 60
    
    # Prediction thresholds (FROZEN - from engine, not tunable)
    TRIGGER_THRESHOLD: float = 1.5
    CONFIRMATION_THRESHOLD: float = 0.3
    REFUTATION_THRESHOLD: float = 3.0
    MAX_HOLDING_BARS: int = 50
    
    # Invalidation thresholds (FROZEN - from engine)
    MIN_CORRELATION: float = 0.20
    MAX_CORRELATION_DROP: float = 0.30
    
    # Default random seed for reproducibility
    RANDOM_SEED: int = 42


BASELINE = BaselineConfig()


# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class ExperimentConfig:
    """
    Complete configuration for a single experiment run.
    
    INVARIANT: Exactly ONE dimension differs from baseline.
    This is enforced at construction time.
    """
    # Experiment identification
    experiment_id: str
    dimension_varied: ExperimentDimension
    
    # Symbol configuration
    symbol_a: str
    symbol_b: str
    
    # Timeframe
    timeframe: Timeframe
    
    # Window parameters
    zscore_window: int
    correlation_window: int
    
    # Engine thresholds (ALWAYS baseline - not varied)
    trigger_threshold: float = BASELINE.TRIGGER_THRESHOLD
    confirmation_threshold: float = BASELINE.CONFIRMATION_THRESHOLD
    refutation_threshold: float = BASELINE.REFUTATION_THRESHOLD
    max_holding_bars: int = BASELINE.MAX_HOLDING_BARS
    min_correlation: float = BASELINE.MIN_CORRELATION
    max_correlation_drop: float = BASELINE.MAX_CORRELATION_DROP
    
    # Data parameters
    n_bars: int = 2000
    random_seed: int = BASELINE.RANDOM_SEED
    use_synthetic: bool = True  # False for live MT5 data
    
    @property
    def pair(self) -> Tuple[str, str]:
        return (self.symbol_a, self.symbol_b)
    
    @property
    def config_hash(self) -> str:
        """
        Deterministic hash of configuration for audit trail.
        """
        config_dict = {
            'symbol_a': self.symbol_a,
            'symbol_b': self.symbol_b,
            'timeframe': self.timeframe.value,
            'zscore_window': self.zscore_window,
            'correlation_window': self.correlation_window,
            'n_bars': self.n_bars,
            'random_seed': self.random_seed,
            'use_synthetic': self.use_synthetic,
        }
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def validate_single_dimension(self) -> bool:
        """
        Verify that EXACTLY one dimension differs from baseline.
        
        Returns True if valid, raises ValueError otherwise.
        """
        differences = []
        
        if (self.symbol_a, self.symbol_b) != (BASELINE.SYMBOL_A, BASELINE.SYMBOL_B):
            differences.append(ExperimentDimension.SYMBOL_PAIR)
        
        if self.timeframe != BASELINE.TIMEFRAME:
            differences.append(ExperimentDimension.TIMEFRAME)
        
        if self.zscore_window != BASELINE.ZSCORE_WINDOW:
            differences.append(ExperimentDimension.ZSCORE_WINDOW)
        
        if self.correlation_window != BASELINE.CORRELATION_WINDOW:
            differences.append(ExperimentDimension.CORRELATION_WINDOW)
        
        if self.random_seed != BASELINE.RANDOM_SEED:
            differences.append(ExperimentDimension.RANDOM_SEED)
        
        if len(differences) == 0:
            # Baseline experiment - valid
            return True
        
        if len(differences) == 1:
            if differences[0] != self.dimension_varied:
                raise ValueError(
                    f"Declared dimension {self.dimension_varied} does not match "
                    f"actual varied dimension {differences[0]}"
                )
            return True
        
        raise ValueError(
            f"Experiment violates single-dimension constraint. "
            f"Multiple dimensions varied: {differences}"
        )


# =============================================================================
# EXPERIMENT FACTORIES
# =============================================================================

def create_baseline_experiment(
    experiment_id: str,
    n_bars: int = 2000,
) -> ExperimentConfig:
    """Create an experiment with all baseline parameters."""
    return ExperimentConfig(
        experiment_id=experiment_id,
        dimension_varied=ExperimentDimension.RANDOM_SEED,  # Nominal
        symbol_a=BASELINE.SYMBOL_A,
        symbol_b=BASELINE.SYMBOL_B,
        timeframe=BASELINE.TIMEFRAME,
        zscore_window=BASELINE.ZSCORE_WINDOW,
        correlation_window=BASELINE.CORRELATION_WINDOW,
        n_bars=n_bars,
        random_seed=BASELINE.RANDOM_SEED,
    )


def create_symbol_pair_experiment(
    experiment_id: str,
    symbol_a: str,
    symbol_b: str,
    n_bars: int = 2000,
) -> ExperimentConfig:
    """Create an experiment varying only the symbol pair."""
    return ExperimentConfig(
        experiment_id=experiment_id,
        dimension_varied=ExperimentDimension.SYMBOL_PAIR,
        symbol_a=symbol_a,
        symbol_b=symbol_b,
        timeframe=BASELINE.TIMEFRAME,
        zscore_window=BASELINE.ZSCORE_WINDOW,
        correlation_window=BASELINE.CORRELATION_WINDOW,
        n_bars=n_bars,
        random_seed=BASELINE.RANDOM_SEED,
    )


def create_timeframe_experiment(
    experiment_id: str,
    timeframe: Timeframe,
    n_bars: int = 2000,
) -> ExperimentConfig:
    """Create an experiment varying only the timeframe."""
    return ExperimentConfig(
        experiment_id=experiment_id,
        dimension_varied=ExperimentDimension.TIMEFRAME,
        symbol_a=BASELINE.SYMBOL_A,
        symbol_b=BASELINE.SYMBOL_B,
        timeframe=timeframe,
        zscore_window=BASELINE.ZSCORE_WINDOW,
        correlation_window=BASELINE.CORRELATION_WINDOW,
        n_bars=n_bars,
        random_seed=BASELINE.RANDOM_SEED,
    )


def create_zscore_window_experiment(
    experiment_id: str,
    zscore_window: int,
    n_bars: int = 2000,
) -> ExperimentConfig:
    """Create an experiment varying only the Z-score window."""
    return ExperimentConfig(
        experiment_id=experiment_id,
        dimension_varied=ExperimentDimension.ZSCORE_WINDOW,
        symbol_a=BASELINE.SYMBOL_A,
        symbol_b=BASELINE.SYMBOL_B,
        timeframe=BASELINE.TIMEFRAME,
        zscore_window=zscore_window,
        correlation_window=BASELINE.CORRELATION_WINDOW,
        n_bars=n_bars,
        random_seed=BASELINE.RANDOM_SEED,
    )


def create_correlation_window_experiment(
    experiment_id: str,
    correlation_window: int,
    n_bars: int = 2000,
) -> ExperimentConfig:
    """Create an experiment varying only the correlation window."""
    return ExperimentConfig(
        experiment_id=experiment_id,
        dimension_varied=ExperimentDimension.CORRELATION_WINDOW,
        symbol_a=BASELINE.SYMBOL_A,
        symbol_b=BASELINE.SYMBOL_B,
        timeframe=BASELINE.TIMEFRAME,
        zscore_window=BASELINE.ZSCORE_WINDOW,
        correlation_window=correlation_window,
        n_bars=n_bars,
        random_seed=BASELINE.RANDOM_SEED,
    )


def create_seed_experiment(
    experiment_id: str,
    random_seed: int,
    n_bars: int = 2000,
) -> ExperimentConfig:
    """Create an experiment varying only the random seed (synthetic data)."""
    return ExperimentConfig(
        experiment_id=experiment_id,
        dimension_varied=ExperimentDimension.RANDOM_SEED,
        symbol_a=BASELINE.SYMBOL_A,
        symbol_b=BASELINE.SYMBOL_B,
        timeframe=BASELINE.TIMEFRAME,
        zscore_window=BASELINE.ZSCORE_WINDOW,
        correlation_window=BASELINE.CORRELATION_WINDOW,
        n_bars=n_bars,
        random_seed=random_seed,
    )


# =============================================================================
# EXPERIMENT BATCH
# =============================================================================

@dataclass(frozen=True)
class ExperimentBatch:
    """
    A batch of related experiments for systematic evaluation.
    
    All experiments in a batch MUST vary the same dimension.
    """
    batch_id: str
    dimension: ExperimentDimension
    experiments: Tuple[ExperimentConfig, ...]
    description: str = ""
    
    def __post_init__(self):
        # Validate all experiments vary the declared dimension
        for exp in self.experiments:
            if exp.dimension_varied != self.dimension:
                raise ValueError(
                    f"Experiment {exp.experiment_id} varies {exp.dimension_varied}, "
                    f"but batch dimension is {self.dimension}"
                )
