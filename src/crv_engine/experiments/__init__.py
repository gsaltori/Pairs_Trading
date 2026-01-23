"""
CRV Experiment Harness

A reproducible experiment platform for evaluating the CRV engine
under controlled variation.
"""

# Core experiment framework
from .config import (
    Timeframe,
    ExperimentDimension,
    BaselineConfig,
    ExperimentConfig,
    ExperimentBatch,
    BASELINE,
    create_baseline_experiment,
    create_symbol_pair_experiment,
    create_timeframe_experiment,
    create_zscore_window_experiment,
    create_correlation_window_experiment,
    create_seed_experiment,
)

from .hypothesis import (
    HypothesisVerdict,
    ComparisonOperator,
    FalsificationCriterion,
    SampleSizeRequirement,
    HypothesisSpec,
    HypothesisResult,
    create_edge_exists_hypothesis,
    create_timeframe_invariance_hypothesis,
    create_confirmation_speed_hypothesis,
)

from .metrics import (
    ExtendedMetrics,
    ResolutionRecord,
    compute_extended_metrics,
    interpret_erv,
    interpret_timing_asymmetry,
)

from .output import (
    SCHEMA_VERSION,
    ExperimentOutput,
    ExperimentOutputBuilder,
    AggregatedResults,
    aggregate_experiments,
)

from .runner import (
    ExperimentRunner,
    run_single_experiment,
    run_batch_experiment,
)

# Walk-forward temporal analysis
from .walk_forward import (
    WalkForwardConfig,
    BlockResult,
    BlockVerdict,
    RegimeDistribution,
    TemporalStabilityMetrics,
    EdgeStabilityClass,
    WalkForwardOutput,
    WALK_FORWARD_SCHEMA_VERSION,
    compute_temporal_stability,
)

from .walk_forward_runner import (
    WalkForwardRunner,
    WalkForwardResult,
    BlockExecutor,
    BlockExecutionResult,
    run_walk_forward,
)

# Edge boundary analysis
from .edge_boundary import (
    PredictionObservables,
    BucketMetrics,
    FailureSurface,
    CollapseAnalysis,
    CollapseType,
    EdgeSafeZone,
    EdgeBoundaryOutput,
    EDGE_BOUNDARY_SCHEMA_VERSION,
    CorrelationBucket,
    CorrelationTrendBucket,
    VolatilityRatioBucket,
    SpreadBucket,
    RegimeTransitionBucket,
    classify_correlation_level,
    classify_correlation_trend,
    classify_volatility_ratio,
    classify_spread_state,
    compute_bucket_metrics,
    derive_failure_threshold,
    classify_collapse,
    identify_safe_zone,
)

from .edge_boundary_analyzer import (
    EdgeBoundaryAnalyzer,
    analyze_edge_boundaries,
)

# Trade Gatekeeper (risk firewall)
from .trade_gatekeeper import (
    TradeGatekeeper,
    TradePermission,
    BlockReason,
    create_gatekeeper,
    check_trade_permission,
)


__all__ = [
    # Config
    'Timeframe',
    'ExperimentDimension',
    'BaselineConfig',
    'ExperimentConfig',
    'ExperimentBatch',
    'BASELINE',
    'create_baseline_experiment',
    'create_symbol_pair_experiment',
    'create_timeframe_experiment',
    'create_zscore_window_experiment',
    'create_correlation_window_experiment',
    'create_seed_experiment',
    
    # Hypothesis
    'HypothesisVerdict',
    'ComparisonOperator',
    'FalsificationCriterion',
    'SampleSizeRequirement',
    'HypothesisSpec',
    'HypothesisResult',
    'create_edge_exists_hypothesis',
    'create_timeframe_invariance_hypothesis',
    'create_confirmation_speed_hypothesis',
    
    # Metrics
    'ExtendedMetrics',
    'ResolutionRecord',
    'compute_extended_metrics',
    'interpret_erv',
    'interpret_timing_asymmetry',
    
    # Output
    'SCHEMA_VERSION',
    'ExperimentOutput',
    'ExperimentOutputBuilder',
    'AggregatedResults',
    'aggregate_experiments',
    
    # Runner
    'ExperimentRunner',
    'run_single_experiment',
    'run_batch_experiment',
    
    # Walk-Forward
    'WalkForwardConfig',
    'BlockResult',
    'BlockVerdict',
    'RegimeDistribution',
    'TemporalStabilityMetrics',
    'EdgeStabilityClass',
    'WalkForwardOutput',
    'WALK_FORWARD_SCHEMA_VERSION',
    'compute_temporal_stability',
    'WalkForwardRunner',
    'WalkForwardResult',
    'BlockExecutor',
    'BlockExecutionResult',
    'run_walk_forward',
    
    # Edge Boundary
    'PredictionObservables',
    'BucketMetrics',
    'FailureSurface',
    'CollapseAnalysis',
    'CollapseType',
    'EdgeSafeZone',
    'EdgeBoundaryOutput',
    'EDGE_BOUNDARY_SCHEMA_VERSION',
    'CorrelationBucket',
    'CorrelationTrendBucket',
    'VolatilityRatioBucket',
    'SpreadBucket',
    'RegimeTransitionBucket',
    'classify_correlation_level',
    'classify_correlation_trend',
    'classify_volatility_ratio',
    'classify_spread_state',
    'compute_bucket_metrics',
    'derive_failure_threshold',
    'classify_collapse',
    'identify_safe_zone',
    'EdgeBoundaryAnalyzer',
    'analyze_edge_boundaries',
    
    # Trade Gatekeeper
    'TradeGatekeeper',
    'TradePermission',
    'BlockReason',
    'create_gatekeeper',
    'check_trade_permission',
]
