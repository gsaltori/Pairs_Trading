"""
Experiment Output Schema

Defines structured output format for experiment results.

REQUIREMENTS:
- Machine-readable (JSON)
- One file per experiment
- Includes configuration hash
- Includes hypothesis verdict
- Supports cross-experiment comparison
- Supports statistical aggregation
- Supports post-hoc audit
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import hashlib
from pathlib import Path
import math

from .config import ExperimentConfig
from .hypothesis import HypothesisResult, HypothesisVerdict
from .metrics import ExtendedMetrics


# Local stat functions to avoid import conflict with local statistics.py
def _mean(data: List[float]) -> float:
    return sum(data) / len(data) if data else 0.0

def _stdev(data: List[float]) -> float:
    if len(data) < 2:
        return 0.0
    m = _mean(data)
    return math.sqrt(sum((x - m) ** 2 for x in data) / (len(data) - 1))


# =============================================================================
# OUTPUT SCHEMA VERSION
# =============================================================================

SCHEMA_VERSION = "1.0.0"


# =============================================================================
# EXPERIMENT OUTPUT
# =============================================================================

@dataclass
class ExperimentOutput:
    """
    Complete output of a single experiment run.
    
    This is the canonical format for all experiment results.
    Designed for machine readability and audit trail.
    """
    # ─────────────────────────────────────────────────────────────────────────
    # IDENTIFICATION
    # ─────────────────────────────────────────────────────────────────────────
    schema_version: str
    experiment_id: str
    config_hash: str
    run_timestamp: str  # ISO format
    
    # ─────────────────────────────────────────────────────────────────────────
    # CONFIGURATION (for reproducibility)
    # ─────────────────────────────────────────────────────────────────────────
    config: Dict[str, Any]
    
    # ─────────────────────────────────────────────────────────────────────────
    # RAW RESULTS
    # ─────────────────────────────────────────────────────────────────────────
    metrics: Dict[str, Any]
    
    # ─────────────────────────────────────────────────────────────────────────
    # HYPOTHESIS VERDICTS
    # ─────────────────────────────────────────────────────────────────────────
    hypothesis_results: List[Dict[str, Any]]
    
    # ─────────────────────────────────────────────────────────────────────────
    # EXECUTION METADATA
    # ─────────────────────────────────────────────────────────────────────────
    execution_time_seconds: float
    engine_version: str
    data_source: str  # "synthetic" or "mt5_live"
    
    # ─────────────────────────────────────────────────────────────────────────
    # AUDIT TRAIL
    # ─────────────────────────────────────────────────────────────────────────
    warnings: List[str]
    errors: List[str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'schema_version': self.schema_version,
            'experiment_id': self.experiment_id,
            'config_hash': self.config_hash,
            'run_timestamp': self.run_timestamp,
            'config': self.config,
            'metrics': self.metrics,
            'hypothesis_results': self.hypothesis_results,
            'execution_time_seconds': self.execution_time_seconds,
            'engine_version': self.engine_version,
            'data_source': self.data_source,
            'warnings': self.warnings,
            'errors': self.errors,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def save(self, output_dir: Path) -> Path:
        """
        Save to file in output directory.
        
        Filename format: {experiment_id}_{config_hash}_{timestamp}.json
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = self.run_timestamp.replace(':', '-').replace('.', '-')
        filename = f"{self.experiment_id}_{self.config_hash}_{timestamp}.json"
        filepath = output_dir / filename
        
        with open(filepath, 'w') as f:
            f.write(self.to_json())
        
        return filepath
    
    @classmethod
    def load(cls, filepath: Path) -> 'ExperimentOutput':
        """Load from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return cls(
            schema_version=data['schema_version'],
            experiment_id=data['experiment_id'],
            config_hash=data['config_hash'],
            run_timestamp=data['run_timestamp'],
            config=data['config'],
            metrics=data['metrics'],
            hypothesis_results=data['hypothesis_results'],
            execution_time_seconds=data['execution_time_seconds'],
            engine_version=data['engine_version'],
            data_source=data['data_source'],
            warnings=data.get('warnings', []),
            errors=data.get('errors', []),
        )
    
    @property
    def primary_verdict(self) -> Optional[str]:
        """
        Return the verdict of the first hypothesis, if any.
        
        Convenience property for simple single-hypothesis experiments.
        """
        if self.hypothesis_results:
            return self.hypothesis_results[0].get('verdict')
        return None
    
    @property
    def all_hypotheses_supported(self) -> bool:
        """Check if all hypotheses were supported."""
        if not self.hypothesis_results:
            return False
        return all(
            h.get('verdict') == HypothesisVerdict.SUPPORTED.value
            for h in self.hypothesis_results
        )


# =============================================================================
# OUTPUT BUILDER
# =============================================================================

class ExperimentOutputBuilder:
    """
    Builder for constructing ExperimentOutput objects.
    
    Ensures all required fields are populated.
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.metrics: Optional[ExtendedMetrics] = None
        self.hypothesis_results: List[HypothesisResult] = []
        self.execution_time: float = 0.0
        self.warnings: List[str] = []
        self.errors: List[str] = []
        self.engine_version: str = "0.3.0"  # From __init__.py
        self.start_time: Optional[datetime] = None
    
    def start(self) -> 'ExperimentOutputBuilder':
        """Mark experiment start time."""
        self.start_time = datetime.utcnow()
        return self
    
    def set_metrics(self, metrics: ExtendedMetrics) -> 'ExperimentOutputBuilder':
        """Set computed metrics."""
        self.metrics = metrics
        return self
    
    def add_hypothesis_result(self, result: HypothesisResult) -> 'ExperimentOutputBuilder':
        """Add a hypothesis evaluation result."""
        self.hypothesis_results.append(result)
        return self
    
    def add_warning(self, warning: str) -> 'ExperimentOutputBuilder':
        """Add a warning message."""
        self.warnings.append(warning)
        return self
    
    def add_error(self, error: str) -> 'ExperimentOutputBuilder':
        """Add an error message."""
        self.errors.append(error)
        return self
    
    def build(self) -> ExperimentOutput:
        """
        Build the final ExperimentOutput.
        
        Raises ValueError if required fields are missing.
        """
        if self.metrics is None:
            raise ValueError("Metrics must be set before building output")
        
        end_time = datetime.utcnow()
        if self.start_time:
            self.execution_time = (end_time - self.start_time).total_seconds()
        
        return ExperimentOutput(
            schema_version=SCHEMA_VERSION,
            experiment_id=self.config.experiment_id,
            config_hash=self.config.config_hash,
            run_timestamp=end_time.isoformat(),
            config={
                'symbol_a': self.config.symbol_a,
                'symbol_b': self.config.symbol_b,
                'timeframe': self.config.timeframe.value,
                'zscore_window': self.config.zscore_window,
                'correlation_window': self.config.correlation_window,
                'n_bars': self.config.n_bars,
                'random_seed': self.config.random_seed,
                'use_synthetic': self.config.use_synthetic,
                'dimension_varied': self.config.dimension_varied.value,
            },
            metrics=self.metrics.to_dict(),
            hypothesis_results=[r.to_dict() for r in self.hypothesis_results],
            execution_time_seconds=self.execution_time,
            engine_version=self.engine_version,
            data_source='synthetic' if self.config.use_synthetic else 'mt5_live',
            warnings=self.warnings,
            errors=self.errors,
        )


# =============================================================================
# AGGREGATION UTILITIES
# =============================================================================

@dataclass
class AggregatedResults:
    """
    Aggregated results across multiple experiments.
    
    For statistical comparison and batch analysis.
    """
    batch_id: str
    dimension_varied: str
    experiment_count: int
    
    # Aggregated metrics
    mean_crr: float
    std_crr: float
    min_crr: float
    max_crr: float
    
    mean_erv: float
    std_erv: float
    
    mean_invalidation_rate: float
    
    # Hypothesis summary
    hypotheses_supported: int
    hypotheses_refuted: int
    hypotheses_insufficient_data: int
    
    # Individual experiment summaries
    experiments: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict:
        return {
            'batch_id': self.batch_id,
            'dimension_varied': self.dimension_varied,
            'experiment_count': self.experiment_count,
            'mean_crr': self.mean_crr,
            'std_crr': self.std_crr,
            'min_crr': self.min_crr,
            'max_crr': self.max_crr,
            'mean_erv': self.mean_erv,
            'std_erv': self.std_erv,
            'mean_invalidation_rate': self.mean_invalidation_rate,
            'hypotheses_supported': self.hypotheses_supported,
            'hypotheses_refuted': self.hypotheses_refuted,
            'hypotheses_insufficient_data': self.hypotheses_insufficient_data,
            'experiments': self.experiments,
        }


def aggregate_experiments(
    batch_id: str,
    outputs: List[ExperimentOutput],
) -> AggregatedResults:
    """
    Aggregate results from multiple experiment outputs.
    """
    if not outputs:
        raise ValueError("Cannot aggregate empty list of outputs")
    
    crrs = [o.metrics['crr'] for o in outputs]
    ervs = [o.metrics['erv_per_prediction'] for o in outputs]
    inv_rates = [o.metrics['invalidation_rate'] for o in outputs]
    
    # Count hypothesis verdicts
    supported = 0
    refuted = 0
    insufficient = 0
    
    for output in outputs:
        for h in output.hypothesis_results:
            if h['verdict'] == HypothesisVerdict.SUPPORTED.value:
                supported += 1
            elif h['verdict'] == HypothesisVerdict.REFUTED.value:
                refuted += 1
            else:
                insufficient += 1
    
    return AggregatedResults(
        batch_id=batch_id,
        dimension_varied=outputs[0].config['dimension_varied'],
        experiment_count=len(outputs),
        mean_crr=_mean(crrs),
        std_crr=_stdev(crrs),
        min_crr=min(crrs),
        max_crr=max(crrs),
        mean_erv=_mean(ervs),
        std_erv=_stdev(ervs),
        mean_invalidation_rate=_mean(inv_rates),
        hypotheses_supported=supported,
        hypotheses_refuted=refuted,
        hypotheses_insufficient_data=insufficient,
        experiments=[
            {
                'experiment_id': o.experiment_id,
                'crr': o.metrics['crr'],
                'erv': o.metrics['erv_per_prediction'],
                'verdict': o.primary_verdict,
            }
            for o in outputs
        ],
    )
