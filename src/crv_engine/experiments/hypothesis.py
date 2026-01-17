"""
Hypothesis Specification Framework

Provides a strict structure for expressing explicit, falsifiable hypotheses.

DESIGN PRINCIPLES:
1. Binary outcome only: SUPPORTED or REFUTED
2. No ambiguous interpretation
3. Explicit sample-size requirements
4. Clear falsification criteria

NO STRATEGY LOGIC. NO TUNING. HYPOTHESIS DEFINITION ONLY.
"""

from dataclasses import dataclass, field
from typing import Optional, Callable, Any, Dict, List
from enum import Enum
from abc import ABC, abstractmethod


class HypothesisVerdict(Enum):
    """
    Binary hypothesis outcome.
    
    NO AMBIGUITY ALLOWED.
    """
    SUPPORTED = "SUPPORTED"      # Evidence consistent with hypothesis
    REFUTED = "REFUTED"          # Evidence contradicts hypothesis
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"  # Cannot evaluate (NOT a verdict)


class ComparisonOperator(Enum):
    """Operators for threshold comparisons."""
    GREATER_THAN = ">"
    GREATER_THAN_OR_EQUAL = ">="
    LESS_THAN = "<"
    LESS_THAN_OR_EQUAL = "<="
    EQUAL = "=="
    NOT_EQUAL = "!="


# =============================================================================
# FALSIFICATION CRITERIA
# =============================================================================

@dataclass(frozen=True)
class FalsificationCriterion:
    """
    A single, atomic falsification criterion.
    
    The hypothesis is REFUTED if this criterion evaluates to False.
    """
    metric_name: str
    operator: ComparisonOperator
    threshold: float
    description: str
    
    def evaluate(self, metric_value: float) -> bool:
        """
        Evaluate whether the metric value satisfies the criterion.
        
        Returns True if criterion is MET (hypothesis supported on this criterion).
        Returns False if criterion is VIOLATED (hypothesis refuted on this criterion).
        """
        if self.operator == ComparisonOperator.GREATER_THAN:
            return metric_value > self.threshold
        elif self.operator == ComparisonOperator.GREATER_THAN_OR_EQUAL:
            return metric_value >= self.threshold
        elif self.operator == ComparisonOperator.LESS_THAN:
            return metric_value < self.threshold
        elif self.operator == ComparisonOperator.LESS_THAN_OR_EQUAL:
            return metric_value <= self.threshold
        elif self.operator == ComparisonOperator.EQUAL:
            return metric_value == self.threshold
        elif self.operator == ComparisonOperator.NOT_EQUAL:
            return metric_value != self.threshold
        else:
            raise ValueError(f"Unknown operator: {self.operator}")
    
    def __str__(self) -> str:
        return f"{self.metric_name} {self.operator.value} {self.threshold}"


@dataclass(frozen=True)
class SampleSizeRequirement:
    """
    Minimum sample size for valid hypothesis evaluation.
    
    If sample size is below minimum, verdict is INSUFFICIENT_DATA.
    """
    metric_name: str
    minimum_n: int
    rationale: str
    
    def is_satisfied(self, actual_n: int) -> bool:
        return actual_n >= self.minimum_n


# =============================================================================
# HYPOTHESIS SPECIFICATION
# =============================================================================

@dataclass(frozen=True)
class HypothesisSpec:
    """
    Complete specification of a falsifiable hypothesis.
    
    STRUCTURE:
    - hypothesis_id: Unique identifier
    - name: Human-readable name
    - description: Detailed description of what is being tested
    - null_hypothesis: What we are trying to reject
    - alternative_hypothesis: What we accept if null is rejected
    - falsification_criteria: List of criteria that MUST ALL be met
    - sample_requirements: Minimum data requirements
    - expected_outcome: What we expect (for documentation only)
    
    EVALUATION LOGIC:
    1. Check sample size requirements
    2. If any requirement fails → INSUFFICIENT_DATA
    3. Evaluate ALL falsification criteria
    4. If ANY criterion fails → REFUTED
    5. If ALL criteria pass → SUPPORTED
    """
    hypothesis_id: str
    name: str
    description: str
    
    # Statistical framing
    null_hypothesis: str
    alternative_hypothesis: str
    
    # Falsification criteria (ALL must pass for SUPPORTED)
    falsification_criteria: tuple  # Tuple[FalsificationCriterion, ...]
    
    # Sample size requirements
    sample_requirements: tuple  # Tuple[SampleSizeRequirement, ...]
    
    # Expected outcome (documentation only - does not affect evaluation)
    expected_outcome: Optional[str] = None
    
    # Metadata
    author: str = "CRV Research"
    version: str = "1.0"
    
    def evaluate(
        self,
        metrics: Dict[str, float],
        sample_sizes: Dict[str, int],
    ) -> 'HypothesisResult':
        """
        Evaluate the hypothesis against observed metrics.
        
        Args:
            metrics: Dictionary of metric_name -> value
            sample_sizes: Dictionary of metric_name -> sample count
        
        Returns:
            HypothesisResult with verdict and detailed evaluation
        """
        # Check sample requirements first
        sample_checks = []
        for req in self.sample_requirements:
            actual_n = sample_sizes.get(req.metric_name, 0)
            satisfied = req.is_satisfied(actual_n)
            sample_checks.append({
                'requirement': req.metric_name,
                'minimum': req.minimum_n,
                'actual': actual_n,
                'satisfied': satisfied,
            })
            
            if not satisfied:
                return HypothesisResult(
                    hypothesis_id=self.hypothesis_id,
                    verdict=HypothesisVerdict.INSUFFICIENT_DATA,
                    criteria_results=[],
                    sample_checks=sample_checks,
                    summary=f"Insufficient data: {req.metric_name} has {actual_n} samples, need {req.minimum_n}",
                )
        
        # Evaluate falsification criteria
        criteria_results = []
        all_passed = True
        
        for criterion in self.falsification_criteria:
            metric_value = metrics.get(criterion.metric_name)
            
            if metric_value is None:
                criteria_results.append({
                    'criterion': str(criterion),
                    'metric_value': None,
                    'passed': False,
                    'reason': f"Metric {criterion.metric_name} not found",
                })
                all_passed = False
                continue
            
            passed = criterion.evaluate(metric_value)
            criteria_results.append({
                'criterion': str(criterion),
                'metric_value': metric_value,
                'threshold': criterion.threshold,
                'passed': passed,
                'description': criterion.description,
            })
            
            if not passed:
                all_passed = False
        
        # Determine verdict
        if all_passed:
            verdict = HypothesisVerdict.SUPPORTED
            summary = "All falsification criteria passed"
        else:
            verdict = HypothesisVerdict.REFUTED
            failed = [c['criterion'] for c in criteria_results if not c['passed']]
            summary = f"Failed criteria: {', '.join(failed)}"
        
        return HypothesisResult(
            hypothesis_id=self.hypothesis_id,
            verdict=verdict,
            criteria_results=criteria_results,
            sample_checks=sample_checks,
            summary=summary,
        )


# =============================================================================
# HYPOTHESIS RESULT
# =============================================================================

@dataclass
class HypothesisResult:
    """
    Complete result of hypothesis evaluation.
    
    Contains:
    - Binary verdict
    - Detailed evaluation of each criterion
    - Sample size checks
    - Summary explanation
    """
    hypothesis_id: str
    verdict: HypothesisVerdict
    criteria_results: List[Dict]
    sample_checks: List[Dict]
    summary: str
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary for JSON output."""
        return {
            'hypothesis_id': self.hypothesis_id,
            'verdict': self.verdict.value,
            'criteria_results': self.criteria_results,
            'sample_checks': self.sample_checks,
            'summary': self.summary,
        }
    
    @property
    def is_supported(self) -> bool:
        return self.verdict == HypothesisVerdict.SUPPORTED
    
    @property
    def is_refuted(self) -> bool:
        return self.verdict == HypothesisVerdict.REFUTED
    
    @property
    def has_insufficient_data(self) -> bool:
        return self.verdict == HypothesisVerdict.INSUFFICIENT_DATA


# =============================================================================
# PREDEFINED HYPOTHESIS TEMPLATES
# =============================================================================

def create_edge_exists_hypothesis(
    hypothesis_id: str,
    min_crr: float = 0.55,
    min_testable: int = 30,
    max_invalidation_rate: float = 0.40,
) -> HypothesisSpec:
    """
    Create a hypothesis that tests whether the engine has positive edge.
    
    FALSIFICATION CRITERIA:
    1. CRR must exceed threshold (evidence of directional accuracy)
    2. Invalidation rate must be below threshold (structural stability)
    
    This is the CORE hypothesis for the CRV engine.
    """
    return HypothesisSpec(
        hypothesis_id=hypothesis_id,
        name="Edge Exists",
        description=(
            "Test whether the CRV engine produces predictions with positive edge. "
            "Edge is defined as CRR significantly above random (0.50) with "
            "acceptable invalidation rate."
        ),
        null_hypothesis="CRR ≤ 0.50 (no better than random)",
        alternative_hypothesis=f"CRR > {min_crr} (positive edge)",
        falsification_criteria=(
            FalsificationCriterion(
                metric_name="crr",
                operator=ComparisonOperator.GREATER_THAN_OR_EQUAL,
                threshold=min_crr,
                description=f"CRR must be at least {min_crr:.0%}",
            ),
            FalsificationCriterion(
                metric_name="invalidation_rate",
                operator=ComparisonOperator.LESS_THAN_OR_EQUAL,
                threshold=max_invalidation_rate,
                description=f"Invalidation rate must not exceed {max_invalidation_rate:.0%}",
            ),
        ),
        sample_requirements=(
            SampleSizeRequirement(
                metric_name="testable_count",
                minimum_n=min_testable,
                rationale=f"Need at least {min_testable} testable predictions for statistical validity",
            ),
        ),
        expected_outcome="Hypothesis testing framework - no expected outcome specified",
    )


def create_timeframe_invariance_hypothesis(
    hypothesis_id: str,
    baseline_crr: float,
    tolerance: float = 0.10,
    min_testable: int = 30,
) -> HypothesisSpec:
    """
    Create a hypothesis that edge is invariant to timeframe change.
    
    FALSIFICATION: CRR deviation from baseline exceeds tolerance.
    """
    return HypothesisSpec(
        hypothesis_id=hypothesis_id,
        name="Timeframe Invariance",
        description=(
            "Test whether edge is preserved when changing timeframe. "
            "A robust edge should not be highly sensitive to timeframe selection."
        ),
        null_hypothesis=f"CRR deviation from baseline > {tolerance:.0%}",
        alternative_hypothesis=f"CRR deviation from baseline ≤ {tolerance:.0%}",
        falsification_criteria=(
            FalsificationCriterion(
                metric_name="crr",
                operator=ComparisonOperator.GREATER_THAN_OR_EQUAL,
                threshold=baseline_crr - tolerance,
                description=f"CRR must not fall below {baseline_crr - tolerance:.0%}",
            ),
            FalsificationCriterion(
                metric_name="crr",
                operator=ComparisonOperator.LESS_THAN_OR_EQUAL,
                threshold=baseline_crr + tolerance,
                description=f"CRR must not exceed {baseline_crr + tolerance:.0%} (overfitting indicator)",
            ),
        ),
        sample_requirements=(
            SampleSizeRequirement(
                metric_name="testable_count",
                minimum_n=min_testable,
                rationale=f"Need at least {min_testable} testable predictions",
            ),
        ),
    )


def create_confirmation_speed_hypothesis(
    hypothesis_id: str,
    max_median_bars_to_confirm: int = 25,
    min_confirmed: int = 15,
) -> HypothesisSpec:
    """
    Create a hypothesis about confirmation speed.
    
    FALSIFICATION: Median bars-to-confirmation exceeds threshold.
    
    RATIONALE: If confirmations take too long, the prediction
    may not be capturing true reversion but random drift.
    """
    return HypothesisSpec(
        hypothesis_id=hypothesis_id,
        name="Confirmation Speed",
        description=(
            "Test whether confirmations occur within reasonable time. "
            "Fast confirmations suggest genuine reversion; slow confirmations "
            "may indicate random drift rather than predictable behavior."
        ),
        null_hypothesis=f"Median bars-to-confirmation > {max_median_bars_to_confirm}",
        alternative_hypothesis=f"Median bars-to-confirmation ≤ {max_median_bars_to_confirm}",
        falsification_criteria=(
            FalsificationCriterion(
                metric_name="median_bars_to_confirmation",
                operator=ComparisonOperator.LESS_THAN_OR_EQUAL,
                threshold=max_median_bars_to_confirm,
                description=f"Median confirmation time must not exceed {max_median_bars_to_confirm} bars",
            ),
        ),
        sample_requirements=(
            SampleSizeRequirement(
                metric_name="confirmed_count",
                minimum_n=min_confirmed,
                rationale=f"Need at least {min_confirmed} confirmations to compute reliable median",
            ),
        ),
    )
