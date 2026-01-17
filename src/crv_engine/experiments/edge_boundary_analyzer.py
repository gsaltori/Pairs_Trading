"""
Edge Boundary Analyzer

Analyzes walk-forward results to identify failure boundaries and collapse conditions.

OPERATES ON:
- Walk-forward block results (already computed)
- Per-prediction observables (already captured)

DOES NOT:
- Reprocess raw price data
- Add new indicators
- Modify any thresholds
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timezone
from pathlib import Path
import json
import math

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
    compute_bucket_metrics,
    derive_failure_threshold,
    classify_collapse,
    identify_safe_zone,
)
from .walk_forward import WalkForwardOutput, BlockResult


# =============================================================================
# EDGE BOUNDARY ANALYZER
# =============================================================================

class EdgeBoundaryAnalyzer:
    """
    Analyzes edge failure boundaries from walk-forward results.
    
    RESPONSIBILITIES:
    1. Segment predictions by observable buckets
    2. Compute conditional metrics
    3. Derive empirical failure surfaces
    4. Classify collapse pattern
    5. Identify safe zone (if any)
    
    DOES NOT:
    - Modify walk-forward results
    - Add filters
    - Improve anything
    """
    
    def __init__(
        self,
        crr_threshold: float = 0.55,
        max_inv_rate: float = 0.40,
        min_bucket_samples: int = 10,
        min_failure_samples: int = 15,
        verbose: bool = False,
    ):
        self.crr_threshold = crr_threshold
        self.max_inv_rate = max_inv_rate
        self.min_bucket_samples = min_bucket_samples
        self.min_failure_samples = min_failure_samples
        self.verbose = verbose
    
    def analyze(
        self,
        predictions: List[PredictionObservables],
        walk_forward_output: Optional[WalkForwardOutput] = None,
    ) -> EdgeBoundaryOutput:
        """
        Perform complete edge boundary analysis.
        
        Args:
            predictions: List of predictions with observables
            walk_forward_output: Optional walk-forward results for context
        
        Returns:
            EdgeBoundaryOutput with complete analysis
        """
        if self.verbose:
            print("=" * 70)
            print("EDGE BOUNDARY ANALYSIS")
            print("=" * 70)
            print(f"Total predictions: {len(predictions)}")
        
        # Overall metrics
        testable = [p for p in predictions if p.is_testable]
        confirmed = [p for p in predictions if p.is_confirmed]
        invalidated = [p for p in predictions if p.is_invalidated]
        
        overall_crr = len(confirmed) / len(testable) if testable else 0.0
        overall_inv = len(invalidated) / len(predictions) if predictions else 0.0
        
        if self.verbose:
            print(f"Overall CRR: {overall_crr:.1%}")
            print(f"Overall Invalidation: {overall_inv:.1%}")
            print()
        
        # ─────────────────────────────────────────────────────────────────────
        # 1. BUCKET METRICS
        # ─────────────────────────────────────────────────────────────────────
        if self.verbose:
            print("1. Computing bucket metrics...")
        
        corr_level_metrics = self._compute_correlation_level_metrics(predictions)
        corr_trend_metrics = self._compute_correlation_trend_metrics(predictions)
        vol_ratio_metrics = self._compute_volatility_ratio_metrics(predictions)
        spread_metrics = self._compute_spread_state_metrics(predictions)
        regime_metrics = self._compute_regime_transition_metrics(predictions)
        
        # ─────────────────────────────────────────────────────────────────────
        # 2. FAILURE SURFACES
        # ─────────────────────────────────────────────────────────────────────
        if self.verbose:
            print("2. Deriving failure surfaces...")
        
        failure_surfaces = self._derive_failure_surfaces(predictions)
        
        # ─────────────────────────────────────────────────────────────────────
        # 3. NECESSARY FAILURE CONDITIONS
        # ─────────────────────────────────────────────────────────────────────
        if self.verbose:
            print("3. Identifying necessary failure conditions...")
        
        necessary_conditions = self._identify_necessary_conditions(
            corr_level_metrics,
            corr_trend_metrics,
            vol_ratio_metrics,
            spread_metrics,
            failure_surfaces,
        )
        
        # ─────────────────────────────────────────────────────────────────────
        # 4. NON-FAILURE REGIONS
        # ─────────────────────────────────────────────────────────────────────
        if self.verbose:
            print("4. Mapping non-failure regions...")
        
        non_failure_regions = self._identify_non_failure_regions(
            corr_level_metrics,
            corr_trend_metrics,
            vol_ratio_metrics,
            spread_metrics,
        )
        
        # ─────────────────────────────────────────────────────────────────────
        # 5. COLLAPSE CLASSIFICATION
        # ─────────────────────────────────────────────────────────────────────
        if self.verbose:
            print("5. Classifying collapse pattern...")
        
        collapse_analysis = self._classify_collapse(predictions, walk_forward_output)
        
        # ─────────────────────────────────────────────────────────────────────
        # 6. SAFE ZONE
        # ─────────────────────────────────────────────────────────────────────
        if self.verbose:
            print("6. Identifying safe zone...")
        
        safe_zone = identify_safe_zone(
            predictions,
            crr_threshold=self.crr_threshold,
            max_inv_rate=self.max_inv_rate,
            min_samples=self.min_bucket_samples * 2,
        )
        
        # ─────────────────────────────────────────────────────────────────────
        # 7. FLAGS
        # ─────────────────────────────────────────────────────────────────────
        flags = self._compute_flags(
            failure_surfaces,
            necessary_conditions,
            safe_zone,
            collapse_analysis,
        )
        
        # Build output
        total_blocks = 0
        if walk_forward_output:
            total_blocks = len(walk_forward_output.block_results)
        
        output = EdgeBoundaryOutput(
            schema_version=EDGE_BOUNDARY_SCHEMA_VERSION,
            analysis_timestamp=datetime.now(timezone.utc).isoformat(),
            total_predictions=len(predictions),
            total_blocks=total_blocks,
            overall_crr=overall_crr,
            overall_inv_rate=overall_inv,
            correlation_level_metrics=[m.to_dict() for m in corr_level_metrics],
            correlation_trend_metrics=[m.to_dict() for m in corr_trend_metrics],
            volatility_ratio_metrics=[m.to_dict() for m in vol_ratio_metrics],
            spread_state_metrics=[m.to_dict() for m in spread_metrics],
            regime_transition_metrics=[m.to_dict() for m in regime_metrics],
            failure_surfaces=[f.to_dict() for f in failure_surfaces],
            necessary_failure_conditions=necessary_conditions,
            non_failure_regions=non_failure_regions,
            edge_safe_zone=safe_zone.to_dict() if safe_zone else None,
            collapse_analysis=collapse_analysis.to_dict(),
            flags=flags,
        )
        
        if self.verbose:
            print()
            print("=" * 70)
            print("ANALYSIS COMPLETE")
            print("=" * 70)
            self._print_summary(output)
        
        return output
    
    # =========================================================================
    # BUCKET METRICS COMPUTATION
    # =========================================================================
    
    def _compute_correlation_level_metrics(
        self,
        predictions: List[PredictionObservables],
    ) -> List[BucketMetrics]:
        """Compute metrics segmented by correlation level."""
        metrics = []
        for bucket in CorrelationBucket:
            filtered = [p for p in predictions if p.correlation_bucket == bucket]
            m = compute_bucket_metrics(
                bucket_name="correlation_level",
                bucket_value=bucket.value,
                predictions=filtered,
                crr_threshold=self.crr_threshold,
            )
            metrics.append(m)
        return metrics
    
    def _compute_correlation_trend_metrics(
        self,
        predictions: List[PredictionObservables],
    ) -> List[BucketMetrics]:
        """Compute metrics segmented by correlation trend."""
        metrics = []
        for bucket in CorrelationTrendBucket:
            filtered = [p for p in predictions if p.correlation_trend_bucket == bucket]
            m = compute_bucket_metrics(
                bucket_name="correlation_trend",
                bucket_value=bucket.value,
                predictions=filtered,
                crr_threshold=self.crr_threshold,
            )
            metrics.append(m)
        return metrics
    
    def _compute_volatility_ratio_metrics(
        self,
        predictions: List[PredictionObservables],
    ) -> List[BucketMetrics]:
        """Compute metrics segmented by volatility ratio."""
        metrics = []
        for bucket in VolatilityRatioBucket:
            filtered = [p for p in predictions if p.volatility_bucket == bucket]
            m = compute_bucket_metrics(
                bucket_name="volatility_ratio",
                bucket_value=bucket.value,
                predictions=filtered,
                crr_threshold=self.crr_threshold,
            )
            metrics.append(m)
        return metrics
    
    def _compute_spread_state_metrics(
        self,
        predictions: List[PredictionObservables],
    ) -> List[BucketMetrics]:
        """Compute metrics segmented by spread state."""
        metrics = []
        for bucket in SpreadBucket:
            filtered = [p for p in predictions if p.spread_bucket == bucket]
            m = compute_bucket_metrics(
                bucket_name="spread_state",
                bucket_value=bucket.value,
                predictions=filtered,
                crr_threshold=self.crr_threshold,
            )
            metrics.append(m)
        return metrics
    
    def _compute_regime_transition_metrics(
        self,
        predictions: List[PredictionObservables],
    ) -> List[BucketMetrics]:
        """Compute metrics segmented by regime transition."""
        metrics = []
        for bucket in RegimeTransitionBucket:
            filtered = [p for p in predictions if p.regime_transition_bucket == bucket]
            m = compute_bucket_metrics(
                bucket_name="regime_transition",
                bucket_value=bucket.value,
                predictions=filtered,
                crr_threshold=self.crr_threshold,
            )
            metrics.append(m)
        return metrics
    
    # =========================================================================
    # FAILURE SURFACE DERIVATION
    # =========================================================================
    
    def _derive_failure_surfaces(
        self,
        predictions: List[PredictionObservables],
    ) -> List[FailureSurface]:
        """Derive empirical failure thresholds."""
        surfaces = []
        
        # Correlation level failure (below threshold)
        corr_result = derive_failure_threshold(
            observable_values=[p.correlation for p in predictions],
            outcomes=[p.outcome for p in predictions],
            crr_threshold=self.crr_threshold,
            min_samples=self.min_failure_samples,
            search_direction="below",
        )
        
        if corr_result:
            threshold, crr, inv_rate, n = corr_result
            surfaces.append(FailureSurface(
                observable_name="correlation",
                threshold_value=threshold,
                threshold_direction="below",
                failure_crr=crr,
                failure_inv_rate=inv_rate,
                sample_size=n,
                confidence=min(1.0, n / 50),
                description=f"Edge fails when correlation < {threshold:.2f}",
            ))
        
        # Volatility ratio failure (extreme imbalance)
        vol_result = derive_failure_threshold(
            observable_values=[p.volatility_ratio for p in predictions],
            outcomes=[p.outcome for p in predictions],
            crr_threshold=self.crr_threshold,
            min_samples=self.min_failure_samples,
            search_direction="above",
        )
        
        if vol_result:
            threshold, crr, inv_rate, n = vol_result
            surfaces.append(FailureSurface(
                observable_name="volatility_ratio",
                threshold_value=threshold,
                threshold_direction="above",
                failure_crr=crr,
                failure_inv_rate=inv_rate,
                sample_size=n,
                confidence=min(1.0, n / 50),
                description=f"Edge fails when volatility_ratio > {threshold:.2f}",
            ))
        
        # Spread extreme failure
        spread_result = derive_failure_threshold(
            observable_values=[abs(p.zscore) for p in predictions],
            outcomes=[p.outcome for p in predictions],
            crr_threshold=self.crr_threshold,
            min_samples=self.min_failure_samples,
            search_direction="above",
        )
        
        if spread_result:
            threshold, crr, inv_rate, n = spread_result
            surfaces.append(FailureSurface(
                observable_name="spread_zscore_abs",
                threshold_value=threshold,
                threshold_direction="above",
                failure_crr=crr,
                failure_inv_rate=inv_rate,
                sample_size=n,
                confidence=min(1.0, n / 50),
                description=f"Edge fails when |Z-score| > {threshold:.2f}",
            ))
        
        # Correlation trend failure (deteriorating)
        trend_result = derive_failure_threshold(
            observable_values=[p.correlation_trend for p in predictions],
            outcomes=[p.outcome for p in predictions],
            crr_threshold=self.crr_threshold,
            min_samples=self.min_failure_samples,
            search_direction="below",
        )
        
        if trend_result:
            threshold, crr, inv_rate, n = trend_result
            surfaces.append(FailureSurface(
                observable_name="correlation_trend",
                threshold_value=threshold,
                threshold_direction="below",
                failure_crr=crr,
                failure_inv_rate=inv_rate,
                sample_size=n,
                confidence=min(1.0, n / 50),
                description=f"Edge fails when correlation_trend < {threshold:.3f}",
            ))
        
        return surfaces
    
    # =========================================================================
    # NECESSARY CONDITIONS
    # =========================================================================
    
    def _identify_necessary_conditions(
        self,
        corr_level_metrics: List[BucketMetrics],
        corr_trend_metrics: List[BucketMetrics],
        vol_ratio_metrics: List[BucketMetrics],
        spread_metrics: List[BucketMetrics],
        failure_surfaces: List[FailureSurface],
    ) -> Dict[str, Any]:
        """
        Identify necessary conditions for failure.
        
        A necessary condition means: "if this condition is NOT met, edge CAN succeed"
        Contrapositive: "if edge failed, this condition WAS met"
        """
        conditions = {
            'correlation_requirements': [],
            'volatility_requirements': [],
            'spread_requirements': [],
            'combined_requirements': [],
        }
        
        # From bucket metrics: identify buckets where edge ALWAYS fails
        for m in corr_level_metrics:
            if m.testable_count >= self.min_bucket_samples and m.crr < self.crr_threshold:
                conditions['correlation_requirements'].append({
                    'bucket': m.bucket_value,
                    'crr': m.crr,
                    'sample_size': m.testable_count,
                    'interpretation': f"Edge fails in {m.bucket_value} correlation regime",
                })
        
        for m in vol_ratio_metrics:
            if m.testable_count >= self.min_bucket_samples and m.crr < self.crr_threshold:
                conditions['volatility_requirements'].append({
                    'bucket': m.bucket_value,
                    'crr': m.crr,
                    'sample_size': m.testable_count,
                    'interpretation': f"Edge fails in {m.bucket_value} volatility regime",
                })
        
        for m in spread_metrics:
            if m.testable_count >= self.min_bucket_samples and m.crr < self.crr_threshold:
                conditions['spread_requirements'].append({
                    'bucket': m.bucket_value,
                    'crr': m.crr,
                    'sample_size': m.testable_count,
                    'interpretation': f"Edge fails in {m.bucket_value} spread state",
                })
        
        # From failure surfaces: identify threshold conditions
        for surface in failure_surfaces:
            if surface.confidence > 0.5:
                conditions['combined_requirements'].append({
                    'observable': surface.observable_name,
                    'threshold': surface.threshold_value,
                    'direction': surface.threshold_direction,
                    'failure_crr': surface.failure_crr,
                    'description': surface.description,
                })
        
        return conditions
    
    # =========================================================================
    # NON-FAILURE REGIONS
    # =========================================================================
    
    def _identify_non_failure_regions(
        self,
        corr_level_metrics: List[BucketMetrics],
        corr_trend_metrics: List[BucketMetrics],
        vol_ratio_metrics: List[BucketMetrics],
        spread_metrics: List[BucketMetrics],
    ) -> Dict[str, Any]:
        """
        Identify regions where edge does NOT fail.
        
        This is NOT a recommendation. This is cartography.
        """
        regions = {
            'correlation_levels': [],
            'correlation_trends': [],
            'volatility_ratios': [],
            'spread_states': [],
        }
        
        for m in corr_level_metrics:
            if m.testable_count >= self.min_bucket_samples and m.crr >= self.crr_threshold:
                regions['correlation_levels'].append({
                    'bucket': m.bucket_value,
                    'crr': m.crr,
                    'inv_rate': m.invalidation_rate,
                    'sample_size': m.testable_count,
                })
        
        for m in corr_trend_metrics:
            if m.testable_count >= self.min_bucket_samples and m.crr >= self.crr_threshold:
                regions['correlation_trends'].append({
                    'bucket': m.bucket_value,
                    'crr': m.crr,
                    'inv_rate': m.invalidation_rate,
                    'sample_size': m.testable_count,
                })
        
        for m in vol_ratio_metrics:
            if m.testable_count >= self.min_bucket_samples and m.crr >= self.crr_threshold:
                regions['volatility_ratios'].append({
                    'bucket': m.bucket_value,
                    'crr': m.crr,
                    'inv_rate': m.invalidation_rate,
                    'sample_size': m.testable_count,
                })
        
        for m in spread_metrics:
            if m.testable_count >= self.min_bucket_samples and m.crr >= self.crr_threshold:
                regions['spread_states'].append({
                    'bucket': m.bucket_value,
                    'crr': m.crr,
                    'inv_rate': m.invalidation_rate,
                    'sample_size': m.testable_count,
                })
        
        return regions
    
    # =========================================================================
    # COLLAPSE CLASSIFICATION
    # =========================================================================
    
    def _classify_collapse(
        self,
        predictions: List[PredictionObservables],
        walk_forward_output: Optional[WalkForwardOutput],
    ) -> CollapseAnalysis:
        """Classify the collapse pattern."""
        # Extract time series from walk-forward if available
        if walk_forward_output:
            crr_series = [b['crr'] for b in walk_forward_output.block_results 
                         if b['verdict'] != 'INSUFFICIENT_DATA']
            inv_series = [b['invalidation_rate'] for b in walk_forward_output.block_results
                         if b['verdict'] != 'INSUFFICIENT_DATA']
            
            # Compute regime change rate
            regime_changes = 0
            for b in walk_forward_output.block_results:
                if b.get('regime_distribution') and b['regime_distribution'].get('regime_entropy', 0) > 1.0:
                    regime_changes += 1
            regime_change_rate = regime_changes / len(walk_forward_output.block_results) if walk_forward_output.block_results else 0
            
            # Compute CRR variance
            crr_variance = 0.0
            if len(crr_series) > 1:
                mean_crr = sum(crr_series) / len(crr_series)
                crr_variance = sum((c - mean_crr) ** 2 for c in crr_series) / len(crr_series)
        else:
            # Fallback: compute from predictions directly
            # Group by pseudo-blocks
            block_size = len(predictions) // 5 if len(predictions) >= 50 else len(predictions)
            blocks = [predictions[i:i+block_size] for i in range(0, len(predictions), block_size)]
            
            crr_series = []
            inv_series = []
            for block in blocks:
                testable = [p for p in block if p.is_testable]
                if len(testable) >= 5:
                    confirmed = sum(1 for p in testable if p.is_confirmed)
                    crr_series.append(confirmed / len(testable))
                    inv_series.append(sum(1 for p in block if p.is_invalidated) / len(block))
            
            regime_change_rate = 0.0
            crr_variance = 0.0
            if len(crr_series) > 1:
                mean_crr = sum(crr_series) / len(crr_series)
                crr_variance = sum((c - mean_crr) ** 2 for c in crr_series) / len(crr_series)
        
        return classify_collapse(
            crr_series=crr_series,
            invalidation_series=inv_series,
            regime_change_rate=regime_change_rate,
            crr_variance=crr_variance,
        )
    
    # =========================================================================
    # FLAGS
    # =========================================================================
    
    def _compute_flags(
        self,
        failure_surfaces: List[FailureSurface],
        necessary_conditions: Dict,
        safe_zone: Optional[EdgeSafeZone],
        collapse_analysis: CollapseAnalysis,
    ) -> Dict[str, bool]:
        """Compute summary flags."""
        return {
            'has_failure_boundaries': len(failure_surfaces) > 0,
            'has_correlation_dependency': len(necessary_conditions.get('correlation_requirements', [])) > 0,
            'has_volatility_dependency': len(necessary_conditions.get('volatility_requirements', [])) > 0,
            'has_spread_dependency': len(necessary_conditions.get('spread_requirements', [])) > 0,
            'has_safe_zone': safe_zone is not None,
            'safe_zone_is_narrow': safe_zone is not None and safe_zone.coverage_fraction < 0.3,
            'is_gradual_decay': collapse_analysis.collapse_type == CollapseType.GRADUAL_DECAY,
            'is_regime_flip': collapse_analysis.collapse_type == CollapseType.SUDDEN_REGIME_FLIP,
            'is_structurally_unstable': collapse_analysis.collapse_type == CollapseType.STRUCTURAL_INVALIDATION,
            'is_noise_dominated': collapse_analysis.collapse_type == CollapseType.NOISE_INDUCED_INSTABILITY,
            'edge_is_fragile': (
                (safe_zone is not None and safe_zone.coverage_fraction < 0.3) or
                len(failure_surfaces) >= 3
            ),
        }
    
    # =========================================================================
    # PRINT HELPERS
    # =========================================================================
    
    def _print_summary(self, output: EdgeBoundaryOutput):
        """Print analysis summary."""
        print()
        print("FAILURE SURFACES:")
        print("-" * 50)
        if output.failure_surfaces:
            for fs in output.failure_surfaces:
                print(f"  • {fs['description']}")
                print(f"    CRR in failure region: {fs['failure_crr']:.1%}")
                print(f"    Sample size: {fs['sample_size']}")
        else:
            print("  No empirical failure boundaries detected")
        
        print()
        print("SAFE ZONE:")
        print("-" * 50)
        if output.edge_safe_zone:
            sz = output.edge_safe_zone
            print(f"  Correlation: [{sz['correlation_range'][0]:.2f}, {sz['correlation_range'][1]:.2f}]")
            print(f"  Safe zone CRR: {sz['safe_zone_crr']:.1%}")
            print(f"  Coverage: {sz['coverage_fraction']:.1%} of predictions")
        else:
            print("  No safe zone identified (edge may be universally fragile)")
        
        print()
        print("COLLAPSE CLASSIFICATION:")
        print("-" * 50)
        print(f"  Type: {output.collapse_analysis['collapse_type']}")
        print(f"  Confidence: {output.collapse_analysis['confidence']:.1%}")
        print(f"  {output.collapse_analysis['description']}")
        
        print()
        print("FLAGS:")
        print("-" * 50)
        for flag, value in output.flags.items():
            indicator = "✓" if value else "✗"
            print(f"  [{indicator}] {flag}")


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def analyze_edge_boundaries(
    predictions: List[PredictionObservables],
    walk_forward_output: Optional[WalkForwardOutput] = None,
    crr_threshold: float = 0.55,
    verbose: bool = True,
) -> EdgeBoundaryOutput:
    """
    Convenience function to run edge boundary analysis.
    
    Args:
        predictions: List of predictions with observables
        walk_forward_output: Optional walk-forward results
        crr_threshold: CRR threshold for edge success
        verbose: Print progress
    
    Returns:
        EdgeBoundaryOutput with complete analysis
    """
    analyzer = EdgeBoundaryAnalyzer(
        crr_threshold=crr_threshold,
        verbose=verbose,
    )
    
    return analyzer.analyze(predictions, walk_forward_output)
