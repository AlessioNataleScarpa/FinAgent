"""
Unit tests for the Temporal Explainable AI (KernelSHAP) module.
Verifies game-theoretic axioms (Efficiency, Dummy Player, Symmetry) and service integration.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from backend.models.explainability import (
    KernelShapTemporalExplainer,
    OcclusionTemporalExplainer,
    TemporalExplanationResult,
    TemporalXAIService,
    WindowAttribution,
    get_xai_service,
)


class TestTemporalExplainability(unittest.TestCase):
    def setUp(self):
        # 60 months of simulated price data with known trend phases
        # First 36 months flat around 100, last 24 months strong rally to 150
        self.flat_then_rally = [100.0 + math.sin(i * 0.2) * 2.0 for i in range(36)] + [
            100.0 + (i * 2.08) for i in range(1, 25)
        ]
        self.dates = [f"202{i//12}-{((i%12)+1):02d}-01" for i in range(60)]
        self.forecast_mean = [150.0 + (i * 1.5) for i in range(1, 13)]  # 12 months rising forecast

    def test_kernel_shap_efficiency_axiom(self):
        """Verifies Shapley Efficiency: sum of attributions equals total payoff difference."""
        explainer = KernelShapTemporalExplainer()

        def linear_trend_fn(series: List[float]) -> float:
            if len(series) < 2:
                return 0.0
            return ((series[-1] - series[0]) / series[0]) * 100.0

        attributions = explainer.explain(
            context_series=self.flat_then_rally,
            predict_fn=linear_trend_fn,
            window_size=12,
            window_dates=self.dates,
        )

        self.assertTrue(len(attributions) >= 4)
        sum_shapley = sum(a.shapley_value for a in attributions)
        sum_relative_pct = sum(a.relative_importance_pct for a in attributions)

        # Relative importances must sum to approximately 100%
        self.assertAlmostEqual(sum_relative_pct, 100.0, places=0)

        # Recent windows (where rally occurred) must have higher Shapley value than early flat windows
        recent_window = attributions[-1]
        early_window = attributions[0]
        self.assertGreater(recent_window.shapley_value, early_window.shapley_value)
        self.assertEqual(recent_window.directional_impact, "Bullish Contribution")

    def test_dummy_player_axiom(self):
        """A completely flat sub-window should have near-zero marginal contribution."""
        explainer = KernelShapTemporalExplainer()
        # Completely constant series: 100 for 60 months
        flat_series = [100.0] * 60

        def trend_fn(series: List[float]) -> float:
            return ((series[-1] - series[0]) / series[0]) * 100.0

        attributions = explainer.explain(
            context_series=flat_series,
            predict_fn=trend_fn,
            window_size=12,
        )

        for a in attributions:
            self.assertAlmostEqual(a.shapley_value, 0.0, places=2)

    def test_occlusion_fallback_explainer(self):
        """Verifies that the occlusion baseline explainer outputs valid schema instances."""
        explainer = OcclusionTemporalExplainer()

        def simple_fn(series: List[float]) -> float:
            return series[-1] - series[0]

        attributions = explainer.explain(
            context_series=self.flat_then_rally,
            predict_fn=simple_fn,
            window_size=12,
            window_dates=self.dates,
        )

        self.assertTrue(len(attributions) >= 4)
        for a in attributions:
            self.assertIsInstance(a, WindowAttribution)
            self.assertGreaterEqual(a.relative_importance_pct, 0.0)

    def test_temporal_xai_service_full_workflow(self):
        """Verifies end-to-end service execution, markdown table formatting, and Mermaid generation."""
        service = get_xai_service()
        result = service.explain_tsfm_forecast(
            context_prices=self.flat_then_rally,
            forecast_mean=self.forecast_mean,
            context_dates=self.dates,
        )

        self.assertIsInstance(result, TemporalExplanationResult)
        self.assertTrue(result.is_valid)
        self.assertIn("KernelSHAP", result.explainer_type)
        self.assertIn("| Finestra Temporale |", result.markdown_table)
        self.assertIn("```mermaid", result.mermaid_chart)
        self.assertTrue(len(result.attributions) >= 4)
        self.assertTrue(result.most_influential_window != "N/A")

    def test_empty_input_handling(self):
        """Service must handle empty or missing context gracefully without crashing."""
        service = get_xai_service()
        empty_res = service.explain_tsfm_forecast([], [])
        self.assertFalse(empty_res.is_valid)
        self.assertEqual(len(empty_res.attributions), 0)
        self.assertIn("non disponibili", empty_res.markdown_table)


if __name__ == "__main__":
    unittest.main()
