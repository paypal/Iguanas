import polars as pl
import pytest

from iguanas.metrics import compute_metrics
from iguanas.rule_monitoring import compare_rule_metrics


def _make_metrics(predictions: dict, y: pl.Series) -> pl.DataFrame:
    R = pl.DataFrame(predictions)
    return compute_metrics(R, y)


class TestCompareRuleMetrics:
    def test_delta_and_degraded_with_threshold(self):
        y = pl.Series([True, True, True, False])
        ref = _make_metrics({"rule_A": [True, True, True, False]}, y)
        curr = _make_metrics({"rule_A": [True, True, False, False]}, y)
        result = compare_rule_metrics(ref, curr, thresholds={"precision": 0.05})

        assert result.shape[0] == 1
        assert "precision_ref" in result.columns
        assert "precision_curr" in result.columns
        assert "precision_delta" in result.columns
        assert "precision_degraded" in result.columns
        assert result["precision_delta"][0] == pytest.approx(0.0)   # 1.0 → 1.0 (no FP either way)
        assert result["precision_degraded"][0] is False

    def test_degraded_flag_set_when_drop_exceeds_threshold(self):
        y = pl.Series([True, True, True, False])
        ref = _make_metrics({"rule_A": [True, True, True, False]}, y)
        # Current: adds a false positive → precision drops
        curr = _make_metrics({"rule_A": [True, True, True, True]}, y)
        result = compare_rule_metrics(ref, curr, thresholds={"precision": 0.05})

        assert result["precision_delta"][0] < 0
        assert result["precision_degraded"][0] is True

    def test_no_threshold_any_negative_delta_flagged(self):
        y = pl.Series([True, True, False, False])
        ref = _make_metrics({"rule_A": [True, True, False, False]}, y)
        curr = _make_metrics({"rule_A": [True, False, False, False]}, y)
        result = compare_rule_metrics(ref, curr, thresholds=None)

        assert result["recall_delta"][0] < 0
        assert result["recall_degraded"][0] is True

    def test_no_threshold_zero_delta_not_flagged(self):
        y = pl.Series([True, True, False, False])
        ref = _make_metrics({"rule_A": [True, True, False, False]}, y)
        curr = _make_metrics({"rule_A": [True, True, False, False]}, y)
        result = compare_rule_metrics(ref, curr, thresholds=None)

        assert result["precision_delta"][0] == pytest.approx(0.0)
        assert result["precision_degraded"][0] is False

    def test_inner_join_excludes_missing_rules(self):
        y = pl.Series([True, False, True, False])
        ref = _make_metrics({"rule_A": [True, False, True, False], "rule_B": [True, True, False, False]}, y)
        curr = _make_metrics({"rule_A": [True, False, True, False]}, y)
        result = compare_rule_metrics(ref, curr)

        # Only rule_A appears in both — rule_B is excluded
        assert result.shape[0] == 1
        assert result["rule"][0] == "rule_A"

    def test_multiple_rules_compared(self):
        y = pl.Series([True, True, False, False])
        ref = _make_metrics(
            {"rule_A": [True, True, False, False], "rule_B": [True, False, False, False]}, y
        )
        curr = _make_metrics(
            {"rule_A": [True, False, False, False], "rule_B": [True, True, False, False]}, y
        )
        result = compare_rule_metrics(ref, curr)

        assert result.shape[0] == 2
        rule_a = result.filter(pl.col("rule") == "rule_A")
        rule_b = result.filter(pl.col("rule") == "rule_B")
        assert rule_a["recall_delta"][0] < 0
        assert rule_b["recall_delta"][0] > 0

    def test_weighted_metrics_compared(self):
        y = pl.Series([True, True, False, False])
        weights = pl.Series([2.0, 1.0, 1.0, 1.0])
        R = pl.DataFrame({"rule_A": [True, True, False, False]})
        ref = compute_metrics(R, y, weights=weights)
        curr = compute_metrics(
            pl.DataFrame({"rule_A": [True, False, False, False]}), y, weights=weights
        )
        result = compare_rule_metrics(ref, curr)

        assert "precision_weight_ref" in result.columns
        assert "precision_weight_delta" in result.columns
