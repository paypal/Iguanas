import polars as pl
import pytest

from iguanas.metrics import (
    compute_metrics,
    compute_single_metric,
    count_conditions,
    count_features,
)


class TestComputeMetrics:
    """Test cases for compute_metrics function."""

    def test_basic_metrics_without_weights(self):
        """Test basic metric calculation without weights."""
        # Create simple rule predictions and target
        R = pl.DataFrame(
            {
                "rule1": [True, True, False, False, True],
                "rule2": [True, False, True, False, False],
            }
        )
        y = pl.Series([True, True, False, False, True])

        result = compute_metrics(R, y, weights=None)

        # Check expected columns exist
        assert "rule" in result.columns
        assert "TP" in result.columns
        assert "FP" in result.columns
        assert "TN" in result.columns
        assert "FN" in result.columns
        assert "precision" in result.columns
        assert "recall" in result.columns
        assert "flagged(%)" in result.columns
        assert "good_flagged(%)" in result.columns
        assert "f1" in result.columns
        assert "num_rules" in result.columns

        # Check no weighted columns when weights not provided
        assert "TP_weight" not in result.columns
        assert "precision_weight" not in result.columns

        # Check rule1 metrics: TP=2 (idx 0,4), FP=0, TN=2 (idx 2,3), FN=1 (idx 1 wrong, should be idx1 is TP)
        # Actually: y = [T, T, F, F, T], rule1 = [T, T, F, F, T]
        # TP = y & rule1 = [T, T, F, F, T] = 3
        # FP = ~y & rule1 = [F, F, F, F, F] = 0
        # TN = ~y & ~rule1 = [F, F, T, T, F] = 2
        # FN = y & ~rule1 = [F, F, F, F, F] = 0
        rule1_metrics = result.filter(pl.col("rule") == "rule1")
        assert rule1_metrics["TP"][0] == 3
        assert rule1_metrics["FP"][0] == 0
        assert rule1_metrics["TN"][0] == 2
        assert rule1_metrics["FN"][0] == 0
        assert rule1_metrics["precision"][0] == 1.0  # 3/3
        assert rule1_metrics["recall"][0] == 1.0  # 3/3

    def test_metrics_with_weights(self):
        """Test metric calculation with sample weights."""
        R = pl.DataFrame(
            {
                "rule1": [True, True, False, False],
            }
        )
        y = pl.Series([True, False, True, False])
        weights = pl.Series([1.0, 2.0, 3.0, 4.0])

        result = compute_metrics(R, y, weights=weights)

        # Check weighted columns exist
        assert "TP_weight" in result.columns
        assert "FP_weight" in result.columns
        assert "TN_weight" in result.columns
        assert "FN_weight" in result.columns
        assert "precision_weight" in result.columns
        assert "recall_weight" in result.columns
        assert "f1_weight" in result.columns

        # Calculate expected weighted metrics
        # y = [T, F, T, F], rule1 = [T, T, F, F], weights = [1, 2, 3, 4]
        # TP_weight = weights[y & rule1] = weights[[T,F,F,F]] = 1.0
        # FP_weight = weights[~y & rule1] = weights[[F,T,F,F]] = 2.0
        # TN_weight = weights[~y & ~rule1] = weights[[F,F,F,T]] = 4.0
        # FN_weight = weights[y & ~rule1] = weights[[F,F,T,F]] = 3.0
        rule1_metrics = result.filter(pl.col("rule") == "rule1")
        assert rule1_metrics["TP_weight"][0] == 1.0
        assert rule1_metrics["FP_weight"][0] == 2.0
        assert rule1_metrics["TN_weight"][0] == 4.0
        assert rule1_metrics["FN_weight"][0] == 3.0

    def test_all_true_positive(self):
        """Test when rule perfectly identifies all positive cases."""
        R = pl.DataFrame({"perfect_rule": [True, True, False, False]})
        y = pl.Series([True, True, False, False])

        result = compute_metrics(R, y, weights=None)

        metrics = result.filter(pl.col("rule") == "perfect_rule")
        assert metrics["TP"][0] == 2
        assert metrics["FP"][0] == 0
        assert metrics["TN"][0] == 2
        assert metrics["FN"][0] == 0
        assert metrics["precision"][0] == 1.0
        assert metrics["recall"][0] == 1.0

    def test_all_false_positive(self):
        """Test when rule only generates false positives."""
        R = pl.DataFrame({"bad_rule": [False, False, True, True]})
        y = pl.Series([True, True, False, False])

        result = compute_metrics(R, y, weights=None)

        metrics = result.filter(pl.col("rule") == "bad_rule")
        assert metrics["TP"][0] == 0
        assert metrics["FP"][0] == 2
        assert metrics["TN"][0] == 0
        assert metrics["FN"][0] == 2

    def test_no_predictions(self):
        """Test when rule predicts nothing (all False)."""
        R = pl.DataFrame({"no_pred_rule": [False, False, False, False]})
        y = pl.Series([True, True, False, False])

        result = compute_metrics(R, y, weights=None)

        metrics = result.filter(pl.col("rule") == "no_pred_rule")
        assert metrics["TP"][0] == 0
        assert metrics["FP"][0] == 0
        assert metrics["TN"][0] == 2
        assert metrics["FN"][0] == 2

    def test_all_predictions(self):
        """Test when rule predicts everything (all True)."""
        R = pl.DataFrame({"all_pred_rule": [True, True, True, True]})
        y = pl.Series([True, True, False, False])

        result = compute_metrics(R, y, weights=None)

        metrics = result.filter(pl.col("rule") == "all_pred_rule")
        assert metrics["TP"][0] == 2
        assert metrics["FP"][0] == 2
        assert metrics["TN"][0] == 0
        assert metrics["FN"][0] == 0

    def test_multiple_rules(self):
        """Test computing metrics for multiple rules simultaneously."""
        R = pl.DataFrame(
            {
                "rule1": [True, False, True, False],
                "rule2": [False, True, False, True],
                "rule3": [True, True, False, False],
            }
        )
        y = pl.Series([True, False, True, False])

        result = compute_metrics(R, y, weights=None)

        # Should have 3 rows, one for each rule
        assert result.height == 3
        assert set(result["rule"].to_list()) == {"rule1", "rule2", "rule3"}

    def test_custom_betas(self):
        """Test that the betas parameter controls which F-beta columns are produced."""
        R = pl.DataFrame({"rule1": [True, True, False, False]})
        y = pl.Series([True, False, True, False])

        result = compute_metrics(R, y, betas=[0.5, 2, 3])

        assert "f0.5" in result.columns
        assert "f2" in result.columns
        assert "f3" in result.columns

        # Columns outside the custom list must not be present
        assert "f0.25" not in result.columns
        assert "f1" not in result.columns
        assert "f1.5" not in result.columns

    def test_custom_betas_weighted(self):
        """Test that the betas parameter also controls weighted F-beta columns."""
        R = pl.DataFrame({"rule1": [True, True, False, False]})
        y = pl.Series([True, False, True, False])
        weights = pl.Series([1.0, 2.0, 3.0, 4.0])

        result = compute_metrics(R, y, weights=weights, betas=[1, 2])

        assert "f1_weight" in result.columns
        assert "f2_weight" in result.columns
        assert "f0.25_weight" not in result.columns
        assert "f0.5_weight" not in result.columns
        assert "f1.5_weight" not in result.columns

    def test_num_rules_single(self):
        """Test F-beta score calculations using default betas."""
        R = pl.DataFrame({"rule1": [True, True, False, False]})
        y = pl.Series([True, False, True, False])

        result = compute_metrics(R, y, weights=None)

        # Default betas = [0.25, 0.5, 1, 1.5, 2]
        assert "f0.25" in result.columns
        assert "f0.5" in result.columns
        assert "f1" in result.columns
        assert "f1.5" in result.columns
        assert "f2" in result.columns

        # Non-default betas must not be present
        assert "f0.33" not in result.columns
        assert "f0.67" not in result.columns
        assert "f0.8" not in result.columns
        assert "f1.25" not in result.columns
        assert "f3" not in result.columns
        assert "f4" not in result.columns

    def test_num_rules_single(self):
        """Test num_rules for single rule (no OR operator)."""
        R = pl.DataFrame({"rule1": [True, False, True, False]})
        y = pl.Series([True, False, True, False])

        result = compute_metrics(R, y, weights=None)

        # Single rule should have num_rules = 1
        assert result["num_rules"][0] == 1

    def test_num_rules_combined(self):
        """Test num_rules for combined rules (with OR operator)."""
        R = pl.DataFrame(
            {
                "combined_rule": [True, False, True, False],
            }
        )
        # Simulate a combined rule name with OR operator
        R = R.rename({"combined_rule": '(X["a"] > 1) | (X["b"] < 2)'})
        y = pl.Series([True, False, True, False])

        result = compute_metrics(R, y, weights=None)

        # Rule with one "|" should have num_rules = 2
        assert result["num_rules"][0] == 2

    def test_non_boolean_target_conversion(self):
        """Test that non-boolean target is converted to boolean."""
        R = pl.DataFrame({"rule1": [True, True, False, False]})
        # Use integer target instead of boolean
        y = pl.Series([1, 0, 1, 0])

        result = compute_metrics(R, y, weights=None)

        # Should work without error and produce valid results
        assert result.height == 1
        assert "TP" in result.columns

    def test_flagged_percentage(self):
        """Test flagged(%) calculation."""
        R = pl.DataFrame({"rule1": [True, True, False, False]})
        y = pl.Series([True, False, True, False])

        result = compute_metrics(R, y, weights=None)

        # flagged(%) = (TP + FP) / (TP + FP + TN + FN) * 100
        # 2 out of 4 are flagged = 50%
        assert result["flagged(%)"][0] == 50.0

    def test_good_flagged_percentage(self):
        """Test good_flagged(%) calculation."""
        R = pl.DataFrame({"rule1": [True, True, False, False]})
        y = pl.Series([True, False, True, False])
        # y = [T, F, T, F], rule1 = [T, T, F, F]
        # FP = ~y & rule1 = [F, T, F, F] = 1
        # TN = ~y & ~rule1 = [F, F, F, T] = 1

        result = compute_metrics(R, y, weights=None)

        # good_flagged(%) = FP / (TN + FP) * 100
        # 1 / (1 + 1) * 100 = 50%
        assert result["good_flagged(%)"][0] == 50.0

    def test_weighted_fbeta_scores(self):
        """Test weighted F-beta score calculations using default betas."""
        R = pl.DataFrame({"rule1": [True, True, False, False]})
        y = pl.Series([True, False, True, False])
        weights = pl.Series([1.0, 2.0, 3.0, 4.0])

        result = compute_metrics(R, y, weights=weights)

        # Default betas = [0.25, 0.5, 1, 1.5, 2]
        assert "f0.25_weight" in result.columns
        assert "f0.5_weight" in result.columns
        assert "f1_weight" in result.columns
        assert "f1.5_weight" in result.columns
        assert "f2_weight" in result.columns

        # Non-default betas must not be present
        assert "f0.33_weight" not in result.columns
        assert "f3_weight" not in result.columns
        assert "f4_weight" not in result.columns

    def test_total_and_total_weight(self):
        """Test total and total_weight calculations."""
        R = pl.DataFrame({"rule1": [True, False, True, False]})
        y = pl.Series([True, False, True, False])
        weights = pl.Series([1.0, 2.0, 3.0, 4.0])

        result = compute_metrics(R, y, weights=weights)

        # total should be sum of confusion matrix
        expected_total = result["TP"][0] + result["FP"][0] + result["TN"][0] + result["FN"][0]
        # Note: the code doesn't add "total" column explicitly, but it's implied

        # total_weight should be sum of weighted confusion matrix
        expected_total_weight = (
            result["TP_weight"][0]
            + result["FP_weight"][0]
            + result["TN_weight"][0]
            + result["FN_weight"][0]
        )
        assert result["total_weight"][0] == expected_total_weight


class TestComputeSingleMetric:
    def test_accuracy_no_weights(self):
        # TP=1, FP=1, TN=1, FN=1 → accuracy = 2/4 = 0.5
        combined = pl.Series([True, True, False, False])
        y = pl.Series([True, False, True, False])
        result = compute_single_metric(combined, y, "accuracy")
        assert result == pytest.approx(0.5)

    def test_accuracy_with_weights(self):
        # TP_w=2, FP_w=0, TN_w=3, FN_w=0 → accuracy_w = 5/5 = 1.0
        combined = pl.Series([True, False, False, False])
        y = pl.Series([True, False, False, False])
        weights = pl.Series([2.0, 1.0, 1.0, 1.0])
        result = compute_single_metric(combined, y, "accuracy", weights=weights)
        assert result == pytest.approx(1.0)

    def test_mcc_no_weights(self):
        # TP=2, FP=0, TN=2, FN=0 → MCC = (2*2-0*0)/sqrt(2*2*2*2) = 1.0
        combined = pl.Series([True, True, False, False])
        y = pl.Series([True, True, False, False])
        result = compute_single_metric(combined, y, "mcc")
        assert result == pytest.approx(1.0)

    def test_mcc_with_weights(self):
        combined = pl.Series([True, True, False, False])
        y = pl.Series([True, True, False, False])
        weights = pl.Series([1.0, 1.0, 1.0, 1.0])
        result = compute_single_metric(combined, y, "mcc", weights=weights)
        assert result == pytest.approx(1.0)

    def test_mcc_zero_denom_returns_zero(self):
        # All predictions positive, all targets positive → TN=FP=0 → denom=0
        combined = pl.Series([True, True, True])
        y = pl.Series([True, True, True])
        result = compute_single_metric(combined, y, "mcc")
        assert result == 0.0


class TestComputeMetricsSeries:
    def test_series_input_is_converted_to_frame(self):
        """compute_metrics accepts a pl.Series (single rule) and converts it internally."""
        y_pred = pl.Series("my_rule", [True, True, False, False])
        y = pl.Series([True, False, True, False])
        result = compute_metrics(y_pred, y)
        assert result.shape[0] == 1
        assert result["rule"][0] == "my_rule"
        assert "precision" in result.columns
        assert "recall" in result.columns


class TestCoverageAwareMetrics:
    """lift, wracc, laplace and m_estimate: 100 rows, 10 positives."""

    @staticmethod
    def _problem():
        y = pl.Series("y", [True] * 10 + [False] * 90)
        R = pl.DataFrame(
            {
                "tiny_perfect": [True] * 3 + [False] * 97,
                "broad_useless": [True] * 100,
                "good": [True] * 8 + [False] * 2 + [True] * 10 + [False] * 80,
            }
        )
        return R, y

    def test_known_values(self):
        R, y = self._problem()
        m = compute_metrics(R, y).sort("rule")
        by_rule = {row["rule"]: row for row in m.to_dicts()}

        # base rate 0.10; tiny_perfect covers 3 rows, all positive
        assert by_rule["tiny_perfect"]["lift"] == pytest.approx(10.0)
        assert by_rule["tiny_perfect"]["wracc"] == pytest.approx(3 / 100 - (3 * 10) / 10_000)
        assert by_rule["tiny_perfect"]["laplace"] == pytest.approx(4 / 5)
        assert by_rule["good"]["lift"] == pytest.approx(40 / 9)
        assert by_rule["good"]["wracc"] == pytest.approx(8 / 100 - (18 * 10) / 10_000)

    def test_wracc_demotes_a_tiny_perfect_rule(self):
        """Precision ranks a 3-row rule first; WRAcc must not."""
        R, y = self._problem()
        m = compute_metrics(R, y)
        by_rule = {row["rule"]: row for row in m.to_dicts()}

        assert by_rule["tiny_perfect"]["precision"] > by_rule["good"]["precision"]
        assert by_rule["tiny_perfect"]["wracc"] < by_rule["good"]["wracc"]

    def test_flag_everything_scores_zero(self):
        R, y = self._problem()
        by_rule = {row["rule"]: row for row in compute_metrics(R, y).to_dicts()}

        assert by_rule["broad_useless"]["wracc"] == pytest.approx(0.0)
        assert by_rule["broad_useless"]["lift"] == pytest.approx(1.0)

    def test_m_estimate_shrinks_low_coverage_toward_base_rate(self):
        R, y = self._problem()
        by_rule = {row["rule"]: row for row in compute_metrics(R, y).to_dicts()}

        # precision 1.0 but only 3 rows -> pulled far down toward the 0.1 base rate
        assert by_rule["tiny_perfect"]["m_estimate"] < 0.5
        assert by_rule["tiny_perfect"]["m_estimate"] > 0.1

    @pytest.mark.parametrize("metric", ["lift", "wracc", "laplace", "m_estimate"])
    def test_scalar_path_matches_frame_path(self, metric):
        R, y = self._problem()
        frame = compute_metrics(R, y)[metric].to_list()
        scalar = [compute_single_metric(R[c], y, metric) for c in R.columns]
        assert scalar == pytest.approx(frame)

    @pytest.mark.parametrize("metric", ["lift", "wracc"])
    def test_weighted_variants_present(self, metric):
        R, y = self._problem()
        weights = pl.Series("w", [1.0] * 100)
        m = compute_metrics(R, y, weights=weights)
        assert f"{metric}_weight" in m.columns
        assert m[f"{metric}_weight"].to_list() == pytest.approx(m[metric].to_list())

    def test_empty_rule_is_safe(self):
        y = pl.Series("y", [True, False, True])
        R = pl.DataFrame({"never": [False, False, False]})
        row = compute_metrics(R, y).to_dicts()[0]

        assert row["lift"] == pytest.approx(0.0)
        assert row["wracc"] == pytest.approx(0.0)
        assert row["laplace"] == pytest.approx(0.5)


class TestComplexityMetrics:
    """Complexity is counted in conditions and distinct features."""

    @pytest.mark.parametrize(
        "rule, conditions, features",
        [
            ('(X["a"] > 1)', 1, 1),
            ('(X["a"] > 1) & (X["b"] <= 2)', 2, 2),
            ('(X["a"] > 1) & (X["a"] < 5)', 2, 1),
            ("(X['a'] > 1) | (X['b'] > 2) | (X['c'] > 3)", 3, 3),
            ("rule_A", 0, 0),
            ("", 0, 0),
        ],
    )
    def test_counts(self, rule, conditions, features):
        assert count_conditions(rule) == conditions
        assert count_features(rule) == features

    def test_columns_match_helper_functions(self):
        y = pl.Series("y", [True, False, True, False])
        rules = {
            '(X["a"] > 1)': [True, False, True, False],
            '(X["a"] > 1) & (X["a"] < 5)': [True, False, False, False],
            '(X["a"] > 1) | (X["b"] > 2)': [True, True, True, False],
        }
        R = pl.DataFrame(rules)
        m = compute_metrics(R, y)

        assert m["num_conditions"].to_list() == [count_conditions(r) for r in rules]
        assert m["num_features"].to_list() == [count_features(r) for r in rules]

    def test_a_wider_disjunction_is_more_complex(self):
        y = pl.Series("y", [True, False])
        narrow = '(X["a"] > 1)'
        wide = '(X["a"] > 1) | (X["b"] > 2) | (X["c"] > 3)'
        R = pl.DataFrame({narrow: [True, False], wide: [True, True]})
        m = compute_metrics(R, y)

        assert m["num_conditions"].to_list() == [1, 3]
        assert m["num_features"].to_list() == [1, 3]
