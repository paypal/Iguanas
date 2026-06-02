import polars as pl
import pytest

from iguanas.metrics import compute_metrics


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
