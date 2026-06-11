import pytest
import polars as pl

from iguanas.rule_selection import (
    extract_feature_names_from_rule,
    filter_rules_by_feature_overlap,
    filter_correlated_rules,
    select_best_rule_per_column_combination,
)


class TestExtractColumnNames:
    """Test cases for extract_feature_names_from_rule function."""

    def test_single_column(self):
        """Test extraction from rule with single column."""
        rule = '(X["amount"] > 100)'
        result = extract_feature_names_from_rule(rule)
        assert result == ["amount"]

    def test_multiple_columns(self):
        """Test extraction from rule with multiple columns."""
        rule = '(X["a"] >= 419) & (X["b"] < 1.0)'
        result = extract_feature_names_from_rule(rule)
        assert result == ["a", "b"]

    def test_duplicate_columns(self):
        """Test that duplicates are removed but order is preserved."""
        rule = '(X["a"] > 1) & (X["b"] < 2) & (X["a"] < 10)'
        result = extract_feature_names_from_rule(rule)
        assert result == ["a", "b"]

    def test_no_columns(self):
        """Test rule with no column references."""
        rule = "True"
        result = extract_feature_names_from_rule(rule)
        assert result == []

    def test_complex_column_names(self):
        """Test extraction with complex column names."""
        rule = '(X["column_name_123"] > 5) & (X["another-column"] < 10)'
        result = extract_feature_names_from_rule(rule)
        assert result == ["column_name_123", "another-column"]


class TestFilterSimilarRules:
    """Test cases for filter_rules_by_feature_overlap function."""

    def test_single_rule_returns_unchanged(self):
        """Test that single rule is returned unchanged (line 90 coverage)."""
        R = pl.DataFrame({"rule": ['(X["a"] > 1)'], "score": [0.9]})
        importance = {'(X["a"] > 1)': 0.9}

        result = filter_rules_by_feature_overlap(R, importance, min_difference=1)

        assert result.equals(R)

    def test_empty_rules_returns_unchanged(self):
        """Test that empty DataFrame is returned unchanged (line 90 coverage)."""
        R = pl.DataFrame({"rule": [], "score": []})
        importance = {}

        result = filter_rules_by_feature_overlap(R, importance, min_difference=1)

        assert result.equals(R)

    def test_filter_identical_rules(self):
        """Test filtering of rules with identical columns."""
        R = pl.DataFrame(
            {
                "rule": [
                    '(X["a"] > 1) & (X["b"] < 2)',
                    '(X["a"] > 5) & (X["b"] < 10)',  # Same columns, different thresholds
                ],
                "score": [0.9, 0.85],
            }
        )
        importance = {'(X["a"] > 1) & (X["b"] < 2)': 0.7, '(X["a"] > 5) & (X["b"] < 10)': 0.9}

        result = filter_rules_by_feature_overlap(R, importance, min_difference=1)

        # Should keep only the rule with higher importance
        assert len(result) == 1
        assert result["rule"][0] == '(X["a"] > 5) & (X["b"] < 10)'

    def test_keep_dissimilar_rules(self):
        """Test that dissimilar rules are all kept."""
        R = pl.DataFrame(
            {
                "rule": [
                    '(X["a"] > 1)',
                    '(X["b"] < 2)',
                    '(X["c"] == 3)',
                ],
                "score": [0.9, 0.85, 0.8],
            }
        )
        importance = {'(X["a"] > 1)': 0.9, '(X["b"] < 2)': 0.85, '(X["c"] == 3)': 0.8}

        result = filter_rules_by_feature_overlap(R, importance, min_difference=1)

        # Should keep all rules as they use different columns
        assert len(result) == 3

    def test_min_difference_threshold(self):
        """Test min_difference parameter."""
        R = pl.DataFrame(
            {
                "rule": [
                    '(X["a"] > 1) & (X["b"] < 2)',
                    '(X["a"] > 1) & (X["c"] < 3)',  # Differs by 1 column (max_one_sided_diff=1)
                ],
                "score": [0.9, 0.85],
            }
        )
        importance = {'(X["a"] > 1) & (X["b"] < 2)': 0.9, '(X["a"] > 1) & (X["c"] < 3)': 0.85}

        # With min_difference=1: max_one_sided_diff (1) < 1? NO → NOT similar → both kept
        result_min1 = filter_rules_by_feature_overlap(R, importance, min_difference=1)
        assert len(result_min1) == 2

        # With min_difference=2: max_one_sided_diff (1) < 2? YES → similar → one filtered
        result_min2 = filter_rules_by_feature_overlap(R, importance, min_difference=2)
        assert len(result_min2) == 1
        assert result_min2["rule"][0] == '(X["a"] > 1) & (X["b"] < 2)'  # Higher importance

    def test_custom_rule_column(self):
        """Test with custom rule_column parameter."""
        R = pl.DataFrame({"custom_rule": ['(X["a"] > 1)', '(X["b"] < 2)'], "score": [0.9, 0.85]})
        importance = {'(X["a"] > 1)': 0.9, '(X["b"] < 2)': 0.85}

        result = filter_rules_by_feature_overlap(
            R, importance, min_difference=1, rule_column="custom_rule"
        )

        assert len(result) == 2
        assert "custom_rule" in result.columns

    def test_missing_importance_values(self):
        """Test handling of missing importance values (defaults to 0.0)."""
        R = pl.DataFrame({"rule": ['(X["a"] > 1)', '(X["a"] < 10)'], "score": [0.9, 0.85]})
        # Only provide importance for one rule
        importance = {'(X["a"] > 1)': 0.9}

        result = filter_rules_by_feature_overlap(R, importance, min_difference=1)

        # Should keep the rule with defined importance
        assert len(result) == 1
        assert result["rule"][0] == '(X["a"] > 1)'

    def test_rules_with_no_columns(self):
        """Test handling rules that extract no column names."""
        R = pl.DataFrame({"rule": ["True", "False", "1 == 1"], "score": [0.9, 0.85, 0.8]})
        importance = {"True": 0.9, "False": 0.5, "1 == 1": 0.7}

        result = filter_rules_by_feature_overlap(R, importance, min_difference=1)

        # Rules with no columns are all similar (max_one_sided_diff = 0)
        # Should keep only the highest importance
        assert len(result) == 1
        assert result["rule"][0] == "True"

    def test_transitive_similarity(self):
        """Test transitive similarity scenario (A~B, B~C, but A not similar to C)."""
        R = pl.DataFrame(
            {
                "rule": [
                    '(X["a"] > 1) & (X["b"] < 2)',  # {a, b}
                    '(X["b"] > 1) & (X["c"] < 2)',  # {b, c} - differs from first by 1
                    '(X["c"] > 1) & (X["d"] < 2)',  # {c, d} - differs from second by 1
                ],
                "score": [0.9, 0.85, 0.8],
            }
        )
        importance = {
            '(X["a"] > 1) & (X["b"] < 2)': 0.9,
            '(X["b"] > 1) & (X["c"] < 2)': 0.5,
            '(X["c"] > 1) & (X["d"] < 2)': 0.8,
        }

        result = filter_rules_by_feature_overlap(R, importance, min_difference=2)

        # First and second are similar (differ by 1 < 2), keep first (higher importance)
        # Third is similar to second but second is already filtered, so third is kept
        # This is transitive similarity - greedy algorithm keeps first and third
        assert len(result) == 2
        assert '(X["a"] > 1) & (X["b"] < 2)' in result["rule"].to_list()
        assert '(X["c"] > 1) & (X["d"] < 2)' in result["rule"].to_list()

    def test_identical_columns_always_similar(self):
        """Test that rules with identical columns are always similar regardless of min_difference."""
        R = pl.DataFrame(
            {
                "rule": ['(X["a"] > 1) & (X["b"] < 2)', '(X["a"] > 5) & (X["b"] < 10)'],
                "score": [0.9, 0.85],
            }
        )
        importance = {'(X["a"] > 1) & (X["b"] < 2)': 0.7, '(X["a"] > 5) & (X["b"] < 10)': 0.9}

        # Even with large min_difference, identical column sets are similar
        result = filter_rules_by_feature_overlap(R, importance, min_difference=100)

        assert len(result) == 1
        assert result["rule"][0] == '(X["a"] > 5) & (X["b"] < 10)'

    def test_order_independence_for_importance(self):
        """Test that result is independent of input order (always keeps highest importance)."""
        R1 = pl.DataFrame(
            {
                "rule": ['(X["a"] > 1) & (X["b"] < 2)', '(X["a"] > 5) & (X["b"] < 10)'],
                "score": [0.9, 0.85],
            }
        )
        R2 = pl.DataFrame(
            {
                "rule": ['(X["a"] > 5) & (X["b"] < 10)', '(X["a"] > 1) & (X["b"] < 2)'],
                "score": [0.85, 0.9],
            }
        )
        importance = {'(X["a"] > 1) & (X["b"] < 2)': 0.9, '(X["a"] > 5) & (X["b"] < 10)': 0.5}

        result1 = filter_rules_by_feature_overlap(R1, importance, min_difference=1)
        result2 = filter_rules_by_feature_overlap(R2, importance, min_difference=1)

        # Both should keep the rule with higher importance
        assert len(result1) == 1
        assert len(result2) == 1
        assert result1["rule"][0] == result2["rule"][0]
        assert result1["rule"][0] == '(X["a"] > 1) & (X["b"] < 2)'


class TestFilterCorrelatedRules:
    """Test cases for filter_correlated_rules function."""

    def test_single_column_returns_unchanged(self):
        """Test that single column is returned unchanged."""
        C = pl.DataFrame({"rule_A": [1.0]})
        importance = {"rule_A": 0.9}

        result = filter_correlated_rules(C, importance, max_corr=0.95)

        assert result == ["rule_A"]

    def test_no_correlation_keeps_all(self):
        """Test that uncorrelated features are all kept."""
        C = pl.DataFrame(
            {
                "rule_A": [1.0, 0.1, 0.05],
                "rule_B": [0.1, 1.0, 0.15],
                "rule_C": [0.05, 0.15, 1.0],
            }
        )
        importance = {"rule_A": 0.8, "rule_B": 0.6, "rule_C": 0.9}

        result = filter_correlated_rules(C, importance, max_corr=0.95)

        assert set(result) == {"rule_A", "rule_B", "rule_C"}
        assert len(result) == 3

    def test_high_correlation_filters_less_important(self):
        """Test that highly correlated pairs keep only the most important."""
        C = pl.DataFrame(
            {
                "rule_A": [1.0, 0.98, 0.1],
                "rule_B": [0.98, 1.0, 0.2],
                "rule_C": [0.1, 0.2, 1.0],
            }
        )
        importance = {"rule_A": 0.8, "rule_B": 0.6, "rule_C": 0.9}

        result = filter_correlated_rules(C, importance, max_corr=0.95)

        # rule_A and rule_B are correlated (0.98 > 0.95), keep rule_A (higher importance)
        # rule_C is uncorrelated, keep it
        assert set(result) == {"rule_A", "rule_C"}
        assert len(result) == 2

    def test_negative_correlation_above_threshold(self):
        """Test that absolute correlation is used (negative correlations)."""
        C = pl.DataFrame(
            {
                "rule_A": [1.0, -0.97, 0.1],
                "rule_B": [-0.97, 1.0, 0.2],
                "rule_C": [0.1, 0.2, 1.0],
            }
        )
        importance = {"rule_A": 0.8, "rule_B": 0.6, "rule_C": 0.9}

        result = filter_correlated_rules(C, importance, max_corr=0.95)

        # rule_A and rule_B are correlated (|-0.97| > 0.95), keep rule_A
        assert set(result) == {"rule_A", "rule_C"}

    def test_max_corr_threshold(self):
        """Test max_corr parameter controls filtering."""
        C = pl.DataFrame(
            {
                "rule_A": [1.0, 0.92, 0.1],
                "rule_B": [0.92, 1.0, 0.2],
                "rule_C": [0.1, 0.2, 1.0],
            }
        )
        importance = {"rule_A": 0.8, "rule_B": 0.6, "rule_C": 0.9}

        # With max_corr=0.95, 0.92 is not above threshold, keep all
        result_095 = filter_correlated_rules(C, importance, max_corr=0.95)
        assert len(result_095) == 3

        # With max_corr=0.90, 0.92 is above threshold, filter one
        result_090 = filter_correlated_rules(C, importance, max_corr=0.90)
        assert set(result_090) == {"rule_A", "rule_C"}

    def test_importance_dict_mismatch_raises_error(self):
        """Test that ValueError is raised when importance dict length doesn't match."""
        C = pl.DataFrame(
            {
                "rule_A": [1.0, 0.5],
                "rule_B": [0.5, 1.0],
            }
        )
        # Only one importance value provided
        importance = {"rule_A": 0.8}

        with pytest.raises(ValueError, match="Length of importance dict must match"):
            filter_correlated_rules(C, importance, max_corr=0.95)

    def test_nan_correlation_is_ignored(self):
        """Test that NaN correlations are handled gracefully."""
        C = pl.DataFrame(
            {
                "rule_A": [1.0, float("nan"), 0.1],
                "rule_B": [float("nan"), 1.0, 0.2],
                "rule_C": [0.1, 0.2, 1.0],
            }
        )
        importance = {"rule_A": 0.8, "rule_B": 0.6, "rule_C": 0.9}

        result = filter_correlated_rules(C, importance, max_corr=0.95)

        # NaN correlation is ignored, all columns kept
        assert len(result) == 3

    def test_multiple_correlations_greedy_selection(self):
        """Test greedy algorithm with multiple correlated pairs."""
        C = pl.DataFrame(
            {
                "rule_A": [1.0, 0.97, 0.1, 0.05],
                "rule_B": [0.97, 1.0, 0.96, 0.1],
                "rule_C": [0.1, 0.96, 1.0, 0.15],
                "rule_D": [0.05, 0.1, 0.15, 1.0],
            }
        )
        importance = {"rule_A": 0.9, "rule_B": 0.5, "rule_C": 0.7, "rule_D": 0.8}

        result = filter_correlated_rules(C, importance, max_corr=0.95)

        # rule_A and rule_B correlated (0.97 > 0.95) -> keep rule_A (0.9 > 0.5)
        # rule_B and rule_C correlated (0.96 > 0.95) -> rule_B already removed, so keep rule_C
        # rule_D uncorrelated -> keep
        assert set(result) == {"rule_A", "rule_C", "rule_D"}

    def test_equal_importance_keeps_first(self):
        """Test that when importance is equal, first column is kept."""
        C = pl.DataFrame(
            {
                "rule_A": [1.0, 0.98],
                "rule_B": [0.98, 1.0],
            }
        )
        importance = {"rule_A": 0.8, "rule_B": 0.8}

        result = filter_correlated_rules(C, importance, max_corr=0.95)

        # Equal importance, keep first (rule_A)
        assert result == ["rule_A"]

    def test_preserves_column_order(self):
        """Test that returned column order matches input order."""
        C = pl.DataFrame(
            {
                "rule_Z": [1.0, 0.1, 0.05],
                "rule_A": [0.1, 1.0, 0.15],
                "rule_M": [0.05, 0.15, 1.0],
            }
        )
        importance = {"rule_Z": 0.8, "rule_A": 0.6, "rule_M": 0.9}

        result = filter_correlated_rules(C, importance, max_corr=0.95)

        # Order should be preserved from input
        assert result == ["rule_Z", "rule_A", "rule_M"]

    def test_chain_correlation_removal(self):
        """Test handling of chain correlations (A~B, B removed, C~B)."""
        C = pl.DataFrame(
            {
                "rule_A": [1.0, 0.97, 0.97],
                "rule_B": [0.97, 1.0, 0.97],
                "rule_C": [0.97, 0.97, 1.0],
            }
        )
        importance = {"rule_A": 0.9, "rule_B": 0.5, "rule_C": 0.8}

        result = filter_correlated_rules(C, importance, max_corr=0.95)

        # rule_A and rule_B correlated -> remove rule_B
        # rule_A and rule_C correlated -> remove rule_C (lower importance)
        assert result == ["rule_A"]

    # --- use_abs parameter tests ---

    def test_use_abs_true_filters_negative_correlation(self):
        """use_abs=True (default): strong negative correlation triggers filtering."""
        C = pl.DataFrame(
            {
                "rule_A": [1.0, -0.97, 0.1],
                "rule_B": [-0.97, 1.0, 0.2],
                "rule_C": [0.1, 0.2, 1.0],
            }
        )
        importance = {"rule_A": 0.8, "rule_B": 0.6, "rule_C": 0.9}

        result = filter_correlated_rules(C, importance, max_corr=0.95, use_abs=True)

        # |-0.97| = 0.97 > 0.95 → filter rule_B (lower importance)
        assert set(result) == {"rule_A", "rule_C"}

    def test_use_abs_false_ignores_negative_correlation(self):
        """use_abs=False: strong negative correlation does NOT trigger filtering."""
        C = pl.DataFrame(
            {
                "rule_A": [1.0, -0.97, 0.1],
                "rule_B": [-0.97, 1.0, 0.2],
                "rule_C": [0.1, 0.2, 1.0],
            }
        )
        importance = {"rule_A": 0.8, "rule_B": 0.6, "rule_C": 0.9}

        result = filter_correlated_rules(C, importance, max_corr=0.95, use_abs=False)

        # -0.97 is NOT > 0.95 (raw value is negative) → all rules kept
        assert set(result) == {"rule_A", "rule_B", "rule_C"}

    def test_use_abs_false_still_filters_positive_correlation(self):
        """use_abs=False: strong positive correlation still triggers filtering."""
        C = pl.DataFrame(
            {
                "rule_A": [1.0, 0.97, 0.1],
                "rule_B": [0.97, 1.0, 0.2],
                "rule_C": [0.1, 0.2, 1.0],
            }
        )
        importance = {"rule_A": 0.8, "rule_B": 0.6, "rule_C": 0.9}

        result = filter_correlated_rules(C, importance, max_corr=0.95, use_abs=False)

        # 0.97 > 0.95 → filter rule_B (lower importance)
        assert set(result) == {"rule_A", "rule_C"}

    def test_use_abs_default_is_true(self):
        """Verify default use_abs=True behaviour matches explicit use_abs=True."""
        C = pl.DataFrame(
            {
                "rule_A": [1.0, -0.97],
                "rule_B": [-0.97, 1.0],
            }
        )
        importance = {"rule_A": 0.8, "rule_B": 0.6}

        result_default = filter_correlated_rules(C, importance, max_corr=0.95)
        result_explicit = filter_correlated_rules(C, importance, max_corr=0.95, use_abs=True)

        assert result_default == result_explicit == ["rule_A"]

    def test_use_abs_false_boundary_exactly_at_threshold(self):
        """use_abs=False: correlation exactly at max_corr is NOT filtered (strict >)."""
        C = pl.DataFrame(
            {
                "rule_A": [1.0, 0.95],
                "rule_B": [0.95, 1.0],
            }
        )
        importance = {"rule_A": 0.8, "rule_B": 0.6}

        result = filter_correlated_rules(C, importance, max_corr=0.95, use_abs=False)

        # 0.95 is NOT > 0.95, so no filtering
        assert set(result) == {"rule_A", "rule_B"}


class TestSelectBestRulePerColumnCombination:
    def test_basic_selection(self):
        """Best rule per unique column combination is returned (lines 273-299)."""
        metrics = pl.DataFrame({
            "rule": ['(X["a"] > 1)', '(X["a"] > 2)', '(X["b"] < 3)'],
            "precision": [0.95, 0.98, 0.96],
        })
        result = select_best_rule_per_column_combination(metrics, ranking_metric="precision")
        assert '(X["a"] > 2)' in result
        assert '(X["b"] < 3)' in result
        assert len(result) == 2

    def test_missing_rule_column_raises(self):
        """ValueError when 'rule' column is absent (line 268-269)."""
        metrics = pl.DataFrame({"not_rule": ['(X["a"] > 1)'], "precision": [0.9]})
        with pytest.raises(ValueError, match="must contain a 'rule' column"):
            select_best_rule_per_column_combination(metrics)

    def test_missing_ranking_metric_raises(self):
        """ValueError when ranking_metric column is absent (line 270-271)."""
        metrics = pl.DataFrame({"rule": ['(X["a"] > 1)'], "precision": [0.9]})
        with pytest.raises(ValueError, match="not found in metrics columns"):
            select_best_rule_per_column_combination(metrics, ranking_metric="recall")


class TestFilterCorrelatedRulesColIRemoval:
    def test_col_i_removed_when_less_important(self):
        """When col_j is more important, col_i is removed and inner loop breaks (lines 234-235)."""
        C = pl.DataFrame({
            "rule_A": [1.0, 0.98, 0.1],
            "rule_B": [0.98, 1.0, 0.2],
            "rule_C": [0.1, 0.2, 1.0],
        })
        # rule_B > rule_A in importance, so when (A, B) pair is examined,
        # col_i=A is removed and the break on line 235 is triggered
        importance = {"rule_A": 0.3, "rule_B": 0.9, "rule_C": 0.7}
        result = filter_correlated_rules(C, importance, max_corr=0.95)
        assert "rule_A" not in result
        assert "rule_B" in result
        assert "rule_C" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
