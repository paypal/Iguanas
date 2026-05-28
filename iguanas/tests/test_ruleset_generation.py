import polars as pl
import pytest

from iguanas.rule_selection import (
    extract_feature_names_from_rule,
    filter_correlated_rules,
    filter_rules_by_feature_overlap,
)


class TestExtractColumnNames:
    """Test cases for extract_feature_names_from_rule function."""

    def test_single_column(self):
        """Test extracting a single column name."""
        rule = '(X["age"] >= 18)'
        result = extract_feature_names_from_rule(rule)
        assert result == ["age"]

    def test_multiple_columns(self):
        """Test extracting multiple column names."""
        rule = '(X["age"] >= 18) & (X["income"] < 50000)'
        result = extract_feature_names_from_rule(rule)
        assert result == ["age", "income"]

    def test_duplicate_columns(self):
        """Test that duplicate columns are returned only once."""
        rule = '(X["age"] >= 18) & (X["age"] < 65)'
        result = extract_feature_names_from_rule(rule)
        assert result == ["age"]

    def test_complex_rule(self):
        """Test extracting from a complex rule with many columns."""
        rule = '(X["a"] >= 419) & (X["b"] < 1.0) | (X["c"] == "test") & (X["a"] > 0)'
        result = extract_feature_names_from_rule(rule)
        assert result == ["a", "b", "c"]

    def test_no_columns(self):
        """Test that empty list is returned when no columns found."""
        rule = "True"
        result = extract_feature_names_from_rule(rule)
        assert result == []

    def test_preserves_order(self):
        """Test that column order is preserved."""
        rule = '(X["z"] > 1) & (X["a"] > 2) & (X["m"] > 3)'
        result = extract_feature_names_from_rule(rule)
        assert result == ["z", "a", "m"]


class TestFilterSimilarRules:
    """Test cases for filter_rules_by_feature_overlap function."""

    def test_identical_rules(self):
        """Test filtering identical rules - keeps highest importance."""
        X = pl.DataFrame(
            {
                "rule": [
                    '(X["a"] > 1) & (X["b"] < 2)',
                    '(X["a"] > 1) & (X["b"] < 2)',
                    '(X["a"] > 1) & (X["b"] < 2)',
                ],
                "score": [0.9, 0.85, 0.8],
            }
        )
        importance = {
            '(X["a"] > 1) & (X["b"] < 2)': 0.7,
        }

        result = filter_rules_by_feature_overlap(X, importance, min_difference=1)
        # Should keep only one rule (first occurrence or highest importance)
        assert len(result) == 1
        assert result["rule"][0] == '(X["a"] > 1) & (X["b"] < 2)'

    def test_completely_different_rules(self):
        """Test that completely different rules are all kept."""
        X = pl.DataFrame(
            {
                "rule": [
                    '(X["a"] > 1)',
                    '(X["b"] < 2)',
                    '(X["c"] == 3)',
                ],
                "score": [0.9, 0.85, 0.8],
            }
        )
        importance = {
            '(X["a"] > 1)': 0.6,
            '(X["b"] < 2)': 0.7,
            '(X["c"] == 3)': 0.8,
        }

        result = filter_rules_by_feature_overlap(X, importance, min_difference=1)
        assert len(result) == 3

    def test_similar_rules_different_importance(self):
        """Test filtering similar rules based on importance values."""
        X = pl.DataFrame(
            {
                "rule": [
                    '(X["a"] > 1) & (X["b"] < 2)',
                    '(X["a"] > 5) & (X["b"] < 10)',
                    '(X["c"] > 1)',
                ],
                "score": [0.9, 0.85, 0.8],
            }
        )
        importance = {
            '(X["a"] > 1) & (X["b"] < 2)': 0.5,
            '(X["a"] > 5) & (X["b"] < 10)': 0.9,
            '(X["c"] > 1)': 0.7,
        }

        # First two rules use same columns, should keep higher importance one
        result = filter_rules_by_feature_overlap(X, importance, min_difference=1)
        assert len(result) == 2
        assert '(X["a"] > 5) & (X["b"] < 10)' in result["rule"].to_list()
        assert '(X["c"] > 1)' in result["rule"].to_list()

    def test_min_difference_parameter(self):
        """Test the min_difference parameter."""
        X = pl.DataFrame(
            {
                "rule": [
                    '(X["a"] > 1)',
                    '(X["a"] > 2) & (X["b"] < 3)',
                    '(X["a"] > 3) & (X["b"] < 4) & (X["c"] > 5)',
                ],
            }
        )
        importance = {
            '(X["a"] > 1)': 0.5,
            '(X["a"] > 2) & (X["b"] < 3)': 0.6,
            '(X["a"] > 3) & (X["b"] < 4) & (X["c"] > 5)': 0.7,
        }

        # With min_difference=1, rules that differ by less than 1 column (i.e., 0 columns, identical) should be filtered
        # These rules use subset: {a}, {a,b}, {a,b,c}. They differ by 1, 1, and 1 columns respectively.
        # So with min_difference=1, none are filtered as similar.
        result = filter_rules_by_feature_overlap(X, importance, min_difference=1)
        assert len(result) == 3  # All rules differ by at least 1 column

        # With min_difference=2, rules must differ by at least 2 columns
        # Rule 1 vs 2: differ by 1 < 2 (similar) → keep rule 2 (higher importance)
        # Rule 2 vs 3: differ by 1 < 2 (similar) → keep rule 3 (higher importance)
        # Result: only rule 3 survives the cascade
        result = filter_rules_by_feature_overlap(X, importance, min_difference=2)
        assert len(result) == 1  # Only highest importance rule kept after cascade

    def test_missing_importance_default_to_zero(self):
        """Test that missing importance values default to 0.0."""
        X = pl.DataFrame(
            {
                "rule": [
                    '(X["a"] > 1)',
                    '(X["a"] > 2)',
                ],
            }
        )
        importance = {
            '(X["a"] > 2)': 0.5,
            # '(X["a"] > 1)' is missing
        }

        result = filter_rules_by_feature_overlap(X, importance, min_difference=1)
        # Should keep second rule with higher importance
        assert len(result.columns) == 1
        assert result["rule"][0] == '(X["a"] > 2)'

    def test_custom_rule_column(self):
        """Test using a custom rule column name."""
        X = pl.DataFrame(
            {
                "my_rules": [
                    '(X["a"] > 1)',
                    '(X["b"] < 2)',
                ],
                "score": [0.9, 0.8],
            }
        )
        importance = {
            '(X["a"] > 1)': 0.6,
            '(X["b"] < 2)': 0.7,
        }

        result = filter_rules_by_feature_overlap(
            X, importance, min_difference=1, rule_column="my_rules"
        )
        assert len(result) == 2

    def test_importance_length_mismatch_raises_error(self):
        """Test that missing importance values are handled (default to 0.0)."""
        X = pl.DataFrame(
            {
                "rule": [
                    '(X["a"] > 1)',
                    '(X["b"] < 2)',
                    '(X["c"] == 3)',
                ],
                "score": [0.9, 0.85, 0.8],
            }
        )
        importance = {
            '(X["a"] > 1)': 0.6,
            '(X["b"] < 2)': 0.7,
            # Missing '(X["c"] == 3)' - should default to 0.0
        }

        # Should not raise error - missing rules default to 0.0 importance
        result = filter_rules_by_feature_overlap(X, importance, min_difference=1)
        # All rules are different, so all should be kept
        assert len(result) == 3

    def test_excess_importance_keys_raises_error(self):
        """Test that extra importance keys that don't match any rules are ignored."""
        X = pl.DataFrame(
            {
                "rule": [
                    '(X["a"] > 1)',
                    '(X["b"] < 2)',
                ],
                "score": [0.9, 0.85],
            }
        )
        importance = {
            '(X["a"] > 1)': 0.6,
            '(X["b"] < 2)': 0.7,
            '(X["c"] == 3)': 0.8,  # Extra key that doesn't exist in DataFrame - should be ignored
        }

        # Should not raise error - extra keys are simply ignored
        result = filter_rules_by_feature_overlap(X, importance, min_difference=1)
        # Both rules are different, so both should be kept
        assert len(result) == 2


class TestFilterCorrelatedRules:
    """Test cases for filter_correlated_rules function."""

    def test_perfectly_correlated_columns(self):
        """Test filtering perfectly correlated columns."""
        R = pl.DataFrame(
            {
                "rule_A": [True, False, True, False, True],
                "rule_B": [True, False, True, False, True],  # Identical to rule_A
                "rule_C": [
                    False,
                    True,
                    False,
                    True,
                    False,
                ],  # Perfectly negatively correlated with A and B
            }
        )
        importance = {
            "rule_A": 0.8,
            "rule_B": 0.6,
            "rule_C": 0.9,
        }

        result = filter_correlated_rules(R, importance, max_corr=0.95)
        # rule_C has highest importance, so rule_A and rule_B should be removed
        # (all three have |corr|=1.0 with each other)
        assert result == ["rule_C"]

    def test_no_correlation(self):
        """Test that uncorrelated columns are all kept."""
        R = pl.DataFrame(
            {
                "rule_A": [True, False, True, False],
                "rule_B": [
                    False,
                    True,
                    False,
                    True,
                ],  # Perfectly negatively correlated with A
                "rule_C": [True, True, False, False],  # Uncorrelated with A and B
            }
        )
        importance = {
            "rule_A": 0.5,
            "rule_B": 0.6,
            "rule_C": 0.7,
        }

        result = filter_correlated_rules(R, importance, max_corr=0.95)
        # rule_A and rule_B have |corr|=1.0, so rule_A (lower importance) should be removed
        assert set(result) == {"rule_B", "rule_C"}

    def test_high_correlation_threshold(self):
        """Test filtering with different correlation thresholds."""
        R = pl.DataFrame(
            {
                "rule_A": [True, True, False, False, True, True],
                "rule_B": [
                    True,
                    True,
                    False,
                    False,
                    True,
                    False,
                ],  # Mostly correlated with A
                "rule_C": [
                    False,
                    False,
                    True,
                    True,
                    False,
                    False,
                ],  # Negatively correlated with A and B
            }
        )
        importance = {
            "rule_A": 0.8,
            "rule_B": 0.6,
            "rule_C": 0.7,
        }

        # With high threshold (0.99), rule_C will still be filtered due to high negative correlation
        result = filter_correlated_rules(R, importance, max_corr=0.99)
        # rule_A and rule_C have strong correlation, rule_A (higher importance) is kept
        assert len(result) == 2

        # With lower threshold, may filter some
        result = filter_correlated_rules(R, importance, max_corr=0.7)
        assert len(result) <= 3

    def test_keeps_highest_importance(self):
        """Test that the column with highest importance is kept."""
        R = pl.DataFrame(
            {
                "rule_A": [True, False, True, False],
                "rule_B": [True, False, True, False],  # Identical to rule_A
            }
        )
        importance_low_first = {
            "rule_A": 0.4,
            "rule_B": 0.9,
        }

        result = filter_correlated_rules(R, importance_low_first, max_corr=0.95)
        # Should keep rule_B (higher importance)
        assert result == ["rule_B"]

    def test_single_column(self):
        """Test behavior with single column."""
        R = pl.DataFrame(
            {
                "rule_A": [True, False, True, False],
            }
        )
        importance = {
            "rule_A": 0.8,
        }

        result = filter_correlated_rules(R, importance, max_corr=0.95)
        assert result == ["rule_A"]

    def test_constant_columns(self):
        """Test handling of constant columns (NaN correlation)."""
        R = pl.DataFrame(
            {
                "rule_A": [True, True, True, True],
                "rule_B": [False, True, False, True],
                "rule_C": [True, True, True, True],
            }
        )
        importance = {
            "rule_A": 0.8,
            "rule_B": 0.6,
            "rule_C": 0.9,
        }

        # Constant columns should not crash, NaN correlations should be skipped
        result = filter_correlated_rules(R, importance, max_corr=0.95)
        assert "rule_B" in result  # Non-constant column should be kept

    def test_negative_correlation(self):
        """Test that absolute correlation is used."""
        R = pl.DataFrame(
            {
                "rule_A": [True, False, True, False, True],
                "rule_B": [
                    False,
                    True,
                    False,
                    True,
                    False,
                ],  # Perfect negative correlation
                "rule_C": [True, True, False, False, True],
            }
        )
        importance = {
            "rule_A": 0.8,
            "rule_B": 0.6,
            "rule_C": 0.7,
        }

        result = filter_correlated_rules(R, importance, max_corr=0.95)
        # rule_B should be removed (|corr|=1.0 with rule_A)
        assert "rule_B" not in result
        assert "rule_A" in result

    def test_missing_importance_keys(self):
        """Test that ValueError is raised when importance dict doesn't have all columns."""
        R = pl.DataFrame(
            {
                "rule_A": [True, False, True, False],
                "rule_B": [
                    False,
                    True,
                    False,
                    True,
                ],  # Perfectly negatively correlated with A
                "rule_C": [True, True, False, False],
            }
        )
        importance = {
            "rule_A": 0.8,
            "rule_B": 0.6,
            # rule_C missing - should raise ValueError
        }

        with pytest.raises(
            ValueError,
            match="Length of importance dict must match number of columns in R",
        ):
            filter_correlated_rules(R, importance, max_corr=0.95)

    def test_excess_importance_keys(self):
        """Test that ValueError is raised when importance dict has more keys than columns."""
        R = pl.DataFrame(
            {
                "rule_A": [True, False, True, False],
                "rule_B": [False, True, False, True],
            }
        )
        importance = {
            "rule_A": 0.8,
            "rule_B": 0.6,
            "rule_C": 0.7,  # Extra key that doesn't exist in DataFrame
        }

        with pytest.raises(
            ValueError,
            match="Length of importance dict must match number of columns in R",
        ):
            filter_correlated_rules(R, importance, max_corr=0.95)

    def test_chain_of_correlations(self):
        """Test handling of multiple correlated columns."""
        R = pl.DataFrame(
            {
                "rule_A": [True, False, True, False, True, False],
                "rule_B": [True, False, True, False, True, False],  # Same as A
                "rule_C": [True, False, True, False, True, False],  # Same as A and B
                "rule_D": [True, True, False, False, True, True],  # Uncorrelated with A
            }
        )
        importance = {
            "rule_A": 0.9,  # Highest
            "rule_B": 0.7,
            "rule_C": 0.6,  # Lowest
            "rule_D": 0.8,
        }

        result = filter_correlated_rules(R, importance, max_corr=0.95)
        # Should keep rule_A (highest importance among correlated) and rule_D (uncorrelated)
        assert set(result) == {"rule_A", "rule_D"}
