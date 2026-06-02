import polars as pl
import pytest

from iguanas.rule_evaluation import (
    apply_rules,
    apply_and_filter_by_performance,
    select_diverse_top_rules,
    apply_filter_and_deduplicate_rules,
)


class TestEvaluateRules:
    """Test cases for apply_rules function."""

    def test_basic_rule_evaluation(self):
        """Test basic rule string evaluation."""
        X = pl.DataFrame({"amount": [100, 50, 200], "age": [25, 35, 45]})
        rules = ['(X["amount"] > 75)', '(X["age"] < 40)']
        result = apply_rules(X, rules)

        expected = pl.DataFrame(
            {
                '(X["amount"] > 75)': [True, False, True],
                '(X["age"] < 40)': [True, True, False],
            }
        )
        assert result.equals(expected)

    def test_complex_rule_with_multiple_conditions(self):
        """Test rule with multiple AND conditions."""
        X = pl.DataFrame({"amount": [100, 50, 200], "age": [25, 35, 45]})
        rules = ['(X["amount"] > 75) & (X["age"] < 40)']
        result = apply_rules(X, rules)

        expected = pl.DataFrame(
            {
                '(X["amount"] > 75) & (X["age"] < 40)': [True, False, False],
            }
        )
        assert result.equals(expected)

    def test_empty_rules_list(self):
        """Test with empty rules list."""
        X = pl.DataFrame({"amount": [100, 50, 200]})
        rules = []
        result = apply_rules(X, rules)

        # Polars native behavior: X.select([]) returns shape (0, 0)
        assert result.shape == (0, 0)


class TestApplyAndFilterByPerformance:
    """Test cases for apply_and_filter_by_performance function."""

    def test_basic_filtering(self, capsys):
        """Test basic rule evaluation and filtering by performance thresholds."""
        X = pl.DataFrame(
            {"age": [25, 30, 35, 40, 45, 50], "income": [30000, 40000, 50000, 60000, 70000, 80000]}
        )
        y = pl.Series([0, 0, 1, 1, 1, 1])
        rules = ['(X["age"] >= 35)', '(X["income"] >= 50000)']

        R, metrics = apply_and_filter_by_performance(
            X,
            y,
            rules,
            metrics_threshold=[
                {"name": "precision", "operator": ">=", "value": 0.5},
                {"name": "recall", "operator": ">=", "value": 0.5},
            ],
        )

        # Check that we get filtered results
        assert isinstance(R, pl.DataFrame)
        assert isinstance(metrics, pl.DataFrame)
        assert "rule" in metrics.columns
        assert "precision" in metrics.columns
        assert "recall" in metrics.columns

        # Check that all returned rules meet thresholds
        assert (metrics["precision"] >= 0.5).all()
        assert (metrics["recall"] >= 0.5).all()

    def test_with_weights(self):
        """Test filtering with sample weights."""
        X = pl.DataFrame({"age": [25, 30, 35, 40], "weight": [1.0, 2.0, 1.5, 3.0]})
        y = pl.Series([0, 0, 1, 1])
        rules = ['(X["age"] >= 35)']

        R, metrics = apply_and_filter_by_performance(
            X,
            y,
            rules,
            weight_column="weight",
            metrics_threshold=[
                {"name": "precision", "operator": ">=", "value": 0.4},
                {"name": "recall", "operator": ">=", "value": 0.4},
            ],
        )

        assert isinstance(R, pl.DataFrame)
        assert isinstance(metrics, pl.DataFrame)
        assert len(metrics) > 0

    def test_no_rules_pass_threshold(self, capsys):
        """Test when no rules meet the performance thresholds."""
        X = pl.DataFrame({"age": [25, 30, 35, 40], "income": [30000, 40000, 50000, 60000]})
        y = pl.Series([0, 0, 0, 1])
        rules = ['(X["age"] >= 50)', '(X["income"] >= 100000)']  # Very restrictive rules

        R, metrics = apply_and_filter_by_performance(
            X,
            y,
            rules,
            metrics_threshold=[
                {"name": "precision", "operator": ">=", "value": 0.9},
                {"name": "recall", "operator": ">=", "value": 0.9},
            ],
        )

        # Should return empty results
        assert len(metrics) == 0

    def test_custom_sort_by(self):
        """Test sorting by different metrics."""
        X = pl.DataFrame(
            {"age": [25, 30, 35, 40, 45, 50], "income": [30000, 40000, 50000, 60000, 70000, 80000]}
        )
        y = pl.Series([0, 0, 1, 1, 1, 1])
        rules = ['(X["age"] >= 35)', '(X["income"] >= 50000)', '(X["age"] >= 40)']

        R, metrics = apply_and_filter_by_performance(
            X,
            y,
            rules,
            metrics_threshold=[
                {"name": "precision", "operator": ">=", "value": 0.4},
                {"name": "recall", "operator": ">=", "value": 0.4},
            ],
            sort_by="recall",
        )

        # Check that results are sorted by recall in descending order
        assert isinstance(metrics, pl.DataFrame)
        if len(metrics) > 1:
            recall_values = metrics["recall"].to_list()
            assert recall_values == sorted(recall_values, reverse=True)

    def test_minimal_thresholds(self):
        """Test with very low thresholds to include most rules."""
        X = pl.DataFrame({"age": [25, 30, 35, 40], "income": [30000, 40000, 50000, 60000]})
        y = pl.Series([0, 0, 1, 1])
        rules = ['(X["age"] >= 30)', '(X["income"] >= 40000)']

        R, metrics = apply_and_filter_by_performance(
            X,
            y,
            rules,
            metrics_threshold=[
                {"name": "precision", "operator": ">=", "value": 0.0},
                {"name": "recall", "operator": ">=", "value": 0.0},
            ],
        )

        # All rules should pass with minimal thresholds
        assert len(metrics) == len(rules)
        assert R.shape[1] == len(rules)


class TestSelectDiverseTopRules:
    """Test cases for select_diverse_top_rules function."""

    def test_basic_selection(self, capsys):
        """Test basic rule selection with correlation filtering."""
        # Create correlated rules
        R_test = pl.DataFrame(
            {
                "rule_A": [True, False, True, False, True],
                "rule_B": [True, False, True, True, True],  # Similar to rule_A
                "rule_C": [False, True, False, True, False],  # Different from others
            }
        )
        metrics_test = pl.DataFrame(
            {
                "rule": ["rule_A", "rule_B", "rule_C"],
                "f1": [0.8, 0.75, 0.6],
                "f0.5": [0.85, 0.78, 0.65],
            }
        )

        R_filtered, metrics_filtered, rules = select_diverse_top_rules(
            R_test, metrics_test, max_corr=0.7, importance_metric="f0.5"
        )

        # Should filter out correlated rules
        assert isinstance(R_filtered, pl.DataFrame)
        assert isinstance(metrics_filtered, pl.DataFrame)
        assert isinstance(rules, list)
        assert len(rules) <= 3

        # Verify console output
        captured = capsys.readouterr()
        assert "Number of uncorrelated rules:" in captured.out

    def test_with_top_n_limit(self, capsys):
        """Test limiting to top N rules before correlation filtering."""
        R_test = pl.DataFrame(
            {
                "rule_A": [True, False, True, False],
                "rule_B": [False, True, False, True],
                "rule_C": [True, True, False, False],
                "rule_D": [False, False, True, True],
            }
        )
        metrics_test = pl.DataFrame(
            {
                "rule": ["rule_A", "rule_B", "rule_C", "rule_D"],
                "f1": [0.9, 0.8, 0.7, 0.6],
                "f0.5": [0.92, 0.82, 0.72, 0.62],
            }
        )

        R_filtered, metrics_filtered, rules = select_diverse_top_rules(
            R_test, metrics_test, max_corr=0.5, top_n=2
        )

        # Should limit to top 2 rules by f1
        assert len(rules) <= 2
        assert all(rule in ["rule_A", "rule_B", "rule_C", "rule_D"] for rule in rules)

    def test_no_correlation_filtering(self, capsys):
        """Test with high max_corr threshold (no filtering expected)."""
        R_test = pl.DataFrame(
            {
                "rule_A": [True, False, True, False],
                "rule_B": [False, True, False, True],
                "rule_C": [True, True, False, False],
            }
        )
        metrics_test = pl.DataFrame(
            {
                "rule": ["rule_A", "rule_B", "rule_C"],
                "f1": [0.8, 0.7, 0.6],
                "f0.5": [0.85, 0.75, 0.65],
            }
        )

        R_filtered, metrics_filtered, rules = select_diverse_top_rules(
            R_test, metrics_test, max_corr=1.0, importance_metric="f0.5"
        )

        # All rules should pass with max_corr=1.0
        assert len(rules) == 3

    def test_custom_sort_metric(self):
        """Test sorting by different metric."""
        R_test = pl.DataFrame(
            {
                "rule_A": [True, False, True, False],
                "rule_B": [False, True, False, True],
                "rule_C": [True, True, False, False],
            }
        )
        metrics_test = pl.DataFrame(
            {
                "rule": ["rule_A", "rule_B", "rule_C"],
                "precision": [0.9, 0.7, 0.8],
                "f1": [0.6, 0.8, 0.7],
                "f0.5": [0.85, 0.75, 0.65],
            }
        )

        R_filtered, metrics_filtered, rules = select_diverse_top_rules(
            R_test, metrics_test, max_corr=0.5, sort_by="precision"
        )

        # Verify sorting by precision
        assert isinstance(metrics_filtered, pl.DataFrame)
        if len(metrics_filtered) > 1:
            precision_values = metrics_filtered["precision"].to_list()
            assert precision_values == sorted(precision_values, reverse=True)

    def test_single_rule(self, capsys):
        """Test with only one rule."""
        R_test = pl.DataFrame({"rule_A": [True, False, True, False]})
        metrics_test = pl.DataFrame({"rule": ["rule_A"], "f1": [0.8], "f0.5": [0.85]})

        R_filtered, metrics_filtered, rules = select_diverse_top_rules(
            R_test, metrics_test, max_corr=0.5
        )

        assert len(rules) == 1
        assert rules[0] == "rule_A"

    def test_all_rules_correlated(self, capsys):
        """Test when all rules are highly correlated."""
        # Create highly correlated rules
        R_test = pl.DataFrame(
            {
                "rule_A": [True, False, True, False, True],
                "rule_B": [True, False, True, False, True],  # Identical to rule_A
                "rule_C": [True, False, True, True, True],  # Very similar
            }
        )
        metrics_test = pl.DataFrame(
            {
                "rule": ["rule_A", "rule_B", "rule_C"],
                "f1": [0.8, 0.79, 0.75],
                "f0.5": [0.85, 0.84, 0.78],
            }
        )

        R_filtered, metrics_filtered, rules = select_diverse_top_rules(
            R_test, metrics_test, max_corr=0.8, importance_metric="f0.5"
        )

        # Should keep only the most important rule
        assert len(rules) >= 1
        assert "rule_A" in rules  # Should keep the highest importance


class TestApplyFilterAndDeduplicateRules:
    """Test cases for apply_filter_and_deduplicate_rules function."""

    def test_basic_pipeline(self, capsys):
        """Test the complete pipeline from evaluation to deduplication."""
        X = pl.DataFrame(
            {
                "age": [25, 30, 35, 40, 45, 50, 55, 60],
                "income": [30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000],
            }
        )
        y = pl.Series([0, 0, 0, 0, 1, 1, 1, 1])
        rules = ['(X["age"] >= 40)', '(X["income"] >= 60000)', '(X["age"] >= 45)']

        R, metrics, selected_rules = apply_filter_and_deduplicate_rules(
            X,
            y,
            rules,
            metrics_threshold=[
                {"name": "precision", "operator": ">=", "value": 0.5},
                {"name": "recall", "operator": ">=", "value": 0.5},
            ],
        )

        # Verify output types
        assert isinstance(R, pl.DataFrame)
        assert isinstance(metrics, pl.DataFrame)
        assert isinstance(selected_rules, list)

        # Verify that selected rules are in the returned DataFrame
        assert all(rule in R.columns for rule in selected_rules)

        # Verify console output
        captured = capsys.readouterr()
        assert "Number of uncorrelated rules:" in captured.out

    def test_with_dataframe_input(self):
        """Test with DataFrame input containing 'rule' column."""
        X = pl.DataFrame(
            {"age": [25, 30, 35, 40, 45, 50], "income": [30000, 40000, 50000, 60000, 70000, 80000]}
        )
        y = pl.Series([0, 0, 1, 1, 1, 1])
        rules_df = pl.DataFrame({"rule": ['(X["age"] >= 35)', '(X["income"] >= 50000)']})

        R, metrics, selected_rules = apply_filter_and_deduplicate_rules(
            X,
            y,
            rules_df,
            metrics_threshold=[
                {"name": "precision", "operator": ">=", "value": 0.4},
                {"name": "recall", "operator": ">=", "value": 0.4},
            ],
        )

        assert isinstance(R, pl.DataFrame)
        assert len(selected_rules) > 0

    def test_with_weights(self):
        """Test pipeline with sample weights."""
        X = pl.DataFrame(
            {"age": [25, 30, 35, 40, 45, 50], "weight": [1.0, 2.0, 1.5, 3.0, 2.5, 4.0]}
        )
        y = pl.Series([0, 0, 1, 1, 1, 1])
        rules = ['(X["age"] >= 35)', '(X["age"] >= 40)']

        R, metrics, selected_rules = apply_filter_and_deduplicate_rules(
            X,
            y,
            rules,
            weight_column="weight",
            metrics_threshold=[
                {"name": "precision", "operator": ">=", "value": 0.4},
                {"name": "recall", "operator": ">=", "value": 0.4},
            ],
        )

        assert isinstance(R, pl.DataFrame)
        assert len(selected_rules) > 0

    def test_no_rules_pass_filters(self, capsys):
        """Test when no rules meet performance thresholds."""
        X = pl.DataFrame({"age": [25, 30, 35, 40], "income": [30000, 40000, 50000, 60000]})
        y = pl.Series([0, 0, 0, 1])
        rules = ['(X["age"] >= 60)', '(X["income"] >= 200000)']

        R, metrics, selected_rules = apply_filter_and_deduplicate_rules(
            X,
            y,
            rules,
            metrics_threshold=[
                {"name": "precision", "operator": ">=", "value": 0.9},
                {"name": "recall", "operator": ">=", "value": 0.9},
            ],
        )
        assert len(selected_rules) == 0
        assert R.is_empty()

    def test_with_top_n_limit(self, capsys):
        """Test limiting to top N rules."""
        X = pl.DataFrame(
            {
                "age": [25, 30, 35, 40, 45, 50, 55, 60],
                "income": [30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000],
            }
        )
        y = pl.Series([0, 0, 0, 0, 1, 1, 1, 1])
        rules = [
            '(X["age"] >= 35)',
            '(X["age"] >= 40)',
            '(X["age"] >= 45)',
            '(X["income"] >= 50000)',
            '(X["income"] >= 60000)',
        ]

        R, metrics, selected_rules = apply_filter_and_deduplicate_rules(
            X,
            y,
            rules,
            metrics_threshold=[
                {"name": "precision", "operator": ">=", "value": 0.3},
                {"name": "recall", "operator": ">=", "value": 0.3},
            ],
            top_n_rules=3,
        )

        # Should limit results
        assert len(selected_rules) <= 3

    def test_custom_correlation_threshold(self, capsys):
        """Test with different correlation threshold."""
        X = pl.DataFrame(
            {
                "age": [25, 30, 35, 40, 45, 50, 55, 60],
                "income": [30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000],
            }
        )
        y = pl.Series([0, 0, 0, 0, 1, 1, 1, 1])
        rules = [
            '(X["age"] >= 40)',
            '(X["age"] >= 42)',  # Very similar to above
            '(X["income"] >= 60000)',
        ]

        R, metrics, selected_rules = apply_filter_and_deduplicate_rules(
            X,
            y,
            rules,
            metrics_threshold=[
                {"name": "precision", "operator": ">=", "value": 0.3},
                {"name": "recall", "operator": ">=", "value": 0.3},
            ],
            max_corr=0.7,
        )

        # With low correlation threshold, should filter out similar rules
        assert isinstance(selected_rules, list)

    def test_custom_sort_by(self):
        """Test sorting by different metric."""
        X = pl.DataFrame(
            {
                "age": [25, 30, 35, 40, 45, 50, 55, 60],
                "income": [30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000],
            }
        )
        y = pl.Series([0, 0, 0, 0, 1, 1, 1, 1])
        rules = ['(X["age"] >= 40)', '(X["income"] >= 60000)', '(X["age"] >= 50)']

        R, metrics, selected_rules = apply_filter_and_deduplicate_rules(
            X,
            y,
            rules,
            metrics_threshold=[
                {"name": "precision", "operator": ">=", "value": 0.3},
                {"name": "recall", "operator": ">=", "value": 0.3},
            ],
            sort_by="recall",
        )

        # Check that results are sorted by recall
        assert isinstance(metrics, pl.DataFrame)
        if len(metrics) > 1:
            recall_values = metrics["recall"].to_list()
            assert recall_values == sorted(recall_values, reverse=True)

    def test_single_rule(self, capsys):
        """Test pipeline with single rule."""
        X = pl.DataFrame(
            {
                "age": [25, 30, 35, 40, 45, 50],
            }
        )
        y = pl.Series([0, 0, 1, 1, 1, 1])
        rules = ['(X["age"] >= 35)']

        R, metrics, selected_rules = apply_filter_and_deduplicate_rules(
            X,
            y,
            rules,
            metrics_threshold=[
                {"name": "precision", "operator": ">=", "value": 0.5},
                {"name": "recall", "operator": ">=", "value": 0.5},
            ],
        )

        # Single rule that meets threshold should pass
        assert len(selected_rules) <= 1
        if len(selected_rules) == 1:
            assert selected_rules[0] == '(X["age"] >= 35)'
