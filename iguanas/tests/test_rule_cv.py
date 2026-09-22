import polars as pl
import pytest

from iguanas.rule_cv import identify_unstable_rules, validate_rules_cv


class TestValidateRulesCv:
    @pytest.fixture
    def data(self):
        X = pl.DataFrame({"age": list(range(20, 70)), "income": list(range(10_000, 60_000, 1_000))})
        y = pl.Series([0] * 25 + [1] * 25)
        rules = ['(X["age"] >= 45)', '(X["income"] >= 35000)']
        return X, y, rules

    def test_returns_one_row_per_rule(self, data):
        X, y, rules = data
        result = validate_rules_cv(X, y, rules, n_folds=5, random_state=0)
        assert isinstance(result, pl.DataFrame)
        assert result.shape[0] == len(rules)
        assert "rule" in result.columns

    def test_default_cv_metric_columns_present(self, data):
        X, y, rules = data
        result = validate_rules_cv(X, y, rules, n_folds=5, random_state=0)
        for metric in ("precision", "recall", "f1"):
            assert f"{metric}_cv_mean" in result.columns
            assert f"{metric}_cv_std" in result.columns
            assert f"{metric}_cv_min" in result.columns

    def test_custom_cv_metrics(self, data):
        X, y, rules = data
        result = validate_rules_cv(
            X, y, rules, n_folds=3, cv_metrics=["precision"], random_state=0
        )
        assert "precision_cv_mean" in result.columns
        assert "recall_cv_mean" not in result.columns

    def test_mean_values_in_valid_range(self, data):
        X, y, rules = data
        result = validate_rules_cv(X, y, rules, n_folds=5, random_state=0)
        for col in result.columns:
            if col.endswith("_cv_mean"):
                assert (result[col].drop_nulls() >= 0).all()
                assert (result[col].drop_nulls() <= 1).all()

    def test_min_le_mean(self, data):
        """cv_min must be <= cv_mean for every rule and metric."""
        X, y, rules = data
        result = validate_rules_cv(X, y, rules, n_folds=5, random_state=0)
        for metric in ("precision", "recall", "f1"):
            assert (
                result[f"{metric}_cv_min"].drop_nulls()
                <= result[f"{metric}_cv_mean"].drop_nulls()
            ).all()

    def test_with_weight_column(self, data):
        X, y, rules = data
        X = X.with_columns(pl.Series("w", [1.0] * len(X)))
        result = validate_rules_cv(X, y, rules, n_folds=3, weight_column="w", random_state=0)
        assert result.shape[0] == len(rules)

    def test_no_shuffle(self, data):
        X, y, rules = data
        result = validate_rules_cv(X, y, rules, n_folds=5, shuffle=False)
        assert result.shape[0] == len(rules)

    def test_two_folds(self, data):
        X, y, rules = data
        result = validate_rules_cv(X, y, rules, n_folds=2, random_state=42)
        assert result.shape[0] == len(rules)


class TestIdentifyUnstableRules:
    @pytest.fixture
    def cv_result(self):
        return pl.DataFrame(
            {
                "rule": ["rule_A", "rule_B", "rule_C"],
                "f1_cv_mean": [0.8, 0.5, 0.9],
                "f1_cv_std":  [0.01, 0.12, 0.03],
                "f1_cv_min":  [0.78, 0.35, 0.85],
            }
        )

    def test_flags_high_std_rule(self, cv_result):
        result = identify_unstable_rules(cv_result, metric="f1", max_std=0.05)
        assert "rule_B" in result["rule"].to_list()
        assert "rule_A" not in result["rule"].to_list()

    def test_flags_low_mean_rule(self, cv_result):
        result = identify_unstable_rules(cv_result, metric="f1", max_std=0.05, min_mean=0.6)
        rules = result["rule"].to_list()
        assert "rule_B" in rules  # high std AND low mean

    def test_sorted_by_std_descending(self, cv_result):
        result = identify_unstable_rules(cv_result, metric="f1", max_std=0.0)
        stds = result["f1_cv_std"].to_list()
        assert stds == sorted(stds, reverse=True)

    def test_returns_empty_when_all_stable(self, cv_result):
        result = identify_unstable_rules(cv_result, metric="f1", max_std=1.0)
        assert result.is_empty()

    def test_raises_missing_std_column(self, cv_result):
        with pytest.raises(ValueError, match="not found"):
            identify_unstable_rules(cv_result, metric="recall", max_std=0.05)

    def test_raises_missing_mean_column_when_min_mean_set(self, cv_result):
        # Drop the mean column; std is still present — hits the mean-column error
        partial = cv_result.drop("f1_cv_mean")
        with pytest.raises(ValueError, match="not found"):
            identify_unstable_rules(partial, metric="f1", max_std=0.05, min_mean=0.5)
