import polars as pl
import pytest

from iguanas.rule_explanation import (
    compute_counterfactual,
    compute_coverage_overlap,
    verbalize_rule,
)


class TestVerbalizeRule:
    def test_single_gt_condition(self):
        assert verbalize_rule('(X["age"] > 30)') == "age is greater than 30"

    def test_single_gte_condition(self):
        assert verbalize_rule('(X["score"] >= 0.8)') == "score is at least 0.8"

    def test_single_lt_condition(self):
        assert verbalize_rule('(X["income"] < 50000)') == "income is less than 50000"

    def test_single_lte_condition(self):
        assert verbalize_rule('(X["risk"] <= 0.5)') == "risk is at most 0.5"

    def test_eq_condition(self):
        assert verbalize_rule('(X["flag"] == 1)') == "flag is equal to 1"

    def test_neq_condition(self):
        assert verbalize_rule('(X["status"] != 0)') == "status is not equal to 0"

    def test_and_combination(self):
        result = verbalize_rule('(X["age"] > 30) & (X["income"] < 50000)')
        assert "age is greater than 30" in result
        assert "income is less than 50000" in result
        assert "AND" in result

    def test_or_combination(self):
        result = verbalize_rule('(X["score"] >= 0.8) | (X["flag"] == 1)')
        assert "OR" in result

    def test_no_parentheses_in_output(self):
        result = verbalize_rule('(X["age"] > 30) & (X["income"] < 50000)')
        assert "(" not in result
        assert ")" not in result


class TestComputeCoverageOverlap:
    def test_basic_overlap(self):
        R = pl.DataFrame(
            {
                "rule_A": [True, True, False, False],
                "rule_B": [True, False, True, False],
            }
        )
        result = compute_coverage_overlap(R)
        assert isinstance(result, pl.DataFrame)
        assert result.shape[0] == 1
        assert "jaccard" in result.columns
        assert "flagged_by_both" in result.columns
        assert "flagged_by_either" in result.columns
        # both=1, either=3 → jaccard ≈ 0.333
        assert abs(result["jaccard"][0] - 1 / 3) < 1e-9

    def test_identical_rules(self):
        R = pl.DataFrame(
            {
                "rule_A": [True, True, False],
                "rule_B": [True, True, False],
            }
        )
        result = compute_coverage_overlap(R)
        assert result["jaccard"][0] == pytest.approx(1.0)

    def test_disjoint_rules(self):
        R = pl.DataFrame(
            {
                "rule_A": [True, True, False, False],
                "rule_B": [False, False, True, True],
            }
        )
        result = compute_coverage_overlap(R)
        assert result["jaccard"][0] == pytest.approx(0.0)
        assert result["flagged_by_both"][0] == 0

    def test_single_rule_returns_empty(self):
        R = pl.DataFrame({"rule_A": [True, False, True]})
        result = compute_coverage_overlap(R)
        assert result.is_empty()

    def test_three_rules_pair_count(self):
        R = pl.DataFrame(
            {
                "rule_A": [True, False, False],
                "rule_B": [False, True, False],
                "rule_C": [False, False, True],
            }
        )
        result = compute_coverage_overlap(R)
        # 3 rules → C(3,2) = 3 pairs
        assert result.shape[0] == 3

    def test_sorted_descending_by_jaccard(self):
        R = pl.DataFrame(
            {
                "rule_A": [True, True, False],
                "rule_B": [True, True, False],  # jaccard=1 with A
                "rule_C": [True, False, False],  # jaccard<1 with A
            }
        )
        result = compute_coverage_overlap(R)
        jaccards = result["jaccard"].to_list()
        assert jaccards == sorted(jaccards, reverse=True)


class TestComputeCounterfactual:
    @pytest.fixture
    def sample(self):
        return pl.DataFrame({"age": [45], "income": [80_000]})

    def test_and_rule_flagged_sample(self, sample):
        rule = '(X["age"] > 30) & (X["income"] >= 50000)'
        results = compute_counterfactual(rule, sample)
        assert len(results) == 2
        features = {r["feature"] for r in results}
        assert features == {"age", "income"}

    def test_not_flagged_returns_empty(self):
        sample = pl.DataFrame({"age": [20]})
        rule = '(X["age"] > 30)'
        assert compute_counterfactual(rule, sample) == []

    def test_sorted_by_abs_change(self, sample):
        rule = '(X["age"] > 30) & (X["income"] >= 50000)'
        results = compute_counterfactual(rule, sample)
        changes = [r["abs_change"] for r in results]
        assert changes == sorted(changes)

    def test_single_condition_suggestion_breaks_condition(self):
        sample = pl.DataFrame({"age": [45]})
        rule = '(X["age"] > 30)'
        results = compute_counterfactual(rule, sample)
        assert len(results) == 1
        # Suggested value must be <= threshold (30)
        assert results[0]["suggested_value"] <= 30

    def test_upper_bound_condition(self):
        sample = pl.DataFrame({"risk": [0.3]})
        rule = '(X["risk"] < 0.5)'
        results = compute_counterfactual(rule, sample)
        assert len(results) == 1
        assert results[0]["suggested_value"] >= 0.5

    def test_raises_on_multiple_rows(self):
        sample = pl.DataFrame({"age": [45, 30]})
        with pytest.raises(ValueError, match="exactly 1 row"):
            compute_counterfactual('(X["age"] > 20)', sample)

    def test_unknown_feature_skipped(self):
        sample = pl.DataFrame({"other": [10]})
        rule = '(X["age"] > 30)'
        results = compute_counterfactual(rule, sample)
        assert results == []

    def test_non_numeric_threshold_skipped(self):
        """Conditions with non-numeric thresholds (e.g. boolean literals) are skipped."""
        # flag == True fires on this sample; float("True") raises ValueError → skipped
        sample = pl.DataFrame({"flag": [True], "age": [45]})
        rule = '(X["age"] > 30) & (X["flag"] == True)'
        results = compute_counterfactual(rule, sample)
        # Only the numeric age condition should produce a counterfactual
        assert all(r["feature"] != "flag" for r in results)

    def test_unsatisfied_condition_in_or_rule_skipped(self):
        """In an OR rule, conditions that don't fire for the sample are skipped."""
        sample = pl.DataFrame({"age": [45], "income": [5_000]})
        # Rule fires because age > 30; income >= 100_000 does NOT fire
        rule = '(X["age"] > 30) | (X["income"] >= 100000)'
        results = compute_counterfactual(rule, sample)
        features = {r["feature"] for r in results}
        assert "income" not in features  # income condition was unsatisfied → skipped
        assert "age" in features

    def test_eq_operator_counterfactual(self):
        """== operator: suggested value is current + epsilon."""
        sample = pl.DataFrame({"flag": [1]})
        rule = '(X["flag"] == 1)'
        results = compute_counterfactual(rule, sample)
        assert len(results) == 1
        assert results[0]["feature"] == "flag"
        assert results[0]["suggested_value"] > 1.0

    def test_neq_operator_counterfactual(self):
        """!= operator: suggested value is the forbidden threshold itself."""
        sample = pl.DataFrame({"flag": [1]})
        rule = '(X["flag"] != 0)'
        results = compute_counterfactual(rule, sample)
        assert len(results) == 1
        assert results[0]["feature"] == "flag"
        assert results[0]["suggested_value"] == pytest.approx(0.0)
