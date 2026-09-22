import polars as pl
import pytest

from iguanas.rule_formatting import rule_to_sql
from iguanas.rule_evaluation import apply_rules_lazy


class TestRuleToSql:
    def test_gt_operator(self):
        assert rule_to_sql('(X["age"] > 30)') == "(age > 30.0)"

    def test_gte_operator(self):
        assert rule_to_sql('(X["income"] >= 50000)') == "(income >= 50000.0)"

    def test_lt_operator(self):
        assert rule_to_sql('(X["risk"] < 0.5)') == "(risk < 0.5)"

    def test_lte_operator(self):
        assert rule_to_sql('(X["score"] <= 1.0)') == "(score <= 1.0)"

    def test_eq_operator_translates_to_single_equals(self):
        sql = rule_to_sql('(X["flag"] == 1)')
        assert "= 1.0" in sql
        assert "==" not in sql

    def test_neq_operator(self):
        assert rule_to_sql('(X["status"] != 0)') == "(status != 0.0)"

    def test_and_becomes_AND(self):
        sql = rule_to_sql('(X["age"] > 30) & (X["income"] < 50000)')
        assert " AND " in sql
        assert "&" not in sql

    def test_or_becomes_OR(self):
        sql = rule_to_sql('(X["age"] > 30) | (X["flag"] == 1)')
        assert " OR " in sql
        assert "|" not in sql

    def test_table_alias_prefixes_columns(self):
        sql = rule_to_sql('(X["age"] > 30) & (X["income"] < 50000)', table_alias="t")
        assert "(t.age > 30.0)" in sql
        assert "(t.income < 50000.0)" in sql

    def test_no_table_alias(self):
        sql = rule_to_sql('(X["age"] > 30)', table_alias=None)
        assert "t." not in sql
        assert "(age > 30.0)" in sql

    def test_compound_rule(self):
        rule = '(X["a"] >= 10) & (X["b"] < 5) | (X["c"] != 0)'
        sql = rule_to_sql(rule)
        assert "AND" in sql or "OR" in sql
        assert "X[" not in sql


class TestApplyRulesLazy:
    def test_basic_lazy_evaluation(self):
        X = pl.DataFrame({"age": [25, 35, 45], "income": [30_000, 50_000, 80_000]})
        lf = X.lazy()
        rules = ['(X["age"] >= 35)', '(X["income"] > 40000)']
        result_lazy = apply_rules_lazy(lf, rules)
        result = result_lazy.collect()

        expected = pl.DataFrame(
            {
                '(X["age"] >= 35)': [False, True, True],
                '(X["income"] > 40000)': [False, True, True],
            }
        )
        assert result.equals(expected)

    def test_columns_match_rule_strings(self):
        X = pl.DataFrame({"age": [25, 35]})
        rules = ['(X["age"] > 30)']
        result = apply_rules_lazy(X.lazy(), rules).collect()
        assert result.columns == rules

    def test_empty_rules_returns_empty_lazyframe(self):
        X = pl.DataFrame({"age": [25, 35]})
        result = apply_rules_lazy(X.lazy(), []).collect()
        assert result.shape[1] == 0

    def test_lazy_matches_eager(self):
        from iguanas.rule_evaluation import apply_rules

        X = pl.DataFrame(
            {"amount": [100, 200, 300, 50], "age": [20, 40, 60, 30]}
        )
        rules = [
            '(X["amount"] >= 150)',
            '(X["age"] < 50)',
            '(X["amount"] >= 100) & (X["age"] < 45)',
        ]
        eager = apply_rules(X, rules)
        lazy = apply_rules_lazy(X.lazy(), rules).collect()
        assert eager.equals(lazy)

    def test_returns_lazyframe_type(self):
        X = pl.DataFrame({"age": [25, 35]})
        result = apply_rules_lazy(X.lazy(), ['(X["age"] > 30)'])
        assert isinstance(result, pl.LazyFrame)


class TestRuleToSqlNonNumeric:
    def test_non_numeric_value_quoted_as_string(self):
        """rule_to_sql quotes non-numeric values as SQL string literals."""
        sql = rule_to_sql('(X["category"] == active)')
        assert "= 'active'" in sql
        assert "X[" not in sql
