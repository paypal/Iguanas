import polars as pl
import pytest

from iguanas.rule_fairness import compute_disparate_impact_ratio, compute_subgroup_metrics


class TestComputeSubgroupMetrics:
    @pytest.fixture
    def base_data(self):
        R = pl.DataFrame(
            {
                "rule_A": [True, False, True, False, True, False],
                "rule_B": [True, True, False, False, True, True],
            }
        )
        y = pl.Series([1, 0, 1, 0, 1, 0])
        group = pl.Series(["M", "M", "F", "F", "M", "F"])
        return R, y, group

    def test_returns_dataframe(self, base_data):
        R, y, group = base_data
        result = compute_subgroup_metrics(R, y, group)
        assert isinstance(result, pl.DataFrame)

    def test_group_and_rule_columns_present(self, base_data):
        R, y, group = base_data
        result = compute_subgroup_metrics(R, y, group)
        assert "group" in result.columns
        assert "rule" in result.columns
        assert "group_size" in result.columns

    def test_row_count_equals_groups_times_rules(self, base_data):
        R, y, group = base_data
        result = compute_subgroup_metrics(R, y, group)
        n_groups = group.n_unique()
        n_rules = len(R.columns)
        assert result.shape[0] == n_groups * n_rules

    def test_group_sizes_correct(self, base_data):
        R, y, group = base_data
        result = compute_subgroup_metrics(R, y, group)
        # F appears 3 times, M appears 3 times
        f_rows = result.filter(pl.col("group") == "F")
        m_rows = result.filter(pl.col("group") == "M")
        assert f_rows["group_size"].unique().to_list() == [3]
        assert m_rows["group_size"].unique().to_list() == [3]

    def test_metric_columns_present(self, base_data):
        R, y, group = base_data
        result = compute_subgroup_metrics(R, y, group)
        for col in ("precision", "recall", "f1"):
            assert col in result.columns

    def test_sorted_by_group_then_rule(self, base_data):
        R, y, group = base_data
        result = compute_subgroup_metrics(R, y, group)
        groups = result["group"].to_list()
        rules = result["rule"].to_list()
        expected = sorted(zip(groups, rules))
        actual = list(zip(groups, rules))
        assert actual == expected

    def test_three_groups(self):
        R = pl.DataFrame({"rule_X": [True, False, True, False, True, False]})
        y = pl.Series([1, 0, 1, 0, 1, 0])
        group = pl.Series(["A", "A", "B", "B", "C", "C"])
        result = compute_subgroup_metrics(R, y, group)
        assert result["group"].n_unique() == 3

    def test_empty_R_returns_empty(self):
        R = pl.DataFrame()
        y = pl.Series([1, 0])
        group = pl.Series(["A", "B"])
        result = compute_subgroup_metrics(R, y, group)
        assert result.is_empty()

    def test_with_weights(self, base_data):
        R, y, group = base_data
        weights = pl.Series([1.0, 2.0, 1.5, 3.0, 1.0, 2.5])
        result = compute_subgroup_metrics(R, y, group, weights=weights)
        assert "precision_weight" in result.columns
        assert result.shape[0] > 0

    def test_precision_in_range(self, base_data):
        R, y, group = base_data
        result = compute_subgroup_metrics(R, y, group)
        assert (result["precision"].drop_nulls() >= 0).all()
        assert (result["precision"].drop_nulls() <= 1).all()


class TestComputeDisparateImpactRatio:
    @pytest.fixture
    def subgroup_df(self):
        return pl.DataFrame(
            {
                "group":     ["A", "A", "B", "B"],
                "rule":      ["r1", "r2", "r1", "r2"],
                "precision": [0.8, 0.6, 0.4, 0.9],
            }
        )

    def test_ratio_computed_correctly(self, subgroup_df):
        result = compute_disparate_impact_ratio(subgroup_df, reference_group="A", metric="precision")
        # r1: group B / reference A = 0.4 / 0.8 = 0.5
        r1_row = result.filter(pl.col("rule") == "r1")
        assert r1_row["disparate_impact_ratio"][0] == pytest.approx(0.5)

    def test_reference_group_excluded(self, subgroup_df):
        result = compute_disparate_impact_ratio(subgroup_df, reference_group="A", metric="precision")
        assert "A" not in result["group"].to_list()

    def test_sorted_by_rule_and_ratio(self, subgroup_df):
        result = compute_disparate_impact_ratio(subgroup_df, reference_group="A", metric="precision")
        assert result["rule"].to_list() == sorted(result["rule"].to_list())

    def test_raises_on_missing_column(self, subgroup_df):
        with pytest.raises(ValueError, match="missing columns"):
            compute_disparate_impact_ratio(subgroup_df, reference_group="A", metric="recall")
