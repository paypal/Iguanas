import numpy as np
import polars as pl
import pytest

import iguanas.weight_transformations as wt
from iguanas.weight_transformations import (
    _DEFAULT_POWERS,
    EPS,
    _decreasing_exprs,
    _dispatch,
    _increasing_exprs,
    _power_label,
    _resolve,
    generate_all_weight,
    generate_decreasing_weight,
    generate_increasing_weight,
    select_uncorrelated_weights,
)


@pytest.fixture
def series():
    return pl.Series("x", [1.0, 2.0, 3.0, 4.0, 5.0])


@pytest.fixture
def dataframe():
    return pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})


# ---------------------------------------------------------------------------
# _power_label
# ---------------------------------------------------------------------------


class TestPowerLabel:
    def test_integer_power(self):
        assert _power_label(2.0) == "2"
        assert _power_label(4.0) == "4"
        assert _power_label(1.0) == "1"

    def test_non_integer_power(self):
        assert _power_label(0.25) == "0.25"
        assert _power_label(0.50) == "0.50"


# ---------------------------------------------------------------------------
# _resolve
# ---------------------------------------------------------------------------


class TestResolve:
    def test_default_powers_used_when_none(self, series):
        _, _, powers = _resolve(series, None)
        np.testing.assert_array_equal(powers, _DEFAULT_POWERS)

    def test_custom_powers_preserved(self, series):
        custom = np.array([1.0, 2.0])
        _, _, powers = _resolve(series, custom)
        np.testing.assert_array_equal(powers, custom)

    def test_min_subtracted_from_values(self):
        s = pl.Series("v", [3.0, 5.0, 7.0])
        df, col_name, _ = _resolve(s, None)
        assert col_name == "v"
        assert df["v"].to_list() == pytest.approx([0.0, 2.0, 4.0])

    def test_col_name_matches_series_name(self, series):
        _, col_name, _ = _resolve(series, None)
        assert col_name == "x"


# ---------------------------------------------------------------------------
# _dispatch
# ---------------------------------------------------------------------------


class TestDispatch:
    def test_series_returns_none(self, series):
        result = _dispatch(generate_increasing_weight, series)
        assert result is None

    def test_dataframe_returns_dataframe(self, dataframe):
        result = _dispatch(generate_increasing_weight, dataframe)
        assert isinstance(result, pl.DataFrame)

    def test_dataframe_baseline_appears_once(self, dataframe):
        result = _dispatch(generate_increasing_weight, dataframe)
        assert result.columns.count("Baseline") == 1

    def test_dataframe_both_column_suffixes_present(self, dataframe):
        result = _dispatch(generate_increasing_weight, dataframe)
        assert any(c.endswith("__a") for c in result.columns)
        assert any(c.endswith("__b") for c in result.columns)


# ---------------------------------------------------------------------------
# _increasing_exprs / _decreasing_exprs (via the public API for coverage)
# ---------------------------------------------------------------------------


class TestIncreasingExprs:
    def test_without_quantile_baseline_included(self, series):
        df, col, powers = _resolve(series, np.array([1.0, 2.0]))
        exprs = _increasing_exprs(col, powers)
        result = df.with_columns(exprs).drop(col)
        assert "Baseline" in result.columns

    def test_without_quantile_special_p1_label(self, series):
        # p=1.0 uses label "(1+x)" not "(1+x)^1"
        df, col, powers = _resolve(series, np.array([1.0]))
        exprs = _increasing_exprs(col, powers)
        result = df.with_columns(exprs).drop(col)
        assert "(1+x)__x" in result.columns
        assert "(1+x)^1__x" not in result.columns

    def test_without_quantile_integer_power_label(self, series):
        df, col, powers = _resolve(series, np.array([2.0]))
        exprs = _increasing_exprs(col, powers)
        result = df.with_columns(exprs).drop(col)
        assert "(1+x)^2__x" in result.columns

    def test_without_quantile_non_integer_power_label(self, series):
        df, col, powers = _resolve(series, np.array([0.25]))
        exprs = _increasing_exprs(col, powers)
        result = df.with_columns(exprs).drop(col)
        assert "(1+x)^0.25__x" in result.columns


class TestDecreasingExprs:
    def test_without_quantile_special_p1_label(self, series):
        # p=1.0 uses label "1/(1+x)" not "1/(1+x)^1"
        df, col, powers = _resolve(series, np.array([1.0]))
        exprs = _decreasing_exprs(col, powers)
        result = df.with_columns(exprs).drop(col)
        assert "1/(1+x)__x" in result.columns

    def test_without_quantile_non_integer_power_label(self, series):
        df, col, powers = _resolve(series, np.array([0.25]))
        exprs = _decreasing_exprs(col, powers)
        result = df.with_columns(exprs).drop(col)
        assert "1/(1+x)^0.25__x" in result.columns

    def test_without_quantile_integer_power_label(self, series):
        df, col, powers = _resolve(series, np.array([2.0]))
        exprs = _decreasing_exprs(col, powers)
        result = df.with_columns(exprs).drop(col)
        assert "1/(1+x)^2__x" in result.columns

    def test_log_inverse_column_present(self, series):
        df, col, powers = _resolve(series, np.array([1.0]))
        exprs = _decreasing_exprs(col, powers)
        result = df.with_columns(exprs).drop(col)
        assert "1/log(1+x)__x" in result.columns


# ---------------------------------------------------------------------------
# generate_increasing_weight
# ---------------------------------------------------------------------------


class TestGenerateIncreasingWeight:
    def test_series_returns_dataframe(self, series):
        result = generate_increasing_weight(series)
        assert isinstance(result, pl.DataFrame)

    def test_series_has_baseline(self, series):
        result = generate_increasing_weight(series)
        assert "Baseline" in result.columns

    def test_baseline_is_all_ones(self, series):
        result = generate_increasing_weight(series)
        assert result["Baseline"].to_list() == [1.0] * len(series)

    def test_default_column_count(self, series):
        # Baseline + 5 powers + log = 7
        result = generate_increasing_weight(series)
        assert result.shape[1] == 7

    def test_custom_powers_column_count(self, series):
        # Baseline + 2 powers + log = 4
        result = generate_increasing_weight(series, powers=np.array([1.0, 2.0]))
        assert result.shape[1] == 4

    def test_power1_column_values(self, series):
        # shifted: [0,1,2,3,4]; (1+x)^1 = [1,2,3,4,5]
        result = generate_increasing_weight(series, powers=np.array([1.0]))
        assert result["(1+x)__x"].to_list() == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0])

    def test_power2_column_values(self, series):
        # shifted: [0,1,2,3,4]; (1+x)^2 = [1,4,9,16,25]
        result = generate_increasing_weight(series, powers=np.array([2.0]))
        assert result["(1+x)^2__x"].to_list() == pytest.approx([1.0, 4.0, 9.0, 16.0, 25.0])

    def test_log_column_present(self, series):
        result = generate_increasing_weight(series)
        assert "log(1+x)__x" in result.columns

    def test_increasing_monotone(self, series):
        result = generate_increasing_weight(series, powers=np.array([1.0]))
        vals = result["(1+x)__x"].to_list()
        assert vals == sorted(vals)

    def test_dataframe_input(self, dataframe):
        result = generate_increasing_weight(dataframe)
        assert isinstance(result, pl.DataFrame)
        assert result.columns.count("Baseline") == 1
        assert any(c.endswith("__a") for c in result.columns)
        assert any(c.endswith("__b") for c in result.columns)

    def test_dataframe_column_count(self, dataframe):
        # col "a": 7; col "b": 7 - 1 (Baseline dropped) = 6; total = 13
        result = generate_increasing_weight(dataframe)
        assert result.shape[1] == 13


# ---------------------------------------------------------------------------
# generate_decreasing_weight
# ---------------------------------------------------------------------------


class TestGenerateDecreasingWeight:
    def test_series_returns_dataframe(self, series):
        result = generate_decreasing_weight(series)
        assert isinstance(result, pl.DataFrame)

    def test_series_has_baseline(self, series):
        result = generate_decreasing_weight(series)
        assert "Baseline" in result.columns

    def test_baseline_is_all_ones(self, series):
        result = generate_decreasing_weight(series)
        assert result["Baseline"].to_list() == [1.0] * len(series)

    def test_default_column_count(self, series):
        # Baseline + 5 reciprocal powers + 1/log = 7
        result = generate_decreasing_weight(series)
        assert result.shape[1] == 7

    def test_reciprocal_column_values(self, series):
        # shifted: [0,1,2,3,4]; 1/(1+x) = [1, 0.5, 1/3, 0.25, 0.2]
        result = generate_decreasing_weight(series, powers=np.array([1.0]))
        expected = [1.0, 0.5, 1 / 3, 0.25, 0.2]
        assert result["1/(1+x)__x"].to_list() == pytest.approx(expected)

    def test_decreasing_monotone(self, series):
        result = generate_decreasing_weight(series, powers=np.array([1.0]))
        vals = result["1/(1+x)__x"].to_list()
        assert vals == sorted(vals, reverse=True)

    def test_log_inverse_column_present(self, series):
        result = generate_decreasing_weight(series)
        assert "1/log(1+x)__x" in result.columns

    def test_dataframe_input(self, dataframe):
        result = generate_decreasing_weight(dataframe)
        assert isinstance(result, pl.DataFrame)
        assert result.columns.count("Baseline") == 1
        assert any(c.endswith("__a") for c in result.columns)
        assert any(c.endswith("__b") for c in result.columns)


class TestSelectUncorrelatedWeights:
    @pytest.mark.parametrize(
        "min_corr,max_corr",
        [
            (0.0, 0.5),
            (0.5, 0.5),
            (0.75, 0.25),
        ],
    )
    def test_invalid_correlation_bounds_raises(self, min_corr, max_corr):
        sample_weights = pl.DataFrame({"A": [1.0], "B": [1.0]})
        importance = {"A": 1.0, "B": 2.0}

        with pytest.raises(ValueError, match="min_corr and max_corr must satisfy"):
            select_uncorrelated_weights(
                sample_weights,
                importance,
                target_len=1,
                min_corr=min_corr,
                max_corr=max_corr,
            )

    def test_negative_target_len_raises(self):
        sample_weights = pl.DataFrame({"A": [1.0], "B": [1.0]})
        importance = {"A": 1.0, "B": 2.0}

        with pytest.raises(ValueError, match="target_len must be non-negative"):
            select_uncorrelated_weights(sample_weights, importance, target_len=-1)

    def test_non_positive_step_raises(self):
        sample_weights = pl.DataFrame({"A": [1.0], "B": [1.0]})
        importance = {"A": 1.0, "B": 2.0}

        with pytest.raises(ValueError, match="step must be positive"):
            select_uncorrelated_weights(
                sample_weights,
                importance,
                target_len=1,
                step=0.0,
            )

    def test_returns_closest_filtered_set_for_target_len(self):
        sample_weights = pl.DataFrame(
            {
                "A": [1.0, 0.8, 0.4],
                "B": [0.8, 1.0, 0.7],
                "C": [0.4, 0.7, 1.0],
            }
        )
        importance = {"A": 1.0, "B": 2.0, "C": 3.0}

        selected, corr_value = select_uncorrelated_weights(
            sample_weights,
            importance,
            target_len=2,
            min_corr=0.01,
            max_corr=0.99,
            step=0.01,
        )

        assert selected == ["B", "C"]
        assert corr_value == pytest.approx(0.75)

    def test_returns_next_closest_threshold_when_exact_target_is_impossible(self):
        sample_weights = pl.DataFrame(
            {
                "A": [1.0, 0.9, 0.9],
                "B": [0.9, 1.0, 0.9],
                "C": [0.9, 0.9, 1.0],
            }
        )
        importance = {"A": 1.0, "B": 2.0, "C": 3.0}

        selected, corr_value = select_uncorrelated_weights(
            sample_weights,
            importance,
            target_len=2,
            min_corr=0.01,
            max_corr=0.99,
            step=0.01,
        )

        assert selected == ["A", "B", "C"]
        assert corr_value == pytest.approx(0.90)

    def test_returns_max_when_search_exhausts_without_exact_match(self, monkeypatch):
        call_state = {"count": 0}

        def fake_filter_correlated_rules(R, importance, max_corr, use_abs=False):
            if call_state["count"] == 0:
                call_state["count"] += 1
                assert max_corr == 0.01
                return ["A"]
            if call_state["count"] == 1:
                call_state["count"] += 1
                assert pytest.approx(max_corr, rel=1e-8) == 0.99
                return ["A", "B", "C"]
            call_state["count"] += 1
            return ["A"]

        monkeypatch.setattr(wt, "filter_correlated_rules", fake_filter_correlated_rules)
        selected, corr_value = select_uncorrelated_weights(
            pl.DataFrame({"A": [0.0], "B": [0.0], "C": [0.0]}),
            {"A": 1.0, "B": 2.0, "C": 3.0},
            target_len=2,
            min_corr=0.01,
            max_corr=0.99,
            step=0.01,
        )

        assert selected == ["A", "B", "C"]
        assert corr_value == pytest.approx(0.99)

    def test_returns_minimum_when_target_len_below_range(self):
        sample_weights = pl.DataFrame(
            {
                "A": [1.0, 0.8, 0.4],
                "B": [0.8, 1.0, 0.7],
                "C": [0.4, 0.7, 1.0],
            }
        )
        importance = {"A": 1.0, "B": 2.0, "C": 3.0}

        selected, corr_value = select_uncorrelated_weights(
            sample_weights,
            importance,
            target_len=1,
            min_corr=0.01,
            max_corr=0.99,
            step=0.01,
        )

        assert selected == ["C"]
        assert corr_value == pytest.approx(0.01)

    def test_returns_maximum_when_target_len_above_range(self):
        sample_weights = pl.DataFrame(
            {
                "A": [1.0, 0.8, 0.4],
                "B": [0.8, 1.0, 0.7],
                "C": [0.4, 0.7, 1.0],
            }
        )
        importance = {"A": 1.0, "B": 2.0, "C": 3.0}

        selected, corr_value = select_uncorrelated_weights(
            sample_weights,
            importance,
            target_len=4,
            min_corr=0.01,
            max_corr=0.99,
            step=0.01,
        )

        assert selected == ["A", "B", "C"]
        assert corr_value == pytest.approx(0.99)

    def test_use_abs_false_preserves_negative_correlations(self):
        sample_weights = pl.DataFrame(
            {
                "A": [1.0, -0.95],
                "B": [-0.95, 1.0],
            }
        )
        importance = {"A": 1.0, "B": 2.0}

        selected, corr_value = select_uncorrelated_weights(
            sample_weights,
            importance,
            target_len=2,
            min_corr=0.01,
            max_corr=0.99,
            step=0.01,
            use_abs=False,
        )

        assert selected == ["A", "B"]
        assert corr_value == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# generate_all_weight
# ---------------------------------------------------------------------------


class TestGenerateAllWeight:
    def test_series_returns_dataframe(self, series):
        result = generate_all_weight(series)
        assert isinstance(result, pl.DataFrame)

    def test_series_has_baseline(self, series):
        result = generate_all_weight(series)
        assert "Baseline" in result.columns

    def test_default_column_count(self, series):
        # Baseline + 5 inc powers + log + 5 dec powers + 1/log = 13
        result = generate_all_weight(series)
        assert result.shape[1] == 13

    def test_contains_increasing_columns(self, series):
        result = generate_all_weight(series)
        assert "log(1+x)__x" in result.columns

    def test_contains_decreasing_columns(self, series):
        result = generate_all_weight(series)
        assert "1/log(1+x)__x" in result.columns
        assert "1/(1+x)__x" in result.columns

    def test_union_of_increasing_and_decreasing(self, series):
        inc = generate_increasing_weight(series)
        dec = generate_decreasing_weight(series)
        all_ = generate_all_weight(series)
        expected_cols = set(inc.columns) | (set(dec.columns) - {"Baseline"})
        assert set(all_.columns) == expected_cols

    def test_dataframe_input(self, dataframe):
        result = generate_all_weight(dataframe)
        assert isinstance(result, pl.DataFrame)
        assert result.columns.count("Baseline") == 1
        assert any(c.endswith("__a") for c in result.columns)
        assert any(c.endswith("__b") for c in result.columns)
