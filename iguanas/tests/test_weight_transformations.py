import numpy as np
import polars as pl
import pytest

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

    def test_with_quantile_adds_clipped_columns(self, series):
        qval = float(series.quantile(0.9))
        df, col, powers = _resolve(series, np.array([1.0]))
        exprs = _increasing_exprs(col, powers, qval, 0.9)
        result = df.with_columns(exprs).drop(col)
        assert any("clipped_90th" in c for c in result.columns)


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

    def test_with_quantile_adds_clipped_columns(self, series):
        qval = float(series.quantile(0.9))
        df, col, powers = _resolve(series, np.array([1.0]))
        exprs = _decreasing_exprs(col, powers, qval, 0.9)
        result = df.with_columns(exprs).drop(col)
        assert any("clipped_90th" in c for c in result.columns)


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

    def test_with_quantile_column_count(self, series):
        # Baseline + 5 powers + log + 5 clipped = 12
        result = generate_increasing_weight(series, quantile_value=0.9)
        assert result.shape[1] == 12

    def test_with_quantile_clipped_column_names(self, series):
        result = generate_increasing_weight(series, quantile_value=0.9)
        assert any("clipped_90th" in c for c in result.columns)

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

    def test_with_quantile_column_count(self, series):
        result = generate_decreasing_weight(series, quantile_value=0.9)
        assert result.shape[1] == 12

    def test_with_quantile_clipped_column_names(self, series):
        result = generate_decreasing_weight(series, quantile_value=0.9)
        assert any("clipped_90th" in c for c in result.columns)

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

    def test_with_quantile_column_count(self, series):
        # 13 + 5 inc clipped + 5 dec clipped = 23
        result = generate_all_weight(series, quantile_value=0.9)
        assert result.shape[1] == 23

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
