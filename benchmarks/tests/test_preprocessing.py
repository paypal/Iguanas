"""Regression tests for the shared preprocessing stage."""

from __future__ import annotations

import polars as pl

from benchmarks.preprocessing import SharedPreprocessor


class TestDtypePreservation:
    """Continuous features must survive preprocessing intact.

    A blanket ``.cast(pl.Int8)`` once truncated every continuous column to an
    integer and overflowed outright above 127. That both raised on some
    datasets and -- worse -- silently destroyed the split points rule learners
    depend on wherever it did not raise.
    """

    def test_large_values_are_not_overflowed(self) -> None:
        X = pl.DataFrame({"num_operands": [147.0, 355.0, 12.0]})
        out = SharedPreprocessor(discretize=False).fit_transform(X)
        assert out["num_operands"].to_list() == [147.0, 355.0, 12.0]

    def test_fractional_values_are_not_truncated(self) -> None:
        X = pl.DataFrame({"ratio": [0.1, 0.9, 0.5]})
        out = SharedPreprocessor(discretize=False).fit_transform(X)
        assert out["ratio"].to_list() == [0.1, 0.9, 0.5]
        assert out["ratio"].n_unique() == 3

    def test_dummies_stay_compact(self) -> None:
        X = pl.DataFrame({"cat": ["a", "b", "a"]})
        out = SharedPreprocessor(discretize=False).fit_transform(X)
        assert out.schema["cat_a"] == pl.Int8

    def test_transform_matches_fit_schema(self) -> None:
        train = pl.DataFrame({"x": [1.5, 2.5, 900.0], "cat": ["a", "b", "a"]})
        test = pl.DataFrame({"x": [3.5, 400.0], "cat": ["a", "a"]})
        pre = SharedPreprocessor(discretize=False).fit(train)
        out = pre.transform(test)
        assert out.columns == pre.columns_
        assert out["x"].to_list() == [3.5, 400.0]
