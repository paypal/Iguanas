"""Shared preprocessing applied identically to every model.

Iguanas performs no imputation or encoding of its own -- in the PayPal stack that
is Gators' job -- and neither do most of the rule-learning baselines. Leaving
each model to cope with raw columns would mean comparing preprocessing choices as
much as rule learners, so one Gators pipeline runs ahead of every model:

1. :class:`gators.imputers.NumericImputer` (``strategy="mean"``)
2. :class:`gators.imputers.StringImputer` for categorical nulls
3. :class:`gators.discretizers.QuantileDiscretizer` (optional)
4. one-hot encoding

The discretizer emits readable interval labels such as ``(-inf,3.0]``, so the
one-hot columns a rule learner splits on stay legible.

.. note::

   The final step uses Polars' native ``to_dummies`` rather than
   :class:`gators.encoders.OneHotEncoder`, because gators 1.3.0 calls
   ``pl.concat(how="horizontal_extend")``, a mode current Polars rejects. Swap
   it back once that is fixed upstream.

The pipeline is **fitted on the training data only** -- the generate and select
splits together -- and then applied to test. Imputation values, bin edges and
category levels are computed without labels, so the generate/select boundary
(which exists to stop rule *selection* seeing the data the rules were induced on)
does not apply here; only the held-out test split must be excluded. Fitting on
everything, as the loader used to when it median-filled before any split existed,
leaks test-set statistics into the fill values.

.. note::

   Discretising every numeric column changes what the *baselines* can express:
   gradient-boosted trees and CART lose continuous split points, and Iguanas
   emits ``X["age__(30.0,50.0]"] == 1`` rather than ``X["age"] > 30``. It is the
   right choice when the comparison must include learners that require binary
   features (BRL, CORELS), and it removes BRL's MDLP discretisation cost
   entirely, but it is a protocol decision worth reporting. Set
   ``discretize=False`` to keep continuous features and measure the difference.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import polars as pl

MISSING_LEVEL = "__MISSING__"

# QuantileDiscretizer labels bins as "(-inf,4.0]", "(4.0,6.0]", "(6.0,inf)".
# XGBoost rejects feature names containing '[', ']' or '<', so the interval
# notation is rewritten rather than stripped, to keep the bins legible.
_INTERVAL = re.compile(r"^(?P<col>.+)_\((?P<lo>[^,]+),(?P<hi>[^\]\)]+)[\]\)]$")
_UNSAFE = re.compile(r"[\[\]<>(),]")


def sanitise_column(name: str) -> str:
    """Rewrite a bin label into a name tree libraries accept.

    >>> sanitise_column("age_(-inf,30.0]")
    'age__le_30.0'
    >>> sanitise_column("age_(30.0,50.0]")
    'age__30.0_to_50.0'
    """
    match = _INTERVAL.match(name)
    if match is None:
        return _UNSAFE.sub("_", name)
    col, lo, hi = match.group("col"), match.group("lo"), match.group("hi")
    if lo == "-inf":
        return f"{col}__le_{hi}"
    if hi == "inf":
        return f"{col}__gt_{lo}"
    return f"{col}__{lo}_to_{hi}"


def _sanitise_frame(frame: pl.DataFrame) -> pl.DataFrame:
    renamed: dict[str, str] = {}
    seen: set[str] = set()
    for original in frame.columns:
        candidate = sanitise_column(original)
        suffix = 2
        while candidate in seen:
            candidate = f"{sanitise_column(original)}_{suffix}"
            suffix += 1
        seen.add(candidate)
        renamed[original] = candidate
    return frame.rename(renamed)


@dataclass
class SharedPreprocessor:
    """Mean-impute, optionally quantile-discretize, then one-hot encode.

    Parameters
    ----------
    discretize : bool, default=True
        Quantile-bin numeric columns before encoding.
    num_bins : int, default=8
        Quantile bins per numeric column.
    min_count : int, default=1
        Minimum occurrences for a category to get its own one-hot column.
    """

    discretize: bool = True
    num_bins: int = 8
    min_count: int = 1
    steps_: list[Any] = field(default_factory=list)
    columns_: list[str] = field(default_factory=list)
    dtypes_: dict[str, Any] = field(default_factory=dict)
    fitted_: bool = False

    def _build(self, X: pl.DataFrame) -> list[Any]:
        from gators.discretizers import QuantileDiscretizer
        from gators.imputers import NumericImputer, StringImputer

        numeric = [c for c in X.columns if X.schema[c].is_numeric()]
        categorical = [c for c in X.columns if c not in numeric]

        steps: list[Any] = []
        if numeric:
            steps.append(NumericImputer(strategy="mean", subset=numeric))
        if categorical:
            steps.append(
                StringImputer(strategy="constant", value=MISSING_LEVEL, subset=categorical)
            )
        if self.discretize and numeric:
            steps.append(QuantileDiscretizer(num_bins=self.num_bins, subset=numeric))
        return steps

    @staticmethod
    def _one_hot(frame: pl.DataFrame) -> pl.DataFrame:
        to_encode = [c for c in frame.columns if not frame.schema[c].is_numeric()]
        if to_encode:
            frame = frame.to_dummies(columns=to_encode)
        return _sanitise_frame(frame)

    def fit(self, X: pl.DataFrame, y: pl.Series | None = None) -> SharedPreprocessor:
        self.steps_ = self._build(X)
        frame = X.clone()
        for step in self.steps_:
            frame = step.fit_transform(frame, y)
        encoded = self._one_hot(frame)
        self.columns_ = list(encoded.columns)
        # Dummy columns are cast to Int8 to keep the one-hot block compact, but
        # continuous features must keep their original dtype: blanket-casting
        # the whole frame truncates them to integers (and overflows outright on
        # counts above 127), silently destroying every split point a rule
        # learner could find.
        self.dtypes_ = {
            c: (pl.Int8 if dt in (pl.Boolean, pl.UInt8) else dt)
            for c, dt in encoded.schema.items()
        }
        self.fitted_ = True
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        if not self.fitted_:
            raise RuntimeError("SharedPreprocessor.transform called before fit")
        frame = X.clone()
        for step in self.steps_:
            frame = step.transform(frame)
        frame = self._one_hot(frame)
        # A level absent from this split would otherwise drop its column and
        # silently change the feature space between train and test.
        missing = [c for c in self.columns_ if c not in frame.columns]
        if missing:
            frame = frame.with_columns([pl.lit(0).alias(c) for c in missing])
        return frame.select(self.columns_).cast(self.dtypes_)

    def fit_transform(self, X: pl.DataFrame, y: pl.Series | None = None) -> pl.DataFrame:
        return self.fit(X, y).transform(X)
