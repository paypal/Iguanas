from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, PrivateAttr, field_validator
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from xgboost import XGBClassifier

from .rule_evaluation import apply_and_filter_by_performance, apply_rules
from .rule_generation import rule_grid_search_parallel_scales

_NUMERIC_DTYPES = (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64)


class RuleClassifier(BaseModel, BaseEstimator, ClassifierMixin):
    """Rule-based classifier that selects the single best rule.

    Generates candidate rules, filters by minimum precision and recall,
    then keeps only the top rule ranked by opt_metric.

    Parameters
    ----------
    estimator : XGBClassifier
        XGBoost classifier used for rule generation.
    scale_pos_weight_vec : np.ndarray
        Vector of scale_pos_weight values swept during rule generation.
    opt_metric : str, default="accuracy"
        Metric used to rank candidate rules. The single highest-scoring rule
        is kept. Must be a column produced by compute_metrics.
    metrics_threshold : list[dict[str, Any]] | None, default=None
        List of threshold dicts used to filter candidate rules. Each dict must
        have keys ``"name"`` (metric column), ``"operator"`` (one of
        ``">="``, ``">"``, ``"<="``, ``"<"``, ``"=="``, ``"!="``), and
        ``"value"`` (numeric threshold). All conditions are combined with AND.
        If None, defaults to ``[precision >= 0.2, recall >= 0.2]``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    estimator: XGBClassifier
    scale_pos_weight_vec: np.ndarray
    opt_metric: str = "accuracy"
    metrics_threshold: list[dict[str, Any]] | None = None

    @field_validator("metrics_threshold")
    @classmethod
    def _check_metrics_threshold(cls, v: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        if v is None:
            return v
        for t in v:
            val = t.get("value")
            if val is not None and not (0.0 <= val <= 1.0):
                raise ValueError(
                    f"metrics_threshold value {val!r} is out of range [0, 1] for threshold {t!r}"
                )
        return v

    # Learned attributes (set by fit, not part of the model schema)
    _feature_cols_: list[str] = PrivateAttr(default_factory=list)
    _best_rule_: str = PrivateAttr(default="")

    def fit(self, X: pl.DataFrame, y: pl.Series) -> RuleClassifier:
        """Generate, filter, and select the single best rule from training data.

        Parameters
        ----------
        X : pl.DataFrame
            Feature DataFrame. Only numeric columns are used for rule generation.
        y : pl.Series
            Binary target series.

        Returns
        -------
        RuleClassifier
            Fitted classifier instance (self).
        """
        self._feature_cols_ = [c for c, dt in X.schema.items() if dt in _NUMERIC_DTYPES]

        rules = rule_grid_search_parallel_scales(
            self.estimator,
            X[self._feature_cols_].to_pandas(),
            y.to_pandas(),
            scale_pos_weight_vec=self.scale_pos_weight_vec,
        ).unique("rule")

        _, M = apply_and_filter_by_performance(
            X[self._feature_cols_], y, rules["rule"].to_list(),
            metrics_threshold=self.metrics_threshold,
            sort_by=self.opt_metric,
        )

        self._best_rule_ = M["rule"].item(0) if M.height > 0 else ""
        return self

    def _check_is_fitted(self) -> None:
        if not self._feature_cols_:
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )

    def predict(self, X: pl.DataFrame) -> pl.Series:
        """Predict binary labels using the single best rule.

        Parameters
        ----------
        X : pl.DataFrame
            Feature DataFrame with the same columns seen during fit.

        Returns
        -------
        pl.Series
            Boolean series named "prediction".
        """
        self._check_is_fitted()
        if not self._best_rule_:
            return pl.Series(self._best_rule_, [False] * X.height, dtype=pl.Boolean)

        R = apply_rules(X[self._feature_cols_], [self._best_rule_])
        return R[self._best_rule_]

    def predict_proba(self, X: pl.DataFrame) -> pl.Series:
        """Predict probability using the single best rule.

        - Rule fires  → 1.0
        - Rule does not fire → 0.0

        Parameters
        ----------
        X : pl.DataFrame
            Feature DataFrame with the same columns seen during fit.

        Returns
        -------
        pl.Series
            Float64 series named "proba" with values in {0.0, 1.0}.
        """
        return self.predict(X).cast(pl.Float64)

    def fit_predict(self, X: pl.DataFrame, y: pl.Series) -> pl.Series:
        """Fit classifier and return binary predictions on the same data."""
        return self.fit(X, y).predict(X)
