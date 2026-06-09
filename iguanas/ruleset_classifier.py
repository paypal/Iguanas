from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from xgboost import XGBClassifier

from .rule_combination import combine_rules_greedy
from .rule_evaluation import apply_and_filter_by_performance, apply_rules
from .rule_generation import rule_grid_search_parallel_scales
from .rule_selection import filter_correlated_rules

_NUMERIC_DTYPES = (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64)


class RulesetClassifier(BaseModel, BaseEstimator, ClassifierMixin):
    """End-to-end rule-based classification pipeline.

    The best ruleset is selected through the following steps:

    1. **Rule generation**: candidate rules are extracted from XGBoost decision
       trees trained across a sweep of ``scale_pos_weight`` values.
    2. **Performance filtering**: rules that fail any condition in
       ``metric_thresholds`` are discarded.
    3. **Correlation filtering**: among rules that are correlated above
       ``max_corr``, only the one with the highest ``opt_metric`` score is kept.
    4. **Greedy combination**: starting from the single best rule, rules are
       added one at a time — each iteration picks the candidate that yields the
       largest improvement in ``opt_metric`` when combined (via
       ``combine_operator``) with the already-selected rules. Addition stops
       when no candidate improves the metric by at least ``min_improvement`` or
       when ``max_rules`` rules have been selected.

    The resulting combined rule expression is stored in ``_best_ruleset_`` as a
    string (e.g. ``"(rule_A) | (rule_B) | (rule_C)"``).

    Parameters
    ----------
    estimator : XGBClassifier
        XGBoost classifier used for rule generation.
    scale_pos_weight_vec : np.ndarray
        Vector of scale_pos_weight values swept during rule generation.
    opt_metric : str, default="accuracy"
        Metric used to rank and select candidate rules. Must be a column
        produced by compute_metrics (e.g. "f1", "precision", "recall").
    max_rules : int, default=10
        Maximum number of rules the greedy search may select. Must be > 0.
    metric_thresholds : list[dict[str, Any]] | None, default=None
        List of threshold dicts used to filter candidate rules. Each dict must
        have keys ``"name"`` (metric column), ``"operator"`` (one of
        ``">="``, ``">"``, ``"<="``, ``"<"``, ``"=="``, ``"!="``), and
        ``"value"`` (numeric threshold). All conditions are combined with AND.
        If None, the default threshold of ``apply_and_filter_by_performance``
        is used.
    max_corr : float, default=0.8
        Maximum pairwise correlation allowed between rules; correlated pairs
        are pruned to keep only the highest-ranked one. Must be in [0, 1].
    combine_operator : str, default="or"
        Boolean operator used to combine selected rules: "or" or "and".
    min_improvement : float, default=0.01
        Minimum improvement in opt_metric required to add a new rule to the
        combined ruleset during greedy selection.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    estimator: XGBClassifier
    scale_pos_weight_vec: np.ndarray
    opt_metric: str = "accuracy"
    max_rules: int = Field(default=10, gt=0)
    metric_thresholds: list[dict[str, Any]] | None = None
    metric_weights: pl.Series | None = None

    @field_validator("metric_thresholds")
    @classmethod
    def _check_metric_thresholds(
        cls, v: list[dict[str, Any]] | None
    ) -> list[dict[str, Any]] | None:
        if v is None:
            return v
        for t in v:
            val = t.get("value")
            if val is not None and not (0.0 <= val <= 1.0):
                raise ValueError(
                    f"metric_thresholds value {val!r} is out of range [0, 1] for threshold {t!r}"
                )
        return v

    max_corr: float = Field(default=0.8, ge=0.0, le=1.0)
    combine_operator: str = "or"
    min_improvement: float = Field(default=0.01, ge=0.0, le=1.0)

    # Learned attributes (set by fit, not part of the model schema)
    _feature_cols_: list[str] = PrivateAttr(default_factory=list)
    _best_ruleset_: str = PrivateAttr(default="")

    @field_validator("combine_operator")
    @classmethod
    def _check_operator(cls, v: str) -> str:
        if v not in ("or", "and"):
            raise ValueError(f"combine_operator must be 'or' or 'and', got '{v}'")
        return v

    def fit(
        self, X: pl.DataFrame, y: pl.Series, sample_weights: pl.DataFrame | None = None
    ) -> RulesetClassifier:
        """Generate, filter, and select rules from training data.

        Parameters
        ----------
        X : pl.DataFrame
            Feature DataFrame. Only numeric columns are used for rule generation.
        y : pl.Series
            Binary target series.

        Returns
        -------
        RulesetClassifier
            Fitted pipeline instance (self).
        """
        if self.metric_thresholds is None:
            self.metric_thresholds = [{"name": "accuracy", "operator": ">=", "value": 0.5}]
        self._feature_cols_ = [c for c, dt in X.schema.items() if dt in _NUMERIC_DTYPES]
        rules = rule_grid_search_parallel_scales(
            self.estimator,
            X[self._feature_cols_],
            y,
            scale_pos_weight_vec=self.scale_pos_weight_vec,
            weights_train_vec=sample_weights,
        )
        R, M = apply_and_filter_by_performance(
            X[self._feature_cols_],
            y,
            rules["rule"].to_list(),
            metric_thresholds=self.metric_thresholds,
            sort_by=self.opt_metric,
        )
        candidate_rules = M["rule"].to_list()
        importance = dict(zip(M["rule"], M[self.opt_metric], strict=False))
        candidate_rules = filter_correlated_rules(
            R[candidate_rules],
            importance=importance,
            max_corr=self.max_corr,
        )
        R_greedy = combine_rules_greedy(
            R[candidate_rules],
            y,
            metric=self.opt_metric,
            max_rules=self.max_rules,
            min_improvement=self.min_improvement,
            weights=self.metric_weights,
        )
        self._best_ruleset_ = R_greedy.columns[0]
        return self

    def _check_is_fitted(self) -> None:
        if not self._feature_cols_:
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )

    def predict(self, X: pl.DataFrame) -> pl.Series:
        """Predict binary labels for each sample.

        A sample is positive if any (OR) or all (AND) selected rules fire,
        depending on combine_operator.

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
        combined_name = self._best_ruleset_
        if not self._best_ruleset_:
            return pl.Series(combined_name, [False] * X.height, dtype=pl.Boolean)

        R = apply_rules(X[self._feature_cols_], [self._best_ruleset_])
        return R[self._best_ruleset_]

    def predict_proba(self, X: pl.DataFrame) -> pl.Series:
        """Predict rule-coverage probability for each sample.

        Probability is a piecewise-linear function of the number of selected
        rules that fire for each sample:

        - 0 rules fired  → 0.0
        - 1 rule fired   → 0.5
        - all rules fired → 1.0
        - between 1 and all: linearly interpolated in [0.5, 1.0]

        Parameters
        ----------
        X : pl.DataFrame
            Feature DataFrame with the same columns seen during fit.

        Returns
        -------
        pl.Series
            Float64 series named "proba" with values in [0.0, 1.0].
        """
        self._check_is_fitted()
        combined_name = self._best_ruleset_

        if not self._best_ruleset_:
            return pl.Series(combined_name, [0.0] * X.height, dtype=pl.Float64)

        R = apply_rules(X[self._feature_cols_], [self._best_ruleset_])
        proba_expr = pl.col(self._best_ruleset_).cast(pl.Float64).alias(combined_name)
        return R.select(proba_expr).to_series()

    def fit_predict(self, X: pl.DataFrame, y: pl.Series) -> pl.Series:
        """Fit pipeline and return binary predictions on the same data."""
        return self.fit(X, y).predict(X)
