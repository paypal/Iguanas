"""Iguanas pipeline exposed as a :class:`benchmarks.protocol.NestedModel`.

Every stage the ablations toggle is a constructor flag, so ``ablations.py`` never
has to reach inside the pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from xgboost import XGBClassifier

from iguanas.metrics import compute_metrics, count_conditions
from iguanas.rule_combination import combine_rules_budgeted
from iguanas.rule_evaluation import apply_and_filter_by_performance, apply_rules
from iguanas.rule_generation import rule_grid_search
from iguanas.rule_selection import filter_correlated_rules
from iguanas.weight_transformations import generate_weights, select_uncorrelated_weights

from .config import ExperimentConfig, RuleGenerationConfig, SelectionConfig
from .operating_point import select_rule_by_alert_budget
from .ruleset_selection import NoRulesError, select_ruleset
from .search_wrappers import run_combiner
from .timing import Counters


@dataclass
class _Pool:
    rules: list[str]
    trees_fitted: int


class IguanasAdapter:
    """Generate on one split, select on another, score on a third."""

    def __init__(
        self,
        cfg: ExperimentConfig,
        seed: int,
        *,
        name: str = "iguanas",
        generation: RuleGenerationConfig | None = None,
        selection: SelectionConfig | None = None,
    ) -> None:
        self.name = name
        self.cfg = cfg
        self.seed = seed
        self.gen_cfg = generation or cfg.generation
        self.sel_cfg = selection or cfg.selection
        self._pool: _Pool = _Pool([], 0)
        self._chosen: str | None = None
        self._counters = Counters()
        self.selection_report: dict[str, Any] = {}

    # -- generation ------------------------------------------------------- #

    def _estimator(self) -> XGBClassifier:
        return XGBClassifier(
            n_estimators=self.gen_cfg.n_estimators,
            max_depth=self.gen_cfg.max_depth,
            learning_rate=self.gen_cfg.learning_rate,
            random_state=self.seed,
            n_jobs=self.gen_cfg.n_jobs,
            tree_method="hist",
            verbosity=0,
        )

    def _scale_pos_weights(self, y: np.ndarray, n: int) -> np.ndarray:
        pos = float(np.count_nonzero(y))
        neg = float(len(y) - pos)
        # n<=1 means no grid: the single value is the true balanced ratio
        # (class_weight='balanced' equivalent), not the log-range's start point.
        if n <= 1:
            return np.array([neg / pos if pos else 1.0])
        ratio = max(2.0, neg / pos) if pos else 2.0
        return np.logspace(0.0, math.log10(ratio), num=n)

    def _sample_weights(self, X: pl.DataFrame, y: np.ndarray) -> pl.DataFrame | None:
        """Build weight schedules from variance or generation-split importance."""
        if not self.gen_cfg.use_weight_steering or self.gen_cfg.n_weight_transformations < 1:
            return None
        if self.gen_cfg.weight_feature_mode not in {"variance", "top_importance"}:
            raise ValueError(
                "weight_feature_mode must be 'variance' or 'top_importance'"
            )

        if self.gen_cfg.weight_feature_mode == "variance":
            variances = {c: float(np.nanvar(X[c].to_numpy())) for c in X.columns}
            features = [max(sorted(variances), key=lambda c: variances[c])]
        else:
            model = self._estimator()
            positives = float(np.count_nonzero(y))
            negatives = float(len(y) - positives)
            if positives > 0.0:
                model.set_params(scale_pos_weight=max(1.0, negatives / positives))
            model.fit(X.to_pandas(), y.astype(int))
            ranked = sorted(
                zip(X.columns, model.feature_importances_, strict=True),
                key=lambda item: (-float(item[1]), item[0]),
            )
            features = [
                name for name, importance in ranked
                if float(importance) > 0.0
            ][: self.gen_cfg.n_weight_features]

        frames = []
        for feature in features:
            if float(np.nanvar(X[feature].to_numpy())) <= 0.0:
                continue
            frame = generate_weights(X[feature].cast(pl.Float64).abs())
            frames.append(frame)
        if not frames:
            return None

        candidates = pl.concat(
            [frames[0]] + [frame.drop("Baseline") for frame in frames[1:]],
            how="horizontal",
        )
        importance = {
            name: float(len(candidates.columns) - index)
            for index, name in enumerate(candidates.columns)
        }
        names, _ = select_uncorrelated_weights(
            candidates,
            importance,
            self.gen_cfg.n_weight_transformations,
        )
        keep = names[: self.gen_cfg.n_weight_transformations] or ["Baseline"]
        return candidates.select([c for c in keep if c in candidates.columns])

    def _random_path_pool(
        self,
        X: pl.DataFrame,
        y: np.ndarray,
        scales: np.ndarray,
        weights_df: pl.DataFrame | None,
    ) -> _Pool:
        """Ablation baseline: sample a root-to-leaf path uniformly instead of max gain.

        Sweeps the same (scale x weight) grid as :func:`rule_grid_search` so the
        two extraction strategies fit an identical number of trees.
        """
        rng = np.random.default_rng(self.seed)
        X_pd = X.to_pandas()
        weight_columns: list[pl.Series | None] = (
            [weights_df[c] for c in weights_df.columns] if weights_df is not None else [None]
        )
        rules: list[str] = []
        trees = 0
        for scale in scales:
            for weights in weight_columns:
                model = self._estimator()
                model.set_params(scale_pos_weight=float(scale))
                model.fit(
                    X_pd,
                    y.astype(int),
                    sample_weight=None if weights is None else weights.to_numpy(),
                )
                trees += self.gen_cfg.n_estimators
                frame = model._Booster.trees_to_dataframe()
                for _, tree in frame.groupby("Tree", sort=False):
                    rule = _random_path_rule(tree.reset_index(drop=True), rng)
                    if rule:
                        rules.append(rule)
        return _Pool(sorted(set(rules)), trees)

    def fit_generate(self, X: pl.DataFrame, y: np.ndarray) -> None:
        weights_df = self._sample_weights(X, y)
        n_weight_cols = weights_df.width if weights_df is not None else 1
        n_scales = self.gen_cfg.n_scale_pos_weights
        scales = self._scale_pos_weights(y, n_scales)

        if self.gen_cfg.max_gain_extraction:
            rules_df = rule_grid_search(
                self._estimator(),
                X,
                pl.Series(y.astype(int)),
                scale_pos_weights=scales,
                sample_weights_df=weights_df,
                n_jobs=self.gen_cfg.n_jobs,
                verbose=0,
            )
            rules = (
                sorted(set(rules_df["rule"].to_list())) if not rules_df.is_empty() else []
            )
            trees = n_scales * n_weight_cols * self.gen_cfg.n_estimators
            self._pool = _Pool(rules, trees)
        else:
            self._pool = self._random_path_pool(X, y, scales, weights_df)

        self._counters = Counters(
            trees_fitted=self._pool.trees_fitted,
            rules_generated=len(self._pool.rules),
        )
        if not self._pool.rules:
            raise NoRulesError("rule generation produced no rules")

    # -- selection -------------------------------------------------------- #

    def _thresholds(self) -> list[dict[str, Any]]:
        return [
            {"name": "precision", "operator": ">=", "value": self.sel_cfg.min_precision},
            {"name": "recall", "operator": ">=", "value": self.sel_cfg.min_recall},
        ]

    def _candidate_rules(
        self, X: pl.DataFrame, y_series: pl.Series
    ) -> tuple[pl.DataFrame, list[str]]:
        pool_rules = self._pool.rules
        if self.sel_cfg.max_conditions_per_rule is not None:
            pool_rules = [r for r in pool_rules if count_conditions(r) <= self.sel_cfg.max_conditions_per_rule]
        R, metrics = apply_and_filter_by_performance(
            X,
            y_series,
            pool_rules,
            metric_thresholds=self._thresholds(),
            ranking_metric=self.sel_cfg.shortlist_metric,
        )
        if metrics.is_empty():
            raise NoRulesError("no rule passed the selection-split thresholds")

        # Shortlist on the F-score so candidates are not degenerate, then order by
        # precision: at a fixed alert budget the purest rules are what fit inside it.
        ranked = metrics.head(self.sel_cfg.top_n_rules)
        order_metric = self.sel_cfg.order_metric
        if order_metric in ranked.columns:
            ranked = ranked.sort(order_metric, descending=True)
        rules = ranked["rule"].to_list()
        if self.sel_cfg.use_correlation_dedup and len(rules) > 1:
            importance_metric = (
                order_metric if order_metric in ranked.columns else self.cfg.metric
            )
            importance = dict(zip(rules, ranked[importance_metric].to_list(), strict=True))
            rules = filter_correlated_rules(
                R[rules], importance=importance, max_corr=self.sel_cfg.max_corr
            )
        return R[rules], rules

    def _fit_select_budgeted(
        self, X: pl.DataFrame, y_series: pl.Series, alert_rate: float
    ) -> None:
        """Same shared stage every baseline uses, so only generation differs."""
        chosen, report = select_ruleset(
            X, y_series, self._pool.rules, self.cfg, self.sel_cfg, alert_rate
        )
        self._chosen = chosen
        self.selection_report = report
        self._counters = self._counters.merge(
            Counters(rules_after_filter=int(report["n_candidate_rules"]))
        )

    def fit_select(self, X: pl.DataFrame, y: np.ndarray, alert_rate: float) -> None:
        y_series = pl.Series(y.astype(bool))
        if self.sel_cfg.use_budgeted_combination:
            self._fit_select_budgeted(X, y_series, alert_rate)
            return

        R, rules = self._candidate_rules(X, y_series)
        combination = run_combiner(
            self.sel_cfg.combiner,
            R,
            y_series,
            metric=self.cfg.metric,
            max_rules=self.cfg.max_rules,
            operator=self.sel_cfg.combine_operator,
            min_improvement=self.sel_cfg.min_improvement,
            return_top_k=self.sel_cfg.return_top_k,
            beam_width=self.sel_cfg.beam_width,
        )

        candidates = R
        if not combination.R.is_empty():
            extra = [c for c in combination.R.columns if c not in candidates.columns]
            if extra:
                candidates = pl.concat([candidates, combination.R.select(extra)], how="horizontal")

        metrics = compute_metrics(candidates, y_series)
        chosen = select_rule_by_alert_budget(
            metrics, alert_rate, tie_break_metric=self.sel_cfg.budget_objective
        )
        if chosen is None:
            raise NoRulesError("no candidate ruleset available after combination search")

        self._chosen = chosen
        self._counters = self._counters.merge(combination.counters).merge(
            Counters(rules_after_filter=len(rules))
        )
        self.selection_report = {
            "n_pool_rules": len(self._pool.rules),
            "n_candidate_rules": len(rules),
            "n_combined_candidates": int(candidates.width),
            "combiner": self.sel_cfg.combiner,
            "chosen_rule": chosen,
        }

    # -- scoring ---------------------------------------------------------- #

    def score(self, X: pl.DataFrame) -> np.ndarray:
        if self._chosen is None:
            raise NoRulesError("adapter has not completed its selection stage")
        fired = apply_rules(X, [self._chosen])[self._chosen]
        return fired.cast(pl.Float64).to_numpy()

    def complexity(self) -> int:
        return count_conditions(self._chosen) if self._chosen else 0

    def counters(self) -> Counters:
        return self._counters


def _random_path_rule(tree: pd.DataFrame, rng: np.random.Generator) -> str:
    """Walk one XGBoost tree from root to a leaf, choosing branches uniformly."""
    if tree.empty:
        return ""
    by_id = tree.set_index("ID")
    current = str(tree.iloc[0]["ID"])
    conditions: list[str] = []
    for _ in range(len(tree)):
        row = by_id.loc[current]
        if row["Feature"] == "Leaf":
            break
        go_left = bool(rng.integers(2))
        operator, child = ("<", row["Yes"]) if go_left else (">=", row["No"])
        conditions.append(f'(X["{row["Feature"]}"] {operator} {round(float(row["Split"]), 5)})')
        if not isinstance(child, str):
            break
        current = child
    return " & ".join(conditions)
