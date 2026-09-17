"""Baseline adapters with a uniform interface and graceful degradation.

Complexity is always reported as a **number of conditions** (atomic feature
tests), never a number of rules: a 5-rule model with 6 conditions each is not
simpler than a 12-rule model with 1 condition each, and rule counts are the
usual way that comparison gets fudged.

Every optional third-party import is probed through :func:`_try_import`, which
records an "unavailable" reason instead of raising, so a missing package removes
one row from the results table rather than killing the run.
"""

from __future__ import annotations

import importlib
import re
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
import polars as pl
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from iguanas.metrics import count_conditions
from iguanas.rule_evaluation import apply_rules

from .config import ExperimentConfig
from .rule_extraction import (
    rules_from_brl,
    rules_from_figs,
    rules_from_ripper,
    rules_from_rulefit,
    rules_from_skope,
    rules_from_sklearn_tree,
    rules_from_xgboost,
)
from .ruleset_selection import NoRulesError, select_ruleset
from .timing import Counters

_CONDITION_TOKEN = re.compile(r"[<>=]=?|\bin\b")


@dataclass(frozen=True)
class Unavailable:
    """Why a baseline could not be constructed."""

    name: str
    reason: str


def _try_import(module: str) -> tuple[Any | None, str]:
    try:
        return importlib.import_module(module), ""
    except ImportError as exc:
        return None, f"import {module} failed: {exc}"
    except (RuntimeError, OSError) as exc:  # broken native extension, missing lib
        return None, f"import {module} raised {type(exc).__name__}: {exc}"


def _to_pandas(X: pl.DataFrame) -> pd.DataFrame:
    return X.to_pandas()


def _scores_from(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X))
        return proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else proba.ravel()
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype="float64").ravel()
    return np.asarray(model.predict(X), dtype="float64").ravel()


def _count_conditions_in_text(text: str) -> int:
    return len(_CONDITION_TOKEN.findall(text))


def sklearn_tree_conditions(tree: Any) -> int:
    """Total conditions across all root-to-leaf paths of a fitted sklearn tree."""
    left, right = tree.children_left, tree.children_right
    total = 0
    stack: list[tuple[int, int]] = [(0, 0)]
    while stack:
        node, depth = stack.pop()
        if left[node] == -1:
            total += depth
            continue
        stack.append((left[node], depth + 1))
        stack.append((right[node], depth + 1))
    return total


def _xgb_conditions(booster_frame: pd.DataFrame) -> int:
    total = 0
    for _, tree in booster_frame.groupby("Tree", sort=False):
        parents = {}
        for row in tree.itertuples():
            for child in (row.Yes, row.No):
                if isinstance(child, str):
                    parents[child] = row.ID
        leaves = tree.loc[tree["Feature"] == "Leaf", "ID"].tolist()
        for leaf in leaves:
            depth, node = 0, leaf
            while node in parents:
                node = parents[node]
                depth += 1
            total += depth
    return total


class _BaseBaseline:
    """Common plumbing: fit on generate, no-op select, score on anything."""

    name = "baseline"

    def __init__(self, cfg: ExperimentConfig, seed: int) -> None:
        self.cfg = cfg
        self.seed = seed
        self._counters = Counters()

    def fit_generate(self, X: pl.DataFrame, y: np.ndarray) -> None:
        raise NotImplementedError

    def fit_select(self, X: pl.DataFrame, y: np.ndarray, alert_rate: float) -> None:
        return None

    def score(self, X: pl.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def complexity(self) -> int:
        raise NotImplementedError

    def counters(self) -> Counters:
        return self._counters


class RuleSetMixin:
    """Evaluate a learner as a disjunction of the rules it generated.

    Left to themselves the baselines disagree about what a rule set *is*:
    RuleFit weights rules with signed coefficients and adds linear terms, BRL
    reads an ordered list top-down, FIGS and gradient boosting sum leaf values.
    Those scorers can be thresholded anywhere, so they always fill an alert
    budget exactly, while a genuine disjunction can only occupy the coverage
    levels its rules happen to produce -- which flatters the scorers and
    measures the wrong thing.

    Subclasses supply the generated rules; everything after that is the shared
    pipeline in :func:`benchmarks.ruleset_selection.select_ruleset`.
    """

    name = "ruleset"
    cfg: ExperimentConfig
    _counters: Counters

    def _extract_rules(self) -> list[str]:
        raise NotImplementedError

    def _rule_frame(self, X: pl.DataFrame) -> pl.DataFrame:
        """Frame the rules refer to; overridden where a model bins its own input."""
        return X

    def _remember_rules(self, X: pl.DataFrame) -> None:
        self._rules = self._extract_rules()
        self._chosen: str | None = None
        self.selection_report: dict[str, Any] = {}
        self._counters = self._counters.merge(Counters(rules_generated=len(self._rules)))

    def fit_select(self, X: pl.DataFrame, y: np.ndarray, alert_rate: float) -> None:
        frame = self._rule_frame(X)
        chosen, report = select_ruleset(
            frame,
            pl.Series(y.astype(bool)),
            getattr(self, "_rules", []),
            self.cfg,
            self.cfg.selection,
            alert_rate,
        )
        self._chosen = chosen
        self.selection_report = report
        self._counters = self._counters.merge(
            Counters(rules_after_filter=int(report["n_candidate_rules"]))
        )

    def score(self, X: pl.DataFrame) -> np.ndarray:
        if not self._chosen:
            raise NoRulesError(f"{self.name} has not completed its selection stage")
        frame = self._rule_frame(X)
        fired = apply_rules(frame, [self._chosen])[self._chosen]
        return fired.cast(pl.Float64).to_numpy()

    def complexity(self) -> int:
        return count_conditions(self._chosen) if self._chosen else 0


class DecisionTreeBaseline(RuleSetMixin, _BaseBaseline):
    """CART pruned along its cost-complexity path, tuned on the selection split.

    ``target_conditions`` makes the comparison complexity-matched: the pruning
    level whose condition count is closest to the target is chosen rather than
    the one with the best selection metric.
    """

    name = "decision_tree"

    def __init__(
        self,
        cfg: ExperimentConfig,
        seed: int,
        *,
        target_conditions: int | None = None,
        max_depth: int = 4,
    ) -> None:
        super().__init__(cfg, seed)
        self.target_conditions = target_conditions
        self.max_depth = max_depth
        self._candidates: list[DecisionTreeClassifier] = []
        self._model: DecisionTreeClassifier | None = None

    def _new_tree(self, ccp_alpha: float) -> DecisionTreeClassifier:
        return DecisionTreeClassifier(
            max_depth=self.max_depth,
            random_state=self.seed,
            class_weight="balanced",
            ccp_alpha=ccp_alpha,
        )

    def fit_generate(self, X: pl.DataFrame, y: np.ndarray) -> None:
        X_pd, y_int = _to_pandas(X), y.astype(int)
        path = self._new_tree(0.0).cost_complexity_pruning_path(X_pd, y_int)
        alphas = np.unique(np.clip(path.ccp_alphas, 0.0, None))
        if alphas.size > 12:
            alphas = np.quantile(alphas, np.linspace(0.0, 1.0, 12))
        self._candidates = []
        for alpha in alphas:
            tree = self._new_tree(float(alpha)).fit(X_pd, y_int)
            if tree.tree_.node_count > 1:
                self._candidates.append(tree)
        if not self._candidates:
            self._candidates = [self._new_tree(0.0).fit(X_pd, y_int)]
        self._counters = Counters(trees_fitted=len(self._candidates) + 1)
        # The unpruned tree yields the richest pool; pruning is a selection
        # device for a scorer, and the shared budgeted stage plays that role now.
        self._model = max(self._candidates, key=lambda t: t.tree_.node_count)
        self._feature_names = list(X.columns)
        self._remember_rules(X)

    def _extract_rules(self) -> list[str]:
        if self._model is None:
            return []
        return rules_from_sklearn_tree(self._model, self._feature_names)


class GBMCeilingBaseline(RuleSetMixin, _BaseBaseline):
    """Boosted-tree paths: rule generation by boosting, with no steering.

    Previously reported as a continuous accuracy ceiling. Under disjunction
    semantics it is instead a rule *generator* -- every root-to-leaf path whose
    leaf favours the positive class -- so it stays comparable with the others
    rather than being scored by a mechanism none of them have.
    """

    name = "gbm_ceiling"

    def __init__(
        self, cfg: ExperimentConfig, seed: int, *, n_estimators: int = 200
    ) -> None:
        super().__init__(cfg, seed)
        self.n_estimators = n_estimators
        self._model: XGBClassifier | None = None
        self._feature_names: list[str] = []

    def fit_generate(self, X: pl.DataFrame, y: np.ndarray) -> None:
        pos = float(np.count_nonzero(y))
        neg = float(len(y) - pos)
        model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=4,
            learning_rate=0.1,
            random_state=self.seed,
            n_jobs=self.cfg.generation.n_jobs,
            tree_method="hist",
            scale_pos_weight=(neg / pos) if pos else 1.0,
            verbosity=0,
        )
        model.fit(_to_pandas(X), y.astype(int))
        self._model = model
        self._feature_names = list(X.columns)
        self._counters = Counters(trees_fitted=self.n_estimators)
        self._remember_rules(X)

    def _extract_rules(self) -> list[str]:
        if self._model is None:
            return []
        return rules_from_xgboost(self._model._Booster.trees_to_dataframe())


class _ImodelsBaseline(RuleSetMixin, _BaseBaseline):
    """Shared wrapper for the ``imodels`` rule learners.

    ``class_names`` lists the estimator names that have carried this algorithm
    across imodels releases; the first one present is used.
    """

    module = "imodels"
    class_names: tuple[str, ...] = ()
    needs_discretization = False
    num_bins = 8
    default_kwargs: dict[str, Any] = {}

    def __init__(self, cfg: ExperimentConfig, seed: int, **kwargs: Any) -> None:
        super().__init__(cfg, seed)
        self.kwargs = {**self.default_kwargs, **kwargs}
        self._model: Any = None
        self._discretizer: Any = None
        self._discrete_columns: list[str] = []

    @classmethod
    def _resolve_class(cls) -> tuple[Any | None, str]:
        module, reason = _try_import(cls.module)
        if module is None:
            return None, reason
        for name in cls.class_names:
            estimator_cls = getattr(module, name, None)
            if estimator_cls is not None:
                return estimator_cls, ""
        return None, f"{cls.module} exposes none of {cls.class_names}"

    @classmethod
    def availability(cls) -> str:
        return cls._resolve_class()[1]

    def _build(self) -> Any:
        estimator_cls, reason = self._resolve_class()
        if estimator_cls is None:
            raise ImportError(reason)
        params = dict(self.kwargs)
        try:
            accepted = estimator_cls().get_params()
        except (TypeError, ValueError):
            accepted = {}
        if "random_state" in accepted:
            params.setdefault("random_state", self.seed)
        return estimator_cls(**{k: v for k, v in params.items() if not accepted or k in accepted})

    def _fit_discretizer(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """Quantile-bin then one-hot, for the learners that need binary features.

        Only BRL and CORELS require this; every other model sees the continuous
        columns. imodels' own ``BRLDiscretizer`` is avoided because its
        Fayyad-Irani MDLP implementation costs roughly 8.8 ms per row -- around
        450s on a 51k-row split, ten times the cost of BRL's own MCMC -- which
        makes the baseline unfittable on the larger datasets rather than merely
        slow.
        """
        from benchmarks.preprocessing import SharedPreprocessor

        self._discretizer = SharedPreprocessor(
            discretize=True, num_bins=self.num_bins
        ).fit(pl.from_pandas(X))
        transformed = self._discretizer.transform(pl.from_pandas(X)).to_pandas()
        self._discrete_columns = list(transformed.columns)
        return transformed

    def _prepare(self, X: pl.DataFrame) -> tuple[Any, list[str]]:
        X_pd = _to_pandas(X)
        if not self.needs_discretization:
            return X_pd.to_numpy(), list(X_pd.columns)
        if self._discretizer is None:
            raise RuntimeError(f"{self.name} discretizer is not fitted")
        transformed = self._discretizer.transform(pl.from_pandas(X_pd)).to_pandas()
        aligned = transformed.reindex(columns=self._discrete_columns, fill_value=False)
        return aligned, list(aligned.columns)

    def fit_generate(self, X: pl.DataFrame, y: np.ndarray) -> None:
        X_pd = _to_pandas(X)
        if self.needs_discretization:
            data = self._fit_discretizer(X_pd, y)
            names = list(data.columns)
        else:
            data, names = X_pd.to_numpy(), list(X_pd.columns)
        model = self._build()
        model.fit(data, y.astype(int), feature_names=names)
        self._model = model
        self._feature_names = names
        self._remember_rules(X)

    def _rule_frame(self, X: pl.DataFrame) -> pl.DataFrame:
        if not self.needs_discretization:
            return X
        if self._discretizer is None:
            raise RuntimeError(f"{self.name} discretizer is not fitted")
        return self._discretizer.transform(X)


def _imodels_conditions(model: Any) -> int:
    """Count conditions across the several shapes imodels uses for rule storage."""
    if model is None:
        return 0
    rules = getattr(model, "rules_", None)
    if isinstance(rules, pd.DataFrame) and "rule" in rules.columns:
        active = rules
        if "coef" in rules.columns:
            active = rules.loc[rules["coef"].astype(float) != 0.0]
        return int(sum(_count_conditions_in_text(str(r)) for r in active["rule"]))
    if isinstance(rules, list) and rules:
        total = 0
        for rule in rules:
            if isinstance(rule, tuple) and rule:
                total += _count_conditions_in_text(str(rule[0]))
            elif isinstance(rule, dict):
                total += len(rule.get("feature", []) or []) or _count_conditions_in_text(str(rule))
            else:
                total += _count_conditions_in_text(str(rule))
        return total
    for attr in ("complexity_", "complexity"):
        value = getattr(model, attr, None)
        if isinstance(value, (int, float)):
            return int(value)
    return _count_conditions_in_text(str(model))


class RuleFitBaseline(_ImodelsBaseline):
    name = "rulefit"
    class_names = ("RuleFitClassifier",)

    def _extract_rules(self) -> list[str]:
        return rules_from_rulefit(self._model, set(self._feature_names))


class SkopeRulesBaseline(_ImodelsBaseline):
    name = "skope_rules"
    class_names = ("SkopeRulesClassifier",)

    def _extract_rules(self) -> list[str]:
        return rules_from_skope(self._model, set(self._feature_names))


class BRLBaseline(_ImodelsBaseline):
    name = "brl"
    class_names = ("BayesianRuleListClassifier",)
    needs_discretization = True
    # imodels defaults run 3 chains x 50k MCMC iterations, which dominates a run;
    # these are the smallest settings that still produce a non-trivial list.
    default_kwargs = {"n_chains": 1, "max_iter": 1_000, "listlengthprior": 3}

    def _extract_rules(self) -> list[str]:
        return rules_from_brl(self._model, set(self._feature_names))


class FIGSBaseline(_ImodelsBaseline):
    name = "figs"
    class_names = ("FIGSClassifier",)
    default_kwargs = {"max_depth": 4}

    def _extract_rules(self) -> list[str]:
        return rules_from_figs(self._model, list(self._feature_names))


class CorelsBaseline(_ImodelsBaseline):
    """CORELS certifiably optimal rule lists; needs the optional ``corels`` wheel."""

    name = "corels"
    class_names = ("OptimalRuleListClassifier", "CorelsRuleListClassifier")
    needs_discretization = True

    def _extract_rules(self) -> list[str]:
        return rules_from_brl(self._model, set(self._feature_names))


class RipperBaseline(RuleSetMixin, _BaseBaseline):
    """RIPPER via ``wittgenstein``; already a DNF rule set."""

    name = "ripper"

    def __init__(self, cfg: ExperimentConfig, seed: int) -> None:
        super().__init__(cfg, seed)
        self._model: Any = None
        self._feature_names: list[str] = []

    @classmethod
    def availability(cls) -> str:
        module, reason = _try_import("wittgenstein")
        return reason if module is None else ""

    def fit_generate(self, X: pl.DataFrame, y: np.ndarray) -> None:
        import wittgenstein as lw

        model = lw.RIPPER(random_state=self.seed, max_rule_conds=4)
        model.fit(_to_pandas(X), pd.Series(y.astype(int), name="target"))
        self._model = model
        self._feature_names = list(X.columns)
        self._remember_rules(X)

    def _extract_rules(self) -> list[str]:
        return rules_from_ripper(self._model, set(self._feature_names))


BaselineFactory = Callable[[ExperimentConfig, int], Any]

BASELINE_REGISTRY: dict[str, BaselineFactory] = {
    "decision_tree": DecisionTreeBaseline,
    "gbm_ceiling": GBMCeilingBaseline,
    "rulefit": RuleFitBaseline,
    "skope_rules": SkopeRulesBaseline,
    "brl": BRLBaseline,
    "figs": FIGSBaseline,
    "corels": CorelsBaseline,
    "ripper": RipperBaseline,
}


def baseline_availability(name: str) -> str:
    """Return an empty string if the baseline can be built, else the reason."""
    if name not in BASELINE_REGISTRY:
        return f"unknown baseline {name!r}"
    factory = BASELINE_REGISTRY[name]
    checker = getattr(factory, "availability", None)
    return checker() if callable(checker) else ""


def make_baseline(
    name: str, cfg: ExperimentConfig, seed: int, **kwargs: Any
) -> Any | Unavailable:
    reason = baseline_availability(name)
    if reason:
        return Unavailable(name, reason)
    try:
        return BASELINE_REGISTRY[name](cfg, seed, **kwargs)
    except (TypeError, ValueError, ImportError, RuntimeError) as exc:
        return Unavailable(name, f"construction failed: {type(exc).__name__}: {exc}")


def availability_report(names: tuple[str, ...]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "baseline": name,
                "available": baseline_availability(name) == "",
                "reason": baseline_availability(name),
            }
            for name in names
        ]
    )
