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
    rules_from_rule_list,
    rules_from_rulefit,
    rules_from_skope,
    rules_from_sklearn_tree,
    rules_from_slipper,
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


class GBMCeilingBaseline(RuleSetMixin, _BaseBaseline):
    """Boosted-tree paths: rule generation by boosting, with no steering.

    Previously reported as a continuous accuracy ceiling. Under disjunction
    semantics it is instead a rule *generator* -- every root-to-leaf path whose
    leaf favours the positive class -- so it stays comparable with the others
    rather than being scored by a mechanism none of them have.

    ``n_estimators``/``max_depth`` are matched to :class:`IguanasAdapter`'s
    generation config, and ``scale_pos_weight`` is the class_weight='balanced'
    equivalent, so this isolates extraction strategy (all positive leaves vs.
    one max-gain rule per tree) from differences in model capacity.
    """

    name = "gbm_ceiling"

    def __init__(
        self, cfg: ExperimentConfig, seed: int, *, n_estimators: int = 100
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
            # Matches IguanasAdapter's RuleGenerationConfig.learning_rate default,
            # so the two fit under identical hyperparameters and any remaining
            # difference is attributable to extraction strategy alone.
            learning_rate=0.3,
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
    # RuleFitClassifier.fit has no sample_weight parameter; only enable this
    # for subclasses whose underlying estimator's fit() accepts it.
    supports_sample_weight = False

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
        fit_kwargs: dict[str, Any] = {"feature_names": names}
        if self.supports_sample_weight:
            fit_kwargs["sample_weight"] = _balanced_sample_weight(y)
        model.fit(data, y.astype(int), **fit_kwargs)
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


def _balanced_sample_weight(y: np.ndarray) -> np.ndarray:
    """sklearn's class_weight='balanced' formula, as per-row sample weights.

    Needed because several imodels estimators (e.g. RuleFitClassifier) accept
    neither a ``class_weight`` constructor argument nor ``sample_weight`` in
    ``fit()``, so balancing has to be applied wherever it *is* supported
    (:class:`SkopeRulesBaseline`) rather than uniformly.
    """
    y = y.astype(int)
    classes, counts = np.unique(y, return_counts=True)
    weight_by_class = {c: len(y) / (len(classes) * n) for c, n in zip(classes, counts, strict=True)}
    return np.array([weight_by_class[v] for v in y], dtype=float)


class RuleFitBaseline(_ImodelsBaseline):
    name = "rulefit"
    class_names = ("RuleFitClassifier",)
    # Matches IguanasAdapter's n_estimators/max_depth; already imodels' own
    # defaults, set explicitly so a future imodels release can't silently
    # drift the comparison. RuleFitClassifier has no class_weight/sample_weight
    # hook, so imbalance is not rebalanced here (see _balanced_sample_weight).
    # cv=False: imodels' default cv=True refits an internal regularization-path
    # search regardless of dataset size, dominating runtime; no other baseline
    # tunes its own hyperparameters via internal CV, so this is also fairer.
    default_kwargs = {"n_estimators": 100, "tree_size": 4, "cv": False}

    def _extract_rules(self) -> list[str]:
        return rules_from_rulefit(self._model, set(self._feature_names))


class SkopeRulesBaseline(_ImodelsBaseline):
    name = "skope_rules"
    class_names = ("SkopeRulesClassifier",)
    # imodels defaults are n_estimators=10, max_depth=3; matched here to
    # IguanasAdapter/GBMCeilingBaseline's n_estimators=100, max_depth=4.
    default_kwargs = {"n_estimators": 100, "max_depth": 4}
    # SkopeRulesClassifier.fit(sample_weight=...) exists, so it can be given
    # the class_weight='balanced' equivalent unlike RuleFit.
    supports_sample_weight = True

    def _extract_rules(self) -> list[str]:
        return rules_from_skope(self._model, set(self._feature_names))


class BRLBaseline(_ImodelsBaseline):
    name = "brl"
    class_names = ("BayesianRuleListClassifier",)
    needs_discretization = True
    # num_bins is a plain _ImodelsBaseline attribute (used by _fit_discretizer),
    # NOT a BayesianRuleListClassifier constructor param -- it cannot live in
    # default_kwargs, which only reaches the model via _build()'s get_params()
    # filter and would silently drop it.
    num_bins = 3
    # imodels defaults run 3 chains x 50k MCMC iterations, which dominates a run;
    # max_iter=100 matches the shared "100 trees" iteration budget used by every
    # ensemble baseline (gbm_ceiling/rulefit/skope_rules/slipper n_estimators=100).
    #
    # Runtime does NOT scale with row/feature count alone -- it scales with how
    # many itemsets clear minsupport, which on a *balanced* dataset (e.g.
    # spambase, ~39% positive) is combinatorially more than on the heavily
    # imbalanced ones the rest of the suite is mostly made of. On spambase,
    # maxcardinality=3/num_bins=8/minsupport=0.1 (imodels default) never
    # finished (>20min, well past the intended per-model timeout); the in-process
    # signal.alarm timeout in _greedy_f1_only._Timeout does NOT preempt this --
    # mlxtend's itemset mining runs long C-level calls that only see the pending
    # signal once they return, so a slow fit can silently run for a very long
    # time regardless of FIT_TIMEOUT_SECONDS. minsupport=0.2 (up from 0.1) prunes
    # the survivor count enough to finish spambase in ~18s; num_bins=3 and
    # maxcardinality=2 shrink the candidate itemset count further as a second
    # line of defence. select_ruleset's max_conditions_per_rule=4 still caps the
    # final rule length regardless of what BRL itself mines.
    default_kwargs = {
        "n_chains": 1,
        "max_iter": 100,
        "listlengthprior": 3,
        "maxcardinality": 2,
        "minsupport": 0.2,
    }

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
    # `max_card` is the standalone `corels` package's antecedent-cardinality cap
    # (default 2); not verified against imodels' own wrapper class, since
    # neither is installed in this environment. `_build()` drops unaccepted
    # kwargs via get_params(), so this is a no-op rather than an error if the
    # resolved class uses a different name.
    default_kwargs = {"max_card": 4}

    def _extract_rules(self) -> list[str]:
        return rules_from_brl(self._model, set(self._feature_names))


class SlipperBaseline(_ImodelsBaseline):
    """SLIPPER: AdaBoost over individual conjunctive rules, not trees."""

    name = "slipper"
    class_names = ("SlipperClassifier",)
    # n_estimators=100 matches every other ensemble baseline's iteration budget.
    # SlipperBaseEstimator exposes no rule-length cap, so max_conditions_per_rule=4
    # is only enforced downstream by select_ruleset, not at generation time.
    default_kwargs = {"n_estimators": 100}

    def _extract_rules(self) -> list[str]:
        return rules_from_slipper(self._model, set(self._feature_names))


class GreedyRuleListBaseline(_ImodelsBaseline):
    """Sequential covering: one single-condition split peeled off per depth.

    Despite picking each cutoff by ``criterion='gini'`` (a stump, like a tree
    split), the result is a flat decision *list*, not a branching tree -- the
    "no" branch is never split further, only the "yes" residual is.
    """

    name = "greedy_rule_list"
    class_names = ("GreedyRuleListClassifier",)
    # max_depth caps the list length (each depth contributes one single-condition
    # rule), matched to the shared max_conditions_per_rule=4 policy.
    default_kwargs = {"max_depth": 4}

    def _extract_rules(self) -> list[str]:
        return rules_from_rule_list(self._model, set(self._feature_names))


class OneRBaseline(_ImodelsBaseline):
    """Classic 1R: same decision-list builder as GreedyRuleList, shallower."""

    name = "oner"
    class_names = ("OneRClassifier",)
    default_kwargs = {"max_depth": 4}

    def _extract_rules(self) -> list[str]:
        return rules_from_rule_list(self._model, set(self._feature_names))


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
    "gbm_ceiling": GBMCeilingBaseline,
    "rulefit": RuleFitBaseline,
    "skope_rules": SkopeRulesBaseline,
    "brl": BRLBaseline,
    "figs": FIGSBaseline,
    "corels": CorelsBaseline,
    "ripper": RipperBaseline,
    "slipper": SlipperBaseline,
    "greedy_rule_list": GreedyRuleListBaseline,
    "oner": OneRBaseline,
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
