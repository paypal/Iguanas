"""Three-way nested evaluation protocol.

The single most common source of optimism in rule-learning papers is reusing
one split for rule *generation*, rule *selection* and *reporting*.  A rule pool
mined from a split has already seen that split's noise; picking the best
combination on the same rows then reports a number that is partly a maximum of
noise.

This module enforces a strict separation::

    outer StratifiedKFold      ->  (dev, test)
    inner stratified split(dev) ->  (generate, select)

* rules may only be mined on ``generate``
* thresholding, correlation dedup and combination search may only see ``select``
* every reported number comes from ``test``

The generalisation gap ``metric(select) - metric(test)`` is computed explicitly
for the chosen model, because quantifying that gap is the point of the exercise.
Leakage is prevented by a runtime guard, not by convention: every frame handed
to a model passes through :meth:`LeakGuard.subset`, which raises if the caller
asks for a row that belongs to the fold's test set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from iguanas.metrics import compute_single_metric

from .config import ExperimentConfig
from .operating_point import (
    OperatingPoint,
    apply_threshold,
    choose_threshold,
    precision_recall,
    realised_alert_rate,
)
from .preprocessing import SharedPreprocessor
from .timing import Counters, FitTimeout, time_limit, timed

class LeakageError(AssertionError):
    """Raised when a stage requests rows reserved for the fold's test set."""


@runtime_checkable
class NestedModel(Protocol):
    """Interface every competitor must satisfy to enter the protocol."""

    name: str

    def fit_generate(self, X: pl.DataFrame, y: np.ndarray) -> None: ...

    def fit_select(self, X: pl.DataFrame, y: np.ndarray, alert_rate: float) -> None: ...

    def score(self, X: pl.DataFrame) -> np.ndarray: ...

    def complexity(self) -> int: ...

    def counters(self) -> Counters: ...


@dataclass(frozen=True)
class LeakGuard:
    """Runtime custodian of the fold's test rows."""

    test_index: frozenset[int]
    fold: int
    allow_select_on_test: bool = False

    def check(self, index: np.ndarray, stage: str) -> None:
        overlap = self.test_index.intersection(int(i) for i in index)
        if overlap:
            raise LeakageError(
                f"fold {self.fold}: stage {stage!r} requested {len(overlap)} row(s) "
                f"reserved for test (e.g. {sorted(overlap)[:5]})"
            )

    def subset(
        self, X: pl.DataFrame, y: np.ndarray, index: np.ndarray, stage: str
    ) -> tuple[pl.DataFrame, np.ndarray]:
        exempt = stage == "test" or (stage == "select" and self.allow_select_on_test)
        if not exempt:
            self.check(index, stage)
        return X[index], y[index]


@dataclass(frozen=True)
class NestedSplit:
    fold: int
    seed: int
    generate: np.ndarray
    select: np.ndarray
    test: np.ndarray
    oracle_selection: bool = False

    def guard(self) -> LeakGuard:
        return LeakGuard(
            frozenset(int(i) for i in self.test), self.fold, self.oracle_selection
        )

    def validate(self) -> None:
        parts = {"generate": self.generate, "select": self.select, "test": self.test}
        for name, idx in parts.items():
            if idx.size == 0:
                raise LeakageError(f"fold {self.fold}: split {name!r} is empty")
            if len(set(idx.tolist())) != idx.size:
                raise LeakageError(f"fold {self.fold}: split {name!r} has duplicate rows")
        # 'generate' and 'select' may legitimately coincide (selection_split="train"),
        # and in the oracle arm 'select' is deliberately the test rows; neither may
        # ever leak into a *reported* estimate other than through that arm's label.
        pairs = [("generate", "test")]
        if not self.oracle_selection:
            pairs.append(("select", "test"))
        for a, b in pairs:
            shared = set(parts[a].tolist()) & set(parts[b].tolist())
            if shared:
                raise LeakageError(
                    f"fold {self.fold}: {a!r} and {b!r} overlap on {len(shared)} row(s)"
                )


def make_nested_splits(
    y: np.ndarray, cfg: ExperimentConfig, seed: int
) -> list[NestedSplit]:
    """Build the outer (dev/test) and inner (generate/select) index sets.

    ``cfg.selection_split`` decides where rules are chosen:

    ``"train"``
        Generate and select on the whole development split, report on test. The
        test estimate is unbiased and rule generation sees every training row,
        which matters because a thin candidate pool starves the later stages.
    ``"holdout"``
        Carve a separate selection split out of development, leaving generation
        with the remainder. Unbiased too, and it isolates selection-induced
        optimism, at the cost of data for both stages.
    ``"oracle"``
        Select directly on test. This is **not** a valid estimate of
        generalisation -- it reports the best the selection stage could have
        done had it seen the answers -- and exists only as an upper bound to
        measure how much that stage overfits.
    """
    y_int = np.asarray(y, dtype=int)
    outer = StratifiedKFold(
        n_splits=cfg.n_outer_folds, shuffle=True, random_state=seed
    )
    select_fraction = 1.0 / cfg.n_inner_folds
    splits: list[NestedSplit] = []
    for fold, (dev_idx, test_idx) in enumerate(
        outer.split(np.zeros(len(y_int)), y_int)
    ):
        dev_sorted = np.sort(dev_idx)
        if cfg.selection_split == "holdout":
            gen_idx, sel_idx = train_test_split(
                dev_idx,
                test_size=select_fraction,
                stratify=y_int[dev_idx],
                random_state=seed * 1000 + fold,
                shuffle=True,
            )
            generate, select = np.sort(gen_idx), np.sort(sel_idx)
        elif cfg.selection_split == "oracle":
            generate, select = dev_sorted, np.sort(test_idx)
        else:
            generate = select = dev_sorted
        split = NestedSplit(
            fold=fold,
            seed=seed,
            generate=generate,
            select=select,
            test=np.sort(test_idx),
            oracle_selection=cfg.selection_split == "oracle",
        )
        split.validate()
        splits.append(split)
    return splits


def score_metric(y: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    """Evaluate ``metric`` through iguanas so harness and library agree exactly."""
    return float(
        compute_single_metric(
            pl.Series(np.asarray(y_pred, dtype=bool)),
            pl.Series(np.asarray(y, dtype=bool)),
            metric,
        )
    )


def _ranking_scores(y: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    if len(np.unique(y)) < 2:
        return {"average_precision": float("nan"), "roc_auc": float("nan")}
    return {
        "average_precision": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
    }


@dataclass
class FoldResult:
    dataset: str
    model: str
    seed: int
    fold: int
    alert_rate: float
    status: str = "ok"
    error: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "model": self.model,
            "seed": self.seed,
            "fold": self.fold,
            "target_alert_rate": self.alert_rate,
            "status": self.status,
            "error": self.error,
            **self.metrics,
        }


# Reported for every fold at the chosen operating point. F-beta is swept rather
# than fixed at F1 because the beta that matters depends on the alert budget: at
# a tight budget false positives dominate the cost (beta < 1), at a loose one
# missed positives do (beta > 1). MCC and WRAcc are included because both stay
# informative under extreme imbalance, where F1 and accuracy do not.
_FOLD_METRICS: tuple[str, ...] = (
    "f0.5",
    "f1",
    "f2",
    "mcc",
    "lift",
    "wracc",
)


def _operating_point_metrics(
    y_true: np.ndarray, pred: np.ndarray, prefix: str
) -> dict[str, float]:
    truth = pl.Series(np.asarray(y_true, dtype=bool))
    flagged = pl.Series(np.asarray(pred, dtype=bool))
    return {
        f"{prefix}_{name}": float(compute_single_metric(flagged, truth, name))
        for name in _FOLD_METRICS
    }


def _evaluate_at_operating_point(
    y_select: np.ndarray,
    scores_select: np.ndarray,
    y_test: np.ndarray,
    scores_test: np.ndarray,
    alert_rate: float,
    metric: str,
) -> dict[str, Any]:
    """Fix the threshold on ``select`` only, then apply it unchanged to ``test``."""
    point: OperatingPoint = choose_threshold(scores_select, alert_rate)
    pred_select = apply_threshold(scores_select, point)
    pred_test = apply_threshold(scores_test, point)

    p_sel, r_sel = precision_recall(y_select, pred_select)
    p_test, r_test = precision_recall(y_test, pred_test)
    m_sel = score_metric(y_select, pred_select, metric)
    m_test = score_metric(y_test, pred_test, metric)
    test_rate = realised_alert_rate(pred_test)

    return {
        "threshold": point.threshold,
        "select_alert_rate": point.realised_alert_rate,
        "test_alert_rate": test_rate,
        # Recall is only comparable between models that respected the same alert
        # budget. A model flagging 62% of the population against a 1% budget will
        # top a recall ranking while being operationally useless, so budget
        # compliance travels with the metrics rather than being re-derived later.
        "budget_utilisation": test_rate / point.target_alert_rate
        if point.target_alert_rate
        else float("nan"),
        "within_budget": bool(test_rate <= point.target_alert_rate * 1.1),
        "select_precision": p_sel,
        "select_recall": r_sel,
        f"select_{metric}": m_sel,
        "test_precision": p_test,
        "test_recall": r_test,
        f"test_{metric}": m_test,
        "generalisation_gap": m_sel - m_test,
        "precision_gap": p_sel - p_test,
        "recall_gap": r_sel - r_test,
        **_operating_point_metrics(y_select, pred_select, "select"),
        **_operating_point_metrics(y_test, pred_test, "test"),
        **{f"test_{k}": v for k, v in _ranking_scores(y_test, scores_test).items()},
    }


@dataclass
class PreparedFold:
    """Preprocessed splits for one fold, reused across every alert budget."""

    X_gen: pl.DataFrame
    y_gen: np.ndarray
    X_sel: pl.DataFrame
    y_sel: np.ndarray
    X_test: pl.DataFrame
    y_test: np.ndarray


def prepare_fold(
    X: pl.DataFrame, y: np.ndarray, split: NestedSplit
) -> PreparedFold:
    """Subset and preprocess once; neither step depends on the alert budget.

    Preprocessing is fitted on train (generate + select), never on test.
    Imputation means and category levels use no labels, so the generate/select
    boundary -- which exists to stop rule *selection* seeing the data rules were
    induced on -- does not apply; only the held-out test split must be excluded.

    Discretisation is NOT applied here: only BRL and CORELS require binary
    features, and forcing quantile bins on everything would strip continuous
    split points from the tree-based models. Those two bin their own input in
    ``_fit_discretizer``.
    """
    guard = split.guard()
    X_gen, y_gen = guard.subset(X, y, split.generate, "generate")
    X_sel, y_sel = guard.subset(X, y, split.select, "select")
    X_test, y_test = guard.subset(X, y, split.test, "test")

    preprocessor = SharedPreprocessor(discretize=False).fit(
        pl.concat([X_gen, X_sel], how="vertical")
    )
    return PreparedFold(
        X_gen=preprocessor.transform(X_gen),
        y_gen=y_gen,
        X_sel=preprocessor.transform(X_sel),
        y_sel=y_sel,
        X_test=preprocessor.transform(X_test),
        y_test=y_test,
    )


def run_fold(
    model: NestedModel,
    X: pl.DataFrame,
    y: np.ndarray,
    split: NestedSplit,
    cfg: ExperimentConfig,
    dataset_name: str,
    alert_rate: float,
    prepared: PreparedFold | None = None,
    generate_seconds: float | None = None,
) -> FoldResult:
    """Run one model through one outer fold under the nested protocol.

    When *prepared* is supplied the model is assumed already generated: rule
    generation and preprocessing do not depend on the alert budget, so repeating
    them for every point on the operating curve multiplies the cost of the most
    expensive stage by the size of the budget grid.
    """
    fold = prepared if prepared is not None else prepare_fold(X, y, split)

    result = FoldResult(
        dataset=dataset_name,
        model=model.name,
        seed=split.seed,
        fold=split.fold,
        alert_rate=alert_rate,
    )
    try:
        with time_limit(cfg.fit_timeout_s):
            if prepared is None:
                with timed() as generate_watch:
                    model.fit_generate(fold.X_gen, fold.y_gen)
                generate_elapsed = generate_watch.seconds
            else:
                generate_elapsed = generate_seconds or 0.0
            with timed() as select_watch:
                model.fit_select(fold.X_sel, fold.y_sel, alert_rate)
            with timed() as score_watch:
                scores_sel = model.score(fold.X_sel)
                scores_test = model.score(fold.X_test)
    except FitTimeout as exc:
        result.status = "timeout"
        result.error = str(exc)
        return result
    except (ValueError, KeyError, RuntimeError, ArithmeticError, MemoryError) as exc:
        result.status = "failed"
        result.error = f"{type(exc).__name__}: {exc}"
        return result

    result.metrics = {
        "n_generate": int(len(split.generate)),
        "n_select": int(len(split.select)),
        "n_test": int(len(split.test)),
        "positive_rate_test": float(np.mean(fold.y_test)),
        "complexity_conditions": int(model.complexity()),
        "chosen_rule": getattr(model, "_chosen", "") or "",
        "generate_seconds": generate_elapsed,
        "select_seconds": select_watch.seconds,
        "score_seconds": score_watch.seconds,
        "total_seconds": generate_elapsed + select_watch.seconds + score_watch.seconds,
        **model.counters().as_dict(),
        **_evaluate_at_operating_point(
            fold.y_sel, scores_sel, fold.y_test, scores_test, alert_rate, cfg.metric
        ),
    }
    return result
