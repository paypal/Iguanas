"""Matched-alert-rate operating point protocol.

Disjunction learners emit booleans and therefore occupy a single, uncontrolled
point on the precision/recall curve; full classifiers emit scores and can be
placed anywhere.  Comparing them without matching the *alert rate* (the fraction
of the population flagged for review) compares two different operating regimes.

Every function here fixes that budget first and measures second.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

FLAGGED_PCT_COLUMN = "flagged(%)"


@dataclass(frozen=True)
class OperatingPoint:
    """A concrete decision threshold plus the alert rate it actually realised."""

    target_alert_rate: float
    realised_alert_rate: float
    threshold: float
    n_flagged: int


def _as_scores(scores: np.ndarray) -> np.ndarray:
    arr = np.asarray(scores, dtype="float64").ravel()
    if arr.size == 0:
        raise ValueError("scores must be non-empty")
    return arr


def choose_threshold(scores: np.ndarray, alert_rate: float) -> OperatingPoint:
    """Pick the achievable operating point flagging as close to ``alert_rate`` as possible.

    Only thresholds that sit on an actual score value are considered, so the
    realised rate is always one a model can genuinely produce. Ties are never
    split: if the cut score is duplicated, every duplicate is flagged.

    Operating *under* budget is feasible, not a failure. Interpolating a
    quantile instead breaks binary-output models: a ruleset firing on 0.9% of
    rows, asked for a 5% alert rate, has its threshold driven down to 0 and ends
    up flagging the entire population. Rule learners cannot manufacture alerts
    they have no rule for, so the closest achievable point below budget is used
    and the realised rate is reported.

    When even the single highest-scoring group exceeds the budget, that group is
    returned and the realised rate will overshoot the target.
    """
    if not 0.0 < alert_rate <= 1.0:
        raise ValueError(f"alert_rate must be in (0, 1], got {alert_rate}")
    arr = _as_scores(scores)
    n = arr.size
    budget = alert_rate * n

    ascending = np.sort(arr)
    values = np.unique(arr)[::-1]
    # Rows scoring >= each distinct value: the achievable alert volumes.
    flagged_at = n - np.searchsorted(ascending, values, side="left")

    feasible = flagged_at <= budget
    index = int(np.argmax(np.where(feasible, flagged_at, -1))) if feasible.any() else 0
    threshold = float(values[index])
    flagged = int(flagged_at[index])
    return OperatingPoint(
        target_alert_rate=alert_rate,
        realised_alert_rate=flagged / n,
        threshold=threshold,
        n_flagged=flagged,
    )


def predictions_at_alert_rate(
    scores: np.ndarray, alert_rate: float
) -> tuple[np.ndarray, OperatingPoint]:
    arr = _as_scores(scores)
    point = choose_threshold(arr, alert_rate)
    return arr >= point.threshold, point


def apply_threshold(scores: np.ndarray, point: OperatingPoint) -> np.ndarray:
    """Apply a threshold fitted elsewhere (e.g. on the selection split)."""
    return _as_scores(scores) >= point.threshold


def precision_recall(y: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    y_bool = np.asarray(y, dtype=bool)
    pred = np.asarray(y_pred, dtype=bool)
    tp = float(np.count_nonzero(y_bool & pred))
    fp = float(np.count_nonzero(~y_bool & pred))
    fn = float(np.count_nonzero(y_bool & ~pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return precision, recall


def recall_at_alert_rate(y: np.ndarray, scores: np.ndarray, alert_rate: float) -> float:
    pred, _ = predictions_at_alert_rate(scores, alert_rate)
    return precision_recall(y, pred)[1]


def precision_at_alert_rate(
    y: np.ndarray, scores: np.ndarray, alert_rate: float
) -> float:
    pred, _ = predictions_at_alert_rate(scores, alert_rate)
    return precision_recall(y, pred)[0]


def select_rule_by_alert_budget(
    metrics: pl.DataFrame,
    alert_rate: float,
    *,
    tie_break_metric: str = "f1",
    rule_column: str = "rule",
) -> str | None:
    """Return the best-scoring rule whose ``flagged(%)`` fits within the budget.

    ``metrics`` is a frame as produced by :func:`iguanas.metrics.compute_metrics`.
    Fitting the budget is a *constraint*; *tie_break_metric* is the objective, and
    ``flagged(%)`` only breaks ties between equally good rules.

    Ordering by ``flagged(%)`` instead -- filling the budget -- is actively
    harmful: it returns the broadest rule that fits, which is the least selective
    one. On mammography at a 1% budget that choice scored recall 0.024 at
    precision 0.059, where the WRAcc-best rule within the same budget scored
    recall 0.341 at precision 1.000.

    If every candidate overshoots the budget, the least-overshooting one is
    returned so the caller still gets a usable model; its realised alert rate is
    reported separately and will exceed the target.
    """
    if metrics.is_empty() or rule_column not in metrics.columns:
        return None
    if FLAGGED_PCT_COLUMN not in metrics.columns:
        raise KeyError(f"metrics frame lacks {FLAGGED_PCT_COLUMN!r}")

    budget_pct = alert_rate * 100.0
    sort_by = [FLAGGED_PCT_COLUMN]
    descending = [True]
    if tie_break_metric in metrics.columns:
        sort_by = [tie_break_metric, FLAGGED_PCT_COLUMN]
        descending = [True, True]

    within = metrics.filter(pl.col(FLAGGED_PCT_COLUMN) <= budget_pct).filter(
        pl.col(FLAGGED_PCT_COLUMN) > 0
    )
    if not within.is_empty():
        return str(within.sort(sort_by, descending=descending).row(0, named=True)[rule_column])

    over = metrics.sort(FLAGGED_PCT_COLUMN, descending=False)
    return str(over.row(0, named=True)[rule_column])


def realised_alert_rate(y_pred: np.ndarray) -> float:
    pred = np.asarray(y_pred, dtype=bool)
    return float(pred.mean()) if pred.size else 0.0
