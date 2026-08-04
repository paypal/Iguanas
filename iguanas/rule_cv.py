from __future__ import annotations

import numpy as np
import polars as pl

from .metrics import compute_metrics
from .rule_evaluation import apply_rules


def validate_rules_cv(
    X: pl.DataFrame,
    y: pl.Series,
    rules: list[str],
    n_folds: int = 5,
    cv_metrics: list[str] | None = None,
    weight_column: str | None = None,
    shuffle: bool = True,
    random_state: int | None = None,
) -> pl.DataFrame:
    """Evaluate rule stability across K folds.

    Validates already-generated rules on K held-out folds without re-generating
    them. For each fold the rules are evaluated on the validation split, and the
    mean, standard deviation, and minimum of each requested metric across folds
    are returned. Rules with a high ``{metric}_cv_std`` or a low
    ``{metric}_cv_min`` are likely over-fitted to the training data.

    Parameters
    ----------
    X : pl.DataFrame
        Feature DataFrame. Must contain all columns referenced in ``rules``.
    y : pl.Series
        Target series (boolean or binary).
    rules : list[str]
        Rule expressions to validate. Typically obtained from
        :func:`~iguanas.rule_evaluation.apply_filter_and_deduplicate_rules`
        or similar.
    n_folds : int, default=5
        Number of CV folds. The data is split into ``n_folds`` contiguous
        blocks (after optional shuffling).
    cv_metrics : list[str] | None, default=None
        Metric names to compute CV statistics for. When ``None``, defaults to
        ``["precision", "recall", "f1"]``.
    weight_column : str | None, default=None
        Name of a column in ``X`` to use as sample weights when computing
        metrics. If ``None``, all samples are weighted equally.
    shuffle : bool, default=True
        Whether to shuffle the row indices before splitting into folds.
    random_state : int | None, default=None
        Random seed for reproducibility when ``shuffle=True``.

    Returns
    -------
    pl.DataFrame
        One row per rule with columns:

        - ``rule``
        - ``{metric}_cv_mean`` — mean of the metric across folds
        - ``{metric}_cv_std`` — standard deviation across folds
        - ``{metric}_cv_min`` — worst-fold value (lowest)

        Sorted by ``rule`` name.

    Examples
    --------
    >>> import polars as pl
    >>> X = pl.DataFrame({"age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70]})
    >>> y = pl.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    >>> rules = ['(X["age"] >= 50)']
    >>> validate_rules_cv(X, y, rules, n_folds=2, random_state=0)

    Notes
    -----
    The CV stability scores carry an optimism bias: because ``rules`` are
    generated from the full dataset *before* ``validate_rules_cv`` is called,
    the held-out folds were already seen during rule extraction.  The
    reported ``{metric}_cv_min`` and ``{metric}_cv_std`` are therefore a
    lower bound on overfitting, not a true out-of-sample estimate.  Use
    them to flag unstable rules rather than to estimate deployment
    performance.

    See Also
    --------
    apply_filter_and_deduplicate_rules : Complete evaluation pipeline that
        produces the ``rules`` input for this function.
    """
    if cv_metrics is None:
        cv_metrics = ["precision", "recall", "f1"]

    n = len(y)
    indices = list(range(n))

    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    fold_size = n // n_folds
    fold_frames: list[pl.DataFrame] = []

    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < n_folds - 1 else n
        val_idx = indices[val_start:val_end]

        X_val = X[val_idx]
        y_val = y[val_idx]
        weights_val = X_val[weight_column] if weight_column is not None else None

        R_val = apply_rules(X_val, rules)
        m = compute_metrics(R_val, y_val, weights=weights_val)

        available = [c for c in cv_metrics if c in m.columns]
        fold_frames.append(m.select(["rule"] + available))

    stacked = pl.concat(fold_frames)

    available_metrics = [c for c in cv_metrics if c in stacked.columns]
    agg_exprs = []
    for metric in available_metrics:
        agg_exprs.extend(
            [
                pl.col(metric).mean().alias(f"{metric}_cv_mean"),
                pl.col(metric).std().alias(f"{metric}_cv_std"),
                pl.col(metric).min().alias(f"{metric}_cv_min"),
            ]
        )

    return stacked.group_by("rule").agg(agg_exprs).sort("rule")


def identify_unstable_rules(
    cv_result: pl.DataFrame,
    metric: str = "f1",
    max_std: float = 0.05,
    min_mean: float | None = None,
) -> pl.DataFrame:
    """Return rules whose cross-fold metric is unstable or consistently poor.

    Filters the output of :func:`validate_rules_cv` to surface rules that are
    likely over-fitted (high variance across folds) or simply weak (low mean
    metric).

    Parameters
    ----------
    cv_result : pl.DataFrame
        Output of :func:`validate_rules_cv`.  Must contain columns
        ``{metric}_cv_std`` and (if ``min_mean`` is set) ``{metric}_cv_mean``.
    metric : str, default="f1"
        Metric prefix to inspect (must match one used in :func:`validate_rules_cv`).
    max_std : float, default=0.05
        Rules whose ``{metric}_cv_std`` **exceeds** this threshold are flagged
        as unstable.
    min_mean : float | None, default=None
        If provided, also flag rules whose ``{metric}_cv_mean`` is **below**
        this threshold (consistently weak rules).

    Returns
    -------
    pl.DataFrame
        Subset of ``cv_result`` containing only flagged rules, sorted by
        ``{metric}_cv_std`` descending (most unstable first).

    Raises
    ------
    ValueError
        If required columns are absent from ``cv_result``.

    Examples
    --------
    >>> cv = validate_rules_cv(X, y, rules, n_folds=5)
    >>> identify_unstable_rules(cv, metric="f1", max_std=0.05, min_mean=0.3)
    """
    std_col = f"{metric}_cv_std"
    mean_col = f"{metric}_cv_mean"

    if std_col not in cv_result.columns:
        raise ValueError(
            f"Column '{std_col}' not found. "
            f"Run validate_rules_cv with cv_metrics=['{metric}', ...]."
        )
    if min_mean is not None and mean_col not in cv_result.columns:
        raise ValueError(
            f"Column '{mean_col}' not found. "
            f"Run validate_rules_cv with cv_metrics=['{metric}', ...]."
        )

    mask = pl.col(std_col) > max_std
    if min_mean is not None:
        mask = mask | (pl.col(mean_col) < min_mean)

    return cv_result.filter(mask).sort(std_col, descending=True)
