from __future__ import annotations

import polars as pl

# Columns produced by compute_metrics that are not metrics and should be excluded from comparison.
_NON_METRIC_COLS: frozenset[str] = frozenset(
    {
        "rule",
        "num_rules",
        "TP", "FP", "TN", "FN",
        "TP_weight", "FP_weight", "TN_weight", "FN_weight",
        "total_weight",
    }
)


def compare_rule_metrics(
    ref_metrics: pl.DataFrame,
    curr_metrics: pl.DataFrame,
    thresholds: dict[str, float] | None = None,
) -> pl.DataFrame:
    """Compare rule metrics between a reference period and a current period.

    Takes two :func:`~iguanas.metrics.compute_metrics` outputs and returns the
    per-rule delta for every shared metric column, together with a boolean flag
    indicating whether the rule has degraded beyond an optional threshold.

    Parameters
    ----------
    ref_metrics : pl.DataFrame
        Baseline metrics from :func:`~iguanas.metrics.compute_metrics`.
        Must contain a ``rule`` column.
    curr_metrics : pl.DataFrame
        Current-period metrics from :func:`~iguanas.metrics.compute_metrics`.
        Must contain a ``rule`` column. Only rules present in both DataFrames
        are compared (inner join on ``rule``).
    thresholds : dict[str, float] | None, default=None
        Maximum allowed drop per metric, e.g. ``{"precision": 0.05}`` flags
        rules whose precision fell by more than 5 pp. When ``None``, any
        negative delta is flagged.

    Returns
    -------
    pl.DataFrame
        One row per rule with columns:

        - ``rule``
        - ``{metric}_ref`` — metric value in the reference period
        - ``{metric}_curr`` — metric value in the current period
        - ``{metric}_delta`` — ``curr - ref`` (negative means degradation)
        - ``{metric}_degraded`` — ``True`` when the drop exceeds the threshold

    Examples
    --------
    >>> import polars as pl
    >>> from iguanas.metrics import compute_metrics
    >>> from iguanas.rule_monitoring import compare_rule_metrics
    >>> R_ref = pl.DataFrame({"rule_A": [True, False, True]})
    >>> y_ref = pl.Series([True, True, True])
    >>> R_curr = pl.DataFrame({"rule_A": [True, False, False]})
    >>> y_curr = pl.Series([True, True, True])
    >>> ref = compute_metrics(R_ref, y_ref)
    >>> curr = compute_metrics(R_curr, y_curr)
    >>> compare_rule_metrics(ref, curr, thresholds={"precision": 0.1})
    """
    shared_metrics = [
        c for c in ref_metrics.columns
        if c not in _NON_METRIC_COLS and c in curr_metrics.columns
    ]

    ref_renamed = ref_metrics.select(["rule", *shared_metrics]).rename(
        {c: f"{c}_ref" for c in shared_metrics}
    )
    curr_renamed = curr_metrics.select(["rule", *shared_metrics]).rename(
        {c: f"{c}_curr" for c in shared_metrics}
    )
    joined = ref_renamed.join(curr_renamed, on="rule", how="inner")

    delta_exprs = []
    for metric in shared_metrics:
        ref_col = f"{metric}_ref"
        curr_col = f"{metric}_curr"
        allowed_drop = -abs((thresholds or {}).get(metric, 0.0))
        delta_exprs.append((pl.col(curr_col) - pl.col(ref_col)).alias(f"{metric}_delta"))
        delta_exprs.append(
            ((pl.col(curr_col) - pl.col(ref_col)) < allowed_drop).alias(f"{metric}_degraded")
        )

    return joined.with_columns(delta_exprs)
