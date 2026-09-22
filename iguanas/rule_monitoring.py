"""Performance-degradation monitoring for deployed rules.

Compares two :func:`~iguanas.metrics.compute_metrics` snapshots — a reference
period and a current period — and reports the per-rule change in each shared
metric, flagging rules whose metric fell by more than a user-supplied
threshold.

This is threshold-based degradation monitoring on supervised metrics, not
statistical drift detection. No distributional test is performed: there is no
Kolmogorov-Smirnov test, no Population Stability Index, no Jensen-Shannon or
KL divergence, no change-point detection, and no significance testing. Both
snapshots must already contain labels, so this cannot detect covariate shift
on unlabelled production data.
"""
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
    """Flag per-rule performance degradation between two periods.

    Takes two :func:`~iguanas.metrics.compute_metrics` outputs and returns the
    per-rule delta for every shared metric column, together with a boolean flag
    indicating whether the metric dropped by more than an optional threshold.

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

    Notes
    -----
    What this does: an inner join on ``rule``, an arithmetic difference per
    shared metric column, and a comparison of that difference against a fixed
    threshold.

    What this does **not** do:

    - It is not a statistical drift test. No Kolmogorov-Smirnov test,
      Population Stability Index, Jensen-Shannon/KL divergence, or
      change-point detection is computed.
    - It performs no significance testing and returns no p-value or
      confidence interval, so a flagged drop may be sampling noise —
      particularly for low-volume rules. Inspect the ``TP``/``FP`` counts in
      the source metric tables before acting on a flag.
    - It compares *supervised* metrics only, so both periods must be labelled.
      It cannot detect feature or covariate shift on unlabelled data.
    - It compares exactly two snapshots. There is no trend estimation over a
      series of periods.
    - Rules absent from either input are silently dropped by the inner join.

    Choosing ``thresholds`` is a judgement call: with the default (``None``)
    every negative delta is flagged, including a one-sample fluctuation.

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
