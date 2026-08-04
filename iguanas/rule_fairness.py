from __future__ import annotations

import polars as pl

from .metrics import compute_metrics


def compute_subgroup_metrics(
    R: pl.DataFrame,
    y: pl.Series,
    group_col: pl.Series,
    weights: pl.Series | None = None,
    betas: list[float] | None = None,
) -> pl.DataFrame:
    """Compute rule performance metrics broken down by a protected attribute.

    Evaluates each rule's precision, recall, and other metrics within every
    subgroup defined by the unique values of ``group_col``.  Useful for
    detecting disparate impact — e.g. a rule that has high precision overall
    but systematically mis-fires on a particular demographic group.

    Parameters
    ----------
    R : pl.DataFrame
        Boolean DataFrame of rule predictions (columns = rules, rows = samples).
    y : pl.Series
        Target series (boolean or binary).
    group_col : pl.Series
        Series defining subgroup membership. Any dtype is supported; unique
        values define the groups.  Must have the same length as ``R`` and ``y``.
    weights : pl.Series | None, default=None
        Optional sample weights.  Forwarded to
        :func:`~iguanas.metrics.compute_metrics` for each subgroup.
    betas : list[float] | None, default=None
        F-beta values to compute.  Forwarded to
        :func:`~iguanas.metrics.compute_metrics`.

    Returns
    -------
    pl.DataFrame
        Long-format DataFrame with one row per *(group, rule)* pair.
        Columns:

        - ``group`` — subgroup label
        - ``group_size`` — number of samples in that group
        - ``rule`` — rule expression string
        - All metric columns from :func:`~iguanas.metrics.compute_metrics`
          (TP, FP, precision, recall, f1, …)

        Sorted by ``group`` then ``rule``.  Returns an empty DataFrame if
        ``R`` is empty.

    Examples
    --------
    >>> import polars as pl
    >>> R = pl.DataFrame({
    ...     "rule_A": [True,  False, True,  False, True],
    ...     "rule_B": [False, True,  False, True,  True],
    ... })
    >>> y = pl.Series([1, 0, 1, 0, 1])
    >>> group = pl.Series(["M", "M", "F", "F", "F"])
    >>> compute_subgroup_metrics(R, y, group)

    See Also
    --------
    compute_metrics : Aggregate (non-split) metric computation.
    """
    if R.is_empty():
        return pl.DataFrame()

    groups = group_col.unique().sort().to_list()
    frames: list[pl.DataFrame] = []

    for g in groups:
        mask = group_col == g

        R_g = R.filter(mask)
        y_g = y.filter(mask)
        w_g = weights.filter(mask) if weights is not None else None

        m = compute_metrics(R_g, y_g, weights=w_g, betas=betas)
        m = m.with_columns(
            [
                pl.lit(g).alias("group"),
                pl.lit(mask.sum()).alias("group_size"),
            ]
        )
        frames.append(m)

    result = pl.concat(frames)
    # Reorder: group columns first
    front = ["group", "group_size"]
    rest = [c for c in result.columns if c not in front]
    return result.select(front + rest).sort(["group", "rule"])


def compute_disparate_impact_ratio(
    subgroup_df: pl.DataFrame,
    reference_group: str,
    metric: str = "precision",
) -> pl.DataFrame:
    """Compute the disparate-impact ratio relative to a reference group.

    For each rule and each non-reference group, the ratio is:

    .. math::

        \\text{DIR} = \\frac{\\text{metric}_{\\text{group}}}{\\text{metric}_{\\text{reference}}}

    A ratio below **0.8** (the "four-fifths rule") typically signals
    potentially disparate impact under US EEOC guidelines.

    Parameters
    ----------
    subgroup_df : pl.DataFrame
        Output of :func:`compute_subgroup_metrics` — must contain columns
        ``group``, ``rule``, and the requested ``metric``.
    reference_group : str
        The group whose metric value is used as the denominator.
    metric : str, default="precision"
        Name of the metric column to use for the ratio.

    Returns
    -------
    pl.DataFrame
        One row per *(rule, group)* pair for non-reference groups with columns:

        - ``rule``
        - ``group``
        - ``{metric}_group`` — metric value for this group
        - ``{metric}_reference`` — metric value for the reference group
        - ``disparate_impact_ratio`` — ratio (group / reference); ``null``
          when reference metric is zero or null.

        Sorted by ``rule`` then ``disparate_impact_ratio`` ascending.

    Examples
    --------
    >>> subgroup_df = compute_subgroup_metrics(R, y, group_col)
    >>> compute_disparate_impact_ratio(subgroup_df, reference_group="1", metric="precision")

    See Also
    --------
    compute_subgroup_metrics : Compute per-subgroup metrics first.
    """
    required = {"group", "rule", metric}
    missing = required - set(subgroup_df.columns)
    if missing:
        raise ValueError(f"subgroup_df is missing columns: {missing}")

    ref = subgroup_df.filter(pl.col("group") == reference_group).select(
        ["rule", pl.col(metric).alias(f"{metric}_reference")]
    )
    non_ref = subgroup_df.filter(pl.col("group") != reference_group).select(
        ["rule", "group", pl.col(metric).alias(f"{metric}_group")]
    )

    return (
        non_ref.join(ref, on="rule", how="left")
        .with_columns(
            (pl.col(f"{metric}_group") / pl.col(f"{metric}_reference")).alias(
                "disparate_impact_ratio"
            )
        )
        .select(
            ["rule", "group", f"{metric}_group", f"{metric}_reference", "disparate_impact_ratio"]
        )
        .sort(["rule", "disparate_impact_ratio"])
    )
