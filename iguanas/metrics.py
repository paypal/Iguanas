import polars as pl


def compute_single_metric(
    combined: pl.Series,
    y: pl.Series,
    metric: str,
    weights: pl.Series | None = None,
) -> float:
    """Compute a single performance metric for one boolean prediction series.

    Faster than compute_metrics when only one scalar is needed, because it
    skips computing all 25+ derived metrics. Used internally by
    combine_rules_beam_search during candidate evaluation.

    Parameters
    ----------
    combined : pl.Series
        Boolean prediction series.
    y : pl.Series
        Boolean target series.
    metric : str
        Metric name: "precision", "recall", "accuracy", or an F-beta score (f<number>).
    weights : pl.Series | None, default=None
        Optional sample weights. When provided, all counts use weighted sums.

    Returns
    -------
    float
        The requested metric value.
    """
    y_bool = y.cast(pl.Boolean)
    combined_bool = combined.cast(pl.Boolean)

    if weights is not None:
        TP = float(weights.filter(y_bool & combined_bool).sum())
        FP = float(weights.filter(~y_bool & combined_bool).sum())
        FN = float(weights.filter(y_bool & ~combined_bool).sum())
    else:
        TP = float((y_bool & combined_bool).sum())
        FP = float((~y_bool & combined_bool).sum())
        FN = float((y_bool & ~combined_bool).sum())

    if metric == "precision":
        return TP / (TP + FP) if (TP + FP) > 0 else 0.0
    elif metric == "recall":
        return TP / (TP + FN) if (TP + FN) > 0 else 0.0
    elif metric == "accuracy":
        TN = float((~y_bool & ~combined_bool).sum()) if weights is None else float(weights.filter(~y_bool & ~combined_bool).sum())
        return (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
    elif metric.startswith("f"):
        beta = float(metric[1:])
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        denom = beta ** 2 * precision + recall
        return (1 + beta**2) * precision * recall / denom if denom > 0 else 0.0
    else:
        raise ValueError(
            f"Unsupported metric '{metric}'. Must be 'precision', 'recall', "
            f"'accuracy', or an F-beta score (f<number>)."
        )


def compute_metrics(
    R: pl.DataFrame,
    y: pl.Series,
    weights: pl.Series | None = None,
    betas: list[float] = [0.25, 0.5, 1, 1.5, 2],
) -> pl.DataFrame:
    """Compute comprehensive performance metrics for all rule columns.

    Calculates confusion matrix, precision, recall, F-beta scores, and TPVE metrics
    for each rule. Optionally computes weighted versions of all metrics.

    Parameters
    ----------
    R : pl.DataFrame
        DataFrame with boolean columns representing rule predictions. Each column
        is a rule that evaluates to True/False for each observation.
    y : pl.Series
        Boolean target series indicating true labels (True for positive class).
        Will be cast to Boolean if not already.
    weights : pl.Series | None, default=None
        Optional numeric series for weighted metrics computation. If provided,
        computes both count-based and weighted versions of all metrics.
    betas : list[float], default=[0.25, 0.5, 1, 1.5, 2]
        F-beta values to compute. Each value ``b`` produces a column named
        ``f{b}`` (and ``f{b}_weight`` when *weights* is provided).

    Returns
    -------
    pl.DataFrame
        DataFrame with one row per rule containing:

        - rule: Rule name (column name from R)
        - TP, FP, TN, FN: Confusion matrix counts
        - precision, recall, accuracy: Standard classification metrics
        - flagged(%): Percentage of total flagged as positive
        - good_flagged(%): Percentage of negatives flagged as positive
        - f{b} for each b in *betas*: F-beta scores
        - num_rules: Number of individual rules combined (1 for single rules)

        If weights is provided, additional columns with "_weight" suffix:

        - TP_weight, FP_weight, TN_weight, FN_weight: Weighted confusion matrix
        - total_weight, precision_weight, recall_weight, accuracy_weight: Weighted versions
        - f{b}_weight for each b in *betas*: Weighted F-beta scores

    Examples
    --------
    >>> import polars as pl
    >>> # Count-based metrics only
    >>> metrics_X = compute_metrics(R, y, weights=None)
    >>>
    >>> # Both count and weighted metrics
    >>> metrics_X = compute_metrics(R, y, weights=transaction_amounts)
    >>>
    >>> # Sort by TPVE3 to find best rules
    >>> top_rules = metrics_X.sort("TPVE3", descending=True).head(10)
    """
    if y.dtype != pl.Boolean:
        y = y.cast(pl.Boolean)
    # Compute confusion matrix for all columns
    if weights is not None:
        # Both count and weighted metrics
        metrics_X = pl.DataFrame(
            {
                "rule": R.columns,
                "TP": [(y & R[col]).sum() for col in R.columns],
                "FP": [(~y & R[col]).sum() for col in R.columns],
                "TN": [(~y & ~R[col]).sum() for col in R.columns],
                "FN": [(y & ~R[col]).sum() for col in R.columns],
                "TP_weight": [(weights.filter(y & R[col])).sum() for col in R.columns],
                "FP_weight": [(weights.filter(~y & R[col])).sum() for col in R.columns],
                "TN_weight": [(weights.filter(~y & ~R[col])).sum() for col in R.columns],
                "FN_weight": [(weights.filter(y & ~R[col])).sum() for col in R.columns],
            }
        )
    else:
        # Only count metrics
        metrics_X = pl.DataFrame(
            {
                "rule": R.columns,
                "TP": [(y & R[col]).sum() for col in R.columns],
                "FP": [(~y & R[col]).sum() for col in R.columns],
                "TN": [(~y & ~R[col]).sum() for col in R.columns],
                "FN": [(y & ~R[col]).sum() for col in R.columns],
            }
        )

    # Step 1: Add basic metrics (precision, recall, and accuracy)
    metrics_X = metrics_X.with_columns(
        [
            (pl.col("TP") / (pl.col("TP") + pl.col("FP"))).alias("precision"),
            (pl.col("TP") / (pl.col("TP") + pl.col("FN"))).alias("recall"),
            (
                (pl.col("TP") + pl.col("TN"))
                / (pl.col("TP") + pl.col("FP") + pl.col("TN") + pl.col("FN"))
            ).alias("accuracy"),
        ]
    )

    # Step 2: Build complete list of all derived metrics that depend on precision/recall
    expressions = [
        (
            (pl.col("TP") + pl.col("FP"))
            / (pl.col("TP") + pl.col("FP") + pl.col("TN") + pl.col("FN"))
            * 100
        ).alias("flagged(%)"),
        (pl.col("FP") / (pl.col("TN") + pl.col("FP")) * 100).alias("good_flagged(%)"),
        *[
            (
                (1 + b**2)
                * pl.col("precision")
                * pl.col("recall")
                / (b**2 * pl.col("precision") + pl.col("recall"))
            ).alias(f"f{b:g}")
            for b in betas
        ],
        # Number of rules
        (pl.col("rule").str.count_matches(r"\) \| \(") + 1).alias("num_rules"),
    ]

    if weights is not None:
        # First compute total_weight
        metrics_X = metrics_X.with_columns(
            [
                (
                    pl.col("TP_weight")
                    + pl.col("FP_weight")
                    + pl.col("TN_weight")
                    + pl.col("FN_weight")
                ).alias("total_weight"),
            ]
        )
        # Then compute precision, recall, and accuracy using total_weight
        metrics_X = metrics_X.with_columns(
            [
                (pl.col("TP_weight") / (pl.col("TP_weight") + pl.col("FP_weight"))).alias(
                    "precision_weight"
                ),
                (pl.col("TP_weight") / (pl.col("TP_weight") + pl.col("FN_weight"))).alias(
                    "recall_weight"
                ),
                ((pl.col("TP_weight") + pl.col("TN_weight")) / pl.col("total_weight")).alias(
                    "accuracy_weight"
                ),
            ]
        )
        expressions.extend(
            [
                *[
                    (
                        (1 + b**2)
                        * pl.col("precision_weight")
                        * pl.col("recall_weight")
                        / (b**2 * pl.col("precision_weight") + pl.col("recall_weight"))
                    ).alias(f"f{b:g}_weight")
                    for b in betas
                ],
            ]
        )

    metrics_X = metrics_X.with_columns(expressions)

    return metrics_X
