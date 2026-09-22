import re

import polars as pl

# Default shrinkage strength for the m-estimate. Larger values pull the estimate
# of a low-coverage rule further toward the base rate.
M_ESTIMATE_M = 10.0

_COVERAGE_METRICS = frozenset({"lift", "wracc", "laplace", "m_estimate"})

# A rule references each feature as X["col"] / X['col'], once per condition.
_FEATURE_REF_PATTERN = r"""X\[["'][^"']+["']\]"""


def count_conditions(rule: str) -> int:
    """Number of atomic conditions in a rule expression.

    Complexity is measured in conditions rather than rules: a 3-rule disjunction
    with 12 conditions is not simpler than a 5-rule one with 8.

    Parameters
    ----------
    rule : str
        Rule expression, e.g. ``'(X["a"] > 1) & (X["b"] <= 2)'``.

    Returns
    -------
    int
        Condition count. Zero for a string that references no features, which is
        what a plain rule *name* such as ``"rule_A"`` will yield.

    Examples
    --------
    >>> count_conditions('(X["a"] > 1) & (X["b"] <= 2)')
    2
    """
    return len(re.findall(_FEATURE_REF_PATTERN, rule))


def count_features(rule: str) -> int:
    """Number of *distinct* features a rule expression references.

    Two conditions on the same feature (a range) are easier to read than two
    conditions on different features, so this complements :func:`count_conditions`.

    Examples
    --------
    >>> count_features('(X["a"] > 1) & (X["a"] < 5)')
    1
    """
    return len(set(re.findall(_FEATURE_REF_PATTERN, rule)))


def compute_single_metric(
    y_pred: pl.Series,
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
    y_pred : pl.Series
        Boolean prediction series.
    y : pl.Series
        Boolean target series.
    metric : str
        Metric name: "precision", "recall", "accuracy", "mcc", an F-beta score
        (f<number>), or one of the coverage-aware rule metrics "lift", "wracc",
        "laplace" and "m_estimate".
    weights : pl.Series | None, default=None
        Optional sample weights. When provided, all counts use weighted sums.

    Returns
    -------
    float
        The requested metric value.
    """
    y_bool = y.cast(pl.Boolean)
    y_pred_bool = y_pred.cast(pl.Boolean)

    if weights is not None:
        TP = float(weights.filter(y_bool & y_pred_bool).sum())
        FP = float(weights.filter(~y_bool & y_pred_bool).sum())
        FN = float(weights.filter(y_bool & ~y_pred_bool).sum())
    else:
        TP = float((y_bool & y_pred_bool).sum())
        FP = float((~y_bool & y_pred_bool).sum())
        FN = float((y_bool & ~y_pred_bool).sum())

    if metric == "precision":
        return TP / (TP + FP) if (TP + FP) > 0 else 0.0
    if metric == "recall":
        return TP / (TP + FN) if (TP + FN) > 0 else 0.0
    if metric == "accuracy":
        TN = (
            float((~y_bool & ~y_pred_bool).sum())
            if weights is None
            else float(weights.filter(~y_bool & ~y_pred_bool).sum())
        )
        return (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
    if metric == "mcc":
        TN = (
            float((~y_bool & ~y_pred_bool).sum())
            if weights is None
            else float(weights.filter(~y_bool & ~y_pred_bool).sum())
        )
        denom = ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5
        return (TP * TN - FP * FN) / denom if denom > 0 else 0.0
    if metric in _COVERAGE_METRICS:
        total = float(len(y_bool)) if weights is None else float(weights.sum())
        covered = TP + FP
        positives = TP + FN
        if metric == "laplace":
            return (TP + 1.0) / (covered + 2.0)
        if total <= 0:
            return 0.0
        base_rate = positives / total
        if metric == "m_estimate":
            return (TP + M_ESTIMATE_M * base_rate) / (covered + M_ESTIMATE_M)
        if metric == "wracc":
            return TP / total - (covered * positives) / (total * total)
        if covered <= 0 or base_rate <= 0:
            return 0.0
        return (TP / covered) / base_rate
    if metric.startswith("f"):
        beta = float(metric[1:])
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        denom = beta**2 * precision + recall
        return (1 + beta**2) * precision * recall / denom if denom > 0 else 0.0
    raise ValueError(
        f"Unsupported metric '{metric}'. Must be 'precision', 'recall', "
        f"'accuracy', 'mcc', an F-beta score (f<number>), or one of "
        f"{sorted(_COVERAGE_METRICS)}."
    )


def compute_metrics(
    R: pl.Series | pl.DataFrame,
    y: pl.Series,
    weights: pl.Series | None = None,
    betas: list[float] | None = None,
    m: float = M_ESTIMATE_M,
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
    m : float, default=10.0
        Shrinkage strength for the ``m_estimate`` column. Larger values pull
        low-coverage rules further toward the base rate.

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
        - lift: precision divided by the base rate
        - wracc: weighted relative accuracy, ``coverage * (precision - base_rate)``
        - laplace: ``(TP + 1) / (TP + FP + 2)``
        - m_estimate: precision shrunk toward the base rate by *m*
        - num_rules: Number of individual rules y_pred (1 for single rules)
        - num_conditions: Atomic conditions in the rule expression
        - num_features: Distinct features the rule expression references

        If weights is provided, additional columns with "_weight" suffix:

        - TP_weight, FP_weight, TN_weight, FN_weight: Weighted confusion matrix
        - total_weight, precision_weight, recall_weight, accuracy_weight: Weighted versions
        - f{b}_weight for each b in *betas*: Weighted F-beta scores

    Examples
    --------
    >>> import polars as pl
    >>> # Count-based metrics only
    >>> metrics_df = compute_metrics(R, y, weights=None)
    >>>
    >>> # Both count and weighted metrics
    >>> metrics_df = compute_metrics(R, y, weights=transaction_amounts)
    >>>
    >>> # Sort by TPVE3 to find best rules
    >>> top_rules = metrics_df.sort("TPVE3", descending=True).head(10)
    """
    if betas is None:
        betas = [0.25, 0.5, 1, 1.5, 2]
    if y.dtype != pl.Boolean:
        y = y.cast(pl.Boolean)
    if isinstance(R, pl.Series):
        R = R.to_frame()
    # Compute confusion matrix for all columns
    if weights is not None:
        # Both count and weighted metrics
        metrics_df = pl.DataFrame(
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
        metrics_df = pl.DataFrame(
            {
                "rule": R.columns,
                "TP": [(y & R[col]).sum() for col in R.columns],
                "FP": [(~y & R[col]).sum() for col in R.columns],
                "TN": [(~y & ~R[col]).sum() for col in R.columns],
                "FN": [(y & ~R[col]).sum() for col in R.columns],
            }
        )

    # Step 1: Add basic metrics (precision, recall, and accuracy)
    metrics_df = metrics_df.with_columns(
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
        pl.when(
            (pl.col("TP") + pl.col("FP"))
            * (pl.col("TP") + pl.col("FN"))
            * (pl.col("TN") + pl.col("FP"))
            * (pl.col("TN") + pl.col("FN"))
            == 0
        )
        .then(pl.lit(0.0))
        .otherwise(
            (pl.col("TP") * pl.col("TN") - pl.col("FP") * pl.col("FN")).cast(pl.Float64)
            / (
                (pl.col("TP") + pl.col("FP"))
                * (pl.col("TP") + pl.col("FN"))
                * (pl.col("TN") + pl.col("FP"))
                * (pl.col("TN") + pl.col("FN"))
            ).cast(pl.Float64).sqrt()
        )
        .alias("mcc"),
        # Number of rules
        (pl.col("rule").str.count_matches(r"\) \| \(") + 1).alias("num_rules"),
        # Complexity. Counted from the rule string, so these are 0 when columns
        # carry plain names rather than rule expressions.
        pl.col("rule").str.count_matches(_FEATURE_REF_PATTERN).alias("num_conditions"),
        pl.col("rule")
        .str.extract_all(_FEATURE_REF_PATTERN)
        .list.unique()
        .list.len()
        .alias("num_features"),
    ]

    # Coverage-aware rule quality metrics. Unlike precision these cannot be gamed
    # by a rule firing on a handful of rows, and unlike recall they are not
    # maximised by flagging everything.
    n_expr = (pl.col("TP") + pl.col("FP") + pl.col("TN") + pl.col("FN")).cast(pl.Float64)
    covered_expr = (pl.col("TP") + pl.col("FP")).cast(pl.Float64)
    positives_expr = (pl.col("TP") + pl.col("FN")).cast(pl.Float64)
    tp_expr = pl.col("TP").cast(pl.Float64)
    expressions.extend(
        [
            pl.when((covered_expr <= 0) | (positives_expr <= 0) | (n_expr <= 0))
            .then(pl.lit(0.0))
            .otherwise((tp_expr / covered_expr) / (positives_expr / n_expr))
            .alias("lift"),
            pl.when(n_expr <= 0)
            .then(pl.lit(0.0))
            .otherwise(tp_expr / n_expr - (covered_expr * positives_expr) / (n_expr * n_expr))
            .alias("wracc"),
            ((tp_expr + 1.0) / (covered_expr + 2.0)).alias("laplace"),
            pl.when(n_expr <= 0)
            .then(pl.lit(0.0))
            .otherwise(
                (tp_expr + m * (positives_expr / n_expr)) / (covered_expr + m)
            )
            .alias("m_estimate"),
        ]
    )

    if weights is not None:
        # First compute total_weight
        metrics_df = metrics_df.with_columns(
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
        metrics_df = metrics_df.with_columns(
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
                pl.when(
                    (pl.col("TP_weight") + pl.col("FP_weight"))
                    * (pl.col("TP_weight") + pl.col("FN_weight"))
                    * (pl.col("TN_weight") + pl.col("FP_weight"))
                    * (pl.col("TN_weight") + pl.col("FN_weight"))
                    == 0
                )
                .then(pl.lit(0.0))
                .otherwise(
                    (pl.col("TP_weight") * pl.col("TN_weight") - pl.col("FP_weight") * pl.col("FN_weight")).cast(pl.Float64)
                    / (
                        (pl.col("TP_weight") + pl.col("FP_weight"))
                        * (pl.col("TP_weight") + pl.col("FN_weight"))
                        * (pl.col("TN_weight") + pl.col("FP_weight"))
                        * (pl.col("TN_weight") + pl.col("FN_weight"))
                    ).cast(pl.Float64).sqrt()
                )
                .alias("mcc_weight"),
            ]
        )
        n_w = pl.col("total_weight").cast(pl.Float64)
        covered_w = (pl.col("TP_weight") + pl.col("FP_weight")).cast(pl.Float64)
        positives_w = (pl.col("TP_weight") + pl.col("FN_weight")).cast(pl.Float64)
        tp_w = pl.col("TP_weight").cast(pl.Float64)
        # laplace/m_estimate smooth *counts*, so they have no weighted analogue.
        expressions.extend(
            [
                pl.when((covered_w <= 0) | (positives_w <= 0) | (n_w <= 0))
                .then(pl.lit(0.0))
                .otherwise((tp_w / covered_w) / (positives_w / n_w))
                .alias("lift_weight"),
                pl.when(n_w <= 0)
                .then(pl.lit(0.0))
                .otherwise(tp_w / n_w - (covered_w * positives_w) / (n_w * n_w))
                .alias("wracc_weight"),
            ]
        )

    metrics_df = metrics_df.with_columns(expressions)

    return metrics_df
