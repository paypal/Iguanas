from functools import reduce
from typing import Any

import polars as pl

from .metrics import compute_metrics
from .rule_selection import filter_correlated_rules

# Mapping from operator string to the corresponding Polars Series/Expr method name.
# Defined at module level to avoid reconstruction on every function call.
_OPS: dict[str, str] = {
    ">=": "__ge__",
    ">": "__gt__",
    "<=": "__le__",
    "<": "__lt__",
    "==": "__eq__",
    "!=": "__ne__",
}


def apply_rules(X: pl.DataFrame, rules: list[str]) -> pl.DataFrame:
    r"""Evaluate rule expressions on a DataFrame to produce boolean predictions.

    Takes a list of rule strings (logical conditions) and evaluates them against
    the input DataFrame to produce a boolean DataFrame where each column represents
    the evaluation result of one rule.

    Parameters
    ----------
    X : pl.DataFrame
        Input DataFrame on which to evaluate the rules. Must contain all columns
        referenced in the rule expressions.
    rules : list[str]
        List of rule expressions as strings. Each rule should be a valid
        Polars expression that evaluates to a boolean result.
        Format: ``'(X["column_name"] operator value) [& \| \|] ...'``

    Returns
    -------
    pl.DataFrame
        DataFrame containing only the evaluated rule columns as boolean values.
        Each column name matches the corresponding rule expression string.
        Shape: (len(X), len(rules))

    Examples
    --------
    >>> import polars as pl
    >>> X = pl.DataFrame({"age": [25, 30, 35], "income": [50000, 60000, 70000]})
    >>> rules = ['(X["age"] >= 30)', '(X["income"] > 55000)']
    >>> R = apply_rules(X, rules)
    >>> R.columns
    ['(X["age"] >= 30)', '(X["income"] > 55000)']
    >>> R
    shape: (3, 2)
    ┌──────────────────┬─────────────────────┐
    │ (X["age"] >= 30) │ (X["income"] > ... │
    │ ---              │ ---                 │
    │ bool             │ bool                │
    ╞══════════════════╪═════════════════════╡
    │ false            │ false               │
    │ true             │ true                │
    │ true             │ true                │
    └──────────────────┴─────────────────────┘

    Notes
    -----
    Uses Python's `eval()` with a restricted namespace for security.
    The namespace includes only `pl` (Polars) and `X` (the input DataFrame).
    """
    # Provide explicit namespace for eval to ensure cross-platform compatibility
    namespace = {"pl": pl, "X": X}
    exprs = [eval(rule, namespace).alias(rule) for rule in rules]
    return X.with_columns(exprs).select(rules)


def apply_and_filter_by_performance(
    X: pl.DataFrame,
    y: pl.Series,
    rules: list[str],
    weight_column: str | None = None,
    metric_thresholds: list[dict[str, Any]] | None = None,
    sort_by: str = "precision",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Evaluate rules on a dataset split and filter by performance thresholds.

    Generates rule predictions, computes performance metrics (optionally weighted),
    filters rules that don't meet minimum precision and recall thresholds, and
    returns both the filtered rule predictions and their metrics.

    Parameters
    ----------
    X : pl.DataFrame
        Feature DataFrame on which to evaluate rules.
    y : pl.Series
        Target series with true labels (boolean or binary).
    rules : list[str]
        List of rule expressions as strings to evaluate.
    weight_column : str | None, default=None
        Name of column in X to use as sample weights for metric computation.
        If None, all samples are weighted equally.
    metric_thresholds : list[dict[str, Any]], default=[{"name": "accuracy", "operator": ">=", "value": 0.5}]
        List of threshold dicts, each with keys:

        - ``"name"``: metric column name (e.g. ``"precision"``, ``"recall"``, ``"f1"``).
        - ``"operator"``: comparison string — one of ``">="``, ``">"``, ``"<="``, ``"<"``, ``"=="``, ``"!="``.
        - ``"value"``: numeric threshold.

        All conditions are combined with AND. Rules failing any condition are dropped.
    sort_by : str, default="precision"
        Metric name to sort results by (descending order).
        Must be a valid column name from compute_metrics output.

    Returns
    -------
    R_split : pl.DataFrame
        Boolean DataFrame containing only rules that meet all threshold criteria.
        Columns are rule expressions, rows are samples.
    metrics_split : pl.DataFrame
        Performance metrics for the filtered rules, sorted by `sort_by`.
        Contains columns like 'rule', 'precision', 'recall', 'f1', etc.

    Examples
    --------
    >>> X = pl.DataFrame({"age": [25, 30, 35, 40], "income": [50000, 60000, 70000, 80000]})
    >>> y = pl.Series([0, 0, 1, 1])
    >>> rules = ['(X["age"] >= 30)', '(X["income"] > 55000)']
    >>> metric_thresholds = [{"name": "accuracy", "operator": ">=", "value": 0.5}]
    >>> R, metrics = apply_and_filter_by_performance(X, y, rules, metric_thresholds=metric_thresholds)
    >>> metrics[['rule', 'precision', 'recall']]

    See Also
    --------
    apply_rules : Evaluate rule expressions on a DataFrame
    compute_metrics : Compute performance metrics for rule predictions
    """
    if metric_thresholds is None:
        metric_thresholds = [{"name": "accuracy", "operator": ">=", "value": 0.5}]
    if not rules:
        return pl.DataFrame(), pl.DataFrame()
    R_split = apply_rules(X, rules)
    weights = X[weight_column] if weight_column is not None else None
    metrics_split = compute_metrics(R_split, y, weights=weights).sort(sort_by, descending=True)

    filter_expr = reduce(
        lambda acc, t: acc & getattr(pl.col(t["name"]), _OPS[t["operator"]])(t["value"]),
        metric_thresholds,
        pl.lit(True),
    )
    metrics_split = metrics_split.filter(filter_expr)
    if metrics_split.is_empty():
        return pl.DataFrame(), pl.DataFrame()
    return R_split[metrics_split["rule"].to_list()], metrics_split


def select_diverse_top_rules(
    R: pl.DataFrame,
    metrics: pl.DataFrame,
    max_corr: float = 0.8,
    importance_metric: str = "f0.5",
    top_n: int | None = None,
    sort_by: str = "f1",
) -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
    """Select top performing rules while removing highly correlated duplicates.

    Ranks rules by a performance metric, optionally limits to top N rules,
    then filters out correlated rules to produce a diverse set of high-quality
    rules. When rules are correlated, keeps the one with higher importance.

    Parameters
    ----------
    R : pl.DataFrame
        Boolean DataFrame with rule predictions (columns are rules, rows are samples).
    metrics : pl.DataFrame
        Performance metrics for the rules in R.
        Must contain columns 'rule', importance_metric, and sort_by metric.
    max_corr : float, default=0.8
        Maximum correlation threshold. Rule pairs with correlation > max_corr
        are considered too similar, and only the more important one is kept.
    importance_metric : str, default="f0.5"
        Metric name to use for determining rule importance when filtering
        correlated rules. Higher values indicate more important rules.
    top_n : int | None, default=None
        If specified, limits selection to top N rules by sort_by metric
        before filtering correlations. If None, considers all rules.
    sort_by : str, default="f1"
        Metric name to sort and rank rules by (descending order) before
        applying correlation filtering.

    Returns
    -------
    R_filtered : pl.DataFrame
        Boolean DataFrame containing only the selected uncorrelated rules.
    metrics_filtered : pl.DataFrame
        Performance metrics for the selected rules only.
    selected_rule_list : list[str]
        List of selected rule expressions (column names from R_filtered).

    Examples
    --------
    >>> R = pl.DataFrame({
    ...     "rule_A": [True, False, True, False],
    ...     "rule_B": [True, False, True, True],
    ...     "rule_C": [False, True, False, True]
    ... })
    >>> metrics = pl.DataFrame({
    ...     "rule": ["rule_A", "rule_B", "rule_C"],
    ...     "f1": [0.8, 0.75, 0.6],
    ...     "f0.5": [0.85, 0.78, 0.65]
    ... })
    >>> R_filtered, metrics_filtered, rules = select_diverse_top_rules(
    ...     R, metrics, max_corr=0.9, top_n=2
    ... )
    >>> rules  # ['rule_A', 'rule_C'] - rule_B removed due to correlation with rule_A

    See Also
    --------
    filter_correlated_rules : Filter correlated boolean columns by importance
    """
    # Guard against empty input
    if R.is_empty() or metrics.is_empty() or "rule" not in metrics.columns:
        print("Number of uncorrelated rules: 0")
        return pl.DataFrame(), pl.DataFrame(), []

    # Sort and optionally limit to top N
    metrics_sorted = metrics.sort(sort_by, descending=True)
    if top_n is not None:
        metrics_sorted = metrics_sorted[:top_n]

    # Filter correlated rules — returns list[str]
    importance_dict = dict(metrics_sorted[["rule", importance_metric]].rows())
    selected_rules = filter_correlated_rules(
        R[metrics_sorted["rule"].to_list()], max_corr=max_corr, importance=importance_dict
    )
    R_filtered = R[selected_rules] if selected_rules else pl.DataFrame()

    print(f"Number of uncorrelated rules: {len(selected_rules)}")

    return R_filtered, metrics_sorted.filter(pl.col("rule").is_in(selected_rules)), selected_rules


def apply_filter_and_deduplicate_rules(
    X: pl.DataFrame,
    y: pl.Series,
    rules: list[str] | pl.DataFrame,
    weight_column: str | None = None,
    metric_thresholds: list[dict[str, Any]] | None = None,
    top_n_rules: int | None = None,
    max_corr: float = 0.8,
    sort_by: str = "precision",
) -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
    """Complete pipeline to evaluate and filter rules on a dataset.

    Performs a comprehensive rule evaluation workflow:
    1. Evaluates rules on the dataset and filters by performance thresholds
    2. Selects top N uncorrelated rules based on performance
    3. Returns predictions, metrics, and selected rule list

    Parameters
    ----------
    X : pl.DataFrame
        Feature DataFrame on which to evaluate rules.
    y : pl.Series
        Target series (boolean or binary).
    rules : list[str] | pl.DataFrame
        Either a list of rule expressions as strings, or a DataFrame with a 'rule' column.
    weight_column : str | None, default=None
        Name of weight column in X. If None, unweighted metrics are computed.
    metric_thresholds : list[dict[str, Any]], default=[precision >= 0.2, recall >= 0.2]
        List of threshold dicts forwarded to :func:`apply_and_filter_by_performance`.
        Each dict must have keys ``"name"``, ``"operator"``, and ``"value"``.
        All conditions are combined with AND.
    top_n_rules : int | None, default=None
        Maximum number of rules to keep after sorting by sort_by metric.
        Applied before correlation filtering. If None, keeps all rules that pass thresholds.
    max_corr : float, default=0.8
        Maximum correlation threshold for filtering similar rules.
    sort_by : str, default="precision"
        Metric name to sort results by (descending order).

    Returns
    -------
    R : pl.DataFrame
        Boolean DataFrame with selected rule predictions.
        Shape: (len(X), n_selected_rules)
    metrics : pl.DataFrame
        Performance metrics for selected rules.
    selected_rules : list[str]
        List of selected rule expressions that passed all filters.

    Examples
    --------
    >>> import polars as pl
    >>> # Combine train and test data
    >>> X = pl.concat([X_train, X_test])
    >>> y = pl.concat([y_train, y_test])
    >>> rules = ['(X["age"] >= 30)', '(X["income"] > 55000)']
    >>> thresholds = [{"name": "precision", "operator": ">=", "value": 0.5},
    ...               {"name": "recall",    "operator": ">=", "value": 0.5}]
    >>> R, metrics, selected_rules = apply_filter_and_deduplicate_rules(
    ...     X, y, rules, metric_thresholds=thresholds, top_n_rules=10
    ... )
    >>> print(f"Selected {len(selected_rules)} rules")
    >>> metrics[['rule', 'precision', 'recall', 'f1']]

    Notes
    -----
    The function applies filtering in sequence:
    - Threshold filtering removes low-performing rules
    - Top-N selection limits the rule set size (optional)
    - Correlation filtering ensures diversity in the final rule set

    See Also
    --------
    apply_and_filter_by_performance : Evaluate rules on a single data split
    select_diverse_top_rules : Select top rules while removing correlations
    """
    if metric_thresholds is None:
        metric_thresholds = [
            {"name": "precision", "operator": ">=", "value": 0.2},
            {"name": "recall", "operator": ">=", "value": 0.2},
        ]
    # Convert rules_df to list if needed
    if isinstance(rules, pl.DataFrame):
        rules = rules.select("rule").to_series().to_list()

    # Evaluate and filter by thresholds
    R, metrics = apply_and_filter_by_performance(
        X, y, rules, weight_column, metric_thresholds, sort_by
    )

    # Select top uncorrelated rules
    R, metrics, selected_rules = select_diverse_top_rules(
        R, metrics, max_corr=max_corr, top_n=top_n_rules, sort_by=sort_by
    )

    return R, metrics, selected_rules
