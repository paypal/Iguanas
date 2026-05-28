import math
import re

import numpy as np
import polars as pl
from pydantic import PositiveInt


def extract_feature_names_from_rule(rule: str) -> list[str]:
    """Extract column names from a rule string with X["column_name"] patterns.

    Parameters
    ----------
    rule : str
        Rule string containing X["column_name"] patterns.

    Returns
    -------
    list[str]
        List of unique column names extracted from the rule, in order of appearance.

    Examples
    --------
    >>> rule = '(X["a"] >= 419) & (X["b"] < 1.0)'
    >>> extract_feature_names_from_rule(rule)
    ['a', 'b']
    """
    # Pattern to match X["column_name"] where column_name is between double quotes
    pattern = r'X\["([^"]+)"\]'

    # Find all matches and return unique column names preserving order
    matches = re.findall(pattern, rule)

    # Remove duplicates while preserving order
    seen = set()
    unique_columns = []
    for col in matches:
        if col not in seen:
            seen.add(col)
            unique_columns.append(col)

    return unique_columns


def filter_rules_by_feature_overlap(
    R: pl.DataFrame,
    importance: dict[str, float],
    min_difference: PositiveInt = 1,
    rule_column: str = "rule",
) -> pl.DataFrame:
    """Filter out rules that are too similar based on column usage, keeping the most important.

    Uses a greedy algorithm that processes rules sequentially. Note that this can result
    in keeping rules that are transitively similar (A similar to B, B filtered out,
    C similar to B but not to A, both A and C kept).

    Rules with identical column sets are always considered similar regardless of
    min_difference value (max one-sided difference = 0).

    Parameters
    ----------
    R : pl.DataFrame
        DataFrame with a column containing rule strings (X["column_name"] patterns).
    importance : dict
        Dictionary mapping rule strings to their importance values.
        Keys: rule strings matching those in R[rule_column]
        Values: importance values for each rule (missing rules default to 0.0)
    min_difference : PositiveInt, default=1
        Minimum number of different columns required between two rules.
        If two rules differ by fewer than this many columns, only the one
        with highest importance is kept. Must be >= 1.
    rule_column : str, default="rule"
        Name of the column containing rule strings.

    Returns
    -------
    pl.DataFrame
        Filtered DataFrame with similar rules removed (keeping highest importance).

    Examples
    --------
    >>> import polars as pl
    >>> rules_X = pl.DataFrame({
    ...     'rule': ['(X["a"] > 1) & (X["b"] < 2)',
    ...              '(X["a"] > 1) & (X["c"] < 3)',
    ...              '(X["a"] > 1) & (X["b"] < 2)'],
    ...     'score': [0.9, 0.85, 0.8]
    ... })
    >>> importance = {'(X["a"] > 1) & (X["b"] < 2)': 0.7,
    ...               '(X["a"] > 1) & (X["c"] < 3)': 0.9}
    >>> filter_rules_by_feature_overlap(rules_X, importance, min_difference=1)
    """
    # Get the rule strings from the specified column
    rules = R[rule_column].to_list()

    if len(rules) <= 1:
        return R

    # Extract column names for each rule
    rule_columns = []
    for rule in rules:
        cols = set(extract_feature_names_from_rule(rule))
        rule_columns.append(cols)

    # Track which indices to keep
    indices_to_keep = []

    for i, cols_i in enumerate(rule_columns):
        rule_i = rules[i]
        importance_i = importance.get(rule_i, 0.0)

        # Check if this rule is too similar to any previously kept rule
        similar_index = None
        for j in indices_to_keep:
            cols_j = rule_columns[j]

            # Calculate max one-sided difference (columns unique to each rule)
            cols_only_in_i = cols_i - cols_j
            cols_only_in_j = cols_j - cols_i
            max_one_sided_diff = max(len(cols_only_in_i), len(cols_only_in_j))

            # If the max one-sided difference is less than min_difference, they're too similar
            if max_one_sided_diff < min_difference:
                similar_index = j
                break

        if similar_index is not None:
            # Rules are similar - compare importance values
            rule_j = rules[similar_index]
            importance_j = importance.get(rule_j, 0.0)

            # Replace the kept rule if current one has higher importance
            if importance_i > importance_j:
                indices_to_keep.remove(similar_index)
                indices_to_keep.append(i)
        else:
            # Not similar to any kept rule, so keep it
            indices_to_keep.append(i)

    return R[indices_to_keep]


def filter_correlated_rules(
    R: pl.DataFrame, importance: dict, max_corr: float = 0.95, use_abs: bool = True
) -> list[str]:
    """Filter highly correlated columns, keeping only the most important.

    Accepts either a boolean predictions DataFrame (correlation is computed internally)
    or a pre-computed float correlation matrix. For each pair of columns with correlation
    above max_corr threshold, keeps only the column with higher importance value.

    Parameters
    ----------
    R : pl.DataFrame
        Either a boolean DataFrame of rule predictions (one column per rule, one row per
        sample) or a pre-computed n×n float correlation matrix. When boolean, Pearson
        correlations are computed automatically.
    importance : dict
        Dictionary mapping rule names (column names) to their importance values.
    max_corr : float, default=0.95
        Maximum correlation threshold. Pairs with correlation above this value
        will be filtered to keep only the most important rule.
    use_abs : bool, default=True
        If True, compares the absolute value of the correlation against max_corr,
        treating strong negative correlations (e.g. -0.97) the same as strong
        positive ones. If False, only positive correlations above max_corr trigger
        filtering.

    Returns
    -------
    list[str]
        List of selected columns to keep.

    Raises
    ------
    ValueError
        If length of importance dict doesn't match number of columns in R.

    Examples
    --------
    >>> import polars as pl
    >>> R = pl.DataFrame({
    ...     "rule_A": [True, False, True, False],
    ...     "rule_B": [True, False, True, False],  # identical to rule_A
    ...     "rule_C": [False, True, False, True],
    ... })
    >>> importance = {"rule_A": 0.8, "rule_B": 0.6, "rule_C": 0.9}
    >>> filter_correlated_rules(R, importance, max_corr=0.9).columns
    ['rule_A', 'rule_C']
    """
    if len(R.columns) != len(importance):
        raise ValueError("Length of importance dict must match number of columns in R")
    columns = R.columns

    if len(columns) <= 1:
        return columns

    # If the DataFrame contains boolean predictions, compute the correlation matrix.
    if all(dtype == pl.Boolean for dtype in R.dtypes):
        arr = R.cast(pl.Float64).to_numpy()
        with np.errstate(invalid="ignore"):
            corr_values = np.corrcoef(arr.T)
        C = pl.DataFrame(corr_values, schema=columns)
    else:
        C = R

    columns_to_remove = set()

    for i, col_i in enumerate(columns):
        if col_i in columns_to_remove:
            continue

        for col_j in columns[i + 1 :]:
            if col_j in columns_to_remove:
                continue

            # Get correlation value from the matrix
            corr = C[col_j][i]

            # Handle NaN correlations (e.g., constant columns)
            if corr is None or math.isnan(corr):
                continue

            # If correlation is above threshold, remove the less important column
            corr_value = abs(corr) if use_abs else corr
            if corr_value > max_corr:
                importance_i = importance[col_i]
                importance_j = importance[col_j]

                # Remove the column with lower importance
                if importance_i >= importance_j:
                    columns_to_remove.add(col_j)
                else:
                    columns_to_remove.add(col_i)
                    break  # col_i is removed, no need to check further pairs with it

    # Return the filtered subset of R
    selected_columns = [col for col in columns if col not in columns_to_remove]
    return selected_columns


def select_best_rule_per_column_combination(
    metrics: pl.DataFrame,
    sort_by: str = "precision"
) -> list[str]:
    """
    Select the rule with the highest metric score for each unique column combination.

    Parameters
    ----------
    metrics : pl.DataFrame
        DataFrame containing rule performance metrics. Must have a "rule" column
        and the metric specified in sort_by.
    sort_by : str, default="precision"
        Name of the metric column to use for selecting the best rule in each group.

    Returns
    -------
    list[str]
        Filtered rules with only the best rule for each column combination.

    Examples
    --------
    >>> metrics = pl.DataFrame({
    ...     "rule": ['(X["a"] > 1)', '(X["a"] > 2)', '(X["b"] < 3)'],
    ...     "precision": [0.95, 0.98, 0.96]
    ... })
    >>> select_best_rule_per_column_combination(metrics, sort_by="precision")
    # Returns the rule with highest precision for column "a" and the rule for column "b"
    """
    # Validate inputs
    if "rule" not in metrics.columns:
        raise ValueError("metrics DataFrame must contain a 'rule' column")
    if sort_by not in metrics.columns:
        raise ValueError(f"sort_by metric '{sort_by}' not found in metrics columns")

    # Extract rules and get column combinations
    rules = metrics["rule"].to_list()
    # column_combinations = count_column_combinations(rules)

    # Create a mapping from rule to its column combination
    pattern = r'X\["([^"]+)"\]'
    rule_to_columns = {}
    for rule in rules:
        columns = re.findall(pattern, rule)
        if columns:
            column_tuple = tuple(sorted(set(columns)))
            rule_to_columns[rule] = column_tuple

    # Add column combination as a new column to metrics
    metrics_with_combo = metrics.with_columns(
        pl.col("rule").map_elements(
            lambda r: str(rule_to_columns.get(r, ())),
            return_dtype=pl.String
        ).alias("column_combination")
    )

    # Group by column combination and select the row with max sort_by value
    best_rules = (
        metrics_with_combo
        .sort(sort_by, descending=True)
        .group_by("column_combination", maintain_order=True)
        .first()
        .drop("column_combination")
    )

    return best_rules["rule"].to_list()
