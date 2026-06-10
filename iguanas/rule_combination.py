import heapq
import itertools

import polars as pl

from .metrics import compute_metrics, compute_single_metric


def combine_rules_full_search(
    R: pl.DataFrame,
    n: int = 3,
    max_combinations_per_n: int = 200_000,
    batch_size: int = 50_000,
    operator: str = "or",
) -> pl.DataFrame:
    """Combine rules using logical operations to create new composite rules.

    Generates all possible combinations of 2 to n rules and creates new columns
    where each combination is evaluated using the specified logical operation (OR/AND).
    The combined rule name reflects the operation between component rules.

    Optimized for speed using batch processing and vectorized operations.

    Parameters
    ----------
    R : pl.DataFrame
        DataFrame containing rule columns to be combined. Each column should
        represent a boolean or binary rule evaluation. All columns will be
        used as candidate rules.
    n : int, default=3
        Maximum number of rules to combine. Generates all combinations from
        size 2 up to size n.
    max_combinations_per_n : int, default=250_000
        Maximum number of combinations to generate per combination size.
        If exceeded, only the first max_combinations_per_n are used.
    batch_size : int, default=100_000
        Number of combinations to process in each batch to manage memory.
    operator : str, default='or'
        Boolean operator to apply: 'or' for OR operations (any True),
        'and' for AND operations (all True).

    Returns
    -------
    pl.DataFrame
        DataFrame containing the original rules plus all generated combined
        rules. Combined rule columns are named using the pattern:

        - "(rule1) | (rule2) | ..." for OR operations
        - "(rule1) & (rule2) & ..." for AND operations

    Examples
    --------
    >>> import polars as pl
    >>> R = pl.DataFrame({"rule_A": [1, 0, 1], "rule_B": [0, 1, 1]})
    >>> combine_rules_full_search(R, n=2, operator='or')
    # Returns DataFrame with original columns plus "(rule_A) | (rule_B)"
    >>> combine_rules_full_search(R, n=2, operator='and')
    # Returns DataFrame with original columns plus "(rule_A) & (rule_B)"
    """
    if operator not in ["or", "and"]:
        raise ValueError(f"operator must be 'or' or 'and', got '{operator}'")

    rules = R.columns
    separator = " | " if operator == "or" else " & "

    # Step 1: Generate all combinations across all sizes from 2 to n
    all_combinations = []
    for combo_size in range(2, n + 1):
        combos = list(itertools.combinations(rules, combo_size))

        # Apply limit per combination size
        if len(combos) > max_combinations_per_n:
            combos = combos[:max_combinations_per_n]

        all_combinations.extend(combos)

    if not all_combinations:
        return R

    # Step 2: Process all combinations in batches for memory efficiency
    all_exprs = []
    for batch_start in range(0, len(all_combinations), batch_size):
        batch_end = min(batch_start + batch_size, len(all_combinations))
        batch_combos = all_combinations[batch_start:batch_end]

        for combi in batch_combos:
            rule_name = separator.join(f"({rule})" for rule in combi)

            expr = pl.col(combi[0])
            for rule in combi[1:]:
                if operator == "or":
                    expr = expr | pl.col(rule)
                else:
                    expr = expr & pl.col(rule)
            all_exprs.append(expr.alias(rule_name))

    return R.with_columns(all_exprs)


def combine_rules_cumulative(
    R: pl.DataFrame, output_names: list[str] | None = None, operator: str = "or"
) -> pl.DataFrame:
    """Compute horizontal cumulative boolean operations across all columns.

    Parameters
    ----------
    R : pl.DataFrame
        Input DataFrame. All columns will be used in the cumulative operation.
    output_names : list[str] | None, default=None
        List of names for the output columns. If None, generates names based on operator.
        Must have the same length as R.columns.
    operator : str, default='or'
        Boolean operator to apply:

        - 'or': cumulative OR (any True)
        - 'and': cumulative AND (all True)

    Returns
    -------
    pl.DataFrame
        DataFrame with boolean values:

        - If operator='or': True if at least one condition is True up to that position
        - If operator='and': True if all conditions are True up to that position

    Raises
    ------
    ValueError
        If operator is not 'or' or 'and', or if output_names length doesn't match columns.

    Examples
    --------
    >>> import polars as pl
    >>> R = pl.DataFrame({
    ...     "rule_A": [True, False, True],
    ...     "rule_B": [False, True, True],
    ...     "rule_C": [True, True, False],
    ... })
    >>> combine_rules_cumulative(R, operator="or")
    # Column 1: rule_A | ...; Column 2: rule_A | rule_B | ...; Column 3: all three
    >>> combine_rules_cumulative(R, operator="and", output_names=["step1", "step2", "step3"])
    # Named columns, each True only if all rules up to that position are True
    """
    if operator not in ["or", "and"]:
        raise ValueError(f"operator must be 'or' or 'and', got '{operator}'")

    columns = R.columns

    if output_names is None:
        separator = " | " if operator == "or" else " & "
        output_names = [
            separator.join(f"({col})" for col in columns[: i + 1]) for i in range(len(columns))
        ]
    elif len(output_names) != len(columns):
        raise ValueError(
            f"Length of output_names ({len(output_names)}) must match length of columns ({len(columns)})"
        )

    cumsum_R = R.select(pl.cum_sum_horizontal(*columns)).unnest("cum_sum")

    if operator == "or":
        # Cumulative OR: at least one True (cumsum > 0)
        return cumsum_R.select(
            [
                pl.col(col_name).gt(0).alias(output_name)
                for col_name, output_name in zip(columns, output_names, strict=False)
            ]
        )
    # operator == 'and'
    # Cumulative AND: all True up to position i (cumsum == i+1)
    return cumsum_R.select(
        [
            pl.col(col_name).eq(i + 1).alias(output_name)
            for i, (col_name, output_name) in enumerate(zip(columns, output_names, strict=False))
        ]
    )


def combine_rules_greedy(
    R: pl.DataFrame,
    y: pl.Series,
    metric: str = "f1",
    max_rules: int = 5,
    operator: str = "or",
    weights: pl.Series | None = None,
    min_improvement: float = 0.0,
) -> pl.DataFrame:
    """Greedily select rules that maximize a performance metric.

    Starts with the best single rule, then iteratively adds rules that provide
    the largest metric improvement. Stops when no rule improves the metric by
    at least min_improvement or when max_rules is reached.

    Parameters
    ----------
    R : pl.DataFrame
        DataFrame containing boolean rule columns. All columns will be
        used as candidate rules.
    y : pl.Series
        Boolean target series indicating true labels.
    metric : str, default="f1"
        Performance metric to optimize. Must be a column name produced by
        compute_metrics (e.g., "f1", "accuracy", "precision", "recall").
    max_rules : int, default=5
        Maximum number of rules to select.
    operator : str, default="or"
        Boolean operator for combining rules: 'or' or 'and'.
    weights : pl.Series | None, default=None
        Optional sample weights for weighted metric computation.
    min_improvement : float, default=0.0
        Minimum metric improvement required to add a new rule.

    Returns
    -------
    pl.DataFrame
        DataFrame with single column containing the combined rule.
        Column name reflects the selected rules using the operator.

    Examples
    --------
    >>> import polars as pl
    >>> R = pl.DataFrame({"rule_A": [True, False, True],
    ...                   "rule_B": [False, True, True],
    ...                   "rule_C": [True, True, False]})
    >>> y = pl.Series([True, True, False])
    >>> result_R = combine_rules_greedy(
    ...     R, y, metric="f1", max_rules=2
    ... )
    >>> print(result_R.columns)  # e.g., ['(rule_B) | (rule_A)']

    Raises
    ------
    ValueError
        If operator is not 'or' or 'and', or if metric column not found.
    """
    if operator not in ["or", "and"]:
        raise ValueError(f"operator must be 'or' or 'and', got '{operator}'")

    rules = R.columns
    if not rules:
        raise ValueError("rules list cannot be empty")

    selected_rules = []
    metric_history = {}
    remaining_rules = rules.copy()
    current_best_metric = float("-inf")

    # Evaluate all single rules to find the best starting point
    metrics_R = compute_metrics(R.select(rules), y, weights)

    # Use weighted metric if weights are provided
    metric_to_use = f"{metric}_weight" if weights is not None else metric

    if metric_to_use not in metrics_R.columns:
        raise ValueError(
            f"Metric '{metric_to_use}' not found in computed metrics. "
            f"Available metrics: {list(metrics_R.columns)}"
        )

    # Select best single rule
    best_idx = metrics_R[metric_to_use].arg_max()
    if best_idx is None:
        raise ValueError("Cannot find best rule - all metrics may be null or equal")

    best_rule = metrics_R["rule"].item(best_idx)
    current_best_metric = metrics_R[metric_to_use].item(best_idx)
    selected_rules.append(best_rule)
    remaining_rules.remove(best_rule)
    metric_history[0] = current_best_metric

    # Iteratively add rules that improve the metric
    for iteration in range(1, max_rules):
        if not remaining_rules:
            break

        best_candidate = None
        best_candidate_metric = current_best_metric

        # Create current combined rule
        if operator == "or":
            current_combined = R[selected_rules[0]]
            for rule in selected_rules[1:]:
                current_combined = current_combined | R[rule]
        else:  # 'and'
            current_combined = R[selected_rules[0]]
            for rule in selected_rules[1:]:
                current_combined = current_combined & R[rule]

        # Try adding each remaining rule
        for candidate_rule in remaining_rules:
            if operator == "or":
                test_combined = current_combined | R[candidate_rule]
            else:  # 'and'
                test_combined = current_combined & R[candidate_rule]

            # Evaluate the combination
            test_R = pl.DataFrame({"test_rule": test_combined})
            test_metrics = compute_metrics(test_R, y, weights)
            candidate_metric = test_metrics[metric_to_use].item(0)

            if candidate_metric > best_candidate_metric:
                best_candidate = candidate_rule
                best_candidate_metric = candidate_metric

        # Check if improvement meets threshold
        improvement = best_candidate_metric - current_best_metric
        if best_candidate is None or improvement < min_improvement:
            break

        # Add the best candidate
        selected_rules.append(best_candidate)
        remaining_rules.remove(best_candidate)
        current_best_metric = best_candidate_metric
        metric_history[iteration] = current_best_metric

    # Create final combined rule
    separator = " | " if operator == "or" else " & "
    combined_rule_name = separator.join(f"({rule})" for rule in selected_rules)

    if operator == "or":
        final_combined = R[selected_rules[0]]
        for rule in selected_rules[1:]:
            final_combined = final_combined | R[rule]
    else:  # 'and'
        final_combined = R[selected_rules[0]]
        for rule in selected_rules[1:]:
            final_combined = final_combined & R[rule]

    result_R = pl.DataFrame({combined_rule_name: final_combined})

    return result_R


def combine_rules_beam_search(
    R: pl.DataFrame,
    y: pl.Series,
    metric: str = "f1",
    beam_width: int = 4,
    max_rules: int = 5,
    operator: str = "or",
    weights: pl.Series | None = None,
    min_improvement: float = 0.0,
    return_top_k: int = 10,
) -> pl.DataFrame:
    """Find top rule combinations using beam search.

    Maintains beam_width best partial combinations at each depth level,
    exploring a broader set of combinations than greedy search while
    remaining more efficient than exhaustive search.

    Parameters
    ----------
    R : pl.DataFrame
        DataFrame containing boolean rule columns. All columns will be
        used as candidate rules.
    y : pl.Series
        Boolean target series indicating true labels.
    metric : str, default="f1"
        Performance metric to optimize. Must be a column name produced by
        compute_metrics (e.g., "accuracy", "f1", "precision", "recall").
    beam_width : int, default=4
        Number of best candidates to keep at each depth level.
    max_rules : int, default=5
        Maximum number of rules in a combination.
    operator : str, default="or"
        Boolean operator for combining rules: 'or' or 'and'.
    weights : pl.Series | None, default=None
        Optional sample weights for weighted metric computation.
    min_improvement : float, default=0.0
        Minimum metric improvement required over parent combination to
        add a new rule. Acts as a pruning criterion to avoid expanding
        combinations that don't provide sufficient benefit.
    return_top_k : int, default=10
        Number of top combinations to return.

    Returns
    -------
    pl.DataFrame
        DataFrame containing columns for the top rule combinations found.
        Each column represents one combination, with the column name showing
        the combined rule expression.

    Examples
    --------
    >>> import polars as pl
    >>> R = pl.DataFrame({"rule_A": [True, False, True],
    ...                   "rule_B": [False, True, True],
    ...                   "rule_C": [True, True, False]})
    >>> y = pl.Series([True, True, False])
    >>> result_R = combine_rules_beam_search(
    ...     R, y, metric="f1", beam_width=3, max_rules=2
    ... )
    >>> print(result_R.columns)  # Shows top rule combinations

    Raises
    ------
    ValueError
        If operator is not 'or' or 'and', or if metric column not found.
    """
    if operator not in ["or", "and"]:
        raise ValueError(f"operator must be 'or' or 'and', got '{operator}'")

    rules = R.columns
    if not rules:
        raise ValueError("rules list cannot be empty")

    separator = " | " if operator == "or" else " & "

    # Initialize beam with all single rules
    # Each entry: (rule_list, metric_value, combined_expr, combined_series)
    beam = []

    for rule in rules:
        rule_series = R[rule]
        metric_value = compute_single_metric(rule_series, y, metric, weights)
        beam.append(([rule], metric_value, rule, rule_series))

    # Sort beam by metric value (descending)
    beam.sort(key=lambda x: x[1], reverse=True)
    beam = beam[:beam_width]

    # Track all explored combinations to avoid duplicates
    all_candidates = beam.copy()

    # Expand beam for each depth level
    for _ in range(max_rules):
        new_beam = []

        for rule_list, parent_metric, _, parent_series in beam:
            # Try adding each rule not already in the combination
            for candidate_rule in rules:
                if candidate_rule in rule_list:
                    continue

                # Extend combination using the cached parent series (no OR-chain replay)
                new_rule_list = rule_list + [candidate_rule]
                new_expr = separator.join(f"({r})" for r in new_rule_list)
                if operator == "or":
                    combined = parent_series | R[candidate_rule]
                else:
                    combined = parent_series & R[candidate_rule]

                # Evaluate only the one needed metric
                metric_value = compute_single_metric(combined, y, metric, weights)

                # Only add if improvement meets threshold
                improvement = metric_value - parent_metric
                if improvement >= min_improvement:
                    new_beam.append((new_rule_list, metric_value, new_expr, combined))

        # Sort and keep top beam_width candidates
        new_beam.sort(key=lambda x: x[1], reverse=True)
        beam = new_beam[:beam_width]

        # Add to all candidates
        all_candidates.extend(beam)

    # Sort all candidates and return top k
    all_candidates.sort(key=lambda x: x[1], reverse=True)

    # Remove duplicates while preserving order
    seen = set()
    unique_candidates = []
    for candidate in all_candidates:
        rule_tuple = tuple(sorted(candidate[0]))
        if rule_tuple not in seen:
            seen.add(rule_tuple)
            unique_candidates.append(candidate)
            if len(unique_candidates) >= return_top_k:
                break

    # Build result DataFrame using cached boolean series (no recomputation)
    result_dict = {rule_expr: combined for _, _, rule_expr, combined in unique_candidates}
    return pl.DataFrame(result_dict)


def combine_rules_a_star(
    R: pl.DataFrame,
    y: pl.Series,
    metric: str = "f1",
    max_rules: int = 5,
    operator: str = "or",
    weights: pl.Series | None = None,
    min_improvement: float = 0.0,
    return_top_k: int = 10,
) -> pl.DataFrame:
    """Find top rule combinations using A* search algorithm.

    Uses A* to efficiently explore the space of rule combinations, finding
    optimal or near-optimal combinations by balancing actual performance (g)
    with estimated potential (h). More thorough than greedy or beam search
    when finding the globally best combination is important.

    A* Cost Function:
        - g(n): Negative metric value (better metrics = lower cost)
        - h(n): Optimistic estimate of best possible improvement from remaining rules
        - f(n): g(n) + h(n) (total estimated cost)

    Parameters
    ----------
    R : pl.DataFrame
        DataFrame containing boolean rule columns. All columns will be
        used as candidate rules.
    y : pl.Series
        Boolean target series indicating true labels.
    metric : str, default="f1"
        Performance metric to optimize. Must be a column name produced by
        compute_metrics (e.g., "f1", "accuracy", "precision", "recall").
    max_rules : int, default=5
        Maximum number of rules in a combination.
    operator : str, default="or"
        Boolean operator for combining rules: 'or' or 'and'.
    weights : pl.Series | None, default=None
        Optional sample weights for weighted metric computation.
    min_improvement : float, default=0.0
        Minimum metric improvement required over parent combination to
        expand a node. Acts as a pruning criterion.
    return_top_k : int, default=10
        Number of top combinations to return. Set to 1 for single best.

    Returns
    -------
    pl.DataFrame
        DataFrame containing columns for the top rule combinations found.
        Each column represents one combination, with the column name showing
        the combined rule expression. Ordered by metric value (best first).

    Examples
    --------
    >>> import polars as pl
    >>> R = pl.DataFrame({"rule_A": [True, False, True],
    ...                   "rule_B": [False, True, True],
    ...                   "rule_C": [True, True, False]})
    >>> y = pl.Series([True, True, False])
    >>> # Find single best combination
    >>> best = combine_rules_a_star(R, y, metric="f1", return_top_k=1)
    >>> # Find top 5 combinations
    >>> top_5 = combine_rules_a_star(R, y, metric="f1", return_top_k=5)

    Raises
    ------
    ValueError
        If operator is not 'or' or 'and', or if metric column not found.

    Notes
    -----
    A* is guaranteed to find the optimal solution if the heuristic is admissible
    (never overestimates the true cost). The heuristic used here estimates the
    best possible improvement from remaining rules, which is optimistic and
    thus admissible.
    """
    if operator not in ["or", "and"]:
        raise ValueError(f"operator must be 'or' or 'and', got '{operator}'")

    rules = R.columns
    if not rules:
        raise ValueError("rules list cannot be empty")

    separator = " | " if operator == "or" else " & "
    metric_to_use = f"{metric}_weight" if weights is not None else metric

    # Precompute metrics for all single rules (for heuristic calculation)
    single_rule_metrics = {}
    metrics_R = compute_metrics(R, y, weights)

    if metric_to_use not in metrics_R.columns:
        raise ValueError(
            f"Metric '{metric_to_use}' not found in computed metrics. "
            f"Available metrics: {list(metrics_R.columns)}"
        )

    # Build a rule→row-index map for O(1) lookups instead of O(n) list.index()
    rule_to_idx = {rule: idx for idx, rule in enumerate(rules)}
    for rule in rules:
        single_rule_metrics[rule] = metrics_R[metric_to_use].item(rule_to_idx[rule])

    # Cache for computed combinations to avoid redundant evaluations
    combination_cache: dict[tuple[str, ...], float] = {}

    def compute_combination_metric(rule_list: list[str]) -> float:
        """Compute metric for a combination, using cache if available."""
        rule_tuple = tuple(sorted(rule_list))

        if rule_tuple in combination_cache:
            return combination_cache[rule_tuple]

        # Compute combined rule
        if operator == "or":
            combined = R[rule_list[0]]
            for r in rule_list[1:]:
                combined = combined | R[r]
        else:  # 'and'
            combined = R[rule_list[0]]
            for r in rule_list[1:]:
                combined = combined & R[r]

        # Evaluate metric
        test_R = pl.DataFrame({"test_rule": combined})
        test_metrics = compute_metrics(test_R, y, weights)
        metric_value = test_metrics[metric_to_use].item(0)

        combination_cache[rule_tuple] = metric_value
        return metric_value

    def heuristic(rule_list: list[str], current_metric: float) -> float:
        """
        Admissible heuristic: optimistic estimate of improvement potential.

        Estimates the best possible improvement by assuming we can achieve
        the maximum single-rule improvement for each remaining slot.
        """
        remaining_rules = [r for r in rules if r not in rule_list]
        if not remaining_rules:
            return 0.0

        # Get the best metrics from remaining rules
        remaining_metrics = [single_rule_metrics[r] for r in remaining_rules]
        remaining_metrics.sort(reverse=True)

        # Optimistic: assume we can improve by the best remaining rule's metric
        # This is optimistic because actual combination might not achieve this
        slots_left = max_rules - len(rule_list)

        if slots_left <= 0:
            return 0.0

        # Take top metrics for remaining slots (optimistic estimate)
        best_remaining = remaining_metrics[:slots_left]
        estimated_improvement = sum(best_remaining) / len(best_remaining) if best_remaining else 0.0

        # Return negative (since we want to maximize metric = minimize negative metric)
        return -estimated_improvement

    # Priority queue: (f_score, counter, g_score, rule_list, rule_expr)
    # Counter ensures FIFO for equal f_scores
    counter = 0
    open_set: list[tuple[float, int, float, list[str], str]] = []

    # Initialize with all single rules
    for rule in rules:
        metric_value = single_rule_metrics[rule]
        g_score = -metric_value  # Negative because we minimize cost
        h_score = heuristic([rule], metric_value)
        f_score = g_score + h_score

        heapq.heappush(open_set, (f_score, counter, g_score, [rule], rule))
        counter += 1

    # Track completed combinations (at max depth or promising ones)
    completed_combinations: list[tuple[float, list[str], str]] = []

    # Track explored states to avoid redundant exploration
    explored: set[tuple[str, ...]] = set()

    # A* main loop
    while open_set:
        f_score, _, g_score, rule_list, rule_expr = heapq.heappop(open_set)

        # Skip if already explored this combination
        rule_tuple = tuple(sorted(rule_list))
        if rule_tuple in explored:
            continue
        explored.add(rule_tuple)

        current_metric = -g_score  # Convert back to positive metric

        # If at max depth, save as completed
        if len(rule_list) >= max_rules:
            completed_combinations.append((current_metric, rule_list[:], rule_expr))
            continue

        # Also save current state as a potential solution
        completed_combinations.append((current_metric, rule_list[:], rule_expr))

        # Expand node: try adding each remaining rule
        for candidate_rule in rules:
            if candidate_rule in rule_list:
                continue

            # Create new combination
            new_rule_list = rule_list + [candidate_rule]
            new_rule_tuple = tuple(sorted(new_rule_list))

            # Skip if already explored
            if new_rule_tuple in explored:
                continue

            # Compute metric for new combination
            new_metric = compute_combination_metric(new_rule_list)

            # Check improvement threshold
            improvement = new_metric - current_metric
            if improvement < min_improvement:
                continue

            # Compute costs
            new_g_score = -new_metric
            new_h_score = heuristic(new_rule_list, new_metric)
            new_f_score = new_g_score + new_h_score

            # Create expression
            new_expr = separator.join(f"({r})" for r in new_rule_list)

            # Add to open set
            heapq.heappush(open_set, (new_f_score, counter, new_g_score, new_rule_list, new_expr))
            counter += 1

    # Sort completed combinations by metric (descending)
    completed_combinations.sort(key=lambda x: x[0], reverse=True)

    # Remove duplicates while preserving order
    seen = set()
    unique_combinations = []
    for metric_val, rule_list, rule_expr in completed_combinations:
        rule_tuple = tuple(sorted(rule_list))
        if rule_tuple not in seen:
            seen.add(rule_tuple)
            unique_combinations.append((metric_val, rule_list, rule_expr))
            if len(unique_combinations) >= return_top_k:
                break

    # Build result DataFrame
    result_dict = {}
    for _, rule_list, rule_expr in unique_combinations:
        # Compute the combined rule
        if operator == "or":
            combined = R[rule_list[0]]
            for r in rule_list[1:]:
                combined = combined | R[r]
        else:  # 'and'
            combined = R[rule_list[0]]
            for r in rule_list[1:]:
                combined = combined & R[r]

        result_dict[rule_expr] = combined

    return pl.DataFrame(result_dict)
