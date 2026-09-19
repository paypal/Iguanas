import heapq
import itertools

import polars as pl

from .metrics import compute_metrics, compute_single_metric


def _metric_col(metric: str, weights: pl.Series | None) -> str:
    """Return the metric column name, appending ``'_weight'`` when weights are used."""
    return f"{metric}_weight" if weights is not None else metric


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
    max_combinations_per_n : int, default=200_000
        Maximum number of combinations to generate per combination size.
        If exceeded, only the first max_combinations_per_n are used.
    batch_size : int, default=50_000
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
        combos = list(itertools.islice(itertools.combinations(rules, combo_size), max_combinations_per_n))
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


def combine_rules_budgeted(
    R: pl.DataFrame,
    y: pl.Series,
    max_alert_rate: float,
    max_rules: int = 5,
    weights: pl.Series | None = None,
) -> pl.DataFrame:
    """Build the OR-ruleset that catches the most positives within an alert budget.

    Operational rule systems are constrained by review capacity, not by F1: only
    a fixed fraction of the population can be flagged. This selects a disjunction
    maximising recall subject to that constraint, which is the *budgeted maximum
    coverage* problem. The greedy rule below is its standard
    ``1 - 1/e`` approximation.

    This differs from :func:`combine_rules_greedy` in what it optimises. Greedy
    metric maximisation ignores coverage, so under OR it tends to keep widening
    the ruleset until it flags far more than the budget permits; here a candidate
    is simply infeasible once the union breaches the budget.

    Parameters
    ----------
    R : pl.DataFrame
        DataFrame containing boolean rule columns. All columns are candidates.
    y : pl.Series
        Boolean target series indicating true labels.
    max_alert_rate : float
        Maximum fraction of the population the ruleset may flag, in (0, 1].
        With *weights*, this is a fraction of total weight rather than of rows.
    max_rules : int, default=5
        Maximum number of rules in the returned disjunction.
    weights : pl.Series | None, default=None
        Optional sample weights. When given, both the budget and the positives
        caught are measured in weight rather than row counts.

    Returns
    -------
    pl.DataFrame
        Single boolean column named by the combined rule expression. Empty when
        no single rule fits within the budget.

    Raises
    ------
    ValueError
        If *max_alert_rate* is outside (0, 1], or R has no columns.

    Examples
    --------
    >>> import polars as pl
    >>> R = pl.DataFrame({"a": [True, False, False], "b": [False, True, False]})
    >>> y = pl.Series([True, True, False])
    >>> ruleset = combine_rules_budgeted(R, y, max_alert_rate=0.7)

    Notes
    -----
    Coverage is monotone under OR, so a rule that breaches the budget can never
    be rescued by adding more rules. Each step therefore takes the feasible rule
    with the greatest *marginal* gain in positives caught, breaking ties toward
    the cheaper rule, and stops as soon as nothing feasible adds a positive.
    """
    if not 0.0 < max_alert_rate <= 1.0:
        raise ValueError(f"max_alert_rate must be in (0, 1], got {max_alert_rate}")
    rules = R.columns
    if not rules:
        raise ValueError("rules list cannot be empty")

    y_bool = y.cast(pl.Boolean)
    cols = [R[r].cast(pl.Boolean) for r in rules]
    total = float(len(y_bool)) if weights is None else float(weights.sum())
    budget = max_alert_rate * total

    def mass(mask: pl.Series) -> float:
        return float(mask.sum()) if weights is None else float(weights.filter(mask).sum())

    covered = pl.repeat(False, R.height, eager=True)
    chosen: list[int] = []
    caught = 0.0

    while len(chosen) < max_rules:
        best: tuple[tuple[float, float], int, pl.Series, float] | None = None
        for j in range(len(rules)):
            if j in chosen:
                continue
            union = covered | cols[j]
            cost = mass(union)
            if cost > budget:
                continue
            gain = mass(y_bool & union) - caught
            if gain <= 0:
                continue
            key = (gain, -cost)
            if best is None or key > best[0]:
                best = (key, j, union, cost)
        if best is None:
            break
        (gain, _), j, union, _ = best
        chosen.append(j)
        covered = union
        caught += gain

    if not chosen:
        return pl.DataFrame()

    if len(chosen) == 1:
        expression = rules[chosen[0]]
    else:
        expression = " | ".join(f"({rules[j]})" for j in chosen)
    return pl.DataFrame({expression: covered})


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
    remaining_rules = rules.copy()
    current_best_metric = float("-inf")

    # Evaluate all single rules to find the best starting point
    metrics_R = compute_metrics(R.select(rules), y, weights)

    # Use weighted metric if weights are provided
    metric_to_use = _metric_col(metric, weights)

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

    # Maintain the running combined Series incrementally to avoid rebuilding
    # from scratch on every iteration (O(k) total ops instead of O(k²)).
    current_combined = R[best_rule]

    # Iteratively add rules that improve the metric
    for _ in range(1, max_rules):
        if not remaining_rules:
            break

        best_candidate = None
        best_candidate_metric = current_best_metric

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

        # Add the best candidate and update the running combined Series
        if operator == "or":
            current_combined = current_combined | R[best_candidate]
        else:
            current_combined = current_combined & R[best_candidate]
        selected_rules.append(best_candidate)
        remaining_rules.remove(best_candidate)
        current_best_metric = best_candidate_metric

    separator = " | " if operator == "or" else " & "
    combined_rule_name = separator.join(f"({rule})" for rule in selected_rules)
    result_R = pl.DataFrame({combined_rule_name: current_combined})

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
    evaluating far fewer than exhaustive enumeration. Like greedy search this
    is a heuristic: it offers no optimality guarantee. Use
    :func:`combine_rules_a_star` when the top-k must be provably optimal.

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

    # Expand beam for each depth level. The initial beam already holds size-1
    # combinations, so only max_rules - 1 more rules may be added to reach a
    # combination of size max_rules (mirrors combine_rules_greedy's
    # range(1, max_rules) after its own initial single-rule pick).
    for _ in range(max_rules - 1):
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
    min_improvement: float | None = None,
    return_top_k: int = 10,
    protected: pl.Series | None = None,
    reference_group: object | None = None,
    min_dir: float | None = None,
    max_alert_rate: float | None = None,
    return_diagnostics: bool = False,
) -> pl.DataFrame | tuple[pl.DataFrame, dict]:
    """Find the top rule combinations by best-first branch-and-bound search.

    Searches the space of rule subsets containing between 1 and ``max_rules``
    rules, and returns the ``return_top_k`` best-scoring combinations. Unlike
    greedy or beam search, this returns a *provably* optimal top-k under the
    conditions in Notes.

    Search formulation
    ------------------
    Nodes are rule *subsets*; children extend a subset by one rule. Each node
    carries an **upper bound** on the metric attainable by any descendant. The
    frontier is ordered by that bound (best-first), and a node is discarded when
    its bound cannot beat the current k-th best incumbent. Search stops as soon
    as the best remaining bound falls below the k-th incumbent, at which point
    no unexplored subset can enter the top-k.

    Parameters
    ----------
    R : pl.DataFrame
        DataFrame containing boolean rule columns. All columns are candidates.
    y : pl.Series
        Boolean target series indicating true labels.
    metric : str, default="f1"
        Metric to maximise: "precision", "recall", "accuracy", "mcc", or an
        F-beta score ("f1", "f0.5", "f2", ...).
    max_rules : int, default=5
        Maximum number of rules in a combination.
    operator : str, default="or"
        Boolean operator for combining rules: 'or' or 'and'.
    weights : pl.Series | None, default=None
        Optional sample weights for weighted metric computation.
    min_improvement : float | None, default=None
        If set, a child is discarded unless it improves on its parent by at
        least this much. This is a *heuristic* filter that *forfeits the
        optimality guarantee* (the best subset may only be reachable through a
        temporarily worse ancestor -- under OR-composition, adding a rule
        typically lowers precision before later rules raise recall). Leave as
        None for exact search.
    return_top_k : int, default=10
        Number of top combinations to return. Set to 1 for the single best.
    protected : pl.Series | None, default=None
        Optional protected-attribute series used to enforce a fairness
        constraint on returned combinations.
    reference_group : object | None, default=None
        Value of *protected* to treat as the reference group. Defaults to the
        most frequent value.
    min_dir : float | None, default=None
        Minimum acceptable disparate impact ratio. When set (together with
        *protected*), a combination is only admitted to the results if its DIR
        is at least this value; 0.8 corresponds to the "four-fifths rule".
        Search remains exact over the feasible set.
    max_alert_rate : float | None, default=None
        If set, only combinations flagging at most this fraction of the
        population are admitted. This makes the search the *exact* counterpart
        of :func:`combine_rules_budgeted`, whose greedy rule is a ``1 - 1/e``
        approximation, so the two together measure the approximation gap.
    return_diagnostics : bool, default=False
        If True, return ``(results, diagnostics)`` where diagnostics reports
        node counts and whether the optimality guarantee held.

    Returns
    -------
    pl.DataFrame
        One column per returned combination, named by the combined rule
        expression, ordered best-first.
    dict
        Only when *return_diagnostics* is True. Keys: ``nodes_expanded``,
        ``nodes_pruned``, ``combinations_evaluated``, ``bounds_computed``,
        ``exact``, ``bound_is_tight``.

    Examples
    --------
    >>> import polars as pl
    >>> R = pl.DataFrame({"rule_A": [True, False, True],
    ...                   "rule_B": [False, True, True],
    ...                   "rule_C": [True, True, False]})
    >>> y = pl.Series([True, True, False])
    >>> best = combine_rules_a_star(R, y, metric="f1", return_top_k=1)
    >>> top_5, diag = combine_rules_a_star(R, y, return_top_k=5,
    ...                                    return_diagnostics=True)

    Raises
    ------
    ValueError
        If operator is not 'or' or 'and', or the metric is unsupported.

    Notes
    -----
    **Optimality.** The returned top-k is exact when ``min_improvement`` is None
    and the metric admits a non-trivial bound (see below). Exactness holds over
    the feasible set when a fairness constraint is active.

    **Why a bound rather than an A\\* heuristic.** Metric value is a property of
    a *state*, not a sum of edge costs, so classical A\\* ``f = g + h`` does not
    apply: there is no additive path cost to decompose. This routine is
    therefore a best-first branch-and-bound, which is the correct formulation
    for maximising a non-additive set function. The public name is retained for
    backwards compatibility.

    **Admissibility.** Under OR-composition, coverage is monotonically
    non-decreasing in the rule set: TP and FP can only grow and FN can only
    shrink. Writing ``P`` for total positive mass and ``k`` for the remaining
    slots, an attainable-TP upper bound follows from the union bound,
    ``TP_max = TP + (sum of the k largest per-rule new-TP masses)``, clipped at
    ``P``, while ``FP`` is bounded below by its current value. Every metric
    below is non-decreasing in TP and non-increasing in FP, so evaluating it at
    ``(TP_max, FP_min)`` maximises it over the reachable region and never
    understates what a descendant can achieve:

    - ``recall = TP / P``
    - ``precision = TP / (TP + FP)``
    - ``accuracy = (TP + (N - P) - FP) / N``
    - ``f_beta = (1 + b^2) TP / (TP + b^2 P + FP)``

    The F-beta identity uses ``FN = P - TP`` to eliminate FN, which is what
    makes the bound monotone in only two quantities. Under AND-composition the
    monotonicity reverses -- coverage shrinks, so TP is bounded above by its
    current value and FP is bounded below by subtracting the k largest
    per-rule FP removals -- and the same four expressions apply.

    ``mcc`` has no non-trivial bound implemented and falls back to 1.0. That is
    admissible, so the result is still exact, but no pruning occurs and the
    search degenerates to exhaustive enumeration.

    **Alert budget.** Under OR the constraint is *monotone*: coverage only grows,
    so a node already over budget can never be rescued and its entire subtree is
    discarded -- the constraint makes the search cheaper, not more expensive. It
    also tightens the bound, since any rows a descendant adds consume the
    remaining capacity, capping attainable TP at ``budget - coverage``. Under AND
    coverage shrinks, so an over-budget node may still have feasible descendants;
    there it is excluded from the results but still expanded.

    **Prior behaviour.** Before v1.4 this function ordered the frontier by
    ``-mean(best remaining single-rule metrics)``, which is *not* admissible:
    for disjoint rules under OR the achievable recall gain is the *sum* of
    per-rule gains, so a mean understates it and the optimum could be ordered
    away. It also defaulted to ``min_improvement=0.0``, discarding every
    non-improving child, which reduced the search to hill-climbing. Both are
    fixed here; ``min_improvement`` now defaults to None.
    """
    if operator not in ["or", "and"]:
        raise ValueError(f"operator must be 'or' or 'and', got '{operator}'")
    if max_alert_rate is not None and not 0.0 < max_alert_rate <= 1.0:
        raise ValueError(f"max_alert_rate must be in (0, 1], got {max_alert_rate}")

    rules = R.columns
    if not rules:
        raise ValueError("rules list cannot be empty")

    separator = " | " if operator == "or" else " & "
    n_rules = len(rules)
    rule_cols = [R[r].cast(pl.Boolean) for r in rules]

    y_bool = y.cast(pl.Boolean)
    w = weights
    if w is None:
        total_mass = float(len(y_bool))
        positive_mass = float(y_bool.sum())
    else:
        total_mass = float(w.sum())
        positive_mass = float(w.filter(y_bool).sum())
    negative_mass = total_mass - positive_mass

    # Fail fast on an unsupported metric rather than deep inside the search.
    compute_single_metric(rule_cols[0], y_bool, metric, w)

    def _is_fbeta(name: str) -> bool:
        if len(name) < 2 or not name.startswith("f"):
            return False
        try:
            float(name[1:])
        except ValueError:
            return False
        return True

    has_bound = metric in ("precision", "recall", "accuracy") or _is_fbeta(metric)
    budget = None if max_alert_rate is None else max_alert_rate * total_mass

    def _mass(mask: pl.Series) -> float:
        return float(mask.sum()) if w is None else float(w.filter(mask).sum())

    def _confusion(pred: pl.Series) -> tuple[float, float]:
        return _mass(y_bool & pred), _mass(~y_bool & pred)

    def _metric_ceiling(tp_max: float, fp_min: float) -> float:
        """Metric evaluated at the corner of the reachable (TP, FP) region.

        Every metric below is non-decreasing in TP and non-increasing in FP, so
        this maximises it over the region and is a valid upper bound.
        """
        if metric == "recall":
            return tp_max / positive_mass if positive_mass > 0 else 0.0
        if metric == "precision":
            denom = tp_max + fp_min
            return tp_max / denom if denom > 0 else 0.0
        if metric == "accuracy":
            return (tp_max + negative_mass - fp_min) / total_mass if total_mass > 0 else 0.0
        if _is_fbeta(metric):
            b2 = float(metric[1:]) ** 2
            denom = tp_max + b2 * positive_mass + fp_min
            return (1.0 + b2) * tp_max / denom if denom > 0 else 0.0
        return 1.0  # no non-trivial bound (e.g. mcc): admissible, but no pruning

    def _upper_bound(
        pred: pl.Series, tp: float, fp: float, remaining: range, slots: int
    ) -> float:
        """Upper bound on the metric over every descendant of this node."""
        if slots <= 0 or not remaining:
            return _metric_ceiling(tp, fp)
        if operator == "or":
            # Coverage only grows: FP is floored at its current value, and the
            # union bound caps the TP mass the remaining slots can still add.
            gains = sorted(
                (_mass(y_bool & rule_cols[j] & ~pred) for j in remaining), reverse=True
            )
            addable = sum(gains[:slots])
            if budget is not None:
                # Rows a descendant adds consume the remaining alert capacity,
                # so attainable TP cannot exceed what the budget still allows.
                addable = min(addable, max(0.0, budget - (tp + fp)))
            return _metric_ceiling(min(positive_mass, tp + addable), fp)
        # Coverage only shrinks: TP is capped at its current value and FP can
        # fall by at most the sum of the largest per-rule FP removals.
        drops = sorted((_mass(~y_bool & pred & ~rule_cols[j]) for j in remaining), reverse=True)
        return _metric_ceiling(tp, max(0.0, fp - sum(drops[:slots])))

    group_masks: dict[object, pl.Series] = {}
    dir_threshold: float | None = None
    if protected is not None and min_dir is not None:
        dir_threshold = min_dir
        if reference_group is None:
            reference_group = protected.value_counts(sort=True).row(0)[0]
        group_masks = {g: (protected == g) for g in protected.unique().to_list()}
        if reference_group not in group_masks:
            raise ValueError(f"reference_group {reference_group!r} not present in protected")

    def _subgroup_metric(pred: pl.Series, mask: pl.Series) -> float:
        return compute_single_metric(
            pred.filter(mask),
            y_bool.filter(mask),
            metric,
            None if w is None else w.filter(mask),
        )

    def _dir_ok(pred: pl.Series) -> bool:
        if dir_threshold is None:
            return True
        reference = _subgroup_metric(pred, group_masks[reference_group])
        if reference <= 0:
            return False
        return all(
            _subgroup_metric(pred, mask) / reference >= dir_threshold
            for group, mask in group_masks.items()
            if group != reference_group
        )

    def _within_budget(tp: float, fp: float) -> bool:
        return budget is None or (tp + fp) <= budget

    counter = 0
    nodes_expanded = 0
    nodes_pruned = 0
    combinations_evaluated = 0
    bounds_computed = 0

    # Frontier ordered by upper bound, descending (negated for heapq).
    frontier: list[tuple[float, int, tuple[int, ...]]] = []
    node_state: dict[tuple[int, ...], tuple[pl.Series, float, float, float]] = {}
    # Incumbent top-k as a min-heap, so incumbents[0] is the one to beat.
    incumbents: list[tuple[float, int, tuple[int, ...]]] = []

    def _threshold() -> float:
        return incumbents[0][0] if len(incumbents) >= return_top_k else float("-inf")

    def _offer(idx_tuple: tuple[int, ...], pred: pl.Series, tp: float, fp: float, value: float):
        """Bound the node and admit it to the frontier unless it cannot compete."""
        nonlocal counter, nodes_pruned, bounds_computed
        # Under OR the budget is monotone, so an over-budget node's whole subtree
        # is unreachable and can be discarded outright.
        if operator == "or" and not _within_budget(tp, fp):
            nodes_pruned += 1
            return
        remaining = range(idx_tuple[-1] + 1, n_rules)
        bound = _upper_bound(pred, tp, fp, remaining, max_rules - len(idx_tuple))
        bounds_computed += 1
        if bound <= _threshold():
            nodes_pruned += 1
            return
        node_state[idx_tuple] = (pred, tp, fp, value)
        heapq.heappush(frontier, (-bound, counter, idx_tuple))
        counter += 1

    for i in range(n_rules):
        pred = rule_cols[i]
        tp, fp = _confusion(pred)
        combinations_evaluated += 1
        _offer((i,), pred, tp, fp, compute_single_metric(pred, y_bool, metric, w))

    while frontier:
        neg_bound, _, idx_tuple = heapq.heappop(frontier)
        # The frontier is bound-ordered, so once the best remaining bound cannot
        # beat the k-th incumbent, no unexplored subset ever will.
        if -neg_bound <= _threshold():
            break
        pred, tp, fp, value = node_state.pop(idx_tuple)
        nodes_expanded += 1

        if _within_budget(tp, fp) and _dir_ok(pred):
            heapq.heappush(incumbents, (value, counter, idx_tuple))
            counter += 1
            if len(incumbents) > return_top_k:
                heapq.heappop(incumbents)

        if len(idx_tuple) >= max_rules:
            continue

        # Children only ever append a higher index, so each subset is reached once.
        for j in range(idx_tuple[-1] + 1, n_rules):
            child_pred = (pred | rule_cols[j]) if operator == "or" else (pred & rule_cols[j])
            child_value = compute_single_metric(child_pred, y_bool, metric, w)
            combinations_evaluated += 1
            if min_improvement is not None and child_value - value < min_improvement:
                nodes_pruned += 1
                continue
            child_tp, child_fp = _confusion(child_pred)
            _offer((*idx_tuple, j), child_pred, child_tp, child_fp, child_value)

    result_dict: dict[str, pl.Series] = {}
    for _, _, idx_tuple in sorted(incumbents, key=lambda item: item[0], reverse=True):
        combined = rule_cols[idx_tuple[0]]
        for j in idx_tuple[1:]:
            combined = (combined | rule_cols[j]) if operator == "or" else (combined & rule_cols[j])
        if len(idx_tuple) == 1:
            expr = rules[idx_tuple[0]]
        else:
            expr = separator.join(f"({rules[j]})" for j in idx_tuple)
        result_dict[expr] = combined

    results = pl.DataFrame(result_dict)
    if not return_diagnostics:
        return results
    return results, {
        "nodes_expanded": nodes_expanded,
        "nodes_pruned": nodes_pruned,
        "combinations_evaluated": combinations_evaluated,
        "bounds_computed": bounds_computed,
        "exact": min_improvement is None,
        "bound_is_tight": has_bound,
    }

