import re

# Pre-compiled regex pattern used by simplify_rule and rule_to_sql.
_COND_PATTERN = re.compile(r'\(X\["([^"]+)"\]\s*([><=!]+)\s*([^\)]+)\)')


def simplify_rule(rule: str) -> str:
    """Simplify a rule by removing redundant conditions on the same column.

    When multiple conditions exist on the same column, keeps only the most restrictive:

    - For lower bounds (>, >=): keeps the highest threshold, preferring > over >= when equal
    - For upper bounds (<, <=): keeps the lowest threshold, preferring < over <= when equal

    Parameters
    ----------
    rule : str
        Rule string with conditions like (X["col"] > val) & (X["col"] >= val).

    Returns
    -------
    str
        Simplified rule string with redundant conditions removed.
        Column order is preserved based on first appearance.

    Examples
    --------
    >>> simplify_rule('(X["amount"] >= 100.0) & (X["amount"] > 100.0)')
    '(X["amount"] > 100.0)'

    >>> simplify_rule('(X["amount"] < 100.0) & (X["amount"] <= 100.0)')
    '(X["amount"] < 100.0)'

    >>> simplify_rule('(X["a"] >= 50) & (X["b"] < 10) & (X["a"] > 100)')
    '(X["a"] > 100) & (X["b"] < 10)'
    """
    # Find all conditions with their full match
    matches = [
        (m.group(0), m.group(1), m.group(2), m.group(3)) for m in _COND_PATTERN.finditer(rule)
    ]

    if not matches:
        return rule

    # Track column order based on first appearance
    column_order: list[str] = []

    # Group conditions by column
    column_conditions: dict[str, list[tuple[str, str, float | None, str]]] = {}
    for full_match, col, op, val in matches:
        if col not in column_conditions:
            column_conditions[col] = []
            column_order.append(col)  # Track first appearance
        try:
            numeric_val = float(val.strip())
            column_conditions[col].append((full_match, op, numeric_val, val.strip()))
        except ValueError:
            # Non-numeric value, keep as is
            column_conditions[col].append((full_match, op, None, val.strip()))

    # Determine which conditions to remove
    conditions_to_remove: set[str] = set()

    for _, conds in column_conditions.items():
        if len(conds) <= 1:
            continue

        # Separate by operator type (only numeric values)
        greater_conds = [
            (full, op, num_val, val)
            for full, op, num_val, val in conds
            if op in (">", ">=") and num_val is not None
        ]
        less_conds = [
            (full, op, num_val, val)
            for full, op, num_val, val in conds
            if op in ("<", "<=") and num_val is not None
        ]

        # For greater/greater-equal: keep only the most restrictive (highest value)
        if len(greater_conds) > 1:
            # Find max value and keeper in single pass
            max_val = max(num_val for _, _, num_val, _ in greater_conds)
            # Among max values, prefer > over >=
            keeper: tuple[str, str] | None = None
            for full, op, num_val, _ in greater_conds:
                if num_val == max_val:
                    if keeper is None or (op == ">" and keeper[1] == ">="):
                        keeper = (full, op)

            assert keeper is not None
            # Mark all others for removal (only iterate once)
            conditions_to_remove.update(
                full for full, _, _, _ in greater_conds if full != keeper[0]
            )

        # For less/less-equal: keep only the most restrictive (lowest value)
        if len(less_conds) > 1:
            # Find min value and keeper in single pass
            min_val = min(num_val for _, _, num_val, _ in less_conds)
            # Among min values, prefer < over <=
            less_keeper: tuple[str, str] | None = None
            for full, op, num_val, _ in less_conds:
                if num_val == min_val:
                    if less_keeper is None or (op == "<" and less_keeper[1] == "<="):
                        less_keeper = (full, op)

            assert less_keeper is not None
            # Mark all others for removal (only iterate once)
            conditions_to_remove.update(
                full for full, _, _, _ in less_conds if full != less_keeper[0]
            )

    result_conditions = [
        full
        for col in column_order
        for full, op, num_val, val in column_conditions[col]
        if full not in conditions_to_remove
    ]

    return " & ".join(result_conditions)


# SQL operator mapping: Python == becomes SQL =
_SQL_OP_MAP: dict[str, str] = {"==": "="}


def rule_to_sql(rule: str, table_alias: str | None = None) -> str:
    """Convert a rule expression string to a SQL WHERE clause.

    Translates Iguanas rule notation (``X["col"] op value``) into standard
    SQL predicate syntax suitable for use in a ``WHERE`` or
    ``CASE WHEN`` clause.

    Parameters
    ----------
    rule : str
        Rule expression using ``X["col"]`` notation with ``&`` / ``|``
        operators, e.g. ``'(X["age"] > 30) & (X["income"] < 50000)'``.
    table_alias : str | None, default=None
        Optional table or CTE alias to prefix column references with.
        For example, ``table_alias="t"`` turns ``age > 30`` into
        ``t.age > 30``.

    Returns
    -------
    str
        SQL WHERE clause string.

    Examples
    --------
    >>> rule_to_sql('(X["age"] > 30) & (X["income"] < 50000)')
    '(age > 30.0) AND (income < 50000.0)'

    >>> rule_to_sql('(X["age"] > 30) | (X["flag"] == 1)', table_alias="t")
    '(t.age > 30.0) OR (t.flag = 1.0)'
    """

    def _cond_to_sql(m: re.Match) -> str:
        feature, op, val = m.group(1), m.group(2), m.group(3).strip()
        col_ref = f"{table_alias}.{feature}" if table_alias else feature
        sql_op = _SQL_OP_MAP.get(op, op)
        try:
            val_sql = str(float(val))
        except ValueError:
            val_sql = f"'{val}'"
        return f"({col_ref} {sql_op} {val_sql})"

    sql = _COND_PATTERN.sub(_cond_to_sql, rule)
    sql = re.sub(r"\s*&\s*", " AND ", sql)
    sql = re.sub(r"\s*\|\s*", " OR ", sql)
    return sql
