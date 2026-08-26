"""
Rule formatting utilities for iguanas.

Alongside simplification/SQL conversion, this module provides composable
functions for turning encoded-feature rules (produced by any preprocessing
library, e.g. gators) back into human-readable rules on the original columns.
Each function takes a plain dict/list mapping and a rule string - there's no
dependency on, or knowledge of, any specific encoder's internals.

gators transformer -> rule_formatting function (None if not reversible)
------------------------------------------------------------------------

    GATORS_TRANSFORMER_TO_FUNCTION = {
        # imputers - value None means the imputed value depends on other
        # rows/columns, so no static {col: value} mapping can be built
        "BooleanImputer": "add_missing_value_conditions",
        "GroupByImputer": None,
        "IterativeImputer": None,
        "KNNImputer": None,
        "NumericImputer": "add_missing_value_conditions",
        "StringImputer": "decode_string_imputation",
        # encoders
        "BinaryEncoder": None,  # category split across multiple bit columns, needs joint decoding
        "CatBoostEncoder": "decode_numeric_encodings",
        "CountEncoder": "decode_numeric_encodings",
        "HashEncoder": None,  # hashing is lossy/many-to-one, not invertible
        "LeaveOneOutEncoder": "decode_numeric_encodings",
        "OneHotEncoder": "decode_onehot_encodings",
        "OrdinalEncoder": "decode_numeric_encodings",
        "RareCategoryEncoder": None,  # no output of its own; merge its groups into the downstream encoder's mapping
        "TargetEncoder": "decode_numeric_encodings",
        "WOEEncoder": "decode_numeric_encodings",
        # discretizers
        "CustomDiscretizer": "decode_discretized_bins",
        "EqualLengthDiscretizer": "decode_discretized_bins",
        "EqualSizeDiscretizer": "decode_discretized_bins",
        "GeometricDiscretizer": "decode_discretized_bins",
        "KMeansDiscretizer": "decode_discretized_bins",
        "QuantileDiscretizer": "decode_discretized_bins",
        "TreeBasedDiscretizer": "decode_discretized_bins",
        # scalers - decode_scaled_thresholds takes the scaler's inverse_transform,
        # so it works for any strictly monotonic scaler, not just these
        "ArcSinSquareRootScaler": "decode_scaled_thresholds",
        "ArcSinhScaler": "decode_scaled_thresholds",
        "BoxCox": "decode_scaled_thresholds",
        "Log1pScaler": "decode_scaled_thresholds",
        "MinmaxScaler": "decode_scaled_thresholds",
        "PowerScaler": "decode_scaled_thresholds",
        "RobustScaler": "decode_scaled_thresholds",
        "StandardScaler": "decode_scaled_thresholds",
        "YeoJohnson": "decode_scaled_thresholds",
        # feature generation
        "IsNull": "decode_null_indicators",
    }

Typical usage after fitting a gators preprocessing pipeline and generating
rules with iguanas:

    from functools import partial
    from iguanas.rule_formatting import (
        add_missing_value_conditions,
        decode_numeric_encodings,
        decode_onehot_encodings,
        format_as_boolean_conditions,
        format_floats_as_integers,
        prettify_rules,
        round_thresholds,
    )

    # WOEEncoder.mapping_ is already {col: {category: woe_score}}
    woe_mapping = pipe["WOEEncoder"].mapping_

    # OneHotEncoder.categories -> {encoded_col: (original_col, category)}
    ohe_mapping = {
        f"{col}__{cat}": (col, cat)
        for col, cats in pipe["OneHotEncoder"].categories.items()
        for cat in cats
    }

    # NumericImputer._statistics is empty for min/max/mean strategies, so
    # build the imputed-value dict manually from the training data instead
    imputation_dict = dict(zip(
        pipe["NumericImputerMean"].subset,
        X_train[pipe["NumericImputerMean"].subset].mean().row(0),
    ))

    steps = [
        partial(decode_numeric_encodings, mapping=woe_mapping),
        partial(decode_onehot_encodings, mapping=ohe_mapping, null_category="MISSING"),
        partial(add_missing_value_conditions, mapping=imputation_dict),
        partial(format_floats_as_integers, int_columns=int_columns),
        partial(format_as_boolean_conditions, bool_columns=bool_columns),
        partial(round_thresholds, columns=numeric_columns, ndigits=2),
    ]
    pretty_rules = prettify_rules(raw_rules, steps)
"""

import math
import re
from collections.abc import Callable

# Pre-compiled regex pattern used by all condition-parsing functions below.
# Accepts single or double quotes around the column name (output always uses double quotes).
_COND_PATTERN = re.compile(r'''\(X\[["']([^"']+)["']\]\s*([><=!]+)\s*([^\)]+)\)''')


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

    def _cond_to_sql(m: re.Match[str]) -> str:
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


def format_floats_as_integers(rule: str, int_columns: list[str]) -> str:
    """Convert float thresholds to integers for the given columns.

    Uses ceiling for ``>=``/``<`` and floor for ``>``/``<=``, so the integer
    boundary preserves the original condition's semantics. ``==``/``!=`` are
    left unchanged since there is no boundary to round.

    Parameters
    ----------
    rule : str
        Rule expression using ``X["col"]`` notation.
    int_columns : list[str]
        Columns whose float thresholds should be converted to integers.

    Returns
    -------
    str
        Rule string with integer thresholds for the given columns.

    Examples
    --------
    >>> format_floats_as_integers('(X["a"] >= 0.1) & (X["b"] >= 9.1)', ["a"])
    '(X["a"] >= 1) & (X["b"] >= 9.1)'
    """

    def _convert(m: re.Match[str]) -> str:
        col, op, val = m.group(1), m.group(2), m.group(3).strip()
        if col not in int_columns or op not in (">=", ">", "<=", "<"):
            return m.group(0)
        try:
            float_val = float(val)
        except ValueError:
            return m.group(0)
        int_val = math.ceil(float_val) if op in (">=", "<") else math.floor(float_val)
        return f'(X["{col}"] {op} {int_val})'

    return _COND_PATTERN.sub(_convert, rule)


def add_missing_value_conditions(rule: str, mapping: dict[str, float]) -> str:
    """Append an ``is_null()`` clause to conditions satisfied by an imputed value.

    When a column's nulls were filled with a value that also satisfies an
    existing condition, that condition implicitly matches originally-null
    rows too. This makes that explicit by OR-ing in ``X[col].is_null()``.

    Parameters
    ----------
    rule : str
        Rule expression using ``X["col"]`` notation.
    mapping : dict[str, float]
        Maps column name to the value nulls were imputed with.

    Returns
    -------
    str
        Rule string with ``is_null()`` clauses added where relevant.

    Examples
    --------
    >>> add_missing_value_conditions('(X["a"] < 1)', {"a": 0})
    '(X["a"] < 1 | X["a"].is_null())'
    """
    _OPS = {
        ">=": lambda a, b: a >= b,
        ">": lambda a, b: a > b,
        "<=": lambda a, b: a <= b,
        "<": lambda a, b: a < b,
        "==": lambda a, b: a == b,
        "!=": lambda a, b: a != b,
    }

    def _convert(m: re.Match[str]) -> str:
        col, op, val = m.group(1), m.group(2), m.group(3).strip()
        if col not in mapping or op not in _OPS:
            return m.group(0)
        try:
            threshold = float(val)
        except ValueError:
            return m.group(0)
        if not _OPS[op](mapping[col], threshold):
            return m.group(0)
        return f'(X["{col}"] {op} {val} | X["{col}"].is_null())'

    return _COND_PATTERN.sub(_convert, rule)


def decode_string_imputation(rule: str, mapping: dict[str, str]) -> str:
    """Convert an equality on a string-imputed placeholder into ``is_null()``.

    For columns where nulls were filled with a placeholder string (e.g.
    gators ``StringImputer``'s default ``"MISSING"``), rewrites an equality
    condition on that placeholder back to an is-null check.

    Parameters
    ----------
    rule : str
        Rule expression using ``X["col"]`` notation.
    mapping : dict[str, str]
        Maps column name to the placeholder value strings were imputed with.

    Returns
    -------
    str
        Rule string with placeholder equality conditions decoded.

    Examples
    --------
    >>> decode_string_imputation('(X["status"] == "MISSING")', {"status": "MISSING"})
    'X["status"].is_null()'
    """

    def _convert(m: re.Match[str]) -> str:
        col, op, val = m.group(1), m.group(2), m.group(3).strip()
        if col not in mapping or op not in ("==", "!="):
            return m.group(0)
        if val.strip("\"'") != mapping[col]:
            return m.group(0)
        is_null = op == "=="
        return f'X["{col}"].is_null()' if is_null else f'(~X["{col}"].is_null())'

    return _COND_PATTERN.sub(_convert, rule)


def decode_numeric_encodings(rule: str, mapping: dict[str, dict[str, float]]) -> str:
    """Reverse a numeric category encoding back to the original category labels.

    For encodings where each category maps to a numeric statistic (e.g. WOE
    score, category count, mean target value), finds which categories satisfy
    the condition's operator/threshold and replaces it with an equality (single
    match) or ``.is_in()`` (multiple matches) condition on the original values.

    Parameters
    ----------
    rule : str
        Rule expression using ``X["col"]`` notation, where col holds the
        encoded numeric values.
    mapping : dict[str, dict[str, float]]
        Maps column name to a dict of {category: encoded_value}.

    Returns
    -------
    str
        Rule string with conditions decoded back to category labels.

    Examples
    --------
    >>> mapping = {"A": {"a": 1, "b": 2, "c": 3}}
    >>> decode_numeric_encodings('(X["A"] >= 2)', mapping)
    '(X["A"].is_in(["b", "c"]))'
    """
    _OPS = {
        ">=": lambda a, b: a >= b,
        ">": lambda a, b: a > b,
        "<=": lambda a, b: a <= b,
        "<": lambda a, b: a < b,
        "==": lambda a, b: a == b,
        "!=": lambda a, b: a != b,
    }

    def _convert(m: re.Match[str]) -> str:
        col, op, val = m.group(1), m.group(2), m.group(3).strip()
        if col not in mapping or op not in _OPS:
            return m.group(0)
        try:
            threshold = float(val)
        except ValueError:
            return m.group(0)
        matched = [c for c, v in mapping[col].items() if _OPS[op](v, threshold)]
        if not matched:
            return m.group(0)
        if len(matched) == 1:
            return f'(X["{col}"] == "{matched[0]}")'
        cats = ", ".join(f'"{c}"' for c in matched)
        return f'(X["{col}"].is_in([{cats}]))'

    return _COND_PATTERN.sub(_convert, rule)


_TRUE_VALUES = {"true", "1"}
_FALSE_VALUES = {"false", "0"}


def format_as_boolean_conditions(rule: str, bool_columns: list[str]) -> str:
    """Convert True/False-like condition values to Python booleans.

    Recognises two condition shapes for the given columns:

    - "True"/"true"/"1" and "False"/"false"/"0" (quoted or bare), combined
      with ``==``/``!=``.
    - Raw numeric threshold splits on ``>=``/``>``/``<``/``<=`` (as produced
      by a model trained on a boolean column cast to float), where the
      threshold unambiguously selects only the 0.0 or only the 1.0 value.

    Both shapes are rewritten as ``== True``/``== False``.

    Parameters
    ----------
    rule : str
        Rule expression using ``X["col"]`` notation.
    bool_columns : list[str]
        Columns to treat as boolean.

    Returns
    -------
    str
        Rule string with boolean conditions normalised.

    Examples
    --------
    >>> format_as_boolean_conditions('(X["flag"] != "False")', ["flag"])
    '(X["flag"] == True)'

    >>> format_as_boolean_conditions('(X["flag"] >= 1.0)', ["flag"])
    '(X["flag"] == True)'

    >>> format_as_boolean_conditions('(X["flag"] < 1.0)', ["flag"])
    '(X["flag"] == False)'
    """

    def _convert(m: re.Match[str]) -> str:
        col, op, val = m.group(1), m.group(2), m.group(3).strip()
        if col not in bool_columns:
            return m.group(0)

        if op in ("==", "!="):
            clean_val = val.strip("\"'").lower()
            if clean_val in _TRUE_VALUES:
                is_true = True
            elif clean_val in _FALSE_VALUES:
                is_true = False
            else:
                return m.group(0)
            result = is_true if op == "==" else not is_true
            return f'(X["{col}"] == {result})'

        if op in (">=", ">", "<", "<="):
            try:
                threshold = float(val.strip("\"'"))
            except ValueError:
                return m.group(0)
            # Only 0.0/1.0 are possible, so a threshold split unambiguously
            # selects one of the two values only at these exact boundaries
            # (open/closed bound differs by operator - e.g. ">=1.0" selects
            # only 1.0, but "<=1.0" would trivially select both).
            if op == ">=" and 0 < threshold <= 1:
                return f'(X["{col}"] == True)'
            if op == ">" and 0 <= threshold < 1:
                return f'(X["{col}"] == True)'
            if op == "<" and 0 < threshold <= 1:
                return f'(X["{col}"] == False)'
            if op == "<=" and 0 <= threshold < 1:
                return f'(X["{col}"] == False)'
            return m.group(0)

        return m.group(0)

    return _COND_PATTERN.sub(_convert, rule)


def decode_onehot_encodings(
    rule: str,
    mapping: dict[str, tuple[str, str]],
    null_category: str | None = None,
) -> str:
    """Reverse a one-hot encoding back to a categorical condition.

    One-hot encoders typically produce a binary column per category, split
    by the model at 0.5. This converts those splits back to equality/
    inequality conditions on the original categorical column. If one category
    represents "value was null" (``null_category``), it's rendered as
    ``is_null()``/``~is_null()`` instead of a literal category comparison.

    Parameters
    ----------
    rule : str
        Rule expression using ``X["col"]`` notation, where col is the
        one-hot encoded binary column.
    mapping : dict[str, tuple[str, str]]
        Maps encoded column name to (original_col, category).
    null_category : str | None, optional
        Category value that represents an originally-null value, by default None.

    Returns
    -------
    str
        Rule string with one-hot conditions decoded back to category labels.

    Examples
    --------
    >>> mapping = {"status__active": ("status", "active")}
    >>> decode_onehot_encodings('(X["status__active"] >= 0.5)', mapping)
    '(X["status"] == "active")'
    """

    def _convert(m: re.Match[str]) -> str:
        col, op, val = m.group(1), m.group(2), m.group(3).strip()
        if col not in mapping:
            return m.group(0)
        original_col, category = mapping[col]
        try:
            threshold = float(val.strip("\"'"))
        except ValueError:
            return m.group(0)

        if category == null_category:
            if op in (">=", ">") and threshold >= 0.5:
                return f'X["{original_col}"].is_null()'
            if op in ("<", "<=") and threshold <= 1.0:
                return f'(~X["{original_col}"].is_null())'
            if op in ("==", "!="):
                is_null = (threshold >= 0.5) == (op == "==")
                return (
                    f'X["{original_col}"].is_null()'
                    if is_null
                    else f'(~X["{original_col}"].is_null())'
                )
            return m.group(0)

        is_equal = op in (">=", ">") and threshold >= 0.5
        return (
            f'(X["{original_col}"] == "{category}")'
            if is_equal
            else f'(X["{original_col}"] != "{category}")'
        )

    return _COND_PATTERN.sub(_convert, rule)


def decode_null_indicators(rule: str, mapping: dict[str, str]) -> str:
    """Convert null-indicator binary columns to ``is_null()`` conditions.

    Parameters
    ----------
    rule : str
        Rule expression using ``X["col"]`` notation, where col is a binary
        column indicating whether the original column was null.
    mapping : dict[str, str]
        Maps encoded column name to the original column name.

    Returns
    -------
    str
        Rule string with is-null conditions decoded.

    Examples
    --------
    >>> decode_null_indicators('(X["amount__is_null"] >= 0.5)', {"amount__is_null": "amount"})
    'X["amount"].is_null()'
    """

    def _convert(m: re.Match[str]) -> str:
        col, op, val = m.group(1), m.group(2), m.group(3).strip()
        if col not in mapping:
            return m.group(0)
        original_col = mapping[col]
        try:
            threshold = float(val)
        except ValueError:
            return m.group(0)
        if op in (">=", ">") and threshold >= 0.5:
            return f'X["{original_col}"].is_null()'
        if op in ("<", "<=") and threshold <= 0.5:
            return f'(~X["{original_col}"].is_null())'
        return m.group(0)

    return _COND_PATTERN.sub(_convert, rule)


def decode_discretized_bins(rule: str, mapping: dict[str, list[float]]) -> str:
    """Reverse a discretizer's bin index back to a threshold on the original column.

    Discretizers (equal-width, equal-frequency, quantile, k-means, tree-based,
    geometric, custom bin edges, ...) all replace a numeric column with an
    integer bin index. Given the fitted bin edges, this converts a condition
    on the bin index back to a condition on the original numeric column.

    Parameters
    ----------
    rule : str
        Rule expression using ``X["col"]`` notation, where col holds the bin
        index.
    mapping : dict[str, list[float]]
        Maps column name to its sorted bin edges, where ``edges[i]`` is the
        lower boundary of bin ``i``.

    Returns
    -------
    str
        Rule string with bin-index conditions decoded to original thresholds.

    Examples
    --------
    >>> decode_discretized_bins('(X["amount"] >= 2)', {"amount": [0, 10, 50, 200]})
    '(X["amount"] >= 50)'
    """

    def _convert(m: re.Match[str]) -> str:
        col, op, val = m.group(1), m.group(2), m.group(3).strip()
        if col not in mapping or op not in (">=", ">", "<", "<="):
            return m.group(0)
        try:
            threshold = float(val)
        except ValueError:
            return m.group(0)
        edges = mapping[col]
        bin_idx = math.ceil(threshold)
        if not (0 <= bin_idx < len(edges)):
            return m.group(0)
        new_op = ">=" if op in (">=", ">") else "<"
        return f'(X["{col}"] {new_op} {edges[bin_idx]})'

    return _COND_PATTERN.sub(_convert, rule)


def decode_scaled_thresholds(rule: str, mapping: dict[str, Callable[[float], float]]) -> str:
    """Reverse a monotonic numeric scaling back to a threshold on the original column.

    Works for any strictly increasing scaler (standardisation, min-max,
    log1p, Box-Cox, Yeo-Johnson, arcsinh, power, robust, ...) since the
    ``>=``/``<``/etc. ordering is preserved - pass the scaler's inverse
    transform as the mapping value.

    Parameters
    ----------
    rule : str
        Rule expression using ``X["col"]`` notation, where col holds the
        scaled values.
    mapping : dict[str, Callable[[float], float]]
        Maps column name to a function that inverts the scaling
        (e.g. ``scaler.inverse_transform``).

    Returns
    -------
    str
        Rule string with scaled thresholds decoded back to original values.

    Examples
    --------
    >>> decode_scaled_thresholds('(X["amount"] >= 5.0)', {"amount": lambda x: x * 2})
    '(X["amount"] >= 10.0)'
    """

    def _convert(m: re.Match[str]) -> str:
        col, op, val = m.group(1), m.group(2), m.group(3).strip()
        if col not in mapping:
            return m.group(0)
        try:
            threshold = float(val)
        except ValueError:
            return m.group(0)
        original_threshold = round(mapping[col](threshold), 6)
        return f'(X["{col}"] {op} {original_threshold})'

    return _COND_PATTERN.sub(_convert, rule)


def quote_string_values(rule: str, columns: list[str]) -> str:
    """Wrap bare (unquoted) condition values in double quotes.

    Parameters
    ----------
    rule : str
        Rule expression using ``X["col"]`` notation.
    columns : list[str]
        Columns whose values should be quoted.

    Returns
    -------
    str
        Rule string with bare values quoted.

    Examples
    --------
    >>> quote_string_values('(X["col"] == retail)', ["col"])
    '(X["col"] == "retail")'
    """

    def _convert(m: re.Match[str]) -> str:
        col, op, val = m.group(1), m.group(2), m.group(3).strip()
        if col not in columns or (val.startswith('"') and val.endswith('"')):
            return m.group(0)
        return f'(X["{col}"] {op} "{val}")'

    return _COND_PATTERN.sub(_convert, rule)


def round_thresholds(rule: str, columns: list[str], ndigits: int = 2) -> str:
    """Round numeric thresholds to a fixed number of decimal places.

    Parameters
    ----------
    rule : str
        Rule expression using ``X["col"]`` notation.
    columns : list[str]
        Columns to round.
    ndigits : int, optional
        Decimal places, by default 2.

    Returns
    -------
    str
        Rule string with rounded thresholds.

    Examples
    --------
    >>> round_thresholds('(X["amount"] >= 1234.56789)', ["amount"])
    '(X["amount"] >= 1234.57)'
    """

    def _convert(m: re.Match[str]) -> str:
        col, op, val = m.group(1), m.group(2), m.group(3).strip()
        if col not in columns:
            return m.group(0)
        try:
            rounded = round(float(val), ndigits)
        except ValueError:
            return m.group(0)
        return f'(X["{col}"] {op} {rounded})'

    return _COND_PATTERN.sub(_convert, rule)


def drop_null_clauses(rule: str, columns: list[str]) -> str:
    """Strip ``| X[col].is_null()`` clauses added for always-imputed columns.

    Parameters
    ----------
    rule : str
        Rule expression using ``X["col"]`` notation.
    columns : list[str]
        Columns for which to strip the is-null clause, e.g. because the
        column is never actually null once imputed.

    Returns
    -------
    str
        Rule string with the is-null clauses removed.

    Examples
    --------
    >>> drop_null_clauses('((X["amount"] >= 5.0) | X["amount"].is_null())', ["amount"])
    '(X["amount"] >= 5.0)'
    """
    for col in columns:
        c = re.escape(col)
        rule = re.sub(
            r'\(\((X\["' + c + r'"\][^)]+)\)\s*\|\s*X\["' + c + r'"\]\.is_null\(\)\)',
            r"(\1)",
            rule,
        )
    return rule


def drop_not_null_conditions(rule: str, columns: list[str]) -> str:
    """Drop standalone ``(~X[col].is_null())`` conditions for given columns.

    Parameters
    ----------
    rule : str
        Rule expression using ``X["col"]`` notation.
    columns : list[str]
        Columns for which a not-null condition is trivially true and can
        be dropped, e.g. because the column is never actually null.

    Returns
    -------
    str
        Rule string with the not-null conditions removed.

    Examples
    --------
    >>> drop_not_null_conditions('(X["a"] > 1) & (~X["b"].is_null())', ["b"])
    '(X["a"] > 1)'
    """
    for col in columns:
        c = re.escape(col)
        rule = re.sub(r'\s*&\s*\(~X\["' + c + r'"\]\.is_null\(\)\)', "", rule)
        rule = re.sub(r'\(~X\["' + c + r'"\]\.is_null\(\)\)\s*&\s*', "", rule)
    return rule


def prettify_rules(
    rules: list[str],
    steps: list[Callable[[str], str]],
    column_name_mapping: dict[str, str] | None = None,
) -> list[str]:
    """Apply an ordered list of rule-string transformations to each rule.

    Each step is a plain function taking and returning a rule string (e.g.
    ``simplify_rule``, or ``functools.partial(decode_numeric_encodings,
    mapping=woe_mapping)``), applied in order.

    Parameters
    ----------
    rules : list[str]
        Raw rule strings to prettify.
    steps : list[Callable[[str], str]]
        Ordered transformations to apply to each rule.
    column_name_mapping : dict[str, str] | None, optional
        Maps column name to a display name, applied last, by default None.

    Returns
    -------
    list[str]
        Prettified rule strings.

    Examples
    --------
    >>> from functools import partial
    >>> steps = [
    ...     partial(decode_numeric_encodings, mapping={"A": {"x": 1, "y": 2}}),
    ...     partial(round_thresholds, columns=["amount"]),
    ... ]
    >>> prettify_rules(['(X["A"] >= 2) & (X["amount"] > 1.239)'], steps)
    ['(X["A"] == "y") & (X["amount"] > 1.24)']
    """

    def _apply(rule: str) -> str:
        for step in steps:
            rule = step(rule)
        if column_name_mapping:
            for old_name, new_name in column_name_mapping.items():
                rule = rule.replace(f'X["{old_name}"]', f'X["{new_name}"]')
        return rule

    return [_apply(rule) for rule in rules]
