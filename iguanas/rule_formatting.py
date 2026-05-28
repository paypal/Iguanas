import math
import re


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
    # Pattern to match full conditions including parentheses
    pattern = r'\(X\["([^"]+)"\]\s*([><=!]+)\s*([^\)]+)\)'

    # Find all conditions with their full match
    matches = [(m.group(0), m.group(1), m.group(2), m.group(3)) for m in re.finditer(pattern, rule)]

    if not matches:
        return rule

    # Track column order based on first appearance
    column_order = []

    # Group conditions by column
    column_conditions = {}
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
    conditions_to_remove = set()

    for col, conds in column_conditions.items():
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
            keeper = None
            for full, op, num_val, val in greater_conds:
                if num_val == max_val:
                    if keeper is None or (op == ">" and keeper[1] == ">="):
                        keeper = (full, op)

            # Mark all others for removal (only iterate once)
            conditions_to_remove.update(
                full for full, _, _, _ in greater_conds if full != keeper[0]
            )

        # For less/less-equal: keep only the most restrictive (lowest value)
        if len(less_conds) > 1:
            # Find min value and keeper in single pass
            min_val = min(num_val for _, _, num_val, _ in less_conds)
            # Among min values, prefer < over <=
            keeper = None
            for full, op, num_val, val in less_conds:
                if num_val == min_val:
                    if keeper is None or (op == "<" and keeper[1] == "<="):
                        keeper = (full, op)

            # Mark all others for removal (only iterate once)
            conditions_to_remove.update(full for full, _, _, _ in less_conds if full != keeper[0])

    result_conditions = [
        full
        for col in column_order
        for full, op, num_val, val in column_conditions[col]
        if full not in conditions_to_remove
    ]

    return " & ".join(result_conditions)


def format_floats_as_integers(rule: str, int_columns: list[str]) -> str:
    """
    Convert float values to integers for specified columns in a rule string,
    preserving the validity of the comparison operations.

    For integer columns, fractional thresholds are converted as follows:

    - >= operator: uses ceil(value) to ensure all valid integers are included
    - > operator: uses floor(value) to maintain strict inequality
    - <= operator: uses floor(value) to ensure all valid integers are included
    - < operator: uses ceil(value) to maintain strict inequality

    Parameters
    ----------
    rule : str
        Rule string with conditions like (X["col"] >= 0.1)
    int_columns : list[str]
        List of column names that should have integer thresholds

    Returns
    -------
    str
        Rule string with integer thresholds for specified columns

    Examples
    --------
    >>> format_floats_as_integers('(X["a"] >= 0.1) & (X["b"] >= 9.1)', ["a"])
    '(X["a"] >= 1) & (X["b"] >= 9.1)'

    >>> format_floats_as_integers('(X["a"] > 0.1) & (X["b"] < 10.9)', ["a", "b"])
    '(X["a"] > 0) & (X["b"] < 11)'

    >>> format_floats_as_integers('(X["a"] <= 9.9) & (X["a"] < 5.5)', ["a"])
    '(X["a"] <= 9) & (X["a"] < 6)'
    """
    if not int_columns:
        return rule

    # Pattern to match full conditions including parentheses
    pattern = r'\(X\["([^"]+)"\]\s*([><=!]+)\s*([^\)]+)\)'

    def replace_condition(match):
        col = match.group(1)
        op = match.group(2)
        val_str = match.group(3).strip()

        # Only process if column is in int_columns
        if col not in int_columns:
            return match.group(0)

        # Try to convert value to float
        try:
            val = float(val_str)
        except ValueError:
            # Not a numeric value, keep as is
            return match.group(0)

        if op == ">=":
            # For >=, use ceiling: X["a"] >= 0.1 means X["a"] >= 1
            new_val = math.ceil(val)
        elif op == ">":
            # For >, use floor: X["a"] > 0.1 means X["a"] > 0
            new_val = math.floor(val)
        elif op == "<=":
            # For <=, use floor: X["a"] <= 9.9 means X["a"] <= 9
            new_val = math.floor(val)
        elif op == "<":
            # For <, use ceiling: X["a"] < 9.1 means X["a"] < 10
            new_val = math.ceil(val)
        else:
            # For == or != or other operators, keep as is (or could round)
            return match.group(0)

        # Format the new value as integer
        return f'(X["{col}"] {op} {new_val})'

    result = re.sub(pattern, replace_condition, rule)
    return result


def add_missing_value_conditions(rule: str, nan_mapping: dict[str, int | float]) -> str:
    """
    Add null checks to conditions that would include the NaN replacement value.

    For each condition in the rule, if the column has a NaN replacement value that
    satisfies the condition, adds "| X["column"].isnull()" to include null values.

    Parameters
    ----------
    rule : str
        Rule string with conditions like (X["col"] < val)
    nan_mapping : dict[str, int | float]
        Dictionary mapping column names to their NaN replacement values

    Returns
    -------
    str
        Rule string with null checks added where appropriate

    Examples
    --------
    >>> mapping = {"a": 0, "b": 0.3, "c": 100}
    >>> add_missing_value_conditions('(X["a"] < 1) & (X["b"] >= 3) & (X["c"] > 10)', mapping)
    '(X["a"] < 1 | X["a"].isnull()) & (X["b"] >= 3) & (X["c"] > 10 | X["c"].isnull())'
    """
    if not nan_mapping:
        return rule

    # Pattern to match full conditions including parentheses
    pattern = r'\(X\[(["\'])([^"\']+)\1\]\s*([><=!]+)\s*([^\)]+)\)'

    def check_and_add_null(match):
        col = match.group(2)
        op = match.group(3)
        val_str = match.group(4).strip()

        # Only process if column is in nan_mapping
        if col not in nan_mapping:
            return match.group(0)

        # Try to convert value to float for comparison
        try:
            threshold = float(val_str)
        except ValueError:
            # Not a numeric value, keep as is
            return match.group(0)

        nan_value = nan_mapping[col]

        # Check if the NaN replacement value satisfies the condition
        satisfies_condition = False

        if op == ">=":
            satisfies_condition = nan_value >= threshold
        elif op == ">":
            satisfies_condition = nan_value > threshold
        elif op == "<=":
            satisfies_condition = nan_value <= threshold
        elif op == "<":
            satisfies_condition = nan_value < threshold
        elif op == "==":
            satisfies_condition = nan_value == threshold
        elif op == "!=":
            satisfies_condition = nan_value != threshold

        # If the NaN value satisfies the condition, add null check
        if satisfies_condition:
            return f'(X["{col}"] {op} {val_str} | X["{col}"].isnull())'
        else:
            return match.group(0)

    result = re.sub(pattern, check_and_add_null, rule)
    return result


def decode_scaled_feature_names(rule: str) -> str:
    """_summary_

    Args:
        rule (str): _description_

    Returns:
        str: _description_
    """
    return rule


def decode_math_features(rule: str) -> str:
    """
    Convert conditions on mathematical features back to their original form.

    For each condition on a mathematical feature, finds which original features satisfy the condition and
    replaces it with an appropriate condition.

    Parameters
    ----------
    rule : str
        Rule string with conditions like (X["col"] >= val) where val is the encoded value for mathematical features.

    Returns
    -------
    str
        Rule string with conditions on mathematical features converted to their original form.
    """

    return rule


def decode_rare_category_encodings(rule: str, rare_mapping: dict[str, list[str]]) -> str:
    """
    Convert conditions on encoded categorical columns that represent rare categories back to categorical conditions.

    For each condition on a column with rare categories, finds which rare categories satisfy the condition and
    replaces it with an appropriate categorical condition (.is_in() for multiple values, == for single value).

    Parameters
    ----------
    rule : str
        Rule string with conditions like (X["col"] >= val) where val is the encoded value for rare categories.
    rare_mapping : dict[str, list[str]]
        Dictionary mapping column names to lists of category names that are considered rare.

    Returns
    -------
    str
        Rule string with conditions on rare categories converted to categorical conditions

    Examples
    --------
    >>> mapping = {"A": ["a", "b"], "B": ["x"]}
    >>> decode_rare_category_encodings('(X["A"] >= 1) & (X["B"] == 1)', mapping)
    '(X["A"].is_in(["a", "b"])) & (X["B"] == "x")'
    """
    if not rare_mapping:
        return rule

    # Pattern to match full conditions including parentheses
    pattern = r'\(X\[(["\'])([^"\']+)\1\]\s*([><=!]+)\s*([^\)]+)\)'

    def decode_condition(match):
        quote_char = match.group(1)
        col = match.group(2)
        op = match.group(3)
        val_str = match.group(4).strip()

        # Only process if column is in rare_mapping
        if col not in rare_mapping:
            return match.group(0)

        # Check if the value matches the encoded value for the rare category (assumed to be 1)
        if val_str != "1":
            return match.group(0)

        categories = rare_mapping[col]

        if len(categories) == 0:
            return match.group(0)
        elif len(categories) == 1:
            return f'(X["{col}"] == "{categories[0]}")'
        else:
            categories_str = ", ".join(f'"{cat}"' for cat in categories)
            return f'(X["{col}"].is_in([{categories_str}]))'

    result = re.sub(pattern, decode_condition, rule)
    return result


def decode_numeric_encodings(
    rule: str, encoding_mapping: dict[str, dict[str, int | float]]
) -> str:
    """
    Convert numerical conditions on encoded categorical columns to categorical conditions.

    For each numerical condition, finds which category values satisfy the condition and
    replaces it with an appropriate categorical condition (.is_in() for multiple values,
    == for single value).

    Parameters
    ----------
    rule : str
        Rule string with numerical conditions like (X["col"] >= val)
    encoding_mapping : dict[str, dict[str, int | float]]
        Dictionary mapping column names to their encoding dictionaries.
        Inner dict maps category names to their encoded numerical values.

    Returns
    -------
    str
        Rule string with categorical conditions

    Examples
    --------
    >>> mapping = {"A": {"a": 1, "b": 2, "c": 3}, "B": {"a": -8.1, "b": 1.1, "c": 3}}
    >>> decode_numeric_encodings('(X["A"] >= 2) & (X["B"] < 0)', mapping)
    '(X["A"].is_in(["b", "c"])) & (X["B"] == "a")'

    >>> mapping = {"A": {"x": 1, "y": 2}}
    >>> decode_numeric_encodings('(X["A"] == 1)', mapping)
    '(X["A"] == "x")'
    """
    if not encoding_mapping:
        return rule

    # Pattern to match full conditions including parentheses
    pattern = r'\(X\[(["\'])([^"\']+)\1\]\s*([><=!]+)\s*([^\)]+)\)'

    def decode_condition(match):
        col = match.group(2)
        op = match.group(3)
        val_str = match.group(4).strip()

        # Only process if column is in encoding_mapping
        if col not in encoding_mapping:
            return match.group(0)

        # Try to convert value to float for comparison
        try:
            threshold = float(val_str)
        except ValueError:
            # Not a numeric value, keep as is
            return match.group(0)

        col_mapping = encoding_mapping[col]

        # Find all categories that satisfy the condition
        matching_categories = []

        for category, encoded_value in col_mapping.items():
            satisfies = False

            if op == ">=":
                satisfies = encoded_value >= threshold
            elif op == ">":
                satisfies = encoded_value > threshold
            elif op == "<=":
                satisfies = encoded_value <= threshold
            elif op == "<":
                satisfies = encoded_value < threshold
            elif op == "==":
                satisfies = encoded_value == threshold
            elif op == "!=":
                satisfies = encoded_value != threshold

            if satisfies:
                matching_categories.append(category)

        # Generate the replacement condition
        if len(matching_categories) == 0:
            # No categories match - keep original or return False condition
            # For now, keep original
            return match.group(0)
        elif len(matching_categories) == 1:
            # Single category - use equality
            return f'(X["{col}"] == "{matching_categories[0]}")'
        else:
            # Multiple categories - use is_in (polars method)
            categories_str = ", ".join(f'"{cat}"' for cat in matching_categories)
            return f'(X["{col}"].is_in([{categories_str}]))'

    result = re.sub(pattern, decode_condition, rule)
    return result


def format_as_boolean_conditions(rule: str, bool_columns: list[str]) -> str:
    """
    Convert boolean value representations to actual boolean values in rule strings.

    For columns specified as boolean columns, converts string/numeric representations
    of True/False to actual boolean values and normalizes operators to ==.

    Conversions:

    - "True", "true", 1 with == operator becomes True
    - "True", "true", 1 with != operator becomes False (operator changed to ==)
    - "False", "false", 0 with == operator becomes False
    - "False", "false", 0 with != operator becomes True (operator changed to ==)

    Parameters
    ----------
    rule : str
        Rule string with conditions like (X["col"] == "True")
    bool_columns : list[str]
        List of column names that should have boolean values

    Returns
    -------
    str
        Rule string with boolean values and normalized operators

    Examples
    --------
    >>> format_as_boolean_conditions('(X["flag"] == "True") & (X["active"] != "False")', ["flag", "active"])
    '(X["flag"] == True) & (X["active"] == True)'

    >>> format_as_boolean_conditions('(X["enabled"] != 1) & (X["disabled"] == 0)', ["enabled", "disabled"])
    '(X["enabled"] == False) & (X["disabled"] == False)'

    >>> format_as_boolean_conditions('(X["is_valid"] == "true") & (X["is_ready"] != "false")', ["is_valid", "is_ready"])
    '(X["is_valid"] == True) & (X["is_ready"] == True)'
    """
    if not bool_columns:
        return rule

    # Pattern to match conditions with == or != operators
    pattern = r'\(X\[(["\'])([^"\']+)\1\]\s*(==|!=)\s*([^\)]+)\)'

    def convert_condition(match):
        quote_char = match.group(1)
        col = match.group(2)
        op = match.group(3)
        val_str = match.group(4).strip()

        # Only process if column is in bool_columns
        if col not in bool_columns:
            return match.group(0)

        # Remove quotes if present
        value = val_str.strip('"').strip("'")

        # Determine if this is a True-like or False-like value
        is_true_like = value in ["True", "true", "1"]
        is_false_like = value in ["False", "false", "0"]

        if not (is_true_like or is_false_like):
            # Not a recognized boolean representation
            return match.group(0)

        # Determine the final boolean value based on input and operator
        if is_true_like:
            bool_value = True if op == "==" else False
        else:  # is_false_like
            bool_value = False if op == "==" else True

        # Always use == operator with the converted boolean value
        return f'(X["{col}"] == {bool_value})'

    result = re.sub(pattern, convert_condition, rule)
    return result
