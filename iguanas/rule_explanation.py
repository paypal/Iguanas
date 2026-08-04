from __future__ import annotations

import re
from typing import Any

import polars as pl

from .rule_evaluation import apply_rules

_COND_PATTERN = re.compile(r'\(X\["([^"]+)"\]\s*([><=!]+)\s*([^\)]+)\)')

_OP_TO_WORDS: dict[str, str] = {
    ">": "greater than",
    ">=": "at least",
    "<": "less than",
    "<=": "at most",
    "==": "equal to",
    "!=": "not equal to",
}


def verbalize_rule(rule: str) -> str:
    """Convert a rule expression to a plain-English sentence.

    Parameters
    ----------
    rule : str
        Rule expression using ``X["col"]`` notation with ``&`` / ``|``
        operators.

    Returns
    -------
    str
        Plain-English description of the rule.

    Examples
    --------
    >>> verbalize_rule('(X["age"] > 30) & (X["income"] < 50000)')
    'age is greater than 30 AND income is less than 50000'

    >>> verbalize_rule('(X["score"] >= 0.8) | (X["flag"] == 1)')
    'score is at least 0.8 OR flag is equal to 1'
    """

    def _cond_to_words(m: re.Match) -> str:
        feature, op, val = m.group(1), m.group(2), m.group(3).strip()
        op_word = _OP_TO_WORDS.get(op, op)
        return f"{feature} is {op_word} {val}"

    verbalized = _COND_PATTERN.sub(_cond_to_words, rule)
    verbalized = re.sub(r"\s*&\s*", " AND ", verbalized)
    verbalized = re.sub(r"\s*\|\s*", " OR ", verbalized)
    # Remove leftover grouping parentheses
    verbalized = re.sub(r"[\(\)]", "", verbalized).strip()
    return verbalized


def compute_coverage_overlap(R: pl.DataFrame) -> pl.DataFrame:
    """Compute pairwise Jaccard overlap between rule predictions.

    For each pair of rules, computes the Jaccard similarity:
    *(samples flagged by both) / (samples flagged by either)*.
    A high score means the two rules flag nearly the same population.

    Parameters
    ----------
    R : pl.DataFrame
        Boolean DataFrame of rule predictions (columns = rules, rows = samples).

    Returns
    -------
    pl.DataFrame
        Long-format DataFrame with columns:
        ``rule_a``, ``rule_b``, ``jaccard``, ``flagged_by_both``,
        ``flagged_by_either``.  Only the upper triangle of the pair matrix
        is returned.  Sorted by ``jaccard`` descending (most overlapping first).

    Examples
    --------
    >>> import polars as pl
    >>> R = pl.DataFrame({
    ...     "rule_A": [True, True, False, False],
    ...     "rule_B": [True, False, True, False],
    ... })
    >>> compute_coverage_overlap(R)
    shape: (1, 5)
    ┌────────┬────────┬─────────┬────────────────┬──────────────────┐
    │ rule_a │ rule_b │ jaccard │ flagged_by_both │ flagged_by_either│
    │ ---    │ ---    │ ---     │ ---            │ ---              │
    │ str    │ str    │ f64     │ i64            │ i64              │
    ╞════════╪════════╪═════════╪════════════════╪══════════════════╡
    │ rule_A │ rule_B │ 0.333…  │ 1              │ 3                │
    └────────┴────────┴─────────┴────────────────┴──────────────────┘
    """
    _EMPTY_SCHEMA = {
        "rule_a": pl.String,
        "rule_b": pl.String,
        "jaccard": pl.Float64,
        "flagged_by_both": pl.Int64,
        "flagged_by_either": pl.Int64,
    }

    rules = R.columns
    if len(rules) < 2:
        return pl.DataFrame(schema=_EMPTY_SCHEMA)

    records: list[dict[str, Any]] = []
    for i, rule_a in enumerate(rules):
        for rule_b in rules[i + 1 :]:
            a = R[rule_a]
            b = R[rule_b]
            both = int((a & b).sum())
            either = int((a | b).sum())
            jaccard = both / either if either > 0 else 0.0
            records.append(
                {
                    "rule_a": rule_a,
                    "rule_b": rule_b,
                    "jaccard": jaccard,
                    "flagged_by_both": both,
                    "flagged_by_either": either,
                }
            )

    return pl.DataFrame(records).sort("jaccard", descending=True)


def compute_counterfactual(
    rule: str,
    sample: pl.DataFrame,
    epsilon: float = 1e-6,
) -> list[dict[str, Any]]:
    """Find minimal feature changes to un-flag a sample from a rule.

    For each atomic condition in the rule that is currently satisfied by
    ``sample``, computes the smallest perturbation of that feature that
    would violate the condition. Results are sorted by ``abs_change``
    ascending so the caller can pick the least-effort option.

    **AND rules**: breaking any single condition un-flags the sample.
    The first (cheapest) entry is sufficient.

    **OR rules**: every currently-satisfied condition must be broken to
    un-flag the sample. All returned entries together form the
    counterfactual.

    Parameters
    ----------
    rule : str
        Rule expression to un-flag the sample from.
    sample : pl.DataFrame
        Single-row DataFrame representing the sample to explain. Must
        contain all columns referenced in ``rule``.
    epsilon : float, default=1e-6
        Small perturbation used when breaking strict inequalities.

    Returns
    -------
    list[dict]
        Candidate counterfactuals sorted by ``abs_change`` ascending.
        Each dict has keys:

        - ``feature`` — feature name to change
        - ``condition`` — the atomic condition that would be broken
        - ``current_value`` — current feature value
        - ``suggested_value`` — value that breaks the condition
        - ``abs_change`` — magnitude of the required change

        Returns an empty list if the sample is not flagged by the rule or
        if no atomic conditions can be broken numerically.

    Raises
    ------
    ValueError
        If ``sample`` does not have exactly one row.

    Examples
    --------
    >>> sample = pl.DataFrame({"age": [45], "income": [80_000]})
    >>> rule = '(X["age"] > 30) & (X["income"] >= 50_000)'
    >>> compute_counterfactual(rule, sample)
    [
        {'feature': 'age', 'condition': ..., 'current_value': 45.0,
         'suggested_value': 29.999999, 'abs_change': 15.000001},
        {'feature': 'income', ...},
    ]
    """
    if len(sample) != 1:
        raise ValueError(f"sample must have exactly 1 row, got {len(sample)}")

    # If any column referenced in the rule is absent, the rule cannot be evaluated
    rule_features = {m.group(1) for m in _COND_PATTERN.finditer(rule)}
    if not rule_features.issubset(sample.columns):
        return []

    # Verify the rule actually fires on this sample
    is_flagged = bool(apply_rules(sample, [rule])[rule][0])
    if not is_flagged:
        return []

    results: list[dict[str, Any]] = []

    for feature, op, val_str in _COND_PATTERN.findall(rule):
        val_str = val_str.strip()
        try:
            threshold = float(val_str)
        except ValueError:
            continue  # non-numeric threshold — skip

        current = float(sample[feature][0])

        # Check whether this specific condition is currently satisfied
        cond = f'(X["{feature}"] {op} {val_str})'
        if not bool(apply_rules(sample, [cond])[cond][0]):
            continue

        # Compute the suggested value that breaks this condition
        if op in (">", ">="):
            suggested = threshold - epsilon
        elif op in ("<", "<="):
            suggested = threshold + epsilon
        elif op == "==":
            suggested = current + epsilon
        elif op == "!=":
            suggested = threshold  # set to the forbidden value

        results.append(
            {
                "feature": feature,
                "condition": cond,
                "current_value": current,
                "suggested_value": suggested,
                "abs_change": abs(suggested - current),
            }
        )

    return sorted(results, key=lambda d: d["abs_change"])
