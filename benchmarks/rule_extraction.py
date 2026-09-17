"""Extract each learner's rules into one common form.

The benchmark compares *rule set generators*, so every method has to be reduced
to the same object: a set of boolean rules combined by disjunction, where a row
is flagged if any rule fires. Left alone the baselines do not agree on that --
RuleFit weights its rules with signed coefficients and adds linear terms, BRL
reads an ordered list top-down, FIGS sums tree outputs, gradient boosting sums
leaf weights. Those scoring functions are extra machinery no Iguanas rule set
has, and comparing against them measures the scorer rather than the rules.

Each extractor therefore returns rule expressions in Iguanas' syntax --
``(X["a"] > 1.0) & (X["b"] <= 2.0)`` -- so :func:`iguanas.rule_evaluation.apply_rules`
can evaluate every method's output identically.

Only rules that argue *for* the positive class are kept: a RuleFit rule with a
negative coefficient is evidence against, and OR-ing it in would invert its
meaning.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

# "attr4 <= 2.5 and attr5 > 0.5" -- shared by SkopeRules, RuleFit and BRL.
_CONDITION = re.compile(r"^\s*(?P<col>.+?)\s*(?P<op><=|>=|<|>|==|!=)\s*(?P<value>-?[\d.eE+]+)\s*$")


def _quote(column: str) -> str:
    return f'X["{column}"]'


def conjunction_to_rule(text: str, columns: set[str] | None = None) -> str | None:
    """Convert an ``and``-joined conjunction into an Iguanas rule expression.

    Returns None when any conjunct cannot be parsed, so a malformed rule is
    dropped rather than silently truncated to the part that did parse.
    """
    parts: list[str] = []
    for chunk in re.split(r"\s+and\s+", str(text).strip()):
        if not chunk:
            continue
        match = _CONDITION.match(chunk)
        if match is None:
            return None
        column = match.group("col").strip()
        if columns is not None and column not in columns:
            return None
        parts.append(f'({_quote(column)} {match.group("op")} {match.group("value")})')
    return " & ".join(parts) if parts else None


def rules_from_conjunctions(texts: list[str], columns: set[str] | None = None) -> list[str]:
    out = [conjunction_to_rule(t, columns) for t in texts]
    return [r for r in out if r]


def rules_from_skope(model: Any, columns: set[str] | None = None) -> list[str]:
    raw = getattr(model, "rules_", None) or []
    texts = [r[0] if isinstance(r, tuple) else r for r in raw]
    return rules_from_conjunctions([str(t) for t in texts], columns)


def rules_from_rulefit(model: Any, columns: set[str] | None = None) -> list[str]:
    """Positive-coefficient rule terms only; linear terms are not rules."""
    frame = model._get_rules()
    keep = frame[(frame["type"] == "rule") & (frame["coef"].astype(float) > 0.0)]
    return rules_from_conjunctions([str(t) for t in keep["rule"].tolist()], columns)


def rules_from_brl(model: Any, columns: set[str] | None = None) -> list[str]:
    """Antecedents of the rule list whose consequent favours the positive class.

    BRL is an *ordered* list: a row is scored by the first antecedent it
    matches, and later entries are only reached by rows the earlier ones
    rejected. Read as a flat disjunction the list is meaningless -- its trailing
    entries are near-catch-alls, so OR-ing everything flags the whole population
    at base-rate precision. ``theta`` holds each antecedent's P(class 1) (plus a
    final default), so only those above 0.5 are kept.
    """
    raw = list(getattr(model, "rules_", None) or [])
    texts = [str(r[0] if isinstance(r, tuple) else r) for r in raw]
    theta = np.ravel(getattr(model, "theta", []))
    if theta.size >= len(texts):
        texts = [t for t, p in zip(texts, theta[: len(texts)], strict=True) if p > 0.5]
    return rules_from_conjunctions(texts, columns)


def rules_from_ripper(model: Any, columns: set[str] | None = None) -> list[str]:
    """Wittgenstein rules are ``feature=value`` conjunctions joined by ``^``.

    A value may be a range (``2.0-3.0``), which becomes a pair of bounds.
    """
    ruleset = getattr(model, "ruleset_", None)
    rules: list[str] = []
    for rule in getattr(ruleset, "rules", []) or []:
        parts: list[str] = []
        ok = True
        for cond in getattr(rule, "conds", []) or []:
            column = str(getattr(cond, "feature", ""))
            value = str(getattr(cond, "val", ""))
            if columns is not None and column not in columns:
                ok = False
                break
            span = re.match(r"^(-?[\d.]+)-(-?[\d.]+)$", value)
            if span:
                lo, hi = span.group(1), span.group(2)
                parts.append(f"({_quote(column)} >= {lo}) & ({_quote(column)} <= {hi})")
                continue
            try:
                float(value)
            except ValueError:
                ok = False
                break
            parts.append(f"({_quote(column)} == {value})")
        if ok and parts:
            rules.append(" & ".join(parts))
    return rules


def rules_from_xgboost(frame: Any) -> list[str]:
    """Root-to-leaf paths of a boosted ensemble whose leaf raises the log-odds.

    *frame* is ``Booster.trees_to_dataframe()``. A leaf's ``Gain`` column holds
    its output value, so a positive value is a path arguing for the positive
    class.
    """
    rules: list[str] = []
    for _, tree in frame.groupby("Tree", sort=False):
        nodes = tree.set_index("ID")
        roots = [i for i in nodes.index if nodes.loc[i, "Node"] == 0]
        if not roots:
            continue

        def walk(node_id: str, path: list[str]) -> None:
            row = nodes.loc[node_id]
            if str(row["Feature"]) == "Leaf":
                if float(row["Gain"]) > 0 and path:
                    rules.append(" & ".join(path))
                return
            column, threshold = str(row["Feature"]), float(row["Split"])
            walk(str(row["Yes"]), [*path, f"({_quote(column)} < {threshold})"])
            walk(str(row["No"]), [*path, f"({_quote(column)} >= {threshold})"])

        walk(roots[0], [])
    return rules


def rules_from_sklearn_tree(model: Any, feature_names: list[str]) -> list[str]:
    """Root-to-leaf paths of a fitted sklearn tree whose leaf predicts positive."""
    tree = getattr(model, "tree_", None)
    if tree is None:
        return []
    rules: list[str] = []

    def walk(node: int, path: list[str]) -> None:
        if tree.children_left[node] == tree.children_right[node]:  # leaf
            values = tree.value[node][0]
            if len(values) > 1 and values[1] > values[0] and path:
                rules.append(" & ".join(path))
            return
        column = feature_names[tree.feature[node]]
        threshold = tree.threshold[node]
        walk(tree.children_left[node], [*path, f"({_quote(column)} <= {threshold})"])
        walk(tree.children_right[node], [*path, f"({_quote(column)} > {threshold})"])

    walk(0, [])
    return rules


def rules_from_figs(model: Any, feature_names: list[str]) -> list[str]:
    """Root-to-leaf paths through FIGS' trees whose leaf is majority positive.

    A FIGS node's ``value`` is a class distribution, not a contribution, so the
    test is ``P(positive) > P(negative)``. Testing ``value > 0`` instead accepts
    any leaf containing a single positive row, which on imbalanced data selects
    nearly the whole population.
    """
    rules: list[str] = []

    def walk(node: Any, path: list[str]) -> None:
        if node is None:
            return
        left, right = getattr(node, "left", None), getattr(node, "right", None)
        if left is None and right is None:
            value = np.ravel(getattr(node, "value", []))
            if value.size >= 2 and value[1] > value[0] and path:
                rules.append(" & ".join(path))
            return
        idx = getattr(node, "feature", None)
        threshold = getattr(node, "threshold", None)
        if idx is None or threshold is None or int(idx) >= len(feature_names):
            return
        column = feature_names[int(idx)]
        walk(left, [*path, f"({_quote(column)} <= {threshold})"])
        walk(right, [*path, f"({_quote(column)} > {threshold})"])

    for tree in getattr(model, "trees_", []) or []:
        walk(tree, [])
    return rules
