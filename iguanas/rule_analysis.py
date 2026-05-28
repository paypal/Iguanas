import polars as pl
from iguanas.metrics import compute_metrics
from iguanas.rule_evaluation import apply_rules
import ast
import re
from collections import deque


def _to_py(expr: str) -> str:
    return re.sub(r'\s*&\s*', ' and ', re.sub(r'\s*\|\s*', ' or ', expr))

def _node_to_str(node: ast.AST) -> str:
    if isinstance(node, ast.Compare):
        return f"({ast.unparse(node)})"
    elif isinstance(node, ast.BoolOp):
        op = " & " if isinstance(node.op, ast.And) else " | "
        return op.join(_node_to_str(v) for v in node.values)
    else:
        s = ast.unparse(node)
        return re.sub(r'\sand\s', ' & ', re.sub(r'\sor\s', ' | ', s))

def parse_conditions(expr: str):
    """Parses a boolean expression into a nested dict tree."""
    tree = ast.parse(_to_py(expr), mode="eval")
    return _convert(tree.body)

def _convert(node):
    if isinstance(node, ast.BoolOp):
        op = "&" if isinstance(node.op, ast.And) else "|"
        values = [_convert(v) for v in node.values]
        result = values[0]
        for v in values[1:]:
            result = {"op": op, "left": result, "right": v}
        return result
    elif isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Compare):
        return ast.unparse(node)
    else:
        return ast.unparse(node)

def parse_levels(expr: str) -> list[dict]:
    """
    Parses a boolean expression level by level (BFS), assigning a hierarchical
    index to each child so the original expression can be rebuilt.
    Each level entry: {operator: [(index, expr), ...]}
    Child indices use dot notation to reflect their position in the tree
    (e.g. "1.0" = first child of the item at index "1" in the parent level).

    Example:
        "(A > 1) | (B <= 5 & C < 3) | (D >= 0)"
        -> [
            {"|": [("0", "A > 1"), ("1", "B <= 5 & C < 3"), ("2", "D >= 0")]},
            {"&": [("1.0", "B <= 5"), ("1.1", "C < 3")]},
        ]
    """
    tree = ast.parse(_to_py(expr), mode="eval")

    levels = []
    # queue items: (ast_node, parent_index_string)
    queue = deque([(tree.body, "")])

    while queue:
        next_queue = deque()
        level_entries = []

        for node, parent_idx in queue:
            if isinstance(node, ast.BoolOp):
                op = "&" if isinstance(node.op, ast.And) else "|"
                children = []
                for i, v in enumerate(node.values):
                    child_idx = f"{parent_idx}.{i}" if parent_idx else str(i)
                    children.append((child_idx, _node_to_str(v)))
                    if isinstance(v, ast.BoolOp):
                        next_queue.append((v, child_idx))
                level_entries.append({op: children})

        if level_entries:
            levels.append(level_entries[0] if len(level_entries) == 1 else level_entries)
        queue = next_queue

    return levels


def rebuild_from_levels(levels: list[dict]) -> str:
    """
    Rebuilds the original expression from the output of parse_levels.
    Processes levels bottom-up: deepest compound expressions are collapsed
    first, then their rebuilt string replaces the placeholder in the parent level.
    """
    # Seed the map with all leaf expressions across all levels
    index_map: dict[str, str] = {}
    for entry in levels:
        for e in ([entry] if isinstance(entry, dict) else entry):
            op = next(iter(e))
            for idx, expr in e[op]:
                index_map.setdefault(idx, expr)

    # Collapse bottom-up
    for entry in reversed(levels):
        for e in ([entry] if isinstance(entry, dict) else entry):
            op = next(iter(e))
            children = e[op]
            first_idx = children[0][0]
            parent_idx = first_idx.rsplit(".", 1)[0] if "." in first_idx else None
            rebuilt = f" {op} ".join(f"({index_map[idx]})" for idx, _ in children)
            if parent_idx is None:
                return rebuilt          # reached the root
            index_map[parent_idx] = rebuilt

    return ""


def generate_rule_performance_report(
    rules: str | list[str],
    X: pl.DataFrame,
    y: pl.Series,
    weights: pl.Series | None = None,
) -> pl.DataFrame:
    """
    For each rule in *rules*, parses it into its components (BFS levels),
    evaluates every component on X, computes metrics, and returns a DataFrame
    with one row per component across all rules.

    The ``rule_index`` column uses dot notation with the rule's position
    in the list prepended as the root level, e.g. for the 2nd rule:
      "2.0", "2.1", "2.1.0", "2.1.1", ...

    Parameters
    ----------
    exprs : list[str]
        List of boolean rule expression strings (using & / | operators).
    X : pl.DataFrame
        Feature DataFrame on which to evaluate each component.
    y : pl.Series
        Boolean target series.
    weights : pl.Series | None, default=None
        Optional sample weights passed to compute_metrics.

    Returns
    -------
    pl.DataFrame
        One row per component with columns:
        rule_index, rule, + all metric columns from compute_metrics.
    """
    components = []
    if isinstance(rules, str):
        rules = [rules]
    for rule_idx, expr in enumerate(rules, 0):
        # Level 0: the rule itself
        components.append((str(rule_idx), expr))
        levels = parse_levels(expr)
        for entry in levels:
            for e in ([entry] if isinstance(entry, dict) else entry):
                op = next(iter(e))
                for idx, rule_str in e[op]:
                    components.append((f"{rule_idx}.{idx}", rule_str))

    if not components:
        return pl.DataFrame()

    idxs, rule_strs = zip(*components)

    # Deduplicate across all expressions to avoid duplicate column names
    unique_rules = list(dict.fromkeys(rule_strs))

    # Single batched evaluation + metrics across all expressions
    R_all = apply_rules(X, unique_rules)
    M_all = compute_metrics(R_all, y, weights)

    meta = pl.DataFrame({
        "rule_index": list(idxs),
        "rule": list(rule_strs),
    })

    return meta.join(M_all, on="rule")
