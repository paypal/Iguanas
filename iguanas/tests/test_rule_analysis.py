import ast

import polars as pl
import pytest

from iguanas.rule_analysis import (
    _node_to_str,
    _to_py,
    generate_rule_performance_report,
    parse_conditions,
    parse_levels,
    rebuild_from_levels,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def X():
    return pl.DataFrame(
        {
            "a": [0, 2, 0, 2],
            "b": [4, 4, 6, 6],
            "c": [1, 1, 4, 4],
        }
    )


@pytest.fixture
def y():
    return pl.Series([True, False, True, False])


@pytest.fixture
def weights():
    return pl.Series([1.0, 2.0, 1.0, 2.0])


# ---------------------------------------------------------------------------
# _to_py
# ---------------------------------------------------------------------------


class TestToPy:
    def test_replaces_ampersand_with_and(self):
        assert _to_py("A & B") == "A and B"

    def test_replaces_pipe_with_or(self):
        assert _to_py("A | B") == "A or B"

    def test_replaces_both_operators(self):
        assert _to_py("A & B | C") == "A and B or C"

    def test_handles_no_surrounding_spaces(self):
        assert _to_py("A&B|C") == "A and B or C"

    def test_no_operators_unchanged(self):
        assert _to_py('X["col"] > 1') == 'X["col"] > 1'


# ---------------------------------------------------------------------------
# _node_to_str
# ---------------------------------------------------------------------------


class TestNodeToStr:
    def test_and_node_converted_to_ampersand(self):
        node = ast.parse("A and B", mode="eval").body
        assert _node_to_str(node) == "A & B"

    def test_or_node_converted_to_pipe(self):
        node = ast.parse("A or B", mode="eval").body
        assert _node_to_str(node) == "A | B"

    def test_compare_node_wrapped_in_parens(self):
        # Compare nodes are now wrapped in parens so they are eval-safe
        node = ast.parse("a > 1", mode="eval").body
        assert _node_to_str(node) == "(a > 1)"

    def test_compare_with_subscript_wrapped_in_parens(self):
        node = ast.parse("X['a'] > 1", mode="eval").body
        assert _node_to_str(node) == "(X['a'] > 1)"

    def test_boolop_with_compare_children_parenthesizes_each(self):
        # Each Compare child gets its own parens → eval-safe string
        node = ast.parse("X['a'] > 1 and X['b'] <= 5", mode="eval").body
        result = _node_to_str(node)
        assert result == "(X['a'] > 1) & (X['b'] <= 5)"

    def test_nested_boolop_or_of_and(self):
        # (A and B) or C — A and B are Names (no parens added), joined by |
        node = ast.parse("(A and B) or C", mode="eval").body
        result = _node_to_str(node)
        assert "A & B" in result
        assert "|" in result


# ---------------------------------------------------------------------------
# parse_conditions  (also exercises all _convert branches)
# ---------------------------------------------------------------------------


class TestParseConditions:
    def test_boolop_and_two_values(self):
        result = parse_conditions("A & B")
        assert result == {"op": "&", "left": "A", "right": "B"}

    def test_boolop_or_two_values(self):
        result = parse_conditions("A | B")
        assert result == {"op": "|", "left": "A", "right": "B"}

    def test_boolop_three_values_folds_left(self):
        # A & B & C  → {"op": "&", "left": {"op": "&", "left": "A", "right": "B"}, "right": "C"}
        result = parse_conditions("A & B & C")
        assert result["op"] == "&"
        assert result["right"] == "C"
        assert result["left"]["op"] == "&"
        assert result["left"]["left"] == "A"
        assert result["left"]["right"] == "B"

    def test_name_node_returns_string(self):
        # A single identifier → ast.Name branch
        result = parse_conditions("A")
        assert result == "A"

    def test_compare_node_returns_unparsed_string(self):
        # A > 1 → ast.Compare branch
        result = parse_conditions("A > 1")
        assert "A" in result
        assert "1" in result

    def test_else_branch_constant(self):
        # Literal "1" parses to ast.Constant, which hits the else branch
        result = parse_conditions("1")
        assert result == "1"

    def test_nested_boolop(self):
        result = parse_conditions("A & (B | C)")
        assert result["op"] == "&"
        assert result["left"] == "A"
        assert result["right"]["op"] == "|"


# ---------------------------------------------------------------------------
# parse_levels
# ---------------------------------------------------------------------------


class TestParseLevels:
    def test_flat_expression_returns_empty_list(self):
        # No BoolOp at root → levels should be empty
        assert parse_levels('X["a"] > 1') == []

    def test_single_and_produces_one_level(self):
        levels = parse_levels('(X["a"] > 1) & (X["b"] <= 5)')
        assert len(levels) == 1
        assert "&" in levels[0]
        children = levels[0]["&"]
        assert children[0][0] == "0"
        assert children[1][0] == "1"
        # ast.unparse normalises double quotes → single quotes
        assert "X['a'] > 1" in children[0][1]
        assert "X['b'] <= 5" in children[1][1]

    def test_single_or_produces_one_level(self):
        levels = parse_levels('(X["a"] > 1) | (X["b"] <= 5)')
        assert len(levels) == 1
        assert "|" in levels[0]

    def test_three_or_operands_at_root(self):
        levels = parse_levels('(X["a"] > 1) | (X["b"] <= 5) | (X["c"] < 3)')
        assert len(levels) == 1
        assert len(levels[0]["|"]) == 3

    def test_nested_or_and_produces_two_levels(self):
        expr = '(X["a"] > 1) | ((X["b"] <= 5) & (X["c"] < 3))'
        levels = parse_levels(expr)
        assert len(levels) == 2
        # Level 1: OR with two children; second child is a compound (index "1")
        l1 = levels[0]["|"]
        assert l1[0][0] == "0"
        assert l1[1][0] == "1"
        # Level 2: AND; children carry the dotted parent index "1"
        l2 = levels[1]["&"]
        assert l2[0][0] == "1.0"
        assert l2[1][0] == "1.1"

    def test_multiple_compound_children_produces_list_at_level(self):
        # Both children of OR are AND → level 2 is a list of two dicts
        expr = '((X["a"] > 1) & (X["b"] <= 5)) | ((X["c"] < 3) & (X["a"] <= 0))'
        levels = parse_levels(expr)
        assert len(levels) == 2
        # Level 2 should be a list, not a single dict
        assert isinstance(levels[1], list)
        assert len(levels[1]) == 2
        assert "&" in levels[1][0]
        assert "&" in levels[1][1]

    def test_child_index_uses_dot_notation(self):
        # Verifies the f"{parent_idx}.{i}" branch for nested nodes
        expr = '(X["a"] > 1) | ((X["b"] <= 5) & (X["c"] < 3))'
        levels = parse_levels(expr)
        leaf_indices = [idx for idx, _ in levels[1]["&"]]
        assert all("." in idx for idx in leaf_indices)

    def test_compare_children_are_parenthesized(self):
        # Atomic (Compare) children at any level are wrapped in parens
        levels = parse_levels('(X["a"] > 1) & (X["b"] <= 5)')
        children = levels[0]["&"]
        assert children[0][1] == "(X['a'] > 1)"
        assert children[1][1] == "(X['b'] <= 5)"

    def test_compound_child_has_parenthesized_atoms(self):
        # The compound child string at level 1 has each atomic condition in parens
        expr = '(X["a"] > 1) | ((X["b"] <= 5) & (X["c"] < 3))'
        levels = parse_levels(expr)
        l1_children = levels[0]["|"]
        # First child: simple comparison → parenthesized
        assert l1_children[0][1] == "(X['a'] > 1)"
        # Second child: compound → each atom parenthesized, joined by &
        assert l1_children[1][1] == "(X['b'] <= 5) & (X['c'] < 3)"
        # Level-2 children: each atomic condition parenthesized
        l2_children = levels[1]["&"]
        assert l2_children[0][1] == "(X['b'] <= 5)"
        assert l2_children[1][1] == "(X['c'] < 3)"


# ---------------------------------------------------------------------------
# rebuild_from_levels
# ---------------------------------------------------------------------------


class TestRebuildFromLevels:
    def test_empty_levels_returns_empty_string(self):
        # Hits the final `return ""` fallback
        assert rebuild_from_levels([]) == ""

    def test_single_and_level(self):
        levels = [{"&": [("0", 'X["a"] > 1'), ("1", 'X["b"] <= 5')]}]
        result = rebuild_from_levels(levels)
        assert 'X["a"] > 1' in result
        assert 'X["b"] <= 5' in result
        assert "&" in result

    def test_single_or_level(self):
        levels = [
            {
                "|": [
                    ("0", 'X["a"] > 1'),
                    ("1", 'X["b"] <= 5'),
                    ("2", 'X["c"] < 3'),
                ]
            }
        ]
        result = rebuild_from_levels(levels)
        assert 'X["a"] > 1' in result
        assert "|" in result

    def test_nested_levels_rebuilds_all_conditions(self):
        expr = '(X["a"] > 1) | ((X["b"] <= 5) & (X["c"] < 3))'
        levels = parse_levels(expr)
        result = rebuild_from_levels(levels)
        # ast.unparse uses single quotes
        assert "X['a'] > 1" in result
        assert "X['b'] <= 5" in result
        assert "X['c'] < 3" in result

    def test_multiple_entries_at_level(self):
        # Level 2 is a list → exercises the `else entry` (non-dict) branch
        expr = '((X["a"] > 1) & (X["b"] <= 5)) | ((X["c"] < 3) & (X["a"] <= 0))'
        levels = parse_levels(expr)
        result = rebuild_from_levels(levels)
        # ast.unparse uses single quotes
        assert "X['a'] > 1" in result
        assert "X['b'] <= 5" in result
        assert "X['c'] < 3" in result
        assert "X['a'] <= 0" in result

    def test_roundtrip_single_level(self):
        levels = parse_levels('(X["a"] > 1) & (X["b"] <= 5)')
        result = rebuild_from_levels(levels)
        # Should be non-empty and preserve both conditions
        assert result != ""


# ---------------------------------------------------------------------------
# generate_rule_performance_report
# ---------------------------------------------------------------------------


class TestComputeComponentMetrics:
    def test_single_string_rule_accepted(self, X, y):
        # Exercises the `if isinstance(rules, str): rules = [rules]` branch
        rule = '(X["a"] > 1) & (X["b"] <= 5)'
        result = generate_rule_performance_report(rule, X, y)
        assert isinstance(result, pl.DataFrame)
        assert result.height > 0

    def test_list_of_one_rule(self, X, y):
        rules = ['(X["a"] > 1) & (X["b"] <= 5)']
        result = generate_rule_performance_report(rules, X, y)
        assert result.height > 0
        assert "rule_index" in result.columns
        assert "rule" in result.columns

    def test_empty_rules_returns_empty_dataframe(self, X, y):
        result = generate_rule_performance_report([], X, y)
        assert isinstance(result, pl.DataFrame)
        assert result.is_empty()

    def test_flat_rule_returns_single_row_at_level_0(self, X, y):
        # A single comparison has no sub-components but is itself the root.
        rules = ['(X["a"] > 1)']
        result = generate_rule_performance_report(rules, X, y)
        assert isinstance(result, pl.DataFrame)
        assert result.height == 1
        assert result["rule_index"].to_list() == ["0"]

    def test_multiple_rules_all_represented(self, X, y):
        rules = [
            '(X["a"] > 1) & (X["b"] <= 5)',
            '(X["c"] < 3) | (X["a"] <= 0)',
        ]
        result = generate_rule_performance_report(rules, X, y)
        assert result.height > 0
        assert "0" in result["rule_index"].to_list()
        assert "1" in result["rule_index"].to_list()

    def test_rule_index_zero_based(self, X, y):
        rules = [
            '(X["a"] > 1) | (X["b"] <= 5)',
            '(X["c"] < 3) & (X["a"] <= 0)',
        ]
        result = generate_rule_performance_report(rules, X, y)
        for row in result.iter_rows(named=True):
            root = row["rule_index"].split(".")[0]
            assert root in ("0", "1")

    def test_nested_rule_depth_numbers(self, X, y):
        # A flat OR rule: rule itself + 3 leaf components = 4 rows.
        rule = '(X["a"] > 1) | (X["b"] <= 5) | (X["c"] < 3)'
        result = generate_rule_performance_report(rule, X, y)
        assert result.height == 4
        assert result["rule_index"].str.starts_with("0").all()

    def test_with_weights_adds_weight_columns(self, X, y, weights):
        rule = '(X["a"] > 1) & (X["b"] <= 5)'
        result = generate_rule_performance_report(rule, X, y, weights=weights)
        assert "TP_weight" in result.columns

    def test_without_weights_no_weight_columns(self, X, y):
        rule = '(X["a"] > 1) & (X["b"] <= 5)'
        result = generate_rule_performance_report(rule, X, y)
        assert "TP_weight" not in result.columns

    def test_metric_columns_present(self, X, y):
        rule = '(X["a"] > 1) | (X["b"] <= 5)'
        result = generate_rule_performance_report(rule, X, y)
        for col in ("precision", "recall", "f1"):
            assert col in result.columns

    def test_deduplication_of_shared_sub_conditions(self, X, y):
        # Same atomic condition appears in both rules; apply_rules should not
        # fail with a duplicate column name.
        rules = [
            '(X["a"] > 1) & (X["b"] <= 5)',
            '(X["a"] > 1) | (X["c"] < 3)',
        ]
        result = generate_rule_performance_report(rules, X, y)
        assert result.height > 0

    def test_nested_rule_produces_multiple_depths(self, X, y):
        # Compound sub-expressions are now properly parenthesized, so
        # apply_rules can eval all components including the depth-1 compound.
        rule = '(X["a"] > 1) | ((X["b"] <= 5) & (X["c"] < 3))'
        result = generate_rule_performance_report(rule, X, y)
        # rule itself + 2 children at depth 1 + 2 children at depth 2 = 5 rows
        assert result.height == 5
        dot_counts = result["rule_index"].str.count_matches(r"\.").unique().sort().to_list()
        assert dot_counts == [0, 1, 2]

    def test_nested_rule_level1_compound_is_evald(self, X, y):
        # The compound sub-expression at level 1 must appear in the results
        # with valid (non-null) metrics — proving it was successfully eval'd.
        rule = '(X["a"] > 1) | ((X["b"] <= 5) & (X["c"] < 3))'
        result = generate_rule_performance_report(rule, X, y)
        compound = result.filter(pl.col("rule") == "(X['b'] <= 5) & (X['c'] < 3)")
        assert compound.height == 1
        assert compound["precision"][0] is not None

    def test_nested_rule_precision_values(self, X, y):
        # X = {"a":[0,2,0,2], "b":[4,4,6,6], "c":[1,1,4,4]}, y=[T,F,T,F]
        # (X['b'] <= 5): [T,T,F,F] → TP=1 (row0), FP=1 (row1) → precision=0.5
        rule = '(X["a"] > 1) | ((X["b"] <= 5) & (X["c"] < 3))'
        result = generate_rule_performance_report(rule, X, y)
        b_row = result.filter(pl.col("rule") == "(X['b'] <= 5)")
        assert b_row["precision"][0] == pytest.approx(0.5)

    def test_both_symmetric_compound_children_are_evald(self, X, y):
        # Rule where BOTH depth-1 children are compound
        rule = '((X["a"] > 1) & (X["b"] <= 5)) | ((X["c"] < 3) & (X["a"] <= 0))'
        result = generate_rule_performance_report(rule, X, y)
        # rule itself + 2 compound at depth 1 + 2+2 atomic at depth 2 = 7 rows
        assert result.height == 7
        depth1 = result.filter(pl.col("rule_index").str.count_matches(r"\.") == 1)
        assert depth1.height == 2
        depth2 = result.filter(pl.col("rule_index").str.count_matches(r"\.") == 2)
        assert depth2.height == 4
