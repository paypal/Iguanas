"""Tests for onnx_converter — 100% line and branch coverage."""
from __future__ import annotations

import ast

import numpy as np
import onnxruntime as ort
import pytest

from iguanas.onnx_converter import _BuildCtx, _subscript_col, rules_to_onnx, _DTYPE_MAP


def _run(model, X: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(model.SerializeToString())
    return sess.run(None, {"X": X})[0]


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            rules_to_onnx("")

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            rules_to_onnx([])

    def test_non_string_non_list_raises(self):
        with pytest.raises(TypeError, match="must be a string or a list of strings"):
            rules_to_onnx(123)  # type: ignore[arg-type]

    def test_invalid_dtype_raises(self):
        with pytest.raises(ValueError, match="dtype must be 'f32' or 'f64'"):
            rules_to_onnx('(X["a"] > 1.0)', dtype="f16")

    def test_invalid_syntax_raises(self):
        with pytest.raises(ValueError, match="Invalid rule syntax"):
            rules_to_onnx("this is not valid !!!")

    def test_unsupported_ast_node_raises(self):
        # "1 + 2" is valid Python but not a rule expression
        with pytest.raises(ValueError, match="Unsupported rule expression"):
            rules_to_onnx("1 + 2")

    def test_non_subscript_left_raises(self):
        # "(5 > 3)" has a Constant on the left, not X["col"]
        with pytest.raises(ValueError, match="Expected X\\['col'\\] subscript"):
            rules_to_onnx("(5 > 3)")

    def test_non_constant_key_raises(self):
        # X[a] — unquoted variable as dict key
        with pytest.raises(ValueError, match="Rule column key must be a string literal"):
            rules_to_onnx("(X[a] > 5)")

    def test_non_constant_threshold_raises(self):
        # Threshold is a variable name, not a numeric literal
        with pytest.raises(ValueError, match="Rule threshold must be a numeric literal"):
            rules_to_onnx('(X["a"] > b)')

    def test_negative_threshold(self):
        # Threshold expressed as a unary minus: -5.0 is UnaryOp(USub, Constant(5.0))
        m = rules_to_onnx('(X["a"] >= -5.0)')
        X = np.array([[-4.0], [-5.0], [-6.0]], dtype=np.float32)
        np.testing.assert_array_equal(_run(m, X), [1, 1, 0])


# ---------------------------------------------------------------------------
# dtype parameter
# ---------------------------------------------------------------------------

class TestDtype:
    def test_f32_default(self):
        model = rules_to_onnx('(X["a"] > 1.0)')
        X = np.array([[2.0], [0.5]], dtype=np.float32)
        np.testing.assert_array_equal(_run(model, X), [1, 0])

    def test_f64(self):
        model = rules_to_onnx('(X["a"] > 1.0)', dtype="f64")
        X = np.array([[2.0], [0.5]], dtype=np.float64)
        np.testing.assert_array_equal(_run(model, X), [1, 0])


# ---------------------------------------------------------------------------
# Comparison operators
# ---------------------------------------------------------------------------

class TestOperators:
    def _x(self, val: float, dtype: str = "f32") -> np.ndarray:
        dt = np.float32 if dtype == "f32" else np.float64
        return np.array([[val]], dtype=dt)

    def test_gte(self):
        m = rules_to_onnx('(X["a"] >= 5.0)')
        assert _run(m, self._x(5.0))[0] == 1
        assert _run(m, self._x(4.9))[0] == 0

    def test_gt(self):
        m = rules_to_onnx('(X["a"] > 5.0)')
        assert _run(m, self._x(5.1))[0] == 1
        assert _run(m, self._x(5.0))[0] == 0

    def test_lte(self):
        m = rules_to_onnx('(X["a"] <= 5.0)')
        assert _run(m, self._x(5.0))[0] == 1
        assert _run(m, self._x(5.1))[0] == 0

    def test_lt(self):
        m = rules_to_onnx('(X["a"] < 5.0)')
        assert _run(m, self._x(4.9))[0] == 1
        assert _run(m, self._x(5.0))[0] == 0

    def test_eq(self):
        m = rules_to_onnx('(X["a"] == 5.0)')
        assert _run(m, self._x(5.0))[0] == 1
        assert _run(m, self._x(5.1))[0] == 0

    def test_neq(self):
        m = rules_to_onnx('(X["a"] != 5.0)')
        assert _run(m, self._x(4.0))[0] == 1
        assert _run(m, self._x(5.0))[0] == 0


# ---------------------------------------------------------------------------
# Boolean combinations within a single rule string
# ---------------------------------------------------------------------------

class TestBooleanCombinations:
    def test_and_two_conditions(self):
        m = rules_to_onnx('(X["a"] > 1.0) & (X["b"] < 10.0)')
        X = np.array([[2.0, 5.0], [2.0, 15.0], [0.5, 5.0]], dtype=np.float32)
        np.testing.assert_array_equal(_run(m, X), [1, 0, 0])

    def test_or_two_conditions(self):
        m = rules_to_onnx('(X["a"] > 1.0) | (X["b"] < 10.0)')
        X = np.array([[2.0, 15.0], [0.5, 5.0], [0.5, 15.0]], dtype=np.float32)
        np.testing.assert_array_equal(_run(m, X), [1, 1, 0])

    def test_three_way_and(self):
        # BoolOp.values has 3 elements — tests the multi-child chaining loop
        m = rules_to_onnx('(X["a"] > 1.0) & (X["b"] < 10.0) & (X["c"] >= 5.0)')
        X = np.array([[2.0, 5.0, 5.0], [2.0, 5.0, 4.0]], dtype=np.float32)
        np.testing.assert_array_equal(_run(m, X), [1, 0])

    def test_nested_and_or(self):
        m = rules_to_onnx('((X["a"] > 1.0) & (X["b"] < 10.0)) | (X["c"] >= 20.0)')
        X = np.array(
            [[2.0, 5.0, 1.0], [0.5, 15.0, 20.0], [0.5, 15.0, 1.0]],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(_run(m, X), [1, 1, 0])


# ---------------------------------------------------------------------------
# Multiple rules as a list (OR'd at the top level)
# ---------------------------------------------------------------------------

class TestMultipleRules:
    def test_single_rule_in_list(self):
        m = rules_to_onnx(['(X["a"] >= 3.0)'])
        X = np.array([[3.0], [2.9]], dtype=np.float32)
        np.testing.assert_array_equal(_run(m, X), [1, 0])

    def test_two_rules(self):
        m = rules_to_onnx(['(X["a"] > 5.0)', '(X["b"] < 2.0)'])
        X = np.array([[6.0, 5.0], [1.0, 1.0], [1.0, 5.0]], dtype=np.float32)
        np.testing.assert_array_equal(_run(m, X), [1, 1, 0])

    def test_three_rules(self):
        # Tests that the top-level OR chain handles more than two rules
        m = rules_to_onnx(['(X["a"] > 5.0)', '(X["b"] < 2.0)', '(X["c"] == 0.0)'])
        X = np.array(
            [[6.0, 5.0, 1.0], [1.0, 1.0, 1.0], [1.0, 5.0, 0.0], [1.0, 5.0, 1.0]],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(_run(m, X), [1, 1, 1, 0])


# ---------------------------------------------------------------------------
# Output shape and type
# ---------------------------------------------------------------------------

class TestOutputContract:
    def test_output_dtype_is_int64(self):
        m = rules_to_onnx('(X["a"] > 1.0)')
        out = _run(m, np.array([[2.0], [0.5]], dtype=np.float32))
        assert out.dtype == np.int64

    def test_output_shape_matches_batch(self):
        m = rules_to_onnx('(X["a"] > 1.0)')
        X = np.array([[2.0], [0.5], [3.0]], dtype=np.float32)
        assert _run(m, X).shape == (3,)


# ---------------------------------------------------------------------------
# Metadata properties
# ---------------------------------------------------------------------------

class TestMetadata:
    def test_single_rule_feature_order(self):
        m = rules_to_onnx('(X["age"] > 30.0) & (X["income"] >= 50000.0)')
        props = {p.key: p.value for p in m.metadata_props}
        assert props["feature_0"] == "age"
        assert props["feature_1"] == "income"

    def test_multiple_rules_feature_order(self):
        m = rules_to_onnx(['(X["a"] > 1.0)', '(X["b"] < 2.0)'])
        props = {p.key: p.value for p in m.metadata_props}
        assert props["feature_0"] == "a"
        assert props["feature_1"] == "b"

    def test_shared_feature_across_rules(self):
        # "a" appears in both rules; should only get one index
        m = rules_to_onnx(['(X["a"] > 1.0)', '(X["a"] < 10.0)'])
        props = {p.key: p.value for p in m.metadata_props}
        assert len(props) == 1
        assert props["feature_0"] == "a"


# ---------------------------------------------------------------------------
# _subscript_col unit tests (internal helper)
# ---------------------------------------------------------------------------

class TestSubscriptCol:
    def test_valid_string_key(self):
        node = ast.parse('X["amount"]', mode="eval").body
        assert _subscript_col(node) == "amount"

    def test_non_subscript_raises(self):
        node = ast.parse("5", mode="eval").body
        with pytest.raises(ValueError, match="Expected X\\['col'\\] subscript"):
            _subscript_col(node)

    def test_non_constant_key_raises(self):
        node = ast.parse("X[a]", mode="eval").body
        with pytest.raises(ValueError, match="Rule column key must be a string literal"):
            _subscript_col(node)
