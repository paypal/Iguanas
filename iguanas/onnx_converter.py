"""Convert Iguanas rule strings to an ONNX binary classifier model."""
from __future__ import annotations

import ast

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

_DTYPE_MAP: dict[str, tuple[int, type]] = {
    "f32": (TensorProto.FLOAT, np.float32),
    "f64": (TensorProto.DOUBLE, np.float64),
}

# Minimum opset that covers all operators used (GreaterOrEqual/LessOrEqual=16,
# Equal=19, Cast/Squeeze=25).
_OPSET = 25

_CMP_OPS: dict[type, str] = {
    ast.GtE: "GreaterOrEqual",
    ast.Gt: "Greater",
    ast.LtE: "LessOrEqual",
    ast.Lt: "Less",
    ast.Eq: "Equal",
}


def rules_to_onnx(
    rules: str | list[str],
    dtype: str = "f32",
) -> onnx.ModelProto:
    """Convert Iguanas rule strings to an ONNX binary classifier.

    Parameters
    ----------
    rules : str | list[str]
        One rule string or a list of rule strings.  When a list is supplied,
        the rules are OR'd together — the model outputs 1 if *any* rule fires.
        Each rule must use the ``X["col"] op val`` notation produced by
        Iguanas, where ``op`` is one of ``>=``, ``>``, ``<=``, ``<``,
        ``==``, ``!=``, and conditions may be combined with ``&`` / ``|``.
    dtype : str, default ``"f32"``
        Numeric dtype for the input tensor and thresholds.
        ``"f32"`` → float32, ``"f64"`` → float64.

    Returns
    -------
    onnx.ModelProto
        ONNX model with:

        * input ``X`` shape ``[N, num_features]`` (*dtype* as requested)
        * output ``prediction`` shape ``[N]`` (int64, values 0 or 1)

        Feature names are stored in ``metadata_props`` as
        ``"feature_0"``, ``"feature_1"``, … in first-appearance order.

    Raises
    ------
    ValueError
        If ``rules`` is empty, ``dtype`` is not ``"f32"`` or ``"f64"``, or
        a rule string is syntactically invalid or uses unsupported node types.
    """
    if isinstance(rules, str) and not rules:
        raise ValueError("rules must not be empty")
    elif isinstance(rules, list) and not rules:
        raise ValueError("rules must not be empty")
    elif not isinstance(rules, str | list):
        raise TypeError("rules must be a string or a list of strings")

    if isinstance(rules, str):
        rules = [rules]

    if dtype not in _DTYPE_MAP:
        raise ValueError(f"dtype must be 'f32' or 'f64', got {dtype!r}")

    dtype_onnx, dtype_np = _DTYPE_MAP[dtype]

    parsed: list[ast.expr] = []
    for rule in rules:
        try:
            parsed.append(ast.parse(rule.strip(), mode="eval").body)
        except SyntaxError as exc:
            raise ValueError(f"Invalid rule syntax: {rule!r}") from exc

    feature_index: dict[str, int] = {}
    for tree in parsed:
        _collect_features(tree, feature_index)

    ctx = _BuildCtx(dtype_onnx, dtype_np, feature_index)
    rule_outputs = [ctx.visit(tree) for tree in parsed]

    combined = rule_outputs[0]
    for other in rule_outputs[1:]:
        out = ctx.fresh("top_or")
        ctx.nodes.append(helper.make_node("Or", inputs=[combined, other], outputs=[out]))
        combined = out

    pred_out = "prediction"
    ctx.nodes.append(
        helper.make_node("Cast", inputs=[combined], outputs=[pred_out], to=TensorProto.INT64)
    )

    num_features = len(feature_index)
    graph = helper.make_graph(
        ctx.nodes,
        "iguanas_rules",
        [helper.make_tensor_value_info("X", dtype_onnx, [None, num_features])],
        [helper.make_tensor_value_info(pred_out, TensorProto.INT64, [None])],
        initializer=ctx.initializers,
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET)])

    for feat_name, idx in feature_index.items():
        entry = model.metadata_props.add()
        entry.key = f"feature_{idx}"
        entry.value = feat_name

    onnx.checker.check_model(model)
    return model


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _collect_features(node: ast.expr, feature_index: dict[str, int]) -> None:
    """First-pass traversal: register every X["col"] column name."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitAnd | ast.BitOr):
        _collect_features(node.left, feature_index)
        _collect_features(node.right, feature_index)
    elif isinstance(node, ast.Compare) and isinstance(node.left, ast.Subscript):
        col = _subscript_col(node.left)
        if col not in feature_index:
            feature_index[col] = len(feature_index)


def _subscript_col(node: ast.expr) -> str:
    """Return the column name from an ``X["col"]`` subscript node."""
    if not isinstance(node, ast.Subscript):
        raise ValueError(f"Expected X['col'] subscript, got {ast.dump(node)}")
    idx = node.slice
    if not isinstance(idx, ast.Constant):
        raise ValueError(f"Rule column key must be a string literal, got {ast.dump(idx)}")
    return str(idx.value)


class _BuildCtx:
    """Accumulates ONNX nodes and initializers while traversing a rule AST."""

    def __init__(
        self,
        dtype_onnx: int,
        dtype_np: type,
        feature_index: dict[str, int],
    ) -> None:
        self.dtype_onnx = dtype_onnx
        self.dtype_np = dtype_np
        self.feature_index = feature_index
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []
        self._counter = 0

    def fresh(self, prefix: str) -> str:
        name = f"{prefix}_{self._counter}"
        self._counter += 1
        return name

    def visit(self, node: ast.expr) -> str:
        """Emit ONNX nodes for *node* and return the output tensor name."""
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitAnd | ast.BitOr):
            return self._visit_binop(node)
        if isinstance(node, ast.Compare):
            return self._visit_compare(node)
        raise ValueError(f"Unsupported rule expression: {ast.dump(node)}")

    def _visit_binop(self, node: ast.BinOp) -> str:
        onnx_op = "And" if isinstance(node.op, ast.BitAnd) else "Or"
        left = self.visit(node.left)
        right = self.visit(node.right)
        out = self.fresh(onnx_op.lower())
        self.nodes.append(helper.make_node(onnx_op, inputs=[left, right], outputs=[out]))
        return out

    def _visit_compare(self, node: ast.Compare) -> str:
        col = _subscript_col(node.left)
        op = node.ops[0]
        comparator = node.comparators[0]
        if isinstance(comparator, ast.Constant) and isinstance(comparator.value, int | float):
            threshold = float(comparator.value)
        elif (
            isinstance(comparator, ast.UnaryOp)
            and isinstance(comparator.op, ast.USub)
            and isinstance(comparator.operand, ast.Constant)
            and isinstance(comparator.operand.value, int | float)
        ):
            threshold = -float(comparator.operand.value)
        else:
            raise ValueError(
                f"Rule threshold must be a numeric literal, got {ast.dump(comparator)}"
            )

        feat_out = self._extract_feature(col)
        thresh_name = self._make_scalar(threshold)

        if isinstance(op, ast.NotEq):
            eq_out = self.fresh("eq")
            self.nodes.append(
                helper.make_node("Equal", inputs=[feat_out, thresh_name], outputs=[eq_out])
            )
            cmp_out = self.fresh("neq")
            self.nodes.append(helper.make_node("Not", inputs=[eq_out], outputs=[cmp_out]))
        else:
            onnx_op = _CMP_OPS[type(op)]
            cmp_out = self.fresh("cmp")
            self.nodes.append(
                helper.make_node(onnx_op, inputs=[feat_out, thresh_name], outputs=[cmp_out])
            )
        return cmp_out

    def _extract_feature(self, col: str) -> str:
        """Emit a Gather node slicing column *col* from X → output shape [N]."""
        idx_name = self.fresh("idx")
        # 0-D indices drop the gathered axis, giving output shape [N] from [N, F]
        self.initializers.append(
            numpy_helper.from_array(
                np.array(self.feature_index[col], dtype=np.int64), name=idx_name
            )
        )
        feat_out = self.fresh("feat")
        self.nodes.append(
            helper.make_node("Gather", inputs=["X", idx_name], outputs=[feat_out], axis=1)
        )
        return feat_out

    def _make_scalar(self, value: float) -> str:
        name = self.fresh("thresh")
        self.initializers.append(
            numpy_helper.from_array(np.array(value, dtype=self.dtype_np), name=name)
        )
        return name
