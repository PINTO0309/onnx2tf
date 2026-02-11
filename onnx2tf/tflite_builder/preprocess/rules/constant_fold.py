from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnx
from onnx import helper, numpy_helper

from onnx2tf.tflite_builder.preprocess.pipeline import register_preprocess_rule

CONSTANT_FOLD_RULE_ID = "constant_fold_a5"
_FOLDABLE_OPS = {
    "Identity",
    "Cast",
    "Unsqueeze",
    "Squeeze",
    "Concat",
    "Reshape",
    "Transpose",
    "Gather",
    "Shape",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Neg",
}


def _copy_node(node: onnx.NodeProto) -> onnx.NodeProto:
    cloned = onnx.NodeProto()
    cloned.CopyFrom(node)
    return cloned


def _find_attr(node: onnx.NodeProto, name: str) -> Optional[onnx.AttributeProto]:
    for attr in node.attribute:
        if str(attr.name) == str(name):
            return attr
    return None


def _get_attr_int(node: onnx.NodeProto, name: str, default: int) -> int:
    attr = _find_attr(node, name)
    if attr is None:
        return int(default)
    if attr.type == onnx.AttributeProto.INT:
        return int(attr.i)
    if attr.type == onnx.AttributeProto.FLOAT:
        return int(attr.f)
    return int(default)


def _get_attr_ints(node: onnx.NodeProto, name: str) -> Optional[List[int]]:
    attr = _find_attr(node, name)
    if attr is None:
        return None
    if attr.type == onnx.AttributeProto.INTS:
        return [int(v) for v in attr.ints]
    if attr.type == onnx.AttributeProto.INT:
        return [int(attr.i)]
    return None


def _find_const_tensor_attribute(node: onnx.NodeProto, name: str) -> Optional[onnx.TensorProto]:
    for attr in node.attribute:
        if str(attr.name) != str(name):
            continue
        if attr.type == onnx.AttributeProto.TENSOR:
            return attr.t
    return None


def _build_constant_map(graph: onnx.GraphProto) -> Dict[str, np.ndarray]:
    consts: Dict[str, np.ndarray] = {}
    for init in graph.initializer:
        consts[str(init.name)] = np.asarray(numpy_helper.to_array(init))
    for node in graph.node:
        if str(node.op_type) != "Constant" or len(node.output) < 1:
            continue
        value = _find_const_tensor_attribute(node, "value")
        if value is None:
            continue
        consts[str(node.output[0])] = np.asarray(numpy_helper.to_array(value))
    return consts


def _normalize_axes(axes: Sequence[int], rank: int) -> List[int]:
    normalized = []
    for axis in axes:
        a = int(axis)
        if a < 0:
            a += int(rank)
        normalized.append(a)
    return [int(v) for v in normalized]


def _fold_cast(values: List[np.ndarray], node: onnx.NodeProto) -> Optional[np.ndarray]:
    if len(values) < 1:
        return None
    to = int(_get_attr_int(node, "to", 1))
    np_dtype = {
        onnx.TensorProto.FLOAT: np.float32,
        onnx.TensorProto.FLOAT16: np.float16,
        onnx.TensorProto.DOUBLE: np.float64,
        onnx.TensorProto.INT8: np.int8,
        onnx.TensorProto.INT16: np.int16,
        onnx.TensorProto.INT32: np.int32,
        onnx.TensorProto.INT64: np.int64,
        onnx.TensorProto.UINT8: np.uint8,
        onnx.TensorProto.UINT16: np.uint16,
        onnx.TensorProto.UINT32: np.uint32,
        onnx.TensorProto.UINT64: np.uint64,
        onnx.TensorProto.BOOL: np.bool_,
    }.get(to, None)
    if np_dtype is None:
        return None
    return np.asarray(values[0]).astype(np_dtype)


def _fold_unsqueeze(values: List[np.ndarray], node: onnx.NodeProto) -> Optional[np.ndarray]:
    if len(values) < 1:
        return None
    data = np.asarray(values[0])
    axes = None
    if len(values) >= 2:
        axes = [int(v) for v in np.asarray(values[1]).reshape(-1).tolist()]
    if axes is None:
        axes = _get_attr_ints(node, "axes")
    if axes is None:
        return None
    rank = data.ndim
    normalized = _normalize_axes(axes, rank + 1)
    out = data
    for axis in sorted(normalized):
        out = np.expand_dims(out, axis=axis)
    return np.asarray(out)


def _fold_squeeze(values: List[np.ndarray], node: onnx.NodeProto) -> Optional[np.ndarray]:
    if len(values) < 1:
        return None
    data = np.asarray(values[0])
    axes = None
    if len(values) >= 2:
        axes = [int(v) for v in np.asarray(values[1]).reshape(-1).tolist()]
    if axes is None:
        axes = _get_attr_ints(node, "axes")
    if axes is None or len(axes) == 0:
        return np.squeeze(data)
    normalized = _normalize_axes(axes, data.ndim)
    return np.squeeze(data, axis=tuple(normalized))


def _fold_concat(values: List[np.ndarray], node: onnx.NodeProto) -> Optional[np.ndarray]:
    if len(values) < 1:
        return None
    axis = int(_get_attr_int(node, "axis", 0))
    rank = max(v.ndim for v in values)
    if axis < 0:
        axis += rank
    return np.concatenate([np.asarray(v) for v in values], axis=axis)


def _fold_reshape(values: List[np.ndarray]) -> Optional[np.ndarray]:
    if len(values) < 2:
        return None
    data = np.asarray(values[0])
    shape = [int(v) for v in np.asarray(values[1]).reshape(-1).tolist()]
    return np.reshape(data, shape)


def _fold_transpose(values: List[np.ndarray], node: onnx.NodeProto) -> Optional[np.ndarray]:
    if len(values) < 1:
        return None
    data = np.asarray(values[0])
    perm = None
    if len(values) >= 2:
        perm = [int(v) for v in np.asarray(values[1]).reshape(-1).tolist()]
    if perm is None:
        perm = _get_attr_ints(node, "perm")
    if perm is None:
        perm = [int(v) for v in reversed(range(data.ndim))]
    return np.transpose(data, axes=perm)


def _fold_gather(values: List[np.ndarray], node: onnx.NodeProto) -> Optional[np.ndarray]:
    if len(values) < 2:
        return None
    data = np.asarray(values[0])
    indices = np.asarray(values[1])
    axis = int(_get_attr_int(node, "axis", 0))
    if axis < 0:
        axis += data.ndim
    return np.take(data, indices, axis=axis)


def _fold_shape(values: List[np.ndarray]) -> Optional[np.ndarray]:
    if len(values) < 1:
        return None
    return np.asarray(np.asarray(values[0]).shape, dtype=np.int64)


def _fold_binary(values: List[np.ndarray], op_type: str) -> Optional[np.ndarray]:
    if len(values) < 2:
        return None
    x = np.asarray(values[0])
    y = np.asarray(values[1])
    if op_type == "Add":
        return x + y
    if op_type == "Sub":
        return x - y
    if op_type == "Mul":
        return x * y
    if op_type == "Div":
        return x / y
    return None


def _fold_unary(values: List[np.ndarray], op_type: str) -> Optional[np.ndarray]:
    if len(values) < 1:
        return None
    x = np.asarray(values[0])
    if op_type == "Identity":
        return np.asarray(x)
    if op_type == "Neg":
        return -x
    return None


def _fold_node(
    *,
    node: onnx.NodeProto,
    const_map: Dict[str, np.ndarray],
) -> Optional[np.ndarray]:
    if len(node.output) != 1:
        return None
    op_type = str(node.op_type)
    if op_type not in _FOLDABLE_OPS:
        return None
    input_names = [str(v) for v in node.input if str(v) != ""]
    values: List[np.ndarray] = []
    for name in input_names:
        if name not in const_map:
            return None
        values.append(np.asarray(const_map[name]))
    try:
        if op_type == "Cast":
            return _fold_cast(values, node)
        if op_type == "Unsqueeze":
            return _fold_unsqueeze(values, node)
        if op_type == "Squeeze":
            return _fold_squeeze(values, node)
        if op_type == "Concat":
            return _fold_concat(values, node)
        if op_type == "Reshape":
            return _fold_reshape(values)
        if op_type == "Transpose":
            return _fold_transpose(values, node)
        if op_type == "Gather":
            return _fold_gather(values, node)
        if op_type == "Shape":
            return _fold_shape(values)
        if op_type in {"Add", "Sub", "Mul", "Div"}:
            return _fold_binary(values, op_type)
        if op_type in {"Identity", "Neg"}:
            return _fold_unary(values, op_type)
    except Exception:
        return None
    return None


def apply_constant_fold_rule(onnx_graph: onnx.ModelProto) -> Dict[str, Any]:
    graph = onnx_graph.graph
    const_map = _build_constant_map(graph)
    rewritten_nodes: List[onnx.NodeProto] = []
    matched_nodes = 0
    rewritten_count = 0
    matched_by_op: Dict[str, int] = {}
    rewritten_by_op: Dict[str, int] = {}

    for node in graph.node:
        op_type = str(node.op_type)
        if op_type in _FOLDABLE_OPS:
            matched_nodes += 1
            matched_by_op[op_type] = int(matched_by_op.get(op_type, 0) + 1)
        folded = _fold_node(node=node, const_map=const_map)
        if folded is None:
            rewritten_nodes.append(_copy_node(node))
            continue
        output_name = str(node.output[0])
        const_tensor = numpy_helper.from_array(np.asarray(folded), name=output_name)
        const_node = helper.make_node(
            "Constant",
            [],
            [output_name],
            name=str(node.name) if str(node.name) != "" else f"{output_name}_constant_folded",
            value=const_tensor,
        )
        rewritten_nodes.append(const_node)
        const_map[output_name] = np.asarray(folded)
        rewritten_count += 1
        rewritten_by_op[op_type] = int(rewritten_by_op.get(op_type, 0) + 1)

    if rewritten_count > 0:
        del graph.node[:]
        graph.node.extend(rewritten_nodes)
    return {
        "matched_nodes": int(matched_nodes),
        "rewritten_nodes": int(rewritten_count),
        "changed": bool(rewritten_count > 0),
        "message": (
            f"matched_by_op={matched_by_op}, "
            f"rewritten_by_op={rewritten_by_op}"
        ),
    }


def register_constant_fold_rule() -> None:
    register_preprocess_rule(
        rule_id=CONSTANT_FOLD_RULE_ID,
        callback=apply_constant_fold_rule,
        overwrite=True,
    )

