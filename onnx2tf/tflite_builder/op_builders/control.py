from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import onnx
from onnx import numpy_helper

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR


_IF_NMS_GUARD_ELSE_OPS_WITH_NESTED_IF = [
    "ReduceMax",
    "Cast",
    "Equal",
    "Unsqueeze",
    "Add",
    "Mul",
    "Unsqueeze",
    "Add",
    "Unsqueeze",
    "NonMaxSuppression",
    "Gather",
    "If",
]

_IF_NMS_GUARD_ELSE_OPS_SIMPLE = [
    "ReduceMax",
    "Cast",
    "Unsqueeze",
    "Add",
    "Mul",
    "Unsqueeze",
    "Add",
    "Unsqueeze",
    "NonMaxSuppression",
    "Gather",
    "Squeeze",
]


_IF_GENERIC_BRANCH_SAFE_OPS = {
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Pow",
    "Max",
    "Min",
    "Abs",
    "Neg",
    "Relu",
    "Sigmoid",
    "Tanh",
    "Cast",
    "Identity",
    "Squeeze",
    "Unsqueeze",
    "Reshape",
    "Transpose",
    "Concat",
    "Equal",
    "Greater",
    "GreaterOrEqual",
    "Less",
    "LessOrEqual",
    "Not",
    "And",
    "Or",
    "Xor",
    "ReduceSum",
    "ReduceMean",
    "ReduceMin",
    "ReduceMax",
    "Constant",
    "ConstantOfShape",
    "Shape",
    "Size",
    "Gather",
    "Slice",
    "Pad",
    "Conv",
    "Sqrt",
    "LSTM",
}


def is_supported_if_nms_guard_pattern(node: Any) -> bool:
    then_graph = node.attrs.get("then_branch", None)
    else_graph = node.attrs.get("else_branch", None)
    if then_graph is None or else_graph is None:
        return False
    if not hasattr(then_graph, "node") or not hasattr(else_graph, "node"):
        return False
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        return False
    if len(then_graph.node) != 0:
        return False
    else_ops = [str(n.op_type) for n in else_graph.node]
    if else_ops == _IF_NMS_GUARD_ELSE_OPS_SIMPLE:
        return True
    if else_ops != _IF_NMS_GUARD_ELSE_OPS_WITH_NESTED_IF:
        return False

    nested_if = else_graph.node[-1]
    nested_attrs = {a.name: a for a in nested_if.attribute}
    if "then_branch" not in nested_attrs or "else_branch" not in nested_attrs:
        return False
    nested_then = nested_attrs["then_branch"].g
    nested_else = nested_attrs["else_branch"].g
    if len(nested_then.node) != 1 or len(nested_else.node) != 1:
        return False
    if str(nested_then.node[0].op_type) != "Squeeze":
        return False
    if str(nested_else.node[0].op_type) != "Identity":
        return False
    return True


def _onnx_dim_to_int(dim_proto: Any) -> int:
    if hasattr(dim_proto, "HasField"):
        if dim_proto.HasField("dim_value"):
            return int(dim_proto.dim_value)
        if dim_proto.HasField("dim_param"):
            return -1
    try:
        value = int(getattr(dim_proto, "dim_value", 0))
        if value > 0:
            return value
    except Exception:
        pass
    return -1


def _value_info_shape(value_info: Any) -> list[int]:
    tensor_type = getattr(getattr(value_info, "type", None), "tensor_type", None)
    if tensor_type is None or not hasattr(tensor_type, "shape"):
        return [1]
    dims = [_onnx_dim_to_int(dim) for dim in tensor_type.shape.dim]
    if len(dims) == 0:
        return [1]
    return [int(v) for v in dims]


def _value_info_tflite_dtype(value_info: Any) -> Optional[str]:
    tensor_type = getattr(getattr(value_info, "type", None), "tensor_type", None)
    if tensor_type is None:
        return None
    elem_type = int(getattr(tensor_type, "elem_type", 0))
    name = str(onnx.TensorProto.DataType.Name(elem_type)).upper()
    mapping = {
        "FLOAT": "FLOAT32",
        "FLOAT16": "FLOAT16",
        "DOUBLE": "FLOAT64",
        "INT8": "INT8",
        "INT16": "INT16",
        "INT32": "INT32",
        "INT64": "INT64",
        "UINT8": "UINT8",
        "UINT16": "UINT16",
        "UINT32": "UINT32",
        "UINT64": "UINT64",
        "BOOL": "BOOL",
    }
    return mapping.get(name, None)


def _apply_value_info_hint_to_tensor(
    *,
    tensor_name: str,
    value_info: Any,
    ctx: Any,
) -> None:
    if value_info is None:
        return
    hinted_shape = [int(v) for v in _value_info_shape(value_info)]
    hinted_dtype = _value_info_tflite_dtype(value_info)
    if tensor_name not in ctx.model_ir.tensors:
        ctx.ensure_tensor(tensor_name, dtype=hinted_dtype, shape=hinted_shape)
        return

    tensor = ctx.model_ir.tensors[tensor_name]
    if hinted_dtype is not None:
        tensor.dtype = str(hinted_dtype)
        if hasattr(ctx, "dtype_map") and isinstance(ctx.dtype_map, dict):
            ctx.dtype_map[str(tensor_name)] = str(hinted_dtype)

    hinted_norm_shape = [int(v) if int(v) > 0 else 1 for v in hinted_shape]
    hinted_signature = [int(v) if int(v) > 0 else -1 for v in hinted_shape]
    current_shape = [int(v) for v in list(tensor.shape)]
    current_signature = (
        [int(v) for v in list(tensor.shape_signature)]
        if tensor.shape_signature is not None
        else [int(v) for v in current_shape]
    )
    current_is_unresolved = (
        len(current_shape) == 1
        and int(current_shape[0]) == 1
        and all(int(v) <= 0 for v in current_signature)
    )
    if len(current_shape) != len(hinted_norm_shape) or current_is_unresolved:
        tensor.shape = hinted_norm_shape
        tensor.shape_signature = hinted_signature


def is_supported_if_axis0_add_branch_pattern(node: Any) -> bool:
    then_graph = node.attrs.get("then_branch", None)
    else_graph = node.attrs.get("else_branch", None)
    if then_graph is None or else_graph is None:
        return False
    if not hasattr(then_graph, "node") or not hasattr(else_graph, "node"):
        return False
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        return False
    if len(then_graph.input) != 0 or len(else_graph.input) != 0:
        return False
    if len(then_graph.output) != 1 or len(else_graph.output) != 1:
        return False
    if len(then_graph.node) != 1 or len(else_graph.node) != 1:
        return False

    then_node = then_graph.node[0]
    else_node = else_graph.node[0]
    if str(then_node.op_type) != "Add" or str(else_node.op_type) != "Add":
        return False
    if len(then_node.output) != 1 or len(else_node.output) != 1:
        return False
    if str(then_node.output[0]) != str(then_graph.output[0].name):
        return False
    if str(else_node.output[0]) != str(else_graph.output[0].name):
        return False

    then_shape = _value_info_shape(then_graph.output[0])
    else_shape = _value_info_shape(else_graph.output[0])
    if len(then_shape) < 1 or len(else_shape) < 1:
        return False
    if len(then_shape) != len(else_shape):
        return False
    if int(then_shape[0]) <= 0 or int(else_shape[0]) <= 0:
        return False
    for idx in range(1, len(then_shape)):
        t_dim = int(then_shape[idx])
        e_dim = int(else_shape[idx])
        if t_dim <= 0 or e_dim <= 0 or t_dim != e_dim:
            return False

    then_dtype = _value_info_tflite_dtype(then_graph.output[0])
    else_dtype = _value_info_tflite_dtype(else_graph.output[0])
    if then_dtype is not None and else_dtype is not None and then_dtype != else_dtype:
        return False

    return True


def _sequenceconstruct_node(graph: Any) -> Optional[Any]:
    if not hasattr(graph, "node") or len(graph.node) == 0:
        return None
    last_node = graph.node[-1]
    if str(last_node.op_type) != "SequenceConstruct":
        return None
    if len(last_node.output) != 1:
        return None
    if not hasattr(graph, "output") or len(graph.output) != 1:
        return None
    if str(graph.output[0].name) != str(last_node.output[0]):
        return None
    return last_node


def _is_supported_if_sequenceconstruct_add_branch(graph: Any) -> bool:
    seq_node = _sequenceconstruct_node(graph)
    if seq_node is None:
        return False
    seq_inputs = [str(name) for name in seq_node.input if str(name) != ""]
    if len(seq_inputs) == 0:
        return False

    producers: Dict[str, Any] = {}
    constant_outputs = {str(initializer.name) for initializer in graph.initializer}
    for graph_node in graph.node:
        op_type = str(graph_node.op_type)
        for output_name in graph_node.output:
            out_name = str(output_name)
            if out_name == "":
                continue
            producers[out_name] = graph_node
            if op_type == "Constant":
                constant_outputs.add(out_name)

    for graph_node in graph.node[:-1]:
        op_type = str(graph_node.op_type)
        if op_type not in {"Add", "Constant"}:
            return False

    for seq_input in seq_inputs:
        producer = producers.get(seq_input, None)
        if producer is None or str(producer.op_type) != "Add":
            return False
        add_inputs = [str(name) for name in producer.input if str(name) != ""]
        if len(add_inputs) != 2:
            return False
        const_count = sum(1 for add_input in add_inputs if add_input in constant_outputs)
        if const_count != 1:
            return False
    return True


def is_supported_if_sequenceconstruct_add_branch_pattern(node: Any) -> bool:
    then_graph = node.attrs.get("then_branch", None)
    else_graph = node.attrs.get("else_branch", None)
    if then_graph is None or else_graph is None:
        return False
    if not hasattr(then_graph, "node") or not hasattr(else_graph, "node"):
        return False
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        return False
    if len(then_graph.input) != 0 or len(else_graph.input) != 0:
        return False
    return (
        _is_supported_if_sequenceconstruct_add_branch(then_graph)
        and _is_supported_if_sequenceconstruct_add_branch(else_graph)
    )


def _is_supported_single_add_branch_graph(graph: Any) -> bool:
    if not hasattr(graph, "node") or not hasattr(graph, "output"):
        return False
    if len(graph.input) != 0 or len(graph.output) != 1:
        return False
    if len(graph.node) != 1:
        return False
    add_node = graph.node[0]
    if str(add_node.op_type) != "Add":
        return False
    if len(add_node.input) != 2 or len(add_node.output) != 1:
        return False
    if str(add_node.output[0]) != str(graph.output[0].name):
        return False
    constant_outputs = {str(initializer.name) for initializer in graph.initializer}
    for graph_node in graph.node:
        if str(graph_node.op_type) == "Constant":
            for output_name in graph_node.output:
                if str(output_name) != "":
                    constant_outputs.add(str(output_name))
    input_names = [str(name) for name in add_node.input if str(name) != ""]
    const_count = sum(1 for input_name in input_names if input_name in constant_outputs)
    return const_count == 1


def is_supported_if_nested_reducemin_add_branch_pattern(node: Any) -> bool:
    then_graph = node.attrs.get("then_branch", None)
    else_graph = node.attrs.get("else_branch", None)
    if then_graph is None or else_graph is None:
        return False
    if not hasattr(then_graph, "node") or not hasattr(else_graph, "node"):
        return False
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        return False
    if len(then_graph.input) != 0 or len(else_graph.input) != 0:
        return False
    if len(then_graph.output) != 1 or len(else_graph.output) != 1:
        return False
    if not _is_supported_single_add_branch_graph(then_graph):
        return False

    expected_else_ops = ["ReduceMin", "ReduceMin", "Greater", "If"]
    else_ops = [str(n.op_type) for n in else_graph.node]
    if else_ops != expected_else_ops:
        return False
    nested_if = else_graph.node[-1]
    if len(nested_if.input) != 1 or len(nested_if.output) != 1:
        return False
    if str(nested_if.output[0]) != str(else_graph.output[0].name):
        return False
    nested_attrs = {a.name: a for a in nested_if.attribute}
    if "then_branch" not in nested_attrs or "else_branch" not in nested_attrs:
        return False
    nested_then = nested_attrs["then_branch"].g
    nested_else = nested_attrs["else_branch"].g
    if not _is_supported_single_add_branch_graph(nested_then):
        return False
    if not _is_supported_single_add_branch_graph(nested_else):
        return False

    then_dtype = _value_info_tflite_dtype(then_graph.output[0])
    else_dtype = _value_info_tflite_dtype(else_graph.output[0])
    if then_dtype is not None and else_dtype is not None and then_dtype != else_dtype:
        return False
    return True


def is_supported_if_generic_branch_mux_pattern(node: Any, ctx: Any = None) -> bool:
    then_graph = node.attrs.get("then_branch", None)
    else_graph = node.attrs.get("else_branch", None)
    if then_graph is None or else_graph is None:
        return False
    if not hasattr(then_graph, "node") or not hasattr(else_graph, "node"):
        return False
    if len(node.inputs) != 1:
        return False
    output_count = int(len(node.outputs))
    if output_count <= 0:
        return False
    if len(then_graph.input) != 0 or len(else_graph.input) != 0:
        return False
    if len(then_graph.output) != output_count or len(else_graph.output) != output_count:
        return False

    def _is_safe_graph(graph: Any) -> bool:
        for graph_node in list(getattr(graph, "node", [])):
            op_type = str(graph_node.op_type)
            if op_type == "If":
                attrs = {a.name: a for a in graph_node.attribute}
                if "then_branch" not in attrs or "else_branch" not in attrs:
                    return False
                if len(graph_node.input) != 1 or len(graph_node.output) <= 0:
                    return False
                if not _is_safe_graph(attrs["then_branch"].g):
                    return False
                if not _is_safe_graph(attrs["else_branch"].g):
                    return False
                continue
            if op_type not in _IF_GENERIC_BRANCH_SAFE_OPS:
                return False
        return True

    if not _is_safe_graph(then_graph):
        return False
    if not _is_safe_graph(else_graph):
        return False

    for output_index in range(output_count):
        then_output = then_graph.output[output_index]
        else_output = else_graph.output[output_index]
        then_dtype = _value_info_tflite_dtype(then_output)
        else_dtype = _value_info_tflite_dtype(else_output)
        if then_dtype is not None and else_dtype is not None and then_dtype != else_dtype:
            return False

        then_shape = _value_info_shape(then_output)
        else_shape = _value_info_shape(else_output)
        if not _are_shapes_broadcast_compatible(then_shape, else_shape):
            return False
    return True


def _ensure_graph_initializers(graph: Any, ctx: Any) -> None:
    for initializer in graph.initializer:
        name = str(initializer.name)
        if name in ctx.model_ir.tensors:
            continue
        value = np.asarray(numpy_helper.to_array(initializer))
        ctx.add_const_tensor(name, value)


def _to_tflite_attr_value(attr: Any) -> Any:
    attr_type = int(attr.type)
    if attr_type == int(onnx.AttributeProto.INT):
        return int(attr.i)
    if attr_type == int(onnx.AttributeProto.FLOAT):
        return float(attr.f)
    if attr_type == int(onnx.AttributeProto.INTS):
        return [int(v) for v in attr.ints]
    if attr_type == int(onnx.AttributeProto.FLOATS):
        return [float(v) for v in attr.floats]
    if attr_type == int(onnx.AttributeProto.STRING):
        return attr.s.decode("utf-8")
    if attr_type == int(onnx.AttributeProto.STRINGS):
        return [v.decode("utf-8") for v in attr.strings]
    if attr_type == int(onnx.AttributeProto.GRAPH):
        return attr.g
    if attr_type == int(onnx.AttributeProto.GRAPHS):
        return [g for g in attr.graphs]
    return None


def _constant_node_to_array(node_proto: Any) -> np.ndarray:
    attrs = {str(attr.name): attr for attr in node_proto.attribute}
    if "value" in attrs and attrs["value"].HasField("t"):
        return np.asarray(numpy_helper.to_array(attrs["value"].t))
    if "value_float" in attrs:
        return np.asarray(float(attrs["value_float"].f), dtype=np.float32)
    if "value_floats" in attrs:
        return np.asarray([float(v) for v in attrs["value_floats"].floats], dtype=np.float32)
    if "value_int" in attrs:
        return np.asarray(int(attrs["value_int"].i), dtype=np.int64)
    if "value_ints" in attrs:
        return np.asarray([int(v) for v in attrs["value_ints"].ints], dtype=np.int64)
    if "value_string" in attrs:
        return np.asarray(attrs["value_string"].s.decode("utf-8"))
    if "value_strings" in attrs:
        return np.asarray([v.decode("utf-8") for v in attrs["value_strings"].strings])
    raise NotImplementedError(
        f"Constant in If branch requires supported value attribute. node={node_proto.name}"
    )


def _wrap_node(
    node_proto: Any,
    *,
    input_name_remap: Optional[Dict[str, str]] = None,
    output_name_remap: Optional[Dict[str, str]] = None,
) -> Any:
    remap_in = input_name_remap if isinstance(input_name_remap, dict) else {}
    remap_out = output_name_remap if isinstance(output_name_remap, dict) else {}
    attrs: Dict[str, Any] = {}
    for attr in node_proto.attribute:
        converted = _to_tflite_attr_value(attr)
        if converted is not None:
            attrs[str(attr.name)] = converted

    wrapped = type("BranchNode", (), {})()
    wrapped.name = str(node_proto.name) if str(node_proto.name) != "" else str(node_proto.op_type)
    wrapped.op = str(node_proto.op_type)
    wrapped.attrs = attrs
    wrapped.inputs = [
        type(
            "In",
            (),
            {"name": (remap_in.get(str(name), str(name)) if str(name) != "" else "")},
        )
        for name in node_proto.input
    ]
    wrapped.outputs = [
        type("Out", (), {"name": remap_out.get(str(name), str(name))})
        for name in node_proto.output
        if str(name) != ""
    ]
    return wrapped


def _np_dtype_from_tflite_dtype(tflite_dtype: str) -> np.dtype:
    dtype = str(tflite_dtype).upper()
    mapping = {
        "FLOAT32": np.float32,
        "FLOAT16": np.float16,
        "INT64": np.int64,
        "INT32": np.int32,
    }
    if dtype in mapping:
        return np.dtype(mapping[dtype])
    return np.dtype(np.float32)


def _tflite_dtype_from_np_dtype(np_dtype: np.dtype) -> str:
    dt = np.dtype(np_dtype)
    if np.issubdtype(dt, np.bool_):
        return "BOOL"
    if np.issubdtype(dt, np.int64):
        return "INT64"
    if np.issubdtype(dt, np.int32):
        return "INT32"
    if np.issubdtype(dt, np.float16):
        return "FLOAT16"
    return "FLOAT32"


def _are_shapes_broadcast_compatible(
    lhs_shape: List[int],
    rhs_shape: List[int],
) -> bool:
    lhs = [int(v) for v in list(lhs_shape)]
    rhs = [int(v) for v in list(rhs_shape)]
    max_rank = int(max(len(lhs), len(rhs)))
    for idx in range(1, max_rank + 1):
        lhs_dim = int(lhs[-idx]) if idx <= len(lhs) else 1
        rhs_dim = int(rhs[-idx]) if idx <= len(rhs) else 1
        if lhs_dim > 0 and rhs_dim > 0 and lhs_dim != rhs_dim and lhs_dim != 1 and rhs_dim != 1:
            return False
    return True


def _tensor_shape_and_signature(
    *,
    tensor_name: str,
    ctx: Any,
) -> tuple[List[int], List[int]]:
    shape = [int(v) for v in ctx.get_tensor_shape(tensor_name)]
    signature = [int(v) for v in list(shape)]
    tensor = ctx.model_ir.tensors.get(str(tensor_name), None)
    if tensor is not None and tensor.shape_signature is not None and len(tensor.shape_signature) == len(shape):
        signature = [int(v) for v in list(tensor.shape_signature)]
    return shape, signature


def _broadcast_shape_signature(
    lhs_signature: List[int],
    rhs_signature: List[int],
) -> Optional[List[int]]:
    lhs = [int(v) for v in list(lhs_signature)]
    rhs = [int(v) for v in list(rhs_signature)]
    max_rank = int(max(len(lhs), len(rhs)))
    out_rev: List[int] = []
    for idx in range(1, max_rank + 1):
        lhs_dim = int(lhs[-idx]) if idx <= len(lhs) else 1
        rhs_dim = int(rhs[-idx]) if idx <= len(rhs) else 1
        if lhs_dim > 0 and rhs_dim > 0:
            if lhs_dim == rhs_dim:
                out_rev.append(lhs_dim)
            elif lhs_dim == 1:
                out_rev.append(rhs_dim)
            elif rhs_dim == 1:
                out_rev.append(lhs_dim)
            else:
                return None
        elif lhs_dim > 0 and rhs_dim <= 0:
            out_rev.append(lhs_dim if lhs_dim != 1 else -1)
        elif lhs_dim <= 0 and rhs_dim > 0:
            out_rev.append(rhs_dim if rhs_dim != 1 else -1)
        else:
            out_rev.append(-1)
    return [int(v) for v in reversed(out_rev)]


def _normalize_onnx_axes(axes: np.ndarray, rank: int) -> List[int]:
    normalized: List[int] = []
    for axis in np.asarray(axes, dtype=np.int64).reshape(-1).tolist():
        axis_i = int(axis)
        if axis_i < 0:
            axis_i += int(rank)
        if axis_i < 0 or axis_i >= int(rank):
            raise ValueError(f"axis out of range: axis={axis} rank={rank}")
        normalized.append(axis_i)
    return normalized


def _const_fold_slice(
    data: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    axes: Optional[np.ndarray],
    steps: Optional[np.ndarray],
) -> np.ndarray:
    x = np.asarray(data)
    rank = int(x.ndim)
    starts_v = np.asarray(starts, dtype=np.int64).reshape(-1)
    ends_v = np.asarray(ends, dtype=np.int64).reshape(-1)
    if axes is None:
        axes_v = np.arange(int(starts_v.size), dtype=np.int64)
    else:
        axes_v = np.asarray(axes, dtype=np.int64).reshape(-1)
    if steps is None:
        steps_v = np.ones(int(starts_v.size), dtype=np.int64)
    else:
        steps_v = np.asarray(steps, dtype=np.int64).reshape(-1)
    if int(starts_v.size) != int(ends_v.size) or int(starts_v.size) != int(axes_v.size):
        raise ValueError("Slice const-fold requires starts/ends/axes lengths to match.")
    if int(steps_v.size) != int(starts_v.size):
        raise ValueError("Slice const-fold requires steps length to match starts.")

    slices: List[slice] = [slice(None)] * rank
    for start, end, axis, step in zip(starts_v.tolist(), ends_v.tolist(), axes_v.tolist(), steps_v.tolist()):
        axis_i = int(axis)
        if axis_i < 0:
            axis_i += rank
        if axis_i < 0 or axis_i >= rank:
            raise ValueError(f"Slice axis out of range in const-fold: axis={axis} rank={rank}")
        step_i = int(step)
        if step_i == 0:
            raise ValueError("Slice step must not be 0 in const-fold.")
        dim = int(x.shape[axis_i])
        start_i = int(start)
        end_i = int(end)
        if step_i > 0:
            if start_i < 0:
                start_i += dim
            if end_i < 0:
                end_i += dim
            start_i = min(max(start_i, 0), dim)
            end_i = min(max(end_i, 0), dim)
        else:
            if start_i < 0:
                start_i += dim
            if end_i < 0:
                end_i += dim
            start_i = min(max(start_i, -1), dim - 1)
            end_i = min(max(end_i, -1), dim - 1)
        slices[axis_i] = slice(start_i, end_i, step_i)
    return np.asarray(x[tuple(slices)])


def _add_cond_gate_to_slice_output(
    *,
    cond_name: str,
    candidate_name: str,
    output_name: str,
    ctx: Any,
) -> None:
    ctx.ensure_tensor(cond_name)
    ctx.ensure_tensor(candidate_name)
    ctx.ensure_tensor(output_name)

    cond_i32_name = ctx.add_intermediate_tensor(
        f"{output_name}_if_cond_i32",
        dtype="INT32",
        shape=[],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="CAST",
            inputs=[cond_name],
            outputs=[cond_i32_name],
            options={"inDataType": "BOOL", "outDataType": "INT32"},
        )
    )

    one_name = ctx.add_const_tensor(
        f"{output_name}_if_one",
        np.asarray(1, dtype=np.int32),
    )
    not_cond_i32_name = ctx.add_intermediate_tensor(
        f"{output_name}_if_not_cond_i32",
        dtype="INT32",
        shape=[],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SUB",
            inputs=[one_name, cond_i32_name],
            outputs=[not_cond_i32_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )

    candidate_shape_name = ctx.add_intermediate_tensor(
        f"{output_name}_if_candidate_shape",
        dtype="INT32",
        shape=[1],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SHAPE",
            inputs=[candidate_name],
            outputs=[candidate_shape_name],
            options={"outType": "INT32"},
        )
    )

    gated_size_name = ctx.add_intermediate_tensor(
        f"{output_name}_if_gated_size",
        dtype="INT32",
        shape=[1],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="MUL",
            inputs=[candidate_shape_name, not_cond_i32_name],
            outputs=[gated_size_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )

    begin_name = ctx.add_const_tensor(
        f"{output_name}_if_slice_begin",
        np.asarray([0], dtype=np.int32),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SLICE",
            inputs=[candidate_name, begin_name, gated_size_name],
            outputs=[output_name],
        )
    )


def _lower_graph_nodes(
    *,
    graph: Any,
    ctx: Any,
    input_name_remap: Optional[Dict[str, str]] = None,
    output_name_remap: Optional[Dict[str, str]] = None,
    skip_sequenceconstruct: bool = False,
) -> None:
    from onnx2tf.tflite_builder.dispatcher import dispatch_node

    remap_in = input_name_remap if isinstance(input_name_remap, dict) else {}
    remap_out = output_name_remap if isinstance(output_name_remap, dict) else {}
    value_info_map: Dict[str, Any] = {}
    for value_info in list(getattr(graph, "value_info", [])) + list(getattr(graph, "input", [])) + list(getattr(graph, "output", [])):
        value_name = str(getattr(value_info, "name", ""))
        if value_name != "":
            value_info_map[value_name] = value_info

    for graph_node in graph.node:
        op_type = str(graph_node.op_type)
        for input_name in graph_node.input:
            original_name = str(input_name)
            if original_name == "":
                continue
            mapped_name = remap_in.get(original_name, original_name)
            _apply_value_info_hint_to_tensor(
                tensor_name=mapped_name,
                value_info=value_info_map.get(original_name, None),
                ctx=ctx,
            )
        for output_name in graph_node.output:
            original_name = str(output_name)
            if original_name == "":
                continue
            mapped_name = remap_out.get(original_name, original_name)
            _apply_value_info_hint_to_tensor(
                tensor_name=mapped_name,
                value_info=value_info_map.get(original_name, None),
                ctx=ctx,
            )

        if op_type == "Constant":
            if len(graph_node.output) != 1:
                raise NotImplementedError(
                    f"Constant in branch requires 1 output in flatbuffer_direct. node={graph_node.name}"
                )
            out_name = remap_out.get(str(graph_node.output[0]), str(graph_node.output[0]))
            const_value = _constant_node_to_array(graph_node)
            if out_name in ctx.model_ir.tensors:
                out_tensor = ctx.model_ir.tensors[out_name]
                out_tensor.data = const_value
                out_tensor.dtype = _tflite_dtype_from_np_dtype(const_value.dtype)
                out_tensor.shape = [int(v) for v in const_value.shape]
                out_tensor.shape_signature = [int(v) for v in const_value.shape]
                ctx.constants[out_name] = const_value
            else:
                ctx.add_const_tensor(out_name, const_value)
            continue

        if op_type == "Concat" and len(graph_node.output) == 1:
            concat_inputs = [
                remap_in.get(str(name), str(name))
                for name in graph_node.input
                if str(name) != ""
            ]
            concat_values = [ctx.get_constant_array(name) for name in concat_inputs]
            if len(concat_values) > 0 and all(value is not None for value in concat_values):
                axis = 0
                for attr in graph_node.attribute:
                    if str(attr.name) == "axis":
                        axis = int(attr.i)
                        break
                concat_arrays = [np.asarray(value) for value in concat_values]
                try:
                    concat_result = np.concatenate(concat_arrays, axis=axis)
                except Exception:
                    concat_result = None
                if concat_result is not None:
                    out_name = remap_out.get(str(graph_node.output[0]), str(graph_node.output[0]))
                    if out_name in ctx.model_ir.tensors:
                        out_tensor = ctx.model_ir.tensors[out_name]
                        out_tensor.data = concat_result
                        out_tensor.dtype = _tflite_dtype_from_np_dtype(concat_result.dtype)
                        out_tensor.shape = [int(v) for v in concat_result.shape]
                        out_tensor.shape_signature = [int(v) for v in concat_result.shape]
                        ctx.constants[out_name] = concat_result
                    else:
                        ctx.add_const_tensor(out_name, concat_result)
                    continue

        if op_type == "Slice" and len(graph_node.output) == 1 and len(graph_node.input) >= 3:
            data_name = remap_in.get(str(graph_node.input[0]), str(graph_node.input[0]))
            starts_name = remap_in.get(str(graph_node.input[1]), str(graph_node.input[1]))
            ends_name = remap_in.get(str(graph_node.input[2]), str(graph_node.input[2]))
            data_value = ctx.get_constant_array(data_name)
            starts_value = ctx.get_constant_array(starts_name)
            ends_value = ctx.get_constant_array(ends_name)
            axes_value = None
            steps_value = None
            if len(graph_node.input) >= 4 and str(graph_node.input[3]) != "":
                axes_name = remap_in.get(str(graph_node.input[3]), str(graph_node.input[3]))
                axes_value = ctx.get_constant_array(axes_name)
            if len(graph_node.input) >= 5 and str(graph_node.input[4]) != "":
                steps_name = remap_in.get(str(graph_node.input[4]), str(graph_node.input[4]))
                steps_value = ctx.get_constant_array(steps_name)
            if data_value is not None and starts_value is not None and ends_value is not None:
                try:
                    slice_result = _const_fold_slice(
                        data=np.asarray(data_value),
                        starts=np.asarray(starts_value),
                        ends=np.asarray(ends_value),
                        axes=(np.asarray(axes_value) if axes_value is not None else None),
                        steps=(np.asarray(steps_value) if steps_value is not None else None),
                    )
                except Exception:
                    slice_result = None
                if slice_result is not None:
                    out_name = remap_out.get(str(graph_node.output[0]), str(graph_node.output[0]))
                    if out_name in ctx.model_ir.tensors:
                        out_tensor = ctx.model_ir.tensors[out_name]
                        out_tensor.data = slice_result
                        out_tensor.dtype = _tflite_dtype_from_np_dtype(slice_result.dtype)
                        out_tensor.shape = [int(v) for v in slice_result.shape]
                        out_tensor.shape_signature = [int(v) for v in slice_result.shape]
                        ctx.constants[out_name] = slice_result
                    else:
                        ctx.add_const_tensor(out_name, slice_result)
                    continue

        if op_type == "Unsqueeze" and len(graph_node.output) == 1 and len(graph_node.input) >= 1:
            data_name = remap_in.get(str(graph_node.input[0]), str(graph_node.input[0]))
            data_value = ctx.get_constant_array(data_name)
            axes_value = None
            if len(graph_node.input) >= 2 and str(graph_node.input[1]) != "":
                axes_name = remap_in.get(str(graph_node.input[1]), str(graph_node.input[1]))
                axes_value = ctx.get_constant_array(axes_name)
            if axes_value is None:
                for attr in graph_node.attribute:
                    if str(attr.name) == "axes":
                        axes_value = np.asarray([int(v) for v in list(attr.ints)], dtype=np.int64)
                        break
            if data_value is not None and axes_value is not None:
                try:
                    result = np.asarray(data_value)
                    axes_norm = _normalize_onnx_axes(
                        np.asarray(axes_value),
                        int(result.ndim + np.asarray(axes_value).reshape(-1).size),
                    )
                    for axis in sorted(axes_norm):
                        result = np.expand_dims(result, axis=axis)
                except Exception:
                    result = None
                if result is not None:
                    out_name = remap_out.get(str(graph_node.output[0]), str(graph_node.output[0]))
                    if out_name in ctx.model_ir.tensors:
                        out_tensor = ctx.model_ir.tensors[out_name]
                        out_tensor.data = result
                        out_tensor.dtype = _tflite_dtype_from_np_dtype(result.dtype)
                        out_tensor.shape = [int(v) for v in result.shape]
                        out_tensor.shape_signature = [int(v) for v in result.shape]
                        ctx.constants[out_name] = result
                    else:
                        ctx.add_const_tensor(out_name, result)
                    continue

        if op_type == "Squeeze" and len(graph_node.output) == 1 and len(graph_node.input) >= 1:
            data_name = remap_in.get(str(graph_node.input[0]), str(graph_node.input[0]))
            data_value = ctx.get_constant_array(data_name)
            axes_value = None
            if len(graph_node.input) >= 2 and str(graph_node.input[1]) != "":
                axes_name = remap_in.get(str(graph_node.input[1]), str(graph_node.input[1]))
                axes_value = ctx.get_constant_array(axes_name)
            if axes_value is None:
                for attr in graph_node.attribute:
                    if str(attr.name) == "axes":
                        axes_value = np.asarray([int(v) for v in list(attr.ints)], dtype=np.int64)
                        break
            if data_value is not None:
                try:
                    result = np.asarray(data_value)
                    if axes_value is None:
                        result = np.squeeze(result)
                    else:
                        axes_norm = _normalize_onnx_axes(
                            np.asarray(axes_value),
                            int(result.ndim),
                        )
                        result = np.squeeze(result, axis=tuple(axes_norm))
                except Exception:
                    result = None
                if result is not None:
                    out_name = remap_out.get(str(graph_node.output[0]), str(graph_node.output[0]))
                    if out_name in ctx.model_ir.tensors:
                        out_tensor = ctx.model_ir.tensors[out_name]
                        out_tensor.data = result
                        out_tensor.dtype = _tflite_dtype_from_np_dtype(result.dtype)
                        out_tensor.shape = [int(v) for v in np.asarray(result).shape]
                        out_tensor.shape_signature = [int(v) for v in np.asarray(result).shape]
                        ctx.constants[out_name] = np.asarray(result)
                    else:
                        ctx.add_const_tensor(out_name, np.asarray(result))
                    continue

        if op_type == "Gather" and len(graph_node.input) >= 2 and len(graph_node.output) == 1:
            data_name = remap_in.get(str(graph_node.input[0]), str(graph_node.input[0]))
            indices_name = remap_in.get(str(graph_node.input[1]), str(graph_node.input[1]))
            data_value = ctx.get_constant_array(data_name)
            indices_value = ctx.get_constant_array(indices_name)
            if data_value is not None and indices_value is not None:
                axis = 0
                for attr in graph_node.attribute:
                    if str(attr.name) == "axis":
                        axis = int(attr.i)
                        break
                gather_result = np.take(
                    np.asarray(data_value),
                    np.asarray(indices_value, dtype=np.int64),
                    axis=axis,
                )
                out_name = remap_out.get(str(graph_node.output[0]), str(graph_node.output[0]))
                if out_name in ctx.model_ir.tensors:
                    out_tensor = ctx.model_ir.tensors[out_name]
                    out_tensor.data = gather_result
                    out_tensor.dtype = _tflite_dtype_from_np_dtype(gather_result.dtype)
                    out_tensor.shape = [int(v) for v in gather_result.shape]
                    out_tensor.shape_signature = [int(v) for v in gather_result.shape]
                    ctx.constants[out_name] = gather_result
                else:
                    ctx.add_const_tensor(out_name, gather_result)
                continue

        if op_type == "SequenceConstruct":
            if skip_sequenceconstruct:
                continue
            raise NotImplementedError(
                f"SequenceConstruct in branch is unsupported in flatbuffer_direct. node={graph_node.name}"
            )

        if op_type == "Equal" and len(graph_node.input) >= 2 and len(graph_node.output) == 1:
            lhs_name = remap_in.get(str(graph_node.input[0]), str(graph_node.input[0]))
            rhs_name = remap_in.get(str(graph_node.input[1]), str(graph_node.input[1]))
            lhs = ctx.get_constant_array(lhs_name)
            rhs = ctx.get_constant_array(rhs_name)
            if lhs is not None and rhs is not None:
                out_name = remap_out.get(str(graph_node.output[0]), str(graph_node.output[0]))
                const_value = np.asarray(np.equal(np.asarray(lhs), np.asarray(rhs)), dtype=np.bool_)
                if out_name in ctx.model_ir.tensors:
                    out_tensor = ctx.model_ir.tensors[out_name]
                    out_tensor.data = const_value
                    out_tensor.dtype = _tflite_dtype_from_np_dtype(const_value.dtype)
                    out_tensor.shape = [int(v) for v in const_value.shape]
                    out_tensor.shape_signature = [int(v) for v in const_value.shape]
                    ctx.constants[out_name] = const_value
                else:
                    ctx.add_const_tensor(out_name, const_value)
                continue

        if op_type == "If":
            wrapped_if = _wrap_node(
                graph_node,
                input_name_remap=remap_in,
                output_name_remap=remap_out,
            )
            dispatch_node(wrapped_if, ctx)
            continue

        wrapped = _wrap_node(
            graph_node,
            input_name_remap=remap_in,
            output_name_remap=remap_out,
        )
        if op_type == "NonMaxSuppression":
            for wrapped_output in wrapped.outputs:
                out_name = str(wrapped_output.name)
                ctx.ensure_tensor(out_name, dtype="INT32")
                ctx.model_ir.tensors[out_name].dtype = "INT32"
                if hasattr(ctx, "dtype_map") and isinstance(ctx.dtype_map, dict):
                    ctx.dtype_map[out_name] = "INT32"
        if op_type == "Squeeze" and len(graph_node.input) >= 1 and len(graph_node.output) == 1:
            squeeze_input_name = remap_in.get(
                str(graph_node.input[0]),
                str(graph_node.input[0]),
            )
            squeeze_output_name = remap_out.get(
                str(graph_node.output[0]),
                str(graph_node.output[0]),
            )
            if squeeze_input_name != "" and squeeze_output_name != "":
                axes: List[int] = []
                if len(graph_node.input) >= 2:
                    axes_name = remap_in.get(
                        str(graph_node.input[1]),
                        str(graph_node.input[1]),
                    )
                    axes_value = ctx.get_constant_array(axes_name)
                    if axes_value is not None:
                        axes = [int(v) for v in np.asarray(axes_value).reshape(-1)]
                if len(axes) == 0:
                    for attr in graph_node.attribute:
                        if str(attr.name) == "axes":
                            axes = [int(v) for v in list(attr.ints)]
                            break
                squeeze_input_shape = [int(v) for v in ctx.get_tensor_shape(squeeze_input_name)]
                if int(len(squeeze_input_shape)) == 1 and axes == [1]:
                    # Branch-local shape metadata may collapse [K,1] to rank-1.
                    # Fall back to a rank-preserving alias instead of rejecting axis=1.
                    ctx.ensure_tensor(
                        squeeze_output_name,
                        dtype=str(ctx.get_tensor_dtype(squeeze_input_name)).upper(),
                        shape=[-1],
                    )
                    squeeze_output_tensor = ctx.model_ir.tensors.get(squeeze_output_name, None)
                    if squeeze_output_tensor is not None:
                        squeeze_output_tensor.dtype = str(ctx.get_tensor_dtype(squeeze_input_name)).upper()
                        squeeze_output_tensor.shape = [-1]
                        squeeze_output_tensor.shape_signature = [-1]
                    reshape_shape_name = ctx.add_const_tensor(
                        f"{squeeze_output_name}_if_branch_squeeze_shape",
                        np.asarray([-1], dtype=np.int32),
                    )
                    ctx.add_operator(
                        OperatorIR(
                            op_type="RESHAPE",
                            inputs=[squeeze_input_name, reshape_shape_name],
                            outputs=[squeeze_output_name],
                            options={"newShape": [-1]},
                        )
                    )
                    continue
        dispatch_node(wrapped, ctx)


def _cast_to_dtype_for_if(
    *,
    tensor_name: str,
    target_dtype: str,
    output_name: str,
    suffix: str,
    ctx: Any,
) -> str:
    in_dtype = str(ctx.get_tensor_dtype(tensor_name)).upper()
    if in_dtype == target_dtype:
        return tensor_name
    casted_name = ctx.add_intermediate_tensor(
        f"{output_name}_if_{suffix}_cast",
        dtype=target_dtype,
        shape=[int(v) for v in ctx.get_tensor_shape(tensor_name)],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="CAST",
            inputs=[tensor_name],
            outputs=[casted_name],
            options={"inDataType": in_dtype, "outDataType": target_dtype},
        )
    )
    return casted_name


def _reshape_if_tensor_to_vector(
    *,
    tensor_name: str,
    output_name: str,
    suffix: str,
    ctx: Any,
) -> str:
    reshaped_name = ctx.add_intermediate_tensor(
        f"{output_name}_if_{suffix}_reshape1d",
        dtype=str(ctx.get_tensor_dtype(tensor_name)).upper(),
        shape=[-1],
    )
    reshape_shape_name = ctx.add_const_tensor(
        f"{reshaped_name}_shape",
        np.asarray([-1], dtype=np.int32),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="RESHAPE",
            inputs=[tensor_name, reshape_shape_name],
            outputs=[reshaped_name],
            options={"newShape": [-1]},
        )
    )
    return reshaped_name


def _build_if_axis0_tensor_mux(
    *,
    cond_name: str,
    output_name: str,
    expected_output_dtype: str,
    then_output_name: str,
    then_shape: List[int],
    else_output_name: str,
    else_shape: List[int],
    ctx: Any,
) -> None:
    then_first_dim = int(then_shape[0])
    else_first_dim = int(else_shape[0])
    tail_shape = [int(v) for v in list(then_shape[1:])]
    rank = int(len(then_shape))

    then_output_name = _cast_to_dtype_for_if(
        tensor_name=then_output_name,
        target_dtype=expected_output_dtype,
        output_name=output_name,
        suffix="then",
        ctx=ctx,
    )
    else_output_name = _cast_to_dtype_for_if(
        tensor_name=else_output_name,
        target_dtype=expected_output_dtype,
        output_name=output_name,
        suffix="else",
        ctx=ctx,
    )

    merged_output_name = ctx.add_intermediate_tensor(
        f"{output_name}_if_merged",
        dtype=expected_output_dtype,
        shape=[int(then_first_dim + else_first_dim)] + [int(v) for v in tail_shape],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="CONCATENATION",
            inputs=[then_output_name, else_output_name],
            outputs=[merged_output_name],
            options={"axis": 0, "fusedActivationFunction": "NONE"},
        )
    )

    cond_i32_name = ctx.add_intermediate_tensor(
        f"{output_name}_if_cond_i32",
        dtype="INT32",
        shape=[1],
    )
    cond_dtype = str(ctx.get_tensor_dtype(cond_name)).upper()
    ctx.add_operator(
        OperatorIR(
            op_type="CAST",
            inputs=[cond_name],
            outputs=[cond_i32_name],
            options={"inDataType": cond_dtype, "outDataType": "INT32"},
        )
    )
    one_i32_name = ctx.add_const_tensor(
        f"{output_name}_if_one",
        np.asarray(1, dtype=np.int32),
    )
    not_cond_i32_name = ctx.add_intermediate_tensor(
        f"{output_name}_if_not_cond_i32",
        dtype="INT32",
        shape=[1],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SUB",
            inputs=[one_i32_name, cond_i32_name],
            outputs=[not_cond_i32_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )

    then_first_i32_name = ctx.add_const_tensor(
        f"{output_name}_if_then_first_dim",
        np.asarray(int(then_first_dim), dtype=np.int32),
    )
    else_first_i32_name = ctx.add_const_tensor(
        f"{output_name}_if_else_first_dim",
        np.asarray(int(else_first_dim), dtype=np.int32),
    )

    begin_axis0_name = ctx.add_intermediate_tensor(
        f"{output_name}_if_begin_axis0",
        dtype="INT32",
        shape=[1],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="MUL",
            inputs=[not_cond_i32_name, then_first_i32_name],
            outputs=[begin_axis0_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )

    size_then_name = ctx.add_intermediate_tensor(
        f"{output_name}_if_size_then",
        dtype="INT32",
        shape=[1],
    )
    size_else_name = ctx.add_intermediate_tensor(
        f"{output_name}_if_size_else",
        dtype="INT32",
        shape=[1],
    )
    size_axis0_name = ctx.add_intermediate_tensor(
        f"{output_name}_if_size_axis0",
        dtype="INT32",
        shape=[1],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="MUL",
            inputs=[cond_i32_name, then_first_i32_name],
            outputs=[size_then_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="MUL",
            inputs=[not_cond_i32_name, else_first_i32_name],
            outputs=[size_else_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="ADD",
            inputs=[size_then_name, size_else_name],
            outputs=[size_axis0_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )

    if rank == 1:
        begin_name = begin_axis0_name
        size_name = size_axis0_name
    else:
        begin_tail_name = ctx.add_const_tensor(
            f"{output_name}_if_begin_tail",
            np.asarray([0 for _ in range(rank - 1)], dtype=np.int32),
        )
        size_tail_name = ctx.add_const_tensor(
            f"{output_name}_if_size_tail",
            np.asarray([int(v) for v in tail_shape], dtype=np.int32),
        )
        begin_name = ctx.add_intermediate_tensor(
            f"{output_name}_if_begin",
            dtype="INT32",
            shape=[rank],
        )
        size_name = ctx.add_intermediate_tensor(
            f"{output_name}_if_size",
            dtype="INT32",
            shape=[rank],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CONCATENATION",
                inputs=[begin_axis0_name, begin_tail_name],
                outputs=[begin_name],
                options={"axis": 0, "fusedActivationFunction": "NONE"},
            )
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CONCATENATION",
                inputs=[size_axis0_name, size_tail_name],
                outputs=[size_name],
                options={"axis": 0, "fusedActivationFunction": "NONE"},
            )
        )

    ctx.add_operator(
        OperatorIR(
            op_type="SLICE",
            inputs=[merged_output_name, begin_name, size_name],
            outputs=[output_name],
        )
    )

    output_tensor = ctx.model_ir.tensors.get(output_name, None)
    if output_tensor is not None:
        output_tensor.dtype = expected_output_dtype
        output_tensor.shape = [int(max(then_first_dim, else_first_dim))] + [int(v) for v in tail_shape]
        output_tensor.shape_signature = [-1] + [int(v) for v in tail_shape]


def _build_if_axis0_add_branch_mux(node: Any, ctx: Any) -> None:
    cond_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(cond_name)
    ctx.ensure_tensor(output_name)
    expected_output_dtype = str(ctx.get_tensor_dtype(output_name)).upper()

    then_graph = node.attrs["then_branch"]
    else_graph = node.attrs["else_branch"]
    _ensure_graph_initializers(then_graph, ctx)
    _ensure_graph_initializers(else_graph, ctx)

    then_shape = [int(v) for v in _value_info_shape(then_graph.output[0])]
    else_shape = [int(v) for v in _value_info_shape(else_graph.output[0])]

    then_output_name = ctx.add_intermediate_tensor(
        f"{output_name}_if_then",
        dtype=expected_output_dtype,
        shape=[int(v) for v in list(then_shape)],
    )
    else_output_name = ctx.add_intermediate_tensor(
        f"{output_name}_if_else",
        dtype=expected_output_dtype,
        shape=[int(v) for v in list(else_shape)],
    )

    _lower_graph_nodes(
        graph=then_graph,
        ctx=ctx,
        output_name_remap={
            str(then_graph.output[0].name): then_output_name,
        },
    )
    _lower_graph_nodes(
        graph=else_graph,
        ctx=ctx,
        output_name_remap={
            str(else_graph.output[0].name): else_output_name,
        },
    )

    _build_if_axis0_tensor_mux(
        cond_name=cond_name,
        output_name=output_name,
        expected_output_dtype=expected_output_dtype,
        then_output_name=then_output_name,
        then_shape=then_shape,
        else_output_name=else_output_name,
        else_shape=else_shape,
        ctx=ctx,
    )


def _build_sequenceconstruct_branch_tensor(
    *,
    graph: Any,
    branch_name: str,
    output_name: str,
    expected_output_dtype: str,
    ctx: Any,
) -> tuple[str, List[int]]:
    _ensure_graph_initializers(graph, ctx)
    seq_node = _sequenceconstruct_node(graph)
    if seq_node is None:
        raise NotImplementedError(
            f"If SequenceConstruct-branch expects terminal SequenceConstruct. branch={branch_name}"
        )

    _lower_graph_nodes(
        graph=graph,
        ctx=ctx,
        skip_sequenceconstruct=True,
    )

    seq_input_names = [str(name) for name in seq_node.input if str(name) != ""]
    if len(seq_input_names) == 0:
        raise NotImplementedError(
            f"If SequenceConstruct-branch requires non-empty inputs. branch={branch_name}"
        )

    casted_names: List[str] = []
    input_shapes: List[List[int]] = []
    for idx, seq_input_name in enumerate(seq_input_names):
        ctx.ensure_tensor(seq_input_name)
        casted_name = _cast_to_dtype_for_if(
            tensor_name=seq_input_name,
            target_dtype=expected_output_dtype,
            output_name=output_name,
            suffix=f"{branch_name}_{idx}",
            ctx=ctx,
        )
        shape = [int(v) for v in ctx.get_tensor_shape(casted_name)]
        if len(shape) == 0:
            raise NotImplementedError(
                f"If SequenceConstruct-branch requires rank>=1 tensors. branch={branch_name} tensor={casted_name}"
            )
        casted_names.append(casted_name)
        input_shapes.append(shape)

    base_rank = len(input_shapes[0])
    base_tail = [int(v) for v in input_shapes[0][1:]]
    first_dim_sum = int(input_shapes[0][0])
    if first_dim_sum <= 0:
        raise NotImplementedError(
            f"If SequenceConstruct-branch requires static positive first dim. branch={branch_name}"
        )
    for shape in input_shapes[1:]:
        if len(shape) != base_rank:
            raise NotImplementedError(
                f"If SequenceConstruct-branch requires uniform rank. branch={branch_name}"
            )
        if int(shape[0]) <= 0:
            raise NotImplementedError(
                f"If SequenceConstruct-branch requires static positive first dim. branch={branch_name}"
            )
        for axis, (lhs_dim, rhs_dim) in enumerate(zip(base_tail, shape[1:]), start=1):
            if int(lhs_dim) != int(rhs_dim):
                raise NotImplementedError(
                    (
                        "If SequenceConstruct-branch requires matching trailing dims. "
                        f"branch={branch_name} axis={axis} lhs={lhs_dim} rhs={rhs_dim}"
                    )
                )
        first_dim_sum += int(shape[0])

    if len(casted_names) == 1:
        return casted_names[0], input_shapes[0]

    merged_name = ctx.add_intermediate_tensor(
        f"{output_name}_if_{branch_name}_seq_merged",
        dtype=expected_output_dtype,
        shape=[int(first_dim_sum)] + [int(v) for v in base_tail],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="CONCATENATION",
            inputs=casted_names,
            outputs=[merged_name],
            options={"axis": 0, "fusedActivationFunction": "NONE"},
        )
    )
    return merged_name, [int(first_dim_sum)] + [int(v) for v in base_tail]


def _build_if_sequenceconstruct_add_branch_mux(node: Any, ctx: Any) -> None:
    cond_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(cond_name)
    ctx.ensure_tensor(output_name)
    expected_output_dtype = str(ctx.get_tensor_dtype(output_name)).upper()

    then_graph = node.attrs["then_branch"]
    else_graph = node.attrs["else_branch"]

    then_output_name, then_shape = _build_sequenceconstruct_branch_tensor(
        graph=then_graph,
        branch_name="then",
        output_name=output_name,
        expected_output_dtype=expected_output_dtype,
        ctx=ctx,
    )
    else_output_name, else_shape = _build_sequenceconstruct_branch_tensor(
        graph=else_graph,
        branch_name="else",
        output_name=output_name,
        expected_output_dtype=expected_output_dtype,
        ctx=ctx,
    )

    _build_if_axis0_tensor_mux(
        cond_name=cond_name,
        output_name=output_name,
        expected_output_dtype=expected_output_dtype,
        then_output_name=then_output_name,
        then_shape=then_shape,
        else_output_name=else_output_name,
        else_shape=else_shape,
        ctx=ctx,
    )


def _emit_where_mux(
    *,
    cond_name: str,
    then_name: str,
    else_name: str,
    output_name: str,
    output_shape: List[int],
    output_dtype: str,
    node_name: str,
    ctx: Any,
) -> None:
    from onnx2tf.tflite_builder.dispatcher import dispatch_node

    ctx.ensure_tensor(cond_name)
    ctx.ensure_tensor(then_name)
    ctx.ensure_tensor(else_name)
    ctx.ensure_tensor(output_name, dtype=output_dtype, shape=output_shape)

    where_node = type("IfWhereNode", (), {})()
    where_node.name = node_name
    where_node.op = "Where"
    where_node.attrs = {}
    where_node.inputs = [
        type("In", (), {"name": str(cond_name)}),
        type("In", (), {"name": str(then_name)}),
        type("In", (), {"name": str(else_name)}),
    ]
    where_node.outputs = [
        type("Out", (), {"name": str(output_name)}),
    ]
    dispatch_node(where_node, ctx)


def _resolve_if_branch_output_name(
    *,
    graph_output_name: str,
    remapped_name: str,
    ctx: Any,
) -> str:
    if str(remapped_name) in ctx.model_ir.tensors:
        return str(remapped_name)
    if str(graph_output_name) in ctx.model_ir.tensors:
        return str(graph_output_name)
    raise NotImplementedError(
        "If branch output tensor was not materialized in flatbuffer_direct lowering. "
        f"graph_output={graph_output_name} remapped_output={remapped_name}"
    )


def _build_if_generic_branch_mux(node: Any, ctx: Any) -> None:
    if not is_supported_if_generic_branch_mux_pattern(node, ctx):
        raise NotImplementedError(
            "If generic branch-mux lowering requires 1 condition input, no branch graph inputs, "
            "and matching then/else output signatures."
        )

    cond_name = node.inputs[0].name
    ctx.ensure_tensor(cond_name)

    then_graph = node.attrs["then_branch"]
    else_graph = node.attrs["else_branch"]
    _ensure_graph_initializers(then_graph, ctx)
    _ensure_graph_initializers(else_graph, ctx)

    then_output_remap: Dict[str, str] = {}
    else_output_remap: Dict[str, str] = {}
    node_name = str(node.name) if str(getattr(node, "name", "")) != "" else "if"
    output_count = int(len(node.outputs))
    for output_index in range(output_count):
        then_output_name = str(then_graph.output[output_index].name)
        else_output_name = str(else_graph.output[output_index].name)
        then_output_remap[then_output_name] = f"{node_name}_if_then_output_{output_index}"
        else_output_remap[else_output_name] = f"{node_name}_if_else_output_{output_index}"

    _lower_graph_nodes(
        graph=then_graph,
        ctx=ctx,
        output_name_remap=then_output_remap,
    )
    _lower_graph_nodes(
        graph=else_graph,
        ctx=ctx,
        output_name_remap=else_output_remap,
    )

    for output_index, output_obj in enumerate(node.outputs):
        output_name = str(output_obj.name)
        then_graph_output_name = str(then_graph.output[output_index].name)
        else_graph_output_name = str(else_graph.output[output_index].name)
        then_output_name = _resolve_if_branch_output_name(
            graph_output_name=then_graph_output_name,
            remapped_name=then_output_remap[then_graph_output_name],
            ctx=ctx,
        )
        else_output_name = _resolve_if_branch_output_name(
            graph_output_name=else_graph_output_name,
            remapped_name=else_output_remap[else_graph_output_name],
            ctx=ctx,
        )

        ctx.ensure_tensor(output_name)
        ctx.ensure_tensor(then_output_name)
        ctx.ensure_tensor(else_output_name)
        expected_output_dtype = str(ctx.get_tensor_dtype(output_name)).upper()

        then_output_name = _cast_to_dtype_for_if(
            tensor_name=then_output_name,
            target_dtype=expected_output_dtype,
            output_name=output_name,
            suffix=f"generic_then_{output_index}",
            ctx=ctx,
        )
        else_output_name = _cast_to_dtype_for_if(
            tensor_name=else_output_name,
            target_dtype=expected_output_dtype,
            output_name=output_name,
            suffix=f"generic_else_{output_index}",
            ctx=ctx,
        )

        then_shape, then_signature = _tensor_shape_and_signature(
            tensor_name=then_output_name,
            ctx=ctx,
        )
        else_shape, else_signature = _tensor_shape_and_signature(
            tensor_name=else_output_name,
            ctx=ctx,
        )
        if not _are_shapes_broadcast_compatible(then_signature, else_signature):
            raise NotImplementedError(
                "If generic branch-mux requires broadcast-compatible then/else outputs. "
                f"node={node.name} output_index={output_index} "
                f"then_shape={then_shape} else_shape={else_shape}"
            )
        inferred_signature = _broadcast_shape_signature(then_signature, else_signature)
        if inferred_signature is None:
            raise NotImplementedError(
                "If generic branch-mux failed to infer broadcast shape/signature. "
                f"node={node.name} output_index={output_index} "
                f"then_signature={then_signature} else_signature={else_signature}"
            )
        expected_output_shape = [int(v) if int(v) > 0 else 1 for v in inferred_signature]
        output_tensor = ctx.model_ir.tensors.get(output_name, None)
        if output_tensor is not None:
            output_tensor.shape = [int(v) for v in expected_output_shape]
            output_tensor.shape_signature = [int(v) for v in inferred_signature]

        _emit_where_mux(
            cond_name=cond_name,
            then_name=then_output_name,
            else_name=else_output_name,
            output_name=output_name,
            output_shape=expected_output_shape,
            output_dtype=expected_output_dtype,
            node_name=f"{node_name}_generic_if_where_{output_index}",
            ctx=ctx,
        )


def _build_if_nested_reducemin_add_branch_mux(node: Any, ctx: Any) -> None:
    cond_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(cond_name)
    ctx.ensure_tensor(output_name)
    expected_output_dtype = str(ctx.get_tensor_dtype(output_name)).upper()
    expected_output_shape = [int(v) for v in ctx.get_tensor_shape(output_name)]

    then_graph = node.attrs["then_branch"]
    else_graph = node.attrs["else_branch"]
    _ensure_graph_initializers(then_graph, ctx)
    _ensure_graph_initializers(else_graph, ctx)

    then_output_name = ctx.add_intermediate_tensor(
        f"{output_name}_if_then",
        dtype=expected_output_dtype,
        shape=expected_output_shape,
    )
    _lower_graph_nodes(
        graph=then_graph,
        ctx=ctx,
        output_name_remap={
            str(then_graph.output[0].name): then_output_name,
        },
    )
    then_output_name = _cast_to_dtype_for_if(
        tensor_name=then_output_name,
        target_dtype=expected_output_dtype,
        output_name=output_name,
        suffix="nested_then_top",
        ctx=ctx,
    )

    else_prelude_graph = type("IfElsePreludeGraph", (), {})()
    else_prelude_graph.node = list(else_graph.node[:3])
    _lower_graph_nodes(
        graph=else_prelude_graph,
        ctx=ctx,
    )

    nested_if_node = else_graph.node[3]
    nested_attrs = {a.name: a for a in nested_if_node.attribute}
    nested_then_graph = nested_attrs["then_branch"].g
    nested_else_graph = nested_attrs["else_branch"].g
    _ensure_graph_initializers(nested_then_graph, ctx)
    _ensure_graph_initializers(nested_else_graph, ctx)

    nested_then_output_name = ctx.add_intermediate_tensor(
        f"{output_name}_if_nested_then",
        dtype=expected_output_dtype,
        shape=expected_output_shape,
    )
    nested_else_output_name = ctx.add_intermediate_tensor(
        f"{output_name}_if_nested_else",
        dtype=expected_output_dtype,
        shape=expected_output_shape,
    )
    _lower_graph_nodes(
        graph=nested_then_graph,
        ctx=ctx,
        output_name_remap={
            str(nested_then_graph.output[0].name): nested_then_output_name,
        },
    )
    _lower_graph_nodes(
        graph=nested_else_graph,
        ctx=ctx,
        output_name_remap={
            str(nested_else_graph.output[0].name): nested_else_output_name,
        },
    )

    nested_then_output_name = _cast_to_dtype_for_if(
        tensor_name=nested_then_output_name,
        target_dtype=expected_output_dtype,
        output_name=output_name,
        suffix="nested_then",
        ctx=ctx,
    )
    nested_else_output_name = _cast_to_dtype_for_if(
        tensor_name=nested_else_output_name,
        target_dtype=expected_output_dtype,
        output_name=output_name,
        suffix="nested_else",
        ctx=ctx,
    )

    else_output_name = ctx.add_intermediate_tensor(
        f"{output_name}_if_else",
        dtype=expected_output_dtype,
        shape=expected_output_shape,
    )
    _emit_where_mux(
        cond_name=str(nested_if_node.input[0]),
        then_name=nested_then_output_name,
        else_name=nested_else_output_name,
        output_name=else_output_name,
        output_shape=expected_output_shape,
        output_dtype=expected_output_dtype,
        node_name=f"{node.name}_nested_if_where",
        ctx=ctx,
    )

    _emit_where_mux(
        cond_name=cond_name,
        then_name=then_output_name,
        else_name=else_output_name,
        output_name=output_name,
        output_shape=expected_output_shape,
        output_dtype=expected_output_dtype,
        node_name=f"{node.name}_if_where",
        ctx=ctx,
    )


def build_if_op(node: Any, ctx: Any) -> None:
    if is_supported_if_axis0_add_branch_pattern(node):
        _build_if_axis0_add_branch_mux(node, ctx)
        return

    if is_supported_if_sequenceconstruct_add_branch_pattern(node):
        _build_if_sequenceconstruct_add_branch_mux(node, ctx)
        return

    if is_supported_if_nested_reducemin_add_branch_pattern(node):
        _build_if_nested_reducemin_add_branch_mux(node, ctx)
        return

    if is_supported_if_nms_guard_pattern(node):
        cond_name = node.inputs[0].name
        output_name = node.outputs[0].name

        then_graph = node.attrs["then_branch"]
        else_graph = node.attrs["else_branch"]
        _ensure_graph_initializers(then_graph, ctx)
        _ensure_graph_initializers(else_graph, ctx)

        else_ops = [str(n.op_type) for n in else_graph.node]
        if else_ops == _IF_NMS_GUARD_ELSE_OPS_WITH_NESTED_IF:
            unsqueeze_scores_idx = 3
            nms_idx = 9
        elif else_ops == _IF_NMS_GUARD_ELSE_OPS_SIMPLE:
            unsqueeze_scores_idx = 2
            nms_idx = 8
        else:
            raise NotImplementedError(
                f"If pattern details are not supported by flatbuffer_direct built-in lowering. node={node.name}"
            )

        reduce_max_node = else_graph.node[0]
        cast_node = else_graph.node[1]
        unsqueeze_scores_node = else_graph.node[unsqueeze_scores_idx]
        nms_node = else_graph.node[nms_idx]
        if (
            str(reduce_max_node.op_type) != "ReduceMax"
            or str(cast_node.op_type) != "Cast"
            or str(unsqueeze_scores_node.op_type) != "Unsqueeze"
            or str(nms_node.op_type) != "NonMaxSuppression"
        ):
            raise NotImplementedError(
                f"If pattern details are not supported by flatbuffer_direct built-in lowering. node={node.name}"
            )

        boxes_name = str(reduce_max_node.input[0])
        idxs_name = str(cast_node.input[0])
        scores_name = str(unsqueeze_scores_node.input[0])
        ctx.ensure_tensor(boxes_name)
        ctx.ensure_tensor(scores_name)
        ctx.ensure_tensor(idxs_name)

        boxes_np_dtype = _np_dtype_from_tflite_dtype(ctx.get_tensor_dtype(boxes_name))
        scores_np_dtype = _np_dtype_from_tflite_dtype(ctx.get_tensor_dtype(scores_name))
        idxs_np_dtype = _np_dtype_from_tflite_dtype(ctx.get_tensor_dtype(idxs_name))

        dummy_box_value = float(np.finfo(boxes_np_dtype).min) if np.issubdtype(boxes_np_dtype, np.floating) else 0.0
        dummy_score_value = float(np.finfo(scores_np_dtype).min) if np.issubdtype(scores_np_dtype, np.floating) else 0.0
        if len(nms_node.input) >= 5:
            score_threshold_name = str(nms_node.input[4])
            score_threshold = ctx.get_constant_array(score_threshold_name)
            if score_threshold is not None:
                score_threshold_f = float(np.asarray(score_threshold, dtype=np.float32).reshape(-1)[0])
                dummy_score_value = min(dummy_score_value, float(score_threshold_f - 1.0))

        dummy_box_name = ctx.add_const_tensor(
            f"{output_name}_if_dummy_box",
            np.full((1, 4), dummy_box_value, dtype=boxes_np_dtype),
        )
        dummy_score_name = ctx.add_const_tensor(
            f"{output_name}_if_dummy_score",
            np.asarray([dummy_score_value], dtype=scores_np_dtype),
        )
        dummy_idx_name = ctx.add_const_tensor(
            f"{output_name}_if_dummy_idx",
            np.asarray([0], dtype=idxs_np_dtype),
        )

        boxes_safe_name = ctx.add_intermediate_tensor(
            f"{output_name}_if_boxes_safe",
            dtype=str(ctx.get_tensor_dtype(boxes_name)).upper(),
            shape=[-1, 4],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CONCATENATION",
                inputs=[boxes_name, dummy_box_name],
                outputs=[boxes_safe_name],
                options={"axis": 0, "fusedActivationFunction": "NONE"},
            )
        )

        scores_safe_name = ctx.add_intermediate_tensor(
            f"{output_name}_if_scores_safe",
            dtype=str(ctx.get_tensor_dtype(scores_name)).upper(),
            shape=[-1],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CONCATENATION",
                inputs=[scores_name, dummy_score_name],
                outputs=[scores_safe_name],
                options={"axis": 0, "fusedActivationFunction": "NONE"},
            )
        )

        idxs_safe_name = ctx.add_intermediate_tensor(
            f"{output_name}_if_idxs_safe",
            dtype=str(ctx.get_tensor_dtype(idxs_name)).upper(),
            shape=[-1],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CONCATENATION",
                inputs=[idxs_name, dummy_idx_name],
                outputs=[idxs_safe_name],
                options={"axis": 0, "fusedActivationFunction": "NONE"},
            )
        )

        remap_inputs = {
            boxes_name: boxes_safe_name,
            scores_name: scores_safe_name,
            idxs_name: idxs_safe_name,
        }
        _lower_graph_nodes(
            graph=else_graph,
            ctx=ctx,
            input_name_remap=remap_inputs,
            output_name_remap=None,
        )
        return

    if is_supported_if_generic_branch_mux_pattern(node, ctx):
        _build_if_generic_branch_mux(node, ctx)
        return

    raise NotImplementedError(
        (
            "If pattern is not supported by flatbuffer_direct built-in lowering. "
            "supported patterns: NMS-guard pattern, axis0 Add-branch pattern, "
            "SequenceConstruct Add-branch pattern, nested ReduceMin/Add pattern, "
            "or generic branch-mux pattern (no branch inputs, safe branch-op subset, "
            "with matching "
            "then/else outputs). "
            f"node={node.name}"
        )
    )

def _loop_const_trip_count(node: Any, ctx: Any) -> Optional[int]:
    if len(node.inputs) < 1:
        return None
    trip_count_name = node.inputs[0].name
    if str(trip_count_name) == "":
        return None
    trip_count_arr = ctx.get_constant_array(trip_count_name)
    if trip_count_arr is None:
        return None
    flat = np.asarray(trip_count_arr).reshape(-1)
    if int(flat.size) == 0:
        return None
    return int(flat[0])


def _loop_const_cond(node: Any, ctx: Any) -> Optional[bool]:
    if len(node.inputs) < 2:
        return None
    cond_name = node.inputs[1].name
    if str(cond_name) == "":
        return None
    cond_arr = ctx.get_constant_array(cond_name)
    if cond_arr is None:
        return None
    flat = np.asarray(cond_arr).reshape(-1)
    if int(flat.size) == 0:
        return None
    return bool(flat[0])


def is_supported_loop_static_unroll_pattern(node: Any, ctx: Any) -> bool:
    body = node.attrs.get("body", None)
    if body is None or not hasattr(body, "input") or not hasattr(body, "output"):
        return False
    if len(node.inputs) < 3:
        return False
    trip_count = _loop_const_trip_count(node, ctx)
    if trip_count is None or int(trip_count) < 0:
        return False
    if int(trip_count) > 1024:
        return False
    if _loop_const_cond(node, ctx) is None:
        return False

    state_count = int(len(node.inputs) - 2)
    if state_count <= 0:
        return False
    # Static unroll path supports loop-carried outputs only (no scan outputs).
    if len(node.outputs) != state_count:
        return False
    if len(body.input) != int(2 + state_count):
        return False
    if len(body.output) != int(1 + state_count):
        return False
    return True


def is_supported_loop_while_pattern(node: Any, ctx: Any) -> bool:
    body = node.attrs.get("body", None)
    if body is None or not hasattr(body, "input") or not hasattr(body, "output"):
        return False
    if len(node.inputs) < 3:
        return False
    state_count = int(len(node.inputs) - 2)
    if state_count <= 0:
        return False
    # WHILE lowering currently supports loop-carried outputs only (no scan outputs).
    if len(node.outputs) != state_count:
        return False
    if len(body.input) != int(2 + state_count):
        return False
    if len(body.output) != int(1 + state_count):
        return False
    return True


def _make_unique_tensor_name(*, base_name: str, ctx: Any) -> str:
    name = str(base_name) if str(base_name) != "" else "loop_tensor"
    if name not in ctx.model_ir.tensors and name not in ctx.shape_map:
        return name
    suffix = 1
    while True:
        candidate = f"{name}_{suffix}"
        if candidate not in ctx.model_ir.tensors and candidate not in ctx.shape_map:
            return candidate
        suffix += 1


def _register_tensor_remap_metadata(*, old_name: str, new_name: str, ctx: Any) -> None:
    if str(old_name) in ctx.shape_map and str(new_name) not in ctx.shape_map:
        ctx.shape_map[str(new_name)] = [int(v) for v in list(ctx.shape_map[str(old_name)])]
    if str(old_name) in ctx.dtype_map and str(new_name) not in ctx.dtype_map:
        ctx.dtype_map[str(new_name)] = str(ctx.dtype_map[str(old_name)])


def _copy_tensor_metadata_for_alias(*, src_name: str, dst_name: str, ctx: Any) -> None:
    src_tensor = ctx.model_ir.tensors.get(str(src_name), None)
    dst_tensor = ctx.model_ir.tensors.get(str(dst_name), None)
    if src_tensor is None or dst_tensor is None:
        return
    dst_tensor.dtype = str(src_tensor.dtype)
    dst_tensor.shape = [int(v) for v in list(src_tensor.shape)]
    if src_tensor.shape_signature is not None:
        dst_tensor.shape_signature = [int(v) for v in list(src_tensor.shape_signature)]
    else:
        dst_tensor.shape_signature = [int(v) for v in list(src_tensor.shape)]
    dst_tensor.quantization = src_tensor.quantization


def _emit_tensor_alias_via_reshape(
    *,
    src_name: str,
    dst_name: str,
    alias_base_name: str,
    ctx: Any,
) -> None:
    if str(src_name) == str(dst_name):
        return
    ctx.ensure_tensor(str(src_name))
    src_shape = [int(v) for v in ctx.get_tensor_shape(str(src_name))]
    src_dtype = str(ctx.get_tensor_dtype(str(src_name))).upper()
    ctx.ensure_tensor(str(dst_name), dtype=src_dtype, shape=src_shape)
    _copy_tensor_metadata_for_alias(
        src_name=str(src_name),
        dst_name=str(dst_name),
        ctx=ctx,
    )
    alias_shape_name = ctx.add_const_tensor(
        _make_unique_tensor_name(base_name=f"{alias_base_name}_shape", ctx=ctx),
        np.asarray(src_shape, dtype=np.int32),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="RESHAPE",
            inputs=[str(src_name), str(alias_shape_name)],
            outputs=[str(dst_name)],
            options={"newShape": [int(v) for v in src_shape]},
        )
        )


def _build_loop_static_unroll(node: Any, ctx: Any) -> None:
    body_graph = node.attrs["body"]
    _ensure_graph_initializers(body_graph, ctx)

    trip_count = int(_loop_const_trip_count(node, ctx))
    cond_initial = bool(_loop_const_cond(node, ctx))
    state_input_names = [str(v.name) for v in node.inputs[2:]]
    state_output_names = [str(v.name) for v in node.outputs]
    state_count = int(len(state_input_names))

    for state_input_name in state_input_names:
        ctx.ensure_tensor(state_input_name)

    current_states = [str(v) for v in state_input_names]
    if cond_initial and trip_count > 0:
        body_input_iter_name = str(body_graph.input[0].name)
        body_input_cond_name = str(body_graph.input[1].name)
        body_state_input_names = [str(v.name) for v in body_graph.input[2:2 + state_count]]
        body_cond_output_name = str(body_graph.output[0].name)
        body_state_output_names = [str(v.name) for v in body_graph.output[1:1 + state_count]]

        for iter_idx in range(int(trip_count)):
            iter_const_name = ctx.add_const_tensor(
                _make_unique_tensor_name(base_name=f"{node.name}_loop_iter_{iter_idx}", ctx=ctx),
                np.asarray(iter_idx, dtype=np.int64),
            )
            cond_const_name = ctx.add_const_tensor(
                _make_unique_tensor_name(base_name=f"{node.name}_loop_cond_{iter_idx}", ctx=ctx),
                np.asarray(True, dtype=np.bool_),
            )

            iteration_internal_remap: Dict[str, str] = {}
            for body_node in body_graph.node:
                for output_name in body_node.output:
                    out_name = str(output_name)
                    if out_name == "":
                        continue
                    if out_name in iteration_internal_remap:
                        continue
                    scoped_name = _make_unique_tensor_name(
                        base_name=f"{out_name}_loop_{iter_idx}",
                        ctx=ctx,
                    )
                    iteration_internal_remap[out_name] = scoped_name
                    _register_tensor_remap_metadata(
                        old_name=out_name,
                        new_name=scoped_name,
                        ctx=ctx,
                    )

            input_remap = dict(iteration_internal_remap)
            input_remap[body_input_iter_name] = str(iter_const_name)
            input_remap[body_input_cond_name] = str(cond_const_name)
            for state_idx, body_state_input_name in enumerate(body_state_input_names):
                input_remap[str(body_state_input_name)] = str(current_states[state_idx])

            output_remap = dict(iteration_internal_remap)
            cond_output_scoped = _make_unique_tensor_name(
                base_name=f"{node.name}_loop_cond_out_{iter_idx}",
                ctx=ctx,
            )
            output_remap[str(body_cond_output_name)] = str(cond_output_scoped)
            _register_tensor_remap_metadata(
                old_name=str(body_cond_output_name),
                new_name=str(cond_output_scoped),
                ctx=ctx,
            )

            next_states: List[str] = []
            for state_idx, body_state_output_name in enumerate(body_state_output_names):
                next_state_name = _make_unique_tensor_name(
                    base_name=f"{node.name}_loop_state_{state_idx}_iter_{iter_idx}",
                    ctx=ctx,
                )
                output_remap[str(body_state_output_name)] = str(next_state_name)
                _register_tensor_remap_metadata(
                    old_name=str(body_state_output_name),
                    new_name=str(next_state_name),
                    ctx=ctx,
                )
                next_states.append(str(next_state_name))

            _lower_graph_nodes(
                graph=body_graph,
                ctx=ctx,
                input_name_remap=input_remap,
                output_name_remap=output_remap,
            )

            for state_idx, next_state_name in enumerate(next_states):
                if next_state_name in ctx.model_ir.tensors:
                    continue
                body_state_output_name = str(body_state_output_names[state_idx])
                source_name = input_remap.get(body_state_output_name, None)
                if source_name is None:
                    source_name = str(current_states[state_idx])
                if str(source_name) not in ctx.model_ir.tensors:
                    raise NotImplementedError(
                        (
                            "Loop static unroll could not resolve loop-carried state output. "
                            f"node={node.name} iter={iter_idx} output={body_state_output_name}"
                        )
                    )
                _emit_tensor_alias_via_reshape(
                    src_name=str(source_name),
                    dst_name=str(next_state_name),
                    alias_base_name=f"{node.name}_loop_state_alias_{state_idx}_iter_{iter_idx}",
                    ctx=ctx,
                )

            current_states = [str(v) for v in next_states]

    for state_idx, output_name in enumerate(state_output_names):
        src_name = str(current_states[state_idx])
        _emit_tensor_alias_via_reshape(
            src_name=src_name,
            dst_name=str(output_name),
            alias_base_name=f"{node.name}_loop_final_state_{state_idx}",
            ctx=ctx,
        )


def _build_loop_while(node: Any, ctx: Any) -> None:
    body_graph = node.attrs["body"]
    _ensure_graph_initializers(body_graph, ctx)

    state_input_names = [str(v.name) for v in node.inputs[2:]]
    state_output_names = [str(v.name) for v in node.outputs]
    state_count = int(len(state_input_names))

    max_trip_input_name = str(node.inputs[0].name)
    cond_input_name = str(node.inputs[1].name)
    if max_trip_input_name == "" or cond_input_name == "":
        raise NotImplementedError(
            (
                "Loop WHILE lowering currently requires explicit max_trip_count and condition inputs. "
                f"node={node.name}"
            )
        )

    ctx.ensure_tensor(max_trip_input_name)
    ctx.ensure_tensor(cond_input_name)
    for state_input_name in state_input_names:
        ctx.ensure_tensor(state_input_name)

    iter_dtype = str(ctx.get_tensor_dtype(max_trip_input_name)).upper()
    if iter_dtype not in {"INT32", "INT64"}:
        raise NotImplementedError(
            f"Loop WHILE lowering requires INT32/INT64 max_trip_count. node={node.name} dtype={iter_dtype}"
        )

    cond_dtype = str(ctx.get_tensor_dtype(cond_input_name)).upper()
    while_cond_input_name = cond_input_name
    if cond_dtype != "BOOL":
        cast_cond_name = ctx.add_intermediate_tensor(
            _make_unique_tensor_name(base_name=f"{node.name}_loop_cond_cast", ctx=ctx),
            dtype="BOOL",
            shape=[],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[cond_input_name],
                outputs=[cast_cond_name],
                options={"inDataType": cond_dtype, "outDataType": "BOOL"},
            )
        )
        while_cond_input_name = cast_cond_name

    iter_init_name = ctx.add_const_tensor(
        _make_unique_tensor_name(base_name=f"{node.name}_loop_iter_init", ctx=ctx),
        np.asarray(0, dtype=np.int64 if iter_dtype == "INT64" else np.int32),
    )

    while_input_names = [iter_init_name, max_trip_input_name, while_cond_input_name] + state_input_names
    while_output_names: List[str] = [
        _make_unique_tensor_name(base_name=f"{node.name}_loop_iter_out", ctx=ctx),
        _make_unique_tensor_name(base_name=f"{node.name}_loop_trip_out", ctx=ctx),
        _make_unique_tensor_name(base_name=f"{node.name}_loop_cond_out", ctx=ctx),
    ] + state_output_names

    ctx.ensure_tensor(while_output_names[0], dtype=iter_dtype, shape=[])
    ctx.ensure_tensor(while_output_names[1], dtype=iter_dtype, shape=[])
    ctx.ensure_tensor(while_output_names[2], dtype="BOOL", shape=[])
    for idx, state_output_name in enumerate(state_output_names):
        state_shape = [int(v) for v in ctx.get_tensor_shape(state_input_names[idx])]
        state_dtype = str(ctx.get_tensor_dtype(state_input_names[idx])).upper()
        ctx.ensure_tensor(state_output_name, dtype=state_dtype, shape=state_shape)

    cond_subgraph = ModelIR(name=f"{node.name}_while_cond")
    body_subgraph = ModelIR(name=f"{node.name}_while_body")

    # Build subgraph-local contexts with copied shape/dtype knowledge.
    from onnx2tf.tflite_builder.lower_from_onnx2tf import LoweringContext

    cond_ctx = LoweringContext(
        model_ir=cond_subgraph,
        shape_map=dict(ctx.shape_map),
        dtype_map=dict(ctx.dtype_map),
        constants={},
        onnx_model=getattr(ctx, "onnx_model", None),
        allow_custom_ops=bool(getattr(ctx, "allow_custom_ops", False)),
        custom_op_allowlist=list(getattr(ctx, "custom_op_allowlist", []) or []),
        disable_group_convolution=bool(getattr(ctx, "disable_group_convolution", False)),
        tensor_consumer_count={},
        graph_output_names=None,
        output_nms_with_argmax=bool(getattr(ctx, "output_nms_with_argmax", False)),
        switch_nms_version=str(getattr(ctx, "switch_nms_version", "v4")),
    )
    body_ctx = LoweringContext(
        model_ir=body_subgraph,
        shape_map=dict(ctx.shape_map),
        dtype_map=dict(ctx.dtype_map),
        constants={},
        onnx_model=getattr(ctx, "onnx_model", None),
        allow_custom_ops=bool(getattr(ctx, "allow_custom_ops", False)),
        custom_op_allowlist=list(getattr(ctx, "custom_op_allowlist", []) or []),
        disable_group_convolution=bool(getattr(ctx, "disable_group_convolution", False)),
        tensor_consumer_count={},
        graph_output_names=None,
        output_nms_with_argmax=bool(getattr(ctx, "output_nms_with_argmax", False)),
        switch_nms_version=str(getattr(ctx, "switch_nms_version", "v4")),
    )
    _ensure_graph_initializers(body_graph, body_ctx)

    cond_iter_in = f"{node.name}_while_iter_in"
    cond_trip_in = f"{node.name}_while_trip_in"
    cond_cond_in = f"{node.name}_while_cond_in"
    cond_state_in_names = [f"{node.name}_while_state_{idx}_in" for idx in range(state_count)]
    cond_out_name = f"{node.name}_while_cond_out"

    for tensor_name, dtype_name, shape in [
        (cond_iter_in, iter_dtype, []),
        (cond_trip_in, iter_dtype, []),
        (cond_cond_in, "BOOL", []),
    ]:
        cond_ctx.ensure_tensor(tensor_name, dtype=dtype_name, shape=shape)
    for idx, state_name in enumerate(cond_state_in_names):
        cond_ctx.ensure_tensor(
            state_name,
            dtype=str(ctx.get_tensor_dtype(state_input_names[idx])).upper(),
            shape=[int(v) for v in ctx.get_tensor_shape(state_input_names[idx])],
        )
    cond_ctx.ensure_tensor(cond_out_name, dtype="BOOL", shape=[])

    cond_iter_lt_name = cond_ctx.add_intermediate_tensor(
        f"{node.name}_while_iter_lt_trip",
        dtype="BOOL",
        shape=[],
    )
    cond_ctx.add_operator(
        OperatorIR(
            op_type="LESS",
            inputs=[cond_iter_in, cond_trip_in],
            outputs=[cond_iter_lt_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    cond_ctx.add_operator(
        OperatorIR(
            op_type="LOGICAL_AND",
            inputs=[cond_cond_in, cond_iter_lt_name],
            outputs=[cond_out_name],
            options={},
        )
    )
    cond_subgraph.inputs = [cond_iter_in, cond_trip_in, cond_cond_in] + cond_state_in_names
    cond_subgraph.outputs = [cond_out_name]

    body_iter_in = f"{node.name}_while_iter_in"
    body_trip_in = f"{node.name}_while_trip_in"
    body_cond_in = f"{node.name}_while_cond_in"
    body_state_in_names = [f"{node.name}_while_state_{idx}_in" for idx in range(state_count)]
    body_iter_out = f"{node.name}_while_iter_out"
    body_trip_out = f"{node.name}_while_trip_out"
    body_cond_out = f"{node.name}_while_cond_out"
    body_state_out_names = [f"{node.name}_while_state_{idx}_out" for idx in range(state_count)]

    for tensor_name, dtype_name, shape in [
        (body_iter_in, iter_dtype, []),
        (body_trip_in, iter_dtype, []),
        (body_cond_in, "BOOL", []),
    ]:
        body_ctx.ensure_tensor(tensor_name, dtype=dtype_name, shape=shape)
    for idx, state_name in enumerate(body_state_in_names):
        body_ctx.ensure_tensor(
            state_name,
            dtype=str(ctx.get_tensor_dtype(state_input_names[idx])).upper(),
            shape=[int(v) for v in ctx.get_tensor_shape(state_input_names[idx])],
        )

    iter_plus_one_const_name = body_ctx.add_const_tensor(
        f"{node.name}_while_iter_plus_one_const",
        np.asarray(1, dtype=np.int64 if iter_dtype == "INT64" else np.int32),
    )
    body_ctx.ensure_tensor(body_iter_out, dtype=iter_dtype, shape=[])
    body_ctx.add_operator(
        OperatorIR(
            op_type="ADD",
            inputs=[body_iter_in, iter_plus_one_const_name],
            outputs=[body_iter_out],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    _emit_tensor_alias_via_reshape(
        src_name=body_trip_in,
        dst_name=body_trip_out,
        alias_base_name=f"{node.name}_while_trip_passthrough",
        ctx=body_ctx,
    )

    onnx_body_input_iter_name = str(body_graph.input[0].name)
    onnx_body_input_cond_name = str(body_graph.input[1].name)
    onnx_body_state_input_names = [str(v.name) for v in body_graph.input[2:2 + state_count]]
    onnx_body_cond_output_name = str(body_graph.output[0].name)
    onnx_body_state_output_names = [str(v.name) for v in body_graph.output[1:1 + state_count]]

    body_input_remap: Dict[str, str] = {
        onnx_body_input_iter_name: body_iter_in,
        onnx_body_input_cond_name: body_cond_in,
    }
    for idx, state_input_name in enumerate(onnx_body_state_input_names):
        body_input_remap[state_input_name] = body_state_in_names[idx]

    body_output_remap: Dict[str, str] = {}
    for graph_node in body_graph.node:
        for output_name in graph_node.output:
            out_name = str(output_name)
            if out_name == "":
                continue
            if out_name == onnx_body_cond_output_name:
                mapped_name = str(body_cond_out)
            elif out_name in onnx_body_state_output_names:
                mapped_name = str(body_state_out_names[onnx_body_state_output_names.index(out_name)])
            else:
                mapped_name = _make_unique_tensor_name(
                    base_name=f"{node.name}_while_body_{out_name}",
                    ctx=body_ctx,
                )
                _register_tensor_remap_metadata(
                    old_name=out_name,
                    new_name=mapped_name,
                    ctx=body_ctx,
                )
            body_output_remap[out_name] = mapped_name
            body_input_remap[out_name] = mapped_name

    for idx, state_output_name in enumerate(onnx_body_state_output_names):
        body_output_remap[state_output_name] = body_state_out_names[idx]

    body_ctx.ensure_tensor(body_cond_out, dtype="BOOL", shape=[])
    for idx, state_out_name in enumerate(body_state_out_names):
        body_ctx.ensure_tensor(
            state_out_name,
            dtype=str(ctx.get_tensor_dtype(state_input_names[idx])).upper(),
            shape=[int(v) for v in ctx.get_tensor_shape(state_input_names[idx])],
        )

    _lower_graph_nodes(
        graph=body_graph,
        ctx=body_ctx,
        input_name_remap=body_input_remap,
        output_name_remap=body_output_remap,
    )

    produced_tensor_names = {
        str(out_name)
        for op in body_ctx.model_ir.operators
        for out_name in op.outputs
    }
    produced_tensor_names.update(str(name) for name in body_ctx.constants.keys())

    if body_cond_out not in produced_tensor_names:
        fallback_cond_src = body_input_remap.get(onnx_body_cond_output_name, body_cond_in)
        _emit_tensor_alias_via_reshape(
            src_name=str(fallback_cond_src),
            dst_name=body_cond_out,
            alias_base_name=f"{node.name}_while_cond_passthrough",
            ctx=body_ctx,
        )
        produced_tensor_names.add(str(body_cond_out))
    for idx, state_out_name in enumerate(body_state_out_names):
        if str(state_out_name) in produced_tensor_names:
            continue
        fallback_state_src = body_input_remap.get(onnx_body_state_output_names[idx], body_state_in_names[idx])
        _emit_tensor_alias_via_reshape(
            src_name=str(fallback_state_src),
            dst_name=state_out_name,
            alias_base_name=f"{node.name}_while_state_passthrough_{idx}",
            ctx=body_ctx,
        )
        produced_tensor_names.add(str(state_out_name))

    body_subgraph.inputs = [body_iter_in, body_trip_in, body_cond_in] + body_state_in_names
    body_subgraph.outputs = [body_iter_out, body_trip_out, body_cond_out] + body_state_out_names

    cond_subgraph_index = int(len(ctx.model_ir.subgraphs) + 1)
    ctx.model_ir.subgraphs.append(cond_subgraph)
    body_subgraph_index = int(len(ctx.model_ir.subgraphs) + 1)
    ctx.model_ir.subgraphs.append(body_subgraph)

    ctx.add_operator(
        OperatorIR(
            op_type="WHILE",
            inputs=while_input_names,
            outputs=while_output_names,
            options={
                "condSubgraphIndex": int(cond_subgraph_index),
                "bodySubgraphIndex": int(body_subgraph_index),
            },
        )
    )


def build_loop_op(node: Any, ctx: Any) -> None:
    if is_supported_loop_while_pattern(node, ctx):
        _build_loop_while(node, ctx)
        return
    if is_supported_loop_static_unroll_pattern(node, ctx):
        _build_loop_static_unroll(node, ctx)
        return
    raise NotImplementedError(
        (
            "Loop built-in lowering supports either static-unroll patterns with constant trip_count/cond "
            "or WHILE patterns with loop-carried outputs only (no scan outputs). "
            f"node={node.name}"
        )
    )
