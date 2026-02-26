from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import onnx
from onnx import numpy_helper

from onnx2tf.tflite_builder.ir import OperatorIR


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
    expected_else_ops = [
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
    else_ops = [str(n.op_type) for n in else_graph.node]
    if else_ops != expected_else_ops:
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
        type("In", (), {"name": remap_in.get(str(name), str(name))})
        for name in node_proto.input
        if str(name) != ""
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

    for graph_node in graph.node:
        op_type = str(graph_node.op_type)

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
            if len(graph_node.input) != 1 or len(graph_node.output) != 1:
                raise NotImplementedError(
                    f"If in branch requires 1 input/1 output in flatbuffer_direct. node={graph_node.name}"
                )
            cond_name = remap_in.get(str(graph_node.input[0]), str(graph_node.input[0]))
            cond_value = ctx.get_constant_array(cond_name)
            if cond_value is None:
                raise NotImplementedError(
                    f"If in branch requires constant condition in flatbuffer_direct. node={graph_node.name}"
                )
            cond_bool = bool(np.asarray(cond_value).reshape(-1)[0])
            attrs = {a.name: a for a in graph_node.attribute}
            selected_attr = "then_branch" if cond_bool else "else_branch"
            if selected_attr not in attrs:
                raise NotImplementedError(
                    f"If branch '{selected_attr}' is missing in node={graph_node.name}"
                )
            selected_graph = attrs[selected_attr].g
            _ensure_graph_initializers(selected_graph, ctx)
            selected_output_name = (
                str(selected_graph.output[0].name)
                if len(selected_graph.output) > 0
                else ""
            )
            nested_out_name = str(graph_node.output[0])
            nested_output_remap = dict(remap_out)
            if selected_output_name != "" and nested_out_name != "":
                nested_output_remap[selected_output_name] = remap_out.get(
                    nested_out_name,
                    nested_out_name,
                )

            # The nested If in torchvision NMS heads emits Squeeze(axis=1) for
            # [K,1] selected indices. Branch tensors are out-of-scope for the
            # parent shape map, so validation may see rank=1 placeholders and
            # reject axis=1. Lower this branch explicitly as RESHAPE([-1]).
            if (
                len(selected_graph.node) == 1
                and str(selected_graph.node[0].op_type) == "Squeeze"
                and len(selected_graph.node[0].input) >= 1
                and len(selected_graph.node[0].output) == 1
            ):
                squeeze_input_name = remap_in.get(
                    str(selected_graph.node[0].input[0]),
                    str(selected_graph.node[0].input[0]),
                )
                squeeze_output_name = nested_output_remap.get(
                    str(selected_graph.node[0].output[0]),
                    str(selected_graph.node[0].output[0]),
                )
                ctx.ensure_tensor(squeeze_input_name)
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
                    f"{squeeze_output_name}_if_nested_squeeze_shape",
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

            _lower_graph_nodes(
                graph=selected_graph,
                ctx=ctx,
                input_name_remap=remap_in,
                output_name_remap=nested_output_remap,
            )
            continue

        wrapped = _wrap_node(
            graph_node,
            input_name_remap=remap_in,
            output_name_remap=remap_out,
        )
        if op_type == "NonMaxSuppression":
            for wrapped_output in wrapped.outputs:
                out_name = str(wrapped_output.name)
                ctx.ensure_tensor(out_name, dtype="INT64")
                ctx.model_ir.tensors[out_name].dtype = "INT64"
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

    if not is_supported_if_nms_guard_pattern(node):
        raise NotImplementedError(
            (
                "If pattern is not supported by flatbuffer_direct built-in lowering. "
                "supported patterns: NMS-guard pattern, axis0 Add-branch pattern, "
                "SequenceConstruct Add-branch pattern, nested ReduceMin/Add pattern. "
                f"node={node.name}"
            )
        )

    cond_name = node.inputs[0].name
    output_name = node.outputs[0].name

    then_graph = node.attrs["then_branch"]
    else_graph = node.attrs["else_branch"]
    _ensure_graph_initializers(then_graph, ctx)
    _ensure_graph_initializers(else_graph, ctx)

    reduce_max_node = else_graph.node[0]
    cast_node = else_graph.node[1]
    unsqueeze_scores_node = else_graph.node[3]
    nms_node = else_graph.node[9]
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

    candidate_output_name = str(else_graph.output[0].name)
    ctx.ensure_tensor(candidate_output_name)
    expected_output_dtype = str(ctx.get_tensor_dtype(output_name)).upper()
    candidate_output_dtype = str(ctx.get_tensor_dtype(candidate_output_name)).upper()
    if candidate_output_dtype != expected_output_dtype:
        casted_candidate_name = ctx.add_intermediate_tensor(
            f"{output_name}_if_candidate_cast",
            dtype=expected_output_dtype,
            shape=[int(v) for v in ctx.get_tensor_shape(candidate_output_name)],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[candidate_output_name],
                outputs=[casted_candidate_name],
                options={
                    "inDataType": candidate_output_dtype,
                    "outDataType": expected_output_dtype,
                },
            )
        )
        candidate_output_name = casted_candidate_name

    _add_cond_gate_to_slice_output(
        cond_name=cond_name,
        candidate_name=candidate_output_name,
        output_name=output_name,
        ctx=ctx,
    )
