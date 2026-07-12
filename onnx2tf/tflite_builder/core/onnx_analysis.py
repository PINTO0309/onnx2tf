from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx
from onnx import numpy_helper

from onnx2tf.utils.onnx_litert_runtime import check_model_has_external_data
from onnx2tf.tflite_builder.ir import normalize_onnx_shape
from onnx2tf.tflite_builder.tensor_buffer_builder import tflite_dtype_from_numpy


_ONNX_TYPE_TO_TFLITE_DTYPE = {
    onnx.TensorProto.FLOAT: "FLOAT32",
    onnx.TensorProto.FLOAT16: "FLOAT16",
    onnx.TensorProto.DOUBLE: "FLOAT64",
    onnx.TensorProto.INT8: "INT8",
    onnx.TensorProto.INT16: "INT16",
    onnx.TensorProto.INT32: "INT32",
    onnx.TensorProto.INT64: "INT64",
    onnx.TensorProto.UINT8: "UINT8",
    onnx.TensorProto.UINT16: "UINT16",
    onnx.TensorProto.UINT32: "UINT32",
    onnx.TensorProto.UINT64: "UINT64",
    onnx.TensorProto.BOOL: "BOOL",
    onnx.TensorProto.STRING: "STRING",
}

def _dtype_from_onnx_elem_type(elem_type: Optional[int]) -> str:
    if elem_type is None:
        return "FLOAT32"
    if elem_type not in _ONNX_TYPE_TO_TFLITE_DTYPE:
        raise NotImplementedError(f"Unsupported ONNX dtype in flatbuffer_direct: elem_type={elem_type}")
    return _ONNX_TYPE_TO_TFLITE_DTYPE[elem_type]


def _collect_constant_arrays(onnx_graph: onnx.ModelProto) -> Dict[str, np.ndarray]:
    constants: Dict[str, np.ndarray] = {}
    for ini in onnx_graph.graph.initializer:
        constants[ini.name] = np.asarray(numpy_helper.to_array(ini))
    for node in onnx_graph.graph.node:
        if str(node.op_type) != "Constant" or len(node.output) < 1:
            continue
        output_name = str(node.output[0])
        if output_name == "":
            continue
        value_attr = None
        for attr in node.attribute:
            if str(attr.name) == "value":
                value_attr = attr
                break
        if value_attr is None:
            continue
        try:
            constants[output_name] = np.asarray(numpy_helper.to_array(value_attr.t))
        except Exception:
            continue
    return constants


def _node_attr_int(node: onnx.NodeProto, attr_name: str, default: int) -> int:
    for attr in node.attribute:
        if str(attr.name) != str(attr_name):
            continue
        if attr.type == onnx.AttributeProto.INT:
            return int(attr.i)
        if attr.type == onnx.AttributeProto.INTS and len(attr.ints) > 0:
            return int(attr.ints[0])
    return int(default)


def _node_attr_ints(node: onnx.NodeProto, attr_name: str) -> List[int]:
    for attr in node.attribute:
        if str(attr.name) != str(attr_name):
            continue
        if attr.type == onnx.AttributeProto.INTS:
            return [int(v) for v in attr.ints]
        if attr.type == onnx.AttributeProto.INT:
            return [int(attr.i)]
    return []


def _infer_missing_tensor_ranks_with_axis_constraints(
    *,
    onnx_graph: onnx.ModelProto,
    shape_map: Dict[str, List[Any]],
) -> None:
    rank_map: Dict[str, int] = {}
    leading_dims_proven_missing: set[str] = set()
    for name, shape in shape_map.items():
        if isinstance(shape, list) and len(shape) > 0:
            rank_map[str(name)] = max(int(rank_map.get(str(name), 0)), int(len(shape)))

    constants = _collect_constant_arrays(onnx_graph)
    consumers_by_tensor: Dict[str, List[onnx.NodeProto]] = {}
    for graph_node in onnx_graph.graph.node:
        for input_name in graph_node.input:
            if str(input_name) != "":
                consumers_by_tensor.setdefault(str(input_name), []).append(graph_node)

    def _set_min_rank(
        tensor_name: str,
        rank: int,
        *,
        prove_missing_leading_dims: bool = False,
    ) -> bool:
        name = str(tensor_name)
        if name == "" or int(rank) <= 0:
            return False
        if prove_missing_leading_dims:
            current = shape_map.get(name, None)
            if isinstance(current, list) and 0 < len(current) < int(rank):
                leading_dims_proven_missing.add(name)
        prev = int(rank_map.get(name, 0))
        if int(rank) > prev:
            rank_map[name] = int(rank)
            return True
        return False

    def _unify_ranks(
        names: List[str],
        *,
        prove_short_shapes: bool = False,
    ) -> bool:
        normalized = [str(v) for v in names if str(v) != ""]
        if len(normalized) == 0:
            return False
        max_rank = max(int(rank_map.get(name, 0)) for name in normalized)
        if max_rank <= 0:
            return False
        changed = False
        propagate_missing_leading_dims = any(
            name in leading_dims_proven_missing for name in normalized
        )
        for name in normalized:
            changed = _set_min_rank(name, max_rank) or changed
            current = shape_map.get(name, None)
            if (
                (propagate_missing_leading_dims or prove_short_shapes)
                and isinstance(current, list)
                and 0 < len(current) < int(max_rank)
                and name not in leading_dims_proven_missing
            ):
                leading_dims_proven_missing.add(name)
                changed = True
        return changed

    def _slice_axes(node: onnx.NodeProto) -> List[int]:
        if len(node.input) >= 4 and str(node.input[3]) != "":
            axes_arr = constants.get(str(node.input[3]), None)
            if axes_arr is not None:
                return [int(v) for v in np.asarray(axes_arr).reshape(-1).tolist()]
        attr_axes = _node_attr_ints(node, "axes")
        if len(attr_axes) > 0:
            return [int(v) for v in attr_axes]
        if len(node.input) >= 2 and str(node.input[1]) != "":
            starts_arr = constants.get(str(node.input[1]), None)
            if starts_arr is not None:
                starts_len = int(np.asarray(starts_arr).size)
                return [int(v) for v in range(starts_len)]
        return []

    max_iter = max(1, int(len(onnx_graph.graph.node)) * 6)
    for _ in range(max_iter):
        changed = False
        for node in onnx_graph.graph.node:
            op = str(node.op_type)
            inputs = [str(v) for v in node.input if str(v) != ""]
            outputs = [str(v) for v in node.output if str(v) != ""]
            if len(outputs) == 0:
                continue

            if op in {"QuantizeLinear", "DequantizeLinear", "Identity"}:
                if len(inputs) >= 1:
                    changed = _unify_ranks([inputs[0], outputs[0]]) or changed
                continue

            if op in {"QLinearLeakyRelu", "QLinearSigmoid", "QLinearSoftmax"}:
                if len(inputs) >= 1:
                    changed = _unify_ranks(
                        [inputs[0], outputs[0]],
                        prove_short_shapes=True,
                    ) or changed
                continue

            if op == "Split":
                if len(inputs) >= 1:
                    rank_linked = [inputs[0]] + outputs
                    changed = _unify_ranks(
                        rank_linked,
                        prove_short_shapes=True,
                    ) or changed
                    axis = _node_attr_int(node, "axis", 0)
                    if axis >= 0:
                        min_rank = int(axis) + 1
                        for tensor_name in rank_linked:
                            changed = _set_min_rank(
                                tensor_name,
                                min_rank,
                                prove_missing_leading_dims=True,
                            ) or changed
                continue

            if op in {"Softmax", "LogSoftmax"}:
                if len(inputs) >= 1:
                    changed = _unify_ranks([inputs[0], outputs[0]]) or changed
                    axis = _node_attr_int(node, "axis", 1)
                    if axis >= 0:
                        min_rank = int(axis) + 1
                        changed = _set_min_rank(
                            inputs[0],
                            min_rank,
                            prove_missing_leading_dims=True,
                        ) or changed
                        changed = _set_min_rank(
                            outputs[0],
                            min_rank,
                            prove_missing_leading_dims=True,
                        ) or changed
                continue

            if op == "Flatten":
                if len(inputs) >= 1:
                    axis = _node_attr_int(node, "axis", 1)
                    current_input_shape = shape_map.get(inputs[0], None)
                    trailing_dim_required = False
                    for consumer in consumers_by_tensor.get(outputs[0], []):
                        if (
                            str(consumer.op_type) == "BatchNormalization"
                            and len(consumer.input) >= 2
                        ):
                            scale = constants.get(str(consumer.input[1]), None)
                            if scale is not None and int(np.asarray(scale).size) > 1:
                                trailing_dim_required = True
                                break
                    if (
                        axis >= 0
                        and (
                            (
                                isinstance(current_input_shape, list)
                                and len(current_input_shape) > 0
                            )
                            or trailing_dim_required
                        )
                    ):
                        # A positive Flatten axis identifies all preceding
                        # dimensions.  When shape inference retained a short,
                        # non-empty suffix, preserve one trailing dimension.
                        # A completely unknown input may legally have
                        # axis==rank. Only a downstream channel contract such
                        # as non-scalar BatchNormalization scale proves that a
                        # non-empty trailing product must exist.
                        changed = _set_min_rank(
                            inputs[0],
                            int(axis) + 1,
                            prove_missing_leading_dims=True,
                        ) or changed
                        changed = _set_min_rank(
                            outputs[0],
                            2,
                            prove_missing_leading_dims=True,
                        ) or changed
                continue

            if op == "Slice":
                if len(inputs) >= 1:
                    changed = _unify_ranks([inputs[0], outputs[0]]) or changed
                    axes = _slice_axes(node)
                    positive_axes = [int(v) for v in axes if int(v) >= 0]
                    if len(positive_axes) > 0:
                        min_rank = max(positive_axes) + 1
                        changed = _set_min_rank(
                            inputs[0],
                            min_rank,
                            prove_missing_leading_dims=True,
                        ) or changed
                        changed = _set_min_rank(
                            outputs[0],
                            min_rank,
                            prove_missing_leading_dims=True,
                        ) or changed
                continue

            if op == "QLinearConv":
                if len(inputs) >= 1:
                    changed = _set_min_rank(inputs[0], 4) or changed
                changed = _set_min_rank(outputs[0], 4) or changed
                continue

            if op in {"Concat", "QLinearConcat"}:
                if op == "QLinearConcat":
                    data_inputs = [inputs[idx] for idx in range(2, len(inputs), 3)]
                else:
                    data_inputs = list(inputs)
                rank_linked = [str(v) for v in data_inputs if str(v) != ""] + [outputs[0]]
                changed = _unify_ranks(
                    rank_linked,
                    prove_short_shapes=(op == "QLinearConcat"),
                ) or changed
                axis = _node_attr_int(node, "axis", 1)
                if axis >= 0:
                    min_rank = int(axis) + 1
                    for tensor_name in rank_linked:
                        changed = _set_min_rank(tensor_name, min_rank) or changed
                continue

        if not changed:
            break

    for tensor_name, rank in rank_map.items():
        if int(rank) <= 0:
            continue
        current = shape_map.get(str(tensor_name), None)
        if current is None or len(current) == 0:
            shape_map[str(tensor_name)] = [-1 for _ in range(int(rank))]
        elif (
            len(current) < int(rank)
            and str(tensor_name) in leading_dims_proven_missing
        ):
            # Axis-bearing ops can prove a minimum rank even when ONNX shape
            # inference dropped leading singleton/dynamic dimensions.  Keep
            # the known trailing dimensions and restore only the missing
            # leading dimensions as unknowns.
            shape_map[str(tensor_name)] = [
                -1 for _ in range(int(rank) - len(current))
            ] + list(current)


def _extract_tensor_info(
    onnx_graph: onnx.ModelProto,
) -> Tuple[Dict[str, List[Any]], Dict[str, str]]:
    shape_map: Dict[str, List[Any]] = {}
    dtype_map: Dict[str, str] = {}

    def _fill_value_info(value_info):
        if not value_info.type.HasField("tensor_type"):
            return
        name = value_info.name
        tensor_type = value_info.type.tensor_type
        dims: List[Any] = []
        if tensor_type.HasField("shape"):
            for d in tensor_type.shape.dim:
                if d.HasField("dim_value") and d.dim_value >= 0:
                    dims.append(int(d.dim_value))
                else:
                    dims.append(-1)
        shape_map[name] = dims
        dtype_map[name] = _dtype_from_onnx_elem_type(tensor_type.elem_type)

    for vi in onnx_graph.graph.input:
        _fill_value_info(vi)
    for vi in onnx_graph.graph.value_info:
        _fill_value_info(vi)
    for vi in onnx_graph.graph.output:
        _fill_value_info(vi)

    for ini in onnx_graph.graph.initializer:
        arr = numpy_helper.to_array(ini)
        shape_map[ini.name] = list(arr.shape)
        dtype_map[ini.name] = tflite_dtype_from_numpy(arr.dtype)

    _infer_missing_tensor_ranks_with_axis_constraints(
        onnx_graph=onnx_graph,
        shape_map=shape_map,
    )

    return shape_map, dtype_map


def _collect_dynamic_boundary_tensor_names(onnx_graph: onnx.ModelProto) -> Dict[str, List[str]]:
    initializer_names = {str(ini.name) for ini in onnx_graph.graph.initializer}

    def _has_dynamic_dim(value_info: Any) -> bool:
        if not value_info.type.HasField("tensor_type"):
            return True
        tensor_type = value_info.type.tensor_type
        if not tensor_type.HasField("shape"):
            return True
        for d in tensor_type.shape.dim:
            if not (d.HasField("dim_value") and int(d.dim_value) >= 0):
                return True
        return False

    dynamic_inputs: List[str] = []
    for vi in onnx_graph.graph.input:
        name = str(vi.name)
        if name in initializer_names:
            continue
        if _has_dynamic_dim(vi):
            dynamic_inputs.append(name)

    dynamic_outputs: List[str] = []
    for vi in onnx_graph.graph.output:
        name = str(vi.name)
        if _has_dynamic_dim(vi):
            dynamic_outputs.append(name)

    return {
        "inputs": dynamic_inputs,
        "outputs": dynamic_outputs,
    }


def _build_onnx_boundary_shape_signature_map(
    *,
    onnx_graph: onnx.ModelProto,
    shape_map: Dict[str, List[Any]],
) -> Dict[str, List[int]]:
    signature_map: Dict[str, List[int]] = {}
    initializer_names = {str(ini.name) for ini in onnx_graph.graph.initializer}
    for value_info in list(onnx_graph.graph.input) + list(onnx_graph.graph.output):
        name = str(value_info.name)
        if name == "" or name in initializer_names:
            continue
        raw_shape = shape_map.get(name, None)
        _, signature = normalize_onnx_shape(raw_shape)
        signature_map[name] = [int(v) for v in list(signature)]
    return signature_map


def _align_boundary_signature_to_current_shape(
    *,
    boundary_signature: Optional[List[int]],
    current_shape: Optional[List[int]],
) -> Optional[List[int]]:
    if boundary_signature is None or current_shape is None:
        return None
    signature = [int(v) for v in list(boundary_signature)]
    shape = [int(v) for v in list(current_shape)]
    if len(signature) == 0 or len(signature) != len(shape):
        return None

    # Fast-path when no layout permutation is observed.
    # Keep boundary axis contracts (including singleton dims) whenever the
    # current static shape is compatible on the same axes.
    static_axes = [int(i) for i, v in enumerate(signature) if int(v) >= 0]
    if len(static_axes) > 0 and all(int(signature[i]) == int(shape[i]) for i in static_axes):
        return [int(v) for v in list(signature)]

    aligned = [-1 for _ in range(len(signature))]
    used_axes: set[int] = set()
    static_values = sorted({int(v) for v in signature if int(v) > 1})
    for value in static_values:
        needed = int(sum(1 for v in signature if int(v) == value))
        if needed <= 0:
            continue
        candidate_axes = [int(i) for i, dim in enumerate(shape) if int(dim) == value and int(i) not in used_axes]
        if len(candidate_axes) < needed:
            continue
        for axis in candidate_axes[:needed]:
            aligned[int(axis)] = int(value)
            used_axes.add(int(axis))
    return [int(v) for v in aligned]


def _graph_has_missing_rank_info(onnx_graph: onnx.ModelProto) -> bool:
    value_infos = (
        list(onnx_graph.graph.input)
        + list(onnx_graph.graph.value_info)
        + list(onnx_graph.graph.output)
    )
    for vi in value_infos:
        if not vi.type.HasField("tensor_type"):
            continue
        if not vi.type.tensor_type.HasField("shape"):
            return True
    return False


def _infer_shapes_with_fallback(onnx_graph: onnx.ModelProto) -> onnx.ModelProto:
    if check_model_has_external_data(onnx_graph):
        return onnx_graph

    inferred_graph = onnx_graph
    try:
        inferred_graph = onnx.shape_inference.infer_shapes(inferred_graph)
    except Exception:
        pass

    if not _graph_has_missing_rank_info(inferred_graph):
        return inferred_graph

    try:
        from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
    except Exception:
        return inferred_graph

    try:
        return SymbolicShapeInference.infer_shapes(
            inferred_graph,
            auto_merge=True,
            guess_output_rank=True,
        )
    except TypeError:
        try:
            return SymbolicShapeInference.infer_shapes(inferred_graph)
        except Exception:
            return inferred_graph
    except Exception:
        return inferred_graph
