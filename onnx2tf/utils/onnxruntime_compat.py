from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import onnx
from onnx import numpy_helper


_MICROSOFT_CONTRIB_OPS = {
    "FusedConv",
    "FusedMatMul",
    "Gelu",
    "GroupNorm",
    "Inverse",
    "MultiHeadAttention",
    "QGemm",
    "QLinearAdd",
    "QLinearAveragePool",
    "QLinearConcat",
    "QLinearGlobalAveragePool",
    "QLinearLeakyRelu",
    "QLinearMul",
    "QLinearSigmoid",
    "QLinearSoftmax",
}


def _default_domain_opset(model: onnx.ModelProto) -> int | None:
    versions = [
        int(opset.version)
        for opset in model.opset_import
        if str(opset.domain) in {"", "ai.onnx"}
    ]
    return min(versions) if versions else None


def _static_shape_map(model: onnx.ModelProto) -> Dict[str, List[int]]:
    shapes: Dict[str, List[int]] = {}
    for value in [
        *model.graph.input,
        *model.graph.value_info,
        *model.graph.output,
    ]:
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        dims: List[int] = []
        for dim in tensor_type.shape.dim:
            if not dim.HasField("dim_value") or int(dim.dim_value) <= 0:
                dims = []
                break
            dims.append(int(dim.dim_value))
        if dims:
            shapes[str(value.name)] = dims
    for initializer in model.graph.initializer:
        shapes[str(initializer.name)] = [int(dim) for dim in initializer.dims]
    return shapes


def _node_attribute(node: onnx.NodeProto, name: str, default: object) -> object:
    for attribute in node.attribute:
        if str(attribute.name) == str(name):
            return onnx.helper.get_attribute_value(attribute)
    return default


def _decompose_group_norm_node(
    *,
    node: onnx.NodeProto,
    shape_map: Dict[str, List[int]],
    initializer_map: Dict[str, onnx.TensorProto],
    new_initializers: List[onnx.TensorProto],
) -> Optional[List[onnx.NodeProto]]:
    if str(node.op_type) != "GroupNorm" or str(node.domain) not in {
        "",
        "ai.onnx",
        "com.microsoft",
    }:
        return None
    if len(node.input) != 3 or len(node.output) != 1:
        return None

    input_name, scale_name, bias_name = [str(value) for value in node.input]
    input_shape = shape_map.get(input_name)
    scale_initializer = initializer_map.get(scale_name)
    bias_initializer = initializer_map.get(bias_name)
    if (
        input_shape is None
        or len(input_shape) < 3
        or scale_initializer is None
        or bias_initializer is None
    ):
        return None
    scale = np.asarray(numpy_helper.to_array(scale_initializer)).reshape(-1)
    bias = np.asarray(numpy_helper.to_array(bias_initializer)).reshape(-1)
    if scale.size == 0 or scale.size != bias.size:
        return None

    channels = int(scale.size)
    if int(input_shape[-1]) == channels:
        channel_axis = len(input_shape) - 1
    elif int(input_shape[1]) == channels:
        channel_axis = 1
    else:
        return None
    groups = int(_node_attribute(node, "groups", 1))
    if groups <= 0 or channels % groups != 0:
        return None

    group_size = channels // groups
    if channel_axis == len(input_shape) - 1:
        grouped_shape = [*input_shape[:-1], groups, group_size]
        reduce_axes = [*range(1, len(input_shape) - 1), len(input_shape)]
        affine_shape = [1] * (len(input_shape) - 1) + [channels]
    else:
        grouped_shape = [input_shape[0], groups, group_size, *input_shape[2:]]
        reduce_axes = list(range(2, len(grouped_shape)))
        affine_shape = [1, channels] + [1] * (len(input_shape) - 2)

    prefix = str(node.name or node.output[0]) + "_ort_compat"
    grouped_shape_name = f"{prefix}_grouped_shape"
    original_shape_name = f"{prefix}_original_shape"
    affine_shape_name = f"{prefix}_affine_shape"
    epsilon_name = f"{prefix}_epsilon"
    new_initializers.extend(
        [
            numpy_helper.from_array(
                np.asarray(grouped_shape, dtype=np.int64),
                name=grouped_shape_name,
            ),
            numpy_helper.from_array(
                np.asarray(input_shape, dtype=np.int64),
                name=original_shape_name,
            ),
            numpy_helper.from_array(
                np.asarray(affine_shape, dtype=np.int64),
                name=affine_shape_name,
            ),
            numpy_helper.from_array(
                np.asarray(
                    float(_node_attribute(node, "epsilon", 1e-5)),
                    dtype=scale.dtype,
                ),
                name=epsilon_name,
            ),
        ]
    )

    grouped = f"{prefix}_grouped"
    mean = f"{prefix}_mean"
    centered = f"{prefix}_centered"
    squared = f"{prefix}_squared"
    variance = f"{prefix}_variance"
    variance_epsilon = f"{prefix}_variance_epsilon"
    std = f"{prefix}_std"
    normalized_grouped = f"{prefix}_normalized_grouped"
    normalized = f"{prefix}_normalized"
    scale_broadcast = f"{prefix}_scale"
    bias_broadcast = f"{prefix}_bias"
    scaled = f"{prefix}_scaled"
    affine = (
        f"{prefix}_affine"
        if int(_node_attribute(node, "activation", 0)) == 1
        else str(node.output[0])
    )
    replacement = [
        onnx.helper.make_node(
            "Reshape", [input_name, grouped_shape_name], [grouped], name=f"{prefix}_reshape_grouped"
        ),
        onnx.helper.make_node(
            "ReduceMean", [grouped], [mean], name=f"{prefix}_mean_node", axes=reduce_axes, keepdims=1
        ),
        onnx.helper.make_node("Sub", [grouped, mean], [centered], name=f"{prefix}_center"),
        onnx.helper.make_node("Mul", [centered, centered], [squared], name=f"{prefix}_square"),
        onnx.helper.make_node(
            "ReduceMean", [squared], [variance], name=f"{prefix}_variance_node", axes=reduce_axes, keepdims=1
        ),
        onnx.helper.make_node(
            "Add", [variance, epsilon_name], [variance_epsilon], name=f"{prefix}_add_epsilon"
        ),
        onnx.helper.make_node("Sqrt", [variance_epsilon], [std], name=f"{prefix}_sqrt"),
        onnx.helper.make_node(
            "Div", [centered, std], [normalized_grouped], name=f"{prefix}_normalize"
        ),
        onnx.helper.make_node(
            "Reshape", [normalized_grouped, original_shape_name], [normalized], name=f"{prefix}_reshape_original"
        ),
        onnx.helper.make_node(
            "Reshape", [scale_name, affine_shape_name], [scale_broadcast], name=f"{prefix}_reshape_scale"
        ),
        onnx.helper.make_node(
            "Reshape", [bias_name, affine_shape_name], [bias_broadcast], name=f"{prefix}_reshape_bias"
        ),
        onnx.helper.make_node("Mul", [normalized, scale_broadcast], [scaled], name=f"{prefix}_scale_node"),
        onnx.helper.make_node("Add", [scaled, bias_broadcast], [affine], name=f"{prefix}_bias_node"),
    ]
    if int(_node_attribute(node, "activation", 0)) == 1:
        sigmoid = f"{prefix}_sigmoid"
        replacement.extend(
            [
                onnx.helper.make_node("Sigmoid", [affine], [sigmoid], name=f"{prefix}_sigmoid_node"),
                onnx.helper.make_node(
                    "Mul", [affine, sigmoid], [str(node.output[0])], name=f"{prefix}_swish"
                ),
            ]
        )
    return replacement


def _decompose_group_norm_for_onnxruntime(
    model: onnx.ModelProto,
) -> Dict[str, int]:
    shape_map = _static_shape_map(model)
    initializer_map = {
        str(initializer.name): initializer
        for initializer in model.graph.initializer
    }
    rewritten_nodes: List[onnx.NodeProto] = []
    new_initializers: List[onnx.TensorProto] = []
    rewritten_count = 0
    for node in model.graph.node:
        replacement = _decompose_group_norm_node(
            node=node,
            shape_map=shape_map,
            initializer_map=initializer_map,
            new_initializers=new_initializers,
        )
        if replacement is None:
            rewritten_nodes.append(node)
            continue
        rewritten_nodes.extend(replacement)
        rewritten_count += 1
    if rewritten_count == 0:
        return {}
    del model.graph.node[:]
    model.graph.node.extend(rewritten_nodes)
    model.graph.initializer.extend(new_initializers)
    return {"GroupNorm": rewritten_count}


def prepare_onnx_graph_for_onnxruntime(
    onnx_graph: onnx.ModelProto,
) -> tuple[onnx.ModelProto, Dict[str, int]]:
    """Build an evaluation-only graph compatible with current ONNX Runtime."""

    prepared = onnx.ModelProto()
    prepared.CopyFrom(onnx_graph)

    default_opset = _default_domain_opset(prepared)
    if default_opset is not None and int(default_opset) < 7:
        try:
            prepared = onnx.version_converter.convert_version(prepared, 7)
            default_opset = _default_domain_opset(prepared)
        except Exception:
            pass

    rewritten: Dict[str, int] = _decompose_group_norm_for_onnxruntime(prepared)
    for node in prepared.graph.node:
        if str(node.domain) not in {"", "ai.onnx"}:
            continue
        is_contrib = str(node.op_type) in _MICROSOFT_CONTRIB_OPS
        is_legacy_grid_sample = (
            str(node.op_type) == "GridSample"
            and default_opset is not None
            and int(default_opset) < 16
        )
        if not is_contrib and not is_legacy_grid_sample:
            continue
        node.domain = "com.microsoft"
        op_type = str(node.op_type)
        rewritten[op_type] = int(rewritten.get(op_type, 0)) + 1

    if rewritten and not any(
        str(opset.domain) == "com.microsoft" for opset in prepared.opset_import
    ):
        prepared.opset_import.append(
            onnx.helper.make_operatorsetid("com.microsoft", 1)
        )
    return prepared, rewritten
