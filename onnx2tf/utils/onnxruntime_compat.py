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


def _upgrade_legacy_upsample_for_onnxruntime(
    model: onnx.ModelProto,
) -> Dict[str, int]:
    default_opset = _default_domain_opset(model)
    legacy_upsample_count = int(
        sum(
            1
            for node in model.graph.node
            if str(node.domain) in {"", "ai.onnx"}
            and str(node.op_type) == "Upsample"
        )
    )
    if (
        default_opset is None
        or int(default_opset) >= 10
        or legacy_upsample_count == 0
    ):
        return {}
    try:
        converted = onnx.version_converter.convert_version(model, 11)
    except Exception:
        return {}
    model.CopyFrom(converted)
    return {"LegacyUpsample": legacy_upsample_count}


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


def _graph_static_shapes(graph: onnx.GraphProto) -> Dict[str, List[int]]:
    shapes: Dict[str, List[int]] = {}
    for value in [*graph.input, *graph.value_info, *graph.output]:
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
    for initializer in graph.initializer:
        shapes[str(initializer.name)] = [int(dim) for dim in initializer.dims]
    return shapes


def _graph_tensor_elem_types(graph: onnx.GraphProto) -> Dict[str, int]:
    elem_types: Dict[str, int] = {}
    for value in [*graph.input, *graph.value_info, *graph.output]:
        tensor_type = value.type.tensor_type
        if int(tensor_type.elem_type) != int(onnx.TensorProto.UNDEFINED):
            elem_types[str(value.name)] = int(tensor_type.elem_type)
    for initializer in graph.initializer:
        elem_types[str(initializer.name)] = int(initializer.data_type)
    return elem_types


def _sequenceconstruct_tensor_spec(
    graph: onnx.GraphProto,
) -> Optional[tuple[onnx.NodeProto, List[int], int]]:
    if len(graph.node) == 0 or len(graph.output) != 1:
        return None
    terminal = graph.node[-1]
    if (
        str(terminal.op_type) != "SequenceConstruct"
        or len(terminal.output) != 1
        or str(terminal.output[0]) != str(graph.output[0].name)
        or len(terminal.input) == 0
    ):
        return None
    if any(str(node.op_type) not in {"Add", "Constant"} for node in graph.node[:-1]):
        return None

    shapes = _graph_static_shapes(graph)
    elem_types = _graph_tensor_elem_types(graph)
    input_shapes = [shapes.get(str(name)) for name in terminal.input]
    input_elem_types = [elem_types.get(str(name)) for name in terminal.input]
    if any(shape is None or len(shape) == 0 for shape in input_shapes):
        return None
    if any(elem_type is None for elem_type in input_elem_types):
        return None
    concrete_shapes = [list(shape) for shape in input_shapes if shape is not None]
    rank = len(concrete_shapes[0])
    tail = concrete_shapes[0][1:]
    if any(len(shape) != rank or shape[1:] != tail for shape in concrete_shapes[1:]):
        return None
    concrete_elem_types = [int(value) for value in input_elem_types if value is not None]
    if any(value != concrete_elem_types[0] for value in concrete_elem_types[1:]):
        return None
    output_shape = [sum(int(shape[0]) for shape in concrete_shapes), *tail]
    return terminal, output_shape, concrete_elem_types[0]


def _replace_sequenceconstruct_with_tensor(
    *,
    graph: onnx.GraphProto,
    terminal: onnx.NodeProto,
    output_shape: List[int],
    elem_type: int,
) -> None:
    terminal.domain = ""
    del terminal.attribute[:]
    if len(terminal.input) == 1:
        terminal.op_type = "Identity"
    else:
        terminal.op_type = "Concat"
        terminal.attribute.extend([onnx.helper.make_attribute("axis", 0)])
    output_name = str(terminal.output[0])
    graph.output[0].CopyFrom(
        onnx.helper.make_tensor_value_info(
            output_name,
            int(elem_type),
            [int(dim) for dim in output_shape],
        )
    )


def _repair_if_sequenceconstruct_outputs_for_onnxruntime(
    model: onnx.ModelProto,
) -> Dict[str, int]:
    rewritten_count = 0

    def visit_graph(graph: onnx.GraphProto) -> None:
        nonlocal rewritten_count
        for node in graph.node:
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    visit_graph(attribute.g)
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for child_graph in attribute.graphs:
                        visit_graph(child_graph)
            if str(node.op_type) != "If" or len(node.output) != 1:
                continue
            branch_graphs = {
                str(attribute.name): attribute.g
                for attribute in node.attribute
                if attribute.type == onnx.AttributeProto.GRAPH
            }
            then_graph = branch_graphs.get("then_branch")
            else_graph = branch_graphs.get("else_branch")
            if then_graph is None or else_graph is None:
                continue
            then_spec = _sequenceconstruct_tensor_spec(then_graph)
            else_spec = _sequenceconstruct_tensor_spec(else_graph)
            if then_spec is None or else_spec is None:
                continue
            then_terminal, then_shape, then_elem_type = then_spec
            else_terminal, else_shape, else_elem_type = else_spec
            if (
                len(then_shape) != len(else_shape)
                or then_shape[1:] != else_shape[1:]
                or int(then_elem_type) != int(else_elem_type)
            ):
                continue
            _replace_sequenceconstruct_with_tensor(
                graph=then_graph,
                terminal=then_terminal,
                output_shape=then_shape,
                elem_type=then_elem_type,
            )
            _replace_sequenceconstruct_with_tensor(
                graph=else_graph,
                terminal=else_terminal,
                output_shape=else_shape,
                elem_type=else_elem_type,
            )
            dynamic_output_shape: List[object] = [
                f"{str(node.name or node.output[0])}_axis0",
                *then_shape[1:],
            ]
            for value in [*graph.value_info, *graph.output]:
                if str(value.name) == str(node.output[0]):
                    value.CopyFrom(
                        onnx.helper.make_tensor_value_info(
                            str(node.output[0]),
                            int(then_elem_type),
                            dynamic_output_shape,
                        )
                    )
            rewritten_count += 1

    visit_graph(model.graph)
    return {"IfSequenceConstruct": rewritten_count} if rewritten_count else {}


def _rewrite_integer_matmul_for_onnxruntime(
    model: onnx.ModelProto,
) -> Dict[str, int]:
    integer_types = {
        int(onnx.TensorProto.INT8),
        int(onnx.TensorProto.INT16),
        int(onnx.TensorProto.INT32),
        int(onnx.TensorProto.INT64),
        int(onnx.TensorProto.UINT8),
        int(onnx.TensorProto.UINT16),
        int(onnx.TensorProto.UINT32),
        int(onnx.TensorProto.UINT64),
    }
    rewritten_count = 0

    def visit_graph(graph: onnx.GraphProto) -> None:
        nonlocal rewritten_count
        elem_types = _graph_tensor_elem_types(graph)
        rewritten_nodes: List[onnx.NodeProto] = []
        for node in graph.node:
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    visit_graph(attribute.g)
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for child_graph in attribute.graphs:
                        visit_graph(child_graph)
            if (
                str(node.op_type) != "MatMul"
                or str(node.domain) not in {"", "ai.onnx"}
                or len(node.input) != 2
                or len(node.output) != 1
            ):
                rewritten_nodes.append(node)
                continue
            lhs_type = elem_types.get(str(node.input[0]))
            rhs_type = elem_types.get(str(node.input[1]))
            output_type = elem_types.get(str(node.output[0]))
            if (
                lhs_type not in integer_types
                or rhs_type not in integer_types
                or output_type not in integer_types
            ):
                rewritten_nodes.append(node)
                continue

            prefix = str(node.name or node.output[0]) + "_ort_compat"
            lhs_float = f"{prefix}_lhs_float"
            rhs_float = f"{prefix}_rhs_float"
            result_float = f"{prefix}_result_float"
            rewritten_nodes.extend(
                [
                    onnx.helper.make_node(
                        "Cast",
                        [str(node.input[0])],
                        [lhs_float],
                        name=f"{prefix}_cast_lhs",
                        to=int(onnx.TensorProto.FLOAT),
                    ),
                    onnx.helper.make_node(
                        "Cast",
                        [str(node.input[1])],
                        [rhs_float],
                        name=f"{prefix}_cast_rhs",
                        to=int(onnx.TensorProto.FLOAT),
                    ),
                    onnx.helper.make_node(
                        "MatMul",
                        [lhs_float, rhs_float],
                        [result_float],
                        name=f"{prefix}_matmul",
                    ),
                    onnx.helper.make_node(
                        "Cast",
                        [result_float],
                        [str(node.output[0])],
                        name=f"{prefix}_cast_output",
                        to=int(output_type),
                    ),
                ]
            )
            rewritten_count += 1
        del graph.node[:]
        graph.node.extend(rewritten_nodes)

    visit_graph(model.graph)
    return {"IntegerMatMul": rewritten_count} if rewritten_count else {}


def _rewrite_tensor_optional_has_element_for_onnxruntime(
    model: onnx.ModelProto,
) -> Dict[str, int]:
    rewritten_count = 0

    def visit_graph(
        graph: onnx.GraphProto,
        outer_tensor_names: set[str],
    ) -> None:
        nonlocal rewritten_count
        local_tensor_names = {
            str(value.name)
            for value in [*graph.input, *graph.value_info, *graph.output]
            if value.type.HasField("tensor_type")
        }
        local_tensor_names.update(str(value.name) for value in graph.initializer)
        visible_tensor_names = outer_tensor_names | local_tensor_names
        rewritten_nodes: List[onnx.NodeProto] = []
        for node in graph.node:
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    visit_graph(attribute.g, visible_tensor_names)
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for child_graph in attribute.graphs:
                        visit_graph(child_graph, visible_tensor_names)
            if (
                str(node.op_type) == "OptionalHasElement"
                and str(node.domain) in {"", "ai.onnx"}
                and len(node.input) == 1
                and len(node.output) == 1
                and str(node.input[0]) in visible_tensor_names
            ):
                rewritten_nodes.append(
                    onnx.helper.make_node(
                        "Constant",
                        [],
                        [str(node.output[0])],
                        name=str(node.name),
                        value=numpy_helper.from_array(
                            np.asarray(True, dtype=np.bool_)
                        ),
                    )
                )
                rewritten_count += 1
            else:
                rewritten_nodes.append(node)
        del graph.node[:]
        graph.node.extend(rewritten_nodes)

    visit_graph(model.graph, set())
    return {"TensorOptionalHasElement": rewritten_count} if rewritten_count else {}


def _repair_unknown_rank_conv_io_for_onnxruntime(
    model: onnx.ModelProto,
) -> Dict[str, int]:
    rewritten_count = 0

    def visit_graph(graph: onnx.GraphProto) -> None:
        nonlocal rewritten_count
        initializer_map = {
            str(initializer.name): initializer
            for initializer in graph.initializer
        }
        value_map = {
            str(value.name): value
            for value in [*graph.input, *graph.value_info, *graph.output]
        }
        for node in graph.node:
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    visit_graph(attribute.g)
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for child_graph in attribute.graphs:
                        visit_graph(child_graph)
            if (
                str(node.op_type) not in {"Conv", "FusedConv"}
                or len(node.input) < 2
                or len(node.output) != 1
            ):
                continue
            input_value = value_map.get(str(node.input[0]))
            output_value = value_map.get(str(node.output[0]))
            weights = initializer_map.get(str(node.input[1]))
            if input_value is None or weights is None or len(weights.dims) < 3:
                continue
            input_tensor_type = input_value.type.tensor_type
            input_has_unknown_rank = (
                not input_tensor_type.HasField("shape")
                or len(input_tensor_type.shape.dim) == 0
            )
            if not input_has_unknown_rank:
                continue
            spatial_rank = len(weights.dims) - 2
            group = int(_node_attribute(node, "group", 1))
            input_channels = int(weights.dims[1]) * int(group)
            output_channels = int(weights.dims[0])
            symbolic_input_shape: List[object] = [
                f"{str(node.name or node.input[0])}_batch",
                input_channels,
                *[
                    f"{str(node.name or node.input[0])}_spatial_{index}"
                    for index in range(spatial_rank)
                ],
            ]
            input_value.CopyFrom(
                onnx.helper.make_tensor_value_info(
                    str(node.input[0]),
                    int(input_tensor_type.elem_type),
                    symbolic_input_shape,
                )
            )
            if output_value is not None:
                output_tensor_type = output_value.type.tensor_type
                output_has_unknown_rank = (
                    not output_tensor_type.HasField("shape")
                    or len(output_tensor_type.shape.dim) == 0
                )
                if output_has_unknown_rank:
                    symbolic_output_shape: List[object] = [
                        symbolic_input_shape[0],
                        output_channels,
                        *symbolic_input_shape[2:],
                    ]
                    output_value.CopyFrom(
                        onnx.helper.make_tensor_value_info(
                            str(node.output[0]),
                            int(output_tensor_type.elem_type),
                            symbolic_output_shape,
                        )
                    )
            rewritten_count += 1

    visit_graph(model.graph)
    return {"UnknownRankConv": rewritten_count} if rewritten_count else {}


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

    rewritten: Dict[str, int] = _upgrade_legacy_upsample_for_onnxruntime(
        prepared
    )
    default_opset = _default_domain_opset(prepared)
    for op_type, count in _decompose_group_norm_for_onnxruntime(prepared).items():
        rewritten[op_type] = int(rewritten.get(op_type, 0)) + int(count)
    for op_type, count in _repair_if_sequenceconstruct_outputs_for_onnxruntime(
        prepared
    ).items():
        rewritten[op_type] = int(rewritten.get(op_type, 0)) + int(count)
    for op_type, count in _rewrite_integer_matmul_for_onnxruntime(prepared).items():
        rewritten[op_type] = int(rewritten.get(op_type, 0)) + int(count)
    for op_type, count in _rewrite_tensor_optional_has_element_for_onnxruntime(
        prepared
    ).items():
        rewritten[op_type] = int(rewritten.get(op_type, 0)) + int(count)
    for op_type, count in _repair_unknown_rank_conv_io_for_onnxruntime(
        prepared
    ).items():
        rewritten[op_type] = int(rewritten.get(op_type, 0)) + int(count)
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
