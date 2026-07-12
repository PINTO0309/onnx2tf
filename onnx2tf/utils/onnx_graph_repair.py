from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper


def _attribute_int(node: onnx.NodeProto, name: str, default: int) -> int:
    for attribute in node.attribute:
        if str(attribute.name) == str(name):
            return int(attribute.i)
    return int(default)


def _constant_array(
    name: str,
    *,
    initializers: Dict[str, onnx.TensorProto],
    producers: Dict[str, onnx.NodeProto],
) -> Optional[np.ndarray]:
    initializer = initializers.get(str(name))
    if initializer is not None:
        return np.asarray(numpy_helper.to_array(initializer))
    producer = producers.get(str(name))
    if producer is None or str(producer.op_type) != "Constant":
        return None
    for attribute in producer.attribute:
        if str(attribute.name) == "value" and attribute.HasField("t"):
            return np.asarray(numpy_helper.to_array(attribute.t))
    return None


def _topk_segment_length(
    tensor_name: str,
    *,
    producers: Dict[str, onnx.NodeProto],
    initializers: Dict[str, onnx.TensorProto],
) -> Optional[int]:
    producer = producers.get(str(tensor_name))
    if producer is None:
        return None
    topk_node: Optional[onnx.NodeProto] = None
    if str(producer.op_type) == "TopK":
        topk_node = producer
    elif str(producer.op_type) == "Add":
        for input_name in producer.input:
            candidate = producers.get(str(input_name))
            if candidate is not None and str(candidate.op_type) == "TopK":
                topk_node = candidate
                break
    if topk_node is None or len(topk_node.input) < 2:
        return None
    k_value = _constant_array(
        str(topk_node.input[1]),
        initializers=initializers,
        producers=producers,
    )
    if k_value is None or int(np.asarray(k_value).size) != 1:
        return None
    k = int(np.asarray(k_value, dtype=np.int64).reshape(-1)[0])
    return int(k) if k > 0 else None


def _find_topk_index_concat(
    start_name: str,
    *,
    producers: Dict[str, onnx.NodeProto],
    initializers: Dict[str, onnx.TensorProto],
) -> Optional[Tuple[onnx.NodeProto, List[int]]]:
    queue = deque([(str(start_name), 0)])
    visited: set[str] = set()
    while queue:
        tensor_name, depth = queue.popleft()
        if tensor_name in visited or depth > 12:
            continue
        visited.add(tensor_name)
        producer = producers.get(tensor_name)
        if producer is None:
            continue
        if str(producer.op_type) == "Concat" and len(producer.input) > 0:
            segment_lengths = [
                _topk_segment_length(
                    str(input_name),
                    producers=producers,
                    initializers=initializers,
                )
                for input_name in producer.input
            ]
            if all(length is not None for length in segment_lengths):
                return producer, [int(length) for length in segment_lengths if length is not None]
        for input_name in producer.input:
            if str(input_name) != "":
                queue.append((str(input_name), int(depth) + 1))
    return None


def _nms_guard_capture_names(
    if_node: onnx.NodeProto,
) -> Optional[Tuple[str, str, str]]:
    then_graph: Optional[onnx.GraphProto] = None
    else_graph: Optional[onnx.GraphProto] = None
    for attribute in if_node.attribute:
        if str(attribute.name) == "then_branch":
            then_graph = attribute.g
        elif str(attribute.name) == "else_branch":
            else_graph = attribute.g
    if then_graph is None or else_graph is None or len(then_graph.node) != 0:
        return None
    reduce_nodes = [node for node in else_graph.node if str(node.op_type) == "ReduceMax"]
    cast_nodes = [node for node in else_graph.node if str(node.op_type) == "Cast"]
    nms_nodes = [node for node in else_graph.node if str(node.op_type) == "NonMaxSuppression"]
    if len(reduce_nodes) != 1 or len(cast_nodes) != 1 or len(nms_nodes) != 1:
        return None
    nms_node = nms_nodes[0]
    if len(reduce_nodes[0].input) != 1 or len(cast_nodes[0].input) != 1 or len(nms_node.input) < 2:
        return None
    local_producers = {
        str(output_name): node
        for node in else_graph.node
        for output_name in node.output
        if str(output_name) != ""
    }
    score_name = str(nms_node.input[1])
    visited: set[str] = set()
    while score_name not in visited:
        visited.add(score_name)
        producer = local_producers.get(score_name)
        if producer is None or str(producer.op_type) != "Unsqueeze" or len(producer.input) < 1:
            break
        score_name = str(producer.input[0])
    return (
        str(reduce_nodes[0].input[0]),
        str(score_name),
        str(cast_nodes[0].input[0]),
    )


def _repair_graph(graph: onnx.GraphProto) -> int:
    nodes = list(graph.node)
    producers = {
        str(output_name): node
        for node in nodes
        for output_name in node.output
        if str(output_name) != ""
    }
    initializers = {
        str(initializer.name): initializer
        for initializer in graph.initializer
        if str(initializer.name) != ""
    }
    available = {
        str(value.name)
        for value in graph.input
        if str(value.name) != ""
    } | set(initializers)
    repaired = 0
    rewritten_nodes: List[onnx.NodeProto] = []

    for if_node in nodes:
        captures = _nms_guard_capture_names(if_node) if str(if_node.op_type) == "If" else None
        if captures is not None:
            boxes_name, scores_name, indices_name = captures
            missing_scores = scores_name not in available
            missing_indices = indices_name not in available
            boxes_producer = producers.get(boxes_name)
            if (
                (missing_scores or missing_indices)
                and boxes_name in available
                and boxes_producer is not None
                and str(boxes_producer.op_type) == "Gather"
                and len(boxes_producer.input) >= 2
            ):
                final_indices_name = str(boxes_producer.input[1])
                prefiltered_boxes_name = str(boxes_producer.input[0])
                prefiltered_boxes_producer = producers.get(prefiltered_boxes_name)
                prefilter_indices_name = (
                    str(prefiltered_boxes_producer.input[1])
                    if prefiltered_boxes_producer is not None
                    and str(prefiltered_boxes_producer.op_type) == "Gather"
                    and len(prefiltered_boxes_producer.input) >= 2
                    else ""
                )
                score_candidates = [
                    node
                    for node in rewritten_nodes
                    if str(node.op_type) == "Gather"
                    and len(node.input) >= 2
                    and str(node.input[1]) == prefilter_indices_name
                    and len(node.output) == 1
                    and str(node.output[0]) != prefiltered_boxes_name
                ]
                score_candidate = score_candidates[-1] if score_candidates else None
                level_match = (
                    _find_topk_index_concat(
                        str(score_candidate.input[0]),
                        producers=producers,
                        initializers=initializers,
                    )
                    if score_candidate is not None
                    else None
                )
                if score_candidate is not None and (not missing_indices or level_match is not None):
                    if missing_scores:
                        rewritten_nodes.append(
                            onnx.helper.make_node(
                                "Gather",
                                [str(score_candidate.output[0]), final_indices_name],
                                [scores_name],
                                name=f"{if_node.name}_repair_nms_scores",
                                axis=0,
                            )
                        )
                        available.add(scores_name)
                    if missing_indices and level_match is not None:
                        _, segment_lengths = level_match
                        levels_name = f"{if_node.name}_repair_nms_levels"
                        filtered_levels_name = f"{levels_name}_prefiltered"
                        levels = np.concatenate(
                            [
                                np.full((int(length),), level, dtype=np.int64)
                                for level, length in enumerate(segment_lengths)
                            ],
                            axis=0,
                        )
                        graph.initializer.append(
                            numpy_helper.from_array(levels, name=levels_name)
                        )
                        initializers[levels_name] = graph.initializer[-1]
                        available.add(levels_name)
                        rewritten_nodes.extend(
                            [
                                onnx.helper.make_node(
                                    "Gather",
                                    [levels_name, prefilter_indices_name],
                                    [filtered_levels_name],
                                    name=f"{if_node.name}_repair_nms_levels_prefilter",
                                    axis=0,
                                ),
                                onnx.helper.make_node(
                                    "Gather",
                                    [filtered_levels_name, final_indices_name],
                                    [indices_name],
                                    name=f"{if_node.name}_repair_nms_levels_final",
                                    axis=0,
                                ),
                            ]
                        )
                        available.update({filtered_levels_name, indices_name})
                    if scores_name in available and indices_name in available:
                        repaired += 1

        rewritten_nodes.append(if_node)
        available.update(str(name) for name in if_node.output if str(name) != "")

    if repaired > 0:
        del graph.node[:]
        graph.node.extend(rewritten_nodes)
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                repaired += _repair_graph(attribute.g)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for child_graph in attribute.graphs:
                    repaired += _repair_graph(child_graph)
    return int(repaired)


def repair_missing_torchvision_nms_guard_captures(
    model: onnx.ModelProto,
) -> Dict[str, int]:
    """Restore optimizer-pruned implicit captures in TorchVision NMS guards."""

    repaired = _repair_graph(model.graph)
    return {"TorchVisionNmsGuardCaptures": repaired} if repaired else {}


def _tensor_value_info(
    graph: onnx.GraphProto,
    tensor_name: str,
) -> Optional[onnx.ValueInfoProto]:
    for value in [*graph.input, *graph.value_info, *graph.output]:
        if str(value.name) == str(tensor_name):
            return value
    return None


def _static_tensor_shape(value: onnx.ValueInfoProto) -> Optional[List[int]]:
    tensor_type = value.type.tensor_type
    if not tensor_type.HasField("shape"):
        return None
    shape: List[int] = []
    for dim in tensor_type.shape.dim:
        shape.append(int(dim.dim_value) if dim.HasField("dim_value") else -1)
    return shape


def _loop_body(node: onnx.NodeProto) -> Optional[onnx.GraphProto]:
    bodies = [
        attribute.g
        for attribute in node.attribute
        if str(attribute.name) == "body"
        and attribute.type == onnx.AttributeProto.GRAPH
    ]
    return bodies[0] if len(bodies) == 1 else None


def _graph_captures(graph: onnx.GraphProto) -> set[str]:
    local = {
        str(value.name)
        for value in graph.input
        if str(value.name) != ""
    } | {
        str(initializer.name)
        for initializer in graph.initializer
        if str(initializer.name) != ""
    } | {
        str(output_name)
        for node in graph.node
        for output_name in node.output
        if str(output_name) != ""
    }
    return {
        str(input_name)
        for node in graph.node
        for input_name in node.input
        if str(input_name) != "" and str(input_name) not in local
    }


def _scalar_constant_int(
    name: str,
    *,
    initializers: Dict[str, onnx.TensorProto],
    producers: Dict[str, onnx.NodeProto],
) -> Optional[int]:
    value = _constant_array(
        str(name),
        initializers=initializers,
        producers=producers,
    )
    if value is None or int(np.asarray(value).size) != 1:
        return None
    return int(np.asarray(value, dtype=np.int64).reshape(-1)[0])


def _paste_masks_loop_pattern(
    loop_node: onnx.NodeProto,
    *,
    graph: onnx.GraphProto,
    available: set[str],
    producers: Dict[str, onnx.NodeProto],
    initializers: Dict[str, onnx.TensorProto],
) -> Optional[Tuple[str, onnx.NodeProto, onnx.ValueInfoProto]]:
    body = _loop_body(loop_node)
    if (
        body is None
        or len(loop_node.input) != 3
        or len(loop_node.output) != 1
        or len(body.input) != 3
        or len(body.output) != 2
    ):
        return None
    captures = _graph_captures(body)
    missing = sorted(captures - available)
    present = sorted(captures & available)
    if len(missing) != 1 or len(present) != 1:
        return None

    iteration_name = str(body.input[0].name)
    missing_name = missing[0]
    mask_name = present[0]
    mask_gathers = [
        node
        for node in body.node
        if str(node.op_type) == "Gather"
        and len(node.input) >= 2
        and str(node.input[0]) == mask_name
        and str(node.input[1]) == iteration_name
        and len(node.output) == 1
    ]
    box_gathers = [
        node
        for node in body.node
        if str(node.op_type) == "Gather"
        and len(node.input) >= 2
        and str(node.input[0]) == missing_name
        and str(node.input[1]) == iteration_name
        and len(node.output) == 1
    ]
    if len(mask_gathers) != 1 or len(box_gathers) != 1:
        return None
    box_row_name = str(box_gathers[0].output[0])
    coordinate_indices = sorted(
        index
        for node in body.node
        if str(node.op_type) == "Gather"
        and len(node.input) >= 2
        and str(node.input[0]) == box_row_name
        and (
            index := _scalar_constant_int(
                str(node.input[1]),
                initializers=initializers,
                producers=producers,
            )
        )
        is not None
    )
    if coordinate_indices != [0, 1, 2, 3]:
        return None

    pad_node = producers.get(mask_name)
    if (
        pad_node is None
        or str(pad_node.op_type) != "Pad"
        or len(pad_node.input) < 2
        or len(pad_node.output) != 1
    ):
        return None
    pads = _constant_array(
        str(pad_node.input[1]),
        initializers=initializers,
        producers=producers,
    )
    if pads is None:
        return None
    pads = np.asarray(pads, dtype=np.int64).reshape(-1)
    if (
        len(pads) != 8
        or pads.tolist()[:2] != [0, 0]
        or pads.tolist()[4:6] != [0, 0]
        or int(pads[2]) <= 0
        or int(pads[2]) != int(pads[3])
        or int(pads[2]) != int(pads[6])
        or int(pads[2]) != int(pads[7])
    ):
        return None

    trip_producer = producers.get(str(loop_node.input[0]))
    if (
        trip_producer is None
        or str(trip_producer.op_type) != "Gather"
        or len(trip_producer.input) < 2
        or _scalar_constant_int(
            str(trip_producer.input[1]),
            initializers=initializers,
            producers=producers,
        )
        != 0
    ):
        return None
    shape_producer = producers.get(str(trip_producer.input[0]))
    if (
        shape_producer is None
        or str(shape_producer.op_type) != "Shape"
        or list(shape_producer.input) != [mask_name]
    ):
        return None

    box_outputs: List[onnx.ValueInfoProto] = []
    for value in graph.output:
        shape = _static_tensor_shape(value)
        tensor_type = value.type.tensor_type
        if (
            shape is not None
            and len(shape) == 2
            and int(shape[1]) == 4
            and int(tensor_type.elem_type)
            in {TensorProto.FLOAT, TensorProto.DOUBLE}
            and str(value.name) in available
        ):
            box_outputs.append(value)
    if len(box_outputs) != 1:
        return None
    return missing_name, pad_node, box_outputs[0]


def _make_paste_masks_box_repair_nodes(
    *,
    loop_node: onnx.NodeProto,
    missing_name: str,
    pad_node: onnx.NodeProto,
    boxes: onnx.ValueInfoProto,
    graph: onnx.GraphProto,
) -> List[onnx.NodeProto]:
    prefix = f"{loop_node.name or loop_node.output[0]}_repair_paste_masks"
    float_type = int(boxes.type.tensor_type.elem_type)
    float_dtype = np.float64 if float_type == TensorProto.DOUBLE else np.float32
    constants = {
        "spatial_axis": np.asarray(3, dtype=np.int64),
        "half": np.asarray(0.5, dtype=float_dtype),
        "unsqueeze_axis": np.asarray([1], dtype=np.int64),
    }
    for index in range(4):
        constants[f"coordinate_{index}"] = np.asarray(index, dtype=np.int64)
    for suffix, value in constants.items():
        graph.initializer.append(
            numpy_helper.from_array(value, name=f"{prefix}_{suffix}")
        )

    source_shape = f"{prefix}_source_shape"
    padded_shape = f"{prefix}_padded_shape"
    source_size = f"{prefix}_source_size"
    padded_size = f"{prefix}_padded_size"
    source_size_float = f"{source_size}_float"
    padded_size_float = f"{padded_size}_float"
    scale = f"{prefix}_scale"
    nodes: List[onnx.NodeProto] = [
        onnx.helper.make_node(
            "Shape",
            [str(pad_node.input[0])],
            [source_shape],
            name=f"{prefix}_source_shape_node",
        ),
        onnx.helper.make_node(
            "Shape",
            [str(pad_node.output[0])],
            [padded_shape],
            name=f"{prefix}_padded_shape_node",
        ),
        onnx.helper.make_node(
            "Gather",
            [source_shape, f"{prefix}_spatial_axis"],
            [source_size],
            name=f"{prefix}_source_size_node",
            axis=0,
        ),
        onnx.helper.make_node(
            "Gather",
            [padded_shape, f"{prefix}_spatial_axis"],
            [padded_size],
            name=f"{prefix}_padded_size_node",
            axis=0,
        ),
        onnx.helper.make_node(
            "Cast",
            [source_size],
            [source_size_float],
            name=f"{prefix}_source_size_cast",
            to=float_type,
        ),
        onnx.helper.make_node(
            "Cast",
            [padded_size],
            [padded_size_float],
            name=f"{prefix}_padded_size_cast",
            to=float_type,
        ),
        onnx.helper.make_node(
            "Div",
            [padded_size_float, source_size_float],
            [scale],
            name=f"{prefix}_scale_node",
        ),
    ]
    coordinates: List[str] = []
    for index in range(4):
        coordinate = f"{prefix}_coordinate_{index}_value"
        coordinates.append(coordinate)
        nodes.append(
            onnx.helper.make_node(
                "Gather",
                [str(boxes.name), f"{prefix}_coordinate_{index}"],
                [coordinate],
                name=f"{prefix}_coordinate_{index}_node",
                axis=1,
            )
        )

    x_width = f"{prefix}_x_width_value"
    y_height = f"{prefix}_y_height_value"
    x_half = f"{prefix}_x_half"
    y_half = f"{prefix}_y_half"
    x_center_sum = f"{prefix}_x_center_sum_value"
    y_center_sum = f"{prefix}_y_center_sum_value"
    x_center = f"{prefix}_x_center"
    y_center = f"{prefix}_y_center"
    scaled_x_half = f"{prefix}_scaled_x_half"
    scaled_y_half = f"{prefix}_scaled_y_half"
    expanded = [f"{prefix}_expanded_{index}" for index in range(4)]
    nodes.extend(
        [
            onnx.helper.make_node(
                "Sub", [coordinates[2], coordinates[0]], [x_width], name=f"{prefix}_x_width"
            ),
            onnx.helper.make_node(
                "Sub", [coordinates[3], coordinates[1]], [y_height], name=f"{prefix}_y_height"
            ),
            onnx.helper.make_node(
                "Mul", [x_width, f"{prefix}_half"], [x_half], name=f"{prefix}_x_half_node"
            ),
            onnx.helper.make_node(
                "Mul", [y_height, f"{prefix}_half"], [y_half], name=f"{prefix}_y_half_node"
            ),
            onnx.helper.make_node(
                "Add", [coordinates[2], coordinates[0]], [x_center_sum], name=f"{prefix}_x_center_sum"
            ),
            onnx.helper.make_node(
                "Add", [coordinates[3], coordinates[1]], [y_center_sum], name=f"{prefix}_y_center_sum"
            ),
            onnx.helper.make_node(
                "Mul", [x_center_sum, f"{prefix}_half"], [x_center], name=f"{prefix}_x_center_node"
            ),
            onnx.helper.make_node(
                "Mul", [y_center_sum, f"{prefix}_half"], [y_center], name=f"{prefix}_y_center_node"
            ),
            onnx.helper.make_node(
                "Mul", [x_half, scale], [scaled_x_half], name=f"{prefix}_scale_x"
            ),
            onnx.helper.make_node(
                "Mul", [y_half, scale], [scaled_y_half], name=f"{prefix}_scale_y"
            ),
            onnx.helper.make_node(
                "Sub", [x_center, scaled_x_half], [expanded[0]], name=f"{prefix}_expanded_x0"
            ),
            onnx.helper.make_node(
                "Sub", [y_center, scaled_y_half], [expanded[1]], name=f"{prefix}_expanded_y0"
            ),
            onnx.helper.make_node(
                "Add", [x_center, scaled_x_half], [expanded[2]], name=f"{prefix}_expanded_x1"
            ),
            onnx.helper.make_node(
                "Add", [y_center, scaled_y_half], [expanded[3]], name=f"{prefix}_expanded_y1"
            ),
        ]
    )
    expanded_columns: List[str] = []
    for index, expanded_name in enumerate(expanded):
        column = f"{expanded_name}_column"
        expanded_columns.append(column)
        nodes.append(
            onnx.helper.make_node(
                "Unsqueeze",
                [expanded_name, f"{prefix}_unsqueeze_axis"],
                [column],
                name=f"{prefix}_expanded_{index}_unsqueeze",
            )
        )
    expanded_boxes = f"{prefix}_expanded_boxes"
    nodes.extend(
        [
            onnx.helper.make_node(
                "Concat",
                expanded_columns,
                [expanded_boxes],
                name=f"{prefix}_expanded_boxes_node",
                axis=1,
            ),
            onnx.helper.make_node(
                "Cast",
                [expanded_boxes],
                [missing_name],
                name=f"{prefix}_cast",
                to=TensorProto.INT64,
            ),
        ]
    )
    return nodes


def repair_missing_torchvision_paste_masks_loop_captures(
    model: onnx.ModelProto,
) -> Dict[str, int]:
    """Restore optimizer-pruned expanded-box captures in mask-pasting loops."""

    def visit(graph: onnx.GraphProto) -> int:
        nodes = list(graph.node)
        producers = {
            str(output_name): node
            for node in nodes
            for output_name in node.output
            if str(output_name) != ""
        }
        initializers = {
            str(initializer.name): initializer
            for initializer in graph.initializer
            if str(initializer.name) != ""
        }
        available = {
            str(value.name)
            for value in graph.input
            if str(value.name) != ""
        } | set(initializers) | set(producers)
        rewritten_nodes: List[onnx.NodeProto] = []
        repaired = 0
        for node in nodes:
            pattern = (
                _paste_masks_loop_pattern(
                    node,
                    graph=graph,
                    available=available,
                    producers=producers,
                    initializers=initializers,
                )
                if str(node.op_type) == "Loop"
                else None
            )
            if pattern is not None:
                missing_name, pad_node, boxes = pattern
                rewritten_nodes.extend(
                    _make_paste_masks_box_repair_nodes(
                        loop_node=node,
                        missing_name=missing_name,
                        pad_node=pad_node,
                        boxes=boxes,
                        graph=graph,
                    )
                )
                available.add(missing_name)
                repaired += 1
            rewritten_nodes.append(node)
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    repaired += visit(attribute.g)
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for child_graph in attribute.graphs:
                        repaired += visit(child_graph)
        if repaired > 0:
            del graph.node[:]
            graph.node.extend(rewritten_nodes)
        return repaired

    repaired = visit(model.graph)
    return {"TorchVisionPasteMasksLoopCaptures": repaired} if repaired else {}


def repair_missing_arange_loop_delta_captures(
    model: onnx.ModelProto,
) -> Dict[str, int]:
    """Restore a pruned default ``delta=1`` capture in tf2onnx arange loops.

    Some optimized tf2onnx graphs retain ``prev + .../arange.../delta:0`` in
    Loop bodies after pruning the scalar default step from the parent graph.
    Restrict the repair to the canonical carried-value and scan-output pattern;
    arbitrary missing Loop captures must remain invalid.
    """

    def _repair_graph_arange_delta(graph: onnx.GraphProto) -> int:
        available = {
            str(value.name)
            for value in graph.input
            if str(value.name) != ""
        } | {
            str(initializer.name)
            for initializer in graph.initializer
            if str(initializer.name) != ""
        } | {
            str(output_name)
            for node in graph.node
            for output_name in node.output
            if str(output_name) != ""
        }
        candidate_uses: Dict[str, List[Tuple[onnx.NodeProto, onnx.GraphProto]]] = {}
        child_graphs: List[onnx.GraphProto] = []
        for node in graph.node:
            for attribute in node.attribute:
                nested_graphs: List[onnx.GraphProto] = []
                if attribute.type == onnx.AttributeProto.GRAPH:
                    nested_graphs = [attribute.g]
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    nested_graphs = list(attribute.graphs)
                child_graphs.extend(nested_graphs)
                if str(node.op_type) != "Loop":
                    continue
                for body in nested_graphs:
                    local = {
                        str(value.name)
                        for value in body.input
                        if str(value.name) != ""
                    } | {
                        str(initializer.name)
                        for initializer in body.initializer
                        if str(initializer.name) != ""
                    } | {
                        str(output_name)
                        for body_node in body.node
                        for output_name in body_node.output
                        if str(output_name) != ""
                    }
                    captures = {
                        str(input_name)
                        for body_node in body.node
                        for input_name in body_node.input
                        if str(input_name) != "" and str(input_name) not in local
                    }
                    for capture_name in captures - available:
                        candidate_uses.setdefault(capture_name, []).append((node, body))

        repaired = 0
        for capture_name, uses in candidate_uses.items():
            normalized_name = str(capture_name).lower().replace("__", ":")
            if "arange" not in normalized_name or "delta" not in normalized_name:
                continue
            elem_types: set[int] = set()
            valid = True
            for loop_node, body in uses:
                if (
                    "arange" not in str(loop_node.name).lower()
                    or len(loop_node.input) < 3
                    or len(body.input) < 3
                    or len(body.output) < 3
                ):
                    valid = False
                    break
                carried_name = str(body.input[2].name)
                current_name = str(body.output[1].name)
                scan_name = str(body.output[2].name)
                matching_adds = [
                    body_node
                    for body_node in body.node
                    if str(body_node.op_type) == "Add"
                    and set(str(name) for name in body_node.input)
                    == {carried_name, capture_name}
                    and list(body_node.output) == [current_name]
                ]
                matching_scans = [
                    body_node
                    for body_node in body.node
                    if str(body_node.op_type) == "Identity"
                    and list(body_node.input) == [carried_name]
                    and list(body_node.output) == [scan_name]
                ]
                all_capture_users = [
                    body_node
                    for body_node in body.node
                    if capture_name in body_node.input
                ]
                if (
                    len(matching_adds) != 1
                    or len(matching_scans) != 1
                    or all_capture_users != matching_adds
                ):
                    valid = False
                    break
                elem_types.add(int(body.input[2].type.tensor_type.elem_type))
            if not valid or len(elem_types) != 1:
                continue
            try:
                np_dtype = onnx.helper.tensor_dtype_to_np_dtype(next(iter(elem_types)))
            except Exception:
                continue
            graph.initializer.append(
                numpy_helper.from_array(
                    np.asarray(1, dtype=np_dtype),
                    name=capture_name,
                )
            )
            available.add(capture_name)
            repaired += 1

        for child_graph in child_graphs:
            repaired += _repair_graph_arange_delta(child_graph)
        return repaired

    repaired = _repair_graph_arange_delta(model.graph)
    return {"ArangeLoopDeltaCaptures": repaired} if repaired else {}
