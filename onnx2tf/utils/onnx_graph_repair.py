from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx
from onnx import numpy_helper


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
