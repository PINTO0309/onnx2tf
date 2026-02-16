from __future__ import annotations

from typing import Any, Dict, List, Set

import onnx

from onnx2tf.tflite_builder.preprocess.pipeline import register_preprocess_rule

CLEANUP_UNUSED_INITIALIZERS_RULE_ID = "cleanup_unused_initializers_z9"


def prune_unused_initializers_inplace(
    onnx_graph: onnx.ModelProto,
    *,
    prune_orphan_value_info: bool = True,
) -> Dict[str, Any]:
    graph = onnx_graph.graph

    used_input_names: Set[str] = set()
    for node in graph.node:
        for name in node.input:
            if name != "":
                used_input_names.add(str(name))

    graph_output_names = {str(v.name) for v in graph.output if str(v.name) != ""}

    original_initializer_count = int(len(graph.initializer))
    kept_initializers: List[onnx.TensorProto] = []
    removed_initializer_names: Set[str] = set()
    for initializer in graph.initializer:
        name = str(initializer.name)
        if name == "":
            kept_initializers.append(initializer)
            continue
        if name in used_input_names or name in graph_output_names:
            kept_initializers.append(initializer)
            continue
        removed_initializer_names.add(name)

    removed_count = int(len(removed_initializer_names))
    if removed_count > 0:
        del graph.initializer[:]
        graph.initializer.extend(kept_initializers)

    removed_value_info_count = 0
    if prune_orphan_value_info and removed_count > 0:
        active_tensor_names: Set[str] = set()
        for node in graph.node:
            for name in list(node.input) + list(node.output):
                if name != "":
                    active_tensor_names.add(str(name))
        for value in list(graph.input) + list(graph.output):
            if value.name != "":
                active_tensor_names.add(str(value.name))
        for initializer in graph.initializer:
            if initializer.name != "":
                active_tensor_names.add(str(initializer.name))

        kept_value_infos: List[onnx.ValueInfoProto] = []
        for value_info in graph.value_info:
            value_name = str(value_info.name)
            if value_name in active_tensor_names:
                kept_value_infos.append(value_info)
                continue
            removed_value_info_count += 1
        if removed_value_info_count > 0:
            del graph.value_info[:]
            graph.value_info.extend(kept_value_infos)

    return {
        "removed_initializer_count": removed_count,
        "removed_value_info_count": int(removed_value_info_count),
        "original_initializer_count": int(original_initializer_count),
        "changed": bool(removed_count > 0 or removed_value_info_count > 0),
    }


def apply_cleanup_unused_initializers(onnx_graph: onnx.ModelProto) -> Dict[str, Any]:
    stats = prune_unused_initializers_inplace(
        onnx_graph,
        prune_orphan_value_info=True,
    )
    removed_initializer_count = int(stats.get("removed_initializer_count", 0))
    removed_value_info_count = int(stats.get("removed_value_info_count", 0))
    return {
        "matched_nodes": int(removed_initializer_count),
        "rewritten_nodes": int(removed_initializer_count),
        "changed": bool(stats.get("changed", False)),
        "message": (
            f"removed_initializers={removed_initializer_count} "
            f"removed_value_info={removed_value_info_count}"
        ),
    }


def register_cleanup_unused_initializers_rule() -> None:
    register_preprocess_rule(
        rule_id=CLEANUP_UNUSED_INITIALIZERS_RULE_ID,
        callback=apply_cleanup_unused_initializers,
        overwrite=True,
    )
