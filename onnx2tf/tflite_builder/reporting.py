from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx
from onnx import numpy_helper

from onnx2tf.tflite_builder.core.lowering_context import LoweringContext
from onnx2tf.tflite_builder.core.model_ir_utils import _build_tensor_producer_map
from onnx2tf.tflite_builder.core.node import NodeView as _NodeWrap
from onnx2tf.tflite_builder.core.onnx_analysis import (
    _extract_tensor_info,
    _infer_shapes_with_fallback,
)
from onnx2tf.tflite_builder.core.session import ConversionSession
from onnx2tf.tflite_builder.ir import ModelIR, normalize_onnx_shape
from onnx2tf.tflite_builder.op_registry import (
    NodeValidationError,
    get_custom_op_candidate_ops,
    get_supported_onnx_ops,
    resolve_node_dispatch,
)
from onnx2tf.tflite_builder.tensor_buffer_builder import tflite_dtype_from_numpy


def _build_tensor_consumer_map(model_ir: ModelIR) -> Dict[str, List[int]]:
    consumers: Dict[str, List[int]] = {}
    for op_idx, op in enumerate(model_ir.operators):
        for input_name in op.inputs:
            if input_name not in consumers:
                consumers[input_name] = []
            consumers[input_name].append(op_idx)
    return consumers


def _collect_schema_ops_for_range(
    *,
    opset_min: int = 13,
    opset_max: int = 18,
) -> List[str]:
    schema_ops = set()
    for schema in onnx.defs.get_all_schemas_with_history():
        if schema.domain != "":
            continue
        if int(schema.since_version) < int(opset_min) or int(schema.since_version) > int(opset_max):
            continue
        schema_ops.add(str(schema.name))
    return sorted(schema_ops)


def _build_schema_policy_matrix(
    *,
    schema_ops: set,
    supported_registry_ops: set,
    custom_candidate_ops: set,
) -> List[Dict[str, Any]]:
    matrix: List[Dict[str, Any]] = []
    for op in sorted(list(schema_ops)):
        if op in supported_registry_ops:
            policy = "builtin_supported"
        elif op in custom_candidate_ops:
            policy = "custom_candidate"
        else:
            policy = "explicit_error"
        matrix.append(
            {
                "onnx_op": str(op),
                "policy": str(policy),
            }
        )
    return matrix


def build_op_coverage_report(
    *,
    onnx_graph: onnx.ModelProto,
    output_file_name: str,
    opset_min: int = 13,
    opset_max: int = 18,
    conversion_error: Optional[str] = None,
    allow_custom_ops: bool = False,
    custom_op_allowlist: Optional[List[str]] = None,
    disable_group_convolution: bool = False,
    preprocess_report: Optional[Dict[str, Any]] = None,
    output_nms_with_argmax: bool = False,
    switch_nms_version: str = "v4",
) -> Dict[str, Any]:
    onnx_graph = _infer_shapes_with_fallback(onnx_graph)

    shape_map, dtype_map = _extract_tensor_info(onnx_graph)
    constants: Dict[str, np.ndarray] = {}
    for ini in onnx_graph.graph.initializer:
        constants[ini.name] = np.asarray(numpy_helper.to_array(ini))
    # Producer/consumer information is built once by ConversionSession below.
    tensor_consumer_count: Dict[str, int] = {}
    graph_output_names = [str(o.name) for o in onnx_graph.graph.output]

    model_ir = ModelIR(name=output_file_name)
    model_ir.metadata["tensor_lineage_events"] = []
    session = ConversionSession(
        onnx_model=onnx_graph,
        model_ir=model_ir,
        shape_map=shape_map,
        dtype_map=dtype_map,
        constants=constants,
    )
    ctx = LoweringContext(
        model_ir=model_ir,
        shape_map=shape_map,
        dtype_map=dtype_map,
        constants=constants,
        onnx_model=onnx_graph,
        allow_custom_ops=allow_custom_ops,
        custom_op_allowlist=custom_op_allowlist,
        disable_group_convolution=disable_group_convolution,
        tensor_consumer_count=tensor_consumer_count,
        graph_output_names=graph_output_names,
        output_nms_with_argmax=output_nms_with_argmax,
        switch_nms_version=switch_nms_version,
        session=session,
    )
    initializer_names = {ini.name for ini in onnx_graph.graph.initializer}
    for graph_input in onnx_graph.graph.input:
        if graph_input.name in initializer_names:
            continue
        ctx.ensure_tensor(graph_input.name)
        model_ir.inputs.append(graph_input.name)
    for name, value in constants.items():
        if name not in model_ir.tensors:
            ctx.add_const_tensor(name, value)

    node_reports: List[Dict[str, Any]] = []
    unsupported_nodes: List[Dict[str, Any]] = []
    custom_lowered_nodes: List[Dict[str, Any]] = []
    graph_unique_ops: set = set()
    for node in onnx_graph.graph.node:
        node_name = node.name if node.name else node.op_type
        graph_unique_ops.add(node.op_type)
        if node.op_type == "Constant":
            value_attr = None
            for attr in node.attribute:
                if attr.name == "value":
                    value_attr = attr
                    break
            if value_attr is not None and len(node.output) > 0:
                const_array = np.asarray(numpy_helper.to_array(value_attr.t))
                out_name = node.output[0]
                if out_name in model_ir.tensors:
                    t = model_ir.tensors[out_name]
                    t.data = const_array
                    t.dtype = tflite_dtype_from_numpy(const_array.dtype)
                    t.shape, t.shape_signature = normalize_onnx_shape(list(const_array.shape))
                    constants[out_name] = const_array
                else:
                    _added = ctx.add_const_tensor(out_name, const_array)
                    if _added != out_name:
                        model_ir.tensors[out_name] = model_ir.tensors.pop(_added)
                        model_ir.tensors[out_name].name = out_name
                        constants[out_name] = constants.pop(_added)
            node_reports.append(
                {
                    "node_name": node_name,
                    "onnx_op": "Constant",
                    "supported": True,
                    "reason_code": "handled_inline",
                    "message": "Constant node is handled inline in lowering pass.",
                }
            )
            continue

        wrapped = _NodeWrap(
            node,
            shape_map=ctx.shape_map,
            dtype_map=ctx.dtype_map,
        )
        try:
            resolution = resolve_node_dispatch(wrapped, ctx)
            dispatch_mode = str(resolution.dispatch_mode)
            reason_code = resolution.reason_code
            message = resolution.message
            node_reports.append(
                {
                    "node_name": node_name,
                    "onnx_op": node.op_type,
                    "supported": True,
                    "dispatch_mode": dispatch_mode,
                    "reason_code": reason_code,
                    "message": message,
                }
            )
            if dispatch_mode == "custom":
                custom_lowered_nodes.append(
                    {
                        "node_name": node_name,
                        "onnx_op": node.op_type,
                        "reason_code": reason_code,
                        "message": message,
                    }
                )
        except NodeValidationError as ve:
            issue = ve.to_dict()
            issue["supported"] = False
            issue["dispatch_mode"] = "unsupported"
            node_reports.append(issue)
            unsupported_nodes.append(issue)
        except Exception as ex:
            issue = {
                "node_name": node_name,
                "onnx_op": node.op_type,
                "supported": False,
                "dispatch_mode": "unsupported",
                "reason_code": "validation_exception",
                "message": str(ex),
            }
            node_reports.append(issue)
            unsupported_nodes.append(issue)

    supported_registry_ops = set(get_supported_onnx_ops())
    custom_candidate_ops = set(get_custom_op_candidate_ops())
    schema_ops = set(_collect_schema_ops_for_range(opset_min=opset_min, opset_max=opset_max))
    schema_policy_matrix = _build_schema_policy_matrix(
        schema_ops=schema_ops,
        supported_registry_ops=supported_registry_ops,
        custom_candidate_ops=custom_candidate_ops,
    )
    schema_policy_counts: Dict[str, int] = {
        "builtin_supported": 0,
        "custom_candidate": 0,
        "explicit_error": 0,
    }
    for item in schema_policy_matrix:
        p = str(item["policy"])
        schema_policy_counts[p] = int(schema_policy_counts.get(p, 0) + 1)
    graph_ops = sorted(graph_unique_ops)
    supported_graph_ops = sorted(
        list({r["onnx_op"] for r in node_reports if r["supported"] is True})
    )
    unsupported_graph_ops = sorted(
        list({r["onnx_op"] for r in node_reports if r["supported"] is False})
    )
    reason_counts: Dict[str, int] = {}
    for issue in unsupported_nodes:
        reason = str(issue.get("reason_code", "unknown"))
        reason_counts[reason] = int(reason_counts.get(reason, 0) + 1)

    normalized_allowlist = (
        [str(v).strip() for v in custom_op_allowlist if str(v).strip() != ""]
        if custom_op_allowlist is not None
        else None
    )
    allowlist_set = (
        {str(v) for v in normalized_allowlist}
        if normalized_allowlist is not None
        else set()
    )
    candidate_ops_now_builtin_supported = sorted(
        list(custom_candidate_ops & supported_registry_ops)
    )
    allowlist_builtin_supported_ops = sorted(
        list(allowlist_set & supported_registry_ops)
    )
    allowlist_custom_candidate_ops = sorted(
        list(allowlist_set & custom_candidate_ops)
    )
    allowlist_unknown_ops = sorted(
        list(
            allowlist_set
            - supported_registry_ops
            - custom_candidate_ops
            - schema_ops
        )
    )

    total_nodes = len(node_reports)
    supported_nodes = len([r for r in node_reports if r["supported"] is True])
    coverage = float(supported_nodes / total_nodes) if total_nodes > 0 else 1.0
    report: Dict[str, Any] = {
        "schema_version": 1,
        "target_opset_min": int(opset_min),
        "target_opset_max": int(opset_max),
        "supported_onnx_ops_registry": sorted(list(supported_registry_ops)),
        "custom_op_candidate_ops": sorted(list(custom_candidate_ops)),
        "schema_onnx_ops_target_range": sorted(list(schema_ops)),
        "schema_policy_matrix": schema_policy_matrix,
        "schema_policy_counts": schema_policy_counts,
        "schema_unresolved_ops": [],
        "registry_missing_from_schema_range": sorted(list(schema_ops - supported_registry_ops)),
        "registry_extra_outside_schema_range": sorted(list(supported_registry_ops - schema_ops)),
        "graph_ops": graph_ops,
        "graph_supported_ops": supported_graph_ops,
        "graph_unsupported_ops": unsupported_graph_ops,
        "graph_custom_ops": sorted(list({r["onnx_op"] for r in custom_lowered_nodes})),
        "graph_node_reports": node_reports,
        "custom_lowered_nodes": custom_lowered_nodes,
        "unsupported_nodes": unsupported_nodes,
        "unsupported_reason_counts": reason_counts,
        "graph_summary": {
            "total_nodes": int(total_nodes),
            "supported_nodes": int(supported_nodes),
            "custom_lowered_nodes": int(len(custom_lowered_nodes)),
            "unsupported_nodes": int(total_nodes - supported_nodes),
            "coverage_ratio": float(coverage),
        },
        "custom_op_policy": {
            "allow_custom_ops": bool(allow_custom_ops),
            "custom_op_allowlist": (
                [str(v) for v in normalized_allowlist]
                if normalized_allowlist is not None
                else None
            ),
            "candidate_count": int(len(custom_candidate_ops)),
            "candidate_count_excluding_builtin_supported": int(
                len(custom_candidate_ops - supported_registry_ops)
            ),
            "candidate_ops_now_builtin_supported": candidate_ops_now_builtin_supported,
            "allowlist_builtin_supported_ops": allowlist_builtin_supported_ops,
            "allowlist_custom_candidate_ops": allowlist_custom_candidate_ops,
            "allowlist_unknown_ops": allowlist_unknown_ops,
        },
        "preprocess_report": (
            dict(preprocess_report)
            if isinstance(preprocess_report, dict)
            else {
                "schema_version": 1,
                "pipeline_version": 1,
                "registered_rule_ids": [],
                "enabled_rule_ids": [],
                "applied_rules": [],
                "summary": {
                    "registered_rule_count": 0,
                    "enabled_rule_count": 0,
                    "executed_rule_count": 0,
                    "changed_rule_count": 0,
                    "total_matched_nodes": 0,
                    "total_rewritten_nodes": 0,
                },
            }
        ),
        "conversion_error": conversion_error,
    }
    return report


def write_op_coverage_report(
    *,
    report: Dict[str, Any],
    output_report_path: str,
) -> str:
    import json
    import os

    os.makedirs(os.path.dirname(output_report_path) or ".", exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return output_report_path


def _trace_tensor_rewrite_history(
    *,
    original_name: str,
    lineage_events: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]], List[str]]:
    current_name = str(original_name)
    applied_events: List[Dict[str, Any]] = []
    name_chain: List[str] = [str(original_name)]
    for event in lineage_events:
        kind = str(event.get("kind", ""))
        if kind == "rename_tensor":
            old_name = str(event.get("old_name", ""))
            new_name = str(event.get("new_name", ""))
            if current_name == old_name and old_name != "" and new_name != "":
                current_name = new_name
                applied_events.append(
                    {
                        "event_index": int(event.get("event_index", -1)),
                        "kind": kind,
                        "from": old_name,
                        "to": new_name,
                    }
                )
                name_chain.append(current_name)
        elif kind == "replace_input":
            src_name = str(event.get("src_name", ""))
            dst_name = str(event.get("dst_name", ""))
            if current_name == src_name and src_name != "" and dst_name != "":
                current_name = dst_name
                applied_events.append(
                    {
                        "event_index": int(event.get("event_index", -1)),
                        "kind": kind,
                        "from": src_name,
                        "to": dst_name,
                    }
                )
                name_chain.append(current_name)
    return current_name, applied_events, name_chain


def _build_onnx_tensor_consumer_graph(
    onnx_graph: onnx.ModelProto,
) -> Tuple[Dict[str, List[int]], List[List[str]], List[str]]:
    tensor_consumers: Dict[str, List[int]] = {}
    node_outputs: List[List[str]] = []
    node_op_types: List[str] = []
    for node_index, node in enumerate(onnx_graph.graph.node):
        outputs = [str(v) for v in node.output if str(v) != ""]
        node_outputs.append(outputs)
        node_op_types.append(str(node.op_type))
        for input_name in node.input:
            input_name = str(input_name)
            if input_name == "":
                continue
            if input_name not in tensor_consumers:
                tensor_consumers[input_name] = []
            tensor_consumers[input_name].append(int(node_index))
    return tensor_consumers, node_outputs, node_op_types


def _infer_correspondence_via_downstream(
    *,
    records: List[Dict[str, Any]],
    onnx_graph: onnx.ModelProto,
) -> None:
    record_by_output = {
        str(record.get("onnx_output_name", "")): record
        for record in records
        if str(record.get("onnx_output_name", "")) != ""
    }
    tensor_consumers, node_outputs, node_op_types = _build_onnx_tensor_consumer_graph(onnx_graph)
    resolvable_statuses = {"direct", "rewritten", "inferred"}
    traceable_ops = {
        "Transpose",
        "Reshape",
        "QuantizeLinear",
        "DequantizeLinear",
        "Identity",
        "Cast",
        "Squeeze",
        "Unsqueeze",
        "Flatten",
    }
    max_infer_hops = 4

    changed = True
    while changed:
        changed = False
        for record in records:
            if str(record.get("status", "")) != "missing":
                continue
            start_name = str(record.get("onnx_output_name", ""))
            if start_name == "":
                continue

            queue: List[Tuple[str, int]] = [(start_name, 0)]
            visited: set = {start_name}
            inferred_target: Optional[Dict[str, Any]] = None
            inferred_from_output: Optional[str] = None
            inferred_hops: Optional[int] = None

            while len(queue) > 0 and inferred_target is None:
                tensor_name, hops = queue.pop(0)
                if int(hops) >= int(max_infer_hops):
                    continue
                consumer_indices = tensor_consumers.get(str(tensor_name), [])
                for consumer_index in consumer_indices:
                    if int(consumer_index) < 0 or int(consumer_index) >= len(node_outputs):
                        continue
                    if (
                        int(consumer_index) >= len(node_op_types)
                        or str(node_op_types[int(consumer_index)]) not in traceable_ops
                    ):
                        continue
                    for out_name in node_outputs[int(consumer_index)]:
                        out_name = str(out_name)
                        if out_name == "" or out_name in visited:
                            continue
                        visited.add(out_name)
                        downstream_record = record_by_output.get(out_name, None)
                        if (
                            downstream_record is not None
                            and str(downstream_record.get("status", "")) in resolvable_statuses
                            and bool(downstream_record.get("exists_in_final_model", False))
                        ):
                            inferred_target = downstream_record
                            inferred_from_output = out_name
                            inferred_hops = int(hops + 1)
                            break
                        queue.append((out_name, int(hops + 1)))
                    if inferred_target is not None:
                        break

            if inferred_target is None:
                continue

            resolved_name = str(inferred_target.get("resolved_tflite_tensor_name", ""))
            if resolved_name == "":
                continue

            record["status"] = "inferred"
            record["is_rewritten"] = True
            record["resolved_tflite_tensor_name"] = resolved_name
            record["exists_in_final_model"] = bool(
                inferred_target.get("exists_in_final_model", False)
            )
            record["producer_operator_index"] = inferred_target.get(
                "producer_operator_index", None
            )
            record["producer_op_type"] = inferred_target.get("producer_op_type", None)
            record["consumer_count"] = inferred_target.get("consumer_count", 0)
            record["is_graph_output"] = bool(inferred_target.get("is_graph_output", False))
            record["inferred_from_onnx_output_name"] = str(inferred_from_output)
            record["inferred_hops"] = int(inferred_hops) if inferred_hops is not None else None
            chain = record.get("rewrite_name_chain", [])
            if not isinstance(chain, list):
                chain = [str(start_name)]
            chain = [str(v) for v in chain]
            if resolved_name not in chain:
                chain.append(resolved_name)
            record["rewrite_name_chain"] = chain
            rewrite_events = record.get("rewrite_events", [])
            if not isinstance(rewrite_events, list):
                rewrite_events = []
            rewrite_events.append(
                {
                    "event_index": None,
                    "kind": "inferred_via_downstream",
                    "from": str(start_name),
                    "to": str(resolved_name),
                    "via_onnx_output": str(inferred_from_output),
                    "hops": int(inferred_hops) if inferred_hops is not None else None,
                }
            )
            record["rewrite_events"] = rewrite_events
            changed = True


def build_tensor_correspondence_report(
    *,
    onnx_graph: onnx.ModelProto,
    model_ir: ModelIR,
) -> Dict[str, Any]:
    lineage_events = model_ir.metadata.get("tensor_lineage_events", [])
    if not isinstance(lineage_events, list):
        lineage_events = []
    lineage_events = [
        dict(event) for event in lineage_events if isinstance(event, dict)
    ]

    producers = _build_tensor_producer_map(model_ir)
    consumers = _build_tensor_consumer_map(model_ir)
    model_output_set = set(str(v) for v in model_ir.outputs)
    model_tensor_set = set(str(v) for v in model_ir.tensors.keys())
    channel_last_hint_names = {
        str(v)
        for v in model_ir.metadata.get("assume_channel_last_layout_tensor_names", [])
        if str(v) != ""
    }

    def _is_channel_last_layout_name(tensor_name: str) -> bool:
        normalized_name = str(tensor_name).split(":")[0].lower()
        if normalized_name == "":
            return False
        return (
            "_nwc" in normalized_name
            or "_nhwc" in normalized_name
            or "_ndhwc" in normalized_name
        )

    records: List[Dict[str, Any]] = []
    total_onnx_outputs = 0

    for identifier, node in enumerate(onnx_graph.graph.node, start=1):
        node_name = str(node.name)
        node_op = str(node.op_type)
        for output_index, output_name in enumerate(node.output):
            output_name = str(output_name)
            if output_name == "":
                continue
            total_onnx_outputs += 1

            resolved_name, applied_events, name_chain = _trace_tensor_rewrite_history(
                original_name=output_name,
                lineage_events=lineage_events,
            )
            exists_in_final_model = resolved_name in model_tensor_set
            is_rewritten = str(resolved_name) != str(output_name)

            producer_index = producers.get(resolved_name, None)
            producer_op_type = None
            if producer_index is not None:
                producer_op_type = str(model_ir.operators[int(producer_index)].op_type)
            consumer_count = int(len(consumers.get(resolved_name, [])))
            is_graph_output = bool(resolved_name in model_output_set)

            if not is_rewritten and exists_in_final_model:
                status = "direct"
            elif is_rewritten and exists_in_final_model:
                status = "rewritten"
            elif is_rewritten and not exists_in_final_model:
                status = "rewritten_but_missing"
            else:
                status = "missing"

            records.append(
                {
                    "identifier": int(identifier),
                    "onnx_op_type": node_op,
                    "onnx_op_name": node_name,
                    "onnx_output_index": int(output_index),
                    "onnx_output_name": output_name,
                    "resolved_tflite_tensor_name": str(resolved_name),
                    "exists_in_final_model": bool(exists_in_final_model),
                    "is_graph_output": bool(is_graph_output),
                    "consumer_count": int(consumer_count),
                    "producer_operator_index": (
                        int(producer_index)
                        if producer_index is not None
                        else None
                    ),
                    "producer_op_type": producer_op_type,
                    "assume_channel_last_layout": bool(
                        resolved_name in channel_last_hint_names
                        or _is_channel_last_layout_name(resolved_name)
                    ),
                    "status": status,
                    "is_rewritten": bool(is_rewritten),
                    "rewrite_events": applied_events,
                    "rewrite_name_chain": name_chain,
                }
            )

    for record in records:
        if str(record.get("status", "")) != "missing":
            continue
        if str(record.get("onnx_op_type", "")) != "QLinearConv":
            continue
        onnx_op_name = str(record.get("onnx_op_name", ""))
        if onnx_op_name == "":
            continue
        candidate_name = f"{onnx_op_name}_output_nhwc"
        if candidate_name not in model_tensor_set:
            continue

        producer_index = producers.get(candidate_name, None)
        producer_op_type = None
        if producer_index is not None:
            producer_op_type = str(model_ir.operators[int(producer_index)].op_type)
        consumer_count = int(len(consumers.get(candidate_name, [])))
        is_graph_output = bool(candidate_name in model_output_set)

        record["status"] = "inferred"
        record["is_rewritten"] = True
        record["resolved_tflite_tensor_name"] = str(candidate_name)
        record["exists_in_final_model"] = True
        record["producer_operator_index"] = (
            int(producer_index) if producer_index is not None else None
        )
        record["producer_op_type"] = producer_op_type
        record["consumer_count"] = int(consumer_count)
        record["is_graph_output"] = bool(is_graph_output)
        record["assume_channel_last_layout"] = bool(
            candidate_name in channel_last_hint_names
            or _is_channel_last_layout_name(candidate_name)
        )
        chain = record.get("rewrite_name_chain", [])
        if not isinstance(chain, list):
            chain = [str(record.get("onnx_output_name", ""))]
        chain = [str(v) for v in chain]
        if candidate_name not in chain:
            chain.append(str(candidate_name))
        record["rewrite_name_chain"] = chain
        rewrite_events = record.get("rewrite_events", [])
        if not isinstance(rewrite_events, list):
            rewrite_events = []
        rewrite_events.append(
            {
                "event_index": None,
                "kind": "inferred_via_qlinearconv_output_nhwc",
                "from": str(record.get("onnx_output_name", "")),
                "to": str(candidate_name),
            }
        )
        record["rewrite_events"] = rewrite_events

    _infer_correspondence_via_downstream(
        records=records,
        onnx_graph=onnx_graph,
    )

    direct_count = int(sum(1 for r in records if str(r.get("status", "")) == "direct"))
    rewritten_count = int(sum(1 for r in records if str(r.get("status", "")) == "rewritten"))
    inferred_count = int(sum(1 for r in records if str(r.get("status", "")) == "inferred"))
    rewritten_and_missing_count = int(
        sum(1 for r in records if str(r.get("status", "")) == "rewritten_but_missing")
    )
    missing_count = int(
        sum(
            1
            for r in records
            if str(r.get("status", "")) in {"missing", "rewritten_but_missing"}
        )
    )

    report = {
        "schema_version": 1,
        "summary": {
            "total_onnx_node_outputs": int(total_onnx_outputs),
            "direct_count": int(direct_count),
            "rewritten_count": int(rewritten_count),
            "inferred_count": int(inferred_count),
            "missing_count": int(missing_count),
            "rewritten_but_missing_count": int(rewritten_and_missing_count),
            "lineage_event_count": int(len(lineage_events)),
        },
        "lineage_events": lineage_events,
        "records": records,
    }
    return report


def write_tensor_correspondence_report(
    *,
    report: Dict[str, Any],
    output_report_path: str,
) -> str:
    import json
    import os

    os.makedirs(os.path.dirname(output_report_path) or ".", exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return output_report_path
