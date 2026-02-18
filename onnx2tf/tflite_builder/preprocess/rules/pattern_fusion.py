from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnx
from onnx import helper, numpy_helper

from onnx2tf.tflite_builder.preprocess.pipeline import register_preprocess_rule

PATTERN_FUSION_WAVE2_RULE_ID = "pattern_fusion_wave2"


def _copy_node(node: onnx.NodeProto) -> onnx.NodeProto:
    cloned = onnx.NodeProto()
    cloned.CopyFrom(node)
    return cloned


def _collect_used_names(graph: onnx.GraphProto) -> set[str]:
    used = set()
    for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
        if vi.name != "":
            used.add(str(vi.name))
    for init in graph.initializer:
        if init.name != "":
            used.add(str(init.name))
    for node in graph.node:
        if node.name != "":
            used.add(str(node.name))
        for name in list(node.input) + list(node.output):
            if name != "":
                used.add(str(name))
    return used


def _next_name(used_names: set[str], base: str) -> str:
    candidate = str(base) if str(base) != "" else "tmp"
    if candidate not in used_names:
        used_names.add(candidate)
        return candidate
    i = 1
    while True:
        c = f"{candidate}_{i}"
        if c not in used_names:
            used_names.add(c)
            return c
        i += 1


def _make_node(
    *,
    used_names: set[str],
    op_type: str,
    inputs: Sequence[str],
    outputs: Sequence[str],
    name_base: str,
    attrs: Optional[Dict[str, Any]] = None,
) -> onnx.NodeProto:
    attrs = attrs or {}
    node_name = _next_name(used_names, f"{name_base}_{op_type.lower()}")
    return helper.make_node(
        op_type,
        [str(v) for v in inputs],
        [str(v) for v in outputs],
        name=node_name,
        **attrs,
    )


def _find_const_tensor_attribute(node: onnx.NodeProto, name: str) -> Optional[onnx.TensorProto]:
    for attr in node.attribute:
        if str(attr.name) != str(name):
            continue
        if attr.type == onnx.AttributeProto.TENSOR:
            return attr.t
    return None


def _build_initializer_map(graph: onnx.GraphProto) -> Dict[str, np.ndarray]:
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


def _build_tensor_consumers(nodes: List[onnx.NodeProto]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for node in nodes:
        for name in node.input:
            if name == "":
                continue
            counts[str(name)] = int(counts.get(str(name), 0) + 1)
    return counts


def _graph_output_names(graph: onnx.GraphProto) -> set[str]:
    return {str(v.name) for v in graph.output if str(v.name) != ""}


def _resolve_clip_bounds(
    *,
    node: onnx.NodeProto,
    const_map: Dict[str, np.ndarray],
) -> Tuple[float, float]:
    min_value = float("-inf")
    max_value = float("inf")
    for attr in node.attribute:
        if str(attr.name) == "min" and attr.type in [onnx.AttributeProto.FLOAT, onnx.AttributeProto.INT]:
            min_value = float(attr.f if attr.type == onnx.AttributeProto.FLOAT else attr.i)
        if str(attr.name) == "max" and attr.type in [onnx.AttributeProto.FLOAT, onnx.AttributeProto.INT]:
            max_value = float(attr.f if attr.type == onnx.AttributeProto.FLOAT else attr.i)
    if len(node.input) >= 2 and str(node.input[1]) in const_map:
        values = np.asarray(const_map[str(node.input[1])]).reshape(-1)
        if values.size > 0:
            min_value = float(values[0])
    if len(node.input) >= 3 and str(node.input[2]) in const_map:
        values = np.asarray(const_map[str(node.input[2])]).reshape(-1)
        if values.size > 0:
            max_value = float(values[0])
    return min_value, max_value


def _get_const_scalar(
    *,
    tensor_name: str,
    const_map: Dict[str, np.ndarray],
) -> Optional[float]:
    if str(tensor_name) not in const_map:
        return None
    values = np.asarray(const_map[str(tensor_name)]).reshape(-1)
    if values.size == 0:
        return None
    if not np.allclose(values, values[0], rtol=0.0, atol=0.0):
        return None
    return float(values[0])


def _get_transpose_perm(
    *,
    node: onnx.NodeProto,
    const_map: Dict[str, np.ndarray],
) -> Optional[List[int]]:
    if len(node.input) >= 2 and str(node.input[1]) in const_map:
        arr = np.asarray(const_map[str(node.input[1])]).reshape(-1)
        return [int(v) for v in arr.tolist()]
    for attr in node.attribute:
        if str(attr.name) == "perm":
            if attr.type == onnx.AttributeProto.INTS:
                return [int(v) for v in attr.ints]
    return None


def _get_const_shape_vector(
    *,
    node: onnx.NodeProto,
    input_index: int,
    const_map: Dict[str, np.ndarray],
) -> Optional[List[int]]:
    if len(node.input) <= int(input_index):
        return None
    tensor_name = str(node.input[int(input_index)])
    if tensor_name not in const_map:
        return None
    arr = np.asarray(const_map[tensor_name]).reshape(-1)
    if arr.size == 0:
        return []
    return [int(v) for v in arr.tolist()]


def _fuse_relu_clip_relu6(
    *,
    nodes: List[onnx.NodeProto],
    used_names: set[str],
    const_map: Dict[str, np.ndarray],
    graph_outputs: set[str],
) -> Tuple[List[onnx.NodeProto], int, int]:
    consumer_counts = _build_tensor_consumers(nodes)
    fused_nodes: List[onnx.NodeProto] = []
    matched = 0
    rewritten = 0
    i = 0
    while i < len(nodes):
        if i + 1 >= len(nodes):
            fused_nodes.append(_copy_node(nodes[i]))
            i += 1
            continue
        relu = nodes[i]
        clip = nodes[i + 1]
        if (
            str(relu.op_type) == "Relu"
            and str(clip.op_type) == "Clip"
            and len(relu.output) >= 1
            and len(clip.input) >= 1
            and str(clip.input[0]) == str(relu.output[0])
        ):
            matched += 1
            min_v, max_v = _resolve_clip_bounds(node=clip, const_map=const_map)
            is_relu6 = abs(float(min_v) - 0.0) <= 1e-6 and abs(float(max_v) - 6.0) <= 1e-6
            if not is_relu6:
                fused_nodes.append(_copy_node(relu))
                i += 1
                continue

            relu_in = str(relu.input[0])
            clip_out = str(clip.output[0])
            relu_out = str(relu.output[0])
            can_drop_relu = (
                int(consumer_counts.get(relu_out, 0)) <= 1
                and relu_out not in graph_outputs
            )
            if not can_drop_relu:
                fused_nodes.append(_copy_node(relu))
            fused_nodes.append(
                _make_node(
                    used_names=used_names,
                    op_type="Clip",
                    inputs=[relu_in],
                    outputs=[clip_out],
                    name_base=clip.name or "relu6_fused",
                    attrs={"min": 0.0, "max": 6.0},
                )
            )
            rewritten += 1
            i += 2
            continue

        fused_nodes.append(_copy_node(nodes[i]))
        i += 1
    return fused_nodes, matched, rewritten


def _fuse_gelu_erf_chain(
    *,
    nodes: List[onnx.NodeProto],
    used_names: set[str],
    const_map: Dict[str, np.ndarray],
    graph_outputs: set[str],
) -> Tuple[List[onnx.NodeProto], int, int]:
    consumer_counts = _build_tensor_consumers(nodes)
    consumer_indices: Dict[str, List[int]] = {}
    for node_idx, node in enumerate(nodes):
        for input_name in node.input:
            if input_name == "":
                continue
            key = str(input_name)
            if key not in consumer_indices:
                consumer_indices[key] = []
            consumer_indices[key].append(int(node_idx))

    def _single_consumer_index(
        *,
        tensor_name: str,
        expected_op_type: str,
    ) -> Optional[int]:
        users = consumer_indices.get(str(tensor_name), [])
        if len(users) != 1:
            return None
        user_idx = int(users[0])
        if str(nodes[user_idx].op_type) != str(expected_op_type):
            return None
        return user_idx

    replace_at_index: Dict[int, onnx.NodeProto] = {}
    remove_indices: set[int] = set()
    matched = 0
    rewritten = 0
    for div_idx, div_node in enumerate(nodes):
        if div_idx in remove_indices:
            continue
        if str(div_node.op_type) != "Div":
            continue
        if len(div_node.input) < 2 or len(div_node.output) < 1:
            continue

        x_name = str(div_node.input[0])
        div_out = str(div_node.output[0])
        erf_idx = _single_consumer_index(
            tensor_name=div_out,
            expected_op_type="Erf",
        )
        if erf_idx is None or erf_idx in remove_indices:
            continue
        erf_node = nodes[int(erf_idx)]
        if len(erf_node.input) < 1 or len(erf_node.output) < 1:
            continue
        if str(erf_node.input[0]) != div_out:
            continue

        erf_out = str(erf_node.output[0])
        add_idx = _single_consumer_index(
            tensor_name=erf_out,
            expected_op_type="Add",
        )
        if add_idx is None or add_idx in remove_indices:
            continue
        add_node = nodes[int(add_idx)]
        if len(add_node.input) < 2 or len(add_node.output) < 1:
            continue

        add_inputs = [str(v) for v in add_node.input]
        if erf_out not in add_inputs:
            continue
        add_other = add_inputs[0] if add_inputs[1] == erf_out else add_inputs[1]
        add_const = _get_const_scalar(tensor_name=add_other, const_map=const_map)
        if add_const is None or abs(add_const - 1.0) > 1e-5:
            continue

        add_out = str(add_node.output[0])
        mul1_idx = _single_consumer_index(
            tensor_name=add_out,
            expected_op_type="Mul",
        )
        if mul1_idx is None or mul1_idx in remove_indices:
            continue
        mul1_node = nodes[int(mul1_idx)]
        if len(mul1_node.input) < 2 or len(mul1_node.output) < 1:
            continue
        mul1_inputs = [str(v) for v in mul1_node.input]
        if x_name not in mul1_inputs or add_out not in mul1_inputs:
            continue

        mul1_out = str(mul1_node.output[0])
        mul2_idx = _single_consumer_index(
            tensor_name=mul1_out,
            expected_op_type="Mul",
        )
        if mul2_idx is None or mul2_idx in remove_indices:
            continue
        mul2_node = nodes[int(mul2_idx)]
        if len(mul2_node.input) < 2 or len(mul2_node.output) < 1:
            continue
        mul2_inputs = [str(v) for v in mul2_node.input]
        if mul1_out not in mul2_inputs:
            continue
        mul2_other = mul2_inputs[0] if mul2_inputs[1] == mul1_out else mul2_inputs[1]
        mul2_const = _get_const_scalar(tensor_name=mul2_other, const_map=const_map)
        if mul2_const is None or abs(mul2_const - 0.5) > 1e-5:
            continue

        sqrt2_const = _get_const_scalar(
            tensor_name=str(div_node.input[1]),
            const_map=const_map,
        )
        if sqrt2_const is None or abs(sqrt2_const - float(np.sqrt(2.0))) > 1e-4:
            continue

        matched += 1
        for intermediate in [div_out, erf_out, add_out, mul1_out]:
            if intermediate in graph_outputs or int(consumer_counts.get(intermediate, 0)) > 1:
                raise ValueError(
                    "pattern_fusion_invalid_rewrite "
                    "reason_code=gelu_chain_fanout_conflict "
                    f"tensor={intermediate}"
                )

        matched_indices = {
            int(div_idx),
            int(erf_idx),
            int(add_idx),
            int(mul1_idx),
            int(mul2_idx),
        }
        if any(idx in remove_indices for idx in matched_indices):
            continue

        replace_at_index[int(mul2_idx)] = _make_node(
            used_names=used_names,
            op_type="Gelu",
            inputs=[x_name],
            outputs=[str(mul2_node.output[0])],
            name_base=mul2_node.name or "gelu_fused",
            attrs={"approximate": "none"},
        )
        remove_indices.update(matched_indices)
        rewritten += 1

    fused_nodes: List[onnx.NodeProto] = []
    for node_idx, node in enumerate(nodes):
        if node_idx in replace_at_index:
            fused_nodes.append(replace_at_index[node_idx])
            continue
        if node_idx in remove_indices:
            continue
        fused_nodes.append(_copy_node(node))
    return fused_nodes, matched, rewritten


def _fuse_space_to_depth_chain(
    *,
    nodes: List[onnx.NodeProto],
    used_names: set[str],
    const_map: Dict[str, np.ndarray],
    graph_outputs: set[str],
) -> Tuple[List[onnx.NodeProto], int, int]:
    consumer_counts = _build_tensor_consumers(nodes)
    fused_nodes: List[onnx.NodeProto] = []
    matched = 0
    rewritten = 0
    i = 0
    while i < len(nodes):
        if i + 2 >= len(nodes):
            fused_nodes.append(_copy_node(nodes[i]))
            i += 1
            continue
        n0 = nodes[i]
        n1 = nodes[i + 1]
        n2 = nodes[i + 2]
        if [str(n0.op_type), str(n1.op_type), str(n2.op_type)] != ["Reshape", "Transpose", "Reshape"]:
            fused_nodes.append(_copy_node(nodes[i]))
            i += 1
            continue
        if (
            len(n0.input) < 2
            or len(n1.input) < 1
            or len(n2.input) < 2
            or len(n0.output) < 1
            or len(n1.output) < 1
            or len(n2.output) < 1
            or str(n1.input[0]) != str(n0.output[0])
            or str(n2.input[0]) != str(n1.output[0])
        ):
            fused_nodes.append(_copy_node(nodes[i]))
            i += 1
            continue

        shape1 = _get_const_shape_vector(node=n0, input_index=1, const_map=const_map)
        shape2 = _get_const_shape_vector(node=n2, input_index=1, const_map=const_map)
        perm = _get_transpose_perm(node=n1, const_map=const_map)
        if shape1 is None or shape2 is None or perm is None:
            fused_nodes.append(_copy_node(nodes[i]))
            i += 1
            continue
        if len(shape1) != 6 or len(shape2) != 4 or len(perm) != 6:
            fused_nodes.append(_copy_node(nodes[i]))
            i += 1
            continue
        if [int(v) for v in perm] != [0, 1, 3, 5, 2, 4]:
            fused_nodes.append(_copy_node(nodes[i]))
            i += 1
            continue

        matched += 1
        block_size = int(shape1[3])
        if block_size <= 1:
            raise ValueError(
                "pattern_fusion_invalid_rewrite "
                "reason_code=space_to_depth_invalid_blocksize "
                f"block_size={block_size}"
            )
        if int(shape1[5]) != block_size:
            raise ValueError(
                "pattern_fusion_invalid_rewrite "
                "reason_code=space_to_depth_blocksize_mismatch "
                f"shape1={shape1}"
            )
        if int(shape2[1]) != int(shape1[1]) * block_size * block_size:
            raise ValueError(
                "pattern_fusion_invalid_rewrite "
                "reason_code=space_to_depth_channel_mismatch "
                f"shape1={shape1} shape2={shape2}"
            )
        if int(shape1[2]) != int(shape2[2]) or int(shape1[4]) != int(shape2[3]):
            raise ValueError(
                "pattern_fusion_invalid_rewrite "
                "reason_code=space_to_depth_spatial_mismatch "
                f"shape1={shape1} shape2={shape2}"
            )

        for intermediate in [str(n0.output[0]), str(n1.output[0])]:
            if intermediate in graph_outputs or int(consumer_counts.get(intermediate, 0)) > 1:
                raise ValueError(
                    "pattern_fusion_invalid_rewrite "
                    "reason_code=space_to_depth_fanout_conflict "
                    f"tensor={intermediate}"
                )

        fused_nodes.append(
            _make_node(
                used_names=used_names,
                op_type="SpaceToDepth",
                inputs=[str(n0.input[0])],
                outputs=[str(n2.output[0])],
                name_base=n2.name or "space_to_depth_fused",
                attrs={"blocksize": int(block_size), "mode": "DCR"},
            )
        )
        rewritten += 1
        i += 3
    return fused_nodes, matched, rewritten


def apply_pattern_fusion_wave2(onnx_graph: onnx.ModelProto) -> Dict[str, Any]:
    graph = onnx_graph.graph
    used_names = _collect_used_names(graph)
    graph_outputs = _graph_output_names(graph)

    nodes = [_copy_node(node) for node in graph.node]
    const_map = _build_initializer_map(graph)

    total_matched = 0
    total_rewritten = 0
    matched_by_pattern: Dict[str, int] = {}
    rewritten_by_pattern: Dict[str, int] = {}

    nodes, m, r = _fuse_gelu_erf_chain(
        nodes=nodes,
        used_names=used_names,
        const_map=const_map,
        graph_outputs=graph_outputs,
    )
    total_matched += m
    total_rewritten += r
    matched_by_pattern["gelu_chain"] = int(m)
    rewritten_by_pattern["gelu_chain"] = int(r)

    nodes, m, r = _fuse_relu_clip_relu6(
        nodes=nodes,
        used_names=used_names,
        const_map=const_map,
        graph_outputs=graph_outputs,
    )
    total_matched += m
    total_rewritten += r
    matched_by_pattern["relu_clip_relu6"] = int(m)
    rewritten_by_pattern["relu_clip_relu6"] = int(r)

    nodes, m, r = _fuse_space_to_depth_chain(
        nodes=nodes,
        used_names=used_names,
        const_map=const_map,
        graph_outputs=graph_outputs,
    )
    total_matched += m
    total_rewritten += r
    matched_by_pattern["space_to_depth_chain"] = int(m)
    rewritten_by_pattern["space_to_depth_chain"] = int(r)

    if total_rewritten > 0:
        del graph.node[:]
        graph.node.extend(nodes)

    return {
        "matched_nodes": int(total_matched),
        "rewritten_nodes": int(total_rewritten),
        "changed": bool(total_rewritten > 0),
        "message": (
            f"matched_by_pattern={matched_by_pattern}, "
            f"rewritten_by_pattern={rewritten_by_pattern}"
        ),
    }


def register_pattern_fusion_wave2_rule() -> None:
    register_preprocess_rule(
        rule_id=PATTERN_FUSION_WAVE2_RULE_ID,
        callback=apply_pattern_fusion_wave2,
        overwrite=True,
    )
