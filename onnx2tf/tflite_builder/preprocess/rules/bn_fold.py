from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx
from onnx import helper, numpy_helper

from onnx2tf.tflite_builder.preprocess.pipeline import register_preprocess_rule

BN_FOLD_WAVE0_RULE_ID = "bn_fold_wave0"


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


def _build_const_map(graph: onnx.GraphProto) -> Dict[str, np.ndarray]:
    const_map: Dict[str, np.ndarray] = {}
    for init in graph.initializer:
        const_map[str(init.name)] = np.asarray(numpy_helper.to_array(init))
    for node in graph.node:
        if str(node.op_type) != "Constant" or len(node.output) < 1:
            continue
        for attr in node.attribute:
            if str(attr.name) == "value" and attr.type == onnx.AttributeProto.TENSOR:
                const_map[str(node.output[0])] = np.asarray(numpy_helper.to_array(attr.t))
                break
    return const_map


def _build_producer_index(nodes: List[onnx.NodeProto]) -> Dict[str, int]:
    producers: Dict[str, int] = {}
    for idx, node in enumerate(nodes):
        for out_name in node.output:
            if out_name == "":
                continue
            producers[str(out_name)] = int(idx)
    return producers


def _build_consumer_count(nodes: List[onnx.NodeProto]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for node in nodes:
        for in_name in node.input:
            if in_name == "":
                continue
            key = str(in_name)
            counts[key] = int(counts.get(key, 0) + 1)
    return counts


def _build_rank_map(graph: onnx.GraphProto) -> Dict[str, int]:
    rank_map: Dict[str, int] = {}

    def _record_value_info(vi: onnx.ValueInfoProto) -> None:
        if str(vi.name) == "":
            return
        if not vi.type.HasField("tensor_type"):
            return
        tensor_type = vi.type.tensor_type
        if not tensor_type.HasField("shape"):
            return
        rank_map[str(vi.name)] = int(len(list(tensor_type.shape.dim)))

    for vi in list(graph.input) + list(graph.value_info) + list(graph.output):
        _record_value_info(vi)
    return rank_map


def _bn_epsilon(node: onnx.NodeProto) -> float:
    eps = 1e-5
    for attr in node.attribute:
        if str(attr.name) != "epsilon":
            continue
        if attr.type == onnx.AttributeProto.FLOAT:
            eps = float(attr.f)
        elif attr.type == onnx.AttributeProto.INT:
            eps = float(attr.i)
    return float(eps)


def _reshape_bn_affine(
    *,
    bn_mul: np.ndarray,
    bn_add: np.ndarray,
    rank: int,
) -> Tuple[np.ndarray, np.ndarray]:
    channels = int(np.asarray(bn_mul).reshape(-1).size)
    if rank == 4:
        shape = (1, channels, 1, 1)
    elif rank == 3:
        shape = (1, channels, 1)
    elif rank == 2:
        shape = (1, channels)
    else:
        shape = (channels,)
    return (
        np.asarray(bn_mul, dtype=np.float32).reshape(shape),
        np.asarray(bn_add, dtype=np.float32).reshape(shape),
    )


def _infer_rank_for_dq_bn_path(
    *,
    bn_input: str,
    bn_output: str,
    dq_node: onnx.NodeProto,
    rank_map: Dict[str, int],
    producer_idx: Dict[str, int],
    nodes: List[onnx.NodeProto],
) -> Optional[int]:
    for candidate_name in [bn_input, bn_output]:
        rank = rank_map.get(str(candidate_name), None)
        if rank is not None and int(rank) > 0:
            return int(rank)

    if len(dq_node.input) >= 1:
        dq_raw_input = str(dq_node.input[0])
        rank = rank_map.get(dq_raw_input, None)
        if rank is not None and int(rank) > 0:
            return int(rank)
        up_idx = producer_idx.get(dq_raw_input, None)
        if up_idx is not None:
            up_type = str(nodes[int(up_idx)].op_type)
            if up_type in {"QLinearMatMul", "MatMul", "Gemm"}:
                return 2
            if up_type.startswith("QLinear"):
                return 4
            if up_type in {"Conv", "ConvTranspose", "MaxPool", "AveragePool"}:
                return 4
    return None


def _fold_conv_bn(
    *,
    conv_w: np.ndarray,
    conv_b: Optional[np.ndarray],
    bn_scale: np.ndarray,
    bn_bias: np.ndarray,
    bn_mean: np.ndarray,
    bn_var: np.ndarray,
    epsilon: float,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    w = np.asarray(conv_w, dtype=np.float32)
    if w.ndim < 1:
        return None

    channels = int(w.shape[0])
    scale = np.asarray(bn_scale, dtype=np.float32).reshape(-1)
    bias = np.asarray(bn_bias, dtype=np.float32).reshape(-1)
    mean = np.asarray(bn_mean, dtype=np.float32).reshape(-1)
    var = np.asarray(bn_var, dtype=np.float32).reshape(-1)
    if any(int(v.size) != channels for v in [scale, bias, mean, var]):
        return None

    bn_mul = scale / np.sqrt(var + float(epsilon))
    reshape_dims = [channels] + [1] * (w.ndim - 1)
    w_folded = w * bn_mul.reshape(reshape_dims)

    if conv_b is None:
        b_base = np.zeros((channels,), dtype=np.float32)
    else:
        b_base = np.asarray(conv_b, dtype=np.float32).reshape(-1)
        if int(b_base.size) != channels:
            return None
    b_folded = (b_base - mean) * bn_mul + bias
    return np.asarray(w_folded, dtype=np.float32), np.asarray(b_folded, dtype=np.float32)


def _convtranspose_group(node: onnx.NodeProto) -> int:
    group = 1
    for attr in node.attribute:
        if str(attr.name) == "group":
            if attr.type == onnx.AttributeProto.INT:
                group = int(attr.i)
            break
    return int(group)


def _fold_convtranspose_bn(
    *,
    convt_w: np.ndarray,
    convt_b: Optional[np.ndarray],
    bn_scale: np.ndarray,
    bn_bias: np.ndarray,
    bn_mean: np.ndarray,
    bn_var: np.ndarray,
    epsilon: float,
    group: int,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    w = np.asarray(convt_w, dtype=np.float32)
    if w.ndim < 2:
        return None
    if int(group) <= 0:
        return None

    in_channels = int(w.shape[0])
    out_channels_per_group = int(w.shape[1])
    if in_channels <= 0 or out_channels_per_group <= 0:
        return None
    if in_channels % int(group) != 0:
        return None
    out_channels = int(out_channels_per_group * int(group))

    scale = np.asarray(bn_scale, dtype=np.float32).reshape(-1)
    bias = np.asarray(bn_bias, dtype=np.float32).reshape(-1)
    mean = np.asarray(bn_mean, dtype=np.float32).reshape(-1)
    var = np.asarray(bn_var, dtype=np.float32).reshape(-1)
    if any(int(v.size) != out_channels for v in [scale, bias, mean, var]):
        return None

    bn_mul = scale / np.sqrt(var + float(epsilon))
    in_per_group = int(in_channels // int(group))
    w_folded = np.asarray(w, dtype=np.float32).copy()
    for c in range(in_channels):
        g = int(c // in_per_group)
        start = int(g * out_channels_per_group)
        stop = int(start + out_channels_per_group)
        local_mul = bn_mul[start:stop]
        reshape_dims = [out_channels_per_group] + [1] * (w_folded.ndim - 2)
        w_folded[c, ...] *= local_mul.reshape(reshape_dims)

    if convt_b is None:
        b_base = np.zeros((out_channels,), dtype=np.float32)
    else:
        b_base = np.asarray(convt_b, dtype=np.float32).reshape(-1)
        if int(b_base.size) != out_channels:
            return None
    b_folded = (b_base - mean) * bn_mul + bias
    return np.asarray(w_folded, dtype=np.float32), np.asarray(b_folded, dtype=np.float32)


def _extract_channel_affine_vector(
    *,
    values: np.ndarray,
    out_channels: int,
    rank: int,
) -> Optional[np.ndarray]:
    arr = np.asarray(values, dtype=np.float32)
    if int(arr.size) == 0 or int(out_channels) <= 0 or int(rank) < 2:
        return None
    if int(arr.size) == 1:
        scalar = float(arr.reshape(-1)[0])
        return np.full((int(out_channels),), scalar, dtype=np.float32)

    dims = [int(v) for v in list(arr.shape)]
    if len(dims) > int(rank):
        extra = int(len(dims) - int(rank))
        if any(int(v) != 1 for v in dims[:extra]):
            return None
        dims = dims[extra:]
        arr = np.reshape(arr, dims)
    if len(dims) < int(rank):
        dims = [1] * int(int(rank) - len(dims)) + dims
        arr = np.reshape(arr, dims)

    for axis, dim in enumerate(dims):
        if int(axis) == 1:
            if int(dim) not in [1, int(out_channels)]:
                return None
        else:
            if int(dim) != 1:
                return None

    if int(dims[1]) == 1:
        scalar = float(np.asarray(arr).reshape(-1)[0])
        return np.full((int(out_channels),), scalar, dtype=np.float32)

    index = [0 for _ in range(int(rank))]
    index[1] = slice(None)
    vec = np.asarray(arr, dtype=np.float32)[tuple(index)].reshape(-1)
    if int(vec.size) != int(out_channels):
        return None
    return np.asarray(vec, dtype=np.float32)


def _fold_conv_mul_add_affine(
    *,
    nodes: List[onnx.NodeProto],
    used_names: set[str],
    const_map: Dict[str, np.ndarray],
    graph_outputs: set[str],
) -> Tuple[List[onnx.NodeProto], List[onnx.TensorProto], int, int]:
    if len(nodes) < 3:
        return list(nodes), [], 0, 0

    consumer_count = _build_consumer_count(nodes)
    rewritten_nodes: List[onnx.NodeProto] = []
    new_initializers: List[onnx.TensorProto] = []
    matched_patterns = 0
    rewritten_patterns = 0

    i = 0
    while i < len(nodes):
        if i + 2 >= len(nodes):
            rewritten_nodes.append(_copy_node(nodes[i]))
            i += 1
            continue

        conv_node = nodes[i]
        mul_node = nodes[i + 1]
        add_node = nodes[i + 2]
        conv_type = str(conv_node.op_type)
        if (
            conv_type not in {"Conv", "ConvTranspose"}
            or str(mul_node.op_type) != "Mul"
            or str(add_node.op_type) != "Add"
            or len(conv_node.output) < 1
            or len(mul_node.output) < 1
            or len(add_node.output) < 1
            or len(mul_node.input) < 2
            or len(add_node.input) < 2
        ):
            rewritten_nodes.append(_copy_node(nodes[i]))
            i += 1
            continue

        conv_out = str(conv_node.output[0])
        mul_out = str(mul_node.output[0])
        add_out = str(add_node.output[0])
        mul_inputs = [str(v) for v in mul_node.input]
        add_inputs = [str(v) for v in add_node.input]
        if conv_out not in mul_inputs:
            rewritten_nodes.append(_copy_node(nodes[i]))
            i += 1
            continue
        if mul_out not in add_inputs:
            rewritten_nodes.append(_copy_node(nodes[i]))
            i += 1
            continue

        matched_patterns += 1

        if (
            int(consumer_count.get(conv_out, 0)) != 1
            or int(consumer_count.get(mul_out, 0)) != 1
            or conv_out in graph_outputs
            or mul_out in graph_outputs
        ):
            rewritten_nodes.append(_copy_node(nodes[i]))
            i += 1
            continue

        mul_const_name = mul_inputs[0] if mul_inputs[1] == conv_out else mul_inputs[1]
        add_const_name = add_inputs[0] if add_inputs[1] == mul_out else add_inputs[1]
        mul_const = const_map.get(str(mul_const_name), None)
        add_const = const_map.get(str(add_const_name), None)
        if mul_const is None or add_const is None:
            rewritten_nodes.append(_copy_node(nodes[i]))
            i += 1
            continue

        if len(conv_node.input) < 2:
            rewritten_nodes.append(_copy_node(nodes[i]))
            i += 1
            continue
        conv_w_name = str(conv_node.input[1])
        conv_b_name = str(conv_node.input[2]) if len(conv_node.input) >= 3 else None
        conv_w = const_map.get(conv_w_name, None)
        conv_b = const_map.get(conv_b_name, None) if conv_b_name is not None else None
        if conv_w is None:
            rewritten_nodes.append(_copy_node(nodes[i]))
            i += 1
            continue

        w_arr = np.asarray(conv_w, dtype=np.float32)
        rank = int(w_arr.ndim)
        if rank < 2:
            rewritten_nodes.append(_copy_node(nodes[i]))
            i += 1
            continue

        if conv_type == "Conv":
            out_channels = int(w_arr.shape[0])
        else:
            out_channels_per_group = int(w_arr.shape[1])
            out_channels = int(out_channels_per_group * int(_convtranspose_group(conv_node)))

        mul_vec = _extract_channel_affine_vector(
            values=np.asarray(mul_const),
            out_channels=int(out_channels),
            rank=int(rank),
        )
        add_vec = _extract_channel_affine_vector(
            values=np.asarray(add_const),
            out_channels=int(out_channels),
            rank=int(rank),
        )
        if mul_vec is None or add_vec is None:
            rewritten_nodes.append(_copy_node(nodes[i]))
            i += 1
            continue

        neutral_mean = np.zeros((int(out_channels),), dtype=np.float32)
        neutral_var = np.ones((int(out_channels),), dtype=np.float32)
        if conv_type == "Conv":
            folded = _fold_conv_bn(
                conv_w=w_arr,
                conv_b=conv_b,
                bn_scale=mul_vec,
                bn_bias=add_vec,
                bn_mean=neutral_mean,
                bn_var=neutral_var,
                epsilon=0.0,
            )
        else:
            folded = _fold_convtranspose_bn(
                convt_w=w_arr,
                convt_b=conv_b,
                bn_scale=mul_vec,
                bn_bias=add_vec,
                bn_mean=neutral_mean,
                bn_var=neutral_var,
                epsilon=0.0,
                group=_convtranspose_group(conv_node),
            )
        if folded is None:
            rewritten_nodes.append(_copy_node(nodes[i]))
            i += 1
            continue
        w_folded, b_folded = folded

        op_name = conv_node.name or conv_type.lower()
        new_w_name = _next_name(used_names, f"{op_name}_affine_fold_w")
        new_b_name = _next_name(used_names, f"{op_name}_affine_fold_b")
        new_initializers.extend(
            [
                numpy_helper.from_array(np.asarray(w_folded, dtype=np.float32), name=new_w_name),
                numpy_helper.from_array(np.asarray(b_folded, dtype=np.float32), name=new_b_name),
            ]
        )
        const_map[new_w_name] = np.asarray(w_folded, dtype=np.float32)
        const_map[new_b_name] = np.asarray(b_folded, dtype=np.float32)

        folded_conv = _copy_node(conv_node)
        del folded_conv.input[:]
        folded_conv.input.extend([str(conv_node.input[0]), new_w_name, new_b_name])
        del folded_conv.output[:]
        folded_conv.output.extend([add_out])
        rewritten_nodes.append(folded_conv)

        rewritten_patterns += 1
        i += 3

    return rewritten_nodes, new_initializers, int(matched_patterns), int(rewritten_patterns)


def apply_bn_fold_wave0(onnx_graph: onnx.ModelProto) -> Dict[str, Any]:
    graph = onnx_graph.graph
    nodes = list(graph.node)
    bn_node_count = int(
        sum(1 for n in nodes if str(n.op_type) == "BatchNormalization")
    )
    if len(nodes) == 0:
        return {
            "matched_nodes": 0,
            "rewritten_nodes": 0,
            "changed": False,
            "message": (
                f"bn_nodes_in_graph={bn_node_count} "
                "matched_patterns=0 rewritten_patterns=0"
            ),
        }

    used_names = _collect_used_names(graph)
    const_map = _build_const_map(graph)
    producer_idx = _build_producer_index(nodes)
    consumer_count = _build_consumer_count(nodes)
    rank_map = _build_rank_map(graph)
    graph_outputs = {str(v.name) for v in graph.output if str(v.name) != ""}

    rewritten_nodes_bn: List[onnx.NodeProto] = []
    new_initializers: List[onnx.TensorProto] = []
    skip_indices: set[int] = set()
    replacement_nodes: Dict[int, List[onnx.NodeProto]] = {}

    matched_patterns = 0
    rewritten_patterns = 0

    for bn_idx, bn_node in enumerate(nodes):
        if str(bn_node.op_type) != "BatchNormalization" or bn_idx in skip_indices:
            continue
        if len(bn_node.input) < 5 or len(bn_node.output) < 1:
            continue

        bn_input = str(bn_node.input[0])
        bn_output = str(bn_node.output[0])
        conv_idx = producer_idx.get(bn_input, None)
        if conv_idx is None:
            continue
        producer_node = nodes[int(conv_idx)]
        producer_type = str(producer_node.op_type)
        if producer_type not in {"Conv", "ConvTranspose", "DequantizeLinear"}:
            continue
        if len(producer_node.output) < 1:
            continue
        producer_output = str(producer_node.output[0])
        if producer_output != bn_input:
            continue
        if int(consumer_count.get(producer_output, 0)) != 1:
            continue
        if producer_output in graph_outputs:
            continue

        bn_scale_name = str(bn_node.input[1])
        bn_bias_name = str(bn_node.input[2])
        bn_mean_name = str(bn_node.input[3])
        bn_var_name = str(bn_node.input[4])

        bn_scale = const_map.get(bn_scale_name, None)
        bn_bias = const_map.get(bn_bias_name, None)
        bn_mean = const_map.get(bn_mean_name, None)
        bn_var = const_map.get(bn_var_name, None)
        if (
            bn_scale is None
            or bn_bias is None
            or bn_mean is None
            or bn_var is None
        ):
            continue

        matched_patterns += 1

        if producer_type in {"Conv", "ConvTranspose"}:
            if len(producer_node.input) < 2:
                continue
            w_name = str(producer_node.input[1])
            b_name = str(producer_node.input[2]) if len(producer_node.input) >= 3 else None
            conv_w = const_map.get(w_name, None)
            conv_b = const_map.get(b_name, None) if b_name is not None else None
            if conv_w is None:
                continue

            if producer_type == "Conv":
                folded = _fold_conv_bn(
                    conv_w=conv_w,
                    conv_b=conv_b,
                    bn_scale=bn_scale,
                    bn_bias=bn_bias,
                    bn_mean=bn_mean,
                    bn_var=bn_var,
                    epsilon=_bn_epsilon(bn_node),
                )
            else:
                folded = _fold_convtranspose_bn(
                    convt_w=conv_w,
                    convt_b=conv_b,
                    bn_scale=bn_scale,
                    bn_bias=bn_bias,
                    bn_mean=bn_mean,
                    bn_var=bn_var,
                    epsilon=_bn_epsilon(bn_node),
                    group=_convtranspose_group(producer_node),
                )
            if folded is None:
                continue
            w_folded, b_folded = folded

            op_name = producer_node.name or producer_type.lower()
            new_w_name = _next_name(used_names, f"{op_name}_bn_fold_w")
            new_b_name = _next_name(used_names, f"{op_name}_bn_fold_b")
            new_initializers.extend(
                [
                    numpy_helper.from_array(np.asarray(w_folded, dtype=np.float32), name=new_w_name),
                    numpy_helper.from_array(np.asarray(b_folded, dtype=np.float32), name=new_b_name),
                ]
            )
            const_map[new_w_name] = np.asarray(w_folded, dtype=np.float32)
            const_map[new_b_name] = np.asarray(b_folded, dtype=np.float32)

            folded_op = _copy_node(producer_node)
            del folded_op.input[:]
            folded_op.input.extend([str(producer_node.input[0]), new_w_name, new_b_name])
            del folded_op.output[:]
            folded_op.output.extend([bn_output])

            nodes[int(conv_idx)] = folded_op
            skip_indices.add(int(bn_idx))
            rewritten_patterns += 1
            continue

        if producer_type == "DequantizeLinear":
            rank = _infer_rank_for_dq_bn_path(
                bn_input=bn_input,
                bn_output=bn_output,
                dq_node=producer_node,
                rank_map=rank_map,
                producer_idx=producer_idx,
                nodes=nodes,
            )
            if rank is None:
                continue

            scale = np.asarray(bn_scale, dtype=np.float32).reshape(-1)
            bias = np.asarray(bn_bias, dtype=np.float32).reshape(-1)
            mean = np.asarray(bn_mean, dtype=np.float32).reshape(-1)
            var = np.asarray(bn_var, dtype=np.float32).reshape(-1)
            if not (int(scale.size) == int(bias.size) == int(mean.size) == int(var.size)):
                continue
            bn_mul = scale / np.sqrt(var + float(_bn_epsilon(bn_node)))
            bn_add = bias - mean * bn_mul
            bn_mul_shaped, bn_add_shaped = _reshape_bn_affine(
                bn_mul=np.asarray(bn_mul, dtype=np.float32),
                bn_add=np.asarray(bn_add, dtype=np.float32),
                rank=int(rank),
            )

            bn_name = bn_node.name or "bn"
            mul_const_name = _next_name(used_names, f"{bn_name}_fold_mul")
            add_const_name = _next_name(used_names, f"{bn_name}_fold_add")
            mul_out_name = _next_name(used_names, f"{bn_name}_fold_mul_out")
            new_initializers.extend(
                [
                    numpy_helper.from_array(np.asarray(bn_mul_shaped, dtype=np.float32), name=mul_const_name),
                    numpy_helper.from_array(np.asarray(bn_add_shaped, dtype=np.float32), name=add_const_name),
                ]
            )
            const_map[mul_const_name] = np.asarray(bn_mul_shaped, dtype=np.float32)
            const_map[add_const_name] = np.asarray(bn_add_shaped, dtype=np.float32)

            mul_node = helper.make_node(
                "Mul",
                [bn_input, mul_const_name],
                [mul_out_name],
                name=_next_name(used_names, f"{bn_name}_fold_mul"),
            )
            add_node = helper.make_node(
                "Add",
                [mul_out_name, add_const_name],
                [bn_output],
                name=_next_name(used_names, f"{bn_name}_fold_add"),
            )
            replacement_nodes[int(bn_idx)] = [mul_node, add_node]
            skip_indices.add(int(bn_idx))
            rewritten_patterns += 1
            continue

    for idx, node in enumerate(nodes):
        replacement = replacement_nodes.get(int(idx), None)
        if replacement is not None:
            for rep_node in replacement:
                rewritten_nodes_bn.append(_copy_node(rep_node))
            continue
        if idx in skip_indices:
            continue
        rewritten_nodes_bn.append(_copy_node(node))

    (
        rewritten_nodes,
        affine_initializers,
        affine_matched_patterns,
        affine_rewritten_patterns,
    ) = _fold_conv_mul_add_affine(
        nodes=rewritten_nodes_bn,
        used_names=used_names,
        const_map=const_map,
        graph_outputs=graph_outputs,
    )
    new_initializers.extend(affine_initializers)

    total_rewritten_patterns = int(rewritten_patterns + affine_rewritten_patterns)
    if total_rewritten_patterns > 0:
        del graph.node[:]
        graph.node.extend(rewritten_nodes)
        graph.initializer.extend(new_initializers)

    return {
        "matched_nodes": int(matched_patterns * 2 + affine_matched_patterns * 3),
        "rewritten_nodes": int(rewritten_patterns * 2 + affine_rewritten_patterns * 3),
        "changed": bool(total_rewritten_patterns > 0),
        "message": (
            f"bn_nodes_in_graph={bn_node_count} "
            f"matched_patterns={matched_patterns} rewritten_patterns={rewritten_patterns} "
            f"matched_affine_patterns={affine_matched_patterns} "
            f"rewritten_affine_patterns={affine_rewritten_patterns}"
        ),
    }


def register_bn_fold_wave0_rule() -> None:
    register_preprocess_rule(
        rule_id=BN_FOLD_WAVE0_RULE_ID,
        callback=apply_bn_fold_wave0,
        overwrite=True,
    )
