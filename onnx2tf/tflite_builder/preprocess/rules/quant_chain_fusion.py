from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx
from onnx import helper, numpy_helper

from onnx2tf.tflite_builder.preprocess.pipeline import register_preprocess_rule

QUANT_CHAIN_FUSION_WAVE3_RULE_ID = "quant_chain_fusion_wave3"


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


def _build_initializer_map(graph: onnx.GraphProto) -> Dict[str, np.ndarray]:
    consts: Dict[str, np.ndarray] = {}
    for init in graph.initializer:
        consts[str(init.name)] = np.asarray(numpy_helper.to_array(init))
    for node in graph.node:
        if str(node.op_type) != "Constant" or len(node.output) < 1:
            continue
        for attr in node.attribute:
            if str(attr.name) == "value" and attr.type == onnx.AttributeProto.TENSOR:
                consts[str(node.output[0])] = np.asarray(numpy_helper.to_array(attr.t))
                break
    return consts


def _make_node(
    *,
    used_names: set[str],
    op_type: str,
    inputs: List[str],
    outputs: List[str],
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


def _reshape_bn_affine_for_prelu(
    *,
    bn_mul: np.ndarray,
    bn_add: np.ndarray,
    prelu: onnx.NodeProto,
    const_map: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reshape BN affine terms so ONNX/NCHW broadcasting semantics are preserved
    after fusing Dequantize->BatchNormalization->PRelu->Quantize.

    For conv-like paths, PRelu slope is typically [C,1,1] or [1,C,1,1].
    In those cases, BN affine terms must be reshaped to [1,C,1,1].
    """
    if len(prelu.input) < 2:
        return bn_mul, bn_add

    slope = const_map.get(str(prelu.input[1]), None)
    if slope is None:
        return bn_mul, bn_add

    slope_arr = np.asarray(slope)
    if slope_arr.ndim == 3:
        # ONNX PRelu slope for NCHW conv path: [C,1,1]
        if int(slope_arr.shape[1]) == 1 and int(slope_arr.shape[2]) == 1:
            c = int(slope_arr.shape[0])
            if int(bn_mul.size) == c and int(bn_add.size) == c:
                return bn_mul.reshape(1, c, 1, 1), bn_add.reshape(1, c, 1, 1)
    elif slope_arr.ndim == 4:
        # Sometimes already expanded to [1,C,1,1]
        if (
            int(slope_arr.shape[0]) == 1
            and int(slope_arr.shape[2]) == 1
            and int(slope_arr.shape[3]) == 1
        ):
            c = int(slope_arr.shape[1])
            if int(bn_mul.size) == c and int(bn_add.size) == c:
                return bn_mul.reshape(1, c, 1, 1), bn_add.reshape(1, c, 1, 1)
    elif slope_arr.ndim == 2:
        # Rank-2 path: [1,C]
        if int(slope_arr.shape[0]) == 1:
            c = int(slope_arr.shape[1])
            if int(bn_mul.size) == c and int(bn_add.size) == c:
                return bn_mul.reshape(1, c), bn_add.reshape(1, c)
    elif slope_arr.ndim == 1:
        c = int(slope_arr.shape[0])
        if int(bn_mul.size) == c and int(bn_add.size) == c:
            # Keep [1,C] to avoid invalid trailing-dim broadcast for rank-2 paths.
            return bn_mul.reshape(1, c), bn_add.reshape(1, c)

    return bn_mul, bn_add


def apply_quant_chain_fusion_wave3(onnx_graph: onnx.ModelProto) -> Dict[str, Any]:
    graph = onnx_graph.graph
    nodes = list(graph.node)
    if len(nodes) < 4:
        return {
            "matched_nodes": 0,
            "rewritten_nodes": 0,
            "changed": False,
            "message": "matched_patterns=0 rewritten_patterns=0",
        }

    used_names = _collect_used_names(graph)
    const_map = _build_initializer_map(graph)
    rewritten_nodes: List[onnx.NodeProto] = []
    new_initializers: List[onnx.TensorProto] = []

    matched_patterns = 0
    rewritten_patterns = 0
    i = 0
    while i < len(nodes):
        if i + 3 >= len(nodes):
            rewritten_nodes.append(_copy_node(nodes[i]))
            i += 1
            continue

        dq = nodes[i]
        bn = nodes[i + 1]
        prelu = nodes[i + 2]
        q = nodes[i + 3]
        if [
            str(dq.op_type),
            str(bn.op_type),
            str(prelu.op_type),
            str(q.op_type),
        ] != ["DequantizeLinear", "BatchNormalization", "PRelu", "QuantizeLinear"]:
            rewritten_nodes.append(_copy_node(nodes[i]))
            i += 1
            continue

        if (
            len(dq.output) < 1
            or len(bn.input) < 5
            or len(bn.output) < 1
            or len(prelu.input) < 2
            or len(prelu.output) < 1
            or len(q.input) < 2
            or len(q.output) < 1
        ):
            rewritten_nodes.append(_copy_node(nodes[i]))
            i += 1
            continue

        dq_out = str(dq.output[0])
        bn_in = str(bn.input[0])
        bn_out = str(bn.output[0])
        prelu_in = str(prelu.input[0])
        prelu_out = str(prelu.output[0])
        q_in = str(q.input[0])
        if not (bn_in == dq_out and prelu_in == bn_out and q_in == prelu_out):
            rewritten_nodes.append(_copy_node(nodes[i]))
            i += 1
            continue

        scale = const_map.get(str(bn.input[1]))
        bias = const_map.get(str(bn.input[2]))
        mean = const_map.get(str(bn.input[3]))
        var = const_map.get(str(bn.input[4]))
        if scale is None or bias is None or mean is None or var is None:
            rewritten_nodes.append(_copy_node(nodes[i]))
            i += 1
            continue

        matched_patterns += 1

        scale = np.asarray(scale, dtype=np.float32).reshape(-1)
        bias = np.asarray(bias, dtype=np.float32).reshape(-1)
        mean = np.asarray(mean, dtype=np.float32).reshape(-1)
        var = np.asarray(var, dtype=np.float32).reshape(-1)
        eps = _bn_epsilon(bn)
        bn_mul = scale / np.sqrt(var + eps)
        bn_add = bias - mean * bn_mul
        bn_mul, bn_add = _reshape_bn_affine_for_prelu(
            bn_mul=bn_mul,
            bn_add=bn_add,
            prelu=prelu,
            const_map=const_map,
        )

        mul_init_name = _next_name(used_names, f"{bn.name or 'bn'}_fused_mul")
        add_init_name = _next_name(used_names, f"{bn.name or 'bn'}_fused_add")
        mul_init = numpy_helper.from_array(np.asarray(bn_mul, dtype=np.float32), name=mul_init_name)
        add_init = numpy_helper.from_array(np.asarray(bn_add, dtype=np.float32), name=add_init_name)
        new_initializers.extend([mul_init, add_init])
        const_map[mul_init_name] = np.asarray(bn_mul, dtype=np.float32)
        const_map[add_init_name] = np.asarray(bn_add, dtype=np.float32)

        mul_out = _next_name(used_names, f"{bn.name or 'bn'}_mul_out")
        rewritten_nodes.append(_copy_node(dq))
        rewritten_nodes.append(
            _make_node(
                used_names=used_names,
                op_type="Mul",
                inputs=[dq_out, mul_init_name],
                outputs=[mul_out],
                name_base=bn.name or "bn_fused",
            )
        )
        rewritten_nodes.append(
            _make_node(
                used_names=used_names,
                op_type="Add",
                inputs=[mul_out, add_init_name],
                outputs=[bn_out],
                name_base=bn.name or "bn_fused",
            )
        )
        rewritten_nodes.append(_copy_node(prelu))
        rewritten_nodes.append(_copy_node(q))
        rewritten_patterns += 1
        i += 4

    if rewritten_patterns > 0:
        del graph.node[:]
        graph.node.extend(rewritten_nodes)
        graph.initializer.extend(new_initializers)

    return {
        "matched_nodes": int(matched_patterns * 4),
        "rewritten_nodes": int(rewritten_patterns * 4),
        "changed": bool(rewritten_patterns > 0),
        "message": f"matched_patterns={matched_patterns} rewritten_patterns={rewritten_patterns}",
    }


def register_quant_chain_fusion_wave3_rule() -> None:
    register_preprocess_rule(
        rule_id=QUANT_CHAIN_FUSION_WAVE3_RULE_ID,
        callback=apply_quant_chain_fusion_wave3,
        overwrite=True,
    )
