from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnx
from onnx import helper, numpy_helper

from onnx2tf.tflite_builder.preprocess.pipeline import register_preprocess_rule

PSEUDO_OPS_WAVE1_RULE_ID = "pseudo_ops_wave1"


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
    candidate = str(base)
    if candidate == "":
        candidate = "tmp"
    if candidate not in used_names:
        used_names.add(candidate)
        return candidate
    i = 1
    while True:
        candidate_i = f"{candidate}_{i}"
        if candidate_i not in used_names:
            used_names.add(candidate_i)
            return candidate_i
        i += 1


def _copy_node(node: onnx.NodeProto) -> onnx.NodeProto:
    cloned = onnx.NodeProto()
    cloned.CopyFrom(node)
    return cloned


def _node_attr_float(node: onnx.NodeProto, name: str, default: float) -> float:
    for attr in node.attribute:
        if str(attr.name) != str(name):
            continue
        if attr.type == onnx.AttributeProto.FLOAT:
            return float(attr.f)
        if attr.type == onnx.AttributeProto.INT:
            return float(attr.i)
    return float(default)


def _node_attr_str(node: onnx.NodeProto, name: str, default: str) -> str:
    for attr in node.attribute:
        if str(attr.name) != str(name):
            continue
        if attr.type == onnx.AttributeProto.STRING:
            try:
                return str(attr.s.decode("utf-8"))
            except Exception:
                return str(default)
    return str(default)


def _find_const_tensor_attribute(node: onnx.NodeProto, name: str) -> Optional[onnx.TensorProto]:
    for attr in node.attribute:
        if str(attr.name) != str(name):
            continue
        if attr.type == onnx.AttributeProto.TENSOR:
            return attr.t
    return None


def _make_const_initializer(
    *,
    used_names: set[str],
    name_base: str,
    value: np.ndarray,
) -> onnx.TensorProto:
    name = _next_name(used_names, name_base)
    arr = np.asarray(value)
    return numpy_helper.from_array(arr, name=name)


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


def _rewrite_hardswish(
    *,
    node: onnx.NodeProto,
    used_names: set[str],
) -> Optional[Tuple[List[onnx.NodeProto], List[onnx.TensorProto]]]:
    if str(node.op_type) != "HardSwish" or len(node.input) < 1 or len(node.output) < 1:
        return None
    x_name = str(node.input[0])
    y_name = str(node.output[0])

    c3 = _make_const_initializer(
        used_names=used_names,
        name_base=f"{node.name or 'HardSwish'}_c3",
        value=np.asarray(3.0, dtype=np.float32),
    )
    c6 = _make_const_initializer(
        used_names=used_names,
        name_base=f"{node.name or 'HardSwish'}_c6",
        value=np.asarray(6.0, dtype=np.float32),
    )
    t_add = _next_name(used_names, f"{node.name or 'HardSwish'}_add_out")
    t_clip = _next_name(used_names, f"{node.name or 'HardSwish'}_clip_out")
    t_div = _next_name(used_names, f"{node.name or 'HardSwish'}_div_out")
    rewritten = [
        _make_node(
            used_names=used_names,
            op_type="Add",
            inputs=[x_name, c3.name],
            outputs=[t_add],
            name_base=node.name or "HardSwish",
        ),
        _make_node(
            used_names=used_names,
            op_type="Clip",
            inputs=[t_add],
            outputs=[t_clip],
            name_base=node.name or "HardSwish",
            attrs={"min": 0.0, "max": 6.0},
        ),
        _make_node(
            used_names=used_names,
            op_type="Div",
            inputs=[t_clip, c6.name],
            outputs=[t_div],
            name_base=node.name or "HardSwish",
        ),
        _make_node(
            used_names=used_names,
            op_type="Mul",
            inputs=[x_name, t_div],
            outputs=[y_name],
            name_base=node.name or "HardSwish",
        ),
    ]
    return rewritten, [c3, c6]


def _rewrite_leaky_relu(
    *,
    node: onnx.NodeProto,
    used_names: set[str],
) -> Optional[Tuple[List[onnx.NodeProto], List[onnx.TensorProto]]]:
    if str(node.op_type) != "LeakyRelu" or len(node.input) < 1 or len(node.output) < 1:
        return None
    alpha = _node_attr_float(node, "alpha", 0.01)
    x_name = str(node.input[0])
    y_name = str(node.output[0])
    alpha_const = _make_const_initializer(
        used_names=used_names,
        name_base=f"{node.name or 'LeakyRelu'}_alpha",
        value=np.asarray(alpha, dtype=np.float32),
    )
    t_neg = _next_name(used_names, f"{node.name or 'LeakyRelu'}_neg_out")
    t_pos_relu = _next_name(used_names, f"{node.name or 'LeakyRelu'}_relu_pos_out")
    t_neg_relu = _next_name(used_names, f"{node.name or 'LeakyRelu'}_relu_neg_out")
    t_scaled = _next_name(used_names, f"{node.name or 'LeakyRelu'}_scaled_neg_out")
    rewritten = [
        _make_node(
            used_names=used_names,
            op_type="Neg",
            inputs=[x_name],
            outputs=[t_neg],
            name_base=node.name or "LeakyRelu",
        ),
        _make_node(
            used_names=used_names,
            op_type="Relu",
            inputs=[x_name],
            outputs=[t_pos_relu],
            name_base=node.name or "LeakyRelu",
        ),
        _make_node(
            used_names=used_names,
            op_type="Relu",
            inputs=[t_neg],
            outputs=[t_neg_relu],
            name_base=node.name or "LeakyRelu",
        ),
        _make_node(
            used_names=used_names,
            op_type="Mul",
            inputs=[t_neg_relu, alpha_const.name],
            outputs=[t_scaled],
            name_base=node.name or "LeakyRelu",
        ),
        _make_node(
            used_names=used_names,
            op_type="Sub",
            inputs=[t_pos_relu, t_scaled],
            outputs=[y_name],
            name_base=node.name or "LeakyRelu",
        ),
    ]
    return rewritten, [alpha_const]


def _rewrite_prelu(
    *,
    node: onnx.NodeProto,
    used_names: set[str],
) -> Optional[Tuple[List[onnx.NodeProto], List[onnx.TensorProto]]]:
    if str(node.op_type) != "PRelu" or len(node.input) < 2 or len(node.output) < 1:
        return None
    x_name = str(node.input[0])
    slope_name = str(node.input[1])
    y_name = str(node.output[0])
    t_neg = _next_name(used_names, f"{node.name or 'PRelu'}_neg_out")
    t_pos_relu = _next_name(used_names, f"{node.name or 'PRelu'}_relu_pos_out")
    t_neg_relu = _next_name(used_names, f"{node.name or 'PRelu'}_relu_neg_out")
    t_scaled = _next_name(used_names, f"{node.name or 'PRelu'}_scaled_neg_out")
    rewritten = [
        _make_node(
            used_names=used_names,
            op_type="Neg",
            inputs=[x_name],
            outputs=[t_neg],
            name_base=node.name or "PRelu",
        ),
        _make_node(
            used_names=used_names,
            op_type="Relu",
            inputs=[x_name],
            outputs=[t_pos_relu],
            name_base=node.name or "PRelu",
        ),
        _make_node(
            used_names=used_names,
            op_type="Relu",
            inputs=[t_neg],
            outputs=[t_neg_relu],
            name_base=node.name or "PRelu",
        ),
        _make_node(
            used_names=used_names,
            op_type="Mul",
            inputs=[t_neg_relu, slope_name],
            outputs=[t_scaled],
            name_base=node.name or "PRelu",
        ),
        _make_node(
            used_names=used_names,
            op_type="Sub",
            inputs=[t_pos_relu, t_scaled],
            outputs=[y_name],
            name_base=node.name or "PRelu",
        ),
    ]
    return rewritten, []


def _try_get_pow_exponent(
    *,
    node: onnx.NodeProto,
    const_map: Dict[str, np.ndarray],
) -> Optional[float]:
    if str(node.op_type) != "Pow" or len(node.input) < 2:
        return None
    exponent_name = str(node.input[1])
    if exponent_name not in const_map:
        return None
    values = np.asarray(const_map[exponent_name]).reshape(-1)
    if values.size == 0:
        return None
    if not np.allclose(values, values[0], rtol=0.0, atol=0.0):
        return None
    return float(values[0])


def _rewrite_pow(
    *,
    node: onnx.NodeProto,
    used_names: set[str],
    const_map: Dict[str, np.ndarray],
) -> Optional[Tuple[List[onnx.NodeProto], List[onnx.TensorProto]]]:
    exponent = _try_get_pow_exponent(node=node, const_map=const_map)
    if exponent is None or len(node.input) < 1 or len(node.output) < 1:
        return None
    x_name = str(node.input[0])
    y_name = str(node.output[0])
    if abs(exponent - 1.0) <= 1e-6:
        rewritten = [
            _make_node(
                used_names=used_names,
                op_type="Identity",
                inputs=[x_name],
                outputs=[y_name],
                name_base=node.name or "Pow",
            ),
        ]
        return rewritten, []
    if abs(exponent - 2.0) <= 1e-6:
        rewritten = [
            _make_node(
                used_names=used_names,
                op_type="Mul",
                inputs=[x_name, x_name],
                outputs=[y_name],
                name_base=node.name or "Pow",
            ),
        ]
        return rewritten, []
    if abs(exponent - 0.5) <= 1e-6:
        rewritten = [
            _make_node(
                used_names=used_names,
                op_type="Sqrt",
                inputs=[x_name],
                outputs=[y_name],
                name_base=node.name or "Pow",
            ),
        ]
        return rewritten, []
    return None


def _rewrite_gelu(
    *,
    node: onnx.NodeProto,
    used_names: set[str],
) -> Optional[Tuple[List[onnx.NodeProto], List[onnx.TensorProto]]]:
    if str(node.op_type) not in {"Gelu", "GeLU"} or len(node.input) < 1 or len(node.output) < 1:
        return None
    approximate = _node_attr_str(node, "approximate", "none").lower()
    if approximate not in {"none", "", "tanh"}:
        return None
    x_name = str(node.input[0])
    y_name = str(node.output[0])

    c_half = _make_const_initializer(
        used_names=used_names,
        name_base=f"{node.name or 'Gelu'}_half",
        value=np.asarray(0.5, dtype=np.float32),
    )
    c_one = _make_const_initializer(
        used_names=used_names,
        name_base=f"{node.name or 'Gelu'}_one",
        value=np.asarray(1.0, dtype=np.float32),
    )
    c_poly = _make_const_initializer(
        used_names=used_names,
        name_base=f"{node.name or 'Gelu'}_poly",
        value=np.asarray(0.044715, dtype=np.float32),
    )
    c_scale = _make_const_initializer(
        used_names=used_names,
        name_base=f"{node.name or 'Gelu'}_scale",
        value=np.asarray(np.sqrt(2.0 / np.pi), dtype=np.float32),
    )
    t_x2 = _next_name(used_names, f"{node.name or 'Gelu'}_x2_out")
    t_x3 = _next_name(used_names, f"{node.name or 'Gelu'}_x3_out")
    t_poly = _next_name(used_names, f"{node.name or 'Gelu'}_poly_out")
    t_inner = _next_name(used_names, f"{node.name or 'Gelu'}_inner_out")
    t_scaled = _next_name(used_names, f"{node.name or 'Gelu'}_scaled_out")
    t_tanh = _next_name(used_names, f"{node.name or 'Gelu'}_tanh_out")
    t_add = _next_name(used_names, f"{node.name or 'Gelu'}_add_out")
    t_mul = _next_name(used_names, f"{node.name or 'Gelu'}_mul_out")
    rewritten = [
        _make_node(
            used_names=used_names,
            op_type="Mul",
            inputs=[x_name, x_name],
            outputs=[t_x2],
            name_base=node.name or "Gelu",
        ),
        _make_node(
            used_names=used_names,
            op_type="Mul",
            inputs=[t_x2, x_name],
            outputs=[t_x3],
            name_base=node.name or "Gelu",
        ),
        _make_node(
            used_names=used_names,
            op_type="Mul",
            inputs=[t_x3, c_poly.name],
            outputs=[t_poly],
            name_base=node.name or "Gelu",
        ),
        _make_node(
            used_names=used_names,
            op_type="Add",
            inputs=[x_name, t_poly],
            outputs=[t_inner],
            name_base=node.name or "Gelu",
        ),
        _make_node(
            used_names=used_names,
            op_type="Mul",
            inputs=[t_inner, c_scale.name],
            outputs=[t_scaled],
            name_base=node.name or "Gelu",
        ),
        _make_node(
            used_names=used_names,
            op_type="Tanh",
            inputs=[t_scaled],
            outputs=[t_tanh],
            name_base=node.name or "Gelu",
        ),
        _make_node(
            used_names=used_names,
            op_type="Add",
            inputs=[t_tanh, c_one.name],
            outputs=[t_add],
            name_base=node.name or "Gelu",
        ),
        _make_node(
            used_names=used_names,
            op_type="Mul",
            inputs=[x_name, t_add],
            outputs=[t_mul],
            name_base=node.name or "Gelu",
        ),
        _make_node(
            used_names=used_names,
            op_type="Mul",
            inputs=[t_mul, c_half.name],
            outputs=[y_name],
            name_base=node.name or "Gelu",
        ),
    ]
    return rewritten, [c_half, c_one, c_poly, c_scale]


def apply_pseudo_ops_wave1(onnx_graph: onnx.ModelProto) -> Dict[str, Any]:
    graph = onnx_graph.graph
    used_names = _collect_used_names(graph)
    const_map = _build_initializer_map(graph)

    rewritten_nodes: List[onnx.NodeProto] = []
    new_initializers: List[onnx.TensorProto] = []
    matched_nodes = 0
    rewritten_count = 0
    matched_by_op: Dict[str, int] = {}
    rewritten_by_op: Dict[str, int] = {}

    for node in graph.node:
        node_op = str(node.op_type)
        replacement: Optional[Tuple[List[onnx.NodeProto], List[onnx.TensorProto]]] = None
        if node_op in {"HardSwish", "LeakyRelu", "Pow", "Gelu", "GeLU", "MatMulInteger"}:
            matched_nodes += 1
            matched_by_op[node_op] = int(matched_by_op.get(node_op, 0) + 1)
            if node_op == "HardSwish":
                replacement = _rewrite_hardswish(node=node, used_names=used_names)
            elif node_op == "LeakyRelu":
                replacement = _rewrite_leaky_relu(node=node, used_names=used_names)
            elif node_op == "Pow":
                replacement = _rewrite_pow(node=node, used_names=used_names, const_map=const_map)
            elif node_op in {"Gelu", "GeLU"}:
                replacement = _rewrite_gelu(node=node, used_names=used_names)
            else:
                replacement = None

        if replacement is None:
            rewritten_nodes.append(_copy_node(node))
            continue

        nodes_out, inits_out = replacement
        rewritten_nodes.extend(nodes_out)
        new_initializers.extend(inits_out)
        rewritten_count += 1
        rewritten_by_op[node_op] = int(rewritten_by_op.get(node_op, 0) + 1)

    if rewritten_count > 0:
        del graph.node[:]
        graph.node.extend(rewritten_nodes)
        graph.initializer.extend(new_initializers)

    return {
        "matched_nodes": int(matched_nodes),
        "rewritten_nodes": int(rewritten_count),
        "changed": bool(rewritten_count > 0),
        "message": (
            f"matched_by_op={matched_by_op}, "
            f"rewritten_by_op={rewritten_by_op}"
        ),
    }


def register_pseudo_ops_wave1_rule() -> None:
    register_preprocess_rule(
        rule_id=PSEUDO_OPS_WAVE1_RULE_ID,
        callback=apply_pseudo_ops_wave1,
        overwrite=True,
    )
