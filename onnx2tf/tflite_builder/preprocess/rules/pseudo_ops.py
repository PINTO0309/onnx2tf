from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import onnx
from onnx import helper, numpy_helper

from onnx2tf.tflite_builder.preprocess.pipeline import register_preprocess_rule

PSEUDO_OPS_WAVE1_RULE_ID = "pseudo_ops_wave1"

# NOTE:
# - This map must stay aligned with `onnx2tf.py --replace_to_pseudo_operators` help.
# - `inverse` is special: default behavior is pseudo-lowering, and explicit
#   `-rtpo Inverse` requests keeping the Inverse op as-is.
_PSEUDO_OP_ALIAS_TO_ONNX_OP: Dict[str, str] = {
    "abs": "Abs",
    "acos": "Acos",
    "asin": "Asin",
    "atan": "Atan",
    "erf": "Erf",
    "gathernd": "GatherND",
    "gelu": "Gelu",
    "inverse": "Inverse",
    "leakyrelu": "LeakyRelu",
    "neg": "Neg",
    "pow": "Pow",
    "power": "Pow",
    "prelu": "PRelu",
    "hardswish": "HardSwish",
    "matmulinteger": "MatMulInteger",
}

# Keep existing default behavior and add Inverse pseudo-lowering by default
# (explicit `-rtpo inverse` disables it).
_DEFAULT_TARGET_ONNX_OPS: Set[str] = {
    "LeakyRelu",
    "Pow",
    "MatMulInteger",
    "Inverse",
}

_REQUESTED_PSEUDO_OP_ALIASES: Optional[Set[str]] = None

_ONNX_TENSOR_TYPE_TO_NUMPY: Dict[int, Any] = {
    int(onnx.TensorProto.FLOAT16): np.float16,
    int(onnx.TensorProto.FLOAT): np.float32,
    int(onnx.TensorProto.DOUBLE): np.float64,
    int(onnx.TensorProto.INT8): np.int8,
    int(onnx.TensorProto.INT16): np.int16,
    int(onnx.TensorProto.INT32): np.int32,
    int(onnx.TensorProto.INT64): np.int64,
    int(onnx.TensorProto.UINT8): np.uint8,
    int(onnx.TensorProto.UINT16): np.uint16,
    int(onnx.TensorProto.UINT32): np.uint32,
    int(onnx.TensorProto.UINT64): np.uint64,
    int(onnx.TensorProto.BOOL): np.bool_,
}


def get_supported_pseudo_ops_wave1_aliases() -> List[str]:
    return sorted(list(_PSEUDO_OP_ALIAS_TO_ONNX_OP.keys()))


def configure_pseudo_ops_wave1_targets(
    pseudo_op_aliases: Optional[Sequence[str]],
) -> None:
    global _REQUESTED_PSEUDO_OP_ALIASES
    if pseudo_op_aliases is None:
        _REQUESTED_PSEUDO_OP_ALIASES = None
        return
    normalized: Set[str] = set()
    for alias in pseudo_op_aliases:
        alias_norm = str(alias).strip().lower()
        if alias_norm != "":
            normalized.add(alias_norm)
    _REQUESTED_PSEUDO_OP_ALIASES = normalized if len(normalized) > 0 else None


def _collect_used_names(graph: onnx.GraphProto) -> Set[str]:
    used: Set[str] = set()
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


def _next_name(used_names: Set[str], base: str) -> str:
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


def _node_attr_int(node: onnx.NodeProto, name: str, default: int) -> int:
    for attr in node.attribute:
        if str(attr.name) != str(name):
            continue
        if attr.type == onnx.AttributeProto.INT:
            return int(attr.i)
    return int(default)


def _find_const_tensor_attribute(node: onnx.NodeProto, name: str) -> Optional[onnx.TensorProto]:
    for attr in node.attribute:
        if str(attr.name) != str(name):
            continue
        if attr.type == onnx.AttributeProto.TENSOR:
            return attr.t
    return None


def _make_const_initializer(
    *,
    used_names: Set[str],
    name_base: str,
    value: np.ndarray,
) -> onnx.TensorProto:
    name = _next_name(used_names, name_base)
    arr = np.asarray(value)
    return numpy_helper.from_array(arr, name=name)


def _make_scalar_const_initializer(
    *,
    used_names: Set[str],
    name_base: str,
    value: Any,
    dtype: Any,
) -> onnx.TensorProto:
    return _make_const_initializer(
        used_names=used_names,
        name_base=name_base,
        value=np.asarray(value, dtype=dtype),
    )


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


def _build_elem_type_map(graph: onnx.GraphProto) -> Dict[str, int]:
    elem_type_map: Dict[str, int] = {}
    for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
        name = str(vi.name)
        if name == "":
            continue
        if not vi.type.HasField("tensor_type"):
            continue
        tensor_type = vi.type.tensor_type
        if not tensor_type.HasField("elem_type"):
            continue
        elem_type_map[name] = int(tensor_type.elem_type)
    for init in graph.initializer:
        name = str(init.name)
        if name == "":
            continue
        elem_type_map[name] = int(init.data_type)
    return elem_type_map


def _normalize_dim_value(dim: Any) -> Optional[int]:
    try:
        value = int(dim)
        if value <= 0:
            return None
        return value
    except Exception:
        return None


def _build_tensor_shape_map(
    graph: onnx.GraphProto,
    const_map: Dict[str, np.ndarray],
) -> Dict[str, List[Optional[int]]]:
    shape_map: Dict[str, List[Optional[int]]] = {}
    for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
        name = str(vi.name)
        if name == "" or not vi.type.HasField("tensor_type"):
            continue
        shape_proto = vi.type.tensor_type.shape
        dims: List[Optional[int]] = []
        for dim in shape_proto.dim:
            if dim.HasField("dim_value"):
                dims.append(_normalize_dim_value(dim.dim_value))
            else:
                dims.append(None)
        shape_map[name] = dims

    for init in graph.initializer:
        name = str(init.name)
        if name == "":
            continue
        shape_map[name] = [int(v) for v in list(init.dims)]

    for name, arr in const_map.items():
        if name in shape_map:
            continue
        try:
            shape_map[str(name)] = [int(v) for v in list(np.asarray(arr).shape)]
        except Exception:
            continue
    return shape_map


def _numpy_dtype_for_elem_type(
    elem_type: Optional[int],
    *,
    default: Any = np.float32,
) -> Any:
    if elem_type is None:
        return default
    return _ONNX_TENSOR_TYPE_TO_NUMPY.get(int(elem_type), default)


def _dtype_for_tensor_name(
    *,
    tensor_name: str,
    elem_type_map: Dict[str, int],
    default: Any = np.float32,
) -> Any:
    elem_type = elem_type_map.get(str(tensor_name), None)
    return _numpy_dtype_for_elem_type(elem_type, default=default)


def _dtype_for_node_input(
    *,
    node: onnx.NodeProto,
    elem_type_map: Dict[str, int],
    input_index: int = 0,
    default: Any = np.float32,
) -> Any:
    if len(node.input) <= int(input_index):
        return default
    input_name = str(node.input[int(input_index)])
    return _dtype_for_tensor_name(
        tensor_name=input_name,
        elem_type_map=elem_type_map,
        default=default,
    )


def _dtype_for_node_output(
    *,
    node: onnx.NodeProto,
    elem_type_map: Dict[str, int],
    output_index: int = 0,
    default: Any = np.float32,
) -> Any:
    if len(node.output) <= int(output_index):
        return default
    output_name = str(node.output[int(output_index)])
    return _dtype_for_tensor_name(
        tensor_name=output_name,
        elem_type_map=elem_type_map,
        default=default,
    )


def _is_float_dtype(np_dtype: Any) -> bool:
    try:
        return bool(np.issubdtype(np.dtype(np_dtype), np.floating))
    except Exception:
        return False


def _make_node(
    *,
    used_names: Set[str],
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


def _make_unary_node(
    *,
    used_names: Set[str],
    node: onnx.NodeProto,
    op_type: str,
    input_name: str,
    output_name: str,
) -> onnx.NodeProto:
    return _make_node(
        used_names=used_names,
        op_type=op_type,
        inputs=[input_name],
        outputs=[output_name],
        name_base=node.name or op_type,
    )


def _append_binary(
    *,
    nodes: List[onnx.NodeProto],
    used_names: Set[str],
    node: onnx.NodeProto,
    op_type: str,
    a: str,
    b: str,
    out_base: str,
) -> str:
    out = _next_name(used_names, out_base)
    nodes.append(
        _make_node(
            used_names=used_names,
            op_type=op_type,
            inputs=[a, b],
            outputs=[out],
            name_base=node.name or op_type,
        )
    )
    return out


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


def _rewrite_abs(
    *,
    node: onnx.NodeProto,
    used_names: Set[str],
    elem_type_map: Dict[str, int],
) -> Optional[Tuple[List[onnx.NodeProto], List[onnx.TensorProto]]]:
    if str(node.op_type) != "Abs" or len(node.input) < 1 or len(node.output) < 1:
        return None
    x_name = str(node.input[0])
    y_name = str(node.output[0])
    x_dtype = _dtype_for_node_input(node=node, elem_type_map=elem_type_map, input_index=0, default=np.float32)
    if not _is_float_dtype(x_dtype):
        return None
    x_sq = _next_name(used_names, f"{node.name or 'Abs'}_sq")
    rewritten = [
        _make_node(
            used_names=used_names,
            op_type="Mul",
            inputs=[x_name, x_name],
            outputs=[x_sq],
            name_base=node.name or "Abs",
        ),
        _make_node(
            used_names=used_names,
            op_type="Sqrt",
            inputs=[x_sq],
            outputs=[y_name],
            name_base=node.name or "Abs",
        ),
    ]
    return rewritten, []


def _rewrite_neg(
    *,
    node: onnx.NodeProto,
    used_names: Set[str],
    elem_type_map: Dict[str, int],
) -> Optional[Tuple[List[onnx.NodeProto], List[onnx.TensorProto]]]:
    if str(node.op_type) != "Neg" or len(node.input) < 1 or len(node.output) < 1:
        return None
    x_name = str(node.input[0])
    y_name = str(node.output[0])
    x_dtype = _dtype_for_node_input(node=node, elem_type_map=elem_type_map, input_index=0, default=np.float32)
    neg_one = _make_scalar_const_initializer(
        used_names=used_names,
        name_base=f"{node.name or 'Neg'}_minus_one",
        value=-1,
        dtype=x_dtype,
    )
    rewritten = [
        _make_node(
            used_names=used_names,
            op_type="Mul",
            inputs=[x_name, neg_one.name],
            outputs=[y_name],
            name_base=node.name or "Neg",
        ),
    ]
    return rewritten, [neg_one]


def _rewrite_leaky_relu(
    *,
    node: onnx.NodeProto,
    used_names: Set[str],
    elem_type_map: Dict[str, int],
) -> Optional[Tuple[List[onnx.NodeProto], List[onnx.TensorProto]]]:
    if str(node.op_type) != "LeakyRelu" or len(node.input) < 1 or len(node.output) < 1:
        return None
    x_dtype = _dtype_for_node_input(node=node, elem_type_map=elem_type_map, input_index=0, default=np.float32)
    alpha = _node_attr_float(node, "alpha", 0.01)
    x_name = str(node.input[0])
    y_name = str(node.output[0])

    alpha_const = _make_scalar_const_initializer(
        used_names=used_names,
        name_base=f"{node.name or 'LeakyRelu'}_alpha",
        value=alpha,
        dtype=x_dtype,
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
    used_names: Set[str],
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


def _rewrite_hardswish(
    *,
    node: onnx.NodeProto,
    used_names: Set[str],
    elem_type_map: Dict[str, int],
) -> Optional[Tuple[List[onnx.NodeProto], List[onnx.TensorProto]]]:
    if str(node.op_type) != "HardSwish" or len(node.input) < 1 or len(node.output) < 1:
        return None
    x_name = str(node.input[0])
    y_name = str(node.output[0])
    x_dtype = _dtype_for_node_input(node=node, elem_type_map=elem_type_map, input_index=0, default=np.float32)
    if not _is_float_dtype(x_dtype):
        return None

    c3 = _make_scalar_const_initializer(
        used_names=used_names,
        name_base=f"{node.name or 'HardSwish'}_c3",
        value=3.0,
        dtype=x_dtype,
    )
    c6 = _make_scalar_const_initializer(
        used_names=used_names,
        name_base=f"{node.name or 'HardSwish'}_c6",
        value=6.0,
        dtype=x_dtype,
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


def _rewrite_pow(
    *,
    node: onnx.NodeProto,
    used_names: Set[str],
    const_map: Dict[str, np.ndarray],
) -> Optional[Tuple[List[onnx.NodeProto], List[onnx.TensorProto]]]:
    exponent = _try_get_pow_exponent(node=node, const_map=const_map)
    if exponent is None or len(node.input) < 1 or len(node.output) < 1:
        return None
    x_name = str(node.input[0])
    y_name = str(node.output[0])

    if abs(exponent - 1.0) <= 1e-6:
        return [
            _make_node(
                used_names=used_names,
                op_type="Identity",
                inputs=[x_name],
                outputs=[y_name],
                name_base=node.name or "Pow",
            )
        ], []

    if abs(exponent - 0.5) <= 1e-6:
        return [
            _make_node(
                used_names=used_names,
                op_type="Sqrt",
                inputs=[x_name],
                outputs=[y_name],
                name_base=node.name or "Pow",
            )
        ], []

    if abs(exponent - 2.0) <= 1e-6:
        return [
            _make_node(
                used_names=used_names,
                op_type="Mul",
                inputs=[x_name, x_name],
                outputs=[y_name],
                name_base=node.name or "Pow",
            )
        ], []

    exponent_int = int(round(float(exponent)))
    if exponent_int >= 2 and abs(float(exponent) - float(exponent_int)) <= 1e-6:
        rewritten: List[onnx.NodeProto] = []
        current = x_name
        for i in range(exponent_int - 1):
            out = y_name if i == exponent_int - 2 else _next_name(used_names, f"{node.name or 'Pow'}_mul_{i}")
            rewritten.append(
                _make_node(
                    used_names=used_names,
                    op_type="Mul",
                    inputs=[current, x_name],
                    outputs=[out],
                    name_base=node.name or "Pow",
                )
            )
            current = out
        return rewritten, []

    return None


def _rewrite_atan(
    *,
    node: onnx.NodeProto,
    used_names: Set[str],
    elem_type_map: Dict[str, int],
) -> Optional[Tuple[List[onnx.NodeProto], List[onnx.TensorProto]]]:
    if str(node.op_type) != "Atan" or len(node.input) < 1 or len(node.output) < 1:
        return None
    x_name = str(node.input[0])
    y_name = str(node.output[0])
    x_dtype = _dtype_for_node_input(node=node, elem_type_map=elem_type_map, input_index=0, default=np.float32)
    if not _is_float_dtype(x_dtype):
        return None

    one = _make_scalar_const_initializer(
        used_names=used_names,
        name_base=f"{node.name or 'Atan'}_one",
        value=1.0,
        dtype=x_dtype,
    )
    rewritten = [
        _make_node(
            used_names=used_names,
            op_type="Atan2",
            inputs=[x_name, one.name],
            outputs=[y_name],
            name_base=node.name or "Atan",
        )
    ]
    return rewritten, [one]


def _rewrite_asin(
    *,
    node: onnx.NodeProto,
    used_names: Set[str],
    elem_type_map: Dict[str, int],
) -> Optional[Tuple[List[onnx.NodeProto], List[onnx.TensorProto]]]:
    if str(node.op_type) != "Asin" or len(node.input) < 1 or len(node.output) < 1:
        return None
    x_name = str(node.input[0])
    y_name = str(node.output[0])
    x_dtype = _dtype_for_node_input(node=node, elem_type_map=elem_type_map, input_index=0, default=np.float32)
    if not _is_float_dtype(x_dtype):
        return None

    consts: Dict[str, onnx.TensorProto] = {}
    for name_base, value in [
        ("zero", 0.0),
        ("minus_one", -1.0),
        ("one", 1.0),
        ("two", 2.0),
        ("half_pi", 3.14159265358979 * 0.5),
        ("c0", -0.0187293),
        ("c1", 0.0742610),
        ("c2", -0.2121144),
        ("c3", 1.5707288),
    ]:
        consts[name_base] = _make_scalar_const_initializer(
            used_names=used_names,
            name_base=f"{node.name or 'Asin'}_{name_base}",
            value=value,
            dtype=x_dtype,
        )

    rewritten: List[onnx.NodeProto] = []
    x_abs = _append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Mul", a=x_name, b=x_name, out_base=f"{node.name or 'Asin'}_x_sq")
    x_abs_sqrt = _next_name(used_names, f"{node.name or 'Asin'}_x_abs")
    rewritten.append(_make_unary_node(used_names=used_names, node=node, op_type="Sqrt", input_name=x_abs, output_name=x_abs_sqrt))

    x_min_zero = _append_binary(
        nodes=rewritten,
        used_names=used_names,
        node=node,
        op_type="Min",
        a=x_name,
        b=consts["zero"].name,
        out_base=f"{node.name or 'Asin'}_x_min_zero",
    )
    neg_num = _append_binary(
        nodes=rewritten,
        used_names=used_names,
        node=node,
        op_type="Mul",
        a=x_min_zero,
        b=consts["minus_one"].name,
        out_base=f"{node.name or 'Asin'}_neg_num",
    )
    neg = _append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Div", a=neg_num, b=x_abs_sqrt, out_base=f"{node.name or 'Asin'}_neg")

    y0 = _append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Mul", a=consts["c0"].name, b=x_abs_sqrt, out_base=f"{node.name or 'Asin'}_y0")
    y1 = _append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Add", a=y0, b=consts["c1"].name, out_base=f"{node.name or 'Asin'}_y1")
    y2 = _append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Mul", a=y1, b=x_abs_sqrt, out_base=f"{node.name or 'Asin'}_y2")
    y3 = _append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Add", a=y2, b=consts["c2"].name, out_base=f"{node.name or 'Asin'}_y3")
    y4 = _append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Mul", a=y3, b=x_abs_sqrt, out_base=f"{node.name or 'Asin'}_y4")
    y5 = _append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Add", a=y4, b=consts["c3"].name, out_base=f"{node.name or 'Asin'}_y5")

    one_minus_x = _append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Sub", a=consts["one"].name, b=x_abs_sqrt, out_base=f"{node.name or 'Asin'}_one_minus_x")
    sqrt_term = _next_name(used_names, f"{node.name or 'Asin'}_sqrt_term")
    rewritten.append(_make_unary_node(used_names=used_names, node=node, op_type="Sqrt", input_name=one_minus_x, output_name=sqrt_term))
    mul_term = _append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Mul", a=sqrt_term, b=y5, out_base=f"{node.name or 'Asin'}_mul_term")
    y = _append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Sub", a=consts["half_pi"].name, b=mul_term, out_base=f"{node.name or 'Asin'}_y")

    two_neg = _append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Mul", a=consts["two"].name, b=neg, out_base=f"{node.name or 'Asin'}_two_neg")
    corr = _append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Mul", a=two_neg, b=y, out_base=f"{node.name or 'Asin'}_corr")
    rewritten.append(
        _make_node(
            used_names=used_names,
            op_type="Sub",
            inputs=[y, corr],
            outputs=[y_name],
            name_base=node.name or "Asin",
        )
    )

    return rewritten, list(consts.values())


def _rewrite_acos(
    *,
    node: onnx.NodeProto,
    used_names: Set[str],
    elem_type_map: Dict[str, int],
) -> Optional[Tuple[List[onnx.NodeProto], List[onnx.TensorProto]]]:
    if str(node.op_type) != "Acos" or len(node.input) < 1 or len(node.output) < 1:
        return None
    x_name = str(node.input[0])
    y_name = str(node.output[0])
    x_dtype = _dtype_for_node_input(node=node, elem_type_map=elem_type_map, input_index=0, default=np.float32)
    if not _is_float_dtype(x_dtype):
        return None

    consts: Dict[str, onnx.TensorProto] = {}
    for name_base, value in [
        ("zero", 0.0),
        ("minus_one", -1.0),
        ("one", 1.0),
        ("two", 2.0),
        ("pi", 3.14159265358979),
        ("c0", -0.0187293),
        ("c1", 0.0742610),
        ("c2", -0.2121144),
        ("c3", 1.5707288),
    ]:
        consts[name_base] = _make_scalar_const_initializer(
            used_names=used_names,
            name_base=f"{node.name or 'Acos'}_{name_base}",
            value=value,
            dtype=x_dtype,
        )

    rewritten: List[onnx.NodeProto] = []
    x_sq = _append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Mul", a=x_name, b=x_name, out_base=f"{node.name or 'Acos'}_x_sq")
    x_abs = _next_name(used_names, f"{node.name or 'Acos'}_x_abs")
    rewritten.append(_make_unary_node(used_names=used_names, node=node, op_type="Sqrt", input_name=x_sq, output_name=x_abs))

    x_min_zero = _append_binary(
        nodes=rewritten,
        used_names=used_names,
        node=node,
        op_type="Min",
        a=x_name,
        b=consts["zero"].name,
        out_base=f"{node.name or 'Acos'}_x_min_zero",
    )
    neg_num = _append_binary(
        nodes=rewritten,
        used_names=used_names,
        node=node,
        op_type="Mul",
        a=x_min_zero,
        b=consts["minus_one"].name,
        out_base=f"{node.name or 'Acos'}_neg_num",
    )
    neg = _append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Div", a=neg_num, b=x_abs, out_base=f"{node.name or 'Acos'}_neg")

    y0 = _append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Mul", a=consts["c0"].name, b=x_abs, out_base=f"{node.name or 'Acos'}_y0")
    y1 = _append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Add", a=y0, b=consts["c1"].name, out_base=f"{node.name or 'Acos'}_y1")
    y2 = _append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Mul", a=y1, b=x_abs, out_base=f"{node.name or 'Acos'}_y2")
    y3 = _append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Add", a=y2, b=consts["c2"].name, out_base=f"{node.name or 'Acos'}_y3")
    y4 = _append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Mul", a=y3, b=x_abs, out_base=f"{node.name or 'Acos'}_y4")
    y5 = _append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Add", a=y4, b=consts["c3"].name, out_base=f"{node.name or 'Acos'}_y5")

    one_minus_x = _append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Sub", a=consts["one"].name, b=x_abs, out_base=f"{node.name or 'Acos'}_one_minus_x")
    sqrt_term = _next_name(used_names, f"{node.name or 'Acos'}_sqrt_term")
    rewritten.append(_make_unary_node(used_names=used_names, node=node, op_type="Sqrt", input_name=one_minus_x, output_name=sqrt_term))
    y6 = _append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Mul", a=y5, b=sqrt_term, out_base=f"{node.name or 'Acos'}_y6")

    two_neg = _append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Mul", a=consts["two"].name, b=neg, out_base=f"{node.name or 'Acos'}_two_neg")
    one_minus_two_neg = _append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Sub", a=consts["one"].name, b=two_neg, out_base=f"{node.name or 'Acos'}_one_minus_two_neg")
    y7 = _append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Mul", a=y6, b=one_minus_two_neg, out_base=f"{node.name or 'Acos'}_y7")

    neg_pi = _append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Mul", a=neg, b=consts["pi"].name, out_base=f"{node.name or 'Acos'}_neg_pi")
    rewritten.append(
        _make_node(
            used_names=used_names,
            op_type="Add",
            inputs=[neg_pi, y7],
            outputs=[y_name],
            name_base=node.name or "Acos",
        )
    )
    return rewritten, list(consts.values())


def _rewrite_erf(
    *,
    node: onnx.NodeProto,
    used_names: Set[str],
    elem_type_map: Dict[str, int],
) -> Optional[Tuple[List[onnx.NodeProto], List[onnx.TensorProto]]]:
    if str(node.op_type) != "Erf" or len(node.input) < 1 or len(node.output) < 1:
        return None
    x_name = str(node.input[0])
    y_name = str(node.output[0])
    x_dtype = _dtype_for_node_input(node=node, elem_type_map=elem_type_map, input_index=0, default=np.float32)
    if not _is_float_dtype(x_dtype):
        return None

    consts: Dict[str, onnx.TensorProto] = {}
    for name_base, value in [
        ("eps", 1e-11),
        ("one", 1.0),
        ("minus_one", -1.0),
        ("a1", 0.254829592),
        ("a2", -0.284496736),
        ("a3", 1.421413741),
        ("a4", -1.453152027),
        ("a5", 1.061405429),
        ("p", 0.3275911),
    ]:
        consts[name_base] = _make_scalar_const_initializer(
            used_names=used_names,
            name_base=f"{node.name or 'Erf'}_{name_base}",
            value=value,
            dtype=x_dtype,
        )

    rewritten: List[onnx.NodeProto] = []

    x_abs = _next_name(used_names, f"{node.name or 'Erf'}_x_abs")
    rewritten.append(_make_unary_node(used_names=used_names, node=node, op_type="Abs", input_name=x_name, output_name=x_abs))

    x_abs_eps = _append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Add", a=x_abs, b=consts["eps"].name, out_base=f"{node.name or 'Erf'}_x_abs_eps")
    sign = _append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Div", a=x_name, b=x_abs_eps, out_base=f"{node.name or 'Erf'}_sign")

    p_mul_abs = _append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Mul", a=consts["p"].name, b=x_abs, out_base=f"{node.name or 'Erf'}_p_mul_abs")
    one_plus = _append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Add", a=consts["one"].name, b=p_mul_abs, out_base=f"{node.name or 'Erf'}_one_plus")
    t = _append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Div", a=consts["one"].name, b=one_plus, out_base=f"{node.name or 'Erf'}_t")

    a5t = _append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Mul", a=consts["a5"].name, b=t, out_base=f"{node.name or 'Erf'}_a5t")
    s1 = _append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Add", a=a5t, b=consts["a4"].name, out_base=f"{node.name or 'Erf'}_s1")
    s2 = _append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Mul", a=s1, b=t, out_base=f"{node.name or 'Erf'}_s2")
    s3 = _append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Add", a=s2, b=consts["a3"].name, out_base=f"{node.name or 'Erf'}_s3")
    s4 = _append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Mul", a=s3, b=t, out_base=f"{node.name or 'Erf'}_s4")
    s5 = _append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Add", a=s4, b=consts["a2"].name, out_base=f"{node.name or 'Erf'}_s5")
    s6 = _append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Mul", a=s5, b=t, out_base=f"{node.name or 'Erf'}_s6")
    s7 = _append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Add", a=s6, b=consts["a1"].name, out_base=f"{node.name or 'Erf'}_s7")
    s8 = _append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Mul", a=s7, b=t, out_base=f"{node.name or 'Erf'}_s8")

    x_abs_sq = _append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Mul", a=x_abs, b=x_abs, out_base=f"{node.name or 'Erf'}_x_abs_sq")
    neg_x_abs_sq = _append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Mul", a=x_abs_sq, b=consts["minus_one"].name, out_base=f"{node.name or 'Erf'}_neg_x_abs_sq")
    exp_term = _next_name(used_names, f"{node.name or 'Erf'}_exp_term")
    rewritten.append(_make_unary_node(used_names=used_names, node=node, op_type="Exp", input_name=neg_x_abs_sq, output_name=exp_term))

    poly_exp = _append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Mul", a=s8, b=exp_term, out_base=f"{node.name or 'Erf'}_poly_exp")
    y_core = _append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Sub", a=consts["one"].name, b=poly_exp, out_base=f"{node.name or 'Erf'}_y_core")
    rewritten.append(
        _make_node(
            used_names=used_names,
            op_type="Mul",
            inputs=[sign, y_core],
            outputs=[y_name],
            name_base=node.name or "Erf",
        )
    )

    return rewritten, list(consts.values())


def _rewrite_gelu(
    *,
    node: onnx.NodeProto,
    used_names: Set[str],
    elem_type_map: Dict[str, int],
) -> Optional[Tuple[List[onnx.NodeProto], List[onnx.TensorProto]]]:
    if str(node.op_type) != "Gelu" or len(node.input) < 1 or len(node.output) < 1:
        return None
    x_name = str(node.input[0])
    y_name = str(node.output[0])
    x_dtype = _dtype_for_node_input(node=node, elem_type_map=elem_type_map, input_index=0, default=np.float32)
    if not _is_float_dtype(x_dtype):
        return None

    # tanh-approximation to avoid erf dependency:
    # y = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
    consts: List[onnx.TensorProto] = []
    for name_base, value in [
        ("half", 0.5),
        ("one", 1.0),
        ("k", np.sqrt(2.0 / np.pi)),
        ("c", 0.044715),
    ]:
        consts.append(
            _make_scalar_const_initializer(
                used_names=used_names,
                name_base=f"{node.name or 'Gelu'}_{name_base}",
                value=value,
                dtype=x_dtype,
            )
        )
    c = {v.name.rsplit("_", 1)[-1]: v.name for v in consts}

    rewritten: List[onnx.NodeProto] = []
    x2 = _append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Mul", a=x_name, b=x_name, out_base=f"{node.name or 'Gelu'}_x2")
    x3 = _append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Mul", a=x2, b=x_name, out_base=f"{node.name or 'Gelu'}_x3")
    cx3 = _append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Mul", a=c["c"], b=x3, out_base=f"{node.name or 'Gelu'}_cx3")
    x_plus = _append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Add", a=x_name, b=cx3, out_base=f"{node.name or 'Gelu'}_x_plus")
    k_term = _append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Mul", a=c["k"], b=x_plus, out_base=f"{node.name or 'Gelu'}_k_term")
    tanh_out = _next_name(used_names, f"{node.name or 'Gelu'}_tanh")
    rewritten.append(_make_unary_node(used_names=used_names, node=node, op_type="Tanh", input_name=k_term, output_name=tanh_out))
    one_plus = _append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Add", a=c["one"], b=tanh_out, out_base=f"{node.name or 'Gelu'}_one_plus")
    half_x = _append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Mul", a=c["half"], b=x_name, out_base=f"{node.name or 'Gelu'}_half_x")
    rewritten.append(
        _make_node(
            used_names=used_names,
            op_type="Mul",
            inputs=[half_x, one_plus],
            outputs=[y_name],
            name_base=node.name or "Gelu",
        )
    )

    return rewritten, consts


def _rewrite_matmulinteger(
    *,
    node: onnx.NodeProto,
    used_names: Set[str],
    elem_type_map: Dict[str, int],
) -> Optional[Tuple[List[onnx.NodeProto], List[onnx.TensorProto]]]:
    if str(node.op_type) != "MatMulInteger" or len(node.input) < 2 or len(node.output) < 1:
        return None

    a_name = str(node.input[0])
    b_name = str(node.input[1])
    y_name = str(node.output[0])

    float_to = int(onnx.TensorProto.FLOAT)
    output_elem = elem_type_map.get(y_name, int(onnx.TensorProto.INT32))
    if int(output_elem) not in {int(onnx.TensorProto.INT32), int(onnx.TensorProto.INT64)}:
        output_elem = int(onnx.TensorProto.INT32)

    rewritten: List[onnx.NodeProto] = []

    a_fp = _next_name(used_names, f"{node.name or 'MatMulInteger'}_a_fp")
    b_fp = _next_name(used_names, f"{node.name or 'MatMulInteger'}_b_fp")
    rewritten.append(
        _make_node(
            used_names=used_names,
            op_type="Cast",
            inputs=[a_name],
            outputs=[a_fp],
            name_base=node.name or "MatMulInteger",
            attrs={"to": float_to},
        )
    )
    rewritten.append(
        _make_node(
            used_names=used_names,
            op_type="Cast",
            inputs=[b_name],
            outputs=[b_fp],
            name_base=node.name or "MatMulInteger",
            attrs={"to": float_to},
        )
    )

    a_adj = a_fp
    b_adj = b_fp
    if len(node.input) >= 3 and str(node.input[2]) != "":
        a_zp = str(node.input[2])
        a_zp_fp = _next_name(used_names, f"{node.name or 'MatMulInteger'}_a_zp_fp")
        rewritten.append(
            _make_node(
                used_names=used_names,
                op_type="Cast",
                inputs=[a_zp],
                outputs=[a_zp_fp],
                name_base=node.name or "MatMulInteger",
                attrs={"to": float_to},
            )
        )
        a_adj = _next_name(used_names, f"{node.name or 'MatMulInteger'}_a_adj")
        rewritten.append(
            _make_node(
                used_names=used_names,
                op_type="Sub",
                inputs=[a_fp, a_zp_fp],
                outputs=[a_adj],
                name_base=node.name or "MatMulInteger",
            )
        )

    if len(node.input) >= 4 and str(node.input[3]) != "":
        b_zp = str(node.input[3])
        b_zp_fp = _next_name(used_names, f"{node.name or 'MatMulInteger'}_b_zp_fp")
        rewritten.append(
            _make_node(
                used_names=used_names,
                op_type="Cast",
                inputs=[b_zp],
                outputs=[b_zp_fp],
                name_base=node.name or "MatMulInteger",
                attrs={"to": float_to},
            )
        )
        b_adj = _next_name(used_names, f"{node.name or 'MatMulInteger'}_b_adj")
        rewritten.append(
            _make_node(
                used_names=used_names,
                op_type="Sub",
                inputs=[b_fp, b_zp_fp],
                outputs=[b_adj],
                name_base=node.name or "MatMulInteger",
            )
        )

    mm = _next_name(used_names, f"{node.name or 'MatMulInteger'}_mm")
    mm_round = _next_name(used_names, f"{node.name or 'MatMulInteger'}_mm_round")
    rewritten.append(
        _make_node(
            used_names=used_names,
            op_type="MatMul",
            inputs=[a_adj, b_adj],
            outputs=[mm],
            name_base=node.name or "MatMulInteger",
        )
    )
    rewritten.append(
        _make_node(
            used_names=used_names,
            op_type="Round",
            inputs=[mm],
            outputs=[mm_round],
            name_base=node.name or "MatMulInteger",
        )
    )
    rewritten.append(
        _make_node(
            used_names=used_names,
            op_type="Cast",
            inputs=[mm_round],
            outputs=[y_name],
            name_base=node.name or "MatMulInteger",
            attrs={"to": int(output_elem)},
        )
    )

    return rewritten, []


def _rewrite_gathernd(
    *,
    node: onnx.NodeProto,
    used_names: Set[str],
    shape_map: Dict[str, List[Optional[int]]],
    elem_type_map: Dict[str, int],
) -> Optional[Tuple[List[onnx.NodeProto], List[onnx.TensorProto]]]:
    if str(node.op_type) != "GatherND" or len(node.input) < 2 or len(node.output) < 1:
        return None

    batch_dims = _node_attr_int(node, "batch_dims", 0)
    if batch_dims != 0:
        return None

    params_name = str(node.input[0])
    indices_name = str(node.input[1])
    y_name = str(node.output[0])

    params_shape = shape_map.get(params_name, None)
    indices_shape = shape_map.get(indices_name, None)
    if params_shape is None or indices_shape is None:
        return None
    if len(indices_shape) < 1:
        return None
    idx_dims = indices_shape[-1]
    if idx_dims is None:
        return None
    idx_dims = int(idx_dims)
    if idx_dims <= 0 or idx_dims > len(params_shape):
        return None
    if any(v is None for v in params_shape):
        return None
    if any(v is None for v in indices_shape):
        return None

    params_shape_i = [int(v) for v in params_shape if v is not None]
    indices_shape_i = [int(v) for v in indices_shape if v is not None]
    gather_shape = params_shape_i[idx_dims:]

    axis_step = np.cumprod(np.asarray(params_shape_i[:idx_dims], dtype=np.int64), dtype=np.int64)
    if axis_step.size > 0:
        axis_step = np.roll(axis_step, 1)
        axis_step[0] = 1
        axis_step = axis_step[::-1]
    else:
        axis_step = np.asarray([], dtype=np.int64)

    output_shape = indices_shape_i[:-1] + gather_shape

    indices_dtype = _dtype_for_tensor_name(
        tensor_name=indices_name,
        elem_type_map=elem_type_map,
        default=np.int64,
    )
    try:
        axis_step_arr = axis_step.astype(np.dtype(indices_dtype), copy=False)
    except Exception:
        axis_step_arr = axis_step.astype(np.int64, copy=False)

    shape_params_flat = _make_const_initializer(
        used_names=used_names,
        name_base=f"{node.name or 'GatherND'}_params_flat_shape",
        value=np.asarray([-1] + gather_shape, dtype=np.int64),
    )
    axis_step_init = _make_const_initializer(
        used_names=used_names,
        name_base=f"{node.name or 'GatherND'}_axis_step",
        value=axis_step_arr,
    )
    reduce_axis = _make_const_initializer(
        used_names=used_names,
        name_base=f"{node.name or 'GatherND'}_reduce_axis",
        value=np.asarray([-1], dtype=np.int64),
    )
    output_shape_init = _make_const_initializer(
        used_names=used_names,
        name_base=f"{node.name or 'GatherND'}_output_shape",
        value=np.asarray(output_shape, dtype=np.int64),
    )

    rewritten: List[onnx.NodeProto] = []
    params_flat = _next_name(used_names, f"{node.name or 'GatherND'}_params_flat")
    idx_mul = _next_name(used_names, f"{node.name or 'GatherND'}_idx_mul")
    indices_flat = _next_name(used_names, f"{node.name or 'GatherND'}_indices_flat")
    gathered_flat = _next_name(used_names, f"{node.name or 'GatherND'}_gathered_flat")

    rewritten.append(
        _make_node(
            used_names=used_names,
            op_type="Reshape",
            inputs=[params_name, shape_params_flat.name],
            outputs=[params_flat],
            name_base=node.name or "GatherND",
        )
    )
    rewritten.append(
        _make_node(
            used_names=used_names,
            op_type="Mul",
            inputs=[indices_name, axis_step_init.name],
            outputs=[idx_mul],
            name_base=node.name or "GatherND",
        )
    )
    rewritten.append(
        _make_node(
            used_names=used_names,
            op_type="ReduceSum",
            inputs=[idx_mul, reduce_axis.name],
            outputs=[indices_flat],
            name_base=node.name or "GatherND",
            attrs={"keepdims": 0},
        )
    )
    rewritten.append(
        _make_node(
            used_names=used_names,
            op_type="Gather",
            inputs=[params_flat, indices_flat],
            outputs=[gathered_flat],
            name_base=node.name or "GatherND",
            attrs={"axis": 0},
        )
    )
    rewritten.append(
        _make_node(
            used_names=used_names,
            op_type="Reshape",
            inputs=[gathered_flat, output_shape_init.name],
            outputs=[y_name],
            name_base=node.name or "GatherND",
        )
    )

    return rewritten, [shape_params_flat, axis_step_init, reduce_axis, output_shape_init]


def _make_slice_elem(
    *,
    used_names: Set[str],
    node: onnx.NodeProto,
    input_name: str,
    i: int,
    j: int,
    dtype: Any,
    tag: str,
) -> Tuple[str, List[onnx.NodeProto], List[onnx.TensorProto]]:
    starts = _make_const_initializer(
        used_names=used_names,
        name_base=f"{node.name or 'Inverse'}_{tag}_starts",
        value=np.asarray([i, j], dtype=np.int64),
    )
    ends = _make_const_initializer(
        used_names=used_names,
        name_base=f"{node.name or 'Inverse'}_{tag}_ends",
        value=np.asarray([i + 1, j + 1], dtype=np.int64),
    )
    axes = _make_const_initializer(
        used_names=used_names,
        name_base=f"{node.name or 'Inverse'}_{tag}_axes",
        value=np.asarray([-2, -1], dtype=np.int64),
    )
    steps = _make_const_initializer(
        used_names=used_names,
        name_base=f"{node.name or 'Inverse'}_{tag}_steps",
        value=np.asarray([1, 1], dtype=np.int64),
    )
    out = _next_name(used_names, f"{node.name or 'Inverse'}_{tag}")
    nodes = [
        _make_node(
            used_names=used_names,
            op_type="Slice",
            inputs=[input_name, starts.name, ends.name, axes.name, steps.name],
            outputs=[out],
            name_base=node.name or "Inverse",
        )
    ]
    inits = [starts, ends, axes, steps]
    return out, nodes, inits


def _build_safe_denominator(
    *,
    used_names: Set[str],
    node: onnx.NodeProto,
    det_name: str,
    dtype: Any,
    eps_name_base: str,
) -> Tuple[str, List[onnx.NodeProto], List[onnx.TensorProto]]:
    eps = _make_scalar_const_initializer(
        used_names=used_names,
        name_base=eps_name_base,
        value=1.0e-3 if np.dtype(dtype) == np.float16 else 1.0e-6,
        dtype=dtype,
    )
    det_abs = _next_name(used_names, f"{node.name or 'Inverse'}_det_abs")
    cond = _next_name(used_names, f"{node.name or 'Inverse'}_det_small")
    det_ones = _next_name(used_names, f"{node.name or 'Inverse'}_det_ones")
    det_safe = _next_name(used_names, f"{node.name or 'Inverse'}_det_safe")
    nodes = [
        _make_unary_node(used_names=used_names, node=node, op_type="Abs", input_name=det_name, output_name=det_abs),
        _make_node(
            used_names=used_names,
            op_type="Less",
            inputs=[det_abs, eps.name],
            outputs=[cond],
            name_base=node.name or "Inverse",
        ),
        _make_unary_node(used_names=used_names, node=node, op_type="ConstantOfShape", input_name=det_name, output_name=det_ones),
    ]
    # ConstantOfShape requires shape input; create via Shape(det)
    det_shape = _next_name(used_names, f"{node.name or 'Inverse'}_det_shape")
    nodes[2] = _make_unary_node(used_names=used_names, node=node, op_type="Shape", input_name=det_name, output_name=det_shape)
    ones_init = _make_const_initializer(
        used_names=used_names,
        name_base=f"{node.name or 'Inverse'}_one_scalar",
        value=np.asarray([1.0], dtype=dtype),
    )
    nodes.append(
        _make_node(
            used_names=used_names,
            op_type="ConstantOfShape",
            inputs=[det_shape],
            outputs=[det_ones],
            name_base=node.name or "Inverse",
            attrs={"value": ones_init},
        )
    )
    nodes.append(
        _make_node(
            used_names=used_names,
            op_type="Where",
            inputs=[cond, det_ones, det_name],
            outputs=[det_safe],
            name_base=node.name or "Inverse",
        )
    )
    return det_safe, nodes, [eps]


def _rewrite_inverse_2x2(
    *,
    node: onnx.NodeProto,
    used_names: Set[str],
    dtype: Any,
) -> Tuple[List[onnx.NodeProto], List[onnx.TensorProto]]:
    x_name = str(node.input[0])
    y_name = str(node.output[0])

    rewritten: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    a00, n, i = _make_slice_elem(used_names=used_names, node=node, input_name=x_name, i=0, j=0, dtype=dtype, tag="a00")
    rewritten.extend(n)
    inits.extend(i)
    a01, n, i = _make_slice_elem(used_names=used_names, node=node, input_name=x_name, i=0, j=1, dtype=dtype, tag="a01")
    rewritten.extend(n)
    inits.extend(i)
    a10, n, i = _make_slice_elem(used_names=used_names, node=node, input_name=x_name, i=1, j=0, dtype=dtype, tag="a10")
    rewritten.extend(n)
    inits.extend(i)
    a11, n, i = _make_slice_elem(used_names=used_names, node=node, input_name=x_name, i=1, j=1, dtype=dtype, tag="a11")
    rewritten.extend(n)
    inits.extend(i)

    a00a11 = _append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Mul", a=a00, b=a11, out_base=f"{node.name or 'Inverse'}_a00a11")
    a01a10 = _append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Mul", a=a01, b=a10, out_base=f"{node.name or 'Inverse'}_a01a10")
    det = _append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Sub", a=a00a11, b=a01a10, out_base=f"{node.name or 'Inverse'}_det")

    minus_one = _make_scalar_const_initializer(
        used_names=used_names,
        name_base=f"{node.name or 'Inverse'}_minus_one",
        value=-1.0,
        dtype=dtype,
    )
    inits.append(minus_one)
    neg_a01 = _append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Mul", a=a01, b=minus_one.name, out_base=f"{node.name or 'Inverse'}_neg_a01")
    neg_a10 = _append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Mul", a=a10, b=minus_one.name, out_base=f"{node.name or 'Inverse'}_neg_a10")

    row0 = _next_name(used_names, f"{node.name or 'Inverse'}_row0")
    row1 = _next_name(used_names, f"{node.name or 'Inverse'}_row1")
    adj = _next_name(used_names, f"{node.name or 'Inverse'}_adj")
    rewritten.append(
        _make_node(
            used_names=used_names,
            op_type="Concat",
            inputs=[a11, neg_a01],
            outputs=[row0],
            name_base=node.name or "Inverse",
            attrs={"axis": -1},
        )
    )
    rewritten.append(
        _make_node(
            used_names=used_names,
            op_type="Concat",
            inputs=[neg_a10, a00],
            outputs=[row1],
            name_base=node.name or "Inverse",
            attrs={"axis": -1},
        )
    )
    rewritten.append(
        _make_node(
            used_names=used_names,
            op_type="Concat",
            inputs=[row0, row1],
            outputs=[adj],
            name_base=node.name or "Inverse",
            attrs={"axis": -2},
        )
    )

    det_safe, safe_nodes, safe_inits = _build_safe_denominator(
        used_names=used_names,
        node=node,
        det_name=det,
        dtype=dtype,
        eps_name_base=f"{node.name or 'Inverse'}_eps",
    )
    rewritten.extend(safe_nodes)
    inits.extend(safe_inits)

    rewritten.append(
        _make_node(
            used_names=used_names,
            op_type="Div",
            inputs=[adj, det_safe],
            outputs=[y_name],
            name_base=node.name or "Inverse",
        )
    )

    return rewritten, inits


def _rewrite_inverse_3x3(
    *,
    node: onnx.NodeProto,
    used_names: Set[str],
    dtype: Any,
) -> Tuple[List[onnx.NodeProto], List[onnx.TensorProto]]:
    x_name = str(node.input[0])
    y_name = str(node.output[0])

    rewritten: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    elems: Dict[str, str] = {}
    for i in range(3):
        for j in range(3):
            t, n, init = _make_slice_elem(
                used_names=used_names,
                node=node,
                input_name=x_name,
                i=i,
                j=j,
                dtype=dtype,
                tag=f"a{i}{j}",
            )
            elems[f"a{i}{j}"] = t
            rewritten.extend(n)
            inits.extend(init)

    c00 = _append_binary(
        nodes=rewritten,
        used_names=used_names,
        node=node,
        op_type="Sub",
        a=_append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Mul", a=elems["a11"], b=elems["a22"], out_base=f"{node.name or 'Inverse'}_a11a22"),
        b=_append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Mul", a=elems["a12"], b=elems["a21"], out_base=f"{node.name or 'Inverse'}_a12a21"),
        out_base=f"{node.name or 'Inverse'}_c00",
    )
    c01 = _append_binary(
        nodes=rewritten,
        used_names=used_names,
        node=node,
        op_type="Sub",
        a=_append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Mul", a=elems["a12"], b=elems["a20"], out_base=f"{node.name or 'Inverse'}_a12a20"),
        b=_append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Mul", a=elems["a10"], b=elems["a22"], out_base=f"{node.name or 'Inverse'}_a10a22"),
        out_base=f"{node.name or 'Inverse'}_c01",
    )
    c02 = _append_binary(
        nodes=rewritten,
        used_names=used_names,
        node=node,
        op_type="Sub",
        a=_append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Mul", a=elems["a10"], b=elems["a21"], out_base=f"{node.name or 'Inverse'}_a10a21"),
        b=_append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Mul", a=elems["a11"], b=elems["a20"], out_base=f"{node.name or 'Inverse'}_a11a20"),
        out_base=f"{node.name or 'Inverse'}_c02",
    )

    det0 = _append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Mul", a=elems["a00"], b=c00, out_base=f"{node.name or 'Inverse'}_det0")
    det1 = _append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Mul", a=elems["a01"], b=c01, out_base=f"{node.name or 'Inverse'}_det1")
    det2 = _append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Mul", a=elems["a02"], b=c02, out_base=f"{node.name or 'Inverse'}_det2")
    det01 = _append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Add", a=det0, b=det1, out_base=f"{node.name or 'Inverse'}_det01")
    det = _append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Add", a=det01, b=det2, out_base=f"{node.name or 'Inverse'}_det")

    adj00 = c00
    adj01 = _append_binary(
        nodes=rewritten,
        used_names=used_names,
        node=node,
        op_type="Sub",
        a=_append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Mul", a=elems["a02"], b=elems["a21"], out_base=f"{node.name or 'Inverse'}_a02a21"),
        b=_append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Mul", a=elems["a01"], b=elems["a22"], out_base=f"{node.name or 'Inverse'}_a01a22"),
        out_base=f"{node.name or 'Inverse'}_adj01",
    )
    adj02 = _append_binary(
        nodes=rewritten,
        used_names=used_names,
        node=node,
        op_type="Sub",
        a=_append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Mul", a=elems["a01"], b=elems["a12"], out_base=f"{node.name or 'Inverse'}_a01a12"),
        b=_append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Mul", a=elems["a02"], b=elems["a11"], out_base=f"{node.name or 'Inverse'}_a02a11"),
        out_base=f"{node.name or 'Inverse'}_adj02",
    )
    adj10 = c01
    adj11 = _append_binary(
        nodes=rewritten,
        used_names=used_names,
        node=node,
        op_type="Sub",
        a=_append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Mul", a=elems["a00"], b=elems["a22"], out_base=f"{node.name or 'Inverse'}_a00a22"),
        b=_append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Mul", a=elems["a02"], b=elems["a20"], out_base=f"{node.name or 'Inverse'}_a02a20"),
        out_base=f"{node.name or 'Inverse'}_adj11",
    )
    adj12 = _append_binary(
        nodes=rewritten,
        used_names=used_names,
        node=node,
        op_type="Sub",
        a=_append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Mul", a=elems["a02"], b=elems["a10"], out_base=f"{node.name or 'Inverse'}_a02a10"),
        b=_append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Mul", a=elems["a00"], b=elems["a12"], out_base=f"{node.name or 'Inverse'}_a00a12"),
        out_base=f"{node.name or 'Inverse'}_adj12",
    )
    adj20 = c02
    adj21 = _append_binary(
        nodes=rewritten,
        used_names=used_names,
        node=node,
        op_type="Sub",
        a=_append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Mul", a=elems["a01"], b=elems["a20"], out_base=f"{node.name or 'Inverse'}_a01a20"),
        b=_append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Mul", a=elems["a00"], b=elems["a21"], out_base=f"{node.name or 'Inverse'}_a00a21"),
        out_base=f"{node.name or 'Inverse'}_adj21",
    )
    adj22 = _append_binary(
        nodes=rewritten,
        used_names=used_names,
        node=node,
        op_type="Sub",
        a=_append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Mul", a=elems["a00"], b=elems["a11"], out_base=f"{node.name or 'Inverse'}_a00a11"),
        b=_append_binary(nodes=rewritten, used_names=used_names, node=node, op_type="Mul", a=elems["a01"], b=elems["a10"], out_base=f"{node.name or 'Inverse'}_a01a10"),
        out_base=f"{node.name or 'Inverse'}_adj22",
    )

    row0 = _next_name(used_names, f"{node.name or 'Inverse'}_row0")
    row1 = _next_name(used_names, f"{node.name or 'Inverse'}_row1")
    row2 = _next_name(used_names, f"{node.name or 'Inverse'}_row2")
    adj = _next_name(used_names, f"{node.name or 'Inverse'}_adj")

    rewritten.append(_make_node(used_names=used_names, op_type="Concat", inputs=[adj00, adj01, adj02], outputs=[row0], name_base=node.name or "Inverse", attrs={"axis": -1}))
    rewritten.append(_make_node(used_names=used_names, op_type="Concat", inputs=[adj10, adj11, adj12], outputs=[row1], name_base=node.name or "Inverse", attrs={"axis": -1}))
    rewritten.append(_make_node(used_names=used_names, op_type="Concat", inputs=[adj20, adj21, adj22], outputs=[row2], name_base=node.name or "Inverse", attrs={"axis": -1}))
    rewritten.append(_make_node(used_names=used_names, op_type="Concat", inputs=[row0, row1, row2], outputs=[adj], name_base=node.name or "Inverse", attrs={"axis": -2}))

    det_safe, safe_nodes, safe_inits = _build_safe_denominator(
        used_names=used_names,
        node=node,
        det_name=det,
        dtype=dtype,
        eps_name_base=f"{node.name or 'Inverse'}_eps",
    )
    rewritten.extend(safe_nodes)
    inits.extend(safe_inits)

    rewritten.append(
        _make_node(
            used_names=used_names,
            op_type="Div",
            inputs=[adj, det_safe],
            outputs=[y_name],
            name_base=node.name or "Inverse",
        )
    )
    return rewritten, inits


def _rewrite_inverse(
    *,
    node: onnx.NodeProto,
    used_names: Set[str],
    elem_type_map: Dict[str, int],
    shape_map: Dict[str, List[Optional[int]]],
) -> Optional[Tuple[List[onnx.NodeProto], List[onnx.TensorProto]]]:
    if str(node.op_type) != "Inverse" or len(node.input) < 1 or len(node.output) < 1:
        return None

    x_name = str(node.input[0])
    x_shape = shape_map.get(x_name, None)
    if x_shape is None or len(x_shape) < 2:
        return None

    rows = x_shape[-2]
    cols = x_shape[-1]
    if rows is None or cols is None:
        return None
    if int(rows) != int(cols):
        return None

    x_dtype = _dtype_for_node_input(node=node, elem_type_map=elem_type_map, input_index=0, default=np.float32)
    if not _is_float_dtype(x_dtype):
        return None

    if int(rows) == 2:
        return _rewrite_inverse_2x2(node=node, used_names=used_names, dtype=x_dtype)
    if int(rows) == 3:
        return _rewrite_inverse_3x3(node=node, used_names=used_names, dtype=x_dtype)
    return None


def apply_pseudo_ops_wave1(onnx_graph: onnx.ModelProto) -> Dict[str, Any]:
    graph = onnx_graph.graph
    used_names = _collect_used_names(graph)
    const_map = _build_initializer_map(graph)
    elem_type_map = _build_elem_type_map(graph)
    shape_map = _build_tensor_shape_map(graph, const_map)

    requested_aliases = set(_REQUESTED_PSEUDO_OP_ALIASES or set())
    requested_unknown_aliases = sorted(
        list(
            {
                alias
                for alias in requested_aliases
                if alias not in _PSEUDO_OP_ALIAS_TO_ONNX_OP
            }
        )
    )

    keep_builtin_inverse = "inverse" in requested_aliases

    target_ops = set(_DEFAULT_TARGET_ONNX_OPS)
    if keep_builtin_inverse:
        target_ops.discard("Inverse")
    for alias in requested_aliases:
        if alias == "inverse":
            continue
        mapped = _PSEUDO_OP_ALIAS_TO_ONNX_OP.get(alias, None)
        if mapped is not None and mapped != "":
            target_ops.add(str(mapped))

    rewritten_nodes: List[onnx.NodeProto] = []
    new_initializers: List[onnx.TensorProto] = []
    matched_nodes = 0
    rewritten_count = 0
    matched_by_op: Dict[str, int] = {}
    rewritten_by_op: Dict[str, int] = {}

    for node in graph.node:
        node_op = str(node.op_type)
        replacement: Optional[Tuple[List[onnx.NodeProto], List[onnx.TensorProto]]] = None

        if node_op in target_ops:
            matched_nodes += 1
            matched_by_op[node_op] = int(matched_by_op.get(node_op, 0) + 1)
            if node_op == "Abs":
                replacement = _rewrite_abs(node=node, used_names=used_names, elem_type_map=elem_type_map)
            elif node_op == "Acos":
                replacement = _rewrite_acos(node=node, used_names=used_names, elem_type_map=elem_type_map)
            elif node_op == "Asin":
                replacement = _rewrite_asin(node=node, used_names=used_names, elem_type_map=elem_type_map)
            elif node_op == "Atan":
                replacement = _rewrite_atan(node=node, used_names=used_names, elem_type_map=elem_type_map)
            elif node_op == "Erf":
                replacement = _rewrite_erf(node=node, used_names=used_names, elem_type_map=elem_type_map)
            elif node_op == "GatherND":
                replacement = _rewrite_gathernd(
                    node=node,
                    used_names=used_names,
                    shape_map=shape_map,
                    elem_type_map=elem_type_map,
                )
            elif node_op == "Gelu":
                replacement = _rewrite_gelu(node=node, used_names=used_names, elem_type_map=elem_type_map)
            elif node_op == "HardSwish":
                replacement = _rewrite_hardswish(node=node, used_names=used_names, elem_type_map=elem_type_map)
            elif node_op == "Inverse":
                replacement = _rewrite_inverse(
                    node=node,
                    used_names=used_names,
                    elem_type_map=elem_type_map,
                    shape_map=shape_map,
                )
            elif node_op == "LeakyRelu":
                replacement = _rewrite_leaky_relu(node=node, used_names=used_names, elem_type_map=elem_type_map)
            elif node_op == "MatMulInteger":
                replacement = _rewrite_matmulinteger(node=node, used_names=used_names, elem_type_map=elem_type_map)
            elif node_op == "Neg":
                replacement = _rewrite_neg(node=node, used_names=used_names, elem_type_map=elem_type_map)
            elif node_op == "PRelu":
                replacement = _rewrite_prelu(node=node, used_names=used_names)
            elif node_op == "Pow":
                replacement = _rewrite_pow(node=node, used_names=used_names, const_map=const_map)

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

    msg_parts = [
        f"matched_by_op={matched_by_op}",
        f"rewritten_by_op={rewritten_by_op}",
    ]
    if keep_builtin_inverse:
        msg_parts.append("inverse=kept_builtin_by_request")
    if len(requested_unknown_aliases) > 0:
        msg_parts.append(f"ignored_requested_aliases={requested_unknown_aliases}")

    return {
        "matched_nodes": int(matched_nodes),
        "rewritten_nodes": int(rewritten_count),
        "changed": bool(rewritten_count > 0),
        "message": ", ".join(msg_parts),
    }


def register_pseudo_ops_wave1_rule() -> None:
    register_preprocess_rule(
        rule_id=PSEUDO_OPS_WAVE1_RULE_ID,
        callback=apply_pseudo_ops_wave1,
        overwrite=True,
    )
