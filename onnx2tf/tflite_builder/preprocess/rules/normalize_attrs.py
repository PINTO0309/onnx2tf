from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnx
from onnx import helper, numpy_helper

from onnx2tf.tflite_builder.preprocess.pipeline import register_preprocess_rule

NORMALIZE_ATTRS_RULE_ID = "normalize_attrs_a5"


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


def _make_const_initializer(
    *,
    used_names: set[str],
    name_base: str,
    value: np.ndarray,
) -> onnx.TensorProto:
    name = _next_name(used_names, name_base)
    return numpy_helper.from_array(np.asarray(value), name=name)


def _find_attr(node: onnx.NodeProto, name: str) -> Optional[onnx.AttributeProto]:
    for attr in node.attribute:
        if str(attr.name) == str(name):
            return attr
    return None


def _get_attr_int(node: onnx.NodeProto, name: str, default: int) -> int:
    attr = _find_attr(node, name)
    if attr is None:
        return int(default)
    if attr.type == onnx.AttributeProto.INT:
        return int(attr.i)
    if attr.type == onnx.AttributeProto.FLOAT:
        return int(attr.f)
    return int(default)


def _get_attr_ints(node: onnx.NodeProto, name: str) -> Optional[List[int]]:
    attr = _find_attr(node, name)
    if attr is None:
        return None
    if attr.type == onnx.AttributeProto.INTS:
        return [int(v) for v in attr.ints]
    if attr.type == onnx.AttributeProto.INT:
        return [int(attr.i)]
    return None


def _get_attr_str(node: onnx.NodeProto, name: str, default: str) -> str:
    attr = _find_attr(node, name)
    if attr is None:
        return str(default)
    if attr.type == onnx.AttributeProto.STRING:
        try:
            return str(attr.s.decode("utf-8"))
        except Exception:
            return str(default)
    return str(default)


def _set_int_attr(node: onnx.NodeProto, name: str, value: int) -> None:
    existing = _find_attr(node, name)
    if existing is not None:
        existing.type = onnx.AttributeProto.INT
        existing.i = int(value)
        return
    node.attribute.append(helper.make_attribute(name, int(value)))


def _set_ints_attr(node: onnx.NodeProto, name: str, values: Sequence[int]) -> None:
    existing = _find_attr(node, name)
    vals = [int(v) for v in values]
    if existing is not None:
        existing.type = onnx.AttributeProto.INTS
        del existing.ints[:]
        existing.ints.extend(vals)
        return
    node.attribute.append(helper.make_attribute(name, vals))


def _set_str_attr(node: onnx.NodeProto, name: str, value: str) -> None:
    existing = _find_attr(node, name)
    v = str(value)
    if existing is not None:
        existing.type = onnx.AttributeProto.STRING
        existing.s = v.encode("utf-8")
        return
    node.attribute.append(helper.make_attribute(name, v))


def _build_rank_map(onnx_graph: onnx.ModelProto) -> Dict[str, int]:
    shape_map: Dict[str, int] = {}

    def _set_from_vi(value_info: onnx.ValueInfoProto) -> None:
        if value_info.name == "":
            return
        if not value_info.type.HasField("tensor_type"):
            return
        tt = value_info.type.tensor_type
        if not tt.HasField("shape"):
            return
        shape_map[str(value_info.name)] = int(len(tt.shape.dim))

    for vi in onnx_graph.graph.input:
        _set_from_vi(vi)
    for vi in onnx_graph.graph.value_info:
        _set_from_vi(vi)
    for vi in onnx_graph.graph.output:
        _set_from_vi(vi)
    for init in onnx_graph.graph.initializer:
        arr = np.asarray(numpy_helper.to_array(init))
        shape_map[str(init.name)] = int(arr.ndim)
    return shape_map


def _normalize_axis(axis: int, rank: int) -> int:
    a = int(axis)
    if a < 0:
        a += int(rank)
    return int(a)


def _inverse_perm(perm: Sequence[int]) -> List[int]:
    inv = [-1] * len(perm)
    for i, p in enumerate(perm):
        inv[int(p)] = int(i)
    return [int(v) for v in inv]


def apply_normalize_attrs_rule(onnx_graph: onnx.ModelProto) -> Dict[str, Any]:
    graph = onnx_graph.graph
    used_names = _collect_used_names(graph)
    rank_map = _build_rank_map(onnx_graph)
    added_initializers: List[onnx.TensorProto] = []

    rewritten_nodes: List[onnx.NodeProto] = []
    matched_nodes = 0
    rewritten_count = 0
    inserted_nodes = 0
    details: Dict[str, int] = {
        "transpose_perm_input_added": 0,
        "axes_input_added": 0,
        "axis_normalized": 0,
        "pads_normalized": 0,
        "softmax_transpose_inserted": 0,
    }

    for node in graph.node:
        n = _copy_node(node)
        op = str(n.op_type)
        changed = False

        if op in {"Transpose", "ReduceMean", "ReduceSum", "Squeeze", "Unsqueeze", "Gather", "Softmax", "LpNormalization", "Conv", "AveragePool", "MaxPool", "SpaceToDepth"}:
            matched_nodes += 1

        if op in {"Conv", "AveragePool", "MaxPool"}:
            pads = _get_attr_ints(n, "pads")
            if pads is not None and len(pads) == 2:
                _set_ints_attr(n, "pads", [pads[0], pads[1], pads[0], pads[1]])
                changed = True
                details["pads_normalized"] += 1

        if op == "SpaceToDepth":
            _set_str_attr(n, "mode", _get_attr_str(n, "mode", "DCR").upper())
            block_size = _get_attr_int(n, "blocksize", _get_attr_int(n, "block_size", 0))
            if block_size > 0:
                _set_int_attr(n, "blocksize", int(block_size))
                changed = True

        if op in {"Gather", "Softmax", "LpNormalization"}:
            rank = int(rank_map.get(str(n.input[0]), -1)) if len(n.input) >= 1 else -1
            if rank > 0:
                axis = _get_attr_int(n, "axis", -1 if op != "Gather" else 0)
                norm_axis = _normalize_axis(axis, rank)
                if norm_axis != axis:
                    _set_int_attr(n, "axis", norm_axis)
                    changed = True
                    details["axis_normalized"] += 1

        if op in {"ReduceMean", "ReduceSum", "Squeeze", "Unsqueeze"} and len(n.input) == 1:
            axes = _get_attr_ints(n, "axes")
            if axes is not None:
                rank = int(rank_map.get(str(n.input[0]), -1))
                if rank > 0:
                    axes = [_normalize_axis(v, rank + (1 if op == "Unsqueeze" else 0)) for v in axes]
                axes_init = _make_const_initializer(
                    used_names=used_names,
                    name_base=f"{n.name or op}_axes",
                    value=np.asarray(axes, dtype=np.int64),
                )
                added_initializers.append(axes_init)
                n.input.append(axes_init.name)
                changed = True
                details["axes_input_added"] += 1

        if op == "Transpose" and len(n.input) == 1:
            perm = _get_attr_ints(n, "perm")
            rank = int(rank_map.get(str(n.input[0]), -1))
            if perm is None and rank > 0:
                perm = [int(v) for v in reversed(range(rank))]
            if perm is not None:
                if rank > 0:
                    perm = [_normalize_axis(v, rank) for v in perm]
                perm_init = _make_const_initializer(
                    used_names=used_names,
                    name_base=f"{n.name or 'Transpose'}_perm",
                    value=np.asarray(perm, dtype=np.int64),
                )
                added_initializers.append(perm_init)
                n.input.append(perm_init.name)
                changed = True
                details["transpose_perm_input_added"] += 1

        if op == "Softmax":
            rank = int(rank_map.get(str(n.input[0]), -1)) if len(n.input) >= 1 else -1
            axis = _get_attr_int(n, "axis", 1)
            if rank > 1:
                axis = _normalize_axis(axis, rank)
                if axis != rank - 1:
                    perm_to_last = [int(v) for v in range(rank) if v != axis] + [int(axis)]
                    perm_from_last = _inverse_perm(perm_to_last)
                    pre_name = _next_name(used_names, f"{n.name or 'Softmax'}_pre")
                    post_name = str(n.output[0])

                    perm_to_last_init = _make_const_initializer(
                        used_names=used_names,
                        name_base=f"{n.name or 'Softmax'}_perm_to_last",
                        value=np.asarray(perm_to_last, dtype=np.int64),
                    )
                    perm_from_last_init = _make_const_initializer(
                        used_names=used_names,
                        name_base=f"{n.name or 'Softmax'}_perm_from_last",
                        value=np.asarray(perm_from_last, dtype=np.int64),
                    )
                    added_initializers.extend([perm_to_last_init, perm_from_last_init])

                    pre_node = helper.make_node(
                        "Transpose",
                        [str(n.input[0]), perm_to_last_init.name],
                        [pre_name],
                        name=_next_name(used_names, f"{n.name or 'Softmax'}_pre_transpose"),
                    )
                    soft_out = _next_name(used_names, f"{n.name or 'Softmax'}_softmax_out")
                    soft_node = helper.make_node(
                        "Softmax",
                        [pre_name],
                        [soft_out],
                        name=_next_name(used_names, f"{n.name or 'Softmax'}_normalized"),
                        axis=rank - 1,
                    )
                    post_node = helper.make_node(
                        "Transpose",
                        [soft_out, perm_from_last_init.name],
                        [post_name],
                        name=_next_name(used_names, f"{n.name or 'Softmax'}_post_transpose"),
                    )
                    rewritten_nodes.extend([pre_node, soft_node, post_node])
                    rewritten_count += 1
                    inserted_nodes += 2
                    details["softmax_transpose_inserted"] += 1
                    continue

        rewritten_nodes.append(n)
        if changed:
            rewritten_count += 1

    if len(added_initializers) > 0:
        graph.initializer.extend(added_initializers)
    if rewritten_count > 0:
        del graph.node[:]
        graph.node.extend(rewritten_nodes)

    return {
        "matched_nodes": int(matched_nodes),
        "rewritten_nodes": int(rewritten_count),
        "changed": bool(rewritten_count > 0),
        "message": f"details={details}, inserted_nodes={inserted_nodes}",
    }


def register_normalize_attrs_rule() -> None:
    register_preprocess_rule(
        rule_id=NORMALIZE_ATTRS_RULE_ID,
        callback=apply_normalize_attrs_rule,
        overwrite=True,
    )

