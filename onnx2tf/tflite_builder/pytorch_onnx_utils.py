from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnx


def _clear_onnx_graph_and_node_metadata_in_place(graph: onnx.GraphProto) -> None:
    del graph.metadata_props[:]
    for node in graph.node:
        del node.metadata_props[:]
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                _clear_onnx_graph_and_node_metadata_in_place(attr.g)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for subgraph in attr.graphs:
                    _clear_onnx_graph_and_node_metadata_in_place(subgraph)


def _onnx_node_maps(
    graph: onnx.GraphProto,
) -> Tuple[Dict[str, onnx.NodeProto], Dict[str, List[onnx.NodeProto]]]:
    producer_map: Dict[str, onnx.NodeProto] = {}
    consumer_map: Dict[str, List[onnx.NodeProto]] = {}
    for node in graph.node:
        for output_name in node.output:
            producer_map[str(output_name)] = node
        for input_name in node.input:
            consumer_map.setdefault(str(input_name), []).append(node)
    return producer_map, consumer_map


def _onnx_node_attr(node: onnx.NodeProto, name: str) -> Optional[Any]:
    for attr in node.attribute:
        if attr.name == name:
            return onnx.helper.get_attribute_value(attr)
    return None


def _onnx_set_node_attr(node: onnx.NodeProto, name: str, value: Any) -> None:
    new_attr = onnx.helper.make_attribute(str(name), value)
    for attr_index, attr in enumerate(node.attribute):
        if attr.name == name:
            node.attribute[attr_index].CopyFrom(new_attr)
            return
    node.attribute.append(new_attr)


def _onnx_replace_all_node_inputs(
    graph: onnx.GraphProto,
    *,
    old_name: str,
    new_name: str,
) -> None:
    if old_name == new_name:
        return
    for node in graph.node:
        for input_index, input_name in enumerate(node.input):
            if str(input_name) == str(old_name):
                node.input[input_index] = str(new_name)
    for output in graph.output:
        if str(output.name) == str(old_name):
            output.name = str(new_name)


def _onnx_remove_nodes_by_name(
    graph: onnx.GraphProto,
    node_names: Sequence[str],
) -> None:
    remove_name_set = {str(name) for name in list(node_names)}
    if not remove_name_set:
        return
    kept_nodes = [node for node in graph.node if str(node.name) not in remove_name_set]
    del graph.node[:]
    graph.node.extend(kept_nodes)


def _onnx_repair_inferred_shapes_in_place(model: onnx.ModelProto) -> None:
    original_internal_value_infos: Dict[str, onnx.ValueInfoProto] = {}
    for value_info in list(model.graph.value_info):
        value_info_name = str(getattr(value_info, "name", ""))
        if value_info_name == "":
            continue
        original_internal_value_infos[value_info_name] = onnx.ValueInfoProto()
        original_internal_value_infos[value_info_name].CopyFrom(value_info)
    del model.graph.value_info[:]
    produced_tensor_names = {
        str(output_name)
        for node in model.graph.node
        for output_name in list(node.output)
        if str(output_name) != ""
    }
    for output in model.graph.output:
        if str(output.name) not in produced_tensor_names:
            continue
        tensor_type = getattr(output.type, "tensor_type", None)
        if tensor_type is None:
            continue
        shape = getattr(tensor_type, "shape", None)
        if shape is None:
            continue
        del shape.dim[:]
    try:
        inferred_model = onnx.shape_inference.infer_shapes(
            model,
            check_type=False,
            strict_mode=False,
            data_prop=True,
        )
    except TypeError:
        try:
            inferred_model = onnx.shape_inference.infer_shapes(
                model,
                check_type=False,
                strict_mode=False,
            )
        except Exception:
            inferred_model = None
    except Exception:
        inferred_model = None
    if inferred_model is not None:
        model.CopyFrom(inferred_model)
    if original_internal_value_infos:
        existing_value_infos = {
            str(value_info.name): value_info
            for value_info in list(model.graph.value_info)
            if str(getattr(value_info, "name", "")) != ""
        }
        graph_input_names = {str(value_info.name) for value_info in list(model.graph.input)}
        graph_output_names = {str(value_info.name) for value_info in list(model.graph.output)}
        for value_info_name, original_value_info in original_internal_value_infos.items():
            if value_info_name in graph_input_names or value_info_name in graph_output_names:
                continue
            original_tensor_type = getattr(original_value_info.type, "tensor_type", None)
            if original_tensor_type is None:
                continue
            original_shape = getattr(original_tensor_type, "shape", None)
            original_shape_dims = list(original_shape.dim) if original_shape is not None else []
            original_elem_type = int(getattr(original_tensor_type, "elem_type", 0))
            existing_value_info = existing_value_infos.get(value_info_name)
            if existing_value_info is None:
                restored_value_info = model.graph.value_info.add()
                restored_value_info.CopyFrom(original_value_info)
                existing_value_infos[value_info_name] = restored_value_info
                continue
            existing_tensor_type = getattr(existing_value_info.type, "tensor_type", None)
            if existing_tensor_type is None:
                continue
            if int(getattr(existing_tensor_type, "elem_type", 0)) <= 0 and original_elem_type > 0:
                existing_tensor_type.elem_type = int(original_elem_type)
            existing_shape = getattr(existing_tensor_type, "shape", None)
            if existing_shape is None or len(list(existing_shape.dim)) != 0 or len(original_shape_dims) == 0:
                continue
            del existing_shape.dim[:]
            for original_dim in original_shape_dims:
                target_dim = existing_shape.dim.add()
                if original_dim.HasField("dim_value"):
                    target_dim.dim_value = int(original_dim.dim_value)
                elif original_dim.HasField("dim_param"):
                    target_dim.dim_param = str(original_dim.dim_param)
    producer_map, _ = _onnx_node_maps(model.graph)
    for output in model.graph.output:
        tensor_type = getattr(output.type, "tensor_type", None)
        if tensor_type is None:
            continue
        shape = getattr(tensor_type, "shape", None)
        if shape is None or len(list(shape.dim)) != 0:
            continue
        output_name = str(output.name)
        if output_name == "" or output_name not in producer_map:
            continue
        inferred_shape = _onnx_resolve_static_shape(
            model.graph,
            output_name,
            producer_map=producer_map,
        )
        if inferred_shape is None:
            continue
        for dim_value in inferred_shape:
            shape.dim.add().dim_value = int(dim_value)
    _onnx_restore_missing_internal_pad_value_info_shapes_in_place(model.graph)


def _onnx_get_initializer_index(graph: onnx.GraphProto, name: str) -> Optional[int]:
    for initializer_index, initializer in enumerate(graph.initializer):
        if str(initializer.name) == str(name):
            return int(initializer_index)
    return None


def _onnx_set_initializer_array(
    graph: onnx.GraphProto,
    *,
    name: str,
    array: np.ndarray,
) -> None:
    tensor = onnx.numpy_helper.from_array(np.asarray(array), name=str(name))
    initializer_index = _onnx_get_initializer_index(graph, str(name))
    if initializer_index is None:
        graph.initializer.append(tensor)
    else:
        graph.initializer[initializer_index].CopyFrom(tensor)


def _onnx_make_unique_initializer_name(graph: onnx.GraphProto, base_name: str) -> str:
    existing_names = {str(initializer.name) for initializer in graph.initializer}
    existing_names.update(str(node.output[0]) for node in graph.node if len(node.output) >= 1)
    candidate = str(base_name)
    suffix_index = 0
    while candidate in existing_names:
        suffix_index += 1
        candidate = f"{base_name}_{suffix_index}"
    return candidate


def _onnx_get_initializer_array(
    graph: onnx.GraphProto,
    name: str,
) -> Optional[np.ndarray]:
    initializer_index = _onnx_get_initializer_index(graph, str(name))
    if initializer_index is None:
        return None
    return onnx.numpy_helper.to_array(graph.initializer[initializer_index])


def _onnx_get_initializer_scalar(
    graph: onnx.GraphProto,
    name: str,
) -> Optional[float]:
    initializer_array = _onnx_get_initializer_array(graph, str(name))
    if initializer_array is None:
        return None
    initializer_values = np.asarray(initializer_array)
    if int(initializer_values.size) != 1:
        return None
    return float(initializer_values.reshape(()))


def _onnx_evaluate_constant_scatter_nd(
    *,
    data_array: Any,
    indices_array: Any,
    updates_array: Any,
    reduction: str = "none",
) -> Optional[np.ndarray]:
    try:
        data = np.asarray(data_array)
        indices = np.asarray(indices_array, dtype=np.int64)
        updates = np.asarray(updates_array)
    except Exception:
        return None
    if indices.ndim < 1:
        return None
    index_depth = int(indices.shape[-1])
    if index_depth < 0 or index_depth > int(data.ndim):
        return None
    prefix_shape = tuple(int(v) for v in list(indices.shape[:-1]))
    suffix_shape = tuple(int(v) for v in list(data.shape[index_depth:]))
    if tuple(int(v) for v in list(updates.shape)) != prefix_shape + suffix_shape:
        return None

    try:
        out = np.array(data, copy=True)
    except Exception:
        return None

    flat_indices = indices.reshape((-1, int(index_depth)))
    flat_updates = updates.reshape((int(flat_indices.shape[0]),) + suffix_shape)
    normalized_reduction = str(reduction).lower()
    if normalized_reduction == "":
        normalized_reduction = "none"
    if normalized_reduction not in {"none", "add"}:
        return None

    if int(index_depth) == 0:
        try:
            if normalized_reduction == "add":
                np.add(out, np.sum(flat_updates, axis=0), out=out)
            else:
                if int(flat_updates.shape[0]) != 1:
                    return None
                out[...] = flat_updates[0]
        except Exception:
            return None
        return out

    upper_bounds = np.asarray(data.shape[:index_depth], dtype=np.int64)
    if upper_bounds.size != int(index_depth):
        return None
    if np.any(flat_indices < 0) or np.any(flat_indices >= upper_bounds.reshape((1, -1))):
        return None
    if normalized_reduction == "none":
        seen_indices = {
            tuple(int(v) for v in list(index_row.tolist()))
            for index_row in flat_indices
        }
        if len(seen_indices) != int(flat_indices.shape[0]):
            return None

    try:
        for update_idx, target_index in enumerate(flat_indices):
            target_key = tuple(int(v) for v in list(target_index.tolist()))
            if normalized_reduction == "add":
                out[target_key] += flat_updates[int(update_idx)]
            else:
                out[target_key] = flat_updates[int(update_idx)]
    except Exception:
        return None
    return out


def _onnx_evaluate_constant_reshape(
    *,
    data_array: Any,
    shape_array: Any,
) -> Optional[np.ndarray]:
    try:
        data = np.asarray(data_array)
        shape_values = np.asarray(shape_array, dtype=np.int64).reshape(-1)
    except Exception:
        return None
    if shape_values.ndim != 1 or int(shape_values.size) == 0:
        return None
    if np.sum(shape_values == -1) > 1:
        return None
    if np.any(shape_values < -1):
        return None
    try:
        return np.reshape(data, tuple(int(v) for v in list(shape_values.tolist())))
    except Exception:
        return None


def _onnx_evaluate_constant_binary_elementwise(
    *,
    lhs_array: Any,
    rhs_array: Any,
    op_type: str,
) -> Optional[np.ndarray]:
    try:
        lhs = np.asarray(lhs_array)
        rhs = np.asarray(rhs_array)
    except Exception:
        return None
    try:
        if str(op_type) == "Add":
            out = np.add(lhs, rhs)
        elif str(op_type) == "Sub":
            out = np.subtract(lhs, rhs)
        elif str(op_type) == "Mul":
            out = np.multiply(lhs, rhs)
        elif str(op_type) == "Div":
            out = np.divide(lhs, rhs)
        else:
            return None
    except Exception:
        return None
    if np.issubdtype(out.dtype, np.floating) and not np.all(np.isfinite(out)):
        return None
    return np.asarray(out)


def _onnx_get_value_info_shape(
    graph: onnx.GraphProto,
    name: str,
) -> Optional[List[int]]:
    target_name = str(name)
    for value_info in list(graph.value_info) + list(graph.input) + list(graph.output):
        if str(value_info.name) != target_name:
            continue
        if len(value_info.type.tensor_type.shape.dim) == 0:
            return None
        shape: List[int] = []
        for dim in value_info.type.tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                shape.append(int(dim.dim_value))
            else:
                return None
        return shape
    return None


def _onnx_get_tensor_elem_type(
    graph: onnx.GraphProto,
    name: str,
) -> Optional[int]:
    target_name = str(name)
    for value_info in list(graph.value_info) + list(graph.input) + list(graph.output):
        if str(value_info.name) != target_name:
            continue
        tensor_type = getattr(value_info.type, "tensor_type", None)
        if tensor_type is None:
            return None
        elem_type = int(getattr(tensor_type, "elem_type", 0))
        return elem_type if elem_type > 0 else None
    initializer_index = _onnx_get_initializer_index(graph, target_name)
    if initializer_index is None:
        return None
    elem_type = int(getattr(graph.initializer[int(initializer_index)], "data_type", 0))
    return elem_type if elem_type > 0 else None


_ONNX_STATIC_SHAPE_PASSTHROUGH_OP_TYPES = {
    "Abs",
    "Acos",
    "Asin",
    "Atan",
    "Cast",
    "Ceil",
    "Clip",
    "Cos",
    "Cosh",
    "Elu",
    "Erf",
    "Exp",
    "Floor",
    "HardSigmoid",
    "Identity",
    "LeakyRelu",
    "Log",
    "Neg",
    "Reciprocal",
    "Relu",
    "Round",
    "Selu",
    "Sigmoid",
    "Sin",
    "Sinh",
    "Softplus",
    "Softsign",
    "Sqrt",
    "Tanh",
}


def _onnx_resolve_static_shape(
    graph: onnx.GraphProto,
    name: str,
    *,
    producer_map: Optional[Dict[str, onnx.NodeProto]] = None,
    depth: int = 0,
) -> Optional[List[int]]:
    if depth > 4:
        return None
    shape = _onnx_get_value_info_shape(graph, name)
    if shape is not None:
        return shape
    initializer_array = _onnx_get_initializer_array(graph, name)
    if initializer_array is not None:
        return [int(v) for v in initializer_array.shape]
    if producer_map is None:
        producer_map, _ = _onnx_node_maps(graph)
    producer_node = producer_map.get(str(name))
    if producer_node is None:
        return None
    op_type = str(producer_node.op_type)
    if op_type in _ONNX_STATIC_SHAPE_PASSTHROUGH_OP_TYPES and producer_node.input:
        return _onnx_resolve_static_shape(
            graph,
            str(producer_node.input[0]),
            producer_map=producer_map,
            depth=depth + 1,
        )
    if op_type in {"Add", "Div", "Max", "Min", "Mul", "Sub"} and len(producer_node.input) == 2:
        lhs_shape = _onnx_resolve_static_shape(
            graph,
            str(producer_node.input[0]),
            producer_map=producer_map,
            depth=depth + 1,
        )
        rhs_shape = _onnx_resolve_static_shape(
            graph,
            str(producer_node.input[1]),
            producer_map=producer_map,
            depth=depth + 1,
        )
        if lhs_shape is None:
            return rhs_shape
        if rhs_shape is None:
            return lhs_shape
        max_rank = max(len(lhs_shape), len(rhs_shape))
        lhs_shape = [1] * (max_rank - len(lhs_shape)) + list(lhs_shape)
        rhs_shape = [1] * (max_rank - len(rhs_shape)) + list(rhs_shape)
        resolved: List[int] = []
        for lhs_dim, rhs_dim in zip(lhs_shape, rhs_shape):
            if lhs_dim == rhs_dim:
                resolved.append(int(lhs_dim))
            elif lhs_dim == 1:
                resolved.append(int(rhs_dim))
            elif rhs_dim == 1:
                resolved.append(int(lhs_dim))
            else:
                return None
        return resolved
    if op_type == "Concat":
        axis = int(_onnx_node_attr(producer_node, "axis") or 0)
        input_shapes: List[List[int]] = []
        for input_name in list(producer_node.input):
            input_shape = _onnx_resolve_static_shape(
                graph,
                str(input_name),
                producer_map=producer_map,
                depth=depth + 1,
            )
            if input_shape is None:
                return None
            input_shapes.append(input_shape)
        if not input_shapes:
            return None
        rank = len(input_shapes[0])
        if any(len(input_shape) != rank for input_shape in input_shapes):
            return None
        if axis < 0:
            axis += rank
        if axis < 0 or axis >= rank:
            return None
        base_shape = list(input_shapes[0])
        concat_dim = 0
        for input_shape in input_shapes:
            for dim_index, dim_value in enumerate(input_shape):
                if dim_index == axis:
                    concat_dim += int(dim_value)
                elif int(base_shape[dim_index]) != int(dim_value):
                    return None
        base_shape[axis] = int(concat_dim)
        return base_shape
    if op_type == "Transpose" and producer_node.input:
        perm = [int(v) for v in list(_onnx_node_attr(producer_node, "perm") or [])]
        input_shape = _onnx_resolve_static_shape(
            graph,
            str(producer_node.input[0]),
            producer_map=producer_map,
            depth=depth + 1,
        )
        if input_shape is None or len(perm) != len(input_shape):
            return None
        return [int(input_shape[index]) for index in perm]
    if op_type == "Pad" and len(producer_node.input) >= 2:
        input_shape = _onnx_resolve_static_shape(
            graph,
            str(producer_node.input[0]),
            producer_map=producer_map,
            depth=depth + 1,
        )
        pads_array = _onnx_get_initializer_array(graph, str(producer_node.input[1]))
        if input_shape is None or pads_array is None:
            return None
        pads_values = np.asarray(pads_array, dtype=np.int64).reshape(-1)
        rank = int(len(input_shape))
        if int(pads_values.size) != int(rank) * 2:
            return None
        head_pads = pads_values[:rank]
        tail_pads = pads_values[rank:]
        try:
            return [
                int(int(dim) + int(head_pad) + int(tail_pad))
                for dim, head_pad, tail_pad in zip(input_shape, head_pads, tail_pads)
            ]
        except Exception:
            return None
    return None


def _onnx_restore_missing_internal_pad_value_info_shapes_in_place(
    graph: onnx.GraphProto,
) -> None:
    producer_map, _ = _onnx_node_maps(graph)
    graph_input_names = {str(value_info.name) for value_info in list(graph.input)}
    graph_output_names = {str(value_info.name) for value_info in list(graph.output)}

    def _find_value_info(name: str) -> Optional[onnx.ValueInfoProto]:
        for value_info in list(graph.value_info):
            if str(value_info.name) == str(name):
                return value_info
        return None

    for node in list(graph.node):
        if str(node.op_type) != "Pad" or len(list(node.output)) != 1 or len(list(node.input)) < 1:
            continue
        output_name = str(node.output[0])
        if output_name == "":
            continue
        if _onnx_get_value_info_shape(graph, output_name) is not None:
            continue
        resolved_shape = _onnx_resolve_static_shape(
            graph,
            output_name,
            producer_map=producer_map,
        )
        if resolved_shape is None:
            continue
        elem_type = _onnx_get_tensor_elem_type(graph, str(node.input[0]))
        if elem_type is None:
            continue
        target_value_info = _find_value_info(output_name)
        if target_value_info is None:
            if output_name in graph_input_names or output_name in graph_output_names:
                continue
            target_value_info = graph.value_info.add()
            target_value_info.name = output_name
        tensor_type = target_value_info.type.tensor_type
        tensor_type.elem_type = int(elem_type)
        del tensor_type.shape.dim[:]
        for dim_value in resolved_shape:
            tensor_type.shape.dim.add().dim_value = int(dim_value)


def _onnx_fold_constant_scatter_nd_in_place(graph: onnx.GraphProto) -> None:
    while True:
        changed = False
        for node_index, node in enumerate(list(graph.node)):
            if str(node.op_type) != "ScatterND" or len(list(node.input)) != 3 or len(list(node.output)) != 1:
                continue
            output_name = str(node.output[0])
            if output_name == "":
                continue
            data_array = _onnx_get_initializer_array(graph, str(node.input[0]))
            indices_array = _onnx_get_initializer_array(graph, str(node.input[1]))
            updates_array = _onnx_get_initializer_array(graph, str(node.input[2]))
            if data_array is None or indices_array is None or updates_array is None:
                continue
            folded = _onnx_evaluate_constant_scatter_nd(
                data_array=data_array,
                indices_array=indices_array,
                updates_array=updates_array,
                reduction=str(_onnx_node_attr(node, "reduction") or "none"),
            )
            if folded is None:
                continue
            _onnx_set_initializer_array(graph, name=output_name, array=np.asarray(folded))
            del graph.node[node_index]
            changed = True
            break
        if not changed:
            break


def _onnx_fold_constant_reshape_in_place(graph: onnx.GraphProto) -> None:
    while True:
        changed = False
        for node_index, node in enumerate(list(graph.node)):
            if str(node.op_type) != "Reshape" or len(list(node.input)) != 2 or len(list(node.output)) != 1:
                continue
            output_name = str(node.output[0])
            if output_name == "":
                continue
            data_array = _onnx_get_initializer_array(graph, str(node.input[0]))
            shape_array = _onnx_get_initializer_array(graph, str(node.input[1]))
            if data_array is None or shape_array is None:
                continue
            folded = _onnx_evaluate_constant_reshape(
                data_array=data_array,
                shape_array=shape_array,
            )
            if folded is None:
                continue
            _onnx_set_initializer_array(graph, name=output_name, array=np.asarray(folded))
            del graph.node[node_index]
            changed = True
            break
        if not changed:
            break


def _onnx_fold_constant_binary_elementwise_in_place(graph: onnx.GraphProto) -> None:
    supported_op_types = {"Add", "Sub", "Mul", "Div"}
    while True:
        changed = False
        for node_index, node in enumerate(list(graph.node)):
            if str(node.op_type) not in supported_op_types or len(list(node.input)) != 2 or len(list(node.output)) != 1:
                continue
            output_name = str(node.output[0])
            if output_name == "":
                continue
            lhs_array = _onnx_get_initializer_array(graph, str(node.input[0]))
            rhs_array = _onnx_get_initializer_array(graph, str(node.input[1]))
            if lhs_array is None or rhs_array is None:
                continue
            folded = _onnx_evaluate_constant_binary_elementwise(
                lhs_array=lhs_array,
                rhs_array=rhs_array,
                op_type=str(node.op_type),
            )
            if folded is None:
                continue
            _onnx_set_initializer_array(graph, name=output_name, array=np.asarray(folded))
            del graph.node[node_index]
            changed = True
            break
        if not changed:
            break
