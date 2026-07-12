from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import onnx

from onnx2tf.tflite_builder.pytorch_onnx_utils import (
    _onnx_get_initializer_array,
    _onnx_make_unique_initializer_name,
    _onnx_node_attr,
    _onnx_node_maps,
    _onnx_remove_nodes_by_name,
    _onnx_replace_all_node_inputs,
    _onnx_resolve_static_shape,
    _onnx_set_initializer_array,
)


def _onnx_resolve_rank4_shape(
    graph: onnx.GraphProto,
    name: str,
    *,
    producer_map: Optional[Dict[str, onnx.NodeProto]] = None,
    depth: int = 0,
) -> Optional[List[int]]:
    resolved_shape = _onnx_resolve_static_shape(
        graph,
        name,
        producer_map=producer_map,
        depth=depth,
    )
    if resolved_shape is None or len(resolved_shape) != 4:
        return None
    return resolved_shape


def _onnx_convert_pads_nhwc_to_nchw(pads: Sequence[int] | np.ndarray) -> Optional[np.ndarray]:
    pad_values = np.asarray(list(pads), dtype=np.int64).reshape(-1)
    if pad_values.size != 8:
        return None
    begin = pad_values[:4]
    end = pad_values[4:]
    reorder = [0, 3, 1, 2]
    return np.concatenate([begin[reorder], end[reorder]], axis=0).astype(np.int64)


def _onnx_rewrite_slice_axis_to_nchw_in_place(
    graph: onnx.GraphProto,
    *,
    slice_node: onnx.NodeProto,
) -> bool:
    axes_name = str(slice_node.input[3]) if len(slice_node.input) >= 4 else ""
    axes_array = _onnx_get_initializer_array(graph, axes_name)
    if axes_array is None:
        return False
    axes_values = [int(v) for v in axes_array.reshape(-1)]
    if axes_values != [3]:
        return False
    new_axes_name = _onnx_make_unique_initializer_name(graph, f"{axes_name}_nchw")
    _onnx_set_initializer_array(graph, name=new_axes_name, array=np.asarray([1], dtype=np.int64))
    slice_node.input[3] = str(new_axes_name)
    return True


def _onnx_fold_relu_layout_bridges_in_place(graph: onnx.GraphProto) -> None:
    layout_preserving_unary_op_types = {
        "Clip",
        "Elu",
        "Exp",
        "Identity",
        "LeakyRelu",
        "Relu",
        "Sigmoid",
        "Tanh",
    }
    _, consumer_map = _onnx_node_maps(graph)
    remove_node_names: List[str] = []
    for transpose_node in list(graph.node):
        if str(transpose_node.op_type) != "Transpose":
            continue
        transpose_perm = [int(v) for v in list(_onnx_node_attr(transpose_node, "perm") or [])]
        if not transpose_perm:
            continue
        transpose_consumers = consumer_map.get(str(transpose_node.output[0]), [])
        if len(transpose_consumers) != 1:
            continue
        unary_node = transpose_consumers[0]
        if str(unary_node.op_type) not in layout_preserving_unary_op_types:
            continue
        unary_consumers = consumer_map.get(str(unary_node.output[0]), [])
        if len(unary_consumers) != 1:
            continue
        inverse_node = unary_consumers[0]
        if str(inverse_node.op_type) != "Transpose":
            continue
        inverse_perm = [0] * len(transpose_perm)
        for perm_index, perm_value in enumerate(transpose_perm):
            perm_value_int = int(perm_value)
            if perm_value_int < 0 or perm_value_int >= len(transpose_perm):
                inverse_perm = []
                break
            inverse_perm[perm_value_int] = int(perm_index)
        if not inverse_perm:
            continue
        if [int(v) for v in list(_onnx_node_attr(inverse_node, "perm") or [])] != inverse_perm:
            continue
        unary_node.input[0] = str(transpose_node.input[0])
        unary_node.output[0] = str(inverse_node.output[0])
        remove_node_names.extend([str(transpose_node.name), str(inverse_node.name)])
    _onnx_remove_nodes_by_name(graph, remove_node_names)


def _onnx_fold_reducesum_sigmoid_layout_bridges_in_place(graph: onnx.GraphProto) -> None:
    _, consumer_map = _onnx_node_maps(graph)
    remove_node_names: List[str] = []
    for transpose_node in list(graph.node):
        if str(transpose_node.op_type) != "Transpose":
            continue
        if list(_onnx_node_attr(transpose_node, "perm") or []) != [0, 2, 3, 1]:
            continue
        transpose_consumers = consumer_map.get(str(transpose_node.output[0]), [])
        if len(transpose_consumers) != 1:
            continue
        reduce_node = transpose_consumers[0]
        if str(reduce_node.op_type) != "ReduceSum":
            continue
        axes_name = str(reduce_node.input[1]) if len(reduce_node.input) >= 2 else ""
        axes_array = _onnx_get_initializer_array(graph, axes_name)
        if axes_array is None or [int(v) for v in axes_array.reshape(-1)] != [3]:
            continue
        if int(_onnx_node_attr(reduce_node, "keepdims") or 0) != 1:
            continue
        reduce_consumers = consumer_map.get(str(reduce_node.output[0]), [])
        if len(reduce_consumers) != 1:
            continue
        sigmoid_node = reduce_consumers[0]
        if str(sigmoid_node.op_type) != "Sigmoid":
            continue
        sigmoid_consumers = consumer_map.get(str(sigmoid_node.output[0]), [])
        if len(sigmoid_consumers) != 1:
            continue
        inverse_node = sigmoid_consumers[0]
        if str(inverse_node.op_type) != "Transpose":
            continue
        if list(_onnx_node_attr(inverse_node, "perm") or []) != [0, 3, 1, 2]:
            continue
        new_axes_name = _onnx_make_unique_initializer_name(graph, f"{axes_name}_nchw")
        _onnx_set_initializer_array(graph, name=new_axes_name, array=np.asarray([1], dtype=np.int64))
        reduce_node.input[0] = str(transpose_node.input[0])
        reduce_node.input[1] = str(new_axes_name)
        sigmoid_node.output[0] = str(inverse_node.output[0])
        remove_node_names.extend([str(transpose_node.name), str(inverse_node.name)])
    _onnx_remove_nodes_by_name(graph, remove_node_names)


def _onnx_fold_mul_reducesum_sigmoid_layout_bridges_in_place(graph: onnx.GraphProto) -> None:
    producer_map, consumer_map = _onnx_node_maps(graph)
    remove_node_names: List[str] = []
    for inverse_node in list(graph.node):
        if str(inverse_node.op_type) != "Transpose":
            continue
        if list(_onnx_node_attr(inverse_node, "perm") or []) != [0, 3, 1, 2]:
            continue
        if not inverse_node.input:
            continue
        sigmoid_node = producer_map.get(str(inverse_node.input[0]))
        if sigmoid_node is None or str(sigmoid_node.op_type) != "Sigmoid":
            continue
        if len(consumer_map.get(str(sigmoid_node.output[0]), [])) != 1:
            continue
        if not sigmoid_node.input:
            continue
        reduce_node = producer_map.get(str(sigmoid_node.input[0]))
        if reduce_node is None or str(reduce_node.op_type) != "ReduceSum":
            continue
        if len(consumer_map.get(str(reduce_node.output[0]), [])) != 1:
            continue
        axes_name = str(reduce_node.input[1]) if len(reduce_node.input) >= 2 else ""
        axes_array = _onnx_get_initializer_array(graph, axes_name)
        if axes_array is None or [int(v) for v in axes_array.reshape(-1)] != [3]:
            continue
        if int(_onnx_node_attr(reduce_node, "keepdims") or 0) != 1:
            continue
        if not reduce_node.input:
            continue
        mul_node = producer_map.get(str(reduce_node.input[0]))
        if mul_node is None or str(mul_node.op_type) != "Mul" or len(mul_node.input) != 2:
            continue
        if len(consumer_map.get(str(mul_node.output[0]), [])) != 1:
            continue
        input_transpose_nodes: List[onnx.NodeProto] = []
        for input_name in mul_node.input:
            transpose_node = producer_map.get(str(input_name))
            if transpose_node is None or str(transpose_node.op_type) != "Transpose":
                input_transpose_nodes = []
                break
            input_transpose_nodes.append(transpose_node)
        if len(input_transpose_nodes) != 2:
            continue
        if any(list(_onnx_node_attr(node, "perm") or []) != [0, 2, 3, 1] for node in input_transpose_nodes):
            continue
        if any(len(consumer_map.get(str(node.output[0]), [])) != 1 for node in input_transpose_nodes):
            continue

        for input_index, transpose_node in enumerate(input_transpose_nodes):
            mul_node.input[input_index] = str(transpose_node.input[0])
        new_axes_name = _onnx_make_unique_initializer_name(graph, f"{axes_name}_nchw")
        _onnx_set_initializer_array(graph, name=new_axes_name, array=np.asarray([1], dtype=np.int64))
        reduce_node.input[1] = str(new_axes_name)
        _onnx_replace_all_node_inputs(
            graph,
            old_name=str(inverse_node.output[0]),
            new_name=str(sigmoid_node.output[0]),
        )
        remove_node_names.extend([str(node.name) for node in input_transpose_nodes] + [str(inverse_node.name)])
    _onnx_remove_nodes_by_name(graph, remove_node_names)


def _onnx_fold_inverse_transpose_pairs_in_place(graph: onnx.GraphProto) -> None:
    _, consumer_map = _onnx_node_maps(graph)
    remove_node_names: List[str] = []
    for first_node in list(graph.node):
        if str(first_node.op_type) != "Transpose":
            continue
        first_perm = [int(v) for v in list(_onnx_node_attr(first_node, "perm") or [])]
        if not first_perm:
            continue
        first_consumers = consumer_map.get(str(first_node.output[0]), [])
        if len(first_consumers) != 1:
            continue
        second_node = first_consumers[0]
        if str(second_node.op_type) != "Transpose":
            continue
        second_perm = [int(v) for v in list(_onnx_node_attr(second_node, "perm") or [])]
        inverse_perm = [0] * len(first_perm)
        for perm_index, perm_value in enumerate(first_perm):
            inverse_perm[int(perm_value)] = int(perm_index)
        if second_perm != inverse_perm:
            continue
        _onnx_replace_all_node_inputs(
            graph,
            old_name=str(second_node.output[0]),
            new_name=str(first_node.input[0]),
        )
        remove_node_names.extend([str(first_node.name), str(second_node.name)])
    _onnx_remove_nodes_by_name(graph, remove_node_names)


def _onnx_fold_pad_layout_bridges_in_place(graph: onnx.GraphProto) -> None:
    producer_map, consumer_map = _onnx_node_maps(graph)
    remove_node_names: List[str] = []
    for inverse_transpose_node in list(graph.node):
        if str(inverse_transpose_node.op_type) != "Transpose":
            continue
        if list(_onnx_node_attr(inverse_transpose_node, "perm") or []) != [0, 3, 1, 2]:
            continue
        if not inverse_transpose_node.input or not inverse_transpose_node.output:
            continue
        pad_node = producer_map.get(str(inverse_transpose_node.input[0]))
        if pad_node is None or str(pad_node.op_type) != "Pad" or len(pad_node.input) < 2:
            continue
        if len(consumer_map.get(str(pad_node.output[0]), [])) != 1:
            continue
        input_transpose_node = producer_map.get(str(pad_node.input[0]))
        if input_transpose_node is None or str(input_transpose_node.op_type) != "Transpose":
            continue
        if list(_onnx_node_attr(input_transpose_node, "perm") or []) != [0, 2, 3, 1]:
            continue
        if len(consumer_map.get(str(input_transpose_node.output[0]), [])) != 1:
            continue
        pad_name = str(pad_node.input[1])
        pad_values = _onnx_get_initializer_array(graph, pad_name)
        nchw_pad_values = (
            _onnx_convert_pads_nhwc_to_nchw(pad_values)
            if pad_values is not None
            else None
        )
        if nchw_pad_values is None:
            continue
        new_pad_name = _onnx_make_unique_initializer_name(graph, f"{pad_name}_nchw")
        _onnx_set_initializer_array(graph, name=new_pad_name, array=nchw_pad_values)
        pad_node.input[0] = str(input_transpose_node.input[0])
        pad_node.input[1] = str(new_pad_name)
        _onnx_replace_all_node_inputs(
            graph,
            old_name=str(inverse_transpose_node.output[0]),
            new_name=str(pad_node.output[0]),
        )
        remove_node_names.extend([str(input_transpose_node.name), str(inverse_transpose_node.name)])
    _onnx_remove_nodes_by_name(graph, remove_node_names)


def _onnx_fold_residual_add_layout_bridges_in_place(graph: onnx.GraphProto) -> None:
    layout_preserving_unary_op_types = {
        "Clip",
        "Identity",
        "LeakyRelu",
        "Relu",
        "Sigmoid",
    }
    while True:
        producer_map, consumer_map = _onnx_node_maps(graph)
        optimized = False
        for add_node in list(graph.node):
            if str(add_node.op_type) != "Add" or len(add_node.input) != 2:
                continue
            add_output_name = str(add_node.output[0]) if add_node.output else ""
            if add_output_name == "":
                continue

            rewritten_inputs: List[str] = []
            remove_node_names: List[str] = []
            expected_shape: Optional[List[int]] = None
            valid_inputs = True
            for input_name in list(add_node.input):
                input_str = str(input_name)
                input_producer = producer_map.get(input_str)
                if (
                    input_producer is not None
                    and str(input_producer.op_type) == "Transpose"
                    and list(_onnx_node_attr(input_producer, "perm") or []) == [0, 2, 3, 1]
                    and len(consumer_map.get(str(input_producer.output[0]), [])) == 1
                    and input_producer.input
                ):
                    source_name = str(input_producer.input[0])
                    source_shape = _onnx_resolve_rank4_shape(
                        graph,
                        source_name,
                        producer_map=producer_map,
                    )
                    if source_shape is None or len(source_shape) != 4:
                        valid_inputs = False
                        break
                    if expected_shape is None:
                        expected_shape = [int(v) for v in list(source_shape)]
                    elif [int(v) for v in list(source_shape)] != expected_shape:
                        valid_inputs = False
                        break
                    rewritten_inputs.append(source_name)
                    remove_node_names.append(str(input_producer.name))
                    continue
                input_shape = _onnx_resolve_rank4_shape(
                    graph,
                    input_str,
                    producer_map=producer_map,
                )
                if input_shape is None or len(input_shape) != 4:
                    valid_inputs = False
                    break
                if expected_shape is None:
                    expected_shape = [int(v) for v in list(input_shape)]
                elif [int(v) for v in list(input_shape)] != expected_shape:
                    valid_inputs = False
                    break
                rewritten_inputs.append(input_str)
            if not valid_inputs:
                continue
            if rewritten_inputs == [str(v) for v in list(add_node.input)]:
                continue

            add_consumers = consumer_map.get(add_output_name, [])
            if len(add_consumers) != 1:
                continue
            first_consumer = add_consumers[0]

            trailing_relu_node: Optional[onnx.NodeProto] = None
            trailing_transpose_node: Optional[onnx.NodeProto] = None
            conv_anchor_name = ""

            if str(first_consumer.op_type) in layout_preserving_unary_op_types:
                relu_consumers = consumer_map.get(str(first_consumer.output[0]), [])
                if not relu_consumers:
                    continue
                transpose_consumers = [
                    node
                    for node in relu_consumers
                    if str(node.op_type) == "Transpose"
                    and list(_onnx_node_attr(node, "perm") or []) == [0, 3, 1, 2]
                ]
                if len(transpose_consumers) != 1:
                    continue
                candidate_transpose = transpose_consumers[0]
                transpose_output_consumers = consumer_map.get(str(candidate_transpose.output[0]), [])
                if (
                    not transpose_output_consumers
                    or not all(str(node.op_type) == "Conv" for node in transpose_output_consumers)
                ):
                    continue
                trailing_relu_node = first_consumer
                trailing_transpose_node = candidate_transpose
                conv_anchor_name = str(candidate_transpose.output[0])
            elif (
                str(first_consumer.op_type) == "Transpose"
                and list(_onnx_node_attr(first_consumer, "perm") or []) == [0, 3, 1, 2]
            ):
                transpose_consumers = consumer_map.get(str(first_consumer.output[0]), [])
                if len(transpose_consumers) != 1:
                    continue
                candidate_relu = transpose_consumers[0]
                if str(candidate_relu.op_type) not in layout_preserving_unary_op_types:
                    continue
                relu_consumers = consumer_map.get(str(candidate_relu.output[0]), [])
                if not relu_consumers or not all(str(node.op_type) == "Conv" for node in relu_consumers):
                    continue
                trailing_transpose_node = first_consumer
                trailing_relu_node = candidate_relu
                conv_anchor_name = str(first_consumer.output[0])
            else:
                continue

            for input_index, rewritten_input_name in enumerate(rewritten_inputs):
                add_node.input[input_index] = str(rewritten_input_name)
            if str(trailing_relu_node.input[0]) != add_output_name:
                trailing_relu_node.input[0] = add_output_name
            if str(first_consumer.op_type) in layout_preserving_unary_op_types:
                _onnx_replace_all_node_inputs(
                    graph,
                    old_name=conv_anchor_name,
                    new_name=str(trailing_relu_node.output[0]),
                )
            remove_node_names.append(str(trailing_transpose_node.name))
            _onnx_remove_nodes_by_name(graph, remove_node_names)
            optimized = True
            break
        if not optimized:
            break


def _onnx_remove_passthrough_identity_nodes_in_place(graph: onnx.GraphProto) -> None:
    remove_node_names: List[str] = []
    for node in list(graph.node):
        if str(node.op_type) != "Identity":
            continue
        if len(node.input) != 1 or len(node.output) != 1:
            continue
        input_name = str(node.input[0])
        output_name = str(node.output[0])
        if not input_name or not output_name or input_name == output_name:
            continue
        _onnx_replace_all_node_inputs(
            graph,
            old_name=output_name,
            new_name=input_name,
        )
        remove_node_names.append(str(node.name))
    _onnx_remove_nodes_by_name(graph, remove_node_names)
