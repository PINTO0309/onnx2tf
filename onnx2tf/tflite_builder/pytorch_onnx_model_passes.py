from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import onnx

from onnx2tf.tflite_builder.pytorch_onnx_layout_passes import (
    _onnx_convert_pads_nhwc_to_nchw,
)
from onnx2tf.tflite_builder.pytorch_onnx_utils import (
    _onnx_get_initializer_array,
    _onnx_make_unique_initializer_name,
    _onnx_node_attr,
    _onnx_node_maps,
    _onnx_remove_nodes_by_name,
    _onnx_replace_all_node_inputs,
    _onnx_set_initializer_array,
)


def _onnx_optimize_pidnet_spp_transpose_bridges_in_place(graph: onnx.GraphProto) -> None:
    while True:
        producer_map, consumer_map = _onnx_node_maps(graph)
        optimized = False
        for layout_bridge_node in list(graph.node):
            if str(layout_bridge_node.op_type) != "Transpose":
                continue
            if list(_onnx_node_attr(layout_bridge_node, "perm") or []) != [0, 2, 3, 1]:
                continue
            if not layout_bridge_node.input or not layout_bridge_node.output:
                continue
            base_input_name = str(layout_bridge_node.input[0])
            layout_bridge_output_name = str(layout_bridge_node.output[0])
            layout_bridge_consumers = consumer_map.get(layout_bridge_output_name, [])
            if len(layout_bridge_consumers) < 4:
                continue

            inverse_bridge_node: Optional[onnx.NodeProto] = None
            reduce_mean_node: Optional[onnx.NodeProto] = None
            pad_bridge_nodes: List[onnx.NodeProto] = []
            unsupported_consumer = False
            for consumer_node in layout_bridge_consumers:
                if (
                    str(consumer_node.op_type) == "Transpose"
                    and list(_onnx_node_attr(consumer_node, "perm") or []) == [0, 3, 1, 2]
                ):
                    if inverse_bridge_node is not None:
                        unsupported_consumer = True
                        break
                    inverse_bridge_node = consumer_node
                    continue
                if str(consumer_node.op_type) == "ReduceMean":
                    if reduce_mean_node is not None:
                        unsupported_consumer = True
                        break
                    reduce_mean_node = consumer_node
                    continue
                if str(consumer_node.op_type) == "Pad":
                    pad_bridge_nodes.append(consumer_node)
                    continue
                unsupported_consumer = True
                break
            if unsupported_consumer:
                continue
            if inverse_bridge_node is None or reduce_mean_node is None or len(pad_bridge_nodes) < 2:
                continue

            inverse_bridge_consumers = consumer_map.get(str(inverse_bridge_node.output[0]), [])
            if not inverse_bridge_consumers:
                continue

            mean_axes_name = str(reduce_mean_node.input[1]) if len(reduce_mean_node.input) >= 2 else ""
            mean_axes_array = _onnx_get_initializer_array(graph, mean_axes_name)
            if mean_axes_array is None or [int(v) for v in mean_axes_array.reshape(-1)] != [1, 2]:
                continue
            if int(_onnx_node_attr(reduce_mean_node, "keepdims") or 0) != 1:
                continue
            reduce_mean_consumers = consumer_map.get(str(reduce_mean_node.output[0]), [])
            if len(reduce_mean_consumers) != 1:
                continue
            reduce_mean_mul_node = reduce_mean_consumers[0]
            if str(reduce_mean_mul_node.op_type) != "Mul" or len(reduce_mean_mul_node.input) != 2:
                continue
            reduce_mean_mul_consumers = consumer_map.get(str(reduce_mean_mul_node.output[0]), [])
            if len(reduce_mean_mul_consumers) != 1:
                continue
            reduce_mean_inverse_node = reduce_mean_mul_consumers[0]
            if str(reduce_mean_inverse_node.op_type) != "Transpose":
                continue
            if list(_onnx_node_attr(reduce_mean_inverse_node, "perm") or []) != [0, 3, 1, 2]:
                continue

            pad_rewrites: List[Tuple[onnx.NodeProto, str, onnx.NodeProto]] = []
            pad_pattern_valid = True
            for pad_node in pad_bridge_nodes:
                pad_name = str(pad_node.input[1]) if len(pad_node.input) >= 2 else ""
                pad_values = _onnx_get_initializer_array(graph, pad_name)
                nchw_pad_values = (
                    _onnx_convert_pads_nhwc_to_nchw(pad_values)
                    if pad_values is not None
                    else None
                )
                if not pad_name or nchw_pad_values is None:
                    pad_pattern_valid = False
                    break
                pad_consumers = consumer_map.get(str(pad_node.output[0]), [])
                if len(pad_consumers) != 1:
                    pad_pattern_valid = False
                    break
                pad_inverse_node = pad_consumers[0]
                if str(pad_inverse_node.op_type) != "Transpose":
                    pad_pattern_valid = False
                    break
                if list(_onnx_node_attr(pad_inverse_node, "perm") or []) != [0, 3, 1, 2]:
                    pad_pattern_valid = False
                    break
                pad_inverse_consumers = consumer_map.get(str(pad_inverse_node.output[0]), [])
                if len(pad_inverse_consumers) != 1:
                    pad_pattern_valid = False
                    break
                if str(pad_inverse_consumers[0].op_type) != "AveragePool":
                    pad_pattern_valid = False
                    break
                pad_rewrites.append((pad_node, pad_name, pad_inverse_node))
            if not pad_pattern_valid:
                continue

            for consumer_node in inverse_bridge_consumers:
                for input_index, input_name in enumerate(consumer_node.input):
                    if str(input_name) == str(inverse_bridge_node.output[0]):
                        consumer_node.input[input_index] = base_input_name

            new_mean_axes_name = _onnx_make_unique_initializer_name(graph, f"{mean_axes_name}_nchw")
            _onnx_set_initializer_array(
                graph,
                name=new_mean_axes_name,
                array=np.asarray([2, 3], dtype=np.int64),
            )
            reduce_mean_node.input[0] = base_input_name
            reduce_mean_node.input[1] = str(new_mean_axes_name)

            mul_const_name = str(reduce_mean_mul_node.input[1]) if len(reduce_mean_mul_node.input) >= 2 else ""
            mul_const_array = _onnx_get_initializer_array(graph, mul_const_name)
            if mul_const_name and mul_const_array is not None and len(mul_const_array.shape) == 4:
                _onnx_set_initializer_array(
                    graph,
                    name=mul_const_name,
                    array=np.transpose(mul_const_array, (0, 3, 1, 2)),
                )

            _onnx_replace_all_node_inputs(
                graph,
                old_name=str(reduce_mean_inverse_node.output[0]),
                new_name=str(reduce_mean_mul_node.output[0]),
            )

            for pad_node, pad_name, pad_inverse_node in pad_rewrites:
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
                pad_node.input[0] = base_input_name
                pad_node.input[1] = str(new_pad_name)
                pad_inverse_consumers = consumer_map.get(str(pad_inverse_node.output[0]), [])
                if len(pad_inverse_consumers) == 1:
                    pad_inverse_consumers[0].input[0] = str(pad_node.output[0])

            _onnx_remove_nodes_by_name(
                graph,
                [
                    str(layout_bridge_node.name),
                    str(inverse_bridge_node.name),
                    str(reduce_mean_inverse_node.name),
                    *[str(pad_inverse_node.name) for _, _, pad_inverse_node in pad_rewrites],
                ],
            )
            optimized = True
            break
        if not optimized:
            break


def _onnx_optimize_pphumanseg_add_resize_layout_bridges_in_place(graph: onnx.GraphProto) -> None:
    producer_map, consumer_map = _onnx_node_maps(graph)
    remove_node_names: List[str] = []
    for inverse_transpose_node in list(graph.node):
        if str(inverse_transpose_node.op_type) != "Transpose":
            continue
        if list(_onnx_node_attr(inverse_transpose_node, "perm") or []) != [0, 3, 1, 2]:
            continue
        relu_node_name = str(inverse_transpose_node.input[0]) if inverse_transpose_node.input else ""
        relu_node = producer_map.get(relu_node_name)
        if relu_node is None or str(relu_node.op_type) != "Relu":
            continue
        if len(consumer_map.get(str(relu_node.output[0]), [])) != 1:
            continue
        add_node_name = str(relu_node.input[0]) if relu_node.input else ""
        add_node = producer_map.get(add_node_name)
        if add_node is None or str(add_node.op_type) != "Add" or len(add_node.input) != 2:
            continue
        if len(consumer_map.get(str(add_node.output[0]), [])) != 1:
            continue
        add_input_producers: List[onnx.NodeProto] = []
        for add_input_name in add_node.input:
            producer = producer_map.get(str(add_input_name))
            if producer is None:
                add_input_producers = []
                break
            add_input_producers.append(producer)
        if len(add_input_producers) != 2:
            continue
        if any(str(node.op_type) != "Transpose" for node in add_input_producers):
            continue
        if any(list(_onnx_node_attr(node, "perm") or []) != [0, 2, 3, 1] for node in add_input_producers):
            continue
        if any(len(consumer_map.get(str(node.output[0]), [])) != 1 for node in add_input_producers):
            continue
        resize_nodes = consumer_map.get(str(inverse_transpose_node.output[0]), [])
        if len(resize_nodes) != 1:
            continue
        resize_node = resize_nodes[0]
        if str(resize_node.op_type) != "Resize" or not resize_node.input:
            continue
        if str(resize_node.input[0]) != str(inverse_transpose_node.output[0]):
            continue
        if len(consumer_map.get(str(resize_node.output[0]), [])) != 1:
            continue
        trailing_nodes = consumer_map.get(str(resize_node.output[0]), [])
        if len(trailing_nodes) != 1:
            continue
        trailing_transpose_node = trailing_nodes[0]
        if str(trailing_transpose_node.op_type) != "Transpose":
            continue
        if list(_onnx_node_attr(trailing_transpose_node, "perm") or []) != [0, 2, 3, 1]:
            continue

        add_node.input[0] = str(add_input_producers[0].input[0])
        add_node.input[1] = str(add_input_producers[1].input[0])
        resize_node.input[0] = str(relu_node.output[0])
        remove_node_names.extend(
            [
                str(add_input_producers[0].name),
                str(add_input_producers[1].name),
                str(inverse_transpose_node.name),
            ]
        )
    _onnx_remove_nodes_by_name(graph, remove_node_names)
